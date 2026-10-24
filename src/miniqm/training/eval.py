"""Validation loss evaluation, test-set benchmarking, and SMILES conformer inference."""

import dataclasses
import functools
import json
import logging
import pathlib
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
from rich import progress

from miniqm import jaxtypes as jt
from miniqm import viz
from miniqm.data import batch as batch_lib
from miniqm.data import data_source
from miniqm.diffusion import geometry
from miniqm.diffusion import losses as losses_lib
from miniqm.diffusion import model as model_lib
from miniqm.diffusion import noise as noise_lib
from miniqm.diffusion import sampling as sampling_lib
from miniqm.training import checkpoint

_LOGGER: Final = logging.getLogger(__name__)
# Standard QM9 conformer generation RMSD coverage thresholds (in Angstroms).
_STRICT_RMSD_THRESHOLD_A: Final = 0.5
_LOOSE_RMSD_THRESHOLD_A: Final = 1.0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationMetrics:
    """Teacher-forced single-step validation loss breakdown across an evaluation split."""

    total_loss: float
    aligned_mse_loss: float
    distance_loss: float
    rmsd: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SamplingMetrics:
    """Full reverse-diffusion conformer generation RMSD statistics on a validation batch."""

    sampled_rmsd_mean: float
    sampled_rmsd_median: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSamplingMetrics:
    """Full reverse-diffusion conformer generation RMSD and threshold coverage across a split."""

    sampled_rmsd_mean: float
    sampled_rmsd_median: float
    fraction_below_0_5a: float
    fraction_below_1_0a: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EvalConfig:
    """Configuration for evaluating a trained checkpoint on a QM9 Parquet split."""

    run_dir: pathlib.Path = pathlib.Path("runs/qm9_default")
    test_path: pathlib.Path = pathlib.Path("data/test.parquet")
    checkpoint_name: str = "best.msgpack"
    use_ema: bool = True
    batch_size: int = 64
    max_test_molecules: int | None = None
    num_sampling_steps: int | None = None
    output_path: pathlib.Path | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"Expected positive batch_size, got {self.batch_size}.")
        if self.num_sampling_steps is not None and self.num_sampling_steps <= 0:
            raise ValueError(
                f"Expected positive num_sampling_steps, got {self.num_sampling_steps}."
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TestEvaluationReport:
    """Combined teacher-forced loss and reverse-diffusion sampling report on a test split."""

    step: int
    num_molecules: int
    loss_metrics: ValidationMetrics
    sampling_metrics: DatasetSamplingMetrics


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SampleSmilesConfig:
    """Configuration for generating 3D conformers from a sequence of SMILES strings."""

    run_dir: pathlib.Path = pathlib.Path("runs/qm9_default")
    smiles: tuple[str, ...] = ("CCO", "c1ccccc1")
    smiles_file: pathlib.Path | None = None
    checkpoint_name: str = "best.msgpack"
    use_ema: bool = True
    num_sampling_steps: int | None = None
    output_xyz: pathlib.Path = pathlib.Path("sampled_conformers.xyz")
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_sampling_steps is not None and self.num_sampling_steps <= 0:
            raise ValueError(
                f"Expected positive num_sampling_steps, got {self.num_sampling_steps}."
            )


@jt.typed
def compute_batch_loss(
    params: flax_typing.VariableDict,
    noisy_sample: noise_lib.NoisySample,
    batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> losses_lib.LossOutput:
    """Runs the denoiser on a corrupted sample and evaluates the diffusion loss."""
    denoiser = model_lib.Denoiser(config=model_config)
    predicted_positions = cast(
        "jt.Float[jt.Array, 'batch num_atoms 3']",
        denoiser.apply(
            params,
            noisy_sample.noisy_positions,
            noisy_sample.sigma,
            batch=batch,
        ),
    )
    return losses_lib.compute_diffusion_loss(
        predicted_positions=predicted_positions,
        target_positions=noisy_sample.target_positions,
        sigma=noisy_sample.sigma,
        atom_mask=batch.atom_mask,
        config=loss_config,
    )


@jt.typed
def eval_step(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> losses_lib.LossOutput:
    """Evaluates the diffusion loss on a batch without updating parameters.

    Args:
        params: Model parameters (typically EMA weights).
        rng: PRNG key for random 3D rotation augmentation and noise sampling.
        batch: Padded JAX batch with ground-truth 3D coordinates and molecular graph features.
        model_config: Denoiser architecture and noise schedule configuration.
        loss_config: Diffusion loss weights and distance cutoff configuration.

    Returns:
        Diffusion loss breakdown on the randomly rotated and corrupted batch.
    """
    noisy_sample = noise_lib.make_noisy_sample(
        rng,
        batch.positions,
        batch.atom_mask,
        config=model_config.noise,
    )
    return compute_batch_loss(
        params,
        noisy_sample,
        batch,
        model_config=model_config,
        loss_config=loss_config,
    )


@functools.partial(jax.jit, static_argnames=("model_config", "loss_config"))
@jt.typed
def _jit_eval_step(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> losses_lib.LossOutput:
    """Evaluates the diffusion loss on a batch under a cached JIT compilation."""
    return eval_step(
        params,
        rng,
        batch,
        model_config=model_config,
        loss_config=loss_config,
    )


@jt.typed
def evaluate_loss(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    val_source: data_source.DataSource,
    *,
    batch_size: int,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> ValidationMetrics:
    """Computes mean teacher-forced diffusion loss and RMSD across validation batches.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for validation rotation augmentation and noise sampling.
        val_source: Validation data source.
        batch_size: Validation batch size.
        model_config: Denoiser architecture and noise schedule configuration.
        loss_config: Diffusion loss weights and distance cutoff configuration.

    Returns:
        Mean validation loss components across all batches.
    """
    totals: list[float] = []
    mses: list[float] = []
    dists: list[float] = []
    rmsds: list[float] = []

    step_rng = rng
    for host_batch in val_source.iter_batches(
        batch_size=batch_size,
        drop_last=False,
    ):
        step_rng, batch_rng = jax.random.split(step_rng)
        loss_out = cast(
            "losses_lib.LossOutput",
            _jit_eval_step(
                params,
                batch_rng,
                host_batch.to_jax(),
                model_config=model_config,
                loss_config=loss_config,
            ),
        )
        totals.append(float(loss_out.total_loss))
        mses.append(float(loss_out.aligned_mse_loss))
        dists.append(float(loss_out.distance_loss))
        rmsds.append(float(loss_out.rmsd))

    return ValidationMetrics(
        total_loss=float(np.mean(totals)),
        aligned_mse_loss=float(np.mean(mses)),
        distance_loss=float(np.mean(dists)),
        rmsd=float(np.mean(rmsds)),
    )


@jt.typed
def compute_sampled_rmsd_per_molecule(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: sampling_lib.SamplingConfig = sampling_lib.DEFAULT_SAMPLING_CONFIG,
) -> jt.Float[jt.Array, " batch"]:
    """Samples 3D conformers for a batch and returns per-molecule Kabsch-aligned RMSD.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for reverse-diffusion sampling.
        batch: Padded JAX batch containing target coordinates.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Per-molecule Kabsch-aligned coordinate RMSD in Angstroms.
    """
    denoiser = model_lib.Denoiser(config=model_config)
    generated = sampling_lib.sample_coordinates(
        rng,
        denoiser,
        params,
        batch=batch,
        config=sampling_config,
    )
    aligned = geometry.kabsch_align(
        reference=batch.positions,
        target=generated,
        atom_mask=batch.atom_mask,
    )
    # (batch,) valid atom counts per molecule, clamped to >= 1.
    atom_counts = jnp.maximum(jnp.sum(batch.atom_mask, axis=-1), 1)
    # (batch,) per-molecule Kabsch-aligned coordinate RMSD.
    return jnp.sqrt(jnp.sum(jnp.square(aligned - batch.positions), axis=(-1, -2)) / atom_counts)


@jt.typed
def evaluate_sampling_rmsd(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    val_batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: sampling_lib.SamplingConfig = sampling_lib.DEFAULT_SAMPLING_CONFIG,
) -> SamplingMetrics:
    """Generates 3D conformers from Gaussian noise and computes Kabsch-aligned RMSD.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for reverse-diffusion sampling.
        val_batch: Padded JAX validation batch containing target coordinates.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Mean and median Kabsch-aligned RMSD across the sampled validation batch.
    """
    rmsd_per_mol = compute_sampled_rmsd_per_molecule(
        params,
        rng,
        val_batch,
        model_config=model_config,
        sampling_config=sampling_config,
    )
    return SamplingMetrics(
        sampled_rmsd_mean=float(jnp.mean(rmsd_per_mol)),
        sampled_rmsd_median=float(jnp.median(rmsd_per_mol)),
    )


@jt.typed
def evaluate_dataset_sampling(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    eval_source: data_source.DataSource,
    *,
    batch_size: int,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: sampling_lib.SamplingConfig = sampling_lib.DEFAULT_SAMPLING_CONFIG,
) -> DatasetSamplingMetrics:
    """Evaluates reverse-diffusion conformer generation RMSD across an entire data source.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for reverse-diffusion sampling across batches.
        eval_source: Evaluation data source.
        batch_size: Batch size for reverse-diffusion rollouts.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Mean and median RMSD along with sub-0.5 A and sub-1.0 A coverage fractions.
    """
    batches = tuple(eval_source.iter_batches(batch_size=batch_size, drop_last=False))
    rmsd_chunks: list[np.ndarray] = []
    step_rng = rng

    with progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[bold blue]{task.description}"),
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        progress.TimeElapsedColumn(),
    ) as progress_bar:
        task_id = progress_bar.add_task("Sampling conformers", total=len(batches))
        for host_batch in batches:
            step_rng, batch_rng = jax.random.split(step_rng)
            batch_rmsd = compute_sampled_rmsd_per_molecule(
                params,
                batch_rng,
                host_batch.to_jax(),
                model_config=model_config,
                sampling_config=sampling_config,
            )
            rmsd_chunks.append(np.asarray(batch_rmsd, dtype=np.float32))
            progress_bar.advance(task_id)

    all_rmsds = np.concatenate(rmsd_chunks, axis=0)
    return DatasetSamplingMetrics(
        sampled_rmsd_mean=float(np.mean(all_rmsds)),
        sampled_rmsd_median=float(np.median(all_rmsds)),
        fraction_below_0_5a=float(np.mean(all_rmsds < _STRICT_RMSD_THRESHOLD_A)),
        fraction_below_1_0a=float(np.mean(all_rmsds < _LOOSE_RMSD_THRESHOLD_A)),
    )


def run_evaluation(config: EvalConfig) -> TestEvaluationReport:
    """Evaluates loss and conformer sampling for a saved run on a test split and writes JSON.

    Args:
        config: Test-set evaluation configuration.

    Returns:
        Combined loss and conformer sampling evaluation report.
    """
    loaded = checkpoint.load_run_checkpoint(
        config.run_dir,
        checkpoint_name=config.checkpoint_name,
    )
    active_params = loaded.checkpoint.ema_params if config.use_ema else loaded.checkpoint.params
    sampling_config = (
        dataclasses.replace(loaded.sampling_config, num_steps=config.num_sampling_steps)
        if config.num_sampling_steps is not None
        else loaded.sampling_config
    )

    test_source = data_source.DataSource(
        config.test_path,
        max_molecules=config.max_test_molecules,
    )
    loss_rng, sample_rng = jax.random.split(jax.random.key(config.seed))

    loss_metrics = evaluate_loss(
        active_params,
        loss_rng,
        test_source,
        batch_size=config.batch_size,
        model_config=loaded.model_config,
        loss_config=loaded.loss_config,
    )
    sampling_metrics = evaluate_dataset_sampling(
        active_params,
        sample_rng,
        test_source,
        batch_size=config.batch_size,
        model_config=loaded.model_config,
        sampling_config=sampling_config,
    )
    report = TestEvaluationReport(
        step=loaded.checkpoint.step,
        num_molecules=len(test_source),
        loss_metrics=loss_metrics,
        sampling_metrics=sampling_metrics,
    )

    output_path = config.output_path or (config.run_dir / "test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info(
        "Evaluated %d molecules at step=%d: test_loss=%.4f test_rmsd=%.3f A "
        "sampled_rmsd_mean=%.3f A sampled_rmsd_median=%.3f A (<0.5A=%.1f%%, <1.0A=%.1f%%)",
        report.num_molecules,
        report.step,
        loss_metrics.total_loss,
        loss_metrics.rmsd,
        sampling_metrics.sampled_rmsd_mean,
        sampling_metrics.sampled_rmsd_median,
        100.0 * sampling_metrics.fraction_below_0_5a,
        100.0 * sampling_metrics.fraction_below_1_0a,
    )
    return report


def _resolve_smiles_inputs(config: SampleSmilesConfig) -> tuple[str, ...]:
    """Returns the sequence of SMILES strings from either a file or CLI tuple."""
    if config.smiles_file is not None:
        lines = config.smiles_file.read_text(encoding="utf-8").splitlines()
        cleaned = tuple(line.strip() for line in lines if line.strip())
        if not cleaned:
            raise ValueError(f"Expected at least one SMILES in {config.smiles_file}.")
        return cleaned
    if not config.smiles:
        raise ValueError("Expected non-empty smiles when smiles_file is not set.")
    return config.smiles


def run_smiles_sampling(config: SampleSmilesConfig) -> str:
    """Generates 3D conformers for a list of SMILES strings and writes them to an XYZ file.

    Args:
        config: SMILES inference configuration.

    Returns:
        Concatenated multi-molecule XYZ coordinate string.
    """
    smiles_seq = _resolve_smiles_inputs(config)
    loaded = checkpoint.load_run_checkpoint(
        config.run_dir,
        checkpoint_name=config.checkpoint_name,
    )
    active_params = loaded.checkpoint.ema_params if config.use_ema else loaded.checkpoint.params
    sampling_config = (
        dataclasses.replace(loaded.sampling_config, num_steps=config.num_sampling_steps)
        if config.num_sampling_steps is not None
        else loaded.sampling_config
    )

    host_batch = batch_lib.collate_smiles(smiles_seq)
    denoiser = model_lib.Denoiser(config=loaded.model_config)
    generated = np.asarray(
        sampling_lib.sample_coordinates(
            jax.random.key(config.seed),
            denoiser,
            active_params,
            batch=host_batch.to_jax(),
            config=sampling_config,
        ),
        dtype=np.float32,
    )

    xyz_blocks = tuple(
        viz.format_xyz_block(
            generated[mol_idx],
            host_batch.atom_features[mol_idx],
            host_batch.atom_mask[mol_idx],
            comment=smiles,
        )
        for mol_idx, smiles in enumerate(smiles_seq)
    )
    xyz_content = "".join(xyz_blocks)
    config.output_xyz.parent.mkdir(parents=True, exist_ok=True)
    config.output_xyz.write_text(xyz_content, encoding="utf-8")
    _LOGGER.info(
        "Generated %d conformer(s) from checkpoint step=%d and wrote %s",
        len(smiles_seq),
        loaded.checkpoint.step,
        config.output_xyz,
    )
    return xyz_content
