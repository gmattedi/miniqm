"""Validation loss evaluation, test-set benchmarking, and SMILES conformer inference."""

import dataclasses
import functools
import itertools
import logging
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from rdkit import Chem
from rich import progress

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.data import data_source
from miniqm.diffusion import losses as losses_lib
from miniqm.diffusion import metrics as metrics_lib
from miniqm.diffusion import model as model_lib
from miniqm.diffusion import noise as noise_lib
from miniqm.diffusion import sampling as sampling_lib
from miniqm.training import checkpoint

_LOGGER: Final = logging.getLogger(__name__)
# Standard QM9 conformer generation RMSD coverage thresholds (in Angstroms).
_STRICT_RMSD_THRESHOLD_A: Final = 0.5
_LOOSE_RMSD_THRESHOLD_A: Final = 1.0


class ValidationMetrics(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Teacher-forced single-step validation loss breakdown across an evaluation split."""

    total_loss: float
    aligned_mse_loss: float
    distance_loss: float
    rmsd: float


class SamplingMetrics(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Reverse-diffusion conformer generation RMSD, lDDT, and stereocenter statistics."""

    sampled_rmsd_mean: float
    sampled_rmsd_median: float
    sampled_lddt_mean: float
    sampled_lddt_median: float
    stereocenter_violation_rate: float


class DatasetSamplingMetrics(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Full reverse-diffusion conformer generation RMSD, lDDT, coverage, and stereocenter stats."""

    sampled_rmsd_mean: float
    sampled_rmsd_median: float
    sampled_lddt_mean: float
    sampled_lddt_median: float
    fraction_below_0_5a: float
    fraction_below_1_0a: float
    stereocenter_violation_rate: float


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SampledBatchStats:
    """Per-molecule Kabsch RMSD, local lDDT, and stereocenter counts for generated conformers."""

    rmsd: jt.Float[jt.Array, " batch"]
    lddt: jt.Float[jt.Array, " batch"]
    num_stereo_violations: jt.Int[jt.Array, " batch"]
    num_stereocenters: jt.Int[jt.Array, " batch"]


class TestEvaluationReport(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Combined teacher-forced loss and reverse-diffusion sampling report on a test split."""

    step: int
    num_molecules: int
    loss_metrics: ValidationMetrics
    sampling_metrics: DatasetSamplingMetrics


@jt.typed
def compute_batch_loss(
    params: flax_typing.VariableDict,
    noisy_sample: noise_lib.NoisySample,
    batch: batch_lib.JaxBatch,
    *,
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: config_lib.LossConfig = config_lib.DEFAULT_LOSS_CONFIG,
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
        noise_config=model_config.noise,
    )


@jt.typed
def eval_step(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: config_lib.LossConfig = config_lib.DEFAULT_LOSS_CONFIG,
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
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: config_lib.LossConfig = config_lib.DEFAULT_LOSS_CONFIG,
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
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: config_lib.LossConfig = config_lib.DEFAULT_LOSS_CONFIG,
) -> ValidationMetrics:
    """Computes molecule-weighted mean teacher-forced diffusion loss and metrics across batches.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for validation rotation augmentation and noise sampling.
        val_source: Validation data source.
        batch_size: Validation batch size.
        model_config: Denoiser architecture and noise schedule configuration.
        loss_config: Diffusion loss weights and distance cutoff configuration.

    Returns:
        Molecule-weighted mean validation loss components across all batches.
    """
    batch_outputs: list[tuple[losses_lib.LossOutput, int]] = []
    step_rng = rng
    for host_batch in val_source.iter_batches(batch_size=batch_size, drop_last=False):
        step_rng, batch_rng = jax.random.split(step_rng)
        batch_count, _ = host_batch.atom_mask.shape
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
        batch_outputs.append((loss_out, batch_count))

    weights = np.array(tuple(count for _, count in batch_outputs), dtype=np.float64)
    return ValidationMetrics(
        total_loss=float(
            np.average(tuple(float(out.total_loss) for out, _ in batch_outputs), weights=weights)
        ),
        aligned_mse_loss=float(
            np.average(
                tuple(float(out.aligned_mse_loss) for out, _ in batch_outputs), weights=weights
            )
        ),
        distance_loss=float(
            np.average(tuple(float(out.distance_loss) for out, _ in batch_outputs), weights=weights)
        ),
        rmsd=float(np.average(tuple(float(out.rmsd) for out, _ in batch_outputs), weights=weights)),
    )


@jt.typed
def compute_sampled_metrics_per_molecule(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> SampledBatchStats:
    """Samples 3D conformers for a batch and returns per-molecule RMSD, lDDT, and stereo counts.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for reverse-diffusion sampling.
        batch: Padded JAX batch containing target coordinates.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Per-molecule Kabsch-aligned coordinate RMSD, local lDDT, and stereocenter violation counts.
    """
    denoiser = model_lib.Denoiser(config=model_config)
    generated = sampling_lib.sample_coordinates(
        rng,
        denoiser,
        params,
        batch=batch,
        config=sampling_config,
    )
    rmsd = metrics_lib.rmsd_per_molecule(
        predicted_positions=generated,
        target_positions=batch.positions,
        atom_mask=batch.atom_mask,
    )
    lddt = metrics_lib.lddt_per_molecule(
        predicted_positions=generated,
        target_positions=batch.positions,
        atom_mask=batch.atom_mask,
    )
    stereo_stats = metrics_lib.stereocenter_violations(
        predicted_positions=generated,
        target_positions=batch.positions,
        atom_features=batch.atom_features,
        bond_features=batch.bond_features,
        atom_mask=batch.atom_mask,
    )
    return SampledBatchStats(
        rmsd=rmsd,
        lddt=lddt,
        num_stereo_violations=stereo_stats.num_violations,
        num_stereocenters=stereo_stats.num_stereocenters,
    )


@jt.typed
def evaluate_sampling_rmsd(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    val_batch: batch_lib.JaxBatch,
    *,
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> SamplingMetrics:
    """Generates 3D conformers from Gaussian noise and computes RMSD, lDDT, and stereo violations.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for reverse-diffusion sampling.
        val_batch: Padded JAX validation batch containing target coordinates.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Mean/median Kabsch RMSD and lDDT plus stereocenter violation rate across the batch.
    """
    stats = compute_sampled_metrics_per_molecule(
        params,
        rng,
        val_batch,
        model_config=model_config,
        sampling_config=sampling_config,
    )
    total_violations = int(jnp.sum(stats.num_stereo_violations))
    total_stereocenters = int(jnp.sum(stats.num_stereocenters))
    return SamplingMetrics(
        sampled_rmsd_mean=float(jnp.mean(stats.rmsd)),
        sampled_rmsd_median=float(jnp.median(stats.rmsd)),
        sampled_lddt_mean=float(jnp.mean(stats.lddt)),
        sampled_lddt_median=float(jnp.median(stats.lddt)),
        stereocenter_violation_rate=total_violations / max(total_stereocenters, 1),
    )


@jt.typed
def evaluate_dataset_sampling(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    eval_source: data_source.DataSource,
    *,
    batch_size: int,
    model_config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> DatasetSamplingMetrics:
    """Evaluates reverse-diffusion conformer generation RMSD, lDDT, and stereo across a data source.

    Args:
        params: Model parameters to evaluate (typically EMA weights).
        rng: PRNG key for reverse-diffusion sampling across batches.
        eval_source: Evaluation data source.
        batch_size: Batch size for reverse-diffusion rollouts.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Mean/median RMSD and lDDT, coverage fractions, and stereocenter violation rate.
    """
    batches = tuple(eval_source.iter_batches(batch_size=batch_size, drop_last=False))
    rmsd_chunks: list[np.ndarray] = []
    lddt_chunks: list[np.ndarray] = []
    total_violations = 0
    total_stereocenters = 0
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
            stats = compute_sampled_metrics_per_molecule(
                params,
                batch_rng,
                host_batch.to_jax(),
                model_config=model_config,
                sampling_config=sampling_config,
            )
            rmsd_chunks.append(np.asarray(stats.rmsd, dtype=np.float32))
            lddt_chunks.append(np.asarray(stats.lddt, dtype=np.float32))
            total_violations += int(jnp.sum(stats.num_stereo_violations))
            total_stereocenters += int(jnp.sum(stats.num_stereocenters))
            progress_bar.advance(task_id)

    all_rmsds = np.concatenate(rmsd_chunks, axis=0)
    all_lddts = np.concatenate(lddt_chunks, axis=0)
    return DatasetSamplingMetrics(
        sampled_rmsd_mean=float(np.mean(all_rmsds)),
        sampled_rmsd_median=float(np.median(all_rmsds)),
        sampled_lddt_mean=float(np.mean(all_lddts)),
        sampled_lddt_median=float(np.median(all_lddts)),
        fraction_below_0_5a=float(np.mean(all_rmsds < _STRICT_RMSD_THRESHOLD_A)),
        fraction_below_1_0a=float(np.mean(all_rmsds < _LOOSE_RMSD_THRESHOLD_A)),
        stereocenter_violation_rate=total_violations / max(total_stereocenters, 1),
    )


def run_evaluation(config: config_lib.EvalConfig) -> TestEvaluationReport:
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
        loaded.sampling_config.model_copy(update={"num_steps": config.num_sampling_steps})
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

    output_path = config.output_path or (loaded.run_dir / config_lib.ArtifactName.TEST_METRICS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info(
        "Evaluated %d molecules at step=%d: test_loss=%.4f test_rmsd=%.3f A "
        "sampled_rmsd_mean=%.3f A sampled_lddt_mean=%.3f stereo_violations=%.1f%% "
        "(<0.5A=%.1f%%, <1.0A=%.1f%%)",
        report.num_molecules,
        report.step,
        loss_metrics.total_loss,
        loss_metrics.rmsd,
        sampling_metrics.sampled_rmsd_mean,
        sampling_metrics.sampled_lddt_mean,
        100.0 * sampling_metrics.stereocenter_violation_rate,
        100.0 * sampling_metrics.fraction_below_0_5a,
        100.0 * sampling_metrics.fraction_below_1_0a,
    )
    return report


def _resolve_smiles_inputs(config: config_lib.SampleSmilesConfig) -> tuple[str, ...]:
    """Returns the sequence of SMILES strings from either a file or CLI tuple."""
    if config.smiles_file is not None:
        lines = config.smiles_file.read_text(encoding="utf-8").splitlines()
        cleaned = tuple(filter(None, map(str.strip, lines)))
        if not cleaned:
            raise ValueError(f"Expected at least one SMILES in {config.smiles_file}.")
        return cleaned
    if not config.smiles:
        raise ValueError("Expected non-empty smiles when smiles_file is not set.")
    return config.smiles


@jt.typed
def _build_tagged_conformer(
    smiles: str,
    positions: jt.Float[np.ndarray, "max_atoms 3"],
    *,
    conformer_idx: int,
    seed: int,
) -> Chem.Mol:
    """Builds a 3D conformer molecule annotated with its conformer index and PRNG seed."""
    mol = batch_lib.build_conformer_mol(smiles, positions)
    mol.SetProp(batch_lib.SdfProperty.CONFORMER_ID, str(conformer_idx))
    mol.SetProp(batch_lib.SdfProperty.SEED, str(seed))
    return mol


def run_smiles_sampling(config: config_lib.SampleSmilesConfig) -> tuple[Chem.Mol, ...]:
    """Generates 3D conformers for a list of SMILES strings and writes them to an SDF file.

    Args:
        config: SMILES inference configuration.

    Returns:
        RDKit molecules populated with the sampled 3D conformers.
    """
    smiles = _resolve_smiles_inputs(config)
    loaded = checkpoint.load_run_checkpoint(
        config.run_dir,
        checkpoint_name=config.checkpoint_name,
    )
    active_params = loaded.checkpoint.ema_params if config.use_ema else loaded.checkpoint.params
    sampling_config = (
        loaded.sampling_config.model_copy(update={"num_steps": config.num_sampling_steps})
        if config.num_sampling_steps is not None
        else loaded.sampling_config
    )

    jax_batch = batch_lib.collate_smiles(smiles).to_jax()
    denoiser = model_lib.Denoiser(config=loaded.model_config)
    generated_per_seed = tuple(
        np.asarray(
            sampling_lib.sample_coordinates(
                jax.random.key(config.seed + conf_idx),
                denoiser,
                active_params,
                batch=jax_batch,
                config=sampling_config,
            ),
            dtype=np.float32,
        )
        for conf_idx in range(config.num_conformers)
    )

    conformer_mols = tuple(
        _build_tagged_conformer(
            mol_smiles,
            generated[mol_idx],
            conformer_idx=conf_idx,
            seed=config.seed + conf_idx,
        )
        for (mol_idx, mol_smiles), (conf_idx, generated) in itertools.product(
            enumerate(smiles),
            enumerate(generated_per_seed),
        )
    )
    config.output_sdf.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(config.output_sdf)) as writer:
        for mol in conformer_mols:
            writer.write(mol)
    _LOGGER.info(
        "Generated %d conformer(s) (%d per molecule) from checkpoint step=%d and wrote %s",
        len(conformer_mols),
        config.num_conformers,
        loaded.checkpoint.step,
        config.output_sdf,
    )
    return conformer_mols
