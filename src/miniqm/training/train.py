"""Training state, Optax optimizer schedule, JIT train step, and experiment loop."""

from collections.abc import Iterator, Mapping
import dataclasses
import functools
import json
import logging
import pathlib
import time
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich import progress
import tensorboardX

from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.data import data_source
from miniqm.diffusion import losses as losses_lib
from miniqm.diffusion import model as model_lib
from miniqm.diffusion import noise as noise_lib
from miniqm.diffusion import sampling as sampling_lib
from miniqm.training import checkpoint
from miniqm.training import eval as eval_lib

_LOGGER: Final = logging.getLogger(__name__)
_SPATIAL_DIMS: Final = 3
_BYTES_PER_MB: Final = 1024 * 1024
_TABULATE_WIDTH: Final = 100


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainConfig:
    """Hyperparameters for model architecture, diffusion loss, and Optax optimization.

    Attributes:
        model: Denoiser architecture and EDM noise schedule configuration.
        loss: Diffusion loss weights and distance cutoff configuration.
        learning_rate: Peak learning rate after linear warmup.
        warmup_steps: Number of linear warmup steps from zero to peak learning rate.
        decay_steps: Total steps for the warmup + cosine decay schedule.
        min_learning_rate: Terminal learning rate at the end of cosine decay.
        weight_decay: AdamW decoupled weight decay coefficient.
        max_grad_norm: Global gradient norm clipping threshold.
        ema_decay: Exponential moving average decay factor for evaluation weights.
    """

    model: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG
    loss: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG
    learning_rate: float = 3e-4
    warmup_steps: int = 1_000
    decay_steps: int = 100_000
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(f"Expected positive learning_rate, got {self.learning_rate}.")
        if not (0.0 <= self.min_learning_rate <= self.learning_rate):
            raise ValueError(
                f"Expected 0 <= min_learning_rate <= learning_rate, got "
                f"min_learning_rate={self.min_learning_rate}, learning_rate={self.learning_rate}."
            )
        if self.warmup_steps < 0:
            raise ValueError(f"Expected non-negative warmup_steps, got {self.warmup_steps}.")
        if self.decay_steps <= self.warmup_steps:
            raise ValueError(
                f"Expected decay_steps ({self.decay_steps}) > warmup_steps ({self.warmup_steps})."
            )
        if self.weight_decay < 0.0:
            raise ValueError(f"Expected non-negative weight_decay, got {self.weight_decay}.")
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"Expected positive max_grad_norm, got {self.max_grad_norm}.")
        if not (0.0 <= self.ema_decay <= 1.0):
            raise ValueError(f"Expected ema_decay in [0, 1], got {self.ema_decay}.")


DEFAULT_TRAIN_CONFIG: Final = TrainConfig()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """End-to-end configuration for the training, evaluation, and checkpointing loop."""

    train_path: pathlib.Path = pathlib.Path("data/train.parquet")
    val_path: pathlib.Path = pathlib.Path("data/validation.parquet")
    run_dir: pathlib.Path = pathlib.Path("runs/qm9_default")
    train: TrainConfig = DEFAULT_TRAIN_CONFIG
    sampling: sampling_lib.SamplingConfig = sampling_lib.DEFAULT_SAMPLING_CONFIG
    batch_size: int = 64
    num_steps: int = 50_000
    log_every: int = 50
    eval_every: int = 500
    sample_every: int = 1_000
    max_train_molecules: int | None = None
    max_val_molecules: int | None = 512
    sample_batch_size: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"Expected positive batch_size, got {self.batch_size}.")
        if self.num_steps <= 0:
            raise ValueError(f"Expected positive num_steps, got {self.num_steps}.")
        if self.log_every <= 0:
            raise ValueError(f"Expected positive log_every, got {self.log_every}.")
        if self.eval_every <= 0:
            raise ValueError(f"Expected positive eval_every, got {self.eval_every}.")
        if self.sample_every <= 0:
            raise ValueError(f"Expected positive sample_every, got {self.sample_every}.")
        if self.sample_batch_size <= 0:
            raise ValueError(f"Expected positive sample_batch_size, got {self.sample_batch_size}.")


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainState:
    """Mutable training state carried across optimization steps as a JAX PyTree.

    Attributes:
        step: Current optimization step counter.
        params: Active model parameters updated by the optimizer.
        ema_params: Exponential moving average of model parameters for evaluation/sampling.
        opt_state: Optax optimizer state.
        rng: PRNG key advanced at each training step.
    """

    step: jt.Int[jt.Array, ""]
    params: flax_typing.VariableDict
    ema_params: flax_typing.VariableDict
    opt_state: optax.OptState
    rng: jt.Array


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainStepOutput:
    """Result of a single optimization step.

    Attributes:
        state: Updated training state.
        loss_output: Total loss and diagnostic breakdown from the forward pass.
        grad_norm: Global L2 norm of the unclipped parameter gradients.
        learning_rate: Learning rate applied at the current step.
    """

    state: TrainState
    loss_output: losses_lib.LossOutput
    grad_norm: jt.Float[jt.Array, ""]
    learning_rate: jt.Float[jt.Array, ""]


def learning_rate_schedule(config: TrainConfig = DEFAULT_TRAIN_CONFIG) -> optax.Schedule:
    """Builds the linear warmup and cosine decay learning rate schedule.

    Args:
        config: Training hyperparameter configuration.

    Returns:
        Optax schedule mapping step count to learning rate.
    """
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.min_learning_rate,
    )


def build_optimizer(config: TrainConfig = DEFAULT_TRAIN_CONFIG) -> optax.GradientTransformation:
    """Constructs the global-norm-clipped AdamW optimizer.

    Args:
        config: Training hyperparameter configuration.

    Returns:
        Chained Optax gradient transformation.
    """
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=learning_rate_schedule(config),
            weight_decay=config.weight_decay,
        ),
    )


@jt.typed
def init_train_state(
    rng: jt.Array,
    dummy_batch: batch_lib.JaxBatch,
    *,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> TrainState:
    """Initializes model parameters, EMA weights, and optimizer state from a sample batch.

    Args:
        rng: PRNG key split into parameter initialization and initial step state keys.
        dummy_batch: Padded JAX batch providing input feature shapes.
        config: Training hyperparameter configuration.

    Returns:
        Initial training state at step 0 with EMA weights equal to active weights.
    """
    init_rng, state_rng = jax.random.split(rng)
    batch_size, num_atoms = dummy_batch.atom_mask.shape
    dummy_positions = jnp.zeros((batch_size, num_atoms, _SPATIAL_DIMS), dtype=jnp.float32)
    dummy_sigma = jnp.ones((batch_size,), dtype=jnp.float32)

    denoiser = model_lib.Denoiser(config=config.model)
    params = denoiser.init(
        init_rng,
        dummy_positions,
        dummy_sigma,
        batch=dummy_batch,
    )
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(params)
    return TrainState(
        step=jnp.zeros((), dtype=jnp.int32),
        params=params,
        ema_params=params,
        opt_state=opt_state,
        rng=state_rng,
    )


@jt.typed
def train_step(
    state: TrainState,
    batch: batch_lib.JaxBatch,
    *,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> TrainStepOutput:
    """Executes one forward/backward diffusion training step and updates parameters and EMA.

    Args:
        state: Current training state.
        batch: Padded JAX batch with ground-truth 3D coordinates and molecular graph features.
        config: Training hyperparameter configuration.

    Returns:
        Updated training state, loss breakdown, unclipped gradient norm, and current learning rate.
    """
    step_rng, next_rng = jax.random.split(state.rng)
    noisy_sample = noise_lib.make_noisy_sample(
        step_rng,
        batch.positions,
        batch.atom_mask,
        config=config.model.noise,
    )

    def _loss_fn(
        params: flax_typing.VariableDict,
    ) -> tuple[jt.Float[jt.Array, ""], losses_lib.LossOutput]:
        loss_out = eval_lib.compute_batch_loss(
            params,
            noisy_sample,
            batch,
            model_config=config.model,
            loss_config=config.loss,
        )
        return loss_out.total_loss, loss_out

    (_, loss_output), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
    grad_norm = jnp.asarray(optax.tree.norm(grads), dtype=jnp.float32)

    optimizer = build_optimizer(config)
    updates, next_opt_state = optimizer.update(grads, state.opt_state, state.params)
    next_params = cast("flax_typing.VariableDict", optax.apply_updates(state.params, updates))
    next_ema_params = cast(
        "flax_typing.VariableDict",
        optax.incremental_update(
            next_params,
            state.ema_params,
            step_size=1.0 - config.ema_decay,
        ),
    )
    lr = jnp.asarray(learning_rate_schedule(config)(state.step), dtype=jnp.float32)

    next_state = TrainState(
        step=state.step + 1,
        params=next_params,
        ema_params=next_ema_params,
        opt_state=next_opt_state,
        rng=next_rng,
    )
    return TrainStepOutput(
        state=next_state,
        loss_output=loss_output,
        grad_norm=grad_norm,
        learning_rate=lr,
    )


@jt.typed
def eval_step(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> losses_lib.LossOutput:
    """Evaluates the diffusion loss on a batch without updating parameters.

    Args:
        params: Model parameters (typically EMA weights).
        rng: PRNG key for random 3D rotation augmentation and noise sampling.
        batch: Padded JAX batch with ground-truth 3D coordinates and molecular graph features.
        config: Training hyperparameter configuration.

    Returns:
        Diffusion loss breakdown on the randomly rotated and corrupted batch.
    """
    return eval_lib.eval_step(
        params,
        rng,
        batch,
        model_config=config.model,
        loss_config=config.loss,
    )


def count_parameters(params: flax_typing.VariableDict) -> int:
    """Returns the total number of scalar parameters in a variable dictionary."""
    return sum(int(leaf.size) for leaf in jax.tree.leaves(params))


def format_parameter_summary(params: flax_typing.VariableDict) -> str:
    """Formats a top-level submodule parameter count and memory breakdown.

    Args:
        params: Initialized Flax variable dictionary.

    Returns:
        Single-line summary of parameter counts per top-level submodule and total size.
    """
    root = params.get("params", params)
    module_parts = tuple(f"{name}: {count_parameters(subtree):,}" for name, subtree in root.items())
    total_params = count_parameters(params)
    total_mb = sum(int(leaf.nbytes) for leaf in jax.tree.leaves(params)) / _BYTES_PER_MB
    return f"{', '.join(module_parts)} | total: {total_params:,} ({total_mb:.2f} MB)"


@jt.typed
def tabulate_model(
    dummy_batch: batch_lib.JaxBatch,
    *,
    config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    depth: int = 1,
) -> str:
    """Renders a Flax Linen module summary table for the denoiser.

    Args:
        dummy_batch: Sample padded JAX batch providing input feature shapes.
        config: Denoiser architecture configuration.
        depth: Submodule tree depth to display in the summary table.

    Returns:
        Formatted table summarizing module inputs, outputs, and parameter counts.
    """
    batch_size, num_atoms = dummy_batch.atom_mask.shape
    dummy_positions = jnp.zeros((batch_size, num_atoms, _SPATIAL_DIMS), dtype=jnp.float32)
    dummy_sigma = jnp.ones((batch_size,), dtype=jnp.float32)
    denoiser = model_lib.Denoiser(config=config)
    with jt.disable_typechecking():
        return denoiser.tabulate(
            jax.random.key(0),
            dummy_positions,
            dummy_sigma,
            batch=dummy_batch,
            depth=depth,
            console_kwargs={"width": _TABULATE_WIDTH, "force_terminal": False},
        )


@functools.partial(jax.jit, static_argnames=("config",))
@jt.typed
def _jit_train_step(
    state: TrainState,
    batch: batch_lib.JaxBatch,
    *,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> TrainStepOutput:
    """Executes one compiled diffusion training step."""
    return train_step(state, batch, config=config)


def _infinite_train_batches(
    train_source: data_source.DataSource,
    *,
    batch_size: int,
    seed: int,
) -> Iterator[batch_lib.JaxBatch]:
    """Yields shuffled device batches indefinitely across training epochs."""
    drop_last = len(train_source) >= batch_size
    epoch = 0
    while True:
        epoch_rng = np.random.default_rng(seed + epoch)
        for host_batch in train_source.iter_batches(
            batch_size=batch_size,
            rng=epoch_rng,
            drop_last=drop_last,
        ):
            yield host_batch.to_jax()
        epoch += 1


def _write_scalars(
    writer: tensorboardX.SummaryWriter,
    metrics_path: pathlib.Path,
    *,
    step: int,
    split: str,
    scalars: Mapping[str, float],
) -> None:
    """Logs scalar metrics to both TensorBoardX and the JSONL metrics file."""
    for name, value in scalars.items():
        writer.add_scalar(f"{split}/{name}", value, global_step=step)
    writer.flush()
    record = {"step": step, "split": split, **scalars}
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_run_config(config: RunConfig) -> None:
    """Writes the serialized run configuration to the run directory.

    Args:
        config: End-to-end training and evaluation configuration.
    """
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_dict = dataclasses.asdict(config)
    for path_key in ("train_path", "val_path", "run_dir"):
        config_dict[path_key] = str(config_dict[path_key])
    (config.run_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_progress() -> progress.Progress:
    """Constructs the Rich progress bar for the training step loop."""
    return progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[bold blue]{task.description}"),
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        progress.TextColumn("{task.fields[metrics]}"),
    )


def run_training(config: RunConfig) -> TrainState:
    """Executes the full training, validation, sampling, and checkpointing loop.

    Args:
        config: End-to-end training and evaluation configuration.

    Returns:
        Final training state after all configured optimization steps.
    """
    write_run_config(config)
    checkpoints_dir = config.run_dir / "checkpoints"
    metrics_path = config.run_dir / "metrics.jsonl"

    devices = jax.devices()
    _LOGGER.info(
        "Running on %d %s device(s): %s",
        len(devices),
        jax.default_backend(),
        ", ".join(str(device) for device in devices),
    )

    train_source = data_source.DataSource(
        config.train_path,
        max_molecules=config.max_train_molecules,
    )
    val_source = data_source.DataSource(
        config.val_path,
        max_molecules=config.max_val_molecules,
    )
    sample_val_batch = next(
        val_source.iter_batches(
            batch_size=min(config.sample_batch_size, len(val_source)),
            drop_last=False,
        )
    ).to_jax()

    batch_iter = _infinite_train_batches(
        train_source,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    first_batch = next(batch_iter)

    init_rng, eval_rng, sample_rng = jax.random.split(jax.random.key(config.seed), 3)
    state = init_train_state(init_rng, first_batch, config=config.train)
    _LOGGER.info("Parameter summary: %s", format_parameter_summary(state.params))
    _LOGGER.info(
        "Model architecture:\n%s",
        tabulate_model(first_batch, config=config.train.model),
    )

    best_val_loss = float("inf")
    start_time = time.monotonic()

    with (
        tensorboardX.SummaryWriter(logdir=str(config.run_dir)) as writer,
        _make_progress() as progress_bar,
    ):
        task_id = progress_bar.add_task("Training", total=config.num_steps, metrics="")
        current_batch = first_batch
        for step_idx in range(1, config.num_steps + 1):
            step_out = cast(
                "TrainStepOutput",
                _jit_train_step(state, current_batch, config=config.train),
            )
            state = step_out.state
            progress_bar.advance(task_id)

            if step_idx % config.log_every == 0 or step_idx in {1, config.num_steps}:
                elapsed_s = time.monotonic() - start_time
                train_scalars = {
                    "total_loss": float(step_out.loss_output.total_loss),
                    "aligned_mse_loss": float(step_out.loss_output.aligned_mse_loss),
                    "distance_loss": float(step_out.loss_output.distance_loss),
                    "rmsd": float(step_out.loss_output.rmsd),
                    "grad_norm": float(step_out.grad_norm),
                    "learning_rate": float(step_out.learning_rate),
                    "wall_time_s": elapsed_s,
                }
                _write_scalars(
                    writer,
                    metrics_path,
                    step=step_idx,
                    split="train",
                    scalars=train_scalars,
                )
                progress_bar.update(
                    task_id,
                    metrics=(
                        f"loss={train_scalars['total_loss']:.4f} "
                        f"rmsd={train_scalars['rmsd']:.3f}A "
                        f"lr={train_scalars['learning_rate']:.2e}"
                    ),
                )
                _LOGGER.info(
                    "step=%d/%d train_loss=%.4f rmsd=%.3f A grad_norm=%.2f lr=%.2e",
                    step_idx,
                    config.num_steps,
                    train_scalars["total_loss"],
                    train_scalars["rmsd"],
                    train_scalars["grad_norm"],
                    train_scalars["learning_rate"],
                )

            if step_idx % config.eval_every == 0 or step_idx == config.num_steps:
                val_metrics = eval_lib.evaluate_loss(
                    state.ema_params,
                    eval_rng,
                    val_source,
                    batch_size=config.batch_size,
                    model_config=config.train.model,
                    loss_config=config.train.loss,
                )
                _write_scalars(
                    writer,
                    metrics_path,
                    step=step_idx,
                    split="val",
                    scalars=dataclasses.asdict(val_metrics),
                )
                ckpt_payload = checkpoint.Checkpoint(
                    step=int(state.step),
                    params=state.params,
                    ema_params=state.ema_params,
                )
                checkpoint.save_checkpoint(checkpoints_dir / "latest.msgpack", ckpt_payload)
                if val_metrics.total_loss < best_val_loss:
                    best_val_loss = val_metrics.total_loss
                    checkpoint.save_checkpoint(checkpoints_dir / "best.msgpack", ckpt_payload)
                _LOGGER.info(
                    "step=%d/%d val_loss=%.4f val_rmsd=%.3f A (best=%.4f)",
                    step_idx,
                    config.num_steps,
                    val_metrics.total_loss,
                    val_metrics.rmsd,
                    best_val_loss,
                )

            if step_idx % config.sample_every == 0 or step_idx == config.num_steps:
                sampling_metrics = eval_lib.evaluate_sampling_rmsd(
                    state.ema_params,
                    sample_rng,
                    sample_val_batch,
                    model_config=config.train.model,
                    sampling_config=config.sampling,
                )
                _write_scalars(
                    writer,
                    metrics_path,
                    step=step_idx,
                    split="val_sample",
                    scalars=dataclasses.asdict(sampling_metrics),
                )
                _LOGGER.info(
                    "step=%d/%d sampled_rmsd_mean=%.3f A sampled_rmsd_median=%.3f A",
                    step_idx,
                    config.num_steps,
                    sampling_metrics.sampled_rmsd_mean,
                    sampling_metrics.sampled_rmsd_median,
                )

            if step_idx < config.num_steps:
                current_batch = next(batch_iter)

    return state
