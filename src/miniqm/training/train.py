"""Training state, Optax optimizer schedule, JIT train step, and experiment loop."""

from collections.abc import Iterator, Mapping
import dataclasses
import datetime as dt
import enum
import functools
import json
import logging
import math
import os
import pathlib
import time
from typing import Final, cast

from flax import typing as flax_typing
import jax
from jax import sharding as jax_sharding
import jax.numpy as jnp
import numpy as np
import optax
from rich import progress
import tensorboardX

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.data import data_source
from miniqm.diffusion import losses as losses_lib
from miniqm.diffusion import model as model_lib
from miniqm.diffusion import noise as noise_lib
from miniqm.training import checkpoint
from miniqm.training import eval as eval_lib

_LOGGER: Final = logging.getLogger(__name__)
_SPATIAL_DIMS: Final = 3
_BYTES_PER_MB: Final = 1024 * 1024
_TABULATE_WIDTH: Final = 100


class MetricSplit(enum.StrEnum):
    """Split identifiers written to TensorBoard and JSONL metric logs."""

    TRAIN = "train"
    VAL = "val"
    VAL_SAMPLE = "val_sample"


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainState:
    """Training state PyTree carried across optimization steps."""

    step: jt.Int[jt.Array, ""]
    params: flax_typing.VariableDict
    ema_params: flax_typing.VariableDict
    opt_state: optax.OptState
    rng: jt.Array


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainStepOutput:
    """Updated training state and diagnostic scalars from a single optimization step."""

    state: TrainState
    loss_output: losses_lib.LossOutput
    grad_norm: jt.Float[jt.Array, ""]
    learning_rate: jt.Float[jt.Array, ""]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _DeviceShardings:
    """Primary device and mesh shardings for replicated state and data-parallel batches."""

    primary_device: jax.Device
    replicated_sharding: jax_sharding.NamedSharding
    batch_sharding: jax_sharding.NamedSharding


def learning_rate_schedule(
    config: config_lib.TrainConfig = config_lib.DEFAULT_TRAIN_CONFIG,
) -> optax.Schedule:
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


def build_optimizer(
    config: config_lib.TrainConfig = config_lib.DEFAULT_TRAIN_CONFIG,
) -> optax.GradientTransformation:
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
    config: config_lib.TrainConfig = config_lib.DEFAULT_TRAIN_CONFIG,
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
    config: config_lib.TrainConfig = config_lib.DEFAULT_TRAIN_CONFIG,
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

    @jt.typed
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
    config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG,
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
    config: config_lib.TrainConfig = config_lib.DEFAULT_TRAIN_CONFIG,
) -> TrainStepOutput:
    """Executes one compiled diffusion training step."""
    return train_step(state, batch, config=config)


def configure_cpu_devices(num_devices: int) -> None:
    """Configures the default CPU platform and logical XLA CPU device count.

    Args:
        num_devices: Number of logical XLA CPU devices to expose when uninitialized.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        jax.config.update("jax_num_cpu_devices", num_devices)
    except RuntimeError:
        _LOGGER.debug(
            "JAX backend already initialized; keeping %d active device(s).",
            len(jax.devices()),
        )


def _build_device_shardings(
    *,
    num_devices: int,
    effective_batch_size: int,
) -> _DeviceShardings:
    """Configures CPU devices and builds replicated state and batch shardings."""
    configure_cpu_devices(num_devices)
    devices = jax.devices()[:num_devices]
    active_devices = tuple(devices[: math.gcd(len(devices), effective_batch_size)])
    mesh = jax_sharding.Mesh(active_devices, ("batch",))
    replicated_sharding = jax_sharding.NamedSharding(mesh, jax_sharding.PartitionSpec())
    batch_sharding = jax_sharding.NamedSharding(mesh, jax_sharding.PartitionSpec("batch"))
    _LOGGER.info(
        "Running on %d %s device(s) (batch sharded across %d): %s",
        len(devices),
        jax.default_backend(),
        len(active_devices),
        ", ".join(str(device) for device in active_devices),
    )
    return _DeviceShardings(
        primary_device=devices[0],
        replicated_sharding=replicated_sharding,
        batch_sharding=batch_sharding,
    )


def _infinite_train_batches(
    train_source: data_source.DataSource,
    *,
    batch_size: int,
    seed: int,
    sharding: jax_sharding.Sharding | None = None,
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
            jax_batch = host_batch.to_jax()
            yield (
                jax_batch
                if sharding is None
                else cast("batch_lib.JaxBatch", jax.device_put(jax_batch, sharding))
            )
        epoch += 1


def _write_scalars(
    writer: tensorboardX.SummaryWriter,
    metrics_path: pathlib.Path,
    *,
    step: int,
    split: MetricSplit,
    scalar_by_name: Mapping[str, float],
) -> None:
    """Logs scalar metrics to both TensorBoardX and the JSONL metrics file."""
    for name, value in scalar_by_name.items():
        writer.add_scalar(f"{split}/{name}", value, global_step=step)
    writer.flush()
    record = {"step": step, "split": str(split), **scalar_by_name}
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_run_config(config: config_lib.RunConfig) -> None:
    """Writes the serialized run configuration to the run directory.

    Args:
        config: End-to-end training and evaluation configuration.
    """
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / config_lib.ArtifactName.CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    (config.run_dir / config_lib.ArtifactName.CONFIG).write_text(
        config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _init_timestamped_run_dir(config: config_lib.RunConfig) -> pathlib.Path:
    """Creates a timestamped run subdirectory and writes the serialized configuration."""
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    active_run_dir = config.run_dir / timestamp
    write_run_config(config.model_copy(update={"run_dir": active_run_dir}))
    _LOGGER.info("Saving run artifacts to %s", active_run_dir)
    return active_run_dir


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


def _save_state_checkpoint(
    path: pathlib.Path,
    state: TrainState,
    primary_device: jax.Device,
) -> None:
    """Serializes the current training state onto disk via the primary device."""
    checkpoint_data = checkpoint.Checkpoint(
        step=int(state.step),
        params=jax.device_put(state.params, primary_device),
        ema_params=jax.device_put(state.ema_params, primary_device),
    )
    checkpoint.save_checkpoint(path, checkpoint_data)


def _save_interrupted_checkpoints(
    checkpoints_dir: pathlib.Path,
    state: TrainState,
    primary_device: jax.Device,
    *,
    num_steps: int,
) -> None:
    """Persists the latest (and fallback best) checkpoint when training is interrupted."""
    latest_path = checkpoints_dir / config_lib.CheckpointName.LATEST
    best_path = checkpoints_dir / config_lib.CheckpointName.BEST
    _save_state_checkpoint(latest_path, state, primary_device)
    if not best_path.is_file():
        _save_state_checkpoint(best_path, state, primary_device)
    _LOGGER.warning(
        "Training interrupted at step %d/%d; saved latest checkpoint to %s.",
        int(state.step),
        num_steps,
        latest_path,
    )


def _log_train_step(
    writer: tensorboardX.SummaryWriter,
    metrics_path: pathlib.Path,
    progress_bar: progress.Progress,
    task_id: progress.TaskID,
    *,
    step_idx: int,
    num_steps: int,
    step_out: TrainStepOutput,
    elapsed_s: float,
) -> None:
    """Logs training metrics to TensorBoard, JSONL, the progress bar, and the logger."""
    loss_out = step_out.loss_output
    total_loss = float(loss_out.total_loss)
    rmsd = float(loss_out.rmsd)
    grad_norm = float(step_out.grad_norm)
    learning_rate = float(step_out.learning_rate)
    _write_scalars(
        writer,
        metrics_path,
        step=step_idx,
        split=MetricSplit.TRAIN,
        scalar_by_name={
            "total_loss": total_loss,
            "aligned_mse_loss": float(loss_out.aligned_mse_loss),
            "distance_loss": float(loss_out.distance_loss),
            "rmsd": rmsd,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "wall_time_s": elapsed_s,
        },
    )
    progress_bar.update(
        task_id,
        metrics=(f"loss={total_loss:.4f} rmsd={rmsd:.3f}A lr={learning_rate:.2e}"),
    )
    _LOGGER.info(
        "step=%d/%d train_loss=%.4f rmsd=%.3f A grad_norm=%.2f lr=%.2e",
        step_idx,
        num_steps,
        total_loss,
        rmsd,
        grad_norm,
        learning_rate,
    )


def run_training(config: config_lib.RunConfig) -> TrainState:
    """Executes the full training, validation, sampling, and checkpointing loop.

    Args:
        config: End-to-end training and evaluation configuration.

    Returns:
        Final training state after all configured optimization steps.
    """
    active_run_dir = _init_timestamped_run_dir(config)
    checkpoints_dir = active_run_dir / config_lib.ArtifactName.CHECKPOINTS_DIR
    metrics_path = active_run_dir / config_lib.ArtifactName.METRICS

    configure_cpu_devices(config.num_devices)
    train_source = data_source.DataSource(
        config.train_path,
        max_molecules=config.max_train_molecules,
    )
    val_source = data_source.DataSource(
        config.val_path,
        max_molecules=config.max_val_molecules,
    )
    shardings = _build_device_shardings(
        num_devices=config.num_devices,
        effective_batch_size=min(config.batch_size, len(train_source)),
    )

    sample_val_batch = next(
        val_source.iter_batches(
            batch_size=min(config.sample_batch_size, len(val_source)),
            drop_last=False,
        )
    ).to_jax()

    unsharded_first_batch = next(
        train_source.iter_batches(
            batch_size=config.batch_size,
            drop_last=len(train_source) >= config.batch_size,
        )
    ).to_jax()
    init_rng, eval_rng, sample_rng = jax.random.split(jax.random.key(config.seed), 3)
    state = init_train_state(init_rng, unsharded_first_batch, config=config.train)
    _LOGGER.info("Parameter summary: %s", format_parameter_summary(state.params))
    _LOGGER.info(
        "Model architecture:\n%s",
        tabulate_model(unsharded_first_batch, config=config.train.model),
    )

    state = cast("TrainState", jax.device_put(state, shardings.replicated_sharding))
    batch_iter = _infinite_train_batches(
        train_source,
        batch_size=config.batch_size,
        seed=config.seed,
        sharding=shardings.batch_sharding,
    )
    first_batch = next(batch_iter)

    best_sampled_lddt = -1.0
    start_time = time.perf_counter()

    with (
        tensorboardX.SummaryWriter(logdir=str(active_run_dir)) as writer,
        _make_progress() as progress_bar,
    ):
        task_id = progress_bar.add_task("Training", total=config.num_steps, metrics="")
        current_batch = first_batch
        try:
            for step_idx in range(1, config.num_steps + 1):
                step_out = cast(
                    "TrainStepOutput",
                    _jit_train_step(state, current_batch, config=config.train),
                )
                state = step_out.state
                progress_bar.advance(task_id)

                if step_idx % config.log_every == 0 or step_idx in {1, config.num_steps}:
                    _log_train_step(
                        writer,
                        metrics_path,
                        progress_bar,
                        task_id,
                        step_idx=step_idx,
                        num_steps=config.num_steps,
                        step_out=step_out,
                        elapsed_s=time.perf_counter() - start_time,
                    )

                if step_idx % config.eval_every == 0 or step_idx == config.num_steps:
                    single_device_ema = jax.device_put(state.ema_params, shardings.primary_device)
                    val_metrics = eval_lib.evaluate_loss(
                        single_device_ema,
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
                        split=MetricSplit.VAL,
                        scalar_by_name=val_metrics.model_dump(),
                    )
                    _save_state_checkpoint(
                        checkpoints_dir / config_lib.CheckpointName.LATEST,
                        state,
                        shardings.primary_device,
                    )
                    _LOGGER.info(
                        "step=%d/%d val_loss=%.4f val_rmsd=%.3f A",
                        step_idx,
                        config.num_steps,
                        val_metrics.total_loss,
                        val_metrics.rmsd,
                    )

                if step_idx % config.sample_every == 0 or step_idx == config.num_steps:
                    single_device_ema = jax.device_put(state.ema_params, shardings.primary_device)
                    sampling_metrics = eval_lib.evaluate_sampling_rmsd(
                        single_device_ema,
                        sample_rng,
                        sample_val_batch,
                        model_config=config.train.model,
                        sampling_config=config.sampling,
                    )
                    _write_scalars(
                        writer,
                        metrics_path,
                        step=step_idx,
                        split=MetricSplit.VAL_SAMPLE,
                        scalar_by_name=sampling_metrics.model_dump(),
                    )
                    if sampling_metrics.sampled_lddt_mean > best_sampled_lddt:
                        best_sampled_lddt = sampling_metrics.sampled_lddt_mean
                        _save_state_checkpoint(
                            checkpoints_dir / config_lib.CheckpointName.BEST,
                            state,
                            shardings.primary_device,
                        )
                    _LOGGER.info(
                        "step=%d/%d sampled_rmsd_mean=%.3f A sampled_lddt_mean=%.3f (best=%.3f) "
                        "stereo_violations=%.1f%%",
                        step_idx,
                        config.num_steps,
                        sampling_metrics.sampled_rmsd_mean,
                        sampling_metrics.sampled_lddt_mean,
                        best_sampled_lddt,
                        100.0 * sampling_metrics.stereocenter_violation_rate,
                    )

                if step_idx < config.num_steps:
                    current_batch = next(batch_iter)
        except KeyboardInterrupt:
            _save_interrupted_checkpoints(
                checkpoints_dir,
                state,
                shardings.primary_device,
                num_steps=config.num_steps,
            )

    return state
