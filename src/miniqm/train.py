"""Training state, Optax optimizer schedule, and JIT-ready train/eval steps."""

import dataclasses
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import optax

from miniqm import batch as batch_lib
from miniqm import jaxtypes as jt
from miniqm import losses as losses_lib
from miniqm import model as model_lib
from miniqm import noise as noise_lib

_SPATIAL_DIMS: Final = 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrainConfig:
    """Hyperparameters for model architecture, diffusion loss, and Optax optimization.

    Attributes:
        model: Denoiser architecture and EDM noise schedule configuration.
        loss: Diffusion loss weights and distance cutoff configuration.
        learning_rate: Peak learning rate after linear warmup.
        warmup_steps: Number of linear warmup steps from 0.0 to `learning_rate`.
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
        Initial `TrainState` at step 0 with `ema_params` equal to `params`.
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
def _compute_batch_loss(
    params: flax_typing.VariableDict,
    noisy_sample: noise_lib.NoisySample,
    batch: batch_lib.JaxBatch,
    *,
    config: TrainConfig,
) -> losses_lib.LossOutput:
    """Runs the denoiser on a corrupted sample and evaluates the diffusion training loss."""
    denoiser = model_lib.Denoiser(config=config.model)
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
        config=config.loss,
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
        state: Current `TrainState`.
        batch: Padded JAX batch with ground-truth 3D `positions` and molecular graph features.
        config: Training hyperparameter configuration.

    Returns:
        Updated `TrainState`, loss breakdown, unclipped gradient norm, and current learning rate.
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
        loss_out = _compute_batch_loss(params, noisy_sample, batch, config=config)
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
        params: Model parameters (typically `state.ema_params`).
        rng: PRNG key for random 3D rotation augmentation and noise sampling.
        batch: Padded JAX batch with ground-truth 3D `positions` and molecular graph features.
        config: Training hyperparameter configuration.

    Returns:
        Diffusion loss breakdown on the randomly rotated and corrupted batch.
    """
    noisy_sample = noise_lib.make_noisy_sample(
        rng,
        batch.positions,
        batch.atom_mask,
        config=config.model.noise,
    )
    return _compute_batch_loss(params, noisy_sample, batch, config=config)
