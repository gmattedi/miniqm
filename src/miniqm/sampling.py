"""Reverse-time Karras Euler sampler with stochastic churn for 3D coordinate generation."""

import dataclasses
import logging
from typing import Final

from flax import typing as flax_typing
import jax
import jax.numpy as jnp

from miniqm import batch as batch_lib
from miniqm import geometry, model
from miniqm import jaxtypes as jt
from miniqm import noise as noise_lib

_LOGGER: Final = logging.getLogger(__name__)
_SPATIAL_DIMS: Final = 3


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplingConfig:
    """Reverse integration and stochastic churn parameters for Karras sampling."""

    noise: noise_lib.NoiseConfig = dataclasses.field(default_factory=noise_lib.NoiseConfig)
    num_steps: int = 40
    # Karras et al. (2022) / AlphaFold 3 stochastic churn defaults.
    gamma_0: float = 0.8
    gamma_min: float = 0.1
    noise_scale: float = 1.003
    step_scale: float = 1.5

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"Expected positive num_steps, got {self.num_steps}.")
        if self.gamma_0 < 0.0:
            raise ValueError(f"Expected non-negative gamma_0, got {self.gamma_0}.")
        if self.noise_scale < 0.0:
            raise ValueError(f"Expected non-negative noise_scale, got {self.noise_scale}.")
        if self.step_scale <= 0.0:
            raise ValueError(f"Expected positive step_scale, got {self.step_scale}.")


DEFAULT_SAMPLING_CONFIG: Final = SamplingConfig()


@jt.typed
def _euler_churn_step(
    key: jt.Array,
    positions: jt.Float[jt.Array, "batch num_atoms 3"],
    *,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    sigma_current: jt.Float[jt.Array, ""],
    sigma_next: jt.Float[jt.Array, ""],
    batch: batch_lib.JaxBatch,
    config: SamplingConfig,
) -> jt.Float[jt.Array, "batch num_atoms 3"]:
    """Applies one stochastic-churn corrector and Euler predictor step."""
    batch_size, _, _ = positions.shape
    gamma = jnp.where(sigma_current > config.gamma_min, config.gamma_0, 0.0)
    sigma_hat = sigma_current * (1.0 + gamma)
    churn_std = jnp.sqrt(jnp.maximum(jnp.square(sigma_hat) - jnp.square(sigma_current), 0.0)) * (
        config.noise_scale
    )

    churn_noise = jax.random.normal(key, shape=positions.shape, dtype=positions.dtype)
    noisy_positions = geometry.zero_center_of_mass(
        positions + churn_std * churn_noise,
        batch.atom_mask,
    )

    sigma_per_molecule = jnp.full((batch_size,), sigma_hat, dtype=positions.dtype)
    denoised_positions = jnp.asarray(
        denoiser.apply(
            params,
            noisy_positions,
            sigma_per_molecule,
            batch=batch,
        )
    )

    score_direction = (noisy_positions - denoised_positions) / sigma_hat
    updated_positions = (
        noisy_positions + config.step_scale * (sigma_next - sigma_hat) * score_direction
    )
    return geometry.zero_center_of_mass(updated_positions, batch.atom_mask)


@jt.typed
def sample_coordinates(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: SamplingConfig = DEFAULT_SAMPLING_CONFIG,
) -> jt.Float[jt.Array, "batch num_atoms 3"]:
    """Generates 3D molecular coordinates from Gaussian noise via Euler integration with churn.

    Args:
        key: JAX PRNG key.
        denoiser: Configured Denoiser Linen module.
        params: Initialized model parameters.
        batch: Padded molecular graph features and valid-atom mask.
        config: Reverse integration and stochastic churn parameters.

    Returns:
        Sampled zero-centered 3D coordinates with padded slots zeroed.
    """
    _LOGGER.info(
        "Sampling 3D coordinates over %d steps (gamma_0=%.2f, step_scale=%.2f).",
        config.num_steps,
        config.gamma_0,
        config.step_scale,
    )
    batch_size, num_atoms, _ = batch.atom_features.shape
    schedule = noise_lib.rollout_schedule(config.num_steps, config=config.noise)

    initial_key, loop_key = jax.random.split(key)
    initial_noise = jax.random.normal(
        initial_key,
        shape=(batch_size, num_atoms, _SPATIAL_DIMS),
        dtype=batch.atom_features.dtype,
    )
    initial_positions = geometry.zero_center_of_mass(schedule[0] * initial_noise, batch.atom_mask)
    step_keys = jax.random.split(loop_key, config.num_steps)

    @jt.typed
    def scan_body(
        positions: jt.Float[jt.Array, "batch num_atoms 3"],
        step_inputs: tuple[jt.Array, jt.Float[jt.Array, ""], jt.Float[jt.Array, ""]],
    ) -> tuple[jt.Float[jt.Array, "batch num_atoms 3"], None]:
        """Advances the coordinate trajectory by one reverse diffusion step."""
        step_key, sigma_current, sigma_next = step_inputs
        next_positions = _euler_churn_step(
            step_key,
            positions,
            denoiser=denoiser,
            params=params,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            batch=batch,
            config=config,
        )
        return next_positions, None

    final_positions, _ = jax.lax.scan(
        scan_body,
        initial_positions,
        (step_keys, schedule[:-1], schedule[1:]),
    )
    return final_positions
