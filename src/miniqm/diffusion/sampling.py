"""Reverse-time Karras Euler sampler with stochastic noise injection for 3D coordinates."""

import functools
import logging
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry, model
from miniqm.diffusion import noise as noise_lib

_LOGGER: Final = logging.getLogger(__name__)
_SPATIAL_DIMS: Final = 3


@jt.typed
def _euler_step(
    key: jt.Array,
    positions: jt.Float[jt.Array, "batch num_atoms 3"],
    *,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    sigma_current: jt.Float[jt.Array, ""],
    sigma_next: jt.Float[jt.Array, ""],
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig,
) -> jt.Float[jt.Array, "batch num_atoms 3"]:
    """Applies one stochastic noise expansion and reverse Euler integration step."""
    batch_size, _, _ = positions.shape
    is_stochastic = sigma_current > config.gamma_min
    gamma = jnp.where(is_stochastic, config.gamma_0, 0.0)
    effective_step_scale = jnp.where(is_stochastic, config.step_scale, 1.0)
    sigma_hat = sigma_current * (1.0 + gamma)
    injected_std = (
        jnp.sqrt(jnp.maximum(jnp.square(sigma_hat) - jnp.square(sigma_current), 0.0))
        * config.noise_scale
    )

    injected_noise = jax.random.normal(key, shape=positions.shape, dtype=positions.dtype)
    noisy_positions = geometry.zero_center_of_mass(
        positions + injected_std * injected_noise,
        batch.atom_mask,
    )

    # (batch,) per-molecule noise level broadcast from scalar sigma_hat.
    sigma_per_molecule = jnp.full((batch_size,), sigma_hat, dtype=positions.dtype)
    denoised_positions = jnp.asarray(
        denoiser.apply(
            params,
            noisy_positions,
            sigma_per_molecule,
            batch=batch,
        )
    )

    # (batch, num_atoms, 3) empirical score direction (x_hat - D(x_hat, sigma_hat)) / sigma_hat.
    score_direction = (noisy_positions - denoised_positions) / sigma_hat
    updated_positions = (
        noisy_positions + effective_step_scale * (sigma_next - sigma_hat) * score_direction
    )
    return geometry.zero_center_of_mass(updated_positions, batch.atom_mask)


@functools.partial(jax.jit, static_argnames=("denoiser", "config"))
@jt.typed
def _integrate_trajectory(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig,
) -> jt.Float[jt.Array, "num_frames batch num_atoms 3"]:
    """Integrates the reverse-diffusion coordinate trajectory in a compiled scan loop."""
    batch_size, num_atoms, _ = batch.atom_features.shape
    schedule = noise_lib.rollout_schedule(config.num_steps, config=denoiser.config.noise)

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
    ) -> tuple[jt.Float[jt.Array, "batch num_atoms 3"], jt.Float[jt.Array, "batch num_atoms 3"]]:
        """Advances the coordinate trajectory by one reverse diffusion step."""
        step_key, sigma_current, sigma_next = step_inputs
        next_positions = _euler_step(
            step_key,
            positions,
            denoiser=denoiser,
            params=params,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            batch=batch,
            config=config,
        )
        return next_positions, next_positions

    _, step_frames = jax.lax.scan(
        scan_body,
        initial_positions,
        (step_keys, schedule[:-1], schedule[1:]),
    )
    # (num_steps + 1, batch, num_atoms, 3) full trajectory with initial noise prepended.
    return jnp.concatenate((initial_positions[None, ...], step_frames), axis=0)


@jt.typed
def sample_trajectory(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> jt.Float[jt.Array, "num_frames batch num_atoms 3"]:
    """Generates the full reverse-diffusion coordinate trajectory from Gaussian noise.

    Args:
        key: JAX PRNG key.
        denoiser: Configured denoiser module.
        params: Initialized model parameters.
        batch: Padded molecular graph features and valid-atom mask.
        config: Reverse integration parameters.

    Returns:
        Coordinate trajectory from initial noise (index 0) to final denoised coordinates (index -1).
    """
    _LOGGER.info(
        "Sampling 3D coordinate trajectory over %d steps (gamma_0=%.2f, step_scale=%.2f).",
        config.num_steps,
        config.gamma_0,
        config.step_scale,
    )
    return cast(
        "jt.Float[jt.Array, 'num_frames batch num_atoms 3']",
        _integrate_trajectory(
            key,
            denoiser,
            params,
            batch=batch,
            config=config,
        ),
    )


@jt.typed
def sample_coordinates(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> jt.Float[jt.Array, "batch num_atoms 3"]:
    """Generates 3D molecular coordinates from Gaussian noise via reverse Euler integration.

    Args:
        key: JAX PRNG key.
        denoiser: Configured denoiser module.
        params: Initialized model parameters.
        batch: Padded molecular graph features and valid-atom mask.
        config: Reverse integration parameters.

    Returns:
        Sampled zero-centered 3D coordinates with padded slots zeroed.
    """
    return sample_trajectory(
        key,
        denoiser,
        params,
        batch=batch,
        config=config,
    )[-1]
