"""Reverse-time Karras Euler sampler with stochastic noise injection for 2D+3D molecules."""

import dataclasses
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


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SampledMolecules:
    """Jointly sampled 3D atomic coordinates and continuous 2D atom/bond features."""

    positions: jt.Float[jt.Array, "*batch num_atoms 3"]
    atom_features: jt.Float[jt.Array, "*batch num_atoms num_atom_features"]
    bond_features: jt.Float[jt.Array, "*batch num_atoms num_atoms num_bond_features"]


@jt.typed
def _euler_step(
    key: jt.Array,
    state: SampledMolecules,
    *,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    sigma_current: jt.Float[jt.Array, ""],
    sigma_next: jt.Float[jt.Array, ""],
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig,
) -> SampledMolecules:
    """Applies one stochastic noise expansion and reverse Euler step to 2D+3D state."""
    pos_key, atom_key, bond_key = jax.random.split(key, 3)
    batch_size, _, _ = state.positions.shape
    mask_3d = batch.atom_mask[..., None]
    pair_mask = noise_lib.off_diagonal_pair_mask(batch.atom_mask)

    is_stochastic = sigma_current > config.gamma_min
    gamma = jnp.where(is_stochastic, config.gamma_0, 0.0)
    effective_step_scale = jnp.where(is_stochastic, config.step_scale, 1.0)
    sigma_hat = sigma_current * (1.0 + gamma)
    injected_std = (
        jnp.sqrt(jnp.maximum(jnp.square(sigma_hat) - jnp.square(sigma_current), 0.0))
        * config.noise_scale
    )

    pos_noise = jax.random.normal(pos_key, shape=state.positions.shape)
    atom_noise = jax.random.normal(atom_key, shape=state.atom_features.shape)
    bond_noise = noise_lib.symmetric_bond_noise(bond_key, state.bond_features.shape)

    noisy_positions = geometry.zero_center_of_mass(
        state.positions + injected_std * pos_noise,
        batch.atom_mask,
    )
    noisy_atoms = (state.atom_features + injected_std * atom_noise) * mask_3d
    noisy_bonds = (state.bond_features + injected_std * bond_noise) * pair_mask

    # (batch,) per-molecule noise level broadcast from scalar sigma_hat.
    sigma_per_molecule = jnp.full((batch_size,), sigma_hat)
    denoised = cast(
        "model.DenoiserOutput",
        denoiser.apply(
            params,
            noisy_positions,
            noisy_atoms,
            noisy_bonds,
            sigma_per_molecule,
            batch=batch,
        ),
    )

    step_factor = effective_step_scale * (sigma_next - sigma_hat) / sigma_hat
    updated_positions = geometry.zero_center_of_mass(
        noisy_positions + step_factor * (noisy_positions - denoised.positions),
        batch.atom_mask,
    )
    updated_atoms = (noisy_atoms + step_factor * (noisy_atoms - denoised.atom_features)) * mask_3d
    updated_bonds = (noisy_bonds + step_factor * (noisy_bonds - denoised.bond_features)) * pair_mask
    return SampledMolecules(
        positions=updated_positions,
        atom_features=updated_atoms,
        bond_features=updated_bonds,
    )


@functools.partial(jax.jit, static_argnames=("denoiser", "config"))
@jt.typed
def _integrate_trajectory(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig,
) -> SampledMolecules:
    """Integrates the reverse-diffusion 2D+3D trajectory in a compiled scan loop."""
    batch_size, num_atoms, _ = batch.atom_features.shape
    schedule = noise_lib.rollout_schedule(config.num_steps, config=denoiser.config.noise)

    pos_init_key, atom_init_key, bond_init_key, loop_key = jax.random.split(key, 4)
    mask_3d = batch.atom_mask[..., None]
    pair_mask = noise_lib.off_diagonal_pair_mask(batch.atom_mask)

    initial_positions = geometry.zero_center_of_mass(
        schedule[0]
        * jax.random.normal(
            pos_init_key,
            shape=(batch_size, num_atoms, _SPATIAL_DIMS),
        ),
        batch.atom_mask,
    )
    initial_atoms = (
        schedule[0]
        * jax.random.normal(
            atom_init_key,
            shape=batch.atom_features.shape,
        )
        * mask_3d
    )
    initial_bonds = (
        schedule[0]
        * noise_lib.symmetric_bond_noise(
            bond_init_key,
            batch.bond_features.shape,
        )
        * pair_mask
    )
    initial_state = SampledMolecules(
        positions=initial_positions,
        atom_features=initial_atoms,
        bond_features=initial_bonds,
    )
    step_keys = jax.random.split(loop_key, config.num_steps)

    @jt.typed
    def scan_body(
        state: SampledMolecules,
        step_inputs: tuple[jt.Array, jt.Float[jt.Array, ""], jt.Float[jt.Array, ""]],
    ) -> tuple[SampledMolecules, SampledMolecules]:
        """Advances the 2D+3D trajectory by one reverse diffusion step."""
        step_key, sigma_current, sigma_next = step_inputs
        next_state = _euler_step(
            step_key,
            state,
            denoiser=denoiser,
            params=params,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            batch=batch,
            config=config,
        )
        return next_state, next_state

    _, step_frames = jax.lax.scan(
        scan_body,
        initial_state,
        (step_keys, schedule[:-1], schedule[1:]),
    )
    return SampledMolecules(
        positions=jnp.concatenate(
            (initial_state.positions[None, ...], step_frames.positions), axis=0
        ),
        atom_features=jnp.concatenate(
            (initial_state.atom_features[None, ...], step_frames.atom_features),
            axis=0,
        ),
        bond_features=jnp.concatenate(
            (initial_state.bond_features[None, ...], step_frames.bond_features),
            axis=0,
        ),
    )


@jt.typed
def sample_molecule_trajectory(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> SampledMolecules:
    """Generates the full reverse-diffusion 2D+3D trajectory from Gaussian noise."""
    _LOGGER.info(
        "Sampling 2D+3D molecular trajectory over %d steps (gamma_0=%.2f, step_scale=%.2f).",
        config.num_steps,
        config.gamma_0,
        config.step_scale,
    )
    return cast(
        "SampledMolecules",
        _integrate_trajectory(
            key,
            denoiser,
            params,
            batch=batch,
            config=config,
        ),
    )


@jt.typed
def sample_trajectory(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> jt.Float[jt.Array, "num_frames batch num_atoms 3"]:
    """Generates the full reverse-diffusion 3D coordinate trajectory from Gaussian noise.

    Args:
        key: JAX PRNG key.
        denoiser: Configured denoiser module.
        params: Initialized model parameters.
        batch: Padded molecular graph conditioning features and masks.
        config: Reverse integration parameters.

    Returns:
        Coordinate trajectory from initial noise (index 0) to final denoised coordinates (index -1).
    """
    return sample_molecule_trajectory(
        key,
        denoiser,
        params,
        batch=batch,
        config=config,
    ).positions


@jt.typed
def sample_molecules(
    key: jt.Array,
    denoiser: model.Denoiser,
    params: flax_typing.VariableDict,
    *,
    batch: batch_lib.JaxBatch,
    config: config_lib.SamplingConfig = config_lib.DEFAULT_SAMPLING_CONFIG,
) -> SampledMolecules:
    """Generates 3D coordinates and 2D atom/bond features from Gaussian noise.

    Args:
        key: JAX PRNG key.
        denoiser: Configured denoiser module.
        params: Initialized model parameters.
        batch: Padded molecular graph conditioning features and masks.
        config: Reverse integration parameters.

    Returns:
        Final sampled 3D coordinates and continuous 2D atom/bond features.
    """
    trajectory = sample_molecule_trajectory(
        key,
        denoiser,
        params,
        batch=batch,
        config=config,
    )
    return SampledMolecules(
        positions=trajectory.positions[-1],
        atom_features=trajectory.atom_features[-1],
        bond_features=trajectory.bond_features[-1],
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
        batch: Padded molecular graph conditioning features and masks.
        config: Reverse integration parameters.

    Returns:
        Sampled zero-centered 3D coordinates with padded slots zeroed.
    """
    return sample_molecules(
        key,
        denoiser,
        params,
        batch=batch,
        config=config,
    ).positions
