"""Karras EDM noise schedule, preconditioning factors, and forward coordinate corruption."""

from collections.abc import Sequence
import dataclasses
from typing import Final

import jax
import jax.numpy as jnp

from miniqm import geometry
from miniqm import jaxtypes as jt


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoiseConfig:
    """Karras EDM noise distribution and rollout schedule parameters."""

    # Empirical coordinate std (1.20 A) across zero-centered QM9 train molecules.
    sigma_data: float = 1.2
    s_min: float = 4e-4
    s_max: float = 80.0
    rho: int = 7
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.5


DEFAULT_NOISE_CONFIG: Final = NoiseConfig()


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class PreconditioningFactors:
    """Karras EDM input/output scaling coefficients and loss weight for a given noise level."""

    c_in: jt.Float[jt.Array, "*batch"]
    c_skip: jt.Float[jt.Array, "*batch"]
    c_out: jt.Float[jt.Array, "*batch"]
    loss_weight: jt.Float[jt.Array, "*batch"]


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class NoisySample:
    """Randomly rotated target coordinates paired with Gaussian-corrupted noisy coordinates."""

    target_positions: jt.Float[jt.Array, "*batch num_atoms 3"]
    noisy_positions: jt.Float[jt.Array, "*batch num_atoms 3"]
    sigma: jt.Float[jt.Array, "*batch"]


@jt.typed
def preconditioning_factors(
    sigma: jt.Float[jt.Array, "*batch"],
    *,
    config: NoiseConfig = DEFAULT_NOISE_CONFIG,
) -> PreconditioningFactors:
    """Computes Karras EDM preconditioning coefficients and MSE loss weights.

    Args:
        sigma: Noise standard deviation per sample.
        config: Noise schedule and data scale configuration.

    Returns:
        Input scaling, skip connection, output scaling, and loss weight arrays.
    """
    sigma_data_sq = config.sigma_data**2
    total_var = jnp.square(sigma) + sigma_data_sq
    total_std = jnp.sqrt(total_var)
    return PreconditioningFactors(
        c_in=1.0 / total_std,
        c_skip=sigma_data_sq / total_var,
        c_out=(sigma * config.sigma_data) / total_std,
        loss_weight=total_var / jnp.square(sigma * config.sigma_data),
    )


@jt.typed
def rollout_schedule(
    num_steps: int,
    *,
    config: NoiseConfig = DEFAULT_NOISE_CONFIG,
) -> jt.Float[jt.Array, " num_levels"]:
    """Builds a decreasing Karras polynomial noise schedule over num_steps intervals.

    Args:
        num_steps: Number of reverse integration steps (produces num_steps + 1 levels).
        config: Noise schedule and data scale configuration.

    Returns:
        Monotonically decreasing noise levels from sigma_data * s_max to sigma_data * s_min.

    Raises:
        ValueError: If num_steps is not positive.
    """
    if num_steps <= 0:
        raise ValueError(f"Expected positive num_steps, got {num_steps}.")
    step_fraction = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    inv_rho = 1.0 / config.rho
    s_max_root = config.s_max**inv_rho
    s_min_root = config.s_min**inv_rho
    return config.sigma_data * jnp.power(
        s_max_root + step_fraction * (s_min_root - s_max_root),
        config.rho,
    )


@jt.typed
def sample_train_sigmas(
    key: jt.Array,
    shape: Sequence[int],
    *,
    config: NoiseConfig = DEFAULT_NOISE_CONFIG,
) -> jt.Float[jt.Array, "..."]:
    """Samples training noise standard deviations from a log-normal distribution.

    Args:
        key: JAX PRNG key.
        shape: Output batch dimensions.
        config: Noise distribution configuration.

    Returns:
        Sampled positive noise levels.
    """
    z_normal = jax.random.normal(key, shape=tuple(shape), dtype=jnp.float32)
    return config.sigma_data * jnp.exp(config.log_sigma_mean + config.log_sigma_std * z_normal)


@jt.typed
def make_noisy_sample(
    key: jt.Array,
    positions: jt.Float[jt.Array, "*batch num_atoms 3"],
    atom_mask: jt.Bool[jt.Array, "*batch num_atoms"],
    *,
    config: NoiseConfig = DEFAULT_NOISE_CONFIG,
) -> NoisySample:
    """Rotates ground-truth coordinates randomly and corrupts valid atoms with Gaussian noise.

    Args:
        key: JAX PRNG key.
        positions: Clean atomic coordinates.
        atom_mask: Boolean mask indicating valid atoms.
        config: Noise distribution configuration.

    Returns:
        Rotated target coordinates, noisy coordinates, and sampled noise levels.
    """
    rot_key, sigma_key, noise_key = jax.random.split(key, 3)
    *batch_dims, _, _ = positions.shape

    target_positions = geometry.random_rotation(rot_key, positions, atom_mask)
    sigma = sample_train_sigmas(sigma_key, batch_dims, config=config)
    gaussian_noise = jax.random.normal(noise_key, shape=positions.shape, dtype=positions.dtype)
    # (*batch, num_atoms, 1) expanded valid-atom mask for 3D coordinates.
    mask_3d = atom_mask[..., None]
    # (*batch, num_atoms, 3) broadcast per-molecule sigma (*batch, 1, 1) across atoms and xyz.
    noisy_positions = (target_positions + sigma[..., None, None] * gaussian_noise) * mask_3d
    return NoisySample(
        target_positions=target_positions,
        noisy_positions=noisy_positions,
        sigma=sigma,
    )
