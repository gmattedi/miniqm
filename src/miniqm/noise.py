"""Karras EDM noise schedule, preconditioning factors, and forward coordinate corruption."""

from collections.abc import Sequence
import dataclasses
from typing import Final

import jax
import jax.numpy as jnp

from miniqm import geometry
from miniqm import jaxtypes as jt

# Empirical coordinate std (1.20 A) across zero-centered QM9 train molecules.
SIGMA_DATA: Final[float] = 1.2

# Karras et al. (2022) / AlphaFold 3 noise schedule defaults.
DEFAULT_S_MIN: Final[float] = 4e-4
DEFAULT_S_MAX: Final[float] = 80.0
DEFAULT_RHO: Final[int] = 7
DEFAULT_LOG_SIGMA_MEAN: Final[float] = -1.2
DEFAULT_LOG_SIGMA_STD: Final[float] = 1.5


@jt.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class PreconditioningFactors:
    """Karras EDM input/output scaling coefficients and loss weight for a given noise level."""

    c_in: jt.Float[jt.Array, "*batch"]
    c_skip: jt.Float[jt.Array, "*batch"]
    c_out: jt.Float[jt.Array, "*batch"]
    loss_weight: jt.Float[jt.Array, "*batch"]


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
    sigma_data: float = SIGMA_DATA,
) -> PreconditioningFactors:
    """Computes Karras EDM preconditioning coefficients and MSE loss weights.

    Args:
        sigma: Noise standard deviation per sample.
        sigma_data: Empirical data coordinate standard deviation.

    Returns:
        Input scaling, skip connection, output scaling, and loss weight arrays.
    """
    sigma_data_sq = sigma_data**2
    total_var = jnp.square(sigma) + sigma_data_sq
    total_std = jnp.sqrt(total_var)
    return PreconditioningFactors(
        c_in=1.0 / total_std,
        c_skip=sigma_data_sq / total_var,
        c_out=(sigma * sigma_data) / total_std,
        loss_weight=total_var / jnp.square(sigma * sigma_data),
    )


@jt.typed
def rollout_schedule(
    num_steps: int,
    *,
    s_min: float = DEFAULT_S_MIN,
    s_max: float = DEFAULT_S_MAX,
    rho: int = DEFAULT_RHO,
    sigma_data: float = SIGMA_DATA,
) -> jt.Float[jt.Array, " num_levels"]:
    """Builds a decreasing Karras polynomial noise schedule over num_steps intervals.

    Args:
        num_steps: Number of reverse integration steps (produces num_steps + 1 levels).
        s_min: Minimum relative noise level at t = 1.
        s_max: Maximum relative noise level at t = 0.
        rho: Polynomial schedule curvature exponent.
        sigma_data: Empirical data coordinate standard deviation.

    Returns:
        Monotonically decreasing noise levels from sigma_data * s_max to sigma_data * s_min.

    Raises:
        ValueError: If num_steps is not positive.
    """
    if num_steps <= 0:
        raise ValueError(f"Expected positive num_steps, got {num_steps}.")
    step_fraction = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    inv_rho = 1.0 / rho
    s_max_root = s_max**inv_rho
    s_min_root = s_min**inv_rho
    return sigma_data * jnp.power(
        s_max_root + step_fraction * (s_min_root - s_max_root),
        rho,
    )


@jt.typed
def sample_train_sigmas(
    key: jt.Array,
    shape: Sequence[int],
    *,
    sigma_data: float = SIGMA_DATA,
    mean: float = DEFAULT_LOG_SIGMA_MEAN,
    std: float = DEFAULT_LOG_SIGMA_STD,
) -> jt.Float[jt.Array, "..."]:
    """Samples training noise standard deviations from a log-normal distribution.

    Args:
        key: JAX PRNG key.
        shape: Output batch dimensions.
        sigma_data: Empirical data coordinate standard deviation.
        mean: Mean of ln(sigma / sigma_data).
        std: Standard deviation of ln(sigma / sigma_data).

    Returns:
        Sampled positive noise levels.
    """
    z_normal = jax.random.normal(key, shape=tuple(shape), dtype=jnp.float32)
    return sigma_data * jnp.exp(mean + std * z_normal)


@jt.typed
def make_noisy_sample(
    key: jt.Array,
    positions: jt.Float[jt.Array, "*batch num_atoms 3"],
    atom_mask: jt.Bool[jt.Array, "*batch num_atoms"],
    *,
    sigma_data: float = SIGMA_DATA,
) -> NoisySample:
    """Rotates ground-truth coordinates randomly and corrupts valid atoms with Gaussian noise.

    Args:
        key: JAX PRNG key.
        positions: Clean atomic coordinates.
        atom_mask: Boolean mask indicating valid atoms.
        sigma_data: Empirical data coordinate standard deviation.

    Returns:
        Rotated target coordinates, noisy coordinates, and sampled noise levels.
    """
    rot_key, sigma_key, noise_key = jax.random.split(key, 3)
    *batch_dims, _, _ = positions.shape

    target_positions = geometry.random_rotation(rot_key, positions, atom_mask)
    sigma = sample_train_sigmas(
        sigma_key,
        batch_dims,
        sigma_data=sigma_data,
    )
    gaussian_noise = jax.random.normal(noise_key, shape=positions.shape, dtype=positions.dtype)
    mask_3d = atom_mask[..., None]
    noisy_positions = (target_positions + sigma[..., None, None] * gaussian_noise) * mask_3d
    return NoisySample(
        target_positions=target_positions,
        noisy_positions=noisy_positions,
        sigma=sigma,
    )
