"""Karras EDM noise schedule, preconditioning factors, and forward 2D+3D corruption."""

from collections.abc import Iterable
import dataclasses
from typing import Final

import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry

_SYM_NOISE_SCALE: Final = float(1.0 / jnp.sqrt(2.0))
_BONDED_THRESHOLD: Final = 0.5
_MIN_PARTIAL_KEEP_PROB: Final = 0.2
_MAX_PARTIAL_KEEP_PROB: Final = 0.8
_TILDE_BOND_PROB: Final = 0.5
_INTRINSIC_ATOM_CHANNEL_MASK: Final = (
    True,
    True,
    True,
    True,  # element: C, N, O, F (0..3)
    False,
    False,
    False,
    False,
    False,  # total_deg (4..8)
    False,
    False,
    False,
    False,
    False,  # total_hs (9..13)
    True,
    True,
    True,  # formal charge: -1, 0, +1 (14..16)
    False,
    False,
    False,
    False,  # hybridization (17..20)
    False,
    False,  # is_aromatic, is_in_ring (21..22)
    False,
    False,
    False,  # chirality: none, R, S (23..25)
)


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PreconditioningFactors:
    """Karras EDM input/output scaling coefficients and loss weight for a given noise level."""

    c_in: jt.Float[jt.Array, "*batch"]
    c_skip: jt.Float[jt.Array, "*batch"]
    c_out: jt.Float[jt.Array, "*batch"]
    loss_weight: jt.Float[jt.Array, "*batch"]


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NoisySample:
    """Corrupted 3D coordinates and 2D atom/bond features paired with clean targets and masks."""

    target_positions: jt.Float[jt.Array, "*batch num_atoms 3"]
    noisy_positions: jt.Float[jt.Array, "*batch num_atoms 3"]
    target_atom_features: jt.Float[jt.Array, "*batch num_atoms num_atom_features"]
    noisy_atom_features: jt.Float[jt.Array, "*batch num_atoms num_atom_features"]
    target_bond_features: jt.Float[jt.Array, "*batch num_atoms num_atoms num_bond_features"]
    noisy_bond_features: jt.Float[jt.Array, "*batch num_atoms num_atoms num_bond_features"]
    atom_cond_mask: jt.Bool[jt.Array, "*batch num_atoms num_atom_features"]
    bond_cond_mask: jt.Bool[jt.Array, "*batch num_atoms num_atoms num_bond_features"]
    sigma: jt.Float[jt.Array, "*batch"]


@jt.typed
def preconditioning_factors(
    sigma: jt.Float[jt.Array, "*batch"],
    *,
    config: config_lib.NoiseConfig = config_lib.DEFAULT_NOISE_CONFIG,
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
    config: config_lib.NoiseConfig = config_lib.DEFAULT_NOISE_CONFIG,
) -> jt.Float[jt.Array, " num_levels"]:
    """Builds a decreasing Karras polynomial noise schedule.

    Args:
        num_steps: Number of reverse integration steps.
        config: Noise schedule and data scale configuration.

    Returns:
        Monotonically decreasing noise levels from maximum to minimum sigma.

    Raises:
        ValueError: If the step count is not positive.
    """
    if num_steps <= 0:
        raise ValueError(f"Expected positive num_steps, got {num_steps}.")
    step_fraction = jnp.linspace(0.0, 1.0, num_steps + 1)
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
    shape: Iterable[int],
    *,
    config: config_lib.NoiseConfig = config_lib.DEFAULT_NOISE_CONFIG,
) -> jt.Float[jt.Array, "..."]:
    """Samples training noise standard deviations from a log-normal distribution.

    Args:
        key: JAX PRNG key.
        shape: Output batch dimensions.
        config: Noise distribution configuration.

    Returns:
        Sampled positive noise levels.
    """
    z_normal = jax.random.normal(key, shape=tuple(shape))
    return config.sigma_data * jnp.exp(config.log_sigma_mean + config.log_sigma_std * z_normal)


@jt.typed
def off_diagonal_pair_mask(
    atom_mask: jt.Bool[jt.Array, "*batch num_atoms"],
) -> jt.Bool[jt.Array, "*batch num_atoms num_atoms 1"]:
    """Builds a broadcastable mask over distinct valid atom pairs."""
    *_, num_atoms = atom_mask.shape
    eye = jnp.eye(num_atoms, dtype=jnp.bool_)
    return (atom_mask[..., :, None] & atom_mask[..., None, :] & ~eye)[..., None]


@jt.typed
def symmetric_bond_noise(
    key: jt.Array,
    shape: Iterable[int],
) -> jt.Float[jt.Array, "*batch num_atoms num_atoms num_bond_features"]:
    """Samples symmetric unit-variance Gaussian noise over undirected atom pairs."""
    raw_noise = jax.random.normal(key, shape=tuple(shape))
    return _SYM_NOISE_SCALE * (raw_noise + jnp.swapaxes(raw_noise, -3, -2))


@jt.typed
def _sample_training_cond_masks(
    key: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    config: config_lib.NoiseConfig,
) -> tuple[
    jt.Bool[jt.Array, "*batch num_atoms num_atom_features"],
    jt.Bool[jt.Array, "*batch num_atoms num_atoms num_bond_features"],
]:
    """Samples unconditional, partial-substructure, or full 2D conditioning masks per molecule."""
    regime_key, prob_key, atom_key, tilde_key = jax.random.split(key, 4)
    *batch_dims, num_atoms = batch.atom_mask.shape

    regime_u = jax.random.uniform(regime_key, shape=batch_dims)
    is_uncond = regime_u < config.uncond_prob
    is_partial = (regime_u >= config.uncond_prob) & (
        regime_u < config.uncond_prob + config.partial_cond_prob
    )

    keep_prob = jax.random.uniform(
        prob_key,
        shape=(*batch_dims, 1),
        minval=_MIN_PARTIAL_KEEP_PROB,
        maxval=_MAX_PARTIAL_KEEP_PROB,
    )
    keep_atom = (
        jax.random.bernoulli(atom_key, p=keep_prob, shape=batch.atom_mask.shape) & batch.atom_mask
    )

    pair_valid = off_diagonal_pair_mask(batch.atom_mask)[..., 0]
    is_bonded = (batch.bond_features[..., 0] < _BONDED_THRESHOLD) & pair_valid
    # (*batch, num_atoms) True if atom is covalently bonded to any masked neighbor.
    touches_masked_atom = jnp.any(is_bonded & (~keep_atom[..., None, :]), axis=-1)

    intrinsic_mask = jnp.array(_INTRINSIC_ATOM_CHANNEL_MASK, dtype=jnp.bool_)
    # (*batch, num_atoms, num_atom_features) partial atom conditioning mask.
    partial_atom_cond = keep_atom[..., None] & (~touches_masked_atom[..., None] | intrinsic_mask)

    both_kept = keep_atom[..., :, None] & keep_atom[..., None, :] & pair_valid
    tilde_draw = jax.random.bernoulli(tilde_key, p=_TILDE_BOND_PROB, shape=pair_valid.shape)
    upper_tri = jnp.triu(jnp.ones((num_atoms, num_atoms), dtype=jnp.bool_), k=1)
    upper_draw = tilde_draw & upper_tri
    sym_tilde_draw = upper_draw | jnp.swapaxes(upper_draw, -2, -1)
    tilde_bond = is_bonded & ~both_kept & sym_tilde_draw

    *_, num_bond_features = batch.bond_features.shape
    bond_channel_0 = jnp.arange(num_bond_features) == 0
    # (*batch, num_atoms, num_atoms, num_bond_features) partial bond conditioning mask.
    partial_bond_cond = both_kept[..., None] | (tilde_bond[..., None] & bond_channel_0)

    full_atom_cond = batch.atom_cond_mask & batch.atom_mask[..., None]
    full_bond_cond = batch.bond_cond_mask & pair_valid[..., None]

    # (*batch, num_atoms, num_atom_features) and (*batch, num_atoms, num_atoms, num_bond_features).
    atom_cond_mask = ~is_uncond[..., None, None] & jnp.where(
        is_partial[..., None, None],
        partial_atom_cond,
        full_atom_cond,
    )
    bond_cond_mask = ~is_uncond[..., None, None, None] & jnp.where(
        is_partial[..., None, None, None],
        partial_bond_cond,
        full_bond_cond,
    )
    return atom_cond_mask, bond_cond_mask


@jt.typed
def make_noisy_sample(
    key: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    config: config_lib.NoiseConfig = config_lib.DEFAULT_NOISE_CONFIG,
    sample_cond_masks: bool = False,
) -> NoisySample:
    """Rotates 3D coordinates and corrupts coordinates, atom types, and bonds with Gaussian noise.

    Args:
        key: JAX PRNG key.
        batch: Clean padded molecular batch.
        config: Noise distribution and conditioning mask configuration.
        sample_cond_masks: Whether to randomly mask 2D conditioning into unconditional,
            partial-template, and full-graph regimes.

    Returns:
        Corrupted 3D coordinates and 2D features paired with clean targets and conditioning masks.
    """
    rot_key, sigma_key, pos_noise_key, atom_noise_key, bond_noise_key, cond_key = jax.random.split(
        key, 6
    )
    *batch_dims, _, _ = batch.positions.shape

    target_positions = geometry.random_rotation(rot_key, batch.positions, batch.atom_mask)
    sigma = sample_train_sigmas(sigma_key, batch_dims, config=config)

    # (*batch, num_atoms, 1) expanded valid-atom mask.
    mask_3d = batch.atom_mask[..., None]
    pair_mask = off_diagonal_pair_mask(batch.atom_mask)

    pos_noise = jax.random.normal(pos_noise_key, shape=batch.positions.shape)
    atom_noise = jax.random.normal(atom_noise_key, shape=batch.atom_features.shape)
    bond_noise = symmetric_bond_noise(bond_noise_key, batch.bond_features.shape)

    target_atom_features = batch.atom_features * mask_3d
    target_bond_features = batch.bond_features * pair_mask

    noisy_positions = (target_positions + sigma[..., None, None] * pos_noise) * mask_3d
    noisy_atom_features = (target_atom_features + sigma[..., None, None] * atom_noise) * mask_3d
    noisy_bond_features = (
        target_bond_features + sigma[..., None, None, None] * bond_noise
    ) * pair_mask

    if sample_cond_masks:
        atom_cond_mask, bond_cond_mask = _sample_training_cond_masks(
            cond_key,
            batch,
            config=config,
        )
    else:
        atom_cond_mask = batch.atom_cond_mask & mask_3d
        bond_cond_mask = batch.bond_cond_mask & pair_mask

    return NoisySample(
        target_positions=target_positions,
        noisy_positions=noisy_positions,
        target_atom_features=target_atom_features,
        noisy_atom_features=noisy_atom_features,
        target_bond_features=target_bond_features,
        noisy_bond_features=noisy_bond_features,
        atom_cond_mask=atom_cond_mask,
        bond_cond_mask=bond_cond_mask,
        sigma=sigma,
    )
