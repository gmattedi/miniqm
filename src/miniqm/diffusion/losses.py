"""Differentiable Kabsch-aligned EDM coordinate MSE and local pairwise distance losses."""

import dataclasses
from typing import Final

import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.diffusion import geometry
from miniqm.diffusion import noise as noise_lib

# Prevents infinite gradient of sqrt(r^2) at zero RMSD.
_DIST_EPS: Final = 1e-8


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AlignedMseStats:
    """EDM-weighted Kabsch-aligned MSE loss and unweighted RMSD across a batch."""

    weighted_mse: jt.Float[jt.Array, ""]
    rmsd: jt.Float[jt.Array, ""]


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LossOutput:
    """Combined differentiable diffusion training loss and its components."""

    total_loss: jt.Float[jt.Array, ""]
    aligned_mse_loss: jt.Float[jt.Array, ""]
    distance_loss: jt.Float[jt.Array, ""]
    rmsd: jt.Float[jt.Array, ""]


@jt.typed
def aligned_mse_loss(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    sigma: jt.Float[jt.Array, " batch"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
    config: config_lib.NoiseConfig = config_lib.DEFAULT_NOISE_CONFIG,
) -> AlignedMseStats:
    """Computes Kabsch-aligned EDM-weighted coordinate MSE and unweighted RMSD.

    Args:
        predicted_positions: Denoised atomic coordinates predicted by the model.
        target_positions: Ground-truth atomic coordinates.
        sigma: Noise standard deviation per molecule.
        atom_mask: Valid-atom mask.
        config: Noise schedule and data scale configuration.

    Returns:
        Batch-averaged EDM-weighted MSE and unweighted Kabsch RMSD in Angstroms.
    """
    aligned_target = geometry.kabsch_align(
        reference=predicted_positions,
        target=target_positions,
        atom_mask=atom_mask,
    )
    mask_f32 = atom_mask.astype(predicted_positions.dtype)
    atom_counts = jnp.maximum(jnp.sum(mask_f32, axis=-1), 1.0)

    # (batch, num_atoms) squared Euclidean error per atom, zeroed on padded slots.
    sq_dist_per_atom = jnp.sum(jnp.square(predicted_positions - aligned_target), axis=-1) * mask_f32
    # (batch,) mean squared coordinate error and RMSD per molecule.
    mse_per_mol = jnp.sum(sq_dist_per_atom, axis=-1) / atom_counts
    rmsd_per_mol = jnp.sqrt(mse_per_mol + _DIST_EPS)

    factors = noise_lib.preconditioning_factors(sigma, config=config)
    weighted_mse = jnp.mean(factors.loss_weight * mse_per_mol)
    return AlignedMseStats(
        weighted_mse=weighted_mse,
        rmsd=jnp.mean(rmsd_per_mol),
    )


@jt.typed
def pairwise_distance_loss(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
    cutoff: float = config_lib.DEFAULT_LOSS_CONFIG.distance_cutoff,
) -> jt.Float[jt.Array, ""]:
    """Computes mean squared error on interatomic distances within a target distance cutoff.

    Args:
        predicted_positions: Denoised atomic coordinates predicted by the model.
        target_positions: Ground-truth atomic coordinates.
        atom_mask: Valid-atom mask.
        cutoff: Maximum ground-truth interatomic distance in Angstroms to include.

    Returns:
        Batch-averaged per-molecule mean squared distance error over local atom pairs.
    """
    pairwise = geometry.pairwise_distances_and_mask(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cutoff=cutoff,
    )
    sq_err = jnp.square(pairwise.pred_dists - pairwise.target_dists) * pairwise.pair_mask
    # (batch,) mean squared distance error per molecule across active local pairs.
    num_pairs = jnp.maximum(jnp.sum(pairwise.pair_mask, axis=(-1, -2)), 1.0)
    mse_per_mol = jnp.sum(sq_err, axis=(-1, -2)) / num_pairs
    return jnp.mean(mse_per_mol)


@jt.typed
def compute_diffusion_loss(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    sigma: jt.Float[jt.Array, " batch"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
    config: config_lib.LossConfig = config_lib.DEFAULT_LOSS_CONFIG,
    noise_config: config_lib.NoiseConfig = config_lib.DEFAULT_NOISE_CONFIG,
) -> LossOutput:
    """Computes the combined differentiable Kabsch-aligned EDM MSE and pairwise distance loss.

    Args:
        predicted_positions: Denoised atomic coordinates predicted by the model.
        target_positions: Ground-truth atomic coordinates.
        sigma: Noise standard deviation per molecule.
        atom_mask: Valid-atom mask.
        config: Loss weighting and cutoff configuration.
        noise_config: Karras EDM noise preconditioning configuration.

    Returns:
        Combined scalar loss alongside individual MSE, distance, and RMSD terms.
    """
    mse_stats = aligned_mse_loss(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        sigma=sigma,
        atom_mask=atom_mask,
        config=noise_config,
    )
    dist_loss = pairwise_distance_loss(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cutoff=config.distance_cutoff,
    )
    total_loss = mse_stats.weighted_mse + config.distance_loss_weight * dist_loss
    return LossOutput(
        total_loss=total_loss,
        aligned_mse_loss=mse_stats.weighted_mse,
        distance_loss=dist_loss,
        rmsd=mse_stats.rmsd,
    )
