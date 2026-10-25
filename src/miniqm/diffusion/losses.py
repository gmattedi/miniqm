"""Kabsch-aligned EDM coordinate MSE, local pairwise distance loss, and lDDT scoring."""

from collections.abc import Iterable
import dataclasses
from typing import Final

import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.diffusion import geometry
from miniqm.diffusion import noise as noise_lib

# Prevents infinite gradient of sqrt(r^2) at zero interatomic distance or zero RMSD.
_DIST_EPS: Final = 1e-8
_LDDT_THRESHOLDS: Final = (0.1, 0.25, 0.5, 1.0)


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
    """Combined diffusion training loss and diagnostic components."""

    total_loss: jt.Float[jt.Array, ""]
    aligned_mse_loss: jt.Float[jt.Array, ""]
    distance_loss: jt.Float[jt.Array, ""]
    rmsd: jt.Float[jt.Array, ""]
    lddt: jt.Float[jt.Array, ""]


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PairwiseDistances:
    """Predicted and target interatomic distances paired with a local valid-pair mask."""

    pred_dists: jt.Float[jt.Array, "batch num_atoms num_atoms"]
    target_dists: jt.Float[jt.Array, "batch num_atoms num_atoms"]
    pair_mask: jt.Float[jt.Array, "batch num_atoms num_atoms"]


@jt.typed
def _pairwise_distances_and_mask(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
    cutoff: float,
) -> _PairwiseDistances:
    """Computes pairwise Euclidean distances and the mask of distinct valid pairs within cutoff."""
    _, num_atoms, _ = predicted_positions.shape
    # (batch, num_atoms, num_atoms, 3) pairwise 3D coordinate differences x_i - x_j.
    pred_offsets = predicted_positions[:, :, None, :] - predicted_positions[:, None, :, :]
    target_offsets = target_positions[:, :, None, :] - target_positions[:, None, :, :]

    # (batch, num_atoms, num_atoms) pairwise Euclidean distance matrices.
    pred_dists = jnp.sqrt(jnp.sum(jnp.square(pred_offsets), axis=-1) + _DIST_EPS)
    target_dists = jnp.sqrt(jnp.sum(jnp.square(target_offsets), axis=-1) + _DIST_EPS)

    # (batch, num_atoms, num_atoms) mask for distinct valid atom pairs (i != j) within cutoff.
    valid_pairs = atom_mask[:, :, None] & atom_mask[:, None, :]
    not_self = ~(jnp.eye(num_atoms, dtype=jnp.bool_)[None, :, :])
    within_cutoff = target_dists < cutoff
    pair_mask = (valid_pairs & not_self & within_cutoff).astype(predicted_positions.dtype)
    return _PairwiseDistances(
        pred_dists=pred_dists,
        target_dists=target_dists,
        pair_mask=pair_mask,
    )


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
    pairwise = _pairwise_distances_and_mask(
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
def lddt_per_molecule(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
    cutoff: float = config_lib.DEFAULT_LOSS_CONFIG.distance_cutoff,
    thresholds: Iterable[float] = _LDDT_THRESHOLDS,
) -> jt.Float[jt.Array, " batch"]:
    """Computes per-molecule Local Distance Difference Test (lDDT) scores in [0, 1].

    Args:
        predicted_positions: Predicted 3D atomic coordinates.
        target_positions: Ground-truth 3D atomic coordinates.
        atom_mask: Valid-atom mask.
        cutoff: Inclusion radius in Angstroms on ground-truth interatomic distances.
        thresholds: Distance error tolerances in Angstroms averaged into the score.

    Returns:
        Per-molecule lDDT scores in [0, 1].
    """
    pairwise = _pairwise_distances_and_mask(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cutoff=cutoff,
    )
    dist_err = jnp.abs(pairwise.pred_dists - pairwise.target_dists)
    threshold_values = jnp.asarray(tuple(thresholds), dtype=predicted_positions.dtype)
    # (batch, num_atoms, num_atoms, num_thresholds) boolean satisfaction across tolerances.
    within_thresholds = (dist_err[..., None] < threshold_values).astype(predicted_positions.dtype)
    # (batch, num_atoms, num_atoms) fraction of satisfied distance thresholds per pair.
    pair_scores = jnp.mean(within_thresholds, axis=-1)
    num_pairs = jnp.sum(pairwise.pair_mask, axis=(-1, -2))
    return jnp.where(
        num_pairs > 0.0,
        jnp.sum(pair_scores * pairwise.pair_mask, axis=(-1, -2)) / jnp.maximum(num_pairs, 1.0),
        1.0,
    )


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
    """Computes the combined Kabsch-aligned EDM MSE, local pairwise distance loss, and lDDT.

    Args:
        predicted_positions: Denoised atomic coordinates predicted by the model.
        target_positions: Ground-truth atomic coordinates.
        sigma: Noise standard deviation per molecule.
        atom_mask: Valid-atom mask.
        config: Loss weighting and cutoff configuration.
        noise_config: Karras EDM noise preconditioning configuration.

    Returns:
        Combined scalar loss alongside individual MSE, distance, RMSD, and lDDT metrics.
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
    lddt = jnp.mean(
        lddt_per_molecule(
            predicted_positions=predicted_positions,
            target_positions=target_positions,
            atom_mask=atom_mask,
            cutoff=config.distance_cutoff,
        )
    )
    total_loss = mse_stats.weighted_mse + config.distance_loss_weight * dist_loss
    return LossOutput(
        total_loss=total_loss,
        aligned_mse_loss=mse_stats.weighted_mse,
        distance_loss=dist_loss,
        rmsd=mse_stats.rmsd,
        lddt=lddt,
    )
