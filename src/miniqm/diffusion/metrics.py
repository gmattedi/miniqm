"""Conformer evaluation metrics: Kabsch RMSD, local lDDT, and stereocenter violations."""

from collections.abc import Iterable
import dataclasses
from typing import Final

import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.diffusion import geometry

_DIST_EPS: Final = 1e-8
_LDDT_THRESHOLDS: Final = (0.1, 0.25, 0.5, 1.0)
_ONE_HOT_THRESHOLD: Final = 0.5
_MIN_STEREOCENTER_DEGREE: Final = 3
_MIN_TETRAHEDRAL_ATOMS: Final = 4


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StereocenterStats:
    """Per-molecule stereocenter violation counts and batch-wide violation rate."""

    num_violations: jt.Int[jt.Array, " batch"]
    num_stereocenters: jt.Int[jt.Array, " batch"]
    violation_rate: jt.Float[jt.Array, ""]


@jt.typed
def rmsd_per_molecule(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
) -> jt.Float[jt.Array, " batch"]:
    """Computes per-molecule Kabsch-aligned coordinate RMSD in Angstroms.

    Args:
        predicted_positions: Predicted 3D atomic coordinates.
        target_positions: Ground-truth 3D atomic coordinates.
        atom_mask: Valid-atom mask.

    Returns:
        Per-molecule Kabsch-aligned coordinate RMSD in Angstroms.
    """
    aligned = geometry.kabsch_align(
        reference=target_positions,
        target=predicted_positions,
        atom_mask=atom_mask,
    )
    atom_counts = jnp.maximum(jnp.sum(atom_mask, axis=-1), 1)
    # (batch, num_atoms) squared coordinate error per atom, zeroed on padded slots.
    sq_err_per_atom = jnp.sum(jnp.square(aligned - target_positions), axis=-1) * atom_mask
    return jnp.sqrt(jnp.sum(sq_err_per_atom, axis=-1) / atom_counts + _DIST_EPS)


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
    pairwise = geometry.pairwise_distances_and_mask(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cutoff=cutoff,
    )
    dist_err = jnp.abs(pairwise.pred_dists - pairwise.target_dists)
    threshold_values = jnp.asarray(tuple(thresholds))
    # (batch, num_atoms, num_atoms) fraction of satisfied distance thresholds per pair.
    pair_scores = jnp.mean(dist_err[..., None] < threshold_values, axis=-1)
    num_pairs = jnp.sum(pairwise.pair_mask, axis=(-1, -2))
    return jnp.where(
        num_pairs > 0,
        jnp.sum(pair_scores * pairwise.pair_mask, axis=(-1, -2)) / jnp.maximum(num_pairs, 1),
        1.0,
    )


@jt.typed
def _tetrahedral_volumes(
    positions: jt.Float[jt.Array, "batch num_atoms 3"],
    neighbor_indices: jt.Int[jt.Array, "batch num_atoms 4"],
    has_four_neighbors: jt.Bool[jt.Array, "batch num_atoms"],
) -> jt.Float[jt.Array, "batch num_atoms"]:
    """Computes signed tetrahedral volumes from each atom's bonded neighbor coordinates."""
    batch_size, _, _ = positions.shape
    batch_indices = jnp.arange(batch_size)[:, None, None]
    # (batch, num_atoms, 4, 3) 3D coordinates of the up-to-4 bonded neighbors per atom.
    neighbor_pos = positions[batch_indices, neighbor_indices]
    # Use the 4th neighbor as apex for 4-coordinate centers, or central atom for 3-coordinate.
    ref_pos = jnp.where(has_four_neighbors[..., None], neighbor_pos[:, :, 3], positions)
    u0 = neighbor_pos[:, :, 0] - ref_pos
    u1 = neighbor_pos[:, :, 1] - ref_pos
    u2 = neighbor_pos[:, :, 2] - ref_pos
    return jnp.sum(jnp.cross(u0, u1) * u2, axis=-1)


@jt.typed
def stereocenter_violations(
    *,
    predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    target_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    atom_features: jt.Float[jt.Array, "batch num_atoms num_atom_features"],
    bond_features: jt.Float[jt.Array, "batch num_atoms num_atoms num_bond_features"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
) -> StereocenterStats:
    """Counts tetrahedral stereocenters whose 3D handedness disagrees with the target conformer.

    Args:
        predicted_positions: Predicted 3D atomic coordinates.
        target_positions: Ground-truth 3D atomic coordinates.
        atom_features: Per-atom 2D feature vectors ending with CIP chirality channels [none, R, S].
        bond_features: Pairwise bond feature tensors with bond-order channels at indices 1..4.
        atom_mask: Valid-atom mask.

    Returns:
        Per-molecule violation and stereocenter counts alongside the batch-wide violation fraction.
    """
    batch_size, num_atoms, _ = predicted_positions.shape
    if num_atoms < _MIN_TETRAHEDRAL_ATOMS:
        zeros_i32 = jnp.zeros((batch_size,), dtype=jnp.int32)
        return StereocenterStats(
            num_violations=zeros_i32,
            num_stereocenters=zeros_i32,
            violation_rate=jnp.zeros(()),
        )

    has_cip_stereo = (jnp.sum(atom_features[..., -2:], axis=-1) > _ONE_HOT_THRESHOLD) & atom_mask
    valid_pairs = atom_mask[:, :, None] & atom_mask[:, None, :]
    not_self = ~(jnp.eye(num_atoms, dtype=jnp.bool_)[None, :, :])
    # (batch, num_atoms, num_atoms) covalent adjacency mask across bond-order channels 1..4.
    is_bonded = (
        (jnp.sum(bond_features[..., 1:5], axis=-1) > _ONE_HOT_THRESHOLD) & valid_pairs & not_self
    )

    degree = jnp.sum(is_bonded, axis=-1)
    is_stereocenter = has_cip_stereo & (degree >= _MIN_STEREOCENTER_DEGREE)
    has_four_neighbors = degree >= _MIN_TETRAHEDRAL_ATOMS
    _, neighbor_indices = jax.lax.top_k(1 * is_bonded, k=_MIN_TETRAHEDRAL_ATOMS)

    pred_volumes = _tetrahedral_volumes(predicted_positions, neighbor_indices, has_four_neighbors)
    target_volumes = _tetrahedral_volumes(target_positions, neighbor_indices, has_four_neighbors)
    is_violated = is_stereocenter & ((pred_volumes * target_volumes) <= 0.0)

    num_violations = jnp.sum(is_violated, axis=-1)
    num_stereocenters = jnp.sum(is_stereocenter, axis=-1)
    violation_rate = jnp.sum(num_violations) / jnp.maximum(jnp.sum(num_stereocenters), 1)
    return StereocenterStats(
        num_violations=num_violations,
        num_stereocenters=num_stereocenters,
        violation_rate=violation_rate,
    )
