"""3D coordinate centering, rotation augmentation, and Kabsch alignment in JAX."""

from typing import Final

import jax
import jax.numpy as jnp

from miniqm import jaxtypes as jt

_MIN_NORM: Final = 1e-10  # Floor on Gram-Schmidt vector norms to prevent division by zero.
_SPATIAL_DIMS: Final = 3


@jt.typed
def _masked_centroid(
    positions: jt.Float[jax.Array, "*batch num_atoms 3"],
    atom_mask: jt.Bool[jax.Array, "*batch num_atoms"],
) -> jt.Float[jax.Array, "*batch 1 3"]:
    """Computes the masked mean coordinate across the atom axis."""
    # (*batch, num_atoms, 1) float atom mask for coordinate broadcasting.
    mask = atom_mask[..., None].astype(positions.dtype)
    # (*batch, 1, 1) valid atom count per molecule, clamped to >= 1.
    atom_count = jnp.maximum(mask.sum(axis=-2, keepdims=True), 1.0)
    return (positions * mask).sum(axis=-2, keepdims=True) / atom_count


@jt.typed
def _normalize(
    vectors: jt.Float[jax.Array, "*batch 3"],
) -> jt.Float[jax.Array, "*batch 3"]:
    """Normalizes 3D vectors to unit length with a norm floor."""
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, _MIN_NORM)


@jt.typed
def zero_center_of_mass(
    positions: jt.Float[jax.Array, "*batch num_atoms 3"],
    atom_mask: jt.Bool[jax.Array, "*batch num_atoms"],
) -> jt.Float[jax.Array, "*batch num_atoms 3"]:
    """Subtracts the masked centroid across atoms and zeroes padded positions.

    Args:
        positions: Atomic coordinates.
        atom_mask: Valid-atom mask.

    Returns:
        Centered coordinates with padded atoms set to zero.
    """
    centroid = _masked_centroid(positions, atom_mask)
    return (positions - centroid) * atom_mask[..., None]


@jt.typed
def random_rotation(
    key: jax.Array,
    positions: jt.Float[jax.Array, "*batch num_atoms 3"],
    atom_mask: jt.Bool[jax.Array, "*batch num_atoms"],
) -> jt.Float[jax.Array, "*batch num_atoms 3"]:
    """Applies a uniform random 3D proper rotation per batch element.

    Args:
        key: PRNG key for sampling rotations.
        positions: Atomic coordinates.
        atom_mask: Valid-atom mask.

    Returns:
        Rotated coordinates with padded atoms set to zero.
    """
    *batch_shape, _, _ = positions.shape
    # (*batch, 2, 3) two random 3D vectors for Gram-Schmidt orthogonalization.
    basis = jax.random.normal(key, shape=(*batch_shape, 2, _SPATIAL_DIMS), dtype=positions.dtype)
    first_basis = basis[..., 0, :]
    second_basis = basis[..., 1, :]

    axis_x = _normalize(first_basis)
    projection = jnp.sum(second_basis * axis_x, axis=-1, keepdims=True)
    axis_y = _normalize(second_basis - projection * axis_x)
    axis_z = jnp.cross(axis_x, axis_y)
    # (*batch, 3, 3) proper rotation matrix with orthonormal rows (axis_x, axis_y, axis_z).
    rotation = jnp.stack((axis_x, axis_y, axis_z), axis=-2)

    # (*batch, num_atoms, 3) coordinates rotated via x @ R^T.
    rotated = jnp.einsum(
        "...ni,...ji->...nj",
        positions,
        rotation,
        precision=jax.lax.Precision.HIGHEST,
    )
    return rotated * atom_mask[..., None]


@jt.typed
def kabsch_align(
    *,
    reference: jt.Float[jax.Array, "*batch num_atoms 3"],
    target: jt.Float[jax.Array, "*batch num_atoms 3"],
    atom_mask: jt.Bool[jax.Array, "*batch num_atoms"],
) -> jt.Float[jax.Array, "*batch num_atoms 3"]:
    """Rigidly aligns target coordinates onto reference coordinates over valid atoms.

    Restricts the alignment to proper rotations (det = +1) so chiral enantiomers
    are not reflected onto each other. Detaches the optimal rotation and centroids
    from gradient computation so gradients flow only through target coordinates.

    Args:
        reference: Reference coordinates to align onto.
        target: Coordinates to be rotated and translated.
        atom_mask: Valid-atom mask.

    Returns:
        Aligned target coordinates with padded atoms set to zero.
    """
    # (*batch, num_atoms, 1) float atom mask for coordinate broadcasting.
    mask = atom_mask[..., None].astype(target.dtype)
    stopped_ref = jax.lax.stop_gradient(reference)
    stopped_target = jax.lax.stop_gradient(target)

    ref_centroid = _masked_centroid(stopped_ref, atom_mask)
    target_centroid = _masked_centroid(stopped_target, atom_mask)

    centered_ref = (stopped_ref - ref_centroid) * mask
    centered_target = (stopped_target - target_centroid) * mask

    # (*batch, 3, 3) cross-covariance matrix H = centered_target^T @ centered_ref.
    covariance = jnp.einsum(
        "...ni,...nj->...ij",
        centered_target,
        centered_ref,
        precision=jax.lax.Precision.HIGHEST,
    )
    left_singular, _, right_singular_t = jnp.linalg.svd(covariance, full_matrices=True)

    det_sign = jnp.where(jnp.linalg.det(left_singular @ right_singular_t) < 0.0, -1.0, 1.0)
    ones = jnp.ones_like(det_sign)
    # (*batch, 3, 1) row-wise sign flip [1, 1, sign(det(U @ V^T))] on V^T to enforce det(R) = +1.
    reflection_fix = jnp.stack((ones, ones, det_sign), axis=-1)[..., :, None]
    # (*batch, 3, 3) optimal proper rotation matrix R = U @ diag(1, 1, det_sign) @ V^T.
    rotation = jnp.matmul(
        left_singular,
        right_singular_t * reflection_fix,
        precision=jax.lax.Precision.HIGHEST,
    )

    aligned = (
        jnp.matmul(
            target - target_centroid,
            rotation,
            precision=jax.lax.Precision.HIGHEST,
        )
        + ref_centroid
    )
    return aligned * mask
