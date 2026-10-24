import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import geometry
from miniqm import jaxtypes as jt


@jt.typed
def _tetrahedral_signed_volume(
    positions: jt.Float[jax.Array, "*batch num_atoms 3"],
) -> jt.Float[jax.Array, "*batch"]:
    """Computes the signed scalar triple product of the first four atoms."""
    edge_a = positions[..., 1, :] - positions[..., 0, :]
    edge_b = positions[..., 2, :] - positions[..., 0, :]
    edge_c = positions[..., 3, :] - positions[..., 0, :]
    return jnp.sum(edge_a * jnp.cross(edge_b, edge_c), axis=-1)


@pytest.mark.parametrize(
    ("positions", "atom_mask", "expected_positions"),
    (
        (
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [3.0, 6.0, 9.0],
                    [99.0, -99.0, 42.0],
                ],
                dtype=jnp.float32,
            ),
            jnp.array([True, True, False]),
            jnp.array(
                [
                    [-1.0, -2.0, -3.0],
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
        ),
        (
            jnp.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=jnp.float32,
            ),
            jnp.array([False, False]),
            jnp.zeros((2, 3), dtype=jnp.float32),
        ),
    ),
    ids=("masked_padding", "all_masked_out"),
)
@jt.typed
def test_zero_center_of_mass(
    positions: jt.Float[jax.Array, "num_atoms 3"],
    atom_mask: jt.Bool[jax.Array, "*batch num_atoms"],
    expected_positions: jt.Float[jax.Array, "num_atoms 3"],
) -> None:
    centered_positions = geometry.zero_center_of_mass(positions, atom_mask)
    np.testing.assert_allclose(centered_positions, expected_positions, atol=1e-6)


def test_random_rotation_preserves_distances_and_chirality() -> None:
    chiral_coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.5],
            [50.0, -30.0, 10.0],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array(
        [
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )
    centered_positions = geometry.zero_center_of_mass(
        jnp.stack((chiral_coords, chiral_coords), axis=0),
        atom_mask,
    )

    rotated_positions = geometry.random_rotation(jax.random.key(7), centered_positions, atom_mask)

    orig_diffs = centered_positions[:, :4, None, :] - centered_positions[:, None, :4, :]
    rot_diffs = rotated_positions[:, :4, None, :] - rotated_positions[:, None, :4, :]
    np.testing.assert_allclose(
        jnp.linalg.norm(rot_diffs, axis=-1),
        jnp.linalg.norm(orig_diffs, axis=-1),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _tetrahedral_signed_volume(rotated_positions),
        _tetrahedral_signed_volume(centered_positions),
        atol=1e-5,
    )
    np.testing.assert_array_equal(rotated_positions[:, 4, :], jnp.zeros((2, 3), dtype=jnp.float32))
    assert not jnp.allclose(rotated_positions[0], rotated_positions[1])


def test_kabsch_align_recovers_rotated_and_translated_coordinates() -> None:
    raw_coords = jnp.array(
        [
            [
                [0.1, -0.2, 0.3],
                [1.4, 0.2, -0.1],
                [-0.5, 1.7, 0.2],
                [0.2, -0.3, 2.1],
                [0.0, 0.0, 0.0],
            ],
            [
                [-0.3, 0.4, 0.1],
                [1.1, -0.8, 0.5],
                [0.6, 1.2, -0.9],
                [-1.0, -0.5, 1.8],
                [0.0, 0.0, 0.0],
            ],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array(
        [
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )
    reference = geometry.zero_center_of_mass(raw_coords, atom_mask)
    translation = jnp.array([[[3.5, -2.0, 7.0]], [[-4.0, 1.5, -3.0]]], dtype=jnp.float32)
    pad_garbage = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 8.0, 7.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 4.0, 3.0]],
        ],
        dtype=jnp.float32,
    )
    rotated_positions = geometry.random_rotation(jax.random.key(42), reference, atom_mask)
    target = (rotated_positions + translation) * atom_mask[..., None] + pad_garbage

    aligned_positions = geometry.kabsch_align(
        reference=reference,
        target=target,
        atom_mask=atom_mask,
    )

    np.testing.assert_allclose(aligned_positions, reference, atol=1e-5)


def test_kabsch_align_preserves_chirality_against_mirror_reflection() -> None:
    chiral_coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array([True, True, True, True])
    reference = geometry.zero_center_of_mass(chiral_coords, atom_mask)
    mirror_enantiomer = reference * jnp.array([1.0, 1.0, -1.0], dtype=jnp.float32)

    aligned_positions = geometry.kabsch_align(
        reference=reference,
        target=mirror_enantiomer,
        atom_mask=atom_mask,
    )

    np.testing.assert_allclose(
        _tetrahedral_signed_volume(aligned_positions),
        _tetrahedral_signed_volume(mirror_enantiomer),
        atol=1e-5,
    )
    assert float(jnp.sum((aligned_positions - reference) ** 2)) > 1.0


def test_kabsch_align_gradients_flow_and_match_both_alignment_directions() -> None:
    ground_truth = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.6, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array([True, True, True, True, False])
    ground_truth = geometry.zero_center_of_mass(ground_truth, atom_mask)
    perturbed_positions = geometry.random_rotation(
        jax.random.key(11),
        ground_truth + jnp.array([0.1, -0.05, 0.08], dtype=jnp.float32),
        atom_mask,
    )

    @jt.typed
    def loss_align_pred_to_gt(
        pred: jt.Float[jax.Array, "num_atoms 3"],
    ) -> jt.Float[jax.Array, ""]:
        """Computes squared error when aligning predicted coordinates onto ground truth."""
        aligned_pred = geometry.kabsch_align(
            reference=ground_truth,
            target=pred,
            atom_mask=atom_mask,
        )
        return jnp.sum(((aligned_pred - ground_truth) * atom_mask[..., None]) ** 2)

    @jt.typed
    def loss_align_gt_to_pred(
        pred: jt.Float[jax.Array, "num_atoms 3"],
    ) -> jt.Float[jax.Array, ""]:
        """Computes squared error when aligning ground truth onto predicted coordinates."""
        aligned_gt = geometry.kabsch_align(
            reference=pred,
            target=ground_truth,
            atom_mask=atom_mask,
        )
        return jnp.sum(((pred - aligned_gt) * atom_mask[..., None]) ** 2)

    grad_pred_to_gt = jax.grad(loss_align_pred_to_gt)(perturbed_positions)
    grad_gt_to_pred = jax.grad(loss_align_gt_to_pred)(perturbed_positions)

    assert jnp.all(jnp.isfinite(grad_pred_to_gt))
    assert float(jnp.linalg.norm(grad_pred_to_gt[:4])) > 0.0
    np.testing.assert_array_equal(grad_pred_to_gt[4], jnp.zeros((3,), dtype=jnp.float32))
    np.testing.assert_allclose(grad_pred_to_gt, grad_gt_to_pred, atol=1e-5)
