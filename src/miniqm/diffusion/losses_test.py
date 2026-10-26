import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import jaxtypes as jt
from miniqm.diffusion import geometry, losses


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _TetrahedralBatch:
    positions: jt.Float[jt.Array, "batch num_atoms 3"]
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"]
    sigma: jt.Float[jt.Array, " batch"]


@jt.typed
def _make_tetrahedral_batch() -> _TetrahedralBatch:
    raw_positions = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, 1.6],
                [0.0, 0.0, 0.0],
            ]
        ],
    )
    atom_mask = jnp.array([[True, True, True, True, False]], dtype=jnp.bool_)
    positions = geometry.zero_center_of_mass(raw_positions, atom_mask)
    sigma = jnp.array([1.2])
    return _TetrahedralBatch(positions=positions, atom_mask=atom_mask, sigma=sigma)


def test_compute_diffusion_loss_is_zero_under_proper_rotation():
    batch = _make_tetrahedral_batch()
    rotated_pred = geometry.random_rotation(jax.random.key(0), batch.positions, batch.atom_mask)

    output = losses.compute_diffusion_loss(
        predicted_positions=rotated_pred,
        target_positions=batch.positions,
        sigma=batch.sigma,
        atom_mask=batch.atom_mask,
    )

    np.testing.assert_allclose(float(output.aligned_mse_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(output.distance_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(output.total_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(output.rmsd), 0.0, atol=2e-4)


def test_aligned_mse_penalizes_mirror_enantiomer_while_distance_loss_is_zero():
    batch = _make_tetrahedral_batch()
    reflected_pred = batch.positions * jnp.array([1.0, 1.0, -1.0])

    output = losses.compute_diffusion_loss(
        predicted_positions=reflected_pred,
        target_positions=batch.positions,
        sigma=batch.sigma,
        atom_mask=batch.atom_mask,
    )

    np.testing.assert_allclose(float(output.distance_loss), 0.0, atol=1e-6)
    assert float(output.aligned_mse_loss) > 0.1
    assert float(output.rmsd) > 0.3


@pytest.mark.parametrize(
    ("cutoff", "expected_loss"),
    (
        (3.0, 0.0),
        (10.0, 8.0 / 3.0),
    ),
    ids=("excludes_distant_pair", "includes_distant_pair"),
)
def test_pairwise_distance_loss_respects_cutoff_and_padding(
    cutoff: float,
    expected_loss: float,
):
    target_positions = jnp.array(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    )
    predicted_positions = jnp.array(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0], [99.0, 0.0, 0.0]]],
    )
    atom_mask = jnp.array([[True, True, True, False]], dtype=jnp.bool_)

    dist_loss = losses.pairwise_distance_loss(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cutoff=cutoff,
    )

    np.testing.assert_allclose(float(dist_loss), expected_loss, rtol=1e-5, atol=1e-6)


def test_compute_diffusion_loss_has_finite_gradients_and_zero_padded_grad():
    batch = _make_tetrahedral_batch()
    noisy_pred = (batch.positions + 0.25) * batch.atom_mask[..., None]

    @jt.typed
    def total_loss_fn(
        predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    ) -> jt.Float[jt.Array, ""]:
        return losses.compute_diffusion_loss(
            predicted_positions=predicted_positions,
            target_positions=batch.positions,
            sigma=batch.sigma,
            atom_mask=batch.atom_mask,
        ).total_loss

    grad_at_noisy = jax.grad(total_loss_fn)(noisy_pred)
    grad_at_exact = jax.grad(total_loss_fn)(batch.positions)

    assert bool(jnp.all(jnp.isfinite(grad_at_noisy)))
    assert bool(jnp.all(jnp.isfinite(grad_at_exact)))
    assert float(jnp.linalg.norm(grad_at_noisy[:, :4])) > 0.0
    np.testing.assert_allclose(grad_at_noisy[:, 4:], 0.0, atol=1e-7)


def test_compute_diffusion_loss_includes_atom_and_bond_feature_mse():
    batch = _make_tetrahedral_batch()
    target_atoms = jnp.ones((1, 5, 26)) * batch.atom_mask[..., None]
    pred_atoms = jnp.zeros((1, 5, 26))
    target_bonds = jnp.ones((1, 5, 5, 6))
    pred_bonds = jnp.zeros((1, 5, 5, 6))

    output = losses.compute_diffusion_loss(
        predicted_positions=batch.positions,
        target_positions=batch.positions,
        predicted_atom_features=pred_atoms,
        target_atom_features=target_atoms,
        predicted_bond_features=pred_bonds,
        target_bond_features=target_bonds,
        sigma=batch.sigma,
        atom_mask=batch.atom_mask,
    )

    assert float(output.atom_loss) > 0.0
    assert float(output.bond_loss) > 0.0
    np.testing.assert_allclose(
        float(output.total_loss),
        0.25 * float(output.atom_loss) + 0.5 * float(output.bond_loss),
        rtol=1e-5,
    )
