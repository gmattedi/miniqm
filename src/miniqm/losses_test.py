import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import geometry, losses
from miniqm import jaxtypes as jt


@jt.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class _TetrahedralBatch:
    """Single-molecule batch containing a 4-atom chiral pyramid and 1 padded slot."""

    positions: jt.Float[jt.Array, "batch num_atoms 3"]
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"]
    sigma: jt.Float[jt.Array, " batch"]


@jt.typed
def _make_tetrahedral_batch() -> _TetrahedralBatch:
    """Creates a 1-molecule batch containing a 4-atom non-planar pyramid and 1 padded slot."""
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
        dtype=jnp.float32,
    )
    atom_mask = jnp.array([[True, True, True, True, False]], dtype=jnp.bool_)
    positions = geometry.zero_center_of_mass(raw_positions, atom_mask)
    sigma = jnp.array([1.2], dtype=jnp.float32)
    return _TetrahedralBatch(positions=positions, atom_mask=atom_mask, sigma=sigma)


def test_compute_diffusion_loss_is_zero_under_proper_rotation() -> None:
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


def test_aligned_mse_penalizes_mirror_enantiomer_while_distance_loss_is_zero() -> None:
    batch = _make_tetrahedral_batch()
    reflected_pred = batch.positions * jnp.array([1.0, 1.0, -1.0], dtype=jnp.float32)

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
) -> None:
    target_positions = jnp.array(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    predicted_positions = jnp.array(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0], [99.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array([[True, True, True, False]], dtype=jnp.bool_)

    dist_loss = losses.pairwise_distance_loss(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cutoff=cutoff,
    )

    np.testing.assert_allclose(float(dist_loss), expected_loss, rtol=1e-5, atol=1e-6)


def test_compute_diffusion_loss_has_finite_gradients_and_zero_padded_grad() -> None:
    batch = _make_tetrahedral_batch()
    noisy_pred = (batch.positions + 0.25) * batch.atom_mask[..., None]

    @jt.typed
    def total_loss_fn(
        predicted_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    ) -> jt.Float[jt.Array, ""]:
        """Evaluates total diffusion loss as a scalar function of predicted coordinates."""
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
