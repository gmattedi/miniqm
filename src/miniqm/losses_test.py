import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import geometry, losses
from miniqm import jaxtypes as jt


@jt.typed
def _make_tetrahedral_batch() -> tuple[
    jt.Float[jt.Array, "batch num_atoms 3"],
    jt.Bool[jt.Array, "batch num_atoms"],
    jt.Float[jt.Array, " batch"],
]:
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
    return positions, atom_mask, sigma


def test_compute_diffusion_loss_is_zero_under_proper_rotation():
    target_positions, atom_mask, sigma = _make_tetrahedral_batch()
    rotated_pred = geometry.random_rotation(jax.random.key(0), target_positions, atom_mask)

    output = losses.compute_diffusion_loss(
        predicted_positions=rotated_pred,
        target_positions=target_positions,
        sigma=sigma,
        atom_mask=atom_mask,
    )

    np.testing.assert_allclose(float(output.aligned_mse_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(output.distance_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(output.total_loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(output.rmsd), 0.0, atol=2e-4)


def test_aligned_mse_penalizes_mirror_enantiomer_while_distance_loss_is_zero():
    target_positions, atom_mask, sigma = _make_tetrahedral_batch()
    reflected_pred = target_positions * jnp.array([1.0, 1.0, -1.0], dtype=jnp.float32)

    output = losses.compute_diffusion_loss(
        predicted_positions=reflected_pred,
        target_positions=target_positions,
        sigma=sigma,
        atom_mask=atom_mask,
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


def test_compute_diffusion_loss_has_finite_gradients_and_zero_padded_grad():
    target_positions, atom_mask, sigma = _make_tetrahedral_batch()
    noisy_pred = (target_positions + 0.25) * atom_mask[..., None]

    def total_loss_fn(pred: jt.Float[jt.Array, "batch num_atoms 3"]) -> jt.Array:
        return losses.compute_diffusion_loss(
            predicted_positions=pred,
            target_positions=target_positions,
            sigma=sigma,
            atom_mask=atom_mask,
        ).total_loss

    grad_at_noisy = jax.grad(total_loss_fn)(noisy_pred)
    grad_at_exact = jax.grad(total_loss_fn)(target_positions)

    assert bool(jnp.all(jnp.isfinite(grad_at_noisy)))
    assert bool(jnp.all(jnp.isfinite(grad_at_exact)))
    assert float(jnp.linalg.norm(grad_at_noisy[:, :4])) > 0.0
    np.testing.assert_allclose(grad_at_noisy[:, 4:], 0.0, atol=1e-7)
