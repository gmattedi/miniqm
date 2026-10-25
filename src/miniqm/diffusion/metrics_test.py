import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry, metrics


def test_rmsd_per_molecule_is_zero_under_proper_rotation():
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
    target_positions = geometry.zero_center_of_mass(raw_positions, atom_mask)
    rotated_positions = geometry.random_rotation(jax.random.key(0), target_positions, atom_mask)

    rmsd = metrics.rmsd_per_molecule(
        predicted_positions=rotated_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
    )
    np.testing.assert_allclose(float(rmsd[0]), 0.0, atol=2e-4)


@pytest.mark.parametrize(
    ("stretched_x", "expected_lddt"),
    (
        (1.05, 1.0),
        (1.20, 0.75),
        (1.40, 0.50),
        (1.80, 0.25),
        (2.50, 0.0),
    ),
    ids=("all_pass", "three_pass", "two_pass", "one_pass", "none_pass"),
)
def test_lddt_per_molecule_scores_distance_errors_across_thresholds(
    stretched_x: float,
    expected_lddt: float,
):
    target_positions = jnp.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    )
    predicted_positions = jnp.array(
        [[[0.0, 0.0, 0.0], [stretched_x, 0.0, 0.0], [99.0, 0.0, 0.0]]],
    )
    atom_mask = jnp.array([[True, True, False]], dtype=jnp.bool_)

    scores = metrics.lddt_per_molecule(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_mask=atom_mask,
    )
    np.testing.assert_allclose(float(scores[0]), expected_lddt, atol=1e-6)


@pytest.mark.parametrize(
    ("chiral_smiles", "z_scale", "expected_violations", "expected_rate"),
    (
        ("C[C@H](N)O", 1.0, (0, 0), 0.0),
        ("C[C@H](N)O", -1.0, (1, 0), 1.0),
        ("C[C@](F)(N)O", 1.0, (0, 0), 0.0),
        ("C[C@](F)(N)O", -1.0, (1, 0), 1.0),
    ),
    ids=(
        "tertiary_proper_rotation",
        "tertiary_reflection",
        "quaternary_proper_rotation",
        "quaternary_reflection",
    ),
)
def test_stereocenter_violations_detects_enantiomer_inversion(
    chiral_smiles: str,
    z_scale: float,
    expected_violations: tuple[int, int],
    expected_rate: float,
):
    host_batch = batch_lib.collate_smiles((chiral_smiles, "CC"))
    jax_batch = host_batch.to_jax()
    target_positions = (
        jax.random.normal(jax.random.key(7), jax_batch.positions.shape)
        * jax_batch.atom_mask[..., None]
    )
    rotated_positions = geometry.random_rotation(
        jax.random.key(11),
        target_positions,
        jax_batch.atom_mask,
    )
    predicted_positions = rotated_positions * jnp.array([1.0, 1.0, z_scale])

    stats = metrics.stereocenter_violations(
        predicted_positions=predicted_positions,
        target_positions=target_positions,
        atom_features=jax_batch.atom_features,
        bond_features=jax_batch.bond_features,
        atom_mask=jax_batch.atom_mask,
    )

    assert tuple(int(v) for v in stats.num_stereocenters) == (1, 0)
    assert tuple(int(v) for v in stats.num_violations) == expected_violations
    np.testing.assert_allclose(float(stats.violation_rate), expected_rate, atol=1e-6)


def test_stereocenter_violations_handles_fewer_than_four_atoms():
    positions = jnp.zeros((1, 3, 3))
    atom_features = jnp.zeros((1, 3, batch_lib.NUM_ATOM_FEATURES))
    bond_features = jnp.zeros((1, 3, 3, batch_lib.NUM_BOND_FEATURES))
    atom_mask = jnp.ones((1, 3), dtype=jnp.bool_)

    stats = metrics.stereocenter_violations(
        predicted_positions=positions,
        target_positions=positions,
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
    )

    assert int(stats.num_stereocenters[0]) == 0
    assert int(stats.num_violations[0]) == 0
    assert float(stats.violation_rate) == 0.0
