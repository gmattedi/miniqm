import dataclasses
from typing import Final

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry, model, noise, sampling

_TEST_DENOISER_CONFIG: Final = config_lib.DenoiserConfig(
    transformer=config_lib.TransformerConfig(hidden_dim=16, num_layers=1, num_heads=2),
)


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SamplerTestSetup:
    denoiser: model.Denoiser
    params: flax_typing.VariableDict
    batch: batch_lib.JaxBatch


@jt.typed
def _make_test_setup() -> _SamplerTestSetup:
    atom_key, bond_key, init_key = jax.random.split(jax.random.key(0), 3)
    atom_mask = jnp.array(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=jnp.bool_,
    )
    atom_features = jax.random.normal(
        atom_key,
        (2, 5, batch_lib.NUM_ATOM_FEATURES),
    )
    bond_features = jax.random.normal(
        bond_key,
        (2, 5, 5, batch_lib.NUM_BOND_FEATURES),
    )
    pair_mask = noise.off_diagonal_pair_mask(atom_mask)
    graph_batch = batch_lib.JaxBatch(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
        positions=jnp.zeros((2, 5, 3)),
        atom_cond_mask=jnp.broadcast_to(atom_mask[..., None], atom_features.shape),
        bond_cond_mask=jnp.broadcast_to(pair_mask, bond_features.shape),
    )
    denoiser = model.Denoiser(config=_TEST_DENOISER_CONFIG)
    dummy_positions = jnp.zeros((2, 5, 3))
    dummy_atoms = jnp.zeros_like(atom_features)
    dummy_bonds = jnp.zeros_like(bond_features)
    dummy_sigma = jnp.ones((2,))
    params = denoiser.init(
        init_key,
        dummy_positions,
        dummy_atoms,
        dummy_bonds,
        dummy_sigma,
        batch=graph_batch,
    )
    return _SamplerTestSetup(
        denoiser=denoiser,
        params=params,
        batch=graph_batch,
    )


def test_sample_coordinates_and_molecules_return_zero_centered_masked_outputs():
    setup = _make_test_setup()
    sampling_config = config_lib.SamplingConfig(num_steps=5)

    sampled_positions = sampling.sample_coordinates(
        jax.random.key(2),
        setup.denoiser,
        setup.params,
        batch=setup.batch,
        config=sampling_config,
    )
    sampled_molecules = sampling.sample_molecules(
        jax.random.key(2),
        setup.denoiser,
        setup.params,
        batch=setup.batch,
        config=sampling_config,
    )
    trajectory = sampling.sample_trajectory(
        jax.random.key(2),
        setup.denoiser,
        setup.params,
        batch=setup.batch,
        config=sampling_config,
    )
    mol_trajectory = sampling.sample_molecule_trajectory(
        jax.random.key(2),
        setup.denoiser,
        setup.params,
        batch=setup.batch,
        config=sampling_config,
    )

    assert trajectory.shape == (6, 2, 5, 3)
    assert mol_trajectory.atom_features.shape == (6, 2, 5, batch_lib.NUM_ATOM_FEATURES)
    assert mol_trajectory.bond_features.shape == (6, 2, 5, 5, batch_lib.NUM_BOND_FEATURES)
    np.testing.assert_allclose(trajectory[-1], sampled_positions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        sampled_molecules.positions,
        sampled_positions,
        rtol=1e-6,
        atol=1e-6,
    )
    assert sampled_positions.shape == (2, 5, 3)
    assert bool(jnp.all(jnp.isfinite(sampled_positions)))
    np.testing.assert_allclose(jnp.sum(sampled_positions, axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        sampled_positions * (~setup.batch.atom_mask)[..., None],
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        sampled_molecules.bond_features,
        jnp.swapaxes(sampled_molecules.bond_features, 1, 2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sample_coordinates_matches_analytical_ode_contraction_for_zero_network():
    setup = _make_test_setup()
    zero_params = jax.tree.map(jnp.zeros_like, setup.params)
    ode_config = config_lib.SamplingConfig(num_steps=8, gamma_0=0.0, step_scale=1.0)
    sample_key = jax.random.key(4)

    sampled_positions = sampling.sample_coordinates(
        sample_key,
        setup.denoiser,
        zero_params,
        batch=setup.batch,
        config=ode_config,
    )

    noise_config = setup.denoiser.config.noise
    schedule = noise.rollout_schedule(ode_config.num_steps, config=noise_config)
    factors = noise.preconditioning_factors(schedule[:-1], config=noise_config)
    step_multipliers = 1.0 + ((schedule[1:] - schedule[:-1]) / schedule[:-1]) * (
        1.0 - factors.c_skip
    )
    total_contraction = jnp.prod(step_multipliers)

    pos_init_key, _, _, _ = jax.random.split(sample_key, 4)
    initial_positions = geometry.zero_center_of_mass(
        schedule[0] * jax.random.normal(pos_init_key, (2, 5, 3)),
        setup.batch.atom_mask,
    )
    expected_positions = initial_positions * total_contraction

    np.testing.assert_allclose(sampled_positions, expected_positions, rtol=1e-5, atol=1e-5)


def test_euler_step_uses_unit_step_scale_below_gamma_min():
    setup = _make_test_setup()
    zero_params = jax.tree.map(jnp.zeros_like, setup.params)
    initial_state = sampling.SampledMolecules(
        positions=geometry.zero_center_of_mass(
            jnp.ones((2, 5, 3)),
            setup.batch.atom_mask,
        ),
        atom_features=jnp.ones_like(setup.batch.atom_features),
        bond_features=jnp.ones_like(setup.batch.bond_features),
    )
    sigma_current = jnp.array(0.05)
    sigma_next = jnp.array(0.0)

    # Below gamma_min=0.1, step_scale=1.5 is inactive so the step matches step_scale=1.0.
    step_with_boost = sampling._euler_step(
        jax.random.key(0),
        initial_state,
        denoiser=setup.denoiser,
        params=zero_params,
        sigma_current=sigma_current,
        sigma_next=sigma_next,
        batch=setup.batch,
        config=config_lib.SamplingConfig(gamma_min=0.1, step_scale=1.5),
    )
    step_unit = sampling._euler_step(
        jax.random.key(0),
        initial_state,
        denoiser=setup.denoiser,
        params=zero_params,
        sigma_current=sigma_current,
        sigma_next=sigma_next,
        batch=setup.batch,
        config=config_lib.SamplingConfig(gamma_min=0.1, step_scale=1.0),
    )

    np.testing.assert_allclose(
        step_with_boost.positions,
        step_unit.positions,
        rtol=1e-6,
        atol=1e-6,
    )
