import dataclasses
from typing import Final

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import batch as batch_lib
from miniqm import geometry, model, noise, sampling
from miniqm import jaxtypes as jt

_TEST_DENOISER_CONFIG: Final = model.DenoiserConfig(
    transformer=model.TransformerConfig(hidden_dim=16, num_layers=1, num_heads=2),
)


@jt.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class _SamplerTestSetup:
    """Synthetic 2-molecule graph batch and initialized Denoiser for sampler tests."""

    denoiser: model.Denoiser
    params: flax_typing.VariableDict
    batch: batch_lib.JaxBatch


@jt.typed
def _make_test_setup() -> _SamplerTestSetup:
    """Creates a 2-molecule synthetic graph batch and initialized Denoiser parameters."""
    atom_key, bond_key, init_key = jax.random.split(jax.random.key(0), 3)
    graph_batch = batch_lib.JaxBatch(
        atom_features=jax.random.normal(atom_key, (2, 5, 22), dtype=jnp.float32),
        bond_features=jax.random.normal(bond_key, (2, 5, 5, 6), dtype=jnp.float32),
        atom_mask=jnp.array(
            [
                [True, True, True, True, False],
                [True, True, True, False, False],
            ],
            dtype=jnp.bool_,
        ),
    )
    denoiser = model.Denoiser(config=_TEST_DENOISER_CONFIG)
    dummy_positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    dummy_sigma = jnp.ones((2,), dtype=jnp.float32)
    params = denoiser.init(
        init_key,
        dummy_positions,
        dummy_sigma,
        batch=graph_batch,
    )
    return _SamplerTestSetup(
        denoiser=denoiser,
        params=params,
        batch=graph_batch,
    )


def test_sample_coordinates_returns_zero_centered_masked_positions() -> None:
    setup = _make_test_setup()
    sampling_config = sampling.SamplingConfig(num_steps=5)

    sampled_positions = sampling.sample_coordinates(
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

    assert trajectory.shape == (6, 2, 5, 3)
    np.testing.assert_allclose(trajectory[-1], sampled_positions, rtol=1e-6, atol=1e-6)
    assert sampled_positions.shape == (2, 5, 3)
    assert bool(jnp.all(jnp.isfinite(sampled_positions)))
    np.testing.assert_allclose(jnp.sum(sampled_positions, axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        sampled_positions * (~setup.batch.atom_mask)[..., None],
        0.0,
        atol=1e-7,
    )


def test_sample_coordinates_matches_analytical_ode_contraction_for_zero_network() -> None:
    setup = _make_test_setup()
    zero_params = jax.tree.map(jnp.zeros_like, setup.params)
    ode_config = sampling.SamplingConfig(num_steps=8, gamma_0=0.0, step_scale=1.0)
    sample_key = jax.random.key(4)

    sampled_positions = sampling.sample_coordinates(
        sample_key,
        setup.denoiser,
        zero_params,
        batch=setup.batch,
        config=ode_config,
    )

    schedule = noise.rollout_schedule(ode_config.num_steps, config=ode_config.noise)
    factors = noise.preconditioning_factors(schedule[:-1], config=ode_config.noise)
    step_multipliers = 1.0 + ((schedule[1:] - schedule[:-1]) / schedule[:-1]) * (
        1.0 - factors.c_skip
    )
    total_contraction = jnp.prod(step_multipliers)

    initial_key, _ = jax.random.split(sample_key)
    initial_positions = geometry.zero_center_of_mass(
        schedule[0] * jax.random.normal(initial_key, (2, 5, 3), dtype=jnp.float32),
        setup.batch.atom_mask,
    )
    expected_positions = initial_positions * total_contraction

    np.testing.assert_allclose(sampled_positions, expected_positions, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("num_steps", "gamma_0", "noise_scale", "step_scale", "expected_message"),
    (
        (0, 0.8, 1.003, 1.5, "Expected positive num_steps"),
        (10, -0.1, 1.003, 1.5, "Expected non-negative gamma_0"),
        (10, 0.8, -1.0, 1.5, "Expected non-negative noise_scale"),
        (10, 0.8, 1.003, 0.0, "Expected positive step_scale"),
    ),
    ids=("zero_steps", "negative_gamma", "negative_noise_scale", "zero_step_scale"),
)
def test_sampling_config_rejects_invalid_parameters(
    num_steps: int,
    gamma_0: float,
    noise_scale: float,
    step_scale: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        sampling.SamplingConfig(
            num_steps=num_steps,
            gamma_0=gamma_0,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )
