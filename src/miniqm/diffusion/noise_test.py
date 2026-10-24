from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm.diffusion import noise

_DEFAULT_CONFIG: Final = noise.DEFAULT_NOISE_CONFIG


@pytest.mark.parametrize(
    ("sigma_val", "expected_skip", "expected_out"),
    (
        (0.0, 1.0, 0.0),
        (_DEFAULT_CONFIG.sigma_data, 0.5, _DEFAULT_CONFIG.sigma_data / np.sqrt(2.0)),
    ),
    ids=("zero_noise", "unit_data_noise"),
)
def test_preconditioning_factors_boundary_values(
    sigma_val: float,
    expected_skip: float,
    expected_out: float,
) -> None:
    sigma = jnp.array([sigma_val], dtype=jnp.float32)
    factors = noise.preconditioning_factors(sigma)

    np.testing.assert_allclose(factors.c_skip, [expected_skip], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(factors.c_out, [expected_out], rtol=1e-6, atol=1e-6)


def test_preconditioning_factors_preserve_variance_identities() -> None:
    sigma = jnp.geomspace(1e-3, 80.0, 32, dtype=jnp.float32)
    factors = noise.preconditioning_factors(sigma)
    noisy_var = jnp.square(sigma) + _DEFAULT_CONFIG.sigma_data**2

    np.testing.assert_allclose(jnp.square(factors.c_in) * noisy_var, 1.0, rtol=1e-5)
    np.testing.assert_allclose(
        jnp.square(factors.c_skip) * noisy_var + jnp.square(factors.c_out),
        _DEFAULT_CONFIG.sigma_data**2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        factors.loss_weight * jnp.square(factors.c_out),
        1.0,
        rtol=1e-5,
    )


def test_rollout_schedule_monotonic_and_endpoints() -> None:
    num_steps = 20
    schedule = noise.rollout_schedule(num_steps)

    assert schedule.shape == (num_steps + 1,)
    np.testing.assert_allclose(
        schedule[0],
        _DEFAULT_CONFIG.sigma_data * _DEFAULT_CONFIG.s_max,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        schedule[-1],
        _DEFAULT_CONFIG.sigma_data * _DEFAULT_CONFIG.s_min,
        rtol=1e-5,
    )
    assert bool(jnp.all(jnp.diff(schedule) < 0.0))


def test_rollout_schedule_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="Expected positive num_steps"):
        noise.rollout_schedule(0)


def test_sample_train_sigmas_log_normal_moments() -> None:
    noise_config = noise.NoiseConfig(log_sigma_mean=-1.2, log_sigma_std=1.5)
    sigmas = noise.sample_train_sigmas(
        jax.random.key(0),
        (20_000,),
        config=noise_config,
    )
    log_scaled = jnp.log(sigmas / noise_config.sigma_data)

    assert bool(jnp.all(sigmas > 0.0))
    np.testing.assert_allclose(float(jnp.mean(log_scaled)), -1.2, atol=0.03)
    np.testing.assert_allclose(float(jnp.std(log_scaled)), 1.5, atol=0.03)


def test_make_noisy_sample_preserves_target_geometry_and_masks_padding() -> None:
    positions = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-1.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.array(
        [
            [True, True, True, False],
            [True, True, False, False],
        ],
        dtype=jnp.bool_,
    )

    sample = noise.make_noisy_sample(jax.random.key(7), positions, atom_mask)

    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    orig_dists = jnp.linalg.norm(positions[:, :, None, :] - positions[:, None, :, :], axis=-1)
    target_dists = jnp.linalg.norm(
        sample.target_positions[:, :, None, :] - sample.target_positions[:, None, :, :],
        axis=-1,
    )
    np.testing.assert_allclose(target_dists * pair_mask, orig_dists * pair_mask, atol=1e-5)

    invalid_mask = (~atom_mask)[..., None]
    np.testing.assert_allclose(sample.target_positions * invalid_mask, 0.0, atol=1e-7)
    np.testing.assert_allclose(sample.noisy_positions * invalid_mask, 0.0, atol=1e-7)
    assert not np.allclose(sample.noisy_positions, sample.target_positions)
