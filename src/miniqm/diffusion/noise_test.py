from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import config as config_lib
from miniqm.data import batch as batch_lib
from miniqm.diffusion import noise

_DEFAULT_CONFIG: Final = config_lib.DEFAULT_NOISE_CONFIG


def _make_sample_jax_batch() -> batch_lib.JaxBatch:
    positions = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-1.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
    )
    atom_mask = jnp.array(
        [
            [True, True, True, False],
            [True, True, False, False],
        ],
        dtype=jnp.bool_,
    )
    atom_features = jnp.ones((2, 4, batch_lib.NUM_ATOM_FEATURES)) * atom_mask[..., None]
    bond_features = jnp.zeros((2, 4, 4, batch_lib.NUM_BOND_FEATURES))
    bond_features = bond_features.at[:, 0, 1, 1].set(1.0).at[:, 1, 0, 1].set(1.0)
    atom_cond_mask = jnp.broadcast_to(atom_mask[..., None], atom_features.shape)
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    bond_cond_mask = jnp.broadcast_to(pair_mask[..., None], bond_features.shape)
    return batch_lib.JaxBatch(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
        positions=positions,
        atom_cond_mask=atom_cond_mask,
        bond_cond_mask=bond_cond_mask,
    )


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
):
    sigma = jnp.array([sigma_val])
    factors = noise.preconditioning_factors(sigma)

    np.testing.assert_allclose(factors.c_skip, [expected_skip], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(factors.c_out, [expected_out], rtol=1e-6, atol=1e-6)


def test_preconditioning_factors_preserve_variance_identities():
    sigma = jnp.geomspace(1e-3, 80.0, 32)
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


def test_rollout_schedule_monotonic_and_endpoints():
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


def test_rollout_schedule_and_noise_config_reject_invalid_bounds():
    with pytest.raises(ValueError, match="Expected positive num_steps"):
        noise.rollout_schedule(0)

    with pytest.raises(ValueError, match="s_max"):
        config_lib.NoiseConfig(s_min=10.0, s_max=5.0)

    with pytest.raises(ValueError, match="uncond_prob"):
        config_lib.NoiseConfig(uncond_prob=0.7, partial_cond_prob=0.5)


def test_sample_train_sigmas_log_normal_moments():
    noise_config = config_lib.NoiseConfig(log_sigma_mean=-1.2, log_sigma_std=1.5)
    sigmas = noise.sample_train_sigmas(
        jax.random.key(0),
        (20_000,),
        config=noise_config,
    )
    log_scaled = jnp.log(sigmas / noise_config.sigma_data)

    assert bool(jnp.all(sigmas > 0.0))
    np.testing.assert_allclose(float(jnp.mean(log_scaled)), -1.2, atol=0.03)
    np.testing.assert_allclose(float(jnp.std(log_scaled)), 1.5, atol=0.03)


def test_make_noisy_sample_preserves_target_geometry_and_masks_padding():
    batch = _make_sample_jax_batch()

    sample = noise.make_noisy_sample(jax.random.key(7), batch)

    pair_mask = batch.atom_mask[:, :, None] & batch.atom_mask[:, None, :]
    orig_dists = jnp.linalg.norm(
        batch.positions[:, :, None, :] - batch.positions[:, None, :, :],
        axis=-1,
    )
    target_dists = jnp.linalg.norm(
        sample.target_positions[:, :, None, :] - sample.target_positions[:, None, :, :],
        axis=-1,
    )
    np.testing.assert_allclose(target_dists * pair_mask, orig_dists * pair_mask, atol=1e-5)

    invalid_mask = (~batch.atom_mask)[..., None]
    np.testing.assert_allclose(sample.target_positions * invalid_mask, 0.0, atol=1e-7)
    np.testing.assert_allclose(sample.noisy_positions * invalid_mask, 0.0, atol=1e-7)
    np.testing.assert_allclose(sample.noisy_atom_features * invalid_mask, 0.0, atol=1e-7)
    np.testing.assert_allclose(
        sample.noisy_bond_features,
        jnp.swapaxes(sample.noisy_bond_features, 1, 2),
        atol=1e-6,
    )
    assert not np.allclose(sample.noisy_positions, sample.target_positions)


def test_make_noisy_sample_supports_uncond_and_partial_cond_regimes():
    batch = _make_sample_jax_batch()

    uncond_sample = noise.make_noisy_sample(
        jax.random.key(11),
        batch,
        config=config_lib.NoiseConfig(uncond_prob=1.0, partial_cond_prob=0.0),
        sample_cond_masks=True,
    )
    assert not bool(jnp.any(uncond_sample.atom_cond_mask))
    assert not bool(jnp.any(uncond_sample.bond_cond_mask))

    partial_sample = noise.make_noisy_sample(
        jax.random.key(12),
        batch,
        config=config_lib.NoiseConfig(uncond_prob=0.0, partial_cond_prob=1.0),
        sample_cond_masks=True,
    )
    assert partial_sample.atom_cond_mask.shape == batch.atom_features.shape
    assert partial_sample.bond_cond_mask.shape == batch.bond_features.shape
