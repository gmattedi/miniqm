import dataclasses
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry, model, noise

_TEST_CONFIG: Final = config_lib.DenoiserConfig(
    transformer=config_lib.TransformerConfig(hidden_dim=32, num_layers=2, num_heads=4),
)


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ModelTestInputs:
    noisy_positions: jt.Float[jt.Array, "batch num_atoms 3"]
    noisy_atom_features: jt.Float[jt.Array, "batch num_atoms num_atom_features"]
    noisy_bond_features: jt.Float[jt.Array, "batch num_atoms num_atoms num_bond_features"]
    sigma: jt.Float[jt.Array, " batch"]
    batch: batch_lib.JaxBatch


@jt.typed
def _make_dummy_inputs() -> _ModelTestInputs:
    pos_key, atom_key, bond_key, n_atom_key, n_bond_key = jax.random.split(
        jax.random.key(0),
        5,
    )
    atom_mask = jnp.array(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=jnp.bool_,
    )
    mask_3d = atom_mask[..., None]
    pair_mask = noise.off_diagonal_pair_mask(atom_mask)

    noisy_positions = jax.random.normal(pos_key, (2, 5, 3)) * mask_3d
    noisy_atom_features = (
        jax.random.normal(
            n_atom_key,
            (2, 5, batch_lib.NUM_ATOM_FEATURES),
        )
        * mask_3d
    )
    noisy_bond_features = (
        noise.symmetric_bond_noise(
            n_bond_key,
            (2, 5, 5, batch_lib.NUM_BOND_FEATURES),
        )
        * pair_mask
    )
    sigma = jnp.array([0.5, 2.0])
    atom_features = jax.random.normal(
        atom_key,
        (2, 5, batch_lib.NUM_ATOM_FEATURES),
    )
    bond_features = jax.random.normal(
        bond_key,
        (2, 5, 5, batch_lib.NUM_BOND_FEATURES),
    )
    graph_batch = batch_lib.JaxBatch(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
        positions=jnp.zeros((2, 5, 3)),
        atom_cond_mask=jnp.broadcast_to(mask_3d, atom_features.shape),
        bond_cond_mask=jnp.broadcast_to(pair_mask, bond_features.shape),
    )
    return _ModelTestInputs(
        noisy_positions=noisy_positions,
        noisy_atom_features=noisy_atom_features,
        noisy_bond_features=noisy_bond_features,
        sigma=sigma,
        batch=graph_batch,
    )


def test_denoiser_outputs_zero_centered_and_masked_predictions():
    inputs = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(1),
        inputs.noisy_positions,
        inputs.noisy_atom_features,
        inputs.noisy_bond_features,
        inputs.sigma,
        batch=inputs.batch,
    )

    output = cast(
        "model.DenoiserOutput",
        denoiser.apply(
            params,
            inputs.noisy_positions,
            inputs.noisy_atom_features,
            inputs.noisy_bond_features,
            inputs.sigma,
            batch=inputs.batch,
        ),
    )

    assert output.positions.shape == inputs.noisy_positions.shape
    assert output.atom_features.shape == inputs.noisy_atom_features.shape
    assert output.bond_features.shape == inputs.noisy_bond_features.shape
    np.testing.assert_allclose(jnp.sum(output.positions, axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        output.positions * (~inputs.batch.atom_mask)[..., None],
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        output.atom_features * (~inputs.batch.atom_mask)[..., None],
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        output.bond_features,
        jnp.swapaxes(output.bond_features, 1, 2),
        atol=1e-6,
    )


def test_denoiser_is_invariant_to_padded_slot_perturbations():
    inputs = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(2),
        inputs.noisy_positions,
        inputs.noisy_atom_features,
        inputs.noisy_bond_features,
        inputs.sigma,
        batch=inputs.batch,
    )

    pad_mask = ~inputs.batch.atom_mask
    perturbed_positions = inputs.noisy_positions + 50.0 * pad_mask[..., None]
    perturbed_atoms = inputs.noisy_atom_features + 25.0 * pad_mask[..., None]
    perturbed_batch = dataclasses.replace(
        inputs.batch,
        atom_features=inputs.batch.atom_features + 10.0 * pad_mask[..., None],
    )

    clean_out = cast(
        "model.DenoiserOutput",
        denoiser.apply(
            params,
            inputs.noisy_positions,
            inputs.noisy_atom_features,
            inputs.noisy_bond_features,
            inputs.sigma,
            batch=inputs.batch,
        ),
    )
    perturbed_out = cast(
        "model.DenoiserOutput",
        denoiser.apply(
            params,
            perturbed_positions,
            perturbed_atoms,
            inputs.noisy_bond_features,
            inputs.sigma,
            batch=perturbed_batch,
        ),
    )

    np.testing.assert_allclose(clean_out.positions, perturbed_out.positions, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        clean_out.atom_features,
        perturbed_out.atom_features,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        clean_out.bond_features,
        perturbed_out.bond_features,
        rtol=1e-5,
        atol=1e-5,
    )


def test_denoiser_recovers_centered_input_at_vanishing_noise():
    inputs = _make_dummy_inputs()
    tiny_sigma = jnp.array([1e-7, 1e-7])
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(3),
        inputs.noisy_positions,
        inputs.noisy_atom_features,
        inputs.noisy_bond_features,
        tiny_sigma,
        batch=inputs.batch,
    )

    output = cast(
        "model.DenoiserOutput",
        denoiser.apply(
            params,
            inputs.noisy_positions,
            inputs.noisy_atom_features,
            inputs.noisy_bond_features,
            tiny_sigma,
            batch=inputs.batch,
        ),
    )
    expected_positions = geometry.zero_center_of_mass(
        inputs.noisy_positions,
        inputs.batch.atom_mask,
    )

    np.testing.assert_allclose(output.positions, expected_positions, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        output.atom_features,
        inputs.noisy_atom_features,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        output.bond_features,
        inputs.noisy_bond_features,
        rtol=1e-5,
        atol=1e-5,
    )


def test_denoiser_gradients_are_finite_and_non_zero():
    inputs = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(4),
        inputs.noisy_positions,
        inputs.noisy_atom_features,
        inputs.noisy_bond_features,
        inputs.sigma,
        batch=inputs.batch,
    )

    def loss_fn(model_params: flax_typing.VariableDict) -> jt.Array:
        output = cast(
            "model.DenoiserOutput",
            denoiser.apply(
                model_params,
                inputs.noisy_positions,
                inputs.noisy_atom_features,
                inputs.noisy_bond_features,
                inputs.sigma,
                batch=inputs.batch,
            ),
        )
        return (
            jnp.mean(jnp.square(output.positions))
            + jnp.mean(jnp.square(output.atom_features))
            + jnp.mean(jnp.square(output.bond_features))
        )

    grads = jax.grad(loss_fn)(params)
    grad_leaves = jax.tree.leaves(grads)

    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves)
    assert any(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in grad_leaves)


def test_transformer_config_rejects_incompatible_hidden_dim():
    with pytest.raises(ValueError, match="Expected hidden_dim"):
        config_lib.TransformerConfig(hidden_dim=30, num_layers=1, num_heads=4)
