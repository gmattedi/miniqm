from typing import cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import geometry, model
from miniqm import jaxtypes as jt

_TEST_CONFIG = model.DenoiserConfig(
    transformer=model.TransformerConfig(hidden_dim=32, num_layers=2, num_heads=4),
)


@jt.typed
def _make_dummy_inputs() -> tuple[
    jt.Float[jt.Array, "batch num_atoms 3"],
    jt.Float[jt.Array, " batch"],
    jt.Float[jt.Array, "batch num_atoms 22"],
    jt.Float[jt.Array, "batch num_atoms num_atoms 6"],
    jt.Bool[jt.Array, "batch num_atoms"],
]:
    """Constructs a 2-molecule synthetic batch with padded atom slots for model tests."""
    key = jax.random.key(0)
    pos_key, atom_key, bond_key = jax.random.split(key, 3)
    atom_mask = jnp.array(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=jnp.bool_,
    )
    noisy_positions = (
        jax.random.normal(pos_key, (2, 5, 3), dtype=jnp.float32) * atom_mask[..., None]
    )
    sigma = jnp.array([0.5, 2.0], dtype=jnp.float32)
    atom_features = jax.random.normal(atom_key, (2, 5, 22), dtype=jnp.float32)
    bond_features = jax.random.normal(bond_key, (2, 5, 5, 6), dtype=jnp.float32)
    return noisy_positions, sigma, atom_features, bond_features, atom_mask


def test_denoiser_outputs_zero_centered_and_masked_coordinates():
    noisy_positions, sigma, atom_features, bond_features, atom_mask = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(1),
        noisy_positions,
        sigma,
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
    )

    predicted = cast(
        "jt.Array",
        denoiser.apply(
            params,
            noisy_positions,
            sigma,
            atom_features=atom_features,
            bond_features=bond_features,
            atom_mask=atom_mask,
        ),
    )

    assert predicted.shape == noisy_positions.shape
    np.testing.assert_allclose(jnp.sum(predicted, axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(predicted * (~atom_mask)[..., None], 0.0, atol=1e-7)


def test_denoiser_is_invariant_to_padded_slot_perturbations():
    noisy_positions, sigma, atom_features, bond_features, atom_mask = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(2),
        noisy_positions,
        sigma,
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
    )

    pad_mask = ~atom_mask
    perturbed_positions = noisy_positions + 50.0 * pad_mask[..., None]
    perturbed_atoms = atom_features + 10.0 * pad_mask[..., None]

    clean_out = cast(
        "jt.Array",
        denoiser.apply(
            params,
            noisy_positions,
            sigma,
            atom_features=atom_features,
            bond_features=bond_features,
            atom_mask=atom_mask,
        ),
    )
    perturbed_out = cast(
        "jt.Array",
        denoiser.apply(
            params,
            perturbed_positions,
            sigma,
            atom_features=perturbed_atoms,
            bond_features=bond_features,
            atom_mask=atom_mask,
        ),
    )

    np.testing.assert_allclose(clean_out, perturbed_out, rtol=1e-5, atol=1e-5)


def test_denoiser_recovers_centered_input_at_vanishing_noise():
    noisy_positions, _, atom_features, bond_features, atom_mask = _make_dummy_inputs()
    tiny_sigma = jnp.array([1e-7, 1e-7], dtype=jnp.float32)
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(3),
        noisy_positions,
        tiny_sigma,
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
    )

    predicted = cast(
        "jt.Array",
        denoiser.apply(
            params,
            noisy_positions,
            tiny_sigma,
            atom_features=atom_features,
            bond_features=bond_features,
            atom_mask=atom_mask,
        ),
    )
    expected = geometry.zero_center_of_mass(noisy_positions, atom_mask)

    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-5)


def test_denoiser_gradients_are_finite_and_non_zero():
    noisy_positions, sigma, atom_features, bond_features, atom_mask = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(4),
        noisy_positions,
        sigma,
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
    )

    def loss_fn(model_params: flax_typing.VariableDict) -> jt.Array:
        pred = cast(
            "jt.Array",
            denoiser.apply(
                model_params,
                noisy_positions,
                sigma,
                atom_features=atom_features,
                bond_features=bond_features,
                atom_mask=atom_mask,
            ),
        )
        return jnp.mean(jnp.square(pred))

    grads = jax.grad(loss_fn)(params)
    grad_leaves = jax.tree.leaves(grads)

    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves)
    assert any(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in grad_leaves)


def test_transformer_config_rejects_incompatible_hidden_dim():
    with pytest.raises(ValueError, match="Expected hidden_dim"):
        model.TransformerConfig(hidden_dim=30, num_layers=1, num_heads=4)
