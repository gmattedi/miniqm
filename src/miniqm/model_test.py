import dataclasses
from typing import Final, cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import batch as batch_lib
from miniqm import geometry, model
from miniqm import jaxtypes as jt

_TEST_CONFIG: Final = model.DenoiserConfig(
    transformer=model.TransformerConfig(hidden_dim=32, num_layers=2, num_heads=4),
)


@jt.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class _ModelTestInputs:
    """Synthetic 2-molecule padded batch for Denoiser tests."""

    noisy_positions: jt.Float[jt.Array, "batch num_atoms 3"]
    sigma: jt.Float[jt.Array, " batch"]
    batch: batch_lib.JaxBatch


@jt.typed
def _make_dummy_inputs() -> _ModelTestInputs:
    """Constructs a 2-molecule synthetic batch with padded atom slots for model tests."""
    pos_key, atom_key, bond_key = jax.random.split(jax.random.key(0), 3)
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
    graph_batch = batch_lib.JaxBatch(
        atom_features=jax.random.normal(atom_key, (2, 5, 22), dtype=jnp.float32),
        bond_features=jax.random.normal(bond_key, (2, 5, 5, 6), dtype=jnp.float32),
        atom_mask=atom_mask,
    )
    return _ModelTestInputs(
        noisy_positions=noisy_positions,
        sigma=sigma,
        batch=graph_batch,
    )


def test_denoiser_outputs_zero_centered_and_masked_coordinates() -> None:
    inputs = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(1),
        inputs.noisy_positions,
        inputs.sigma,
        batch=inputs.batch,
    )

    predicted_positions = cast(
        "jt.Array",
        denoiser.apply(
            params,
            inputs.noisy_positions,
            inputs.sigma,
            batch=inputs.batch,
        ),
    )

    assert predicted_positions.shape == inputs.noisy_positions.shape
    np.testing.assert_allclose(jnp.sum(predicted_positions, axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        predicted_positions * (~inputs.batch.atom_mask)[..., None],
        0.0,
        atol=1e-7,
    )


def test_denoiser_is_invariant_to_padded_slot_perturbations() -> None:
    inputs = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(2),
        inputs.noisy_positions,
        inputs.sigma,
        batch=inputs.batch,
    )

    pad_mask = ~inputs.batch.atom_mask
    perturbed_positions = inputs.noisy_positions + 50.0 * pad_mask[..., None]
    perturbed_batch = dataclasses.replace(
        inputs.batch,
        atom_features=inputs.batch.atom_features + 10.0 * pad_mask[..., None],
    )

    clean_out = cast(
        "jt.Array",
        denoiser.apply(
            params,
            inputs.noisy_positions,
            inputs.sigma,
            batch=inputs.batch,
        ),
    )
    perturbed_out = cast(
        "jt.Array",
        denoiser.apply(
            params,
            perturbed_positions,
            inputs.sigma,
            batch=perturbed_batch,
        ),
    )

    np.testing.assert_allclose(clean_out, perturbed_out, rtol=1e-5, atol=1e-5)


def test_denoiser_recovers_centered_input_at_vanishing_noise() -> None:
    inputs = _make_dummy_inputs()
    tiny_sigma = jnp.array([1e-7, 1e-7], dtype=jnp.float32)
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(3),
        inputs.noisy_positions,
        tiny_sigma,
        batch=inputs.batch,
    )

    predicted_positions = cast(
        "jt.Array",
        denoiser.apply(
            params,
            inputs.noisy_positions,
            tiny_sigma,
            batch=inputs.batch,
        ),
    )
    expected_positions = geometry.zero_center_of_mass(
        inputs.noisy_positions,
        inputs.batch.atom_mask,
    )

    np.testing.assert_allclose(predicted_positions, expected_positions, rtol=1e-5, atol=1e-5)


def test_denoiser_gradients_are_finite_and_non_zero() -> None:
    inputs = _make_dummy_inputs()
    denoiser = model.Denoiser(config=_TEST_CONFIG)
    params = denoiser.init(
        jax.random.key(4),
        inputs.noisy_positions,
        inputs.sigma,
        batch=inputs.batch,
    )

    def loss_fn(model_params: flax_typing.VariableDict) -> jt.Array:
        """Computes mean squared predicted coordinate magnitude for gradient verification."""
        predicted_positions = cast(
            "jt.Array",
            denoiser.apply(
                model_params,
                inputs.noisy_positions,
                inputs.sigma,
                batch=inputs.batch,
            ),
        )
        return jnp.mean(jnp.square(predicted_positions))

    grads = jax.grad(loss_fn)(params)
    grad_leaves = jax.tree.leaves(grads)

    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves)
    assert any(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in grad_leaves)


def test_transformer_config_rejects_incompatible_hidden_dim() -> None:
    with pytest.raises(ValueError, match="Expected hidden_dim"):
        model.TransformerConfig(hidden_dim=30, num_layers=1, num_heads=4)
