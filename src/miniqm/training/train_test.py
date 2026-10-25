from collections.abc import Mapping
import functools
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry
from miniqm.training import eval as eval_lib
from miniqm.training import train

_TEST_TRAIN_CONFIG: Final = config_lib.TrainConfig(
    model=config_lib.DenoiserConfig(
        transformer=config_lib.TransformerConfig(hidden_dim=32, num_layers=2, num_heads=4),
        noise=config_lib.NoiseConfig(log_sigma_mean=0.5, log_sigma_std=0.05),
    ),
    learning_rate=3e-3,
    warmup_steps=2,
    decay_steps=50,
    min_learning_rate=1e-4,
    ema_decay=0.9,
)


@jt.typed
def _make_synthetic_batch() -> batch_lib.JaxBatch:
    pos_key, atom_key, bond_key = jax.random.split(jax.random.key(0), 3)
    atom_mask = jnp.array(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=jnp.bool_,
    )
    raw_positions = jax.random.normal(pos_key, (2, 5, 3), dtype=jnp.float32)
    positions = geometry.zero_center_of_mass(raw_positions, atom_mask)
    return batch_lib.JaxBatch(
        atom_features=jax.random.normal(atom_key, (2, 5, 22), dtype=jnp.float32),
        bond_features=jax.random.normal(bond_key, (2, 5, 5, 6), dtype=jnp.float32),
        atom_mask=atom_mask,
        positions=positions,
    )


def test_init_train_state_creates_matching_params_and_ema():
    synthetic_batch = _make_synthetic_batch()
    state = train.init_train_state(
        jax.random.key(1),
        synthetic_batch,
        config=_TEST_TRAIN_CONFIG,
    )

    assert int(state.step) == 0
    param_leaves = jax.tree.leaves(state.params)
    ema_leaves = jax.tree.leaves(state.ema_params)
    assert len(param_leaves) == len(ema_leaves)
    for param_leaf, ema_leaf in zip(param_leaves, ema_leaves, strict=True):
        np.testing.assert_array_equal(param_leaf, ema_leaf)


def test_parameter_summary_and_tabulate_report_named_submodules():
    synthetic_batch = _make_synthetic_batch()
    state = train.init_train_state(
        jax.random.key(1),
        synthetic_batch,
        config=_TEST_TRAIN_CONFIG,
    )
    total_params = train.count_parameters(state.params)
    summary = train.format_parameter_summary(state.params)
    table = train.tabulate_model(synthetic_batch, config=_TEST_TRAIN_CONFIG.model)

    assert total_params > 0
    assert f"total: {total_params:,}" in summary
    assert "atom_proj:" in summary
    assert "block_0:" in summary
    assert "coord_head:" in summary
    assert "Denoiser Summary" in table
    assert "block_0" in table
    assert "coord_head" in table


def test_train_step_updates_parameters_and_ema_weights():
    synthetic_batch = _make_synthetic_batch()
    initial_state = train.init_train_state(
        jax.random.key(2),
        synthetic_batch,
        config=_TEST_TRAIN_CONFIG,
    )
    jit_train_step = jax.jit(functools.partial(train.train_step, config=_TEST_TRAIN_CONFIG))

    step_output = jit_train_step(initial_state, synthetic_batch)
    next_state = step_output.state

    assert int(next_state.step) == 1
    assert bool(jnp.isfinite(step_output.loss_output.total_loss))
    assert float(step_output.grad_norm) > 0.0
    assert bool(jnp.isfinite(step_output.grad_norm))
    assert float(step_output.learning_rate) == pytest.approx(0.0)

    # First step at warmup step 0 has lr=0.0; second step at step 1 has positive lr.
    second_output = jit_train_step(next_state, synthetic_batch)
    updated_state = second_output.state
    assert int(updated_state.step) == 2
    assert float(second_output.learning_rate) == pytest.approx(1.5e-3)

    init_leaves = jax.tree.leaves(initial_state.params)
    new_leaves = jax.tree.leaves(updated_state.params)
    ema_leaves = jax.tree.leaves(updated_state.ema_params)
    assert any(
        not np.allclose(init_leaf, new_leaf)
        for init_leaf, new_leaf in zip(init_leaves, new_leaves, strict=True)
    )
    for init_leaf, new_leaf, ema_leaf in zip(init_leaves, new_leaves, ema_leaves, strict=True):
        expected_ema = (
            _TEST_TRAIN_CONFIG.ema_decay * init_leaf
            + (1.0 - _TEST_TRAIN_CONFIG.ema_decay) * new_leaf
        )
        np.testing.assert_allclose(ema_leaf, expected_ema, rtol=1e-5, atol=1e-6)


def test_eval_step_matches_forward_loss_of_train_step():
    synthetic_batch = _make_synthetic_batch()
    state = train.init_train_state(
        jax.random.key(3),
        synthetic_batch,
        config=_TEST_TRAIN_CONFIG,
    )
    step_rng, _ = jax.random.split(state.rng)

    train_out = train.train_step(state, synthetic_batch, config=_TEST_TRAIN_CONFIG)
    eval_out = jax.jit(
        functools.partial(
            eval_lib.eval_step,
            model_config=_TEST_TRAIN_CONFIG.model,
            loss_config=_TEST_TRAIN_CONFIG.loss,
        )
    )(
        state.params,
        step_rng,
        synthetic_batch,
    )

    np.testing.assert_allclose(
        train_out.loss_output.total_loss,
        eval_out.total_loss,
        rtol=1e-5,
        atol=1e-5,
    )


def test_short_training_loop_reduces_diffusion_loss_on_single_batch():
    synthetic_batch = _make_synthetic_batch()
    state = train.init_train_state(
        jax.random.key(4),
        synthetic_batch,
        config=_TEST_TRAIN_CONFIG,
    )
    jit_train_step = jax.jit(functools.partial(train.train_step, config=_TEST_TRAIN_CONFIG))
    jit_eval_step = jax.jit(
        functools.partial(
            eval_lib.eval_step,
            model_config=_TEST_TRAIN_CONFIG.model,
            loss_config=_TEST_TRAIN_CONFIG.loss,
        )
    )

    eval_rng = jax.random.key(99)
    initial_eval = jit_eval_step(state.params, eval_rng, synthetic_batch)
    for _ in range(25):
        state = jit_train_step(state, synthetic_batch).state
    final_eval = jit_eval_step(state.params, eval_rng, synthetic_batch)

    assert float(final_eval.total_loss) < 0.5 * float(initial_eval.total_loss)


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    (
        ({"learning_rate": 0.0}, "greater than 0"),
        ({"min_learning_rate": 1e-2, "learning_rate": 1e-3}, "min_learning_rate <= learning_rate"),
        ({"warmup_steps": -1}, "greater than or equal to 0"),
        ({"warmup_steps": 100, "decay_steps": 100}, "decay_steps"),
        ({"weight_decay": -0.1}, "greater than or equal to 0"),
        ({"max_grad_norm": 0.0}, "greater than 0"),
        ({"ema_decay": 1.5}, "less than or equal to 1"),
    ),
    ids=(
        "zero_lr",
        "min_lr_exceeds_lr",
        "negative_warmup",
        "decay_not_greater_than_warmup",
        "negative_weight_decay",
        "zero_grad_norm",
        "ema_decay_above_one",
    ),
)
def test_train_config_rejects_invalid_hyperparameters(
    kwargs: Mapping[str, float],
    error_match: str,
):
    with pytest.raises(ValueError, match=error_match):
        config_lib.TrainConfig(
            learning_rate=float(kwargs.get("learning_rate", 3e-4)),
            min_learning_rate=float(kwargs.get("min_learning_rate", 1e-5)),
            warmup_steps=int(kwargs.get("warmup_steps", 1_000)),
            decay_steps=int(kwargs.get("decay_steps", 100_000)),
            weight_decay=float(kwargs.get("weight_decay", 1e-4)),
            max_grad_norm=float(kwargs.get("max_grad_norm", 1.0)),
            ema_decay=float(kwargs.get("ema_decay", 0.999)),
        )


def test_run_config_rejects_invalid_num_devices():
    with pytest.raises(ValueError, match="greater than 0"):
        config_lib.RunConfig(num_devices=0)
