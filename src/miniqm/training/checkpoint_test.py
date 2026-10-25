import pathlib
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from miniqm import config as config_lib
from miniqm.data import batch as batch_lib
from miniqm.training import checkpoint, train

_TINY_TRAIN_CONFIG: Final = config_lib.TrainConfig(
    model=config_lib.DenoiserConfig(
        transformer=config_lib.TransformerConfig(hidden_dim=16, num_layers=1, num_heads=2),
    ),
    ema_decay=0.8,
)


def _make_dummy_batch() -> batch_lib.JaxBatch:
    return batch_lib.JaxBatch(
        atom_features=jnp.ones((2, 4, 22), dtype=jnp.float32),
        bond_features=jnp.ones((2, 4, 4, 6), dtype=jnp.float32),
        atom_mask=jnp.array([[True, True, False, False], [True, True, True, False]]),
        positions=jnp.zeros((2, 4, 3), dtype=jnp.float32),
    )


def test_save_and_load_checkpoint_round_trip(tmp_path: pathlib.Path):
    dummy_batch = _make_dummy_batch()
    initial_state = train.init_train_state(
        jax.random.key(0),
        dummy_batch,
        config=_TINY_TRAIN_CONFIG,
    )
    stepped_state = train.train_step(
        initial_state,
        dummy_batch,
        config=_TINY_TRAIN_CONFIG,
    ).state

    ckpt_path = tmp_path / "checkpoints" / "model.msgpack"
    checkpoint.save_checkpoint(
        ckpt_path,
        checkpoint.Checkpoint(
            step=int(stepped_state.step),
            params=stepped_state.params,
            ema_params=stepped_state.ema_params,
        ),
    )

    template_state = train.init_train_state(
        jax.random.key(99),
        dummy_batch,
        config=_TINY_TRAIN_CONFIG,
    )
    restored = checkpoint.load_checkpoint(
        ckpt_path,
        checkpoint.Checkpoint(
            step=int(template_state.step),
            params=template_state.params,
            ema_params=template_state.ema_params,
        ),
    )

    assert restored.step == 1
    for expected, actual in zip(
        jax.tree.leaves(stepped_state.params),
        jax.tree.leaves(restored.params),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    for expected, actual in zip(
        jax.tree.leaves(stepped_state.ema_params),
        jax.tree.leaves(restored.ema_params),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_load_run_checkpoint_restores_configs_and_weights(tmp_path: pathlib.Path):
    dummy_batch = _make_dummy_batch()
    state = train.init_train_state(
        jax.random.key(3),
        dummy_batch,
        config=_TINY_TRAIN_CONFIG,
    )
    parent_dir = tmp_path / "experiment"
    run_dir = parent_dir / "20261024_220000"
    run_config = config_lib.RunConfig(run_dir=run_dir, train=_TINY_TRAIN_CONFIG)
    train.write_run_config(run_config)
    checkpoint.save_checkpoint(
        run_dir / "checkpoints" / config_lib.CheckpointName.BEST,
        checkpoint.Checkpoint(step=7, params=state.params, ema_params=state.ema_params),
    )

    loaded = checkpoint.load_run_checkpoint(parent_dir)

    assert loaded.run_dir == run_dir
    assert loaded.model_config == _TINY_TRAIN_CONFIG.model
    assert loaded.checkpoint.step == 7
    for expected, actual in zip(
        jax.tree.leaves(state.ema_params),
        jax.tree.leaves(loaded.checkpoint.ema_params),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_resolve_run_dir_picks_latest_timestamp_and_errors_when_missing(
    tmp_path: pathlib.Path,
):
    parent_dir = tmp_path / "experiment"
    older = parent_dir / "20261024_100000"
    newer = parent_dir / "20261024_220000"
    for run_dir in (older, newer):
        train.write_run_config(config_lib.RunConfig(run_dir=run_dir, train=_TINY_TRAIN_CONFIG))

    assert checkpoint.resolve_run_dir(parent_dir) == newer

    with pytest.raises(FileNotFoundError, match=r"No config\.json found"):
        checkpoint.resolve_run_dir(tmp_path / "empty")
