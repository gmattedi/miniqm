"""MessagePack serialization and restoration for model and EMA parameters."""

import dataclasses
import enum
import json
import os
import pathlib
from typing import Final, cast

from flax import serialization
from flax import typing as flax_typing
import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm.data import batch as batch_lib
from miniqm.diffusion import model as model_lib

_SPATIAL_DIMS: Final = 3


class _CheckpointKey(enum.StrEnum):
    """Dictionary keys stored inside a MessagePack checkpoint payload."""

    STEP = enum.auto()
    PARAMS = enum.auto()
    EMA_PARAMS = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """Serialized training checkpoint containing step count, active weights, and EMA weights."""

    step: int
    params: flax_typing.VariableDict
    ema_params: flax_typing.VariableDict


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LoadedRun:
    """Restored model configuration, loss/sampling settings, and checkpoint from a run directory."""

    run_dir: pathlib.Path
    model_config: config_lib.DenoiserConfig
    loss_config: config_lib.LossConfig
    sampling_config: config_lib.SamplingConfig
    checkpoint: Checkpoint


def save_checkpoint(path: pathlib.Path, checkpoint_data: Checkpoint) -> None:
    """Serializes step count, active weights, and EMA weights to a MessagePack file.

    Args:
        path: Destination MessagePack file path.
        checkpoint_data: Checkpoint payload to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        _CheckpointKey.STEP: checkpoint_data.step,
        _CheckpointKey.PARAMS: checkpoint_data.params,
        _CheckpointKey.EMA_PARAMS: checkpoint_data.ema_params,
    }
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(serialization.to_bytes(payload))
    tmp_path.replace(path)


def load_checkpoint(path: pathlib.Path, template: Checkpoint) -> Checkpoint:
    """Restores step count, active weights, and EMA weights from a MessagePack file.

    Args:
        path: Source MessagePack file path.
        template: Template checkpoint providing the expected parameter PyTree structure.

    Returns:
        Restored checkpoint with step count, active weights, and EMA weights.
    """
    template_payload = {
        _CheckpointKey.STEP: template.step,
        _CheckpointKey.PARAMS: template.params,
        _CheckpointKey.EMA_PARAMS: template.ema_params,
    }
    restored = cast(
        "dict[str, object]",
        serialization.from_bytes(template_payload, path.read_bytes()),
    )
    return Checkpoint(
        step=cast("int", restored[_CheckpointKey.STEP]),
        params=cast("flax_typing.VariableDict", restored[_CheckpointKey.PARAMS]),
        ema_params=cast("flax_typing.VariableDict", restored[_CheckpointKey.EMA_PARAMS]),
    )


def resolve_run_dir(run_dir: pathlib.Path) -> pathlib.Path:
    """Resolves a run directory or parent experiment directory to the latest timestamped run.

    Args:
        run_dir: Direct run directory or parent directory containing timestamped runs.

    Returns:
        Directory containing the saved run configuration and checkpoints.

    Raises:
        FileNotFoundError: If neither the directory nor any immediate child has a saved config.
    """
    if (run_dir / config_lib.ArtifactName.CONFIG).is_file():
        return run_dir
    if run_dir.is_dir():
        candidates = tuple(
            path.parent for path in run_dir.glob(f"*/{config_lib.ArtifactName.CONFIG}")
        )
        if candidates:
            return max(candidates, key=lambda path: path.name)
    raise FileNotFoundError(
        f"No {config_lib.ArtifactName.CONFIG} found in {run_dir} or its subdirectories."
    )


def load_run_checkpoint(
    run_dir: pathlib.Path,
    *,
    checkpoint_name: str = config_lib.CheckpointName.BEST,
) -> LoadedRun:
    """Restores configurations and checkpoint weights from a training run directory.

    Args:
        run_dir: Training run directory or parent experiment directory containing timestamped runs.
        checkpoint_name: Checkpoint filename inside the checkpoints subdirectory.

    Returns:
        Restored model, loss, and sampling configurations alongside the loaded checkpoint.
    """
    resolved_dir = resolve_run_dir(run_dir)
    saved_config = json.loads(
        (resolved_dir / config_lib.ArtifactName.CONFIG).read_text(encoding="utf-8")
    )
    model_config = config_lib.DenoiserConfig.model_validate(saved_config["train"]["model"])
    loss_config = config_lib.LossConfig.model_validate(saved_config["train"]["loss"])
    sampling_config = config_lib.SamplingConfig.model_validate(saved_config["sampling"])

    dummy_positions = jnp.zeros((1, batch_lib.MAX_ATOMS, _SPATIAL_DIMS))
    dummy_atoms = jnp.zeros((1, batch_lib.MAX_ATOMS, batch_lib.NUM_ATOM_FEATURES))
    dummy_bonds = jnp.zeros(
        (1, batch_lib.MAX_ATOMS, batch_lib.MAX_ATOMS, batch_lib.NUM_BOND_FEATURES),
    )
    dummy_batch = batch_lib.JaxBatch(
        atom_features=dummy_atoms,
        bond_features=dummy_bonds,
        atom_mask=jnp.ones((1, batch_lib.MAX_ATOMS), dtype=jnp.bool_),
        positions=dummy_positions,
        atom_cond_mask=jnp.ones_like(dummy_atoms, dtype=jnp.bool_),
        bond_cond_mask=jnp.ones_like(dummy_bonds, dtype=jnp.bool_),
    )
    dummy_sigma = jnp.ones((1,))
    template_params = model_lib.Denoiser(config=model_config).init(
        jax.random.key(0),
        dummy_positions,
        dummy_atoms,
        dummy_bonds,
        dummy_sigma,
        batch=dummy_batch,
    )
    template = Checkpoint(step=0, params=template_params, ema_params=template_params)
    checkpoint_path = resolved_dir / config_lib.ArtifactName.CHECKPOINTS_DIR / checkpoint_name
    checkpoint_data = load_checkpoint(checkpoint_path, template)
    return LoadedRun(
        run_dir=resolved_dir,
        model_config=model_config,
        loss_config=loss_config,
        sampling_config=sampling_config,
        checkpoint=checkpoint_data,
    )
