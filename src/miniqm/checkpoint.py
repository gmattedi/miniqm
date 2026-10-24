"""MessagePack serialization and restoration for model and EMA parameters."""

import dataclasses
import pathlib
from typing import cast

from flax import serialization
from flax import typing as flax_typing


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """Serialized training checkpoint containing step count, active weights, and EMA weights."""

    step: int
    params: flax_typing.VariableDict
    ema_params: flax_typing.VariableDict


def save_checkpoint(path: pathlib.Path, ckpt: Checkpoint) -> None:
    """Serializes step, active `params`, and `ema_params` to a MessagePack file.

    Args:
        path: Destination `.msgpack` file path.
        ckpt: Checkpoint payload to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": ckpt.step,
        "params": ckpt.params,
        "ema_params": ckpt.ema_params,
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_bytes(serialization.to_bytes(payload))
    tmp_path.replace(path)


def load_checkpoint(path: pathlib.Path, template: Checkpoint) -> Checkpoint:
    """Restores step, `params`, and `ema_params` from a MessagePack file.

    Args:
        path: Source `.msgpack` file path.
        template: Template `Checkpoint` providing the expected parameter PyTree structure.

    Returns:
        Restored `Checkpoint` with `step`, `params`, and `ema_params`.
    """
    template_dict = {
        "step": template.step,
        "params": template.params,
        "ema_params": template.ema_params,
    }
    restored = cast(
        "dict[str, object]",
        serialization.from_bytes(template_dict, path.read_bytes()),
    )
    return Checkpoint(
        step=cast("int", restored["step"]),
        params=cast("flax_typing.VariableDict", restored["params"]),
        ema_params=cast("flax_typing.VariableDict", restored["ema_params"]),
    )
