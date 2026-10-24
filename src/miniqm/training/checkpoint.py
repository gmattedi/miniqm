"""MessagePack serialization and restoration for model and EMA parameters."""

import dataclasses
import json
import pathlib
from typing import Final, cast

from flax import serialization
from flax import typing as flax_typing
import jax
import jax.numpy as jnp

from miniqm.data import batch as batch_lib
from miniqm.diffusion import losses as losses_lib
from miniqm.diffusion import model as model_lib
from miniqm.diffusion import noise as noise_lib
from miniqm.diffusion import sampling as sampling_lib

_SPATIAL_DIMS: Final = 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """Serialized training checkpoint containing step count, active weights, and EMA weights."""

    step: int
    params: flax_typing.VariableDict
    ema_params: flax_typing.VariableDict


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LoadedRun:
    """Restored model configuration, loss/sampling settings, and checkpoint from a run directory."""

    model_config: model_lib.DenoiserConfig
    loss_config: losses_lib.LossConfig
    sampling_config: sampling_lib.SamplingConfig
    checkpoint: Checkpoint


def save_checkpoint(path: pathlib.Path, ckpt: Checkpoint) -> None:
    """Serializes step count, active weights, and EMA weights to a MessagePack file.

    Args:
        path: Destination MessagePack file path.
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
    """Restores step count, active weights, and EMA weights from a MessagePack file.

    Args:
        path: Source MessagePack file path.
        template: Template checkpoint providing the expected parameter PyTree structure.

    Returns:
        Restored checkpoint with step count, active weights, and EMA weights.
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


def load_run_checkpoint(
    run_dir: pathlib.Path,
    *,
    checkpoint_name: str = "best.msgpack",
) -> LoadedRun:
    """Restores configurations and checkpoint weights from a training run directory.

    Args:
        run_dir: Training run directory containing saved configuration and checkpoints.
        checkpoint_name: Checkpoint filename inside the checkpoints subdirectory.

    Returns:
        Restored model, loss, and sampling configurations alongside the loaded checkpoint.
    """
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_raw = raw["train"]["model"]
    model_config = model_lib.DenoiserConfig(
        fourier=model_lib.FourierNoiseConfig(**model_raw["fourier"]),
        transformer=model_lib.TransformerConfig(**model_raw["transformer"]),
        noise=noise_lib.NoiseConfig(**model_raw["noise"]),
    )
    loss_raw = raw["train"]["loss"]
    loss_config = losses_lib.LossConfig(
        noise=noise_lib.NoiseConfig(**loss_raw["noise"]),
        distance_loss_weight=float(loss_raw["distance_loss_weight"]),
        distance_cutoff=float(loss_raw["distance_cutoff"]),
    )
    sampling_raw = raw["sampling"]
    sampling_config = sampling_lib.SamplingConfig(
        noise=noise_lib.NoiseConfig(**sampling_raw["noise"]),
        num_steps=int(sampling_raw["num_steps"]),
        gamma_0=float(sampling_raw["gamma_0"]),
        gamma_min=float(sampling_raw["gamma_min"]),
        noise_scale=float(sampling_raw["noise_scale"]),
        step_scale=float(sampling_raw["step_scale"]),
    )

    dummy_batch = batch_lib.JaxBatch(
        atom_features=jnp.zeros(
            (1, batch_lib.MAX_ATOMS, batch_lib.NUM_ATOM_FEATURES),
            dtype=jnp.float32,
        ),
        bond_features=jnp.zeros(
            (1, batch_lib.MAX_ATOMS, batch_lib.MAX_ATOMS, batch_lib.NUM_BOND_FEATURES),
            dtype=jnp.float32,
        ),
        atom_mask=jnp.ones((1, batch_lib.MAX_ATOMS), dtype=jnp.bool_),
    )
    dummy_positions = jnp.zeros((1, batch_lib.MAX_ATOMS, _SPATIAL_DIMS), dtype=jnp.float32)
    dummy_sigma = jnp.ones((1,), dtype=jnp.float32)
    template_params = model_lib.Denoiser(config=model_config).init(
        jax.random.key(0),
        dummy_positions,
        dummy_sigma,
        batch=dummy_batch,
    )
    template = Checkpoint(step=0, params=template_params, ema_params=template_params)
    ckpt = load_checkpoint(run_dir / "checkpoints" / checkpoint_name, template)
    return LoadedRun(
        model_config=model_config,
        loss_config=loss_config,
        sampling_config=sampling_config,
        checkpoint=ckpt,
    )
