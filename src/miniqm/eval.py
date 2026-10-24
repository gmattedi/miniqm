"""Validation loss evaluation and reverse-diffusion conformer RMSD metrics."""

import dataclasses
import functools
from typing import cast

from flax import typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np

from miniqm import batch as batch_lib
from miniqm import data_source, geometry
from miniqm import jaxtypes as jt
from miniqm import losses as losses_lib
from miniqm import model as model_lib
from miniqm import noise as noise_lib
from miniqm import sampling as sampling_lib


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationMetrics:
    """Teacher-forced single-step validation loss breakdown across an evaluation split."""

    total_loss: float
    aligned_mse_loss: float
    distance_loss: float
    rmsd: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SamplingMetrics:
    """Full reverse-diffusion conformer generation RMSD statistics on a validation batch."""

    sampled_rmsd_mean: float
    sampled_rmsd_median: float


@jt.typed
def compute_batch_loss(
    params: flax_typing.VariableDict,
    noisy_sample: noise_lib.NoisySample,
    batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> losses_lib.LossOutput:
    """Runs the denoiser on a corrupted sample and evaluates the diffusion loss."""
    denoiser = model_lib.Denoiser(config=model_config)
    predicted_positions = cast(
        "jt.Float[jt.Array, 'batch num_atoms 3']",
        denoiser.apply(
            params,
            noisy_sample.noisy_positions,
            noisy_sample.sigma,
            batch=batch,
        ),
    )
    return losses_lib.compute_diffusion_loss(
        predicted_positions=predicted_positions,
        target_positions=noisy_sample.target_positions,
        sigma=noisy_sample.sigma,
        atom_mask=batch.atom_mask,
        config=loss_config,
    )


@jt.typed
def eval_step(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> losses_lib.LossOutput:
    """Evaluates the diffusion loss on a batch without updating parameters.

    Args:
        params: Model parameters (typically `state.ema_params`).
        rng: PRNG key for random 3D rotation augmentation and noise sampling.
        batch: Padded JAX batch with ground-truth 3D `positions` and molecular graph features.
        model_config: Denoiser architecture and noise schedule configuration.
        loss_config: Diffusion loss weights and distance cutoff configuration.

    Returns:
        Diffusion loss breakdown on the randomly rotated and corrupted batch.
    """
    noisy_sample = noise_lib.make_noisy_sample(
        rng,
        batch.positions,
        batch.atom_mask,
        config=model_config.noise,
    )
    return compute_batch_loss(
        params,
        noisy_sample,
        batch,
        model_config=model_config,
        loss_config=loss_config,
    )


@jt.typed
def evaluate_loss(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    val_source: data_source.DataSource,
    *,
    batch_size: int,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    loss_config: losses_lib.LossConfig = losses_lib.DEFAULT_LOSS_CONFIG,
) -> ValidationMetrics:
    """Computes mean teacher-forced diffusion loss and RMSD across validation batches.

    Args:
        params: Model parameters to evaluate (typically `state.ema_params`).
        rng: PRNG key for validation rotation augmentation and noise sampling.
        val_source: Validation `DataSource`.
        batch_size: Validation batch size.
        model_config: Denoiser architecture and noise schedule configuration.
        loss_config: Diffusion loss weights and distance cutoff configuration.

    Returns:
        Mean validation loss components across all batches.
    """
    jit_eval = jax.jit(
        functools.partial(eval_step, model_config=model_config, loss_config=loss_config)
    )
    totals: list[float] = []
    mses: list[float] = []
    dists: list[float] = []
    rmsds: list[float] = []

    step_rng = rng
    for host_batch in val_source.iter_batches(
        batch_size=batch_size,
        drop_last=False,
    ):
        step_rng, batch_rng = jax.random.split(step_rng)
        loss_out = cast("losses_lib.LossOutput", jit_eval(params, batch_rng, host_batch.to_jax()))
        totals.append(float(loss_out.total_loss))
        mses.append(float(loss_out.aligned_mse_loss))
        dists.append(float(loss_out.distance_loss))
        rmsds.append(float(loss_out.rmsd))

    return ValidationMetrics(
        total_loss=float(np.mean(totals)),
        aligned_mse_loss=float(np.mean(mses)),
        distance_loss=float(np.mean(dists)),
        rmsd=float(np.mean(rmsds)),
    )


@jt.typed
def evaluate_sampling_rmsd(
    params: flax_typing.VariableDict,
    rng: jt.Array,
    val_batch: batch_lib.JaxBatch,
    *,
    model_config: model_lib.DenoiserConfig = model_lib.DEFAULT_DENOISER_CONFIG,
    sampling_config: sampling_lib.SamplingConfig = sampling_lib.DEFAULT_SAMPLING_CONFIG,
) -> SamplingMetrics:
    """Generates 3D conformers from Gaussian noise and computes Kabsch-aligned RMSD.

    Args:
        params: Model parameters to evaluate (typically `state.ema_params`).
        rng: PRNG key for reverse-diffusion sampling.
        val_batch: Padded JAX validation batch containing target `positions`.
        model_config: Denoiser architecture configuration.
        sampling_config: Reverse Euler integration configuration.

    Returns:
        Mean and median Kabsch-aligned RMSD across the sampled validation batch.
    """
    denoiser = model_lib.Denoiser(config=model_config)
    generated = sampling_lib.sample_coordinates(
        rng,
        denoiser,
        params,
        batch=val_batch,
        config=sampling_config,
    )
    aligned = geometry.kabsch_align(
        reference=val_batch.positions,
        target=generated,
        atom_mask=val_batch.atom_mask,
    )
    # (batch,) valid atom counts per molecule, clamped to >= 1.
    atom_counts = jnp.maximum(jnp.sum(val_batch.atom_mask, axis=-1), 1)
    # (batch,) per-molecule Kabsch-aligned coordinate RMSD.
    rmsd_per_mol = jnp.sqrt(
        jnp.sum(jnp.square(aligned - val_batch.positions), axis=(-1, -2)) / atom_counts
    )
    return SamplingMetrics(
        sampled_rmsd_mean=float(jnp.mean(rmsd_per_mol)),
        sampled_rmsd_median=float(jnp.median(rmsd_per_mol)),
    )
