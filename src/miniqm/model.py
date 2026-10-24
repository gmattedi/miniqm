"""Karras EDM 3D molecular coordinate Transformer denoiser."""

import dataclasses
import logging
from typing import Final, override

from flax import linen as nn
import jax
import jax.numpy as jnp

from miniqm import batch as batch_lib
from miniqm import geometry
from miniqm import jaxtypes as jt
from miniqm import noise as noise_lib

_LOGGER: Final = logging.getLogger(__name__)

_SPATIAL_DIMS: Final = 3
_MASK_FILL_VALUE: Final = -1e9
# Prevents infinite gradient of sqrt(r^2) at zero interatomic distance.
_DIST_EPS: Final = 1e-8


@dataclasses.dataclass(frozen=True, kw_only=True)
class TransformerConfig:
    """Architecture hyperparameters for the pair-biased Transformer stack."""

    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_factor: int = 4

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0 or self.hidden_dim % 2 != 0:
            raise ValueError(
                f"Expected hidden_dim ({self.hidden_dim}) divisible by "
                f"num_heads ({self.num_heads}) and 2."
            )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FourierNoiseConfig:
    """Sinusoidal Fourier embedding parameters for log-noise conditioning."""

    # Karras et al. (2022) c_noise(sigma) = 0.25 * ln(sigma / sigma_data).
    log_sigma_scale: float = 0.25
    max_period: float = 100.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoiserConfig:
    """Complete configuration for the 3D molecular coordinate Denoiser."""

    transformer: TransformerConfig = dataclasses.field(default_factory=TransformerConfig)
    fourier: FourierNoiseConfig = dataclasses.field(default_factory=FourierNoiseConfig)
    noise: noise_lib.NoiseConfig = dataclasses.field(default_factory=noise_lib.NoiseConfig)


DEFAULT_DENOISER_CONFIG: Final = DenoiserConfig()


@jt.typed
def _fourier_noise_embedding(
    sigma: jt.Float[jt.Array, " batch"],
    *,
    embedding_dim: int,
    sigma_data: float,
    config: FourierNoiseConfig,
) -> jt.Float[jt.Array, "batch {embedding_dim}"]:
    """Embeds scaled log-noise levels using sinusoidal log-spaced frequencies."""
    half_dim = embedding_dim // 2
    c_noise = config.log_sigma_scale * jnp.log(sigma / sigma_data)
    freq_exponents = jnp.linspace(0.0, 1.0, half_dim, dtype=jnp.float32)
    frequencies = jnp.exp(-jnp.log(config.max_period) * freq_exponents)
    # (batch, half_dim) outer product of c_noise (batch, 1) and frequencies (1, half_dim).
    angles = c_noise[:, None] * frequencies[None, :]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


@jt.typed
def _pairwise_geometry_features(
    scaled_positions: jt.Float[jt.Array, "batch num_atoms 3"],
    bond_features: jt.Float[jt.Array, "batch num_atoms num_atoms num_bond_features"],
    atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
) -> jt.Float[jt.Array, "batch num_atoms num_atoms num_pair_features"]:
    """Combines 2D bond features with 3D pairwise displacements and Euclidean distances."""
    # (batch, num_atoms, num_atoms, 1) pairwise valid-atom mask.
    pair_mask = (atom_mask[:, :, None] & atom_mask[:, None, :])[..., None]
    # (batch, num_atoms, num_atoms, 3) pairwise displacement vectors x_i - x_j.
    offsets = scaled_positions[:, :, None, :] - scaled_positions[:, None, :, :]
    # (batch, num_atoms, num_atoms, 1) pairwise Euclidean distances ||x_i - x_j||.
    distances = jnp.sqrt(jnp.sum(jnp.square(offsets), axis=-1, keepdims=True) + _DIST_EPS)
    return jnp.concatenate((bond_features, offsets, distances), axis=-1) * pair_mask


class _TransformerBlock(nn.Module):
    """Pair-biased multi-head self-attention block with adaptive noise conditioning."""

    config: TransformerConfig

    @override
    @jt.typed
    @nn.compact
    def __call__(
        self,
        act: jt.Float[jt.Array, "batch num_atoms hidden_dim"],
        *,
        noise_cond: jt.Float[jt.Array, "batch hidden_dim"],
        pair_bias: jt.Float[jt.Array, "batch num_heads num_atoms num_atoms"],
        atom_mask: jt.Bool[jt.Array, "batch num_atoms"],
    ) -> jt.Float[jt.Array, "batch num_atoms hidden_dim"]:
        """Updates atom activations via pair-biased self-attention and a feed-forward network.

        Args:
            act: Per-atom hidden activations.
            noise_cond: Per-molecule noise conditioning vector.
            pair_bias: Per-head pairwise attention logit bias.
            atom_mask: Boolean mask indicating valid atoms.

        Returns:
            Updated per-atom activations with padded positions zeroed out.
        """
        batch_size, num_atoms, _ = act.shape
        hidden_dim = self.config.hidden_dim
        num_heads = self.config.num_heads
        head_dim = hidden_dim // num_heads
        # (batch, num_atoms, 1) expanded valid-atom mask.
        mask_3d = atom_mask[..., None]

        mod_params = nn.Dense(2 * hidden_dim)(nn.silu(noise_cond))
        # (batch, 1, hidden_dim) per-molecule AdaLN scale and shift broadcast across atoms.
        scale, shift = jnp.split(mod_params[:, None, :], 2, axis=-1)
        norm_act = nn.LayerNorm()(act) * (1.0 + scale) + shift

        qkv = nn.Dense(3 * hidden_dim, use_bias=False)(norm_act)
        # (batch, num_atoms, 3, num_heads, head_dim) unpacked into Q, K, V per head.
        qkv = qkv.reshape(batch_size, num_atoms, 3, num_heads, head_dim)
        queries, keys, values = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )

        # (batch, num_heads, num_atoms, num_atoms) scaled dot-product attention logits.
        attn_logits = jnp.einsum("bihd,bjhd->bhij", queries, keys) / jnp.sqrt(jnp.float32(head_dim))
        attn_logits = attn_logits + pair_bias
        # (batch, 1, 1, num_atoms) valid-key mask broadcast across heads and query atoms.
        key_mask = atom_mask[:, None, None, :]
        attn_logits = jnp.where(key_mask, attn_logits, _MASK_FILL_VALUE)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # (batch, num_atoms, hidden_dim) attention-weighted values merged across heads.
        attn_out = jnp.einsum("bhij,bjhd->bihd", attn_weights, values).reshape(
            batch_size,
            num_atoms,
            hidden_dim,
        )
        act = jnp.asarray((act + nn.Dense(hidden_dim)(attn_out)) * mask_3d)

        mlp_hidden = nn.Dense(hidden_dim * self.config.mlp_factor)(nn.LayerNorm()(act))
        mlp_out = nn.Dense(hidden_dim)(nn.silu(mlp_hidden))
        return jnp.asarray((act + mlp_out) * mask_3d)


class Denoiser(nn.Module):
    """Karras-preconditioned Transformer that predicts clean 3D molecular coordinates."""

    config: DenoiserConfig = DEFAULT_DENOISER_CONFIG

    @override
    @jt.typed
    @nn.compact
    def __call__(
        self,
        noisy_positions: jt.Float[jt.Array, "batch num_atoms 3"],
        sigma: jt.Float[jt.Array, " batch"],
        *,
        batch: batch_lib.JaxBatch,
    ) -> jt.Float[jt.Array, "batch num_atoms 3"]:
        """Predicts clean, zero-centered 3D atomic coordinates from noisy coordinates.

        Args:
            noisy_positions: Gaussian-corrupted atomic coordinates.
            sigma: Noise standard deviation per molecule.
            batch: Padded molecular graph features and valid-atom mask.

        Returns:
            Denoised zero-centered atomic coordinates with padded slots zeroed.
        """
        transformer_config = self.config.transformer
        _LOGGER.info(
            "Tracing Denoiser forward pass (hidden_dim=%d, num_layers=%d, num_heads=%d).",
            transformer_config.hidden_dim,
            transformer_config.num_layers,
            transformer_config.num_heads,
        )

        batch_size, num_atoms, _ = noisy_positions.shape
        # (batch, num_atoms, 1) expanded valid-atom mask.
        mask_3d = batch.atom_mask[..., None]

        factors = noise_lib.preconditioning_factors(sigma, config=self.config.noise)
        # (batch, num_atoms, 3) input coordinates scaled by per-molecule c_in (batch, 1, 1).
        scaled_positions = (factors.c_in[:, None, None] * noisy_positions) * mask_3d

        fourier_sigma = _fourier_noise_embedding(
            sigma,
            embedding_dim=transformer_config.hidden_dim,
            sigma_data=self.config.noise.sigma_data,
            config=self.config.fourier,
        )
        noise_cond = nn.Dense(transformer_config.hidden_dim)(
            nn.silu(nn.Dense(transformer_config.hidden_dim)(fourier_sigma))
        )

        atom_inputs = jnp.concatenate((batch.atom_features, scaled_positions), axis=-1)
        # (batch, num_atoms, hidden_dim) atom embeddings plus broadcast noise_cond (batch, 1, D).
        act = (
            nn.Dense(transformer_config.hidden_dim)(atom_inputs) + noise_cond[:, None, :]
        ) * mask_3d

        pair_inputs = _pairwise_geometry_features(
            scaled_positions,
            batch.bond_features,
            batch.atom_mask,
        )
        pair_hidden = nn.silu(
            nn.Dense(transformer_config.hidden_dim // 2)(nn.LayerNorm()(pair_inputs))
        )
        pair_biases = nn.Dense(transformer_config.num_layers * transformer_config.num_heads)(
            pair_hidden
        )
        # (batch, num_layers, num_heads, num_atoms, num_atoms) per-layer attention logit biases.
        pair_biases = pair_biases.reshape(
            batch_size,
            num_atoms,
            num_atoms,
            transformer_config.num_layers,
            transformer_config.num_heads,
        ).transpose(0, 3, 4, 1, 2)

        for layer_idx in range(transformer_config.num_layers):
            act = _TransformerBlock(config=transformer_config)(
                act,
                noise_cond=noise_cond,
                pair_bias=pair_biases[:, layer_idx],
                atom_mask=batch.atom_mask,
            )

        raw_output = nn.Dense(_SPATIAL_DIMS)(nn.LayerNorm()(act))
        # (batch, num_atoms, 3) Karras skip + output combination with (batch, 1, 1) coefficients.
        denoised_positions = (
            factors.c_skip[:, None, None] * noisy_positions
            + factors.c_out[:, None, None] * raw_output
        )
        return geometry.zero_center_of_mass(denoised_positions, batch.atom_mask)
