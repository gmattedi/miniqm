"""Karras EDM joint 2D+3D molecular Transformer denoiser."""

import dataclasses
import logging
from typing import Final, override

from flax import linen as nn
import jax
import jax.numpy as jnp

from miniqm import config as config_lib
from miniqm import jaxtypes as jt
from miniqm.data import batch as batch_lib
from miniqm.diffusion import geometry
from miniqm.diffusion import noise as noise_lib

_LOGGER: Final = logging.getLogger(__name__)

_SPATIAL_DIMS: Final = 3
_NUM_RBF: Final = 16
# Covalent and non-bonded heavy-atom distance range in Angstroms for radial basis expansion.
_RBF_MIN_DIST: Final = 0.8
_RBF_MAX_DIST: Final = 3.2
_NUM_GEOM_FEATURES: Final = _SPATIAL_DIMS + 1 + _NUM_RBF
_MASK_FILL_VALUE: Final = -1e9
# Prevents infinite gradient of sqrt(r^2) at zero interatomic distance.
_DIST_EPS: Final = 1e-8
_HALF_WEIGHT: Final = 0.5


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DenoiserOutput:
    """Jointly denoised 3D atomic coordinates and 2D one-hot atom/bond features."""

    positions: jt.Float[jt.Array, "batch num_atoms 3"]
    atom_features: jt.Float[jt.Array, "batch num_atoms num_atom_features"]
    bond_features: jt.Float[jt.Array, "batch num_atoms num_atoms num_bond_features"]


class _TransformerBlock(nn.Module):
    """Pair-biased multi-head self-attention block with adaptive noise conditioning."""

    config: config_lib.TransformerConfig

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
            atom_mask: Valid-atom mask.

        Returns:
            Updated per-atom activations with padded positions zeroed out.
        """
        batch_size, num_atoms, _ = act.shape
        hidden_dim = self.config.hidden_dim
        num_heads = self.config.num_heads
        head_dim = hidden_dim // num_heads
        # (batch, num_atoms, 1) expanded valid-atom mask.
        mask_3d = atom_mask[..., None]

        mod_params = nn.Dense(2 * hidden_dim, name="adaln_proj")(nn.silu(noise_cond))
        # (batch, 1, hidden_dim) per-molecule AdaLN scale and shift broadcast across atoms.
        scale, shift = jnp.split(mod_params[:, None, :], 2, axis=-1)
        norm_act = nn.LayerNorm(name="attn_norm")(act) * (1.0 + scale) + shift

        qkv = nn.Dense(3 * hidden_dim, use_bias=False, name="qkv_proj")(norm_act)
        # (batch, num_atoms, 3, num_heads, head_dim) unpacked into Q, K, V per head.
        qkv = qkv.reshape(batch_size, num_atoms, 3, num_heads, head_dim)
        queries, keys, values = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )

        # (batch, num_heads, num_atoms, num_atoms) scaled dot-product attention logits.
        attn_logits = jnp.einsum("bihd,bjhd->bhij", queries, keys) / jnp.sqrt(head_dim)
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
        act = jnp.asarray((act + nn.Dense(hidden_dim, name="out_proj")(attn_out)) * mask_3d)

        mlp_hidden = nn.Dense(hidden_dim * self.config.mlp_factor, name="mlp_fc1")(
            nn.LayerNorm(name="mlp_norm")(act)
        )
        mlp_out = nn.Dense(hidden_dim, name="mlp_fc2")(nn.silu(mlp_hidden))
        return jnp.asarray((act + mlp_out) * mask_3d)


class Denoiser(nn.Module):
    """Karras-preconditioned Transformer that jointly denoises 3D coordinates and 2D graphs."""

    config: config_lib.DenoiserConfig = config_lib.DEFAULT_DENOISER_CONFIG

    @override
    @jt.typed
    @nn.compact
    def __call__(
        self,
        noisy_positions: jt.Float[jt.Array, "batch num_atoms 3"],
        noisy_atom_features: jt.Float[jt.Array, "batch num_atoms num_atom_features"],
        noisy_bond_features: jt.Float[jt.Array, "batch num_atoms num_atoms num_bond_features"],
        sigma: jt.Float[jt.Array, " batch"],
        *,
        batch: batch_lib.JaxBatch,
    ) -> DenoiserOutput:
        """Predicts clean 3D coordinates and 2D one-hot atom/bond features from noisy inputs.

        Args:
            noisy_positions: Gaussian-corrupted atomic coordinates.
            noisy_atom_features: Gaussian-corrupted one-hot atom features.
            noisy_bond_features: Gaussian-corrupted symmetric one-hot bond features.
            sigma: Noise standard deviation per molecule.
            batch: Padded molecular graph conditioning features and masks.

        Returns:
            Denoised zero-centered 3D coordinates and 2D atom/bond features.
        """
        transformer_config = self.config.transformer
        _LOGGER.info(
            "Tracing Denoiser forward pass (hidden_dim=%d, num_layers=%d, num_heads=%d).",
            transformer_config.hidden_dim,
            transformer_config.num_layers,
            transformer_config.num_heads,
        )

        batch_size, num_atoms, _ = noisy_positions.shape
        pair_dim = transformer_config.hidden_dim // 2
        # (batch, num_atoms, 1) expanded valid-atom mask.
        mask_3d = batch.atom_mask[..., None]
        # (batch, num_atoms, num_atoms, 1) expanded distinct valid-pair mask (i != j).
        pair_mask = noise_lib.off_diagonal_pair_mask(batch.atom_mask)

        factors = noise_lib.preconditioning_factors(sigma, config=self.config.noise)
        # (batch, 1, 1) and (batch, 1, 1, 1) broadcastable input scaling factors.
        c_in_atom = factors.c_in[:, None, None]
        c_in_pair = factors.c_in[:, None, None, None]

        scaled_positions = (c_in_atom * noisy_positions) * mask_3d
        scaled_atoms = (c_in_atom * noisy_atom_features) * mask_3d
        scaled_bonds = (c_in_pair * noisy_bond_features) * pair_mask

        cond_atom_mask = batch.atom_cond_mask & mask_3d
        cond_atom_features = batch.atom_features * cond_atom_mask
        cond_bond_mask = batch.bond_cond_mask & pair_mask
        cond_bond_features = batch.bond_features * cond_bond_mask

        fourier_sigma = _fourier_noise_embedding(
            sigma,
            embedding_dim=transformer_config.hidden_dim,
            sigma_data=self.config.noise.sigma_data,
            config=self.config.fourier,
        )
        # (batch, 1) fraction of active atom slots per molecule for size conditioning.
        atom_fraction = jnp.sum(batch.atom_mask, axis=-1, keepdims=True) / max(num_atoms, 1)
        sigma_inputs = jnp.concatenate((fourier_sigma, atom_fraction), axis=-1)
        noise_cond = nn.Dense(transformer_config.hidden_dim, name="sigma_mlp_1")(
            nn.silu(nn.Dense(transformer_config.hidden_dim, name="sigma_mlp_0")(sigma_inputs))
        )

        atom_indices = jnp.arange(num_atoms)
        atom_inputs = jnp.concatenate(
            (scaled_atoms, cond_atom_features, cond_atom_mask),
            axis=-1,
        )
        # (batch, num_atoms, hidden_dim) 2D atom, 3D coordinate, and sequence-index projections.
        act = (
            nn.Dense(transformer_config.hidden_dim, name="atom_proj")(atom_inputs)
            + nn.Dense(transformer_config.hidden_dim, use_bias=False, name="pos_proj")(
                scaled_positions
            )
            + nn.Embed(
                batch_lib.MAX_ATOMS,
                transformer_config.hidden_dim,
                name="atom_index_embed",
            )(atom_indices)[None, :, :]
            + noise_cond[:, None, :]
        ) * mask_3d

        geom_inputs = _pairwise_geometry_features(scaled_positions, pair_mask)
        bond_inputs = (
            jnp.concatenate(
                (scaled_bonds, cond_bond_features, cond_bond_mask),
                axis=-1,
            )
            * pair_mask
        )
        init_node_pair = nn.Dense(pair_dim, use_bias=False, name="pair_init_node_proj")(act)
        pair_hidden = (
            nn.silu(
                nn.Dense(pair_dim, name="pair_mlp_0")(geom_inputs)
                + nn.Dense(pair_dim, use_bias=False, name="pair_bond_proj")(bond_inputs)
                + init_node_pair[:, :, None, :]
                + init_node_pair[:, None, :, :]
            )
            * pair_mask
        )
        pair_biases = nn.Dense(
            transformer_config.num_layers * transformer_config.num_heads,
            name="pair_mlp_1",
        )(pair_hidden)
        # (batch, num_layers, num_heads, num_atoms, num_atoms) per-layer attention logit biases.
        pair_biases = pair_biases.reshape(
            batch_size,
            num_atoms,
            num_atoms,
            transformer_config.num_layers,
            transformer_config.num_heads,
        ).transpose(0, 3, 4, 1, 2)

        for layer_idx in range(transformer_config.num_layers):
            act = _TransformerBlock(config=transformer_config, name=f"block_{layer_idx}")(
                act,
                noise_cond=noise_cond,
                pair_bias=pair_biases[:, layer_idx],
                atom_mask=batch.atom_mask,
            )

        norm_act = nn.LayerNorm(name="output_norm")(act)
        raw_positions = nn.Dense(_SPATIAL_DIMS, name="coord_head")(norm_act)
        raw_atoms = nn.Dense(batch_lib.NUM_ATOM_FEATURES, name="atom_head")(norm_act)

        # (batch, 1, 1) and (batch, 1, 1, 1) broadcastable skip and output scaling factors.
        c_skip_atom = factors.c_skip[:, None, None]
        c_out_atom = factors.c_out[:, None, None]
        c_skip_pair = factors.c_skip[:, None, None, None]
        c_out_pair = factors.c_out[:, None, None, None]

        denoised_positions = geometry.zero_center_of_mass(
            c_skip_atom * noisy_positions + c_out_atom * raw_positions,
            batch.atom_mask,
        )
        denoised_atoms = (c_skip_atom * noisy_atom_features + c_out_atom * raw_atoms) * mask_3d

        node_sum_proj = nn.Dense(pair_dim, name="pair_node_proj")(norm_act)
        node_prod_proj = nn.Dense(pair_dim, use_bias=False, name="pair_prod_proj")(norm_act)
        denoised_geom = _pairwise_geometry_features(denoised_positions, pair_mask)
        # (batch, num_atoms, num_atoms, pair_dim) symmetric pairwise activations for bond logits.
        pair_act = nn.silu(
            pair_hidden
            + node_sum_proj[:, :, None, :]
            + node_sum_proj[:, None, :, :]
            + node_prod_proj[:, :, None, :] * node_prod_proj[:, None, :, :]
            + nn.Dense(pair_dim, use_bias=False, name="denoised_geom_proj")(denoised_geom)
        )
        sym_pair_act = _HALF_WEIGHT * (pair_act + jnp.swapaxes(pair_act, 1, 2))
        bond_hidden = nn.silu(nn.Dense(pair_dim, name="bond_mlp")(sym_pair_act))
        raw_bonds = nn.Dense(batch_lib.NUM_BOND_FEATURES, name="bond_head")(bond_hidden)
        raw_bonds = _HALF_WEIGHT * (raw_bonds + jnp.swapaxes(raw_bonds, 1, 2))
        denoised_bonds = (c_skip_pair * noisy_bond_features + c_out_pair * raw_bonds) * pair_mask

        return DenoiserOutput(
            positions=denoised_positions,
            atom_features=denoised_atoms,
            bond_features=denoised_bonds,
        )


@jt.typed
def _fourier_noise_embedding(
    sigma: jt.Float[jt.Array, " batch"],
    *,
    embedding_dim: int,
    sigma_data: float,
    config: config_lib.FourierNoiseConfig,
) -> jt.Float[jt.Array, "batch {embedding_dim}"]:
    """Embeds scaled log-noise levels using sinusoidal log-spaced frequencies."""
    half_dim = embedding_dim // 2
    c_noise = config.log_sigma_scale * jnp.log(sigma / sigma_data)
    freq_exponents = jnp.linspace(0.0, 1.0, half_dim)
    frequencies = jnp.exp(-jnp.log(config.max_period) * freq_exponents)
    # (batch, half_dim) outer product of c_noise (batch, 1) and frequencies (1, half_dim).
    angles = c_noise[:, None] * frequencies[None, :]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


@jt.typed
def _pairwise_geometry_features(
    positions: jt.Float[jt.Array, "batch num_atoms 3"],
    pair_mask: jt.Bool[jt.Array, "batch num_atoms num_atoms 1"],
) -> jt.Float[jt.Array, f"batch num_atoms num_atoms {_NUM_GEOM_FEATURES}"]:
    """Computes masked 3D pairwise displacements, distances, and Gaussian RBF features."""
    # (batch, num_atoms, num_atoms, 3) pairwise displacement vectors x_i - x_j.
    offsets = positions[:, :, None, :] - positions[:, None, :, :]
    # (batch, num_atoms, num_atoms, 1) pairwise Euclidean distances ||x_i - x_j||.
    distances = jnp.sqrt(jnp.sum(jnp.square(offsets), axis=-1, keepdims=True) + _DIST_EPS)
    centers = jnp.linspace(_RBF_MIN_DIST, _RBF_MAX_DIST, _NUM_RBF)
    rbf_width = (_RBF_MAX_DIST - _RBF_MIN_DIST) / (_NUM_RBF - 1)
    # (batch, num_atoms, num_atoms, _NUM_RBF) Gaussian radial basis expansion.
    rbf = jnp.exp(-_HALF_WEIGHT * jnp.square((distances - centers) / rbf_width))
    return jnp.concatenate((offsets, distances, rbf), axis=-1) * pair_mask
