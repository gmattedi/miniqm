"""Interactive 3D molecular and diffusion-trajectory viewers using py3Dmol."""

from typing import Final

import numpy as np
import py3Dmol

from miniqm import jaxtypes as jt

_ELEMENT_SYMBOLS: Final = ("H", "C", "N", "O", "F")
_NUM_ELEMENTS: Final = len(_ELEMENT_SYMBOLS)
_MIN_TARGET_RMS: Final = 0.8
_RMS_FLOOR: Final = 1e-8
_FINAL_FRAME_REPEATS: Final = 8


@jt.typed
def atom_symbols_from_features(
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
) -> tuple[str, ...]:
    """Decodes valid-atom element symbols from one-hot atom feature vectors.

    Args:
        atom_features: Padded per-atom feature matrix whose first 5 channels are one-hot elements.
        atom_mask: Boolean mask indicating valid atoms.

    Returns:
        Tuple of atomic element symbols (H, C, N, O, F) for valid atoms.
    """
    valid_element_one_hot = atom_features[atom_mask, :_NUM_ELEMENTS]
    element_indices = np.argmax(valid_element_one_hot, axis=-1)
    return tuple(_ELEMENT_SYMBOLS[int(idx)] for idx in element_indices)


@jt.typed
def format_xyz_block(
    positions: jt.Float[np.ndarray, "num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    comment: str = "",
) -> str:
    """Formats a single molecule's valid atoms into an XYZ coordinate block.

    Args:
        positions: Padded 3D coordinates in Angstroms.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Boolean mask indicating valid atoms.
        comment: Optional header comment line.

    Returns:
        Standard XYZ-formatted string for the molecule's valid atoms.
    """
    symbols = atom_symbols_from_features(atom_features, atom_mask)
    valid_xyz = positions[atom_mask]
    atom_lines = tuple(
        f"{symbol:<2s} {float(valid_xyz[atom_idx, 0]):12.6f} "
        f"{float(valid_xyz[atom_idx, 1]):12.6f} {float(valid_xyz[atom_idx, 2]):12.6f}"
        for atom_idx, symbol in enumerate(symbols)
    )
    return "\n".join((str(len(symbols)), comment, *atom_lines)) + "\n"


@jt.typed
def format_xyz_trajectory(
    trajectory: jt.Float[np.ndarray, "num_frames num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
) -> str:
    """Formats a sequence of 3D coordinate frames into a multi-frame XYZ string.

    Args:
        trajectory: Padded 3D coordinate frames.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Boolean mask indicating valid atoms.

    Returns:
        Concatenated multi-frame XYZ string.
    """
    blocks = tuple(
        format_xyz_block(
            trajectory[frame_idx],
            atom_features,
            atom_mask,
            comment=f"frame {frame_idx}",
        )
        for frame_idx in range(trajectory.shape[0])
    )
    return "".join(blocks)


@jt.typed
def show_molecule(
    positions: jt.Float[np.ndarray, "num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    width: int = 450,
    height: int = 350,
) -> py3Dmol.view:
    """Creates an interactive 3D ball-and-stick viewer for a single molecule.

    Args:
        positions: Padded 3D coordinates.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Boolean mask indicating valid atoms.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        Configured 3D viewer instance.
    """
    xyz_block = format_xyz_block(positions, atom_features, atom_mask)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(xyz_block, "xyz")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    viewer.zoomTo()
    return viewer


@jt.typed
def show_overlay(
    reference_positions: jt.Float[np.ndarray, "num_atoms 3"],
    candidate_positions: jt.Float[np.ndarray, "num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    width: int = 500,
    height: int = 380,
) -> py3Dmol.view:
    """Creates a 3D viewer overlaying reference (translucent) and candidate (cyan) conformers.

    Args:
        reference_positions: Ground-truth padded 3D coordinates.
        candidate_positions: Predicted/aligned padded 3D coordinates.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Boolean mask indicating valid atoms.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        Configured 3D viewer instance containing both models.
    """
    ref_xyz = format_xyz_block(reference_positions, atom_features, atom_mask, comment="reference")
    cand_xyz = format_xyz_block(candidate_positions, atom_features, atom_mask, comment="candidate")

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(ref_xyz, "xyz")
    viewer.setStyle(
        {"model": 0},
        {
            "stick": {"radius": 0.12, "opacity": 0.45, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.2, "opacity": 0.45, "colorscheme": "Jmol"},
        },
    )
    viewer.addModel(cand_xyz, "xyz")
    viewer.setStyle(
        {"model": 1},
        {
            "stick": {"radius": 0.16, "colorscheme": "cyanCarbon"},
            "sphere": {"scale": 0.26, "colorscheme": "cyanCarbon"},
        },
    )
    viewer.zoomTo()
    return viewer


@jt.typed
def normalize_trajectory_frames(
    trajectory: jt.Float[np.ndarray, "num_frames num_atoms 3"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
) -> jt.Float[np.ndarray, "num_frames num_atoms 3"]:
    """Normalizes each trajectory frame's RMS radius to the final frame's molecular scale.

    Args:
        trajectory: Padded 3D coordinate frames across reverse diffusion steps.
        atom_mask: Boolean mask indicating valid atoms.

    Returns:
        Coordinate trajectory where every frame matches the final molecular RMS spread.
    """
    valid_coords = trajectory[:, atom_mask, :]
    frame_rms = np.sqrt(np.mean(np.square(valid_coords), axis=(1, 2)))
    target_rms = max(float(frame_rms[-1]), _MIN_TARGET_RMS)
    scales = np.where(
        frame_rms > _RMS_FLOOR,
        target_rms / np.maximum(frame_rms, _RMS_FLOOR),
        1.0,
    )
    return (trajectory * scales[:, None, None]).astype(trajectory.dtype, copy=False)


@jt.typed
def show_trajectory(
    trajectory: jt.Float[np.ndarray, "num_frames num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    normalize_scale: bool = True,
    interval_ms: int = 60,
    width: int = 500,
    height: int = 380,
) -> py3Dmol.view:
    """Creates an animated 3D viewer for a multi-frame diffusion coordinate trajectory.

    Args:
        trajectory: Padded 3D coordinate frames.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Boolean mask indicating valid atoms.
        normalize_scale: Whether to rescale frames to a consistent molecular RMS spread.
        interval_ms: Delay between animation frames in milliseconds.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        Configured 3D viewer instance with loop animation enabled.
    """
    display_traj = (
        normalize_trajectory_frames(trajectory, atom_mask) if normalize_scale else trajectory
    )
    final_hold = np.repeat(display_traj[-1:], _FINAL_FRAME_REPEATS, axis=0)
    padded_traj = np.concatenate((display_traj, final_hold), axis=0)
    xyz_traj = format_xyz_trajectory(padded_traj, atom_features, atom_mask)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModelsAsFrames(xyz_traj, "xyz")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}})
    viewer.zoomTo()
    viewer.animate({"loop": "forward", "interval": interval_ms})
    return viewer
