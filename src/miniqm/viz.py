"""Interactive 3D molecular and diffusion-trajectory viewers using py3Dmol."""

from typing import Final

import immutabledict
import numpy as np
import py3Dmol

from miniqm import jaxtypes as jt

_ELEMENT_SYMBOLS: Final = ("H", "C", "N", "O", "F")
_NUM_ELEMENTS: Final = len(_ELEMENT_SYMBOLS)
_MIN_TARGET_RMS: Final = 0.8
_RMS_FLOOR: Final = 1e-8
_FINAL_FRAME_REPEATS: Final = 8
_XYZ_FORMAT: Final = "xyz"
_DEFAULT_MOLECULE_WIDTH: Final = 450
_DEFAULT_MOLECULE_HEIGHT: Final = 350
_DEFAULT_VIEWER_WIDTH: Final = 500
_DEFAULT_VIEWER_HEIGHT: Final = 380
_DEFAULT_INTERVAL_MS: Final = 60

_MOLECULE_STYLE: Final = immutabledict.immutabledict(
    {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}}
)
_REFERENCE_OVERLAY_STYLE: Final = immutabledict.immutabledict(
    {
        "stick": {"radius": 0.12, "opacity": 0.45, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.2, "opacity": 0.45, "colorscheme": "Jmol"},
    }
)
_CANDIDATE_OVERLAY_STYLE: Final = immutabledict.immutabledict(
    {
        "stick": {"radius": 0.16, "colorscheme": "cyanCarbon"},
        "sphere": {"scale": 0.26, "colorscheme": "cyanCarbon"},
    }
)
_TRAJECTORY_STYLE: Final = immutabledict.immutabledict(
    {"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}}
)


@jt.typed
def atom_symbols_from_features(
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
) -> tuple[str, ...]:
    """Decodes valid-atom element symbols from one-hot atom feature vectors.

    Args:
        atom_features: Padded per-atom feature matrix whose first 5 channels are one-hot elements.
        atom_mask: Valid-atom mask.

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
    """Formats the valid atoms of a molecule into an XYZ coordinate block.

    Args:
        positions: Padded 3D coordinates in Angstroms.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Valid-atom mask.
        comment: Optional header comment line.

    Returns:
        Standard XYZ string for the valid atoms of the molecule.
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
        atom_mask: Valid-atom mask.

    Returns:
        Concatenated multi-frame XYZ string.
    """
    num_frames, _, _ = trajectory.shape
    blocks = tuple(
        format_xyz_block(
            trajectory[frame_idx],
            atom_features,
            atom_mask,
            comment=f"frame {frame_idx}",
        )
        for frame_idx in range(num_frames)
    )
    return "".join(blocks)


@jt.typed
def show_molecule(
    positions: jt.Float[np.ndarray, "num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    width: int = _DEFAULT_MOLECULE_WIDTH,
    height: int = _DEFAULT_MOLECULE_HEIGHT,
) -> py3Dmol.view:
    """Creates an interactive 3D ball-and-stick viewer for a single molecule.

    Args:
        positions: Padded 3D coordinates.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Valid-atom mask.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        Configured 3D viewer instance.
    """
    xyz_block = format_xyz_block(positions, atom_features, atom_mask)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(xyz_block, _XYZ_FORMAT)
    viewer.setStyle(dict(_MOLECULE_STYLE))
    viewer.zoomTo()
    return viewer


@jt.typed
def show_overlay(
    reference_positions: jt.Float[np.ndarray, "num_atoms 3"],
    candidate_positions: jt.Float[np.ndarray, "num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    width: int = _DEFAULT_VIEWER_WIDTH,
    height: int = _DEFAULT_VIEWER_HEIGHT,
) -> py3Dmol.view:
    """Creates a 3D viewer overlaying reference (translucent) and candidate (cyan) conformers.

    Args:
        reference_positions: Ground-truth padded 3D coordinates.
        candidate_positions: Predicted/aligned padded 3D coordinates.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Valid-atom mask.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        Configured 3D viewer instance containing both models.
    """
    reference_xyz = format_xyz_block(
        reference_positions, atom_features, atom_mask, comment="reference"
    )
    candidate_xyz = format_xyz_block(
        candidate_positions, atom_features, atom_mask, comment="candidate"
    )

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(reference_xyz, _XYZ_FORMAT)
    viewer.setStyle({"model": 0}, dict(_REFERENCE_OVERLAY_STYLE))
    viewer.addModel(candidate_xyz, _XYZ_FORMAT)
    viewer.setStyle({"model": 1}, dict(_CANDIDATE_OVERLAY_STYLE))
    viewer.zoomTo()
    return viewer


@jt.typed
def normalize_trajectory_frames(
    trajectory: jt.Float[np.ndarray, "num_frames num_atoms 3"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
) -> jt.Float[np.ndarray, "num_frames num_atoms 3"]:
    """Normalizes the RMS radius of each trajectory frame to the scale of the final frame.

    Args:
        trajectory: Padded 3D coordinate frames across reverse diffusion steps.
        atom_mask: Valid-atom mask.

    Returns:
        Coordinate trajectory where every frame matches the final molecular RMS spread.
    """
    valid_coords = trajectory[:, atom_mask, :]
    # (num_frames,) RMS coordinate radius across valid atoms for each frame.
    frame_rms = np.sqrt(np.mean(np.square(valid_coords), axis=(1, 2)))
    target_rms = max(float(frame_rms[-1]), _MIN_TARGET_RMS)
    scales = np.where(
        frame_rms > _RMS_FLOOR,
        target_rms / np.maximum(frame_rms, _RMS_FLOOR),
        1.0,
    )
    # (num_frames, num_atoms, 3) trajectory scaled by per-frame (num_frames, 1, 1) multipliers.
    return (trajectory * scales[:, None, None]).astype(trajectory.dtype, copy=False)


@jt.typed
def show_trajectory(
    trajectory: jt.Float[np.ndarray, "num_frames num_atoms 3"],
    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " num_atoms"],
    *,
    normalize_scale: bool = True,
    interval_ms: int = _DEFAULT_INTERVAL_MS,
    width: int = _DEFAULT_VIEWER_WIDTH,
    height: int = _DEFAULT_VIEWER_HEIGHT,
) -> py3Dmol.view:
    """Creates an animated 3D viewer for a multi-frame diffusion coordinate trajectory.

    Args:
        trajectory: Padded 3D coordinate frames.
        atom_features: Padded per-atom feature matrix.
        atom_mask: Valid-atom mask.
        normalize_scale: Whether to rescale frames to a consistent molecular RMS spread.
        interval_ms: Delay between animation frames in milliseconds.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        Configured 3D viewer instance with forward loop animation enabled.
    """
    display_trajectory = (
        normalize_trajectory_frames(trajectory, atom_mask) if normalize_scale else trajectory
    )
    final_hold = np.repeat(display_trajectory[-1:], _FINAL_FRAME_REPEATS, axis=0)
    padded_trajectory = np.concatenate((display_trajectory, final_hold), axis=0)
    xyz_trajectory = format_xyz_trajectory(padded_trajectory, atom_features, atom_mask)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModelsAsFrames(xyz_trajectory, _XYZ_FORMAT)
    viewer.setStyle(dict(_TRAJECTORY_STYLE))
    viewer.zoomTo()
    viewer.animate({"loop": "forward", "interval": interval_ms})
    return viewer
