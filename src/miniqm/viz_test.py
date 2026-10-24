import numpy as np
import py3Dmol

from miniqm import viz
from miniqm.data import batch


def test_format_xyz_block_and_trajectory_exclude_padded_atoms() -> None:
    graph = batch.featurize_smiles("CO")
    num_valid = graph.atom_features.shape[0]
    padded_features = np.zeros((num_valid + 2, graph.atom_features.shape[1]), dtype=np.float32)
    padded_features[:num_valid] = graph.atom_features
    atom_mask = np.array([True] * num_valid + [False, False], dtype=np.bool_)
    positions = np.arange((num_valid + 2) * 3, dtype=np.float32).reshape(num_valid + 2, 3)

    symbols = viz.atom_symbols_from_features(padded_features, atom_mask)
    xyz_block = viz.format_xyz_block(positions, padded_features, atom_mask, comment="methanol")

    assert symbols[0] == "C"
    assert symbols[1] == "O"
    assert len(symbols) == num_valid
    lines = xyz_block.strip().splitlines()
    assert lines[0] == str(num_valid)
    assert lines[1] == "methanol"
    assert len(lines) == num_valid + 2

    trajectory = np.stack((positions, positions + 1.0), axis=0)
    xyz_traj = viz.format_xyz_trajectory(trajectory, padded_features, atom_mask)
    assert xyz_traj.count("frame ") == 2


def test_py3dmol_viewers_construct_without_error() -> None:
    graph = batch.featurize_smiles("C")
    num_atoms = graph.atom_features.shape[0]
    atom_mask = np.ones((num_atoms,), dtype=np.bool_)
    positions = np.zeros((num_atoms, 3), dtype=np.float32)
    trajectory = np.zeros((4, num_atoms, 3), dtype=np.float32)

    mol_view = viz.show_molecule(positions, graph.atom_features, atom_mask)
    overlay_view = viz.show_overlay(positions, positions + 0.1, graph.atom_features, atom_mask)
    traj_view = viz.show_trajectory(trajectory, graph.atom_features, atom_mask)

    assert isinstance(mol_view, py3Dmol.view)
    assert isinstance(overlay_view, py3Dmol.view)
    assert isinstance(traj_view, py3Dmol.view)


def test_normalize_trajectory_frames_rescales_high_variance_frames() -> None:
    atom_mask = np.array([True, True, False], dtype=np.bool_)
    final_frame = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    noisy_frame = final_frame * 50.0
    trajectory = np.stack((noisy_frame, final_frame), axis=0)

    normalized = viz.normalize_trajectory_frames(trajectory, atom_mask)
    np.testing.assert_allclose(normalized[0, atom_mask], final_frame[atom_mask], rtol=1e-5)
    np.testing.assert_allclose(normalized[1, atom_mask], final_frame[atom_mask], rtol=1e-5)
