import numpy as np
import py3Dmol

from miniqm import viz
from miniqm.data import batch


def test_format_xyz_block_and_trajectory_exclude_padded_atoms():
    graph = batch.featurize_smiles("CO")
    num_valid, num_atom_features = graph.atom_features.shape
    padded_features = np.zeros((num_valid + 2, num_atom_features), dtype=np.float32)
    padded_features[:num_valid] = graph.atom_features
    atom_mask = np.array((*(True,) * num_valid, False, False), dtype=np.bool_)
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
    xyz_trajectory = viz.format_xyz_trajectory(trajectory, padded_features, atom_mask)
    assert xyz_trajectory.count("frame ") == 2


def test_py3dmol_viewers_construct_without_error():
    graph = batch.featurize_smiles("C")
    num_atoms, _ = graph.atom_features.shape
    atom_mask = np.ones((num_atoms,), dtype=np.bool_)
    positions = np.zeros((num_atoms, 3), dtype=np.float32)
    trajectory = np.zeros((4, num_atoms, 3), dtype=np.float32)

    molecule_view = viz.show_molecule(positions, graph.atom_features, atom_mask)
    overlay_view = viz.show_overlay(
        reference_positions=positions,
        candidate_positions=positions + 0.1,
        atom_features=graph.atom_features,
        atom_mask=atom_mask,
    )
    trajectory_view = viz.show_trajectory(trajectory, graph.atom_features, atom_mask)

    assert isinstance(molecule_view, py3Dmol.view)
    assert isinstance(overlay_view, py3Dmol.view)
    assert isinstance(trajectory_view, py3Dmol.view)


def test_normalize_trajectory_frames_rescales_high_variance_frames():
    atom_mask = np.array([True, True, False], dtype=np.bool_)
    final_frame = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    noisy_frame = final_frame * 50.0
    trajectory = np.stack((noisy_frame, final_frame), axis=0)

    normalized = viz.normalize_trajectory_frames(trajectory, atom_mask)
    np.testing.assert_allclose(normalized[0, atom_mask], final_frame[atom_mask], rtol=1e-5)
    np.testing.assert_allclose(normalized[1, atom_mask], final_frame[atom_mask], rtol=1e-5)


def test_format_xyz_trajectory_supports_per_frame_features_and_valence_conditioning():
    template_batch = batch.collate_smiles(("U~U(=U)",))
    atom_mask = template_batch.atom_mask[0]
    max_atoms, num_atom_features = template_batch.atom_features[0].shape
    trajectory = np.zeros((2, max_atoms, 3), dtype=np.float32)

    frame_0_atoms = np.zeros((max_atoms, num_atom_features), dtype=np.float32)
    frame_0_atoms[0, 0] = 5.0  # C
    frame_0_atoms[1, 0] = 5.0  # C
    frame_0_atoms[2, 2] = 5.0  # O

    frame_1_atoms = np.zeros((max_atoms, num_atom_features), dtype=np.float32)
    frame_1_atoms[0, 0] = 5.0  # C
    frame_1_atoms[1, 2] = 5.0  # O (infeasible for ~U(=U) requiring valence >= 3)
    frame_1_atoms[1, 1] = 2.0  # N (next highest feasible element with valence >= 3)
    frame_1_atoms[2, 0] = 5.0  # C
    atom_trajectory = np.stack((frame_0_atoms, frame_1_atoms), axis=0)

    xyz_trajectory = viz.format_xyz_trajectory(
        trajectory,
        atom_trajectory,
        atom_mask,
        cond_atom_features=template_batch.atom_features[0],
        atom_cond_mask=template_batch.atom_cond_mask[0],
        cond_bond_features=template_batch.bond_features[0],
        bond_cond_mask=template_batch.bond_cond_mask[0],
    )
    lines = xyz_trajectory.strip().splitlines()
    frame_0_elements = tuple(line.split()[0] for line in lines[2:5])
    frame_1_elements = tuple(line.split()[0] for line in lines[7:10])
    assert frame_0_elements == ("C", "C", "O")
    assert frame_1_elements == ("C", "N", "C")

    viewer = viz.show_trajectory(
        trajectory,
        atom_trajectory,
        atom_mask,
        cond_atom_features=template_batch.atom_features[0],
        atom_cond_mask=template_batch.atom_cond_mask[0],
        cond_bond_features=template_batch.bond_features[0],
        bond_cond_mask=template_batch.bond_cond_mask[0],
    )
    assert isinstance(viewer, py3Dmol.view)
