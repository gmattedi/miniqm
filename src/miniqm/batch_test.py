import dataclasses

import jax
import numpy as np
import pytest

from miniqm import batch, schema


def _make_labels() -> schema.QM9Labels:
    """Builds synthetic QM9 scalar labels for batch tests."""
    return schema.QM9Labels(
        rotational_a=157.7,
        rotational_b=157.7,
        rotational_c=157.7,
        mu=0.0,
        alpha=13.2,
        homo=-0.38,
        lumo=0.11,
        gap=0.50,
        r2=35.3,
        zpve=0.044,
        energy_u0=-40.47,
        energy_u=-40.47,
        enthalpy_h=-40.47,
        free_energy_g=-40.49,
        heat_capacity_cv=6.46,
        u0_atomization=-0.62,
        u_atomization=-0.62,
        h_atomization=-0.63,
        g_atomization=-0.59,
    )


def _make_hf_molecule(*, offset: float = 2.0) -> schema.Molecule:
    """Builds a 2-atom hydrogen fluoride Molecule translated along the x-axis."""
    return schema.Molecule(
        index=1,
        tag="gdb",
        smiles="F",
        smiles_relaxed="F",
        atomic_numbers=np.array([9, 1], dtype=np.int64),
        positions=np.array(
            [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        mulliken_charges=np.array([-0.4, 0.4], dtype=np.float32),
        labels=_make_labels(),
    )


def _make_expected_hf_features() -> batch.GraphFeatures:
    """Builds the expected 2D atom and bond features for hydrogen fluoride."""
    # F (Z=9, deg=1, charge=0, SP3, not aromatic, not ring, achiral)
    fluorine_vec = (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,  # H, C, N, O, F
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,  # deg 1, 2, 3, 4, other
        0.0,
        1.0,
        0.0,  # charge -1, 0, +1
        0.0,
        0.0,
        1.0,
        0.0,  # SP, SP2, SP3, other
        0.0,
        0.0,  # aromatic, in_ring
        1.0,
        0.0,
        0.0,  # chiral none, R, S
    )
    # H (Z=1, deg=1, charge=0, other hybridization, not aromatic, not ring, achiral)
    hydrogen_vec = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    atom_features = np.array([fluorine_vec, hydrogen_vec], dtype=np.float32)

    no_bond = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    single_bond = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    bond_features = np.array(
        [
            [no_bond, single_bond],
            [single_bond, no_bond],
        ],
        dtype=np.float32,
    )
    return batch.GraphFeatures(
        atom_features=atom_features,
        bond_features=bond_features,
    )


def test_featurize_smiles_matches_expected_hf_graph() -> None:
    features = batch.featurize_smiles("F")

    assert features == _make_expected_hf_features()


def test_featurize_smiles_distinguishes_enantiomers() -> None:
    r_features = batch.featurize_smiles("C[C@H](O)F")
    s_features = batch.featurize_smiles("C[C@@H](O)F")

    assert r_features != s_features
    # Chiral carbon at index 1 has (none, R, S) one-hot channels in the last three columns.
    np.testing.assert_array_equal(
        r_features.atom_features[1, -3:],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        s_features.atom_features[1, -3:],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_featurize_molecule_aligns_inline_bracket_hydrogens() -> None:
    # Methylammonium [CH3][NH3+] where the N-attached hydrogens appear immediately after N
    # at indices 2, 3, 4 before the C-attached hydrogens at indices 5, 6, 7.
    zwitterion = schema.Molecule(
        index=10,
        tag="gdb",
        smiles="C[NH3+]",
        smiles_relaxed="C[NH3+]",
        atomic_numbers=np.array([6, 7, 1, 1, 1, 1, 1, 1], dtype=np.int64),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],  # C (0)
                [1.5, 0.0, 0.0],  # N+ (1)
                [2.0, 0.8, 0.0],  # H on N (2)
                [2.0, -0.8, 0.0],  # H on N (3)
                [2.0, 0.0, 0.8],  # H on N (4)
                [-0.5, 0.8, 0.0],  # H on C (5)
                [-0.5, -0.8, 0.0],  # H on C (6)
                [-0.5, 0.0, 0.8],  # H on C (7)
            ],
            dtype=np.float32,
        ),
        mulliken_charges=np.zeros((8,), dtype=np.float32),
        labels=_make_labels(),
    )

    features = batch.featurize_molecule(zwitterion)

    # Atom 1 is N, atoms 2..4 are bonded to N (index 1), and atoms 5..7 are bonded to C (index 0).
    assert features.atom_features[1, 2] == 1.0
    for n_h_idx in (2, 3, 4):
        assert features.bond_features[1, n_h_idx, 1] == 1.0
        assert features.bond_features[0, n_h_idx, 0] == 1.0
    for c_h_idx in (5, 6, 7):
        assert features.bond_features[0, c_h_idx, 1] == 1.0
        assert features.bond_features[1, c_h_idx, 0] == 1.0


def test_featurize_molecule_aligns_reordered_smiles_branches() -> None:
    # Propene where XYZ orders carbons as C_methyl (0), C_central (1), C_vinyl (2),
    # while smiles_relaxed writes the vinyl (=C) branch before the methyl (C) branch.
    propene = schema.Molecule(
        index=20,
        tag="gdb",
        smiles="CC=C",
        smiles_relaxed="C(=C)C",
        atomic_numbers=np.array([6, 6, 6, 1, 1, 1, 1, 1, 1], dtype=np.int64),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],  # C0 methyl (3 Hs: 3, 4, 5)
                [1.5, 0.0, 0.0],  # C1 central (1 H: 6)
                [2.8, 0.0, 0.0],  # C2 vinyl (2 Hs: 7, 8)
                [-0.5, 0.8, 0.0],  # H3 on C0
                [-0.5, -0.8, 0.0],  # H4 on C0
                [-0.5, 0.0, 0.8],  # H5 on C0
                [1.5, 1.0, 0.0],  # H6 on C1
                [3.3, 0.8, 0.0],  # H7 on C2
                [3.3, -0.8, 0.0],  # H8 on C2
            ],
            dtype=np.float32,
        ),
        mulliken_charges=np.zeros((9,), dtype=np.float32),
        labels=_make_labels(),
    )

    features = batch.featurize_molecule(propene)

    # C0-C1 is SINGLE (channel 1), C1-C2 is DOUBLE (channel 2).
    assert features.bond_features[0, 1, 1] == 1.0
    assert features.bond_features[1, 2, 2] == 1.0


def test_collate_batch_pads_and_centers_positions() -> None:
    mol = _make_hf_molecule(offset=2.0)
    expected_hf = _make_expected_hf_features()

    collated_batch = batch.collate_batch((mol,), max_atoms=4)

    expected_atom_features = np.zeros((1, 4, 22), dtype=np.float32)
    expected_atom_features[0, :2] = expected_hf.atom_features
    expected_bond_features = np.zeros((1, 4, 4, 6), dtype=np.float32)
    expected_bond_features[0, :2, :2] = expected_hf.bond_features
    expected_mask = np.array([[True, True, False, False]], dtype=np.bool_)
    expected_positions = np.array(
        [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    expected_labels = np.array([dataclasses.astuple(mol.labels)], dtype=np.float32)

    expected_batch = batch.Batch(
        atom_features=expected_atom_features,
        bond_features=expected_bond_features,
        atom_mask=expected_mask,
        positions=expected_positions,
        labels=expected_labels,
    )
    assert collated_batch == expected_batch
    assert len(jax.tree.leaves(collated_batch.to_jax())) == 5


def test_featurize_smiles_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="Failed to parse SMILES"):
        batch.featurize_smiles("not_a_smiles")


def test_collate_batch_validates_preconditions() -> None:
    with pytest.raises(ValueError, match="Expected non-empty sequence"):
        batch.collate_batch(())

    with pytest.raises(ValueError, match="exceeding max_atoms=1"):
        batch.collate_batch((_make_hf_molecule(),), max_atoms=1)
