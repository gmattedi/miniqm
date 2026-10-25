import jax
import numpy as np
import pytest

from miniqm.data import batch, schema


def _make_cf_molecule(*, offset: float = 2.0) -> schema.Molecule:
    return schema.Molecule(
        index=1,
        tag="gdb",
        smiles="CF",
        smiles_relaxed="CF",
        atomic_numbers=np.array([6, 9, 1, 1, 1], dtype=np.int64),
        positions=np.array(
            [
                [offset, 0.0, 0.0],
                [offset + 1.4, 0.0, 0.0],
                [offset - 0.4, 0.9, 0.0],
                [offset - 0.4, -0.5, 0.8],
                [offset - 0.4, -0.5, -0.8],
            ],
            dtype=np.float32,
        ),
    )


def _make_expected_cf_features() -> batch.GraphFeatures:
    # C (Z=6, total_deg=4, total_hs=3, charge=0, SP3, not aromatic, not ring, achiral)
    carbon_features = (
        1.0,
        0.0,
        0.0,
        0.0,  # C, N, O, F
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,  # total_deg 1, 2, 3, 4, other
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,  # total_hs 0, 1, 2, 3, 4
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
    # F (Z=9, total_deg=1, total_hs=0, charge=0, SP3, not aromatic, not ring, achiral)
    fluorine_features = (
        0.0,
        0.0,
        0.0,
        1.0,  # C, N, O, F
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,  # total_deg 1, 2, 3, 4, other
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,  # total_hs 0, 1, 2, 3, 4
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
    atom_features = np.array([carbon_features, fluorine_features], dtype=np.float32)

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


def test_featurize_smiles_matches_expected_cf_graph():
    features = batch.featurize_smiles("CF")

    assert features == _make_expected_cf_features()


def test_featurize_smiles_distinguishes_enantiomers():
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


def test_featurize_molecule_perceives_3d_chirality_without_smiles_stereo_tags():
    # 1-fluoroethanol with an achiral SMILES string ("CC(O)F") and opposite 3D tetrahedral
    # enantiomer coordinates on the central carbon (index 1).
    atomic_numbers = np.array([6, 6, 8, 9, 1, 1, 1, 1, 1], dtype=np.int64)
    base_positions = np.array(
        [
            [-1.4, 0.0, 0.0],  # C0 methyl
            [0.0, 0.0, 0.0],  # C1 chiral center
            [0.5, 1.3, 0.4],  # O2
            [0.5, -0.8, 1.0],  # F3
            [-1.8, 0.9, 0.0],  # H4 on C0
            [-1.8, -0.5, 0.8],  # H5 on C0
            [-1.8, -0.5, -0.8],  # H6 on C0
            [0.4, -0.4, -0.9],  # H7 on C1
            [1.4, 1.3, 0.4],  # H8 on O2
        ],
        dtype=np.float32,
    )
    mirrored_positions = base_positions * np.array([1.0, 1.0, -1.0], dtype=np.float32)

    mol_a = schema.Molecule(
        index=5,
        tag="gdb",
        smiles="CC(O)F",
        smiles_relaxed="CC(O)F",
        atomic_numbers=atomic_numbers,
        positions=base_positions,
    )
    mol_b = schema.Molecule(
        index=6,
        tag="gdb",
        smiles="CC(O)F",
        smiles_relaxed="CC(O)F",
        atomic_numbers=atomic_numbers,
        positions=mirrored_positions,
    )

    features_a = batch.featurize_molecule(mol_a)
    features_b = batch.featurize_molecule(mol_b)

    assert features_a.atom_features.shape == (4, batch.NUM_ATOM_FEATURES)
    assert features_a.atom_features[1, -3] == 0.0
    assert features_b.atom_features[1, -3] == 0.0
    assert not np.array_equal(
        features_a.atom_features[1, -2:],
        features_b.atom_features[1, -2:],
    )


def test_featurize_molecule_aligns_inline_bracket_hydrogens():
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
    )

    features = batch.featurize_molecule(zwitterion)

    assert features.atom_features.shape == (2, batch.NUM_ATOM_FEATURES)
    # Atom 0 is C (channel 0), Atom 1 is N+ (channel 1, formal charge +1 at channel 16).
    assert features.atom_features[0, 0] == 1.0
    assert features.atom_features[1, 1] == 1.0
    assert features.atom_features[1, 16] == 1.0
    assert features.bond_features[0, 1, 1] == 1.0


def test_featurize_molecule_aligns_reordered_smiles_branches_via_substructure():
    # Propene where XYZ orders carbons as C_methyl (0), C_central (1), C_vinyl (2),
    # while both SMILES strings write C_central (=C_vinyl) C_methyl so heavy-order matching
    # returns None and substructure matching aligns the 3D connectivity.
    propene = schema.Molecule(
        index=20,
        tag="gdb",
        smiles="C(=C)C",
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
    )

    features = batch.featurize_molecule(propene)

    # C0-C1 is SINGLE (channel 1), C1-C2 is DOUBLE (channel 2).
    assert features.bond_features[0, 1, 1] == 1.0
    assert features.bond_features[1, 2, 2] == 1.0


def test_collate_batch_pads_and_centers_positions():
    mol = _make_cf_molecule(offset=2.0)
    expected_cf = _make_expected_cf_features()

    collated_batch = batch.collate_batch((mol,), max_atoms=4)

    expected_atom_features = np.zeros((1, 4, batch.NUM_ATOM_FEATURES), dtype=np.float32)
    expected_atom_features[0, :2] = expected_cf.atom_features
    expected_bond_features = np.zeros((1, 4, 4, batch.NUM_BOND_FEATURES), dtype=np.float32)
    expected_bond_features[0, :2, :2] = expected_cf.bond_features
    expected_mask = np.array([[True, True, False, False]], dtype=np.bool_)
    expected_positions = np.array(
        [[[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )

    expected_batch = batch.Batch(
        atom_features=expected_atom_features,
        bond_features=expected_bond_features,
        atom_mask=expected_mask,
        positions=expected_positions,
    )
    assert collated_batch == expected_batch
    assert len(jax.tree.leaves(collated_batch.to_jax())) == 4


def test_featurize_smiles_rejects_invalid_smiles():
    with pytest.raises(ValueError, match="Failed to parse SMILES"):
        batch.featurize_smiles("not_a_smiles")


def test_collate_smiles_pads_graph_features():
    expected_cf = _make_expected_cf_features()
    smiles_batch = batch.collate_smiles(("CF",), max_atoms=4)

    expected_atom_features = np.zeros((1, 4, batch.NUM_ATOM_FEATURES), dtype=np.float32)
    expected_atom_features[0, :2] = expected_cf.atom_features
    expected_bond_features = np.zeros((1, 4, 4, batch.NUM_BOND_FEATURES), dtype=np.float32)
    expected_bond_features[0, :2, :2] = expected_cf.bond_features
    expected_mask = np.array([[True, True, False, False]], dtype=np.bool_)

    np.testing.assert_array_equal(smiles_batch.atom_features, expected_atom_features)
    np.testing.assert_array_equal(smiles_batch.bond_features, expected_bond_features)
    np.testing.assert_array_equal(smiles_batch.atom_mask, expected_mask)
    assert smiles_batch.positions.shape == (1, 4, 3)

    conformer_mol = batch.build_conformer_mol(
        "CF",
        np.array(
            [[0.0, 0.0, 0.0], [1.38, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert conformer_mol.GetNumAtoms() == 2
    assert conformer_mol.GetNumConformers() == 1
    assert conformer_mol.GetProp(batch.SdfProperty.SMILES) == "CF"
    np.testing.assert_allclose(
        conformer_mol.GetConformer().GetPositions(),
        np.array([[0.0, 0.0, 0.0], [1.38, 0.0, 0.0]], dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("fn", "arg", "max_atoms", "error_match"),
    (
        (batch.collate_batch, (), 4, "Expected non-empty sequence"),
        (batch.collate_smiles, (), 4, "Expected non-empty sequence"),
        (batch.collate_batch, (_make_cf_molecule(),), 1, "exceeding max_atoms=1"),
    ),
    ids=("empty_molecules", "empty_smiles", "exceeds_max_atoms"),
)
def test_collate_validates_preconditions(
    fn: object,
    arg: tuple[object, ...],
    max_atoms: int,
    error_match: str,
):
    assert callable(fn)
    with pytest.raises(ValueError, match=error_match):
        fn(arg, max_atoms=max_atoms)
