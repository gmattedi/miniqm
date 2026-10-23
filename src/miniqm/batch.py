"""2D molecular graph featurization from SMILES and padded batch construction."""

from collections.abc import Sequence
import dataclasses
from typing import Final, override

import jax
import jaxtyping as jt
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from miniqm import schema

MAX_ATOMS: Final = 29
_SPATIAL_DIMS: Final = 3

_ELEMENTS: Final = (1, 6, 7, 8, 9)
_DEGREES: Final = (1, 2, 3, 4)
_FORMAL_CHARGES: Final = (-1, 0, 1)
_HYBRIDIZATIONS: Final = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
)
_CHIRAL_CODES: Final = ("R", "S")
_BOND_TYPES: Final = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class GraphFeatures:
    """Unpadded 2D molecular graph features derived from a SMILES structure."""

    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"]
    bond_features: jt.Float[np.ndarray, "num_atoms num_atoms num_bond_features"]

    __hash__ = None

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphFeatures):
            return NotImplemented
        return np.array_equal(self.atom_features, other.atom_features) and np.array_equal(
            self.bond_features, other.bond_features
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Batch:
    """Padded batch of molecular graphs, zero-COM 3D coordinates, and QM9 labels."""

    atom_features: jt.Float[np.ndarray, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[np.ndarray, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[np.ndarray, "batch max_atoms"]
    positions: jt.Float[np.ndarray, "batch max_atoms 3"]
    labels: jt.Float[np.ndarray, "batch num_labels"]

    __hash__ = None

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Batch):
            return NotImplemented
        return (
            np.array_equal(self.atom_features, other.atom_features)
            and np.array_equal(self.bond_features, other.bond_features)
            and np.array_equal(self.atom_mask, other.atom_mask)
            and np.allclose(self.positions, other.positions)
            and np.allclose(self.labels, other.labels)
        )


def _one_hot(value: object, choices: Sequence[object]) -> tuple[float, ...]:
    return tuple(1.0 if value == choice else 0.0 for choice in choices)


def _one_hot_with_other(value: object, choices: Sequence[object]) -> tuple[float, ...]:
    matched = tuple(1.0 if value == choice else 0.0 for choice in choices)
    other = 0.0 if any(matched) else 1.0
    return (*matched, other)


def _encode_atom(atom: Chem.Atom) -> tuple[float, ...]:
    cahn_ingold_prelog = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
    chirality_one_hot = (
        1.0 if cahn_ingold_prelog not in _CHIRAL_CODES else 0.0,
        *_one_hot(cahn_ingold_prelog, _CHIRAL_CODES),
    )
    return (
        *_one_hot(atom.GetAtomicNum(), _ELEMENTS),
        *_one_hot_with_other(atom.GetTotalDegree(), _DEGREES),
        *_one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES),
        *_one_hot_with_other(atom.GetHybridization(), _HYBRIDIZATIONS),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        *chirality_one_hot,
    )


def _featurize_rdkit_mol(mol: Chem.Mol) -> GraphFeatures:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    num_atoms = mol.GetNumAtoms()

    atom_features = np.array(
        tuple(_encode_atom(atom) for atom in mol.GetAtoms()),
        dtype=np.float32,
    )

    num_bond_types = len(_BOND_TYPES)
    # Channels: [NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC, is_in_ring].
    bond_features = np.zeros((num_atoms, num_atoms, num_bond_types + 2), dtype=np.float32)
    bond_features[:, :, 0] = 1.0

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_vec = (
            0.0,
            *_one_hot(bond.GetBondType(), _BOND_TYPES),
            1.0 if bond.IsInRing() else 0.0,
        )
        bond_features[begin_idx, end_idx] = bond_vec
        bond_features[end_idx, begin_idx] = bond_vec

    return GraphFeatures(atom_features=atom_features, bond_features=bond_features)


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    # RDKit stubs annotate MolFromSmiles as returning Mol, but it returns None on invalid SMILES.
    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    return mol


def featurize_smiles(smiles: str) -> GraphFeatures:
    """Computes unpadded 2D atom and bond features directly from a SMILES string.

    Args:
        smiles: Input SMILES string.

    Returns:
        Unpadded atom and bond feature arrays with explicit hydrogens added.
    """
    base_mol = _mol_from_smiles(smiles)
    if base_mol is None:
        raise ValueError(f"Failed to parse SMILES {smiles!r}.")
    mol_with_hs = Chem.AddHs(base_mol)
    return _featurize_rdkit_mol(mol_with_hs)


def _try_align_by_heavy_order(
    mol_with_hs: Chem.Mol,
    xyz_heavy_indices: Sequence[int],
    xyz_hs_per_heavy: Sequence[Sequence[int]],
    xyz_heavy_sig: Sequence[tuple[int, int]],
) -> Chem.Mol | None:
    mol_heavy_atoms = tuple(atom for atom in mol_with_hs.GetAtoms() if atom.GetAtomicNum() > 1)
    mol_heavy_sig = tuple(
        (
            atom.GetAtomicNum(),
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1),
        )
        for atom in mol_heavy_atoms
    )
    if mol_heavy_sig != xyz_heavy_sig:
        return None

    mol_to_xyz = [0] * mol_with_hs.GetNumAtoms()
    for slot, heavy_atom in enumerate(mol_heavy_atoms):
        mol_to_xyz[xyz_heavy_indices[slot]] = heavy_atom.GetIdx()
        mol_hs = tuple(
            neighbor.GetIdx()
            for neighbor in heavy_atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        for xyz_h_idx, mol_h_idx in zip(xyz_hs_per_heavy[slot], mol_hs, strict=True):
            mol_to_xyz[xyz_h_idx] = mol_h_idx
    return Chem.RenumberAtoms(mol_with_hs, mol_to_xyz)


def _try_align_by_substructure(
    molecule: schema.Molecule,
    candidate_mols: Sequence[Chem.Mol],
) -> Chem.Mol | None:
    rw_mol = Chem.RWMol()
    conformer = Chem.Conformer(molecule.num_atoms)
    for atom_idx, atomic_num in enumerate(molecule.atomic_numbers):
        rw_mol.AddAtom(Chem.Atom(int(atomic_num)))
        pos = molecule.positions[atom_idx]
        conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    xyz_mol = rw_mol.GetMol()
    xyz_mol.AddConformer(conformer, assignId=True)
    rdDetermineBonds.DetermineConnectivity(xyz_mol)

    query_params = Chem.AdjustQueryParameters.NoAdjustments()
    query_params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(xyz_mol, query_params)

    for mol_with_hs in candidate_mols:
        match = mol_with_hs.GetSubstructMatch(query)
        if len(match) == molecule.num_atoms:
            return Chem.RenumberAtoms(mol_with_hs, list(match))
    return None


def _parse_aligned_mol(molecule: schema.Molecule) -> Chem.Mol:
    xyz_atomic_numbers = tuple(int(atomic_num) for atomic_num in molecule.atomic_numbers)
    xyz_heavy_indices = tuple(
        xyz_idx for xyz_idx, atomic_num in enumerate(xyz_atomic_numbers) if atomic_num > 1
    )
    candidate_mols = tuple(
        Chem.AddHs(parsed)
        for candidate_smiles in (molecule.smiles_relaxed, molecule.smiles)
        if (parsed := _mol_from_smiles(candidate_smiles)) is not None
        and Chem.AddHs(parsed).GetNumAtoms() == molecule.num_atoms
    )
    if not candidate_mols:
        raise ValueError(f"Could not parse SMILES for molecule index={molecule.index}.")
    if not xyz_heavy_indices:
        return candidate_mols[0]

    heavy_positions = molecule.positions[np.asarray(xyz_heavy_indices)]
    xyz_hs_per_heavy: tuple[list[int], ...] = tuple([] for _ in xyz_heavy_indices)
    for xyz_idx, atomic_num in enumerate(xyz_atomic_numbers):
        if atomic_num == 1:
            distances = np.linalg.norm(heavy_positions - molecule.positions[xyz_idx], axis=-1)
            xyz_hs_per_heavy[int(np.argmin(distances))].append(xyz_idx)

    xyz_heavy_sig = tuple(
        (xyz_atomic_numbers[heavy_idx], len(xyz_hs_per_heavy[slot]))
        for slot, heavy_idx in enumerate(xyz_heavy_indices)
    )
    for mol_with_hs in candidate_mols:
        aligned = _try_align_by_heavy_order(
            mol_with_hs, xyz_heavy_indices, xyz_hs_per_heavy, xyz_heavy_sig
        )
        if aligned is not None:
            return aligned

    substruct_aligned = _try_align_by_substructure(molecule, candidate_mols)
    if substruct_aligned is not None:
        return substruct_aligned

    raise ValueError(
        f"Could not align SMILES {molecule.smiles!r} or {molecule.smiles_relaxed!r} "
        f"to atomic_numbers for molecule index={molecule.index}."
    )


def featurize_molecule(molecule: schema.Molecule) -> GraphFeatures:
    """Computes 2D graph features aligned to the atom ordering of a QM9 molecule.

    Args:
        molecule: Unpadded QM9 molecule record.

    Returns:
        Unpadded atom and bond feature arrays matching the input coordinate atom order.
    """
    aligned_mol = _parse_aligned_mol(molecule)
    return _featurize_rdkit_mol(aligned_mol)


def center_positions(
    positions: jt.Float[np.ndarray, "num_atoms 3"],
) -> jt.Float[np.ndarray, "num_atoms 3"]:
    """Subtracts the centroid so coordinates have zero center of mass.

    Args:
        positions: [num_atoms, 3] Cartesian coordinates.

    Returns:
        Zero-centered [num_atoms, 3] coordinates.
    """
    return positions - positions.mean(axis=0, keepdims=True)


def collate_batch(
    molecules: Sequence[schema.Molecule],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a sequence of molecules into a fixed-shape Batch.

    Args:
        molecules: Non-empty sequence of unpadded QM9 molecules.
        max_atoms: Padded atom dimension.

    Returns:
        Padded Batch with zero-centered coordinates and boolean atom mask.
    """
    if not molecules:
        raise ValueError("Expected non-empty sequence of molecules to collate.")

    features_per_mol = tuple(featurize_molecule(mol) for mol in molecules)
    batch_size = len(molecules)
    _, num_atom_features = features_per_mol[0].atom_features.shape
    _, _, num_bond_features = features_per_mol[0].bond_features.shape

    atom_features = np.zeros((batch_size, max_atoms, num_atom_features), dtype=np.float32)
    bond_features = np.zeros(
        (batch_size, max_atoms, max_atoms, num_bond_features), dtype=np.float32
    )
    atom_mask = np.zeros((batch_size, max_atoms), dtype=np.bool_)
    positions = np.zeros((batch_size, max_atoms, _SPATIAL_DIMS), dtype=np.float32)
    labels = np.array(
        tuple(dataclasses.astuple(mol.labels) for mol in molecules),
        dtype=np.float32,
    )

    for batch_idx, (mol, graph) in enumerate(zip(molecules, features_per_mol, strict=True)):
        num_atoms = mol.num_atoms
        if num_atoms > max_atoms:
            raise ValueError(
                f"Molecule index={mol.index} has {num_atoms} atoms, exceeding {max_atoms=}."
            )
        atom_features[batch_idx, :num_atoms] = graph.atom_features
        bond_features[batch_idx, :num_atoms, :num_atoms] = graph.bond_features
        atom_mask[batch_idx, :num_atoms] = True
        positions[batch_idx, :num_atoms] = center_positions(mol.positions)

    return Batch(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
        positions=positions,
        labels=labels,
    )
