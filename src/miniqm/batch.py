"""2D molecular graph featurization from SMILES and padded batch construction."""

from collections.abc import Iterator, Sequence
import dataclasses
from typing import Final, override

import jax
import jaxtyping as jt
import numpy as np
from rdkit import Chem

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


def _parse_aligned_mol(molecule: schema.Molecule) -> Chem.Mol:
    xyz_atomic_numbers = tuple(int(atomic_num) for atomic_num in molecule.atomic_numbers)
    xyz_heavy = tuple(atomic_num for atomic_num in xyz_atomic_numbers if atomic_num > 1)

    selected_mol: Chem.Mol | None = None
    for candidate_smiles in (molecule.smiles_relaxed, molecule.smiles):
        parsed = _mol_from_smiles(candidate_smiles)
        if parsed is None:
            continue
        parsed_heavy = tuple(
            atom.GetAtomicNum() for atom in parsed.GetAtoms() if atom.GetAtomicNum() > 1
        )
        with_hs = Chem.AddHs(parsed)
        if parsed_heavy == xyz_heavy and with_hs.GetNumAtoms() == molecule.num_atoms:
            selected_mol = with_hs
            break

    if selected_mol is None:
        raise ValueError(
            f"Could not align SMILES {molecule.smiles!r} or {molecule.smiles_relaxed!r} "
            f"to atomic_numbers for molecule index={molecule.index}."
        )

    if not xyz_heavy:
        return selected_mol

    # Align heavy atoms in order and match each hydrogen to its bonded heavy atom in 3D.
    mol_heavy_indices = tuple(
        atom.GetIdx() for atom in selected_mol.GetAtoms() if atom.GetAtomicNum() > 1
    )
    xyz_heavy_indices = tuple(
        xyz_idx for xyz_idx, atomic_num in enumerate(xyz_atomic_numbers) if atomic_num > 1
    )
    heavy_positions = molecule.positions[np.asarray(xyz_heavy_indices)]

    mol_hs_by_heavy: dict[int, list[int]] = {mol_idx: [] for mol_idx in mol_heavy_indices}
    for atom in selected_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            parent_mol_idx = atom.GetNeighbors()[0].GetIdx()
            mol_hs_by_heavy[parent_mol_idx].append(atom.GetIdx())

    xyz_to_mol: list[int] = []
    heavy_cursor = 0
    for xyz_idx, atomic_num in enumerate(xyz_atomic_numbers):
        if atomic_num > 1:
            xyz_to_mol.append(mol_heavy_indices[heavy_cursor])
            heavy_cursor += 1
        else:
            distances = np.linalg.norm(heavy_positions - molecule.positions[xyz_idx], axis=-1)
            nearest_heavy_slot = int(np.argmin(distances))
            parent_mol_idx = mol_heavy_indices[nearest_heavy_slot]
            xyz_to_mol.append(mol_hs_by_heavy[parent_mol_idx].pop(0))

    return Chem.RenumberAtoms(selected_mol, xyz_to_mol)


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


def iter_batches(
    molecules: Sequence[schema.Molecule],
    batch_size: int,
    *,
    rng: np.random.Generator | None = None,
    drop_last: bool = False,
    max_atoms: int = MAX_ATOMS,
) -> Iterator[Batch]:
    """Yields padded batches from a sequence of molecules.

    Args:
        molecules: Dataset of unpadded QM9 molecules.
        batch_size: Number of molecules per batch.
        rng: Optional NumPy random generator for shuffling indices before batching.
        drop_last: Whether to drop the trailing batch if smaller than batch_size.
        max_atoms: Padded atom dimension.

    Yields:
        Collated Batch instances.
    """
    if batch_size <= 0:
        raise ValueError(f"Expected positive batch_size, got {batch_size}.")

    num_molecules = len(molecules)
    indices = rng.permutation(num_molecules) if rng is not None else np.arange(num_molecules)

    for start_idx in range(0, num_molecules, batch_size):
        batch_indices = indices[start_idx : start_idx + batch_size]
        if drop_last and batch_indices.size < batch_size:
            break
        batch_molecules = tuple(molecules[int(mol_idx)] for mol_idx in batch_indices)
        yield collate_batch(batch_molecules, max_atoms=max_atoms)
