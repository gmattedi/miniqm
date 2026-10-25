"""2D molecular graph featurization from SMILES and padded batch construction."""

from collections.abc import Collection, Iterable, Sequence
import dataclasses
import enum
import functools
import itertools
from typing import Final, override

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

from miniqm import jaxtypes as jt
from miniqm.data import schema

MAX_ATOMS: Final = 29
_SPATIAL_DIMS: Final = 3
_MAX_COVALENT_BOND_A: Final = 2.1
_CIP_CODE_PROP: Final = "_CIPCode"

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
NUM_ATOM_FEATURES: Final = (
    len(_ELEMENTS)
    + (len(_DEGREES) + 1)
    + len(_FORMAL_CHARGES)
    + (len(_HYBRIDIZATIONS) + 1)
    + 2
    + (len(_CHIRAL_CODES) + 1)
)
NUM_BOND_FEATURES: Final = len(_BOND_TYPES) + 2


class SdfProperty(enum.StrEnum):
    """RDKit and SDF property keys attached to generated conformers."""

    NAME = "_Name"
    SMILES = "SMILES"
    CONFORMER_ID = "CONFORMER_ID"
    SEED = "SEED"


@jt.typed
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
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class JaxBatch:
    """Padded JAX batch of 2D molecular graphs and zero-centered 3D coordinates."""

    atom_features: jt.Float[jt.Array, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[jt.Array, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[jt.Array, "batch max_atoms"]
    positions: jt.Float[jt.Array, "batch max_atoms 3"]


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Batch:
    """Padded NumPy batch of 2D molecular graphs and zero-centered 3D coordinates."""

    atom_features: jt.Float[np.ndarray, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[np.ndarray, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[np.ndarray, "batch max_atoms"]
    positions: jt.Float[np.ndarray, "batch max_atoms 3"]

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
        )

    def to_jax(self) -> JaxBatch:
        """Converts all NumPy batch arrays to JAX device arrays."""
        return JaxBatch(
            atom_features=jnp.asarray(self.atom_features),
            bond_features=jnp.asarray(self.bond_features),
            atom_mask=jnp.asarray(self.atom_mask),
            positions=jnp.asarray(self.positions),
        )


@functools.cache
def _disable_rdkit_logging() -> None:
    """Suppresses RDKit console warnings once on first molecule parse."""
    rdBase.DisableLog("rdApp.*")


def _one_hot(
    value: object,
    choices: Iterable[object],
    *,
    include_unknown: bool = False,
) -> tuple[float, ...]:
    """Encodes a value as a one-hot tuple over choices, with an optional catch-all channel."""
    matched_flags = tuple(1.0 if value == choice else 0.0 for choice in choices)
    if not include_unknown:
        return matched_flags
    unknown_flag = 0.0 if any(matched_flags) else 1.0
    return (*matched_flags, unknown_flag)


def _encode_atom(atom: Chem.Atom) -> tuple[float, ...]:
    """Builds the 1D feature vector for a single RDKit atom."""
    cahn_ingold_prelog = atom.GetProp(_CIP_CODE_PROP) if atom.HasProp(_CIP_CODE_PROP) else ""
    chirality_one_hot = (
        1.0 if cahn_ingold_prelog not in _CHIRAL_CODES else 0.0,
        *_one_hot(cahn_ingold_prelog, _CHIRAL_CODES),
    )
    return (
        *_one_hot(atom.GetAtomicNum(), _ELEMENTS),
        *_one_hot(atom.GetTotalDegree(), _DEGREES, include_unknown=True),
        *_one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES),
        *_one_hot(atom.GetHybridization(), _HYBRIDIZATIONS, include_unknown=True),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        *chirality_one_hot,
    )


def _featurize_rdkit_mol(mol: Chem.Mol) -> GraphFeatures:
    """Extracts atom and bond feature arrays from an RDKit molecule."""
    if mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol)
    else:
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
        bond_channel_values = (
            0.0,
            *_one_hot(bond.GetBondType(), _BOND_TYPES),
            1.0 if bond.IsInRing() else 0.0,
        )
        bond_features[begin_idx, end_idx] = bond_channel_values
        bond_features[end_idx, begin_idx] = bond_channel_values

    return GraphFeatures(atom_features=atom_features, bond_features=bond_features)


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parses a SMILES string into an RDKit molecule, or None if invalid."""
    _disable_rdkit_logging()
    # RDKit stubs annotate MolFromSmiles as returning Mol, but it returns None on invalid SMILES.
    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    return mol


def featurize_smiles(smiles: str) -> GraphFeatures:
    """Computes unpadded 2D atom and bond features directly from a SMILES string.

    Args:
        smiles: Input SMILES string.

    Returns:
        Unpadded atom and bond feature arrays with explicit hydrogens added.

    Raises:
        ValueError: If the SMILES string cannot be parsed by RDKit.
    """
    base_mol = _mol_from_smiles(smiles)
    if base_mol is None:
        raise ValueError(f"Failed to parse SMILES {smiles!r}.")
    mol_with_hs = Chem.AddHs(base_mol)
    return _featurize_rdkit_mol(mol_with_hs)


@jt.typed
def _attach_3d_conformer(
    mol: Chem.Mol,
    positions: jt.Float[np.ndarray, "num_atoms 3"],
) -> None:
    """Attaches a 3D conformer populated from Cartesian coordinates in Angstroms."""
    num_atoms, _ = positions.shape
    conformer = Chem.Conformer(num_atoms)
    for atom_idx in range(num_atoms):
        pos = positions[atom_idx]
        conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)


@jt.typed
def _has_valid_3d_bond_lengths(
    mol: Chem.Mol,
    positions: jt.Float[np.ndarray, "num_atoms 3"],
) -> bool:
    """Checks whether all bonded atom pairs fall within covalent bond distance in 3D."""
    for bond in mol.GetBonds():
        begin_pos = positions[bond.GetBeginAtomIdx()]
        end_pos = positions[bond.GetEndAtomIdx()]
        if float(np.linalg.norm(begin_pos - end_pos)) > _MAX_COVALENT_BOND_A:
            return False
    return True


def _heavy_atoms_and_signature(
    mol_with_hs: Chem.Mol,
) -> tuple[tuple[Chem.Atom, ...], tuple[tuple[int, int], ...]]:
    """Extracts heavy atoms and their (atomic_number, attached_hydrogen_count) signature."""
    heavy_atoms: list[Chem.Atom] = []
    signature: list[tuple[int, int]] = []
    for atom in mol_with_hs.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        heavy_atoms.append(atom)
        num_attached_hs = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                num_attached_hs += 1
        signature.append((atom.GetAtomicNum(), num_attached_hs))
    return tuple(heavy_atoms), tuple(signature)


def _attached_hydrogen_indices(heavy_atom: Chem.Atom) -> tuple[int, ...]:
    """Returns indices of hydrogen neighbors bonded to a heavy atom."""
    neighbors = heavy_atom.GetNeighbors()
    is_hydrogen = (neighbor.GetAtomicNum() == 1 for neighbor in neighbors)
    return tuple(int(neighbor.GetIdx()) for neighbor in itertools.compress(neighbors, is_hydrogen))


@jt.typed
def _try_align_by_heavy_order(
    mol_with_hs: Chem.Mol,
    xyz_heavy_indices: Iterable[int],
    xyz_hs_per_heavy: Iterable[Iterable[int]],
    xyz_heavy_signature: tuple[tuple[int, int], ...],
    positions: jt.Float[np.ndarray, "num_atoms 3"],
) -> Chem.Mol | None:
    """Attempts to renumber an RDKit molecule assuming heavy atoms match XYZ order."""
    mol_heavy_atoms, mol_heavy_signature = _heavy_atoms_and_signature(mol_with_hs)
    if mol_heavy_signature != xyz_heavy_signature:
        return None

    mol_to_xyz = [0] * mol_with_hs.GetNumAtoms()
    for heavy_atom, xyz_heavy_idx, xyz_h_group in zip(
        mol_heavy_atoms,
        xyz_heavy_indices,
        xyz_hs_per_heavy,
        strict=True,
    ):
        mol_to_xyz[xyz_heavy_idx] = heavy_atom.GetIdx()
        mol_hs = _attached_hydrogen_indices(heavy_atom)
        for xyz_h_idx, mol_h_idx in zip(xyz_h_group, mol_hs, strict=True):
            mol_to_xyz[xyz_h_idx] = mol_h_idx

    renumbered = Chem.RenumberAtoms(mol_with_hs, mol_to_xyz)
    if not _has_valid_3d_bond_lengths(renumbered, positions):
        return None
    return renumbered


def _try_align_by_substructure(
    molecule: schema.Molecule,
    candidate_mols: Iterable[Chem.Mol],
) -> Chem.Mol | None:
    """Attempts to renumber an RDKit molecule via 3D-inferred bond substructure matching."""
    rw_mol = Chem.RWMol()
    for atomic_num in molecule.atomic_numbers:
        rw_mol.AddAtom(Chem.Atom(int(atomic_num)))
    xyz_mol = rw_mol.GetMol()
    _attach_3d_conformer(xyz_mol, molecule.positions)
    rdDetermineBonds.DetermineConnectivity(xyz_mol)

    query_params = Chem.AdjustQueryParameters.NoAdjustments()
    query_params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(xyz_mol, query_params)

    for mol_with_hs in candidate_mols:
        match = mol_with_hs.GetSubstructMatch(query)
        if len(match) == molecule.num_atoms:
            return Chem.RenumberAtoms(mol_with_hs, list(match))
    return None


def _build_candidate_mols(molecule: schema.Molecule) -> tuple[Chem.Mol, ...]:
    """Parses unique candidate SMILES strings with explicit hydrogens matching atom count."""
    candidates: list[Chem.Mol] = []
    for candidate_smiles in dict.fromkeys((molecule.smiles, molecule.smiles_relaxed)):
        parsed = _mol_from_smiles(candidate_smiles)
        if parsed is None:
            continue
        mol_with_hs = Chem.AddHs(parsed)
        if mol_with_hs.GetNumAtoms() == molecule.num_atoms:
            candidates.append(mol_with_hs)
    return tuple(candidates)


def _parse_aligned_mol(molecule: schema.Molecule) -> Chem.Mol:
    """Parses a SMILES string and renumbers its atoms to match XYZ coordinate order."""
    xyz_atomic_numbers = tuple(int(atomic_num) for atomic_num in molecule.atomic_numbers)
    xyz_heavy_indices: list[int] = []
    for xyz_idx, atomic_num in enumerate(xyz_atomic_numbers):
        if atomic_num > 1:
            xyz_heavy_indices.append(xyz_idx)

    candidate_mols = _build_candidate_mols(molecule)
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

    xyz_heavy_signature = tuple(
        (xyz_atomic_numbers[heavy_idx], len(xyz_h_group))
        for heavy_idx, xyz_h_group in zip(xyz_heavy_indices, xyz_hs_per_heavy, strict=True)
    )
    for mol_with_hs in candidate_mols:
        aligned_mol = _try_align_by_heavy_order(
            mol_with_hs,
            xyz_heavy_indices,
            xyz_hs_per_heavy,
            xyz_heavy_signature,
            molecule.positions,
        )
        if aligned_mol is not None:
            return aligned_mol

    substruct_aligned_mol = _try_align_by_substructure(molecule, candidate_mols)
    if substruct_aligned_mol is not None:
        return substruct_aligned_mol

    raise ValueError(
        f"Could not align SMILES {molecule.smiles!r} or {molecule.smiles_relaxed!r} "
        f"to atomic_numbers for molecule index={molecule.index}."
    )


def featurize_molecule(molecule: schema.Molecule) -> GraphFeatures:
    """Computes 2D graph features aligned to the atom ordering of a molecule.

    Args:
        molecule: Unpadded molecule record with 3D coordinates.

    Returns:
        Unpadded atom and bond feature arrays matching the input coordinate atom order.
    """
    aligned_mol = _parse_aligned_mol(molecule)
    _attach_3d_conformer(aligned_mol, molecule.positions)
    return _featurize_rdkit_mol(aligned_mol)


@jt.typed
def center_positions(
    positions: jt.Float[np.ndarray, "num_atoms 3"],
) -> jt.Float[np.ndarray, "num_atoms 3"]:
    """Subtracts the centroid so coordinates have zero center of mass.

    Args:
        positions: Cartesian atomic coordinates.

    Returns:
        Zero-centered coordinates.
    """
    return positions - positions.mean(axis=0, keepdims=True)


def _pad_graph_features(
    graphs: Sequence[GraphFeatures],
    *,
    max_atoms: int,
) -> Batch:
    """Pads a sequence of 2D molecular graph features into a fixed-shape batch."""
    batch_size = len(graphs)
    _, num_atom_features = graphs[0].atom_features.shape
    _, _, num_bond_features = graphs[0].bond_features.shape

    atom_features = np.zeros((batch_size, max_atoms, num_atom_features), dtype=np.float32)
    bond_features = np.zeros(
        (batch_size, max_atoms, max_atoms, num_bond_features), dtype=np.float32
    )
    atom_mask = np.zeros((batch_size, max_atoms), dtype=np.bool_)
    positions = np.zeros((batch_size, max_atoms, _SPATIAL_DIMS), dtype=np.float32)

    for batch_idx, graph in enumerate(graphs):
        num_atoms, _ = graph.atom_features.shape
        if num_atoms > max_atoms:
            raise ValueError(
                f"Molecule at batch_idx={batch_idx} has {num_atoms} atoms, exceeding {max_atoms=}."
            )
        atom_features[batch_idx, :num_atoms] = graph.atom_features
        bond_features[batch_idx, :num_atoms, :num_atoms] = graph.bond_features
        atom_mask[batch_idx, :num_atoms] = True

    return Batch(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
        positions=positions,
    )


def collate_smiles(
    smiles: Collection[str],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a collection of SMILES strings into a fixed-shape batch.

    Args:
        smiles: Non-empty collection of SMILES strings.
        max_atoms: Padded atom dimension.

    Returns:
        Padded batch of 2D graph features and valid-atom masks for conformer generation.

    Raises:
        ValueError: If smiles is empty or a molecule exceeds max_atoms.
    """
    if not smiles:
        raise ValueError("Expected non-empty sequence of SMILES strings to collate.")
    graphs = tuple(featurize_smiles(smi) for smi in smiles)
    return _pad_graph_features(graphs, max_atoms=max_atoms)


@jt.typed
def build_conformer_mol(
    smiles: str,
    positions: jt.Float[np.ndarray, "max_atoms 3"],
) -> Chem.Mol:
    """Builds an explicit-hydrogen RDKit molecule with a 3D conformer from padded coordinates.

    Args:
        smiles: Input SMILES string defining the molecular graph.
        positions: Padded 3D coordinates in Angstroms ordered by explicit-hydrogen SMILES atoms.

    Returns:
        RDKit molecule with explicit hydrogens and a single 3D conformer attached.

    Raises:
        ValueError: If the SMILES string is invalid or has more atoms than positions.
    """
    base_mol = _mol_from_smiles(smiles)
    if base_mol is None:
        raise ValueError(f"Failed to parse SMILES {smiles!r}.")
    mol_with_hs = Chem.AddHs(base_mol)
    num_atoms = mol_with_hs.GetNumAtoms()
    max_atoms, _ = positions.shape
    if num_atoms > max_atoms:
        raise ValueError(f"Molecule {smiles!r} has {num_atoms} atoms, exceeding {max_atoms=}.")
    _attach_3d_conformer(mol_with_hs, positions[:num_atoms])
    mol_with_hs.SetProp(SdfProperty.NAME, smiles)
    mol_with_hs.SetProp(SdfProperty.SMILES, smiles)
    return mol_with_hs


def collate_batch(
    molecules: Collection[schema.Molecule],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a collection of molecules into a fixed-shape batch.

    Args:
        molecules: Non-empty collection of unpadded molecules.
        max_atoms: Padded atom dimension.

    Returns:
        Padded batch with zero-centered coordinates and valid-atom mask.

    Raises:
        ValueError: If molecules is empty or a molecule exceeds max_atoms.
    """
    if not molecules:
        raise ValueError("Expected non-empty sequence of molecules to collate.")

    graphs = tuple(featurize_molecule(mol) for mol in molecules)
    graph_batch = _pad_graph_features(graphs, max_atoms=max_atoms)
    batch_size = len(molecules)
    positions = np.zeros((batch_size, max_atoms, _SPATIAL_DIMS), dtype=np.float32)

    for batch_idx, mol in enumerate(molecules):
        positions[batch_idx, : mol.num_atoms] = center_positions(mol.positions)

    return dataclasses.replace(graph_batch, positions=positions)
