"""2D molecular graph featurization from SMILES and padded batch construction."""

from collections.abc import Sequence
import dataclasses
from typing import Final, override

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

from miniqm import jaxtypes as jt
from miniqm.data import schema

rdBase.DisableLog("rdApp.*")

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
NUM_ATOM_FEATURES: Final = (
    len(_ELEMENTS)
    + (len(_DEGREES) + 1)
    + len(_FORMAL_CHARGES)
    + (len(_HYBRIDIZATIONS) + 1)
    + 2
    + (len(_CHIRAL_CODES) + 1)
)
NUM_BOND_FEATURES: Final = len(_BOND_TYPES) + 2


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
    """Padded JAX batch of molecular graphs, zero-centered 3D coordinates, and QM9 labels."""

    atom_features: jt.Float[jt.Array, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[jt.Array, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[jt.Array, "batch max_atoms"]
    positions: jt.Float[jt.Array, "*pos_dims"] = dataclasses.field(
        default_factory=lambda: jnp.empty((0, 0, _SPATIAL_DIMS), dtype=jnp.float32)
    )
    labels: jt.Float[jt.Array, "*label_dims"] = dataclasses.field(
        default_factory=lambda: jnp.empty((0, 0), dtype=jnp.float32)
    )


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Batch:
    """Padded NumPy batch of molecular graphs, zero-centered 3D coordinates, and QM9 labels."""

    atom_features: jt.Float[np.ndarray, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[np.ndarray, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[np.ndarray, "batch max_atoms"]
    positions: jt.Float[np.ndarray, "*pos_dims"] = dataclasses.field(
        default_factory=lambda: np.empty((0, 0, _SPATIAL_DIMS), dtype=np.float32)
    )
    labels: jt.Float[np.ndarray, "*label_dims"] = dataclasses.field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )

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

    def to_jax(self) -> JaxBatch:
        """Converts all NumPy batch arrays to JAX device arrays."""
        return JaxBatch(
            atom_features=jnp.asarray(self.atom_features),
            bond_features=jnp.asarray(self.bond_features),
            atom_mask=jnp.asarray(self.atom_mask),
            positions=jnp.asarray(self.positions),
            labels=jnp.asarray(self.labels),
        )


def _one_hot(
    value: object,
    choices: Sequence[object],
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
    cahn_ingold_prelog = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
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
    """Parses a SMILES string into an RDKit molecule, or None if invalid."""
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
    """Attempts to renumber an RDKit molecule assuming heavy atoms match XYZ order."""
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
    """Attempts to renumber an RDKit molecule via 3D-inferred bond substructure matching."""
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
    """Parses a SMILES string and renumbers its atoms to match XYZ coordinate order."""
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
        aligned_mol = _try_align_by_heavy_order(
            mol_with_hs, xyz_heavy_indices, xyz_hs_per_heavy, xyz_heavy_sig
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
    """Computes 2D graph features aligned to the atom ordering of a QM9 molecule.

    Args:
        molecule: Unpadded QM9 molecule record.

    Returns:
        Unpadded atom and bond feature arrays matching the input coordinate atom order.
    """
    aligned_mol = _parse_aligned_mol(molecule)
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
    )


def collate_smiles(
    smiles_seq: Sequence[str],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a sequence of SMILES strings into a fixed-shape batch.

    Args:
        smiles_seq: Non-empty sequence of SMILES strings.
        max_atoms: Padded atom dimension.

    Returns:
        Padded batch of 2D graph features and valid-atom masks for conformer generation.
    """
    if not smiles_seq:
        raise ValueError("Expected non-empty sequence of SMILES strings to collate.")
    graphs = tuple(featurize_smiles(smiles) for smiles in smiles_seq)
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
    """
    base_mol = _mol_from_smiles(smiles)
    if base_mol is None:
        raise ValueError(f"Failed to parse SMILES {smiles!r}.")
    mol_with_hs = Chem.AddHs(base_mol)
    num_atoms = mol_with_hs.GetNumAtoms()
    max_atoms, _ = positions.shape
    if num_atoms > max_atoms:
        raise ValueError(f"Molecule {smiles!r} has {num_atoms} atoms, exceeding {max_atoms=}.")
    conformer = Chem.Conformer(num_atoms)
    for atom_idx in range(num_atoms):
        pos = positions[atom_idx]
        conformer.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol_with_hs.AddConformer(conformer, assignId=True)
    mol_with_hs.SetProp("_Name", smiles)
    mol_with_hs.SetProp("SMILES", smiles)
    return mol_with_hs


def collate_batch(
    molecules: Sequence[schema.Molecule],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a sequence of molecules into a fixed-shape batch.

    Args:
        molecules: Non-empty sequence of unpadded QM9 molecules.
        max_atoms: Padded atom dimension.

    Returns:
        Padded batch with zero-centered coordinates and boolean atom mask.
    """
    if not molecules:
        raise ValueError("Expected non-empty sequence of molecules to collate.")

    graphs = tuple(featurize_molecule(mol) for mol in molecules)
    graph_batch = _pad_graph_features(graphs, max_atoms=max_atoms)
    batch_size = len(molecules)
    positions = np.zeros((batch_size, max_atoms, _SPATIAL_DIMS), dtype=np.float32)
    labels = np.array(
        tuple(dataclasses.astuple(mol.labels) for mol in molecules),
        dtype=np.float32,
    )

    for batch_idx, mol in enumerate(molecules):
        positions[batch_idx, : mol.num_atoms] = center_positions(mol.positions)

    return dataclasses.replace(graph_batch, positions=positions, labels=labels)
