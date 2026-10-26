"""2D molecular graph featurization from SMILES and padded batch construction."""

from collections.abc import Collection, Iterable, Sequence
import dataclasses
import enum
import functools
import itertools
import re
from typing import Final, cast, override

import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

from miniqm import jaxtypes as jt
from miniqm.data import schema

MAX_ATOMS: Final = 9
_SPATIAL_DIMS: Final = 3
_MAX_COVALENT_BOND_A: Final = 2.1
_CIP_CODE_PROP: Final = "_CIPCode"
_UNSPECIFIED_BOND_PROP: Final = "_UnspecifiedBond"
_BARE_URANIUM_PATTERN: Final = re.compile(r"(?<!\[)U(?!\])")
_MASKED_LOGIT: Final = -1e9
_BOND_ACTIVE_THRESHOLD: Final = 0.5
_SINGLE_BOND_VALENCE: Final = 1.0
_HALF_WEIGHT: Final = 0.5

_ELEMENTS: Final = (6, 7, 8, 9)
_SYMBOL_BY_ELEMENT: Final = immutabledict.immutabledict({6: "C", 7: "N", 8: "O", 9: "F"})
_DEGREES: Final = (1, 2, 3, 4)
_TOTAL_HS: Final = (0, 1, 2, 3, 4)
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
_MAX_VALENCE_BY_ELEMENT: Final = immutabledict.immutabledict({6: 4.0, 7: 3.0, 8: 2.0, 9: 1.0})
_VALENCE_BY_BOND_TYPE: Final = immutabledict.immutabledict(
    {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 1.5,
    }
)
_ELEMENT_SLICE: Final = slice(0, len(_ELEMENTS))
_TOTAL_HS_START: Final = len(_ELEMENTS) + (len(_DEGREES) + 1)
_TOTAL_HS_SLICE: Final = slice(_TOTAL_HS_START, _TOTAL_HS_START + len(_TOTAL_HS))
_CHARGE_START: Final = _TOTAL_HS_START + len(_TOTAL_HS)
_CHARGE_SLICE: Final = slice(_CHARGE_START, _CHARGE_START + len(_FORMAL_CHARGES))
_RING_INDEX: Final = _CHARGE_START + len(_FORMAL_CHARGES) + (len(_HYBRIDIZATIONS) + 1) + 1
_CHIRAL_SLICE: Final = slice(_RING_INDEX + 1, _RING_INDEX + 1 + len(_CHIRAL_CODES) + 1)
NUM_ATOM_FEATURES: Final = (
    len(_ELEMENTS)
    + (len(_DEGREES) + 1)
    + len(_TOTAL_HS)
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
    """Unpadded 2D molecular graph features and per-channel conditioning masks."""

    atom_features: jt.Float[np.ndarray, "num_atoms num_atom_features"]
    bond_features: jt.Float[np.ndarray, "num_atoms num_atoms num_bond_features"]
    atom_cond_mask: jt.Bool[np.ndarray, "num_atoms num_atom_features"]
    bond_cond_mask: jt.Bool[np.ndarray, "num_atoms num_atoms num_bond_features"]

    __hash__ = None

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphFeatures):
            return NotImplemented
        return (
            np.array_equal(self.atom_features, other.atom_features)
            and np.array_equal(self.bond_features, other.bond_features)
            and np.array_equal(self.atom_cond_mask, other.atom_cond_mask)
            and np.array_equal(self.bond_cond_mask, other.bond_cond_mask)
        )


@jax.tree_util.register_dataclass
@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class JaxBatch:
    """Padded JAX batch of 2D molecular graphs, conditioning masks, and 3D coordinates."""

    atom_features: jt.Float[jt.Array, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[jt.Array, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[jt.Array, "batch max_atoms"]
    positions: jt.Float[jt.Array, "batch max_atoms 3"]
    atom_cond_mask: jt.Bool[jt.Array, "batch max_atoms num_atom_features"]
    bond_cond_mask: jt.Bool[jt.Array, "batch max_atoms max_atoms num_bond_features"]


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Batch:
    """Padded NumPy batch of 2D molecular graphs, conditioning masks, and 3D coordinates."""

    atom_features: jt.Float[np.ndarray, "batch max_atoms num_atom_features"]
    bond_features: jt.Float[np.ndarray, "batch max_atoms max_atoms num_bond_features"]
    atom_mask: jt.Bool[np.ndarray, "batch max_atoms"]
    positions: jt.Float[np.ndarray, "batch max_atoms 3"]
    atom_cond_mask: jt.Bool[np.ndarray, "batch max_atoms num_atom_features"]
    bond_cond_mask: jt.Bool[np.ndarray, "batch max_atoms max_atoms num_bond_features"]

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
            and np.array_equal(self.atom_cond_mask, other.atom_cond_mask)
            and np.array_equal(self.bond_cond_mask, other.bond_cond_mask)
        )

    def to_jax(self) -> JaxBatch:
        """Converts all NumPy batch arrays to JAX device arrays."""
        return JaxBatch(
            atom_features=jnp.asarray(self.atom_features),
            bond_features=jnp.asarray(self.bond_features),
            atom_mask=jnp.asarray(self.atom_mask),
            positions=jnp.asarray(self.positions),
            atom_cond_mask=jnp.asarray(self.atom_cond_mask),
            bond_cond_mask=jnp.asarray(self.bond_cond_mask),
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _BondCandidate:
    """Ranked candidate bond between two valid heavy atoms."""

    is_cond_bonded: bool
    is_order_cond: bool
    margin: float
    begin_idx: int
    end_idx: int
    bond_type: Chem.BondType


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
    """Builds the 1D feature vector for a single heavy RDKit atom."""
    cahn_ingold_prelog = atom.GetProp(_CIP_CODE_PROP) if atom.HasProp(_CIP_CODE_PROP) else ""
    chirality_one_hot = (
        1.0 if cahn_ingold_prelog not in _CHIRAL_CODES else 0.0,
        *_one_hot(cahn_ingold_prelog, _CHIRAL_CODES),
    )
    return (
        *_one_hot(atom.GetAtomicNum(), _ELEMENTS),
        *_one_hot(atom.GetTotalDegree(), _DEGREES, include_unknown=True),
        *_one_hot(atom.GetTotalNumHs(), _TOTAL_HS),
        *_one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES),
        *_one_hot(atom.GetHybridization(), _HYBRIDIZATIONS, include_unknown=True),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        *chirality_one_hot,
    )


def _strip_hydrogens_preserving_cip(mol: Chem.Mol) -> Chem.Mol:
    """Removes explicit hydrogens while preserving 3D-perceived CIP codes on heavy atoms."""
    atoms = tuple(mol.GetAtoms())
    heavy_atoms = tuple(itertools.compress(atoms, (atom.GetAtomicNum() > 1 for atom in atoms)))
    heavy_cip_codes = tuple(
        atom.GetProp(_CIP_CODE_PROP) if atom.HasProp(_CIP_CODE_PROP) else "" for atom in heavy_atoms
    )
    heavy_mol = Chem.RemoveAllHs(mol)
    for atom, cip_code in zip(heavy_mol.GetAtoms(), heavy_cip_codes, strict=True):
        if cip_code:
            atom.SetProp(_CIP_CODE_PROP, cip_code)
    return heavy_mol


def _atom_touches_unknown(atom: Chem.Atom, is_dummy: Sequence[bool]) -> bool:
    """Returns True if an atom has a dummy neighbor or an unspecified (~) incident bond."""
    atom_idx = atom.GetIdx()
    return any(
        bond.HasProp(_UNSPECIFIED_BOND_PROP) or is_dummy[bond.GetOtherAtomIdx(atom_idx)]
        for bond in atom.GetBonds()
    )


def _build_atom_cond_mask(
    atom: Chem.Atom,
    *,
    is_dummy: Sequence[bool],
) -> np.ndarray:
    """Builds the 1D boolean conditioning mask for a single heavy atom."""
    atom_idx = atom.GetIdx()
    if is_dummy[atom_idx]:
        return np.zeros((NUM_ATOM_FEATURES,), dtype=np.bool_)
    if not _atom_touches_unknown(atom, is_dummy):
        return np.ones((NUM_ATOM_FEATURES,), dtype=np.bool_)

    mask = np.zeros((NUM_ATOM_FEATURES,), dtype=np.bool_)
    mask[_ELEMENT_SLICE] = True
    mask[_CHARGE_SLICE] = True
    if atom.IsInRing():
        mask[_RING_INDEX - 1 : _RING_INDEX + 1] = True
    if atom.HasProp(_CIP_CODE_PROP):
        mask[_CHIRAL_SLICE] = True
    return mask


def _featurize_rdkit_mol(mol: Chem.Mol) -> GraphFeatures:
    """Extracts heavy-atom and bond feature arrays and conditioning masks from an RDKit molecule."""
    if mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    heavy_mol = _strip_hydrogens_preserving_cip(mol)
    num_atoms = heavy_mol.GetNumAtoms()
    atoms = tuple(heavy_mol.GetAtoms())
    is_dummy = tuple(atom.GetAtomicNum() not in _ELEMENTS for atom in atoms)

    atom_features = np.zeros((num_atoms, NUM_ATOM_FEATURES), dtype=np.float32)
    atom_cond_mask = np.zeros((num_atoms, NUM_ATOM_FEATURES), dtype=np.bool_)
    for atom_idx, atom in enumerate(atoms):
        atom_cond_mask[atom_idx] = _build_atom_cond_mask(atom, is_dummy=is_dummy)
        if not is_dummy[atom_idx]:
            atom_features[atom_idx] = np.asarray(_encode_atom(atom), dtype=np.float32)

    # Channels: [NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC, is_in_ring].
    bond_features = np.zeros((num_atoms, num_atoms, NUM_BOND_FEATURES), dtype=np.float32)
    bond_cond_mask = np.zeros((num_atoms, num_atoms, NUM_BOND_FEATURES), dtype=np.bool_)
    known_atom_mask = ~np.asarray(is_dummy, dtype=np.bool_)
    known_pair_mask = (known_atom_mask[:, None] & known_atom_mask[None, :]) | np.eye(
        num_atoms, dtype=np.bool_
    )
    bond_features[known_pair_mask, 0] = 1.0
    bond_cond_mask[known_pair_mask, :] = True

    for bond in heavy_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if bond.HasProp(_UNSPECIFIED_BOND_PROP):
            unspecified_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unspecified_mask = (True, False, False, False, False, False)
            bond_features[begin_idx, end_idx] = unspecified_values
            bond_features[end_idx, begin_idx] = unspecified_values
            bond_cond_mask[begin_idx, end_idx] = unspecified_mask
            bond_cond_mask[end_idx, begin_idx] = unspecified_mask
        else:
            bond_channel_values = (
                0.0,
                *_one_hot(bond.GetBondType(), _BOND_TYPES),
                1.0 if bond.IsInRing() else 0.0,
            )
            ring_known = (not is_dummy[begin_idx] and not is_dummy[end_idx]) or bond.IsInRing()
            known_bond_mask = (True, True, True, True, True, ring_known)
            bond_features[begin_idx, end_idx] = bond_channel_values
            bond_features[end_idx, begin_idx] = bond_channel_values
            bond_cond_mask[begin_idx, end_idx] = known_bond_mask
            bond_cond_mask[end_idx, begin_idx] = known_bond_mask

    return GraphFeatures(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_cond_mask=atom_cond_mask,
        bond_cond_mask=bond_cond_mask,
    )


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parses a SMILES string into an RDKit molecule, or None if invalid."""
    _disable_rdkit_logging()
    # RDKit stubs annotate MolFromSmiles as returning Mol, but it returns None on invalid SMILES.
    mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
    return mol


def _parse_template_smiles(smiles: str) -> Chem.Mol | None:
    """Parses a SMILES or template string supporting U/* dummy atoms and ~ dummy bonds."""
    normalized = _BARE_URANIUM_PATTERN.sub("[U]", smiles)
    if "~" not in normalized:
        return _mol_from_smiles(normalized)

    _disable_rdkit_logging()
    smarts_mol = cast("Chem.Mol | None", Chem.MolFromSmarts(normalized))
    mol = cast(
        "Chem.Mol | None",
        Chem.MolFromSmiles(normalized.replace("~", "-"), sanitize=False),
    )
    if smarts_mol is None or mol is None or smarts_mol.GetNumAtoms() != mol.GetNumAtoms():
        return None
    for smarts_bond in smarts_mol.GetBonds():
        if smarts_bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            mol_bond = cast(
                "Chem.Bond | None",
                mol.GetBondBetweenAtoms(
                    smarts_bond.GetBeginAtomIdx(),
                    smarts_bond.GetEndAtomIdx(),
                ),
            )
            if mol_bond is not None:
                mol_bond.SetIntProp(_UNSPECIFIED_BOND_PROP, 1)
                if (
                    mol_bond.GetBeginAtom().GetIsAromatic()
                    and mol_bond.GetEndAtom().GetIsAromatic()
                ):
                    mol_bond.SetBondType(Chem.BondType.AROMATIC)
    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    return mol


def featurize_smiles(smiles: str) -> GraphFeatures:
    """Computes unpadded heavy-atom 2D features and conditioning masks from a SMILES or template.

    Supports dummy atoms (U or *) and unspecified bonds (~).

    Args:
        smiles: Input SMILES or template string.

    Returns:
        Unpadded heavy-atom and bond feature arrays with per-channel conditioning masks.

    Raises:
        ValueError: If the SMILES or template string cannot be parsed by RDKit.
    """
    base_mol = _parse_template_smiles(smiles)
    if base_mol is None:
        raise ValueError(f"Failed to parse SMILES {smiles!r}.")
    return _featurize_rdkit_mol(base_mol)


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
    xyz_heavy_indices = tuple(
        itertools.compress(
            range(len(xyz_atomic_numbers)),
            (atomic_num > 1 for atomic_num in xyz_atomic_numbers),
        )
    )

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
    atom_cond_mask = np.zeros((batch_size, max_atoms, num_atom_features), dtype=np.bool_)
    bond_cond_mask = np.zeros((batch_size, max_atoms, max_atoms, num_bond_features), dtype=np.bool_)

    for batch_idx, graph in enumerate(graphs):
        num_atoms, _ = graph.atom_features.shape
        if num_atoms > max_atoms:
            raise ValueError(
                f"Molecule at batch_idx={batch_idx} has {num_atoms} atoms, exceeding {max_atoms=}."
            )
        atom_features[batch_idx, :num_atoms] = graph.atom_features
        bond_features[batch_idx, :num_atoms, :num_atoms] = graph.bond_features
        atom_mask[batch_idx, :num_atoms] = True
        atom_cond_mask[batch_idx, :num_atoms] = graph.atom_cond_mask
        bond_cond_mask[batch_idx, :num_atoms, :num_atoms] = graph.bond_cond_mask

    return Batch(
        atom_features=atom_features,
        bond_features=bond_features,
        atom_mask=atom_mask,
        positions=positions,
        atom_cond_mask=atom_cond_mask,
        bond_cond_mask=bond_cond_mask,
    )


def collate_smiles(
    smiles: Collection[str],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a collection of SMILES or template strings into a fixed-shape batch.

    Args:
        smiles: Non-empty collection of SMILES or template strings.
        max_atoms: Padded atom dimension.

    Returns:
        Padded batch of 2D graph features, conditioning masks, and valid-atom masks.

    Raises:
        ValueError: If smiles is empty or a molecule exceeds max_atoms.
    """
    if not smiles:
        raise ValueError("Expected non-empty sequence of SMILES strings to collate.")
    graphs = tuple(featurize_smiles(smi) for smi in smiles)
    return _pad_graph_features(graphs, max_atoms=max_atoms)


@jt.typed
def _min_conditioned_bond_valences(
    indexed_atoms: Collection[tuple[int, int]],
    *,
    cond_bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"] | None,
    bond_cond_mask: jt.Bool[np.ndarray, "max_atoms max_atoms num_bond_features"] | None,
) -> tuple[float, ...]:
    """Computes the minimum heavy-atom valence required by conditioned bonds per valid atom."""
    min_valences = [0.0] * len(indexed_atoms)
    if cond_bond_features is None or bond_cond_mask is None:
        return tuple(min_valences)
    num_bond_classes = len(_BOND_TYPES) + 1
    for (dst_i, src_i), (dst_j, src_j) in itertools.combinations(indexed_atoms, 2):
        if (
            bool(bond_cond_mask[src_i, src_j, 0])
            and float(cond_bond_features[src_i, src_j, 0]) < _BOND_ACTIVE_THRESHOLD
        ):
            valence = (
                _VALENCE_BY_BOND_TYPE[
                    _BOND_TYPES[
                        int(np.argmax(cond_bond_features[src_i, src_j, 1:num_bond_classes]))
                    ]
                ]
                if bool(bond_cond_mask[src_i, src_j, 1])
                else _SINGLE_BOND_VALENCE
            )
            min_valences[dst_i] += valence
            min_valences[dst_j] += valence
    return tuple(min_valences)


@jt.typed
def _resolve_effective_atoms(
    atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"],
    *,
    cond_atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"] | None,
    atom_cond_mask: jt.Bool[np.ndarray, "max_atoms num_atom_features"] | None,
) -> jt.Float[np.ndarray, "max_atoms num_atom_features"]:
    """Overlays conditioned atom feature channels onto predicted atom features."""
    if cond_atom_features is None or atom_cond_mask is None:
        return atom_features
    return np.where(
        atom_cond_mask,
        np.where(cond_atom_features > 0.0, 1.0, _MASKED_LOGIT),
        atom_features,
    )


@jt.typed
def _decode_atom_element_and_hs(
    src_i: int,
    min_bond_valence: float,
    *,
    effective_atoms: jt.Float[np.ndarray, "max_atoms num_atom_features"],
    cond_atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"] | None,
    atom_cond_mask: jt.Bool[np.ndarray, "max_atoms num_atom_features"] | None,
) -> tuple[int, float]:
    """Selects the highest-logit valence-compatible atomic number and conditioned H count."""
    cond_hs = (
        float(_TOTAL_HS[int(np.argmax(cond_atom_features[src_i, _TOTAL_HS_SLICE]))])
        if (
            cond_atom_features is not None
            and atom_cond_mask is not None
            and bool(np.all(atom_cond_mask[src_i, _TOTAL_HS_SLICE]))
        )
        else 0.0
    )
    required_valence = cond_hs + min_bond_valence
    feasible_mask = np.array(
        [_MAX_VALENCE_BY_ELEMENT[el] >= required_valence for el in _ELEMENTS],
        dtype=np.bool_,
    )
    element_logits = np.where(
        feasible_mask,
        effective_atoms[src_i, _ELEMENT_SLICE],
        _MASKED_LOGIT,
    )
    return _ELEMENTS[int(np.argmax(element_logits))], cond_hs


@jt.typed
def decode_atom_symbols(
    atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"],
    atom_mask: jt.Bool[np.ndarray, " max_atoms"],
    *,
    cond_atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"] | None = None,
    atom_cond_mask: jt.Bool[np.ndarray, "max_atoms num_atom_features"] | None = None,
    cond_bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"] | None = None,
    bond_cond_mask: jt.Bool[np.ndarray, "max_atoms max_atoms num_bond_features"] | None = None,
) -> tuple[str, ...]:
    """Decodes valence-compatible element symbols for valid atoms from feature arrays.

    Args:
        atom_features: Padded per-atom feature matrix.
        atom_mask: Valid-atom mask.
        cond_atom_features: Optional ground-truth or template atom feature matrix.
        atom_cond_mask: Optional boolean mask of conditioned atom feature channels.
        cond_bond_features: Optional ground-truth or template bond feature tensor.
        bond_cond_mask: Optional boolean mask of conditioned bond feature channels.

    Returns:
        Tuple of atomic element symbols (C, N, O, F) for valid atoms.
    """
    effective_atoms = _resolve_effective_atoms(
        atom_features,
        cond_atom_features=cond_atom_features,
        atom_cond_mask=atom_cond_mask,
    )
    indexed_atoms = tuple(enumerate(int(idx) for idx in np.flatnonzero(atom_mask)))
    min_bond_valences = _min_conditioned_bond_valences(
        indexed_atoms,
        cond_bond_features=cond_bond_features,
        bond_cond_mask=bond_cond_mask,
    )
    symbols: list[str] = []
    for dst_i, src_i in indexed_atoms:
        atomic_num, _ = _decode_atom_element_and_hs(
            src_i,
            min_bond_valences[dst_i],
            effective_atoms=effective_atoms,
            cond_atom_features=cond_atom_features,
            atom_cond_mask=atom_cond_mask,
        )
        symbols.append(_SYMBOL_BY_ELEMENT[atomic_num])
    return tuple(symbols)


@jt.typed
def _add_decoded_atoms(
    rw_mol: Chem.RWMol,
    indexed_atoms: Collection[tuple[int, int]],
    *,
    effective_atoms: jt.Float[np.ndarray, "max_atoms num_atom_features"],
    cond_atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"] | None,
    atom_cond_mask: jt.Bool[np.ndarray, "max_atoms num_atom_features"] | None,
    cond_bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"] | None,
    bond_cond_mask: jt.Bool[np.ndarray, "max_atoms max_atoms num_bond_features"] | None,
) -> list[float]:
    """Adds valence-compatible decoded heavy atoms and returns remaining valences."""
    min_bond_valences = _min_conditioned_bond_valences(
        indexed_atoms,
        cond_bond_features=cond_bond_features,
        bond_cond_mask=bond_cond_mask,
    )
    remaining_valences: list[float] = []
    for dst_i, src_i in indexed_atoms:
        atomic_num, cond_hs = _decode_atom_element_and_hs(
            src_i,
            min_bond_valences[dst_i],
            effective_atoms=effective_atoms,
            cond_atom_features=cond_atom_features,
            atom_cond_mask=atom_cond_mask,
        )
        charge = _FORMAL_CHARGES[int(np.argmax(effective_atoms[src_i, _CHARGE_SLICE]))]
        rd_atom = Chem.Atom(atomic_num)
        rd_atom.SetFormalCharge(charge)
        rw_mol.AddAtom(rd_atom)
        remaining_valences.append(max(_MAX_VALENCE_BY_ELEMENT[atomic_num] + charge - cond_hs, 0.0))
    return remaining_valences


@jt.typed
def _rank_bond_candidates(
    indexed_atoms: Iterable[tuple[int, int]],
    *,
    effective_bonds: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"],
    cond_bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"] | None,
    bond_cond_mask: jt.Bool[np.ndarray, "max_atoms max_atoms num_bond_features"] | None,
) -> tuple[_BondCandidate, ...]:
    """Ranks non-excluded atom pairs by bond conditioning status and predicted logit margin."""
    num_bond_classes = len(_BOND_TYPES) + 1
    candidates: list[_BondCandidate] = []
    for (dst_i, src_i), (dst_j, src_j) in itertools.combinations(indexed_atoms, 2):
        pair_logits = _HALF_WEIGHT * (
            effective_bonds[src_i, src_j, :num_bond_classes]
            + effective_bonds[src_j, src_i, :num_bond_classes]
        )
        is_cond_pair = bond_cond_mask is not None and bool(bond_cond_mask[src_i, src_j, 0])
        is_cond_none = (
            is_cond_pair
            and cond_bond_features is not None
            and float(cond_bond_features[src_i, src_j, 0]) > _BOND_ACTIVE_THRESHOLD
        )
        if is_cond_none:
            continue
        is_order_cond = bond_cond_mask is not None and bool(bond_cond_mask[src_i, src_j, 1])
        best_order_idx = int(np.argmax(pair_logits[1:])) + 1
        margin = float(pair_logits[best_order_idx] - pair_logits[0])
        candidates.append(
            _BondCandidate(
                is_cond_bonded=is_cond_pair,
                is_order_cond=is_order_cond,
                margin=margin,
                begin_idx=dst_i,
                end_idx=dst_j,
                bond_type=_BOND_TYPES[best_order_idx - 1],
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.is_cond_bonded,
                candidate.is_order_cond,
                candidate.margin,
            ),
            reverse=True,
        )
    )


@jt.typed
def _build_raw_generated_mol(
    valid_indices: Iterable[int],
    atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"],
    bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"],
    *,
    cond_atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"] | None = None,
    atom_cond_mask: jt.Bool[np.ndarray, "max_atoms num_atom_features"] | None = None,
    cond_bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"] | None = None,
    bond_cond_mask: jt.Bool[np.ndarray, "max_atoms max_atoms num_bond_features"] | None = None,
    allow_aromatic: bool,
) -> Chem.Mol:
    """Constructs an unsanitized RDKit molecule respecting atomic valences and connectivity."""
    rw_mol = Chem.RWMol()
    indexed_atoms = tuple(enumerate(int(idx) for idx in valid_indices))
    remaining_valences = _add_decoded_atoms(
        rw_mol,
        indexed_atoms,
        effective_atoms=atom_features,
        cond_atom_features=cond_atom_features,
        atom_cond_mask=atom_cond_mask,
        cond_bond_features=cond_bond_features,
        bond_cond_mask=bond_cond_mask,
    )
    ranked_candidates = _rank_bond_candidates(
        indexed_atoms,
        effective_bonds=bond_features,
        cond_bond_features=cond_bond_features,
        bond_cond_mask=bond_cond_mask,
    )

    for candidate in ranked_candidates:
        if not candidate.is_cond_bonded and candidate.margin <= 0.0:
            continue
        bond_type = candidate.bond_type
        if (
            not allow_aromatic
            and not candidate.is_order_cond
            and bond_type == Chem.BondType.AROMATIC
        ):
            bond_type = Chem.BondType.SINGLE
        valence = _VALENCE_BY_BOND_TYPE[bond_type]
        begin_idx = candidate.begin_idx
        end_idx = candidate.end_idx
        if not candidate.is_order_cond and (
            remaining_valences[begin_idx] < valence or remaining_valences[end_idx] < valence
        ):
            bond_type = Chem.BondType.SINGLE
            valence = _SINGLE_BOND_VALENCE
        if candidate.is_cond_bonded or (
            remaining_valences[begin_idx] >= valence and remaining_valences[end_idx] >= valence
        ):
            rw_mol.AddBond(begin_idx, end_idx, bond_type)
            remaining_valences[begin_idx] -= valence
            remaining_valences[end_idx] -= valence

    for candidate in ranked_candidates:
        fragments = Chem.GetMolFrags(rw_mol)
        if len(fragments) <= 1:
            break
        begin_idx = candidate.begin_idx
        end_idx = candidate.end_idx
        in_same_fragment = any(begin_idx in frag and end_idx in frag for frag in fragments)
        if (
            not in_same_fragment
            and remaining_valences[begin_idx] >= _SINGLE_BOND_VALENCE
            and remaining_valences[end_idx] >= _SINGLE_BOND_VALENCE
        ):
            rw_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
            remaining_valences[begin_idx] -= _SINGLE_BOND_VALENCE
            remaining_valences[end_idx] -= _SINGLE_BOND_VALENCE

    return rw_mol.GetMol()


@jt.typed
def build_generated_mol(
    positions: jt.Float[np.ndarray, "max_atoms 3"],
    atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"],
    bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"],
    atom_mask: jt.Bool[np.ndarray, " max_atoms"],
    *,
    cond_atom_features: jt.Float[np.ndarray, "max_atoms num_atom_features"] | None = None,
    atom_cond_mask: jt.Bool[np.ndarray, "max_atoms num_atom_features"] | None = None,
    cond_bond_features: jt.Float[np.ndarray, "max_atoms max_atoms num_bond_features"] | None = None,
    bond_cond_mask: jt.Bool[np.ndarray, "max_atoms max_atoms num_bond_features"] | None = None,
    prompt_smiles: str = "",
) -> Chem.Mol:
    """Reconstructs a heavy-atom RDKit molecule and 3D conformer from predicted graph features.

    Args:
        positions: Padded 3D coordinates in Angstroms.
        atom_features: Padded per-atom continuous/one-hot feature matrix.
        bond_features: Padded pairwise continuous/one-hot bond feature tensor.
        atom_mask: Valid-atom mask.
        cond_atom_features: Optional ground-truth or template atom feature matrix.
        atom_cond_mask: Optional boolean mask of conditioned atom feature channels.
        cond_bond_features: Optional ground-truth or template bond feature tensor.
        bond_cond_mask: Optional boolean mask of conditioned bond feature channels.
        prompt_smiles: Optional molecule name/template identifier for SDF properties.

    Returns:
        RDKit molecule with decoded atom types, bond orders, and 3D conformer attached.
    """
    _disable_rdkit_logging()
    effective_atoms = _resolve_effective_atoms(
        atom_features,
        cond_atom_features=cond_atom_features,
        atom_cond_mask=atom_cond_mask,
    )
    effective_bonds = (
        np.where(
            bond_cond_mask,
            np.where(cond_bond_features > 0.0, 1.0, _MASKED_LOGIT),
            bond_features,
        )
        if cond_bond_features is not None and bond_cond_mask is not None
        else bond_features
    )
    valid_indices = np.flatnonzero(atom_mask)
    mol = _build_raw_generated_mol(
        valid_indices,
        effective_atoms,
        effective_bonds,
        cond_atom_features=cond_atom_features,
        atom_cond_mask=atom_cond_mask,
        cond_bond_features=cond_bond_features,
        bond_cond_mask=bond_cond_mask,
        allow_aromatic=True,
    )
    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        mol = _build_raw_generated_mol(
            valid_indices,
            effective_atoms,
            effective_bonds,
            cond_atom_features=cond_atom_features,
            atom_cond_mask=atom_cond_mask,
            cond_bond_features=cond_bond_features,
            bond_cond_mask=bond_cond_mask,
            allow_aromatic=False,
        )
        Chem.SanitizeMol(mol, catchErrors=True)
    _attach_3d_conformer(mol, positions[valid_indices])
    if mol.GetNumAtoms() > 0:
        Chem.AssignStereochemistryFrom3D(mol)
    resolved_smiles = Chem.MolToSmiles(mol) if mol.GetNumAtoms() > 0 else prompt_smiles
    mol.SetProp(SdfProperty.NAME, prompt_smiles or resolved_smiles)
    mol.SetProp(SdfProperty.SMILES, resolved_smiles)
    return mol


@jt.typed
def build_conformer_mol(
    smiles: str,
    positions: jt.Float[np.ndarray, "max_atoms 3"],
) -> Chem.Mol:
    """Builds a heavy-atom RDKit molecule with a 3D conformer from padded coordinates.

    Args:
        smiles: Input SMILES string defining the molecular graph.
        positions: Padded 3D coordinates in Angstroms ordered by heavy SMILES atoms.

    Returns:
        Heavy-atom RDKit molecule with a single 3D conformer attached.

    Raises:
        ValueError: If the SMILES string is invalid or has more heavy atoms than positions.
    """
    base_mol = _parse_template_smiles(smiles)
    if base_mol is None:
        raise ValueError(f"Failed to parse SMILES {smiles!r}.")
    heavy_mol = Chem.RemoveAllHs(base_mol)
    num_atoms = heavy_mol.GetNumAtoms()
    max_atoms, _ = positions.shape
    if num_atoms > max_atoms:
        raise ValueError(f"Molecule {smiles!r} has {num_atoms} atoms, exceeding {max_atoms=}.")
    _attach_3d_conformer(heavy_mol, positions[:num_atoms])
    heavy_mol.SetProp(SdfProperty.NAME, smiles)
    heavy_mol.SetProp(SdfProperty.SMILES, smiles)
    return heavy_mol


def collate_batch(
    molecules: Collection[schema.Molecule],
    *,
    max_atoms: int = MAX_ATOMS,
) -> Batch:
    """Featurizes and pads a collection of molecules into a fixed-shape heavy-atom batch.

    Args:
        molecules: Non-empty collection of unpadded molecules.
        max_atoms: Padded heavy-atom dimension.

    Returns:
        Padded batch with zero-centered heavy-atom coordinates and valid-atom mask.

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
        heavy_positions = mol.positions[mol.atomic_numbers > 1]
        num_heavy, _ = heavy_positions.shape
        positions[batch_idx, :num_heavy] = center_positions(heavy_positions)

    return dataclasses.replace(graph_batch, positions=positions)
