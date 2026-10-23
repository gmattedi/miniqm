"""Data structures for single unpadded QM9 molecules and their quantum chemical labels."""

from collections.abc import Mapping
import dataclasses
import enum
import pathlib
from typing import Self, override

import numpy as np
import pyarrow.parquet as pq

from miniqm import jaxtypes as jt


class Column(enum.StrEnum):
    """Column names in the QM9 Parquet splits."""

    INDEX = enum.auto()
    TAG = enum.auto()
    NUM_ATOMS = enum.auto()
    SMILES = enum.auto()
    SMILES_RELAXED = enum.auto()
    CHARGES = enum.auto()
    POSITIONS = enum.auto()
    MULLIKEN_CHARGES = enum.auto()
    A = enum.auto()
    B = enum.auto()
    C = enum.auto()
    MU = enum.auto()
    ALPHA = enum.auto()
    HOMO = enum.auto()
    LUMO = enum.auto()
    GAP = enum.auto()
    R2 = enum.auto()
    ZPVE = enum.auto()
    U0 = enum.auto()
    U = enum.auto()
    H = enum.auto()
    G = enum.auto()
    CV = enum.auto()
    U0_ATOMIZATION = enum.auto()
    U_ATOMIZATION = enum.auto()
    H_ATOMIZATION = enum.auto()
    G_ATOMIZATION = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QM9Labels:
    """Scalar quantum chemical properties from the QM9 dataset (Ramakrishnan et al., 2014)."""

    rotational_a: float  # GHz.
    rotational_b: float  # GHz.
    rotational_c: float  # GHz.
    mu: float  # Dipole moment (Debye).
    alpha: float  # Isotropic polarizability (bohr^3).
    homo: float  # HOMO energy (Hartree).
    lumo: float  # LUMO energy (Hartree).
    gap: float  # HOMO-LUMO gap (Hartree).
    r2: float  # Electronic spatial extent (bohr^2).
    zpve: float  # Zero-point vibrational energy (Hartree).
    energy_u0: float  # Internal energy at 0 K (Hartree).
    energy_u: float  # Internal energy at 298.15 K (Hartree).
    enthalpy_h: float  # Enthalpy at 298.15 K (Hartree).
    free_energy_g: float  # Free energy at 298.15 K (Hartree).
    heat_capacity_cv: float  # Heat capacity at 298.15 K (cal / (mol K)).
    u0_atomization: float  # Atomization energy at 0 K (Hartree).
    u_atomization: float  # Atomization energy at 298.15 K (Hartree).
    h_atomization: float  # Atomization enthalpy at 298.15 K (Hartree).
    g_atomization: float  # Atomization free energy at 298.15 K (Hartree).

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        """Builds scalar labels from a QM9 row mapping.

        Args:
            row: Mapping keyed by QM9 Parquet column names.

        Returns:
            Parsed scalar quantum chemical properties.
        """
        return cls(
            rotational_a=_as_float(row[Column.A]),
            rotational_b=_as_float(row[Column.B]),
            rotational_c=_as_float(row[Column.C]),
            mu=_as_float(row[Column.MU]),
            alpha=_as_float(row[Column.ALPHA]),
            homo=_as_float(row[Column.HOMO]),
            lumo=_as_float(row[Column.LUMO]),
            gap=_as_float(row[Column.GAP]),
            r2=_as_float(row[Column.R2]),
            zpve=_as_float(row[Column.ZPVE]),
            energy_u0=_as_float(row[Column.U0]),
            energy_u=_as_float(row[Column.U]),
            enthalpy_h=_as_float(row[Column.H]),
            free_energy_g=_as_float(row[Column.G]),
            heat_capacity_cv=_as_float(row[Column.CV]),
            u0_atomization=_as_float(row[Column.U0_ATOMIZATION]),
            u_atomization=_as_float(row[Column.U_ATOMIZATION]),
            h_atomization=_as_float(row[Column.H_ATOMIZATION]),
            g_atomization=_as_float(row[Column.G_ATOMIZATION]),
        )


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Molecule:
    """Single QM9 molecule with unpadded per-atom arrays and scalar labels."""

    index: int  # 1-based GDB9 index.
    tag: str
    smiles: str
    smiles_relaxed: str
    atomic_numbers: jt.Int[np.ndarray, "num_atoms"]
    positions: jt.Float[np.ndarray, "num_atoms 3"]  # Angstroms.
    mulliken_charges: jt.Float[np.ndarray, "num_atoms"]  # Elementary charge.
    labels: QM9Labels

    __hash__ = None

    def __post_init__(self) -> None:
        if self.atomic_numbers.size == 0:
            raise ValueError("Expected non-empty atomic_numbers.")

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented
        return (
            self.index == other.index
            and self.tag == other.tag
            and self.smiles == other.smiles
            and self.smiles_relaxed == other.smiles_relaxed
            and self.labels == other.labels
            and np.array_equal(self.atomic_numbers, other.atomic_numbers)
            and np.array_equal(self.positions, other.positions)
            and np.array_equal(self.mulliken_charges, other.mulliken_charges)
        )

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return int(self.atomic_numbers.size)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        """Constructs an unpadded molecule from a padded QM9 row mapping.

        Args:
            row: Mapping keyed by QM9 Parquet column names.

        Returns:
            Unpadded molecule instance with sliced atom arrays.
        """
        num_atoms = _as_int(row[Column.NUM_ATOMS])
        atomic_numbers = np.asarray(row[Column.CHARGES], dtype=np.int64)[:num_atoms]
        positions = np.asarray(row[Column.POSITIONS], dtype=np.float32)[:num_atoms]
        mulliken_charges = np.asarray(row[Column.MULLIKEN_CHARGES], dtype=np.float32)[:num_atoms]
        return cls(
            index=_as_int(row[Column.INDEX]),
            tag=_as_str(row[Column.TAG]),
            smiles=_as_str(row[Column.SMILES]),
            smiles_relaxed=_as_str(row[Column.SMILES_RELAXED]),
            atomic_numbers=atomic_numbers,
            positions=positions,
            mulliken_charges=mulliken_charges,
            labels=QM9Labels.from_row(row),
        )


def _as_float(value: object) -> float:
    """Converts a numeric scalar to a Python float."""
    if isinstance(value, int | float | np.floating):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {value!r}.")


def _as_int(value: object) -> int:
    """Converts an integer scalar to a Python int."""
    if isinstance(value, int | np.integer):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {value!r}.")


def _as_str(value: object) -> str:
    """Validates that a value is a string."""
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected str, got {value!r}.")


def load_molecules(
    parquet_path: pathlib.Path,
    *,
    max_molecules: int | None = None,
) -> tuple[Molecule, ...]:
    """Loads molecules from a QM9 split Parquet file.

    Args:
        parquet_path: Path to a QM9 split Parquet file.
        max_molecules: Optional cap on the number of rows to load.

    Returns:
        Unpadded molecules in row order.
    """
    if max_molecules is not None and max_molecules <= 0:
        raise ValueError(f"Expected positive max_molecules, got {max_molecules}.")
    table = pq.read_table(parquet_path)
    if max_molecules is not None:
        table = table.slice(0, max_molecules)
    return tuple(Molecule.from_row(row) for row in table.to_pylist())
