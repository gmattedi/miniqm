"""Data structures for unpadded molecules and their 3D atomic coordinates."""

from collections.abc import Mapping
import dataclasses
import enum
import pathlib
from typing import Self, override

import numpy as np
import pyarrow.parquet as pq

from miniqm import jaxtypes as jt


class Split(enum.StrEnum):
    """Standard dataset split names."""

    TRAIN = enum.auto()
    VALIDATION = enum.auto()
    TEST = enum.auto()


class Column(enum.StrEnum):
    """Column names in the molecule Parquet splits."""

    INDEX = enum.auto()
    TAG = enum.auto()
    NUM_ATOMS = enum.auto()
    SMILES = enum.auto()
    SMILES_RELAXED = enum.auto()
    CHARGES = enum.auto()
    POSITIONS = enum.auto()


@jt.typed
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Molecule:
    """Single molecule with unpadded atomic numbers and 3D coordinates."""

    index: int  # 1-based molecule index.
    tag: str
    smiles: str
    smiles_relaxed: str
    atomic_numbers: jt.Int[np.ndarray, "num_atoms"]
    positions: jt.Float[np.ndarray, "num_atoms 3"]  # Angstroms.

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
            and np.array_equal(self.atomic_numbers, other.atomic_numbers)
            and np.array_equal(self.positions, other.positions)
        )

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return int(self.atomic_numbers.size)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        """Constructs an unpadded molecule from a padded Parquet row mapping.

        Args:
            row: Mapping keyed by Parquet column names.

        Returns:
            Unpadded molecule instance with sliced atom arrays.
        """
        num_atoms = _as_int(row[Column.NUM_ATOMS])
        atomic_numbers = np.asarray(row[Column.CHARGES], dtype=np.int64)[:num_atoms]
        positions = np.asarray(row[Column.POSITIONS], dtype=np.float32)[:num_atoms]
        return cls(
            index=_as_int(row[Column.INDEX]),
            tag=_as_str(row[Column.TAG]),
            smiles=_as_str(row[Column.SMILES]),
            smiles_relaxed=_as_str(row[Column.SMILES_RELAXED]),
            atomic_numbers=atomic_numbers,
            positions=positions,
        )


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
    """Loads molecules from a split Parquet file.

    Args:
        parquet_path: Path to a split Parquet file.
        max_molecules: Optional cap on the number of rows to load.

    Returns:
        Unpadded molecules in row order.

    Raises:
        ValueError: If max_molecules is not positive.
    """
    if max_molecules is not None and max_molecules <= 0:
        raise ValueError(f"Expected positive max_molecules, got {max_molecules}.")
    table = pq.read_table(parquet_path, columns=list(Column))
    if max_molecules is not None:
        table = table.slice(0, max_molecules)
    return tuple(Molecule.from_row(row) for row in table.to_pylist())
