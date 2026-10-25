import dataclasses
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm import jaxtypes as jt
from miniqm.data import schema


def _make_sample_row() -> dict[str, object]:
    charges = (6, 1, 1, 1, 1, *(0,) * 24)
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        *((0.0, 0.0, 0.0),) * 24,
    )
    return {
        schema.Column.INDEX: 1,
        schema.Column.TAG: "gdb",
        schema.Column.NUM_ATOMS: 5,
        schema.Column.SMILES: "C",
        schema.Column.SMILES_RELAXED: "C",
        schema.Column.CHARGES: charges,
        schema.Column.POSITIONS: positions,
    }


def _make_expected_molecule() -> schema.Molecule:
    return schema.Molecule(
        index=1,
        tag="gdb",
        smiles="C",
        smiles_relaxed="C",
        atomic_numbers=np.array([6, 1, 1, 1, 1], dtype=np.int64),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_molecule_from_row_slices_padding():
    molecule = schema.Molecule.from_row(_make_sample_row())

    assert molecule == _make_expected_molecule()
    assert molecule.num_atoms == 5


def test_load_molecules_roundtrip(tmp_path: pathlib.Path):
    parquet_path = tmp_path / "sample.parquet"
    table = pa.Table.from_pylist((_make_sample_row(),))
    pq.write_table(table, parquet_path)

    molecules = schema.load_molecules(parquet_path)

    assert molecules == (_make_expected_molecule(),)


@pytest.mark.parametrize(
    ("atomic_numbers", "positions", "expected_error"),
    (
        (
            np.array((), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
            ValueError,
        ),
        (
            np.ones((5,), dtype=np.int64),
            np.zeros((4, 3), dtype=np.float32),
            jt.TypeCheckError,
        ),
    ),
    ids=("empty_atoms", "mismatched_positions"),
)
@jt.typed
def test_molecule_validates_shapes(
    atomic_numbers: jt.Int[np.ndarray, "..."],
    positions: jt.Float[np.ndarray, "..."],
    expected_error: type[Exception],
):
    base_molecule = _make_expected_molecule()

    with pytest.raises(expected_error, match=r"atomic_numbers|positions"):
        dataclasses.replace(
            base_molecule,
            atomic_numbers=atomic_numbers,
            positions=positions,
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error_match"),
    (
        (schema.Column.NUM_ATOMS, "five", "Expected int-compatible value"),
        (schema.Column.TAG, 123, "Expected str"),
    ),
    ids=("bad_int", "bad_str"),
)
def test_molecule_from_row_rejects_invalid_scalar_types(
    field_name: str,
    bad_value: object,
    error_match: str,
):
    row = {**_make_sample_row(), field_name: bad_value}

    with pytest.raises(TypeError, match=error_match):
        schema.Molecule.from_row(row)
