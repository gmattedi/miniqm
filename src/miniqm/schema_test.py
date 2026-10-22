import dataclasses
import pathlib

import jaxtyping as jt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm import schema


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
    mulliken = (-0.4, 0.1, 0.1, 0.1, 0.1, *(0.0,) * 24)
    return {
        schema.Column.INDEX: 1,
        schema.Column.TAG: "gdb",
        schema.Column.NUM_ATOMS: 5,
        schema.Column.SMILES: "C",
        schema.Column.SMILES_RELAXED: "C",
        schema.Column.CHARGES: charges,
        schema.Column.POSITIONS: positions,
        schema.Column.MULLIKEN_CHARGES: mulliken,
        schema.Column.A: 157.7,
        schema.Column.B: 157.7,
        schema.Column.C: 157.7,
        schema.Column.MU: 0.0,
        schema.Column.ALPHA: 13.2,
        schema.Column.HOMO: -0.38,
        schema.Column.LUMO: 0.11,
        schema.Column.GAP: 0.50,
        schema.Column.R2: 35.3,
        schema.Column.ZPVE: 0.044,
        schema.Column.U0: -40.47,
        schema.Column.U: -40.47,
        schema.Column.H: -40.47,
        schema.Column.G: -40.49,
        schema.Column.CV: 6.46,
        schema.Column.U0_ATOMIZATION: -0.62,
        schema.Column.U_ATOMIZATION: -0.62,
        schema.Column.H_ATOMIZATION: -0.63,
        schema.Column.G_ATOMIZATION: -0.59,
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
        mulliken_charges=np.array([-0.4, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        labels=schema.QM9Labels(
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
    ("atomic_numbers", "positions", "mulliken_charges", "error_match"),
    (
        (
            np.array((), dtype=np.int64),
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((5,), dtype=np.float32),
            "Expected non-empty 1D atomic_numbers",
        ),
        (
            np.ones((5,), dtype=np.int64),
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((5,), dtype=np.float32),
            r"Expected positions shape \(5, 3\)",
        ),
        (
            np.ones((5,), dtype=np.int64),
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
            r"Expected mulliken_charges shape \(5,\)",
        ),
    ),
)
def test_molecule_post_init_validates_shapes(
    atomic_numbers: jt.Int[np.ndarray, "..."],
    positions: jt.Float[np.ndarray, "..."],
    mulliken_charges: jt.Float[np.ndarray, "..."],
    error_match: str,
):
    base_molecule = _make_expected_molecule()

    with pytest.raises(ValueError, match=error_match):
        dataclasses.replace(
            base_molecule,
            atomic_numbers=atomic_numbers,
            positions=positions,
            mulliken_charges=mulliken_charges,
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error_match"),
    (
        (schema.Column.NUM_ATOMS, "five", "Expected int-compatible value"),
        (schema.Column.TAG, 123, "Expected str"),
        (schema.Column.MU, "zero", "Expected float-compatible value"),
    ),
)
def test_molecule_from_row_rejects_invalid_scalar_types(
    field_name: str,
    bad_value: object,
    error_match: str,
):
    row = {**_make_sample_row(), field_name: bad_value}

    with pytest.raises(TypeError, match=error_match):
        schema.Molecule.from_row(row)
