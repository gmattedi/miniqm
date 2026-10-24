import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm.data import batch, data_source, schema


def _make_sample_row(*, index: int, mu: float) -> dict[str, object]:
    """Builds a padded 29-slot QM9 row dictionary for a 2-atom HF molecule."""
    charges = (9, 1, *(0,) * 27)
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        *((0.0, 0.0, 0.0),) * 27,
    )
    mulliken = (-0.4, 0.4, *(0.0,) * 27)
    return {
        schema.Column.INDEX: index,
        schema.Column.TAG: "gdb",
        schema.Column.NUM_ATOMS: 2,
        schema.Column.SMILES: "F",
        schema.Column.SMILES_RELAXED: "F",
        schema.Column.CHARGES: charges,
        schema.Column.POSITIONS: positions,
        schema.Column.MULLIKEN_CHARGES: mulliken,
        schema.Column.A: 157.7,
        schema.Column.B: 157.7,
        schema.Column.C: 157.7,
        schema.Column.MU: mu,
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


def _write_sample_parquet(tmp_path: pathlib.Path, num_rows: int = 3) -> pathlib.Path:
    """Writes a synthetic QM9 Parquet file with num_rows HF molecules."""
    parquet_path = tmp_path / "split.parquet"
    rows = tuple(
        _make_sample_row(index=row_idx + 1, mu=float(row_idx)) for row_idx in range(num_rows)
    )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return parquet_path


def test_data_source_loads_and_iterates_batches(tmp_path: pathlib.Path) -> None:
    parquet_path = _write_sample_parquet(tmp_path, num_rows=3)
    molecules = schema.load_molecules(parquet_path)

    source = data_source.DataSource(parquet_path, max_atoms=4)
    batches = tuple(source.iter_batches(batch_size=2, drop_last=False))

    assert len(source) == 3
    assert batches == (
        batch.collate_batch(molecules[:2], max_atoms=4),
        batch.collate_batch(molecules[2:], max_atoms=4),
    )


def test_data_source_drop_last_and_max_molecules(tmp_path: pathlib.Path) -> None:
    parquet_path = _write_sample_parquet(tmp_path, num_rows=3)

    capped_source = data_source.DataSource(parquet_path, max_atoms=4, max_molecules=2)
    full_source = data_source.DataSource(parquet_path, max_atoms=4)
    dropped_batches = tuple(
        full_source.iter_batches(
            batch_size=2,
            rng=np.random.default_rng(0),
            drop_last=True,
        )
    )

    assert len(capped_source) == 2
    assert len(dropped_batches) == 1
    assert dropped_batches[0].positions.shape == (2, 4, 3)


def test_data_source_rejects_invalid_arguments(tmp_path: pathlib.Path) -> None:
    parquet_path = _write_sample_parquet(tmp_path, num_rows=2)

    with pytest.raises(ValueError, match="Expected positive max_molecules"):
        data_source.DataSource(parquet_path, max_molecules=0)

    source = data_source.DataSource(parquet_path, max_atoms=4)
    with pytest.raises(ValueError, match="Expected positive batch_size"):
        next(source.iter_batches(batch_size=0))
