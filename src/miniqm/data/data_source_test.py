import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm.data import batch, data_source, schema


def _make_sample_row(*, index: int) -> dict[str, object]:
    charges = (9, 1, *(0,) * 27)
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        *((0.0, 0.0, 0.0),) * 27,
    )
    return {
        schema.Column.INDEX: index,
        schema.Column.TAG: "gdb",
        schema.Column.NUM_ATOMS: 2,
        schema.Column.SMILES: "F",
        schema.Column.SMILES_RELAXED: "F",
        schema.Column.CHARGES: charges,
        schema.Column.POSITIONS: positions,
    }


def _write_sample_parquet(tmp_path: pathlib.Path, *, num_rows: int = 3) -> pathlib.Path:
    parquet_path = tmp_path / "split.parquet"
    rows = tuple(_make_sample_row(index=row_idx + 1) for row_idx in range(num_rows))
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return parquet_path


def test_data_source_loads_and_iterates_batches(tmp_path: pathlib.Path):
    parquet_path = _write_sample_parquet(tmp_path, num_rows=3)
    molecules = schema.load_molecules(parquet_path)

    source = data_source.DataSource(parquet_path, max_atoms=4)
    batches = tuple(source.iter_batches(batch_size=2, drop_last=False))

    assert len(source) == 3
    assert (tmp_path / "split_collated_4.npz").is_file()
    assert batches == (
        batch.collate_batch(molecules[:2], max_atoms=4),
        batch.collate_batch(molecules[2:], max_atoms=4),
    )

    cached_source = data_source.DataSource(parquet_path, max_atoms=4)
    assert tuple(cached_source.iter_batches(batch_size=2, drop_last=False)) == batches


def test_collate_molecules_parallel_chunks(tmp_path: pathlib.Path):
    parquet_path = _write_sample_parquet(tmp_path, num_rows=3)
    molecules = schema.load_molecules(parquet_path)

    collated = data_source._collate_molecules(
        molecules,
        max_atoms=4,
        max_workers=2,
        parallel_min_molecules=2,
        chunk_size=1,
    )

    assert collated == batch.collate_batch(molecules, max_atoms=4)


def test_data_source_drop_last_and_max_molecules(tmp_path: pathlib.Path):
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


def test_data_source_rejects_invalid_arguments(tmp_path: pathlib.Path):
    parquet_path = _write_sample_parquet(tmp_path, num_rows=2)

    with pytest.raises(ValueError, match="Expected positive max_molecules"):
        data_source.DataSource(parquet_path, max_molecules=0)

    source = data_source.DataSource(parquet_path, max_atoms=4)
    with pytest.raises(ValueError, match="Expected positive batch_size"):
        next(source.iter_batches(batch_size=0))
