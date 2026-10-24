from collections.abc import Mapping
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm import runner, schema, train


def _write_split_parquet(path: pathlib.Path, num_rows: int = 4) -> pathlib.Path:
    """Writes a synthetic QM9 Parquet file with `num_rows` 2-atom HF molecules."""
    charges = (9, 1, *(0,) * 27)
    positions = ((0.0, 0.0, 0.0), (0.92, 0.0, 0.0), *((0.0, 0.0, 0.0),) * 27)
    mulliken = (-0.4, 0.4, *(0.0,) * 27)
    rows = tuple(
        {
            schema.Column.INDEX: row_idx + 1,
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
            schema.Column.MU: 1.8,
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
        for row_idx in range(num_rows)
    )
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_main_cli_writes_tensorboard_events_checkpoints_and_metrics(
    tmp_path: pathlib.Path,
) -> None:
    parquet_path = _write_split_parquet(tmp_path / "split.parquet", num_rows=4)
    run_dir = tmp_path / "cli_run"

    runner.main(
        (
            "--train-path",
            str(parquet_path),
            "--val-path",
            str(parquet_path),
            "--run-dir",
            str(run_dir),
            "--batch-size",
            "2",
            "--num-steps",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "2",
            "--sample-every",
            "2",
            "--train.model.transformer.hidden-dim",
            "16",
            "--train.model.transformer.num-layers",
            "1",
            "--train.model.transformer.num-heads",
            "2",
            "--sampling.num-steps",
            "3",
        )
    )

    assert (run_dir / "config.json").is_file()
    assert (run_dir / "checkpoints" / "latest.msgpack").is_file()
    assert (run_dir / "checkpoints" / "best.msgpack").is_file()

    metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = tuple(json.loads(line) for line in metrics_lines)
    splits_logged = {record["split"] for record in records}
    assert splits_logged == {"train", "val", "val_sample"}

    event_files = tuple(run_dir.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    (
        ({"batch_size": 0}, "positive batch_size"),
        ({"num_steps": 0}, "positive num_steps"),
        ({"log_every": 0}, "positive log_every"),
        ({"eval_every": 0}, "positive eval_every"),
        ({"sample_every": 0}, "positive sample_every"),
        ({"sample_batch_size": 0}, "positive sample_batch_size"),
    ),
)
def test_run_config_rejects_non_positive_counts(
    kwargs: Mapping[str, int],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        train.RunConfig(
            batch_size=kwargs.get("batch_size", 64),
            num_steps=kwargs.get("num_steps", 10),
            log_every=kwargs.get("log_every", 5),
            eval_every=kwargs.get("eval_every", 5),
            sample_every=kwargs.get("sample_every", 5),
            sample_batch_size=kwargs.get("sample_batch_size", 8),
        )
