from collections.abc import Mapping
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm import config as config_lib
from miniqm import runner
from miniqm.data import schema
from miniqm.training import checkpoint, train


def _write_split_parquet(path: pathlib.Path, num_rows: int = 4) -> pathlib.Path:
    charges = (9, 1, *(0,) * 27)
    positions = ((0.0, 0.0, 0.0), (0.92, 0.0, 0.0), *((0.0, 0.0, 0.0),) * 27)
    rows = tuple(
        {
            schema.Column.INDEX: row_idx + 1,
            schema.Column.TAG: "gdb",
            schema.Column.NUM_ATOMS: 2,
            schema.Column.SMILES: "F",
            schema.Column.SMILES_RELAXED: "F",
            schema.Column.CHARGES: charges,
            schema.Column.POSITIONS: positions,
        }
        for row_idx in range(num_rows)
    )
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_main_cli_writes_tensorboard_events_checkpoints_and_metrics(
    tmp_path: pathlib.Path,
):
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

    active_run_dir = checkpoint.resolve_run_dir(run_dir)
    assert active_run_dir.parent == run_dir
    assert (active_run_dir / config_lib.ArtifactName.CONFIG).is_file()
    assert (active_run_dir / "checkpoints" / config_lib.CheckpointName.LATEST).is_file()
    assert (active_run_dir / "checkpoints" / config_lib.CheckpointName.BEST).is_file()

    metrics_lines = (
        (active_run_dir / config_lib.ArtifactName.METRICS)
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    records = tuple(json.loads(line) for line in metrics_lines)
    splits_logged = {record["split"] for record in records}
    assert splits_logged == {
        train.MetricSplit.TRAIN,
        train.MetricSplit.VAL,
        train.MetricSplit.VAL_SAMPLE,
    }

    event_files = tuple(active_run_dir.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1

    runner.eval_main(
        (
            "--run-dir",
            str(run_dir),
            "--test-path",
            str(parquet_path),
            "--batch-size",
            "2",
            "--num-sampling-steps",
            "2",
        )
    )
    assert (active_run_dir / config_lib.ArtifactName.TEST_METRICS).is_file()

    sdf_path = tmp_path / "sampled.sdf"
    runner.sample_main(
        (
            "--run-dir",
            str(run_dir),
            "--smiles",
            "F",
            "O",
            "--num-sampling-steps",
            "2",
            "--output-sdf",
            str(sdf_path),
        )
    )
    assert sdf_path.is_file()
    assert "$$$$" in sdf_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"batch_size": 0},
        {"num_steps": 0},
        {"log_every": 0},
        {"eval_every": 0},
        {"sample_every": 0},
        {"sample_batch_size": 0},
    ),
    ids=(
        "batch_size",
        "num_steps",
        "log_every",
        "eval_every",
        "sample_every",
        "sample_batch_size",
    ),
)
def test_run_config_rejects_non_positive_counts(
    kwargs: Mapping[str, int],
):
    with pytest.raises(ValueError, match="greater than 0"):
        config_lib.RunConfig(
            batch_size=kwargs.get("batch_size", 64),
            num_steps=kwargs.get("num_steps", 10),
            log_every=kwargs.get("log_every", 5),
            eval_every=kwargs.get("eval_every", 5),
            sample_every=kwargs.get("sample_every", 5),
            sample_batch_size=kwargs.get("sample_batch_size", 8),
        )
