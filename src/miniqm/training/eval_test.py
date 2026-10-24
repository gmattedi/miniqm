import json
import pathlib
from typing import Final

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniqm.data import data_source, schema
from miniqm.diffusion import losses, model, sampling
from miniqm.training import checkpoint, train
from miniqm.training import eval as eval_lib

_TINY_MODEL_CONFIG: Final = model.DenoiserConfig(
    transformer=model.TransformerConfig(hidden_dim=16, num_layers=1, num_heads=2),
)


def _write_sample_parquet(path: pathlib.Path, num_rows: int = 3) -> pathlib.Path:
    """Writes a synthetic 2-atom HF QM9 Parquet split."""
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


def test_evaluate_loss_and_sampling_rmsd_return_finite_metrics(tmp_path: pathlib.Path) -> None:
    parquet_path = _write_sample_parquet(tmp_path / "val.parquet", num_rows=3)
    val_source = data_source.DataSource(parquet_path, max_atoms=4)
    val_batch = next(val_source.iter_batches(batch_size=3, drop_last=False)).to_jax()
    state = train.init_train_state(
        jax.random.key(0),
        val_batch,
        config=train.TrainConfig(model=_TINY_MODEL_CONFIG),
    )

    val_metrics = eval_lib.evaluate_loss(
        state.ema_params,
        jax.random.key(1),
        val_source,
        batch_size=2,
        model_config=_TINY_MODEL_CONFIG,
        loss_config=losses.DEFAULT_LOSS_CONFIG,
    )
    sampling_metrics = eval_lib.evaluate_sampling_rmsd(
        state.ema_params,
        jax.random.key(2),
        val_batch,
        model_config=_TINY_MODEL_CONFIG,
        sampling_config=sampling.SamplingConfig(num_steps=3),
    )

    assert np.isfinite(val_metrics.total_loss)
    assert np.isfinite(val_metrics.rmsd)
    assert val_metrics.rmsd >= 0.0
    assert np.isfinite(sampling_metrics.sampled_rmsd_mean)
    assert np.isfinite(sampling_metrics.sampled_rmsd_median)
    assert sampling_metrics.sampled_rmsd_mean >= 0.0


def test_run_evaluation_and_smiles_sampling_write_outputs(tmp_path: pathlib.Path) -> None:
    parquet_path = _write_sample_parquet(tmp_path / "test.parquet", num_rows=3)
    test_source = data_source.DataSource(parquet_path)
    first_batch = next(test_source.iter_batches(batch_size=3, drop_last=False)).to_jax()
    train_config = train.TrainConfig(model=_TINY_MODEL_CONFIG)
    state = train.init_train_state(jax.random.key(0), first_batch, config=train_config)

    run_dir = tmp_path / "run"
    train.write_run_config(
        train.RunConfig(
            run_dir=run_dir,
            train=train_config,
            sampling=sampling.SamplingConfig(num_steps=3),
        )
    )
    checkpoint.save_checkpoint(
        run_dir / "checkpoints" / "best.msgpack",
        checkpoint.Checkpoint(step=5, params=state.params, ema_params=state.ema_params),
    )

    report = eval_lib.run_evaluation(
        eval_lib.EvalConfig(
            run_dir=run_dir,
            test_path=parquet_path,
            batch_size=2,
            num_sampling_steps=2,
        )
    )
    assert report.step == 5
    assert report.num_molecules == 3
    assert np.isfinite(report.loss_metrics.total_loss)
    assert np.isfinite(report.sampling_metrics.sampled_rmsd_mean)
    assert 0.0 <= report.sampling_metrics.fraction_below_0_5a <= 1.0
    assert 0.0 <= report.sampling_metrics.fraction_below_1_0a <= 1.0
    saved_report = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved_report["step"] == 5

    smiles_file = tmp_path / "molecules.smi"
    smiles_file.write_text("F\nO\n", encoding="utf-8")
    output_xyz = tmp_path / "out.xyz"
    xyz_content = eval_lib.run_smiles_sampling(
        eval_lib.SampleSmilesConfig(
            run_dir=run_dir,
            smiles_file=smiles_file,
            num_sampling_steps=2,
            output_xyz=output_xyz,
        )
    )
    assert output_xyz.is_file()
    assert xyz_content.startswith("2\nF\n")
    assert "\n3\nO\n" in xyz_content


def test_eval_and_sample_configs_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive batch_size"):
        eval_lib.EvalConfig(batch_size=0)

    with pytest.raises(ValueError, match="positive num_sampling_steps"):
        eval_lib.EvalConfig(num_sampling_steps=0)

    with pytest.raises(ValueError, match="positive num_sampling_steps"):
        eval_lib.SampleSmilesConfig(num_sampling_steps=0)
