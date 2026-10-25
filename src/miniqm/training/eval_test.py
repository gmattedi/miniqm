import json
import pathlib
from typing import Final

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from miniqm import config as config_lib
from miniqm.data import batch as batch_lib
from miniqm.data import data_source, schema
from miniqm.training import checkpoint, train
from miniqm.training import eval as eval_lib

_TINY_MODEL_CONFIG: Final = config_lib.DenoiserConfig(
    transformer=config_lib.TransformerConfig(hidden_dim=16, num_layers=1, num_heads=2),
)


def _write_sample_parquet(path: pathlib.Path, *, num_rows: int = 3) -> pathlib.Path:
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


def test_evaluate_loss_and_sampling_rmsd_return_finite_metrics(tmp_path: pathlib.Path):
    parquet_path = _write_sample_parquet(tmp_path / "val.parquet", num_rows=3)
    val_source = data_source.DataSource(parquet_path, max_atoms=4)
    val_batch = next(val_source.iter_batches(batch_size=3, drop_last=False)).to_jax()
    state = train.init_train_state(
        jax.random.key(0),
        val_batch,
        config=config_lib.TrainConfig(model=_TINY_MODEL_CONFIG),
    )

    val_metrics = eval_lib.evaluate_loss(
        state.ema_params,
        jax.random.key(1),
        val_source,
        batch_size=2,
        model_config=_TINY_MODEL_CONFIG,
        loss_config=config_lib.DEFAULT_LOSS_CONFIG,
    )
    sampling_metrics = eval_lib.evaluate_sampling_rmsd(
        state.ema_params,
        jax.random.key(2),
        val_batch,
        model_config=_TINY_MODEL_CONFIG,
        sampling_config=config_lib.SamplingConfig(num_steps=3),
    )

    assert np.isfinite(val_metrics.total_loss)
    assert np.isfinite(val_metrics.rmsd)
    assert val_metrics.rmsd >= 0.0
    assert np.isfinite(sampling_metrics.sampled_rmsd_mean)
    assert np.isfinite(sampling_metrics.sampled_rmsd_median)
    assert sampling_metrics.sampled_rmsd_mean >= 0.0
    assert 0.0 <= sampling_metrics.sampled_lddt_mean <= 1.0
    assert 0.0 <= sampling_metrics.sampled_lddt_median <= 1.0
    assert 0.0 <= sampling_metrics.stereocenter_violation_rate <= 1.0


def test_run_evaluation_and_smiles_sampling_write_outputs(tmp_path: pathlib.Path):
    parquet_path = _write_sample_parquet(tmp_path / "test.parquet", num_rows=3)
    test_source = data_source.DataSource(parquet_path)
    first_batch = next(test_source.iter_batches(batch_size=3, drop_last=False)).to_jax()
    train_config = config_lib.TrainConfig(model=_TINY_MODEL_CONFIG)
    state = train.init_train_state(jax.random.key(0), first_batch, config=train_config)

    run_dir = tmp_path / "run"
    train.write_run_config(
        config_lib.RunConfig(
            run_dir=run_dir,
            train=train_config,
            sampling=config_lib.SamplingConfig(num_steps=3),
        )
    )
    checkpoint.save_checkpoint(
        run_dir / "checkpoints" / config_lib.CheckpointName.BEST,
        checkpoint.Checkpoint(step=5, params=state.params, ema_params=state.ema_params),
    )

    report = eval_lib.run_evaluation(
        config_lib.EvalConfig(
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
    assert 0.0 <= report.sampling_metrics.sampled_lddt_mean <= 1.0
    assert 0.0 <= report.sampling_metrics.fraction_below_0_5a <= 1.0
    assert 0.0 <= report.sampling_metrics.fraction_below_1_0a <= 1.0
    assert 0.0 <= report.sampling_metrics.stereocenter_violation_rate <= 1.0
    saved_report = json.loads(
        (run_dir / config_lib.ArtifactName.TEST_METRICS).read_text(encoding="utf-8")
    )
    assert saved_report["step"] == 5

    smiles_file = tmp_path / "molecules.smi"
    smiles_file.write_text("F\nCO\nC~U\n", encoding="utf-8")
    output_sdf = tmp_path / "out.sdf"
    mols = eval_lib.run_smiles_sampling(
        config_lib.SampleSmilesConfig(
            run_dir=run_dir,
            smiles_file=smiles_file,
            num_conformers=2,
            num_sampling_steps=2,
            output_sdf=output_sdf,
            seed=10,
        )
    )
    assert len(mols) == 6
    assert output_sdf.is_file()
    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False)
    loaded_mols = tuple(supplier)
    assert len(loaded_mols) == 6
    assert loaded_mols[0].GetNumAtoms() == 1
    assert loaded_mols[0].GetProp(batch_lib.SdfProperty.CONFORMER_ID) == "0"
    assert loaded_mols[0].GetProp(batch_lib.SdfProperty.SEED) == "10"
    assert loaded_mols[1].GetNumAtoms() == 1
    assert loaded_mols[1].GetProp(batch_lib.SdfProperty.CONFORMER_ID) == "1"
    assert loaded_mols[1].GetProp(batch_lib.SdfProperty.SEED) == "11"
    assert loaded_mols[2].GetNumAtoms() == 2
    assert loaded_mols[4].GetNumAtoms() == 2
    assert loaded_mols[4].GetAtomWithIdx(0).GetSymbol() == "C"
    assert loaded_mols[4].GetBondBetweenAtoms(0, 1) is not None


def test_resolve_smiles_inputs_validates_sources(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="Expected non-empty smiles"):
        eval_lib._resolve_smiles_inputs(config_lib.SampleSmilesConfig(smiles=()))

    with pytest.raises(FileNotFoundError):
        eval_lib._resolve_smiles_inputs(
            config_lib.SampleSmilesConfig(smiles_file=tmp_path / "missing.smi")
        )

    empty_file = tmp_path / "empty.smi"
    empty_file.write_text("\n   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected at least one SMILES"):
        eval_lib._resolve_smiles_inputs(config_lib.SampleSmilesConfig(smiles_file=empty_file))
