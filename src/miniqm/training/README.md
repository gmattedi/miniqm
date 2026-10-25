<!-- AGENT NOTICE: Keep this README synchronized with the training, evaluation, and checkpointing implementations in this directory whenever code changes. -->

# Training, Evaluation & Checkpointing

`miniqm.training` manages optimization state, multi-device CPU batch sharding, periodic validation, conformer sampling benchmarks, and MessagePack checkpointing:

* `train.py` defines `TrainState` and runs the compiled training step (`train_step`) with global-norm-clipped AdamW (`warmup_cosine_decay_schedule`), shards training batches across logical XLA CPU devices via `NamedSharding`, and updates exponential moving average (`ema_params`) weights at every step.
* `train.py` writes each training run into a timestamped subdirectory (`<run_dir>/<YYYYMMDD_HHMMSS>/`), logs submodule parameter counts and a Flax `tabulate` summary table, displays a live Rich progress bar, logs scalar metrics (coordinate, distance, atom, and bond losses) to TensorBoard and `metrics.jsonl`, and saves `best.msgpack` when validation `sampled_lddt_mean` improves.
* `eval.py` implements molecule-weighted validation loss evaluation (`evaluate_loss`), validation conformer sampling (`evaluate_dataset_sampling`), full-split test evaluation (`run_evaluation`), and multi-conformer SMILES/template-to-3D SDF generation (`run_smiles_sampling`).
* `checkpoint.py` resolves direct or timestamped parent run directories (`resolve_run_dir`), serializes and restores `Checkpoint` states (`step`, `params`, `ema_params`) via `flax.serialization` with atomic file replacement, and reconstructs full `LoadedRun` bundles from `config.json`.
