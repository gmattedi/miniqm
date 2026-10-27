# Training, Evaluation & Checkpointing

`miniqm.training` manages optimization state, multi-device CPU batch sharding, periodic validation, conformer sampling benchmarks, and MessagePack checkpointing:

* Runs the compiled training step with global-norm-clipped AdamW and a warmup cosine decay schedule, shards batches across logical XLA CPU devices, and updates exponential moving average (EMA) weights at every step.
* Writes each training run into a timestamped subdirectory, logs parameter counts and scalar loss components to TensorBoard and JSON Lines, and saves the best checkpoint when mean validation sampled lDDT improves.
* Evaluates molecule-weighted validation losses, validation conformer sampling metrics, full test-split benchmarks, and multi-conformer SMILES/template-to-3D SDF generation.
* Resolves direct or timestamped parent run directories and serializes model and EMA parameters with atomic file replacement.
