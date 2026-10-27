# miniqm Package

`miniqm` is a JAX library for heavy-atom QM9 graph featurization, Karras EDM joint 2D+3D molecular diffusion, Optax training, and interactive 3D molecular visualization:

* `data/` converts QM9 splits and SMILES/SMARTS templates into padded heavy-atom molecular graph batches, per-channel conditioning masks, and device arrays.
* `diffusion/` implements 3D rigid-body geometry, the Karras noise schedule, random subgraph conditioning masks, the pair-biased Transformer denoiser, Kabsch-aligned coordinate and one-hot 2D losses, evaluation metrics, and reverse Euler sampling.
* `training/` manages optimizer schedules, exponential moving average (EMA) weights, checkpoint serialization, validation metrics, and the training loop.
* Top-level modules define Pydantic model and training configurations, runtime shape-checked array type aliases, CLI entrypoints, and `py3Dmol` 3D viewers for single molecules, conformer overlays, and diffusion trajectories.
