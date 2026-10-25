<!-- AGENT NOTICE: Keep this README synchronized with the package layout and top-level utilities in this directory whenever code changes. -->

# miniqm Package

Core library implementing QM9 graph featurization, Karras EDM 3D coordinate diffusion, Optax training, and interactive 3D molecular visualization.

* `data/` converts QM9 Parquet splits and SMILES strings into padded 2D molecular graph batches and device arrays.
* `diffusion/` implements 3D rigid-body geometry, the Karras noise schedule, the pair-biased Transformer denoiser, Kabsch-aligned losses, and Euler conformer sampling.
* `training/` manages optimizer schedules, EMA weights, checkpoint serialization, validation metrics, and the training loop.
* Top-level utilities provide centralized Pydantic configurations (`config.py`), runtime shape-checked array typing (`jaxtypes.py`), CLI entrypoints (`runner.py`), and `py3Dmol` 3D viewers (`viz.py`) for single molecules, conformer overlays, and multi-frame diffusion trajectories.
