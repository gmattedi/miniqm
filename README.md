<!-- AGENT NOTICE: Keep this README synchronized with the repository structure, CLI behavior, and modeling pipeline whenever code changes. -->

# miniqm

`miniqm` is a joint 2D+3D molecular diffusion model trained on heavy-atom QM9 graphs (`MAX_ATOMS = 9`) in JAX, Flax, and Optax.

It jointly denoises 3D Cartesian coordinates in $\mathbb{R}^3$, 26-channel one-hot heavy-atom features, and 6-channel one-hot bond features without $\mathrm{SO}(3)$-equivariant layers, supporting unconditional molecule generation, partial substructure/template conditioning (`U` / `*` dummy atoms and `~` dummy bonds), and full 2D SMILES conformer generation:

* Ground-truth heavy-atom conformers are centered at the origin and augmented with uniform random 3D rotations at every training step.
* A pair-biased Transformer denoiser with Karras EDM preconditioning predicts clean coordinates, atom features, and symmetric bond features from Gaussian-corrupted inputs and optional per-channel 2D conditioning masks.
* Training randomly samples full 2D conditioning, partial subgraph conditioning, or unconditional masks, minimizing a Kabsch-aligned coordinate MSE, a local pairwise interatomic distance loss, and EDM-weighted one-hot atom and bond MSE losses.
* Reverse Euler integration with stochastic noise injection generates 3D conformers and 2D molecular graphs from Gaussian noise down to $\sigma_{\min}$.

## Quickstart

Prepare the QM9 train, validation, and test Parquet splits:

```bash
uv run python data/download_qm9.py
```

Run training and periodic validation:

```bash
uv run miniqm-train --run-dir runs/qm9_experiment
```

Evaluate a saved run on the test split or sample molecules/conformers from full SMILES, partial templates, or unconditional dummy templates:

```bash
uv run miniqm-eval --run-dir runs/qm9_experiment --test-path data/test.parquet
uv run miniqm-sample --run-dir runs/qm9_experiment --smiles CCO "c1ccccc1~U~U(=U)" "U.U.U.U.U" --num-conformers 5 --output-sdf conformers.sdf
```

Run lint, type checking, and tests:

```bash
uv run pre-commit run --all-files
uv run pytest
```
