<!-- AGENT NOTICE: Keep this README synchronized with the repository structure, CLI behavior, and modeling pipeline whenever code changes. -->

# miniqm

`miniqm` is a 3D molecular coordinate diffusion model trained on the QM9 dataset in JAX, Flax, and Optax.

It predicts equilibrium 3D atomic coordinates directly in Cartesian $\mathbb{R}^3$ from a 2D molecular graph (atoms, bonds, ring membership, and formal stereochemistry) without $\mathrm{SO}(3)$-equivariant layers:

* Ground-truth conformers are centered at the origin and augmented with uniform random 3D rotations at every training step.
* A pair-biased Transformer denoiser with Karras EDM preconditioning predicts clean coordinates from Gaussian-corrupted inputs.
* Training minimizes a Kabsch-aligned coordinate MSE combined with a local pairwise interatomic distance loss.
* Conformer generation integrates the reverse probability flow ODE with stochastic churn from Gaussian noise down to $\sigma_{\min}$.

## Quickstart

Prepare the QM9 train, validation, and test Parquet splits:

```bash
uv run python data/download_qm9.py
```

Run training and periodic validation:

```bash
uv run miniqm-train --run-dir runs/qm9_experiment
```

Evaluate a saved run on the test split or sample 3D conformers from arbitrary SMILES:

```bash
uv run miniqm-eval --run-dir runs/qm9_experiment --test-path data/test.parquet
uv run miniqm-sample --run-dir runs/qm9_experiment --smiles CCO c1ccccc1 --output-xyz conformers.xyz
```

Run lint, type checking, and tests:

```bash
uv run pre-commit run --all-files
uv run pytest
```
