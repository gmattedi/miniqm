# miniqm

`miniqm` is a small JAX/Flax diffusion model that generates QM9 molecules in 2D and 3D at the same time.

This is a small model built to demonstrate how basic diffusion models can be used for ligand conformer generation and generative chemistry tasks,
please consider this an academic exercise rather than a tool that can be relied on.

Given a molecule with up to 9 heavy atoms (C, N, O, F), the model jointly denoises 3D Cartesian coordinates, one-hot atom features, and one-hot bond features. By masking out different parts of the 2D graph, the same checkpoint handles three tasks:

```text
"CCO"               -> 3D conformer generation from a full SMILES string
"c1ccccc1~U~U(=U)"  -> scaffold completion (phenyl ring fixed; U atoms and ~ bonds denoised)
"U.U.U.U.U"         -> unconditional 2D+3D molecule generation from dummy atoms
```

![Reverse diffusion rollout](assets/reverse_diffusion.gif)

## Quickstart

Install dependencies with `uv` and download the QM9 splits:

```bash
uv sync
uv run python data/download_qm9.py
```

### Sample Molecules & Evaluate

`miniqm-sample` and `miniqm-eval` use the pretrained checkpoint in `pretrained_weights/` by default:

```bash
# Sample 3D conformers and 2D+3D molecules into an SDF file:
uv run miniqm-sample \
    --smiles CCO c1ccccc1 "c1ccccc1~U~U(=U)" "U.U.U.U.U" \
    --num-conformers 5 \
    --output-sdf sampled_conformers.sdf

# Evaluate test-split losses and sampled conformer RMSD, lDDT, and stereochemistry:
uv run miniqm-eval
```

### Train from Scratch

Launch a training run sharded across logical CPU devices:

```bash
uv run miniqm-train --run-dir runs/qm9_experiment
```

Pass `--run-dir runs/qm9_experiment` to `miniqm-sample` or `miniqm-eval` to use a custom run. Both commands automatically pick the latest timestamped run directory inside the given path.

## How It Works

* Explicit hydrogens are stripped, leaving heavy atoms padded to a fixed size. Each molecule is represented by zero-centered 3D coordinates, one-hot atom features (element, degree, attached H count, formal charge, hybridization, aromaticity, ring membership, and $R$/$S$ stereochemistry), and symmetric one-hot bond features (bond order and ring membership).
* Boolean masks over the atom and bond channels specify which parts of the 2D graph are known. During training, each molecule is randomly assigned full 2D conditioning, partial subgraph conditioning, or no 2D conditioning. At inference, SMILES/SMARTS prompts use `U` (or `*`) for unknown atoms and `~` for bonds of unknown order.
* Training samples log-normal noise levels $\sigma$ and adds Gaussian noise to coordinates, atom features, and symmetric bond features. The network uses Karras EDM preconditioning ($c_{\mathrm{in}}, c_{\mathrm{skip}}, c_{\mathrm{out}}$), and reverse sampling integrates an Euler trajectory with optional stochastic churn.
* Rather than using $\mathrm{SE}(3)$-equivariant layers, the denoiser is a standard Transformer with pairwise attention biases. Atom tokens embed noisy coordinates, noisy and conditioned atom features, atom index, and noise level. Pair representations combine noisy and conditioned bond features with 3D pairwise displacements, distances, and Gaussian radial basis functions to produce attention biases for every layer and symmetric bond predictions at the output.
* Target coordinates are randomly rotated in 3D during training. The coordinate loss aligns the ground-truth conformer onto the detached prediction via Kabsch SVD (restricted to proper rotations, $\det R = +1$, to preserve chirality) and combines EDM-weighted coordinate MSE, local pairwise distance MSE, and atom/bond feature MSE.
* For full SMILES inputs, sampled 3D coordinates are attached directly to the RDKit graph. For partial or unconditional prompts, predicted atom and bond logits are decoded greedily into a valence-valid RDKit molecule.

## Interactive Notebooks

The `notebooks/` directory walks through each part of the pipeline with 2D RDKit drawings and interactive `py3Dmol` 3D viewers:

* QM9 batch loading, heavy-atom count and radius distributions, and 2D molecular graph featurization.
* Center-of-mass zeroing, random 3D rotations, chirality preservation, and Kabsch alignment.
* Karras EDM preconditioning curves, single-batch training convergence, and animated 3D reverse-diffusion trajectories.
* Test-set evaluation and animated 2D+3D sampling trajectories for full SMILES (`CCO`), partial scaffolds (`c1ccccc1~U~U(=U)`), and unconditional generation (`U.U.U.U.U`).
