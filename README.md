<!-- AGENT NOTICE: Keep this README synchronized with the repository structure, CLI behavior, and modeling pipeline whenever code changes. -->

# miniqm

`miniqm` is a compact JAX, Flax, and Optax implementation of joint 2D+3D molecular diffusion on heavy-atom QM9 graphs.

Instead of using specialized $\mathrm{SE}(3)$-equivariant layers, a standard pair-biased Transformer denoises 3D Cartesian coordinates in $\mathbb{R}^3$, one-hot heavy-atom features, and one-hot bond features jointly under Karras EDM preconditioning. Per-channel conditioning masks allow the same checkpoint to perform 3D conformer generation from full SMILES, scaffold-constrained 2D+3D completion from partial templates, and unconditional de novo molecule generation.

![Reverse diffusion rollout](assets/reverse_diffusion.gif)

## Representation & Conditioning

Each QM9 molecule is stripped of explicit hydrogens and padded to $N$ heavy atoms:

* Coordinates $\mathbf{x} \in \mathbb{R}^{N \times 3}$ hold zero-center-of-mass Cartesian positions in Angstroms. During training, each molecule is rotated by a uniform random proper rotation $R \in \mathrm{SO}(3)$ sampled via Gram-Schmidt orthogonalization.
* Atom features $\mathbf{a} \in \{0, 1\}^{N \times D_{\mathrm{atom}}}$ concatenate one-hot blocks for heavy-atom element, total degree, total attached hydrogens, formal charge, hybridization, aromaticity, ring membership, and 3D Cahn-Ingold-Prelog ($R$/$S$) stereochemistry.
* Bond features $\mathbf{b} \in \{0, 1\}^{N \times N \times D_{\mathrm{bond}}}$ form a symmetric pair tensor encoding bond order and ring membership.

Boolean conditioning masks over the atom and bond feature channels control which 2D attributes are known. Unconditioned channels are zeroed in the conditioning input and denoised from Gaussian noise alongside coordinates. During training, each molecule is assigned one of three conditioning regimes:

* Full 2D graph conditioning. All atom and bond channels are known, and the model predicts 3D conformer coordinates.
* Partial subgraph conditioning. A random subset of atoms and incident bonds is masked out.
* Unconditional generation. Only the valid-atom count mask is given, and both the 2D graph and 3D coordinates are denoised from scratch.

At inference time, SMILES or SMARTS strings are parsed into the same mask format using `U` (or `*`) for unconditioned atoms and `~` for unconditioned bonds:

```text
"CCO"               -> full 2D conditioning (conformer generation)
"c1ccccc1~U~U(=U)"  -> phenyl ring fixed; substituent atoms and ~ bonds denoised
"U.U.U.U.U"         -> unconditioned heavy atoms; full 2D+3D de novo generation
```

For full SMILES prompts, sampled 3D coordinates are attached directly to the input RDKit graph. For partial or unconditional templates, continuous bond and atom predictions are decoded into a valence-consistent RDKit molecule. The decoder combines 2D bond logits with interatomic distances to suppress long-range false bonds, enforces per-atom bond-order capacities, and adjusts elements when needed to satisfy open valences.

## Diffusion Formulation

### Karras EDM Preconditioning

Given noise level $\sigma > 0$ and data standard deviation $\sigma_{\mathrm{data}}$, the preconditioning coefficients are

$$c_{\mathrm{skip}}(\sigma) = \frac{\sigma_{\mathrm{data}}^2}{\sigma^2 + \sigma_{\mathrm{data}}^2}, \quad c_{\mathrm{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\mathrm{data}}}{\sqrt{\sigma^2 + \sigma_{\mathrm{data}}^2}}, \quad c_{\mathrm{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\mathrm{data}}^2}}, \quad \lambda(\sigma) = \frac{\sigma^2 + \sigma_{\mathrm{data}}^2}{(\sigma \cdot \sigma_{\mathrm{data}})^2}.$$

Training samples $\ln \sigma$ from a normal distribution clipped to $[\sigma_{\min}, \sigma_{\max}]$. Independent Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ is added to coordinates (zero-centered over valid atoms) and to unconditioned atom and symmetric bond channels. The denoiser $D_\theta$ on $z = (\mathbf{x}, \mathbf{a}, \mathbf{b})$ is parameterized as

$$D_\theta(z_t, \sigma) = c_{\mathrm{skip}}(\sigma)\, z_t + c_{\mathrm{out}}(\sigma)\, F_\theta\!\left(c_{\mathrm{in}}(\sigma)\, z_t,\, \frac{1}{4}\ln \sigma\right).$$

### Pair-Biased Transformer Architecture

$F_\theta$ operates on atom tokens $h_i \in \mathbb{R}^{d}$ and pair representations $p_{ij} \in \mathbb{R}^{d}$:

* Atom embeddings sum linear projections of scaled noisy coordinates $c_{\mathrm{in}}(\sigma)\mathbf{x}_i$, scaled noisy atom features, masked ground-truth conditioning features, and the binary conditioning mask.
* Pair embeddings sum linear projections of scaled noisy bond features, conditioned bond features, pairwise coordinate displacements $c_{\mathrm{in}}(\sigma)(\mathbf{x}_i - \mathbf{x}_j)$, and interatomic distances $\|c_{\mathrm{in}}(\sigma)(\mathbf{x}_i - \mathbf{x}_j)\|_2$.
* A sinusoidal Fourier encoder maps $\frac{1}{4}\ln \sigma$ into a conditioning vector. Zero-initialized adaptive LayerNorm scale, shift, and residual gates modulate the pre-norm Transformer blocks with this vector.
* Each Transformer block biases multi-head self-attention logits with a learned projection of $p_{ij}$ and updates atom tokens through a gated SiLU MLP. Pair representations are updated via outer sums of projected atom tokens and symmetrized as $p_{ij} \leftarrow \frac{1}{2}(p_{ij} + p_{ji})$.
* Output heads project final atom tokens to coordinate updates (re-centered to zero center of mass) and atom logits, and project final pair tokens to symmetric bond logits.

### Training Loss & Reverse Sampling

The training objective combines four terms weighted by $\lambda(\sigma)$:

$$\mathcal{L} = \mathcal{L}_{\mathrm{coord}} + w_{\mathrm{dist}}\,\mathcal{L}_{\mathrm{dist}} + w_{\mathrm{atom}}\,\mathcal{L}_{\mathrm{atom}} + w_{\mathrm{bond}}\,\mathcal{L}_{\mathrm{bond}}$$

* $\mathcal{L}_{\mathrm{coord}}$ is the EDM-weighted MSE between predicted coordinates $\hat{\mathbf{x}}_0$ and ground-truth coordinates $\mathbf{x}_0^{\mathrm{aligned}}$ rigid-body aligned onto $\mathrm{stop\_gradient}(\hat{\mathbf{x}}_0)$ via Kabsch SVD ($\det R = +1$).
* $\mathcal{L}_{\mathrm{dist}}$ is an auxiliary MSE on local pairwise interatomic distances.
* $\mathcal{L}_{\mathrm{atom}}$ and $\mathcal{L}_{\mathrm{bond}}$ are EDM-weighted MSE losses on the continuous one-hot atom and bond targets.

At inference time, reverse integration steps through the polynomial noise schedule:

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N_{\mathrm{steps}} - 1}\left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)\right)^\rho.$$

Each step applies an Euler update with optional Karras stochastic churn. Conditioned 2D channels are re-pinned after the final denoising step.

## Quickstart

Install dependencies with `uv` and download the QM9 splits:

```bash
uv sync
uv run python data/download_qm9.py
```

### Sampling & Evaluation with Pretrained Weights

`miniqm-sample` and `miniqm-eval` load the pretrained checkpoint in `pretrained_weights/` by default:

```bash
# Sample 3D conformers and 2D+3D molecules from full SMILES, partial templates, or dummy atoms:
uv run miniqm-sample \
    --smiles CCO c1ccccc1 "c1ccccc1~U~U(=U)" "U.U.U.U.U" \
    --num-conformers 5 \
    --output-sdf sampled_conformers.sdf

# Evaluate teacher-forced loss and sampled conformer RMSD / lDDT on the test split:
uv run miniqm-eval
```

### Training From Scratch

Launch a training run sharded across logical XLA CPU devices:

```bash
uv run miniqm-train --run-dir runs/qm9_experiment
```

Pass `--run-dir runs/qm9_experiment` to `miniqm-sample` or `miniqm-eval` to evaluate a custom run. Both commands automatically resolve the latest timestamped run directory inside the given path.

## Interactive Notebooks

The `notebooks/` directory walks through each stage of the pipeline with 2D RDKit drawings and interactive `py3Dmol` 3D viewers:

* Loading pre-collated QM9 batches, inspecting heavy-atom count and RMS radius distributions, and plotting 2D molecular graphs and bond adjacency matrices.
* Center-of-mass zeroing, uniform random 3D rotations, proper-rotation Kabsch alignment, and mirror-enantiomer chirality preservation.
* Karras EDM preconditioning curves, single-batch denoiser training convergence, and animated 3D reverse-diffusion trajectories.
* Evaluating the pretrained checkpoint on test-set conformer RMSD and lDDT, and animating 3D trajectories for full SMILES (`CCO`), partial templates (`c1ccccc1~U~U(=U)`), and de novo generation (`U.U.U.U.U`).
