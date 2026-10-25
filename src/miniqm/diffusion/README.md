<!-- AGENT NOTICE: Keep this README synchronized with the geometry, noise, model, loss, and sampling implementations in this directory whenever code changes. -->

# Joint 2D+3D Molecular Diffusion

`miniqm.diffusion` implements Karras EDM joint diffusion over 3D Cartesian coordinates and continuous one-hot 2D atom and bond features:

* `geometry.py` provides masked center-of-mass zeroing, Gram-Schmidt uniform random 3D rotations, and SVD-based Kabsch alignment restricted to proper rotations ($\det R = +1$) so mirror enantiomers are never superimposed.
* `noise.py` implements log-normal training noise sampling, Karras polynomial rollout schedules, EDM preconditioning factors ($c_{\mathrm{in}}, c_{\mathrm{skip}}, c_{\mathrm{out}}$), random training conditioning masks (full 2D graph, random subgraph, or unconditional), and joint Gaussian corruption of coordinates, one-hot atom features, and symmetric one-hot bond features.
* `model.py` defines the pair-biased Transformer `Denoiser` conditioned on sinusoidal Fourier log-noise embeddings via adaptive LayerNorm, combining noisy and conditioned 2D atom/bond features with 3D coordinates, pairwise displacement vectors, and interatomic distances to predict clean coordinates, atom logits, and symmetric bond logits (`DenoiserOutput`).
* `losses.py` and `metrics.py` implement the differentiable training objective (EDM-weighted Kabsch-aligned coordinate MSE, local pairwise distance MSE within $5.0\text{ \AA}$, and one-hot atom and bond MSE) alongside evaluation metrics (per-molecule RMSD, superposition-free local lDDT, tetrahedral stereocenter inversion rates, and 2D atom/bond reconstruction accuracy).
* `sampling.py` runs reverse-time Euler integration with stochastic noise injection across coordinates, atom features, and symmetric bond features (`sample_molecules`, `sample_coordinates`, `sample_trajectory`, and `sample_molecule_trajectory`).
