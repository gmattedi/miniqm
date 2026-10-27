# Joint 2D+3D Molecular Diffusion

`miniqm.diffusion` implements Karras EDM joint diffusion over 3D Cartesian coordinates and continuous one-hot 2D atom and bond features:

* Masked center-of-mass zeroing, Gram-Schmidt uniform random 3D rotations, and SVD-based Kabsch alignment restricted to proper rotations ($\det R = +1$) so mirror enantiomers are never superimposed.
* Log-normal training noise sampling, Karras polynomial rollout schedules, EDM preconditioning factors ($c_{\mathrm{in}}, c_{\mathrm{skip}}, c_{\mathrm{out}}$), random training conditioning masks (full 2D graph, random subgraph, or unconditional), and joint Gaussian corruption of coordinates, atom features, and symmetric bond features.
* Pair-biased Transformer denoiser conditioned on sinusoidal Fourier log-noise embeddings via adaptive LayerNorm. Combines noisy and conditioned 2D atom/bond features with 3D coordinates, pairwise displacement vectors, and interatomic distances to predict clean coordinates, atom logits, and symmetric bond logits.
* Differentiable training losses (EDM-weighted Kabsch-aligned coordinate MSE, local pairwise distance MSE, and one-hot atom and bond MSE) and evaluation metrics (per-molecule RMSD, superposition-free local lDDT, tetrahedral stereocenter inversion rates, and 2D atom/bond reconstruction accuracy).
* Reverse-time Euler integration with stochastic noise injection across coordinates, atom features, and symmetric bond features.
