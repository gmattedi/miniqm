<!-- AGENT NOTICE: Keep this README synchronized with the geometry, noise, model, loss, and sampling implementations in this directory whenever code changes. -->

# 3D Coordinate Diffusion

`miniqm.diffusion` implements Karras EDM coordinate diffusion directly in $\mathbb{R}^3$:

* Masked center-of-mass zeroing, Gram-Schmidt uniform random 3D rotations, and SVD-based Kabsch alignment restricted to proper rotations ($\det R = +1$) so mirror enantiomers are never superimposed.
* Log-normal training noise sampling, Karras polynomial rollout schedules, and EDM input/skip/output preconditioning factors ($c_{\mathrm{in}}, c_{\mathrm{skip}}, c_{\mathrm{out}}$) calibrated to the empirical coordinate standard deviation of centered QM9 molecules ($\sigma_{\mathrm{data}} = 1.2\text{ \AA}$).
* A pair-biased Transformer denoiser conditioned on sinusoidal Fourier log-noise embeddings via adaptive LayerNorm, combining 2D bond features with 3D pairwise displacement vectors and interatomic distances.
* A training objective summing EDM-weighted Kabsch-aligned coordinate MSE and per-molecule local pairwise distance MSE within a $5.0\text{ \AA}$ cutoff, paired with superposition-free local lDDT scoring at $(0.1, 0.25, 0.5, 1.0)\text{ \AA}$ tolerances.
* Reverse-time Euler integration with stochastic noise injection that evolves centered Gaussian noise into 3D molecular coordinates and records full multi-frame trajectories.
