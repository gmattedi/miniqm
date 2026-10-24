<!-- AGENT NOTICE: Keep this README synchronized with the checkpointing, evaluation, and training loop logic in this directory whenever code changes. -->

# Training, Evaluation & Checkpointing

Manages optimization state, validation metrics, and checkpoint persistence for the coordinate denoiser:

* Builds a global-norm-clipped AdamW optimizer with linear warmup and cosine learning-rate decay, maintaining an exponential moving average (EMA) of model parameters across steps.
* Evaluates both single-step teacher-forced diffusion loss across validation splits and full reverse-diffusion conformer generation RMSD after Kabsch alignment to ground truth.
* Serializes step count, active parameters, and EMA parameters to MessagePack checkpoints, tracking both the latest checkpoint and the best validation EMA checkpoint while logging architecture/parameter summaries, a live progress bar, TensorBoard event files, and JSONL scalars.
