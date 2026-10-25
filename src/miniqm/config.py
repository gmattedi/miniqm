"""Pydantic configuration models and artifact enums for miniqm."""

import enum
import pathlib
from typing import Annotated, Final, Self

import pydantic

from miniqm.data import schema

DEFAULT_TRAIN_PATH: Final = pathlib.Path(f"data/{schema.Split.TRAIN}.parquet")
DEFAULT_VAL_PATH: Final = pathlib.Path(f"data/{schema.Split.VALIDATION}.parquet")
DEFAULT_TEST_PATH: Final = pathlib.Path(f"data/{schema.Split.TEST}.parquet")
DEFAULT_RUN_DIR: Final = pathlib.Path("runs/qm9_experiment")
DEFAULT_SAMPLED_SDF: Final = pathlib.Path("sampled_conformers.sdf")
DEFAULT_SMILES: Final = ("CCO", "c1ccccc1")


class CheckpointName(enum.StrEnum):
    """Standard MessagePack checkpoint filenames inside `<run_dir>/checkpoints/`."""

    LATEST = "latest.msgpack"
    BEST = "best.msgpack"


class ArtifactName(enum.StrEnum):
    """Standard run directory artifact filenames and subdirectories."""

    CONFIG = "config.json"
    METRICS = "metrics.jsonl"
    TEST_METRICS = "test_metrics.json"
    CHECKPOINTS_DIR = "checkpoints"


class NoiseConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Karras EDM noise distribution and rollout schedule parameters.

    Attributes:
        sigma_data: Empirical coordinate standard deviation on centered QM9 (Angstroms).
        s_min: Minimum relative noise multiplier at the end of the sampling rollout.
        s_max: Maximum relative noise multiplier at the start of the sampling rollout.
        rho: Polynomial exponent controlling step spacing between `s_max` and `s_min`.
        log_sigma_mean: Mean of the log-normal training noise distribution `ln(sigma / sigma_data)`.
        log_sigma_std: Standard deviation of the log-normal training noise distribution.
    """

    sigma_data: pydantic.PositiveFloat = 1.2
    s_min: pydantic.PositiveFloat = 4e-4
    s_max: pydantic.PositiveFloat = 80.0
    rho: pydantic.PositiveInt = 7
    log_sigma_mean: float = -1.2
    log_sigma_std: pydantic.PositiveFloat = 1.5

    @pydantic.model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.s_max <= self.s_min:
            raise ValueError(f"Expected s_max ({self.s_max}) > s_min ({self.s_min}).")
        return self


class TransformerConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Architecture hyperparameters for the pair-biased Transformer stack.

    Attributes:
        hidden_dim: Atom token embedding and attention hidden dimension.
        num_layers: Number of pre-norm Transformer blocks.
        num_heads: Number of attention heads per block (must divide `hidden_dim`).
        mlp_factor: Expansion multiplier for the SiLU feed-forward hidden dimension.
    """

    hidden_dim: pydantic.PositiveInt = 128
    num_layers: pydantic.PositiveInt = 4
    num_heads: pydantic.PositiveInt = 4
    mlp_factor: pydantic.PositiveInt = 4

    @pydantic.model_validator(mode="after")
    def _validate_dimensions(self) -> Self:
        if self.hidden_dim % self.num_heads != 0 or self.hidden_dim % 2 != 0:
            raise ValueError(
                f"Expected hidden_dim ({self.hidden_dim}) divisible by "
                f"num_heads ({self.num_heads}) and 2."
            )
        return self


class FourierNoiseConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Sinusoidal Fourier embedding parameters for log-noise conditioning.

    Attributes:
        log_sigma_scale: Scaling factor applied to `log(sigma / sigma_data)` before encoding.
        max_period: Maximum sinusoidal period controlling the lowest frequency band.
    """

    log_sigma_scale: pydantic.PositiveFloat = 0.25
    max_period: Annotated[float, pydantic.Field(gt=1.0)] = 100.0


class DenoiserConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Complete configuration for the 3D molecular coordinate denoiser.

    Attributes:
        transformer: Pair-biased Transformer backbone hyperparameters.
        fourier: Sinusoidal Fourier noise encoder hyperparameters.
        noise: Karras EDM preconditioning and noise schedule parameters.
    """

    transformer: TransformerConfig = TransformerConfig()
    fourier: FourierNoiseConfig = FourierNoiseConfig()
    noise: NoiseConfig = NoiseConfig()


class LossConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Loss weighting and cutoff parameters for 3D coordinate diffusion training.

    Attributes:
        distance_loss_weight: Multiplier on the auxiliary pairwise distance MSE term.
        distance_cutoff: Upper pairwise distance threshold (Angstroms) for auxiliary supervision.
    """

    distance_loss_weight: pydantic.NonNegativeFloat = 0.25
    distance_cutoff: pydantic.PositiveFloat = 5.0


class SamplingConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Reverse Euler integration and stochastic noise injection parameters.

    Attributes:
        num_steps: Number of reverse discretization steps across the Karras schedule.
        gamma_0: Fractional noise expansion rate when `sigma > gamma_min`.
        gamma_min: Noise threshold (Angstroms) below which stochastic noise injection is disabled.
        noise_scale: Standard deviation multiplier for injected Gaussian noise.
        step_scale: Euler velocity multiplier applied during stochastic noise injection steps.
    """

    num_steps: pydantic.PositiveInt = 40
    gamma_0: pydantic.NonNegativeFloat = 0.8
    gamma_min: pydantic.NonNegativeFloat = 0.1
    noise_scale: pydantic.NonNegativeFloat = 1.003
    step_scale: pydantic.PositiveFloat = 1.5


class TrainConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Hyperparameters for model architecture, diffusion loss, and Optax optimization.

    Attributes:
        model: Denoiser Transformer and Karras preconditioning configuration.
        loss: Kabsch-aligned MSE and local pairwise distance loss weights.
        learning_rate: Peak AdamW learning rate after linear warmup.
        warmup_steps: Number of linear warmup steps from `0.0` to `learning_rate`.
        decay_steps: Total steps over which the warmup + cosine decay schedule spans.
        min_learning_rate: Floor learning rate at the end of cosine decay.
        weight_decay: Decoupled AdamW weight decay coefficient.
        max_grad_norm: Global gradient norm clipping threshold.
        ema_decay: Exponential moving average decay rate for evaluation parameters.
    """

    model: DenoiserConfig = DenoiserConfig()
    loss: LossConfig = LossConfig()
    learning_rate: pydantic.PositiveFloat = 3e-4
    warmup_steps: pydantic.NonNegativeInt = 1_000
    decay_steps: pydantic.PositiveInt = 100_000
    min_learning_rate: pydantic.NonNegativeFloat = 1e-5
    weight_decay: pydantic.NonNegativeFloat = 1e-4
    max_grad_norm: pydantic.PositiveFloat = 1.0
    ema_decay: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.999

    @pydantic.model_validator(mode="after")
    def _validate_schedule_bounds(self) -> Self:
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                f"Expected 0 <= min_learning_rate <= learning_rate, got "
                f"min_learning_rate={self.min_learning_rate}, learning_rate={self.learning_rate}."
            )
        if self.decay_steps <= self.warmup_steps:
            raise ValueError(
                f"Expected decay_steps ({self.decay_steps}) > warmup_steps ({self.warmup_steps})."
            )
        return self


class RunConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """End-to-end configuration for the training, evaluation, and checkpointing loop.

    Attributes:
        train_path: Path to the training Parquet split.
        val_path: Path to the validation Parquet split.
        run_dir: Parent directory where timestamped run subdirectories are created.
        train: Model, loss, and optimizer hyperparameters.
        sampling: Reverse-ODE Euler sampling schedule configuration.
        batch_size: Number of molecules per optimization step.
        num_steps: Total number of optimization steps to run.
        log_every: Step interval for writing training scalars and console logs.
        eval_every: Step interval for computing validation loss and saving `latest.msgpack`.
        sample_every: Step interval for running reverse-ODE sampling on a validation batch.
        max_train_molecules: Optional cap on loaded training molecules (for debugging/tests).
        max_val_molecules: Optional cap on loaded validation molecules.
        sample_batch_size: Number of validation molecules used for periodic conformer sampling.
        num_devices: Number of XLA devices to shard each training batch across.
        seed: Base PRNG seed for weight initialization, data shuffling, and evaluation.
    """

    train_path: pathlib.Path = DEFAULT_TRAIN_PATH
    val_path: pathlib.Path = DEFAULT_VAL_PATH
    run_dir: pathlib.Path = DEFAULT_RUN_DIR
    train: TrainConfig = TrainConfig()
    sampling: SamplingConfig = SamplingConfig()
    batch_size: pydantic.PositiveInt = 64
    num_steps: pydantic.PositiveInt = 50_000
    log_every: pydantic.PositiveInt = 50
    eval_every: pydantic.PositiveInt = 500
    sample_every: pydantic.PositiveInt = 1_000
    max_train_molecules: pydantic.PositiveInt | None = None
    max_val_molecules: pydantic.PositiveInt | None = 512
    sample_batch_size: pydantic.PositiveInt = 32
    num_devices: pydantic.PositiveInt = 8
    seed: int = 0


class EvalConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Configuration for evaluating a trained checkpoint on a Parquet split.

    Attributes:
        run_dir: Run directory (or parent experiment directory) containing `config.json`.
        test_path: Path to the evaluation Parquet split.
        checkpoint_name: Filename inside `<run_dir>/checkpoints/` to evaluate.
        use_ema: Whether to evaluate `ema_params` (`True`) or raw `params` (`False`).
        batch_size: Number of molecules per evaluation batch.
        max_test_molecules: Optional cap on loaded molecules from `test_path`.
        num_sampling_steps: Optional override for `SamplingConfig.num_steps`.
        output_path: Optional path to write the JSON evaluation report.
        seed: PRNG seed for teacher-forced noise and reverse-ODE sampling.
    """

    run_dir: pathlib.Path = DEFAULT_RUN_DIR
    test_path: pathlib.Path = DEFAULT_TEST_PATH
    checkpoint_name: str = CheckpointName.BEST
    use_ema: bool = True
    batch_size: pydantic.PositiveInt = 64
    max_test_molecules: pydantic.PositiveInt | None = None
    num_sampling_steps: pydantic.PositiveInt | None = None
    output_path: pathlib.Path | None = None
    seed: int = 0


class SampleSmilesConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Configuration for generating 3D conformers from a sequence of SMILES strings.

    Attributes:
        run_dir: Run directory (or parent experiment directory) containing `config.json`.
        smiles: SMILES strings to generate 3D conformers for.
        smiles_file: Optional newline-delimited SMILES file (takes precedence over `smiles`).
        checkpoint_name: Filename inside `<run_dir>/checkpoints/` to load.
        use_ema: Whether to sample with `ema_params` (`True`) or raw `params` (`False`).
        num_conformers: Number of conformers to sample per SMILES (using `seed + conformer_idx`).
        num_sampling_steps: Optional override for `SamplingConfig.num_steps`.
        output_sdf: Output SDF file path where generated 3D conformers are written.
        seed: Base PRNG seed for reverse-ODE coordinate sampling.
    """

    run_dir: pathlib.Path = DEFAULT_RUN_DIR
    smiles: tuple[str, ...] = DEFAULT_SMILES
    smiles_file: pathlib.Path | None = None
    checkpoint_name: str = CheckpointName.BEST
    use_ema: bool = True
    num_conformers: pydantic.PositiveInt = 1
    num_sampling_steps: pydantic.PositiveInt | None = None
    output_sdf: pathlib.Path = DEFAULT_SAMPLED_SDF
    seed: int = 0


DEFAULT_NOISE_CONFIG: Final = NoiseConfig()
DEFAULT_DENOISER_CONFIG: Final = DenoiserConfig()
DEFAULT_LOSS_CONFIG: Final = LossConfig()
DEFAULT_SAMPLING_CONFIG: Final = SamplingConfig()
DEFAULT_TRAIN_CONFIG: Final = TrainConfig()
