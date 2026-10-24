"""CLI entrypoints for training, test-set evaluation, and SMILES conformer inference."""

from collections.abc import Sequence
import logging

from rich import logging as rich_logging
import tyro

from miniqm.training import eval as eval_lib
from miniqm.training import train


def _configure_logging() -> None:
    """Configures Rich console logging for CLI entrypoints."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_logging.RichHandler(rich_tracebacks=True, show_path=False)],
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parses training CLI arguments via tyro and runs the training loop."""
    _configure_logging()
    config = tyro.cli(train.RunConfig, args=argv)
    train.run_training(config)


def eval_main(argv: Sequence[str] | None = None) -> None:
    """Parses evaluation CLI arguments via tyro and evaluates a checkpoint on a test split."""
    _configure_logging()
    config = tyro.cli(eval_lib.EvalConfig, args=argv)
    eval_lib.run_evaluation(config)


def sample_main(argv: Sequence[str] | None = None) -> None:
    """Parses SMILES inference CLI arguments via tyro and writes sampled XYZ conformers."""
    _configure_logging()
    config = tyro.cli(eval_lib.SampleSmilesConfig, args=argv)
    eval_lib.run_smiles_sampling(config)


if __name__ == "__main__":
    main()
