"""CLI entrypoints for training, test-set evaluation, and SMILES conformer inference."""

from collections.abc import Sequence
import logging
import os
from typing import Final

from rich import logging as rich_logging
import tyro

from miniqm.training import eval as eval_lib
from miniqm.training import train

_LOGGER: Final = logging.getLogger(__name__)
_SIGINT_EXIT_CODE: Final = 130


def _configure_logging() -> None:
    """Configures default CPU platform and Rich console logging for CLI entrypoints."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
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
    try:
        train.run_training(config)
    except KeyboardInterrupt as exc:
        _LOGGER.warning("Training interrupted by user.")
        raise SystemExit(_SIGINT_EXIT_CODE) from exc


def eval_main(argv: Sequence[str] | None = None) -> None:
    """Parses evaluation CLI arguments via tyro and evaluates a checkpoint on a test split."""
    _configure_logging()
    config = tyro.cli(eval_lib.EvalConfig, args=argv)
    try:
        eval_lib.run_evaluation(config)
    except KeyboardInterrupt as exc:
        _LOGGER.warning("Evaluation interrupted by user.")
        raise SystemExit(_SIGINT_EXIT_CODE) from exc


def sample_main(argv: Sequence[str] | None = None) -> None:
    """Parses SMILES inference CLI arguments via tyro and writes sampled SDF conformers."""
    _configure_logging()
    config = tyro.cli(eval_lib.SampleSmilesConfig, args=argv)
    try:
        eval_lib.run_smiles_sampling(config)
    except KeyboardInterrupt as exc:
        _LOGGER.warning("Sampling interrupted by user.")
        raise SystemExit(_SIGINT_EXIT_CODE) from exc


if __name__ == "__main__":
    main()
