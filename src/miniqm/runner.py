"""CLI entrypoint for training and evaluating the miniqm diffusion model."""

from collections.abc import Sequence
import logging

from rich import logging as rich_logging
import tyro

from miniqm.training import train


def main(argv: Sequence[str] | None = None) -> None:
    """Parses CLI arguments via tyro and runs the training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_logging.RichHandler(rich_tracebacks=True, show_path=False)],
    )
    config = tyro.cli(train.RunConfig, args=argv)
    train.run_training(config)


if __name__ == "__main__":
    main()
