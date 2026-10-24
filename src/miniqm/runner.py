"""CLI entrypoint for training and evaluating the miniqm diffusion model."""

from collections.abc import Sequence
import logging

import tyro

from miniqm import train


def main(argv: Sequence[str] | None = None) -> None:
    """Parses `RunConfig` CLI arguments via `tyro` and runs the training loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = tyro.cli(train.RunConfig, args=argv)
    train.run_training(config)


if __name__ == "__main__":
    main()
