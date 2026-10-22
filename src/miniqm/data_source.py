"""In-memory QM9 data source that pre-collates a Parquet split and yields batches."""

from collections.abc import Iterator
import logging
import pathlib
from typing import Final

import numpy as np

from miniqm import batch, schema

_LOGGER: Final = logging.getLogger(__name__)


class DataSource:
    """Loads a QM9 Parquet split into a pre-collated array store and yields batches."""

    def __init__(
        self,
        parquet_path: pathlib.Path,
        *,
        max_atoms: int = batch.MAX_ATOMS,
        max_molecules: int | None = None,
    ) -> None:
        molecules = schema.load_molecules(parquet_path, max_molecules=max_molecules)
        self._num_molecules = len(molecules)
        self._dataset = batch.collate_batch(molecules, max_atoms=max_atoms)
        _LOGGER.info(
            "Initialized DataSource from %s with %d molecules (max_atoms=%d).",
            parquet_path,
            self._num_molecules,
            max_atoms,
        )

    def __len__(self) -> int:
        return self._num_molecules

    def iter_batches(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        drop_last: bool = False,
    ) -> Iterator[batch.Batch]:
        """Yields padded batches sliced from the pre-collated dataset.

        Args:
            batch_size: Number of molecules per batch.
            rng: Optional NumPy random generator for shuffling indices before batching.
            drop_last: Whether to drop the trailing batch if smaller than batch_size.

        Yields:
            Sliced Batch instances.
        """
        if batch_size <= 0:
            raise ValueError(f"Expected positive batch_size, got {batch_size}.")

        indices = (
            rng.permutation(self._num_molecules)
            if rng is not None
            else np.arange(self._num_molecules)
        )

        for start_idx in range(0, self._num_molecules, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            if drop_last and batch_indices.size < batch_size:
                break
            yield batch.Batch(
                atom_features=self._dataset.atom_features[batch_indices],
                bond_features=self._dataset.bond_features[batch_indices],
                atom_mask=self._dataset.atom_mask[batch_indices],
                positions=self._dataset.positions[batch_indices],
                labels=self._dataset.labels[batch_indices],
            )
