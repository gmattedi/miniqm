"""In-memory QM9 data source that pre-collates a Parquet split and yields batches."""

from collections.abc import Iterator, Sequence
from concurrent import futures
import logging
import multiprocessing
import os
import pathlib
from typing import Final

import numpy as np
from rich import progress

from miniqm.data import batch, schema

_LOGGER: Final = logging.getLogger(__name__)
_PARALLEL_MIN_MOLECULES: Final = 512
_CHUNK_SIZE: Final = 2_048
_MAX_WORKERS: Final = 16


def _collate_chunk(args: tuple[Sequence[schema.Molecule], int]) -> batch.Batch:
    """Featurizes and collates a slice of molecules inside a worker process."""
    molecules, max_atoms = args
    return batch.collate_batch(molecules, max_atoms=max_atoms)


def _concatenate_batches(batches: Sequence[batch.Batch]) -> batch.Batch:
    """Concatenates a non-empty sequence of padded batches along the molecule axis."""
    if len(batches) == 1:
        return batches[0]
    return batch.Batch(
        atom_features=np.concatenate(tuple(b.atom_features for b in batches), axis=0),
        bond_features=np.concatenate(tuple(b.bond_features for b in batches), axis=0),
        atom_mask=np.concatenate(tuple(b.atom_mask for b in batches), axis=0),
        positions=np.concatenate(tuple(b.positions for b in batches), axis=0),
        labels=np.concatenate(tuple(b.labels for b in batches), axis=0),
    )


def _collate_molecules(
    molecules: Sequence[schema.Molecule],
    *,
    max_atoms: int,
    max_workers: int = _MAX_WORKERS,
) -> batch.Batch:
    """Collates molecules into a single batch, parallelizing large splits across CPU workers."""
    num_workers = min(max_workers, os.cpu_count() or 1)
    if len(molecules) < _PARALLEL_MIN_MOLECULES or num_workers <= 1:
        return batch.collate_batch(molecules, max_atoms=max_atoms)

    chunks = tuple(
        (molecules[idx : idx + _CHUNK_SIZE], max_atoms)
        for idx in range(0, len(molecules), _CHUNK_SIZE)
    )
    mp_context = multiprocessing.get_context("forkserver")
    with futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        collated_chunks = tuple(
            progress.track(
                executor.map(_collate_chunk, chunks),
                total=len(chunks),
                description=f"Featurizing {len(molecules):,} molecules ({num_workers} workers)",
            )
        )
    return _concatenate_batches(collated_chunks)


def _cache_path(parquet_path: pathlib.Path, *, max_atoms: int) -> pathlib.Path:
    """Returns the sibling NPZ cache path for a full Parquet split."""
    return parquet_path.with_name(f"{parquet_path.stem}_collated_{max_atoms}.npz")


def _load_cached_batch(cache_path: pathlib.Path) -> batch.Batch:
    """Loads a pre-collated batch from an NPZ archive."""
    with np.load(cache_path) as archive:
        return batch.Batch(
            atom_features=archive["atom_features"],
            bond_features=archive["bond_features"],
            atom_mask=archive["atom_mask"],
            positions=archive["positions"],
            labels=archive["labels"],
        )


def _save_cached_batch(cache_path: pathlib.Path, dataset: batch.Batch) -> None:
    """Atomically writes a pre-collated batch to an NPZ archive."""
    assert dataset.positions is not None
    assert dataset.labels is not None
    tmp_path = cache_path.with_suffix(".npz.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            atom_features=dataset.atom_features,
            bond_features=dataset.bond_features,
            atom_mask=dataset.atom_mask,
            positions=dataset.positions,
            labels=dataset.labels,
        )
    tmp_path.replace(cache_path)


class DataSource:
    """Loads a QM9 Parquet split into a pre-collated array store and yields batches."""

    def __init__(
        self,
        parquet_path: pathlib.Path,
        *,
        max_atoms: int = batch.MAX_ATOMS,
        max_molecules: int | None = None,
        use_cache: bool = True,
    ) -> None:
        cache_file = _cache_path(parquet_path, max_atoms=max_atoms)
        can_use_cache = use_cache and max_molecules is None and parquet_path.is_file()
        if (
            can_use_cache
            and cache_file.is_file()
            and cache_file.stat().st_mtime >= parquet_path.stat().st_mtime
        ):
            self._dataset = _load_cached_batch(cache_file)
            self._num_molecules = int(self._dataset.atom_mask.shape[0])
            _LOGGER.info(
                "Loaded cached DataSource from %s with %d molecules (max_atoms=%d).",
                cache_file,
                self._num_molecules,
                max_atoms,
            )
            return

        molecules = schema.load_molecules(parquet_path, max_molecules=max_molecules)
        self._num_molecules = len(molecules)
        self._dataset = _collate_molecules(molecules, max_atoms=max_atoms)
        if can_use_cache:
            _save_cached_batch(cache_file, self._dataset)
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
            drop_last: Whether to drop the trailing batch if smaller than the target batch size.

        Yields:
            Sliced batches.
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
