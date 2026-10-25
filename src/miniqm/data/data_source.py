"""In-memory data source that pre-collates a Parquet split and yields batches."""

from collections.abc import Collection, Iterator, Sequence
import enum
import itertools
import logging
import os
import pathlib
from typing import Final

import joblib
import numpy as np
from rich import progress

from miniqm.data import batch, schema

_LOGGER: Final = logging.getLogger(__name__)
_PARALLEL_MIN_MOLECULES: Final = 512
_CHUNK_SIZE: Final = 2_048
_MAX_WORKERS: Final = 16


class _CacheKey(enum.StrEnum):
    """Array keys stored in pre-collated NPZ split caches."""

    ATOM_FEATURES = enum.auto()
    BOND_FEATURES = enum.auto()
    ATOM_MASK = enum.auto()
    POSITIONS = enum.auto()


class DataSource:
    """Pre-collated molecular array store that yields padded batches."""

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
            cached_batch = _load_cached_batch(cache_file, max_atoms=max_atoms)
            if cached_batch is not None:
                self._dataset = cached_batch
                num_molecules, _ = self._dataset.atom_mask.shape
                self._num_molecules = int(num_molecules)
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

        Raises:
            ValueError: If batch_size is not positive.
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
                atom_cond_mask=self._dataset.atom_cond_mask[batch_indices],
                bond_cond_mask=self._dataset.bond_cond_mask[batch_indices],
            )


def _concatenate_batches(batches: Sequence[batch.Batch]) -> batch.Batch:
    """Concatenates a non-empty sequence of padded batches along the molecule axis."""
    if len(batches) == 1:
        return batches[0]
    return batch.Batch(
        atom_features=np.concatenate(
            tuple(sub_batch.atom_features for sub_batch in batches), axis=0
        ),
        bond_features=np.concatenate(
            tuple(sub_batch.bond_features for sub_batch in batches), axis=0
        ),
        atom_mask=np.concatenate(tuple(sub_batch.atom_mask for sub_batch in batches), axis=0),
        positions=np.concatenate(tuple(sub_batch.positions for sub_batch in batches), axis=0),
        atom_cond_mask=np.concatenate(
            tuple(sub_batch.atom_cond_mask for sub_batch in batches), axis=0
        ),
        bond_cond_mask=np.concatenate(
            tuple(sub_batch.bond_cond_mask for sub_batch in batches), axis=0
        ),
    )


def _collate_molecules(
    molecules: Collection[schema.Molecule],
    *,
    max_atoms: int,
    max_workers: int = _MAX_WORKERS,
    parallel_min_molecules: int = _PARALLEL_MIN_MOLECULES,
    chunk_size: int = _CHUNK_SIZE,
) -> batch.Batch:
    """Collates molecules into a single batch, parallelizing large splits across CPU workers."""
    num_workers = min(max_workers, os.cpu_count() or 1)
    if len(molecules) < parallel_min_molecules or num_workers <= 1:
        return batch.collate_batch(molecules, max_atoms=max_atoms)

    chunks = tuple(itertools.batched(molecules, chunk_size))
    jobs = (joblib.delayed(batch.collate_batch)(chunk, max_atoms=max_atoms) for chunk in chunks)
    collated_chunks = tuple(
        progress.track(
            joblib.Parallel(n_jobs=num_workers, return_as="generator")(jobs),
            total=len(chunks),
            description=f"Featurizing {len(molecules):,} molecules ({num_workers} workers)",
        )
    )
    return _concatenate_batches(collated_chunks)


def _cache_path(parquet_path: pathlib.Path, *, max_atoms: int) -> pathlib.Path:
    """Returns the sibling NPZ cache path for a full Parquet split."""
    return parquet_path.with_name(f"{parquet_path.stem}_collated_{max_atoms}.npz")


def _load_cached_batch(cache_path: pathlib.Path, *, max_atoms: int) -> batch.Batch | None:
    """Loads a pre-collated batch from an NPZ archive if its schema and dimensions match."""
    with np.load(cache_path) as archive:
        if not all(key in archive for key in _CacheKey):
            return None
        atom_features = archive[_CacheKey.ATOM_FEATURES]
        bond_features = archive[_CacheKey.BOND_FEATURES]
        _, cached_max_atoms, cached_num_atom_features = atom_features.shape
        if cached_max_atoms != max_atoms or cached_num_atom_features != batch.NUM_ATOM_FEATURES:
            return None
        _, bond_atoms_i, bond_atoms_j, cached_num_bond_features = bond_features.shape
        if (
            bond_atoms_i != max_atoms
            or bond_atoms_j != max_atoms
            or cached_num_bond_features != batch.NUM_BOND_FEATURES
        ):
            return None
        atom_mask = archive[_CacheKey.ATOM_MASK]
        atom_cond_mask = np.broadcast_to(atom_mask[..., None], atom_features.shape).copy()
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        bond_cond_mask = np.broadcast_to(pair_mask[..., None], bond_features.shape).copy()
        return batch.Batch(
            atom_features=atom_features,
            bond_features=bond_features,
            atom_mask=atom_mask,
            positions=archive[_CacheKey.POSITIONS],
            atom_cond_mask=atom_cond_mask,
            bond_cond_mask=bond_cond_mask,
        )


def _save_cached_batch(cache_path: pathlib.Path, dataset: batch.Batch) -> None:
    """Atomically writes a pre-collated batch to an NPZ archive."""
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.npz.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            **{
                _CacheKey.ATOM_FEATURES: dataset.atom_features,
                _CacheKey.BOND_FEATURES: dataset.bond_features,
                _CacheKey.ATOM_MASK: dataset.atom_mask,
                _CacheKey.POSITIONS: dataset.positions,
            },
        )
    tmp_path.replace(cache_path)
