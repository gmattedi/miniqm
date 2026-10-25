"""Downloads the QM9 dataset and writes the DimeNet train/validation/test splits to Parquet."""

from collections.abc import Container, Iterable, Mapping
from collections.abc import Set as AbstractSet
import logging
import pathlib
import tarfile
from typing import Final
import urllib.request

import immutabledict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from miniqm.data import schema

_LOGGER: Final = logging.getLogger(__name__)

_UNCHARACTERIZED_URL: Final = "https://ndownloader.figshare.com/files/3195404"
_MOLECULES_URL: Final = "https://ndownloader.figshare.com/files/3195389"

_TOTAL_MOLECULES: Final = 133_885
_CHARACTERIZED_MOLECULES: Final = 130_831
_MAX_ATOMS: Final = 29
_NUM_COORDS: Final = 3

_UNCHARACTERIZED_START_LINE: Final = 9
_UNCHARACTERIZED_END_LINE: Final = -1
_ATOM_LINE_START: Final = 2
_SMILES_LINE_OFFSET: Final = 3
_LOG_EVERY_MOLECULES: Final = 25_000
_PARQUET_COMPRESSION: Final = "zstd"

# DimeNet split configuration from tensorflow_datasets (seed 42, 110k/10k/10,831).
_DIMENET_SEED: Final = 42
_DIMENET_TRAIN_SIZE: Final = 110_000
_DIMENET_VALIDATION_SIZE: Final = 10_000

_ATOMIC_NUMBERS_BY_SYMBOL: Final = immutabledict.immutabledict(
    {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
    }
)


def _fetch_text(url: str) -> str:
    """Downloads a UTF-8 text file from a URL."""
    with urllib.request.urlopen(url) as response:
        payload = response.read()
        assert isinstance(payload, bytes)
        return payload.decode("utf-8")


def _load_uncharacterized_ids() -> frozenset[int]:
    """Loads the 1-based indices of the 3,054 uncharacterized QM9 molecules."""
    lines = _fetch_text(_UNCHARACTERIZED_URL).splitlines()[
        _UNCHARACTERIZED_START_LINE:_UNCHARACTERIZED_END_LINE
    ]
    return frozenset(int(line.split()[0]) for line in lines)


def _get_dimenet_split_ids(
    uncharacterized_ids: AbstractSet[int],
) -> dict[schema.Split, tuple[int, ...]]:
    """Computes the DimeNet train/validation/test 1-based molecule IDs."""
    characterized_ids = np.array(
        sorted(set(range(1, _TOTAL_MOLECULES + 1)) - uncharacterized_ids),
        dtype=np.int64,
    )
    if characterized_ids.size != _CHARACTERIZED_MOLECULES:
        raise ValueError(
            f"Expected {_CHARACTERIZED_MOLECULES} characterized molecules, "
            f"got {characterized_ids.size}."
        )

    # Uses legacy RandomState to match the exact permutation in tensorflow_datasets.
    rng = np.random.RandomState(_DIMENET_SEED)
    permuted_ids = rng.permutation(characterized_ids)
    train_end = _DIMENET_TRAIN_SIZE
    val_end = _DIMENET_TRAIN_SIZE + _DIMENET_VALIDATION_SIZE
    return {
        schema.Split.TRAIN: tuple(int(mol_id) for mol_id in permuted_ids[:train_end]),
        schema.Split.VALIDATION: tuple(int(mol_id) for mol_id in permuted_ids[train_end:val_end]),
        schema.Split.TEST: tuple(int(mol_id) for mol_id in permuted_ids[val_end:]),
    }


def _parse_float(token: str) -> float:
    """Parses a QM9 float token, including Fortran exponential notation."""
    return float(token.replace("*^", "e"))


def _parse_molecule_xyz(xyz_text: str) -> dict[str, object]:
    """Parses a single QM9 coordinate block into a zero-padded metadata and coordinate record."""
    lines = xyz_text.splitlines()
    num_atoms = int(lines[0].rstrip())

    tag, mol_index_str, *_ = lines[1].split()
    mol_index = int(mol_index_str)

    charges = np.zeros(_MAX_ATOMS, dtype=np.int64)
    positions = np.zeros((_MAX_ATOMS, _NUM_COORDS), dtype=np.float32)

    for atom_idx, atom_line in enumerate(lines[_ATOM_LINE_START : _ATOM_LINE_START + num_atoms]):
        symbol, *coord_tokens, _ = atom_line.split()
        charges[atom_idx] = _ATOMIC_NUMBERS_BY_SYMBOL[symbol]
        positions[atom_idx] = tuple(_parse_float(token) for token in coord_tokens)

    smiles_gdb17, smiles_relaxed = lines[num_atoms + _SMILES_LINE_OFFSET].split()

    return {
        schema.Column.INDEX: mol_index,
        schema.Column.TAG: tag,
        schema.Column.NUM_ATOMS: num_atoms,
        schema.Column.SMILES: smiles_gdb17,
        schema.Column.SMILES_RELAXED: smiles_relaxed,
        schema.Column.CHARGES: charges.tolist(),
        schema.Column.POSITIONS: positions.tolist(),
    }


def _load_characterized_molecules(
    uncharacterized_ids: Container[int],
) -> dict[int, dict[str, object]]:
    """Streams the remote molecule archive and parses all characterized entries."""
    _LOGGER.info("Streaming characterized molecules from %s ...", _MOLECULES_URL)
    molecules_by_id: dict[int, dict[str, object]] = {}

    with (
        urllib.request.urlopen(_MOLECULES_URL) as response,
        tarfile.open(fileobj=response, mode="r|bz2") as archive,
    ):
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xyz"):
                continue
            # Filenames are formatted as dsgdb9nsd_XXXXXX.xyz.
            mol_id = int(member.name.rsplit("_", maxsplit=1)[-1].removesuffix(".xyz"))
            if mol_id in uncharacterized_ids:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not extract {member.name}.")
            xyz_text = extracted.read().decode("utf-8")
            molecules_by_id[mol_id] = _parse_molecule_xyz(xyz_text)
            if len(molecules_by_id) % _LOG_EVERY_MOLECULES == 0:
                _LOGGER.info(
                    "Parsed %d / %d molecules ...",
                    len(molecules_by_id),
                    _CHARACTERIZED_MOLECULES,
                )

    if len(molecules_by_id) != _CHARACTERIZED_MOLECULES:
        raise ValueError(
            f"Expected {_CHARACTERIZED_MOLECULES} parsed molecules, got {len(molecules_by_id)}."
        )
    return molecules_by_id


def _build_parquet_schema() -> pa.Schema:
    """Builds the Arrow schema for molecule metadata and 3D coordinates."""
    return pa.schema(
        (
            pa.field(schema.Column.INDEX, pa.int64()),
            pa.field(schema.Column.TAG, pa.string()),
            pa.field(schema.Column.NUM_ATOMS, pa.int64()),
            pa.field(schema.Column.SMILES, pa.string()),
            pa.field(schema.Column.SMILES_RELAXED, pa.string()),
            pa.field(schema.Column.CHARGES, pa.list_(pa.int64(), _MAX_ATOMS)),
            pa.field(
                schema.Column.POSITIONS,
                pa.list_(pa.list_(pa.float32(), _NUM_COORDS), _MAX_ATOMS),
            ),
        )
    )


def _write_split_parquet(
    split_ids: Iterable[int],
    molecules_by_id: Mapping[int, Mapping[str, object]],
    parquet_schema: pa.Schema,
    output_path: pathlib.Path,
) -> None:
    """Writes a split of molecule records to a Parquet file."""
    rows = tuple(molecules_by_id[mol_id] for mol_id in split_ids)
    table = pa.Table.from_pylist(rows, schema=parquet_schema)
    pq.write_table(table, output_path, compression=_PARQUET_COMPRESSION)
    _LOGGER.info("Wrote %d rows to %s", table.num_rows, output_path)


def main() -> None:
    """Downloads QM9 and writes DimeNet train, validation, and test Parquet files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    output_dir = pathlib.Path(__file__).resolve().parent

    uncharacterized_ids = _load_uncharacterized_ids()
    split_ids_by_name = _get_dimenet_split_ids(uncharacterized_ids)
    molecules_by_id = _load_characterized_molecules(uncharacterized_ids)

    parquet_schema = _build_parquet_schema()
    for split_name, split_ids in split_ids_by_name.items():
        output_path = output_dir / f"{split_name}.parquet"
        _write_split_parquet(split_ids, molecules_by_id, parquet_schema, output_path)


if __name__ == "__main__":
    main()
