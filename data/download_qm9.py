"""Downloads the QM9 dataset and writes the DimeNet train/validation/test splits to Parquet."""

from collections.abc import Mapping, Sequence
import logging
import pathlib
import tarfile
from typing import Final
import urllib.request

import immutabledict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_LOGGER: Final = logging.getLogger(__name__)

_ATOMREF_URL: Final = "https://ndownloader.figshare.com/files/3195395"
_UNCHARACTERIZED_URL: Final = "https://ndownloader.figshare.com/files/3195404"
_MOLECULES_URL: Final = "https://ndownloader.figshare.com/files/3195389"

_TOTAL_MOLECULES: Final = 133_885
_CHARACTERIZED_MOLECULES: Final = 130_831
_MAX_ATOMS: Final = 29
_NUM_COORDS: Final = 3

# DimeNet split configuration from tensorflow_datasets (seed 42, 110k/10k/10,831).
_DIMENET_SEED: Final = 42
_DIMENET_TRAIN_SIZE: Final = 110_000
_DIMENET_VALIDATION_SIZE: Final = 10_000

_ATOMIC_NUMBERS_BY_SYMBOL: Final[Mapping[str, int]] = immutabledict.immutabledict(
    {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
    }
)

_SCALAR_FLOAT_LABELS: Final = (
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
)
_ATOMIZATION_TARGETS: Final = ("U0", "U", "H", "G")


def _fetch_text(url: str) -> str:
    """Downloads a UTF-8 text file from a URL."""
    with urllib.request.urlopen(url) as response:
        payload = response.read()
        assert isinstance(payload, bytes)
        return payload.decode("utf-8")


def _load_atomref() -> dict[str, dict[str, float]]:
    """Loads per-element reference thermochemical energies indexed by symbol."""
    columns = ("zpve", "U0", "U", "H", "G", "Cv")
    atomref_by_symbol: dict[str, dict[str, float]] = {}
    for line in _fetch_text(_ATOMREF_URL).splitlines()[5:10]:
        symbol, *values = line.split()
        atomref_by_symbol[symbol] = dict(zip(columns, map(float, values), strict=True))
    return atomref_by_symbol


def _load_uncharacterized_ids() -> frozenset[int]:
    """Loads the 1-based indices of the 3,054 uncharacterized QM9 molecules."""
    lines = _fetch_text(_UNCHARACTERIZED_URL).splitlines()[9:-1]
    return frozenset(int(line.split()[0]) for line in lines)


def _get_dimenet_split_ids(uncharacterized_ids: frozenset[int]) -> dict[str, tuple[int, ...]]:
    """Computes the DimeNet train/validation/test 1-based molecule IDs."""
    characterized_ids = np.array(
        sorted(set(range(1, _TOTAL_MOLECULES + 1)) - uncharacterized_ids),
        dtype=np.int64,
    )
    if characterized_ids.size != _CHARACTERIZED_MOLECULES:
        raise ValueError(
            f"Expected {_CHARACTERIZED_MOLECULES} characterized molecules, "
            f"got {characterized_ids.size}"
        )

    # Uses legacy RandomState to match the exact permutation in tensorflow_datasets.
    rng = np.random.RandomState(_DIMENET_SEED)
    permuted_ids = rng.permutation(characterized_ids)
    train_end = _DIMENET_TRAIN_SIZE
    val_end = _DIMENET_TRAIN_SIZE + _DIMENET_VALIDATION_SIZE
    return {
        "train": tuple(int(mol_id) for mol_id in permuted_ids[:train_end]),
        "validation": tuple(int(mol_id) for mol_id in permuted_ids[train_end:val_end]),
        "test": tuple(int(mol_id) for mol_id in permuted_ids[val_end:]),
    }


def _parse_float(token: str) -> float:
    """Parses a QM9 float token, handling Fortran '*^' exponential notation."""
    return float(token.replace("*^", "e"))


def _parse_molecule_xyz(
    xyz_text: str,
    atomref_by_symbol: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Parses a single QM9 .xyz file into the zero-padded TFDS feature dictionary."""
    lines = xyz_text.splitlines()
    num_atoms = int(lines[0].rstrip())

    tag, mol_index_str, *scalar_tokens = lines[1].split()
    mol_index = int(mol_index_str)
    scalar_values = dict(zip(_SCALAR_FLOAT_LABELS, map(_parse_float, scalar_tokens), strict=True))

    charges = np.zeros(_MAX_ATOMS, dtype=np.int64)
    mulliken_charges = np.zeros(_MAX_ATOMS, dtype=np.float32)
    positions = np.zeros((_MAX_ATOMS, _NUM_COORDS), dtype=np.float32)
    atom_symbols: list[str] = []

    for atom_idx, atom_line in enumerate(lines[2 : 2 + num_atoms]):
        symbol, *coord_tokens, mulliken_token = atom_line.split()
        atom_symbols.append(symbol)
        charges[atom_idx] = _ATOMIC_NUMBERS_BY_SYMBOL[symbol]
        positions[atom_idx] = tuple(_parse_float(token) for token in coord_tokens)
        mulliken_charges[atom_idx] = _parse_float(mulliken_token)

    frequencies = tuple(_parse_float(token) for token in lines[num_atoms + 2].split())
    smiles_gdb17, smiles_relaxed = lines[num_atoms + 3].split()
    inchi_corina, inchi_relaxed = lines[num_atoms + 4].split()

    atomization_values = {
        f"{target_name}_atomization": scalar_values[target_name]
        - sum(atomref_by_symbol[symbol][target_name] for symbol in atom_symbols)
        for target_name in _ATOMIZATION_TARGETS
    }

    return {
        "index": mol_index,
        "tag": tag,
        "num_atoms": num_atoms,
        "SMILES": smiles_gdb17,
        "SMILES_relaxed": smiles_relaxed,
        "InChI": inchi_corina,
        "InChI_relaxed": inchi_relaxed,
        "charges": charges.tolist(),
        "Mulliken_charges": mulliken_charges.tolist(),
        "positions": positions.tolist(),
        "frequencies": frequencies,
        **scalar_values,
        **atomization_values,
    }


def _load_characterized_molecules(
    uncharacterized_ids: frozenset[int],
    atomref_by_symbol: Mapping[str, Mapping[str, float]],
) -> dict[int, dict[str, object]]:
    """Streams the remote .xyz.tar.bz2 archive and parses all characterized molecules."""
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
                raise ValueError(f"Could not extract {member.name}")
            xyz_text = extracted.read().decode("utf-8")
            molecules_by_id[mol_id] = _parse_molecule_xyz(xyz_text, atomref_by_symbol)
            if len(molecules_by_id) % 25_000 == 0:
                _LOGGER.info(
                    "Parsed %d / %d molecules ...",
                    len(molecules_by_id),
                    _CHARACTERIZED_MOLECULES,
                )

    if len(molecules_by_id) != _CHARACTERIZED_MOLECULES:
        raise ValueError(
            f"Expected {_CHARACTERIZED_MOLECULES} parsed molecules, got {len(molecules_by_id)}"
        )
    return molecules_by_id


def _build_parquet_schema() -> pa.Schema:
    """Builds the Arrow schema matching the TFDS QM9 feature types and shapes."""
    scalar_fields = tuple(pa.field(label_name, pa.float32()) for label_name in _SCALAR_FLOAT_LABELS)
    atomization_fields = tuple(
        pa.field(f"{target_name}_atomization", pa.float32()) for target_name in _ATOMIZATION_TARGETS
    )
    return pa.schema(
        (
            pa.field("index", pa.int64()),
            pa.field("tag", pa.string()),
            pa.field("num_atoms", pa.int64()),
            pa.field("SMILES", pa.string()),
            pa.field("SMILES_relaxed", pa.string()),
            pa.field("InChI", pa.string()),
            pa.field("InChI_relaxed", pa.string()),
            pa.field("charges", pa.list_(pa.int64(), _MAX_ATOMS)),
            pa.field("Mulliken_charges", pa.list_(pa.float32(), _MAX_ATOMS)),
            pa.field("positions", pa.list_(pa.list_(pa.float32(), _NUM_COORDS), _MAX_ATOMS)),
            pa.field("frequencies", pa.list_(pa.float32())),
            *scalar_fields,
            *atomization_fields,
        )
    )


def _write_split_parquet(
    split_ids: Sequence[int],
    molecules_by_id: Mapping[int, Mapping[str, object]],
    schema: pa.Schema,
    output_path: pathlib.Path,
) -> None:
    """Writes a split of QM9 records to a Parquet file."""
    rows = tuple(molecules_by_id[mol_id] for mol_id in split_ids)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path, compression="zstd")
    _LOGGER.info("Wrote %d rows to %s", table.num_rows, output_path)


def main() -> None:
    """Downloads QM9 and writes DimeNet train, validation, and test Parquet files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    output_dir = pathlib.Path(__file__).resolve().parent

    atomref_by_symbol = _load_atomref()
    uncharacterized_ids = _load_uncharacterized_ids()
    split_ids_by_name = _get_dimenet_split_ids(uncharacterized_ids)
    molecules_by_id = _load_characterized_molecules(uncharacterized_ids, atomref_by_symbol)

    schema = _build_parquet_schema()
    for split_name, split_ids in split_ids_by_name.items():
        output_path = output_dir / f"{split_name}.parquet"
        _write_split_parquet(split_ids, molecules_by_id, schema, output_path)


if __name__ == "__main__":
    main()
