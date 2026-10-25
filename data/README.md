<!-- AGENT NOTICE: Keep this README synchronized with the files and data preparation workflow in this directory whenever code changes. -->

# QM9 Dataset Preparation

`download_qm9.py` is a data ingestion script that downloads the raw QM9 archive from Figshare and writes the standard DimeNet splits (`train.parquet`, `validation.parquet`, and `test.parquet`):

* Filters out the 3,054 uncharacterized molecules, leaving 130,831 characterized entries partitioned into 110,000 training, 10,000 validation, and 10,831 test molecules (seed 42).
* Parses Fortran-formatted coordinates, atomic numbers, and GDB-17 / relaxed SMILES strings into Zstandard-compressed Parquet tables padded to 29 atoms (`index`, `tag`, `num_atoms`, `smiles`, `smiles_relaxed`, `charges`, and `positions`).
