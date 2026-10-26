<!-- AGENT NOTICE: Keep this README synchronized with the files and data preparation workflow in this directory whenever code changes. -->

# QM9 Dataset Preparation

This directory holds the QM9 dataset downloader and the resulting Parquet splits.

* Downloads the raw QM9 archive from Figshare, filters out uncharacterized molecules, and partitions the remaining entries into the standard DimeNet training, validation, and test splits.
* Parses Fortran-formatted coordinates, atomic numbers, and GDB-17 / relaxed SMILES strings into Zstandard-compressed Parquet tables.
