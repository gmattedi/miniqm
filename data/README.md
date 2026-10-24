<!-- AGENT NOTICE: Keep this README synchronized with the data preparation script and Parquet split layout whenever code changes. -->

# QM9 Dataset Splits

Downloads the raw QM9 archive (133,885 small organic molecules with up to 29 atoms of H, C, N, O, and F), filters out the 3,054 molecules that fail geometry consistency checks, and writes deterministic 80/10/10 train, validation, and test Parquet files.

Each row stores the 1-based GDB9 index, SMILES strings (initial and B3LYP-relaxed), atomic numbers, DFT equilibrium 3D coordinates in Angstroms, Mulliken partial charges, and 19 scalar quantum chemical properties.
