<!-- AGENT NOTICE: Keep this README synchronized with the schema, featurization, and batching logic in this directory whenever code changes. -->

# Data Loading & 2D Molecular Featurization

`miniqm.data` is the data-loading and 2D molecular featurization package that reads Parquet splits, aligns heavy-atom RDKit molecular graphs to 3D coordinate atom orderings, and collates padded JAX batches:

* `schema.py` defines `Split`, `Column`, and `Molecule` (unpadded atomic numbers and 3D coordinates in Angstroms) and slices out padding slots when loading Parquet splits.
* `batch.py` aligns SMILES graphs (`smiles` and `smiles_relaxed`) to the heavy-atom XYZ ordering (`MAX_ATOMS = 9`) using 3D-validated matching and substructure matching (`rdDetermineBonds`). It perceives 3D Cahn-Ingold-Prelog (`R`/`S`) stereochemistry from the coordinates and encodes 26-dimensional one-hot heavy-atom features (element, total degree, total hydrogens, formal charge, hybridization, aromaticity, ring membership, and CIP chirality) and 6-dimensional one-hot bond features (`NONE`, `SINGLE`, `DOUBLE`, `TRIPLE`, `AROMATIC`, and ring membership).
* `batch.py` centers 3D coordinates at the origin, parses SMILES/SMARTS templates with dummy atoms (`U`, `[U]`, `*`) and dummy bonds (`~`) into per-channel conditioning masks (`collate_smiles`), reconstructs sanitized RDKit molecules from denoised one-hot predictions (`build_generated_mol`) or known graphs (`build_conformer_mol`), and converts padded NumPy batches (`Batch`) into JAX PyTrees (`JaxBatch`).
* `data_source.py` provides `DataSource`, which parallelizes featurization across `forkserver` worker processes, caches pre-collated `<split>_collated_<max_atoms>.npz` archives on disk, and yields shuffled or sequential batches.
