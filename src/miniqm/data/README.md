<!-- AGENT NOTICE: Keep this README synchronized with the schema, featurization, and batching logic in this directory whenever code changes. -->

# Data Loading & 2D Molecular Featurization

`miniqm.data` is the data-loading and 2D molecular featurization package that reads Parquet splits, aligns RDKit molecular graphs to 3D coordinate atom orderings, and collates padded JAX batches:

* `schema.py` defines `Split`, `Column`, and `Molecule` (unpadded atomic numbers and 3D coordinates in Angstroms) and slices out padding slots when loading Parquet splits.
* `batch.py` aligns SMILES graphs (`smiles` and `smiles_relaxed`) to the XYZ atom ordering using 3D-validated heavy-atom matching and substructure matching (`rdDetermineBonds`). It perceives 3D Cahn-Ingold-Prelog (`R`/`S`) stereochemistry from the coordinates and encodes 22-dimensional atom features and 6-dimensional bond features.
* `batch.py` centers 3D coordinates at the origin, collates inference-only SMILES strings (`collate_smiles`), builds explicit-hydrogen RDKit 3D conformers (`build_conformer_mol`), and converts padded NumPy batches (`Batch`) into JAX PyTrees (`JaxBatch`).
* `data_source.py` provides `DataSource`, which parallelizes featurization across `forkserver` worker processes, caches pre-collated `<split>_collated_<max_atoms>.npz` archives on disk, and yields shuffled or sequential batches.
