# Data Loading & 2D Molecular Featurization

`miniqm.data` loads QM9 splits, aligns heavy-atom RDKit molecular graphs to 3D coordinate atom orderings, and collates padded JAX batches:

* Defines the dataset schema and unpadded molecule container (atomic numbers and 3D coordinates in Angstroms), slicing out padding slots when loading splits.
* Aligns SMILES graphs to the heavy-atom XYZ ordering using 3D-validated bond perception and substructure matching. Perceives 3D Cahn-Ingold-Prelog ($R$/$S$) stereochemistry from the coordinates and encodes one-hot heavy-atom and bond features.
* Centers 3D coordinates at the origin, parses SMILES/SMARTS templates with dummy atoms (`U`, `[U]`, `*`) and dummy bonds (`~`) into per-channel conditioning masks, and reconstructs sanitized RDKit molecules from denoised predictions or known graphs.
* Parallelizes featurization across worker processes, caches pre-collated archives on disk, and yields shuffled or sequential JAX batches.
