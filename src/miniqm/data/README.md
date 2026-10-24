<!-- AGENT NOTICE: Keep this README synchronized with the schema, featurization, and batching logic in this directory whenever code changes. -->

# Data & Graph Featurization

Reads QM9 Parquet splits into unpadded molecule records, aligns 2D SMILES topology to 3D XYZ atom ordering, and collates fixed-shape batches padded to 29 atoms.

* Atom features encode one-hot atomic number (H, C, N, O, F), total degree, formal charge, hybridization, aromaticity, ring membership, and Cahn-Ingold-Prelog R/S chirality.
* Bond features encode pairwise bond order (none, single, double, triple, aromatic) and ring membership across all atom pairs.
* SMILES atom order is aligned to the XYZ coordinate order via heavy-atom signature matching with a 3D bond-inference substructure fallback.
* Coordinates are centered to zero center of mass per molecule before padding, and host NumPy batches convert to JAX PyTree batches on demand.
