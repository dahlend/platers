# platers-build

Index building tools for plate solving.

Builds the quad-based `.qidx` indices that [`platers-core`](https://crates.io/crates/platers-core)
solves against. `TiledBuilder` writes a per-scale set of HEALPix tiles (one file
per tile/scale) so a solve loads only the tiles its field touches;
`merge_scale_indices` collapses those tiles into one all-sky index per scale for
fast blind solving. `import_catalog` ingests Gaia DR3 / Tycho-2 exports into the
`platers-core` star format.

The crate ships binaries for the pipeline:

- `build_catalog` -- ingest a catalog export into a parquet star catalog
- `build_index` -- build the tiled `.qidx` index from a catalog
- `merge_scale_qidx` -- merge tiles into per-scale all-sky indices

Part of the [Platers](https://github.com/ddahlen/platers) workspace; see the
workspace readme for the full pipeline. Licensed under MIT OR Apache-2.0.
