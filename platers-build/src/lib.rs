//! Index building tools for plate solving.
//!
//! This crate builds the quad-based `.qidx` indices used for blind astrometric
//! calibration. [`TiledBuilder`] writes a per-scale set of `HEALPix` tiles (one file
//! per tile/scale) so a solve loads only the tiles its field touches;
//! [`merge_scale_indices`] collapses those tiles into one all-sky index per scale
//! for fast blind solving. The on-disk format is [`platers_core::flatindex`].

pub mod builder;
pub mod catalog_import;
pub mod merge;
pub mod tiled;

pub use builder::{IndexBuilder, IndexConfig, IndexedQuad};
pub use catalog_import::{
    import_catalog, CatalogSource, GaiaDr3, ImportFilter, ImportStats, Tycho2,
};
pub use merge::merge_scale_indices;
pub use tiled::{tile_depth_for_fov, TiledBuilder};
