//! Core plate solving library for astronomical astrometry.
//!
//! This crate provides the fundamental data structures and algorithms for blind
//! astrometric calibration using geometric hash-based pattern matching: given a
//! list of detected star positions (from any source extractor) and a pre-built
//! quad index, it recovers the image's WCS -- where on the sky it was taken, at
//! what scale and rotation.
//!
//! ## Solving
//!
//! ```no_run
//! use platers_core::{DetectedField, IndexSet, PlateSolver, QueryConfig, ScaleRange};
//!
//! # fn main() -> platers_core::PlatersResult<()> {
//! let index = IndexSet::load_from_directory("index")?;
//! let stars = vec![/* DetectedStar { x, y, flux } from your source extractor */];
//! let field = DetectedField::new(stars, 2048, 1489);
//! let config = QueryConfig {
//!     // A pixel-scale hint restricts the search to the matching index
//!     // tiers -- typically 10-100x faster than a scale-blind solve.
//!     scale_hint: Some(ScaleRange::from_nominal(0.39, 0.05)),
//!     ..QueryConfig::default()
//! };
//! let solution = PlateSolver::new(index, config).solve(&field)?;
//! println!("center: {:?}", solution.wcs.center);
//! # Ok(())
//! # }
//! ```
//!
//! Complete runnable programs live in this crate's `examples/` directory
//! (`cargo run --release -p platers-core --example solve_from_json`); building
//! an index is covered by the workspace readme and the `platers-build` crate.

pub mod catalog;
pub mod catalog_index;
pub mod errors;
pub mod flatindex;
pub mod geometry;
pub mod index;
pub mod query;
pub mod refinement;
pub mod solver;
pub mod spatial;
pub mod types;
pub mod verification;
pub mod wcs;

pub use catalog::{load_catalog_parquet, save_catalog_parquet, CATALOG_EPOCH};
pub use catalog_index::CatalogIndex;
pub use errors::{Error, PlatersResult};
pub use index::{CatalogQuad, IndexSet, IndexStats, LoadedIndex, QuadMatch};
pub use query::{ImageQuad, ImageQuadGenerator, PositionHint, QueryConfig, ScaleRange};
pub use refinement::{IterativeRefiner, RefinementConfig, RefinementResult, StarMatch};
pub use solver::{PlateSolver, QueryStats, SolveResult};
pub use types::{DetectedField, DetectedStar, PixelCoord, SkyCoord, Star};
pub use verification::{VerificationConfig, VerificationResult, Verifier};
pub use wcs::{Projector, WcsHypothesis};
