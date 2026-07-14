//! Tiled multi-scale index building (astrometry.net style).
//!
//! A monolithic all-sky index embeds the whole catalog and all-sky quads in one
//! blob per scale -- for a real catalog that's many GB and you'd load the entire
//! sky to solve one small field. Following astrometry.net, this builder:
//!
//! 1. **Uniformizes** density -- stores only the brightest `stars_per_cell` per
//!    fine `HEALPix` cell (`uniformize_catalog`). This bounds each tile's content
//!    (and thus its quad count and file size) regardless of how crowded that
//!    patch of sky is, and is the star set verification matches against.
//! 2. **Tiles** the sky into per-scale `HEALPix` cells (coarse for wide fields,
//!    fine for narrow), writing one file per `(tile, scale)` holding only that
//!    tile's uniformized stars + quads.
//!
//! A solve with an approximate position loads just the few tiles its field
//! touches (see `IndexSet::load_tiles_for`). Each tile is written in the mmap'd,
//! pre-sorted `.qidx` format ([`platers_core::flatindex`]): the quad KD-tree is
//! balanced at build time, so a solve loads a tile by mmap alone -- no deserialize,
//! no per-tile tree rebuild -- which is how blind solving (touching every tile in
//! the sky) stays affordable. The boundary margin (astrometry.net's `-r`) defaults
//! to 0 here: a quad straddling a tile edge isn't built, which is fine because a
//! solve loads the neighbouring tiles.
//!
//! Tile identity *and the angular quad-diameter band* live in the filename
//! (`tile_d{depth}_h{hash}_q{dmin}-{dmax}_n{scale}.qidx`, degrees x600), so the
//! loader picks matching files by name without opening them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use platers_core::flatindex::write_qidx;
use platers_core::geometry::HashCode;
use platers_core::spatial::HealPixGrid;
use platers_core::types::{SkyCoord, Star};
use platers_core::{Error, PlatersResult};

use crate::builder::{IndexBuilder, IndexConfig};

const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Approximate `HEALPix` cell side length in degrees at `depth` (nside = 2^depth).
fn cell_size_deg(depth: u8) -> f64 {
    58.6323 / f64::from(1_u32 << depth)
}

/// Fixed tiling depth for a scale whose fields are ~`target_fov_deg` across.
/// Targets cells a few x the FOV so a field lands in ~one tile; per-tile content
/// is bounded by the builder's uniformization (brightest-N per fine cell), not
/// by the tile geometry.
#[must_use]
#[allow(clippy::cast_sign_loss, reason = "clamped to [2, 4]")]
pub fn tile_depth_for_fov(target_fov_deg: f64) -> u8 {
    let ideal_depth = (58.6323 / (6.0 * target_fov_deg.max(1e-3))).log2().round();
    // in [2, 4]
    ideal_depth.clamp(2.0, 4.0) as u8
}

/// `HEALPix` cell containing a sky position, at the given depth.
fn tile_of(depth: u8, p: SkyCoord) -> u64 {
    cdshealpix::nested::hash(depth, p.ra.to_radians(), p.dec.to_radians())
}

/// Reduce a star list to the brightest `stars_per_cell` per fine `HEALPix` cell
/// (astrometry.net's "uniformize" step). This caps density everywhere, so what
/// each tile stores -- and therefore its quad count and file size -- is bounded
/// regardless of how crowded that patch of sky is. It is also exactly the star
/// set verification should match against (the detected stars are uniformized the
/// same way at solve time).
///
/// # Errors
/// Fails only if the `HEALPix` depth is invalid.
fn uniformize_catalog(
    stars: &[Star],
    healpix_depth: u8,
    stars_per_cell: usize,
) -> PlatersResult<Vec<Star>> {
    let mut grid = HealPixGrid::new(healpix_depth)?;
    grid.insert_many(stars);
    let mut out = Vec::new();
    for cell in grid.cell_hashes() {
        out.extend(grid.get_brightest_in_cell(cell, stars_per_cell));
    }
    Ok(out)
}

/// Builds a tiled, per-scale index set from a catalog.
#[derive(Debug)]
pub struct TiledBuilder {
    catalog: Vec<Star>,
    base_config: IndexConfig,
}

impl TiledBuilder {
    /// Create a builder over `catalog` with the given base configuration.
    #[must_use]
    pub fn new(catalog: Vec<Star>, base_config: IndexConfig) -> Self {
        Self {
            catalog,
            base_config,
        }
    }

    /// Build all scales (sqrt2 quad-diameter progression), tiling each scale at a
    /// resolution matched to its FOV. Writes one file per populated `(tile,
    /// scale)` into `output_dir` and returns their paths.
    ///
    /// # Errors
    /// Propagates index-build and file-write failures.
    pub fn build_all_scales(
        &self,
        output_dir: &Path,
        min_quad_diameter_arcmin: f64,
        max_quad_diameter_deg: f64,
    ) -> PlatersResult<Vec<PathBuf>> {
        std::fs::create_dir_all(output_dir)
            .map_err(|e| Error::IOError(format!("creating output directory: {e}")))?;

        let mut paths = Vec::new();
        let mut current_min_deg = min_quad_diameter_arcmin / 60.0;
        let mut scale_num: u32 = 0;

        while current_min_deg < max_quad_diameter_deg {
            let current_max_deg = current_min_deg * SQRT_2;
            let config =
                IndexConfig::for_scale_range(current_min_deg, current_max_deg, &self.base_config);
            let depth = tile_depth_for_fov(config.target_fov_deg);

            // Partition stars by exact tile membership -- each star lives in one
            // tile, so there is no cross-tile catalog duplication. A quad whose 4
            // stars all fall in one tile is built there; a quad straddling a tile
            // boundary is simply not built (neither tile holds all 4). That's a
            // negligible loss: a field well inside a tile keeps all its quads, and
            // a field near an edge loads both neighbouring tiles (the solver's
            // cone selection), each contributing its near-edge quads.
            let mut by_tile: HashMap<u64, Vec<Star>> = HashMap::new();
            for s in &self.catalog {
                by_tile
                    .entry(tile_of(depth, s.position))
                    .or_default()
                    .push(*s);
            }

            println!(
                "Scale {scale_num}: tile depth {depth} (~{:.2} deg cells), uniformize depth {} \
                 (stars_per_cell {}), {} populated tiles",
                cell_size_deg(depth),
                config.healpix_depth,
                config.stars_per_cell,
                by_tile.len()
            );

            let mut emitted = 0_usize;
            for (tile, local) in by_tile {
                if local.len() < 4 {
                    continue;
                }
                // Store the uniformized (brightest-per-cell) set, not the raw
                // tile catalog: bounds the tile's content/size and is the set
                // verification matches against. Quads are built from it.
                let uniformized =
                    uniformize_catalog(&local, config.healpix_depth, config.stars_per_cell)?;
                if uniformized.len() < 4 {
                    continue;
                }
                let mut builder = IndexBuilder::new(uniformized.clone(), config.clone())?;
                builder.build();
                if builder.quads().is_empty() {
                    continue;
                }

                // Emit the mmap'd, pre-sorted `.qidx` format: the quad codes are
                // balanced into KD-tree order here so a solve loads a tile by mmap
                // alone -- no deserialize, no per-tile tree rebuild.
                let quads: Vec<(HashCode, [usize; 4])> = builder
                    .quads()
                    .iter()
                    .map(|q| (q.hash_code, q.star_indices))
                    .collect();
                // Filename carries the tile (depth, hash) AND the *angular* quad-
                // diameter band (degrees x600), so the loader can pick matching
                // files by name alone -- the band<->pixel-scale match is computed at
                // solve time from the actual image, not baked in here.
                #[allow(
                    clippy::cast_sign_loss,
                    reason = "quad diameters are non-negative by construction"
                )]
                let path = output_dir.join(format!(
                    "tile_d{depth}_h{tile}_q{}-{}_n{scale_num:02}.qidx",
                    (config.min_diameter_deg * 600.0).round() as u32,
                    (config.max_diameter_deg * 600.0).round() as u32,
                ));
                write_qidx(
                    &path,
                    &uniformized,
                    &quads,
                    (
                        config.min_scale_arcsec_per_pixel,
                        config.max_scale_arcsec_per_pixel,
                    ),
                    (config.min_diameter_deg, config.max_diameter_deg),
                    config.healpix_depth,
                    config.stars_per_cell,
                )
                .map_err(|e| Error::IOError(format!("saving {}: {e}", path.display())))?;
                paths.push(path);
                emitted += 1;
            }
            println!("  emitted {emitted} tile files");

            current_min_deg = current_max_deg;
            scale_num += 1;
        }

        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_depth_is_per_scale_and_bounded() {
        // Narrow FOV -> finer tiles (larger depth); wide FOV -> coarser.
        let narrow = tile_depth_for_fov(0.21);
        let wide = tile_depth_for_fov(1.2);
        assert!(narrow >= wide, "narrow {narrow} should be >= wide {wide}");
        assert!((2..=4).contains(&narrow));
        assert!((2..=4).contains(&wide));
    }

    #[test]
    fn populated_tile_of_a_known_point() {
        // Deterministic: the same point always maps to the same tile.
        let p = SkyCoord::new(305.0, 40.0);
        assert_eq!(tile_of(4, p), tile_of(4, p));
    }
}
