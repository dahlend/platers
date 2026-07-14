//! Merge a directory of per-tile indices into one all-sky `.qidx` *per scale*.
//!
//! A blind solve has no position hint, so it must consider every tile in the sky.
//! Searching thousands of small per-tile KD-trees means thousands of tree descents
//! per image quad; merging a scale's tiles into a single balanced tree turns that
//! into one descent per image quad (astrometry.net's per-scale all-sky index). The
//! fine per-tile directory is still what position-hinted solves use (its cone
//! pruning needs the per-tile `HEALPix` identity); the merged files are a blind-only
//! artifact, kept in their own directory.
//!
//! Each output is named `allsky_q{dmin}-{dmax}_n{scale}.qidx` -- the `q` band lets
//! the loader pick it by scale, and the `allsky_` prefix marks it as covering the
//! whole sky (so it always passes a position cone, which a blind hit's final
//! hinted refine relies on).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use platers_core::flatindex::write_qidx;
use platers_core::geometry::HashCode;
use platers_core::index::LoadedIndex;
use platers_core::types::Star;
use platers_core::{Error, PlatersResult};

/// Parse the trailing `_n{scale}.{ext}` of an index filename into its scale index.
fn scale_num_of(path: &Path) -> Option<u32> {
    let name = path.file_name()?.to_str()?;
    let (_, rest) = name.rsplit_once("_n")?;
    rest.split_once('.')?.0.parse().ok()
}

/// Merge every per-tile index in `src_dir` into one all-sky `.qidx` per scale,
/// written to `dst_dir`. Returns the paths written.
///
/// Tiles are grouped by their scale number; within a scale, all stars are
/// concatenated (quad star indices remapped to the combined list) and all quads
/// pooled into one balanced tree. The angular quad-diameter band is the union of
/// the merged tiles' bands.
///
/// # Errors
/// Propagates directory-read, tile-load, and file-write failures.
pub fn merge_scale_indices(src_dir: &Path, dst_dir: &Path) -> PlatersResult<Vec<PathBuf>> {
    std::fs::create_dir_all(dst_dir)
        .map_err(|e| Error::IOError(format!("creating dst dir: {e}")))?;

    let mut by_scale: BTreeMap<u32, Vec<PathBuf>> = BTreeMap::new();
    for entry in std::fs::read_dir(src_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("qidx") {
            continue;
        }
        if let Some(n) = scale_num_of(&path) {
            by_scale.entry(n).or_default().push(path);
        }
    }

    let mut written = Vec::new();
    for (scale, paths) in by_scale {
        let mut stars: Vec<Star> = Vec::new();
        let mut quads: Vec<(HashCode, [usize; 4])> = Vec::new();
        let mut band = (f64::INFINITY, f64::NEG_INFINITY);
        let mut healpix_depth = 0_u8;
        let mut stars_per_cell = 0_usize;

        for path in &paths {
            let idx = LoadedIndex::open(path)
                .map_err(|e| Error::IOError(format!("loading {}: {e}", path.display())))?;
            let offset = stars.len();
            stars.extend(idx.catalog());
            for qi in 0..idx.num_quads() {
                let q = idx.quad(qi);
                let remapped = [
                    q.star_indices[0] + offset,
                    q.star_indices[1] + offset,
                    q.star_indices[2] + offset,
                    q.star_indices[3] + offset,
                ];
                quads.push((q.hash_code, remapped));
            }
            band.0 = band.0.min(idx.diameter_range.0);
            band.1 = band.1.max(idx.diameter_range.1);
            healpix_depth = idx.config.healpix_depth;
            stars_per_cell = idx.config.stars_per_cell;
        }

        #[allow(
            clippy::cast_sign_loss,
            reason = "quad diameters are non-negative by construction"
        )]
        let out = dst_dir.join(format!(
            "allsky_q{}-{}_n{scale:02}.qidx",
            (band.0 * 600.0).round() as u32,
            (band.1 * 600.0).round() as u32,
        ));
        // Scale band (arcsec/px) is derived at solve time from the angular `q`
        // band + image width, so the stored scale range here is only metadata;
        // carry a permissive placeholder.
        write_qidx(
            &out,
            &stars,
            &quads,
            (0.0, 1000.0),
            band,
            healpix_depth,
            stars_per_cell,
        )?;
        println!(
            "scale {scale:02}: merged {} tiles -> {} stars, {} quads -> {}",
            paths.len(),
            stars.len(),
            quads.len(),
            out.display()
        );
        written.push(out);
    }
    Ok(written)
}
