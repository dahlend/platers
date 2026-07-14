//! Index loading and matching for plate solving queries.
//!
//! This module handles:
//! - Loading serialized indices from disk
//! - Filtering indices by scale range
//! - Matching image quads to catalog quads
//! - Managing multi-scale index sets

use crate::{
    errors::{Error, PlatersResult},
    flatindex::FlatCodeIndex,
    geometry::HashCode,
    types::{SkyCoord, Star},
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The `HEALPix` tile a tiled index file covers (`depth` = order, `hash` = nested
/// cell). Recovered from the filename; `None` for monolithic (all-sky) indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRef {
    /// `HEALPix` order (`nside = 2^depth`).
    pub depth: u8,
    /// Nested `HEALPix` cell index.
    pub hash: u64,
}

/// Encoding factor for the quad-diameter range in a tiled filename (degrees x
/// this, as an integer).
const DIAMETER_FILENAME_SCALE: f64 = 600.0;

/// Parse the **quad-diameter** band (degrees) from the `q{dmin}-{dmax}_...` part of
/// a filename. The band is stored in *angular* terms (image-size-agnostic); the
/// pixel-scale match is computed at solve time from the actual image.
fn parse_band(after_q: &str) -> Option<(f64, f64)> {
    let (dmin_str, dmax_str) = after_q.split('_').next()?.split_once('-')?;
    Some((
        dmin_str.parse::<f64>().ok()? / DIAMETER_FILENAME_SCALE,
        dmax_str.parse::<f64>().ok()? / DIAMETER_FILENAME_SCALE,
    ))
}

/// Parse an index filename into its sky coverage and quad-diameter band, letting
/// the loader pick files by name alone (no deserialization). Two forms:
/// - `tile_d{depth}_h{hash}_q{dmin}-{dmax}_...` -> a `HEALPix` **tile** (cone-prunable).
/// - `allsky_q{dmin}-{dmax}_...` -> an **all-sky** per-scale index (`tile = None`):
///   it covers the whole sky, so it always intersects any position cone. This is
///   the merged, fast-blind artifact (one balanced tree per scale).
fn parse_index_meta(path: &Path) -> Option<(Option<TileRef>, (f64, f64))> {
    let name = path.file_name()?.to_str()?;
    if let Some(rest) = name.strip_prefix("tile_d") {
        let (depth_str, rest) = rest.split_once("_h")?;
        let (hash_str, rest) = rest.split_once("_q")?;
        let tile = TileRef {
            depth: depth_str.parse().ok()?,
            hash: hash_str.parse().ok()?,
        };
        Some((Some(tile), parse_band(rest)?))
    } else if let Some(rest) = name.strip_prefix("allsky_q") {
        Some((None, parse_band(rest)?))
    } else {
        None
    }
}

/// Useful quad-diameter as a fraction of the image width: a quad smaller than
/// `F_LO` of the frame is noise-dominated, larger than `F_HI` is poorly framed.
const QUAD_FRACTION_LO: f64 = 0.15;
const QUAD_FRACTION_HI: f64 = 0.45;

/// Does a quad-diameter band `(d_lo, d_hi)` degrees produce usefully-sized quads
/// for an image `image_width` px wide at some pixel scale in `scale` arcsec/px?
///
/// A quad of angular diameter `D` spans `D_arcsec / s` pixels, i.e. a fraction
/// `(D_arcsec / s) / W` of the width. Requiring that fraction to lie in
/// `[F_LO, F_HI]` gives, for the band, a pixel-scale range
/// `[d_lo*3600 / (F_HI*W), d_hi*3600 / (F_LO*W)]`; the band matches if that
/// overlaps the requested `scale`. This derives the band<->pixel-scale mapping
/// from the *actual* image width -- no fixed image-size assumption.
#[must_use]
pub fn band_matches_image(d_lo: f64, d_hi: f64, image_width: usize, scale: (f64, f64)) -> bool {
    if image_width == 0 {
        return true;
    }
    let w = image_width as f64;
    let band_s_lo = d_lo * 3600.0 / (QUAD_FRACTION_HI * w);
    let band_s_hi = d_hi * 3600.0 / (QUAD_FRACTION_LO * w);
    band_s_lo <= scale.1 && band_s_hi >= scale.0
}

/// The tiled index files in `dir` whose quad-diameter band fits `image_width` +
/// `scale_range` -- i.e. every tile a field of this scale could match, across the
/// whole sky. A blind solve iterates these (each loaded exactly once) until one
/// yields a confident pose. Non-tiled files are skipped; the list is sorted for
/// deterministic scan order.
///
/// # Errors
/// Returns an error if the directory cannot be read.
pub fn matching_tile_paths<P: AsRef<Path>>(
    dir: P,
    image_width: usize,
    scale_range: Option<(f64, f64)>,
) -> PlatersResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir.as_ref())? {
        let path = entry?.path();
        let Some((_tile, (d_lo, d_hi))) = parse_index_meta(&path) else {
            continue;
        };
        if scale_range.is_none_or(|sr| band_matches_image(d_lo, d_hi, image_width, sr)) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Whether a `HEALPix` tile's cell could intersect a cone `(center, radius_deg)`.
/// Generous (over-includes) so a relevant tile is never wrongly dropped.
fn tile_intersects_cone(tile: TileRef, center: SkyCoord, radius_deg: f64) -> bool {
    let (lon, lat) = cdshealpix::nested::center(tile.depth, tile.hash);
    let cell_center = SkyCoord::new_normalized(lon.to_degrees(), lat.to_degrees());
    let cell_circumradius = 58.6323 / f64::from(1_u32 << tile.depth) * 0.8;
    center.angular_distance(&cell_center) <= radius_deg + cell_circumradius
}

/// A loaded index ready for querying -- a thin view over an mmap'd `.qidx` file.
/// The catalog stars and quads are *not* copied into owned vectors; they are read
/// straight out of the mmap via the accessors ([`star`](Self::star),
/// [`quad`](Self::quad), [`catalog`](Self::catalog)) and the pre-sorted quad/star
/// trees are searched in place.
#[derive(Debug, Clone)]
pub struct LoadedIndex {
    /// Path to the index file
    pub path: PathBuf,

    /// Index configuration
    pub config: IndexMetadata,

    /// Scale range this index covers (arcsec/pixel)
    pub scale_range: (f64, f64),

    /// Quad diameter range (degrees)
    pub diameter_range: (f64, f64),

    /// `HEALPix` tile this file covers, if it is a tiled index (`None` = all-sky).
    pub tile: Option<TileRef>,

    /// mmap'd, pre-sorted quad + star backing (`.qidx`) -- the sole storage. Quad
    /// matching and the catalog spatial queries search it in place; the accessors
    /// reconstruct `Star`/`CatalogQuad` on demand without owning a copy.
    flat: Arc<FlatCodeIndex>,
}

impl LoadedIndex {
    /// Open an index file. The only on-disk format is `.qidx`
    /// ([`from_qidx`](Self::from_qidx)); the extension is required.
    ///
    /// # Errors
    /// Propagates [`from_qidx`](Self::from_qidx)'s error; errors on a non-`.qidx`
    /// extension.
    pub fn open<P: AsRef<Path>>(path: P) -> PlatersResult<Self> {
        let path = path.as_ref();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("qidx") {
            return Err(Error::ValueError(format!(
                "not a .qidx index: {}",
                path.display()
            )));
        }
        Self::from_qidx(path)
    }

    /// Load an index from an mmap'd, pre-sorted `.qidx` file. Opening is just the
    /// mmap plus a header check -- no tree construction. The `catalog` and `quads`
    /// vectors are materialized (an `O(n)` copy) so the rest of the solver sees an
    /// ordinary [`LoadedIndex`]; quad matching and the catalog spatial queries run
    /// over the mapped, pre-built trees.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened/mapped or is not a valid
    /// `.qidx`.
    pub fn from_qidx<P: AsRef<Path>>(path: P) -> PlatersResult<Self> {
        let path = path.as_ref();
        let flat = Arc::new(FlatCodeIndex::open(path)?);
        let config = IndexMetadata {
            healpix_depth: flat.healpix_depth,
            stars_per_cell: flat.stars_per_cell,
            min_diameter_deg: flat.diameter_range.0,
            max_diameter_deg: flat.diameter_range.1,
        };
        let tile = parse_index_meta(path).and_then(|(t, _)| t);
        Ok(Self {
            path: path.to_path_buf(),
            config,
            scale_range: flat.scale_range,
            diameter_range: flat.diameter_range,
            tile,
            flat,
        })
    }

    /// The mmap'd, pre-sorted `.qidx` backing -- quad matching and the catalog
    /// spatial queries (`nearest`/`stars_near`) run over its mapped, pre-built
    /// trees.
    #[must_use]
    pub(crate) fn flat(&self) -> &Arc<FlatCodeIndex> {
        &self.flat
    }

    /// Number of catalog stars in this index.
    #[must_use]
    pub fn num_stars(&self) -> usize {
        self.flat.num_stars()
    }

    /// Number of quads in this index.
    #[must_use]
    pub fn num_quads(&self) -> usize {
        self.flat.num_quads()
    }

    /// Force this index's mapping resident (see `FlatCodeIndex::prefault`).
    /// Returns the mapping size in bytes.
    #[must_use]
    pub fn prefault(&self) -> usize {
        self.flat.prefault()
    }

    /// Catalog star `i` (original star order), read from the mmap.
    #[must_use]
    pub fn star(&self, i: usize) -> Star {
        self.flat.star(i)
    }

    /// Quad `i` (star indices + hash code), read from the mmap.
    #[must_use]
    pub fn quad(&self, i: usize) -> CatalogQuad {
        CatalogQuad {
            star_indices: self.flat.quad_star_indices(i),
            hash_code: self.flat.quad_hash(i),
            quad_index: i,
        }
    }

    /// Materialize the full catalog as an owned vector. This is an `O(n)` copy out
    /// of the mmap -- use it only for bulk/offline consumers (e.g. merging); the
    /// solve path reads stars individually via [`star`](Self::star).
    #[must_use]
    pub fn catalog(&self) -> Vec<Star> {
        self.flat.catalog()
    }

    /// Whether this index's sky coverage could intersect a cone `(center,
    /// radius_deg)`. Monolithic (all-sky) indices always match; tiled indices
    /// match when their `HEALPix` cell is within the cone (generous, so a tile is
    /// never wrongly excluded).
    #[must_use]
    pub fn intersects_cone(&self, center: SkyCoord, radius_deg: f64) -> bool {
        self.tile
            .is_none_or(|tile| tile_intersects_cone(tile, center, radius_deg))
    }

    /// Check if this index is suitable for a given scale range.
    /// Whether this index's quad-diameter band yields usefully-sized quads for an
    /// image `image_width` px wide at pixel scales in `scale` (arcsec/px). The
    /// match is computed from the band's *angular* diameter and the *actual*
    /// image width -- see [`band_matches_image`].
    #[must_use]
    pub fn matches_field(&self, image_width: usize, scale: (f64, f64)) -> bool {
        band_matches_image(
            self.diameter_range.0,
            self.diameter_range.1,
            image_width,
            scale,
        )
    }

    /// Find candidate quads matching a hash code.
    ///
    /// Returns quads within the tolerance distance of the target hash code.
    #[must_use]
    pub fn find_matching_quads(&self, hash_code: &HashCode, tolerance: f64) -> Vec<QuadMatch> {
        // Search the mmap'd pre-sorted quad tree in place, then read each matched
        // quad's stars straight out of the mmap (no owned catalog/quad vectors).
        let num_stars = self.num_stars();
        self.flat
            .search(&hash_code.components, tolerance)
            .into_iter()
            .filter_map(|(distance, idx)| {
                let quad = self.quad(idx);
                // A corrupt index file could carry star ids past the catalog;
                // skip such quads rather than indexing out of bounds.
                if quad.star_indices.iter().any(|&si| si >= num_stars) {
                    return None;
                }
                Some(QuadMatch {
                    catalog_stars: [
                        self.flat.star(quad.star_indices[0]),
                        self.flat.star(quad.star_indices[1]),
                        self.flat.star(quad.star_indices[2]),
                        self.flat.star(quad.star_indices[3]),
                    ],
                    catalog_quad: quad,
                    hash_distance: distance,
                })
            })
            .collect()
    }

    /// Get statistics about this index.
    #[must_use]
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_stars: self.num_stars(),
            num_quads: self.num_quads(),
            scale_range: self.scale_range,
            diameter_range: self.diameter_range,
            healpix_depth: self.config.healpix_depth,
        }
    }
}

/// Metadata about an index.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// `HEALPix` depth of the build grid.
    pub healpix_depth: u8,
    /// Brightest-per-cell cap used at build time.
    pub stars_per_cell: usize,
    /// Minimum quad diameter in degrees.
    pub min_diameter_deg: f64,
    /// Maximum quad diameter in degrees.
    pub max_diameter_deg: f64,
}

/// A quad from a catalog.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CatalogQuad {
    /// Indices of the four stars in the catalog
    pub star_indices: [usize; 4],

    /// Hash code for this quad
    pub hash_code: HashCode,

    /// Index of this quad in the quad list
    pub quad_index: usize,
}

/// A match between an image quad and a catalog quad.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct QuadMatch {
    /// The catalog quad that matched
    pub catalog_quad: CatalogQuad,

    /// The four catalog stars forming this quad
    pub catalog_stars: [Star; 4],

    /// Distance between hash codes
    pub hash_distance: f64,
}

/// Statistics about an index.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of catalog stars.
    pub num_stars: usize,
    /// Number of quads.
    pub num_quads: usize,
    /// (min, max) pixel scale covered (arcsec/pixel).
    pub scale_range: (f64, f64),
    /// (min, max) quad diameter covered (degrees).
    pub diameter_range: (f64, f64),
    /// `HEALPix` depth of the build grid.
    pub healpix_depth: u8,
}

/// Multi-scale index set for efficient querying.
#[derive(Debug, Clone)]
pub struct IndexSet {
    /// All loaded indices
    indices: Vec<LoadedIndex>,

    /// Whether the indices are sorted by scale
    sorted: bool,
}

impl IndexSet {
    /// Create a new empty index set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            sorted: false,
        }
    }

    /// Add an index to the set.
    pub fn add(&mut self, index: LoadedIndex) {
        self.indices.push(index);
        self.sorted = false;
    }

    /// Load every `.qidx` index file in a directory into one in-memory set
    /// (eager, no cone/scale pruning). For solving directly against a pre-loaded
    /// set; for lazy, field-pruned loading of a large tiled directory use
    /// [`load_tiles_for`](Self::load_tiles_for).
    ///
    /// # Errors
    /// Returns an error if the directory cannot be read or a `.qidx` file fails to
    /// open.
    pub fn load_from_directory<P: AsRef<Path>>(dir: P) -> PlatersResult<Self> {
        let dir = dir.as_ref();
        let mut set = Self::new();

        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) == Some("qidx") {
                set.add(LoadedIndex::from_qidx(&path)?);
            }
        }

        set.sort_by_scale();
        Ok(set)
    }

    /// Lazily load only the index files in `dir` that a solve needs: the field fit
    /// (the file's angular quad-diameter band vs `image_width` + the `scale_range`
    /// hint -- see [`band_matches_image`]) and the sky cone are both checked against
    /// each file's **name**, so non-matching tiles are never deserialized. This is
    /// the all-sky RAM win -- a tiled directory can be enormous, but only the few
    /// tiles under the field are read. `tile_...` files are cone-pruned; `allsky_...`
    /// per-scale files always pass the cone (they cover the whole sky). Files with
    /// neither recognized name are skipped.
    ///
    /// # Errors
    /// Returns an error if the directory cannot be read or a selected tile fails
    /// to deserialize.
    pub fn load_tiles_for<P: AsRef<Path>>(
        dir: P,
        image_width: usize,
        scale_range: Option<(f64, f64)>,
        cone: Option<(SkyCoord, f64)>,
    ) -> PlatersResult<Self> {
        let mut set = Self::new();
        for entry in std::fs::read_dir(dir.as_ref())? {
            let path = entry?.path();
            let Some((tile, (d_lo, d_hi))) = parse_index_meta(&path) else {
                continue;
            };
            // Band produces useful quads for this image + scale?
            if scale_range.is_some_and(|sr| !band_matches_image(d_lo, d_hi, image_width, sr)) {
                continue;
            }
            // Cone prune: a HEALPix tile must intersect the cone; an all-sky index
            // (`tile = None`) covers everything, so it always passes.
            if let Some(tile) = tile {
                if cone.is_some_and(|(c, r)| !tile_intersects_cone(tile, c, r)) {
                    continue;
                }
            }
            set.add(LoadedIndex::open(&path)?);
        }
        set.sort_by_scale();
        Ok(set)
    }

    /// Sort indices by minimum scale (for efficient filtering).
    pub fn sort_by_scale(&mut self) {
        self.indices.sort_by(|a, b| {
            a.scale_range
                .0
                .partial_cmp(&b.scale_range.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.sorted = true;
    }

    /// Select indices for a solve: by **field fit** (the band's angular quad
    /// diameter vs the image width + pixel-scale hint -- see
    /// [`band_matches_image`]) and by sky **cone**. The cone prunes tiled indices
    /// to the tiles whose cell intersects `(center, radius_deg)`; monolithic
    /// indices ignore it. `image_width` of 0 or `None` filters disable the
    /// respective check; both `None` yields every index (blind, all-sky).
    #[must_use]
    pub fn select_for(
        &self,
        image_width: usize,
        scale_range: Option<(f64, f64)>,
        cone: Option<(SkyCoord, f64)>,
    ) -> Vec<&LoadedIndex> {
        self.indices
            .iter()
            .filter(|idx| {
                scale_range.is_none_or(|sr| idx.matches_field(image_width, sr))
                    && cone.is_none_or(|(c, r)| idx.intersects_cone(c, r))
            })
            .collect()
    }

    /// Get all indices (for blind solving without scale hints).
    #[must_use]
    pub fn all_indices(&self) -> &[LoadedIndex] {
        &self.indices
    }

    /// Get the number of indices in this set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Pre-fault every loaded index so the whole working set is resident before
    /// serving (uniform query latency from the first request). Returns the total
    /// bytes touched.
    pub fn prefault(&self) -> usize {
        self.indices.iter().map(LoadedIndex::prefault).sum()
    }

    /// Check if the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get combined statistics for all indices.
    pub fn total_stats(&self) -> IndexStats {
        let total_stars: usize = self.indices.iter().map(LoadedIndex::num_stars).sum();
        let total_quads: usize = self.indices.iter().map(LoadedIndex::num_quads).sum();

        let min_scale = self
            .indices
            .iter()
            .map(|i| i.scale_range.0)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_scale = self
            .indices
            .iter()
            .map(|i| i.scale_range.1)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        IndexStats {
            num_stars: total_stars,
            num_quads: total_quads,
            scale_range: (min_scale, max_scale),
            diameter_range: (0.0, 0.0), // Not meaningful for combined stats
            healpix_depth: self.indices.first().map_or(0, |i| i.config.healpix_depth),
        }
    }
}

impl Default for IndexSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_set_creation() {
        let set = IndexSet::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn parse_index_meta_tile_vs_allsky() {
        // A HEALPix tile: cone-prunable, band parsed (degrees = encoded / 600).
        let tile = parse_index_meta(Path::new("tile_d4_h523_q120-170_n04.qidx"));
        let (t, band) = tile.expect("tile parse");
        let t = t.expect("tile present");
        assert_eq!((t.depth, t.hash), (4, 523));
        assert!((band.0 - 0.2).abs() < 1e-9 && (band.1 - 170.0 / 600.0).abs() < 1e-9);

        // An all-sky per-scale index: no tile (always passes a cone), same band.
        let allsky = parse_index_meta(Path::new("allsky_q120-170_n04.qidx"));
        let (t, band) = allsky.expect("allsky parse");
        assert!(t.is_none(), "all-sky index must have no tile");
        assert!((band.0 - 0.2).abs() < 1e-9);

        // Unrecognized names are skipped.
        assert!(parse_index_meta(Path::new("catalog.parquet")).is_none());
    }

    #[test]
    fn test_scale_range_matching() {
        // This would need a real index to test properly
        // For now, just test the logic
        let query_range = (1.0, 2.0);
        let index_range = (1.5, 2.5);

        // Ranges overlap
        let overlaps = query_range.0 <= index_range.1 && query_range.1 >= index_range.0;
        assert!(overlaps);

        // Non-overlapping ranges
        let query_range = (1.0, 1.5);
        let index_range = (2.0, 3.0);
        let overlaps = query_range.0 <= index_range.1 && query_range.1 >= index_range.0;
        assert!(!overlaps);
    }
}
