//! Spatial index over a star catalog for fast "stars near this point" queries.
//!
//! Both solver stages need the same primitive: given a sky position and a
//! radius, return the catalog stars inside that cone (and the single nearest
//! star, for verification). This module provides that primitive once, backed
//! by a KD-tree over **3D unit vectors** so the distance metric is the true
//! great-circle distance -- correct across the RA = 0/360 wrap and at the poles,
//! unlike a naive 2D (RA, Dec) tree.

use crate::flatindex::FlatCodeIndex;
use crate::geometry::{chord_sq_for_angle, chord_sq_to_arcsec};
use crate::index::LoadedIndex;
use crate::types::{SkyCoord, Star};
use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use std::sync::Arc;

/// KD-tree node bucket size. Small buckets favor query speed over build speed,
/// which suits our "build once, query many" usage.
const BUCKET: usize = 32;

/// 3D unit-vector KD-tree: key = unit vector, value = index into `stars`.
type Tree = KdTree<f64, u64, 3, BUCKET, u32>;

/// Use the `Flat` forest only when the number of sub-indices
/// is at most this. A forest query fans out over *every* sub-tree (`nearest` takes
/// the global min), so cost grows with the index count; past this point, building
/// one merged in-memory tree and amortizing it over many queries is cheaper. Few
/// large all-sky indices (merged blind, or a hinted solve touching a handful of
/// tiles) stay under it and skip the build; a blind sweep over a 128-tile batch
/// exceeds it and uses one built tree.
const FLAT_FOREST_MAX_INDICES: usize = 16;

/// A star catalog with a spatial index for cone and nearest-neighbor queries.
///
/// Two backings with the same query surface:
/// - `Kiddo`: an in-memory KD-tree built from an owned star list (used for the
///   locally-bounded catalog around a hint, and for a many-tile batch where a
///   single built tree beats fanning out over many small ones).
/// - `Flat`: a *forest* of the pre-built, mmap'd star trees of the loaded indices
///   (`FlatCodeIndex`). No spatial tree is built at solve time -- the key win for
///   blind solving against merged all-sky indices, where building a kiddo tree
///   over millions of stars per solve dominated. A query fans out over the forest:
///   `nearest` takes the global minimum, `stars_near` the union.
#[derive(Debug)]
pub struct CatalogIndex(Inner);

/// Private backing so the storage strategy can change without an API break.
#[derive(Debug)]
enum Inner {
    /// In-memory KD-tree over an owned star list.
    Kiddo {
        /// The owned stars, indexed by the tree's item ids.
        stars: Vec<Star>,
        /// KD-tree over the stars' unit vectors.
        tree: Tree,
    },
    /// Forest of mmap'd, pre-built star trees (one per `.qidx` index).
    Flat {
        /// The mmap'd indices whose embedded star trees are queried in place.
        indices: Vec<Arc<FlatCodeIndex>>,
    },
}

impl CatalogIndex {
    /// Build an index over the given stars (in-memory KD-tree).
    #[must_use]
    pub fn new(stars: Vec<Star>) -> Self {
        let mut tree = Tree::with_capacity(stars.len());
        for (i, star) in stars.iter().enumerate() {
            tree.add(&star.position.to_unit_vector(), i as u64);
        }
        Self(Inner::Kiddo { stars, tree })
    }

    /// Build an index from several loaded sub-indices.
    ///
    /// With at most `FLAT_FOREST_MAX_INDICES` indices, this borrows their
    /// pre-built star trees into a flat forest -- no tree
    /// construction at all, which is what makes blind solving against merged
    /// all-sky indices fast. Past that (a many-tile batch, where fan-out would cost
    /// more than a build) it concatenates the materialized catalogs and builds one
    /// in-memory tree (duplicates across scales are harmless coincident points).
    #[must_use]
    pub fn from_indices(indices: &[&LoadedIndex]) -> Self {
        if !indices.is_empty() && indices.len() <= FLAT_FOREST_MAX_INDICES {
            let flats = indices
                .iter()
                .map(|i| Arc::clone(i.flat()))
                .collect::<Vec<_>>();
            return Self(Inner::Flat { indices: flats });
        }
        let total: usize = indices.iter().map(|i| i.num_stars()).sum();
        let mut stars = Vec::with_capacity(total);
        for index in indices {
            stars.extend(index.catalog());
        }
        Self::new(stars)
    }

    /// All catalog stars within `radius_deg` of `center` (great-circle).
    #[must_use]
    pub fn stars_near(&self, center: SkyCoord, radius_deg: f64) -> Vec<Star> {
        match &self.0 {
            Inner::Kiddo { stars, tree } => {
                if stars.is_empty() {
                    return Vec::new();
                }
                let q = center.to_unit_vector();
                let chord_sq = chord_sq_for_angle(radius_deg.to_radians());
                tree.within_unsorted::<SquaredEuclidean>(&q, chord_sq)
                    .into_iter()
                    .map(|n| stars[n.item as usize])
                    .collect()
            }
            Inner::Flat { indices } => indices
                .iter()
                .flat_map(|idx| idx.stars_near(center, radius_deg))
                .collect(),
        }
    }

    /// The single nearest catalog star to `sky`, with its angular distance in
    /// arcseconds. Returns `None` only if the catalog is empty.
    #[must_use]
    pub fn nearest(&self, sky: SkyCoord) -> Option<(Star, f64)> {
        match &self.0 {
            Inner::Kiddo { stars, tree } => {
                if stars.is_empty() {
                    return None;
                }
                let q = sky.to_unit_vector();
                let nn = tree.nearest_one::<SquaredEuclidean>(&q);
                Some((stars[nn.item as usize], chord_sq_to_arcsec(nn.distance)))
            }
            // Global nearest across the forest = the closest of each tree's nearest.
            Inner::Flat { indices } => indices
                .iter()
                .filter_map(|idx| idx.nearest(sky))
                .min_by(|a, b| a.1.total_cmp(&b.1)),
        }
    }

    /// Number of stars in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.0 {
            Inner::Kiddo { stars, .. } => stars.len(),
            Inner::Flat { indices } => indices.iter().map(|i| i.num_stars()).sum(),
        }
    }

    /// Whether the index contains no stars.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn star(ra: f64, dec: f64) -> Star {
        Star::with_id(ra, dec, 10.0, 0)
    }

    #[test]
    fn nearest_finds_closest() {
        let cat = CatalogIndex::new(vec![star(10.0, 20.0), star(200.0, -30.0), star(11.0, 20.0)]);
        let (found, arcsec) = cat.nearest(SkyCoord::new(10.1, 20.0)).unwrap();
        assert!(
            (found.position.ra - 10.0).abs() < 1e-9,
            "got RA {}",
            found.position.ra
        );
        // 0.1 deg at this dec is ~360 arcsec scaled by cos(dec); just sanity-bound it.
        assert!(arcsec > 0.0 && arcsec < 600.0, "arcsec = {arcsec}");
    }

    #[test]
    fn stars_near_respects_radius() {
        let cat = CatalogIndex::new(vec![
            star(180.0, 45.0),
            star(180.2, 45.0),
            star(181.0, 45.0),
            star(0.0, -80.0),
        ]);
        // 0.5 deg cone around (180,45): the chord captures the first two, not (181,45).
        let near = cat.stars_near(SkyCoord::new(180.0, 45.0), 0.5);
        assert!(
            near.len() >= 2,
            "expected at least the two closest, got {}",
            near.len()
        );
        assert!(
            near.iter().all(|s| s.position.dec > 0.0),
            "polar star leaked in"
        );
    }

    #[test]
    fn handles_ra_wrap() {
        // Two stars straddling RA = 0/360 are ~0.2 deg apart on the sphere; a
        // naive 2D (RA, Dec) tree would think they are ~359.8 deg apart.
        let cat = CatalogIndex::new(vec![star(359.9, 0.0), star(0.1, 0.0)]);
        let near = cat.stars_near(SkyCoord::new(0.0, 0.0), 0.5);
        assert_eq!(near.len(), 2, "RA-wrap neighbors not both found");
    }

    #[test]
    fn empty_index() {
        let cat = CatalogIndex::new(Vec::new());
        assert!(cat.is_empty());
        assert!(cat.nearest(SkyCoord::new(0.0, 0.0)).is_none());
        assert!(cat.stars_near(SkyCoord::new(0.0, 0.0), 1.0).is_empty());
    }
}
