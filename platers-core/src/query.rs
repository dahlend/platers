//! Query processing and hypothesis generation for plate solving.
//!
//! This module implements the core plate solving query engine that:
//! - Generates quads from detected stars
//! - Matches quads against index using hash codes
//! - Generates WCS hypotheses from matches
//! - Optimizes for known FOV/pixel scale (10-100x speedup)

use crate::{
    errors::PlatersResult,
    geometry::{compute_hash_code_pixels, HashCode, Quad},
    types::{DetectedStar, PixelCoord},
};

/// Configuration for query processing.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Scale hint (HIGHLY RECOMMENDED for 10-100x speedup)
    pub scale_hint: Option<ScaleRange>,

    /// Position hint (optional, for further optimization)
    pub position_hint: Option<PositionHint>,

    /// Maximum number of stars to use for quad generation
    pub max_stars_for_quads: usize,

    /// Ceiling on image quads to generate (the adaptive budget scales with the
    /// star count up to this cap)
    pub max_quads_to_try: usize,

    /// Maximum hypotheses to verify before giving up (a per-solve ceiling,
    /// shared across the breadth-first and full-ball search passes)
    pub max_hypotheses: usize,

    /// Hash code search radius (Euclidean distance in 4D space)
    pub hash_code_tolerance: f64,

    /// Observation epoch as a Julian year (e.g. `2021.4`). When set, catalog
    /// stars that carry a proper motion are propagated from the catalog
    /// reference epoch to this one before refinement matching. `None` (the
    /// default) uses catalog positions as stored.
    pub observation_epoch: Option<f64>,
}

impl Default for QueryConfig {
    /// Production-sized budgets (matching the CLI and server): a confident
    /// solution exits early, so easy fields never pay the full ceiling; a hard
    /// field gets a real search rather than giving up after a token effort.
    fn default() -> Self {
        Self {
            scale_hint: None,
            position_hint: None,
            max_stars_for_quads: 50,
            max_quads_to_try: 300_000,
            max_hypotheses: 200_000,
            hash_code_tolerance: 0.01,
            observation_epoch: None,
        }
    }
}

/// Scale range for FOV optimization.
///
/// Providing a scale hint enables 10-100x speedup by:
/// - Loading only 1-3 relevant sub-indices instead of all
/// - Filtering hypotheses early based on implied scale
///
/// Most users know their pixel scale to +/-1-5%, making this the common case.
#[derive(Debug, Clone)]
pub struct ScaleRange {
    /// Minimum pixel scale (arcsec/pixel)
    pub min_arcsec_per_pixel: f64,
    /// Maximum pixel scale (arcsec/pixel)
    pub max_arcsec_per_pixel: f64,
    /// Tolerance as fraction (e.g., 0.05 = 5%)
    pub tolerance: f64,
}

impl ScaleRange {
    /// Create a scale range from a nominal value and tolerance.
    ///
    /// # Example
    /// ```
    /// use platers_core::query::ScaleRange;
    ///
    /// // 0.39 arcsec/pixel +/- 5%
    /// let range = ScaleRange::from_nominal(0.39, 0.05);
    /// assert!(range.contains(0.39));
    /// assert!(range.contains(0.37)); // 5% below
    /// assert!(range.contains(0.41)); // 5% above
    /// ```
    #[must_use]
    pub fn from_nominal(nominal_arcsec_per_pixel: f64, tolerance: f64) -> Self {
        let min = nominal_arcsec_per_pixel * (1.0 - tolerance);
        let max = nominal_arcsec_per_pixel * (1.0 + tolerance);
        Self {
            min_arcsec_per_pixel: min,
            max_arcsec_per_pixel: max,
            tolerance,
        }
    }

    /// Create a scale range from image FOV and dimensions.
    ///
    /// # Example
    /// ```
    /// use platers_core::query::ScaleRange;
    ///
    /// // 13.4' x 9.8' field, 2048 x 1489 pixels
    /// let range = ScaleRange::from_fov(
    ///     0.224, // width in degrees
    ///     0.164, // height in degrees
    ///     2048,  // width in pixels
    ///     1489,  // height in pixels
    ///     0.05   // 5% tolerance
    /// );
    /// ```
    #[must_use]
    pub fn from_fov(
        fov_width_deg: f64,
        fov_height_deg: f64,
        image_width_px: usize,
        image_height_px: usize,
        tolerance: f64,
    ) -> Self {
        // Guard zero dimensions so the scale cannot go infinite.
        let scale_x = (fov_width_deg * 3600.0) / image_width_px.max(1) as f64;
        let scale_y = (fov_height_deg * 3600.0) / image_height_px.max(1) as f64;
        let avg_scale = f64::midpoint(scale_x, scale_y);

        Self::from_nominal(avg_scale, tolerance)
    }

    /// Check if a pixel scale is within this range.
    #[must_use]
    pub fn contains(&self, scale_arcsec_per_pixel: f64) -> bool {
        let min_with_tolerance = self.min_arcsec_per_pixel * (1.0 - self.tolerance);
        let max_with_tolerance = self.max_arcsec_per_pixel * (1.0 + self.tolerance);
        scale_arcsec_per_pixel >= min_with_tolerance && scale_arcsec_per_pixel <= max_with_tolerance
    }

    /// Check if this range overlaps with another range.
    #[must_use]
    pub fn overlaps(&self, other: &(f64, f64)) -> bool {
        let (other_min, other_max) = other;
        let self_min = self.min_arcsec_per_pixel * (1.0 - self.tolerance);
        let self_max = self.max_arcsec_per_pixel * (1.0 + self.tolerance);

        // Ranges overlap if they are not disjoint
        !(self_max < *other_min || self_min > *other_max)
    }
}

/// Position hint for sky location.
#[derive(Debug, Clone)]
pub struct PositionHint {
    /// Right ascension (degrees)
    pub ra: f64,
    /// Declination (degrees)
    pub dec: f64,
    /// Search radius (degrees)
    pub radius: f64,
}

impl PositionHint {
    /// Create a position hint: a cone of `radius` degrees around (`ra`, `dec`).
    #[must_use]
    pub const fn new(ra: f64, dec: f64, radius: f64) -> Self {
        Self { ra, dec, radius }
    }
}

/// A quad generated from detected stars in an image.
#[derive(Debug, Clone)]
pub struct ImageQuad {
    /// The four stars forming this quad (in pixel coordinates)
    pub stars: [DetectedStar; 4],
    /// Indices of stars in the original detection list
    pub star_indices: [usize; 4],
    /// Hash code for this quad
    pub hash_code: HashCode,
    /// Quad geometry in pixel space
    pub geometry: Quad<PixelCoord>,
}

impl ImageQuad {
    /// Create a new image quad from detected stars.
    ///
    /// # Errors
    /// Returns an error if the quad is degenerate or invalid.
    pub fn new(stars: [DetectedStar; 4], star_indices: [usize; 4]) -> PlatersResult<Self> {
        let coords: Vec<PixelCoord> = stars.iter().map(|s| PixelCoord::new(s.x, s.y)).collect();

        // Build geometry quad
        let geometry = Quad::new([coords[0], coords[1], coords[2], coords[3]], star_indices);

        let hash_code = compute_hash_code_pixels(&[
            geometry.stars[0],
            geometry.stars[1],
            geometry.stars[2],
            geometry.stars[3],
        ])?;

        Ok(Self {
            stars,
            star_indices,
            hash_code,
            geometry,
        })
    }

    /// Get the diameter of this quad in pixels.
    #[must_use]
    pub fn diameter_pixels(&self) -> f64 {
        // Calculate maximum pairwise distance
        let mut max_dist = 0.0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dx = self.geometry.stars[i].x - self.geometry.stars[j].x;
                let dy = self.geometry.stars[i].y - self.geometry.stars[j].y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > max_dist {
                    max_dist = dist;
                }
            }
        }
        max_dist
    }
}

/// Uniformize a detected-star list to match how the index selected stars.
///
/// The index builds quads from the **brightest `stars_per_cell` stars in each
/// `HEALPix` cell**, so an image must reproduce that *per-cell* selection for its
/// quads to coincide with index quads. Taking the globally-brightest N across
/// the whole image (the naive approach) picks a different star set and the
/// hashes rarely match -- the root cause of "no hypotheses generated" on sparse
/// indices.
///
/// This bins the detected stars onto an image-space grid whose cell size equals
/// the index's `HEALPix` cell projected to pixels
/// (`cell_pixels = healpix_cell_arcsec / pixel_scale`, per astrometry.net's
/// `verify_get_uniformize_scale`), and keeps the brightest `stars_per_cell` in
/// each bin. Returns the union, brightest-first.
///
/// `healpix_cell_arcsec` is the index's cell side length; `pixel_scale_arcsec`
/// is the (hinted/expected) image scale; `stars_per_cell` matches the index.
#[must_use]
#[allow(
    clippy::cast_sign_loss,
    reason = "negative pixel coordinates saturate to bin 0, which is the intended bin"
)]
pub fn uniformize_field(
    stars: &[DetectedStar],
    image_width: usize,
    image_height: usize,
    healpix_cell_arcsec: f64,
    pixel_scale_arcsec: f64,
    stars_per_cell: usize,
) -> Vec<DetectedStar> {
    use std::collections::HashMap;

    // Cell side in pixels; grid counts across the image (>= 1).
    let cell_pixels = (healpix_cell_arcsec / pixel_scale_arcsec).max(1.0);
    let nw = ((image_width as f64 / cell_pixels).round() as usize).max(1);
    let nh = ((image_height as f64 / cell_pixels).round() as usize).max(1);

    // With a 1x1 grid, uniformizing is a no-op; fall back to all stars.
    if nw == 1 && nh == 1 {
        return stars.to_vec();
    }

    let mut bins: HashMap<(usize, usize), Vec<DetectedStar>> = HashMap::new();
    for s in stars {
        let bx = ((s.x / image_width as f64) * nw as f64) as usize;
        let by = ((s.y / image_height as f64) * nh as f64) as usize;
        let key = (bx.min(nw - 1), by.min(nh - 1));
        bins.entry(key).or_default().push(*s);
    }

    let mut out = Vec::new();
    for stars_in_bin in bins.values_mut() {
        // Brightest first (flux desc), keep top `stars_per_cell`.
        DetectedStar::sort_brightest_first(stars_in_bin);
        stars_in_bin.truncate(stars_per_cell);
        out.extend_from_slice(stars_in_bin);
    }
    DetectedStar::sort_brightest_first(&mut out);
    out
}

/// Enumerate distance-bounded quads -- the **single rule used by both the index
/// builder and the field generator**, so a field quad coincides with a catalog
/// quad over the same four stars.
///
/// Callers pass stars **brightest-first**. For each star `a`, this gathers up to
/// `max_neighbors` other stars within `radius2` of it (squared metric `dist2`, in
/// the caller's space -- squared pixels for a field, squared chord on the unit
/// sphere for the catalog) and emits every quad `{a} + (3-subset of those
/// neighbours)`. Results are deduplicated and sorted (bright-star quads first).
///
/// Because the hash code of four stars is canonical, the *only* thing needed for a
/// field quad and a catalog quad to match is that **both sides build a quad from
/// the same four stars** -- which this guarantees when both run this enumeration
/// over the same (uniformized) local star set.
#[must_use]
pub fn enumerate_distance_bounded_quads(
    n: usize,
    dist2: impl Fn(usize, usize) -> f64,
    radius2: f64,
    max_neighbors: usize,
) -> Vec<[usize; 4]> {
    use std::collections::HashSet;
    let mut set: HashSet<[usize; 4]> = HashSet::new();
    for a in 0..n {
        // Near stars (brightest first, since the caller sorted), capped.
        let mut nbrs: Vec<usize> = Vec::new();
        for b in 0..n {
            if b == a {
                continue;
            }
            if dist2(a, b) <= radius2 {
                nbrs.push(b);
                if nbrs.len() >= max_neighbors {
                    break;
                }
            }
        }
        // Quads = anchor + every 3-subset of its near stars.
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                for k in (j + 1)..nbrs.len() {
                    let mut q = [a, nbrs[i], nbrs[j], nbrs[k]];
                    q.sort_unstable();
                    let _ = set.insert(q);
                }
            }
        }
    }
    let mut combos: Vec<[usize; 4]> = set.into_iter().collect();
    // Order by the DEEPEST member's brightness rank (members are ascending, so
    // that's `q[3]`): all-bright quads first, then progressively deeper stars.
    // Lexicographic order would instead drain the generation budget into every
    // combination containing star 0 before star 1 ever anchors a quad -- with a
    // large neighbour cap that starves all but the brightest few anchors.
    combos.sort_unstable_by_key(|q| (q[3], q[2], q[1], q[0]));
    combos
}

/// Generator for image quads from detected stars.
///
/// Two modes:
/// - **Local** (preferred, when a quad scale is known): only sets of four stars
///   that are *mutually close* -- all within `radius_px` of the brightest, the
///   pixel image of a catalog quad's diameter -- are emitted. This mirrors how the
///   index builds quads (within a `HEALPix` cell), so image and catalog quads
///   correspond; it also bounds the count (no `C(n,4)` blow-up) and, crucially,
///   solves *wide* fields, where global enumeration never reaches the per-cell
///   quads within budget.
/// - **Global** (fallback, no scale info): every `C(n,4)` combination in
///   lexicographic order. Fine for narrow fields (~one cell); the broken case for
///   wide ones.
#[derive(Debug)]
pub struct ImageQuadGenerator {
    /// Detected stars sorted by brightness (descending).
    stars: Vec<DetectedStar>,
    /// Number of stars considered (`min(stars.len(), max_stars)`).
    n: usize,
    /// Global mode: the next lexicographic 4-index combination, or `None` when
    /// exhausted. `None` here in local mode.
    next_combo: Option<[usize; 4]>,
    /// Local mode: the precomputed, distance-bounded 4-index combinations and a
    /// cursor into them. Empty/unused in global mode.
    local_combos: Vec<[usize; 4]>,
    cursor: usize,
}

impl ImageQuadGenerator {
    /// Create a global-mode generator (every `C(n,4)`). Use when no quad scale is
    /// available; prefer [`new_local`](Self::new_local) otherwise.
    #[must_use]
    pub fn new(mut stars: Vec<DetectedStar>, max_stars: usize) -> Self {
        Self::sort_and_truncate(&mut stars, max_stars);
        let n = stars.len();
        let next_combo = (n >= 4).then_some([0, 1, 2, 3]);
        Self {
            stars,
            n,
            next_combo,
            local_combos: Vec::new(),
            cursor: 0,
        }
    }

    /// Create a local-mode generator: only quads whose four stars lie within
    /// `radius_px` of their brightest member (~ a catalog quad's diameter in
    /// pixels). `max_neighbors` caps how many near stars each anchor pairs with,
    /// mirroring the index's brightest-per-cell selection.
    #[must_use]
    pub fn new_local(
        mut stars: Vec<DetectedStar>,
        max_stars: usize,
        radius_px: f64,
        max_neighbors: usize,
    ) -> Self {
        Self::sort_and_truncate(&mut stars, max_stars);
        let combos = Self::distance_bounded_combos(&stars, radius_px, max_neighbors);
        Self {
            n: stars.len(),
            stars,
            next_combo: None,
            local_combos: combos,
            cursor: 0,
        }
    }

    fn sort_and_truncate(stars: &mut Vec<DetectedStar>, max_stars: usize) {
        DetectedStar::sort_brightest_first(stars);
        if stars.len() > max_stars {
            stars.truncate(max_stars);
        }
    }

    /// All four-star index sets in which every star is within `radius_px` of the
    /// set's brightest member, via the shared [`enumerate_distance_bounded_quads`].
    fn distance_bounded_combos(
        stars: &[DetectedStar],
        radius_px: f64,
        max_neighbors: usize,
    ) -> Vec<[usize; 4]> {
        enumerate_distance_bounded_quads(
            stars.len(),
            |a, b| {
                let dx = stars[a].x - stars[b].x;
                let dy = stars[a].y - stars[b].y;
                dx * dx + dy * dy
            },
            radius_px * radius_px,
            max_neighbors,
        )
    }

    /// Advance a strictly-increasing 4-index combination to the next one in
    /// lexicographic order over `0..n`, or `None` when exhausted.
    fn advance(mut c: [usize; 4], n: usize) -> Option<[usize; 4]> {
        let mut p = 3;
        loop {
            let max_for_p = n - 4 + p;
            if c[p] < max_for_p {
                c[p] += 1;
                for q in (p + 1)..4 {
                    c[q] = c[q - 1] + 1;
                }
                return Some(c);
            }
            if p == 0 {
                return None;
            }
            p -= 1;
        }
    }

    /// Generate the next quad, or `None` when exhausted. Combinations that
    /// form a degenerate quad are skipped.
    pub fn next_quad(&mut self) -> Option<ImageQuad> {
        // Local mode: walk the precomputed combos.
        while self.cursor < self.local_combos.len() {
            let indices = self.local_combos[self.cursor];
            self.cursor += 1;
            if let Some(q) = self.try_quad(indices) {
                return Some(q);
            }
        }
        // Global mode: walk the lexicographic cursor.
        while let Some(indices) = self.next_combo {
            self.next_combo = Self::advance(indices, self.n);
            if let Some(q) = self.try_quad(indices) {
                return Some(q);
            }
        }
        None
    }

    fn try_quad(&self, indices: [usize; 4]) -> Option<ImageQuad> {
        let stars = [
            self.stars[indices[0]],
            self.stars[indices[1]],
            self.stars[indices[2]],
            self.stars[indices[3]],
        ];
        ImageQuad::new(stars, indices).ok()
    }

    /// Get the number of detected stars.
    #[must_use]
    pub fn num_stars(&self) -> usize {
        self.stars.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniformization keeps the brightest `stars_per_cell` per grid bin and
    /// drops the rest, so a bin crowded with faint stars can't dominate.
    #[test]
    fn test_uniformize_field_keeps_brightest_per_bin() {
        // 1000x1000 image; cell 250 arcsec at 1 arcsec/px = 250 px -> 4x4 grid.
        // Put 5 stars in the top-left bin (0..250, 0..250) with varying flux,
        // and 1 star in a far bin.
        let mut stars = vec![
            DetectedStar::new(10.0, 10.0, 100.0),
            DetectedStar::new(20.0, 20.0, 500.0), // brightest in bin
            DetectedStar::new(30.0, 30.0, 50.0),
            DetectedStar::new(40.0, 40.0, 300.0),
            DetectedStar::new(50.0, 50.0, 10.0), // faintest, should be dropped
            DetectedStar::new(900.0, 900.0, 5.0), // lone star in another bin
        ];

        // Keep top 3 per bin.
        let out = uniformize_field(&stars, 1000, 1000, 250.0, 1.0, 3);

        // Top-left bin had 5 -> keep 3 brightest (500, 300, 100); the lone star
        // in its own bin survives. Total 4.
        assert_eq!(out.len(), 4, "got {out:?}");
        // The two faintest in the crowded bin (50, 10) must be gone.
        assert!(!out.iter().any(|s| s.flux == 50.0));
        assert!(!out.iter().any(|s| s.flux == 10.0));
        // The lone faint star (5.0) survives -- it's alone in its bin.
        assert!(out.iter().any(|s| s.flux == 5.0));
        // Output is brightest-first.
        assert!(out.windows(2).all(|w| w[0].flux >= w[1].flux));

        // A 1x1 grid (cell larger than the image) is a no-op.
        stars.truncate(6);
        let nop = uniformize_field(&stars, 1000, 1000, 1_000_000.0, 1.0, 3);
        assert_eq!(nop.len(), stars.len());
    }

    #[test]
    fn test_scale_range_from_nominal() {
        let range = ScaleRange::from_nominal(0.39, 0.05);
        assert!(range.contains(0.39));
        assert!(range.contains(0.37)); // 5% below
        assert!(range.contains(0.41)); // 5% above
        assert!(!range.contains(0.30)); // Too far below
        assert!(!range.contains(0.50)); // Too far above
    }

    #[test]
    fn test_scale_range_from_fov() {
        // SDSS-like: 13.4' x 9.8', 2048 x 1489 pixels
        let range = ScaleRange::from_fov(0.224, 0.164, 2048, 1489, 0.05);

        // Should be around 0.396 arcsec/pixel
        assert!(range.contains(0.396));
        assert!(range.min_arcsec_per_pixel < 0.40);
        assert!(range.max_arcsec_per_pixel > 0.39);
    }

    #[test]
    fn test_scale_range_overlaps() {
        let range = ScaleRange::from_nominal(0.39, 0.05);

        // Overlapping ranges
        assert!(range.overlaps(&(0.35, 0.40)));
        assert!(range.overlaps(&(0.38, 0.42)));
        assert!(range.overlaps(&(0.30, 0.50)));

        // Non-overlapping ranges
        assert!(!range.overlaps(&(0.10, 0.20)));
        assert!(!range.overlaps(&(0.80, 0.90)));
    }

    #[test]
    fn test_image_quad_generator() {
        let stars = vec![
            DetectedStar::new(100.0, 100.0, 1000.0),
            DetectedStar::new(200.0, 100.0, 900.0),
            DetectedStar::new(100.0, 200.0, 800.0),
            DetectedStar::new(200.0, 200.0, 700.0),
            DetectedStar::new(150.0, 150.0, 600.0),
        ];

        let mut generator = ImageQuadGenerator::new(stars, 5);
        assert_eq!(generator.num_stars(), 5);

        // Should generate at least one quad
        let quad_result = generator.next_quad();
        assert!(quad_result.is_some());

        if let Some(quad) = quad_result {
            // Quad should have 4 stars
            assert_eq!(quad.stars.len(), 4);
            assert_eq!(quad.star_indices.len(), 4);

            // Hash code should be valid (components in [0, 1])
            assert!(quad.hash_code.components[0] >= 0.0 && quad.hash_code.components[0] <= 1.0);
        }
    }

    /// The resumable cursor must enumerate exactly the C(n,4) unique
    /// 4-combinations, each strictly increasing, in lexicographic order, with
    /// no repeats.
    #[test]
    fn test_quad_generator_enumerates_all_combinations() {
        // 8 collinear-free stars on a jittered grid so quads are non-degenerate.
        let stars: Vec<DetectedStar> = (0..8)
            .map(|i| {
                let f = f64::from(i);
                let x = 100.0 + f64::from(i % 4) * 130.0 + f * 3.0;
                let y = 100.0 + f64::from(i / 4) * 170.0 + f * 5.0;
                DetectedStar::new(x, y, 1000.0 - f)
            })
            .collect();

        let mut g = ImageQuadGenerator::new(stars, 8);
        let mut combos = Vec::new();
        while let Some(q) = g.next_quad() {
            combos.push(q.star_indices);
        }

        // Every emitted combo is strictly increasing.
        for c in &combos {
            assert!(
                c[0] < c[1] && c[1] < c[2] && c[2] < c[3],
                "not increasing: {c:?}"
            );
        }
        // Strictly lexicographically increasing (ordered, no repeats).
        for w in combos.windows(2) {
            assert!(
                w[0] < w[1],
                "out of order or repeated: {:?} !< {:?}",
                w[0],
                w[1]
            );
        }
        // Exactly C(8,4) = 70 combinations (no quad is degenerate here).
        assert_eq!(combos.len(), 70, "expected C(8,4)=70, got {}", combos.len());
    }

    #[test]
    fn test_query_config_default() {
        let config = QueryConfig::default();
        assert!(config.scale_hint.is_none());
        assert_eq!(config.max_stars_for_quads, 50);
        assert_eq!(config.hash_code_tolerance, 0.01);
    }
}
