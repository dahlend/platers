//! Common test utilities and harness for integration tests.

use platers_build::{IndexConfig, TiledBuilder};
use platers_core::{
    load_catalog_parquet, DetectedStar, IndexSet, PixelCoord, PlateSolver, PlatersResult,
    QueryConfig, SkyCoord, Star, VerificationConfig, WcsHypothesis,
};
use std::path::PathBuf;
use std::sync::OnceLock;

/// The fixture catalog and its built tiled `.qidx` index directory, built **once
/// per test process** and cached. The index is built from the committed fixture
/// catalog (no pre-built index files needed) into a per-process temp dir.
fn fixture_index() -> &'static (Vec<Star>, PathBuf) {
    static CACHE: OnceLock<(Vec<Star>, PathBuf)> = OnceLock::new();
    CACHE.get_or_init(|| {
        let catalog_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fixture_catalog.parquet");
        let catalog = load_catalog_parquet(&catalog_path)
            .unwrap_or_else(|e| panic!("load fixture catalog {}: {e}", catalog_path.display()));

        let dir = std::env::temp_dir().join(format!("platers_fixture_idx_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = TiledBuilder::new(catalog.clone(), IndexConfig::default())
            .build_all_scales(&dir, 3.0, 0.4)
            .expect("build fixture tiled index");
        (catalog, dir)
    })
}

/// Build a *merged* all-sky `IndexSet` from the same fixture tiles the tiled
/// [`TestHarness`] uses, so a test can compare tiled-directory vs merged-set
/// layouts on identical data (image-quad generation must not depend on layout).
///
/// # Panics
/// Panics if the fixture tiles cannot be merged or loaded.
#[must_use]
pub fn merged_fixture_index_set() -> IndexSet {
    let (_catalog, tiled_dir) = fixture_index();
    let merged_dir =
        std::env::temp_dir().join(format!("platers_fixture_merged_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&merged_dir);
    let _paths =
        platers_build::merge_scale_indices(tiled_dir, &merged_dir).expect("merge fixture tiles");
    IndexSet::load_from_directory(&merged_dir).expect("load merged fixture index")
}

/// Test harness that manages indices and provides helper methods.
#[derive(Debug)]
pub struct TestHarness {
    /// Loaded index set
    pub index_set: IndexSet,
    /// Full catalog for verification
    pub catalog: Vec<Star>,
}

impl TestHarness {
    /// Create a new test harness. The fixture catalog and its tiled `.qidx` index
    /// are built once per test process (see `fixture_index`) and an in-memory
    /// index set is loaded from the cached index directory.
    ///
    /// # Errors
    /// Returns an error if the cached fixture index directory cannot be loaded.
    pub fn new() -> PlatersResult<Self> {
        let (catalog, index_dir) = fixture_index();
        let index_set = IndexSet::load_from_directory(index_dir)?;
        Ok(Self {
            index_set,
            catalog: catalog.clone(),
        })
    }

    /// Create a solver with default configuration.
    #[must_use]
    pub fn create_solver(&self) -> PlateSolver {
        let query_config = QueryConfig::default();
        let verification_config = VerificationConfig::default();

        PlateSolver::with_verification(self.index_set.clone(), query_config, verification_config)
    }

    /// Create a solver with custom configuration.
    #[must_use]
    pub fn create_solver_with_config(
        &self,
        query_config: QueryConfig,
        verification_config: VerificationConfig,
    ) -> PlateSolver {
        PlateSolver::with_verification(self.index_set.clone(), query_config, verification_config)
    }

    /// Get catalog for verification.
    #[must_use]
    pub fn catalog(&self) -> &[Star] {
        &self.catalog
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new().expect("Failed to create test harness")
    }
}

/// Ground truth for a test case.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// True WCS parameters
    pub wcs: WcsHypothesis,
    /// True sky positions of detected stars
    pub true_positions: Vec<SkyCoord>,
}

/// Configuration for generating a test case.
#[derive(Debug, Clone)]
pub struct TestCaseConfig {
    /// Field center (RA, Dec in degrees)
    pub center: SkyCoord,
    /// Field of view width in arcminutes
    pub fov_width_arcmin: f64,
    /// Field of view height in arcminutes
    pub fov_height_arcmin: f64,
    /// Image width in pixels
    pub image_width: usize,
    /// Image height in pixels
    pub image_height: usize,
    /// Rotation angle in degrees
    pub rotation_deg: f64,
    /// Number of stars to include
    pub num_stars: usize,
    /// Position noise (Gaussian sigma in pixels)
    pub position_noise_pixels: f64,
    /// Global photometric offset (magnitudes) added to *every* detection's measured
    /// brightness -- models the catalog and the image being in **different bands**
    /// (a zero-point shift). NOTE: the solver selects quad stars by brightness
    /// *ordering* (brightest-per-cell), and a uniform offset is order-preserving, so
    /// a pure global offset is invariant to the matcher -- it scales all fluxes
    /// equally and changes nothing. Modeled mainly to combine with [`Self::magnitude_noise`]
    /// (the genuinely-perturbing color/scatter term) for a realistic "wrong band" field.
    pub magnitude_offset: f64,
    /// Per-star photometric scatter: Gaussian sigma (magnitudes) of an arbitrary
    /// offset applied independently to each detection's measured brightness. Models
    /// the color-dependent part of a band change (blue vs red stars shift by
    /// different amounts) plus measurement scatter. `0.0` (default) = exact fluxes.
    /// Unlike the global offset, this *reorders* stars, so it stresses brightest-
    /// per-cell quad selection: near a `HEALPix` cell's brightness boundary it can flip
    /// which star the field treats as brightest, so the field may build a quad the
    /// index never did. Star *selection* (which stars land on the image) stays on
    /// true magnitude -- only the reported flux is perturbed, as in a real detector.
    pub magnitude_noise: f64,
    /// Magnitude limit (fainter stars excluded)
    pub magnitude_limit: f64,
    /// Random seed for reproducible noise (None = random)
    pub random_seed: Option<u64>,
    /// Radial (barrel/pincushion) distortion coefficient `k`. Each ideal pixel
    /// `(x, y)` is displaced from the image center by `(1 + k*r_n^2)`, where
    /// `r_n` is the radius normalized so the image corner has `r_n = 1`. `0.0`
    /// (default) = no distortion; small values (e.g. `0.01`) model real optics.
    /// The ground-truth WCS stays *linear*, so an undistorted fit leaves a
    /// radius-dependent residual that SIP refinement should remove.
    pub radial_distortion: f64,
    /// Remove the k brightest detections entirely -- models saturated stars a
    /// PSF-matched finder misses. Nasty for matching: the index anchors its
    /// quads on exactly these stars. Applied before [`Self::dropout_fraction`].
    pub drop_brightest: usize,
    /// Probability that each remaining detection is deleted (detection
    /// incompleteness: clouds, chip defects, a shallow exposure). `0.0`
    /// (default) = complete detection.
    pub dropout_fraction: f64,
    /// Number of spurious (non-catalog) detections to inject at random on-image
    /// positions, with fluxes up to brighter than every real star -- models hot
    /// pixels, cosmic rays, satellites, and blends. Appended *after*
    /// `GroundTruth::true_positions`, which only parallels the real detections.
    pub num_spurious: usize,
    /// The k brightest detections each gain a *duplicate* detection a pixel or
    /// two away with comparable flux -- a deblending failure on a bright (often
    /// saturated) star. Worst case for matching: the brightest stars are the
    /// index's quad anchors, and both copies compete for brightest-per-cell
    /// selection. Duplicates are appended (no `true_positions` entry).
    pub num_duplicates: usize,
}

impl Default for TestCaseConfig {
    fn default() -> Self {
        Self {
            center: SkyCoord::new(180.0, 0.0), // Random position
            fov_width_arcmin: 30.0,
            fov_height_arcmin: 20.0,
            image_width: 2048,
            image_height: 1489,
            rotation_deg: 0.0,
            num_stars: 50,
            position_noise_pixels: 0.0,
            magnitude_offset: 0.0,
            magnitude_noise: 0.0,
            magnitude_limit: 12.0,
            random_seed: Some(42), // Default deterministic seed
            radial_distortion: 0.0,
            drop_brightest: 0,
            dropout_fraction: 0.0,
            num_spurious: 0,
            num_duplicates: 0,
        }
    }
}

impl TestCaseConfig {
    /// Create a new test case configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the field center.
    #[must_use]
    pub fn center(mut self, ra: f64, dec: f64) -> Self {
        self.center = SkyCoord::new(ra, dec);
        self
    }

    /// Set the field of view in arcminutes.
    #[must_use]
    pub fn fov(mut self, width_arcmin: f64, height_arcmin: f64) -> Self {
        self.fov_width_arcmin = width_arcmin;
        self.fov_height_arcmin = height_arcmin;
        self
    }

    /// Set the image dimensions in pixels.
    #[must_use]
    pub fn image_size(mut self, width: usize, height: usize) -> Self {
        self.image_width = width;
        self.image_height = height;
        self
    }

    /// Set the rotation angle in degrees.
    #[must_use]
    pub fn rotation(mut self, degrees: f64) -> Self {
        self.rotation_deg = degrees;
        self
    }

    /// Set the number of stars.
    #[must_use]
    pub fn stars(mut self, count: usize) -> Self {
        self.num_stars = count;
        self
    }

    /// Set the position noise in pixels.
    #[must_use]
    pub fn noise(mut self, sigma_pixels: f64) -> Self {
        self.position_noise_pixels = sigma_pixels;
        self
    }

    /// Set the per-star photometric scatter: Gaussian sigma in magnitudes of an
    /// arbitrary per-star brightness offset (see [`Self::magnitude_noise`]).
    #[must_use]
    pub fn mag_noise(mut self, sigma_mag: f64) -> Self {
        self.magnitude_noise = sigma_mag;
        self
    }

    /// Set the global photometric offset in magnitudes -- a uniform "different band"
    /// zero-point shift on every star (see [`Self::magnitude_offset`]).
    #[must_use]
    pub fn mag_offset(mut self, offset_mag: f64) -> Self {
        self.magnitude_offset = offset_mag;
        self
    }

    /// Set the magnitude limit.
    #[must_use]
    pub fn mag_limit(mut self, limit: f64) -> Self {
        self.magnitude_limit = limit;
        self
    }

    /// Set the radial distortion coefficient (see [`Self::radial_distortion`]).
    #[must_use]
    pub fn distortion(mut self, k: f64) -> Self {
        self.radial_distortion = k;
        self
    }

    /// Remove the k brightest detections (see [`Self::drop_brightest`]).
    #[must_use]
    pub fn drop_brightest(mut self, k: usize) -> Self {
        self.drop_brightest = k;
        self
    }

    /// Set the random-dropout probability (see [`Self::dropout_fraction`]).
    #[must_use]
    pub fn dropout(mut self, fraction: f64) -> Self {
        self.dropout_fraction = fraction;
        self
    }

    /// Inject spurious detections (see [`Self::num_spurious`]).
    #[must_use]
    pub fn spurious(mut self, n: usize) -> Self {
        self.num_spurious = n;
        self
    }

    /// Duplicate the k brightest detections (see [`Self::num_duplicates`]).
    #[must_use]
    pub fn duplicates(mut self, k: usize) -> Self {
        self.num_duplicates = k;
        self
    }

    /// Set the random seed for deterministic noise generation.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Use random (non-deterministic) noise.
    #[must_use]
    pub fn random(mut self) -> Self {
        self.random_seed = None;
        self
    }

    /// Calculate pixel scale in arcsec/pixel.
    #[must_use]
    pub fn pixel_scale_arcsec(&self) -> f64 {
        (self.fov_width_arcmin * 60.0) / self.image_width as f64
    }
}

/// Displace an ideal pixel by a radial barrel/pincushion distortion.
///
/// `r_n` is the radius from the image center normalized so the corner is 1.0;
/// the point is scaled by `(1 + k * r_n^2)`. With `k = 0` this is a no-op.
fn apply_radial_distortion(pixel: PixelCoord, config: &TestCaseConfig) -> PixelCoord {
    let k = config.radial_distortion;
    if k == 0.0 {
        return pixel;
    }
    let cx = config.image_width as f64 / 2.0;
    let cy = config.image_height as f64 / 2.0;
    let dx = pixel.x - cx;
    let dy = pixel.y - cy;
    // Normalize by the corner radius so `r_n in [0, 1]` across the image.
    let corner = (cx * cx + cy * cy).sqrt();
    let r_n2 = (dx * dx + dy * dy) / (corner * corner);
    let factor = 1.0 + k * r_n2;
    PixelCoord::new(cx + dx * factor, cy + dy * factor)
}

/// Generate a test case with known ground truth.
///
/// # Errors
/// Currently infallible in practice; kept fallible for harness compatibility.
///
/// # Panics
/// Panics if a catalog magnitude is NaN (the brightness sort is unwrapped).
pub fn generate_test_case(
    config: &TestCaseConfig,
    catalog: &[Star],
) -> PlatersResult<(Vec<DetectedStar>, GroundTruth)> {
    // Create ground truth WCS
    let scale_arcsec_per_pixel = config.pixel_scale_arcsec();
    let wcs = WcsHypothesis::new(
        config.center,
        scale_arcsec_per_pixel,
        config.rotation_deg,
        config.image_width,
        config.image_height,
    );

    // Find stars in the field
    let fov_radius_deg = (config.fov_width_arcmin / 60.0).max(config.fov_height_arcmin / 60.0);

    println!(
        "  Looking for stars within {:.2} deg of ({:.2}, {:.2})",
        fov_radius_deg, config.center.ra, config.center.dec
    );
    println!("  Magnitude limit: {:.1}", config.magnitude_limit);
    println!(
        "  Catalog has {} stars (sample mag: {:.1})",
        catalog.len(),
        catalog.first().map_or(0.0, |s| s.magnitude)
    );

    // Stars in the cone within the magnitude limit. Project each to pixels and
    // keep only those that actually fall on the image, THEN take the brightest
    // `num_stars`. (Truncating to brightest-N *before* the in-bounds check is a
    // bug: the brightest stars are scattered across the whole cone, so most
    // project outside the rectangular image and the field ends up nearly empty.)
    let mut in_image: Vec<(Star, PixelCoord)> = catalog
        .iter()
        .filter(|star| {
            let distance = star.position.angular_distance(&config.center);
            distance < fov_radius_deg && star.magnitude < config.magnitude_limit
        })
        .filter_map(|star| {
            let pixel = wcs.sky_to_pixel(star.position).ok()?;
            // Apply optional radial distortion (the WCS itself stays linear).
            let pixel = apply_radial_distortion(pixel, config);
            let in_bounds = pixel.x >= 0.0
                && pixel.x < config.image_width as f64
                && pixel.y >= 0.0
                && pixel.y < config.image_height as f64;
            in_bounds.then_some((*star, pixel))
        })
        .collect();

    println!("  Found {} stars on the image", in_image.len());

    // Brightest first, keep the requested number.
    in_image.sort_by(|a, b| a.0.magnitude.partial_cmp(&b.0.magnitude).unwrap());
    in_image.truncate(config.num_stars);

    // Convert to detected stars with noise
    let mut detected_stars = Vec::new();
    let mut true_positions = Vec::new();

    for (star, pixel) in &in_image {
        // Add noise
        let noisy_pixel = if config.position_noise_pixels > 0.0 {
            use rand::Rng;
            use rand::SeedableRng;
            use rand_distr::StandardNormal;

            let (noise_x, noise_y): (f64, f64) = if let Some(seed) = config.random_seed {
                // Deterministic: seed + star index for reproducibility.
                let star_seed = seed.wrapping_add(detected_stars.len() as u64);
                let mut rng = rand::rngs::StdRng::seed_from_u64(star_seed);
                (rng.sample(StandardNormal), rng.sample(StandardNormal))
            } else {
                let mut rng = rand::thread_rng();
                (rng.sample(StandardNormal), rng.sample(StandardNormal))
            };

            PixelCoord::new(
                pixel.x + noise_x * config.position_noise_pixels,
                pixel.y + noise_y * config.position_noise_pixels,
            )
        } else {
            *pixel
        };

        // Photometric model on the *measured* magnitude (not which stars were
        // selected): a uniform global offset (band/zero-point) plus an arbitrary
        // per-star scatter (color terms + measurement noise). The per-star draw is
        // decorrelated from the position-noise stream via a distinct seed salt.
        let per_star = if config.magnitude_noise > 0.0 {
            use rand::Rng;
            use rand::SeedableRng;
            use rand_distr::StandardNormal;

            let draw: f64 = if let Some(seed) = config.random_seed {
                let star_seed =
                    seed.wrapping_add(0x5EED_0FF5_u64.wrapping_add(detected_stars.len() as u64));
                let mut rng = rand::rngs::StdRng::seed_from_u64(star_seed);
                rng.sample(StandardNormal)
            } else {
                rand::thread_rng().sample(StandardNormal)
            };
            draw * config.magnitude_noise
        } else {
            0.0
        };
        let measured_magnitude = star.magnitude + config.magnitude_offset + per_star;

        detected_stars.push(DetectedStar {
            x: noisy_pixel.x,
            y: noisy_pixel.y,
            flux: 10.0_f64.powf(-0.4 * measured_magnitude),
        });
        true_positions.push(star.position);
    }

    apply_corruption(config, &mut detected_stars, &mut true_positions);

    let ground_truth = GroundTruth {
        wcs,
        true_positions,
    };

    Ok((detected_stars, ground_truth))
}

/// Post-detection corruption, modeling a poor extraction (see the
/// `drop_brightest` / `dropout_fraction` / `num_spurious` config fields).
/// Order matters: saturation removes the brightest stars first, random
/// incompleteness thins what remains, then contaminants are appended (to
/// `detected` only -- `true_positions` parallels just the real detections).
fn apply_corruption(
    config: &TestCaseConfig,
    detected: &mut Vec<DetectedStar>,
    true_positions: &mut Vec<SkyCoord>,
) {
    use rand::Rng;
    use rand::SeedableRng;

    // `detected` is brightest-first (built from the magnitude-sorted list), so
    // saturation is dropping the head.
    if config.drop_brightest > 0 {
        let k = config.drop_brightest.min(detected.len());
        let _ = detected.drain(..k);
        let _ = true_positions.drain(..k);
    }

    let mut rng = match config.random_seed {
        // Distinct salt so corruption is decorrelated from the noise streams.
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(0x00BA_D1D0)),
        None => rand::rngs::StdRng::from_entropy(),
    };

    if config.dropout_fraction > 0.0 {
        let keep: Vec<bool> = detected
            .iter()
            .map(|_| rng.gen_range(0.0..1.0) >= config.dropout_fraction)
            .collect();
        let mut it = keep.iter();
        detected.retain(|_| *it.next().unwrap_or(&true));
        let mut it = keep.iter();
        true_positions.retain(|_| *it.next().unwrap_or(&true));
    }

    if config.num_duplicates > 0 && !detected.is_empty() {
        // `detected` is still brightest-first here (dropout preserves order),
        // so the deblending failures land on the quad-anchor stars.
        let k = config.num_duplicates.min(detected.len());
        for i in 0..k {
            let twin = detected[i];
            detected.push(DetectedStar {
                x: twin.x + rng.gen_range(-2.0..2.0),
                y: twin.y + rng.gen_range(-2.0..2.0),
                flux: twin.flux * rng.gen_range(0.6..1.0),
            });
        }
    }

    if config.num_spurious > 0 && !detected.is_empty() {
        let max_flux = detected.iter().map(|d| d.flux).fold(f64::MIN, f64::max);
        for _ in 0..config.num_spurious {
            // Anywhere on the image, up to brighter than every real star -- a
            // bright contaminant is the worst case, since brightest-per-cell
            // uniformization is then forced to build quads on it.
            detected.push(DetectedStar {
                x: rng.gen_range(0.0..config.image_width as f64),
                y: rng.gen_range(0.0..config.image_height as f64),
                flux: max_flux * rng.gen_range(0.2..2.0),
            });
        }
    }
}

/// Validate a solve result against ground truth.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the solve succeeded
    pub success: bool,
    /// Position error in arcseconds (RMS)
    pub position_error_arcsec: f64,
    /// Scale error as percentage
    pub scale_error_percent: f64,
    /// Rotation error in degrees
    pub rotation_error_deg: f64,
    /// Number of matched stars
    pub num_matches: usize,
}

impl ValidationResult {
    /// Check if validation passed based on thresholds.
    #[must_use]
    pub fn passes(&self, max_pos_error: f64, max_scale_error: f64, max_rot_error: f64) -> bool {
        self.success
            && self.position_error_arcsec < max_pos_error
            && self.scale_error_percent < max_scale_error
            && self.rotation_error_deg < max_rot_error
    }
}

/// Validate a WCS solution against ground truth.
#[must_use]
pub fn validate_solution(
    solved_wcs: &WcsHypothesis,
    ground_truth: &GroundTruth,
) -> ValidationResult {
    // Position error
    let center_distance = solved_wcs.center.angular_distance(&ground_truth.wcs.center);
    let position_error_arcsec = center_distance * 3600.0; // Convert to arcsec

    // Scale error
    let scale_error_percent = ((solved_wcs.scale_arcsec_per_pixel()
        - ground_truth.wcs.scale_arcsec_per_pixel())
        / ground_truth.wcs.scale_arcsec_per_pixel())
    .abs()
        * 100.0;

    // Rotation error (handle wrap-around)
    let mut rotation_error = (solved_wcs.rotation_deg() - ground_truth.wcs.rotation_deg()).abs();
    if rotation_error > 180.0 {
        rotation_error = 360.0 - rotation_error;
    }

    ValidationResult {
        success: true,
        position_error_arcsec,
        scale_error_percent,
        rotation_error_deg: rotation_error,
        num_matches: 0, // Will be filled in by caller if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_creation() {
        let harness = TestHarness::new();
        assert!(harness.is_ok());
        let harness = harness.unwrap();
        assert!(!harness.index_set.is_empty());
        assert!(!harness.catalog.is_empty());
    }

    #[test]
    fn test_case_config_builder() {
        let config = TestCaseConfig::new()
            .center(180.0, 45.0)
            .fov(20.0, 15.0)
            .image_size(1024, 768)
            .rotation(45.0)
            .stars(30)
            .noise(0.5);

        assert_eq!(config.center.ra, 180.0);
        assert_eq!(config.center.dec, 45.0);
        assert_eq!(config.fov_width_arcmin, 20.0);
        assert_eq!(config.rotation_deg, 45.0);
        assert_eq!(config.num_stars, 30);
        assert_eq!(config.position_noise_pixels, 0.5);
    }

    #[test]
    fn test_pixel_scale_calculation() {
        let config = TestCaseConfig::new().fov(30.0, 20.0).image_size(2048, 1489);

        let scale = config.pixel_scale_arcsec();
        // 30 arcmin = 1800 arcsec, divided by 2048 pixels
        let expected = 1800.0 / 2048.0;
        assert!((scale - expected).abs() < 0.01);
    }
}
