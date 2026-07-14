//! Bayesian verification of WCS hypotheses.
//!
//! This module implements statistical verification to rank WCS hypotheses
//! by their probability of being correct. It uses a Bayesian approach comparing:
//! - **Foreground model**: Probability of observed matches given correct WCS
//! - **Background model**: Probability of matches from random coincidences
//!
//! Based on Section 5.4 of the astrometry.net paper (arXiv:0910.2233v1).

use crate::{
    catalog_index::CatalogIndex,
    types::{DetectedStar, PixelCoord, Star},
    wcs::WcsHypothesis,
};
use std::f64::consts::PI;

/// Configuration for Bayesian verification.
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Positional error model (sigma in arcseconds)
    /// Typical value: 0.1-1.0 arcsec depending on detection quality
    pub sigma_arcsec: f64,

    /// Background star density (stars per square degree)
    /// Typical value: 100-10,000 depending on Galactic latitude
    pub background_density_per_sqdeg: f64,

    /// Search radius for matching (in arcseconds)
    /// Stars within this radius are considered matches
    pub match_radius_arcsec: f64,

    /// Minimum number of matched stars required
    pub min_matches: usize,

    /// Log-odds threshold for accepting a hypothesis
    /// Higher = more stringent (fewer false positives)
    /// Typical value: 10-20
    pub log_odds_threshold: f64,

    /// Maximum number of stars to verify (for performance)
    pub max_stars_to_verify: usize,

    /// Fraction of detected stars expected to have *no* catalog counterpart
    /// (artifacts, cosmic rays, planets, bandpass mismatch). This sets the
    /// "distractor" floor each star contributes when it has no good match.
    /// astrometry.net calls this `distractors`; typical 0.1-0.25.
    pub distractor_fraction: f64,

    /// Abandon a hypothesis early once its running log-odds falls below this
    /// (it cannot realistically recover). Only applied after at least
    /// `2 * min_matches` stars have been scored. Large-negative.
    pub early_bail_log_odds: f64,

    /// Accept a hypothesis early once its running log-odds exceeds this, to
    /// avoid scoring all `max_stars_to_verify` stars on an obvious winner.
    pub early_accept_log_odds: f64,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            sigma_arcsec: 0.5,
            background_density_per_sqdeg: 1000.0,
            match_radius_arcsec: 2.0,
            min_matches: 4,
            log_odds_threshold: 15.0,
            max_stars_to_verify: 100,
            distractor_fraction: 0.1,
            early_bail_log_odds: -20.0,
            early_accept_log_odds: 50.0,
        }
    }
}

/// Result of hypothesis verification.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationResult {
    /// The WCS hypothesis being verified
    pub wcs: WcsHypothesis,

    /// Number of foreground matches (likely real)
    pub num_matches: usize,

    /// Number of stars checked
    pub num_stars_checked: usize,

    /// Log-odds ratio (`log(P_fg` / `P_bg`))
    /// Positive = likely correct, negative = likely wrong
    pub log_odds: f64,

    /// Whether this hypothesis passes the threshold
    pub passes_threshold: bool,

    /// List of matched catalog stars
    pub matched_stars: Vec<Star>,

    /// List of matched image stars
    pub matched_detections: Vec<DetectedStar>,

    /// Match distances in arcseconds
    pub match_distances: Vec<f64>,
}

impl VerificationResult {
    /// Get the probability ratio P(correct) / P(wrong).
    #[must_use]
    pub fn probability_ratio(&self) -> f64 {
        self.log_odds.exp()
    }

    /// Get match rate (fraction of stars that matched).
    #[must_use]
    pub fn match_rate(&self) -> f64 {
        self.num_matches as f64 / self.num_stars_checked.max(1) as f64
    }

    /// Get mean match distance in arcseconds.
    #[must_use]
    pub fn mean_match_distance(&self) -> f64 {
        if self.match_distances.is_empty() {
            return f64::INFINITY;
        }
        self.match_distances.iter().sum::<f64>() / self.match_distances.len() as f64
    }

    /// Get RMS match distance in arcseconds.
    #[must_use]
    pub fn rms_match_distance(&self) -> f64 {
        if self.match_distances.is_empty() {
            return f64::INFINITY;
        }
        let sum_sq: f64 = self.match_distances.iter().map(|d| d * d).sum();
        (sum_sq / self.match_distances.len() as f64).sqrt()
    }
}

/// Bayesian verifier for WCS hypotheses.
#[derive(Debug)]
pub struct Verifier {
    config: VerificationConfig,
}

impl Verifier {
    /// Create a new verifier with the given configuration.
    #[must_use]
    pub fn new(config: VerificationConfig) -> Self {
        Self { config }
    }

    /// Verify a WCS hypothesis against detected stars and catalog.
    ///
    /// Accumulates a **distance-weighted** log-odds: each detected star, in
    /// brightness order, contributes the log-ratio of a Gaussian foreground
    /// (centered on its nearest catalog star) against a uniform + distractor
    /// background. Closer matches contribute more; far/no matches contribute the
    /// distractor floor (slightly negative). This is the astrometry.net model
    /// (Paper Sec. 5.4 / `verify.c`); unlike a count-only score it keeps the
    /// per-star distance information. Early bail/accept short-circuit obvious
    /// losers and winners.
    #[must_use]
    pub fn verify(
        &self,
        wcs: &WcsHypothesis,
        detected_stars: &[DetectedStar],
        catalog: &CatalogIndex,
    ) -> VerificationResult {
        let num_stars_to_check = detected_stars.len().min(self.config.max_stars_to_verify);

        // Build the projector once (parsing the fitsy WCS is expensive). An
        // invalid WCS yields an empty (rejected) result.
        let Ok(projector) = wcs.projector() else {
            return Self::rejected(wcs, num_stars_to_check);
        };

        // Per-star likelihood constants (see `star_log_odds`).
        let sigma = self.config.sigma_arcsec.max(1e-6);
        let two_sigma2 = 2.0 * sigma * sigma;
        // log of the foreground Gaussian peak vs. the uniform background. The
        // background is the chance a random star lands within ~1 sigma^2 cell:
        // bg_density (per deg^2) * (sigma in deg)^2. fg peak is 1/(2*pi*sigma^2)
        // normalized the same way, so the peak log-ratio reduces to:
        let sigma_deg = sigma / 3600.0;
        let bg_in_cell =
            (self.config.background_density_per_sqdeg * sigma_deg * sigma_deg).max(1e-12);
        let peak_log_odds = (1.0 / (2.0 * PI * bg_in_cell)).ln();
        // Distractor floor: a star with no good match still gets this (<= 0).
        let distractor_log_odds = self.config.distractor_fraction.max(1e-9).ln();

        let mut total_log_odds = 0.0;
        let mut num_matches = 0;
        let mut matched_stars = Vec::new();
        let mut matched_detections = Vec::new();
        let mut match_distances = Vec::new();

        for (i, detection) in detected_stars.iter().take(num_stars_to_check).enumerate() {
            let pixel_pos = PixelCoord::new(detection.x, detection.y);
            let Ok(sky_pos) = projector.pixel_to_sky(pixel_pos) else {
                continue; // Skip stars that fail projection
            };

            // Foreground term from the nearest catalog star's distance. A star
            // counts as a "match" (recorded for stats/refinement) only when it
            // is within the match radius and beats the distractor floor;
            // otherwise it contributes that floor.
            let fg_log_odds = match catalog.nearest(sky_pos) {
                Some((nearest_star, distance_arcsec)) => {
                    let d2 = distance_arcsec * distance_arcsec;
                    let star_fg = peak_log_odds - d2 / two_sigma2;
                    if distance_arcsec <= self.config.match_radius_arcsec
                        && star_fg > distractor_log_odds
                    {
                        num_matches += 1;
                        matched_stars.push(nearest_star);
                        matched_detections.push(*detection);
                        match_distances.push(distance_arcsec);
                        star_fg
                    } else {
                        distractor_log_odds
                    }
                }
                None => distractor_log_odds,
            };

            total_log_odds += fg_log_odds;

            // Early exits, but only once enough stars have been scored to be
            // meaningful.
            if i + 1 >= 2 * self.config.min_matches {
                if total_log_odds >= self.config.early_accept_log_odds {
                    break;
                }
                if total_log_odds <= self.config.early_bail_log_odds {
                    break;
                }
            }
        }

        let passes_threshold = total_log_odds >= self.config.log_odds_threshold
            && num_matches >= self.config.min_matches;

        VerificationResult {
            wcs: wcs.clone(),
            num_matches,
            num_stars_checked: num_stars_to_check,
            log_odds: total_log_odds,
            passes_threshold,
            matched_stars,
            matched_detections,
            match_distances,
        }
    }

    /// An empty, rejected result (used when the WCS can't be parsed).
    fn rejected(wcs: &WcsHypothesis, num_stars_checked: usize) -> VerificationResult {
        VerificationResult {
            wcs: wcs.clone(),
            num_matches: 0,
            num_stars_checked,
            log_odds: f64::NEG_INFINITY,
            passes_threshold: false,
            matched_stars: Vec::new(),
            matched_detections: Vec::new(),
            match_distances: Vec::new(),
        }
    }

    /// Verify multiple hypotheses and return them sorted by log-odds (best first).
    #[must_use]
    pub fn verify_and_rank(
        &self,
        hypotheses: &[WcsHypothesis],
        detected_stars: &[DetectedStar],
        catalog: &CatalogIndex,
    ) -> Vec<VerificationResult> {
        let mut results: Vec<VerificationResult> = hypotheses
            .iter()
            .map(|wcs| self.verify(wcs, detected_stars, catalog))
            .collect();

        // Sort by log-odds (descending)
        results.sort_by(|a, b| {
            b.log_odds
                .partial_cmp(&a.log_odds)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Find the best verified hypothesis, or None if none pass threshold.
    #[must_use]
    pub fn find_best(
        &self,
        hypotheses: &[WcsHypothesis],
        detected_stars: &[DetectedStar],
        catalog: &CatalogIndex,
    ) -> Option<VerificationResult> {
        let results = self.verify_and_rank(hypotheses, detected_stars, catalog);
        results.into_iter().find(|r| r.passes_threshold)
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new(VerificationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SkyCoord;

    #[test]
    fn test_verifier_creation() {
        let verifier = Verifier::default();
        assert_eq!(verifier.config.sigma_arcsec, 0.5);
        assert_eq!(verifier.config.min_matches, 4);
    }

    #[test]
    fn test_verification_config_default() {
        let config = VerificationConfig::default();
        assert!(config.sigma_arcsec > 0.0);
        assert!(config.background_density_per_sqdeg > 0.0);
        assert!(config.match_radius_arcsec > 0.0);
        assert!(config.min_matches >= 3);
    }

    /// The distance-weighted model should: (a) score an exact-match field far
    /// higher than a field with no counterparts, and (b) score a *closer* set
    /// of matches higher than a more scattered set with the same count -- the
    /// property a count-only model cannot express.
    #[test]
    fn test_distance_weighted_log_odds() {
        let verifier = Verifier::default();

        // Ground-truth WCS: 1 arcsec/px, no rotation, centered at (180, 45).
        let wcs = WcsHypothesis::new(SkyCoord::new(180.0, 45.0), 1.0, 0.0, 1024, 1024);

        // 12 detections spread across the image; project each to sky to get the
        // "true" catalog position, then build catalogs with varying accuracy.
        let projector = wcs.projector().unwrap();
        let detections: Vec<DetectedStar> = (0..12_i32)
            .map(|i| {
                let f = f64::from(i);
                DetectedStar::new(100.0 + f * 70.0, 200.0 + f * 50.0, 1000.0 - f)
            })
            .collect();
        let truth: Vec<SkyCoord> = detections
            .iter()
            .map(|d| projector.pixel_to_sky(PixelCoord::new(d.x, d.y)).unwrap())
            .collect();

        let to_star = |s: &SkyCoord| Star::with_id(s.ra, s.dec, 10.0, 0);

        // Exact catalog (0 arcsec residual).
        let exact = CatalogIndex::new(truth.iter().map(to_star).collect());
        // Same stars, nudged ~0.4 arcsec each (still within match radius).
        let nudge_deg = 0.4 / 3600.0;
        let near = CatalogIndex::new(
            truth
                .iter()
                .map(|s| to_star(&SkyCoord::new(s.ra + nudge_deg, s.dec)))
                .collect(),
        );
        // A wrong field entirely (far away).
        let wrong = CatalogIndex::new(vec![to_star(&SkyCoord::new(10.0, -30.0))]);

        let lo_exact = verifier.verify(&wcs, &detections, &exact).log_odds;
        let lo_near = verifier.verify(&wcs, &detections, &near).log_odds;
        let lo_wrong = verifier.verify(&wcs, &detections, &wrong).log_odds;

        assert!(
            lo_exact > lo_near,
            "exact matches should beat nudged: {lo_exact} vs {lo_near}"
        );
        assert!(
            lo_near > lo_wrong,
            "real (if noisy) field should beat a wrong field: {lo_near} vs {lo_wrong}"
        );
        assert!(lo_wrong < 0.0, "wrong field should be negative: {lo_wrong}");
    }

    #[test]
    fn test_verification_result_methods() {
        let wcs = WcsHypothesis::new(SkyCoord::new(180.0, 45.0), 1.0, 0.0, 1024, 1024);

        let result = VerificationResult {
            wcs,
            num_matches: 8,
            num_stars_checked: 10,
            log_odds: 15.0,
            passes_threshold: true,
            matched_stars: Vec::new(),
            matched_detections: Vec::new(),
            match_distances: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        };

        assert_eq!(result.match_rate(), 0.8);
        assert!(result.probability_ratio() > 1e6); // exp(15) ~ 3 million

        let mean_dist = result.mean_match_distance();
        assert!((mean_dist - 0.45).abs() < 0.01);

        let rms_dist = result.rms_match_distance();
        assert!(rms_dist > mean_dist); // RMS > mean for non-zero spread
    }

    #[test]
    fn test_verify_and_rank() {
        let verifier = Verifier::default();

        // Create simple test data
        let detections = vec![
            DetectedStar::new(100.0, 100.0, 1000.0),
            DetectedStar::new(200.0, 100.0, 900.0),
        ];

        let catalog = CatalogIndex::new(vec![Star {
            position: SkyCoord::new(180.0, 45.0),
            magnitude: 10.0,
            id: Some(1),
            proper_motion: None,
        }]);

        let hypotheses = vec![
            WcsHypothesis::new(SkyCoord::new(180.0, 45.0), 1.0, 0.0, 1024, 1024),
            WcsHypothesis::new(SkyCoord::new(181.0, 45.0), 1.0, 0.0, 1024, 1024),
        ];

        let results = verifier.verify_and_rank(&hypotheses, &detections, &catalog);

        // Should return 2 results
        assert_eq!(results.len(), 2);

        // Should be sorted by log-odds (best first)
        assert!(results[0].log_odds >= results[1].log_odds);
    }
}
