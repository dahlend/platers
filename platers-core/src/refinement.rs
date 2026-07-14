//! Iterative WCS refinement using multi-star least-squares fitting.
//!
//! This module implements a two-stage approach to plate solving:
//! 1. Coarse solve: Single quad match (existing solver) -> ~6-20 arcmin accuracy
//! 2. Iterative refinement: Multi-star fitting -> <1 arcmin accuracy
//!
//! The refinement process:
//! - Uses initial WCS to match many stars (20-50) between image and catalog
//! - Refits WCS using least-squares with all matched stars
//! - Iterates with progressively tighter match radii and outlier rejection
//! - Converges in 2-3 iterations to sub-arcminute accuracy

use crate::catalog_index::CatalogIndex;
use crate::errors::{Error, PlatersResult};
use crate::types::{DetectedStar, PixelCoord, SkyCoord, Star};
use crate::wcs::WcsHypothesis;
use tracing::debug;

/// Configuration for iterative WCS refinement.
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Maximum number of refinement iterations
    pub max_iterations: usize,

    /// Initial match radius in arcseconds (iteration 1)
    pub initial_radius_arcsec: f64,

    /// Final match radius in arcseconds (last iteration)
    pub final_radius_arcsec: f64,

    /// Minimum number of matched stars required for refinement
    pub min_stars: usize,

    /// Outlier rejection threshold in standard deviations
    /// Stars with residuals > `outlier_sigma` * RMS are rejected
    pub outlier_sigma: f64,

    /// Convergence threshold in arcseconds
    /// Stop iterating when RMS improvement < this value
    pub convergence_threshold: f64,

    /// Optional SIP distortion order to fit on the *final* WCS (`None` = linear
    /// only). SIP needs many well-spread stars to be well-conditioned, so it is
    /// applied only when at least `sip_min_stars` matched; otherwise the linear
    /// fit is used. Off by default -- enable for fields with real optical
    /// distortion.
    pub sip_order: Option<u32>,

    /// Minimum matched stars required before attempting a SIP fit.
    pub sip_min_stars: usize,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            // Start wide enough to tolerate a coarse-seed pointing error of
            // tens of arcsec (the radius shrinks toward `final` each iteration).
            // A too-tight initial radius is the classic refinement failure: if
            // the seed is offset by more than the radius, *zero* stars match.
            // The measured sweet spot recovers 20"+ seed errors to sub-arcsec.
            initial_radius_arcsec: 30.0,
            final_radius_arcsec: 1.0,   // Tight final match
            min_stars: 10,              // Need at least 10 stars
            outlier_sigma: 3.0,         // 3-sigma clipping
            convergence_threshold: 0.1, // 0.1 arcsec improvement
            sip_order: None,            // linear by default
            sip_min_stars: 25,          // SIP needs many well-spread stars
        }
    }
}

/// A matched pair of image star and catalog star.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StarMatch {
    /// Detected star in image coordinates
    pub image_star: DetectedStar,

    /// Corresponding catalog star
    pub catalog_star: Star,

    /// Astrometric residual in arcseconds (distance between projected positions)
    pub residual_arcsec: f64,
}

/// Result of iterative WCS refinement.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RefinementResult {
    /// Refined WCS solution
    pub refined_wcs: WcsHypothesis,

    /// Stars that were matched and used in final fit
    pub matched_stars: Vec<StarMatch>,

    /// RMS residual of final fit in arcseconds
    pub rms_residual_arcsec: f64,

    /// Number of iterations performed
    pub iterations: usize,

    /// Whether refinement converged (true) or hit max iterations (false)
    pub converged: bool,
}

/// Iterative WCS refiner using multi-star least-squares fitting.
#[derive(Debug)]
pub struct IterativeRefiner {
    config: RefinementConfig,
}

impl IterativeRefiner {
    /// Create a new refiner with the given configuration.
    #[must_use]
    pub fn new(config: RefinementConfig) -> Self {
        Self { config }
    }
}

impl Default for IterativeRefiner {
    fn default() -> Self {
        Self::new(RefinementConfig::default())
    }
}

impl IterativeRefiner {
    /// Refine a WCS solution using iterative multi-star fitting.
    ///
    /// # Arguments
    /// * `initial_wcs` - Starting WCS from coarse solve (single quad)
    /// * `detected_stars` - All detected stars in the image
    /// * `catalog` - Reference catalog stars in the field
    ///
    /// # Returns
    /// Refined WCS with improved accuracy, or error if refinement fails.
    ///
    /// # Errors
    /// Returns error if:
    /// - Too few stars can be matched (<`min_stars`)
    /// - WCS fitting fails (degenerate configuration)
    pub fn refine(
        &self,
        initial_wcs: WcsHypothesis,
        detected_stars: &[DetectedStar],
        catalog: &CatalogIndex,
    ) -> PlatersResult<RefinementResult> {
        let mut current_wcs = initial_wcs;
        let mut previous_rms = f64::MAX;

        for iteration in 0..self.config.max_iterations {
            // Compute match radius for this iteration (progressively tighter)
            let t = iteration as f64 / (self.config.max_iterations - 1).max(1) as f64;
            let radius = self.config.initial_radius_arcsec
                + t * (self.config.final_radius_arcsec - self.config.initial_radius_arcsec);

            // Match stars using current WCS
            let mut matches = Self::match_stars(&current_wcs, detected_stars, catalog, radius)?;

            if matches.len() < self.config.min_stars {
                return Err(Error::InsufficientData(format!(
                    "Insufficient matched stars for refinement: {} < {}",
                    matches.len(),
                    self.config.min_stars
                )));
            }

            // Reject outliers (not in first iteration)
            if iteration > 0 {
                Self::reject_outliers(&mut matches, self.config.outlier_sigma);

                if matches.len() < self.config.min_stars {
                    return Err(Error::InsufficientData(format!(
                        "Too many outliers rejected, insufficient stars remaining: {}",
                        matches.len()
                    )));
                }
            }

            // Compute RMS residual before fitting
            let rms = Self::compute_rms(&matches);

            // Check convergence
            let improvement = previous_rms - rms;
            if iteration > 0 && improvement < self.config.convergence_threshold {
                // Converged -- do the final (optionally SIP) fit on these matches,
                // then recompute the residuals under it. The `rms` above describes
                // the PREVIOUS fit; reporting it would hide exactly what the final
                // fit adds (for SIP, the whole point).
                let refined_wcs = self
                    .fit_final(&matches, current_wcs.image_width, current_wcs.image_height)
                    .unwrap_or(current_wcs);
                let final_rms = Self::refit_residuals(&refined_wcs, &mut matches);
                return Ok(RefinementResult {
                    refined_wcs,
                    matched_stars: matches,
                    rms_residual_arcsec: final_rms,
                    iterations: iteration,
                    converged: true,
                });
            }

            // Refit WCS from matched stars (linear per-iteration).
            current_wcs = Self::fit_wcs_from_matches(
                &matches,
                current_wcs.image_width,
                current_wcs.image_height,
            )?;

            previous_rms = rms;
        }

        // Max iterations reached -- final (optionally SIP) fit, residuals
        // recomputed under it (see the converged path).
        let mut final_matches = Self::match_stars(
            &current_wcs,
            detected_stars,
            catalog,
            self.config.final_radius_arcsec,
        )?;

        let refined_wcs = self
            .fit_final(
                &final_matches,
                current_wcs.image_width,
                current_wcs.image_height,
            )
            .unwrap_or(current_wcs);
        let final_rms = Self::refit_residuals(&refined_wcs, &mut final_matches);

        Ok(RefinementResult {
            refined_wcs,
            matched_stars: final_matches,
            rms_residual_arcsec: final_rms,
            iterations: self.config.max_iterations,
            converged: false,
        })
    }

    /// Match image stars to catalog stars using a WCS projection.
    ///
    /// Projects catalog stars to pixel coordinates and matches to detected stars.
    /// This is more robust than projecting detected stars to sky, because catalog
    /// positions are exact while detected positions have noise.
    fn match_stars(
        wcs: &WcsHypothesis,
        detected_stars: &[DetectedStar],
        catalog: &CatalogIndex,
        radius_arcsec: f64,
    ) -> PlatersResult<Vec<StarMatch>> {
        let mut matches = Vec::new();

        // Convert radius from arcsec to pixels (approximate)
        let radius_pixels = radius_arcsec / wcs.scale_arcsec_per_pixel();

        // Restrict to catalog stars that could land on the image: a cone around
        // the field center covering the image half-diagonal plus the match
        // radius. Avoids projecting the whole catalog every iteration.
        let half_diag_px = 0.5 * (wcs.image_width as f64).hypot(wcs.image_height as f64);
        let field_radius_deg =
            (half_diag_px * wcs.scale_arcsec_per_pixel() + radius_arcsec) / 3600.0;
        let candidates = catalog.stars_near(wcs.center, field_radius_deg);

        // Build the projector once and reuse it for every candidate.
        let projector = wcs.projector()?;

        // For each candidate catalog star, project to pixels and find the
        // nearest detected star.
        for catalog_star in &candidates {
            // Project catalog star to image
            let Ok(pixel_pos) = projector.sky_to_pixel(catalog_star.position) else {
                continue; // Star not in image
            };

            // Check if in image bounds
            if pixel_pos.x < 0.0
                || pixel_pos.x >= wcs.image_width as f64
                || pixel_pos.y < 0.0
                || pixel_pos.y >= wcs.image_height as f64
            {
                continue;
            }

            // Find nearest detected star within radius
            let mut best_match: Option<(&DetectedStar, f64)> = None;

            for detected in detected_stars {
                let dx = detected.x - pixel_pos.x;
                let dy = detected.y - pixel_pos.y;
                let distance_pixels = (dx * dx + dy * dy).sqrt();

                if distance_pixels < radius_pixels {
                    if let Some((_, best_dist)) = best_match {
                        if distance_pixels < best_dist {
                            best_match = Some((detected, distance_pixels));
                        }
                    } else {
                        best_match = Some((detected, distance_pixels));
                    }
                }
            }

            if let Some((detected, dist_pixels)) = best_match {
                // Convert distance to arcseconds for residual
                let residual_arcsec = dist_pixels * wcs.scale_arcsec_per_pixel();

                matches.push(StarMatch {
                    image_star: *detected,
                    catalog_star: *catalog_star,
                    residual_arcsec,
                });
            }
        }

        Ok(matches)
    }

    /// Fit a new WCS from star matches using least-squares.
    ///
    /// Uses the same algorithm as `WcsHypothesis::from_quad_match` but with
    /// N stars instead of 4, providing better accuracy.
    fn fit_wcs_from_matches(
        matches: &[StarMatch],
        image_width: usize,
        image_height: usize,
    ) -> PlatersResult<WcsHypothesis> {
        let image_coords: Vec<PixelCoord> = matches
            .iter()
            .map(|m| PixelCoord::new(m.image_star.x, m.image_star.y))
            .collect();
        let catalog_coords: Vec<SkyCoord> =
            matches.iter().map(|m| m.catalog_star.position).collect();

        // Per-iteration fits stay linear (SIP is applied once at the end via
        // `fit_final` -- fitting SIP on a not-yet-converged pose is ill-advised).
        WcsHypothesis::from_star_matches(&image_coords, &catalog_coords, image_width, image_height)
    }

    /// Final fit, optionally adding SIP distortion when configured and there are
    /// enough well-spread matches. Falls back to the linear fit if SIP fails.
    fn fit_final(
        &self,
        matches: &[StarMatch],
        image_width: usize,
        image_height: usize,
    ) -> PlatersResult<WcsHypothesis> {
        let image_coords: Vec<PixelCoord> = matches
            .iter()
            .map(|m| PixelCoord::new(m.image_star.x, m.image_star.y))
            .collect();
        let catalog_coords: Vec<SkyCoord> =
            matches.iter().map(|m| m.catalog_star.position).collect();

        if let Some(order) = self.config.sip_order {
            if matches.len() >= self.config.sip_min_stars {
                match WcsHypothesis::from_star_matches_sip(
                    &image_coords,
                    &catalog_coords,
                    image_width,
                    image_height,
                    Some(order),
                ) {
                    Ok(sip) => return Ok(sip),
                    // Ill-conditioned SIP fit -- fall through to linear.
                    Err(e) => debug!("SIP fit failed ({e}); using the linear fit"),
                }
            }
        }
        WcsHypothesis::from_star_matches(&image_coords, &catalog_coords, image_width, image_height)
    }

    /// Reject outlier matches based on residual threshold.
    ///
    /// Removes matches with residuals > sigma * RMS.
    fn reject_outliers(matches: &mut Vec<StarMatch>, sigma: f64) {
        if matches.is_empty() {
            return;
        }

        let rms = Self::compute_rms(matches);
        let threshold = sigma * rms;

        matches.retain(|m| m.residual_arcsec <= threshold);
    }

    /// Compute RMS residual of matches.
    /// Recompute each match's residual under `wcs` -- measured pixel -> sky via
    /// the forward projection, so a fitted SIP distortion is honored -- and
    /// return the RMS. The stored residuals describe the fit *before* the last
    /// one; the reported RMS must describe the final fit. Leaves the stored
    /// residuals untouched (and returns their RMS) if the WCS cannot project.
    fn refit_residuals(wcs: &WcsHypothesis, matches: &mut [StarMatch]) -> f64 {
        if let Ok(projector) = wcs.projector() {
            for m in matches.iter_mut() {
                let pixel = PixelCoord::new(m.image_star.x, m.image_star.y);
                if let Ok(sky) = projector.pixel_to_sky(pixel) {
                    m.residual_arcsec = sky.angular_distance(&m.catalog_star.position) * 3600.0;
                }
            }
        }
        Self::compute_rms(matches)
    }

    fn compute_rms(matches: &[StarMatch]) -> f64 {
        if matches.is_empty() {
            return 0.0;
        }

        let sum_sq: f64 = matches.iter().map(|m| m.residual_arcsec.powi(2)).sum();
        (sum_sq / matches.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_config_default() {
        let config = RefinementConfig::default();
        assert!(config.max_iterations >= 3);
        assert_eq!(config.min_stars, 10);
        // The initial radius must comfortably exceed a typical coarse-seed error
        // (tens of arcsec) and shrink toward the tight final radius.
        assert!(config.initial_radius_arcsec >= 20.0);
        assert!(config.initial_radius_arcsec > config.final_radius_arcsec);
    }

    #[test]
    fn test_refiner_creation() {
        let refiner = IterativeRefiner::default();
        assert!(refiner.config.max_iterations >= 3);
    }
}
