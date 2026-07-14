//! Core data types for plate solving.

use crate::errors::{Error, PlatersResult};
use serde::{Deserialize, Serialize};

/// A celestial coordinate (RA, Dec) in degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyCoord {
    /// Right Ascension in degrees (0-360)
    pub ra: f64,
    /// Declination in degrees (-90 to 90)
    pub dec: f64,
}

impl SkyCoord {
    /// Create a new sky coordinate.
    ///
    /// # Panics
    /// Panics if RA is not in [0, 360) or Dec is not in [-90, 90].
    #[must_use]
    pub fn new(ra: f64, dec: f64) -> Self {
        assert!(
            (0.0..360.0).contains(&ra),
            "RA must be in [0, 360), got {ra}"
        );
        assert!(
            (-90.0..=90.0).contains(&dec),
            "Dec must be in [-90, 90], got {dec}"
        );
        Self { ra, dec }
    }

    /// Construct from possibly-out-of-range *computed* values, wrapping RA into
    /// `[0, 360)` and clamping Dec into `[-90, 90]`.
    ///
    /// Use this on values produced by projection or least-squares fitting, which
    /// can legitimately land just outside the valid range near the RA 0/360 seam
    /// or the poles. [`new`](Self::new) panics on out-of-range input as a
    /// programming-error guard for *user-supplied* data; computed paths should
    /// normalize instead of risking a panic mid-solve.
    #[must_use]
    pub fn new_normalized(ra: f64, dec: f64) -> Self {
        Self {
            ra: ra.rem_euclid(360.0),
            dec: dec.clamp(-90.0, 90.0),
        }
    }

    /// The position as a 3D unit vector `[cosdelta*cosalpha, cosdelta*sinalpha, sindelta]`. This is
    /// the canonical Cartesian form used for wrap-safe spherical math.
    #[must_use]
    pub fn to_unit_vector(&self) -> [f64; 3] {
        let ra = self.ra.to_radians();
        let dec = self.dec.to_radians();
        let cos_dec = dec.cos();
        [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
    }

    /// Inverse of [`to_unit_vector`](Self::to_unit_vector): recover (RA, Dec)
    /// from a Cartesian vector. The vector need not be unit length (it's
    /// normalized here), so a summed/averaged vector maps correctly. RA is
    /// wrapped into `[0, 360)` and Dec clamped, so the result is always valid.
    #[must_use]
    pub fn from_unit_vector(v: [f64; 3]) -> Self {
        let [x, y, z] = v;
        let norm = (x * x + y * y + z * z).sqrt();
        let dec = if norm > 0.0 {
            (z / norm).clamp(-1.0, 1.0).asin().to_degrees()
        } else {
            0.0
        };
        let ra = y.atan2(x).to_degrees().rem_euclid(360.0);
        Self { ra, dec }
    }

    /// Try to create a new sky coordinate, returning an error if invalid.
    ///
    /// # Errors
    /// [`Error::InvalidCoordinate`] if RA is not in [0, 360) or Dec is not in [-90, 90].
    pub fn try_new(ra: f64, dec: f64) -> PlatersResult<Self> {
        if !(0.0..360.0).contains(&ra) {
            return Err(Error::InvalidCoordinate(format!(
                "RA must be in [0, 360), got {ra}"
            )));
        }
        if !(-90.0..=90.0).contains(&dec) {
            return Err(Error::InvalidCoordinate(format!(
                "Dec must be in [-90, 90], got {dec}"
            )));
        }
        Ok(Self { ra, dec })
    }

    /// Angular distance to another coordinate in degrees.
    #[must_use]
    pub fn angular_distance(&self, other: &Self) -> f64 {
        // Haversine formula
        let d_dec = (other.dec - self.dec).to_radians();
        let d_ra = (other.ra - self.ra).to_radians();
        let dec1 = self.dec.to_radians();
        let dec2 = other.dec.to_radians();

        let a = (d_dec / 2.0).sin().powi(2) + dec1.cos() * dec2.cos() * (d_ra / 2.0).sin().powi(2);
        // Rounding can push `a` past 1.0 for near-antipodal points; clamp so
        // `asin` cannot go NaN.
        let c = 2.0 * a.min(1.0).sqrt().asin();
        c.to_degrees()
    }

    /// Compute the centroid (mean position) of four sky coordinates.
    ///
    /// Averages on the unit sphere, so the result is correct across the RA
    /// 0/360 seam and near the poles -- unlike a naive arithmetic mean of RA/Dec,
    /// which produces a wildly wrong center for a quad straddling RA 0.
    #[must_use]
    pub fn centroid_of_four(coords: &[Self; 4]) -> Self {
        let mut sum = [0.0; 3];
        for c in coords {
            let v = c.to_unit_vector();
            sum[0] += v[0];
            sum[1] += v[1];
            sum[2] += v[2];
        }
        let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        if norm < 1e-12 {
            // Degenerate (e.g. antipodal cancellation); no meaningful mean.
            return coords[0];
        }
        Self::from_unit_vector(sum)
    }
}

/// A star in the reference catalog.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Star {
    /// Position on the sky, at the catalog reference epoch
    /// ([`CATALOG_EPOCH`](crate::catalog::CATALOG_EPOCH)) when the star has a
    /// proper motion
    pub position: SkyCoord,
    /// Magnitude in the reference band
    pub magnitude: f64,
    /// Optional catalog ID
    pub id: Option<u64>,
    /// Proper motion `[mu_alpha*cos(delta), mu_delta]` in mas/year (the Gaia
    /// convention: the RA component already includes the `cos(delta)` factor).
    /// `None` when the source catalog carries no proper motion.
    #[serde(default)]
    pub proper_motion: Option<[f32; 2]>,
}

impl Star {
    /// Create a new star.
    #[must_use]
    pub fn new(ra: f64, dec: f64, magnitude: f64) -> Self {
        Self {
            position: SkyCoord::new(ra, dec),
            magnitude,
            id: None,
            proper_motion: None,
        }
    }

    /// Create a new star with an ID.
    #[must_use]
    pub fn with_id(ra: f64, dec: f64, magnitude: f64, id: u64) -> Self {
        Self {
            position: SkyCoord::new(ra, dec),
            magnitude,
            id: Some(id),
            proper_motion: None,
        }
    }

    /// The star's position propagated from the catalog reference epoch
    /// ([`CATALOG_EPOCH`](crate::catalog::CATALOG_EPOCH)) to `epoch` (a Julian
    /// year, e.g. `2021.4`) along its proper motion. Identity for stars
    /// without one.
    ///
    /// Linear tangent-plane propagation: exact to well below a mas over
    /// decade-scale baselines, except within arcminutes of a celestial pole
    /// (where the `cos(delta)` division is clamped rather than allowed to
    /// blow up).
    #[must_use]
    pub fn position_at_epoch(&self, epoch: f64) -> SkyCoord {
        let Some([pm_ra_cosd, pm_dec]) = self.proper_motion else {
            return self.position;
        };
        let dt_years = epoch - crate::catalog::CATALOG_EPOCH;
        // mas -> degrees.
        let d_dec = f64::from(pm_dec) * dt_years / 3.6e6;
        let cos_dec = self.position.dec.to_radians().cos().max(1e-6);
        let d_ra = f64::from(pm_ra_cosd) * dt_years / (3.6e6 * cos_dec);
        SkyCoord::new_normalized(self.position.ra + d_ra, self.position.dec + d_dec)
    }
}

/// A detected star in pixel coordinates (output from star detection software).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectedStar {
    /// X pixel coordinate
    pub x: f64,
    /// Y pixel coordinate
    pub y: f64,
    /// Flux/brightness in arbitrary units
    pub flux: f64,
}

impl DetectedStar {
    /// Create a new detected star.
    #[must_use]
    pub const fn new(x: f64, y: f64, flux: f64) -> Self {
        Self { x, y, flux }
    }

    /// Euclidean distance to another pixel coordinate.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Sort detections brightest-first (descending flux).
    ///
    /// Stable, and treats incomparable (NaN) fluxes as equal so a non-finite
    /// flux cannot panic the sort.
    pub fn sort_brightest_first(stars: &mut [Self]) {
        stars.sort_by(|a, b| {
            b.flux
                .partial_cmp(&a.flux)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Convert an array of 4 detected stars to pixel coordinates.
    #[must_use]
    pub fn array_to_pixel_coords(stars: &[Self; 4]) -> [PixelCoord; 4] {
        [
            stars[0].into(),
            stars[1].into(),
            stars[2].into(),
            stars[3].into(),
        ]
    }
}

/// A field of detected stars together with the image dimensions.
///
/// This is the input to a plate solve: the star list (from any detection
/// software) plus the width/height of the image it came from, which the solver
/// needs to anchor the WCS at the image center.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedField {
    /// Detected stars in pixel coordinates.
    pub stars: Vec<DetectedStar>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl DetectedField {
    /// Create a new detected field.
    #[must_use]
    pub fn new(stars: Vec<DetectedStar>, width: usize, height: usize) -> Self {
        Self {
            stars,
            width,
            height,
        }
    }

    /// Length of the image diagonal in pixels.
    #[must_use]
    pub fn diagonal_px(&self) -> f64 {
        // Computed in f64 so absurd dimensions cannot overflow the integer square.
        (self.width as f64).hypot(self.height as f64)
    }
}

/// Pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelCoord {
    /// X pixel coordinate
    pub x: f64,
    /// Y pixel coordinate
    pub y: f64,
}

impl PixelCoord {
    /// Create a new pixel coordinate.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another pixel coordinate.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Compute the centroid (mean position) of four pixel coordinates.
    #[must_use]
    pub fn centroid_of_four(coords: &[Self; 4]) -> Self {
        let x = (coords[0].x + coords[1].x + coords[2].x + coords[3].x) / 4.0;
        let y = (coords[0].y + coords[1].y + coords[2].y + coords[3].y) / 4.0;
        Self { x, y }
    }
}

impl From<DetectedStar> for PixelCoord {
    fn from(star: DetectedStar) -> Self {
        Self {
            x: star.x,
            y: star.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sky_coord_angular_distance() {
        let coord1 = SkyCoord::new(0.0, 0.0);
        let coord2 = SkyCoord::new(1.0, 0.0);
        let dist = coord1.angular_distance(&coord2);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "RA")]
    fn test_sky_coord_invalid_ra() {
        let _ = SkyCoord::new(361.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Dec")]
    fn test_sky_coord_invalid_dec() {
        let _ = SkyCoord::new(0.0, 91.0);
    }

    #[test]
    fn test_sky_coord_try_new_valid() {
        let coord = SkyCoord::try_new(180.0, 45.0);
        assert!(coord.is_ok());
        let coord = coord.unwrap();
        assert_eq!(coord.ra, 180.0);
        assert_eq!(coord.dec, 45.0);
    }

    #[test]
    fn test_sky_coord_try_new_invalid_ra() {
        let coord = SkyCoord::try_new(361.0, 0.0);
        assert!(matches!(coord, Err(Error::InvalidCoordinate(_))));
    }

    #[test]
    fn test_sky_coord_try_new_invalid_dec() {
        let coord = SkyCoord::try_new(0.0, 91.0);
        assert!(matches!(coord, Err(Error::InvalidCoordinate(_))));
    }

    #[test]
    fn test_detected_star_distance() {
        let star1 = DetectedStar::new(0.0, 0.0, 100.0);
        let star2 = DetectedStar::new(3.0, 4.0, 100.0);
        let dist = star1.distance(&star2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_new_normalized_wraps_ra_and_clamps_dec() {
        // RA just below 0 and at/above 360 wrap into [0, 360); Dec clamps.
        let a = SkyCoord::new_normalized(-0.3, 91.0);
        assert!((a.ra - 359.7).abs() < 1e-9, "ra={}", a.ra);
        assert_eq!(a.dec, 90.0);
        let b = SkyCoord::new_normalized(360.2, -95.0);
        assert!((b.ra - 0.2).abs() < 1e-9, "ra={}", b.ra);
        assert_eq!(b.dec, -90.0);
        // In-range values pass through unchanged.
        let c = SkyCoord::new_normalized(180.0, 45.0);
        assert_eq!((c.ra, c.dec), (180.0, 45.0));
    }

    #[test]
    fn test_unit_vector_round_trip() {
        for (ra, dec) in [(0.0, 0.0), (359.9, 0.0), (180.0, 45.0), (90.0, -89.0)] {
            let c = SkyCoord::new(ra, dec);
            let back = SkyCoord::from_unit_vector(c.to_unit_vector());
            assert!(c.angular_distance(&back) < 1e-9, "{ra},{dec} -> {back:?}");
        }
    }

    #[test]
    fn test_centroid_of_four_across_ra_seam() {
        // A quad straddling RA 0/360. The true center is RA ~ 0, Dec ~ 0.
        // A naive arithmetic mean of RA would give ~ 180 (catastrophically wrong);
        // the spherical centroid must stay near the seam.
        let quad = [
            SkyCoord::new(359.5, 0.5),
            SkyCoord::new(0.5, 0.5),
            SkyCoord::new(359.5, -0.5),
            SkyCoord::new(0.5, -0.5),
        ];
        let c = SkyCoord::centroid_of_four(&quad);
        // Distance from the true center (RA 0, Dec 0) must be small.
        let dist = SkyCoord::new(0.0, 0.0).angular_distance(&c);
        assert!(
            dist < 0.01,
            "centroid {c:?} is {dist} deg from the seam center"
        );
        // And explicitly NOT near 180.
        assert!(
            !(90.0..270.0).contains(&c.ra),
            "centroid RA wrapped to the wrong side: {}",
            c.ra
        );
    }

    #[test]
    fn test_centroid_of_four_simple_field() {
        // Away from the seam it still behaves like an ordinary mean.
        let quad = [
            SkyCoord::new(179.9, 44.9),
            SkyCoord::new(180.1, 44.9),
            SkyCoord::new(179.9, 45.1),
            SkyCoord::new(180.1, 45.1),
        ];
        let c = SkyCoord::centroid_of_four(&quad);
        assert!(SkyCoord::new(180.0, 45.0).angular_distance(&c) < 1e-3);
    }

    /// Proper-motion propagation: Gaia-convention components over a 10-year
    /// baseline, including the `cos(delta)` de-projection of the RA component.
    #[test]
    fn test_position_at_epoch() {
        let mut star = Star::new(100.0, 60.0, 9.0);

        // No proper motion: identity at any epoch.
        assert_eq!(star.position_at_epoch(2026.0), star.position);

        // 1000 mas/yr in RA*cos(dec), -500 mas/yr in Dec, for 10 years:
        // dec moves -5000 mas; RA moves +10000 mas / cos(60 deg) = +20000 mas
        // in coordinate terms.
        star.proper_motion = Some([1000.0, -500.0]);
        let p = star.position_at_epoch(crate::catalog::CATALOG_EPOCH + 10.0);
        assert!((p.dec - (60.0 - 5000.0 / 3.6e6)).abs() < 1e-9);
        assert!((p.ra - (100.0 + 10_000.0 / 3.6e6 / 60.0_f64.to_radians().cos())).abs() < 1e-9);
        // The on-sky displacement is sqrt(1000^2 + 500^2) mas/yr x 10 yr.
        let moved = star.position.angular_distance(&p) * 3.6e6;
        assert!((moved - 10.0 * (1000.0_f64.powi(2) + 500.0_f64.powi(2)).sqrt()).abs() < 5.0);
    }
}
