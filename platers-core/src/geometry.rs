//! Geometric hash code computation and coordinate transformations.
//!
//! This module implements the geometric hashing algorithm for quad-based
//! astrometric pattern matching, as described in the astrometry.net paper.

use crate::errors::{Error, PlatersResult};
use crate::types::{PixelCoord, SkyCoord};
use serde::{Deserialize, Serialize};

/// A quad of four stars used for pattern matching.
///
/// The quad consists of four stars where:
/// - Stars A and B are the most widely separated pair (defining the quad diameter)
/// - Stars C and D lie within the circle having AB as diameter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad<T> {
    /// The four stars forming the quad (ordered: A, B, C, D)
    pub stars: [T; 4],
    /// Indices of the stars in the original list
    pub indices: [usize; 4],
}

impl<T: Copy> Quad<T> {
    /// Create a new quad from four stars and their indices.
    #[must_use]
    pub const fn new(stars: [T; 4], indices: [usize; 4]) -> Self {
        Self { stars, indices }
    }
}

/// A 4D geometric hash code for a quad.
///
/// The hash code is invariant to translation, rotation, and scaling.
/// It represents the normalized positions of stars C and D in a coordinate
/// system where A is at the origin and B is at (1, 1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HashCode {
    /// Hash code components: (`x_c`, `y_c`, `x_d`, `y_d`)
    pub components: [f64; 4],
}

impl HashCode {
    /// Create a new hash code from components.
    ///
    /// # Errors
    /// [`Error::Geometry`] if components are not in [0, 1].
    pub fn new(components: [f64; 4]) -> PlatersResult<Self> {
        for (i, &val) in components.iter().enumerate() {
            if !(0.0..=1.0).contains(&val) {
                return Err(Error::Geometry(format!(
                    "Component {i} = {val} is not in [0, 1]"
                )));
            }
        }
        Ok(Self { components })
    }

    /// Create a hash code without validation (for internal use).
    #[must_use]
    pub const fn new_unchecked(components: [f64; 4]) -> Self {
        Self { components }
    }

    /// Get the hash code as a slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[f64; 4] {
        &self.components
    }

    /// Compute Euclidean distance to another hash code in 4D space.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..4 {
            let diff = self.components[i] - other.components[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Compute the geometric hash code for a quad of pixel coordinates.
///
/// The hash code is computed by:
/// 1. Finding the most widely separated pair (A, B)
/// 2. Defining a local coordinate system with A at origin, B at (1, 1)
/// 3. Transforming the other two stars (C, D) to this coordinate system
/// 4. Applying symmetry breaking to ensure uniqueness
///
/// # Errors
/// [`Error::Geometry`] if the stars are too close together
/// or in a degenerate configuration.
pub fn compute_hash_code_pixels(stars: &[PixelCoord; 4]) -> PlatersResult<HashCode> {
    let (hash, _perm) = compute_hash_code_pixels_with_permutation(stars)?;
    Ok(hash)
}

/// Compute hash code for pixel coordinates and return the canonical permutation.
///
/// Returns (`hash_code`, permutation) where `permutation[i]` is the index in the
/// original array of the i-th star in canonical (A,B,C,D) order where A-B is the
/// widest pair.
///
/// # Errors
/// [`Error::Geometry`] if the stars are too close together or in a degenerate
/// configuration.
pub fn compute_hash_code_pixels_with_permutation(
    stars: &[PixelCoord; 4],
) -> PlatersResult<(HashCode, [usize; 4])> {
    // Find the most widely separated pair
    let (a_idx, b_idx) = find_max_separation_pair_pixels(stars)?;

    let a = &stars[a_idx];
    let b = &stars[b_idx];

    // Get the other two stars
    let (c_idx_temp, d_idx_temp) = get_remaining_indices(a_idx, b_idx);
    let c = &stars[c_idx_temp];
    let d = &stars[d_idx_temp];

    // Compute hash code and check if C/D need swapping for canonical order
    let (hash_code, c_d_swapped) = compute_hash_from_points_with_swap(a, b, c, d)?;

    // Build permutation array
    let (c_idx, d_idx) = if c_d_swapped {
        (d_idx_temp, c_idx_temp)
    } else {
        (c_idx_temp, d_idx_temp)
    };

    let permutation = [a_idx, b_idx, c_idx, d_idx];
    Ok((hash_code, permutation))
}

/// Compute hash code for sky coordinates (RA/Dec).
///
/// For small fields, we can use gnomonic (tangent plane) projection
/// to approximate the spherical geometry as flat.
///
/// CRITICAL: To ensure hash codes match between sky and pixel coordinates,
/// we must find the widest pair in PROJECTED space (Euclidean distance),
/// not in angular distance. Otherwise, spherical distortion can cause
/// different pairs to be selected, producing completely different hash codes.
///
/// # Errors
/// [`Error::Geometry`] if the stars are too close together.
pub fn compute_hash_code_sky(stars: &[SkyCoord; 4]) -> PlatersResult<HashCode> {
    let (hash, _perm) = compute_hash_code_sky_with_permutation(stars)?;
    Ok(hash)
}

/// Compute hash code for sky coordinates and return the canonical permutation.
///
/// # Errors
/// [`Error::Geometry`] if the stars are too close together.
pub fn compute_hash_code_sky_with_permutation(
    stars: &[SkyCoord; 4],
) -> PlatersResult<(HashCode, [usize; 4])> {
    // Project to tangent plane around stars[0]
    let p0 = &stars[0];
    let projected = [
        gnomonic_project(p0, &stars[0]),
        gnomonic_project(p0, &stars[1]),
        gnomonic_project(p0, &stars[2]),
        gnomonic_project(p0, &stars[3]),
    ];

    // Find widest pair in projected space
    let (a_idx, b_idx) = find_max_separation_pair_pixels(&projected)?;

    let a = &projected[a_idx];
    let b = &projected[b_idx];

    // Get the other two stars
    let (c_idx_temp, d_idx_temp) = get_remaining_indices(a_idx, b_idx);
    let c = &projected[c_idx_temp];
    let d = &projected[d_idx_temp];

    // Compute hash code and check if C/D need swapping
    let (hash_code, c_d_swapped) = compute_hash_from_points_with_swap(a, b, c, d)?;

    // Build permutation array
    let (c_idx, d_idx) = if c_d_swapped {
        (d_idx_temp, c_idx_temp)
    } else {
        (c_idx_temp, d_idx_temp)
    };

    let permutation = [a_idx, b_idx, c_idx, d_idx];
    Ok((hash_code, permutation))
}

/// Find the most widely separated pair of pixels.
fn find_max_separation_pair_pixels(stars: &[PixelCoord; 4]) -> PlatersResult<(usize, usize)> {
    let mut max_dist = 0.0;
    let mut max_pair = (0, 1);

    for i in 0..4 {
        for j in (i + 1)..4 {
            let dist = stars[i].distance(&stars[j]);
            if dist > max_dist {
                max_dist = dist;
                max_pair = (i, j);
            }
        }
    }

    // Check for degenerate case
    if max_dist < 1e-6 {
        return Err(Error::Geometry(
            "All stars are too close together".to_string(),
        ));
    }

    Ok(max_pair)
}

/// The two indices in `0..4` that are neither `a_idx` nor `b_idx`.
///
/// Computed as the set complement rather than matched, so it never panics on
/// unexpected input. `a_idx`/`b_idx` are always the distinct widest-separation
/// pair from [`find_max_separation_pair_pixels`], which leaves exactly two; the
/// `0` fallbacks are unreachable under that contract.
fn get_remaining_indices(a_idx: usize, b_idx: usize) -> (usize, usize) {
    let mut rest = (0..4).filter(|&i| i != a_idx && i != b_idx);
    (rest.next().unwrap_or(0), rest.next().unwrap_or(0))
}

/// Compute hash code and return whether C and D were swapped.
///
/// Returns (`hash_code`, `c_d_swapped`) where `c_d_swapped` indicates if C and D
/// were swapped during symmetry breaking to create the canonical ordering.
fn compute_hash_from_points_with_swap(
    a: &PixelCoord,
    b: &PixelCoord,
    c: &PixelCoord,
    d: &PixelCoord,
) -> PlatersResult<(HashCode, bool)> {
    // Vector from A to B
    let ab_x = b.x - a.x;
    let ab_y = b.y - a.y;

    // Magnitude of AB
    let ab_mag = (ab_x * ab_x + ab_y * ab_y).sqrt();

    if ab_mag < 1e-10 {
        return Err(Error::Geometry(
            "Stars A and B are at the same position".to_string(),
        ));
    }

    // Transform C and D to local coordinates
    // In this system, A is at (0, 0) and B is at (1, 1)
    let (mut x_c, mut y_c) = transform_to_local(a, ab_x, ab_y, ab_mag, c);
    let (mut x_d, mut y_d) = transform_to_local(a, ab_x, ab_y, ab_mag, d);

    // Apply symmetry breaking to ensure uniqueness
    // Rule 1: x_c <= x_d
    let mut c_d_swapped = false;
    if x_c > x_d {
        std::mem::swap(&mut x_c, &mut x_d);
        std::mem::swap(&mut y_c, &mut y_d);
        c_d_swapped = true;
    }

    // Rule 2: x_c + x_d <= 1 (if not, flip both horizontally)
    if x_c + x_d > 1.0 {
        x_c = 1.0 - x_c;
        x_d = 1.0 - x_d;
    }

    // Clamp to [0, 1] to handle numerical errors
    let x_c = x_c.clamp(0.0, 1.0);
    let y_c = y_c.clamp(0.0, 1.0);
    let x_d = x_d.clamp(0.0, 1.0);
    let y_d = y_d.clamp(0.0, 1.0);

    Ok((HashCode::new_unchecked([x_c, y_c, x_d, y_d]), c_d_swapped))
}

/// Transform a point to local coordinates where A is at origin and B is at (1, 1).
fn transform_to_local(
    a: &PixelCoord,
    ab_x: f64,
    ab_y: f64,
    ab_mag: f64,
    point: &PixelCoord,
) -> (f64, f64) {
    // Vector from A to point
    let ap_x = point.x - a.x;
    let ap_y = point.y - a.y;

    // Project onto AB direction (normalized)
    let along = (ap_x * ab_x + ap_y * ab_y) / (ab_mag * ab_mag);

    // Perpendicular component
    let perp_x = ap_x - along * ab_x;
    let perp_y = ap_y - along * ab_y;
    let perp = (perp_x * perp_x + perp_y * perp_y).sqrt();

    // Determine sign of perpendicular component
    // Cross product to determine which side of AB the point is on
    let cross = ab_x * ap_y - ab_y * ap_x;
    let perp_signed = if cross >= 0.0 { perp } else { -perp };

    // Normalize to (1, 1) coordinate system
    // B is at distance ab_mag along AB, which maps to (1, 1)
    let x = along;
    let y = perp_signed / ab_mag;

    // Transform to make B at (1, 1) instead of (1, 0)
    // This is the astrometry.net convention
    (x, x + y)
}

/// Squared chord length on the unit sphere for a given angular radius.
///
/// For angle `theta`, the chord between two unit vectors is `2 sin(theta/2)`, so
/// the squared chord is `2 (1 - cos theta)`. KD-tree range queries over unit
/// vectors use squared Euclidean distance, which equals this squared chord.
#[must_use]
pub fn chord_sq_for_angle(angle_rad: f64) -> f64 {
    let h = (angle_rad / 2.0).sin();
    4.0 * h * h
}

/// Convert a squared chord length back to an angular distance in arcseconds.
///
/// Inverse of [`chord_sq_for_angle`]: `chord = 2 sin(theta/2)`, so
/// `theta = 2 asin(chord/2)`.
#[must_use]
pub fn chord_sq_to_arcsec(chord_sq: f64) -> f64 {
    let half_chord = (chord_sq.max(0.0).sqrt() / 2.0).clamp(-1.0, 1.0);
    let angle_rad = 2.0 * half_chord.asin();
    angle_rad.to_degrees() * 3600.0
}

/// Simple gnomonic projection for small fields.
///
/// Projects a point on the sphere to a tangent plane at the reference point.
///
/// CRITICAL: We negate the x-coordinate to match the pixel coordinate convention:
///
/// - In sky coordinates: RA increases eastward (counterclockwise when viewed from north pole)
/// - In pixel coordinates: x increases rightward (which is westward on sky)
///
/// This ensures geometric hash codes match between pixel and sky quads.
///
/// We use a standard tangent plane projection with cos(dec) correction applied at
/// the reference declination, matching how WCS transformations work.
fn gnomonic_project(reference: &SkyCoord, point: &SkyCoord) -> PixelCoord {
    // Standard tangent plane projection
    // Apply cos(dec) at the REFERENCE declination for consistency

    let d_ra = point.ra - reference.ra;
    let d_dec = point.dec - reference.dec;

    // Apply cos(dec) correction to RA at the reference declination
    // Negate to match pixel coordinate convention (x increases westward)
    let x = -d_ra * reference.dec.to_radians().cos();
    let y = d_dec;

    // Scale to pixel-like units
    // (1 degree ~ 3600 arcsec, so use a similar scale)
    let scale = 3600.0;

    PixelCoord::new(x * scale, y * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hash_code_creation() {
        let hash = HashCode::new([0.5, 0.5, 0.3, 0.7]).unwrap();
        assert_eq!(hash.components, [0.5, 0.5, 0.3, 0.7]);
    }

    #[test]
    fn test_hash_code_invalid() {
        assert!(HashCode::new([1.5, 0.5, 0.3, 0.7]).is_err());
        assert!(HashCode::new([-0.1, 0.5, 0.3, 0.7]).is_err());
    }

    #[test]
    fn test_hash_code_distance() {
        let h1 = HashCode::new_unchecked([0.5, 0.5, 0.3, 0.7]);
        let h2 = HashCode::new_unchecked([0.5, 0.5, 0.3, 0.7]);
        assert_relative_eq!(h1.distance(&h2), 0.0, epsilon = 1e-10);

        let h3 = HashCode::new_unchecked([0.6, 0.5, 0.3, 0.7]);
        let dist = h1.distance(&h3);
        assert_relative_eq!(dist, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_hash_simple_square() {
        // Square with corners at (0,0), (1,0), (0,1), (1,1)
        let stars = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(1.0, 0.0),
            PixelCoord::new(0.0, 1.0),
            PixelCoord::new(1.0, 1.0),
        ];

        let hash = compute_hash_code_pixels(&stars).unwrap();

        // All components should be in [0, 1]
        for &val in &hash.components {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_compute_hash_invariance_translation() {
        // Test that hash code is invariant to translation
        let stars1 = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(10.0, 0.0),
            PixelCoord::new(3.0, 4.0),
            PixelCoord::new(7.0, 6.0),
        ];

        // Translate by (100, 200)
        let stars2 = [
            PixelCoord::new(100.0, 200.0),
            PixelCoord::new(110.0, 200.0),
            PixelCoord::new(103.0, 204.0),
            PixelCoord::new(107.0, 206.0),
        ];

        let hash1 = compute_hash_code_pixels(&stars1).unwrap();
        let hash2 = compute_hash_code_pixels(&stars2).unwrap();

        // Should be identical (or very close due to numerical errors)
        assert_relative_eq!(hash1.distance(&hash2), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_hash_invariance_scaling() {
        // Test that hash code is invariant to scaling
        let stars1 = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(10.0, 0.0),
            PixelCoord::new(3.0, 4.0),
            PixelCoord::new(7.0, 6.0),
        ];

        // Scale by 2.0
        let stars2 = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(20.0, 0.0),
            PixelCoord::new(6.0, 8.0),
            PixelCoord::new(14.0, 12.0),
        ];

        let hash1 = compute_hash_code_pixels(&stars1).unwrap();
        let hash2 = compute_hash_code_pixels(&stars2).unwrap();

        assert_relative_eq!(hash1.distance(&hash2), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_hash_invariance_rotation() {
        // Test rotation invariance - rotate by 90 degrees
        let stars1 = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(10.0, 0.0),
            PixelCoord::new(3.0, 4.0),
            PixelCoord::new(7.0, 6.0),
        ];

        // Rotate 90 degrees counterclockwise around origin
        let stars2 = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(0.0, 10.0),
            PixelCoord::new(-4.0, 3.0),
            PixelCoord::new(-6.0, 7.0),
        ];

        let hash1 = compute_hash_code_pixels(&stars1).unwrap();
        let hash2 = compute_hash_code_pixels(&stars2).unwrap();

        assert_relative_eq!(hash1.distance(&hash2), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_degenerate_quad() {
        // All stars at the same position
        let stars = [
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(0.0, 0.0),
            PixelCoord::new(0.0, 0.0),
        ];

        assert!(compute_hash_code_pixels(&stars).is_err());
    }

    #[test]
    fn test_sky_coord_hash() {
        // Test with sky coordinates
        let stars = [
            SkyCoord::new(180.0, 0.0),
            SkyCoord::new(180.1, 0.0),
            SkyCoord::new(180.0, 0.05),
            SkyCoord::new(180.05, 0.05),
        ];

        let hash = compute_hash_code_sky(&stars).unwrap();

        // All components should be in [0, 1]
        for &val in &hash.components {
            assert!((0.0..=1.0).contains(&val), "Component {val} not in [0, 1]");
        }
    }
}
