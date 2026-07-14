//! `HEALPix` binning for density uniformization.
//!
//! [`HealPixGrid`] bins stars into equal-area sky cells and answers
//! "brightest N per cell" -- the uniformization primitive index building is
//! built on. Catalog *queries* (cone / nearest-neighbor) live in
//! [`crate::catalog_index`], not here.

use crate::errors::{Error, PlatersResult};
use crate::types::{SkyCoord, Star};
use std::collections::HashMap;

/// HEALPix-based spatial index for stars.
///
/// Divides the sky into equal-area cells for efficient spatial queries.
#[derive(Debug)]
pub struct HealPixGrid {
    /// `HEALPix` depth (resolution)
    depth: u8,
    /// Stars organized by `HEALPix` cell
    cells: HashMap<u64, Vec<Star>>,
    /// Total number of stars
    total_stars: usize,
}

impl HealPixGrid {
    /// Create a new `HEALPix` grid with the given depth.
    ///
    /// # Errors
    /// [`Error::Spatial`] if depth is invalid (> 29).
    pub fn new(depth: u8) -> PlatersResult<Self> {
        if depth > 29 {
            return Err(Error::Spatial(format!(
                "HEALPix depth {depth} exceeds maximum of 29"
            )));
        }

        Ok(Self {
            depth,
            cells: HashMap::new(),
            total_stars: 0,
        })
    }

    /// Insert a star into the grid.
    pub fn insert(&mut self, star: Star) {
        let hash = self.get_cell_hash(&star.position);
        self.cells.entry(hash).or_default().push(star);
        self.total_stars += 1;
    }

    /// Insert multiple stars.
    pub fn insert_many(&mut self, stars: &[Star]) {
        for star in stars {
            self.insert(*star);
        }
    }

    /// Get the `HEALPix` cell hash for a coordinate.
    fn get_cell_hash(&self, coord: &SkyCoord) -> u64 {
        // Use cdshealpix library
        let lon = coord.ra.to_radians();
        let lat = coord.dec.to_radians();
        cdshealpix::nested::hash(self.depth, lon, lat)
    }

    /// Get the N brightest stars from a cell.
    #[must_use]
    pub fn get_brightest_in_cell(&self, cell_hash: u64, n: usize) -> Vec<Star> {
        if let Some(stars) = self.cells.get(&cell_hash) {
            let mut sorted: Vec<Star> = stars.clone();
            sorted.sort_by(|a, b| {
                a.magnitude
                    .partial_cmp(&b.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted.truncate(n);
            sorted
        } else {
            Vec::new()
        }
    }

    /// Get all cell hashes in the grid.
    #[must_use]
    pub fn cell_hashes(&self) -> Vec<u64> {
        self.cells.keys().copied().collect()
    }

    /// Get the number of cells with stars.
    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Get the total number of stars.
    #[must_use]
    pub const fn total_stars(&self) -> usize {
        self.total_stars
    }

    /// Get the depth of the grid.
    #[must_use]
    pub const fn depth(&self) -> u8 {
        self.depth
    }

    /// Get the approximate cell size in degrees.
    #[must_use]
    pub fn cell_size_degrees(&self) -> f64 {
        let n_cells = 12_u64 * 4_u64.pow(u32::from(self.depth));
        let cell_area = 4.0 * std::f64::consts::PI / n_cells as f64;
        cell_area.sqrt().to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healpix_grid_creation() {
        let grid = HealPixGrid::new(5).unwrap();
        assert_eq!(grid.depth(), 5);
        assert_eq!(grid.total_stars(), 0);
    }

    #[test]
    fn test_healpix_grid_invalid_depth() {
        assert!(HealPixGrid::new(30).is_err());
    }

    #[test]
    fn test_healpix_insert() {
        let mut grid = HealPixGrid::new(5).unwrap();
        let star = Star::new(180.0, 0.0, 10.0);
        grid.insert(star);

        assert_eq!(grid.total_stars(), 1);
        assert_eq!(grid.num_cells(), 1);
    }

    #[test]
    fn test_healpix_insert_many() {
        let mut grid = HealPixGrid::new(5).unwrap();
        let stars = vec![
            Star::new(180.0, 0.0, 10.0),
            Star::new(180.1, 0.0, 11.0),
            Star::new(0.0, 45.0, 12.0),
        ];
        grid.insert_many(&stars);

        assert_eq!(grid.total_stars(), 3);
    }

    #[test]
    fn test_healpix_get_brightest() {
        let mut grid = HealPixGrid::new(5).unwrap();

        // Insert stars at the exact same position (same cell)
        grid.insert(Star::new(180.0, 0.0, 10.0));
        grid.insert(Star::new(180.0, 0.0, 8.0));
        grid.insert(Star::new(180.0, 0.0, 12.0));

        let coord = SkyCoord::new(180.0, 0.0);
        let hash = grid.get_cell_hash(&coord);

        let brightest = grid.get_brightest_in_cell(hash, 2);
        assert_eq!(brightest.len(), 2);
        assert_eq!(brightest[0].magnitude, 8.0);
        assert_eq!(brightest[1].magnitude, 10.0);
    }
}
