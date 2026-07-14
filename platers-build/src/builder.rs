//! Index builder for creating quad-based plate solving indices.

use platers_core::{
    geometry::{chord_sq_for_angle, compute_hash_code_sky, HashCode},
    query::enumerate_distance_bounded_quads,
    types::Star,
    Error, PlatersResult,
};
use std::collections::HashMap;

/// Near-neighbours per anchor the *catalog* pairs with. The field generator uses a
/// few more (`stars_per_cell + slack`), so the catalog's quads stay a **subset** of
/// the field's over the same stars -- every catalog quad can be hit.
const CATALOG_MAX_NEIGHBORS: usize = 12;

/// Target number of quads per build-grid `HEALPix` cell -- the **fixed quad density
/// per area** quota. A cell with fewer available in-band quads keeps them all
/// (fills sparse sky, where matching was failing); a cell with more keeps the
/// brightest-first subset (caps dense sky, bounding the index). This decouples
/// quad coverage from the wildly-varying star density. See `build`.
const TARGET_QUADS_PER_CELL: usize = 16;

/// Configuration for index building.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// `HEALPix` depth for spatial partitioning
    pub healpix_depth: u8,
    /// Number of stars to select per `HEALPix` cell
    pub stars_per_cell: usize,
    /// Minimum quad diameter in degrees
    pub min_diameter_deg: f64,
    /// Maximum quad diameter in degrees
    pub max_diameter_deg: f64,

    // Scale-range metadata for FOV optimization.
    /// Target field of view this index covers (degrees)
    pub target_fov_deg: f64,
    /// Minimum pixel scale this index is optimized for (arcsec/pixel)
    pub min_scale_arcsec_per_pixel: f64,
    /// Maximum pixel scale this index is optimized for (arcsec/pixel)
    pub max_scale_arcsec_per_pixel: f64,

    /// Target uniform **stellar density** for the index, in stars per square
    /// degree. When set, `for_scale_range` derives `stars_per_cell` per tier from
    /// the tier's `HEALPix` cell area, so every scale carries the same sky density
    /// (and the catalog can be far deeper than the index needs). `None` keeps the
    /// flat `stars_per_cell`.
    pub target_density_per_deg2: Option<f64>,
}

/// Area of one `HEALPix` cell at `depth` (nside = 2^depth), in square degrees.
/// Whole sky is 41252.96 deg^2 over `12 x 4^depth` equal-area cells.
#[must_use]
pub fn cell_area_deg2(depth: u8) -> f64 {
    41_252.961_25 / (12.0 * 4_f64.powi(i32::from(depth)))
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            healpix_depth: 5,
            stars_per_cell: 10,
            min_diameter_deg: 0.05, // ~3 arcmin
            max_diameter_deg: 0.1,  // ~6 arcmin
            target_fov_deg: 0.3,    // ~18 arcmin (3x max_diameter)
            min_scale_arcsec_per_pixel: 0.3,
            max_scale_arcsec_per_pixel: 0.6,
            target_density_per_deg2: None,
        }
    }
}

impl IndexConfig {
    /// Create a config for a specific scale range (multi-scale index building).
    #[must_use]
    pub fn for_scale_range(
        min_diameter_deg: f64,
        max_diameter_deg: f64,
        base_config: &Self,
    ) -> Self {
        let target_fov = max_diameter_deg * 3.0;

        // Build-grid cells at ~1/3 of the target FOV.
        let cell_size_deg = target_fov / 3.0;
        let healpix_depth = Self::compute_healpix_depth(cell_size_deg);

        // The pixel-scale range this tier serves: a quad is usefully sized when
        // it spans a moderate fraction of the image width, so a mid-band quad
        // diameter of 35% of a 3000 px frame sets the lower scale bound and 20%
        // of a 1500 px frame the upper. Kept deliberately narrow (~3.5x) so
        // scale hints prune tiers aggressively.
        let quad_diameter_deg = f64::midpoint(min_diameter_deg, max_diameter_deg);
        let quad_diameter_arcsec = quad_diameter_deg * 3600.0;
        let min_scale = quad_diameter_arcsec / (3000.0 * 0.35);
        let max_scale = quad_diameter_arcsec / (1500.0 * 0.20);

        // If a uniform stellar density is requested, derive this tier's
        // `stars_per_cell` from its cell area so every scale carries the same
        // stars/deg^2 (finer tiers -> smaller cells -> fewer per cell). Otherwise keep
        // the flat `stars_per_cell` from the base config.
        #[allow(clippy::cast_sign_loss, reason = "stellar density is non-negative")]
        let stars_per_cell = match base_config.target_density_per_deg2 {
            Some(density) => ((density * cell_area_deg2(healpix_depth)).round() as usize).max(4),
            None => base_config.stars_per_cell,
        };

        Self {
            healpix_depth,
            stars_per_cell,
            min_diameter_deg,
            max_diameter_deg,
            target_fov_deg: target_fov,
            min_scale_arcsec_per_pixel: min_scale,
            max_scale_arcsec_per_pixel: max_scale,
            ..base_config.clone()
        }
    }

    /// Compute appropriate `HEALPix` depth for a given cell size.
    #[allow(clippy::cast_sign_loss, reason = "clamped to [0, 29]")]
    fn compute_healpix_depth(cell_size_deg: f64) -> u8 {
        // HEALPix has 12 x 4^depth cells total
        // Average cell area ~ 4pi / (12 x 4^depth) steradians
        // We want sqrt(area) ~ cell_size_deg

        // Approximate: depth ~ log4(4pi / (12 x cell_size_deg^2))
        let cell_size_rad = cell_size_deg.to_radians();
        let area_sr = cell_size_rad * cell_size_rad;
        let total_area_sr = 4.0 * std::f64::consts::PI;
        let num_cells = total_area_sr / area_sr;
        let depth_float = (num_cells / 12.0).log2() / 2.0;

        // Clamp to valid range [0, 29]
        depth_float.clamp(0.0, 29.0).round() as u8
    }
}

/// A quad in the index with its hash code and metadata.
#[derive(Debug, Clone)]
pub struct IndexedQuad {
    /// The hash code for this quad
    pub hash_code: HashCode,
    /// Indices of the four stars in the catalog
    pub star_indices: [usize; 4],
    /// `HEALPix` cell this quad belongs to
    pub cell_hash: u64,
    /// Quad diameter in degrees
    pub diameter: f64,
}

/// Index builder that constructs quad indices from star catalogs.
#[derive(Debug)]
pub struct IndexBuilder {
    /// Configuration
    config: IndexConfig,
    /// Input catalog of stars (expected already density-capped/uniformized by the
    /// caller, e.g. `TiledBuilder`; the builder does no further thinning).
    catalog: Vec<Star>,
    /// Built quads
    quads: Vec<IndexedQuad>,
}

impl IndexBuilder {
    /// Create a new index builder from a catalog.
    ///
    /// # Errors
    /// [`Error::InsufficientData`] if the catalog is empty.
    pub fn new(catalog: Vec<Star>, config: IndexConfig) -> PlatersResult<Self> {
        if catalog.is_empty() {
            return Err(Error::InsufficientData(
                "Cannot build index from empty catalog".to_string(),
            ));
        }
        Ok(Self {
            config,
            catalog,
            quads: Vec::new(),
        })
    }

    /// Build the index by constructing quads, at a **fixed quad density per sky
    /// area** (`TARGET_QUADS_PER_CELL` per build-grid `HEALPix` cell).
    ///
    /// Quads are enumerated with the **same rule as the field generator**
    /// ([`enumerate_distance_bounded_quads`]) -- for each star, quads of it plus a
    /// 3-subset of its nearby stars within `max_diameter_deg` -- so a field quad and
    /// a catalog quad coincide over the same four stars (the prerequisite for
    /// matching). Each in-band quad is charged to the `HEALPix` cell of its brightest
    /// star; a cell keeps the **brightest-first** quads up to the target, dropping
    /// the rest. This fills sparse sky (build everything available -- where matching
    /// was failing for lack of catalog quads) and caps dense sky (bounding the
    /// index), decoupling coverage from the wildly varying star density.
    pub fn build(&mut self) {
        // Brightest-first (lowest magnitude), carrying the original catalog index.
        let mut order: Vec<usize> = (0..self.catalog.len()).collect();
        order.sort_by(|&a, &b| {
            self.catalog[a]
                .magnitude
                .partial_cmp(&self.catalog[b].magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let uv: Vec<[f64; 3]> = order
            .iter()
            .map(|&i| self.catalog[i].position.to_unit_vector())
            .collect();

        // Neighbour radius = max quad diameter, as a squared chord on the unit
        // sphere; cap neighbours per anchor to bound enumeration.
        let radius2 = chord_sq_for_angle(self.config.max_diameter_deg.to_radians());
        let combos = enumerate_distance_bounded_quads(
            uv.len(),
            |a, b| {
                let d = [
                    uv[a][0] - uv[b][0],
                    uv[a][1] - uv[b][1],
                    uv[a][2] - uv[b][2],
                ];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            },
            radius2,
            CATALOG_MAX_NEIGHBORS,
        );

        // `combos` are sorted ascending by brightness rank, so `c[0]` is the
        // brightest star of the quad and brighter quads come first -- exactly the
        // order the per-cell quota should keep.
        let depth = self.config.healpix_depth;
        let mut per_cell: HashMap<u64, usize> = HashMap::new();
        for c in combos {
            let owner = self.catalog[order[c[0]]].position;
            let cell =
                cdshealpix::nested::hash(depth, owner.ra.to_radians(), owner.dec.to_radians());
            let count = per_cell.entry(cell).or_insert(0);
            if *count >= TARGET_QUADS_PER_CELL {
                continue; // cell quota reached
            }
            let coords = [
                self.catalog[order[c[0]]].position,
                self.catalog[order[c[1]]].position,
                self.catalog[order[c[2]]].position,
                self.catalog[order[c[3]]].position,
            ];
            // Quad diameter (max pairwise separation); keep only in-band quads.
            let mut diameter = 0.0;
            for i in 0..4 {
                for j in (i + 1)..4 {
                    diameter = f64::max(diameter, coords[i].angular_distance(&coords[j]));
                }
            }
            if diameter < self.config.min_diameter_deg || diameter > self.config.max_diameter_deg {
                continue;
            }
            let Ok(hash_code) = compute_hash_code_sky(&coords) else {
                continue; // degenerate quad
            };
            *count += 1;
            self.quads.push(IndexedQuad {
                hash_code,
                star_indices: [order[c[0]], order[c[1]], order[c[2]], order[c[3]]],
                cell_hash: cell,
                diameter,
            });
        }
    }

    /// Get the built quads.
    #[must_use]
    pub fn quads(&self) -> &[IndexedQuad] {
        &self.quads
    }

    /// Get the catalog.
    #[must_use]
    pub fn catalog(&self) -> &[Star] {
        &self.catalog
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &IndexConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_catalog() -> Vec<Star> {
        // Create a small test catalog with stars in a known pattern
        vec![
            Star::new(180.0, 0.0, 10.0),
            Star::new(180.05, 0.0, 10.5),
            Star::new(180.0, 0.05, 11.0),
            Star::new(180.05, 0.05, 11.5),
            Star::new(180.1, 0.0, 12.0),
            Star::new(180.0, 0.1, 12.5),
            Star::new(180.1, 0.1, 13.0),
            Star::new(180.15, 0.05, 13.5),
        ]
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.healpix_depth, 5);
        assert_eq!(config.stars_per_cell, 10);
    }

    #[test]
    fn test_index_builder_creation() {
        let catalog = create_test_catalog();
        let config = IndexConfig::default();
        let builder = IndexBuilder::new(catalog, config);
        assert!(builder.is_ok());
    }

    #[test]
    fn test_index_builder_empty_catalog() {
        let catalog = Vec::new();
        let config = IndexConfig::default();
        let builder = IndexBuilder::new(catalog, config);
        assert!(builder.is_err());
    }

    #[test]
    fn test_index_building() {
        let catalog = create_test_catalog();
        let config = IndexConfig {
            healpix_depth: 3, // Lower depth for small catalog
            stars_per_cell: 8,
            min_diameter_deg: 0.03,
            max_diameter_deg: 0.15,
            ..IndexConfig::default()
        };

        let mut builder = IndexBuilder::new(catalog, config).unwrap();
        builder.build();

        // Should have built some quads
        assert!(!builder.quads().is_empty());
        println!("Built {} quads from test catalog", builder.quads().len());
    }

    #[test]
    fn test_quad_diameter_constraints() {
        let catalog = vec![
            Star::new(180.0, 0.0, 10.0),
            Star::new(180.001, 0.0, 10.0), // Very close
            Star::new(180.002, 0.0, 10.0),
            Star::new(180.003, 0.0, 10.0),
        ];

        let config = IndexConfig {
            healpix_depth: 3,
            stars_per_cell: 4,
            min_diameter_deg: 0.01, // Larger than star separations
            max_diameter_deg: 0.02,
            ..IndexConfig::default()
        };

        let mut builder = IndexBuilder::new(catalog, config).unwrap();
        builder.build();

        // Should not build quads because stars are too close
        assert_eq!(builder.quads().len(), 0);
    }
}
