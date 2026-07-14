//! High-level plate solving query API.
//!
//! This module provides the main entry point for plate solving queries,
//! orchestrating all the components:
//! - Image quad generation
//! - Index loading and filtering
//! - Quad matching
//! - WCS hypothesis generation
//! - Optional iterative refinement for improved accuracy

use crate::{
    catalog_index::CatalogIndex,
    errors::{Error, PlatersResult},
    geometry::{compute_hash_code_pixels_with_permutation, compute_hash_code_sky_with_permutation},
    index::{IndexSet, LoadedIndex, QuadMatch},
    query::{ImageQuad, ImageQuadGenerator, PositionHint, QueryConfig},
    refinement::{IterativeRefiner, RefinementConfig, RefinementResult},
    types::{DetectedField, DetectedStar, PixelCoord, SkyCoord, Star},
    verification::{VerificationConfig, VerificationResult, Verifier},
    wcs::WcsHypothesis,
};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use tracing::{debug, info};

/// Log-odds at which we stop searching: 20.0 ~ e^20 ~ 5x10^8 Bayes factor.
const EARLY_TERMINATION_THRESHOLD: f64 = 20.0;
/// Reject hypotheses implying an unrealistically small pixel scale (arcsec/px).
const MIN_REALISTIC_SCALE: f64 = 0.01;
/// Reject hypotheses implying an unrealistically large pixel scale (arcsec/px).
const MAX_REALISTIC_SCALE: f64 = 20.0;
/// Image quads processed per interleaved match+verify chunk. Balances
/// early-exit latency (smaller = stop sooner) against per-chunk overhead.
const INTERLEAVE_CHUNK: usize = 256;
/// Cap on catalog-quad candidates kept per image quad (closest by hash
/// distance). See the selectivity note in `chunk_candidates`.
const MAX_MATCHES_PER_QUAD: usize = 16;
/// Near-neighbours per anchor for image-side quad generation. The *builder*
/// enumerates each anchor's 12 nearest stars at the tier's own density; the
/// image list carries ~2x that density (the uniformization slack) plus
/// non-catalog contaminants, so covering the same *angular* neighbourhood --
/// and with it the tier's widest quads, whose companions rank ~20th-40th
/// nearest at image density -- needs a proportionally larger count.
const IMAGE_MAX_NEIGHBORS: usize = 30;

/// A candidate image-quad <-> catalog-quad correspondence awaiting verification.
#[derive(Debug, Clone)]
struct CandidateMatch {
    image_coords: [PixelCoord; 4],
    catalog_coords: [SkyCoord; 4],
}

/// The result of the interleaved match+verify search ([`PlateSolver::search_chunked`]):
/// the best verified hypothesis plus the counts needed to assemble the `SolveResult`.
struct SearchOutcome {
    num_hypotheses_tested: usize,
    num_quads_matched: usize,
    best_log_odds: f64,
    best_result: Option<VerificationResult>,
}

/// Result of a plate solving query.
#[non_exhaustive]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SolveResult {
    /// Best WCS solution found (after refinement if enabled)
    pub wcs: WcsHypothesis,

    /// Verification result for the best solution
    pub verification: VerificationResult,

    /// Number of hypotheses tested
    pub num_hypotheses_tested: usize,

    /// Number of quads matched
    pub num_quads_matched: usize,

    /// Number of image quads generated
    pub num_image_quads: usize,

    /// Paths of the index files used for matching
    pub indices_used: Vec<PathBuf>,

    /// Whether a solution was found
    pub solved: bool,

    /// Refinement result (if refinement was performed)
    pub refinement: Option<RefinementResult>,
}

/// Statistics about query performance.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct QueryStats {
    /// Number of image quads generated
    pub num_image_quads: usize,

    /// Number of catalog quads checked
    pub num_catalog_quads_checked: usize,

    /// Number of indices searched
    pub num_indices_searched: usize,

    /// Number of indices skipped (due to scale filtering)
    pub num_indices_skipped: usize,

    /// Number of WCS hypotheses generated
    pub num_hypotheses: usize,
}

/// High-level plate solver.
#[derive(Debug)]
pub struct PlateSolver {
    /// Set of indices to query
    index_set: IndexSet,

    /// Query configuration
    config: QueryConfig,

    /// Verification configuration
    verification_config: VerificationConfig,

    /// If set, the solver loads *tiled* index files lazily from this directory on
    /// each solve -- only the tiles under the field (by scale + position cone) are
    /// read into memory. This is how an all-sky tiled index stays affordable: the
    /// directory may be enormous, but a solve touches only a few tiles. When
    /// `None`, the pre-loaded `index_set` is used as-is.
    tile_dir: Option<PathBuf>,
}

impl PlateSolver {
    /// Create a new plate solver with the given index set and configuration.
    #[must_use]
    pub fn new(index_set: IndexSet, config: QueryConfig) -> Self {
        Self {
            index_set,
            config,
            verification_config: VerificationConfig::default(),
            tile_dir: None,
        }
    }

    /// Create a new plate solver with custom verification config.
    #[must_use]
    pub fn with_verification(
        index_set: IndexSet,
        config: QueryConfig,
        verification_config: VerificationConfig,
    ) -> Self {
        Self {
            index_set,
            config,
            verification_config,
            tile_dir: None,
        }
    }

    /// Create a solver over a **tiled** index directory: each solve lazily loads
    /// only the tile files under its field (see the `tile_dir` field). Requires a
    /// position hint in the query config -- that's what selects the tiles.
    #[must_use]
    pub fn from_tile_directory(
        tile_dir: PathBuf,
        config: QueryConfig,
        verification_config: VerificationConfig,
    ) -> Self {
        Self {
            index_set: IndexSet::new(),
            config,
            verification_config,
            tile_dir: Some(tile_dir),
        }
    }

    /// Blind solve: find the field with **no position hint**, by scanning the
    /// tiled index's files in batches and running a coarse solve over each batch,
    /// stopping at the first batch that yields a confident pose. Each tile is read
    /// exactly once (we iterate the files, not overlapping sky cones), so this is
    /// a single linear pass over the index with early exit. Correct because the
    /// quad *match* localizes the field -- a batch without the field yields no
    /// passing hypothesis.
    ///
    /// On a hit, a final hinted [`solve`](Self::solve) is run at the located
    /// centre to load the full neighbourhood and refine; a spurious hit (which
    /// then fails to re-solve) is skipped and the scan continues.
    ///
    /// **For fast blind, point this at a directory of merged `allsky_...` per-scale
    /// indices** (see `platers_build::merge_scale_indices`): one balanced tree per
    /// scale means a handful of files instead of thousands of tiles, so the scan
    /// is one batch rather than a sky-wide sweep -- dramatically fewer tree
    /// descents. Pointing at the fine per-tile directory still works
    /// (RAM-bounded, each tile read once) but is much slower when the field
    /// isn't near the front of the scan order.
    ///
    /// Requires a tiled directory ([`from_tile_directory`](Self::from_tile_directory));
    /// a **scale hint is strongly recommended**. Any position hint is ignored.
    ///
    /// # Errors
    /// Returns an error if not in tiled mode, or if no batch yields a confident
    /// solution (the field isn't covered by the index, or is unsolvable).
    pub fn solve_blind(&self, field: &DetectedField) -> PlatersResult<SolveResult> {
        /// Tiles loaded (in parallel) and searched together per scan step.
        const BLIND_BATCH: usize = 128;

        let Some(dir) = self.tile_dir.clone() else {
            return Err(Error::ValueError(
                "blind solving requires a tiled index directory (PlateSolver::from_tile_directory)"
                    .to_string(),
            ));
        };
        let scale_range = self
            .config
            .scale_hint
            .as_ref()
            .map(|s| (s.min_arcsec_per_pixel, s.max_arcsec_per_pixel));

        let paths = crate::index::matching_tile_paths(&dir, field.width, scale_range)?;
        if paths.is_empty() {
            return Err(Error::NoSolution(
                "blind solve: no index tiles match the image scale".to_string(),
            ));
        }
        info!(
            "Blind: scanning {} tiles in batches of {BLIND_BATCH}",
            paths.len()
        );

        for batch in paths.chunks(BLIND_BATCH) {
            // Load this batch of tiles in parallel (deserialization dominates the
            // scan; a corrupt tile is skipped rather than aborting the blind solve).
            let loaded: Vec<LoadedIndex> = batch
                .par_iter()
                .filter_map(|path| LoadedIndex::open(path).ok())
                .collect();
            let mut set = IndexSet::new();
            for idx in loaded {
                set.add(idx);
            }
            // Scout: coarse solve over just this batch, no position hint.
            let mut config = self.config.clone();
            config.position_hint = None;
            let scout = Self {
                index_set: set,
                config,
                verification_config: self.verification_config.clone(),
                tile_dir: None,
            };
            let Ok(coarse) = scout.solve_coarse(field) else {
                continue;
            };

            // Confident hit: re-solve hinted at the located centre to pull in the
            // full neighbourhood and refine. If that fails, the hit was spurious;
            // keep scanning.
            let c = coarse.wcs.center;
            info!("Blind: candidate match near ({:.3}, {:.3})", c.ra, c.dec);
            let mut hinted = self.config.clone();
            hinted.position_hint = Some(PositionHint {
                ra: c.ra,
                dec: c.dec,
                radius: 1.0,
            });
            let solver = Self {
                index_set: IndexSet::new(),
                config: hinted,
                verification_config: self.verification_config.clone(),
                tile_dir: Some(dir.clone()),
            };
            if let Ok(result) = solver.solve(field) {
                return Ok(result);
            }
        }
        Err(Error::NoSolution(
            "blind solve: no confident match found across the index".to_string(),
        ))
    }

    /// Coarse solve only (Stage 1): quad matching + verification, returning the
    /// best hypothesis with its WCS re-anchored to the image center.
    ///
    /// This is the fast "good first guess" -- accurate scale/rotation and an
    /// image-centered field position, but no multi-star refinement. Most callers
    /// want [`solve`](Self::solve), which runs this and then refines. Use
    /// `solve_coarse` when you specifically want the unrefined pose (e.g. to
    /// measure coarse accuracy, or as a seed for custom refinement).
    ///
    /// Steps:
    /// 1. Generate quads from detected stars
    /// 2. Filter indices by scale (if scale hint provided)
    /// 3. Match image quads to catalog quads
    /// 4. Generate + verify WCS hypotheses, keep the best
    ///
    /// # Errors
    /// Returns an error if no solution can be found or if there are I/O errors.
    pub fn solve_coarse(&self, field: &DetectedField) -> PlatersResult<SolveResult> {
        // Stage 1 -- index selection. Derive the scale/cone filters, load the index
        // files they select, and pick the sub-indices to search. `owner` keeps any
        // lazily-loaded tile set alive for the rest of this solve; `indices_to_search`
        // borrows from it (or from the pre-loaded set), so both stay in this scope.
        let (scale_range, cone) = self.compute_filters(field);
        let owner = self.load_index_owner(field.width, scale_range, cone)?;
        let index_set: &IndexSet = owner.as_ref().unwrap_or(&self.index_set);
        let indices_to_search: Vec<&LoadedIndex> =
            index_set.select_for(field.width, scale_range, cone);

        info!(
            "Searching {} indices (of {} total)",
            indices_to_search.len(),
            self.index_set.len()
        );

        if indices_to_search.is_empty() {
            return Err(Error::NoSolution(
                "No indices match the scale range".to_string(),
            ));
        }

        // Stage 2 -- uniformize the field and generate image quads, pooling over
        // every selected scale tier's grid (see `generate_image_quads`).
        let image_quads = self.generate_image_quads(field, &indices_to_search)?;

        // Stage 4 -- build the catalog spatial index once for verification.
        let catalog_index = self.build_catalog_index(&indices_to_search);

        // Stage 5 -- interleaved match + verify with early termination, in two
        // passes. Pass 1 is breadth-first: only each image quad's closest-by-hash
        // candidate, so the hypothesis budget reaches EVERY generated quad even
        // when a dense hash region floods the tolerance ball (the true match is
        // almost always the closest candidate). Pass 2 (only if pass 1 found
        // nothing) descends into the full tolerance ball, depth-first. The
        // `max_hypotheses` ceiling is per SOLVE, not per pass: pass 2 gets only
        // whatever pass 1 left unspent.
        let mut outcome = self.search_chunked(
            &image_quads,
            &indices_to_search,
            &catalog_index,
            field,
            cone,
            true,
            self.config.max_hypotheses,
        );
        if !outcome
            .best_result
            .as_ref()
            .is_some_and(|r| r.passes_threshold)
        {
            let deep = self.search_chunked(
                &image_quads,
                &indices_to_search,
                &catalog_index,
                field,
                cone,
                false,
                self.config
                    .max_hypotheses
                    .saturating_sub(outcome.num_hypotheses_tested),
            );
            outcome = SearchOutcome {
                num_hypotheses_tested: outcome.num_hypotheses_tested + deep.num_hypotheses_tested,
                num_quads_matched: outcome.num_quads_matched.max(deep.num_quads_matched),
                best_log_odds: outcome.best_log_odds.max(deep.best_log_odds),
                best_result: match (outcome.best_result, deep.best_result) {
                    (Some(a), Some(b)) => Some(if a.log_odds >= b.log_odds { a } else { b }),
                    (a, b) => a.or(b),
                },
            };
        }

        // Stage 6 -- assemble the coarse result (re-anchored to the image center).
        Self::assemble_coarse_result(outcome, &indices_to_search, image_quads.len())
    }

    /// Stage 1a (see [`solve_coarse`](Self::solve_coarse)): derive the scale-range
    /// and sky-cone filters from the query hints. The cone covers the hint
    /// uncertainty plus the field's own angular half-diagonal (so every in-field
    /// star is reachable) and prunes *tiled* index files to the few tiles the field
    /// can touch; for monolithic all-sky indices it's a no-op.
    fn compute_filters(
        &self,
        field: &DetectedField,
    ) -> (Option<(f64, f64)>, Option<(SkyCoord, f64)>) {
        let scale_range = self
            .config
            .scale_hint
            .as_ref()
            .map(|s| (s.min_arcsec_per_pixel, s.max_arcsec_per_pixel));
        let cone = self.config.position_hint.as_ref().map(|hint| {
            // Field size needs a pixel scale; use the scale hint midpoint, or a
            // conservative default when blind in scale.
            let scale_arcsec = scale_range.map_or(2.0, |(lo, hi)| f64::midpoint(lo, hi));
            let field_half_diag_deg = 0.5 * field.diagonal_px() * scale_arcsec / 3600.0;
            (
                SkyCoord::new_normalized(hint.ra, hint.dec),
                hint.radius + field_half_diag_deg + 0.1,
            )
        });

        if let Some((lo, hi)) = scale_range {
            info!("Using scale hint: {lo:.4} - {hi:.4} arcsec/pixel");
        }
        if let Some((c, r)) = cone {
            info!(
                "Using position cone: {r:.3} deg around ({:.4}, {:.4})",
                c.ra, c.dec
            );
        }
        (scale_range, cone)
    }

    /// Stage 1b (see [`solve_coarse`](Self::solve_coarse)): in tiled mode, lazily
    /// load only the tile files under the field and return them (the caller keeps
    /// them alive); in pre-loaded mode return `None` so the caller uses
    /// `self.index_set`.
    fn load_index_owner(
        &self,
        image_width: usize,
        scale_range: Option<(f64, f64)>,
        cone: Option<(SkyCoord, f64)>,
    ) -> PlatersResult<Option<IndexSet>> {
        let Some(dir) = &self.tile_dir else {
            return Ok(None);
        };
        let set = IndexSet::load_tiles_for(dir, image_width, scale_range, cone)?;
        info!(
            "Lazy-loaded {} tile file(s) from {}",
            set.len(),
            dir.display()
        );
        Ok(Some(set))
    }

    /// Stage 2+3 (see [`solve_coarse`](Self::solve_coarse)): uniformize the field
    /// and generate image quads.
    ///
    /// With a scale hint, the field is uniformized to brightest-per-HEALPix-cell to
    /// mirror how the index selected stars. A single grid can't serve every case:
    /// a *coarse* grid keeps only the few bright, widely-spaced stars (right for a
    /// deep survey frame, whose fainter detections aren't in the catalog), while a
    /// *fine* grid keeps many stars (right for a dense field that needs the
    /// redundancy). The selected index spans several scale tiers, each with its own
    /// grid -- so we uniformize and generate at **every** distinct tier grid and pool
    /// the quads. The pool then covers all scales, and -- critically -- depends only
    /// on *which* tiers are present, not on their order, so a tiled directory and a
    /// merged all-sky set produce identical quads. Without a scale hint, falls back
    /// to one global pass.
    fn generate_image_quads(
        &self,
        field: &DetectedField,
        indices_to_search: &[&LoadedIndex],
    ) -> PlatersResult<Vec<ImageQuad>> {
        // Adaptive quad budget. Generation order is brightness-local, so the rank
        // of the *true* quad grows with how many stars compete to anchor quads: a
        // sparse field surfaces it within a few thousand draws, while a rich
        // deep-survey frame can bury it past 100k. Scale the budget with the
        // participating star count, with a floor so tiny fields still get enough
        // draws, and clamp to `max_quads_to_try` (the ceiling) so a dense field
        // can't blow up runtime.
        const QUADS_PER_STAR: usize = 1500;
        const MIN_QUAD_BUDGET: usize = 50_000;
        let ceiling = self.config.max_quads_to_try;
        let budget =
            (field.stars.len() * QUADS_PER_STAR).clamp(MIN_QUAD_BUDGET.min(ceiling), ceiling);
        let mut image_quads = Vec::new();

        let Some(ref scale_range) = self.config.scale_hint else {
            // No scale: one global pass, truncated to `max_stars_for_quads`.
            let mut quad_gen =
                ImageQuadGenerator::new(field.stars.clone(), self.config.max_stars_for_quads);
            while image_quads.len() < budget {
                match quad_gen.next_quad() {
                    Some(q) => image_quads.push(q),
                    None => break,
                }
            }
            if image_quads.is_empty() {
                return Err(Error::InsufficientData(
                    "No valid quads could be generated from detected stars".to_string(),
                ));
            }
            return Ok(image_quads);
        };

        let pixel_scale = f64::midpoint(
            scale_range.min_arcsec_per_pixel,
            scale_range.max_arcsec_per_pixel,
        );

        // Distinct tier grids (depth, stars_per_cell) with each grid's own
        // widest quad diameter, order-independent. The generation radius must be
        // TIER-LOCAL, mirroring the builder: with one global (widest-tier)
        // radius, the brightness-capped neighbour list of every anchor fills
        // with far-but-bright stars and a fine tier's genuinely local quads
        // become unreachable.
        let mut grids: Vec<(u8, usize, f64)> = Vec::new();
        for index in indices_to_search {
            let key = (index.config.healpix_depth, index.config.stars_per_cell);
            let diam = index.diameter_range.1;
            match grids.iter_mut().find(|(d, s, _)| (*d, *s) == key) {
                Some((_, _, max_diam)) => *max_diam = max_diam.max(diam),
                None => grids.push((key.0, key.1, diam)),
            }
        }
        grids.sort_unstable_by_key(|&(d, s, _)| (d, s));
        if !grids.iter().any(|&(_, _, diam)| diam > 0.0) {
            return Err(Error::InsufficientData(
                "No valid quads could be generated from detected stars".to_string(),
            ));
        }
        let per_grid = (budget / grids.len().max(1)).max(1);

        for &(depth, stars_per_cell, max_diam_deg) in &grids {
            let radius_px = (max_diam_deg * 3600.0 / pixel_scale) * 1.2;
            if radius_px <= 0.0 {
                continue;
            }
            let cell_arcsec = crate::spatial::HealPixGrid::new(depth)
                .map_or(0.0, |g| g.cell_size_degrees())
                * 3600.0;
            // Keep 2x the index's per-cell count: the image's brightness ordering
            // is in the *detector's* band, the index's in catalog G, and in a
            // crowded cell (many stars within the band-difference scatter) the
            // per-cell winners diverge -- the index's choices must survive the
            // reshuffle or no shared quad exists. High-airmass survey frames at
            // low galactic latitude are the worst case (extinction reorders by
            // up to a magnitude).
            let keep_per_cell = stars_per_cell * 2;
            let uniform = if cell_arcsec > 0.0 {
                crate::query::uniformize_field(
                    &field.stars,
                    field.width,
                    field.height,
                    cell_arcsec,
                    pixel_scale,
                    keep_per_cell,
                )
            } else {
                field.stars.clone()
            };
            let mut quad_gen = ImageQuadGenerator::new_local(
                uniform.clone(),
                uniform.len(),
                radius_px,
                IMAGE_MAX_NEIGHBORS,
            );
            let mut from_grid = 0;
            while from_grid < per_grid && image_quads.len() < budget {
                match quad_gen.next_quad() {
                    Some(q) => {
                        image_quads.push(q);
                        from_grid += 1;
                    }
                    None => break,
                }
            }
        }

        if image_quads.is_empty() {
            return Err(Error::InsufficientData(
                "No valid quads could be generated from detected stars".to_string(),
            ));
        }
        info!(
            "Generated {} image quads (pooled over {} tier grid(s), budget {} for {} stars)",
            image_quads.len(),
            grids.len(),
            budget,
            field.stars.len()
        );
        Ok(image_quads)
    }

    /// Stage 4 (see [`solve_coarse`](Self::solve_coarse)): build the catalog spatial
    /// index over all selected sub-indices once, so verification does O(log n)
    /// nearest-neighbor lookups instead of a per-hypothesis linear scan. When a
    /// position hint is given, bound the index to that region up front so the tree
    /// (and queries) stay small.
    fn build_catalog_index(&self, indices_to_search: &[&LoadedIndex]) -> CatalogIndex {
        let catalog_index = if let Some(ref hint) = self.config.position_hint {
            let full = CatalogIndex::from_indices(indices_to_search);
            // Pad the hint radius so stars near the image edges are still included
            // for any plausible pointing within the hint.
            let nearby = full.stars_near(
                SkyCoord::new_normalized(hint.ra, hint.dec),
                hint.radius + 1.0,
            );
            CatalogIndex::new(nearby)
        } else {
            CatalogIndex::from_indices(indices_to_search)
        };
        info!("Catalog index built with {} stars", catalog_index.len());
        catalog_index
    }

    /// Stage 5 (see [`solve_coarse`](Self::solve_coarse)): interleaved match +
    /// verify with early termination. Rather than match ALL image quads into one big
    /// candidate list and only then verify (making a hard field pay the full
    /// match+canonicalize cost before any early-out), process image quads in chunks:
    /// build that chunk's candidates ([`chunk_candidates`](Self::chunk_candidates)),
    /// verify them in parallel, and stop as soon as a confident solution appears. An
    /// easy field exits after the first chunk; a hard field still scans everything.
    /// Mirrors astrometry.net's `logratio_stoplooking`.
    ///
    /// With `best_only`, each image quad contributes only its closest-by-hash
    /// candidate (the breadth-first pass -- see
    /// [`solve_coarse`](Self::solve_coarse)). `budget` caps the number of
    /// hypotheses this call may verify (the caller splits `max_hypotheses`
    /// across the two passes).
    ///
    /// The result is DETERMINISTIC despite the parallel verification: the
    /// accepted hypothesis is the first (in candidate order) to clear the
    /// early-termination threshold, not whichever thread happens to win a race,
    /// and the reported hypothesis count is the serial-equivalent count up to
    /// that winner. Same input, same output -- run to run.
    #[expect(
        clippy::too_many_arguments,
        reason = "internal stage function; the arguments are the solve pipeline's state"
    )]
    fn search_chunked(
        &self,
        image_quads: &[ImageQuad],
        indices_to_search: &[&LoadedIndex],
        catalog_index: &CatalogIndex,
        field: &DetectedField,
        cone: Option<(SkyCoord, f64)>,
        best_only: bool,
        budget: usize,
    ) -> SearchOutcome {
        let mut outcome = SearchOutcome {
            num_hypotheses_tested: 0,
            num_quads_matched: 0,
            best_log_odds: f64::NEG_INFINITY,
            best_result: None,
        };
        if budget == 0 {
            return outcome;
        }
        info!(
            "Matching {} image quads against {} indices (interleaved, {})...",
            image_quads.len(),
            indices_to_search.len(),
            if best_only {
                "closest candidates"
            } else {
                "full tolerance ball"
            }
        );

        let verifier = Verifier::new(self.verification_config.clone());
        let num_tested = AtomicUsize::new(0);
        // Best-so-far as (log_odds, candidate ordinal, result). Ties in log-odds
        // break toward the earlier candidate, so the kept result is a function of
        // the candidate order alone, never of which thread locked first.
        let best_slot: Mutex<(f64, usize, Option<VerificationResult>)> =
            Mutex::new((f64::NEG_INFINITY, usize::MAX, None));

        for chunk in image_quads.chunks(INTERLEAVE_CHUNK) {
            // Build this chunk's candidate correspondences (match + canonical
            // ordering), bounding the up-front match cost to the chunk.
            let candidates = self.chunk_candidates(chunk, indices_to_search, cone, best_only);
            if candidates.is_empty() {
                continue;
            }
            let base = outcome.num_quads_matched;
            outcome.num_quads_matched += candidates.len();

            // Respect the hypothesis budget across chunks.
            let remaining = budget.saturating_sub(num_tested.load(Ordering::Relaxed));
            if remaining == 0 {
                break;
            }
            let to_test = candidates.len().min(remaining);

            // Verify this chunk's candidates in parallel. Early exit is keyed to
            // the smallest candidate index that clears the early-termination
            // threshold: threads skip only candidates AFTER the current winner,
            // so every candidate before the final winner is always verified and
            // the winner is exactly what a serial scan would accept. (No earlier
            // candidate can out-score it: scoring above the threshold at a
            // smaller index would have claimed the winner slot itself.)
            let winner = AtomicUsize::new(usize::MAX);
            let tested_at_chunk_start = num_tested.load(Ordering::Relaxed);
            candidates[..to_test]
                .par_iter()
                .enumerate()
                .for_each(|(i, candidate)| {
                    if i > winner.load(Ordering::Relaxed) {
                        return;
                    }

                    let Some(wcs) = self.hypothesis_from(candidate, field) else {
                        return;
                    };
                    let result = verifier.verify(&wcs, &field.stars, catalog_index);
                    let tested_count = num_tested.fetch_add(1, Ordering::Relaxed) + 1;

                    {
                        let mut best = best_slot
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if result.log_odds > best.0 || (result.log_odds == best.0 && base + i < best.1)
                        {
                            debug!(
                                "New best #{}: log_odds={:.2}, matches={}, center=({:.3}, {:.3}), scale={:.4}",
                                tested_count,
                                result.log_odds,
                                result.num_matches,
                                wcs.center.ra,
                                wcs.center.dec,
                                wcs.scale_arcsec_per_pixel()
                            );
                            *best = (result.log_odds, base + i, Some(result.clone()));
                        }
                    }

                    if result.passes_threshold && result.log_odds > EARLY_TERMINATION_THRESHOLD {
                        let prev = winner.fetch_min(i, Ordering::SeqCst);
                        if i < prev {
                            info!(
                                "Found confident solution! log-odds={:.1}",
                                result.log_odds
                            );
                        }
                    }
                });

            let w = winner.load(Ordering::Relaxed);
            if w != usize::MAX {
                // Threads in flight past the winner also verified and counted
                // themselves; rebuild the serial-equivalent state so the output
                // does not depend on scheduling. The count is the candidates up
                // to the winner that reached the verifier, and the kept result
                // is the winner's own (re-verified -- cheap, once per solve; an
                // in-flight later candidate may have posted a higher score that
                // a serial scan would never have tested).
                let serial_tested = candidates[..=w]
                    .iter()
                    .filter(|c| self.hypothesis_from(c, field).is_some())
                    .count();
                num_tested.store(tested_at_chunk_start + serial_tested, Ordering::Relaxed);
                if let Some(wcs) = self.hypothesis_from(&candidates[w], field) {
                    let result = verifier.verify(&wcs, &field.stars, catalog_index);
                    let mut best = best_slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *best = (result.log_odds, base + w, Some(result));
                }
                break;
            }
        }

        let (best_log_odds, _, best_result) = best_slot
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        outcome.num_hypotheses_tested = num_tested.into_inner();
        outcome.best_log_odds = best_log_odds;
        outcome.best_result = best_result;
        outcome
    }

    /// The cheap pre-verification gates: fit a WCS from the candidate
    /// correspondence and require a physically plausible (and hint-consistent)
    /// pixel scale. `None` means the candidate never reaches the verifier.
    fn hypothesis_from(
        &self,
        candidate: &CandidateMatch,
        field: &DetectedField,
    ) -> Option<WcsHypothesis> {
        let wcs = WcsHypothesis::from_quad_match(
            &candidate.image_coords,
            &candidate.catalog_coords,
            field.width,
            field.height,
        )
        .ok()?;
        let (scale_lo, scale_hi) = self.scale_bounds();
        let scale = wcs.scale_arcsec_per_pixel();
        if scale < scale_lo || scale > scale_hi {
            return None;
        }
        Some(wcs)
    }

    /// The acceptable pixel-scale range for a hypothesis: the scale hint clamped
    /// to the physically realistic bounds, or the realistic bounds alone when
    /// solving blind in scale.
    fn scale_bounds(&self) -> (f64, f64) {
        self.config
            .scale_hint
            .as_ref()
            .map_or((MIN_REALISTIC_SCALE, MAX_REALISTIC_SCALE), |r| {
                (
                    r.min_arcsec_per_pixel.max(MIN_REALISTIC_SCALE),
                    r.max_arcsec_per_pixel.min(MAX_REALISTIC_SCALE),
                )
            })
    }

    /// Match one chunk of image quads against the selected indices and return the
    /// candidate correspondences in canonical star order.
    ///
    /// Each quad is tried at BOTH parities. The geometric hash encodes the
    /// handedness of the quad (the sign of C/D perpendicular to the A-B axis), so
    /// a mirror-flipped field -- a North-down / negative-vs-positive-parity
    /// detector, e.g. ZTF vs LCO -- hashes to the *reflection* of the catalog code
    /// and never matches. The mirror variant negates one pixel axis; either way
    /// the candidate is built from the *real* pixel positions, so
    /// `from_quad_match`'s free fit recovers the true pose at whichever parity.
    fn chunk_candidates(
        &self,
        chunk: &[ImageQuad],
        indices_to_search: &[&LoadedIndex],
        cone: Option<(SkyCoord, f64)>,
        best_only: bool,
    ) -> Vec<CandidateMatch> {
        // Quads are independent, and the hash-tree searches dominate a hard
        // field's wall time, so match them in parallel. `flat_map_iter` keeps
        // the chunk's quad order, so the candidate stream (and therefore the
        // budget's reach) is identical to the serial version.
        chunk
            .par_iter()
            .flat_map_iter(|image_quad| {
                let mut candidates: Vec<CandidateMatch> = Vec::new();
                let image_pixels = DetectedStar::array_to_pixel_coords(&image_quad.stars);
                let mirrored_pixels: [PixelCoord; 4] =
                    image_pixels.map(|p| PixelCoord { x: -p.x, y: p.y });

                for quad_pixels in [&image_pixels, &mirrored_pixels] {
                    let Ok((hash, perm)) = compute_hash_code_pixels_with_permutation(quad_pixels)
                    else {
                        continue; // degenerate image quad
                    };

                    // Gather this quad-parity's tolerance-ball matches across
                    // every index, dropping matches outside the position cone.
                    // The cone is the same padded one used to select tiles
                    // (hint radius + the field's half-diagonal), so a quad
                    // anchored near the frame edge of an off-center field is
                    // never rejected here after the tile filter deliberately
                    // included its region.
                    let mut matches: Vec<QuadMatch> = Vec::new();
                    for index in indices_to_search {
                        matches.extend(
                            index
                                .find_matching_quads(&hash, self.config.hash_code_tolerance)
                                .into_iter()
                                .filter(|m| {
                                    let Some((hint_center, radius)) = &cone else {
                                        return true;
                                    };
                                    let catalog_sky = m.catalog_stars.map(|s| s.position);
                                    SkyCoord::centroid_of_four(&catalog_sky)
                                        .angular_distance(hint_center)
                                        <= *radius
                                }),
                        );
                    }

                    if best_only {
                        // Breadth-first pass: this quad-parity contributes only
                        // its single closest-by-hash candidate.
                        if let Some(m) = matches
                            .iter()
                            .min_by(|a, b| a.hash_distance.total_cmp(&b.hash_distance))
                        {
                            if let Some(c) = Self::candidate_from(m, &image_pixels, perm) {
                                candidates.push(c);
                            }
                        }
                        continue;
                    }

                    // An unselective quad -- one whose tolerance ball holds
                    // hundreds of catalog quads -- carries almost no
                    // information, yet verifying its whole ball exhausts the
                    // hypothesis budget before selective quads deeper in the
                    // enumeration are ever reached. The true match sits at
                    // noise-scale hash distance, so keep only the closest few:
                    // a flat cap over ALL searched indices, so the per-quad
                    // budget does not multiply with the tile/tier count.
                    if matches.len() > MAX_MATCHES_PER_QUAD {
                        matches.sort_by(|a, b| a.hash_distance.total_cmp(&b.hash_distance));
                        matches.truncate(MAX_MATCHES_PER_QUAD);
                    }
                    candidates.extend(
                        matches
                            .iter()
                            .filter_map(|m| Self::candidate_from(m, &image_pixels, perm)),
                    );
                }
                candidates
            })
            .collect()
    }

    /// Pair a catalog quad match with the real image pixels, both in canonical
    /// star order. `perm` indexes the four stars (identity is unchanged by the
    /// mirror). `None` if the catalog quad is degenerate.
    fn candidate_from(
        quad_match: &QuadMatch,
        image_pixels: &[PixelCoord; 4],
        perm: [usize; 4],
    ) -> Option<CandidateMatch> {
        let catalog_sky = quad_match.catalog_stars.map(|s| s.position);
        let (_cat_hash, cat_perm) = compute_hash_code_sky_with_permutation(&catalog_sky).ok()?;
        Some(CandidateMatch {
            image_coords: [
                image_pixels[perm[0]],
                image_pixels[perm[1]],
                image_pixels[perm[2]],
                image_pixels[perm[3]],
            ],
            catalog_coords: [
                catalog_sky[cat_perm[0]],
                catalog_sky[cat_perm[1]],
                catalog_sky[cat_perm[2]],
                catalog_sky[cat_perm[3]],
            ],
        })
    }

    /// Stage 6 (see [`solve_coarse`](Self::solve_coarse)): turn the search outcome
    /// into a `SolveResult` -- error if nothing matched or nothing passed the
    /// threshold, then re-anchor the accepted pose so its reported center is the
    /// *image* center rather than the quad centroid (falls back to the
    /// quad-anchored WCS if the re-anchor fails).
    fn assemble_coarse_result(
        outcome: SearchOutcome,
        indices_to_search: &[&LoadedIndex],
        num_image_quads: usize,
    ) -> PlatersResult<SolveResult> {
        let SearchOutcome {
            num_hypotheses_tested,
            num_quads_matched,
            best_log_odds,
            best_result,
        } = outcome;

        if num_quads_matched == 0 {
            return Err(Error::NoSolution(
                "No matching quads found in any index".to_string(),
            ));
        }
        info!(
            "Tested {} hypotheses from {} quad matches",
            num_hypotheses_tested, num_quads_matched
        );

        let best_result = best_result
            .ok_or_else(|| Error::NoSolution("No hypotheses were generated".to_string()))?;
        if !best_result.passes_threshold {
            return Err(Error::NoSolution(format!(
                "No hypothesis passed verification threshold (best log-odds: {best_log_odds:.2})"
            )));
        }

        let coarse_wcs = best_result
            .wcs
            .re_anchored_to_image_center()
            .unwrap_or_else(|_| best_result.wcs.clone());

        info!(
            "Solution found: log-odds={:.1}, matches={}, center=({:.3}, {:.3})",
            best_result.log_odds,
            best_result.num_matches,
            coarse_wcs.center.ra,
            coarse_wcs.center.dec
        );

        Ok(SolveResult {
            wcs: coarse_wcs,
            verification: best_result.clone(),
            num_hypotheses_tested,
            num_quads_matched,
            num_image_quads,
            indices_used: indices_to_search
                .iter()
                .map(|idx| idx.path.clone())
                .collect(),
            solved: true,
            refinement: None,
        })
    }

    /// Solve for WCS -- the main entry point (Stage 1 + Stage 2).
    ///
    /// Runs a coarse solve ([`solve_coarse`](Self::solve_coarse)) and then
    /// iterative multi-star refinement, returning the refined WCS. If refinement
    /// cannot run (too few matched stars, etc.) the accurate-enough coarse result
    /// is returned instead, so a solvable field always yields a solution.
    ///
    /// For the unrefined coarse pose, use [`solve_coarse`](Self::solve_coarse);
    /// to control the refinement parameters, use
    /// [`solve_with_refinement`](Self::solve_with_refinement).
    ///
    /// # Errors
    /// Returns an error only if no coarse solution can be found.
    pub fn solve(&self, field: &DetectedField) -> PlatersResult<SolveResult> {
        self.solve_with_refinement(field, None)
    }

    /// Solve, then refine with an explicit refinement configuration.
    ///
    /// Same as [`solve`](Self::solve) but lets the caller supply a
    /// [`RefinementConfig`] (`None` = default). Falls back to the coarse result
    /// if refinement cannot run.
    ///
    /// # Errors
    /// Returns an error only if no coarse solution can be found.
    pub fn solve_with_refinement(
        &self,
        field: &DetectedField,
        refinement_config: Option<RefinementConfig>,
    ) -> PlatersResult<SolveResult> {
        self.solve_with_refinement_against(field, refinement_config, None)
    }

    /// Solve, then refine against an external dense catalog instead of the index's
    /// own embedded stars.
    ///
    /// The on-disk index stores only the *uniformized* (brightest-per-cell) subset
    /// of the catalog, which is all that quad matching needs. The final least-squares
    /// / SIP fit, however, is better conditioned with a denser star list around the
    /// solved center -- more matched stars, lower residuals. A caller that keeps the
    /// full catalog resident (e.g. the server) passes it here as a [`CatalogIndex`].
    ///
    /// `external_catalog = None` reproduces [`solve_with_refinement`](Self::solve_with_refinement) exactly
    /// (refine against the index's own stars), so existing callers are unaffected.
    ///
    /// # Errors
    /// Returns an error only if no coarse solution can be found.
    pub fn solve_with_refinement_against(
        &self,
        field: &DetectedField,
        refinement_config: Option<RefinementConfig>,
        external_catalog: Option<&CatalogIndex>,
    ) -> PlatersResult<SolveResult> {
        let coarse = self.solve_coarse(field)?;
        Ok(self.apply_refinement(coarse, &field.stars, refinement_config, external_catalog))
    }

    /// Refine a coarse solve result in place. On any refinement failure the
    /// coarse `result` is returned unchanged (refinement is best-effort polish).
    fn apply_refinement(
        &self,
        mut result: SolveResult,
        detected_stars: &[DetectedStar],
        refinement_config: Option<RefinementConfig>,
        external_catalog: Option<&CatalogIndex>,
    ) -> SolveResult {
        debug!(
            "Refining: initial center=({:.6}, {:.6}), scale={:.4} arcsec/px",
            result.wcs.center.ra,
            result.wcs.center.dec,
            result.wcs.scale_arcsec_per_pixel()
        );

        // Spatial index over the catalog for refinement matching, bounded to the
        // solved field. Pad by the image half-diagonal so edge stars are included.
        let half_diag_deg = 0.5
            * (result.wcs.image_width as f64).hypot(result.wcs.image_height as f64)
            * result.wcs.scale_arcsec_per_pixel()
            / 3600.0;
        let cone_radius = half_diag_deg + 0.1;

        // Source the refinement stars. Prefer a dense external catalog when the
        // caller supplies one (more stars around the center -> a better-conditioned
        // least-squares / SIP fit). Otherwise fall back to the index's own embedded
        // (uniformized) stars near the solved center.
        let catalog_index = if let Some(external) = external_catalog {
            CatalogIndex::new(
                self.at_observation_epoch(external.stars_near(result.wcs.center, cone_radius)),
            )
        } else {
            let scale_range = self
                .config
                .scale_hint
                .as_ref()
                .map(|s| (s.min_arcsec_per_pixel, s.max_arcsec_per_pixel));
            let cone = Some((result.wcs.center, cone_radius));

            // In tile mode `self.index_set` is empty, so load the tiles around the
            // solved center lazily; otherwise reuse the pre-loaded set (scale-
            // filtered -- each sub-index embeds its own catalog, so `all_indices()`
            // would build a needlessly large tree). Refinement is best-effort: a
            // tile-load failure just returns the coarse result.
            let image_width = result.wcs.image_width;
            let lazy_set;
            let indices_to_use: Vec<&LoadedIndex> = if let Some(dir) = &self.tile_dir {
                match IndexSet::load_tiles_for(dir, image_width, scale_range, cone) {
                    Ok(set) => {
                        lazy_set = set;
                        lazy_set.all_indices().iter().collect()
                    }
                    Err(e) => {
                        debug!("tile load for refinement failed ({e}); returning coarse result");
                        return result;
                    }
                }
            } else {
                self.index_set.select_for(image_width, scale_range, cone)
            };
            let full = CatalogIndex::from_indices(&indices_to_use);
            CatalogIndex::new(
                self.at_observation_epoch(full.stars_near(result.wcs.center, cone_radius)),
            )
        };

        let config = refinement_config.unwrap_or_default();
        let refiner = IterativeRefiner::new(config);

        match refiner.refine(result.wcs.clone(), detected_stars, &catalog_index) {
            Ok(refinement_result) => {
                info!(
                    "Refined: {} stars, {} iters, RMS={:.3} arcsec, center=({:.6}, {:.6})",
                    refinement_result.matched_stars.len(),
                    refinement_result.iterations,
                    refinement_result.rms_residual_arcsec,
                    refinement_result.refined_wcs.center.ra,
                    refinement_result.refined_wcs.center.dec,
                );
                result.wcs = refinement_result.refined_wcs.clone();
                result.refinement = Some(refinement_result);
            }
            Err(e) => {
                debug!("Refinement skipped ({e}); returning coarse result");
            }
        }
        result
    }

    /// Propagate catalog stars to the query's observation epoch along their
    /// proper motions. A no-op when no epoch was given, or for stars without
    /// a proper motion (index-embedded stars never carry one). Runs on the
    /// cone-selected refinement subset, so the cost is negligible.
    fn at_observation_epoch(&self, mut stars: Vec<Star>) -> Vec<Star> {
        let Some(epoch) = self.config.observation_epoch else {
            return stars;
        };
        for star in &mut stars {
            star.position = star.position_at_epoch(epoch);
        }
        stars
    }

    /// Get statistics about the index set.
    #[must_use]
    pub fn index_stats(&self) -> QueryStats {
        let total_stats = self.index_set.total_stats();

        QueryStats {
            num_image_quads: 0,
            num_catalog_quads_checked: total_stats.num_quads,
            num_indices_searched: self.index_set.len(),
            num_indices_skipped: 0,
            num_hypotheses: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DetectedStar;

    #[test]
    fn test_solver_creation() {
        let index_set = IndexSet::new();
        let config = QueryConfig::default();
        let _solver = PlateSolver::new(index_set, config);
    }

    #[test]
    fn test_solve_with_no_indices() {
        let index_set = IndexSet::new();
        let config = QueryConfig::default();
        let solver = PlateSolver::new(index_set, config);

        let stars = vec![
            DetectedStar::new(100.0, 100.0, 1000.0),
            DetectedStar::new(200.0, 100.0, 900.0),
            DetectedStar::new(100.0, 200.0, 800.0),
            DetectedStar::new(200.0, 200.0, 700.0),
        ];

        // Should fail with no indices
        let result = solver.solve(&DetectedField::new(stars, 512, 512));
        assert!(result.is_err());
    }
}
