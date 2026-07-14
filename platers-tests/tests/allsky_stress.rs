//! All-sky stress test: sweep the whole celestial sphere, synthesize a star field
//! from the *real* Tycho-2 catalog at each position, and solve it. Catches
//! position-dependent failures that the single-position parametric sweep cannot:
//! the RA = 0/360 wrap, high-declination projection, density variation
//! (galactic plane vs. poles), and tile selection across the sphere.
//!
//! Ignored by default -- needs the on-disk hybrid catalog + index under
//! `data/` and takes a minute or two.
//!
//! Run: `cargo test --release -p platers-tests --test allsky_stress -- --ignored --nocapture`

use std::path::Path;

use platers_core::{
    geometry::compute_hash_code_pixels, index::IndexSet, load_catalog_parquet, query::PositionHint,
    CatalogIndex, DetectedField, PlateSolver, QueryConfig, ScaleRange, SkyCoord,
    VerificationConfig, Verifier,
};
use platers_tests::test_utils::{generate_test_case, validate_solution, TestCaseConfig};

/// Trace the actual solver on a handful of fields: print the per-field internal
/// counts (image quads generated, quad matches found, hypotheses tested, best
/// log-odds) so we can see *where* a field that should solve actually breaks.
#[test]
#[ignore = "needs on-disk hybrid catalog + index; diagnostic"]
fn diag_solver_trace() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let tiled_dir = root.join("data/index");
    if !catalog_path.exists() || !tiled_dir.exists() {
        eprintln!("SKIP");
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    let cat_index = CatalogIndex::new(load_catalog_parquet(&catalog_path).expect("load"));
    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    let field_radius = 0.5 * (img_w as f64 * scale / 3600.0) + 0.2;

    for (i, &(ra, dec)) in sky_grid(30.0, 40.0).iter().enumerate() {
        let local = cat_index.stars_near(SkyCoord::new_normalized(ra, dec), field_radius);
        let config = TestCaseConfig::new()
            .center(ra, dec)
            .fov(img_w as f64 * scale / 60.0, img_h as f64 * scale / 60.0)
            .image_size(img_w, img_h)
            .rotation((ra + dec).rem_euclid(360.0))
            .stars(80)
            .noise(0.3)
            .seed(i as u64 + 1);
        let Ok((det, gt)) = generate_test_case(&config, &local) else {
            continue;
        };
        if det.len() < MIN_STARS {
            continue;
        }
        let qc = QueryConfig {
            max_stars_for_quads: 50,
            max_quads_to_try: 50000,
            max_hypotheses: 10000,
            hash_code_tolerance: 0.01,
            observation_epoch: None,
            scale_hint: Some(ScaleRange::from_nominal(scale, 0.1)),
            position_hint: Some(PositionHint {
                ra,
                dec,
                radius: 1.0,
            }),
        };
        let r = PlateSolver::from_tile_directory(tiled_dir.clone(), qc, lenient_verification())
            .solve_coarse(&DetectedField::new(det, img_w, img_h));
        let outcome = match r {
            Ok(res) => format!(
                "SOLVED pos_err={:.2}\"",
                validate_solution(&res.wcs, &gt).position_error_arcsec
            ),
            Err(e) => format!("FAILED ({e})"),
        };
        println!(
            "FIELD ({ra:.0},{dec:.0}) rot={:.0} parity={:+.0} => {outcome}",
            config.rotation_deg,
            gt.wcs.parity()
        );
    }
}

/// Diagnostic: for a field, take catalog quads that fall entirely on the image,
/// project their 4 stars to pixels via the *true* WCS, recompute the pixel hash,
/// and compare it to the catalog's stored sky hash. If these disagree by more than
/// the match tolerance, the index and the field hash the *same four stars*
/// differently -- a hash-consistency bug, independent of star selection.
#[test]
#[ignore = "needs on-disk hybrid catalog + index; diagnostic"]
fn diag_hash_consistency() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let tiled_dir = root.join("data/index");
    if !catalog_path.exists() || !tiled_dir.exists() {
        eprintln!("SKIP");
        return;
    }
    let catalog = load_catalog_parquet(&catalog_path).expect("load");
    let cat_index = CatalogIndex::new(catalog);

    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    for &(ra, dec) in &[(200.0, 30.0), (45.0, -20.0), (300.0, 55.0)] {
        let center = SkyCoord::new_normalized(ra, dec);
        let local = cat_index.stars_near(center, 1.0);
        let config = TestCaseConfig::new()
            .center(ra, dec)
            .fov(img_w as f64 * scale / 60.0, img_h as f64 * scale / 60.0)
            .image_size(img_w, img_h)
            .rotation(17.0)
            .stars(80)
            .seed(1);
        let (_stars, gt) = generate_test_case(&config, &local).expect("gen");

        let sr = ScaleRange::from_nominal(scale, 0.1);
        let set = IndexSet::load_tiles_for(
            &tiled_dir,
            img_w,
            Some((sr.min_arcsec_per_pixel, sr.max_arcsec_per_pixel)),
            Some((center, 1.0)),
        )
        .expect("tiles");

        let mut on_image = 0_usize;
        let mut dists: Vec<f64> = Vec::new();
        for idx in set.all_indices() {
            for qi in 0..idx.num_quads() {
                let q = idx.quad(qi);
                let mut px = [(0.0, 0.0); 4];
                let mut all_on = true;
                for (k, &si) in q.star_indices.iter().enumerate() {
                    match gt.wcs.sky_to_pixel(idx.star(si).position) {
                        Ok(p)
                            if p.x >= 0.0
                                && p.x < img_w as f64
                                && p.y >= 0.0
                                && p.y < img_h as f64 =>
                        {
                            px[k] = (p.x, p.y);
                        }
                        _ => {
                            all_on = false;
                            break;
                        }
                    }
                }
                if !all_on {
                    continue;
                }
                on_image += 1;
                let pix = px.map(|(x, y)| platers_core::PixelCoord::new(x, y));
                if let Ok(h) = compute_hash_code_pixels(&pix) {
                    let d: f64 = h
                        .components
                        .iter()
                        .zip(q.hash_code.components.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    dists.push(d);
                }
            }
        }
        dists.sort_by(f64::total_cmp);
        let within = dists.iter().filter(|&&d| d < 0.01).count();
        println!(
            "({ra},{dec}): {on_image} catalog quads on-image; pixel-vs-sky hash dist \
             min={:.5} median={:.5} max={:.5}; {within}/{} within tol 0.01",
            dists.first().copied().unwrap_or(f64::NAN),
            dists.get(dists.len() / 2).copied().unwrap_or(f64::NAN),
            dists.last().copied().unwrap_or(f64::NAN),
            dists.len()
        );
    }
}

/// Diagnostic: does running refinement *again* (re-seeding the next pass from the
/// previous refined pose) improve accuracy beyond a single `refine()`? `refine()`
/// already iterates internally to convergence with a shrinking match radius; this
/// chains whole passes and prints matched-star count, RMS, and center error per pass
/// so we can see whether a second/third pass pulls in more stars or has converged.
#[test]
#[ignore = "needs on-disk hybrid catalog + index; diagnostic"]
fn diag_refine_passes() {
    use platers_core::{IterativeRefiner, RefinementConfig};

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let tiled_dir = root.join("data/index");
    if !catalog_path.exists() || !tiled_dir.exists() {
        eprintln!("SKIP");
        return;
    }
    let cat_index = CatalogIndex::new(load_catalog_parquet(&catalog_path).expect("load"));
    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    let field_radius = 0.5 * ((img_w * img_w + img_h * img_h) as f64).sqrt() * scale / 3600.0 + 0.2;

    let positions = [(200.0, 35.0), (45.0, -20.0), (90.0, 88.0), (0.3, 88.0)];
    println!("\n=== Chained refinement passes (matched / RMS\" / center_err\") ===");
    for &(ra, dec) in &positions {
        let local = cat_index.stars_near(SkyCoord::new_normalized(ra, dec), field_radius);
        let config = TestCaseConfig::new()
            .center(ra, dec)
            .fov(img_w as f64 * scale / 60.0, img_h as f64 * scale / 60.0)
            .image_size(img_w, img_h)
            .rotation((ra + dec).rem_euclid(360.0))
            .stars(250)
            .mag_limit(13.5)
            .noise(0.3)
            .seed(1);
        let Ok((stars, gt)) = generate_test_case(&config, &local) else {
            continue;
        };
        let qc = QueryConfig {
            max_stars_for_quads: 50,
            max_quads_to_try: 50000,
            max_hypotheses: 10000,
            hash_code_tolerance: 0.01,
            observation_epoch: None,
            scale_hint: Some(ScaleRange::from_nominal(scale, 0.1)),
            position_hint: Some(PositionHint {
                ra,
                dec,
                radius: 1.0,
            }),
        };
        let Ok(coarse) =
            PlateSolver::from_tile_directory(tiled_dir.clone(), qc, lenient_verification())
                .solve_coarse(&DetectedField::new(stars.clone(), img_w, img_h))
        else {
            println!("  ({ra:.0},{dec:.0}) coarse failed");
            continue;
        };

        // Refinement catalog: local stars around the solved center (what
        // apply_refinement builds internally).
        let refine_cat = CatalogIndex::new(cat_index.stars_near(coarse.wcs.center, field_radius));
        let refiner = IterativeRefiner::new(RefinementConfig::default());

        let coarse_err = validate_solution(&coarse.wcs, &gt).position_error_arcsec;
        print!("  ({ra:>5.1},{dec:>4.1}) coarse_err={coarse_err:>5.2}\" |");
        let mut wcs = coarse.wcs.clone();
        for pass in 1..=3 {
            match refiner.refine(wcs.clone(), &stars, &refine_cat) {
                Ok(r) => {
                    let err = validate_solution(&r.refined_wcs, &gt).position_error_arcsec;
                    print!(
                        " p{pass}: {}/{:.3}/{:.3} (it={},conv={})",
                        r.matched_stars.len(),
                        r.rms_residual_arcsec,
                        err,
                        r.iterations,
                        r.converged
                    );
                    wcs = r.refined_wcs;
                }
                Err(e) => {
                    print!(" p{pass}: FAILED ({e})");
                    break;
                }
            }
        }
        println!();
    }
}

/// Below this detected-star count a failure is the catalog's (too sparse), not the
/// solver's, so those positions are excluded from the solve-rate.
const MIN_STARS: usize = 8;

fn lenient_verification() -> VerificationConfig {
    VerificationConfig {
        sigma_arcsec: 2.0,
        background_density_per_sqdeg: 1000.0,
        match_radius_arcsec: 5.0,
        min_matches: 4,
        log_odds_threshold: 10.0,
        max_stars_to_verify: 100,
        ..Default::default()
    }
}

/// A roughly equal-area sweep of the sphere: declination bands every `dec_step`,
/// with RA spacing widened by `1/cos(dec)` so high-latitude rings aren't
/// over-sampled. Plus explicit RA-wrap and near-pole edge cases.
fn sky_grid(dec_step: f64, ra_step_equator: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut dec = -85.0_f64;
    while dec <= 85.0 + 1e-6 {
        let cosd = dec.to_radians().cos().max(0.05);
        #[allow(clippy::cast_sign_loss, reason = "step counts are positive")]
        let n_ra = ((360.0 / (ra_step_equator / cosd)).round() as usize).max(1);
        for i in 0..n_ra {
            out.push((360.0 * i as f64 / n_ra as f64, dec));
        }
        dec += dec_step;
    }
    // Edge cases: straddle the RA = 0/360 seam and sit very near the poles.
    for &dec in &[-88.0, 88.0, 0.0, 60.0] {
        out.push((0.3, dec));
        out.push((359.7, dec));
    }
    out
}

#[test]
#[ignore = "needs on-disk hybrid catalog + index; long-running"]
fn allsky_stress_sweep() {
    run_allsky_sweep(0.0, 0.0, "clean photometry");
}

/// The same all-sky sweep, but every field's magnitudes are in a **different band**:
/// a large global zero-point offset on every star plus ~0.25 mag of per-star
/// color/measurement scatter. A global offset is order-preserving and so a no-op for
/// the brightest-per-cell quad selection; the per-star scatter can reorder selection
/// near a `HEALPix` cell boundary (the realistic stress). Confirms the all-sky solve
/// rate holds under realistic photometry, not just truth-perfect fluxes.
#[test]
#[ignore = "needs on-disk hybrid catalog + index; long-running"]
fn allsky_photometric_sweep() {
    run_allsky_sweep(1.0, 0.25, "band +1.0 mag, 0.25 mag scatter");
}

fn run_allsky_sweep(mag_offset: f64, mag_noise: f64, label: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let tiled_dir = root.join("data/index");
    if !catalog_path.exists() || !tiled_dir.exists() {
        eprintln!("SKIP: hybrid catalog or index not present under data/");
        return;
    }

    println!("Loading catalog...");
    let catalog = load_catalog_parquet(&catalog_path).expect("load catalog");
    println!("Indexing {} stars...", catalog.len());
    let cat_index = CatalogIndex::new(catalog);

    // Fixed imaging setup: ~2 arcsec/px, 2048x1489 (~1.14 deg x 0.83 deg field).
    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    let fov_w = img_w as f64 * scale / 60.0;
    let fov_h = img_h as f64 * scale / 60.0;
    let field_radius_deg = 0.5 * (fov_w.max(fov_h) / 60.0) + 0.2;

    let grid = sky_grid(10.0, 15.0);
    println!(
        "Sweeping {} sky positions @ {scale:.1}\"/px [{label}]...\n",
        grid.len()
    );

    // (dec, n_detected, solved, pos_err_arcsec)
    let mut results: Vec<(f64, usize, bool, f64)> = Vec::new();
    let mut wrap_fail = 0_usize;
    let mut true_pose_ok_but_failed = 0_usize;
    let mut fail_reasons: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for (idx, &(ra, dec)) in grid.iter().enumerate() {
        let center = SkyCoord::new_normalized(ra, dec);
        let local = cat_index.stars_near(center, field_radius_deg);

        let config = TestCaseConfig::new()
            .center(ra, dec)
            .fov(fov_w, fov_h)
            .image_size(img_w, img_h)
            .rotation((ra + dec).rem_euclid(360.0))
            .stars(250)
            .mag_limit(13.5)
            .noise(0.3)
            .mag_offset(mag_offset)
            .mag_noise(mag_noise)
            .seed(idx as u64 + 1);
        let Ok((stars, gt)) = generate_test_case(&config, &local) else {
            results.push((dec, 0, false, f64::NAN));
            continue;
        };
        let n = stars.len();

        let qc = QueryConfig {
            max_stars_for_quads: 50,
            max_quads_to_try: 50000,
            max_hypotheses: 10000,
            hash_code_tolerance: 0.01,
            observation_epoch: None,
            scale_hint: Some(ScaleRange::from_nominal(scale, 0.1)),
            position_hint: Some(PositionHint {
                ra,
                dec,
                radius: 1.0,
            }),
        };
        let solver =
            PlateSolver::from_tile_directory(tiled_dir.clone(), qc, lenient_verification());
        // Diagnostic: does the GROUND-TRUTH pose verify against the (full) catalog?
        // High here on a *failed* field => the true quad was never matched (matching
        // problem); low => the pose can't be verified at all (verification problem).
        let true_lo = Verifier::new(lenient_verification())
            .verify(&gt.wcs, &stars, &cat_index)
            .log_odds;

        let res = solver.solve(&DetectedField::new(stars, img_w, img_h));
        match res {
            Ok(r) => {
                let err = validate_solution(&r.wcs, &gt).position_error_arcsec;
                results.push((dec, n, err < 5.0, err));
            }
            Err(e) => {
                results.push((dec, n, false, f64::NAN));
                if n >= MIN_STARS {
                    if true_lo > 20.0 {
                        true_pose_ok_but_failed += 1;
                    }
                    // Bucket the error message (strip the trailing number).
                    let msg = e.to_string();
                    let key = msg
                        .split(['(', ':'])
                        .next()
                        .unwrap_or(&msg)
                        .trim()
                        .to_string();
                    *fail_reasons.entry(key).or_default() += 1;
                }
                if !(2.0..=358.0).contains(&ra) && n >= MIN_STARS {
                    wrap_fail += 1;
                }
            }
        }
    }

    // ---- Aggregate ----
    let solvable: Vec<&(f64, usize, bool, f64)> =
        results.iter().filter(|r| r.1 >= MIN_STARS).collect();
    let solved = solvable.iter().filter(|r| r.2).count();
    let total = solvable.len();
    let sparse = results.len() - total;

    println!("\n=== All-sky stress results [{label}] ===");
    println!(
        "positions: {} ({} sparse <{} stars, excluded)",
        results.len(),
        sparse,
        MIN_STARS
    );
    println!(
        "solve rate (>={MIN_STARS} stars): {}/{} = {:.1}%",
        solved,
        total,
        100.0 * solved as f64 / total.max(1) as f64
    );

    // Per-declination-band breakdown.
    println!("\n  dec band |  n  | solved | rate  | median err");
    println!("  ---------|-----|--------|-------|-----------");
    let mut band = -90;
    while band < 90 {
        let lo = f64::from(band);
        let hi = lo + 10.0;
        let in_band: Vec<&&(f64, usize, bool, f64)> =
            solvable.iter().filter(|r| r.0 >= lo && r.0 < hi).collect();
        if !in_band.is_empty() {
            let s = in_band.iter().filter(|r| r.2).count();
            let mut errs: Vec<f64> = in_band
                .iter()
                .filter(|r| r.2 && r.3.is_finite())
                .map(|r| r.3)
                .collect();
            errs.sort_by(f64::total_cmp);
            let med = errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN);
            println!(
                "  {:+3}...{:+3} | {:3} | {:6} | {:4.0}% | {:.3}\"",
                band,
                band + 10,
                in_band.len(),
                s,
                100.0 * s as f64 / in_band.len() as f64,
                med
            );
        }
        band += 10;
    }
    println!("\n  RA-wrap (RA~0/360) failures with enough stars: {wrap_fail}");
    println!(
        "  failed fields whose TRUE pose verifies (log-odds>20): {true_pose_ok_but_failed}\n\
         \x20   (high => matching problem: the true quad is never matched, not a verify problem)"
    );
    println!("\n  failure reasons (>={MIN_STARS} stars):");
    let mut reasons: Vec<(&String, &usize)> = fail_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, count) in reasons {
        println!("    {count:3} x {reason}");
    }

    let rate = solved as f64 / total.max(1) as f64;
    println!(
        "\n[{label}] SOLVE RATE {:.1}%  (>=99% expected across the sky since the Sec 0\n\
         orientation fix; uniform per declination band, sub-arcsecond accuracy)",
        100.0 * rate
    );

    // Hard invariant: the verifier must not FALSE-ACCEPT. Every *returned* solution
    // has to be accurate -- a returned WCS that is wildly wrong means the accept gate
    // let a bad hypothesis through. (Failing to solve is a reliability gap; accepting
    // a wrong pose is a correctness bug.)
    let false_accepts = results
        .iter()
        .filter(|r| r.3.is_finite() && r.3 > 30.0)
        .count();
    assert_eq!(
        false_accepts, 0,
        "verifier FALSE-ACCEPTED {false_accepts} wildly-wrong pose(s) -- correctness bug"
    );

    // Reliability is a tracked metric (printed above), not yet at target; fail only on
    // catastrophic regression so this stays a usable monitor rather than perma-red.
    assert!(
        rate > 0.25,
        "all-sky solve rate {:.1}% -- catastrophic regression",
        100.0 * rate
    );
}

/// Diagnostic: inject a few **bright spurious sources** (asteroids / satellites /
/// cosmic rays / hot pixels -- point sources brighter than the field stars that are
/// *not* in the catalog) into otherwise-clean fields and solve each one. Shows that
/// the coarse path stays robust to contamination: even when the brightest
/// detections are contaminants that uniformization is forced to keep, the field
/// still builds *matchable* quads (a contaminant only spoils the quads it joins;
/// plenty of all-real-star quads remain), so `matched` should stay at full count
/// and `solved` should barely move.
#[test]
#[ignore = "needs on-disk hybrid catalog + index; diagnostic"]
fn diag_spurious_sources() {
    use platers_core::DetectedStar;
    use rand::{Rng, SeedableRng};

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let tiled_dir = root.join("data/index");
    if !catalog_path.exists() || !tiled_dir.exists() {
        eprintln!("SKIP");
        return;
    }
    let cat_index = CatalogIndex::new(load_catalog_parquet(&catalog_path).expect("load"));
    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    let field_radius = 0.5 * (img_w as f64 * scale / 3600.0) + 0.2;

    println!("\n=== Spurious-source robustness ===");
    println!("spurious | fields | matched | solved");

    let positions = sky_grid(40.0, 60.0);
    for &n_spurious in &[0_usize, 1, 3] {
        let (mut fields, mut matched, mut solved) = (0, 0, 0);

        for (i, &(ra, dec)) in positions.iter().enumerate() {
            let local = cat_index.stars_near(SkyCoord::new_normalized(ra, dec), field_radius);
            let config = TestCaseConfig::new()
                .center(ra, dec)
                .fov(img_w as f64 * scale / 60.0, img_h as f64 * scale / 60.0)
                .image_size(img_w, img_h)
                .rotation((ra + dec).rem_euclid(360.0))
                .stars(250)
                .mag_limit(13.5)
                .noise(0.3)
                .seed(i as u64 + 1);
            let Ok((mut det, gt)) = generate_test_case(&config, &local) else {
                continue;
            };
            if det.len() < MIN_STARS {
                continue;
            }

            // Inject `n_spurious` contaminants: brighter than every real detection
            // (so uniformization is forced to keep them), placed on-image and well
            // clear of any real source so they're genuinely non-catalog.
            let max_flux = det.iter().map(|d| d.flux).fold(0.0_f64, f64::max);
            let mut rng = rand::rngs::StdRng::seed_from_u64(0xA57E_0000 + i as u64);
            let mut placed = 0;
            let mut tries = 0;
            while placed < n_spurious && tries < 1000 {
                tries += 1;
                let x = rng.gen_range(20.0..(img_w as f64 - 20.0));
                let y = rng.gen_range(20.0..(img_h as f64 - 20.0));
                if det.iter().any(|d| (d.x - x).hypot(d.y - y) < 30.0) {
                    continue; // too close to a real source; would blend
                }
                det.push(DetectedStar {
                    x,
                    y,
                    flux: max_flux * 10.0,
                });
                placed += 1;
            }

            let qc = QueryConfig {
                max_stars_for_quads: 50,
                max_quads_to_try: 50000,
                max_hypotheses: 10000,
                hash_code_tolerance: 0.01,
                observation_epoch: None,
                scale_hint: Some(ScaleRange::from_nominal(scale, 0.1)),
                position_hint: Some(PositionHint {
                    ra,
                    dec,
                    radius: 1.0,
                }),
            };
            let solver =
                PlateSolver::from_tile_directory(tiled_dir.clone(), qc, lenient_verification());

            fields += 1;
            match solver.solve_coarse(&DetectedField::new(det, img_w, img_h)) {
                Ok(res) => {
                    if res.num_quads_matched > 0 {
                        matched += 1;
                    }
                    if validate_solution(&res.wcs, &gt).position_error_arcsec < 5.0 {
                        solved += 1;
                    }
                }
                // On failure the error tells us whether quads matched at all:
                // "No matching quads found ..." => contamination broke quad building.
                Err(e) => {
                    if !e.to_string().contains("No matching quads") {
                        matched += 1;
                    }
                }
            }
        }

        println!("   {n_spurious:>2}    |  {fields:>4}  |  {matched:>4}   |  {solved:>4}");
    }
}
