//! Integration tests for solver with refinement, end-to-end against the
//! committed fixture index.
//!
//! These exercise the full coarse->refine path against the small committed
//! fixture index. Detected stars are produced by projecting *real catalog
//! stars* through a known WCS (`generate_test_case`), so the field's quads
//! actually exist in the index -- a blind solve can match them. (Synthesized
//! random pixel positions would leave a blind solve nothing to match.)
//!
//! Note on expectations: since the coarse stage re-anchors the field center to
//! the image center, the *coarse* solve is already sub-arcsec on clean data.
//! So the contract these tests pin is not "refine is Nx better than coarse"
//! but "refine yields a sub-arcsec, low-RMS solution and does not regress
//! relative to coarse."

use platers_core::types::{DetectedStar, SkyCoord};
use platers_core::{
    query::PositionHint, refinement::RefinementConfig, CatalogIndex, DetectedField, QueryConfig,
    ScaleRange, VerificationConfig,
};
use platers_tests::test_utils::{
    generate_test_case, validate_solution, TestCaseConfig, TestHarness,
};

/// 1.0 arcsec/pixel over a 2048x1489 frame -> FOV in arcminutes.
const FOV_W_ARCMIN: f64 = 2048.0 / 60.0; // ~34.13'
const FOV_H_ARCMIN: f64 = 1489.0 / 60.0; // ~24.82'

fn query_config(pixel_scale: f64, center: SkyCoord) -> QueryConfig {
    QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000,
        max_hypotheses: 10000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.1)),
        position_hint: Some(PositionHint {
            ra: center.ra,
            dec: center.dec,
            radius: 1.0,
        }),
    }
}

fn verification_config() -> VerificationConfig {
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

/// Refinement yields a sub-arcsec, low-RMS solution and does not regress vs the
/// coarse solve.
#[test]
fn test_refinement_accuracy_improvement() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(15.0)
        .stars(80)
        .noise(0.3)
        .seed(7);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");
    assert!(
        detected_stars.len() >= 20,
        "fixture field too sparse: {} stars",
        detected_stars.len()
    );

    let solver = harness.create_solver_with_config(
        query_config(config.pixel_scale_arcsec(), config.center),
        verification_config(),
    );

    let field = DetectedField::new(detected_stars, config.image_width, config.image_height);

    // Coarse-only baseline.
    let coarse = solver.solve_coarse(&field).expect("Coarse solve failed");
    let coarse_v = validate_solution(&coarse.wcs, &ground_truth);

    // Coarse -> refine (default path).
    let refined = solver
        .solve_with_refinement(&field, None)
        .expect("Refined solve failed");
    let refined_v = validate_solution(&refined.wcs, &ground_truth);

    println!(
        "coarse: pos={:.3}\" scale={:.3}% rot={:.3} deg  refined: pos={:.3}\" scale={:.3}% rot={:.3} deg",
        coarse_v.position_error_arcsec,
        coarse_v.scale_error_percent,
        coarse_v.rotation_error_deg,
        refined_v.position_error_arcsec,
        refined_v.scale_error_percent,
        refined_v.rotation_error_deg,
    );

    // Refined solution must be high quality on clean data.
    assert!(
        refined_v.position_error_arcsec < 2.0,
        "refined position error too large: {:.3}\"",
        refined_v.position_error_arcsec
    );
    assert!(
        refined_v.scale_error_percent < 0.5,
        "refined scale error too large: {:.3}%",
        refined_v.scale_error_percent
    );
    assert!(
        refined_v.rotation_error_deg < 0.5,
        "refined rotation error too large: {:.3} deg",
        refined_v.rotation_error_deg
    );

    // Refinement must not regress relative to coarse (allow a small slack for
    // noise jitter -- both are already excellent post re-anchor).
    assert!(
        refined_v.position_error_arcsec <= coarse_v.position_error_arcsec + 0.5,
        "refinement regressed position: coarse {:.3}\" -> refined {:.3}\"",
        coarse_v.position_error_arcsec,
        refined_v.position_error_arcsec
    );

    let refinement = refined
        .refinement
        .as_ref()
        .expect("refinement should have run");
    println!(
        "refinement: iters={} converged={} matched={} rms={:.3}\"",
        refinement.iterations,
        refinement.converged,
        refinement.matched_stars.len(),
        refinement.rms_residual_arcsec
    );
    assert!(
        refinement.rms_residual_arcsec < 1.0,
        "RMS residual should be sub-arcsec: {:.3}\"",
        refinement.rms_residual_arcsec
    );
}

/// The dense-catalog refinement path (what the server uses, keeping the full
/// catalog resident) produces a sound, sub-arcsec fit with a healthy match count and
/// is at least as accurate as refining against the index's own embedded stars.
///
/// Note: on this synthetic fixture the detections *are* uniformized catalog stars,
/// so the index already matches them 1:1 and the dense catalog cannot show a higher
/// match count -- the denser star list instead trims to fewer but cleaner matches
/// (lower RMS). The match-count win appears on real frames, where detections include
/// faint stars the index's brightest-per-cell uniformization dropped. Here we assert
/// the path is sound and not a regression.
#[test]
fn test_refinement_against_dense_catalog() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(30.0)
        .stars(80)
        .noise(0.5)
        .seed(7);

    let (detected_stars, _ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    let solver = harness.create_solver_with_config(
        query_config(config.pixel_scale_arcsec(), config.center),
        verification_config(),
    );

    let field = DetectedField::new(detected_stars, config.image_width, config.image_height);

    // Default path: refine against the index's own embedded (uniformized) stars.
    let index_only = solver
        .solve_with_refinement(&field, None)
        .expect("index-only solve failed");

    // Dense path: refine against the full fixture catalog (superset of the index's
    // uniformized stars) -- what the server passes in.
    let dense = CatalogIndex::new(harness.catalog().to_vec());
    let with_catalog = solver
        .solve_with_refinement_against(&field, None, Some(&dense))
        .expect("dense-catalog solve failed");

    let r_idx = index_only
        .refinement
        .as_ref()
        .expect("index-only refinement ran");
    let r_cat = with_catalog
        .refinement
        .as_ref()
        .expect("dense-catalog refinement ran");

    println!(
        "index-only: matched={} rms={:.3}\"  dense-catalog: matched={} rms={:.3}\"",
        r_idx.matched_stars.len(),
        r_idx.rms_residual_arcsec,
        r_cat.matched_stars.len(),
        r_cat.rms_residual_arcsec,
    );

    // Healthy match count (comfortably above the SIP minimum), sub-arcsec, and no
    // worse than index-only (here it is better -- fewer but cleaner matches).
    assert!(
        r_cat.matched_stars.len() >= 20,
        "dense-catalog refinement too few matches for a stable fit: {}",
        r_cat.matched_stars.len(),
    );
    assert!(
        r_cat.rms_residual_arcsec < 1.0,
        "dense-catalog RMS not sub-arcsec: {:.3}\"",
        r_cat.rms_residual_arcsec,
    );
    assert!(
        r_cat.rms_residual_arcsec <= r_idx.rms_residual_arcsec + 0.15,
        "dense-catalog refinement regressed RMS: index {:.3}\" -> dense {:.3}\"",
        r_idx.rms_residual_arcsec,
        r_cat.rms_residual_arcsec,
    );
}

/// A custom (tighter) refinement config still produces a clean, low-RMS solve
/// with a healthy number of matched stars.
#[test]
fn test_refinement_with_custom_config() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(30.0)
        .stars(80)
        .noise(0.5)
        .seed(11);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    let solver = harness.create_solver_with_config(
        query_config(config.pixel_scale_arcsec(), config.center),
        verification_config(),
    );

    // A custom config that differs from the default (tighter final radius, more
    // iterations, stricter outlier rejection and convergence) but keeps a wide
    // *initial* capture radius. The initial radius must exceed the coarse pose
    // error: on a noisy dense field the coarse 4-star pose can be ~10-12" off
    // (refinement is exactly what tightens it), so a too-tight initial radius
    // (e.g. 5") sits below that error and the match set becomes a knife-edge --
    // sometimes catching the full field, sometimes only a handful. This is the
    // Phase-4 wide-initial-radius lesson: size the capture radius to the *seed*
    // error, not the target precision.
    let custom = RefinementConfig {
        initial_radius_arcsec: 25.0,
        final_radius_arcsec: 0.5,
        max_iterations: 6,
        min_stars: 15,
        outlier_sigma: 2.5,
        convergence_threshold: 0.05,
        ..Default::default()
    };

    let result = solver
        .solve_with_refinement(
            &DetectedField::new(detected_stars, config.image_width, config.image_height),
            Some(custom),
        )
        .expect("Solve failed");

    let validation = validate_solution(&result.wcs, &ground_truth);
    assert!(
        validation.position_error_arcsec < 2.0,
        "position error too large: {:.3}\"",
        validation.position_error_arcsec
    );

    let refinement = result
        .refinement
        .as_ref()
        .expect("refinement should have run");
    println!(
        "custom refinement: iters={} converged={} matched={} rms={:.3}\"",
        refinement.iterations,
        refinement.converged,
        refinement.matched_stars.len(),
        refinement.rms_residual_arcsec
    );
    assert!(
        refinement.matched_stars.len() >= 15,
        "should match at least 15 stars, matched {}",
        refinement.matched_stars.len()
    );
    assert!(
        refinement.rms_residual_arcsec < 1.0,
        "RMS should be sub-arcsec: {:.3}\"",
        refinement.rms_residual_arcsec
    );
}

/// A field of uniformly-random detections (no real catalog correspondence). The
/// correct outcome is rejection: there is no field to find.
fn random_field(num: usize, w: usize, h: usize, seed: u64) -> Vec<DetectedStar> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..num)
        .map(|_| DetectedStar {
            x: rng.r#gen::<f64>() * w as f64,
            y: rng.r#gen::<f64>() * h as f64,
            flux: 1000.0 + rng.r#gen::<f64>() * 500.0,
        })
        .collect()
}

/// The solver must not manufacture false accepts. Across many fields of pure
/// random junk (no real catalog correspondence), every one must be rejected -- a
/// promising-looking coarse hypothesis must never be coaxed past the verifier.
/// Runs on any catalog (including the fixture in CI); the dense regional set,
/// with far more index quads, is the stronger stress.
#[test]
fn test_random_fields_are_rejected() {
    let harness = TestHarness::new().expect("Failed to create test harness");
    let (w, h) = (2048, 1489);
    let qc = query_config(1.0, SkyCoord::new(180.0, 45.0));
    let vc = verification_config();

    let trials = 16;
    let mut false_accepts = 0;
    for seed in 0..trials {
        let stars = random_field(80, w, h, seed);
        if harness
            .create_solver_with_config(qc.clone(), vc.clone())
            .solve(&DetectedField::new(stars, w, h))
            .is_ok()
        {
            false_accepts += 1;
        }
    }
    println!("false accepts over {trials} random fields: {false_accepts}");
    assert_eq!(
        false_accepts, 0,
        "solver should reject all {trials} random fields, got {false_accepts} false accepts"
    );
}

/// Photometric robustness: the detector's magnitudes may be in a **different band**
/// than the catalog -- a large arbitrary *global* zero-point offset on every star --
/// on top of ~0.25 mag of per-star color/measurement scatter. The solver picks quad
/// stars by brightness *ordering* (brightest-per-HEALPix-cell), so:
///   - a global offset is order-preserving and must be a no-op, and
///   - 0.25 mag of per-star scatter can flip selection near a cell boundary, yet
///     enough cells keep their true brightest star that the field still solves.
///
/// Every (offset, seed) combination must still solve to good accuracy.
#[test]
fn test_solves_with_band_offset_and_photometric_scatter() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Arbitrary global band offsets (mag): brighter and fainter zero-points.
    let global_offsets = [-2.0, -0.7, 0.9, 2.5];
    let mut solved = 0;
    let mut attempted = 0;

    for (i, &offset) in global_offsets.iter().enumerate() {
        for seed in 0..4_u64 {
            let config = TestCaseConfig::new()
                .center(180.0, 45.0)
                .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
                .image_size(2048, 1489)
                .rotation(20.0 * i as f64 + 5.0 * seed as f64)
                .stars(80)
                .noise(0.3)
                .mag_offset(offset)
                .mag_noise(0.25)
                .seed(seed + 1);
            let (detected_stars, ground_truth) =
                generate_test_case(&config, harness.catalog()).expect("generate");
            if detected_stars.len() < 20 {
                continue; // too sparse to be a fair attempt
            }
            attempted += 1;

            let solver = harness.create_solver_with_config(
                query_config(config.pixel_scale_arcsec(), config.center),
                verification_config(),
            );
            match solver.solve(&DetectedField::new(
                detected_stars,
                config.image_width,
                config.image_height,
            )) {
                Ok(result) => {
                    let v = validate_solution(&result.wcs, &ground_truth);
                    assert!(
                        v.position_error_arcsec < 5.0,
                        "offset {offset:+.1} mag, seed {seed}: solved but inaccurate \
                         ({:.2}\")",
                        v.position_error_arcsec
                    );
                    solved += 1;
                }
                Err(e) => {
                    println!("offset {offset:+.1} mag, seed {seed}: FAILED ({e})");
                }
            }
        }
    }

    println!("photometric robustness: solved {solved}/{attempted}");
    // The fixture field is dense; 0.25 mag scatter + a band offset should not break
    // it. Require every attempt to solve (a single failure is a real regression).
    assert!(attempted >= 8, "too few fair attempts: {attempted}");
    assert_eq!(
        solved, attempted,
        "all band-offset + photometric-scatter fields should solve: {solved}/{attempted}"
    );
}

/// Layout independence: the same field must solve -- to the same pose -- against
/// a *tiled* index directory and a *merged* all-sky set built from the same
/// tiles. Image-quad generation pools quads over every tier grid, so the result
/// must depend only on *which* tiers are present, never on how they are laid
/// out on disk or the order they load in.
#[test]
fn test_tiled_and_merged_indices_solve_consistently() {
    use platers_core::PlateSolver;
    use platers_tests::test_utils::merged_fixture_index_set;

    let harness = TestHarness::new().expect("Failed to create test harness"); // tiled dir
    let merged = merged_fixture_index_set();

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(25.0)
        .stars(80)
        .noise(0.3)
        .seed(5);
    let (stars, gt) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");
    let qc = || query_config(config.pixel_scale_arcsec(), config.center);
    let field = DetectedField::new(stars, config.image_width, config.image_height);

    let tiled =
        PlateSolver::with_verification(harness.index_set.clone(), qc(), verification_config())
            .solve(&field)
            .expect("tiled-layout solve failed");
    let merged = PlateSolver::with_verification(merged, qc(), verification_config())
        .solve(&field)
        .expect("merged-layout solve failed");

    let vt = validate_solution(&tiled.wcs, &gt);
    let vm = validate_solution(&merged.wcs, &gt);
    assert!(
        vt.position_error_arcsec < 5.0,
        "tiled inaccurate: {:.2}\"",
        vt.position_error_arcsec
    );
    assert!(
        vm.position_error_arcsec < 5.0,
        "merged inaccurate: {:.2}\"",
        vm.position_error_arcsec
    );
    let center_diff = tiled.wcs.center.angular_distance(&merged.wcs.center) * 3600.0;
    assert!(
        center_diff < 2.0,
        "tiled vs merged disagree by {center_diff:.2}\" -- layout-dependent solve"
    );
}

/// Parity: a mirror-flipped field must still solve. Real detectors come in both
/// parities (e.g. North-down ZTF vs North-up LCO), and the geometric hash encodes
/// handedness, so the solver tries both parities. Mirroring a field across the x
/// axis flips its parity while keeping the same sky stars; it must still solve to
/// the same center. (Before the both-parities fix, the opposite-parity field was
/// unsolvable -- half of all real images.)
#[test]
fn test_solves_mirror_flipped_parity() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(15.0)
        .stars(80)
        .noise(0.3)
        .seed(11);
    let (detected, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");
    assert!(
        detected.len() >= 20,
        "fixture field too sparse: {}",
        detected.len()
    );

    // Mirror across the x axis: same sky stars, opposite parity.
    let w = config.image_width as f64;
    let mirrored: Vec<DetectedStar> = detected
        .iter()
        .map(|d| DetectedStar {
            x: w - 1.0 - d.x,
            y: d.y,
            flux: d.flux,
        })
        .collect();

    let solver = harness.create_solver_with_config(
        query_config(config.pixel_scale_arcsec(), config.center),
        verification_config(),
    );
    let result = solver
        .solve(&DetectedField::new(
            mirrored,
            config.image_width,
            config.image_height,
        ))
        .expect("mirror-flipped (opposite-parity) field should solve");
    let v = validate_solution(&result.wcs, &ground_truth);
    assert!(
        v.position_error_arcsec < 5.0,
        "mirror-flipped solve inaccurate: {:.2}\"",
        v.position_error_arcsec
    );
    assert!(
        result.wcs.parity() * ground_truth.wcs.parity() < 0.0,
        "mirrored field must solve to the OPPOSITE parity of the truth WCS \
         (got {}, truth {})",
        result.wcs.parity(),
        ground_truth.wcs.parity()
    );
}

/// SIP distortion fitting, end to end. A field with real (barrel) radial
/// distortion is solved twice: with the default config, whose solution must
/// stay a pure linear WCS (SIP is opt-in), and with `sip_order: Some(3)`, whose
/// solution must carry fitted SIP polynomials and a lower final residual than
/// the linear fit of the same matches. Order 3 is the smallest that can
/// represent the fixture's distortion: the `(1 + k*r_n^2)` displacement is
/// cubic in pixel coordinates (`x * (x^2 + y^2)`).
#[test]
fn test_sip_refinement_fits_radial_distortion() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(20.0)
        .stars(150)
        .noise(0.1)
        .distortion(0.002)
        .seed(17);
    let (detected, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    let solver = harness.create_solver_with_config(
        query_config(config.pixel_scale_arcsec(), config.center),
        verification_config(),
    );
    let field = DetectedField::new(detected, config.image_width, config.image_height);
    // Refine against the dense fixture catalog (the server's configuration):
    // SIP needs many well-spread matches.
    let dense = CatalogIndex::new(harness.catalog().to_vec());

    let linear = solver
        .solve_with_refinement_against(&field, None, Some(&dense))
        .expect("distorted field should solve with the default (linear) config");
    assert!(
        linear.wcs.sip.is_none(),
        "SIP must be off by default: the default config produced a SIP WCS"
    );

    let sip_config = RefinementConfig {
        sip_order: Some(3),
        ..RefinementConfig::default()
    };
    let with_sip = solver
        .solve_with_refinement_against(&field, Some(sip_config), Some(&dense))
        .expect("distorted field should solve with SIP enabled");

    let r_lin = linear.refinement.as_ref().expect("linear refinement ran");
    let r_sip = with_sip.refinement.as_ref().expect("SIP refinement ran");
    println!(
        "linear: matched={} rms={:.3}\"  sip(3): matched={} rms={:.3}\"",
        r_lin.matched_stars.len(),
        r_lin.rms_residual_arcsec,
        r_sip.matched_stars.len(),
        r_sip.rms_residual_arcsec,
    );

    assert!(
        with_sip.wcs.sip.is_some(),
        "SIP order 3 requested with {} matched stars, but no SIP was fitted",
        r_sip.matched_stars.len()
    );
    assert!(
        r_sip.rms_residual_arcsec < r_lin.rms_residual_arcsec,
        "SIP fit must beat the linear fit on a distorted field: \
         linear {:.3}\" -> sip {:.3}\"",
        r_lin.rms_residual_arcsec,
        r_sip.rms_residual_arcsec,
    );
    let v = validate_solution(&with_sip.wcs, &ground_truth);
    assert!(
        v.position_error_arcsec < 5.0,
        "SIP solve inaccurate: {:.2}\"",
        v.position_error_arcsec
    );
}

/// Proper-motion propagation, end to end. The refinement catalog stores
/// positions 10 years of proper motion AWAY from where the detections actually
/// are (mimicking an epoch-2016 catalog matched against a later frame); with
/// `observation_epoch` set, the solver must propagate the catalog to the frame
/// date and refine to a tighter residual than the epoch-less solve.
#[test]
fn test_observation_epoch_propagates_proper_motion() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(FOV_W_ARCMIN, FOV_H_ARCMIN)
        .image_size(2048, 1489)
        .rotation(10.0)
        .stars(80)
        .noise(0.1)
        .seed(23);
    let (detected, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");
    let field = DetectedField::new(detected, config.image_width, config.image_height);

    // Detections sit at the TRUE (frame-epoch) positions. Build a catalog whose
    // stored positions are displaced by -0.5" in Dec with a proper motion that
    // walks them back over a 10-year baseline: +50 mas/yr x 10 yr = +0.5".
    let epoch = platers_core::CATALOG_EPOCH + 10.0;
    let displaced: Vec<_> = harness
        .catalog()
        .iter()
        .map(|s| {
            let mut moved = *s;
            moved.position = SkyCoord::new(s.position.ra, s.position.dec - 0.5 / 3600.0);
            moved.proper_motion = Some([0.0, 50.0]);
            moved
        })
        .collect();
    let dense = CatalogIndex::new(displaced);

    let solve = |with_epoch: bool| {
        let mut qc = query_config(config.pixel_scale_arcsec(), config.center);
        qc.observation_epoch = with_epoch.then_some(epoch);
        let solver = harness.create_solver_with_config(qc, verification_config());
        solver
            .solve_with_refinement_against(&field, None, Some(&dense))
            .expect("field should solve")
    };

    let stale = solve(false);
    let propagated = solve(true);
    // A COHERENT catalog displacement does not inflate the fit residual -- the
    // refit absorbs it straight into the pose. Its signature is a biased
    // center: the stale solve lands ~0.5" from the truth, the propagated one
    // on it.
    let err_stale = validate_solution(&stale.wcs, &ground_truth).position_error_arcsec;
    let err_prop = validate_solution(&propagated.wcs, &ground_truth).position_error_arcsec;
    println!("stale-epoch center error={err_stale:.3}\"  propagated={err_prop:.3}\"");

    assert!(
        err_stale > 0.35,
        "the stale-epoch solve should carry most of the 0.5\" catalog displacement, got {err_stale:.3}\""
    );
    assert!(
        err_prop < 0.15,
        "epoch propagation should recover the true positions, got {err_prop:.3}\""
    );
}
