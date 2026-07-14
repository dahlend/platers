//! Robustness to degraded input: missing stars (random incompleteness and
//! saturated-brightest dropout), spurious detections, and pure garbage.
//!
//! All fixture-backed and deterministic (fixed seeds), so these run in CI. The
//! solver gets a scale hint but *no position hint* -- the harder, more honest
//! setting -- and production verification defaults, so a pass here is a real
//! robustness statement, and the garbage test doubles as a false-positive check.

use platers_core::{DetectedField, DetectedStar, QueryConfig, ScaleRange, VerificationConfig};
use platers_tests::test_utils::{generate_test_case, TestCaseConfig, TestHarness};

const SEEDS: [u64; 5] = [1, 2, 3, 4, 5];
/// Solved fields must land within this distance of the true center.
const MAX_CENTER_ERR_ARCSEC: f64 = 10.0;

/// Base field: 30' x 20' at (180, 45) on 2048x1489, 60 stars, 0.3 px noise.
fn base_config(seed: u64) -> TestCaseConfig {
    TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .rotation((seed * 73 % 360) as f64)
        .stars(60)
        .noise(0.3)
        .seed(seed)
}

/// Solve one generated field; `Some(center_error_arcsec)` if it solved.
fn solve_once(harness: &TestHarness, config: &TestCaseConfig) -> Option<f64> {
    let (stars, truth) = generate_test_case(config, harness.catalog()).expect("generate");
    let query = QueryConfig {
        scale_hint: Some(ScaleRange::from_nominal(config.pixel_scale_arcsec(), 0.1)),
        ..QueryConfig::default()
    };
    let solver = harness.create_solver_with_config(query, VerificationConfig::default());
    let field = DetectedField::new(stars, config.image_width, config.image_height);
    let result = solver.solve(&field).ok()?;
    Some(result.wcs.center.angular_distance(&truth.wcs.center) * 3600.0)
}

/// Solve the base field under `corrupt` across all seeds; return the number
/// solved, asserting every solved field is accurate.
fn solve_rate(
    harness: &TestHarness,
    label: &str,
    corrupt: impl Fn(TestCaseConfig) -> TestCaseConfig,
) -> usize {
    let mut solved = 0;
    for &seed in &SEEDS {
        let config = corrupt(base_config(seed));
        if let Some(err_arcsec) = solve_once(harness, &config) {
            assert!(
                err_arcsec < MAX_CENTER_ERR_ARCSEC,
                "{label} seed {seed}: solved but {err_arcsec:.2} arcsec from truth"
            );
            solved += 1;
        }
    }
    println!("{label}: {solved}/{} solved", SEEDS.len());
    solved
}

#[test]
fn dropout_incompleteness() {
    let harness = TestHarness::default();
    let clean = solve_rate(&harness, "dropout  0%", |c| c);
    let light = solve_rate(&harness, "dropout 15%", |c| c.dropout(0.15));
    let heavy = solve_rate(&harness, "dropout 30%", |c| c.dropout(0.30));

    assert_eq!(clean, SEEDS.len(), "clean fields must all solve");
    assert!(light >= 4, "15% dropout should solve nearly all: {light}/5");
    assert!(
        heavy >= 3,
        "30% dropout should still mostly solve: {heavy}/5"
    );
}

#[test]
fn missing_brightest_stars() {
    let harness = TestHarness::default();
    let few = solve_rate(&harness, "drop brightest  4", |c| c.drop_brightest(4));
    let many = solve_rate(&harness, "drop brightest 10", |c| c.drop_brightest(10));

    assert!(
        few >= 4,
        "4 saturated stars should solve nearly all: {few}/5"
    );
    assert!(
        many >= 3,
        "10 saturated stars should mostly solve: {many}/5"
    );
}

#[test]
fn spurious_contamination() {
    let harness = TestHarness::default();
    let light = solve_rate(&harness, "spurious 10%", |c| c.spurious(6));
    let heavy = solve_rate(&harness, "spurious 30%", |c| c.spurious(18));

    assert!(
        light >= 4,
        "10% contamination should solve nearly all: {light}/5"
    );
    assert!(
        heavy >= 3,
        "30% contamination should mostly solve: {heavy}/5"
    );
}

#[test]
fn duplicate_detections() {
    let harness = TestHarness::default();
    // Deblending failures on the brightest stars -- the index's quad anchors
    // each appear twice, a pixel or two apart.
    let light = solve_rate(&harness, "duplicates  4", |c| c.duplicates(4));
    let heavy = solve_rate(&harness, "duplicates 12", |c| c.duplicates(12));

    assert!(
        light >= 4,
        "4 duplicated anchors should solve nearly all: {light}/5"
    );
    assert!(
        heavy >= 3,
        "12 duplicated anchors should mostly solve: {heavy}/5"
    );
}

#[test]
fn combined_degradation() {
    let harness = TestHarness::default();
    // Saturation + incompleteness + contamination + deblending failures +
    // heavier centroid noise, at once: a genuinely bad frame.
    let solved = solve_rate(&harness, "combined degradation", |c| {
        c.drop_brightest(3)
            .dropout(0.15)
            .spurious(8)
            .duplicates(3)
            .noise(0.5)
    });
    assert!(
        solved >= 3,
        "a bad-but-real frame should mostly solve: {solved}/5"
    );
}

/// Per-seed trace of the combined-degradation config, for diagnosing which
/// frame fails and where in the pipeline it dies. Run with:
/// `cargo test -p platers-tests --test degraded_input_tests -- --ignored --nocapture diag`
#[test]
#[ignore = "diagnostic; prints per-seed solver internals"]
fn diag_combined_degradation_per_seed() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();
    let harness = TestHarness::default();
    for &seed in &SEEDS {
        let config = base_config(seed)
            .drop_brightest(3)
            .dropout(0.15)
            .spurious(8)
            .duplicates(3)
            .noise(0.5);
        let (stars, truth) = generate_test_case(&config, harness.catalog()).expect("generate");
        println!(
            "\n=== seed {seed}: rotation {} deg, {} detections after corruption ===",
            config.rotation_deg,
            stars.len()
        );
        let query = QueryConfig {
            scale_hint: Some(ScaleRange::from_nominal(config.pixel_scale_arcsec(), 0.1)),
            ..QueryConfig::default()
        };
        let solver = harness.create_solver_with_config(query, VerificationConfig::default());
        let field = DetectedField::new(stars, config.image_width, config.image_height);
        let outcome = solver.solve(&field);
        match &outcome {
            Ok(r) => println!(
                "seed {seed}: SOLVED, err {:.2} arcsec, log-odds {:.1}, {} matches",
                r.wcs.center.angular_distance(&truth.wcs.center) * 3600.0,
                r.verification.log_odds,
                r.verification.num_matches
            ),
            Err(e) => println!("seed {seed}: FAILED: {e}"),
        }
        if outcome.is_ok() {
            continue;
        }

        // Attribution for a failing seed: drop one corruption at a time to
        // identify which (combination) tips it past the solvable boundary.
        let variants: [(&str, fn(TestCaseConfig) -> TestCaseConfig); 5] = [
            ("without drop_brightest", |c| c.drop_brightest(0)),
            ("without dropout", |c| c.dropout(0.0)),
            ("without spurious", |c| c.spurious(0)),
            ("without duplicates", |c| c.duplicates(0)),
            ("without noise", |c| c.noise(0.3)),
        ];
        for (label, tweak) in variants {
            let config = tweak(
                base_config(seed)
                    .drop_brightest(3)
                    .dropout(0.15)
                    .spurious(8)
                    .duplicates(3)
                    .noise(0.5),
            );
            let outcome = solve_once(&harness, &config)
                .map_or_else(|| "FAILED".to_string(), |e| format!("solved ({e:.2}\")"));
            println!("  {label}: {outcome}");
        }
    }
}

/// Pure garbage must yield NO solution -- a wrong-but-confident WCS on random
/// input would be far worse than an error.
#[test]
fn garbage_input_never_solves() {
    use rand::{Rng, SeedableRng};

    let harness = TestHarness::default();
    for &seed in &SEEDS {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let stars: Vec<DetectedStar> = (0..60)
            .map(|_| DetectedStar {
                x: rng.gen_range(0.0..2048.0),
                y: rng.gen_range(0.0..1489.0),
                flux: rng.gen_range(1.0..1e5),
            })
            .collect();
        let query = QueryConfig {
            scale_hint: Some(ScaleRange::from_nominal(0.88, 0.1)),
            ..QueryConfig::default()
        };
        let solver = harness.create_solver_with_config(query, VerificationConfig::default());
        let result = solver.solve(&DetectedField::new(stars, 2048, 1489));
        assert!(
            result.is_err(),
            "seed {seed}: random detections produced a 'solution' (log-odds {:?})",
            result.map(|r| r.verification.log_odds)
        );
    }
    println!("garbage: 0/{} false positives", SEEDS.len());
}
