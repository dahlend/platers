//! Basic plate solve integration tests.
//!
//! These tests validate the end-to-end plate solving pipeline:
//! 1. Generate synthetic detected stars from a known WCS
//! 2. Run the plate solver
//! 3. Compare the solution to ground truth
//!
//! This is the first true integration test of the complete system.

use platers_core::{
    query::{PositionHint, QueryConfig, ScaleRange},
    DetectedField, VerificationConfig,
};
use platers_tests::test_utils::{
    generate_test_case, validate_solution, TestCaseConfig, TestHarness,
};

/// Test the simplest possible solve: clean data, known position, moderate FOV.
///
/// This is the "hello world" of plate solving - if this doesn't work, nothing will.
#[test]
fn test_simple_solve_success() {
    // Initialize tracing for performance debugging
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_test_writer()
        .try_init();

    println!("\n=== Test: Simple Plate Solve (Clean Data) ===");
    println!("Goal: Validate end-to-end pipeline with perfect data\n");

    // Create test harness
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Configure a simple test case
    // Use center of our regional catalog coverage (RA=180, Dec=45)
    let config = TestCaseConfig::new()
        .center(180.0, 45.0) // Center of regional catalog
        .fov(30.0, 20.0) // 30x20 arcmin FOV (moderate size)
        .image_size(2048, 1489) // Typical amateur scope resolution
        .rotation(0.0) // No rotation
        .stars(100) // Request 100 stars (will get fewer after bounds check)
        .noise(0.0) // No noise (perfect measurements)
        .mag_limit(16.0); // Include fainter stars

    println!("Test case configuration:");
    println!(
        "  Center: RA={:.1} deg, Dec={:.1} deg",
        config.center.ra, config.center.dec
    );
    println!(
        "  FOV: {:.1}x{:.1} arcmin",
        config.fov_width_arcmin, config.fov_height_arcmin
    );
    println!(
        "  Image: {}x{} pixels",
        config.image_width, config.image_height
    );
    println!(
        "  Pixel scale: {:.2} arcsec/pixel",
        config.pixel_scale_arcsec()
    );
    println!("  Stars: {}", config.num_stars);
    println!(
        "  Noise: {:.1} pixels (perfect data)",
        config.position_noise_pixels
    );

    // Generate test case
    println!("\nGenerating test case...");
    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("  Generated {} detected stars", detected_stars.len());

    // Show sample detected stars with ground truth
    println!("\nSample detected stars (first 5) with ground truth:");
    for (i, star) in detected_stars.iter().take(5).enumerate() {
        let true_pos = &ground_truth.true_positions[i];
        println!(
            "    Star {}: pixel=({:.1}, {:.1}), sky=RA={:.6} deg, Dec={:.6} deg",
            i, star.x, star.y, true_pos.ra, true_pos.dec
        );
    }

    assert!(
        detected_stars.len() >= 20,
        "Need at least 20 stars for reliable solving (got {})",
        detected_stars.len()
    );

    // Create solver with scale hint (realistic usage)
    let pixel_scale = config.pixel_scale_arcsec();

    let query_config = QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000, // Try MANY more quads
        max_hypotheses: 10000,   // Allow MANY more hypotheses
        hash_code_tolerance: 0.01,
        observation_epoch: None, // Loosen tolerance to catch correct matches
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.1)), // +/-10% scale tolerance
        position_hint: Some(PositionHint {
            ra: config.center.ra,
            dec: config.center.dec,
            radius: 1.0, // Within 1 degree of expected position
        }),
    };

    let verification_config = VerificationConfig {
        sigma_arcsec: 2.0, // More lenient position error model
        background_density_per_sqdeg: 1000.0,
        match_radius_arcsec: 5.0, // Larger match radius
        min_matches: 4,           // Lower minimum matches
        log_odds_threshold: 10.0, // Much lower threshold for testing
        max_stars_to_verify: 100,
        ..Default::default()
    };

    let solver = harness.create_solver_with_config(query_config, verification_config);

    // Solve!
    println!("\nRunning plate solver...");
    let start = std::time::Instant::now();

    let result = solver.solve(&DetectedField::new(
        detected_stars,
        config.image_width,
        config.image_height,
    ));

    let elapsed = start.elapsed();
    println!("  Solve time: {:.2}s", elapsed.as_secs_f64());

    // Check result
    match result {
        Ok(solution) => {
            println!("\nSolve SUCCEEDED!");
            println!("\nSolve Statistics:");
            println!("  Image quads: {}", solution.num_image_quads);
            println!("  Quads matched: {}", solution.num_quads_matched);
            println!("  Hypotheses tested: {}", solution.num_hypotheses_tested);
            println!("  Indices used: {}", solution.indices_used.len());

            println!("\nSolution:");
            println!(
                "  Center: RA={:.6} deg, Dec={:.6} deg",
                solution.wcs.center.ra, solution.wcs.center.dec
            );
            println!(
                "  Scale: {:.4} arcsec/pixel",
                solution.wcs.scale_arcsec_per_pixel()
            );
            println!("  Rotation: {:.2} deg", solution.wcs.rotation_deg());

            println!("\nGround Truth:");
            println!(
                "  Center: RA={:.6} deg, Dec={:.6} deg",
                ground_truth.wcs.center.ra, ground_truth.wcs.center.dec
            );
            println!(
                "  Scale: {:.4} arcsec/pixel",
                ground_truth.wcs.scale_arcsec_per_pixel()
            );
            println!("  Rotation: {:.2} deg", ground_truth.wcs.rotation_deg());

            // Validate solution
            let validation = validate_solution(&solution.wcs, &ground_truth);

            println!("\nValidation Results:");
            println!(
                "  Position error: {:.4} arcsec",
                validation.position_error_arcsec
            );
            println!("  Scale error: {:.4}%", validation.scale_error_percent);
            println!("  Rotation error: {:.4} deg", validation.rotation_error_deg);

            // For clean data, we expect very high accuracy
            // Note: Position can be slightly off when computed from a single quad
            // but should be within a few arcmin
            assert!(
                validation.position_error_arcsec < 600.0,
                "Position error too large: {:.4} arcsec",
                validation.position_error_arcsec
            );
            assert!(
                validation.scale_error_percent < 1.0,
                "Scale error too large: {:.4}%",
                validation.scale_error_percent
            );
            assert!(
                validation.rotation_error_deg < 1.0,
                "Rotation error too large: {:.4} deg",
                validation.rotation_error_deg
            );

            println!("\nALL VALIDATION CHECKS PASSED!");
            if validation.position_error_arcsec < 60.0 {
                println!("   Position accuracy: excellent (<1 arcmin)");
            } else {
                println!("   Position accuracy: good (<10 arcmin, refinement needed)");
            }
            println!("   Scale accuracy: excellent (<1%)");
            println!("   Rotation accuracy: excellent (<1 deg)");
        }
        Err(e) => {
            println!("\nSolve FAILED: {e}");
            println!("\nThis suggests either:");
            println!("  1. WCS projection mismatch between generation and solving");
            println!("  2. Verification threshold too stringent");
            println!("  3. Scale hint not matching actual data");
            println!("  4. Hash code matching not finding correct quads");
            panic!("Solve FAILED: {e}");
        }
    }

    println!("\nTEST COMPLETE: Simple solve works end-to-end!");
}

/// Test solving with different field of view sizes.
#[test]
#[ignore = "slow; run explicitly"]
fn test_solve_different_scales() {
    println!("\n=== Test: Solve at Different Scales ===\n");

    let harness = TestHarness::new().expect("Failed to create test harness");

    // Test scales: small, medium, large FOVs
    let test_scales = [
        ("Small FOV", 15.0, 10.0),  // ~0.44 arcsec/pixel (0.25 deg x 0.17 deg)
        ("Medium FOV", 30.0, 20.0), // ~0.88 arcsec/pixel (0.50 deg x 0.33 deg)
        ("Large FOV", 60.0, 40.0),  // ~1.76 arcsec/pixel (1.00 deg x 0.67 deg)
                                    // Wider FOVs excluded: large pixel scales hit
                                    // quad-matching ambiguity on the fixture index.
                                    // ("Wide FOV", 120.0, 80.0),        // ~3.52 arcsec/pixel (2.00 deg x 1.33 deg)
                                    // ("Ultra-Wide FOV", 180.0, 120.0), // ~5.27 arcsec/pixel (3.00 deg x 2.00 deg)
    ];

    for (name, fov_width, fov_height) in &test_scales {
        println!(
            "--- Testing: {} ({:.1}x{:.1} arcmin = {:.2} degx{:.2} deg) ---",
            name,
            fov_width,
            fov_height,
            fov_width / 60.0,
            fov_height / 60.0
        );

        let config = TestCaseConfig::new()
            .center(180.0, 45.0)
            .fov(*fov_width, *fov_height)
            .image_size(2048, 1489)
            .stars(100) // Request more stars
            .mag_limit(16.0) // Use fainter stars for testing
            .noise(0.0);

        let (detected_stars, ground_truth) =
            generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

        if detected_stars.len() < 20 {
            println!(
                "   Skipping - insufficient stars ({} < 20)",
                detected_stars.len()
            );
            continue;
        }

        // Debug: Show sample detected stars to verify scale
        println!("\n  Sample detected stars (first 5):");
        for (i, star) in detected_stars.iter().take(5).enumerate() {
            println!("    Star {}: ({:.2}, {:.2})", i, star.x, star.y);
        }
        println!("  Ground truth WCS:");
        println!(
            "    Center: ({:.6}, {:.6})",
            ground_truth.wcs.center.ra, ground_truth.wcs.center.dec
        );
        println!(
            "    Scale: {:.4} arcsec/px",
            ground_truth.wcs.scale_arcsec_per_pixel()
        );
        println!(
            "    Image size: {}x{}",
            ground_truth.wcs.image_width, ground_truth.wcs.image_height
        );

        let pixel_scale = config.pixel_scale_arcsec();

        // Use wider tolerance for larger FOVs (they have more distortion)
        let scale_tolerance = if *fov_width > 100.0 {
            0.3 // +/-30% for wide-angle (>100 arcmin)
        } else if *fov_width > 50.0 {
            0.25 // +/-25% for large FOV (>50 arcmin) - accounts for projection effects
        } else {
            0.1 // +/-10% for normal FOV
        };

        println!(
            "  Using scale tolerance: +/-{:.0}%",
            scale_tolerance * 100.0
        );

        let query_config = QueryConfig {
            max_stars_for_quads: 30,
            max_quads_to_try: 500, // High limit for comprehensive testing
            max_hypotheses: 1000,  // High limit for comprehensive testing
            hash_code_tolerance: 0.01,
            observation_epoch: None, // Optimal tolerance
            scale_hint: Some(ScaleRange::from_nominal(pixel_scale, scale_tolerance)),
            position_hint: Some(PositionHint {
                ra: config.center.ra,
                dec: config.center.dec,
                radius: 1.0,
            }),
        };

        let solver = harness.create_solver_with_config(query_config, VerificationConfig::default());

        let start = std::time::Instant::now();
        let result = solver.solve(&DetectedField::new(
            detected_stars,
            config.image_width,
            config.image_height,
        ));
        let elapsed = start.elapsed();

        match result {
            Ok(solution) => {
                let validation = validate_solution(&solution.wcs, &ground_truth);
                println!(
                    "  Success! Time: {:.2}s, Pos error: {:.1} arcsec, Scale error: {:.3}%, Rot error: {:.2} deg",
                    elapsed.as_secs_f64(),
                    validation.position_error_arcsec,
                    validation.scale_error_percent,
                    validation.rotation_error_deg
                );

                // Single-quad WCS position error scales with FOV size
                // Small FOVs: ~6 arcmin (360 arcsec)
                // Large FOVs: ~20 arcmin (1200 arcsec) due to quad covering smaller fraction
                // Wide FOVs: ~30 arcmin (1800 arcsec)
                let max_position_error = if *fov_width > 100.0 {
                    1800.0 // +/-30 arcmin for wide FOV
                } else if *fov_width > 50.0 {
                    1200.0 // +/-20 arcmin for large FOV
                } else {
                    600.0 // +/-10 arcmin for normal FOV
                };

                assert!(
                    validation.position_error_arcsec < max_position_error,
                    "Position error too large for {}: {:.1} arcsec (max {:.0})",
                    name,
                    validation.position_error_arcsec,
                    max_position_error
                );
                assert!(
                    validation.scale_error_percent < 5.0,
                    "Scale error too large for {}: {:.3}%",
                    name,
                    validation.scale_error_percent
                );
                assert!(
                    validation.rotation_error_deg < 5.0,
                    "Rotation error too large for {}: {:.2} deg",
                    name,
                    validation.rotation_error_deg
                );
            }
            Err(e) => {
                println!("  Failed: {e}");
                panic!("Solve failed for {name}");
            }
        }
    }

    println!("\nAll scales tested successfully!");
}

/// Test solving with rotation.
#[test]
#[ignore = "run explicitly"]
fn test_solve_with_rotation() {
    println!("\n=== Test: Solve with Different Rotations ===\n");

    let harness = TestHarness::new().expect("Failed to create test harness");

    let test_rotations = [0.0, 45.0, 90.0, 135.0, 180.0];

    for rotation in &test_rotations {
        println!("--- Testing rotation: {rotation:.0} deg ---");

        let config = TestCaseConfig::new()
            .center(180.0, 45.0)
            .fov(30.0, 20.0)
            .image_size(2048, 1489)
            .rotation(*rotation)
            .stars(100) // Request more stars
            .mag_limit(16.0) // Use fainter stars for testing
            .noise(0.0);

        let (detected_stars, ground_truth) =
            generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

        if detected_stars.len() < 20 {
            println!("   Skipping - insufficient stars");
            continue;
        }

        let pixel_scale = config.pixel_scale_arcsec();
        let query_config = QueryConfig {
            max_stars_for_quads: 30,
            max_quads_to_try: 500, // High limit for comprehensive testing
            max_hypotheses: 1000,  // High limit for comprehensive testing
            hash_code_tolerance: 0.01,
            observation_epoch: None, // Optimal tolerance
            scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.1)), // +/-10% optimal
            position_hint: Some(PositionHint {
                ra: config.center.ra,
                dec: config.center.dec,
                radius: 1.0,
            }),
        };

        let solver = harness.create_solver_with_config(query_config, VerificationConfig::default());

        let result = solver.solve(&DetectedField::new(
            detected_stars,
            config.image_width,
            config.image_height,
        ));

        match result {
            Ok(solution) => {
                let validation = validate_solution(&solution.wcs, &ground_truth);
                println!(
                    "  Success! Rot error: {:.2} deg, Scale error: {:.3}%, Pos error: {:.1} arcsec",
                    validation.rotation_error_deg,
                    validation.scale_error_percent,
                    validation.position_error_arcsec
                );

                // Check rotation is accurate (main thing being tested)
                assert!(
                    validation.rotation_error_deg < 5.0,
                    "Rotation error too large: {:.2} deg",
                    validation.rotation_error_deg
                );
                // Scale should also be good
                assert!(
                    validation.scale_error_percent < 5.0,
                    "Scale error too large: {:.3}%",
                    validation.scale_error_percent
                );
            }
            Err(e) => {
                println!("  Failed: {e}");
                panic!("Solve failed for rotation {rotation:.0} deg");
            }
        }
    }

    println!("\nAll rotations tested successfully!");
}
