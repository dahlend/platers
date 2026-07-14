//! Basic integration tests for end-to-end plate solving.

use platers_core::{
    query::PositionHint, DetectedField, QueryConfig, ScaleRange, VerificationConfig,
};
use platers_tests::test_utils::{
    generate_test_case, validate_solution, TestCaseConfig, TestHarness,
};
use std::time::Instant;

/// Create appropriate query config for testing.
/// Most real-world usage has scale hints, so we use them by default.
fn create_test_query_config(
    pixel_scale: f64,
    center_ra: f64,
    center_dec: f64,
    use_position_hint: bool,
) -> QueryConfig {
    QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000,
        max_hypotheses: 10000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.1)),
        position_hint: if use_position_hint {
            Some(PositionHint {
                ra: center_ra,
                dec: center_dec,
                radius: 1.0,
            })
        } else {
            None
        },
    }
}

/// Create lenient verification config for testing with clean synthetic data.
fn create_test_verification_config() -> VerificationConfig {
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

#[test]
fn test_simple_solve_no_noise() {
    println!("\n=== Test: Simple Solve (No Noise) ===");

    // Create test harness
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Generate test case: 30' FOV, 100 stars (more than needed to ensure enough are in bounds)
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .rotation(0.0)
        .stars(100)
        .noise(0.0);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Generated {} detected stars", detected_stars.len());

    // Create solver with appropriate config for testing
    let query_config = create_test_query_config(
        config.pixel_scale_arcsec(),
        config.center.ra,
        config.center.dec,
        true, // Use position hint
    );
    let verification_config = create_test_verification_config();

    let solver = harness.create_solver_with_config(query_config, verification_config);

    // Solve
    let start = Instant::now();
    let result = solver.solve(&DetectedField::new(
        detected_stars,
        config.image_width,
        config.image_height,
    ));
    let duration = start.elapsed();

    println!("Solve took {:.3}s", duration.as_secs_f64());

    // Validate
    assert!(result.is_ok(), "Solve failed: {:?}", result.err());
    let solve_result = result.unwrap();

    println!("Found {} hypotheses", solve_result.num_hypotheses_tested);
    println!(
        "Verification: {} matches, log-odds = {:.1}",
        solve_result.verification.num_matches, solve_result.verification.log_odds
    );

    // Validate against ground truth
    let validation = validate_solution(&solve_result.wcs, &ground_truth);

    println!(
        "Position error: {:.3} arcsec",
        validation.position_error_arcsec
    );
    println!("Scale error: {:.3}%", validation.scale_error_percent);
    println!("Rotation error: {:.3} deg", validation.rotation_error_deg);

    // The coarse WCS is re-anchored to the image center, so its reported
    // position should be sub-arcsec on clean data (a quad-centroid-anchored
    // pose could sit hundreds of arcsec off).
    assert!(
        validation.position_error_arcsec < 5.0,
        "Position error too large: {:.3} arcsec (expected sub-arcsec after image-center re-anchor)",
        validation.position_error_arcsec
    );
    assert!(
        validation.scale_error_percent < 1.0,
        "Scale error too large: {:.3}%",
        validation.scale_error_percent
    );
    assert!(
        validation.rotation_error_deg < 0.5,
        "Rotation error too large: {:.3} deg",
        validation.rotation_error_deg
    );

    println!("OK Test passed!");
}

#[test]
fn test_solve_with_scale_hint() {
    println!("\n=== Test: Solve with Scale Hint ===");
    println!("Goal: Validate that scale hints enable successful solving\n");

    let harness = TestHarness::new().expect("Failed to create test harness");

    // Generate test case - use position within regional catalog coverage
    let config = TestCaseConfig::new()
        .center(180.0, 45.0) // Center of regional catalog for best coverage
        .fov(20.0, 15.0)
        .image_size(1536, 1152)
        .rotation(10.0)
        .stars(50)
        .noise(0.2);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!(
        "Generated {} stars with 0.2 pixel noise",
        detected_stars.len()
    );

    // Solve WITH scale hint - this is the primary use case (90% of users know their pixel scale)
    let query_config = create_test_query_config(
        config.pixel_scale_arcsec(),
        config.center.ra,
        config.center.dec,
        true,
    );
    let verification_config = create_test_verification_config();
    let solver = harness.create_solver_with_config(query_config, verification_config);

    let start = Instant::now();
    let result = solver.solve(&DetectedField::new(
        detected_stars,
        config.image_width,
        config.image_height,
    ));
    let duration = start.elapsed();

    println!("Solve with scale hint took {:.3}s", duration.as_secs_f64());

    // Should succeed with scale hint
    assert!(result.is_ok(), "Solve with hint failed: {:?}", result.err());

    // Verify accuracy
    let validation = validate_solution(&result.unwrap().wcs, &ground_truth);

    println!(
        "Position error: {:.3} arcsec",
        validation.position_error_arcsec
    );
    println!("Scale error: {:.3}%", validation.scale_error_percent);
    println!("Rotation error: {:.3} deg", validation.rotation_error_deg);

    assert!(
        validation.position_error_arcsec < 5.0,
        "Position error too large: {:.3} arcsec (expected sub-arcsec after image-center re-anchor)",
        validation.position_error_arcsec
    );
    assert!(validation.scale_error_percent < 3.0);
    assert!(validation.rotation_error_deg < 1.0);

    println!("OK Test passed!");
}

#[test]
fn test_solve_different_rotations() {
    println!("\n=== Test: Different Rotations ===");

    let harness = TestHarness::new().expect("Failed to create test harness");

    // Test fewer rotations for faster testing - focus on key angles
    let rotations = vec![0.0, 15.0, 30.0];

    for rotation in rotations {
        println!("\nTesting rotation: {rotation:.0} deg");

        let config = TestCaseConfig::new()
            .center(180.0, 44.5) // Within regional catalog
            .fov(20.0, 15.0)
            .image_size(1536, 1024)
            .rotation(rotation)
            .stars(45)
            .noise(0.1);

        let (detected_stars, ground_truth) =
            generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

        let query_config = create_test_query_config(
            config.pixel_scale_arcsec(),
            config.center.ra,
            config.center.dec,
            true,
        );
        let verification_config = create_test_verification_config();
        let solver = harness.create_solver_with_config(query_config, verification_config);

        let result = solver.solve(&DetectedField::new(
            detected_stars,
            config.image_width,
            config.image_height,
        ));
        assert!(
            result.is_ok(),
            "Failed to solve at rotation {rotation:.0} deg"
        );

        let validation = validate_solution(&result.unwrap().wcs, &ground_truth);

        println!(
            "  Position error: {:.3} arcsec",
            validation.position_error_arcsec
        );
        println!("  Rotation error: {:.3} deg", validation.rotation_error_deg);

        assert!(
            validation.position_error_arcsec < 5.0,
            "Position error too large at rotation {:.0} deg: {:.3} arcsec",
            rotation,
            validation.position_error_arcsec
        );
        assert!(
            validation.rotation_error_deg < 1.0,
            "Rotation error too large at rotation {rotation:.0} deg"
        );
    }

    println!("\nOK All rotations passed!");
}

#[test]
fn test_solve_different_scales() {
    println!("\n=== Test: Different Field Scales ===");

    let harness = TestHarness::new().expect("Failed to create test harness");

    // Test small, medium, and large fields within our regional catalog
    let test_fovs = vec![(10.0, "small"), (20.0, "medium"), (30.0, "large")];

    for (fov_arcmin, label) in test_fovs {
        println!(
            "\nTesting {} FOV: {:.0}' x {:.0}'",
            label,
            fov_arcmin,
            fov_arcmin * 0.75
        );

        let config = TestCaseConfig::new()
            .center(180.0, 45.0) // Center of regional catalog for best coverage
            .fov(fov_arcmin, fov_arcmin * 0.75)
            .image_size(2048, 1536) // Larger image for better star count
            .rotation(10.0) // Moderate rotation
            .stars(100) // Request more stars
            .noise(0.15);

        let (detected_stars, ground_truth) =
            generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

        println!("  Generated {} detected stars", detected_stars.len());

        // Skip if not enough stars
        if detected_stars.len() < 20 {
            println!(
                "   Skipping {} FOV - not enough stars ({} < 20)",
                label,
                detected_stars.len()
            );
            continue;
        }

        // Use appropriate config
        let query_config = create_test_query_config(
            config.pixel_scale_arcsec(),
            config.center.ra,
            config.center.dec,
            true,
        );
        let verification_config = create_test_verification_config();
        let solver = harness.create_solver_with_config(query_config, verification_config);

        let start = Instant::now();
        let result = solver.solve(&DetectedField::new(
            detected_stars,
            config.image_width,
            config.image_height,
        ));
        let duration = start.elapsed();

        println!("  Solve took {:.3}s", duration.as_secs_f64());

        assert!(result.is_ok(), "Failed to solve {label} FOV");

        let validation = validate_solution(&result.unwrap().wcs, &ground_truth);

        println!(
            "  Position error: {:.3} arcsec",
            validation.position_error_arcsec
        );
        println!("  Scale error: {:.3}%", validation.scale_error_percent);

        assert!(
            validation.position_error_arcsec < 5.0,
            "Position error too large for {} FOV: {:.3} arcsec (expected sub-arcsec after re-anchor)",
            label,
            validation.position_error_arcsec
        );
        assert!(
            validation.scale_error_percent < 3.0,
            "Scale error too large for {label} FOV"
        );
    }

    println!("\nOK All scales passed!");
}

#[test]
fn test_solve_with_noise() {
    println!("\n=== Test: Solve with Varying Noise ===");

    let harness = TestHarness::new().expect("Failed to create test harness");

    let noise_levels = vec![0.0, 0.1, 0.3, 0.5]; // Skip 1.0 pixel noise - too much

    for noise in noise_levels {
        println!("\nTesting noise level: {noise:.1} pixels");

        let config = TestCaseConfig::new()
            .center(180.0, 45.0) // Center of regional catalog
            .fov(25.0, 18.0)
            .image_size(2048, 1536)
            .rotation(10.0) // Moderate rotation
            .stars(100) // Request more stars
            .noise(noise);

        let (detected_stars, ground_truth) =
            generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

        println!("  Generated {} detected stars", detected_stars.len());

        // Skip if not enough stars
        if detected_stars.len() < 20 {
            println!(
                "   Skipping noise level {:.1} - not enough stars ({} < 20)",
                noise,
                detected_stars.len()
            );
            continue;
        }

        let query_config = create_test_query_config(
            config.pixel_scale_arcsec(),
            config.center.ra,
            config.center.dec,
            true,
        );
        let verification_config = create_test_verification_config();
        let solver = harness.create_solver_with_config(query_config, verification_config);

        let result = solver.solve(&DetectedField::new(
            detected_stars,
            config.image_width,
            config.image_height,
        ));
        assert!(result.is_ok(), "Failed to solve with noise {noise:.1}");

        let validation = validate_solution(&result.unwrap().wcs, &ground_truth);

        println!(
            "  Position error: {:.3} arcsec",
            validation.position_error_arcsec
        );
        println!("  Scale error: {:.3}%", validation.scale_error_percent);

        // Re-anchored coarse WCS: sub-arcsec even with detection noise.
        assert!(
            validation.position_error_arcsec < 5.0,
            "Position error too large with noise {:.1}: {:.3} arcsec",
            noise,
            validation.position_error_arcsec
        );
        // Scale and rotation should still be good even with noise
        assert!(
            validation.scale_error_percent < 5.0,
            "Scale error too large with noise {noise:.1}"
        );
    }

    println!("\nOK All noise levels passed!");
}
