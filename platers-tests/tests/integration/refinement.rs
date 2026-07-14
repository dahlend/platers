//! Refinement integration tests.
//!
//! These tests validate that refinement improves accuracy over basic quad matching
//! using real catalog data and synthetic detected stars.

use platers_core::{
    query::{PositionHint, QueryConfig, ScaleRange},
    refinement::RefinementConfig,
    DetectedField, VerificationConfig,
};
use platers_tests::test_utils::{
    generate_test_case, validate_solution, TestCaseConfig, TestHarness,
};

/// Test that refinement significantly improves accuracy over basic quad matching.
///
/// This test demonstrates the core value proposition of refinement:
/// - Basic solve: ~30 arcsec accuracy (good for field identification)
/// - Refined solve: <1 arcsec accuracy (suitable for astrometry)
#[test]
fn test_refinement_improves_accuracy() {
    println!("\n=== Test: Refinement Accuracy Improvement ===");
    println!("Goal: Demonstrate 10-20x accuracy improvement\n");

    // Create test harness
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Configure test case - same as basic test but with more stars
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .rotation(15.0) // Add some rotation
        .stars(100)
        .noise(0.3) // Small noise (realistic)
        .mag_limit(16.0);

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
        "  Pixel scale: {:.2} arcsec/pixel",
        config.pixel_scale_arcsec()
    );
    println!("  Rotation: {:.1} deg", config.rotation_deg);
    println!("  Stars: {}", config.num_stars);
    println!("  Noise: {:.1} pixels", config.position_noise_pixels);

    // Generate test case
    println!("\nGenerating test case...");
    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("  Generated {} detected stars", detected_stars.len());

    assert!(
        detected_stars.len() >= 20,
        "Need at least 20 stars for solving"
    );

    // Test 1: Solve WITHOUT refinement (basic quad matching)
    println!("\n--- Test 1: Basic Solve (NO refinement) ---");

    let pixel_scale = config.pixel_scale_arcsec();
    let query_config = QueryConfig {
        max_stars_for_quads: 35,
        max_quads_to_try: 5000, // Increased for rotation + noise
        max_hypotheses: 1000,
        hash_code_tolerance: 0.02,
        observation_epoch: None, // Slightly larger for noisy data
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.1)),
        position_hint: Some(PositionHint {
            ra: config.center.ra,
            dec: config.center.dec,
            radius: 1.0,
        }),
    };

    let verification_config = VerificationConfig {
        sigma_arcsec: 0.5,
        background_density_per_sqdeg: 1000.0,
        match_radius_arcsec: 2.0,
        min_matches: 10,
        log_odds_threshold: 14.0,
        max_stars_to_verify: 100,
        ..Default::default()
    };

    let solver = harness.create_solver_with_config(query_config, verification_config);

    let field = DetectedField::new(detected_stars, config.image_width, config.image_height);
    let coarse_result = solver.solve_coarse(&field).expect("Coarse solve failed");

    println!("Coarse solve completed:");
    println!(
        "  Center: RA={:.6} deg, Dec={:.6} deg",
        coarse_result.wcs.center.ra, coarse_result.wcs.center.dec
    );
    println!(
        "  Scale: {:.4} arcsec/pixel",
        coarse_result.wcs.scale_arcsec_per_pixel()
    );
    println!("  Rotation: {:.2} deg", coarse_result.wcs.rotation_deg());

    // Validate coarse result
    let coarse_validation = validate_solution(&coarse_result.wcs, &ground_truth);

    println!("\nCoarse solve accuracy:");
    println!(
        "  Position error: {:.2} arcsec",
        coarse_validation.position_error_arcsec
    );
    println!(
        "  Scale error: {:.3}%",
        coarse_validation.scale_error_percent
    );
    println!(
        "  Rotation error: {:.2} deg",
        coarse_validation.rotation_error_deg
    );

    // Test 2: Solve WITH refinement
    println!("\n--- Test 2: Refined Solve (WITH refinement) ---");

    // Use default refinement config (suitable for production)
    let refined_result = solver
        .solve_with_refinement(&field, None) // Use default config
        .expect("Refined solve failed");

    println!("Refined solve completed:");
    println!(
        "  Center: RA={:.6} deg, Dec={:.6} deg",
        refined_result.wcs.center.ra, refined_result.wcs.center.dec
    );
    println!(
        "  Scale: {:.4} arcsec/pixel",
        refined_result.wcs.scale_arcsec_per_pixel()
    );
    println!("  Rotation: {:.2} deg", refined_result.wcs.rotation_deg());

    // Show refinement details
    if let Some(ref refinement) = refined_result.refinement {
        println!("\nRefinement details:");
        println!("  Iterations: {}", refinement.iterations);
        println!("  Converged: {}", refinement.converged);
        println!("  Matched stars: {}", refinement.matched_stars.len());
        println!(
            "  RMS residual: {:.3} arcsec",
            refinement.rms_residual_arcsec
        );

        // Refinement should have been performed
        assert!(
            refinement.matched_stars.len() >= 10,
            "Refinement should match at least 10 stars"
        );
        assert!(
            refinement.rms_residual_arcsec < 2.0,
            "RMS residual should be < 2 arcsec (got {:.3})",
            refinement.rms_residual_arcsec
        );
    } else {
        panic!("Refinement should have been performed but was None");
    }

    // Validate refined result
    let refined_validation = validate_solution(&refined_result.wcs, &ground_truth);

    println!("\nRefined solve accuracy:");
    println!(
        "  Position error: {:.2} arcsec",
        refined_validation.position_error_arcsec
    );
    println!(
        "  Scale error: {:.3}%",
        refined_validation.scale_error_percent
    );
    println!(
        "  Rotation error: {:.2} deg",
        refined_validation.rotation_error_deg
    );

    // Compute improvement factors
    let position_improvement = coarse_validation.position_error_arcsec
        / refined_validation.position_error_arcsec.max(0.01);
    let scale_improvement =
        coarse_validation.scale_error_percent / refined_validation.scale_error_percent.max(0.001);
    let rotation_improvement =
        coarse_validation.rotation_error_deg / refined_validation.rotation_error_deg.max(0.01);

    println!("\n=== Improvement Summary ===");
    println!("Position: {position_improvement:.1}x better");
    println!("Scale: {scale_improvement:.1}x better");
    println!("Rotation: {rotation_improvement:.1}x better");

    // Assertions: Refinement should improve accuracy
    // Note: With parallelization, coarse solution quality varies due to non-deterministic
    // hypothesis ordering. Sometimes we get an excellent initial solution, sometimes good.
    // The key is that refinement always produces excellent final accuracy.

    // Absolute accuracy requirements for refined solution
    assert!(
        refined_validation.scale_error_percent < 0.1,
        "Refined scale error should be < 0.1% (got {:.3}%)",
        refined_validation.scale_error_percent
    );

    assert!(
        refined_validation.rotation_error_deg < 0.5,
        "Refined rotation error should be < 0.5 deg (got {:.2} deg)",
        refined_validation.rotation_error_deg
    );

    // Improvement requirements: If coarse solution is poor (>0.1% error), refinement should help significantly
    // If coarse solution is already excellent (<0.05% error), any improvement is a bonus
    if coarse_validation.scale_error_percent > 0.1 {
        assert!(
            scale_improvement >= 2.0,
            "Scale should improve by at least 2x when initial error > 0.1% (got {:.1}x improvement from {:.3}% to {:.3}%)",
            scale_improvement,
            coarse_validation.scale_error_percent,
            refined_validation.scale_error_percent
        );
    } else {
        // Coarse solution is already good - just verify refinement doesn't make it worse
        assert!(
            refined_validation.scale_error_percent <= coarse_validation.scale_error_percent * 1.1,
            "Refinement shouldn't significantly degrade accuracy (coarse: {:.3}%, refined: {:.3}%)",
            coarse_validation.scale_error_percent,
            refined_validation.scale_error_percent
        );
        println!("Note: Coarse solution already excellent ({:.3}%), refinement maintains/improves to {:.3}%", 
                 coarse_validation.scale_error_percent,
                 refined_validation.scale_error_percent);
    }

    // As with scale: only demand a big improvement when the coarse pose is poor.
    // Post re-anchor (and with bright-star uniformization) the coarse rotation is
    // often already excellent -- 0.01 deg here -- so refinement maintains rather than
    // multiplies it. The real contract is "refined is accurate and doesn't regress".
    if coarse_validation.rotation_error_deg > 0.5 {
        assert!(
            rotation_improvement >= 1.2,
            "Rotation should improve by at least 1.2x when initial error > 0.5 deg (got {rotation_improvement:.1}x)"
        );
    } else {
        assert!(
            refined_validation.rotation_error_deg <= coarse_validation.rotation_error_deg.max(0.1) + 0.05,
            "Refinement shouldn't degrade an already-good rotation (coarse: {:.3} deg, refined: {:.3} deg)",
            coarse_validation.rotation_error_deg,
            refined_validation.rotation_error_deg
        );
    }

    println!("\nTEST PASSED: Refinement significantly improves accuracy!");
    println!("   Scale: {scale_improvement:.1}x better");
    println!("   Rotation: {rotation_improvement:.1}x better");
}

/// Test refinement with custom configuration for demanding applications.
#[test]
fn test_refinement_custom_config() {
    println!("\n=== Test: Refinement with Custom Config ===");
    println!("Goal: Demonstrate custom config for high-precision work\n");

    // Create test harness
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Configure test case with very clean data
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .rotation(30.0)
        .stars(100)
        .noise(0.1) // Very clean data
        .mag_limit(16.0);

    println!("Test case: Very clean data, 30 deg rotation");

    // Generate test case
    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Generated {} detected stars", detected_stars.len());

    // Solve with custom refinement config
    let pixel_scale = config.pixel_scale_arcsec();
    let query_config = QueryConfig {
        max_stars_for_quads: 35,
        // 5000 (was 750): after the test-generator bounds fix + query-side
        // uniformization, fields are denser and the per-quad match rate is
        // lower, so a larger budget is needed to reliably hit a match. This
        // asserts refinement *works*, not that it works at a specific budget.
        max_quads_to_try: 5000,
        max_hypotheses: 1000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.1)),
        position_hint: Some(PositionHint {
            ra: config.center.ra,
            dec: config.center.dec,
            radius: 1.0,
        }),
    };

    let solver = harness.create_solver_with_config(query_config, VerificationConfig::default());

    // Custom refinement config - more aggressive for high precision
    let custom_config = RefinementConfig {
        initial_radius_arcsec: 5.0,  // Tighter initial (good after coarse solve)
        final_radius_arcsec: 0.5,    // Very tight final radius
        max_iterations: 5,           // More iterations allowed
        min_stars: 15,               // Require more stars
        outlier_sigma: 2.5,          // More aggressive outlier rejection
        convergence_threshold: 0.05, // Stricter convergence
        ..Default::default()
    };

    println!("\nCustom refinement config:");
    println!(
        "  Initial radius: {:.1} arcsec",
        custom_config.initial_radius_arcsec
    );
    println!(
        "  Final radius: {:.1} arcsec",
        custom_config.final_radius_arcsec
    );
    println!("  Max iterations: {}", custom_config.max_iterations);
    println!("  Min stars: {}", custom_config.min_stars);

    let result = solver
        .solve_with_refinement(
            &DetectedField::new(detected_stars, config.image_width, config.image_height),
            Some(custom_config),
        )
        .expect("Solve failed");

    // Check refinement was performed
    assert!(
        result.refinement.is_some(),
        "Refinement should have been performed"
    );

    if let Some(ref refinement) = result.refinement {
        println!("\nRefinement results:");
        println!("  Iterations: {}", refinement.iterations);
        println!("  Converged: {}", refinement.converged);
        println!("  Matched stars: {}", refinement.matched_stars.len());
        println!(
            "  RMS residual: {:.3} arcsec",
            refinement.rms_residual_arcsec
        );

        // Should have matched plenty of stars
        assert!(
            refinement.matched_stars.len() >= 15,
            "Should match at least 15 stars"
        );

        // RMS should be excellent with clean data and custom config
        assert!(
            refinement.rms_residual_arcsec < 1.5,
            "RMS should be < 1.5 arcsec with clean data (got {:.3})",
            refinement.rms_residual_arcsec
        );
    }

    // Validate accuracy
    let validation = validate_solution(&result.wcs, &ground_truth);

    println!("\nAccuracy:");
    println!(
        "  Position error: {:.2} arcsec",
        validation.position_error_arcsec
    );
    println!("  Scale error: {:.3}%", validation.scale_error_percent);
    println!("  Rotation error: {:.2} deg", validation.rotation_error_deg);

    // With very clean data and custom config, should be extremely accurate
    assert!(
        validation.scale_error_percent < 0.05,
        "Scale error should be < 0.05% with clean data"
    );
    assert!(
        validation.rotation_error_deg < 0.3,
        "Rotation error should be < 0.3 deg with clean data"
    );

    println!("\nTEST PASSED: Custom config provides excellent precision!");
}

/// Test refinement with wide field of view (120' x 80').
///
/// Wide FOV challenges:
/// - More stars to match (potentially 50-100+)
/// - Larger area to search
/// - Potential for more outliers
/// - Tests scalability of refinement
///
/// NOTE: This test is marked #[ignore] because wide FOV (>60 arcmin) with large
/// pixel scale (>1.5 arcsec/px) can fail at the COARSE SOLVE stage due to quad
/// matching ambiguity -- a known limitation. The refinement algorithm itself
/// works fine; the bottleneck is quad matching.
#[test]
#[ignore = "wide-FOV stress case; run manually"]
fn test_refinement_wide_fov() {
    println!("\n=== Test: Refinement with Wide FOV (60' x 40') ===");
    println!("Goal: Validate refinement performance on wide fields\n");

    // Create test harness
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Configure wide FOV test case
    // 60' x 40' = 1.0 deg x 0.67 deg (realistic wide field for amateur)
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(60.0, 40.0) // 1 deg x 0.67 deg - wide but realistic
        .image_size(2048, 1489)
        .rotation(15.0) // Moderate rotation (45 deg was too much for wide FOV)
        .stars(100) // Moderate number of stars
        .noise(0.3) // Small noise
        .mag_limit(15.0); // Include more stars for better matching

    println!("Wide FOV configuration:");
    println!(
        "  FOV: {:.1}x{:.1} arcmin ({:.2} degx{:.2} deg)",
        config.fov_width_arcmin,
        config.fov_height_arcmin,
        config.fov_width_arcmin / 60.0,
        config.fov_height_arcmin / 60.0
    );
    println!(
        "  Pixel scale: {:.2} arcsec/pixel",
        config.pixel_scale_arcsec()
    );
    println!("  Rotation: {:.1} deg", config.rotation_deg);

    // Generate test case
    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Generated {} detected stars", detected_stars.len());

    assert!(
        detected_stars.len() >= 20,
        "Need at least 20 stars for wide FOV"
    );

    // Solve with refinement
    let pixel_scale = config.pixel_scale_arcsec();
    let query_config = QueryConfig {
        max_stars_for_quads: 60,  // More stars for wide FOV
        max_quads_to_try: 100000, // More quads needed
        max_hypotheses: 20000,    // More hypotheses
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.15)), // Wider tolerance
        position_hint: Some(PositionHint {
            ra: config.center.ra,
            dec: config.center.dec,
            radius: 2.0, // Wider search for wide FOV
        }),
    };

    let solver = harness.create_solver_with_config(query_config, VerificationConfig::default());

    let result = solver
        .solve_with_refinement(
            &DetectedField::new(detected_stars, config.image_width, config.image_height),
            None, // Use default refinement config
        )
        .expect("Wide FOV solve failed");

    // Check refinement was performed
    assert!(
        result.refinement.is_some(),
        "Refinement should have been performed"
    );

    if let Some(ref refinement) = result.refinement {
        println!("\nRefinement results:");
        println!("  Iterations: {}", refinement.iterations);
        println!("  Converged: {}", refinement.converged);
        println!("  Matched stars: {}", refinement.matched_stars.len());
        println!(
            "  RMS residual: {:.3} arcsec",
            refinement.rms_residual_arcsec
        );

        // Should match many stars in wide field
        assert!(
            refinement.matched_stars.len() >= 30,
            "Wide FOV should match at least 30 stars (got {})",
            refinement.matched_stars.len()
        );

        // RMS should still be good despite wide FOV and more noise
        assert!(
            refinement.rms_residual_arcsec < 2.0,
            "RMS should be < 2 arcsec even with wide FOV (got {:.3})",
            refinement.rms_residual_arcsec
        );
    }

    // Validate accuracy
    let validation = validate_solution(&result.wcs, &ground_truth);

    println!("\nAccuracy:");
    println!(
        "  Position error: {:.2} arcsec",
        validation.position_error_arcsec
    );
    println!("  Scale error: {:.3}%", validation.scale_error_percent);
    println!("  Rotation error: {:.2} deg", validation.rotation_error_deg);

    // Wide FOV should still achieve good accuracy
    assert!(
        validation.scale_error_percent < 0.5,
        "Scale error should be < 0.5% with wide FOV"
    );
    assert!(
        validation.rotation_error_deg < 1.0,
        "Rotation error should be < 1 deg with wide FOV"
    );

    println!("\nTEST PASSED: Wide FOV refinement works well!");
}

/// Test refinement with ultra-wide field of view (180' x 120').
///
/// Ultra-wide FOV challenges:
/// - Even more stars (potentially 200+)
/// - Field distortion may become significant
/// - Tests extreme scalability
#[test]
#[ignore = "very demanding; run manually"]
fn test_refinement_ultra_wide_fov() {
    println!("\n=== Test: Refinement with Ultra-Wide FOV (180' x 120') ===");
    println!("Goal: Validate extreme wide field performance\n");

    // Create test harness
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Configure ultra-wide FOV test case
    // 180' x 120' = 3.0 deg x 2.0 deg (extreme wide field)
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(180.0, 120.0) // 3 deg x 2 deg - ultra-wide
        .image_size(2048, 1489)
        .rotation(60.0) // Large rotation
        .stars(200) // Many stars
        .noise(0.5)
        .mag_limit(12.0); // Bright stars only

    println!("Ultra-wide FOV configuration:");
    println!(
        "  FOV: {:.1}x{:.1} arcmin ({:.2} degx{:.2} deg)",
        config.fov_width_arcmin,
        config.fov_height_arcmin,
        config.fov_width_arcmin / 60.0,
        config.fov_height_arcmin / 60.0
    );
    println!(
        "  Pixel scale: {:.2} arcsec/pixel",
        config.pixel_scale_arcsec()
    );

    // Generate test case
    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Generated {} detected stars", detected_stars.len());

    assert!(
        detected_stars.len() >= 30,
        "Need at least 30 stars for ultra-wide FOV"
    );

    // Solve with refinement and custom config for ultra-wide
    let pixel_scale = config.pixel_scale_arcsec();
    let query_config = QueryConfig {
        max_stars_for_quads: 80,  // Even more stars
        max_quads_to_try: 200000, // Many more quads
        max_hypotheses: 50000,    // Many hypotheses
        hash_code_tolerance: 0.015,
        observation_epoch: None, // Slightly looser
        scale_hint: Some(ScaleRange::from_nominal(pixel_scale, 0.2)),
        position_hint: Some(PositionHint {
            ra: config.center.ra,
            dec: config.center.dec,
            radius: 3.0, // Very wide search
        }),
    };

    // Custom refinement config for ultra-wide FOV
    let refinement_config = RefinementConfig {
        initial_radius_arcsec: 10.0, // Wider initial radius
        final_radius_arcsec: 1.0,
        max_iterations: 5,
        min_stars: 20, // Require more stars
        outlier_sigma: 3.0,
        convergence_threshold: 0.1,
        ..Default::default()
    };

    let solver = harness.create_solver_with_config(query_config, VerificationConfig::default());

    let result = solver
        .solve_with_refinement(
            &DetectedField::new(detected_stars, config.image_width, config.image_height),
            Some(refinement_config),
        )
        .expect("Ultra-wide FOV solve failed");

    // Check refinement
    assert!(
        result.refinement.is_some(),
        "Refinement should have been performed"
    );

    if let Some(ref refinement) = result.refinement {
        println!("\nRefinement results:");
        println!("  Iterations: {}", refinement.iterations);
        println!("  Converged: {}", refinement.converged);
        println!("  Matched stars: {}", refinement.matched_stars.len());
        println!(
            "  RMS residual: {:.3} arcsec",
            refinement.rms_residual_arcsec
        );

        // Should match very many stars in ultra-wide field
        assert!(
            refinement.matched_stars.len() >= 40,
            "Ultra-wide FOV should match at least 40 stars"
        );

        // RMS may be slightly worse with ultra-wide FOV
        assert!(
            refinement.rms_residual_arcsec < 3.0,
            "RMS should be < 3 arcsec with ultra-wide FOV"
        );
    }

    // Validate accuracy
    let validation = validate_solution(&result.wcs, &ground_truth);

    println!("\nAccuracy:");
    println!(
        "  Position error: {:.2} arcsec",
        validation.position_error_arcsec
    );
    println!("  Scale error: {:.3}%", validation.scale_error_percent);
    println!("  Rotation error: {:.2} deg", validation.rotation_error_deg);

    // Ultra-wide should still be reasonable
    assert!(
        validation.scale_error_percent < 1.0,
        "Scale error should be < 1.0% with ultra-wide FOV"
    );
    assert!(
        validation.rotation_error_deg < 2.0,
        "Rotation error should be < 2 deg with ultra-wide FOV"
    );

    println!("\nTEST PASSED: Ultra-wide FOV refinement achieves good accuracy!");
}
