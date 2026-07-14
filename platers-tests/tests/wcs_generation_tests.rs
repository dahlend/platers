//! Test WCS generation from quad matches

use platers_core::{wcs::WcsHypothesis, PixelCoord};
use platers_tests::test_utils::{generate_test_case, TestCaseConfig, TestHarness};

#[test]
fn test_wcs_from_generated_quads() {
    println!("\n=== Test: WCS from Generated Quads ===\n");

    let harness = TestHarness::new().expect("Failed to create harness");

    // Generate test case
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .stars(100)
        .noise(0.0)
        .mag_limit(16.0);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Ground truth WCS:");
    println!(
        "  Center: RA={:.6} deg, Dec={:.6} deg",
        ground_truth.wcs.center.ra, ground_truth.wcs.center.dec
    );
    println!(
        "  Scale: {:.6} arcsec/pixel",
        ground_truth.wcs.scale_arcsec_per_pixel()
    );
    println!("  Rotation: {:.6} deg", ground_truth.wcs.rotation_deg());
    println!(
        "  Ref pixel: ({:.2}, {:.2})",
        ground_truth.wcs.reference_pixel.x, ground_truth.wcs.reference_pixel.y
    );

    // Take first 4 detected stars and create a quad
    assert!(detected_stars.len() >= 4, "Not enough detected stars");

    let image_stars = [
        PixelCoord::new(detected_stars[0].x, detected_stars[0].y),
        PixelCoord::new(detected_stars[1].x, detected_stars[1].y),
        PixelCoord::new(detected_stars[2].x, detected_stars[2].y),
        PixelCoord::new(detected_stars[3].x, detected_stars[3].y),
    ];

    // Get the catalog stars that match these detected stars
    let catalog_stars = [
        ground_truth.true_positions[0],
        ground_truth.true_positions[1],
        ground_truth.true_positions[2],
        ground_truth.true_positions[3],
    ];

    println!("\nQuad correspondence:");
    for i in 0..4 {
        println!(
            "  Star {}: pixel=({:.2}, {:.2}) -> sky=({:.6} deg, {:.6} deg)",
            i, image_stars[i].x, image_stars[i].y, catalog_stars[i].ra, catalog_stars[i].dec
        );
    }

    // Generate WCS from this quad match
    let computed_wcs = WcsHypothesis::from_quad_match(
        &image_stars,
        &catalog_stars,
        config.image_width,
        config.image_height,
    )
    .expect("Failed to compute WCS from quad match");

    println!("\nComputed WCS from quad:");
    println!(
        "  Center: RA={:.6} deg, Dec={:.6} deg",
        computed_wcs.center.ra, computed_wcs.center.dec
    );
    println!(
        "  Scale: {:.6} arcsec/pixel",
        computed_wcs.scale_arcsec_per_pixel()
    );
    println!("  Rotation: {:.6} deg", computed_wcs.rotation_deg());
    println!(
        "  Ref pixel: ({:.2}, {:.2})",
        computed_wcs.reference_pixel.x, computed_wcs.reference_pixel.y
    );

    // Compare
    let center_error = ground_truth
        .wcs
        .center
        .angular_distance(&computed_wcs.center)
        * 3600.0;
    let scale_error = ((computed_wcs.scale_arcsec_per_pixel()
        - ground_truth.wcs.scale_arcsec_per_pixel())
        / ground_truth.wcs.scale_arcsec_per_pixel()
        * 100.0)
        .abs();
    let mut rotation_error = (computed_wcs.rotation_deg() - ground_truth.wcs.rotation_deg()).abs();
    if rotation_error > 180.0 {
        rotation_error = 360.0 - rotation_error;
    }

    println!("\nErrors:");
    println!("  Center: {center_error:.4} arcsec");
    println!("  Scale: {scale_error:.4}%");
    println!("  Rotation: {rotation_error:.4} deg");

    // Now test if this computed WCS can verify correctly
    let verifier = platers_core::verification::Verifier::new(platers_core::VerificationConfig {
        sigma_arcsec: 2.0,
        background_density_per_sqdeg: 1000.0,
        match_radius_arcsec: 5.0,
        min_matches: 4,
        log_odds_threshold: 10.0,
        max_stars_to_verify: 100,
        ..Default::default()
    });

    let catalog_index = platers_core::CatalogIndex::new(harness.catalog().to_vec());
    let result = verifier.verify(&computed_wcs, &detected_stars, &catalog_index);

    println!("\nVerification of computed WCS:");
    println!(
        "  Matches: {}/{}",
        result.num_matches, result.num_stars_checked
    );
    println!("  Log-odds: {:.2}", result.log_odds);
    println!("  Passes: {}", result.passes_threshold);
    if result.num_matches > 0 {
        println!(
            "  Mean distance: {:.4} arcsec",
            result.mean_match_distance()
        );
    }

    println!("\nTest complete");
}
