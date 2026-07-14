//! Test verification with known good WCS

use platers_core::{verification::Verifier, CatalogIndex, VerificationConfig};
use platers_tests::test_utils::{generate_test_case, TestCaseConfig, TestHarness};

#[test]
fn test_verify_ground_truth_wcs() {
    println!("\n=== Test: Verify Ground Truth WCS ===\n");

    let harness = TestHarness::new().expect("Failed to create harness");

    // Generate test case
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .stars(100)
        .noise(0.0) // No noise for perfect match
        .mag_limit(16.0);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Generated {} detected stars", detected_stars.len());
    println!("Ground truth WCS:");
    println!(
        "  Center: RA={:.6} deg, Dec={:.6} deg",
        ground_truth.wcs.center.ra, ground_truth.wcs.center.dec
    );
    println!(
        "  Scale: {:.4} arcsec/pixel",
        ground_truth.wcs.scale_arcsec_per_pixel()
    );

    // Try different verification configs
    let configs = vec![
        (
            "Very Lenient",
            VerificationConfig {
                sigma_arcsec: 5.0,
                background_density_per_sqdeg: 1000.0,
                match_radius_arcsec: 10.0,
                min_matches: 3,
                log_odds_threshold: 5.0,
                max_stars_to_verify: 100,
                ..Default::default()
            },
        ),
        (
            "Lenient",
            VerificationConfig {
                sigma_arcsec: 2.0,
                background_density_per_sqdeg: 1000.0,
                match_radius_arcsec: 5.0,
                min_matches: 4,
                log_odds_threshold: 10.0,
                max_stars_to_verify: 100,
                ..Default::default()
            },
        ),
        ("Default", VerificationConfig::default()),
    ];

    for (name, ver_config) in configs {
        println!("\n--- Testing: {name} ---");
        println!("  sigma: {:.2} arcsec", ver_config.sigma_arcsec);
        println!(
            "  match_radius: {:.2} arcsec",
            ver_config.match_radius_arcsec
        );
        println!("  threshold: {:.2}", ver_config.log_odds_threshold);

        let verifier = Verifier::new(ver_config);

        // Verify the ground truth WCS
        let catalog = CatalogIndex::new(harness.catalog().to_vec());
        let result = verifier.verify(&ground_truth.wcs, &detected_stars, &catalog);

        println!("\n  Results:");
        println!(
            "    Matches: {}/{}",
            result.num_matches, result.num_stars_checked
        );
        println!("    Log-odds: {:.2}", result.log_odds);
        println!("    Passes threshold: {}", result.passes_threshold);
        println!("    Match rate: {:.1}%", result.match_rate() * 100.0);

        if !result.match_distances.is_empty() {
            println!(
                "    Mean distance: {:.4} arcsec",
                result.mean_match_distance()
            );
            println!(
                "    Max distance: {:.4} arcsec",
                result
                    .match_distances
                    .iter()
                    .fold(0.0_f64, |a, &b| a.max(b))
            );
        }
    }

    println!("\nVerification test complete");
}

#[test]
fn test_verify_slightly_wrong_wcs() {
    println!("\n=== Test: Verify Slightly Wrong WCS ===\n");

    let harness = TestHarness::new().expect("Failed to create harness");

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(30.0, 20.0)
        .image_size(2048, 1489)
        .stars(100)
        .noise(0.0)
        .mag_limit(16.0);

    let (detected_stars, ground_truth) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    println!("Testing WCS with small errors:");

    // Test WCS with slightly wrong center
    let mut wrong_wcs = ground_truth.wcs.clone();
    wrong_wcs.center.ra += 0.01; // 0.01 degree = 36 arcsec error

    let verifier = Verifier::new(VerificationConfig {
        sigma_arcsec: 2.0,
        background_density_per_sqdeg: 1000.0,
        match_radius_arcsec: 5.0,
        min_matches: 4,
        log_odds_threshold: 10.0,
        max_stars_to_verify: 100,
        ..Default::default()
    });

    let catalog = CatalogIndex::new(harness.catalog().to_vec());
    let result = verifier.verify(&wrong_wcs, &detected_stars, &catalog);

    println!("  RA offset: +0.01 deg (+36 arcsec)");
    println!(
        "  Matches: {}/{}",
        result.num_matches, result.num_stars_checked
    );
    println!("  Log-odds: {:.2}", result.log_odds);
    println!("  Passes: {}", result.passes_threshold);

    println!("\nTest complete");
}
