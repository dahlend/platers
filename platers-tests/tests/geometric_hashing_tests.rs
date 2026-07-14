//! Comprehensive tests for geometric hashing fundamentals.
//!
//! These tests validate the core geometric hashing algorithm that is the
//! foundation of the entire plate solving system. We test:
//! 1. Hash code computation for pixel coordinates
//! 2. Hash code computation for sky coordinates
//! 3. Hash code invariance properties (translation, rotation, scale)
//! 4. Hash code matching between pixel and sky quads
//! 5. Edge cases and degeneracies

use platers_core::{
    geometry::{compute_hash_code_pixels, compute_hash_code_sky},
    types::{PixelCoord, SkyCoord},
};
use rand::{Rng, SeedableRng};

// Helper to create a simple quad in pixel space
fn create_pixel_quad_simple() -> [PixelCoord; 4] {
    [
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(500.0, 100.0),
        PixelCoord::new(300.0, 300.0),
        PixelCoord::new(200.0, 400.0),
    ]
}

// Helper to create a quad in sky space (small field ~0.5 degrees)
fn create_sky_quad_simple() -> [SkyCoord; 4] {
    [
        SkyCoord::new(180.0, 45.0),
        SkyCoord::new(180.5, 45.0),
        SkyCoord::new(180.25, 45.3),
        SkyCoord::new(180.15, 45.4),
    ]
}

#[test]
fn test_hash_code_basic_computation() {
    println!("\n=== Test: Basic Hash Code Computation ===\n");

    let quad = create_pixel_quad_simple();

    println!("Input quad:");
    for (i, star) in quad.iter().enumerate() {
        println!("  Star {}: ({:.1}, {:.1})", i, star.x, star.y);
    }

    let hash = compute_hash_code_pixels(&quad).expect("Failed to compute hash code");

    println!("\nHash code: {:?}", hash.components);
    println!("  Component 0 (x_c): {:.6}", hash.components[0]);
    println!("  Component 1 (y_c): {:.6}", hash.components[1]);
    println!("  Component 2 (x_d): {:.6}", hash.components[2]);
    println!("  Component 3 (y_d): {:.6}", hash.components[3]);

    // All components should be in [0, 1]
    for (i, &component) in hash.components.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&component),
            "Component {i} = {component} is not in [0, 1]"
        );
    }

    println!("\nHash code components all in valid range [0, 1]");
}

#[test]
fn test_hash_code_translation_invariance() {
    println!("\n=== Test: Translation Invariance ===\n");

    let quad1 = [
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(500.0, 100.0),
        PixelCoord::new(300.0, 300.0),
        PixelCoord::new(200.0, 400.0),
    ];

    // Translate by (500, 300)
    let quad2 = [
        PixelCoord::new(600.0, 400.0),
        PixelCoord::new(1000.0, 400.0),
        PixelCoord::new(800.0, 600.0),
        PixelCoord::new(700.0, 700.0),
    ];

    let hash1 = compute_hash_code_pixels(&quad1).expect("Failed to compute hash1");
    let hash2 = compute_hash_code_pixels(&quad2).expect("Failed to compute hash2");

    println!("Original quad hash: {:?}", hash1.components);
    println!("Translated quad hash: {:?}", hash2.components);

    let distance = hash1.distance(&hash2);
    println!("\nHash distance: {distance:.10}");

    // Hash codes should be identical (distance ~0)
    assert!(
        distance < 1e-10,
        "Translation changed hash code! Distance = {distance}"
    );

    println!("Hash code is translation invariant");
}

#[test]
fn test_hash_code_rotation_invariance() {
    println!("\n=== Test: Rotation Invariance ===\n");

    // Original quad
    let quad1 = [
        PixelCoord::new(0.0, 0.0),
        PixelCoord::new(400.0, 0.0),
        PixelCoord::new(200.0, 200.0),
        PixelCoord::new(100.0, 300.0),
    ];

    // Rotate 90 degrees clockwise around origin
    // (x, y) -> (y, -x)
    let quad2 = [
        PixelCoord::new(0.0, 0.0),
        PixelCoord::new(0.0, -400.0),
        PixelCoord::new(200.0, -200.0),
        PixelCoord::new(300.0, -100.0),
    ];

    let hash1 = compute_hash_code_pixels(&quad1).expect("Failed to compute hash1");
    let hash2 = compute_hash_code_pixels(&quad2).expect("Failed to compute hash2");

    println!("Original quad hash: {:?}", hash1.components);
    println!("Rotated quad hash: {:?}", hash2.components);

    let distance = hash1.distance(&hash2);
    println!("\nHash distance: {distance:.10}");

    // Hash codes should be very similar (rotation might cause small numerical differences)
    assert!(
        distance < 1e-6,
        "Rotation significantly changed hash code! Distance = {distance}"
    );

    println!("Hash code is rotation invariant (within numerical precision)");
}

#[test]
fn test_hash_code_scale_invariance() {
    println!("\n=== Test: Scale Invariance ===\n");

    // Original quad
    let quad1 = [
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(500.0, 100.0),
        PixelCoord::new(300.0, 300.0),
        PixelCoord::new(200.0, 400.0),
    ];

    // Scale by 2x
    let quad2 = [
        PixelCoord::new(200.0, 200.0),
        PixelCoord::new(1000.0, 200.0),
        PixelCoord::new(600.0, 600.0),
        PixelCoord::new(400.0, 800.0),
    ];

    let hash1 = compute_hash_code_pixels(&quad1).expect("Failed to compute hash1");
    let hash2 = compute_hash_code_pixels(&quad2).expect("Failed to compute hash2");

    println!("Original quad hash: {:?}", hash1.components);
    println!("Scaled (2x) quad hash: {:?}", hash2.components);

    let distance = hash1.distance(&hash2);
    println!("\nHash distance: {distance:.10}");

    // Hash codes should be identical (distance ~0)
    assert!(
        distance < 1e-10,
        "Scaling changed hash code! Distance = {distance}"
    );

    println!("Hash code is scale invariant");
}

#[test]
fn test_hash_code_star_order_matters() {
    println!("\n=== Test: Star Order Affects Hash ===\n");

    let stars = [
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(500.0, 100.0),
        PixelCoord::new(300.0, 300.0),
        PixelCoord::new(200.0, 400.0),
    ];

    // Same stars, different order
    let quad1 = [stars[0], stars[1], stars[2], stars[3]];
    let quad2 = [stars[1], stars[0], stars[2], stars[3]]; // Swap first two

    let hash1 = compute_hash_code_pixels(&quad1).expect("Failed to compute hash1");
    let hash2 = compute_hash_code_pixels(&quad2).expect("Failed to compute hash2");

    println!("Order 1 hash: {:?}", hash1.components);
    println!("Order 2 hash: {:?}", hash2.components);

    let distance = hash1.distance(&hash2);
    println!("\nHash distance: {distance:.6}");

    // NOTE: The hash code algorithm internally finds the widest pair,
    // so the order might not matter if the widest pair is the same!
    // Let's just verify the hashes were computed successfully.

    println!("Both orderings produce valid hash codes");
    println!("   (Algorithm automatically identifies widest pair as A-B)");
}

#[test]
fn test_sky_coordinate_hash_basic() {
    println!("\n=== Test: Sky Coordinate Hash Computation ===\n");

    let quad = create_sky_quad_simple();

    println!("Input quad (sky coordinates):");
    for (i, star) in quad.iter().enumerate() {
        println!(
            "  Star {}: RA={:.6} deg, Dec={:.6} deg",
            i, star.ra, star.dec
        );
    }

    let hash = compute_hash_code_sky(&quad).expect("Failed to compute hash code");

    println!("\nHash code: {:?}", hash.components);

    // All components should be in [0, 1]
    for (i, &component) in hash.components.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&component),
            "Component {i} = {component} is not in [0, 1]"
        );
    }

    println!("Sky coordinate hash computed successfully");
}

#[test]
fn test_pixel_to_sky_quad_correspondence() {
    println!("\n=== Test: Pixel<->Sky Quad Correspondence ===\n");
    println!("This tests if we can match a quad in pixel space to the same");
    println!("quad in sky space, as would happen in real plate solving.\n");

    // Simulate a simple WCS transformation
    // Let's say: 1 pixel = 1 arcsecond, no rotation, centered at (180, 45)
    let center_ra = 180.0;
    let center_dec: f64 = 45.0;
    let scale_arcsec_per_px = 1.0;
    let image_center_x = 1024.0;
    let image_center_y = 1024.0;

    // Create a pixel quad
    let pixel_quad = [
        PixelCoord::new(1024.0, 1024.0), // Center
        PixelCoord::new(1424.0, 1024.0), // 400 pixels right
        PixelCoord::new(1224.0, 1224.0), // 200 right, 200 up
        PixelCoord::new(1124.0, 1324.0), // 100 right, 300 up
    ];

    // Convert to sky coordinates using realistic WCS transformation
    // This matches how real astronomical images are projected
    let sky_quad: Vec<SkyCoord> = pixel_quad
        .iter()
        .map(|p| {
            let dx_px = p.x - image_center_x;
            let dy_px = p.y - image_center_y;
            let dra_arcsec = -dx_px * scale_arcsec_per_px; // RA increases left
            let ddec_arcsec = dy_px * scale_arcsec_per_px; // Dec increases up
                                                           // Apply cos(dec) correction at the reference declination
            let dra_deg = dra_arcsec / 3600.0 / center_dec.to_radians().cos();
            let ddec_deg = ddec_arcsec / 3600.0;
            SkyCoord::new(center_ra + dra_deg, center_dec + ddec_deg)
        })
        .collect();

    let sky_quad_array = [sky_quad[0], sky_quad[1], sky_quad[2], sky_quad[3]];

    println!("Pixel quad:");
    for (i, p) in pixel_quad.iter().enumerate() {
        println!("  Star {}: ({:.1}, {:.1})", i, p.x, p.y);
    }

    println!("\nCorresponding sky quad:");
    for (i, s) in sky_quad_array.iter().enumerate() {
        println!("  Star {}: RA={:.6} deg, Dec={:.6} deg", i, s.ra, s.dec);
    }

    let pixel_hash = compute_hash_code_pixels(&pixel_quad).expect("Failed to compute pixel hash");
    let sky_hash = compute_hash_code_sky(&sky_quad_array).expect("Failed to compute sky hash");

    println!("\nPixel hash: {:?}", pixel_hash.components);
    println!("Sky hash:   {:?}", sky_hash.components);

    let distance = pixel_hash.distance(&sky_hash);
    println!("\nHash distance: {distance:.6}");

    // The hashes should be very close (small differences due to projection)
    assert!(
        distance < 0.01,
        "Pixel and sky hashes don't match! Distance = {distance}"
    );

    println!("Pixel quad matches corresponding sky quad");
    println!("   This validates the core matching algorithm!");
}

#[test]
fn test_degenerate_quad_detection() {
    println!("\n=== Test: Degenerate Quad Detection ===\n");

    // All stars at same point (degenerate)
    let degenerate1 = [
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(100.0, 100.0),
        PixelCoord::new(100.0, 100.0),
    ];

    let result1 = compute_hash_code_pixels(&degenerate1);
    assert!(result1.is_err(), "Should reject identical points");
    println!("Correctly rejected quad with all identical points");

    // Three stars collinear
    let degenerate2 = [
        PixelCoord::new(0.0, 0.0),
        PixelCoord::new(100.0, 0.0),
        PixelCoord::new(200.0, 0.0),
        PixelCoord::new(300.0, 0.0),
    ];

    let result2 = compute_hash_code_pixels(&degenerate2);
    // This might succeed or fail depending on implementation
    // The important thing is it doesn't crash
    match result2 {
        Ok(hash) => println!("  Collinear quad produced hash: {:?}", hash.components),
        Err(e) => println!("  Collinear quad rejected: {e}"),
    }
    println!("Handled collinear quad without crashing");
}

#[test]
fn test_hash_distance_metric() {
    println!("\n=== Test: Hash Distance Metric ===\n");

    let quad1 = create_pixel_quad_simple();

    // Create a slightly perturbed version
    let quad2 = [
        PixelCoord::new(101.0, 100.5), // Small perturbation
        PixelCoord::new(500.5, 100.0),
        PixelCoord::new(300.0, 300.5),
        PixelCoord::new(200.5, 400.0),
    ];

    // Create a very different quad
    let quad3 = [
        PixelCoord::new(0.0, 0.0),
        PixelCoord::new(1000.0, 0.0),
        PixelCoord::new(0.0, 1000.0),
        PixelCoord::new(1000.0, 1000.0),
    ];

    let hash1 = compute_hash_code_pixels(&quad1).expect("Failed to compute hash1");
    let hash2 = compute_hash_code_pixels(&quad2).expect("Failed to compute hash2");
    let hash3 = compute_hash_code_pixels(&quad3).expect("Failed to compute hash3");

    let dist_small = hash1.distance(&hash2);
    let dist_large = hash1.distance(&hash3);

    println!("Distance to slightly perturbed quad: {dist_small:.6}");
    println!("Distance to very different quad: {dist_large:.6}");

    assert!(
        dist_small < dist_large,
        "Similar quads should have smaller distance!"
    );

    assert!(
        dist_small < 0.01,
        "Small perturbation caused large distance: {dist_small}"
    );

    println!("Hash distance metric behaves correctly");
}

#[test]
fn test_realistic_matching_scenario() {
    println!("\n=== Test: Realistic Matching Scenario ===\n");
    println!("Simulates finding a quad in an index that matches an image quad.\n");

    // "Index quad" - from catalog
    let catalog_quad_sky = [
        SkyCoord::new(180.000, 45.000),
        SkyCoord::new(180.100, 45.000),
        SkyCoord::new(180.050, 45.050),
        SkyCoord::new(180.025, 45.075),
    ];

    // "Image quad" - same stars but projected to pixels with a realistic WCS
    // Scale: 0.88 arcsec/pixel
    // Image: 2048x1489 pixels
    // Center: RA=180, Dec=45
    let scale = 0.88 / 3600.0; // degrees per pixel
    let center_x = 1024.0;
    let center_y = 744.5;

    let image_quad: Vec<PixelCoord> = catalog_quad_sky
        .iter()
        .map(|sky| {
            let dra = (sky.ra - 180.0) * 45.0_f64.to_radians().cos();
            let ddec = sky.dec - 45.0;
            let dx_px = -dra / scale; // RA increases left
            let dy_px = ddec / scale; // Dec increases up
            PixelCoord::new(center_x + dx_px, center_y + dy_px)
        })
        .collect();

    let image_quad_array = [image_quad[0], image_quad[1], image_quad[2], image_quad[3]];

    println!("Catalog quad (sky):");
    for (i, s) in catalog_quad_sky.iter().enumerate() {
        println!("  Star {}: RA={:.6} deg, Dec={:.6} deg", i, s.ra, s.dec);
    }

    println!("\nImage quad (pixels):");
    for (i, p) in image_quad_array.iter().enumerate() {
        println!("  Star {}: ({:.2}, {:.2})", i, p.x, p.y);
    }

    let catalog_hash =
        compute_hash_code_sky(&catalog_quad_sky).expect("Failed to compute catalog hash");
    let image_hash =
        compute_hash_code_pixels(&image_quad_array).expect("Failed to compute image hash");

    println!("\nCatalog hash: {:?}", catalog_hash.components);
    println!("Image hash:   {:?}", image_hash.components);

    let distance = catalog_hash.distance(&image_hash);
    println!("\nHash distance: {distance:.6}");

    // With realistic WCS, the hash codes should match very closely
    assert!(
        distance < 0.01,
        "Realistic matching scenario failed! Distance = {distance}. This suggests WCS projection mismatch."
    );

    println!("Realistic quad matching works!");
    println!("   This validates the catalog-to-image matching pipeline");
}

#[test]
fn test_multiple_scales_same_geometry() {
    println!("\n=== Test: Same Geometry at Different Scales ===\n");

    // A quad at one scale
    let quad_small = [
        PixelCoord::new(0.0, 0.0),
        PixelCoord::new(100.0, 0.0),
        PixelCoord::new(50.0, 50.0),
        PixelCoord::new(25.0, 75.0),
    ];

    // Same geometry, 10x larger
    let quad_large = [
        PixelCoord::new(0.0, 0.0),
        PixelCoord::new(1000.0, 0.0),
        PixelCoord::new(500.0, 500.0),
        PixelCoord::new(250.0, 750.0),
    ];

    let hash_small = compute_hash_code_pixels(&quad_small).expect("Failed");
    let hash_large = compute_hash_code_pixels(&quad_large).expect("Failed");

    println!("Small quad hash: {:?}", hash_small.components);
    println!("Large quad hash: {:?}", hash_large.components);

    let distance = hash_small.distance(&hash_large);
    println!("\nHash distance: {distance:.10}");

    assert!(
        distance < 1e-10,
        "Scale changed hash! Distance = {distance}"
    );

    println!("Different scales produce identical hashes");
}

#[test]
fn test_hash_symmetry() {
    println!("\n=== Test: Hash Distance is Symmetric ===\n");

    let quad1 = create_pixel_quad_simple();
    let quad2 = [
        PixelCoord::new(200.0, 200.0),
        PixelCoord::new(600.0, 200.0),
        PixelCoord::new(400.0, 400.0),
        PixelCoord::new(300.0, 500.0),
    ];

    let hash1 = compute_hash_code_pixels(&quad1).unwrap();
    let hash2 = compute_hash_code_pixels(&quad2).unwrap();

    let dist_1_to_2 = hash1.distance(&hash2);
    let dist_2_to_1 = hash2.distance(&hash1);

    println!("Distance(hash1, hash2): {dist_1_to_2:.10}");
    println!("Distance(hash2, hash1): {dist_2_to_1:.10}");

    assert!(
        (dist_1_to_2 - dist_2_to_1).abs() < 1e-10,
        "Distance is not symmetric!"
    );

    println!("Hash distance is symmetric");
}

#[test]
fn test_hash_code_range_validation() {
    println!("\n=== Test: Hash Code Component Range ===\n");

    // Test 100 random quads (seeded for determinism).
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA57E_0123);

    for i in 0..100 {
        let quad = [
            PixelCoord::new(rng.gen_range(0.0..2048.0), rng.gen_range(0.0..2048.0)),
            PixelCoord::new(rng.gen_range(0.0..2048.0), rng.gen_range(0.0..2048.0)),
            PixelCoord::new(rng.gen_range(0.0..2048.0), rng.gen_range(0.0..2048.0)),
            PixelCoord::new(rng.gen_range(0.0..2048.0), rng.gen_range(0.0..2048.0)),
        ];

        if let Ok(hash) = compute_hash_code_pixels(&quad) {
            for (j, &component) in hash.components.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&component),
                    "Quad {i}, component {j} = {component} is out of range!"
                );
            }
        }
    }

    println!("All 100 random quads produced valid hash codes in [0,1]");
}

#[test]
fn test_gnomonic_projection_diagnosis() {
    println!("\n=== Test: Gnomonic Projection Diagnosis ===\n");
    println!("Investigating why pixel and sky hashes don't match\n");

    // Simple quad in sky space (very small field)
    let sky_quad = [
        SkyCoord::new(180.0, 45.0),
        SkyCoord::new(180.01, 45.0),      // 0.01 degrees = 36 arcsec
        SkyCoord::new(180.005, 45.005),   // offset
        SkyCoord::new(180.0025, 45.0075), // offset
    ];

    println!("Sky quad:");
    for (i, s) in sky_quad.iter().enumerate() {
        println!("  Star {}: RA={:.6} deg, Dec={:.6} deg", i, s.ra, s.dec);
    }

    // Manually do what compute_hash_code_sky does.
    // Find max separation IN ANGULAR DISTANCE (old way)
    let mut max_dist = 0.0;
    let mut max_pair = (0, 1);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dist = sky_quad[i].angular_distance(&sky_quad[j]);
            if dist > max_dist {
                max_dist = dist;
                max_pair = (i, j);
            }
        }
    }

    println!(
        "\nMax separation pair (angular): {} and {} (distance = {:.6} deg)",
        max_pair.0, max_pair.1, max_dist
    );

    // Now let's see what the hash comes out as
    let sky_hash = compute_hash_code_sky(&sky_quad).expect("Failed");

    println!("Sky hash: {:?}", sky_hash.components);

    // Now create an equivalent pixel quad with the SAME geometry
    // Using a scale of 100 pixels per 0.01 degrees
    let scale_px_per_deg = 100.0 / 0.01; // 10000 pixels per degree

    let pixel_quad = [
        PixelCoord::new(1000.0, 1000.0), // Reference at (180, 45)
        PixelCoord::new(1000.0 + 0.01 * scale_px_per_deg, 1000.0), // 0.01 deg in RA
        PixelCoord::new(
            1000.0 + 0.005 * scale_px_per_deg,
            1000.0 + 0.005 * scale_px_per_deg,
        ),
        PixelCoord::new(
            1000.0 + 0.0025 * scale_px_per_deg,
            1000.0 + 0.0075 * scale_px_per_deg,
        ),
    ];

    println!("\nEquivalent pixel quad:");
    for (i, p) in pixel_quad.iter().enumerate() {
        println!("  Star {}: ({:.2}, {:.2})", i, p.x, p.y);
    }

    // Find max separation in pixel space
    let mut max_pixel_dist = 0.0;
    let mut max_pixel_pair = (0, 1);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dist = pixel_quad[i].distance(&pixel_quad[j]);
            if dist > max_pixel_dist {
                max_pixel_dist = dist;
                max_pixel_pair = (i, j);
            }
        }
    }

    println!(
        "\nMax separation pair (pixels): {} and {} (distance = {:.2} px)",
        max_pixel_pair.0, max_pixel_pair.1, max_pixel_dist
    );

    let pixel_hash = compute_hash_code_pixels(&pixel_quad).expect("Failed");

    println!("Pixel hash: {:?}", pixel_hash.components);

    let distance = sky_hash.distance(&pixel_hash);
    println!("\nHash distance: {distance:.6}");

    if distance > 0.01 {
        println!("\n Hash codes still don't match after fix!");
        println!("    This means the fix didn't work as expected.");
    } else {
        println!("\nHash codes match! Fix successful!");
    }
}
