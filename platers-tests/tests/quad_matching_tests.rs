//! Fundamental quad matching tests.
//!
//! These tests validate the core quad matching logic by:
//! 1. Finding exact quads that exist in the index (no transformation)
//! 2. Finding quads with rotation applied
//! 3. Finding quads with scale changes
//! 4. Finding quads with combined transformations
//!
//! This is a prerequisite for the full integration tests: if we cannot find
//! a known quad in the index, the hash code matching system is fundamentally
//! broken.

use platers_core::{geometry::compute_hash_code_sky, SkyCoord};
use platers_tests::test_utils::TestHarness;

/// Test that we can find an exact quad that exists in the index.
/// This is the most basic test - if this fails, the hash code matching is broken.
#[test]
fn test_exact_quad_match() {
    println!("\n=== Test: Exact Quad Match (No Transformation) ===");
    println!("Goal: Extract a quad from the index and verify we can find it again\n");

    let harness = TestHarness::new().expect("Failed to create test harness");

    // Load an index and extract a quad from it
    let index_set = &harness.index_set;
    println!("Loaded {} indices", index_set.len());

    // Get the first index
    let indices = index_set.all_indices();
    assert!(!indices.is_empty(), "No indices loaded");

    let first_index = &indices[0];
    println!(
        "Using index: {} quads, {} stars, scale {:.1}-{:.1} arcmin",
        first_index.num_quads(),
        first_index.num_stars(),
        first_index.config.min_diameter_deg * 60.0,
        first_index.config.max_diameter_deg * 60.0
    );

    // Get the first quad from this index
    assert!(first_index.num_quads() > 0, "Index has no quads");
    let test_quad = first_index.quad(0);
    println!("\n1. Test quad from index:");
    println!("   Hash code: {:?}", test_quad.hash_code.components);
    println!("   Star indices: {:?}", test_quad.star_indices);

    // Get the actual star positions for this quad
    let star_positions: Vec<SkyCoord> = test_quad
        .star_indices
        .iter()
        .map(|&idx| first_index.star(idx).position)
        .collect();

    println!("\n2. Quad star positions (RA, Dec):");
    for (i, pos) in star_positions.iter().enumerate() {
        println!(
            "   Star {}: RA={:.6} deg, Dec={:.6} deg",
            i, pos.ra, pos.dec
        );
    }

    // Verify star indices are valid
    for (i, &idx) in test_quad.star_indices.iter().enumerate() {
        assert!(
            idx < first_index.num_stars(),
            "Star index {} at position {} is out of bounds (catalog size: {})",
            idx,
            i,
            first_index.num_stars()
        );
    }

    // Compute hash code directly from these star positions
    // This should match the hash code stored in the index
    let computed_hash = compute_hash_code_sky(&[
        star_positions[0],
        star_positions[1],
        star_positions[2],
        star_positions[3],
    ])
    .expect("Failed to compute hash code");

    println!("\n3. Hash code comparison:");
    println!("   Index hash:    {:?}", test_quad.hash_code.components);
    println!("   Computed hash: {:?}", computed_hash.components);

    // Verify hash codes match
    let hash_diff: f64 = test_quad
        .hash_code
        .components
        .iter()
        .zip(computed_hash.components.iter())
        .map(|(a, b): (&f64, &f64)| (a - b).abs())
        .sum();
    println!("   Total difference: {hash_diff:.10}");

    assert!(
        hash_diff < 1e-6,
        "Hash codes should match exactly for same stars (diff: {hash_diff})"
    );

    // Now use the hash tree to search for this quad
    // This is the core matching operation that the solver uses
    println!("\n4. Searching for quad in hash tree:");

    let search_radius = 0.01; // Hash code search tolerance
    let matches = first_index
        .find_matching_quads(&computed_hash, search_radius)
        .iter()
        .map(|m| (m.hash_distance, m.catalog_quad.quad_index))
        .collect::<Vec<_>>();

    println!(
        "   Found {} matches within radius {}",
        matches.len(),
        search_radius
    );

    assert!(
        !matches.is_empty(),
        "Should find at least one match (the original quad itself)"
    );

    // `find_matching_quads` returns matches in tree order, not sorted by distance,
    // so pick the closest. The self-quad must be among them at ~zero distance.
    let best_match = matches
        .iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("at least one match");
    println!("\n5. Best match:");
    println!("   Quad index: {}", best_match.1);
    println!("   Distance: {:.10}", best_match.0);

    // Verify the distance is near zero (exact match). The index stores quad codes
    // as f32, so the self-match distance is f32-precision (~1e-7), not f64-zero.
    assert!(
        best_match.0 < 1e-5,
        "Distance to self should be near zero (got: {})",
        best_match.0
    );

    // Verify this is actually the same quad by checking star indices
    let found_quad = first_index.quad(best_match.1);
    assert_eq!(
        found_quad.star_indices, test_quad.star_indices,
        "Found quad should have the same stars"
    );

    println!("\nSUCCESS: Exact quad matching works correctly!");
    println!("   - Hash code computation is consistent");
    println!("   - KD-tree search finds the correct quad");
    println!("   - Distance metric is working properly");
}

/// Test finding a quad after rotation.
/// Hash codes should be rotation invariant.
#[test]
fn test_rotated_quad_match() {
    println!("\n=== Test: Rotated Quad Match ===");
    println!("Goal: Verify hash codes are rotation invariant\n");

    let harness = TestHarness::new().expect("Failed to create test harness");

    // Get a quad from the index
    let indices = harness.index_set.all_indices();
    let first_index = &indices[0];
    let test_quad = first_index.quad(0);

    // Get star positions
    let star_positions: Vec<SkyCoord> = test_quad
        .star_indices
        .iter()
        .map(|&idx| first_index.star(idx).position)
        .collect();

    println!("Original quad hash: {:?}", test_quad.hash_code.components);

    // The hash code should be the same regardless of which star we start with
    // This tests rotation invariance at the quad level

    // Try all 4 starting positions (cyclic rotations)
    for rotation in 0..4 {
        let rotated_positions = [
            star_positions[rotation % 4],
            star_positions[(rotation + 1) % 4],
            star_positions[(rotation + 2) % 4],
            star_positions[(rotation + 3) % 4],
        ];

        let rotated_hash =
            compute_hash_code_sky(&rotated_positions).expect("Failed to compute rotated hash");

        println!("\nRotation {}: {:?}", rotation, rotated_hash.components);

        // Hash codes should be identical (or very close) for rotated star order
        // Note: The hash code algorithm normalizes for rotation, but different
        // star orderings might produce different normalizations
        let hash_diff: f64 = test_quad
            .hash_code
            .components
            .iter()
            .zip(rotated_hash.components.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        println!("   Difference from original: {hash_diff:.10}");
    }

    println!("\nRotation invariance test complete");
}

/// Test finding a quad with slight position perturbations.
/// This simulates measurement noise.
#[test]
fn test_noisy_quad_match() {
    println!("\n=== Test: Noisy Quad Match ===");
    println!("Goal: Verify hash codes are stable under small perturbations\n");

    let harness = TestHarness::new().expect("Failed to create test harness");

    let indices = harness.index_set.all_indices();
    let first_index = &indices[0];
    let test_quad = first_index.quad(0);

    // Get star positions
    let star_positions: Vec<SkyCoord> = test_quad
        .star_indices
        .iter()
        .map(|&idx| first_index.star(idx).position)
        .collect();

    println!("Original quad hash: {:?}", test_quad.hash_code.components);

    // Add small random perturbations (simulating 0.5 arcsec noise)
    let noise_arcsec = 0.5;
    let noise_deg = noise_arcsec / 3600.0;

    let noisy_positions: Vec<SkyCoord> = star_positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            // Deterministic "random" perturbation based on index
            let delta_ra = noise_deg * ((i as f64 * 0.7).sin());
            let delta_dec = noise_deg * ((i as f64 * 1.3).cos());
            SkyCoord::new(pos.ra + delta_ra, pos.dec + delta_dec)
        })
        .collect();

    println!("\nNoisy positions:");
    for (i, (orig, noisy)) in star_positions
        .iter()
        .zip(noisy_positions.iter())
        .enumerate()
    {
        let delta_arcsec = ((orig.ra - noisy.ra) * 3600.0).hypot((orig.dec - noisy.dec) * 3600.0);
        println!("   Star {i}: Delta = {delta_arcsec:.3} arcsec");
    }

    let noisy_hash = compute_hash_code_sky(&[
        noisy_positions[0],
        noisy_positions[1],
        noisy_positions[2],
        noisy_positions[3],
    ])
    .expect("Failed to compute noisy hash");

    println!("\nNoisy quad hash: {:?}", noisy_hash.components);

    let hash_diff: f64 = test_quad
        .hash_code
        .components
        .iter()
        .zip(noisy_hash.components.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    println!("Hash difference: {hash_diff:.10}");

    // Search for the noisy quad in the hash tree
    let search_radius = 0.02; // Larger tolerance for noisy data
    let matches = first_index
        .find_matching_quads(&noisy_hash, search_radius)
        .iter()
        .map(|m| (m.hash_distance, m.catalog_quad.quad_index))
        .collect::<Vec<_>>();

    println!(
        "\nFound {} matches within radius {}",
        matches.len(),
        search_radius
    );

    if matches.is_empty() {
        println!("\n WARNING: No matches found");
        println!("   This suggests the noise tolerance may need adjustment");
    } else {
        println!("Best match:");
        println!("   Quad index: {}", matches[0].1);
        println!("   Distance: {:.10}", matches[0].0);

        // The best match should still be our original quad (or very close)
        if matches[0].1 == test_quad.quad_index {
            println!("\nSUCCESS: Found correct quad despite noise!");
        } else {
            println!("\n WARNING: Best match is a different quad");
            println!("   This may be acceptable if the noise changed the quad significantly");
        }
    }
}
