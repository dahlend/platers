//! Tiled, lazy-loading index: build a tiled index from the fixture catalog,
//! confirm the cone prunes which tiles load, and solve via lazy loading.

use platers_build::{merge_scale_indices, IndexConfig, TiledBuilder};
use platers_core::{
    index::IndexSet, query::PositionHint, types::SkyCoord, DetectedField, PlateSolver, QueryConfig,
    ScaleRange, VerificationConfig,
};
use platers_tests::test_utils::{
    generate_test_case, validate_solution, TestCaseConfig, TestHarness,
};

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

#[test]
fn tiled_build_lazy_load_and_solve() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    // Build a tiled index from the catalog into a unique temp dir.
    let dir = std::env::temp_dir().join(format!("platers_tiled_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let builder = TiledBuilder::new(harness.catalog().to_vec(), IndexConfig::default());
    let paths = builder
        .build_all_scales(&dir, 3.0, 0.4)
        .expect("tiled build failed");
    assert!(!paths.is_empty(), "tiled build should emit files");

    let scale = ScaleRange::from_nominal(1.0, 0.1);
    let scale_range = Some((scale.min_arcsec_per_pixel, scale.max_arcsec_per_pixel));

    // Pruning: a cone far from the (180,45) catalog selects no tiles to load;
    // a cone on the field selects some.
    let far = IndexSet::load_tiles_for(
        &dir,
        2048,
        scale_range,
        Some((SkyCoord::new(10.0, -30.0), 1.0)),
    )
    .expect("load_tiles_for");
    assert_eq!(far.len(), 0, "a far cone should load zero tiles");
    let near = IndexSet::load_tiles_for(
        &dir,
        2048,
        scale_range,
        Some((SkyCoord::new(180.0, 45.0), 1.5)),
    )
    .expect("load_tiles_for");
    assert!(
        !near.is_empty(),
        "a cone on the field should load some tiles"
    );

    // End-to-end: a field at the catalog center solves via lazy tile loading.
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(2048.0 / 60.0, 1489.0 / 60.0)
        .image_size(2048, 1489)
        .rotation(20.0)
        .stars(60)
        .noise(0.3)
        .seed(4);
    let (stars, gt) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");
    let qc = QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000,
        max_hypotheses: 10000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(config.pixel_scale_arcsec(), 0.1)),
        position_hint: Some(PositionHint {
            ra: 180.0,
            dec: 45.0,
            radius: 1.0,
        }),
    };
    let result = PlateSolver::from_tile_directory(dir.clone(), qc, lenient_verification())
        .solve(&DetectedField::new(
            stars,
            config.image_width,
            config.image_height,
        ))
        .expect("lazy tiled solve failed");
    let v = validate_solution(&result.wcs, &gt);
    assert!(
        v.position_error_arcsec < 3.0,
        "tiled lazy solve inaccurate: {:.3}\"",
        v.position_error_arcsec
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Merge the per-tile index into one all-sky `.qidx` per scale and blind-solve
/// (no position hint) against it. Exercises the fast-blind path end to end: the
/// `allsky_` files are picked by scale band, always pass the (absent) cone, the
/// merged tree localizes the field, and the final hinted refine loads the all-sky
/// index correctly.
#[test]
fn merged_allsky_blind_solve() {
    let harness = TestHarness::new().expect("Failed to create test harness");

    let tiles = std::env::temp_dir().join(format!("platers_blind_tiles_{}", std::process::id()));
    let merged = std::env::temp_dir().join(format!("platers_blind_merged_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tiles);
    let _ = std::fs::remove_dir_all(&merged);

    let builder = TiledBuilder::new(harness.catalog().to_vec(), IndexConfig::default());
    let _ = builder
        .build_all_scales(&tiles, 3.0, 0.4)
        .expect("tiled build failed");
    let allsky = merge_scale_indices(&tiles, &merged).expect("merge failed");
    assert!(!allsky.is_empty(), "merge should emit all-sky files");
    assert!(
        allsky.iter().all(|p| p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("allsky_"))),
        "merged files should use the allsky_ prefix"
    );

    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .fov(2048.0 / 60.0, 1489.0 / 60.0)
        .image_size(2048, 1489)
        .rotation(20.0)
        .stars(60)
        .noise(0.3)
        .seed(4);
    let (stars, gt) =
        generate_test_case(&config, harness.catalog()).expect("Failed to generate test case");

    // No position hint -- this is the blind path.
    let qc = QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000,
        max_hypotheses: 10000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(config.pixel_scale_arcsec(), 0.1)),
        position_hint: None,
    };
    let result = PlateSolver::from_tile_directory(merged.clone(), qc, lenient_verification())
        .solve_blind(&DetectedField::new(
            stars,
            config.image_width,
            config.image_height,
        ))
        .expect("merged all-sky blind solve failed");
    let v = validate_solution(&result.wcs, &gt);
    assert!(
        v.position_error_arcsec < 3.0,
        "merged blind solve inaccurate: {:.3}\"",
        v.position_error_arcsec
    );

    let _ = std::fs::remove_dir_all(&tiles);
    let _ = std::fs::remove_dir_all(&merged);
}
