//! Blind-solve benchmark over a Tycho-2 index with no position hint, timing the
//! per-tile `.qidx` directory against the merged all-sky `.qidx` directory on the
//! *same* synthetic field. Ignored by default -- it needs the on-disk indexes
//! under `data/` and takes tens of seconds.
//!
//! Run: `cargo test --release -p platers-tests --test blind_bench -- --ignored --nocapture`

use std::path::Path;
use std::time::Instant;

use platers_core::index::{IndexSet, LoadedIndex};
use platers_core::{
    load_catalog_parquet, types::DetectedStar, DetectedField, PlateSolver, QueryConfig, ScaleRange,
    VerificationConfig,
};
use platers_tests::test_utils::{generate_test_case, validate_solution, TestCaseConfig};

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

/// Diagnostic: isolate whether the merged-index blind failure is candidate
/// flooding (exhausting the hypothesis budget before the true match is tested)
/// vs. a verification/locality problem. A HINTED solve at the true centre applies
/// the position pre-filter + local catalog bound; if it solves, the field IS in
/// the merged index and verifiable, so the blind failure is flooding.
#[test]
#[ignore = "needs on-disk hybrid index; diagnostic"]
fn diag_merged_hinted_vs_blind() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let merged_dir = root.join("data/index");
    if !catalog_path.exists() || !merged_dir.exists() {
        eprintln!("SKIP: merged index or catalog not present");
        return;
    }
    let catalog = load_catalog_parquet(&catalog_path).expect("load catalog");
    let scale = 2.0;
    let config = TestCaseConfig::new()
        .center(305.0, 40.0)
        .fov(2048.0 * scale / 60.0, 1489.0 * scale / 60.0)
        .image_size(2048, 1489)
        .rotation(33.0)
        .stars(300)
        .noise(0.3)
        .seed(7);
    let (stars, gt) = generate_test_case(&config, &catalog).expect("generate field");
    println!(
        "Field: {} stars @ {:.3}\"/px",
        stars.len(),
        config.pixel_scale_arcsec()
    );

    let base = || QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000,
        max_hypotheses: 10000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(config.pixel_scale_arcsec(), 0.1)),
        position_hint: None,
    };

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();

    // Drive solve_coarse DIRECTLY on the single matching merged file (the scout's
    // inner call), in-memory, no position hint -- isolating merged matching+verify
    // from the blind batch machinery so the info logs show num quads matched,
    // hypotheses tested, and best log-odds.
    let merged_file = std::fs::read_dir(&merged_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.to_string_lossy().contains("_n04."))
        .expect("n04 merged file");
    println!("Driving solve_coarse on {}", merged_file.display());
    let idx = LoadedIndex::open(&merged_file).expect("open merged");
    println!(
        "merged n04: {} stars, {} quads",
        idx.num_stars(),
        idx.num_quads()
    );
    let mut set = IndexSet::new();
    set.add(idx);
    let solver = PlateSolver::new(set, base());
    let t0 = Instant::now();
    match solver.solve_coarse(&DetectedField::new(
        stars,
        config.image_width,
        config.image_height,
    )) {
        Ok(res) => println!(
            "solve_coarse: solved in {:.1}s pos_err={:.3}\" log_odds={:.1}",
            t0.elapsed().as_secs_f64(),
            validate_solution(&res.wcs, &gt).position_error_arcsec,
            res.verification.log_odds
        ),
        Err(e) => println!(
            "solve_coarse: FAILED in {:.1}s: {e}",
            t0.elapsed().as_secs_f64()
        ),
    }
}

/// Benchmark the common case: a *tiled* index with a rough position + scale hint
/// (the normal "I roughly know where this points" plate-solve). Times the coarse
/// solve and the full solve (coarse + refinement) across sky positions, including
/// the lazy per-solve tile load.
#[test]
#[ignore = "needs on-disk hybrid index; benchmark"]
fn bench_tiled_hinted() {
    use platers_core::{query::PositionHint, CatalogIndex, SkyCoord};

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let tiled_dir = root.join("data/index");
    if !catalog_path.exists() || !tiled_dir.exists() {
        eprintln!("SKIP: tiled index or catalog not present");
        return;
    }
    let cat_index = CatalogIndex::new(load_catalog_parquet(&catalog_path).expect("load"));
    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    let field_radius = 0.5 * ((img_w * img_w + img_h * img_h) as f64).sqrt() * scale / 3600.0 + 0.2;

    let positions = [
        (45.0, -20.0),
        (120.0, 10.0),
        (200.0, 35.0),
        (260.0, -50.0),
        (300.0, 60.0),
        (15.0, 80.0),
        (180.0, 0.0),
        (330.0, -75.0),
    ];

    println!("\n-- tiled hinted solve (scale + 1 deg position hint) --");
    println!("  pos          | coarse | full(+refine) | pos_err");
    let (mut coarse_times, mut full_times) = (Vec::new(), Vec::new());
    for (i, &(ra, dec)) in positions.iter().enumerate() {
        let local = cat_index.stars_near(SkyCoord::new_normalized(ra, dec), field_radius);
        let config = TestCaseConfig::new()
            .center(ra, dec)
            .fov(img_w as f64 * scale / 60.0, img_h as f64 * scale / 60.0)
            .image_size(img_w, img_h)
            .rotation((ra + dec).rem_euclid(360.0))
            .stars(250)
            .mag_limit(13.5)
            .noise(0.3)
            .seed(i as u64 + 1);
        let Ok((stars, gt)) = generate_test_case(&config, &local) else {
            continue;
        };
        let qc = QueryConfig {
            max_stars_for_quads: 50,
            max_quads_to_try: 50000,
            max_hypotheses: 10000,
            hash_code_tolerance: 0.01,
            observation_epoch: None,
            scale_hint: Some(ScaleRange::from_nominal(scale, 0.1)),
            position_hint: Some(PositionHint {
                ra,
                dec,
                radius: 1.0,
            }),
        };
        let field = DetectedField::new(stars, img_w, img_h);
        let t0 = Instant::now();
        let coarse =
            PlateSolver::from_tile_directory(tiled_dir.clone(), qc.clone(), lenient_verification())
                .solve_coarse(&field);
        let coarse_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let full = PlateSolver::from_tile_directory(tiled_dir.clone(), qc, lenient_verification())
            .solve(&field);
        let full_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let err = full.as_ref().map_or(f64::NAN, |r| {
            validate_solution(&r.wcs, &gt).position_error_arcsec
        });
        println!("  ({ra:>5.0},{dec:>4.0}) | {coarse_ms:>4.0}ms | {full_ms:>8.0}ms | {err:>6.2}\"");
        if coarse.is_ok() {
            coarse_times.push(coarse_ms);
        }
        if full.is_ok() {
            full_times.push(full_ms);
        }
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    };
    println!(
        "\n  median coarse: {:.0}ms   median full(+refine): {:.0}ms   (n={})",
        med(coarse_times.clone()),
        med(full_times.clone()),
        full_times.len()
    );
}

/// Does blind solving work with **no scale hint**? Runs `solve_blind` on the merged
/// all-sky index both with and without a scale hint (and never a position hint) and
/// reports solved/time/accuracy/recovered-scale. Without a scale hint, every scale
/// tier is scanned and the scout uses global quad generation with a wide scale
/// acceptance window.
#[test]
#[ignore = "needs on-disk hybrid index; diagnostic"]
fn diag_blind_no_scale() {
    use platers_core::{query::PositionHint, CatalogIndex, ScaleRange, SkyCoord};

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let merged_dir = root.join("data/index");
    if !catalog_path.exists() || !merged_dir.exists() {
        eprintln!("SKIP: merged index or catalog not present");
        return;
    }
    let cat_index = CatalogIndex::new(load_catalog_parquet(&catalog_path).expect("load"));
    let scale = 2.0;
    let (img_w, img_h) = (2048_usize, 1489_usize);
    let field_radius = 0.5 * ((img_w * img_w + img_h * img_h) as f64).sqrt() * scale / 3600.0 + 0.2;

    let (ra, dec) = (200.0, 35.0);
    let local = cat_index.stars_near(SkyCoord::new_normalized(ra, dec), field_radius);
    let config = TestCaseConfig::new()
        .center(ra, dec)
        .fov(img_w as f64 * scale / 60.0, img_h as f64 * scale / 60.0)
        .image_size(img_w, img_h)
        .rotation(47.0)
        .stars(250)
        .mag_limit(13.5)
        .noise(0.3)
        .seed(3);
    let (stars, gt) = generate_test_case(&config, &local).expect("generate");
    println!(
        "Field at ({ra},{dec}), true scale {scale}\"/px, {} stars",
        stars.len()
    );
    let field = DetectedField::new(stars, img_w, img_h);

    // _ = scale hint variant; the position hint is always None for blind.
    for (label, scale_hint) in [
        (
            "with scale hint ",
            Some(ScaleRange::from_nominal(scale, 0.1)),
        ),
        ("NO scale hint   ", None),
    ] {
        let qc = QueryConfig {
            max_stars_for_quads: 50,
            max_quads_to_try: 50000,
            max_hypotheses: 10000,
            hash_code_tolerance: 0.01,
            observation_epoch: None,
            scale_hint,
            position_hint: None::<PositionHint>,
        };
        let solver =
            PlateSolver::from_tile_directory(merged_dir.clone(), qc, lenient_verification());
        let t0 = Instant::now();
        match solver.solve_blind(&field) {
            Ok(res) => {
                let v = validate_solution(&res.wcs, &gt);
                println!(
                    "  {label} SOLVED in {:5.2}s  pos_err={:.2}\"  recovered_scale={:.3}\"/px",
                    t0.elapsed().as_secs_f64(),
                    v.position_error_arcsec,
                    res.wcs.scale_arcsec_per_pixel()
                );
            }
            Err(e) => println!(
                "  {label} FAILED  in {:5.2}s: {e}",
                t0.elapsed().as_secs_f64()
            ),
        }
    }
}

#[test]
#[ignore = "needs on-disk Tycho-2 index; long-running benchmark"]
fn bench_blind_bin_vs_qidx() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let catalog_path = root.join("data/catalog.parquet");
    let qidx_dir = root.join("data/index");
    let merged_dir = root.join("data/index");
    if !catalog_path.exists() {
        eprintln!("SKIP: {} not present", catalog_path.display());
        return;
    }

    let catalog = load_catalog_parquet(&catalog_path).expect("load catalog");
    println!("Catalog: {} stars", catalog.len());

    // A ~2 arcsec/px, 2048x1489 field in dense Cygnus -- many detections -> many
    // image quads, which is the workload that stresses per-tile quad matching.
    let scale = 2.0;
    let config = TestCaseConfig::new()
        .center(305.0, 40.0)
        .fov(2048.0 * scale / 60.0, 1489.0 * scale / 60.0)
        .image_size(2048, 1489)
        .rotation(33.0)
        .stars(300)
        .noise(0.3)
        .seed(7);
    let (stars, gt) = generate_test_case(&config, &catalog).expect("generate field");
    println!(
        "Field: {} detected stars, pixel scale {:.3} arcsec/px",
        stars.len(),
        config.pixel_scale_arcsec()
    );

    let qc = || QueryConfig {
        max_stars_for_quads: 50,
        max_quads_to_try: 50000,
        max_hypotheses: 10000,
        hash_code_tolerance: 0.01,
        observation_epoch: None,
        scale_hint: Some(ScaleRange::from_nominal(config.pixel_scale_arcsec(), 0.1)),
        position_hint: None,
    };

    let field = DetectedField::new(stars.clone(), config.image_width, config.image_height);
    for (label, dir) in [
        (".qidx tiles", &qidx_dir),
        (".qidx merged/scale", &merged_dir),
    ] {
        if !dir.exists() {
            eprintln!("SKIP {label}: {} not present", dir.display());
            continue;
        }
        let solver = PlateSolver::from_tile_directory(dir.clone(), qc(), lenient_verification());
        let t0 = Instant::now();
        let result = solver.solve_blind(&field);
        let dt = t0.elapsed();
        match result {
            Ok(r) => {
                let v = validate_solution(&r.wcs, &gt);
                println!(
                    "{label:18} solved in {:6.1}s  pos_err={:.3}\"  scale_err={:.2}%",
                    dt.as_secs_f64(),
                    v.position_error_arcsec,
                    v.scale_error_percent
                );
            }
            Err(e) => println!("{label:18} FAILED in {:6.1}s: {e}", dt.as_secs_f64()),
        }
    }

    // Worst case: random detections that match nothing, so blind sweeps EVERY
    // matching tile (no early exit). This is where the per-tile load+match cost --
    // the thing `.qidx` changes -- actually shows. Same field count/scale.
    println!("\n-- full-sweep (random detections, matches nothing) --");
    let noise: Vec<DetectedStar> = {
        // Deterministic xorshift; no rand dep.
        let mut s = 0x00C0_FFEE_1234_5678_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1_u64 << 53) as f64
        };
        (0..stars.len())
            .map(|_| DetectedStar {
                x: next() * config.image_width as f64,
                y: next() * config.image_height as f64,
                flux: 1000.0 + next() * 1000.0,
            })
            .collect()
    };
    let noise_field = DetectedField::new(noise, config.image_width, config.image_height);
    for (label, dir) in [
        (".qidx tiles", &qidx_dir),
        (".qidx merged/scale", &merged_dir),
    ] {
        if !dir.exists() {
            continue;
        }
        let solver = PlateSolver::from_tile_directory(dir.clone(), qc(), lenient_verification());
        let t0 = Instant::now();
        let r = solver.solve_blind(&noise_field);
        println!(
            "{label:18} swept all tiles in {:6.1}s ({})",
            t0.elapsed().as_secs_f64(),
            if r.is_ok() {
                "spurious solve!"
            } else {
                "no match (expected)"
            }
        );
    }
}
