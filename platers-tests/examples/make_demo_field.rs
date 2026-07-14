//! Generate a synthetic detected-star field over the committed fixture catalog,
//! for the readme quickstart -- a working solve with no multi-GB dataset.
//!
//! Writes a JSON star list (default `demo_stars.json`) whose true pose is known,
//! then prints the `platers-cli` command that solves it and the ground truth to
//! compare against.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p platers-tests --example make_demo_field -- [out.json]
//! ```

use platers_core::{load_catalog_parquet, Error, PlatersResult};
use platers_tests::test_utils::{generate_test_case, TestCaseConfig};

fn main() -> PlatersResult<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "demo_stars.json".to_string());

    let catalog_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/fixture_catalog.parquet"
    );
    let catalog = load_catalog_parquet(catalog_path)?;

    // The fixture catalog is regional, centered at (180, 45); the config
    // defaults give a 30' x 20' field on a 2048 x 1489 detector with a little
    // per-star position noise, like a real extraction.
    let config = TestCaseConfig::new()
        .center(180.0, 45.0)
        .stars(60)
        .noise(0.3);
    let (stars, truth) = generate_test_case(&config, &catalog)?;

    let file =
        std::fs::File::create(&out).map_err(|e| Error::IOError(format!("creating {out}: {e}")))?;
    serde_json::to_writer_pretty(file, &stars)
        .map_err(|e| Error::IOError(format!("writing {out}: {e}")))?;

    println!("\nWrote {} stars to {out}", stars.len());
    println!(
        "Ground truth: center RA {:.6} deg, Dec {:.6} deg, scale {:.4} arcsec/px, rotation {:.1} deg",
        truth.wcs.center.ra,
        truth.wcs.center.dec,
        truth.wcs.scale_arcsec_per_pixel(),
        truth.wcs.rotation_deg()
    );
    println!("\nSolve it with:\n");
    println!(
        "  cargo run --release -p platers-cli -- solve \\\n      --input {out} --index-dir demo_index \\\n      --width {} --height {} --scale {:.2}",
        config.image_width,
        config.image_height,
        truth.wcs.scale_arcsec_per_pixel()
    );
    Ok(())
}
