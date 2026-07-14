//! Solve a field from a JSON star list against an on-disk index.
//!
//! The solver ingests *pre-extracted* detections (0-based pixel coordinates,
//! arbitrary flux units) -- run your own source detection first:
//!
//! ```json
//! [{"x": 512.5, "y": 512.5, "flux": 1000.0}, ...]
//! ```
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p platers-core --example solve_from_json -- \
//!     stars.json data/index <width> <height> [scale_arcsec_per_pixel]
//! ```

use std::path::Path;

use platers_core::{
    types::{DetectedField, DetectedStar},
    Error, IndexSet, PlateSolver, PlatersResult, QueryConfig, ScaleRange,
};

fn main() -> PlatersResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        return Err(Error::ValueError(
            "usage: solve_from_json <stars.json> <index_dir> <width> <height> \
             [scale_arcsec_per_pixel]"
                .into(),
        ));
    }
    let stars_file = std::fs::File::open(&args[1])
        .map_err(|e| Error::IOError(format!("opening {}: {e}", args[1])))?;
    let stars: Vec<DetectedStar> = serde_json::from_reader(stars_file)
        .map_err(|e| Error::IOError(format!("parsing star-list JSON: {e}")))?;
    let width: usize = args[3]
        .parse()
        .map_err(|e| Error::ValueError(format!("parsing width: {e}")))?;
    let height: usize = args[4]
        .parse()
        .map_err(|e| Error::ValueError(format!("parsing height: {e}")))?;
    let scale: Option<f64> = args
        .get(5)
        .map(|s| {
            s.parse()
                .map_err(|e| Error::ValueError(format!("parsing scale: {e}")))
        })
        .transpose()?;

    let index = IndexSet::load_from_directory(Path::new(&args[2]))
        .map_err(|e| Error::IOError(format!("loading index from {}: {e}", args[2])))?;
    println!("{} stars, {} index tier(s)", stars.len(), index.len());

    // A scale hint restricts the search to the index tiers covering that pixel
    // scale (10-100x faster than a blind sweep).
    let config = QueryConfig {
        scale_hint: scale.map(|s| ScaleRange::from_nominal(s, 0.1)),
        ..QueryConfig::default()
    };

    let field = DetectedField::new(stars, width, height);
    let solution = PlateSolver::new(index, config).solve(&field)?;
    println!(
        "solved: center RA {:.6} deg, Dec {:.6} deg",
        solution.wcs.center.ra, solution.wcs.center.dec
    );
    println!(
        "  scale {:.4} arcsec/px, rotation {:.3} deg",
        solution.wcs.scale_arcsec_per_pixel(),
        solution.wcs.rotation_deg()
    );
    println!(
        "  {} matched stars, log-odds {:.1}, {} hypotheses tested",
        solution.verification.num_matches,
        solution.verification.log_odds,
        solution.num_hypotheses_tested
    );
    if let Some(refinement) = &solution.refinement {
        println!(
            "  refined: RMS residual {:.2} arcsec over {} stars",
            refinement.rms_residual_arcsec,
            refinement.matched_stars.len()
        );
    }
    Ok(())
}
