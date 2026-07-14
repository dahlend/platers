//! Platers CLI - Command-line interface for astronomical plate solving
//!
//! This tool solves astrometry (finds WCS solutions) from lists of detected stars.
//! It does NOT perform star detection - use tools like `SExtractor`, photutils, or
//! custom detection pipelines to generate star lists first.

use clap::{Args, Parser, Subcommand};
use platers_core::{
    load_catalog_parquet,
    types::{DetectedField, DetectedStar},
    CatalogIndex, Error, IndexSet, PlateSolver, PlatersResult, PositionHint, QueryConfig,
    RefinementConfig, ScaleRange,
};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "platers")]
#[command(about = "Astronomical plate solving from detected star lists", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve astrometry from a star list
    Solve(SolveArgs),

    /// Show information about index files
    Info {
        /// Index directory
        #[arg(short = 'I', long)]
        index_dir: PathBuf,
    },
}

/// Arguments for `platers solve`.
#[derive(Args)]
struct SolveArgs {
    /// Input file with detected stars (JSON format)
    #[arg(short, long)]
    input: PathBuf,

    /// Index directory containing plate solving indices
    #[arg(short = 'I', long)]
    index_dir: PathBuf,

    /// Image width in pixels
    #[arg(short = 'W', long)]
    width: usize,

    /// Image height in pixels
    #[arg(short = 'H', long)]
    height: usize,

    /// Pixel scale in arcsec/pixel (if known)
    #[arg(short = 's', long)]
    scale: Option<f64>,

    /// Pixel scale uncertainty (default: 0.1 = +/-10%)
    #[arg(long, default_value = "0.1")]
    scale_uncertainty: f64,

    /// RA hint in degrees (if known)
    #[arg(long, allow_hyphen_values = true)]
    ra: Option<f64>,

    /// Dec hint in degrees (if known; negative values allowed)
    #[arg(long, allow_hyphen_values = true)]
    dec: Option<f64>,

    /// Position search radius in degrees (default: 5.0, matching the server)
    #[arg(long, default_value = "5.0")]
    radius: f64,

    /// Ceiling on image quads to generate (the actual budget scales with star
    /// count up to this cap; default: 300000)
    #[arg(long, default_value = "300000")]
    max_quads: usize,

    /// Max WCS hypotheses to verify (default: 200000, matching the server)
    #[arg(long, default_value = "200000")]
    max_hypotheses: usize,

    /// Dense star catalog (parquet) to refine against instead of the index's
    /// own embedded stars -- more stars around the solved center give a
    /// better-conditioned refinement / SIP fit, and it is where proper
    /// motions live (`--epoch` needs it).
    #[arg(long)]
    catalog: Option<PathBuf>,

    /// Observation epoch as a decimal Julian year (e.g. 2021.4). Catalog
    /// stars with known proper motion are propagated from the catalog epoch
    /// (2016.0, Gaia DR3) to this date before refinement. Off by default:
    /// catalog positions are used as stored.
    #[arg(long)]
    epoch: Option<f64>,

    /// Fit SIP distortion polynomials of this order (2-5) during refinement.
    /// Off by default: the solution is a pure linear TAN WCS. Needs a
    /// well-populated field (at least ~25 matched stars) or the fit falls back
    /// to linear.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=5))]
    sip_order: Option<u32>,

    /// Output file for WCS solution (JSON format)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> PlatersResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Solve(args) => solve_command(&args),
        Commands::Info { index_dir } => info_command(&index_dir),
    }
}

fn solve_command(args: &SolveArgs) -> PlatersResult<()> {
    // Initialize logging if verbose
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    // Read detected stars from input file
    println!("Reading stars from {}...", args.input.display());
    let stars = read_star_list(&args.input)?;
    println!("  Loaded {} stars", stars.len());

    // Load indices
    println!("Loading indices from {}...", args.index_dir.display());
    let index_set = IndexSet::load_from_directory(&args.index_dir)
        .map_err(|e| Error::IOError(format!("Failed to load indices: {e}")))?;
    println!("  Loaded {} indices", index_set.len());

    // Build query configuration
    let mut query_config = QueryConfig {
        max_quads_to_try: args.max_quads,
        max_hypotheses: args.max_hypotheses,
        ..QueryConfig::default()
    };

    if let Some(pixel_scale) = args.scale {
        println!(
            "Using scale hint: {:.2} arcsec/pixel (+/-{:.0}%)",
            pixel_scale,
            args.scale_uncertainty * 100.0
        );
        query_config.scale_hint = Some(ScaleRange::from_nominal(
            pixel_scale,
            args.scale_uncertainty,
        ));
    }

    if let (Some(ra), Some(dec)) = (args.ra, args.dec) {
        println!(
            "Using position hint: RA={ra:.2} deg, Dec={dec:.2} deg, radius={:.2} deg",
            args.radius
        );
        query_config.position_hint = Some(PositionHint::new(ra, dec, args.radius));
    }

    if let Some(epoch) = args.epoch {
        println!("Using observation epoch: {epoch:.2} (proper-motion propagation)");
        query_config.observation_epoch = Some(epoch);
    }

    // Create solver
    let solver = PlateSolver::new(index_set, query_config);
    let field = DetectedField::new(stars, args.width, args.height);

    // Solve. `--sip-order` opts into a SIP distortion fit on the final refined
    // WCS; the default refinement stays linear. `--catalog` refines against a
    // dense external catalog (the server's configuration) instead of the
    // index's embedded stars.
    let refinement_config = args.sip_order.map(|order| RefinementConfig {
        sip_order: Some(order),
        ..RefinementConfig::default()
    });
    let catalog = match &args.catalog {
        Some(path) => {
            println!("Loading catalog from {}...", path.display());
            let stars = load_catalog_parquet(path)
                .map_err(|e| Error::IOError(format!("Failed to load catalog: {e}")))?;
            println!("  Loaded {} catalog stars", stars.len());
            Some(CatalogIndex::new(stars))
        }
        None => None,
    };
    println!("\nSolving...");
    let start = std::time::Instant::now();

    match solver.solve_with_refinement_against(&field, refinement_config, catalog.as_ref()) {
        Ok(solution) => {
            let elapsed = start.elapsed();

            println!("\nSOLVED in {:.2}s", elapsed.as_secs_f64());
            println!("\nWCS Solution:");
            println!(
                "  Center: RA={:.6} deg, Dec={:.6} deg",
                solution.wcs.center.ra, solution.wcs.center.dec
            );
            println!(
                "  Pixel scale: {:.4} arcsec/pixel",
                solution.wcs.scale_arcsec_per_pixel()
            );
            println!("  Rotation: {:.3} deg", solution.wcs.rotation_deg());
            match (&solution.wcs.sip, args.sip_order) {
                (Some(sip), _) => println!("  SIP distortion: order {} fitted", sip.a.order),
                (None, Some(_)) => {
                    println!("  SIP distortion: requested but not fitted (too few matched stars)");
                }
                (None, None) => {}
            }
            println!("\nVerification:");
            println!("  Matched stars: {}", solution.verification.num_matches);
            println!("  Log-odds: {:.1}", solution.verification.log_odds);

            // Compute RMS from match distances
            if !solution.verification.match_distances.is_empty() {
                let sum_sq: f64 = solution
                    .verification
                    .match_distances
                    .iter()
                    .map(|d| d * d)
                    .sum();
                let rms = (sum_sq / solution.verification.match_distances.len() as f64).sqrt();
                println!("  RMS residual: {rms:.2} arcsec");
            }

            println!("\nPerformance:");
            println!("  Hypotheses tested: {}", solution.num_hypotheses_tested);
            println!("  Solve time: {:.2}s", elapsed.as_secs_f64());

            // Write output if requested
            if let Some(output_path) = &args.output {
                write_solution(&solution, output_path)?;
                println!("\nSolution written to {}", output_path.display());
            }

            Ok(())
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("\nFAILED to solve in {:.2}s", elapsed.as_secs_f64());
            println!("Error: {e}");
            Err(e)
        }
    }
}

fn info_command(index_dir: &Path) -> PlatersResult<()> {
    println!("Loading indices from {}...\n", index_dir.display());

    let index_set = IndexSet::load_from_directory(index_dir)
        .map_err(|e| Error::IOError(format!("Failed to load indices: {e}")))?;

    println!("Index Set Information");
    println!("=====================");
    println!("Total indices: {}", index_set.len());
    println!("\nIndex Details:");

    for (i, index) in index_set.all_indices().iter().enumerate() {
        println!("\nIndex {i}:");
        println!(
            "  File: {}",
            index
                .path
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
        );
        println!(
            "  Pixel scale: {:.2} - {:.2} arcsec/pixel",
            index.scale_range.0, index.scale_range.1
        );
        println!(
            "  Quad diameter: {:.2} - {:.2} arcmin",
            index.diameter_range.0 * 60.0,
            index.diameter_range.1 * 60.0
        );
        println!("  Number of quads: {}", index.num_quads());
        println!("  Number of stars: {}", index.num_stars());
    }

    Ok(())
}

/// Read star list from JSON file
fn read_star_list(path: &Path) -> PlatersResult<Vec<DetectedStar>> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::IOError(format!("Failed to open {}: {e}", path.display())))?;

    let stars: Vec<DetectedStar> = serde_json::from_reader(file).map_err(|e| {
        Error::IOError(format!("Failed to parse JSON from {}: {e}", path.display()))
    })?;

    Ok(stars)
}

/// Write solution to JSON file
fn write_solution(solution: &platers_core::SolveResult, path: &Path) -> PlatersResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| Error::IOError(format!("Failed to create {}: {e}", path.display())))?;

    serde_json::to_writer_pretty(file, solution)
        .map_err(|e| Error::IOError(format!("Failed to write JSON to {}: {e}", path.display())))?;

    Ok(())
}
