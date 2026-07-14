//! CLI tool for building a tiled `.qidx` plate-solving index from a star catalog.
//!
//! Writes a per-scale set of `HEALPix` tiles into the output directory (one
//! `.qidx` file per tile/scale). For fast blind solving, merge the tiles into
//! per-scale all-sky indices afterwards with `merge_scale_qidx`.

use clap::Parser;
use platers_build::{IndexConfig, TiledBuilder};
use platers_core::load_catalog_parquet;
use platers_core::{Error, PlatersResult};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "build_index")]
#[command(about = "Build a tiled .qidx plate-solving index from a star catalog")]
struct Cli {
    /// Input catalog file (Parquet format)
    #[arg(short, long)]
    catalog: PathBuf,

    /// Output directory for the tiled index
    #[arg(short, long)]
    output: PathBuf,

    /// Minimum quad diameter (arcminutes)
    #[arg(long, default_value = "2.0")]
    min_scale_arcmin: f64,

    /// Maximum quad diameter (degrees)
    #[arg(long, default_value = "1.0")]
    max_scale_deg: f64,

    /// Number of stars to select per `HEALPix` cell (uniformization cap). Ignored
    /// when --target-density is set.
    #[arg(long, default_value = "10")]
    stars_per_cell: usize,

    /// Target uniform stellar density for the index, in stars per square degree.
    /// When set, `stars_per_cell` is derived per scale tier from the tier's cell
    /// area so every scale carries this same sky density -- letting the catalog be
    /// far deeper than the index needs.
    #[arg(long)]
    target_density: Option<f64>,
}

fn main() -> PlatersResult<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    println!("Building tiled index from {}", cli.catalog.display());
    let start = Instant::now();
    let catalog = load_catalog_parquet(&cli.catalog)
        .map_err(|e| Error::IOError(format!("Failed to load catalog: {e}")))?;
    println!(
        "Loaded {} stars in {:.2}s",
        catalog.len(),
        start.elapsed().as_secs_f64()
    );

    let base_config = IndexConfig {
        stars_per_cell: cli.stars_per_cell,
        target_density_per_deg2: cli.target_density,
        ..IndexConfig::default()
    };
    if let Some(d) = cli.target_density {
        println!("Targeting uniform stellar density {d} stars/deg^2 (per-tier stars_per_cell)");
    }

    let builder = TiledBuilder::new(catalog, base_config);
    let start = Instant::now();
    let paths = builder.build_all_scales(&cli.output, cli.min_scale_arcmin, cli.max_scale_deg)?;

    println!(
        "Built {} tile files in {:.2}s -> {}",
        paths.len(),
        start.elapsed().as_secs_f64(),
        cli.output.display()
    );
    Ok(())
}
