//! CLI: build a platers star catalog (Parquet) from a real source catalog.
//!
//! Ingests a locally-downloaded, delimited export (Gaia DR3 CSV from the ESA
//! archive, or Tycho-2 TSV from `VizieR`), filters to a magnitude limit (and
//! optionally a sky region), and writes the `ra/dec/mag/id` Parquet that
//! `build_index` consumes.
//!
//! Examples:
//! ```text
//! # Gaia DR3, all-sky to G <= 12.5
//! build_catalog --source gaia -i gaia_source.csv --max-mag 12.5 -o catalog.parquet
//!
//! # Tycho-2 (VizieR TSV), a 5 deg cone to V <= 12
//! build_catalog --source tycho2 -i tyc2.tsv --max-mag 12.0 \
//!     --center-ra 180 --center-dec 45 --radius-deg 5 -o catalog.parquet
//! ```

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use platers_build::catalog_import::{import_catalog, CatalogSource, GaiaDr3, ImportFilter, Tycho2};
use platers_core::{save_catalog_parquet, types::SkyCoord};
use platers_core::{Error, PlatersResult};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Source {
    /// Gaia DR3 `gaia_source` export (CSV); magnitude = Gaia G.
    Gaia,
    /// Tycho-2 (`VizieR`, I/259) export (TSV); magnitude = Johnson V from VT/BT.
    Tycho2,
}

#[derive(Parser, Debug)]
#[command(name = "build_catalog")]
#[command(about = "Build a platers star catalog (Parquet) from a real source catalog")]
struct Cli {
    /// Source catalog format.
    #[arg(long, value_enum)]
    source: Source,

    /// Input file: a delimited export with a header row. Use `-` for stdin.
    #[arg(short, long)]
    input: PathBuf,

    /// Output Parquet path.
    #[arg(short, long)]
    output: PathBuf,

    /// Faintest magnitude to keep (in the source's band).
    #[arg(long, default_value = "12.5")]
    max_mag: f64,

    /// Brightest magnitude to keep (optional).
    #[arg(long)]
    min_mag: Option<f64>,

    /// Region cone center RA (deg) -- requires `--center-dec` and `--radius-deg`.
    #[arg(long, requires = "center_dec", requires = "radius_deg")]
    center_ra: Option<f64>,
    /// Region cone center Dec (deg).
    #[arg(long)]
    center_dec: Option<f64>,
    /// Region cone radius (deg).
    #[arg(long)]
    radius_deg: Option<f64>,

    /// Field delimiter override (default: per source -- `,` for Gaia, tab for Tycho-2).
    #[arg(long)]
    delimiter: Option<char>,
}

/// Guess the field delimiter from the first non-comment, non-empty line of a
/// file by counting candidates. Returns `None` for stdin (`-`) or on read error.
fn detect_delimiter(input: &std::path::Path) -> Option<u8> {
    use std::io::BufRead;
    if input.as_os_str() == "-" {
        return None;
    }
    let file = File::open(input).ok()?;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        return [b',', b'\t', b';', b'|']
            .into_iter()
            .max_by_key(|&d| line.bytes().filter(|&b| b == d).count())
            .filter(|&d| line.bytes().any(|b| b == d));
    }
    None
}

fn main() -> PlatersResult<()> {
    let cli = Cli::parse();

    let source: Box<dyn CatalogSource> = match cli.source {
        Source::Gaia => Box::new(GaiaDr3),
        Source::Tycho2 => Box::new(Tycho2),
    };

    // Delimiter: explicit override, else auto-detect from the file's header
    // (VizieR returns comma-CSV via TAP but tab-TSV via asu-tsv), else the
    // source's default.
    let delimiter = match cli.delimiter {
        Some(c) => u8::try_from(c).map_err(|_| {
            Error::ValueError("--delimiter must be a single-byte (ASCII) character".to_string())
        })?,
        None => detect_delimiter(&cli.input).unwrap_or_else(|| source.default_delimiter()),
    };

    let region = match (cli.center_ra, cli.center_dec, cli.radius_deg) {
        (Some(ra), Some(dec), Some(r)) => Some((SkyCoord::new_normalized(ra, dec), r)),
        _ => None,
    };
    let filter = ImportFilter {
        max_mag: Some(cli.max_mag),
        min_mag: cli.min_mag,
        region,
    };

    println!(
        "Source: {} (magnitude band: {})",
        source.name(),
        source.magnitude_band()
    );
    println!("Magnitude limit: <= {}", cli.max_mag);
    if let Some((c, r)) = filter.region {
        println!("Region: {:.3} deg of ({:.4}, {:.4})", r, c.ra, c.dec);
    }

    let reader: Box<dyn Read> = if cli.input.as_os_str() == "-" {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(BufReader::new(File::open(&cli.input).map_err(|e| {
            Error::IOError(format!("opening input {}: {e}", cli.input.display()))
        })?))
    };

    let (stars, stats) = import_catalog(reader, delimiter, source.as_ref(), &filter)?;

    println!(
        "Read {} records -> kept {} (filtered {}, unparsed {})",
        stats.read, stats.kept, stats.filtered, stats.unparsed
    );

    if stars.is_empty() {
        return Err(Error::InsufficientData(
            "no stars kept -- check the source format, delimiter, and magnitude limit".to_string(),
        ));
    }

    save_catalog_parquet(&cli.output, &stars)
        .map_err(|e| Error::IOError(format!("writing {}: {e}", cli.output.display())))?;
    println!("Wrote {} stars to {}", stars.len(), cli.output.display());
    Ok(())
}
