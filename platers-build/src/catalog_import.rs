//! Ingest real star catalogs into the platers `Star` format.
//!
//! Real plate solving needs an index built from a real catalog, not synthetic
//! stars. This module turns a locally-downloaded, delimited catalog export
//! (Gaia DR3 from the ESA archive, Tycho-2 from `VizieR`, ...) into a `Vec<Star>`
//! that [`platers_core::save_catalog_parquet`] can write and `build_index` can
//! consume.
//!
//! ## Why delimited-with-header rather than fixed-width
//!
//! Both supported sources are readily exported as CSV/TSV with a **named header
//! row** (Gaia archive emits clean CSV; `VizieR` can emit either). Parsing by
//! column *name* is robust -- it survives column reordering and extra columns --
//! whereas hard-coded byte offsets silently produce garbage if the upstream
//! format shifts. Comment lines (`#...`) and unparseable rows (e.g. `VizieR`'s units
//! / dashed-separator lines) are skipped and counted, never fatal.
//!
//! ## Pluggability
//!
//! A [`CatalogSource`] knows how to map one header row to a `Star` (including
//! the source's magnitude convention). Add a source by implementing the trait;
//! the read/filter/collect pipeline in [`import_catalog`] is shared.

use std::collections::HashMap;
use std::io::Read;

use platers_core::types::{SkyCoord, Star};
use platers_core::{Error, PlatersResult};

/// One parsed record, addressable by the file's header column names.
#[derive(Debug)]
pub struct Row<'a> {
    headers: &'a HashMap<String, usize>,
    record: &'a csv::StringRecord,
}

impl Row<'_> {
    /// The trimmed, non-empty value of `column`, if present.
    #[must_use]
    pub fn get(&self, column: &str) -> Option<&str> {
        let idx = *self.headers.get(column)?;
        let v = self.record.get(idx)?.trim();
        (!v.is_empty()).then_some(v)
    }

    /// First present value among `columns` (for sources whose exports vary the
    /// column name, e.g. `ra` vs `RA_ICRS`).
    #[must_use]
    pub fn get_any(&self, columns: &[&str]) -> Option<&str> {
        columns.iter().find_map(|c| self.get(c))
    }

    /// Parse the first present column among `columns` as a finite `f64`.
    #[must_use]
    pub fn f64_any(&self, columns: &[&str]) -> Option<f64> {
        self.get_any(columns)
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite())
    }
}

/// A real catalog format: maps a delimited file's columns to platers `Star`s.
pub trait CatalogSource {
    /// Stable identifier (used in messages).
    fn name(&self) -> &'static str;

    /// Default field delimiter for this source's usual export.
    fn default_delimiter(&self) -> u8;

    /// Human-readable magnitude band the `--max-mag` limit applies to.
    fn magnitude_band(&self) -> &'static str;

    /// Build a `Star` from one row, or `None` to skip it (missing/invalid
    /// fields, or a non-data line like a units/separator row).
    fn parse_row(&self, row: &Row<'_>) -> Option<Star>;
}

/// Construct a `Star`, normalizing the position so out-of-range or
/// seam-straddling RA/Dec never panics (real exports can carry RA = 360.0).
fn make_star(ra: f64, dec: f64, magnitude: f64, id: Option<u64>) -> Option<Star> {
    if !ra.is_finite() || !dec.is_finite() || !magnitude.is_finite() {
        return None;
    }
    Some(Star {
        position: SkyCoord::new_normalized(ra, dec),
        magnitude,
        id,
        proper_motion: None,
    })
}

/// Gaia DR3 (`gaia_source`), as exported from the ESA Gaia archive (CSV).
///
/// Magnitude is Gaia **G** (`phot_g_mean_mag`). G ~ Johnson V to within a few
/// tenths for most stars; the `--max-mag` limit is applied in G.
#[derive(Debug, Default, Clone, Copy)]
pub struct GaiaDr3;

impl CatalogSource for GaiaDr3 {
    fn name(&self) -> &'static str {
        "gaia-dr3"
    }
    fn default_delimiter(&self) -> u8 {
        b','
    }
    fn magnitude_band(&self) -> &'static str {
        "Gaia G"
    }
    fn parse_row(&self, row: &Row<'_>) -> Option<Star> {
        let ra = row.f64_any(&["ra", "RA_ICRS", "ra_deg"])?;
        let dec = row.f64_any(&["dec", "DE_ICRS", "dec_deg"])?;
        let mag = row.f64_any(&["phot_g_mean_mag", "Gmag"])?;
        let id = row
            .get_any(&["source_id", "SOURCE_ID", "Source"])
            .and_then(|s| s.parse::<u64>().ok());
        make_star(ra, dec, mag, id)
    }
}

/// Tycho-2, as exported from `VizieR` (catalog I/259).
///
/// Magnitude is **Johnson V** derived from the Tycho photometry,
/// `V = VT - 0.090*(BT - VT)` (the CDS-documented transform); when BT is absent
/// VT is used directly. Position is the mean ICRS position (`RAmdeg`/`DEmdeg`),
/// falling back to the observed position when the mean is not given.
#[derive(Debug, Default, Clone, Copy)]
pub struct Tycho2;

impl CatalogSource for Tycho2 {
    fn name(&self) -> &'static str {
        "tycho2"
    }
    fn default_delimiter(&self) -> u8 {
        b'\t'
    }
    fn magnitude_band(&self) -> &'static str {
        "Johnson V (from Tycho VT/BT)"
    }
    fn parse_row(&self, row: &Row<'_>) -> Option<Star> {
        let ra = row.f64_any(&["RAmdeg", "RA_ICRS", "RAdeg", "ra"])?;
        let dec = row.f64_any(&["DEmdeg", "DE_ICRS", "DEdeg", "dec"])?;
        let vt = row.f64_any(&["VTmag", "VT"]);
        let bt = row.f64_any(&["BTmag", "BT"]);
        let v = johnson_v(bt, vt)?;
        // TYC identifier "TYC1-TYC2-TYC3" -> a stable packed u64, if present.
        let id = row
            .get_any(&["TYC", "Tycho", "TYC1"])
            .and_then(|s| tyc_to_id(s, row));
        make_star(ra, dec, v, id)
    }
}

/// Johnson V from Tycho VT/BT. Requires VT; BT optional (improves the estimate).
fn johnson_v(bt: Option<f64>, vt: Option<f64>) -> Option<f64> {
    let vt = vt?;
    Some(match bt {
        Some(bt) => vt - 0.090 * (bt - vt),
        None => vt,
    })
}

/// Pack a Tycho identifier into a u64 (`TYC1*1e6 + TYC2*10 + TYC3`). Accepts a
/// combined field -- either dash-separated ("1-8-1", `VizieR`) or space-separated
/// ("0001 00008 1", the raw CDS `tyc2.dat`) -- or separate `TYC1`/`TYC2`/`TYC3`
/// columns.
fn tyc_to_id(first: &str, row: &Row<'_>) -> Option<u64> {
    let parts: Vec<u64> = if first.contains(['-', ' ']) {
        first
            .split(['-', ' '])
            .filter(|s| !s.is_empty())
            .filter_map(|p| p.parse().ok())
            .collect()
    } else {
        ["TYC1", "TYC2", "TYC3"]
            .iter()
            .filter_map(|c| row.get(c).and_then(|s| s.parse().ok()))
            .collect()
    };
    match parts.as_slice() {
        [t1, t2, t3] => Some(t1 * 1_000_000 + t2 * 10 + t3),
        _ => None,
    }
}

/// Magnitude / region cut applied during import.
#[derive(Debug, Clone, Default)]
pub struct ImportFilter {
    /// Keep stars brighter-or-equal to this (i.e. magnitude <= `max_mag`).
    pub max_mag: Option<f64>,
    /// Keep stars fainter-or-equal to this (magnitude >= `min_mag`).
    pub min_mag: Option<f64>,
    /// Optional cone: `(center, radius_degrees)`.
    pub region: Option<(SkyCoord, f64)>,
}

impl ImportFilter {
    fn keep(&self, star: &Star) -> bool {
        if self.max_mag.is_some_and(|m| star.magnitude > m) {
            return false;
        }
        if self.min_mag.is_some_and(|m| star.magnitude < m) {
            return false;
        }
        if let Some((center, radius)) = self.region {
            if star.position.angular_distance(&center) > radius {
                return false;
            }
        }
        true
    }
}

/// Counts from an import run.
#[derive(Debug, Default, Clone, Copy)]
pub struct ImportStats {
    /// Data records read (excludes comment lines).
    pub read: usize,
    /// Records that did not parse into a star (missing fields, units/separator
    /// lines, bad numbers).
    pub unparsed: usize,
    /// Parsed stars dropped by the magnitude/region filter.
    pub filtered: usize,
    /// Stars kept.
    pub kept: usize,
}

/// Read a delimited catalog export and return the kept stars plus counts.
///
/// `delimiter` is the field separator (comma for Gaia CSV, tab for `VizieR` TSV).
/// Lines beginning with `#` are treated as comments and skipped.
///
/// # Errors
/// Returns an error if the header row cannot be read or a record is malformed at
/// the CSV layer (not for individual unparseable *values* -- those are counted in
/// [`ImportStats::unparsed`]).
pub fn import_catalog<R: Read>(
    reader: R,
    delimiter: u8,
    source: &dyn CatalogSource,
    filter: &ImportFilter,
) -> PlatersResult<(Vec<Star>, ImportStats)> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .comment(Some(b'#'))
        .flexible(true)
        .from_reader(reader);

    let headers: HashMap<String, usize> = rdr
        .headers()
        .map_err(|e| Error::IOError(format!("reading header row: {e}")))?
        .iter()
        .enumerate()
        .map(|(i, h)| (h.trim().to_string(), i))
        .collect();

    let mut stars = Vec::new();
    let mut stats = ImportStats::default();
    let mut record = csv::StringRecord::new();
    while rdr
        .read_record(&mut record)
        .map_err(|e| Error::IOError(format!("reading record: {e}")))?
    {
        stats.read += 1;
        let row = Row {
            headers: &headers,
            record: &record,
        };
        match source.parse_row(&row) {
            None => stats.unparsed += 1,
            Some(star) if filter.keep(&star) => {
                stars.push(star);
                stats.kept += 1;
            }
            Some(_) => stats.filtered += 1,
        }
    }
    Ok((stars, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn import_str(
        data: &str,
        delim: u8,
        src: &dyn CatalogSource,
        filter: &ImportFilter,
    ) -> (Vec<Star>, ImportStats) {
        import_catalog(data.as_bytes(), delim, src, filter).unwrap()
    }

    #[test]
    fn gaia_csv_parses_named_columns() {
        let csv = "source_id,ra,dec,phot_g_mean_mag\n\
                   42,180.0,45.0,11.2\n\
                   43,10.5,-20.0,14.8\n";
        let (stars, stats) = import_str(csv, b',', &GaiaDr3, &ImportFilter::default());
        assert_eq!(stats.kept, 2);
        assert_eq!(stars[0].id, Some(42));
        assert!((stars[0].position.ra - 180.0).abs() < 1e-9);
        assert!((stars[0].magnitude - 11.2).abs() < 1e-9);
    }

    #[test]
    fn gaia_mag_limit_filters() {
        let csv = "source_id,ra,dec,phot_g_mean_mag\n\
                   1,180.0,45.0,11.0\n\
                   2,181.0,45.0,12.9\n\
                   3,182.0,45.0,13.4\n";
        let filter = ImportFilter {
            max_mag: Some(12.5),
            ..Default::default()
        };
        let (stars, stats) = import_str(csv, b',', &GaiaDr3, &filter);
        assert_eq!(stats.kept, 1);
        assert_eq!(stats.filtered, 2);
        assert_eq!(stars[0].id, Some(1));
    }

    #[test]
    fn tycho2_derives_johnson_v_and_skips_separator_lines() {
        // VizieR-style TSV: header, a units line, a dashed line, then data.
        let tsv = "RAmdeg\tDEmdeg\tBTmag\tVTmag\tTYC1\tTYC2\tTYC3\n\
                   deg\tdeg\tmag\tmag\t\t\t\n\
                   ------\t------\t------\t------\t----\t----\t----\n\
                   2.0\t3.0\t11.5\t11.0\t1\t8\t1\n";
        let (stars, stats) = import_str(tsv, b'\t', &Tycho2, &ImportFilter::default());
        assert_eq!(stats.kept, 1, "stats: {stats:?}");
        assert!(stats.unparsed >= 2, "units + dash lines should be skipped");
        // V = VT - 0.090*(BT-VT) = 11.0 - 0.090*0.5 = 10.955
        assert!(
            (stars[0].magnitude - 10.955).abs() < 1e-6,
            "{}",
            stars[0].magnitude
        );
        assert_eq!(stars[0].id, Some(1_000_000 + 8 * 10 + 1));
    }

    #[test]
    fn tycho2_falls_back_to_vt_without_bt() {
        let tsv = "RAmdeg\tDEmdeg\tVTmag\n\
                   2.0\t3.0\t9.5\n";
        let (stars, _) = import_str(tsv, b'\t', &Tycho2, &ImportFilter::default());
        assert_eq!(stars.len(), 1);
        assert!((stars[0].magnitude - 9.5).abs() < 1e-9);
    }

    #[test]
    fn region_cone_filters() {
        let csv = "source_id,ra,dec,phot_g_mean_mag\n\
                   1,180.0,45.0,10.0\n\
                   2,181.0,45.0,10.0\n\
                   3,10.0,-30.0,10.0\n";
        let filter = ImportFilter {
            region: Some((SkyCoord::new(180.0, 45.0), 2.0)),
            ..Default::default()
        };
        let (stars, stats) = import_str(csv, b',', &GaiaDr3, &filter);
        assert_eq!(stats.kept, 2);
        assert!(stars
            .iter()
            .all(|s| s.position.angular_distance(&SkyCoord::new(180.0, 45.0)) <= 2.0));
    }

    #[test]
    fn comment_lines_are_skipped() {
        let csv = "# generated by VizieR\nsource_id,ra,dec,phot_g_mean_mag\n1,180.0,45.0,10.0\n";
        let (stars, _) = import_str(csv, b',', &GaiaDr3, &ImportFilter::default());
        assert_eq!(stars.len(), 1);
    }
}
