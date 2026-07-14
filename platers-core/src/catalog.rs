//! Catalog I/O and star structures.

use crate::errors::{Error, PlatersResult};
use crate::types::{SkyCoord, Star};
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// The Julian year all catalog positions are referenced to (the Gaia DR3
/// epoch; the build pipeline propagates Tycho-2 positions here too). Star
/// proper motions displace positions relative to this epoch -- see
/// [`Star::position_at_epoch`].
pub const CATALOG_EPOCH: f64 = 2016.0;

/// Create the catalog schema.
fn catalog_schema() -> Schema {
    Schema::new(vec![
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
        Field::new("mag", DataType::Float64, false),
        Field::new("id", DataType::Int64, true),
        Field::new("pmra", DataType::Float32, true),
        Field::new("pmdec", DataType::Float32, true),
    ])
}

/// Save stars to a Parquet file.
///
/// The file will be compressed with Snappy and written in batches for efficiency.
///
/// # Errors
/// [`Error::IOError`] if the file cannot be created or written.
pub fn save_catalog_parquet<P: AsRef<Path>>(path: P, stars: &[Star]) -> PlatersResult<()> {
    // Write in batches for memory efficiency.
    const BATCH_SIZE: usize = 100_000;

    let file = File::create(path)?;
    let schema = Arc::new(catalog_schema());

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    for chunk in stars.chunks(BATCH_SIZE) {
        let ra_values: Vec<f64> = chunk.iter().map(|s| s.position.ra).collect();
        let dec_values: Vec<f64> = chunk.iter().map(|s| s.position.dec).collect();
        let mag_values: Vec<f64> = chunk.iter().map(|s| s.magnitude).collect();
        let id_values: Vec<Option<i64>> =
            chunk.iter().map(|s| s.id.map(u64::cast_signed)).collect();
        let pmra_values: Vec<Option<f32>> = chunk
            .iter()
            .map(|s| s.proper_motion.map(|pm| pm[0]))
            .collect();
        let pmdec_values: Vec<Option<f32>> = chunk
            .iter()
            .map(|s| s.proper_motion.map(|pm| pm[1]))
            .collect();

        // Typed as `ArrayRef` (`Arc<dyn Array>`) so the concrete arrays coerce
        // via unsizing without explicit `as` casts.
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(ra_values)),
            Arc::new(Float64Array::from(dec_values)),
            Arc::new(Float64Array::from(mag_values)),
            Arc::new(Int64Array::from(id_values)),
            Arc::new(Float32Array::from(pmra_values)),
            Arc::new(Float32Array::from(pmdec_values)),
        ];

        let batch = RecordBatch::try_new(schema.clone(), columns)?;

        writer.write(&batch)?;
    }

    // `close` returns file metadata we don't need.
    let _ = writer.close()?;
    Ok(())
}

/// Load stars from a Parquet file.
///
/// Expected schema:
/// - `ra`: f64 (degrees, at [`CATALOG_EPOCH`])
/// - `dec`: f64 (degrees, at [`CATALOG_EPOCH`])
/// - `mag`: f64 (magnitude)
/// - `id`: i64 (optional)
/// - `pmra`, `pmdec`: f32 mas/yr, Gaia convention (optional columns -- a
///   catalog written without them loads with `proper_motion: None`)
///
/// # Errors
/// [`Error::IOError`] if the file cannot be read, or [`Error::Catalog`] if a
/// required column is missing or has the wrong type.
pub fn load_catalog_parquet<P: AsRef<Path>>(path: P) -> PlatersResult<Vec<Star>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let arrow_reader = builder.build()?;

    let mut stars = Vec::new();

    for batch_result in arrow_reader {
        let batch = batch_result?;

        let ra_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Catalog("RA column not found or wrong type".to_string()))?;

        let dec_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Catalog("Dec column not found or wrong type".to_string()))?;

        let mag_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Catalog("Mag column not found or wrong type".to_string()))?;

        let id_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| Error::Catalog("ID column not found or wrong type".to_string()))?;

        // Proper-motion columns are optional: a catalog without them loads
        // with `proper_motion: None` (positions are then treated as epoch-less,
        // and `position_at_epoch` is the identity).
        let pm_arrays = match (batch.column_by_name("pmra"), batch.column_by_name("pmdec")) {
            (Some(pmra), Some(pmdec)) => {
                let pmra = pmra
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        Error::Catalog("pmra column has wrong type (expected f32)".to_string())
                    })?;
                let pmdec = pmdec
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        Error::Catalog("pmdec column has wrong type (expected f32)".to_string())
                    })?;
                Some((pmra, pmdec))
            }
            _ => None,
        };

        for i in 0..batch.num_rows() {
            let ra = ra_array.value(i);
            let dec = dec_array.value(i);
            let mag = mag_array.value(i);
            let id = if id_array.is_null(i) {
                None
            } else {
                Some(id_array.value(i).cast_unsigned())
            };
            let proper_motion = pm_arrays.and_then(|(pmra, pmdec)| {
                (!pmra.is_null(i) && !pmdec.is_null(i)).then(|| [pmra.value(i), pmdec.value(i)])
            });

            stars.push(Star {
                position: SkyCoord::new_normalized(ra, dec),
                magnitude: mag,
                id,
                proper_motion,
            });
        }
    }

    Ok(stars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_roundtrip() -> PlatersResult<()> {
        let mut fast = Star::with_id(180.1, 0.1, 13.0, 2);
        fast.proper_motion = Some([1000.0, -500.0]);
        let stars = vec![
            Star::with_id(180.0, 0.0, 12.5, 1),
            fast,
            Star::new(180.2, 0.2, 14.5),
        ];

        // Unique per-process path so parallel test runs cannot collide.
        let temp_file = std::env::temp_dir().join(format!(
            "platers_catalog_roundtrip_{}.parquet",
            std::process::id()
        ));

        // Write
        save_catalog_parquet(&temp_file, &stars)?;

        // Read
        let loaded_stars = load_catalog_parquet(&temp_file)?;

        assert_eq!(stars.len(), loaded_stars.len());

        for (original, loaded) in stars.iter().zip(loaded_stars.iter()) {
            assert_eq!(original.position.ra, loaded.position.ra);
            assert_eq!(original.position.dec, loaded.position.dec);
            assert_eq!(original.magnitude, loaded.magnitude);
            assert_eq!(original.id, loaded.id);
            assert_eq!(original.proper_motion, loaded.proper_motion);
        }

        // Cleanup (ignore failure -- the temp file may already be gone).
        let _ = std::fs::remove_file(temp_file);

        Ok(())
    }
}
