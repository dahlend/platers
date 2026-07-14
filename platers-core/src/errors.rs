//! Errors which may be raised by this crate.
//!
//! There is a single [`Error`] enum for the entire crate, and every fallible
//! function returns a [`PlatersResult`]. Variants carry a plain description of
//! what went wrong; conversions from underlying error types (IO, parquet,
//! arrow) are provided so that `?` works transparently.

use std::{error, fmt, io};

/// Result type used throughout this crate.
pub type PlatersResult<T> = Result<T, Error>;

/// Possible errors raised by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Input/output failed (filesystem, parquet, or arrow).
    IOError(String),

    /// Catalog data is malformed or missing required columns.
    Catalog(String),

    /// Geometry is degenerate or a coordinate transform failed.
    Geometry(String),

    /// A WCS could not be built, parsed, or fitted.
    InvalidWcs(String),

    /// Spatial index construction or query failed.
    Spatial(String),

    /// A coordinate is outside of its valid range.
    InvalidCoordinate(String),

    /// An input value was invalid or a required option was not provided.
    ValueError(String),

    /// Not enough input data to attempt the operation.
    InsufficientData(String),

    /// The solver ran to completion without finding a confident match.
    NoSolution(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IOError(s)
            | Self::Catalog(s)
            | Self::Geometry(s)
            | Self::InvalidWcs(s)
            | Self::Spatial(s)
            | Self::InvalidCoordinate(s)
            | Self::ValueError(s)
            | Self::InsufficientData(s)
            | Self::NoSolution(s) => write!(f, "{s}"),
        }
    }
}

impl error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::IOError(error.to_string())
    }
}

impl From<parquet::errors::ParquetError> for Error {
    fn from(error: parquet::errors::ParquetError) -> Self {
        Self::IOError(error.to_string())
    }
}

impl From<arrow::error::ArrowError> for Error {
    fn from(error: arrow::error::ArrowError) -> Self {
        Self::IOError(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_passes_message_through() {
        let err = Error::InvalidCoordinate("RA out of range".to_string());
        assert_eq!(err.to_string(), "RA out of range");
    }

    #[test]
    fn io_error_converts() {
        let err: Error = io::Error::other("disk on fire").into();
        assert_eq!(err, Error::IOError("disk on fire".to_string()));
    }
}
