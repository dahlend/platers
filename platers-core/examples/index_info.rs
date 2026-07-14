//! Print per-tier statistics for an index directory.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p platers-core --example index_info -- index
//! ```

use std::path::Path;

use platers_core::{Error, IndexSet, PlatersResult};

fn main() -> PlatersResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir] = args.as_slice() else {
        return Err(Error::ValueError("usage: index_info <index_dir>".into()));
    };
    let index = IndexSet::load_from_directory(Path::new(dir))
        .map_err(|e| Error::IOError(format!("loading index from {dir}: {e}")))?;
    println!("{} index tier(s) in {dir}", index.len());
    for idx in index.all_indices() {
        println!(
            "  {}: scale {:.2}-{:.2} arcsec/px, quad diameter {:.1}-{:.1} arcmin, \
             {} quads, {} stars",
            idx.path.display(),
            idx.scale_range.0,
            idx.scale_range.1,
            idx.diameter_range.0 * 60.0,
            idx.diameter_range.1 * 60.0,
            idx.num_quads(),
            idx.num_stars()
        );
    }
    Ok(())
}
