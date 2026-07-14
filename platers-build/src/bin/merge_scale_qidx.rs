//! Merge a directory of per-tile `.qidx` indices into one all-sky `.qidx` per
//! scale, for fast blind solving. See [`platers_build::merge`].
//!
//! Usage: `merge_scale_qidx <src_dir> <dst_dir>`

use std::path::PathBuf;

use platers_build::merge_scale_indices;
use platers_core::PlatersResult;

fn main() -> PlatersResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <src_dir> <dst_dir>", args[0]);
        std::process::exit(2);
    }
    let src = PathBuf::from(&args[1]);
    let dst = PathBuf::from(&args[2]);
    let written = merge_scale_indices(&src, &dst)?;
    println!(
        "Wrote {} all-sky scale indices to {}",
        written.len(),
        dst.display()
    );
    Ok(())
}
