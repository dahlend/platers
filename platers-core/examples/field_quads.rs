//! Diagnostic: for a sky position, report each index tier's quads centered
//! within a radius, and -- given a star list with reference sky positions --
//! how many of those quads are fully covered by the extraction and how far
//! their image-side hash codes sit from the stored catalog codes.
//!
//! This separates the classic failure modes of a non-solving field: no index
//! coverage (few quads in field), an extraction problem (member stars missing),
//! or a geometry problem (members present but hashes out of tolerance).
//!
//! The star list is a JSON array of `[ra, dec, x, y]` rows: raw pixel
//! positions plus their sky positions through a reference WCS (e.g. the
//! frame's header WCS).
//!
//! Usage: `field_quads <index_dir> <ra> <dec> <radius_deg> [stars.json]`

use platers_core::geometry::compute_hash_code_pixels;
use platers_core::{Error, IndexSet, PixelCoord, PlatersResult, SkyCoord};

fn main() -> PlatersResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err(Error::ValueError(
            "usage: field_quads <index_dir> <ra> <dec> <radius_deg> [stars.json]".into(),
        ));
    }
    let index = IndexSet::load_from_directory(&args[1])?;
    let center = SkyCoord::new_normalized(args[2].parse().unwrap(), args[3].parse().unwrap());
    let radius: f64 = args[4].parse().unwrap();

    let (extracted, pixels): (Vec<SkyCoord>, Vec<PixelCoord>) = if args.len() > 5 {
        let text = std::fs::read_to_string(&args[5])?;
        let rows: Vec<(f64, f64, f64, f64)> = serde_json::from_str(&text)
            .map_err(|e| Error::ValueError(format!("parsing {}: {e}", args[5])))?;
        rows.into_iter()
            .map(|(ra, dec, x, y)| (SkyCoord::new_normalized(ra, dec), PixelCoord::new(x, y)))
            .unzip()
    } else {
        (Vec::new(), Vec::new())
    };

    for idx in index.all_indices() {
        let mut in_field = 0_usize;
        let mut fully_extracted = 0_usize;
        let mut member_hits = 0_usize;
        let mut members = 0_usize;
        let mut hash_dists: Vec<f64> = Vec::new();
        for qi in 0..idx.num_quads() {
            let quad = idx.quad(qi);
            let stars: Vec<SkyCoord> = quad
                .star_indices
                .iter()
                .map(|&si| idx.star(si).position)
                .collect();
            let centroid = SkyCoord::centroid_of_four(&[stars[0], stars[1], stars[2], stars[3]]);
            if centroid.angular_distance(&center) > radius {
                continue;
            }
            in_field += 1;
            if extracted.is_empty() {
                continue;
            }
            // Match each member to its nearest extracted star within 3 arcsec.
            let matched: Vec<Option<usize>> = stars
                .iter()
                .map(|s| {
                    (0..extracted.len())
                        .filter(|&i| extracted[i].angular_distance(s) * 3600.0 < 3.0)
                        .min_by(|&a, &b| {
                            extracted[a]
                                .angular_distance(s)
                                .total_cmp(&extracted[b].angular_distance(s))
                        })
                })
                .collect();
            let hits = matched.iter().flatten().count();
            members += 4;
            member_hits += hits;
            if hits == 4 {
                fully_extracted += 1;
                // Image-side hash from the matched pixel positions, both
                // parities, vs the stored catalog hash.
                let px: Vec<PixelCoord> = matched.iter().map(|m| pixels[m.unwrap()]).collect();
                let img = [px[0], px[1], px[2], px[3]];
                let mirrored = img.map(|p| PixelCoord::new(-p.x, p.y));
                let mut best = f64::INFINITY;
                for quad_px in [img, mirrored] {
                    if let Ok(h) = compute_hash_code_pixels(&quad_px) {
                        best = best.min(h.distance(&quad.hash_code));
                    }
                }
                hash_dists.push(best);
            }
        }
        hash_dists.sort_by(f64::total_cmp);
        let quantile = |q: f64| -> f64 {
            if hash_dists.is_empty() {
                f64::NAN
            } else {
                #[allow(clippy::cast_sign_loss, reason = "quantile q is in [0, 1]")]
                let i = ((hash_dists.len() - 1) as f64 * q) as usize;
                hash_dists[i]
            }
        };
        let hit_rate = if members > 0 {
            100.0 * member_hits as f64 / members as f64
        } else {
            0.0
        };
        println!(
            "{}: {} quads in field, {} fully extracted (member hit rate {hit_rate:.0}%), \
             hash dist p10/p50/p90 = {:.4}/{:.4}/{:.4}",
            idx.path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            in_field,
            fully_extracted,
            quantile(0.1),
            quantile(0.5),
            quantile(0.9),
        );
    }
    Ok(())
}
