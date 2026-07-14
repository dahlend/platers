//! mmap'd, pre-sorted quad index -- the `.qidx` format.
//!
//! The balanced 4D KD-tree over the quad hash codes is built **once, at
//! index-build time**, and written to disk in the exact array order the search
//! walks (an *implicit* median tree: the node of any sub-range `[lo, hi)` is its
//! midpoint, splitting on dimension `depth % 4`). At solve time the file is
//! **mmap'd** and the range search runs directly over the borrowed `[f32; 4]`
//! code array -- no deserialize, no tree construction, and the OS page cache
//! keeps a hot index resident across solves. A blind solve, which touches many
//! tiles, therefore pays no per-tile tree rebuild.
//!
//! The same layout pre-builds the **3D star tree** (`star_kd`) used for
//! verification, so a solve does no spatial-tree construction over the catalog
//! either -- just mmap and query.
//!
//! ## Layout (`.qidx`, little-endian, all sections 16-byte aligned)
//! - `Header` (96 bytes): magic, version, counts, scale/diameter bands, and the
//!   byte offset of each section.
//! - `quads`: `n_quads x QuadRec` -- the `[f32; 4]` code interleaved with the
//!   four catalog-star indices, in pre-sorted (implicit-median) tree order.
//! - `star_pos`: `n_stars x [f64; 2]` -- (ra, dec) degrees, original star order.
//! - `star_meta`: `n_stars x StarMeta` -- id (`u64::MAX` = none) and magnitude.
//! - `star_kd`: `n_stars x StarNode` -- the stars' `f32` unit vectors in
//!   implicit-median 3D tree order, each tagged with its `star_pos` index.
//!
//! The reader casts each section straight out of the mmap with `bytemuck` (the
//! 16-byte record sizes and 16-aligned offsets keep every cast aligned, since the
//! mmap base is page-aligned). `FlatCodeIndex::search` is the quad matcher for
//! [`crate::index::LoadedIndex`] -- a within-tolerance range query over the code
//! tree, tested against brute force.

use crate::errors::{Error, PlatersResult};
use crate::geometry::{chord_sq_for_angle, HashCode};
use crate::types::{SkyCoord, Star};
use bytemuck::{Pod, Zeroable};
use memmap2::{Advice, Mmap};

/// Page stride for pre-faulting. 4 KiB is the smallest common page size, so
/// touching every 4 KiB touches at least once per page on any platform.
const PAGE_SIZE: usize = 4096;
use std::path::Path;

/// File magic for a `.qidx` index.
const MAGIC: [u8; 8] = *b"PLQTIDX\0";
/// On-disk format version (bumped on any incompatible layout change).
const FORMAT_VERSION: u32 = 1;
/// Sentinel star id meaning "no id" (catalog star had `id == None`).
const NO_ID: u64 = u64::MAX;

/// Fixed-size file header. `#[repr(C)]` with naturally-aligned fields and no
/// padding (the 32-byte prefix is 8-aligned, so the `f64`/`u64` tail is too),
/// which keeps it a valid `Pod`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version: u32,
    n_quads: u32,
    n_stars: u32,
    /// `HEALPix` depth of the build grid (uniformization meta; carried for tools).
    healpix_depth: u32,
    /// Brightest-per-cell cap used at build (uniformization meta).
    stars_per_cell: u32,
    _pad0: u32,
    scale_min: f64,
    scale_max: f64,
    diam_min: f64,
    diam_max: f64,
    off_quads: u64,
    off_star_pos: u64,
    off_star_meta: u64,
    off_star_kd: u64,
}

const HEADER_SIZE: usize = size_of::<Header>();

/// One quad: its 4D hash code interleaved with the four catalog-star indices it
/// is built from. Stored in implicit-median KD-tree order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct QuadRec {
    code: [f32; 4],
    star_ids: [u32; 4],
}

/// Per-star metadata: catalog id (`NO_ID` = none) and magnitude.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StarMeta {
    id: u64,
    magnitude: f32,
    _pad: f32,
}

/// One node of the 3D star KD-tree: a star's unit vector (`f32`, ~0.02" on the
/// sphere -- ample for matching) plus its index into the `star_pos`/`star_meta`
/// arrays. Stored in implicit-median tree order (split dim = `depth % 3`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct StarNode {
    xyz: [f32; 3],
    idx: u32,
}

/// Round a byte length up to the next 16-byte boundary (keeps every section
/// 16-aligned, so `bytemuck` slice casts out of the page-aligned mmap are sound).
const fn align16(n: usize) -> usize {
    (n + 15) & !15
}

/// Arrange `recs` in place into implicit-median KD-tree order: the node of any
/// sub-range `[lo, hi)` is its midpoint, partitioned so the left half holds the
/// smaller values in dimension `depth % 4`. Recursing the same way at search time
/// reproduces the tree without any stored child pointers.
fn arrange(recs: &mut [QuadRec], depth: u32) {
    if recs.len() <= 1 {
        return;
    }
    let dim = (depth % 4) as usize;
    let mid = recs.len() / 2;
    let _ = recs.select_nth_unstable_by(mid, |a, b| {
        a.code[dim]
            .partial_cmp(&b.code[dim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (left, rest) = recs.split_at_mut(mid);
    arrange(left, depth + 1);
    arrange(&mut rest[1..], depth + 1);
}

/// Implicit-median arrange for the 3D star tree (split dim = `depth % 3`).
fn arrange_stars(nodes: &mut [StarNode], depth: u32) {
    if nodes.len() <= 1 {
        return;
    }
    let dim = (depth % 3) as usize;
    let mid = nodes.len() / 2;
    let _ = nodes.select_nth_unstable_by(mid, |a, b| {
        a.xyz[dim]
            .partial_cmp(&b.xyz[dim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (left, rest) = nodes.split_at_mut(mid);
    arrange_stars(left, depth + 1);
    arrange_stars(&mut rest[1..], depth + 1);
}

/// Squared Euclidean distance between an `f64` query vector and an `f32` node
/// vector (the star tree stores `f32` unit vectors).
fn dist2(q: &[f64; 3], xyz: &[f32; 3]) -> f64 {
    q.iter()
        .zip(xyz.iter())
        .map(|(&a, &b)| {
            let df = a - f64::from(b);
            df * df
        })
        .sum()
}

/// Nearest-neighbor descent over the implicit-median 3D star tree on `[lo, hi)`.
/// Updates `best = (dist2, star_index)`. Visits the near child first, then the far
/// child only when the splitting plane is closer than the current best -- standard
/// KD nearest-neighbor pruning.
fn nn_recurse(
    nodes: &[StarNode],
    q: &[f64; 3],
    lo: usize,
    hi: usize,
    depth: u32,
    best: &mut (f64, usize),
) {
    if lo >= hi {
        return;
    }
    let mid = lo + (hi - lo) / 2;
    let node = nodes[mid];
    let d2 = dist2(q, &node.xyz);
    if d2 < best.0 {
        *best = (d2, node.idx as usize);
    }
    let dim = (depth % 3) as usize;
    let diff = q[dim] - f64::from(node.xyz[dim]);
    let (near_lo, near_hi, far_lo, far_hi) = if diff <= 0.0 {
        (lo, mid, mid + 1, hi)
    } else {
        (mid + 1, hi, lo, mid)
    };
    nn_recurse(nodes, q, near_lo, near_hi, depth + 1, best);
    if diff * diff < best.0 {
        nn_recurse(nodes, q, far_lo, far_hi, depth + 1, best);
    }
}

/// Write a `.qidx` file from a star list and its quads.
///
/// `quads` are `(hash_code, star_indices)` pairs; the star indices reference
/// `stars`. The codes are pre-sorted into KD-tree order here so the reader never
/// has to build a tree.
///
/// # Errors
/// Returns an error on I/O failure, or if a star index is out of range.
pub fn write_qidx<P: AsRef<Path>>(
    path: P,
    stars: &[Star],
    quads: &[(HashCode, [usize; 4])],
    scale_range: (f64, f64),
    diameter_range: (f64, f64),
    healpix_depth: u8,
    stars_per_cell: usize,
) -> PlatersResult<()> {
    let n_stars = stars.len();
    let mut recs: Vec<QuadRec> = Vec::with_capacity(quads.len());
    for (hash, idx) in quads {
        let mut star_ids = [0_u32; 4];
        for (slot, &si) in star_ids.iter_mut().zip(idx.iter()) {
            if si >= n_stars {
                return Err(Error::ValueError(format!(
                    "quad star index {si} out of range (n_stars = {n_stars})"
                )));
            }
            *slot = si as u32;
        }
        recs.push(QuadRec {
            code: [
                hash.components[0] as f32,
                hash.components[1] as f32,
                hash.components[2] as f32,
                hash.components[3] as f32,
            ],
            star_ids,
        });
    }
    arrange(&mut recs, 0);

    let star_pos: Vec<[f64; 2]> = stars
        .iter()
        .map(|s| [s.position.ra, s.position.dec])
        .collect();
    let star_meta: Vec<StarMeta> = stars
        .iter()
        .map(|s| StarMeta {
            id: s.id.unwrap_or(NO_ID),
            magnitude: s.magnitude as f32,
            _pad: 0.0,
        })
        .collect();

    // Pre-build the 3D star KD-tree: unit vectors in implicit-median order, each
    // tagged with its original star index (so quad star ids and `star_pos`/`meta`
    // stay in input order). A solve then queries this in place -- no per-solve
    // spatial-tree build over millions of stars.
    let mut star_nodes: Vec<StarNode> = stars
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let v = s.position.to_unit_vector();
            StarNode {
                xyz: [v[0] as f32, v[1] as f32, v[2] as f32],
                idx: i as u32,
            }
        })
        .collect();
    arrange_stars(&mut star_nodes, 0);

    let off_quads = align16(HEADER_SIZE);
    let off_star_pos = align16(off_quads + size_of_val(recs.as_slice()));
    let off_star_meta = align16(off_star_pos + size_of_val(star_pos.as_slice()));
    let off_star_kd = align16(off_star_meta + size_of_val(star_meta.as_slice()));
    let total = align16(off_star_kd + size_of_val(star_nodes.as_slice()));

    let header = Header {
        magic: MAGIC,
        version: FORMAT_VERSION,
        n_quads: recs.len() as u32,
        n_stars: n_stars as u32,
        healpix_depth: u32::from(healpix_depth),
        stars_per_cell: stars_per_cell as u32,
        _pad0: 0,
        scale_min: scale_range.0,
        scale_max: scale_range.1,
        diam_min: diameter_range.0,
        diam_max: diameter_range.1,
        off_quads: off_quads as u64,
        off_star_pos: off_star_pos as u64,
        off_star_meta: off_star_meta as u64,
        off_star_kd: off_star_kd as u64,
    };

    let mut buf = vec![0_u8; total];
    buf[..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
    {
        let bytes = bytemuck::cast_slice(recs.as_slice());
        buf[off_quads..off_quads + bytes.len()].copy_from_slice(bytes);
    }
    {
        let bytes = bytemuck::cast_slice(star_pos.as_slice());
        buf[off_star_pos..off_star_pos + bytes.len()].copy_from_slice(bytes);
    }
    {
        let bytes = bytemuck::cast_slice(star_meta.as_slice());
        buf[off_star_meta..off_star_meta + bytes.len()].copy_from_slice(bytes);
    }
    {
        let bytes = bytemuck::cast_slice(star_nodes.as_slice());
        buf[off_star_kd..off_star_kd + bytes.len()].copy_from_slice(bytes);
    }

    let tmp = path.as_ref().with_extension("qidx.tmp");
    std::fs::write(&tmp, &buf)
        .map_err(|e| Error::IOError(format!("writing {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path.as_ref())
        .map_err(|e| Error::IOError(format!("finalizing {}: {e}", path.as_ref().display())))?;
    Ok(())
}

/// An mmap'd, pre-sorted quad index. Borrows its arrays directly from the mapped
/// file -- opening is just the mmap plus a header check, with no per-tile tree
/// construction.
#[derive(Debug)]
pub(crate) struct FlatCodeIndex {
    mmap: Mmap,
    n_quads: usize,
    n_stars: usize,
    off_quads: usize,
    off_star_pos: usize,
    off_star_meta: usize,
    off_star_kd: usize,
    /// (min, max) pixel scale this index covers (arcsec/pixel).
    pub(crate) scale_range: (f64, f64),
    /// (min, max) quad diameter this index covers (degrees).
    pub(crate) diameter_range: (f64, f64),
    /// `HEALPix` build-grid depth.
    pub(crate) healpix_depth: u8,
    /// Brightest-per-cell cap used at build time.
    pub(crate) stars_per_cell: usize,
}

impl FlatCodeIndex {
    /// Open and validate a `.qidx` file, mapping it into memory.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened/mapped, the magic or version
    /// is wrong, or the section offsets do not fit the file.
    pub(crate) fn open<P: AsRef<Path>>(path: P) -> PlatersResult<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::IOError(format!("opening {}: {e}", path.display())))?;
        // SAFETY: we only ever read the mapping, and the file is a fixed index
        // artifact; a concurrent truncation would be a build/deploy bug, not a
        // normal condition.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::IOError(format!("mmap {}: {e}", path.display())))?;

        if mmap.len() < HEADER_SIZE {
            return Err(Error::IOError(format!(
                "{}: too small to be a qidx file",
                path.display()
            )));
        }
        let header: Header = *bytemuck::from_bytes(&mmap[..HEADER_SIZE]);
        if header.magic != MAGIC {
            return Err(Error::IOError(format!(
                "{}: bad magic (not a qidx file)",
                path.display()
            )));
        }
        if header.version != FORMAT_VERSION {
            return Err(Error::IOError(format!(
                "{}: qidx version {} != supported {FORMAT_VERSION}",
                path.display(),
                header.version
            )));
        }

        let n_quads = header.n_quads as usize;
        let n_stars = header.n_stars as usize;
        let off_quads = header.off_quads as usize;
        let off_star_pos = header.off_star_pos as usize;
        let off_star_meta = header.off_star_meta as usize;
        let off_star_kd = header.off_star_kd as usize;
        let end = |off: usize, count: usize, rec: usize| {
            count
                .checked_mul(rec)
                .and_then(|bytes| off.checked_add(bytes))
        };
        let fits = [
            end(off_quads, n_quads, size_of::<QuadRec>()),
            end(off_star_pos, n_stars, 16),
            end(off_star_meta, n_stars, size_of::<StarMeta>()),
            end(off_star_kd, n_stars, size_of::<StarNode>()),
        ]
        .iter()
        .all(|e| e.is_some_and(|x| x <= mmap.len()));
        if !fits {
            return Err(Error::IOError(format!(
                "{}: section offsets exceed file length",
                path.display()
            )));
        }
        // The writer 16-aligns every section. Require that here so the
        // `bytemuck::cast_slice` accessors can never hit an alignment panic on a
        // corrupt or hand-crafted header (the mmap base is page-aligned, so a
        // 16-aligned offset is aligned for every record type).
        let aligned = [off_quads, off_star_pos, off_star_meta, off_star_kd]
            .iter()
            .all(|&off| off >= HEADER_SIZE && off % 16 == 0);
        if !aligned {
            return Err(Error::IOError(format!(
                "{}: misaligned section offsets",
                path.display()
            )));
        }

        Ok(Self {
            mmap,
            n_quads,
            n_stars,
            off_quads,
            off_star_pos,
            off_star_meta,
            off_star_kd,
            scale_range: (header.scale_min, header.scale_max),
            diameter_range: (header.diam_min, header.diam_max),
            healpix_depth: header.healpix_depth as u8,
            stars_per_cell: header.stars_per_cell as usize,
        })
    }

    /// Force the whole mapping resident: advise the kernel we will need it, then
    /// touch one byte per page so the pages are faulted in now rather than on the
    /// first query (uniform latency from request #1). Returns the mapping size in
    /// bytes. Best-effort -- the advise is ignored where `madvise` is unavailable.
    #[must_use]
    pub(crate) fn prefault(&self) -> usize {
        let _ = self.mmap.advise(Advice::WillNeed);
        let bytes: &[u8] = &self.mmap;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < bytes.len() {
            acc = acc.wrapping_add(bytes[i]);
            i += PAGE_SIZE;
        }
        let _ = std::hint::black_box(acc);
        bytes.len()
    }

    fn quads(&self) -> &[QuadRec] {
        bytemuck::cast_slice(
            &self.mmap[self.off_quads..self.off_quads + self.n_quads * size_of::<QuadRec>()],
        )
    }

    fn star_pos(&self) -> &[[f64; 2]] {
        bytemuck::cast_slice(&self.mmap[self.off_star_pos..self.off_star_pos + self.n_stars * 16])
    }

    fn star_meta(&self) -> &[StarMeta] {
        bytemuck::cast_slice(
            &self.mmap
                [self.off_star_meta..self.off_star_meta + self.n_stars * size_of::<StarMeta>()],
        )
    }

    fn star_kd(&self) -> &[StarNode] {
        bytemuck::cast_slice(
            &self.mmap[self.off_star_kd..self.off_star_kd + self.n_stars * size_of::<StarNode>()],
        )
    }

    /// Reconstruct catalog star `i` (original star order) from `star_pos`/`meta`,
    /// reading the mmap directly (no owned copy of the catalog).
    #[must_use]
    pub(crate) fn star(&self, i: usize) -> Star {
        let p = self.star_pos()[i];
        let m = self.star_meta()[i];
        Star {
            position: SkyCoord::new_normalized(p[0], p[1]),
            magnitude: f64::from(m.magnitude),
            id: (m.id != NO_ID).then_some(m.id),
            // The `.qidx` format does not store proper motions (index star
            // positions only feed quad matching + verification, whose
            // tolerances dwarf a decade of PM).
            proper_motion: None,
        }
    }

    /// Number of quads in the index.
    #[must_use]
    pub(crate) fn num_quads(&self) -> usize {
        self.n_quads
    }

    /// Number of stars in the index.
    #[must_use]
    pub(crate) fn num_stars(&self) -> usize {
        self.n_stars
    }

    /// Materialize the star catalog (for building a [`crate::catalog_index::CatalogIndex`]
    /// and for verification). This is an `O(n_stars)` copy -- no tree construction.
    #[must_use]
    pub(crate) fn catalog(&self) -> Vec<Star> {
        (0..self.n_stars).map(|i| self.star(i)).collect()
    }

    /// All catalog stars within `radius_deg` of `center` (great-circle), via the
    /// pre-built 3D star tree -- no spatial-tree build at solve time.
    #[must_use]
    pub(crate) fn stars_near(&self, center: SkyCoord, radius_deg: f64) -> Vec<Star> {
        let nodes = self.star_kd();
        if nodes.is_empty() {
            return Vec::new();
        }
        let q = center.to_unit_vector();
        let r2 = chord_sq_for_angle(radius_deg.to_radians());
        let r = r2.sqrt();
        let mut out = Vec::new();
        let mut stack: Vec<(usize, usize, u32)> = vec![(0, nodes.len(), 0)];
        while let Some((lo, hi, depth)) = stack.pop() {
            if lo >= hi {
                continue;
            }
            let mid = lo + (hi - lo) / 2;
            let node = nodes[mid];
            // The bounds check guards against a corrupt file whose node ids
            // point past the catalog; such nodes are skipped, not fatal.
            if dist2(&q, &node.xyz) <= r2 && (node.idx as usize) < self.n_stars {
                out.push(self.star(node.idx as usize));
            }
            let dim = (depth % 3) as usize;
            let diff = q[dim] - f64::from(node.xyz[dim]);
            if diff <= r {
                stack.push((lo, mid, depth + 1));
            }
            if diff >= -r {
                stack.push((mid + 1, hi, depth + 1));
            }
        }
        out
    }

    /// The single nearest catalog star to `sky`, with its great-circle distance in
    /// arcseconds. The nearest node is found over the `f32` star tree; the returned
    /// distance is recomputed from the star's full-precision position. `None` only
    /// if the index has no stars.
    #[must_use]
    pub(crate) fn nearest(&self, sky: SkyCoord) -> Option<(Star, f64)> {
        let nodes = self.star_kd();
        if nodes.is_empty() {
            return None;
        }
        let q = sky.to_unit_vector();
        let mut best = (f64::INFINITY, 0_usize); // (dist2, star index)
        nn_recurse(nodes, &q, 0, nodes.len(), 0, &mut best);
        if best.1 >= self.n_stars {
            // Corrupt file: the winning node's id points past the catalog.
            return None;
        }
        let star = self.star(best.1);
        Some((star, sky.angular_distance(&star.position) * 3600.0))
    }

    /// The four catalog-star indices of quad `quad_idx` (in this index's catalog
    /// order, i.e. the order [`catalog`](Self::catalog) returns).
    #[must_use]
    pub(crate) fn quad_star_indices(&self, quad_idx: usize) -> [usize; 4] {
        let q = self.quads()[quad_idx];
        [
            q.star_ids[0] as usize,
            q.star_ids[1] as usize,
            q.star_ids[2] as usize,
            q.star_ids[3] as usize,
        ]
    }

    /// Find every quad whose code is within Euclidean `tolerance` of `query`,
    /// returning `(distance, quad_idx)` pairs. It walks the implicit-median tree,
    /// pruning a far subtree only when the split plane is more than `tolerance`
    /// away.
    #[must_use]
    pub(crate) fn search(&self, query: &[f64; 4], tolerance: f64) -> Vec<(f64, usize)> {
        let codes = self.quads();
        let tol2 = tolerance * tolerance;
        let mut out = Vec::new();
        // Explicit stack of (lo, hi, depth) sub-ranges to visit.
        let mut stack: Vec<(usize, usize, u32)> = vec![(0, codes.len(), 0)];
        while let Some((lo, hi, depth)) = stack.pop() {
            if lo >= hi {
                continue;
            }
            let mid = lo + (hi - lo) / 2;
            let c = codes[mid].code;
            let mut d2 = 0.0;
            for k in 0..4 {
                let df = query[k] - f64::from(c[k]);
                d2 += df * df;
            }
            if d2 <= tol2 {
                out.push((d2.sqrt(), mid));
            }
            let dim = (depth % 4) as usize;
            let diff = query[dim] - f64::from(c[dim]);
            if diff <= tolerance {
                stack.push((lo, mid, depth + 1));
            }
            if diff >= -tolerance {
                stack.push((mid + 1, hi, depth + 1));
            }
        }
        out
    }

    /// The hash code of quad `quad_idx` (as stored, `f32`-precision).
    #[must_use]
    pub(crate) fn quad_hash(&self, quad_idx: usize) -> HashCode {
        let c = self.quads()[quad_idx].code;
        HashCode::new_unchecked([
            f64::from(c[0]),
            f64::from(c[1]),
            f64::from(c[2]),
            f64::from(c[3]),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_codes(n: usize) -> Vec<[f64; 4]> {
        // Deterministic pseudo-random codes in [0, 1)^4 (no rand dep needed).
        let mut s = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1_u64 << 53) as f64
        };
        (0..n).map(|_| [next(), next(), next(), next()]).collect()
    }

    #[test]
    fn flat_search_matches_bruteforce() {
        let codes = rng_codes(2000);
        let stars: Vec<Star> = (0..codes.len() * 4)
            .map(|i| Star::with_id(i as f64 % 360.0, 0.0, 10.0, i as u64))
            .collect();
        let quads: Vec<(HashCode, [usize; 4])> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    HashCode::new_unchecked(*c),
                    [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3],
                )
            })
            .collect();

        let dir = std::env::temp_dir().join(format!("qidx_eq_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.qidx");
        write_qidx(&path, &stars, &quads, (1.0, 2.0), (0.1, 0.2), 5, 10).unwrap();
        let flat = FlatCodeIndex::open(&path).unwrap();

        let tol = 0.03;
        for q in rng_codes(200) {
            // The tree search must return exactly the codes a brute-force scan
            // finds within `tol`. Compare the SET of f32 code-vectors (the flat
            // file is re-sorted, so indices differ).
            let mut a: Vec<[u32; 4]> = flat
                .search(&q, tol)
                .into_iter()
                .map(|(_, idx)| flat.quads()[idx].code.map(f32::to_bits))
                .collect();
            let mut b: Vec<[u32; 4]> = codes
                .iter()
                .filter(|c| {
                    let d2: f64 = q
                        .iter()
                        .zip(c.iter())
                        .map(|(a, b)| {
                            let df = a - f64::from(*b as f32);
                            df * df
                        })
                        .sum();
                    d2.sqrt() <= tol
                })
                .map(|c| c.map(|v| (v as f32).to_bits()))
                .collect();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "flat vs brute-force mismatch for query {q:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_catalog_and_quads() {
        let stars = vec![
            Star::with_id(10.0, 20.0, 9.5, 42),
            Star::new(11.0, 20.0, 10.0),
            Star::with_id(12.0, 21.0, 8.0, 7),
            Star::with_id(13.0, 19.0, 11.0, 9),
        ];
        let quads = vec![(HashCode::new_unchecked([0.1, 0.2, 0.3, 0.4]), [0, 1, 2, 3])];
        let dir = std::env::temp_dir().join(format!("qidx_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.qidx");
        write_qidx(&path, &stars, &quads, (1.0, 2.0), (0.1, 0.2), 5, 10).unwrap();

        let flat = FlatCodeIndex::open(&path).unwrap();
        assert_eq!(flat.num_quads(), 1);
        assert_eq!(flat.num_stars(), 4);
        assert_eq!(flat.scale_range, (1.0, 2.0));
        assert_eq!(flat.diameter_range, (0.1, 0.2));

        let cat = flat.catalog();
        assert_eq!(cat.len(), 4);
        assert_eq!(cat[0].id, Some(42));
        assert_eq!(cat[1].id, None);
        assert!((cat[0].position.ra - 10.0).abs() < 1e-9);

        let idx = flat.quad_star_indices(0);
        assert_eq!(idx, [0, 1, 2, 3]);
    }

    #[test]
    fn star_tree_matches_bruteforce() {
        // Pseudo-random stars over the sphere; compare the mmap'd star tree's
        // nearest + stars_near against brute force.
        let mut s = 0xDEAD_BEEF_0000_0001_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1_u64 << 53) as f64
        };
        let stars: Vec<Star> = (0..3000_u32)
            .map(|i| Star::with_id(next() * 360.0, next() * 180.0 - 90.0, 10.0, u64::from(i)))
            .collect();
        let quads = vec![(HashCode::new_unchecked([0.1, 0.2, 0.3, 0.4]), [0, 1, 2, 3])];

        let dir = std::env::temp_dir().join(format!("qidx_star_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.qidx");
        write_qidx(&path, &stars, &quads, (1.0, 2.0), (0.1, 0.2), 5, 10).unwrap();
        let flat = FlatCodeIndex::open(&path).unwrap();

        for _ in 0..100 {
            let query = SkyCoord::new_normalized(next() * 360.0, next() * 180.0 - 90.0);

            // nearest: same star id as a brute-force scan.
            let (got, _d) = flat.nearest(query).unwrap();
            let bf = stars
                .iter()
                .min_by(|a, b| {
                    query
                        .angular_distance(&a.position)
                        .total_cmp(&query.angular_distance(&b.position))
                })
                .unwrap();
            assert_eq!(got.id, bf.id, "nearest mismatch at {query:?}");

            // stars_near: same SET of ids as a brute-force radius scan.
            let radius = 5.0;
            let mut tree_ids: Vec<u64> = flat
                .stars_near(query, radius)
                .iter()
                .filter_map(|s| s.id)
                .collect();
            let mut bf_ids: Vec<u64> = stars
                .iter()
                .filter(|s| query.angular_distance(&s.position) <= radius)
                .filter_map(|s| s.id)
                .collect();
            tree_ids.sort_unstable();
            bf_ids.sort_unstable();
            assert_eq!(tree_ids, bf_ids, "stars_near mismatch at {query:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
