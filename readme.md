<p align="center">
  <img src="assets/logo.svg" alt="Platers logo" width="130">
</p>

# Platers

Platers is a Rust library and toolset for computing the astrometry of a
sky image -- *plate solving*. It identifies star asterisms with geometric hashing and
is built to solve **many images quickly**: the index is memory-mapped and stays
resident, so each solve runs without re-reading the sky.

Platers produces a **first-pass** WCS -- a sub-arcsecond solution good enough to
stand on its own or to seed a later step. Squeezing out state-of-the-art
astrometry takes a further refinement pass against a deep catalog, which is
outside Platers' scope.

## How it works

A solve takes a list of detected source pixel positions (plus brightnesses) and
recovers the WCS -- where on the sky the image was taken, at what scale and rotation.

1. **Uniformize** the detected stars on a grid sized to the index's `HEALPix`
   cells (brightest-per-cell), so density is even regardless of how crowded the
   field is.
2. **Build quads** -- 4-star asterisms -- and hash each into a 4-D geometric code that
   is invariant to translation, scale, and rotation (and matched in both parities).
3. **Match** image-quad codes against the catalog index via a KD-tree range query.
4. **Verify** each candidate pose with a distance-weighted Bayesian test
   (log-odds), so a real match is distinguished from chance alignments.
5. **Refine** the winning pose with a least-squares TAN fit against the catalog
   stars around the solved center.

## Workspace layout

| crate | purpose |
|---|---|
| `platers-core` | core solving library -- quad hashing, KD-tree matching, verification, refinement, the mmap'd `.qidx` index format |
| `platers-build` | index-building tools (`build_index`, `merge_scale_qidx`, `build_catalog`) |
| `platers-cli` | command-line solver (`solve`, `info`) |
| `platers-server` | web service -- submit pixel detections over HTTP, get a WCS back |
| `platers-tests` | integration tests and benchmarks |

## Quickstart (no dataset needed)

The real dataset is multi-GB and takes hours to build (below), but you can see a
full solve in under a minute using the committed ~3 MB test-fixture catalog:

```bash
# 1. Build a tiny index from the fixture catalog (seconds)
cargo run --release -p platers-build --bin build_index -- \
    --catalog platers-tests/fixtures/fixture_catalog.parquet \
    --output demo_tiles --min-scale-arcmin 3.0 --max-scale-deg 0.4
cargo run --release -p platers-build --bin merge_scale_qidx -- demo_tiles demo_index

# 2. Generate a synthetic detected-star field over that catalog
#    (writes demo_stars.json and prints the ground-truth pose)
cargo run --release -p platers-tests --example make_demo_field

# 3. Solve it and compare against the printed ground truth
cargo run --release -p platers-cli -- solve \
    --input demo_stars.json --index-dir demo_index \
    --width 2048 --height 1489 --scale 0.88
```

The fixture covers one sky region (around RA 180, Dec +45), so it only solves
fields generated over it -- for real images, build the all-sky dataset below.

## The dataset

The default dataset lives in `data/` (gitignored -- it's large and rebuildable):

- **`data/catalog.parquet`** -- the reference catalog: Gaia DR3 to G <= 13.5
  (Gaia-preferred), plus Tycho-2 for the bright stars Gaia misses. ~11.2M stars.
  Used for verification, refinement, and distortion (SIP) fitting.
- **`data/index/`** -- the all-sky quad index (`.qidx`), built at a uniform target
  **stellar density** (~70 stars/deg^2) rather than a flat magnitude cut, so index
  size is decoupled from catalog depth and small fields still get enough stars.
  (Without `--target-density`, the builder falls back to a flat `--stars-per-cell`
  cap, default 10 -- what the quickstart above uses.)
- **`data/sources/`** -- the inputs the catalog is rebuilt from (raw Gaia tiles and
  the Tycho-2 catalog).

A catalog is just a Parquet table of `ra, dec, mag` in degrees (plus an `id`).

## Building the dataset

Needs `pip install numpy scipy pyarrow astroquery`, ~10 GB free disk, and
patience: step 1 runs for **hours** (400 async archive jobs, but resumable --
re-run until it prints `(complete)`), and step 3 is CPU-bound for **tens of
minutes to hours**. Always build with `--release`.

```bash
# 0. One-time: convert a Tycho-2 export (VizieR I/259, delimited w/ header)
#    into the bright-star fill catalog
cargo run --release -p platers-build --bin build_catalog -- \
    --source tycho2 --input tycho2.tsv --max-mag 12.0 \
    --output data/sources/tycho2_full/catalog.parquet

# 1. Fetch Gaia DR3 to G<=13.5, tiled and resumable (slow -- see above)
python scripts/fetch_gaia_allsky.py --maglim 13.5

# 2. Merge Gaia + Tycho-2 bright fill into the reference catalog (minutes;
#    Gaia wins any duplicate within 2", Tycho-2 fills the saturated bright end)
python scripts/build_allsky_hybrid.py --out data/catalog.parquet

# 3. Build the density-targeted quad index, then merge per-scale all-sky files
cargo run --release -p platers-build --bin build_index -- \
    --catalog data/catalog.parquet --output data/index_tiles \
    --min-scale-arcmin 3.0 --max-scale-deg 0.45 --target-density 70
cargo run --release -p platers-build --bin merge_scale_qidx -- \
    data/index_tiles data/index

# 4. Sanity-check the result (per-tier scales, quad/star counts)
cargo run --release -p platers-cli -- info --index-dir data/index
```

Steps 0-2 only change with a new magnitude limit or source catalog; steps 3-4
re-run from the existing `data/catalog.parquet` whenever index parameters
change. `data/index_tiles` is an intermediate and can be deleted after the
merge.

## Solving

The solver works on a **pre-extracted** star list -- it does *not* detect stars, so
run your detection (`photutils`, `SExtractor`, `retego`, ...) first and hand it a JSON array
of 0-based pixel positions and brightnesses:

```json
[
  {"x": 512.5, "y": 512.5, "flux": 1000.0},
  {"x": 312.8, "y": 215.6, "flux": 850.0}
]
```

```bash
cargo run --release -p platers-cli -- solve \
    --input stars.json --index-dir data/index \
    --width 2048 --height 4096 \
    --scale 1.0 --scale-uncertainty 0.1 \
    --ra 214.6 --dec -1.7 --radius 3.0 \
    --output solution.json --verbose
```

Scale and position hints are optional but make solving much faster; with none, the
solver falls back to a (slower) blind search. By default the final fit refines
against the index's own (uniformized) stars; `--catalog data/catalog.parquet`
instead refines against the dense catalog around the solved center -- more matched
stars and lower residuals, the same as the server. `--sip-order N` (2-5, off by
default) additionally fits SIP distortion polynomials of order `N` to the final
solution -- for optics with real field distortion; the default solution is a pure
linear TAN WCS. `--epoch YEAR` (e.g. `--epoch 2021.4`, off by default) propagates
catalog proper motions from the catalog epoch (2016.0) to the observation date
before refinement -- worth ~0.1" for frames taken years from the epoch (needs
`--catalog` pointed at a catalog with the `pmra`/`pmdec` columns, since the index's
own stars carry no proper motions). `--output` writes the result as JSON -- the
fitted WCS plus the verification (matched-star count and log-odds) and solve counts.
`--max-quads` / `--max-hypotheses` cap the search (defaults 300000 / 200000).
`--verbose` streams the quad/hypothesis counts and log-odds as it runs, and
`platers-cli info --index-dir data/index` summarizes an index (scales, quad/star
counts, diameter ranges).

To use the library directly, see `platers-core/examples/`
(`cargo run --release -p platers-core --example solve_from_json`).

To go straight from a raw FITS frame, `scripts/fits_solve.py` extracts sources
(PSF-matched detection plus recovery of saturated bright stars) and solves against
`data/index`, scoring the result against the frame's header WCS when present.
It additionally needs `pip install numpy astropy scipy retego` (or `photutils`
in place of `retego` for the fallback extractor, `--extractor photutils`; or
`--extractor cat` to read sources from an embedded SExtractor table instead of
detecting them):

```bash
python scripts/fits_solve.py frame.fits.gz --solve --radius 3.0
```

## Running the server

For repeated solving, run the service instead of the CLI: it loads the index and
catalog once, keeps them resident (no per-request reload), and answers over HTTP.

```bash
cargo run --release -p platers-server
```

Defaults: serves on `127.0.0.1:8080`, loads `data/index` + `data/catalog.parquet`,
and pre-faults the index into RAM before accepting requests. Flags:

```bash
cargo run --release -p platers-server -- \
    --index-dir data/index --catalog data/catalog.parquet \
    --bind 0.0.0.0:8080 \
    --max-concurrency 8 \      # simultaneous solves (default: CPU cores)
    --max-queued 32 \          # extra solves allowed to queue before 503 (default: 4x concurrency)
    --solve-timeout-ms 10000 \ # per-solve wall-clock deadline -> 504 (default: 30000)
    --failure-log failures.jsonl \  # append unsolved frames for offline replay
    --cors-origin '*' \        # allow browser cross-origin calls (repeatable; omit to disable)
    --no-prefault              # skip the startup pre-fault (faster boot, colder first requests)
```

Endpoints:
- `POST /solve` -- body `{ "stars": [{"x","y","flux"}, ...], "width", "height",
  "scale", "scale_uncertainty"?, "ra"?, "dec"?, "radius"?, "sip_order"?, "epoch"? }`.
  `scale` (arcsec/pixel) is required; position is optional; `sip_order` (2-5) opts
  into a SIP distortion fit (default: linear WCS); `epoch` (a Julian year, e.g.
  `2021.4`) propagates catalog proper motions to the observation date. Returns the
  WCS solution, or `{"solved": false, ...}` if it could not solve. Under load it
  sheds with `503` (`Retry-After`).
- `GET /healthz` -- readiness (ready once the dataset is loaded).
- `GET /info` -- per-scale index stats and catalog size.
- `GET /metrics` -- Prometheus counters (outcomes, solve durations).

```bash
curl -s http://127.0.0.1:8080/solve -H 'Content-Type: application/json' \
    -d '{"stars":[{"x":512.5,"y":512.5,"flux":1000}, ...],
         "width":2048,"height":2048,"scale":1.0,"ra":214.6,"dec":-1.7,"radius":3.0}'
```

## Querying from Python

Failing to solve is not an HTTP error -- check the `solved` field on a 200.
Under load the server sheds with 503 (`Retry-After`); a solve past the
deadline is a 504. Needs at least 8 stars; only the brightest 500 are used.

```python
import requests

r = requests.post("http://127.0.0.1:8080/solve", json={
    "stars": [{"x": x, "y": y, "flux": f} for x, y, f in detections],  # 0-based pixels
    "width": 2048, "height": 2048,
    "scale": 1.0,                  # arcsec/pixel, required
    "ra": 214.6, "dec": -1.7, "radius": 3.0,   # optional position hint
}, timeout=60)
sol = r.json()
if sol["solved"]:
    w = sol["wcs"]     # center.{ra,dec}, reference_pixel.{x,y}, cd1_1..cd2_2 (deg/px)
    print(w["center"], sol["verification"]["num_matches"], sol["verification"]["log_odds"])
```

The `wcs` object maps directly onto a FITS TAN WCS (astropy: `crval` =
`center`, `cd` = the CD matrix, `crpix` = `reference_pixel` **+ 1** -- platers
is 0-based, FITS is 1-based).

## Notable details

- **Density-targeted, per-tier index.** Each scale tier carries a uniform sky
  density; fine (small-FOV) tiers reach deeper than wide-field tiers, so small fields
  stay solvable without bloating the index.
- **Both-parity matching.** The geometric hash is not reflection-invariant, so each
  image quad is searched in both handednesses (image flips/negative-parity CDs).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion in this work
by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
