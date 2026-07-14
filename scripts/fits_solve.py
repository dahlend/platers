#!/usr/bin/env python3
"""Extract sources from a real FITS image and plate-solve it with platers.

This bridges the gap between a raw `.fits` frame and the platers solver, which
ingests a *pre-extracted* star list (JSON `[{"x", "y", "flux"}, ...]`, 0-based
pixels). It also reads the frame's own header WCS, when present, as ground truth
so a solve can be scored automatically.

Pipeline
--------
1. Read the image (first 2-D HDU, e.g. LCO `SCI`) and its header WCS.
2. Detect point sources with `retego` (PSF-matched filter + saturated-star
   recovery, the default), or with photutils `DAOStarFinder`, or read an embedded
   SExtractor `CAT` table (`--extractor photutils|cat`).
3. Keep the brightest N (Tycho-2 is shallow; the solver wants bright stars).
4. Write the platers star list JSON.
5. With `--solve`, invoke the platers CLI and compare its center / scale /
   rotation to the header WCS.

Examples
--------
    # just extract sources -> stars.json
    python scripts/fits_solve.py frame.fits

    # extract and solve against the merged all-sky hybrid index, scored vs header
    python scripts/fits_solve.py frame.fits --solve \
        --index-dir data/index

    # batch over the bundled LCO frames
    python scripts/fits_solve.py *-e91.fits --solve \
        --index-dir data/index

Requires: numpy, astropy, scipy, and retego (or photutils for the fallback
extractor: `--extractor photutils`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def first_image_hdu(hdul):
    """The first HDU holding a 2-D image (LCO frames put it in `SCI`)."""
    for hdu in hdul:
        if hdu.data is not None and np.ndim(hdu.data) == 2:
            return hdu
    raise ValueError("no 2-D image HDU found")


def extract_photutils(data, fwhm, threshold_sigma, nstars, edge_margin):
    """Detect point sources with DAOStarFinder; return (N, 3) of x, y, flux.

    Coordinates are 0-based (numpy/array convention), which is exactly what the
    platers solver expects for image pixel coordinates.
    """
    from photutils.detection import DAOStarFinder

    data = np.asarray(data, dtype=float)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    # Lower `sharplo` only (default 0.2 -> 0.0): bright Tycho-2-magnitude stars --
    # the ones the index needs -- *saturate* in survey frames, giving flat-topped
    # (low-sharpness) profiles the default cut rejects. Keep `sharphi`/roundness at
    # their defaults so cosmic rays/hot pixels (high-sharpness spikes) stay rejected.
    # (For saturated survey data, the `retego` extractor is far better -- see below.)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std, sharplo=0.0)
    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        return np.empty((0, 3))

    h, w = data.shape
    x = np.asarray(sources["xcentroid"])
    y = np.asarray(sources["ycentroid"])
    flux = np.asarray(sources["flux"])

    # Drop detections hugging the edges (truncated PSFs give biased centroids).
    keep = (
        (x >= edge_margin)
        & (x < w - edge_margin)
        & (y >= edge_margin)
        & (y < h - edge_margin)
        & np.isfinite(flux)
        & (flux > 0)
    )
    x, y, flux = x[keep], y[keep], flux[keep]

    order = np.argsort(flux)[::-1][:nstars]  # brightest first
    return np.column_stack([x[order], y[order], flux[order]])


def recover_saturated_sources(data, sat_level, edge_margin, min_pixels=4, dilate=3):
    """Recover blown-out bright stars the PSF-matched finder misses.

    The index's quad anchors are the sky's brightest stars per cell -- and in
    crowded (galactic-plane) fields those are G~9-11 stars whose cores bloom far
    past any fitter's tolerance, so the PSF pipeline drops them entirely. Without
    them no correct quad forms. Recover them
    geometrically: threshold at the saturation level, DILATE to merge bleed-trail
    fragments into one region, then take the flux-weighted centroid (~0.5" -- enough
    to anchor a quad).

    Returns `(centroids, label_map)`: centroids is (N,3) x,y,counts (0-based;
    counts only rank the recovered stars relative to one another), and label_map
    is the labeled dilated saturation mask (0 = unsaturated), used to cull
    trail/halo fragments measured inside a region.
    """
    empty = np.empty((0, 3)), None
    if sat_level is None or not np.isfinite(sat_level):
        return empty
    from scipy import ndimage

    h, w = data.shape
    mask = ndimage.binary_dilation(data >= sat_level * 0.95, iterations=dilate)
    lbl, n = ndimage.label(mask)
    if n == 0:
        return empty
    coms = ndimage.center_of_mass(data, lbl, range(1, n + 1))
    counts = ndimage.sum(data, lbl, range(1, n + 1))
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    rows = [
        (cx, cy, c)
        for (cy, cx), c, s in zip(coms, counts, sizes)
        if s >= min_pixels
        and edge_margin <= cx < w - edge_margin
        and edge_margin <= cy < h - edge_margin
    ]
    return (np.array(rows) if rows else np.empty((0, 3))), lbl


def extract_retego(
    data,
    nstars,
    edge_margin,
    snr=5.0,
    fwhm=3.0,
    sharpness_factor=1.6,
    bg_order=2,
    sat_level=None,
    min_reliability=0.8,
):
    """Detect + measure sources with the `retego` pipeline; return (N,3) x,y,flux.

    retego uses a PSF-matched matched filter (better completeness than a Gaussian
    kernel finder), calibrates the PSF from the detections, and measures each source
    with saturation accounting -- which matters because bright Tycho-2-magnitude
    stars saturate in survey frames yet are exactly the index's quad anchors.
    Mildly saturated sources are kept (the fitter accounts for clipped pixels);
    heavily bloomed ones are recovered geometrically afterwards (see
    `recover_saturated_sources`). Coordinates are 0-based, as the solver expects.
    """
    import retego

    data = np.ascontiguousarray(data, dtype=np.float32)
    img = retego.Image(data)

    if sat_level is None or not np.isfinite(sat_level):
        try:
            sat_level = float(retego.estimate_saturation_level(img))
        except Exception:  # noqa: BLE001 - saturation estimate is best-effort
            sat_level = None

    # Remove the sky with a low-order polynomial background, not a single global
    # median: real frames have gradients/vignetting that a constant leaves behind,
    # seeding spurious detections in the bright regions. Fall back to a constant
    # fit if the polynomial fails.
    try:
        bgm = retego.BackgroundPolynomial.fit(img, bg_order, 3.0)
    except Exception:  # noqa: BLE001 - background fit is best-effort
        bgm = retego.BackgroundPolynomial.fit(img, 0, 3.0)
    frame = retego.ReducedFrame.reduce(img, bgm, saturation_level=sat_level)

    sigma = fwhm / 2.3548
    cands = retego.detect_candidates_psf_matched(frame, retego.GaussianPSF(sigma), snr)
    if not cands:
        return np.empty((0, 3))

    # Measure only the brightest candidates, not all of them. A single survey
    # exposure yields tens of thousands of candidates (mostly faint cosmic rays);
    # measuring all is slow AND lets faint-but-high-response cosmic rays into the
    # brightest-N. The strongest detections are star-dominated, so this is both much
    # faster (~3x) and cleaner (it's what lets the most CR-heavy frames solve).
    cands = sorted(cands, key=lambda c: c.brightness, reverse=True)[
        : max(4 * nstars, 800)
    ]
    try:
        psf = retego.fit_gaussian_psf(frame, cands)
    except Exception:  # noqa: BLE001 - PSF calibration is best-effort
        psf = retego.GaussianPSF(sigma)
    measurements = retego.measure_candidates(frame, cands, psf)
    sms = [m for m in measurements.per_candidate if m is not None]
    if not sms:
        return np.empty((0, 3))

    # Cuts on the measured sources:
    #  - saturated sources are NOT cut: retego masks clipped pixels and fits the
    #    wings, so both flux and centroid of a measured saturated star are sound
    #    (and the index anchors on exactly these stars).
    #  - `on_trail`: fragments sitting on a *neighbor's* bleed trail are junk.
    #  - `sharpness`: cosmic rays and hot pixels are sharp spikes. In single survey
    #    exposures (PTF/ZTF) they can be the *brightest* detections, crowding real
    #    catalog stars out of the per-cell quad anchors. The cut is ADAPTIVE --
    #    `factor x median sharpness` of the bright detections -- because the stellar
    #    sharpness scale depends on the PSF/seeing.
    h, w = data.shape
    base = [
        s
        for s in sms
        if np.isfinite(s.x)
        and np.isfinite(s.y)
        and np.isfinite(s.flux)
        and s.flux > 0
        and not s.on_trail
        and np.isfinite(s.sharpness)
        and edge_margin <= s.x < w - edge_margin
        and edge_margin <= s.y < h - edge_margin
    ]
    if not base:
        return np.empty((0, 3))

    # Reference the sharpness cut to the BRIGHT detections, not all: the full list
    # is dominated by faint cosmic rays, whose sharpness median can exceed a star's.
    ref_n = min(len(base), max(nstars, 50))
    bright = sorted(base, key=lambda s: s.flux, reverse=True)[:ref_n]
    sharp_cut = float(np.median([s.sharpness for s in bright])) * sharpness_factor
    # 4th column: retego's per-source reliability (population purity at the
    # source's significance) -- low values flag trails, ghosts, and other
    # poorly behaved detections.
    rows = [
        (s.x, s.y, s.flux, s.reliability if s.reliability is not None else 1.0)
        for s in base
        if s.sharpness <= sharp_cut
    ]
    arr = np.array(rows) if rows else np.empty((0, 4))

    sat, lbl = recover_saturated_sources(data, sat_level, edge_margin)
    if len(arr):
        # Two junk signals, one exemption:
        #  - inside a (dilated) saturated region away from its centroid = a
        #    bleed-trail / halo FRAGMENT (a bright star's trail can fragment
        #    into dozens of top-ranked fakes);
        #  - reliability < 0.8 = retego's own trails/ghosts/junk flag.
        # A source coinciding with a region centroid is the saturated star
        # itself and is always kept: real bloomed anchors can share a
        # junk-poisoned significance bucket on trail-heavy frames, since
        # reliability is a per-bucket population statistic.
        low_rel = arr[:, 3] < min_reliability
        if len(sat):
            ix = np.clip(arr[:, 0].round().astype(int), 0, lbl.shape[1] - 1)
            iy = np.clip(arr[:, 1].round().astype(int), 0, lbl.shape[0] - 1)
            in_region = lbl[iy, ix] > 0
            dmin = np.min(
                np.hypot(
                    arr[:, 0][:, None] - sat[None, :, 0],
                    arr[:, 1][:, None] - sat[None, :, 1],
                ),
                axis=1,
            )
            arr = arr[~((in_region | low_rel) & (dmin > 3.0))]
        else:
            arr = arr[~low_rel]
    if len(sat):
        # Add regions with no measured counterpart (the monsters the PSF path
        # never produced), ranked ABOVE every PSF detection: their counts are
        # meaningless but they must survive the brightest-N cut.
        if len(arr):
            fresh = [
                r
                for r in sat
                if not (np.hypot(arr[:, 0] - r[0], arr[:, 1] - r[1]) < 6.0).any()
            ]
            sat = np.array(fresh) if fresh else np.empty((0, 3))
        if len(sat):
            top = float(arr[:, 2].max()) if len(arr) else 1.0
            rank = np.argsort(np.argsort(sat[:, 2]))  # brightest region ranks highest
            sat = np.column_stack([sat[:, 0], sat[:, 1], top * 2.0 + rank, np.ones(len(sat))])
            arr = np.vstack([sat, arr]) if len(arr) else sat

    if len(arr) == 0:
        return arr
    return arr[np.argsort(arr[:, 2])[::-1][:nstars], :3]


def extract_cat_extension(hdul, nstars, edge_margin, shape):
    """Read an embedded SExtractor/SEP `CAT` table (LCO BANZAI frames have one).

    SExtractor pixel coordinates are 1-based (FITS convention) -> subtract 1.
    """
    cat = None
    for hdu in hdul:
        if hdu.name.upper() == "CAT" and hdu.data is not None:
            cat = hdu.data
            break
    if cat is None:
        raise ValueError("no CAT extension in this file")

    cols = {c.upper(): c for c in cat.columns.names}

    def pick(*names):
        for n in names:
            if n in cols:
                return np.asarray(cat[cols[n]], dtype=float)
        raise KeyError(f"CAT missing any of {names}; has {list(cols)}")

    x = pick("X", "XWIN_IMAGE", "X_IMAGE") - 1.0
    y = pick("Y", "YWIN_IMAGE", "Y_IMAGE") - 1.0
    flux = pick("FLUX", "FLUX_AUTO", "FLUX_BEST", "PEAK")

    h, w = shape
    keep = (
        (x >= edge_margin)
        & (x < w - edge_margin)
        & (y >= edge_margin)
        & (y < h - edge_margin)
        & np.isfinite(flux)
        & (flux > 0)
    )
    x, y, flux = x[keep], y[keep], flux[keep]
    order = np.argsort(flux)[::-1][:nstars]
    return np.column_stack([x[order], y[order], flux[order]])


def header_truth(header, width, height):
    """Ground-truth (center RA/Dec, pixel scale arcsec/px, rotation deg) from the
    header WCS, or None if the frame has no celestial WCS."""
    try:
        wcs = WCS(header)
        if not wcs.has_celestial:
            return None
        wcs = wcs.celestial
        # FITS-standard geometric center (0-based): pixel centers sit at
        # integers in the 1-based convention, so the array spans [0.5, N+0.5]
        # and the true center is (N-1)/2 here -- matching platers' own
        # reference_pixel, which is anchored at the same point.
        center = wcs.pixel_to_world((width - 1) / 2.0, (height - 1) / 2.0)
        scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
        cd = wcs.pixel_scale_matrix
        rotation = float(np.degrees(np.arctan2(cd[1, 0], cd[1, 1])))
        return {
            "ra": float(center.ra.deg),
            "dec": float(center.dec.deg),
            "scale": scale,
            "rotation": rotation,
        }
    except Exception as e:  # noqa: BLE001 - header WCS is best-effort
        print(f"  (no usable header WCS: {e})")
        return None


def pointing_from_header(header):
    """The telescope's *commanded* pointing (sexagesimal RA/DEC-type keywords),
    independent of the solved CRVAL -- the realistic "I roughly know where I pointed"
    hint. Handles the common keyword spellings (RA/DEC, OBJRA/OBJDEC, TELRA/TELDEC).
    Returns (ra_deg, dec_deg) or None."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    for ra_key, dec_key in [("RA", "DEC"), ("OBJRA", "OBJDEC"), ("TELRA", "TELDEC")]:
        if ra_key in header and dec_key in header:
            try:
                c = SkyCoord(
                    str(header[ra_key]), str(header[dec_key]), unit=(u.hourangle, u.deg)
                )
                return float(c.ra.deg), float(c.dec.deg)
            except Exception:  # noqa: BLE001
                continue
    return None


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcseconds."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cos = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) * 3600.0)


def epoch_from_header(header):
    """Observation epoch as a decimal Julian year, from OBSMJD / MJD-OBS /
    DATE-OBS -- or None. Lets the solver propagate catalog proper motions to
    the frame's date."""
    mjd = None
    for key in ("OBSMJD", "MJD-OBS", "MJDMID"):
        if key in header:
            try:
                mjd = float(header[key])
                break
            except (TypeError, ValueError):
                continue
    if mjd is None and "DATE-OBS" in header:
        try:
            from astropy.time import Time

            mjd = Time(header["DATE-OBS"]).mjd
        except Exception:  # noqa: BLE001
            return None
    if mjd is None:
        return None
    return 2000.0 + (mjd - 51544.5) / 365.25


def run_platers(args, stars_json, width, height, scale, pointing, truth, epoch=None):
    """Invoke the platers CLI and compare to the header WCS. `pointing` is the
    (ra, dec) hint or None (position-blind). Returns the parsed solution, or None."""
    out_path = stars_json.with_suffix(".solution.json")
    # Numeric options are passed as `--key=value` so a negative declination
    # isn't mis-parsed as a flag by the CLI.
    cmd = args.platers_cmd.split() + [
        "solve",
        "--input",
        str(stars_json),
        "--index-dir",
        str(args.index_dir),
        f"--width={width}",
        f"--height={height}",
        f"--scale-uncertainty={args.scale_uncertainty}",
        "--output",
        str(out_path),
    ]
    if scale is not None:
        cmd += [f"--scale={scale:.4f}"]
    if pointing is not None:
        cmd += [
            f"--ra={pointing[0]:.5f}",
            f"--dec={pointing[1]:.5f}",
            f"--radius={args.radius}",
        ]
    if epoch is not None:
        cmd += [f"--epoch={epoch:.3f}"]

    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(args.repo_root), capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        print("  platers did not solve:")
        for line in tail:
            print(f"     {line}")
        return None

    sol = json.loads(out_path.read_text())
    wcs = sol["wcs"]
    ra, dec = wcs["center"]["ra"], wcs["center"]["dec"]
    print(
        f"  solved: RA={ra:.5f} deg  Dec={dec:.5f} deg  "
        f"matches={sol['verification']['num_matches']}  "
        f"log_odds={sol['verification']['log_odds']:.1f}"
    )
    if truth is not None:
        err = angular_sep_arcsec(ra, dec, truth["ra"], truth["dec"])
        verdict = "PASS" if err < 30.0 else "OFF"
        print(
            f'  vs header: center error = {err:.2f}"  [{verdict}]  '
            f"(true RA={truth['ra']:.5f} deg Dec={truth['dec']:.5f} deg)"
        )
    return sol


def process(path: Path, args) -> bool:
    print(f"\n=== {path.name} ===")
    with fits.open(path) as hdul:
        img = first_image_hdu(hdul)
        data = img.data
        height, width = data.shape
        header = img.header
        truth = header_truth(header, width, height)
        pointing = None if args.blind else pointing_from_header(header)
        epoch = epoch_from_header(header)

        if args.extractor == "cat":
            sources = extract_cat_extension(
                hdul, args.nstars, args.edge_margin, data.shape
            )
            method = "embedded CAT table"
        elif args.extractor == "photutils":
            sources = extract_photutils(
                data, args.fwhm, args.threshold, args.nstars, args.edge_margin
            )
            method = "photutils DAOStarFinder"
        else:
            sat_level = None
            if not args.no_saturated_recovery:
                sat_level = args.saturation
                if sat_level is None:
                    for kw in ("SATURVAL", "SATURATE", "SATURAT", "MAXLIN"):
                        if kw in header:
                            sat_level = float(header[kw])
                            break
            sources = extract_retego(
                data,
                args.nstars,
                args.edge_margin,
                snr=args.threshold,
                fwhm=args.fwhm,
                sharpness_factor=args.sharpness_factor,
                bg_order=args.bg_order,
                sat_level=sat_level,
                min_reliability=args.min_reliability,
            )
            method = "retego PSF-matched"
            if sat_level is not None:
                method += f" (saturation-aware, sat={sat_level:.0f})"

    print(f"  {width}x{height} px; extracted {len(sources)} sources ({method})")
    if truth is not None:
        print(
            f"  header WCS: RA={truth['ra']:.5f} deg Dec={truth['dec']:.5f} deg "
            f'scale={truth["scale"]:.3f}"/px rot={truth["rotation"]:.2f} deg'
        )
    if len(sources) < 8:
        print("  too few sources to solve")
        return False

    stars = [{"x": float(x), "y": float(y), "flux": float(f)} for x, y, f in sources]
    out_json = path.with_suffix(".stars.json")
    out_json.write_text(json.dumps(stars))
    print(f"  wrote {out_json.name}")

    if not args.solve:
        scale = truth["scale"] if truth else "<unknown>"
        print(
            f"  to solve: platers solve --input {out_json.name} "
            f"--index-dir {args.index_dir} --width {width} --height {height} "
            f"--scale {scale}"
        )
        return True

    scale = truth["scale"] if (truth and not args.no_scale_hint) else args.scale
    if pointing is not None:
        print(
            f"  pointing hint (commanded): RA={pointing[0]:.5f} deg Dec={pointing[1]:.5f} deg"
        )
    sol = run_platers(args, out_json, width, height, scale, pointing, truth, epoch=epoch)
    return sol is not None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("fits", nargs="+", type=Path, help="FITS image(s)")
    p.add_argument(
        "--solve", action="store_true", help="run the platers CLI on each frame"
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="directory of .qidx indices (default: data/index)",
    )
    p.add_argument(
        "--nstars", type=int, default=150, help="keep the brightest N sources"
    )
    p.add_argument(
        "--fwhm", type=float, default=4.0, help="DAOStarFinder FWHM in pixels"
    )
    p.add_argument(
        "--threshold", type=float, default=5.0, help="detection threshold in sigma"
    )
    p.add_argument(
        "--bg-order",
        type=int,
        default=2,
        help="retego polynomial background order (0=constant; default 2)",
    )
    p.add_argument(
        "--sharpness-factor",
        type=float,
        default=1.6,
        help="retego: drop sources sharper than factor*median sharpness "
        "(adaptive cosmic-ray/hot-pixel cut; default 1.6)",
    )
    p.add_argument(
        "--min-reliability",
        type=float,
        default=0.8,
        help="retego: drop sources with reliability below this (trails, ghosts, "
        "junk); sources at a saturated-region centroid are always kept",
    )
    p.add_argument(
        "--edge-margin",
        type=float,
        default=16.0,
        help="drop sources within N px of an edge",
    )
    p.add_argument(
        "--saturation",
        type=float,
        default=None,
        help="saturation ADU level for recovering blown-out bright stars "
        "(default: read SATURVAL/SATURATE from the header)",
    )
    p.add_argument(
        "--no-saturated-recovery",
        action="store_true",
        help="disable geometric recovery of saturated bright stars (the "
        "catalog's quad anchors); recovery is on by default when a "
        "saturation level is available",
    )
    p.add_argument(
        "--extractor",
        choices=["retego", "photutils", "cat"],
        default="retego",
        help="source extractor (default: retego -- PSF-matched, saturation-aware)",
    )
    p.add_argument(
        "--blind",
        action="store_true",
        help="don't pass a position hint (default: hint with the header's "
        "commanded RA/Dec -- note the CLI's solve() floods on an all-sky index)",
    )
    p.add_argument(
        "--no-scale-hint",
        action="store_true",
        help="don't pass a scale hint (default: use the header pixel scale)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="pixel scale arcsec/px to hint when there's no header WCS",
    )
    p.add_argument("--scale-uncertainty", type=float, default=0.1)
    p.add_argument(
        "--radius", type=float, default=1.0, help="position-hint radius in degrees"
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="process N frames in parallel (extraction dominates; 4-6 is a good "
        "choice on a modern desktop)",
    )
    p.add_argument(
        "--platers-cmd",
        default=None,
        help="how to invoke the platers CLI (default: the prebuilt "
        "target/release/platers-cli if present, else `cargo run`)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root to run the CLI from",
    )
    args = p.parse_args()

    # Prefer the prebuilt binary -- `cargo run` re-checks the build on every frame,
    # which dominates wall time across a batch. Run `cargo build --release -p
    # platers-cli` once first.
    if args.platers_cmd is None:
        binary = args.repo_root / "target" / "release" / "platers-cli"
        args.platers_cmd = (
            str(binary)
            if binary.exists()
            else "cargo run -q --release -p platers-cli --"
        )

    if args.jobs > 1:
        # Frames are independent; extraction dominates wall time, so a small
        # process pool gives a near-linear speedup. Keep the solver's internal
        # rayon pool from oversubscribing alongside N parallel extractions.
        import os

        os.environ.setdefault(
            "RAYON_NUM_THREADS", str(max(2, (os.cpu_count() or 8) // args.jobs))
        )
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.jobs) as pool:
            results = []
            for ok, text in pool.imap(_process_one, [(f, args) for f in args.fits]):
                sys.stdout.write(text)
                results.append(ok)
    else:
        # One unreadable frame must not kill a batch: count it as unsolved and go on.
        results = [_print_and_return(*_process_one((f, args))) for f in args.fits]
    if args.solve:
        ok = sum(results)
        print(f"\n=== {ok}/{len(results)} solved ===")
        sys.exit(0 if ok == len(results) else 1)


def _process_one(payload):
    """Process one frame, capturing its output (for ordered parallel logs).
    One unreadable frame must not kill a batch: errors count as unsolved."""
    import contextlib
    import io

    f, args = payload
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            ok = process(f, args)
        except Exception as e:  # noqa: BLE001 - per-frame isolation
            print(f"\n=== {f.name} ===\n  ERROR: {e}")
            ok = False
    return ok, buf.getvalue()


def _print_and_return(ok, text):
    sys.stdout.write(text)
    return ok


if __name__ == "__main__":
    main()
