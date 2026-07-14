#!/usr/bin/env python3
"""Assemble the all-sky hybrid catalog: Gaia DR3 (deep, accurate) + Tycho-2 fill.

Concatenates the tiled Gaia fetch, then adds Tycho-2 stars ONLY where Gaia has no
counterpart (the bright stars Gaia saturates/misses, G<~6 and bright-end gaps).
Gaia is preferred on any duplicate, so a Tycho star is kept only if no Gaia star
lies within --match-radius of it. Uses a HEALPix bucket join so the dedup is
all-sky without an O(N^2) cross-match.

Usage: python scripts/build_allsky_hybrid.py \
    --gaia-tiles data/sources/gaia_allsky_tiles --tycho2 data/sources/tycho2_full/catalog.parquet \
    --out data/catalog.parquet
"""

from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


def radec_to_vec(ra, dec):
    r = np.radians(ra)
    d = np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gaia-tiles", default="data/sources/gaia_allsky_tiles")
    p.add_argument("--tycho2", default="data/sources/tycho2_full/catalog.parquet")
    p.add_argument("--out", default="data/catalog.parquet")
    p.add_argument(
        "--match-radius",
        type=float,
        default=2.0,
        help="Gaia<->Tycho dedup radius (arcsec)",
    )
    p.add_argument(
        "--nside", type=int, default=64, help="HEALPix nside for the dedup bucket join"
    )
    args = p.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.gaia_tiles, "box_*.parquet")))
    if not tiles:
        raise SystemExit(f"no Gaia tiles in {args.gaia_tiles}")
    gra, gdec, gmag, gpmra, gpmdec = [], [], [], [], []
    for t in tiles:
        tb = pq.read_table(t)
        n = tb.num_rows
        gra.append(np.asarray(tb["ra"]))
        gdec.append(np.asarray(tb["dec"]))
        gmag.append(np.asarray(tb["mag"]))
        # Tiles without PM columns: PM unknown -> NaN.
        for col, acc in (("pmra", gpmra), ("pmdec", gpmdec)):
            if col in tb.column_names:
                acc.append(np.asarray(tb[col].to_pandas(), dtype=np.float32))
            else:
                acc.append(np.full(n, np.nan, dtype=np.float32))
    gra = np.concatenate(gra)
    gdec = np.concatenate(gdec)
    gmag = np.concatenate(gmag)
    gpmra = np.concatenate(gpmra)
    gpmdec = np.concatenate(gpmdec)
    print(
        f"Gaia: {len(gra):,} stars from {len(tiles)} tiles (G {gmag.min():.1f}-{gmag.max():.1f})"
    )

    ty = pq.read_table(args.tycho2)
    tra = np.asarray(ty["ra"])
    tdec = np.asarray(ty["dec"])
    tmag = np.asarray(ty["mag"])
    print(f"Tycho-2: {len(tra):,} stars (mag {tmag.min():.1f}-{tmag.max():.1f})")

    # Gaia-preferred dedup via a KD-tree on 3D unit vectors: keep a Tycho star only
    # if the nearest Gaia star is farther than match-radius. cKDTree handles ~11M
    # points easily (chord distance <-> angular separation is monotonic for small
    # angles, so a chord upper bound is exact for a 2" cut).

    gvec = radec_to_vec(gra, gdec)
    tvec = radec_to_vec(tra, tdec)
    chord_thresh = 2.0 * np.sin(np.radians(args.match_radius / 3600.0) / 2.0)
    tree = cKDTree(gvec)
    dist, _ = tree.query(tvec, k=1, distance_upper_bound=chord_thresh)
    keep = ~np.isfinite(dist)  # inf => no Gaia neighbour within the cut => keep Tycho
    print(
        f'Tycho-2 kept (no Gaia within {args.match_radius}"): {int(keep.sum()):,} '
        f"(mostly bright: median G {np.median(tmag[keep]):.1f})"
    )

    ra = np.concatenate([gra, tra[keep]])
    dec = np.concatenate([gdec, tdec[keep]])
    mag = np.concatenate([gmag, tmag[keep]])
    # Tycho-2 survivors (the few brightest stars with no Gaia counterpart) have
    # no PM in our reduced Tycho parquet: leave PM null so the solver treats
    # their positions as epoch-less rather than propagating garbage.
    pmra = np.concatenate([gpmra, np.full(int(keep.sum()), np.nan, dtype=np.float32)])
    pmdec = np.concatenate([gpmdec, np.full(int(keep.sum()), np.nan, dtype=np.float32)])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "ra": pa.array(ra),
                "dec": pa.array(dec),
                "mag": pa.array(mag),
                "id": pa.array(np.arange(len(ra), dtype=np.int64)),
                "pmra": pa.array(pmra, from_pandas=True),
                "pmdec": pa.array(pmdec, from_pandas=True),
            }
        ),
        args.out,
    )
    print(
        f"hybrid all-sky: {len(ra):,} stars -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
