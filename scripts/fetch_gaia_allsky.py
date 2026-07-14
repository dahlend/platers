#!/usr/bin/env python3
"""Resumable tiled all-sky Gaia DR3 fetch to a magnitude limit.

A single all-sky query for G<=13.5 is ~11M rows and the archive times out on big
jobs, so we tile the sky into RA/Dec boxes and fetch each as its own async job,
caching one parquet per box. Re-running skips boxes already on disk, so a flaky
archive just means re-running until every tile is present. Boxes tile cleanly
(half-open intervals), so there's no overlap to dedup.

Each tile carries `pmra`/`pmdec` (mas/yr, Gaia convention; positions are epoch
2016.0) for proper-motion propagation at solve time. The downstream build also
tolerates tiles without those columns (PM treated as unknown).

Usage: python scripts/fetch_gaia_allsky.py [--maglim 13.5] [--out DIR]
"""

from __future__ import annotations
import argparse, os, sys, time, warnings
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maglim", type=float, default=13.5)
    p.add_argument("--out", default="gaia_tiles")
    p.add_argument("--dec-step", type=float, default=9.0)
    p.add_argument("--ra-step", type=float, default=18.0)
    p.add_argument("--retries", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1

    dec_edges = np.arange(-90.0, 90.0 + 1e-6, args.dec_step)
    ra_edges = np.arange(0.0, 360.0 + 1e-6, args.ra_step)
    boxes = [
        (d0, d1, r0, r1)
        for d0, d1 in zip(dec_edges[:-1], dec_edges[1:])
        for r0, r1 in zip(ra_edges[:-1], ra_edges[1:])
    ]
    print(
        f"{len(boxes)} boxes ({args.dec_step} degx{args.ra_step} deg), maglim G<={args.maglim} -> {args.out}"
    )

    done = fetched = 0
    for i, (d0, d1, r0, r1) in enumerate(boxes):
        path = os.path.join(args.out, f"box_{i:04d}.parquet")
        if os.path.exists(path):
            done += 1
            continue
        q = (
            f"SELECT ra, dec, pmra, pmdec, phot_g_mean_mag AS g "
            f"FROM gaiadr3.gaia_source "
            f"WHERE phot_g_mean_mag <= {args.maglim} "
            f"AND dec >= {d0} AND dec < {d1} AND ra >= {r0} AND ra < {r1}"
        )
        for attempt in range(args.retries):
            try:
                r = Gaia.launch_job_async(q).get_results()
                ra = np.asarray(r["ra"], float)
                dec = np.asarray(r["dec"], float)
                g = np.asarray(r["g"], float)
                # Masked (missing) PM -> NaN -> null in the parquet.
                pmra = np.ma.filled(np.ma.asarray(r["pmra"], float), np.nan)
                pmdec = np.ma.filled(np.ma.asarray(r["pmdec"], float), np.nan)
                pq.write_table(
                    pa.table(
                        {
                            "ra": pa.array(ra),
                            "dec": pa.array(dec),
                            "mag": pa.array(g),
                            "pmra": pa.array(pmra.astype(np.float32), from_pandas=True),
                            "pmdec": pa.array(pmdec.astype(np.float32), from_pandas=True),
                        }
                    ),
                    path,
                )
                fetched += 1
                print(
                    f"  box {i:04d} dec[{d0:.0f},{d1:.0f}) ra[{r0:.0f},{r1:.0f}): "
                    f"{len(ra):>7,} stars  [{done + fetched}/{len(boxes)}]"
                )
                break
            except Exception as e:  # noqa: BLE001
                wait = 5 * (attempt + 1)
                print(
                    f"  box {i:04d} attempt {attempt + 1} failed ({str(e)[:60]}); retry in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
        else:
            print(f"  box {i:04d} GAVE UP after {args.retries} tries", file=sys.stderr)

    remaining = sum(
        1
        for i in range(len(boxes))
        if not os.path.exists(os.path.join(args.out, f"box_{i:04d}.parquet"))
    )
    print(
        f"done: {len(boxes) - remaining}/{len(boxes)} tiles present"
        + (
            f"  ({remaining} still missing -- re-run to retry)"
            if remaining
            else "  (complete)"
        )
    )
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
