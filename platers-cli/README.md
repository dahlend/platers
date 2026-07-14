# platers-cli

Command-line interface for plate solving.

Solves the WCS of an image from a list of pre-extracted star positions (JSON,
0-based pixels) against a [`platers-core`](https://crates.io/crates/platers-core)
index. It does not detect stars -- run your own source extractor first and hand
it the star list.

```bash
platers-cli solve \
    --input stars.json --index-dir data/index \
    --width 2048 --height 1489 \
    --scale 0.39 --scale-uncertainty 0.05 \
    --output solution.json
```

`platers-cli info --index-dir data/index` summarizes an index. Scale and
position hints are optional but make solving much faster.

Part of the [Platers](https://github.com/ddahlen/platers) workspace; see the
workspace readme for extracting stars from raw FITS frames. Licensed under
MIT OR Apache-2.0.
