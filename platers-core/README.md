# platers-core

Core plate solving library for astronomical astrometry.

Given a list of detected star positions (from any source extractor) and a
pre-built quad index, `platers-core` recovers an image's WCS -- where on the sky
it was taken, at what scale and rotation -- using geometric hash-based pattern
matching. The index is memory-mapped and stays resident, so repeated solves run
without re-reading the sky.

```rust
use platers_core::{DetectedField, IndexSet, PlateSolver, QueryConfig, ScaleRange};

let index = IndexSet::load_from_directory("data/index")?;
let field = DetectedField::new(stars, 2048, 1489); // stars: Vec<DetectedStar>
let config = QueryConfig {
    scale_hint: Some(ScaleRange::from_nominal(0.39, 0.05)),
    ..QueryConfig::default()
};
let solution = PlateSolver::new(index, config).solve(&field)?;
```

Runnable examples are in `examples/`. Building an index is the job of the
`platers-build` crate.

Part of the [Platers](https://github.com/ddahlen/platers) workspace; see the
workspace readme for the full pipeline. Licensed under MIT OR Apache-2.0.
