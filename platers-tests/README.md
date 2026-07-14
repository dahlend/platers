# platers-tests

Internal integration tests and benchmarks for the
[Platers](https://github.com/ddahlen/platers) workspace.

Not published (`publish = false`). Holds the shared test harness
(`src/test_utils.rs` -- synthetic field generation with configurable degradation,
a committed fixture index) and the cross-crate integration/benchmark suites that
exercise the full solve pipeline end to end. Run with `cargo test`; the
dataset-dependent and long-running cases are `#[ignore]`d by default.
