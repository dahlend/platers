# platers-server

Web service for plate solving.

Loads a [`platers-core`](https://crates.io/crates/platers-core) index and its
catalog once at startup and keeps them resident, so solve requests reuse the
memory-mapped data instead of reloading it per query. Solve a frame by POSTing a
star list; failures are counted, logged, and optionally persisted for offline
replay.

```bash
platers-server --index-dir data/index --catalog data/catalog.parquet \
    --bind 127.0.0.1:8080
```

Endpoints: `POST /solve`, `GET /healthz`, `GET /info`, `GET /metrics`
(Prometheus). Under load the server sheds with `503`; a solve past its deadline
returns `504`.

Part of the [Platers](https://github.com/ddahlen/platers) workspace; see the
workspace readme for the request/response schema. Licensed under MIT OR
Apache-2.0.
