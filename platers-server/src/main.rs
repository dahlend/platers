//! platers-server entry point: load the dataset once, then serve.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use platers_server::{build_router, cors_layer, AppState};

/// Plate-solving web service.
#[derive(Parser, Debug)]
#[command(name = "platers-server", about = "Plate-solving web service")]
struct Args {
    /// Directory of merged `.qidx` index files.
    #[arg(long, default_value = "data/index")]
    index_dir: PathBuf,
    /// Catalog parquet used for refinement.
    #[arg(long, default_value = "data/catalog.parquet")]
    catalog: PathBuf,
    /// Address to bind the HTTP server to.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    /// Maximum concurrent solves (default: available parallelism). Each solve is
    /// rayon-parallel, so additional requests queue behind this cap.
    #[arg(long)]
    max_concurrency: Option<usize>,
    /// Extra solves allowed to queue beyond --max-concurrency before requests are
    /// shed with 503 (default: 4x max-concurrency).
    #[arg(long)]
    max_queued: Option<usize>,
    /// Hard per-solve wall-clock timeout in milliseconds (default: 30000).
    #[arg(long, default_value = "30000")]
    solve_timeout_ms: u64,
    /// Skip pre-faulting the index into RAM at startup. By default the whole index
    /// is touched so it is resident before serving (uniform latency); pass this for
    /// a faster boot at the cost of cold-page faults on early requests.
    #[arg(long)]
    no_prefault: bool,
    /// Append unsolved frames (request + diagnostics) to this JSONL file for offline
    /// replay. Omit to disable the failure log.
    #[arg(long)]
    failure_log: Option<PathBuf>,
    /// Allow browser cross-origin requests from these origins (repeatable; `*` allows
    /// any). Omit to disable CORS -- the default; non-browser clients are unaffected
    /// either way.
    #[arg(long)]
    cors_origin: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let max_concurrency = args.max_concurrency.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get)
    });
    let max_queued = args.max_queued.unwrap_or(max_concurrency.saturating_mul(4));
    let solve_timeout = Duration::from_millis(args.solve_timeout_ms);

    let started = Instant::now();
    tracing::info!(
        index_dir = %args.index_dir.display(),
        catalog = %args.catalog.display(),
        max_concurrency,
        max_queued,
        solve_timeout_ms = args.solve_timeout_ms,
        "loading dataset"
    );
    let state = Arc::new(AppState::load(
        &args.index_dir,
        &args.catalog,
        max_concurrency,
        max_queued,
        solve_timeout,
        args.failure_log.as_deref(),
    )?);
    tracing::info!(
        indices = state.index.len(),
        catalog_stars = state.catalog.len(),
        elapsed_s = started.elapsed().as_secs_f64(),
        "dataset loaded"
    );

    if args.no_prefault {
        tracing::info!("skipping index pre-fault (--no-prefault)");
    } else {
        let pf = Instant::now();
        let bytes = state.index.prefault();
        tracing::info!(
            gib = bytes as f64 / 1e9,
            elapsed_s = pf.elapsed().as_secs_f64(),
            "index pre-faulted resident"
        );
    }

    let mut app = build_router(state);
    if let Some(cors) = cors_layer(&args.cors_origin) {
        tracing::info!(origins = ?args.cors_origin, "CORS enabled");
        app = app.layer(cors);
    }

    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    tracing::info!(addr = %args.bind, "listening");
    // `into_make_service_with_connect_info` exposes the peer address to handlers
    // (the `ConnectInfo` extractor), for requester-IP logging.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
