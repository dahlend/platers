//! Web service for plate solving.
//!
//! Loads the index and catalog once at startup and keeps them resident, so solve
//! requests reuse the memory-mapped multi-GB data instead of reloading it per query
//! (the point of the service over the CLI).
//!
//! Provides `/healthz`, `/info`, `/metrics`, and `POST /solve`. Failures are a
//! first-class concern: every unsolved frame is counted, logged with the solver's
//! diagnostics, and (optionally) persisted to a failure log for offline replay.

// axum handlers are async fns by contract; some do not `.await` anything.
#![allow(
    clippy::unused_async,
    reason = "axum handlers must be async fns even when they do not await"
)]

use std::fmt::Write as _;
use std::io::Write as _;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{ConnectInfo, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use platers_core::types::{DetectedField, DetectedStar};
use platers_core::{
    load_catalog_parquet, CatalogIndex, Error, IndexSet, PlateSolver, PlatersResult, PositionHint,
    QueryConfig, RefinementConfig, ScaleRange,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};

/// Default fractional scale uncertainty when the request omits it (+/-10%).
const DEFAULT_SCALE_UNCERTAINTY: f64 = 0.1;
/// Default position-cone radius (degrees) when ra/dec are given but radius is not.
const DEFAULT_RADIUS_DEG: f64 = 5.0;
/// Upper bound on `width`/`height` (pixels). Far beyond any real detector; guards
/// the geometry math against absurd dimensions.
const MAX_DIMENSION: usize = 100_000;
/// Minimum detected stars required to attempt a solve.
const MIN_STARS: usize = 8;
/// Cap on detected stars per request; the faintest beyond this are dropped. The
/// quad budget saturates well below this, so extra stars only add overhead.
const MAX_STARS: usize = 500;

/// Outcome counters and timing, scraped at `/metrics`.
#[derive(Debug, Default)]
pub struct Metrics {
    solved: AtomicU64,
    no_solution: AtomicU64,
    bad_request: AtomicU64,
    error: AtomicU64,
    /// Shed at the bounded-queue limit (HTTP 503).
    rejected: AtomicU64,
    /// Exceeded the per-solve deadline (HTTP 504).
    timed_out: AtomicU64,
    /// Summed solve duration (ms) over actual solve attempts.
    solve_ms_total: AtomicU64,
    solve_attempts: AtomicU64,
}

impl Metrics {
    fn record_solved(&self, solve_ms: u64) {
        let _ = self.solved.fetch_add(1, Ordering::Relaxed);
        let _ = self.solve_ms_total.fetch_add(solve_ms, Ordering::Relaxed);
        let _ = self.solve_attempts.fetch_add(1, Ordering::Relaxed);
    }

    fn record_no_solution(&self, solve_ms: u64) {
        let _ = self.no_solution.fetch_add(1, Ordering::Relaxed);
        let _ = self.solve_ms_total.fetch_add(solve_ms, Ordering::Relaxed);
        let _ = self.solve_attempts.fetch_add(1, Ordering::Relaxed);
    }

    fn record_bad_request(&self) {
        let _ = self.bad_request.fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self) {
        let _ = self.error.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rejected(&self) {
        let _ = self.rejected.fetch_add(1, Ordering::Relaxed);
    }

    fn record_timed_out(&self, solve_ms: u64) {
        let _ = self.timed_out.fetch_add(1, Ordering::Relaxed);
        let _ = self.solve_ms_total.fetch_add(solve_ms, Ordering::Relaxed);
        let _ = self.solve_attempts.fetch_add(1, Ordering::Relaxed);
    }

    fn render_prometheus(&self) -> String {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut s = String::new();
        let _ = writeln!(
            s,
            "# HELP platers_solves_total Solve requests by outcome.\n\
             # TYPE platers_solves_total counter\n\
             platers_solves_total{{outcome=\"solved\"}} {}\n\
             platers_solves_total{{outcome=\"no_solution\"}} {}\n\
             platers_solves_total{{outcome=\"bad_request\"}} {}\n\
             platers_solves_total{{outcome=\"error\"}} {}\n\
             platers_solves_total{{outcome=\"rejected\"}} {}\n\
             platers_solves_total{{outcome=\"timeout\"}} {}",
            load(&self.solved),
            load(&self.no_solution),
            load(&self.bad_request),
            load(&self.error),
            load(&self.rejected),
            load(&self.timed_out),
        );
        let _ = writeln!(
            s,
            "# HELP platers_solve_duration_ms_total Summed solve duration over attempts.\n\
             # TYPE platers_solve_duration_ms_total counter\n\
             platers_solve_duration_ms_total {}\n\
             # HELP platers_solve_attempts_total Solve attempts (solved + no_solution).\n\
             # TYPE platers_solve_attempts_total counter\n\
             platers_solve_attempts_total {}",
            load(&self.solve_ms_total),
            load(&self.solve_attempts),
        );
        s
    }
}

/// Cap on the failure-log file size. Each record embeds the request's full star
/// list, so sustained failure traffic would otherwise grow the file until the
/// disk fills; past the cap new records are dropped (with a one-time warning).
const MAX_FAILURE_LOG_BYTES: u64 = 1 << 30; // 1 GiB

/// Append-only log of failed solves (one JSON object per line), so unsolved frames
/// can be replayed offline as the catalog/solver improve. Size-capped (see
/// `MAX_FAILURE_LOG_BYTES`).
#[derive(Debug)]
pub struct FailureLog {
    file: Mutex<std::fs::File>,
    /// Bytes in the file (resumes from the existing length on open).
    written: AtomicU64,
    cap_reported: std::sync::atomic::AtomicBool,
}

impl FailureLog {
    /// Open (creating/appending) the failure log at `path`.
    ///
    /// # Errors
    /// [`Error::IOError`] if the file cannot be opened for appending.
    pub fn open(path: &Path) -> PlatersResult<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| Error::IOError(format!("opening failure log {}: {e}", path.display())))?;
        let written = file.metadata().map_or(0, |m| m.len());
        Ok(Self {
            file: Mutex::new(file),
            written: AtomicU64::new(written),
            cap_reported: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Append one record. Best-effort: a write/lock failure is logged, not fatal,
    /// and records past the size cap are dropped.
    fn record(&self, record: &serde_json::Value) {
        let line = format!("{record}\n");
        let projected =
            self.written.fetch_add(line.len() as u64, Ordering::Relaxed) + line.len() as u64;
        if projected > MAX_FAILURE_LOG_BYTES {
            if !self.cap_reported.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    cap_bytes = MAX_FAILURE_LOG_BYTES,
                    "failure log reached its size cap; dropping further records"
                );
            }
            return;
        }
        let Ok(mut file) = self.file.lock() else {
            tracing::error!("failure log mutex poisoned");
            return;
        };
        if let Err(e) = file.write_all(line.as_bytes()) {
            tracing::error!(error = %e, "failed to append to failure log");
        }
    }
}

/// Shared, read-only server state, loaded once and reused across all requests.
///
/// The index is memory-mapped behind `Arc`s (so an `IndexSet` clone is a refcount
/// bump, not a data copy) and the catalog sits behind an `Arc`; handlers reuse the
/// resident data with no reload.
#[derive(Debug)]
pub struct AppState {
    /// Memory-mapped merged all-sky quad index.
    pub index: IndexSet,
    /// Dense catalog (full Gaia + Tycho) for refinement / SIP fitting.
    pub catalog: Arc<CatalogIndex>,
    /// Caps concurrent (CPU-bound, rayon-parallel) solves; held until the solve
    /// completes (even one orphaned by a timeout), so concurrency stays bounded.
    pub solve_permits: Arc<Semaphore>,
    /// Caps total in-flight solves (running + queued); excess is shed with 503.
    pub admission: Arc<Semaphore>,
    /// Hard wall-clock deadline per solve request.
    pub solve_timeout: Duration,
    /// Outcome counters / timing for `/metrics`.
    pub metrics: Metrics,
    /// Optional append-only log of failed solves.
    pub failure_log: Option<FailureLog>,
}

impl AppState {
    /// Load the merged index directory and the catalog parquet into resident state.
    ///
    /// `max_concurrency` caps simultaneous solves; up to `max_queued` more may wait
    /// before requests are shed (503). `solve_timeout` bounds each solve.
    /// `failure_log` (if set) records each unsolved frame for offline replay.
    ///
    /// # Errors
    /// [`Error::IOError`] if the index, catalog, or failure log cannot be opened.
    pub fn load(
        index_dir: &Path,
        catalog_path: &Path,
        max_concurrency: usize,
        max_queued: usize,
        solve_timeout: Duration,
        failure_log: Option<&Path>,
    ) -> PlatersResult<Self> {
        let index = IndexSet::load_from_directory(index_dir).map_err(|e| {
            Error::IOError(format!("loading index from {}: {e}", index_dir.display()))
        })?;
        let stars = load_catalog_parquet(catalog_path).map_err(|e| {
            Error::IOError(format!(
                "loading catalog from {}: {e}",
                catalog_path.display()
            ))
        })?;
        let failure_log = failure_log.map(FailureLog::open).transpose()?;
        let max_concurrency = max_concurrency.max(1);
        Ok(Self {
            index,
            catalog: Arc::new(CatalogIndex::new(stars)),
            solve_permits: Arc::new(Semaphore::new(max_concurrency)),
            admission: Arc::new(Semaphore::new(max_concurrency + max_queued)),
            solve_timeout,
            metrics: Metrics::default(),
            failure_log,
        })
    }
}

/// Build the HTTP router over the shared state.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/info", get(info))
        .route("/metrics", get(metrics))
        .route("/solve", post(solve))
        .with_state(state)
}

/// Build a CORS layer from a list of allowed origins, or `None` to leave CORS off.
///
/// An empty list disables CORS entirely. A list containing `"*"` allows any origin;
/// otherwise only the listed origins are allowed. Applies to `GET`/`POST` with a
/// `content-type` request header (preflight is handled automatically). CORS is a
/// browser mechanism only -- it does not restrict non-browser clients.
#[must_use]
pub fn cors_layer(origins: &[String]) -> Option<CorsLayer> {
    if origins.is_empty() {
        return None;
    }
    let layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE]);
    let layer = if origins.iter().any(|o| o == "*") {
        layer.allow_origin(Any)
    } else {
        let list: Vec<HeaderValue> = origins
            .iter()
            .filter_map(|o| HeaderValue::from_str(o).ok())
            .collect();
        layer.allow_origin(list)
    };
    Some(layer)
}

/// Liveness/readiness. The server only starts accepting connections after the index
/// and catalog are loaded, so reaching this handler means it is ready to serve.
async fn healthz() -> &'static str {
    "ok"
}

/// One scale tier of the loaded index.
#[derive(Serialize)]
struct ScaleInfo {
    file: String,
    scale_arcsec_per_pixel: (f64, f64),
    quad_diameter_arcmin: (f64, f64),
    num_quads: usize,
    num_stars: usize,
}

/// Summary of the resident dataset.
#[derive(Serialize)]
struct InfoResponse {
    num_indices: usize,
    catalog_stars: usize,
    scales: Vec<ScaleInfo>,
}

/// Index summary: per-scale quad/star counts and diameter ranges, plus the resident
/// catalog size.
async fn info(State(state): State<Arc<AppState>>) -> Json<InfoResponse> {
    let scales = state
        .index
        .all_indices()
        .iter()
        .map(|idx| ScaleInfo {
            file: idx
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            scale_arcsec_per_pixel: idx.scale_range,
            quad_diameter_arcmin: (idx.diameter_range.0 * 60.0, idx.diameter_range.1 * 60.0),
            num_quads: idx.num_quads(),
            num_stars: idx.num_stars(),
        })
        .collect();
    Json(InfoResponse {
        num_indices: state.index.len(),
        catalog_stars: state.catalog.len(),
        scales,
    })
}

/// Prometheus-format outcome counters and solve timing.
async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render_prometheus(),
    )
        .into_response()
}

/// A `POST /solve` request: a detected-star list plus image geometry and hints.
///
/// `scale` (pixel scale, arcsec/pixel) is required -- it restricts the search to the
/// matching scale tier(s). Position (`ra`/`dec`/`radius`) is optional; without it the
/// solve is a position-blind sweep over those tiers.
#[derive(Debug, Deserialize)]
struct SolveRequest {
    #[serde(default)]
    stars: Vec<DetectedStar>,
    width: Option<usize>,
    height: Option<usize>,
    scale: Option<f64>,
    scale_uncertainty: Option<f64>,
    ra: Option<f64>,
    dec: Option<f64>,
    radius: Option<f64>,
    sip_order: Option<u32>,
    epoch: Option<f64>,
}

/// First non-empty value of `name` (split on `,` for forwarding lists), trimmed.
fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// First parseable IP from `name` (handles `X-Forwarded-For` lists).
fn header_ip(headers: &HeaderMap, name: &str) -> Option<IpAddr> {
    header_value(headers, name).and_then(|s| s.parse().ok())
}

/// Who made the request, for logs and the failure record.
struct Requester {
    ip: Option<IpAddr>,
    user_agent: Option<String>,
}

impl Requester {
    /// Prefer the proxy-forwarded client IP (`X-Forwarded-For` / `X-Real-IP`), else
    /// the direct TCP peer.
    fn new(headers: &HeaderMap, peer: Option<SocketAddr>) -> Self {
        let ip = header_ip(headers, "x-forwarded-for")
            .or_else(|| header_ip(headers, "x-real-ip"))
            .or_else(|| peer.map(|a| a.ip()));
        Self {
            ip,
            user_agent: header_value(headers, header::USER_AGENT.as_str()),
        }
    }

    /// Truncated client IP for privacy: IPv4 drops the last octet (a `/24`), IPv6
    /// the last 80 bits (a `/48`). Coarse enough that it no longer identifies an
    /// individual, while still useful for correlation/diagnostics.
    fn ip_truncated(&self) -> String {
        match self.ip {
            Some(IpAddr::V4(v4)) => {
                let o = v4.octets();
                format!("{}.{}.{}.0", o[0], o[1], o[2])
            }
            Some(IpAddr::V6(v6)) => {
                let mut s = v6.segments();
                for seg in &mut s[3..] {
                    *seg = 0;
                }
                Ipv6Addr::from(s).to_string()
            }
            None => "unknown".to_owned(),
        }
    }
}

/// Build a `{ "error": message }` JSON response with the given status, recording a
/// `bad_request` and a warning. Boxed so a validation `Result` stays small.
fn reject_boxed(state: &AppState, client_ip: &str, message: &str) -> Box<Response> {
    state.metrics.record_bad_request();
    tracing::warn!(
        outcome = "bad_request",
        requester_ip = client_ip,
        reason = message,
        "solve rejected"
    );
    Box::new(
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": message })),
        )
            .into_response(),
    )
}

/// Current wall-clock in epoch milliseconds (for failure-log records).
fn epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// A validated `/solve` request, ready to run.
struct ValidatedSolve {
    field: DetectedField,
    config: QueryConfig,
    scale: f64,
    scale_uncertainty: f64,
    /// The request's position hint (`ra`, `dec`, `radius`), kept for the
    /// failure log.
    position: [Option<f64>; 3],
    /// Requested SIP distortion order (`None` = linear WCS, the default).
    sip_order: Option<u32>,
}

/// Validate a `/solve` request into a runnable [`ValidatedSolve`], or the
/// 400 response to send back (recorded as a `bad_request`).
fn validate_request(
    state: &AppState,
    client_ip: &str,
    req: SolveRequest,
) -> Result<ValidatedSolve, Box<Response>> {
    let scale = match req.scale {
        Some(s) if s.is_finite() && s > 0.0 => s,
        Some(_) => {
            return Err(reject_boxed(
                state,
                client_ip,
                "`scale` must be a positive, finite number",
            ))
        }
        None => {
            return Err(reject_boxed(
                state,
                client_ip,
                "`scale` (arcsec/pixel) is required",
            ))
        }
    };
    let (Some(width), Some(height)) = (req.width, req.height) else {
        return Err(reject_boxed(
            state,
            client_ip,
            "`width` and `height` are required",
        ));
    };
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(reject_boxed(
            state,
            client_ip,
            &format!("`width` and `height` must be in 1..={MAX_DIMENSION}"),
        ));
    }

    // Drop any non-finite detections (a NaN/Inf coordinate would poison the quad
    // geometry), then require a usable minimum.
    let mut stars: Vec<DetectedStar> = req
        .stars
        .into_iter()
        .filter(|s| s.x.is_finite() && s.y.is_finite() && s.flux.is_finite())
        .collect();
    if stars.len() < MIN_STARS {
        return Err(reject_boxed(
            state,
            client_ip,
            &format!(
                "need at least {MIN_STARS} stars with finite coordinates, got {}",
                stars.len()
            ),
        ));
    }
    // Cap to the brightest MAX_STARS to bound per-request work.
    if stars.len() > MAX_STARS {
        DetectedStar::sort_brightest_first(&mut stars);
        stars.truncate(MAX_STARS);
    }

    let scale_uncertainty = req.scale_uncertainty.unwrap_or(DEFAULT_SCALE_UNCERTAINTY);
    // A negative uncertainty inverts the scale range (guaranteed no-solution);
    // >= 1 makes the lower bound non-positive and defeats the scale filter.
    if !(0.0..1.0).contains(&scale_uncertainty) {
        return Err(reject_boxed(
            state,
            client_ip,
            "`scale_uncertainty` must be a fraction in [0, 1)",
        ));
    }
    // Quad/hypothesis budgets are the `QueryConfig` defaults, shared with the CLI.
    let mut config = QueryConfig {
        scale_hint: Some(ScaleRange::from_nominal(scale, scale_uncertainty)),
        ..QueryConfig::default()
    };
    if let (Some(ra), Some(dec)) = (req.ra, req.dec) {
        if !ra.is_finite() || !dec.is_finite() || !(-90.0..=90.0).contains(&dec) {
            return Err(reject_boxed(
                state,
                client_ip,
                "`ra` must be finite and `dec` in [-90, 90]",
            ));
        }
        let radius = req.radius.unwrap_or(DEFAULT_RADIUS_DEG);
        if !(radius > 0.0 && radius <= 180.0) {
            return Err(reject_boxed(
                state,
                client_ip,
                "`radius` must be in (0, 180] degrees",
            ));
        }
        config.position_hint = Some(PositionHint::new(ra, dec, radius));
    }

    // SIP is opt-in; the order is bounded (higher orders need more matched
    // stars than any request can guarantee, and only invite ill-conditioning).
    if let Some(order) = req.sip_order {
        if !(2..=5).contains(&order) {
            return Err(reject_boxed(
                state,
                client_ip,
                "`sip_order` must be in [2, 5]",
            ));
        }
    }

    // Observation epoch (Julian year) for proper-motion propagation; bounded
    // to the plausible range of astronomical imagery.
    if let Some(epoch) = req.epoch {
        if !(1850.0..=2150.0).contains(&epoch) {
            return Err(reject_boxed(
                state,
                client_ip,
                "`epoch` must be a Julian year in [1850, 2150]",
            ));
        }
        config.observation_epoch = Some(epoch);
    }

    Ok(ValidatedSolve {
        field: DetectedField::new(stars, width, height),
        config,
        scale,
        scale_uncertainty,
        position: [req.ra, req.dec, req.radius],
        sip_order: req.sip_order,
    })
}

impl ValidatedSolve {
    /// One failure-log record: the outcome plus enough of the request to replay
    /// the frame offline (see [`FailureLog`]).
    fn failure_record(
        &self,
        outcome: &str,
        reason: &str,
        client_ip: &str,
        user_agent: Option<&String>,
    ) -> serde_json::Value {
        serde_json::json!({
            "ts_ms": epoch_ms(),
            "outcome": outcome,
            "reason": reason,
            "requester": { "ip": client_ip, "user_agent": user_agent },
            "request": {
                "width": self.field.width, "height": self.field.height,
                "scale": self.scale, "scale_uncertainty": self.scale_uncertainty,
                "ra": self.position[0], "dec": self.position[1], "radius": self.position[2],
                "sip_order": self.sip_order,
                "epoch": self.config.observation_epoch,
                "n_stars": self.field.stars.len(), "stars": self.field.stars,
            },
        })
    }
}

/// Solve a frame against the resident index, refining against the resident catalog.
///
/// A frame that does not solve returns 200 with `{ "solved": false, "reason": ... }`
/// and is counted/logged/persisted as a failure; only malformed requests (400) and
/// internal faults (500) are HTTP errors. The solve is CPU-bound, so it runs on a
/// blocking thread behind a concurrency permit.
async fn solve(
    State(state): State<Arc<AppState>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(req): Json<SolveRequest>,
) -> Response {
    let requester = Requester::new(&headers, peer.map(|c| c.0));
    let client_ip = requester.ip_truncated();

    let validated = match validate_request(&state, &client_ip, req) {
        Ok(v) => Arc::new(v),
        Err(response) => return *response,
    };
    let n_stars = validated.field.stars.len();
    let scale = validated.scale;
    let position_given = validated.config.position_hint.is_some();

    // --- admission control: shed load past the bounded queue (running + queued) ---
    let Ok(_admit) = state.admission.clone().try_acquire_owned() else {
        state.metrics.record_rejected();
        tracing::warn!(
            outcome = "rejected",
            requester_ip = client_ip.as_str(),
            "at capacity; shedding request"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            Json(serde_json::json!({ "error": "server at capacity, retry shortly" })),
        )
            .into_response();
    };

    // --- run the CPU-bound solve under a concurrency permit + wall-clock deadline ---
    let Ok(running) = state.solve_permits.clone().acquire_owned().await else {
        state.metrics.record_error();
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "solve semaphore closed" })),
        )
            .into_response();
    };

    let solve_state = state.clone();
    let task_input = validated.clone();
    let started = Instant::now();
    // Move the running permit into the task so it is held until the solve actually
    // finishes -- even if a timeout stops us awaiting it -- keeping concurrency bounded.
    let join = tokio::task::spawn_blocking(move || {
        let _running = running;
        let solver = PlateSolver::new(solve_state.index.clone(), task_input.config.clone());
        let refinement = task_input.sip_order.map(|order| RefinementConfig {
            sip_order: Some(order),
            ..RefinementConfig::default()
        });
        solver.solve_with_refinement_against(
            &task_input.field,
            refinement,
            Some(solve_state.catalog.as_ref()),
        )
    });
    let timed = tokio::time::timeout(state.solve_timeout, join).await;
    let solve_ms = started.elapsed().as_millis();

    // Deadline exceeded: the orphaned task finishes within its bounded budget and
    // releases its permit; we reply now so the client is not left hanging.
    let Ok(joined) = timed else {
        state.metrics.record_timed_out(solve_ms as u64);
        tracing::warn!(
            outcome = "timeout",
            solve_ms = solve_ms as u64,
            n_stars,
            scale,
            position_given,
            requester_ip = client_ip.as_str(),
            "solve exceeded deadline"
        );
        if let Some(log) = &state.failure_log {
            log.record(&validated.failure_record(
                "timeout",
                "solve exceeded deadline",
                &client_ip,
                requester.user_agent.as_ref(),
            ));
        }
        return (
            StatusCode::GATEWAY_TIMEOUT,
            Json(serde_json::json!({ "solved": false, "reason": "solve exceeded deadline" })),
        )
            .into_response();
    };

    match joined {
        // Solved.
        Ok(Ok(result)) => {
            state.metrics.record_solved(solve_ms as u64);
            tracing::info!(
                outcome = "solved",
                solve_ms = solve_ms as u64,
                n_stars,
                scale,
                position_given,
                requester_ip = client_ip.as_str(),
                log_odds = result.verification.log_odds,
                num_matches = result.verification.num_matches,
                "solved"
            );
            Json(result).into_response()
        }
        // Ran but found no solution -- a tracked failure, not an HTTP error.
        Ok(Err(e)) => {
            let reason = e.to_string();
            state.metrics.record_no_solution(solve_ms as u64);
            tracing::warn!(
                outcome = "no_solution",
                solve_ms = solve_ms as u64,
                n_stars,
                scale,
                position_given,
                requester_ip = client_ip.as_str(),
                reason = %reason,
                "no solution"
            );
            if let Some(log) = &state.failure_log {
                log.record(&validated.failure_record(
                    "no_solution",
                    &reason,
                    &client_ip,
                    requester.user_agent.as_ref(),
                ));
            }
            (
                StatusCode::OK,
                Json(serde_json::json!({ "solved": false, "reason": reason })),
            )
                .into_response()
        }
        // The solve task panicked.
        Err(join_err) => {
            let reason = format!("solve task failed: {join_err}");
            state.metrics.record_error();
            tracing::error!(
                outcome = "error",
                requester_ip = client_ip.as_str(),
                reason = %reason,
                "solve task panicked"
            );
            if let Some(log) = &state.failure_log {
                log.record(&validated.failure_record(
                    "error",
                    &reason,
                    &client_ip,
                    requester.user_agent.as_ref(),
                ));
            }
            // The panic detail stays in the server log / failure log; clients
            // get a generic message (panic strings can leak internals).
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error during solve" })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{healthz, Metrics};

    #[tokio::test]
    async fn healthz_returns_ok() {
        assert_eq!(healthz().await, "ok");
    }

    #[test]
    fn metrics_render_counts() {
        let m = Metrics::default();
        m.record_solved(10);
        m.record_solved(20);
        m.record_solved(30);
        m.record_no_solution(5);
        let out = m.render_prometheus();
        assert!(out.contains("platers_solves_total{outcome=\"solved\"} 3"));
        assert!(out.contains("platers_solves_total{outcome=\"no_solution\"} 1"));
        assert!(out.contains("platers_solve_attempts_total 4"));
    }

    #[test]
    fn cors_layer_toggles() {
        assert!(super::cors_layer(&[]).is_none(), "empty -> off");
        assert!(super::cors_layer(&["*".to_owned()]).is_some(), "* -> on");
        assert!(
            super::cors_layer(&["https://app.example.com".to_owned()]).is_some(),
            "specific origin -> on"
        );
    }
}
