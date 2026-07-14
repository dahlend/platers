//! Performance and validation metrics collection.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Simple metrics for a single solve attempt (for parametric testing)
#[derive(Debug, Clone)]
pub struct SolveMetrics {
    /// Whether the solve succeeded
    pub solved: bool,
    /// Total solve time in milliseconds
    pub solve_time_ms: u64,
    /// Position error in arcseconds
    pub position_error_arcsec: f64,
    /// Scale error as percentage
    pub scale_error_percent: f64,
    /// Rotation error in degrees
    pub rotation_error_deg: f64,
    /// Number of hypotheses tested
    pub num_hypotheses: usize,
    /// Number of matched stars
    pub num_matches: usize,
}

/// Aggregate statistics across multiple solves (for parametric testing)
#[derive(Debug, Clone)]
pub struct ParametricMetrics {
    /// Total number of tests run
    pub num_tests: usize,
    /// Number of successful solves
    pub num_solved: usize,
    /// Number of failed solves
    pub num_failed: usize,
    /// Solve rate as a percentage
    pub solve_rate_percent: f64,

    /// Mean solve time (ms)
    pub mean_solve_time_ms: f64,
    /// Median solve time (ms)
    pub median_solve_time_ms: f64,
    /// 95th-percentile solve time (ms)
    pub solve_time_p95: f64,

    /// Median position error (arcsec) - for solved cases only
    pub position_error_p50: f64,
    /// 95th-percentile position error (arcsec) - for solved cases only
    pub position_error_p95: f64,

    /// Median scale error (percent) - for solved cases only
    pub scale_error_p50: f64,
    /// 95th-percentile scale error (percent) - for solved cases only
    pub scale_error_p95: f64,

    /// Median rotation error (degrees) - for solved cases only
    pub rotation_error_p50: f64,
    /// 95th-percentile rotation error (degrees) - for solved cases only
    pub rotation_error_p95: f64,

    /// Mean number of hypotheses tested (solved cases only)
    pub mean_hypotheses: f64,
}

impl ParametricMetrics {
    /// Compute aggregate statistics from a collection of solve metrics
    ///
    /// # Panics
    /// Panics if any recorded metric is NaN (float comparisons are unwrapped).
    #[must_use]
    pub fn from_results(results: &[SolveMetrics]) -> Self {
        let num_tests = results.len();
        let num_solved = results.iter().filter(|r| r.solved).count();
        let num_failed = num_tests - num_solved;
        let solve_rate_percent = if num_tests > 0 {
            (num_solved as f64 / num_tests as f64) * 100.0
        } else {
            0.0
        };

        // Timing statistics (all tests)
        let mut solve_times: Vec<f64> = results.iter().map(|r| r.solve_time_ms as f64).collect();
        solve_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_solve_time_ms = if solve_times.is_empty() {
            0.0
        } else {
            solve_times.iter().sum::<f64>() / solve_times.len() as f64
        };
        let median_solve_time_ms = percentile(&solve_times, 50.0);
        let solve_time_p95 = percentile(&solve_times, 95.0);

        // Accuracy statistics (solved cases only)
        let solved_results: Vec<_> = results.iter().filter(|r| r.solved).collect();

        let mut position_errors: Vec<f64> = solved_results
            .iter()
            .map(|r| r.position_error_arcsec)
            .collect();
        position_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut scale_errors: Vec<f64> = solved_results
            .iter()
            .map(|r| r.scale_error_percent)
            .collect();
        scale_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut rotation_errors: Vec<f64> = solved_results
            .iter()
            .map(|r| r.rotation_error_deg)
            .collect();
        rotation_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let position_error_p50 = percentile(&position_errors, 50.0);
        let position_error_p95 = percentile(&position_errors, 95.0);
        let scale_error_p50 = percentile(&scale_errors, 50.0);
        let scale_error_p95 = percentile(&scale_errors, 95.0);
        let rotation_error_p50 = percentile(&rotation_errors, 50.0);
        let rotation_error_p95 = percentile(&rotation_errors, 95.0);

        let mean_hypotheses = if solved_results.is_empty() {
            0.0
        } else {
            solved_results
                .iter()
                .map(|r| r.num_hypotheses as f64)
                .sum::<f64>()
                / solved_results.len() as f64
        };

        Self {
            num_tests,
            num_solved,
            num_failed,
            solve_rate_percent,
            mean_solve_time_ms,
            median_solve_time_ms,
            solve_time_p95,
            position_error_p50,
            position_error_p95,
            scale_error_p50,
            scale_error_p95,
            rotation_error_p50,
            rotation_error_p95,
            mean_hypotheses,
        }
    }
}

/// Timing metrics for a solve operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Total solve time
    pub total_duration: Duration,
    /// Time spent generating image quads
    pub quad_generation_duration: Option<Duration>,
    /// Time spent searching indices
    pub index_search_duration: Option<Duration>,
    /// Time spent on verification
    pub verification_duration: Option<Duration>,
}

impl TimingMetrics {
    /// Create new timing metrics with just total duration.
    #[must_use]
    pub fn new(total_duration: Duration) -> Self {
        Self {
            total_duration,
            quad_generation_duration: None,
            index_search_duration: None,
            verification_duration: None,
        }
    }

    /// Get total time in seconds.
    #[must_use]
    pub fn total_secs(&self) -> f64 {
        self.total_duration.as_secs_f64()
    }

    /// Get total time in milliseconds.
    #[must_use]
    pub fn total_millis(&self) -> u128 {
        self.total_duration.as_millis()
    }
}

/// Accuracy metrics for a solve result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Position error in arcseconds (RMS)
    pub position_error_arcsec: f64,
    /// Scale error as percentage
    pub scale_error_percent: f64,
    /// Rotation error in degrees
    pub rotation_error_deg: f64,
    /// Number of matched stars
    pub num_matches: usize,
    /// Match rate (fraction of stars matched)
    pub match_rate: f64,
}

impl AccuracyMetrics {
    /// Check if metrics pass given thresholds.
    #[must_use]
    pub fn passes(&self, max_pos_error: f64, max_scale_error: f64, max_rot_error: f64) -> bool {
        self.position_error_arcsec < max_pos_error
            && self.scale_error_percent < max_scale_error
            && self.rotation_error_deg < max_rot_error
    }
}

/// Complete metrics for a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseMetrics {
    /// Test case identifier
    pub case_id: String,
    /// Whether the solve succeeded
    pub success: bool,
    /// Timing metrics
    pub timing: TimingMetrics,
    /// Accuracy metrics (if succeeded)
    pub accuracy: Option<AccuracyMetrics>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Statistics aggregated across multiple test cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Total number of test cases
    pub total_cases: usize,
    /// Number of successful solves
    pub num_success: usize,
    /// Number of failed solves
    pub num_failures: usize,
    /// Success rate
    pub success_rate: f64,

    /// Median solve time in seconds (for successful cases)
    pub median_time_secs: f64,
    /// Mean solve time in seconds (for successful cases)
    pub mean_time_secs: f64,
    /// 95th-percentile solve time in seconds (for successful cases)
    pub p95_time_secs: f64,
    /// Minimum solve time in seconds (for successful cases)
    pub min_time_secs: f64,
    /// Maximum solve time in seconds (for successful cases)
    pub max_time_secs: f64,

    /// Median position error in arcseconds (for successful cases)
    pub median_pos_error_arcsec: f64,
    /// Mean position error in arcseconds (for successful cases)
    pub mean_pos_error_arcsec: f64,
    /// RMS position error in arcseconds (for successful cases)
    pub rms_pos_error_arcsec: f64,
    /// 95th-percentile position error in arcseconds (for successful cases)
    pub p95_pos_error_arcsec: f64,

    /// Median scale error as percentage (for successful cases)
    pub median_scale_error_percent: f64,
    /// Mean scale error as percentage (for successful cases)
    pub mean_scale_error_percent: f64,
    /// 95th-percentile scale error as percentage (for successful cases)
    pub p95_scale_error_percent: f64,

    /// Median rotation error in degrees (for successful cases)
    pub median_rotation_error_deg: f64,
    /// Mean rotation error in degrees (for successful cases)
    pub mean_rotation_error_deg: f64,
    /// 95th-percentile rotation error in degrees (for successful cases)
    pub p95_rotation_error_deg: f64,
}

impl AggregateMetrics {
    /// Compute aggregate metrics from a collection of test case metrics.
    ///
    /// # Panics
    /// Panics if any recorded metric is NaN (float comparisons are unwrapped).
    #[must_use]
    pub fn from_test_cases(cases: &[TestCaseMetrics]) -> Self {
        let total_cases = cases.len();
        let num_success = cases.iter().filter(|c| c.success).count();
        let num_failures = total_cases - num_success;
        let success_rate = if total_cases > 0 {
            num_success as f64 / total_cases as f64
        } else {
            0.0
        };

        // Collect successful cases
        let successful: Vec<_> = cases.iter().filter(|c| c.success).collect();

        // Timing statistics
        let mut times: Vec<f64> = successful.iter().map(|c| c.timing.total_secs()).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (median_time, mean_time, p95_time, min_time, max_time) = if times.is_empty() {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let median = percentile(&times, 50.0);
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let p95 = percentile(&times, 95.0);
            let min = *times.first().unwrap();
            let max = *times.last().unwrap();
            (median, mean, p95, min, max)
        };

        // Accuracy statistics
        let accuracies: Vec<_> = successful
            .iter()
            .filter_map(|c| c.accuracy.as_ref())
            .collect();

        let (median_pos, mean_pos, rms_pos, p95_pos) = if accuracies.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let mut pos_errors: Vec<f64> =
                accuracies.iter().map(|a| a.position_error_arcsec).collect();
            pos_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = percentile(&pos_errors, 50.0);
            let mean = pos_errors.iter().sum::<f64>() / pos_errors.len() as f64;
            let rms =
                (pos_errors.iter().map(|x| x * x).sum::<f64>() / pos_errors.len() as f64).sqrt();
            let p95 = percentile(&pos_errors, 95.0);
            (median, mean, rms, p95)
        };

        let (median_scale, mean_scale, p95_scale) = if accuracies.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mut scale_errors: Vec<f64> =
                accuracies.iter().map(|a| a.scale_error_percent).collect();
            scale_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = percentile(&scale_errors, 50.0);
            let mean = scale_errors.iter().sum::<f64>() / scale_errors.len() as f64;
            let p95 = percentile(&scale_errors, 95.0);
            (median, mean, p95)
        };

        let (median_rot, mean_rot, p95_rot) = if accuracies.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mut rot_errors: Vec<f64> =
                accuracies.iter().map(|a| a.rotation_error_deg).collect();
            rot_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = percentile(&rot_errors, 50.0);
            let mean = rot_errors.iter().sum::<f64>() / rot_errors.len() as f64;
            let p95 = percentile(&rot_errors, 95.0);
            (median, mean, p95)
        };

        Self {
            total_cases,
            num_success,
            num_failures,
            success_rate,
            median_time_secs: median_time,
            mean_time_secs: mean_time,
            p95_time_secs: p95_time,
            min_time_secs: min_time,
            max_time_secs: max_time,
            median_pos_error_arcsec: median_pos,
            mean_pos_error_arcsec: mean_pos,
            rms_pos_error_arcsec: rms_pos,
            p95_pos_error_arcsec: p95_pos,
            median_scale_error_percent: median_scale,
            mean_scale_error_percent: mean_scale,
            p95_scale_error_percent: p95_scale,
            median_rotation_error_deg: median_rot,
            mean_rotation_error_deg: mean_rot,
            p95_rotation_error_deg: p95_rot,
        }
    }

    /// Print a formatted summary report.
    pub fn print_summary(&self) {
        println!("\n=== Test Suite Summary ===");
        println!("Total Cases: {}", self.total_cases);
        println!(
            "Successes: {} ({:.1}%)",
            self.num_success,
            self.success_rate * 100.0
        );
        println!("Failures: {}", self.num_failures);

        if self.num_success > 0 {
            println!("\n--- Timing Statistics ---");
            println!("  Median: {:.3}s", self.median_time_secs);
            println!("  Mean:   {:.3}s", self.mean_time_secs);
            println!("  95th percentile: {:.3}s", self.p95_time_secs);
            println!(
                "  Range: [{:.3}s, {:.3}s]",
                self.min_time_secs, self.max_time_secs
            );

            println!("\n--- Position Accuracy ---");
            println!("  Median: {:.3}\"", self.median_pos_error_arcsec);
            println!("  Mean:   {:.3}\"", self.mean_pos_error_arcsec);
            println!("  RMS:    {:.3}\"", self.rms_pos_error_arcsec);
            println!("  95th percentile: {:.3}\"", self.p95_pos_error_arcsec);

            println!("\n--- Scale Accuracy ---");
            println!("  Median: {:.3}%", self.median_scale_error_percent);
            println!("  Mean:   {:.3}%", self.mean_scale_error_percent);
            println!("  95th percentile: {:.3}%", self.p95_scale_error_percent);

            println!("\n--- Rotation Accuracy ---");
            println!("  Median: {:.3} deg", self.median_rotation_error_deg);
            println!("  Mean:   {:.3} deg", self.mean_rotation_error_deg);
            println!("  95th percentile: {:.3} deg", self.p95_rotation_error_deg);
        }
        println!("========================\n");
    }
}

/// Calculate percentile from sorted data.
#[allow(clippy::cast_sign_loss, reason = "percentile p is in [0, 100]")]
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[idx.min(sorted_data.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_aggregate_metrics() {
        let cases = vec![
            TestCaseMetrics {
                case_id: "1".to_string(),
                success: true,
                timing: TimingMetrics::new(Duration::from_secs(1)),
                accuracy: Some(AccuracyMetrics {
                    position_error_arcsec: 0.5,
                    scale_error_percent: 1.0,
                    rotation_error_deg: 0.1,
                    num_matches: 10,
                    match_rate: 0.9,
                }),
                error: None,
            },
            TestCaseMetrics {
                case_id: "2".to_string(),
                success: true,
                timing: TimingMetrics::new(Duration::from_secs(2)),
                accuracy: Some(AccuracyMetrics {
                    position_error_arcsec: 1.0,
                    scale_error_percent: 2.0,
                    rotation_error_deg: 0.2,
                    num_matches: 12,
                    match_rate: 0.95,
                }),
                error: None,
            },
        ];

        let metrics = AggregateMetrics::from_test_cases(&cases);
        assert_eq!(metrics.total_cases, 2);
        assert_eq!(metrics.num_success, 2);
        assert_eq!(metrics.success_rate, 1.0);
        assert!(metrics.median_time_secs > 0.0);
    }
}
