//! Exponential metrics use complete UTC daily end-equity returns, independently
//! of the legacy minimum-equity and nested-window analysis definitions.
use serde::Serialize;

const DAY_MS: u64 = 86_400_000;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EwMetrics {
    pub adg: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub omega_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EwMetadata {
    pub half_life_days: f64,
    pub definition_version: u32,
    pub observation_count: usize,
    pub effective_sample_size: f64,
    pub first_return_end_ms: Option<u64>,
    pub last_return_end_ms: Option<u64>,
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct EwAnalysis {
    pub metrics: Option<EwMetrics>,
    pub metadata: EwMetadata,
}

pub fn analyze_ew_equity(
    equity: &[f64],
    timestamps_ms: &[u64],
    half_life_days: f64,
    candle_interval_minutes: u64,
) -> Result<EwAnalysis, String> {
    if !half_life_days.is_finite() || half_life_days < 1.0 {
        return Err("ew_half_life_days must be finite and at least 1 day".into());
    }
    let interval = candle_interval_minutes
        .checked_mul(60_000)
        .filter(|&v| v > 0 && DAY_MS % v == 0)
        .ok_or("EW candle interval must divide a UTC day")?;
    if equity.len() != timestamps_ms.len() {
        return Err("EW equity and timestamps must have equal lengths".into());
    }
    if equity.iter().any(|v| !v.is_finite()) {
        return Err("EW equity contains a nonfinite sample".into());
    }
    if timestamps_ms.windows(2).any(|t| t[1] <= t[0]) {
        return Err("EW timestamps must increase strictly".into());
    }
    let mut result = EwAnalysis {
        metrics: None,
        metadata: EwMetadata {
            half_life_days,
            definition_version: 1,
            observation_count: 0,
            effective_sample_size: 0.0,
            first_return_end_ms: None,
            last_return_end_ms: None,
            status: "insufficient_observations",
        },
    };
    // A ruined curve remains unavailable even if ruin falls in an excluded
    // partial day. Never remove a loss observation to make log growth usable.
    if equity.iter().any(|&v| v <= 0.0) {
        result.metadata.status = "nonpositive_equity";
        return Ok(result);
    }
    if timestamps_ms.iter().any(|t| t % interval != 0)
        || timestamps_ms.windows(2).any(|t| t[1] - t[0] != interval)
    {
        result.metadata.status = "missing_samples";
        return Ok(result);
    }
    let mut returns = Vec::new();
    let mut ends = Vec::new();
    let samples_per_day = (DAY_MS / interval) as usize;
    for (i, &ts) in timestamps_ms.iter().enumerate() {
        // The sample clock is the candle's open; its equity is evaluated at
        // that candle's close. Prior-day final equity is the required boundary.
        if ts % DAY_MS == DAY_MS - interval && i >= samples_per_day {
            let r = (equity[i] - equity[i - samples_per_day]) / equity[i - samples_per_day];
            if !r.is_finite() || r <= -1.0 {
                return Err("EW daily return is not finite or representable".into());
            }
            returns.push(r);
            ends.push(ts.checked_add(interval).ok_or("EW timestamp overflow")?);
        }
    }
    result.metadata.observation_count = returns.len();
    result.metadata.first_return_end_ms = ends.first().copied();
    result.metadata.last_return_end_ms = ends.last().copied();
    if returns.is_empty() {
        return Ok(result);
    }
    // Subtract the newest valid return clock before exponentiating. The common
    // evaluation-end factor cancels in normalization, avoiding underflow.
    let newest = *ends.last().unwrap();
    let weights: Vec<f64> = ends
        .iter()
        .map(|t| 2.0_f64.powf(-((newest - t) as f64 / DAY_MS as f64) / half_life_days))
        .collect();
    let sum_w: f64 = weights.iter().sum();
    result.metadata.effective_sample_size =
        sum_w * sum_w / weights.iter().map(|w| w * w).sum::<f64>();
    if returns.len() < 2 {
        return Ok(result);
    }
    let mean = returns
        .iter()
        .zip(&weights)
        .map(|(r, w)| r * w)
        .sum::<f64>()
        / sum_w;
    let variance = returns
        .iter()
        .zip(&weights)
        .map(|(r, w)| w * (r - mean).powi(2))
        .sum::<f64>()
        / sum_w;
    let downside = returns
        .iter()
        .zip(&weights)
        .map(|(r, w)| w * r.min(0.0).powi(2))
        .sum::<f64>()
        / sum_w;
    let gains = returns
        .iter()
        .zip(&weights)
        .map(|(r, w)| w * r.max(0.0))
        .sum::<f64>()
        / sum_w;
    let losses = returns
        .iter()
        .zip(&weights)
        .map(|(r, w)| w * (-r).max(0.0))
        .sum::<f64>()
        / sum_w;
    let log_mean = returns
        .iter()
        .zip(&weights)
        .map(|(r, w)| w * r.ln_1p())
        .sum::<f64>()
        / sum_w;
    let metrics = EwMetrics {
        adg: log_mean.exp_m1(),
        sharpe_ratio: (mean / variance.sqrt().max(1e-12)).clamp(-1000.0, 1000.0),
        sortino_ratio: (mean / downside.sqrt().max(1e-12)).clamp(-1000.0, 1000.0),
        omega_ratio: (gains / losses.max(1e-12)).clamp(0.0, 1000.0),
    };
    if [
        metrics.adg,
        metrics.sharpe_ratio,
        metrics.sortino_ratio,
        metrics.omega_ratio,
    ]
    .iter()
    .any(|v| !v.is_finite())
    {
        return Err("EW weighted moments are not finite".into());
    }
    result.metrics = Some(metrics);
    result.metadata.status = "available";
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn curve(daily: &[f64]) -> (Vec<f64>, Vec<u64>) {
        // Starts at a prior-day final candle and includes all following days.
        let mut eq = Vec::new();
        let mut ts = Vec::new();
        for i in 0..=(daily.len() - 1) * 1440 {
            eq.push(daily[i / 1440]);
            ts.push(DAY_MS - 60_000 + i as u64 * 60_000);
        }
        (eq, ts)
    }
    #[test]
    fn hand_computed_growth_and_downside_use_all_weights() {
        let (e, t) = curve(&[100.0, 110.0, 99.0]);
        let a = analyze_ew_equity(&e, &t, 1.0, 1).unwrap();
        let m = a.metrics.unwrap();
        let mean = (-0.1 + 0.5 * 0.1) / 1.5;
        let std = ((0.5 * (0.1_f64 - mean).powi(2) + (-0.1_f64 - mean).powi(2)) / 1.5).sqrt();
        assert!((m.adg - ((0.5 * 1.1_f64.ln() + 0.9_f64.ln()) / 1.5).exp_m1()).abs() < 1e-12);
        assert!((m.sharpe_ratio - mean / std).abs() < 1e-12);
        assert!((m.sortino_ratio - mean / (0.01_f64 / 1.5).sqrt()).abs() < 1e-12);
        assert!((m.omega_ratio - 0.5).abs() < 1e-12);
        assert!((a.metadata.effective_sample_size - 1.8).abs() < 1e-12);
    }
    #[test]
    fn flat_constant_and_partial_days() {
        for daily in [&[100.0, 100.0, 100.0][..], &[100.0, 110.0, 121.0][..]] {
            let (e, t) = curve(daily);
            let m = analyze_ew_equity(&e, &t, 180.5, 1)
                .unwrap()
                .metrics
                .unwrap();
            if daily[1] == 100.0 {
                assert_eq!(m.adg, 0.0);
                assert_eq!(m.sharpe_ratio, 0.0);
                assert_eq!(m.omega_ratio, 0.0);
            } else {
                assert!((m.adg - 0.1).abs() < 1e-12);
                assert_eq!(m.sharpe_ratio, 1000.0);
                assert_eq!(m.sortino_ratio, 1000.0);
                assert_eq!(m.omega_ratio, 1000.0);
            }
        }
        let (e, t) = curve(&[100.0, 110.0, 121.0, 133.1]);
        let a = analyze_ew_equity(&e[1..e.len() - 1], &t[1..t.len() - 1], 2.0, 1).unwrap();
        assert_eq!(a.metadata.observation_count, 1);
        assert!(a.metrics.is_none());
    }
    #[test]
    fn malformed_ruined_and_gap_inputs() {
        let (e, t) = curve(&[100.0, 110.0, 121.0]);
        for h in [0.0, 0.9, f64::NAN, f64::INFINITY] {
            assert!(analyze_ew_equity(&e, &t, h, 1).is_err());
        }
        let mut bad = e.clone();
        bad[4] = f64::NAN;
        assert!(analyze_ew_equity(&bad, &t, 1.0, 1).is_err());
        bad[4] = 0.0;
        assert_eq!(
            analyze_ew_equity(&bad, &t, 1.0, 1).unwrap().metadata.status,
            "nonpositive_equity"
        );
        let mut gap = t.clone();
        gap[4] += 60_000;
        assert!(analyze_ew_equity(&e, &gap, 1.0, 1).is_err());
        let mut e = e;
        let mut t = t;
        e.remove(4);
        t.remove(4);
        assert_eq!(
            analyze_ew_equity(&e, &t, 1.0, 1).unwrap().metadata.status,
            "missing_samples"
        );
    }
    #[test]
    fn five_minute_cadence_and_sparse_data() {
        let (e, t) = curve(&[100.0, 110.0, 121.0]);
        // Re-clock final candles to the supported five-minute close boundary.
        let e: Vec<_> = e.iter().step_by(5).copied().collect();
        let t: Vec<_> = t.iter().step_by(5).map(|v| v - 240_000).collect();
        assert_eq!(
            analyze_ew_equity(&e, &t, 180.0, 5)
                .unwrap()
                .metadata
                .observation_count,
            2
        );
        assert_eq!(
            analyze_ew_equity(&[], &[], 180.0, 1)
                .unwrap()
                .metadata
                .status,
            "insufficient_observations"
        );
    }
}
