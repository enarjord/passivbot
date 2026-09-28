//! Signed RMS directionality from completed one-minute closes.
//! A deterministic 20-span window gives live, restart and CPU the same inputs.
//! Weights are exponential; this is not a simple moving average.
pub fn warmup_returns(span: f64) -> Result<usize, String> {
    if !span.is_finite() || !(1.0..=100_000.0).contains(&span) {
        return Err("unilateralness span must be finite and between 1 and 100000".into());
    }
    Ok((span * 20.0).ceil() as usize)
}

pub fn signed_rms(closes: &[f64], span: f64) -> Result<f64, String> {
    let n = warmup_returns(span)?;
    if closes.len() < n + 1 {
        return Err("incomplete unilateralness warmup".into());
    }
    let closes = &closes[closes.len() - n - 1..];
    let alpha = 2.0 / (span + 1.0);
    let mut mean = 0.0;
    let mut square = 0.0;
    for pair in closes.windows(2) {
        if pair.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("invalid unilateralness close".into());
        }
        let r = pair[1].ln() - pair[0].ln();
        mean += alpha * (r - mean);
        square += alpha * (r * r - square);
    }
    Ok(if square == 0.0 {
        0.0
    } else {
        (mean / square.sqrt()).clamp(-1.0, 1.0)
    })
}

/// A finite exponentially weighted window represented by two aggregate stacks.
/// Each return is pushed once and transferred at most once: amortized O(1)
/// work per close, O(window) memory, and no subtraction of nearly equal moments.
/// Grouping changes floating-point rounding, but never the retained return set.
pub(crate) struct RollingRms {
    window: usize,
    alpha: f64,
    decay: f64,
    previous_log: Option<f64>,
    front: Vec<WeightedReturn>,
    back: Vec<WeightedReturn>,
    back_decay: f64,
    #[cfg(test)]
    aggregations: usize,
}

#[derive(Clone, Copy, Default)]
struct WeightedReturn {
    value: f64,
    mean: f64,
    square: f64,
}

impl RollingRms {
    pub(crate) fn new(span: f64) -> Result<Self, String> {
        let alpha = 2.0 / (span + 1.0);
        Ok(Self {
            window: warmup_returns(span)?,
            alpha,
            decay: 1.0 - alpha,
            previous_log: None,
            front: Vec::new(),
            back: Vec::new(),
            back_decay: 1.0,
            #[cfg(test)]
            aggregations: 0,
        })
    }

    pub(crate) fn push_close(&mut self, close: f64) -> Result<(), String> {
        if !close.is_finite() || close <= 0.0 {
            return Err("invalid unilateralness close".into());
        }
        let log_close = close.ln();
        let previous = self.previous_log.replace(log_close);
        let Some(previous) = previous else {
            return Ok(());
        };
        let value = log_close - previous;
        if self.front.len() + self.back.len() == self.window {
            if self.front.is_empty() {
                let mut aggregate = WeightedReturn::default();
                let mut power = 1.0;
                while let Some(item) = self.back.pop() {
                    aggregate.value = item.value;
                    aggregate.mean += self.alpha * item.value * power;
                    aggregate.square += self.alpha * item.value * item.value * power;
                    self.front.push(aggregate);
                    power *= self.decay;
                    #[cfg(test)]
                    {
                        self.aggregations += 1;
                    }
                }
                self.back_decay = 1.0;
            }
            self.front.pop();
        }
        let mut aggregate = self.back.last().copied().unwrap_or_default();
        aggregate.value = value;
        aggregate.mean += self.alpha * (value - aggregate.mean);
        aggregate.square += self.alpha * (value * value - aggregate.square);
        self.back.push(aggregate);
        self.back_decay *= self.decay;
        #[cfg(test)]
        {
            self.aggregations += 1;
        }
        Ok(())
    }

    pub(crate) fn score(&self) -> Option<f64> {
        if self.front.len() + self.back.len() != self.window {
            return None;
        }
        let front = self.front.last().copied().unwrap_or_default();
        let back = self.back.last().copied().unwrap_or_default();
        let mean = front.mean * self.back_decay + back.mean;
        let square = front.square * self.back_decay + back.square;
        Some(if square == 0.0 {
            0.0
        } else {
            (mean / square.sqrt()).clamp(-1.0, 1.0)
        })
    }
}

#[pyo3::prelude::pyfunction]
pub fn calc_signed_unilateralness(closes: Vec<f64>, span: f64) -> pyo3::PyResult<f64> {
    signed_rms(&closes, span).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direction_flat_decay_and_fractional_span() {
        let span = 10.5;
        let n = warmup_returns(span).unwrap();
        let up: Vec<f64> = (0..=n).map(|i| (i as f64 * 0.001).exp()).collect();
        let down: Vec<f64> = up.iter().map(|v| 1.0 / v).collect();
        let u = signed_rms(&up, span).unwrap();
        assert!((u - 1.0).abs() < 1e-12);
        assert!((signed_rms(&down, span).unwrap() + u).abs() < 1e-12);
        let mut flat = up.clone();
        flat.extend(vec![*up.last().unwrap(); 20]);
        assert!(
            (signed_rms(&flat, span).unwrap() - u * (1.0 - 2.0 / (span + 1.0)).powf(10.0)).abs()
                < 1e-12
        );
        assert_eq!(signed_rms(&vec![100.0; n + 1], span).unwrap(), 0.0);
        assert!(signed_rms(&up[..n], span).is_err());
    }

    #[test]
    fn rolling_matches_replay_across_transfers_and_flat_tails() {
        for span in [1.0, 1.01, 10.5, 60.0, 240.5] {
            let n = warmup_returns(span).unwrap();
            let mut rolling = RollingRms::new(span).unwrap();
            let mut closes = Vec::new();
            let mut log_price = 0.0;
            for k in 0..3 * n + 100 {
                if k < n + 100 {
                    log_price += 0.003 * (k as f64 * 0.7).sin() + 0.0001;
                }
                closes.push(log_price.exp());
                rolling.push_close(*closes.last().unwrap()).unwrap();
                if k < n {
                    assert!(rolling.score().is_none());
                } else if k % 13 == 0 || k % n <= 1 {
                    let replay = signed_rms(&closes, span).unwrap();
                    assert!(
                        (rolling.score().unwrap() - replay).abs() < 2e-12,
                        "span={span} candle={k} rolling={:?} replay={replay}",
                        rolling.score()
                    );
                }
                assert!(rolling.front.len() + rolling.back.len() <= n);
            }
            assert_eq!(rolling.score(), Some(0.0));
            // Each return is aggregated on arrival and transferred at most once.
            assert!(rolling.aggregations <= 2 * (closes.len() - 1));
        }
    }

    #[test]
    fn rolling_drops_large_shock_without_cancellation_residue() {
        let span = 10.5;
        let n = warmup_returns(span).unwrap();
        let mut rolling = RollingRms::new(span).unwrap();
        let mut closes = vec![1e-300];
        rolling.push_close(closes[0]).unwrap();
        for k in 1..=3 * n {
            closes.push(1.0 + (k % 2) as f64 * f64::EPSILON);
            rolling.push_close(*closes.last().unwrap()).unwrap();
            if k >= n {
                assert!(
                    (rolling.score().unwrap() - signed_rms(&closes, span).unwrap()).abs() < 2e-12
                );
            }
        }
    }

    #[test]
    fn rolling_rejects_bad_closes_without_corrupting_history() {
        let mut rolling = RollingRms::new(1.0).unwrap();
        rolling.push_close(100.0).unwrap();
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(rolling.push_close(invalid).is_err());
        }
        for _ in 0..20 {
            rolling.push_close(100.0).unwrap();
        }
        assert_eq!(rolling.score(), Some(0.0));
        assert!(RollingRms::new(0.0).is_err());
    }

    #[test]
    #[ignore = "manual CPU benchmark; use --release and --nocapture"]
    fn benchmark_rolling_against_replay() {
        use std::hint::black_box;
        use std::time::Instant;
        let span = 60.0;
        let n = warmup_returns(span).unwrap();
        let bars = 20_000;
        let closes: Vec<f64> = (0..bars)
            .map(|k| (0.02 * (k as f64 * 0.017).sin() + 0.00001 * k as f64).exp())
            .collect();
        let start = Instant::now();
        let reference: Vec<f64> = (n..bars)
            .map(|k| black_box(signed_rms(&closes[k - n..=k], span).unwrap()))
            .collect();
        let replay_time = start.elapsed();
        let start = Instant::now();
        let mut rolling = RollingRms::new(span).unwrap();
        let actual: Vec<f64> = closes
            .iter()
            .filter_map(|close| {
                rolling.push_close(*close).unwrap();
                black_box(rolling.score())
            })
            .collect();
        let rolling_time = start.elapsed();
        assert_eq!(reference.len(), actual.len());
        let error = reference
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(error < 2e-12, "maximum score error {error}");
        println!("os={} arch={} debug={} bars={bars} span={span} replay={replay_time:?} rolling={rolling_time:?} max_error={error}",
            std::env::consts::OS, std::env::consts::ARCH, cfg!(debug_assertions));
    }
}
