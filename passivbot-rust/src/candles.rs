use crate::constants::{CLOSE, HIGH, LOW, VOLUME};
use ndarray::ArrayView3;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Bucket {
    pub ts: i64,
    pub h: f32,
    pub l: f32,
    pub c: f32,
    pub v: f32, // base volume (assumed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Metric {
    Close,
    Volume,
    Nrr,
}

#[derive(Default)]
struct AggCache {
    pub tf_min: u32,
    pub last_bucket_idx: i64,
    pub buckets: Vec<Bucket>,
}

#[derive(Default)]
struct EmaState {
    pub span: u32,
    pub last_bucket_idx: i64,
    pub last_ema: f64,
}

/// BacktestCandleManager: lazy tf aggregation + EMA caching for backtester
///
/// This is a scaffold. It provides the public interface and basic structures
/// but defers full implementation to future commits. Current methods either
/// return placeholders or no-op while keeping signatures stable.
pub struct BacktestCandleManager {
    /// reference timeline start (first 1m candle ts). If unknown, keep 0.
    start_ts_ms: i64,
    /// per (coin_idx, tf) aggregation cache
    agg: HashMap<(usize, u32), AggCache>,
    /// per (coin_idx, tf, metric, span) EMA state
    ema: HashMap<(usize, u32, Metric, u32), EmaState>,
    /// number of coins tracked (index validity checks)
    n_coins: usize,
}

impl BacktestCandleManager {
    pub fn new_empty() -> Self {
        BacktestCandleManager {
            start_ts_ms: 0,
            agg: HashMap::new(),
            ema: HashMap::new(),
            n_coins: 0,
        }
    }

    /// Optionally provide initial context like number of coins and first ts.
    pub fn with_context(n_coins: usize, start_ts_ms: i64) -> Self {
        BacktestCandleManager {
            start_ts_ms,
            agg: HashMap::new(),
            ema: HashMap::new(),
            n_coins,
        }
    }

    /// Return the close of candle at 1m index k for coin idx.
    /// Placeholder: returns NaN until wired to underlying data.
    pub fn get_last_close(&mut self, _k: usize, _idx: usize) -> f32 {
        f32::NAN
    }

    /// Return the high of candle at 1m index k for coin idx.
    pub fn get_last_high(&mut self, _k: usize, _idx: usize) -> f32 {
        f32::NAN
    }

    /// Return the low of candle at 1m index k for coin idx.
    pub fn get_last_low(&mut self, _k: usize, _idx: usize) -> f32 {
        f32::NAN
    }

    /// Return last EMA(close) at 1m step k for coin idx, span, timeframe (minutes).
    /// Placeholder: returns NaN; caching to be implemented.
    pub fn get_last_ema_close(
        &mut self,
        k: usize,
        idx: usize,
        span: u32,
        tf_min: u32,
        hlcvs: &ArrayView3<'_, f64>,
    ) -> f64 {
        self.ensure_ema_up_to(k, idx, span, tf_min, hlcvs, Metric::Close)
    }

    /// Return last EMA(volume) at 1m step k for coin idx, span, timeframe (minutes).
    pub fn get_last_ema_volume(
        &mut self,
        k: usize,
        idx: usize,
        span: u32,
        tf_min: u32,
        hlcvs: &ArrayView3<'_, f64>,
    ) -> f64 {
        self.ensure_ema_up_to(k, idx, span, tf_min, hlcvs, Metric::Volume)
    }

    /// Return last EMA(NRR=(high-low)/max(close, 1e-12)) at 1m step k for coin idx, span, tf.
    pub fn get_last_ema_nrr(
        &mut self,
        k: usize,
        idx: usize,
        span: u32,
        tf_min: u32,
        hlcvs: &ArrayView3<'_, f64>,
    ) -> f64 {
        self.ensure_ema_up_to(k, idx, span, tf_min, hlcvs, Metric::Nrr)
    }

    fn ensure_ema_up_to(
        &mut self,
        k: usize,
        idx: usize,
        span: u32,
        tf_min: u32,
        hlcvs: &ArrayView3<'_, f64>,
        metric: Metric,
    ) -> f64 {
        let key = (idx, tf_min, metric, span);
        let state = self.ema.entry(key).or_insert(EmaState {
            span,
            last_bucket_idx: -1,
            last_ema: f64::NAN,
        });

        let tf = tf_min.max(1) as usize;
        // Determine last completed bucket at step k
        let end_bucket: isize = if tf == 1 {
            k as isize
        } else {
            (k as isize / tf as isize) - 1
        };
        if end_bucket < 0 {
            // no completed buckets yet
            return f64::NAN;
        }

        // If already up-to-date
        if state.last_bucket_idx >= end_bucket as i64 && state.last_ema.is_finite() {
            return state.last_ema;
        }

        // Compute starting bucket and seed EMA
        let mut b_start = 0isize;
        let span_i = span as isize;
        let b_end = end_bucket;
        if b_end - span_i + 1 > 0 {
            b_start = b_end - span_i + 1;
        }

        // If we have previous state behind b_start, continue from there
        let mut ema: f64;
        let mut b_iter_start = b_start;
        if state.last_bucket_idx >= 0 && state.last_bucket_idx >= b_start as i64 {
            // continue from stored state
            ema = state.last_ema;
            b_iter_start = state.last_bucket_idx as isize + 1;
        } else {
            // seed from first available bucket value
            let x0 = Self::bucket_metric(idx, b_start as usize, tf, hlcvs, &metric);
            ema = x0;
            b_iter_start = b_start + 1;
        }

        // EMA recurrence
        let alpha = 2.0 / (span as f64 + 1.0);
        let one_minus = 1.0 - alpha;
        for b in b_iter_start..=b_end {
            let x = Self::bucket_metric(idx, b as usize, tf, hlcvs, &metric);
            ema = one_minus * ema + alpha * x;
        }

        state.last_bucket_idx = b_end as i64;
        state.last_ema = ema;
        ema
    }

    fn bucket_metric(
        idx: usize,
        b: usize,
        tf: usize,
        hlcvs: &ArrayView3<'_, f64>,
        metric: &Metric,
    ) -> f64 {
        let start = b * tf;
        let end = start + tf; // exclusive
        let n_ts = hlcvs.shape()[0];
        let last = (end - 1).min(n_ts - 1);

        match metric {
            Metric::Close => hlcvs[[last, idx, CLOSE]],
            Metric::Volume => {
                let mut sum = 0.0f64;
                let stop = end.min(n_ts);
                for k in start..stop {
                    sum += hlcvs[[k, idx, VOLUME]].max(0.0);
                }
                sum
            }
            Metric::Nrr => {
                let mut h = f64::NEG_INFINITY;
                let mut l = f64::INFINITY;
                let stop = end.min(n_ts);
                for k in start..stop {
                    let hk = hlcvs[[k, idx, HIGH]];
                    let lk = hlcvs[[k, idx, LOW]];
                    if hk > h {
                        h = hk;
                    }
                    if lk < l {
                        l = lk;
                    }
                }
                let c = hlcvs[[last, idx, CLOSE]];
                let denom = if c.abs() < 1e-12 { 1e-12 } else { c.abs() };
                (h - l) / denom
            }
        }
    }
}
