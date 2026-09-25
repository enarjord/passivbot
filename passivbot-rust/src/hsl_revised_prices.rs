//! Causal, bounded historical close projection for revised HSL only.
//! Estimated rows are ephemeral risk inputs, never a factual candle ledger.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MINUTE: i64 = 60_000;
const MAX_WINDOW: i64 = 90 * 24 * 60 * MINUTE;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candle {
    pub start: i64,
    pub minutes: i64,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub open: Option<f64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub high: Option<f64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub low: Option<f64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub close: Option<f64>,
    pub available_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub start: i64,
    pub end: i64,
    pub candles: Vec<Candle>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Price {
    pub timestamp: i64,
    pub close: f64,
    pub resolution_minutes: i64,
    pub source_end: i64,
    pub carried: bool,
}

#[derive(Debug, Serialize)]
pub struct Prices {
    pub rows: Vec<Price>,
    pub reasons: BTreeSet<String>,
}

fn positive(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v > 0.0)
}

pub fn minute_prices(input: &Input) -> Result<Prices, String> {
    minute_prices_from_candles(input.start, input.end, &input.candles, 0)
}

fn minute_prices_from_candles(
    start: i64,
    end: i64,
    candles: &[Candle],
    observation_offset: i64,
) -> Result<Prices, String> {
    let window = end
        .checked_sub(start)
        .filter(|v| (0..=MAX_WINDOW).contains(v))
        .ok_or("invalid revised HSL price interval (maximum 90 days)")?;
    let mut reasons = BTreeSet::new();
    // Collect contiguous scalar rows rather than allocating a tree for every
    // timestamp/resolution. Sorting below preserves finest-source arbitration.
    let mut candidates = Vec::with_capacity(candles.len());
    for candle in candles {
        // Match scalar transport validation even for a row later excluded by
        // the window or resolution policy; never wrap an observation clock.
        let available_at = candle
            .available_at
            .map(|at| {
                at.checked_add(observation_offset)
                    .ok_or("candle observation timestamp overflow")
            })
            .transpose()?;
        if ![1, 5, 15, 60].contains(&candle.minutes) {
            reasons.insert("unsupported_candle_resolution".into());
            continue;
        }
        let Some(finish) = candle.start.checked_add(candle.minutes * MINUTE) else {
            continue;
        };
        if finish > end
            || finish < start
            || available_at.is_some_and(|at| at > end)
            || (candle.minutes > 1 && candle.start < start)
        {
            continue;
        }
        let Some(close) = positive(candle.close) else {
            reasons.insert("unusable_historical_candle".into());
            continue;
        };
        if candle.minutes == 1 {
            candidates.push(Price {
                timestamp: finish,
                close,
                resolution_minutes: 1,
                source_end: finish,
                carried: false,
            });
            continue;
        }
        let path: Vec<f64> = {
            let (Some(open), Some(high), Some(low)) = (
                positive(candle.open),
                positive(candle.high),
                positive(candle.low),
            ) else {
                reasons.insert("unusable_historical_candle".into());
                continue;
            };
            if low > open.min(close) || open.max(close) > high {
                reasons.insert("unusable_historical_candle".into());
                continue;
            }
            let waypoints = [
                (0, open),
                (candle.minutes / 3, if close >= open { low } else { high }),
                (
                    2 * candle.minutes / 3,
                    if close >= open { high } else { low },
                ),
                (candle.minutes - 1, close),
            ];
            (0..candle.minutes)
                .map(|i| {
                    let w = waypoints
                        .windows(2)
                        .find(|w| w[0].0 <= i && i <= w[1].0)
                        .unwrap();
                    let fraction = (i - w[0].0) as f64 / (w[1].0 - w[0].0) as f64;
                    // Convex weights preserve finite, positive parent bounds.
                    ((1.0 - fraction) * w[0].1 + fraction * w[1].1)
                        .clamp(w[0].1.min(w[1].1), w[0].1.max(w[1].1))
                })
                .collect()
        };
        for (index, price) in path.into_iter().enumerate() {
            let timestamp = candle.start + (index as i64 + 1) * MINUTE;
            let row = Price {
                timestamp,
                close: price,
                resolution_minutes: candle.minutes,
                source_end: finish,
                carried: false,
            };
            candidates.push(row);
        }
    }
    candidates.sort_unstable_by_key(|row| (row.timestamp, row.resolution_minutes));
    let mut selected: Vec<Price> = Vec::new();
    for group in candidates
        .chunk_by(|a, b| a.timestamp == b.timestamp && a.resolution_minutes == b.resolution_minutes)
    {
        let first = &group[0];
        if group
            .iter()
            .any(|row| row.close != first.close || row.source_end != first.source_end)
        {
            // Any disagreement disputes the entire resolution, including later
            // duplicates. Still inspect coarser groups for their diagnostics.
            reasons.insert("candle_conflict".into());
        } else if selected
            .last()
            .is_none_or(|row| row.timestamp != first.timestamp)
        {
            selected.push(first.clone());
        }
    }
    drop(candidates);
    let mut rows = Vec::with_capacity((window / MINUTE + 1) as usize);
    if let Some(first) = selected.first() {
        let first_time = first.timestamp;
        let mut previous = first;
        let mut next = 0;
        let offset = (MINUTE - start.rem_euclid(MINUTE)) % MINUTE;
        let mut timestamp = start.checked_add(offset);
        while let Some(t) = timestamp.filter(|t| *t <= end) {
            while next < selected.len() && selected[next].timestamp < t {
                next += 1;
            }
            let observed = selected.get(next).filter(|row| row.timestamp == t);
            if let Some(observed) = observed {
                previous = observed;
            }
            let mut row = previous.clone();
            row.timestamp = t;
            row.carried = observed.is_none();
            if row.carried {
                reasons.insert(
                    if t < first_time {
                        "backfilled_price"
                    } else {
                        "forward_filled_price"
                    }
                    .into(),
                );
            }
            if row.resolution_minutes > 1 {
                reasons.insert("coarse_candle".into());
            }
            rows.push(row);
            timestamp = t.checked_add(MINUTE);
        }
    }
    if rows.is_empty() {
        reasons.insert("no_historical_candles".into());
    }
    Ok(Prices { rows, reasons })
}

#[pyfunction]
pub fn hsl_revised_prices(input_json: &str) -> PyResult<String> {
    let input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = minute_prices(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

// The live adapter already owns validated, immutable scalar observations. Avoid
// encoding those as JSON and returning per-minute provenance which that consumer
// immediately discards. Both bindings use exactly the same projection above.
type CandleRow = (
    i64,
    i64,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<i64>,
);
type PriceGrid = (BTreeMap<String, f64>, Option<(f64, i64)>, Vec<String>);
type NativeProjection = (RevisedHslPriceGrid, Option<(f64, i64)>, Vec<String>);

/// Immutable factual input, never a retained trading permission. Only the shared
/// source projection can construct it; Python exports are detached copies.
#[pyclass(module = "passivbot_rust", frozen)]
pub struct RevisedHslPriceGrid {
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) prices: BTreeMap<i64, f64>,
}

#[pymethods]
impl RevisedHslPriceGrid {
    pub fn values(&self) -> BTreeMap<String, f64> {
        self.prices
            .iter()
            .map(|(t, p)| (t.to_string(), *p))
            .collect()
    }
}

#[pyfunction]
pub fn hsl_revised_price_grid(
    start: i64,
    end: i64,
    candles: Vec<CandleRow>,
) -> PyResult<PriceGrid> {
    let (grid, last, reasons) = hsl_revised_native_price_grid(start, end, candles)?;
    Ok((grid.values(), last, reasons))
}

/// Immutable source rows, not a projected history or a trading permission.
/// Projection rechecks the exact window and observation clock on every call.
#[pyclass(module = "passivbot_rust", frozen)]
pub struct RevisedHslCandleSource {
    candles: Vec<Candle>,
}

#[pymethods]
impl RevisedHslCandleSource {
    #[new]
    fn new(candles: Vec<CandleRow>) -> Self {
        Self {
            candles: candles
                .into_iter()
                .map(
                    |(start, minutes, open, high, low, close, available_at)| Candle {
                        start,
                        minutes,
                        open,
                        high,
                        low,
                        close,
                        available_at,
                    },
                )
                .collect(),
        }
    }

    #[pyo3(signature = (start, end, observation_offset=0))]
    fn project(&self, start: i64, end: i64, observation_offset: i64) -> PyResult<NativeProjection> {
        let result = minute_prices_from_candles(start, end, &self.candles, observation_offset)
            .map_err(PyValueError::new_err)?;
        let last = result.rows.last().map(|row| (row.close, row.source_end));
        Ok((
            RevisedHslPriceGrid {
                start,
                end,
                prices: result
                    .rows
                    .into_iter()
                    .map(|row| (row.timestamp, row.close))
                    .collect(),
            },
            last,
            result.reasons.into_iter().collect(),
        ))
    }
}

#[pyfunction]
pub fn hsl_revised_native_price_grid(
    start: i64,
    end: i64,
    candles: Vec<CandleRow>,
) -> PyResult<NativeProjection> {
    RevisedHslCandleSource::new(candles).project(start, end, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_source_rechecks_observation_clock_and_interval() {
        let candles = vec![Candle {
            start: 0,
            minutes: 1,
            open: None,
            high: None,
            low: None,
            close: Some(100.0),
            available_at: Some(MINUTE),
        }];
        let visible = minute_prices_from_candles(0, MINUTE, &candles, 0).unwrap();
        assert_eq!(visible.rows.len(), 2);
        assert_eq!(visible.rows[0].close, 100.0);
        assert!(minute_prices_from_candles(0, MINUTE, &candles, 1)
            .unwrap()
            .rows
            .is_empty());
        assert!(minute_prices_from_candles(0, MINUTE - 1, &candles, 0)
            .unwrap()
            .rows
            .is_empty());
        assert!(minute_prices_from_candles(0, MINUTE, &candles, i64::MAX).is_err());
        assert_eq!(
            minute_prices_from_candles(0, MINUTE, &candles, 0)
                .unwrap()
                .rows
                .len(),
            2
        );
    }

    #[test]
    fn empty_prices_are_an_explicit_minimal_history_input() {
        let out = minute_prices(&Input {
            start: 0,
            end: MINUTE,
            candles: vec![],
        })
        .unwrap();
        assert!(out.rows.is_empty());
        assert!(out.reasons.contains("no_historical_candles"));
    }

    #[test]
    fn allocation_is_bounded() {
        assert!(minute_prices(&Input {
            start: 0,
            end: MAX_WINDOW + 1,
            candles: vec![]
        })
        .is_err());
        assert!(minute_prices(&Input {
            start: i64::MIN,
            end: i64::MAX,
            candles: vec![]
        })
        .is_err());
    }
}
