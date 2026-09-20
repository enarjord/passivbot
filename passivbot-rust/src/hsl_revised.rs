//! Revised HSL numerical kernel. Not yet connected to any trading controller.
//!
//! Inputs are already scoped currency observations, not individual drawdown ratios.
//! Reconstruction, lifecycle and execution adapters must not use this kernel alone
//! as a complete HSL controller.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub timestamp_ms: i64,
    pub realized: f64,
    pub unrealized: f64,
}

#[derive(Debug, Serialize)]
pub struct Signal {
    pub equity: Vec<f64>,
    pub peaks: Vec<f64>,
    pub raw: Vec<f64>,
    pub ema: Vec<f64>,
    pub panic: Vec<bool>,
    pub numeric_range_approximation: bool,
}

fn bounded(value: f64, approximate: &mut bool) -> f64 {
    if value.is_finite() {
        value
    } else {
        *approximate = true;
        if value.is_sign_negative() {
            -f64::MAX
        } else {
            f64::MAX
        }
    }
}

/// Batch reference semantics: final equity equals current budget, EMA starts at
/// the first raw drawdown, and updates in the same minute replace that minute's
/// contribution. Peak tracking still sees each ordered boundary sample.
pub fn signal(
    rows: &[Observation],
    budget: f64,
    span: f64,
    threshold: f64,
    entry_reference: Option<f64>,
) -> Result<Signal, String> {
    let anchor = rows.last().ok_or("invalid revised HSL signal inputs")?;
    signal_with_anchor(rows, budget, span, threshold, entry_reference, anchor)
}

/// Episode resets clear peak/EMA, while currency rebasing retains the shared
/// current scope endpoint. A completed episode may end at nonpositive equity.
pub fn signal_with_anchor(
    rows: &[Observation],
    budget: f64,
    span: f64,
    threshold: f64,
    entry_reference: Option<f64>,
    anchor: &Observation,
) -> Result<Signal, String> {
    if !anchor.realized.is_finite()
        || !anchor.unrealized.is_finite()
        || rows.is_empty()
        || !budget.is_finite()
        || budget <= 0.0
        || !span.is_finite()
        || span < 1.0
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
        || entry_reference.is_some_and(|p| !p.is_finite())
        || rows
            .iter()
            .any(|r| !r.realized.is_finite() || !r.unrealized.is_finite())
        || rows
            .windows(2)
            .any(|w| w[0].timestamp_ms > w[1].timestamp_ms)
    {
        return Err("invalid revised HSL signal inputs".into());
    }
    let mut out = Signal {
        equity: Vec::with_capacity(rows.len()),
        peaks: Vec::with_capacity(rows.len()),
        raw: Vec::with_capacity(rows.len()),
        ema: Vec::with_capacity(rows.len()),
        panic: Vec::with_capacity(rows.len()),
        numeric_range_approximation: false,
    };
    let last = anchor;
    // Subtract the common currency offset first: a large realized baseline must
    // not erase an otherwise representable unrealized change or the budget.
    for row in rows {
        let realized_delta = row.realized - last.realized;
        let unrealized_delta = row.unrealized - last.unrealized;
        let direct = realized_delta + unrealized_delta;
        let delta = if direct.is_finite() {
            direct
        } else {
            // Opposite oversized deltas may cancel to a representable result.
            // Do not saturate each independently and erase that residual risk.
            out.numeric_range_approximation = true;
            let scaled = (row.realized * 0.25 - last.realized * 0.25)
                + (row.unrealized * 0.25 - last.unrealized * 0.25);
            bounded(scaled * 4.0, &mut out.numeric_range_approximation)
        };
        out.equity.push(bounded(
            budget + delta,
            &mut out.numeric_range_approximation,
        ));
    }
    // Exact endpoint identity, including after range-limited history estimation.
    if rows
        .last()
        .is_some_and(|r| r.realized == anchor.realized && r.unrealized == anchor.unrealized)
    {
        *out.equity.last_mut().unwrap() = budget;
    }
    let mut peak = entry_reference.unwrap_or(out.equity[0]);
    let alpha = 2.0 / (span + 1.0);
    let mut previous_minute = None;
    let mut baseline: Option<f64> = None;
    for (row, equity) in rows.iter().zip(out.equity.iter().copied()) {
        peak = peak.max(equity);
        let drawdown = if peak > 0.0 {
            // Division first avoids overflow when peak and negative equity have
            // opposite extreme magnitudes. Clamp only an unrepresentable ratio.
            bounded(
                if equity >= 0.0 {
                    (peak - equity) / peak
                } else {
                    1.0 - equity / peak
                },
                &mut out.numeric_range_approximation,
            )
        } else {
            // Explicit reference rule for a nonpositive historical peak.
            1.0
        };
        let minute = row.timestamp_ms.div_euclid(60_000);
        if previous_minute != Some(minute) {
            baseline = out.ema.last().copied();
        }
        let ema = baseline.map_or(drawdown, |prior| {
            bounded(
                alpha * drawdown + (1.0 - alpha) * prior,
                &mut out.numeric_range_approximation,
            )
        });
        out.peaks.push(peak);
        out.raw.push(drawdown);
        out.ema.push(ema);
        out.panic.push(drawdown.min(ema) > threshold);
        previous_minute = Some(minute);
    }
    Ok(out)
}

#[pyfunction(name = "hsl_revised_signal", signature = (rows, budget, span, threshold, entry_reference=None))]
pub fn signal_py(
    rows: Vec<(i64, f64, f64)>,
    budget: f64,
    span: f64,
    threshold: f64,
    entry_reference: Option<f64>,
) -> PyResult<String> {
    let rows: Vec<_> = rows
        .into_iter()
        .map(|(timestamp_ms, realized, unrealized)| Observation {
            timestamp_ms,
            realized,
            unrealized,
        })
        .collect();
    let result =
        signal(&rows, budget, span, threshold, entry_reference).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(values: &[f64]) -> Vec<Observation> {
        values
            .iter()
            .enumerate()
            .map(|(i, &unrealized)| Observation {
                timestamp_ms: i as i64 * 60_000,
                realized: 0.0,
                unrealized,
            })
            .collect()
    }

    #[test]
    fn known_equity_drawdown_and_threshold() {
        let r = signal(&rows(&[0.0, -100.0, -200.0]), 800.0, 3.0, 0.125, None).unwrap();
        assert_eq!(r.equity, [1000.0, 900.0, 800.0]);
        assert_eq!(r.ema, [0.0, 0.05, 0.125]);
        assert_eq!(r.panic, [false, false, false]);
    }

    #[test]
    fn within_minute_replaces_ema_and_retains_boundary_peak() {
        let mut input = rows(&[0.0, -100.0, -200.0]);
        input[2].timestamp_ms = 60_001;
        let r = signal(&input, 800.0, 3.0, 0.075, None).unwrap();
        assert_eq!(r.ema, [0.0, 0.05, 0.1]);
        assert_eq!(r.panic, [false, false, true]);
    }

    #[test]
    fn singleton_reference_has_no_implicit_warmup() {
        let r = signal(&rows(&[-100.0]), 1000.0, 1e6, 0.09, Some(1100.0)).unwrap();
        assert_eq!(r.raw, r.ema);
        assert!(r.panic[0]);
    }

    #[test]
    fn currency_offsets_do_not_erase_small_unrealized_changes() {
        let mut input = rows(&[0.0, -10.0]);
        for r in &mut input {
            r.realized = 1e100;
        }
        let r = signal(&input, 100.0, 1.0, 0.05, None).unwrap();
        assert_eq!(r.equity, [110.0, 100.0]);
        assert!(r.panic[1]);
    }

    #[test]
    fn extreme_finite_history_returns_bounded_visible_estimate() {
        let r = signal(
            &rows(&[f64::MAX, -f64::MAX, -f64::MAX / 2.0]),
            1.0,
            2.5,
            0.1,
            None,
        )
        .unwrap();
        assert!(r.numeric_range_approximation);
        assert!(r
            .equity
            .iter()
            .chain(&r.peaks)
            .chain(&r.raw)
            .chain(&r.ema)
            .all(|v| v.is_finite()));
        assert_eq!(r.equity[2], 1.0);
        assert!(r.panic[2]);
    }

    #[test]
    fn invalid_input_is_not_a_false_decision() {
        assert!(signal(&[], 1.0, 1.0, 0.1, None).is_err());
        assert!(signal(&rows(&[f64::NAN]), 1.0, 1.0, 0.1, None).is_err());
        assert!(signal(&rows(&[0.0]), 0.0, 1.0, 0.1, None).is_err());
        assert!(signal(&rows(&[0.0]), 1.0, 0.0, 0.1, None).is_err());
    }

    #[test]
    fn opposite_overflowing_deltas_keep_their_finite_residual() {
        let input = [
            Observation {
                timestamp_ms: 0,
                realized: f64::MAX,
                unrealized: -f64::MAX / 2.0,
            },
            Observation {
                timestamp_ms: 60_000,
                realized: -f64::MAX,
                unrealized: f64::MAX,
            },
        ];
        let r = signal(&input, 1.0, 1.0, 0.1, None).unwrap();
        assert!(r.numeric_range_approximation);
        assert!(r.equity[0] > f64::MAX / 3.0);
        assert!(r.panic[1]);
    }
}
