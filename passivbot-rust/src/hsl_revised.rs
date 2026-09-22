//! Revised HSL numerical kernel shared by diagnostic and trading-controller consumers.
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
    #[serde(skip)]
    pub(crate) last_peak_delta: Option<f64>,
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

pub(crate) fn validate_settings(span: f64, threshold: f64) -> Result<(), String> {
    if !span.is_finite()
        || span < 1.0
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
    {
        return Err("invalid revised HSL signal inputs".into());
    }
    Ok(())
}

/// One observation with an exact reconstructed peak-to-current currency loss.
/// The peak includes current UPNL, so realizing an existing loss is neutral.
pub(crate) fn singleton_from_loss(
    budget: f64,
    upnl: f64,
    loss: f64,
    span: f64,
    threshold: f64,
) -> Result<Signal, String> {
    validate_settings(span, threshold)?;
    if !budget.is_finite() || budget <= 0.0 || !upnl.is_finite() || !loss.is_finite() {
        return Err("invalid revised HSL signal inputs".into());
    }
    let loss = loss.max(0.0);
    let mut reasons = std::collections::BTreeSet::new();
    let equity = crate::hsl_revised_sum::currency_sum(&[budget, upnl], &mut reasons);
    let peak = crate::hsl_revised_sum::currency_sum(&[budget, upnl, loss], &mut reasons);
    let scale = if budget.abs().max(upnl.abs()).max(loss) > f64::MAX / 8.0 {
        0.125
    } else {
        1.0
    };
    let denominator = crate::hsl_revised_sum::currency_sum(
        &[budget * scale, upnl * scale, loss * scale],
        &mut reasons,
    );
    let mut approximate = !reasons.is_empty();
    let raw = if denominator > 0.0 {
        bounded(loss * scale / denominator, &mut approximate)
    } else {
        1.0
    };
    Ok(Signal {
        equity: vec![equity],
        peaks: vec![peak],
        raw: vec![raw],
        ema: vec![raw],
        panic: vec![raw > threshold],
        numeric_range_approximation: approximate,
        last_peak_delta: None,
    })
}

/// Batch reference semantics: final equity equals current budget plus current UPNL, EMA starts at
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
    signal_with_references(
        rows,
        budget,
        span,
        threshold,
        entry_reference,
        None,
        &[],
        anchor,
    )
}

/// A reference relative to current budget preserves small losses which would
/// disappear if the caller rounded budget + reference before signal evaluation.
pub(crate) fn signal_with_references(
    rows: &[Observation],
    budget: f64,
    span: f64,
    threshold: f64,
    entry_reference: Option<f64>,
    entry_reference_delta: Option<f64>,
    point_references: &[Option<f64>],
    anchor: &Observation,
) -> Result<Signal, String> {
    let mut out = Signal {
        equity: Vec::with_capacity(rows.len()),
        peaks: Vec::with_capacity(rows.len()),
        raw: Vec::with_capacity(rows.len()),
        ema: Vec::with_capacity(rows.len()),
        panic: Vec::with_capacity(rows.len()),
        numeric_range_approximation: false,
        last_peak_delta: None,
    };
    if !point_references.is_empty() && point_references.len() != rows.len() {
        return Err("invalid revised HSL signal inputs".into());
    }
    let summary = visit_signal(
        rows.len(),
        |i| rows[i],
        |i| point_references.get(i).copied().flatten(),
        budget,
        span,
        threshold,
        entry_reference,
        entry_reference_delta,
        anchor,
        |_, equity, peak, raw, ema, panic| {
            out.equity.push(equity);
            out.peaks.push(peak);
            out.raw.push(raw);
            out.ema.push(ema);
            out.panic.push(panic);
        },
    )?;
    out.numeric_range_approximation = summary.numeric_range_approximation;
    out.last_peak_delta = summary.last_peak_delta;
    Ok(out)
}

pub(crate) struct SignalSummary {
    pub numeric_range_approximation: bool,
    pub last_peak_delta: Option<f64>,
}

/// The same arithmetic can stream to its controller without allocating the five
/// diagnostic output arrays. The public numerical API collects this stream.
pub(crate) fn visit_signal(
    count: usize,
    row_at: impl Fn(usize) -> Observation,
    reference_at: impl Fn(usize) -> Option<f64>,
    budget: f64,
    span: f64,
    threshold: f64,
    entry_reference: Option<f64>,
    entry_reference_delta: Option<f64>,
    anchor: &Observation,
    mut visit: impl FnMut(usize, f64, f64, f64, f64, bool),
) -> Result<SignalSummary, String> {
    validate_settings(span, threshold)?;
    if !anchor.realized.is_finite()
        || !anchor.unrealized.is_finite()
        || count == 0
        || !budget.is_finite()
        || budget <= 0.0
        || entry_reference.is_some_and(|p| !p.is_finite())
        || entry_reference_delta.is_some_and(|p| !p.is_finite())
        || (entry_reference.is_some() && entry_reference_delta.is_some())
        || (0..count).filter_map(&reference_at).any(|r| !r.is_finite())
        || (entry_reference.is_some() && (0..count).any(|i| reference_at(i).is_some()))
        || (0..count)
            .map(&row_at)
            .any(|r| !r.realized.is_finite() || !r.unrealized.is_finite())
        || (1..count).any(|i| row_at(i - 1).timestamp_ms > row_at(i).timestamp_ms)
    {
        return Err("invalid revised HSL signal inputs".into());
    }
    // Keep extreme historical currency values scaled through ratio calculation.
    // Saturating an oversized peak before division can erase a real drawdown.
    let scale = if budget.abs() > f64::MAX / 8.0
        || anchor.realized.abs() > f64::MAX / 8.0
        || (0..count)
            .map(&row_at)
            .any(|r| r.realized.abs().max(r.unrealized.abs()) > f64::MAX / 8.0)
        || entry_reference.is_some_and(|r| r.abs() > f64::MAX / 8.0)
        || entry_reference_delta.is_some_and(|r| r.abs() > f64::MAX / 8.0)
        || (0..count)
            .filter_map(&reference_at)
            .any(|r| r.abs() > f64::MAX / 8.0)
    {
        0.125
    } else {
        1.0
    };
    let budget = budget * scale;
    let entry_reference = entry_reference.map(|r| r * scale);
    let entry_reference_delta = entry_reference_delta.map(|r| r * scale);
    let mut approximate = scale != 1.0;
    // Keep the same currency subtraction order as the collected reference.
    let delta = |row: &Observation, approximate: &mut bool| {
        let direct = (row.realized * scale - anchor.realized * scale) + row.unrealized * scale;
        if direct.is_finite() {
            direct
        } else {
            *approximate = true;
            let scaled = (row.realized * 0.25 - anchor.realized * 0.25) + row.unrealized * 0.25;
            bounded(scaled * 4.0, approximate)
        }
    };
    let first_delta = delta(&row_at(0), &mut approximate);
    let mut peak = entry_reference.unwrap_or(bounded(budget + first_delta, &mut approximate));
    let mut peak_delta = entry_reference.is_none().then_some(
        entry_reference_delta.map_or(first_delta, |reference| reference.max(first_delta)),
    );
    if let Some(relative_peak) = peak_delta {
        peak = bounded(budget + relative_peak, &mut approximate);
    }
    let alpha = 2.0 / (span + 1.0);
    let mut previous_minute = None;
    let mut baseline: Option<f64> = None;
    let mut previous_ema = None;
    for index in 0..count {
        let row = row_at(index);
        let current_delta = delta(&row, &mut approximate);
        let equity = if index + 1 == count
            && row.realized == anchor.realized
            && row.unrealized == anchor.unrealized
        {
            bounded(budget + anchor.unrealized * scale, &mut approximate)
        } else {
            bounded(budget + current_delta, &mut approximate)
        };
        peak = peak.max(equity);
        if let Some(relative_peak) = &mut peak_delta {
            *relative_peak = relative_peak.max(current_delta);
            if let Some(reference) = reference_at(index) {
                *relative_peak = relative_peak.max(reference * scale);
                peak = bounded(budget + *relative_peak, &mut approximate);
            }
        }
        let drawdown = if peak > 0.0 && peak_delta.is_some() {
            let relative_peak = peak_delta.unwrap();
            let loss = relative_peak - current_delta;
            let denominator = budget + relative_peak;
            bounded(
                if loss.is_finite() && denominator.is_finite() {
                    loss / denominator
                } else {
                    // Power-of-two scaling avoids overflow in either positive
                    // denominator or opposite-sign loss subtraction.
                    (relative_peak * 0.25 - current_delta * 0.25)
                        / (budget * 0.25 + relative_peak * 0.25)
                },
                &mut approximate,
            )
        } else if peak > 0.0 {
            // Division first avoids overflow when peak and negative equity have
            // opposite extreme magnitudes. Clamp only an unrepresentable ratio.
            bounded(
                if equity >= 0.0 {
                    (peak - equity) / peak
                } else {
                    1.0 - equity / peak
                },
                &mut approximate,
            )
        } else {
            // Explicit reference rule for a nonpositive historical peak.
            1.0
        };
        let minute = row.timestamp_ms.div_euclid(60_000);
        if previous_minute != Some(minute) {
            baseline = previous_ema;
        }
        let ema = baseline.map_or(drawdown, |prior| {
            bounded(alpha * drawdown + (1.0 - alpha) * prior, &mut approximate)
        });
        previous_ema = Some(ema);
        let displayed_equity = if scale == 1.0 {
            equity
        } else {
            bounded(equity / scale, &mut approximate)
        };
        let displayed_peak = if scale == 1.0 {
            peak
        } else {
            bounded(peak / scale, &mut approximate)
        };
        visit(
            index,
            displayed_equity,
            displayed_peak,
            drawdown,
            ema,
            drawdown.min(ema) > threshold,
        );
        previous_minute = Some(minute);
    }
    Ok(SignalSummary {
        numeric_range_approximation: approximate,
        last_peak_delta: if scale == 1.0 && !approximate {
            peak_delta
        } else {
            None
        },
    })
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
    use crate::hsl_revised_sum::currency_sum;

    #[test]
    fn relative_entry_peak_retains_loss_below_budget_ulp() {
        let row = Observation {
            timestamp_ms: 0,
            realized: 0.0,
            unrealized: -1.0,
        };
        let risk = signal_with_references(&[row], 1e16, 10_000.5, 0.0, None, Some(0.0), &[], &row)
            .unwrap();
        assert_eq!(risk.raw[0], 1e-16);
        assert_eq!(risk.ema[0], risk.raw[0]);
        assert!(risk.panic[0]);
    }

    #[test]
    fn currency_sum_retains_residuals_across_extreme_permutations() {
        for (terms, expected) in [
            ([-1e16, -1.0, 1e16], -1.0),
            ([1e308, -1e-300, -1e308], -1e-300),
            ([1e308, 1e308, -1e308], 1e308),
            ([-1e308, -1e308, 1e308], -1e308),
        ] {
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let mut reasons = std::collections::BTreeSet::new();
                assert_eq!(
                    currency_sum(&order.map(|i| terms[i]), &mut reasons),
                    expected
                );
            }
        }
        for sign in [-1.0, 1.0] {
            let mut reasons = std::collections::BTreeSet::new();
            assert_eq!(
                currency_sum(&[sign * 1e308; 3], &mut reasons),
                sign * f64::MAX
            );
            assert!(reasons.contains("numeric_range_approximation"));
        }
    }

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
        let r = signal(&rows(&[0.0, -100.0, -200.0]), 1000.0, 3.0, 0.125, None).unwrap();
        assert_eq!(r.equity, [1000.0, 900.0, 800.0]);
        assert_eq!(r.ema, [0.0, 0.05, 0.125]);
        assert_eq!(r.panic, [false, false, false]);
    }

    #[test]
    fn within_minute_replaces_ema_and_retains_boundary_peak() {
        let mut input = rows(&[0.0, -100.0, -200.0]);
        input[2].timestamp_ms = 60_001;
        let r = signal(&input, 1000.0, 3.0, 0.075, None).unwrap();
        assert_eq!(r.ema, [0.0, 0.05, 0.1]);
        assert_eq!(r.panic, [false, false, true]);
    }

    #[test]
    fn singleton_reference_has_no_implicit_warmup() {
        let r = signal(&rows(&[-100.0]), 1000.0, 1e6, 0.09, Some(1000.0)).unwrap();
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
        assert_eq!(r.equity, [100.0, 90.0]);
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
        assert_eq!(r.equity[2], 1.0 - f64::MAX / 2.0);
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
