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

/// One current observation with an independently reconstructed peak-to-current
/// currency loss. Keep the loss through division: budget + loss may round back
/// to budget even though their ratio is representable and relevant to a threshold.
pub(crate) fn singleton_from_loss(
    budget: f64,
    loss: f64,
    span: f64,
    threshold: f64,
) -> Result<Signal, String> {
    validate_settings(span, threshold)?;
    if !budget.is_finite() || budget <= 0.0 || !loss.is_finite() {
        return Err("invalid revised HSL signal inputs".into());
    }
    let loss = loss.max(0.0);
    let total = budget + loss;
    let mut approximate = false;
    let peak = bounded(total, &mut approximate);
    let raw = if total.is_finite() {
        loss / total
    } else {
        // Positive operands: half scaling cannot overflow or cancel a loss.
        (loss * 0.5) / (budget * 0.5 + loss * 0.5)
    };
    Ok(Signal {
        equity: vec![budget],
        peaks: vec![peak],
        raw: vec![raw],
        ema: vec![raw],
        panic: vec![raw > threshold],
        numeric_range_approximation: approximate,
    })
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
    validate_settings(span, threshold)?;
    if !anchor.realized.is_finite()
        || !anchor.unrealized.is_finite()
        || rows.is_empty()
        || !budget.is_finite()
        || budget <= 0.0
        || entry_reference.is_some_and(|p| !p.is_finite())
        || entry_reference_delta.is_some_and(|p| !p.is_finite())
        || (entry_reference.is_some() && entry_reference_delta.is_some())
        || (!point_references.is_empty() && point_references.len() != rows.len())
        || point_references.iter().flatten().any(|r| !r.is_finite())
        || (entry_reference.is_some() && point_references.iter().any(Option::is_some))
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
    let mut deltas = Vec::with_capacity(rows.len());
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
        deltas.push(delta);
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
    // An explicit absolute reference already has the caller's float precision.
    // Otherwise retain the relative currency peak, so adding a large budget
    // cannot erase an independently representable peak-to-current loss.
    let mut peak_delta = entry_reference
        .is_none()
        .then_some(entry_reference_delta.map_or(deltas[0], |reference| reference.max(deltas[0])));
    if let Some(relative_peak) = peak_delta {
        peak = bounded(budget + relative_peak, &mut out.numeric_range_approximation);
    }
    let alpha = 2.0 / (span + 1.0);
    let mut previous_minute = None;
    let mut baseline: Option<f64> = None;
    for (index, (row, equity)) in rows.iter().zip(out.equity.iter().copied()).enumerate() {
        peak = peak.max(equity);
        if let Some(relative_peak) = &mut peak_delta {
            *relative_peak = relative_peak.max(deltas[index]);
            if let Some(reference) = point_references.get(index).copied().flatten() {
                *relative_peak = relative_peak.max(reference);
                peak = bounded(
                    budget + *relative_peak,
                    &mut out.numeric_range_approximation,
                );
            }
        }
        let drawdown = if peak > 0.0 && peak_delta.is_some() {
            let relative_peak = peak_delta.unwrap();
            let loss = relative_peak - deltas[index];
            let denominator = budget + relative_peak;
            bounded(
                if loss.is_finite() && denominator.is_finite() {
                    loss / denominator
                } else {
                    // Power-of-two scaling avoids overflow in either positive
                    // denominator or opposite-sign loss subtraction.
                    (relative_peak * 0.25 - deltas[index] * 0.25)
                        / (budget * 0.25 + relative_peak * 0.25)
                },
                &mut out.numeric_range_approximation,
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
    use crate::hsl_revised_sum::currency_sum;

    #[test]
    fn relative_entry_peak_retains_loss_below_budget_ulp() {
        let row = Observation {
            timestamp_ms: 0,
            realized: 0.0,
            unrealized: -1.0,
        };
        let risk = signal_with_references(&[row], 1e16, 10_000.5, 0.0, None, Some(1.0), &[], &row)
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
