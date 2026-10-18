//! Factual price-grid normalization and configured budget support for v9.
use crate::hsl_intent::{self as controller, Decision};
use crate::hsl_snapshot::{self as snapshot, Input as Snapshot, Mode};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureBudget {
    pub wallet_exposure_limit: f64,
    pub total_wallet_exposure_limit: f64,
    pub we_excess_allowance_pct: f64,
}

/// Configured headroom, never actual current exposure, scales the slot budget.
pub(crate) fn budget(
    balance: f64,
    mode: &Mode,
    slots: u64,
    scale: bool,
    exposure: Option<&ExposureBudget>,
) -> Result<f64, String> {
    if scale && !matches!(mode, Mode::Coin) {
        return Err("HSL excess allowance budget scaling requires coin mode".into());
    }
    let divisor = if matches!(mode, Mode::Coin) { slots } else { 1 };
    if divisor == 0 {
        return Err("inactive HSL slot budget".into());
    }
    let mut multiplier = 1.0;
    if scale {
        let e = exposure.ok_or("HSL budget scaling requires current exposure configuration")?;
        if !e.wallet_exposure_limit.is_finite()
            || !e.total_wallet_exposure_limit.is_finite()
            || e.total_wallet_exposure_limit < 0.0
            || !e.we_excess_allowance_pct.is_finite()
            || e.we_excess_allowance_pct < 0.0
        {
            return Err("invalid HSL exposure budget inputs".into());
        }
        let base = if e.wallet_exposure_limit < 0.0 {
            e.total_wallet_exposure_limit / divisor as f64
        } else {
            e.wallet_exposure_limit
        };
        let params = crate::types::BotParams {
            total_wallet_exposure_limit: e.total_wallet_exposure_limit,
            risk_we_excess_allowance_pct: e.we_excess_allowance_pct,
            ..Default::default()
        };
        multiplier += crate::entries::effective_we_excess_allowance_pct(&params, base);
    }
    let result = balance / divisor as f64 * multiplier;
    if !result.is_finite() {
        return Err("invalid HSL balance budget".into());
    }
    Ok(result)
}

#[derive(Clone, Debug, Serialize)]
pub struct Output {
    pub decision: Option<Decision>,
    /// Reconstructed in-window lifecycle, for diagnostics only. Repeated evaluation
    /// may revise these events as facts/budget change; this is not an execution log.
    pub events: Vec<controller::LifecycleEvent>,
    pub reasons: BTreeSet<String>,
    pub observations: usize,
    pub episodes: usize,
}

/// Normalize only selected pairs to one bounded minute grid. Source-resolution
/// projection is already Rust-owned; these observations retain that causal cut.
/// Whole-scope absence remains sparse: never make a fictitious flat minute tape.
pub(crate) fn normalize(
    snapshot: &mut Snapshot,
    reasons: &mut BTreeSet<String>,
) -> Result<bool, String> {
    normalize_inner(snapshot, reasons, false)
}

/// The continuity ledger has already admitted every row against its own actual
/// receipt clock. A later correction can improve an older bootstrap prefix.
pub(crate) fn normalize_admitted(
    snapshot: &mut Snapshot,
    reasons: &mut BTreeSet<String>,
) -> Result<bool, String> {
    normalize_inner(snapshot, reasons, true)
}

fn normalize_inner(
    snapshot: &mut Snapshot,
    reasons: &mut BTreeSet<String>,
    admitted: bool,
) -> Result<bool, String> {
    let window = snapshot
        .now
        .checked_sub(snapshot.start)
        .filter(|w| (0..=90 * 86_400_000).contains(w))
        .ok_or("invalid HSL evaluation interval")?;
    let selected = snapshot::select(snapshot)?;
    let keys: BTreeSet<_> = selected
        .iter()
        .map(|p| {
            (
                p.symbol.clone(),
                p.position.pside == crate::hsl_history::PositionSide::Long,
            )
        })
        .collect();
    let mut histories = Vec::new();
    for (index, pair) in snapshot.pairs.iter().enumerate() {
        if !keys.contains(&(
            pair.symbol.clone(),
            pair.position.pside == crate::hsl_history::PositionSide::Long,
        )) {
            continue;
        }
        let prices: BTreeMap<_, _> = pair
            .prices
            .iter()
            .filter(|(t, p)| {
                snapshot.start <= **t
                    && **t
                        <= if admitted {
                            snapshot.now
                        } else {
                            snapshot.now.min(pair.prices_at)
                        }
                    && p.is_finite()
                    && **p > 0.0
            })
            .map(|(t, p)| (*t, *p))
            .collect();
        histories.push((index, prices));
    }
    let candle_free = histories.iter().all(|(_, p)| p.is_empty());
    if candle_free {
        reasons.insert("candle_free_reference".into());
        for (index, _) in histories {
            snapshot.pairs[index].prices.clear();
        }
        return Ok(true);
    }
    let mut grid = Vec::with_capacity((window / 60_000 + 2) as usize);
    let offset = (60_000 - snapshot.start.rem_euclid(60_000)) % 60_000;
    let mut next = snapshot.start.checked_add(offset);
    while let Some(t) = next.filter(|t| *t <= snapshot.now) {
        grid.push(t);
        next = t.checked_add(60_000);
    }
    // prepare() replaces this endpoint with observed current position and mark.
    if grid.last().copied() != Some(snapshot.now) {
        grid.push(snapshot.now);
    }
    for (index, prices) in histories {
        let pair = &mut snapshot.pairs[index];
        let mut aligned = BTreeMap::new();
        if let Some((_, first)) = prices.first_key_value() {
            for &t in &grid {
                let observed = prices.range(..=t).next_back();
                let value = observed.map_or(*first, |(_, p)| *p);
                if !prices.contains_key(&t) {
                    reasons.insert(
                        if observed.is_some() {
                            "forward_filled_price"
                        } else {
                            "backfilled_price"
                        }
                        .into(),
                    );
                }
                aligned.insert(t, value);
            }
        } else {
            // Mixed scope: preserve other pairs' history. Current mark is an
            // explicit estimator-only approximation for this absent price tape.
            reasons.insert("current_mark_history_estimate".into());
            aligned.extend(grid.iter().map(|t| (*t, pair.position.mark)));
        }
        pair.prices = aligned;
        // Rows have now been computed at evaluation time from causally eligible
        // observations. Original source skew remains in the collected reasons.
        pair.prices_at = snapshot.now;
    }
    Ok(false)
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn bounded_headroom_scales_only_explicit_coin_budgets() {
        let mut e = ExposureBudget {
            wallet_exposure_limit: -1.0,
            total_wallet_exposure_limit: 2.5,
            we_excess_allowance_pct: 0.44,
        };
        assert_eq!(budget(1000.0, &Mode::Coin, 10, false, None).unwrap(), 100.0);
        assert_eq!(
            budget(1000.0, &Mode::Coin, 10, true, Some(&e)).unwrap(),
            144.0
        );
        assert_eq!(
            budget(1000.0, &Mode::Coin, 1, true, Some(&e)).unwrap(),
            1000.0
        );
        e.wallet_exposure_limit = 2.0;
        assert_eq!(
            budget(1000.0, &Mode::Coin, 10, true, Some(&e)).unwrap(),
            125.0
        );
        e.total_wallet_exposure_limit = 0.0;
        assert_eq!(
            budget(1000.0, &Mode::Coin, 10, true, Some(&e)).unwrap(),
            100.0
        );
        assert!(budget(1000.0, &Mode::Coin, 10, true, None).is_err());
        assert!(budget(1000.0, &Mode::Pside, 1, true, Some(&e)).is_err());
        e.we_excess_allowance_pct = f64::NAN;
        assert!(budget(1000.0, &Mode::Coin, 10, true, Some(&e)).is_err());
    }
}
