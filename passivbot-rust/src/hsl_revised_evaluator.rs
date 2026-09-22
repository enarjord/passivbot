//! Snapshot-to-permission composition shared by revised runtime adapters.
//! No prior decision, saved EMA, journal or cached permission is an input.
use crate::hsl_revised::validate_settings;
use crate::hsl_revised_controller::{self as controller, Decision, Intervention, Restart};
use crate::hsl_revised_snapshot::{self as snapshot, Input as Snapshot, Mode};
use crate::hsl_revised_trace::{compose_prepared, compose_with_cashflow_peaks};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub snapshot: Snapshot,
    pub slots: u64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub span: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub restart: Restart,
    pub intervention: Intervention,
}

#[derive(Debug, Serialize)]
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
fn normalize(snapshot: &mut Snapshot, reasons: &mut BTreeSet<String>) -> Result<bool, String> {
    let window = snapshot
        .now
        .checked_sub(snapshot.start)
        .filter(|w| (0..=90 * 86_400_000).contains(w))
        .ok_or("invalid revised HSL evaluation interval")?;
    let selected = snapshot::select(snapshot)?;
    let keys: BTreeSet<_> = selected
        .iter()
        .map(|p| {
            (
                p.symbol.clone(),
                p.position.pside == crate::hsl_revised_history::PositionSide::Long,
            )
        })
        .collect();
    let mut histories = Vec::new();
    for (index, pair) in snapshot.pairs.iter().enumerate() {
        if !keys.contains(&(
            pair.symbol.clone(),
            pair.position.pside == crate::hsl_revised_history::PositionSide::Long,
        )) {
            continue;
        }
        let prices: BTreeMap<_, _> = pair
            .prices
            .iter()
            .filter(|(t, p)| {
                snapshot.start <= **t
                    && **t <= snapshot.now.min(pair.prices_at)
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

/// True only when normalization would leave every selected price and source
/// timestamp unchanged. Validation/reconstruction always runs before this check.
fn already_normalized(snapshot: &Snapshot) -> Result<bool, String> {
    let selected = snapshot::select(snapshot)?;
    if selected.is_empty() {
        return Ok(false);
    }
    let offset = (60_000 - snapshot.start.rem_euclid(60_000)) % 60_000;
    let first = snapshot
        .start
        .checked_add(offset)
        .filter(|t| *t <= snapshot.now)
        .unwrap_or(snapshot.now);
    Ok(selected.iter().all(|pair| {
        if pair.prices_at != snapshot.now {
            return false;
        }
        let mut expected = Some(first);
        for (&timestamp, &price) in &pair.prices {
            if expected != Some(timestamp) || !price.is_finite() || price <= 0.0 {
                return false;
            }
            expected = if timestamp == snapshot.now {
                None
            } else {
                Some(
                    timestamp
                        .checked_add(60_000)
                        .filter(|t| *t <= snapshot.now)
                        .unwrap_or(snapshot.now),
                )
            };
        }
        expected.is_none()
    }))
}

pub fn evaluate(mut input: Input) -> Result<Output, String> {
    validate_settings(input.span, input.threshold)?;
    input
        .snapshot
        .now
        .checked_sub(input.snapshot.start)
        .filter(|w| (0..=90 * 86_400_000).contains(w))
        .ok_or("invalid revised HSL evaluation interval")?;
    if input.cooldown_ms < 0 {
        return Err("negative revised HSL cooldown".into());
    }
    // Validate current observations before inactivity or price approximation.
    // Historical defects remain diagnostic, never an old readiness certificate.
    let prepared = snapshot::prepare(&input.snapshot)?;
    let mut reasons = prepared.reasons.clone();
    if matches!(input.snapshot.mode, Mode::Coin) && input.slots == 0 {
        reasons.insert("inactive_scope".into());
        return Ok(Output {
            decision: None,
            events: Vec::new(),
            reasons,
            observations: 0,
            episodes: 0,
        });
    }
    let budget = if matches!(input.snapshot.mode, Mode::Coin) {
        input.snapshot.balance / input.slots as f64
    } else {
        input.snapshot.balance
    };
    let trace = if already_normalized(&input.snapshot)? {
        compose_prepared(&input.snapshot, prepared, false)?
    } else {
        drop(prepared); // Do not retain an unused full history during reconstruction.
        let candle_free = normalize(&mut input.snapshot, &mut reasons)?;
        compose_with_cashflow_peaks(&input.snapshot, candle_free)?
    };
    reasons.extend(trace.reasons);
    let episodes = trace.episodes.len();
    let replay = controller::replay_latest_with_events(&controller::Input {
        episodes: trace.episodes,
        now: input.snapshot.now,
        start: input.snapshot.start,
        budget,
        span: input.span,
        threshold: input.threshold,
        cooldown_ms: input.cooldown_ms,
        restart: input.restart,
        intervention: input.intervention,
    })?;
    let decisions = replay.decisions;
    if replay.numeric_range_approximation {
        reasons.insert("numeric_range_approximation".into());
    }
    let observations = replay.observations;
    let decision = decisions
        .into_iter()
        .last()
        .ok_or("empty revised HSL evaluation")?;
    Ok(Output {
        decision: Some(decision),
        events: replay.events,
        reasons,
        observations,
        episodes,
    })
}

#[pyfunction]
pub fn hsl_revised_evaluate(input_json: &str) -> PyResult<String> {
    let input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let output = evaluate(input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compact metadata and immutable native price inputs use the same evaluator as
/// the JSON/reference boundary. A grid must belong to this exact observation cut;
/// callers cannot combine it with a second price source or silently drop pairs.
fn attach_grids(
    input: &mut Input,
    grids: &[&crate::hsl_revised_prices::RevisedHslPriceGrid],
) -> Result<(), String> {
    if grids.len() != input.snapshot.pairs.len() {
        return Err("revised HSL price grid count mismatch".into());
    }
    for (pair, grid) in input.snapshot.pairs.iter_mut().zip(grids) {
        if grid.start != input.snapshot.start
            || grid.end != input.snapshot.now
            || pair.prices_at != grid.end
            || !pair.prices.is_empty()
        {
            return Err("revised HSL price grid observation mismatch".into());
        }
        pair.prices = grid.prices.clone();
    }
    Ok(())
}

#[pyfunction]
pub fn hsl_revised_evaluate_grids(
    input_json: &str,
    grids: Vec<PyRef<'_, crate::hsl_revised_prices::RevisedHslPriceGrid>>,
) -> PyResult<String> {
    let mut input: Input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let refs: Vec<_> = grids.iter().map(|grid| &**grid).collect();
    attach_grids(&mut input, &refs).map_err(PyValueError::new_err)?;
    let output = evaluate(input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_price_inputs_preserve_values_and_reject_other_observation_cuts() {
        use crate::hsl_revised_prices::RevisedHslPriceGrid;
        let mut input = fixture();
        input.snapshot.pairs[0].prices.clear();
        let grid = RevisedHslPriceGrid {
            start: input.snapshot.start,
            end: input.snapshot.now,
            prices: BTreeMap::from([(60000, 98.0), (120000, 95.0)]),
        };
        assert!(attach_grids(&mut input, &[]).is_err());
        attach_grids(&mut input, &[&grid]).unwrap();
        assert_eq!(input.snapshot.pairs[0].prices, grid.prices);
        assert!(attach_grids(&mut input, &[&grid]).is_err());
        input.snapshot.pairs[0].prices.clear();
        input.snapshot.start += 1;
        assert!(attach_grids(&mut input, &[&grid]).is_err());
    }

    fn fixture() -> Input {
        serde_json::from_value(json!({
            "snapshot": {
                "now": 180000, "start": 0, "balance": 1000,
                "balance_at": 180000, "config_at": 0, "max_current_age_ms": 120000,
                "mode": "unified", "pairs": [{
                    "symbol": "A", "position": {"size": 10, "basis": 100,
                        "mark": 100, "multiplier": 1, "inverse": false, "pside": "long"},
                    "position_at": 180000, "mark_at": 180000,
                    "fills_started_at": 180000, "fills_at": 180000,
                    "prices_at": 180000, "fills": [], "prices": {"60000": 110},
                    "revisions": [0,0,0,0]
                }]
            },
            "slots": 1, "span": 1, "threshold": 0.05, "cooldown_ms": 0,
            "restart": "always", "intervention": "panic"
        }))
        .unwrap()
    }

    #[test]
    fn selected_sparse_prices_reach_current_permission() {
        let output = evaluate(fixture()).unwrap();
        assert_eq!(output.observations, 4);
        assert!(output.reasons.contains("backfilled_price"));
        assert!(output.reasons.contains("forward_filled_price"));
        let decision = output.decision.unwrap();
        assert_eq!(decision.action, controller::Action::Panic);
        assert!((decision.raw - 100.0 / 1100.0).abs() < 1e-14);
    }
    #[test]
    fn dense_reuse_matches_full_normalization_and_reconstruction() {
        for (start, now) in [(0, 180_000), (1, 180_001), (179_999, 180_001), (1, 1)] {
            let mut input = fixture();
            let snapshot = &mut input.snapshot;
            snapshot.start = start;
            snapshot.now = now;
            snapshot.balance_at = now;
            let pair = &mut snapshot.pairs[0];
            pair.position_at = now;
            pair.mark_at = now;
            pair.prices_at = now;
            pair.fills_started_at = Some(now);
            pair.fills_at = Some(now);
            pair.prices = BTreeMap::from([(start, 110.0), (now, 100.0)]);
            normalize(snapshot, &mut BTreeSet::new()).unwrap();
            assert!(already_normalized(snapshot).unwrap());
            let reused =
                compose_prepared(snapshot, snapshot::prepare(snapshot).unwrap(), false).unwrap();
            let mut reasons = BTreeSet::new();
            assert!(!normalize(snapshot, &mut reasons).unwrap());
            assert!(reasons.is_empty());
            let rebuilt = compose_with_cashflow_peaks(snapshot, false).unwrap();
            assert_eq!(
                serde_json::to_value(reused).unwrap(),
                serde_json::to_value(rebuilt).unwrap()
            );
        }
    }

    #[test]
    fn changed_or_incomplete_price_inputs_use_full_normalization() {
        for change in 0..6 {
            let mut input = fixture();
            let snapshot = &mut input.snapshot;
            normalize(snapshot, &mut BTreeSet::new()).unwrap();
            assert!(already_normalized(snapshot).unwrap());
            let pair = &mut snapshot.pairs[0];
            match change {
                0 => {
                    pair.prices.remove(&60_000);
                }
                1 => {
                    pair.prices.insert(30_000, 100.0);
                }
                2 => {
                    pair.prices.insert(60_000, f64::NAN);
                }
                3 => {
                    pair.prices.insert(60_000, 0.0);
                }
                4 => {
                    pair.prices_at -= 1;
                }
                _ => {
                    pair.prices.clear();
                }
            }
            assert!(!already_normalized(snapshot).unwrap());
        }
    }
}
