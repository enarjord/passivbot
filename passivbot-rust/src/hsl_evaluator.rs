//! Snapshot-to-permission composition shared by hsl runtime adapters.
//! No prior decision, saved EMA, journal or cached permission is an input.
use crate::hsl::validate_settings;
use crate::hsl_controller::{self as controller, Decision, Restart};
use crate::hsl_snapshot::{self as snapshot, Input as Snapshot, Mode};
use crate::hsl_trace::{compose_prepared, compose_with_cashflow_peaks};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Deserialize)]
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
    if !result.is_finite() || result <= 0.0 {
        return Err("invalid HSL balance budget".into());
    }
    Ok(result)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub snapshot: Snapshot,
    pub slots: u64,
    #[serde(default)]
    pub scale_budget_with_excess_allowance: bool,
    #[serde(default)]
    pub exposure_budget: Option<ExposureBudget>,
    #[serde(deserialize_with = "crate::hsl_json::number")]
    pub span: f64,
    #[serde(deserialize_with = "crate::hsl_json::number")]
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub restart: Restart,
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
    #[serde(skip)]
    pub(crate) cursor: Option<controller::Cursor>,
}

/// Normalize only selected pairs to one bounded minute grid. Source-resolution
/// projection is already Rust-owned; these observations retain that causal cut.
/// Whole-scope absence remains sparse: never make a fictitious flat minute tape.
fn normalize(snapshot: &mut Snapshot, reasons: &mut BTreeSet<String>) -> Result<bool, String> {
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

/// Native live grids contain every minute but not the non-minute current cut.
/// Reconstruction already appends that cut using the current observed mark.
/// Reuse is allowed only when normalization cannot change any historical price.
fn normalized_minute_prefix(snapshot: &Snapshot) -> Result<bool, String> {
    if snapshot.now.rem_euclid(60_000) == 0 {
        return Ok(false);
    }
    let selected = snapshot::select(snapshot)?;
    if selected.is_empty() {
        return Ok(false);
    }
    let first = snapshot
        .start
        .checked_add((60_000 - snapshot.start.rem_euclid(60_000)) % 60_000);
    Ok(selected.iter().all(|pair| {
        if pair.prices_at != snapshot.now || pair.prices.is_empty() {
            return false;
        }
        let mut expected = first;
        for (&timestamp, &price) in &pair.prices {
            if expected != Some(timestamp)
                || timestamp >= snapshot.now
                || !price.is_finite()
                || price <= 0.0
            {
                return false;
            }
            expected = timestamp.checked_add(60_000);
        }
        expected.is_some_and(|t| t > snapshot.now)
    }))
}

pub fn evaluate(input: Input) -> Result<Output, String> {
    evaluate_inner::<false>(input)
}

pub(crate) fn evaluate_for_simulator(input: Input) -> Result<Output, String> {
    evaluate_inner::<true>(input)
}

fn evaluate_inner<const SEED: bool>(input: Input) -> Result<Output, String> {
    evaluate_with_reuse::<SEED, true>(input)
}

fn evaluate_with_reuse<const SEED: bool, const REUSE: bool>(
    mut input: Input,
) -> Result<Output, String> {
    validate_settings(input.span, input.threshold)?;
    input
        .snapshot
        .now
        .checked_sub(input.snapshot.start)
        .filter(|w| (0..=90 * 86_400_000).contains(w))
        .ok_or("invalid HSL evaluation interval")?;
    if input.cooldown_ms < 0 {
        return Err("negative HSL cooldown".into());
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
            cursor: None,
        });
    }
    let budget = budget(
        input.snapshot.balance,
        &input.snapshot.mode,
        input.slots,
        input.scale_budget_with_excess_allowance,
        input.exposure_budget.as_ref(),
    )?;
    let trace = if REUSE && already_normalized(&input.snapshot)? {
        compose_prepared(&input.snapshot, prepared, false)?
    } else if REUSE && normalized_minute_prefix(&input.snapshot)? {
        // Preserve the reference normalizer's endpoint diagnostic as well as values.
        reasons.insert("forward_filled_price".into());
        compose_prepared(&input.snapshot, prepared, false)?
    } else {
        drop(prepared); // Do not retain an unused full history during reconstruction.
        let candle_free = normalize(&mut input.snapshot, &mut reasons)?;
        compose_with_cashflow_peaks(&input.snapshot, candle_free)?
    };
    reasons.extend(trace.reasons);
    let episodes = trace.episodes.len();
    let replay_input = controller::Input {
        episodes: trace.episodes,
        now: input.snapshot.now,
        start: input.snapshot.start,
        budget,
        span: input.span,
        threshold: input.threshold,
        cooldown_ms: input.cooldown_ms,
        restart: input.restart,
    };
    let replay = if SEED {
        controller::replay_latest_with_seed(&replay_input)
    } else {
        controller::replay_latest_with_events(&replay_input)
    }?;
    let decisions = replay.decisions;
    if replay.numeric_range_approximation {
        reasons.insert("numeric_range_approximation".into());
    }
    let observations = replay.observations;
    let decision = decisions.into_iter().last().ok_or("empty HSL evaluation")?;
    Ok(Output {
        decision: Some(decision),
        events: replay.events,
        reasons,
        observations,
        episodes,
        cursor: replay.cursor,
    })
}

#[pyfunction]
pub fn hsl_evaluate(input_json: &str) -> PyResult<String> {
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
    grids: &[&crate::hsl_prices::HslPriceGrid],
) -> Result<(), String> {
    if grids.len() != input.snapshot.pairs.len() {
        return Err("HSL price grid count mismatch".into());
    }
    for (pair, grid) in input.snapshot.pairs.iter_mut().zip(grids) {
        if grid.start != input.snapshot.start
            || grid.end != input.snapshot.now
            || pair.prices_at != grid.end
            || !pair.prices.is_empty()
        {
            return Err("HSL price grid observation mismatch".into());
        }
        pair.prices = grid.prices.clone();
    }
    Ok(())
}

#[pyfunction]
pub fn hsl_evaluate_grids(
    input_json: &str,
    grids: Vec<PyRef<'_, crate::hsl_prices::HslPriceGrid>>,
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
        use crate::hsl_prices::HslPriceGrid;
        let mut input = fixture();
        input.snapshot.pairs[0].prices.clear();
        let grid = HslPriceGrid {
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
            "restart": "always"
        }))
        .unwrap()
    }

    #[test]
    fn minute_prefix_reuse_preserves_complete_reference_outputs() {
        let mut actions = BTreeSet::new();
        let mut reused = 0;
        for case in 0..2304 {
            let make = || {
                let mut value = json!({
                    "snapshot": {"now": 900001, "start": 0, "balance": 1000,
                        "balance_at": 900001, "config_at": 900001, "max_current_age_ms": 10000,
                        "mode": "unified", "pairs": [{"symbol": "A",
                            "position": {"size": 10, "basis": 100, "mark": 70,
                                "multiplier": 1, "inverse": false, "pside": "long"},
                            "position_at": 900001, "mark_at": 900001, "prices_at": 900001,
                            "fills_started_at": 900001, "fills_at": 900001,
                            "fills": [], "prices": {}, "revisions": [0,0,0,0]}]},
                    "slots": 1, "span": 2.5, "threshold": 0.05,
                    "cooldown_ms": 3600000, "restart": "always"
                });
                let now = 900000 + [0, 1, 12345, 59999][case % 4];
                let start = [0, 1, 59999][case / 4 % 3];
                let short = case & 16 != 0;
                let flat = case & 128 != 0;
                let sign = if short { -1 } else { 1 };
                let side = if short { "short" } else { "long" };
                let mode = ["coin", "pside", "unified"][case / 32 % 3];
                let snapshot = &mut value["snapshot"];
                snapshot["now"] = json!(now);
                snapshot["start"] = json!(start);
                snapshot["balance_at"] = json!(now);
                snapshot["config_at"] = json!(now);
                snapshot["mode"] = json!(mode);
                snapshot["balance"] = json!([1000, 250, 100][case % 3]);
                if mode != "unified" {
                    snapshot["pside"] = json!(side);
                }
                if mode == "coin" {
                    snapshot["symbol"] = json!("A");
                }
                let pair = &mut snapshot["pairs"][0];
                for key in [
                    "position_at",
                    "mark_at",
                    "prices_at",
                    "fills_started_at",
                    "fills_at",
                ] {
                    pair[key] = json!(now);
                }
                pair["position"]["pside"] = json!(side);
                pair["position"]["size"] = json!(if flat { 0 } else { 10 * sign });
                pair["position"]["inverse"] = json!(case & 512 != 0);
                pair["position"]["mark"] = json!([70, 100, 130][case % 3]);
                let first = start + (60000 - start % 60000) % 60000;
                for t in (first..now).step_by(60000) {
                    pair["prices"][t.to_string()] =
                        json!([80, 100, 120][(t as usize / 60000 + case) % 3]);
                }
                match case / 256 % 6 {
                    1 => {
                        pair["prices"]
                            .as_object_mut()
                            .unwrap()
                            .remove(&first.to_string());
                    }
                    2 => {
                        pair["prices"]["60001"] = json!(99);
                    }
                    3 => {
                        pair["prices"] = json!({});
                    }
                    4 => {
                        pair["prices"][now.to_string()] = json!(110);
                    }
                    5 => {
                        pair["prices"]["120000"] = json!(0);
                    }
                    _ => {}
                }
                pair["fills"] = json!([{"identity": "open", "timestamp": 120000,
                    "delta": 10 * sign, "price": 100, "realized": 0, "fee": -0.25,
                    "sequence": 0, "revision": 0}]);
                if flat {
                    pair["fills"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"identity":"close",
                        "timestamp": 600000, "delta": -10 * sign, "price": if short {120} else {80},
                        "realized": -200, "fee": -0.25, "sequence": 1, "revision": 0}));
                }
                if mode != "coin" {
                    let mut second = snapshot["pairs"][0].clone();
                    second["symbol"] = json!("B");
                    second["position"]["mark"] = json!(95);
                    if case % 8 == 2 {
                        second["prices"][now.to_string()] = json!(95);
                    }
                    snapshot["pairs"].as_array_mut().unwrap().push(second);
                }
                value["span"] = json!([1.0, 2.5, 703.0][case % 3]);
                value["restart"] = json!(if case & 64 != 0 { "never" } else { "always" });
                value["cooldown_ms"] = json!([0, 60000, 3600000][case % 3]);
                value["slots"] = json!(if case % 17 == 0 { 0 } else { 1 });
                serde_json::from_value::<Input>(value).unwrap()
            };
            if normalized_minute_prefix(&make().snapshot).unwrap() {
                reused += 1;
            }
            let fast = evaluate_with_reuse::<false, true>(make()).unwrap();
            let reference = evaluate_with_reuse::<false, false>(make()).unwrap();
            if let Some(decision) = &fast.decision {
                actions.insert(format!("{:?}", decision.action));
            }
            assert_eq!(
                serde_json::to_value(&fast).unwrap(),
                serde_json::to_value(&reference).unwrap(),
                "case {case}"
            );
            // Simulator seed/cursor behavior remains unchanged too.
            let fast = evaluate_with_reuse::<true, true>(make()).unwrap();
            let reference = evaluate_with_reuse::<true, false>(make()).unwrap();
            assert_eq!(
                serde_json::to_value(&fast).unwrap(),
                serde_json::to_value(&reference).unwrap(),
                "seed case {case}"
            );
            assert_eq!(
                format!("{:?}", fast.cursor),
                format!("{:?}", reference.cursor),
                "cursor case {case}"
            );
        }
        assert!(reused > 100);
        assert!(
            actions.contains("Normal") && actions.contains("Panic") && actions.contains("Halted"),
            "{actions:?}"
        );
    }

    #[test]
    fn missing_opening_uses_current_basis_without_inventing_old_exposure() {
        let output = evaluate(fixture()).unwrap();
        assert_eq!(output.observations, 1);
        assert!(output.reasons.contains("backfilled_price"));
        assert!(output.reasons.contains("forward_filled_price"));
        let decision = output.decision.unwrap();
        assert_eq!(decision.action, controller::Action::Normal);
        assert_eq!(decision.raw, 0.0);
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
