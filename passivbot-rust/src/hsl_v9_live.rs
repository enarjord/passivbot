//! Bounded live fact reconstruction for the forward v9 core. No previous
//! permission, order tag, terminal flag or persisted decision is an input.
use crate::hsl_decay::{self, Event, Policy, ScopeState};
use crate::hsl_intent::{Action, Decision, LifecycleEvent};
use crate::hsl_snapshot::Input as Snapshot;
use crate::hsl_support::{budget, ExposureBudget};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[path = "hsl_v9_continuity.rs"]
mod continuity;
pub use continuity::HslV9Replay;

fn exact_upnl(values: impl IntoIterator<Item = f64>) -> Result<f64, String> {
    let mut sum = crate::hsl_sum::CurrencySum::new();
    for value in values {
        if !value.is_finite() {
            return Err("nonfinite scoped v9 UPNL constituent".into());
        }
        sum.add(value);
    }
    let value = sum.value(&mut BTreeSet::new());
    if !value.is_finite() {
        return Err("nonfinite scoped v9 UPNL".into());
    }
    Ok(value)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetFact {
    pub timestamp: i64,
    /// Capture provenance; timestamp is the coherent budget source clock.
    #[serde(default)]
    pub observed_at: Option<i64>,
    /// -1 before the first fill; n after zero-based fill n in that timestamp's
    /// canonical cohort; i64::MAX for a later current-account observation.
    pub order: i64,
    pub balance: f64,
    pub slots: u64,
    pub scale_budget_with_excess_allowance: bool,
    pub exposure_budget: Option<ExposureBudget>,
    pub estimated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub snapshot: Snapshot,
    pub slots: u64,
    pub scale_budget_with_excess_allowance: bool,
    pub exposure_budget: Option<ExposureBudget>,
    pub lookback_span_minutes: f64,
    pub span: f64,
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub budget_observations: Vec<BudgetFact>,
    /// Previously supported prices explicitly disputed by corrected source facts.
    #[serde(default)]
    pub invalidated_price_symbols: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub unavailable_reason: Option<String>,
    pub decision: Option<Decision>,
    pub risk: Option<hsl_decay::Decision>,
    pub events: Vec<LifecycleEvent>,
    pub reasons: BTreeSet<String>,
    pub observations: usize,
    pub episodes: usize,
}

#[derive(Clone, Default)]
struct Recorder {
    state: ScopeState,
    red_at: Option<i64>,
    flat_at: Option<i64>,
    events: Vec<LifecycleEvent>,
    observations: usize,
    episodes: usize,
    exposed: bool,
}

impl Recorder {
    fn record(&mut self, d: &hsl_decay::Decision) {
        self.observations += 1;
        if d.exposed && !self.exposed {
            self.episodes += 1;
            self.red_at = None;
            self.flat_at = None;
        }
        if d.exposed && d.red && self.red_at.is_none() {
            self.red_at = Some(d.timestamp_ms);
            self.events.push(LifecycleEvent {
                timestamp: d.timestamp_ms,
                kind: "red",
                red_at: d.timestamp_ms,
                flat_at: None,
                reason: "drawdown",
                raw: Some(d.raw),
                ema: Some(d.ema),
            });
        }
        if d.flatten_qualified {
            let red = self.red_at.unwrap_or(d.timestamp_ms);
            self.red_at = Some(red);
            self.flat_at = Some(d.timestamp_ms);
            self.events.push(LifecycleEvent {
                timestamp: d.timestamp_ms,
                kind: "flat",
                red_at: red,
                flat_at: self.flat_at,
                reason: "stop_flattened",
                raw: Some(d.raw),
                ema: Some(d.ema),
            });
        } else if (d.exposed && !d.red) || (!d.exposed && d.action == Action::Normal) {
            if let (Some(red), Some(flat)) = (self.red_at, self.flat_at) {
                self.events.push(LifecycleEvent {
                    timestamp: d.timestamp_ms,
                    kind: "restart",
                    red_at: red,
                    flat_at: Some(flat),
                    reason: "cooldown_complete",
                    raw: None,
                    ema: None,
                });
            }
            self.red_at = None;
            self.flat_at = None;
        }
        self.exposed = d.exposed;
    }

    fn observe(&mut self, policy: &Policy, event: Event) -> Result<(), String> {
        let d = self.state.observe(policy, event)?;
        self.record(&d);
        Ok(())
    }
}

fn scope_budget(
    input: &Input,
    timestamp: i64,
    order: i64,
    reasons: &mut BTreeSet<String>,
) -> Result<f64, String> {
    if let Some(fact) = input
        .budget_observations
        .iter()
        .rev()
        .find(|f| (f.timestamp, f.order) <= (timestamp, order))
    {
        if fact.estimated {
            reasons.insert("historical_budget_estimate".into());
        }
        budget(
            fact.balance,
            &input.snapshot.mode,
            fact.slots,
            fact.scale_budget_with_excess_allowance,
            fact.exposure_budget.as_ref(),
        )
    } else {
        reasons.insert("historical_budget_estimate".into());
        budget(
            input.snapshot.balance,
            &input.snapshot.mode,
            input.slots,
            input.scale_budget_with_excess_allowance,
            input.exposure_budget.as_ref(),
        )
    }
}

pub fn evaluate(input: Input) -> Result<Output, String> {
    continuity::one_shot(input)
}

#[pyfunction]
pub fn hsl_v9_evaluate(input_json: &str) -> PyResult<String> {
    let input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let output = evaluate(input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn hsl_v9_evaluate_grids(
    input_json: &str,
    grids: Vec<PyRef<'_, crate::hsl_prices::HslPriceGrid>>,
) -> PyResult<String> {
    let mut input: Input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if grids.len() != input.snapshot.pairs.len() {
        return Err(PyValueError::new_err("v9 price grid count mismatch"));
    }
    for (pair, grid) in input.snapshot.pairs.iter_mut().zip(grids) {
        if grid.start != input.snapshot.start
            || grid.end != input.snapshot.now
            || pair.prices_at != grid.end
            || !pair.prices.is_empty()
        {
            return Err(PyValueError::new_err("v9 price grid observation mismatch"));
        }
        pair.prices = grid.prices.clone();
    }
    let output = evaluate(input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealizedFill {
    identity: String,
    timestamp: i64,
    net: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealizedInput {
    span_minutes: f64,
    start: i64,
    now: i64,
    known_origin: bool,
    fills: Vec<RealizedFill>,
}

/// Account-wide continuous memory consumes the manager's canonical ordered
/// facts. Same-time cashflows remain individual events, never a netted cohort.
#[pyfunction]
pub fn realized_memory_v9(input_json: &str) -> PyResult<String> {
    let input: RealizedInput =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if input.start < 0 || input.start > input.now {
        return Err(PyValueError::new_err("invalid v9 realized-memory interval"));
    }
    let policy = Policy::new(input.span_minutes, 1.0, 1.0, 0).map_err(PyValueError::new_err)?;
    let mut memory = hsl_decay::RealizedMemory::new(input.span_minutes, input.start)
        .map_err(PyValueError::new_err)?;
    let mut previous = input.start;
    let mut identities = BTreeSet::new();
    let mut same_time = false;
    for (i, fill) in input.fills.iter().enumerate() {
        if fill.timestamp < previous
            || fill.timestamp > input.now
            || fill.identity.is_empty()
            || !identities.insert(&fill.identity)
        {
            return Err(PyValueError::new_err(
                "unordered, duplicate or future v9 realized fill",
            ));
        }
        same_time |= i > 0 && fill.timestamp == previous;
        memory
            .fill(fill.timestamp, fill.net)
            .map_err(PyValueError::new_err)?;
        previous = fill.timestamp;
    }
    memory
        .advance_to(input.now)
        .map_err(PyValueError::new_err)?;
    let minutes = input.now.div_euclid(60_000) - input.start.div_euclid(60_000);
    // Diagnostic attenuation only, not the state-update arithmetic graph.
    let attenuation = (policy.slow_beta as f64).powf(minutes as f64);
    serde_json::to_string(&serde_json::json!({
        "drawdown": memory.drawdown(), "start": input.start, "now": input.now,
        "fills": input.fills.len(), "unknown_prefix": !input.known_origin,
        "prefix_attenuation": attenuation, "same_time_order_from_canonical_manager": same_time,
    }))
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn calc_auto_unstuck_allowance_v9(
    balance: f64,
    loss_pct: f64,
    twel: f64,
    drawdown: f64,
) -> PyResult<f64> {
    hsl_decay::allowance_for(balance, loss_pct, twel, drawdown).map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn fixture() -> Value {
        json!({
            "snapshot": { "now":120000,"start":0,"balance":500.0,
                "balance_at":120000,"config_at":120000,"max_current_age_ms":1,
                "mode":"coin","pside":"long","symbol":"A",
                "global_fill_sequence":true,"fills_before_same_time_price":false,
                "pairs":[{"symbol":"A","position":{"size":0.0,"basis":0.0,
                    "mark":90.0,"multiplier":1.0,"inverse":false,"pside":"long"},
                    "position_at":120000,"mark_at":120000,"prices_at":120000,
                    "fills_started_at":120000,"fills_at":120000,"revisions":[0,0,0,0],
                    "prices":{"0":100.0,"60000":90.0,"120000":90.0},
                    "fills":[{"identity":"entry","timestamp":1,"delta":1.0,"price":100.0,
                        "realized":0.0,"fee":0.0,"sequence":0,"revision":0},
                        {"identity":"manual-close","timestamp":60001,"delta":-1.0,"price":90.0,
                        "realized":-10.0,"fee":1.0,"sequence":1,"revision":0}]}]},
            "slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,
            "lookback_span_minutes":3.0,"span":1.0,"threshold":0.095,"cooldown_ms":120000,
            "budget_observations":[
                {"timestamp":0,"order":-1,"balance":100.0,"slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false},
                {"timestamp":1,"order":0,"balance":100.0,"slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false},
                {"timestamp":60001,"order":0,"balance":91.0,"slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false},
                {"timestamp":120000,"order":9223372036854775807i64,"balance":500.0,"slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false}]
        })
    }

    #[test]
    fn final_rebate_green_keeps_pre_red_cooldown_and_event_budget_after_deposit() {
        let output = evaluate(serde_json::from_value(fixture()).unwrap()).unwrap();
        let risk = output.risk.unwrap();
        assert_eq!(risk.action, Action::Halted);
        assert!(!risk.red);
        assert_eq!(risk.cooldown_until, Some(180001));
        assert_eq!(risk.drawdown, 9.0);
        assert!(risk.raw < 0.095);
        let decision = output.decision.unwrap();
        assert_eq!(decision.flat_at, Some(60001));
        assert_eq!(decision.red_at, Some(60000));
        assert!(!output.reasons.contains("historical_budget_estimate"));
    }

    #[test]
    fn repeated_reconstruction_and_new_current_budget_do_not_reclassify_terminal() {
        let first = evaluate(serde_json::from_value(fixture()).unwrap()).unwrap();
        let mut changed = fixture();
        changed["snapshot"]["balance"] = json!(1000.0);
        changed["budget_observations"][3]["balance"] = json!(1000.0);
        let next = evaluate(serde_json::from_value(changed).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(first.decision).unwrap(),
            serde_json::to_value(next.decision).unwrap()
        );
    }

    #[test]
    fn old_restart_selector_and_ambiguous_budget_order_are_rejected() {
        let mut old = fixture();
        old["restart"] = json!("never");
        assert!(serde_json::from_value::<Input>(old).is_err());
        let mut bad = fixture();
        bad["budget_observations"][2] = bad["budget_observations"][1].clone();
        assert!(evaluate(serde_json::from_value(bad).unwrap()).is_err());
    }

    #[test]
    fn no_candle_or_fill_history_still_assesses_known_current_exposure() {
        let mut value = fixture();
        let pair = &mut value["snapshot"]["pairs"][0];
        pair["fills"] = json!([]);
        pair["prices"] = json!({});
        pair["position"]["size"] = json!(10.0);
        pair["position"]["basis"] = json!(100.0);
        value["span"] = json!(1000000.0);
        let output = evaluate(serde_json::from_value(value).unwrap()).unwrap();
        assert_eq!(output.decision.unwrap().action, Action::Panic);
        let risk = output.risk.unwrap();
        assert_eq!(risk.drawdown, 100.0);
        assert_eq!(risk.raw, risk.ema);
        assert!(output.reasons.contains("unknown_prefix_current_basis_seed"));
    }

    #[test]
    fn guessed_historical_basis_cannot_hide_observed_loss_with_new_fast_warmup() {
        let mut value = fixture();
        let pair = &mut value["snapshot"]["pairs"][0];
        pair["prices"] = json!({});
        pair["position"]["size"] = json!(10.0);
        pair["position"]["basis"] = json!(100.0);
        pair["fills"] = json!([{"identity":"damaged-quantity","timestamp":60000,
            "delta":null,"price":90.0,"realized":-200.0,"fee":-1.0,
            "sequence":0,"revision":0}]);
        value["span"] = json!(1000000.0);
        let output = evaluate(serde_json::from_value(value).unwrap()).unwrap();
        let risk = output.risk.unwrap();
        assert_eq!(risk.action, Action::Panic);
        assert!(risk.ema > 0.19);
        // Seed loss100 is repaid by the estimated historical UPNL transition;
        // known netloss201 then one decay and current UPNL loss100 are separate.
        assert_eq!(risk.drawdown, 200.5);
        assert_eq!(risk.raw, (200.5_f32 / 600.5_f32) as f64);
    }

    #[test]
    fn realized_memory_clips_each_ordered_fill_and_queries_do_not_net_cohorts() {
        let input = json!({"span_minutes":3,"start":0,"now":60000,"known_origin":true,
            "fills":[{"identity":"profit","timestamp":0,"net":10},
                     {"identity":"loss","timestamp":0,"net":-15}]})
        .to_string();
        let output: Value = serde_json::from_str(&realized_memory_v9(&input).unwrap()).unwrap();
        assert_eq!(output["drawdown"], json!(7.5));
        assert_eq!(output["unknown_prefix"], json!(false));
    }
}
