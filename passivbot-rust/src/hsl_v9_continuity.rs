//! Disposable forward cache over retained factual observations. The immutable
//! first snapshot is the bootstrap initializer; later quotes cannot replace it.
//! No serialized risk state, permission or cooldown is accepted from Python.
use super::{exact_upnl, scope_budget, BudgetFact, Input, Output, Recorder};
use crate::hsl_decay::{Event, Policy};
use crate::hsl_history::{self, Fill, Position, PositionSide};
use crate::hsl_prices::HslPriceGrid;
use crate::hsl_snapshot;
use crate::hsl_support::{budget, normalize_admitted};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

type PairKey = (String, u8);

fn input_value(input: &Input) -> Value {
    let snapshot = &input.snapshot;
    let pairs:Vec<_>=snapshot.pairs.iter().map(|p|json!({
        "symbol":p.symbol,"position":p.position,"position_at":p.position_at,"mark_at":p.mark_at,
        "fills_started_at":p.fills_started_at,"fills_at":p.fills_at,"prices_at":p.prices_at,
        "fills":p.fills,"prices":p.prices.iter().map(|(t,p)|(t.to_string(),*p)).collect::<BTreeMap<_,_>>(),
        "revisions":p.revisions,"fills_position_anchor":p.fills_position_anchor.as_ref().map(|a|json!({
            "position_at":a.position_at,"size":a.size,"basis":a.basis,"multiplier":a.multiplier,
            "inverse":a.inverse,"pside":a.pside,"revision":a.revision}))
    })).collect();
    let mode = match snapshot.mode {
        hsl_snapshot::Mode::Coin => "coin",
        hsl_snapshot::Mode::Pside => "pside",
        hsl_snapshot::Mode::Unified => "portfolio",
    };
    json!({"snapshot":{"flat_coin":snapshot.flat_coin.as_ref().map(|f|json!({
            "symbol":f.symbol,"pside":f.pside,"position_at":f.position_at,"fills_at":f.fills_at,"history_start":f.history_start})),
        "global_fill_sequence":snapshot.global_fill_sequence,"fills_before_same_time_price":snapshot.fills_before_same_time_price,
        "now":snapshot.now,"start":snapshot.start,"balance":snapshot.balance,"balance_at":snapshot.balance_at,
        "config_at":snapshot.config_at,"max_current_age_ms":snapshot.max_current_age_ms,
        "mode":mode,"symbol":snapshot.symbol,"pside":snapshot.pside,"pairs":pairs},
        "slots":input.slots,"scale_budget_with_excess_allowance":input.scale_budget_with_excess_allowance,
        "exposure_budget":input.exposure_budget,"lookback_span_minutes":input.lookback_span_minutes,
        "span":input.span,"threshold":input.threshold,"cooldown_ms":input.cooldown_ms,
        "budget_observations":input.budget_observations,"invalidated_price_symbols":input.invalidated_price_symbols})
}

pub(super) fn one_shot(input: Input) -> Result<Output, String> {
    HslV9Replay::default().consume(input_value(&input), input)
}
fn key(symbol: &str, side: PositionSide) -> PairKey {
    (symbol.to_owned(), u8::from(side == PositionSide::Short))
}
fn direction(p: &Position) -> f64 {
    if p.pside == PositionSide::Long {
        1.0
    } else {
        -1.0
    }
}
fn held(pairs: &BTreeMap<PairKey, Position>) -> bool {
    pairs.values().any(|p| p.size != 0.0)
}
fn upnl(pairs: &BTreeMap<PairKey, Position>) -> Result<f64, String> {
    exact_upnl(pairs.values().map(|p| p.pnl(p.size, p.basis, p.mark)))
}

#[derive(Clone, Debug, PartialEq)]
struct Step {
    timestamp: i64,
    budget: f64,
    upnl: f64,
    net: f64,
    exposed: bool,
    seed: bool,
}
impl Step {
    fn apply(&self, policy: &Policy, recorder: &mut Recorder) -> Result<(), String> {
        if self.seed {
            let result =
                recorder
                    .state
                    .seed_unknown(policy, self.timestamp, self.budget, self.upnl)?;
            recorder.record(&result);
            Ok(())
        } else {
            recorder.observe(
                policy,
                Event {
                    timestamp_ms: self.timestamp,
                    budget: self.budget,
                    upnl: self.upnl,
                    net_realized_pnl: self.net,
                    exposed: self.exposed,
                },
            )
        }
    }
}
fn push(
    steps: &mut Vec<Step>,
    timestamp: i64,
    budget: f64,
    pairs: &BTreeMap<PairKey, Position>,
    net: f64,
    seed: bool,
) -> Result<(), String> {
    let exposed = held(pairs);
    let was_exposed = steps.last().is_some_and(|s| s.exposed);
    if !was_exposed && !exposed && net != 0.0 {
        return Ok(());
    }
    steps.push(Step {
        timestamp,
        budget,
        upnl: upnl(pairs)?,
        net,
        exposed,
        seed,
    });
    Ok(())
}

fn push_cashflow(
    steps: &mut Vec<Step>,
    timestamp: i64,
    budget: f64,
    pairs: &BTreeMap<PairKey, Position>,
    net: f64,
    reasons: &mut BTreeSet<String>,
) -> Result<(), String> {
    if net != 0.0 && !held(pairs) && !steps.last().is_some_and(|s| s.exposed) {
        reasons.insert("flat_cashflow_without_exposure".into());
    }
    push(steps, timestamp, budget, pairs, net, false)
}

#[derive(Clone)]
struct PositionFact {
    position: Position,
    position_at: Option<i64>,
    /// Latest actual same-value read, not evidence between the endpoints.
    position_latest_at: Option<i64>,
    mark_at: Option<i64>,
}
#[derive(Clone)]
struct PairFacts {
    metadata: Position,
    initial_mark_at: i64,
    prices: BTreeMap<i64, f64>,
    fills: Vec<Fill>,
    /// Complete position observations, not replay-derived after-states.
    positions: BTreeMap<i64, PositionFact>,
}
#[derive(Clone)]
struct Ledger {
    anchor: Value,
    origin: i64,
    anchor_now: i64,
    last_now: i64,
    policy_key: Value,
    pairs: BTreeMap<PairKey, PairFacts>,
    budgets: BTreeMap<(i64, i64), BudgetFact>,
    global_sequence: bool,
    /// Only identities admitted at the immutable capture can precede its account read.
    anchor_fill_ids: BTreeMap<PairKey, BTreeSet<String>>,
}
impl Ledger {
    fn policy_key(value: &Value) -> Value {
        json!([
            value["snapshot"]["mode"],
            value["snapshot"]["symbol"],
            value["snapshot"]["pside"],
            value["lookback_span_minutes"],
            value["span"],
            value["threshold"],
            value["cooldown_ms"]
        ])
    }
    fn new(raw: Value, input: &Input) -> Self {
        Self {
            origin: input.snapshot.start,
            anchor_now: input.snapshot.now,
            last_now: input.snapshot.now,
            policy_key: Self::policy_key(&raw),
            anchor: raw,
            pairs: BTreeMap::new(),
            budgets: BTreeMap::new(),
            global_sequence: input.snapshot.global_fill_sequence,
            anchor_fill_ids: input
                .snapshot
                .pairs
                .iter()
                .map(|pair| {
                    (
                        key(&pair.symbol, pair.position.pside),
                        hsl_snapshot::causal_fills(pair, input.snapshot.now)
                            .iter()
                            .map(|fill| fill.identity.clone())
                            .collect(),
                    )
                })
                .collect(),
        }
    }
    fn merge(
        &mut self,
        raw: &Value,
        input: &Input,
        reasons: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        let new_policy = Self::policy_key(raw);
        if new_policy.as_array().unwrap()[..3] != self.policy_key.as_array().unwrap()[..3] {
            return Err("live HSL scope topology changed within one factual owner".into());
        }
        self.policy_key = new_policy;
        if input.snapshot.now < self.last_now {
            return Err("live HSL observation clock moved backwards".into());
        }
        if input.snapshot.start < self.origin {
            reasons.insert("history_before_fixed_bootstrap_origin_ignored".into());
        }
        let current: BTreeMap<_, _> = hsl_snapshot::select(&input.snapshot)?
            .into_iter()
            .map(|p| (key(&p.symbol, p.position.pside), p))
            .collect();
        // A canonical fill tape replaces its covered suffix. A failed fetch
        // does not alter that tape; the owner retains its existing facts.
        for (k, facts) in &mut self.pairs {
            let quarantined = current.get(k).map_or_else(BTreeSet::new, |pair| {
                hsl_snapshot::quarantined_fill_revisions(pair, input.snapshot.now)
            });
            let novel_quarantined: BTreeSet<_> = quarantined
                .iter()
                .copied()
                .filter(|revision| {
                    // Compare with the complete prior receipt, before replacing
                    // its covered suffix. An older repeated capture cannot
                    // invalidate an already admitted variant of this revision.
                    current[k]
                        .fills
                        .iter()
                        .filter(|fill| (fill.identity.as_str(), fill.revision) == *revision)
                        .any(|fill| !facts.fills.contains(fill))
                })
                .collect();
            let fill_end = current
                .get(k)
                .and_then(|pair| pair.fills_at)
                .unwrap_or(input.snapshot.now)
                .min(input.snapshot.now);
            facts.fills.retain(|f| {
                let revision = (f.identity.as_str(), f.revision);
                // A newly seen future variant quarantines its whole
                // revision, including any retained older member. A wholly
                // known repeated group keeps all members: deleting only
                // its covered member could resurrect a conflicting sibling.
                !novel_quarantined.contains(&revision)
                    && (quarantined.contains(&revision)
                        || f.timestamp < input.snapshot.start
                        || f.timestamp > fill_end)
            });
            if input.invalidated_price_symbols.contains(&k.0) {
                let price_end = current
                    .get(k)
                    .map_or(input.snapshot.now, |pair| pair.prices_at)
                    .min(input.snapshot.now);
                facts
                    .prices
                    .retain(|t, _| *t < input.snapshot.start || *t > price_end);
            }
            if !current.contains_key(k) {
                let mut flat = facts.metadata.clone();
                flat.size = 0.0;
                flat.basis = 0.0;
                facts.positions.insert(
                    input.snapshot.now,
                    PositionFact {
                        position: flat,
                        position_at: None,
                        position_latest_at: None,
                        mark_at: None,
                    },
                );
            }
        }
        for (k, pair) in current {
            let facts = self.pairs.entry(k).or_insert_with(|| PairFacts {
                metadata: pair.position.clone(),
                initial_mark_at: pair.mark_at,
                prices: BTreeMap::new(),
                fills: vec![],
                positions: BTreeMap::new(),
            });
            if facts.metadata.multiplier != pair.position.multiplier
                || facts.metadata.inverse != pair.position.inverse
            {
                return Err("live HSL contract metadata changed within retained history".into());
            }
            // The first supported mark is a factual birth anchor for a pair
            // discovered after bootstrap. Later quote previews cannot rebase it.
            // Both raw snapshot preparation and ledger ingestion quarantine a
            // future variant's entire revision before canonical selection.
            facts.fills.extend(
                hsl_snapshot::causal_fills(pair, input.snapshot.now)
                    .iter()
                    .filter(|f| f.timestamp >= self.origin)
                    .cloned(),
            );
            // Missing source observations do not delete supported old facts.
            // Explicit corrected prices replace the same event-time sample.
            facts.prices.extend(
                pair.prices
                    .range(..=input.snapshot.now.min(pair.prices_at))
                    .filter(|(t, _)| **t >= self.origin)
                    .map(|(&t, &p)| (t, p)),
            );
            // Retain the account source clock separately from quote/capture
            // clocks. The event tape values inventory using only marks which
            // were supported at its causal source event.
            let observed = input.snapshot.now;
            match facts.positions.get(&observed) {
                Some(previous)
                    if previous.position.size == pair.position.size
                        && previous.position.basis == pair.position.basis =>
                {
                    let previous = facts.positions.get_mut(&observed).unwrap();
                    previous.position_latest_at = Some(
                        previous
                            .position_latest_at
                            .map_or(pair.position_at, |at| at.max(pair.position_at)),
                    );
                    previous.position_at = Some(
                        previous
                            .position_at
                            .map_or(pair.position_at, |at| at.min(pair.position_at)),
                    );
                }
                _ => {
                    facts.positions.insert(
                        observed,
                        PositionFact {
                            position: pair.position.clone(),
                            position_at: Some(pair.position_at),
                            position_latest_at: Some(pair.position_at),
                            mark_at: Some(pair.mark_at),
                        },
                    );
                }
            }
        }
        for fact in &input.budget_observations {
            self.budgets
                .insert((fact.timestamp, fact.order), fact.clone());
        }
        self.last_now = input.snapshot.now;
        self.global_sequence &= input.snapshot.global_fill_sequence;
        Ok(())
    }
    fn bootstrap_fill(&self, pair: &PairKey, fill: &Fill) -> bool {
        if fill.timestamp != self.anchor_now {
            return fill.timestamp < self.anchor_now;
        }
        // A later receipt may share the initializer's exchange millisecond.
        // Keep the captured fill cohort as its factual phase boundary; never
        // insert a newly observed same-clock execution before that account read.
        self.anchor_fill_ids
            .get(pair)
            .is_some_and(|ids| ids.contains(&fill.identity))
    }

    fn bootstrap(&self) -> Result<Input, String> {
        let mut raw = self.anchor.clone();
        let original: BTreeMap<_, _> = raw["snapshot"]["pairs"]
            .as_array()
            .ok_or("invalid bootstrap pairs")?
            .iter()
            .map(|p| {
                (
                    (
                        p["symbol"].as_str().unwrap().to_owned(),
                        u8::from(p["position"]["pside"] == "short"),
                    ),
                    p.clone(),
                )
            })
            .collect();
        let mut pairs = vec![];
        for (k, facts) in &self.pairs {
            if !original.contains_key(k) && !facts.fills.iter().any(|f| self.bootstrap_fill(k, f)) {
                continue;
            }
            let mut pair = if let Some(p) = original.get(k) {
                p.clone()
            } else {
                let mut position = facts.metadata.clone();
                position.size = 0.0;
                position.basis = 0.0;
                json!({"symbol":k.0,"position":position,"position_at":self.anchor_now,
                    "mark_at":self.anchor_now,"fills_started_at":null,"fills_at":null,
                    "prices_at":self.anchor_now,"prices":{},"fills":[],"revisions":[0,0,0,0],"fills_position_anchor":null})
            };
            // Improve provenance only from an independently retained identical
            // actual position read; the raw numerical initializer stays fixed.
            let source: Position =
                serde_json::from_value(pair["position"].clone()).map_err(|e| e.to_string())?;
            let oldest = self.anchor_now
                - raw["snapshot"]["max_current_age_ms"]
                    .as_i64()
                    .ok_or("invalid anchor freshness")?;
            if let Some(at) = facts
                .positions
                .values()
                .filter(|f| {
                    f.position.size == source.size
                        && f.position.basis == source.basis
                        && f.position.multiplier == source.multiplier
                        && f.position.inverse == source.inverse
                })
                .filter_map(|f| f.position_at)
                .filter(|at| oldest <= *at && *at <= self.anchor_now)
                .min()
            {
                let initial = pair["position_at"]
                    .as_i64()
                    .ok_or("invalid anchor position clock")?;
                pair["position_at"] = json!(initial.min(at));
            }
            pair["prices"] = serde_json::to_value(
                facts
                    .prices
                    .range(self.origin..=self.anchor_now)
                    .map(|(&t, &p)| (t.to_string(), p))
                    .collect::<BTreeMap<_, _>>(),
            )
            .map_err(|e| e.to_string())?;
            pair["fills"] = serde_json::to_value(
                facts
                    .fills
                    .iter()
                    .filter(|f| self.bootstrap_fill(k, f))
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())?;
            pairs.push(pair);
        }
        raw["snapshot"]["pairs"] = Value::Array(pairs);
        if !raw["snapshot"]["pairs"].as_array().unwrap().is_empty() {
            raw["snapshot"].as_object_mut().unwrap().remove("flat_coin");
        }
        raw["budget_observations"] = serde_json::to_value(
            self.budgets
                .values()
                .filter(|f| f.timestamp <= self.anchor_now)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        serde_json::from_value(raw).map_err(|e| e.to_string())
    }
}

/// A replay cache owns only retained normalized facts and a disposable Rust
/// fold. clear_derived_cache rebuilds exactly from the same factual initializer.
#[pyclass(module = "passivbot_rust")]
#[derive(Clone, Default)]
pub struct HslV9Replay {
    ledger: Option<Ledger>,
    steps: Vec<Step>,
    recorder: Recorder,
    cache_valid: bool,
    rebuilds: usize,
    advances: usize,
}

fn attach_grids(input: &mut Input, grids: Vec<PyRef<'_, HslPriceGrid>>) -> Result<(), String> {
    if grids.len() != input.snapshot.pairs.len() {
        return Err("v9 continuity price grid count mismatch".into());
    }
    for (pair, grid) in input.snapshot.pairs.iter_mut().zip(grids) {
        if grid.start != input.snapshot.start
            || grid.end != input.snapshot.now
            || pair.prices_at != grid.end
            || !pair.prices.is_empty()
        {
            return Err("v9 continuity price grid observation mismatch".into());
        }
        pair.prices = grid.prices.clone();
    }
    Ok(())
}

fn bootstrap_steps(
    ledger: &Ledger,
    reasons: &mut BTreeSet<String>,
) -> Result<(Vec<Step>, BTreeMap<PairKey, Position>), TapeError> {
    let mut input = ledger.bootstrap()?;
    normalize_admitted(&mut input.snapshot, reasons)?;
    let prepared = hsl_snapshot::prepare_admitted(&input.snapshot)?;
    reasons.extend(prepared.reasons.iter().cloned());
    let selected = hsl_snapshot::select(&input.snapshot)?;
    let keys: Vec<_> = selected
        .iter()
        .map(|p| key(&p.symbol, p.position.pside))
        .collect();
    // Reconstruction terminates at each original position source observation.
    // A later admitted fill between that source and the capture is forward
    // economics after that inventory, even though it precedes anchor_now.
    let fill_tapes: Vec<Vec<Fill>> = selected
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            prepared.pairs[i]
                .history
                .events
                .iter()
                .map(|event| event.fill.clone())
                .chain(
                    hsl_history::canonical_fills(
                        &pair.fills,
                        input.snapshot.start,
                        input.snapshot.now,
                        pair.position.pside,
                        reasons,
                    )
                    .into_iter()
                    .filter(|fill| {
                        fill.timestamp > pair.position_at && ledger.bootstrap_fill(&keys[i], fill)
                    }),
                )
                .collect()
        })
        .collect();
    let has_source_tail = fill_tapes
        .iter()
        .zip(&prepared.pairs)
        .any(|(fills, pair)| fills.len() > pair.history.events.len());
    let mut positions = BTreeMap::new();
    for (i, p) in selected.iter().enumerate() {
        let mut position = p.position.clone();
        position.size = prepared.pairs[i].history.opening_size;
        position.basis = prepared.pairs[i].history.opening_basis;
        position.mark = p
            .prices
            .range(..=input.snapshot.start)
            .next_back()
            .map_or(p.position.mark, |(_, v)| *v);
        positions.insert(keys[i].clone(), position);
    }
    let mut steps = vec![];
    if held(&positions) {
        reasons.insert("unknown_prefix_current_basis_seed".into());
        reasons.insert("continuity_unknown_prefix_anchor_retained".into());
        let observed = selected
            .iter()
            .map(|p| (key(&p.symbol, p.position.pside), p.position.clone()))
            .collect();
        let b = budget(
            input.snapshot.balance,
            &input.snapshot.mode,
            input.slots,
            input.scale_budget_with_excess_allowance,
            input.exposure_budget.as_ref(),
        )?;
        push(&mut steps, input.snapshot.start, b, &observed, 0.0, true)?;
    }
    let mut timeline = Vec::<(i64, u8, i64, usize, usize)>::new();
    for (i, p) in selected.iter().enumerate() {
        for (&t, _) in &p.prices {
            if t.rem_euclid(60_000) == 0 {
                timeline.push((t, 0, 0, i, 0));
            }
        }
    }
    let mut cohorts = BTreeMap::<i64, Vec<(usize, usize)>>::new();
    for (i, fills) in fill_tapes.iter().enumerate() {
        for (j, fill) in fills.iter().enumerate() {
            cohorts.entry(fill.timestamp).or_default().push((i, j));
        }
        if has_source_tail {
            // The captured inventory belongs to this original source clock,
            // before the appended execution tail. It is not a new observation
            // at the later capture and cannot borrow that capture's quote.
            timeline.push((
                selected[i].position_at.max(input.snapshot.start),
                2,
                0,
                i,
                0,
            ));
            // Only a still-unvalued unknown episode may consume this quote,
            // once both the original inventory and its actual mark are known.
            timeline.push((
                selected[i]
                    .mark_at
                    .max(selected[i].position_at)
                    .max(input.snapshot.start),
                4,
                0,
                i,
                0,
            ));
        }
    }
    for (t, cohort) in cohorts {
        let sequences: BTreeSet<_> = cohort
            .iter()
            .filter_map(|(i, j)| fill_tapes[*i][*j].sequence)
            .collect();
        let one_pair = cohort.iter().all(|(i, _)| *i == cohort[0].0);
        let exact = (one_pair || ledger.global_sequence) && sequences.len() == cohort.len();
        if cohort.len() > 1 && !exact {
            reasons.insert("ambiguous_same_time_fill_order".into());
        }
        for (i, j) in cohort {
            timeline.push((
                t,
                1,
                if exact {
                    fill_tapes[i][j].sequence.unwrap()
                } else {
                    0
                },
                i,
                j,
            ));
        }
    }
    if !selected.is_empty() {
        for fact in input
            .budget_observations
            .iter()
            .filter(|f| !f.estimated && f.timestamp >= input.snapshot.start)
        {
            timeline.push((fact.timestamp, 3, 0, 0, 0));
        }
    }
    timeline.sort();
    let mut cursor = 0;
    let mut fill_time = None;
    let mut ordinal = -1;
    let mut marks = BTreeSet::new();
    let mut source_valuation = false;
    let mut pending_unknown = false;
    while cursor < timeline.len() {
        let (t, phase, _, i, j) = timeline[cursor];
        if phase == 0 {
            let end = cursor + timeline[cursor..].partition_point(|e| e.0 == t && e.1 == 0);
            for &(_, _, _, i, _) in &timeline[cursor..end] {
                // The historical estimator can backfill an absent pair tape
                // from its capture quote. That is not causal mark support for
                // an original source inventory followed by a real execution.
                if !has_source_tail
                    || ledger.pairs[&keys[i]]
                        .prices
                        .range(..=t)
                        .next_back()
                        .is_some()
                {
                    positions.get_mut(&keys[i]).unwrap().mark = selected[i].prices[&t];
                    marks.insert(keys[i].clone());
                }
            }
            let b = scope_budget(&input, t, -1, reasons)?;
            if source_valuation {
                if positions
                    .iter()
                    .all(|(k, p)| p.size == 0.0 || marks.contains(k))
                {
                    causal_valuation(
                        &mut steps,
                        t,
                        b,
                        &positions,
                        &marks,
                        &mut pending_unknown,
                        reasons,
                    )?;
                }
            } else {
                push(&mut steps, t, b, &positions, 0.0, false)?;
            }
            cursor = end;
        } else if phase == 2 {
            let end = cursor + timeline[cursor..].partition_point(|e| e.0 == t && e.1 == phase);
            let was_exposed = steps.last().is_some_and(|s| s.exposed);
            for &(_, _, _, i, _) in &timeline[cursor..end] {
                let p = positions.get_mut(&keys[i]).unwrap();
                let observed = &selected[i].position;
                if p.size != observed.size {
                    reasons.insert("current_quantity_reconciliation".into());
                }
                if p.size != 0.0 && p.basis != observed.basis {
                    reasons.insert("current_basis_reconciliation".into());
                }
                p.size = observed.size;
                p.basis = observed.basis;
            }
            source_valuation = true;
            pending_unknown |= !was_exposed && held(&positions);
            if positions
                .iter()
                .all(|(k, p)| p.size == 0.0 || marks.contains(k))
            {
                let b = scope_budget(&input, t, i64::MAX, reasons)?;
                causal_valuation(
                    &mut steps,
                    t,
                    b,
                    &positions,
                    &marks,
                    &mut pending_unknown,
                    reasons,
                )?;
            }
            cursor = end;
        } else if phase == 3 {
            let end = cursor + timeline[cursor..].partition_point(|e| e.0 == t && e.1 == phase);
            let b = scope_budget(&input, t, i64::MAX, reasons)?;
            if source_valuation {
                if positions
                    .iter()
                    .all(|(k, p)| p.size == 0.0 || marks.contains(k))
                {
                    causal_valuation(
                        &mut steps,
                        t,
                        b,
                        &positions,
                        &marks,
                        &mut pending_unknown,
                        reasons,
                    )?;
                }
            } else {
                push(&mut steps, t, b, &positions, 0.0, false)?;
            }
            cursor = end;
        } else if phase == 4 {
            let end = cursor + timeline[cursor..].partition_point(|e| e.0 == t && e.1 == phase);
            if pending_unknown {
                for &(_, _, _, i, _) in &timeline[cursor..end] {
                    let p = positions.get_mut(&keys[i]).unwrap();
                    if p.size != 0.0 && !marks.contains(&keys[i]) {
                        p.mark = selected[i].position.mark;
                        marks.insert(keys[i].clone());
                    }
                }
                if positions
                    .iter()
                    .all(|(k, p)| p.size == 0.0 || marks.contains(k))
                {
                    let b = scope_budget(&input, t, i64::MAX, reasons)?;
                    causal_valuation(
                        &mut steps,
                        t,
                        b,
                        &positions,
                        &marks,
                        &mut pending_unknown,
                        reasons,
                    )?;
                }
            }
            cursor = end;
        } else {
            if fill_time != Some(t) {
                fill_time = Some(t);
                ordinal = -1;
            }
            let pre = scope_budget(&input, t, ordinal, reasons)?;
            ordinal += 1;
            let post = scope_budget(&input, t, ordinal, reasons)?;
            let fill = &fill_tapes[i][j];
            if let Some(price) = fill.price.filter(|p| p.is_finite() && *p > 0.0) {
                for (k, p) in &mut positions {
                    if k.0 == keys[i].0 {
                        p.mark = price;
                        marks.insert(k.clone());
                    }
                }
            }
            if source_valuation {
                causal_valuation(
                    &mut steps,
                    t,
                    pre,
                    &positions,
                    &marks,
                    &mut pending_unknown,
                    reasons,
                )?;
            } else {
                push(&mut steps, t, pre, &positions, 0.0, false)?;
            }
            let p = positions.get_mut(&keys[i]).unwrap();
            let net = if let Some(e) = prepared.pairs[i].history.events.get(j) {
                p.size = direction(p) * e.after;
                p.basis = e.basis;
                if j + 1 == prepared.pairs[i].history.events.len() {
                    if let Some(r) = &prepared.pairs[i].history.reconciliation {
                        if r.applied_at == t && r.after == 0.0 {
                            p.size = 0.0;
                            p.basis = 0.0;
                        }
                    }
                }
                exact_upnl([e.gross_realized, e.fee])?
            } else {
                forward_fill(p, fill, reasons)?
            };
            if !input
                .budget_observations
                .iter()
                .any(|f| f.timestamp == t && f.order == ordinal && !f.estimated)
            {
                reasons.insert("historical_fill_budget_estimate".into());
            }
            push_cashflow(&mut steps, t, post, &positions, net, reasons)?;
            cursor += 1;
        }
    }
    let before = positions.clone();
    for (i, p) in selected.iter().enumerate() {
        if fill_tapes[i].len() > prepared.pairs[i].history.events.len() {
            // This source observation predates a supported execution. Never
            // replay it again at the later capture and resurrect old inventory.
            continue;
        }
        let value = positions.get_mut(&keys[i]).unwrap();
        value.size = p.position.size;
        value.basis = p.position.basis;
    }
    let changed = positions
        .iter()
        .any(|(k, p)| before[k].size != p.size || before[k].basis != p.basis);
    if changed || steps.is_empty() {
        let b = scope_budget(&input, input.snapshot.now, i64::MAX, reasons)?;
        if !steps.last().is_some_and(|s| s.exposed) && held(&positions) {
            reasons.insert("unknown_prefix_current_basis_seed".into());
            let observed = selected
                .iter()
                .map(|p| (key(&p.symbol, p.position.pside), p.position.clone()))
                .collect();
            push(&mut steps, input.snapshot.now, b, &observed, 0.0, true)?;
        }
        push(&mut steps, input.snapshot.now, b, &positions, 0.0, false)?;
    }
    if source_valuation
        && ((pending_unknown && held(&positions))
            || positions
                .iter()
                .any(|(k, p)| p.size != 0.0 && !marks.contains(k)))
    {
        return Err(TapeError::Unavailable("causal_position_mark_unavailable"));
    }
    Ok((steps, positions))
}

fn forward_fill(
    p: &mut Position,
    fill: &Fill,
    reasons: &mut BTreeSet<String>,
) -> Result<f64, String> {
    let dir = direction(p);
    let delta = fill.delta.filter(|v| v.is_finite()).unwrap_or_else(|| {
        reasons.insert("invalid_quantity".into());
        0.0
    }) * dir;
    let before = p.size.abs();
    let price = fill
        .price
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or_else(|| {
            reasons.insert("estimated_fill_price".into());
            if p.basis > 0.0 {
                p.basis
            } else {
                p.mark
            }
        });
    let gross = fill.realized.filter(|v| v.is_finite()).unwrap_or_else(|| {
        reasons.insert("estimated_realized_pnl".into());
        if delta < 0.0 {
            p.pnl(dir * before.min(-delta), p.basis, price)
        } else {
            0.0
        }
    });
    let fee = fill.fee.filter(|v| v.is_finite()).unwrap_or_else(|| {
        reasons.insert("unknown_fee".into());
        0.0
    });
    let mut after = exact_upnl([before, delta])?;
    let scale = before.max(delta.abs());
    if after != 0.0
        && after.abs() <= 8.0 * f64::EPSILON * scale
        && p.quantity_step.is_none_or(|q| after.abs() < q / 2.0)
    {
        reasons.insert("quantity_roundoff".into());
        after = 0.0;
    }
    if after < 0.0 {
        reasons.insert("local_quantity_reconciliation".into());
        after = 0.0;
    }
    if delta > 0.0 {
        let total = before + delta;
        let weight = delta / total;
        p.basis = if p.inverse && before > 0.0 && p.basis > 0.0 {
            1.0 / ((1.0 - weight) / p.basis + weight / price)
        } else {
            (1.0 - weight) * p.basis + weight * price
        };
    }
    p.size = dir * after;
    if after == 0.0 {
        p.basis = 0.0;
    }
    if !p.size.is_finite() || !p.basis.is_finite() {
        return Err("nonfinite v9 forward position accounting".into());
    }
    exact_upnl([gross, fee])
}

#[derive(Debug)]
enum TapeError {
    Invalid(String),
    Unavailable(&'static str),
}
impl From<String> for TapeError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

fn causal_valuation(
    steps: &mut Vec<Step>,
    timestamp: i64,
    budget: f64,
    positions: &BTreeMap<PairKey, Position>,
    marks: &BTreeSet<PairKey>,
    pending_unknown: &mut bool,
    reasons: &mut BTreeSet<String>,
) -> Result<(), TapeError> {
    if positions
        .iter()
        .any(|(k, p)| p.size != 0.0 && !marks.contains(k))
    {
        return Err(TapeError::Unavailable("causal_position_mark_unavailable"));
    }
    if *pending_unknown && held(positions) {
        reasons.insert("unknown_prefix_causal_mark_seed".into());
        push(steps, timestamp, budget, positions, 0.0, true)?;
        *pending_unknown = false;
    } else {
        push(steps, timestamp, budget, positions, 0.0, false)?;
    }
    Ok(())
}

fn event_tape(
    ledger: &Ledger,
    reasons: &mut BTreeSet<String>,
) -> Result<(Vec<Step>, BTreeMap<PairKey, Position>), TapeError> {
    let (mut steps, mut positions) = bootstrap_steps(ledger, reasons)?;
    let mut marks: BTreeSet<_> = positions.keys().cloned().collect();
    let mut pending_unknown = false;
    let mut input = ledger.bootstrap()?;
    input.budget_observations = ledger.budgets.values().cloned().collect();
    let mut fills = BTreeMap::<PairKey, Vec<Fill>>::new();
    let mut account = BTreeMap::<PairKey, Vec<&PositionFact>>::new();
    // mark, pre-fill account, execution, ambiguous/post-fill account, budget,
    // first supported unknown-episode quote. Source clocks own economic time.
    let mut timeline = Vec::<(i64, u8, i64, PairKey, usize)>::new();
    let mut cohorts = BTreeMap::<i64, Vec<(PairKey, usize)>>::new();
    for (k, facts) in &ledger.pairs {
        positions.entry(k.clone()).or_insert_with(|| {
            let mut p = facts.metadata.clone();
            p.size = 0.0;
            p.basis = 0.0;
            p
        });
        if !marks.contains(k) {
            // A newly selected pair may already have a causal completed mark
            // at/before the immutable anchor, or share a marked instrument.
            // Its later current quote is never used for this lookup.
            let supported = facts
                .prices
                .range(..=ledger.anchor_now)
                .next_back()
                .map(|(_, price)| *price)
                .or_else(|| {
                    positions
                        .iter()
                        .find(|(pair, _)| pair.0 == k.0 && marks.contains(*pair))
                        .map(|(_, p)| p.mark)
                });
            if let Some(price) = supported {
                positions.get_mut(k).unwrap().mark = price;
                marks.insert(k.clone());
            }
        }
        let canonical = hsl_history::canonical_fills(
            &facts.fills,
            ledger.anchor_now,
            ledger.last_now,
            facts.metadata.pside,
            reasons,
        )
        .into_iter()
        .filter(|f| !ledger.bootstrap_fill(k, f))
        .collect::<Vec<_>>();
        for (i, f) in canonical.iter().enumerate() {
            if f.timestamp == ledger.anchor_now {
                reasons.insert("ambiguous_bootstrap_fill_phase".into());
            }
            cohorts.entry(f.timestamp).or_default().push((k.clone(), i));
        }
        fills.insert(k.clone(), canonical);
        for (&t, _) in facts.prices.range((
            std::ops::Bound::Excluded(ledger.anchor_now),
            std::ops::Bound::Included(ledger.last_now),
        )) {
            if t.rem_euclid(60_000) == 0 {
                timeline.push((t, 0, 0, k.clone(), 0));
            }
        }
        if !marks.contains(k) {
            timeline.push((
                facts.initial_mark_at.max(ledger.anchor_now),
                5,
                0,
                k.clone(),
                0,
            ));
        }
        let initial = ledger.anchor["snapshot"]["pairs"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|p| {
                p["symbol"].as_str() == Some(k.0.as_str())
                    && u8::from(p["position"]["pside"] == "short") == k.1
            });
        let mut previous = initial.map_or((0.0, 0.0, ledger.anchor_now), |p| {
            (
                p["position"]["size"].as_f64().unwrap(),
                p["position"]["basis"].as_f64().unwrap(),
                p["position_at"].as_i64().unwrap(),
            )
        });
        let mut endpoints = vec![];
        for (&captured, fact) in facts.positions.range((
            std::ops::Bound::Included(ledger.anchor_now),
            std::ops::Bound::Included(ledger.last_now),
        )) {
            let mut sources = vec![
                fact.position_at.unwrap_or(captured),
                fact.position_latest_at.unwrap_or(captured),
            ];
            sources.sort_unstable();
            sources.dedup();
            for source in sources {
                endpoints.push((source, captured, fact));
            }
        }
        // Corrected source facts can arrive in another capture order. Compare
        // inventory and fill crossings in causal source order, retaining the
        // original capture clock only as same-source receipt provenance.
        endpoints.sort_by_key(|(source, captured, _)| (*source, *captured));
        let mut observations = vec![];
        for (source, _captured, fact) in endpoints {
            let changed = fact.position.size != previous.0 || fact.position.basis != previous.1;
            let after_execution = facts.fills.iter().any(|f| {
                f.delta.is_some_and(|d| d != 0.0)
                    && previous.2 < f.timestamp
                    && f.timestamp <= source
            });
            if changed || after_execution {
                let at = source.max(ledger.anchor_now);
                let equal_execution = fills[k].iter().any(|f| f.timestamp == at);
                let phase = if equal_execution && source == at {
                    3
                } else {
                    1
                };
                if equal_execution && source == at {
                    reasons.insert("ambiguous_position_fill_phase".into());
                }
                // Keep source ordering even where several older source clocks
                // are clamped onto the immutable initializer boundary. Indices
                // retain capture order for actual equal-source observations.
                timeline.push((at, phase, source, k.clone(), observations.len()));
                observations.push(fact);
            }
            previous = (
                fact.position.size,
                fact.position.basis,
                previous.2.max(source),
            );
        }
        account.insert(k.clone(), observations);
    }
    for (t, cohort) in cohorts {
        let one_pair = cohort.iter().all(|(k, _)| k == &cohort[0].0);
        let sequences: BTreeSet<_> = cohort
            .iter()
            .filter_map(|(k, i)| fills[k][*i].sequence)
            .collect();
        let exact = (one_pair || ledger.global_sequence) && sequences.len() == cohort.len();
        if cohort.len() > 1 && !exact {
            reasons.insert("ambiguous_same_time_fill_order".into());
        }
        for (k, i) in cohort {
            let sequence = if exact {
                fills[&k][i].sequence.unwrap()
            } else {
                0
            };
            timeline.push((t, 2, sequence, k, i));
        }
    }
    if let Some(k) = positions.keys().next() {
        for fact in ledger
            .budgets
            .values()
            .filter(|f| !f.estimated && f.timestamp > ledger.anchor_now)
        {
            if fact.observed_at.is_some()
                && fact.order == i64::MAX
                && fills
                    .values()
                    .flatten()
                    .any(|f| f.timestamp == fact.timestamp)
            {
                reasons.insert("ambiguous_budget_fill_phase".into());
            }
            timeline.push((fact.timestamp, 4, 0, k.clone(), 0));
        }
    }
    timeline.sort();
    let mut cursor = 0;
    let mut fill_time = None;
    let mut ordinal = -1;
    while cursor < timeline.len() {
        let (t, phase, _, ref k, i) = timeline[cursor];
        if phase == 2 {
            if fill_time != Some(t) {
                fill_time = Some(t);
                ordinal = -1;
            }
            let pre = scope_budget(&input, t, ordinal, reasons)?;
            ordinal += 1;
            let post = scope_budget(&input, t, ordinal, reasons)?;
            let f = &fills[k][i];
            if let Some(price) = f.price.filter(|v| v.is_finite() && *v > 0.0) {
                for (pair, p) in &mut positions {
                    if pair.0 == k.0 {
                        p.mark = price;
                        marks.insert(pair.clone());
                    }
                }
            }
            causal_valuation(
                &mut steps,
                t,
                pre,
                &positions,
                &marks,
                &mut pending_unknown,
                reasons,
            )?;
            let net = forward_fill(positions.get_mut(k).unwrap(), f, reasons)?;
            if !input
                .budget_observations
                .iter()
                .any(|f| f.timestamp == t && f.order == ordinal && !f.estimated)
            {
                reasons.insert("historical_fill_budget_estimate".into());
            }
            if positions
                .iter()
                .any(|(k, p)| p.size != 0.0 && !marks.contains(k))
            {
                return Err(TapeError::Unavailable("causal_position_mark_unavailable"));
            }
            push_cashflow(&mut steps, t, post, &positions, net, reasons)?;
            cursor += 1;
        } else {
            let end = cursor + timeline[cursor..].partition_point(|e| e.0 == t && e.1 == phase);
            let core_exposed = steps.last().is_some_and(|s| s.exposed);
            let mut changed = phase == 0 || phase == 4;
            for (_, _, _, pair, index) in &timeline[cursor..end] {
                if phase == 4 {
                    continue;
                }
                let p = positions.get_mut(pair).unwrap();
                if phase == 0 {
                    p.mark = ledger.pairs[pair].prices[&t];
                    marks.insert(pair.clone());
                } else if phase == 5 {
                    // A current quote initializes an otherwise unpriceable new
                    // unknown episode at its own observation clock only.
                    if pending_unknown && !marks.contains(pair) && p.size != 0.0 {
                        p.mark = ledger.pairs[pair].metadata.mark;
                        marks.insert(pair.clone());
                        changed = true;
                    }
                } else {
                    let actual = &account[pair][*index].position;
                    changed |= p.size != actual.size || p.basis != actual.basis;
                    if p.size != actual.size {
                        reasons.insert("current_quantity_reconciliation".into());
                    }
                    if p.size != 0.0 && p.basis != actual.basis {
                        reasons.insert("current_basis_reconciliation".into());
                    }
                    p.size = actual.size;
                    p.basis = actual.basis;
                }
            }
            if changed {
                let complete = positions
                    .iter()
                    .all(|(k, p)| p.size == 0.0 || marks.contains(k));
                if !core_exposed && held(&positions) {
                    pending_unknown = true;
                }
                if !complete && pending_unknown {
                    cursor = end;
                    continue;
                }
                let b = scope_budget(&input, t, if phase <= 1 { -1 } else { i64::MAX }, reasons)?;
                causal_valuation(
                    &mut steps,
                    t,
                    b,
                    &positions,
                    &marks,
                    &mut pending_unknown,
                    reasons,
                )?;
            }
            cursor = end;
        }
    }
    if pending_unknown {
        return Err(TapeError::Unavailable("causal_position_mark_unavailable"));
    }
    Ok((steps, positions))
}

impl HslV9Replay {
    fn evaluate_inner(&mut self, raw: Value, input: Input) -> Result<Output, String> {
        let policy = Policy::new(
            input.lookback_span_minutes,
            input.span,
            input.threshold,
            input.cooldown_ms,
        )?;
        if input
            .snapshot
            .now
            .checked_sub(input.snapshot.start)
            .is_none_or(|v| !(0..=90 * 86_400_000).contains(&v))
        {
            return Err("invalid v9 live continuity acquisition interval".into());
        }
        if input
            .budget_observations
            .windows(2)
            .any(|w| (w[0].timestamp, w[0].order) >= (w[1].timestamp, w[1].order))
            || input.budget_observations.iter().any(|f| {
                f.timestamp < 0
                    || f.timestamp > input.snapshot.now
                    || f.observed_at
                        .is_some_and(|at| at < f.timestamp || at > input.snapshot.now)
            })
        {
            return Err("unordered or future v9 continuity budget observations".into());
        }
        let mut reasons = BTreeSet::new();
        let prepared = hsl_snapshot::prepare(&input.snapshot)?;
        reasons.extend(prepared.reasons);
        if matches!(input.snapshot.mode, hsl_snapshot::Mode::Coin) && input.slots == 0 {
            if hsl_snapshot::select(&input.snapshot)?
                .iter()
                .any(|pair| pair.position.size != 0.0)
            {
                return Err("held enabled HSL scope has zero configured slots".into());
            }
            reasons.insert("inactive_scope".into());
            return Ok(Output {
                unavailable_reason: None,
                decision: None,
                risk: None,
                events: vec![],
                reasons,
                observations: 0,
                episodes: 0,
            });
        }
        for f in &input.budget_observations {
            budget(
                f.balance,
                &input.snapshot.mode,
                f.slots,
                f.scale_budget_with_excess_allowance,
                f.exposure_budget.as_ref(),
            )?;
        }
        let initialized = self.ledger.is_none();
        if initialized {
            self.ledger = Some(Ledger::new(raw.clone(), &input));
        }
        let ledger = self.ledger.as_mut().unwrap();
        let policy_changed = ledger.policy_key != Ledger::policy_key(&raw);
        ledger.merge(&raw, &input, &mut reasons)?;
        let (steps, terminal) = match event_tape(ledger, &mut reasons) {
            Ok(steps) => steps,
            Err(TapeError::Invalid(error)) => return Err(error),
            Err(TapeError::Unavailable(reason)) => {
                self.cache_valid = false;
                return Ok(Output {
                    unavailable_reason: Some(reason.into()),
                    decision: None,
                    risk: None,
                    events: vec![],
                    reasons,
                    observations: 0,
                    episodes: 0,
                });
            }
        };
        let selected = hsl_snapshot::select(&input.snapshot)?;
        let current: BTreeMap<_, _> = selected
            .iter()
            .map(|p| (key(&p.symbol, p.position.pside), p.position.clone()))
            .collect();
        // A partial execution can leave both snapshots exposed while changing
        // their UPNL. A quote preview cannot apply the older account quantity
        // after the newer economic fill; that would debit the same loss twice.
        // Exposure mismatches retain the existing atomic preview validation.
        let stale_quantity = held(&current) == held(&terminal)
            && selected.iter().any(|pair| {
                let k = key(&pair.symbol, pair.position.pside);
                terminal.get(&k).is_some_and(|p| {
                    (p.size != pair.position.size || p.basis != pair.position.basis)
                        && ledger.pairs[&k].fills.iter().any(|f| {
                            f.delta.is_some_and(|d| d != 0.0)
                                && pair.position_at < f.timestamp
                                && f.timestamp <= input.snapshot.now
                        })
                })
            });
        if stale_quantity {
            self.cache_valid = false;
            return Ok(Output {
                unavailable_reason: Some("causal_position_snapshot_precedes_fill".into()),
                decision: None,
                risk: None,
                events: vec![],
                reasons,
                observations: 0,
                episodes: 0,
            });
        }
        let append = self.cache_valid
            && !policy_changed
            && steps.len() >= self.steps.len()
            && steps[..self.steps.len()] == self.steps;
        let first = if append { self.steps.len() } else { 0 };
        if !append {
            self.recorder = Recorder::default();
            self.rebuilds += 1;
        }
        for step in &steps[first..] {
            step.apply(&policy, &mut self.recorder)?;
            self.advances += 1;
        }
        self.steps = steps;
        self.cache_valid = true;
        if input.snapshot.now - ledger.origin > 90 * 86_400_000 {
            reasons.insert("continuity_retained_beyond_bootstrap_window".into());
        }
        let current_budget = budget(
            input.snapshot.balance,
            &input.snapshot.mode,
            input.slots,
            input.scale_budget_with_excess_allowance,
            input.exposure_budget.as_ref(),
        )?;
        let preview = self.recorder.state.preview(
            &policy,
            Event {
                timestamp_ms: input.snapshot.now,
                budget: current_budget,
                upnl: upnl(&current)?,
                net_realized_pnl: 0.0,
                exposed: held(&current),
            },
        )?;
        let mut report = self.recorder.clone();
        report.record(&preview);
        let (red_at, flat_at) = if preview.action == crate::hsl_intent::Action::Normal {
            (None, None)
        } else {
            (report.red_at, report.flat_at)
        };
        let decision = crate::hsl_intent::Decision {
            timestamp: input.snapshot.now,
            action: preview.action,
            red_at,
            flat_at,
            reason: if preview.action == crate::hsl_intent::Action::Panic {
                "drawdown"
            } else if preview.action == crate::hsl_intent::Action::Halted {
                "stop_flattened"
            } else {
                "green"
            },
            raw: preview.raw,
            ema: preview.ema,
            numeric_range_approximation: false,
        };
        Ok(Output {
            unavailable_reason: None,
            decision: Some(decision),
            risk: Some(preview),
            events: report.events,
            reasons,
            observations: report.observations,
            episodes: report.episodes,
        })
    }
    fn consume(&mut self, raw: Value, input: Input) -> Result<Output, String> {
        // Producer failures leave both facts and the derived cache untouched.
        let mut next = self.clone();
        let output = next.evaluate_inner(raw, input)?;
        *self = next;
        Ok(output)
    }
}

#[pymethods]
impl HslV9Replay {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn evaluate(&mut self, input_json: &str) -> PyResult<String> {
        let raw: Value =
            serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let input: Input = serde_json::from_value(raw.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let output = self.consume(raw, input).map_err(PyValueError::new_err)?;
        serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn evaluate_grids(
        &mut self,
        input_json: &str,
        grids: Vec<PyRef<'_, HslPriceGrid>>,
    ) -> PyResult<String> {
        let raw: Value =
            serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut input: Input = serde_json::from_value(raw.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        attach_grids(&mut input, grids).map_err(PyValueError::new_err)?;
        let output = self.consume(raw, input).map_err(PyValueError::new_err)?;
        serde_json::to_string(&output).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn clear_derived_cache(&mut self) {
        self.cache_valid = false;
    }

    fn cache_info(&self) -> BTreeMap<String, usize> {
        BTreeMap::from([
            ("rebuilds".into(), self.rebuilds),
            ("advanced_events".into(), self.advances),
            ("committed_events".into(), self.steps.len()),
            (
                "retained_pairs".into(),
                self.ledger.as_ref().map_or(0, |l| l.pairs.len()),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(now: i64, start: i64, mark: f64) -> Value {
        json!({"snapshot":{"now":now,"start":start,"balance":1000.0,"balance_at":now,
            "config_at":now,"max_current_age_ms":60000,"mode":"coin","pside":"long","symbol":"A",
            "pairs":[{"symbol":"A","position":{"size":1.0,"basis":100.0,"mark":mark,
                "multiplier":1.0,"inverse":false,"pside":"long"},"position_at":now,"mark_at":now,
                "fills_started_at":null,"fills_at":null,"prices_at":now,"prices":{},"fills":[],
                "revisions":[0,0,0,0],"fills_position_anchor":null}]},
            "slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,
            "lookback_span_minutes":43200.0,"span":1000000.0,"threshold":0.005,"cooldown_ms":60000,
            "budget_observations":[{"timestamp":0,"order":-1,"balance":1000.0,"slots":1,
                "scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":true}]})
    }
    fn run(replay: &mut HslV9Replay, value: Value) -> Output {
        replay
            .consume(value.clone(), serde_json::from_value(value).unwrap())
            .unwrap()
    }
    fn risk(value: &Output) -> Value {
        serde_json::to_value(&value.risk).unwrap()
    }

    #[test]
    fn quotes_replace_preview_without_reseeding_or_becoming_samples() {
        let mut replay = HslV9Replay::default();
        let first = run(&mut replay, facts(60001, 0, 90.0));
        assert_eq!(first.risk.unwrap().action, crate::hsl_intent::Action::Panic);
        let before = replay.advances;
        let green = run(&mut replay, facts(60002, 0, 100.0));
        assert_eq!(
            green.risk.unwrap().action,
            crate::hsl_intent::Action::Normal
        );
        let red = run(&mut replay, facts(60003, 0, 90.0));
        assert_eq!(red.risk.unwrap().action, crate::hsl_intent::Action::Panic);
        assert_eq!(replay.advances, before);
        assert_eq!(replay.rebuilds, 1);
    }

    #[test]
    fn first_and_repeated_capture_match_the_shared_stateless_graph_exactly() {
        let mut replay = HslV9Replay::default();
        let mut input = facts(60001, 0, 90.0);
        input["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0,"60000":90.0});
        input["budget_observations"]
            .as_array_mut()
            .unwrap()
            .push(json!({"timestamp":60001,
            "order":i64::MAX,"balance":1000.0,"slots":1,"scale_budget_with_excess_allowance":false,
            "exposure_budget":null,"estimated":false}));
        let expected =
            serde_json::to_value(one_shot(serde_json::from_value(input.clone()).unwrap()).unwrap())
                .unwrap();
        let first = serde_json::to_value(run(&mut replay, input.clone())).unwrap();
        let repeated = serde_json::to_value(run(&mut replay, input)).unwrap();
        assert_eq!(first, expected);
        assert_eq!(repeated, expected);
        assert_eq!(replay.rebuilds, 1);
    }

    #[test]
    fn same_anchor_inventory_correction_commits_before_preview_and_survives_refold() {
        for size in [0.0, 2.0] {
            let mut replay = HslV9Replay::default();
            let mut first = facts(60001, 0, 90.0);
            first["span"] = json!(1.0);
            run(&mut replay, first.clone());
            let initial_steps = replay.steps.clone();
            let advances = replay.advances;
            run(&mut replay, first.clone());
            assert_eq!(replay.steps, initial_steps);
            assert_eq!(replay.advances, advances);
            let mut correction = first.clone();
            correction["snapshot"]["pairs"][0]["position"]["size"] = json!(size);
            if size == 0.0 {
                correction["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
            }
            let corrected = run(&mut replay, correction.clone());
            assert_eq!(corrected.risk.as_ref().unwrap().exposed, size != 0.0);
            let last = replay.steps.last().unwrap();
            assert_eq!(last.timestamp, 60001);
            assert_eq!(last.exposed, size != 0.0);
            assert_eq!(last.upnl, -10.0 * size);
            let advances = replay.advances;
            let repeated = run(&mut replay, correction.clone());
            assert_eq!(risk(&repeated), risk(&corrected));
            assert_eq!(replay.advances, advances);
            replay.clear_derived_cache();
            let rebuilt = run(&mut replay, correction);
            assert_eq!(risk(&rebuilt), risk(&corrected));
            assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
        }
    }

    #[test]
    fn same_anchor_inventory_correction_follows_fill_and_its_budget() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(60001, 0, 90.0);
        first["span"] = json!(1.0);
        run(&mut replay, first.clone());
        let mut corrected = first.clone();
        corrected["snapshot"]["balance"] = json!(999.0);
        corrected["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        corrected["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        corrected["snapshot"]["pairs"][0]["fills"] =
            json!([execution("fee", 60001, None, -1.0, 0)]);
        corrected["budget_observations"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "timestamp":60001,"order":0,"balance":999.0,"slots":1,
                "scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false
            }));
        run(&mut replay, corrected);
        let cohort: Vec<_> = replay
            .steps
            .iter()
            .filter(|s| s.timestamp == 60001)
            .collect();
        let economic = cohort.iter().position(|s| s.net == -1.0).unwrap();
        assert!(cohort[economic].exposed);
        assert_eq!(cohort[economic].budget, 999.0);
        assert!(!cohort.last().unwrap().exposed);
        assert_eq!(cohort.last().unwrap().budget, 999.0);
        assert!(economic < cohort.len() - 1);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn preview_history_cannot_change_next_committed_minute() {
        let mut direct = HslV9Replay::default();
        let mut queried = HslV9Replay::default();
        run(&mut direct, facts(60001, 0, 90.0));
        run(&mut queried, facts(60001, 0, 90.0));
        run(&mut queried, facts(60002, 0, 100.0));
        run(&mut queried, facts(60003, 0, 1.0));
        let mut next = facts(120001, 60000, 80.0);
        next["snapshot"]["pairs"][0]["prices"] = json!({"120000":80.0});
        let a = run(&mut direct, next.clone());
        let b = run(&mut queried, next);
        assert_eq!(risk(&a), risk(&b));
        assert_eq!(queried.rebuilds, 1);
    }

    #[test]
    fn correction_and_fold_loss_rebuild_from_the_same_factual_anchor() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(60001, 0, 90.0);
        first["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0,"60000":90.0});
        run(&mut replay, first.clone());
        let mut corrected = facts(120001, 0, 90.0);
        corrected["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0,"60000":50.0,"120000":90.0});
        let output = run(&mut replay, corrected.clone());
        assert_eq!(replay.rebuilds, 2);
        replay.clear_derived_cache();
        let rebuilt = run(&mut replay, corrected);
        assert_eq!(risk(&output), risk(&rebuilt));
        assert_eq!(replay.rebuilds, 3);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn state_age_is_not_limited_by_each_acquisition_window() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 90.0);
        first["span"] = json!(1.0);
        run(&mut replay, first);
        let now = 90 * 86_400_000 + 60001;
        let mut next = facts(now, 60001, 90.0);
        next["span"] = json!(1.0);
        let output = run(&mut replay, next);
        assert!(output
            .reasons
            .contains("continuity_retained_beyond_bootstrap_window"));
        assert_eq!(replay.ledger.as_ref().unwrap().origin, 0);
        assert_eq!(replay.rebuilds, 1);
    }

    #[test]
    fn malformed_event_does_not_mutate_facts_or_forward_cache() {
        let mut replay = HslV9Replay::default();
        run(&mut replay, facts(60001, 0, 90.0));
        let before = replay.steps.clone();
        let anchor = replay.ledger.as_ref().unwrap().anchor.clone();
        let mut bad = facts(60002, 0, 90.0);
        bad["snapshot"]["pairs"][0]["position"]["basis"] = json!(-1.0);
        assert!(replay
            .consume(bad.clone(), serde_json::from_value(bad).unwrap())
            .is_err());
        assert_eq!(replay.steps, before);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, anchor);
    }

    fn execution(id: &str, timestamp: i64, delta: Option<f64>, net: f64, sequence: i64) -> Value {
        json!({"identity":id,"timestamp":timestamp,"delta":delta,"price":100.0,
            "realized":net,"fee":0.0,"sequence":sequence,"revision":0})
    }

    #[test]
    fn forward_fills_preserve_individual_clipping_and_proven_sequence() {
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        first["snapshot"]["pairs"][0]["fills"] = json!([execution("entry", 0, Some(1.0), 0.0, 0)]);
        let mut loss_last = HslV9Replay::default();
        let mut profit_last = HslV9Replay::default();
        run(&mut loss_last, first.clone());
        run(&mut profit_last, first);
        let mut next = facts(60001, 0, 100.0);
        next["span"] = json!(1.0);
        next["snapshot"]["pairs"][0]["fills"] = json!([
            execution("entry", 0, Some(1.0), 0.0, 0),
            execution("profit", 60000, None, 100.0, 0),
            execution("loss", 60000, None, -50.0, 1)
        ]);
        let a = run(&mut loss_last, next.clone());
        next["snapshot"]["pairs"][0]["fills"][1]["sequence"] = json!(1);
        next["snapshot"]["pairs"][0]["fills"][2]["sequence"] = json!(0);
        let b = run(&mut profit_last, next);
        assert_eq!(a.risk.unwrap().drawdown, 50.0);
        assert_eq!(b.risk.unwrap().drawdown, 0.0);
        assert_eq!(loss_last.rebuilds, 1);
        assert_eq!(profit_last.rebuilds, 1);
    }

    #[test]
    fn unchanged_prewrite_snapshot_cannot_reopen_a_later_same_clock_fill() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        run(&mut replay, first.clone());
        let mut before_write = facts(60000, 0, 100.0);
        before_write["span"] = json!(1.0);
        before_write["snapshot"]["pairs"][0]["fills"] =
            json!([execution("loss", 30000, None, -10.0, 0)]);
        before_write["snapshot"]["pairs"][0]["position_at"] = json!(59999);
        run(&mut replay, before_write.clone());
        let mut flat = facts(60001, 0, 100.0);
        flat["span"] = json!(1.0);
        flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["fills"] = json!([
            execution("loss", 30000, None, -10.0, 0),
            execution("exit", 60000, Some(-1.0), 0.0, 0)
        ]);
        let result = run(&mut replay, flat.clone());
        assert_eq!(
            result.risk.as_ref().unwrap().action,
            crate::hsl_intent::Action::Halted
        );
        assert_eq!(result.risk.as_ref().unwrap().cooldown_until, Some(120000));
        let exit = replay
            .steps
            .iter()
            .rposition(|s| s.timestamp == 60000 && !s.exposed)
            .unwrap();
        assert!(replay.steps[exit..].iter().all(|s| !s.exposed));
        let committed = replay.steps.clone();
        let advances = replay.advances;
        for now in [60002, 60003] {
            flat["snapshot"]["now"] = json!(now);
            flat["snapshot"]["balance_at"] = json!(now);
            flat["snapshot"]["config_at"] = json!(now);
            for field in ["position_at", "mark_at", "prices_at"] {
                flat["snapshot"]["pairs"][0][field] = json!(now);
            }
            let repeated = run(&mut replay, flat.clone());
            assert_eq!(repeated.risk.as_ref().unwrap().cooldown_until, Some(120000));
            assert_eq!(replay.steps, committed);
            assert_eq!(replay.advances, advances);
        }
        let stable = serde_json::to_value(run(&mut replay, flat.clone())).unwrap();
        replay.clear_derived_cache();
        assert_eq!(
            serde_json::to_value(run(&mut replay, flat)).unwrap(),
            stable
        );
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn unchanged_postfill_account_fact_still_reconciles_missing_inventory() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        run(&mut replay, first);
        let mut observed = facts(60001, 0, 90.0);
        observed["span"] = json!(1.0);
        observed["snapshot"]["pairs"][0]["fills"] =
            json!([execution("unbalanced-entry", 59999, Some(1.0), 0.0, 0)]);
        observed["snapshot"]["pairs"][0]["prices"] = json!({"60000":90.0});
        run(&mut replay, observed.clone());
        observed["snapshot"]["now"] = json!(120001);
        observed["snapshot"]["balance_at"] = json!(120001);
        observed["snapshot"]["config_at"] = json!(120001);
        observed["snapshot"]["pairs"][0]["position_at"] = json!(120001);
        observed["snapshot"]["pairs"][0]["mark_at"] = json!(120001);
        observed["snapshot"]["pairs"][0]["prices_at"] = json!(120001);
        run(&mut replay, observed);
        assert!(replay
            .steps
            .iter()
            .any(|s| s.timestamp == 60001 && s.upnl == -10.0));
    }

    #[test]
    fn unchanged_account_read_before_a_newer_execution_cannot_resurrect_inventory() {
        let mut replay = HslV9Replay::default();
        let first = facts(1, 0, 100.0);
        run(&mut replay, first.clone());
        let committed = replay.steps.clone();
        let mut stale = facts(60001, 0, 100.0);
        stale["snapshot"]["pairs"][0]["position_at"] = json!(59999);
        stale["snapshot"]["pairs"][0]["fills"] =
            json!([execution("exit", 60000, Some(-1.0), 0.0, 0)]);
        assert!(replay
            .consume(stale.clone(), serde_json::from_value(stale).unwrap())
            .is_err());
        assert_eq!(replay.steps, committed);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn unchanged_snapshot_preserves_true_same_quantity_roundtrip_and_entry_fee() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        run(&mut replay, first);
        let mut before_write = facts(60000, 0, 100.0);
        before_write["span"] = json!(1.0);
        run(&mut replay, before_write.clone());
        let mut after = facts(60001, 0, 100.0);
        after["span"] = json!(1.0);
        after["snapshot"]["pairs"][0]["fills"] = json!([
            execution("exit", 60000, Some(-1.0), -10.0, 0),
            execution("entry", 60000, Some(1.0), -1.0, 1)
        ]);
        let result = run(&mut replay, after);
        assert!(result.risk.as_ref().unwrap().exposed);
        assert_eq!(result.risk.as_ref().unwrap().drawdown, 1.0);
        let net: Vec<_> = replay
            .steps
            .iter()
            .filter(|s| s.net != 0.0)
            .map(|s| s.net)
            .collect();
        assert_eq!(net, vec![-10.0, -1.0]);
    }

    #[test]
    fn newly_discovered_fill_at_anchor_follows_immutable_account_initializer() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(60001, 0, 90.0);
        first["span"] = json!(1.0);
        run(&mut replay, first.clone());
        let mut flat = facts(60002, 0, 90.0);
        flat["span"] = json!(1.0);
        flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        let mut exit = execution("exit", 60001, Some(-1.0), -10.0, 0);
        exit["price"] = json!(90.0);
        flat["snapshot"]["pairs"][0]["fills"] = json!([exit]);
        let result = run(&mut replay, flat);
        assert!(result.reasons.contains("ambiguous_bootstrap_fill_phase"));
        assert_eq!(
            result.risk.as_ref().unwrap().action,
            crate::hsl_intent::Action::Halted
        );
        let economic = replay.steps.iter().rposition(|s| s.net == -10.0).unwrap();
        assert!(replay.steps[economic..].iter().all(|s| !s.exposed));
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn forward_flatten_cooldown_survives_sliding_tape_and_balance_change() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        first["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        first["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        run(&mut replay, first);
        let mut open = facts(60001, 0, 90.0);
        open["span"] = json!(1.0);
        open["snapshot"]["pairs"][0]["prices"] = json!({"60000":90.0});
        open["snapshot"]["pairs"][0]["fills"] =
            json!([execution("entry", 10000, Some(1.0), -1.0, 0)]);
        assert_eq!(
            run(&mut replay, open.clone()).risk.unwrap().action,
            crate::hsl_intent::Action::Panic
        );
        let mut close = facts(60003, 0, 90.0);
        close["span"] = json!(1.0);
        close["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        close["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        let mut exit = execution("exit", 60002, Some(-1.0), -11.0, 0);
        exit["price"] = json!(90.0);
        close["snapshot"]["pairs"][0]["fills"] =
            json!([execution("entry", 10000, Some(1.0), -1.0, 0), exit]);
        let stopped = run(&mut replay, close.clone()).risk.unwrap();
        assert_eq!(stopped.action, crate::hsl_intent::Action::Halted);
        assert_eq!(stopped.cooldown_until, Some(120002));
        close["snapshot"]["now"] = json!(90000);
        close["snapshot"]["start"] = json!(70000);
        close["snapshot"]["balance"] = json!(10000.0);
        close["snapshot"]["balance_at"] = json!(90000);
        close["snapshot"]["config_at"] = json!(90000);
        let pair = &mut close["snapshot"]["pairs"][0];
        pair["fills"] = json!([]);
        pair["position_at"] = json!(90000);
        pair["mark_at"] = json!(90000);
        pair["prices_at"] = json!(90000);
        let later = run(&mut replay, close).risk.unwrap();
        assert_eq!(later.action, crate::hsl_intent::Action::Halted);
        assert_eq!(later.cooldown_until, stopped.cooldown_until);
        assert_eq!(replay.rebuilds, 1);
    }

    #[test]
    fn budget_observation_commits_retained_mark_and_next_ema_baseline() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 90.0);
        first["span"] = json!(3.0);
        run(&mut replay, first);
        let mut changed = facts(60001, 0, 90.0);
        changed["span"] = json!(3.0);
        changed["snapshot"]["balance"] = json!(2000.0);
        let fact = json!({"timestamp":60001,"order":i64::MAX,"balance":2000.0,"slots":1,
            "scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false});
        changed["budget_observations"]
            .as_array_mut()
            .unwrap()
            .push(fact.clone());
        let observed = run(&mut replay, changed.clone());
        assert!(replay
            .steps
            .iter()
            .any(|s| s.timestamp == 60001 && s.budget == 2000.0));
        assert_eq!(
            replay.recorder.state.current(60001).ema,
            observed.risk.as_ref().unwrap().ema
        );
        let mut reference = replay.recorder.state.clone();
        let policy = Policy::new(43200.0, 3.0, 0.005, 60000).unwrap();
        let expected = reference
            .observe(
                &policy,
                Event {
                    timestamp_ms: 120000,
                    budget: 2000.0,
                    upnl: -10.0,
                    net_realized_pnl: 0.0,
                    exposed: true,
                },
            )
            .unwrap();
        let mut next = facts(120001, 0, 90.0);
        next["span"] = json!(3.0);
        next["snapshot"]["balance"] = json!(2000.0);
        next["budget_observations"]
            .as_array_mut()
            .unwrap()
            .push(fact);
        next["snapshot"]["pairs"][0]["prices"] = json!({"120000":90.0});
        let output = run(&mut replay, next).risk.unwrap();
        assert_eq!(output.drawdown, expected.drawdown);
        assert_eq!(output.ema, expected.ema);
    }

    #[test]
    fn policy_change_refolds_same_initializer_and_retained_facts() {
        let mut replay = HslV9Replay::default();
        let first = facts(60001, 0, 90.0);
        assert_eq!(
            run(&mut replay, first.clone()).risk.unwrap().action,
            crate::hsl_intent::Action::Panic
        );
        let mut changed = facts(60002, 0, 90.0);
        changed["threshold"] = json!(0.5);
        changed["span"] = json!(3.0);
        changed["lookback_span_minutes"] = json!(288.0);
        changed["cooldown_ms"] = json!(12345);
        let output = run(&mut replay, changed);
        assert_eq!(
            output.risk.unwrap().action,
            crate::hsl_intent::Action::Normal
        );
        assert_eq!(replay.rebuilds, 2);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn reconciliation_seed_never_uses_a_later_quote_at_an_earlier_clock() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(60000, 0, 100.0);
        first["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        first["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        run(&mut replay, first);
        let mut observed = facts(120001, 0, 90.0);
        observed["snapshot"]["pairs"][0]["position_at"] = json!(119999);
        observed["snapshot"]["pairs"][0]["mark_at"] = json!(120001);
        run(&mut replay, observed);
        let seed = replay.steps.iter().find(|s| s.seed).unwrap();
        assert_eq!(seed.timestamp, 119999);
        // The earlier retained mark was 100. The quote at 120001 is preview
        // only and cannot change the seed or the prior minute's sample.
        assert_eq!(seed.upnl, 0.0);
        assert!(replay.steps.iter().all(|s| s.upnl == 0.0));
        let fact = &replay.ledger.as_ref().unwrap().pairs[&("A".into(), 0)].positions[&120001];
        assert_eq!(fact.position_at, Some(119999));
        assert_eq!(fact.mark_at, Some(120001));
    }

    #[test]
    fn changed_prewrite_inventory_uses_source_clock_before_later_discovered_exit() {
        for known in [false, true] {
            let mut replay = HslV9Replay::default();
            let mut first = facts(1, 0, 100.0);
            first["span"] = json!(1.0);
            let entry = execution("entry", 0, Some(1.0), 0.0, 0);
            if known {
                first["snapshot"]["pairs"][0]["fills"] = json!([entry.clone()]);
            }
            run(&mut replay, first.clone());
            let mut before = facts(60000, 0, 90.0);
            before["span"] = json!(1.0);
            before["snapshot"]["pairs"][0]["position"]["size"] = json!(2.0);
            before["snapshot"]["pairs"][0]["position_at"] = json!(59999);
            let addition = execution("addition", 59999, Some(1.0), 0.0, 0);
            if known {
                before["snapshot"]["pairs"][0]["fills"] = json!([entry.clone(), addition.clone()]);
            }
            run(&mut replay, before.clone());
            let mut exit = execution("exit", 60000, Some(-2.0), -20.0, 0);
            exit["price"] = json!(90.0);
            let mut flat = facts(90000, 0, 90.0);
            flat["span"] = json!(1.0);
            flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
            flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
            flat["snapshot"]["pairs"][0]["fills"] = if known {
                json!([entry, addition, exit])
            } else {
                json!([exit])
            };
            let output = run(&mut replay, flat.clone());
            assert_eq!(output.risk.as_ref().unwrap().cooldown_until, Some(120000));
            let flat_step = replay
                .steps
                .iter()
                .rposition(|s| s.timestamp == 60000 && !s.exposed)
                .unwrap();
            assert!(replay.steps[flat_step..].iter().all(|s| !s.exposed));
            assert!(replay
                .steps
                .iter()
                .filter(|s| s.timestamp == 59999)
                .all(|s| s.upnl == 0.0));
            let committed = replay.steps.clone();
            for now in [90001, 90002] {
                flat["snapshot"]["now"] = json!(now);
                flat["snapshot"]["balance_at"] = json!(now);
                flat["snapshot"]["config_at"] = json!(now);
                for field in ["position_at", "mark_at", "prices_at"] {
                    flat["snapshot"]["pairs"][0][field] = json!(now);
                }
                assert_eq!(
                    run(&mut replay, flat.clone()).risk.unwrap().cooldown_until,
                    Some(120000)
                );
                assert_eq!(replay.steps, committed);
            }
            let before = serde_json::to_value(run(&mut replay, flat.clone())).unwrap();
            replay.clear_derived_cache();
            assert_eq!(
                serde_json::to_value(run(&mut replay, flat)).unwrap(),
                before
            );
            assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
        }
    }

    #[test]
    fn causal_balance_before_exit_prevents_false_red_terminal_reclassification() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        let entry = execution("entry", 0, Some(1.0), 0.0, 0);
        first["snapshot"]["pairs"][0]["fills"] = json!([entry.clone()]);
        run(&mut replay, first.clone());
        let mut before = facts(60000, 0, 90.0);
        before["span"] = json!(1.0);
        before["snapshot"]["balance"] = json!(10000.0);
        before["snapshot"]["balance_at"] = json!(59999);
        before["snapshot"]["pairs"][0]["position_at"] = json!(59999);
        before["snapshot"]["pairs"][0]["fills"] = json!([entry.clone()]);
        let budget = json!({"timestamp":59999,"observed_at":60000,"order":i64::MAX,
            "balance":10000.0,"slots":1,"scale_budget_with_excess_allowance":false,
            "exposure_budget":null,"estimated":false});
        before["budget_observations"]
            .as_array_mut()
            .unwrap()
            .push(budget.clone());
        assert!(!run(&mut replay, before).risk.unwrap().red);
        let mut flat = facts(90000, 0, 90.0);
        flat["span"] = json!(1.0);
        flat["snapshot"]["balance"] = json!(10000.0);
        flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        let mut exit = execution("exit", 60000, Some(-1.0), -10.0, 0);
        exit["price"] = json!(90.0);
        flat["snapshot"]["pairs"][0]["fills"] = json!([entry, exit]);
        flat["budget_observations"]
            .as_array_mut()
            .unwrap()
            .push(budget);
        let output = run(&mut replay, flat.clone());
        assert_eq!(
            output.risk.as_ref().unwrap().action,
            crate::hsl_intent::Action::Normal
        );
        assert_eq!(output.risk.as_ref().unwrap().cooldown_until, None);
        assert!(replay
            .steps
            .iter()
            .filter(|s| s.timestamp == 60000)
            .all(|s| s.budget == 10000.0));
        let stable = serde_json::to_value(output).unwrap();
        replay.clear_derived_cache();
        assert_eq!(
            serde_json::to_value(run(&mut replay, flat)).unwrap(),
            stable
        );
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    fn aggregate(mut value: Value) -> Value {
        value["snapshot"]["mode"] = json!("portfolio");
        value["snapshot"]["pside"] = Value::Null;
        value["snapshot"]["symbol"] = Value::Null;
        value
    }

    #[test]
    fn newly_discovered_unknown_episode_waits_for_its_own_supported_mark_clock() {
        let mut replay = HslV9Replay::default();
        let mut first = aggregate(facts(60000, 0, 100.0));
        first["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        first["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        run(&mut replay, first.clone());
        let mut observed = aggregate(facts(120001, 0, 90.0));
        observed["snapshot"]["pairs"][0]["symbol"] = json!("B");
        observed["snapshot"]["pairs"][0]["position_at"] = json!(119999);
        let output = run(&mut replay, observed.clone());
        assert_eq!(output.unavailable_reason, None);
        assert!(output.reasons.contains("unknown_prefix_causal_mark_seed"));
        let seed = replay.steps.iter().find(|s| s.seed).unwrap();
        assert_eq!(seed.timestamp, 120001);
        assert_eq!(seed.upnl, -10.0);
        assert_eq!(output.risk.as_ref().unwrap().drawdown, 10.0);
        assert!(replay
            .steps
            .iter()
            .filter(|s| s.timestamp < 120001)
            .all(|s| !s.exposed));
        let stable = serde_json::to_value(output).unwrap();
        replay.clear_derived_cache();
        assert_eq!(
            serde_json::to_value(run(&mut replay, observed)).unwrap(),
            stable
        );
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn already_exposed_aggregate_cannot_consume_a_new_pairs_future_quote() {
        let mut replay = HslV9Replay::default();
        let first = aggregate(facts(60000, 0, 100.0));
        run(&mut replay, first.clone());
        let mut observed = aggregate(facts(120001, 0, 100.0));
        let mut other = observed["snapshot"]["pairs"][0].clone();
        other["symbol"] = json!("B");
        other["position"]["mark"] = json!(90.0);
        other["position_at"] = json!(119999);
        observed["snapshot"]["pairs"]
            .as_array_mut()
            .unwrap()
            .push(other);
        let unavailable = run(&mut replay, observed.clone());
        assert_eq!(
            unavailable.unavailable_reason.as_deref(),
            Some("causal_position_mark_unavailable")
        );
        assert!(unavailable.decision.is_none() && unavailable.risk.is_none());
        assert!(!replay.cache_valid);
        replay.clear_derived_cache();
        assert_eq!(
            serde_json::to_value(run(&mut replay, observed.clone())).unwrap(),
            serde_json::to_value(unavailable).unwrap()
        );
        // A factual completed mark before the inventory source clock repairs
        // the causal gap. It is not a quote copied backwards.
        observed["snapshot"]["pairs"][1]["prices"] = json!({"60000":95.0});
        let repaired = run(&mut replay, observed);
        assert_eq!(repaired.unavailable_reason, None);
        assert!(repaired.risk.unwrap().exposed);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn equal_source_and_fill_clock_discloses_phase_ambiguity() {
        let mut replay = HslV9Replay::default();
        let first = facts(1, 0, 100.0);
        run(&mut replay, first);
        let mut changed = facts(60001, 0, 100.0);
        changed["snapshot"]["pairs"][0]["position"]["size"] = json!(2.0);
        changed["snapshot"]["pairs"][0]["position_at"] = json!(60000);
        changed["snapshot"]["pairs"][0]["fills"] =
            json!([execution("entry", 60000, Some(1.0), -1.0, 0)]);
        changed["budget_observations"].as_array_mut().unwrap().push(json!({
            "timestamp":60000,"observed_at":60001,"order":i64::MAX,"balance":2000.0,
            "slots":1,"scale_budget_with_excess_allowance":false,"exposure_budget":null,"estimated":false
        }));
        let output = run(&mut replay, changed);
        assert!(output.reasons.contains("ambiguous_position_fill_phase"));
        assert!(output.reasons.contains("ambiguous_budget_fill_phase"));
    }

    #[test]
    fn held_zero_slot_scope_errors_while_flat_zero_slot_scope_is_inactive() {
        let mut replay = HslV9Replay::default();
        let mut value = facts(1, 0, 100.0);
        value["slots"] = json!(0);
        assert!(replay
            .consume(
                value.clone(),
                serde_json::from_value(value.clone()).unwrap()
            )
            .unwrap_err()
            .contains("zero configured slots"));
        assert!(replay.ledger.is_none());
        value["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        value["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        let output = run(&mut replay, value);
        assert!(output.decision.is_none() && output.unavailable_reason.is_none());
        assert!(output.reasons.contains("inactive_scope"));
    }

    #[test]
    fn stale_partial_fill_quantity_is_unavailable_without_committing_quote_or_false_red() {
        for short in [false, true] {
            for portfolio in [false, true] {
                let dir = if short { -1.0 } else { 1.0 };
                let side = if short { "short" } else { "long" };
                let make = |now, size, mark| {
                    let mut value = facts(now, 0, mark);
                    value["span"] = json!(1.0);
                    value["threshold"] = json!(0.012);
                    value["snapshot"]["pside"] = json!(side);
                    value["snapshot"]["pairs"][0]["position"]["pside"] = json!(side);
                    value["snapshot"]["pairs"][0]["position"]["size"] = json!(dir * size);
                    if portfolio {
                        value = aggregate(value);
                        let mut other = value["snapshot"]["pairs"][0].clone();
                        other["symbol"] = json!("B");
                        other["position"]["mark"] = json!(100.0);
                        other["position"]["size"] = json!(dir);
                        value["snapshot"]["pairs"]
                            .as_array_mut()
                            .unwrap()
                            .push(other);
                    }
                    value
                };
                let mut replay = HslV9Replay::default();
                let first = make(1, 1.0, 100.0);
                run(&mut replay, first.clone());
                let before_steps = replay.steps.clone();
                let advances = replay.advances;
                let mark = if short { 110.0 } else { 90.0 };
                let mut stale = make(60001, 1.0, mark);
                stale["snapshot"]["pairs"][0]["position_at"] = json!(59999);
                let mut exit = execution("partial", 60000, Some(-dir * 0.5), -5.0, 0);
                exit["price"] = json!(mark);
                stale["snapshot"]["pairs"][0]["fills"] = json!([exit.clone()]);
                let unavailable = run(&mut replay, stale.clone());
                assert_eq!(
                    unavailable.unavailable_reason.as_deref(),
                    Some("causal_position_snapshot_precedes_fill")
                );
                assert!(unavailable.risk.is_none() && unavailable.decision.is_none());
                assert_eq!(replay.advances, advances);
                assert_eq!(replay.steps, before_steps);
                replay.clear_derived_cache();
                assert_eq!(
                    serde_json::to_value(run(&mut replay, stale)).unwrap(),
                    serde_json::to_value(unavailable).unwrap()
                );
                let mut recovered = make(60002, 0.5, mark);
                recovered["snapshot"]["pairs"][0]["fills"] = json!([exit]);
                let output = run(&mut replay, recovered.clone());
                assert_eq!(output.unavailable_reason, None);
                assert_eq!(output.risk.as_ref().unwrap().drawdown, 10.0);
                assert_eq!(
                    output.risk.as_ref().unwrap().action,
                    crate::hsl_intent::Action::Normal
                );
                let stable = serde_json::to_value(output).unwrap();
                replay.clear_derived_cache();
                assert_eq!(
                    serde_json::to_value(run(&mut replay, recovered)).unwrap(),
                    stable
                );
                assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
            }
        }
    }

    #[test]
    fn actual_same_value_source_endpoints_preserve_postfill_reconciliation_and_cooldown() {
        for reverse in [false, true] {
            let mut replay = HslV9Replay::default();
            let mut first = facts(1, 0, 100.0);
            first["span"] = json!(1.0);
            first["threshold"] = json!(0.012);
            run(&mut replay, first.clone());
            let mut account = facts(60001, 0, 90.0);
            account["span"] = json!(1.0);
            account["threshold"] = json!(0.012);
            let mut partial = execution("partial", 60000, Some(-0.5), -5.0, 0);
            partial["price"] = json!(90.0);
            account["snapshot"]["pairs"][0]["fills"] = json!([partial.clone()]);
            let sources = if reverse {
                [60001, 59999]
            } else {
                [59999, 60001]
            };
            for (i, source) in sources.into_iter().enumerate() {
                account["snapshot"]["pairs"][0]["position_at"] = json!(source);
                let output = run(&mut replay, account.clone());
                if !reverse && i == 0 {
                    assert_eq!(
                        output.unavailable_reason.as_deref(),
                        Some("causal_position_snapshot_precedes_fill")
                    );
                } else {
                    assert_eq!(output.unavailable_reason, None);
                    assert_eq!(output.risk.unwrap().drawdown, 15.0);
                }
            }
            let reconciliation = replay.steps.iter().find(|s| s.timestamp == 60001).unwrap();
            assert_eq!(reconciliation.upnl, -10.0);
            let fact = &replay.ledger.as_ref().unwrap().pairs[&("A".into(), 0)].positions[&60001];
            assert_eq!(fact.position_at, Some(59999));
            assert_eq!(fact.position_latest_at, Some(60001));
            let committed = replay.steps.clone();
            let advances = replay.advances;
            run(&mut replay, account.clone());
            assert_eq!(replay.steps, committed);
            assert_eq!(replay.advances, advances);
            let stable = serde_json::to_value(run(&mut replay, account.clone())).unwrap();
            replay.clear_derived_cache();
            assert_eq!(
                serde_json::to_value(run(&mut replay, account)).unwrap(),
                stable
            );
            let mut flat = facts(60002, 0, 90.0);
            flat["span"] = json!(1.0);
            flat["threshold"] = json!(0.012);
            flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
            flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
            let mut exit = execution("rebated_exit", 60002, Some(-1.0), 0.0, 0);
            exit["price"] = json!(90.0);
            flat["snapshot"]["pairs"][0]["fills"] = json!([partial, exit]);
            let terminal = run(&mut replay, flat.clone());
            assert_eq!(terminal.risk.as_ref().unwrap().cooldown_until, Some(120002));
            assert_eq!(terminal.risk.as_ref().unwrap().drawdown, 5.0);
            let stable = serde_json::to_value(terminal).unwrap();
            replay.clear_derived_cache();
            assert_eq!(
                serde_json::to_value(run(&mut replay, flat)).unwrap(),
                stable
            );
            assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
        }
    }

    #[test]
    fn corrected_position_sources_are_sorted_before_no_op_detection() {
        let mut ordered = HslV9Replay::default();
        let mut corrected = HslV9Replay::default();
        let mut first = facts(1, 0, 100.0);
        first["span"] = json!(1.0);
        first["threshold"] = json!(0.012);
        run(&mut ordered, first.clone());
        run(&mut corrected, first.clone());
        for (replay, readings) in [
            (&mut ordered, [(40000, 1.0), (50000, 2.0), (60000, 1.0)]),
            (&mut corrected, [(50000, 2.0), (40000, 1.0), (60000, 1.0)]),
        ] {
            for (i, (source, size)) in readings.into_iter().enumerate() {
                let mut account = facts(60001 + i as i64 * 10000, 0, 90.0);
                account["span"] = json!(1.0);
                account["threshold"] = json!(0.012);
                account["snapshot"]["pairs"][0]["position_at"] = json!(source);
                account["snapshot"]["pairs"][0]["position"]["size"] = json!(size);
                run(replay, account);
            }
        }
        assert_eq!(corrected.steps, ordered.steps);
        assert_eq!(corrected.ledger.as_ref().unwrap().anchor, first);
        let mut flat = facts(90001, 0, 90.0);
        flat["span"] = json!(1.0);
        flat["threshold"] = json!(0.012);
        flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        let mut exit = execution("exit", 90000, Some(-1.0), -10.0, 0);
        exit["price"] = json!(90.0);
        flat["snapshot"]["pairs"][0]["fills"] = json!([exit]);
        let terminal = run(&mut corrected, flat.clone());
        let expected = run(&mut ordered, flat.clone());
        assert_eq!(risk(&terminal), risk(&expected));
        assert_eq!(
            terminal.risk.as_ref().unwrap().action,
            crate::hsl_intent::Action::Normal
        );
        assert_eq!(terminal.risk.as_ref().unwrap().cooldown_until, None);
        corrected.clear_derived_cache();
        assert_eq!(
            serde_json::to_value(run(&mut corrected, flat)).unwrap(),
            serde_json::to_value(terminal).unwrap()
        );
    }

    #[test]
    fn flat_cashflow_keeps_diagnostic_without_inventing_an_episode() {
        let mut replay = HslV9Replay::default();
        let mut input = facts(60001, 0, 100.0);
        input["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        input["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        input["snapshot"]["pairs"][0]["fills"] =
            json!([execution("cashflow", 60000, Some(0.0), -100.0, 0)]);
        let output = run(&mut replay, input);
        assert!(output.reasons.contains("flat_cashflow_without_exposure"));
        assert_eq!(output.episodes, 0);
        assert!(!output.risk.unwrap().exposed);
    }

    #[test]
    fn supported_earlier_identical_position_read_improves_provenance_only() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(60001, 0, 90.0);
        first["snapshot"]["pairs"][0]["fills_started_at"] = json!(59900);
        first["snapshot"]["pairs"][0]["fills_at"] = json!(59950);
        let initial = run(&mut replay, first.clone());
        assert!(initial.reasons.contains("fills_before_position"));
        let mut corrected = first.clone();
        corrected["snapshot"]["pairs"][0]["position_at"] = json!(59800);
        let output = run(&mut replay, corrected);
        assert!(!output.reasons.contains("fills_before_position"));
        assert_eq!(risk(&output), risk(&initial));
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
    }

    #[test]
    fn forward_fill_capture_cut_advances_once_and_matches_causal_reference() {
        let mut replay = HslV9Replay::default();
        let mut reference = HslV9Replay::default();
        run(&mut replay, facts(1, 0, 100.0));
        run(&mut reference, facts(1, 0, 100.0));
        let mut captured = facts(200, 0, 100.0);
        captured["span"] = json!(1.0);
        captured["snapshot"]["pairs"][0]["fills_at"] = json!(100);
        captured["snapshot"]["pairs"][0]["fills"] =
            json!([execution("outside_capture", 110, Some(0.0), -100.0, 0)]);
        let mut causal = captured.clone();
        causal["snapshot"]["pairs"][0]["fills"] = json!([]);
        let excluded = run(&mut replay, captured.clone());
        assert_eq!(risk(&excluded), risk(&run(&mut reference, causal)));
        assert!(excluded.reasons.contains("post_capture_fill"));
        assert_eq!(excluded.risk.unwrap().drawdown, 0.0);
        captured["snapshot"]["pairs"][0]["fills_at"] = json!(200);
        let included = run(&mut replay, captured.clone());
        assert_eq!(included.risk.as_ref().unwrap().drawdown, 100.0);
        let committed = replay.steps.clone();
        assert_eq!(risk(&run(&mut replay, captured.clone())), risk(&included));
        assert_eq!(replay.steps, committed);
        replay.clear_derived_cache();
        assert_eq!(
            serde_json::to_value(run(&mut replay, captured)).unwrap(),
            serde_json::to_value(included).unwrap()
        );
    }

    #[test]
    fn quarantined_revision_cannot_resurrect_its_retained_member_outside_new_window() {
        let mut replay = HslV9Replay::default();
        run(&mut replay, facts(1, 0, 100.0));
        let mut earlier = facts(100, 0, 100.0);
        earlier["snapshot"]["pairs"][0]["fills"] =
            json!([execution("revision", 50, Some(0.0), -10.0, 0)]);
        assert_eq!(run(&mut replay, earlier).risk.unwrap().drawdown, 10.0);
        let mut later = facts(200, 60, 100.0);
        later["snapshot"]["pairs"][0]["fills_at"] = json!(100);
        later["snapshot"]["pairs"][0]["fills"] =
            json!([execution("revision", 150, Some(0.0), -100.0, 0)]);
        let quarantined = run(&mut replay, later.clone());
        assert_eq!(quarantined.risk.as_ref().unwrap().drawdown, 0.0);
        assert!(replay
            .ledger
            .as_ref()
            .unwrap()
            .pairs
            .values()
            .all(|p| p.fills.is_empty()));
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, later)), risk(&quarantined));
    }

    #[test]
    fn future_higher_revision_does_not_replace_causal_revision_until_its_cut() {
        let mut replay = HslV9Replay::default();
        run(&mut replay, facts(1, 0, 100.0));
        let old = execution("revision", 50, Some(0.0), -10.0, 0);
        let mut corrected = execution("revision", 150, Some(0.0), -20.0, 0);
        corrected["revision"] = json!(1);
        let mut input = facts(200, 0, 100.0);
        input["snapshot"]["pairs"][0]["fills_at"] = json!(100);
        input["snapshot"]["pairs"][0]["fills"] = json!([old, corrected]);
        assert_eq!(run(&mut replay, input.clone()).risk.unwrap().drawdown, 10.0);
        input["snapshot"]["pairs"][0]["fills_at"] = json!(200);
        assert_eq!(run(&mut replay, input).risk.unwrap().drawdown, 20.0);
    }

    #[test]
    fn deferred_equal_anchor_fill_is_not_in_the_immutable_admitted_cohort() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(100, 0, 100.0);
        first["snapshot"]["pairs"][0]["fills_at"] = json!(90);
        first["snapshot"]["pairs"][0]["fills"] =
            json!([execution("not_captured", 100, Some(0.0), -100.0, 0)]);
        assert_eq!(run(&mut replay, first.clone()).risk.unwrap().drawdown, 0.0);
        let mut admitted = facts(200, 0, 100.0);
        admitted["snapshot"]["pairs"][0]["fills"] = first["snapshot"]["pairs"][0]["fills"].clone();
        admitted["snapshot"]["pairs"][0]["fills_at"] = json!(200);
        let output = run(&mut replay, admitted.clone());
        assert_eq!(output.risk.as_ref().unwrap().drawdown, 100.0);
        assert!(output.reasons.contains("ambiguous_bootstrap_fill_phase"));
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, admitted)), risk(&output));
    }

    #[test]
    fn later_receipts_repair_the_prefix_without_forging_anchor_capture_clocks() {
        let mut replay = HslV9Replay::default();
        let mut first = facts(100, 0, 100.0);
        first["snapshot"]["pairs"][0]["fills_at"] = json!(90);
        first["snapshot"]["pairs"][0]["prices_at"] = json!(90);
        run(&mut replay, first.clone());
        let mut later = facts(200, 0, 100.0);
        later["snapshot"]["pairs"][0]["fills_at"] = json!(200);
        later["snapshot"]["pairs"][0]["fills"] = json!([execution(
            "later_historical_receipt",
            95,
            Some(0.0),
            -100.0,
            0
        )]);
        let repaired = run(&mut replay, later.clone());
        assert_eq!(repaired.risk.as_ref().unwrap().drawdown, 100.0);
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, first);
        assert_eq!(
            replay
                .ledger
                .as_ref()
                .unwrap()
                .bootstrap()
                .unwrap()
                .snapshot
                .pairs[0]
                .fills_at,
            Some(90)
        );
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, later)), risk(&repaired));

        let mut prices = HslV9Replay::default();
        let mut initial = facts(60001, 0, 100.0);
        initial["snapshot"]["pairs"][0]["prices_at"] = json!(59999);
        run(&mut prices, initial.clone());
        let mut later = facts(120001, 0, 100.0);
        later["snapshot"]["pairs"][0]["prices"] = json!({"60000":150.0});
        assert_eq!(run(&mut prices, later).risk.unwrap().drawdown, 50.0);
        assert_eq!(prices.ledger.as_ref().unwrap().anchor, initial);
    }

    #[test]
    fn post_position_pre_capture_exit_keeps_its_actual_execution_cooldown() {
        let entry = execution("entry", 0, Some(1.0), 0.0, 0);
        let mut exit = execution("exit", 95, Some(-1.0), -10.0, 1);
        exit["price"] = json!(90.0);
        let mut initial = facts(100, 0, 100.0);
        initial["span"] = json!(1.0);
        initial["snapshot"]["pairs"][0]["position_at"] = json!(90);
        initial["snapshot"]["pairs"][0]["fills_at"] = json!(90);
        initial["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
        initial["snapshot"]["pairs"][0]["fills"] = json!([entry.clone()]);
        let mut replay = HslV9Replay::default();
        run(&mut replay, initial.clone());
        let mut flat = facts(200, 0, 90.0);
        flat["span"] = json!(1.0);
        flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
        flat["snapshot"]["pairs"][0]["fills"] = json!([entry, exit]);
        flat["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
        let stopped = run(&mut replay, flat.clone());
        assert_eq!(stopped.risk.as_ref().unwrap().drawdown, 10.0);
        assert_eq!(stopped.risk.as_ref().unwrap().cooldown_until, Some(60095));
        assert_eq!(stopped.decision.as_ref().unwrap().flat_at, Some(95));
        assert!(!replay.steps.iter().any(|step| step.timestamp == 100));
        assert_eq!(
            risk(&stopped),
            risk(&run(&mut HslV9Replay::default(), flat.clone()))
        );
        assert_eq!(replay.ledger.as_ref().unwrap().anchor, initial);
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, flat)), risk(&stopped));
    }

    #[test]
    fn post_position_pre_capture_partial_fill_blocks_a_stale_preview_until_fresh() {
        let mut initial = facts(100, 0, 100.0);
        initial["snapshot"]["pairs"][0]["position_at"] = json!(90);
        initial["snapshot"]["pairs"][0]["fills_at"] = json!(90);
        initial["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
        let mut replay = HslV9Replay::default();
        run(&mut replay, initial);
        let mut partial = facts(200, 0, 90.0);
        partial["snapshot"]["pairs"][0]["position_at"] = json!(90);
        partial["snapshot"]["pairs"][0]["fills"] =
            json!([execution("partial", 95, Some(-0.5), -5.0, 0)]);
        partial["snapshot"]["pairs"][0]["fills"][0]["price"] = json!(90.0);
        let blocked = run(&mut replay, partial.clone());
        assert_eq!(
            blocked.unavailable_reason.as_deref(),
            Some("causal_position_snapshot_precedes_fill")
        );
        assert!(blocked.risk.is_none());
        partial["snapshot"]["pairs"][0]["position_at"] = json!(200);
        partial["snapshot"]["pairs"][0]["position"]["size"] = json!(0.5);
        let recovered = run(&mut replay, partial.clone());
        assert_eq!(recovered.risk.as_ref().unwrap().drawdown, 10.0);
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, partial)), risk(&recovered));
    }

    #[test]
    fn unknown_source_inventory_precedes_its_exit_and_waits_for_a_causal_mark() {
        for earlier_mark in [false, true] {
            let mut initial = facts(100, 0, 100.0);
            initial["span"] = json!(1.0);
            initial["snapshot"]["pairs"][0]["position_at"] = json!(90);
            initial["snapshot"]["pairs"][0]["fills_at"] = json!(90);
            if earlier_mark {
                initial["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
            }
            let mut replay = HslV9Replay::default();
            run(&mut replay, initial.clone());
            let mut flat = facts(200, 0, 90.0);
            flat["span"] = json!(1.0);
            flat["snapshot"]["pairs"][0]["position"]["size"] = json!(0.0);
            flat["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
            let mut exit = execution("unknown_exit", 95, Some(-1.0), -10.0, 0);
            exit["price"] = json!(90.0);
            flat["snapshot"]["pairs"][0]["fills"] = json!([exit]);
            if earlier_mark {
                flat["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
            }
            let result = run(&mut replay, flat.clone());
            assert_eq!(result.risk.as_ref().unwrap().drawdown, 10.0);
            assert_eq!(result.risk.as_ref().unwrap().cooldown_until, Some(60095));
            assert_eq!(result.decision.as_ref().unwrap().flat_at, Some(95));
            assert!(!replay.steps.iter().any(|s| s.timestamp == 100));
            let seed = replay.steps.iter().find(|s| s.seed).unwrap();
            assert_eq!(seed.timestamp, if earlier_mark { 90 } else { 95 });
            assert_eq!(seed.upnl, if earlier_mark { 0.0 } else { -10.0 });
            assert_eq!(replay.ledger.as_ref().unwrap().anchor, initial);
            replay.clear_derived_cache();
            assert_eq!(risk(&run(&mut replay, flat)), risk(&result));
        }
    }

    #[test]
    fn bootstrap_source_tail_does_not_borrow_another_held_pairs_capture_quote() {
        let mut initial = aggregate(facts(100, 0, 100.0));
        initial["span"] = json!(1.0);
        initial["snapshot"]["pairs"][0]["position_at"] = json!(90);
        initial["snapshot"]["pairs"][0]["fills_at"] = json!(90);
        let mut other = initial["snapshot"]["pairs"][0].clone();
        other["symbol"] = json!("B");
        initial["snapshot"]["pairs"]
            .as_array_mut()
            .unwrap()
            .push(other);
        initial["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
        let mut replay = HslV9Replay::default();
        run(&mut replay, initial);
        let mut later = aggregate(facts(200, 0, 90.0));
        later["span"] = json!(1.0);
        later["snapshot"]["pairs"][0]["position"]["size"] = json!(0.5);
        later["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
        let mut exit = execution("partial", 95, Some(-0.5), -5.0, 0);
        exit["price"] = json!(90.0);
        later["snapshot"]["pairs"][0]["fills"] = json!([exit]);
        let mut other = facts(200, 0, 80.0)["snapshot"]["pairs"][0].clone();
        other["symbol"] = json!("B");
        later["snapshot"]["pairs"]
            .as_array_mut()
            .unwrap()
            .push(other);
        let unavailable = run(&mut replay, later.clone());
        assert_eq!(
            unavailable.unavailable_reason.as_deref(),
            Some("causal_position_mark_unavailable")
        );
        assert!(unavailable.risk.is_none());
        later["snapshot"]["pairs"][1]["prices"] = json!({"0":100.0});
        let recovered = run(&mut replay, later.clone());
        assert_eq!(recovered.risk.as_ref().unwrap().drawdown, 30.0);
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, later)), risk(&recovered));
    }

    #[test]
    fn bootstrap_terminal_unknown_episode_uses_only_its_own_supported_quote_clock() {
        for partial in [false, true] {
            let mut initial = aggregate(facts(100, 0, 100.0));
            initial["span"] = json!(1.0);
            initial["snapshot"]["pairs"][0]["position_at"] = json!(90);
            initial["snapshot"]["pairs"][0]["fills_at"] = json!(90);
            let mut other = initial["snapshot"]["pairs"][0].clone();
            other["symbol"] = json!("B");
            other["position_at"] = json!(100);
            initial["snapshot"]["pairs"]
                .as_array_mut()
                .unwrap()
                .push(other);
            initial["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
            let mut replay = HslV9Replay::default();
            run(&mut replay, initial.clone());
            let mut later = aggregate(facts(200, 0, 90.0));
            later["span"] = json!(1.0);
            later["snapshot"]["pairs"][0]["position"]["size"] =
                json!(if partial { 0.5 } else { 0.0 });
            if !partial {
                later["snapshot"]["pairs"][0]["position"]["basis"] = json!(0.0);
            }
            later["snapshot"]["pairs"][0]["prices"] = json!({"0":100.0});
            let mut exit = execution(
                "exit",
                95,
                Some(if partial { -0.5 } else { -1.0 }),
                if partial { -5.0 } else { -10.0 },
                0,
            );
            exit["price"] = json!(90.0);
            later["snapshot"]["pairs"][0]["fills"] = json!([exit]);
            let mut other = facts(200, 0, 80.0)["snapshot"]["pairs"][0].clone();
            other["symbol"] = json!("B");
            later["snapshot"]["pairs"]
                .as_array_mut()
                .unwrap()
                .push(other);
            let result = run(&mut replay, later.clone());
            if partial {
                assert_eq!(
                    result.unavailable_reason.as_deref(),
                    Some("causal_position_mark_unavailable")
                );
                assert!(result.risk.is_none());
            } else {
                assert_eq!(result.risk.as_ref().unwrap().drawdown, 20.0);
                let seed = replay
                    .steps
                    .iter()
                    .find(|s| s.timestamp == 100 && s.seed)
                    .unwrap();
                assert_eq!(seed.upnl, 0.0);
                assert!(replay.steps.iter().any(|s| s.timestamp == 95 && !s.exposed));
            }
            assert_eq!(replay.ledger.as_ref().unwrap().anchor, initial);
            replay.clear_derived_cache();
            assert_eq!(
                serde_json::to_value(run(&mut replay, later)).unwrap(),
                serde_json::to_value(result).unwrap()
            );
        }
    }

    #[test]
    fn scalar_price_capture_cut_excludes_future_peak_then_admits_it() {
        let mut replay = HslV9Replay::default();
        let mut reference = HslV9Replay::default();
        run(&mut replay, facts(1, 0, 100.0));
        run(&mut reference, facts(1, 0, 100.0));
        let mut input = facts(120001, 0, 100.0);
        input["snapshot"]["pairs"][0]["prices_at"] = json!(59999);
        input["snapshot"]["pairs"][0]["prices"] = json!({"60000":150.0});
        let mut causal = input.clone();
        causal["snapshot"]["pairs"][0]["prices"] = json!({});
        let excluded = run(&mut replay, input.clone());
        assert_eq!(risk(&excluded), risk(&run(&mut reference, causal)));
        assert!(excluded.reasons.contains("post_capture_price"));
        input["snapshot"]["pairs"][0]["prices_at"] = json!(120001);
        let included = run(&mut replay, input.clone());
        assert_eq!(included.risk.as_ref().unwrap().drawdown, 50.0);
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, input)), risk(&included));
    }

    #[test]
    fn regressed_source_cuts_retain_independently_supported_later_facts() {
        let mut replay = HslV9Replay::default();
        run(&mut replay, facts(1, 0, 100.0));
        let mut supported = facts(200, 0, 100.0);
        supported["snapshot"]["pairs"][0]["fills_at"] = json!(200);
        supported["snapshot"]["pairs"][0]["fills"] =
            json!([execution("supported", 150, Some(0.0), -20.0, 0)]);
        run(&mut replay, supported);
        let mut older = facts(300, 0, 100.0);
        older["snapshot"]["pairs"][0]["fills_at"] = json!(100);
        let retained = run(&mut replay, older.clone());
        assert_eq!(retained.risk.as_ref().unwrap().drawdown, 20.0);
        older["snapshot"]["pairs"][0]["fills"] =
            json!([execution("supported", 150, Some(0.0), -20.0, 0)]);
        assert_eq!(run(&mut replay, older.clone()).risk.unwrap().drawdown, 20.0);
        replay.clear_derived_cache();
        assert_eq!(risk(&run(&mut replay, older)), risk(&retained));

        let mut prices = HslV9Replay::default();
        run(&mut prices, facts(1, 0, 100.0));
        let mut supported = facts(60001, 0, 100.0);
        supported["snapshot"]["pairs"][0]["prices"] = json!({"60000":150.0});
        run(&mut prices, supported);
        let mut older = facts(120001, 0, 100.0);
        older["invalidated_price_symbols"] = json!(["A"]);
        older["snapshot"]["pairs"][0]["prices_at"] = json!(59999);
        older["snapshot"]["pairs"][0]["prices"] = json!({"60000":50.0});
        assert_eq!(run(&mut prices, older).risk.unwrap().drawdown, 50.0);
    }

    #[test]
    fn regressed_cut_keeps_a_wholly_known_conflicting_revision_group_intact() {
        let mut replay = HslV9Replay::default();
        run(&mut replay, facts(1, 0, 100.0));
        let variants = json!([
            execution("conflict", 95, Some(0.0), -10.0, 0),
            execution("conflict", 150, Some(0.0), -20.0, 0)
        ]);
        let mut supported = facts(200, 0, 100.0);
        supported["snapshot"]["pairs"][0]["fills_at"] = json!(200);
        supported["snapshot"]["pairs"][0]["fills"] = variants.clone();
        let conflict = run(&mut replay, supported);
        assert!(conflict.reasons.contains("conflicting_identity"));
        assert_eq!(conflict.risk.unwrap().drawdown, 0.0);

        let mut older = facts(300, 0, 100.0);
        older["snapshot"]["pairs"][0]["fills_at"] = json!(100);
        older["snapshot"]["pairs"][0]["fills"] = variants;
        let repeated = run(&mut replay, older.clone());
        assert!(repeated.reasons.contains("conflicting_identity"));
        assert_eq!(repeated.risk.as_ref().unwrap().drawdown, 0.0);
        assert_eq!(
            replay
                .ledger
                .as_ref()
                .unwrap()
                .pairs
                .values()
                .next()
                .unwrap()
                .fills
                .len(),
            2
        );
        replay.clear_derived_cache();
        let refolded = run(&mut replay, older);
        assert!(refolded.reasons.contains("conflicting_identity"));
        assert_eq!(risk(&refolded), risk(&repeated));
    }
}
