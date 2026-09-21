//! Immutable revised HSL snapshot preparation. Boundary uncertainty is not a risk veto.
use crate::hsl_revised_history::{self as history, Fill, History, Position, PositionSide};
use crate::hsl_revised_sum::CurrencySum;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionAnchor {
    pub position_at: i64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub size: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub basis: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub multiplier: f64,
    pub inverse: bool,
    pub pside: PositionSide,
    pub revision: u64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub symbol: String,
    pub position: Position,
    pub position_at: i64,
    pub mark_at: i64,
    pub fills_started_at: Option<i64>,
    pub fills_at: Option<i64>,
    pub prices_at: i64,
    pub fills: Vec<Fill>,
    #[serde(deserialize_with = "crate::hsl_revised_json::prices")]
    pub prices: BTreeMap<i64, f64>,
    pub revisions: [u64; 4],
    pub fills_position_anchor: Option<PositionAnchor>,
}
impl Pair {
    fn anchored(&self) -> bool {
        self.fills_position_anchor.as_ref().is_some_and(|a| {
            a.position_at == self.position_at
                && a.revision == self.revisions[0]
                && a.size == self.position.size
                && a.basis == self.position.basis
                && a.multiplier == self.position.multiplier
                && a.inverse == self.position.inverse
                && a.pside == self.position.pside
        })
    }
}
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Coin,
    Pside,
    Unified,
}
/// Explicitly observed flat coin with an empty retained fill tape. This carries
/// no invented quote; it cannot be used to discard retained history or exposure.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatCoin {
    pub symbol: String,
    pub pside: PositionSide,
    pub position_at: i64,
    pub fills_at: i64,
    pub history_start: i64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    #[serde(default)]
    pub flat_coin: Option<FlatCoin>,
    /// True only for a producer with one execution sequence across every pair.
    #[serde(default)]
    pub global_fill_sequence: bool,
    #[serde(default = "history::default_fills_before_price")]
    pub fills_before_same_time_price: bool,
    pub now: i64,
    pub start: i64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub balance: f64,
    pub balance_at: i64,
    pub config_at: i64,
    pub max_current_age_ms: i64,
    pub mode: Mode,
    pub pside: Option<PositionSide>,
    pub symbol: Option<String>,
    pub pairs: Vec<Pair>,
}
#[derive(Debug, Serialize)]
pub struct PreparedPair {
    pub symbol: String,
    pub pside: PositionSide,
    pub history: History,
    pub fills: Vec<Fill>,
}
#[derive(Debug, Serialize)]
pub struct Consumed {
    pub symbol: String,
    pub pside: PositionSide,
    pub count: usize,
}
#[derive(Debug, Serialize)]
pub struct Boundary {
    pub timestamp: i64,
    pub pnl: f64,
    pub upnl: f64,
    pub consumed: Vec<Consumed>,
}
#[derive(Debug, Serialize)]
pub struct Output {
    pub pairs: Vec<PreparedPair>,
    pub boundaries: Vec<Boundary>,
    pub reasons: BTreeSet<String>,
}
fn side_key(s: PositionSide) -> u8 {
    if s == PositionSide::Long {
        0
    } else {
        1
    }
}
pub(crate) fn select(input: &Input) -> Result<Vec<&Pair>, String> {
    match input.mode {
        Mode::Unified if input.pside.is_some() || input.symbol.is_some() => {
            return Err("unified has no side/symbol selector".into())
        }
        Mode::Coin if input.pside.is_none() || input.symbol.is_none() => {
            return Err("coin requires side and symbol".into())
        }
        Mode::Pside if input.pside.is_none() || input.symbol.is_some() => {
            return Err("pside requires only side".into())
        }
        _ => {}
    }
    let mut keys = BTreeSet::new();
    for p in &input.pairs {
        if !keys.insert((&p.symbol, side_key(p.position.pside))) {
            return Err("duplicate current scope position".into());
        }
    }
    let mut pairs: Vec<_> = input
        .pairs
        .iter()
        .filter(|p| {
            input.pside.is_none_or(|s| s == p.position.pside)
                && input.symbol.as_ref().is_none_or(|s| s == &p.symbol)
        })
        .collect();
    pairs.sort_by_key(|p| (&p.symbol, side_key(p.position.pside)));
    if let Some(flat) = &input.flat_coin {
        if !matches!(input.mode, Mode::Coin)
            || !pairs.is_empty()
            || input.symbol.as_ref() != Some(&flat.symbol)
            || input.pside != Some(flat.pside)
            || flat.history_start != input.start
            || flat.position_at < input.now.saturating_sub(input.max_current_age_ms)
            || flat.position_at > input.now
            || flat.fills_at != flat.position_at
        {
            return Err("invalid explicit flat coin observation".into());
        }
    }
    if matches!(input.mode, Mode::Coin) && pairs.is_empty() && input.flat_coin.is_none() {
        return Err("missing current coin position; absent is not flat".into());
    }
    Ok(pairs)
}
fn latest(fills: &[Fill]) -> Vec<&Fill> {
    let mut versions = BTreeMap::<&str, u64>::new();
    for f in fills {
        versions
            .entry(&f.identity)
            .and_modify(|r| *r = (*r).max(f.revision))
            .or_insert(f.revision);
    }
    fills
        .iter()
        .filter(|f| versions[f.identity.as_str()] == f.revision)
        .collect()
}
fn causal_fills(pair: &Pair, now: i64) -> Vec<Fill> {
    let end = now.min(pair.fills_at.unwrap_or(now));
    let mut versions = BTreeMap::<(&str, u64), Vec<&Fill>>::new();
    for f in &pair.fills {
        versions
            .entry((&f.identity, f.revision))
            .or_default()
            .push(f);
    }
    // A future variant quarantines its entire revision before canonical selection.
    versions
        .values()
        .filter(|v| v.iter().all(|f| f.timestamp <= end))
        .flat_map(|v| v.iter().map(|f| (*f).clone()))
        .collect()
}
pub fn prepare(input: &Input) -> Result<Output, String> {
    if input.start > input.now
        || !input.balance.is_finite()
        || input.balance <= 0.0
        || input.max_current_age_ms < 0
        || input.config_at > input.now
    {
        return Err("invalid minimum HSL snapshot".into());
    }
    let oldest = input.now.saturating_sub(input.max_current_age_ms);
    let fresh = |t| oldest <= t && t <= input.now;
    if !fresh(input.balance_at) {
        return Err("unusable current balance observation".into());
    }
    let selected = select(input)?;
    let mut reasons = BTreeSet::new();
    let mut pairs = Vec::new();
    for p in &selected {
        if !fresh(p.position_at)
            || p.mark_at > input.now
            || (p.position.size != 0.0 && !fresh(p.mark_at))
        {
            return Err("unusable current position/mark observation".into());
        }
        if p.position.size == 0.0 && !fresh(p.mark_at) {
            // Current UPNL is exactly zero for a confirmed-flat pair. Its last
            // factual close still values retained history after a delisting.
            reasons.insert("stale_flat_mark".into());
        }
        if p.prices_at > input.now
            || p.fills_at.is_some_and(|t| t > input.now)
            || p.fills_started_at.is_some_and(|t| t > input.now)
            || p.fills_started_at
                .zip(p.fills_at)
                .is_some_and(|(a, b)| a > b)
        {
            return Err("invalid source capture interval".into());
        }
        if p.fills_started_at.is_none() || p.fills_at.is_none() {
            reasons.insert("fill_capture_unknown".into());
        }
        if p.fills_started_at
            .is_some_and(|t| t < p.position_at || (t == p.position_at && !p.anchored()))
        {
            reasons.insert("fills_before_position".into());
        }
        if p.prices_at < p.mark_at {
            reasons.insert("prices_before_mark".into());
        }
        if p.prices.keys().any(|t| *t > p.prices_at) {
            reasons.insert("post_capture_price".into());
        }
        if p.position_at != input.balance_at || p.mark_at != input.balance_at {
            reasons.insert("snapshot_skew".into());
        }
        for f in latest(&p.fills) {
            if p.position_at < f.timestamp && f.timestamp <= input.now {
                reasons.insert("post_position_fill".into());
            }
            if f.timestamp == p.position_at && !(input.global_fill_sequence && p.anchored()) {
                reasons.insert("position_fill_timestamp_tie".into());
            }
            if f.timestamp > input.now.min(p.fills_at.unwrap_or(input.now)) {
                reasons.insert("post_capture_fill".into());
            }
        }
        let causal = causal_fills(p, input.now);
        let mut fills = history::canonical_fills(
            &causal,
            input.start,
            input.now,
            p.position.pside,
            &mut reasons,
        );
        let limit = p.position_at.min(p.fills_at.unwrap_or(input.now));
        fills.retain(|f| f.timestamp <= limit);
        let prices = p
            .prices
            .iter()
            .filter(|(t, _)| input.start <= **t && **t <= input.now.min(p.prices_at))
            .map(|(t, p)| (*t, *p))
            .collect();
        let h = history::reconstruct(&history::Input {
            fills_before_same_time_price: input.fills_before_same_time_price,
            start: input.start,
            end: input.now,
            position: p.position.clone(),
            fills: fills.clone(),
            prices,
        })?;
        reasons.extend(h.reasons.iter().cloned());
        pairs.push(PreparedPair {
            symbol: p.symbol.clone(),
            pside: p.position.pside,
            history: h,
            fills,
        });
    }
    let mut sizes: Vec<_> = pairs
        .iter()
        .zip(&selected)
        .map(|(h, p)| {
            h.history
                .events
                .first()
                .map_or(p.position.size.abs(), |e| e.before)
        })
        .collect();
    let mut counts = vec![0; pairs.len()];
    let mut cashflows = CurrencySum::new();
    let mut timeline = Vec::new();
    for (pi, p) in pairs.iter().enumerate() {
        timeline.extend(
            p.history
                .events
                .iter()
                .enumerate()
                .map(|(fi, e)| (e.fill.timestamp, pi, fi)),
        );
    }
    timeline.sort_unstable();
    let mut boundaries = Vec::new();
    let mut cursor = 0;
    while cursor < timeline.len() {
        let t = timeline[cursor].0;
        let end = cursor + timeline[cursor..].partition_point(|v| v.0 == t);
        let cohort = &timeline[cursor..end];
        let one_pair = cohort.iter().all(|v| v.1 == cohort[0].1);
        let sequences: BTreeSet<_> = cohort
            .iter()
            .filter_map(|v| pairs[v.1].history.events[v.2].fill.sequence)
            .collect();
        let exact = (one_pair || input.global_fill_sequence) && sequences.len() == cohort.len();
        let mut ordered = cohort.to_vec();
        if exact {
            ordered.sort_by_key(|v| pairs[v.1].history.events[v.2].fill.sequence);
        }
        let groups: Vec<&[(i64, usize, usize)]> = if exact {
            ordered.chunks(1).collect()
        } else {
            vec![&ordered]
        };
        for group in groups {
            let had_exposure = sizes.iter().any(|q| *q != 0.0)
                || group.iter().any(|v| {
                    let e = &pairs[v.1].history.events[v.2];
                    e.before > 0.0 || e.after > 0.0
                });
            for &(_, pi, fi) in group {
                let e = &pairs[pi].history.events[fi];
                sizes[pi] = if fi + 1 == pairs[pi].history.events.len()
                    && pairs[pi]
                        .history
                        .reconciliation
                        .as_ref()
                        .is_some_and(|r| r.after == 0.0)
                {
                    0.0
                } else {
                    e.after
                };
                counts[pi] = fi + 1;
                cashflows.add(e.gross_realized);
                cashflows.add(e.fee);
            }
            if sizes.iter().any(|q| *q != 0.0) {
                continue;
            }
            if !had_exposure {
                continue;
            }
            if selected.iter().any(|p| t > p.position_at) {
                reasons.insert("boundary_after_position_anchor".into());
            }
            // This is the reconciler's estimated path. Observation skew and
            // historical quality remain diagnostics, never a second flat veto.
            let mut mixed = BTreeMap::<usize, BTreeSet<bool>>::new();
            for &(_, pi, fi) in group {
                mixed.entry(pi).or_default().insert(
                    pairs[pi].history.events[fi]
                        .fill
                        .delta
                        .is_some_and(|q| q > 0.0),
                );
            }
            let ambiguous = !exact && mixed.values().any(|v| v.len() > 1);
            if ambiguous {
                reasons.insert("estimated_flat".into());
            }
            let pnl = cashflows.value(&mut reasons);
            boundaries.push(Boundary {
                timestamp: t,
                pnl,
                upnl: 0.0,
                consumed: pairs
                    .iter()
                    .zip(&counts)
                    .map(|(p, n)| Consumed {
                        symbol: p.symbol.clone(),
                        pside: p.pside,
                        count: *n,
                    })
                    .collect(),
            });
        }
        cursor = end;
    }
    Ok(Output {
        pairs,
        boundaries,
        reasons,
    })
}
#[pyfunction]
pub fn hsl_revised_snapshot(input_json: &str) -> PyResult<String> {
    let input: Input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = prepare(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_flat_uses_last_retained_fill_but_residual_exposure_does_not() {
        let value = serde_json::json!({
            "now": 100, "start": 0, "balance": 100.0, "balance_at": 100,
            "config_at": 100, "max_current_age_ms": 10, "mode": "unified",
            "pairs": [{"symbol": "A", "position": {"size": 0.0, "basis": 0.0,
                "mark": 80.0, "multiplier": 1.0, "inverse": false, "pside": "long"},
                "position_at": 100, "mark_at": 100, "prices_at": 100,
                "fills_started_at": 100, "fills_at": 100, "prices": {}, "revisions": [0,0,0,0],
                "fills": [
                    {"identity": "open", "timestamp": 1, "delta": 3.0, "price": 100.0,
                     "realized": 0.0, "fee": 0.0, "revision": 0},
                    {"identity": "partial", "timestamp": 2, "delta": -1.0, "price": 80.0,
                     "realized": -20.0, "fee": 0.0, "revision": 0}]}]
        });
        let mut input: Input = serde_json::from_value(value).unwrap();
        let output = prepare(&input).unwrap();
        assert_eq!(output.boundaries.len(), 1);
        assert_eq!(output.boundaries[0].timestamp, 2);
        assert_eq!(output.boundaries[0].consumed[0].count, 2);
        assert!(output.reasons.contains("current_flat_timestamp_estimate"));
        input.pairs[0].position.size = 2.0;
        input.pairs[0].position.basis = 100.0;
        assert!(prepare(&input).unwrap().boundaries.is_empty());
    }

    #[test]
    fn historical_flat_survives_capture_skew_and_missing_receipts() {
        let value = serde_json::json!({
            "now": 100, "start": 0, "balance": 1000.0, "balance_at": 100,
            "config_at": 100, "max_current_age_ms": 10, "mode": "unified",
            "pairs": [{"symbol": "A", "position": {"size": 1.0, "basis": 100.0,
                "mark": 100.0, "multiplier": 1.0, "inverse": false, "pside": "long"},
                "position_at": 100, "mark_at": 100, "prices_at": 100,
                "fills_started_at": 95, "fills_at": 99, "prices": {}, "revisions": [0,0,0,0],
                "fills": [
                    {"identity": "open", "timestamp": 10, "delta": 10.0, "price": 100.0,
                     "realized": 0.0, "fee": 0.0, "revision": 0},
                    {"identity": "close", "timestamp": 20, "delta": -10.0, "price": 80.0,
                     "realized": -200.0, "fee": 0.0, "revision": 0},
                    {"identity": "reopen", "timestamp": 90, "delta": 1.0, "price": 100.0,
                     "realized": 0.0, "fee": 0.0, "revision": 0}]}]
        });
        let mut input: Input = serde_json::from_value(value).unwrap();
        let output = prepare(&input).unwrap();
        assert!(output.reasons.contains("fills_before_position"));
        assert_eq!(output.boundaries.len(), 1);
        assert_eq!(output.boundaries[0].timestamp, 20);
        for started in [10, 20] {
            input.pairs[0].fills_started_at = Some(started);
            assert_eq!(prepare(&input).unwrap().boundaries.len(), 1);
        }
        input.pairs[0].fills_started_at = None;
        assert_eq!(prepare(&input).unwrap().boundaries.len(), 1);
    }

    #[test]
    fn future_variant_quarantines_whole_revision_without_erasing_older_evidence() {
        let original = Fill {
            identity: "fill".into(),
            timestamp: 10,
            delta: Some(-1.0),
            price: Some(90.0),
            realized: Some(-10.0),
            fee: Some(0.0),
            sequence: None,
            revision: 0,
        };
        let mut expired = original.clone();
        expired.timestamp = -10;
        expired.revision = 1;
        let mut future = expired.clone();
        future.timestamp = 101;
        let p = Pair {
            symbol: "TEST".into(),
            position: Position {
                size: 0.0,
                basis: 0.0,
                mark: 90.0,
                multiplier: 1.0,
                quantity_step: None,
                inverse: false,
                pside: PositionSide::Long,
            },
            position_at: 100,
            mark_at: 100,
            fills_started_at: Some(100),
            fills_at: Some(100),
            prices_at: 100,
            fills: vec![original.clone(), expired, future],
            prices: BTreeMap::new(),
            revisions: [0; 4],
            fills_position_anchor: None,
        };
        assert_eq!(causal_fills(&p, 100), vec![original]);
    }
}
