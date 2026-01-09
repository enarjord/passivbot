//! Hedge overlay (WIP).
//!
//! Implements a reactive market-neutral overlay:
//! - Base is long-only or short-only.
//! - Hedge positions are on the opposite pside.
//! - One-way constraint (v0): never hold long and short simultaneously on the same symbol.
//! - Signed conventions: buy qty/long psize positive; sell qty/short psize negative.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::types::ExchangeParams;
use crate::utils::{calc_new_psize_pprice, cost_to_qty, qty_to_cost, round_, round_dn, round_up};

#[derive(Clone, Debug, PartialEq)]
pub struct HedgePosition {
    pub idx: usize,
    /// Signed size: longs positive, shorts negative.
    pub size: f64,
    /// Average entry price (pprice).
    pub price: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HedgeSymbol {
    pub idx: usize,
    pub bid: f64,
    pub ask: f64,
    pub volume_score: f64,
    pub volatility_score: f64,
    pub exchange_params: ExchangeParams,
    pub effective_min_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesiredBaseOrder {
    pub idx: usize,
    /// Signed qty following internal conventions (buy > 0, sell < 0).
    pub qty: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HedgeAction {
    OpenOrIncrease,
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HedgeOrder {
    pub idx: usize,
    /// Signed qty: positive = buy, negative = sell.
    pub qty: f64,
    pub price: f64,
    pub action: HedgeAction,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HedgeCycleOutput {
    pub orders: Vec<HedgeOrder>,
    /// In one-way mode, base entry orders on these symbols must be gated/canceled this cycle.
    pub gate_base_entries: HashSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HedgeMode {
    /// Base is long-only, hedges are shorts.
    HedgeShortsForLongs,
    /// Base is short-only, hedges are longs.
    HedgeLongsForShorts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HedgeConfig {
    /// 0 disables hedging; 1 targets equal hedge/base exposure; 0.5 targets 50% hedge exposure, etc.
    pub threshold: f64,
    /// Tolerance is absolute exposure band: `tolerance_band = base_twel * tolerance_pct`.
    pub tolerance_pct: f64,
    /// Cap looseness multiplier.
    pub hedge_excess_allowance_pct: f64,
    /// Max number of hedge positions. If 0, default to base side `n_positions`.
    pub max_n_positions: usize,
    /// Minimum fraction of remaining hedge budget to spend per allocation step (avoids churn).
    pub allocation_min_fraction: f64,
    /// Eligible symbols for opening new hedges (e.g. approved_coins.short in HedgeShortsForLongs).
    pub approved_hedge_symbols: Vec<usize>,
    /// v0: one-way only.
    pub one_way: bool,
    /// Hedge direction.
    pub mode: HedgeMode,
}

impl Default for HedgeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            tolerance_pct: 0.05,
            hedge_excess_allowance_pct: 0.20,
            max_n_positions: 0,
            allocation_min_fraction: 0.10,
            approved_hedge_symbols: Vec::new(),
            one_way: true,
            mode: HedgeMode::HedgeShortsForLongs,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HedgeError {
    pub message: String,
}

impl HedgeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

fn market_price(sym: &HedgeSymbol) -> f64 {
    (sym.bid + sym.ask) * 0.5
}

fn effective_min_entry_qty(effective_min_cost: f64, entry_price: f64, ep: &ExchangeParams) -> f64 {
    f64::max(
        ep.min_qty,
        round_up(
            cost_to_qty(effective_min_cost, entry_price, ep.c_mult),
            ep.qty_step,
        ),
    )
}

/// Exposure uses mark-to-market price for true market-neutral hedging.
/// This means exposure reflects current risk, not historical entry cost.
fn position_exposure(pos: &HedgePosition, sym: &HedgeSymbol, balance: f64) -> f64 {
    if pos.size == 0.0 {
        return 0.0;
    }
    qty_to_cost(pos.size.abs(), market_price(sym), sym.exchange_params.c_mult) / balance
}

/// Borda ranking: low volatility + high volume. Lower score is better.
fn rank_assets_borda(eligible: &[HedgeSymbol]) -> Vec<HedgeSymbol> {
    if eligible.is_empty() {
        return vec![];
    }
    let mut by_vol = eligible.to_vec();
    by_vol.sort_by(|a, b| a.volatility_score.partial_cmp(&b.volatility_score).unwrap());
    let mut by_volume = eligible.to_vec();
    by_volume.sort_by(|a, b| b.volume_score.partial_cmp(&a.volume_score).unwrap());

    let mut vol_rank: HashMap<usize, usize> = HashMap::new();
    for (i, s) in by_vol.iter().enumerate() {
        vol_rank.insert(s.idx, i);
    }
    let mut volume_rank: HashMap<usize, usize> = HashMap::new();
    for (i, s) in by_volume.iter().enumerate() {
        volume_rank.insert(s.idx, i);
    }
    let mut scored: Vec<(usize, HedgeSymbol)> = eligible
        .iter()
        .map(|s| {
            let vr = *vol_rank.get(&s.idx).unwrap_or(&eligible.len());
            let pr = *volume_rank.get(&s.idx).unwrap_or(&eligible.len());
            (vr + pr, s.clone())
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0));
    scored.into_iter().map(|(_, s)| s).collect()
}

fn underwaterness_long(pprice: f64, sym: &HedgeSymbol) -> f64 {
    if pprice <= 0.0 {
        return 0.0;
    }
    1.0 - market_price(sym) / pprice
}

fn underwaterness_short(pprice: f64, sym: &HedgeSymbol) -> f64 {
    if pprice <= 0.0 {
        return 0.0;
    }
    market_price(sym) / pprice - 1.0
}

fn validate_symbols_contiguous(symbols: &[HedgeSymbol]) -> Result<(), HedgeError> {
    for (i, s) in symbols.iter().enumerate() {
        if s.idx != i {
            return Err(HedgeError::new(format!(
                "symbols must be indexed by idx (symbols[i].idx == i); got symbols[{}].idx={}",
                i, s.idx
            )));
        }
        if !(s.bid.is_finite() && s.ask.is_finite() && s.bid > 0.0 && s.ask > 0.0) {
            return Err(HedgeError::new(format!(
                "invalid order book for idx {}: bid={} ask={}",
                s.idx, s.bid, s.ask
            )));
        }
        if !(s.effective_min_cost.is_finite() && s.effective_min_cost > 0.0) {
            return Err(HedgeError::new(format!(
                "non-finite effective_min_cost for idx {}: {}",
                s.idx, s.effective_min_cost
            )));
        }
    }
    Ok(())
}

fn validate_config(cfg: &HedgeConfig) -> Result<(), HedgeError> {
    for (name, v) in [
        ("threshold", cfg.threshold),
        ("tolerance_pct", cfg.tolerance_pct),
        ("hedge_excess_allowance_pct", cfg.hedge_excess_allowance_pct),
        ("allocation_min_fraction", cfg.allocation_min_fraction),
    ] {
        if !v.is_finite() {
            return Err(HedgeError::new(format!(
                "hedge config {} is non-finite: {}",
                name, v
            )));
        }
    }
    if cfg.threshold < 0.0 {
        return Err(HedgeError::new("hedge.threshold must be >= 0.0"));
    }
    if cfg.tolerance_pct < 0.0 {
        return Err(HedgeError::new("hedge.tolerance_pct must be >= 0.0"));
    }
    if cfg.hedge_excess_allowance_pct < 0.0 {
        return Err(HedgeError::new(
            "hedge.hedge_excess_allowance_pct must be >= 0.0",
        ));
    }
    if cfg.allocation_min_fraction <= 0.0 || cfg.allocation_min_fraction > 1.0 {
        return Err(HedgeError::new(
            "hedge.allocation_min_fraction must be in (0.0, 1.0]",
        ));
    }
    Ok(())
}

fn base_entry_wants_symbol(mode: HedgeMode, desired_qty: f64) -> bool {
    match mode {
        HedgeMode::HedgeShortsForLongs => desired_qty > 0.0,
        HedgeMode::HedgeLongsForShorts => desired_qty < 0.0,
    }
}

fn hedge_entry_price(mode: HedgeMode, sym: &HedgeSymbol) -> f64 {
    match mode {
        HedgeMode::HedgeShortsForLongs => sym.ask, // sell to open/increase short (maker)
        HedgeMode::HedgeLongsForShorts => sym.bid, // buy to open/increase long (maker)
    }
}

fn hedge_close_price(mode: HedgeMode, sym: &HedgeSymbol) -> f64 {
    match mode {
        HedgeMode::HedgeShortsForLongs => sym.bid, // buy to close short (maker)
        HedgeMode::HedgeLongsForShorts => sym.ask, // sell to close long (maker)
    }
}

fn hedge_close_qty_for_full_close(mode: HedgeMode, pos: &HedgePosition, sym: &HedgeSymbol) -> f64 {
    let qty_step = sym.exchange_params.qty_step;
    let abs_size = pos.size.abs();
    let rounded = round_(abs_size, qty_step);
    if rounded == 0.0 { abs_size } else { rounded }.copysign(match mode {
        HedgeMode::HedgeShortsForLongs => 1.0,  // buy
        HedgeMode::HedgeLongsForShorts => -1.0, // sell
    })
}

/// Compute hedge orders and collision gating.
///
/// - `symbols` must be indexed by `idx` (i.e. `symbols[i].idx == i`).
/// - `positions_long/short` are the current open positions per pside (signed sizes).
/// - `desired_base_orders` are the base ideal orders for this cycle (used for collisions/exclusions).
/// - `base_twel` and `base_n_positions` are taken from the base pside's bot params.
pub fn compute_hedge_cycle(
    cfg: &HedgeConfig,
    symbols: &[HedgeSymbol],
    positions_long: &[HedgePosition],
    positions_short: &[HedgePosition],
    desired_base_orders: &[DesiredBaseOrder],
    balance: f64,
    base_twel: f64,
    base_n_positions: usize,
) -> Result<HedgeCycleOutput, HedgeError> {
    validate_config(cfg)?;
    validate_symbols_contiguous(symbols)?;

    if !(balance.is_finite() && balance > 0.0) {
        return Err(HedgeError::new(format!(
            "balance must be > 0 and finite; got {}",
            balance
        )));
    }
    if !(base_twel.is_finite() && base_twel >= 0.0) {
        return Err(HedgeError::new(format!(
            "base_twel must be >= 0 and finite; got {}",
            base_twel
        )));
    }

    let (base_positions, hedge_positions) = match cfg.mode {
        HedgeMode::HedgeShortsForLongs => (positions_long, positions_short),
        HedgeMode::HedgeLongsForShorts => (positions_short, positions_long),
    };

    // One-way invariant (v0).
    if cfg.one_way {
        let mut long_set: HashSet<usize> = HashSet::new();
        for p in positions_long.iter().filter(|p| p.size != 0.0) {
            long_set.insert(p.idx);
        }
        for p in positions_short.iter().filter(|p| p.size != 0.0) {
            if long_set.contains(&p.idx) {
                return Err(HedgeError::new(format!(
                    "one-way violation: both long and short positions open on idx {}",
                    p.idx
                )));
            }
        }
    }

    let mut out = HedgeCycleOutput::default();

    let base_open: HashSet<usize> = base_positions
        .iter()
        .filter(|p| p.size != 0.0)
        .map(|p| p.idx)
        .collect();

    // Base entry intents for one-way exclusions and collision gating.
    let base_entry_intents: HashSet<usize> = desired_base_orders
        .iter()
        .filter(|o| base_entry_wants_symbol(cfg.mode, o.qty))
        .map(|o| o.idx)
        .collect();

    let hedge_map: HashMap<usize, HedgePosition> = hedge_positions
        .iter()
        .filter(|p| p.size != 0.0)
        .cloned()
        .map(|p| (p.idx, p))
        .collect();

    // One-way collisions: if base wants to open on idx where a hedge exists, close hedge and gate base entry.
    if cfg.one_way {
        for idx in &base_entry_intents {
            if let Some(pos) = hedge_map.get(idx) {
                if *idx >= symbols.len() {
                    return Err(HedgeError::new(format!(
                        "missing symbol market data for collision close idx {}",
                        idx
                    )));
                }
                let sym = &symbols[*idx];
                let close_qty = hedge_close_qty_for_full_close(cfg.mode, pos, sym);
                if close_qty != 0.0 {
                    out.orders.push(HedgeOrder {
                        idx: *idx,
                        qty: close_qty,
                        price: hedge_close_price(cfg.mode, sym),
                        action: HedgeAction::Close,
                        reason: "collision_with_base".to_string(),
                    });
                    out.gate_base_entries.insert(*idx);
                }
            }
        }
    }

    // Active hedges excluding collisions we are closing.
    let mut active_hedges: Vec<HedgePosition> = hedge_positions
        .iter()
        .filter(|p| p.size != 0.0 && !out.gate_base_entries.contains(&p.idx))
        .cloned()
        .collect();

    // Compute exposures.
    let mut gross_base: f64 = 0.0;
    for p in base_positions.iter().filter(|p| p.size != 0.0) {
        let sym = &symbols[p.idx];
        gross_base += position_exposure(p, sym, balance);
    }
    let mut gross_hedge: f64 = 0.0;
    for p in active_hedges.iter().filter(|p| p.size != 0.0) {
        let sym = &symbols[p.idx];
        gross_hedge += position_exposure(p, sym, balance);
    }

    let target_hedge = gross_base * cfg.threshold;
    let tolerance_band = base_twel * cfg.tolerance_pct;
    let diff = gross_hedge - target_hedge;
    if diff.abs() <= tolerance_band + 1e-12 {
        return Ok(out);
    }

    // Determine per-position cap.
    let max_positions_cfg = if cfg.max_n_positions == 0 {
        base_n_positions
    } else {
        cfg.max_n_positions
    };
    let max_positions = max_positions_cfg.max(1);
    let cap_exposure =
        (base_twel * cfg.threshold / max_positions as f64) * (1.0 + cfg.hedge_excess_allowance_pct);

    // Reduce hedge exposure: close least underwater first (full close).
    if diff > tolerance_band {
        // least underwater first
        active_hedges.sort_by(|a, b| {
            let sa = &symbols[a.idx];
            let sb = &symbols[b.idx];
            let ua = match cfg.mode {
                HedgeMode::HedgeShortsForLongs => underwaterness_short(a.price, sa),
                HedgeMode::HedgeLongsForShorts => underwaterness_long(a.price, sa),
            };
            let ub = match cfg.mode {
                HedgeMode::HedgeShortsForLongs => underwaterness_short(b.price, sb),
                HedgeMode::HedgeLongsForShorts => underwaterness_long(b.price, sb),
            };
            ua.partial_cmp(&ub).unwrap()
        });

        for pos in active_hedges.iter() {
            if gross_hedge <= target_hedge + tolerance_band {
                break;
            }
            let sym = &symbols[pos.idx];
            let close_qty = hedge_close_qty_for_full_close(cfg.mode, pos, sym);
            if close_qty == 0.0 {
                continue;
            }
            out.orders.push(HedgeOrder {
                idx: pos.idx,
                qty: close_qty,
                price: hedge_close_price(cfg.mode, sym),
                action: HedgeAction::Close,
                reason: "rebalance_reduce".to_string(),
            });
            gross_hedge -= position_exposure(pos, sym, balance);
        }
        return Ok(out);
    }

    // Add/open hedges if underexposed.
    let mut remaining_budget_notional = (target_hedge - gross_hedge).max(0.0) * balance;
    if remaining_budget_notional <= 0.0 {
        return Ok(out);
    }

    // Build eligible symbols list:
    // - restricted to approved_hedge_symbols for *opening*
    // - exclude base open symbols and base entry intents (one-way)
    let approved_set: HashSet<usize> = cfg.approved_hedge_symbols.iter().copied().collect();
    let mut eligible: Vec<HedgeSymbol> = Vec::new();
    for &idx in &cfg.approved_hedge_symbols {
        if idx >= symbols.len() {
            continue;
        }
        if cfg.one_way && (base_open.contains(&idx) || base_entry_intents.contains(&idx)) {
            continue;
        }
        eligible.push(symbols[idx].clone());
    }
    let ranked = rank_assets_borda(&eligible);

    // Working hedge state (simulate in-cycle allocations).
    let mut hedge_state: HashMap<usize, HedgePosition> = HashMap::new();
    for p in active_hedges.iter().cloned() {
        hedge_state.insert(p.idx, p);
    }

    // Bootstrap: open minimum-size hedges on as many distinct eligible symbols as possible.
    let mut open_count = hedge_state.len();
    for sym in ranked.iter() {
        if remaining_budget_notional <= 0.0 {
            break;
        }
        if open_count >= max_positions {
            break;
        }
        if hedge_state.contains_key(&sym.idx) {
            continue;
        }
        let entry_price = hedge_entry_price(cfg.mode, sym);
        let min_qty =
            effective_min_entry_qty(sym.effective_min_cost, entry_price, &sym.exchange_params);
        let min_cost = qty_to_cost(min_qty, entry_price, sym.exchange_params.c_mult);
        let cap_notional = cap_exposure * balance;
        if cap_notional + 1e-12 < min_cost {
            continue;
        }
        if remaining_budget_notional + 1e-12 < min_cost {
            continue;
        }
        let signed_qty = match cfg.mode {
            HedgeMode::HedgeShortsForLongs => -min_qty,
            HedgeMode::HedgeLongsForShorts => min_qty,
        };
        out.orders.push(HedgeOrder {
            idx: sym.idx,
            qty: signed_qty,
            price: entry_price,
            action: HedgeAction::OpenOrIncrease,
            reason: "rebalance_open_min".to_string(),
        });
        hedge_state.insert(
            sym.idx,
            HedgePosition {
                idx: sym.idx,
                size: signed_qty,
                price: entry_price,
            },
        );
        remaining_budget_notional -= min_cost;
        open_count += 1;
    }

    // Allocation: add to most underwater hedges, chunked to avoid churn.
    // Aggregate qty per idx into one order per symbol per cycle (deterministic).
    let mut qty_by_idx: HashMap<usize, f64> = HashMap::new();

    // Safety: cap the number of allocations.
    let mut iter: usize = 0;
    while remaining_budget_notional > 0.0 {
        iter += 1;
        if iter > 200 {
            break;
        }
        let target_hedge_now = gross_base * cfg.threshold;
        let current_gross_hedge: f64 = hedge_state
            .values()
            .map(|p| position_exposure(p, &symbols[p.idx], balance))
            .sum();
        let need = (target_hedge_now - current_gross_hedge).max(0.0) * balance;
        if need <= tolerance_band * balance {
            break;
        }
        if remaining_budget_notional <= 0.0 {
            break;
        }

        // Choose most underwater hedge with remaining cap room.
        let mut best: Option<(f64, usize)> = None;
        for (idx, pos) in hedge_state.iter() {
            if pos.size == 0.0 {
                continue;
            }
            if !approved_set.contains(idx) {
                continue;
            }
            let sym = &symbols[*idx];
            let pos_exposure = position_exposure(pos, sym, balance);
            if pos_exposure >= cap_exposure - 1e-12 {
                continue;
            }
            let u = match cfg.mode {
                HedgeMode::HedgeShortsForLongs => underwaterness_short(pos.price, sym),
                HedgeMode::HedgeLongsForShorts => underwaterness_long(pos.price, sym),
            };
            best = match best {
                None => Some((u, *idx)),
                Some((bu, bidx)) => {
                    if u > bu + 1e-12 {
                        Some((u, *idx))
                    } else if (u - bu).abs() <= 1e-12 && *idx < bidx {
                        Some((u, *idx))
                    } else {
                        Some((bu, bidx))
                    }
                }
            };
        }
        let Some((_u, idx)) = best else { break };

        let sym = &symbols[idx];
        let entry_price = hedge_entry_price(cfg.mode, sym);
        let min_qty =
            effective_min_entry_qty(sym.effective_min_cost, entry_price, &sym.exchange_params);
        let min_cost = qty_to_cost(min_qty, entry_price, sym.exchange_params.c_mult);

        let pos = hedge_state.get(&idx).cloned().unwrap_or(HedgePosition {
            idx,
            size: 0.0,
            price: 0.0,
        });
        let current_exposure = position_exposure(&pos, sym, balance);
        let remaining_cap_notional = ((cap_exposure - current_exposure).max(0.0)) * balance;
        if remaining_cap_notional + 1e-12 < min_cost {
            break;
        }

        let chunk = (cfg.allocation_min_fraction * remaining_budget_notional)
            .max(min_cost)
            .min(need)
            .min(remaining_cap_notional)
            .min(remaining_budget_notional);
        if chunk + 1e-12 < min_cost {
            break;
        }

        let mut qty = cost_to_qty(chunk, entry_price, sym.exchange_params.c_mult);
        qty = round_dn(qty, sym.exchange_params.qty_step);
        if qty + 1e-12 < min_qty {
            qty = min_qty;
        }
        if qty <= 0.0 {
            break;
        }
        let signed_qty = match cfg.mode {
            HedgeMode::HedgeShortsForLongs => -qty,
            HedgeMode::HedgeLongsForShorts => qty,
        };
        let cost_after = qty_to_cost(qty, entry_price, sym.exchange_params.c_mult);
        if cost_after + 1e-12 < min_cost {
            break;
        }
        if cost_after > remaining_budget_notional + 1e-12 {
            break;
        }

        // Update aggregate.
        *qty_by_idx.entry(idx).or_insert(0.0) += signed_qty;

        // Update simulated psize/pprice.
        let (new_psize, new_pprice) = calc_new_psize_pprice(
            pos.size,
            pos.price,
            signed_qty,
            entry_price,
            sym.exchange_params.qty_step,
        );
        hedge_state.insert(
            idx,
            HedgePosition {
                idx,
                size: new_psize,
                price: new_pprice,
            },
        );
        remaining_budget_notional -= cost_after;
    }

    for (idx, qty) in qty_by_idx {
        if qty == 0.0 {
            continue;
        }
        let sym = &symbols[idx];
        let entry_price = hedge_entry_price(cfg.mode, sym);
        out.orders.push(HedgeOrder {
            idx,
            qty,
            price: entry_price,
            action: HedgeAction::OpenOrIncrease,
            reason: "rebalance_add".to_string(),
        });
    }

    Ok(out)
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(idx: usize, bid: f64, ask: f64, vol: f64, vola: f64, min_cost: f64) -> HedgeSymbol {
        HedgeSymbol {
            idx,
            bid,
            ask,
            volume_score: vol,
            volatility_score: vola,
            exchange_params: ExchangeParams {
                qty_step: 0.01,
                price_step: 0.01,
                min_qty: 0.01,
                min_cost,
                c_mult: 1.0,
            },
            effective_min_cost: min_cost,
        }
    }

    #[test]
    fn test_rank_borda_prefers_low_vol_high_volume() {
        let a = sym(0, 10.0, 10.0, 1000.0, 0.1, 1.0);
        let b = sym(1, 10.0, 10.0, 900.0, 0.2, 1.0);
        let c = sym(2, 10.0, 10.0, 800.0, 0.05, 1.0);
        let ranked = rank_assets_borda(&[a.clone(), b.clone(), c.clone()]);
        assert_eq!(ranked[0].idx, 0); // best volume, decent vol
        assert_eq!(ranked[1].idx, 2); // best vol
    }

    #[test]
    fn test_no_action_within_tolerance() {
        let cfg = HedgeConfig {
            threshold: 1.0,
            tolerance_pct: 0.10,
            approved_hedge_symbols: vec![1],
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1.0, 1.0, 1.0),
        ];
        let longs = vec![HedgePosition {
            idx: 0,
            size: 1.0,
            price: 10.0,
        }];
        let shorts = vec![HedgePosition {
            idx: 1,
            size: -1.0,
            price: 10.0,
        }];
        let res = compute_hedge_cycle(&cfg, &symbols, &longs, &shorts, &[], 100.0, 1.5, 1).unwrap();
        assert!(res.orders.is_empty() && res.gate_base_entries.is_empty());
    }

    #[test]
    fn test_add_shorts_when_underhedged() {
        let cfg = HedgeConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_hedge_symbols: vec![1, 2],
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1000.0, 0.1, 1.0),
            sym(2, 10.0, 10.0, 900.0, 0.2, 1.0),
        ];
        let longs = vec![HedgePosition {
            idx: 0,
            size: 10.0,
            price: 10.0,
        }]; // exposure = 1.0
        let shorts: Vec<HedgePosition> = vec![]; // none
        let res = compute_hedge_cycle(&cfg, &symbols, &longs, &shorts, &[], 100.0, 1.5, 1).unwrap();
        assert!(res.orders.iter().any(|o| o.qty < 0.0));
    }

    #[test]
    fn test_reduce_shorts_when_overhedged() {
        let cfg = HedgeConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_hedge_symbols: vec![1],
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1.0, 1.0, 1.0),
        ];
        let longs = vec![HedgePosition {
            idx: 0,
            size: 5.0,
            price: 10.0,
        }]; // exposure 0.5
        let shorts = vec![HedgePosition {
            idx: 1,
            size: -10.0,
            price: 10.0,
        }]; // exposure 1.0
        let res = compute_hedge_cycle(&cfg, &symbols, &longs, &shorts, &[], 100.0, 1.5, 1).unwrap();
        assert!(res.orders.iter().any(|o| o.qty > 0.0));
    }

    #[test]
    fn test_collision_gates_base_entry_and_closes_hedge() {
        let cfg = HedgeConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_hedge_symbols: vec![0],
            ..Default::default()
        };
        let symbols = vec![sym(0, 10.0, 10.0, 1.0, 1.0, 1.0)];
        let longs = vec![HedgePosition {
            idx: 0,
            size: 10.0,
            price: 10.0,
        }];
        let shorts = vec![HedgePosition {
            idx: 0,
            size: -1.0,
            price: 10.0,
        }];
        let desired = vec![DesiredBaseOrder { idx: 0, qty: 1.0 }];
        let res =
            compute_hedge_cycle(&cfg, &symbols, &longs, &shorts, &desired, 100.0, 1.5, 1).unwrap();
        assert!(res.gate_base_entries.contains(&0));
        assert!(res
            .orders
            .iter()
            .any(|o| o.qty > 0.0 && o.action == HedgeAction::Close));
    }

    #[test]
    fn test_short_only_mode_adds_longs() {
        let cfg = HedgeConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_hedge_symbols: vec![1],
            mode: HedgeMode::HedgeLongsForShorts,
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1.0, 1.0, 1.0),
        ];
        let shorts = vec![HedgePosition {
            idx: 0,
            size: -10.0,
            price: 10.0,
        }]; // base exposure 1.0
        let hedges: Vec<HedgePosition> = vec![];
        let res =
            compute_hedge_cycle(&cfg, &symbols, &hedges, &shorts, &[], 100.0, 1.5, 1).unwrap();
        assert!(res
            .orders
            .iter()
            .any(|o| o.qty > 0.0 && o.action == HedgeAction::OpenOrIncrease));
    }
}
