//! Mirror overlay (market-neutral hedging).
//!
//! Implements a reactive market-neutral overlay:
//! - Base is long-only or short-only.
//! - Mirror positions are on the opposite pside, mirroring base exposure.
//! - One-way constraint (v0): never hold long and short simultaneously on the same symbol.
//! - Signed conventions: buy qty/long psize positive; sell qty/short psize negative.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::types::ExchangeParams;
use crate::utils::{calc_new_psize_pprice, cost_to_qty, qty_to_cost, round_, round_dn, round_up};

#[derive(Clone, Debug, PartialEq)]
pub struct MirrorPosition {
    pub idx: usize,
    /// Signed size: longs positive, shorts negative.
    pub size: f64,
    /// Average entry price (pprice).
    pub price: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MirrorSymbol {
    pub idx: usize,
    pub bid: f64,
    pub ask: f64,
    pub volume_score: f64,
    pub volatility_score: f64,
    pub exchange_params: ExchangeParams,
    pub effective_min_cost: f64,
    /// EMA upper band for this symbol (used for EMA-based gating).
    pub ema_upper: f64,
    /// EMA lower band for this symbol (used for EMA-based gating).
    pub ema_lower: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesiredBaseOrder {
    pub idx: usize,
    /// Signed qty following internal conventions (buy > 0, sell < 0).
    pub qty: f64,
    /// Order price, used to compute exposure from imminent orders.
    pub price: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MirrorAction {
    OpenOrIncrease,
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MirrorOrder {
    pub idx: usize,
    /// Signed qty: positive = buy, negative = sell.
    pub qty: f64,
    pub price: f64,
    pub action: MirrorAction,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MirrorCycleOutput {
    pub orders: Vec<MirrorOrder>,
    /// In one-way mode, base entry orders on these symbols must be gated/canceled this cycle.
    pub gate_base_entries: HashSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirrorMode {
    /// Base is long-only, hedges are shorts.
    MirrorShortsForLongs,
    /// Base is short-only, hedges are longs.
    MirrorLongsForShorts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirrorConfig {
    /// 0 disables hedging; 1 targets equal hedge/base exposure; 0.5 targets 50% hedge exposure, etc.
    pub threshold: f64,
    /// Tolerance is absolute exposure band: `tolerance_band = base_twel * tolerance_pct`.
    pub tolerance_pct: f64,
    /// Cap looseness multiplier.
    pub mirror_excess_allowance_pct: f64,
    /// Max number of hedge positions. If 0, default to base side `n_positions`.
    pub max_n_positions: usize,
    /// Minimum fraction of remaining hedge budget to spend per allocation step (avoids churn).
    pub allocation_min_fraction: f64,
    /// Eligible symbols for opening new hedges (e.g. approved_coins.short in MirrorShortsForLongs).
    pub approved_mirror_symbols: Vec<usize>,
    /// Hedge direction.
    pub mode: MirrorMode,
    /// EMA distance for opening new mirror positions (0 = disabled, no EMA gating).
    /// For MirrorShortsForLongs: only open shorts when price > ema_upper * (1 + ema_dist_entry)
    /// For MirrorLongsForShorts: only open longs when price < ema_lower * (1 - ema_dist_entry)
    /// E.g., 0.01 = require price to be 1% beyond EMA band before opening mirror position.
    #[serde(default)]
    pub ema_dist_entry: f64,
}

impl Default for MirrorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            tolerance_pct: 0.05,
            mirror_excess_allowance_pct: 0.20,
            max_n_positions: 0,
            allocation_min_fraction: 0.10,
            approved_mirror_symbols: Vec::new(),
            mode: MirrorMode::MirrorShortsForLongs,
            ema_dist_entry: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MirrorError {
    pub message: String,
}

impl MirrorError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

fn market_price(sym: &MirrorSymbol) -> f64 {
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
fn position_exposure(pos: &MirrorPosition, sym: &MirrorSymbol, balance: f64) -> f64 {
    if pos.size == 0.0 {
        return 0.0;
    }
    qty_to_cost(
        pos.size.abs(),
        market_price(sym),
        sym.exchange_params.c_mult,
    ) / balance
}

/// Borda ranking: low volatility + high volume. Lower score is better.
fn rank_assets_borda(eligible: &[MirrorSymbol]) -> Vec<MirrorSymbol> {
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
    let mut scored: Vec<(usize, MirrorSymbol)> = eligible
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

fn underwaterness_long(pprice: f64, sym: &MirrorSymbol) -> f64 {
    if pprice <= 0.0 {
        return 0.0;
    }
    1.0 - market_price(sym) / pprice
}

fn underwaterness_short(pprice: f64, sym: &MirrorSymbol) -> f64 {
    if pprice <= 0.0 {
        return 0.0;
    }
    market_price(sym) / pprice - 1.0
}

/// Check if price satisfies EMA gating condition for opening new mirror positions.
/// - `ema_dist_entry = 0.0`: gate exactly at the EMA band
/// - Positive values: require price extended beyond the EMA band
/// - Negative values: allow entries inside the EMA band
/// - To disable gating, use a large negative value (e.g., -1.0)
fn ema_gate_allows_entry(mode: MirrorMode, sym: &MirrorSymbol, ema_dist_entry: f64) -> bool {
    // If EMA bands are not available (0.0), allow entry
    if sym.ema_upper <= 0.0 || sym.ema_lower <= 0.0 {
        return true;
    }

    let price = market_price(sym);
    match mode {
        // For shorting: only open when price is ABOVE upper EMA band (extended upward)
        MirrorMode::MirrorShortsForLongs => {
            let threshold = sym.ema_upper * (1.0 + ema_dist_entry);
            price >= threshold
        }
        // For longing: only open when price is BELOW lower EMA band (extended downward)
        MirrorMode::MirrorLongsForShorts => {
            let threshold = sym.ema_lower * (1.0 - ema_dist_entry);
            price <= threshold
        }
    }
}

fn validate_symbols_contiguous(symbols: &[MirrorSymbol]) -> Result<(), MirrorError> {
    for (i, s) in symbols.iter().enumerate() {
        if s.idx != i {
            return Err(MirrorError::new(format!(
                "symbols must be indexed by idx (symbols[i].idx == i); got symbols[{}].idx={}",
                i, s.idx
            )));
        }
        if !(s.bid.is_finite() && s.ask.is_finite() && s.bid > 0.0 && s.ask > 0.0) {
            return Err(MirrorError::new(format!(
                "invalid order book for idx {}: bid={} ask={}",
                s.idx, s.bid, s.ask
            )));
        }
        if !(s.effective_min_cost.is_finite() && s.effective_min_cost > 0.0) {
            return Err(MirrorError::new(format!(
                "non-finite effective_min_cost for idx {}: {}",
                s.idx, s.effective_min_cost
            )));
        }
    }
    Ok(())
}

fn validate_config(cfg: &MirrorConfig) -> Result<(), MirrorError> {
    for (name, v) in [
        ("threshold", cfg.threshold),
        ("tolerance_pct", cfg.tolerance_pct),
        (
            "mirror_excess_allowance_pct",
            cfg.mirror_excess_allowance_pct,
        ),
        ("allocation_min_fraction", cfg.allocation_min_fraction),
        ("ema_dist_entry", cfg.ema_dist_entry),
    ] {
        if !v.is_finite() {
            return Err(MirrorError::new(format!(
                "mirror config {} is non-finite: {}",
                name, v
            )));
        }
    }
    if cfg.threshold < 0.0 {
        return Err(MirrorError::new("mirror.threshold must be >= 0.0"));
    }
    if cfg.tolerance_pct < 0.0 {
        return Err(MirrorError::new("mirror.tolerance_pct must be >= 0.0"));
    }
    if cfg.mirror_excess_allowance_pct < 0.0 {
        return Err(MirrorError::new(
            "mirror.mirror_excess_allowance_pct must be >= 0.0",
        ));
    }
    if cfg.allocation_min_fraction <= 0.0 || cfg.allocation_min_fraction > 1.0 {
        return Err(MirrorError::new(
            "mirror.allocation_min_fraction must be in (0.0, 1.0]",
        ));
    }
    Ok(())
}

fn base_entry_wants_symbol(mode: MirrorMode, desired_qty: f64) -> bool {
    match mode {
        MirrorMode::MirrorShortsForLongs => desired_qty > 0.0,
        MirrorMode::MirrorLongsForShorts => desired_qty < 0.0,
    }
}

fn hedge_entry_price(mode: MirrorMode, sym: &MirrorSymbol) -> f64 {
    match mode {
        MirrorMode::MirrorShortsForLongs => sym.ask, // sell to open/increase short (maker)
        MirrorMode::MirrorLongsForShorts => sym.bid, // buy to open/increase long (maker)
    }
}

fn hedge_close_price(mode: MirrorMode, sym: &MirrorSymbol) -> f64 {
    match mode {
        MirrorMode::MirrorShortsForLongs => sym.bid, // buy to close short (maker)
        MirrorMode::MirrorLongsForShorts => sym.ask, // sell to close long (maker)
    }
}

fn hedge_close_qty_for_full_close(
    mode: MirrorMode,
    pos: &MirrorPosition,
    sym: &MirrorSymbol,
) -> f64 {
    let qty_step = sym.exchange_params.qty_step;
    let abs_size = pos.size.abs();
    let rounded = round_(abs_size, qty_step);
    if rounded == 0.0 { abs_size } else { rounded }.copysign(match mode {
        MirrorMode::MirrorShortsForLongs => 1.0,  // buy
        MirrorMode::MirrorLongsForShorts => -1.0, // sell
    })
}

/// Compute hedge orders and collision gating.
///
/// - `symbols` must be indexed by `idx` (i.e. `symbols[i].idx == i`).
/// - `positions_long/short` are the current open positions per pside (signed sizes).
/// - `desired_base_orders` are the base ideal orders for this cycle (used for collisions/exclusions).
/// - `base_twel` and `base_n_positions` are taken from the base pside's bot params.
/// - `hedge_mode` controls whether simultaneous long/short positions are allowed on same symbol.
///   If false (one-way mode), mirror enforces single-side-per-symbol constraint.
/// - `debug` enables debug output to trace mirror decisions.
pub fn compute_mirror_cycle(
    cfg: &MirrorConfig,
    symbols: &[MirrorSymbol],
    positions_long: &[MirrorPosition],
    positions_short: &[MirrorPosition],
    desired_base_orders: &[DesiredBaseOrder],
    balance: f64,
    base_twel: f64,
    base_n_positions: usize,
    hedge_mode: bool,
    debug: bool,
) -> Result<MirrorCycleOutput, MirrorError> {
    validate_config(cfg)?;
    validate_symbols_contiguous(symbols)?;

    if !(balance.is_finite() && balance > 0.0) {
        return Err(MirrorError::new(format!(
            "balance must be > 0 and finite; got {}",
            balance
        )));
    }
    if !(base_twel.is_finite() && base_twel >= 0.0) {
        return Err(MirrorError::new(format!(
            "base_twel must be >= 0 and finite; got {}",
            base_twel
        )));
    }

    let (base_positions, hedge_positions) = match cfg.mode {
        MirrorMode::MirrorShortsForLongs => (positions_long, positions_short),
        MirrorMode::MirrorLongsForShorts => (positions_short, positions_long),
    };

    // One-way invariant: if hedge_mode is false, we enforce single-side-per-symbol.
    if !hedge_mode {
        let mut long_set: HashSet<usize> = HashSet::new();
        for p in positions_long.iter().filter(|p| p.size != 0.0) {
            long_set.insert(p.idx);
        }
        for p in positions_short.iter().filter(|p| p.size != 0.0) {
            if long_set.contains(&p.idx) {
                return Err(MirrorError::new(format!(
                    "one-way violation: both long and short positions open on idx {}",
                    p.idx
                )));
            }
        }
    }

    let mut out = MirrorCycleOutput::default();

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

    let hedge_map: HashMap<usize, MirrorPosition> = hedge_positions
        .iter()
        .filter(|p| p.size != 0.0)
        .cloned()
        .map(|p| (p.idx, p))
        .collect();

    // One-way collisions: if base wants to open on idx where a hedge exists, close hedge and gate base entry.
    if !hedge_mode {
        for idx in &base_entry_intents {
            if let Some(pos) = hedge_map.get(idx) {
                if *idx >= symbols.len() {
                    return Err(MirrorError::new(format!(
                        "missing symbol market data for collision close idx {}",
                        idx
                    )));
                }
                let sym = &symbols[*idx];
                let close_qty = hedge_close_qty_for_full_close(cfg.mode, pos, sym);
                if close_qty != 0.0 {
                    out.orders.push(MirrorOrder {
                        idx: *idx,
                        qty: close_qty,
                        price: hedge_close_price(cfg.mode, sym),
                        action: MirrorAction::Close,
                        reason: "collision_with_base".to_string(),
                    });
                    out.gate_base_entries.insert(*idx);
                }
            }
        }
    }

    // Active hedges excluding collisions.
    let mut active_hedges: Vec<MirrorPosition> = hedge_positions
        .iter()
        .filter(|p| p.size != 0.0 && !out.gate_base_entries.contains(&p.idx))
        .cloned()
        .collect();

    // Compute exposures from current positions.
    let mut gross_base: f64 = 0.0;
    let mut gross_base_positions: f64 = 0.0;
    for p in base_positions.iter().filter(|p| p.size != 0.0) {
        let sym = &symbols[p.idx];
        let exp = position_exposure(p, sym, balance);
        gross_base += exp;
        gross_base_positions += exp;
    }

    // Add exposure from imminent base orders (at market price) for proactive mirroring.
    // These orders are highly likely to fill next cycle, so we target their projected exposure.
    let mut gross_base_orders: f64 = 0.0;
    for o in desired_base_orders {
        if o.idx < symbols.len() {
            let sym = &symbols[o.idx];
            // Order exposure = |qty| * price * c_mult / balance
            let exp = o.qty.abs() * o.price * sym.exchange_params.c_mult / balance;
            gross_base += exp;
            gross_base_orders += exp;
        }
    }

    let mut gross_hedge: f64 = 0.0;
    for p in active_hedges.iter().filter(|p| p.size != 0.0) {
        let sym = &symbols[p.idx];
        gross_hedge += position_exposure(p, sym, balance);
    }

    let target_hedge = gross_base * cfg.threshold;
    // Tolerance band relative to target, not max TWEL.
    // This ensures mirror acts when meaningfully out of balance relative to target.
    let tolerance_band = target_hedge * cfg.tolerance_pct;
    let diff = gross_hedge - target_hedge;

    if debug {
        eprintln!(
            "[MIRROR DEBUG] gross_base={:.6} (positions={:.6}, orders={:.6}), gross_hedge={:.6}, target={:.6}, tol_band={:.6}, diff={:.6}",
            gross_base, gross_base_positions, gross_base_orders, gross_hedge, target_hedge, tolerance_band, diff
        );
    }

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
    let cap_exposure = (base_twel * cfg.threshold / max_positions as f64)
        * (1.0 + cfg.mirror_excess_allowance_pct);

    // Reduce hedge exposure: close least underwater first (full close).
    if diff > tolerance_band {
        // least underwater first
        active_hedges.sort_by(|a, b| {
            let sa = &symbols[a.idx];
            let sb = &symbols[b.idx];
            let ua = match cfg.mode {
                MirrorMode::MirrorShortsForLongs => underwaterness_short(a.price, sa),
                MirrorMode::MirrorLongsForShorts => underwaterness_long(a.price, sa),
            };
            let ub = match cfg.mode {
                MirrorMode::MirrorShortsForLongs => underwaterness_short(b.price, sb),
                MirrorMode::MirrorLongsForShorts => underwaterness_long(b.price, sb),
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
            out.orders.push(MirrorOrder {
                idx: pos.idx,
                qty: close_qty,
                price: hedge_close_price(cfg.mode, sym),
                action: MirrorAction::Close,
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
    // - restricted to approved_mirror_symbols for *opening*
    // - exclude base open symbols and base entry intents (one-way)
    // - apply EMA gating: only open new positions when price is extended beyond EMA
    let approved_set: HashSet<usize> = cfg.approved_mirror_symbols.iter().copied().collect();
    let mut eligible: Vec<MirrorSymbol> = Vec::new();
    for &idx in &cfg.approved_mirror_symbols {
        if idx >= symbols.len() {
            continue;
        }
        if !hedge_mode && (base_open.contains(&idx) || base_entry_intents.contains(&idx)) {
            continue;
        }
        // EMA gating: only include symbols where price is extended beyond EMA
        let sym = &symbols[idx];
        if !ema_gate_allows_entry(cfg.mode, sym, cfg.ema_dist_entry) {
            continue;
        }
        eligible.push(sym.clone());
    }
    let ranked = rank_assets_borda(&eligible);

    // Working hedge state (simulate in-cycle allocations).
    let mut hedge_state: HashMap<usize, MirrorPosition> = HashMap::new();
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
            MirrorMode::MirrorShortsForLongs => -min_qty,
            MirrorMode::MirrorLongsForShorts => min_qty,
        };
        out.orders.push(MirrorOrder {
            idx: sym.idx,
            qty: signed_qty,
            price: entry_price,
            action: MirrorAction::OpenOrIncrease,
            reason: "rebalance_open_min".to_string(),
        });
        hedge_state.insert(
            sym.idx,
            MirrorPosition {
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
                MirrorMode::MirrorShortsForLongs => underwaterness_short(pos.price, sym),
                MirrorMode::MirrorLongsForShorts => underwaterness_long(pos.price, sym),
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
        let Some((_u, idx)) = best else {
            break;
        };

        let sym = &symbols[idx];
        let entry_price = hedge_entry_price(cfg.mode, sym);
        let min_qty =
            effective_min_entry_qty(sym.effective_min_cost, entry_price, &sym.exchange_params);
        let min_cost = qty_to_cost(min_qty, entry_price, sym.exchange_params.c_mult);

        let pos = hedge_state.get(&idx).cloned().unwrap_or(MirrorPosition {
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
            MirrorMode::MirrorShortsForLongs => -qty,
            MirrorMode::MirrorLongsForShorts => qty,
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
            MirrorPosition {
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
        out.orders.push(MirrorOrder {
            idx,
            qty,
            price: entry_price,
            action: MirrorAction::OpenOrIncrease,
            reason: "rebalance_add".to_string(),
        });
    }

    Ok(out)
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(idx: usize, bid: f64, ask: f64, vol: f64, vola: f64, min_cost: f64) -> MirrorSymbol {
        let price = (bid + ask) * 0.5;
        MirrorSymbol {
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
            // Default EMA bands to market price (no gating effect)
            ema_upper: price,
            ema_lower: price,
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
        let cfg = MirrorConfig {
            threshold: 1.0,
            tolerance_pct: 0.10,
            approved_mirror_symbols: vec![1],
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1.0, 1.0, 1.0),
        ];
        let longs = vec![MirrorPosition {
            idx: 0,
            size: 1.0,
            price: 10.0,
        }];
        let shorts = vec![MirrorPosition {
            idx: 1,
            size: -1.0,
            price: 10.0,
        }];
        // hedge_mode=false means one-way enforcement (equivalent to old one_way=true)
        let res = compute_mirror_cycle(
            &cfg,
            &symbols,
            &longs,
            &shorts,
            &[],
            100.0,
            1.5,
            1,
            false,
            false,
        )
        .unwrap();
        assert!(res.orders.is_empty() && res.gate_base_entries.is_empty());
    }

    #[test]
    fn test_add_shorts_when_underhedged() {
        let cfg = MirrorConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_mirror_symbols: vec![1, 2],
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1000.0, 0.1, 1.0),
            sym(2, 10.0, 10.0, 900.0, 0.2, 1.0),
        ];
        let longs = vec![MirrorPosition {
            idx: 0,
            size: 10.0,
            price: 10.0,
        }]; // exposure = 1.0
        let shorts: Vec<MirrorPosition> = vec![]; // none
        let res = compute_mirror_cycle(
            &cfg,
            &symbols,
            &longs,
            &shorts,
            &[],
            100.0,
            1.5,
            1,
            false,
            false,
        )
        .unwrap();
        assert!(res.orders.iter().any(|o| o.qty < 0.0));
    }

    #[test]
    fn test_reduce_shorts_when_overhedged() {
        let cfg = MirrorConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_mirror_symbols: vec![1],
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1.0, 1.0, 1.0),
        ];
        let longs = vec![MirrorPosition {
            idx: 0,
            size: 5.0,
            price: 10.0,
        }]; // exposure 0.5
        let shorts = vec![MirrorPosition {
            idx: 1,
            size: -10.0,
            price: 10.0,
        }]; // exposure 1.0
        let res = compute_mirror_cycle(
            &cfg,
            &symbols,
            &longs,
            &shorts,
            &[],
            100.0,
            1.5,
            1,
            false,
            false,
        )
        .unwrap();
        assert!(res.orders.iter().any(|o| o.qty > 0.0));
    }

    #[test]
    fn test_collision_gates_base_entry_and_closes_hedge() {
        let cfg = MirrorConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_mirror_symbols: vec![0],
            ..Default::default()
        };
        let symbols = vec![sym(0, 10.0, 10.0, 1.0, 1.0, 1.0)];
        let longs = vec![MirrorPosition {
            idx: 0,
            size: 10.0,
            price: 10.0,
        }];
        let shorts = vec![MirrorPosition {
            idx: 0,
            size: -1.0,
            price: 10.0,
        }];
        let desired = vec![DesiredBaseOrder {
            idx: 0,
            qty: 1.0,
            price: 10.0,
        }];
        let res = compute_mirror_cycle(
            &cfg, &symbols, &longs, &shorts, &desired, 100.0, 1.5, 1, false, false,
        )
        .unwrap();
        assert!(res.gate_base_entries.contains(&0));
        assert!(res
            .orders
            .iter()
            .any(|o| o.qty > 0.0 && o.action == MirrorAction::Close));
    }

    #[test]
    fn test_short_only_mode_adds_longs() {
        let cfg = MirrorConfig {
            threshold: 1.0,
            tolerance_pct: 0.0,
            approved_mirror_symbols: vec![1],
            mode: MirrorMode::MirrorLongsForShorts,
            ..Default::default()
        };
        let symbols = vec![
            sym(0, 10.0, 10.0, 1.0, 1.0, 1.0),
            sym(1, 10.0, 10.0, 1.0, 1.0, 1.0),
        ];
        let shorts = vec![MirrorPosition {
            idx: 0,
            size: -10.0,
            price: 10.0,
        }]; // base exposure 1.0
        let hedges: Vec<MirrorPosition> = vec![];
        let res = compute_mirror_cycle(
            &cfg,
            &symbols,
            &hedges,
            &shorts,
            &[],
            100.0,
            1.5,
            1,
            false,
            false,
        )
        .unwrap();
        assert!(res
            .orders
            .iter()
            .any(|o| o.qty > 0.0 && o.action == MirrorAction::OpenOrIncrease));
    }
}
