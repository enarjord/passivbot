//! Hedge overlay (experimental, isolated).
//!
//! This module is intentionally self contained and not wired into the rest of the code base.
//! It encodes the long-only + mirrored shorts rules (1-7) and the symmetric short-only + mirrored longs.
//! Defaults are hardcoded; plumbing from config will be added later.
//! Quantities follow core conventions: base longs/hedge longs positive, base shorts/hedge shorts negative.

use std::collections::{HashMap, HashSet};

use crate::types::ExchangeParams;
use crate::utils::{cost_to_qty, qty_to_cost, round_, round_up};

/// Default parameters (hardcoded for the prototype).
const DEFAULT_TOLERANCE: f64 = 0.05;
const DEFAULT_HEDGE_EXCESS_ALLOWANCE_PCT: f64 = 0.20;

#[derive(Clone, Debug, PartialEq)]
pub struct HedgePosition {
    pub idx: usize,
    /// Signed size: longs positive, shorts negative.
    pub size: f64,
    pub price: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HedgeAsset {
    pub idx: usize,
    pub bid: f64,
    pub ask: f64,
    pub volume_score: f64,
    pub volatility_score: f64,
    pub exchange_params: ExchangeParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesiredBaseOrder {
    pub idx: usize,
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
pub struct HedgeResult {
    pub orders: Vec<HedgeOrder>,
    pub deferred_base_longs: HashSet<usize>,
    pub deferred_base_shorts: HashSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HedgeMode {
    /// Base is long-only, hedges are shorts.
    HedgeShortsForLongs,
    /// Base is short-only, hedges are longs.
    HedgeLongsForShorts,
}

/// Gross exposure = sum(|size| * price) / balance.
fn gross_exposure(positions: &[HedgePosition], balance: f64) -> f64 {
    positions
        .iter()
        .map(|p| qty_to_cost(p.size.abs(), p.price, 1.0) / balance)
        .sum()
}

/// Borda ranking: low volatility + high volume. Lower score is better.
fn rank_assets_borda(eligible: &[HedgeAsset]) -> Vec<HedgeAsset> {
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
    let mut scored: Vec<(usize, HedgeAsset)> = eligible
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

/// Reduce slot count until per-slot cost meets min_cost.
fn compute_slot_count(
    target_exposure: f64,
    balance: f64,
    max_slots: usize,
    min_cost: f64,
) -> usize {
    if max_slots == 0 {
        return 0;
    }
    let mut slots = max_slots;
    loop {
        let per_slot_notional = (target_exposure * balance) / (slots as f64);
        if per_slot_notional + 1e-12 >= min_cost {
            return slots;
        }
        if slots == 1 {
            return 1;
        }
        slots -= 1;
    }
}

/// Compute desired hedge orders.
pub fn compute_hedge_orders(
    base_mode: HedgeMode,
    base_positions: &[HedgePosition],
    hedge_positions: &[HedgePosition],
    balance: f64,
    twel: f64,
    eligible_assets: &[HedgeAsset],
    desired_base_orders: &[DesiredBaseOrder],
    one_way: bool,
) -> HedgeResult {
    let mut result = HedgeResult::default();
    if balance <= 0.0 {
        return result;
    }

    let desired_base_set: HashSet<usize> = desired_base_orders.iter().map(|o| o.idx).collect();

    // Map current hedge positions by idx.
    let hedge_map: HashMap<usize, HedgePosition> = hedge_positions
        .iter()
        .cloned()
        .map(|p| (p.idx, p))
        .collect();

    // Collision handling for one-way: if base wants an idx already hedged on opposite side, close hedge and defer base order.
    if one_way {
        for idx in &desired_base_set {
            if let Some(hp) = hedge_map.get(idx) {
                if let Some(asset) = eligible_assets.iter().find(|s| s.idx == *idx) {
                    let qty = round_(hp.size.abs(), asset.exchange_params.qty_step);
                    if qty > 0.0 {
                        // Close the hedge (direction depends on hedge mode).
                        let close_qty = match base_mode {
                            HedgeMode::HedgeShortsForLongs => qty, // buy to close short => +qty
                            HedgeMode::HedgeLongsForShorts => -qty, // sell to close long => -qty
                        };
                        result.orders.push(HedgeOrder {
                            idx: *idx,
                            qty: close_qty,
                            price: asset.bid,
                            action: HedgeAction::Close,
                            reason: "collision_with_base".to_string(),
                        });
                    }
                    match base_mode {
                        HedgeMode::HedgeShortsForLongs => {
                            result.deferred_base_longs.insert(*idx);
                        }
                        HedgeMode::HedgeLongsForShorts => {
                            result.deferred_base_shorts.insert(*idx);
                        }
                    }
                }
            }
        }
    }

    // Active hedge positions excluding collisions we just closed.
    let active_hedges: Vec<HedgePosition> = hedge_positions
        .iter()
        .filter(|p| !desired_base_set.contains(&p.idx))
        .cloned()
        .collect();

    let gross_base = gross_exposure(base_positions, balance);
    let gross_hedge = gross_exposure(&active_hedges, balance);
    let net = gross_base - gross_hedge;

    // Tolerance band.
    if net.abs() <= DEFAULT_TOLERANCE * twel {
        return result;
    }

    // Target hedge exposure mirrors gross_base.
    let target_hedge_exposure = gross_base;
    let remaining_hedge_exposure = (target_hedge_exposure - gross_hedge).max(0.0);
    let mut reduce_hedge_exposure = (gross_hedge - target_hedge_exposure).max(0.0);

    // Filter eligible hedge assets (exclude desired base assets if one-way).
    let eligible_filtered: Vec<HedgeAsset> = eligible_assets
        .iter()
        .filter(|s| !one_way || !desired_base_set.contains(&s.idx))
        .cloned()
        .collect();
    let ranked = rank_assets_borda(&eligible_filtered);
    if ranked.is_empty() {
        return result;
    }

    // Slot count and per-slot cap.
    let max_slots = ranked.len().min(base_positions.len().max(1));
    let slots = compute_slot_count(
        target_hedge_exposure,
        balance,
        max_slots,
        ranked[0].exchange_params.min_cost,
    );
    let per_slot_target = if slots > 0 {
        (target_hedge_exposure * balance) / slots as f64
    } else {
        0.0
    };
    let per_slot_cap = per_slot_target * (1.0 + DEFAULT_HEDGE_EXCESS_ALLOWANCE_PCT);

    // Working map of hedge state.
    let hedge_state: HashMap<usize, HedgePosition> = ranked
        .iter()
        .map(|s| {
            let p = active_hedges
                .iter()
                .find(|p| p.idx == s.idx)
                .cloned()
                .unwrap_or(HedgePosition {
                    idx: s.idx,
                    size: 0.0,
                    price: s.ask,
                });
            (s.idx, p)
        })
        .collect();

    // Reduce hedge if overexposed: close least underwater first.
    if reduce_hedge_exposure > DEFAULT_TOLERANCE * twel {
        let mut hedges_vec: Vec<(f64, HedgeAsset, HedgePosition)> = ranked
            .iter()
            .filter_map(|asset| {
                let pos = hedge_state.get(&asset.idx)?;
                if pos.size == 0.0 {
                    return None;
                }
                let underwater = match base_mode {
                    HedgeMode::HedgeShortsForLongs => {
                        if pos.price > 0.0 {
                            asset.ask / pos.price - 1.0
                        } else {
                            0.0
                        }
                    }
                    HedgeMode::HedgeLongsForShorts => {
                        if pos.price > 0.0 {
                            asset.bid / pos.price - 1.0
                        } else {
                            0.0
                        }
                    }
                };
                Some((underwater, asset.clone(), pos.clone()))
            })
            .collect();
        // least underwater first
        hedges_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (_, asset, pos) in hedges_vec {
            if reduce_hedge_exposure <= 0.0 {
                break;
            }
            let notional = qty_to_cost(pos.size.abs(), pos.price, asset.exchange_params.c_mult);
            let reduce_notional = notional.min(reduce_hedge_exposure * balance);
            let mut qty = cost_to_qty(reduce_notional, asset.bid, asset.exchange_params.c_mult);
            qty = round_(qty, asset.exchange_params.qty_step);
            if qty <= 0.0 {
                continue;
            }
            let signed_qty = match base_mode {
                HedgeMode::HedgeShortsForLongs => qty,  // buy to close short
                HedgeMode::HedgeLongsForShorts => -qty, // sell to close long
            };
            result.orders.push(HedgeOrder {
                idx: asset.idx,
                qty: signed_qty,
                price: asset.bid,
                action: HedgeAction::Close,
                reason: "rebalance_reduce".to_string(),
            });
            reduce_hedge_exposure -=
                qty_to_cost(qty.abs(), asset.bid, asset.exchange_params.c_mult) / balance;
        }
        return result;
    }

    // Add/open hedges if underexposed.
    if remaining_hedge_exposure > DEFAULT_TOLERANCE * twel {
        let mut hedges_vec: Vec<(f64, HedgeAsset, HedgePosition)> = ranked
            .iter()
            .filter_map(|asset| {
                let pos = hedge_state.get(&asset.idx)?.clone();
                let underwater = if pos.size != 0.0 && pos.price > 0.0 {
                    match base_mode {
                        HedgeMode::HedgeShortsForLongs => asset.ask / pos.price - 1.0,
                        HedgeMode::HedgeLongsForShorts => asset.bid / pos.price - 1.0,
                    }
                } else {
                    1.0 // seed new positions first
                };
                Some((underwater, asset.clone(), pos))
            })
            .collect();
        // most underwater first
        hedges_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut allowance_notional = remaining_hedge_exposure * balance;

        for (_, asset, pos) in hedges_vec {
            if allowance_notional <= 0.0 {
                break;
            }
            let current_notional =
                qty_to_cost(pos.size.abs(), pos.price, asset.exchange_params.c_mult);
            let target_notional = per_slot_target.min(per_slot_cap);
            if current_notional >= target_notional - 1e-12 {
                continue;
            }
            let to_add_notional = (target_notional - current_notional).min(allowance_notional);
            if to_add_notional <= 0.0 {
                continue;
            }
            let mut qty = cost_to_qty(to_add_notional, asset.ask, asset.exchange_params.c_mult);
            qty = round_up(qty, asset.exchange_params.qty_step);
            if qty <= 0.0 {
                continue;
            }
            // Respect min_cost/min_qty
            let cost_after = qty_to_cost(qty, asset.ask, asset.exchange_params.c_mult);
            if cost_after < asset.exchange_params.min_cost - 1e-9
                || qty < asset.exchange_params.min_qty - 1e-9
            {
                continue;
            }
            let signed_qty = match base_mode {
                HedgeMode::HedgeShortsForLongs => -qty, // sell to open/increase short
                HedgeMode::HedgeLongsForShorts => qty,  // buy to open/increase long
            };
            result.orders.push(HedgeOrder {
                idx: asset.idx,
                qty: signed_qty,
                price: asset.ask,
                action: HedgeAction::OpenOrIncrease,
                reason: "rebalance_add".to_string(),
            });
            allowance_notional -= cost_after;
        }
    }

    result
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(idx: usize, bid: f64, ask: f64, vol: f64, vola: f64, min_cost: f64) -> HedgeAsset {
        HedgeAsset {
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
        }
    }

    #[test]
    fn test_rank_borda_prefers_low_vol_high_volume() {
        let a = asset(0, 10.0, 10.0, 1000.0, 0.1, 1.0);
        let b = asset(1, 10.0, 10.0, 900.0, 0.2, 1.0);
        let c = asset(2, 10.0, 10.0, 800.0, 0.05, 1.0);
        let ranked = rank_assets_borda(&[a.clone(), b.clone(), c.clone()]);
        assert_eq!(ranked[0].idx, 0); // best volume, decent vol
        assert_eq!(ranked[1].idx, 2); // best vol
    }

    #[test]
    fn test_no_action_within_tolerance() {
        let longs = vec![HedgePosition {
            idx: 0,
            size: 1.0,
            price: 10.0,
        }];
        let shorts: Vec<HedgePosition> = vec![HedgePosition {
            idx: 1,
            size: -1.0,
            price: 10.0,
        }];
        let eligible = vec![asset(1, 10.0, 10.0, 1000.0, 0.1, 1.0)];
        let res = compute_hedge_orders(
            HedgeMode::HedgeShortsForLongs,
            &longs,
            &shorts,
            100.0,
            1.5,
            &eligible,
            &[],
            false,
        );
        assert!(res.orders.is_empty());
    }

    #[test]
    fn test_add_shorts_when_underhedged() {
        let longs = vec![HedgePosition {
            idx: 0,
            size: 10.0,
            price: 10.0,
        }]; // exposure = 1.0
        let shorts: Vec<HedgePosition> = vec![]; // none
        let eligible = vec![
            asset(1, 10.0, 10.0, 1000.0, 0.1, 1.0),
            asset(2, 10.0, 10.0, 900.0, 0.2, 1.0),
        ];
        let res = compute_hedge_orders(
            HedgeMode::HedgeShortsForLongs,
            &longs,
            &shorts,
            100.0,
            1.5,
            &eligible,
            &[],
            false,
        );
        assert!(res.orders.iter().any(|o| o.qty < 0.0));
    }

    #[test]
    fn test_reduce_shorts_when_overhedged() {
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
        let eligible = vec![asset(1, 10.0, 10.0, 1000.0, 0.1, 1.0)];
        let res = compute_hedge_orders(
            HedgeMode::HedgeShortsForLongs,
            &longs,
            &shorts,
            100.0,
            1.5,
            &eligible,
            &[],
            false,
        );
        assert!(res.orders.iter().any(|o| o.qty > 0.0));
    }

    #[test]
    fn test_collision_defers_long_and_closes_short() {
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
        let eligible = vec![asset(0, 10.0, 10.0, 1000.0, 0.1, 1.0)];
        let desired_long = vec![DesiredBaseOrder { idx: 0 }];
        let res = compute_hedge_orders(
            HedgeMode::HedgeShortsForLongs,
            &longs,
            &shorts,
            100.0,
            1.5,
            &eligible,
            &desired_long,
            true,
        );
        assert!(res.deferred_base_longs.contains(&0));
        assert!(res
            .orders
            .iter()
            .any(|o| o.qty > 0.0 && o.action == HedgeAction::Close));
    }

    #[test]
    fn test_short_only_mode_adds_longs() {
        let shorts = vec![HedgePosition {
            idx: 0,
            size: -10.0,
            price: 10.0,
        }]; // base exposure 1.0
        let hedges: Vec<HedgePosition> = vec![];
        let eligible = vec![asset(1, 10.0, 10.0, 1000.0, 0.1, 1.0)];
        let res = compute_hedge_orders(
            HedgeMode::HedgeLongsForShorts,
            &shorts,
            &hedges,
            100.0,
            1.5,
            &eligible,
            &[],
            false,
        );
        assert!(res
            .orders
            .iter()
            .any(|o| o.qty > 0.0 && o.action == HedgeAction::OpenOrIncrease));
    }
}
