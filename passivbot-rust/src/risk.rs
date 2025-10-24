use crate::constants::{LONG, SHORT};
use crate::types::{Order, OrderType};
use crate::utils::{
    calc_new_psize_pprice, calc_pprice_diff_int, calc_pside_price_diff_int, calc_wallet_exposure,
    round_dn, round_up,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct GateEntriesPosition {
    pub idx: usize,
    pub position_size: f64,
    pub position_price: f64,
    pub c_mult: f64,
}

#[derive(Clone, Debug)]
pub struct GateEntriesCandidate {
    pub idx: usize,
    pub qty: f64,
    pub price: f64,
    pub qty_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
    pub c_mult: f64,
    pub market_price: f64,
    pub order_type: OrderType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateEntriesDecision {
    pub idx: usize,
    pub qty: f64,
    pub price: f64,
    pub order_type: OrderType,
}

pub fn gate_entries_by_twel(
    pside: usize,
    balance: f64,
    total_wallet_exposure_limit: f64,
    positions: &[GateEntriesPosition],
    entries: &[GateEntriesCandidate],
) -> Vec<GateEntriesDecision> {
    const EXPOSURE_EPS: f64 = 1e-12;
    const QTY_EPS: f64 = 1e-12;

    if balance <= 0.0 || total_wallet_exposure_limit <= 0.0 {
        return Vec::new();
    }

    #[derive(Clone)]
    struct CandidateInternal {
        data: GateEntriesCandidate,
        distance: f64,
        original_order: usize,
    }

    let mut current_positions: HashMap<usize, (f64, f64, f64)> = HashMap::new();
    let mut current_twe = 0.0_f64;
    for pos in positions {
        if !pos.position_price.is_finite() || pos.position_price <= 0.0 {
            continue;
        }
        if !pos.position_size.is_finite() {
            continue;
        }
        let abs_psize = pos.position_size.abs();
        if abs_psize <= QTY_EPS {
            current_positions.insert(pos.idx, (0.0, pos.position_price, pos.c_mult));
            continue;
        }
        let exposure = calc_wallet_exposure(pos.c_mult, balance, abs_psize, pos.position_price);
        if !exposure.is_finite() {
            continue;
        }
        current_twe += exposure;
        current_positions.insert(pos.idx, (abs_psize, pos.position_price, pos.c_mult));
    }
    if current_twe >= total_wallet_exposure_limit - EXPOSURE_EPS {
        return Vec::new();
    }

    let mut candidates: Vec<CandidateInternal> = Vec::with_capacity(entries.len());
    for (original_order, entry) in entries.iter().enumerate() {
        if !entry.price.is_finite() || entry.price <= 0.0 {
            continue;
        }
        if !entry.market_price.is_finite() || entry.market_price <= 0.0 {
            continue;
        }
        if !entry.qty.is_finite() || entry.qty <= QTY_EPS {
            continue;
        }
        let qty_step = if entry.qty_step > 0.0 {
            entry.qty_step
        } else {
            continue;
        };
        let distance = calc_pside_price_diff_int(pside, entry.market_price, entry.price).abs();
        candidates.push(CandidateInternal {
            data: GateEntriesCandidate {
                qty: entry.qty.abs(),
                qty_step,
                ..entry.clone()
            },
            distance,
            original_order,
        });
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut included: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| (idx, candidate.data.qty))
        .collect();

    let compute_twe_if_filled = |selection: &[(usize, f64)]| -> f64 {
        let mut pos_state = current_positions.clone();
        for (cand_idx, qty) in selection {
            let qty = qty.max(0.0);
            if qty <= QTY_EPS {
                continue;
            }
            let candidate = &candidates[*cand_idx];
            let entry = pos_state.entry(candidate.data.idx).or_insert((
                0.0,
                candidate.data.price,
                candidate.data.c_mult,
            ));
            let (psize, pprice, c_mult) = *entry;
            let (new_psize, new_pprice) = calc_new_psize_pprice(
                psize,
                pprice,
                qty,
                candidate.data.price,
                candidate.data.qty_step,
            );
            *entry = (new_psize.abs(), new_pprice, c_mult);
        }
        let mut twe = 0.0_f64;
        for (_idx, (psize, pprice, c_mult)) in pos_state.iter() {
            if *psize <= QTY_EPS || *pprice <= 0.0 {
                continue;
            }
            let exposure = calc_wallet_exposure(*c_mult, balance, *psize, *pprice);
            if exposure.is_finite() {
                twe += exposure;
            }
        }
        twe
    };

    let mut twe_if_filled = compute_twe_if_filled(&included);
    if twe_if_filled < total_wallet_exposure_limit - EXPOSURE_EPS {
        let mut decisions: Vec<(usize, GateEntriesDecision)> = included
            .into_iter()
            .map(|(cand_idx, qty)| {
                let candidate = &candidates[cand_idx];
                (
                    candidate.original_order,
                    GateEntriesDecision {
                        idx: candidate.data.idx,
                        qty,
                        price: candidate.data.price,
                        order_type: candidate.data.order_type,
                    },
                )
            })
            .collect();
        decisions.sort_by_key(|(order_idx, _)| *order_idx);
        return decisions
            .into_iter()
            .map(|(_, decision)| decision)
            .collect();
    }

    let mut removal_order: Vec<usize> = (0..candidates.len()).collect();
    removal_order.sort_by(|a, b| {
        candidates[*b]
            .distance
            .partial_cmp(&candidates[*a].distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut removed_stack: Vec<(usize, f64)> = Vec::new();
    for cand_idx in removal_order {
        if twe_if_filled < total_wallet_exposure_limit - EXPOSURE_EPS {
            break;
        }
        if let Some(pos) = included.iter().position(|(idx, _)| *idx == cand_idx) {
            let entry = included.remove(pos);
            twe_if_filled = compute_twe_if_filled(&included);
            removed_stack.push(entry);
        }
    }

    if twe_if_filled >= total_wallet_exposure_limit - EXPOSURE_EPS {
        return Vec::new();
    }

    if let Some((cand_idx, original_qty)) = removed_stack.pop() {
        let candidate = &candidates[cand_idx];
        let mut lo = 0.0_f64;
        let mut hi = original_qty;
        let mut best_qty = 0.0_f64;

        for _ in 0..64 {
            let mid = (lo + hi) / 2.0;
            let mid_rd = round_dn(mid, candidate.data.qty_step);
            if mid_rd <= QTY_EPS {
                hi = mid;
                continue;
            }
            let mut trial = included.clone();
            trial.push((cand_idx, mid_rd));
            let twe_trial = compute_twe_if_filled(&trial);
            if twe_trial < total_wallet_exposure_limit - EXPOSURE_EPS {
                best_qty = mid_rd;
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let meets_min_qty =
            candidate.data.min_qty <= QTY_EPS || best_qty >= candidate.data.min_qty - QTY_EPS;
        let meets_min_cost = candidate.data.min_cost <= QTY_EPS
            || best_qty * candidate.data.price * candidate.data.c_mult
                >= candidate.data.min_cost - 1e-9;

        if best_qty > QTY_EPS && meets_min_qty && meets_min_cost {
            included.push((cand_idx, best_qty));
            twe_if_filled = compute_twe_if_filled(&included);
            if twe_if_filled >= total_wallet_exposure_limit - EXPOSURE_EPS {
                included.pop();
            }
        }
    }

    let mut decisions: Vec<(usize, GateEntriesDecision)> = included
        .into_iter()
        .map(|(cand_idx, qty)| {
            let candidate = &candidates[cand_idx];
            (
                candidate.original_order,
                GateEntriesDecision {
                    idx: candidate.data.idx,
                    qty,
                    price: candidate.data.price,
                    order_type: candidate.data.order_type,
                },
            )
        })
        .collect();
    decisions.sort_by_key(|(order, _)| *order);
    decisions
        .into_iter()
        .map(|(_, decision)| decision)
        .collect()
}

#[derive(Clone, Debug)]
pub struct TwelEnforcerInputPosition {
    pub idx: usize,
    pub position_size: f64,
    pub position_price: f64,
    pub market_price: f64,
    pub base_wallet_exposure_limit: f64,
    pub risk_wel_enforcer_threshold: f64,
    pub risk_we_excess_allowance_pct: f64,
    pub c_mult: f64,
    pub qty_step: f64,
    pub price_step: f64,
}

pub fn calc_twel_enforcer_actions(
    pside: usize,
    threshold: f64,
    total_wallet_exposure_limit: f64,
    balance: f64,
    positions: &[TwelEnforcerInputPosition],
    skip_idx: Option<usize>,
) -> Vec<(usize, Order)> {
    if threshold < 0.0 || total_wallet_exposure_limit <= 0.0 || balance <= 0.0 {
        return Vec::new();
    }
    let limit = total_wallet_exposure_limit * threshold;
    if limit <= 0.0 {
        return Vec::new();
    }
    #[derive(Clone, Debug)]
    struct Candidate {
        idx: usize,
        exposure: f64,
        target_psize: f64,
        initial_abs_psize: f64,
        abs_psize: f64,
        position_price: f64,
        market_price: f64,
        qty_step: f64,
        price_step: f64,
        c_mult: f64,
        price_diff: f64,
        psize_to_close: f64,
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(positions.len());
    let mut total_exposure = 0.0f64;
    for pos in positions {
        if !pos.position_price.is_finite() || pos.position_price <= 0.0 {
            continue;
        }
        let abs_psize = pos.position_size.abs();
        if abs_psize <= f64::EPSILON {
            continue;
        }
        let exposure = calc_wallet_exposure(pos.c_mult, balance, abs_psize, pos.position_price);
        if !exposure.is_finite() {
            continue;
        }
        total_exposure += exposure;

        let base_limit = pos.base_wallet_exposure_limit.max(0.0);
        if base_limit <= 0.0 {
            continue;
        }
        // Candidate eligibility: positions whose WE > WEL_base (not boosted, and decoupled from WEL threshold)
        if exposure <= base_limit {
            continue;
        }
        if let Some(skip) = skip_idx {
            if pos.idx == skip {
                continue;
            }
        }
        let market_price = if pos.market_price.is_finite() && pos.market_price > 0.0 {
            pos.market_price
        } else {
            pos.position_price
        };
        // Floor psize at WEL_base (do not reduce below base limit)
        let target_psize = (base_limit * balance) / (pos.position_price * pos.c_mult);
        let price_diff = calc_pprice_diff_int(pside, pos.position_price, market_price);
        candidates.push(Candidate {
            idx: pos.idx,
            exposure,
            target_psize,
            initial_abs_psize: abs_psize,
            abs_psize,
            position_price: pos.position_price,
            market_price,
            qty_step: pos.qty_step,
            price_step: pos.price_step,
            c_mult: pos.c_mult,
            price_diff,
            psize_to_close: 0.0,
        });
    }

    let exposure_tolerance = 1e-9;
    let qty_tolerance = 1e-9;
    if candidates.is_empty() || !(total_exposure > limit + exposure_tolerance) {
        return Vec::new();
    }

    loop {
        // Strict: continue until TWE < limit (with small tolerance)
        if total_exposure < limit - exposure_tolerance {
            break;
        }
        let mut best_idx: Option<usize> = None;
        let mut best_metric = f64::MAX;
        for (idx, candidate) in candidates.iter().enumerate() {
            if candidate.abs_psize <= candidate.target_psize + qty_tolerance {
                continue;
            }
            if candidate.price_diff < best_metric {
                best_metric = candidate.price_diff;
                best_idx = Some(idx);
            }
        }
        let Some(candidate_idx) = best_idx else {
            break;
        };
        let candidate = &mut candidates[candidate_idx];
        let max_reducible_psize = (candidate.abs_psize - candidate.target_psize).max(0.0);
        if max_reducible_psize <= qty_tolerance {
            continue;
        }
        let needed_exposure = (total_exposure - limit).max(0.0);
        if needed_exposure <= exposure_tolerance {
            break;
        }
        let needed_psize =
            (needed_exposure * balance) / (candidate.position_price * candidate.c_mult);
        let raw_reduce_psize = max_reducible_psize.min(needed_psize.max(0.0));
        if raw_reduce_psize <= qty_tolerance {
            continue;
        }
        let mut qty_reduce = round_up(raw_reduce_psize, candidate.qty_step);
        if qty_reduce > max_reducible_psize {
            qty_reduce = max_reducible_psize;
        }
        if qty_reduce <= qty_tolerance {
            continue;
        }
        let new_abs_psize = (candidate.abs_psize - qty_reduce).max(0.0);
        let new_exposure = calc_wallet_exposure(
            candidate.c_mult,
            balance,
            new_abs_psize,
            candidate.position_price,
        );
        let actual_reduce = (candidate.exposure - new_exposure).max(0.0);
        if actual_reduce <= exposure_tolerance {
            candidate.exposure = new_exposure;
            candidate.abs_psize = new_abs_psize;
            break;
        }
        candidate.abs_psize = new_abs_psize;
        candidate.exposure = new_exposure;
        candidate.psize_to_close += qty_reduce;
        total_exposure -= actual_reduce;
    }

    let mut actions: Vec<(usize, Order)> = Vec::new();
    for candidate in candidates {
        if candidate.psize_to_close <= qty_tolerance {
            continue;
        }
        let available_qty = (candidate.initial_abs_psize - candidate.abs_psize).max(0.0);
        if available_qty <= qty_tolerance {
            continue;
        }
        let mut qty_to_close = candidate.psize_to_close.min(available_qty);
        qty_to_close = round_dn(qty_to_close, candidate.qty_step);
        if qty_to_close <= qty_tolerance {
            continue;
        }
        let mut price = candidate.market_price;
        if !price.is_finite() || price <= 0.0 {
            price = candidate.position_price;
        }
        if price <= 0.0 {
            continue;
        }
        let price = match pside {
            LONG => {
                let adjusted = price * 0.9995;
                let rounded = round_dn(adjusted, candidate.price_step);
                if rounded <= 0.0 {
                    candidate.price_step
                } else {
                    rounded
                }
            }
            SHORT => {
                let adjusted = price * 1.0005;
                let rounded = round_up(adjusted, candidate.price_step);
                if rounded <= 0.0 {
                    candidate.price_step
                } else {
                    rounded
                }
            }
            _ => price,
        };
        let qty = match pside {
            LONG => -qty_to_close,
            SHORT => qty_to_close,
            _ => 0.0,
        };
        if qty.abs() <= qty_tolerance {
            continue;
        }
        let order_type = match pside {
            LONG => OrderType::CloseAutoReduceTwelLong,
            SHORT => OrderType::CloseAutoReduceTwelShort,
            _ => OrderType::Empty,
        };
        actions.push((
            candidate.idx,
            Order {
                qty,
                price,
                order_type,
            },
        ));
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(
        idx: usize,
        psize: f64,
        pprice: f64,
        mprice: f64,
        wel_base: f64,
        c_mult: f64,
        qty_step: f64,
        price_step: f64,
    ) -> TwelEnforcerInputPosition {
        TwelEnforcerInputPosition {
            idx,
            position_size: psize,
            position_price: pprice,
            market_price: mprice,
            base_wallet_exposure_limit: wel_base,
            risk_wel_enforcer_threshold: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            c_mult,
            qty_step,
            price_step,
        }
    }

    fn gate_pos(idx: usize, psize: f64, pprice: f64, c_mult: f64) -> GateEntriesPosition {
        GateEntriesPosition {
            idx,
            position_size: psize,
            position_price: pprice,
            c_mult,
        }
    }

    fn gate_entry(
        idx: usize,
        qty: f64,
        price: f64,
        market_price: f64,
        qty_step: f64,
        min_qty: f64,
        min_cost: f64,
        c_mult: f64,
        order_type: OrderType,
    ) -> GateEntriesCandidate {
        GateEntriesCandidate {
            idx,
            qty,
            price,
            qty_step,
            min_qty,
            min_cost,
            c_mult,
            market_price,
            order_type,
        }
    }

    #[test]
    fn test_twel_reducer_basic_two_positions() {
        // Two long positions, each exposure > WEL_base; total exceeds TWEL_target.
        let balance = 1000.0;
        let wel_base = 0.5; // per coin
        let twel = 0.9; // total limit
        let threshold = 1.0; // target == twel
                             // Position A: psize 8 at 50 => WE = 0.4
                             // Position B: psize 12 at 50 => WE = 0.6
                             // Total 1.0 > twel 0.9; need reduce 0.1 exposure strictly below
        let positions = vec![
            pos(0, 8.0, 50.0, 50.0, wel_base, 1.0, 0.1, 0.1),
            pos(1, 12.0, 50.0, 49.0, wel_base, 1.0, 0.1, 0.1),
        ];
        let actions = calc_twel_enforcer_actions(LONG, threshold, twel, balance, &positions, None);
        assert!(!actions.is_empty(), "should emit reduction actions");
        // Apply reductions and verify TWE < twel and no position below WEL_base
        let mut psize = vec![8.0, 12.0];
        let pprice = vec![50.0, 50.0];
        let c_mult = 1.0;
        for (idx, order) in actions {
            let i = idx;
            // long close qty negative
            let dq = order.qty.abs();
            psize[i] = (psize[i] - dq).max(0.0);
        }
        let we0 = calc_wallet_exposure(c_mult, balance, psize[0], pprice[0]);
        let we1 = calc_wallet_exposure(c_mult, balance, psize[1], pprice[1]);
        assert!(we0 >= wel_base - 1e-9, "reduced below WEL_base for pos0");
        assert!(we1 >= wel_base - 1e-9, "reduced below WEL_base for pos1");
        let twe = we0 + we1;
        assert!(
            twe < twel - 1e-12,
            "TWE not strictly below target: {} >= {}",
            twe,
            twel
        );
    }

    #[test]
    fn test_twel_reducer_selects_lowest_price_diff() {
        // One pos deep underwater (higher price_diff), one near breakeven; near breakeven should be reduced first (lowest diff)
        let balance = 1000.0;
        let wel_base = 0.4;
        let twel = 0.7;
        let positions = vec![
            // idx 0: position_price 100, market 60 (underwater, large diff)
            pos(0, 8.0, 100.0, 60.0, wel_base, 1.0, 0.1, 0.1),
            // idx 1: position_price 100, market 99 (near breakeven, small diff)
            pos(1, 7.5, 100.0, 99.0, wel_base, 1.0, 0.1, 0.1),
        ];
        let actions = calc_twel_enforcer_actions(LONG, 1.0, twel, balance, &positions, None);
        assert!(!actions.is_empty());
        // First action should target idx 1 (lowest diff)
        let first_idx = actions[0].0;
        assert_eq!(first_idx, 1);
    }

    #[test]
    fn test_gate_entries_blocks_when_twe_if_filled_exceeds() {
        let balance = 1000.0;
        let twel = 1.0;
        let positions = vec![gate_pos(0, 0.0, 0.0, 1.0)];
        let order_type = OrderType::EntryGridNormalLong;
        let entries = vec![
            gate_entry(0, 5.0, 100.0, 100.0, 0.01, 0.0, 0.0, 1.0, order_type),
            gate_entry(0, 6.0, 100.0, 90.0, 0.01, 0.0, 0.0, 1.0, order_type),
        ];
        let gated = gate_entries_by_twel(LONG, balance, twel, &positions, &entries);
        assert!(!gated.is_empty());
        let mut psize = 0.0;
        let mut pprice = 0.0;
        for decision in gated {
            let template = entries
                .iter()
                .find(|e| e.idx == decision.idx && (e.price - decision.price).abs() < 1e-12)
                .expect("matching entry template");
            let (nps, npp) = calc_new_psize_pprice(
                psize,
                pprice,
                decision.qty,
                decision.price,
                template.qty_step,
            );
            psize = nps;
            pprice = npp;
        }
        let twe = calc_wallet_exposure(
            1.0,
            balance,
            psize.abs(),
            if pprice > 0.0 { pprice } else { 100.0 },
        );
        assert!(
            twe < twel - 1e-12,
            "gated twe {} not strictly below twel {}",
            twe,
            twel
        );
    }

    #[test]
    fn test_gate_entries_blocks_when_current_twe_at_limit() {
        let balance = 1000.0;
        let twel = 0.5;
        // Existing position already at limit
        let positions = vec![gate_pos(0, 5.0, 100.0, 1.0)];
        let entries = vec![gate_entry(
            0,
            1.0,
            100.0,
            100.0,
            0.01,
            0.0,
            0.0,
            1.0,
            OrderType::EntryGridNormalLong,
        )];
        let gated = gate_entries_by_twel(LONG, balance, twel, &positions, &entries);
        assert!(
            gated.is_empty(),
            "expected no entries when current twe meets or exceeds limit"
        );
    }

    #[test]
    fn test_gate_entries_allows_when_below_limit() {
        let balance = 1000.0;
        let twel = 1.0;
        let positions = vec![gate_pos(0, 0.0, 0.0, 1.0)];
        let entries = vec![gate_entry(
            0,
            4.0,
            100.0,
            100.0,
            0.01,
            0.0,
            0.0,
            1.0,
            OrderType::EntryGridNormalLong,
        )];
        let gated = gate_entries_by_twel(LONG, balance, twel, &positions, &entries);
        assert_eq!(gated.len(), 1);
        assert!((gated[0].qty - 4.0).abs() < 1e-12);
        assert!((gated[0].price - 100.0).abs() < 1e-12);
    }
}
