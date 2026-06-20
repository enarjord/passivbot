use crate::constants::{LONG, SHORT};
use crate::entries::calc_min_entry_qty;
use crate::types::{ExchangeParams, Order, OrderType, TwelEnforcerPolicy};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_pprice_diff_int, calc_pside_price_diff_int, calc_wallet_exposure, cost_to_qty,
    quantize_price, quantize_qty, round_dn, round_up, RoundingMode,
};
use std::collections::HashMap;

fn exposure_to_psize(exposure: f64, balance: f64, price: f64, c_mult: f64) -> f64 {
    if exposure <= 0.0 || balance <= 0.0 || price <= 0.0 || c_mult <= 0.0 {
        0.0
    } else {
        exposure * balance / (price * c_mult)
    }
}

#[derive(Clone, Copy, Debug)]
struct LeastStuckCandidate {
    collection_idx: usize,
    symbol_idx: usize,
    price_diff: f64,
}

fn compare_least_stuck(a: LeastStuckCandidate, b: LeastStuckCandidate) -> std::cmp::Ordering {
    match a
        .price_diff
        .partial_cmp(&b.price_diff)
        .unwrap_or(std::cmp::Ordering::Equal)
    {
        std::cmp::Ordering::Equal => a.symbol_idx.cmp(&b.symbol_idx),
        other => other,
    }
}

fn least_stuck_order(candidates: Vec<LeastStuckCandidate>) -> Vec<usize> {
    let mut candidates: Vec<LeastStuckCandidate> = candidates
        .into_iter()
        .filter(|candidate| candidate.price_diff.is_finite())
        .collect();
    candidates.sort_by(|a, b| compare_least_stuck(*a, *b));
    candidates
        .into_iter()
        .map(|candidate| candidate.collection_idx)
        .collect()
}

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
    pub original_order: usize,
}

pub fn gate_entries_by_twel(
    pside: usize,
    balance: f64,
    total_wallet_exposure_limit: f64,
    positions: &[GateEntriesPosition],
    entries: &[GateEntriesCandidate],
) -> Vec<GateEntriesDecision> {
    // Public/PyO3 reference helper. The live/backtest production gate lives in
    // orchestrator.rs and is covered by parity regressions there.
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
        let distance = calc_pside_price_diff_int(pside, entry.market_price, entry.price);
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
                        original_order: candidate.original_order,
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
                    original_order: candidate.original_order,
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
pub struct UnstuckPositionInput {
    pub idx: usize,
    pub side: usize,
    pub position_size: f64,
    pub position_price: f64,
    pub wallet_exposure_limit: f64,
    pub effective_we_excess_allowance_pct: f64,
    pub unstuck_threshold: f64,
    pub unstuck_close_pct: f64,
    pub unstuck_ema_dist: f64,
    pub unstuck_loss_allowance_pct: f64,
    pub total_wallet_exposure_limit: f64,
    pub ema_band_upper: f64,
    pub ema_band_lower: f64,
    pub current_price: f64,
    pub price_step: f64,
    pub qty_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
    pub c_mult: f64,
}

pub fn calc_unstucking_action(
    balance: f64,
    allowance_long: f64,
    allowance_short: f64,
    positions: &[UnstuckPositionInput],
) -> Option<(usize, usize, Order)> {
    calc_unstucking_action_with_allowance(balance, positions, |input| match input.side {
        LONG => allowance_long,
        SHORT => allowance_short,
        _ => 0.0,
    })
}

pub fn calc_unstucking_action_with_position_allowances(
    balance: f64,
    pnl_cumsum_max: f64,
    pnl_cumsum_last: f64,
    positions: &[UnstuckPositionInput],
) -> Option<(usize, usize, Order)> {
    calc_unstucking_action_with_allowance(balance, positions, |input| {
        if input.unstuck_loss_allowance_pct <= 0.0
            || input.total_wallet_exposure_limit <= 0.0
            || !input.unstuck_loss_allowance_pct.is_finite()
            || !input.total_wallet_exposure_limit.is_finite()
        {
            return 0.0;
        }
        calc_auto_unstuck_allowance(
            balance,
            input.unstuck_loss_allowance_pct * input.total_wallet_exposure_limit,
            pnl_cumsum_max,
            pnl_cumsum_last,
        )
    })
}

fn calc_unstucking_action_with_allowance<F>(
    balance: f64,
    positions: &[UnstuckPositionInput],
    mut allowance_for: F,
) -> Option<(usize, usize, Order)>
where
    F: FnMut(&UnstuckPositionInput) -> f64,
{
    if balance <= 0.0 || positions.is_empty() {
        return None;
    }

    #[derive(Clone, Copy, Debug)]
    struct Candidate<'a> {
        input: &'a UnstuckPositionInput,
        pprice_diff: f64,
    }

    let mut stuck_positions: Vec<Candidate> = Vec::new();

    for input in positions {
        let allowance = allowance_for(input);
        if allowance <= 0.0 {
            continue;
        }

        if !input.position_price.is_finite() || input.position_price <= 0.0 {
            continue;
        }
        let size_abs = input.position_size.abs();
        if size_abs <= f64::EPSILON {
            continue;
        }

        let wallet_exposure =
            calc_wallet_exposure(input.c_mult, balance, size_abs, input.position_price);
        let allowance_multiplier = 1.0 + input.effective_we_excess_allowance_pct.max(0.0);
        let effective_wel = input.wallet_exposure_limit * allowance_multiplier;

        let unstuck_threshold = input.unstuck_threshold;
        if unstuck_threshold < 0.0 {
            continue;
        }
        if effective_wel > 0.0 && wallet_exposure / effective_wel <= unstuck_threshold {
            continue;
        }

        if !input.current_price.is_finite() || input.current_price <= 0.0 {
            continue;
        }

        let price_step = if input.price_step > 0.0 {
            input.price_step
        } else {
            0.0
        };

        let ema_price = match input.side {
            LONG => {
                if !input.ema_band_upper.is_finite() || input.ema_band_upper <= 0.0 {
                    continue;
                }
                let target = input.ema_band_upper * (1.0 + input.unstuck_ema_dist);
                let rounded = if price_step > 0.0 {
                    round_up(target, price_step)
                } else {
                    target
                };
                if !rounded.is_finite() || rounded <= 0.0 {
                    continue;
                }
                rounded
            }
            SHORT => {
                if !input.ema_band_lower.is_finite() || input.ema_band_lower <= 0.0 {
                    continue;
                }
                let target = input.ema_band_lower * (1.0 - input.unstuck_ema_dist);
                let rounded = if price_step > 0.0 {
                    round_dn(target, price_step)
                } else {
                    target
                };
                if !rounded.is_finite() || rounded <= 0.0 {
                    continue;
                }
                rounded
            }
            _ => continue,
        };

        let meets_trigger = match input.side {
            LONG => input.current_price >= ema_price,
            SHORT => input.current_price <= ema_price,
            _ => false,
        };
        if !meets_trigger {
            continue;
        }

        let pprice_diff =
            calc_pprice_diff_int(input.side, input.position_price, input.current_price);
        if !pprice_diff.is_finite() {
            continue;
        }

        stuck_positions.push(Candidate { input, pprice_diff });
    }

    if stuck_positions.is_empty() {
        return None;
    }

    let candidate_order = least_stuck_order(
        stuck_positions
            .iter()
            .enumerate()
            .map(|(collection_idx, candidate)| LeastStuckCandidate {
                collection_idx,
                symbol_idx: candidate.input.idx,
                price_diff: candidate.pprice_diff,
            })
            .collect(),
    );

    for candidate_idx in candidate_order {
        let input = stuck_positions[candidate_idx].input;
        let allowance = allowance_for(input);
        if allowance <= 0.0 {
            continue;
        }

        let exchange_params = ExchangeParams {
            qty_step: input.qty_step,
            price_step: input.price_step,
            min_qty: input.min_qty,
            min_cost: input.min_cost,
            c_mult: input.c_mult,
            ..Default::default()
        };
        let min_entry_qty = calc_min_entry_qty(input.current_price, &exchange_params);
        let allowance_multiplier = 1.0 + input.effective_we_excess_allowance_pct.max(0.0);
        let effective_wel = input.wallet_exposure_limit * allowance_multiplier;

        match input.side {
            LONG => {
                let size_abs = input.position_size.max(0.0);
                if size_abs <= f64::EPSILON {
                    continue;
                }
                let target_qty = cost_to_qty(
                    balance * effective_wel * input.unstuck_close_pct,
                    input.current_price,
                    input.c_mult,
                );
                let target_qty = round_dn(target_qty, input.qty_step).max(0.0);
                let mut close_qty = -f64::min(size_abs, f64::max(min_entry_qty, target_qty));
                if close_qty == 0.0 {
                    continue;
                }
                let pnl_if_closed = calc_pnl_long(
                    input.position_price,
                    input.current_price,
                    close_qty,
                    input.c_mult,
                );
                let pnl_abs = pnl_if_closed.abs();
                if pnl_if_closed < 0.0 && pnl_abs > allowance {
                    let scaled_qty = close_qty.abs() * (allowance / pnl_abs);
                    let scaled_qty = f64::min(size_abs, scaled_qty);
                    let scaled_qty = f64::max(min_entry_qty, round_dn(scaled_qty, input.qty_step));
                    close_qty = -scaled_qty;
                }
                if close_qty == 0.0 {
                    continue;
                }
                return Some((
                    input.idx,
                    LONG,
                    Order {
                        qty: close_qty,
                        price: input.current_price,
                        order_type: OrderType::CloseUnstuckLong,
                    },
                ));
            }
            SHORT => {
                let size_abs = input.position_size.abs();
                if size_abs <= f64::EPSILON {
                    continue;
                }
                let target_qty = cost_to_qty(
                    balance * effective_wel * input.unstuck_close_pct,
                    input.current_price,
                    input.c_mult,
                );
                let target_qty = round_dn(target_qty, input.qty_step).max(0.0);
                let mut close_qty = f64::min(size_abs, f64::max(min_entry_qty, target_qty));
                if close_qty == 0.0 {
                    continue;
                }
                let pnl_if_closed = calc_pnl_short(
                    input.position_price,
                    input.current_price,
                    close_qty,
                    input.c_mult,
                );
                let pnl_abs = pnl_if_closed.abs();
                if pnl_if_closed < 0.0 && pnl_abs > allowance {
                    let scaled_qty = close_qty * (allowance / pnl_abs);
                    let scaled_qty = f64::min(size_abs, scaled_qty);
                    let scaled_qty = f64::max(min_entry_qty, round_dn(scaled_qty, input.qty_step));
                    close_qty = scaled_qty;
                }
                if close_qty == 0.0 {
                    continue;
                }
                return Some((
                    input.idx,
                    SHORT,
                    Order {
                        qty: close_qty,
                        price: input.current_price,
                        order_type: OrderType::CloseUnstuckShort,
                    },
                ));
            }
            _ => continue,
        }
    }

    None
}

#[derive(Clone, Debug)]
pub struct TwelEnforcerInputPosition {
    pub idx: usize,
    pub position_size: f64,
    pub position_price: f64,
    pub market_price: f64,
    pub base_wallet_exposure_limit: f64,
    pub c_mult: f64,
    pub qty_step: f64,
    pub price_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
}

pub fn calc_twel_enforcer_actions(
    pside: usize,
    threshold: f64,
    total_wallet_exposure_limit: f64,
    effective_n_positions: usize,
    balance: f64,
    positions: &[TwelEnforcerInputPosition],
    policy: TwelEnforcerPolicy,
    skip_idx: Option<usize>,
) -> Vec<(usize, Order)> {
    if threshold <= 0.0
        || total_wallet_exposure_limit <= 0.0
        || balance <= 0.0
        || effective_n_positions == 0
        || !threshold.is_finite()
        || !total_wallet_exposure_limit.is_finite()
        || !balance.is_finite()
    {
        return Vec::new();
    }
    let limit = total_wallet_exposure_limit * threshold;
    if !limit.is_finite() || limit <= 0.0 {
        return Vec::new();
    }
    #[derive(Clone, Debug)]
    struct Candidate {
        idx: usize,
        exposure: f64,
        initial_abs_psize: f64,
        position_price: f64,
        market_price: f64,
        qty_step: f64,
        price_step: f64,
        min_qty: f64,
        min_cost: f64,
        c_mult: f64,
        adverse_loss: f64,
    }

    let mut valid_positions: Vec<Candidate> = Vec::with_capacity(positions.len());
    let mut total_exposure = 0.0f64;
    for pos in positions {
        if !pos.position_price.is_finite() || pos.position_price <= 0.0 {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid position_price {}",
                pos.idx,
                pos.position_price
            );
            continue;
        }
        let abs_psize = pos.position_size.abs();
        if abs_psize <= f64::EPSILON {
            log::error!(
                "TWEL enforcer input rejected: idx={} zero position_size",
                pos.idx
            );
            continue;
        }
        if pos.c_mult <= 0.0 || !pos.c_mult.is_finite() {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid c_mult {}",
                pos.idx,
                pos.c_mult
            );
            continue;
        }
        if pos.qty_step <= 0.0 || !pos.qty_step.is_finite() {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid qty_step {}",
                pos.idx,
                pos.qty_step
            );
            continue;
        }
        if pos.price_step <= 0.0 || !pos.price_step.is_finite() {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid price_step {}",
                pos.idx,
                pos.price_step
            );
            continue;
        }
        if pos.min_qty < 0.0 || !pos.min_qty.is_finite() {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid min_qty {}",
                pos.idx,
                pos.min_qty
            );
            continue;
        }
        if pos.min_cost < 0.0 || !pos.min_cost.is_finite() {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid min_cost {}",
                pos.idx,
                pos.min_cost
            );
            continue;
        }
        let exposure = calc_wallet_exposure(pos.c_mult, balance, abs_psize, pos.position_price);
        if !exposure.is_finite() {
            log::error!(
                "TWEL enforcer input rejected: idx={} exposure non-finite",
                pos.idx
            );
            continue;
        }
        total_exposure += exposure;

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
        let base_limit = pos.base_wallet_exposure_limit.max(0.0);
        if base_limit <= 0.0 {
            log::error!(
                "TWEL enforcer input rejected: idx={} invalid base WEL {}",
                pos.idx,
                pos.base_wallet_exposure_limit
            );
            continue;
        }
        let projected_pnl = match pside {
            LONG => calc_pnl_long(pos.position_price, market_price, -abs_psize, pos.c_mult),
            SHORT => calc_pnl_short(pos.position_price, market_price, abs_psize, pos.c_mult),
            _ => 0.0,
        };
        let adverse_loss = if projected_pnl.is_finite() {
            (-projected_pnl).max(0.0)
        } else {
            f64::INFINITY
        };
        valid_positions.push(Candidate {
            idx: pos.idx,
            exposure,
            initial_abs_psize: abs_psize,
            position_price: pos.position_price,
            market_price,
            qty_step: pos.qty_step,
            price_step: pos.price_step,
            min_qty: pos.min_qty.max(0.0),
            min_cost: pos.min_cost.max(0.0),
            c_mult: pos.c_mult,
            adverse_loss,
        });
    }

    let exposure_tolerance = 1e-9;
    let qty_tolerance = 1e-9;
    if valid_positions.is_empty() || !(total_exposure > limit + exposure_tolerance) {
        return Vec::new();
    }

    let overweight_target = limit / effective_n_positions as f64;
    let mut candidates: Vec<Candidate> = valid_positions
        .into_iter()
        .filter(|candidate| match policy {
            TwelEnforcerPolicy::ReduceOverweight => {
                candidate.exposure > overweight_target + exposure_tolerance
            }
            TwelEnforcerPolicy::ReducePortfolio => candidate.exposure > exposure_tolerance,
        })
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }

    candidates.sort_by(|a, b| {
        a.adverse_loss
            .partial_cmp(&b.adverse_loss)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.idx.cmp(&b.idx))
    });

    let mut actions: Vec<(usize, Order)> = Vec::new();
    for candidate in candidates {
        let needed_exposure = (total_exposure - limit).max(0.0);
        let exposure_to_cut = needed_exposure.min(candidate.exposure);
        let mut qty_to_close = exposure_to_psize(
            exposure_to_cut,
            balance,
            candidate.position_price,
            candidate.c_mult,
        );
        qty_to_close = round_up(qty_to_close, candidate.qty_step);
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
        let exchange_params = ExchangeParams {
            qty_step: candidate.qty_step,
            price_step: candidate.price_step,
            min_qty: candidate.min_qty,
            min_cost: candidate.min_cost,
            c_mult: candidate.c_mult,
            ..Default::default()
        };
        let min_entry_qty = calc_min_entry_qty(price, &exchange_params);
        let mut abs_qty = qty_to_close
            .max(min_entry_qty)
            .min(candidate.initial_abs_psize);
        abs_qty = round_up(abs_qty, candidate.qty_step);
        if abs_qty > candidate.initial_abs_psize {
            abs_qty = candidate.initial_abs_psize;
        }
        if candidate.initial_abs_psize > min_entry_qty + qty_tolerance
            && abs_qty < min_entry_qty - qty_tolerance
        {
            continue;
        }
        let qty = match pside {
            LONG => -abs_qty,
            SHORT => abs_qty,
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
        let mut order = Order {
            qty,
            price,
            order_type,
        };
        order.price = quantize_price(
            order.price,
            candidate.price_step,
            RoundingMode::Nearest,
            "calc_twel_enforcer_actions::price",
        );
        order.qty = quantize_qty(
            order.qty,
            candidate.qty_step,
            RoundingMode::Nearest,
            "calc_twel_enforcer_actions::qty",
        );
        if order.qty.abs() > candidate.initial_abs_psize + qty_tolerance {
            order.qty = match pside {
                LONG => -candidate.initial_abs_psize,
                SHORT => candidate.initial_abs_psize,
                _ => 0.0,
            };
        }
        let remaining_psize = (candidate.initial_abs_psize - order.qty.abs()).max(0.0);
        let new_exposure = calc_wallet_exposure(
            candidate.c_mult,
            balance,
            remaining_psize,
            candidate.position_price,
        );
        let actual_reduce = (candidate.exposure - new_exposure).max(0.0);
        if actual_reduce <= exposure_tolerance {
            continue;
        }
        total_exposure -= actual_reduce;
        actions.push((candidate.idx, order));
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::calc_pprice_diff_int;

    fn pos(
        idx: usize,
        psize: f64,
        pprice: f64,
        mprice: f64,
        wel_base: f64,
        c_mult: f64,
        qty_step: f64,
        price_step: f64,
        min_qty: f64,
        min_cost: f64,
    ) -> TwelEnforcerInputPosition {
        TwelEnforcerInputPosition {
            idx,
            position_size: psize,
            position_price: pprice,
            market_price: mprice,
            base_wallet_exposure_limit: wel_base,
            c_mult,
            qty_step,
            price_step,
            min_qty,
            min_cost,
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
        let wel_base = 0.4; // per coin
        let twel = 0.9; // total limit
        let threshold = 1.0; // target == twel
                             // Position A: psize 8 at 50 => WE = 0.4
                             // Position B: psize 12 at 50 => WE = 0.6
                             // Total 1.0 > twel 0.9; need reduce 0.1 exposure strictly below
        let positions = vec![
            pos(0, 8.0, 50.0, 50.0, wel_base, 1.0, 0.1, 0.1, 0.1, 0.0),
            pos(1, 12.0, 50.0, 49.0, wel_base, 1.0, 0.1, 0.1, 0.1, 0.0),
        ];
        let actions = calc_twel_enforcer_actions(
            LONG,
            threshold,
            twel,
            2,
            balance,
            &positions,
            TwelEnforcerPolicy::ReduceOverweight,
            None,
        );
        assert!(!actions.is_empty(), "should emit reduction actions");
        // Only the most above-floor position (idx 1) should be targeted
        for (idx, _) in &actions {
            assert_eq!(*idx, 1, "expected reductions to target idx 1 only");
        }
        // Apply reductions and verify TWE <= twel while only the least underwater position is trimmed
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
        assert!(
            (we0 - wel_base).abs() < 1e-9,
            "pos0 should remain at its floor share; got {}",
            we0
        );
        assert!(
            (we1 - 0.5).abs() < 1e-9,
            "pos1 should reduce to relieve TWE; got {}",
            we1
        );
        let twe = we0 + we1;
        assert!(
            twe <= twel + 1e-12,
            "TWE not at or below target: {} > {}",
            twe,
            twel
        );
    }

    #[test]
    fn test_twel_reducer_threshold_adjusts_per_position_floor() {
        let balance = 1000.0;
        let wel_base = 0.75;
        let twel = 1.5;
        let threshold = 0.99;
        let positions = vec![
            pos(0, 15.0, 50.0, 49.0, wel_base, 1.0, 0.1, 0.1, 0.1, 0.0),
            pos(1, 15.0, 50.0, 49.5, wel_base, 1.0, 0.1, 0.1, 0.1, 0.0),
        ];

        let actions = calc_twel_enforcer_actions(
            LONG,
            threshold,
            twel,
            2,
            balance,
            &positions,
            TwelEnforcerPolicy::ReduceOverweight,
            None,
        );
        assert!(
            !actions.is_empty(),
            "thresholded TWEL target should be reachable even when positions start at WEL"
        );

        let mut psize = vec![15.0, 15.0];
        for (idx, order) in actions {
            psize[idx] = (psize[idx] - order.qty.abs()).max(0.0);
        }
        let twe = psize
            .iter()
            .map(|size| calc_wallet_exposure(1.0, balance, *size, 50.0))
            .sum::<f64>();
        let qty_step_exposure = 0.1 * 50.0 / balance;
        assert!(
            twe <= twel * threshold + qty_step_exposure + 1e-12,
            "TWE should reduce to within one qty step of the thresholded TWEL target: {} > {}",
            twe,
            twel * threshold
        );
    }

    #[test]
    fn test_twel_reducer_cascades_to_next_position() {
        let balance = 1000.0;
        let twel = 0.4;
        let positions = vec![
            pos(0, 4.0, 50.0, 50.0, 0.2, 1.0, 0.1, 0.1, 0.1, 0.0),
            pos(1, 12.0, 50.0, 48.0, 0.2, 1.0, 0.1, 0.1, 0.1, 0.0),
        ];
        let actions = calc_twel_enforcer_actions(
            LONG,
            1.0,
            twel,
            2,
            balance,
            &positions,
            TwelEnforcerPolicy::ReduceOverweight,
            None,
        );
        assert!(
            actions.iter().all(|(idx, _)| *idx == 1),
            "only idx 1 should be reduced"
        );
        let mut psize = vec![4.0, 12.0];
        let pprice = vec![50.0, 50.0];
        let c_mult = 1.0;
        for (idx, order) in actions {
            let i = idx;
            let dq = order.qty.abs();
            psize[i] = (psize[i] - dq).max(0.0);
        }
        let we0 = calc_wallet_exposure(c_mult, balance, psize[0], pprice[0]);
        let we1 = calc_wallet_exposure(c_mult, balance, psize[1], pprice[1]);
        assert!(
            (we0 - 0.2).abs() < 1e-9,
            "position 0 should stay at its floor; exposure {}",
            we0
        );
        assert!(
            (we1 - 0.2).abs() < 1e-9,
            "position 1 should settle at floor exposure 0.2; got {}",
            we1
        );
        let twe = we0 + we1;
        assert!(
            twe <= twel + 1e-12,
            "TWE should not exceed limit after cascading reductions: {} > {}",
            twe,
            twel
        );
    }

    #[test]
    fn test_twel_reducer_selects_lowest_price_diff() {
        // One pos deep underwater (higher price_diff), one near breakeven; near breakeven should be reduced first (lowest diff)
        let balance = 1000.0;
        let wel_base = 0.4;
        let twel = 1.2;
        let positions = vec![
            // idx 0: position_price 100, market 60 (underwater, large diff)
            pos(0, 8.0, 100.0, 60.0, wel_base, 1.0, 0.1, 0.1, 0.1, 0.0),
            // idx 1: position_price 100, market 99 (near breakeven, small diff)
            pos(1, 7.5, 100.0, 99.0, wel_base, 1.0, 0.1, 0.1, 0.1, 0.0),
        ];
        let actions = calc_twel_enforcer_actions(
            LONG,
            1.0,
            twel,
            2,
            balance,
            &positions,
            TwelEnforcerPolicy::ReduceOverweight,
            None,
        );
        assert!(!actions.is_empty());
        // The action with the smallest price diff should target idx 1
        let price_diff = |idx: usize| -> f64 {
            let pos = positions.iter().find(|p| p.idx == idx).unwrap();
            calc_pprice_diff_int(LONG, pos.position_price, pos.market_price)
        };
        let best_idx = actions
            .iter()
            .min_by(|(a_idx, _), (b_idx, _)| {
                price_diff(*a_idx)
                    .partial_cmp(&price_diff(*b_idx))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| *idx)
            .unwrap();
        assert_eq!(best_idx, 1);
    }

    #[test]
    fn test_twel_reducer_emits_one_order_per_reduce_portfolio_candidate() {
        let balance = 1000.0;
        let twel = 0.9;
        let positions = vec![
            pos(0, 4.0, 100.0, 110.0, 0.5, 1.0, 0.1, 0.1, 0.1, 0.0),
            pos(1, 4.0, 100.0, 109.0, 0.5, 1.0, 0.1, 0.1, 0.1, 0.0),
            pos(2, 2.0, 100.0, 108.0, 0.5, 1.0, 0.1, 0.1, 0.1, 0.0),
        ];

        let actions = calc_twel_enforcer_actions(
            LONG,
            1.0,
            twel,
            3,
            balance,
            &positions,
            TwelEnforcerPolicy::ReducePortfolio,
            None,
        );

        let action_idxs: Vec<usize> = actions.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(
            action_idxs,
            vec![0, 1, 2],
            "reduce_portfolio should emit one order per managed open-position candidate"
        );
    }

    #[test]
    fn test_twel_reducer_second_pass_breaks_below_position_floor() {
        let balance = 1000.0;
        let twel = 1.0;
        let wel_base = 0.2;
        let mut positions = Vec::new();
        for idx in 0..9 {
            let market_price = if idx == 0 { 100.0 } else { 90.0 };
            positions.push(pos(
                idx,
                1.2,
                100.0,
                market_price,
                wel_base,
                1.0,
                0.01,
                0.01,
                0.01,
                0.0,
            ));
        }

        let actions = calc_twel_enforcer_actions(
            LONG,
            1.0,
            twel,
            8,
            balance,
            &positions,
            TwelEnforcerPolicy::ReducePortfolio,
            None,
        );
        assert!(
            actions.iter().any(|(idx, _)| *idx == 0),
            "second pass should reduce the least-stuck position even when all positions are at/below floor"
        );

        let mut psizes = vec![1.2; 9];
        for (idx, order) in actions {
            psizes[idx] = (psizes[idx] - order.qty.abs()).max(0.0);
        }
        let exposures: Vec<f64> = psizes
            .iter()
            .map(|psize| calc_wallet_exposure(1.0, balance, *psize, 100.0))
            .collect();
        let twe: f64 = exposures.iter().sum();
        assert!(
            twe <= twel + 1e-12,
            "second pass should bring TWE to target; got {} > {}",
            twe,
            twel
        );
        let floor = (twel / 8.0_f64).min(wel_base);
        assert!(
            exposures[0] < floor - 1e-12,
            "least-stuck position should be allowed below floor in the second pass"
        );
    }

    #[test]
    fn test_twel_reducer_never_closes_more_than_position_after_rounding() {
        let positions = vec![pos(0, 0.26, 100.0, 100.0, 0.1, 1.0, 0.1, 0.1, 0.0, 30.0)];

        let actions = calc_twel_enforcer_actions(
            LONG,
            1.0,
            0.0001,
            1,
            1000.0,
            &positions,
            TwelEnforcerPolicy::ReducePortfolio,
            None,
        );

        assert!(!actions.is_empty(), "expected TWEL reducer action");
        for (_idx, order) in actions {
            assert!(
                order.qty.abs() <= 0.26 + 1e-12,
                "TWEL close qty must not exceed live position size; got {}",
                order.qty
            );
        }
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
