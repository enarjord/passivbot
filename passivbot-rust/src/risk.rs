use crate::constants::{LONG, SHORT};
use crate::types::{Order, OrderType};
use crate::utils::{calc_pprice_diff_int, calc_wallet_exposure, round_dn, round_up};

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
        target_limit: f64,
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
        let allowed_limit = base_limit * (1.0 + pos.risk_we_excess_allowance_pct.max(0.0));
        let target_limit = if pos.risk_wel_enforcer_threshold > 0.0 {
            allowed_limit * pos.risk_wel_enforcer_threshold
        } else {
            allowed_limit
        };
        if exposure <= target_limit {
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
        let target_psize = (target_limit * balance) / (pos.position_price * pos.c_mult);
        let price_diff = calc_pprice_diff_int(pside, pos.position_price, market_price);
        candidates.push(Candidate {
            idx: pos.idx,
            exposure,
            target_limit,
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

    if candidates.is_empty() || total_exposure <= limit {
        return Vec::new();
    }

    let exposure_tolerance = 1e-9;
    let qty_tolerance = 1e-9;

    loop {
        if total_exposure <= limit + exposure_tolerance {
            break;
        }
        let mut best_idx: Option<usize> = None;
        let mut best_metric = f64::MAX;
        for (idx, candidate) in candidates.iter().enumerate() {
            if candidate.exposure <= candidate.target_limit + exposure_tolerance {
                continue;
            }
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
            candidate.exposure = candidate.target_limit;
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
            candidate.exposure = candidate.target_limit;
            continue;
        }
        let mut qty_reduce = round_up(raw_reduce_psize, candidate.qty_step);
        if qty_reduce > max_reducible_psize {
            qty_reduce = max_reducible_psize;
        }
        if qty_reduce <= qty_tolerance {
            candidate.exposure = candidate.target_limit;
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
