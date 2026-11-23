use crate::entries::calc_min_entry_qty;
use crate::orchestrator::{ExecutionType, PositionSide};
use crate::types::{ExchangeParams, OrderBook, Position};
use crate::utils::{
    calc_pnl_long, calc_pnl_short, round_dn, round_up, tolerant_round_dn_preserve_step,
    tolerant_round_up_preserve_step,
};
use serde::{Deserialize, Serialize};

pub(crate) fn required_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

/// Validate dormant caps too. A numeric cap is never an enablement signal.
pub(crate) fn active_gate_cap(enabled: bool, cap: Option<f64>) -> Result<Option<f64>, String> {
    if cap.is_some_and(|cap| !(0.0..1.0).contains(&cap)) {
        return Err("loss_gate_max_drawdown_fraction must be finite and in [0, 1)".into());
    }
    if enabled {
        return cap
            .map(Some)
            .ok_or_else(|| "enabled loss gate requires loss_gate_max_drawdown_fraction".into());
    }
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CloseProjection {
    pub execution_price: f64,
    pub immediately_executable: bool,
    pub minimum_qty: f64,
    position_price: f64,
    pside: PositionSide,
    c_mult: f64,
    fee_fraction: f64,
}

impl CloseProjection {
    pub(crate) fn pnl(self, qty: f64) -> f64 {
        let gross_pnl = match self.pside {
            PositionSide::Long => {
                calc_pnl_long(self.position_price, self.execution_price, qty, self.c_mult)
            }
            PositionSide::Short => {
                calc_pnl_short(self.position_price, self.execution_price, qty, self.c_mult)
            }
        };
        gross_pnl - qty.abs() * self.execution_price * self.c_mult * self.fee_fraction
    }

    pub(crate) fn loss(self, qty: f64) -> f64 {
        (-self.pnl(qty)).max(0.0)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReservationInput {
    pub pside: PositionSide,
    pub qty: f64,
    pub price: f64,
    pub execution_type: ExecutionType,
    pub post_only: bool,
    pub position: Position,
    pub order_book: OrderBook,
    pub exchange: ExchangeParams,
    pub market_order_slippage_fraction: f64,
    pub loss_gate_enabled: bool,
    #[serde(deserialize_with = "required_optional")]
    pub loss_gate_max_drawdown_fraction: Option<f64>,
}

#[derive(Serialize)]
pub(crate) struct ReservationOutput {
    pub projected_pnl: f64,
    pub estimated_loss: f64,
    pub execution_price: f64,
    pub immediately_executable: bool,
}

impl ReservationInput {
    pub(crate) fn evaluate(self) -> Result<ReservationOutput, String> {
        active_gate_cap(self.loss_gate_enabled, self.loss_gate_max_drawdown_fraction)?;
        self.exchange.validate_required()?;
        let projection = project_close(
            self.pside,
            self.qty,
            self.price,
            self.execution_type,
            self.post_only,
            &self.position,
            &self.order_book,
            &self.exchange,
            self.market_order_slippage_fraction,
        )
        .ok_or("invalid close reservation inputs")?;
        let qty = self.qty.abs().min(self.position.size.abs());
        let projected_pnl = projection.pnl(qty);
        if !projected_pnl.is_finite() {
            return Err("nonfinite close reservation projection".into());
        }
        Ok(ReservationOutput {
            projected_pnl,
            estimated_loss: (-projected_pnl).max(0.0),
            execution_price: projection.execution_price,
            immediately_executable: projection.immediately_executable,
        })
    }
}

/// Use the submitted limit as the conservative price for a crossing limit.
/// Market sizing uses executable touch while its loss uses adverse slippage.
pub(crate) fn project_close(
    pside: PositionSide,
    qty: f64,
    limit_price: f64,
    execution_type: ExecutionType,
    post_only: bool,
    position: &Position,
    book: &OrderBook,
    exchange: &ExchangeParams,
    slippage: f64,
) -> Option<CloseProjection> {
    let buy = pside == PositionSide::Short;
    let correct_signs = if buy {
        qty > 0.0 && position.size < 0.0
    } else {
        qty < 0.0 && position.size > 0.0
    };
    if !correct_signs
        || !qty.is_finite()
        || ![
            position.size.abs(),
            position.price,
            limit_price,
            book.bid,
            book.ask,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0)
        || !slippage.is_finite()
        || slippage < 0.0
    {
        return None;
    }
    let touch = if buy { book.ask } else { book.bid };
    let crosses = if buy {
        limit_price >= touch
    } else {
        limit_price <= touch
    };
    let is_market = execution_type == ExecutionType::Market;
    let immediately_executable = is_market || (!post_only && crosses);
    let fee_fraction = if immediately_executable {
        exchange.taker_fee_fraction
    } else {
        exchange.maker_fee_fraction
    };
    let price_step = exchange.price_step.max(f64::EPSILON);
    let execution_price = if !is_market {
        limit_price
    } else if buy {
        round_up(touch * (1.0 + slippage), price_step)
    } else {
        round_dn(touch * (1.0 - slippage), price_step).max(price_step)
    };
    let minimum_price = if is_market { touch } else { limit_price };
    let minimum = calc_min_entry_qty(minimum_price, exchange);
    let minimum_qty = tolerant_round_up_preserve_step(minimum, exchange.qty_step);
    let projection = CloseProjection {
        execution_price,
        immediately_executable,
        minimum_qty,
        position_price: position.price,
        pside,
        c_mult: exchange.c_mult,
        fee_fraction,
    };
    (execution_price.is_finite()
        && execution_price > 0.0
        && minimum_qty.is_finite()
        && projection
            .pnl(qty.abs().min(position.size.abs()))
            .is_finite())
    .then_some(projection)
}

/// Normalize order ranking to a shared binary32 lattice with a 2^-20 fraction
/// quantum. Keep economic projections at their original precision.
fn close_rank_bucket(fraction: f64) -> f32 {
    const SCALE: f32 = 1_048_576.0;
    let fraction = fraction as f32;
    // At |fraction| >= 16, binary32 already has a spacing of at least 2^-20.
    let ranked = if fraction.abs() < 16.0 {
        (fraction * SCALE).round() / SCALE
    } else {
        fraction
    };
    if ranked == 0.0 {
        0.0
    } else {
        ranked
    }
}

/// Executable closes compare entry-relative touch; passive closes compare
/// distance to their own resting touch first.
pub(crate) fn close_fill_priority(
    pside: PositionSide,
    price: f64,
    position_price: f64,
    book: &OrderBook,
    immediately_executable: bool,
) -> (bool, f32, f32) {
    let short = pside == PositionSide::Short;
    let (gap, reference_price) = if immediately_executable {
        (0.0, if short { book.ask } else { book.bid })
    } else {
        let touch = if short { book.bid } else { book.ask };
        (close_rank_bucket(((price - touch) / touch).abs()), price)
    };
    let distance = if short {
        (position_price - reference_price) / position_price
    } else {
        (reference_price - position_price) / position_price
    };
    (immediately_executable, gap, close_rank_bucket(distance))
}

pub(crate) fn compare_close_fill_priority(
    (a_immediate, a_gap, a_distance): (bool, f32, f32),
    (b_immediate, b_gap, b_distance): (bool, f32, f32),
) -> std::cmp::Ordering {
    b_immediate
        .cmp(&a_immediate)
        .then_with(|| a_gap.total_cmp(&b_gap))
        .then_with(|| b_distance.total_cmp(&a_distance))
}

/// Round at the venue step only after computing the positive allowance target.
/// A below-minimum lot may be kept solely as an already-finalized whole close.
pub(crate) fn size_to_loss_budget(
    proposed: f64,
    position: f64,
    qty_step: f64,
    projection: CloseProjection,
    headroom: Option<f64>,
    allow_below_minimum_full_close: bool,
) -> Option<f64> {
    let upper = proposed.min(position);
    let tolerance = f64::EPSILON * upper.max(projection.minimum_qty) * 4.0;
    let full_close_exception = allow_below_minimum_full_close
        && upper == position
        && position + tolerance < projection.minimum_qty;
    let legal_upper = if full_close_exception {
        upper
    } else {
        tolerant_round_dn_preserve_step(upper, qty_step).min(upper)
    };
    if legal_upper <= 0.0
        || (!full_close_exception && legal_upper + tolerance < projection.minimum_qty)
    {
        return None;
    }
    let Some(headroom) = headroom else {
        return Some(legal_upper);
    };
    let loss = projection.loss(legal_upper);
    if loss <= headroom {
        return Some(legal_upper);
    }
    if headroom <= 0.0 {
        return None;
    }
    if full_close_exception {
        return Some(legal_upper);
    }
    // Even if the quotient underflows to zero, one venue step is the least
    // positive legal order. The projected loss of that lot consumes headroom.
    let raw = legal_upper * (headroom / loss);
    let rounded = tolerant_round_up_preserve_step(raw, qty_step)
        .max(qty_step)
        .max(projection.minimum_qty);
    Some(rounded.min(legal_upper))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct UnstuckExposure {
    pub quantity: f64,
    pub entry_cost_per_unit: f64,
    pub balance: f64,
    pub effective_wel: f64,
    pub trigger: f64,
}

impl UnstuckExposure {
    pub(crate) fn ratio(self, qty: f64, pnl: f64) -> Option<f64> {
        let balance = self.balance + pnl;
        if !(balance.is_finite() && balance > 0.0 && qty < self.quantity && qty >= 0.0) {
            return None;
        }
        let ratio =
            (self.quantity - qty) * self.entry_cost_per_unit / (balance * self.effective_wel);
        ratio.is_finite().then_some(ratio)
    }

    pub(crate) fn upper_qty(self, step: f64, projection: CloseProjection) -> Option<f64> {
        let before = self.ratio(0.0, 0.0)?;
        if !(self.trigger > 0.0 && self.trigger < before && step > 0.0) {
            return None;
        }
        // The sign of the derivative is constant as long as the projected
        // balance is positive. Reject a neutral/increasing exposure ratio.
        let unit_pnl = projection.pnl(1.0);
        let direction = self.quantity.mul_add(unit_pnl, self.balance);
        if !(direction.is_finite() && direction > 0.0) {
            return None;
        }

        // Binary search integer venue lots. This avoids dividing by a nearly
        // zero coefficient and tests the same projected PnL used for sizing.
        let count = (self.quantity / step).ceil() + 1.0;
        if !(1.0 < count && count <= (1u64 << f64::MANTISSA_DIGITS) as f64) {
            return None;
        }
        let mut lo = 0;
        let mut hi = count as u64;
        let lot = |count| tolerant_round_dn_preserve_step(count as f64 * step, step);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let qty = lot(mid);
            if self
                .ratio(qty, projection.pnl(qty))
                .is_some_and(|ratio| ratio >= self.trigger)
            {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let qty = lot(lo);
        self.ratio(qty, projection.pnl(qty))
            .filter(|ratio| qty > 0.0 && *ratio < before && *ratio >= self.trigger)
            .map(|_| qty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection(
        pside: PositionSide,
        price: f64,
        step: f64,
        minimum: f64,
    ) -> (CloseProjection, ExchangeParams) {
        let size = if pside == PositionSide::Long {
            1000.0
        } else {
            -1000.0
        };
        let exchange = ExchangeParams {
            qty_step: step,
            min_qty: minimum,
            min_cost: 0.0,
            c_mult: 1.0,
            maker_fee_fraction: 0.0,
            taker_fee_fraction: 0.0,
            ..ExchangeParams::default()
        };
        let projection = project_close(
            pside,
            -size,
            price,
            ExecutionType::Limit,
            true,
            &Position { size, price: 100.0 },
            &OrderBook {
                bid: price,
                ask: price,
            },
            &exchange,
            0.0,
        )
        .unwrap();
        (projection, exchange)
    }

    #[test]
    fn close_rank_buckets_preserve_meaningful_differences_and_canonicalize_zero() {
        let quantum = 1.0f64 / 1_048_576.0;
        for sign in [-1.0, 1.0] {
            assert_eq!(
                close_rank_bucket(sign * 0.1399999999999999),
                close_rank_bucket(sign * 0.14),
            );
            assert_ne!(
                close_rank_bucket(sign * 0.14),
                close_rank_bucket(sign * 0.14001),
            );
            assert_eq!(
                close_rank_bucket(sign * quantum * 0.5),
                (sign * quantum) as f32,
            );
            assert_eq!(
                close_rank_bucket(sign * quantum * 0.25).to_bits(),
                0.0f32.to_bits(),
            );
            assert_eq!(close_rank_bucket(sign * 16.0), (sign * 16.0) as f32);
            assert_eq!(
                close_rank_bucket(sign * f64::from(f32::MAX)),
                (sign as f32) * f32::MAX
            );
        }
    }

    #[test]
    fn passive_close_ranks_use_absolute_resting_gap_then_side_adjusted_distance() {
        let long = close_fill_priority(
            PositionSide::Long,
            57.0,
            50.0,
            &OrderBook {
                bid: 49.0,
                ask: 50.0,
            },
            false,
        );
        let short = close_fill_priority(
            PositionSide::Short,
            86.0,
            100.0,
            &OrderBook {
                bid: 100.0,
                ask: 101.0,
            },
            false,
        );
        assert_eq!(
            long,
            (false, close_rank_bucket(0.14), close_rank_bucket(0.14))
        );
        assert_eq!(
            compare_close_fill_priority(long, short),
            std::cmp::Ordering::Equal
        );
        let farther = close_fill_priority(
            PositionSide::Short,
            85.999,
            100.0,
            &OrderBook {
                bid: 100.0,
                ask: 101.0,
            },
            false,
        );
        assert_eq!(
            compare_close_fill_priority(long, farther),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn allowance_sizing_rounds_up_to_one_legal_lot_and_preserves_exact_steps() {
        for (side, price) in [(PositionSide::Long, 99.0), (PositionSide::Short, 101.0)] {
            for (step, minimum, budget, expected) in [
                (0.7, 0.0, 100.0, 100.1),
                (0.1, 0.0, 100.0, 100.0),
                (1.0, 120.0, 100.0, 120.0),
                (1.0, 120.0, f64::from(f32::MIN_POSITIVE), 120.0),
            ] {
                let (projection, _) = projection(side, price, step, minimum);
                let qty = size_to_loss_budget(500.5, 1000.0, step, projection, Some(budget), false);
                assert_eq!(qty, Some(expected));
                assert_eq!(projection.loss(qty.unwrap()), expected);
                assert_eq!(
                    size_to_loss_budget(500.5, 1000.0, step, projection, Some(0.0), false),
                    None
                );
            }
        }
    }

    #[test]
    fn venue_minimum_and_position_dust_respect_quantity_caps() {
        let (projection, _) = projection(PositionSide::Long, 99.0, 0.1, 0.5);
        assert_eq!(
            size_to_loss_budget(0.4, 0.4, 0.1, projection, Some(0.01), true),
            Some(0.4)
        );
        assert_eq!(
            size_to_loss_budget(0.4, 0.4, 0.1, projection, Some(0.0), true),
            None
        );
        assert_eq!(
            size_to_loss_budget(0.4, 0.4, 0.1, projection, Some(10.0), false),
            None
        );
        assert_eq!(
            size_to_loss_budget(0.4, 0.6, 0.1, projection, Some(10.0), true),
            None
        );
    }

    #[test]
    fn exposure_bound_reprojects_balance_and_checks_realized_liquidity() {
        let exposure = UnstuckExposure {
            quantity: 10.0,
            entry_cost_per_unit: 100.0,
            balance: 1000.0,
            effective_wel: 1.0,
            trigger: 0.5,
        };
        for (side, loss_price, gain_price) in [
            (PositionSide::Long, 90.0, 110.0),
            (PositionSide::Short, 110.0, 90.0),
        ] {
            let (loss, _) = projection(side, loss_price, 0.1, 6.0);
            assert_eq!(exposure.upper_qty(0.1, loss), Some(5.2));
            assert!(exposure.ratio(5.2, loss.pnl(5.2)).unwrap() >= exposure.trigger);
            assert!(exposure.ratio(5.3, loss.pnl(5.3)).unwrap() < exposure.trigger);
            assert_eq!(
                size_to_loss_budget(5.2, 10.0, 0.1, loss, Some(30.0), false),
                None
            );

            let (gain, _) = projection(side, gain_price, 0.1, 0.0);
            assert_eq!(exposure.upper_qty(0.1, gain), Some(4.7));
            assert_eq!(
                size_to_loss_budget(4.7, 10.0, 0.1, gain, Some(0.0), false),
                Some(4.7)
            );

            let undercapitalized = UnstuckExposure {
                balance: 50.0,
                ..exposure
            };
            assert_eq!(undercapitalized.upper_qty(0.1, loss), None);
            assert_eq!(undercapitalized.ratio(5.0, loss.pnl(5.0)), None);
        }
    }

    #[test]
    fn execution_routes_apply_limit_price_and_the_applicable_signed_fee() {
        let (_, mut exchange) = projection(PositionSide::Long, 100.0, 0.1, 0.0);
        exchange.maker_fee_fraction = -0.001;
        exchange.taker_fee_fraction = 0.002;
        let position = Position {
            size: 3.0,
            price: 99.0,
        };
        let book = OrderBook {
            bid: 100.0,
            ask: 101.0,
        };
        let projection = |route, post_only| {
            project_close(
                PositionSide::Long,
                -3.0,
                99.0,
                route,
                post_only,
                &position,
                &book,
                &exchange,
                0.01,
            )
            .unwrap()
        };
        let crossing = projection(ExecutionType::Limit, false);
        assert!(crossing.immediately_executable);
        assert_eq!(crossing.execution_price, 99.0);
        assert_eq!(crossing.pnl(3.0), -0.594);
        let posted = projection(ExecutionType::Limit, true);
        assert!(!posted.immediately_executable);
        assert_eq!(posted.pnl(3.0), 0.297);
        let market = projection(ExecutionType::Market, false);
        assert!(market.immediately_executable);
        assert_eq!(market.execution_price, 99.0);
        assert_eq!(market.pnl(3.0), crossing.pnl(3.0));
    }
}
