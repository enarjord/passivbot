use super::{
    BinaryOutcomeMarketSpec, LiquidityRole, Outcome, OutcomeError, OutcomeFeeSchedule, OutcomeFill,
    OutcomeFillAccounting, OutcomeLedger, OutcomeLimitOrder, OutcomeOrderSide, OutcomePriceGrid,
    OutcomeSettlement,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

const PRICE_EPSILON: f64 = 1e-12;
const QTY_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeCandle {
    pub timestamp_ms: u64,
    /// Native book from which the trades in this candle originated.
    pub outcome: Outcome,
    /// Prices are expressed in the canonical YES coordinate, including for NO-book candles.
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OutcomeCandle {
    pub fn validate(&self, payout_unit: f64) -> Result<(), OutcomeError> {
        for price in [self.open, self.high, self.low, self.close] {
            if !price.is_finite() || price < 0.0 || price > payout_unit {
                return Err(OutcomeError::InvalidPrice(price));
            }
        }
        if self.high + PRICE_EPSILON < self.low
            || self.open < self.low - PRICE_EPSILON
            || self.open > self.high + PRICE_EPSILON
            || self.close < self.low - PRICE_EPSILON
            || self.close > self.high + PRICE_EPSILON
        {
            return Err(OutcomeError::InvalidPrice(self.close));
        }
        if !self.volume.is_finite() || self.volume < 0.0 {
            return Err(OutcomeError::InvalidQuantity(self.volume));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestingOutcomeOrder {
    pub order: OutcomeLimitOrder,
    pub remaining_qty: f64,
    sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulatedOutcomeFill {
    pub order_id: String,
    pub fill: OutcomeFill,
    pub accounting: OutcomeFillAccounting,
}

/// Deterministic single-market execution and accounting kernel.
///
/// The initial fill policy consumes trade-derived one-second candles. Positive volume and a
/// strict trade-through are required: low < bid or high > ask. A touch does not fill. As in the
/// existing candle backtester, candle volume is an eligibility gate rather than a simulated
/// quantity cap. Within the bot's eligible orders, canonical price and placement sequence decide
/// priority.
#[derive(Debug, Clone)]
pub struct SingleOutcomeSimulator {
    market: BinaryOutcomeMarketSpec,
    fee_schedule: OutcomeFeeSchedule,
    ledger: OutcomeLedger,
    open_orders: Vec<RestingOutcomeOrder>,
    fills: Vec<SimulatedOutcomeFill>,
    next_sequence: u64,
}

impl SingleOutcomeSimulator {
    pub fn new(
        market: BinaryOutcomeMarketSpec,
        fee_schedule: OutcomeFeeSchedule,
        starting_collateral: f64,
    ) -> Result<Self, OutcomeError> {
        market.validate()?;
        fee_schedule.validate()?;
        let ledger = OutcomeLedger::new(starting_collateral, market.payout_unit)?;
        Ok(Self {
            market,
            fee_schedule,
            ledger,
            open_orders: Vec::new(),
            fills: Vec::new(),
            next_sequence: 0,
        })
    }

    pub fn place_order(
        &mut self,
        order: OutcomeLimitOrder,
        timestamp_ms: u64,
    ) -> Result<(), OutcomeError> {
        self.ensure_trading(timestamp_ms)?;
        self.expire_orders(timestamp_ms);
        self.market.validate()?;
        if order.close_all {
            let residual_qty = self.ledger.yes_qty() - self.ledger.no_qty();
            let closes_full_residual = match order.outcome {
                Outcome::Yes => {
                    residual_qty > QTY_EPSILON && (order.qty - residual_qty).abs() <= QTY_EPSILON
                }
                Outcome::No => {
                    residual_qty < -QTY_EPSILON && (order.qty + residual_qty).abs() <= QTY_EPSILON
                }
            };
            if !closes_full_residual {
                return Err(OutcomeError::InvalidQuantity(order.qty));
            }
            order.validate_close_all(&self.market)?;
        } else {
            self.market.validate_order(&order)?;
        }
        if !order.post_only {
            return Err(OutcomeError::UnsupportedOrderFeature(
                "non_post_only".to_string(),
            ));
        }
        if order.post_only && !self.market.capabilities.supports_post_only {
            return Err(OutcomeError::UnsupportedOrderFeature(
                "post_only".to_string(),
            ));
        }
        if order.expires_at_ms.is_some() && !self.market.capabilities.supports_gtd {
            return Err(OutcomeError::UnsupportedOrderFeature("gtd".to_string()));
        }
        if order
            .expires_at_ms
            .is_some_and(|expires_at_ms| expires_at_ms <= timestamp_ms)
        {
            return Err(OutcomeError::InvalidMarket(
                "order expiry must be later than its placement timestamp".to_string(),
            ));
        }
        if self
            .open_orders
            .iter()
            .any(|resting| resting.order.order_id == order.order_id)
        {
            return Err(OutcomeError::DuplicateOrderId(order.order_id));
        }

        match order.side {
            OutcomeOrderSide::Buy => {
                let required = self.collateral_reservation(&order, order.qty)?;
                let available = self.available_collateral();
                if required > available + QTY_EPSILON {
                    return Err(OutcomeError::InsufficientCollateral {
                        required,
                        available,
                    });
                }
            }
            OutcomeOrderSide::Sell => {
                let available = self.available_inventory(order.outcome);
                if order.qty > available + QTY_EPSILON {
                    return Err(OutcomeError::InsufficientInventory {
                        outcome: order.outcome,
                        required: order.qty,
                        available,
                    });
                }
            }
        }

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        let remaining_qty = order.qty;
        self.open_orders.push(RestingOutcomeOrder {
            order,
            remaining_qty,
            sequence,
        });
        Ok(())
    }

    pub fn cancel_order(&mut self, order_id: &str) -> Result<RestingOutcomeOrder, OutcomeError> {
        let index = self
            .open_orders
            .iter()
            .position(|resting| resting.order.order_id == order_id)
            .ok_or_else(|| OutcomeError::UnknownOrderId(order_id.to_string()))?;
        Ok(self.open_orders.remove(index))
    }

    pub fn update_price_grid(
        &mut self,
        price_grid: OutcomePriceGrid,
        timestamp_ms: u64,
    ) -> Result<(), OutcomeError> {
        self.ensure_market_open(timestamp_ms)?;
        self.market.replace_price_grid(price_grid)?;
        // A venue grid transition makes any incompatible resting price non-executable.
        self.open_orders.retain(|resting| {
            if resting.order.close_all {
                resting.order.validate_close_all(&self.market).is_ok()
            } else {
                self.market.validate_order(&resting.order).is_ok()
            }
        });
        Ok(())
    }

    pub fn process_candle(
        &mut self,
        candle: &OutcomeCandle,
    ) -> Result<Vec<SimulatedOutcomeFill>, OutcomeError> {
        self.ensure_market_open(candle.timestamp_ms)?;
        self.expire_orders(candle.timestamp_ms);
        candle.validate(self.market.payout_unit)?;
        if candle.volume <= 0.0 {
            return Ok(Vec::new());
        }

        let mut candidates: Vec<(usize, f64, u64, bool)> = self
            .open_orders
            .iter()
            .enumerate()
            .filter_map(|(index, resting)| {
                if resting.remaining_qty <= QTY_EPSILON {
                    return None;
                }
                if !self.market.capabilities.complementary_books_merged
                    && resting.order.outcome != candle.outcome
                {
                    return None;
                }
                let order_exposure_delta = resting.order.canonical_yes_exposure_delta();
                let order_canonical_price = resting
                    .order
                    .canonical_yes_price(self.market.payout_unit)
                    .ok()?;
                let traded_through = if order_exposure_delta > 0.0 {
                    candle.low < order_canonical_price
                } else {
                    candle.high > order_canonical_price
                };
                traded_through.then_some((
                    index,
                    order_canonical_price,
                    resting.sequence,
                    order_exposure_delta > 0.0,
                ))
            })
            .collect();

        candidates.sort_by(|left, right| {
            let side_order = right.3.cmp(&left.3);
            let price_order = if left.3 {
                right.1.partial_cmp(&left.1)
            } else {
                left.1.partial_cmp(&right.1)
            }
            .unwrap_or(Ordering::Equal);
            side_order
                .then(price_order)
                .then_with(|| left.2.cmp(&right.2))
        });

        let mut produced = Vec::new();
        for (index, _, _, _) in candidates {
            let resting = &self.open_orders[index];
            let fill_qty = resting.remaining_qty;
            let order = resting.order.clone();
            let fill = OutcomeFill {
                timestamp_ms: candle.timestamp_ms,
                outcome: order.outcome,
                side: order.side,
                price: order.price,
                qty: fill_qty,
                liquidity_role: LiquidityRole::Maker,
            };
            let accounting = self.ledger.apply_fill(&fill, &self.fee_schedule)?;
            self.open_orders[index].remaining_qty -= fill_qty;
            let simulated_fill = SimulatedOutcomeFill {
                order_id: order.order_id,
                fill,
                accounting,
            };
            self.fills.push(simulated_fill.clone());
            produced.push(simulated_fill);
        }
        self.open_orders
            .retain(|resting| resting.remaining_qty > QTY_EPSILON);
        Ok(produced)
    }

    pub fn settle(
        &mut self,
        timestamp_ms: u64,
        yes_fraction: f64,
    ) -> Result<OutcomeSettlement, OutcomeError> {
        if timestamp_ms < self.market.trading_closes_ms {
            return Err(OutcomeError::MarketNotTrading(timestamp_ms));
        }
        self.open_orders.clear();
        self.ledger
            .settle_with_fee_schedule(yes_fraction, &self.fee_schedule)
    }

    pub fn worst_case_settlement_equity(&self) -> Result<f64, OutcomeError> {
        self.ledger.worst_case_settlement_equity(&self.fee_schedule)
    }

    pub fn split(&mut self, qty: f64, yes_reference_price: f64) -> Result<(), OutcomeError> {
        if !self.market.capabilities.supports_split {
            return Err(OutcomeError::UnsupportedOrderFeature("split".to_string()));
        }
        let required = qty * self.market.payout_unit;
        let available = self.available_collateral();
        if required > available + QTY_EPSILON {
            return Err(OutcomeError::InsufficientCollateral {
                required,
                available,
            });
        }
        self.ledger.split(qty, yes_reference_price)
    }

    pub fn merge(&mut self, qty: f64) -> Result<f64, OutcomeError> {
        if !self.market.capabilities.supports_merge {
            return Err(OutcomeError::UnsupportedOrderFeature("merge".to_string()));
        }
        for outcome in [Outcome::Yes, Outcome::No] {
            let available = self.available_inventory(outcome);
            if qty > available + QTY_EPSILON {
                return Err(OutcomeError::InsufficientInventory {
                    outcome,
                    required: qty,
                    available,
                });
            }
        }
        self.ledger.merge(qty)
    }

    pub fn expire_orders(&mut self, timestamp_ms: u64) -> Vec<RestingOutcomeOrder> {
        let mut retained = Vec::with_capacity(self.open_orders.len());
        let mut expired = Vec::new();
        for order in self.open_orders.drain(..) {
            let is_expired = timestamp_ms >= self.market.trading_closes_ms
                || order
                    .order
                    .expires_at_ms
                    .is_some_and(|expiry| timestamp_ms >= expiry);
            if is_expired {
                expired.push(order);
            } else {
                retained.push(order);
            }
        }
        self.open_orders = retained;
        expired
    }

    pub fn reserved_collateral(&self) -> f64 {
        self.open_orders
            .iter()
            .filter(|resting| resting.order.side == OutcomeOrderSide::Buy)
            .map(|resting| {
                self.collateral_reservation(&resting.order, resting.remaining_qty)
                    .unwrap_or(f64::INFINITY)
            })
            .sum()
    }

    pub fn available_collateral(&self) -> f64 {
        (self.ledger.collateral() - self.reserved_collateral()).max(0.0)
    }

    pub fn reserved_inventory(&self, outcome: Outcome) -> f64 {
        self.open_orders
            .iter()
            .filter(|resting| {
                resting.order.outcome == outcome && resting.order.side == OutcomeOrderSide::Sell
            })
            .map(|resting| resting.remaining_qty)
            .sum()
    }

    pub fn available_inventory(&self, outcome: Outcome) -> f64 {
        let inventory = match outcome {
            Outcome::Yes => self.ledger.yes_qty(),
            Outcome::No => self.ledger.no_qty(),
        };
        (inventory - self.reserved_inventory(outcome)).max(0.0)
    }

    pub fn ledger(&self) -> &OutcomeLedger {
        &self.ledger
    }

    pub fn open_orders(&self) -> &[RestingOutcomeOrder] {
        &self.open_orders
    }

    pub fn fills(&self) -> &[SimulatedOutcomeFill] {
        &self.fills
    }

    fn ensure_trading(&self, timestamp_ms: u64) -> Result<(), OutcomeError> {
        if self.ledger.is_settled()
            || timestamp_ms < self.market.order_entry_opens_ms
            || timestamp_ms >= self.market.trading_closes_ms
        {
            Err(OutcomeError::MarketNotTrading(timestamp_ms))
        } else {
            Ok(())
        }
    }

    fn ensure_market_open(&self, timestamp_ms: u64) -> Result<(), OutcomeError> {
        if self.ledger.is_settled()
            || timestamp_ms < self.market.trading_opens_ms
            || timestamp_ms >= self.market.trading_closes_ms
        {
            Err(OutcomeError::MarketNotTrading(timestamp_ms))
        } else {
            Ok(())
        }
    }

    fn collateral_reservation(
        &self,
        order: &OutcomeLimitOrder,
        qty: f64,
    ) -> Result<f64, OutcomeError> {
        let fee = self.fee_schedule.calculate(
            &OutcomeFill {
                timestamp_ms: self.market.trading_opens_ms,
                outcome: order.outcome,
                side: order.side,
                price: order.price,
                qty,
                liquidity_role: LiquidityRole::Maker,
            },
            self.market.payout_unit,
        )?;
        // Rebate proceeds are not available until the order fills.
        Ok(qty * order.price + fee.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{
        OutcomeFeeFormula, OutcomeFeeIncidence, OutcomePriceGrid, OutcomeVenueCapabilities,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual {actual} != expected {expected}"
        );
    }

    fn market(merged: bool) -> BinaryOutcomeMarketSpec {
        BinaryOutcomeMarketSpec {
            venue: "fixture".to_string(),
            market_id: "market-1".to_string(),
            yes_asset_id: "yes-1".to_string(),
            no_asset_id: "no-1".to_string(),
            payout_unit: 1.0,
            min_price: 0.001,
            max_price: 0.999,
            price_grid: OutcomePriceGrid::FixedStep { step: 0.001 },
            qty_step: 0.1,
            min_qty: 0.1,
            min_notional: 0.0,
            trading_opens_ms: 1_000,
            order_entry_opens_ms: 1_000,
            trading_closes_ms: 10_000,
            scheduled_event_ms: 10_000,
            capabilities: OutcomeVenueCapabilities {
                complementary_books_merged: merged,
                supports_split: true,
                supports_merge: true,
                supports_redeem: true,
                supports_post_only: true,
                supports_gtd: true,
                sell_requires_inventory: true,
            },
        }
    }

    fn order(
        id: &str,
        outcome: Outcome,
        side: OutcomeOrderSide,
        price: f64,
        qty: f64,
    ) -> OutcomeLimitOrder {
        OutcomeLimitOrder {
            order_id: id.to_string(),
            outcome,
            side,
            price,
            qty,
            close_all: false,
            post_only: true,
            expires_at_ms: None,
        }
    }

    fn candle(outcome: Outcome, high: f64, low: f64, close: f64, volume: f64) -> OutcomeCandle {
        OutcomeCandle {
            timestamp_ms: 2_000,
            outcome,
            open: close,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn exact_touch_and_zero_volume_do_not_fill_but_trade_through_does() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 10.0).unwrap();
        simulator
            .place_order(
                order("buy-yes", Outcome::Yes, OutcomeOrderSide::Buy, 0.4, 2.0),
                1_500,
            )
            .unwrap();
        assert!(simulator
            .process_candle(&candle(Outcome::Yes, 0.45, 0.4, 0.42, 2.0))
            .unwrap()
            .is_empty());
        assert!(simulator
            .process_candle(&candle(Outcome::Yes, 0.45, 0.39, 0.42, 0.0))
            .unwrap()
            .is_empty());

        let fills = simulator
            .process_candle(&candle(Outcome::Yes, 0.45, 0.4 - 1e-13, 0.42, 0.7))
            .unwrap();
        assert_eq!(fills.len(), 1);
        assert_close(fills[0].fill.qty, 2.0);
        assert_close(fills[0].fill.price, 0.4);
        assert!(simulator.open_orders().is_empty());
        assert_close(simulator.ledger().yes_qty(), 2.0);
    }

    #[test]
    fn price_grid_change_refreshes_simulator_executable_bounds() {
        let mut initial_market = market(false);
        initial_market.min_price = 0.01;
        initial_market.max_price = 0.99;
        initial_market.price_grid = OutcomePriceGrid::FixedStep { step: 0.01 };
        let mut simulator =
            SingleOutcomeSimulator::new(initial_market, OutcomeFeeSchedule::zero(), 10.0).unwrap();

        simulator
            .update_price_grid(OutcomePriceGrid::FixedStep { step: 0.001 }, 1_500)
            .unwrap();
        simulator
            .place_order(
                order(
                    "new-upper-bound",
                    Outcome::Yes,
                    OutcomeOrderSide::Buy,
                    0.999,
                    0.1,
                ),
                1_600,
            )
            .unwrap();

        assert_close(simulator.market.min_price, 0.001);
        assert_close(simulator.market.max_price, 0.999);
    }

    #[test]
    fn non_post_only_orders_are_rejected_until_taker_execution_is_modeled() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 10.0).unwrap();
        let mut taker = order("taker", Outcome::Yes, OutcomeOrderSide::Buy, 0.4, 1.0);
        taker.post_only = false;

        assert_eq!(
            simulator.place_order(taker, 1_500),
            Err(OutcomeError::UnsupportedOrderFeature(
                "non_post_only".to_string()
            ))
        );
        assert!(simulator.open_orders().is_empty());
    }

    #[test]
    fn merged_book_matches_complementary_native_actions_in_canonical_price_priority() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(true), OutcomeFeeSchedule::zero(), 10.0).unwrap();
        simulator
            .place_order(
                order("buy-yes", Outcome::Yes, OutcomeOrderSide::Buy, 0.34, 1.0),
                1_500,
            )
            .unwrap();
        let fills = simulator
            .process_candle(&candle(Outcome::No, 0.34, 0.33, 0.335, 0.4))
            .unwrap();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].order_id, "buy-yes");
        assert_close(fills[0].accounting.canonical_yes_price, 0.34);
        assert_close(simulator.ledger().net_yes_exposure(), 1.0);
    }

    #[test]
    fn separate_books_do_not_cross_fill_complementary_assets() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 10.0).unwrap();
        simulator
            .place_order(
                order("buy-yes", Outcome::Yes, OutcomeOrderSide::Buy, 0.34, 1.0),
                1_500,
            )
            .unwrap();
        assert!(simulator
            .process_candle(&candle(Outcome::No, 0.67, 0.66, 0.665, 1.0))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn canonical_candle_can_fill_bids_and_asks_but_not_at_exact_touch() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 10.0).unwrap();
        simulator.split(2.0, 0.5).unwrap();
        simulator
            .place_order(
                order("bid", Outcome::Yes, OutcomeOrderSide::Buy, 0.4, 1.0),
                1_500,
            )
            .unwrap();
        simulator
            .place_order(
                order("ask", Outcome::Yes, OutcomeOrderSide::Sell, 0.6, 1.0),
                1_500,
            )
            .unwrap();
        let fills = simulator
            .process_candle(&candle(Outcome::Yes, 0.6 + 1e-13, 0.4 - 1e-13, 0.5, 0.1))
            .unwrap();
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].order_id, "bid");
        assert_eq!(fills[1].order_id, "ask");
    }

    #[test]
    fn reservations_prevent_oversubscribed_buy_orders() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 1.0).unwrap();
        simulator
            .place_order(
                order("first", Outcome::Yes, OutcomeOrderSide::Buy, 0.6, 1.0),
                1_500,
            )
            .unwrap();
        let before = simulator.open_orders().to_vec();
        assert!(matches!(
            simulator.place_order(
                order("second", Outcome::No, OutcomeOrderSide::Buy, 0.5, 1.0),
                1_500
            ),
            Err(OutcomeError::InsufficientCollateral { .. })
        ));
        assert_eq!(simulator.open_orders(), before.as_slice());
        assert_close(simulator.reserved_collateral(), 0.6);
        assert_close(simulator.available_collateral(), 0.4);
    }

    #[test]
    fn close_all_sell_may_clear_aligned_residual_below_minimum_quantity() {
        let mut outcome_market = market(false);
        outcome_market.min_qty = 1.0;
        outcome_market.min_notional = 0.25;
        let mut simulator =
            SingleOutcomeSimulator::new(outcome_market, OutcomeFeeSchedule::zero(), 10.0).unwrap();
        simulator
            .place_order(
                order("buy-yes", Outcome::Yes, OutcomeOrderSide::Buy, 0.7, 2.0),
                1_500,
            )
            .unwrap();
        simulator
            .place_order(
                order("buy-no", Outcome::No, OutcomeOrderSide::Buy, 0.7, 1.5),
                1_500,
            )
            .unwrap();
        simulator
            .process_candle(&candle(Outcome::Yes, 0.71, 0.69, 0.7, 1.0))
            .unwrap();
        simulator
            .process_candle(&candle(Outcome::No, 0.71, 0.69, 0.7, 1.0))
            .unwrap();
        assert_close(
            simulator.ledger().yes_qty() - simulator.ledger().no_qty(),
            0.5,
        );

        let mut ordinary_dust = order(
            "ordinary-dust",
            Outcome::Yes,
            OutcomeOrderSide::Sell,
            0.6,
            0.5,
        );
        assert_eq!(
            simulator.place_order(ordinary_dust.clone(), 2_500),
            Err(OutcomeError::InvalidQuantity(0.5))
        );
        ordinary_dust.order_id = "close-all".to_string();
        ordinary_dust.close_all = true;
        simulator.place_order(ordinary_dust, 2_500).unwrap();

        let mut false_close = order(
            "false-close",
            Outcome::Yes,
            OutcomeOrderSide::Sell,
            0.6,
            0.4,
        );
        false_close.close_all = true;
        assert_eq!(
            simulator.place_order(false_close, 2_500),
            Err(OutcomeError::InvalidQuantity(0.4))
        );
    }

    #[test]
    fn close_all_sell_remains_subject_to_minimum_notional() {
        let mut outcome_market = market(false);
        outcome_market.min_qty = 0.1;
        outcome_market.min_notional = 0.9;
        let mut simulator =
            SingleOutcomeSimulator::new(outcome_market, OutcomeFeeSchedule::zero(), 10.0).unwrap();
        simulator
            .place_order(
                order("buy-yes", Outcome::Yes, OutcomeOrderSide::Buy, 0.7, 2.0),
                1_500,
            )
            .unwrap();
        simulator
            .place_order(
                order("buy-no", Outcome::No, OutcomeOrderSide::Buy, 0.95, 1.0),
                1_500,
            )
            .unwrap();
        simulator
            .process_candle(&candle(Outcome::Yes, 0.71, 0.69, 0.7, 1.0))
            .unwrap();
        simulator
            .process_candle(&candle(Outcome::No, 0.96, 0.94, 0.95, 1.0))
            .unwrap();
        let mut below_notional = order(
            "below-notional",
            Outcome::Yes,
            OutcomeOrderSide::Sell,
            0.5,
            1.0,
        );
        below_notional.close_all = true;

        assert_eq!(
            simulator.place_order(below_notional, 2_500),
            Err(OutcomeError::InvalidQuantity(1.0))
        );
    }

    #[test]
    fn maker_fee_is_reserved_and_applied_on_fill() {
        let fees = OutcomeFeeSchedule {
            maker_rate: 0.01,
            taker_rate: 0.02,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.0,
        };
        let mut simulator = SingleOutcomeSimulator::new(market(false), fees, 1.0).unwrap();
        simulator
            .place_order(
                order("buy", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                1_500,
            )
            .unwrap();
        assert_close(simulator.reserved_collateral(), 0.505);
        simulator
            .process_candle(&candle(Outcome::Yes, 0.51, 0.499, 0.5, 1.0))
            .unwrap();
        assert_close(simulator.ledger().collateral(), 0.495);
        assert_close(simulator.ledger().fees_paid(), 0.005);
    }

    #[test]
    fn maker_rebate_is_not_counted_as_resting_collateral() {
        let fees = OutcomeFeeSchedule {
            maker_rate: -0.01,
            taker_rate: 0.02,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.0,
        };
        let mut simulator = SingleOutcomeSimulator::new(market(false), fees, 0.5).unwrap();
        simulator
            .place_order(
                order("buy", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                1_500,
            )
            .unwrap();
        assert_close(simulator.reserved_collateral(), 0.5);
        assert_close(simulator.available_collateral(), 0.0);
        simulator
            .process_candle(&candle(Outcome::Yes, 0.51, 0.499, 0.5, 1.0))
            .unwrap();
        assert_close(simulator.ledger().collateral(), 0.005);
        assert_close(simulator.ledger().rebates_earned(), 0.005);
    }

    #[test]
    fn inventory_reduction_fee_does_not_reserve_opening_fee_and_settlement_fee_applies() {
        let fees = OutcomeFeeSchedule {
            maker_rate: 0.01,
            taker_rate: 0.02,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::InventoryReductionOnly,
            settlement_rate: 0.02,
        };
        let mut simulator = SingleOutcomeSimulator::new(market(false), fees, 1.0).unwrap();
        simulator
            .place_order(
                order("buy", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                1_500,
            )
            .unwrap();
        assert_close(simulator.reserved_collateral(), 0.5);
        simulator
            .process_candle(&candle(Outcome::Yes, 0.51, 0.499, 0.5, 1.0))
            .unwrap();
        assert_close(simulator.ledger().fees_paid(), 0.0);
        let settlement = simulator.settle(10_000, 1.0).unwrap();
        assert_close(settlement.fee, 0.02);
        assert_close(simulator.ledger().collateral(), 1.48);
        assert_close(simulator.ledger().fees_paid(), 0.02);
    }

    #[test]
    fn worst_case_settlement_equity_includes_settlement_fee() {
        let schedule = OutcomeFeeSchedule {
            maker_rate: 0.0,
            taker_rate: 0.0,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.02,
        };
        let mut simulator = SingleOutcomeSimulator::new(market(false), schedule, 2.0).unwrap();
        simulator
            .place_order(
                OutcomeLimitOrder {
                    order_id: "yes".to_string(),
                    outcome: Outcome::Yes,
                    side: OutcomeOrderSide::Buy,
                    price: 0.4,
                    qty: 1.0,
                    close_all: false,
                    post_only: true,
                    expires_at_ms: None,
                },
                1_000,
            )
            .unwrap();
        simulator
            .place_order(
                OutcomeLimitOrder {
                    order_id: "no".to_string(),
                    outcome: Outcome::No,
                    side: OutcomeOrderSide::Buy,
                    price: 0.5,
                    qty: 1.0,
                    close_all: false,
                    post_only: true,
                    expires_at_ms: None,
                },
                1_000,
            )
            .unwrap();
        simulator
            .process_candle(&OutcomeCandle {
                timestamp_ms: 2_000,
                outcome: Outcome::Yes,
                open: 0.5,
                high: 0.5,
                low: 0.39,
                close: 0.5,
                volume: 1.0,
            })
            .unwrap();
        simulator
            .process_candle(&OutcomeCandle {
                timestamp_ms: 2_000,
                outcome: Outcome::No,
                open: 0.5,
                high: 0.51,
                low: 0.49,
                close: 0.5,
                volume: 1.0,
            })
            .unwrap();

        assert_close(simulator.worst_case_settlement_equity().unwrap(), 2.08);
    }

    #[test]
    fn expiry_releases_reservation_and_settlement_cancels_all_orders() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 1.0).unwrap();
        let mut expiring = order("expiring", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0);
        expiring.expires_at_ms = Some(3_000);
        simulator.place_order(expiring, 1_500).unwrap();
        assert_close(simulator.available_collateral(), 0.5);
        assert_eq!(simulator.expire_orders(3_000).len(), 1);
        assert_close(simulator.available_collateral(), 1.0);

        simulator
            .place_order(
                order("until-close", Outcome::No, OutcomeOrderSide::Buy, 0.5, 1.0),
                3_500,
            )
            .unwrap();
        simulator.settle(10_000, 1.0).unwrap();
        assert!(simulator.open_orders().is_empty());
        assert!(simulator.ledger().is_settled());
    }

    #[test]
    fn order_entry_respects_acceptance_after_open_and_close_after_scheduled_event() {
        let mut delayed = market(false);
        delayed.order_entry_opens_ms = 2_000;
        delayed.scheduled_event_ms = 5_000;
        delayed.trading_closes_ms = 6_000;
        let mut simulator =
            SingleOutcomeSimulator::new(delayed, OutcomeFeeSchedule::zero(), 1.0).unwrap();

        assert_eq!(
            simulator
                .place_order(
                    order("too-early", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                    1_500,
                )
                .unwrap_err(),
            OutcomeError::MarketNotTrading(1_500)
        );
        simulator
            .place_order(
                order("accepted", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                2_000,
            )
            .unwrap();
    }

    #[test]
    fn rejects_gtd_order_expired_at_placement() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 1.0).unwrap();
        let mut expired = order("expired", Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0);
        expired.expires_at_ms = Some(1_500);

        assert!(matches!(
            simulator.place_order(expired, 1_500),
            Err(OutcomeError::InvalidMarket(message))
                if message.contains("later than its placement")
        ));
        assert!(simulator.open_orders().is_empty());
    }

    #[test]
    fn split_and_merge_respect_open_order_reservations() {
        let mut simulator =
            SingleOutcomeSimulator::new(market(false), OutcomeFeeSchedule::zero(), 2.0).unwrap();
        simulator
            .place_order(
                order(
                    "reserved-buy",
                    Outcome::Yes,
                    OutcomeOrderSide::Buy,
                    0.5,
                    2.0,
                ),
                1_500,
            )
            .unwrap();
        assert!(matches!(
            simulator.split(1.1, 0.5),
            Err(OutcomeError::InsufficientCollateral { .. })
        ));
        simulator.cancel_order("reserved-buy").unwrap();
        simulator.split(2.0, 0.4).unwrap();
        simulator
            .place_order(
                order(
                    "reserved-sell",
                    Outcome::No,
                    OutcomeOrderSide::Sell,
                    0.6,
                    1.5,
                ),
                1_500,
            )
            .unwrap();
        assert!(matches!(
            simulator.merge(1.0),
            Err(OutcomeError::InsufficientInventory {
                outcome: Outcome::No,
                ..
            })
        ));
        simulator.cancel_order("reserved-sell").unwrap();
        assert_close(simulator.merge(2.0).unwrap(), 0.0);
        assert_close(simulator.ledger().collateral(), 2.0);
    }

    #[test]
    fn split_and_merge_require_venue_capabilities() {
        let mut unsupported = market(false);
        unsupported.capabilities.supports_split = false;
        unsupported.capabilities.supports_merge = false;
        let mut simulator =
            SingleOutcomeSimulator::new(unsupported, OutcomeFeeSchedule::zero(), 2.0).unwrap();

        assert_eq!(
            simulator.split(1.0, 0.5),
            Err(OutcomeError::UnsupportedOrderFeature("split".to_string()))
        );
        assert_eq!(
            simulator.merge(1.0),
            Err(OutcomeError::UnsupportedOrderFeature("merge".to_string()))
        );
    }
}
