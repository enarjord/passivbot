use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod backtest;
pub mod simulator;
pub mod strategy;

const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum OutcomeError {
    InvalidMarket(String),
    InvalidPrice(f64),
    InvalidQuantity(f64),
    InvalidFee(f64),
    InvalidSettlementFraction(f64),
    InsufficientCollateral {
        required: f64,
        available: f64,
    },
    InsufficientInventory {
        outcome: Outcome,
        required: f64,
        available: f64,
    },
    DuplicateOrderId(String),
    UnknownOrderId(String),
    MarketNotTrading(u64),
    UnsupportedOrderFeature(String),
    AlreadySettled,
}

impl Display for OutcomeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMarket(message) => write!(f, "invalid outcome market: {message}"),
            Self::InvalidPrice(price) => write!(f, "invalid outcome price: {price}"),
            Self::InvalidQuantity(qty) => write!(f, "invalid outcome quantity: {qty}"),
            Self::InvalidFee(fee) => write!(f, "invalid outcome fee: {fee}"),
            Self::InvalidSettlementFraction(fraction) => {
                write!(f, "invalid settlement fraction: {fraction}")
            }
            Self::InsufficientCollateral {
                required,
                available,
            } => write!(
                f,
                "insufficient outcome collateral: required {required}, available {available}"
            ),
            Self::InsufficientInventory {
                outcome,
                required,
                available,
            } => write!(
                f,
                "insufficient {outcome} inventory: required {required}, available {available}"
            ),
            Self::DuplicateOrderId(order_id) => {
                write!(f, "duplicate outcome order ID: {order_id}")
            }
            Self::UnknownOrderId(order_id) => write!(f, "unknown outcome order ID: {order_id}"),
            Self::MarketNotTrading(timestamp_ms) => {
                write!(f, "outcome market is not trading at {timestamp_ms}")
            }
            Self::UnsupportedOrderFeature(feature) => {
                write!(f, "unsupported outcome order feature: {feature}")
            }
            Self::AlreadySettled => write!(f, "outcome market is already settled"),
        }
    }
}

impl Error for OutcomeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Yes,
    No,
}

impl Display for Outcome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yes => f.write_str("YES"),
            Self::No => f.write_str("NO"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeOrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiquidityRole {
    Maker,
    Taker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeFeeFormula {
    /// Fee equals native trade notional multiplied by the applicable rate.
    Notional,
    /// Fee equals qty * payout_unit * rate * probability * (1 - probability).
    ProbabilityVariance,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeFeeIncidence {
    /// Charge every fill according to its liquidity role.
    #[default]
    EveryFill,
    /// Charge only fills which reduce token inventory. In the current fully collateralized,
    /// inventory-backed model, buys add inventory and sells reduce it.
    InventoryReductionOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeFeeSchedule {
    pub maker_rate: f64,
    pub taker_rate: f64,
    pub formula: OutcomeFeeFormula,
    #[serde(default)]
    pub incidence: OutcomeFeeIncidence,
    /// Rate charged against collateral paid out at settlement. This is intentionally separate
    /// from the trade formula because settlement fees are payout-notional fees.
    #[serde(default)]
    pub settlement_rate: f64,
}

impl OutcomeFeeSchedule {
    pub const fn zero() -> Self {
        Self {
            maker_rate: 0.0,
            taker_rate: 0.0,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), OutcomeError> {
        if !self.maker_rate.is_finite()
            || !self.taker_rate.is_finite()
            || !self.settlement_rate.is_finite()
        {
            return Err(OutcomeError::InvalidMarket(
                "fee rates must be finite; negative rates represent rebates".to_string(),
            ));
        }
        Ok(())
    }

    pub fn calculate(&self, fill: &OutcomeFill, payout_unit: f64) -> Result<f64, OutcomeError> {
        self.validate()?;
        validate_payout_unit(payout_unit)?;
        fill.validate(payout_unit)?;
        if self.incidence == OutcomeFeeIncidence::InventoryReductionOnly
            && fill.side == OutcomeOrderSide::Buy
        {
            return Ok(0.0);
        }
        let rate = match fill.liquidity_role {
            LiquidityRole::Maker => self.maker_rate,
            LiquidityRole::Taker => self.taker_rate,
        };
        let fee = match self.formula {
            OutcomeFeeFormula::Notional => fill.qty * fill.price * rate,
            OutcomeFeeFormula::ProbabilityVariance => {
                let probability = fill.price / payout_unit;
                fill.qty * payout_unit * rate * probability * (1.0 - probability)
            }
        };
        if fee.is_finite() {
            Ok(fee)
        } else {
            Err(OutcomeError::InvalidFee(fee))
        }
    }

    pub fn calculate_settlement_fee(&self, collateral_payout: f64) -> Result<f64, OutcomeError> {
        self.validate()?;
        if !collateral_payout.is_finite() || collateral_payout < 0.0 {
            return Err(OutcomeError::InvalidFee(collateral_payout));
        }
        let fee = collateral_payout * self.settlement_rate;
        if fee.is_finite() {
            Ok(fee)
        } else {
            Err(OutcomeError::InvalidFee(fee))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeVenueCapabilities {
    pub complementary_books_merged: bool,
    pub supports_split: bool,
    pub supports_merge: bool,
    pub supports_redeem: bool,
    pub supports_post_only: bool,
    pub supports_gtd: bool,
    pub sell_requires_inventory: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutcomePriceGrid {
    FixedStep {
        step: f64,
    },
    SignificantFigures {
        max_significant_figures: u32,
        max_decimal_places: u32,
    },
}

impl OutcomePriceGrid {
    pub fn validate(&self) -> Result<(), OutcomeError> {
        match self {
            Self::FixedStep { step } if step.is_finite() && *step > 0.0 => Ok(()),
            Self::SignificantFigures {
                max_significant_figures,
                max_decimal_places,
            } if *max_significant_figures > 0
                && *max_significant_figures <= 15
                && *max_decimal_places <= 15 =>
            {
                Ok(())
            }
            _ => Err(OutcomeError::InvalidMarket(
                "invalid outcome price grid".to_string(),
            )),
        }
    }

    pub fn increment_at(&self, price: f64) -> Result<f64, OutcomeError> {
        self.validate()?;
        if !price.is_finite() || price < 0.0 {
            return Err(OutcomeError::InvalidPrice(price));
        }
        match self {
            Self::FixedStep { step } => Ok(*step),
            Self::SignificantFigures {
                max_significant_figures,
                max_decimal_places,
            } => {
                let decimal_increment = 10_f64.powi(-(*max_decimal_places as i32));
                if price == 0.0 {
                    return Ok(decimal_increment);
                }
                let magnitude = price.abs().log10().floor() as i32;
                let significant_increment =
                    10_f64.powi(magnitude - *max_significant_figures as i32 + 1);
                Ok(decimal_increment.max(significant_increment))
            }
        }
    }

    pub fn round_down(&self, price: f64) -> Result<f64, OutcomeError> {
        let increment = self.increment_at(price)?;
        Ok(((price + EPSILON) / increment).floor() * increment)
    }

    pub fn round_up(&self, price: f64) -> Result<f64, OutcomeError> {
        let increment = self.increment_at(price)?;
        let rounded = ((price - EPSILON) / increment).ceil() * increment;
        // Rounding upward can cross a power-of-ten boundary where the significant-figure
        // increment changes. Apply the new increment once more in that case.
        let next_increment = self.increment_at(rounded)?;
        Ok(((rounded - EPSILON) / next_increment).ceil() * next_increment)
    }

    pub fn is_valid(&self, price: f64) -> bool {
        self.round_down(price)
            .is_ok_and(|rounded| (rounded - price).abs() <= EPSILON.max(price.abs() * 1e-12))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinaryOutcomeMarketSpec {
    pub venue: String,
    pub market_id: String,
    pub yes_asset_id: String,
    pub no_asset_id: String,
    pub payout_unit: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub price_grid: OutcomePriceGrid,
    pub qty_step: f64,
    pub min_qty: f64,
    pub min_notional: f64,
    pub trading_opens_ms: u64,
    pub trading_closes_ms: u64,
    pub scheduled_resolution_ms: u64,
    pub capabilities: OutcomeVenueCapabilities,
}

impl BinaryOutcomeMarketSpec {
    pub fn validate(&self) -> Result<(), OutcomeError> {
        if self.venue.trim().is_empty()
            || self.market_id.trim().is_empty()
            || self.yes_asset_id.trim().is_empty()
            || self.no_asset_id.trim().is_empty()
        {
            return Err(OutcomeError::InvalidMarket(
                "venue, market, and YES/NO asset IDs are required".to_string(),
            ));
        }
        if self.yes_asset_id == self.no_asset_id {
            return Err(OutcomeError::InvalidMarket(
                "YES and NO asset IDs must differ".to_string(),
            ));
        }
        validate_payout_unit(self.payout_unit)?;
        if !self.min_price.is_finite()
            || !self.max_price.is_finite()
            || self.min_price < 0.0
            || self.max_price > self.payout_unit
            || self.min_price >= self.max_price
        {
            return Err(OutcomeError::InvalidMarket(
                "price bounds must satisfy 0 <= min_price < max_price <= payout_unit".to_string(),
            ));
        }
        self.price_grid.validate()?;
        if !self.qty_step.is_finite()
            || self.qty_step <= 0.0
            || !self.min_qty.is_finite()
            || self.min_qty <= 0.0
            || !self.min_notional.is_finite()
            || self.min_notional < 0.0
        {
            return Err(OutcomeError::InvalidMarket(
                "qty_step and min_qty must be positive; min_notional must be non-negative"
                    .to_string(),
            ));
        }
        if self.trading_opens_ms >= self.trading_closes_ms
            || self.trading_closes_ms > self.scheduled_resolution_ms
        {
            return Err(OutcomeError::InvalidMarket(
                "timestamps must satisfy trading_open < trading_close <= scheduled_resolution"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn round_price_down(&self, price: f64) -> Result<f64, OutcomeError> {
        self.price_grid.round_down(price)
    }

    pub fn round_price_up(&self, price: f64) -> Result<f64, OutcomeError> {
        self.price_grid.round_up(price)
    }

    pub fn validate_order(&self, order: &OutcomeLimitOrder) -> Result<(), OutcomeError> {
        self.validate()?;
        order.validate(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeLimitOrder {
    pub order_id: String,
    pub outcome: Outcome,
    pub side: OutcomeOrderSide,
    pub price: f64,
    pub qty: f64,
    pub post_only: bool,
    pub expires_at_ms: Option<u64>,
}

impl OutcomeLimitOrder {
    pub fn validate(&self, market: &BinaryOutcomeMarketSpec) -> Result<(), OutcomeError> {
        if self.order_id.trim().is_empty() {
            return Err(OutcomeError::InvalidMarket(
                "outcome order_id is required".to_string(),
            ));
        }
        if !self.price.is_finite()
            || self.price < market.min_price
            || self.price > market.max_price
            || !market.price_grid.is_valid(self.price)
        {
            return Err(OutcomeError::InvalidPrice(self.price));
        }
        if !self.qty.is_finite()
            || self.qty < market.min_qty
            || !is_step_aligned(self.qty, market.qty_step)
        {
            return Err(OutcomeError::InvalidQuantity(self.qty));
        }
        if self.qty * self.price + EPSILON < market.min_notional {
            return Err(OutcomeError::InvalidQuantity(self.qty));
        }
        if let Some(expires_at_ms) = self.expires_at_ms {
            if expires_at_ms <= market.trading_opens_ms || expires_at_ms > market.trading_closes_ms
            {
                return Err(OutcomeError::InvalidMarket(
                    "order expiry must be after trading open and no later than trading close"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_yes_price(&self, payout_unit: f64) -> Result<f64, OutcomeError> {
        canonical_yes_price(self.outcome, self.price, payout_unit)
    }

    pub fn canonical_yes_exposure_delta(&self) -> f64 {
        canonical_yes_exposure_delta(self.outcome, self.side, self.qty)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeFill {
    pub timestamp_ms: u64,
    pub outcome: Outcome,
    pub side: OutcomeOrderSide,
    pub price: f64,
    pub qty: f64,
    pub liquidity_role: LiquidityRole,
}

impl OutcomeFill {
    pub fn validate(&self, payout_unit: f64) -> Result<(), OutcomeError> {
        validate_price(self.price, payout_unit)?;
        validate_qty(self.qty)
    }

    pub fn canonical_yes_price(&self, payout_unit: f64) -> Result<f64, OutcomeError> {
        canonical_yes_price(self.outcome, self.price, payout_unit)
    }

    pub fn canonical_yes_exposure_delta(&self) -> f64 {
        canonical_yes_exposure_delta(self.outcome, self.side, self.qty)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeFillAccounting {
    pub gross_cash_delta: f64,
    pub fee: f64,
    pub inventory_cost_removed: f64,
    pub realized_trading_pnl_delta: f64,
    pub canonical_yes_price: f64,
    pub canonical_yes_exposure_delta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeSettlement {
    pub yes_fraction: f64,
    pub yes_payout: f64,
    pub no_payout: f64,
    pub collateral_payout: f64,
    pub fee: f64,
    pub inventory_cost_removed: f64,
    pub realized_settlement_pnl: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct OutcomeInventory {
    qty: f64,
    cost: f64,
}

impl OutcomeInventory {
    fn add(&mut self, qty: f64, cost: f64) {
        self.qty += qty;
        self.cost += cost;
    }

    fn remove(&mut self, outcome: Outcome, qty: f64) -> Result<f64, OutcomeError> {
        validate_qty(qty)?;
        if qty > self.qty + EPSILON {
            return Err(OutcomeError::InsufficientInventory {
                outcome,
                required: qty,
                available: self.qty,
            });
        }
        let removed_qty = qty.min(self.qty);
        let removed_cost = if removed_qty >= self.qty - EPSILON {
            self.cost
        } else {
            self.cost * (removed_qty / self.qty)
        };
        self.qty = (self.qty - removed_qty).max(0.0);
        self.cost = (self.cost - removed_cost).max(0.0);
        if self.qty <= EPSILON {
            self.qty = 0.0;
            self.cost = 0.0;
        }
        Ok(removed_cost)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeLedger {
    starting_collateral: f64,
    collateral: f64,
    payout_unit: f64,
    yes: OutcomeInventory,
    no: OutcomeInventory,
    realized_trading_pnl: f64,
    realized_merge_pnl: f64,
    realized_settlement_pnl: f64,
    fees_paid: f64,
    rebates_earned: f64,
    settled: bool,
}

impl OutcomeLedger {
    pub fn new(starting_collateral: f64, payout_unit: f64) -> Result<Self, OutcomeError> {
        if !starting_collateral.is_finite() || starting_collateral < 0.0 {
            return Err(OutcomeError::InvalidMarket(
                "starting collateral must be finite and non-negative".to_string(),
            ));
        }
        validate_payout_unit(payout_unit)?;
        Ok(Self {
            starting_collateral,
            collateral: starting_collateral,
            payout_unit,
            yes: OutcomeInventory::default(),
            no: OutcomeInventory::default(),
            realized_trading_pnl: 0.0,
            realized_merge_pnl: 0.0,
            realized_settlement_pnl: 0.0,
            fees_paid: 0.0,
            rebates_earned: 0.0,
            settled: false,
        })
    }

    pub fn apply_fill(
        &mut self,
        fill: &OutcomeFill,
        fee_schedule: &OutcomeFeeSchedule,
    ) -> Result<OutcomeFillAccounting, OutcomeError> {
        let fee = fee_schedule.calculate(fill, self.payout_unit)?;
        self.apply_fill_with_fee(fill, fee)
    }

    pub fn apply_fill_with_fee(
        &mut self,
        fill: &OutcomeFill,
        fee: f64,
    ) -> Result<OutcomeFillAccounting, OutcomeError> {
        self.ensure_unsettled()?;
        fill.validate(self.payout_unit)?;
        if !fee.is_finite() {
            return Err(OutcomeError::InvalidFee(fee));
        }

        let gross = fill.qty * fill.price;
        let (gross_cash_delta, inventory_cost_removed, realized_trading_pnl_delta) = match fill.side
        {
            OutcomeOrderSide::Buy => {
                // A rebate is credited after execution and must not be used to
                // fund the gross purchase itself.
                let required = gross + fee.max(0.0);
                self.ensure_collateral(required)?;
                self.collateral -= gross + fee;
                self.inventory_mut(fill.outcome).add(fill.qty, gross);
                (-gross, 0.0, 0.0)
            }
            OutcomeOrderSide::Sell => {
                let available = self.inventory(fill.outcome).qty;
                if fill.qty > available + EPSILON {
                    return Err(OutcomeError::InsufficientInventory {
                        outcome: fill.outcome,
                        required: fill.qty,
                        available,
                    });
                }
                if self.collateral + gross + EPSILON < fee {
                    return Err(OutcomeError::InsufficientCollateral {
                        required: fee,
                        available: self.collateral + gross,
                    });
                }
                let removed_cost = self
                    .inventory_mut(fill.outcome)
                    .remove(fill.outcome, fill.qty)?;
                self.collateral += gross - fee;
                let realized = gross - removed_cost;
                self.realized_trading_pnl += realized;
                (gross, removed_cost, realized)
            }
        };
        if fee >= 0.0 {
            self.fees_paid += fee;
        } else {
            self.rebates_earned += -fee;
        }

        Ok(OutcomeFillAccounting {
            gross_cash_delta,
            fee,
            inventory_cost_removed,
            realized_trading_pnl_delta,
            canonical_yes_price: fill.canonical_yes_price(self.payout_unit)?,
            canonical_yes_exposure_delta: fill.canonical_yes_exposure_delta(),
        })
    }

    pub fn split(&mut self, qty: f64, yes_reference_price: f64) -> Result<(), OutcomeError> {
        self.ensure_unsettled()?;
        validate_qty(qty)?;
        validate_price(yes_reference_price, self.payout_unit)?;
        let required = qty * self.payout_unit;
        self.ensure_collateral(required)?;
        self.collateral -= required;
        self.yes.add(qty, qty * yes_reference_price);
        self.no
            .add(qty, qty * (self.payout_unit - yes_reference_price));
        Ok(())
    }

    pub fn merge(&mut self, qty: f64) -> Result<f64, OutcomeError> {
        self.ensure_unsettled()?;
        validate_qty(qty)?;
        if qty > self.yes.qty + EPSILON {
            return Err(OutcomeError::InsufficientInventory {
                outcome: Outcome::Yes,
                required: qty,
                available: self.yes.qty,
            });
        }
        if qty > self.no.qty + EPSILON {
            return Err(OutcomeError::InsufficientInventory {
                outcome: Outcome::No,
                required: qty,
                available: self.no.qty,
            });
        }
        let yes_cost = self.yes.remove(Outcome::Yes, qty)?;
        let no_cost = self.no.remove(Outcome::No, qty)?;
        let payout = qty * self.payout_unit;
        let pnl = payout - yes_cost - no_cost;
        self.collateral += payout;
        self.realized_merge_pnl += pnl;
        Ok(pnl)
    }

    pub fn settle(&mut self, yes_fraction: f64) -> Result<OutcomeSettlement, OutcomeError> {
        self.settle_with_fee(yes_fraction, 0.0)
    }

    pub fn settle_with_fee_schedule(
        &mut self,
        yes_fraction: f64,
        fee_schedule: &OutcomeFeeSchedule,
    ) -> Result<OutcomeSettlement, OutcomeError> {
        self.ensure_unsettled()?;
        validate_settlement_fraction(yes_fraction)?;
        let collateral_payout = self.settlement_collateral_payout(yes_fraction);
        let fee = fee_schedule.calculate_settlement_fee(collateral_payout)?;
        self.settle_with_fee(yes_fraction, fee)
    }

    pub fn settle_with_fee(
        &mut self,
        yes_fraction: f64,
        fee: f64,
    ) -> Result<OutcomeSettlement, OutcomeError> {
        self.ensure_unsettled()?;
        validate_settlement_fraction(yes_fraction)?;
        if !fee.is_finite() {
            return Err(OutcomeError::InvalidFee(fee));
        }
        let yes_payout = self.payout_unit * yes_fraction;
        let no_payout = self.payout_unit * (1.0 - yes_fraction);
        let collateral_payout = self.settlement_collateral_payout(yes_fraction);
        let inventory_cost_removed = self.yes.cost + self.no.cost;
        let realized_settlement_pnl = collateral_payout - inventory_cost_removed;
        if self.collateral + collateral_payout + EPSILON < fee {
            return Err(OutcomeError::InsufficientCollateral {
                required: fee,
                available: self.collateral + collateral_payout,
            });
        }

        self.collateral += collateral_payout - fee;
        self.yes = OutcomeInventory::default();
        self.no = OutcomeInventory::default();
        self.realized_settlement_pnl += realized_settlement_pnl;
        if fee >= 0.0 {
            self.fees_paid += fee;
        } else {
            self.rebates_earned += -fee;
        }
        self.settled = true;

        Ok(OutcomeSettlement {
            yes_fraction,
            yes_payout,
            no_payout,
            collateral_payout,
            fee,
            inventory_cost_removed,
            realized_settlement_pnl,
        })
    }

    fn settlement_collateral_payout(&self, yes_fraction: f64) -> f64 {
        let yes_payout = self.payout_unit * yes_fraction;
        let no_payout = self.payout_unit * (1.0 - yes_fraction);
        self.yes.qty * yes_payout + self.no.qty * no_payout
    }

    pub fn collateral(&self) -> f64 {
        self.collateral
    }

    pub fn yes_qty(&self) -> f64 {
        self.yes.qty
    }

    pub fn no_qty(&self) -> f64 {
        self.no.qty
    }

    pub fn yes_cost(&self) -> f64 {
        self.yes.cost
    }

    pub fn no_cost(&self) -> f64 {
        self.no.cost
    }

    pub fn paired_qty(&self) -> f64 {
        self.yes.qty.min(self.no.qty)
    }

    pub fn net_yes_exposure(&self) -> f64 {
        self.yes.qty - self.no.qty
    }

    pub fn fees_paid(&self) -> f64 {
        self.fees_paid
    }

    pub fn rebates_earned(&self) -> f64 {
        self.rebates_earned
    }

    pub fn realized_trading_pnl(&self) -> f64 {
        self.realized_trading_pnl
    }

    pub fn realized_merge_pnl(&self) -> f64 {
        self.realized_merge_pnl
    }

    pub fn realized_settlement_pnl(&self) -> f64 {
        self.realized_settlement_pnl
    }

    pub fn net_realized_pnl(&self) -> f64 {
        self.realized_trading_pnl + self.realized_merge_pnl + self.realized_settlement_pnl
            - self.fees_paid
            + self.rebates_earned
    }

    pub fn equity_at_yes_price(&self, yes_price: f64) -> Result<f64, OutcomeError> {
        validate_price(yes_price, self.payout_unit)?;
        Ok(self.collateral
            + self.yes.qty * yes_price
            + self.no.qty * (self.payout_unit - yes_price))
    }

    pub fn worst_case_settlement_equity(&self) -> f64 {
        self.collateral + self.paired_qty() * self.payout_unit
    }

    pub fn total_return(&self) -> f64 {
        self.collateral - self.starting_collateral
    }

    pub fn is_settled(&self) -> bool {
        self.settled
    }

    fn ensure_unsettled(&self) -> Result<(), OutcomeError> {
        if self.settled {
            Err(OutcomeError::AlreadySettled)
        } else {
            Ok(())
        }
    }

    fn ensure_collateral(&self, required: f64) -> Result<(), OutcomeError> {
        if required <= self.collateral + EPSILON {
            Ok(())
        } else {
            Err(OutcomeError::InsufficientCollateral {
                required,
                available: self.collateral,
            })
        }
    }

    fn inventory(&self, outcome: Outcome) -> &OutcomeInventory {
        match outcome {
            Outcome::Yes => &self.yes,
            Outcome::No => &self.no,
        }
    }

    fn inventory_mut(&mut self, outcome: Outcome) -> &mut OutcomeInventory {
        match outcome {
            Outcome::Yes => &mut self.yes,
            Outcome::No => &mut self.no,
        }
    }
}

pub fn canonical_yes_price(
    outcome: Outcome,
    native_price: f64,
    payout_unit: f64,
) -> Result<f64, OutcomeError> {
    validate_price(native_price, payout_unit)?;
    Ok(match outcome {
        Outcome::Yes => native_price,
        Outcome::No => payout_unit - native_price,
    })
}

pub fn canonical_yes_exposure_delta(outcome: Outcome, side: OutcomeOrderSide, qty: f64) -> f64 {
    match (outcome, side) {
        (Outcome::Yes, OutcomeOrderSide::Buy) | (Outcome::No, OutcomeOrderSide::Sell) => qty,
        (Outcome::Yes, OutcomeOrderSide::Sell) | (Outcome::No, OutcomeOrderSide::Buy) => -qty,
    }
}

fn validate_payout_unit(payout_unit: f64) -> Result<(), OutcomeError> {
    if payout_unit.is_finite() && payout_unit > 0.0 {
        Ok(())
    } else {
        Err(OutcomeError::InvalidMarket(
            "payout_unit must be finite and positive".to_string(),
        ))
    }
}

fn validate_price(price: f64, payout_unit: f64) -> Result<(), OutcomeError> {
    validate_payout_unit(payout_unit)?;
    if price.is_finite() && price >= 0.0 && price <= payout_unit {
        Ok(())
    } else {
        Err(OutcomeError::InvalidPrice(price))
    }
}

fn validate_qty(qty: f64) -> Result<(), OutcomeError> {
    if qty.is_finite() && qty > 0.0 {
        Ok(())
    } else {
        Err(OutcomeError::InvalidQuantity(qty))
    }
}

fn validate_settlement_fraction(fraction: f64) -> Result<(), OutcomeError> {
    if fraction.is_finite() && (0.0..=1.0).contains(&fraction) {
        Ok(())
    } else {
        Err(OutcomeError::InvalidSettlementFraction(fraction))
    }
}

fn is_step_aligned(value: f64, step: f64) -> bool {
    let units = value / step;
    (units - units.round()).abs() <= EPSILON * units.abs().max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual {actual} != expected {expected}"
        );
    }

    fn fill(outcome: Outcome, side: OutcomeOrderSide, price: f64, qty: f64) -> OutcomeFill {
        OutcomeFill {
            timestamp_ms: 1,
            outcome,
            side,
            price,
            qty,
            liquidity_role: LiquidityRole::Maker,
        }
    }

    #[test]
    fn complementary_native_actions_share_canonical_yes_intent() {
        let buy_yes = fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.335, 2.0);
        let sell_no = fill(Outcome::No, OutcomeOrderSide::Sell, 0.665, 2.0);
        assert_close(buy_yes.canonical_yes_price(1.0).unwrap(), 0.335);
        assert_close(sell_no.canonical_yes_price(1.0).unwrap(), 0.335);
        assert_close(buy_yes.canonical_yes_exposure_delta(), 2.0);
        assert_close(sell_no.canonical_yes_exposure_delta(), 2.0);

        let sell_yes = fill(Outcome::Yes, OutcomeOrderSide::Sell, 0.335, 2.0);
        let buy_no = fill(Outcome::No, OutcomeOrderSide::Buy, 0.665, 2.0);
        assert_close(sell_yes.canonical_yes_exposure_delta(), -2.0);
        assert_close(buy_no.canonical_yes_exposure_delta(), -2.0);
    }

    #[test]
    fn buying_both_sides_and_settling_locks_same_profit_for_either_outcome() {
        for yes_fraction in [0.0, 1.0] {
            let mut ledger = OutcomeLedger::new(1.0, 1.0).unwrap();
            ledger
                .apply_fill(
                    &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.256, 1.0),
                    &OutcomeFeeSchedule::zero(),
                )
                .unwrap();
            ledger
                .apply_fill(
                    &fill(Outcome::No, OutcomeOrderSide::Buy, 0.252, 1.0),
                    &OutcomeFeeSchedule::zero(),
                )
                .unwrap();
            assert_close(ledger.paired_qty(), 1.0);
            assert_close(ledger.net_yes_exposure(), 0.0);
            assert_close(ledger.worst_case_settlement_equity(), 1.492);

            let settlement = ledger.settle(yes_fraction).unwrap();
            assert_close(settlement.realized_settlement_pnl, 0.492);
            assert_close(ledger.collateral(), 1.492);
            assert_close(ledger.total_return(), 0.492);
            assert_close(ledger.net_realized_pnl(), 0.492);
        }
    }

    #[test]
    fn buy_yes_then_sell_yes_matches_buy_yes_then_buy_complement() {
        let mut round_trip = OutcomeLedger::new(1.0, 1.0).unwrap();
        round_trip
            .apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.256, 1.0),
                &OutcomeFeeSchedule::zero(),
            )
            .unwrap();
        round_trip
            .apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Sell, 0.748, 1.0),
                &OutcomeFeeSchedule::zero(),
            )
            .unwrap();
        assert_close(round_trip.collateral(), 1.492);
        assert_close(round_trip.net_realized_pnl(), 0.492);

        let mut complement = OutcomeLedger::new(1.0, 1.0).unwrap();
        complement
            .apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.256, 1.0),
                &OutcomeFeeSchedule::zero(),
            )
            .unwrap();
        complement
            .apply_fill(
                &fill(Outcome::No, OutcomeOrderSide::Buy, 0.252, 1.0),
                &OutcomeFeeSchedule::zero(),
            )
            .unwrap();
        complement.settle(1.0).unwrap();
        assert_close(complement.collateral(), round_trip.collateral());
        assert_close(complement.net_realized_pnl(), round_trip.net_realized_pnl());
    }

    #[test]
    fn split_and_merge_conserve_equity() {
        let mut ledger = OutcomeLedger::new(10.0, 1.0).unwrap();
        ledger.split(4.0, 0.35).unwrap();
        assert_close(ledger.collateral(), 6.0);
        assert_close(ledger.yes_cost(), 1.4);
        assert_close(ledger.no_cost(), 2.6);
        assert_close(ledger.equity_at_yes_price(0.35).unwrap(), 10.0);
        assert_close(ledger.merge(4.0).unwrap(), 0.0);
        assert_close(ledger.collateral(), 10.0);
        assert_close(ledger.net_realized_pnl(), 0.0);
    }

    #[test]
    fn probability_variance_fee_is_symmetric_between_complements() {
        let schedule = OutcomeFeeSchedule {
            maker_rate: 0.0,
            taker_rate: 0.07,
            formula: OutcomeFeeFormula::ProbabilityVariance,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.0,
        };
        let mut yes = fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.3, 10.0);
        yes.liquidity_role = LiquidityRole::Taker;
        let mut no = fill(Outcome::No, OutcomeOrderSide::Buy, 0.7, 10.0);
        no.liquidity_role = LiquidityRole::Taker;
        assert_close(
            schedule.calculate(&yes, 1.0).unwrap(),
            schedule.calculate(&no, 1.0).unwrap(),
        );
        assert_close(schedule.calculate(&yes, 1.0).unwrap(), 0.147);
    }

    #[test]
    fn fees_reduce_cash_and_net_realized_pnl_without_changing_cost_basis() {
        let mut ledger = OutcomeLedger::new(10.0, 1.0).unwrap();
        ledger
            .apply_fill_with_fee(&fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.4, 2.0), 0.01)
            .unwrap();
        ledger
            .apply_fill_with_fee(&fill(Outcome::Yes, OutcomeOrderSide::Sell, 0.5, 2.0), 0.02)
            .unwrap();
        assert_close(ledger.collateral(), 10.17);
        assert_close(ledger.fees_paid(), 0.03);
        assert_close(ledger.net_realized_pnl(), 0.17);
    }

    #[test]
    fn signed_maker_fee_credits_rebate_separately() {
        let schedule = OutcomeFeeSchedule {
            maker_rate: -0.01,
            taker_rate: 0.02,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.0,
        };
        let mut ledger = OutcomeLedger::new(1.0, 1.0).unwrap();
        let accounting = ledger
            .apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                &schedule,
            )
            .unwrap();
        assert_close(accounting.fee, -0.005);
        assert_close(ledger.collateral(), 0.505);
        assert_close(ledger.fees_paid(), 0.0);
        assert_close(ledger.rebates_earned(), 0.005);
        assert_close(ledger.net_realized_pnl(), 0.005);
    }

    #[test]
    fn inventory_reduction_incidence_charges_sells_but_not_opening_buys() {
        let schedule = OutcomeFeeSchedule {
            maker_rate: 0.01,
            taker_rate: 0.02,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::InventoryReductionOnly,
            settlement_rate: 0.0,
        };
        let buy = fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.4, 2.0);
        let sell = fill(Outcome::Yes, OutcomeOrderSide::Sell, 0.5, 2.0);
        assert_close(schedule.calculate(&buy, 1.0).unwrap(), 0.0);
        assert_close(schedule.calculate(&sell, 1.0).unwrap(), 0.01);

        let mut ledger = OutcomeLedger::new(10.0, 1.0).unwrap();
        let buy_accounting = ledger.apply_fill(&buy, &schedule).unwrap();
        let sell_accounting = ledger.apply_fill(&sell, &schedule).unwrap();
        assert_close(buy_accounting.fee, 0.0);
        assert_close(sell_accounting.fee, 0.01);
        assert_close(ledger.fees_paid(), 0.01);
        assert_close(ledger.net_realized_pnl(), 0.19);
    }

    #[test]
    fn settlement_fee_is_deducted_reported_and_included_in_net_pnl() {
        let schedule = OutcomeFeeSchedule {
            maker_rate: 0.0,
            taker_rate: 0.0,
            formula: OutcomeFeeFormula::Notional,
            incidence: OutcomeFeeIncidence::EveryFill,
            settlement_rate: 0.02,
        };
        let mut ledger = OutcomeLedger::new(2.0, 1.0).unwrap();
        ledger
            .apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.4, 2.0),
                &schedule,
            )
            .unwrap();
        let settlement = ledger.settle_with_fee_schedule(1.0, &schedule).unwrap();
        assert_close(settlement.collateral_payout, 2.0);
        assert_close(settlement.fee, 0.04);
        assert_close(settlement.realized_settlement_pnl, 1.2);
        assert_close(ledger.collateral(), 3.16);
        assert_close(ledger.fees_paid(), 0.04);
        assert_close(ledger.net_realized_pnl(), 1.16);
    }

    #[test]
    fn missing_fee_incidence_and_settlement_rate_keep_legacy_payload_behavior() {
        let schedule: OutcomeFeeSchedule =
            serde_json::from_str(r#"{"maker_rate":0.01,"taker_rate":0.02,"formula":"notional"}"#)
                .unwrap();
        assert_eq!(schedule.incidence, OutcomeFeeIncidence::EveryFill);
        assert_close(schedule.settlement_rate, 0.0);
        assert_close(
            schedule
                .calculate(&fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0), 1.0)
                .unwrap(),
            0.005,
        );
    }

    #[test]
    fn rebate_cannot_fund_gross_purchase() {
        let mut ledger = OutcomeLedger::new(0.495, 1.0).unwrap();
        let initial = ledger.clone();
        assert!(matches!(
            ledger
                .apply_fill_with_fee(&fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0), -0.005,),
            Err(OutcomeError::InsufficientCollateral { .. })
        ));
        assert_eq!(ledger, initial);
    }

    #[test]
    fn insufficient_inventory_and_collateral_fail_without_mutating_ledger() {
        let mut ledger = OutcomeLedger::new(0.5, 1.0).unwrap();
        let initial = ledger.clone();
        assert!(matches!(
            ledger.apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.6, 1.0),
                &OutcomeFeeSchedule::zero()
            ),
            Err(OutcomeError::InsufficientCollateral { .. })
        ));
        assert_eq!(ledger, initial);
        assert!(matches!(
            ledger.apply_fill(
                &fill(Outcome::No, OutcomeOrderSide::Sell, 0.4, 1.0),
                &OutcomeFeeSchedule::zero()
            ),
            Err(OutcomeError::InsufficientInventory { .. })
        ));
        assert_eq!(ledger, initial);
    }

    #[test]
    fn excessive_authoritative_fee_fails_without_mutating_inventory() {
        let mut ledger = OutcomeLedger::new(1.0, 1.0).unwrap();
        ledger
            .apply_fill(
                &fill(Outcome::Yes, OutcomeOrderSide::Buy, 1.0, 1.0),
                &OutcomeFeeSchedule::zero(),
            )
            .unwrap();
        let before_rejected_sell = ledger.clone();
        assert!(matches!(
            ledger.apply_fill_with_fee(&fill(Outcome::Yes, OutcomeOrderSide::Sell, 0.0, 1.0), 0.1),
            Err(OutcomeError::InsufficientCollateral { .. })
        ));
        assert_eq!(ledger, before_rejected_sell);
    }

    #[test]
    fn settled_ledger_rejects_further_mutation() {
        let mut ledger = OutcomeLedger::new(1.0, 1.0).unwrap();
        ledger.settle(0.5).unwrap();
        assert_eq!(
            ledger.split(1.0, 0.5).unwrap_err(),
            OutcomeError::AlreadySettled
        );
        assert_eq!(
            ledger
                .apply_fill(
                    &fill(Outcome::Yes, OutcomeOrderSide::Buy, 0.5, 1.0),
                    &OutcomeFeeSchedule::zero()
                )
                .unwrap_err(),
            OutcomeError::AlreadySettled
        );
    }

    #[test]
    fn significant_figure_grid_matches_hyperliquid_style_probability_prices() {
        let grid = OutcomePriceGrid::SignificantFigures {
            max_significant_figures: 5,
            max_decimal_places: 8,
        };
        assert!(grid.is_valid(0.94379));
        assert!(grid.is_valid(0.056251));
        assert!(!grid.is_valid(0.943791));
        assert!(!grid.is_valid(0.0562511));
        assert_close(grid.round_down(0.943799).unwrap(), 0.94379);
        assert_close(grid.round_up(0.0562511).unwrap(), 0.056252);
    }
}
