use super::{BinaryOutcomeMarketSpec, Outcome, OutcomeError, OutcomeOrderSide};
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeEmaAnchorExecutionMode {
    AccumulatePairs,
    InventoryAware,
    YesOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorParams {
    pub ema_span_fast_seconds: f64,
    pub ema_span_slow_seconds: f64,
    /// Number of dense one-second observations required before quote generation.
    #[serde(default)]
    pub ema_warmup_seconds: u64,
    /// Absolute probability-point offset, not a multiplicative price percentage.
    pub quote_offset: f64,
    /// Maximum absolute probability-point shift at the residual inventory limit.
    pub inventory_skew: f64,
    pub clip_qty: f64,
    pub max_total_inventory_qty: f64,
    pub max_abs_residual_qty: f64,
    pub min_locked_pair_edge: f64,
    pub estimated_fee_per_share: f64,
    /// Before the final cutoff, quote only passive sales of the excess outcome token.
    /// This reduces both residual exposure and gross inventory instead of adding the
    /// missing complement late in the lifecycle. Zero disables the phase.
    #[serde(default)]
    pub risk_reduction_only_ms_before_close: u64,
    pub entry_cutoff_ms_before_close: u64,
    pub execution_mode: OutcomeEmaAnchorExecutionMode,
}

impl OutcomeEmaAnchorParams {
    pub fn validate(&self) -> Result<(), OutcomeError> {
        for (name, value) in [
            ("ema_span_fast_seconds", self.ema_span_fast_seconds),
            ("ema_span_slow_seconds", self.ema_span_slow_seconds),
            ("clip_qty", self.clip_qty),
            ("max_total_inventory_qty", self.max_total_inventory_qty),
            ("max_abs_residual_qty", self.max_abs_residual_qty),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(OutcomeError::InvalidMarket(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        for (name, value) in [
            ("quote_offset", self.quote_offset),
            ("inventory_skew", self.inventory_skew),
            ("min_locked_pair_edge", self.min_locked_pair_edge),
            ("estimated_fee_per_share", self.estimated_fee_per_share),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OutcomeError::InvalidMarket(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if self.ema_span_fast_seconds > self.ema_span_slow_seconds {
            return Err(OutcomeError::InvalidMarket(
                "ema_span_fast_seconds must not exceed ema_span_slow_seconds".to_string(),
            ));
        }
        if self.risk_reduction_only_ms_before_close > 0
            && self.risk_reduction_only_ms_before_close < self.entry_cutoff_ms_before_close
        {
            return Err(OutcomeError::InvalidMarket(
                "risk_reduction_only_ms_before_close must be zero or no less than ".to_string()
                    + "entry_cutoff_ms_before_close",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeInventorySnapshot {
    pub yes_qty: f64,
    pub no_qty: f64,
    pub yes_average_cost: f64,
    pub no_average_cost: f64,
    pub free_collateral: f64,
}

impl OutcomeInventorySnapshot {
    pub fn validate(&self, payout_unit: f64) -> Result<(), OutcomeError> {
        for (name, value) in [
            ("yes_qty", self.yes_qty),
            ("no_qty", self.no_qty),
            ("free_collateral", self.free_collateral),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OutcomeError::InvalidMarket(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        for (name, value, qty) in [
            ("yes_average_cost", self.yes_average_cost, self.yes_qty),
            ("no_average_cost", self.no_average_cost, self.no_qty),
        ] {
            if !value.is_finite()
                || value < 0.0
                || value > payout_unit
                || (qty <= EPSILON && value.abs() > EPSILON)
            {
                return Err(OutcomeError::InvalidMarket(format!(
                    "{name} is inconsistent with inventory"
                )));
            }
        }
        Ok(())
    }

    pub fn residual_qty(&self) -> f64 {
        self.yes_qty - self.no_qty
    }

    pub fn total_inventory_qty(&self) -> f64 {
        self.yes_qty + self.no_qty
    }
}

impl OutcomeNativeOrderIntent {
    fn canonical_exposure_delta(&self) -> f64 {
        let sign = match (self.outcome, self.side) {
            (Outcome::Yes, OutcomeOrderSide::Buy) | (Outcome::No, OutcomeOrderSide::Sell) => 1.0,
            (Outcome::Yes, OutcomeOrderSide::Sell) | (Outcome::No, OutcomeOrderSide::Buy) => -1.0,
        };
        sign * self.qty
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeNativeOrderIntent {
    pub outcome: Outcome,
    pub side: OutcomeOrderSide,
    pub native_price: f64,
    pub canonical_yes_price: f64,
    pub qty: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorQuotes {
    pub timestamp_ms: u64,
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub inventory_shift: f64,
    pub canonical_bid: Option<OutcomeNativeOrderIntent>,
    pub canonical_ask: Option<OutcomeNativeOrderIntent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorState {
    ema_fast: f64,
    ema_slow: f64,
    observations: u64,
    initialized: bool,
}

impl Default for OutcomeEmaAnchorState {
    fn default() -> Self {
        Self {
            ema_fast: 0.0,
            ema_slow: 0.0,
            observations: 0,
            initialized: false,
        }
    }
}

impl OutcomeEmaAnchorState {
    pub fn update(
        &mut self,
        canonical_close: f64,
        payout_unit: f64,
        params: &OutcomeEmaAnchorParams,
    ) -> Result<(), OutcomeError> {
        params.validate()?;
        if !payout_unit.is_finite()
            || payout_unit <= 0.0
            || !canonical_close.is_finite()
            || canonical_close < 0.0
            || canonical_close > payout_unit
        {
            return Err(OutcomeError::InvalidPrice(canonical_close));
        }
        if !self.initialized {
            self.ema_fast = canonical_close;
            self.ema_slow = canonical_close;
            self.observations = 1;
            self.initialized = true;
            return Ok(());
        }
        let fast_alpha = 2.0 / (params.ema_span_fast_seconds + 1.0);
        let slow_alpha = 2.0 / (params.ema_span_slow_seconds + 1.0);
        self.ema_fast = self.ema_fast * (1.0 - fast_alpha) + canonical_close * fast_alpha;
        self.ema_slow = self.ema_slow * (1.0 - slow_alpha) + canonical_close * slow_alpha;
        self.observations = self.observations.saturating_add(1);
        Ok(())
    }

    pub fn quote(
        &self,
        timestamp_ms: u64,
        canonical_close: f64,
        market: &BinaryOutcomeMarketSpec,
        params: &OutcomeEmaAnchorParams,
        inventory: &OutcomeInventorySnapshot,
    ) -> Result<OutcomeEmaAnchorQuotes, OutcomeError> {
        market.validate()?;
        params.validate()?;
        inventory.validate(market.payout_unit)?;
        if !self.initialized {
            return Err(OutcomeError::InvalidMarket(
                "outcome EMA anchor is not initialized".to_string(),
            ));
        }
        if !canonical_close.is_finite()
            || canonical_close < 0.0
            || canonical_close > market.payout_unit
        {
            return Err(OutcomeError::InvalidPrice(canonical_close));
        }
        if timestamp_ms < market.trading_opens_ms
            || timestamp_ms >= market.trading_closes_ms
            || self.observations < params.ema_warmup_seconds
            || market.trading_closes_ms.saturating_sub(timestamp_ms)
                <= params.entry_cutoff_ms_before_close
        {
            return Ok(OutcomeEmaAnchorQuotes {
                timestamp_ms,
                ema_fast: self.ema_fast,
                ema_slow: self.ema_slow,
                inventory_shift: 0.0,
                canonical_bid: None,
                canonical_ask: None,
            });
        }

        let canonical_min = market.min_price.max(market.payout_unit - market.max_price);
        let canonical_max = market.max_price.min(market.payout_unit - market.min_price);
        if canonical_min >= canonical_max {
            return Err(OutcomeError::InvalidMarket(
                "YES and NO native price bounds have no common canonical domain".to_string(),
            ));
        }
        let residual_ratio =
            (inventory.residual_qty() / params.max_abs_residual_qty).clamp(-1.0, 1.0);
        let inventory_shift = residual_ratio * params.inventory_skew;
        let lower_anchor = self.ema_fast.min(self.ema_slow);
        let upper_anchor = self.ema_fast.max(self.ema_slow);
        let raw_bid = canonical_close
            .min(lower_anchor - params.quote_offset - inventory_shift)
            .clamp(canonical_min, canonical_max);
        let raw_ask = canonical_close
            .max(upper_anchor + params.quote_offset - inventory_shift)
            .clamp(canonical_min, canonical_max);
        let mut bid_price = market
            .round_price_down(raw_bid)?
            .min(market.payout_unit - market.round_price_up(market.payout_unit - raw_bid)?)
            .max(canonical_min);
        let mut ask_price = market
            .round_price_up(raw_ask)?
            .max(market.payout_unit - market.round_price_down(market.payout_unit - raw_ask)?)
            .min(canonical_max);
        let remaining_ms = market.trading_closes_ms.saturating_sub(timestamp_ms);
        if params.risk_reduction_only_ms_before_close > 0
            && remaining_ms <= params.risk_reduction_only_ms_before_close
        {
            let residual = inventory.residual_qty();
            let (bid, ask) = if residual > EPSILON {
                let exit_price = market
                    .round_price_up(canonical_close)?
                    .max(canonical_min)
                    .min(canonical_max);
                (
                    None,
                    sell_intent(
                        Outcome::Yes,
                        exit_price,
                        params.clip_qty.min(residual).min(inventory.yes_qty),
                        market,
                    ),
                )
            } else if residual < -EPSILON {
                let native_exit = market
                    .round_price_up(market.payout_unit - canonical_close)?
                    .max(market.min_price)
                    .min(market.max_price);
                (
                    sell_intent(
                        Outcome::No,
                        native_exit,
                        params.clip_qty.min(-residual).min(inventory.no_qty),
                        market,
                    ),
                    None,
                )
            } else {
                (None, None)
            };
            return Ok(OutcomeEmaAnchorQuotes {
                timestamp_ms,
                ema_fast: self.ema_fast,
                ema_slow: self.ema_slow,
                inventory_shift,
                canonical_bid: bid,
                canonical_ask: ask,
            });
        }
        if params.execution_mode == OutcomeEmaAnchorExecutionMode::AccumulatePairs {
            if inventory.residual_qty() > EPSILON {
                // A completed pair is settlement-neutral. Once YES inventory is ahead, quote
                // the NO complement as aggressively as the last fill and locked-edge floor
                // permit instead of waiting only for the EMA ask to trade through.
                let completion_floor = inventory.yes_average_cost
                    + params.min_locked_pair_edge
                    + 2.0 * params.estimated_fee_per_share;
                let target = canonical_close.max(completion_floor).min(canonical_max);
                let rounded = market
                    .round_price_up(target)?
                    .max(
                        market.payout_unit
                            - market.round_price_down(market.payout_unit - target)?,
                    )
                    .min(canonical_max);
                ask_price = rounded;
            } else if inventory.residual_qty() < -EPSILON {
                let completion_ceiling = market.payout_unit
                    - inventory.no_average_cost
                    - params.min_locked_pair_edge
                    - 2.0 * params.estimated_fee_per_share;
                let target = canonical_close.min(completion_ceiling).max(canonical_min);
                let rounded = market
                    .round_price_down(target)?
                    .min(market.payout_unit - market.round_price_up(market.payout_unit - target)?)
                    .max(canonical_min);
                bid_price = rounded;
            }
        }
        let completion_overlap = ask_price - bid_price <= EPSILON;
        if completion_overlap
            && !(params.execution_mode == OutcomeEmaAnchorExecutionMode::AccumulatePairs
                && inventory.residual_qty().abs() > EPSILON)
        {
            return Ok(OutcomeEmaAnchorQuotes {
                timestamp_ms,
                ema_fast: self.ema_fast,
                ema_slow: self.ema_slow,
                inventory_shift,
                canonical_bid: None,
                canonical_ask: None,
            });
        }

        let remaining_inventory =
            (params.max_total_inventory_qty - inventory.total_inventory_qty()).max(0.0);
        let pair_cost_per_qty =
            bid_price + (market.payout_unit - ask_price) + 2.0 * params.estimated_fee_per_share;
        let affordable_pair_qty = if pair_cost_per_qty > 0.0 {
            inventory.free_collateral / pair_cost_per_qty
        } else {
            0.0
        };
        let buy_qty = params
            .clip_qty
            .min(remaining_inventory * 0.5)
            .min(affordable_pair_qty);
        let mut bid = self.bid_intent(bid_price, buy_qty, market, params, inventory);
        let mut ask = self.ask_intent(ask_price, buy_qty, market, params, inventory);
        bid = cap_intent_to_residual_bounds(
            bid,
            inventory.residual_qty(),
            params.max_abs_residual_qty,
            market,
        );
        ask = cap_intent_to_residual_bounds(
            ask,
            inventory.residual_qty(),
            params.max_abs_residual_qty,
            market,
        );

        if completion_overlap {
            if inventory.residual_qty() > EPSILON {
                bid = None;
            } else {
                ask = None;
            }
        }
        if params.execution_mode == OutcomeEmaAnchorExecutionMode::AccumulatePairs {
            if inventory.residual_qty() > EPSILON {
                bid = None;
            } else if inventory.residual_qty() < -EPSILON {
                ask = None;
            }
        }
        if inventory.residual_qty() >= params.max_abs_residual_qty - EPSILON {
            bid = None;
        }
        if inventory.residual_qty() <= -params.max_abs_residual_qty + EPSILON {
            ask = None;
        }
        if inventory.residual_qty().abs() <= EPSILON
            && ask_price - bid_price
                < params.min_locked_pair_edge + 2.0 * params.estimated_fee_per_share
        {
            bid = None;
            ask = None;
        }
        if let Some(intent) = bid {
            if intent.side == OutcomeOrderSide::Buy
                && intent.outcome == Outcome::Yes
                && inventory.no_qty > inventory.yes_qty + EPSILON
            {
                let edge = market.payout_unit
                    - bid_price
                    - inventory.no_average_cost
                    - 2.0 * params.estimated_fee_per_share;
                if edge + EPSILON < params.min_locked_pair_edge {
                    bid = None;
                }
            }
        }
        if let Some(intent) = ask {
            if intent.side == OutcomeOrderSide::Buy
                && intent.outcome == Outcome::No
                && inventory.yes_qty > inventory.no_qty + EPSILON
            {
                let no_price = market.payout_unit - ask_price;
                let edge = market.payout_unit
                    - inventory.yes_average_cost
                    - no_price
                    - 2.0 * params.estimated_fee_per_share;
                if edge + EPSILON < params.min_locked_pair_edge {
                    ask = None;
                }
            }
        }
        Ok(OutcomeEmaAnchorQuotes {
            timestamp_ms,
            ema_fast: self.ema_fast,
            ema_slow: self.ema_slow,
            inventory_shift,
            canonical_bid: bid,
            canonical_ask: ask,
        })
    }

    fn bid_intent(
        &self,
        canonical_price: f64,
        buy_qty: f64,
        market: &BinaryOutcomeMarketSpec,
        params: &OutcomeEmaAnchorParams,
        inventory: &OutcomeInventorySnapshot,
    ) -> Option<OutcomeNativeOrderIntent> {
        match params.execution_mode {
            OutcomeEmaAnchorExecutionMode::AccumulatePairs
            | OutcomeEmaAnchorExecutionMode::YesOnly => {
                buy_intent(Outcome::Yes, canonical_price, buy_qty, market)
            }
            OutcomeEmaAnchorExecutionMode::InventoryAware => {
                let sell_qty = params.clip_qty.min(inventory.no_qty);
                let native_price = market.payout_unit - canonical_price;
                if sell_qty >= market.min_qty
                    && sell_qty * native_price + EPSILON >= market.min_notional
                {
                    Some(OutcomeNativeOrderIntent {
                        outcome: Outcome::No,
                        side: OutcomeOrderSide::Sell,
                        native_price,
                        canonical_yes_price: canonical_price,
                        qty: round_down(sell_qty, market.qty_step),
                    })
                } else {
                    buy_intent(Outcome::Yes, canonical_price, buy_qty, market)
                }
            }
        }
    }

    fn ask_intent(
        &self,
        canonical_price: f64,
        buy_qty: f64,
        market: &BinaryOutcomeMarketSpec,
        params: &OutcomeEmaAnchorParams,
        inventory: &OutcomeInventorySnapshot,
    ) -> Option<OutcomeNativeOrderIntent> {
        match params.execution_mode {
            OutcomeEmaAnchorExecutionMode::AccumulatePairs => buy_intent(
                Outcome::No,
                market.payout_unit - canonical_price,
                buy_qty,
                market,
            ),
            OutcomeEmaAnchorExecutionMode::InventoryAware
            | OutcomeEmaAnchorExecutionMode::YesOnly => {
                let sell_qty = params.clip_qty.min(inventory.yes_qty);
                if sell_qty >= market.min_qty
                    && sell_qty * canonical_price + EPSILON >= market.min_notional
                {
                    Some(OutcomeNativeOrderIntent {
                        outcome: Outcome::Yes,
                        side: OutcomeOrderSide::Sell,
                        native_price: canonical_price,
                        canonical_yes_price: canonical_price,
                        qty: round_down(sell_qty, market.qty_step),
                    })
                } else if params.execution_mode == OutcomeEmaAnchorExecutionMode::InventoryAware {
                    buy_intent(
                        Outcome::No,
                        market.payout_unit - canonical_price,
                        buy_qty,
                        market,
                    )
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorObservation {
    pub timestamp_ms: u64,
    pub close: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorPlanInput {
    pub market: BinaryOutcomeMarketSpec,
    pub strategy_params: OutcomeEmaAnchorParams,
    pub observations: Vec<OutcomeEmaAnchorObservation>,
    pub inventory: OutcomeInventorySnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorPlanOutput {
    pub strategy_kind: String,
    pub observation_start_ms: u64,
    pub observation_end_ms: u64,
    pub observation_count: usize,
    pub quotes: OutcomeEmaAnchorQuotes,
}

pub fn plan_outcome_ema_anchor(
    input: &OutcomeEmaAnchorPlanInput,
) -> Result<OutcomeEmaAnchorPlanOutput, OutcomeError> {
    input.market.validate()?;
    input.strategy_params.validate()?;
    input.inventory.validate(input.market.payout_unit)?;
    if input.observations.is_empty() {
        return Err(OutcomeError::InvalidMarket(
            "outcome EMA-anchor planning requires observations".to_string(),
        ));
    }

    let mut state = OutcomeEmaAnchorState::default();
    for (index, observation) in input.observations.iter().enumerate() {
        if observation.timestamp_ms % 1_000 != 0 {
            return Err(OutcomeError::InvalidMarket(
                "outcome EMA-anchor observations must be second-aligned".to_string(),
            ));
        }
        if observation.timestamp_ms < input.market.trading_opens_ms
            || observation.timestamp_ms >= input.market.trading_closes_ms
        {
            return Err(OutcomeError::MarketNotTrading(observation.timestamp_ms));
        }
        if index > 0
            && observation.timestamp_ms != input.observations[index - 1].timestamp_ms + 1_000
        {
            return Err(OutcomeError::InvalidMarket(
                "outcome EMA-anchor observations must be contiguous".to_string(),
            ));
        }
        state.update(
            observation.close,
            input.market.payout_unit,
            &input.strategy_params,
        )?;
    }

    let first = input.observations.first().expect("non-empty checked");
    let last = input.observations.last().expect("non-empty checked");
    let quotes = state.quote(
        last.timestamp_ms.saturating_add(1_000),
        last.close,
        &input.market,
        &input.strategy_params,
        &input.inventory,
    )?;
    Ok(OutcomeEmaAnchorPlanOutput {
        strategy_kind: "ema_anchor_outcome".to_string(),
        observation_start_ms: first.timestamp_ms,
        observation_end_ms: last.timestamp_ms.saturating_add(1_000),
        observation_count: input.observations.len(),
        quotes,
    })
}

fn buy_intent(
    outcome: Outcome,
    native_price: f64,
    qty: f64,
    market: &BinaryOutcomeMarketSpec,
) -> Option<OutcomeNativeOrderIntent> {
    let qty = round_down(qty, market.qty_step);
    if qty < market.min_qty
        || qty * native_price + EPSILON < market.min_notional
        || native_price < market.min_price
        || native_price > market.max_price
        || !market.price_grid.is_valid(native_price)
    {
        return None;
    }
    let canonical_yes_price = match outcome {
        Outcome::Yes => native_price,
        Outcome::No => market.payout_unit - native_price,
    };
    Some(OutcomeNativeOrderIntent {
        outcome,
        side: OutcomeOrderSide::Buy,
        native_price,
        canonical_yes_price,
        qty,
    })
}

fn sell_intent(
    outcome: Outcome,
    native_price: f64,
    qty: f64,
    market: &BinaryOutcomeMarketSpec,
) -> Option<OutcomeNativeOrderIntent> {
    let qty = round_down(qty, market.qty_step);
    if qty < market.min_qty
        || qty * native_price + EPSILON < market.min_notional
        || native_price < market.min_price
        || native_price > market.max_price
        || !market.price_grid.is_valid(native_price)
    {
        return None;
    }
    let canonical_yes_price = match outcome {
        Outcome::Yes => native_price,
        Outcome::No => market.payout_unit - native_price,
    };
    Some(OutcomeNativeOrderIntent {
        outcome,
        side: OutcomeOrderSide::Sell,
        native_price,
        canonical_yes_price,
        qty,
    })
}

fn cap_intent_to_residual_bounds(
    intent: Option<OutcomeNativeOrderIntent>,
    residual_qty: f64,
    max_abs_residual_qty: f64,
    market: &BinaryOutcomeMarketSpec,
) -> Option<OutcomeNativeOrderIntent> {
    let intent = intent?;
    let exposure_delta = intent.canonical_exposure_delta();
    let residual_headroom = if exposure_delta > 0.0 {
        max_abs_residual_qty - residual_qty
    } else {
        max_abs_residual_qty + residual_qty
    }
    .max(0.0);
    let qty = intent.qty.min(residual_headroom);
    match intent.side {
        OutcomeOrderSide::Buy => buy_intent(intent.outcome, intent.native_price, qty, market),
        OutcomeOrderSide::Sell => sell_intent(intent.outcome, intent.native_price, qty, market),
    }
}

fn round_down(value: f64, step: f64) -> f64 {
    ((value + EPSILON) / step).floor() * step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::OutcomeVenueCapabilities;

    fn market() -> BinaryOutcomeMarketSpec {
        BinaryOutcomeMarketSpec {
            venue: "fixture".to_string(),
            market_id: "outcome".to_string(),
            yes_asset_id: "yes".to_string(),
            no_asset_id: "no".to_string(),
            payout_unit: 1.0,
            min_price: 0.001,
            max_price: 0.999,
            price_grid: super::super::OutcomePriceGrid::FixedStep { step: 0.001 },
            qty_step: 0.1,
            min_qty: 0.1,
            min_notional: 0.0,
            trading_opens_ms: 1_000,
            trading_closes_ms: 100_000,
            scheduled_resolution_ms: 100_000,
            capabilities: OutcomeVenueCapabilities {
                complementary_books_merged: false,
                supports_split: true,
                supports_merge: true,
                supports_redeem: true,
                supports_post_only: true,
                supports_gtd: true,
                sell_requires_inventory: true,
            },
        }
    }

    fn params(mode: OutcomeEmaAnchorExecutionMode) -> OutcomeEmaAnchorParams {
        OutcomeEmaAnchorParams {
            ema_span_fast_seconds: 10.0,
            ema_span_slow_seconds: 60.0,
            ema_warmup_seconds: 0,
            quote_offset: 0.01,
            inventory_skew: 0.02,
            clip_qty: 1.0,
            max_total_inventory_qty: 100.0,
            max_abs_residual_qty: 10.0,
            min_locked_pair_edge: 0.005,
            estimated_fee_per_share: 0.001,
            risk_reduction_only_ms_before_close: 10_000,
            entry_cutoff_ms_before_close: 5_000,
            execution_mode: mode,
        }
    }

    fn flat_inventory() -> OutcomeInventorySnapshot {
        OutcomeInventorySnapshot {
            yes_qty: 0.0,
            no_qty: 0.0,
            yes_average_cost: 0.0,
            no_average_cost: 0.0,
            free_collateral: 100.0,
        }
    }

    #[test]
    fn accumulate_mode_maps_canonical_bid_to_buy_yes_and_ask_to_buy_no() {
        let mut state = OutcomeEmaAnchorState::default();
        state
            .update(
                0.5,
                market().payout_unit,
                &params(OutcomeEmaAnchorExecutionMode::AccumulatePairs),
            )
            .unwrap();
        let quotes = state
            .quote(
                2_000,
                0.5,
                &market(),
                &params(OutcomeEmaAnchorExecutionMode::AccumulatePairs),
                &flat_inventory(),
            )
            .unwrap();
        let bid = quotes.canonical_bid.unwrap();
        let ask = quotes.canonical_ask.unwrap();
        assert_eq!(
            (bid.outcome, bid.side),
            (Outcome::Yes, OutcomeOrderSide::Buy)
        );
        assert_eq!(
            (ask.outcome, ask.side),
            (Outcome::No, OutcomeOrderSide::Buy)
        );
        assert!((bid.native_price - 0.49).abs() < 1e-12);
        assert!((ask.native_price - 0.49).abs() < 1e-12);
        assert!((ask.canonical_yes_price - 0.51).abs() < 1e-12);
    }

    #[test]
    fn quote_sizes_cannot_cross_configured_residual_bounds() {
        let mut state = OutcomeEmaAnchorState::default();
        let mut parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        parameters.clip_qty = 25.0;
        parameters.max_abs_residual_qty = 10.0;
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();

        let quotes = state
            .quote(2_000, 0.5, &market(), &parameters, &flat_inventory())
            .unwrap();

        assert_eq!(quotes.canonical_bid.unwrap().qty, 10.0);
        assert_eq!(quotes.canonical_ask.unwrap().qty, 10.0);
    }

    #[test]
    fn absolute_offsets_and_price_bounds_remain_symmetric_near_extremes() {
        let mut state = OutcomeEmaAnchorState::default();
        state
            .update(
                0.005,
                market().payout_unit,
                &params(OutcomeEmaAnchorExecutionMode::AccumulatePairs),
            )
            .unwrap();
        let quotes = state
            .quote(
                2_000,
                0.005,
                &market(),
                &params(OutcomeEmaAnchorExecutionMode::AccumulatePairs),
                &flat_inventory(),
            )
            .unwrap();
        assert!((quotes.canonical_bid.unwrap().canonical_yes_price - 0.001).abs() < 1e-12);
        assert!((quotes.canonical_ask.unwrap().canonical_yes_price - 0.015).abs() < 1e-12);
    }

    #[test]
    fn positive_yes_residual_quotes_only_the_missing_complement() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 10.0,
            no_qty: 0.0,
            yes_average_cost: 0.4,
            no_average_cost: 0.0,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(2_000, 0.5, &market(), &parameters, &inventory)
            .unwrap();
        assert!(quotes.canonical_bid.is_none());
        assert!(quotes.inventory_shift > 0.0);
        let completion = quotes.canonical_ask.unwrap();
        assert_eq!(
            (completion.outcome, completion.side),
            (Outcome::No, OutcomeOrderSide::Buy)
        );
        assert!(completion.canonical_yes_price < 0.51);
    }

    #[test]
    fn accumulate_mode_quotes_complement_at_edge_constrained_current_price() {
        let mut state = OutcomeEmaAnchorState::default();
        let mut parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        parameters.inventory_skew = 0.0;
        state
            .update(0.6, market().payout_unit, &parameters)
            .unwrap();
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 1.0,
            no_qty: 0.0,
            yes_average_cost: 0.4,
            no_average_cost: 0.0,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(2_000, 0.5, &market(), &parameters, &inventory)
            .unwrap();
        let completion = quotes.canonical_ask.unwrap();

        assert_eq!(
            (completion.outcome, completion.side),
            (Outcome::No, OutcomeOrderSide::Buy)
        );
        assert!((completion.canonical_yes_price - 0.5).abs() < 1e-12);
        assert!((completion.native_price - 0.5).abs() < 1e-12);
    }

    #[test]
    fn accumulate_mode_never_completes_pair_below_locked_edge_floor() {
        let mut state = OutcomeEmaAnchorState::default();
        let mut parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        parameters.inventory_skew = 0.0;
        state
            .update(0.6, market().payout_unit, &parameters)
            .unwrap();
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 1.0,
            no_qty: 0.0,
            yes_average_cost: 0.7,
            no_average_cost: 0.0,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(2_000, 0.5, &market(), &parameters, &inventory)
            .unwrap();
        let completion = quotes.canonical_ask.unwrap();
        let locked_edge = market().payout_unit
            - inventory.yes_average_cost
            - completion.native_price
            - 2.0 * parameters.estimated_fee_per_share;

        assert!(locked_edge + 1e-12 >= parameters.min_locked_pair_edge);
        assert!((completion.canonical_yes_price - 0.707).abs() < 1e-12);
    }

    #[test]
    fn inventory_aware_mode_sells_existing_tokens_before_buying_complements() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::InventoryAware);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 2.0,
            no_qty: 3.0,
            yes_average_cost: 0.4,
            no_average_cost: 0.4,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(2_000, 0.5, &market(), &parameters, &inventory)
            .unwrap();
        let bid = quotes.canonical_bid.unwrap();
        let ask = quotes.canonical_ask.unwrap();
        assert_eq!(
            (bid.outcome, bid.side),
            (Outcome::No, OutcomeOrderSide::Sell)
        );
        assert_eq!(
            (ask.outcome, ask.side),
            (Outcome::Yes, OutcomeOrderSide::Sell)
        );
    }

    #[test]
    fn yes_only_mode_never_invents_a_sell_without_yes_inventory() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::YesOnly);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let flat = state
            .quote(2_000, 0.5, &market(), &parameters, &flat_inventory())
            .unwrap();
        assert!(flat.canonical_bid.is_some());
        assert!(flat.canonical_ask.is_none());
    }

    #[test]
    fn entry_cutoff_disables_both_quotes() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let quotes = state
            .quote(95_000, 0.5, &market(), &parameters, &flat_inventory())
            .unwrap();
        assert!(quotes.canonical_bid.is_none());
        assert!(quotes.canonical_ask.is_none());
    }

    #[test]
    fn risk_reduction_window_sells_excess_yes_instead_of_adding_no() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 2.0,
            no_qty: 1.0,
            yes_average_cost: 0.4,
            no_average_cost: 0.4,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(91_000, 0.5, &market(), &parameters, &inventory)
            .unwrap();

        assert!(quotes.canonical_bid.is_none());
        let exit = quotes.canonical_ask.unwrap();
        assert_eq!(
            (exit.outcome, exit.side),
            (Outcome::Yes, OutcomeOrderSide::Sell)
        );
        assert!((exit.native_price - 0.5).abs() < 1e-12);
        assert!(exit.canonical_exposure_delta() < 0.0);
    }

    #[test]
    fn risk_reduction_window_sells_only_excess_no() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 1.0,
            no_qty: 2.0,
            yes_average_cost: 0.4,
            no_average_cost: 0.4,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(91_000, 0.5, &market(), &parameters, &inventory)
            .unwrap();

        assert!(quotes.canonical_ask.is_none());
        let exit = quotes.canonical_bid.unwrap();
        assert_eq!(
            (exit.outcome, exit.side),
            (Outcome::No, OutcomeOrderSide::Sell)
        );
        assert!((exit.native_price - 0.5).abs() < 1e-12);
        assert!(exit.canonical_exposure_delta() > 0.0);
    }

    #[test]
    fn risk_reduction_window_does_not_open_fresh_inventory_when_flat() {
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let quotes = state
            .quote(91_000, 0.5, &market(), &parameters, &flat_inventory())
            .unwrap();

        assert!(quotes.canonical_bid.is_none());
        assert!(quotes.canonical_ask.is_none());
    }

    #[test]
    fn risk_reduction_no_sale_uses_the_native_no_significant_figure_grid() {
        let mut significant_market = market();
        significant_market.price_grid = super::super::OutcomePriceGrid::SignificantFigures {
            max_significant_figures: 3,
            max_decimal_places: 6,
        };
        let mut state = OutcomeEmaAnchorState::default();
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        state
            .update(0.333_333, significant_market.payout_unit, &parameters)
            .unwrap();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: 1.0,
            no_qty: 2.0,
            yes_average_cost: 0.4,
            no_average_cost: 0.4,
            free_collateral: 100.0,
        };
        let quotes = state
            .quote(
                91_000,
                0.333_333,
                &significant_market,
                &parameters,
                &inventory,
            )
            .unwrap();

        let exit = quotes.canonical_bid.unwrap();
        assert_eq!(
            (exit.outcome, exit.side),
            (Outcome::No, OutcomeOrderSide::Sell)
        );
        assert!(significant_market.price_grid.is_valid(exit.native_price));
        assert!((exit.native_price - 0.667).abs() < 1e-12);
    }

    #[test]
    fn stateless_planner_reconstructs_state_and_rejects_gaps() {
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        let input = OutcomeEmaAnchorPlanInput {
            market: market(),
            strategy_params: parameters,
            observations: vec![
                OutcomeEmaAnchorObservation {
                    timestamp_ms: 1_000,
                    close: 0.5,
                },
                OutcomeEmaAnchorObservation {
                    timestamp_ms: 2_000,
                    close: 0.51,
                },
            ],
            inventory: flat_inventory(),
        };
        let output = plan_outcome_ema_anchor(&input).unwrap();
        assert_eq!(output.strategy_kind, "ema_anchor_outcome");
        assert_eq!(output.observation_count, 2);
        assert!(output.quotes.canonical_bid.is_some());
        assert!(output.quotes.canonical_ask.is_some());

        let mut gapped = input;
        gapped.observations[1].timestamp_ms = 3_000;
        assert!(matches!(
            plan_outcome_ema_anchor(&gapped),
            Err(OutcomeError::InvalidMarket(_))
        ));
    }

    #[test]
    fn stateless_planner_uses_completed_candle_boundary_for_cutoff() {
        let parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        let input = OutcomeEmaAnchorPlanInput {
            market: market(),
            strategy_params: parameters,
            observations: vec![OutcomeEmaAnchorObservation {
                timestamp_ms: 94_000,
                close: 0.5,
            }],
            inventory: flat_inventory(),
        };

        let output = plan_outcome_ema_anchor(&input).unwrap();

        assert_eq!(output.observation_end_ms, 95_000);
        assert_eq!(output.quotes.timestamp_ms, 95_000);
        assert!(output.quotes.canonical_bid.is_none());
        assert!(output.quotes.canonical_ask.is_none());
    }

    #[test]
    fn ema_observations_accept_the_market_payout_range() {
        let mut larger_payout_market = market();
        larger_payout_market.payout_unit = 2.0;
        larger_payout_market.max_price = 1.999;
        let input = OutcomeEmaAnchorPlanInput {
            market: larger_payout_market,
            strategy_params: params(OutcomeEmaAnchorExecutionMode::AccumulatePairs),
            observations: vec![OutcomeEmaAnchorObservation {
                timestamp_ms: 1_000,
                close: 1.2,
            }],
            inventory: flat_inventory(),
        };

        let output = plan_outcome_ema_anchor(&input).unwrap();

        assert!((output.quotes.ema_fast - 1.2).abs() < 1e-12);
        assert!((output.quotes.ema_slow - 1.2).abs() < 1e-12);
    }

    #[test]
    fn warmup_suppresses_quotes_until_enough_dense_seconds_exist() {
        let mut parameters = params(OutcomeEmaAnchorExecutionMode::AccumulatePairs);
        parameters.ema_warmup_seconds = 3;
        let mut state = OutcomeEmaAnchorState::default();
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let warming = state
            .quote(2_000, 0.5, &market(), &parameters, &flat_inventory())
            .unwrap();
        assert!(warming.canonical_bid.is_none());
        assert!(warming.canonical_ask.is_none());

        state
            .update(0.5, market().payout_unit, &parameters)
            .unwrap();
        let ready = state
            .quote(3_000, 0.5, &market(), &parameters, &flat_inventory())
            .unwrap();
        assert!(ready.canonical_bid.is_some());
        assert!(ready.canonical_ask.is_some());
    }
}
