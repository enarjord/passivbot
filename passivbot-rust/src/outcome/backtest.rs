use super::simulator::{OutcomeCandle, SimulatedOutcomeFill, SingleOutcomeSimulator};
use super::strategy::{
    OutcomeEmaAnchorParams, OutcomeEmaAnchorState, OutcomeInventorySnapshot,
    OutcomeNativeOrderIntent,
};
use super::{
    BinaryOutcomeMarketSpec, Outcome, OutcomeError, OutcomeFeeSchedule, OutcomeLimitOrder,
    OutcomeOrderSide, OutcomePriceGrid, OutcomeSettlement,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutcomeBacktestAction {
    PlaceOrder {
        timestamp_ms: u64,
        order: OutcomeLimitOrder,
    },
    CancelOrder {
        timestamp_ms: u64,
        order_id: String,
    },
    Split {
        timestamp_ms: u64,
        qty: f64,
        yes_reference_price: f64,
    },
    Merge {
        timestamp_ms: u64,
        qty: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomePriceGridChange {
    pub timestamp_ms: u64,
    pub old_grid: OutcomePriceGrid,
    pub new_grid: OutcomePriceGrid,
}

impl OutcomeBacktestAction {
    fn timestamp_ms(&self) -> u64 {
        match self {
            Self::PlaceOrder { timestamp_ms, .. }
            | Self::CancelOrder { timestamp_ms, .. }
            | Self::Split { timestamp_ms, .. }
            | Self::Merge { timestamp_ms, .. } => *timestamp_ms,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleOutcomeBacktestInput {
    pub market: BinaryOutcomeMarketSpec,
    pub fee_schedule: OutcomeFeeSchedule,
    pub starting_collateral: f64,
    pub actions: Vec<OutcomeBacktestAction>,
    pub candles: Vec<OutcomeCandle>,
    #[serde(default)]
    pub price_grid_changes: Vec<OutcomePriceGridChange>,
    pub settlement_time_ms: u64,
    pub yes_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeSignalCandle {
    pub timestamp_ms: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OutcomeSignalCandle {
    fn validate(&self, payout_unit: f64) -> Result<(), OutcomeError> {
        OutcomeCandle {
            timestamp_ms: self.timestamp_ms,
            outcome: super::Outcome::Yes,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
        }
        .validate(payout_unit)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEmaAnchorBacktestInput {
    pub market: BinaryOutcomeMarketSpec,
    pub fee_schedule: OutcomeFeeSchedule,
    pub starting_collateral: f64,
    pub strategy_params: OutcomeEmaAnchorParams,
    pub signal_candles: Vec<OutcomeSignalCandle>,
    pub execution_candles: Vec<OutcomeCandle>,
    #[serde(default)]
    pub price_grid_changes: Vec<OutcomePriceGridChange>,
    pub settlement_time_ms: u64,
    pub yes_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleOutcomeBacktestOutput {
    pub strategy_kind: String,
    pub market_id: String,
    pub trading_open_time_ms: u64,
    pub settlement_time_ms: u64,
    pub starting_collateral: f64,
    pub ending_collateral: f64,
    pub fills: Vec<SimulatedOutcomeFill>,
    pub orders_placed_count: usize,
    pub fills_count: usize,
    pub maker_fills_count: usize,
    pub traded_notional: f64,
    pub fees_paid: f64,
    pub rebates_earned: f64,
    pub gross_spread_pnl: f64,
    pub settlement_pnl: f64,
    pub pre_settlement_yes_qty: f64,
    pub pre_settlement_no_qty: f64,
    pub pre_settlement_yes_cost: f64,
    pub pre_settlement_no_cost: f64,
    pub pre_settlement_paired_qty: f64,
    pub pre_settlement_net_yes_exposure: f64,
    pub pre_settlement_worst_case_equity: f64,
    pub max_paired_qty: f64,
    pub max_abs_residual_qty: f64,
    pub cumulative_yes_buy_qty: f64,
    pub cumulative_no_buy_qty: f64,
    pub pair_completion_ratio: f64,
    pub time_weighted_abs_residual_qty: f64,
    pub time_weighted_total_inventory_qty: f64,
    pub worst_case_settlement_equity_min: f64,
    pub settlement: OutcomeSettlement,
    pub fill_model: String,
}

enum TimelineEvent<'a> {
    PriceGridChange(usize, &'a OutcomePriceGridChange),
    Action(usize, &'a OutcomeBacktestAction),
    Candle(usize, &'a OutcomeCandle),
}

impl TimelineEvent<'_> {
    fn timestamp_ms(&self) -> u64 {
        match self {
            Self::PriceGridChange(_, change) => change.timestamp_ms,
            Self::Action(_, action) => action.timestamp_ms(),
            Self::Candle(_, candle) => candle.timestamp_ms,
        }
    }

    fn priority(&self) -> u8 {
        match self {
            Self::PriceGridChange(_, _) => 0,
            Self::Action(_, _) => 1,
            Self::Candle(_, _) => 2,
        }
    }

    fn sequence(&self) -> usize {
        match self {
            Self::PriceGridChange(sequence, _)
            | Self::Action(sequence, _)
            | Self::Candle(sequence, _) => *sequence,
        }
    }
}

fn validate_price_grid_change(
    market: &BinaryOutcomeMarketSpec,
    change: &OutcomePriceGridChange,
) -> Result<(), OutcomeError> {
    if change.timestamp_ms < market.trading_opens_ms
        || change.timestamp_ms >= market.trading_closes_ms
    {
        return Err(OutcomeError::MarketNotTrading(change.timestamp_ms));
    }
    change.old_grid.validate()?;
    change.new_grid.validate()?;
    if change.old_grid == change.new_grid {
        return Err(OutcomeError::InvalidMarket(
            "outcome price-grid change must change the grid".to_string(),
        ));
    }
    Ok(())
}

fn apply_price_grid_changes_before(
    simulator: &mut SingleOutcomeSimulator,
    current_market: &mut BinaryOutcomeMarketSpec,
    changes: &[OutcomePriceGridChange],
    next_change: &mut usize,
    boundary_ms: u64,
) -> Result<(), OutcomeError> {
    while *next_change < changes.len() && changes[*next_change].timestamp_ms < boundary_ms {
        let change = changes[*next_change];
        if change.old_grid != current_market.price_grid {
            return Err(OutcomeError::InvalidMarket(
                "outcome price-grid change does not continue the prior grid".to_string(),
            ));
        }
        simulator.update_price_grid(change.new_grid, change.timestamp_ms)?;
        current_market.price_grid = change.new_grid;
        *next_change += 1;
    }
    Ok(())
}

pub fn run_single_outcome_backtest(
    input: &SingleOutcomeBacktestInput,
) -> Result<SingleOutcomeBacktestOutput, OutcomeError> {
    input.market.validate()?;
    input.fee_schedule.validate()?;
    if input.settlement_time_ms < input.market.trading_closes_ms {
        return Err(OutcomeError::InvalidMarket(
            "settlement_time_ms must be no earlier than trading close".to_string(),
        ));
    }
    let mut simulator = SingleOutcomeSimulator::new(
        input.market.clone(),
        input.fee_schedule,
        input.starting_collateral,
    )?;
    let mut timeline: Vec<TimelineEvent<'_>> = input
        .price_grid_changes
        .iter()
        .enumerate()
        .map(|(sequence, change)| TimelineEvent::PriceGridChange(sequence, change))
        .chain(
            input
                .actions
                .iter()
                .enumerate()
                .map(|(sequence, action)| TimelineEvent::Action(sequence, action)),
        )
        .chain(
            input
                .candles
                .iter()
                .enumerate()
                .map(|(sequence, candle)| TimelineEvent::Candle(sequence, candle)),
        )
        .collect();
    for change in &input.price_grid_changes {
        validate_price_grid_change(&input.market, change)?;
    }
    timeline.sort_by_key(|event| (event.timestamp_ms(), event.priority(), event.sequence()));

    let mut current_price_grid = input.market.price_grid;
    let mut max_paired_qty: f64 = 0.0;
    let mut max_abs_residual_qty: f64 = 0.0;
    let mut worst_case_settlement_equity_min = input.starting_collateral;
    let mut residual_qty_time_area_ms = 0.0;
    let mut total_inventory_time_area_ms = 0.0;
    let mut last_inventory_time_ms = input.market.trading_opens_ms;
    for event in timeline {
        if event.timestamp_ms() >= input.settlement_time_ms {
            return Err(OutcomeError::InvalidMarket(
                "outcome actions and candles must precede settlement".to_string(),
            ));
        }
        if event.timestamp_ms() < last_inventory_time_ms {
            return Err(OutcomeError::InvalidMarket(
                "outcome timeline event predates trading open".to_string(),
            ));
        }
        accumulate_inventory_time(
            &simulator,
            last_inventory_time_ms,
            event.timestamp_ms(),
            &mut residual_qty_time_area_ms,
            &mut total_inventory_time_area_ms,
        );
        last_inventory_time_ms = event.timestamp_ms();
        match event {
            TimelineEvent::PriceGridChange(_, change) => {
                if change.old_grid != current_price_grid {
                    return Err(OutcomeError::InvalidMarket(
                        "outcome price-grid change does not continue the prior grid".to_string(),
                    ));
                }
                simulator.update_price_grid(change.new_grid, change.timestamp_ms)?;
                current_price_grid = change.new_grid;
            }
            TimelineEvent::Action(_, action) => match action {
                OutcomeBacktestAction::PlaceOrder {
                    timestamp_ms,
                    order,
                } => simulator.place_order(order.clone(), *timestamp_ms)?,
                OutcomeBacktestAction::CancelOrder {
                    timestamp_ms,
                    order_id,
                } => {
                    simulator.expire_orders(*timestamp_ms);
                    simulator.cancel_order(order_id)?;
                }
                OutcomeBacktestAction::Split {
                    timestamp_ms,
                    qty,
                    yes_reference_price,
                } => {
                    if *timestamp_ms < input.market.trading_opens_ms
                        || *timestamp_ms >= input.market.trading_closes_ms
                    {
                        return Err(OutcomeError::MarketNotTrading(*timestamp_ms));
                    }
                    simulator.split(*qty, *yes_reference_price)?;
                }
                OutcomeBacktestAction::Merge { timestamp_ms, qty } => {
                    if *timestamp_ms < input.market.trading_opens_ms
                        || *timestamp_ms >= input.market.trading_closes_ms
                    {
                        return Err(OutcomeError::MarketNotTrading(*timestamp_ms));
                    }
                    simulator.merge(*qty)?;
                }
            },
            TimelineEvent::Candle(_, candle) => {
                simulator.process_candle(candle)?;
            }
        }
        max_paired_qty = max_paired_qty.max(simulator.ledger().paired_qty());
        max_abs_residual_qty =
            max_abs_residual_qty.max(simulator.ledger().net_yes_exposure().abs());
        worst_case_settlement_equity_min =
            worst_case_settlement_equity_min.min(simulator.worst_case_settlement_equity()?);
    }
    accumulate_inventory_time(
        &simulator,
        last_inventory_time_ms,
        input.settlement_time_ms,
        &mut residual_qty_time_area_ms,
        &mut total_inventory_time_area_ms,
    );

    let paired_locked_pnl = locked_pair_pnl(
        input.market.payout_unit,
        simulator.ledger().yes_qty(),
        simulator.ledger().yes_cost(),
        simulator.ledger().no_qty(),
        simulator.ledger().no_cost(),
    );
    let pre_settlement_trading_pnl = simulator.ledger().realized_trading_pnl();
    let pre_settlement_merge_pnl = simulator.ledger().realized_merge_pnl();
    let pre_settlement_yes_qty = simulator.ledger().yes_qty();
    let pre_settlement_no_qty = simulator.ledger().no_qty();
    let pre_settlement_yes_cost = simulator.ledger().yes_cost();
    let pre_settlement_no_cost = simulator.ledger().no_cost();
    let pre_settlement_paired_qty = simulator.ledger().paired_qty();
    let pre_settlement_net_yes_exposure = simulator.ledger().net_yes_exposure();
    let pre_settlement_worst_case_equity = simulator.worst_case_settlement_equity()?;
    let fills = simulator.fills().to_vec();
    let (cumulative_yes_buy_qty, cumulative_no_buy_qty) = cumulative_buy_qty(&fills);
    let pair_completion_ratio = completion_ratio(cumulative_yes_buy_qty, cumulative_no_buy_qty);
    let lifecycle_duration_ms = (input.settlement_time_ms - input.market.trading_opens_ms) as f64;
    let traded_notional = fills
        .iter()
        .map(|fill| fill.fill.price * fill.fill.qty)
        .sum();
    let settlement = simulator.settle(input.settlement_time_ms, input.yes_fraction)?;
    let ending_collateral = simulator.ledger().collateral();
    let fees_paid = simulator.ledger().fees_paid();
    let rebates_earned = simulator.ledger().rebates_earned();

    Ok(SingleOutcomeBacktestOutput {
        strategy_kind: "explicit_actions".to_string(),
        market_id: input.market.market_id.clone(),
        trading_open_time_ms: input.market.trading_opens_ms,
        settlement_time_ms: input.settlement_time_ms,
        starting_collateral: input.starting_collateral,
        ending_collateral,
        orders_placed_count: input
            .actions
            .iter()
            .filter(|action| matches!(action, OutcomeBacktestAction::PlaceOrder { .. }))
            .count(),
        fills_count: fills.len(),
        maker_fills_count: fills.len(),
        traded_notional,
        fees_paid,
        rebates_earned,
        gross_spread_pnl: pre_settlement_trading_pnl + pre_settlement_merge_pnl + paired_locked_pnl,
        settlement_pnl: settlement.realized_settlement_pnl - paired_locked_pnl,
        pre_settlement_yes_qty,
        pre_settlement_no_qty,
        pre_settlement_yes_cost,
        pre_settlement_no_cost,
        pre_settlement_paired_qty,
        pre_settlement_net_yes_exposure,
        pre_settlement_worst_case_equity,
        max_paired_qty,
        max_abs_residual_qty,
        cumulative_yes_buy_qty,
        cumulative_no_buy_qty,
        pair_completion_ratio,
        time_weighted_abs_residual_qty: residual_qty_time_area_ms / lifecycle_duration_ms,
        time_weighted_total_inventory_qty: total_inventory_time_area_ms / lifecycle_duration_ms,
        worst_case_settlement_equity_min,
        fills,
        settlement,
        fill_model: "trade_derived_1s_strict_cross_no_volume_cap".to_string(),
    })
}

pub fn run_outcome_ema_anchor_backtest(
    input: &OutcomeEmaAnchorBacktestInput,
) -> Result<SingleOutcomeBacktestOutput, OutcomeError> {
    input.market.validate()?;
    input.fee_schedule.validate()?;
    input.strategy_params.validate()?;
    if input.settlement_time_ms < input.market.trading_closes_ms {
        return Err(OutcomeError::InvalidMarket(
            "settlement_time_ms must be no earlier than trading close".to_string(),
        ));
    }
    if input.signal_candles.is_empty() {
        return Err(OutcomeError::InvalidMarket(
            "EMA-anchor outcome backtest requires signal candles".to_string(),
        ));
    }
    let mut signals = input.signal_candles.clone();
    signals.sort_by_key(|candle| candle.timestamp_ms);
    for (index, candle) in signals.iter().enumerate() {
        candle.validate(input.market.payout_unit)?;
        if candle.timestamp_ms % 1_000 != 0 {
            return Err(OutcomeError::InvalidMarket(
                "outcome signal candles must be second-aligned".to_string(),
            ));
        }
        if index > 0 && candle.timestamp_ms != signals[index - 1].timestamp_ms + 1_000 {
            return Err(OutcomeError::InvalidMarket(
                "outcome signal candles must be contiguous; collector gaps are unavailable"
                    .to_string(),
            ));
        }
        if candle.timestamp_ms < input.market.trading_opens_ms
            || candle.timestamp_ms >= input.market.trading_closes_ms
        {
            return Err(OutcomeError::MarketNotTrading(candle.timestamp_ms));
        }
    }
    let signal_times: HashSet<u64> = signals.iter().map(|candle| candle.timestamp_ms).collect();
    let mut price_grid_changes = input.price_grid_changes.clone();
    for change in &price_grid_changes {
        validate_price_grid_change(&input.market, change)?;
    }
    price_grid_changes.sort_by_key(|change| change.timestamp_ms);
    let mut executions_by_time: BTreeMap<u64, Vec<OutcomeCandle>> = BTreeMap::new();
    for candle in &input.execution_candles {
        candle.validate(input.market.payout_unit)?;
        if !signal_times.contains(&candle.timestamp_ms) {
            return Err(OutcomeError::InvalidMarket(
                "execution candle has no corresponding canonical signal candle".to_string(),
            ));
        }
        executions_by_time
            .entry(candle.timestamp_ms)
            .or_default()
            .push(candle.clone());
    }

    let mut simulator = SingleOutcomeSimulator::new(
        input.market.clone(),
        input.fee_schedule,
        input.starting_collateral,
    )?;
    let mut state = OutcomeEmaAnchorState::default();
    let mut max_paired_qty: f64 = 0.0;
    let mut max_abs_residual_qty: f64 = 0.0;
    let mut worst_case_settlement_equity_min = input.starting_collateral;
    let mut residual_qty_time_area_ms = 0.0;
    let mut total_inventory_time_area_ms = 0.0;
    let mut last_inventory_time_ms = input.market.trading_opens_ms;
    let mut next_order_sequence = 0_u64;
    let mut orders_placed_count = 0_usize;
    let mut current_market = input.market.clone();
    let mut next_price_grid_change = 0_usize;

    for signal in &signals {
        accumulate_inventory_time(
            &simulator,
            last_inventory_time_ms,
            signal.timestamp_ms,
            &mut residual_qty_time_area_ms,
            &mut total_inventory_time_area_ms,
        );
        // A grid change stamped at the bucket boundary is authoritative for every fill in that
        // second. Changes later inside an aggregated one-second bucket cannot be ordered against
        // its fills, so they apply after execution and before quotes for the next bucket.
        apply_price_grid_changes_before(
            &mut simulator,
            &mut current_market,
            &price_grid_changes,
            &mut next_price_grid_change,
            signal.timestamp_ms.saturating_add(1),
        )?;
        if let Some(execution_candles) = executions_by_time.get(&signal.timestamp_ms) {
            for execution_candle in execution_candles {
                simulator.process_candle(execution_candle)?;
            }
        }
        let resting_ids: Vec<String> = simulator
            .open_orders()
            .iter()
            .map(|resting| resting.order.order_id.clone())
            .collect();
        for order_id in resting_ids {
            simulator.cancel_order(&order_id)?;
        }
        let signal_second_end_ms = signal.timestamp_ms.saturating_add(1_000);
        apply_price_grid_changes_before(
            &mut simulator,
            &mut current_market,
            &price_grid_changes,
            &mut next_price_grid_change,
            signal_second_end_ms,
        )?;
        update_risk_extrema(
            &simulator,
            &mut max_paired_qty,
            &mut max_abs_residual_qty,
            &mut worst_case_settlement_equity_min,
        )?;

        state.update(
            signal.close,
            input.market.payout_unit,
            &input.strategy_params,
        )?;
        let ledger = simulator.ledger();
        let inventory = OutcomeInventorySnapshot {
            yes_qty: ledger.yes_qty(),
            no_qty: ledger.no_qty(),
            yes_available_qty: Some(simulator.available_inventory(Outcome::Yes)),
            no_available_qty: Some(simulator.available_inventory(Outcome::No)),
            yes_average_cost: average_cost(ledger.yes_cost(), ledger.yes_qty()),
            no_average_cost: average_cost(ledger.no_cost(), ledger.no_qty()),
            free_collateral: simulator.available_collateral(),
        };
        let quotes = state.quote(
            signal_second_end_ms,
            signal.close,
            &current_market,
            &input.strategy_params,
            &inventory,
        )?;
        for intent in [quotes.canonical_bid, quotes.canonical_ask]
            .into_iter()
            .flatten()
        {
            next_order_sequence = next_order_sequence.saturating_add(1);
            simulator.place_order(
                limit_order_from_intent(intent, format!("ema-outcome-{next_order_sequence}")),
                signal_second_end_ms,
            )?;
            orders_placed_count += 1;
        }
        let signal_second_end_ms = signal.timestamp_ms.saturating_add(1_000);
        accumulate_inventory_time(
            &simulator,
            signal.timestamp_ms,
            signal_second_end_ms,
            &mut residual_qty_time_area_ms,
            &mut total_inventory_time_area_ms,
        );
        last_inventory_time_ms = signal_second_end_ms;
    }
    accumulate_inventory_time(
        &simulator,
        last_inventory_time_ms,
        input.settlement_time_ms,
        &mut residual_qty_time_area_ms,
        &mut total_inventory_time_area_ms,
    );
    update_risk_extrema(
        &simulator,
        &mut max_paired_qty,
        &mut max_abs_residual_qty,
        &mut worst_case_settlement_equity_min,
    )?;

    let pre_settlement_worst_case_equity = simulator.worst_case_settlement_equity()?;
    let ledger = simulator.ledger();
    let paired_locked_pnl = locked_pair_pnl(
        input.market.payout_unit,
        ledger.yes_qty(),
        ledger.yes_cost(),
        ledger.no_qty(),
        ledger.no_cost(),
    );
    let pre_settlement_trading_pnl = ledger.realized_trading_pnl();
    let pre_settlement_merge_pnl = ledger.realized_merge_pnl();
    let pre_settlement_yes_qty = ledger.yes_qty();
    let pre_settlement_no_qty = ledger.no_qty();
    let pre_settlement_yes_cost = ledger.yes_cost();
    let pre_settlement_no_cost = ledger.no_cost();
    let pre_settlement_paired_qty = ledger.paired_qty();
    let pre_settlement_net_yes_exposure = ledger.net_yes_exposure();
    let fills = simulator.fills().to_vec();
    let (cumulative_yes_buy_qty, cumulative_no_buy_qty) = cumulative_buy_qty(&fills);
    let pair_completion_ratio = completion_ratio(cumulative_yes_buy_qty, cumulative_no_buy_qty);
    let lifecycle_duration_ms = (input.settlement_time_ms - input.market.trading_opens_ms) as f64;
    let traded_notional = fills
        .iter()
        .map(|fill| fill.fill.price * fill.fill.qty)
        .sum();
    let settlement = simulator.settle(input.settlement_time_ms, input.yes_fraction)?;
    let ending_collateral = simulator.ledger().collateral();
    let fees_paid = simulator.ledger().fees_paid();
    let rebates_earned = simulator.ledger().rebates_earned();

    Ok(SingleOutcomeBacktestOutput {
        strategy_kind: "ema_anchor_outcome".to_string(),
        market_id: input.market.market_id.clone(),
        trading_open_time_ms: input.market.trading_opens_ms,
        settlement_time_ms: input.settlement_time_ms,
        starting_collateral: input.starting_collateral,
        ending_collateral,
        orders_placed_count,
        fills_count: fills.len(),
        maker_fills_count: fills.len(),
        traded_notional,
        fees_paid,
        rebates_earned,
        gross_spread_pnl: pre_settlement_trading_pnl + pre_settlement_merge_pnl + paired_locked_pnl,
        settlement_pnl: settlement.realized_settlement_pnl - paired_locked_pnl,
        pre_settlement_yes_qty,
        pre_settlement_no_qty,
        pre_settlement_yes_cost,
        pre_settlement_no_cost,
        pre_settlement_paired_qty,
        pre_settlement_net_yes_exposure,
        pre_settlement_worst_case_equity,
        max_paired_qty,
        max_abs_residual_qty,
        cumulative_yes_buy_qty,
        cumulative_no_buy_qty,
        pair_completion_ratio,
        time_weighted_abs_residual_qty: residual_qty_time_area_ms / lifecycle_duration_ms,
        time_weighted_total_inventory_qty: total_inventory_time_area_ms / lifecycle_duration_ms,
        worst_case_settlement_equity_min,
        fills,
        settlement,
        fill_model: "trade_derived_1s_strict_cross_no_volume_cap".to_string(),
    })
}

fn limit_order_from_intent(
    intent: OutcomeNativeOrderIntent,
    order_id: String,
) -> OutcomeLimitOrder {
    OutcomeLimitOrder {
        order_id,
        outcome: intent.outcome,
        side: intent.side,
        price: intent.native_price,
        qty: intent.qty,
        post_only: true,
        expires_at_ms: None,
    }
}

fn average_cost(cost: f64, qty: f64) -> f64 {
    if qty > 1e-12 {
        cost / qty
    } else {
        0.0
    }
}

fn locked_pair_pnl(
    payout_unit: f64,
    yes_qty: f64,
    yes_cost: f64,
    no_qty: f64,
    no_cost: f64,
) -> f64 {
    let paired_qty = yes_qty.min(no_qty);
    paired_qty * (payout_unit - average_cost(yes_cost, yes_qty) - average_cost(no_cost, no_qty))
}

fn update_risk_extrema(
    simulator: &SingleOutcomeSimulator,
    max_paired_qty: &mut f64,
    max_abs_residual_qty: &mut f64,
    worst_case_settlement_equity_min: &mut f64,
) -> Result<(), OutcomeError> {
    *max_paired_qty = max_paired_qty.max(simulator.ledger().paired_qty());
    *max_abs_residual_qty = max_abs_residual_qty.max(simulator.ledger().net_yes_exposure().abs());
    *worst_case_settlement_equity_min =
        worst_case_settlement_equity_min.min(simulator.worst_case_settlement_equity()?);
    Ok(())
}

fn accumulate_inventory_time(
    simulator: &SingleOutcomeSimulator,
    start_ms: u64,
    end_ms: u64,
    residual_qty_time_area_ms: &mut f64,
    total_inventory_time_area_ms: &mut f64,
) {
    let duration_ms = end_ms.saturating_sub(start_ms) as f64;
    *residual_qty_time_area_ms += simulator.ledger().net_yes_exposure().abs() * duration_ms;
    *total_inventory_time_area_ms +=
        (simulator.ledger().yes_qty() + simulator.ledger().no_qty()) * duration_ms;
}

fn cumulative_buy_qty(fills: &[SimulatedOutcomeFill]) -> (f64, f64) {
    let mut yes = 0.0;
    let mut no = 0.0;
    for simulated in fills {
        if simulated.fill.side != OutcomeOrderSide::Buy {
            continue;
        }
        match simulated.fill.outcome {
            Outcome::Yes => yes += simulated.fill.qty,
            Outcome::No => no += simulated.fill.qty,
        }
    }
    (yes, no)
}

fn completion_ratio(yes_buy_qty: f64, no_buy_qty: f64) -> f64 {
    let larger = yes_buy_qty.max(no_buy_qty);
    if larger > 0.0 {
        yes_buy_qty.min(no_buy_qty) / larger
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::simulator::OutcomeCandle;
    use crate::outcome::strategy::{OutcomeEmaAnchorExecutionMode, OutcomeEmaAnchorParams};
    use crate::outcome::{
        Outcome, OutcomeFeeFormula, OutcomeOrderSide, OutcomePriceGrid, OutcomeVenueCapabilities,
    };

    fn fixture_market() -> BinaryOutcomeMarketSpec {
        BinaryOutcomeMarketSpec {
            venue: "fixture".to_string(),
            market_id: "binary-1".to_string(),
            yes_asset_id: "yes".to_string(),
            no_asset_id: "no".to_string(),
            payout_unit: 1.0,
            min_price: 0.001,
            max_price: 0.999,
            price_grid: OutcomePriceGrid::FixedStep { step: 0.001 },
            qty_step: 0.1,
            min_qty: 0.1,
            min_notional: 0.0,
            trading_opens_ms: 1_000,
            order_entry_opens_ms: 1_000,
            trading_closes_ms: 5_000,
            scheduled_event_ms: 5_000,
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

    fn order(id: &str, outcome: Outcome, price: f64) -> OutcomeLimitOrder {
        OutcomeLimitOrder {
            order_id: id.to_string(),
            outcome,
            side: OutcomeOrderSide::Buy,
            price,
            qty: 1.0,
            post_only: true,
            expires_at_ms: None,
        }
    }

    fn outcome_strategy_params() -> OutcomeEmaAnchorParams {
        OutcomeEmaAnchorParams {
            ema_span_fast_seconds: 2.0,
            ema_span_slow_seconds: 4.0,
            ema_warmup_seconds: 0,
            quote_offset: 0.01,
            inventory_skew: 0.0,
            clip_qty: 1.0,
            max_total_inventory_qty: 10.0,
            max_abs_residual_qty: 5.0,
            min_locked_pair_edge: 0.005,
            estimated_fee_per_share: 0.0,
            risk_reduction_only_ms_before_close: 0,
            entry_cutoff_ms_before_close: 0,
            execution_mode: OutcomeEmaAnchorExecutionMode::AccumulatePairs,
        }
    }

    #[test]
    fn explicit_single_market_run_fills_both_books_and_settles() {
        let output = run_single_outcome_backtest(&SingleOutcomeBacktestInput {
            market: fixture_market(),
            fee_schedule: OutcomeFeeSchedule {
                maker_rate: 0.01,
                taker_rate: 0.02,
                formula: OutcomeFeeFormula::Notional,
                incidence: crate::outcome::OutcomeFeeIncidence::EveryFill,
                settlement_rate: 0.0,
            },
            starting_collateral: 10.0,
            actions: vec![
                OutcomeBacktestAction::PlaceOrder {
                    timestamp_ms: 1_500,
                    order: order("yes", Outcome::Yes, 0.4),
                },
                OutcomeBacktestAction::PlaceOrder {
                    timestamp_ms: 1_500,
                    order: order("no", Outcome::No, 0.5),
                },
            ],
            candles: vec![
                OutcomeCandle {
                    timestamp_ms: 2_000,
                    outcome: Outcome::Yes,
                    open: 0.42,
                    high: 0.45,
                    low: 0.399,
                    close: 0.42,
                    volume: 0.1,
                },
                OutcomeCandle {
                    timestamp_ms: 2_000,
                    outcome: Outcome::No,
                    open: 0.48,
                    high: 0.501,
                    low: 0.45,
                    close: 0.48,
                    volume: 0.1,
                },
            ],
            price_grid_changes: vec![],
            settlement_time_ms: 5_000,
            yes_fraction: 1.0,
        })
        .unwrap();

        assert_eq!(output.fills_count, 2);
        assert_eq!(
            output.fill_model,
            "trade_derived_1s_strict_cross_no_volume_cap"
        );
        assert!((output.ending_collateral - 10.091).abs() < 1e-12);
        assert!((output.fees_paid - 0.009).abs() < 1e-12);
        assert!((output.max_paired_qty - 1.0).abs() < 1e-12);
        assert!((output.max_abs_residual_qty - 1.0).abs() < 1e-12);
        assert!((output.cumulative_yes_buy_qty - 1.0).abs() < 1e-12);
        assert!((output.cumulative_no_buy_qty - 1.0).abs() < 1e-12);
        assert!((output.pair_completion_ratio - 1.0).abs() < 1e-12);
        assert!(output.time_weighted_abs_residual_qty.abs() < 1e-12);
        assert!((output.time_weighted_total_inventory_qty - 1.5).abs() < 1e-12);
    }

    #[test]
    fn zero_volume_and_exact_touch_do_not_fill_in_full_runner() {
        let output = run_single_outcome_backtest(&SingleOutcomeBacktestInput {
            market: fixture_market(),
            fee_schedule: OutcomeFeeSchedule::zero(),
            starting_collateral: 10.0,
            actions: vec![OutcomeBacktestAction::PlaceOrder {
                timestamp_ms: 1_500,
                order: order("yes", Outcome::Yes, 0.4),
            }],
            candles: vec![
                OutcomeCandle {
                    timestamp_ms: 2_000,
                    outcome: Outcome::Yes,
                    open: 0.45,
                    high: 0.5,
                    low: 0.39,
                    close: 0.45,
                    volume: 0.0,
                },
                OutcomeCandle {
                    timestamp_ms: 3_000,
                    outcome: Outcome::Yes,
                    open: 0.45,
                    high: 0.5,
                    low: 0.4,
                    close: 0.45,
                    volume: 1.0,
                },
            ],
            price_grid_changes: vec![],
            settlement_time_ms: 5_000,
            yes_fraction: 0.0,
        })
        .unwrap();
        assert_eq!(output.fills_count, 0);
        assert_eq!(output.ending_collateral, 10.0);
    }

    #[test]
    fn price_grid_changes_are_replayed_before_later_orders() {
        let mut market = fixture_market();
        market.price_grid = OutcomePriceGrid::FixedStep { step: 0.01 };
        let output = run_single_outcome_backtest(&SingleOutcomeBacktestInput {
            market,
            fee_schedule: OutcomeFeeSchedule::zero(),
            starting_collateral: 10.0,
            actions: vec![OutcomeBacktestAction::PlaceOrder {
                timestamp_ms: 1_600,
                order: order("yes", Outcome::Yes, 0.495),
            }],
            candles: vec![OutcomeCandle {
                timestamp_ms: 2_000,
                outcome: Outcome::Yes,
                open: 0.5,
                high: 0.5,
                low: 0.494,
                close: 0.5,
                volume: 1.0,
            }],
            price_grid_changes: vec![OutcomePriceGridChange {
                timestamp_ms: 1_500,
                old_grid: OutcomePriceGrid::FixedStep { step: 0.01 },
                new_grid: OutcomePriceGrid::FixedStep { step: 0.001 },
            }],
            settlement_time_ms: 5_000,
            yes_fraction: 1.0,
        })
        .unwrap();

        assert_eq!(output.fills_count, 1);
    }

    #[test]
    fn ema_anchor_outcome_uses_current_signal_to_quote_for_next_second() {
        let output = run_outcome_ema_anchor_backtest(&OutcomeEmaAnchorBacktestInput {
            market: fixture_market(),
            fee_schedule: OutcomeFeeSchedule::zero(),
            starting_collateral: 10.0,
            strategy_params: outcome_strategy_params(),
            signal_candles: (1..5)
                .map(|second| OutcomeSignalCandle {
                    timestamp_ms: second * 1_000,
                    open: 0.5,
                    high: 0.5,
                    low: 0.5,
                    close: 0.5,
                    volume: if second == 1 { 1.0 } else { 0.0 },
                })
                .collect(),
            execution_candles: vec![
                OutcomeCandle {
                    timestamp_ms: 2_000,
                    outcome: Outcome::Yes,
                    open: 0.5,
                    high: 0.5,
                    low: 0.489,
                    close: 0.5,
                    volume: 0.01,
                },
                OutcomeCandle {
                    timestamp_ms: 2_000,
                    outcome: Outcome::No,
                    open: 0.5,
                    high: 0.511,
                    low: 0.5,
                    close: 0.5,
                    volume: 0.01,
                },
            ],
            price_grid_changes: vec![],
            settlement_time_ms: 5_000,
            yes_fraction: 1.0,
        })
        .unwrap();

        assert_eq!(output.strategy_kind, "ema_anchor_outcome");
        assert_eq!(output.fills_count, 2);
        assert!((output.ending_collateral - 10.02).abs() < 1e-12);
        assert!((output.gross_spread_pnl - 0.02).abs() < 1e-12);
        assert!(output.settlement_pnl.abs() < 1e-12);
        assert_eq!(output.max_paired_qty, 1.0);
        assert_eq!(output.pair_completion_ratio, 1.0);
        assert!(output.time_weighted_abs_residual_qty.abs() < 1e-12);
        assert!((output.time_weighted_total_inventory_qty - 1.5).abs() < 1e-12);
    }

    #[test]
    fn ema_grid_change_at_bucket_start_precedes_same_second_fills() {
        let mut market = fixture_market();
        market.price_grid = OutcomePriceGrid::FixedStep { step: 0.001 };
        let mut params = outcome_strategy_params();
        params.quote_offset = 0.005;
        let output = run_outcome_ema_anchor_backtest(&OutcomeEmaAnchorBacktestInput {
            market,
            fee_schedule: OutcomeFeeSchedule::zero(),
            starting_collateral: 10.0,
            strategy_params: params,
            signal_candles: (1..5)
                .map(|second| OutcomeSignalCandle {
                    timestamp_ms: second * 1_000,
                    open: 0.5,
                    high: 0.5,
                    low: 0.5,
                    close: 0.5,
                    volume: if second == 1 { 1.0 } else { 0.0 },
                })
                .collect(),
            execution_candles: vec![OutcomeCandle {
                timestamp_ms: 2_000,
                outcome: Outcome::Yes,
                open: 0.5,
                high: 0.5,
                low: 0.494,
                close: 0.5,
                volume: 1.0,
            }],
            price_grid_changes: vec![OutcomePriceGridChange {
                timestamp_ms: 2_000,
                old_grid: OutcomePriceGrid::FixedStep { step: 0.001 },
                new_grid: OutcomePriceGrid::FixedStep { step: 0.01 },
            }],
            settlement_time_ms: 5_000,
            yes_fraction: 1.0,
        })
        .unwrap();

        assert_eq!(output.fills_count, 0);
    }

    #[test]
    fn ema_anchor_outcome_rejects_unknown_signal_gaps() {
        let result = run_outcome_ema_anchor_backtest(&OutcomeEmaAnchorBacktestInput {
            market: fixture_market(),
            fee_schedule: OutcomeFeeSchedule::zero(),
            starting_collateral: 10.0,
            strategy_params: outcome_strategy_params(),
            signal_candles: vec![
                OutcomeSignalCandle {
                    timestamp_ms: 1_000,
                    open: 0.5,
                    high: 0.5,
                    low: 0.5,
                    close: 0.5,
                    volume: 1.0,
                },
                OutcomeSignalCandle {
                    timestamp_ms: 3_000,
                    open: 0.5,
                    high: 0.5,
                    low: 0.5,
                    close: 0.5,
                    volume: 1.0,
                },
            ],
            execution_candles: vec![],
            price_grid_changes: vec![],
            settlement_time_ms: 5_000,
            yes_fraction: 1.0,
        });
        assert!(matches!(result, Err(OutcomeError::InvalidMarket(_))));
    }
}
