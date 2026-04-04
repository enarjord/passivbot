//! Single-source order orchestration.
//!
//! This module centralises the sequencing currently duplicated between backtester and live bot.
//! It computes the complete set of desired orders for all symbols/sides given a deterministic
//! snapshot of state and configuration. No defaults, no silent fallbacks: callers must provide
//! fully-populated inputs.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Backtest-only performance hint: allow next-only vs full-grid expansion on a per-symbol basis.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryPeekHints {
    pub expand_grid_long: HashSet<usize>,
    pub expand_grid_short: HashSet<usize>,
    pub expand_close_long: HashSet<usize>,
    pub expand_close_short: HashSet<usize>,
}

mod core {
    use crate::coin_selection::{
        select_forager_candidates, ForagerCandidate, ForagerPositionSide, ForagerSelectionConfig,
        ForagerSelectionError,
    };
    use crate::constants::{LONG, SHORT};
    use crate::entries::calc_min_entry_qty;
    use crate::risk::{
        calc_twel_enforcer_actions, calc_unstucking_action, GateEntriesPosition,
        TwelEnforcerInputPosition, UnstuckPositionInput,
    };
    use crate::strategies::{
        generate_orders as generate_strategy_orders, parse_strategy_params, strategy_ema_spans,
        strategy_entry_volatility_span_hours, strategy_initial_entry_offset, NextStepHint,
        PeekBehavior, StrategyKind, StrategyRequest, StrategySide,
    };
    use crate::types::{
        BotParams, BotParamsPair, EMABands, ExchangeParams, OrderBook, OrderType, Position,
        RuntimeBudgetState, StateParams, TrailingPriceBundle,
    };
    use crate::utils::{
        calc_new_psize_pprice, calc_order_price_diff_ask, calc_order_price_diff_bid, calc_pnl_long,
        calc_pnl_short, calc_pside_price_diff_int, calc_wallet_exposure, round_, round_dn,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum PositionSide {
        Long,
        Short,
    }

    impl PositionSide {
        #[inline]
        pub const fn to_pside_int(self) -> usize {
            match self {
                PositionSide::Long => LONG,
                PositionSide::Short => SHORT,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum TradingMode {
        Normal,
        Panic,
        GracefulStop,
        TpOnly,
        Manual,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ExecutionType {
        Limit,
        Market,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct IdealOrder {
        pub symbol_idx: usize,
        pub pside: PositionSide,
        /// Signed: buy qty > 0, sell qty < 0
        pub qty: f64,
        pub price: f64,
        pub order_type: OrderType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ExecutableOrder {
        pub symbol_idx: usize,
        pub pside: PositionSide,
        pub qty: f64,
        pub price: f64,
        pub order_type: OrderType,
        pub execution_type: ExecutionType,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case", deny_unknown_fields)]
    pub enum OrchestratorWarning {
        DisabledPsideHasPosition {
            symbol_idx: usize,
            pside: PositionSide,
        },
        NonTradableHasPosition {
            symbol_idx: usize,
            pside: PositionSide,
        },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct LossGateBlock {
        pub symbol_idx: usize,
        pub pside: PositionSide,
        pub order_type: OrderType,
        pub qty: f64,
        pub price: f64,
        pub projected_pnl: f64,
        pub balance_before: f64,
        pub projected_balance_after: f64,
        pub balance_peak: f64,
        pub balance_floor: f64,
        pub max_realized_loss_pct: f64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case", deny_unknown_fields)]
    pub enum OrchestratorError {
        InvalidOrderBook {
            symbol_idx: usize,
        },
        /// `symbols` must be indexed by `symbol_idx` (i.e. `symbols[i].symbol_idx == i`).
        NonContiguousSymbolIdx {
            pos: usize,
            symbol_idx: usize,
        },
        NonFiniteInput {
            field: &'static str,
            symbol_idx: Option<usize>,
        },
        MissingEma {
            symbol_idx: usize,
        },
    }

    #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct OrchestratorDiagnostics {
        pub warnings: Vec<OrchestratorWarning>,
        #[serde(default)]
        pub loss_gate_blocks: Vec<LossGateBlock>,
        #[serde(default)]
        pub symbol_states: Vec<SymbolStateDiagnostic>,
    }

    #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct OrchestratorOutput {
        pub orders: Vec<ExecutableOrder>,
        pub diagnostics: OrchestratorDiagnostics,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct SymbolSideStateDiagnostic {
        pub input_mode: Option<TradingMode>,
        pub effective_mode: TradingMode,
        pub active: bool,
        pub allow_initial: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct SymbolStateDiagnostic {
        pub symbol_idx: usize,
        pub long: SymbolSideStateDiagnostic,
        pub short: SymbolSideStateDiagnostic,
    }

    /// EMA values keyed by span (same numeric spans as used in config/bot params).
    ///
    /// Kept as a small vector to avoid float keys requiring `Ord`.
    pub type EmaBySpan = Vec<(f64, f64)>;

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct EmaTimeframeBundle {
        pub close: EmaBySpan,
        pub log_range: EmaBySpan,
        pub volume: EmaBySpan,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct EmaBundle {
        pub m1: EmaTimeframeBundle,
        pub h1: EmaTimeframeBundle,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct OrchestratorGlobal {
        pub filter_by_min_effective_cost: bool,
        #[serde(default)]
        pub market_orders_allowed: bool,
        #[serde(default = "default_market_order_near_touch_threshold")]
        pub market_order_near_touch_threshold: f64,
        #[serde(default)]
        pub panic_close_market: bool,
        pub unstuck_allowance_long: f64,
        pub unstuck_allowance_short: f64,
        /// Fraction of peak balance that may be realized as drawdown before lossy closes are blocked.
        /// <=0 blocks all lossy closes; >=1 disables gating.
        #[serde(default = "default_max_realized_loss_pct")]
        pub max_realized_loss_pct: f64,
        /// Gross realized pnl cumsum peak from fill history (statelessly reconstructed).
        #[serde(default)]
        pub realized_pnl_cumsum_max: f64,
        /// Gross realized pnl cumsum current value from fill history.
        #[serde(default)]
        pub realized_pnl_cumsum_last: f64,
        /// If true, output orders are globally sorted by the canonical (live-bot) distance metric.
        /// Backtest does not require this global ordering and may disable it for performance.
        pub sort_global: bool,
        /// Global bot params (not modifiable by per-coin overrides).
        pub global_bot_params: BotParamsPair,
        /// If false (one-way mode), only one position side can exist per coin at a time.
        /// When no position exists on either side, the side closer to its EMA entry band wins.
        #[serde(default = "default_hedge_mode")]
        pub hedge_mode: bool,
        #[serde(default)]
        pub strategy_kind: StrategyKind,
    }

    fn default_hedge_mode() -> bool {
        true
    }

    fn default_max_realized_loss_pct() -> f64 {
        1.0
    }

    fn default_market_order_near_touch_threshold() -> f64 {
        0.001
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct SymbolSideInput {
        /// `None` means “forager-eligible default”.
        /// `Some(Normal)` means “forced Normal” (always selected).
        pub mode: Option<TradingMode>,
        pub position: Position,
        pub trailing: TrailingPriceBundle,
        /// Per-symbol/per-pside params after applying coin_overrides.
        pub bot_params: BotParams,
        #[serde(default)]
        pub strategy_params: Option<Value>,
        #[serde(default)]
        pub runtime_budget: Option<RuntimeBudgetState>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct NextCandle {
        pub low: f64,
        pub high: f64,
        pub tradable: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct SymbolInput {
        pub symbol_idx: usize,
        pub order_book: OrderBook, // must have bid>0 and ask>0
        pub exchange: ExchangeParams,
        pub tradable: bool,
        /// Backtest-only hint: next candle range for "peek fill" decisions.
        /// `None` => unknown (live mode), default to full-grid expansion.
        pub next_candle: Option<NextCandle>,
        pub effective_min_cost: f64,
        pub emas: EmaBundle,
        pub long: SymbolSideInput,
        pub short: SymbolSideInput,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct OrchestratorInput {
        /// Hysteresis-snapped balance used for sizing/order-shaping logic.
        pub balance: f64,
        /// True/raw balance used for risk/accounting gates.
        #[serde(default = "default_balance_raw")]
        pub balance_raw: f64,
        pub global: OrchestratorGlobal,
        pub symbols: Vec<SymbolInput>,
        /// Backtest-only performance hint: allow next-only vs full-grid expansion.
        pub peek_hints: Option<super::EntryPeekHints>,
    }

    fn default_balance_raw() -> f64 {
        f64::NAN
    }

    fn input_balance_raw(input: &OrchestratorInput) -> f64 {
        if input.balance_raw.is_finite() {
            input.balance_raw
        } else {
            input.balance
        }
    }

    pub fn is_close_order_type(order_type: OrderType) -> bool {
        use OrderType::*;
        matches!(
            order_type,
            CloseGridLong
                | CloseTrailingLong
                | CloseUnstuckLong
                | CloseAutoReduceTwelLong
                | ClosePanicLong
                | CloseAutoReduceWelLong
                | CloseEmaAnchorLong
                | CloseGridShort
                | CloseTrailingShort
                | CloseUnstuckShort
                | CloseAutoReduceTwelShort
                | ClosePanicShort
                | CloseAutoReduceWelShort
                | CloseEmaAnchorShort
        )
    }

    fn is_panic_close_order_type(order_type: OrderType) -> bool {
        matches!(
            order_type,
            OrderType::ClosePanicLong | OrderType::ClosePanicShort
        )
    }

    fn current_market_price(order_book: &OrderBook) -> f64 {
        if order_book.bid.is_finite()
            && order_book.ask.is_finite()
            && order_book.bid > 0.0
            && order_book.ask > 0.0
        {
            (order_book.bid + order_book.ask) * 0.5
        } else {
            order_book.bid.max(order_book.ask)
        }
    }

    fn should_use_market_execution(
        order: &IdealOrder,
        global: &OrchestratorGlobal,
        order_book: &OrderBook,
    ) -> bool {
        if is_panic_close_order_type(order.order_type) {
            let pside_market = match order.pside {
                PositionSide::Long => {
                    global.global_bot_params.long.hsl_enabled
                        && global.global_bot_params.long.hsl_panic_close_order_type == "market"
                }
                PositionSide::Short => {
                    global.global_bot_params.short.hsl_enabled
                        && global.global_bot_params.short.hsl_panic_close_order_type == "market"
                }
            };
            return pside_market || global.panic_close_market;
        }
        if !global.market_orders_allowed {
            return false;
        }
        let market_price = current_market_price(order_book);
        if !market_price.is_finite() || market_price <= 0.0 {
            return false;
        }
        if order.qty > 0.0 && order.price >= market_price {
            return true;
        }
        if order.qty < 0.0 && order.price <= market_price {
            return true;
        }
        let diff = if order.qty > 0.0 {
            calc_order_price_diff_bid(order.price, market_price).abs()
        } else {
            calc_order_price_diff_ask(order.price, market_price).abs()
        };
        diff <= global.market_order_near_touch_threshold.max(0.0)
    }

    fn to_executable_order(
        order: IdealOrder,
        global: &OrchestratorGlobal,
        order_book: &OrderBook,
    ) -> ExecutableOrder {
        let execution_type = if should_use_market_execution(&order, global, order_book) {
            ExecutionType::Market
        } else {
            ExecutionType::Limit
        };
        ExecutableOrder {
            symbol_idx: order.symbol_idx,
            pside: order.pside,
            qty: order.qty,
            price: order.price,
            order_type: order.order_type,
            execution_type,
        }
    }

    fn is_pside_enabled(global: &BotParamsPair, pside: PositionSide) -> bool {
        let bp = match pside {
            PositionSide::Long => &global.long,
            PositionSide::Short => &global.short,
        };
        bp.total_wallet_exposure_limit > 0.0 && bp.n_positions > 0
    }

    fn ema_lookup(map: &EmaBySpan, span: f64) -> Option<f64> {
        if !span.is_finite() {
            return None;
        }
        for (k, v) in map.iter().copied() {
            let diff = (k - span).abs();
            let allowed = 1e-9_f64.max(span.abs() * 1e-12);
            if diff <= allowed {
                return Some(v);
            }
        }
        None
    }

    fn derive_ema_bands(
        symbol_idx: usize,
        emas: &EmaBundle,
        strategy_params: &crate::strategies::StrategyParams,
    ) -> Result<EMABands, OrchestratorError> {
        let (ema_span_0, ema_span_1) = strategy_ema_spans(strategy_params);
        let ema0 = ema_lookup(&emas.m1.close, ema_span_0)
            .ok_or(OrchestratorError::MissingEma { symbol_idx })?;
        let ema1 = ema_lookup(&emas.m1.close, ema_span_1)
            .ok_or(OrchestratorError::MissingEma { symbol_idx })?;
        let ema2_span = (ema_span_0 * ema_span_1).sqrt();
        let ema2 = ema_lookup(&emas.m1.close, ema2_span)
            .ok_or(OrchestratorError::MissingEma { symbol_idx })?;
        let lower = ema0.min(ema1).min(ema2);
        let upper = ema0.max(ema1).max(ema2);
        if !(lower.is_finite() && upper.is_finite() && lower > 0.0 && upper > 0.0) {
            return Err(OrchestratorError::NonFiniteInput {
                field: "ema_bands",
                symbol_idx: Some(symbol_idx),
            });
        }
        Ok(EMABands { upper, lower })
    }

    fn derive_entry_volatility_logrange_ema_1h(
        symbol_idx: usize,
        emas: &EmaBundle,
        strategy_params: &crate::strategies::StrategyParams,
    ) -> Result<f64, OrchestratorError> {
        let Some(span) = strategy_entry_volatility_span_hours(strategy_params) else {
            return Ok(0.0);
        };
        if span <= 0.0 {
            return Ok(0.0);
        }
        let v = ema_lookup(&emas.h1.log_range, span)
            .ok_or(OrchestratorError::MissingEma { symbol_idx })?;
        if !(v.is_finite() && v >= 0.0) {
            return Err(OrchestratorError::NonFiniteInput {
                field: "entry_volatility_logrange_ema_1h",
                symbol_idx: Some(symbol_idx),
            });
        }
        Ok(v)
    }

    fn map_forager_selection_error(err: ForagerSelectionError) -> OrchestratorError {
        match err {
            ForagerSelectionError::InvalidPositionSide(_) => OrchestratorError::NonFiniteInput {
                field: "forager_position_side",
                symbol_idx: None,
            },
            ForagerSelectionError::NonFiniteInput { field, index } => {
                OrchestratorError::NonFiniteInput {
                    field,
                    symbol_idx: Some(index),
                }
            }
        }
    }

    fn require_forager_input(
        symbol_idx: usize,
        field: &'static str,
        value: f64,
    ) -> Result<f64, OrchestratorError> {
        if !value.is_finite() {
            return Err(OrchestratorError::NonFiniteInput {
                field,
                symbol_idx: Some(symbol_idx),
            });
        }
        Ok(value)
    }

    fn fallback_configured_wallet_exposure_limit(bot: &BotParams) -> f64 {
        if bot.wallet_exposure_limit > 0.0 {
            bot.wallet_exposure_limit
        } else if bot.n_positions > 0 {
            bot.total_wallet_exposure_limit / bot.n_positions as f64
        } else {
            0.0
        }
    }

    fn resolve_runtime_budget(
        side: &SymbolSideInput,
        effective_n_positions: usize,
    ) -> RuntimeBudgetState {
        if let Some(runtime_budget) = side.runtime_budget.clone() {
            return runtime_budget;
        }
        let configured_wallet_exposure_limit =
            fallback_configured_wallet_exposure_limit(&side.bot_params);
        RuntimeBudgetState {
            configured_wallet_exposure_limit,
            effective_wallet_exposure_limit: configured_wallet_exposure_limit,
            configured_n_positions: side.bot_params.n_positions,
            effective_n_positions,
        }
    }

    fn effective_min_cost_is_low_enough(
        balance: f64,
        filter_enabled: bool,
        effective_min_cost: f64,
        bot: &BotParams,
        runtime_budget: &RuntimeBudgetState,
    ) -> bool {
        if !filter_enabled {
            return true;
        }
        if !(balance.is_finite()
            && balance > 0.0
            && effective_min_cost.is_finite()
            && effective_min_cost > 0.0)
        {
            return false;
        }
        let base_limit = runtime_budget.effective_wallet_exposure_limit;
        let allowance_pct = bot.risk_we_excess_allowance_pct;
        let allowance_multiplier = 1.0 + allowance_pct.max(0.0);
        let effective_limit = base_limit * allowance_multiplier;
        if !(effective_limit.is_finite() && effective_limit > 0.0) {
            return false;
        }
        let req = balance * effective_limit * bot.entry_initial_qty_pct;
        req >= effective_min_cost
    }

    fn market_price_for_order_side(ob: &OrderBook, qty: f64) -> f64 {
        if qty >= 0.0 {
            ob.bid
        } else {
            ob.ask
        }
    }

    fn order_price_diff_strict(order: &IdealOrder, ob: &OrderBook) -> f64 {
        let market_price = market_price_for_order_side(ob, order.qty);
        if order.qty >= 0.0 {
            calc_order_price_diff_bid(order.price, market_price)
        } else {
            calc_order_price_diff_ask(order.price, market_price)
        }
    }

    fn canonical_sort_key(
        order: &IdealOrder,
        ob: &OrderBook,
    ) -> (f64, usize, u8, u8, u16, f64, f64) {
        let diff = order_price_diff_strict(order, ob);
        let pside_rank = match order.pside {
            PositionSide::Long => 0u8,
            PositionSide::Short => 1u8,
        };
        let is_close_rank = if is_close_order_type(order.order_type) {
            0u8
        } else {
            1u8
        };
        (
            diff,
            order.symbol_idx,
            pside_rank,
            is_close_rank,
            order.order_type.id(),
            order.price,
            order.qty,
        )
    }

    fn trim_closes_to_position(
        pside: PositionSide,
        closes: &mut Vec<IdealOrder>,
        pos_size: f64,
        ob: &OrderBook,
        exchange: &ExchangeParams,
    ) {
        const EPS: f64 = 1e-12;
        if closes.is_empty() {
            return;
        }
        let pos_abs = pos_size.abs();
        if pos_abs <= EPS {
            closes.clear();
            return;
        }
        // Close orders must respect effective min qty/min cost (via `calc_min_entry_qty`),
        // except when the position itself is smaller than effective min qty, in which case we
        // allow closing the full position size.
        let close_side_qty = match pside {
            PositionSide::Long => -1.0,
            PositionSide::Short => 1.0,
        };
        let min_entry_qty =
            calc_min_entry_qty(market_price_for_order_side(ob, close_side_qty), exchange);
        let allow_below_min = pos_abs < min_entry_qty - EPS;

        // Drop any dust close orders up-front.
        if !allow_below_min && min_entry_qty > EPS {
            closes.retain(|o| o.qty.abs() + EPS >= min_entry_qty);
            if closes.is_empty() {
                return;
            }
        }

        // If the position itself is below effective min qty, collapse to a single close which
        // closes the entire position.
        if allow_below_min {
            // pick closest-to-fill existing close if present; otherwise keep first
            closes.sort_by(|a, b| {
                let da = order_price_diff_strict(a, ob);
                let db = order_price_diff_strict(b, ob);
                da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.order_type.id().cmp(&b.order_type.id()))
                    .then_with(|| {
                        a.price
                            .partial_cmp(&b.price)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            let mut keep = closes[0].clone();
            keep.qty = match pside {
                PositionSide::Long => -pos_abs,
                PositionSide::Short => pos_abs,
            };
            closes.clear();
            closes.push(keep);
            return;
        }

        let enforce_no_dust_remainder = |closes: &mut Vec<IdealOrder>| {
            if closes.is_empty() {
                return;
            }
            // Compute remainder after all closes; if remainder is non-zero but smaller than the
            // effective min qty at all close prices, absorb it into the closest-to-fill close.
            let mut total_abs = 0.0_f64;
            let mut min_req = f64::INFINITY;
            for o in closes.iter() {
                let qa = o.qty.abs();
                if qa <= EPS {
                    continue;
                }
                total_abs += qa;
                let req = calc_min_entry_qty(o.price, exchange);
                if req.is_finite() && req > 0.0 {
                    min_req = min_req.min(req);
                }
            }
            if !min_req.is_finite() {
                return;
            }
            let remainder = pos_abs - total_abs;
            if remainder <= EPS {
                return;
            }
            if remainder + EPS >= min_req {
                return;
            }
            // Absorb into furthest-from-fill close (largest diff).
            //
            // Rationale: when we have a tiny remainder after filtering/trimming (smaller than
            // effective min qty), we must not strand an uncloseable dust position. Absorbing the
            // remainder into the least-likely-to-fill close matches the existing “trim furthest
            // first” heuristic and aligns legacy/orchestrator distribution.
            closes.sort_by(|a, b| {
                let da = order_price_diff_strict(a, ob);
                let db = order_price_diff_strict(b, ob);
                da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.order_type.id().cmp(&b.order_type.id()))
                    .then_with(|| {
                        a.price
                            .partial_cmp(&b.price)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            if closes.is_empty() {
                return;
            }
            // Set the selected close to close exactly the remaining position after other closes.
            // After sorting by diff asc, the selected is the last (furthest).
            let mut sum_other = 0.0_f64;
            if closes.len() > 1 {
                for o in closes.iter().take(closes.len() - 1) {
                    sum_other += o.qty.abs();
                }
            }
            let needed = (pos_abs - sum_other).max(0.0);
            let needed_q =
                round_(needed, exchange.qty_step).min(round_(pos_abs, exchange.qty_step));
            let i = closes.len().saturating_sub(1);
            closes[i].qty = match pside {
                PositionSide::Long => -needed_q,
                PositionSide::Short => needed_q,
            };
        };

        // Compute total close abs qty; validate sign consistency.
        let mut total = 0.0_f64;
        for o in closes.iter() {
            match pside {
                PositionSide::Long => {
                    if o.qty >= 0.0 {
                        continue;
                    }
                    total += o.qty.abs();
                }
                PositionSide::Short => {
                    if o.qty <= 0.0 {
                        continue;
                    }
                    total += o.qty.abs();
                }
            }
        }
        if total <= pos_abs + EPS {
            enforce_no_dust_remainder(closes);
            return;
        }

        // Sort furthest-from-fill first.
        closes.sort_by(|a, b| {
            let da = order_price_diff_strict(a, ob);
            let db = order_price_diff_strict(b, ob);
            match db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => a
                    .order_type
                    .id()
                    .cmp(&b.order_type.id())
                    .then_with(|| {
                        a.price
                            .partial_cmp(&b.price)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| {
                        a.qty
                            .partial_cmp(&b.qty)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }),
                other => other,
            }
        });

        let mut excess = total - pos_abs;
        let mut trimmed: Vec<IdealOrder> = Vec::with_capacity(closes.len());
        for mut o in closes.drain(..) {
            if excess <= EPS {
                trimmed.push(o);
                continue;
            }
            let qty_abs = o.qty.abs();
            if qty_abs <= EPS {
                continue;
            }
            if qty_abs <= excess + EPS {
                // drop whole order
                excess -= qty_abs;
                continue;
            }
            let mut new_abs = qty_abs - excess;
            // `qty_abs` and `excess` are intended to be `qty_step`-snapped; using nearest rounding
            // avoids occasional one-step under-trims due to float representation noise.
            new_abs = round_(new_abs, exchange.qty_step);
            if new_abs <= EPS {
                continue;
            }
            if !allow_below_min && new_abs < min_entry_qty {
                // Drop; we prefer removing furthest rather than keeping dust.
                excess = 0.0;
                continue;
            }
            o.qty = if o.qty.is_sign_negative() {
                -new_abs
            } else {
                new_abs
            };
            excess = 0.0;
            trimmed.push(o);
        }
        *closes = trimmed;
        enforce_no_dust_remainder(closes);
    }

    fn projected_close_pnl(
        order: &IdealOrder,
        pos: &Position,
        exchange: &ExchangeParams,
    ) -> Option<f64> {
        const EPS: f64 = 1e-12;
        if !order.price.is_finite() || order.price <= 0.0 {
            return None;
        }
        if !pos.price.is_finite() || pos.price <= 0.0 {
            return None;
        }
        if !exchange.c_mult.is_finite() || exchange.c_mult <= 0.0 {
            return None;
        }
        match order.pside {
            PositionSide::Long => {
                let size_abs = pos.size.max(0.0);
                if size_abs <= EPS || order.qty >= 0.0 {
                    return None;
                }
                let close_qty = order.qty.abs().min(size_abs);
                if close_qty <= EPS {
                    return None;
                }
                Some(calc_pnl_long(
                    pos.price,
                    order.price,
                    close_qty,
                    exchange.c_mult,
                ))
            }
            PositionSide::Short => {
                let size_abs = pos.size.abs();
                if size_abs <= EPS || order.qty <= 0.0 {
                    return None;
                }
                let close_qty = order.qty.min(size_abs);
                if close_qty <= EPS {
                    return None;
                }
                Some(calc_pnl_short(
                    pos.price,
                    order.price,
                    close_qty,
                    exchange.c_mult,
                ))
            }
        }
    }

    fn gate_lossy_closes_by_peak_balance(
        input: &OrchestratorInput,
        per_long: &mut [Option<PerSymbolOrders>],
        per_short: &mut [Option<PerSymbolOrders>],
        diagnostics: &mut OrchestratorDiagnostics,
    ) {
        let max_loss_pct = input.global.max_realized_loss_pct;
        if !max_loss_pct.is_finite() || max_loss_pct >= 1.0 {
            return;
        }
        let pct = max_loss_pct.max(0.0);
        let balance_raw = input_balance_raw(input);
        if !balance_raw.is_finite() || balance_raw <= 0.0 {
            return;
        }
        let pnl_max = input.global.realized_pnl_cumsum_max;
        let pnl_last = input.global.realized_pnl_cumsum_last;
        if !pnl_max.is_finite() || !pnl_last.is_finite() {
            return;
        }
        let balance_peak = balance_raw + (pnl_max - pnl_last);
        if !balance_peak.is_finite() || balance_peak <= 0.0 {
            return;
        }
        let balance_floor = balance_peak * (1.0 - pct);
        if !balance_floor.is_finite() {
            return;
        }

        for s in per_long.iter_mut().filter_map(|v| v.as_mut()) {
            let exchange = match input.symbols.get(s.symbol_idx) {
                Some(sym) => &sym.exchange,
                None => continue,
            };
            let mut kept: Vec<IdealOrder> = Vec::with_capacity(s.closes.len());
            for order in s.closes.drain(..) {
                if !is_close_order_type(order.order_type)
                    || is_panic_close_order_type(order.order_type)
                {
                    kept.push(order);
                    continue;
                }
                let Some(projected_pnl) = projected_close_pnl(&order, &s.pos, exchange) else {
                    kept.push(order);
                    continue;
                };
                let projected_balance_after = balance_raw + projected_pnl;
                if projected_pnl < 0.0 && projected_balance_after < balance_floor - 1e-12 {
                    diagnostics.loss_gate_blocks.push(LossGateBlock {
                        symbol_idx: order.symbol_idx,
                        pside: order.pside,
                        order_type: order.order_type,
                        qty: order.qty,
                        price: order.price,
                        projected_pnl,
                        balance_before: balance_raw,
                        projected_balance_after,
                        balance_peak,
                        balance_floor,
                        max_realized_loss_pct: pct,
                    });
                    continue;
                }
                kept.push(order);
            }
            s.closes = kept;
        }

        for s in per_short.iter_mut().filter_map(|v| v.as_mut()) {
            let exchange = match input.symbols.get(s.symbol_idx) {
                Some(sym) => &sym.exchange,
                None => continue,
            };
            let mut kept: Vec<IdealOrder> = Vec::with_capacity(s.closes.len());
            for order in s.closes.drain(..) {
                if !is_close_order_type(order.order_type)
                    || is_panic_close_order_type(order.order_type)
                {
                    kept.push(order);
                    continue;
                }
                let Some(projected_pnl) = projected_close_pnl(&order, &s.pos, exchange) else {
                    kept.push(order);
                    continue;
                };
                let projected_balance_after = balance_raw + projected_pnl;
                if projected_pnl < 0.0 && projected_balance_after < balance_floor - 1e-12 {
                    diagnostics.loss_gate_blocks.push(LossGateBlock {
                        symbol_idx: order.symbol_idx,
                        pside: order.pside,
                        order_type: order.order_type,
                        qty: order.qty,
                        price: order.price,
                        projected_pnl,
                        balance_before: balance_raw,
                        projected_balance_after,
                        balance_peak,
                        balance_floor,
                        max_realized_loss_pct: pct,
                    });
                    continue;
                }
                kept.push(order);
            }
            s.closes = kept;
        }
    }

    fn compute_effective_n_positions(
        n_positions_cfg: usize,
        eligible_len: usize,
        forced_normals_len: usize,
    ) -> usize {
        let base = n_positions_cfg.min(eligible_len);
        base.max(forced_normals_len)
    }

    fn fill_forced_normal_indices(
        symbols: &[SymbolInput],
        pside: PositionSide,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        for s in symbols {
            let mode = match pside {
                PositionSide::Long => s.long.mode,
                PositionSide::Short => s.short.mode,
            };
            if mode == Some(TradingMode::Normal) {
                out.push(s.symbol_idx);
            }
        }
    }

    fn one_way_allows_initial_slot(
        symbols: &[SymbolInput],
        symbol_idx: usize,
        pside: PositionSide,
        hedge_mode: bool,
    ) -> bool {
        if hedge_mode {
            return true;
        }
        let s = &symbols[symbol_idx];
        match pside {
            PositionSide::Long => s.short.position.size == 0.0,
            PositionSide::Short => s.long.position.size == 0.0,
        }
    }

    fn build_forager_candidates_into(
        symbols: &[SymbolInput],
        pside: PositionSide,
        strategy_kind: StrategyKind,
        hedge_mode: bool,
        filter_enabled: bool,
        balance: f64,
        effective_n_positions: usize,
        active_flags: Option<&[bool]>,
        cfg: &ForagerSelectionConfig,
        out: &mut Vec<ForagerCandidate>,
    ) -> Result<(), OrchestratorError> {
        let volume_required = cfg.volume_drop_pct > 0.0 || cfg.weights.volume != 0.0;
        let volatility_required = cfg.weights.volatility != 0.0;
        let ema_readiness_required = cfg.weights.ema_readiness != 0.0;
        out.clear();
        out.reserve(symbols.len());
        for s in symbols {
            let side = match pside {
                PositionSide::Long => &s.long,
                PositionSide::Short => &s.short,
            };
            let strategy_params = parse_strategy_params(
                strategy_kind,
                side.strategy_params.as_ref(),
                &side.bot_params,
            )
            .map_err(|_| OrchestratorError::NonFiniteInput {
                field: "strategy_params",
                symbol_idx: Some(s.symbol_idx),
            })?;
            // For selection of coins to occupy available slots for initial entries:
            // - We rank across all coins (including those with positions), matching legacy.
            // - We exclude modes which categorically block initial entries when `psize == 0.0`.
            let mode_no_pos = effective_mode(side.mode, false);
            let can_open_initial = should_generate_entries(mode_no_pos, false, true);
            let already_active = active_flags
                .and_then(|flags| flags.get(s.symbol_idx))
                .copied()
                .unwrap_or(false);
            let enabled = s.tradable
                && !already_active
                && one_way_allows_initial_slot(symbols, s.symbol_idx, pside, hedge_mode)
                && can_open_initial
                && effective_min_cost_is_low_enough(
                    balance,
                    filter_enabled,
                    s.effective_min_cost,
                    &side.bot_params,
                    &resolve_runtime_budget(side, effective_n_positions),
                );
            if !enabled {
                out.push(ForagerCandidate {
                    index: s.symbol_idx,
                    enabled: false,
                    volume_score: 0.0,
                    volatility_score: 0.0,
                    bid: 0.0,
                    ask: 0.0,
                    ema_lower: 0.0,
                    ema_upper: 0.0,
                    entry_initial_ema_dist: 0.0,
                });
                continue;
            }
            let volume_score = if volume_required {
                require_forager_input(
                    s.symbol_idx,
                    "forager_volume_score",
                    ema_lookup(&s.emas.m1.volume, side.bot_params.filter_volume_ema_span).ok_or(
                        OrchestratorError::MissingEma {
                            symbol_idx: s.symbol_idx,
                        },
                    )?,
                )?
            } else {
                0.0
            };
            let volatility_score = if volatility_required {
                require_forager_input(
                    s.symbol_idx,
                    "forager_volatility_score",
                    ema_lookup(
                        &s.emas.m1.log_range,
                        side.bot_params.filter_volatility_ema_span,
                    )
                    .ok_or(OrchestratorError::MissingEma {
                        symbol_idx: s.symbol_idx,
                    })?,
                )?
            } else {
                0.0
            };
            let (bid, ask, ema_lower, ema_upper, entry_initial_ema_dist) = if ema_readiness_required
            {
                let ema_bands = match derive_ema_bands(s.symbol_idx, &s.emas, &strategy_params) {
                    Ok(v) => v,
                    Err(OrchestratorError::MissingEma { .. })
                    | Err(OrchestratorError::NonFiniteInput {
                        field: "ema_bands", ..
                    }) => {
                        out.push(ForagerCandidate {
                            index: s.symbol_idx,
                            enabled: false,
                            volume_score: 0.0,
                            volatility_score: 0.0,
                            bid: 0.0,
                            ask: 0.0,
                            ema_lower: 0.0,
                            ema_upper: 0.0,
                            entry_initial_ema_dist: 0.0,
                        });
                        continue;
                    }
                    Err(err) => return Err(err),
                };
                let entry_initial_ema_dist = strategy_initial_entry_offset(&strategy_params);
                let entry_initial_ema_dist = match require_forager_input(
                    s.symbol_idx,
                    "entry_initial_ema_dist",
                    entry_initial_ema_dist,
                ) {
                    Ok(v) => v,
                    Err(_) => {
                        out.push(ForagerCandidate {
                            index: s.symbol_idx,
                            enabled: false,
                            volume_score: 0.0,
                            volatility_score: 0.0,
                            bid: 0.0,
                            ask: 0.0,
                            ema_lower: 0.0,
                            ema_upper: 0.0,
                            entry_initial_ema_dist: 0.0,
                        });
                        continue;
                    }
                };
                (
                    require_forager_input(s.symbol_idx, "forager_market_bid", s.order_book.bid)?,
                    require_forager_input(s.symbol_idx, "forager_market_ask", s.order_book.ask)?,
                    ema_bands.lower,
                    ema_bands.upper,
                    entry_initial_ema_dist,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };
            out.push(ForagerCandidate {
                index: s.symbol_idx,
                enabled,
                volume_score,
                volatility_score,
                bid,
                ask,
                ema_lower,
                ema_upper,
                entry_initial_ema_dist,
            });
        }
        Ok(())
    }

    fn calc_panic_close(
        symbol_idx: usize,
        pside: PositionSide,
        pos: &Position,
        ob: &OrderBook,
        exchange: &ExchangeParams,
    ) -> Option<IdealOrder> {
        if pos.size == 0.0 {
            return None;
        }
        let qty = match pside {
            PositionSide::Long => -pos.size.abs(),
            PositionSide::Short => pos.size.abs(),
        };
        let price = match pside {
            PositionSide::Long => ob.ask - exchange.price_step,
            PositionSide::Short => ob.bid + exchange.price_step,
        };
        if !(price.is_finite() && price > 0.0 && qty.is_finite() && qty != 0.0) {
            return None;
        }
        Some(IdealOrder {
            symbol_idx,
            pside,
            qty,
            price,
            order_type: match pside {
                PositionSide::Long => OrderType::ClosePanicLong,
                PositionSide::Short => OrderType::ClosePanicShort,
            },
        })
    }

    fn should_generate_entries(mode: TradingMode, has_pos: bool, allow_initial: bool) -> bool {
        match mode {
            TradingMode::Manual => false,
            TradingMode::Panic => false,
            TradingMode::TpOnly => false,
            TradingMode::GracefulStop => has_pos, // only blocks initial entries
            TradingMode::Normal => has_pos || allow_initial,
        }
    }

    fn should_generate_closes(mode: TradingMode, has_pos: bool) -> bool {
        match mode {
            TradingMode::Manual => false,
            TradingMode::GracefulStop => has_pos,
            TradingMode::TpOnly => has_pos,
            TradingMode::Panic => has_pos,
            TradingMode::Normal => has_pos,
        }
    }

    fn effective_mode(mode: Option<TradingMode>, has_pos: bool) -> TradingMode {
        match mode {
            Some(TradingMode::GracefulStop) if has_pos => TradingMode::Normal,
            Some(m) => m,
            None => TradingMode::Normal,
        }
    }

    fn strategy_kind_for_symbol_side(global: &OrchestratorGlobal) -> StrategyKind {
        global.strategy_kind
    }

    fn strategy_params_for_symbol_side(
        global: &OrchestratorGlobal,
        side: &SymbolSideInput,
    ) -> Result<crate::strategies::StrategyParams, OrchestratorError> {
        parse_strategy_params(
            strategy_kind_for_symbol_side(global),
            side.strategy_params.as_ref(),
            &side.bot_params,
        )
        .map_err(|_| OrchestratorError::NonFiniteInput {
            field: "strategy_params",
            symbol_idx: None,
        })
    }

    fn append_strategy_orders_as_ideal(
        target: &mut Vec<IdealOrder>,
        orders: Vec<crate::types::Order>,
        symbol_idx: usize,
        pside: PositionSide,
    ) {
        for order in orders {
            target.push(IdealOrder {
                symbol_idx,
                pside,
                qty: order.qty,
                price: order.price,
                order_type: order.order_type,
            });
        }
    }

    #[derive(Clone)]
    struct PerSymbolOrders {
        symbol_idx: usize,
        entries: Vec<IdealOrder>,
        closes: Vec<IdealOrder>,
        pos: Position,
        mode: TradingMode,
    }

    /// Reusable buffers for performance-critical backtest loops.
    ///
    /// Callers must ensure outputs don't depend on previous contents; `compute_ideal_orders_with_workspace`
    /// fully overwrites the relevant buffers each call.
    #[derive(Default)]
    pub struct OrchestratorWorkspace {
        actives_long: Vec<bool>,
        actives_short: Vec<bool>,
        per_long: Vec<Option<PerSymbolOrders>>,
        per_short: Vec<Option<PerSymbolOrders>>,
        forced_long: Vec<usize>,
        forced_short: Vec<usize>,
        features: Vec<ForagerCandidate>,
        gate_positions_long: Vec<GateEntriesPosition>,
        gate_positions_short: Vec<GateEntriesPosition>,
        twel_positions: Vec<TwelEnforcerInputPosition>,
        unstuck_inputs: Vec<UnstuckPositionInput>,
        all_entries: Vec<IdealOrder>,
        gate_current_positions: Vec<Option<(f64, f64, f64)>>,
        gate_scratch: Vec<Option<(f64, f64, f64, f64)>>,
        gate_keep: Vec<u8>,
        gate_qty_by_order_idx: Vec<f64>,
        gate_out: Vec<IdealOrder>,
        /// One-way mode: per-symbol flags to block initial entries
        one_way_block_initial_long: Vec<bool>,
        one_way_block_initial_short: Vec<bool>,
    }

    fn gate_entries_by_twel_deterministic(
        pside: PositionSide,
        balance: f64,
        total_wallet_exposure_limit: f64,
        positions: &[GateEntriesPosition],
        entries: &mut Vec<IdealOrder>,
        symbols: &[SymbolInput],
        current_positions: &mut Vec<Option<(f64, f64, f64)>>,
        scratch: &mut Vec<Option<(f64, f64, f64, f64)>>,
        keep: &mut Vec<u8>,
        qty_by_order_idx: &mut Vec<f64>,
        out: &mut Vec<IdealOrder>,
    ) {
        const QTY_EPS: f64 = 1e-12;
        const EXP_EPS: f64 = 1e-12;
        if balance <= 0.0 || total_wallet_exposure_limit <= 0.0 || entries.is_empty() {
            entries.clear();
            return;
        }

        #[derive(Clone)]
        struct Candidate {
            order_idx: usize,
            symbol_idx: usize,
            qty: f64,
            price: f64,
            qty_step: f64,
            c_mult: f64,
            order_type: OrderType,
            distance: f64,
            effective_min_qty: f64,
        }

        current_positions.resize(symbols.len(), None);
        current_positions.fill(None);
        let mut current_twe = 0.0_f64;
        for pos in positions {
            if !pos.position_price.is_finite()
                || pos.position_price <= 0.0
                || !pos.position_size.is_finite()
            {
                continue;
            }
            let abs_psize = pos.position_size.abs();
            let exposure = calc_wallet_exposure(pos.c_mult, balance, abs_psize, pos.position_price);
            if exposure.is_finite() {
                current_twe += exposure;
            }
            if pos.idx < current_positions.len() {
                current_positions[pos.idx] = Some((abs_psize, pos.position_price, pos.c_mult));
            }
        }
        if current_twe >= total_wallet_exposure_limit - EXP_EPS {
            entries.clear();
            return;
        }

        let pside_int = pside.to_pside_int();
        let mut candidates: Vec<Candidate> = Vec::with_capacity(entries.len());
        for (order_idx, o) in entries.iter().enumerate() {
            if !o.price.is_finite() || o.price <= 0.0 || !o.qty.is_finite() {
                continue;
            }
            let qty_abs = o.qty.abs();
            if qty_abs <= QTY_EPS {
                continue;
            }
            let sym = match symbols.get(o.symbol_idx) {
                Some(v) => v,
                None => continue,
            };
            let ob = &sym.order_book;
            let exch = &sym.exchange;
            let effective_min_qty = calc_min_entry_qty(o.price, exch);
            if effective_min_qty.is_finite()
                && effective_min_qty > QTY_EPS
                && qty_abs + QTY_EPS < effective_min_qty
            {
                // Never include dust entries.
                continue;
            }
            let market_price = match pside {
                PositionSide::Long => ob.bid,
                PositionSide::Short => ob.ask,
            };
            if !(market_price.is_finite() && market_price > 0.0) {
                continue;
            }
            let distance = calc_pside_price_diff_int(pside_int, market_price, o.price);
            candidates.push(Candidate {
                order_idx,
                symbol_idx: o.symbol_idx,
                qty: qty_abs,
                price: o.price,
                qty_step: exch.qty_step,
                c_mult: exch.c_mult,
                order_type: o.order_type,
                distance,
                effective_min_qty,
            });
        }
        if candidates.is_empty() {
            entries.clear();
            return;
        }

        // Compute TWEL if all `selection` entries were filled, using a scratch vec (O(1) by idx)
        // to avoid hashing and repeated allocation in hot loops.
        scratch.resize(symbols.len(), None);
        let mut compute_twe_if_filled = |selection: &[(usize, f64)]| -> f64 {
            scratch.fill(None);
            let mut twe = current_twe;
            for (cand_idx, qty) in selection {
                let candidate = &candidates[*cand_idx];
                let qty = qty.max(0.0);
                if qty <= QTY_EPS {
                    continue;
                }
                let sym_idx = candidate.symbol_idx;
                if sym_idx >= scratch.len() {
                    continue;
                }

                let (psize, pprice, c_mult, exposure) = if let Some(v) = scratch[sym_idx] {
                    v
                } else if let Some((psize, pprice, c_mult)) = current_positions[sym_idx] {
                    let exp = calc_wallet_exposure(c_mult, balance, psize, pprice);
                    (psize, pprice, c_mult, exp)
                } else {
                    (0.0, candidate.price, candidate.c_mult, 0.0)
                };

                let old_exp = if exposure.is_finite() { exposure } else { 0.0 };
                let (new_psize, new_pprice) =
                    calc_new_psize_pprice(psize, pprice, qty, candidate.price, candidate.qty_step);
                let new_psize = new_psize.abs();
                let new_exp =
                    if new_psize <= QTY_EPS || !(new_pprice.is_finite() && new_pprice > 0.0) {
                        0.0
                    } else {
                        let exp = calc_wallet_exposure(c_mult, balance, new_psize, new_pprice);
                        if exp.is_finite() {
                            exp
                        } else {
                            0.0
                        }
                    };

                twe += new_exp - old_exp;
                scratch[sym_idx] = Some((new_psize, new_pprice, c_mult, new_exp));
            }
            twe
        };

        // `included` preserves candidate iteration order (ascending `cand_idx`) for determinism.
        // Keep it as a plain vec of (cand_idx, qty) during the removal phase, matching the
        // old behavior exactly.
        let mut included: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.qty))
            .collect();
        let mut twe_if_filled = compute_twe_if_filled(&included);
        if twe_if_filled < total_wallet_exposure_limit - EXP_EPS {
            // keep all
            return;
        }

        let mut removal_order: Vec<usize> = (0..candidates.len()).collect();
        removal_order.sort_by(|a, b| {
            let ca = &candidates[*a];
            let cb = &candidates[*b];
            cb.distance
                .partial_cmp(&ca.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ca.symbol_idx.cmp(&cb.symbol_idx))
                .then_with(|| ca.order_type.id().cmp(&cb.order_type.id()))
                .then_with(|| {
                    ca.price
                        .partial_cmp(&cb.price)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    ca.qty
                        .partial_cmp(&cb.qty)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| ca.order_idx.cmp(&cb.order_idx))
        });

        let mut removed_stack: Vec<(usize, f64)> = Vec::new();
        for cand_idx in removal_order {
            if twe_if_filled < total_wallet_exposure_limit - EXP_EPS {
                break;
            }
            // `cand_idx` exists at most once in `included`; avoid the O(n) search by tracking it
            // as we remove (included is always in ascending order).
            if let Ok(pos) = included.binary_search_by_key(&cand_idx, |(idx, _)| *idx) {
                let entry = included.remove(pos);
                twe_if_filled = compute_twe_if_filled(&included);
                removed_stack.push(entry);
            }
        }

        if twe_if_filled >= total_wallet_exposure_limit - EXP_EPS {
            entries.clear();
            return;
        }

        if let Some((cand_idx, original_qty)) = removed_stack.pop() {
            // binary search qty for last removed (closest among removed)
            let mut lo = 0.0_f64;
            let mut hi = original_qty;
            for _ in 0..50 {
                let mid = (lo + hi) / 2.0;
                // Important: preserve legacy behavior where the trial candidate is processed last
                // (it was appended to the end of `included`).
                included.push((cand_idx, mid));
                let twe = compute_twe_if_filled(&included);
                included.pop();
                if twe >= total_wallet_exposure_limit - EXP_EPS {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let candidate = &candidates[cand_idx];
            let final_qty = round_dn(lo, candidate.qty_step);
            if final_qty > QTY_EPS
                && (!candidate.effective_min_qty.is_finite()
                    || candidate.effective_min_qty <= QTY_EPS
                    || final_qty + QTY_EPS >= candidate.effective_min_qty)
            {
                included.push((cand_idx, final_qty));
            }
        }

        // Apply included decisions back into entries.
        keep.resize(entries.len(), 0);
        keep.fill(0);
        qty_by_order_idx.resize(entries.len(), f64::NAN);
        qty_by_order_idx.fill(f64::NAN);
        for (cand_idx, qty) in included {
            let c = &candidates[cand_idx];
            keep[c.order_idx] = 1;
            qty_by_order_idx[c.order_idx] = qty;
        }
        out.clear();
        out.reserve(entries.len());
        for (order_idx, mut o) in entries.drain(..).enumerate() {
            if keep[order_idx] == 0 {
                continue;
            }
            let qty_abs = qty_by_order_idx[order_idx];
            if qty_abs.is_finite() {
                o.qty = match pside {
                    PositionSide::Long => qty_abs.abs(),
                    PositionSide::Short => -qty_abs.abs(),
                };
            }
            // Final dust guard (e.g. binary-search partial could still end up below effective min qty).
            let exch = match symbols.get(o.symbol_idx).map(|s| &s.exchange) {
                Some(v) => v,
                None => continue,
            };
            let min_qty = calc_min_entry_qty(o.price, exch);
            if min_qty.is_finite() && min_qty > QTY_EPS && o.qty.abs() + QTY_EPS < min_qty {
                continue;
            }
            out.push(o);
        }
        std::mem::swap(entries, out);
    }

    pub fn compute_ideal_orders(
        input: &OrchestratorInput,
    ) -> Result<OrchestratorOutput, OrchestratorError> {
        let mut workspace = OrchestratorWorkspace::default();
        compute_ideal_orders_with_workspace(input, &mut workspace)
    }

    pub fn compute_ideal_orders_with_workspace(
        input: &OrchestratorInput,
        workspace: &mut OrchestratorWorkspace,
    ) -> Result<OrchestratorOutput, OrchestratorError> {
        if !input.balance.is_finite() {
            return Err(OrchestratorError::NonFiniteInput {
                field: "balance",
                symbol_idx: None,
            });
        }
        if input.balance_raw.is_infinite() {
            return Err(OrchestratorError::NonFiniteInput {
                field: "balance_raw",
                symbol_idx: None,
            });
        }
        let mut diagnostics = OrchestratorDiagnostics::default();

        // Validate invariants:
        // - order books must be present and non-zero (no silent fallbacks)
        // - symbols must be indexed by `symbol_idx` for O(1) access in hot loops
        let n_symbols = input.symbols.len();
        for (pos, s) in input.symbols.iter().enumerate() {
            if !(s.order_book.bid.is_finite()
                && s.order_book.ask.is_finite()
                && s.order_book.bid > 0.0
                && s.order_book.ask > 0.0)
            {
                return Err(OrchestratorError::InvalidOrderBook {
                    symbol_idx: s.symbol_idx,
                });
            }
            if s.symbol_idx != pos {
                return Err(OrchestratorError::NonContiguousSymbolIdx {
                    pos,
                    symbol_idx: s.symbol_idx,
                });
            }
        }

        // Compute selection sets and effective n_positions per pside.
        let enabled_long = is_pside_enabled(&input.global.global_bot_params, PositionSide::Long);
        let enabled_short = is_pside_enabled(&input.global.global_bot_params, PositionSide::Short);

        fill_forced_normal_indices(
            &input.symbols,
            PositionSide::Long,
            &mut workspace.forced_long,
        );
        fill_forced_normal_indices(
            &input.symbols,
            PositionSide::Short,
            &mut workspace.forced_short,
        );

        let eligible_long = input.symbols.iter().filter(|s| s.tradable).count();
        let eligible_short = eligible_long;

        let enp_long = compute_effective_n_positions(
            input.global.global_bot_params.long.n_positions,
            eligible_long,
            workspace.forced_long.len(),
        );
        let enp_short = compute_effective_n_positions(
            input.global.global_bot_params.short.n_positions,
            eligible_short,
            workspace.forced_short.len(),
        );

        // Active sets per pside:
        // - Always include all current positions (even if > n_positions), so we keep managing them.
        // - Only when there are open slots (`positions < effective_n_positions`), add forced normals
        //   then preferred coins until reaching the cap.
        workspace.actives_long.resize(n_symbols, false);
        workspace.actives_short.resize(n_symbols, false);
        for v in workspace.actives_long.iter_mut() {
            *v = false;
        }
        for v in workspace.actives_short.iter_mut() {
            *v = false;
        }

        let actives_long = &mut workspace.actives_long;
        let mut actives_long_count: usize = 0;
        for s in input.symbols.iter().filter(|s| s.long.position.size != 0.0) {
            if !actives_long[s.symbol_idx] {
                actives_long[s.symbol_idx] = true;
                actives_long_count += 1;
            }
        }
        if enabled_long && actives_long_count < enp_long {
            for idx in &workspace.forced_long {
                if actives_long_count >= enp_long {
                    break;
                }
                if !actives_long[*idx]
                    && one_way_allows_initial_slot(
                        &input.symbols,
                        *idx,
                        PositionSide::Long,
                        input.global.hedge_mode,
                    )
                {
                    actives_long[*idx] = true;
                    actives_long_count += 1;
                }
            }
            if actives_long_count < enp_long {
                let cfg = ForagerSelectionConfig {
                    slots_to_fill: enp_long.saturating_sub(actives_long_count),
                    volume_drop_pct: input.global.global_bot_params.long.forager_volume_drop_pct,
                    weights: input
                        .global
                        .global_bot_params
                        .long
                        .forager_score_weights
                        .clone(),
                    require_forager: true,
                    position_side: ForagerPositionSide::Long,
                };
                build_forager_candidates_into(
                    &input.symbols,
                    PositionSide::Long,
                    input.global.strategy_kind,
                    input.global.hedge_mode,
                    input.global.filter_by_min_effective_cost,
                    input.balance,
                    enp_long,
                    Some(actives_long),
                    &cfg,
                    &mut workspace.features,
                )?;
                for idx in select_forager_candidates(&workspace.features, &cfg)
                    .map_err(map_forager_selection_error)?
                {
                    if actives_long_count >= enp_long {
                        break;
                    }
                    if !actives_long[idx] {
                        actives_long[idx] = true;
                        actives_long_count += 1;
                    }
                }
            }
        }

        let actives_short = &mut workspace.actives_short;
        let mut actives_short_count: usize = 0;
        for s in input
            .symbols
            .iter()
            .filter(|s| s.short.position.size != 0.0)
        {
            if !actives_short[s.symbol_idx] {
                actives_short[s.symbol_idx] = true;
                actives_short_count += 1;
            }
        }
        if enabled_short && actives_short_count < enp_short {
            for idx in &workspace.forced_short {
                if actives_short_count >= enp_short {
                    break;
                }
                if !actives_short[*idx]
                    && one_way_allows_initial_slot(
                        &input.symbols,
                        *idx,
                        PositionSide::Short,
                        input.global.hedge_mode,
                    )
                {
                    actives_short[*idx] = true;
                    actives_short_count += 1;
                }
            }
            if actives_short_count < enp_short {
                let cfg = ForagerSelectionConfig {
                    slots_to_fill: enp_short.saturating_sub(actives_short_count),
                    volume_drop_pct: input.global.global_bot_params.short.forager_volume_drop_pct,
                    weights: input
                        .global
                        .global_bot_params
                        .short
                        .forager_score_weights
                        .clone(),
                    require_forager: true,
                    position_side: ForagerPositionSide::Short,
                };
                build_forager_candidates_into(
                    &input.symbols,
                    PositionSide::Short,
                    input.global.strategy_kind,
                    input.global.hedge_mode,
                    input.global.filter_by_min_effective_cost,
                    input.balance,
                    enp_short,
                    Some(actives_short),
                    &cfg,
                    &mut workspace.features,
                )?;
                for idx in select_forager_candidates(&workspace.features, &cfg)
                    .map_err(map_forager_selection_error)?
                {
                    if actives_short_count >= enp_short {
                        break;
                    }
                    if !actives_short[idx] {
                        actives_short[idx] = true;
                        actives_short_count += 1;
                    }
                }
            }
        }

        // One-way mode: compute per-symbol initial entry blocking
        workspace
            .one_way_block_initial_long
            .resize(n_symbols, false);
        workspace
            .one_way_block_initial_short
            .resize(n_symbols, false);
        for v in workspace.one_way_block_initial_long.iter_mut() {
            *v = false;
        }
        for v in workspace.one_way_block_initial_short.iter_mut() {
            *v = false;
        }
        if !input.global.hedge_mode {
            for s in &input.symbols {
                let has_long = s.long.position.size != 0.0;
                let has_short = s.short.position.size != 0.0;
                let idx = s.symbol_idx;

                if has_long && !has_short {
                    // Long position exists - block short initial entries
                    workspace.one_way_block_initial_short[idx] = true;
                } else if has_short && !has_long {
                    // Short position exists - block long initial entries
                    workspace.one_way_block_initial_long[idx] = true;
                } else if has_long && has_short {
                    // Both positions exist - violation of one-way mode; block both initial entries.
                    workspace.one_way_block_initial_long[idx] = true;
                    workspace.one_way_block_initial_short[idx] = true;
                } else {
                    // No position on either side - decide based on eligibility and EMA band distance
                    let long_enabled = enabled_long
                        && should_generate_entries(effective_mode(s.long.mode, false), false, true);
                    let short_enabled = enabled_short
                        && should_generate_entries(
                            effective_mode(s.short.mode, false),
                            false,
                            true,
                        );

                    if long_enabled && !short_enabled {
                        workspace.one_way_block_initial_short[idx] = true;
                        continue;
                    }
                    if short_enabled && !long_enabled {
                        workspace.one_way_block_initial_long[idx] = true;
                        continue;
                    }
                    if !long_enabled && !short_enabled {
                        workspace.one_way_block_initial_long[idx] = true;
                        workspace.one_way_block_initial_short[idx] = true;
                        continue;
                    }

                    // Both sides are eligible - choose based on EMA band distance.
                    let strategy_params_long =
                        strategy_params_for_symbol_side(&input.global, &s.long);
                    let strategy_params_short =
                        strategy_params_for_symbol_side(&input.global, &s.short);
                    let ema_bands_long = strategy_params_long
                        .as_ref()
                        .ok()
                        .and_then(|params| derive_ema_bands(idx, &s.emas, params).ok());
                    let ema_bands_short = strategy_params_short
                        .as_ref()
                        .ok()
                        .and_then(|params| derive_ema_bands(idx, &s.emas, params).ok());

                    if let (
                        Some(bands_long),
                        Some(bands_short),
                        Ok(params_long),
                        Ok(params_short),
                    ) = (
                        ema_bands_long,
                        ema_bands_short,
                        strategy_params_long,
                        strategy_params_short,
                    ) {
                        let entry_threshold_long =
                            bands_long.lower * (1.0 - strategy_initial_entry_offset(&params_long));
                        let entry_threshold_short = bands_short.upper
                            * (1.0 + strategy_initial_entry_offset(&params_short));

                        let dist_long = entry_threshold_long / s.order_book.bid - 1.0;
                        let dist_short = 1.0 - entry_threshold_short / s.order_book.ask;

                        // Block the side that's farther from triggering (smaller distance).
                        // Tie-break: favor long.
                        if dist_long >= dist_short {
                            workspace.one_way_block_initial_short[idx] = true;
                        } else {
                            workspace.one_way_block_initial_long[idx] = true;
                        }
                    }
                    // If EMA bands can't be derived, allow both (no blocking)
                }
            }
        }

        // Per-symbol order generation
        workspace.per_long.resize_with(n_symbols, || None);
        workspace.per_short.resize_with(n_symbols, || None);
        for v in workspace.per_long.iter_mut() {
            *v = None;
        }
        for v in workspace.per_short.iter_mut() {
            *v = None;
        }
        let per_long = &mut workspace.per_long;
        let per_short = &mut workspace.per_short;

        for s in &input.symbols {
            // LONG
            {
                let has_pos = s.long.position.size != 0.0;
                if !enabled_long && has_pos {
                    diagnostics
                        .warnings
                        .push(OrchestratorWarning::DisabledPsideHasPosition {
                            symbol_idx: s.symbol_idx,
                            pside: PositionSide::Long,
                        });
                }
                if !s.tradable && has_pos {
                    diagnostics
                        .warnings
                        .push(OrchestratorWarning::NonTradableHasPosition {
                            symbol_idx: s.symbol_idx,
                            pside: PositionSide::Long,
                        });
                }
                let mode = effective_mode(s.long.mode, has_pos);
                let mode = if !enabled_long {
                    TradingMode::Manual
                } else {
                    mode
                };

                let allow_initial = actives_long[s.symbol_idx]
                    && !workspace.one_way_block_initial_long[s.symbol_idx]
                    && effective_min_cost_is_low_enough(
                        input.balance,
                        input.global.filter_by_min_effective_cost,
                        s.effective_min_cost,
                        &s.long.bot_params,
                        &resolve_runtime_budget(&s.long, enp_long),
                    );

                let mut entries: Vec<IdealOrder> = Vec::new();
                let mut closes: Vec<IdealOrder> = Vec::new();

                if mode == TradingMode::Panic {
                    if let Some(p) = calc_panic_close(
                        s.symbol_idx,
                        PositionSide::Long,
                        &s.long.position,
                        &s.order_book,
                        &s.exchange,
                    ) {
                        closes.push(p);
                    }
                } else {
                    let wants_entries = should_generate_entries(mode, has_pos, allow_initial);
                    let wants_closes = should_generate_closes(mode, has_pos);
                    if wants_entries || wants_closes {
                        let strategy_params =
                            strategy_params_for_symbol_side(&input.global, &s.long)?;
                        let ema_bands = derive_ema_bands(s.symbol_idx, &s.emas, &strategy_params)?;
                        let entry_volatility_logrange_ema_1h =
                            derive_entry_volatility_logrange_ema_1h(
                                s.symbol_idx,
                                &s.emas,
                                &strategy_params,
                            )?;
                        let state = StateParams {
                            balance: input.balance,
                            order_book: s.order_book.clone(),
                            ema_bands,
                            entry_volatility_logrange_ema_1h,
                        };
                        let runtime_budget = resolve_runtime_budget(&s.long, enp_long);
                        let generated = generate_strategy_orders(
                            strategy_kind_for_symbol_side(&input.global),
                            StrategySide::Long,
                            StrategyRequest {
                                wants_entries,
                                wants_closes,
                                exchange: &s.exchange,
                                state: &state,
                                bot_params: &s.long.bot_params,
                                strategy_params: &strategy_params,
                                runtime_budget,
                                position: &s.long.position,
                                trailing: &s.long.trailing,
                                next_candle: s.next_candle.as_ref().map(|nc| NextStepHint {
                                    low: nc.low,
                                    high: nc.high,
                                    tradable: nc.tradable,
                                }),
                                peek: input.peek_hints.as_ref().map(|hints| PeekBehavior {
                                    expand_entries: hints.expand_grid_long.contains(&s.symbol_idx),
                                    expand_closes: hints.expand_close_long.contains(&s.symbol_idx),
                                }),
                            },
                        );
                        append_strategy_orders_as_ideal(
                            &mut entries,
                            generated.entries,
                            s.symbol_idx,
                            PositionSide::Long,
                        );
                        append_strategy_orders_as_ideal(
                            &mut closes,
                            generated.closes,
                            s.symbol_idx,
                            PositionSide::Long,
                        );
                    }
                }

                per_long[s.symbol_idx] = Some(PerSymbolOrders {
                    symbol_idx: s.symbol_idx,
                    entries,
                    closes,
                    pos: s.long.position,
                    mode,
                });
            }

            // SHORT
            {
                let has_pos = s.short.position.size != 0.0;
                if !enabled_short && has_pos {
                    diagnostics
                        .warnings
                        .push(OrchestratorWarning::DisabledPsideHasPosition {
                            symbol_idx: s.symbol_idx,
                            pside: PositionSide::Short,
                        });
                }
                if !s.tradable && has_pos {
                    diagnostics
                        .warnings
                        .push(OrchestratorWarning::NonTradableHasPosition {
                            symbol_idx: s.symbol_idx,
                            pside: PositionSide::Short,
                        });
                }
                let mode = effective_mode(s.short.mode, has_pos);
                let mode = if !enabled_short {
                    TradingMode::Manual
                } else {
                    mode
                };

                let allow_initial = actives_short[s.symbol_idx]
                    && !workspace.one_way_block_initial_short[s.symbol_idx]
                    && effective_min_cost_is_low_enough(
                        input.balance,
                        input.global.filter_by_min_effective_cost,
                        s.effective_min_cost,
                        &s.short.bot_params,
                        &resolve_runtime_budget(&s.short, enp_short),
                    );

                let mut entries: Vec<IdealOrder> = Vec::new();
                let mut closes: Vec<IdealOrder> = Vec::new();

                if mode == TradingMode::Panic {
                    if let Some(p) = calc_panic_close(
                        s.symbol_idx,
                        PositionSide::Short,
                        &s.short.position,
                        &s.order_book,
                        &s.exchange,
                    ) {
                        closes.push(p);
                    }
                } else {
                    let wants_entries = should_generate_entries(mode, has_pos, allow_initial);
                    let wants_closes = should_generate_closes(mode, has_pos);
                    if wants_entries || wants_closes {
                        let strategy_params =
                            strategy_params_for_symbol_side(&input.global, &s.short)?;
                        let ema_bands = derive_ema_bands(s.symbol_idx, &s.emas, &strategy_params)?;
                        let entry_volatility_logrange_ema_1h =
                            derive_entry_volatility_logrange_ema_1h(
                                s.symbol_idx,
                                &s.emas,
                                &strategy_params,
                            )?;
                        let state = StateParams {
                            balance: input.balance,
                            order_book: s.order_book.clone(),
                            ema_bands,
                            entry_volatility_logrange_ema_1h,
                        };
                        let runtime_budget = resolve_runtime_budget(&s.short, enp_short);
                        let generated = generate_strategy_orders(
                            strategy_kind_for_symbol_side(&input.global),
                            StrategySide::Short,
                            StrategyRequest {
                                wants_entries,
                                wants_closes,
                                exchange: &s.exchange,
                                state: &state,
                                bot_params: &s.short.bot_params,
                                strategy_params: &strategy_params,
                                runtime_budget,
                                position: &s.short.position,
                                trailing: &s.short.trailing,
                                next_candle: s.next_candle.as_ref().map(|nc| NextStepHint {
                                    low: nc.low,
                                    high: nc.high,
                                    tradable: nc.tradable,
                                }),
                                peek: input.peek_hints.as_ref().map(|hints| PeekBehavior {
                                    expand_entries: hints.expand_grid_short.contains(&s.symbol_idx),
                                    expand_closes: hints.expand_close_short.contains(&s.symbol_idx),
                                }),
                            },
                        );
                        append_strategy_orders_as_ideal(
                            &mut entries,
                            generated.entries,
                            s.symbol_idx,
                            PositionSide::Short,
                        );
                        append_strategy_orders_as_ideal(
                            &mut closes,
                            generated.closes,
                            s.symbol_idx,
                            PositionSide::Short,
                        );
                    }
                }

                per_short[s.symbol_idx] = Some(PerSymbolOrders {
                    symbol_idx: s.symbol_idx,
                    entries,
                    closes,
                    pos: s.short.position,
                    mode,
                });
            }
        }

        // Unstuck: select one global unstuck close and add it (do not replace).
        workspace.unstuck_inputs.clear();
        for s in per_long.iter().filter_map(|v| v.as_ref()) {
            if matches!(s.mode, TradingMode::Manual | TradingMode::Panic) || s.pos.size == 0.0 {
                continue;
            }
            let sym = &input.symbols[s.symbol_idx];
            let bot = &sym.long.bot_params;
            let runtime_budget = resolve_runtime_budget(&sym.long, enp_long);
            let enabled = bot.unstuck_loss_allowance_pct > 0.0
                && bot.unstuck_close_pct > 0.0
                && bot.unstuck_threshold > 0.0;
            if !enabled {
                continue;
            }
            let strategy_params = strategy_params_for_symbol_side(&input.global, &sym.long)?;
            let ema_bands = derive_ema_bands(s.symbol_idx, &sym.emas, &strategy_params)?;
            workspace.unstuck_inputs.push(UnstuckPositionInput {
                idx: s.symbol_idx,
                side: LONG,
                position_size: s.pos.size,
                position_price: s.pos.price,
                wallet_exposure_limit: runtime_budget.effective_wallet_exposure_limit,
                risk_we_excess_allowance_pct: bot.risk_we_excess_allowance_pct,
                unstuck_threshold: bot.unstuck_threshold,
                unstuck_close_pct: bot.unstuck_close_pct,
                unstuck_ema_dist: bot.unstuck_ema_dist,
                ema_band_upper: ema_bands.upper,
                ema_band_lower: ema_bands.lower,
                current_price: sym.order_book.ask,
                price_step: sym.exchange.price_step,
                qty_step: sym.exchange.qty_step,
                min_qty: sym.exchange.min_qty,
                min_cost: sym.exchange.min_cost,
                c_mult: sym.exchange.c_mult,
            });
        }
        for s in per_short.iter().filter_map(|v| v.as_ref()) {
            if matches!(s.mode, TradingMode::Manual | TradingMode::Panic) || s.pos.size == 0.0 {
                continue;
            }
            let sym = &input.symbols[s.symbol_idx];
            let bot = &sym.short.bot_params;
            let runtime_budget = resolve_runtime_budget(&sym.short, enp_short);
            let enabled = bot.unstuck_loss_allowance_pct > 0.0
                && bot.unstuck_close_pct > 0.0
                && bot.unstuck_threshold > 0.0;
            if !enabled {
                continue;
            }
            let strategy_params = strategy_params_for_symbol_side(&input.global, &sym.short)?;
            let ema_bands = derive_ema_bands(s.symbol_idx, &sym.emas, &strategy_params)?;
            workspace.unstuck_inputs.push(UnstuckPositionInput {
                idx: s.symbol_idx,
                side: SHORT,
                position_size: s.pos.size,
                position_price: s.pos.price,
                wallet_exposure_limit: runtime_budget.effective_wallet_exposure_limit,
                risk_we_excess_allowance_pct: bot.risk_we_excess_allowance_pct,
                unstuck_threshold: bot.unstuck_threshold,
                unstuck_close_pct: bot.unstuck_close_pct,
                unstuck_ema_dist: bot.unstuck_ema_dist,
                ema_band_upper: ema_bands.upper,
                ema_band_lower: ema_bands.lower,
                current_price: sym.order_book.bid,
                price_step: sym.exchange.price_step,
                qty_step: sym.exchange.qty_step,
                min_qty: sym.exchange.min_qty,
                min_cost: sym.exchange.min_cost,
                c_mult: sym.exchange.c_mult,
            });
        }
        if let Some((idx, side, order)) = calc_unstucking_action(
            input_balance_raw(input),
            input.global.unstuck_allowance_long,
            input.global.unstuck_allowance_short,
            &workspace.unstuck_inputs,
        ) {
            let ideal = IdealOrder {
                symbol_idx: idx,
                pside: if side == LONG {
                    PositionSide::Long
                } else {
                    PositionSide::Short
                },
                qty: order.qty,
                price: order.price,
                order_type: order.order_type,
            };
            match ideal.pside {
                PositionSide::Long => {
                    if let Some(s) = per_long.get_mut(idx).and_then(|v| v.as_mut()) {
                        s.closes.push(ideal);
                    }
                }
                PositionSide::Short => {
                    if let Some(s) = per_short.get_mut(idx).and_then(|v| v.as_mut()) {
                        s.closes.push(ideal);
                    }
                }
            }
        }

        // TWEL enforcer: add auto-reduce closes in addition to normal closes.
        if enabled_long {
            workspace.twel_positions.clear();
            for s in per_long.iter().filter_map(|v| v.as_ref()) {
                if matches!(s.mode, TradingMode::Manual | TradingMode::Panic) || s.pos.size == 0.0 {
                    continue;
                }
                // Skip TWEL enforcer for positions already running WEL auto-reduce.
                if s.closes
                    .iter()
                    .any(|o| o.order_type == OrderType::CloseAutoReduceWelLong)
                {
                    continue;
                }
                let sym = &input.symbols[s.symbol_idx];
                let runtime_budget = resolve_runtime_budget(&sym.long, enp_long);
                workspace.twel_positions.push(TwelEnforcerInputPosition {
                    idx: s.symbol_idx,
                    position_size: s.pos.size,
                    position_price: s.pos.price,
                    market_price: sym.order_book.bid,
                    base_wallet_exposure_limit: runtime_budget.effective_wallet_exposure_limit,
                    c_mult: sym.exchange.c_mult,
                    qty_step: sym.exchange.qty_step,
                    price_step: sym.exchange.price_step,
                    min_qty: sym.exchange.min_qty,
                    min_cost: sym.exchange.min_cost,
                });
            }
            let actions = calc_twel_enforcer_actions(
                LONG,
                input
                    .global
                    .global_bot_params
                    .long
                    .risk_twel_enforcer_threshold,
                input
                    .global
                    .global_bot_params
                    .long
                    .total_wallet_exposure_limit,
                enp_long,
                input_balance_raw(input),
                &workspace.twel_positions,
                None,
            );
            for (idx, order) in actions {
                if let Some(s) = per_long.get_mut(idx).and_then(|v| v.as_mut()) {
                    s.closes.push(IdealOrder {
                        symbol_idx: idx,
                        pside: PositionSide::Long,
                        qty: order.qty,
                        price: order.price,
                        order_type: order.order_type,
                    });
                }
            }
        }
        if enabled_short {
            workspace.twel_positions.clear();
            for s in per_short.iter().filter_map(|v| v.as_ref()) {
                if matches!(s.mode, TradingMode::Manual | TradingMode::Panic) || s.pos.size == 0.0 {
                    continue;
                }
                if s.closes
                    .iter()
                    .any(|o| o.order_type == OrderType::CloseAutoReduceWelShort)
                {
                    continue;
                }
                let sym = &input.symbols[s.symbol_idx];
                let runtime_budget = resolve_runtime_budget(&sym.short, enp_short);
                workspace.twel_positions.push(TwelEnforcerInputPosition {
                    idx: s.symbol_idx,
                    position_size: s.pos.size,
                    position_price: s.pos.price,
                    market_price: sym.order_book.ask,
                    base_wallet_exposure_limit: runtime_budget.effective_wallet_exposure_limit,
                    c_mult: sym.exchange.c_mult,
                    qty_step: sym.exchange.qty_step,
                    price_step: sym.exchange.price_step,
                    min_qty: sym.exchange.min_qty,
                    min_cost: sym.exchange.min_cost,
                });
            }
            let actions = calc_twel_enforcer_actions(
                SHORT,
                input
                    .global
                    .global_bot_params
                    .short
                    .risk_twel_enforcer_threshold,
                input
                    .global
                    .global_bot_params
                    .short
                    .total_wallet_exposure_limit,
                enp_short,
                input_balance_raw(input),
                &workspace.twel_positions,
                None,
            );
            for (idx, order) in actions {
                if let Some(s) = per_short.get_mut(idx).and_then(|v| v.as_mut()) {
                    s.closes.push(IdealOrder {
                        symbol_idx: idx,
                        pside: PositionSide::Short,
                        qty: order.qty,
                        price: order.price,
                        order_type: order.order_type,
                    });
                }
            }
        }

        // Trim closes per symbol to position size (furthest-first).
        for s in per_long.iter_mut().filter_map(|v| v.as_mut()) {
            let sym = &input.symbols[s.symbol_idx];
            trim_closes_to_position(
                PositionSide::Long,
                &mut s.closes,
                s.pos.size,
                &sym.order_book,
                &sym.exchange,
            );
        }
        for s in per_short.iter_mut().filter_map(|v| v.as_mut()) {
            let sym = &input.symbols[s.symbol_idx];
            trim_closes_to_position(
                PositionSide::Short,
                &mut s.closes,
                s.pos.size,
                &sym.order_book,
                &sym.exchange,
            );
        }

        // Global realized-loss gate for close orders (all close types except panic).
        gate_lossy_closes_by_peak_balance(input, per_long, per_short, &mut diagnostics);

        // Portfolio TWEL gating of entries per pside (reuse workspace buffers).
        workspace.gate_positions_long.clear();
        workspace.gate_positions_short.clear();
        for s in input.symbols.iter() {
            if s.long.position.size != 0.0 {
                workspace.gate_positions_long.push(GateEntriesPosition {
                    idx: s.symbol_idx,
                    position_size: s.long.position.size,
                    position_price: s.long.position.price,
                    c_mult: s.exchange.c_mult,
                });
            }
            if s.short.position.size != 0.0 {
                workspace.gate_positions_short.push(GateEntriesPosition {
                    idx: s.symbol_idx,
                    position_size: s.short.position.size,
                    position_price: s.short.position.price,
                    c_mult: s.exchange.c_mult,
                });
            }
        }

        if enabled_long {
            workspace.all_entries.clear();
            for s in per_long.iter_mut().filter_map(|v| v.as_mut()) {
                workspace.all_entries.append(&mut s.entries);
            }
            gate_entries_by_twel_deterministic(
                PositionSide::Long,
                input_balance_raw(input),
                input
                    .global
                    .global_bot_params
                    .long
                    .total_wallet_exposure_limit
                    .max(0.0),
                &workspace.gate_positions_long,
                &mut workspace.all_entries,
                &input.symbols,
                &mut workspace.gate_current_positions,
                &mut workspace.gate_scratch,
                &mut workspace.gate_keep,
                &mut workspace.gate_qty_by_order_idx,
                &mut workspace.gate_out,
            );
            for e in workspace.all_entries.drain(..) {
                if let Some(s) = per_long.get_mut(e.symbol_idx).and_then(|v| v.as_mut()) {
                    s.entries.push(e);
                }
            }
        } else {
            for s in per_long.iter_mut().filter_map(|v| v.as_mut()) {
                s.entries.clear();
            }
        }

        if enabled_short {
            workspace.all_entries.clear();
            for s in per_short.iter_mut().filter_map(|v| v.as_mut()) {
                workspace.all_entries.append(&mut s.entries);
            }
            gate_entries_by_twel_deterministic(
                PositionSide::Short,
                input_balance_raw(input),
                input
                    .global
                    .global_bot_params
                    .short
                    .total_wallet_exposure_limit
                    .max(0.0),
                &workspace.gate_positions_short,
                &mut workspace.all_entries,
                &input.symbols,
                &mut workspace.gate_current_positions,
                &mut workspace.gate_scratch,
                &mut workspace.gate_keep,
                &mut workspace.gate_qty_by_order_idx,
                &mut workspace.gate_out,
            );
            for e in workspace.all_entries.drain(..) {
                if let Some(s) = per_short.get_mut(e.symbol_idx).and_then(|v| v.as_mut()) {
                    s.entries.push(e);
                }
            }
        } else {
            for s in per_short.iter_mut().filter_map(|v| v.as_mut()) {
                s.entries.clear();
            }
        }

        // If global output sorting is disabled (backtest), we still need deterministic per-symbol
        // ordering for fill-priority. Sort per-symbol entries/closes by the canonical key.
        if !input.global.sort_global {
            for s in per_long.iter_mut().filter_map(|v| v.as_mut()) {
                let ob = &input.symbols[s.symbol_idx].order_book;
                s.closes.sort_by(|a, b| {
                    let ka = canonical_sort_key(a, ob);
                    let kb = canonical_sort_key(b, ob);
                    ka.0.partial_cmp(&kb.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| ka.4.cmp(&kb.4))
                        .then_with(|| ka.5.partial_cmp(&kb.5).unwrap_or(std::cmp::Ordering::Equal))
                        .then_with(|| ka.6.partial_cmp(&kb.6).unwrap_or(std::cmp::Ordering::Equal))
                });
                s.entries.sort_by(|a, b| {
                    let ka = canonical_sort_key(a, ob);
                    let kb = canonical_sort_key(b, ob);
                    ka.0.partial_cmp(&kb.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| ka.4.cmp(&kb.4))
                        .then_with(|| ka.5.partial_cmp(&kb.5).unwrap_or(std::cmp::Ordering::Equal))
                        .then_with(|| ka.6.partial_cmp(&kb.6).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            for s in per_short.iter_mut().filter_map(|v| v.as_mut()) {
                let ob = &input.symbols[s.symbol_idx].order_book;
                s.closes.sort_by(|a, b| {
                    let ka = canonical_sort_key(a, ob);
                    let kb = canonical_sort_key(b, ob);
                    ka.0.partial_cmp(&kb.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| ka.4.cmp(&kb.4))
                        .then_with(|| ka.5.partial_cmp(&kb.5).unwrap_or(std::cmp::Ordering::Equal))
                        .then_with(|| ka.6.partial_cmp(&kb.6).unwrap_or(std::cmp::Ordering::Equal))
                });
                s.entries.sort_by(|a, b| {
                    let ka = canonical_sort_key(a, ob);
                    let kb = canonical_sort_key(b, ob);
                    ka.0.partial_cmp(&kb.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| ka.4.cmp(&kb.4))
                        .then_with(|| ka.5.partial_cmp(&kb.5).unwrap_or(std::cmp::Ordering::Equal))
                        .then_with(|| ka.6.partial_cmp(&kb.6).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
        }

        // Collect and (optionally) globally sort.
        let mut total_orders: usize = 0;
        for s in per_long.iter().filter_map(|v| v.as_ref()) {
            total_orders += s.closes.len() + s.entries.len();
        }
        for s in per_short.iter().filter_map(|v| v.as_ref()) {
            total_orders += s.closes.len() + s.entries.len();
        }
        let mut orders: Vec<IdealOrder> = Vec::with_capacity(total_orders);
        for s in per_long.iter_mut().filter_map(|v| v.as_mut()) {
            orders.append(&mut s.closes);
            orders.append(&mut s.entries);
        }
        for s in per_short.iter_mut().filter_map(|v| v.as_mut()) {
            orders.append(&mut s.closes);
            orders.append(&mut s.entries);
        }

        if input.global.sort_global {
            orders.sort_by(|a, b| {
                let oba = &input.symbols[a.symbol_idx].order_book;
                let obb = &input.symbols[b.symbol_idx].order_book;
                let ka = canonical_sort_key(a, oba);
                let kb = canonical_sort_key(b, obb);
                ka.0.partial_cmp(&kb.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| ka.1.cmp(&kb.1))
                    .then_with(|| ka.2.cmp(&kb.2))
                    .then_with(|| ka.3.cmp(&kb.3))
                    .then_with(|| ka.4.cmp(&kb.4))
                    .then_with(|| ka.5.partial_cmp(&kb.5).unwrap_or(std::cmp::Ordering::Equal))
                    .then_with(|| ka.6.partial_cmp(&kb.6).unwrap_or(std::cmp::Ordering::Equal))
            });
        }

        let orders = orders
            .into_iter()
            .map(|order| {
                let order_book = &input.symbols[order.symbol_idx].order_book;
                to_executable_order(order, &input.global, order_book)
            })
            .collect();

        diagnostics.symbol_states = input
            .symbols
            .iter()
            .map(|s| {
                let long_mode = per_long[s.symbol_idx]
                    .as_ref()
                    .map(|state| state.mode)
                    .unwrap_or_else(|| effective_mode(s.long.mode, s.long.position.size != 0.0));
                let short_mode = per_short[s.symbol_idx]
                    .as_ref()
                    .map(|state| state.mode)
                    .unwrap_or_else(|| effective_mode(s.short.mode, s.short.position.size != 0.0));
                let long_allow_initial = workspace.actives_long[s.symbol_idx]
                    && !workspace.one_way_block_initial_long[s.symbol_idx]
                    && effective_min_cost_is_low_enough(
                        input.balance,
                        input.global.filter_by_min_effective_cost,
                        s.effective_min_cost,
                        &s.long.bot_params,
                        &resolve_runtime_budget(&s.long, enp_long),
                    );
                let short_allow_initial = workspace.actives_short[s.symbol_idx]
                    && !workspace.one_way_block_initial_short[s.symbol_idx]
                    && effective_min_cost_is_low_enough(
                        input.balance,
                        input.global.filter_by_min_effective_cost,
                        s.effective_min_cost,
                        &s.short.bot_params,
                        &resolve_runtime_budget(&s.short, enp_short),
                    );
                SymbolStateDiagnostic {
                    symbol_idx: s.symbol_idx,
                    long: SymbolSideStateDiagnostic {
                        input_mode: s.long.mode,
                        effective_mode: long_mode,
                        active: workspace.actives_long[s.symbol_idx],
                        allow_initial: long_allow_initial,
                    },
                    short: SymbolSideStateDiagnostic {
                        input_mode: s.short.mode,
                        effective_mode: short_mode,
                        active: workspace.actives_short[s.symbol_idx],
                        allow_initial: short_allow_initial,
                    },
                }
            })
            .collect();

        Ok(OrchestratorOutput {
            orders,
            diagnostics,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashSet;

        fn make_basic_symbol(idx: usize) -> SymbolInput {
            let mut emas = EmaBundle::default();
            emas.m1.close.push((10.0_f64, 100.0));
            emas.m1.close.push((20.0_f64, 100.0));
            emas.m1.close.push(((10.0_f64 * 20.0_f64).sqrt(), 100.0));
            emas.m1.volume.push((10.0_f64, 1.0));
            emas.m1.volume.push((60.0_f64, 1.0));
            emas.m1.log_range.push((10.0_f64, 1.0));
            emas.m1.log_range.push((60.0_f64, 1.0));
            emas.h1.log_range.push((1.0_f64, 1.0));

            let mut bp = BotParams::default();
            bp.ema_span_0 = 10.0;
            bp.ema_span_1 = 20.0;
            bp.entry_volatility_ema_span_hours = 1.0;
            bp.filter_volume_ema_span = 10.0;
            bp.filter_volatility_ema_span = 10.0;
            bp.total_wallet_exposure_limit = 1.0;
            bp.n_positions = 1;
            bp.wallet_exposure_limit = 1.0;
            bp.entry_initial_qty_pct = 1.0;

            SymbolInput {
                symbol_idx: idx,
                order_book: OrderBook {
                    bid: 100.0,
                    ask: 100.0,
                },
                exchange: ExchangeParams {
                    qty_step: 0.1,
                    price_step: 0.1,
                    min_qty: 0.0,
                    min_cost: 0.0,
                    c_mult: 1.0,
                    ..Default::default()
                },
                tradable: true,
                next_candle: None,
                effective_min_cost: 0.0,
                emas,
                long: SymbolSideInput {
                    mode: None,
                    position: Position::default(),
                    trailing: TrailingPriceBundle::default(),
                    bot_params: bp.clone(),
                    strategy_params: None,
                    runtime_budget: None,
                },
                short: SymbolSideInput {
                    mode: None,
                    position: Position::default(),
                    trailing: TrailingPriceBundle::default(),
                    bot_params: bp,
                    strategy_params: None,
                    runtime_budget: None,
                },
            }
        }

        fn make_basic_global() -> OrchestratorGlobal {
            OrchestratorGlobal {
                filter_by_min_effective_cost: false,
                market_orders_allowed: false,
                market_order_near_touch_threshold: 0.001,
                panic_close_market: false,
                unstuck_allowance_long: 0.0,
                unstuck_allowance_short: 0.0,
                max_realized_loss_pct: 1.0,
                realized_pnl_cumsum_max: 0.0,
                realized_pnl_cumsum_last: 0.0,
                sort_global: true,
                global_bot_params: BotParamsPair::default(),
                hedge_mode: true,
                strategy_kind: StrategyKind::TrailingGrid,
            }
        }

        #[test]
        fn market_execution_uses_near_touch_threshold() {
            let mut global = make_basic_global();
            global.market_orders_allowed = true;
            global.market_order_near_touch_threshold = 0.001;
            let order_book = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let order = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 100.05,
                order_type: OrderType::CloseGridLong,
            };
            assert!(should_use_market_execution(&order, &global, &order_book));
            let executable = to_executable_order(order, &global, &order_book);
            assert_eq!(executable.execution_type, ExecutionType::Market);
        }

        #[test]
        fn market_execution_uses_crossing_rules_for_buys_and_sells() {
            let mut global = make_basic_global();
            global.market_orders_allowed = true;
            global.market_order_near_touch_threshold = 0.0;
            let order_book = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let buy = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: 1.0,
                price: 100.0,
                order_type: OrderType::EntryGridNormalLong,
            };
            let sell = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 100.0,
                order_type: OrderType::CloseGridLong,
            };
            assert!(should_use_market_execution(&buy, &global, &order_book));
            assert!(should_use_market_execution(&sell, &global, &order_book));
            assert_eq!(
                to_executable_order(buy, &global, &order_book).execution_type,
                ExecutionType::Market
            );
            assert_eq!(
                to_executable_order(sell, &global, &order_book).execution_type,
                ExecutionType::Market
            );
        }

        #[test]
        fn market_execution_disabled_keeps_non_panic_orders_limit() {
            let global = make_basic_global();
            let order_book = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let order = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 100.0,
                order_type: OrderType::CloseGridLong,
            };
            assert!(!should_use_market_execution(&order, &global, &order_book));
            assert_eq!(
                to_executable_order(order, &global, &order_book).execution_type,
                ExecutionType::Limit
            );
        }

        #[test]
        fn panic_close_respects_panic_close_market_flag() {
            let mut global = make_basic_global();
            let order_book = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let order = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 50.0,
                order_type: OrderType::ClosePanicLong,
            };
            assert_eq!(
                to_executable_order(order.clone(), &global, &order_book).execution_type,
                ExecutionType::Limit
            );
            global.panic_close_market = true;
            assert_eq!(
                to_executable_order(order, &global, &order_book).execution_type,
                ExecutionType::Market
            );
        }

        #[test]
        fn next_candle_peek_expands_only_when_next_would_fill() {
            let mut sym = make_basic_symbol(0);
            sym.long.bot_params.entry_grid_spacing_pct = 0.01;
            sym.long.bot_params.entry_grid_double_down_factor = 1.2;
            sym.long.bot_params.entry_initial_qty_pct = 0.01;
            sym.long.bot_params.wallet_exposure_limit = 100.0;
            sym.long.position = Position {
                size: 100.0,
                price: 100.0,
            };
            sym.next_candle = Some(NextCandle {
                low: 1e9,
                high: 1e9,
                tradable: true,
            });

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: {
                        let mut pair = BotParamsPair::default();
                        pair.long.total_wallet_exposure_limit = 1000.0;
                        pair.long.n_positions = 1;
                        pair
                    },
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym.clone()],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let n_entries_no_fill = out
                .orders
                .iter()
                .filter(|o| o.pside == PositionSide::Long && !is_close_order_type(o.order_type))
                .count();
            assert_eq!(n_entries_no_fill, 1);

            let mut sym_fill = sym;
            sym_fill.next_candle = Some(NextCandle {
                low: 0.0,
                high: 0.0,
                tradable: true,
            });
            let input_fill = OrchestratorInput {
                symbols: vec![sym_fill],
                ..input
            };
            let out_fill = compute_ideal_orders(&input_fill).unwrap();
            let n_entries_fill = out_fill
                .orders
                .iter()
                .filter(|o| o.pside == PositionSide::Long && !is_close_order_type(o.order_type))
                .count();
            assert!(n_entries_fill > 1);
        }

        #[test]
        fn adaptive_grid_long_entry_output_regression() {
            let mut sym = make_basic_symbol(0);
            sym.long.bot_params.entry_initial_ema_dist = -0.01;
            sym.short.bot_params.total_wallet_exposure_limit = 0.0;
            sym.short.bot_params.n_positions = 0;

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;
            global_bp.short.total_wallet_exposure_limit = 0.0;
            global_bp.short.n_positions = 0;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let actual: Vec<(PositionSide, f64, f64, OrderType)> = out
                .orders
                .iter()
                .map(|o| (o.pside, o.qty, o.price, o.order_type))
                .collect();
            let expected = vec![
                (
                    PositionSide::Long,
                    1.0,
                    100.0,
                    OrderType::EntryInitialNormalLong,
                ),
                (
                    PositionSide::Long,
                    1.02,
                    98.0,
                    OrderType::EntryGridNormalLong,
                ),
                (
                    PositionSide::Long,
                    2.02,
                    97.01,
                    OrderType::EntryGridNormalLong,
                ),
                (
                    PositionSide::Long,
                    4.04,
                    96.04,
                    OrderType::EntryGridNormalLong,
                ),
                (
                    PositionSide::Long,
                    2.27,
                    95.07,
                    OrderType::EntryGridCroppedLong,
                ),
            ];
            assert_eq!(actual, expected);
        }

        #[test]
        fn adaptive_grid_short_entry_output_regression() {
            let mut sym = make_basic_symbol(0);
            sym.long.bot_params.total_wallet_exposure_limit = 0.0;
            sym.long.bot_params.n_positions = 0;
            sym.short.bot_params.total_wallet_exposure_limit = 1.0;
            sym.short.bot_params.n_positions = 1;
            sym.short.bot_params.entry_initial_ema_dist = 0.01;

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 0.0;
            global_bp.long.n_positions = 0;
            global_bp.short.total_wallet_exposure_limit = 1.0;
            global_bp.short.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let actual: Vec<(PositionSide, f64, f64, OrderType)> = out
                .orders
                .iter()
                .map(|o| (o.pside, o.qty, o.price, o.order_type))
                .collect();
            let expected = vec![
                (
                    PositionSide::Short,
                    -0.99,
                    101.0,
                    OrderType::EntryInitialNormalShort,
                ),
                (
                    PositionSide::Short,
                    -0.99,
                    103.02,
                    OrderType::EntryGridNormalShort,
                ),
                (
                    PositionSide::Short,
                    -1.98,
                    104.06,
                    OrderType::EntryGridNormalShort,
                ),
                (
                    PositionSide::Short,
                    -5.63,
                    105.1,
                    OrderType::EntryGridInflatedShort,
                ),
            ];
            assert_eq!(actual, expected);
        }

        #[test]
        fn canonical_sort_close_before_entry_on_ties() {
            let ob = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let close = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 100.0,
                order_type: OrderType::CloseGridLong,
            };
            let entry = IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: 1.0,
                price: 100.0,
                order_type: OrderType::EntryGridNormalLong,
            };
            let kc = canonical_sort_key(&close, &ob);
            let ke = canonical_sort_key(&entry, &ob);
            assert!(kc.3 < ke.3);
        }

        #[test]
        fn graceful_stop_blocks_only_initial_entries() {
            let mut sym = make_basic_symbol(0);
            sym.long.mode = Some(TradingMode::GracefulStop);
            sym.long.position = Position {
                size: 0.0,
                price: 0.0,
            };

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;
            global_bp.long.forager_volume_drop_pct = 0.0;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };
            let out = compute_ideal_orders(&input).unwrap();
            // With no position and GracefulStop, we should not emit any entries.
            assert!(out.orders.iter().all(|o| is_close_order_type(o.order_type)));
        }

        #[test]
        fn one_way_blocks_disabled_long_allows_short_initials() {
            let sym = make_basic_symbol(0);

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 0.0;
            global_bp.long.n_positions = 1;
            global_bp.short.total_wallet_exposure_limit = 1.0;
            global_bp.short.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: false,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let has_long_entries = out
                .orders
                .iter()
                .any(|o| o.pside == PositionSide::Long && !is_close_order_type(o.order_type));
            let has_short_entries = out
                .orders
                .iter()
                .any(|o| o.pside == PositionSide::Short && !is_close_order_type(o.order_type));
            assert!(!has_long_entries);
            assert!(has_short_entries);
        }

        #[test]
        fn one_way_blocks_disabled_short_allows_long_initials() {
            let sym = make_basic_symbol(0);

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;
            global_bp.short.total_wallet_exposure_limit = 0.0;
            global_bp.short.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: false,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let has_long_entries = out
                .orders
                .iter()
                .any(|o| o.pside == PositionSide::Long && !is_close_order_type(o.order_type));
            let has_short_entries = out
                .orders
                .iter()
                .any(|o| o.pside == PositionSide::Short && !is_close_order_type(o.order_type));
            assert!(has_long_entries);
            assert!(!has_short_entries);
        }

        #[test]
        fn one_way_short_forager_skips_coin_already_held_long() {
            let mut sym0 = make_basic_symbol(0);
            let mut sym1 = make_basic_symbol(1);

            sym0.long.position = Position {
                size: 1.0,
                price: 100.0,
            };
            sym0.emas.m1.volume = vec![(10.0, 10.0)];
            sym0.emas.m1.log_range = vec![(10.0, 10.0)];
            sym1.emas.m1.volume = vec![(10.0, 5.0)];
            sym1.emas.m1.log_range = vec![(10.0, 5.0)];

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 0.0;
            global_bp.long.n_positions = 1;
            global_bp.short.total_wallet_exposure_limit = 1.0;
            global_bp.short.n_positions = 1;
            global_bp.long.forager_volume_drop_pct = 0.0;
            global_bp.short.forager_volume_drop_pct = 0.0;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: false,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym0, sym1],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let short_entry_symbol_idxs: Vec<usize> = out
                .orders
                .iter()
                .filter(|o| o.pside == PositionSide::Short && !is_close_order_type(o.order_type))
                .map(|o| o.symbol_idx)
                .collect();
            assert_eq!(short_entry_symbol_idxs, vec![1]);
        }

        #[test]
        fn one_way_long_forager_skips_coin_already_held_short() {
            let mut sym0 = make_basic_symbol(0);
            let mut sym1 = make_basic_symbol(1);

            sym0.short.position = Position {
                size: -1.0,
                price: 100.0,
            };
            sym0.emas.m1.volume = vec![(10.0, 10.0)];
            sym0.emas.m1.log_range = vec![(10.0, 10.0)];
            sym1.emas.m1.volume = vec![(10.0, 5.0)];
            sym1.emas.m1.log_range = vec![(10.0, 5.0)];

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;
            global_bp.short.total_wallet_exposure_limit = 1.0;
            global_bp.short.n_positions = 1;
            global_bp.long.forager_volume_drop_pct = 0.0;
            global_bp.short.forager_volume_drop_pct = 0.0;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: false,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym0, sym1],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let long_entry_symbol_idxs: Vec<usize> = out
                .orders
                .iter()
                .filter(|o| o.pside == PositionSide::Long && !is_close_order_type(o.order_type))
                .map(|o| o.symbol_idx)
                .collect();
            assert_eq!(long_entry_symbol_idxs, vec![1]);
        }

        #[test]
        fn forager_readiness_invalid_candidate_is_skipped_instead_of_panicking() {
            let mut sym0 = make_basic_symbol(0);
            let mut sym1 = make_basic_symbol(1);

            sym0.emas.m1.close = vec![(10.0, 0.0), (14.142135623730951, 0.0), (20.0, 0.0)];
            sym0.emas.m1.volume = vec![(10.0, 10.0)];
            sym0.emas.m1.log_range = vec![(10.0, 1.0)];

            sym1.emas.m1.close = vec![(10.0, 100.0), (14.142135623730951, 100.0), (20.0, 100.0)];
            sym1.emas.m1.volume = vec![(10.0, 9.0)];
            sym1.emas.m1.log_range = vec![(10.0, 1.0)];

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;
            global_bp.long.forager_volume_drop_pct = 0.0;
            global_bp.long.forager_score_weights.ema_readiness = 1.0;
            global_bp.long.forager_score_weights.volume = 0.0;
            global_bp.long.forager_score_weights.volatility = 0.0;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym0, sym1],
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let long_entry_symbol_idxs: Vec<usize> = out
                .orders
                .iter()
                .filter(|o| o.pside == PositionSide::Long && !is_close_order_type(o.order_type))
                .map(|o| o.symbol_idx)
                .collect();
            assert_eq!(long_entry_symbol_idxs, vec![1]);
        }

        #[test]
        fn non_contiguous_symbol_idx_is_rejected() {
            let sym0 = make_basic_symbol(0);
            let mut sym1 = make_basic_symbol(0);
            sym1.order_book = OrderBook {
                bid: 101.0,
                ask: 101.0,
            };

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;
            global_bp.short.total_wallet_exposure_limit = 1.0;
            global_bp.short.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym0, sym1],
                peek_hints: None,
            };

            let err = compute_ideal_orders(&input).unwrap_err();
            assert_eq!(
                err,
                OrchestratorError::NonContiguousSymbolIdx {
                    pos: 1,
                    symbol_idx: 0
                }
            );
        }

        #[test]
        fn trim_closes_prefers_keeping_closer_orders_long() {
            let ob = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let exchange = ExchangeParams {
                qty_step: 0.1,
                price_step: 0.1,
                min_qty: 0.0,
                min_cost: 0.0,
                c_mult: 1.0,
                ..Default::default()
            };
            let mut closes = vec![
                IdealOrder {
                    symbol_idx: 0,
                    pside: PositionSide::Long,
                    qty: -1.0,
                    price: 101.0,
                    order_type: OrderType::CloseGridLong,
                },
                IdealOrder {
                    symbol_idx: 0,
                    pside: PositionSide::Long,
                    qty: -1.0,
                    price: 100.5,
                    order_type: OrderType::CloseGridLong,
                },
            ];
            trim_closes_to_position(PositionSide::Long, &mut closes, 1.5, &ob, &exchange);
            let total: f64 = closes.iter().map(|o| o.qty.abs()).sum();
            assert!(total <= 1.5 + 1e-9);
            assert_eq!(closes.len(), 2);
            // The furthest close (101.0) should be trimmed first.
            let trimmed = closes
                .iter()
                .find(|o| (o.price - 101.0).abs() < 1e-9)
                .unwrap();
            assert!(
                (trimmed.qty + 0.5).abs() < 1e-9,
                "expected -0.5, got {}",
                trimmed.qty
            );
        }

        #[test]
        fn slot_accounting_blocks_extra_initial_entries() {
            // Regression: only allow opening initial positions on empty slots.
            // If `n_positions=4` and we already have 2 positions, we must allow at most 2
            // additional initial-entry coins, even if selection would pick 4 candidates.
            let mut syms: Vec<SymbolInput> = (0..6).map(make_basic_symbol).collect();

            // Existing long positions occupy slots.
            syms[0].long.position = Position {
                size: 1.0,
                price: 100.0,
            };
            syms[1].long.position = Position {
                size: 1.0,
                price: 100.0,
            };

            // Make non-position coins appear more volatile so selection prefers them.
            for idx in 0..6 {
                let v = if idx >= 2 { 10.0 } else { 1.0 };
                syms[idx].emas.m1.log_range.clear();
                syms[idx].emas.m1.log_range.push((10.0_f64, v));
            }

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1000.0;
            global_bp.long.n_positions = 4;
            global_bp.long.forager_volume_drop_pct = 0.0;
            global_bp.long.filter_volume_ema_span = 10.0;
            global_bp.long.filter_volatility_ema_span = 10.0;
            // disable short for this test
            global_bp.short.total_wallet_exposure_limit = 0.0;
            global_bp.short.n_positions = 0;

            let input = OrchestratorInput {
                balance: 1_000_000.0,
                balance_raw: 1_000_000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: syms,
                peek_hints: None,
            };

            let out = compute_ideal_orders(&input).unwrap();
            let mut entry_syms: HashSet<usize> = HashSet::new();
            for o in out.orders {
                if o.pside == PositionSide::Long && o.qty > 0.0 && o.symbol_idx >= 2 {
                    entry_syms.insert(o.symbol_idx);
                }
            }
            assert_eq!(entry_syms.len(), 2, "should only fill remaining 2 slots");
            assert!(entry_syms.contains(&2));
            assert!(entry_syms.contains(&3));
        }

        #[test]
        fn twel_gating_removes_furthest_entry_first() {
            let balance = 1000.0;
            let total_wel = 0.015; // $15
            let ob = OrderBook {
                bid: 20.0,
                ask: 20.0,
            };
            let exchange = ExchangeParams {
                qty_step: 0.1,
                price_step: 0.1,
                min_qty: 0.0,
                min_cost: 0.0,
                c_mult: 1.0,
                ..Default::default()
            };
            let mut sym = make_basic_symbol(0);
            sym.order_book = ob;
            sym.exchange = exchange;
            let symbols = vec![sym];

            let positions: Vec<GateEntriesPosition> = Vec::new();
            let mut entries = vec![
                IdealOrder {
                    symbol_idx: 0,
                    pside: PositionSide::Long,
                    qty: 1.0,
                    price: 10.0, // further from fill
                    order_type: OrderType::EntryGridNormalLong,
                },
                IdealOrder {
                    symbol_idx: 0,
                    pside: PositionSide::Long,
                    qty: 1.0,
                    price: 20.0, // closest to fill
                    order_type: OrderType::EntryGridNormalLong,
                },
            ];

            let mut current_positions: Vec<Option<(f64, f64, f64)>> = Vec::new();
            let mut scratch: Vec<Option<(f64, f64, f64, f64)>> = Vec::new();
            let mut keep: Vec<u8> = Vec::new();
            let mut qty_by_order_idx: Vec<f64> = Vec::new();
            let mut out: Vec<IdealOrder> = Vec::new();
            gate_entries_by_twel_deterministic(
                PositionSide::Long,
                balance,
                total_wel,
                &positions,
                &mut entries,
                &symbols,
                &mut current_positions,
                &mut scratch,
                &mut keep,
                &mut qty_by_order_idx,
                &mut out,
            );

            assert_eq!(entries.len(), 1);
            assert!((entries[0].price - 20.0).abs() < 1e-9);
            // $15 / $20 = 0.75, rounded down to qty_step 0.1 => 0.7
            assert!(
                (entries[0].qty - 0.7).abs() < 1e-9,
                "qty {}",
                entries[0].qty
            );
        }

        #[test]
        fn twel_gating_drops_dust_partial_entry() {
            // One large entry would exceed TWEL; partial re-add would land below effective min qty,
            // and must therefore be dropped (no dust orders).
            let symbol_idx = 0usize;
            let balance = 1000.0;
            let total_wel = 0.1; // allow only qty < 1.0 at price 100

            let ob = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let exchange = ExchangeParams {
                qty_step: 1.0,
                price_step: 0.1,
                min_qty: 10.0,
                min_cost: 0.0,
                c_mult: 1.0,
                ..Default::default()
            };
            let mut sym = make_basic_symbol(symbol_idx);
            sym.order_book = ob;
            sym.exchange = exchange;
            let symbols = vec![sym];

            let positions: Vec<GateEntriesPosition> = Vec::new();
            let mut entries = vec![IdealOrder {
                symbol_idx,
                pside: PositionSide::Long,
                qty: 100.0,
                price: 100.0,
                order_type: OrderType::EntryTrailingNormalLong,
            }];

            let mut current_positions: Vec<Option<(f64, f64, f64)>> = Vec::new();
            let mut scratch: Vec<Option<(f64, f64, f64, f64)>> = Vec::new();
            let mut keep: Vec<u8> = Vec::new();
            let mut qty_by_order_idx: Vec<f64> = Vec::new();
            let mut out: Vec<IdealOrder> = Vec::new();
            gate_entries_by_twel_deterministic(
                PositionSide::Long,
                balance,
                total_wel,
                &positions,
                &mut entries,
                &symbols,
                &mut current_positions,
                &mut scratch,
                &mut keep,
                &mut qty_by_order_idx,
                &mut out,
            );

            assert!(entries.is_empty(), "expected dust entry to be dropped");
        }

        #[test]
        fn closes_respect_effective_min_qty_except_tiny_position() {
            let ob = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            let exchange = ExchangeParams {
                qty_step: 1.0,
                price_step: 0.1,
                min_qty: 10.0,
                min_cost: 0.0,
                c_mult: 1.0,
                ..Default::default()
            };

            // pos larger than effective min => drop dust close
            let mut closes = vec![IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 101.0,
                order_type: OrderType::CloseGridLong,
            }];
            trim_closes_to_position(PositionSide::Long, &mut closes, 100.0, &ob, &exchange);
            assert!(closes.is_empty());

            // pos smaller than effective min => allow full close of tiny pos
            let mut closes = vec![IdealOrder {
                symbol_idx: 0,
                pside: PositionSide::Long,
                qty: -1.0,
                price: 101.0,
                order_type: OrderType::CloseGridLong,
            }];
            trim_closes_to_position(PositionSide::Long, &mut closes, 5.0, &ob, &exchange);
            assert_eq!(closes.len(), 1);
            assert!((closes[0].qty + 5.0).abs() < 1e-9);
        }

        #[test]
        fn realized_loss_gate_blocks_auto_reduce_orders() {
            let mut sym = make_basic_symbol(0);
            sym.long.position = Position {
                size: 10.0,
                price: 100.0,
            };
            sym.order_book = OrderBook {
                bid: 80.0,
                ask: 80.0,
            };
            sym.long.bot_params.wallet_exposure_limit = 0.5;
            sym.long.bot_params.risk_wel_enforcer_threshold = 1.0;
            sym.long.bot_params.total_wallet_exposure_limit = 1.0;
            sym.long.bot_params.n_positions = 1;

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;

            let input_open = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp.clone(),
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym.clone()],
                peek_hints: None,
            };
            let out_open = compute_ideal_orders(&input_open).unwrap();
            assert!(
                out_open
                    .orders
                    .iter()
                    .any(|o| o.order_type == OrderType::CloseAutoReduceWelLong),
                "expected WEL auto-reduce order when gate is disabled"
            );

            let mut input_blocked = input_open.clone();
            input_blocked.global.max_realized_loss_pct = 0.01;
            let out_blocked = compute_ideal_orders(&input_blocked).unwrap();
            assert!(
                out_blocked
                    .orders
                    .iter()
                    .all(|o| o.order_type != OrderType::CloseAutoReduceWelLong),
                "expected WEL auto-reduce order to be blocked by realized-loss gate"
            );
            assert!(
                out_blocked
                    .diagnostics
                    .loss_gate_blocks
                    .iter()
                    .any(|b| b.order_type == OrderType::CloseAutoReduceWelLong),
                "expected loss-gate diagnostic for blocked auto-reduce order"
            );
        }

        #[test]
        fn realized_loss_gate_missing_balance_raw_falls_back_to_balance() {
            let mut sym = make_basic_symbol(0);
            sym.long.position = Position {
                size: 10.0,
                price: 100.0,
            };
            sym.order_book = OrderBook {
                bid: 80.0,
                ask: 80.0,
            };
            sym.long.bot_params.wallet_exposure_limit = 0.5;
            sym.long.bot_params.risk_wel_enforcer_threshold = 1.0;
            sym.long.bot_params.total_wallet_exposure_limit = 1.0;
            sym.long.bot_params.n_positions = 1;

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                // Missing from JSON deserialization path defaults to NaN.
                balance_raw: f64::NAN,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 0.01,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };
            let out = compute_ideal_orders(&input).unwrap();
            assert!(
                out.orders
                    .iter()
                    .all(|o| o.order_type != OrderType::CloseAutoReduceWelLong),
                "expected missing balance_raw to fall back to snapped balance and gate closes"
            );
            assert!(
                out.diagnostics
                    .loss_gate_blocks
                    .iter()
                    .any(|b| b.order_type == OrderType::CloseAutoReduceWelLong
                        && (b.balance_before - 1000.0).abs() < 1e-9),
                "expected diagnostics to reflect snapped-balance fallback as balance_before"
            );
        }

        #[test]
        fn realized_loss_gate_non_positive_balance_raw_returns_early() {
            for raw_balance in [0.0, -1.0] {
                let mut sym = make_basic_symbol(0);
                sym.long.position = Position {
                    size: 10.0,
                    price: 100.0,
                };
                sym.order_book = OrderBook {
                    bid: 80.0,
                    ask: 80.0,
                };
                sym.long.bot_params.wallet_exposure_limit = 0.5;
                sym.long.bot_params.risk_wel_enforcer_threshold = 1.0;
                sym.long.bot_params.total_wallet_exposure_limit = 1.0;
                sym.long.bot_params.n_positions = 1;

                let mut global_bp = BotParamsPair::default();
                global_bp.long.total_wallet_exposure_limit = 1.0;
                global_bp.long.n_positions = 1;

                let input = OrchestratorInput {
                    balance: 1000.0,
                    balance_raw: raw_balance,
                    global: OrchestratorGlobal {
                        filter_by_min_effective_cost: false,
                        market_orders_allowed: false,
                        market_order_near_touch_threshold: 0.001,
                        panic_close_market: false,
                        unstuck_allowance_long: 0.0,
                        unstuck_allowance_short: 0.0,
                        max_realized_loss_pct: 0.01,
                        realized_pnl_cumsum_max: 0.0,
                        realized_pnl_cumsum_last: 0.0,
                        sort_global: true,
                        global_bot_params: global_bp,
                        hedge_mode: true,
                        strategy_kind: StrategyKind::TrailingGrid,
                    },
                    symbols: vec![sym],
                    peek_hints: None,
                };
                let out = compute_ideal_orders(&input).unwrap();
                assert!(
                    out.orders
                        .iter()
                        .any(|o| o.order_type == OrderType::CloseAutoReduceWelLong),
                    "expected non-positive balance_raw={} to early-return and keep close order",
                    raw_balance
                );
                assert!(
                    out.diagnostics.loss_gate_blocks.is_empty(),
                    "expected non-positive balance_raw={} to skip loss-gate diagnostics",
                    raw_balance
                );
            }
        }

        #[test]
        fn twel_enforcer_uses_balance_raw_not_snapped() {
            // Scenario: position has 500 cost, snapped balance = 1000 (WE = 0.5),
            // raw balance = 800 (WE = 0.625). TWEL = 0.6, threshold = 1.0.
            // With snapped: WE(0.5) < limit(0.6) → no auto-reduce.
            // With raw:     WE(0.625) > limit(0.6) → triggers auto-reduce.
            let mut sym = make_basic_symbol(0);
            sym.long.position = Position {
                size: 5.0,
                price: 100.0,
            };
            sym.order_book = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            // Keep WEL reducer inactive while leaving enough reducible size for TWEL reducer.
            sym.long.bot_params.wallet_exposure_limit = 0.4;
            sym.long.bot_params.risk_wel_enforcer_threshold = 2.0;
            sym.long.bot_params.total_wallet_exposure_limit = 0.6;
            sym.long.bot_params.n_positions = 1;

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 0.6;
            global_bp.long.risk_twel_enforcer_threshold = 1.0;
            global_bp.long.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,    // snapped: WE = 500/1000 = 0.5 (under limit)
                balance_raw: 800.0, // raw: WE = 500/800 = 0.625 (over limit)
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };
            let out = compute_ideal_orders(&input).unwrap();
            assert!(
                out.orders
                    .iter()
                    .any(|o| o.order_type == OrderType::CloseAutoReduceTwelLong),
                "TWEL enforcer should trigger using raw balance (WE=0.625 > 0.6), \
                 not snapped balance (WE=0.5 < 0.6). Orders: {:?}",
                out.orders.iter().map(|o| &o.order_type).collect::<Vec<_>>()
            );
        }

        #[test]
        fn twel_entry_gate_uses_balance_raw_not_snapped() {
            // Scenario: no position, TWEL = 0.01 ($10 budget), entry qty*price = $20.
            // With snapped balance = 1000: budget = $10, entry $20 gets trimmed/gated.
            // With raw balance = 500: budget = $5, entry $20 gets trimmed even more.
            // Verify the gating uses raw by checking the resulting entry qty.
            let mut sym = make_basic_symbol(0);
            sym.order_book = OrderBook {
                bid: 100.0,
                ask: 100.0,
            };
            sym.long.bot_params.wallet_exposure_limit = 1.0;
            sym.long.bot_params.total_wallet_exposure_limit = 0.01;
            sym.long.bot_params.n_positions = 1;
            sym.long.bot_params.entry_initial_qty_pct = 1.0;
            sym.exchange.qty_step = 0.001;
            sym.exchange.min_qty = 0.001;

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 0.01;
            global_bp.long.n_positions = 1;

            // With raw balance 500: TWEL budget = 500*0.01 = $5, so max qty = 5/100 = 0.05
            // With snapped balance 1000: TWEL budget = 1000*0.01 = $10, so max qty = 10/100 = 0.1
            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 500.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };
            let out = compute_ideal_orders(&input).unwrap();
            let entry_orders: Vec<_> = out
                .orders
                .iter()
                .filter(|o| {
                    matches!(
                        o.order_type,
                        OrderType::EntryGridNormalLong
                            | OrderType::EntryInitialNormalLong
                            | OrderType::EntryInitialPartialLong
                    )
                })
                .collect();
            if !entry_orders.is_empty() {
                // If entries exist, their total cost must fit within raw balance budget ($5)
                let total_cost: f64 = entry_orders.iter().map(|o| o.qty * o.price).sum();
                assert!(
                    total_cost <= 500.0 * 0.01 + 1e-6,
                    "Entry cost {:.4} should be gated by raw balance budget (500*0.01=5), not snapped (1000*0.01=10)",
                    total_cost
                );
            }
        }

        #[test]
        fn realized_loss_gate_does_not_block_panic_orders() {
            let mut sym = make_basic_symbol(0);
            sym.long.mode = Some(TradingMode::Panic);
            sym.long.position = Position {
                size: 2.0,
                price: 100.0,
            };
            sym.order_book = OrderBook {
                bid: 50.0,
                ask: 50.0,
            };

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    max_realized_loss_pct: 0.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };
            let out = compute_ideal_orders(&input).unwrap();
            assert_eq!(out.orders.len(), 1);
            assert_eq!(out.orders[0].order_type, OrderType::ClosePanicLong);
            assert!(
                out.diagnostics.loss_gate_blocks.is_empty(),
                "panic orders should bypass realized-loss gate"
            );
        }

        #[test]
        fn panic_emits_single_close_only() {
            let mut sym = make_basic_symbol(0);
            sym.long.mode = Some(TradingMode::Panic);
            sym.long.position = Position {
                size: 1.0,
                price: 100.0,
            };
            sym.order_book = OrderBook {
                bid: 100.0,
                ask: 101.0,
            };

            let mut global_bp = BotParamsPair::default();
            global_bp.long.total_wallet_exposure_limit = 1.0;
            global_bp.long.n_positions = 1;

            let input = OrchestratorInput {
                balance: 1000.0,
                balance_raw: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    market_orders_allowed: false,
                    market_order_near_touch_threshold: 0.001,
                    panic_close_market: false,
                    unstuck_allowance_long: 1000.0,
                    unstuck_allowance_short: 1000.0,
                    max_realized_loss_pct: 1.0,
                    realized_pnl_cumsum_max: 0.0,
                    realized_pnl_cumsum_last: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
                    strategy_kind: StrategyKind::TrailingGrid,
                },
                symbols: vec![sym],
                peek_hints: None,
            };
            let out = compute_ideal_orders(&input).unwrap();
            assert_eq!(out.orders.len(), 1);
            assert_eq!(out.orders[0].order_type, OrderType::ClosePanicLong);
        }
    }
}

pub use core::*;
