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
    use crate::closes::{
        calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
    };
    use crate::coin_selection::{select_coins, CoinFeature, SelectionConfig};
    use crate::constants::{LONG, SHORT};
    use crate::entries::{
        calc_entries_long, calc_entries_short, calc_min_entry_qty, calc_next_entry_long,
        calc_next_entry_short,
    };
    use crate::risk::{
        calc_twel_enforcer_actions, calc_unstucking_action, GateEntriesPosition,
        TwelEnforcerInputPosition, UnstuckPositionInput,
    };
    use crate::types::{
        BotParams, BotParamsPair, EMABands, ExchangeParams, OrderBook, OrderType, Position,
        StateParams, TrailingPriceBundle,
    };
    use crate::utils::{
        calc_new_psize_pprice, calc_order_price_diff_ask, calc_order_price_diff_bid,
        calc_pside_price_diff_int, calc_wallet_exposure, round_, round_dn,
    };
    use serde::{Deserialize, Serialize};

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
    }

    #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct OrchestratorOutput {
        pub orders: Vec<IdealOrder>,
        pub diagnostics: OrchestratorDiagnostics,
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
        pub unstuck_allowance_long: f64,
        pub unstuck_allowance_short: f64,
        /// If true, output orders are globally sorted by the canonical (live-bot) distance metric.
        /// Backtest does not require this global ordering and may disable it for performance.
        pub sort_global: bool,
        /// Global bot params (not modifiable by per-coin overrides).
        pub global_bot_params: BotParamsPair,
        /// If false (one-way mode), only one position side can exist per coin at a time.
        /// When no position exists on either side, the side closer to its EMA entry band wins.
        #[serde(default = "default_hedge_mode")]
        pub hedge_mode: bool,
    }

    fn default_hedge_mode() -> bool {
        true
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
        pub balance: f64,
        pub global: OrchestratorGlobal,
        pub symbols: Vec<SymbolInput>,
        /// Backtest-only performance hint: allow next-only vs full-grid expansion.
        pub peek_hints: Option<super::EntryPeekHints>,
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
                | CloseGridShort
                | CloseTrailingShort
                | CloseUnstuckShort
                | CloseAutoReduceTwelShort
                | ClosePanicShort
                | CloseAutoReduceWelShort
        )
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
        bot: &BotParams,
    ) -> Result<EMABands, OrchestratorError> {
        let ema0 = ema_lookup(&emas.m1.close, bot.ema_span_0)
            .ok_or(OrchestratorError::MissingEma { symbol_idx })?;
        let ema1 = ema_lookup(&emas.m1.close, bot.ema_span_1)
            .ok_or(OrchestratorError::MissingEma { symbol_idx })?;
        let ema2_span = (bot.ema_span_0 * bot.ema_span_1).sqrt();
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
        bot: &BotParams,
    ) -> Result<f64, OrchestratorError> {
        let span = bot.entry_volatility_ema_span_hours;
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

    fn effective_min_cost_is_low_enough(
        balance: f64,
        filter_enabled: bool,
        effective_min_cost: f64,
        bot: &BotParams,
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
        let base_limit = bot.wallet_exposure_limit;
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

    #[inline]
    fn would_fill_next_candle(next_low: f64, next_high: f64, qty: f64, price: f64) -> bool {
        if qty > 0.0 {
            next_low < price
        } else if qty < 0.0 {
            next_high > price
        } else {
            false
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

    fn build_forager_features_into(
        symbols: &[SymbolInput],
        pside: PositionSide,
        span_volume: f64,
        span_volatility: f64,
        filter_enabled: bool,
        balance: f64,
        out: &mut Vec<CoinFeature>,
    ) {
        out.clear();
        out.reserve(symbols.len());
        for s in symbols {
            let side = match pside {
                PositionSide::Long => &s.long,
                PositionSide::Short => &s.short,
            };
            // For selection of coins to occupy available slots for initial entries:
            // - We rank across all coins (including those with positions), matching legacy.
            // - We exclude modes which categorically block initial entries when `psize == 0.0`.
            let mode_no_pos = effective_mode(side.mode, false);
            let can_open_initial = should_generate_entries(mode_no_pos, false, true);
            let enabled = s.tradable
                && can_open_initial
                && effective_min_cost_is_low_enough(
                    balance,
                    filter_enabled,
                    s.effective_min_cost,
                    &side.bot_params,
                );
            let volume_score = ema_lookup(&s.emas.m1.volume, span_volume).unwrap_or(0.0);
            let volatility_score = ema_lookup(&s.emas.m1.log_range, span_volatility).unwrap_or(0.0);
            out.push(CoinFeature {
                index: s.symbol_idx,
                enabled,
                volume_score,
                volatility_score,
            });
        }
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
        features: Vec<CoinFeature>,
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
                if !actives_long[*idx] {
                    actives_long[*idx] = true;
                    actives_long_count += 1;
                }
            }
            if actives_long_count < enp_long {
                // Preferred coin ordering for filling remaining empty slots (volume -> clip -> volatility),
                // matching the legacy backtest selection.
                build_forager_features_into(
                    &input.symbols,
                    PositionSide::Long,
                    input.global.global_bot_params.long.filter_volume_ema_span,
                    input
                        .global
                        .global_bot_params
                        .long
                        .filter_volatility_ema_span,
                    input.global.filter_by_min_effective_cost,
                    input.balance,
                    &mut workspace.features,
                );
                let cfg = SelectionConfig {
                    max_positions: enp_long,
                    volume_drop_pct: input.global.global_bot_params.long.filter_volume_drop_pct,
                    volatility_drop_pct: input
                        .global
                        .global_bot_params
                        .long
                        .filter_volatility_drop_pct,
                    require_forager: true,
                };
                for idx in select_coins(&workspace.features, &cfg) {
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
                if !actives_short[*idx] {
                    actives_short[*idx] = true;
                    actives_short_count += 1;
                }
            }
            if actives_short_count < enp_short {
                build_forager_features_into(
                    &input.symbols,
                    PositionSide::Short,
                    input.global.global_bot_params.short.filter_volume_ema_span,
                    input
                        .global
                        .global_bot_params
                        .short
                        .filter_volatility_ema_span,
                    input.global.filter_by_min_effective_cost,
                    input.balance,
                    &mut workspace.features,
                );
                let cfg = SelectionConfig {
                    max_positions: enp_short,
                    volume_drop_pct: input.global.global_bot_params.short.filter_volume_drop_pct,
                    volatility_drop_pct: input
                        .global
                        .global_bot_params
                        .short
                        .filter_volatility_drop_pct,
                    require_forager: true,
                };
                for idx in select_coins(&workspace.features, &cfg) {
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
                    let ema_bands_long = derive_ema_bands(idx, &s.emas, &s.long.bot_params);
                    let ema_bands_short = derive_ema_bands(idx, &s.emas, &s.short.bot_params);

                    if let (Ok(bands_long), Ok(bands_short)) = (ema_bands_long, ema_bands_short) {
                        let entry_threshold_long =
                            bands_long.lower * (1.0 - s.long.bot_params.entry_initial_ema_dist);
                        let entry_threshold_short =
                            bands_short.upper * (1.0 + s.short.bot_params.entry_initial_ema_dist);

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
                        let ema_bands =
                            derive_ema_bands(s.symbol_idx, &s.emas, &s.long.bot_params)?;
                        let entry_volatility_logrange_ema_1h =
                            derive_entry_volatility_logrange_ema_1h(
                                s.symbol_idx,
                                &s.emas,
                                &s.long.bot_params,
                            )?;
                        let state = StateParams {
                            balance: input.balance,
                            order_book: s.order_book.clone(),
                            ema_bands,
                            entry_volatility_logrange_ema_1h,
                        };

                        if wants_entries {
                            if let Some(hints) = input.peek_hints.as_ref() {
                                let expand_entries = hints.expand_grid_long.contains(&s.symbol_idx);
                                if expand_entries {
                                    for e in calc_entries_long(
                                        &s.exchange,
                                        &state,
                                        &s.long.bot_params,
                                        &s.long.position,
                                        &s.long.trailing,
                                    ) {
                                        if e.qty != 0.0 {
                                            entries.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Long,
                                                qty: e.qty,
                                                price: e.price,
                                                order_type: e.order_type,
                                            });
                                        }
                                    }
                                } else {
                                    let e = calc_next_entry_long(
                                        &s.exchange,
                                        &state,
                                        &s.long.bot_params,
                                        &s.long.position,
                                        &s.long.trailing,
                                    );
                                    if e.qty != 0.0 {
                                        entries.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Long,
                                            qty: e.qty,
                                            price: e.price,
                                            order_type: e.order_type,
                                        });
                                    }
                                }
                            } else if let Some(nc) = s.next_candle.as_ref() {
                                let e = calc_next_entry_long(
                                    &s.exchange,
                                    &state,
                                    &s.long.bot_params,
                                    &s.long.position,
                                    &s.long.trailing,
                                );
                                let expand_entries = nc.tradable
                                    && would_fill_next_candle(nc.low, nc.high, e.qty, e.price);
                                if expand_entries {
                                    for e in calc_entries_long(
                                        &s.exchange,
                                        &state,
                                        &s.long.bot_params,
                                        &s.long.position,
                                        &s.long.trailing,
                                    ) {
                                        if e.qty != 0.0 {
                                            entries.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Long,
                                                qty: e.qty,
                                                price: e.price,
                                                order_type: e.order_type,
                                            });
                                        }
                                    }
                                } else if e.qty != 0.0 {
                                    entries.push(IdealOrder {
                                        symbol_idx: s.symbol_idx,
                                        pside: PositionSide::Long,
                                        qty: e.qty,
                                        price: e.price,
                                        order_type: e.order_type,
                                    });
                                }
                            } else {
                                for e in calc_entries_long(
                                    &s.exchange,
                                    &state,
                                    &s.long.bot_params,
                                    &s.long.position,
                                    &s.long.trailing,
                                ) {
                                    if e.qty != 0.0 {
                                        entries.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Long,
                                            qty: e.qty,
                                            price: e.price,
                                            order_type: e.order_type,
                                        });
                                    }
                                }
                            }
                        }

                        if wants_closes {
                            if let Some(hints) = input.peek_hints.as_ref() {
                                let expand_closes = hints.expand_close_long.contains(&s.symbol_idx);
                                if expand_closes {
                                    for c in calc_closes_long(
                                        &s.exchange,
                                        &state,
                                        &s.long.bot_params,
                                        &s.long.position,
                                        &s.long.trailing,
                                    ) {
                                        if c.qty != 0.0 {
                                            closes.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Long,
                                                qty: c.qty,
                                                price: c.price,
                                                order_type: c.order_type,
                                            });
                                        }
                                    }
                                } else {
                                    let c = calc_next_close_long(
                                        &s.exchange,
                                        &state,
                                        &s.long.bot_params,
                                        &s.long.position,
                                        &s.long.trailing,
                                    );
                                    if c.qty != 0.0 {
                                        closes.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Long,
                                            qty: c.qty,
                                            price: c.price,
                                            order_type: c.order_type,
                                        });
                                    }
                                }
                            } else if let Some(nc) = s.next_candle.as_ref() {
                                let c = calc_next_close_long(
                                    &s.exchange,
                                    &state,
                                    &s.long.bot_params,
                                    &s.long.position,
                                    &s.long.trailing,
                                );
                                let expand_closes = nc.tradable
                                    && would_fill_next_candle(nc.low, nc.high, c.qty, c.price);
                                if expand_closes {
                                    for c in calc_closes_long(
                                        &s.exchange,
                                        &state,
                                        &s.long.bot_params,
                                        &s.long.position,
                                        &s.long.trailing,
                                    ) {
                                        if c.qty != 0.0 {
                                            closes.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Long,
                                                qty: c.qty,
                                                price: c.price,
                                                order_type: c.order_type,
                                            });
                                        }
                                    }
                                } else if c.qty != 0.0 {
                                    closes.push(IdealOrder {
                                        symbol_idx: s.symbol_idx,
                                        pside: PositionSide::Long,
                                        qty: c.qty,
                                        price: c.price,
                                        order_type: c.order_type,
                                    });
                                }
                            } else {
                                for c in calc_closes_long(
                                    &s.exchange,
                                    &state,
                                    &s.long.bot_params,
                                    &s.long.position,
                                    &s.long.trailing,
                                ) {
                                    if c.qty != 0.0 {
                                        closes.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Long,
                                            qty: c.qty,
                                            price: c.price,
                                            order_type: c.order_type,
                                        });
                                    }
                                }
                            }
                        }
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
                        let ema_bands =
                            derive_ema_bands(s.symbol_idx, &s.emas, &s.short.bot_params)?;
                        let entry_volatility_logrange_ema_1h =
                            derive_entry_volatility_logrange_ema_1h(
                                s.symbol_idx,
                                &s.emas,
                                &s.short.bot_params,
                            )?;
                        let state = StateParams {
                            balance: input.balance,
                            order_book: s.order_book.clone(),
                            ema_bands,
                            entry_volatility_logrange_ema_1h,
                        };

                        if wants_entries {
                            if let Some(hints) = input.peek_hints.as_ref() {
                                let expand_entries =
                                    hints.expand_grid_short.contains(&s.symbol_idx);
                                if expand_entries {
                                    for e in calc_entries_short(
                                        &s.exchange,
                                        &state,
                                        &s.short.bot_params,
                                        &s.short.position,
                                        &s.short.trailing,
                                    ) {
                                        if e.qty != 0.0 {
                                            entries.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Short,
                                                qty: e.qty,
                                                price: e.price,
                                                order_type: e.order_type,
                                            });
                                        }
                                    }
                                } else {
                                    let e = calc_next_entry_short(
                                        &s.exchange,
                                        &state,
                                        &s.short.bot_params,
                                        &s.short.position,
                                        &s.short.trailing,
                                    );
                                    if e.qty != 0.0 {
                                        entries.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Short,
                                            qty: e.qty,
                                            price: e.price,
                                            order_type: e.order_type,
                                        });
                                    }
                                }
                            } else if let Some(nc) = s.next_candle.as_ref() {
                                let e = calc_next_entry_short(
                                    &s.exchange,
                                    &state,
                                    &s.short.bot_params,
                                    &s.short.position,
                                    &s.short.trailing,
                                );
                                let expand_entries = nc.tradable
                                    && would_fill_next_candle(nc.low, nc.high, e.qty, e.price);
                                if expand_entries {
                                    for e in calc_entries_short(
                                        &s.exchange,
                                        &state,
                                        &s.short.bot_params,
                                        &s.short.position,
                                        &s.short.trailing,
                                    ) {
                                        if e.qty != 0.0 {
                                            entries.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Short,
                                                qty: e.qty,
                                                price: e.price,
                                                order_type: e.order_type,
                                            });
                                        }
                                    }
                                } else if e.qty != 0.0 {
                                    entries.push(IdealOrder {
                                        symbol_idx: s.symbol_idx,
                                        pside: PositionSide::Short,
                                        qty: e.qty,
                                        price: e.price,
                                        order_type: e.order_type,
                                    });
                                }
                            } else {
                                for e in calc_entries_short(
                                    &s.exchange,
                                    &state,
                                    &s.short.bot_params,
                                    &s.short.position,
                                    &s.short.trailing,
                                ) {
                                    if e.qty != 0.0 {
                                        entries.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Short,
                                            qty: e.qty,
                                            price: e.price,
                                            order_type: e.order_type,
                                        });
                                    }
                                }
                            }
                        }

                        if wants_closes {
                            if let Some(hints) = input.peek_hints.as_ref() {
                                let expand_closes =
                                    hints.expand_close_short.contains(&s.symbol_idx);
                                if expand_closes {
                                    for c in calc_closes_short(
                                        &s.exchange,
                                        &state,
                                        &s.short.bot_params,
                                        &s.short.position,
                                        &s.short.trailing,
                                    ) {
                                        if c.qty != 0.0 {
                                            closes.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Short,
                                                qty: c.qty,
                                                price: c.price,
                                                order_type: c.order_type,
                                            });
                                        }
                                    }
                                } else {
                                    let c = calc_next_close_short(
                                        &s.exchange,
                                        &state,
                                        &s.short.bot_params,
                                        &s.short.position,
                                        &s.short.trailing,
                                    );
                                    if c.qty != 0.0 {
                                        closes.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Short,
                                            qty: c.qty,
                                            price: c.price,
                                            order_type: c.order_type,
                                        });
                                    }
                                }
                            } else if let Some(nc) = s.next_candle.as_ref() {
                                let c = calc_next_close_short(
                                    &s.exchange,
                                    &state,
                                    &s.short.bot_params,
                                    &s.short.position,
                                    &s.short.trailing,
                                );
                                let expand_closes = nc.tradable
                                    && would_fill_next_candle(nc.low, nc.high, c.qty, c.price);
                                if expand_closes {
                                    for c in calc_closes_short(
                                        &s.exchange,
                                        &state,
                                        &s.short.bot_params,
                                        &s.short.position,
                                        &s.short.trailing,
                                    ) {
                                        if c.qty != 0.0 {
                                            closes.push(IdealOrder {
                                                symbol_idx: s.symbol_idx,
                                                pside: PositionSide::Short,
                                                qty: c.qty,
                                                price: c.price,
                                                order_type: c.order_type,
                                            });
                                        }
                                    }
                                } else if c.qty != 0.0 {
                                    closes.push(IdealOrder {
                                        symbol_idx: s.symbol_idx,
                                        pside: PositionSide::Short,
                                        qty: c.qty,
                                        price: c.price,
                                        order_type: c.order_type,
                                    });
                                }
                            } else {
                                for c in calc_closes_short(
                                    &s.exchange,
                                    &state,
                                    &s.short.bot_params,
                                    &s.short.position,
                                    &s.short.trailing,
                                ) {
                                    if c.qty != 0.0 {
                                        closes.push(IdealOrder {
                                            symbol_idx: s.symbol_idx,
                                            pside: PositionSide::Short,
                                            qty: c.qty,
                                            price: c.price,
                                            order_type: c.order_type,
                                        });
                                    }
                                }
                            }
                        }
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
            let enabled = bot.unstuck_loss_allowance_pct > 0.0
                && bot.unstuck_close_pct > 0.0
                && bot.unstuck_threshold > 0.0;
            if !enabled {
                continue;
            }
            let ema_bands = derive_ema_bands(s.symbol_idx, &sym.emas, bot)?;
            workspace.unstuck_inputs.push(UnstuckPositionInput {
                idx: s.symbol_idx,
                side: LONG,
                position_size: s.pos.size,
                position_price: s.pos.price,
                wallet_exposure_limit: bot.wallet_exposure_limit,
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
            let enabled = bot.unstuck_loss_allowance_pct > 0.0
                && bot.unstuck_close_pct > 0.0
                && bot.unstuck_threshold > 0.0;
            if !enabled {
                continue;
            }
            let ema_bands = derive_ema_bands(s.symbol_idx, &sym.emas, bot)?;
            workspace.unstuck_inputs.push(UnstuckPositionInput {
                idx: s.symbol_idx,
                side: SHORT,
                position_size: s.pos.size,
                position_price: s.pos.price,
                wallet_exposure_limit: bot.wallet_exposure_limit,
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
            input.balance,
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
                let bot = &sym.long.bot_params;
                workspace.twel_positions.push(TwelEnforcerInputPosition {
                    idx: s.symbol_idx,
                    position_size: s.pos.size,
                    position_price: s.pos.price,
                    market_price: sym.order_book.bid,
                    base_wallet_exposure_limit: bot.wallet_exposure_limit,
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
                input.balance,
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
                let bot = &sym.short.bot_params;
                workspace.twel_positions.push(TwelEnforcerInputPosition {
                    idx: s.symbol_idx,
                    position_size: s.pos.size,
                    position_price: s.pos.price,
                    market_price: sym.order_book.ask,
                    base_wallet_exposure_limit: bot.wallet_exposure_limit,
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
                input.balance,
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
                input.balance,
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
                input.balance,
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
            emas.m1.log_range.push((10.0_f64, 1.0));
            emas.h1.log_range.push((1.0_f64, 1.0));

            let mut bp = BotParams::default();
            bp.ema_span_0 = 10.0;
            bp.ema_span_1 = 20.0;
            bp.entry_volatility_ema_span_hours = 1.0;
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
                },
                short: SymbolSideInput {
                    mode: None,
                    position: Position::default(),
                    trailing: TrailingPriceBundle::default(),
                    bot_params: bp,
                },
            }
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
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    sort_global: true,
                    global_bot_params: {
                        let mut pair = BotParamsPair::default();
                        pair.long.total_wallet_exposure_limit = 1000.0;
                        pair.long.n_positions = 1;
                        pair
                    },
                    hedge_mode: true,
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
            global_bp.long.filter_volume_drop_pct = 0.0;
            global_bp.long.filter_volatility_drop_pct = 0.0;

            let input = OrchestratorInput {
                balance: 1000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
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
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: false,
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
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: false,
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
        fn non_contiguous_symbol_idx_is_rejected() {
            let mut sym0 = make_basic_symbol(0);
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
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
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
            global_bp.long.filter_volume_drop_pct = 0.0;
            global_bp.long.filter_volatility_drop_pct = 0.0;
            global_bp.long.filter_volume_ema_span = 10.0;
            global_bp.long.filter_volatility_ema_span = 10.0;
            // disable short for this test
            global_bp.short.total_wallet_exposure_limit = 0.0;
            global_bp.short.n_positions = 0;

            let input = OrchestratorInput {
                balance: 1_000_000.0,
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 0.0,
                    unstuck_allowance_short: 0.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
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
                global: OrchestratorGlobal {
                    filter_by_min_effective_cost: false,
                    unstuck_allowance_long: 1000.0,
                    unstuck_allowance_short: 1000.0,
                    sort_global: true,
                    global_bot_params: global_bp,
                    hedge_mode: true,
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
