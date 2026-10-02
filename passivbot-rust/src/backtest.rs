#[path = "backtest_hsl_analysis.rs"]
mod hsl_analysis;
#[path = "backtest_hsl_cache.rs"]
mod hsl_cache;
#[path = "backtest_hsl.rs"]
mod hsl_inputs;
#[path = "backtest_hsl_report.rs"]
mod hsl_report;
#[path = "backtest_hsl_runtime.rs"]
pub(crate) mod hsl_runtime;

use crate::analysis::{analyze_equity_series, calc_fill_activity_metrics, FillActivityMetrics};
use crate::constants::{CLOSE, HIGH, LONG, LOW, SHORT, VOLUME};
use crate::entries::{
    calc_min_entry_qty, effective_we_excess_allowance_pct,
    wallet_exposure_limit_with_allowance_from_base,
};
use crate::orchestrator;
use crate::orchestrator::{
    EmaBundle as OrchestratorEmaBundle, EmaTimeframeBundle as OrchestratorEmaTimeframeBundle,
    EntryPeekHints, ForagerHysteresisState,
};
use crate::strategies::{
    parse_strategy_params, strategy_ema_spans, strategy_entry_volatility_span_hours,
    strategy_has_trailing, strategy_initial_qty_pct, strategy_needs_log_range_1h,
    strategy_needs_log_range_1m, strategy_offset_volatility_span_minutes, StrategyParams,
};
use crate::trailing::{reset_trailing_bundle, update_trailing_bundle_with_candle};
use crate::types::{
    BacktestParams, Balance, BotParams, BotParamsPair, EMABands, Equities,
    EquityHardStopLossConfig, ExchangeParams, Fill, Order, OrderBook, OrderType, Position,
    RuntimeBudgetState, RuntimeBudgetStatePair, StrategyParamsPairValue, TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_wallet_exposure, hysteresis, qty_to_cost, round_, round_dn, round_up,
};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::HashSet;

// Orchestrator-only: legacy backtest order-generation path removed in this branch.
const DEBUG_DUMP_ORDERS: bool = false;
const DEBUG_TRACE_BALANCE: bool = false;
const DEBUG_DUMP_UNSTUCK_CALC: bool = false;
// Runtime profiler for the orchestrator path. Enable by setting env var `PASSIVBOT_ORCH_PROFILE=1`.
const ORCH_PROFILE_ENV: &str = "PASSIVBOT_ORCH_PROFILE";
// Limit debug snapshots to a narrow window in the backtest.
const DEBUG_MAX_STEPS: usize = 0;
// Optional extra window to capture debug orders beyond DEBUG_MAX_STEPS (inclusive bounds).
// Set to None to disable.
const DEBUG_EXTRA_WINDOW: Option<(usize, usize)> = None;
// Optional coin filter for debug dumps (by coin name, e.g. Some("LINK")); None dumps all coins.
const DEBUG_COIN_FILTER: Option<&str> = None;
// Optional window for unstuck debug dump (inclusive bounds). Set to None to disable.
const DEBUG_UNSTUCK_WINDOW: Option<(usize, usize)> = None;
// Optional coin filter for unstuck debug dump (by coin name, e.g. Some("LINK")); None dumps all.
const DEBUG_UNSTUCK_COIN_FILTER: Option<&str> = None;
// Optional window for balance trace debug (inclusive bounds). Set to None to disable.
const DEBUG_TRACE_WINDOW: Option<(usize, usize)> = None;
// Optional coin filter for balance trace debug (by coin name, e.g. Some("SOL")); None dumps all.
const DEBUG_TRACE_COIN_FILTER: Option<&str> = None;
use ndarray::{ArrayView1, ArrayView3};
use serde_json;
use std::collections::VecDeque;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Serialize)]
struct UnstuckCalcDebug {
    step: usize,
    coin: String,
    idx: usize,
    side: usize,
    balance: f64,
    balance_bits: u64,
    allowance: f64,
    allowance_bits: u64,
    position_size: f64,
    position_size_bits: u64,
    position_price: f64,
    position_price_bits: u64,
    current_price: f64,
    current_price_bits: u64,
    ema_band_upper: f64,
    ema_band_upper_bits: u64,
    ema_band_lower: f64,
    ema_band_lower_bits: u64,
    wallet_exposure_limit: f64,
    wallet_exposure_limit_bits: u64,
    risk_we_excess_allowance_pct: f64,
    unstuck_threshold: f64,
    unstuck_close_pct: f64,
    unstuck_ema_dist: f64,
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    effective_wel: f64,
    effective_wel_bits: u64,
    wallet_exposure: f64,
    wallet_exposure_bits: u64,
    ema_price_target: f64,
    ema_price_target_bits: u64,
    ema_price_rounded: f64,
    ema_price_rounded_bits: u64,
    meets_trigger: bool,
    min_entry_qty: f64,
    min_entry_qty_bits: u64,
    target_qty_raw: f64,
    target_qty_raw_bits: u64,
    target_qty_dn: f64,
    target_qty_dn_bits: u64,
    close_qty_pre_allowance: f64,
    close_qty_pre_allowance_bits: u64,
    pnl_if_closed: f64,
    pnl_if_closed_bits: u64,
    close_qty_final: f64,
    close_qty_final_bits: u64,
}

fn calc_effective_min_cost(price: f64, exchange: &ExchangeParams) -> f64 {
    qty_to_cost(calc_min_entry_qty(price, exchange), price, exchange.c_mult)
}

#[derive(Clone, Default, Copy, Debug)]
pub struct EmaAlphas {
    pub long: Alphas,
    pub short: Alphas,
    pub unstuck_long: Alphas,
    pub unstuck_short: Alphas,
    pub vol_alpha_long: f64,
    pub vol_alpha_short: f64,
    pub log_range_alpha_long: f64,
    pub log_range_alpha_short: f64,
    pub volatility_ema_1m_alpha_long: f64,
    pub volatility_ema_1m_alpha_short: f64,
    pub volatility_ema_1h_alpha_long: f64,
    pub volatility_ema_1h_alpha_short: f64,
}

#[derive(Clone, Default, Copy, Debug)]
pub struct Alphas {
    pub alphas: [f64; 3],
}

#[derive(Debug)]
pub struct EMAs {
    pub unstuck_long: [f64; 3],
    pub unstuck_long_num: [f64; 3],
    pub unstuck_long_den: [f64; 3],
    pub unstuck_short: [f64; 3],
    pub unstuck_short_num: [f64; 3],
    pub unstuck_short_den: [f64; 3],
    pub long: [f64; 3],
    pub long_num: [f64; 3],
    pub long_den: [f64; 3],
    pub short: [f64; 3],
    pub short_num: [f64; 3],
    pub short_den: [f64; 3],
    pub vol_long: f64,
    pub vol_long_num: f64,
    pub vol_long_den: f64,
    pub vol_short: f64,
    pub vol_short_num: f64,
    pub vol_short_den: f64,
    pub log_range_long: f64,
    pub log_range_long_num: f64,
    pub log_range_long_den: f64,
    pub log_range_short: f64,
    pub log_range_short_num: f64,
    pub log_range_short_den: f64,
    pub volatility_ema_1m_long: f64,
    pub volatility_ema_1m_long_num: f64,
    pub volatility_ema_1m_long_den: f64,
    pub volatility_ema_1m_short: f64,
    pub volatility_ema_1m_short_num: f64,
    pub volatility_ema_1m_short_den: f64,
    pub volatility_ema_1h_long: f64,
    pub volatility_ema_1h_long_num: f64,
    pub volatility_ema_1h_long_den: f64,
    pub volatility_ema_1h_short: f64,
    pub volatility_ema_1h_short_num: f64,
    pub volatility_ema_1h_short_den: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct HourBucket {
    pub high: f64,
    pub low: f64,
}

impl Default for HourBucket {
    fn default() -> Self {
        HourBucket {
            high: 0.0,
            low: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EffectiveNPositions {
    pub long: usize,
    pub short: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct StrategyParamsPair {
    long: StrategyParams,
    short: StrategyParams,
}

#[derive(Debug, Clone, Copy)]
struct OrchestratorM1LogRangeSlots {
    forager_long: usize,
    forager_short: usize,
    offset_long: Option<usize>,
    offset_short: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct OrchestratorH1LogRangeSlots {
    long: Option<usize>,
    short: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct OrchestratorEmaSlots {
    m1_volume_long: usize,
    m1_volume_short: usize,
    m1_log_range: OrchestratorM1LogRangeSlots,
    h1_log_range: OrchestratorH1LogRangeSlots,
}

fn make_orchestrator_ema_slots(
    strategy_params: &StrategyParamsPair,
    bot_params_master: &BotParamsPair,
) -> OrchestratorEmaSlots {
    let _ = bot_params_master;
    let mut next_m1_log_range = 0usize;
    let forager_long = next_m1_log_range;
    next_m1_log_range += 1;
    let forager_short = next_m1_log_range;
    next_m1_log_range += 1;
    let offset_long =
        if strategy_offset_volatility_span_minutes(&strategy_params.long).unwrap_or(0.0) > 0.0 {
            let idx = next_m1_log_range;
            next_m1_log_range += 1;
            Some(idx)
        } else {
            None
        };
    let offset_short =
        if strategy_offset_volatility_span_minutes(&strategy_params.short).unwrap_or(0.0) > 0.0 {
            let idx = next_m1_log_range;
            Some(idx)
        } else {
            None
        };

    let mut next_h1_log_range = 0usize;
    let h1_long =
        if strategy_entry_volatility_span_hours(&strategy_params.long).unwrap_or(0.0) > 0.0 {
            let idx = next_h1_log_range;
            next_h1_log_range += 1;
            Some(idx)
        } else {
            None
        };
    let h1_short =
        if strategy_entry_volatility_span_hours(&strategy_params.short).unwrap_or(0.0) > 0.0 {
            let idx = next_h1_log_range;
            Some(idx)
        } else {
            None
        };

    OrchestratorEmaSlots {
        m1_volume_long: 0,
        m1_volume_short: 1,
        m1_log_range: OrchestratorM1LogRangeSlots {
            forager_long,
            forager_short,
            offset_long,
            offset_short,
        },
        h1_log_range: OrchestratorH1LogRangeSlots {
            long: h1_long,
            short: h1_short,
        },
    }
}

impl EMAs {
    pub fn compute_unstuck_bands(&self, pside: usize) -> EMABands {
        let (upper, lower) = match pside {
            LONG => (
                *self
                    .unstuck_long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                *self
                    .unstuck_long
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            ),
            SHORT => (
                *self
                    .unstuck_short
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                *self
                    .unstuck_short
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            ),
            _ => panic!("Invalid pside"),
        };
        EMABands { upper, lower }
    }
}

#[inline(always)]
fn update_adjusted_ema(value: f64, alpha: f64, numerator: &mut f64, denominator: &mut f64) -> f64 {
    if !value.is_finite() {
        return if *denominator > 0.0 {
            *numerator / *denominator
        } else {
            value
        };
    }
    if alpha <= 0.0 || !alpha.is_finite() {
        return if *denominator > 0.0 {
            *numerator / *denominator
        } else {
            value
        };
    }
    let one_minus_alpha = 1.0 - alpha;
    let new_num = alpha * value + one_minus_alpha * *numerator;
    let new_den = alpha + one_minus_alpha * *denominator;
    if !new_den.is_finite() || new_den <= f64::MIN_POSITIVE {
        *numerator = alpha * value;
        *denominator = alpha;
        return value;
    }
    *numerator = new_num;
    *denominator = new_den;
    new_num / new_den
}

#[derive(Debug)]
pub struct OpenOrders {
    pub long: Vec<OpenOrderBundle>,
    pub short: Vec<OpenOrderBundle>,
}

#[derive(Debug, Default)]
pub struct OpenOrderBundle {
    pub entries: Vec<BacktestOrder>,
    pub closes: Vec<BacktestOrder>,
}

#[derive(Debug, Clone)]
pub struct BacktestOrder {
    pub order: Order,
    pub execution_type: orchestrator::ExecutionType,
}

#[derive(Clone, Copy, Debug)]
struct RollingPnlEvent {
    k: usize,
    pnl: f64,
    abs_cumulative_after: f64,
    seq: usize,
}

#[derive(Clone, Copy, Debug)]
struct RollingPnlPeakCandidate {
    seq: usize,
    abs_cumulative_after: f64,
}

#[derive(Default, Debug)]
pub struct TrailingPrices {
    pub long: Vec<TrailingPriceBundle>,
    pub short: Vec<TrailingPriceBundle>,
}

#[derive(Debug)]
pub struct Positions {
    pub long: Vec<Position>,
    pub short: Vec<Position>,
}

impl OpenOrders {
    fn new(n_coins: usize) -> Self {
        Self {
            long: (0..n_coins).map(|_| OpenOrderBundle::default()).collect(),
            short: (0..n_coins).map(|_| OpenOrderBundle::default()).collect(),
        }
    }

    fn clear_all(&mut self) {
        for bundle in self.long.iter_mut() {
            bundle.entries.clear();
            bundle.closes.clear();
        }
        for bundle in self.short.iter_mut() {
            bundle.entries.clear();
            bundle.closes.clear();
        }
    }
}

impl TrailingPrices {
    fn new(n_coins: usize) -> Self {
        Self {
            long: vec![TrailingPriceBundle::default(); n_coins],
            short: vec![TrailingPriceBundle::default(); n_coins],
        }
    }
}

impl Positions {
    fn new(n_coins: usize) -> Self {
        Self {
            long: vec![Position::default(); n_coins],
            short: vec![Position::default(); n_coins],
        }
    }
}

pub struct TrailingEnabled {
    long: bool,
    short: bool,
}

#[derive(Debug)]
pub struct TradingEnabled {
    long: bool,
    short: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HardStopMetrics {
    pub triggers: u32,
    pub triggers_per_year: f64,
    pub triggers_long: u32,
    pub triggers_short: u32,
    pub halt_to_restart_equity_loss_pct: f64,
    pub restarts: u32,
    pub restarts_per_year: f64,
    pub restarts_per_year_long: f64,
    pub restarts_per_year_short: f64,
    pub restarts_long: u32,
    pub restarts_short: u32,
    pub time_in_red_pct: f64,
    pub duration_minutes_mean: f64,
    pub duration_minutes_max: f64,
    pub trigger_drawdown_mean: f64,
    pub panic_close_loss_sum: f64,
    pub panic_close_loss_max: f64,
    pub panic_close_loss_drawdown_pct_min: f64,
    pub panic_close_loss_drawdown_pct_mean: f64,
    pub panic_close_loss_drawdown_pct_max: f64,
    pub flatten_time_minutes_mean: f64,
    pub post_restart_retrigger_pct: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyEquityMetrics {
    pub gain_strategy_eq: f64,
    pub adg_strategy_eq: f64,
    pub adg_rolling_hmean_strategy_eq: f64,
    pub adg_time_integrated_strategy_eq: f64,
    pub positive_gain_participation_strategy_eq: f64,
    pub mdg_strategy_eq: f64,
    pub sharpe_ratio_strategy_eq: f64,
    pub sortino_ratio_strategy_eq: f64,
    pub omega_ratio_strategy_eq: f64,
    pub expected_shortfall_1pct_strategy_eq: f64,
    pub calmar_ratio_strategy_eq: f64,
    pub sterling_ratio_strategy_eq: f64,
    pub drawdown_worst_strategy_eq: f64,
    pub drawdown_worst_ema_strategy_eq: f64,
    pub drawdown_worst_mean_1pct_strategy_eq: f64,
    pub drawdown_worst_mean_1pct_ema_strategy_eq: f64,
    pub strategy_eq_underwater_pct_mean: f64,
    pub strategy_eq_underwater_pct_median: f64,
    pub strategy_eq_recovery_days_mean: f64,
    pub strategy_eq_recovery_days_median: f64,
    pub strategy_eq_recovery_days_p95: f64,
    pub strategy_eq_recovery_days_p99: f64,
    pub strategy_eq_recovery_days_mean_worst_5pct: f64,
    pub strategy_eq_recovery_days_mean_worst_1pct: f64,
    pub strategy_eq_recovery_days_max: f64,
    pub peak_recovery_hours_strategy_eq: f64,
    pub peak_recovery_days_strategy_eq: f64,
    pub adg_strategy_eq_w: f64,
    pub mdg_strategy_eq_w: f64,
    pub sharpe_ratio_strategy_eq_w: f64,
    pub sortino_ratio_strategy_eq_w: f64,
    pub omega_ratio_strategy_eq_w: f64,
    pub calmar_ratio_strategy_eq_w: f64,
    pub sterling_ratio_strategy_eq_w: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyEquityMetricsBundle {
    pub overall: StrategyEquityMetrics,
    pub long: StrategyEquityMetrics,
    pub short: StrategyEquityMetrics,
}

#[derive(Debug, Clone, Copy)]
struct OrderFillExecution {
    price: f64,
    fee_rate: f64,
    liquidity: &'static str,
}

// RollingSum (SMA) removed — volume & log range are now tracked via EMAs in `EMAs`.

pub struct Backtest<'a> {
    hlcvs: ArrayView3<'a, f64>,
    btc_usd_prices: ArrayView1<'a, f64>, // Change to ArrayView1 (1D view)
    active_coin_indices: Vec<usize>,
    interval_ms: u64,
    bot_params_master: BotParamsPair,
    bot_params: Vec<BotParamsPair>,
    bot_params_original: Vec<BotParamsPair>,
    strategy_params: Vec<StrategyParamsPair>,
    strategy_kind: crate::strategies::StrategyKind,
    runtime_budget: Vec<RuntimeBudgetStatePair>,
    configured_n_positions: EffectiveNPositions,
    effective_n_positions: EffectiveNPositions,
    exchange_params_list: Vec<ExchangeParams>,
    backtest_params: BacktestParams,
    pub balance: Balance,
    n_coins: usize,
    ema_alphas: Vec<EmaAlphas>,
    emas: Vec<EMAs>,
    unilateralness: Vec<Vec<(f64, crate::unilateralness::RollingRms)>>,
    unilateralness_step: Option<usize>,
    orchestrator_ema_slots: Vec<OrchestratorEmaSlots>,
    needs_volume_ema_long: bool,
    needs_volume_ema_short: bool,
    needs_log_range_long: bool,
    needs_log_range_short: bool,
    needs_volatility_ema_1m_long: bool,
    needs_volatility_ema_1m_short: bool,
    needs_volatility_ema_1h_long: bool,
    needs_volatility_ema_1h_short: bool,
    coin_first_valid_idx: Vec<usize>,
    coin_last_valid_idx: Vec<usize>,
    coin_trade_start_idx: Vec<usize>,
    trade_activation_logged: Vec<bool>,
    // Wall-clock timestamp (ms) of the first candle; assumes 1m spacing
    first_timestamp_ms: u64,
    // Latest computed hourly boundary (aligned to whole hours)
    last_hour_boundary_ms: u64,
    // Latest 1h bucket per coin (overwritten each new hour)
    latest_hour: Vec<HourBucket>,
    warmup_bars: usize,
    current_step: usize,
    positions: Positions,
    open_orders: OpenOrders,
    trailing_prices: TrailingPrices,
    pnl_cumsum_running: f64,
    pnl_cumsum_max: f64,
    pnl_cumsum_running_net: f64,
    pnl_cumsum_max_net: f64,
    pnl_cumsum_running_net_pside: [f64; 2],
    pnl_cumsum_running_net_coin_pside: [Vec<f64>; 2],
    pnl_cumsum_max_net_coin_pside: [Vec<f64>; 2],
    pnl_lookback_bars: usize,
    pnl_events: VecDeque<RollingPnlEvent>,
    pnl_event_seq: usize,
    rolling_pnl_peak_candidates: VecDeque<RollingPnlPeakCandidate>,
    coin_pnl_events: [Vec<VecDeque<RollingPnlEvent>>; 2],
    coin_pnl_event_seq: usize,
    coin_rolling_pnl_peak_candidates: [Vec<VecDeque<RollingPnlPeakCandidate>>; 2],
    fills: Vec<Fill>,
    trading_enabled: TradingEnabled,
    trailing_enabled: Vec<TrailingEnabled>,
    any_trailing_long: bool,
    any_trailing_short: bool,
    equities: Equities,
    last_valid_timestamps: Vec<Option<usize>>,
    did_fill_long: Vec<bool>,
    did_fill_short: Vec<bool>,
    last_increase_fill_timestamp_long: Vec<Option<u64>>,
    last_increase_fill_timestamp_short: Vec<Option<u64>>,
    pub total_wallet_exposures: Vec<f64>,
    // removed rolling_volume_sum & buffer — replaced by per-coin EMAs in `emas`
    equity_tracking_active: bool,
    debug_writer: Option<DebugOrderWriter>,
    debug_balance_writer: Option<DebugBalanceWriter>,
    orchestrator_input_cache: Option<orchestrator::OrchestratorInput>,
    orchestrator_workspace: orchestrator::OrchestratorWorkspace,
    orch_profile: Option<OrchProfile>,
    max_tradable_coins_seen: EffectiveNPositions,
    hsl_scopes: Vec<hsl_runtime::Scope>,
    hsl_cutoffs: hsl_cache::Cutoffs,
    hsl_traces: hsl_cache::Traces,
    hsl_report: hsl_report::Report,
    btc_collateral_initialized: bool,
    strategy_equity_series: Vec<f64>,
    strategy_equity_series_pside: [Vec<f64>; 2],
    strategy_equity_timestamps_ms_pside: [Vec<u64>; 2],
    liquidated: bool,
    final_hard_stop_metrics: Option<HardStopMetrics>,
    final_strategy_equity_metrics: Option<StrategyEquityMetricsBundle>,
}

#[derive(Debug, Serialize)]
struct DebugOrder {
    qty: f64,
    price: f64,
    order_type_id: u16,
    reduce_only: bool,
}

#[derive(Debug, Serialize)]
struct DebugOrderSnapshot {
    step: usize,
    side: &'static str,
    idx: usize,
    coin: String,
    stage: &'static str,
    pos_size: f64,
    pos_price: f64,
    close_price: f64,
    entries: Vec<DebugOrder>,
    closes: Vec<DebugOrder>,
}

struct DebugOrderWriter {
    writer: BufWriter<File>,
    has_entries: bool,
    closed: bool,
}

impl DebugOrderWriter {
    fn new_for_mode() -> Option<Self> {
        Self::new("debug_orders_orchestrator.json")
    }

    fn new(fname: &str) -> Option<Self> {
        let mut writer = BufWriter::new(File::create(fname).ok()?);
        writer.write_all(b"[").ok()?;
        Some(Self {
            writer,
            has_entries: false,
            closed: false,
        })
    }

    fn write_snapshot(&mut self, snapshot: &DebugOrderSnapshot) {
        if self.closed {
            return;
        }
        if self.has_entries {
            let _ = self.writer.write_all(b",");
        }
        if serde_json::to_writer(&mut self.writer, snapshot).is_ok() {
            let _ = self.writer.write_all(b"\n");
            self.has_entries = true;
        }
    }

    fn finish(&mut self) {
        if self.closed {
            return;
        }
        let _ = self.writer.write_all(b"]");
        let _ = self.writer.flush();
        self.closed = true;
    }
}

impl Drop for DebugOrderWriter {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Debug, Clone, Copy)]
struct BalanceSnapshot {
    usd_cash_wallet: f64,
    usd_total_balance: f64,
    usd_total_balance_rounded: f64,
    btc_cash_wallet: f64,
    btc_total_balance: f64,
}

#[derive(Debug, Serialize)]
struct DebugBalanceTraceRecord {
    step: usize,
    timestamp_ms: u64,
    coin: String,
    event: &'static str,
    order_type_id: u16,
    fill_qty: f64,
    fill_price: f64,
    pnl: f64,
    fee_paid: f64,
    btc_price: f64,
    usd_cash_wallet_before: f64,
    usd_cash_wallet_after: f64,
    usd_total_balance_before: f64,
    usd_total_balance_after: f64,
    usd_total_balance_rounded_before: f64,
    usd_total_balance_rounded_after: f64,
    btc_cash_wallet_before: f64,
    btc_cash_wallet_after: f64,
    btc_total_balance_before: f64,
    btc_total_balance_after: f64,
}

struct DebugBalanceWriter {
    writer: BufWriter<File>,
}

impl DebugBalanceWriter {
    fn new_for_mode() -> Option<Self> {
        Self::new("debug_balance_orchestrator.jsonl")
    }

    fn new(fname: &str) -> Option<Self> {
        Some(Self {
            writer: BufWriter::new(File::create(fname).ok()?),
        })
    }

    fn write_record(&mut self, record: &DebugBalanceTraceRecord) {
        if serde_json::to_writer(&mut self.writer, record).is_ok() {
            let _ = self.writer.write_all(b"\n");
        }
    }

    fn finish(&mut self) {
        let _ = self.writer.flush();
    }
}

impl Drop for DebugBalanceWriter {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Debug, Default, Serialize)]
struct OrchProfile {
    mode: &'static str,
    steps: u64,
    total_ns: u64,
    clear_orders_ns: u64,
    peek_hints_ns: u64,
    input_update_ns: u64,
    compute_ns: u64,
    distribute_ns: u64,
    sort_bundles_ns: u64,
}

impl OrchProfile {
    #[inline]
    fn add_ns(field: &mut u64, elapsed: std::time::Duration) {
        *field = field.saturating_add(elapsed.as_nanos() as u64);
    }

    fn write_to_path<P: AsRef<Path>>(&self, path: P) {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if create_dir_all(parent).is_err() {
                return;
            }
        }
        if let Ok(file) = File::create(path) {
            let mut w = BufWriter::new(file);
            let _ = serde_json::to_writer_pretty(&mut w, self);
            let _ = w.flush();
        }
    }

    fn write_to_file(&self) {
        self.write_to_path("measurements/orch_profile_orchestrator.json");
    }
}

fn calc_entry_balance_pct(
    params: &BotParams,
    effective_n_positions: usize,
    entry_initial_qty_pct: f64,
) -> f64 {
    if effective_n_positions == 0 {
        return 0.0;
    }
    let base_limit = params.total_wallet_exposure_limit / effective_n_positions as f64;
    let allowance_multiplier = 1.0 + effective_we_excess_allowance_pct(params, base_limit);
    base_limit * entry_initial_qty_pct * allowance_multiplier
}

fn configured_wallet_exposure_limit(params: &BotParams) -> f64 {
    if params.wallet_exposure_limit > 0.0 {
        params.wallet_exposure_limit
    } else if params.n_positions > 0 {
        params.total_wallet_exposure_limit / params.n_positions as f64
    } else {
        0.0
    }
}

fn make_runtime_budget_state(
    params: &BotParams,
    effective_n_positions: usize,
) -> RuntimeBudgetState {
    let configured_wallet_exposure_limit = configured_wallet_exposure_limit(params);
    RuntimeBudgetState {
        configured_wallet_exposure_limit,
        effective_wallet_exposure_limit: configured_wallet_exposure_limit,
        configured_n_positions: params.n_positions,
        effective_n_positions,
    }
}

fn parse_strategy_params_pair(
    strategy_kind: crate::strategies::StrategyKind,
    raw: &StrategyParamsPairValue,
    bot_params: &BotParamsPair,
) -> Result<StrategyParamsPair, String> {
    Ok(StrategyParamsPair {
        long: parse_strategy_params(
            strategy_kind,
            crate::strategies::StrategySide::Long,
            Some(&raw.long),
            &bot_params.long,
        )?,
        short: parse_strategy_params(
            strategy_kind,
            crate::strategies::StrategySide::Short,
            Some(&raw.short),
            &bot_params.short,
        )?,
    })
}

#[cfg(test)]
fn test_trailing_martingale_params_value_from_flat(bot_params: &BotParams) -> serde_json::Value {
    crate::strategies::TrailingMartingaleParams {
        volatility_ema_span_1h: bot_params.entry_volatility_ema_span_1h,
        volatility_ema_span_1m: bot_params.entry_volatility_ema_span_1m,
        entry: crate::strategies::TrailingMartingaleEntryParams {
            ema_span_0: bot_params.ema_span_0,
            ema_span_1: bot_params.ema_span_1,
            double_down_factor: bot_params.entry_grid_double_down_factor,
            ema_gate_mode: crate::strategies::EmaGateMode::Initial,
            initial_ema_dist: bot_params.entry_initial_ema_dist,
            initial_qty_pct: bot_params.entry_initial_qty_pct,
            threshold_base_pct: bot_params.entry_grid_spacing_pct,
            threshold_we_weight: bot_params.entry_we_weight,
            threshold_volatility_1h_weight: bot_params.entry_weight_volatility_1h,
            threshold_volatility_1m_weight: bot_params.entry_weight_volatility_1m,
            retracement_base_pct: bot_params.entry_trailing_retracement_pct,
            retracement_we_weight: bot_params.entry_we_weight,
            retracement_volatility_1h_weight: bot_params.entry_weight_volatility_1h,
            retracement_volatility_1m_weight: bot_params.entry_weight_volatility_1m,
        },
        close: crate::strategies::TrailingMartingaleCloseParams {
            qty_pct: bot_params.close_grid_qty_pct,
            threshold_base_pct: bot_params.close_trailing_threshold_pct,
            threshold_volatility_1h_weight: bot_params.close_weight_volatility_1h,
            threshold_volatility_1m_weight: bot_params.close_weight_volatility_1m,
            retracement_base_pct: bot_params.close_trailing_retracement_pct,
            retracement_volatility_1h_weight: bot_params.close_weight_volatility_1h,
            retracement_volatility_1m_weight: bot_params.close_weight_volatility_1m,
            ..Default::default()
        },
    }
    .to_value()
}

#[cfg(test)]
fn test_strategy_params_pair_value_from_flat(
    bot_params: &BotParamsPair,
) -> StrategyParamsPairValue {
    StrategyParamsPairValue {
        long: test_trailing_martingale_params_value_from_flat(&bot_params.long),
        short: test_trailing_martingale_params_value_from_flat(&bot_params.short),
    }
}

impl<'a> Backtest<'a> {
    #[inline]
    fn snapshot_balance(&self) -> BalanceSnapshot {
        BalanceSnapshot {
            usd_cash_wallet: self.balance.usd_cash_wallet,
            usd_total_balance: self.balance.usd_total_balance,
            usd_total_balance_rounded: self.balance.usd_total_balance_rounded,
            btc_cash_wallet: self.balance.btc_cash_wallet,
            btc_total_balance: self.balance.btc_total_balance,
        }
    }

    fn debug_dump_unstuck_calc(&mut self, k: usize, idx: usize, side: usize) {
        if !DEBUG_DUMP_UNSTUCK_CALC {
            return;
        }
        if DEBUG_UNSTUCK_WINDOW
            .map(|(start, end)| k < start || k > end)
            .unwrap_or(true)
        {
            return;
        }

        let coin = self
            .backtest_params
            .coins
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("idx_{idx}"));
        if let Some(want) = DEBUG_UNSTUCK_COIN_FILTER {
            if coin != want {
                return;
            }
        }

        let position = match side {
            LONG => self.positions.long[idx],
            SHORT => self.positions.short[idx],
            _ => return,
        };
        if position.size == 0.0 || !position.price.is_finite() || position.price <= 0.0 {
            return;
        }

        let balance = self.balance.usd_total_balance_rounded;
        let balance_raw = self.balance.usd_total_balance;
        let (effective_cumsum_max, effective_cumsum_last) = self.effective_pnl_cumsum(k);
        let bp = self.bp(idx, side);
        let allowance = match side {
            LONG => {
                if bp.unstuck_enabled && bp.unstuck_loss_allowance_pct > 0.0 {
                    calc_auto_unstuck_allowance(
                        balance_raw,
                        bp.unstuck_loss_allowance_pct * bp.total_wallet_exposure_limit,
                        effective_cumsum_max,
                        effective_cumsum_last,
                    )
                } else {
                    0.0
                }
            }
            SHORT => {
                if bp.unstuck_enabled && bp.unstuck_loss_allowance_pct > 0.0 {
                    calc_auto_unstuck_allowance(
                        balance_raw,
                        bp.unstuck_loss_allowance_pct * bp.total_wallet_exposure_limit,
                        effective_cumsum_max,
                        effective_cumsum_last,
                    )
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let runtime_budget = self.runtime_budget(idx, side);
        let ema_bands = self.emas[idx].compute_unstuck_bands(side);
        let current_price = self.hlcvs_value(k, idx, CLOSE);
        let ex = &self.exchange_params_list[idx];

        let size_abs = position.size.abs();
        let effective_wel = wallet_exposure_limit_with_allowance_from_base(
            bp,
            runtime_budget.effective_wallet_exposure_limit,
        );
        let wallet_exposure = calc_wallet_exposure(ex.c_mult, balance, size_abs, position.price);
        let ema_price_target = match side {
            LONG => ema_bands.upper * (1.0 + bp.unstuck_ema_dist),
            SHORT => ema_bands.lower * (1.0 - bp.unstuck_ema_dist),
            _ => 0.0,
        };
        let ema_price_rounded = match side {
            LONG => round_up(ema_price_target, ex.price_step),
            SHORT => round_dn(ema_price_target, ex.price_step),
            _ => 0.0,
        };
        let meets_trigger = match side {
            LONG => current_price >= ema_price_rounded,
            SHORT => current_price <= ema_price_rounded,
            _ => false,
        };

        let min_entry_qty = calc_min_entry_qty(current_price, ex);
        let target_qty_raw = crate::utils::cost_to_qty(
            balance * effective_wel * bp.unstuck_close_pct,
            current_price,
            ex.c_mult,
        );
        let target_qty_dn = round_dn(target_qty_raw, ex.qty_step).max(0.0);
        let close_qty_pre_allowance = match side {
            LONG => -f64::min(size_abs, f64::max(min_entry_qty, target_qty_dn)),
            SHORT => f64::min(size_abs, f64::max(min_entry_qty, target_qty_dn)),
            _ => 0.0,
        };

        let pnl_if_closed = match side {
            LONG => calc_pnl_long(
                position.price,
                current_price,
                close_qty_pre_allowance,
                ex.c_mult,
            ),
            SHORT => calc_pnl_short(
                position.price,
                current_price,
                close_qty_pre_allowance,
                ex.c_mult,
            ),
            _ => 0.0,
        };

        let mut close_qty_final = close_qty_pre_allowance;
        if allowance > 0.0 && pnl_if_closed < 0.0 {
            let pnl_abs = pnl_if_closed.abs();
            if pnl_abs > allowance {
                let scaled_qty = close_qty_pre_allowance.abs() * (allowance / pnl_abs);
                let scaled_qty = f64::min(size_abs, scaled_qty);
                let scaled_qty = f64::max(min_entry_qty, round_dn(scaled_qty, ex.qty_step));
                close_qty_final = match side {
                    LONG => -scaled_qty,
                    SHORT => scaled_qty,
                    _ => close_qty_pre_allowance,
                };
            }
        }

        let payload = UnstuckCalcDebug {
            step: k,
            coin,
            idx,
            side,
            balance,
            balance_bits: balance.to_bits(),
            allowance,
            allowance_bits: allowance.to_bits(),
            position_size: position.size,
            position_size_bits: position.size.to_bits(),
            position_price: position.price,
            position_price_bits: position.price.to_bits(),
            current_price,
            current_price_bits: current_price.to_bits(),
            ema_band_upper: ema_bands.upper,
            ema_band_upper_bits: ema_bands.upper.to_bits(),
            ema_band_lower: ema_bands.lower,
            ema_band_lower_bits: ema_bands.lower.to_bits(),
            wallet_exposure_limit: runtime_budget.effective_wallet_exposure_limit,
            wallet_exposure_limit_bits: runtime_budget.effective_wallet_exposure_limit.to_bits(),
            risk_we_excess_allowance_pct: bp.risk_we_excess_allowance_pct,
            unstuck_threshold: bp.unstuck_threshold,
            unstuck_close_pct: bp.unstuck_close_pct,
            unstuck_ema_dist: bp.unstuck_ema_dist,
            qty_step: ex.qty_step,
            price_step: ex.price_step,
            min_qty: ex.min_qty,
            min_cost: ex.min_cost,
            c_mult: ex.c_mult,
            effective_wel,
            effective_wel_bits: effective_wel.to_bits(),
            wallet_exposure,
            wallet_exposure_bits: wallet_exposure.to_bits(),
            ema_price_target,
            ema_price_target_bits: ema_price_target.to_bits(),
            ema_price_rounded,
            ema_price_rounded_bits: ema_price_rounded.to_bits(),
            meets_trigger,
            min_entry_qty,
            min_entry_qty_bits: min_entry_qty.to_bits(),
            target_qty_raw,
            target_qty_raw_bits: target_qty_raw.to_bits(),
            target_qty_dn,
            target_qty_dn_bits: target_qty_dn.to_bits(),
            close_qty_pre_allowance,
            close_qty_pre_allowance_bits: close_qty_pre_allowance.to_bits(),
            pnl_if_closed,
            pnl_if_closed_bits: pnl_if_closed.to_bits(),
            close_qty_final,
            close_qty_final_bits: close_qty_final.to_bits(),
        };

        let fname = "debug_unstuck_calc_orchestrator.json";
        if let Ok(mut f) = File::create(fname) {
            let _ = serde_json::to_writer_pretty(&mut f, &payload);
            let _ = writeln!(f);
        }
    }

    fn record_balance_trace(
        &mut self,
        k: usize,
        idx: usize,
        event: &'static str,
        order: &Order,
        fill_qty: f64,
        fill_price: f64,
        pnl: f64,
        fee_paid: f64,
        before: BalanceSnapshot,
        after: BalanceSnapshot,
    ) {
        if !DEBUG_TRACE_BALANCE {
            return;
        }
        if DEBUG_TRACE_WINDOW
            .map(|(start, end)| k < start || k > end)
            .unwrap_or(false)
        {
            return;
        }

        let coin = self
            .backtest_params
            .coins
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("idx_{idx}"));
        if let Some(want) = DEBUG_TRACE_COIN_FILTER {
            if coin != want {
                return;
            }
        }

        let Some(writer) = self.debug_balance_writer.as_mut() else {
            return;
        };

        let timestamp_ms = self.first_timestamp_ms + (k as u64) * self.interval_ms;
        let record = DebugBalanceTraceRecord {
            step: k,
            timestamp_ms,
            coin,
            event,
            order_type_id: order.order_type.id(),
            fill_qty,
            fill_price,
            pnl,
            fee_paid,
            btc_price: self.btc_usd_prices[k],
            usd_cash_wallet_before: before.usd_cash_wallet,
            usd_cash_wallet_after: after.usd_cash_wallet,
            usd_total_balance_before: before.usd_total_balance,
            usd_total_balance_after: after.usd_total_balance,
            usd_total_balance_rounded_before: before.usd_total_balance_rounded,
            usd_total_balance_rounded_after: after.usd_total_balance_rounded,
            btc_cash_wallet_before: before.btc_cash_wallet,
            btc_cash_wallet_after: after.btc_cash_wallet,
            btc_total_balance_before: before.btc_total_balance,
            btc_total_balance_after: after.btc_total_balance,
        };
        writer.write_record(&record);
    }

    fn configured_mode(&self, idx: usize, pside: usize) -> Option<orchestrator::TradingMode> {
        let side = match pside {
            LONG => &self.bot_params[idx].long,
            SHORT => &self.bot_params[idx].short,
            _ => return None,
        };
        if !side.entry_eligible {
            Some(orchestrator::TradingMode::GracefulStop)
        } else if side.is_forced_active {
            Some(orchestrator::TradingMode::Normal)
        } else {
            None
        }
    }

    fn build_orchestrator_input_iter<I>(
        &mut self,
        k: usize,
        peek_hints: Option<EntryPeekHints>,
        forager_hysteresis: Option<ForagerHysteresisState>,
        indices: I,
    ) -> orchestrator::OrchestratorInput
    where
        I: IntoIterator<Item = usize>,
    {
        let balance = self.balance.usd_total_balance_rounded;
        let balance_raw = self.balance.usd_total_balance;
        let (effective_cumsum_max, effective_cumsum_last) = self.effective_pnl_cumsum(k);

        let long_allowance = if self.bot_params_master.long.unstuck_enabled
            && self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0
        {
            calc_auto_unstuck_allowance(
                balance_raw,
                self.bot_params_master.long.unstuck_loss_allowance_pct
                    * self.bot_params_master.long.total_wallet_exposure_limit,
                effective_cumsum_max,
                effective_cumsum_last,
            )
        } else {
            0.0
        };
        let short_allowance = if self.bot_params_master.short.unstuck_enabled
            && self.bot_params_master.short.unstuck_loss_allowance_pct > 0.0
        {
            calc_auto_unstuck_allowance(
                balance_raw,
                self.bot_params_master.short.unstuck_loss_allowance_pct
                    * self.bot_params_master.short.total_wallet_exposure_limit,
                effective_cumsum_max,
                effective_cumsum_last,
            )
        } else {
            0.0
        };

        let symbols: Vec<orchestrator::SymbolInput> = indices
            .into_iter()
            .map(|idx| {
                let (start, end) = self.coin_valid_range(idx).unwrap_or((0, 0));
                let price_idx = k.clamp(start, end);
                let close_price = self.hlcvs_value(price_idx, idx, CLOSE).max(f64::EPSILON);

                let order_book = OrderBook {
                    bid: close_price,
                    ask: close_price,
                };
                let exchange = self.exchange_params_list[idx].clone();
                let effective_min_cost = calc_effective_min_cost(close_price, &exchange);

                let tradable = self.coin_is_tradeable_at(idx, k);
                let next_candle = if k + 1 < self.hlcvs.shape()[0] {
                    let tradable_next = self.coin_is_tradeable_at(idx, k + 1);
                    let (low, high) = if tradable_next {
                        (
                            self.hlcvs_value(k + 1, idx, LOW),
                            self.hlcvs_value(k + 1, idx, HIGH),
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    Some(orchestrator::NextCandle {
                        limit_order_fill_buffer_pct: self
                            .backtest_params
                            .limit_order_fill_buffer_pct,
                        low,
                        high,
                        tradable: tradable_next,
                    })
                } else {
                    None
                };
                let within_valid_range_now = self.coin_is_within_valid_range_at(idx, k);

                let pos_long = self.positions.long[idx];
                let pos_short = self.positions.short[idx];

                let mut mode_long: Option<orchestrator::TradingMode> =
                    self.configured_mode(idx, LONG);
                let mut mode_short: Option<orchestrator::TradingMode> =
                    self.configured_mode(idx, SHORT);

                if let Some(delist_timestamp) = self.last_valid_timestamps[idx] {
                    if k >= delist_timestamp {
                        if pos_long.size != 0.0 {
                            mode_long = Some(orchestrator::TradingMode::Panic);
                        }
                        if pos_short.size != 0.0 {
                            mode_short = Some(orchestrator::TradingMode::Panic);
                        }
                    }
                } else {
                    if !within_valid_range_now && pos_long.size != 0.0 {
                        mode_long = Some(orchestrator::TradingMode::Panic);
                    }
                    if !within_valid_range_now && pos_short.size != 0.0 {
                        mode_short = Some(orchestrator::TradingMode::Panic);
                    }
                }

                if self.backtest_params.filter_by_min_effective_cost {
                    if !self.coin_passes_min_effective_cost(idx, LONG) && pos_long.size == 0.0 {
                        mode_long = Some(orchestrator::TradingMode::GracefulStop);
                    }
                    if !self.coin_passes_min_effective_cost(idx, SHORT) && pos_short.size == 0.0 {
                        mode_short = Some(orchestrator::TradingMode::GracefulStop);
                    }
                }
                self.apply_hard_stop_mode_overrides(
                    idx,
                    &mut mode_long,
                    &mut mode_short,
                    pos_long,
                    pos_short,
                );

                let mut m1 = OrchestratorEmaTimeframeBundle::default();
                let mut h1 = OrchestratorEmaTimeframeBundle::default();

                {
                    let strategy = &self.strategy_params[idx].long;
                    let (span0, span1) = strategy_ema_spans(strategy);
                    let mut spans = [span0, span1, (span0 * span1).sqrt()];
                    spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    for (span, value) in spans.into_iter().zip(self.emas[idx].long.into_iter()) {
                        m1.close.push((span, value));
                    }
                }
                {
                    let strategy = &self.strategy_params[idx].short;
                    let (span0, span1) = strategy_ema_spans(strategy);
                    let mut spans = [span0, span1, (span0 * span1).sqrt()];
                    spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    for (span, value) in spans.into_iter().zip(self.emas[idx].short.into_iter()) {
                        m1.close.push((span, value));
                    }
                }

                for (bot, values) in [
                    (&self.bot_params[idx].long, self.emas[idx].unstuck_long),
                    (&self.bot_params[idx].short, self.emas[idx].unstuck_short),
                ] {
                    if bot.unstuck_enabled && bot.unstuck_ema_gating_enabled {
                        for (span, value) in unstuck_spans(bot).into_iter().zip(values) {
                            if !m1.close.iter().any(|(existing, _)| *existing == span) {
                                m1.close.push((span, value));
                            }
                        }
                    }
                }
                let vol_span_long = self.bot_params_master.long.filter_volume_ema_span_1m as f64;
                let vol_span_short = self.bot_params_master.short.filter_volume_ema_span_1m as f64;
                let lr_span_long = self.bot_params_master.long.filter_volatility_ema_span_1m as f64;
                let lr_span_short =
                    self.bot_params_master.short.filter_volatility_ema_span_1m as f64;

                debug_assert_eq!(
                    self.bot_params[idx].long.filter_volume_ema_span_1m as f64, vol_span_long,
                    "coin {} long filter_volume_ema_span_1m differs from master",
                    idx
                );
                debug_assert_eq!(
                    self.bot_params[idx].short.filter_volume_ema_span_1m as f64, vol_span_short,
                    "coin {} short filter_volume_ema_span_1m differs from master",
                    idx
                );
                debug_assert_eq!(
                    self.bot_params[idx].long.filter_volatility_ema_span_1m as f64, lr_span_long,
                    "coin {} long filter_volatility_ema_span_1m differs from master",
                    idx
                );
                debug_assert_eq!(
                    self.bot_params[idx].short.filter_volatility_ema_span_1m as f64, lr_span_short,
                    "coin {} short filter_volatility_ema_span_1m differs from master",
                    idx
                );

                m1.signed_unilateralness = self.unilateralness_at(idx, k);
                m1.volume.push((vol_span_long, self.emas[idx].vol_long));
                m1.volume.push((vol_span_short, self.emas[idx].vol_short));
                m1.log_range
                    .push((lr_span_long, self.emas[idx].log_range_long));
                m1.log_range
                    .push((lr_span_short, self.emas[idx].log_range_short));
                if let Some(span) =
                    strategy_offset_volatility_span_minutes(&self.strategy_params[idx].long)
                {
                    if span > 0.0 {
                        m1.log_range
                            .push((span, self.emas[idx].volatility_ema_1m_long));
                    }
                }
                if let Some(span) =
                    strategy_offset_volatility_span_minutes(&self.strategy_params[idx].short)
                {
                    if span > 0.0 {
                        m1.log_range
                            .push((span, self.emas[idx].volatility_ema_1m_short));
                    }
                }

                if let Some(span) =
                    strategy_entry_volatility_span_hours(&self.strategy_params[idx].long)
                {
                    if span > 0.0 {
                        h1.log_range
                            .push((span, self.emas[idx].volatility_ema_1h_long));
                    }
                }
                if let Some(span) =
                    strategy_entry_volatility_span_hours(&self.strategy_params[idx].short)
                {
                    if span > 0.0 {
                        h1.log_range
                            .push((span, self.emas[idx].volatility_ema_1h_short));
                    }
                }

                let emas = OrchestratorEmaBundle { m1, h1 };

                let trailing_long = self.trailing_prices.long[idx].clone();
                let trailing_short = self.trailing_prices.short[idx].clone();

                orchestrator::SymbolInput {
                    symbol_idx: idx,
                    order_book,
                    exchange,
                    tradable,
                    allow_missing_strategy_inputs: false,
                    unilateralness_warmup_spans: self.unilateralness_warmup_spans(idx, k),
                    unilateralness_unavailable: Default::default(),
                    next_candle,
                    effective_min_cost,
                    emas,
                    forager_m1: None,
                    long: orchestrator::SymbolSideInput {
                        mode: mode_long,
                        position: pos_long,
                        trailing: trailing_long,
                        trailing_available: true,
                        last_increase_fill_timestamp_ms: self.last_increase_fill_timestamp_long
                            [idx],
                        bot_params: self.hsl_order_params(LONG, idx),
                        strategy_params: None,
                        parsed_strategy_params: Some(self.strategy_params[idx].long),
                        runtime_budget: Some(self.runtime_budget[idx].long.clone()),
                    },
                    short: orchestrator::SymbolSideInput {
                        mode: mode_short,
                        position: pos_short,
                        trailing: trailing_short,
                        trailing_available: true,
                        last_increase_fill_timestamp_ms: self.last_increase_fill_timestamp_short
                            [idx],
                        bot_params: self.hsl_order_params(SHORT, idx),
                        strategy_params: None,
                        parsed_strategy_params: Some(self.strategy_params[idx].short),
                        runtime_budget: Some(self.runtime_budget[idx].short.clone()),
                    },
                }
            })
            .collect();

        orchestrator::OrchestratorInput {
            timestamp_ms: self.first_timestamp_ms + (k as u64) * self.interval_ms,
            balance,
            balance_raw,
            global: orchestrator::OrchestratorGlobal {
                filter_by_min_effective_cost: self.backtest_params.filter_by_min_effective_cost,
                market_orders_allowed: self.backtest_params.market_orders_allowed,
                market_order_near_touch_threshold: self
                    .backtest_params
                    .market_order_near_touch_threshold,
                market_order_slippage_pct: self.backtest_params.market_order_slippage_pct,
                panic_close_market: false,
                auto_unstuck_allowed: Some(true),
                unstuck_allowance_long: long_allowance,
                unstuck_allowance_short: short_allowance,
                max_realized_loss_pct: self.backtest_params.max_realized_loss_pct,
                realized_pnl_cumsum_max: effective_cumsum_max,
                realized_pnl_cumsum_last: effective_cumsum_last,
                sort_global: false,
                global_bot_params: self.bot_params_master.clone(),
                hedge_mode: self.backtest_params.hedge_mode,
                strategy_kind: self.strategy_kind,
            },
            symbols,
            peek_hints,
            forager_hysteresis,
        }
    }

    fn get_orchestrator_input_cached(
        &mut self,
        k: usize,
        peek_hints: Option<EntryPeekHints>,
        forager_hysteresis: Option<ForagerHysteresisState>,
    ) -> orchestrator::OrchestratorInput {
        // Take ownership temporarily to avoid borrow conflicts while we also read from `self`.
        let mut input = self.orchestrator_input_cache.take().unwrap_or_else(|| {
            self.build_orchestrator_input_iter(k, None, forager_hysteresis.clone(), 0..self.n_coins)
        });

        input.balance = self.balance.usd_total_balance_rounded;
        input.balance_raw = self.balance.usd_total_balance;
        input.timestamp_ms = self.first_timestamp_ms + (k as u64) * self.interval_ms;

        let balance_raw = input.balance_raw;
        let (effective_cumsum_max, effective_cumsum_last) = self.effective_pnl_cumsum(k);
        input.global.unstuck_allowance_long = if self.bot_params_master.long.unstuck_enabled
            && self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0
        {
            calc_auto_unstuck_allowance(
                balance_raw,
                self.bot_params_master.long.unstuck_loss_allowance_pct
                    * self.bot_params_master.long.total_wallet_exposure_limit,
                effective_cumsum_max,
                effective_cumsum_last,
            )
        } else {
            0.0
        };
        input.global.unstuck_allowance_short = if self.bot_params_master.short.unstuck_enabled
            && self.bot_params_master.short.unstuck_loss_allowance_pct > 0.0
        {
            calc_auto_unstuck_allowance(
                balance_raw,
                self.bot_params_master.short.unstuck_loss_allowance_pct
                    * self.bot_params_master.short.total_wallet_exposure_limit,
                effective_cumsum_max,
                effective_cumsum_last,
            )
        } else {
            0.0
        };
        input.global.max_realized_loss_pct = self.backtest_params.max_realized_loss_pct;
        input.global.realized_pnl_cumsum_max = effective_cumsum_max;
        input.global.realized_pnl_cumsum_last = effective_cumsum_last;

        input.peek_hints = peek_hints;
        input.forager_hysteresis = forager_hysteresis;

        for sym in input.symbols.iter_mut() {
            let idx = sym.symbol_idx;
            let (start, end) = self.coin_valid_range(idx).unwrap_or((0, 0));
            let price_idx = k.clamp(start, end);
            let close_price = self.hlcvs_value(price_idx, idx, CLOSE).max(f64::EPSILON);

            sym.order_book.bid = close_price;
            sym.order_book.ask = close_price;
            sym.tradable = self.coin_is_tradeable_at(idx, k);
            sym.next_candle = if k + 1 < self.hlcvs.shape()[0] {
                let tradable_next = self.coin_is_tradeable_at(idx, k + 1);
                let (low, high) = if tradable_next {
                    (
                        self.hlcvs_value(k + 1, idx, LOW),
                        self.hlcvs_value(k + 1, idx, HIGH),
                    )
                } else {
                    (0.0, 0.0)
                };
                Some(orchestrator::NextCandle {
                    limit_order_fill_buffer_pct: self.backtest_params.limit_order_fill_buffer_pct,
                    low,
                    high,
                    tradable: tradable_next,
                })
            } else {
                None
            };

            let exchange = &sym.exchange;
            sym.effective_min_cost = calc_effective_min_cost(close_price, exchange);

            let pos_long = self.positions.long[idx];
            let pos_short = self.positions.short[idx];
            sym.long.position = pos_long;
            sym.short.position = pos_short;

            sym.long.trailing = self.trailing_prices.long[idx].clone();
            sym.short.trailing = self.trailing_prices.short[idx].clone();
            sym.long.last_increase_fill_timestamp_ms = self.last_increase_fill_timestamp_long[idx];
            sym.short.last_increase_fill_timestamp_ms =
                self.last_increase_fill_timestamp_short[idx];

            sym.long.runtime_budget = Some(self.runtime_budget[idx].long.clone());
            sym.short.runtime_budget = Some(self.runtime_budget[idx].short.clone());

            let within_valid_range_now = self.coin_is_within_valid_range_at(idx, k);
            let mut mode_long: Option<orchestrator::TradingMode> = self.configured_mode(idx, LONG);
            let mut mode_short: Option<orchestrator::TradingMode> =
                self.configured_mode(idx, SHORT);

            if let Some(delist_timestamp) = self.last_valid_timestamps[idx] {
                if k >= delist_timestamp {
                    if pos_long.size != 0.0 {
                        mode_long = Some(orchestrator::TradingMode::Panic);
                    }
                    if pos_short.size != 0.0 {
                        mode_short = Some(orchestrator::TradingMode::Panic);
                    }
                }
            } else {
                if !within_valid_range_now && pos_long.size != 0.0 {
                    mode_long = Some(orchestrator::TradingMode::Panic);
                }
                if !within_valid_range_now && pos_short.size != 0.0 {
                    mode_short = Some(orchestrator::TradingMode::Panic);
                }
            }

            if self.backtest_params.filter_by_min_effective_cost {
                if !self.coin_passes_min_effective_cost(idx, LONG) && pos_long.size == 0.0 {
                    mode_long = Some(orchestrator::TradingMode::GracefulStop);
                }
                if !self.coin_passes_min_effective_cost(idx, SHORT) && pos_short.size == 0.0 {
                    mode_short = Some(orchestrator::TradingMode::GracefulStop);
                }
            }
            self.apply_hard_stop_mode_overrides(
                idx,
                &mut mode_long,
                &mut mode_short,
                pos_long,
                pos_short,
            );

            sym.long.mode = mode_long;
            sym.short.mode = mode_short;
            if self.hsl_enabled() {
                let long = self.hsl_order_params(LONG, idx);
                let short = self.hsl_order_params(SHORT, idx);
                sym.long.bot_params.hsl_enabled = long.hsl_enabled;
                sym.long.bot_params.hsl_panic_close_order_type = long.hsl_panic_close_order_type;
                sym.short.bot_params.hsl_enabled = short.hsl_enabled;
                sym.short.bot_params.hsl_panic_close_order_type = short.hsl_panic_close_order_type;
            }

            sym.emas.m1.signed_unilateralness = self.unilateralness_at(idx, k);
            sym.unilateralness_warmup_spans = self.unilateralness_warmup_spans(idx, k);
            // Update EMA values (spans are stable; we overwrite only values).
            // m1.close: 3 long then 3 short.
            if sym.emas.m1.close.len() >= 6 {
                for (i, v) in self.emas[idx].long.iter().copied().enumerate() {
                    sym.emas.m1.close[i].1 = v;
                }
                for (i, v) in self.emas[idx].short.iter().copied().enumerate() {
                    sym.emas.m1.close[i + 3].1 = v;
                }
            }
            for (bot, values) in [
                (&self.bot_params[idx].long, self.emas[idx].unstuck_long),
                (&self.bot_params[idx].short, self.emas[idx].unstuck_short),
            ] {
                for (span, value) in unstuck_spans(bot).into_iter().zip(values) {
                    for (stored_span, stored_value) in sym.emas.m1.close.iter_mut().skip(6) {
                        if *stored_span == span {
                            *stored_value = value;
                        }
                    }
                }
            }
            let slots = self.orchestrator_ema_slots[idx];
            if sym.emas.m1.volume.len() > slots.m1_volume_short {
                sym.emas.m1.volume[slots.m1_volume_long].1 = self.emas[idx].vol_long;
                sym.emas.m1.volume[slots.m1_volume_short].1 = self.emas[idx].vol_short;
            }
            if sym.emas.m1.log_range.len() > slots.m1_log_range.forager_short {
                sym.emas.m1.log_range[slots.m1_log_range.forager_long].1 =
                    self.emas[idx].log_range_long;
                sym.emas.m1.log_range[slots.m1_log_range.forager_short].1 =
                    self.emas[idx].log_range_short;
            }
            if let Some(slot) = slots.m1_log_range.offset_long {
                if sym.emas.m1.log_range.len() > slot {
                    sym.emas.m1.log_range[slot].1 = self.emas[idx].volatility_ema_1m_long;
                }
            }
            if let Some(slot) = slots.m1_log_range.offset_short {
                if sym.emas.m1.log_range.len() > slot {
                    sym.emas.m1.log_range[slot].1 = self.emas[idx].volatility_ema_1m_short;
                }
            }
            if let Some(slot) = slots.h1_log_range.long {
                if sym.emas.h1.log_range.len() > slot {
                    sym.emas.h1.log_range[slot].1 = self.emas[idx].volatility_ema_1h_long;
                }
            }
            if let Some(slot) = slots.h1_log_range.short {
                if sym.emas.h1.log_range.len() > slot {
                    sym.emas.h1.log_range[slot].1 = self.emas[idx].volatility_ema_1h_short;
                }
            }
        }

        input
    }
    #[inline(always)]
    fn col(&self, idx: usize) -> usize {
        self.active_coin_indices[idx]
    }

    #[inline(always)]
    fn hlcvs_value(&self, row: usize, coin_idx: usize, feature: usize) -> f64 {
        let col = self.col(coin_idx);
        self.hlcvs[[row, col, feature]]
    }

    #[cfg(test)]
    pub fn new(
        hlcvs: ArrayView3<'a, f64>,
        btc_usd_prices: ArrayView1<'a, f64>,
        bot_params: Vec<BotParamsPair>,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
        let strategy_params = bot_params
            .iter()
            .map(test_strategy_params_pair_value_from_flat)
            .collect();
        Self::new_with_strategy_params(
            hlcvs,
            btc_usd_prices,
            crate::strategies::StrategyKind::TrailingMartingale,
            bot_params,
            strategy_params,
            exchange_params_list,
            backtest_params,
        )
    }

    pub fn new_with_strategy_params(
        hlcvs: ArrayView3<'a, f64>,
        btc_usd_prices: ArrayView1<'a, f64>,
        strategy_kind: crate::strategies::StrategyKind,
        bot_params: Vec<BotParamsPair>,
        strategy_params: Vec<StrategyParamsPairValue>,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
        assert!(
            backtest_params.maker_fee.is_finite(),
            "backtest maker_fee must be finite"
        );
        assert!(
            backtest_params.taker_fee.is_finite(),
            "backtest taker_fee must be finite"
        );
        crate::limit_fills::validate_buffer(backtest_params.limit_order_fill_buffer_pct)
            .expect("invalid backtest fill buffer");
        let mut balance = Balance::default();
        balance.btc_collateral_cap = backtest_params.btc_collateral_cap.max(0.0);
        balance.btc_collateral_ltv_cap = backtest_params.btc_collateral_ltv_cap;
        balance.use_btc_collateral = balance.btc_collateral_cap > 0.0;

        let starting_balance = backtest_params.starting_balance;
        let initial_btc_price = btc_usd_prices[0].max(f64::EPSILON);

        balance.usd_cash_wallet = starting_balance;
        balance.btc_cash_wallet = 0.0;
        balance.usd_total_balance =
            (balance.btc_cash_wallet * initial_btc_price) + balance.usd_cash_wallet;
        balance.btc_total_balance = if initial_btc_price > 0.0 {
            balance.usd_total_balance / initial_btc_price
        } else {
            0.0
        };
        balance.usd_total_balance_rounded = balance.usd_total_balance;

        let n_timesteps = hlcvs.shape()[0];
        let total_cols = hlcvs.shape()[1];
        let mut active_coin_indices = backtest_params
            .active_coin_indices
            .clone()
            .unwrap_or_else(|| (0..bot_params.len()).collect());
        if active_coin_indices.len() != bot_params.len() {
            active_coin_indices = (0..bot_params.len()).collect();
        }
        for &col in &active_coin_indices {
            assert!(
                col < total_cols,
                "active coin index {} exceeds available columns {}",
                col,
                total_cols
            );
        }
        let n_coins = active_coin_indices.len();
        assert_eq!(
            bot_params.len(),
            n_coins,
            "bot params length ({}) does not match active coin indices ({})",
            bot_params.len(),
            n_coins
        );
        assert_eq!(
            strategy_params.len(),
            n_coins,
            "strategy params length ({}) does not match active coin indices ({})",
            strategy_params.len(),
            n_coins
        );
        let mut first_valid_idx = backtest_params.first_valid_indices.clone();
        if first_valid_idx.len() != n_coins {
            first_valid_idx = vec![0usize; n_coins];
        }
        let mut last_valid_idx = backtest_params.last_valid_indices.clone();
        if last_valid_idx.len() != n_coins {
            last_valid_idx = vec![n_timesteps.saturating_sub(1); n_coins];
        }
        let warmup_minutes = if backtest_params.warmup_minutes.len() == n_coins {
            backtest_params.warmup_minutes.clone()
        } else {
            vec![0usize; n_coins]
        };
        let mut trade_start_idx = if backtest_params.trade_start_indices.len() == n_coins {
            backtest_params.trade_start_indices.clone()
        } else {
            vec![0usize; n_coins]
        };
        let mut trade_activation_logged = vec![false; n_coins];

        for i in 0..n_coins {
            let mut first = first_valid_idx[i];
            if first >= n_timesteps {
                first = n_timesteps.saturating_sub(1);
            }
            let mut last = last_valid_idx[i];
            if last >= n_timesteps {
                last = n_timesteps.saturating_sub(1);
            }
            if last < first {
                last = first;
            }
            first_valid_idx[i] = first;
            last_valid_idx[i] = last;
            let warm = warmup_minutes.get(i).copied().unwrap_or(0);
            let interval = backtest_params.candle_interval_minutes.max(1) as usize;
            let warm_bars = if interval > 1 {
                (warm + interval - 1) / interval
            } else {
                warm
            };
            let provided_trade_idx = trade_start_idx[i];
            let trade_idx = first
                .saturating_add(warm_bars)
                .min(last)
                .max(provided_trade_idx);
            // RMS readiness is scoped to its consuming side/order branch.
            trade_start_idx[i] = trade_idx;

            let expected_trade_idx = first.saturating_add(warm_bars).min(last);
            debug_assert!(
                trade_idx >= expected_trade_idx,
                "trade start index mismatch for coin {}: expected at least {} but got {}",
                i,
                expected_trade_idx,
                trade_idx
            );
            trade_activation_logged[i] = false;
        }

        let initial_emas = (0..n_coins)
            .map(|i| {
                let start_idx = first_valid_idx
                    .get(i)
                    .copied()
                    .unwrap_or(0)
                    .min(n_timesteps.saturating_sub(1));
                let col = active_coin_indices[i];
                let close_price = hlcvs[[start_idx, col, CLOSE]];
                let base_close = if close_price.is_finite() {
                    close_price
                } else {
                    0.0
                };
                let volume = hlcvs[[start_idx, col, VOLUME]];
                let base_volume = if volume.is_finite() {
                    volume.max(0.0)
                } else {
                    0.0
                };
                // Convert base volume to quote volume using typical price
                // This matches live bot's get_latest_ema_quote_volume() calculation
                let high = hlcvs[[start_idx, col, HIGH]];
                let low = hlcvs[[start_idx, col, LOW]];
                let typical_price = if high.is_finite() && low.is_finite() && base_close > 0.0 {
                    (high + low + base_close) / 3.0
                } else {
                    base_close.max(1.0) // Fallback to close price or 1.0
                };
                let quote_volume = base_volume * typical_price;
                EMAs {
                    unstuck_long: [base_close; 3],
                    unstuck_long_num: [base_close; 3],
                    unstuck_long_den: [1.0; 3],
                    unstuck_short: [base_close; 3],
                    unstuck_short_num: [base_close; 3],
                    unstuck_short_den: [1.0; 3],
                    long: [base_close; 3],
                    long_num: [base_close; 3],
                    long_den: [1.0; 3],
                    short: [base_close; 3],
                    short_num: [base_close; 3],
                    short_den: [1.0; 3],
                    vol_long: quote_volume,
                    vol_long_num: quote_volume,
                    vol_long_den: 1.0,
                    vol_short: quote_volume,
                    vol_short_num: quote_volume,
                    vol_short_den: 1.0,
                    log_range_long: 0.0,
                    log_range_long_num: 0.0,
                    log_range_long_den: 1.0,
                    log_range_short: 0.0,
                    log_range_short_num: 0.0,
                    log_range_short_den: 1.0,
                    volatility_ema_1m_long: 0.0,
                    volatility_ema_1m_long_num: 0.0,
                    volatility_ema_1m_long_den: 1.0,
                    volatility_ema_1m_short: 0.0,
                    volatility_ema_1m_short_num: 0.0,
                    volatility_ema_1m_short_den: 1.0,
                    volatility_ema_1h_long: 0.0,
                    volatility_ema_1h_long_num: 0.0,
                    volatility_ema_1h_long_den: 1.0,
                    volatility_ema_1h_short: 0.0,
                    volatility_ema_1h_short_num: 0.0,
                    volatility_ema_1h_short_den: 1.0,
                }
            })
            .collect();
        let equities = Equities::default();

        // Normalize per-position WELs the same way the Python config path does:
        // zero means "derive from TWEL / n_positions", negative means dynamic.
        let mut bot_params = bot_params;
        let bot_params_original = bot_params.clone();
        for bp in bot_params.iter_mut() {
            if bp.long.wallet_exposure_limit == 0.0 && bp.long.n_positions > 0 {
                bp.long.wallet_exposure_limit =
                    bp.long.total_wallet_exposure_limit / bp.long.n_positions as f64;
            }
            if bp.short.wallet_exposure_limit == 0.0 && bp.short.n_positions > 0 {
                bp.short.wallet_exposure_limit =
                    bp.short.total_wallet_exposure_limit / bp.short.n_positions as f64;
            }
        }
        let strategy_params_parsed: Vec<StrategyParamsPair> = strategy_params
            .iter()
            .zip(bot_params.iter())
            .map(|(raw, bp)| parse_strategy_params_pair(strategy_kind, raw, bp))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|err| panic!("failed to parse strategy params for backtest: {err}"));

        // init bot params
        let configured_n_positions = EffectiveNPositions {
            long: bot_params[0].long.n_positions,
            short: bot_params[0].short.n_positions,
        };
        let mut bot_params_master = bot_params[0].clone();
        bot_params_master.long.n_positions = n_coins.min(bot_params_master.long.n_positions);
        bot_params_master.short.n_positions = n_coins.min(bot_params_master.short.n_positions);

        let effective_n_positions = configured_n_positions.clone();
        let runtime_budget = bot_params
            .iter()
            .map(|bp| RuntimeBudgetStatePair {
                long: make_runtime_budget_state(&bp.long, configured_n_positions.long),
                short: make_runtime_budget_state(&bp.short, configured_n_positions.short),
            })
            .collect();
        let orchestrator_ema_slots: Vec<OrchestratorEmaSlots> = strategy_params_parsed
            .iter()
            .map(|sp| make_orchestrator_ema_slots(sp, &bot_params_master))
            .collect();

        // Calculate EMA alphas for each coin, adjusted for candle interval
        let interval = backtest_params.candle_interval_minutes;
        let ema_alphas: Vec<EmaAlphas> = bot_params
            .iter()
            .zip(strategy_params_parsed.iter())
            .map(|(bp, sp)| calc_ema_alphas(bp, sp, interval))
            .collect();
        let mut warmup_bars = backtest_params.global_warmup_bars;
        if warmup_bars == 0 {
            // Zero is the legacy automatic non-RMS warmup sentinel, including
            // direct callers. RMS readiness remains separately consumer-scoped.
            warmup_bars = calc_warmup_bars(&bot_params, &strategy_params_parsed);
        }

        let trailing_enabled: Vec<TrailingEnabled> = strategy_params_parsed
            .iter()
            .map(|sp| TrailingEnabled {
                long: strategy_has_trailing(&sp.long),
                short: strategy_has_trailing(&sp.short),
            })
            .collect();
        let any_trailing_long = trailing_enabled.iter().any(|te| te.long);
        let any_trailing_short = trailing_enabled.iter().any(|te| te.short);
        let needs_log_range_long = bot_params
            .iter()
            .any(|bp| bp.long.forager_score_weights.volatility != 0.0)
            || strategy_params_parsed
                .iter()
                .any(|sp| strategy_needs_log_range_1m(&sp.long));
        let needs_log_range_short = bot_params
            .iter()
            .any(|bp| bp.short.forager_score_weights.volatility != 0.0)
            || strategy_params_parsed
                .iter()
                .any(|sp| strategy_needs_log_range_1m(&sp.short));
        let needs_volatility_ema_1m_long = strategy_params_parsed
            .iter()
            .any(|sp| strategy_needs_log_range_1m(&sp.long));
        let needs_volatility_ema_1m_short = strategy_params_parsed
            .iter()
            .any(|sp| strategy_needs_log_range_1m(&sp.short));
        let needs_volatility_ema_1h_long = strategy_params_parsed
            .iter()
            .any(|sp| strategy_needs_log_range_1h(&sp.long));
        let needs_volatility_ema_1h_short = strategy_params_parsed
            .iter()
            .any(|sp| strategy_needs_log_range_1h(&sp.short));
        let btc_collateral_initialized = !balance.use_btc_collateral;

        let rms_enabled = crate::unilateralness::backtest_enabled_sides(
            &bot_params, backtest_params.dynamic_wel_by_tradability,
        );
        Backtest {
            hlcvs,
            btc_usd_prices,
            active_coin_indices,
            interval_ms: backtest_params.candle_interval_minutes * 60_000,
            bot_params_master: bot_params_master.clone(),
            bot_params: bot_params.clone(),
            bot_params_original,
            strategy_params: strategy_params_parsed,
            strategy_kind,
            runtime_budget,
            configured_n_positions,
            effective_n_positions,
            exchange_params_list,
            backtest_params: backtest_params.clone(),
            balance,
            n_coins,
            ema_alphas,
            emas: initial_emas,
            unilateralness: bot_params
                .iter()
                .zip(rms_enabled.iter())
                .map(|(pair, enabled)| {
                    let mut trackers = Vec::new();
                    for (bp, enabled) in [&pair.long, &pair.short].iter().zip(enabled) {
                        if *enabled
                            && !trackers.iter().any(|(span, _)| *span == bp.unilateralness_ema_span_1m)
                        {
                            trackers.push((
                                bp.unilateralness_ema_span_1m,
                                crate::unilateralness::RollingRms::new(bp.unilateralness_ema_span_1m)
                                    .expect("validated unilateralness span"),
                            ));
                        }
                    }
                    trackers
                })
                .collect(),
            unilateralness_step: None,
            orchestrator_ema_slots,
            needs_volume_ema_long: bot_params.iter().any(|bp| {
                bp.long.forager_volume_drop_pct != 0.0
                    || bp.long.forager_score_weights.volume != 0.0
            }),
            needs_volume_ema_short: bot_params.iter().any(|bp| {
                bp.short.forager_volume_drop_pct != 0.0
                    || bp.short.forager_score_weights.volume != 0.0
            }),
            needs_log_range_long,
            needs_log_range_short,
            needs_volatility_ema_1m_long,
            needs_volatility_ema_1m_short,
            needs_volatility_ema_1h_long,
            needs_volatility_ema_1h_short,
            coin_first_valid_idx: first_valid_idx,
            coin_last_valid_idx: last_valid_idx,
            coin_trade_start_idx: trade_start_idx,
            trade_activation_logged,
            positions: Positions::new(n_coins),
            first_timestamp_ms: backtest_params.first_timestamp_ms,
            last_hour_boundary_ms: (backtest_params.first_timestamp_ms / 3_600_000) * 3_600_000,
            latest_hour: vec![HourBucket::default(); n_coins],
            warmup_bars,
            current_step: 0,
            open_orders: OpenOrders::new(n_coins),
            trailing_prices: TrailingPrices::new(n_coins),
            pnl_cumsum_running: 0.0,
            pnl_cumsum_max: 0.0,
            pnl_cumsum_running_net: 0.0,
            pnl_cumsum_max_net: 0.0,
            pnl_cumsum_running_net_pside: [0.0, 0.0],
            pnl_cumsum_running_net_coin_pside: [vec![0.0; n_coins], vec![0.0; n_coins]],
            pnl_cumsum_max_net_coin_pside: [vec![0.0; n_coins], vec![0.0; n_coins]],
            pnl_lookback_bars: if backtest_params.pnls_max_lookback_days < 0.0 {
                usize::MAX
            } else {
                let minutes = backtest_params.pnls_max_lookback_days * 24.0 * 60.0;
                (minutes / backtest_params.candle_interval_minutes.max(1) as f64)
                    .ceil()
                    .max(1.0) as usize
            },
            pnl_events: VecDeque::new(),
            pnl_event_seq: 0,
            rolling_pnl_peak_candidates: VecDeque::new(),
            coin_pnl_events: [
                (0..n_coins).map(|_| VecDeque::new()).collect(),
                (0..n_coins).map(|_| VecDeque::new()).collect(),
            ],
            coin_pnl_event_seq: 0,
            coin_rolling_pnl_peak_candidates: [
                (0..n_coins).map(|_| VecDeque::new()).collect(),
                (0..n_coins).map(|_| VecDeque::new()).collect(),
            ],
            fills: Vec::new(),
            trading_enabled: TradingEnabled {
                long: bot_params
                    .iter()
                    .any(|bp| bp.long.entry_eligible && bp.long.wallet_exposure_limit != 0.0)
                    && bot_params_master.long.n_positions > 0,
                short: bot_params
                    .iter()
                    .any(|bp| bp.short.entry_eligible && bp.short.wallet_exposure_limit != 0.0)
                    && bot_params_master.short.n_positions > 0,
            },
            trailing_enabled,
            any_trailing_long,
            any_trailing_short,
            equities: equities,
            last_valid_timestamps: vec![None; n_coins],
            did_fill_long: vec![false; n_coins],
            did_fill_short: vec![false; n_coins],
            last_increase_fill_timestamp_long: vec![None; n_coins],
            last_increase_fill_timestamp_short: vec![None; n_coins],
            total_wallet_exposures: Vec::with_capacity(n_timesteps),
            equity_tracking_active: false,
            debug_writer: if DEBUG_DUMP_ORDERS {
                DebugOrderWriter::new_for_mode()
            } else {
                None
            },
            debug_balance_writer: if DEBUG_TRACE_BALANCE {
                DebugBalanceWriter::new_for_mode()
            } else {
                None
            },
            orchestrator_input_cache: None,
            orchestrator_workspace: orchestrator::OrchestratorWorkspace::default(),
            orch_profile: std::env::var(ORCH_PROFILE_ENV)
                .ok()
                .as_deref()
                .filter(|v| *v == "1")
                .map(|_| OrchProfile {
                    mode: "orchestrator",
                    ..OrchProfile::default()
                }),
            max_tradable_coins_seen: EffectiveNPositions { long: 0, short: 0 },
            hsl_scopes: Vec::new(),
            hsl_cutoffs: std::collections::BTreeMap::new(),
            hsl_traces: std::collections::BTreeMap::new(),
            hsl_report: hsl_report::Report::new(
                backtest_params.hsl_detailed_report && !backtest_params.metrics_only,
                !backtest_params.metrics_only,
            ),
            btc_collateral_initialized,
            strategy_equity_series: Vec::new(),
            strategy_equity_series_pside: [Vec::new(), Vec::new()],
            strategy_equity_timestamps_ms_pside: [Vec::new(), Vec::new()],
            liquidated: false,
            final_hard_stop_metrics: None,
            final_strategy_equity_metrics: None,
            // EMAs already initialized in `emas`; no rolling buffers needed
        }
    }

    pub fn liquidated(&self) -> bool {
        self.liquidated
    }

    fn validate_candle_coverage(&self) -> Result<(), String> {
        let n_timesteps = self.hlcvs.shape()[0];
        for idx in 0..self.n_coins {
            // Empty declared ranges represent symbols absent for this dataset.
            if let (Some(&first), Some(&last)) = (
                self.backtest_params.first_valid_indices.get(idx),
                self.backtest_params.last_valid_indices.get(idx),
            ) {
                if first >= n_timesteps || last < first {
                    continue;
                }
            }
            if let Some((first, last)) = self.coin_valid_range(idx) {
                let requires_rms = !self.unilateralness[idx].is_empty();
                for k in first..=last {
                    if requires_rms && self.hlcvs_value(k, idx, CLOSE) <= 0.0 {
                        return Err(format!(
                            "backtest RMS requires positive closes: coin {} index {} candle {}",
                            self.backtest_params.coins[idx], idx, k,
                        ));
                    }
                    for field in [HIGH, LOW, CLOSE] {
                        if !self.hlcvs_value(k, idx, field).is_finite() {
                            return Err(format!(
                                    "backtest requires contiguous finite H/L/C within each valid range: coin {} index {} candle {} field {}",
                                    self.backtest_params.coins[idx], idx, k, field,
                                ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_held_position_valuation(&self, k: usize) -> Result<(), String> {
        for idx in 0..self.n_coins {
            if self.positions.long[idx].size == 0.0 && self.positions.short[idx].size == 0.0 {
                continue;
            }
            if !self.coin_is_valid_at(idx, k) || !self.hlcvs_value(k, idx, CLOSE).is_finite() {
                return Err(format!(
                    "missing held-position valuation candle: coin {} index {} candle {}",
                    self.backtest_params.coins[idx], idx, k,
                ));
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<(Vec<Fill>, Equities), String> {
        self.validate_candle_coverage()?;
        self.update_unilateralness(0)?;
        let n_timesteps = self.hlcvs.shape()[0];

        // --- register first & last valid candle for every coin ---
        for idx in 0..self.n_coins {
            if let Some((_start, end)) = self.coin_valid_range(idx) {
                if end.saturating_add(1400) < n_timesteps {
                    // add only if delisted more than one day before last timestamp
                    self.last_valid_timestamps[idx] = Some(end);
                }
            }
        }

        let warmup_bars = self.warmup_bars.max(1);
        let guard_timestamp_ms = self
            .backtest_params
            .requested_start_timestamp_ms
            .max(self.first_timestamp_ms);
        for k in 1..(n_timesteps - 1) {
            self.current_step = k;
            self.validate_held_position_valuation(k)?;
            self.update_unilateralness(k)?;
            for idx in 0..self.n_coins {
                if !self.trade_activation_logged[idx] && self.coin_is_tradeable_at(idx, k) {
                    self.trade_activation_logged[idx] = true;
                }
                if k < self.coin_trade_start_idx[idx] && self.coin_is_valid_at(idx, k) {
                    debug_assert!(
                        !self.coin_is_tradeable_at(idx, k),
                        "coin {} flagged tradeable too early at k {} (trade_start {})",
                        idx,
                        k,
                        self.coin_trade_start_idx[idx]
                    );
                }
            }
            self.check_for_fills(k)?;
            self.update_emas(k);
            self.update_rounded_balance(k);
            self.update_trailing_prices(k);
            if self.equity_tracking_active
                && self.balance.usd_total_balance.is_finite()
                && (self.balance.usd_total_balance <= 0.0
                    || (self.fills.last().is_some_and(|fill| fill.index == k)
                        && self.hsl_fill_is_terminal(k)))
            {
                self.update_equities(k);
                if !self.check_and_apply_liquidation(k) {
                    return Err(format!(
                        "depleted balance did not trigger liquidation at k {}: {}",
                        k, self.balance.usd_total_balance
                    ));
                }
                self.record_hsl_analysis(k, true);
                break;
            }
            let current_ts = self.first_timestamp_ms + (k as u64) * self.interval_ms;
            if k > warmup_bars && current_ts >= guard_timestamp_ms {
                if self.update_n_positions_and_wallet_exposure_limits(k) {
                    self.equity_tracking_active = true;
                }
                self.initialize_btc_collateral_if_needed(k);
                self.update_hsl(k)?;
                self.update_open_orders_all(k)?;
            }
            self.force_close_delisted_positions(k)?;
            if self.equity_tracking_active {
                self.update_equities(k);
                if self.check_and_apply_liquidation(k) {
                    self.record_hsl_analysis(k, false);
                    break;
                }
                self.record_hsl_analysis(k, false);
                self.record_total_wallet_exposure();
            }
        }
        if let Some(mut writer) = self.debug_writer.take() {
            writer.finish();
        }
        if let Some(mut writer) = self.debug_balance_writer.take() {
            writer.finish();
        }
        if let Some(profile) = self.orch_profile.take() {
            profile.write_to_file();
        }
        self.final_hard_stop_metrics = Some(self.hard_stop_metrics());
        self.final_strategy_equity_metrics = Some(self.strategy_equity_metrics_for_analysis());
        let fills = std::mem::take(&mut self.fills);
        let equities = std::mem::take(&mut self.equities);
        Ok((fills, equities))
    }

    #[inline(always)]
    fn liquidation_equity_floor_usd(&self) -> f64 {
        let threshold = self.backtest_params.liquidation_threshold.max(0.0);
        self.backtest_params.starting_balance.max(0.0) * threshold
    }

    fn check_and_apply_liquidation(&mut self, k: usize) -> bool {
        let Some(&equity_usd) = self.equities.usd_total_equity.last() else {
            return false;
        };
        let floor_usd = self.liquidation_equity_floor_usd();
        let raw_balance_depleted =
            self.balance.usd_total_balance.is_finite() && self.balance.usd_total_balance <= 0.0;
        let liquidated = raw_balance_depleted
            || if floor_usd > 0.0 {
                equity_usd <= floor_usd
            } else {
                equity_usd <= 0.0
            };
        if !liquidated {
            return false;
        }
        self.liquidated = true;

        if let Some(last_usd_equity) = self.equities.usd_total_equity.last_mut() {
            *last_usd_equity = floor_usd.max(0.0);
        }
        if let Some(last_btc_equity) = self.equities.btc_total_equity.last_mut() {
            let btc_price = self.btc_usd_prices[k].max(f64::EPSILON);
            *last_btc_equity = floor_usd.max(0.0) / btc_price;
        }
        true
    }

    fn update_n_positions_and_wallet_exposure_limits(&mut self, k: usize) -> bool {
        let eligible: Vec<usize> = (0..self.n_coins)
            .filter(|&idx| self.coin_is_tradeable_at(idx, k))
            .collect();

        if eligible.is_empty() {
            return false; // nothing tradable right now
        }

        let eligible_long: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&idx| {
                self.bot_params_original[idx].long.entry_eligible
                    && self.bot_params_original[idx].long.wallet_exposure_limit != 0.0
            })
            .collect();
        let eligible_short: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&idx| {
                self.bot_params_original[idx].short.entry_eligible
                    && self.bot_params_original[idx].short.wallet_exposure_limit != 0.0
            })
            .collect();

        let tradable_long_now = eligible_long.len();
        let tradable_short_now = eligible_short.len();
        let tradable_long_for_denom = if self.backtest_params.dynamic_wel_by_tradability {
            // Grow-only tradable universe: once a coin has been tradable, later delistings
            // do not reduce the denominator.
            self.max_tradable_coins_seen.long =
                self.max_tradable_coins_seen.long.max(tradable_long_now);
            self.max_tradable_coins_seen.long
        } else {
            tradable_long_now
        };
        let tradable_short_for_denom = if self.backtest_params.dynamic_wel_by_tradability {
            self.max_tradable_coins_seen.short =
                self.max_tradable_coins_seen.short.max(tradable_short_now);
            self.max_tradable_coins_seen.short
        } else {
            tradable_short_now
        };

        // ---------- 2. denominator/effective position counts ----------
        self.effective_n_positions.long = if self.backtest_params.dynamic_wel_by_tradability {
            self.configured_n_positions
                .long
                .min(tradable_long_for_denom)
        } else {
            self.configured_n_positions.long
        };
        self.effective_n_positions.short = if self.backtest_params.dynamic_wel_by_tradability {
            self.configured_n_positions
                .short
                .min(tradable_short_for_denom)
        } else {
            self.configured_n_positions.short
        };

        // avoid division by zero (possible directly after a delisting)
        if self.effective_n_positions.long == 0 && self.effective_n_positions.short == 0 {
            return false;
        }

        // ---------- 3. dynamic WELs ----------
        let dyn_wel_long_base = if self.effective_n_positions.long > 0 {
            self.bot_params_master.long.total_wallet_exposure_limit
                / self.effective_n_positions.long as f64
        } else {
            0.0
        };
        let dyn_wel_short_base = if self.effective_n_positions.short > 0 {
            self.bot_params_master.short.total_wallet_exposure_limit
                / self.effective_n_positions.short as f64
        } else {
            0.0
        };

        // ---------- 4. apply runtime budgets without mutating config ----------
        for runtime_budget in self.runtime_budget.iter_mut() {
            runtime_budget.long.effective_n_positions = self.effective_n_positions.long;
            runtime_budget.short.effective_n_positions = self.effective_n_positions.short;
        }
        for &idx in &eligible_long {
            if self.bot_params_original[idx].long.wallet_exposure_limit < 0.0 {
                self.runtime_budget[idx]
                    .long
                    .effective_wallet_exposure_limit = dyn_wel_long_base;
            }
        }
        for &idx in &eligible_short {
            if self.bot_params_original[idx].short.wallet_exposure_limit < 0.0 {
                self.runtime_budget[idx]
                    .short
                    .effective_wallet_exposure_limit = dyn_wel_short_base;
            }
        }
        true
    }

    #[inline(always)]
    fn update_rounded_balance(&mut self, k: usize) {
        if self.balance.use_btc_collateral {
            // 1. raw, unrounded totals
            self.balance.usd_total_balance = (self.balance.btc_cash_wallet
                * self.btc_usd_prices[k])
                + self.balance.usd_cash_wallet;
            self.balance.btc_total_balance =
                self.balance.usd_total_balance / self.btc_usd_prices[k];

            // 2. apply hysteresis rounding
            self.balance.usd_total_balance_rounded = hysteresis(
                self.balance.usd_total_balance,
                self.balance.usd_total_balance_rounded,
                0.02,
            );
        }
    }

    fn initialize_btc_collateral_if_needed(&mut self, k: usize) {
        if self.btc_collateral_initialized || !self.balance.use_btc_collateral {
            return;
        }

        let btc_price = self.btc_usd_prices[k].max(f64::EPSILON);
        let total_balance = self.balance.usd_total_balance;
        let btc_value = self.balance.btc_collateral_cap * total_balance;
        self.balance.btc_cash_wallet = btc_value / btc_price;
        self.balance.usd_cash_wallet = total_balance - btc_value;
        self.balance.usd_total_balance =
            (self.balance.btc_cash_wallet * btc_price) + self.balance.usd_cash_wallet;
        self.balance.btc_total_balance = self.balance.usd_total_balance / btc_price;
        self.balance.usd_total_balance_rounded = self.balance.usd_total_balance;
        self.btc_collateral_initialized = true;
    }

    #[inline(always)]
    fn bp(&self, coin_idx: usize, pside: usize) -> &BotParams {
        match pside {
            0 => &self.bot_params[coin_idx].long,
            1 => &self.bot_params[coin_idx].short,
            _ => unreachable!("invalid pside"),
        }
    }

    #[inline(always)]
    fn runtime_budget(&self, coin_idx: usize, pside: usize) -> &RuntimeBudgetState {
        match pside {
            0 => &self.runtime_budget[coin_idx].long,
            1 => &self.runtime_budget[coin_idx].short,
            _ => unreachable!("invalid pside"),
        }
    }

    #[inline(always)]
    fn coin_valid_range(&self, idx: usize) -> Option<(usize, usize)> {
        if idx >= self.coin_first_valid_idx.len() {
            return None;
        }
        let start = self.coin_first_valid_idx[idx];
        let end = self.coin_last_valid_idx[idx];
        if start > end {
            None
        } else {
            Some((start, end))
        }
    }

    #[inline(always)]
    fn coin_is_within_valid_range_at(&self, idx: usize, k: usize) -> bool {
        self.coin_valid_range(idx)
            .map(|(start, end)| k >= start && k <= end)
            .unwrap_or(false)
    }

    #[inline(always)]
    fn coin_is_valid_at(&self, idx: usize, k: usize) -> bool {
        if !self.coin_is_within_valid_range_at(idx, k) {
            return false;
        }
        let high = self.hlcvs_value(k, idx, HIGH);
        let low = self.hlcvs_value(k, idx, LOW);
        let close = self.hlcvs_value(k, idx, CLOSE);
        !(high.is_nan() && low.is_nan() && close.is_nan())
    }

    #[inline(always)]
    fn coin_is_tradeable_at(&self, idx: usize, k: usize) -> bool {
        if idx >= self.coin_trade_start_idx.len() {
            return false;
        }
        let trade_start = self.coin_trade_start_idx[idx];
        self.coin_is_valid_at(idx, k) && k >= trade_start
    }

    fn coin_passes_min_effective_cost(&self, idx: usize, pside: usize) -> bool {
        if !self.backtest_params.filter_by_min_effective_cost {
            return true;
        }
        if idx >= self.exchange_params_list.len() {
            return false;
        }
        let price_idx = self
            .current_step
            .min(self.hlcvs.shape()[0].saturating_sub(1));
        let price = self.hlcvs_value(price_idx, idx, CLOSE);
        if !price.is_finite() || price <= 0.0 {
            return false;
        }
        let exchange = &self.exchange_params_list[idx];
        let min_cost = calc_effective_min_cost(price, exchange);
        let bot = self.bp(idx, pside);
        let strategy = match pside {
            LONG => &self.strategy_params[idx].long,
            SHORT => &self.strategy_params[idx].short,
            _ => return false,
        };
        let entry_initial_qty_pct = strategy_initial_qty_pct(strategy);
        if entry_initial_qty_pct <= 0.0 {
            return false;
        }
        let base_limit = self
            .runtime_budget(idx, pside)
            .effective_wallet_exposure_limit;
        if base_limit <= 0.0 {
            return false;
        }
        let effective_limit = wallet_exposure_limit_with_allowance_from_base(bot, base_limit);
        let projected_cost =
            self.balance.usd_total_balance * effective_limit * entry_initial_qty_pct;
        projected_cost >= min_cost
    }

    fn prune_rolling_pnl_window(&mut self, k: usize) {
        if self.pnl_lookback_bars == 0 || self.pnl_lookback_bars == usize::MAX {
            return;
        }
        while let Some(event) = self.pnl_events.front().copied() {
            if k.saturating_sub(event.k) > self.pnl_lookback_bars {
                self.pnl_events.pop_front();
                if self
                    .rolling_pnl_peak_candidates
                    .front()
                    .map(|candidate| candidate.seq == event.seq)
                    .unwrap_or(false)
                {
                    self.rolling_pnl_peak_candidates.pop_front();
                }
            } else {
                break;
            }
        }
    }

    fn record_rolling_pnl(&mut self, k: usize, pnl: f64) {
        if self.pnl_lookback_bars == 0 || self.pnl_lookback_bars == usize::MAX {
            return;
        }
        self.prune_rolling_pnl_window(k);
        let abs_cumulative_after = self.pnl_cumsum_running_net;
        let seq = self.pnl_event_seq;
        self.pnl_event_seq = self.pnl_event_seq.saturating_add(1);
        self.pnl_events.push_back(RollingPnlEvent {
            k,
            pnl,
            abs_cumulative_after,
            seq,
        });
        while let Some(candidate) = self.rolling_pnl_peak_candidates.back().copied() {
            if candidate.abs_cumulative_after <= abs_cumulative_after {
                self.rolling_pnl_peak_candidates.pop_back();
            } else {
                break;
            }
        }
        self.rolling_pnl_peak_candidates
            .push_back(RollingPnlPeakCandidate {
                seq,
                abs_cumulative_after,
            });
    }

    fn prune_coin_rolling_pnl_window(&mut self, k: usize, idx: usize, pside: usize) {
        if self.pnl_lookback_bars == 0 || self.pnl_lookback_bars == usize::MAX {
            return;
        }
        while let Some(event) = self.coin_pnl_events[pside][idx].front().copied() {
            if k.saturating_sub(event.k) > self.pnl_lookback_bars {
                self.coin_pnl_events[pside][idx].pop_front();
                if self.coin_rolling_pnl_peak_candidates[pside][idx]
                    .front()
                    .map(|candidate| candidate.seq == event.seq)
                    .unwrap_or(false)
                {
                    self.coin_rolling_pnl_peak_candidates[pside][idx].pop_front();
                }
            } else {
                break;
            }
        }
    }

    fn record_coin_rolling_pnl(&mut self, k: usize, idx: usize, pside: usize, pnl: f64) {
        self.pnl_cumsum_running_net_coin_pside[pside][idx] += pnl;
        self.pnl_cumsum_max_net_coin_pside[pside][idx] = self.pnl_cumsum_max_net_coin_pside[pside]
            [idx]
            .max(self.pnl_cumsum_running_net_coin_pside[pside][idx]);
        if self.pnl_lookback_bars == 0 || self.pnl_lookback_bars == usize::MAX {
            return;
        }
        self.prune_coin_rolling_pnl_window(k, idx, pside);
        let abs_cumulative_after = self.pnl_cumsum_running_net_coin_pside[pside][idx];
        let seq = self.coin_pnl_event_seq;
        self.coin_pnl_event_seq = self.coin_pnl_event_seq.saturating_add(1);
        self.coin_pnl_events[pside][idx].push_back(RollingPnlEvent {
            k,
            pnl,
            abs_cumulative_after,
            seq,
        });
        while let Some(candidate) = self.coin_rolling_pnl_peak_candidates[pside][idx]
            .back()
            .copied()
        {
            if candidate.abs_cumulative_after <= abs_cumulative_after {
                self.coin_rolling_pnl_peak_candidates[pside][idx].pop_back();
            } else {
                break;
            }
        }
        self.coin_rolling_pnl_peak_candidates[pside][idx].push_back(RollingPnlPeakCandidate {
            seq,
            abs_cumulative_after,
        });
    }

    #[inline]
    fn effective_pnl_cumsum(&mut self, k: usize) -> (f64, f64) {
        if self.pnl_lookback_bars == usize::MAX {
            return (self.pnl_cumsum_max_net, self.pnl_cumsum_running_net);
        }
        if self.pnl_lookback_bars > 0 {
            self.prune_rolling_pnl_window(k);
            if self.pnl_events.is_empty() {
                return (0.0, 0.0);
            }
            // Match the live contract exactly: filter fills inside the active lookback window,
            // then compute cumsum.max() / cumsum[-1] over just that filtered sequence.
            // We keep the state incrementally for speed, but derive both values from the same
            // absolute cumulative basis so the rolling peak can never fall below the current sum.
            let base_abs_cumsum = self
                .pnl_events
                .front()
                .map(|event| event.abs_cumulative_after - event.pnl)
                .unwrap_or(0.0);
            let rolling_peak = self
                .rolling_pnl_peak_candidates
                .front()
                .map(|candidate| candidate.abs_cumulative_after - base_abs_cumsum)
                .unwrap_or(0.0)
                .max(0.0);
            let rolling_current = self.pnl_cumsum_running_net - base_abs_cumsum;
            return (rolling_peak, rolling_current);
        }
        (self.pnl_cumsum_max_net, self.pnl_cumsum_running_net)
    }

    fn apply_hard_stop_mode_overrides(
        &mut self,
        idx: usize,
        mode_long: &mut Option<orchestrator::TradingMode>,
        mode_short: &mut Option<orchestrator::TradingMode>,
        _pos_long: Position,
        _pos_short: Position,
    ) {
        self.apply_hsl_modes(idx, mode_long, mode_short);
    }

    fn update_balance(&mut self, k: usize, pnl: f64, fee_paid: f64) {
        const CONVERSION_FEE_RATE: f64 = 0.001;

        // Apply fees immediately to the USD balance
        self.balance.usd_cash_wallet += fee_paid;

        let btc_price = self.btc_usd_prices[k].max(f64::EPSILON);
        self.balance.usd_cash_wallet += pnl;

        if self.balance.use_btc_collateral {
            let btc_value = self.balance.btc_cash_wallet * btc_price;
            let equity = btc_value + self.balance.usd_cash_wallet;

            if equity > 0.0 {
                let current_ratio = btc_value / equity;
                let target_cap = self.balance.btc_collateral_cap.max(0.0);
                let debt = if self.balance.usd_cash_wallet < 0.0 {
                    -self.balance.usd_cash_wallet
                } else {
                    0.0
                };
                let ltv = debt / equity;

                if target_cap > 0.0 && current_ratio + 1e-12 < target_cap {
                    let ltv_allows = match self.balance.btc_collateral_ltv_cap {
                        Some(cap) if cap.is_finite() && cap > 0.0 => ltv + 1e-12 < cap,
                        _ => true,
                    };

                    if ltv_allows {
                        let mut usd_to_spend = (target_cap - current_ratio) * equity;

                        if let Some(cap) = self.balance.btc_collateral_ltv_cap {
                            if cap.is_finite() && cap > 0.0 {
                                let max_debt = cap * equity;
                                let allowable_extra_debt = (max_debt - debt).max(0.0);
                                if usd_to_spend > allowable_extra_debt {
                                    usd_to_spend = allowable_extra_debt;
                                }
                            }
                        }

                        if usd_to_spend > 0.0 {
                            self.balance.usd_cash_wallet -= usd_to_spend;
                            let usd_after_fee = usd_to_spend * (1.0 - CONVERSION_FEE_RATE);
                            self.balance.btc_cash_wallet += usd_after_fee / btc_price;
                        }
                    }
                }
            } else {
                // Account is effectively depleted; reset BTC balance
                self.balance.btc_cash_wallet = 0.0;
            }
        } else {
            self.balance.usd_total_balance = self.balance.usd_cash_wallet;
            self.balance.usd_total_balance_rounded = self.balance.usd_cash_wallet;
            self.balance.btc_total_balance = self.balance.usd_total_balance / btc_price;
            return;
        }

        // Update total balances based on latest BTC amount and USD balance
        let new_btc_value = self.balance.btc_cash_wallet * btc_price;
        self.balance.usd_total_balance = new_btc_value + self.balance.usd_cash_wallet;
        self.balance.btc_total_balance = self.balance.usd_total_balance / btc_price;
        self.balance.usd_total_balance_rounded = hysteresis(
            self.balance.usd_total_balance,
            self.balance.usd_total_balance_rounded,
            0.02,
        );
    }

    fn current_usd_equity_at(&self, k: usize) -> f64 {
        let mut equity_usd = self.balance.usd_total_balance;

        for (idx, position) in self.positions.long.iter().enumerate() {
            if position.size == 0.0 || !self.coin_is_valid_at(idx, k) {
                continue;
            }
            let current_price = self.hlcvs_value(k, idx, CLOSE);
            if !current_price.is_finite() {
                continue;
            }
            let upnl = calc_pnl_long(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity_usd += upnl;
        }

        for (idx, position) in self.positions.short.iter().enumerate() {
            if position.size == 0.0 || !self.coin_is_valid_at(idx, k) {
                continue;
            }
            let current_price = self.hlcvs_value(k, idx, CLOSE);
            if !current_price.is_finite() {
                continue;
            }
            let upnl = calc_pnl_short(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity_usd += upnl;
        }

        equity_usd
    }

    fn record_hard_stop_panic_close_loss(
        &mut self,
        pside: usize,
        idx: usize,
        k: usize,
        net_pnl: f64,
    ) {
        let key = self.hsl_report_key(pside, idx);
        // Fills precede the bar's ordinary balance revaluation. Observe
        // current collateral here without mutating trading balances.
        let balance = if self.balance.use_btc_collateral {
            self.balance.btc_cash_wallet * self.btc_usd_prices[k] + self.balance.usd_cash_wallet
        } else {
            self.balance.usd_total_balance
        };
        let equity =
            balance + self.unrealized_pnl_pside(LONG, k) + self.unrealized_pnl_pside(SHORT, k);
        self.hsl_report.panic_fill(key, net_pnl, equity);
    }

    fn update_equities(&mut self, k: usize) {
        let equity_usd = self.current_usd_equity_at(k);
        let btc_price = self.btc_usd_prices[k].max(f64::EPSILON);
        let equity_btc = equity_usd / btc_price;

        // Finally push the results into the Equities struct
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * self.interval_ms;
        self.equities.usd_total_equity.push(equity_usd);
        self.equities.btc_total_equity.push(equity_btc);
        self.equities.timestamps_ms.push(timestamp_ms);
    }

    fn record_total_wallet_exposure(&mut self) {
        // For analysis time series we record the net TWE (long + short, where short is negative).
        let (_, _, twe_net) = self.compute_twe_components();
        self.total_wallet_exposures.push(twe_net);
    }

    fn compute_twe_components(&self) -> (f64, f64, f64) {
        let mut twe_long = 0.0;
        let mut twe_short = 0.0;
        for (idx, position) in self.positions.long.iter().enumerate() {
            if position.size != 0.0 {
                twe_long += calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_balance,
                    position.size.abs(),
                    position.price,
                );
            }
        }
        for (idx, position) in self.positions.short.iter().enumerate() {
            if position.size != 0.0 {
                twe_short -= calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_balance,
                    position.size.abs(),
                    position.price,
                );
            }
        }
        let twe_net = twe_long + twe_short;
        (twe_long, twe_short, twe_net)
    }

    fn check_for_fills(&mut self, k: usize) -> Result<(), String> {
        self.did_fill_long.fill(false);
        self.did_fill_short.fill(false);
        if self.trading_enabled.long
            || (self.hsl_enabled() && self.positions.long.iter().any(|p| p.size != 0.0))
        {
            for idx in 0..self.n_coins {
                // Process close fills long
                if !self.open_orders.long[idx].closes.is_empty() {
                    let mut closes_to_process = Vec::new();
                    {
                        for close_order in &self.open_orders.long[idx].closes {
                            if let Some(exec) = self.order_fill_execution(k, idx, close_order) {
                                closes_to_process.push((close_order.order, exec));
                            }
                        }
                    }
                    for (order, exec) in closes_to_process {
                        if self.positions.long[idx].size != 0.0 {
                            self.did_fill_long[idx] = true;
                            self.process_close_fill_long(k, idx, &order, exec)?;
                        }
                    }
                }
                // Process entry fills long
                if self.trading_enabled.long && !self.open_orders.long[idx].entries.is_empty() {
                    let mut entries_to_process = Vec::new();
                    {
                        for entry_order in &self.open_orders.long[idx].entries {
                            if let Some(exec) = self.order_fill_execution(k, idx, entry_order) {
                                entries_to_process.push((entry_order.order, exec));
                            }
                        }
                    }
                    for (order, exec) in entries_to_process {
                        if self.hsl_enabled()
                            && self.hsl_action(LONG, idx) != crate::hsl_controller::Action::Normal
                        {
                            continue;
                        }
                        self.did_fill_long[idx] = true;
                        self.last_increase_fill_timestamp_long[idx] =
                            Some(self.first_timestamp_ms + (k as u64) * self.interval_ms);
                        self.process_entry_fill_long(k, idx, &order, exec);
                    }
                }
            }
        }
        if self.trading_enabled.short
            || (self.hsl_enabled() && self.positions.short.iter().any(|p| p.size != 0.0))
        {
            for idx in 0..self.n_coins {
                // Process close fills short
                if !self.open_orders.short[idx].closes.is_empty() {
                    let mut closes_to_process = Vec::new();
                    {
                        for close_order in &self.open_orders.short[idx].closes {
                            if let Some(exec) = self.order_fill_execution(k, idx, close_order) {
                                closes_to_process.push((close_order.order, exec));
                            }
                        }
                    }
                    for (order, exec) in closes_to_process {
                        if self.positions.short[idx].size != 0.0 {
                            self.did_fill_short[idx] = true;
                            self.process_close_fill_short(k, idx, &order, exec)?;
                        }
                    }
                }
                // Process entry fills short
                if self.trading_enabled.short && !self.open_orders.short[idx].entries.is_empty() {
                    let mut entries_to_process = Vec::new();
                    {
                        for entry_order in &self.open_orders.short[idx].entries {
                            if let Some(exec) = self.order_fill_execution(k, idx, entry_order) {
                                entries_to_process.push((entry_order.order, exec));
                            }
                        }
                    }
                    for (order, exec) in entries_to_process {
                        if self.hsl_enabled()
                            && self.hsl_action(SHORT, idx) != crate::hsl_controller::Action::Normal
                        {
                            continue;
                        }
                        self.did_fill_short[idx] = true;
                        self.last_increase_fill_timestamp_short[idx] =
                            Some(self.first_timestamp_ms + (k as u64) * self.interval_ms);
                        self.process_entry_fill_short(k, idx, &order, exec);
                    }
                }
            }
        }
        Ok(())
    }

    /// Consume an exact fill boundary before another fill can reopen its scope.
    /// Account for a scope flattening before a possible same-bar re-entry.
    fn finish_hard_stop_episode_at_fill(
        &mut self,
        k: usize,
        idx: usize,
        filled_pside: usize,
    ) -> Result<(), String> {
        if !self.balance.usd_total_balance.is_finite() {
            return Err(format!("non-finite balance at HSL fill boundary: k {}", k));
        }
        self.finish_hsl_flat(k, idx, filled_pside)
    }

    fn process_close_fill_long(
        &mut self,
        k: usize,
        idx: usize,
        close_fill: &Order,
        exec: OrderFillExecution,
    ) -> Result<(), String> {
        let current_position = self.positions.long[idx];
        let mut new_psize = round_(
            current_position.size + close_fill.qty,
            self.exchange_params_list[idx].qty_step,
        );
        let mut adjusted_close_qty = close_fill.qty;
        if new_psize < 0.0 {
            println!("warning: close qty greater than psize long");
            println!("coin: {}", self.backtest_params.coins[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", close_fill);
            println!("bot config: {:?}", self.bp(idx, LONG));
            new_psize = 0.0;
            adjusted_close_qty = -current_position.size;
        }
        let fee_paid = -qty_to_cost(
            adjusted_close_qty,
            exec.price,
            self.exchange_params_list[idx].c_mult,
        ) * exec.fee_rate;
        let pnl = calc_pnl_long(
            current_position.price,
            exec.price,
            adjusted_close_qty,
            self.exchange_params_list[idx].c_mult,
        );
        self.pnl_cumsum_running += pnl;
        self.pnl_cumsum_max = self.pnl_cumsum_max.max(self.pnl_cumsum_running);
        self.pnl_cumsum_running_net += pnl + fee_paid;
        self.pnl_cumsum_max_net = self.pnl_cumsum_max_net.max(self.pnl_cumsum_running_net);
        self.pnl_cumsum_running_net_pside[LONG] += pnl + fee_paid;
        self.record_rolling_pnl(k, pnl + fee_paid);
        self.record_coin_rolling_pnl(k, idx, LONG, pnl + fee_paid);
        if matches!(close_fill.order_type, OrderType::ClosePanicLong) {
            self.record_hard_stop_panic_close_loss(LONG, idx, k, pnl + fee_paid);
        }
        let balance_before = self.snapshot_balance();
        self.update_balance(k, pnl, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "close_long",
            close_fill,
            adjusted_close_qty,
            exec.price,
            pnl,
            fee_paid,
            balance_before,
            balance_after,
        );

        let current_pprice = current_position.price;
        if new_psize == 0.0 {
            self.positions.long[idx] = Position::default();
        } else {
            self.positions.long[idx].size = new_psize;
        }
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * self.interval_ms;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                current_pprice,
            ) * new_psize.signum()
        } else {
            0.0
        };
        let (twe_long, twe_short, twe_net) = self.compute_twe_components();
        self.fills.push(Fill {
            index: k, // index minute
            timestamp_ms,
            coin: self.backtest_params.coins[idx].clone(), // coin
            pnl,                                           // realized pnl
            fee_paid,                                      // fee paid
            usd_total_balance: self.balance.usd_total_balance,
            btc_cash_wallet: self.balance.btc_cash_wallet,
            usd_cash_wallet: self.balance.usd_cash_wallet,
            btc_price: self.btc_usd_prices[k],         // Added
            fill_qty: adjusted_close_qty,              // fill qty
            fill_price: exec.price,                    // fill price
            position_size: new_psize,                  // psize after fill
            position_price: current_pprice,            // pprice after fill
            order_type: close_fill.order_type.clone(), // fill type
            liquidity: exec.liquidity.to_string(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
        if new_psize == 0.0 && current_position.size != 0.0 {
            self.finish_hard_stop_episode_at_fill(k, idx, LONG)?;
        }
        Ok(())
    }

    fn process_close_fill_short(
        &mut self,
        k: usize,
        idx: usize,
        order: &Order,
        exec: OrderFillExecution,
    ) -> Result<(), String> {
        let current_position = self.positions.short[idx];
        let mut new_psize = round_(
            current_position.size + order.qty,
            self.exchange_params_list[idx].qty_step,
        );
        let mut adjusted_close_qty = order.qty;
        if new_psize > 0.0 {
            println!("warning: close qty greater than psize short");
            println!("coin: {}", self.backtest_params.coins[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", order);
            new_psize = 0.0;
            adjusted_close_qty = current_position.size.abs();
        }
        let fee_paid = -qty_to_cost(
            adjusted_close_qty,
            exec.price,
            self.exchange_params_list[idx].c_mult,
        ) * exec.fee_rate;
        let pnl = calc_pnl_short(
            current_position.price,
            exec.price,
            adjusted_close_qty,
            self.exchange_params_list[idx].c_mult,
        );
        self.pnl_cumsum_running += pnl;
        self.pnl_cumsum_max = self.pnl_cumsum_max.max(self.pnl_cumsum_running);
        self.pnl_cumsum_running_net += pnl + fee_paid;
        self.pnl_cumsum_max_net = self.pnl_cumsum_max_net.max(self.pnl_cumsum_running_net);
        self.pnl_cumsum_running_net_pside[SHORT] += pnl + fee_paid;
        self.record_rolling_pnl(k, pnl + fee_paid);
        self.record_coin_rolling_pnl(k, idx, SHORT, pnl + fee_paid);
        if matches!(order.order_type, OrderType::ClosePanicShort) {
            self.record_hard_stop_panic_close_loss(SHORT, idx, k, pnl + fee_paid);
        }
        let balance_before = self.snapshot_balance();
        self.update_balance(k, pnl, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "close_short",
            order,
            adjusted_close_qty,
            exec.price,
            pnl,
            fee_paid,
            balance_before,
            balance_after,
        );

        let current_pprice = current_position.price;
        if new_psize == 0.0 {
            self.positions.short[idx] = Position::default();
        } else {
            self.positions.short[idx].size = new_psize;
        }
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * self.interval_ms;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                current_pprice,
            ) * new_psize.signum()
        } else {
            0.0
        };
        let (twe_long, twe_short, twe_net) = self.compute_twe_components();
        self.fills.push(Fill {
            index: k, // index minute
            timestamp_ms,
            coin: self.backtest_params.coins[idx].clone(), // coin
            pnl,                                           // realized pnl
            fee_paid,                                      // fee paid
            usd_total_balance: self.balance.usd_total_balance,
            btc_cash_wallet: self.balance.btc_cash_wallet,
            usd_cash_wallet: self.balance.usd_cash_wallet,
            btc_price: self.btc_usd_prices[k],
            fill_qty: adjusted_close_qty,
            fill_price: exec.price,
            position_size: new_psize,
            position_price: current_pprice,
            order_type: order.order_type.clone(),
            liquidity: exec.liquidity.to_string(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
        if new_psize == 0.0 && current_position.size != 0.0 {
            self.finish_hard_stop_episode_at_fill(k, idx, SHORT)?;
        }
        Ok(())
    }

    fn process_entry_fill_long(
        &mut self,
        k: usize,
        idx: usize,
        order: &Order,
        exec: OrderFillExecution,
    ) {
        // long entry fill
        let fee_paid = -qty_to_cost(order.qty, exec.price, self.exchange_params_list[idx].c_mult)
            * exec.fee_rate;
        self.pnl_cumsum_running_net += fee_paid;
        self.pnl_cumsum_max_net = self.pnl_cumsum_max_net.max(self.pnl_cumsum_running_net);
        self.pnl_cumsum_running_net_pside[LONG] += fee_paid;
        self.record_rolling_pnl(k, fee_paid);
        self.record_coin_rolling_pnl(k, idx, LONG, fee_paid);
        let balance_before = self.snapshot_balance();
        self.update_balance(k, 0.0, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "entry_long",
            order,
            order.qty,
            exec.price,
            0.0,
            fee_paid,
            balance_before,
            balance_after,
        );

        let (new_psize, new_pprice) = calc_new_psize_pprice(
            self.positions.long[idx].size,
            self.positions.long[idx].price,
            order.qty,
            exec.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.long[idx].size = new_psize;
        self.positions.long[idx].price = new_pprice;
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * self.interval_ms;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                new_pprice,
            ) * new_psize.signum()
        } else {
            0.0
        };
        let (twe_long, twe_short, twe_net) = self.compute_twe_components();
        self.fills.push(Fill {
            index: k,
            timestamp_ms,
            coin: self.backtest_params.coins[idx].clone(),
            pnl: 0.0,
            fee_paid,
            usd_total_balance: self.balance.usd_total_balance,
            btc_cash_wallet: self.balance.btc_cash_wallet,
            usd_cash_wallet: self.balance.usd_cash_wallet,
            btc_price: self.btc_usd_prices[k],
            fill_qty: order.qty,
            fill_price: exec.price,
            position_size: self.positions.long[idx].size,
            position_price: self.positions.long[idx].price,
            order_type: order.order_type.clone(),
            liquidity: exec.liquidity.to_string(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
    }

    fn process_entry_fill_short(
        &mut self,
        k: usize,
        idx: usize,
        order: &Order,
        exec: OrderFillExecution,
    ) {
        // short entry fill
        let fee_paid = -qty_to_cost(order.qty, exec.price, self.exchange_params_list[idx].c_mult)
            * exec.fee_rate;
        self.pnl_cumsum_running_net += fee_paid;
        self.pnl_cumsum_max_net = self.pnl_cumsum_max_net.max(self.pnl_cumsum_running_net);
        self.pnl_cumsum_running_net_pside[SHORT] += fee_paid;
        self.record_rolling_pnl(k, fee_paid);
        self.record_coin_rolling_pnl(k, idx, SHORT, fee_paid);
        let balance_before = self.snapshot_balance();
        self.update_balance(k, 0.0, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "entry_short",
            order,
            order.qty,
            exec.price,
            0.0,
            fee_paid,
            balance_before,
            balance_after,
        );
        let (new_psize, new_pprice) = calc_new_psize_pprice(
            self.positions.short[idx].size,
            self.positions.short[idx].price,
            order.qty,
            exec.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.short[idx].size = new_psize;
        self.positions.short[idx].price = new_pprice;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                new_pprice,
            ) * new_psize.signum()
        } else {
            0.0
        };
        let (twe_long, twe_short, twe_net) = self.compute_twe_components();
        self.fills.push(Fill {
            index: k,
            timestamp_ms: self.first_timestamp_ms + (k as u64) * self.interval_ms,
            coin: self.backtest_params.coins[idx].clone(),
            pnl: 0.0,
            fee_paid,
            usd_total_balance: self.balance.usd_total_balance,
            btc_cash_wallet: self.balance.btc_cash_wallet,
            usd_cash_wallet: self.balance.usd_cash_wallet,
            btc_price: self.btc_usd_prices[k],
            fill_qty: order.qty,
            fill_price: exec.price,
            position_size: self.positions.short[idx].size,
            position_price: self.positions.short[idx].price,
            order_type: order.order_type.clone(),
            liquidity: exec.liquidity.to_string(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
    }

    fn update_trailing_prices(&mut self, k: usize) {
        // ----- LONG side -----
        if self.trading_enabled.long && self.any_trailing_long {
            for (idx, position) in self.positions.long.iter().enumerate() {
                if position.size == 0.0 {
                    continue;
                }
                if !self.trailing_enabled[idx].long {
                    continue;
                }
                if !self.coin_is_valid_at(idx, k) {
                    continue;
                }
                let fill_long = self.did_fill_long[idx];
                let col = self.active_coin_indices[idx];
                let low = self.hlcvs[[k, col, LOW]];
                let high = self.hlcvs[[k, col, HIGH]];
                let close = self.hlcvs[[k, col, CLOSE]];
                let bundle = &mut self.trailing_prices.long[idx];
                if fill_long {
                    reset_trailing_bundle(bundle);
                } else {
                    update_trailing_bundle_with_candle(bundle, high, low, close);
                }
            }
        }

        // ----- SHORT side -----
        if self.trading_enabled.short && self.any_trailing_short {
            for (idx, position) in self.positions.short.iter().enumerate() {
                if position.size == 0.0 {
                    continue;
                }
                if !self.trailing_enabled[idx].short {
                    continue;
                }
                if !self.coin_is_valid_at(idx, k) {
                    continue;
                }
                let fill_short = self.did_fill_short[idx];
                let col = self.col(idx);
                let low = self.hlcvs[[k, col, LOW]];
                let high = self.hlcvs[[k, col, HIGH]];
                let close = self.hlcvs[[k, col, CLOSE]];
                let bundle = &mut self.trailing_prices.short[idx];
                if fill_short {
                    reset_trailing_bundle(bundle);
                } else {
                    update_trailing_bundle_with_candle(bundle, high, low, close);
                }
            }
        }
    }

    fn order_can_fill(&self, k: usize, idx: usize, order: &Order) -> bool {
        if self.hsl_enabled()
            && matches!(
                order.order_type,
                OrderType::ClosePanicLong | OrderType::ClosePanicShort
            )
        {
            // Protective closes require a real current candle, not entry warmup.
            self.coin_is_valid_at(idx, k)
        } else {
            self.coin_is_tradeable_at(idx, k)
        }
    }

    fn order_filled(&self, k: usize, idx: usize, order: &Order) -> bool {
        if !self.order_can_fill(k, idx, order) {
            return false;
        }
        // check if filled in current candle (pass k+1 to check if will fill in next candle)
        crate::limit_fills::crosses_limit(
            self.hlcvs_value(k, idx, LOW),
            self.hlcvs_value(k, idx, HIGH),
            order.qty,
            order.price,
            self.backtest_params.limit_order_fill_buffer_pct,
        )
    }

    fn force_close_delisted_positions(&mut self, k: usize) -> Result<(), String> {
        for idx in 0..self.n_coins {
            if self.last_valid_timestamps.get(idx).copied().flatten() != Some(k) {
                continue;
            }
            if !self.coin_is_valid_at(idx, k) {
                continue;
            }

            let mut closed_any = false;
            let long_size = self.positions.long[idx].size;
            if long_size > 0.0 {
                let close_qty = -long_size;
                if let Some(price) = self.market_fill_price_for_qty(k, idx, close_qty) {
                    let order = Order {
                        qty: close_qty,
                        price,
                        order_type: OrderType::ClosePanicLong,
                    };
                    let exec = OrderFillExecution {
                        price,
                        fee_rate: self.exchange_params_list[idx].taker_fee,
                        liquidity: "taker",
                    };
                    self.did_fill_long[idx] = true;
                    self.process_close_fill_long(k, idx, &order, exec)?;
                    closed_any = true;
                }
            }

            let short_size = self.positions.short[idx].size;
            if short_size < 0.0 {
                let close_qty = -short_size;
                if let Some(price) = self.market_fill_price_for_qty(k, idx, close_qty) {
                    let order = Order {
                        qty: close_qty,
                        price,
                        order_type: OrderType::ClosePanicShort,
                    };
                    let exec = OrderFillExecution {
                        price,
                        fee_rate: self.exchange_params_list[idx].taker_fee,
                        liquidity: "taker",
                    };
                    self.did_fill_short[idx] = true;
                    self.process_close_fill_short(k, idx, &order, exec)?;
                    closed_any = true;
                }
            }

            if closed_any {
                self.open_orders.long[idx] = OpenOrderBundle::default();
                self.open_orders.short[idx] = OpenOrderBundle::default();
            }
        }
        Ok(())
    }

    fn hard_stop_coin_slot_n_positions(&self, pside: usize) -> usize {
        if self.backtest_params.dynamic_wel_by_tradability {
            match pside {
                LONG => self.effective_n_positions.long,
                SHORT => self.effective_n_positions.short,
                _ => 0,
            }
        } else {
            match pside {
                LONG => self.configured_n_positions.long,
                SHORT => self.configured_n_positions.short,
                _ => 0,
            }
        }
    }

    fn unrealized_pnl_pside(&self, pside: usize, k: usize) -> f64 {
        let mut upnl = 0.0;
        match pside {
            LONG => {
                for (idx, position) in self.positions.long.iter().enumerate() {
                    if position.size == 0.0 {
                        continue;
                    }
                    if !self.coin_is_valid_at(idx, k) {
                        continue;
                    }
                    let current_price = self.hlcvs_value(k, idx, CLOSE);
                    if !current_price.is_finite() {
                        continue;
                    }
                    upnl += calc_pnl_long(
                        position.price,
                        current_price,
                        position.size,
                        self.exchange_params_list[idx].c_mult,
                    );
                }
            }
            SHORT => {
                for (idx, position) in self.positions.short.iter().enumerate() {
                    if position.size == 0.0 {
                        continue;
                    }
                    if !self.coin_is_valid_at(idx, k) {
                        continue;
                    }
                    let current_price = self.hlcvs_value(k, idx, CLOSE);
                    if !current_price.is_finite() {
                        continue;
                    }
                    upnl += calc_pnl_short(
                        position.price,
                        current_price,
                        position.size,
                        self.exchange_params_list[idx].c_mult,
                    );
                }
            }
            _ => unreachable!("invalid pside"),
        }
        upnl
    }

    fn market_fill_price_for_qty(&self, k: usize, idx: usize, qty: f64) -> Option<f64> {
        let close_price = self.hlcvs_value(k, idx, CLOSE).max(f64::EPSILON);
        let price_step = self.exchange_params_list[idx].price_step.max(f64::EPSILON);
        let slippage_pct = self.backtest_params.market_order_slippage_pct.max(0.0);
        if qty > 0.0 {
            let slipped = close_price * (1.0 + slippage_pct);
            Some(round_up(slipped, price_step).max(price_step))
        } else if qty < 0.0 {
            let slipped = close_price * (1.0 - slippage_pct);
            Some(round_dn(slipped, price_step).max(price_step))
        } else {
            None
        }
    }

    fn market_fill_price(&self, k: usize, idx: usize, order: &Order) -> Option<f64> {
        if !self.order_can_fill(k, idx, order) {
            return None;
        }
        self.market_fill_price_for_qty(k, idx, order.qty)
    }

    fn order_uses_market_execution(&self, _idx: usize, order: &BacktestOrder) -> bool {
        order.execution_type == orchestrator::ExecutionType::Market
    }

    fn order_fill_execution(
        &self,
        k: usize,
        idx: usize,
        order: &BacktestOrder,
    ) -> Option<OrderFillExecution> {
        if self.order_uses_market_execution(idx, order) {
            return self
                .market_fill_price(k, idx, &order.order)
                .map(|price| OrderFillExecution {
                    price,
                    fee_rate: self.exchange_params_list[idx].taker_fee,
                    liquidity: "taker",
                });
        }
        if self.order_filled(k, idx, &order.order) {
            return Some(OrderFillExecution {
                price: order.order.price,
                fee_rate: self.exchange_params_list[idx].maker_fee,
                liquidity: "maker",
            });
        }
        None
    }

    fn update_open_orders_all(&mut self, k: usize) -> Result<(), String> {
        self.update_open_orders_all_orchestrator(k)
    }

    fn forager_hysteresis_state_from_open_orders(&self) -> ForagerHysteresisState {
        let mut incumbent_long: HashSet<usize> = HashSet::new();
        let mut incumbent_short: HashSet<usize> = HashSet::new();
        for (idx, bundle) in self.open_orders.long.iter().enumerate() {
            let flat = self
                .positions
                .long
                .get(idx)
                .map(|pos| pos.size == 0.0)
                .unwrap_or(true);
            if flat && !bundle.entries.is_empty() {
                incumbent_long.insert(idx);
            }
        }
        for (idx, bundle) in self.open_orders.short.iter().enumerate() {
            let flat = self
                .positions
                .short
                .get(idx)
                .map(|pos| pos.size == 0.0)
                .unwrap_or(true);
            if flat && !bundle.entries.is_empty() {
                incumbent_short.insert(idx);
            }
        }
        ForagerHysteresisState {
            score_hysteresis_pct: self.backtest_params.forager_score_hysteresis_pct,
            incumbent_long,
            incumbent_short,
        }
    }

    fn update_open_orders_all_orchestrator(&mut self, k: usize) -> Result<(), String> {
        let total_t0 = Instant::now();
        if let Some(p) = self.orch_profile.as_mut() {
            p.steps = p.steps.saturating_add(1);
        }

        let t0 = Instant::now();
        let forager_hysteresis = self.forager_hysteresis_state_from_open_orders();
        self.open_orders.clear_all();
        if let Some(p) = self.orch_profile.as_mut() {
            OrchProfile::add_ns(&mut p.clear_orders_ns, t0.elapsed());
        }

        // Backtest-only peek: if next order will fill next candle, expand the full grid.
        // The orchestrator can do this internally when provided `next_candle` in the input.
        let t0 = Instant::now();
        let peek_hints: Option<EntryPeekHints> = None;
        if let Some(p) = self.orch_profile.as_mut() {
            OrchProfile::add_ns(&mut p.peek_hints_ns, t0.elapsed());
        }

        // Debug: dump exact unstuck calculation inputs/components for parity investigations.
        let long_position_keys: Vec<usize> = self
            .positions
            .long
            .iter()
            .enumerate()
            .filter_map(|(idx, position)| (position.size != 0.0).then_some(idx))
            .collect();
        for idx in long_position_keys {
            self.debug_dump_unstuck_calc(k, idx, LONG);
        }
        let short_position_keys: Vec<usize> = self
            .positions
            .short
            .iter()
            .enumerate()
            .filter_map(|(idx, position)| (position.size != 0.0).then_some(idx))
            .collect();
        for idx in short_position_keys {
            self.debug_dump_unstuck_calc(k, idx, SHORT);
        }

        let (res, input_update_elapsed, compute_elapsed) = {
            let t0 = Instant::now();
            let input = self.get_orchestrator_input_cached(k, peek_hints, Some(forager_hysteresis));
            let input_update_elapsed = t0.elapsed();

            let t1 = Instant::now();
            let mut res = orchestrator::compute_ideal_orders_with_workspace(
                &input,
                &mut self.orchestrator_workspace,
            )
            .map_err(|e| format!("orchestrator error at k {}: {:?}", k, e))?;
            if self.hsl_enabled() {
                res.orders.retain(|order| {
                    let side = if order.pside == orchestrator::PositionSide::Long {
                        LONG
                    } else {
                        SHORT
                    };
                    self.hsl_action(side, order.symbol_idx) == crate::hsl_controller::Action::Normal
                });
                res.orders.extend(self.hsl_protective_orders(k)?);
            }
            let compute_elapsed = t1.elapsed();
            self.orchestrator_input_cache = Some(input);
            (res, input_update_elapsed, compute_elapsed)
        };
        if let Some(p) = self.orch_profile.as_mut() {
            OrchProfile::add_ns(&mut p.input_update_ns, input_update_elapsed);
            OrchProfile::add_ns(&mut p.compute_ns, compute_elapsed);
        }

        let t0 = Instant::now();
        for o in res.orders {
            let order = Order {
                qty: o.qty,
                price: o.price,
                order_type: o.order_type,
            };
            let bt_order = BacktestOrder {
                order: order.clone(),
                execution_type: o.execution_type,
            };
            match o.pside {
                orchestrator::PositionSide::Long => {
                    let bundle = &mut self.open_orders.long[o.symbol_idx];
                    if orchestrator::is_close_order_type(order.order_type) {
                        bundle.closes.push(bt_order);
                    } else {
                        bundle.entries.push(bt_order);
                    }
                }
                orchestrator::PositionSide::Short => {
                    let bundle = &mut self.open_orders.short[o.symbol_idx];
                    if orchestrator::is_close_order_type(order.order_type) {
                        bundle.closes.push(bt_order);
                    } else {
                        bundle.entries.push(bt_order);
                    }
                }
            }
        }
        if let Some(p) = self.orch_profile.as_mut() {
            OrchProfile::add_ns(&mut p.distribute_ns, t0.elapsed());
        }

        // The orchestrator guarantees deterministic per-symbol entry/close ordering; we preserve
        // insertion order here to avoid any extra per-step sort pass in the backtester.

        self.record_debug_orders_stage(k, "orch_final");

        if let Some(p) = self.orch_profile.as_mut() {
            OrchProfile::add_ns(&mut p.total_ns, total_t0.elapsed());
        }
        Ok(())
    }

    fn record_debug_orders_stage(&mut self, k: usize, stage: &'static str) {
        if !DEBUG_DUMP_ORDERS
            || (k > DEBUG_MAX_STEPS
                && DEBUG_EXTRA_WINDOW
                    .map(|(start, end)| k < start || k > end)
                    .unwrap_or(true))
        {
            return;
        }
        if self.debug_writer.is_none() {
            return;
        };

        let want_coin = DEBUG_COIN_FILTER;
        let mut snapshots: Vec<DebugOrderSnapshot> = Vec::new();

        for (idx, bundle) in self.open_orders.long.iter().enumerate() {
            if bundle.entries.is_empty() && bundle.closes.is_empty() {
                continue;
            }
            let coin = self
                .backtest_params
                .coins
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("idx_{idx}"));
            if let Some(w) = want_coin {
                if coin != w {
                    continue;
                }
            }
            let position = self.positions.long[idx];
            let close_price = self.hlcvs_value(k, idx, CLOSE);
            let mut entries = Vec::with_capacity(bundle.entries.len());
            for o in &bundle.entries {
                entries.push(DebugOrder {
                    qty: o.order.qty,
                    price: o.order.price,
                    order_type_id: o.order.order_type.id(),
                    reduce_only: false,
                });
            }
            let mut closes = Vec::with_capacity(bundle.closes.len());
            for o in &bundle.closes {
                closes.push(DebugOrder {
                    qty: o.order.qty,
                    price: o.order.price,
                    order_type_id: o.order.order_type.id(),
                    reduce_only: true,
                });
            }
            let snapshot = DebugOrderSnapshot {
                step: k,
                side: "long",
                idx,
                coin,
                stage,
                pos_size: position.size,
                pos_price: position.price,
                close_price,
                entries,
                closes,
            };
            snapshots.push(snapshot);
        }
        for (idx, bundle) in self.open_orders.short.iter().enumerate() {
            if bundle.entries.is_empty() && bundle.closes.is_empty() {
                continue;
            }
            let coin = self
                .backtest_params
                .coins
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("idx_{idx}"));
            if let Some(w) = want_coin {
                if coin != w {
                    continue;
                }
            }
            let position = self.positions.short[idx];
            let close_price = self.hlcvs_value(k, idx, CLOSE);
            let mut entries = Vec::with_capacity(bundle.entries.len());
            for o in &bundle.entries {
                entries.push(DebugOrder {
                    qty: o.order.qty,
                    price: o.order.price,
                    order_type_id: o.order.order_type.id(),
                    reduce_only: false,
                });
            }
            let mut closes = Vec::with_capacity(bundle.closes.len());
            for o in &bundle.closes {
                closes.push(DebugOrder {
                    qty: o.order.qty,
                    price: o.order.price,
                    order_type_id: o.order.order_type.id(),
                    reduce_only: true,
                });
            }
            let snapshot = DebugOrderSnapshot {
                step: k,
                side: "short",
                idx,
                coin,
                stage,
                pos_size: position.size,
                pos_price: position.price,
                close_price,
                entries,
                closes,
            };
            snapshots.push(snapshot);
        }

        let Some(writer) = self.debug_writer.as_mut() else {
            return;
        };
        for s in &snapshots {
            writer.write_snapshot(s);
        }
    }

    fn unilateralness_warmup_spans(&self, idx: usize, k: usize) -> Vec<f64> {
        let Some((start, end)) = self.coin_valid_range(idx) else {
            return Vec::new();
        };
        self.unilateralness[idx].iter()
            .filter(|(span, _)| {
                k <= end
                    && k < start.saturating_add(
                        crate::unilateralness::warmup_returns(*span)
                            .expect("validated unilateralness span"),
                    )
            })
            .map(|(span, _)| *span)
            .collect()
    }

    fn update_unilateralness(&mut self, k: usize) -> Result<(), String> {
        for idx in 0..self.n_coins {
            if self.unilateralness[idx].is_empty() || !self.coin_is_valid_at(idx, k) {
                continue;
            }
            let close = self.hlcvs_value(k, idx, CLOSE);
            for (_, tracker) in &mut self.unilateralness[idx] {
                tracker.push_close(close)?;
            }
        }
        self.unilateralness_step = Some(k);
        Ok(())
    }

    #[inline]
    fn unilateralness_at(&self, idx: usize, k: usize) -> Vec<(f64, f64)> {
        if self.unilateralness_step == Some(k) {
            if !self.coin_is_valid_at(idx, k) {
                return Vec::new();
            }
            return self.unilateralness[idx]
                .iter()
                .filter_map(|(span, tracker)| tracker.score().map(|value| (*span, value)))
                .collect();
        }
        // Standalone snapshots may request another index. Replay that window
        // rather than use a cached score from a different candle.
        let mut out = Vec::new();
        for (span, _) in &self.unilateralness[idx] {
            let span = *span;
            let n =
                crate::unilateralness::warmup_returns(span).expect("validated unilateralness span");
            let (start, end) = self.coin_valid_range(idx).unwrap_or((0, 0));
            if k < start.saturating_add(n) || k > end {
                continue;
            }
            let closes: Vec<f64> = (k - n..=k)
                .map(|j| self.hlcvs_value(j, idx, CLOSE))
                .collect();
            let score = crate::unilateralness::signed_rms(&closes, span)
                .expect("validated unilateralness candle window");
            out.push((span, score));
        }
        out
    }

    fn update_emas(&mut self, k: usize) {
        // Compute/refresh latest 1h bucket on whole-hour boundaries
        let current_ts = self.first_timestamp_ms + (k as u64) * self.interval_ms;
        let hour_boundary = (current_ts / 3_600_000u64) * 3_600_000u64;
        if hour_boundary > self.last_hour_boundary_ms {
            // window is from max(first_ts, last_boundary) to previous minute
            let window_start_ms = self.first_timestamp_ms.max(self.last_hour_boundary_ms);
            if current_ts > window_start_ms + self.interval_ms {
                let start_idx =
                    ((window_start_ms - self.first_timestamp_ms) / self.interval_ms) as usize;
                let end_idx = if k == 0 { 0usize } else { k - 1 };
                if end_idx >= start_idx {
                    for i in 0..self.n_coins {
                        if let Some((coin_start, coin_end)) = self.coin_valid_range(i) {
                            let start = start_idx.max(coin_start);
                            let end = end_idx.min(coin_end);
                            if start > end {
                                continue;
                            }
                            let mut h = f64::MIN;
                            let mut l = f64::MAX;
                            let mut seen = false;
                            for j in start..=end {
                                let high = self.hlcvs_value(j, i, HIGH);
                                let low = self.hlcvs_value(j, i, LOW);
                                if !(high.is_finite() && low.is_finite()) {
                                    continue;
                                }
                                if high > h {
                                    h = high;
                                }
                                if low < l {
                                    l = low;
                                }
                                seen = true;
                            }
                            if !seen {
                                continue;
                            }
                            self.latest_hour[i] = HourBucket { high: h, low: l };
                        }
                    }
                }
            }
            self.last_hour_boundary_ms = hour_boundary;

            // Update hourly log-range EMAs for entry volatility adjustments
            if self.needs_volatility_ema_1h_long || self.needs_volatility_ema_1h_short {
                for i in 0..self.n_coins {
                    if self.coin_valid_range(i).is_none() {
                        continue;
                    }
                    let bucket = &self.latest_hour[i];
                    if bucket.high <= 0.0
                        || bucket.low <= 0.0
                        || !bucket.high.is_finite()
                        || !bucket.low.is_finite()
                    {
                        continue;
                    }
                    let hour_log_range = (bucket.high / bucket.low).ln();
                    let alpha_long = self.ema_alphas[i].volatility_ema_1h_alpha_long;
                    let alpha_short = self.ema_alphas[i].volatility_ema_1h_alpha_short;
                    let emas = &mut self.emas[i];
                    if self.needs_volatility_ema_1h_long && alpha_long > 0.0 {
                        emas.volatility_ema_1h_long = update_adjusted_ema(
                            hour_log_range,
                            alpha_long,
                            &mut emas.volatility_ema_1h_long_num,
                            &mut emas.volatility_ema_1h_long_den,
                        );
                    }
                    if self.needs_volatility_ema_1h_short && alpha_short > 0.0 {
                        emas.volatility_ema_1h_short = update_adjusted_ema(
                            hour_log_range,
                            alpha_short,
                            &mut emas.volatility_ema_1h_short_num,
                            &mut emas.volatility_ema_1h_short_den,
                        );
                    }
                }
            }
        }
        for i in 0..self.n_coins {
            if !self.coin_is_valid_at(i, k) {
                continue;
            }
            let close_price = self.hlcvs_value(k, i, CLOSE);
            if !close_price.is_finite() {
                continue;
            }
            let vol_raw = self.hlcvs_value(k, i, VOLUME);
            let vol_base = if vol_raw.is_finite() {
                f64::max(0.0, vol_raw)
            } else {
                0.0
            };
            let high = self.hlcvs_value(k, i, HIGH);
            let low = self.hlcvs_value(k, i, LOW);
            if !high.is_finite() || !low.is_finite() {
                continue;
            }
            // Convert base volume to quote volume using typical price
            // This matches live bot's get_latest_ema_quote_volume() calculation
            let typical_price = (high + low + close_price) / 3.0;
            let vol = vol_base * typical_price;

            let long_alphas = &self.ema_alphas[i].long.alphas;
            let short_alphas = &self.ema_alphas[i].short.alphas;

            let emas = &mut self.emas[i];

            // price EMAs (3 levels)
            for z in 0..3 {
                emas.unstuck_long[z] = update_adjusted_ema(
                    close_price,
                    self.ema_alphas[i].unstuck_long.alphas[z],
                    &mut emas.unstuck_long_num[z],
                    &mut emas.unstuck_long_den[z],
                );
                emas.unstuck_short[z] = update_adjusted_ema(
                    close_price,
                    self.ema_alphas[i].unstuck_short.alphas[z],
                    &mut emas.unstuck_short_num[z],
                    &mut emas.unstuck_short_den[z],
                );
                emas.long[z] = update_adjusted_ema(
                    close_price,
                    long_alphas[z],
                    &mut emas.long_num[z],
                    &mut emas.long_den[z],
                );
                emas.short[z] = update_adjusted_ema(
                    close_price,
                    short_alphas[z],
                    &mut emas.short_num[z],
                    &mut emas.short_den[z],
                );
            }

            // volume EMAs (single value per pside)
            if self.needs_volume_ema_long || self.needs_volume_ema_short {
                if self.needs_volume_ema_long {
                    let vol_alpha_long = self.ema_alphas[i].vol_alpha_long;
                    emas.vol_long = update_adjusted_ema(
                        vol,
                        vol_alpha_long,
                        &mut emas.vol_long_num,
                        &mut emas.vol_long_den,
                    );
                }
                if self.needs_volume_ema_short {
                    let vol_alpha_short = self.ema_alphas[i].vol_alpha_short;
                    emas.vol_short = update_adjusted_ema(
                        vol,
                        vol_alpha_short,
                        &mut emas.vol_short_num,
                        &mut emas.vol_short_den,
                    );
                }
            }

            // log range metric: ln(high / low)
            if self.needs_log_range_long
                || self.needs_log_range_short
                || self.needs_volatility_ema_1m_long
                || self.needs_volatility_ema_1m_short
            {
                let log_range = if high > 0.0 && low > 0.0 {
                    (high / low).ln()
                } else {
                    0.0
                };
                if self.needs_log_range_long {
                    emas.log_range_long = update_adjusted_ema(
                        log_range,
                        self.ema_alphas[i].log_range_alpha_long,
                        &mut emas.log_range_long_num,
                        &mut emas.log_range_long_den,
                    );
                }
                if self.needs_log_range_short {
                    emas.log_range_short = update_adjusted_ema(
                        log_range,
                        self.ema_alphas[i].log_range_alpha_short,
                        &mut emas.log_range_short_num,
                        &mut emas.log_range_short_den,
                    );
                }
                if self.needs_volatility_ema_1m_long {
                    let alpha = self.ema_alphas[i].volatility_ema_1m_alpha_long;
                    if alpha > 0.0 {
                        emas.volatility_ema_1m_long = update_adjusted_ema(
                            log_range,
                            alpha,
                            &mut emas.volatility_ema_1m_long_num,
                            &mut emas.volatility_ema_1m_long_den,
                        );
                    }
                }
                if self.needs_volatility_ema_1m_short {
                    let alpha = self.ema_alphas[i].volatility_ema_1m_alpha_short;
                    if alpha > 0.0 {
                        emas.volatility_ema_1m_short = update_adjusted_ema(
                            log_range,
                            alpha,
                            &mut emas.volatility_ema_1m_short_num,
                            &mut emas.volatility_ema_1m_short_den,
                        );
                    }
                }
            }
        }
    }

    pub fn initial_entry_balance_pct(&self) -> (f64, f64) {
        let default_long_qty_pct = self.bot_params_master.long.entry_initial_qty_pct;
        let default_short_qty_pct = self.bot_params_master.short.entry_initial_qty_pct;
        let (long_qty_pct, short_qty_pct) = self
            .strategy_params
            .first()
            .map(|params| {
                (
                    strategy_initial_qty_pct(&params.long),
                    strategy_initial_qty_pct(&params.short),
                )
            })
            .unwrap_or((default_long_qty_pct, default_short_qty_pct));
        let long = calc_entry_balance_pct(
            &self.bot_params_master.long,
            self.effective_n_positions.long,
            long_qty_pct,
        );
        let short = calc_entry_balance_pct(
            &self.bot_params_master.short,
            self.effective_n_positions.short,
            short_qty_pct,
        );
        (long, short)
    }

    pub fn fill_activity_metrics_for_analysis(
        &self,
        fills: &[Fill],
        equities: &Equities,
    ) -> FillActivityMetrics {
        let long_slots = if self.bot_params_master.long.n_positions > 0
            && self.bot_params_master.long.total_wallet_exposure_limit > 0.0
        {
            self.bot_params_master.long.n_positions
        } else {
            0
        };
        let short_slots = if self.bot_params_master.short.n_positions > 0
            && self.bot_params_master.short.total_wallet_exposure_limit > 0.0
        {
            self.bot_params_master.short.n_positions
        } else {
            0
        };
        calc_fill_activity_metrics(fills, &equities.timestamps_ms, long_slots, short_slots)
    }

    fn strategy_equity_metrics_from_series(
        &self,
        strategy_equity_series: &[f64],
        drawdown_ema_samples: Option<&[f64]>,
        strategy_equity_timestamps_ms: Option<&[u64]>,
    ) -> StrategyEquityMetrics {
        let sample_count = strategy_equity_series
            .len()
            .min(self.equities.timestamps_ms.len());
        if sample_count < 2 {
            return StrategyEquityMetrics::default();
        }
        let timestamps_offset = self.equities.timestamps_ms.len() - sample_count;
        let series = &strategy_equity_series[strategy_equity_series.len() - sample_count..];
        let timestamps = &self.equities.timestamps_ms[timestamps_offset..];
        let daily_metric_timestamps = strategy_equity_timestamps_ms
            .map(|values| {
                assert_eq!(
                    values.len(),
                    strategy_equity_series.len(),
                    "per-side strategy-equity timestamps must align with samples"
                );
                &values[values.len() - sample_count..]
            })
            .unwrap_or(timestamps);
        let drawdowns = calc_strategy_equity_drawdowns(series);
        let drawdown_emas = drawdown_ema_samples.map(|samples| {
            let ema_sample_count = sample_count.min(samples.len());
            &samples[samples.len() - ema_sample_count..]
        });

        let compute_metrics = |series: &[f64],
                               timestamps_ms: &[u64],
                               daily_metric_timestamps_ms: &[u64],
                               drawdowns: &[f64],
                               drawdown_emas: Option<&[f64]>| {
            let equity_metrics = analyze_equity_series(series, timestamps_ms);
            let drawdown_worst = drawdowns
                .iter()
                .fold(0.0_f64, |max_dd, &x| max_dd.max(x.abs()));
            let daily_worst_drawdowns =
                daily_worst_positive_drawdowns(drawdowns, daily_metric_timestamps_ms, series.len());
            let drawdown_worst_mean_1pct = mean_worst_1pct_abs(&daily_worst_drawdowns);
            let strategy_eq_underwater_pct_mean = mean_abs(&daily_worst_drawdowns);
            let strategy_eq_underwater_pct_median = median_abs(&daily_worst_drawdowns);
            let drawdown_worst_ema_strategy_eq = drawdown_emas
                .map(|values| {
                    values
                        .iter()
                        .fold(0.0_f64, |max_dd, &x| max_dd.max(x.abs()))
                })
                .unwrap_or(0.0);
            let drawdown_worst_mean_1pct_ema_strategy_eq =
                drawdown_emas.map(mean_worst_1pct_abs).unwrap_or(0.0);
            StrategyEquityMetrics {
                gain_strategy_eq: equity_metrics.gain,
                adg_strategy_eq: equity_metrics.adg,
                adg_rolling_hmean_strategy_eq: equity_metrics.adg_rolling_hmean,
                adg_time_integrated_strategy_eq: equity_metrics.adg_time_integrated,
                positive_gain_participation_strategy_eq: equity_metrics.positive_gain_participation,
                mdg_strategy_eq: equity_metrics.mdg,
                sharpe_ratio_strategy_eq: equity_metrics.sharpe_ratio,
                sortino_ratio_strategy_eq: equity_metrics.sortino_ratio,
                omega_ratio_strategy_eq: equity_metrics.omega_ratio,
                expected_shortfall_1pct_strategy_eq: equity_metrics.expected_shortfall_1pct,
                calmar_ratio_strategy_eq: equity_metrics.adg / drawdown_worst.max(1e-12),
                sterling_ratio_strategy_eq: equity_metrics.adg
                    / drawdown_worst_mean_1pct.max(1e-12),
                drawdown_worst_strategy_eq: drawdown_worst,
                drawdown_worst_ema_strategy_eq,
                drawdown_worst_mean_1pct_strategy_eq: drawdown_worst_mean_1pct,
                drawdown_worst_mean_1pct_ema_strategy_eq,
                strategy_eq_underwater_pct_mean,
                strategy_eq_underwater_pct_median,
                ..StrategyEquityMetrics::default()
            }
        };

        let full = compute_metrics(
            series,
            timestamps,
            daily_metric_timestamps,
            &drawdowns,
            drawdown_emas,
        );
        let recovery = calc_strategy_eq_recovery_days(series, timestamps);
        let peak_recovery_days_strategy_eq = recovery.max;
        let peak_recovery_hours_strategy_eq = peak_recovery_days_strategy_eq * 24.0;
        let n = sample_count;
        let mut subset_metrics = Vec::with_capacity(10);
        subset_metrics.push(StrategyEquityMetrics {
            strategy_eq_recovery_days_mean: recovery.mean,
            strategy_eq_recovery_days_median: recovery.median,
            strategy_eq_recovery_days_p95: recovery.p95,
            strategy_eq_recovery_days_p99: recovery.p99,
            strategy_eq_recovery_days_mean_worst_5pct: recovery.mean_worst_5pct,
            strategy_eq_recovery_days_mean_worst_1pct: recovery.mean_worst_1pct,
            strategy_eq_recovery_days_max: recovery.max,
            peak_recovery_hours_strategy_eq,
            peak_recovery_days_strategy_eq,
            ..full
        });
        for i in 1..10 {
            let fraction = 1.0 / (1.0 + i as f64);
            let start_idx = (n as f64 - fraction * (n as f64)).round() as usize;
            let subset_series = &series[start_idx..];
            if subset_series.is_empty() {
                break;
            }
            let subset_timestamps = &timestamps[start_idx..];
            let subset_daily_metric_timestamps = &daily_metric_timestamps[start_idx..];
            let subset_drawdowns = &drawdowns[start_idx..];
            let mut subset_metric = compute_metrics(
                subset_series,
                subset_timestamps,
                subset_daily_metric_timestamps,
                subset_drawdowns,
                // Only the full-window EMA metrics are returned; weighted metrics
                // below consume no EMA fields. Avoid sorting unused suffix samples.
                None,
            );
            subset_metric.peak_recovery_days_strategy_eq =
                calc_strategy_eq_recovery_days(subset_series, subset_timestamps).max;
            subset_metric.peak_recovery_hours_strategy_eq =
                subset_metric.peak_recovery_days_strategy_eq * 24.0;
            subset_metrics.push(subset_metric);
        }

        StrategyEquityMetrics {
            strategy_eq_recovery_days_mean: subset_metrics[0].strategy_eq_recovery_days_mean,
            strategy_eq_recovery_days_median: subset_metrics[0].strategy_eq_recovery_days_median,
            strategy_eq_recovery_days_p95: subset_metrics[0].strategy_eq_recovery_days_p95,
            strategy_eq_recovery_days_p99: subset_metrics[0].strategy_eq_recovery_days_p99,
            strategy_eq_recovery_days_mean_worst_5pct: subset_metrics[0]
                .strategy_eq_recovery_days_mean_worst_5pct,
            strategy_eq_recovery_days_mean_worst_1pct: subset_metrics[0]
                .strategy_eq_recovery_days_mean_worst_1pct,
            strategy_eq_recovery_days_max: subset_metrics[0].strategy_eq_recovery_days_max,
            peak_recovery_hours_strategy_eq: subset_metrics[0].peak_recovery_hours_strategy_eq,
            peak_recovery_days_strategy_eq: subset_metrics[0].peak_recovery_days_strategy_eq,
            adg_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.adg_strategy_eq)
                .sum::<f64>()
                / 10.0,
            mdg_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.mdg_strategy_eq)
                .sum::<f64>()
                / 10.0,
            sharpe_ratio_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.sharpe_ratio_strategy_eq)
                .sum::<f64>()
                / 10.0,
            sortino_ratio_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.sortino_ratio_strategy_eq)
                .sum::<f64>()
                / 10.0,
            omega_ratio_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.omega_ratio_strategy_eq)
                .sum::<f64>()
                / 10.0,
            calmar_ratio_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.calmar_ratio_strategy_eq)
                .sum::<f64>()
                / 10.0,
            sterling_ratio_strategy_eq_w: subset_metrics
                .iter()
                .map(|m| m.sterling_ratio_strategy_eq)
                .sum::<f64>()
                / 10.0,
            ..full
        }
    }

    pub fn strategy_equity_metrics_for_analysis(&self) -> StrategyEquityMetricsBundle {
        if self.equities.timestamps_ms.is_empty() {
            if let Some(metrics) = self.final_strategy_equity_metrics {
                return metrics;
            }
        }
        self.hsl_strategy_metrics()
    }

    pub fn strategy_equity_series_for_artifacts(&self) -> &[f64] {
        &self.strategy_equity_series
    }

    pub fn hard_stop_metrics(&self) -> HardStopMetrics {
        if self.equities.timestamps_ms.is_empty() {
            if let Some(metrics) = self.final_hard_stop_metrics {
                return metrics;
            }
        }
        let minutes = match (
            self.equities.timestamps_ms.first(),
            self.equities.timestamps_ms.last(),
        ) {
            (Some(first), Some(last)) => (last.saturating_sub(*first) as f64 / 60_000.0).max(1.0),
            _ => 0.0,
        };
        self.hsl_report
            .metrics(self.backtest_params.starting_balance, minutes)
    }
}

fn mean_worst_1pct_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    let by_magnitude = |a: &f64, b: &f64| {
        a.abs().partial_cmp(&b.abs()).unwrap_or_else(|| {
            if a.is_nan() && b.is_nan() {
                Ordering::Equal
            } else if a.is_nan() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
    };
    let cutoff_index = std::cmp::max(1, (sorted.len() as f64 * 0.01) as usize);
    let worst_n = std::cmp::min(cutoff_index, sorted.len());
    let split = sorted.len() - worst_n;
    if sorted.iter().all(|x| x.is_finite()) {
        // Select in linear time, then sort only the tail to preserve the exact
        // ascending summation order of the full-sort reference (including ties).
        sorted.select_nth_unstable_by(split, by_magnitude);
        sorted[split..].sort_by(by_magnitude);
    } else {
        // Preserve the existing ordering/NaN contract for exceptional inputs.
        sorted.sort_by(by_magnitude);
    }
    sorted[split..].iter().map(|x| x.abs()).sum::<f64>() / worst_n as f64
}

fn mean_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|x| x.abs()).sum::<f64>() / values.len() as f64
}

fn median_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.iter().map(|x| x.abs()).collect::<Vec<_>>();
    sorted.sort_by(|a, b| {
        a.partial_cmp(b).unwrap_or_else(|| {
            if a.is_nan() && b.is_nan() {
                Ordering::Equal
            } else if a.is_nan() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
    });
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct StrategyEqRecoveryDays {
    mean: f64,
    median: f64,
    p95: f64,
    p99: f64,
    mean_worst_5pct: f64,
    mean_worst_1pct: f64,
    max: f64,
}

fn calc_strategy_eq_recovery_days(series: &[f64], timestamps_ms: &[u64]) -> StrategyEqRecoveryDays {
    if series.is_empty() || timestamps_ms.is_empty() {
        return StrategyEqRecoveryDays::default();
    }
    let n = series.len().min(timestamps_ms.len());
    if n == 0 {
        return StrategyEqRecoveryDays::default();
    }

    let final_ts = timestamps_ms[n - 1];
    let mut durations_ms = vec![0_u64; n];
    let mut pending: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let value = series[i];
        while let Some(&idx) = pending.last() {
            if value > series[idx] {
                durations_ms[idx] = timestamps_ms[i].saturating_sub(timestamps_ms[idx]);
                pending.pop();
            } else {
                break;
            }
        }
        pending.push(i);
    }
    for idx in pending {
        durations_ms[idx] = final_ts.saturating_sub(timestamps_ms[idx]);
    }
    summarize_recovery_durations_days(&mut durations_ms)
}

fn summarize_recovery_durations_days(durations_ms: &mut [u64]) -> StrategyEqRecoveryDays {
    if durations_ms.is_empty() {
        return StrategyEqRecoveryDays::default();
    }
    durations_ms.sort_unstable();
    let denom = 86_400_000.0;
    let mean =
        durations_ms.iter().map(|x| *x as f64).sum::<f64>() / (durations_ms.len() as f64 * denom);
    let median = percentile_sorted_u64(durations_ms, 50.0) / denom;
    StrategyEqRecoveryDays {
        mean,
        median,
        p95: percentile_sorted_u64(durations_ms, 95.0) / denom,
        p99: percentile_sorted_u64(durations_ms, 99.0) / denom,
        mean_worst_5pct: mean_worst_pct_sorted_u64(durations_ms, 5.0) / denom,
        mean_worst_1pct: mean_worst_pct_sorted_u64(durations_ms, 1.0) / denom,
        max: durations_ms.last().copied().unwrap_or(0) as f64 / denom,
    }
}

fn percentile_sorted_u64(sorted: &[u64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let pct = percentile.clamp(0.0, 100.0);
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower] as f64
    } else {
        let weight = rank - lower as f64;
        sorted[lower] as f64 + (sorted[upper] as f64 - sorted[lower] as f64) * weight
    }
}

fn mean_worst_pct_sorted_u64(sorted: &[u64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let clamped = pct.clamp(0.0, 100.0);
    let cutoff_index = ((sorted.len() as f64) * clamped / 100.0).max(1.0) as usize;
    let worst_n = cutoff_index.min(sorted.len());
    sorted[sorted.len() - worst_n..]
        .iter()
        .map(|x| *x as f64)
        .sum::<f64>()
        / worst_n as f64
}

fn calc_strategy_equity_drawdowns(values: &[f64]) -> Vec<f64> {
    let mut drawdowns = Vec::with_capacity(values.len());
    let mut running = f64::NEG_INFINITY;
    for &v in values {
        running = running.max(v);
        let denom = running.abs().max(1e-12);
        drawdowns.push((running - v) / denom);
    }
    drawdowns
}

fn daily_worst_positive_drawdowns(
    drawdowns: &[f64],
    timestamps_ms: &[u64],
    expected_len: usize,
) -> Vec<f64> {
    if drawdowns.is_empty() {
        return Vec::new();
    }

    let use_timestamps = !timestamps_ms.is_empty() && timestamps_ms.len() == expected_len;
    let mut daily_worst = Vec::new();
    let mut current_day = if use_timestamps {
        (timestamps_ms[0] / 86_400_000) as usize
    } else {
        0
    };
    let mut current_worst = drawdowns[0];
    for (i, &drawdown) in drawdowns.iter().enumerate() {
        let day = if use_timestamps {
            (timestamps_ms[i] / 86_400_000) as usize
        } else {
            i / 1440
        };
        if day > current_day {
            daily_worst.push(current_worst);
            current_day = day;
            current_worst = drawdown;
        } else {
            current_worst = current_worst.max(drawdown);
        }
    }
    daily_worst.push(current_worst);
    daily_worst
}

fn unstuck_spans(bot: &BotParams) -> [f64; 3] {
    let mut spans = [
        bot.unstuck_ema_span_0,
        bot.unstuck_ema_span_1,
        (bot.unstuck_ema_span_0 * bot.unstuck_ema_span_1).sqrt(),
    ];
    spans.sort_by(f64::total_cmp);
    spans
}

fn calc_ema_alphas(
    bot_params_pair: &BotParamsPair,
    strategy_params_pair: &StrategyParamsPair,
    interval: u64,
) -> EmaAlphas {
    let interval_f = interval as f64;
    let clamp_alpha = |alpha: f64| {
        if !alpha.is_finite() {
            0.0
        } else if alpha < 0.0 {
            0.0
        } else if alpha > 1.0 {
            1.0
        } else {
            alpha
        }
    };

    // EMA spans are in minutes. Divide by interval to get number of candle periods.
    let (long_span_0, long_span_1) = strategy_ema_spans(&strategy_params_pair.long);
    let mut ema_spans_long = [long_span_0, long_span_1, (long_span_0 * long_span_1).sqrt()];
    ema_spans_long.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (short_span_0, short_span_1) = strategy_ema_spans(&strategy_params_pair.short);
    let mut ema_spans_short = [
        short_span_0,
        short_span_1,
        (short_span_0 * short_span_1).sqrt(),
    ];
    ema_spans_short.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Price EMAs - spans are in minutes, convert to candle periods
    let ema_alphas_long = ema_spans_long.map(|x| clamp_alpha(2.0 / (x / interval_f + 1.0)));
    let ema_alphas_short = ema_spans_short.map(|x| clamp_alpha(2.0 / (x / interval_f + 1.0)));

    EmaAlphas {
        long: Alphas {
            alphas: ema_alphas_long,
        },
        short: Alphas {
            alphas: ema_alphas_short,
        },
        unstuck_long: Alphas {
            alphas: unstuck_spans(&bot_params_pair.long)
                .map(|x| clamp_alpha(2.0 / (x / interval_f + 1.0))),
        },
        unstuck_short: Alphas {
            alphas: unstuck_spans(&bot_params_pair.short)
                .map(|x| clamp_alpha(2.0 / (x / interval_f + 1.0))),
        },
        // EMA spans for the volume/log range filters (alphas precomputed from spans)
        vol_alpha_long: clamp_alpha(
            2.0 / (bot_params_pair.long.filter_volume_ema_span_1m as f64 / interval_f + 1.0),
        ),
        vol_alpha_short: clamp_alpha(
            2.0 / (bot_params_pair.short.filter_volume_ema_span_1m as f64 / interval_f + 1.0),
        ),
        log_range_alpha_long: clamp_alpha(
            2.0 / (bot_params_pair.long.filter_volatility_ema_span_1m as f64 / interval_f + 1.0),
        ),
        log_range_alpha_short: clamp_alpha(
            2.0 / (bot_params_pair.short.filter_volatility_ema_span_1m as f64 / interval_f + 1.0),
        ),
        volatility_ema_1m_alpha_long: {
            let span =
                strategy_offset_volatility_span_minutes(&strategy_params_pair.long).unwrap_or(0.0);
            if span > 0.0 {
                clamp_alpha(2.0 / (span / interval_f + 1.0))
            } else {
                0.0
            }
        },
        volatility_ema_1m_alpha_short: {
            let span =
                strategy_offset_volatility_span_minutes(&strategy_params_pair.short).unwrap_or(0.0);
            if span > 0.0 {
                clamp_alpha(2.0 / (span / interval_f + 1.0))
            } else {
                0.0
            }
        },
        // Note: entry_volatility spans are in HOURS and computed from hourly buckets,
        // so they do NOT need interval adjustment (hourly buckets are calendar-based)
        volatility_ema_1h_alpha_long: {
            let span =
                strategy_entry_volatility_span_hours(&strategy_params_pair.long).unwrap_or(0.0);
            if span > 0.0 {
                2.0 / (span.max(1.0) + 1.0)
            } else {
                0.0
            }
        },
        volatility_ema_1h_alpha_short: {
            let span =
                strategy_entry_volatility_span_hours(&strategy_params_pair.short).unwrap_or(0.0);
            if span > 0.0 {
                2.0 / (span.max(1.0) + 1.0)
            } else {
                0.0
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{
        TrailingGridV7CloseParams, TrailingGridV7EntryParams, TrailingGridV7Params,
        TrailingMartingaleCloseParams, TrailingMartingaleEntryParams, TrailingMartingaleParams,
    };
    use crate::types::EquityHardStopLossConfig;
    use ndarray::{Array1, Array3};
    use std::fs;

    fn tm_params_for_ema_tests(bot_params: &BotParams) -> TrailingMartingaleParams {
        TrailingMartingaleParams {
            volatility_ema_span_1h: bot_params.entry_volatility_ema_span_1h,
            volatility_ema_span_1m: bot_params.entry_volatility_ema_span_1m,
            entry: TrailingMartingaleEntryParams {
                ema_span_0: bot_params.ema_span_0,
                ema_span_1: bot_params.ema_span_1,
                double_down_factor: bot_params.entry_grid_double_down_factor,
                ema_gate_mode: crate::strategies::EmaGateMode::Initial,
                initial_ema_dist: bot_params.entry_initial_ema_dist,
                initial_qty_pct: bot_params.entry_initial_qty_pct,
                threshold_base_pct: bot_params.entry_grid_spacing_pct,
                threshold_we_weight: bot_params.entry_we_weight,
                threshold_volatility_1h_weight: bot_params.entry_weight_volatility_1h,
                threshold_volatility_1m_weight: bot_params.entry_weight_volatility_1m,
                retracement_base_pct: bot_params.entry_trailing_retracement_pct,
                retracement_we_weight: bot_params.entry_we_weight,
                retracement_volatility_1h_weight: bot_params.entry_weight_volatility_1h,
                retracement_volatility_1m_weight: bot_params.entry_weight_volatility_1m,
            },
            close: TrailingMartingaleCloseParams {
                qty_pct: bot_params.close_grid_qty_pct,
                threshold_base_pct: bot_params.close_trailing_threshold_pct,
                threshold_volatility_1h_weight: bot_params.close_weight_volatility_1h,
                threshold_volatility_1m_weight: bot_params.close_weight_volatility_1m,
                retracement_base_pct: bot_params.close_trailing_retracement_pct,
                retracement_volatility_1h_weight: bot_params.close_weight_volatility_1h,
                retracement_volatility_1m_weight: bot_params.close_weight_volatility_1m,
                ..Default::default()
            },
        }
    }

    fn strategy_pair_for_ema_tests(bot_params: &BotParamsPair) -> StrategyParamsPair {
        StrategyParamsPair {
            long: StrategyParams::TrailingMartingale(tm_params_for_ema_tests(&bot_params.long)),
            short: StrategyParams::TrailingMartingale(tm_params_for_ema_tests(&bot_params.short)),
        }
    }

    #[test]
    fn effective_min_cost_uses_executable_min_qty() {
        let exchange = ExchangeParams {
            qty_step: 1.0,
            min_qty: 0.0,
            min_cost: 0.1,
            c_mult: 1.0,
            ..Default::default()
        };
        let price = 88.165;
        let effective_min_cost = calc_effective_min_cost(price, &exchange);
        assert!((effective_min_cost - price).abs() < 1e-12);
    }

    #[test]
    fn btc_collateral_initializes_at_trade_start_price() {
        let hlcvs = Array3::from_shape_vec((3, 1, 4), vec![1.0; 3 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![100.0, 200.0, 300.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.ema_span_0 = 1.0;
        bp_pair.long.ema_span_1 = 1.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 60_000,
            first_valid_indices: vec![0],
            last_valid_indices: vec![2],
            warmup_minutes: vec![1],
            trade_start_indices: vec![1],
            global_warmup_bars: 1,
            btc_collateral_cap: 0.9,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            forager_score_hysteresis_pct: 0.0,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        assert!(bt.balance.use_btc_collateral);
        assert!(!bt.btc_collateral_initialized);
        assert_eq!(bt.balance.usd_cash_wallet, 1000.0);
        assert_eq!(bt.balance.btc_cash_wallet, 0.0);
        assert_eq!(bt.balance.usd_total_balance, 1000.0);

        bt.initialize_btc_collateral_if_needed(1);
        assert!(bt.btc_collateral_initialized);
        assert!((bt.balance.btc_cash_wallet - 4.5).abs() < 1e-12);
        assert!((bt.balance.usd_cash_wallet - 100.0).abs() < 1e-12);
        assert!((bt.balance.usd_total_balance - 1000.0).abs() < 1e-12);

        bt.initialize_btc_collateral_if_needed(2);
        assert!((bt.balance.btc_cash_wallet - 4.5).abs() < 1e-12);
        assert!((bt.balance.usd_cash_wallet - 100.0).abs() < 1e-12);
    }

    #[test]
    fn forced_active_coin_sets_orchestrator_normal_mode() {
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;
        bp_pair.long.is_forced_active = true;
        bp_pair.short.n_positions = 1;
        bp_pair.short.total_wallet_exposure_limit = 1.0;
        bp_pair.short.wallet_exposure_limit = 1.0;
        bp_pair.short.ema_span_0 = 10.0;
        bp_pair.short.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            forager_score_hysteresis_pct: 0.0,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        let input = bt.get_orchestrator_input_cached(1, None, None);
        assert_eq!(
            input.symbols[0].long.mode,
            Some(orchestrator::TradingMode::Normal)
        );
        assert_eq!(input.symbols[0].short.mode, None);
    }

    #[test]
    fn missing_candles_reject_held_valuation_and_preserve_unheld_boundaries() {
        let mut values = vec![1.0; 3 * 4];
        values[4] = f64::NAN;
        values[5] = f64::NAN;
        values[6] = f64::NAN;
        values[7] = f64::NAN;
        let hlcvs = Array3::from_shape_vec((3, 1, 4), values).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 3]);
        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;
        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![2],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            forager_score_hysteresis_pct: 0.0,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            candle_interval_minutes: 1,
        };
        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        assert!(bt.coin_is_valid_at(0, 0));
        assert!(!bt.coin_is_valid_at(0, 1));
        assert!(!bt.coin_is_tradeable_at(0, 1));
        assert!(bt.coin_is_valid_at(0, 2));

        bt.positions.long[0] = Position {
            size: 100.0,
            price: 2.0,
        };
        assert_eq!(bt.current_usd_equity_at(0), 900.0);
        assert_eq!(bt.current_usd_equity_at(2), 900.0);
        let error = bt.run().err().expect("missing candle must reject backtest");
        assert!(error.contains("contiguous finite H/L/C"), "{error}");
        assert!(error.contains("candle 1"), "{error}");
        assert!(bt.equities.timestamps_ms.is_empty());
        assert!(bt.validate_held_position_valuation(1).is_err());

        let input = bt.build_orchestrator_input_iter(1, None, None, 0..1);
        assert!(!input.symbols[0].tradable);
        assert_ne!(
            input.symbols[0].long.mode,
            Some(orchestrator::TradingMode::Panic),
            "an internal data gap must not be mistaken for a delist"
        );
        // Missing rows outside the declared listing window are permitted while flat.
        bt.positions.long[0] = Position::default();
        bt.coin_first_valid_idx[0] = 2;
        bt.coin_last_valid_idx[0] = 2;
        assert!(bt.validate_candle_coverage().is_ok());
        assert!(bt.validate_held_position_valuation(1).is_ok());
        bt.coin_first_valid_idx[0] = 0;
        bt.coin_last_valid_idx[0] = 0;
        assert!(bt.validate_candle_coverage().is_ok());
        assert!(bt.validate_held_position_valuation(1).is_ok());
        bt.positions.short[0] = Position {
            size: -100.0,
            price: 2.0,
        };
        assert!(bt.validate_held_position_valuation(1).is_err());
    }

    #[test]
    fn panic_market_close_fills_next_bar_as_taker() {
        let hlcvs = Array3::from_shape_vec(
            (2, 1, 4),
            vec![
                101.0, 99.0, 100.0, 1.0, //
                101.0, 99.0, 100.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let hs = EquityHardStopLossConfig::default();

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0002,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: hs,
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.5,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );
        bt.positions.long[0] = Position {
            size: 1.0,
            price: 100.0,
        };
        bt.open_orders.long[0].closes.push(BacktestOrder {
            order: Order {
                qty: -1.0,
                price: 200.0,
                order_type: OrderType::ClosePanicLong,
            },
            execution_type: orchestrator::ExecutionType::Market,
        });

        bt.check_for_fills(1).unwrap();

        assert_eq!(bt.positions.long[0].size, 0.0);
        assert_eq!(bt.fills.len(), 1);
        assert_eq!(bt.fills[0].order_type, OrderType::ClosePanicLong);
        assert!((bt.fills[0].fill_price - 99.95).abs() < 1e-12);
        assert!(bt.fills[0].fee_paid < 0.0);
    }

    #[test]
    fn panic_limit_close_still_requires_cross() {
        let hlcvs = Array3::from_shape_vec(
            (2, 1, 4),
            vec![
                101.0, 99.0, 100.0, 1.0, //
                101.0, 99.0, 100.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let hs = EquityHardStopLossConfig::default();

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0002,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: hs,
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );
        bt.positions.long[0] = Position {
            size: 1.0,
            price: 100.0,
        };
        bt.open_orders.long[0].closes.push(BacktestOrder {
            order: Order {
                qty: -1.0,
                price: 200.0,
                order_type: OrderType::ClosePanicLong,
            },
            execution_type: orchestrator::ExecutionType::Limit,
        });

        bt.check_for_fills(1).unwrap();

        assert_ne!(bt.positions.long[0].size, 0.0);
        assert!(bt.fills.is_empty());
    }

    #[test]
    fn delisted_open_positions_are_realized_on_last_valid_candle() {
        let n_timesteps = 1_505;
        let mut hlcvs = Array3::<f64>::zeros((n_timesteps, 1, 4));
        for k in 0..n_timesteps {
            hlcvs[[k, 0, HIGH]] = 90.0;
            hlcvs[[k, 0, LOW]] = 90.0;
            hlcvs[[k, 0, CLOSE]] = 90.0;
            hlcvs[[k, 0, VOLUME]] = 1.0;
        }
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; n_timesteps]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;
        bp_pair.short.n_positions = 1;
        bp_pair.short.total_wallet_exposure_limit = 1.0;
        bp_pair.short.wallet_exposure_limit = 1.0;
        bp_pair.short.ema_span_0 = 10.0;
        bp_pair.short.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.001,
            coins: vec!["DELIST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![30],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );
        bt.positions.long[0] = Position {
            size: 1.0,
            price: 100.0,
        };
        bt.positions.short[0] = Position {
            size: -1.0,
            price: 100.0,
        };

        let (fills, _) = bt.run().unwrap();

        assert_eq!(bt.positions.long[0].size, 0.0);
        assert_eq!(bt.positions.short[0].size, 0.0);
        let close_long = fills
            .iter()
            .find(|fill| fill.order_type == OrderType::ClosePanicLong)
            .expect("expected delisting panic close fill");
        assert_eq!(close_long.index, 30);
        assert_eq!(close_long.fill_qty, -1.0);
        assert_eq!(close_long.fill_price, 90.0);
        assert!(close_long.pnl < 0.0);
        assert!(close_long.fee_paid < 0.0);

        let close_short = fills
            .iter()
            .find(|fill| fill.order_type == OrderType::ClosePanicShort)
            .expect("expected delisting panic short close fill");
        assert_eq!(close_short.index, 30);
        assert_eq!(close_short.fill_qty, 1.0);
        assert_eq!(close_short.fill_price, 90.0);
        assert!(close_short.pnl > 0.0);
        assert!(close_short.fee_paid < 0.0);
    }

    #[test]
    fn non_panic_market_entry_uses_slippage_exchange_taker_fee_and_liquidity_tag() {
        let hlcvs = Array3::from_shape_vec(
            (2, 1, 4),
            vec![
                101.0, 99.0, 100.0, 1.0, //
                101.0, 99.0, 100.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0002,
            taker_fee: 0.00099,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: true,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams {
                taker_fee: 0.00055,
                ..Default::default()
            }],
            &backtest_params,
        );
        bt.open_orders.long[0].entries.push(BacktestOrder {
            order: Order {
                qty: 1.0,
                price: 100.0,
                order_type: OrderType::EntryGridNormalLong,
            },
            execution_type: orchestrator::ExecutionType::Market,
        });

        bt.check_for_fills(1).unwrap();

        assert_ne!(bt.positions.long[0].size, 0.0);
        assert_eq!(bt.fills.len(), 1);
        assert_eq!(bt.fills[0].order_type, OrderType::EntryGridNormalLong);
        assert_eq!(bt.fills[0].liquidity, "taker");
        assert!((bt.fills[0].fill_price - 100.05).abs() < 1e-12);
        assert!((bt.fills[0].fee_paid + 100.05 * 0.00055).abs() < 1e-12);
    }

    #[test]
    fn limit_entry_uses_exchange_maker_fee_and_liquidity_tag() {
        let hlcvs = Array3::from_shape_vec(
            (2, 1, 4),
            vec![
                101.0, 99.0, 100.0, 1.0, //
                101.0, 99.0, 100.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.00099,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: true,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams {
                maker_fee: 0.0002,
                ..Default::default()
            }],
            &backtest_params,
        );
        bt.open_orders.long[0].entries.push(BacktestOrder {
            order: Order {
                qty: 1.0,
                price: 100.0,
                order_type: OrderType::EntryGridNormalLong,
            },
            execution_type: orchestrator::ExecutionType::Limit,
        });

        bt.check_for_fills(1).unwrap();

        assert_ne!(bt.positions.long[0].size, 0.0);
        assert_eq!(bt.fills.len(), 1);
        assert_eq!(bt.fills[0].order_type, OrderType::EntryGridNormalLong);
        assert_eq!(bt.fills[0].liquidity, "maker");
        assert_eq!(bt.fills[0].fill_price, 100.0);
        assert!((bt.fills[0].fee_paid + 100.0 * 0.0002).abs() < 1e-12);
    }

    #[test]
    fn limit_fill_buffer_covers_entries_closes_market_bypass_and_peek_cache() {
        let hlcvs = Array3::from_shape_vec(
            (3, 1, 4),
            vec![
                100.005,
                99.995,
                100.0,
                1.0,
                100.0 * (1.0 + 0.0001),
                100.0 * (1.0 - 0.0001),
                100.0,
                1.0,
                100.02,
                99.98,
                100.0,
                1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 3]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        bp_pair.long.hsl_panic_close_order_type = "limit".to_string();
        bp_pair.short.hsl_panic_close_order_type = "limit".to_string();
        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.00099,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![2],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: true,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0001,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams {
                maker_fee: 0.0002,
                ..Default::default()
            }],
            &backtest_params,
        );
        for (qty, order_type) in [
            (1.0, OrderType::EntryGridNormalLong),
            (-1.0, OrderType::EntryGridNormalShort),
            (-1.0, OrderType::CloseGridLong),
            (1.0, OrderType::CloseGridShort),
            (-1.0, OrderType::ClosePanicLong),
            (1.0, OrderType::ClosePanicShort),
        ] {
            let order = BacktestOrder {
                order: Order {
                    qty,
                    price: 100.0,
                    order_type,
                },
                execution_type: orchestrator::ExecutionType::Limit,
            };
            assert!(bt.order_fill_execution(0, 0, &order).is_none());
            assert!(bt.order_fill_execution(1, 0, &order).is_none());
            let fill = bt.order_fill_execution(2, 0, &order).unwrap();
            assert_eq!(fill.price, 100.0);
            assert_eq!(fill.fee_rate, 0.0002);
            assert_eq!(fill.liquidity, "maker");
            if !matches!(
                order_type,
                OrderType::ClosePanicLong | OrderType::ClosePanicShort
            ) {
                let market = BacktestOrder {
                    execution_type: orchestrator::ExecutionType::Market,
                    ..order
                };
                let fill = bt.order_fill_execution(0, 0, &market).unwrap();
                assert_eq!(fill.liquidity, "taker");
            }
        }
        // The initial and cached next-candle hints must carry the same buffer.
        let input = bt.get_orchestrator_input_cached(0, None, None);
        assert_eq!(
            input.symbols[0]
                .next_candle
                .as_ref()
                .unwrap()
                .limit_order_fill_buffer_pct,
            0.0001
        );
        bt.orchestrator_input_cache = Some(input);
        let input = bt.get_orchestrator_input_cached(1, None, None);
        assert_eq!(
            input.symbols[0]
                .next_candle
                .as_ref()
                .unwrap()
                .limit_order_fill_buffer_pct,
            0.0001
        );
    }

    #[test]
    fn trailing_grid_v7_zero_cooldown_fills_multiple_grid_entries_in_one_candle() {
        let hlcvs = Array3::from_shape_vec(
            (2, 1, 4),
            vec![
                101.0, 99.0, 100.0, 1.0, //
                101.0, 80.0, 90.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.wallet_exposure_limit = 1.0;
        bp_pair.long.risk_entry_cooldown_minutes = 0.0;
        bp_pair.short.n_positions = 0;
        bp_pair.short.total_wallet_exposure_limit = 0.0;
        bp_pair.short.wallet_exposure_limit = 0.0;

        let strategy = TrailingGridV7Params {
            ema_span_0: 1.0,
            ema_span_1: 1.0,
            entry: TrailingGridV7EntryParams {
                grid_double_down_factor: 1.0,
                grid_spacing_pct: 0.02,
                initial_qty_pct: 0.1,
                trailing_double_down_factor: 1.0,
                trailing_grid_ratio: -0.072114,
                trailing_retracement_pct: 0.037427,
                trailing_threshold_pct: 0.01,
                volatility_ema_span_hours: 1.0,
                ..Default::default()
            },
            close: TrailingGridV7CloseParams::default(),
        };
        let strategy_params = StrategyParamsPairValue {
            long: serde_json::to_value(strategy).unwrap(),
            short: serde_json::to_value(strategy).unwrap(),
        };
        let backtest_params = BacktestParams {
            starting_balance: 1_000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };
        let exchange = ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
        };

        let mut bt = Backtest::new_with_strategy_params(
            hlcvs.view(),
            btc_usd_prices.view(),
            crate::strategies::StrategyKind::TrailingGridV7,
            vec![bp_pair],
            vec![strategy_params],
            vec![exchange],
            &backtest_params,
        );

        bt.bot_params[0].long.risk_entry_cooldown_minutes = 0.05;
        bt.orchestrator_input_cache = None;
        bt.update_open_orders_all(0).unwrap();
        assert_eq!(
            bt.open_orders.long[0].entries.len(),
            1,
            "positive cooldown must still stage only one v7 add order"
        );

        bt.bot_params[0].long.risk_entry_cooldown_minutes = 0.0;
        bt.orchestrator_input_cache = None;
        bt.update_open_orders_all(0).unwrap();
        let staged_entry_count = bt.open_orders.long[0].entries.len();
        assert!(
            staged_entry_count >= 2,
            "expected the v7 grid leg to stage multiple entries, got {staged_entry_count}"
        );

        bt.check_for_fills(1).unwrap();

        let same_candle_entries: Vec<&Fill> = bt
            .fills
            .iter()
            .filter(|fill| fill.index == 1 && fill.fill_qty > 0.0)
            .collect();
        assert_eq!(same_candle_entries.len(), staged_entry_count);
        assert!(same_candle_entries.len() >= 2);
        assert!(same_candle_entries
            .iter()
            .all(|fill| fill.timestamp_ms == 60_000));
    }

    #[test]
    fn trailing_grid_v7_fills_unstuck_and_trailing_close_in_same_candle() {
        let hlcvs = Array3::from_shape_vec(
            (2, 1, 4),
            vec![
                100.0, 100.0, 100.0, 1.0, //
                102.0, 99.0, 100.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 10.0;
        bp_pair.long.wallet_exposure_limit = 10.0;
        bp_pair.long.risk_wel_enforcer_enabled = false;
        bp_pair.long.risk_twel_enforcer_enabled = false;
        bp_pair.long.unstuck_enabled = true;
        bp_pair.long.unstuck_ema_gating_enabled = false;
        bp_pair.long.unstuck_close_pct = 0.0195;
        bp_pair.long.unstuck_threshold = 0.9;
        bp_pair.long.unstuck_loss_allowance_pct = 0.01;
        bp_pair.short.n_positions = 0;
        bp_pair.short.total_wallet_exposure_limit = 0.0;
        bp_pair.short.wallet_exposure_limit = 0.0;

        let strategy = TrailingGridV7Params {
            ema_span_0: 1.0,
            ema_span_1: 1.0,
            entry: TrailingGridV7EntryParams::default(),
            close: TrailingGridV7CloseParams {
                trailing_grid_ratio: 1.0,
                trailing_qty_pct: 0.2624,
                trailing_retracement_pct: 0.005,
                trailing_threshold_pct: 0.01,
                ..Default::default()
            },
        };
        let strategy_params = StrategyParamsPairValue {
            long: serde_json::to_value(strategy).unwrap(),
            short: serde_json::to_value(strategy).unwrap(),
        };
        let backtest_params = BacktestParams {
            starting_balance: 1_000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };
        let exchange = ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
        };

        let mut bt = Backtest::new_with_strategy_params(
            hlcvs.view(),
            btc_usd_prices.view(),
            crate::strategies::StrategyKind::TrailingGridV7,
            vec![bp_pair],
            vec![strategy_params],
            vec![exchange],
            &backtest_params,
        );
        bt.positions.long[0] = Position {
            size: 95.6,
            price: 100.0,
        };
        bt.trailing_prices.long[0] = TrailingPriceBundle {
            min_since_open: 99.0,
            max_since_min: 102.0,
            max_since_open: 102.0,
            min_since_max: 100.0,
        };
        bt.orchestrator_input_cache = None;
        bt.update_open_orders_all(0).unwrap();

        let staged_closes = &bt.open_orders.long[0].closes;
        assert_eq!(staged_closes.len(), 2);
        assert!(staged_closes.iter().any(|order| {
            order.order.order_type == OrderType::CloseUnstuckLong
                && (order.order.qty + 1.95).abs() < 1e-12
        }));
        assert!(staged_closes.iter().any(|order| {
            order.order.order_type == OrderType::CloseTrailingLong
                && (order.order.qty + 26.24).abs() < 1e-12
        }));

        bt.check_for_fills(1).unwrap();

        let same_candle_closes: Vec<&Fill> = bt
            .fills
            .iter()
            .filter(|fill| fill.index == 1 && fill.fill_qty < 0.0)
            .collect();
        assert_eq!(same_candle_closes.len(), 2);
        assert!(same_candle_closes
            .iter()
            .any(|fill| fill.order_type == OrderType::CloseUnstuckLong));
        assert!(same_candle_closes
            .iter()
            .any(|fill| fill.order_type == OrderType::CloseTrailingLong));
        assert!(same_candle_closes
            .iter()
            .all(|fill| fill.timestamp_ms == 60_000));
    }

    #[test]
    fn worst_percentile_selection_matches_full_sort_bit_for_bit() {
        fn reference(values: &[f64]) -> f64 {
            if values.is_empty() {
                return 0.0;
            }
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| {
                a.abs().partial_cmp(&b.abs()).unwrap_or_else(|| {
                    if a.is_nan() && b.is_nan() {
                        Ordering::Equal
                    } else if a.is_nan() {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
            });
            let cutoff_index = std::cmp::max(1, (sorted.len() as f64 * 0.01) as usize);
            let worst_n = std::cmp::min(cutoff_index, sorted.len());
            sorted[sorted.len() - worst_n..]
                .iter()
                .map(|x| x.abs())
                .sum::<f64>()
                / worst_n as f64
        }

        let mut seed = 0x123456789abcdef_u64;
        for n in [0, 1, 2, 99, 100, 101, 199, 200, 201, 10001] {
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let magnitude = ((seed >> 32) % 1000) as f64 / 37.0;
                    if i % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            for data in [
                values.clone(),
                vec![0.0; n],
                vec![-1.0; n],
                values.iter().map(|v| v * 1e200).collect(),
                values.iter().map(|v| v * 1e-200).collect(),
            ] {
                assert_eq!(
                    mean_worst_1pct_abs(&data).to_bits(),
                    reference(&data).to_bits(),
                    "n={n}"
                );
                let mut reversed = data.clone();
                reversed.reverse();
                assert_eq!(
                    mean_worst_1pct_abs(&reversed).to_bits(),
                    reference(&reversed).to_bits()
                );
            }
        }
        for data in [
            vec![f64::NAN],
            vec![0.0, f64::NAN, -1.0],
            vec![f64::INFINITY, -f64::INFINITY, f64::NAN],
        ] {
            assert_eq!(
                mean_worst_1pct_abs(&data).to_bits(),
                reference(&data).to_bits()
            );
        }
    }

    #[test]
    fn strategy_equity_drawdowns_are_positive_severity_samples() {
        let mut series = vec![100.0; 198];
        series.push(1.0);
        series.push(50.0);

        let drawdowns = calc_strategy_equity_drawdowns(&series);

        assert!(drawdowns.iter().all(|x| *x >= 0.0));
        assert!((drawdowns[198] - 0.99).abs() < 1e-12);
        assert!((drawdowns[199] - 0.5).abs() < 1e-12);
        assert!((mean_worst_1pct_abs(&drawdowns) - 0.745).abs() < 1e-12);
    }

    #[test]
    fn daily_worst_positive_drawdowns_preserve_intraday_strategy_eq_stress() {
        let series = vec![100.0, 50.0, 110.0, 109.0];
        let timestamps = vec![0, 3_600_000, 86_400_000, 90_000_000];
        let drawdowns = calc_strategy_equity_drawdowns(&series);
        let daily_worst = daily_worst_positive_drawdowns(&drawdowns, &timestamps, series.len());

        assert_eq!(daily_worst.len(), 2);
        assert!((daily_worst[0] - 0.5).abs() < 1e-12);
        assert!((daily_worst[1] - (1.0 / 110.0)).abs() < 1e-12);
        assert!((mean_worst_1pct_abs(&daily_worst) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn strategy_eq_underwater_pct_uses_daily_worst_drawdowns() {
        let day = 86_400_000;
        let series = vec![100.0, 50.0, 110.0, 109.0, 110.0, 99.0];
        let timestamps = vec![
            0,
            3_600_000,
            day,
            day + 3_600_000,
            2 * day,
            2 * day + 3_600_000,
        ];
        let drawdowns = calc_strategy_equity_drawdowns(&series);
        let daily_worst = daily_worst_positive_drawdowns(&drawdowns, &timestamps, series.len());

        assert_eq!(daily_worst.len(), 3);
        assert!((mean_abs(&daily_worst) - ((0.5 + (1.0 / 110.0) + 0.1) / 3.0)).abs() < 1e-12);
        assert!((median_abs(&daily_worst) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn strategy_eq_recovery_days_measure_time_to_strictly_exceed_each_sample() {
        let day = 86_400_000;
        let series = vec![100.0, 90.0, 95.0, 101.0, 100.0, 102.0];
        let timestamps: Vec<u64> = (0..series.len()).map(|i| i as u64 * day).collect();

        let recovery = calc_strategy_eq_recovery_days(&series, &timestamps);

        assert!((recovery.mean - (8.0 / 6.0)).abs() < 1e-12);
        assert!((recovery.median - 1.0).abs() < 1e-12);
        assert!((recovery.p95 - 2.75).abs() < 1e-12);
        assert!((recovery.p99 - 2.95).abs() < 1e-12);
        assert!((recovery.mean_worst_5pct - 3.0).abs() < 1e-12);
        assert!((recovery.mean_worst_1pct - 3.0).abs() < 1e-12);
        assert!((recovery.max - 3.0).abs() < 1e-12);
    }

    #[test]
    fn strategy_eq_recovery_days_keeps_equal_equity_unresolved_until_strictly_higher() {
        let day = 86_400_000;
        let series = vec![100.0, 100.0, 101.0];
        let timestamps: Vec<u64> = (0..series.len()).map(|i| i as u64 * day).collect();

        let recovery = calc_strategy_eq_recovery_days(&series, &timestamps);

        assert!((recovery.mean - 1.0).abs() < 1e-12);
        assert!((recovery.median - 1.0).abs() < 1e-12);
        assert!((recovery.max - 2.0).abs() < 1e-12);
    }

    #[test]
    fn hard_stop_side_metrics_use_runtime_series_shared_max_and_sample_timestamps() {
        let hlcvs = Array3::from_shape_vec((4, 1, 4), vec![1.0; 4 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.short.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.short.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;
        bp_pair.short.ema_span_0 = 10.0;
        bp_pair.short.ema_span_1 = 20.0;
        bp_pair.long.hsl_enabled = true;
        bp_pair.short.hsl_enabled = true;

        let hs = EquityHardStopLossConfig::default();

        let backtest_params = BacktestParams {
            starting_balance: 100.0,
            maker_fee: 0.0,
            taker_fee: 0.0,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![3],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 365.0,
            liquidation_threshold: 0.05,
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            equity_hard_stop_loss: hs,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.equities.timestamps_ms = vec![0, 60_000, 120_000, 180_000];
        bt.strategy_equity_series_pside[LONG] = vec![100.0, 90.0, 90.0, 90.0];
        bt.strategy_equity_timestamps_ms_pside[LONG] = vec![0, 60_000, 120_000, 180_000];
        bt.hsl_report.signal_emas[1] = vec![0.0, 0.10, 0.10, 0.10];
        bt.strategy_equity_series_pside[SHORT] = vec![100.0, 90.0, 100.0, 90.0];
        bt.strategy_equity_timestamps_ms_pside[SHORT] = vec![0, 60_000, 120_000, 180_000];
        bt.hsl_report.signal_emas[2] = vec![0.0, 0.01, 0.005, 0.02];

        let strategy_metrics = bt.strategy_equity_metrics_for_analysis();
        assert!((strategy_metrics.long.drawdown_worst_ema_strategy_eq - 0.10).abs() < 1e-12);
        assert!((strategy_metrics.short.drawdown_worst_ema_strategy_eq - 0.02).abs() < 1e-12);
        assert!(
            (strategy_metrics
                .long
                .drawdown_worst_ema_strategy_eq
                .max(strategy_metrics.short.drawdown_worst_ema_strategy_eq)
                - 0.10)
                .abs()
                < 1e-12
        );
        assert!(
            (strategy_metrics
                .long
                .drawdown_worst_mean_1pct_ema_strategy_eq
                .max(
                    strategy_metrics
                        .short
                        .drawdown_worst_mean_1pct_ema_strategy_eq
                )
                - 0.10)
                .abs()
                < 1e-12
        );

        // Reproduce a halted-controller tail-alignment edge over 200 days.
        // The raw stresses of 90% late on day 0 and 80% early on day 1 are
        // separate daily worst samples with their true controller timestamps.
        // Tail-aligning the shortened series to account timestamps merges both
        // into one day and changes the worst-1% mean from 85% to 50%.
        let day_ms = 86_400_000_u64;
        let mut actual_timestamps = Vec::with_capacity(600);
        let mut series = Vec::with_capacity(600);
        for day in 0..200_u64 {
            actual_timestamps.extend([day * day_ms, day * day_ms + 60_000, day * day_ms + 120_000]);
            let peak = 100.0 + day as f64;
            let first_drawdown = if day == 1 { 0.8 } else { 0.1 };
            let second_drawdown = if day == 0 { 0.9 } else { 0.1 };
            series.extend([
                peak,
                peak * (1.0 - first_drawdown),
                peak * (1.0 - second_drawdown),
            ]);
        }
        bt.equities.timestamps_ms = actual_timestamps.clone();
        bt.equities.timestamps_ms.push(200 * day_ms);

        let actual =
            bt.strategy_equity_metrics_from_series(&series, None, Some(&actual_timestamps));
        let legacy_tail_aligned = bt.strategy_equity_metrics_from_series(&series, None, None);

        assert!((actual.drawdown_worst_mean_1pct_strategy_eq - 0.85).abs() < 1e-9);
        assert!((legacy_tail_aligned.drawdown_worst_mean_1pct_strategy_eq - 0.50).abs() < 1e-9);
    }

    #[test]
    fn liquidation_threshold_clamps_equity_and_stops() {
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.equities.timestamps_ms.push(0);
        bt.equities.usd_total_equity.push(-10.0);
        bt.equities.btc_total_equity.push(-0.001);
        assert!(bt.check_and_apply_liquidation(0));
        assert!((bt.equities.usd_total_equity[0] - 50.0).abs() < 1e-12);
        assert!((bt.equities.btc_total_equity[0] - 0.0025).abs() < 1e-12);
    }

    #[test]
    fn nonpositive_raw_balance_liquidates_even_when_equity_is_above_floor() {
        for raw_balance in [0.0, -1.0] {
            let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
            let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

            let mut bp_pair = BotParamsPair::default();
            bp_pair.long.n_positions = 1;
            bp_pair.long.ema_span_0 = 10.0;
            bp_pair.long.ema_span_1 = 20.0;

            let backtest_params = BacktestParams {
                starting_balance: 1000.0,
                maker_fee: 0.0,
                taker_fee: 0.00055,
                coins: vec!["TEST".to_string()],
                active_coin_indices: None,
                first_timestamp_ms: 0,
                requested_start_timestamp_ms: 0,
                first_valid_indices: vec![0],
                last_valid_indices: vec![1],
                warmup_minutes: vec![0],
                trade_start_indices: vec![0],
                global_warmup_bars: 0,
                btc_collateral_cap: 0.0,
                btc_collateral_ltv_cap: None,
                metrics_only: true,
                hsl_detailed_report: false,
                skip_btc_analysis: false,
                filter_by_min_effective_cost: false,
                dynamic_wel_by_tradability: true,
                hedge_mode: true,
                max_realized_loss_pct: 1.0,
                pnls_max_lookback_days: 30.0,
                liquidation_threshold: 0.05,
                equity_hard_stop_loss: EquityHardStopLossConfig::default(),
                market_orders_allowed: false,
                market_order_near_touch_threshold: 0.001,
                market_order_slippage_pct: 0.0005,
                limit_order_fill_buffer_pct: 0.0,
                forager_score_hysteresis_pct: 0.0,
                candle_interval_minutes: 1,
            };

            let mut bt = Backtest::new(
                hlcvs.view(),
                btc_usd_prices.view(),
                vec![bp_pair],
                vec![ExchangeParams::default()],
                &backtest_params,
            );

            bt.balance.usd_total_balance = raw_balance;
            bt.equities.timestamps_ms.push(0);
            bt.equities.usd_total_equity.push(100.0);
            bt.equities.btc_total_equity.push(0.005);

            assert!(bt.check_and_apply_liquidation(0));
            assert!(bt.liquidated());
            assert!((bt.equities.usd_total_equity[0] - 50.0).abs() < 1e-12);
            assert!((bt.equities.btc_total_equity[0] - 0.0025).abs() < 1e-12);
        }
    }

    #[test]
    fn depleted_raw_balance_preempts_coin_hsl_slot_budget_error() {
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![100.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.5;
        bp_pair.long.wallet_exposure_limit = 1.5;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;
        bp_pair.long.hsl_enabled = true;

        let hs = EquityHardStopLossConfig::default();

        let backtest_params = BacktestParams {
            starting_balance: 100.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: hs,
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.balance.usd_total_balance = -1.0;
        bt.equities.timestamps_ms.push(0);
        bt.equities.usd_total_equity.push(100.0);
        bt.equities.btc_total_equity.push(0.005);

        assert!(bt.check_and_apply_liquidation(0));
        assert!(bt.liquidated());
        assert!(bt.hsl_scopes.is_empty());
    }

    #[test]
    fn post_fill_depleted_balance_liquidates_before_orchestrator() {
        let hlcvs = Array3::from_shape_vec(
            (4, 1, 4),
            vec![
                101.0, 99.0, 100.0, 1.0, //
                101.0, 99.0, 100.0, 1.0, //
                2.0, 1.0, 1.0, 1.0, //
                2.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 5.0;
        bp_pair.long.wallet_exposure_limit = 5.0;
        bp_pair.long.ema_span_0 = 1.0;
        bp_pair.long.ema_span_1 = 1.0;

        let backtest_params = BacktestParams {
            starting_balance: 100.0,
            maker_fee: 0.0,
            taker_fee: 0.0,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![3],
            warmup_minutes: vec![0],
            trade_start_indices: vec![2],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams {
                maker_fee: 0.0,
                taker_fee: 0.0,
                ..Default::default()
            }],
            &backtest_params,
        );
        bt.positions.long[0] = Position {
            size: 2.0,
            price: 100.0,
        };
        bt.open_orders.long[0].closes.push(BacktestOrder {
            order: Order {
                qty: -2.0,
                price: 1.0,
                order_type: OrderType::CloseGridLong,
            },
            execution_type: orchestrator::ExecutionType::Limit,
        });
        bt.equity_tracking_active = true;

        let (fills, equities) = bt.run().expect("depleted balance should liquidate");

        assert!(bt.liquidated());
        assert_eq!(fills.len(), 1);
        assert!(fills[0].usd_total_balance < 0.0);
        assert_eq!(equities.timestamps_ms, vec![60_000, 120_000]);
        assert_eq!(equities.usd_total_equity, vec![100.0, 5.0]);
        assert!((equities.btc_total_equity[1] - 0.00025).abs() < 1e-12);
    }

    #[test]
    fn liquidation_floor_sample_is_reflected_in_hsl_metrics() {
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.balance.usd_total_balance = 1000.0;

        bt.equities.timestamps_ms.push(0);
        bt.equities.usd_total_equity.push(1000.0);
        bt.equities.btc_total_equity.push(0.05);
        bt.record_hsl_analysis(0, false);

        bt.equities.timestamps_ms.push(60_000);
        bt.equities.usd_total_equity.push(40.0);
        bt.equities.btc_total_equity.push(0.002);
        assert!(bt.check_and_apply_liquidation(1));
        bt.pnl_cumsum_running_net = -960.0;
        bt.record_hsl_analysis(1, true);

        let strategy_metrics = bt.strategy_equity_metrics_for_analysis();
        assert!((bt.equities.usd_total_equity[1] - 50.0).abs() < 1e-12);
        assert!((strategy_metrics.overall.drawdown_worst_strategy_eq - 0.96).abs() < 1e-12);
    }

    #[test]
    fn cached_orchestrator_input_updates_dynamic_wallet_exposure_limit() {
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.wallet_exposure_limit = 0.1;
        bp_pair.long.entry_initial_qty_pct = 0.1;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        let input = bt.get_orchestrator_input_cached(1, None, None);
        let runtime_budget = input.symbols[0]
            .long
            .runtime_budget
            .as_ref()
            .expect("expected long runtime budget");
        assert!(
            (runtime_budget.effective_wallet_exposure_limit - 0.1).abs() < 1e-12,
            "expected cached input WEL to match initial runtime budget"
        );
        bt.orchestrator_input_cache = Some(input);

        bt.runtime_budget[0].long.effective_wallet_exposure_limit = 0.2;

        let input = bt.get_orchestrator_input_cached(1, None, None);
        let runtime_budget = input.symbols[0]
            .long
            .runtime_budget
            .as_ref()
            .expect("expected refreshed long runtime budget");
        assert!(
            (runtime_budget.effective_wallet_exposure_limit - 0.2).abs() < 1e-12,
            "expected cached input WEL to update after runtime budget change"
        );
        bt.orchestrator_input_cache = Some(input);
    }

    #[test]
    fn orchestrator_input_routes_snapped_and_raw_balances_correctly() {
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 0.5;
        bp_pair.long.unstuck_loss_allowance_pct = 0.2;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: -1.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.balance.usd_total_balance = 200.0;
        bt.balance.usd_total_balance_rounded = 100.0;
        bt.pnl_cumsum_max = 10.0;
        bt.pnl_cumsum_running = 0.0;
        bt.pnl_cumsum_max_net = 10.0;
        bt.pnl_cumsum_running_net = 0.0;

        let input = bt.get_orchestrator_input_cached(1, None, None);
        assert!(
            (input.balance - 100.0).abs() < 1e-12,
            "expected snapped balance to route to input.balance"
        );
        assert!(
            (input.balance_raw - 200.0).abs() < 1e-12,
            "expected raw balance to route to input.balance_raw"
        );

        let allowance_pct = 0.2 * 0.5;
        let expected_from_raw = calc_auto_unstuck_allowance(
            200.0,
            allowance_pct,
            bt.pnl_cumsum_max_net,
            bt.pnl_cumsum_running_net,
        );
        let expected_from_snapped = calc_auto_unstuck_allowance(
            100.0,
            allowance_pct,
            bt.pnl_cumsum_max_net,
            bt.pnl_cumsum_running_net,
        );
        assert!(
            (input.global.unstuck_allowance_long - expected_from_raw).abs() < 1e-12,
            "expected unstuck allowance to use raw balance"
        );
        assert!(
            (input.global.unstuck_allowance_long - expected_from_snapped).abs() > 1e-9,
            "allowance should differ from snapped-balance path in this scenario"
        );
    }

    #[test]
    fn backtest_balance_raw_refreshed_on_each_cached_call() {
        // Verify that balance_raw is updated from self.balance.usd_total_balance
        // on each call to get_orchestrator_input_cached, even when the cache is reused.
        let hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 0.5;
        bp_pair.long.unstuck_loss_allowance_pct = 0.2;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 0.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        // Step 1: initial balance
        bt.balance.usd_total_balance = 1000.0;
        bt.balance.usd_total_balance_rounded = 1000.0;
        bt.pnl_cumsum_max = 0.0;
        bt.pnl_cumsum_running = 0.0;
        bt.pnl_cumsum_max_net = 0.0;
        bt.pnl_cumsum_running_net = 0.0;

        let input1 = bt.get_orchestrator_input_cached(1, None, None);
        assert!(
            (input1.balance_raw - 1000.0).abs() < 1e-12,
            "first call: balance_raw should be 1000"
        );
        assert!(
            (input1.balance - 1000.0).abs() < 1e-12,
            "first call: balance should be 1000"
        );
        // Return the input to the cache
        bt.orchestrator_input_cache = Some(input1);

        // Step 2: simulate a fill that changes raw balance but snapped stays
        bt.balance.usd_total_balance = 1050.0; // raw changed (profit fill)
        bt.balance.usd_total_balance_rounded = 1000.0; // snapped stays (hysteresis)
        bt.pnl_cumsum_max = 50.0;
        bt.pnl_cumsum_running = 50.0;
        bt.pnl_cumsum_max_net = 50.0;
        bt.pnl_cumsum_running_net = 50.0;

        let input2 = bt.get_orchestrator_input_cached(1, None, None);
        assert!(
            (input2.balance_raw - 1050.0).abs() < 1e-12,
            "second call: balance_raw should have updated to 1050"
        );
        assert!(
            (input2.balance - 1000.0).abs() < 1e-12,
            "second call: snapped balance should still be 1000"
        );

        // Verify the unstuck allowance used the new raw balance, not the old one
        let allowance_pct = 0.2 * 0.5;
        let expected_allowance = calc_auto_unstuck_allowance(1050.0, allowance_pct, 50.0, 50.0);
        assert!(
            (input2.global.unstuck_allowance_long - expected_allowance).abs() < 1e-12,
            "unstuck allowance should use updated raw balance (1050), got {}",
            input2.global.unstuck_allowance_long
        );
    }

    #[test]
    fn rolling_effective_pnl_cumsum_uses_true_window_and_expires_without_new_fills() {
        let hlcvs = Array3::from_shape_vec((4, 1, 4), vec![1.0; 4 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![3],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 1.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 24 * 60,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        record_realized_pnl_for_test(&mut bt, 0, 100.0);
        record_realized_pnl_for_test(&mut bt, 1, -90.0);

        let (peak1, current1) = bt.effective_pnl_cumsum(1);
        assert!((peak1 - 100.0).abs() < 1e-12);
        assert!((current1 - 10.0).abs() < 1e-12);

        let (peak2, current2) = bt.effective_pnl_cumsum(2);
        assert!(
            peak2.abs() < 1e-12,
            "expected stale positive peak to expire to the zero baseline"
        );
        assert!(
            (current2 - -90.0).abs() < 1e-12,
            "expected only the k=1 fill to remain active at k=2"
        );

        let (peak3, current3) = bt.effective_pnl_cumsum(3);
        assert!(
            peak3.abs() < 1e-12 && current3.abs() < 1e-12,
            "expected rolling pnl window to decay to zero after all fills expire"
        );
    }

    #[test]
    fn rolling_pnl_window_expiry_restores_unstuck_allowance_after_stale_peak_ages_out() {
        let hlcvs = Array3::from_shape_vec((4, 1, 4), vec![1.0; 4 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.unstuck_loss_allowance_pct = 0.02;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![3],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 1.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 24 * 60,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.balance.usd_total_balance = 1000.0;
        bt.balance.usd_total_balance_rounded = 1000.0;

        record_realized_pnl_for_test(&mut bt, 0, 100.0);
        record_realized_pnl_for_test(&mut bt, 1, -90.0);

        let input1 = bt.get_orchestrator_input_cached(1, None, None);
        assert!(
            input1.global.unstuck_allowance_long.abs() < 1e-12,
            "expected stale positive peak to suppress allowance while still in-window"
        );

        let input2 = bt.get_orchestrator_input_cached(2, None, None);
        assert!(
            input2.global.realized_pnl_cumsum_max.abs() < 1e-12,
            "expected rolling peak to decay to the zero baseline after the old positive fill expires"
        );
        assert!(
            (input2.global.realized_pnl_cumsum_last - -90.0).abs() < 1e-12,
            "expected rolling current pnl to keep only the still-active fill"
        );
        assert!(
            input2.global.unstuck_allowance_long.abs() < 1e-12,
            "expected allowance to stay exhausted while current realized PnL is below the zero baseline"
        );
    }

    fn naive_live_style_effective_pnl_cumsum(
        events: &[(usize, f64)],
        k: usize,
        lookback_bars: usize,
    ) -> (f64, f64) {
        let active: Vec<f64> = events
            .iter()
            .filter(|(event_k, _)| {
                lookback_bars == usize::MAX || k.saturating_sub(*event_k) <= lookback_bars
            })
            .map(|(_, pnl)| *pnl)
            .collect();
        if active.is_empty() {
            return (0.0, 0.0);
        }
        let mut cumsum = 0.0;
        let mut peak: f64 = 0.0;
        for pnl in active {
            cumsum += pnl;
            peak = peak.max(cumsum);
        }
        (peak, cumsum)
    }

    fn record_realized_pnl_for_test(bt: &mut Backtest, k: usize, pnl: f64) {
        bt.pnl_cumsum_running += pnl;
        bt.pnl_cumsum_max = bt.pnl_cumsum_max.max(bt.pnl_cumsum_running);
        bt.pnl_cumsum_running_net += pnl;
        bt.pnl_cumsum_max_net = bt.pnl_cumsum_max_net.max(bt.pnl_cumsum_running_net);
        bt.record_rolling_pnl(k, pnl);
    }

    #[test]
    fn rolling_effective_pnl_cumsum_keeps_peak_at_or_above_current_after_window_slide() {
        let hlcvs = Array3::from_shape_vec((4, 1, 4), vec![1.0; 4 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![3],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 2.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 24 * 60,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        // True in-window sequence at k=3 with a 2-bar lookback:
        // k=1: +50
        // k=2: -120
        // k=3: +190
        // current rolling pnl = 120, and the in-window peak should also be 120.
        record_realized_pnl_for_test(&mut bt, 0, 100.0);
        record_realized_pnl_for_test(&mut bt, 1, 50.0);
        record_realized_pnl_for_test(&mut bt, 2, -120.0);
        record_realized_pnl_for_test(&mut bt, 3, 190.0);

        let (peak, current) = bt.effective_pnl_cumsum(3);

        assert!(
            peak >= current - 1e-12,
            "rolling peak must never fall below current rolling pnl: peak={}, current={}",
            peak,
            current
        );
        assert!(
            (peak - 120.0).abs() < 1e-12,
            "expected in-window rolling peak to track the later high after the old base expired"
        );
        assert!(
            (current - 120.0).abs() < 1e-12,
            "expected in-window rolling current pnl to equal the surviving 3-event sum"
        );
    }

    #[test]
    fn rolling_pnl_rebase_bug_does_not_inflate_unstuck_allowance() {
        let hlcvs = Array3::from_shape_vec((4, 1, 4), vec![1.0; 4 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.unstuck_loss_allowance_pct = 0.01;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![3],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 2.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 24 * 60,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        bt.balance.usd_total_balance = 1000.0;
        bt.balance.usd_total_balance_rounded = 1000.0;

        record_realized_pnl_for_test(&mut bt, 0, 100.0);
        record_realized_pnl_for_test(&mut bt, 1, 50.0);
        record_realized_pnl_for_test(&mut bt, 2, -120.0);
        record_realized_pnl_for_test(&mut bt, 3, 190.0);

        let input = bt.get_orchestrator_input_cached(3, None, None);

        assert!(
            input.global.realized_pnl_cumsum_max >= input.global.realized_pnl_cumsum_last - 1e-12,
            "realized pnl peak must not be below current: peak={}, current={}",
            input.global.realized_pnl_cumsum_max,
            input.global.realized_pnl_cumsum_last
        );
        assert!(
            (input.global.realized_pnl_cumsum_max - 120.0).abs() < 1e-12,
            "expected current window peak to equal the current rolling pnl at the new high"
        );
        assert!(
            (input.global.realized_pnl_cumsum_last - 120.0).abs() < 1e-12,
            "expected current rolling pnl to equal the surviving 3-event sum"
        );
        assert!(
            (input.global.unstuck_allowance_long - 10.0).abs() < 1e-12,
            "expected allowance to remain anchored to balance when current rolling pnl is at the window peak"
        );
    }

    #[test]
    fn rolling_effective_pnl_cumsum_matches_naive_live_style_reference() {
        let hlcvs = Array3::from_shape_vec((7, 1, 4), vec![1.0; 7 * 1 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 7]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.unstuck_loss_allowance_pct = 0.01;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![6],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 2.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 24 * 60,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        let events = [
            (0usize, 100.0),
            (1, 50.0),
            (2, -120.0),
            (3, 190.0),
            (5, -30.0),
        ];
        let mut next_event_idx = 0usize;

        for k in 0..=6 {
            while next_event_idx < events.len() && events[next_event_idx].0 == k {
                let pnl = events[next_event_idx].1;
                record_realized_pnl_for_test(&mut bt, k, pnl);
                next_event_idx += 1;
            }
            let expected = naive_live_style_effective_pnl_cumsum(&events[..next_event_idx], k, 2);
            let actual = bt.effective_pnl_cumsum(k);
            assert!(
                (actual.0 - expected.0).abs() < 1e-12 && (actual.1 - expected.1).abs() < 1e-12,
                "expected live-style rolling pnl stats at k={} to be {:?}, got {:?}",
                k,
                expected,
                actual
            );
            let input = bt.get_orchestrator_input_cached(k, None, None);
            let expected_allowance =
                calc_auto_unstuck_allowance(1000.0, 0.01, expected.0, expected.1);
            assert!(
                (input.global.unstuck_allowance_long - expected_allowance).abs() < 1e-12,
                "expected unstuck allowance at k={} to match live-style reference",
                k
            );
        }
    }

    #[test]
    fn dynamic_wel_by_tradability_uses_non_shrinking_tradable_count() {
        let hlcvs = Array3::from_shape_vec((6, 3, 4), vec![1.0; 6 * 3 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 6]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 3;
        bp_pair.long.total_wallet_exposure_limit = 1.5;
        bp_pair.long.wallet_exposure_limit = -1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0, 0, 0],
            last_valid_indices: vec![5, 2, 5],
            warmup_minutes: vec![0, 0, 1],
            trade_start_indices: vec![0, 0, 1],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair.clone(), bp_pair.clone(), bp_pair],
            vec![
                ExchangeParams::default(),
                ExchangeParams::default(),
                ExchangeParams::default(),
            ],
            &backtest_params,
        );

        assert!(bt.update_n_positions_and_wallet_exposure_limits(0));
        assert!(
            (bt.runtime_budget[0].long.effective_wallet_exposure_limit - 0.75).abs() < 1e-12,
            "k=0 expected wel 1.5/2"
        );
        assert!(
            (bt.bot_params[0].long.wallet_exposure_limit + 1.0).abs() < 1e-12,
            "configured bot params must remain unchanged"
        );
        assert_eq!(bt.effective_n_positions.long, 2);

        assert!(bt.update_n_positions_and_wallet_exposure_limits(1));
        assert!(
            (bt.runtime_budget[0].long.effective_wallet_exposure_limit - 0.5).abs() < 1e-12,
            "k=1 expected wel 1.5/3 after third coin becomes tradable"
        );
        assert_eq!(bt.effective_n_positions.long, 3);

        assert!(bt.update_n_positions_and_wallet_exposure_limits(4));
        assert!(
            (bt.runtime_budget[0].long.effective_wallet_exposure_limit - 0.5).abs() < 1e-12,
            "k=4 expected wel to remain 1.5/3 after B delists"
        );
        assert_eq!(bt.effective_n_positions.long, 3);
    }

    #[test]
    fn dynamic_wel_by_tradability_uses_side_specific_eligible_counts() {
        let hlcvs = Array3::from_shape_vec((6, 4, 4), vec![1.0; 6 * 4 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 6]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 4;
        bp_pair.long.total_wallet_exposure_limit = 1.2;
        bp_pair.long.wallet_exposure_limit = -1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;
        bp_pair.short.n_positions = 4;
        bp_pair.short.total_wallet_exposure_limit = 2.0;
        bp_pair.short.wallet_exposure_limit = -1.0;
        bp_pair.short.ema_span_0 = 10.0;
        bp_pair.short.ema_span_1 = 20.0;

        let mut short_only = bp_pair.clone();
        short_only.long.entry_eligible = false;
        short_only.long.n_positions = 0;
        short_only.long.total_wallet_exposure_limit = 0.0;
        short_only.long.wallet_exposure_limit = 0.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0, 0, 0, 0],
            last_valid_indices: vec![5, 5, 5, 5],
            warmup_minutes: vec![0, 0, 0, 0],
            trade_start_indices: vec![0, 0, 0, 0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            forager_score_hysteresis_pct: 0.0,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair.clone(), bp_pair.clone(), bp_pair, short_only],
            vec![
                ExchangeParams::default(),
                ExchangeParams::default(),
                ExchangeParams::default(),
                ExchangeParams::default(),
            ],
            &backtest_params,
        );

        assert!(bt.update_n_positions_and_wallet_exposure_limits(0));
        assert_eq!(bt.effective_n_positions.long, 3);
        assert_eq!(bt.effective_n_positions.short, 4);
        assert!((bt.runtime_budget[0].long.effective_wallet_exposure_limit - 0.4).abs() < 1e-12);
        assert!((bt.runtime_budget[0].short.effective_wallet_exposure_limit - 0.5).abs() < 1e-12);
        assert_eq!(
            bt.runtime_budget[3].long.effective_wallet_exposure_limit,
            0.0
        );
        let input = bt.build_orchestrator_input_iter(0, None, None, 0..4);
        assert_eq!(
            input.symbols[3].long.mode,
            Some(orchestrator::TradingMode::GracefulStop)
        );
        assert_eq!(input.symbols[3].short.mode, None);
    }

    #[test]
    fn fixed_backtest_wel_uses_configured_n_positions() {
        let hlcvs = Array3::from_shape_vec((6, 2, 4), vec![1.0; 6 * 2 * 4]).unwrap();
        let btc_usd_prices = Array1::from_vec(vec![20_000.0; 6]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 4;
        bp_pair.long.total_wallet_exposure_limit = 2.0;
        bp_pair.long.wallet_exposure_limit = -1.0;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["A".to_string(), "B".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0, 0],
            last_valid_indices: vec![5, 1],
            warmup_minutes: vec![0, 0],
            trade_start_indices: vec![0, 0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair.clone(), bp_pair],
            vec![ExchangeParams::default(), ExchangeParams::default()],
            &backtest_params,
        );

        assert!(bt.update_n_positions_and_wallet_exposure_limits(0));
        assert!(
            (bt.runtime_budget[0].long.effective_wallet_exposure_limit - 0.5).abs() < 1e-12,
            "k=0 expected fixed wel 2.0/4"
        );
        assert!(
            (bt.bot_params[0].long.wallet_exposure_limit + 1.0).abs() < 1e-12,
            "configured bot params must remain unchanged"
        );
        assert_eq!(bt.effective_n_positions.long, 4);

        assert!(bt.update_n_positions_and_wallet_exposure_limits(4));
        assert!(
            (bt.runtime_budget[0].long.effective_wallet_exposure_limit - 0.5).abs() < 1e-12,
            "k=4 expected fixed wel to stay 2.0/4 despite fewer tradable coins"
        );
        assert_eq!(bt.effective_n_positions.long, 4);
    }

    #[test]
    fn independent_unstuck_emas_refresh_in_cached_orchestrator_input() {
        let mut hlcvs = Array3::from_shape_vec((2, 1, 4), vec![1.0; 2 * 1 * 4]).unwrap();
        hlcvs[[1, 0, CLOSE]] = 1.5;
        let btc_usd_prices = Array1::from_vec(vec![20_000.0, 20_000.0]);

        let mut bp_pair = BotParamsPair::default();
        bp_pair.long.n_positions = 1;
        bp_pair.long.total_wallet_exposure_limit = 1.0;
        bp_pair.long.wallet_exposure_limit = 0.1;
        bp_pair.long.entry_initial_qty_pct = 0.1;
        bp_pair.long.ema_span_0 = 10.0;
        bp_pair.long.ema_span_1 = 20.0;

        bp_pair.long.unstuck_ema_span_0 = 7.5;
        bp_pair.long.unstuck_ema_span_1 = 31.25;
        bp_pair.short.unstuck_ema_span_0 = 10.0;
        bp_pair.short.unstuck_ema_span_1 = 20.0;
        let backtest_params = BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.00055,
            coins: vec!["TEST".to_string()],
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0],
            last_valid_indices: vec![1],
            warmup_minutes: vec![0],
            trade_start_indices: vec![0],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: true,
            hsl_detailed_report: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 30.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0005,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        let first = bt.get_orchestrator_input_cached(0, None, None);
        assert_eq!(first.symbols[0].emas.m1.close.len(), 9);
        bt.orchestrator_input_cache = Some(first);
        bt.update_emas(1);
        let refreshed = bt.get_orchestrator_input_cached(1, None, None);
        let fresh = bt.build_orchestrator_input_iter(1, None, None, 0..1);
        assert_eq!(
            refreshed.symbols[0].emas.m1.close,
            fresh.symbols[0].emas.m1.close
        );
        for (span, value) in unstuck_spans(&bt.bot_params[0].long)
            .into_iter()
            .zip(bt.emas[0].unstuck_long)
        {
            let alpha = 2.0 / (span + 1.0);
            let expected = alpha * 1.5 + (1.0 - alpha);
            assert!((value - expected).abs() < 1e-12);
            assert!(refreshed.symbols[0].emas.m1.close.contains(&(span, value)));
        }
        // A migrated pair uses exactly the existing strategy EMA arithmetic.
        assert_eq!(bt.emas[0].unstuck_short, bt.emas[0].long);
    }

    #[test]
    fn disabled_unstuck_gate_does_not_extend_warmup() {
        let mut bp = BotParamsPair::default();
        let strategies = strategy_pair_for_ema_tests(&bp);
        let baseline = calc_warmup_bars(&[bp.clone()], &[strategies.clone()]);
        bp.long.unstuck_loss_allowance_pct = 0.01;
        bp.long.unstuck_close_pct = 0.1;
        bp.long.unstuck_threshold = 0.9;
        bp.long.total_wallet_exposure_limit = 1.0;
        bp.long.n_positions = 1;
        bp.long.unstuck_ema_span_0 = 400_000.25;
        bp.long.unstuck_ema_span_1 = 500_000.25;
        bp.long.unstuck_enabled = false;
        assert_eq!(
            calc_warmup_bars(&[bp.clone()], &[strategies.clone()]),
            baseline
        );
        bp.long.unstuck_enabled = true;
        bp.long.unstuck_ema_gating_enabled = false;
        assert_eq!(
            calc_warmup_bars(&[bp.clone()], &[strategies.clone()]),
            baseline
        );
        bp.long.unstuck_ema_gating_enabled = true;
        assert_eq!(
            calc_warmup_bars(&[bp.clone()], &[strategies.clone()]),
            500_001
        );
        for control in 0..5 {
            let mut inactive = bp.clone();
            match control {
                0 => inactive.long.unstuck_loss_allowance_pct = 0.0,
                1 => inactive.long.unstuck_close_pct = 0.0,
                2 => inactive.long.unstuck_threshold = 0.0,
                3 => inactive.long.total_wallet_exposure_limit = 0.0,
                _ => inactive.long.n_positions = 0,
            }
            assert_eq!(
                calc_warmup_bars(&[inactive], &[strategies.clone()]),
                baseline
            );
        }
    }

    #[test]
    fn independent_unstuck_alphas_preserve_fractional_minutes_at_every_interval() {
        let mut bp = BotParamsPair::default();
        bp.long.unstuck_ema_span_0 = 17.25;
        bp.long.unstuck_ema_span_1 = 211.75;
        let strategies = strategy_pair_for_ema_tests(&bp);
        for interval in [1, 5, 15] {
            let alphas = calc_ema_alphas(&bp, &strategies, interval);
            for (span, alpha) in unstuck_spans(&bp.long)
                .into_iter()
                .zip(alphas.unstuck_long.alphas)
            {
                assert!((alpha - (2.0 / (span / interval as f64 + 1.0)).min(1.0)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_ema_alpha_interval_1_matches_original_formula() {
        // With interval=1, alpha should equal 2/(span+1) (the original formula)
        let mut bp = BotParamsPair::default();
        bp.long.ema_span_0 = 100.0;
        bp.long.ema_span_1 = 200.0;
        bp.short.ema_span_0 = 50.0;
        bp.short.ema_span_1 = 150.0;
        bp.long.filter_volume_ema_span_1m = 300.0;
        bp.short.filter_volume_ema_span_1m = 400.0;
        bp.long.filter_volatility_ema_span_1m = 500.0;
        bp.short.filter_volatility_ema_span_1m = 600.0;
        let strategy_pair = strategy_pair_for_ema_tests(&bp);

        let alphas = calc_ema_alphas(&bp, &strategy_pair, 1);

        // span2 = sqrt(100*200) = 141.42..., sorted: [100, 141.42, 200]
        let span2_long = (100.0f64 * 200.0).sqrt();
        let expected_long = [
            2.0 / (100.0 + 1.0),
            2.0 / (span2_long + 1.0),
            2.0 / (200.0 + 1.0),
        ];
        for (i, &expected) in expected_long.iter().enumerate() {
            assert!(
                (alphas.long.alphas[i] - expected).abs() < 1e-12,
                "long alpha[{}]: expected {}, got {}",
                i,
                expected,
                alphas.long.alphas[i]
            );
        }

        assert!((alphas.vol_alpha_long - 2.0 / 301.0).abs() < 1e-12);
        assert!((alphas.vol_alpha_short - 2.0 / 401.0).abs() < 1e-12);
        assert!((alphas.log_range_alpha_long - 2.0 / 501.0).abs() < 1e-12);
        assert!((alphas.log_range_alpha_short - 2.0 / 601.0).abs() < 1e-12);
    }

    #[test]
    fn test_ema_alpha_interval_5_adjusts_correctly() {
        // With interval=5, a 60-minute span becomes 12 candle periods
        // alpha = 2 / (60/5 + 1) = 2/13
        let mut bp = BotParamsPair::default();
        bp.long.ema_span_0 = 60.0;
        bp.long.ema_span_1 = 60.0; // same so span2=60 too
        bp.short.ema_span_0 = 60.0;
        bp.short.ema_span_1 = 60.0;
        let strategy_pair = strategy_pair_for_ema_tests(&bp);

        let alphas = calc_ema_alphas(&bp, &strategy_pair, 5);

        let expected = 2.0 / (60.0 / 5.0 + 1.0); // 2/13
        for i in 0..3 {
            assert!(
                (alphas.long.alphas[i] - expected).abs() < 1e-12,
                "long alpha[{}]: expected {}, got {}",
                i,
                expected,
                alphas.long.alphas[i]
            );
        }
    }

    #[test]
    fn test_make_orchestrator_ema_slots_assigns_fixed_positions() {
        let strategy_pair = StrategyParamsPair {
            long: StrategyParams::EmaAnchor(crate::strategies::EmaAnchorParams {
                offset_volatility_ema_span_1m: 30.0,
                offset_volatility_1m_weight: 2.0,
                offset_volatility_ema_span_1h: 12.0,
                offset_volatility_1h_weight: 1.0,
                ..Default::default()
            }),
            short: StrategyParams::EmaAnchor(crate::strategies::EmaAnchorParams {
                offset_volatility_ema_span_1m: 45.0,
                offset_volatility_1m_weight: 3.0,
                offset_volatility_ema_span_1h: 18.0,
                offset_volatility_1h_weight: 0.5,
                ..Default::default()
            }),
        };

        let slots = make_orchestrator_ema_slots(&strategy_pair, &BotParamsPair::default());

        assert_eq!(slots.m1_volume_long, 0);
        assert_eq!(slots.m1_volume_short, 1);
        assert_eq!(slots.m1_log_range.forager_long, 0);
        assert_eq!(slots.m1_log_range.forager_short, 1);
        assert_eq!(slots.m1_log_range.offset_long, Some(2));
        assert_eq!(slots.m1_log_range.offset_short, Some(3));
        assert_eq!(slots.h1_log_range.long, Some(0));
        assert_eq!(slots.h1_log_range.short, Some(1));
    }

    #[test]
    fn test_orch_profile_write_to_path_creates_parent_directory() {
        let base = std::env::temp_dir().join(format!(
            "passivbot-orch-profile-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let out_path = base
            .join("measurements")
            .join("orch_profile_orchestrator.json");
        let profile = OrchProfile {
            mode: "test",
            steps: 1,
            total_ns: 2,
            clear_orders_ns: 3,
            peek_hints_ns: 4,
            input_update_ns: 5,
            compute_ns: 6,
            distribute_ns: 7,
            sort_bundles_ns: 8,
        };

        profile.write_to_path(&out_path);

        let contents = fs::read_to_string(&out_path).expect("profile file should be written");
        assert!(contents.contains("\"mode\": \"test\""));

        let _ = fs::remove_file(&out_path);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn test_ema_alpha_hourly_volatility_not_adjusted() {
        // entry_volatility spans are in hours and calendar-based; should NOT change with interval
        let mut bp = BotParamsPair::default();
        bp.long.entry_volatility_ema_span_1h = 24.0;
        bp.short.entry_volatility_ema_span_1h = 48.0;
        let strategy_pair = strategy_pair_for_ema_tests(&bp);

        let alphas_1 = calc_ema_alphas(&bp, &strategy_pair, 1);
        let alphas_5 = calc_ema_alphas(&bp, &strategy_pair, 5);

        assert!(
            (alphas_1.volatility_ema_1h_alpha_long - alphas_5.volatility_ema_1h_alpha_long).abs()
                < 1e-12,
            "hourly volatility alpha should not change with interval"
        );
        assert!(
            (alphas_1.volatility_ema_1h_alpha_short - alphas_5.volatility_ema_1h_alpha_short).abs()
                < 1e-12,
            "hourly volatility alpha should not change with interval"
        );
    }

    #[test]
    fn test_hourly_volatility_span_below_one_hour_floors_alpha_to_one() {
        let strategy_pair = StrategyParamsPair {
            long: StrategyParams::TrailingGridV7(TrailingGridV7Params {
                entry: TrailingGridV7EntryParams {
                    volatility_ema_span_hours: 0.75,
                    ..Default::default()
                },
                ..Default::default()
            }),
            short: StrategyParams::TrailingGridV7(TrailingGridV7Params {
                entry: TrailingGridV7EntryParams {
                    volatility_ema_span_hours: 0.25,
                    ..Default::default()
                },
                ..Default::default()
            }),
        };

        let alphas = calc_ema_alphas(&BotParamsPair::default(), &strategy_pair, 1);

        assert_eq!(alphas.volatility_ema_1h_alpha_long, 1.0);
        assert_eq!(alphas.volatility_ema_1h_alpha_short, 1.0);
    }
}

fn calc_warmup_bars(bot_params: &[BotParamsPair], strategy_params: &[StrategyParamsPair]) -> usize {
    let mut max_span_minutes = 0.0f64;

    for (pair, strategy_pair) in bot_params.iter().zip(strategy_params.iter()) {
        let (long_span_0, long_span_1) = strategy_ema_spans(&strategy_pair.long);
        let (short_span_0, short_span_1) = strategy_ema_spans(&strategy_pair.short);
        let spans_long = [
            long_span_0,
            long_span_1,
            if pair.long.unstuck_enabled
                && pair.long.unstuck_ema_gating_enabled
                && pair.long.unstuck_loss_allowance_pct > 0.0
                && pair.long.unstuck_close_pct > 0.0
                && pair.long.unstuck_threshold > 0.0
                && pair.long.total_wallet_exposure_limit > 0.0
                && pair.long.n_positions > 0
            {
                pair.long.unstuck_ema_span_0
            } else {
                0.0
            },
            if pair.long.unstuck_enabled
                && pair.long.unstuck_ema_gating_enabled
                && pair.long.unstuck_loss_allowance_pct > 0.0
                && pair.long.unstuck_close_pct > 0.0
                && pair.long.unstuck_threshold > 0.0
                && pair.long.total_wallet_exposure_limit > 0.0
                && pair.long.n_positions > 0
            {
                pair.long.unstuck_ema_span_1
            } else {
                0.0
            },
            pair.long.filter_volume_ema_span_1m as f64,
            pair.long.filter_volatility_ema_span_1m as f64,
            strategy_entry_volatility_span_hours(&strategy_pair.long).unwrap_or(0.0) * 60.0,
        ];
        let spans_short = [
            short_span_0,
            short_span_1,
            if pair.short.unstuck_enabled
                && pair.short.unstuck_ema_gating_enabled
                && pair.short.unstuck_loss_allowance_pct > 0.0
                && pair.short.unstuck_close_pct > 0.0
                && pair.short.unstuck_threshold > 0.0
                && pair.short.total_wallet_exposure_limit > 0.0
                && pair.short.n_positions > 0
            {
                pair.short.unstuck_ema_span_0
            } else {
                0.0
            },
            if pair.short.unstuck_enabled
                && pair.short.unstuck_ema_gating_enabled
                && pair.short.unstuck_loss_allowance_pct > 0.0
                && pair.short.unstuck_close_pct > 0.0
                && pair.short.unstuck_threshold > 0.0
                && pair.short.total_wallet_exposure_limit > 0.0
                && pair.short.n_positions > 0
            {
                pair.short.unstuck_ema_span_1
            } else {
                0.0
            },
            pair.short.filter_volume_ema_span_1m as f64,
            pair.short.filter_volatility_ema_span_1m as f64,
            strategy_entry_volatility_span_hours(&strategy_pair.short).unwrap_or(0.0) * 60.0,
        ];
        for span in spans_long.iter().chain(spans_short.iter()) {
            if span.is_finite() {
                max_span_minutes = max_span_minutes.max(*span);
            }
        }
    }

    max_span_minutes.ceil() as usize
}
