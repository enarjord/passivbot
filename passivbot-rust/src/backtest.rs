use crate::constants::{CLOSE, HIGH, LONG, LOW, SHORT, VOLUME};
use crate::entries::calc_min_entry_qty;
use crate::orchestrator;
use crate::orchestrator::{
    EmaBundle as OrchestratorEmaBundle, EmaTimeframeBundle as OrchestratorEmaTimeframeBundle,
    EntryPeekHints,
};
use crate::trailing::{reset_trailing_bundle, update_trailing_bundle_with_candle};
use crate::types::{
    BacktestParams, Balance, BotParams, BotParamsPair, EMABands, Equities, ExchangeParams, Fill,
    Order, OrderBook, Position, Positions, TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_wallet_exposure, hysteresis, qty_to_cost, round_, round_dn, round_up,
};
use serde::Serialize;

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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
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

#[derive(Clone, Default, Copy, Debug)]
pub struct EmaAlphas {
    pub long: Alphas,
    pub short: Alphas,
    pub vol_alpha_long: f64,
    pub vol_alpha_short: f64,
    pub log_range_alpha_long: f64,
    pub log_range_alpha_short: f64,
    pub entry_volatility_logrange_ema_1h_alpha_long: f64,
    pub entry_volatility_logrange_ema_1h_alpha_short: f64,
}

#[derive(Clone, Default, Copy, Debug)]
pub struct Alphas {
    pub alphas: [f64; 3],
}

#[derive(Debug)]
pub struct EMAs {
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
    pub entry_volatility_logrange_ema_1h_long: f64,
    pub entry_volatility_logrange_ema_1h_long_num: f64,
    pub entry_volatility_logrange_ema_1h_long_den: f64,
    pub entry_volatility_logrange_ema_1h_short: f64,
    pub entry_volatility_logrange_ema_1h_short_num: f64,
    pub entry_volatility_logrange_ema_1h_short_den: f64,
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

impl EMAs {
    pub fn compute_bands(&self, pside: usize) -> EMABands {
        let (upper, lower) = match pside {
            LONG => (
                *self
                    .long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                *self
                    .long
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            ),
            SHORT => (
                *self
                    .short
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                *self
                    .short
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

#[derive(Debug, Default)]
pub struct OpenOrders {
    pub long: BTreeMap<usize, OpenOrderBundle>,
    pub short: BTreeMap<usize, OpenOrderBundle>,
}

#[derive(Debug, Default)]
pub struct OpenOrderBundle {
    pub entries: Vec<Order>,
    pub closes: Vec<Order>,
}

#[derive(Default, Debug)]
pub struct TrailingPrices {
    pub long: HashMap<usize, TrailingPriceBundle>,
    pub short: HashMap<usize, TrailingPriceBundle>,
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

// RollingSum (SMA) removed — volume & log range are now tracked via EMAs in `EMAs`.

pub struct Backtest<'a> {
    hlcvs: ArrayView3<'a, f64>,
    btc_usd_prices: ArrayView1<'a, f64>, // Change to ArrayView1 (1D view)
    active_coin_indices: Vec<usize>,
    interval_ms: u64,
    bot_params_master: BotParamsPair,
    bot_params: Vec<BotParamsPair>,
    bot_params_original: Vec<BotParamsPair>,
    effective_n_positions: EffectiveNPositions,
    exchange_params_list: Vec<ExchangeParams>,
    backtest_params: BacktestParams,
    pub balance: Balance,
    n_coins: usize,
    ema_alphas: Vec<EmaAlphas>,
    emas: Vec<EMAs>,
    needs_volume_ema_long: bool,
    needs_volume_ema_short: bool,
    needs_log_range_long: bool,
    needs_log_range_short: bool,
    needs_entry_volatility_logrange_ema_1h_long: bool,
    needs_entry_volatility_logrange_ema_1h_short: bool,
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
    fills: Vec<Fill>,
    trading_enabled: TradingEnabled,
    trailing_enabled: Vec<TrailingEnabled>,
    any_trailing_long: bool,
    any_trailing_short: bool,
    equities: Equities,
    last_valid_timestamps: HashMap<usize, usize>,
    first_valid_timestamps: HashMap<usize, usize>,
    did_fill_long: HashSet<usize>,
    did_fill_short: HashSet<usize>,
    pub total_wallet_exposures: Vec<f64>,
    // removed rolling_volume_sum & buffer — replaced by per-coin EMAs in `emas`
    equity_tracking_active: bool,
    debug_writer: Option<DebugOrderWriter>,
    debug_balance_writer: Option<DebugBalanceWriter>,
    orchestrator_input_cache: Option<orchestrator::OrchestratorInput>,
    orchestrator_workspace: orchestrator::OrchestratorWorkspace,
    orch_profile: Option<OrchProfile>,
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

    fn write_to_file(&self) {
        let fname = "measurements/orch_profile_orchestrator.json";
        if let Ok(file) = File::create(fname) {
            let mut w = BufWriter::new(file);
            let _ = serde_json::to_writer_pretty(&mut w, self);
            let _ = w.flush();
        }
    }
}

fn calc_entry_balance_pct(params: &BotParams, effective_n_positions: usize) -> f64 {
    if effective_n_positions == 0 {
        return 0.0;
    }
    let allowance_multiplier = 1.0 + params.risk_we_excess_allowance_pct.max(0.0);
    params.total_wallet_exposure_limit * params.entry_initial_qty_pct * allowance_multiplier
        / effective_n_positions as f64
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

    fn debug_dump_unstuck_calc(&self, k: usize, idx: usize, side: usize) {
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
            LONG => self.positions.long.get(&idx).copied().unwrap_or_default(),
            SHORT => self.positions.short.get(&idx).copied().unwrap_or_default(),
            _ => return,
        };
        if position.size == 0.0 || !position.price.is_finite() || position.price <= 0.0 {
            return;
        }

        let balance = self.balance.usd_total_balance_rounded;
        let balance_raw = self.balance.usd_total_balance;
        let allowance = match side {
            LONG => {
                if self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0 {
                    calc_auto_unstuck_allowance(
                        balance_raw,
                        self.bot_params_master.long.unstuck_loss_allowance_pct
                            * self.bot_params_master.long.total_wallet_exposure_limit,
                        self.pnl_cumsum_max,
                        self.pnl_cumsum_running,
                    )
                } else {
                    0.0
                }
            }
            SHORT => {
                if self.bot_params_master.short.unstuck_loss_allowance_pct > 0.0 {
                    calc_auto_unstuck_allowance(
                        balance_raw,
                        self.bot_params_master.short.unstuck_loss_allowance_pct
                            * self.bot_params_master.short.total_wallet_exposure_limit,
                        self.pnl_cumsum_max,
                        self.pnl_cumsum_running,
                    )
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let bp = self.bp(idx, side);
        let ema_bands = self.emas[idx].compute_bands(side);
        let current_price = self.hlcvs_value(k, idx, CLOSE);
        let ex = &self.exchange_params_list[idx];

        let size_abs = position.size.abs();
        let allowance_multiplier = 1.0 + bp.risk_we_excess_allowance_pct.max(0.0);
        let effective_wel = bp.wallet_exposure_limit * allowance_multiplier;
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
            wallet_exposure_limit: bp.wallet_exposure_limit,
            wallet_exposure_limit_bits: bp.wallet_exposure_limit.to_bits(),
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

    fn build_orchestrator_input_iter<I>(
        &self,
        k: usize,
        peek_hints: Option<EntryPeekHints>,
        indices: I,
    ) -> orchestrator::OrchestratorInput
    where
        I: IntoIterator<Item = usize>,
    {
        let balance = self.balance.usd_total_balance_rounded;
        let balance_raw = self.balance.usd_total_balance;

        let long_allowance = if self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0 {
            calc_auto_unstuck_allowance(
                balance_raw,
                self.bot_params_master.long.unstuck_loss_allowance_pct
                    * self.bot_params_master.long.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            )
        } else {
            0.0
        };
        let short_allowance = if self.bot_params_master.short.unstuck_loss_allowance_pct > 0.0 {
            calc_auto_unstuck_allowance(
                balance_raw,
                self.bot_params_master.short.unstuck_loss_allowance_pct
                    * self.bot_params_master.short.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
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
                let effective_min_cost =
                    qty_to_cost(exchange.min_qty, close_price, exchange.c_mult)
                        .max(exchange.min_cost);

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
                        low,
                        high,
                        tradable: tradable_next,
                    })
                } else {
                    None
                };
                let valid_now = self.coin_is_valid_at(idx, k);

                let pos_long = *self
                    .positions
                    .long
                    .get(&idx)
                    .unwrap_or(&Position::default());
                let pos_short = *self
                    .positions
                    .short
                    .get(&idx)
                    .unwrap_or(&Position::default());

                let mut mode_long: Option<orchestrator::TradingMode> = None;
                let mut mode_short: Option<orchestrator::TradingMode> = None;

                // Backtest delist behaviour: if a coin is delisted (ends early), switch to panic at the
                // last valid candle so we force a close while the market is still "tradeable".
                if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
                    if k >= delist_timestamp {
                        if pos_long.size != 0.0 {
                            mode_long = Some(orchestrator::TradingMode::Panic);
                        }
                        if pos_short.size != 0.0 {
                            mode_short = Some(orchestrator::TradingMode::Panic);
                        }
                    }
                } else {
                    // Fallback: if data is already invalid and we still have a position, panic as well.
                    if !valid_now && pos_long.size != 0.0 {
                        mode_long = Some(orchestrator::TradingMode::Panic);
                    }
                    if !valid_now && pos_short.size != 0.0 {
                        mode_short = Some(orchestrator::TradingMode::Panic);
                    }
                }

                // filter_by_min_effective_cost => GracefulStop (blocks only initial entries).
                if self.backtest_params.filter_by_min_effective_cost {
                    if !self.coin_passes_min_effective_cost(idx, LONG) && pos_long.size == 0.0 {
                        mode_long = Some(orchestrator::TradingMode::GracefulStop);
                    }
                    if !self.coin_passes_min_effective_cost(idx, SHORT) && pos_short.size == 0.0 {
                        mode_short = Some(orchestrator::TradingMode::GracefulStop);
                    }
                }

                // Build EMA bundle (per-coin spans; must match how EMAs were computed).
                let mut m1 = OrchestratorEmaTimeframeBundle::default();
                let mut h1 = OrchestratorEmaTimeframeBundle::default();

                // 1m close EMAs (3 spans) per pside
                {
                    let bp = &self.bot_params[idx].long;
                    let mut spans = [
                        bp.ema_span_0,
                        bp.ema_span_1,
                        (bp.ema_span_0 * bp.ema_span_1).sqrt(),
                    ];
                    spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    for (span, value) in spans.into_iter().zip(self.emas[idx].long.into_iter()) {
                        m1.close.push((span, value));
                    }
                }
                {
                    let bp = &self.bot_params[idx].short;
                    let mut spans = [
                        bp.ema_span_0,
                        bp.ema_span_1,
                        (bp.ema_span_0 * bp.ema_span_1).sqrt(),
                    ];
                    spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    for (span, value) in spans.into_iter().zip(self.emas[idx].short.into_iter()) {
                        m1.close.push((span, value));
                    }
                }

                // 1m volume/log-range EMAs (used by forager); these are global by spec.
                // We assume the spans match across coins (as the backtest EMA alphas were built per-coin).
                let vol_span_long = self.bot_params_master.long.filter_volume_ema_span as f64;
                let vol_span_short = self.bot_params_master.short.filter_volume_ema_span as f64;
                let lr_span_long = self.bot_params_master.long.filter_volatility_ema_span as f64;
                let lr_span_short = self.bot_params_master.short.filter_volatility_ema_span as f64;

                debug_assert_eq!(
                    self.bot_params[idx].long.filter_volume_ema_span as f64, vol_span_long,
                    "coin {} long filter_volume_ema_span differs from master",
                    idx
                );
                debug_assert_eq!(
                    self.bot_params[idx].short.filter_volume_ema_span as f64, vol_span_short,
                    "coin {} short filter_volume_ema_span differs from master",
                    idx
                );
                debug_assert_eq!(
                    self.bot_params[idx].long.filter_volatility_ema_span as f64, lr_span_long,
                    "coin {} long filter_volatility_ema_span differs from master",
                    idx
                );
                debug_assert_eq!(
                    self.bot_params[idx].short.filter_volatility_ema_span as f64, lr_span_short,
                    "coin {} short filter_volatility_ema_span differs from master",
                    idx
                );

                m1.volume.push((vol_span_long, self.emas[idx].vol_long));
                m1.volume.push((vol_span_short, self.emas[idx].vol_short));
                m1.log_range
                    .push((lr_span_long, self.emas[idx].log_range_long));
                m1.log_range
                    .push((lr_span_short, self.emas[idx].log_range_short));

                // 1h log-range EMA for grid spacing/trailing volatility weight (per-coin span).
                {
                    let span = self.bot_params[idx].long.entry_volatility_ema_span_hours;
                    if span > 0.0 {
                        h1.log_range
                            .push((span, self.emas[idx].entry_volatility_logrange_ema_1h_long));
                    }
                }
                {
                    let span = self.bot_params[idx].short.entry_volatility_ema_span_hours;
                    if span > 0.0 {
                        h1.log_range
                            .push((span, self.emas[idx].entry_volatility_logrange_ema_1h_short));
                    }
                }

                let emas = OrchestratorEmaBundle { m1, h1 };

                let trailing_long = self
                    .trailing_prices
                    .long
                    .get(&idx)
                    .cloned()
                    .unwrap_or_default();
                let trailing_short = self
                    .trailing_prices
                    .short
                    .get(&idx)
                    .cloned()
                    .unwrap_or_default();

                orchestrator::SymbolInput {
                    symbol_idx: idx,
                    order_book,
                    exchange,
                    tradable,
                    next_candle,
                    effective_min_cost,
                    emas,
                    long: orchestrator::SymbolSideInput {
                        mode: mode_long,
                        position: pos_long,
                        trailing: trailing_long,
                        bot_params: self.bot_params[idx].long.clone(),
                    },
                    short: orchestrator::SymbolSideInput {
                        mode: mode_short,
                        position: pos_short,
                        trailing: trailing_short,
                        bot_params: self.bot_params[idx].short.clone(),
                    },
                }
            })
            .collect();

        orchestrator::OrchestratorInput {
            balance,
            balance_raw,
            global: orchestrator::OrchestratorGlobal {
                filter_by_min_effective_cost: self.backtest_params.filter_by_min_effective_cost,
                unstuck_allowance_long: long_allowance,
                unstuck_allowance_short: short_allowance,
                max_realized_loss_pct: self.backtest_params.max_realized_loss_pct,
                realized_pnl_cumsum_max: self.pnl_cumsum_max,
                realized_pnl_cumsum_last: self.pnl_cumsum_running,
                sort_global: false,
                global_bot_params: self.bot_params_master.clone(),
                hedge_mode: self.backtest_params.hedge_mode,
            },
            symbols,
            peek_hints,
        }
    }

    fn get_orchestrator_input_cached(
        &mut self,
        k: usize,
        peek_hints: Option<EntryPeekHints>,
    ) -> orchestrator::OrchestratorInput {
        // Take ownership temporarily to avoid borrow conflicts while we also read from `self`.
        let mut input = self
            .orchestrator_input_cache
            .take()
            .unwrap_or_else(|| self.build_orchestrator_input_iter(k, None, 0..self.n_coins));

        input.balance = self.balance.usd_total_balance_rounded;
        input.balance_raw = self.balance.usd_total_balance;

        let balance_raw = input.balance_raw;
        input.global.unstuck_allowance_long =
            if self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0 {
                calc_auto_unstuck_allowance(
                    balance_raw,
                    self.bot_params_master.long.unstuck_loss_allowance_pct
                        * self.bot_params_master.long.total_wallet_exposure_limit,
                    self.pnl_cumsum_max,
                    self.pnl_cumsum_running,
                )
            } else {
                0.0
            };
        input.global.unstuck_allowance_short =
            if self.bot_params_master.short.unstuck_loss_allowance_pct > 0.0 {
                calc_auto_unstuck_allowance(
                    balance_raw,
                    self.bot_params_master.short.unstuck_loss_allowance_pct
                        * self.bot_params_master.short.total_wallet_exposure_limit,
                    self.pnl_cumsum_max,
                    self.pnl_cumsum_running,
                )
            } else {
                0.0
            };
        input.global.max_realized_loss_pct = self.backtest_params.max_realized_loss_pct;
        input.global.realized_pnl_cumsum_max = self.pnl_cumsum_max;
        input.global.realized_pnl_cumsum_last = self.pnl_cumsum_running;

        input.peek_hints = peek_hints;

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
                    low,
                    high,
                    tradable: tradable_next,
                })
            } else {
                None
            };

            let exchange = &sym.exchange;
            sym.effective_min_cost =
                qty_to_cost(exchange.min_qty, close_price, exchange.c_mult).max(exchange.min_cost);

            let pos_long = *self
                .positions
                .long
                .get(&idx)
                .unwrap_or(&Position::default());
            let pos_short = *self
                .positions
                .short
                .get(&idx)
                .unwrap_or(&Position::default());
            sym.long.position = pos_long;
            sym.short.position = pos_short;

            sym.long.trailing = self
                .trailing_prices
                .long
                .get(&idx)
                .cloned()
                .unwrap_or_default();
            sym.short.trailing = self
                .trailing_prices
                .short
                .get(&idx)
                .cloned()
                .unwrap_or_default();

            // Bot params are mostly static, but `wallet_exposure_limit` may be dynamically updated
            // per timestep based on `effective_n_positions` and eligible coin count.
            sym.long.bot_params.wallet_exposure_limit =
                self.bot_params[idx].long.wallet_exposure_limit;
            sym.short.bot_params.wallet_exposure_limit =
                self.bot_params[idx].short.wallet_exposure_limit;

            let valid_now = self.coin_is_valid_at(idx, k);
            let mut mode_long: Option<orchestrator::TradingMode> = None;
            let mut mode_short: Option<orchestrator::TradingMode> = None;

            if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
                if k >= delist_timestamp {
                    if pos_long.size != 0.0 {
                        mode_long = Some(orchestrator::TradingMode::Panic);
                    }
                    if pos_short.size != 0.0 {
                        mode_short = Some(orchestrator::TradingMode::Panic);
                    }
                }
            } else {
                if !valid_now && pos_long.size != 0.0 {
                    mode_long = Some(orchestrator::TradingMode::Panic);
                }
                if !valid_now && pos_short.size != 0.0 {
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

            sym.long.mode = mode_long;
            sym.short.mode = mode_short;

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
            if sym.emas.m1.volume.len() >= 2 {
                sym.emas.m1.volume[0].1 = self.emas[idx].vol_long;
                sym.emas.m1.volume[1].1 = self.emas[idx].vol_short;
            }
            if sym.emas.m1.log_range.len() >= 2 {
                sym.emas.m1.log_range[0].1 = self.emas[idx].log_range_long;
                sym.emas.m1.log_range[1].1 = self.emas[idx].log_range_short;
            }
            match sym.emas.h1.log_range.len() {
                2 => {
                    sym.emas.h1.log_range[0].1 =
                        self.emas[idx].entry_volatility_logrange_ema_1h_long;
                    sym.emas.h1.log_range[1].1 =
                        self.emas[idx].entry_volatility_logrange_ema_1h_short;
                }
                1 => {
                    let span0 = sym.emas.h1.log_range[0].0;
                    if (span0 - self.bot_params[idx].long.entry_volatility_ema_span_hours).abs()
                        < 1e-12
                    {
                        sym.emas.h1.log_range[0].1 =
                            self.emas[idx].entry_volatility_logrange_ema_1h_long;
                    } else {
                        sym.emas.h1.log_range[0].1 =
                            self.emas[idx].entry_volatility_logrange_ema_1h_short;
                    }
                }
                _ => {}
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

    pub fn new(
        hlcvs: ArrayView3<'a, f64>,
        btc_usd_prices: ArrayView1<'a, f64>,
        bot_params: Vec<BotParamsPair>,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
        let mut balance = Balance::default();
        balance.btc_collateral_cap = backtest_params.btc_collateral_cap.max(0.0);
        balance.btc_collateral_ltv_cap = backtest_params.btc_collateral_ltv_cap;
        balance.use_btc_collateral = balance.btc_collateral_cap > 0.0;

        let starting_balance = backtest_params.starting_balance;
        let initial_btc_price = btc_usd_prices[0].max(f64::EPSILON);

        if balance.use_btc_collateral {
            let btc_value = balance.btc_collateral_cap * starting_balance;
            balance.btc_cash_wallet = btc_value / initial_btc_price;
            balance.usd_cash_wallet = starting_balance - btc_value;
        } else {
            balance.usd_cash_wallet = starting_balance;
            balance.btc_cash_wallet = 0.0;
        }
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
            let mut trade_idx = first.saturating_add(warm_bars);
            if trade_idx > last {
                trade_idx = last;
            }
            trade_start_idx[i] = trade_idx;

            let expected_trade_idx = first.saturating_add(warm_bars).min(last);
            debug_assert_eq!(
                trade_idx, expected_trade_idx,
                "trade start index mismatch for coin {}: expected {} but got {}",
                i, expected_trade_idx, trade_idx
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
                    entry_volatility_logrange_ema_1h_long: 0.0,
                    entry_volatility_logrange_ema_1h_long_num: 0.0,
                    entry_volatility_logrange_ema_1h_long_den: 1.0,
                    entry_volatility_logrange_ema_1h_short: 0.0,
                    entry_volatility_logrange_ema_1h_short_num: 0.0,
                    entry_volatility_logrange_ema_1h_short_den: 1.0,
                }
            })
            .collect();
        let equities = Equities::default();

        // init bot params
        let mut bot_params_master = bot_params[0].clone();
        bot_params_master.long.n_positions = n_coins.min(bot_params_master.long.n_positions);
        bot_params_master.short.n_positions = n_coins.min(bot_params_master.short.n_positions);

        // Store original bot params to preserve dynamic WEL indicators
        let bot_params_original = bot_params.clone();

        let effective_n_positions = EffectiveNPositions {
            long: bot_params_master.long.n_positions,
            short: bot_params_master.short.n_positions,
        };

        // Calculate EMA alphas for each coin, adjusted for candle interval
        let interval = backtest_params.candle_interval_minutes;
        let ema_alphas: Vec<EmaAlphas> = bot_params
            .iter()
            .map(|bp| calc_ema_alphas(bp, interval))
            .collect();
        let mut warmup_bars = backtest_params.global_warmup_bars;
        if warmup_bars == 0 {
            warmup_bars = calc_warmup_bars(&bot_params);
        }

        let trailing_enabled: Vec<TrailingEnabled> = bot_params
            .iter()
            .map(|bp| TrailingEnabled {
                long: bp.long.close_trailing_grid_ratio != 0.0
                    || bp.long.entry_trailing_grid_ratio != 0.0,
                short: bp.short.close_trailing_grid_ratio != 0.0
                    || bp.short.entry_trailing_grid_ratio != 0.0,
            })
            .collect();
        let any_trailing_long = trailing_enabled.iter().any(|te| te.long);
        let any_trailing_short = trailing_enabled.iter().any(|te| te.short);

        Backtest {
            hlcvs,
            btc_usd_prices,
            active_coin_indices,
            interval_ms: backtest_params.candle_interval_minutes * 60_000,
            bot_params_master: bot_params_master.clone(),
            bot_params: bot_params.clone(),
            bot_params_original,
            effective_n_positions,
            exchange_params_list,
            backtest_params: backtest_params.clone(),
            balance,
            n_coins,
            ema_alphas,
            emas: initial_emas,
            needs_volume_ema_long: bot_params
                .iter()
                .any(|bp| bp.long.filter_volume_drop_pct != 0.0),
            needs_volume_ema_short: bot_params
                .iter()
                .any(|bp| bp.short.filter_volume_drop_pct != 0.0),
            needs_log_range_long: bot_params.iter().any(|bp| {
                bp.long.entry_grid_spacing_volatility_weight != 0.0
                    || bp.long.entry_trailing_threshold_volatility_weight != 0.0
                    || bp.long.entry_trailing_retracement_volatility_weight != 0.0
                    || bp.long.entry_trailing_grid_ratio != 0.0
            }),
            needs_log_range_short: bot_params.iter().any(|bp| {
                bp.short.entry_grid_spacing_volatility_weight != 0.0
                    || bp.short.entry_trailing_threshold_volatility_weight != 0.0
                    || bp.short.entry_trailing_retracement_volatility_weight != 0.0
                    || bp.short.entry_trailing_grid_ratio != 0.0
            }),
            needs_entry_volatility_logrange_ema_1h_long: bot_params
                .iter()
                .any(|bp| bp.long.entry_volatility_ema_span_hours > 0.0),
            needs_entry_volatility_logrange_ema_1h_short: bot_params
                .iter()
                .any(|bp| bp.short.entry_volatility_ema_span_hours > 0.0),
            coin_first_valid_idx: first_valid_idx,
            coin_last_valid_idx: last_valid_idx,
            coin_trade_start_idx: trade_start_idx,
            trade_activation_logged,
            positions: Positions::default(),
            first_timestamp_ms: backtest_params.first_timestamp_ms,
            last_hour_boundary_ms: (backtest_params.first_timestamp_ms / 3_600_000) * 3_600_000,
            latest_hour: vec![HourBucket::default(); n_coins],
            warmup_bars,
            current_step: 0,
            open_orders: OpenOrders::default(),
            trailing_prices: TrailingPrices::default(),
            pnl_cumsum_running: 0.0,
            pnl_cumsum_max: 0.0,
            fills: Vec::new(),
            trading_enabled: TradingEnabled {
                long: bot_params
                    .iter()
                    .any(|bp| bp.long.wallet_exposure_limit != 0.0)
                    && bot_params_master.long.n_positions > 0,
                short: bot_params
                    .iter()
                    .any(|bp| bp.short.wallet_exposure_limit != 0.0)
                    && bot_params_master.short.n_positions > 0,
            },
            trailing_enabled,
            any_trailing_long,
            any_trailing_short,
            equities: equities,
            last_valid_timestamps: HashMap::new(),
            first_valid_timestamps: HashMap::new(),
            did_fill_long: HashSet::new(),
            did_fill_short: HashSet::new(),
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
            // EMAs already initialized in `emas`; no rolling buffers needed
        }
    }

    pub fn run(&mut self) -> (Vec<Fill>, Equities) {
        let n_timesteps = self.hlcvs.shape()[0];
        for idx in 0..self.n_coins {
            self.trailing_prices
                .long
                .insert(idx, TrailingPriceBundle::default());
            self.trailing_prices
                .short
                .insert(idx, TrailingPriceBundle::default());
        }

        // --- register first & last valid candle for every coin ---
        for idx in 0..self.n_coins {
            if let Some((start, end)) = self.coin_valid_range(idx) {
                self.first_valid_timestamps.insert(idx, start);
                if end.saturating_add(1400) < n_timesteps {
                    // add only if delisted more than one day before last timestamp
                    self.last_valid_timestamps.insert(idx, end);
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
            self.check_for_fills(k);
            self.update_emas(k);
            self.update_rounded_balance(k);
            self.update_trailing_prices(k);
            let current_ts = self.first_timestamp_ms + (k as u64) * self.interval_ms;
            if k > warmup_bars && current_ts >= guard_timestamp_ms {
                if self.update_n_positions_and_wallet_exposure_limits(k) {
                    self.equity_tracking_active = true;
                }
                self.update_open_orders_all(k);
            }
            if self.equity_tracking_active {
                self.update_equities(k);
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
        let fills = std::mem::take(&mut self.fills);
        let equities = std::mem::take(&mut self.equities);
        (fills, equities)
    }

    fn update_n_positions_and_wallet_exposure_limits(&mut self, k: usize) -> bool {
        let eligible: Vec<usize> = (0..self.n_coins)
            .filter(|&idx| self.coin_is_tradeable_at(idx, k))
            .collect();

        if eligible.is_empty() {
            return false; // nothing tradable right now
        }

        // ---------- 2. effective position counts ----------
        self.effective_n_positions.long =
            self.bot_params_master.long.n_positions.min(eligible.len());
        self.effective_n_positions.short =
            self.bot_params_master.short.n_positions.min(eligible.len());

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

        // ---------- 4. apply to every eligible coin ----------
        for &idx in &eligible {
            // long side
            if self.bot_params_original[idx].long.wallet_exposure_limit < 0.0 {
                self.bot_params[idx].long.wallet_exposure_limit = dyn_wel_long_base;
            }
            // short side
            if self.bot_params_original[idx].short.wallet_exposure_limit < 0.0 {
                self.bot_params[idx].short.wallet_exposure_limit = dyn_wel_short_base;
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

    #[inline(always)]
    fn bp(&self, coin_idx: usize, pside: usize) -> &BotParams {
        match pside {
            0 => &self.bot_params[coin_idx].long,
            1 => &self.bot_params[coin_idx].short,
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
    fn coin_is_valid_at(&self, idx: usize, k: usize) -> bool {
        self.coin_valid_range(idx)
            .map(|(start, end)| k >= start && k <= end)
            .unwrap_or(false)
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
        let min_cost = qty_to_cost(exchange.min_qty, price, exchange.c_mult).max(exchange.min_cost);
        let bot = self.bp(idx, pside);
        if bot.entry_initial_qty_pct <= 0.0 {
            return false;
        }
        let base_limit = bot.wallet_exposure_limit;
        if base_limit <= 0.0 {
            return false;
        }
        let allowance_multiplier = 1.0 + bot.risk_we_excess_allowance_pct.max(0.0);
        let effective_limit = base_limit * allowance_multiplier;
        let projected_cost =
            self.balance.usd_total_balance * effective_limit * bot.entry_initial_qty_pct;
        projected_cost >= min_cost
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

    fn update_equities(&mut self, k: usize) {
        // Start with the “running totals” in our Balance struct
        let mut equity_usd = self.balance.usd_total_balance;
        let btc_price = self.btc_usd_prices[k].max(f64::EPSILON);
        let mut equity_btc = self.balance.btc_total_balance;

        // Add the unrealized PNL of all positions
        let mut long_keys: Vec<usize> = self.positions.long.keys().cloned().collect();
        long_keys.sort();
        for idx in long_keys {
            let position = &self.positions.long[&idx];
            if !self.coin_is_valid_at(idx, k) {
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
            equity_btc += upnl / btc_price;
        }

        let mut short_keys: Vec<usize> = self.positions.short.keys().cloned().collect();
        short_keys.sort();
        for idx in short_keys {
            let position = &self.positions.short[&idx];
            if !self.coin_is_valid_at(idx, k) {
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
            equity_btc += upnl / btc_price;
        }

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
        // Deterministic summation order (HashMap iteration order is randomized per process).
        let mut long_keys: Vec<usize> = self.positions.long.keys().copied().collect();
        long_keys.sort_unstable();
        for idx in long_keys {
            let position = self.positions.long.get(&idx).expect("idx from keys");
            if position.size != 0.0 {
                twe_long += calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_balance,
                    position.size.abs(),
                    position.price,
                );
            }
        }
        let mut short_keys: Vec<usize> = self.positions.short.keys().copied().collect();
        short_keys.sort_unstable();
        for idx in short_keys {
            let position = self.positions.short.get(&idx).expect("idx from keys");
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

    fn check_for_fills(&mut self, k: usize) {
        self.did_fill_long.clear();
        self.did_fill_short.clear();
        if self.trading_enabled.long {
            // `BTreeMap` keys are already sorted.
            let open_orders_keys_long: Vec<usize> = self.open_orders.long.keys().cloned().collect();
            for idx in open_orders_keys_long {
                // Process close fills long
                if !self.open_orders.long[&idx].closes.is_empty() {
                    let mut closes_to_process = Vec::new();
                    {
                        for close_order in &self.open_orders.long[&idx].closes {
                            if self.order_filled(k, idx, close_order) {
                                closes_to_process.push(close_order.clone());
                            }
                        }
                    }
                    for order in closes_to_process {
                        //if order.qty != 0.0 && self.positions.long.contains_key(&idx) && self.positions.long.contains_key(&idx)
                        //if order.qty != 0.0 && self.get_position
                        if self.positions.long.contains_key(&idx) {
                            self.did_fill_long.insert(idx);
                            self.process_close_fill_long(k, idx, &order);
                        }
                    }
                }
                // Process entry fills long
                if !self.open_orders.long[&idx].entries.is_empty() {
                    let mut entries_to_process = Vec::new();
                    {
                        for entry_order in &self.open_orders.long[&idx].entries {
                            if self.order_filled(k, idx, entry_order) {
                                entries_to_process.push(entry_order.clone());
                            }
                        }
                    }
                    for order in entries_to_process {
                        self.did_fill_long.insert(idx);
                        self.process_entry_fill_long(k, idx, &order);
                    }
                }
            }
        }
        if self.trading_enabled.short {
            // `BTreeMap` keys are already sorted.
            let open_orders_keys_short: Vec<usize> =
                self.open_orders.short.keys().cloned().collect();
            for idx in open_orders_keys_short {
                // Process close fills short
                if !self.open_orders.short[&idx].closes.is_empty() {
                    let mut closes_to_process = Vec::new();
                    {
                        for close_order in &self.open_orders.short[&idx].closes {
                            if self.order_filled(k, idx, close_order) {
                                closes_to_process.push(close_order.clone());
                            }
                        }
                    }
                    for order in closes_to_process {
                        if self.positions.short.contains_key(&idx) {
                            self.did_fill_short.insert(idx);
                            self.process_close_fill_short(k, idx, &order);
                        }
                    }
                }
                // Process entry fills short
                if !self.open_orders.short[&idx].entries.is_empty() {
                    let mut entries_to_process = Vec::new();
                    {
                        for entry_order in &self.open_orders.short[&idx].entries {
                            if self.order_filled(k, idx, entry_order) {
                                entries_to_process.push(entry_order.clone());
                            }
                        }
                    }
                    for order in entries_to_process {
                        self.did_fill_short.insert(idx);
                        self.process_entry_fill_short(k, idx, &order);
                    }
                }
            }
        }
    }

    fn process_close_fill_long(&mut self, k: usize, idx: usize, close_fill: &Order) {
        let mut new_psize = round_(
            self.positions.long[&idx].size + close_fill.qty,
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
            adjusted_close_qty = -self.positions.long[&idx].size;
        }
        let fee_paid = -qty_to_cost(
            adjusted_close_qty,
            close_fill.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let pnl = calc_pnl_long(
            self.positions.long[&idx].price,
            close_fill.price,
            adjusted_close_qty,
            self.exchange_params_list[idx].c_mult,
        );
        self.pnl_cumsum_running += pnl;
        self.pnl_cumsum_max = self.pnl_cumsum_max.max(self.pnl_cumsum_running);
        let balance_before = self.snapshot_balance();
        self.update_balance(k, pnl, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "close_long",
            close_fill,
            adjusted_close_qty,
            close_fill.price,
            pnl,
            fee_paid,
            balance_before,
            balance_after,
        );

        let current_pprice = self.positions.long[&idx].price;
        if new_psize == 0.0 {
            self.positions.long.remove(&idx);
        } else {
            self.positions.long.get_mut(&idx).unwrap().size = new_psize;
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
            fill_price: close_fill.price,              // fill price
            position_size: new_psize,                  // psize after fill
            position_price: current_pprice,            // pprice after fill
            order_type: close_fill.order_type.clone(), // fill type
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
    }

    fn process_close_fill_short(&mut self, k: usize, idx: usize, order: &Order) {
        let mut new_psize = round_(
            self.positions.short[&idx].size + order.qty,
            self.exchange_params_list[idx].qty_step,
        );
        let mut adjusted_close_qty = order.qty;
        if new_psize > 0.0 {
            println!("warning: close qty greater than psize short");
            println!("coin: {}", self.backtest_params.coins[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", order);
            new_psize = 0.0;
            adjusted_close_qty = self.positions.short[&idx].size.abs();
        }
        let fee_paid = -qty_to_cost(
            adjusted_close_qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let pnl = calc_pnl_short(
            self.positions.short[&idx].price,
            order.price,
            adjusted_close_qty,
            self.exchange_params_list[idx].c_mult,
        );
        self.pnl_cumsum_running += pnl;
        self.pnl_cumsum_max = self.pnl_cumsum_max.max(self.pnl_cumsum_running);
        let balance_before = self.snapshot_balance();
        self.update_balance(k, pnl, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "close_short",
            order,
            adjusted_close_qty,
            order.price,
            pnl,
            fee_paid,
            balance_before,
            balance_after,
        );

        let current_pprice = self.positions.short[&idx].price;
        if new_psize == 0.0 {
            self.positions.short.remove(&idx);
        } else {
            self.positions.short.get_mut(&idx).unwrap().size = new_psize;
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
            fill_price: order.price,
            position_size: new_psize,
            position_price: current_pprice,
            order_type: order.order_type.clone(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
    }

    fn process_entry_fill_long(&mut self, k: usize, idx: usize, order: &Order) {
        // long entry fill
        let fee_paid = -qty_to_cost(
            order.qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let balance_before = self.snapshot_balance();
        self.update_balance(k, 0.0, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "entry_long",
            order,
            order.qty,
            order.price,
            0.0,
            fee_paid,
            balance_before,
            balance_after,
        );

        let position_entry = self
            .positions
            .long
            .entry(idx)
            .or_insert(Position::default());
        let (new_psize, new_pprice) = calc_new_psize_pprice(
            position_entry.size,
            position_entry.price,
            order.qty,
            order.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.long.get_mut(&idx).unwrap().size = new_psize;
        self.positions.long.get_mut(&idx).unwrap().price = new_pprice;
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
            fill_price: order.price,
            position_size: self.positions.long[&idx].size,
            position_price: self.positions.long[&idx].price,
            order_type: order.order_type.clone(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
    }

    fn process_entry_fill_short(&mut self, k: usize, idx: usize, order: &Order) {
        // short entry fill
        let fee_paid = -qty_to_cost(
            order.qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let balance_before = self.snapshot_balance();
        self.update_balance(k, 0.0, fee_paid);
        let balance_after = self.snapshot_balance();
        self.record_balance_trace(
            k,
            idx,
            "entry_short",
            order,
            order.qty,
            order.price,
            0.0,
            fee_paid,
            balance_before,
            balance_after,
        );
        let position_entry = self
            .positions
            .short
            .entry(idx)
            .or_insert(Position::default());
        let (new_psize, new_pprice) = calc_new_psize_pprice(
            position_entry.size,
            position_entry.price,
            order.qty,
            order.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.short.get_mut(&idx).unwrap().size = new_psize;
        self.positions.short.get_mut(&idx).unwrap().price = new_pprice;
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
            fill_price: order.price,
            position_size: self.positions.short[&idx].size,
            position_price: self.positions.short[&idx].price,
            order_type: order.order_type.clone(),
            wallet_exposure,
            twe_long,
            twe_short,
            twe_net,
        });
    }

    fn update_trailing_prices(&mut self, k: usize) {
        // ----- LONG side -----
        if self.trading_enabled.long && self.any_trailing_long {
            for (&idx, _) in &self.positions.long {
                if !self.trailing_enabled[idx].long {
                    continue;
                }
                if !self.coin_is_valid_at(idx, k) {
                    continue;
                }
                let fill_long = self.did_fill_long.contains(&idx);
                let col = self.active_coin_indices[idx];
                let low = self.hlcvs[[k, col, LOW]];
                let high = self.hlcvs[[k, col, HIGH]];
                let close = self.hlcvs[[k, col, CLOSE]];
                let bundle = self.trailing_prices.long.entry(idx).or_default();
                if fill_long {
                    reset_trailing_bundle(bundle);
                } else {
                    update_trailing_bundle_with_candle(bundle, high, low, close);
                }
            }
        }

        // ----- SHORT side -----
        if self.trading_enabled.short && self.any_trailing_short {
            for (&idx, _) in &self.positions.short {
                if !self.trailing_enabled[idx].short {
                    continue;
                }
                if !self.coin_is_valid_at(idx, k) {
                    continue;
                }
                let fill_short = self.did_fill_short.contains(&idx);
                let col = self.col(idx);
                let low = self.hlcvs[[k, col, LOW]];
                let high = self.hlcvs[[k, col, HIGH]];
                let close = self.hlcvs[[k, col, CLOSE]];
                let bundle = self.trailing_prices.short.entry(idx).or_default();
                if fill_short {
                    reset_trailing_bundle(bundle);
                } else {
                    update_trailing_bundle_with_candle(bundle, high, low, close);
                }
            }
        }
    }

    fn order_filled(&self, k: usize, idx: usize, order: &Order) -> bool {
        if !self.coin_is_tradeable_at(idx, k) {
            return false;
        }
        // check if filled in current candle (pass k+1 to check if will fill in next candle)
        if order.qty > 0.0 {
            self.hlcvs_value(k, idx, LOW) < order.price
        } else if order.qty < 0.0 {
            self.hlcvs_value(k, idx, HIGH) > order.price
        } else {
            false
        }
    }

    fn update_open_orders_all(&mut self, k: usize) {
        self.update_open_orders_all_orchestrator(k);
    }

    fn update_open_orders_all_orchestrator(&mut self, k: usize) {
        let total_t0 = Instant::now();
        if let Some(p) = self.orch_profile.as_mut() {
            p.steps = p.steps.saturating_add(1);
        }

        let t0 = Instant::now();
        self.open_orders.long.clear();
        self.open_orders.short.clear();
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
        for (&idx, _pos) in self.positions.long.iter() {
            self.debug_dump_unstuck_calc(k, idx, LONG);
        }
        for (&idx, _pos) in self.positions.short.iter() {
            self.debug_dump_unstuck_calc(k, idx, SHORT);
        }

        let (res, input_update_elapsed, compute_elapsed) = {
            let t0 = Instant::now();
            let input = self.get_orchestrator_input_cached(k, peek_hints);
            let input_update_elapsed = t0.elapsed();

            let t1 = Instant::now();
            let res = orchestrator::compute_ideal_orders_with_workspace(
                &input,
                &mut self.orchestrator_workspace,
            )
            .unwrap_or_else(|e| panic!("orchestrator error at k {}: {:?}", k, e));
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
            match o.pside {
                orchestrator::PositionSide::Long => {
                    let bundle = self.open_orders.long.entry(o.symbol_idx).or_default();
                    if orchestrator::is_close_order_type(order.order_type) {
                        bundle.closes.push(order);
                    } else {
                        bundle.entries.push(order);
                    }
                }
                orchestrator::PositionSide::Short => {
                    let bundle = self.open_orders.short.entry(o.symbol_idx).or_default();
                    if orchestrator::is_close_order_type(order.order_type) {
                        bundle.closes.push(order);
                    } else {
                        bundle.entries.push(order);
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

        for (&idx, bundle) in self.open_orders.long.iter() {
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
            let position = self.positions.long.get(&idx).copied().unwrap_or_default();
            let close_price = self.hlcvs_value(k, idx, CLOSE);
            let mut entries = Vec::with_capacity(bundle.entries.len());
            for o in &bundle.entries {
                entries.push(DebugOrder {
                    qty: o.qty,
                    price: o.price,
                    order_type_id: o.order_type.id(),
                    reduce_only: false,
                });
            }
            let mut closes = Vec::with_capacity(bundle.closes.len());
            for o in &bundle.closes {
                closes.push(DebugOrder {
                    qty: o.qty,
                    price: o.price,
                    order_type_id: o.order_type.id(),
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
        for (&idx, bundle) in self.open_orders.short.iter() {
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
            let position = self.positions.short.get(&idx).copied().unwrap_or_default();
            let close_price = self.hlcvs_value(k, idx, CLOSE);
            let mut entries = Vec::with_capacity(bundle.entries.len());
            for o in &bundle.entries {
                entries.push(DebugOrder {
                    qty: o.qty,
                    price: o.price,
                    order_type_id: o.order_type.id(),
                    reduce_only: false,
                });
            }
            let mut closes = Vec::with_capacity(bundle.closes.len());
            for o in &bundle.closes {
                closes.push(DebugOrder {
                    qty: o.qty,
                    price: o.price,
                    order_type_id: o.order_type.id(),
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

    #[inline]
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
            if self.needs_entry_volatility_logrange_ema_1h_long
                || self.needs_entry_volatility_logrange_ema_1h_short
            {
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
                    let alpha_long = self.ema_alphas[i].entry_volatility_logrange_ema_1h_alpha_long;
                    let alpha_short =
                        self.ema_alphas[i].entry_volatility_logrange_ema_1h_alpha_short;
                    let emas = &mut self.emas[i];
                    if self.needs_entry_volatility_logrange_ema_1h_long && alpha_long > 0.0 {
                        emas.entry_volatility_logrange_ema_1h_long = update_adjusted_ema(
                            hour_log_range,
                            alpha_long,
                            &mut emas.entry_volatility_logrange_ema_1h_long_num,
                            &mut emas.entry_volatility_logrange_ema_1h_long_den,
                        );
                    }
                    if self.needs_entry_volatility_logrange_ema_1h_short && alpha_short > 0.0 {
                        emas.entry_volatility_logrange_ema_1h_short = update_adjusted_ema(
                            hour_log_range,
                            alpha_short,
                            &mut emas.entry_volatility_logrange_ema_1h_short_num,
                            &mut emas.entry_volatility_logrange_ema_1h_short_den,
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
            if self.needs_log_range_long || self.needs_log_range_short {
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
            }
        }
    }

    pub fn initial_entry_balance_pct(&self) -> (f64, f64) {
        let long = calc_entry_balance_pct(
            &self.bot_params_master.long,
            self.effective_n_positions.long,
        );
        let short = calc_entry_balance_pct(
            &self.bot_params_master.short,
            self.effective_n_positions.short,
        );
        (long, short)
    }
}

fn calc_ema_alphas(bot_params_pair: &BotParamsPair, interval: u64) -> EmaAlphas {
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
    let mut ema_spans_long = [
        bot_params_pair.long.ema_span_0,
        bot_params_pair.long.ema_span_1,
        (bot_params_pair.long.ema_span_0 * bot_params_pair.long.ema_span_1).sqrt(),
    ];
    ema_spans_long.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ema_spans_short = [
        bot_params_pair.short.ema_span_0,
        bot_params_pair.short.ema_span_1,
        (bot_params_pair.short.ema_span_0 * bot_params_pair.short.ema_span_1).sqrt(),
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
        // EMA spans for the volume/log range filters (alphas precomputed from spans)
        vol_alpha_long: clamp_alpha(
            2.0 / (bot_params_pair.long.filter_volume_ema_span as f64 / interval_f + 1.0),
        ),
        vol_alpha_short: clamp_alpha(
            2.0 / (bot_params_pair.short.filter_volume_ema_span as f64 / interval_f + 1.0),
        ),
        log_range_alpha_long: clamp_alpha(
            2.0 / (bot_params_pair.long.filter_volatility_ema_span as f64 / interval_f + 1.0),
        ),
        log_range_alpha_short: clamp_alpha(
            2.0 / (bot_params_pair.short.filter_volatility_ema_span as f64 / interval_f + 1.0),
        ),
        // Note: entry_volatility spans are in HOURS and computed from hourly buckets,
        // so they do NOT need interval adjustment (hourly buckets are calendar-based)
        entry_volatility_logrange_ema_1h_alpha_long: {
            let span = bot_params_pair.long.entry_volatility_ema_span_hours;
            if span > 0.0 {
                2.0 / (span + 1.0)
            } else {
                0.0
            }
        },
        entry_volatility_logrange_ema_1h_alpha_short: {
            let span = bot_params_pair.short.entry_volatility_ema_span_hours;
            if span > 0.0 {
                2.0 / (span + 1.0)
            } else {
                0.0
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

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
            filter_by_min_effective_cost: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            candle_interval_minutes: 1,
        };

        let mut bt = Backtest::new(
            hlcvs.view(),
            btc_usd_prices.view(),
            vec![bp_pair],
            vec![ExchangeParams::default()],
            &backtest_params,
        );

        let input = bt.get_orchestrator_input_cached(1, None);
        assert!(
            (input.symbols[0].long.bot_params.wallet_exposure_limit - 0.1).abs() < 1e-12,
            "expected cached input WEL to match initial bot_params"
        );
        bt.orchestrator_input_cache = Some(input);

        bt.bot_params[0].long.wallet_exposure_limit = 0.2;

        let input = bt.get_orchestrator_input_cached(1, None);
        assert!(
            (input.symbols[0].long.bot_params.wallet_exposure_limit - 0.2).abs() < 1e-12,
            "expected cached input WEL to update after bot_params change"
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
            filter_by_min_effective_cost: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
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

        let input = bt.get_orchestrator_input_cached(1, None);
        assert!(
            (input.balance - 100.0).abs() < 1e-12,
            "expected snapped balance to route to input.balance"
        );
        assert!(
            (input.balance_raw - 200.0).abs() < 1e-12,
            "expected raw balance to route to input.balance_raw"
        );

        let allowance_pct = 0.2 * 0.5;
        let expected_from_raw =
            calc_auto_unstuck_allowance(200.0, allowance_pct, bt.pnl_cumsum_max, bt.pnl_cumsum_running);
        let expected_from_snapped =
            calc_auto_unstuck_allowance(100.0, allowance_pct, bt.pnl_cumsum_max, bt.pnl_cumsum_running);
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
            filter_by_min_effective_cost: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
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

        let input1 = bt.get_orchestrator_input_cached(1, None);
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
        bt.balance.usd_total_balance = 1050.0;  // raw changed (profit fill)
        bt.balance.usd_total_balance_rounded = 1000.0;  // snapped stays (hysteresis)
        bt.pnl_cumsum_max = 50.0;
        bt.pnl_cumsum_running = 50.0;

        let input2 = bt.get_orchestrator_input_cached(1, None);
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
        let expected_allowance = calc_auto_unstuck_allowance(
            1050.0, allowance_pct, 50.0, 50.0,
        );
        assert!(
            (input2.global.unstuck_allowance_long - expected_allowance).abs() < 1e-12,
            "unstuck allowance should use updated raw balance (1050), got {}",
            input2.global.unstuck_allowance_long
        );
    }

    #[test]
    fn test_ema_alpha_interval_1_matches_original_formula() {
        // With interval=1, alpha should equal 2/(span+1) (the original formula)
        let mut bp = BotParamsPair::default();
        bp.long.ema_span_0 = 100.0;
        bp.long.ema_span_1 = 200.0;
        bp.short.ema_span_0 = 50.0;
        bp.short.ema_span_1 = 150.0;
        bp.long.filter_volume_ema_span = 300.0;
        bp.short.filter_volume_ema_span = 400.0;
        bp.long.filter_volatility_ema_span = 500.0;
        bp.short.filter_volatility_ema_span = 600.0;

        let alphas = calc_ema_alphas(&bp, 1);

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

        let alphas = calc_ema_alphas(&bp, 5);

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
    fn test_ema_alpha_hourly_volatility_not_adjusted() {
        // entry_volatility spans are in hours and calendar-based; should NOT change with interval
        let mut bp = BotParamsPair::default();
        bp.long.entry_volatility_ema_span_hours = 24.0;
        bp.short.entry_volatility_ema_span_hours = 48.0;

        let alphas_1 = calc_ema_alphas(&bp, 1);
        let alphas_5 = calc_ema_alphas(&bp, 5);

        assert!(
            (alphas_1.entry_volatility_logrange_ema_1h_alpha_long
                - alphas_5.entry_volatility_logrange_ema_1h_alpha_long)
                .abs()
                < 1e-12,
            "hourly volatility alpha should not change with interval"
        );
        assert!(
            (alphas_1.entry_volatility_logrange_ema_1h_alpha_short
                - alphas_5.entry_volatility_logrange_ema_1h_alpha_short)
                .abs()
                < 1e-12,
            "hourly volatility alpha should not change with interval"
        );
    }
}

fn calc_warmup_bars(bot_params: &[BotParamsPair]) -> usize {
    let mut max_span_minutes = 0.0f64;

    for pair in bot_params {
        let spans_long = [
            pair.long.ema_span_0,
            pair.long.ema_span_1,
            pair.long.filter_volume_ema_span as f64,
            pair.long.filter_volatility_ema_span as f64,
            pair.long.entry_volatility_ema_span_hours * 60.0,
        ];
        let spans_short = [
            pair.short.ema_span_0,
            pair.short.ema_span_1,
            pair.short.filter_volume_ema_span as f64,
            pair.short.filter_volatility_ema_span as f64,
            pair.short.entry_volatility_ema_span_hours * 60.0,
        ];
        for span in spans_long.iter().chain(spans_short.iter()) {
            if span.is_finite() {
                max_span_minutes = max_span_minutes.max(*span);
            }
        }
    }

    max_span_minutes.ceil() as usize
}
