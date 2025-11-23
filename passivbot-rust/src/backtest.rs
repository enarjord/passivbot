use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::coin_selection::{select_coins, CoinFeature, SelectionConfig};
use crate::constants::{CLOSE, HIGH, LONG, LOW, NO_POS, SHORT, VOLUME};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_next_entry_long, calc_next_entry_short,
};
use crate::risk::{
    calc_twel_enforcer_actions, calc_unstucking_action, gate_entries_by_twel, GateEntriesCandidate,
    GateEntriesPosition, TwelEnforcerInputPosition, UnstuckPositionInput,
};
use crate::trailing::{reset_trailing_bundle, update_trailing_bundle_with_candle};
use crate::types::{
    BacktestParams, Balance, BotParams, BotParamsPair, EMABands, Equities, ExchangeParams, Fill,
    Order, OrderBook, OrderType, Position, Positions, StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_wallet_exposure, hysteresis, qty_to_cost, round_, round_dn,
};
use ndarray::{ArrayView1, ArrayView3};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Default, Copy, Debug)]
pub struct EmaAlphas {
    pub long: Alphas,
    pub short: Alphas,
    pub vol_alpha_long: f64,
    pub vol_alpha_short: f64,
    pub log_range_alpha_long: f64,
    pub log_range_alpha_short: f64,
    pub grid_log_range_alpha_long: f64,
    pub grid_log_range_alpha_short: f64,
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
    pub grid_log_range_long: f64,
    pub grid_log_range_long_num: f64,
    pub grid_log_range_long_den: f64,
    pub grid_log_range_short: f64,
    pub grid_log_range_short_num: f64,
    pub grid_log_range_short_den: f64,
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
pub struct Actives {
    long: HashSet<usize>,
    short: HashSet<usize>,
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
    needs_grid_log_range_long: bool,
    needs_grid_log_range_short: bool,
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
    actives: Actives,
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
            let mut trade_idx = first.saturating_add(warm);
            if trade_idx > last {
                trade_idx = last;
            }
            trade_start_idx[i] = trade_idx;

            let expected_trade_idx = first.saturating_add(warm).min(last);
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
                EMAs {
                    long: [base_close; 3],
                    long_num: [base_close; 3],
                    long_den: [1.0; 3],
                    short: [base_close; 3],
                    short_num: [base_close; 3],
                    short_den: [1.0; 3],
                    vol_long: base_volume,
                    vol_long_num: base_volume,
                    vol_long_den: 1.0,
                    vol_short: base_volume,
                    vol_short_num: base_volume,
                    vol_short_den: 1.0,
                    log_range_long: 0.0,
                    log_range_long_num: 0.0,
                    log_range_long_den: 1.0,
                    log_range_short: 0.0,
                    log_range_short_num: 0.0,
                    log_range_short_den: 1.0,
                    grid_log_range_long: 0.0,
                    grid_log_range_long_num: 0.0,
                    grid_log_range_long_den: 1.0,
                    grid_log_range_short: 0.0,
                    grid_log_range_short_num: 0.0,
                    grid_log_range_short_den: 1.0,
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

        // Calculate EMA alphas for each coin
        let ema_alphas: Vec<EmaAlphas> = bot_params.iter().map(|bp| calc_ema_alphas(bp)).collect();
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
            needs_grid_log_range_long: bot_params
                .iter()
                .any(|bp| bp.long.entry_volatility_ema_span_hours > 0.0),
            needs_grid_log_range_short: bot_params
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
            actives: Actives::default(),
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
            let current_ts = self.first_timestamp_ms + (k as u64) * 60_000u64;
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

    pub fn calc_preferred_coins(&mut self, pside: usize) -> Vec<usize> {
        let max_positions = match pside {
            LONG => self.effective_n_positions.long,
            SHORT => self.effective_n_positions.short,
            _ => 0,
        };
        if max_positions == 0 {
            return Vec::new();
        }

        if self.n_coins <= max_positions && !self.backtest_params.filter_by_min_effective_cost {
            return (0..self.n_coins).collect();
        }

        let volume_drop_pct = match pside {
            LONG => self.bot_params_master.long.filter_volume_drop_pct,
            SHORT => self.bot_params_master.short.filter_volume_drop_pct,
            _ => 0.0,
        };
        let volatility_drop_pct = match pside {
            LONG => self.bot_params_master.long.filter_volatility_drop_pct,
            SHORT => self.bot_params_master.short.filter_volatility_drop_pct,
            _ => 0.0,
        };

        let features: Vec<CoinFeature> = (0..self.n_coins)
            .map(|idx| CoinFeature {
                index: idx,
                enabled: self.coin_passes_min_effective_cost(idx, pside),
                volume_score: match pside {
                    LONG => self.emas[idx].vol_long,
                    SHORT => self.emas[idx].vol_short,
                    _ => 0.0,
                },
                volatility_score: match pside {
                    LONG => self.emas[idx].log_range_long,
                    SHORT => self.emas[idx].log_range_short,
                    _ => 0.0,
                },
            })
            .collect();

        let config = SelectionConfig {
            max_positions,
            volume_drop_pct,
            volatility_drop_pct,
            require_forager: true,
        };

        select_coins(&features, &config)
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

    fn create_state_params(&self, k: usize, idx: usize, pside: usize) -> StateParams {
        let mut close_price = self.hlcvs_value(k, idx, CLOSE);
        if !close_price.is_finite() {
            close_price = 0.0;
        }
        StateParams {
            balance: self.balance.usd_total_balance_rounded,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: self.emas[idx].compute_bands(pside),
            grid_log_range: match pside {
                LONG => self.emas[idx].grid_log_range_long,
                SHORT => self.emas[idx].grid_log_range_short,
                _ => 0.0,
            },
        }
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
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * 60_000;
        self.equities.usd_total_equity.push(equity_usd);
        self.equities.btc_total_equity.push(equity_btc);
        self.equities.timestamps_ms.push(timestamp_ms);
    }

    fn record_total_wallet_exposure(&mut self) {
        let total_wallet_exposure = self.compute_total_wallet_exposure();
        self.total_wallet_exposures.push(total_wallet_exposure);
    }

    fn compute_total_wallet_exposure(&self) -> f64 {
        let mut total = 0.0;
        for (&idx, position) in &self.positions.long {
            if position.size != 0.0 {
                total += calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_balance,
                    position.size.abs(),
                    position.price,
                );
            }
        }
        for (&idx, position) in &self.positions.short {
            if position.size != 0.0 {
                total += calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_balance,
                    position.size.abs(),
                    position.price,
                );
            }
        }
        total
    }

    fn update_actives_long(&mut self) -> Vec<usize> {
        let n_positions = self.effective_n_positions.long;

        let mut current_positions: Vec<usize> = self.positions.long.keys().cloned().collect();
        current_positions.sort();
        let preferred_coins = if current_positions.len() < n_positions {
            self.calc_preferred_coins(LONG)
        } else {
            Vec::new()
        };

        let actives = &mut self.actives.long;
        actives.clear();

        for &idx in &current_positions {
            actives.insert(idx);
        }

        let mut actives_without_pos = Vec::new();
        for &idx in &preferred_coins {
            if actives.len() >= n_positions {
                break;
            }
            if actives.insert(idx) {
                actives_without_pos.push(idx);
            }
        }

        actives_without_pos
    }

    fn update_actives_short(&mut self) -> Vec<usize> {
        let n_positions = self.effective_n_positions.short;

        let mut current_positions: Vec<usize> = self.positions.short.keys().cloned().collect();
        current_positions.sort();

        let preferred_coins = if current_positions.len() < n_positions {
            self.calc_preferred_coins(SHORT)
        } else {
            Vec::new()
        };

        let actives = &mut self.actives.short;
        actives.clear();

        for &idx in &current_positions {
            actives.insert(idx);
        }

        let mut actives_without_pos = Vec::new();
        for &idx in &preferred_coins {
            if actives.len() >= n_positions {
                break;
            }
            if actives.insert(idx) {
                actives_without_pos.push(idx);
            }
        }

        actives_without_pos
    }

    fn check_for_fills(&mut self, k: usize) {
        self.did_fill_long.clear();
        self.did_fill_short.clear();
        if self.trading_enabled.long {
            let mut open_orders_keys_long: Vec<usize> =
                self.open_orders.long.keys().cloned().collect();
            open_orders_keys_long.sort();
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
            let mut open_orders_keys_short: Vec<usize> =
                self.open_orders.short.keys().cloned().collect();
            open_orders_keys_short.sort();
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
        self.update_balance(k, pnl, fee_paid);

        let current_pprice = self.positions.long[&idx].price;
        if new_psize == 0.0 {
            self.positions.long.remove(&idx);
        } else {
            self.positions.long.get_mut(&idx).unwrap().size = new_psize;
        }
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * 60_000;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                current_pprice,
            )
        } else {
            0.0
        };
        let total_wallet_exposure = self.compute_total_wallet_exposure();
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
            total_wallet_exposure,
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
        self.update_balance(k, pnl, fee_paid);

        let current_pprice = self.positions.short[&idx].price;
        if new_psize == 0.0 {
            self.positions.short.remove(&idx);
        } else {
            self.positions.short.get_mut(&idx).unwrap().size = new_psize;
        }
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * 60_000;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                current_pprice,
            )
        } else {
            0.0
        };
        let total_wallet_exposure = self.compute_total_wallet_exposure();
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
            total_wallet_exposure,
        });
    }

    fn process_entry_fill_long(&mut self, k: usize, idx: usize, order: &Order) {
        // long entry fill
        let fee_paid = -qty_to_cost(
            order.qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        self.update_balance(k, 0.0, fee_paid);

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
        let timestamp_ms = self.first_timestamp_ms + (k as u64) * 60_000;
        let wallet_exposure = if new_psize != 0.0 {
            calc_wallet_exposure(
                self.exchange_params_list[idx].c_mult,
                self.balance.usd_total_balance,
                new_psize.abs(),
                new_pprice,
            )
        } else {
            0.0
        };
        let total_wallet_exposure = self.compute_total_wallet_exposure();
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
            total_wallet_exposure,
        });
    }

    fn process_entry_fill_short(&mut self, k: usize, idx: usize, order: &Order) {
        // short entry fill
        let fee_paid = -qty_to_cost(
            order.qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        self.update_balance(k, 0.0, fee_paid);
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
            )
        } else {
            0.0
        };
        let total_wallet_exposure = self.compute_total_wallet_exposure();
        self.fills.push(Fill {
            index: k,
            timestamp_ms: self.first_timestamp_ms + (k as u64) * 60_000,
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
            total_wallet_exposure,
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

    fn update_open_orders_long_single(&mut self, k: usize, idx: usize) {
        if !self.coin_is_valid_at(idx, k) {
            return;
        }
        let state_params = self.create_state_params(k, idx, LONG);
        let position = self
            .positions
            .long
            .get(&idx)
            .cloned()
            .unwrap_or(Position::default());

        // check if coin is delisted; if so, close pos as unstuck close
        if self.positions.long.contains_key(&idx) {
            if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
                if k >= delist_timestamp {
                    self.open_orders.long.entry(idx).or_default().closes = vec![Order {
                        qty: -self.positions.long[&idx].size,
                        price: round_(
                            f64::min(
                                self.hlcvs_value(k, idx, HIGH)
                                    - self.exchange_params_list[idx].price_step,
                                self.positions.long[&idx].price,
                            ),
                            self.exchange_params_list[idx].price_step,
                        ),
                        order_type: OrderType::CloseUnstuckLong,
                    }];
                    self.open_orders
                        .long
                        .entry(idx)
                        .or_default()
                        .entries
                        .clear();
                    return;
                }
            }
        }

        let next_entry_order = calc_next_entry_long(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, LONG),
            &position,
            &self.trailing_prices.long[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_entry_order) {
            self.open_orders.long.entry(idx).or_default().entries = calc_entries_long(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, LONG),
                &position,
                &self.trailing_prices.long[&idx],
            );
        } else {
            self.open_orders.long.entry(idx).or_default().entries = [next_entry_order].to_vec();
        }
        let next_close_order = calc_next_close_long(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, LONG),
            &position,
            &self.trailing_prices.long[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_close_order) {
            // calc all orders
            self.open_orders.long.entry(idx).or_default().closes = calc_closes_long(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, LONG),
                &position,
                &self.trailing_prices.long[&idx],
            );
        } else {
            self.open_orders.long.entry(idx).or_default().closes = [next_close_order].to_vec();
        }
    }

    fn update_open_orders_short_single(&mut self, k: usize, idx: usize) {
        if !self.coin_is_valid_at(idx, k) {
            return;
        }
        let state_params = self.create_state_params(k, idx, SHORT);
        let position = self
            .positions
            .short
            .get(&idx)
            .cloned()
            .unwrap_or(Position::default());

        // check if coin is delisted; if so, close pos as unstuck close
        if self.positions.short.contains_key(&idx) {
            if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
                if k >= delist_timestamp {
                    self.open_orders.short.entry(idx).or_default().closes = vec![Order {
                        qty: self.positions.short[&idx].size.abs(),
                        price: round_(
                            f64::max(
                                self.hlcvs_value(k, idx, LOW)
                                    + self.exchange_params_list[idx].price_step,
                                self.positions.short[&idx].price,
                            ),
                            self.exchange_params_list[idx].price_step,
                        ),
                        order_type: OrderType::CloseUnstuckShort,
                    }];
                    self.open_orders
                        .short
                        .entry(idx)
                        .or_default()
                        .entries
                        .clear();
                    return;
                }
            }
        }
        let next_entry_order = calc_next_entry_short(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, SHORT),
            &position,
            &self.trailing_prices.short[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_entry_order) {
            self.open_orders.short.entry(idx).or_default().entries = calc_entries_short(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, SHORT),
                &position,
                &self.trailing_prices.short[&idx],
            );
        } else {
            self.open_orders.short.entry(idx).or_default().entries = [next_entry_order].to_vec();
        }

        let next_close_order = calc_next_close_short(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, SHORT),
            &position,
            &self.trailing_prices.short[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_close_order) {
            self.open_orders.short.entry(idx).or_default().closes = calc_closes_short(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, SHORT),
                &position,
                &self.trailing_prices.short[&idx],
            );
        } else {
            self.open_orders.short.entry(idx).or_default().closes = [next_close_order].to_vec()
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

    fn calc_unstucking_close(&mut self, k: usize) -> (usize, usize, Order) {
        let balance = self.balance.usd_total_balance_rounded;
        if balance <= 0.0 {
            return (NO_POS, NO_POS, Order::default());
        }

        let long_allowance = if self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0 {
            calc_auto_unstuck_allowance(
                balance,
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
                balance,
                self.bot_params_master.short.unstuck_loss_allowance_pct
                    * self.bot_params_master.short.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            )
        } else {
            0.0
        };

        if long_allowance <= 0.0 && short_allowance <= 0.0 {
            return (NO_POS, NO_POS, Order::default());
        }

        let mut inputs: Vec<UnstuckPositionInput> = Vec::new();

        if long_allowance > 0.0 {
            let mut keys: Vec<usize> = self.positions.long.keys().cloned().collect();
            keys.sort_unstable();
            for idx in keys {
                if !self.coin_is_tradeable_at(idx, k) {
                    continue;
                }
                if let Some(position) = self.positions.long.get(&idx) {
                    let ema_bands = self.emas[idx].compute_bands(LONG);
                    let current_price = self.hlcvs_value(k, idx, CLOSE);
                    inputs.push(UnstuckPositionInput {
                        idx,
                        side: LONG,
                        position_size: position.size,
                        position_price: position.price,
                        wallet_exposure_limit: self.bp(idx, LONG).wallet_exposure_limit,
                        risk_we_excess_allowance_pct: self
                            .bp(idx, LONG)
                            .risk_we_excess_allowance_pct,
                        unstuck_threshold: self.bp(idx, LONG).unstuck_threshold,
                        unstuck_close_pct: self.bp(idx, LONG).unstuck_close_pct,
                        unstuck_ema_dist: self.bp(idx, LONG).unstuck_ema_dist,
                        ema_band_upper: ema_bands.upper,
                        ema_band_lower: ema_bands.lower,
                        current_price,
                        price_step: self.exchange_params_list[idx].price_step,
                        qty_step: self.exchange_params_list[idx].qty_step,
                        min_qty: self.exchange_params_list[idx].min_qty,
                        min_cost: self.exchange_params_list[idx].min_cost,
                        c_mult: self.exchange_params_list[idx].c_mult,
                    });
                }
            }
        }

        if short_allowance > 0.0 {
            let mut keys: Vec<usize> = self.positions.short.keys().cloned().collect();
            keys.sort_unstable();
            for idx in keys {
                if !self.coin_is_tradeable_at(idx, k) {
                    continue;
                }
                if let Some(position) = self.positions.short.get(&idx) {
                    let ema_bands = self.emas[idx].compute_bands(SHORT);
                    let current_price = self.hlcvs_value(k, idx, CLOSE);
                    inputs.push(UnstuckPositionInput {
                        idx,
                        side: SHORT,
                        position_size: position.size,
                        position_price: position.price,
                        wallet_exposure_limit: self.bp(idx, SHORT).wallet_exposure_limit,
                        risk_we_excess_allowance_pct: self
                            .bp(idx, SHORT)
                            .risk_we_excess_allowance_pct,
                        unstuck_threshold: self.bp(idx, SHORT).unstuck_threshold,
                        unstuck_close_pct: self.bp(idx, SHORT).unstuck_close_pct,
                        unstuck_ema_dist: self.bp(idx, SHORT).unstuck_ema_dist,
                        ema_band_upper: ema_bands.upper,
                        ema_band_lower: ema_bands.lower,
                        current_price,
                        price_step: self.exchange_params_list[idx].price_step,
                        qty_step: self.exchange_params_list[idx].qty_step,
                        min_qty: self.exchange_params_list[idx].min_qty,
                        min_cost: self.exchange_params_list[idx].min_cost,
                        c_mult: self.exchange_params_list[idx].c_mult,
                    });
                }
            }
        }

        if let Some((idx, side, order)) =
            calc_unstucking_action(balance, long_allowance, short_allowance, &inputs)
        {
            return (idx, side, order);
        }

        (NO_POS, NO_POS, Order::default())
    }
    fn calc_twel_enforcer_orders(
        &self,
        k: usize,
        skip: Option<(usize, usize)>,
    ) -> Vec<(usize, usize, Order)> {
        let mut results: Vec<(usize, usize, Order)> = Vec::new();
        let balance = self.balance.usd_total_balance_rounded;
        if balance <= 0.0 {
            return results;
        }

        let skip_long = skip.filter(|(_, side)| *side == LONG).map(|(idx, _)| idx);
        let skip_short = skip.filter(|(_, side)| *side == SHORT).map(|(idx, _)| idx);

        let wel_blocked_long: HashSet<usize> = self
            .open_orders
            .long
            .iter()
            .filter_map(|(&idx, orders)| {
                if orders
                    .closes
                    .iter()
                    .any(|o| o.order_type == OrderType::CloseAutoReduceWelLong)
                {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let wel_blocked_short: HashSet<usize> = self
            .open_orders
            .short
            .iter()
            .filter_map(|(&idx, orders)| {
                if orders
                    .closes
                    .iter()
                    .any(|o| o.order_type == OrderType::CloseAutoReduceWelShort)
                {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        let long_threshold = self.bot_params_master.long.risk_twel_enforcer_threshold;
        if long_threshold >= 0.0 {
            let total_wel_long = self
                .bot_params_master
                .long
                .total_wallet_exposure_limit
                .max(0.0);
            if total_wel_long > 0.0 {
                let mut inputs: Vec<TwelEnforcerInputPosition> =
                    Vec::with_capacity(self.positions.long.len());
                let mut indices: Vec<usize> = self.positions.long.keys().cloned().collect();
                indices.sort_unstable();
                for idx in indices {
                    if wel_blocked_long.contains(&idx) {
                        continue;
                    }
                    if let Some(position) = self.positions.long.get(&idx) {
                        let market_price = self.hlcvs_value(k, idx, CLOSE);
                        inputs.push(TwelEnforcerInputPosition {
                            idx,
                            position_size: position.size,
                            position_price: position.price,
                            market_price,
                            base_wallet_exposure_limit: self.bp(idx, LONG).wallet_exposure_limit,
                            c_mult: self.exchange_params_list[idx].c_mult,
                            qty_step: self.exchange_params_list[idx].qty_step,
                            price_step: self.exchange_params_list[idx].price_step,
                            min_qty: self.exchange_params_list[idx].min_qty,
                        });
                    }
                }
                let actions = calc_twel_enforcer_actions(
                    LONG,
                    long_threshold,
                    total_wel_long,
                    self.effective_n_positions.long.max(1),
                    balance,
                    &inputs,
                    skip_long,
                );
                for (idx, order) in actions {
                    results.push((idx, LONG, order));
                }
            }
        }

        let short_threshold = self.bot_params_master.short.risk_twel_enforcer_threshold;
        if short_threshold >= 0.0 {
            let total_wel_short = self
                .bot_params_master
                .short
                .total_wallet_exposure_limit
                .max(0.0);
            if total_wel_short > 0.0 {
                let mut inputs: Vec<TwelEnforcerInputPosition> =
                    Vec::with_capacity(self.positions.short.len());
                let mut indices: Vec<usize> = self.positions.short.keys().cloned().collect();
                indices.sort_unstable();
                for idx in indices {
                    if wel_blocked_short.contains(&idx) {
                        continue;
                    }
                    if let Some(position) = self.positions.short.get(&idx) {
                        let market_price = self.hlcvs_value(k, idx, CLOSE);
                        inputs.push(TwelEnforcerInputPosition {
                            idx,
                            position_size: position.size,
                            position_price: position.price,
                            market_price,
                            base_wallet_exposure_limit: self.bp(idx, SHORT).wallet_exposure_limit,
                            c_mult: self.exchange_params_list[idx].c_mult,
                            qty_step: self.exchange_params_list[idx].qty_step,
                            price_step: self.exchange_params_list[idx].price_step,
                            min_qty: self.exchange_params_list[idx].min_qty,
                        });
                    }
                }

                let actions = calc_twel_enforcer_actions(
                    SHORT,
                    short_threshold,
                    total_wel_short,
                    self.effective_n_positions.short.max(1),
                    balance,
                    &inputs,
                    skip_short,
                );
                for (idx, order) in actions {
                    results.push((idx, SHORT, order));
                }
            }
        }

        results
    }

    fn update_open_orders_all(&mut self, k: usize) {
        self.open_orders.long.clear();
        self.open_orders.short.clear();
        if self.trading_enabled.long {
            let mut active_long_indices: Vec<usize> = self.positions.long.keys().cloned().collect();
            if self.positions.long.len() != self.effective_n_positions.long {
                self.update_actives_long();
                active_long_indices = self.actives.long.iter().cloned().collect();
            }
            active_long_indices.sort();
            for &idx in &active_long_indices {
                if self.coin_is_tradeable_at(idx, k) {
                    self.update_open_orders_long_single(k, idx);
                }
            }
        }
        if self.trading_enabled.short {
            let mut active_short_indices: Vec<usize> =
                self.positions.short.keys().cloned().collect();
            if self.positions.short.len() != self.effective_n_positions.short {
                self.update_actives_short();
                active_short_indices = self.actives.short.iter().cloned().collect();
            }
            active_short_indices.sort();
            for &idx in &active_short_indices {
                if self.coin_is_tradeable_at(idx, k) {
                    self.update_open_orders_short_single(k, idx);
                }
            }
        }

        let (unstucking_idx, unstucking_pside, unstucking_close) = self.calc_unstucking_close(k);
        if unstucking_pside != NO_POS {
            match unstucking_pside {
                LONG => {
                    self.open_orders
                        .long
                        .entry(unstucking_idx)
                        .or_default()
                        .closes = vec![unstucking_close];
                }
                SHORT => {
                    self.open_orders
                        .short
                        .entry(unstucking_idx)
                        .or_default()
                        .closes = vec![unstucking_close];
                }
                _ => unreachable!(),
            }
        }

        let skip = if unstucking_pside != NO_POS {
            Some((unstucking_idx, unstucking_pside))
        } else {
            None
        };
        let enforcer_orders = self.calc_twel_enforcer_orders(k, skip);
        for (idx, pside, order) in enforcer_orders {
            match pside {
                LONG => self
                    .open_orders
                    .long
                    .entry(idx)
                    .or_default()
                    .closes
                    .push(order),
                SHORT => self
                    .open_orders
                    .short
                    .entry(idx)
                    .or_default()
                    .closes
                    .push(order),
                _ => (),
            }
        }

        if self.trading_enabled.long {
            let long_indices: Vec<usize> = self.open_orders.long.keys().cloned().collect();
            for idx in long_indices {
                let has_auto = self
                    .open_orders
                    .long
                    .get(&idx)
                    .map(|orders| {
                        orders.closes.iter().any(|o| {
                            matches!(
                                o.order_type,
                                OrderType::CloseAutoReduceWelLong
                                    | OrderType::CloseAutoReduceTwelLong
                            )
                        })
                    })
                    .unwrap_or(false);
                if !has_auto {
                    continue;
                }
                let position_abs = self
                    .positions
                    .long
                    .get(&idx)
                    .map(|pos| pos.size.abs())
                    .unwrap_or(0.0);
                if let Some(orders) = self.open_orders.long.get_mut(&idx) {
                    Self::trim_reduce_only_orders(
                        &mut orders.closes,
                        position_abs,
                        LONG,
                        &self.exchange_params_list[idx],
                    );
                }
            }
        }
        if self.trading_enabled.short {
            let short_indices: Vec<usize> = self.open_orders.short.keys().cloned().collect();
            for idx in short_indices {
                let has_auto = self
                    .open_orders
                    .short
                    .get(&idx)
                    .map(|orders| {
                        orders.closes.iter().any(|o| {
                            matches!(
                                o.order_type,
                                OrderType::CloseAutoReduceWelShort
                                    | OrderType::CloseAutoReduceTwelShort
                            )
                        })
                    })
                    .unwrap_or(false);
                if !has_auto {
                    continue;
                }
                let position_abs = self
                    .positions
                    .short
                    .get(&idx)
                    .map(|pos| pos.size.abs())
                    .unwrap_or(0.0);
                if let Some(orders) = self.open_orders.short.get_mut(&idx) {
                    Self::trim_reduce_only_orders(
                        &mut orders.closes,
                        position_abs,
                        SHORT,
                        &self.exchange_params_list[idx],
                    );
                }
            }
        }

        self.gate_entries_portfolio(k, LONG);
        self.gate_entries_portfolio(k, SHORT);
    }

    fn gate_entries_portfolio(&mut self, k: usize, side: usize) {
        let (trading_enabled, total_limit) = match side {
            LONG => (
                self.trading_enabled.long,
                self.bot_params_master.long.total_wallet_exposure_limit,
            ),
            SHORT => (
                self.trading_enabled.short,
                self.bot_params_master.short.total_wallet_exposure_limit,
            ),
            _ => (false, 0.0),
        };
        if !trading_enabled {
            return;
        }
        let open_orders_side = match side {
            LONG => &mut self.open_orders.long,
            SHORT => &mut self.open_orders.short,
            _ => return,
        };
        if total_limit <= 0.0 {
            for bundle in open_orders_side.values_mut() {
                bundle.entries.clear();
            }
            return;
        }
        let balance = self.balance.usd_total_balance_rounded;
        if balance <= 0.0 {
            for bundle in open_orders_side.values_mut() {
                bundle.entries.clear();
            }
            return;
        }
        let positions_payload: Vec<GateEntriesPosition> = match side {
            LONG => {
                let mut indices: Vec<usize> = self.positions.long.keys().cloned().collect();
                indices.sort_unstable();
                indices
                    .into_iter()
                    .filter_map(|idx| {
                        self.positions
                            .long
                            .get(&idx)
                            .map(|pos| GateEntriesPosition {
                                idx,
                                position_size: pos.size,
                                position_price: pos.price,
                                c_mult: self.exchange_params_list[idx].c_mult,
                            })
                    })
                    .collect()
            }
            SHORT => {
                let mut indices: Vec<usize> = self.positions.short.keys().cloned().collect();
                indices.sort_unstable();
                indices
                    .into_iter()
                    .filter_map(|idx| {
                        self.positions
                            .short
                            .get(&idx)
                            .map(|pos| GateEntriesPosition {
                                idx,
                                position_size: pos.size,
                                position_price: pos.price,
                                c_mult: self.exchange_params_list[idx].c_mult,
                            })
                    })
                    .collect()
            }
            _ => Vec::new(),
        };

        let mut entry_candidates: Vec<GateEntriesCandidate> = Vec::new();
        {
            let mut order_indices: Vec<usize> = open_orders_side.keys().cloned().collect();
            order_indices.sort_unstable();
            for idx in order_indices {
                let col = self.active_coin_indices[idx];
                let raw_market_price = self.hlcvs[[k, col, CLOSE]];
                let bundle = match open_orders_side.get(&idx) {
                    Some(bundle) => bundle,
                    None => continue,
                };
                if bundle.entries.is_empty() {
                    continue;
                }
                let params = &self.exchange_params_list[idx];
                for order in &bundle.entries {
                    let qty = order.qty.abs();
                    if qty <= 0.0 || order.price <= 0.0 {
                        continue;
                    }
                    let mut market_price = raw_market_price;
                    if !market_price.is_finite() || market_price <= 0.0 {
                        market_price = if order.price > 0.0 {
                            order.price
                        } else {
                            params.price_step
                        };
                    }
                    entry_candidates.push(GateEntriesCandidate {
                        idx,
                        qty,
                        price: order.price,
                        qty_step: params.qty_step,
                        min_qty: params.min_qty,
                        min_cost: params.min_cost,
                        c_mult: params.c_mult,
                        market_price,
                        order_type: order.order_type,
                    });
                }
            }
        }
        if entry_candidates.is_empty() {
            return;
        }
        let gated = gate_entries_by_twel(
            side,
            balance,
            total_limit,
            &positions_payload,
            &entry_candidates,
        );
        let mut new_entries_map: HashMap<usize, Vec<Order>> = HashMap::new();
        for decision in gated {
            let qty_signed = if side == LONG {
                decision.qty
            } else {
                -decision.qty
            };
            new_entries_map
                .entry(decision.idx)
                .or_default()
                .push(Order {
                    qty: qty_signed,
                    price: decision.price,
                    order_type: decision.order_type,
                });
        }
        for (&idx, bundle) in open_orders_side.iter_mut() {
            if bundle.entries.is_empty() {
                continue;
            }
            if let Some(new_entries) = new_entries_map.remove(&idx) {
                bundle.entries = new_entries;
            } else {
                bundle.entries.clear();
            }
        }
    }

    #[inline]
    fn trim_reduce_only_orders(
        closes: &mut Vec<Order>,
        position_abs: f64,
        side: usize,
        exchange_params: &ExchangeParams,
    ) {
        const QTY_TOLERANCE: f64 = 1e-9;
        if closes.is_empty() {
            return;
        }
        if position_abs <= QTY_TOLERANCE {
            closes.clear();
            return;
        }

        let mut closes_sorted = closes.clone();
        closes_sorted.sort_by(|a, b| match side {
            LONG => a.price.partial_cmp(&b.price).unwrap_or(Ordering::Equal),
            SHORT => b.price.partial_cmp(&a.price).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        });

        let mut trimmed: Vec<Order> = Vec::with_capacity(closes.len());
        let mut remaining = position_abs;

        for mut order in closes_sorted {
            if remaining <= QTY_TOLERANCE {
                break;
            }
            let mut qty_abs = order.qty.abs().min(remaining);
            qty_abs = round_dn(qty_abs, exchange_params.qty_step);
            if qty_abs <= QTY_TOLERANCE {
                continue;
            }
            order.qty = if order.qty.is_sign_negative() {
                -qty_abs
            } else {
                qty_abs
            };
            trimmed.push(order);
            remaining -= qty_abs;
        }

        trimmed.sort_by(|a, b| match side {
            LONG => a.price.partial_cmp(&b.price).unwrap_or(Ordering::Equal),
            SHORT => b.price.partial_cmp(&a.price).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        });

        *closes = trimmed;
    }

    fn update_emas(&mut self, k: usize) {
        // Compute/refresh latest 1h bucket on whole-hour boundaries
        let current_ts = self.first_timestamp_ms + (k as u64) * 60_000u64;
        let hour_boundary = (current_ts / 3_600_000u64) * 3_600_000u64;
        if hour_boundary > self.last_hour_boundary_ms {
            // window is from max(first_ts, last_boundary) to previous minute
            let window_start_ms = self.first_timestamp_ms.max(self.last_hour_boundary_ms);
            if current_ts > window_start_ms + 60_000 {
                let start_idx = ((window_start_ms - self.first_timestamp_ms) / 60_000u64) as usize;
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

            // Update hourly log-range EMAs for grid spacing adjustments
            if self.needs_grid_log_range_long || self.needs_grid_log_range_short {
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
                    let grid_alpha_long = self.ema_alphas[i].grid_log_range_alpha_long;
                    let grid_alpha_short = self.ema_alphas[i].grid_log_range_alpha_short;
                    let emas = &mut self.emas[i];
                    if self.needs_grid_log_range_long && grid_alpha_long > 0.0 {
                        emas.grid_log_range_long = update_adjusted_ema(
                            hour_log_range,
                            grid_alpha_long,
                            &mut emas.grid_log_range_long_num,
                            &mut emas.grid_log_range_long_den,
                        );
                    }
                    if self.needs_grid_log_range_short && grid_alpha_short > 0.0 {
                        emas.grid_log_range_short = update_adjusted_ema(
                            hour_log_range,
                            grid_alpha_short,
                            &mut emas.grid_log_range_short_num,
                            &mut emas.grid_log_range_short_den,
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
            let vol = if vol_raw.is_finite() {
                f64::max(0.0, vol_raw)
            } else {
                0.0
            };
            let high = self.hlcvs_value(k, i, HIGH);
            let low = self.hlcvs_value(k, i, LOW);
            if !high.is_finite() || !low.is_finite() {
                continue;
            }

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

fn calc_ema_alphas(bot_params_pair: &BotParamsPair) -> EmaAlphas {
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

    let ema_alphas_long = ema_spans_long.map(|x| 2.0 / (x + 1.0));

    let ema_alphas_short = ema_spans_short.map(|x| 2.0 / (x + 1.0));

    EmaAlphas {
        long: Alphas {
            alphas: ema_alphas_long,
        },
        short: Alphas {
            alphas: ema_alphas_short,
        },
        // EMA spans for the volume/log range filters (alphas precomputed from spans)
        vol_alpha_long: 2.0 / (bot_params_pair.long.filter_volume_ema_span as f64 + 1.0),
        vol_alpha_short: 2.0 / (bot_params_pair.short.filter_volume_ema_span as f64 + 1.0),
        log_range_alpha_long: 2.0 / (bot_params_pair.long.filter_volatility_ema_span as f64 + 1.0),
        log_range_alpha_short: 2.0
            / (bot_params_pair.short.filter_volatility_ema_span as f64 + 1.0),
        grid_log_range_alpha_long: {
            let span = bot_params_pair.long.entry_volatility_ema_span_hours;
            if span > 0.0 {
                2.0 / (span + 1.0)
            } else {
                0.0
            }
        },
        grid_log_range_alpha_short: {
            let span = bot_params_pair.short.entry_volatility_ema_span_hours;
            if span > 0.0 {
                2.0 / (span + 1.0)
            } else {
                0.0
            }
        },
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
