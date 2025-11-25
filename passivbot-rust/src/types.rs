use core::str::FromStr;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{PyArray1, PyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{Py, PyResult, Python};
use serde::Serialize;
use std::collections::HashMap;
use strum_macros::{Display, EnumIter, EnumString};

/// Canonical metadata describing one coin/contract entry inside a unified HLCV tensor.
///
/// The numeric fields mirror the values currently distributed through `mss` dictionaries in
/// Python.  Keeping them here ensures every column in the `(T, N, 4)` cube has an aligned
/// metadata record with deterministic indexing.
#[derive(Clone, Debug, Serialize)]
pub struct CoinMeta {
    /// Zero-based index that lines up with the second dimension of the HLCV tensor.
    pub index: usize,
    /// Full CCXT-style symbol (e.g. "BTC/USDT:USDT").
    pub symbol: String,
    /// Shorthand coin ticker (e.g. "BTC") assuming an implied USD quote.
    pub coin: String,
    /// Normalized exchange identifier (e.g. "binanceusdm").
    pub exchange: String,
    /// Quote asset, extracted from the symbol for readability.
    pub quote: String,
    /// Base asset, extracted from the symbol for readability.
    pub base: String,
    /// Smallest allowed quantity increment.
    pub qty_step: f64,
    /// Smallest allowed price increment.
    pub price_step: f64,
    /// Minimal order quantity enforced by the venue.
    pub min_qty: f64,
    /// Minimal notional/trade value enforced by the venue.
    pub min_cost: f64,
    /// Contract multiplier (lot size) used for sized calculations.
    pub c_mult: f64,
    /// Maker fee rate associated with this market (fractional, not percentage).
    pub maker_fee: f64,
    /// Taker fee rate associated with this market (fractional, not percentage).
    pub taker_fee: f64,
    /// First index (inclusive) in the HLCV tensor that contains valid data for this coin.
    pub first_valid_index: usize,
    /// Last index (inclusive) in the HLCV tensor that contains valid data for this coin.
    pub last_valid_index: usize,
    /// Warmup bars (expressed in minutes) required before this coin may trade.
    pub warmup_minutes: usize,
    /// Index in the HLCV tensor where trading is allowed to begin (>= `first_valid_index`).
    pub trade_start_index: usize,
}

/// Metadata describing how an HLCV tensor was produced.  It travels alongside the raw
/// `(time, coin, feature)` cube so both Python and Rust callers can reason about how to
/// slice or merge the data without extra dictionaries.
#[derive(Clone, Debug, Serialize)]
pub struct HlcvsMeta {
    pub requested_start_timestamp_ms: u64,
    pub effective_start_timestamp_ms: u64,
    pub warmup_minutes_requested: u64,
    pub warmup_minutes_provided: u64,
    pub coins: Vec<CoinMeta>,
}

/// Represents a fully-qualified HLCV tensor and its associated metadata.  The NumPy arrays are
/// stored as owned Python references so Rust callers can obtain zero-copy views as needed while
/// Python retains control over the underlying memory.
#[derive(Clone)]
pub struct HlcvsBundle {
    pub hlcvs: Py<PyArray3<f64>>,
    pub btc_usd: Py<PyArray1<f64>>,
    pub timestamps: Py<PyArray1<i64>>,
    pub meta: HlcvsMeta,
}

impl HlcvsBundle {
    pub fn coin_meta_by_index(&self, idx: usize) -> Option<&CoinMeta> {
        self.meta.coins.iter().find(|coin| coin.index == idx)
    }

    pub fn coin_meta_by_symbol(&self, symbol: &str) -> Option<&CoinMeta> {
        self.meta
            .coins
            .iter()
            .find(|coin| coin.symbol == symbol || coin.coin == symbol)
    }

    pub fn coins_len(&self) -> usize {
        self.meta.coins.len()
    }

    pub fn validate_shapes(&self, py: Python<'_>) -> PyResult<()> {
        let hlcvs_ref = self.hlcvs.bind(py);
        let shape = hlcvs_ref.shape();
        if shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "hlcvs must be a 3D array, got ndim={}",
                shape.len()
            )));
        }

        let n_timesteps = shape[0];
        let n_coins = shape[1];
        if n_coins != self.meta.coins.len() {
            return Err(PyValueError::new_err(format!(
                "coin metadata length ({}) does not match hlcvs coin dimension ({})",
                self.meta.coins.len(),
                n_coins
            )));
        }

        let timestamps_ref = self.timestamps.bind(py);
        if timestamps_ref.len() != n_timesteps {
            return Err(PyValueError::new_err(format!(
                "timestamps length ({}) does not match hlcvs timesteps ({})",
                timestamps_ref.len(),
                n_timesteps
            )));
        }

        let btc_ref = self.btc_usd.bind(py);
        if btc_ref.len() != n_timesteps {
            return Err(PyValueError::new_err(format!(
                "btc_usd length ({}) does not match hlcvs timesteps ({})",
                btc_ref.len(),
                n_timesteps
            )));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct ExchangeParams {
    pub qty_step: f64,
    pub price_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
    pub c_mult: f64,
}

impl Default for ExchangeParams {
    fn default() -> Self {
        ExchangeParams {
            qty_step: 0.00001,
            price_step: 0.00001,
            min_qty: 0.00001,
            min_cost: 1.0,
            c_mult: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BacktestParams {
    pub starting_balance: f64,
    pub maker_fee: f64,
    pub coins: Vec<String>,
    pub active_coin_indices: Option<Vec<usize>>,
    pub first_timestamp_ms: u64,
    pub requested_start_timestamp_ms: u64,
    pub first_valid_indices: Vec<usize>,
    pub last_valid_indices: Vec<usize>,
    pub warmup_minutes: Vec<usize>,
    pub trade_start_indices: Vec<usize>,
    pub global_warmup_bars: usize,
    pub btc_collateral_cap: f64,
    pub btc_collateral_ltv_cap: Option<f64>,
    pub metrics_only: bool,
    pub filter_by_min_effective_cost: bool,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Position {
    pub size: f64,
    pub price: f64,
}

#[derive(Debug, Default)]
pub struct Positions {
    pub long: HashMap<usize, Position>,
    pub short: HashMap<usize, Position>,
}

#[derive(Debug, Default, Clone)]
pub struct EMABands {
    pub upper: f64,
    pub lower: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub qty: f64,
    pub price: f64,
    pub order_type: OrderType,
}

impl Default for Order {
    fn default() -> Self {
        Order {
            qty: 0.0,
            price: 0.0,
            order_type: OrderType::Empty,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OrderBook {
    pub bid: f64,
    pub ask: f64,
}

#[derive(Debug, Default, Clone)]
pub struct StateParams {
    pub balance: f64,
    pub order_book: OrderBook,
    pub ema_bands: EMABands,
    pub grid_log_range: f64,
}

#[derive(Clone, Default, Debug)]
pub struct BotParamsPair {
    pub long: BotParams,
    pub short: BotParams,
}

#[derive(Clone, Default, Debug)]
pub struct BotParams {
    pub close_grid_markup_end: f64,
    pub close_grid_markup_start: f64,
    pub close_grid_qty_pct: f64,
    pub close_trailing_retracement_pct: f64,
    pub close_trailing_grid_ratio: f64,
    pub close_trailing_qty_pct: f64,
    pub close_trailing_threshold_pct: f64,
    pub entry_grid_double_down_factor: f64,
    pub entry_grid_spacing_volatility_weight: f64,
    pub entry_grid_spacing_we_weight: f64,
    pub entry_grid_spacing_pct: f64,
    pub entry_volatility_ema_span_hours: f64,
    pub entry_initial_ema_dist: f64,
    pub entry_initial_qty_pct: f64,
    pub entry_trailing_double_down_factor: f64,
    pub entry_trailing_retracement_pct: f64,
    pub entry_trailing_retracement_we_weight: f64,
    pub entry_trailing_retracement_volatility_weight: f64,
    pub entry_trailing_grid_ratio: f64,
    pub entry_trailing_threshold_pct: f64,
    pub entry_trailing_threshold_we_weight: f64,
    pub entry_trailing_threshold_volatility_weight: f64,
    pub filter_volatility_ema_span: f64,
    pub filter_volatility_drop_pct: f64,
    pub filter_volume_ema_span: f64,
    pub filter_volume_drop_pct: f64,
    pub ema_span_0: f64,
    pub ema_span_1: f64,
    pub n_positions: usize,
    pub total_wallet_exposure_limit: f64,
    pub wallet_exposure_limit: f64, // per-position base limit (without excess allowance)
    pub risk_wel_enforcer_threshold: f64,
    pub risk_twel_enforcer_threshold: f64,
    pub risk_we_excess_allowance_pct: f64,
    pub unstuck_close_pct: f64,
    pub unstuck_ema_dist: f64,
    pub unstuck_loss_allowance_pct: f64,
    pub unstuck_threshold: f64,
}

#[derive(Debug)]
pub struct TrailingPriceBundle {
    pub min_since_open: f64,
    pub max_since_min: f64,
    pub max_since_open: f64,
    pub min_since_max: f64,
}
impl Default for TrailingPriceBundle {
    fn default() -> Self {
        TrailingPriceBundle {
            min_since_open: f64::MAX,
            max_since_min: 0.0,
            max_since_open: 0.0,
            min_since_max: f64::MAX,
        }
    }
}

#[repr(u16)]
#[derive(
    Debug,
    PartialEq,
    Eq,
    Clone,
    Copy,
    IntoPrimitive,
    TryFromPrimitive,
    EnumString,
    Display,
    EnumIter,
)]
#[strum(serialize_all = "snake_case")]
pub enum OrderType {
    EntryInitialNormalLong = 0,
    EntryInitialPartialLong = 1,
    EntryTrailingNormalLong = 2,
    EntryTrailingCroppedLong = 3,
    EntryGridNormalLong = 4,
    EntryGridCroppedLong = 5,
    EntryGridInflatedLong = 6,

    CloseGridLong = 7,
    CloseTrailingLong = 8,
    CloseUnstuckLong = 9,
    CloseAutoReduceTwelLong = 10,

    EntryInitialNormalShort = 11,
    EntryInitialPartialShort = 12,
    EntryTrailingNormalShort = 13,
    EntryTrailingCroppedShort = 14,
    EntryGridNormalShort = 15,
    EntryGridCroppedShort = 16,
    EntryGridInflatedShort = 17,

    CloseGridShort = 18,
    CloseTrailingShort = 19,
    CloseUnstuckShort = 20,
    CloseAutoReduceTwelShort = 21,

    ClosePanicLong = 22,
    ClosePanicShort = 23,
    CloseAutoReduceWelLong = 24,
    CloseAutoReduceWelShort = 25,

    Empty = 65535,
}

impl OrderType {
    #[inline]
    pub const fn id(self) -> u16 {
        self as u16
    }
    #[inline]
    pub fn from_snake(s: &str) -> Option<Self> {
        OrderType::from_str(s).ok()
    }
}

impl OrderType {
    #[inline]
    pub const fn is_long(self) -> bool {
        use OrderType::*;
        matches!(
            self,
            EntryInitialNormalLong
                | EntryInitialPartialLong
                | EntryTrailingNormalLong
                | EntryTrailingCroppedLong
                | EntryGridNormalLong
                | EntryGridCroppedLong
                | EntryGridInflatedLong
                | CloseGridLong
                | CloseTrailingLong
                | CloseUnstuckLong
                | CloseAutoReduceTwelLong
                | CloseAutoReduceWelLong
                | ClosePanicLong
        )
    }
}

#[derive(Default)]
pub struct Balance {
    pub usd_cash_wallet: f64,                // raw usd wallet balance
    pub usd_total_balance: f64,              // usd cash + btc converted to usd
    pub usd_total_balance_rounded: f64,      // rounded usd total for hysteresis logic
    pub btc_cash_wallet: f64,                // raw btc wallet balance
    pub btc_total_balance: f64,              // btc cash + usd converted to btc
    pub use_btc_collateral: bool,            // whether to use btc as collateral
    pub btc_collateral_cap: f64,             // target/cap ratio of collateral held in btc
    pub btc_collateral_ltv_cap: Option<f64>, // optional LTV ceiling when topping up btc
}

#[derive(Default, Clone)]
pub struct Equities {
    pub timestamps_ms: Vec<u64>,
    pub usd_total_equity: Vec<f64>,
    pub btc_total_equity: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub index: usize,
    pub timestamp_ms: u64,
    pub coin: String,
    pub pnl: f64,
    pub fee_paid: f64,
    pub usd_total_balance: f64,
    pub btc_cash_wallet: f64,
    pub usd_cash_wallet: f64,
    pub btc_price: f64,
    pub fill_qty: f64,
    pub fill_price: f64,
    pub position_size: f64,
    pub position_price: f64,
    pub order_type: OrderType,
    pub wallet_exposure: f64,
    pub total_wallet_exposure: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    pub adg: f64,
    pub mdg: f64,
    pub gain: f64,
    pub adg_pnl: f64,
    pub mdg_pnl: f64,
    pub sharpe_ratio_pnl: f64,
    pub sortino_ratio_pnl: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub omega_ratio: f64,
    pub expected_shortfall_1pct: f64,
    pub calmar_ratio: f64,
    pub sterling_ratio: f64,
    pub drawdown_worst: f64,
    pub drawdown_worst_mean_1pct: f64,
    pub equity_balance_diff_neg_max: f64,
    pub equity_balance_diff_neg_mean: f64,
    pub equity_balance_diff_pos_max: f64,
    pub equity_balance_diff_pos_mean: f64,
    pub loss_profit_ratio: f64,
    pub peak_recovery_hours_equity: f64,
    pub peak_recovery_hours_pnl: f64,

    pub equity_choppiness: f64,
    pub equity_jerkiness: f64,
    pub exponential_fit_error: f64,
    pub equity_choppiness_w: f64,
    pub equity_jerkiness_w: f64,
    pub exponential_fit_error_w: f64,

    pub positions_held_per_day: f64,
    pub position_held_hours_mean: f64,
    pub position_held_hours_max: f64,
    pub position_held_hours_median: f64,
    pub position_unchanged_hours_max: f64,

    pub adg_w: f64,
    pub adg_pnl_w: f64,
    pub mdg_pnl_w: f64,
    pub sharpe_ratio_pnl_w: f64,
    pub sortino_ratio_pnl_w: f64,
    pub mdg_w: f64,
    pub sharpe_ratio_w: f64,
    pub sortino_ratio_w: f64,
    pub omega_ratio_w: f64,
    pub calmar_ratio_w: f64,
    pub sterling_ratio_w: f64,
    pub loss_profit_ratio_w: f64,
    pub volume_pct_per_day_avg: f64,
    pub volume_pct_per_day_avg_w: f64,

    pub total_wallet_exposure_max: f64,
    pub total_wallet_exposure_mean: f64,
    pub total_wallet_exposure_median: f64,
    pub entry_initial_balance_pct_long: f64,
    pub entry_initial_balance_pct_short: f64,
}

impl Default for Analysis {
    fn default() -> Self {
        Analysis {
            adg: 0.0,
            mdg: 0.0,
            gain: 0.0,
            adg_pnl: 0.0,
            mdg_pnl: 0.0,
            sharpe_ratio_pnl: 0.0,
            sortino_ratio_pnl: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            omega_ratio: 0.0,
            expected_shortfall_1pct: 0.0,
            calmar_ratio: 0.0,
            sterling_ratio: 0.0,
            drawdown_worst: 1.0,
            drawdown_worst_mean_1pct: 1.0,
            equity_balance_diff_neg_max: 1.0,
            equity_balance_diff_neg_mean: 1.0,
            equity_balance_diff_pos_max: 1.0,
            equity_balance_diff_pos_mean: 1.0,
            loss_profit_ratio: 1.0,
            peak_recovery_hours_equity: 0.0,
            peak_recovery_hours_pnl: 0.0,
            equity_choppiness: 1.0,
            equity_jerkiness: 1.0,
            exponential_fit_error: 1.0,
            positions_held_per_day: 0.0,
            position_held_hours_mean: 0.0,
            position_held_hours_max: 0.0,
            position_held_hours_median: 0.0,
            position_unchanged_hours_max: 0.0,
            adg_w: 0.0,
            adg_pnl_w: 0.0,
            mdg_pnl_w: 0.0,
            sharpe_ratio_pnl_w: 0.0,
            sortino_ratio_pnl_w: 0.0,
            mdg_w: 0.0,
            sharpe_ratio_w: 0.0,
            sortino_ratio_w: 0.0,
            omega_ratio_w: 0.0,
            calmar_ratio_w: 0.0,
            sterling_ratio_w: 0.0,
            loss_profit_ratio_w: 1.0,
            equity_choppiness_w: 1.0,
            equity_jerkiness_w: 1.0,
            exponential_fit_error_w: 1.0,
            volume_pct_per_day_avg: 0.0,
            volume_pct_per_day_avg_w: 0.0,
            total_wallet_exposure_max: 0.0,
            total_wallet_exposure_mean: 0.0,
            total_wallet_exposure_median: 0.0,
            entry_initial_balance_pct_long: 0.0,
            entry_initial_balance_pct_short: 0.0,
        }
    }
}
