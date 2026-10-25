use core::str::FromStr;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{Py, PyResult, Python};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use strum_macros::{Display, EnumIter, EnumString};

/// Independent admission policy for each position side. This does not change exchange minima.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MinimumCostFilter {
    pub long: bool,
    pub short: bool,
}

impl MinimumCostFilter {
    pub fn uniform(enabled: bool) -> Self {
        Self {
            long: enabled,
            short: enabled,
        }
    }
    pub fn for_side(self, side: usize) -> bool {
        match side {
            crate::constants::LONG => self.long,
            crate::constants::SHORT => self.short,
            _ => panic!("invalid position side"),
        }
    }
}

/// Independent permission to increase a position side. Capacity and operational
/// modes remain separate; this never disables held-position close/risk handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SideActivation {
    pub long: bool,
    pub short: bool,
}

impl SideActivation {
    pub const fn uniform(enabled: bool) -> Self {
        Self {
            long: enabled,
            short: enabled,
        }
    }

    pub fn for_side(self, side: usize) -> bool {
        match side {
            crate::constants::LONG => self.long,
            crate::constants::SHORT => self.short,
            _ => panic!("invalid position side"),
        }
    }
}

/// Fixed side-specific normalized score margins; no independent optimizer genes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForagerScoreHysteresis {
    pub long: f64,
    pub short: f64,
}

impl ForagerScoreHysteresis {
    pub const fn uniform(margin: f64) -> Self {
        Self {
            long: margin,
            short: margin,
        }
    }
}

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
    pub maker_fee_fraction: f64,
    /// Taker fee rate associated with this market (fractional, not percentage).
    pub taker_fee_fraction: f64,
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
    pub active_coin_indices: Option<Vec<usize>>,
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
    pub eligibility: Option<Py<PyArray2<u8>>>,
    pub eligibility_rows: Option<Py<PyArray1<i32>>>,
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
            let active_indices = self.meta.active_coin_indices.as_ref().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "coin metadata length ({}) does not match hlcvs coin dimension ({})",
                    self.meta.coins.len(),
                    n_coins
                ))
            })?;
            if active_indices.len() != self.meta.coins.len() {
                return Err(PyValueError::new_err(format!(
                    "active_coin_indices length ({}) does not match coin metadata length ({})",
                    active_indices.len(),
                    self.meta.coins.len()
                )));
            }
            let mut seen = HashSet::with_capacity(active_indices.len());
            for (idx, &col) in active_indices.iter().enumerate() {
                if col >= n_coins {
                    return Err(PyValueError::new_err(format!(
                        "active coin index {} exceeds hlcvs coin dimension {}",
                        col, n_coins
                    )));
                }
                if !seen.insert(col) {
                    return Err(PyValueError::new_err(format!(
                        "active coin index {} appears more than once",
                        col
                    )));
                }
                if self.meta.coins[idx].index != col {
                    return Err(PyValueError::new_err(format!(
                        "coin metadata index ({}) does not match active coin index ({}) at position {}",
                        self.meta.coins[idx].index, col, idx
                    )));
                }
            }
        } else if let Some(active_indices) = &self.meta.active_coin_indices {
            if active_indices.len() != self.meta.coins.len() {
                return Err(PyValueError::new_err(format!(
                    "active_coin_indices length ({}) does not match coin metadata length ({})",
                    active_indices.len(),
                    self.meta.coins.len()
                )));
            }
            let mut seen = HashSet::with_capacity(active_indices.len());
            for (idx, &col) in active_indices.iter().enumerate() {
                if col >= n_coins {
                    return Err(PyValueError::new_err(format!(
                        "active coin index {} exceeds hlcvs coin dimension {}",
                        col, n_coins
                    )));
                }
                if !seen.insert(col) {
                    return Err(PyValueError::new_err(format!(
                        "active coin index {} appears more than once",
                        col
                    )));
                }
                if self.meta.coins[idx].index != col {
                    return Err(PyValueError::new_err(format!(
                        "coin metadata index ({}) does not match active coin index ({}) at position {}",
                        self.meta.coins[idx].index, col, idx
                    )));
                }
            }
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

        match (&self.eligibility, &self.eligibility_rows) {
            (None, None) => (),
            (Some(values), Some(rows)) => {
                let values = values.bind(py).readonly();
                let rows = rows.bind(py).readonly();
                let a = values.as_array();
                let r = rows.as_array();
                if a.nrows() == 0
                    || a.ncols() != n_coins
                    || a.iter().any(|&x| x > 1)
                    || r.len() != n_timesteps
                    || r.iter().any(|&x| x < 0 || x as usize >= a.nrows())
                {
                    return Err(PyValueError::new_err(
                        "invalid Organillo binary eligibility/row mapping",
                    ));
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Organillo eligibility and row mapping must be supplied together",
                ))
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExchangeParams {
    pub qty_step: f64,
    pub price_step: f64,
    pub min_qty: f64,
    pub min_cost: f64,
    pub c_mult: f64,
    pub maker_fee_fraction: f64,
    pub taker_fee_fraction: f64,
}

impl Default for ExchangeParams {
    fn default() -> Self {
        ExchangeParams {
            qty_step: 0.00001,
            price_step: 0.00001,
            min_qty: 0.00001,
            min_cost: 1.0,
            c_mult: 1.0,
            maker_fee_fraction: 0.0002,
            taker_fee_fraction: 0.00055,
        }
    }
}

impl ExchangeParams {
    pub fn validate_required(&self) -> Result<(), String> {
        fn finite_positive(value: f64, field: &str) -> Result<(), String> {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("{field} must be finite and > 0"))
            }
        }
        fn finite_non_negative(value: f64, field: &str) -> Result<(), String> {
            if value.is_finite() && value >= 0.0 {
                Ok(())
            } else {
                Err(format!("{field} must be finite and >= 0"))
            }
        }
        fn finite(value: f64, field: &str) -> Result<(), String> {
            if value.is_finite() {
                Ok(())
            } else {
                Err(format!("{field} must be finite"))
            }
        }

        finite_positive(self.qty_step, "qty_step")?;
        finite_positive(self.price_step, "price_step")?;
        finite_non_negative(self.min_qty, "min_qty")?;
        finite_non_negative(self.min_cost, "min_cost")?;
        finite_positive(self.c_mult, "c_mult")?;
        finite(self.maker_fee_fraction, "maker_fee_fraction")?;
        finite(self.taker_fee_fraction, "taker_fee_fraction")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EquityHardStopLossConfig {
    pub hsl: Option<crate::backtest::hsl_runtime::Config>,
}

impl Default for EquityHardStopLossConfig {
    fn default() -> Self {
        Self {
            hsl: Some(crate::backtest::hsl_runtime::Config::default()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BacktestParams {
    pub side_activation: SideActivation,
    pub minimum_coin_age_days: [f64; 2],
    pub inception_timestamps_ms: Vec<Option<u64>>,
    pub starting_balance: f64,
    pub maker_fee_fraction: f64,
    pub taker_fee_fraction: f64,
    pub coins: Vec<String>,
    pub active_coin_indices: Option<Vec<usize>>,
    pub first_timestamp_ms: u64,
    pub requested_start_timestamp_ms: u64,
    pub first_valid_indices: Vec<usize>,
    pub last_valid_indices: Vec<usize>,
    pub warmup_minutes: Vec<usize>,
    pub trade_start_indices: Vec<usize>,
    pub global_warmup_bars: usize,
    pub btc_collateral_target_ratio: f64,
    pub max_usd_debt_to_wallet_value_ratio: Option<f64>,
    pub metrics_only: bool,
    pub ew_half_life_days: Option<f64>,
    pub hsl_detailed_report: bool,
    pub skip_btc_analysis: bool,
    pub filter_by_min_effective_cost: MinimumCostFilter,
    pub dynamic_wel_by_tradability: bool,
    pub hedge_mode: bool,
    pub loss_gate_max_drawdown_fraction: f64,
    pub unstuck_memory_ema_span_minutes: f64,
    pub loss_gate_memory_ema_span_minutes: f64,
    pub equity_floor_fraction: f64,
    pub equity_hard_stop_loss: EquityHardStopLossConfig,
    pub market_orders_allowed: bool,
    pub market_order_near_touch_fraction: f64,
    pub market_order_slippage_fraction: f64,
    pub limit_order_fill_buffer_fraction: f64,
    pub forager_score_hysteresis: ForagerScoreHysteresis,
    pub candle_interval_minutes: u64, // 1 for 1m candles (default), 5 for 5m, etc.
}

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub size: f64,
    pub price: f64,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EMABands {
    pub upper: f64,
    pub lower: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderBook {
    pub bid: f64,
    pub ask: f64,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateParams {
    pub balance: f64,
    pub order_book: OrderBook,
    pub ema_bands: EMABands,
    pub log_range_ema_1m: f64,
    pub log_range_ema_1h: f64,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotParamsPair {
    pub long: BotParams,
    pub short: BotParams,
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBudgetState {
    pub configured_wallet_exposure_limit_ratio: f64,
    pub effective_wallet_exposure_limit_ratio: f64,
    pub configured_n_positions: usize,
    pub effective_n_positions: usize,
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBudgetStatePair {
    pub long: RuntimeBudgetState,
    pub short: RuntimeBudgetState,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StrategyParamsPairValue {
    pub long: Value,
    pub short: Value,
}

fn default_hsl_enabled() -> bool {
    false
}

fn default_hsl_panic_close_order_type() -> String {
    "market".to_string()
}

fn default_true() -> bool {
    true
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, EnumString, Display,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TwelEnforcerPolicy {
    #[default]
    ReduceOverweight,
    ReducePortfolio,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForagerScoreWeights {
    pub volume: f64,
    pub ema_readiness: f64,
    pub log_range: f64,
    #[serde(default)]
    pub unilateralness: f64,
}

impl Default for ForagerScoreWeights {
    fn default() -> Self {
        Self {
            volume: 0.0,
            ema_readiness: 0.0,
            log_range: 1.0,
            unilateralness: 0.0,
        }
    }
}

impl ForagerScoreWeights {
    pub fn canonicalize(&self) -> Result<Self, String> {
        let values = [
            self.volume,
            self.ema_readiness,
            self.log_range,
            self.unilateralness,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("forager_score_weights must be finite and non-negative".to_string());
        }
        let total = self.volume + self.ema_readiness + self.log_range + self.unilateralness;
        if total <= 0.0 {
            return Ok(Self {
                volume: 0.0,
                ema_readiness: 1.0,
                log_range: 0.0,
                unilateralness: 0.0,
            });
        }
        Ok(Self {
            volume: self.volume / total,
            ema_readiness: self.ema_readiness / total,
            log_range: self.log_range / total,
            unilateralness: self.unilateralness / total,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotParams {
    #[serde(default)]
    pub close_grid_qty_fraction: f64,
    #[serde(default)]
    pub close_trailing_retracement_base_fraction: f64,
    #[serde(default)]
    pub close_trailing_qty_fraction: f64,
    #[serde(default)]
    pub close_trailing_threshold_base_fraction: f64,
    #[serde(default)]
    pub close_log_range_1h_coefficient: f64,
    #[serde(default)]
    pub close_log_range_1m_coefficient: f64,
    #[serde(default)]
    pub entry_grid_reentry_qty_multiplier: f64,
    #[serde(default)]
    pub entry_grid_threshold_base_fraction: f64,
    #[serde(default)]
    pub entry_log_range_ema_span_1h_candles: f64,
    #[serde(default)]
    pub entry_log_range_ema_span_minutes: f64,
    #[serde(default)]
    pub entry_log_range_1h_coefficient: f64,
    #[serde(default)]
    pub entry_log_range_1m_coefficient: f64,
    #[serde(default)]
    pub entry_exposure_ratio_coefficient: f64,
    #[serde(default)]
    pub entry_initial_ema_distance_fraction: f64,
    #[serde(default)]
    pub entry_initial_qty_fraction: f64,
    #[serde(default)]
    pub entry_trailing_reentry_qty_multiplier: f64,
    #[serde(default)]
    pub entry_trailing_retracement_base_fraction: f64,
    #[serde(default)]
    pub entry_trailing_threshold_base_fraction: f64,
    pub forager_log_range_ema_span_minutes: f64,
    pub forager_volume_ema_span_minutes: f64,
    #[serde(default, skip_serializing, rename = "filter_volatility_drop_pct")]
    pub _legacy_filter_volatility_drop_pct: f64,
    #[serde(default)]
    pub forager_score_weights: ForagerScoreWeights,
    #[serde(default)]
    pub is_forced_active: bool,
    /// Whether this symbol+position-side may open new positions.
    ///
    /// Backtests set this from side-specific approved-coins membership.  Live
    /// callers retain their existing mode-based eligibility contract through
    /// the default value.
    #[serde(default = "default_true")]
    pub entry_eligible: bool,
    #[serde(default)]
    pub ema_span_0_minutes: f64,
    #[serde(default)]
    pub ema_span_1_minutes: f64,
    #[serde(default = "default_hsl_enabled")]
    pub hsl_enabled: bool,
    #[serde(default = "default_hsl_panic_close_order_type")]
    pub hsl_panic_close_order_type: String,
    #[serde(default)]
    pub risk_entry_cooldown_minutes: f64,
    #[serde(default)]
    pub entry_cooldown_min_duration_minutes: f64,
    #[serde(default)]
    pub entry_cooldown_max_duration_minutes: Option<f64>,
    #[serde(default)]
    pub entry_cooldown_weights_minutes: crate::entry_cooldown::CooldownWeights,
    #[serde(default = "default_unilateralness_span")]
    pub unilateralness_ema_span_minutes: f64,
    pub n_positions: usize,
    pub total_wallet_exposure_limit_ratio: f64,
    pub wallet_exposure_limit_ratio: f64, // per-position base limit (without excess allowance)
    #[serde(default = "default_true")]
    pub risk_wel_enforcer_enabled: bool,
    pub risk_wel_enforcer_threshold_ratio: f64,
    #[serde(default = "default_true")]
    pub risk_twel_enforcer_enabled: bool,
    #[serde(default)]
    pub risk_twel_enforcer_policy: TwelEnforcerPolicy,
    pub risk_twel_enforcer_threshold_ratio: f64,
    pub risk_we_excess_allowance_fraction: f64,
    #[serde(default = "default_true")]
    pub unstuck_enabled: bool,
    #[serde(default = "default_true")]
    pub unstuck_ema_gating_enabled: bool,
    pub unstuck_close_fraction: f64,
    pub unstuck_ema_distance_fraction: f64,
    pub unstuck_ema_span_0_minutes: f64,
    pub unstuck_ema_span_1_minutes: f64,
    pub unstuck_loss_allowance_fraction: f64,
    pub unstuck_exposure_trigger_ratio: f64,
}

impl Default for BotParams {
    fn default() -> Self {
        Self {
            close_grid_qty_fraction: 0.0,
            close_trailing_retracement_base_fraction: 0.0,
            close_trailing_qty_fraction: 0.0,
            close_trailing_threshold_base_fraction: 0.0,
            close_log_range_1h_coefficient: 0.0,
            close_log_range_1m_coefficient: 0.0,
            entry_grid_reentry_qty_multiplier: 0.0,
            entry_grid_threshold_base_fraction: 0.0,
            entry_log_range_ema_span_1h_candles: 0.0,
            entry_log_range_ema_span_minutes: 0.0,
            entry_log_range_1h_coefficient: 0.0,
            entry_log_range_1m_coefficient: 0.0,
            entry_exposure_ratio_coefficient: 0.0,
            entry_initial_ema_distance_fraction: 0.0,
            entry_initial_qty_fraction: 0.0,
            entry_trailing_reentry_qty_multiplier: 0.0,
            entry_trailing_retracement_base_fraction: 0.0,
            entry_trailing_threshold_base_fraction: 0.0,
            forager_log_range_ema_span_minutes: 0.0,
            forager_volume_ema_span_minutes: 0.0,
            _legacy_filter_volatility_drop_pct: 0.0,
            forager_score_weights: ForagerScoreWeights::default(),
            is_forced_active: false,
            entry_eligible: true,
            ema_span_0_minutes: 0.0,
            ema_span_1_minutes: 0.0,
            hsl_enabled: default_hsl_enabled(),
            hsl_panic_close_order_type: default_hsl_panic_close_order_type(),
            risk_entry_cooldown_minutes: 0.0,
            entry_cooldown_min_duration_minutes: 0.0,
            entry_cooldown_max_duration_minutes: None,
            entry_cooldown_weights_minutes: Default::default(),
            unilateralness_ema_span_minutes: 60.0,
            n_positions: 0,
            total_wallet_exposure_limit_ratio: 0.0,
            wallet_exposure_limit_ratio: 0.0,
            risk_wel_enforcer_enabled: true,
            risk_wel_enforcer_threshold_ratio: 0.0,
            risk_twel_enforcer_enabled: true,
            risk_twel_enforcer_policy: TwelEnforcerPolicy::default(),
            risk_twel_enforcer_threshold_ratio: 0.0,
            risk_we_excess_allowance_fraction: 0.0,
            unstuck_enabled: true,
            unstuck_ema_gating_enabled: true,
            unstuck_close_fraction: 0.0,
            unstuck_ema_distance_fraction: 0.0,
            unstuck_ema_span_0_minutes: 60.0,
            unstuck_ema_span_1_minutes: 60.0,
            unstuck_loss_allowance_fraction: 0.0,
            unstuck_exposure_trigger_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeOrderContext {
    pub effective_wallet_exposure_limit_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    Serialize,
    Deserialize,
    IntoPrimitive,
    TryFromPrimitive,
    EnumString,
    Display,
    EnumIter,
)]
#[serde(rename_all = "snake_case")]
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
    EntryEmaAnchorLong = 26,
    CloseEmaAnchorLong = 27,
    EntryEmaAnchorShort = 28,
    CloseEmaAnchorShort = 29,

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
                | EntryEmaAnchorLong
                | CloseEmaAnchorLong
        )
    }

    #[inline]
    pub const fn is_entry(self) -> bool {
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
                | EntryInitialNormalShort
                | EntryInitialPartialShort
                | EntryTrailingNormalShort
                | EntryTrailingCroppedShort
                | EntryGridNormalShort
                | EntryGridCroppedShort
                | EntryGridInflatedShort
                | EntryEmaAnchorLong
                | EntryEmaAnchorShort
        )
    }

    #[inline]
    pub const fn is_close(self) -> bool {
        use OrderType::*;
        matches!(
            self,
            CloseGridLong
                | CloseTrailingLong
                | CloseUnstuckLong
                | CloseAutoReduceTwelLong
                | CloseGridShort
                | CloseTrailingShort
                | CloseUnstuckShort
                | CloseAutoReduceTwelShort
                | ClosePanicLong
                | ClosePanicShort
                | CloseAutoReduceWelLong
                | CloseAutoReduceWelShort
                | CloseEmaAnchorLong
                | CloseEmaAnchorShort
        )
    }
}

#[derive(Default)]
pub struct Balance {
    pub usd_cash_wallet: f64,           // raw usd wallet balance
    pub usd_total_balance: f64,         // usd cash + btc converted to usd
    pub usd_total_balance_rounded: f64, // rounded usd total for hysteresis logic
    pub btc_cash_wallet: f64,           // raw btc wallet balance
    pub btc_total_balance: f64,         // btc cash + usd converted to btc
    pub use_btc_collateral: bool,       // whether to use btc as collateral
    // BTC market value / total wallet value; values above one use USD debt.
    pub btc_collateral_target_ratio: f64,
    // max(-USD cash, 0) / the same total wallet value when topping up BTC.
    pub max_usd_debt_to_wallet_value_ratio: Option<f64>,
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
    pub liquidity: String,
    pub wallet_exposure: f64, // signed: long positive, short negative
    pub twe_long: f64,
    pub twe_short: f64, // signed negative
    pub twe_net: f64,   // twe_long + twe_short
}

#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    pub adg_fraction: f64,
    pub mdg_fraction: f64,
    pub gain_multiplier: f64,
    pub liquidated: bool,
    pub backtest_completion_fraction: f64,
    pub daily_realized_pnl_mean_fraction: f64,
    pub daily_realized_pnl_median_fraction: f64,
    pub sharpe_ratio_realized_pnl: f64,
    pub sortino_ratio_realized_pnl: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub omega_ratio: f64,
    pub expected_shortfall_1pct_fraction: f64,
    pub calmar_ratio: f64,
    pub sterling_ratio: f64,
    pub drawdown_max_fraction: f64,
    pub drawdown_daily_worst_1pct_mean_fraction: f64,
    pub gain_multiplier_strategy_equity: f64,
    pub adg_fraction_strategy_equity: f64,
    pub adg_rolling_hmean_fraction_strategy_equity: f64,
    pub adg_time_integrated_fraction_strategy_equity: f64,
    pub positive_log_return_participation_score_strategy_equity: f64,
    pub mdg_fraction_strategy_equity: f64,
    pub sharpe_ratio_strategy_equity: f64,
    pub sortino_ratio_strategy_equity: f64,
    pub omega_ratio_strategy_equity: f64,
    pub expected_shortfall_1pct_fraction_strategy_equity: f64,
    pub calmar_ratio_strategy_equity: f64,
    pub sterling_ratio_strategy_equity: f64,
    pub drawdown_max_fraction_strategy_equity: f64,
    pub drawdown_max_fraction_strategy_equity_long: f64,
    pub drawdown_max_fraction_strategy_equity_short: f64,
    pub hsl_signal_ema_max_fraction: f64,
    pub hsl_signal_ema_max_fraction_long: f64,
    pub hsl_signal_ema_max_fraction_short: f64,
    pub drawdown_daily_worst_1pct_mean_fraction_strategy_equity: f64,
    pub drawdown_daily_worst_1pct_mean_fraction_strategy_equity_long: f64,
    pub drawdown_daily_worst_1pct_mean_fraction_strategy_equity_short: f64,
    pub hsl_signal_ema_worst_1pct_mean_fraction: f64,
    pub hsl_signal_ema_worst_1pct_mean_fraction_long: f64,
    pub hsl_signal_ema_worst_1pct_mean_fraction_short: f64,
    pub pain_index_fraction_strategy_equity: f64,
    pub ulcer_index_fraction_strategy_equity: f64,
    pub recovery_mean_days_strategy_equity: f64,
    pub recovery_median_days_strategy_equity: f64,
    pub recovery_p95_days_strategy_equity: f64,
    pub recovery_p99_days_strategy_equity: f64,
    pub recovery_worst_5pct_mean_days_strategy_equity: f64,
    pub recovery_worst_1pct_mean_days_strategy_equity: f64,
    pub recovery_max_days_strategy_equity: f64,
    pub recovery_max_hours_strategy_equity: f64,
    pub recovery_max_hours_strategy_equity_long: f64,
    pub recovery_max_hours_strategy_equity_short: f64,
    pub recovery_max_days_strategy_equity_long: f64,
    pub recovery_max_days_strategy_equity_short: f64,
    pub equity_balance_gap_negative_max_fraction: f64,
    pub equity_balance_gap_negative_mean_fraction: f64,
    pub equity_balance_gap_positive_max_fraction: f64,
    pub equity_balance_gap_positive_mean_fraction: f64,
    pub adg_to_paper_loss_max_ratio: f64,
    pub adg_to_paper_loss_mean_ratio: f64,
    pub adg_to_absolute_net_exposure_max_ratio: f64,
    pub adg_to_absolute_net_exposure_mean_ratio: f64,
    pub loss_profit_ratio: f64,
    pub loss_profit_ratio_long: f64,
    pub loss_profit_ratio_short: f64,
    pub long_pnl_to_total_pnl_ratio: f64,
    pub peak_recovery_max_hours: f64,
    pub peak_recovery_max_hours_realized_pnl: f64,
    pub peak_recovery_max_days: f64,
    pub peak_recovery_max_days_realized_pnl: f64,

    pub equity_choppiness_ratio: f64,
    pub equity_jerkiness_ratio: f64,
    pub exponential_fit_mse_log_ratio_squared: f64,

    pub position_episodes_per_day: f64,
    pub position_held_time_weighted_mean_hours: f64,
    pub position_held_mean_hours: f64,
    pub position_held_max_hours: f64,
    pub position_held_median_hours: f64,
    pub position_unchanged_max_hours: f64,
    pub position_held_mean_days: f64,
    pub position_held_max_days: f64,
    pub position_held_median_days: f64,
    pub position_unchanged_max_days: f64,
    pub trade_win_fraction: f64,
    pub trade_loss_max_fraction: f64,
    pub trade_loss_mean_fraction: f64,
    pub trade_loss_median_fraction: f64,
    pub fills_active_days_count: f64,
    pub fills_active_day_fraction: f64,
    pub fills_active_symbols_count: f64,
    pub analysis_duration_days: f64,
    pub fills_count: f64,
    pub fills_close_count: f64,
    pub fills_entry_count: f64,
    pub fills_count_long: f64,
    pub fills_count_short: f64,
    pub entry_to_close_fill_count_ratio: f64,
    pub fills_gap_max_days: f64,
    pub fills_gap_mean_hours: f64,
    pub fills_gap_median_hours: f64,
    pub fills_gap_p95_hours: f64,
    pub fills_gap_p99_hours: f64,
    pub fills_gap_time_weighted_mean_hours: f64,
    pub fills_per_day: f64,
    pub fills_close_per_day: f64,
    pub fills_entry_per_day: f64,
    pub fills_per_day_long: f64,
    pub fills_per_day_per_position_slot: f64,
    pub fills_per_day_per_position_slot_long: f64,
    pub fills_per_day_per_position_slot_short: f64,
    pub fills_per_day_short: f64,
    pub fills_top_symbol_fraction: f64,

    pub active_day_turnover_proxy_mean_ratio: f64,

    pub absolute_net_wallet_exposure_max_ratio: f64,
    pub absolute_net_wallet_exposure_mean_ratio: f64,
    pub wallet_exposure_rms_ratio_long: f64,
    pub wallet_exposure_rms_ratio_short: f64,
    pub gross_wallet_exposure_rms_ratio: f64,
    pub wallet_exposure_utilization_rms_ratio_long: f64,
    pub wallet_exposure_utilization_rms_ratio_short: f64,
    pub position_exposure_utilization_rms_ratio_long: f64,
    pub position_exposure_utilization_rms_ratio_short: f64,
    pub absolute_net_wallet_exposure_median_ratio: f64,
    pub high_exposure_mean_hours_long: f64,
    pub high_exposure_max_hours_long: f64,
    pub high_exposure_mean_hours_short: f64,
    pub high_exposure_max_hours_short: f64,
    pub high_exposure_mean_days_long: f64,
    pub high_exposure_max_days_long: f64,
    pub high_exposure_mean_days_short: f64,
    pub high_exposure_max_days_short: f64,
    pub initial_entry_balance_fraction_long: f64,
    pub initial_entry_balance_fraction_short: f64,
    pub entry_interval_mean_hours: f64,
    pub entry_interval_median_hours: f64,
    pub entry_interval_p95_hours: f64,
    pub entry_interval_p99_hours: f64,
    pub entry_interval_max_hours: f64,

    pub hsl_trigger_count: u32,
    pub hsl_triggers_per_year: f64,
    pub hsl_trigger_count_long: u32,
    pub hsl_trigger_count_short: u32,
    pub hsl_panic_loss_starting_balance_fraction: f64,
    pub hsl_restart_count: u32,
    pub hsl_restarts_per_year: f64,
    pub hsl_restarts_per_year_long: f64,
    pub hsl_restarts_per_year_short: f64,
    pub hsl_restart_count_long: u32,
    pub hsl_restart_count_short: u32,
    pub hsl_red_time_fraction: f64,
    pub hsl_red_duration_mean_minutes: f64,
    pub hsl_red_duration_max_minutes: f64,
    pub hsl_trigger_signal_mean_fraction: f64,
    pub hsl_panic_fill_loss_sum_usd: f64,
    pub hsl_panic_fill_loss_max_usd: f64,
    pub hsl_panic_episode_loss_min_fraction: f64,
    pub hsl_panic_episode_loss_mean_fraction: f64,
    pub hsl_panic_episode_loss_max_fraction: f64,
    pub hsl_flatten_time_mean_minutes: f64,
    pub hsl_restart_retrigger_fraction: f64,
}

impl Default for Analysis {
    fn default() -> Self {
        Analysis {
            adg_fraction: 0.0,
            mdg_fraction: 0.0,
            gain_multiplier: 0.0,
            liquidated: false,
            backtest_completion_fraction: 0.0,
            daily_realized_pnl_mean_fraction: 0.0,
            daily_realized_pnl_median_fraction: 0.0,
            sharpe_ratio_realized_pnl: 0.0,
            sortino_ratio_realized_pnl: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            omega_ratio: 0.0,
            expected_shortfall_1pct_fraction: 0.0,
            calmar_ratio: 0.0,
            sterling_ratio: 0.0,
            drawdown_max_fraction: 1.0,
            drawdown_daily_worst_1pct_mean_fraction: 1.0,
            gain_multiplier_strategy_equity: 0.0,
            adg_fraction_strategy_equity: 0.0,
            adg_rolling_hmean_fraction_strategy_equity: 0.0,
            adg_time_integrated_fraction_strategy_equity: 0.0,
            positive_log_return_participation_score_strategy_equity: 0.0,
            mdg_fraction_strategy_equity: 0.0,
            sharpe_ratio_strategy_equity: 0.0,
            sortino_ratio_strategy_equity: 0.0,
            omega_ratio_strategy_equity: 0.0,
            expected_shortfall_1pct_fraction_strategy_equity: 0.0,
            calmar_ratio_strategy_equity: 0.0,
            sterling_ratio_strategy_equity: 0.0,
            drawdown_max_fraction_strategy_equity: 0.0,
            drawdown_max_fraction_strategy_equity_long: 0.0,
            drawdown_max_fraction_strategy_equity_short: 0.0,
            hsl_signal_ema_max_fraction: 0.0,
            hsl_signal_ema_max_fraction_long: 0.0,
            hsl_signal_ema_max_fraction_short: 0.0,
            drawdown_daily_worst_1pct_mean_fraction_strategy_equity: 0.0,
            drawdown_daily_worst_1pct_mean_fraction_strategy_equity_long: 0.0,
            drawdown_daily_worst_1pct_mean_fraction_strategy_equity_short: 0.0,
            hsl_signal_ema_worst_1pct_mean_fraction: 0.0,
            hsl_signal_ema_worst_1pct_mean_fraction_long: 0.0,
            hsl_signal_ema_worst_1pct_mean_fraction_short: 0.0,
            pain_index_fraction_strategy_equity: 0.0,
            ulcer_index_fraction_strategy_equity: 0.0,
            recovery_mean_days_strategy_equity: 0.0,
            recovery_median_days_strategy_equity: 0.0,
            recovery_p95_days_strategy_equity: 0.0,
            recovery_p99_days_strategy_equity: 0.0,
            recovery_worst_5pct_mean_days_strategy_equity: 0.0,
            recovery_worst_1pct_mean_days_strategy_equity: 0.0,
            recovery_max_days_strategy_equity: 0.0,
            recovery_max_hours_strategy_equity: 0.0,
            recovery_max_hours_strategy_equity_long: 0.0,
            recovery_max_hours_strategy_equity_short: 0.0,
            recovery_max_days_strategy_equity_long: 0.0,
            recovery_max_days_strategy_equity_short: 0.0,
            equity_balance_gap_negative_max_fraction: 1.0,
            equity_balance_gap_negative_mean_fraction: 1.0,
            equity_balance_gap_positive_max_fraction: 1.0,
            equity_balance_gap_positive_mean_fraction: 1.0,
            adg_to_paper_loss_max_ratio: 0.0,
            adg_to_paper_loss_mean_ratio: 0.0,
            adg_to_absolute_net_exposure_max_ratio: 0.0,
            adg_to_absolute_net_exposure_mean_ratio: 0.0,
            loss_profit_ratio: 1.0,
            loss_profit_ratio_long: 1.0,
            loss_profit_ratio_short: 1.0,
            long_pnl_to_total_pnl_ratio: 0.5,
            peak_recovery_max_hours: 0.0,
            peak_recovery_max_hours_realized_pnl: 0.0,
            peak_recovery_max_days: 0.0,
            peak_recovery_max_days_realized_pnl: 0.0,
            equity_choppiness_ratio: 1.0,
            equity_jerkiness_ratio: 1.0,
            exponential_fit_mse_log_ratio_squared: 1.0,
            position_episodes_per_day: 0.0,
            position_held_time_weighted_mean_hours: 0.0,
            position_held_mean_hours: 0.0,
            position_held_max_hours: 0.0,
            position_held_median_hours: 0.0,
            position_unchanged_max_hours: 0.0,
            position_held_mean_days: 0.0,
            position_held_max_days: 0.0,
            position_held_median_days: 0.0,
            position_unchanged_max_days: 0.0,
            trade_win_fraction: 0.0,
            trade_loss_max_fraction: 0.0,
            trade_loss_mean_fraction: 0.0,
            trade_loss_median_fraction: 0.0,
            fills_active_days_count: 0.0,
            fills_active_day_fraction: 0.0,
            fills_active_symbols_count: 0.0,
            analysis_duration_days: 0.0,
            fills_count: 0.0,
            fills_close_count: 0.0,
            fills_entry_count: 0.0,
            fills_count_long: 0.0,
            fills_count_short: 0.0,
            entry_to_close_fill_count_ratio: 0.0,
            fills_gap_max_days: 0.0,
            fills_gap_mean_hours: 0.0,
            fills_gap_median_hours: 0.0,
            fills_gap_p95_hours: 0.0,
            fills_gap_p99_hours: 0.0,
            fills_gap_time_weighted_mean_hours: 0.0,
            fills_per_day: 0.0,
            fills_close_per_day: 0.0,
            fills_entry_per_day: 0.0,
            fills_per_day_long: 0.0,
            fills_per_day_per_position_slot: 0.0,
            fills_per_day_per_position_slot_long: 0.0,
            fills_per_day_per_position_slot_short: 0.0,
            fills_per_day_short: 0.0,
            fills_top_symbol_fraction: 0.0,
            active_day_turnover_proxy_mean_ratio: 0.0,
            absolute_net_wallet_exposure_max_ratio: 0.0,
            absolute_net_wallet_exposure_mean_ratio: 0.0,
            wallet_exposure_rms_ratio_long: f64::NAN,
            wallet_exposure_rms_ratio_short: f64::NAN,
            gross_wallet_exposure_rms_ratio: f64::NAN,
            wallet_exposure_utilization_rms_ratio_long: f64::NAN,
            wallet_exposure_utilization_rms_ratio_short: f64::NAN,
            position_exposure_utilization_rms_ratio_long: f64::NAN,
            position_exposure_utilization_rms_ratio_short: f64::NAN,
            absolute_net_wallet_exposure_median_ratio: 0.0,
            high_exposure_mean_hours_long: 0.0,
            high_exposure_max_hours_long: 0.0,
            high_exposure_mean_hours_short: 0.0,
            high_exposure_max_hours_short: 0.0,
            high_exposure_mean_days_long: 0.0,
            high_exposure_max_days_long: 0.0,
            high_exposure_mean_days_short: 0.0,
            high_exposure_max_days_short: 0.0,
            initial_entry_balance_fraction_long: 0.0,
            initial_entry_balance_fraction_short: 0.0,
            entry_interval_mean_hours: 0.0,
            entry_interval_median_hours: 0.0,
            entry_interval_p95_hours: 0.0,
            entry_interval_p99_hours: 0.0,
            entry_interval_max_hours: 0.0,
            hsl_trigger_count: 0,
            hsl_triggers_per_year: 0.0,
            hsl_trigger_count_long: 0,
            hsl_trigger_count_short: 0,
            hsl_panic_loss_starting_balance_fraction: 0.0,
            hsl_restart_count: 0,
            hsl_restarts_per_year: 0.0,
            hsl_restarts_per_year_long: 0.0,
            hsl_restarts_per_year_short: 0.0,
            hsl_restart_count_long: 0,
            hsl_restart_count_short: 0,
            hsl_red_time_fraction: 0.0,
            hsl_red_duration_mean_minutes: 0.0,
            hsl_red_duration_max_minutes: 0.0,
            hsl_trigger_signal_mean_fraction: 0.0,
            hsl_panic_fill_loss_sum_usd: 0.0,
            hsl_panic_fill_loss_max_usd: 0.0,
            hsl_panic_episode_loss_min_fraction: 0.0,
            hsl_panic_episode_loss_mean_fraction: 0.0,
            hsl_panic_episode_loss_max_fraction: 0.0,
            hsl_flatten_time_mean_minutes: 0.0,
            hsl_restart_retrigger_fraction: 0.0,
        }
    }
}

fn default_unilateralness_span() -> f64 {
    60.0
}
