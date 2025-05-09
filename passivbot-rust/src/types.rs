use serde::Serialize;
use std::collections::HashMap;
use std::fmt;

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

impl Order {
    pub fn new(qty: f64, price: f64, order_type: OrderType) -> Self {
        Order {
            qty,
            price,
            order_type,
        }
    }
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
}

#[derive(Clone, Default, Debug)]
pub struct BotParamsPair {
    pub long: BotParams,
    pub short: BotParams,
}

#[derive(Clone, Default, Debug)]
pub struct BotParams {
    pub close_grid_markup_range: f64,
    pub close_grid_min_markup: f64,
    pub close_grid_qty_pct: f64,
    pub close_trailing_retracement_pct: f64,
    pub close_trailing_grid_ratio: f64,
    pub close_trailing_qty_pct: f64,
    pub close_trailing_threshold_pct: f64,
    pub enforce_exposure_limit: bool,
    pub entry_grid_double_down_factor: f64,
    pub entry_grid_spacing_weight: f64,
    pub entry_grid_spacing_pct: f64,
    pub entry_initial_ema_dist: f64,
    pub entry_initial_qty_pct: f64,
    pub entry_trailing_double_down_factor: f64,
    pub entry_trailing_retracement_pct: f64,
    pub entry_trailing_grid_ratio: f64,
    pub entry_trailing_threshold_pct: f64,
    pub filter_noisiness_rolling_window: usize,
    pub filter_volume_rolling_window: usize,
    pub filter_volume_drop_pct: f64,
    pub ema_span_0: f64,
    pub ema_span_1: f64,
    pub n_positions: usize,
    pub total_wallet_exposure_limit: f64,
    pub wallet_exposure_limit: f64, // is total_wallet_exposure_limit / n_positions
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OrderType {
    EntryInitialNormalLong,
    EntryInitialPartialLong,
    EntryTrailingNormalLong,
    EntryTrailingCroppedLong,
    EntryGridNormalLong,
    EntryGridCroppedLong,
    EntryGridInflatedLong,

    CloseGridLong,
    CloseTrailingLong,
    CloseUnstuckLong,
    CloseAutoReduceLong,

    EntryInitialNormalShort,
    EntryInitialPartialShort,
    EntryTrailingNormalShort,
    EntryTrailingCroppedShort,
    EntryGridNormalShort,
    EntryGridCroppedShort,
    EntryGridInflatedShort,

    CloseGridShort,
    CloseTrailingShort,
    CloseUnstuckShort,
    CloseAutoReduceShort,

    Empty,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OrderType::EntryInitialNormalLong => write!(f, "entry_initial_normal_long"),
            OrderType::EntryInitialPartialLong => write!(f, "entry_initial_partial_long"),
            OrderType::EntryTrailingNormalLong => write!(f, "entry_trailing_normal_long"),
            OrderType::EntryTrailingCroppedLong => write!(f, "entry_trailing_cropped_long"),
            OrderType::EntryGridNormalLong => write!(f, "entry_grid_normal_long"),
            OrderType::EntryGridCroppedLong => write!(f, "entry_grid_cropped_long"),
            OrderType::EntryGridInflatedLong => write!(f, "entry_grid_inflated_long"),
            OrderType::CloseGridLong => write!(f, "close_grid_long"),
            OrderType::CloseTrailingLong => write!(f, "close_trailing_long"),
            OrderType::CloseUnstuckLong => write!(f, "close_unstuck_long"),
            OrderType::CloseAutoReduceLong => write!(f, "close_auto_reduce_long"),
            OrderType::EntryInitialNormalShort => write!(f, "entry_initial_normal_short"),
            OrderType::EntryInitialPartialShort => write!(f, "entry_initial_partial_short"),
            OrderType::EntryTrailingNormalShort => write!(f, "entry_trailing_normal_short"),
            OrderType::EntryTrailingCroppedShort => write!(f, "entry_trailing_cropped_short"),
            OrderType::EntryGridNormalShort => write!(f, "entry_grid_normal_short"),
            OrderType::EntryGridCroppedShort => write!(f, "entry_grid_cropped_short"),
            OrderType::EntryGridInflatedShort => write!(f, "entry_grid_inflated_short"),
            OrderType::CloseGridShort => write!(f, "close_grid_short"),
            OrderType::CloseTrailingShort => write!(f, "close_trailing_short"),
            OrderType::CloseUnstuckShort => write!(f, "close_unstuck_short"),
            OrderType::CloseAutoReduceShort => write!(f, "close_auto_reduce_short"),
            OrderType::Empty => write!(f, "empty"),
        }
    }
}

#[derive(Default)]
pub struct Balance {
    pub usd: f64,                    // usd balance
    pub usd_total: f64,              // total in usd
    pub usd_total_rounded: f64,      // total in usd rounded for calculations
    pub usd_total_rounded_last: f64, // last usd total rounded for hysterisis
    pub btc: f64,                    // btc balance
    pub btc_total: f64,              // total in btc
    pub use_btc_collateral: bool,    // whether to use btc as collateral
}

#[derive(Default, Clone)]
pub struct Equities {
    pub usd: Vec<f64>,
    pub btc: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub index: usize,
    pub coin: String,
    pub pnl: f64,
    pub fee_paid: f64,
    pub balance_usd_total: f64,
    pub balance_btc: f64, // Added: BTC balance after fill
    pub balance_usd: f64, // Added: USD balance after fill
    pub btc_price: f64,   // Added: BTC/USD price at time of fill
    pub fill_qty: f64,
    pub fill_price: f64,
    pub position_size: f64,
    pub position_price: f64,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    pub adg: f64,
    pub mdg: f64,
    pub gain: f64,
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
    pub mdg_w: f64,
    pub sharpe_ratio_w: f64,
    pub sortino_ratio_w: f64,
    pub omega_ratio_w: f64,
    pub calmar_ratio_w: f64,
    pub sterling_ratio_w: f64,
    pub loss_profit_ratio_w: f64,
    pub volume_pct_per_day_avg: f64,
    pub volume_pct_per_day_avg_w: f64,
}

impl Default for Analysis {
    fn default() -> Self {
        Analysis {
            adg: 0.0,
            mdg: 0.0,
            gain: 0.0,
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
            equity_choppiness: 1.0,
            equity_jerkiness: 1.0,
            exponential_fit_error: 1.0,
            positions_held_per_day: 0.0,
            position_held_hours_mean: 0.0,
            position_held_hours_max: 0.0,
            position_held_hours_median: 0.0,
            position_unchanged_hours_max: 0.0,
            adg_w: 0.0,
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
        }
    }
}
