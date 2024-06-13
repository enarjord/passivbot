use std::fmt;

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

pub struct Position {
    pub size: f64,
    pub price: f64,
}

#[derive(Debug, Default, Clone)]
pub struct EMABands {
    pub upper: f64,
    pub lower: f64,
}

#[derive(Debug)]
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
}

#[derive(Clone, Default)]
pub struct BotParamsLongShort {
    pub long: BotParams,
    pub short: BotParams,
}

#[derive(Clone, Default)]
pub struct BotParams {
    pub close_grid_markup_range: f64,
    pub close_grid_min_markup: f64,
    pub close_grid_qty_pct: f64,
    pub close_trailing_drawdown_pct: f64,
    pub close_trailing_grid_ratio: f64,
    pub close_trailing_threshold_pct: f64,
    pub entry_grid_double_down_factor: f64,
    pub entry_grid_spacing_weight: f64,
    pub entry_grid_spacing_pct: f64,
    pub entry_initial_ema_dist: f64,
    pub entry_initial_qty_pct: f64,
    pub entry_trailing_drawdown_pct: f64,
    pub entry_trailing_grid_ratio: f64,
    pub entry_trailing_threshold_pct: f64,
    pub ema_span0: f64,
    pub ema_span1: f64,
    pub n_positions: usize,
    pub total_wallet_exposure_limit: f64,
    pub wallet_exposure_limit: f64, // is total_wallet_exposure_limit / n_positions
    pub unstuck_close_pct: f64,
    pub unstuck_ema_dist: f64,
    pub unstuck_loss_allowance_pct: f64,
    pub unstuck_threshold: f64,
}

#[derive(Debug, PartialEq, Eq)]
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
            OrderType::Empty => write!(f, "empty"),
        }
    }
}
