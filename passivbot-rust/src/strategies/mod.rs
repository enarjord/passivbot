pub mod ema_anchor;
pub mod registry;
pub mod spec;
pub mod trailing_grid;

use crate::types::{
    BotParams, ExchangeParams, Order, Position, RuntimeBudgetState, StateParams,
    TrailingPriceBundle,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StrategyKind {
    #[default]
    TrailingGrid,
    EmaAnchor,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GridClosePriceAnchor {
    #[default]
    PositionPrice,
    EmaBandUpper,
    EmaBandLower,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct TrailingGridParams {
    pub close_grid_markup_end: f64,
    pub close_grid_markup_start: f64,
    pub close_grid_qty_pct: f64,
    pub grid_close_price_anchor: GridClosePriceAnchor,
    pub close_trailing_grid_ratio: f64,
    pub close_trailing_qty_pct: f64,
    pub close_trailing_retracement_pct: f64,
    pub close_trailing_threshold_pct: f64,
    pub close_weight_volatility_1h: f64,
    pub close_weight_volatility_1m: f64,
    pub ema_span_0: f64,
    pub ema_span_1: f64,
    pub entry_grid_double_down_factor: f64,
    pub entry_grid_spacing_pct: f64,
    pub entry_initial_ema_dist: f64,
    pub entry_initial_qty_pct: f64,
    pub entry_trailing_double_down_factor: f64,
    pub entry_trailing_grid_ratio: f64,
    pub entry_trailing_retracement_pct: f64,
    pub entry_trailing_threshold_pct: f64,
    pub entry_volatility_ema_span_hours: f64,
    pub entry_volatility_ema_span_minutes: f64,
    pub entry_weight_volatility_1h: f64,
    pub entry_weight_volatility_1m: f64,
    pub entry_we_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailingGridEntryParams {
    pub entry_grid_double_down_factor: f64,
    pub entry_grid_spacing_pct: f64,
    pub entry_initial_ema_dist: f64,
    pub entry_initial_qty_pct: f64,
    pub entry_trailing_double_down_factor: f64,
    pub entry_trailing_grid_ratio: f64,
    pub entry_trailing_retracement_pct: f64,
    pub entry_trailing_threshold_pct: f64,
    pub entry_weight_volatility_1h: f64,
    pub entry_weight_volatility_1m: f64,
    pub entry_we_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailingGridCloseParams {
    pub close_grid_markup_end: f64,
    pub close_grid_markup_start: f64,
    pub close_grid_qty_pct: f64,
    pub grid_close_price_anchor: GridClosePriceAnchor,
    pub close_trailing_grid_ratio: f64,
    pub close_trailing_qty_pct: f64,
    pub close_trailing_retracement_pct: f64,
    pub close_trailing_threshold_pct: f64,
    pub close_weight_volatility_1h: f64,
    pub close_weight_volatility_1m: f64,
}

impl TrailingGridParams {
    pub fn from_bot_params(bot_params: &BotParams) -> Self {
        Self {
            close_grid_markup_end: bot_params.close_grid_markup_end,
            close_grid_markup_start: bot_params.close_grid_markup_start,
            close_grid_qty_pct: bot_params.close_grid_qty_pct,
            grid_close_price_anchor: GridClosePriceAnchor::PositionPrice,
            close_trailing_grid_ratio: bot_params.close_trailing_grid_ratio,
            close_trailing_qty_pct: bot_params.close_trailing_qty_pct,
            close_trailing_retracement_pct: bot_params.close_trailing_retracement_pct,
            close_trailing_threshold_pct: bot_params.close_trailing_threshold_pct,
            close_weight_volatility_1h: bot_params.close_weight_volatility_1h,
            close_weight_volatility_1m: bot_params.close_weight_volatility_1m,
            ema_span_0: bot_params.ema_span_0,
            ema_span_1: bot_params.ema_span_1,
            entry_grid_double_down_factor: bot_params.entry_grid_double_down_factor,
            entry_grid_spacing_pct: bot_params.entry_grid_spacing_pct,
            entry_initial_ema_dist: bot_params.entry_initial_ema_dist,
            entry_initial_qty_pct: bot_params.entry_initial_qty_pct,
            entry_trailing_double_down_factor: bot_params.entry_trailing_double_down_factor,
            entry_trailing_grid_ratio: bot_params.entry_trailing_grid_ratio,
            entry_trailing_retracement_pct: bot_params.entry_trailing_retracement_pct,
            entry_trailing_threshold_pct: bot_params.entry_trailing_threshold_pct,
            entry_volatility_ema_span_hours: bot_params.entry_volatility_ema_span_hours,
            entry_volatility_ema_span_minutes: bot_params.entry_volatility_ema_span_minutes,
            entry_weight_volatility_1h: bot_params.entry_weight_volatility_1h,
            entry_weight_volatility_1m: bot_params.entry_weight_volatility_1m,
            entry_we_weight: bot_params.entry_we_weight,
        }
    }

    pub fn entry_params(&self) -> TrailingGridEntryParams {
        TrailingGridEntryParams {
            entry_grid_double_down_factor: self.entry_grid_double_down_factor,
            entry_grid_spacing_pct: self.entry_grid_spacing_pct,
            entry_initial_ema_dist: self.entry_initial_ema_dist,
            entry_initial_qty_pct: self.entry_initial_qty_pct,
            entry_trailing_double_down_factor: self.entry_trailing_double_down_factor,
            entry_trailing_grid_ratio: self.entry_trailing_grid_ratio,
            entry_trailing_retracement_pct: self.entry_trailing_retracement_pct,
            entry_trailing_threshold_pct: self.entry_trailing_threshold_pct,
            entry_weight_volatility_1h: self.entry_weight_volatility_1h,
            entry_weight_volatility_1m: self.entry_weight_volatility_1m,
            entry_we_weight: self.entry_we_weight,
        }
    }

    pub fn close_params(&self) -> TrailingGridCloseParams {
        TrailingGridCloseParams {
            close_grid_markup_end: self.close_grid_markup_end,
            close_grid_markup_start: self.close_grid_markup_start,
            close_grid_qty_pct: self.close_grid_qty_pct,
            grid_close_price_anchor: self.grid_close_price_anchor,
            close_trailing_grid_ratio: self.close_trailing_grid_ratio,
            close_trailing_qty_pct: self.close_trailing_qty_pct,
            close_trailing_retracement_pct: self.close_trailing_retracement_pct,
            close_trailing_threshold_pct: self.close_trailing_threshold_pct,
            close_weight_volatility_1h: self.close_weight_volatility_1h,
            close_weight_volatility_1m: self.close_weight_volatility_1m,
        }
    }

    pub fn to_value(&self) -> Value {
        serde_json::json!({
            "close_grid_markup_end": self.close_grid_markup_end,
            "close_grid_markup_start": self.close_grid_markup_start,
            "close_grid_qty_pct": self.close_grid_qty_pct,
            "grid_close_price_anchor": self.grid_close_price_anchor,
            "close_trailing_grid_ratio": self.close_trailing_grid_ratio,
            "close_trailing_qty_pct": self.close_trailing_qty_pct,
            "close_trailing_retracement_pct": self.close_trailing_retracement_pct,
            "close_trailing_threshold_pct": self.close_trailing_threshold_pct,
            "close_weight_volatility_1h": self.close_weight_volatility_1h,
            "close_weight_volatility_1m": self.close_weight_volatility_1m,
            "ema_span_0": self.ema_span_0,
            "ema_span_1": self.ema_span_1,
            "entry_grid_double_down_factor": self.entry_grid_double_down_factor,
            "entry_grid_spacing_pct": self.entry_grid_spacing_pct,
            "entry_initial_ema_dist": self.entry_initial_ema_dist,
            "entry_initial_qty_pct": self.entry_initial_qty_pct,
            "entry_trailing_double_down_factor": self.entry_trailing_double_down_factor,
            "entry_trailing_grid_ratio": self.entry_trailing_grid_ratio,
            "entry_trailing_retracement_pct": self.entry_trailing_retracement_pct,
            "entry_trailing_threshold_pct": self.entry_trailing_threshold_pct,
            "entry_volatility_ema_span_hours": self.entry_volatility_ema_span_hours,
            "entry_volatility_ema_span_minutes": self.entry_volatility_ema_span_minutes,
            "entry_weight_volatility_1h": self.entry_weight_volatility_1h,
            "entry_weight_volatility_1m": self.entry_weight_volatility_1m,
            "entry_we_weight": self.entry_we_weight,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EmaAnchorParams {
    pub base_qty_pct: f64,
    pub ema_span_0: f64,
    pub ema_span_1: f64,
    pub entry_double_down_factor: f64,
    pub offset: f64,
    pub offset_volatility_ema_span_minutes: f64,
    pub offset_volatility_1m_weight: f64,
    pub entry_volatility_ema_span_hours: f64,
    pub offset_volatility_1h_weight: f64,
    pub offset_psize_weight: f64,
}

impl Default for EmaAnchorParams {
    fn default() -> Self {
        Self {
            base_qty_pct: 0.01,
            ema_span_0: 200.0,
            ema_span_1: 800.0,
            entry_double_down_factor: 0.0,
            offset: 0.002,
            offset_volatility_ema_span_minutes: 60.0,
            offset_volatility_1m_weight: 0.0,
            entry_volatility_ema_span_hours: 24.0,
            offset_volatility_1h_weight: 0.0,
            offset_psize_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrategyParams {
    TrailingGrid(TrailingGridParams),
    EmaAnchor(EmaAnchorParams),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategySide {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PeekBehavior {
    pub expand_entries: bool,
    pub expand_closes: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct NextStepHint {
    pub low: f64,
    pub high: f64,
    pub tradable: bool,
}

#[derive(Debug, Default)]
pub struct GeneratedOrders {
    pub entries: Vec<Order>,
    pub closes: Vec<Order>,
}

pub struct StrategyRequest<'a> {
    pub wants_entries: bool,
    pub wants_closes: bool,
    pub exchange: &'a ExchangeParams,
    pub state: &'a StateParams,
    pub bot_params: &'a BotParams,
    pub strategy_params: &'a StrategyParams,
    pub runtime_budget: RuntimeBudgetState,
    pub position: &'a Position,
    pub trailing: &'a TrailingPriceBundle,
    pub next_candle: Option<NextStepHint>,
    pub peek: Option<PeekBehavior>,
}

pub fn parse_strategy_params(
    kind: StrategyKind,
    side: StrategySide,
    raw: Option<&Value>,
    _bot_params: &BotParams,
) -> Result<StrategyParams, String> {
    match kind {
        StrategyKind::TrailingGrid => {
            let value = raw.ok_or_else(|| {
                "trailing_grid requires per-side strategy_params in orchestrator input".to_string()
            })?;
            let params = serde_json::from_value::<TrailingGridParams>(value.clone())
                .map_err(|err| format!("failed to parse trailing_grid strategy params: {err}"))?;
            validate_grid_close_price_anchor(side, params.grid_close_price_anchor)?;
            Ok(StrategyParams::TrailingGrid(params))
        }
        StrategyKind::EmaAnchor => {
            let value = raw.ok_or_else(|| {
                "ema_anchor requires per-side strategy_params in orchestrator input".to_string()
            })?;
            serde_json::from_value::<EmaAnchorParams>(value.clone())
                .map(StrategyParams::EmaAnchor)
                .map_err(|err| format!("failed to parse ema_anchor strategy params: {err}"))
        }
    }
}

pub fn parse_grid_close_price_anchor_value(
    side: StrategySide,
    raw: &str,
) -> Result<GridClosePriceAnchor, String> {
    let normalized = raw.trim().to_lowercase();
    match (side, normalized.as_str()) {
        (_, "" | "position_price" | "pprice") => Ok(GridClosePriceAnchor::PositionPrice),
        (StrategySide::Long, "ema_band" | "ema_band_upper") => {
            Ok(GridClosePriceAnchor::EmaBandUpper)
        }
        (StrategySide::Short, "ema_band" | "ema_band_lower") => {
            Ok(GridClosePriceAnchor::EmaBandLower)
        }
        (StrategySide::Long, _) => {
            Err(
                "long trailing_grid grid_close_price_anchor must be one of {'position_price', 'pprice', 'ema_band', 'ema_band_upper'}".to_string()
            )
        }
        (StrategySide::Short, _) => {
            Err(
                "short trailing_grid grid_close_price_anchor must be one of {'position_price', 'pprice', 'ema_band', 'ema_band_lower'}".to_string()
            )
        }
    }
}

pub fn validate_grid_close_price_anchor(
    side: StrategySide,
    anchor: GridClosePriceAnchor,
) -> Result<(), String> {
    match (side, anchor) {
        (StrategySide::Long, GridClosePriceAnchor::EmaBandLower) => Err(
            "long trailing_grid grid_close_price_anchor must be position_price or ema_band_upper"
                .to_string(),
        ),
        (StrategySide::Short, GridClosePriceAnchor::EmaBandUpper) => Err(
            "short trailing_grid grid_close_price_anchor must be position_price or ema_band_lower"
                .to_string(),
        ),
        _ => Ok(()),
    }
}

pub fn strategy_ema_spans(params: &StrategyParams) -> (f64, f64) {
    match params {
        StrategyParams::TrailingGrid(params) => (params.ema_span_0, params.ema_span_1),
        StrategyParams::EmaAnchor(params) => (params.ema_span_0, params.ema_span_1),
    }
}

pub fn strategy_entry_volatility_span_hours(params: &StrategyParams) -> Option<f64> {
    match params {
        StrategyParams::TrailingGrid(params) => Some(params.entry_volatility_ema_span_hours),
        StrategyParams::EmaAnchor(params) => Some(params.entry_volatility_ema_span_hours),
    }
}

pub fn strategy_offset_volatility_span_minutes(params: &StrategyParams) -> Option<f64> {
    match params {
        StrategyParams::TrailingGrid(params) => Some(params.entry_volatility_ema_span_minutes),
        StrategyParams::EmaAnchor(params) => Some(params.offset_volatility_ema_span_minutes),
    }
}

pub fn strategy_initial_entry_offset(params: &StrategyParams) -> f64 {
    match params {
        StrategyParams::TrailingGrid(params) => params.entry_initial_ema_dist,
        StrategyParams::EmaAnchor(params) => params.offset,
    }
}

pub fn strategy_needs_log_range_1m(params: &StrategyParams) -> bool {
    match params {
        StrategyParams::TrailingGrid(params) => {
            (params.entry_weight_volatility_1m != 0.0 || params.close_weight_volatility_1m != 0.0)
                && params.entry_volatility_ema_span_minutes > 0.0
        }
        StrategyParams::EmaAnchor(params) => {
            params.offset_volatility_1m_weight != 0.0
                && params.offset_volatility_ema_span_minutes > 0.0
        }
    }
}

pub fn strategy_needs_log_range_1h(params: &StrategyParams) -> bool {
    match params {
        StrategyParams::TrailingGrid(params) => {
            (params.entry_weight_volatility_1h != 0.0 || params.close_weight_volatility_1h != 0.0)
                && params.entry_volatility_ema_span_hours > 0.0
        }
        StrategyParams::EmaAnchor(params) => {
            params.offset_volatility_1h_weight != 0.0
                && params.entry_volatility_ema_span_hours > 0.0
        }
    }
}

pub fn strategy_has_trailing(params: &StrategyParams) -> bool {
    match params {
        StrategyParams::TrailingGrid(params) => {
            params.close_trailing_grid_ratio != 0.0 || params.entry_trailing_grid_ratio != 0.0
        }
        StrategyParams::EmaAnchor(_) => false,
    }
}

pub fn generate_orders(
    kind: StrategyKind,
    side: StrategySide,
    request: StrategyRequest<'_>,
) -> GeneratedOrders {
    match kind {
        StrategyKind::TrailingGrid => trailing_grid::generate_orders(side, request),
        StrategyKind::EmaAnchor => ema_anchor::generate_orders(side, request),
    }
}
