pub mod ema_anchor;
pub mod registry;
pub mod spec;
pub mod trailing_martingale;

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
    #[serde(rename = "trailing_martingale")]
    TrailingMartingale,
    EmaAnchor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EmaGateMode {
    Disabled,
    All,
    #[default]
    Initial,
    Reentry,
}

impl EmaGateMode {
    pub fn gates_initial(self) -> bool {
        matches!(self, Self::All | Self::Initial)
    }

    pub fn gates_reentry(self) -> bool {
        matches!(self, Self::All | Self::Reentry)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct TrailingMartingaleParams {
    pub log_range_ema_span_1h_candles: f64,
    pub log_range_ema_span_minutes: f64,
    pub entry: TrailingMartingaleEntryParams,
    pub close: TrailingMartingaleCloseParams,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct TrailingMartingaleEntryParams {
    pub ema_span_0_minutes: f64,
    pub ema_span_1_minutes: f64,
    pub reentry_qty_multiplier: f64,
    pub ema_gate_mode: EmaGateMode,
    pub initial_ema_distance_fraction: f64,
    pub initial_qty_fraction: f64,
    pub threshold_base_fraction: f64,
    pub threshold_exposure_ratio_coefficient: f64,
    pub threshold_log_range_1h_coefficient: f64,
    pub threshold_log_range_1m_coefficient: f64,
    pub retracement_base_fraction: f64,
    pub retracement_exposure_ratio_coefficient: f64,
    pub retracement_log_range_1h_coefficient: f64,
    pub retracement_log_range_1m_coefficient: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct TrailingMartingaleCloseParams {
    pub qty_fraction: f64,
    pub threshold_base_fraction: f64,
    pub threshold_exposure_ratio_coefficient: f64,
    pub threshold_log_range_1h_coefficient: f64,
    pub threshold_log_range_1m_coefficient: f64,
    pub retracement_base_fraction: f64,
    pub retracement_log_range_1h_coefficient: f64,
    pub retracement_log_range_1m_coefficient: f64,
}

impl TrailingMartingaleParams {
    pub fn entry_params(&self) -> TrailingMartingaleEntryParams {
        self.entry
    }

    pub fn close_params(&self) -> TrailingMartingaleCloseParams {
        self.close
    }

    #[cfg(test)]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("trailing_martingale params serialize")
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EmaAnchorParams {
    pub base_qty_fraction: f64,
    pub ema_span_0_minutes: f64,
    pub ema_span_1_minutes: f64,
    pub entry_qty_exposure_coefficient: f64,
    pub offset_fraction: f64,
    pub offset_log_range_ema_span_minutes: f64,
    pub offset_log_range_1m_coefficient: f64,
    pub offset_log_range_ema_span_1h_candles: f64,
    pub offset_log_range_1h_coefficient: f64,
    pub inventory_offset_exposure_coefficient: f64,
}

impl Default for EmaAnchorParams {
    fn default() -> Self {
        Self {
            base_qty_fraction: 0.01,
            ema_span_0_minutes: 200.0,
            ema_span_1_minutes: 800.0,
            entry_qty_exposure_coefficient: 0.0,
            offset_fraction: 0.002,
            offset_log_range_ema_span_minutes: 60.0,
            offset_log_range_1m_coefficient: 0.0,
            offset_log_range_ema_span_1h_candles: 24.0,
            offset_log_range_1h_coefficient: 0.0,
            inventory_offset_exposure_coefficient: 0.1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrategyParams {
    TrailingMartingale(TrailingMartingaleParams),
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
    pub limit_order_fill_buffer_pct: f64,
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
    _side: StrategySide,
    raw: Option<&Value>,
    _bot_params: &BotParams,
) -> Result<StrategyParams, String> {
    match kind {
        StrategyKind::TrailingMartingale => {
            let value = raw.ok_or_else(|| {
                "trailing_martingale requires per-side strategy_params in orchestrator input"
                    .to_string()
            })?;
            let params = serde_json::from_value::<TrailingMartingaleParams>(value.clone())
                .map_err(|err| {
                    format!("failed to parse trailing_martingale strategy params: {err}")
                })?;
            Ok(StrategyParams::TrailingMartingale(params))
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

pub fn strategy_price_ema_spans_minutes(params: &StrategyParams) -> (f64, f64) {
    match params {
        StrategyParams::TrailingMartingale(params) => {
            (params.entry.ema_span_0_minutes, params.entry.ema_span_1_minutes)
        }
        StrategyParams::EmaAnchor(params) => (params.ema_span_0_minutes, params.ema_span_1_minutes),
    }
}

pub fn strategy_log_range_ema_span_1h_candles(params: &StrategyParams) -> Option<f64> {
    match params {
        StrategyParams::TrailingMartingale(params) => Some(params.log_range_ema_span_1h_candles),
        StrategyParams::EmaAnchor(params) => Some(params.offset_log_range_ema_span_1h_candles),
    }
}

pub fn strategy_log_range_ema_span_minutes(params: &StrategyParams) -> Option<f64> {
    match params {
        StrategyParams::TrailingMartingale(params) => Some(params.log_range_ema_span_minutes),
        StrategyParams::EmaAnchor(params) => Some(params.offset_log_range_ema_span_minutes),
    }
}

pub fn strategy_initial_entry_offset_fraction(params: &StrategyParams) -> f64 {
    match params {
        StrategyParams::TrailingMartingale(params) => params.entry.initial_ema_distance_fraction,
        StrategyParams::EmaAnchor(params) => params.offset_fraction,
    }
}

pub fn strategy_initial_qty_fraction(params: &StrategyParams) -> f64 {
    match params {
        StrategyParams::TrailingMartingale(params) => params.entry.initial_qty_fraction,
        StrategyParams::EmaAnchor(params) => params.base_qty_fraction,
    }
}

pub fn strategy_entry_retracement_enabled(params: &StrategyParams) -> bool {
    match params {
        StrategyParams::TrailingMartingale(params) => params.entry.retracement_base_fraction > 0.0,
        StrategyParams::EmaAnchor(_) => false,
    }
}

pub fn strategy_requires_sequential_entry_staging(params: &StrategyParams) -> bool {
    strategy_entry_retracement_enabled(params)
}

pub fn strategy_needs_log_range_1m(params: &StrategyParams) -> bool {
    strategy_needs_log_range_1m_for_request(params, true, true)
}

pub fn strategy_needs_log_range_1m_for_request(
    params: &StrategyParams,
    wants_entries: bool,
    wants_closes: bool,
) -> bool {
    match params {
        StrategyParams::TrailingMartingale(params) => {
            ((wants_entries
                && (params.entry.threshold_log_range_1m_coefficient != 0.0
                    || params.entry.retracement_log_range_1m_coefficient != 0.0))
                || (wants_closes
                    && (params.close.threshold_log_range_1m_coefficient != 0.0
                        || params.close.retracement_log_range_1m_coefficient != 0.0)))
                && params.log_range_ema_span_minutes > 0.0
        }
        StrategyParams::EmaAnchor(params) => {
            (wants_entries || wants_closes)
                && params.offset_log_range_1m_coefficient != 0.0
                && params.offset_log_range_ema_span_minutes > 0.0
        }
    }
}

pub fn strategy_needs_log_range_1h(params: &StrategyParams) -> bool {
    strategy_needs_log_range_1h_for_request(params, true, true)
}

pub fn strategy_needs_log_range_1h_for_request(
    params: &StrategyParams,
    wants_entries: bool,
    wants_closes: bool,
) -> bool {
    match params {
        StrategyParams::TrailingMartingale(params) => {
            ((wants_entries
                && (params.entry.threshold_log_range_1h_coefficient != 0.0
                    || params.entry.retracement_log_range_1h_coefficient != 0.0))
                || (wants_closes
                    && (params.close.threshold_log_range_1h_coefficient != 0.0
                        || params.close.retracement_log_range_1h_coefficient != 0.0)))
                && params.log_range_ema_span_1h_candles > 0.0
        }
        StrategyParams::EmaAnchor(params) => {
            (wants_entries || wants_closes)
                && params.offset_log_range_1h_coefficient != 0.0
                && params.offset_log_range_ema_span_1h_candles > 0.0
        }
    }
}

pub fn strategy_has_trailing(params: &StrategyParams) -> bool {
    strategy_needs_trailing_for_request(params, true, true)
}

pub fn strategy_needs_trailing_for_request(
    params: &StrategyParams,
    wants_entries: bool,
    wants_closes: bool,
) -> bool {
    match params {
        StrategyParams::TrailingMartingale(params) => {
            (wants_entries && params.entry.retracement_base_fraction > 0.0)
                || (wants_closes && params.close.retracement_base_fraction > 0.0)
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
        StrategyKind::TrailingMartingale => trailing_martingale::generate_orders(side, request),
        StrategyKind::EmaAnchor => ema_anchor::generate_orders(side, request),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trailing_input_needs_are_request_scoped_for_supported_strategies() {
        let mut martingale = TrailingMartingaleParams::default();
        martingale.entry.retracement_base_fraction = 0.01;
        let martingale = StrategyParams::TrailingMartingale(martingale);
        assert!(strategy_needs_trailing_for_request(
            &martingale,
            true,
            false
        ));
        assert!(!strategy_needs_trailing_for_request(
            &martingale,
            false,
            true
        ));
        assert!(strategy_requires_sequential_entry_staging(&martingale));
        let ema_anchor = StrategyParams::EmaAnchor(EmaAnchorParams::default());
        assert!(!strategy_needs_trailing_for_request(
            &ema_anchor,
            true,
            true
        ));
        assert!(!strategy_requires_sequential_entry_staging(&ema_anchor));
    }

    #[test]
    fn martingale_entry_volatility_is_not_required_for_close_only_request() {
        let mut params = TrailingMartingaleParams::default();
        params.entry.threshold_log_range_1h_coefficient = 1.0;
        params.log_range_ema_span_1h_candles = 4.0;
        let params = StrategyParams::TrailingMartingale(params);
        assert!(strategy_needs_log_range_1h(&params));
        assert!(strategy_needs_log_range_1h_for_request(
            &params, true, false
        ));
        assert!(!strategy_needs_log_range_1h_for_request(
            &params, false, true
        ));
    }

    #[test]
    fn martingale_rejects_retired_grid_fields_instead_of_reinterpreting_them() {
        let raw = json!({"entry":{"trailing_grid_ratio":0.5}});
        let error = parse_strategy_params(
            StrategyKind::TrailingMartingale,
            StrategySide::Long,
            Some(&raw),
            &BotParams::default(),
        )
        .unwrap_err();
        assert!(error.contains("trailing_grid_ratio"));
    }
}
