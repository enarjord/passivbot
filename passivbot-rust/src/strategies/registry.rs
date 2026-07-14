use super::spec::{ema_anchor_spec, trailing_grid_v7_spec, trailing_martingale_spec, StrategySpec};
use super::StrategyKind;

pub fn strategy_kind_names() -> Vec<&'static str> {
    vec!["trailing_martingale", "ema_anchor", "trailing_grid_v7"]
}

pub fn strategy_kind_from_name(name: &str) -> Option<StrategyKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "trailing_martingale" => Some(StrategyKind::TrailingMartingale),
        "ema_anchor" => Some(StrategyKind::EmaAnchor),
        "trailing_grid_v7" => Some(StrategyKind::TrailingGridV7),
        _ => None,
    }
}

pub fn strategy_spec(kind: StrategyKind) -> StrategySpec {
    match kind {
        StrategyKind::TrailingMartingale => trailing_martingale_spec(),
        StrategyKind::EmaAnchor => ema_anchor_spec(),
        StrategyKind::TrailingGridV7 => trailing_grid_v7_spec(),
    }
}
