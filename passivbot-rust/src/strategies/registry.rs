use super::spec::{ema_anchor_spec, trailing_grid_spec, StrategySpec};
use super::StrategyKind;

pub fn strategy_kind_from_name(name: &str) -> Option<StrategyKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "trailing_grid" => Some(StrategyKind::TrailingGrid),
        "ema_anchor" => Some(StrategyKind::EmaAnchor),
        _ => None,
    }
}

pub fn strategy_spec(kind: StrategyKind) -> StrategySpec {
    match kind {
        StrategyKind::TrailingGrid => trailing_grid_spec(),
        StrategyKind::EmaAnchor => ema_anchor_spec(),
    }
}
