use super::spec::{ema_anchor_spec, trailing_martingale_spec, StrategySpec};
use super::StrategyKind;

pub fn strategy_kind_names() -> Vec<&'static str> {
    vec!["trailing_martingale", "ema_anchor"]
}

pub fn strategy_kind_from_name(name: &str) -> Option<StrategyKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "trailing_martingale" => Some(StrategyKind::TrailingMartingale),
        "ema_anchor" => Some(StrategyKind::EmaAnchor),
        _ => None,
    }
}

pub fn strategy_spec(kind: StrategyKind) -> StrategySpec {
    match kind {
        StrategyKind::TrailingMartingale => trailing_martingale_spec(),
        StrategyKind::EmaAnchor => ema_anchor_spec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_registry_exposes_only_supported_kinds() {
        assert_eq!(
            strategy_kind_names(),
            vec!["trailing_martingale", "ema_anchor"]
        );
        assert_eq!(strategy_kind_from_name("trailing_grid_v7"), None);
        assert!(serde_json::from_str::<StrategyKind>("\"trailing_grid_v7\"").is_err());
        for name in strategy_kind_names() {
            let kind = strategy_kind_from_name(name).unwrap();
            assert_eq!(strategy_spec(kind).strategy_kind, name);
        }
    }
}
