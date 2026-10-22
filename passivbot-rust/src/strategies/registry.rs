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

    #[test]
    fn strategy_metadata_shares_common_defaults_without_merging_side_bounds() {
        for name in strategy_kind_names() {
            let spec = strategy_spec(strategy_kind_from_name(name).unwrap());
            assert_eq!(spec.defaults["long"], spec.defaults["short"]);
            for long in spec.parameters.iter().filter(|param| param.side == "long") {
                let short = spec
                    .parameters
                    .iter()
                    .find(|param| param.side == "short" && param.name == long.name)
                    .unwrap();
                assert_eq!(long.default, short.default);
                assert_eq!(long.optimize_key, format!("long_{}", long.name));
                assert_eq!(short.optimize_key, format!("short_{}", short.name));
            }
        }
        let tm = strategy_spec(StrategyKind::TrailingMartingale);
        assert_eq!(tm.defaults["long"]["ema_span_0"], 790.0);
        assert_eq!(tm.defaults["short"]["ema_span_0"], 790.0);
        assert_ne!(
            tm.optimize_bounds["long_ema_span_0"],
            tm.optimize_bounds["short_ema_span_0"]
        );
    }
}
