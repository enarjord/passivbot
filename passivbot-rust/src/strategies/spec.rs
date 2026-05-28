use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize)]
pub struct StrategyParameterSpec {
    pub side: &'static str,
    pub name: &'static str,
    pub config_path: Vec<&'static str>,
    pub optimize_key: String,
    pub default: f64,
    pub bounds: Vec<f64>,
    pub mirror_from: Option<&'static str>,
    pub legacy_config_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StrategySpec {
    pub strategy_kind: &'static str,
    pub defaults: BTreeMap<&'static str, BTreeMap<&'static str, f64>>,
    pub optimize_bounds: BTreeMap<String, Vec<f64>>,
    pub parameters: Vec<StrategyParameterSpec>,
}

struct ParamSeed {
    name: &'static str,
    long_default: f64,
    short_default: f64,
    long_bounds: &'static [f64],
    short_bounds: &'static [f64],
}

struct NestedParamSeed {
    name: &'static str,
    path: &'static [&'static str],
    long_default: f64,
    short_default: f64,
    long_bounds: &'static [f64],
    short_bounds: &'static [f64],
}

const TRAILING_MARTINGALE_PARAM_SEEDS: &[NestedParamSeed] = &[
    NestedParamSeed {
        name: "ema_span_0",
        path: &["ema_span_0"],
        long_default: 110.0,
        short_default: 100.0,
        long_bounds: &[100.0, 1440.0, 10.0],
        short_bounds: &[100.0, 1440.0, 10.0],
    },
    NestedParamSeed {
        name: "ema_span_1",
        path: &["ema_span_1"],
        long_default: 260.0,
        short_default: 100.0,
        long_bounds: &[100.0, 1440.0, 10.0],
        short_bounds: &[100.0, 1440.0, 10.0],
    },
    NestedParamSeed {
        name: "volatility_ema_span_1h",
        path: &["volatility_ema_span_1h"],
        long_default: 1787.0,
        short_default: 672.0,
        long_bounds: &[672.0, 2016.0, 1.0],
        short_bounds: &[672.0, 2016.0, 1.0],
    },
    NestedParamSeed {
        name: "volatility_ema_span_1m",
        path: &["volatility_ema_span_1m"],
        long_default: 44.0,
        short_default: 5.0,
        long_bounds: &[5.0, 720.0, 1.0],
        short_bounds: &[5.0, 720.0, 1.0],
    },
    NestedParamSeed {
        name: "entry_double_down_factor",
        path: &["entry", "double_down_factor"],
        long_default: 1.01,
        short_default: 0.5,
        long_bounds: &[0.5, 1.5, 0.01],
        short_bounds: &[0.5, 1.5, 0.01],
    },
    NestedParamSeed {
        name: "entry_initial_qty_pct",
        path: &["entry", "initial_qty_pct"],
        long_default: 0.0283,
        short_default: 0.005,
        long_bounds: &[0.005, 0.1, 0.0001],
        short_bounds: &[0.005, 0.1, 0.0001],
    },
    NestedParamSeed {
        name: "entry_initial_ema_dist",
        path: &["entry", "initial_ema_dist"],
        long_default: 0.0115,
        short_default: -0.1,
        long_bounds: &[-0.1, 0.02, 0.0001],
        short_bounds: &[-0.1, 0.02, 0.0001],
    },
    NestedParamSeed {
        name: "entry_threshold_base_pct",
        path: &["entry", "threshold_base_pct"],
        long_default: 0.0194,
        short_default: 0.0,
        long_bounds: &[0.0, 0.04, 1e-05],
        short_bounds: &[0.0, 0.04, 1e-05],
    },
    NestedParamSeed {
        name: "entry_threshold_we_weight",
        path: &["entry", "threshold_we_weight"],
        long_default: 3.578,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    NestedParamSeed {
        name: "entry_threshold_volatility_1h_weight",
        path: &["entry", "threshold_volatility_1h_weight"],
        long_default: 1.5,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_threshold_volatility_1m_weight",
        path: &["entry", "threshold_volatility_1m_weight"],
        long_default: 4.66,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_retracement_base_pct",
        path: &["entry", "retracement_base_pct"],
        long_default: 0.00008,
        short_default: 0.00001,
        long_bounds: &[0.00001, 0.015, 1e-05],
        short_bounds: &[0.00001, 0.015, 1e-05],
    },
    NestedParamSeed {
        name: "entry_retracement_we_weight",
        path: &["entry", "retracement_we_weight"],
        long_default: 0.032,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    NestedParamSeed {
        name: "entry_retracement_volatility_1h_weight",
        path: &["entry", "retracement_volatility_1h_weight"],
        long_default: 31.4,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_retracement_volatility_1m_weight",
        path: &["entry", "retracement_volatility_1m_weight"],
        long_default: 49.79,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "close_qty_pct",
        path: &["close", "qty_pct"],
        long_default: 0.67,
        short_default: 0.05,
        long_bounds: &[0.05, 1.0, 0.01],
        short_bounds: &[0.05, 1.0, 0.01],
    },
    NestedParamSeed {
        name: "close_threshold_base_pct",
        path: &["close", "threshold_base_pct"],
        long_default: 0.01035,
        short_default: -0.05,
        long_bounds: &[-0.05, 0.02, 1e-05],
        short_bounds: &[-0.05, 0.02, 1e-05],
    },
    NestedParamSeed {
        name: "close_threshold_we_weight",
        path: &["close", "threshold_we_weight"],
        long_default: -0.0119,
        short_default: -0.1,
        long_bounds: &[-0.1, 0.1, 0.0001],
        short_bounds: &[-0.1, 0.1, 0.0001],
    },
    NestedParamSeed {
        name: "close_threshold_volatility_1h_weight",
        path: &["close", "threshold_volatility_1h_weight"],
        long_default: 0.01,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "close_threshold_volatility_1m_weight",
        path: &["close", "threshold_volatility_1m_weight"],
        long_default: 1.4,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "close_retracement_base_pct",
        path: &["close", "retracement_base_pct"],
        long_default: 0.00001,
        short_default: 0.00001,
        long_bounds: &[0.00001, 0.01, 1e-05],
        short_bounds: &[0.00001, 0.01, 1e-05],
    },
    NestedParamSeed {
        name: "close_retracement_volatility_1h_weight",
        path: &["close", "retracement_volatility_1h_weight"],
        long_default: 2.34,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
    NestedParamSeed {
        name: "close_retracement_volatility_1m_weight",
        path: &["close", "retracement_volatility_1m_weight"],
        long_default: 5.63,
        short_default: 0.01,
        long_bounds: &[0.01, 70.0, 0.01],
        short_bounds: &[0.01, 70.0, 0.01],
    },
];

const EMA_ANCHOR_PARAM_SEEDS: &[ParamSeed] = &[
    ParamSeed {
        name: "base_qty_pct",
        long_default: 0.01,
        short_default: 0.01,
        long_bounds: &[0.001, 0.05, 0.0001],
        short_bounds: &[0.001, 0.05, 0.0001],
    },
    ParamSeed {
        name: "ema_span_0",
        long_default: 200.0,
        short_default: 200.0,
        long_bounds: &[20.0, 1440.0, 1.0],
        short_bounds: &[20.0, 1440.0, 1.0],
    },
    ParamSeed {
        name: "ema_span_1",
        long_default: 800.0,
        short_default: 800.0,
        long_bounds: &[20.0, 1440.0, 1.0],
        short_bounds: &[20.0, 1440.0, 1.0],
    },
    ParamSeed {
        name: "entry_double_down_factor",
        long_default: 0.0,
        short_default: 0.0,
        long_bounds: &[0.0, 2.0, 0.01],
        short_bounds: &[0.0, 2.0, 0.01],
    },
    ParamSeed {
        name: "offset",
        long_default: 0.002,
        short_default: 0.002,
        long_bounds: &[0.0, 0.05, 0.0001],
        short_bounds: &[0.0, 0.05, 0.0001],
    },
    ParamSeed {
        name: "offset_volatility_ema_span_1m",
        long_default: 60.0,
        short_default: 60.0,
        long_bounds: &[5.0, 720.0, 1.0],
        short_bounds: &[5.0, 720.0, 1.0],
    },
    ParamSeed {
        name: "offset_volatility_1m_weight",
        long_default: 0.0,
        short_default: 0.0,
        long_bounds: &[0.0, 40.0, 0.1],
        short_bounds: &[0.0, 40.0, 0.1],
    },
    ParamSeed {
        name: "offset_volatility_ema_span_1h",
        long_default: 24.0,
        short_default: 24.0,
        long_bounds: &[1.0, 672.0, 1.0],
        short_bounds: &[1.0, 672.0, 1.0],
    },
    ParamSeed {
        name: "offset_volatility_1h_weight",
        long_default: 0.0,
        short_default: 0.0,
        long_bounds: &[0.0, 40.0, 0.1],
        short_bounds: &[0.0, 40.0, 0.1],
    },
    ParamSeed {
        name: "offset_psize_weight",
        long_default: 0.1,
        short_default: 0.1,
        long_bounds: &[0.0, 2.0, 0.001],
        short_bounds: &[0.0, 2.0, 0.001],
    },
];

fn build_strategy_spec(
    strategy_kind: &'static str,
    param_seeds: &[ParamSeed],
    legacy_path_builder: impl Fn(&str, &'static str) -> Vec<String>,
) -> StrategySpec {
    let mut defaults: BTreeMap<&'static str, BTreeMap<&'static str, f64>> = BTreeMap::new();
    defaults.insert("long", BTreeMap::new());
    defaults.insert("short", BTreeMap::new());

    let mut optimize_bounds = BTreeMap::new();
    let mut parameters = Vec::with_capacity(param_seeds.len() * 2);

    for side in ["long", "short"] {
        for seed in param_seeds {
            let (default, bounds) = if side == "long" {
                (seed.long_default, seed.long_bounds)
            } else {
                (seed.short_default, seed.short_bounds)
            };
            defaults
                .get_mut(side)
                .expect("strategy defaults initialized for side")
                .insert(seed.name, default);
            let optimize_key = format!("{side}_{}", seed.name);
            optimize_bounds.insert(optimize_key.clone(), bounds.to_vec());
            parameters.push(StrategyParameterSpec {
                side,
                name: seed.name,
                config_path: vec!["strategy", side, seed.name],
                optimize_key,
                default,
                bounds: bounds.to_vec(),
                mirror_from: None,
                legacy_config_paths: legacy_path_builder(side, seed.name),
            });
        }
    }

    StrategySpec {
        strategy_kind,
        defaults,
        optimize_bounds,
        parameters,
    }
}

fn build_nested_strategy_spec(
    strategy_kind: &'static str,
    param_seeds: &[NestedParamSeed],
) -> StrategySpec {
    let mut defaults: BTreeMap<&'static str, BTreeMap<&'static str, f64>> = BTreeMap::new();
    defaults.insert("long", BTreeMap::new());
    defaults.insert("short", BTreeMap::new());

    let mut optimize_bounds = BTreeMap::new();
    let mut parameters = Vec::with_capacity(param_seeds.len() * 2);

    for side in ["long", "short"] {
        for seed in param_seeds {
            let (default, bounds) = if side == "long" {
                (seed.long_default, seed.long_bounds)
            } else {
                (seed.short_default, seed.short_bounds)
            };
            defaults
                .get_mut(side)
                .expect("strategy defaults initialized for side")
                .insert(seed.name, default);
            let optimize_key = format!("{side}_{}", seed.name);
            optimize_bounds.insert(optimize_key.clone(), bounds.to_vec());
            let mut config_path = vec!["strategy", side];
            config_path.extend_from_slice(seed.path);
            parameters.push(StrategyParameterSpec {
                side,
                name: seed.name,
                config_path,
                optimize_key,
                default,
                bounds: bounds.to_vec(),
                mirror_from: None,
                legacy_config_paths: Vec::new(),
            });
        }
    }

    StrategySpec {
        strategy_kind,
        defaults,
        optimize_bounds,
        parameters,
    }
}

pub fn trailing_martingale_spec() -> StrategySpec {
    build_nested_strategy_spec("trailing_martingale", TRAILING_MARTINGALE_PARAM_SEEDS)
}

pub fn ema_anchor_spec() -> StrategySpec {
    build_strategy_spec("ema_anchor", EMA_ANCHOR_PARAM_SEEDS, |_side, _name| {
        Vec::new()
    })
}
