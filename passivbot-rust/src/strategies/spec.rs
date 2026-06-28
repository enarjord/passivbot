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
pub struct StrategyFixedParameterSpec {
    pub side: &'static str,
    pub name: &'static str,
    pub config_path: Vec<&'static str>,
    pub default: &'static str,
    pub allowed_values: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StrategySpec {
    pub strategy_kind: &'static str,
    pub defaults: BTreeMap<&'static str, BTreeMap<&'static str, f64>>,
    pub optimize_bounds: BTreeMap<String, Vec<f64>>,
    pub parameters: Vec<StrategyParameterSpec>,
    pub fixed_parameters: Vec<StrategyFixedParameterSpec>,
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
        long_default: 270.0,
        short_default: 60.0,
        long_bounds: &[100.0, 2880.0, 10.0],
        short_bounds: &[100.0, 2880.0, 10.0],
    },
    NestedParamSeed {
        name: "ema_span_1",
        path: &["ema_span_1"],
        long_default: 360.0,
        short_default: 60.0,
        long_bounds: &[100.0, 2880.0, 10.0],
        short_bounds: &[100.0, 2880.0, 10.0],
    },
    NestedParamSeed {
        name: "volatility_ema_span_1h",
        path: &["volatility_ema_span_1h"],
        long_default: 881.0,
        short_default: 672.0,
        long_bounds: &[168.0, 2016.0, 1.0],
        short_bounds: &[168.0, 2016.0, 1.0],
    },
    NestedParamSeed {
        name: "volatility_ema_span_1m",
        path: &["volatility_ema_span_1m"],
        long_default: 1377.0,
        short_default: 5.0,
        long_bounds: &[5.0, 1440.0, 1.0],
        short_bounds: &[5.0, 1440.0, 1.0],
    },
    NestedParamSeed {
        name: "entry_double_down_factor",
        path: &["entry", "double_down_factor"],
        long_default: 0.78,
        short_default: 0.2,
        long_bounds: &[0.2, 1.0, 0.01],
        short_bounds: &[0.2, 1.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_initial_qty_pct",
        path: &["entry", "initial_qty_pct"],
        long_default: 0.0179,
        short_default: 0.005,
        long_bounds: &[0.005, 0.1, 0.0001],
        short_bounds: &[0.005, 0.1, 0.0001],
    },
    NestedParamSeed {
        name: "entry_initial_ema_dist",
        path: &["entry", "initial_ema_dist"],
        long_default: 0.0175,
        short_default: -0.1,
        long_bounds: &[-0.1, 0.02, 0.0001],
        short_bounds: &[-0.1, 0.02, 0.0001],
    },
    NestedParamSeed {
        name: "entry_threshold_base_pct",
        path: &["entry", "threshold_base_pct"],
        long_default: 0.0321,
        short_default: 0.001,
        long_bounds: &[0.001, 0.035, 0.0001],
        short_bounds: &[0.001, 0.035, 0.0001],
    },
    NestedParamSeed {
        name: "entry_threshold_we_weight",
        path: &["entry", "threshold_we_weight"],
        long_default: 1.973,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    NestedParamSeed {
        name: "entry_threshold_volatility_1h_weight",
        path: &["entry", "threshold_volatility_1h_weight"],
        long_default: 12.07,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_threshold_volatility_1m_weight",
        path: &["entry", "threshold_volatility_1m_weight"],
        long_default: 34.58,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_retracement_base_pct",
        path: &["entry", "retracement_base_pct"],
        long_default: 0.0043,
        short_default: -0.01,
        long_bounds: &[0.0001, 0.01, 0.0001],
        short_bounds: &[0.0001, 0.01, 0.0001],
    },
    NestedParamSeed {
        name: "entry_retracement_we_weight",
        path: &["entry", "retracement_we_weight"],
        long_default: 2.913,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    NestedParamSeed {
        name: "entry_retracement_volatility_1h_weight",
        path: &["entry", "retracement_volatility_1h_weight"],
        long_default: 27.22,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "entry_retracement_volatility_1m_weight",
        path: &["entry", "retracement_volatility_1m_weight"],
        long_default: 8.57,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "close_qty_pct",
        path: &["close", "qty_pct"],
        long_default: 0.55,
        short_default: 0.05,
        long_bounds: &[0.05, 1.0, 0.01],
        short_bounds: &[0.05, 1.0, 0.01],
    },
    NestedParamSeed {
        name: "close_threshold_base_pct",
        path: &["close", "threshold_base_pct"],
        long_default: -0.0175,
        short_default: -0.02,
        long_bounds: &[-0.02, 0.02, 0.0001],
        short_bounds: &[-0.02, 0.02, 0.0001],
    },
    NestedParamSeed {
        name: "close_threshold_we_weight",
        path: &["close", "threshold_we_weight"],
        long_default: 0.0158,
        short_default: -0.1,
        long_bounds: &[-0.1, 0.1, 0.0001],
        short_bounds: &[-0.1, 0.1, 0.0001],
    },
    NestedParamSeed {
        name: "close_threshold_volatility_1h_weight",
        path: &["close", "threshold_volatility_1h_weight"],
        long_default: 0.12,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "close_threshold_volatility_1m_weight",
        path: &["close", "threshold_volatility_1m_weight"],
        long_default: 9.89,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "close_retracement_base_pct",
        path: &["close", "retracement_base_pct"],
        long_default: 0.0055,
        short_default: 0.0,
        long_bounds: &[0.0001, 0.01, 0.0001],
        short_bounds: &[0.0001, 0.01, 0.0001],
    },
    NestedParamSeed {
        name: "close_retracement_volatility_1h_weight",
        path: &["close", "retracement_volatility_1h_weight"],
        long_default: 30.15,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
    NestedParamSeed {
        name: "close_retracement_volatility_1m_weight",
        path: &["close", "retracement_volatility_1m_weight"],
        long_default: 6.19,
        short_default: 0.01,
        long_bounds: &[0.01, 40.0, 0.01],
        short_bounds: &[0.01, 40.0, 0.01],
    },
];

const TRAILING_GRID_V7_PARAM_SEEDS: &[NestedParamSeed] = &[
    NestedParamSeed {
        name: "ema_span_0",
        path: &["ema_span_0"],
        long_default: 385.0,
        short_default: 415.0,
        long_bounds: &[200.0, 1440.0, 5.0],
        short_bounds: &[200.0, 1440.0, 5.0],
    },
    NestedParamSeed {
        name: "ema_span_1",
        path: &["ema_span_1"],
        long_default: 620.0,
        short_default: 265.0,
        long_bounds: &[200.0, 1440.0, 5.0],
        short_bounds: &[200.0, 1440.0, 5.0],
    },
    NestedParamSeed {
        name: "entry_grid_double_down_factor",
        path: &["entry", "grid_double_down_factor"],
        long_default: 1.39,
        short_default: 0.57,
        long_bounds: &[0.5, 1.5, 0.01],
        short_bounds: &[0.5, 1.5, 0.01],
    },
    NestedParamSeed {
        name: "entry_grid_spacing_pct",
        path: &["entry", "grid_spacing_pct"],
        long_default: 0.02312,
        short_default: 0.03575,
        long_bounds: &[0.01, 0.04, 1e-05],
        short_bounds: &[0.01, 0.04, 1e-05],
    },
    NestedParamSeed {
        name: "entry_grid_spacing_we_weight",
        path: &["entry", "grid_spacing_we_weight"],
        long_default: 0.6766,
        short_default: 4.2641,
        long_bounds: &[0.0, 5.0, 0.0001],
        short_bounds: &[0.0, 5.0, 0.0001],
    },
    NestedParamSeed {
        name: "entry_grid_spacing_volatility_weight",
        path: &["entry", "grid_spacing_volatility_weight"],
        long_default: 17.8,
        short_default: 10.4,
        long_bounds: &[1.0, 40.0, 0.1],
        short_bounds: &[1.0, 40.0, 0.1],
    },
    NestedParamSeed {
        name: "entry_initial_ema_dist",
        path: &["entry", "initial_ema_dist"],
        long_default: 0.0078,
        short_default: 0.0093,
        long_bounds: &[-0.01, 0.01, 0.0001],
        short_bounds: &[-0.01, 0.01, 0.0001],
    },
    NestedParamSeed {
        name: "entry_initial_qty_pct",
        path: &["entry", "initial_qty_pct"],
        long_default: 0.0122,
        short_default: 0.0122,
        long_bounds: &[0.01, 0.03, 0.0001],
        short_bounds: &[0.01, 0.03, 0.0001],
    },
    NestedParamSeed {
        name: "entry_trailing_double_down_factor",
        path: &["entry", "trailing_double_down_factor"],
        long_default: 1.0,
        short_default: 0.85,
        long_bounds: &[0.5, 1.5, 0.01],
        short_bounds: &[0.5, 1.5, 0.01],
    },
    NestedParamSeed {
        name: "entry_trailing_grid_ratio",
        path: &["entry", "trailing_grid_ratio"],
        long_default: -0.32,
        short_default: -0.7,
        long_bounds: &[-0.8, -0.2, 0.01],
        short_bounds: &[-0.8, -0.2, 0.01],
    },
    NestedParamSeed {
        name: "entry_trailing_retracement_pct",
        path: &["entry", "trailing_retracement_pct"],
        long_default: 0.01498,
        short_default: 0.00135,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    NestedParamSeed {
        name: "entry_trailing_retracement_we_weight",
        path: &["entry", "trailing_retracement_we_weight"],
        long_default: 4.958,
        short_default: 0.381,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    NestedParamSeed {
        name: "entry_trailing_retracement_volatility_weight",
        path: &["entry", "trailing_retracement_volatility_weight"],
        long_default: 37.9,
        short_default: 10.9,
        long_bounds: &[1.0, 40.0, 0.1],
        short_bounds: &[1.0, 40.0, 0.1],
    },
    NestedParamSeed {
        name: "entry_trailing_threshold_pct",
        path: &["entry", "trailing_threshold_pct"],
        long_default: 0.00215,
        short_default: 0.00186,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    NestedParamSeed {
        name: "entry_trailing_threshold_we_weight",
        path: &["entry", "trailing_threshold_we_weight"],
        long_default: 4.243,
        short_default: 0.132,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    NestedParamSeed {
        name: "entry_trailing_threshold_volatility_weight",
        path: &["entry", "trailing_threshold_volatility_weight"],
        long_default: 15.2,
        short_default: 13.4,
        long_bounds: &[1.0, 40.0, 0.1],
        short_bounds: &[1.0, 40.0, 0.1],
    },
    NestedParamSeed {
        name: "entry_volatility_ema_span_hours",
        path: &["entry", "volatility_ema_span_hours"],
        long_default: 1909.0,
        short_default: 1381.0,
        long_bounds: &[672.0, 2016.0, 1.0],
        short_bounds: &[672.0, 2016.0, 1.0],
    },
    NestedParamSeed {
        name: "close_grid_markup_start",
        path: &["close", "grid_markup_start"],
        long_default: 0.01041,
        short_default: 0.00402,
        long_bounds: &[0.0015, 0.012, 1e-05],
        short_bounds: &[0.0015, 0.012, 1e-05],
    },
    NestedParamSeed {
        name: "close_grid_markup_end",
        path: &["close", "grid_markup_end"],
        long_default: 0.00241,
        short_default: 0.00223,
        long_bounds: &[-0.1, 0.012, 1e-05],
        short_bounds: &[-0.1, 0.012, 1e-05],
    },
    NestedParamSeed {
        name: "close_grid_qty_pct",
        path: &["close", "grid_qty_pct"],
        long_default: 0.88,
        short_default: 0.13,
        long_bounds: &[0.05, 1.0, 0.01],
        short_bounds: &[0.05, 1.0, 0.01],
    },
    NestedParamSeed {
        name: "close_trailing_grid_ratio",
        path: &["close", "trailing_grid_ratio"],
        long_default: -0.07,
        short_default: -0.03,
        long_bounds: &[-1.0, 1.0, 0.01],
        short_bounds: &[-1.0, 1.0, 0.01],
    },
    NestedParamSeed {
        name: "close_trailing_qty_pct",
        path: &["close", "trailing_qty_pct"],
        long_default: 0.89,
        short_default: 0.87,
        long_bounds: &[0.05, 1.0, 0.01],
        short_bounds: &[0.05, 1.0, 0.01],
    },
    NestedParamSeed {
        name: "close_trailing_retracement_pct",
        path: &["close", "trailing_retracement_pct"],
        long_default: 0.00413,
        short_default: 0.00389,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    NestedParamSeed {
        name: "close_trailing_threshold_pct",
        path: &["close", "trailing_threshold_pct"],
        long_default: 0.0125,
        short_default: 0.01045,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
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
        fixed_parameters: Vec::new(),
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
        fixed_parameters: Vec::new(),
    }
}

pub fn trailing_martingale_spec() -> StrategySpec {
    let mut spec =
        build_nested_strategy_spec("trailing_martingale", TRAILING_MARTINGALE_PARAM_SEEDS);
    for side in ["long", "short"] {
        spec.fixed_parameters.push(StrategyFixedParameterSpec {
            side,
            name: "entry_ema_gate_mode",
            config_path: vec!["strategy", side, "entry", "ema_gate_mode"],
            default: "all",
            allowed_values: vec!["disabled", "all", "initial", "reentry"],
        });
    }
    spec
}

pub fn trailing_grid_v7_spec() -> StrategySpec {
    build_nested_strategy_spec("trailing_grid_v7", TRAILING_GRID_V7_PARAM_SEEDS)
}

pub fn ema_anchor_spec() -> StrategySpec {
    build_strategy_spec("ema_anchor", EMA_ANCHOR_PARAM_SEEDS, |_side, _name| {
        Vec::new()
    })
}
