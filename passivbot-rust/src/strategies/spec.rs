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

const TRAILING_GRID_PARAM_SEEDS: &[ParamSeed] = &[
    ParamSeed {
        name: "close_grid_markup_end",
        long_default: 0.0094,
        short_default: 0.0015,
        long_bounds: &[0.0015, 0.012, 1e-05],
        short_bounds: &[0.0015, 0.012, 1e-05],
    },
    ParamSeed {
        name: "close_grid_markup_start",
        long_default: 0.00634,
        short_default: 0.0015,
        long_bounds: &[0.0015, 0.012, 1e-05],
        short_bounds: &[0.0015, 0.012, 1e-05],
    },
    ParamSeed {
        name: "close_grid_qty_pct",
        long_default: 0.51,
        short_default: 0.05,
        long_bounds: &[0.05, 1.0, 0.01],
        short_bounds: &[0.05, 1.0, 0.01],
    },
    ParamSeed {
        name: "close_trailing_grid_ratio",
        long_default: -0.76,
        short_default: -1.0,
        long_bounds: &[-1.0, 1.0, 0.01],
        short_bounds: &[-1.0, 1.0, 0.01],
    },
    ParamSeed {
        name: "close_trailing_qty_pct",
        long_default: 0.05,
        short_default: 0.05,
        long_bounds: &[0.05, 1.0, 0.01],
        short_bounds: &[0.05, 1.0, 0.01],
    },
    ParamSeed {
        name: "close_trailing_retracement_pct",
        long_default: 0.00279,
        short_default: 0.001,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    ParamSeed {
        name: "close_trailing_threshold_pct",
        long_default: 0.001,
        short_default: 0.001,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    ParamSeed {
        name: "ema_span_0",
        long_default: 770.0,
        short_default: 100.0,
        long_bounds: &[200.0, 1440.0, 10.0],
        short_bounds: &[200.0, 1440.0, 10.0],
    },
    ParamSeed {
        name: "ema_span_1",
        long_default: 210.0,
        short_default: 100.0,
        long_bounds: &[200.0, 1440.0, 10.0],
        short_bounds: &[200.0, 1440.0, 10.0],
    },
    ParamSeed {
        name: "entry_grid_double_down_factor",
        long_default: 0.73,
        short_default: 0.5,
        long_bounds: &[0.5, 0.9, 0.01],
        short_bounds: &[0.5, 0.9, 0.01],
    },
    ParamSeed {
        name: "entry_grid_spacing_pct",
        long_default: 0.033,
        short_default: 0.025,
        long_bounds: &[0.01, 0.04, 1e-05],
        short_bounds: &[0.01, 0.04, 1e-05],
    },
    ParamSeed {
        name: "entry_grid_spacing_volatility_weight",
        long_default: 2.4,
        short_default: 1.0,
        long_bounds: &[1.0, 40.0, 0.1],
        short_bounds: &[1.0, 40.0, 0.1],
    },
    ParamSeed {
        name: "entry_grid_spacing_we_weight",
        long_default: 0.135,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.0001],
        short_bounds: &[0.0, 5.0, 0.0001],
    },
    ParamSeed {
        name: "entry_initial_ema_dist",
        long_default: 0.0097,
        short_default: -0.01,
        long_bounds: &[-0.01, 0.01, 0.0001],
        short_bounds: &[-0.01, 0.01, 0.0001],
    },
    ParamSeed {
        name: "entry_initial_qty_pct",
        long_default: 0.0276,
        short_default: 0.01,
        long_bounds: &[0.01, 0.03, 0.0001],
        short_bounds: &[0.01, 0.03, 0.0001],
    },
    ParamSeed {
        name: "entry_trailing_double_down_factor",
        long_default: 0.9,
        short_default: 0.5,
        long_bounds: &[0.5, 1.0, 0.01],
        short_bounds: &[0.5, 1.0, 0.01],
    },
    ParamSeed {
        name: "entry_trailing_grid_ratio",
        long_default: -0.5,
        short_default: -0.5,
        long_bounds: &[-0.8, -0.2, 0.01],
        short_bounds: &[-0.8, -0.2, 0.01],
    },
    ParamSeed {
        name: "entry_trailing_retracement_pct",
        long_default: 0.0276,
        short_default: 0.001,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    ParamSeed {
        name: "entry_trailing_retracement_volatility_weight",
        long_default: 87.0,
        short_default: 1.0,
        long_bounds: &[1.0, 40.0, 0.1],
        short_bounds: &[1.0, 40.0, 0.1],
    },
    ParamSeed {
        name: "entry_trailing_retracement_we_weight",
        long_default: 3.97,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    ParamSeed {
        name: "entry_trailing_threshold_pct",
        long_default: 0.0029,
        short_default: 0.001,
        long_bounds: &[0.001, 0.015, 1e-05],
        short_bounds: &[0.001, 0.015, 1e-05],
    },
    ParamSeed {
        name: "entry_trailing_threshold_volatility_weight",
        long_default: 76.0,
        short_default: 1.0,
        long_bounds: &[1.0, 40.0, 0.1],
        short_bounds: &[1.0, 40.0, 0.1],
    },
    ParamSeed {
        name: "entry_trailing_threshold_we_weight",
        long_default: 1.31,
        short_default: 0.0,
        long_bounds: &[0.0, 5.0, 0.001],
        short_bounds: &[0.0, 5.0, 0.001],
    },
    ParamSeed {
        name: "entry_volatility_ema_span_hours",
        long_default: 1690.0,
        short_default: 672.0,
        long_bounds: &[672.0, 2016.0, 1.0],
        short_bounds: &[672.0, 2016.0, 1.0],
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
        name: "offset_volatility_ema_span_minutes",
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
        name: "entry_volatility_ema_span_hours",
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
            defaults.get_mut(side).unwrap().insert(seed.name, default);
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

pub fn trailing_grid_spec() -> StrategySpec {
    build_strategy_spec("trailing_grid", TRAILING_GRID_PARAM_SEEDS, |side, name| {
        vec![format!("bot.{side}.{name}")]
    })
}

pub fn ema_anchor_spec() -> StrategySpec {
    build_strategy_spec("ema_anchor", EMA_ANCHOR_PARAM_SEEDS, |_side, _name| {
        Vec::new()
    })
}
