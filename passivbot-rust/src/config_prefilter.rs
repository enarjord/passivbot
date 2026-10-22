//! Hypothetical normalized strategy probes, never live inputs or candle replay.
use crate::closes::{calc_close_retracement_multiplier, calc_close_threshold_pct};
use crate::entries::{
    calc_entry_distance_multiplier, calc_entry_retracement_multiplier, calc_next_entry_long,
    calc_next_entry_short, gate_reentry_price_ask, gate_reentry_price_bid,
    wallet_exposure_limit_with_allowance,
};
use crate::strategies::TrailingMartingaleParams;
use crate::types::{
    BotParams, EMABands, ExchangeParams, OrderBook, Position, RuntimeOrderContext, StateParams,
    TrailingPriceBundle,
};
use crate::utils::calc_new_psize_pprice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    side: String,
    strategy: TrailingMartingaleParams,
    wallet_exposure_limit: f64,
    total_wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    cooldown_base: f64,
    cooldown_min: f64,
    cooldown_max: Option<f64>,
    cooldown_exposure_weight: f64,
    cooldown_adverse_weight: f64,
    exposure_ratios: Vec<f64>,
    volatility_ema_1m: f64,
    volatility_ema_1h: f64,
    adverse_directionality: f64,
    grid_max_entries: usize,
    calculate_grid: bool,
}

#[derive(Debug, Serialize)]
struct Observation {
    exposure_ratio: f64,
    entry_close_span: f64,
    entry_threshold: f64,
    entry_retracement: f64,
    close_retracement: f64,
    entry_cooldown_minutes: f64,
}

#[derive(Debug, Serialize)]
struct Measurements {
    observations: Vec<Observation>,
    grid_span: Option<f64>,
    grid_complete: bool,
    grid_entries: usize,
}

fn bot(p: &Probe) -> BotParams {
    BotParams {
        wallet_exposure_limit: p.wallet_exposure_limit,
        total_wallet_exposure_limit: p.total_wallet_exposure_limit,
        risk_we_excess_allowance_pct: p.risk_we_excess_allowance_pct,
        risk_entry_cooldown_minutes: p.cooldown_base,
        entry_cooldown_min_duration_minutes: p.cooldown_min,
        entry_cooldown_max_duration_minutes: p.cooldown_max,
        entry_cooldown_weights_minutes: crate::entry_cooldown::CooldownWeights {
            exposure_ratio: p.cooldown_exposure_weight,
            adverse_directionality: p.cooldown_adverse_weight,
        },
        ..Default::default()
    }
}

fn distances(p: &Probe, exposure: f64, allowed: f64) -> (f64, f64) {
    let e = &p.strategy.entry;
    (
        e.threshold_base_pct.max(0.0)
            * calc_entry_distance_multiplier(
                exposure,
                allowed,
                e,
                p.volatility_ema_1h,
                p.volatility_ema_1m,
            ),
        e.retracement_base_pct.max(0.0)
            * calc_entry_retracement_multiplier(
                exposure,
                allowed,
                e,
                p.volatility_ema_1h,
                p.volatility_ema_1m,
            ),
    )
}

// Fill each Rust-sized entry along a declared adverse excursion plus the minimum
// trailing confirmation. Balance and EMA reference stay at 1, fees are zero, and
// quantity/price steps are 1e-12. No closes or elapsed-time policy are simulated.
fn grid(
    p: &Probe,
    bp: &BotParams,
    runtime: &RuntimeOrderContext,
    allowed: f64,
) -> (f64, bool, usize) {
    let long = p.side == "long";
    let exchange = ExchangeParams {
        qty_step: 1e-12,
        price_step: 1e-12,
        min_qty: 1e-12,
        min_cost: 0.0,
        c_mult: 1.0,
        maker_fee: 0.0,
        taker_fee: 0.0,
    };
    let mut position = Position {
        size: 0.0,
        price: 0.0,
    };
    let mut first = 0.0;
    let mut last = 0.0;
    let mut count = 0;
    for _ in 0..p.grid_max_entries {
        let exposure = position.size.abs() * position.price;
        if exposure >= allowed * 0.999 {
            return ((last / first - 1.0_f64).abs(), true, count);
        }
        let (distance, retracement) = distances(p, exposure, allowed);
        let mut touch = 1.0;
        let mut trailing = TrailingPriceBundle::default();
        if position.size != 0.0 {
            let threshold = position.price * (if long { 1.0 - distance } else { 1.0 + distance });
            if !threshold.is_finite() || threshold <= exchange.price_step {
                break;
            }
            let extreme = threshold * (if long { 1.0 - 1e-9 } else { 1.0 + 1e-9 });
            touch = extreme
                * (if long {
                    1.0 + retracement + 1e-9
                } else {
                    1.0 - retracement - 1e-9
                });
            if !touch.is_finite() || touch <= exchange.price_step {
                break;
            }
            trailing = TrailingPriceBundle {
                min_since_open: if long { extreme } else { touch },
                max_since_min: if long { touch } else { extreme },
                max_since_open: if long { touch } else { extreme },
                min_since_max: if long { extreme } else { touch },
            };
        }
        let state = StateParams {
            balance: 1.0,
            order_book: OrderBook {
                bid: touch,
                ask: touch,
            },
            ema_bands: EMABands {
                lower: 1.0,
                upper: 1.0,
            },
            volatility_ema_1m: p.volatility_ema_1m,
            volatility_ema_1h: p.volatility_ema_1h,
        };
        let entry = if long {
            calc_next_entry_long(
                &exchange,
                &state,
                bp,
                runtime,
                &p.strategy.entry,
                &position,
                &trailing,
            )
        } else {
            calc_next_entry_short(
                &exchange,
                &state,
                bp,
                runtime,
                &p.strategy.entry,
                &position,
                &trailing,
            )
        };
        if entry.qty == 0.0
            || !entry.qty.is_finite()
            || !entry.price.is_finite()
            || entry.price <= 0.0
        {
            break;
        }
        let (size, price) = calc_new_psize_pprice(
            position.size,
            position.price,
            entry.qty,
            entry.price,
            exchange.qty_step,
        );
        if size.abs() * price <= exposure {
            break;
        }
        position = Position { size, price };
        if count == 0 {
            first = entry.price;
        }
        last = entry.price;
        count += 1;
    }
    let span = if first > 0.0 {
        (last / first - 1.0_f64).abs()
    } else {
        0.0
    };
    (
        span,
        position.size.abs() * position.price >= allowed * 0.999,
        count,
    )
}

fn measure(p: &Probe) -> Result<Measurements, String> {
    if p.side != "long" && p.side != "short" {
        return Err("invalid probe side".into());
    }
    if p.exposure_ratios.is_empty()
        || p.exposure_ratios.len() > 64
        || p.exposure_ratios
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0 || *x >= 0.999)
        || !(2..=4096).contains(&p.grid_max_entries)
    {
        return Err("invalid probe exposure ratios or grid bound".into());
    }
    for value in [
        p.wallet_exposure_limit,
        p.total_wallet_exposure_limit,
        p.risk_we_excess_allowance_pct,
        p.volatility_ema_1m,
        p.volatility_ema_1h,
        p.adverse_directionality,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err("invalid probe numeric input".into());
        }
    }
    fn finite(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(map) => map.values().all(finite),
            serde_json::Value::Number(number) => number.as_f64().is_some_and(f64::is_finite),
            serde_json::Value::String(_) => true,
            _ => false,
        }
    }
    if !finite(&serde_json::to_value(p.strategy).map_err(|e| e.to_string())?) {
        return Err("non-finite strategy probe input".into());
    }
    let bp = bot(p);
    crate::entry_cooldown::validate(&bp)?;
    let runtime = RuntimeOrderContext {
        effective_wallet_exposure_limit: p.wallet_exposure_limit,
    };
    let allowed = wallet_exposure_limit_with_allowance(&bp, &runtime);
    if !allowed.is_finite() || allowed <= 0.0 {
        return Err("probe requires positive exposure capacity".into());
    }
    let exchange = ExchangeParams {
        qty_step: 1e-12,
        price_step: 1e-12,
        min_qty: 1e-12,
        min_cost: 0.0,
        ..Default::default()
    };
    let normalized_state = StateParams {
        balance: 1.0,
        order_book: OrderBook { bid: 1.0, ask: 1.0 },
        ema_bands: EMABands {
            lower: 1.0,
            upper: 1.0,
        },
        volatility_ema_1m: p.volatility_ema_1m,
        volatility_ema_1h: p.volatility_ema_1h,
    };
    let flat = Position {
        size: 0.0,
        price: 0.0,
    };
    let initial = if p.side == "long" {
        calc_next_entry_long(
            &exchange,
            &normalized_state,
            &bp,
            &runtime,
            &p.strategy.entry,
            &flat,
            &TrailingPriceBundle::default(),
        )
    } else {
        calc_next_entry_short(
            &exchange,
            &normalized_state,
            &bp,
            &runtime,
            &p.strategy.entry,
            &flat,
            &TrailingPriceBundle::default(),
        )
    };
    // Reentry-only gates do not affect the flat initial order above. Ask the
    // runtime gate directly so every EMA mode measures its actual entry branch.
    let reentry_gate_price = if p.side == "long" {
        gate_reentry_price_bid(1.0, &exchange, &normalized_state, &p.strategy.entry)
    } else {
        gate_reentry_price_ask(1.0, &exchange, &normalized_state, &p.strategy.entry)
    };
    let reentry_distance = (reentry_gate_price - 1.0).abs();
    let mut observations = Vec::new();
    for &ratio in &p.exposure_ratios {
        let (mut distance, mut retracement) = distances(p, ratio * allowed, allowed);
        let initial_distance = if p.side == "long" {
            (1.0 - initial.price).max(0.0)
        } else {
            (initial.price - 1.0).max(0.0)
        };
        if ratio * allowed < initial.qty.abs() * 0.8 {
            distance = initial_distance;
            retracement = 0.0;
        } else if p.strategy.entry.ema_gate_mode.gates_reentry() {
            distance = distance.max(reentry_distance);
        }
        let state = StateParams {
            balance: 1.0,
            volatility_ema_1m: p.volatility_ema_1m,
            volatility_ema_1h: p.volatility_ema_1h,
            ..Default::default()
        };
        let position = Position {
            size: ratio * allowed,
            price: 1.0,
        };
        let close = calc_close_threshold_pct(
            &exchange,
            &state,
            &bp,
            &runtime,
            &p.strategy.close,
            &position,
        );
        let close_retracement = p.strategy.close.retracement_base_pct.max(0.0)
            * calc_close_retracement_multiplier(&state, &p.strategy.close);
        // A nonpositive trailing close threshold has no profit-excursion gate.
        let span = (distance
            + if p.strategy.close.retracement_base_pct > 0.0 {
                close.max(0.0)
            } else {
                close
            })
        .max(0.0);
        let cooldown = crate::entry_cooldown::effective_duration(
            &bp,
            Some(ratio * allowed / p.wallet_exposure_limit),
            Some(p.adverse_directionality),
        )?;
        if [distance, span, retracement, close_retracement, cooldown]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err("non-finite config probe result".into());
        }
        observations.push(Observation {
            exposure_ratio: ratio,
            entry_close_span: span,
            entry_threshold: distance,
            entry_retracement: retracement,
            close_retracement,
            entry_cooldown_minutes: cooldown,
        });
    }
    let (grid_span, grid_complete, grid_entries) = if p.calculate_grid {
        let (span, complete, entries) = grid(p, &bp, &runtime, allowed);
        if !span.is_finite() {
            return Err("non-finite grid span".into());
        }
        (Some(span), complete, entries)
    } else {
        (None, true, 0)
    };
    Ok(Measurements {
        observations,
        grid_span,
        grid_complete,
        grid_entries,
    })
}

#[pyfunction]
pub fn config_prefilter_metrics_json(raw: &str) -> PyResult<String> {
    let input: Probe =
        serde_json::from_str(raw).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = measure(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe() -> Probe {
        Probe {
            side: "long".into(),
            strategy: TrailingMartingaleParams::default(),
            wallet_exposure_limit: 1.0,
            total_wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            cooldown_base: 0.0,
            cooldown_min: 0.0,
            cooldown_max: None,
            cooldown_exposure_weight: 0.0,
            cooldown_adverse_weight: 0.0,
            exposure_ratios: vec![0.8],
            volatility_ema_1m: 0.0,
            volatility_ema_1h: 0.0,
            adverse_directionality: 0.0,
            grid_max_entries: 1024,
            calculate_grid: true,
        }
    }

    #[test]
    fn geometry_uses_separate_entry_and_close_weights() {
        let mut p = probe();
        p.strategy.entry.threshold_base_pct = 0.03;
        p.strategy.entry.threshold_we_weight = 5.0;
        p.strategy.entry.retracement_base_pct = 0.001;
        p.strategy.entry.retracement_we_weight = 0.1;
        p.strategy.close.threshold_base_pct = 0.01;
        p.strategy.close.threshold_we_weight = -0.005;
        let result = measure(&p).unwrap();
        assert!((result.observations[0].entry_close_span - 0.156).abs() < 1e-12);
        assert!((result.observations[0].entry_threshold - 0.15).abs() < 1e-12);
        assert!((result.observations[0].entry_retracement - 0.00108).abs() < 1e-12);
    }

    #[test]
    fn entry_threshold_uses_exposure_and_declared_volatility_on_both_sides() {
        for side in ["long", "short"] {
            let mut p = probe();
            p.side = side.into();
            p.calculate_grid = false;
            p.strategy.entry.threshold_base_pct = 0.04;
            p.strategy.entry.threshold_we_weight = 5.0;
            p.strategy.entry.threshold_volatility_1m_weight = 10.0;
            p.strategy.entry.threshold_volatility_1h_weight = 2.0;
            p.volatility_ema_1m = 0.001;
            p.volatility_ema_1h = 0.01;
            p.exposure_ratios = vec![0.5, 0.8, 0.95];
            let result = measure(&p).unwrap();
            for (row, expected) in result.observations.iter().zip([0.1412, 0.2012, 0.2312]) {
                assert!((row.entry_threshold - expected).abs() < 1e-12);
            }
            // Close geometry must not change the independent entry measurement.
            p.strategy.close.threshold_base_pct = 0.5;
            p.strategy.close.retracement_base_pct = 0.0;
            let changed = measure(&p).unwrap();
            assert_eq!(
                changed.observations[1].entry_threshold,
                result.observations[1].entry_threshold
            );
            assert!(
                changed.observations[1].entry_close_span > result.observations[1].entry_close_span
            );
        }
    }

    #[test]
    fn entry_threshold_matches_runtime_order_for_every_ema_mode() {
        use crate::strategies::EmaGateMode;
        for side in ["long", "short"] {
            for mode in [
                EmaGateMode::Disabled,
                EmaGateMode::Initial,
                EmaGateMode::Reentry,
                EmaGateMode::All,
            ] {
                let mut p = probe();
                p.side = side.into();
                p.calculate_grid = false;
                p.strategy.entry.ema_gate_mode = mode;
                p.strategy.entry.initial_qty_pct = 0.01;
                p.strategy.entry.initial_ema_dist = 0.6;
                p.strategy.entry.threshold_base_pct = 0.03;
                p.strategy.entry.threshold_we_weight = 0.0;
                p.strategy.entry.retracement_base_pct = 0.0;
                let exchange = ExchangeParams {
                    qty_step: 1e-12,
                    price_step: 1e-12,
                    min_qty: 1e-12,
                    min_cost: 0.0,
                    c_mult: 1.0,
                    ..Default::default()
                };
                let state = StateParams {
                    balance: 1.0,
                    order_book: OrderBook { bid: 1.0, ask: 1.0 },
                    ema_bands: EMABands {
                        lower: 1.0,
                        upper: 1.0,
                    },
                    ..Default::default()
                };
                let runtime = RuntimeOrderContext {
                    effective_wallet_exposure_limit: 1.0,
                };
                let position = Position {
                    size: if side == "long" { 0.8 } else { -0.8 },
                    price: 1.0,
                };
                let entry_fn = if side == "long" {
                    calc_next_entry_long
                } else {
                    calc_next_entry_short
                };
                let order = entry_fn(
                    &exchange,
                    &state,
                    &bot(&p),
                    &runtime,
                    &p.strategy.entry,
                    &position,
                    &TrailingPriceBundle::default(),
                );
                assert!(order.qty.abs() > 0.0);
                let threshold = measure(&p).unwrap().observations[0].entry_threshold;
                assert!((threshold - (order.price - 1.0).abs()).abs() < 1e-12);
                assert!((threshold - if mode.gates_reentry() { 0.6 } else { 0.03 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn grid_accumulates_fills_to_capacity_and_is_bounded() {
        for side in ["long", "short"] {
            let mut p = probe();
            p.side = side.into();
            p.strategy.entry.initial_qty_pct = 0.1;
            p.strategy.entry.double_down_factor = 1.0;
            p.strategy.entry.threshold_base_pct = 0.05;
            let result = measure(&p).unwrap();
            assert!(result.grid_complete);
            assert!(result.grid_entries > 2);
            assert!(result.grid_span.unwrap() > 0.05);
            p.grid_max_entries = 2;
            assert!(!measure(&p).unwrap().grid_complete);
        }
    }

    #[test]
    fn partial_initial_and_ema_gates_use_their_own_price_branch() {
        let mut p = probe();
        p.calculate_grid = false;
        p.strategy.entry.initial_qty_pct = 0.9;
        p.strategy.entry.threshold_base_pct = 0.5;
        p.strategy.entry.retracement_base_pct = 0.1;
        p.exposure_ratios = vec![0.25];
        let row = &measure(&p).unwrap().observations[0];
        assert_eq!(row.entry_close_span, 0.0);
        assert_eq!(row.entry_threshold, 0.0);
        assert_eq!(row.entry_retracement, 0.0);
        p.strategy.entry.initial_qty_pct = 0.01;
        p.strategy.entry.initial_ema_dist = 0.6;
        p.strategy.entry.ema_gate_mode = crate::strategies::EmaGateMode::All;
        let result = measure(&p).unwrap();
        assert!((result.observations[0].entry_close_span - 0.6).abs() < 1e-9);
        assert!((result.observations[0].entry_threshold - 0.6).abs() < 1e-9);
    }
}
