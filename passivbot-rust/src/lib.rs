mod age_admission;
mod analysis;
mod backtest;
mod closes;
mod coin_selection;
mod config_prefilter;
mod constants;
mod dynamic;
mod entries;
mod entry_cooldown;
mod ew;
mod exposure_rms;
mod gpu;
mod hsl_decay;
mod hsl_history;
mod hsl_intent;
mod hsl_json;
mod hsl_prices;
mod hsl_snapshot;
mod hsl_sum;
mod hsl_support;
mod hsl_v9_live;
mod limit_fills;
mod orchestrator;
mod python;
mod risk;
mod strategies;
mod trailing;
mod types;
mod unilateralness;
mod utils;

use coin_selection::{select_coin_indices_py, select_forager_candidates_py};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use python::*;
use utils::*;

#[pyfunction]
#[pyo3(signature = (now_ms, inception_ms, minimum_days))]
fn coin_age_eligible(
    now_ms: &PyAny,
    inception_ms: Option<&PyAny>,
    minimum_days: &PyAny,
) -> PyResult<bool> {
    let minimum_days = python::minimum_age_days_from_py(minimum_days)?;
    let now_ms = python::age_timestamp_from_py(now_ms)?;
    let inception_ms = inception_ms
        .map(python::age_timestamp_from_py)
        .transpose()?;
    Ok(age_admission::eligible(now_ms, inception_ms, minimum_days))
}

#[pyfunction]
#[pyo3(signature = (timestamps, inception_ms, minimum_days))]
fn coin_age_first_eligible_index(
    timestamps: Vec<&PyAny>,
    inception_ms: Option<&PyAny>,
    minimum_days: &PyAny,
) -> PyResult<usize> {
    let minimum_days = python::minimum_age_days_from_py(minimum_days)?;
    let timestamps: Vec<u64> = timestamps
        .into_iter()
        .map(python::age_timestamp_from_py)
        .collect::<PyResult<_>>()?;
    let inception_ms = inception_ms
        .map(python::age_timestamp_from_py)
        .transpose()?;
    if timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "age admission requires increasing timestamps",
        ));
    }
    Ok(age_admission::first_eligible_index(
        &timestamps,
        inception_ms,
        minimum_days,
    ))
}

#[pyfunction]
fn gpu_ema_bootstrap_counts_py<'py>(
    py: Python<'py>,
    spans: numpy::PyReadonlyArray2<'py, f64>,
    interval_minutes: f64,
    available: u64,
) -> PyResult<Bound<'py, numpy::PyArray2<i32>>> {
    use numpy::IntoPyArray;
    let spans = spans.as_array();
    if spans.ncols() != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "GPU EMA bootstrap requires eight original-f64 span columns",
        ));
    }
    let mut counts =
        numpy::ndarray::Array2::zeros((spans.nrows(), gpu::MARKET_EMA_BOOTSTRAP_COUNT_COLS));
    for (mut output, input) in counts.outer_iter_mut().zip(spans.outer_iter()) {
        let row = gpu::market_ema_bootstrap_counts(
            std::array::from_fn(|i| input[i]),
            interval_minutes,
            available,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        for (value, count) in output.iter_mut().zip(row) {
            *value = count;
        }
    }
    Ok(counts.into_pyarray_bound(py))
}

#[pyfunction]
fn runtime_build_info(py: Python<'_>) -> PyResult<PyObject> {
    let info = PyDict::new_bound(py);
    info.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    info.set_item(
        "source_fingerprint",
        option_env!("PASSIVBOT_RUST_SOURCE_FINGERPRINT").unwrap_or("unknown"),
    )?;
    Ok(info.into())
}

#[pyfunction]
fn mps_ema_anchor_multicoin_source_py() -> &'static str {
    gpu::mps_ema_anchor_multicoin_source()
}

#[pyfunction]
fn mps_ema_anchor_multicoin_long_source_py() -> &'static str {
    gpu::mps_ema_anchor_multicoin_long_source()
}

#[pyfunction]
fn mps_trailing_martingale_multicoin_source_py() -> &'static str {
    gpu::mps_trailing_martingale_multicoin_source()
}

#[pyfunction]
fn mps_strategy_eq_recovery_distribution_source_py() -> &'static str {
    gpu::mps_strategy_eq_recovery_distribution_source()
}

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coin_age_eligible, m)?)?;
    m.add_function(wrap_pyfunction!(coin_age_first_eligible_index, m)?)?;
    m.add_function(wrap_pyfunction!(
        config_prefilter::config_prefilter_metrics_json,
        m
    )?)?;
    m.add_class::<HlcvsBundlePy>()?;
    m.add_function(wrap_pyfunction!(
        unilateralness::calc_signed_unilateralness,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        entry_cooldown::entry_cooldown_durations_json,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(runtime_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ew_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_v9_live::hsl_v9_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_v9_live::hsl_v9_evaluate_grids, m)?)?;
    m.add_class::<hsl_v9_live::HslV9Replay>()?;
    m.add_function(wrap_pyfunction!(hsl_v9_live::realized_memory_v9, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_v9_live::realized_memories_v9, m)?)?;
    m.add_function(wrap_pyfunction!(
        hsl_v9_live::calc_auto_unstuck_allowance_v9,
        m
    )?)?;
    m.add_class::<hsl_prices::HslPriceGrid>()?;
    m.add_class::<hsl_prices::HslCandleSource>()?;
    m.add_function(wrap_pyfunction!(hsl_prices::hsl_native_price_grid, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_history::hsl_history, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_prices::hsl_prices, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_prices::hsl_price_grid, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_snapshot::hsl_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(mps_ema_anchor_multicoin_source_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        mps_ema_anchor_multicoin_long_source_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mps_trailing_martingale_multicoin_source_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mps_strategy_eq_recovery_distribution_source_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utils::gpu_decimal_metadata_py, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_ema_bootstrap_counts_py, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_gpu_market_execution_ticks, m)?)?;
    m.add(
        "GPU_MARKET_EXECUTION_SELL_LANE",
        GPU_MARKET_EXECUTION_SELL_LANE,
    )?;
    m.add(
        "GPU_MARKET_EXECUTION_BUY_LANE",
        GPU_MARKET_EXECUTION_BUY_LANE,
    )?;
    m.add_function(wrap_pyfunction!(round_, m)?)?;
    m.add_function(wrap_pyfunction!(round_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dn, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic_dn, m)?)?;
    m.add_function(wrap_pyfunction!(calc_diff, m)?)?;
    m.add_function(wrap_pyfunction!(qty_to_cost, m)?)?;
    m.add_function(wrap_pyfunction!(cost_to_qty, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pnl_long, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pnl_short, m)?)?;
    m.add_function(wrap_pyfunction!(calc_wallet_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(calc_new_psize_pprice, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_entry_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        calc_trailing_martingale_close_diagnostic_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(calc_next_close_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_entries_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_entry_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_close_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_entries_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_closes_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_closes_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_twel_enforcer_orders_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(hysteresis, m)?)?;
    m.add_function(wrap_pyfunction!(ema_last_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_min_entry_qty_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pprice_diff_int, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pside_price_diff_int, m)?)?;
    m.add_function(wrap_pyfunction!(calc_price_diff_pside_int, m)?)?;
    m.add_function(wrap_pyfunction!(calc_order_price_diff, m)?)?;
    m.add_function(wrap_pyfunction!(order_type_id_to_snake, m)?)?;
    m.add_function(wrap_pyfunction!(all_order_types_ids, m)?)?;
    m.add_function(wrap_pyfunction!(order_type_snake_to_id, m)?)?;
    m.add_function(wrap_pyfunction!(get_order_id_type_from_string_alias, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ideal_orders_json, m)?)?;
    m.add_function(wrap_pyfunction!(compute_protective_closes_json, m)?)?;
    m.add_function(wrap_pyfunction!(get_strategy_spec, m)?)?;
    m.add_function(wrap_pyfunction!(get_strategy_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(gate_entries_by_twel_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_unstucking_close_py, m)?)?;
    m.add_function(wrap_pyfunction!(trailing_bundle_default_py, m)?)?;
    m.add_function(wrap_pyfunction!(update_trailing_bundle_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_coin_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_forager_candidates_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ema_anchor_quote_series_py, m)?)?;

    Ok(())
}
