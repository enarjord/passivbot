use crate::analysis::{
    analyze_backtest_pair_with_options, analyze_backtest_usd_with_profile, FillActivityMetrics,
};
use crate::backtest::BacktestEngine;
use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
    calc_trailing_martingale_close_diagnostic,
};
use crate::constants::{LONG, SHORT};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_next_entry_long, calc_next_entry_short,
};
use crate::risk::{
    calc_twel_enforcer_actions, calc_unstucking_action, gate_entries_by_twel, GateEntriesCandidate,
    GateEntriesDecision, GateEntriesPosition, TwelEnforcerInputPosition, UnstuckPositionInput,
};
use crate::strategies::ema_anchor::calc_quote_prices as calc_ema_anchor_quote_prices;
use crate::strategies::registry::{strategy_kind_from_name, strategy_kind_names, strategy_spec};
use crate::strategies::{
    EmaAnchorParams, EmaGateMode, StrategySide, TrailingMartingaleCloseParams,
    TrailingMartingaleEntryParams,
};
use crate::trailing::{
    trailing_bundle_to_tuple, tuple_to_trailing_bundle, update_trailing_bundle_sequence,
};
use crate::types::{Analysis, OrderType};
use crate::types::{
    BacktestParams, BotParams, BotParamsPair, CoinMeta, EMABands, Equities,
    EquityHardStopLossConfig, ExchangeParams, ForagerScoreWeights, HlcvsBundle, HlcvsMeta,
    OrderBook, Position, RuntimeOrderContext, StateParams, StrategyParamsPairValue,
    TrailingPriceBundle, TwelEnforcerPolicy,
};
use ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::PyObject;
use serde::Serialize;
use serde_json::{self, Value};
use std::str::FromStr;
use std::time::Instant;

type BacktestPyResult = (PyObject, PyObject, Py<PyDict>, Py<PyDict>, PyObject);

/// Native CPU market execution counts, SELL lane0 and BUY lane1.
///
/// The borrowed f64 input may be strided. Windows use global inclusive row
/// indices; only cells outside a declared window receive the zero sentinel.
#[pyfunction]
#[pyo3(signature = (closes_f64_2d_strided, price_steps_f64_list, slippage_f64, first_valid_indices_list, last_valid_indices_list, row_offset_int=0))]
pub fn prepare_gpu_market_execution_ticks<'py>(
    py: Python<'py>,
    closes_f64_2d_strided: PyReadonlyArray2<'py, f64>,
    price_steps_f64_list: Vec<f64>,
    slippage_f64: f64,
    first_valid_indices_list: Vec<i64>,
    last_valid_indices_list: Vec<i64>,
    row_offset_int: i64,
) -> PyResult<Bound<'py, PyArray3<i32>>> {
    let closes = closes_f64_2d_strided.as_array();
    let (rows, coins) = closes.dim();
    if coins == 0
        || price_steps_f64_list.len() != coins
        || first_valid_indices_list.len() != coins
        || last_valid_indices_list.len() != coins
    {
        return Err(PyValueError::new_err(
            "GPU market closes require a nonempty coin axis and one step/window per coin",
        ));
    }
    if !slippage_f64.is_finite() || slippage_f64 < 0.0 {
        return Err(PyValueError::new_err(
            "GPU market slippage must be finite and nonnegative",
        ));
    }
    let row_count = i64::try_from(rows)
        .map_err(|_| PyValueError::new_err("GPU market row count exceeds int64"))?;
    if row_offset_int < 0 || row_offset_int.checked_add(row_count).is_none() {
        return Err(PyValueError::new_err(
            "GPU market row offset must be nonnegative and its chunk end must fit int64",
        ));
    }
    for coin in 0..coins {
        let step = price_steps_f64_list[coin];
        if !step.is_finite() || step < 0.0 {
            return Err(PyValueError::new_err(format!(
                "GPU market price step must be finite and nonnegative for coin {coin}"
            )));
        }
        let first = first_valid_indices_list[coin];
        let last = last_valid_indices_list[coin];
        if first <= last && first < 0 {
            return Err(PyValueError::new_err(format!(
                "Active GPU market window must use nonnegative indices for coin {coin}"
            )));
        }
    }
    let count = rows
        .checked_mul(coins)
        .and_then(|cells| cells.checked_mul(2))
        .filter(|count| *count <= isize::MAX as usize / std::mem::size_of::<i32>())
        .ok_or_else(|| {
            PyValueError::new_err("GPU market output shape overflows addressable size")
        })?;
    let mut values = Vec::<i32>::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| PyMemoryError::new_err("Unable to allocate GPU market tick output"))?;
    values.resize(count, 0);
    for row in 0..rows {
        let global_row = row_offset_int + row as i64;
        for coin in 0..coins {
            if global_row < first_valid_indices_list[coin]
                || global_row > last_valid_indices_list[coin]
            {
                continue;
            }
            let ticks = crate::utils::gpu_market_execution_tick_counts(
                closes[(row, coin)],
                price_steps_f64_list[coin],
                slippage_f64,
            )
            .map_err(|message| {
                PyValueError::new_err(format!("{message}; global row {global_row}, coin {coin}"))
            })?;
            let start = (row * coins + coin) * 2;
            values[start..start + 2].copy_from_slice(&ticks);
        }
    }
    let output = Array3::from_shape_vec((rows, coins, 2), values)
        .map_err(|_| PyValueError::new_err("Invalid GPU market tick output shape"))?;
    Ok(output.into_pyarray_bound(py))
}

const OPTIMIZE_PROFILE_ENV: &str = "PASSIVBOT_OPTIMIZE_PROFILE";
const RUST_PROFILE_KEY: &str = "_rust_profile";

fn rust_profile_enabled() -> bool {
    match std::env::var(OPTIMIZE_PROFILE_ENV) {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !value.is_empty() && value != "0" && value != "false" && value != "no"
        }
        Err(_) => false,
    }
}

fn profile_start(enabled: bool) -> Option<Instant> {
    if enabled {
        Some(Instant::now())
    } else {
        None
    }
}

fn profile_add(
    profile: &mut Vec<(&'static str, f64)>,
    key: &'static str,
    started: Option<Instant>,
) {
    if let Some(started) = started {
        profile.push((key, started.elapsed().as_secs_f64() * 1000.0));
    }
}

fn profile_to_py_dict(
    py: Python<'_>,
    profile: &[(&'static str, f64)],
    total_started: Option<Instant>,
) -> PyResult<Py<PyDict>> {
    let py_profile = PyDict::new_bound(py);
    for (key, value) in profile {
        py_profile.set_item(*key, *value)?;
    }
    if let Some(started) = total_started {
        py_profile.set_item("rust_total_ms", started.elapsed().as_secs_f64() * 1000.0)?;
    }
    Ok(py_profile.unbind())
}

#[pyfunction(signature = (equity, timestamps_ms, half_life_days, candle_interval_minutes=1))]
pub fn calculate_ew_metrics(
    py: Python<'_>,
    equity: Vec<f64>,
    timestamps_ms: Vec<u64>,
    half_life_days: f64,
    candle_interval_minutes: u64,
) -> PyResult<PyObject> {
    let result = crate::ew::analyze_ew_equity(
        &equity,
        &timestamps_ms,
        half_life_days,
        candle_interval_minutes,
    )
    .map_err(PyValueError::new_err)?;
    Ok(struct_to_py_dict(py, &result)?.into_py(py))
}

fn attach_ew_analysis(
    py: Python<'_>,
    dict: &Py<PyDict>,
    curve_name: &str,
    curve: &crate::ew::EwAnalysis,
    strategy: &crate::ew::EwAnalysis,
) -> PyResult<()> {
    for (analysis, suffix) in [(curve, "_ew"), (strategy, "_strategy_eq_ew")] {
        if let Some(metrics) = analysis.metrics {
            for (key, value) in [
                ("adg", metrics.adg),
                ("sharpe_ratio", metrics.sharpe_ratio),
                ("sortino_ratio", metrics.sortino_ratio),
                ("omega_ratio", metrics.omega_ratio),
            ] {
                dict.bind(py).set_item(format!("{key}{suffix}"), value)?;
            }
        }
    }
    let metadata = PyDict::new_bound(py);
    metadata.set_item(curve_name, struct_to_py_dict(py, &curve.metadata)?)?;
    metadata.set_item("strategy_eq", struct_to_py_dict(py, &strategy.metadata)?)?;
    dict.bind(py).set_item("_ew_metadata", metadata)?;
    Ok(())
}



fn apply_fill_activity_metrics(analysis: &mut Analysis, metrics: FillActivityMetrics) {
    analysis.fills_active_days_count = metrics.fills_active_days_count;
    analysis.fills_active_days_ratio = metrics.fills_active_days_ratio;
    analysis.fills_active_symbols_count = metrics.fills_active_symbols_count;
    analysis.fills_analysis_duration_days = metrics.fills_analysis_duration_days;
    analysis.fills_count = metrics.fills_count;
    analysis.fills_count_close = metrics.fills_count_close;
    analysis.fills_count_entry = metrics.fills_count_entry;
    analysis.fills_count_long = metrics.fills_count_long;
    analysis.fills_count_short = metrics.fills_count_short;
    analysis.fills_entry_per_close = metrics.fills_entry_per_close;
    analysis.fills_gap_longest_days = metrics.fills_gap_longest_days;
    analysis.fills_gap_mean_hours = metrics.fills_gap_mean_hours;
    analysis.fills_gap_median_hours = metrics.fills_gap_median_hours;
    analysis.fills_gap_p95_hours = metrics.fills_gap_p95_hours;
    analysis.fills_gap_p99_hours = metrics.fills_gap_p99_hours;
    analysis.fills_gap_time_weighted_mean_hours = metrics.fills_gap_time_weighted_mean_hours;
    analysis.fills_per_day = metrics.fills_per_day;
    analysis.fills_per_day_close = metrics.fills_per_day_close;
    analysis.fills_per_day_entry = metrics.fills_per_day_entry;
    analysis.fills_per_day_long = metrics.fills_per_day_long;
    analysis.fills_per_day_per_position_slot = metrics.fills_per_day_per_position_slot;
    analysis.fills_per_day_per_position_slot_long = metrics.fills_per_day_per_position_slot_long;
    analysis.fills_per_day_per_position_slot_short = metrics.fills_per_day_per_position_slot_short;
    analysis.fills_per_day_short = metrics.fills_per_day_short;
    analysis.fills_top_symbol_share = metrics.fills_top_symbol_share;
}

fn calc_backtest_completion_ratio(
    equities: &Equities,
    backtest_params: &BacktestParams,
    n_timesteps: usize,
) -> f64 {
    let interval_ms = backtest_params
        .candle_interval_minutes
        .max(1)
        .saturating_mul(60_000);
    let requested_start_ts = backtest_params.requested_start_timestamp_ms;
    let requested_end_ts = backtest_params
        .first_timestamp_ms
        .saturating_add((n_timesteps as u64).saturating_mul(interval_ms));
    if requested_end_ts <= requested_start_ts {
        return 1.0;
    }
    let Some(&first_equity_ts) = equities.timestamps_ms.first() else {
        return 0.0;
    };
    let Some(&last_equity_ts) = equities.timestamps_ms.last() else {
        return 0.0;
    };
    if last_equity_ts < first_equity_ts {
        return 0.0;
    }
    let final_covered_ts = last_equity_ts.saturating_add(interval_ms);
    let covered_end_ts = requested_end_ts.min(final_covered_ts);
    let covered_span = covered_end_ts.saturating_sub(requested_start_ts);
    let requested_span = requested_end_ts.saturating_sub(requested_start_ts);
    if requested_span == 0 {
        return 1.0;
    }
    ((covered_span as f64) / (requested_span as f64)).clamp(0.0, 1.0)
}

#[pyclass(name = "HlcvsBundle", module = "passivbot_rust", unsendable)]
pub struct HlcvsBundlePy {
    pub inner: HlcvsBundle,
}

#[pymethods]
impl HlcvsBundlePy {
    #[new]
    #[pyo3(signature = (hlcvs, btc_usd, timestamps, meta, eligibility=None, eligibility_rows=None))]
    pub fn new(
        hlcvs: Py<PyArray3<f64>>,
        btc_usd: Py<PyArray1<f64>>,
        timestamps: Py<PyArray1<i64>>,
        meta: &Bound<'_, PyAny>,
        eligibility: Option<Py<PyArray2<u8>>>,
        eligibility_rows: Option<Py<PyArray1<i32>>>,
    ) -> PyResult<Self> {
        let parsed_meta = hlcvs_meta_from_py(meta)?;
        let bundle = HlcvsBundle {
            hlcvs,
            btc_usd,
            timestamps,
            meta: parsed_meta,
            eligibility,
            eligibility_rows,
        };
        Python::with_gil(|py| bundle.validate_shapes(py))?;
        Ok(Self { inner: bundle })
    }

    #[getter]
    pub fn hlcvs<'py>(&self, py: Python<'py>) -> PyObject {
        self.inner.hlcvs.clone_ref(py).into_py(py)
    }

    #[getter]
    pub fn btc_usd<'py>(&self, py: Python<'py>) -> PyObject {
        self.inner.btc_usd.clone_ref(py).into_py(py)
    }

    #[getter]
    pub fn timestamps<'py>(&self, py: Python<'py>) -> PyObject {
        self.inner.timestamps.clone_ref(py).into_py(py)
    }

    #[getter]
    pub fn eligibility(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .eligibility
            .as_ref()
            .map(|a| a.clone_ref(py).into_py(py))
    }
    #[getter]
    pub fn eligibility_rows(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .eligibility_rows
            .as_ref()
            .map(|a| a.clone_ref(py).into_py(py))
    }

    #[getter]
    pub fn meta<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(hlcvs_meta_to_dict(py, &self.inner.meta)?.into_py(py))
    }

    pub fn coin_meta<'py>(&self, symbol: &str, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if let Some(meta) = self.inner.coin_meta_by_symbol(symbol) {
            Ok(Some(coin_meta_to_dict(py, meta)?.into_py(py)))
        } else {
            Ok(None)
        }
    }

    pub fn coins_len(&self) -> usize {
        self.inner.coins_len()
    }
}

fn hlcvs_meta_from_py(any: &Bound<'_, PyAny>) -> PyResult<HlcvsMeta> {
    let dict = any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("hlcvs meta must be a dict"))?;

    let requested = get_u64(dict, "requested_start_timestamp_ms")?;
    let effective = get_u64(dict, "effective_start_timestamp_ms")?;
    let warm_req = get_u64(dict, "warmup_minutes_requested")?;
    let warm_prov = get_u64(dict, "warmup_minutes_provided")?;

    let coins_obj = dict
        .get_item("coins")?
        .ok_or_else(|| PyValueError::new_err("hlcvs meta missing 'coins'"))?;
    let coins_list = coins_obj
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("'coins' must be a list"))?;
    let mut coins = Vec::with_capacity(coins_list.len());
    for (idx, item) in coins_list.iter().enumerate() {
        let coin_dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("coin metadata entries must be dicts"))?;
        coins.push(parse_coin_meta(coin_dict, idx)?);
    }
    let active_coin_indices = match dict.get_item("active_coin_indices")? {
        Some(item) if !item.is_none() => Some(item.extract::<Vec<usize>>()?),
        _ => None,
    };

    Ok(HlcvsMeta {
        requested_start_timestamp_ms: requested,
        effective_start_timestamp_ms: effective,
        warmup_minutes_requested: warm_req,
        warmup_minutes_provided: warm_prov,
        coins,
        active_coin_indices,
    })
}

fn get_u64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<u64> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("hlcvs meta missing '{key}'")))?
        .extract::<u64>()
        .map_err(|_| PyValueError::new_err(format!("hlcvs meta field '{key}' must be int")))
}

fn parse_coin_meta(dict: &Bound<'_, PyDict>, default_index: usize) -> PyResult<CoinMeta> {
    let index = dict
        .get_item("index")?
        .map(|val| val.extract::<usize>())
        .transpose()?
        .unwrap_or(default_index);
    let symbol = dict
        .get_item("symbol")?
        .ok_or_else(|| PyValueError::new_err("coin meta missing 'symbol'"))?
        .extract::<String>()?;
    let coin = dict
        .get_item("coin")?
        .map(|val| val.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| symbol.clone());
    let exchange = dict
        .get_item("exchange")?
        .ok_or_else(|| PyValueError::new_err("coin meta missing 'exchange'"))?
        .extract::<String>()?;
    let quote = dict
        .get_item("quote")?
        .ok_or_else(|| PyValueError::new_err("coin meta missing 'quote'"))?
        .extract::<String>()?;
    let base = dict
        .get_item("base")?
        .ok_or_else(|| PyValueError::new_err("coin meta missing 'base'"))?
        .extract::<String>()?;

    macro_rules! fval {
        ($key:literal) => {
            dict.get_item($key)?
                .ok_or_else(|| PyValueError::new_err(format!("coin meta missing '{}'", $key)))?
                .extract::<f64>()
                .map_err(|_| {
                    PyValueError::new_err(format!("coin meta field '{}' must be float", $key))
                })?
        };
    }

    macro_rules! usize_val {
        ($key:literal) => {
            dict.get_item($key)?
                .ok_or_else(|| PyValueError::new_err(format!("coin meta missing '{}'", $key)))?
                .extract::<usize>()
                .map_err(|_| {
                    PyValueError::new_err(format!("coin meta field '{}' must be int", $key))
                })?
        };
    }

    Ok(CoinMeta {
        index,
        symbol,
        coin,
        exchange,
        quote,
        base,
        qty_step: fval!("qty_step"),
        price_step: fval!("price_step"),
        min_qty: fval!("min_qty"),
        min_cost: fval!("min_cost"),
        c_mult: fval!("c_mult"),
        maker_fee: fval!("maker_fee"),
        taker_fee: fval!("taker_fee"),
        first_valid_index: usize_val!("first_valid_index"),
        last_valid_index: usize_val!("last_valid_index"),
        warmup_minutes: usize_val!("warmup_minutes"),
        trade_start_index: usize_val!("trade_start_index"),
    })
}

fn coin_meta_to_dict<'py>(py: Python<'py>, meta: &CoinMeta) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("index", meta.index)?;
    dict.set_item("symbol", &meta.symbol)?;
    dict.set_item("coin", &meta.coin)?;
    dict.set_item("exchange", &meta.exchange)?;
    dict.set_item("quote", &meta.quote)?;
    dict.set_item("base", &meta.base)?;
    dict.set_item("qty_step", meta.qty_step)?;
    dict.set_item("price_step", meta.price_step)?;
    dict.set_item("min_qty", meta.min_qty)?;
    dict.set_item("min_cost", meta.min_cost)?;
    dict.set_item("c_mult", meta.c_mult)?;
    dict.set_item("maker_fee", meta.maker_fee)?;
    dict.set_item("taker_fee", meta.taker_fee)?;
    dict.set_item("first_valid_index", meta.first_valid_index)?;
    dict.set_item("last_valid_index", meta.last_valid_index)?;
    dict.set_item("warmup_minutes", meta.warmup_minutes)?;
    dict.set_item("trade_start_index", meta.trade_start_index)?;
    Ok(dict)
}

fn hlcvs_meta_to_dict<'py>(py: Python<'py>, meta: &HlcvsMeta) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "requested_start_timestamp_ms",
        meta.requested_start_timestamp_ms,
    )?;
    dict.set_item(
        "effective_start_timestamp_ms",
        meta.effective_start_timestamp_ms,
    )?;
    dict.set_item("warmup_minutes_requested", meta.warmup_minutes_requested)?;
    dict.set_item("warmup_minutes_provided", meta.warmup_minutes_provided)?;
    let coins = PyList::empty_bound(py);
    for entry in &meta.coins {
        coins.append(coin_meta_to_dict(py, entry)?)?;
    }
    dict.set_item("coins", coins)?;
    if let Some(active_coin_indices) = &meta.active_coin_indices {
        dict.set_item("active_coin_indices", active_coin_indices)?;
    }
    Ok(dict)
}

#[pyfunction]
pub fn trailing_bundle_default_py() -> (f64, f64, f64, f64) {
    let bundle = TrailingPriceBundle::default();
    trailing_bundle_to_tuple(&bundle)
}

#[pyfunction]
#[pyo3(signature = (highs, lows, closes, bundle=None))]
pub fn update_trailing_bundle_py(
    highs: PyReadonlyArray1<'_, f64>,
    lows: PyReadonlyArray1<'_, f64>,
    closes: PyReadonlyArray1<'_, f64>,
    bundle: Option<(f64, f64, f64, f64)>,
) -> PyResult<(f64, f64, f64, f64)> {
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;

    let len = highs.len();
    if lows.len() != len || closes.len() != len {
        return Err(PyValueError::new_err(
            "highs, lows, and closes must have the same length",
        ));
    }

    let mut bundle = match bundle {
        Some(values) => tuple_to_trailing_bundle(values),
        None => TrailingPriceBundle::default(),
    };

    update_trailing_bundle_sequence(&mut bundle, highs, lows, closes);

    Ok(trailing_bundle_to_tuple(&bundle))
}

fn parse_strategy_side_str(side: &str) -> PyResult<StrategySide> {
    match side {
        "long" => Ok(StrategySide::Long),
        "short" => Ok(StrategySide::Short),
        _ => Err(PyValueError::new_err(
            "side must be either 'long' or 'short'",
        )),
    }
}

#[inline]
fn clamp_alpha(alpha: f64) -> f64 {
    if !alpha.is_finite() || alpha < 0.0 {
        0.0
    } else if alpha > 1.0 {
        1.0
    } else {
        alpha
    }
}

#[inline(always)]
fn update_adjusted_ema_py(
    value: f64,
    alpha: f64,
    numerator: &mut f64,
    denominator: &mut f64,
) -> f64 {
    if !value.is_finite() {
        return if *denominator > 0.0 {
            *numerator / *denominator
        } else {
            value
        };
    }
    if alpha <= 0.0 || !alpha.is_finite() {
        return if *denominator > 0.0 {
            *numerator / *denominator
        } else {
            value
        };
    }
    let one_minus_alpha = 1.0 - alpha;
    let new_num = alpha * value + one_minus_alpha * *numerator;
    let new_den = alpha + one_minus_alpha * *denominator;
    if !new_den.is_finite() || new_den <= f64::MIN_POSITIVE {
        *numerator = alpha * value;
        *denominator = alpha;
        return value;
    }
    *numerator = new_num;
    *denominator = new_den;
    new_num / new_den
}

fn validate_ema_anchor_series_inputs(
    timestamps: &[i64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    balances: &[f64],
    position_sizes: &[f64],
    price_step: f64,
) -> PyResult<()> {
    let len = timestamps.len();
    if len == 0 {
        return Err(PyValueError::new_err(
            "timestamps/highs/lows/closes must not be empty",
        ));
    }
    if highs.len() != len
        || lows.len() != len
        || closes.len() != len
        || balances.len() != len
        || position_sizes.len() != len
    {
        return Err(PyValueError::new_err(
            "timestamps, highs, lows, closes, balances, and position_sizes must have the same length",
        ));
    }
    if !price_step.is_finite() || price_step <= 0.0 {
        return Err(PyValueError::new_err("price_step must be > 0"));
    }
    for i in 0..len {
        if i > 0 && timestamps[i] <= timestamps[i - 1] {
            return Err(PyValueError::new_err(format!(
                "timestamps must be strictly increasing; index {i} ({}) <= previous ({})",
                timestamps[i],
                timestamps[i - 1]
            )));
        }
        if !highs[i].is_finite() || highs[i] <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "highs[{i}] must be finite and > 0"
            )));
        }
        if !lows[i].is_finite() || lows[i] <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "lows[{i}] must be finite and > 0"
            )));
        }
        if !closes[i].is_finite() || closes[i] <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "closes[{i}] must be finite and > 0"
            )));
        }
        if !balances[i].is_finite() || balances[i] < 0.0 {
            return Err(PyValueError::new_err(format!(
                "balances[{i}] must be finite and >= 0"
            )));
        }
        if !position_sizes[i].is_finite() {
            return Err(PyValueError::new_err(format!(
                "position_sizes[{i}] must be finite"
            )));
        }
    }
    Ok(())
}

fn calc_ema_anchor_quote_series(
    side: StrategySide,
    timestamps: &[i64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    balances: &[f64],
    position_sizes: &[f64],
    price_step: f64,
    params: EmaAnchorParams,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    validate_ema_anchor_series_inputs(
        timestamps,
        highs,
        lows,
        closes,
        balances,
        position_sizes,
        price_step,
    )?;

    let exchange = ExchangeParams {
        price_step,
        ..Default::default()
    };

    let ema_alpha_0 = clamp_alpha(2.0 / (params.ema_span_0 + 1.0));
    let ema_alpha_1 = clamp_alpha(2.0 / (params.ema_span_1 + 1.0));
    let ema_alpha_2 = clamp_alpha(2.0 / ((params.ema_span_0 * params.ema_span_1).sqrt() + 1.0));
    let vol_1m_alpha = if params.offset_volatility_ema_span_1m > 0.0 {
        clamp_alpha(2.0 / (params.offset_volatility_ema_span_1m + 1.0))
    } else {
        0.0
    };
    let vol_1h_alpha = if params.offset_volatility_ema_span_1h > 0.0 {
        clamp_alpha(2.0 / (params.offset_volatility_ema_span_1h + 1.0))
    } else {
        0.0
    };

    let base_close = closes[0];
    let mut ema_0_num = base_close;
    let mut ema_1_num = base_close;
    let mut ema_2_num = base_close;
    let mut ema_0_den = 1.0;
    let mut ema_1_den = 1.0;
    let mut ema_2_den = 1.0;

    let mut vol_1m_num = 0.0;
    let mut vol_1m_den = 1.0;

    let mut vol_1h = 0.0;
    let mut vol_1h_num = 0.0;
    let mut vol_1h_den = 1.0;

    let first_ts_ms = timestamps[0] as u64;
    let mut current_hour_boundary = (first_ts_ms / 3_600_000u64) * 3_600_000u64;
    let mut bucket_high = f64::MIN;
    let mut bucket_low = f64::MAX;
    let mut bucket_seen = false;

    let mut bids = Vec::with_capacity(timestamps.len());
    let mut asks = Vec::with_capacity(timestamps.len());

    for i in 0..timestamps.len() {
        let current_ts = timestamps[i] as u64;
        let high = highs[i];
        let low = lows[i];
        let close = closes[i];
        let hour_boundary = (current_ts / 3_600_000u64) * 3_600_000u64;
        if hour_boundary > current_hour_boundary {
            if bucket_seen && bucket_high > 0.0 && bucket_low > 0.0 {
                let hour_log_range = (bucket_high / bucket_low).ln();
                vol_1h = update_adjusted_ema_py(
                    hour_log_range,
                    vol_1h_alpha,
                    &mut vol_1h_num,
                    &mut vol_1h_den,
                );
            }
            current_hour_boundary = hour_boundary;
            bucket_high = f64::MIN;
            bucket_low = f64::MAX;
        }

        let ema_0 = update_adjusted_ema_py(close, ema_alpha_0, &mut ema_0_num, &mut ema_0_den);
        let ema_1 = update_adjusted_ema_py(close, ema_alpha_1, &mut ema_1_num, &mut ema_1_den);
        let ema_2 = update_adjusted_ema_py(close, ema_alpha_2, &mut ema_2_num, &mut ema_2_den);

        let log_range = (high / low).ln();
        let vol_1m =
            update_adjusted_ema_py(log_range, vol_1m_alpha, &mut vol_1m_num, &mut vol_1m_den);

        let lower = ema_0.min(ema_1).min(ema_2);
        let upper = ema_0.max(ema_1).max(ema_2);
        let state = StateParams {
            balance: balances[i],
            order_book: OrderBook {
                bid: close,
                ask: close,
            },
            ema_bands: EMABands { upper, lower },
            volatility_ema_1m: vol_1m,
            volatility_ema_1h: vol_1h,
        };
        let (bid, ask) =
            calc_ema_anchor_quote_prices(&state, &exchange, &params, position_sizes[i], 1.0);
        bids.push(bid);
        asks.push(ask);

        if high > bucket_high {
            bucket_high = high;
        }
        if low < bucket_low {
            bucket_low = low;
        }
        bucket_seen = true;
    }

    match side {
        StrategySide::Long | StrategySide::Short => Ok((bids, asks)),
    }
}

#[pyfunction(signature = (
    side,
    timestamps,
    highs,
    lows,
    closes,
    balances,
    position_sizes,
    *,
    price_step,
    ema_span_0,
    ema_span_1,
    offset,
    offset_volatility_ema_span_1m,
    offset_volatility_1m_weight,
    offset_volatility_ema_span_1h,
    offset_volatility_1h_weight,
    offset_psize_weight
))]
pub fn calc_ema_anchor_quote_series_py<'py>(
    py: Python<'py>,
    side: &str,
    timestamps: PyReadonlyArray1<'_, i64>,
    highs: PyReadonlyArray1<'_, f64>,
    lows: PyReadonlyArray1<'_, f64>,
    closes: PyReadonlyArray1<'_, f64>,
    balances: PyReadonlyArray1<'_, f64>,
    position_sizes: PyReadonlyArray1<'_, f64>,
    price_step: f64,
    ema_span_0: f64,
    ema_span_1: f64,
    offset: f64,
    offset_volatility_ema_span_1m: f64,
    offset_volatility_1m_weight: f64,
    offset_volatility_ema_span_1h: f64,
    offset_volatility_1h_weight: f64,
    offset_psize_weight: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let side = parse_strategy_side_str(side)?;
    let params = EmaAnchorParams {
        base_qty_pct: 0.0,
        ema_span_0,
        ema_span_1,
        entry_double_down_factor: 0.0,
        offset,
        offset_volatility_ema_span_1m,
        offset_volatility_1m_weight,
        offset_volatility_ema_span_1h,
        offset_volatility_1h_weight,
        offset_psize_weight,
    };
    let (bids, asks) = calc_ema_anchor_quote_series(
        side,
        timestamps.as_slice()?,
        highs.as_slice()?,
        lows.as_slice()?,
        closes.as_slice()?,
        balances.as_slice()?,
        position_sizes.as_slice()?,
        price_step,
        params,
    )?;
    Ok((
        bids.into_pyarray_bound(py).unbind(),
        asks.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
pub fn gate_entries_by_twel_py(
    side: &str,
    balance: f64,
    total_wallet_exposure_limit: f64,
    positions: &Bound<'_, PyList>,
    entries: &Bound<'_, PyList>,
) -> PyResult<Vec<(usize, f64, f64, u16)>> {
    let positions = positions.as_ref();
    let entries = entries.as_ref();
    let side_code = match side {
        "long" => LONG,
        "short" => SHORT,
        _ => {
            return Err(PyValueError::new_err(
                "side must be either 'long' or 'short'",
            ))
        }
    };
    if balance <= 0.0 || total_wallet_exposure_limit <= 0.0 {
        return Ok(vec![]);
    }

    let positions_len = positions.len()?;
    let mut positions_vec: Vec<GateEntriesPosition> = Vec::with_capacity(positions_len);
    for item in positions.iter()? {
        let item = item?;
        let dict = item.downcast::<PyDict>()?;
        let idx = dict
            .get_item("idx")?
            .ok_or_else(|| PyValueError::new_err("position missing 'idx'"))?
            .extract::<usize>()?;
        let position_size = dict
            .get_item("position_size")?
            .ok_or_else(|| PyValueError::new_err("position missing 'position_size'"))?
            .extract::<f64>()?;
        let position_price = dict
            .get_item("position_price")?
            .ok_or_else(|| PyValueError::new_err("position missing 'position_price'"))?
            .extract::<f64>()?;
        let c_mult = dict
            .get_item("c_mult")?
            .ok_or_else(|| PyValueError::new_err("position missing 'c_mult'"))?
            .extract::<f64>()?;
        positions_vec.push(GateEntriesPosition {
            idx,
            position_size,
            position_price,
            c_mult,
        });
    }

    let entries_len = entries.len()?;
    let mut entries_vec: Vec<GateEntriesCandidate> = Vec::with_capacity(entries_len);
    for item in entries.iter()? {
        let item = item?;
        let dict = item.downcast::<PyDict>()?;
        let idx = dict
            .get_item("idx")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'idx'"))?
            .extract::<usize>()?;
        let qty = dict
            .get_item("qty")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'qty'"))?
            .extract::<f64>()?;
        let price = dict
            .get_item("price")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'price'"))?
            .extract::<f64>()?;
        let qty_step = dict
            .get_item("qty_step")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'qty_step'"))?
            .extract::<f64>()?;
        let min_qty = dict
            .get_item("min_qty")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'min_qty'"))?
            .extract::<f64>()?;
        let min_cost = dict
            .get_item("min_cost")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'min_cost'"))?
            .extract::<f64>()?;
        let c_mult = dict
            .get_item("c_mult")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'c_mult'"))?
            .extract::<f64>()?;
        let market_price = dict
            .get_item("market_price")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'market_price'"))?
            .extract::<f64>()?;
        let order_type_id = dict
            .get_item("order_type_id")?
            .ok_or_else(|| PyValueError::new_err("entry missing 'order_type_id'"))?
            .extract::<u16>()?;
        let order_type = OrderType::try_from(order_type_id).map_err(|_| {
            PyValueError::new_err("unknown order_type_id provided to gate_entries_by_twel")
        })?;
        entries_vec.push(GateEntriesCandidate {
            idx,
            qty,
            price,
            qty_step,
            min_qty,
            min_cost,
            c_mult,
            market_price,
            order_type,
        });
    }

    let gated = gate_entries_by_twel(
        side_code,
        balance,
        total_wallet_exposure_limit,
        &positions_vec,
        &entries_vec,
    );

    let mut result: Vec<(usize, f64, f64, u16)> = Vec::with_capacity(gated.len());
    for GateEntriesDecision {
        idx,
        qty,
        price,
        order_type,
        ..
    } in gated
    {
        result.push((idx, qty, price, order_type.id()));
    }
    Ok(result)
}

#[pyfunction]
pub fn calc_unstucking_close_py(
    balance: f64,
    allowance_long: f64,
    allowance_short: f64,
    positions: &Bound<'_, PyList>,
) -> PyResult<Option<(usize, usize, f64, f64, u16)>> {
    let positions = positions.as_ref();
    let positions_len = positions.len()?;
    let mut inputs: Vec<UnstuckPositionInput> = Vec::with_capacity(positions_len);
    for item in positions.iter()? {
        let item = item?;
        let dict = item.downcast::<PyDict>()?;
        let idx = dict
            .get_item("idx")?
            .ok_or_else(|| PyValueError::new_err("position missing 'idx'"))?
            .extract::<usize>()?;
        let side_str: String = dict
            .get_item("side")?
            .ok_or_else(|| PyValueError::new_err("position missing 'side'"))?
            .extract::<String>()?;
        let side = match side_str.as_str() {
            "long" => LONG,
            "short" => SHORT,
            _ => {
                return Err(PyValueError::new_err(
                    "position side must be 'long' or 'short'",
                ))
            }
        };
        let position_size = dict
            .get_item("position_size")?
            .ok_or_else(|| PyValueError::new_err("position missing 'position_size'"))?
            .extract::<f64>()?;
        let position_price = dict
            .get_item("position_price")?
            .ok_or_else(|| PyValueError::new_err("position missing 'position_price'"))?
            .extract::<f64>()?;
        let wallet_exposure_limit = dict
            .get_item("wallet_exposure_limit")?
            .ok_or_else(|| PyValueError::new_err("position missing 'wallet_exposure_limit'"))?
            .extract::<f64>()?;
        let effective_we_excess_allowance_pct =
            if let Some(value) = dict.get_item("effective_we_excess_allowance_pct")? {
                value.extract::<f64>()?
            } else {
                dict.get_item("risk_we_excess_allowance_pct")?
                    .ok_or_else(|| {
                        PyValueError::new_err("position missing 'risk_we_excess_allowance_pct'")
                    })?
                    .extract::<f64>()?
            };
        let unstuck_threshold = dict
            .get_item("unstuck_threshold")?
            .ok_or_else(|| PyValueError::new_err("position missing 'unstuck_threshold'"))?
            .extract::<f64>()?;
        let unstuck_close_pct = dict
            .get_item("unstuck_close_pct")?
            .ok_or_else(|| PyValueError::new_err("position missing 'unstuck_close_pct'"))?
            .extract::<f64>()?;
        let unstuck_ema_gating_enabled =
            if let Some(value) = dict.get_item("unstuck_ema_gating_enabled")? {
                value.extract::<bool>()?
            } else {
                true
            };
        let unstuck_ema_dist = dict
            .get_item("unstuck_ema_dist")?
            .ok_or_else(|| PyValueError::new_err("position missing 'unstuck_ema_dist'"))?
            .extract::<f64>()?;
        let ema_band_upper = dict
            .get_item("ema_band_upper")?
            .ok_or_else(|| PyValueError::new_err("position missing 'ema_band_upper'"))?
            .extract::<f64>()?;
        let ema_band_lower = dict
            .get_item("ema_band_lower")?
            .ok_or_else(|| PyValueError::new_err("position missing 'ema_band_lower'"))?
            .extract::<f64>()?;
        let current_price = dict
            .get_item("current_price")?
            .ok_or_else(|| PyValueError::new_err("position missing 'current_price'"))?
            .extract::<f64>()?;
        let price_step = dict
            .get_item("price_step")?
            .ok_or_else(|| PyValueError::new_err("position missing 'price_step'"))?
            .extract::<f64>()?;
        let qty_step = dict
            .get_item("qty_step")?
            .ok_or_else(|| PyValueError::new_err("position missing 'qty_step'"))?
            .extract::<f64>()?;
        let min_qty = dict
            .get_item("min_qty")?
            .ok_or_else(|| PyValueError::new_err("position missing 'min_qty'"))?
            .extract::<f64>()?;
        let min_cost = dict
            .get_item("min_cost")?
            .ok_or_else(|| PyValueError::new_err("position missing 'min_cost'"))?
            .extract::<f64>()?;
        let c_mult = dict
            .get_item("c_mult")?
            .ok_or_else(|| PyValueError::new_err("position missing 'c_mult'"))?
            .extract::<f64>()?;

        inputs.push(UnstuckPositionInput {
            idx,
            side,
            position_size,
            position_price,
            wallet_exposure_limit,
            effective_we_excess_allowance_pct,
            unstuck_threshold,
            unstuck_close_pct,
            unstuck_ema_gating_enabled,
            unstuck_ema_dist,
            unstuck_loss_allowance_pct: 0.0,
            total_wallet_exposure_limit: 0.0,
            ema_band_upper,
            ema_band_lower,
            current_price,
            price_step,
            qty_step,
            min_qty,
            min_cost,
            c_mult,
        });
    }

    let result = calc_unstucking_action(balance, allowance_long, allowance_short, &inputs);
    Ok(result.map(|(idx, side, order)| (idx, side, order.qty, order.price, order.order_type.id())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prepare_freethreaded_python;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn strategy_bindings_reject_unknown_and_non_string_keys_before_projection() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            for kind in ["trailing_martingale", "ema_anchor"] {
                let value = if kind == "trailing_martingale" {
                    serde_json::to_value(crate::strategies::TrailingMartingaleParams::default())
                        .unwrap()
                } else {
                    serde_json::to_value(EmaAnchorParams::default()).unwrap()
                };
                let mut paths = vec![vec![], vec!["long"], vec!["short"]];
                if kind == "trailing_martingale" {
                    paths.extend([
                        vec!["long", "entry"],
                        vec!["long", "close"],
                        vec!["short", "entry"],
                        vec!["short", "close"],
                    ]);
                }
                for path in paths {
                    for non_string in [false, true] {
                        let raw = serde_json::json!({"long":value,"short":value});
                        let object = json_value_to_py(py, &raw).unwrap();
                        let dict = object.bind(py).downcast::<PyDict>().unwrap();
                        let mut target = dict.as_gil_ref();
                        for part in &path {
                            target = target
                                .get_item(part)
                                .unwrap()
                                .unwrap()
                                .downcast::<PyDict>()
                                .unwrap();
                        }
                        if non_string {
                            target.set_item(7, 1.0).unwrap();
                        } else {
                            target.set_item("typo_unknown_key", 1.0).unwrap();
                        }
                        let error = strategy_params_pair_from_dict(dict.as_gil_ref(), kind)
                            .expect_err(&format!("{kind} {path:?} non_string={non_string}"));
                        let expected = if non_string {
                            "non-string key"
                        } else {
                            "typo_unknown_key"
                        };
                        assert!(error.to_string().contains(expected), "{error}");
                    }
                }
            }
        });
    }

    #[test]
    fn strategy_binding_optional_defaults_remain_unchanged() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = crate::strategies::TrailingMartingaleParams::default();
            let object = struct_to_py_dict(py, &value).unwrap();
            let dict = object.bind(py);
            dict.get_item("entry")
                .unwrap()
                .unwrap()
                .downcast::<PyDict>()
                .unwrap()
                .del_item("ema_gate_mode")
                .unwrap();
            let parsed = trailing_martingale_strategy_params_from_dict(dict.as_gil_ref()).unwrap();
            assert_eq!(parsed["entry"]["ema_gate_mode"], "initial");

            let object = struct_to_py_dict(py, &EmaAnchorParams::default()).unwrap();
            let dict = object.bind(py);
            let optional = [
                "entry_double_down_factor",
                "offset_volatility_ema_span_1m",
                "offset_volatility_1m_weight",
                "offset_volatility_ema_span_1h",
                "offset_volatility_1h_weight",
            ];
            for key in optional {
                dict.del_item(key).unwrap();
            }
            let parsed = ema_anchor_strategy_params_from_dict(dict.as_gil_ref()).unwrap();
            for key in optional {
                assert_eq!(parsed[key], 0.0, "{key}");
            }
        });
    }

    #[test]
    fn strategy_bindings_reject_nonfinite_values_before_json_null_reaches_backtest() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            for kind in ["trailing_martingale", "ema_anchor"] {
                let value = if kind == "trailing_martingale" {
                    serde_json::to_value(crate::strategies::TrailingMartingaleParams::default())
                        .unwrap()
                } else {
                    serde_json::to_value(EmaAnchorParams::default()).unwrap()
                };
                let fields = if kind == "trailing_martingale" {
                    vec![
                        (None, "volatility_ema_span_1m"),
                        (Some("entry"), "double_down_factor"),
                        (Some("close"), "qty_pct"),
                    ]
                } else {
                    vec![(None, "offset")]
                };
                for pside in ["long", "short"] {
                    for (section, field) in &fields {
                        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                            let raw = serde_json::json!({"long":value,"short":value});
                            let object = json_value_to_py(py, &raw).unwrap();
                            let dict = object.bind(py).downcast::<PyDict>().unwrap();
                            let mut target = dict
                                .get_item(pside)
                                .unwrap()
                                .unwrap()
                                .downcast_into::<PyDict>()
                                .unwrap();
                            if let Some(section) = section {
                                target = target
                                    .get_item(section)
                                    .unwrap()
                                    .unwrap()
                                    .downcast_into::<PyDict>()
                                    .unwrap();
                            }
                            target.set_item(field, nonfinite).unwrap();
                            let error = strategy_params_pair_from_dict(dict.as_gil_ref(), kind)
                                .expect_err("nonfinite input must fail at the Python binding");
                            assert!(error.to_string().contains(field), "{error}");
                            assert!(error.to_string().contains("must be finite"), "{error}");
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn native_backtest_rejects_retired_rolling_gate_horizon() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            for horizon in [-1.0, 0.0, 43_200.0] {
                let input = PyDict::new_bound(py);
                input
                    .set_item("realized_loss_gate_lookback_minutes", horizon)
                    .unwrap();
                let error = backtest_params_from_dict(input.as_gil_ref()).unwrap_err();
                assert!(error
                    .to_string()
                    .contains("realized_loss_gate_lookback_minutes is retired"));
            }
        });
    }

    #[test]
    fn martingale_close_diagnostic_keeps_its_shared_bot_input_reader() {
        let exchange = ExchangeParams::default();
        let state = StateParams {
            balance: 1000.0,
            order_book: OrderBook {
                bid: 110.0,
                ask: 110.1,
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            total_wallet_exposure_limit: 1.0,
            n_positions: 1,
            ..Default::default()
        };
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.25,
            threshold_base_pct: 0.01,
            ..Default::default()
        };
        let trailing = TrailingPriceBundle::default();
        let runtime = RuntimeOrderContext {
            effective_wallet_exposure_limit: 1.0,
        };
        for (side, name, size) in [
            (StrategySide::Long, "long", 5.0),
            (StrategySide::Short, "short", -5.0),
        ] {
            let position = Position { size, price: 100.0 };
            let input = serde_json::json!({
                "pside": name, "exchange": exchange, "state": state,
                "position": position, "trailing": trailing, "close_params": close,
                "bot_params": {"wallet_exposure_limit":1.0,"total_wallet_exposure_limit":1.0,"n_positions":1},
                "runtime": {"effective_wallet_exposure_limit":1.0},
            });
            let actual: Value = serde_json::from_str(
                &calc_trailing_martingale_close_diagnostic_py(&input.to_string()).unwrap(),
            )
            .unwrap();
            let expected = calc_trailing_martingale_close_diagnostic(
                side, &exchange, &state, &bot, &runtime, &close, &position, &trailing,
            );
            assert_eq!(actual, serde_json::to_value(expected).unwrap());
        }
    }

    #[test]
    fn test_gate_entries_blocks_when_twe_if_filled_exceeds() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let side = "long";
            let balance = 1000.0;
            let twel = 1.0;
            // One position small
            let positions = PyList::empty_bound(py);
            let p0 = PyDict::new_bound(py);
            p0.set_item("idx", 0usize).unwrap();
            p0.set_item("position_size", 0.0f64).unwrap();
            p0.set_item("position_price", 0.0f64).unwrap();
            p0.set_item("c_mult", 1.0f64).unwrap();
            positions.append(p0).unwrap();
            // Two entries that together would push twe_if_filled >= 1.0
            let entries = PyList::empty_bound(py);
            let e1 = PyDict::new_bound(py);
            e1.set_item("idx", 0usize).unwrap();
            e1.set_item("qty", 5.0f64).unwrap(); // cost 5*100/1000 = 0.5
            e1.set_item("price", 100.0f64).unwrap();
            e1.set_item("qty_step", 0.01f64).unwrap();
            e1.set_item("min_qty", 0.0f64).unwrap();
            e1.set_item("min_cost", 0.0f64).unwrap();
            e1.set_item("c_mult", 1.0f64).unwrap();
            e1.set_item("market_price", 100.0f64).unwrap();
            e1.set_item("order_type_id", OrderType::EntryGridNormalLong.id())
                .unwrap();
            entries.append(e1).unwrap();
            let e2 = PyDict::new_bound(py);
            e2.set_item("idx", 0usize).unwrap();
            e2.set_item("qty", 6.0f64).unwrap(); // +0.6 exposure -> total 1.1 > 1.0
            e2.set_item("price", 100.0f64).unwrap();
            e2.set_item("qty_step", 0.01f64).unwrap();
            e2.set_item("min_qty", 0.0f64).unwrap();
            e2.set_item("min_cost", 0.0f64).unwrap();
            e2.set_item("c_mult", 1.0f64).unwrap();
            e2.set_item("market_price", 90.0f64).unwrap();
            e2.set_item("order_type_id", OrderType::EntryGridNormalLong.id())
                .unwrap();
            entries.append(e2).unwrap();

            let res = gate_entries_by_twel_py(side, balance, twel, &positions, &entries).unwrap();
            // Should not allow both; either prune one or adjust last
            assert!(!res.is_empty());
            // Simulate twe_if_filled with returned
            let mut psize = 0.0f64;
            let mut pprice = 0.0f64;
            for (idx, qty, price, _ot) in res.into_iter() {
                let (_i, _qty, _price) = (idx, qty, price);
                let (nps, npp) =
                    crate::utils::calc_new_psize_pprice(psize, pprice, qty, price, 0.01);
                psize = nps;
                pprice = npp;
            }
            let twe = crate::utils::calc_wallet_exposure(
                1.0,
                balance,
                psize.abs(),
                if pprice > 0.0 { pprice } else { 100.0 },
            );
            assert!(
                twe < twel - 1e-12,
                "gated twe {} not strictly below {}",
                twe,
                twel
            );
        });
    }
}

#[pyfunction]
pub fn run_backtest(
    hlcvs: PyReadonlyArray3<f64>,
    btc_usd: PyReadonlyArray1<f64>,
    bot_params: &Bound<'_, PyAny>,
    strategy_params: &Bound<'_, PyAny>,
    exchange_params_list: &Bound<'_, PyAny>,
    backtest_params_dict: &Bound<'_, PyDict>,
) -> PyResult<BacktestPyResult> {
    run_backtest_core::<false>(
        hlcvs,
        btc_usd,
        bot_params,
        strategy_params,
        exchange_params_list,
        backtest_params_dict,
        None,
        None,
    )
}

#[pyfunction]
pub fn run_backtest_bundle(
    bundle: &HlcvsBundlePy,
    bot_params: &Bound<'_, PyAny>,
    strategy_params: &Bound<'_, PyAny>,
    exchange_params_list: &Bound<'_, PyAny>,
    backtest_params_dict: &Bound<'_, PyDict>,
) -> PyResult<BacktestPyResult> {
    let py = bot_params.py();
    let hlcvs = bundle.inner.hlcvs.bind(py).readonly();
    let btc = bundle.inner.btc_usd.bind(py).readonly();
    if let (Some(values), Some(rows)) = (&bundle.inner.eligibility, &bundle.inner.eligibility_rows)
    {
        let values = values.bind(py).readonly();
        let rows = rows.bind(py).readonly();
        run_backtest_core::<true>(
            hlcvs,
            btc,
            bot_params,
            strategy_params,
            exchange_params_list,
            backtest_params_dict,
            Some(values),
            Some(rows),
        )
    } else {
        run_backtest_core::<false>(
            hlcvs,
            btc,
            bot_params,
            strategy_params,
            exchange_params_list,
            backtest_params_dict,
            None,
            None,
        )
    }
}

fn run_backtest_core<'py, const ORGANILLO: bool>(
    hlcvs: PyReadonlyArray3<'py, f64>,
    btc_usd: PyReadonlyArray1<'py, f64>,
    bot_params: &Bound<'py, PyAny>,
    strategy_params: &Bound<'py, PyAny>,
    exchange_params_list: &Bound<'py, PyAny>,
    backtest_params_dict: &Bound<'py, PyDict>,
    eligibility: Option<PyReadonlyArray2<'py, u8>>,
    eligibility_rows: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<BacktestPyResult> {
    let profile_enabled = rust_profile_enabled();
    let profile_total_start = profile_start(profile_enabled);
    let mut rust_profile = Vec::new();
    let input_parse_start = profile_start(profile_enabled);
    let hlcvs_rust = hlcvs.as_array();
    let btc_usd_rust = btc_usd.as_array();

    if hlcvs_rust.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "Expected 3D HLCV array, got ndim={}",
            hlcvs_rust.ndim()
        )));
    }

    let n_timesteps = hlcvs_rust.shape()[0];
    if btc_usd_rust.len() != n_timesteps {
        return Err(PyValueError::new_err(format!(
            "BTC/USD data length ({}) does not match HLCV timesteps ({})",
            btc_usd_rust.len(),
            n_timesteps
        )));
    }

    let strategy_kind_name = backtest_params_dict
        .get_item("strategy_kind")?
        .map(|item| item.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "trailing_martingale".to_string());
    let strategy_kind = strategy_kind_from_name(&strategy_kind_name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown strategy_kind for backtest params: {}",
            strategy_kind_name
        ))
    })?;
    let backtest_params = backtest_params_from_dict(backtest_params_dict.as_gil_ref())?;

    let bot_params_py_list_bound = bot_params
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("bot_params must be a list[dict] (one per coin)"))?;
    let bot_params_py_list = bot_params_py_list_bound.as_gil_ref();

    let bot_params_len = bot_params_py_list.len();
    let mut bot_params_vec = Vec::with_capacity(bot_params_len);
    for item in bot_params_py_list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each bot_params element must be a dict"))?;
        bot_params_vec.push(bot_params_pair_from_dict(
            dict,
            backtest_params.equity_hard_stop_loss.hsl.is_some(),
        )?);
    }
    if backtest_params.candle_interval_minutes != 1
        && crate::unilateralness::backtest_enabled_sides(
            &bot_params_vec,
            backtest_params.dynamic_wel_by_tradability,
        )
        .iter()
        .flatten()
        .any(|enabled| *enabled)
    {
        return Err(PyValueError::new_err(
            "RMS unilateralness requires completed one-minute candles",
        ));
    }

    let strategy_params_py_list_bound = strategy_params.downcast::<PyList>().map_err(|_| {
        PyValueError::new_err("strategy_params must be a list[dict] (one per coin)")
    })?;
    let strategy_params_py_list = strategy_params_py_list_bound.as_gil_ref();
    if strategy_params_py_list.len() != bot_params_len {
        return Err(PyValueError::new_err(format!(
            "strategy_params length ({}) does not match bot_params length ({})",
            strategy_params_py_list.len(),
            bot_params_len
        )));
    }
    let mut strategy_params_vec = Vec::with_capacity(bot_params_len);
    for item in strategy_params_py_list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each strategy_params element must be a dict"))?;
        strategy_params_vec.push(strategy_params_pair_from_dict(dict, &strategy_kind_name)?);
    }

    let exchange_params_list_bound = exchange_params_list
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("Unsupported data type for exchange_params_list"))?;
    let exchange_params_list = exchange_params_list_bound.as_gil_ref();
    let list_len = exchange_params_list.len();
    let mut exchange_params = Vec::with_capacity(list_len);
    for item in exchange_params_list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("Unsupported data type in exchange_params_list"))?;
        exchange_params.push(exchange_params_from_dict(dict)?);
    }
    profile_add(&mut rust_profile, "rust_input_parse_ms", input_parse_start);

    let metrics_only = backtest_params.metrics_only;
    let skip_btc_analysis = metrics_only
        && backtest_params.skip_btc_analysis
        && backtest_params.btc_collateral_cap <= 0.0;
    let init_start = profile_start(profile_enabled);
    let mut backtest = BacktestEngine::<ORGANILLO>::new_with_strategy_params(
        hlcvs_rust,
        btc_usd_rust,
        strategy_kind,
        bot_params_vec,
        strategy_params_vec,
        exchange_params,
        &backtest_params,
    );
    backtest.eligibility = eligibility.as_ref().map(|a| a.as_array());
    backtest.eligibility_rows = eligibility_rows.as_ref().map(|a| a.as_array());
    profile_add(&mut rust_profile, "rust_backtest_init_ms", init_start);

    // Run the backtest and process results
    Python::with_gil(move |py| {
        let simulation_start = profile_start(profile_enabled);
        let (fills, equities) = backtest.run().map_err(PyValueError::new_err)?;
        profile_add(&mut rust_profile, "rust_simulation_ms", simulation_start);
        let (entry_pct_long, entry_pct_short) = backtest.initial_entry_balance_pct();
        let (mut analysis_usd, mut analysis_btc) = if skip_btc_analysis {
            let (analysis_usd, analysis_profile) = analyze_backtest_usd_with_profile(
                &fills,
                &equities,
                &backtest.total_wallet_exposures,
                backtest.liquidated(),
                profile_enabled,
            );
            if profile_enabled {
                rust_profile.push(("rust_analysis_usd_ms", analysis_profile.analysis_usd_ms));
                rust_profile.push(("rust_analysis_btc_ms", 0.0));
                rust_profile.push(("rust_btc_fill_prepare_ms", 0.0));
                rust_profile.push(("rust_analysis_pair_ms", analysis_profile.total_ms));
                rust_profile.push(("rust_btc_analysis_skipped", 1.0));
            }
            (analysis_usd, Analysis::default())
        } else if profile_enabled {
            let ((analysis_usd, analysis_btc), analysis_profile) =
                analyze_backtest_pair_with_options(
                    &fills,
                    &equities,
                    backtest.balance.use_btc_collateral,
                    &backtest.total_wallet_exposures,
                    backtest.liquidated(),
                    true,
                );
            rust_profile.push(("rust_analysis_usd_ms", analysis_profile.analysis_usd_ms));
            rust_profile.push((
                "rust_btc_fill_prepare_ms",
                analysis_profile.btc_fill_prepare_ms,
            ));
            rust_profile.push(("rust_analysis_btc_ms", analysis_profile.analysis_btc_ms));
            rust_profile.push(("rust_analysis_pair_ms", analysis_profile.total_ms));
            (analysis_usd, analysis_btc)
        } else {
            analyze_backtest_pair_with_options(
                &fills,
                &equities,
                backtest.balance.use_btc_collateral,
                &backtest.total_wallet_exposures,
                backtest.liquidated(),
                false,
            )
            .0
        };
        backtest.exposure_rms.apply(&mut analysis_usd);
        backtest.exposure_rms.apply(&mut analysis_btc);
        analysis_usd.entry_initial_balance_pct_long = entry_pct_long;
        analysis_usd.entry_initial_balance_pct_short = entry_pct_short;
        analysis_btc.entry_initial_balance_pct_long = entry_pct_long;
        analysis_btc.entry_initial_balance_pct_short = entry_pct_short;
        let completion_start = profile_start(profile_enabled);
        let completion_ratio =
            calc_backtest_completion_ratio(&equities, &backtest_params, n_timesteps);
        analysis_usd.backtest_completion_ratio = completion_ratio;
        analysis_btc.backtest_completion_ratio = completion_ratio;
        profile_add(
            &mut rust_profile,
            "rust_completion_ratio_ms",
            completion_start,
        );
        let fill_activity_start = profile_start(profile_enabled);
        let fill_activity = backtest.fill_activity_metrics_for_analysis(&fills, &equities);
        apply_fill_activity_metrics(&mut analysis_usd, fill_activity);
        apply_fill_activity_metrics(&mut analysis_btc, fill_activity);
        profile_add(
            &mut rust_profile,
            "rust_fill_activity_ms",
            fill_activity_start,
        );

        let hard_stop_metrics_start = profile_start(profile_enabled);
        let hs = backtest.hard_stop_metrics();
        profile_add(
            &mut rust_profile,
            "rust_hard_stop_metrics_ms",
            hard_stop_metrics_start,
        );
        let strategy_metrics_start = profile_start(profile_enabled);
        let strategy = backtest.strategy_equity_metrics_for_analysis();
        profile_add(
            &mut rust_profile,
            "rust_strategy_equity_metrics_ms",
            strategy_metrics_start,
        );
        analysis_usd.hard_stop_triggers = hs.triggers;
        analysis_usd.hard_stop_triggers_per_year = hs.triggers_per_year;
        analysis_usd.hard_stop_triggers_long = hs.triggers_long;
        analysis_usd.hard_stop_triggers_short = hs.triggers_short;
        analysis_usd.hard_stop_halt_to_restart_equity_loss_pct = hs.halt_to_restart_equity_loss_pct;
        analysis_usd.hard_stop_restarts = hs.restarts;
        analysis_usd.hard_stop_restarts_per_year = hs.restarts_per_year;
        analysis_usd.hard_stop_restarts_per_year_long = hs.restarts_per_year_long;
        analysis_usd.hard_stop_restarts_per_year_short = hs.restarts_per_year_short;
        analysis_usd.hard_stop_restarts_long = hs.restarts_long;
        analysis_usd.hard_stop_restarts_short = hs.restarts_short;
        analysis_usd.hard_stop_time_in_red_pct = hs.time_in_red_pct;
        analysis_usd.hard_stop_duration_minutes_mean = hs.duration_minutes_mean;
        analysis_usd.hard_stop_duration_minutes_max = hs.duration_minutes_max;
        analysis_usd.hard_stop_trigger_drawdown_mean = hs.trigger_drawdown_mean;
        analysis_usd.hard_stop_panic_close_loss_sum = hs.panic_close_loss_sum;
        analysis_usd.hard_stop_panic_close_loss_max = hs.panic_close_loss_max;
        analysis_usd.hard_stop_panic_close_loss_drawdown_pct_min =
            hs.panic_close_loss_drawdown_pct_min;
        analysis_usd.hard_stop_panic_close_loss_drawdown_pct_mean =
            hs.panic_close_loss_drawdown_pct_mean;
        analysis_usd.hard_stop_panic_close_loss_drawdown_pct_max =
            hs.panic_close_loss_drawdown_pct_max;
        analysis_usd.hard_stop_flatten_time_minutes_mean = hs.flatten_time_minutes_mean;
        analysis_usd.hard_stop_post_restart_retrigger_pct = hs.post_restart_retrigger_pct;
        analysis_usd.drawdown_worst_strategy_eq = strategy.overall.drawdown_worst_strategy_eq;
        analysis_usd.drawdown_worst_strategy_eq_long = strategy.long.drawdown_worst_strategy_eq;
        analysis_usd.drawdown_worst_strategy_eq_short = strategy.short.drawdown_worst_strategy_eq;
        analysis_usd.drawdown_worst_ema_strategy_eq = strategy
            .long
            .drawdown_worst_ema_strategy_eq
            .max(strategy.short.drawdown_worst_ema_strategy_eq);
        analysis_usd.drawdown_worst_ema_strategy_eq_long =
            strategy.long.drawdown_worst_ema_strategy_eq;
        analysis_usd.drawdown_worst_ema_strategy_eq_short =
            strategy.short.drawdown_worst_ema_strategy_eq;
        analysis_usd.drawdown_worst_mean_1pct_strategy_eq =
            strategy.overall.drawdown_worst_mean_1pct_strategy_eq;
        analysis_usd.drawdown_worst_mean_1pct_strategy_eq_long =
            strategy.long.drawdown_worst_mean_1pct_strategy_eq;
        analysis_usd.drawdown_worst_mean_1pct_strategy_eq_short =
            strategy.short.drawdown_worst_mean_1pct_strategy_eq;
        analysis_usd.drawdown_worst_mean_1pct_ema_strategy_eq = strategy
            .long
            .drawdown_worst_mean_1pct_ema_strategy_eq
            .max(strategy.short.drawdown_worst_mean_1pct_ema_strategy_eq);
        analysis_usd.drawdown_worst_mean_1pct_ema_strategy_eq_long =
            strategy.long.drawdown_worst_mean_1pct_ema_strategy_eq;
        analysis_usd.drawdown_worst_mean_1pct_ema_strategy_eq_short =
            strategy.short.drawdown_worst_mean_1pct_ema_strategy_eq;
        analysis_usd.strategy_eq_underwater_pct_mean =
            strategy.overall.strategy_eq_underwater_pct_mean;
        analysis_usd.strategy_eq_underwater_pct_median =
            strategy.overall.strategy_eq_underwater_pct_median;
        analysis_usd.strategy_eq_recovery_days_mean =
            strategy.overall.strategy_eq_recovery_days_mean;
        analysis_usd.strategy_eq_recovery_days_median =
            strategy.overall.strategy_eq_recovery_days_median;
        analysis_usd.strategy_eq_recovery_days_p95 = strategy.overall.strategy_eq_recovery_days_p95;
        analysis_usd.strategy_eq_recovery_days_p99 = strategy.overall.strategy_eq_recovery_days_p99;
        analysis_usd.strategy_eq_recovery_days_mean_worst_5pct =
            strategy.overall.strategy_eq_recovery_days_mean_worst_5pct;
        analysis_usd.strategy_eq_recovery_days_mean_worst_1pct =
            strategy.overall.strategy_eq_recovery_days_mean_worst_1pct;
        analysis_usd.strategy_eq_recovery_days_max = strategy.overall.strategy_eq_recovery_days_max;
        analysis_usd.peak_recovery_hours_strategy_eq =
            strategy.overall.peak_recovery_hours_strategy_eq;
        analysis_usd.peak_recovery_hours_strategy_eq_long =
            strategy.long.peak_recovery_hours_strategy_eq;
        analysis_usd.peak_recovery_hours_strategy_eq_short =
            strategy.short.peak_recovery_hours_strategy_eq;
        analysis_usd.peak_recovery_days_strategy_eq =
            strategy.overall.peak_recovery_days_strategy_eq;
        analysis_usd.peak_recovery_days_strategy_eq_long =
            strategy.long.peak_recovery_days_strategy_eq;
        analysis_usd.peak_recovery_days_strategy_eq_short =
            strategy.short.peak_recovery_days_strategy_eq;
        analysis_usd.gain_strategy_eq = strategy.overall.gain_strategy_eq;
        analysis_usd.adg_strategy_eq = strategy.overall.adg_strategy_eq;
        analysis_usd.adg_rolling_hmean_strategy_eq = strategy.overall.adg_rolling_hmean_strategy_eq;
        analysis_usd.adg_time_integrated_strategy_eq =
            strategy.overall.adg_time_integrated_strategy_eq;
        analysis_usd.positive_gain_participation_strategy_eq =
            strategy.overall.positive_gain_participation_strategy_eq;
        analysis_usd.mdg_strategy_eq = strategy.overall.mdg_strategy_eq;
        analysis_usd.sharpe_ratio_strategy_eq = strategy.overall.sharpe_ratio_strategy_eq;
        analysis_usd.sortino_ratio_strategy_eq = strategy.overall.sortino_ratio_strategy_eq;
        analysis_usd.omega_ratio_strategy_eq = strategy.overall.omega_ratio_strategy_eq;
        analysis_usd.expected_shortfall_1pct_strategy_eq =
            strategy.overall.expected_shortfall_1pct_strategy_eq;
        analysis_usd.calmar_ratio_strategy_eq = strategy.overall.calmar_ratio_strategy_eq;
        analysis_usd.sterling_ratio_strategy_eq = strategy.overall.sterling_ratio_strategy_eq;
        analysis_btc.hard_stop_triggers = hs.triggers;
        analysis_btc.hard_stop_triggers_per_year = hs.triggers_per_year;
        analysis_btc.hard_stop_triggers_long = hs.triggers_long;
        analysis_btc.hard_stop_triggers_short = hs.triggers_short;
        analysis_btc.hard_stop_halt_to_restart_equity_loss_pct = hs.halt_to_restart_equity_loss_pct;
        analysis_btc.hard_stop_restarts = hs.restarts;
        analysis_btc.hard_stop_restarts_per_year = hs.restarts_per_year;
        analysis_btc.hard_stop_restarts_per_year_long = hs.restarts_per_year_long;
        analysis_btc.hard_stop_restarts_per_year_short = hs.restarts_per_year_short;
        analysis_btc.hard_stop_restarts_long = hs.restarts_long;
        analysis_btc.hard_stop_restarts_short = hs.restarts_short;
        analysis_btc.hard_stop_time_in_red_pct = hs.time_in_red_pct;
        analysis_btc.hard_stop_duration_minutes_mean = hs.duration_minutes_mean;
        analysis_btc.hard_stop_duration_minutes_max = hs.duration_minutes_max;
        analysis_btc.hard_stop_trigger_drawdown_mean = hs.trigger_drawdown_mean;
        analysis_btc.hard_stop_panic_close_loss_sum = hs.panic_close_loss_sum;
        analysis_btc.hard_stop_panic_close_loss_max = hs.panic_close_loss_max;
        analysis_btc.hard_stop_panic_close_loss_drawdown_pct_min =
            hs.panic_close_loss_drawdown_pct_min;
        analysis_btc.hard_stop_panic_close_loss_drawdown_pct_mean =
            hs.panic_close_loss_drawdown_pct_mean;
        analysis_btc.hard_stop_panic_close_loss_drawdown_pct_max =
            hs.panic_close_loss_drawdown_pct_max;
        analysis_btc.hard_stop_flatten_time_minutes_mean = hs.flatten_time_minutes_mean;
        analysis_btc.hard_stop_post_restart_retrigger_pct = hs.post_restart_retrigger_pct;
        analysis_btc.drawdown_worst_strategy_eq = strategy.overall.drawdown_worst_strategy_eq;
        analysis_btc.drawdown_worst_strategy_eq_long = strategy.long.drawdown_worst_strategy_eq;
        analysis_btc.drawdown_worst_strategy_eq_short = strategy.short.drawdown_worst_strategy_eq;
        analysis_btc.drawdown_worst_ema_strategy_eq = strategy
            .long
            .drawdown_worst_ema_strategy_eq
            .max(strategy.short.drawdown_worst_ema_strategy_eq);
        analysis_btc.drawdown_worst_ema_strategy_eq_long =
            strategy.long.drawdown_worst_ema_strategy_eq;
        analysis_btc.drawdown_worst_ema_strategy_eq_short =
            strategy.short.drawdown_worst_ema_strategy_eq;
        analysis_btc.drawdown_worst_mean_1pct_strategy_eq =
            strategy.overall.drawdown_worst_mean_1pct_strategy_eq;
        analysis_btc.drawdown_worst_mean_1pct_strategy_eq_long =
            strategy.long.drawdown_worst_mean_1pct_strategy_eq;
        analysis_btc.drawdown_worst_mean_1pct_strategy_eq_short =
            strategy.short.drawdown_worst_mean_1pct_strategy_eq;
        analysis_btc.drawdown_worst_mean_1pct_ema_strategy_eq = strategy
            .long
            .drawdown_worst_mean_1pct_ema_strategy_eq
            .max(strategy.short.drawdown_worst_mean_1pct_ema_strategy_eq);
        analysis_btc.drawdown_worst_mean_1pct_ema_strategy_eq_long =
            strategy.long.drawdown_worst_mean_1pct_ema_strategy_eq;
        analysis_btc.drawdown_worst_mean_1pct_ema_strategy_eq_short =
            strategy.short.drawdown_worst_mean_1pct_ema_strategy_eq;
        analysis_btc.strategy_eq_underwater_pct_mean =
            strategy.overall.strategy_eq_underwater_pct_mean;
        analysis_btc.strategy_eq_underwater_pct_median =
            strategy.overall.strategy_eq_underwater_pct_median;
        analysis_btc.strategy_eq_recovery_days_mean =
            strategy.overall.strategy_eq_recovery_days_mean;
        analysis_btc.strategy_eq_recovery_days_median =
            strategy.overall.strategy_eq_recovery_days_median;
        analysis_btc.strategy_eq_recovery_days_p95 = strategy.overall.strategy_eq_recovery_days_p95;
        analysis_btc.strategy_eq_recovery_days_p99 = strategy.overall.strategy_eq_recovery_days_p99;
        analysis_btc.strategy_eq_recovery_days_mean_worst_5pct =
            strategy.overall.strategy_eq_recovery_days_mean_worst_5pct;
        analysis_btc.strategy_eq_recovery_days_mean_worst_1pct =
            strategy.overall.strategy_eq_recovery_days_mean_worst_1pct;
        analysis_btc.strategy_eq_recovery_days_max = strategy.overall.strategy_eq_recovery_days_max;
        analysis_btc.peak_recovery_hours_strategy_eq =
            strategy.overall.peak_recovery_hours_strategy_eq;
        analysis_btc.peak_recovery_hours_strategy_eq_long =
            strategy.long.peak_recovery_hours_strategy_eq;
        analysis_btc.peak_recovery_hours_strategy_eq_short =
            strategy.short.peak_recovery_hours_strategy_eq;
        analysis_btc.peak_recovery_days_strategy_eq =
            strategy.overall.peak_recovery_days_strategy_eq;
        analysis_btc.peak_recovery_days_strategy_eq_long =
            strategy.long.peak_recovery_days_strategy_eq;
        analysis_btc.peak_recovery_days_strategy_eq_short =
            strategy.short.peak_recovery_days_strategy_eq;
        analysis_btc.gain_strategy_eq = strategy.overall.gain_strategy_eq;
        analysis_btc.adg_strategy_eq = strategy.overall.adg_strategy_eq;
        analysis_btc.adg_rolling_hmean_strategy_eq = strategy.overall.adg_rolling_hmean_strategy_eq;
        analysis_btc.adg_time_integrated_strategy_eq =
            strategy.overall.adg_time_integrated_strategy_eq;
        analysis_btc.positive_gain_participation_strategy_eq =
            strategy.overall.positive_gain_participation_strategy_eq;
        analysis_btc.mdg_strategy_eq = strategy.overall.mdg_strategy_eq;
        analysis_btc.sharpe_ratio_strategy_eq = strategy.overall.sharpe_ratio_strategy_eq;
        analysis_btc.sortino_ratio_strategy_eq = strategy.overall.sortino_ratio_strategy_eq;
        analysis_btc.omega_ratio_strategy_eq = strategy.overall.omega_ratio_strategy_eq;
        analysis_btc.expected_shortfall_1pct_strategy_eq =
            strategy.overall.expected_shortfall_1pct_strategy_eq;
        analysis_btc.calmar_ratio_strategy_eq = strategy.overall.calmar_ratio_strategy_eq;
        analysis_btc.sterling_ratio_strategy_eq = strategy.overall.sterling_ratio_strategy_eq;

        if backtest_params.equity_hard_stop_loss.hsl.is_some() {
            for analysis in [&mut analysis_usd, &mut analysis_btc] {
                analysis.drawdown_worst_ema_strategy_eq =
                    strategy.overall.drawdown_worst_ema_strategy_eq;
                analysis.drawdown_worst_mean_1pct_ema_strategy_eq =
                    strategy.overall.drawdown_worst_mean_1pct_ema_strategy_eq;
            }
        }
        let analysis_dict_start = profile_start(profile_enabled);
        let py_analysis_usd = struct_to_py_dict(py, &analysis_usd)?;
        let py_analysis_btc = if skip_btc_analysis {
            PyDict::new_bound(py).unbind()
        } else {
            struct_to_py_dict(py, &analysis_btc)?
        };
        if let Some(half_life_days) = backtest_params.ew_half_life_days {
            let calc = |series: &[f64]| -> PyResult<crate::ew::EwAnalysis> {
                let mut result = crate::ew::analyze_ew_equity(
                    series,
                    &equities.timestamps_ms,
                    half_life_days,
                    backtest_params.candle_interval_minutes,
                )
                .map_err(PyValueError::new_err)?;
                if backtest.liquidated() {
                    result.metrics = None;
                    result.metadata.status = "liquidated";
                }
                Ok(result)
            };
            let strategy_ew = calc(backtest.strategy_equity_series_for_artifacts())?;
            let usd_ew = calc(&equities.usd_total_equity)?;
            attach_ew_analysis(py, &py_analysis_usd, "usd", &usd_ew, &strategy_ew)?;
            if !skip_btc_analysis {
                let btc_ew = calc(&equities.btc_total_equity)?;
                attach_ew_analysis(py, &py_analysis_btc, "btc", &btc_ew, &strategy_ew)?;
            }
        }
        let hsl_report = hsl_report_to_py(py, &backtest)?;
        profile_add(
            &mut rust_profile,
            "rust_analysis_py_dict_ms",
            analysis_dict_start,
        );
        if metrics_only {
            let py_extra = if profile_enabled || hsl_report.is_some() {
                let py_extra = PyDict::new_bound(py);
                if profile_enabled {
                    py_extra.set_item(
                        RUST_PROFILE_KEY,
                        profile_to_py_dict(py, &rust_profile, profile_total_start)?,
                    )?;
                }
                if let Some(report) = &hsl_report {
                    py_extra.set_item("hsl", report)?;
                }
                py_extra.into_py(py)
            } else {
                py.None().into_py(py)
            };
            let equities_timestamps_array =
                Array2::from_shape_fn((equities.timestamps_ms.len(), 1), |(i, _)| {
                    equities.timestamps_ms[i] as f64
                })
                .into_pyarray_bound(py)
                .unbind();
            return Ok((
                py.None().into_py(py),
                equities_timestamps_array.into_py(py),
                py_analysis_usd,
                py_analysis_btc,
                py_extra,
            ));
        }
        let artifact_conversion_start = profile_start(profile_enabled);
        let py_hard_stop_plot = PyDict::new_bound(py);
        if let Some(report) = hsl_report {
            py_hard_stop_plot.set_item("hsl", report)?;
        }
        let mut py_fills = Array2::from_elem((fills.len(), 19), py.None());
        for (i, fill) in fills.iter().enumerate() {
            py_fills[(i, 0)] = fill.index.into_py(py);
            py_fills[(i, 1)] = (fill.timestamp_ms as i64).into_py(py);
            py_fills[(i, 2)] = <String as Clone>::clone(&fill.coin).into_py(py);
            py_fills[(i, 3)] = fill.pnl.into_py(py);
            py_fills[(i, 4)] = fill.fee_paid.into_py(py);
            py_fills[(i, 5)] = fill.usd_total_balance.into_py(py);
            py_fills[(i, 6)] = fill.btc_cash_wallet.into_py(py);
            py_fills[(i, 7)] = fill.usd_cash_wallet.into_py(py);
            py_fills[(i, 8)] = fill.btc_price.into_py(py);
            py_fills[(i, 9)] = fill.fill_qty.into_py(py);
            py_fills[(i, 10)] = fill.fill_price.into_py(py);
            py_fills[(i, 11)] = fill.position_size.into_py(py);
            py_fills[(i, 12)] = fill.position_price.into_py(py);
            py_fills[(i, 13)] = fill.order_type.to_string().into_py(py);
            py_fills[(i, 14)] = fill.liquidity.clone().into_py(py);
            py_fills[(i, 15)] = fill.wallet_exposure.into_py(py);
            py_fills[(i, 16)] = fill.twe_long.into_py(py);
            py_fills[(i, 17)] = fill.twe_short.into_py(py);
            py_fills[(i, 18)] = fill.twe_net.into_py(py);
        }

        let strategy_equity_series = backtest.strategy_equity_series_for_artifacts();
        if strategy_equity_series.len() != equities.timestamps_ms.len() {
            return Err(PyValueError::new_err(
                "hsl strategy-equity observations do not match equity timestamps",
            ));
        }
        let equities_array =
            Array2::from_shape_fn((equities.timestamps_ms.len(), 4), |(i, j)| match j {
                0 => equities.timestamps_ms[i] as f64,
                1 => equities.usd_total_equity[i],
                2 => equities.btc_total_equity[i],
                3 => strategy_equity_series[i],
                _ => 0.0,
            })
            .into_pyarray_bound(py)
            .unbind();
        let fills_array = py_fills.into_pyarray_bound(py).unbind();
        profile_add(
            &mut rust_profile,
            "rust_artifact_conversion_ms",
            artifact_conversion_start,
        );
        if profile_enabled {
            py_hard_stop_plot.set_item(
                RUST_PROFILE_KEY,
                profile_to_py_dict(py, &rust_profile, profile_total_start)?,
            )?;
        }
        Ok((
            fills_array.into_py(py),
            equities_array.into_py(py),
            py_analysis_usd,
            py_analysis_btc,
            py_hard_stop_plot.into_py(py),
        ))
    })
}

/// Preserve the public report schema while avoiding a huge intermediate JSON tree.
/// Static field names and enum strings are shared; each row/list remains independent.
pub(crate) fn hsl_report_to_py<const ORGANILLO: bool>(
    py: Python<'_>,
    backtest: &BacktestEngine<'_, ORGANILLO>,
) -> PyResult<Option<PyObject>> {
    let Some(metadata) = backtest
        .hsl_report_metadata()
        .map_err(PyValueError::new_err)?
    else {
        return Ok(None);
    };
    let report = json_value_to_py(py, &metadata)?;
    let report_dict = report.bind(py).downcast::<PyDict>()?;
    macro_rules! fields {
        ($dict:ident, $row:ident, $($field:ident),+ $(,)?) => {
            $($dict.set_item(pyo3::intern!(py, stringify!($field)), &$row.$field)?;)+
        };
    }
    let samples = PyList::empty_bound(py);
    for row in backtest.hsl_samples() {
        let dict = PyDict::new_bound(py);
        fields!(dict, row, sequence, timestamp, side, coin, raw, ema, red_at, flat_at, reasons);
        dict.set_item(
            pyo3::intern!(py, "phase"),
            pyo3::types::PyString::intern_bound(py, row.phase),
        )?;
        let action = row.action.map(|action| match action {
            crate::hsl_intent::Action::Normal => pyo3::intern!(py, "normal"),
            crate::hsl_intent::Action::Panic => pyo3::intern!(py, "panic"),
            crate::hsl_intent::Action::Halted => pyo3::intern!(py, "halted"),
        });
        dict.set_item(pyo3::intern!(py, "action"), action)?;
        samples.append(dict)?;
    }
    let events = PyList::empty_bound(py);
    for row in backtest.hsl_events() {
        let dict = PyDict::new_bound(py);
        fields!(
            dict,
            row,
            sequence,
            observed_at,
            side,
            coin,
            kind,
            reconstructed_at,
            reason
        );
        events.append(dict)?;
    }
    report_dict.set_item(pyo3::intern!(py, "samples"), samples)?;
    report_dict.set_item(pyo3::intern!(py, "events"), events)?;
    Ok(Some(report))
}

fn struct_to_py_dict<T: Serialize + ?Sized>(py: Python<'_>, obj: &T) -> PyResult<Py<PyDict>> {
    let json_value = serde_json::to_value(obj).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialization error: {}", e))
    })?;
    match json_value {
        Value::Object(map) => {
            let py_dict = PyDict::new_bound(py);
            for (key, value) in map.iter() {
                py_dict.set_item(key, json_value_to_py(py, value)?)?;
            }
            Ok(py_dict.unbind())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Serialized object did not produce a JSON object",
        )),
    }
}

fn json_value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(item) => item.into_py(py),
        Value::Number(item) => {
            if let Some(value) = item.as_i64() {
                value.into_py(py)
            } else if let Some(value) = item.as_u64() {
                value.into_py(py)
            } else if let Some(value) = item.as_f64() {
                value.into_py(py)
            } else {
                py.None()
            }
        }
        Value::String(item) => item.into_py(py),
        Value::Array(items) => {
            let py_list = PyList::empty_bound(py);
            for item in items {
                py_list.append(json_value_to_py(py, item)?)?;
            }
            py_list.into_py(py)
        }
        Value::Object(items) => {
            let py_dict = PyDict::new_bound(py);
            for (key, item) in items {
                py_dict.set_item(key, json_value_to_py(py, item)?)?;
            }
            py_dict.into_py(py)
        }
    })
}

fn hsl_from_dict(dict: &PyDict) -> PyResult<crate::backtest::hsl_runtime::Config> {
    use crate::backtest::hsl_runtime::{Config, Policy};
    fn keys(dict: &PyDict, allowed: &[&str]) -> PyResult<()> {
        for key in dict.keys().iter() {
            let key: String = key.extract()?;
            if !allowed.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!("unknown HSL field: {key}")));
            }
        }
        Ok(())
    }
    fn policy(dict: &PyDict) -> PyResult<Policy> {
        keys(
            dict,
            &[
                "enabled",
                "scale_budget_with_excess_allowance",
                "red_threshold",
                "memory_ema_span_minutes",
                "drawdown_ratio_ema_span_minutes",
                "cooldown_minutes_after_red",
                "panic_close_order_type",
            ],
        )?;
        Ok(Policy {
            enabled: extract_value(dict, "enabled")?,
            scale_budget_with_excess_allowance: match dict
                .get_item("scale_budget_with_excess_allowance")?
            {
                Some(value) => value.extract::<bool>()?,
                None => false,
            },
            red_threshold: extract_value(dict, "red_threshold")?,
            memory_ema_span_minutes: extract_value(dict, "memory_ema_span_minutes")?,
            drawdown_ratio_ema_span_minutes: extract_value(
                dict,
                "drawdown_ratio_ema_span_minutes",
            )?,
            cooldown_minutes_after_red: extract_value(dict, "cooldown_minutes_after_red")?,
            panic_close_order_type: extract_value(dict, "panic_close_order_type")?,
        })
    }
    fn pair(value: &PyAny) -> PyResult<[Policy; 2]> {
        let items: Vec<&PyDict> = value.extract()?;
        if items.len() != 2 {
            return Err(PyValueError::new_err(
                "HSL sides require exactly long and short",
            ));
        }
        Ok([policy(items[0])?, policy(items[1])?])
    }
    keys(dict, &["engine", "mode", "sides", "portfolio", "coins"])?;
    let portfolio: Option<&PyDict> = extract_value(dict, "portfolio")?;
    let coins: &PyDict = extract_value(dict, "coins")?;
    let mut coin_policies = std::collections::BTreeMap::new();
    for (key, value) in coins.iter() {
        coin_policies.insert(key.extract::<String>()?, pair(value)?);
    }
    Ok(Config {
        mode: extract_value(dict, "mode")?,
        sides: pair(extract_value::<&PyAny>(dict, "sides")?)?,
        portfolio: portfolio.map(policy).transpose()?,
        coins: coin_policies,
    })
}

fn backtest_params_from_dict(dict: &PyDict) -> PyResult<BacktestParams> {
    if dict.contains("calculate_legacy_w")? {
        return Err(PyValueError::new_err("calculate_legacy_w is retired; only ordinary and explicit _ew metrics remain"));
    }

    if dict.contains("lookback_ema_span_minutes")? {
        return Err(PyValueError::new_err(
            "lookback_ema_span_minutes is retired; supply independent risk memory spans",
        ));
    }
    if dict.contains("realized_loss_gate_lookback_minutes")? {
        return Err(PyValueError::new_err(
            "realized_loss_gate_lookback_minutes is retired; use loss_gate_memory_ema_span_minutes",
        ));
    }
    let parse_hsl_cfg = |parent: &PyDict, key: &str| -> PyResult<EquityHardStopLossConfig> {
        let item = parent
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("missing required key: {key}")))?;
        let cfg = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("{key} must be a dict")))?;
        let engine = extract_optional_string(cfg, "engine", "hsl")?;
        if engine != "hsl" {
            return Err(PyValueError::new_err(
                "unsupported HSL engine; legacy HSL has been removed; migrate to HSL",
            ));
        }
        let hsl = hsl_from_dict(cfg)?;
        Ok(EquityHardStopLossConfig {
            hsl: Some(hsl),
            ..EquityHardStopLossConfig::default()
        })
    };
    let hard_stop_cfg = parse_hsl_cfg(dict, "equity_hard_stop_loss")?;

    let params = BacktestParams {
        starting_balance: extract_value(dict, "starting_balance")?,
        maker_fee: extract_value(dict, "maker_fee")?,
        taker_fee: extract_value(dict, "taker_fee")?,
        coins: extract_value(dict, "coins")?,
        active_coin_indices: match dict.get_item("active_coin_indices")? {
            Some(item) if !item.is_none() => Some(item.extract::<Vec<usize>>()?),
            _ => None,
        },
        first_timestamp_ms: extract_value(dict, "first_timestamp_ms")?,
        requested_start_timestamp_ms: extract_value(dict, "requested_start_timestamp_ms")?,
        first_valid_indices: extract_value(dict, "first_valid_indices")?,
        last_valid_indices: extract_value(dict, "last_valid_indices")?,
        warmup_minutes: extract_value(dict, "warmup_minutes")?,
        trade_start_indices: extract_value(dict, "trade_start_indices")?,
        global_warmup_bars: extract_value(dict, "global_warmup_bars")?,
        btc_collateral_cap: extract_value(dict, "btc_collateral_cap")?,
        btc_collateral_ltv_cap: match dict.get_item("btc_collateral_ltv_cap")? {
            Some(item) if !item.is_none() => Some(item.extract::<f64>()?),
            _ => None,
        },
        metrics_only: dict
            .get_item("metrics_only")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        ew_half_life_days: match dict.get_item("ew_half_life_days")? {
            Some(item) if !item.is_none() => {
                let value = item.extract::<f64>()?;
                if !value.is_finite() || value < 1.0 {
                    return Err(PyValueError::new_err(
                        "ew_half_life_days must be finite and at least 1 day",
                    ));
                }
                Some(value)
            }
            _ => None,
        },
        hsl_detailed_report: dict
            .get_item("hsl_detailed_report")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        skip_btc_analysis: dict
            .get_item("skip_btc_analysis")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        filter_by_min_effective_cost: dict
            .get_item("filter_by_min_effective_cost")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        dynamic_wel_by_tradability: extract_value(dict, "dynamic_wel_by_tradability")?,
        hedge_mode: dict
            .get_item("hedge_mode")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(true),
        max_realized_loss_pct: dict
            .get_item("max_realized_loss_pct")?
            .map(|item| item.extract::<f64>())
            .transpose()?
            .unwrap_or(1.0),
        unstuck_memory_ema_span_minutes: extract_value(dict, "unstuck_memory_ema_span_minutes")?,
        loss_gate_memory_ema_span_minutes: extract_value(
            dict,
            "loss_gate_memory_ema_span_minutes",
        )?,
        liquidation_threshold: dict
            .get_item("liquidation_threshold")?
            .map(|item| item.extract::<f64>())
            .transpose()?
            .unwrap_or(0.05),
        equity_hard_stop_loss: hard_stop_cfg,
        market_orders_allowed: dict
            .get_item("market_orders_allowed")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        market_order_near_touch_threshold: dict
            .get_item("market_order_near_touch_threshold")?
            .map(|item| item.extract::<f64>())
            .transpose()?
            .unwrap_or(0.001),
        // Legacy native payloads predate this optional simulation setting.
        // Default only absence; explicit malformed values still fail below.
        limit_order_fill_buffer_pct: dict
            .get_item("limit_order_fill_buffer_pct")?
            .map(|item| item.extract::<f64>())
            .transpose()?
            .unwrap_or(0.0),
        market_order_slippage_pct: dict
            .get_item("market_order_slippage_pct")?
            .map(|item| item.extract::<f64>())
            .transpose()?
            .unwrap_or(0.0005),
        forager_score_hysteresis_pct: dict
            .get_item("forager_score_hysteresis_pct")?
            .map(|item| item.extract::<f64>())
            .transpose()?
            .unwrap_or(0.02),
        candle_interval_minutes: dict
            .get_item("candle_interval_minutes")?
            .map(|item| item.extract::<u64>())
            .transpose()?
            .unwrap_or(1), // default to 1m candles
    };
    for span in [
        params.unstuck_memory_ema_span_minutes,
        params.loss_gate_memory_ema_span_minutes,
    ] {
        crate::hsl_decay::RealizedMemory::new(
            span,
            i64::try_from(params.first_timestamp_ms)
                .map_err(|_| PyValueError::new_err("risk timestamp exceeds i64"))?,
        )
        .map_err(PyValueError::new_err)?;
    }
    crate::limit_fills::validate_buffer(params.limit_order_fill_buffer_pct)
        .map_err(PyValueError::new_err)?;
    if let Some(hsl) = &params.equity_hard_stop_loss.hsl {
        hsl.validate(&params.coins, params.candle_interval_minutes)
            .map_err(PyValueError::new_err)?;
    }
    Ok(params)
}

fn exchange_params_from_dict(dict: &PyDict) -> PyResult<ExchangeParams> {
    let params = ExchangeParams {
        qty_step: extract_value(dict, "qty_step")?,
        price_step: extract_value(dict, "price_step")?,
        min_qty: extract_value(dict, "min_qty")?,
        min_cost: extract_value(dict, "min_cost")?,
        c_mult: extract_value(dict, "c_mult")?,
        maker_fee: extract_value(dict, "maker_fee")?,
        taker_fee: extract_value(dict, "taker_fee")?,
    };
    params.validate_required().map_err(PyValueError::new_err)?;
    Ok(params)
}

fn bot_params_pair_from_dict(dict: &PyDict, hsl: bool) -> PyResult<BotParamsPair> {
    Ok(BotParamsPair {
        long: bot_params_from_dict(extract_value(dict, "long")?, hsl)?,
        short: bot_params_from_dict(extract_value(dict, "short")?, hsl)?,
    })
}

fn trailing_martingale_strategy_params_from_dict(dict: &PyDict) -> PyResult<Value> {
    let entry = extract_value::<&PyDict>(dict, "entry")?;
    let close = extract_value::<&PyDict>(dict, "close")?;
    let params = serde_json::json!({
        "volatility_ema_span_1h": extract_value::<f64>(dict, "volatility_ema_span_1h")?,
        "volatility_ema_span_1m": extract_value::<f64>(dict, "volatility_ema_span_1m")?,
        "entry": {
            "ema_span_0": extract_value::<f64>(entry, "ema_span_0")?,
            "ema_span_1": extract_value::<f64>(entry, "ema_span_1")?,
            "double_down_factor": extract_value::<f64>(entry, "double_down_factor")?,
            "ema_gate_mode": extract_optional_string(entry, "ema_gate_mode", "initial")?,
            "initial_ema_dist": extract_value::<f64>(entry, "initial_ema_dist")?,
            "initial_qty_pct": extract_value::<f64>(entry, "initial_qty_pct")?,
            "threshold_base_pct": extract_value::<f64>(entry, "threshold_base_pct")?,
            "threshold_we_weight": extract_value::<f64>(entry, "threshold_we_weight")?,
            "threshold_volatility_1h_weight": extract_value::<f64>(entry, "threshold_volatility_1h_weight")?,
            "threshold_volatility_1m_weight": extract_value::<f64>(entry, "threshold_volatility_1m_weight")?,
            "retracement_base_pct": extract_value::<f64>(entry, "retracement_base_pct")?,
            "retracement_we_weight": extract_value::<f64>(entry, "retracement_we_weight")?,
            "retracement_volatility_1h_weight": extract_value::<f64>(entry, "retracement_volatility_1h_weight")?,
            "retracement_volatility_1m_weight": extract_value::<f64>(entry, "retracement_volatility_1m_weight")?,
        },
        "close": {
            "qty_pct": extract_value::<f64>(close, "qty_pct")?,
            "threshold_base_pct": extract_value::<f64>(close, "threshold_base_pct")?,
            "threshold_we_weight": extract_value::<f64>(close, "threshold_we_weight")?,
            "threshold_volatility_1h_weight": extract_value::<f64>(close, "threshold_volatility_1h_weight")?,
            "threshold_volatility_1m_weight": extract_value::<f64>(close, "threshold_volatility_1m_weight")?,
            "retracement_base_pct": extract_value::<f64>(close, "retracement_base_pct")?,
            "retracement_volatility_1h_weight": extract_value::<f64>(close, "retracement_volatility_1h_weight")?,
            "retracement_volatility_1m_weight": extract_value::<f64>(close, "retracement_volatility_1m_weight")?,
        },
    });
    validate_strategy_py_dict_keys(dict, "strategy_params.trailing_martingale", &params)?;
    Ok(params)
}

fn ema_anchor_strategy_params_from_dict(dict: &PyDict) -> PyResult<Value> {
    let params = serde_json::json!({
        "base_qty_pct": extract_value::<f64>(dict, "base_qty_pct")?,
        "ema_span_0": extract_value::<f64>(dict, "ema_span_0")?,
        "ema_span_1": extract_value::<f64>(dict, "ema_span_1")?,
        "entry_double_down_factor": extract_optional_f64(dict, "entry_double_down_factor")?,
        "offset": extract_value::<f64>(dict, "offset")?,
        "offset_volatility_ema_span_1m": extract_optional_f64(dict, "offset_volatility_ema_span_1m")?,
        "offset_volatility_1m_weight": extract_optional_f64(dict, "offset_volatility_1m_weight")?,
        "offset_volatility_ema_span_1h": extract_optional_f64(dict, "offset_volatility_ema_span_1h")?,
        "offset_volatility_1h_weight": extract_optional_f64(dict, "offset_volatility_1h_weight")?,
        "offset_psize_weight": extract_value::<f64>(dict, "offset_psize_weight")?,
    });
    validate_strategy_py_dict_keys(dict, "strategy_params.ema_anchor", &params)?;
    Ok(params)
}

fn validate_py_dict_keys(dict: &PyDict, context: &str, allowed: &[&str]) -> PyResult<()> {
    for (key, _value) in dict.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| PyValueError::new_err(format!("{} contains a non-string key", context)))?;
        if !allowed.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "{} contains unknown key: {}",
                context, key
            )));
        }
    }
    Ok(())
}

fn validate_strategy_py_dict_keys(dict: &PyDict, context: &str, recognized: &Value) -> PyResult<()> {
    // Compare the original input with the fields this binding actually
    // transports. Do not let projection erase a typo before strict Rust serde.
    let object = recognized
        .as_object()
        .ok_or_else(|| PyValueError::new_err("strategy binding must construct an object"))?;
    let allowed: Vec<&str> = object.keys().map(String::as_str).collect();
    validate_py_dict_keys(dict, context, &allowed)?;
    for (key, child) in object {
        if child.is_object() {
            validate_strategy_py_dict_keys(
                extract_value::<&PyDict>(dict, key)?,
                &format!("{context}.{key}"),
                child,
            )?;
        } else if child.is_null() {
            // All numeric fields were extracted as f64 above. json! maps
            // NaN/infinity to null; reject it before backtest serde can panic.
            return Err(PyValueError::new_err(format!(
                "{context}.{key} must be finite"
            )));
        }
    }
    Ok(())
}

fn strategy_params_pair_from_dict(
    dict: &PyDict,
    strategy_kind: &str,
) -> PyResult<StrategyParamsPairValue> {
    validate_py_dict_keys(dict, "strategy_params", &["long", "short"])?;
    let long_dict = dict
        .get_item("long")?
        .ok_or_else(|| PyValueError::new_err("missing required key: long"))?
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("strategy_params.long must be a dict"))?;
    let short_dict = dict
        .get_item("short")?
        .ok_or_else(|| PyValueError::new_err("missing required key: short"))?
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("strategy_params.short must be a dict"))?;
    let (long, short) = match strategy_kind {
        "trailing_martingale" => (
            trailing_martingale_strategy_params_from_dict(long_dict)?,
            trailing_martingale_strategy_params_from_dict(short_dict)?,
        ),
        "ema_anchor" => (
            ema_anchor_strategy_params_from_dict(long_dict)?,
            ema_anchor_strategy_params_from_dict(short_dict)?,
        ),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown strategy kind for strategy params: {}",
                strategy_kind
            )))
        }
    };
    Ok(StrategyParamsPairValue { long, short })
}

fn extract_optional_f64(dict: &PyDict, key: &str) -> PyResult<f64> {
    Ok(match dict.get_item(key)? {
        Some(item) => item.extract::<f64>()?,
        None => 0.0,
    })
}

fn extract_optional_bool(dict: &PyDict, key: &str, default: bool) -> PyResult<bool> {
    Ok(match dict.get_item(key)? {
        Some(item) => item.extract::<bool>()?,
        None => default,
    })
}

fn extract_optional_string(dict: &PyDict, key: &str, default: &str) -> PyResult<String> {
    Ok(match dict.get_item(key)? {
        Some(item) => item.extract::<String>()?,
        None => default.to_string(),
    })
}

fn extract_optional_twel_enforcer_policy(dict: &PyDict) -> PyResult<TwelEnforcerPolicy> {
    Ok(match dict.get_item("risk_twel_enforcer_policy")? {
        Some(item) => {
            if item.is_none() {
                TwelEnforcerPolicy::default()
            } else {
                let raw = item.extract::<String>()?;
                TwelEnforcerPolicy::from_str(raw.trim()).map_err(|_| {
                    PyValueError::new_err(format!(
                        "risk_twel_enforcer_policy must be one of: reduce_overweight, reduce_portfolio; got {:?}",
                        raw
                    ))
                })?
            }
        }
        None => TwelEnforcerPolicy::default(),
    })
}

fn bot_params_from_dict(dict: &PyDict, hsl: bool) -> PyResult<BotParams> {
    for key in [
        "forager_volume_drop_pct",
        "filter_volume_drop_pct",
        "risk_twel_entry_gate_enabled",
    ] {
        if dict.contains(key)? {
            return Err(PyValueError::new_err(format!(
                "{key} is retired in canonical v9 inputs"
            )));
        }
    }
    // Retired producer fields must not silently change native risk semantics.
    for key in ["risk_we_excess_allowance_mode", "we_excess_allowance_mode"] {
        if dict.contains(key)? {
            return Err(PyValueError::new_err(format!(
                "{key} is retired; normalize the config with the current loader. \
                 Excess allowance is always bounded by side TWEL; explicit raw sizing \
                 cannot be migrated automatically. Review headroom and re-backtest before removing the field."
            )));
        }
    }
    let risk_wel_enforcer_threshold: f64 = extract_value(dict, "risk_wel_enforcer_threshold")?;
    let risk_twel_enforcer_threshold: f64 = extract_value(dict, "risk_twel_enforcer_threshold")?;
    let risk_we_excess_allowance_pct: f64 = extract_value(dict, "risk_we_excess_allowance_pct")?;
    let total_wallet_exposure_limit: f64 = extract_value(dict, "total_wallet_exposure_limit")?;
    let wallet_exposure_limit_raw: f64 = extract_value(dict, "wallet_exposure_limit")?;
    let n_positions_float: f64 = extract_value(dict, "n_positions")?;
    let n_positions = n_positions_float.round() as usize;
    let hsl = if hsl {
        for key in dict.keys().iter() {
            if key.extract::<String>()?.starts_with("hsl_") {
                return Err(PyValueError::new_err(
                    "HSL policies belong only in equity_hard_stop_loss",
                ));
            }
        }
        // Only the explicit hsl scope policy can enable protective orders.
        // HSL policies are transported separately from ordinary order parameters.
        BotParams::default()
    } else {
        return Err(PyValueError::new_err("obsolete backtest HSL payload"));
    };
    // Callers resolve live fixed denominators before building orchestrator input.
    // Preserve zero here: per-symbol zero is the explicit side-disable sentinel.
    let wallet_exposure_limit = wallet_exposure_limit_raw;

    let params = BotParams {
        close_grid_qty_pct: extract_optional_f64(dict, "close_grid_qty_pct")?,
        close_trailing_retracement_pct: extract_optional_f64(
            dict,
            "close_trailing_retracement_pct",
        )?,
        close_trailing_qty_pct: extract_optional_f64(dict, "close_trailing_qty_pct")?,
        close_trailing_threshold_pct: extract_optional_f64(dict, "close_trailing_threshold_pct")?,
        close_weight_volatility_1h: extract_optional_f64(dict, "close_weight_volatility_1h")?,
        close_weight_volatility_1m: extract_optional_f64(dict, "close_weight_volatility_1m")?,
        entry_grid_double_down_factor: extract_optional_f64(dict, "entry_grid_double_down_factor")?,
        entry_grid_spacing_pct: extract_optional_f64(dict, "entry_grid_spacing_pct")?,
        entry_volatility_ema_span_1h: match dict.get_item("entry_volatility_ema_span_1h")? {
            Some(item) => item.extract::<f64>()?,
            None => match dict.get_item("entry_log_range_ema_span_hours")? {
                Some(item) => item.extract::<f64>()?,
                None => 0.0,
            },
        },
        entry_volatility_ema_span_1m: extract_optional_f64(dict, "entry_volatility_ema_span_1m")?,
        entry_weight_volatility_1h: extract_optional_f64(dict, "entry_weight_volatility_1h")?,
        entry_weight_volatility_1m: extract_optional_f64(dict, "entry_weight_volatility_1m")?,
        entry_we_weight: extract_optional_f64(dict, "entry_we_weight")?,
        entry_initial_ema_dist: extract_optional_f64(dict, "entry_initial_ema_dist")?,
        entry_initial_qty_pct: extract_optional_f64(dict, "entry_initial_qty_pct")?,
        entry_trailing_double_down_factor: extract_optional_f64(
            dict,
            "entry_trailing_double_down_factor",
        )?,
        entry_trailing_retracement_pct: extract_optional_f64(
            dict,
            "entry_trailing_retracement_pct",
        )?,
        entry_trailing_threshold_pct: extract_optional_f64(dict, "entry_trailing_threshold_pct")?,
        filter_volatility_ema_span_1m: extract_value_with_fallback(
            dict,
            "forager_volatility_ema_span_1m",
            "filter_volatility_ema_span_1m",
        )
        .or_else(|_| {
            extract_value_with_fallback(
                dict,
                "filter_volatility_ema_span_1m",
                "filter_log_range_ema_span",
            )
        })?,
        filter_volume_ema_span_1m: extract_value_with_fallback(
            dict,
            "forager_volume_ema_span_1m",
            "filter_volume_ema_span_1m",
        )?,
        _legacy_filter_volatility_drop_pct: 0.0,
        forager_score_weights: dict
            .get_item("forager_score_weights")
            .ok()
            .flatten()
            .map(|_| extract_forager_score_weights(dict))
            .transpose()?
            .unwrap_or_default(),
        is_forced_active: extract_optional_bool(dict, "is_forced_active", false)?,
        entry_eligible: extract_optional_bool(dict, "entry_eligible", true)?,
        ema_span_0: extract_optional_f64(dict, "ema_span_0")?,
        ema_span_1: extract_optional_f64(dict, "ema_span_1")?,
        hsl_enabled: hsl.hsl_enabled,
        hsl_panic_close_order_type: hsl.hsl_panic_close_order_type,
        entry_cooldown_min_duration_minutes: extract_optional_f64(
            dict,
            "entry_cooldown_min_duration_minutes",
        )?,
        entry_cooldown_max_duration_minutes: match dict
            .get_item("entry_cooldown_max_duration_minutes")?
        {
            Some(v) if !v.is_none() => Some(v.extract()?),
            _ => None,
        },
        entry_cooldown_weights_minutes: match dict.get_item("entry_cooldown_weights_minutes")? {
            Some(v) => {
                let w = v.downcast::<PyDict>()?;
                crate::entry_cooldown::CooldownWeights {
                    exposure_ratio: extract_value(w, "exposure_ratio")?,
                    adverse_directionality: extract_value(w, "adverse_directionality")?,
                }
            }
            None => Default::default(),
        },
        unilateralness_ema_span_1m: match dict.get_item("unilateralness_ema_span_1m")? {
            Some(v) => v.extract()?,
            None => 60.0,
        },
        risk_entry_cooldown_minutes: extract_optional_f64(dict, "risk_entry_cooldown_minutes")?,
        n_positions,
        total_wallet_exposure_limit,
        wallet_exposure_limit,
        risk_wel_enforcer_enabled: extract_optional_bool(dict, "risk_wel_enforcer_enabled", true)?,
        risk_wel_enforcer_threshold,
        risk_twel_enforcer_enabled: extract_optional_bool(
            dict,
            "risk_twel_enforcer_enabled",
            true,
        )?,
        risk_twel_enforcer_policy: extract_optional_twel_enforcer_policy(dict)?,
        risk_twel_enforcer_threshold,
        risk_we_excess_allowance_pct,
        unstuck_enabled: extract_optional_bool(dict, "unstuck_enabled", true)?,
        unstuck_ema_gating_enabled: extract_optional_bool(
            dict,
            "unstuck_ema_gating_enabled",
            true,
        )?,
        unstuck_close_pct: extract_value(dict, "unstuck_close_pct")?,
        unstuck_ema_dist: extract_value(dict, "unstuck_ema_dist")?,
        unstuck_ema_span_0: extract_value(dict, "unstuck_ema_span_0")?,
        unstuck_ema_span_1: extract_value(dict, "unstuck_ema_span_1")?,
        unstuck_loss_allowance_pct: extract_value(dict, "unstuck_loss_allowance_pct")?,
        unstuck_threshold: extract_value(dict, "unstuck_threshold")?,
    };
    crate::entry_cooldown::validate(&params).map_err(PyValueError::new_err)?;
    crate::unilateralness::warmup_returns(params.unilateralness_ema_span_1m)
        .map_err(PyValueError::new_err)?;
    Ok(params)
}

fn extract_forager_score_weights(dict: &PyDict) -> PyResult<ForagerScoreWeights> {
    let weights_dict = dict
        .get_item("forager_score_weights")
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>("Key 'forager_score_weights' not found")
        })?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Value for key 'forager_score_weights' is None",
            )
        })?
        .downcast::<PyDict>()?;
    let weights = ForagerScoreWeights {
        volume: extract_value(weights_dict, "volume")?,
        ema_readiness: extract_value(weights_dict, "ema_readiness")?,
        volatility: extract_value(weights_dict, "volatility")?,
        unilateralness: extract_optional_f64(weights_dict, "unilateralness")?,
    };
    weights.canonicalize().map_err(PyValueError::new_err)
}

fn validate_forager_score_weights_pair(bot_params: &BotParamsPair) -> PyResult<()> {
    for (pside, params) in [("long", &bot_params.long), ("short", &bot_params.short)] {
        if let Err(err) = params.forager_score_weights.canonicalize() {
            return Err(PyValueError::new_err(format!("bot.{pside}.{err}")));
        }
    }
    Ok(())
}

fn validate_hsl_panic_close_order_type_pair(bot_params: &BotParamsPair) -> PyResult<()> {
    for (pside, params) in [("long", &bot_params.long), ("short", &bot_params.short)] {
        match params.hsl_panic_close_order_type.as_str() {
            "market" | "limit" => {}
            raw => {
                return Err(PyValueError::new_err(format!(
                    "bot.{pside}.hsl_panic_close_order_type must be one of: market, limit; got {:?}",
                    raw
                )));
            }
        }
    }
    Ok(())
}

fn validate_finite_range(
    path: &str,
    value: f64,
    min_value: f64,
    max_value: Option<f64>,
    min_inclusive: bool,
) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{path} must be finite")));
    }
    let min_ok = if min_inclusive {
        value >= min_value
    } else {
        value > min_value
    };
    if !min_ok {
        let op = if min_inclusive { ">=" } else { ">" };
        return Err(PyValueError::new_err(format!(
            "{path} must be finite and {op} {min_value}"
        )));
    }
    if let Some(max_value) = max_value {
        if value > max_value {
            return Err(PyValueError::new_err(format!(
                "{path} must be finite and <= {max_value}"
            )));
        }
    }
    Ok(())
}

fn validate_hsl_risk_unstuck_bot_params(
    path_prefix: &str,
    pside: &str,
    params: &BotParams,
) -> PyResult<()> {
    match params.hsl_panic_close_order_type.as_str() {
        "market" | "limit" => {}
        raw => {
            return Err(PyValueError::new_err(format!(
                "{path_prefix}.hsl_panic_close_order_type must be one of: market, limit; got {:?}",
                raw
            )));
        }
    }
    validate_finite_range(
        &format!("{path_prefix}.risk_entry_cooldown_minutes"),
        params.risk_entry_cooldown_minutes,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.total_wallet_exposure_limit"),
        params.total_wallet_exposure_limit,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.wallet_exposure_limit"),
        params.wallet_exposure_limit,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.risk_wel_enforcer_threshold"),
        params.risk_wel_enforcer_threshold,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.risk_twel_enforcer_threshold"),
        params.risk_twel_enforcer_threshold,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.risk_we_excess_allowance_pct"),
        params.risk_we_excess_allowance_pct,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.unstuck_close_pct"),
        params.unstuck_close_pct,
        0.0,
        Some(1.0),
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.unstuck_loss_allowance_pct"),
        params.unstuck_loss_allowance_pct,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        &format!("{path_prefix}.unstuck_threshold"),
        params.unstuck_threshold,
        0.0,
        None,
        true,
    )?;
    for (key, span) in [
        ("unstuck_ema_span_0", params.unstuck_ema_span_0),
        ("unstuck_ema_span_1", params.unstuck_ema_span_1),
    ] {
        if !span.is_finite() || span <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "{path_prefix}.{key} must be positive and finite"
            )));
        }
    }
    if !params.unstuck_ema_dist.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{path_prefix}.unstuck_ema_dist must be finite"
        )));
    }
    if pside == "long" && params.unstuck_ema_dist <= -1.0 {
        return Err(PyValueError::new_err(format!(
            "{path_prefix}.unstuck_ema_dist must be > -1.0"
        )));
    }
    if pside == "short" && params.unstuck_ema_dist >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "{path_prefix}.unstuck_ema_dist must be < 1.0"
        )));
    }
    Ok(())
}

fn validate_hsl_risk_unstuck_orchestrator_input(
    input: &crate::orchestrator::OrchestratorInput,
) -> PyResult<()> {
    validate_hsl_risk_unstuck_bot_params("bot.long", "long", &input.global.global_bot_params.long)?;
    validate_hsl_risk_unstuck_bot_params(
        "bot.short",
        "short",
        &input.global.global_bot_params.short,
    )?;
    for symbol in &input.symbols {
        validate_hsl_risk_unstuck_bot_params(
            &format!("symbols[{}].long.bot_params", symbol.symbol_idx),
            "long",
            &symbol.long.bot_params,
        )?;
        validate_hsl_risk_unstuck_bot_params(
            &format!("symbols[{}].short.bot_params", symbol.symbol_idx),
            "short",
            &symbol.short.bot_params,
        )?;
    }
    Ok(())
}

fn validate_orchestrator_account_risk_inputs(
    input: &crate::orchestrator::OrchestratorInput,
) -> PyResult<()> {
    validate_finite_range("balance", input.balance, 0.0, None, false)?;
    let balance_raw = if input.balance_raw.is_finite() {
        input.balance_raw
    } else {
        input.balance
    };
    validate_finite_range("balance_raw", balance_raw, 0.0, None, false)?;
    validate_finite_range(
        "global.max_realized_loss_pct",
        input.global.max_realized_loss_pct,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        "global.unstuck_allowance_long",
        input.global.unstuck_allowance_long,
        0.0,
        None,
        true,
    )?;
    validate_finite_range(
        "global.unstuck_allowance_short",
        input.global.unstuck_allowance_short,
        0.0,
        None,
        true,
    )?;
    Ok(())
}

fn extract_value<'a, T: pyo3::FromPyObject<'a>>(dict: &'a PyDict, key: &str) -> PyResult<T> {
    dict.get_item(key)
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Key '{}' not found", key))
        })?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Value for key '{}' is None",
                key
            ))
        })
        .and_then(pyo3::FromPyObject::extract)
}

fn extract_value_with_fallback<'a, T: pyo3::FromPyObject<'a>>(
    dict: &'a PyDict,
    primary: &str,
    fallback: &str,
) -> PyResult<T> {
    if let Some(item) = dict.get_item(primary)? {
        return item.extract::<T>();
    }
    if let Some(item) = dict.get_item(fallback)? {
        return item.extract::<T>();
    }
    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
        "Keys '{}' or '{}' not found",
        primary, fallback
    )))
}

fn make_trailing_martingale_entry_params(
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    entry_weight_volatility_1h: f64,
    entry_weight_volatility_1m: f64,
    entry_we_weight: f64,
) -> TrailingMartingaleEntryParams {
    let _ = (
        entry_trailing_double_down_factor,
        entry_trailing_threshold_pct,
    );
    TrailingMartingaleEntryParams {
        ema_span_0: 0.0,
        ema_span_1: 0.0,
        double_down_factor: entry_grid_double_down_factor,
        ema_gate_mode: EmaGateMode::Initial,
        initial_ema_dist: entry_initial_ema_dist,
        initial_qty_pct: entry_initial_qty_pct,
        threshold_base_pct: entry_grid_spacing_pct,
        threshold_we_weight: entry_we_weight,
        threshold_volatility_1h_weight: entry_weight_volatility_1h,
        threshold_volatility_1m_weight: entry_weight_volatility_1m,
        retracement_base_pct: entry_trailing_retracement_pct,
        retracement_we_weight: entry_we_weight,
        retracement_volatility_1h_weight: entry_weight_volatility_1h,
        retracement_volatility_1m_weight: entry_weight_volatility_1m,
    }
}

fn make_trailing_martingale_close_params(
    close_grid_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    close_weight_volatility_1h: f64,
    close_weight_volatility_1m: f64,
) -> TrailingMartingaleCloseParams {
    TrailingMartingaleCloseParams {
        qty_pct: close_grid_qty_pct,
        threshold_base_pct: close_trailing_threshold_pct,
        threshold_we_weight: 0.0,
        threshold_volatility_1h_weight: close_weight_volatility_1h,
        threshold_volatility_1m_weight: close_weight_volatility_1m,
        retracement_base_pct: close_trailing_retracement_pct,
        retracement_volatility_1h_weight: close_weight_volatility_1h,
        retracement_volatility_1m_weight: close_weight_volatility_1m,
    }
}

fn make_runtime_order_context(wallet_exposure_limit: f64) -> RuntimeOrderContext {
    RuntimeOrderContext {
        effective_wallet_exposure_limit: wallet_exposure_limit,
    }
}

fn json_f64(value: &Value, key: &str, default: f64) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn json_usize(value: &Value, key: &str, default: usize) -> usize {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn json_bool(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn bot_params_for_close_diagnostic_json(value: &Value) -> PyResult<BotParams> {
    let mut bot = BotParams::default();
    bot.wallet_exposure_limit = json_f64(value, "wallet_exposure_limit", 0.0);
    bot.total_wallet_exposure_limit = json_f64(value, "total_wallet_exposure_limit", 0.0);
    bot.n_positions = json_usize(value, "n_positions", 1);
    bot.risk_we_excess_allowance_pct = json_f64(value, "risk_we_excess_allowance_pct", 0.0);
    bot.risk_wel_enforcer_enabled = json_bool(value, "risk_wel_enforcer_enabled", true);
    bot.risk_wel_enforcer_threshold = json_f64(value, "risk_wel_enforcer_threshold", 0.0);
    Ok(bot)
}

#[pyfunction]
pub fn calc_trailing_martingale_close_diagnostic_py(input_json: &str) -> PyResult<String> {
    let input: Value = serde_json::from_str(input_json).map_err(|err| {
        PyValueError::new_err(format!(
            "invalid trailing_martingale close diagnostic json: {err}"
        ))
    })?;
    let side = match input.get("pside").and_then(Value::as_str).unwrap_or("long") {
        "long" => StrategySide::Long,
        "short" => StrategySide::Short,
        other => {
            return Err(PyValueError::new_err(format!(
                "pside must be long or short; got {other:?}"
            )))
        }
    };
    let exchange: ExchangeParams = serde_json::from_value(
        input
            .get("exchange")
            .cloned()
            .ok_or_else(|| PyValueError::new_err("missing exchange"))?,
    )
    .map_err(|err| PyValueError::new_err(format!("invalid exchange: {err}")))?;
    let state: StateParams = serde_json::from_value(
        input
            .get("state")
            .cloned()
            .ok_or_else(|| PyValueError::new_err("missing state"))?,
    )
    .map_err(|err| PyValueError::new_err(format!("invalid state: {err}")))?;
    let position: Position = serde_json::from_value(
        input
            .get("position")
            .cloned()
            .ok_or_else(|| PyValueError::new_err("missing position"))?,
    )
    .map_err(|err| PyValueError::new_err(format!("invalid position: {err}")))?;
    let trailing: TrailingPriceBundle = serde_json::from_value(
        input
            .get("trailing")
            .cloned()
            .ok_or_else(|| PyValueError::new_err("missing trailing"))?,
    )
    .map_err(|err| PyValueError::new_err(format!("invalid trailing: {err}")))?;
    let close_params: TrailingMartingaleCloseParams = serde_json::from_value(
        input
            .get("close_params")
            .cloned()
            .ok_or_else(|| PyValueError::new_err("missing close_params"))?,
    )
    .map_err(|err| PyValueError::new_err(format!("invalid close_params: {err}")))?;
    let bot = bot_params_for_close_diagnostic_json(
        input
            .get("bot_params")
            .ok_or_else(|| PyValueError::new_err("missing bot_params"))?,
    )?;
    let runtime = RuntimeOrderContext {
        effective_wallet_exposure_limit: input
            .get("runtime")
            .and_then(|runtime| runtime.get("effective_wallet_exposure_limit"))
            .and_then(Value::as_f64)
            .unwrap_or(bot.wallet_exposure_limit),
    };
    let out = calc_trailing_martingale_close_diagnostic(
        side,
        &exchange,
        &state,
        &bot,
        &runtime,
        &close_params,
        &position,
        &trailing,
    );
    serde_json::to_string(&out)
        .map_err(|err| PyValueError::new_err(format!("failed to serialize diagnostic: {err}")))
}

#[pyfunction]
pub fn calc_next_entry_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    entry_weight_volatility_1h: f64,
    entry_weight_volatility_1m: f64,
    entry_we_weight: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_lower: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    order_book_bid: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };
    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            bid: order_book_bid,
            ..Default::default()
        },
        ema_bands: EMABands {
            lower: ema_bands_lower,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };
    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        ..Default::default()
    };
    let entry_params = make_trailing_martingale_entry_params(
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
    );
    let position = Position {
        size: position_size,
        price: position_price,
    };
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let next_entry = calc_next_entry_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &entry_params,
        &position,
        &trailing_price_bundle,
    );

    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}

#[pyfunction(signature = (
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    close_grid_qty_pct,
    close_trailing_qty_pct,
    close_trailing_retracement_pct,
    close_trailing_threshold_pct,
    wallet_exposure_limit,
    risk_we_excess_allowance_pct,
    risk_wel_enforcer_threshold,
    balance,
    position_size,
    position_price,
    min_since_open,
    max_since_min,
    max_since_open,
    min_since_max,
    order_book_ask,
    ema_bands_upper = 0.0,
    volatility_ema_1h = 0.0,
    volatility_ema_1m = 0.0,
    close_weight_volatility_1h = 0.0,
    close_weight_volatility_1m = 0.0
))]
pub fn calc_next_close_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_qty_pct: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    risk_wel_enforcer_threshold: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_ask: f64,
    ema_bands_upper: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    close_weight_volatility_1h: f64,
    close_weight_volatility_1m: f64,
) -> PyResult<(f64, f64, String)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };
    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };
    let bot_params = BotParams {
        close_grid_qty_pct,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
        ..Default::default()
    };
    let close_params = make_trailing_martingale_close_params(
        close_grid_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
    );
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let position = Position {
        size: position_size,
        price: position_price,
    };
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let next_entry = calc_next_close_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &close_params,
        &position,
        &trailing_price_bundle,
    );
    Ok((
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    ))
}

#[pyfunction]
pub fn calc_next_entry_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    entry_weight_volatility_1h: f64,
    entry_weight_volatility_1m: f64,
    entry_we_weight: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_upper: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    order_book_ask: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };
    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };
    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        ..Default::default()
    };
    let entry_params = make_trailing_martingale_entry_params(
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
    );
    let position = Position {
        size: position_size,
        price: position_price,
    };
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let next_entry = calc_next_entry_short(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &entry_params,
        &position,
        &trailing_price_bundle,
    );

    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}

#[pyfunction(signature = (
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    close_grid_qty_pct,
    close_trailing_qty_pct,
    close_trailing_retracement_pct,
    close_trailing_threshold_pct,
    wallet_exposure_limit,
    risk_we_excess_allowance_pct,
    risk_wel_enforcer_threshold,
    balance,
    position_size,
    position_price,
    min_since_open,
    max_since_min,
    max_since_open,
    min_since_max,
    order_book_bid,
    ema_bands_lower = 0.0,
    volatility_ema_1h = 0.0,
    volatility_ema_1m = 0.0,
    close_weight_volatility_1h = 0.0,
    close_weight_volatility_1m = 0.0
))]
pub fn calc_next_close_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_qty_pct: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    risk_wel_enforcer_threshold: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_bid: f64,
    ema_bands_lower: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    close_weight_volatility_1h: f64,
    close_weight_volatility_1m: f64,
) -> PyResult<(f64, f64, String)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };
    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            bid: order_book_bid,
            ..Default::default()
        },
        ema_bands: EMABands {
            lower: ema_bands_lower,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };
    let bot_params = BotParams {
        close_grid_qty_pct,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
        ..Default::default()
    };
    let close_params = make_trailing_martingale_close_params(
        close_grid_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
    );
    let position = Position {
        size: position_size,
        price: position_price,
    };
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let next_entry = calc_next_close_short(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &close_params,
        &position,
        &trailing_price_bundle,
    );
    Ok((
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    ))
}

#[pyfunction]
pub fn calc_entries_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    entry_weight_volatility_1h: f64,
    entry_weight_volatility_1m: f64,
    entry_we_weight: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_lower: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    order_book_bid: f64,
) -> Vec<(f64, f64, u16)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };

    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            bid: order_book_bid,
            ..Default::default()
        },
        ema_bands: EMABands {
            lower: ema_bands_lower,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };

    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        ..Default::default()
    };
    let entry_params = make_trailing_martingale_entry_params(
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
    );

    let position = Position {
        size: position_size,
        price: position_price,
    };
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let entries = calc_entries_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &entry_params,
        &position,
        &trailing_price_bundle,
    );

    // Convert entries to Python-compatible format
    entries
        .into_iter()
        .map(|order| (order.qty, order.price, order.order_type.id()))
        .collect()
}

#[pyfunction]
pub fn calc_entries_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    entry_weight_volatility_1h: f64,
    entry_weight_volatility_1m: f64,
    entry_we_weight: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_upper: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    order_book_ask: f64,
) -> Vec<(f64, f64, u16)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };

    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };

    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        ..Default::default()
    };
    let entry_params = make_trailing_martingale_entry_params(
        entry_grid_double_down_factor,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        entry_weight_volatility_1h,
        entry_weight_volatility_1m,
        entry_we_weight,
    );

    let position = Position {
        size: position_size,
        price: position_price,
    };
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let entries = calc_entries_short(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &entry_params,
        &position,
        &trailing_price_bundle,
    );

    // Convert entries to Python-compatible format
    entries
        .into_iter()
        .map(|order| (order.qty, order.price, order.order_type.id()))
        .collect()
}

#[pyfunction]
pub fn calc_min_entry_qty_py(
    price: f64,
    c_mult: f64,
    qty_step: f64,
    min_qty: f64,
    min_cost: f64,
) -> f64 {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step: 0.0,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };
    crate::entries::calc_min_entry_qty(price, &exchange_params)
}

#[pyfunction(signature = (
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    close_grid_qty_pct,
    close_trailing_qty_pct,
    close_trailing_retracement_pct,
    close_trailing_threshold_pct,
    wallet_exposure_limit,
    risk_we_excess_allowance_pct,
    risk_wel_enforcer_threshold,
    balance,
    position_size,
    position_price,
    min_since_open,
    max_since_min,
    max_since_open,
    min_since_max,
    order_book_ask,
    ema_bands_upper = 0.0,
    volatility_ema_1h = 0.0,
    volatility_ema_1m = 0.0,
    close_weight_volatility_1h = 0.0,
    close_weight_volatility_1m = 0.0
))]
pub fn calc_closes_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_qty_pct: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    risk_wel_enforcer_threshold: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_ask: f64,
    ema_bands_upper: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    close_weight_volatility_1h: f64,
    close_weight_volatility_1m: f64,
) -> PyResult<Vec<(f64, f64, u16)>> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };

    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };

    let bot_params = BotParams {
        close_grid_qty_pct,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
        ..Default::default()
    };
    let close_params = make_trailing_martingale_close_params(
        close_grid_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
    );
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);

    let position = Position {
        size: position_size,
        price: position_price,
    };
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let closes = calc_closes_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &close_params,
        &position,
        &trailing_price_bundle,
    );

    // Convert closes to Python-compatible format
    Ok(closes
        .into_iter()
        .map(|order| (order.qty, order.price, order.order_type.id()))
        .collect())
}

#[pyfunction(signature = (
    qty_step,
    price_step,
    min_qty,
    min_cost,
    c_mult,
    close_grid_qty_pct,
    close_trailing_qty_pct,
    close_trailing_retracement_pct,
    close_trailing_threshold_pct,
    wallet_exposure_limit,
    risk_we_excess_allowance_pct,
    risk_wel_enforcer_threshold,
    balance,
    position_size,
    position_price,
    min_since_open,
    max_since_min,
    max_since_open,
    min_since_max,
    order_book_bid,
    ema_bands_lower = 0.0,
    volatility_ema_1h = 0.0,
    volatility_ema_1m = 0.0,
    close_weight_volatility_1h = 0.0,
    close_weight_volatility_1m = 0.0
))]
pub fn calc_closes_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_qty_pct: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    risk_we_excess_allowance_pct: f64,
    risk_wel_enforcer_threshold: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_bid: f64,
    ema_bands_lower: f64,
    volatility_ema_1h: f64,
    volatility_ema_1m: f64,
    close_weight_volatility_1h: f64,
    close_weight_volatility_1m: f64,
) -> PyResult<Vec<(f64, f64, u16)>> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
        ..Default::default()
    };

    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            bid: order_book_bid,
            ..Default::default()
        },
        ema_bands: EMABands {
            lower: ema_bands_lower,
            ..Default::default()
        },
        volatility_ema_1h,
        volatility_ema_1m: volatility_ema_1m,
        ..Default::default()
    };

    let bot_params = BotParams {
        close_grid_qty_pct,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
        ..Default::default()
    };
    let close_params = make_trailing_martingale_close_params(
        close_grid_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        close_weight_volatility_1h,
        close_weight_volatility_1m,
    );
    let runtime_context = make_runtime_order_context(wallet_exposure_limit);
    let position = Position {
        size: position_size,
        price: position_price,
    };
    let trailing_price_bundle = TrailingPriceBundle {
        min_since_open: min_since_open,
        max_since_min: max_since_min,
        max_since_open: max_since_open,
        min_since_max: min_since_max,
    };
    let closes = calc_closes_short(
        &exchange_params,
        &state_params,
        &bot_params,
        &runtime_context,
        &close_params,
        &position,
        &trailing_price_bundle,
    );

    // Convert closes to Python-compatible format
    Ok(closes
        .into_iter()
        .map(|order| (order.qty, order.price, order.order_type.id()))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    side,
    red_threshold,
    total_wallet_exposure_limit,
    effective_n_positions,
    balance,
    positions,
    skip_idx=None,
    policy=None
))]
pub fn calc_twel_enforcer_orders_py(
    side: &str,
    red_threshold: f64,
    total_wallet_exposure_limit: f64,
    effective_n_positions: usize,
    balance: f64,
    positions: &Bound<'_, PyList>,
    skip_idx: Option<usize>,
    policy: Option<&str>,
) -> PyResult<Vec<(usize, f64, f64, u16)>> {
    let positions = positions.as_ref();
    let side_code = match side {
        "long" => LONG,
        "short" => SHORT,
        _ => {
            return Err(PyValueError::new_err(
                "side must be either 'long' or 'short'",
            ))
        }
    };
    let policy = match policy {
        Some(raw) => TwelEnforcerPolicy::from_str(raw.trim()).map_err(|_| {
            PyValueError::new_err(format!(
                "policy must be one of: reduce_overweight, reduce_portfolio; got {:?}",
                raw
            ))
        })?,
        None => TwelEnforcerPolicy::default(),
    };
    let positions_len = positions.len()?;
    let mut parsed_positions: Vec<TwelEnforcerInputPosition> = Vec::with_capacity(positions_len);
    for item in positions.iter()? {
        let item = item?;
        let dict = item.downcast::<PyDict>()?;
        parsed_positions.push(TwelEnforcerInputPosition {
            idx: dict
                .get_item("idx")?
                .ok_or_else(|| PyValueError::new_err("twel enforcer position missing 'idx'"))?
                .extract::<usize>()?,
            position_size: dict
                .get_item("position_size")?
                .ok_or_else(|| {
                    PyValueError::new_err("twel enforcer position missing 'position_size'")
                })?
                .extract::<f64>()?,
            position_price: dict
                .get_item("position_price")?
                .ok_or_else(|| {
                    PyValueError::new_err("twel enforcer position missing 'position_price'")
                })?
                .extract::<f64>()?,
            market_price: dict
                .get_item("market_price")?
                .ok_or_else(|| {
                    PyValueError::new_err("twel enforcer position missing 'market_price'")
                })?
                .extract::<f64>()?,
            is_managed_candidate: dict
                .get_item("is_managed_candidate")?
                .map(|value| value.extract::<bool>())
                .transpose()?
                .unwrap_or(true),
            c_mult: dict
                .get_item("c_mult")?
                .ok_or_else(|| PyValueError::new_err("twel enforcer position missing 'c_mult'"))?
                .extract::<f64>()?,
            qty_step: dict
                .get_item("qty_step")?
                .ok_or_else(|| PyValueError::new_err("twel enforcer position missing 'qty_step'"))?
                .extract::<f64>()?,
            price_step: dict
                .get_item("price_step")?
                .ok_or_else(|| {
                    PyValueError::new_err("twel enforcer position missing 'price_step'")
                })?
                .extract::<f64>()?,
            min_qty: dict
                .get_item("min_qty")?
                .ok_or_else(|| PyValueError::new_err("twel enforcer position missing 'min_qty'"))?
                .extract::<f64>()?,
            min_cost: dict
                .get_item("min_cost")?
                .ok_or_else(|| PyValueError::new_err("twel enforcer position missing 'min_cost'"))?
                .extract::<f64>()?,
        });
    }

    let actions = calc_twel_enforcer_actions(
        side_code,
        red_threshold,
        total_wallet_exposure_limit,
        effective_n_positions,
        balance,
        &parsed_positions,
        policy,
        skip_idx,
    );
    Ok(actions
        .into_iter()
        .map(|(idx, order)| (idx, order.qty, order.price, order.order_type.id()))
        .collect())
}

#[pyfunction]
pub fn order_type_id_to_snake(id: u16) -> PyResult<String> {
    let ot = OrderType::try_from(id)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("unknown order type id"))?;
    Ok(ot.to_string())
}

#[pyfunction]
pub fn all_order_types_ids(py: Python<'_>) -> PyResult<Py<PyDict>> {
    use strum::IntoEnumIterator;
    let d = PyDict::new_bound(py);
    for ot in OrderType::iter() {
        let id: u16 = ot.into();
        d.set_item(id, ot.to_string())?;
    }
    Ok(d.unbind())
}

#[pyfunction]
pub fn order_type_snake_to_id(name: &str) -> PyResult<u16> {
    OrderType::from_str(name)
        .map(|ot| ot.id())
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("unknown order type name"))
}

#[pyfunction(name = "get_order_id_type_from_string")]
pub fn get_order_id_type_from_string_alias(name: &str) -> PyResult<u16> {
    order_type_snake_to_id(name)
}

// -------- Orchestrator JSON API --------

#[pyfunction]
pub fn compute_protective_closes_json(input_json: &str) -> PyResult<String> {
    let inputs: Vec<crate::orchestrator::ProtectiveCloseInput> =
        serde_json::from_str(input_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let orders = crate::orchestrator::compute_protective_closes(&inputs)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    serde_json::to_string(&orders)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn compute_ideal_orders_json(input_json: &str) -> PyResult<String> {
    let input: crate::orchestrator::OrchestratorInput =
        serde_json::from_str(input_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to parse OrchestratorInput JSON: {}",
                e
            ))
        })?;
    validate_orchestrator_account_risk_inputs(&input)?;
    validate_forager_score_weights_pair(&input.global.global_bot_params)?;
    for symbol in &input.symbols {
        for side in [&symbol.long, &symbol.short] {
            crate::entry_cooldown::validate(&side.bot_params).map_err(PyValueError::new_err)?;
            crate::unilateralness::warmup_returns(side.bot_params.unilateralness_ema_span_1m)
                .map_err(PyValueError::new_err)?;
        }
    }
    validate_hsl_panic_close_order_type_pair(&input.global.global_bot_params)?;
    validate_hsl_risk_unstuck_orchestrator_input(&input)?;

    let out = crate::orchestrator::compute_ideal_orders(&input).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "orchestrator compute_ideal_orders failed: {:?}",
            e
        ))
    })?;

    serde_json::to_string(&out).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to serialize OrchestratorOutput JSON: {}",
            e
        ))
    })
}

#[pyfunction]
pub fn get_strategy_spec(py: Python<'_>, strategy_kind: &str) -> PyResult<Py<PyDict>> {
    let kind = strategy_kind_from_name(strategy_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown strategy kind: {}", strategy_kind))
    })?;
    struct_to_py_dict(py, &strategy_spec(kind))
}

#[pyfunction]
pub fn get_strategy_kinds() -> Vec<&'static str> {
    strategy_kind_names()
}
