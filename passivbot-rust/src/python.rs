use crate::analysis::analyze_backtest_pair;
use crate::backtest::Backtest;
use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_next_entry_long, calc_next_entry_short,
};
use crate::types::OrderType;
use crate::types::{
    BacktestParams, BotParams, BotParamsPair, EMABands, ExchangeParams, OrderBook, Position,
    StateParams, TrailingPriceBundle,
};
use memmap::MmapOptions;
use ndarray::{Array1, Array2, ArrayView};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::Serialize;
use std::fs::File;
use std::str::FromStr;

#[pyfunction]
pub fn run_backtest(
    shared_memory_file: &str,           // Existing HLCV shared memory file
    hlcvs_shape: (usize, usize, usize), // Shape of HLCV data
    hlcvs_dtype: &str,                  // Dtype of HLCV data
    btc_usd_shared_memory_file: &str,   // New BTC/USD shared memory file
    btc_usd_dtype: &str,                // Dtype of BTC/USD data
    bot_params: &PyAny,                 // Bot parameters per coin
    exchange_params_list: &PyAny,       // Exchange parameters
    backtest_params_dict: &PyDict,      // Backtest parameters
) -> PyResult<(
    Py<PyArray2<PyObject>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyDict>,
    Py<PyDict>,
)> {
    // Open and map the HLCV shared memory file
    let file = File::open(shared_memory_file)
        .map_err(|e| PyValueError::new_err(format!("Unable to open shared memory file: {}", e)))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| PyValueError::new_err(format!("Unable to map HLCV file: {}", e)))?
    };
    let hlcvs_rust = unsafe {
        match hlcvs_dtype {
            "<f8" => ArrayView::from_shape_ptr(hlcvs_shape, mmap.as_ptr() as *const f64),
            _ => return Err(PyValueError::new_err("Unsupported dtype for HLCV data")),
        }
    };

    // Open and map the BTC/USD shared memory file
    let btc_usd_file = File::open(btc_usd_shared_memory_file).map_err(|e| {
        PyValueError::new_err(format!("Unable to open BTC/USD shared memory file: {}", e))
    })?;
    let btc_usd_mmap = unsafe {
        MmapOptions::new()
            .map(&btc_usd_file)
            .map_err(|e| PyValueError::new_err(format!("Unable to map BTC/USD file: {}", e)))?
    };
    let n_timesteps = hlcvs_shape.0; // Number of timesteps from HLCV shape
    let btc_usd_rust = unsafe {
        match btc_usd_dtype {
            "<f8" => ArrayView::from_shape_ptr((n_timesteps,), btc_usd_mmap.as_ptr() as *const f64),
            _ => return Err(PyValueError::new_err("Unsupported dtype for BTC/USD data")),
        }
    };

    // Ensure BTC/USD data length matches HLCV timesteps
    if btc_usd_rust.len() != n_timesteps {
        return Err(PyValueError::new_err(format!(
            "BTC/USD data length ({}) does not match HLCV timesteps ({})",
            btc_usd_rust.len(),
            n_timesteps
        )));
    }

    let bot_params_py_list = bot_params
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("bot_params must be a list[dict] (one per coin)"))?;

    let mut bot_params_vec = Vec::with_capacity(bot_params_py_list.len());
    for item in bot_params_py_list {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each bot_params element must be a dict"))?;
        bot_params_vec.push(bot_params_pair_from_dict(dict)?);
    }

    let exchange_params = {
        let mut params_vec = Vec::new();
        if let Ok(py_list) = exchange_params_list.downcast::<PyList>() {
            for py_dict in py_list.iter() {
                if let Ok(dict) = py_dict.downcast::<PyDict>() {
                    let params = exchange_params_from_dict(dict)?;
                    params_vec.push(params);
                } else {
                    return Err(PyValueError::new_err(
                        "Unsupported data type in exchange_params_list",
                    ));
                }
            }
        } else {
            return Err(PyValueError::new_err(
                "Unsupported data type for exchange_params_list",
            ));
        }
        params_vec
    };

    let backtest_params = backtest_params_from_dict(backtest_params_dict)?;
    let mut backtest = Backtest::new(
        &hlcvs_rust,
        &btc_usd_rust,
        bot_params_vec,
        exchange_params,
        &backtest_params,
    );

    // Run the backtest and process results
    Python::with_gil(|py| {
        let (fills, equities) = backtest.run();
        let (analysis_usd, analysis_btc) = analyze_backtest_pair(
            &fills,
            &equities,
            backtest.balance.use_btc_collateral,
            &backtest.total_wallet_exposures,
        );

        // Create a dictionary to store analysis results using a more concise approach
        let py_analysis_usd = struct_to_py_dict(py, &analysis_usd)?;
        let py_analysis_btc = struct_to_py_dict(py, &analysis_btc)?;
        let mut py_fills = Array2::from_elem((fills.len(), 15), py.None());
        for (i, fill) in fills.iter().enumerate() {
            py_fills[(i, 0)] = fill.index.into_py(py);
            py_fills[(i, 1)] = <String as Clone>::clone(&fill.coin).into_py(py);
            py_fills[(i, 2)] = fill.pnl.into_py(py);
            py_fills[(i, 3)] = fill.fee_paid.into_py(py);
            py_fills[(i, 4)] = fill.balance_usd_total.into_py(py);
            py_fills[(i, 5)] = fill.balance_btc.into_py(py);
            py_fills[(i, 6)] = fill.balance_usd.into_py(py);
            py_fills[(i, 7)] = fill.btc_price.into_py(py);
            py_fills[(i, 8)] = fill.fill_qty.into_py(py);
            py_fills[(i, 9)] = fill.fill_price.into_py(py);
            py_fills[(i, 10)] = fill.position_size.into_py(py);
            py_fills[(i, 11)] = fill.position_price.into_py(py);
            py_fills[(i, 12)] = fill.order_type.to_string().into_py(py);
            py_fills[(i, 13)] = fill.wallet_exposure.into_py(py);
            py_fills[(i, 14)] = fill.total_wallet_exposure.into_py(py);
        }

        let py_equities_usd = Array1::from_vec(equities.usd)
            .into_pyarray_bound(py)
            .unbind();
        let py_equities_btc = Array1::from_vec(equities.btc)
            .into_pyarray_bound(py)
            .unbind();
        Ok((
            py_fills.into_pyarray_bound(py).unbind(),
            py_equities_usd,
            py_equities_btc,
            py_analysis_usd.into(),
            py_analysis_btc.into(),
        ))
    })
}

fn struct_to_py_dict<'py, T: Serialize + ?Sized>(
    py: Python<'py>,
    obj: &T,
) -> PyResult<&'py PyDict> {
    // Convert struct to JSON string
    let json_str = serde_json::to_string(obj).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialization error: {}", e))
    })?;

    // Use Python's json module to convert to a Python dict
    let json = py.import("json")?;
    let py_obj = json.call_method1("loads", (json_str,))?;

    // Convert to PyDict
    py_obj.downcast::<PyDict>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("Failed to convert to Python dict")
    })
}

fn backtest_params_from_dict(dict: &PyDict) -> PyResult<BacktestParams> {
    Ok(BacktestParams {
        starting_balance: extract_value(dict, "starting_balance").unwrap_or_default(),
        maker_fee: extract_value(dict, "maker_fee").unwrap_or_default(),
        coins: extract_value(dict, "coins").unwrap_or_default(),
        // First timestamp (ms); default to 0 if not provided
        first_timestamp_ms: extract_value(dict, "first_timestamp_ms").unwrap_or(0u64),
        requested_start_timestamp_ms: extract_value(dict, "requested_start_timestamp_ms")
            .unwrap_or(0u64),
        first_valid_indices: dict
            .get_item("first_valid_indices")?
            .map(|item| item.extract::<Vec<usize>>())
            .transpose()?
            .unwrap_or_default(),
        last_valid_indices: dict
            .get_item("last_valid_indices")?
            .map(|item| item.extract::<Vec<usize>>())
            .transpose()?
            .unwrap_or_default(),
        warmup_minutes: dict
            .get_item("warmup_minutes")?
            .map(|item| item.extract::<Vec<usize>>())
            .transpose()?
            .unwrap_or_default(),
        trade_start_indices: dict
            .get_item("trade_start_indices")?
            .map(|item| item.extract::<Vec<usize>>())
            .transpose()?
            .unwrap_or_default(),
        global_warmup_bars: extract_value(dict, "global_warmup_bars").unwrap_or(0usize),
    })
}

fn exchange_params_from_dict(dict: &PyDict) -> PyResult<ExchangeParams> {
    Ok(ExchangeParams {
        qty_step: extract_value(dict, "qty_step").unwrap_or_default(),
        price_step: extract_value(dict, "price_step").unwrap_or_default(),
        min_qty: extract_value(dict, "min_qty").unwrap_or_default(),
        min_cost: extract_value(dict, "min_cost").unwrap_or_default(),
        c_mult: extract_value(dict, "c_mult").unwrap_or_default(),
    })
}

fn bot_params_pair_from_dict(dict: &PyDict) -> PyResult<BotParamsPair> {
    Ok(BotParamsPair {
        long: bot_params_from_dict(extract_value(dict, "long")?)?,
        short: bot_params_from_dict(extract_value(dict, "short")?)?,
    })
}

fn extract_bool_value(dict: &PyDict, key: &str) -> PyResult<bool> {
    if let Ok(val) = extract_value::<bool>(dict, key) {
        Ok(val)
    } else if let Ok(val) = extract_value::<i64>(dict, key) {
        Ok(val != 0)
    } else if let Ok(val) = extract_value::<usize>(dict, key) {
        Ok(val != 0)
    } else if let Ok(val) = extract_value::<f64>(dict, key) {
        Ok(val != 0.0)
    } else {
        // If none of the above types match, try to get the value as a bool
        extract_value::<bool>(dict, key)
    }
}

fn extract_grid_spacing_we_weight(dict: &PyDict) -> PyResult<f64> {
    if let Some(obj) = dict.get_item("entry_grid_spacing_we_weight")? {
        obj.extract::<f64>()
    } else {
        extract_value(dict, "entry_grid_spacing_we_weight")
    }
}

fn bot_params_from_dict(dict: &PyDict) -> PyResult<BotParams> {
    Ok(BotParams {
        close_grid_markup_end: extract_value(dict, "close_grid_markup_end")?,
        close_grid_markup_start: extract_value(dict, "close_grid_markup_start")?,
        close_grid_qty_pct: extract_value(dict, "close_grid_qty_pct")?,
        close_trailing_retracement_pct: extract_value(dict, "close_trailing_retracement_pct")?,
        close_trailing_grid_ratio: extract_value(dict, "close_trailing_grid_ratio")?,
        close_trailing_qty_pct: extract_value(dict, "close_trailing_qty_pct")?,
        close_trailing_threshold_pct: extract_value(dict, "close_trailing_threshold_pct")?,
        enforce_exposure_limit: extract_bool_value(dict, "enforce_exposure_limit")?,
        entry_grid_double_down_factor: extract_value(dict, "entry_grid_double_down_factor")?,
        entry_grid_spacing_log_weight: extract_value(dict, "entry_grid_spacing_log_weight")?,
        entry_grid_spacing_we_weight: extract_grid_spacing_we_weight(dict)?,
        entry_grid_spacing_pct: extract_value(dict, "entry_grid_spacing_pct")?,
        entry_grid_spacing_log_span_hours: extract_value(
            dict,
            "entry_grid_spacing_log_span_hours",
        )?,
        entry_initial_ema_dist: extract_value(dict, "entry_initial_ema_dist")?,
        entry_initial_qty_pct: extract_value(dict, "entry_initial_qty_pct")?,
        entry_trailing_double_down_factor: extract_value(
            dict,
            "entry_trailing_double_down_factor",
        )?,
        entry_trailing_retracement_pct: extract_value(dict, "entry_trailing_retracement_pct")?,
        entry_trailing_grid_ratio: extract_value(dict, "entry_trailing_grid_ratio")?,
        entry_trailing_threshold_pct: extract_value(dict, "entry_trailing_threshold_pct")?,
        filter_log_range_ema_span: extract_value(dict, "filter_log_range_ema_span")?,
        filter_volume_ema_span: extract_value(dict, "filter_volume_ema_span")?,
        filter_volume_drop_pct: extract_value(dict, "filter_volume_drop_pct")?,
        ema_span_0: extract_value(dict, "ema_span_0")?,
        ema_span_1: extract_value(dict, "ema_span_1")?,
        n_positions: {
            let n_positions_float: f64 = extract_value(dict, "n_positions")?;
            n_positions_float.round() as usize
        },
        total_wallet_exposure_limit: extract_value(dict, "total_wallet_exposure_limit")?,
        wallet_exposure_limit: extract_value(dict, "wallet_exposure_limit")?,
        unstuck_close_pct: extract_value(dict, "unstuck_close_pct")?,
        unstuck_ema_dist: extract_value(dict, "unstuck_ema_dist")?,
        unstuck_loss_allowance_pct: extract_value(dict, "unstuck_loss_allowance_pct")?,
        unstuck_threshold: extract_value(dict, "unstuck_threshold")?,
    })
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

#[pyfunction]
pub fn calc_next_entry_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_log_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_lower: f64,
    grid_log_range: f64,
    order_book_bid: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
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
        grid_log_range,
        ..Default::default()
    };
    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_log_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
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
    let next_entry = calc_next_entry_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        &trailing_price_bundle,
    );

    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}

#[pyfunction]
pub fn calc_next_close_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_markup_end: f64,
    close_grid_markup_start: f64,
    close_grid_qty_pct: f64,
    close_trailing_grid_ratio: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    enforce_exposure_limit: bool,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_ask: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
    };
    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ..Default::default()
    };
    let bot_params = BotParams {
        close_grid_markup_end,
        close_grid_markup_start,
        close_grid_qty_pct,
        close_trailing_grid_ratio,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        enforce_exposure_limit,
        wallet_exposure_limit,
        ..Default::default()
    };
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
        &position,
        &trailing_price_bundle,
    );
    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}

#[pyfunction]
pub fn calc_next_entry_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_log_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_upper: f64,
    grid_log_range: f64,
    order_book_ask: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
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
        grid_log_range,
        ..Default::default()
    };
    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_log_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
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
    let next_entry = calc_next_entry_short(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        &trailing_price_bundle,
    );

    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}

#[pyfunction]
pub fn calc_next_close_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_markup_end: f64,
    close_grid_markup_start: f64,
    close_grid_qty_pct: f64,
    close_trailing_grid_ratio: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    enforce_exposure_limit: bool,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_bid: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
    };
    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            bid: order_book_bid,
            ..Default::default()
        },
        ..Default::default()
    };
    let bot_params = BotParams {
        close_grid_markup_end,
        close_grid_markup_start,
        close_grid_qty_pct,
        close_trailing_grid_ratio,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        enforce_exposure_limit,
        wallet_exposure_limit,
        ..Default::default()
    };
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
    let next_entry = calc_next_close_short(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        &trailing_price_bundle,
    );
    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}

#[pyfunction]
pub fn calc_entries_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_log_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_lower: f64,
    grid_log_range: f64,
    order_book_bid: f64,
) -> Vec<(f64, f64, u16)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
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
        grid_log_range,
        ..Default::default()
    };

    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_log_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        wallet_exposure_limit,
        ..Default::default()
    };

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
    let entries = calc_entries_long(
        &exchange_params,
        &state_params,
        &bot_params,
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
    entry_grid_spacing_log_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    ema_bands_upper: f64,
    grid_log_range: f64,
    order_book_ask: f64,
) -> Vec<(f64, f64, u16)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
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
        grid_log_range,
        ..Default::default()
    };

    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_log_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_threshold_pct,
        wallet_exposure_limit,
        ..Default::default()
    };

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
    let entries = calc_entries_short(
        &exchange_params,
        &state_params,
        &bot_params,
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
    };
    crate::entries::calc_min_entry_qty(price, &exchange_params)
}

#[pyfunction]
pub fn calc_closes_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_markup_end: f64,
    close_grid_markup_start: f64,
    close_grid_qty_pct: f64,
    close_trailing_grid_ratio: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    enforce_exposure_limit: bool,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_ask: f64,
) -> Vec<(f64, f64, u16)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
    };

    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ..Default::default()
    };

    let bot_params = BotParams {
        close_grid_markup_end,
        close_grid_markup_start,
        close_grid_qty_pct,
        close_trailing_grid_ratio,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        enforce_exposure_limit,
        wallet_exposure_limit,
        ..Default::default()
    };

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
        &position,
        &trailing_price_bundle,
    );

    // Convert closes to Python-compatible format
    closes
        .into_iter()
        .map(|order| (order.qty, order.price, order.order_type.id()))
        .collect()
}

#[pyfunction]
pub fn calc_closes_short_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_markup_end: f64,
    close_grid_markup_start: f64,
    close_grid_qty_pct: f64,
    close_trailing_grid_ratio: f64,
    close_trailing_qty_pct: f64,
    close_trailing_retracement_pct: f64,
    close_trailing_threshold_pct: f64,
    enforce_exposure_limit: bool,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_since_open: f64,
    max_since_min: f64,
    max_since_open: f64,
    min_since_max: f64,
    order_book_bid: f64,
) -> Vec<(f64, f64, u16)> {
    let exchange_params = ExchangeParams {
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult,
    };

    let state_params = StateParams {
        balance,
        order_book: OrderBook {
            bid: order_book_bid,
            ..Default::default()
        },
        ..Default::default()
    };

    let bot_params = BotParams {
        close_grid_markup_end,
        close_grid_markup_start,
        close_grid_qty_pct,
        close_trailing_grid_ratio,
        close_trailing_qty_pct,
        close_trailing_retracement_pct,
        close_trailing_threshold_pct,
        enforce_exposure_limit,
        wallet_exposure_limit,
        ..Default::default()
    };
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
        &position,
        &trailing_price_bundle,
    );

    // Convert closes to Python-compatible format
    closes
        .into_iter()
        .map(|order| (order.qty, order.price, order.order_type.id()))
        .collect()
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
    let d = PyDict::new(py);
    for ot in OrderType::iter() {
        let id: u16 = ot.into();
        d.set_item(id, ot.to_string())?;
    }
    Ok(d.into())
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
