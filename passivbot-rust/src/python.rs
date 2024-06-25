use crate::backtest::Backtest;
use crate::closes::{calc_next_close_long, calc_next_grid_close_long, calc_trailing_close_long};
use crate::entries::{calc_grid_entry_long, calc_next_entry_long, calc_trailing_entry_long};
use crate::types::{
    BacktestParams, BotParams, BotParamsPair, EMABands, ExchangeParams, Order, OrderBook, Position,
    StateParams,
};
use ndarray::{Array2, ArrayBase, ArrayD};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

#[pyfunction]
pub fn run_backtest(
    hlcs: PyReadonlyArray3<f64>,
    noisiness_indices: &PyAny,
    bot_params_pair_dict: &PyDict,
    exchange_params_list: &PyAny,
    backtest_params_dict: &PyDict,
) -> PyResult<Py<PyArray2<PyObject>>> {
    let hlcs_rust = hlcs.as_array();

    let noisiness_indices_rust: Array2<i32> =
        if let Ok(arr) = noisiness_indices.downcast::<PyArray2<i32>>() {
            unsafe { arr.as_array().to_owned() }
        } else if let Ok(arr) = noisiness_indices.downcast::<PyArray2<i64>>() {
            let noisiness_indices_i64: ArrayBase<_, _> = unsafe { arr.as_array() };
            noisiness_indices_i64.mapv(|x| x as i32)
        } else {
            return Err(PyValueError::new_err(
                "Unsupported data type for noisiness_indices",
            ));
        };

    let bot_params_pair = bot_params_pair_from_dict(bot_params_pair_dict)?;
    // convert exchange_params_dict to Vector<ExchangeParams>
    // the python type is list of dicts: [{str: float}]
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
        hlcs_rust.to_owned(),
        noisiness_indices_rust,
        bot_params_pair,
        exchange_params,
        &backtest_params,
    );
    // Convert fills to a 2D array with mixed types
    Python::with_gil(|py| {
        let fills = backtest.run();
        let mut py_fills = Array2::from_elem((fills.len(), 10), py.None());

        for (i, fill) in fills.iter().enumerate() {
            py_fills[(i, 0)] = fill.index.into_py(py);
            py_fills[(i, 1)] = <String as Clone>::clone(&fill.symbol).into_py(py);
            py_fills[(i, 2)] = fill.pnl.into_py(py);
            py_fills[(i, 3)] = fill.fee_paid.into_py(py);
            py_fills[(i, 4)] = fill.balance.into_py(py);
            py_fills[(i, 5)] = fill.fill_qty.into_py(py);
            py_fills[(i, 6)] = fill.fill_price.into_py(py);
            py_fills[(i, 7)] = fill.position_size.into_py(py);
            py_fills[(i, 8)] = fill.position_price.into_py(py);
            py_fills[(i, 9)] = fill.order_type.to_string().into_py(py);
        }

        Ok(py_fills.into_pyarray(py).to_owned())
    })
}

fn backtest_params_from_dict(dict: &PyDict) -> PyResult<BacktestParams> {
    Ok(BacktestParams {
        starting_balance: extract_value(dict, "starting_balance").unwrap_or_default(),
        maker_fee: extract_value(dict, "maker_fee").unwrap_or_default(),
        symbols: extract_value(dict, "symbols").unwrap_or_default(),
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

fn bot_params_from_dict(dict: &PyDict) -> PyResult<BotParams> {
    Ok(BotParams {
        close_grid_markup_range: extract_value(dict, "close_grid_markup_range")?,
        close_grid_min_markup: extract_value(dict, "close_grid_min_markup")?,
        close_grid_qty_pct: extract_value(dict, "close_grid_qty_pct")?,
        close_trailing_drawdown_pct: extract_value(dict, "close_trailing_drawdown_pct")?,
        close_trailing_grid_ratio: extract_value(dict, "close_trailing_grid_ratio")?,
        close_trailing_threshold_pct: extract_value(dict, "close_trailing_threshold_pct")?,
        entry_grid_double_down_factor: extract_value(dict, "entry_grid_double_down_factor")?,
        entry_grid_spacing_weight: extract_value(dict, "entry_grid_spacing_weight")?,
        entry_grid_spacing_pct: extract_value(dict, "entry_grid_spacing_pct")?,
        entry_initial_ema_dist: extract_value(dict, "entry_initial_ema_dist")?,
        entry_initial_qty_pct: extract_value(dict, "entry_initial_qty_pct")?,
        entry_trailing_drawdown_pct: extract_value(dict, "entry_trailing_drawdown_pct")?,
        entry_trailing_grid_ratio: extract_value(dict, "entry_trailing_grid_ratio")?,
        entry_trailing_threshold_pct: extract_value(dict, "entry_trailing_threshold_pct")?,
        ema_span0: extract_value(dict, "ema_span0")?,
        ema_span1: extract_value(dict, "ema_span1")?,
        n_positions: extract_value(dict, "n_positions")?,
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
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Value is None"))
        .and_then(pyo3::FromPyObject::extract)
}

#[pyfunction]
pub fn calc_next_grid_close_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    close_grid_markup_range: f64,
    close_grid_min_markup: f64,
    close_grid_qty_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
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
        close_grid_markup_range,
        close_grid_min_markup,
        close_grid_qty_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
    let position = Position {
        size: position_size,
        price: position_price,
    };

    let order = calc_next_grid_close_long(&exchange_params, &state_params, &bot_params, &position);
    (order.qty, order.price, order.order_type.to_string())
}

#[pyfunction]
pub fn calc_trailing_close_long_py(
    price_step: f64,
    order_book_ask: f64,
    max_price_since_open: f64,
    min_price_since_max: f64,
    close_trailing_threshold_pct: f64,
    close_trailing_drawdown_pct: f64,
    position_size: f64,
    position_price: f64,
) -> (f64, f64, String) {
    let exchange_params = ExchangeParams {
        price_step,
        ..Default::default()
    };
    let state_params = StateParams {
        order_book: OrderBook {
            ask: order_book_ask,
            ..Default::default()
        },
        ..Default::default()
    };
    let bot_params = BotParams {
        close_trailing_drawdown_pct: close_trailing_drawdown_pct,
        close_trailing_threshold_pct: close_trailing_threshold_pct,
        ..Default::default()
    };
    let position = Position {
        size: position_size,
        price: position_price,
    };

    let order = calc_trailing_close_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        max_price_since_open,
        min_price_since_max,
    );
    (order.qty, order.price, order.order_type.to_string())
}

#[pyfunction]
pub fn calc_grid_entry_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    balance: f64,
    order_book_bid: f64,
    ema_bands_lower: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    wallet_exposure_limit: f64,
    position_size: f64,
    position_price: f64,
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
    };
    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
    let position = Position {
        size: position_size,
        price: position_price,
    };

    let order = calc_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
    (order.qty, order.price, order.order_type.to_string())
}

#[pyfunction]
pub fn calc_trailing_entry_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    balance: f64,
    order_book_bid: f64,
    entry_grid_double_down_factor: f64,
    entry_initial_qty_pct: f64,
    wallet_exposure_limit: f64,
    position_size: f64,
    position_price: f64,
    min_price_since_open: f64,
    max_price_since_min: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_drawdown_pct: f64,
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
        entry_grid_double_down_factor,
        entry_initial_qty_pct,
        entry_trailing_threshold_pct,
        entry_trailing_drawdown_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
    let position = Position {
        size: position_size,
        price: position_price,
    };

    let order = calc_trailing_entry_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        min_price_since_open,
        max_price_since_min,
    );
    (order.qty, order.price, order.order_type.to_string())
}

#[pyfunction]
pub fn calc_next_entry_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_drawdown_pct: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_price_since_open: f64,
    max_price_since_min: f64,
    ema_bands_lower: f64,
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
        ..Default::default()
    };
    let bot_params = BotParams {
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_drawdown_pct,
        entry_trailing_grid_ratio,
        entry_trailing_threshold_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
    let position = Position {
        size: position_size,
        price: position_price,
    };

    let next_entry = calc_next_entry_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        min_price_since_open,
        max_price_since_min,
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
    close_grid_markup_range: f64,
    close_grid_min_markup: f64,
    close_grid_qty_pct: f64,
    close_trailing_drawdown_pct: f64,
    close_trailing_grid_ratio: f64,
    close_trailing_threshold_pct: f64,
    wallet_exposure_limit: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
    min_price_since_open: f64,
    max_price_since_min: f64,
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
        close_grid_markup_range,
        close_grid_min_markup,
        close_grid_qty_pct,
        close_trailing_drawdown_pct,
        close_trailing_grid_ratio,
        close_trailing_threshold_pct,
        wallet_exposure_limit,
        ..Default::default()
    };
    let position = Position {
        size: position_size,
        price: position_price,
    };

    let next_entry = calc_next_close_long(
        &exchange_params,
        &state_params,
        &bot_params,
        &position,
        min_price_since_open,
        max_price_since_min,
    );

    (
        next_entry.qty,
        next_entry.price,
        next_entry.order_type.to_string(),
    )
}
