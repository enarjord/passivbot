use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;
use std::collections::HashSet;

#[pyclass]
#[derive(Debug, Default)]
pub struct LiveConfig {
    close_drawdown_pct: f64,
    close_threshold_pct: f64,
    ema_span_0: f64,
    ema_span_1: f64,
    initial_ema_dist: f64,
    initial_qty_pct: f64,
    n_positions: i32,
    reentry_ddown_factor: f64,
    reentry_exposure_weighting: f64,
    reentry_spacing_factor: f64,
    total_wallet_exposure_limit: f64,
    unstuck_close_pct: f64,
    unstuck_ema_dist: f64,
    unstuck_loss_allowance_pct: f64,
    unstuck_threshold: f64,
}

impl LiveConfig {
    fn from_dict(dict: &PyDict) -> PyResult<Self> {
        Ok(LiveConfig {
            close_drawdown_pct: extract_value(dict, "close_drawdown_pct")?,
            close_threshold_pct: extract_value(dict, "close_threshold_pct")?,
            ema_span_0: extract_value(dict, "ema_span_0")?,
            ema_span_1: extract_value(dict, "ema_span_1")?,
            initial_ema_dist: extract_value(dict, "initial_ema_dist")?,
            initial_qty_pct: extract_value(dict, "initial_qty_pct")?,
            n_positions: extract_value(dict, "n_positions")?,
            reentry_ddown_factor: extract_value(dict, "reentry_ddown_factor")?,
            reentry_exposure_weighting: extract_value(dict, "reentry_exposure_weighting")?,
            reentry_spacing_factor: extract_value(dict, "reentry_spacing_factor")?,
            total_wallet_exposure_limit: extract_value(dict, "total_wallet_exposure_limit")?,
            unstuck_close_pct: extract_value(dict, "unstuck_close_pct")?,
            unstuck_ema_dist: extract_value(dict, "unstuck_ema_dist")?,
            unstuck_loss_allowance_pct: extract_value(dict, "unstuck_loss_allowance_pct")?,
            unstuck_threshold: extract_value(dict, "unstuck_threshold")?,
        })
    }
}

#[pyclass]
#[derive(Debug, Default)]
pub struct BacktestConfig {
    long: LiveConfig,
    short: LiveConfig,
    starting_balance: f64,
    maker_fee: f64,
    c_mults: Vec<f64>,
    symbols: Vec<String>,
    qty_steps: Vec<f64>,
    price_steps: Vec<f64>,
    min_costs: Vec<f64>,
    min_qtys: Vec<f64>,
}

impl BacktestConfig {
    fn from_dict(dict: &PyDict) -> PyResult<Self> {
        Ok(BacktestConfig {
            long: LiveConfig::from_dict(extract_value(dict, "long")?)?,
            short: LiveConfig::from_dict(extract_value(dict, "short")?)?,
            starting_balance: extract_value(dict, "starting_balance")?,
            maker_fee: extract_value(dict, "maker_fee")?,
            c_mults: extract_value(dict, "c_mults")?,
            symbols: extract_value(dict, "symbols")?,
            qty_steps: extract_value(dict, "qty_steps")?,
            price_steps: extract_value(dict, "price_steps")?,
            min_costs: extract_value(dict, "min_costs")?,
            min_qtys: extract_value(dict, "min_qtys")?,
        })
    }
}

#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct Position {
    size: f64,
    price: f64,
}

#[pyclass]
#[derive(Debug)]
pub struct OpenOrder {
    qty: f64,
    price: f64,
    order_type: String,
}

#[pyclass]
#[derive(Debug)]
pub struct Fill {
    minute: i32,
    symbol: String,
    realized_pnl: f64,
    fee_paid: f64,
    balance_after_fill: f64,
    equity_after_fill: f64,
    fill_qty: f64,
    fill_price: f64,
    psize_after_fill: f64,
    pprice_after_fill: f64,
}

#[pyclass]
#[derive(Debug)]
pub struct Stat {
    minute: i32,
    positions_long: Array1<Position>,
    positions_short: Array1<Position>,
    hlc: Array2<f64>,
    balance: f64,
    equity: f64,
}

fn extract_value<'a, T: pyo3::FromPyObject<'a>>(dict: &'a PyDict, key: &str) -> PyResult<T> {
    dict.get_item(key)
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Key '{}' not found", key))
        })?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Value is None"))
        .and_then(pyo3::FromPyObject::extract)
}

fn repeat_elements_to_rows(arr: ArrayView1<f64>, n: usize) -> Array2<f64> {
    let rows = arr.len();
    Array2::from_shape_fn((rows, n), |(i, _)| arr[i])
}

fn prepare_emas_forager(
    spans_long: [f64; 2],
    spans_short: [f64; 2],
    hlcs_first: ArrayView2<f64>,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let prepare_spans = |spans: [f64; 2]| -> Array1<f64> {
        let mut vec = vec![spans[0], spans[1], (spans[0] * spans[1]).sqrt()];
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Array1::from_vec(vec).mapv(|x| if x < 1.0 { 1.0 } else { x })
    };

    let spans_long = prepare_spans(spans_long);
    let spans_short = prepare_spans(spans_short);

    let hlcs_first_col2 = hlcs_first.column(2);
    let emas_long = repeat_elements_to_rows(hlcs_first_col2, 3);
    let emas_short = repeat_elements_to_rows(hlcs_first_col2, 3);

    let compute_alphas = |spans: &Array1<f64>| {
        let alphas = spans.mapv(|x| 2.0 / (x + 1.0));
        let alphas_inv = alphas.mapv(|x| 1.0 - x);
        (alphas, alphas_inv)
    };

    let (alphas_long, alphas_inv_long) = compute_alphas(&spans_long);
    let (alphas_short, alphas_inv_short) = compute_alphas(&spans_short);

    (
        emas_long,
        emas_short,
        alphas_long,
        alphas_inv_long,
        alphas_short,
        alphas_inv_short,
    )
}

fn update_emas_inplace(
    previous_emas: &mut Array2<f64>,
    alphas: &[f64],
    alphas_inv: &[f64],
    current_prices: &[f64],
) {
    for (mut emas, &price) in previous_emas.outer_iter_mut().zip(current_prices) {
        for (ema, (&alpha, &alpha_inv)) in emas.iter_mut().zip(alphas.iter().zip(alphas_inv.iter()))
        {
            *ema = *ema * alpha_inv + price * alpha;
        }
    }
}

#[pyfunction]
pub fn run_backtest(
    py: Python,
    hlcs: PyReadonlyArray3<f64>,
    noisiness_indices: PyReadonlyArray2<i32>,
    config: &Bound<PyAny>,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    let dict: &PyDict = config.extract()?;
    let config = BacktestConfig::from_dict(dict)?;

    let enabled_long = config.long.n_positions > 0;
    let enabled_short = config.short.n_positions > 0;

    let hlcs_array = hlcs.as_array();
    let spans_long = [config.long.ema_span_0, config.long.ema_span_1];
    let spans_short = [config.short.ema_span_0, config.short.ema_span_1];
    let hlcs_first = hlcs_array.slice(s![0, .., ..]).to_owned();

    let (
        mut emas_long,
        mut emas_short,
        alphas_long,
        alphas_inv_long,
        alphas_short,
        alphas_inv_short,
    ) = prepare_emas_forager(spans_long, spans_short, hlcs_first.view());

    let num_steps = hlcs_array.shape()[0];
    let mut tmp_emas = Array3::zeros((num_steps, emas_long.shape()[0], emas_long.shape()[1]));

    for k in 0..num_steps {
        if enabled_long {
            let current_prices = hlcs_array.slice(s![k, .., 2]).to_owned();
            update_emas_inplace(
                &mut emas_long,
                alphas_long.as_slice().unwrap(),
                alphas_inv_long.as_slice().unwrap(),
                current_prices.as_slice().unwrap(),
            );
            tmp_emas.slice_mut(s![k, .., ..]).assign(&emas_long);
        }
    }

    let py_emas: Vec<Py<PyArray2<f64>>> = tmp_emas
        .outer_iter()
        .map(|ema| PyArray2::from_owned_array(py, ema.to_owned()).to_owned())
        .collect();

    Ok(py_emas)
}
