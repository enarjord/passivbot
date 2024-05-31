use ndarray::s;
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Debug)]
pub struct BacktestConfig {
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

fn extract_value<'a, T: pyo3::FromPyObject<'a>>(dict: &'a PyDict, key: &str) -> PyResult<T> {
    let item = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Key '{}' not found", key))
    })?;
    item.extract()
}

impl BacktestConfig {
    fn new(dict: &PyDict) -> PyResult<Self> {
        let close_drawdown_pct = extract_value(dict, "close_drawdown_pct")?;
        let close_threshold_pct = extract_value(dict, "close_threshold_pct")?;
        let ema_span_0 = extract_value(dict, "ema_span_0")?;
        let ema_span_1 = extract_value(dict, "ema_span_1")?;
        let initial_ema_dist = extract_value(dict, "initial_ema_dist")?;
        let initial_qty_pct = extract_value(dict, "initial_qty_pct")?;
        let n_positions = extract_value(dict, "n_positions")?;
        let reentry_ddown_factor = extract_value(dict, "reentry_ddown_factor")?;
        let reentry_exposure_weighting = extract_value(dict, "reentry_exposure_weighting")?;
        let reentry_spacing_factor = extract_value(dict, "reentry_spacing_factor")?;
        let total_wallet_exposure_limit = extract_value(dict, "total_wallet_exposure_limit")?;
        let unstuck_close_pct = extract_value(dict, "unstuck_close_pct")?;
        let unstuck_ema_dist = extract_value(dict, "unstuck_ema_dist")?;
        let unstuck_loss_allowance_pct = extract_value(dict, "unstuck_loss_allowance_pct")?;
        let unstuck_threshold = extract_value(dict, "unstuck_threshold")?;

        Ok(BacktestConfig {
            close_drawdown_pct,
            close_threshold_pct,
            ema_span_0,
            ema_span_1,
            initial_ema_dist,
            initial_qty_pct,
            n_positions,
            reentry_ddown_factor,
            reentry_exposure_weighting,
            reentry_spacing_factor,
            total_wallet_exposure_limit,
            unstuck_close_pct,
            unstuck_ema_dist,
            unstuck_loss_allowance_pct,
            unstuck_threshold,
        })
    }
}

#[pyfunction]
pub fn run_backtest(
    hlcs: PyReadonlyArray3<f64>,
    noisiness_indices: PyReadonlyArray2<i32>,
    config: &Bound<PyAny>, // Use Bound<PyAny> instead of PyDict
) -> PyResult<()> {
    // Use `extract` method to convert `&Bound<PyAny>` to `&PyDict`
    let dict: &PyDict = config.extract()?;
    let shape_hlcs = hlcs.shape();
    let shape_noisiness_indices = noisiness_indices.shape();
    let config = BacktestConfig::new(dict)?;

    println!("Shape of the hlcs PyArray3: {:?}", shape_hlcs);
    println!(
        "Shape of the noisiness_indices PyArray2: {:?}",
        shape_noisiness_indices
    );
    println!("{:?}", config);

    if shape_hlcs[0] > 0 && shape_hlcs[1] > 0 && shape_hlcs[2] > 0 {
        println!(
            "First element of hlcs: {:?}",
            hlcs.as_array().slice(s![0, 0, 0])
        );
    } else {
        println!("hlcs array is empty");
    }

    if shape_noisiness_indices[0] > 0 && shape_noisiness_indices[1] > 0 {
        println!(
            "First element of noisiness_indices: {:?}",
            noisiness_indices.as_array().slice(s![0, 0])
        );
    } else {
        println!("noisiness_indices array is empty");
    }

    Ok(())
}
