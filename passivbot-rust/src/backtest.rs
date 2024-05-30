use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Debug)]
struct ForagerConfig {
    total_wallet_exposure_limit: f64,
    n_positions: i32,
    unstuck_loss_allowance_pct: f64,
    unstuck_threshold: f64,
    unstuck_close_pct: f64,
    unstuck_ema_dist: f64,
    ddown_factor: f64,
    ema_span_0: f64,
    ema_span_1: f64,
    initial_eprice_ema_dist: f64,
    initial_qty_pct: f64,
    markup_range: f64,
    min_markup: f64,
    n_close_orders: i32,
    rentry_pprice_dist: f64,
    rentry_pprice_dist_wallet_exposure_weighting: f64,
}

fn extract_value<'a, T: pyo3::FromPyObject<'a>>(dict: &'a Bound<PyDict>, key: &str) -> PyResult<T> {
    dict.get_item(key).map_err(|err| err)?.map_or_else(
        || {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Key '{}' not found",
                key
            )))
        },
        |value| value.extract(),
    )
}

impl ForagerConfig {
    fn new(dict: Bound<PyDict>) -> PyResult<Self> {
        let total_wallet_exposure_limit = extract_value(&dict, "total_wallet_exposure_limit")?;
        let n_positions = extract_value(&dict, "n_positions")?;
        let unstuck_loss_allowance_pct = extract_value(&dict, "unstuck_loss_allowance_pct")?;
        let unstuck_threshold = extract_value(&dict, "unstuck_threshold")?;
        let unstuck_close_pct = extract_value(&dict, "unstuck_close_pct")?;
        let unstuck_ema_dist = extract_value(&dict, "unstuck_ema_dist")?;
        let ddown_factor = extract_value(&dict, "ddown_factor")?;
        let ema_span_0 = extract_value(&dict, "ema_span_0")?;
        let ema_span_1 = extract_value(&dict, "ema_span_1")?;
        let initial_eprice_ema_dist = extract_value(&dict, "initial_eprice_ema_dist")?;
        let initial_qty_pct = extract_value(&dict, "initial_qty_pct")?;
        let markup_range = extract_value(&dict, "markup_range")?;
        let min_markup = extract_value(&dict, "min_markup")?;
        let n_close_orders = extract_value(&dict, "n_close_orders")?;
        let rentry_pprice_dist = extract_value(&dict, "rentry_pprice_dist")?;
        let rentry_pprice_dist_wallet_exposure_weighting =
            extract_value(&dict, "rentry_pprice_dist_wallet_exposure_weighting")?;

        Ok(ForagerConfig {
            total_wallet_exposure_limit,
            n_positions,
            unstuck_loss_allowance_pct,
            unstuck_threshold,
            unstuck_close_pct,
            unstuck_ema_dist,
            ddown_factor,
            ema_span_0,
            ema_span_1,
            initial_eprice_ema_dist,
            initial_qty_pct,
            markup_range,
            min_markup,
            n_close_orders,
            rentry_pprice_dist,
            rentry_pprice_dist_wallet_exposure_weighting,
        })
    }
}

#[pyfunction]
pub fn to_struct(dict: Bound<PyDict>) -> PyResult<ForagerConfig> {
    let forager_config = ForagerConfig::new(dict)?;
    println!("{:?}", forager_config);
    Ok(forager_config)
}

#[pyfunction]
pub fn print_dict(dict: Bound<PyDict>) -> PyResult<()> {
    for (key, value) in dict.iter() {
        println!("Key: {:?}, Value: {:?}", key, value);
    }
    Ok(())
}
