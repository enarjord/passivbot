use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;
use std::collections::HashSet;

#[pyclass]
#[derive(Debug)]
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

#[pyclass]
#[derive(Debug)]
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

#[pyclass]
#[derive(Debug, Clone)]
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
    positions_long: Vec<Position>,
    positions_short: Vec<Position>,
    hlc: Array2<f64>,
    balance: f64,
    equity: f64,
}

fn extract_value<'a, T: pyo3::FromPyObject<'a>>(dict: &'a PyDict, key: &str) -> PyResult<T> {
    let item = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Key '{}' not found", key))
    })?;
    item.extract()
}

impl BacktestConfig {
    fn new(dict: &PyDict) -> PyResult<Self> {
        let long_dict: &PyDict = extract_value(dict, "long")?;
        let short_dict: &PyDict = extract_value(dict, "short")?;

        let long = LiveConfig {
            close_drawdown_pct: extract_value(long_dict, "close_drawdown_pct")?,
            close_threshold_pct: extract_value(long_dict, "close_threshold_pct")?,
            ema_span_0: extract_value(long_dict, "ema_span_0")?,
            ema_span_1: extract_value(long_dict, "ema_span_1")?,
            initial_ema_dist: extract_value(long_dict, "initial_ema_dist")?,
            initial_qty_pct: extract_value(long_dict, "initial_qty_pct")?,
            n_positions: extract_value(long_dict, "n_positions")?,
            reentry_ddown_factor: extract_value(long_dict, "reentry_ddown_factor")?,
            reentry_exposure_weighting: extract_value(long_dict, "reentry_exposure_weighting")?,
            reentry_spacing_factor: extract_value(long_dict, "reentry_spacing_factor")?,
            total_wallet_exposure_limit: extract_value(long_dict, "total_wallet_exposure_limit")?,
            unstuck_close_pct: extract_value(long_dict, "unstuck_close_pct")?,
            unstuck_ema_dist: extract_value(long_dict, "unstuck_ema_dist")?,
            unstuck_loss_allowance_pct: extract_value(long_dict, "unstuck_loss_allowance_pct")?,
            unstuck_threshold: extract_value(long_dict, "unstuck_threshold")?,
        };
        let short = LiveConfig {
            close_drawdown_pct: extract_value(short_dict, "close_drawdown_pct")?,
            close_threshold_pct: extract_value(short_dict, "close_threshold_pct")?,
            ema_span_0: extract_value(short_dict, "ema_span_0")?,
            ema_span_1: extract_value(short_dict, "ema_span_1")?,
            initial_ema_dist: extract_value(short_dict, "initial_ema_dist")?,
            initial_qty_pct: extract_value(short_dict, "initial_qty_pct")?,
            n_positions: extract_value(short_dict, "n_positions")?,
            reentry_ddown_factor: extract_value(short_dict, "reentry_ddown_factor")?,
            reentry_exposure_weighting: extract_value(short_dict, "reentry_exposure_weighting")?,
            reentry_spacing_factor: extract_value(short_dict, "reentry_spacing_factor")?,
            total_wallet_exposure_limit: extract_value(short_dict, "total_wallet_exposure_limit")?,
            unstuck_close_pct: extract_value(short_dict, "unstuck_close_pct")?,
            unstuck_ema_dist: extract_value(short_dict, "unstuck_ema_dist")?,
            unstuck_loss_allowance_pct: extract_value(short_dict, "unstuck_loss_allowance_pct")?,
            unstuck_threshold: extract_value(short_dict, "unstuck_threshold")?,
        };
        let starting_balance = extract_value(dict, "starting_balance")?;
        let maker_fee = extract_value(dict, "maker_fee")?;
        let c_mults = extract_value(dict, "c_mults")?;
        let symbols = extract_value(dict, "symbols")?;
        let qty_steps = extract_value(dict, "qty_steps")?;
        let price_steps = extract_value(dict, "price_steps")?;
        let min_costs = extract_value(dict, "min_costs")?;
        let min_qtys = extract_value(dict, "min_qtys")?;

        Ok(BacktestConfig {
            long,
            short,
            starting_balance,
            maker_fee,
            c_mults,
            symbols,
            qty_steps,
            price_steps,
            min_costs,
            min_qtys,
        })
    }
}

fn repeat_elements_to_rows(arr: ArrayView1<f64>, n: usize) -> Array2<f64> {
    let rows = arr.len();
    let mut result = Array2::<f64>::zeros((rows, n));
    for i in 0..rows {
        for j in 0..n {
            result[(i, j)] = arr[i];
        }
    }
    result
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
    let mut spans_long_vec = vec![
        spans_long[0],
        spans_long[1],
        (spans_long[0] * spans_long[1]).sqrt(),
    ];
    spans_long_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spans_long = Array1::from_vec(spans_long_vec).mapv(|x| if x < 1.0 { 1.0 } else { x });

    let mut spans_short_vec = vec![
        spans_short[0],
        spans_short[1],
        (spans_short[0] * spans_short[1]).sqrt(),
    ];
    spans_short_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spans_short = Array1::from_vec(spans_short_vec).mapv(|x| if x < 1.0 { 1.0 } else { x });

    let hlcs_first_col2 = hlcs_first.column(2);
    let emas_long = repeat_elements_to_rows(hlcs_first_col2, 3);
    let emas_short = repeat_elements_to_rows(hlcs_first_col2, 3);

    let alphas_long = spans_long.mapv(|x| 2.0 / (x + 1.0));
    let alphas_inv_long = alphas_long.mapv(|x| 1.0 - x);
    let alphas_short = spans_short.mapv(|x| 2.0 / (x + 1.0));
    let alphas_inv_short = alphas_short.mapv(|x| 1.0 - x);

    (
        emas_long,
        emas_short,
        alphas_long,
        alphas_inv_long,
        alphas_short,
        alphas_inv_short,
    )
}

//fn multiply_arrays(arr0: &Array1<f64>, arr1: &Array1<f64>) -> Array2<f64> {
//    let mut result = Array2::<f64>::zeros((arr1.len(), arr0.len()));
//    for i in 0..arr1.len() {
//        for j in 0..arr0.len() {
//            result[(i, j)] = arr1[i] * arr0[j];
//        }
//    }
//    result
//}
//
//fn calc_next_ema_multiple(
//    alphas: &Array1<f64>,
//    alphas_: &Array1<f64>,
//    emas: &Array2<f64>,
//    closes: &Array1<f64>,
//) -> Array2<f64> {
//    multiply_arrays(alphas, closes) + alphas_ * emas
//}

#[pyfunction]
pub fn run_backtest(
    hlcs: PyReadonlyArray3<f64>,
    noisiness_indices: PyReadonlyArray2<i32>,
    config: &Bound<PyAny>, // Use Bound<PyAny> instead of PyDict
) -> PyResult<()> {
    // Use `extract` method to convert `&Bound<PyAny>` to `&PyDict`
    let shape_hlcs = hlcs.shape();
    let shape_noisiness_indices = noisiness_indices.shape();
    let dict: &PyDict = config.extract()?;
    let config = BacktestConfig::new(dict)?;

    let enabled_long = config.long.n_positions > 0;
    let enabled_short = config.short.n_positions > 0;
    let wallet_exposure_limit_long = if enabled_long {
        config.long.total_wallet_exposure_limit / config.long.n_positions as f64
    } else {
        0.0
    };
    let wallet_exposure_limit_short = if enabled_short {
        config.short.total_wallet_exposure_limit / config.short.n_positions as f64
    } else {
        0.0
    };

    let mut positions_long: Vec<Position> = vec![
        Position {
            size: 0.0,
            price: 0.0
        };
        hlcs.shape()[0]
    ];
    let mut positions_short: Vec<Position> = vec![
        Position {
            size: 0.0,
            price: 0.0
        };
        hlcs.shape()[0]
    ];

    let mut has_pos_long: HashSet<i32> = HashSet::new();
    let mut has_pos_short: HashSet<i32> = HashSet::new();
    let mut is_stuck_long: HashSet<i32> = HashSet::new();
    let mut is_stuck_short: HashSet<i32> = HashSet::new();
    let mut active_longs: HashSet<i32> = HashSet::new();
    let mut active_shorts: HashSet<i32> = HashSet::new();

    let mut open_orders_entry_long: Vec<OpenOrder> = Vec::new();
    let mut open_orders_close_long: Vec<OpenOrder> = Vec::new();
    let mut open_orders_entry_short: Vec<OpenOrder> = Vec::new();
    let mut open_orders_close_short: Vec<OpenOrder> = Vec::new();

    let mut fills: Vec<Fill> = Vec::new();
    let mut stats: Vec<Stat> = Vec::new();

    let mut unstuck_order: (i32, i32, OpenOrder) = (
        0, // pside: 0 (long) / 1 (short)
        0, // symbol idx
        OpenOrder {
            qty: 0.0,
            price: 0.0,
            order_type: String::from(""),
        },
    );

    let mut unstuck_closes: Vec<OpenOrder> = vec![OpenOrder {
        qty: 0.0,
        price: 0.0,
        order_type: String::from(""),
    }];

    let mut pnl_cumsum_max: f64 = 0.0;
    let mut pnl_cumsum_running: f64 = 0.0;

    let spans_long = [config.long.ema_span_0, config.long.ema_span_1];
    let spans_short = [config.short.ema_span_0, config.short.ema_span_1];
    let hlcs_array = hlcs.as_array();
    let hlcs_first = hlcs_array.slice(s![0, .., ..]).to_owned();

    let (
        mut emas_long,
        mut emas_short,
        alphas_long,
        alphas_inv_long,
        alphas_short,
        alphas_inv_short,
    ) = prepare_emas_forager(spans_long, spans_short, hlcs_first.view());

    // start python code
    //tmp_emas = [emas_long]
    //for k in range(1, len(hlcs)):
    //    if enabled_long:
    //        # calc emas
    //        emas_long = calc_next_ema_multiple(alphas_long, alphas__long, emas_long, hlcs[k, :, 2])
    //        tmp_emas.append(emas_long)
    //return tmp_emas
    // end python code

    println!("emas_long:\n{:?}", emas_long);
    println!("emas_short:\n{:?}", emas_short);
    println!("alphas_long:\n{:?}", alphas_long);
    println!("alphas_inv_long:\n{:?}", alphas_inv_long);
    println!("alphas_short:\n{:?}", alphas_short);
    println!("alphas_inv_short:\n{:?}", alphas_inv_short);

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
