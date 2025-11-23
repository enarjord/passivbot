use crate::analysis::analyze_backtest_pair;
use crate::backtest::Backtest;
use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::constants::{LONG, SHORT};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_next_entry_long, calc_next_entry_short,
};
use crate::risk::{
    calc_twel_enforcer_actions, calc_unstucking_action, gate_entries_by_twel, GateEntriesCandidate,
    GateEntriesDecision, GateEntriesPosition, TwelEnforcerInputPosition, UnstuckPositionInput,
};
use crate::trailing::{
    trailing_bundle_to_tuple, tuple_to_trailing_bundle, update_trailing_bundle_sequence,
};
use crate::types::OrderType;
use crate::types::{
    BacktestParams, BotParams, BotParamsPair, CoinMeta, EMABands, ExchangeParams, HlcvsBundle,
    HlcvsMeta, OrderBook, Position, StateParams, TrailingPriceBundle,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::PyObject;
use serde::Serialize;
use std::str::FromStr;

type BacktestPyResult = (PyObject, PyObject, Py<PyDict>, Py<PyDict>);

#[pyclass(name = "HlcvsBundle", module = "passivbot_rust", unsendable)]
pub struct HlcvsBundlePy {
    pub inner: HlcvsBundle,
}

#[pymethods]
impl HlcvsBundlePy {
    #[new]
    #[pyo3(signature = (hlcvs, btc_usd, timestamps, meta))]
    pub fn new(
        hlcvs: Py<PyArray3<f64>>,
        btc_usd: Py<PyArray1<f64>>,
        timestamps: Py<PyArray1<i64>>,
        meta: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let parsed_meta = hlcvs_meta_from_py(meta)?;
        let bundle = HlcvsBundle {
            hlcvs,
            btc_usd,
            timestamps,
            meta: parsed_meta,
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

    Ok(HlcvsMeta {
        requested_start_timestamp_ms: requested,
        effective_start_timestamp_ms: effective,
        warmup_minutes_requested: warm_req,
        warmup_minutes_provided: warm_prov,
        coins,
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
        let risk_we_excess_allowance_pct = dict
            .get_item("risk_we_excess_allowance_pct")?
            .ok_or_else(|| {
                PyValueError::new_err("position missing 'risk_we_excess_allowance_pct'")
            })?
            .extract::<f64>()?;
        let unstuck_threshold = dict
            .get_item("unstuck_threshold")?
            .ok_or_else(|| PyValueError::new_err("position missing 'unstuck_threshold'"))?
            .extract::<f64>()?;
        let unstuck_close_pct = dict
            .get_item("unstuck_close_pct")?
            .ok_or_else(|| PyValueError::new_err("position missing 'unstuck_close_pct'"))?
            .extract::<f64>()?;
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
            risk_we_excess_allowance_pct,
            unstuck_threshold,
            unstuck_close_pct,
            unstuck_ema_dist,
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
    exchange_params_list: &Bound<'_, PyAny>,
    backtest_params_dict: &Bound<'_, PyDict>,
) -> PyResult<BacktestPyResult> {
    run_backtest_core(
        hlcvs,
        btc_usd,
        bot_params,
        exchange_params_list,
        backtest_params_dict,
    )
}

#[pyfunction]
pub fn run_backtest_bundle(
    bundle: &HlcvsBundlePy,
    bot_params: &Bound<'_, PyAny>,
    exchange_params_list: &Bound<'_, PyAny>,
    backtest_params_dict: &Bound<'_, PyDict>,
) -> PyResult<BacktestPyResult> {
    let py = bot_params.py();
    let hlcvs = bundle.inner.hlcvs.bind(py).readonly();
    let btc = bundle.inner.btc_usd.bind(py).readonly();
    run_backtest_core(
        hlcvs,
        btc,
        bot_params,
        exchange_params_list,
        backtest_params_dict,
    )
}

fn run_backtest_core<'py>(
    hlcvs: PyReadonlyArray3<'py, f64>,
    btc_usd: PyReadonlyArray1<'py, f64>,
    bot_params: &Bound<'py, PyAny>,
    exchange_params_list: &Bound<'py, PyAny>,
    backtest_params_dict: &Bound<'py, PyDict>,
) -> PyResult<BacktestPyResult> {
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
        bot_params_vec.push(bot_params_pair_from_dict(dict)?);
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

    let backtest_params = backtest_params_from_dict(backtest_params_dict.as_gil_ref())?;
    let metrics_only = backtest_params.metrics_only;
    let mut backtest = Backtest::new(
        hlcvs_rust,
        btc_usd_rust,
        bot_params_vec,
        exchange_params,
        &backtest_params,
    );

    // Run the backtest and process results
    Python::with_gil(|py| {
        let (fills, equities) = backtest.run();
        let (entry_pct_long, entry_pct_short) = backtest.initial_entry_balance_pct();
        let (mut analysis_usd, mut analysis_btc) = analyze_backtest_pair(
            &fills,
            &equities,
            backtest.balance.use_btc_collateral,
            &backtest.total_wallet_exposures,
        );
        analysis_usd.entry_initial_balance_pct_long = entry_pct_long;
        analysis_usd.entry_initial_balance_pct_short = entry_pct_short;
        analysis_btc.entry_initial_balance_pct_long = entry_pct_long;
        analysis_btc.entry_initial_balance_pct_short = entry_pct_short;

        // Create a dictionary to store analysis results using a more concise approach
        let py_analysis_usd = struct_to_py_dict(py, &analysis_usd)?;
        let py_analysis_btc = struct_to_py_dict(py, &analysis_btc)?;
        if metrics_only {
            return Ok((
                py.None().into_py(py),
                py.None().into_py(py),
                py_analysis_usd,
                py_analysis_btc,
            ));
        }
        let mut py_fills = Array2::from_elem((fills.len(), 16), py.None());
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
            py_fills[(i, 14)] = fill.wallet_exposure.into_py(py);
            py_fills[(i, 15)] = fill.total_wallet_exposure.into_py(py);
        }

        let equities_array =
            Array2::from_shape_fn((equities.timestamps_ms.len(), 3), |(i, j)| match j {
                0 => equities.timestamps_ms[i] as f64,
                1 => equities.usd_total_equity[i],
                2 => equities.btc_total_equity[i],
                _ => 0.0,
            })
            .into_pyarray_bound(py)
            .unbind();
        let fills_array = py_fills.into_pyarray_bound(py).unbind();
        Ok((
            fills_array.into_py(py),
            equities_array.into_py(py),
            py_analysis_usd,
            py_analysis_btc,
        ))
    })
}

fn struct_to_py_dict<T: Serialize + ?Sized>(py: Python<'_>, obj: &T) -> PyResult<Py<PyDict>> {
    // Convert struct to JSON string
    let json_str = serde_json::to_string(obj).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialization error: {}", e))
    })?;

    // Use Python's json module to convert to a Python dict
    let json = py.import_bound("json")?;
    let py_obj_any = json.call_method1("loads", (json_str,))?.unbind();
    let py_obj_bound = py_obj_any.bind(py);

    // Convert to PyDict
    let py_dict_bound = py_obj_bound.downcast::<PyDict>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("Failed to convert to Python dict")
    })?;
    Ok(py_dict_bound.clone().unbind())
}

fn backtest_params_from_dict(dict: &PyDict) -> PyResult<BacktestParams> {
    Ok(BacktestParams {
        starting_balance: extract_value(dict, "starting_balance").unwrap_or_default(),
        maker_fee: extract_value(dict, "maker_fee").unwrap_or_default(),
        coins: extract_value(dict, "coins").unwrap_or_default(),
        active_coin_indices: dict
            .get_item("active_coin_indices")?
            .map(|item| item.extract::<Vec<usize>>())
            .transpose()?,
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
        btc_collateral_cap: extract_value(dict, "btc_collateral_cap").unwrap_or(0.0f64),
        btc_collateral_ltv_cap: match dict.get_item("btc_collateral_ltv_cap")? {
            Some(item) if !item.is_none() => Some(item.extract::<f64>()?),
            _ => None,
        },
        metrics_only: dict
            .get_item("metrics_only")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        filter_by_min_effective_cost: dict
            .get_item("filter_by_min_effective_cost")?
            .map(|item| item.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
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

fn extract_grid_spacing_we_weight(dict: &PyDict) -> PyResult<f64> {
    if let Some(obj) = dict.get_item("entry_grid_spacing_we_weight")? {
        obj.extract::<f64>()
    } else {
        extract_value(dict, "entry_grid_spacing_we_weight")
    }
}

fn bot_params_from_dict(dict: &PyDict) -> PyResult<BotParams> {
    let risk_wel_enforcer_threshold = match dict.get_item("risk_wel_enforcer_threshold")? {
        Some(item) => item.extract::<f64>()?,
        None => 1.0,
    };
    let risk_twel_enforcer_threshold = match dict.get_item("risk_twel_enforcer_threshold")? {
        Some(item) => item.extract::<f64>()?,
        None => 1.0,
    };
    let risk_we_excess_allowance_pct: f64 = extract_value(dict, "risk_we_excess_allowance_pct")?;
    let total_wallet_exposure_limit: f64 = extract_value(dict, "total_wallet_exposure_limit")?;
    let wallet_exposure_limit_raw: f64 = extract_value(dict, "wallet_exposure_limit")?;
    let n_positions_float: f64 = extract_value(dict, "n_positions")?;
    let n_positions = n_positions_float.round() as usize;
    let wallet_exposure_limit = if wallet_exposure_limit_raw < 0.0 {
        wallet_exposure_limit_raw
    } else if wallet_exposure_limit_raw > 0.0 {
        wallet_exposure_limit_raw
    } else if n_positions > 0 {
        total_wallet_exposure_limit / n_positions as f64
    } else {
        0.0
    };

    Ok(BotParams {
        close_grid_markup_end: extract_value(dict, "close_grid_markup_end")?,
        close_grid_markup_start: extract_value(dict, "close_grid_markup_start")?,
        close_grid_qty_pct: extract_value(dict, "close_grid_qty_pct")?,
        close_trailing_retracement_pct: extract_value(dict, "close_trailing_retracement_pct")?,
        close_trailing_grid_ratio: extract_value(dict, "close_trailing_grid_ratio")?,
        close_trailing_qty_pct: extract_value(dict, "close_trailing_qty_pct")?,
        close_trailing_threshold_pct: extract_value(dict, "close_trailing_threshold_pct")?,
        entry_grid_double_down_factor: extract_value(dict, "entry_grid_double_down_factor")?,
        entry_grid_spacing_volatility_weight: extract_value(
            dict,
            "entry_grid_spacing_volatility_weight",
        )?,
        entry_grid_spacing_we_weight: extract_grid_spacing_we_weight(dict)?,
        entry_grid_spacing_pct: extract_value(dict, "entry_grid_spacing_pct")?,
        entry_volatility_ema_span_hours: extract_value_with_fallback(
            dict,
            "entry_volatility_ema_span_hours",
            "entry_log_range_ema_span_hours",
        )?,
        entry_initial_ema_dist: extract_value(dict, "entry_initial_ema_dist")?,
        entry_initial_qty_pct: extract_value(dict, "entry_initial_qty_pct")?,
        entry_trailing_double_down_factor: extract_value(
            dict,
            "entry_trailing_double_down_factor",
        )?,
        entry_trailing_retracement_pct: extract_value(dict, "entry_trailing_retracement_pct")?,
        entry_trailing_retracement_we_weight: extract_value(
            dict,
            "entry_trailing_retracement_we_weight",
        )?,
        entry_trailing_retracement_volatility_weight: extract_value(
            dict,
            "entry_trailing_retracement_volatility_weight",
        )?,
        entry_trailing_grid_ratio: extract_value(dict, "entry_trailing_grid_ratio")?,
        entry_trailing_threshold_pct: extract_value(dict, "entry_trailing_threshold_pct")?,
        entry_trailing_threshold_we_weight: extract_value(
            dict,
            "entry_trailing_threshold_we_weight",
        )?,
        entry_trailing_threshold_volatility_weight: extract_value(
            dict,
            "entry_trailing_threshold_volatility_weight",
        )?,
        filter_volatility_ema_span: extract_value_with_fallback(
            dict,
            "filter_volatility_ema_span",
            "filter_log_range_ema_span",
        )?,
        filter_volume_ema_span: extract_value(dict, "filter_volume_ema_span")?,
        filter_volume_drop_pct: extract_value(dict, "filter_volume_drop_pct")?,
        filter_volatility_drop_pct: extract_value(dict, "filter_volatility_drop_pct")?,
        ema_span_0: extract_value(dict, "ema_span_0")?,
        ema_span_1: extract_value(dict, "ema_span_1")?,
        n_positions,
        total_wallet_exposure_limit,
        wallet_exposure_limit,
        risk_wel_enforcer_threshold,
        risk_twel_enforcer_threshold,
        risk_we_excess_allowance_pct,
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

#[pyfunction]
pub fn calc_next_entry_long_py(
    qty_step: f64,
    price_step: f64,
    min_qty: f64,
    min_cost: f64,
    c_mult: f64,
    entry_grid_double_down_factor: f64,
    entry_grid_spacing_volatility_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_retracement_we_weight: f64,
    entry_trailing_retracement_volatility_weight: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_threshold_we_weight: f64,
    entry_trailing_threshold_volatility_weight: f64,
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
        entry_grid_spacing_volatility_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_retracement_we_weight,
        entry_trailing_retracement_volatility_weight,
        entry_trailing_threshold_pct,
        entry_trailing_threshold_we_weight,
        entry_trailing_threshold_volatility_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
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
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
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
    entry_grid_spacing_volatility_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_retracement_we_weight: f64,
    entry_trailing_retracement_volatility_weight: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_threshold_we_weight: f64,
    entry_trailing_threshold_volatility_weight: f64,
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
        entry_grid_spacing_volatility_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_retracement_we_weight,
        entry_trailing_retracement_volatility_weight,
        entry_trailing_threshold_pct,
        entry_trailing_threshold_we_weight,
        entry_trailing_threshold_volatility_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
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
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
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
    entry_grid_spacing_volatility_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_retracement_we_weight: f64,
    entry_trailing_retracement_volatility_weight: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_threshold_we_weight: f64,
    entry_trailing_threshold_volatility_weight: f64,
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
        entry_grid_spacing_volatility_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_retracement_we_weight,
        entry_trailing_retracement_volatility_weight,
        entry_trailing_threshold_pct,
        entry_trailing_threshold_we_weight,
        entry_trailing_threshold_volatility_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
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
    entry_grid_spacing_volatility_weight: f64,
    entry_grid_spacing_we_weight: f64,
    entry_grid_spacing_pct: f64,
    entry_initial_ema_dist: f64,
    entry_initial_qty_pct: f64,
    entry_trailing_double_down_factor: f64,
    entry_trailing_grid_ratio: f64,
    entry_trailing_retracement_pct: f64,
    entry_trailing_retracement_we_weight: f64,
    entry_trailing_retracement_volatility_weight: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_threshold_we_weight: f64,
    entry_trailing_threshold_volatility_weight: f64,
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
        entry_grid_spacing_volatility_weight,
        entry_grid_spacing_we_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_double_down_factor,
        entry_trailing_grid_ratio,
        entry_trailing_retracement_pct,
        entry_trailing_retracement_we_weight,
        entry_trailing_retracement_volatility_weight,
        entry_trailing_threshold_pct,
        entry_trailing_threshold_we_weight,
        entry_trailing_threshold_volatility_weight,
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
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
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
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
        wallet_exposure_limit,
        risk_we_excess_allowance_pct,
        risk_wel_enforcer_threshold,
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
pub fn calc_twel_enforcer_orders_py(
    side: &str,
    threshold: f64,
    total_wallet_exposure_limit: f64,
    effective_n_positions: usize,
    balance: f64,
    positions: &Bound<'_, PyList>,
    skip_idx: Option<usize>,
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
            base_wallet_exposure_limit: dict
                .get_item("base_wallet_exposure_limit")?
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "twel enforcer position missing 'base_wallet_exposure_limit'",
                    )
                })?
                .extract::<f64>()?,
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
        });
    }

    let actions = calc_twel_enforcer_actions(
        side_code,
        threshold,
        total_wallet_exposure_limit,
        effective_n_positions,
        balance,
        &parsed_positions,
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
