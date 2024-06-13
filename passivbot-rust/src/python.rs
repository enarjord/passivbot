use crate::backtest::{Backtest, Fill, Stat};
use crate::grids::{
    calc_next_close, calc_next_entry, calc_next_grid_close_long, calc_next_grid_entry_long,
};
use crate::trailing::{calc_trailing_close_long, calc_trailing_entry_long};
use crate::types::{BotParams, EMABands, ExchangeParams, Order, OrderBook, Position, StateParams};
use ndarray::ArrayD;
use numpy::PyArray;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

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
pub fn calc_next_grid_entry_long_py(
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

    let order = calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
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
pub fn calc_next_entry_py(
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

    let next_entry = calc_next_entry(
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
pub fn calc_next_close_py(
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

    let next_entry = calc_next_close(
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
