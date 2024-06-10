use crate::grids::{
    calc_next_entry, calc_next_grid_entry_long, BotParams, EMABands, ExchangeParams, OrderBook,
    Position, StateParams,
};
use crate::trailing::{calc_trailing_close_long, calc_trailing_entry_long};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
pub fn calc_trailing_close_long_py(
    price_step: f64,
    order_book_ask: f64,
    highest_since_position_open: f64,
    lowest_since_highest: f64,
    close_trailing_threshold_pct: f64,
    close_trailing_drawdown_pct: f64,
    position_size: f64,
    position_price: f64,
) -> (f64, f64, String) {
    // these aren't used by calc_trailing_close_long()
    let qty_step = 0.0;
    let min_qty = 0.0;
    let min_cost = 0.0;
    let c_mult = 0.0;
    let balance = 0.0;
    let order_book_bid = 0.0;
    let ema_bands_upper = 0.0;
    let ema_bands_lower = 0.0;
    let close_grid_markup_range = 0.0;
    let close_grid_min_markup = 0.0;
    let close_grid_n_orders = 0.0;
    let close_trailing_grid_ratio = 0.0;
    let entry_grid_double_down_factor = 0.0;
    let entry_grid_spacing_weight = 0.0;
    let entry_grid_spacing_pct = 0.0;
    let entry_initial_ema_dist = 0.0;
    let entry_initial_qty_pct = 0.0;
    let entry_trailing_drawdown_pct = 0.0;
    let entry_trailing_grid_ratio = 0.0;
    let entry_trailing_threshold_pct = 0.0;
    let n_positions = 0;
    let total_wallet_exposure_limit = 0.0;
    let wallet_exposure_limit = 0.0;
    let unstuck_close_pct = 0.0;
    let unstuck_ema_dist = 0.0;
    let unstuck_loss_allowance_pct = 0.0;
    let unstuck_threshold = 0.0;

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
            ask: order_book_ask,
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            lower: ema_bands_lower,
        },
    };

    let bot_params = BotParams {
        close_grid_markup_range,
        close_grid_min_markup,
        close_grid_n_orders,
        close_trailing_drawdown_pct,
        close_trailing_grid_ratio,
        close_trailing_threshold_pct,
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_drawdown_pct,
        entry_trailing_grid_ratio,
        entry_trailing_threshold_pct,
        n_positions,
        total_wallet_exposure_limit,
        wallet_exposure_limit,
        unstuck_close_pct,
        unstuck_ema_dist,
        unstuck_loss_allowance_pct,
        unstuck_threshold,
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
        highest_since_position_open,
        lowest_since_highest,
    );
    (order.qty, order.price, order.description)
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
    // these aren't used by calc_next_grid_entry_long()
    let order_book_ask = 0.0;
    let ema_bands_upper = 0.0;
    let close_grid_markup_range = 0.0;
    let close_grid_min_markup = 0.0;
    let close_grid_n_orders = 0.0;
    let close_trailing_drawdown_pct = 0.0;
    let close_trailing_grid_ratio = 0.0;
    let close_trailing_threshold_pct = 0.0;
    let entry_trailing_drawdown_pct = 0.0;
    let entry_trailing_grid_ratio = 0.0;
    let entry_trailing_threshold_pct = 0.0;
    let n_positions = 0;
    let total_wallet_exposure_limit = 0.0;
    let unstuck_close_pct = 0.0;
    let unstuck_ema_dist = 0.0;
    let unstuck_loss_allowance_pct = 0.0;
    let unstuck_threshold = 0.0;

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
            ask: order_book_ask,
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            lower: ema_bands_lower,
        },
    };

    let bot_params = BotParams {
        close_grid_markup_range,
        close_grid_min_markup,
        close_grid_n_orders,
        close_trailing_drawdown_pct,
        close_trailing_grid_ratio,
        close_trailing_threshold_pct,
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_drawdown_pct,
        entry_trailing_grid_ratio,
        entry_trailing_threshold_pct,
        n_positions,
        total_wallet_exposure_limit,
        wallet_exposure_limit,
        unstuck_close_pct,
        unstuck_ema_dist,
        unstuck_loss_allowance_pct,
        unstuck_threshold,
    };

    let position = Position {
        size: position_size,
        price: position_price,
    };

    let order = calc_next_grid_entry_long(&exchange_params, &state_params, &bot_params, &position);
    (order.qty, order.price, order.description)
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
    lowest_since_position_open: f64,
    highest_since_lowest: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_drawdown_pct: f64,
) -> (f64, f64, String) {
    // these aren't used by calc_trailing_entry_long()
    let order_book_ask = 0.0;
    let ema_bands_upper = 0.0;
    let ema_bands_lower = 0.0;
    let entry_grid_spacing_pct = 0.0;
    let entry_grid_spacing_weight = 0.0;
    let entry_trailing_grid_ratio = 0.0;
    let entry_initial_ema_dist = 0.0;
    let close_grid_markup_range = 0.0;
    let close_grid_min_markup = 0.0;
    let close_grid_n_orders = 0.0;
    let close_trailing_drawdown_pct = 0.0;
    let close_trailing_grid_ratio = 0.0;
    let close_trailing_threshold_pct = 0.0;
    let n_positions = 0;
    let total_wallet_exposure_limit = 0.0;
    let unstuck_close_pct = 0.0;
    let unstuck_ema_dist = 0.0;
    let unstuck_loss_allowance_pct = 0.0;
    let unstuck_threshold = 0.0;

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
            ask: order_book_ask,
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            lower: ema_bands_lower,
        },
    };

    let bot_params = BotParams {
        close_grid_markup_range,
        close_grid_min_markup,
        close_grid_n_orders,
        close_trailing_drawdown_pct,
        close_trailing_grid_ratio,
        close_trailing_threshold_pct,
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_drawdown_pct,
        entry_trailing_grid_ratio,
        entry_trailing_threshold_pct,
        n_positions,
        total_wallet_exposure_limit,
        wallet_exposure_limit,
        unstuck_close_pct,
        unstuck_ema_dist,
        unstuck_loss_allowance_pct,
        unstuck_threshold,
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
        lowest_since_position_open,
        highest_since_lowest,
    );
    (order.qty, order.price, order.description)
}

#[pyfunction]
pub fn calc_next_entry_py(
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
    lowest_since_position_open: f64,
    highest_since_lowest: f64,
    entry_trailing_threshold_pct: f64,
    entry_trailing_drawdown_pct: f64,
    entry_trailing_grid_ratio: f64,
) -> (f64, f64, String) {
    // these aren't used by calc_next_entry()
    let order_book_ask = 0.0;
    let ema_bands_upper = 0.0;
    let close_grid_markup_range = 0.0;
    let close_grid_min_markup = 0.0;
    let close_grid_n_orders = 0.0;
    let close_trailing_drawdown_pct = 0.0;
    let close_trailing_grid_ratio = 0.0;
    let close_trailing_threshold_pct = 0.0;
    let n_positions = 0;
    let total_wallet_exposure_limit = 0.0;
    let unstuck_close_pct = 0.0;
    let unstuck_ema_dist = 0.0;
    let unstuck_loss_allowance_pct = 0.0;
    let unstuck_threshold = 0.0;

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
            ask: order_book_ask,
        },
        ema_bands: EMABands {
            upper: ema_bands_upper,
            lower: ema_bands_lower,
        },
    };

    let bot_params = BotParams {
        close_grid_markup_range,
        close_grid_min_markup,
        close_grid_n_orders,
        close_trailing_drawdown_pct,
        close_trailing_grid_ratio,
        close_trailing_threshold_pct,
        entry_grid_double_down_factor,
        entry_grid_spacing_weight,
        entry_grid_spacing_pct,
        entry_initial_ema_dist,
        entry_initial_qty_pct,
        entry_trailing_drawdown_pct,
        entry_trailing_grid_ratio,
        entry_trailing_threshold_pct,
        n_positions,
        total_wallet_exposure_limit,
        wallet_exposure_limit,
        unstuck_close_pct,
        unstuck_ema_dist,
        unstuck_loss_allowance_pct,
        unstuck_threshold,
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
        lowest_since_position_open,
        highest_since_lowest,
    );

    (next_entry.qty, next_entry.price, next_entry.description)
}
