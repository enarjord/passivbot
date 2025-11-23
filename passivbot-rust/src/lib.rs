mod analysis;
mod backtest;
mod closes;
mod coin_selection;
mod constants;
mod entries;
mod python;
mod risk;
mod trailing;
mod types;
mod utils;

use coin_selection::select_coin_indices_py;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use python::*;
use utils::*;

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HlcvsBundlePy>()?;
    m.add_function(wrap_pyfunction!(round_, m)?)?;
    m.add_function(wrap_pyfunction!(round_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dn, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dynamic_dn, m)?)?;
    m.add_function(wrap_pyfunction!(calc_diff, m)?)?;
    m.add_function(wrap_pyfunction!(qty_to_cost, m)?)?;
    m.add_function(wrap_pyfunction!(cost_to_qty, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pnl_long, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pnl_short, m)?)?;
    m.add_function(wrap_pyfunction!(calc_wallet_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(calc_new_psize_pprice, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_entry_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_close_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_entries_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_entry_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_close_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_entries_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_closes_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_closes_short_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_twel_enforcer_orders_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(calc_auto_unstuck_allowance, m)?)?;
    m.add_function(wrap_pyfunction!(hysteresis, m)?)?;
    m.add_function(wrap_pyfunction!(calc_min_entry_qty_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pprice_diff_int, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pside_price_diff_int, m)?)?;
    m.add_function(wrap_pyfunction!(calc_price_diff_pside_int, m)?)?;
    m.add_function(wrap_pyfunction!(calc_order_price_diff, m)?)?;
    m.add_function(wrap_pyfunction!(order_type_id_to_snake, m)?)?;
    m.add_function(wrap_pyfunction!(all_order_types_ids, m)?)?;
    m.add_function(wrap_pyfunction!(order_type_snake_to_id, m)?)?;
    m.add_function(wrap_pyfunction!(get_order_id_type_from_string_alias, m)?)?;
    m.add_function(wrap_pyfunction!(gate_entries_by_twel_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_unstucking_close_py, m)?)?;
    m.add_function(wrap_pyfunction!(trailing_bundle_default_py, m)?)?;
    m.add_function(wrap_pyfunction!(update_trailing_bundle_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_coin_indices_py, m)?)?;

    Ok(())
}
