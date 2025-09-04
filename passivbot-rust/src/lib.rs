mod analysis;
mod backtest;
mod closes;
mod constants;
mod entries;
mod python;
mod trailing_flip;
mod types;
mod utils;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use python::*;
use trailing_flip::*;
use utils::*;

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(calc_auto_unstuck_allowance, m)?)?;
    m.add_function(wrap_pyfunction!(hysteresis_rounding, m)?)?;
    m.add_function(wrap_pyfunction!(calc_min_entry_qty_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_pprice_diff_int, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_trailing_flip, m)?)?;
    m.add_function(wrap_pyfunction!(order_type_id_to_snake, m)?)?;
    m.add_function(wrap_pyfunction!(all_order_types_ids, m)?)?;
    m.add_function(wrap_pyfunction!(order_type_snake_to_id, m)?)?;
    m.add_function(wrap_pyfunction!(get_order_id_type_from_string_alias, m)?)?;

    Ok(())
}
