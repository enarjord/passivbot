mod backtest;
mod grids;
mod python;
mod trailing;
mod utils;

use backtest::*;
use grids::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use python::*;
use trailing::*;
use utils::*;

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(round_, m)?)?;
    m.add_function(wrap_pyfunction!(round_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dn, m)?)?;
    m.add_function(wrap_pyfunction!(calc_diff, m)?)?;
    m.add_function(wrap_pyfunction!(qty_to_cost, m)?)?;
    m.add_function(wrap_pyfunction!(cost_to_qty, m)?)?;
    m.add_function(wrap_pyfunction!(calc_new_psize_pprice, m)?)?;
    m.add_function(wrap_pyfunction!(calc_next_grid_entry_long_py, m)?)?;
    m.add_function(wrap_pyfunction!(calc_trailing_entry_long_py, m)?)?;

    Ok(())
}
