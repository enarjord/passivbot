mod backtest;
mod grids;
mod utils;

use backtest::*;
use grids::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use utils::*;

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(round_, m)?)?;
    m.add_function(wrap_pyfunction!(round_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dn, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(cost_to_qty, m)?)?;
    m.add_function(wrap_pyfunction!(qty_to_cost, m)?)?;
    m.add_function(wrap_pyfunction!(calc_min_entry_qty, m)?)?;

    Ok(())
}
