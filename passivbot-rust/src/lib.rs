mod backtest;
mod grids;
mod python;
mod utils;

use backtest::*;
use grids::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use python::*;
use utils::*;

/// A Python module implemented in Rust.
#[pymodule]
fn passivbot_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(round_, m)?)?;
    m.add_function(wrap_pyfunction!(round_up, m)?)?;
    m.add_function(wrap_pyfunction!(round_dn, m)?)?;

    Ok(())
}
