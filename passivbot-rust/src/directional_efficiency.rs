//! Causal, signed directional efficiency of a completed close window.
//! Zero means no net movement; +1/-1 means monotonic up/down movement.
use crate::types::BotParams;
use pyo3::prelude::*;

pub const DEFAULT_LOOKBACK: f64 = 60.0;
pub const MAX_LOOKBACK: f64 = 10_080.0;

pub fn default_lookback() -> f64 {
    DEFAULT_LOOKBACK
}

pub fn validate_params(params: &BotParams) -> Result<(), &'static str> {
    for window in [
        params.forager_directional_efficiency_lookback_minutes,
        params.risk_directional_efficiency_lookback_minutes,
    ] {
        if !window.is_finite() || window.fract() != 0.0 || !(1.0..=MAX_LOOKBACK).contains(&window) {
            return Err("directional efficiency lookback must be an integer in [1, 10080]");
        }
    }
    if !params.forager_directional_efficiency_penalty.is_finite()
        || !(0.0..=1.0).contains(&params.forager_directional_efficiency_penalty)
    {
        return Err("directional efficiency penalty must be in [0, 1]");
    }
    if !params
        .risk_directional_efficiency_cooldown_minutes
        .is_finite()
        || !(0.0..=MAX_LOOKBACK).contains(&params.risk_directional_efficiency_cooldown_minutes)
    {
        return Err("directional efficiency cooldown must be in [0, 10080]");
    }
    Ok(())
}

pub fn required_windows(params: &BotParams) -> Vec<usize> {
    let mut windows = Vec::new();
    if params.forager_directional_efficiency_penalty > 0.0 {
        windows.push(params.forager_directional_efficiency_lookback_minutes as usize);
    }
    if params.risk_directional_efficiency_cooldown_minutes > 0.0 {
        windows.push(params.risk_directional_efficiency_lookback_minutes as usize);
    }
    windows.sort_unstable();
    windows.dedup();
    windows
}

pub fn signed_efficiency(closes: &[f64]) -> Result<f64, &'static str> {
    if closes.len() < 2 || closes.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Err("directional efficiency requires at least two finite positive closes");
    }
    let mut previous = closes[0].ln();
    let first = previous;
    let mut path = 0.0;
    for close in &closes[1..] {
        let current = close.ln();
        path += (current - previous).abs();
        previous = current;
    }
    Ok(if path == 0.0 {
        0.0
    } else {
        ((previous - first) / path).clamp(-1.0, 1.0)
    })
}

#[pyfunction]
pub fn calc_directional_efficiency(closes: Vec<f64>) -> PyResult<f64> {
    signed_efficiency(&closes).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_paths() {
        assert_eq!(signed_efficiency(&[100.0, 101.0, 105.0]).unwrap(), 1.0);
        assert_eq!(signed_efficiency(&[105.0, 101.0, 100.0]).unwrap(), -1.0);
        assert_eq!(signed_efficiency(&[100.0, 105.0, 100.0]).unwrap(), 0.0);
        assert_eq!(signed_efficiency(&[100.0; 4]).unwrap(), 0.0);
        let x = signed_efficiency(&[100.0, 110.0, 105.0]).unwrap();
        assert!(x > 0.0 && x < 1.0);
    }

    #[test]
    fn invalid_history_is_not_a_neutral_signal() {
        for closes in [
            vec![],
            vec![1.0],
            vec![0.0, 1.0],
            vec![1.0, f64::NAN],
            vec![1.0, f64::INFINITY],
        ] {
            assert!(signed_efficiency(&closes).is_err());
        }
    }
}
