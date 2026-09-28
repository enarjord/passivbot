//! Signed RMS directionality from completed one-minute closes.
//! A deterministic 20-span replay makes live, restart and CPU inputs identical.
//! Weights are exponential; this is not a simple moving average.
pub fn warmup_returns(span: f64) -> Result<usize, String> {
    if !span.is_finite() || !(1.0..=100_000.0).contains(&span) {
        return Err("unilateralness span must be finite and between 1 and 100000".into());
    }
    Ok((span * 20.0).ceil() as usize)
}

pub fn signed_rms(closes: &[f64], span: f64) -> Result<f64, String> {
    let n = warmup_returns(span)?;
    if closes.len() < n + 1 {
        return Err("incomplete unilateralness warmup".into());
    }
    let closes = &closes[closes.len() - n - 1..];
    let alpha = 2.0 / (span + 1.0);
    let mut mean = 0.0;
    let mut square = 0.0;
    for pair in closes.windows(2) {
        if pair.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("invalid unilateralness close".into());
        }
        let r = pair[1].ln() - pair[0].ln();
        mean += alpha * (r - mean);
        square += alpha * (r * r - square);
    }
    Ok(if square == 0.0 {
        0.0
    } else {
        (mean / square.sqrt()).clamp(-1.0, 1.0)
    })
}

#[pyo3::prelude::pyfunction]
pub fn calc_signed_unilateralness(closes: Vec<f64>, span: f64) -> pyo3::PyResult<f64> {
    signed_rms(&closes, span).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direction_flat_decay_and_fractional_span() {
        let span = 10.5;
        let n = warmup_returns(span).unwrap();
        let up: Vec<f64> = (0..=n).map(|i| (i as f64 * 0.001).exp()).collect();
        let down: Vec<f64> = up.iter().map(|v| 1.0 / v).collect();
        let u = signed_rms(&up, span).unwrap();
        assert!((u - 1.0).abs() < 1e-12);
        assert!((signed_rms(&down, span).unwrap() + u).abs() < 1e-12);
        let mut flat = up.clone();
        flat.extend(vec![*up.last().unwrap(); 20]);
        assert!(
            (signed_rms(&flat, span).unwrap() - u * (1.0 - 2.0 / (span + 1.0)).powf(10.0)).abs()
                < 1e-12
        );
        assert_eq!(signed_rms(&vec![100.0; n + 1], span).unwrap(), 0.0);
        assert!(signed_rms(&up[..n], span).is_err());
    }
}
