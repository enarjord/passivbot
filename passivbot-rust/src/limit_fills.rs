//! Backtest limit eligibility, shared with the next-candle ladder-expansion hint.

pub fn validate_buffer(buffer: f64) -> Result<(), &'static str> {
    if buffer.is_finite() && (0.0..1.0).contains(&buffer) {
        Ok(())
    } else {
        Err("limit_order_fill_buffer_pct must be finite and in [0, 1)")
    }
}

#[inline]
pub fn crosses_limit(low: f64, high: f64, qty: f64, price: f64, buffer: f64) -> bool {
    if qty > 0.0 {
        low < price * (1.0 - buffer)
    } else if qty < 0.0 {
        high > price * (1.0 + buffer)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_boundaries_and_original_limit_price_reference() {
        for buffer in [0.0, 0.0001, 0.01, 0.5] {
            let bid_boundary = 100.0 * (1.0 - buffer);
            let ask_boundary = 100.0 * (1.0 + buffer);
            assert!(!crosses_limit(bid_boundary, 200.0, 1.0, 100.0, buffer));
            assert!(!crosses_limit(1.0, ask_boundary, -1.0, 100.0, buffer));
            assert!(crosses_limit(
                bid_boundary - 1e-8,
                200.0,
                1.0,
                100.0,
                buffer
            ));
            assert!(crosses_limit(1.0, ask_boundary + 1e-8, -1.0, 100.0, buffer));
            assert!(!crosses_limit(
                bid_boundary + 1e-8,
                200.0,
                1.0,
                100.0,
                buffer
            ));
            assert!(!crosses_limit(
                1.0,
                ask_boundary - 1e-8,
                -1.0,
                100.0,
                buffer
            ));
        }
        assert!(!crosses_limit(0.0, 200.0, 0.0, 100.0, 0.0));
    }

    #[test]
    fn increasing_buffer_only_removes_fixed_order_fills() {
        for qty in [-1.0, 1.0] {
            for i in -100..100 {
                let excursion = i as f64 * 0.00001;
                let low = 100.0 * (1.0 - excursion);
                let high = 100.0 * (1.0 + excursion);
                let mut missed = false;
                for buffer in [0.0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001] {
                    let filled = crosses_limit(low, high, qty, 100.0, buffer);
                    assert!(!(missed && filled));
                    missed |= !filled;
                }
                assert_eq!(
                    crosses_limit(low, high, qty, 100.0, 0.0),
                    if qty > 0.0 { low < 100.0 } else { high > 100.0 }
                );
            }
        }
    }

    #[test]
    fn invalid_buffers_fail() {
        for value in [-0.001, 1.0, 2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(validate_buffer(value).is_err());
        }
        assert!(validate_buffer(0.0).is_ok());
        assert!(validate_buffer(0.999).is_ok());
    }
}
