//! One nominal-budget-relative position exposure moment for the core observer.

pub fn position_moment(e: f64, wel: f64, limit: f64) -> f64 {
    if e == 0.0 {
        return 0.0;
    }
    if e.is_finite() && e > 0.0 && wel.is_finite() && wel > 0.0 && limit.is_finite() && limit > 0.0
    {
        e * (e / wel) / limit
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_moments_use_individual_budgets_and_scale_with_both_budgets() {
        let sum = position_moment(0.3, 0.4, 2.0) + position_moment(0.2, 0.8, 2.0);
        assert!((sum - (0.09 / 0.8 + 0.04 / 1.6)).abs() < 1e-12);
        for wel in [0.5, 0.25] {
            let base = position_moment(0.5, wel, 1.0);
            let scaled = position_moment(0.75, wel * 1.5, 1.5);
            assert!((base - scaled).abs() < 1e-12);
        }
    }

    #[test]
    fn flat_is_zero_and_invalid_budgets_cannot_hide_positive_load() {
        assert_eq!(position_moment(0.0, f64::NAN, 0.0), 0.0);
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(position_moment(0.5, invalid, 1.0).is_nan());
            assert!(position_moment(0.5, 0.5, invalid).is_nan());
        }
    }
}
