use crate::types::TrailingPriceBundle;

#[inline]
pub fn reset_trailing_bundle(bundle: &mut TrailingPriceBundle) {
    *bundle = TrailingPriceBundle::default();
}

#[inline]
pub fn update_trailing_bundle_with_candle(
    bundle: &mut TrailingPriceBundle,
    high: f64,
    low: f64,
    close: f64,
) {
    if !high.is_finite() || !low.is_finite() || !close.is_finite() {
        return;
    }

    if low < bundle.min_since_open {
        bundle.min_since_open = low;
        bundle.max_since_min = close;
    } else {
        bundle.max_since_min = bundle.max_since_min.max(high);
    }

    if high > bundle.max_since_open {
        bundle.max_since_open = high;
        bundle.min_since_max = close;
    } else {
        bundle.min_since_max = bundle.min_since_max.min(low);
    }
}

pub fn update_trailing_bundle_sequence(
    bundle: &mut TrailingPriceBundle,
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) {
    for ((&high, &low), &close) in highs.iter().zip(lows.iter()).zip(closes.iter()) {
        update_trailing_bundle_with_candle(bundle, high, low, close);
    }
}

#[inline]
pub fn trailing_bundle_to_tuple(bundle: &TrailingPriceBundle) -> (f64, f64, f64, f64) {
    (
        bundle.min_since_open,
        bundle.max_since_min,
        bundle.max_since_open,
        bundle.min_since_max,
    )
}

#[inline]
pub fn tuple_to_trailing_bundle(values: (f64, f64, f64, f64)) -> TrailingPriceBundle {
    TrailingPriceBundle {
        min_since_open: values.0,
        max_since_min: values.1,
        max_since_open: values.2,
        min_since_max: values.3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_bundle_updates_correctly() {
        let mut bundle = TrailingPriceBundle::default();
        let highs = [11.0, 12.0, 9.0, 15.0];
        let lows = [9.0, 10.0, 7.5, 13.0];
        let closes = [10.0, 11.0, 8.0, 14.0];

        update_trailing_bundle_sequence(&mut bundle, &highs, &lows, &closes);

        assert_eq!(bundle.min_since_open, 7.5);
        assert_eq!(bundle.max_since_min, 15.0);
        assert_eq!(bundle.max_since_open, 15.0);
        assert_eq!(bundle.min_since_max, 14.0);
    }

    #[test]
    fn trailing_bundle_resets_to_default() {
        let mut bundle = TrailingPriceBundle::default();
        update_trailing_bundle_with_candle(&mut bundle, 11.0, 9.0, 10.0);
        assert_ne!(bundle.min_since_open, f64::MAX);
        reset_trailing_bundle(&mut bundle);
        assert_eq!(
            bundle.min_since_open,
            TrailingPriceBundle::default().min_since_open
        );
        assert_eq!(
            bundle.max_since_min,
            TrailingPriceBundle::default().max_since_min
        );
        assert_eq!(
            bundle.max_since_open,
            TrailingPriceBundle::default().max_since_open
        );
        assert_eq!(
            bundle.min_since_max,
            TrailingPriceBundle::default().min_since_max
        );
    }

    #[test]
    fn trailing_bundle_struct_is_unaffected_by_non_finite() {
        let mut bundle = TrailingPriceBundle::default();
        update_trailing_bundle_with_candle(&mut bundle, 11.0, 9.0, 10.0);
        let expected = TrailingPriceBundle {
            min_since_open: 9.0,
            max_since_min: 10.0,
            max_since_open: 11.0,
            min_since_max: 10.0,
        };
        assert_eq!(bundle.min_since_open, expected.min_since_open);
        assert_eq!(bundle.max_since_min, expected.max_since_min);
        assert_eq!(bundle.max_since_open, expected.max_since_open);
        assert_eq!(bundle.min_since_max, expected.min_since_max);

        // These non-finite inputs should be ignored
        update_trailing_bundle_with_candle(&mut bundle, f64::NAN, 10.0, 10.0);
        assert_eq!(bundle.min_since_open, expected.min_since_open);
        assert_eq!(bundle.max_since_min, expected.max_since_min);
        assert_eq!(bundle.max_since_open, expected.max_since_open);
        assert_eq!(bundle.min_since_max, expected.min_since_max);

        update_trailing_bundle_with_candle(&mut bundle, 10.0, f64::INFINITY, 10.5);
        assert_eq!(bundle.min_since_open, expected.min_since_open);
        assert_eq!(bundle.max_since_min, expected.max_since_min);
        assert_eq!(bundle.max_since_open, expected.max_since_open);
        assert_eq!(bundle.min_since_max, expected.min_since_max);

        update_trailing_bundle_with_candle(&mut bundle, 10.0, 9.0, f64::NAN);
        assert_eq!(bundle.min_since_open, expected.min_since_open);
        assert_eq!(bundle.max_since_min, expected.max_since_min);
        assert_eq!(bundle.max_since_open, expected.max_since_open);
        assert_eq!(bundle.min_since_max, expected.min_since_max);
    }

    #[test]
    fn trailing_bundle_sequence_matches_incremental_updates() {
        let highs = [11.0, 9.5, 12.0, 11.5];
        let lows = [9.0, 8.0, 10.5, 10.0];
        let closes = [10.0, 9.0, 11.0, 10.2];

        let mut seq_bundle = TrailingPriceBundle::default();
        update_trailing_bundle_sequence(&mut seq_bundle, &highs, &lows, &closes);

        let mut incremental_bundle = TrailingPriceBundle::default();
        for i in 0..highs.len() {
            update_trailing_bundle_with_candle(
                &mut incremental_bundle,
                highs[i],
                lows[i],
                closes[i],
            );
        }

        assert_eq!(seq_bundle.min_since_open, incremental_bundle.min_since_open);
        assert_eq!(seq_bundle.max_since_min, incremental_bundle.max_since_min);
        assert_eq!(seq_bundle.max_since_open, incremental_bundle.max_since_open);
        assert_eq!(seq_bundle.min_since_max, incremental_bundle.min_since_max);
    }

    #[test]
    fn trailing_bundle_handles_mixed_validity_sequence() {
        let highs = [11.0, f64::NAN, 12.0];
        let lows = [9.0, 8.0, f64::INFINITY];
        let closes = [10.0, 9.5, 11.0];

        let mut bundle = TrailingPriceBundle::default();
        update_trailing_bundle_sequence(&mut bundle, &highs, &lows, &closes);

        // Only the first candle is valid
        assert_eq!(bundle.min_since_open, 9.0);
        assert_eq!(bundle.max_since_min, 10.0);
        assert_eq!(bundle.max_since_open, 11.0);
        assert_eq!(bundle.min_since_max, 10.0);
    }
}
