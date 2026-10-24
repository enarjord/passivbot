//! Initial-position admission from factual inception and the decision clock.

const DAY_MS: f64 = 86_400_000.0;

pub fn validate_minimum_days(days: f64) -> Result<(), &'static str> {
    if days.is_finite() && days >= 0.0 {
        Ok(())
    } else {
        Err("minimum_coin_age_days must be finite and nonnegative")
    }
}

pub fn eligible(now_ms: u64, inception_ms: Option<u64>, minimum_days: f64) -> bool {
    if minimum_days == 0.0 {
        return true;
    }
    let Some(inception) = inception_ms.filter(|timestamp| *timestamp > 0) else {
        return false;
    };
    let Some(elapsed_ms) = now_ms.checked_sub(inception) else {
        return false;
    };
    elapsed_ms as f64 / DAY_MS >= minimum_days
}

/// Compile the same monotone predicate to a replay index, without f32 day/timestamp loss.
pub fn first_eligible_index(timestamps: &[u64], inception_ms: Option<u64>, days: f64) -> usize {
    timestamps.partition_point(|timestamp| !eligible(*timestamp, inception_ms, days))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fractional_boundary_and_unknown_facts() {
        let inception = 1_000;
        assert!(!eligible(inception + 43_199_999, Some(inception), 0.5));
        assert!(eligible(inception + 43_200_000, Some(inception), 0.5));
        assert!(!eligible(999, Some(inception), 0.5));
        assert!(!eligible(u64::MAX, None, 0.5));
        assert!(!eligible(u64::MAX, Some(0), 0.5));
        assert!(eligible(0, None, 0.0));
        assert!(eligible(0, Some(inception), 0.0));
        assert!(!eligible(u64::MAX, Some(inception), f64::MAX));
    }

    #[test]
    fn replay_index_uses_the_same_boundary() {
        let timestamps = [1_000, 31_000, 61_000, 91_000];
        assert_eq!(
            first_eligible_index(&timestamps, Some(1_000), 1.0 / 1_440.0),
            2
        );
        assert_eq!(first_eligible_index(&timestamps, None, 1.0), 4);
        assert_eq!(first_eligible_index(&timestamps, None, 0.0), 0);
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(validate_minimum_days(invalid).is_err());
        }
    }
}
