//! Entry timing policy shared by live planning and CPU backtests.
//!
//! Duration evaluation is separate from elapsed-time enforcement so future
//! modifiers need not duplicate timestamp, rounding, or ladder semantics.

#[derive(Clone, Copy, Debug)]
pub(crate) struct EntryCooldown {
    base_duration_minutes: f64,
}

impl EntryCooldown {
    pub(crate) fn new(base_duration_minutes: f64) -> Self {
        Self { base_duration_minutes }
    }

    fn duration_minutes(self) -> f64 {
        self.base_duration_minutes
    }

    pub(crate) fn is_active(self, now_ms: u64, last_increase_fill_ms: Option<u64>) -> bool {
        let minutes = self.duration_minutes();
        if !minutes.is_finite() || minutes <= 0.0 {
            return false;
        }
        let Some(last_fill_ms) = last_increase_fill_ms else {
            return false;
        };
        let delay_ms = (minutes * 60_000.0).ceil() as u64;
        now_ms < last_fill_ms.saturating_add(delay_ms)
    }

    pub(crate) fn allows_full_entry_ladder(self, entry_retracement_enabled: bool) -> bool {
        let minutes = self.duration_minutes();
        !entry_retracement_enabled && minutes.is_finite() && minutes == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_minutes_round_up_and_expire_at_the_boundary() {
        let policy = EntryCooldown::new(0.000025); // 1.5 milliseconds
        assert!(policy.is_active(101, Some(100)));
        assert!(!policy.is_active(102, Some(100)));
        assert!(!policy.is_active(100, None));
        assert!(EntryCooldown::new(f64::MAX).is_active(u64::MAX - 1, Some(100)));
        assert!(!EntryCooldown::new(f64::MAX).is_active(u64::MAX, Some(100)));
    }

    #[test]
    fn zero_duration_and_retracement_keep_existing_ladder_semantics() {
        let zero = EntryCooldown::new(0.0);
        assert!(!zero.is_active(100, Some(100)));
        assert!(zero.allows_full_entry_ladder(false));
        assert!(!zero.allows_full_entry_ladder(true));
        assert!(!EntryCooldown::new(0.05).allows_full_entry_ladder(false));
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let policy = EntryCooldown::new(invalid);
            assert!(!policy.is_active(100, Some(100)));
            assert!(!policy.allows_full_entry_ladder(false));
        }
    }
}
