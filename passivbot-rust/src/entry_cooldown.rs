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

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooldownWeights {
    pub exposure_ratio: f64,
    pub adverse_directionality: f64,
}

pub(crate) fn validate(params: &crate::types::BotParams) -> Result<(), String> {
    let w = params.entry_cooldown_weights_minutes;
    for value in [
        params.risk_entry_cooldown_minutes,
        params.entry_cooldown_min_duration_minutes,
        w.exposure_ratio,
        w.adverse_directionality,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err("entry cooldown values must be finite and nonnegative".into());
        }
    }
    match params.entry_cooldown_max_duration_minutes {
        Some(max) if !max.is_finite() || max < params.entry_cooldown_min_duration_minutes => {
            return Err("entry cooldown ceiling must be finite and >= floor".into())
        }
        None if w.exposure_ratio > 0.0 || w.adverse_directionality > 0.0 => {
            return Err("adaptive entry cooldown requires a finite ceiling".into())
        }
        _ => {}
    }
    if !params.unilateralness_ema_span_1m.is_finite() || params.unilateralness_ema_span_1m < 1.0 {
        return Err("unilateralness EMA span must be finite and >= 1".into());
    }
    Ok(())
}

/// Nonnegative modifiers cannot change a duration already pinned to its ceiling.
/// Callers validate the policy before using this input-independence check.
pub(crate) fn constant_duration(params: &crate::types::BotParams) -> Option<f64> {
    params.entry_cooldown_max_duration_minutes.filter(|max| {
        params.risk_entry_cooldown_minutes.max(params.entry_cooldown_min_duration_minutes) >= *max
    })
}

pub(crate) fn uses_adverse_rms(params: &crate::types::BotParams) -> bool {
    params.entry_cooldown_weights_minutes.adverse_directionality > 0.0
        && constant_duration(params).is_none()
}

/// Optional inputs are read only for enabled weights. Exposure is not capped at one.
pub(crate) fn effective_duration(
    params: &crate::types::BotParams,
    exposure_ratio: Option<f64>,
    adverse_score: Option<f64>,
) -> Result<f64, String> {
    validate(params)?;
    if let Some(minutes) = constant_duration(params) {
        return Ok(minutes);
    }
    let mut minutes = params.risk_entry_cooldown_minutes;
    for (weight, input) in [
        (
            params.entry_cooldown_weights_minutes.exposure_ratio,
            exposure_ratio,
        ),
        (
            params.entry_cooldown_weights_minutes.adverse_directionality,
            adverse_score,
        ),
    ] {
        if weight > 0.0 {
            let value = input.ok_or("missing enabled entry cooldown input")?;
            if !value.is_finite() || value < 0.0 {
                return Err("invalid entry cooldown input".into());
            }
            minutes += weight * value;
        }
    }
    minutes = minutes.max(params.entry_cooldown_min_duration_minutes);
    if let Some(max) = params.entry_cooldown_max_duration_minutes {
        minutes = minutes.min(max);
    }
    Ok(minutes)
}

#[pyo3::prelude::pyfunction]
pub fn entry_cooldown_durations_json(raw: &str) -> pyo3::PyResult<String> {
    crate::orchestrator::entry_cooldown_durations_json(raw)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}
