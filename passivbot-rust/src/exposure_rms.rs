//! Candle-level exposure moments; budgets are nominal (exclude excess allowance).
use crate::types::Analysis;

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

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn budget_weighted_candle_moments_include_flat_capacity_and_gross_hedges() {
        let mut state = ExposureRms::default();
        state.record([&[(0.3, 0.4), (0.2, 0.8)], &[(0.4, 0.5)]], [2.0, 1.0]);
        state.record([&[], &[]], [2.0, 1.0]);
        let mut a = Analysis::default();
        state.apply(&mut a);
        close(a.total_wallet_exposure_rms_long, (0.25_f64 / 2.0).sqrt());
        close(a.total_wallet_exposure_rms_short, (0.16_f64 / 2.0).sqrt());
        close(a.total_wallet_exposure_rms_gross, (0.81_f64 / 2.0).sqrt());
        close(
            a.total_wallet_exposure_utilization_rms_long,
            (0.0625_f64 / 2.0).sqrt(),
        );
        close(
            a.position_exposure_utilization_rms_long,
            ((0.09 / 0.8 + 0.04 / 1.6) / 2.0_f64).sqrt(),
        );
        close(
            a.position_exposure_utilization_rms_short,
            (0.32_f64 / 2.0).sqrt(),
        );
    }

    #[test]
    fn utilization_is_scale_invariant_and_uses_each_samples_budget() {
        let mut a = ExposureRms::default();
        let mut b = ExposureRms::default();
        for wel in [0.5, 0.25] {
            a.record([&[(0.5, wel)], &[]], [1.0, 0.0]);
            b.record([&[(0.75, wel * 1.5)], &[]], [1.5, 0.0]);
        }
        let mut x = Analysis::default();
        let mut y = Analysis::default();
        a.apply(&mut x);
        b.apply(&mut y);
        close(
            x.total_wallet_exposure_utilization_rms_long,
            y.total_wallet_exposure_utilization_rms_long,
        );
        close(
            x.position_exposure_utilization_rms_long,
            y.position_exposure_utilization_rms_long,
        );
        close(x.position_exposure_utilization_rms_long, 0.75_f64.sqrt());
        close(
            y.total_wallet_exposure_rms_long,
            x.total_wallet_exposure_rms_long * 1.5,
        );
    }

    #[test]
    fn empty_is_undefined_flat_is_zero_and_invalid_budgets_do_not_hide_load() {
        let mut a = Analysis::default();
        let mut state = ExposureRms::default();
        state.apply(&mut a);
        assert!(a.total_wallet_exposure_rms_gross.is_nan());
        state.record([&[], &[(0.0, f64::NAN)]], [0.0, f64::NAN]);
        state.apply(&mut a);
        close(a.position_exposure_utilization_rms_short, 0.0);
        close(a.total_wallet_exposure_utilization_rms_long, 0.0);
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut state = ExposureRms::default();
            state.record([&[(0.5, invalid)], &[]], [1.0, 0.0]);
            state.apply(&mut a);
            assert!(a.position_exposure_utilization_rms_long.is_nan());
            close(a.total_wallet_exposure_rms_long, 0.5);
            let mut state = ExposureRms::default();
            state.record([&[(0.5, 0.5)], &[]], [invalid, 0.0]);
            state.apply(&mut a);
            assert!(a.total_wallet_exposure_utilization_rms_long.is_nan());
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct ExposureRms {
    samples: u64,
    sums: [f64; 7],
}

impl ExposureRms {
    /// One sample includes flat sides and unoccupied capacity. Each position is
    /// (absolute wallet exposure, current runtime base WEL).
    #[cfg(test)]
    pub fn record(&mut self, positions: [&[(f64, f64)]; 2], limits: [f64; 2]) {
        let mut exposure = [0.0; 2];
        let mut position_moment = [0.0; 2];
        for side in 0..2 {
            for &(e, wel) in positions[side] {
                exposure[side] += e;
                if e == 0.0 {
                    continue;
                }
                position_moment[side] += self::position_moment(e, wel, limits[side]);
            }
        }
        self.record_moments(exposure, position_moment, limits);
    }

    pub fn record_moments(&mut self, exposure: [f64; 2], position: [f64; 2], limits: [f64; 2]) {
        self.samples += 1;
        let ratio = |s: usize| {
            let e = exposure[s];
            if e == 0.0 {
                0.0
            } else if e.is_finite() && e > 0.0 && limits[s].is_finite() && limits[s] > 0.0 {
                e / limits[s]
            } else {
                f64::NAN
            }
        };
        let values = [
            exposure[0] * exposure[0],
            exposure[1] * exposure[1],
            (exposure[0] + exposure[1]).powi(2),
            ratio(0).powi(2),
            ratio(1).powi(2),
            position[0],
            position[1],
        ];
        for (sum, value) in self.sums.iter_mut().zip(values) {
            *sum += value;
        }
    }

    pub fn apply(&self, analysis: &mut Analysis) {
        let rms = |i: usize| {
            if self.samples == 0 {
                f64::NAN
            } else {
                (self.sums[i] / self.samples as f64).sqrt()
            }
        };
        analysis.total_wallet_exposure_rms_long = rms(0);
        analysis.total_wallet_exposure_rms_short = rms(1);
        analysis.total_wallet_exposure_rms_gross = rms(2);
        analysis.total_wallet_exposure_utilization_rms_long = rms(3);
        analysis.total_wallet_exposure_utilization_rms_short = rms(4);
        analysis.position_exposure_utilization_rms_long = rms(5);
        analysis.position_exposure_utilization_rms_short = rms(6);
    }
}
