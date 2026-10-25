#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicDistanceInputs {
    pub log_range_ema_1m: f64,
    pub log_range_ema_1h: f64,
    pub log_range_1m_coefficient: f64,
    pub log_range_1h_coefficient: f64,
    pub wallet_exposure_ratio: Option<f64>,
    pub weight_wallet_exposure: f64,
    pub min_multiplier: f64,
}

impl Default for DynamicDistanceInputs {
    fn default() -> Self {
        Self {
            log_range_ema_1m: 0.0,
            log_range_ema_1h: 0.0,
            log_range_1m_coefficient: 0.0,
            log_range_1h_coefficient: 0.0,
            wallet_exposure_ratio: None,
            weight_wallet_exposure: 0.0,
            min_multiplier: 1.0,
        }
    }
}

pub fn calc_dynamic_distance_multiplier(inputs: DynamicDistanceInputs) -> f64 {
    let we_term = inputs.wallet_exposure_ratio.unwrap_or(0.0) * inputs.weight_wallet_exposure;
    let vol_term = inputs.log_range_ema_1h * inputs.log_range_1h_coefficient
        + inputs.log_range_ema_1m * inputs.log_range_1m_coefficient;
    let multiplier = 1.0 + vol_term + we_term;
    if !(multiplier.is_finite() && inputs.min_multiplier.is_finite()) {
        panic!("non-finite dynamic distance multiplier input");
    }
    multiplier.max(inputs.min_multiplier)
}
