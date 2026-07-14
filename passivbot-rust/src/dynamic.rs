#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicDistanceInputs {
    pub volatility_ema_1m: f64,
    pub volatility_ema_1h: f64,
    pub weight_volatility_1m: f64,
    pub weight_volatility_1h: f64,
    pub wallet_exposure_ratio: Option<f64>,
    pub weight_wallet_exposure: f64,
    pub min_multiplier: f64,
}

impl Default for DynamicDistanceInputs {
    fn default() -> Self {
        Self {
            volatility_ema_1m: 0.0,
            volatility_ema_1h: 0.0,
            weight_volatility_1m: 0.0,
            weight_volatility_1h: 0.0,
            wallet_exposure_ratio: None,
            weight_wallet_exposure: 0.0,
            min_multiplier: 1.0,
        }
    }
}

pub fn calc_dynamic_distance_multiplier(inputs: DynamicDistanceInputs) -> f64 {
    let we_term = inputs.wallet_exposure_ratio.unwrap_or(0.0) * inputs.weight_wallet_exposure;
    let vol_term = inputs.volatility_ema_1h * inputs.weight_volatility_1h
        + inputs.volatility_ema_1m * inputs.weight_volatility_1m;
    let multiplier = 1.0 + vol_term + we_term;
    if !(multiplier.is_finite() && inputs.min_multiplier.is_finite()) {
        panic!("non-finite dynamic distance multiplier input");
    }
    multiplier.max(inputs.min_multiplier)
}
