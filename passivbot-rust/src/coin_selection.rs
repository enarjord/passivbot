use crate::types::ForagerScoreWeights;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::FromPyObject;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct CoinFeature {
    pub index: usize,
    pub enabled: bool,
    pub volume_score: f64,
    pub volatility_score: f64,
    pub ema_readiness_score: f64,
}

#[derive(Debug, Clone)]
pub struct SelectionConfig {
    pub slots_to_fill: usize,
    pub volume_drop_pct: f64,
    pub weights: ForagerScoreWeights,
    pub require_forager: bool,
}

impl SelectionConfig {
    fn clamp_pct(value: f64) -> f64 {
        if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn volume_drop(&self) -> f64 {
        Self::clamp_pct(self.volume_drop_pct)
    }
}

pub fn select_coins(features: &[CoinFeature], cfg: &SelectionConfig) -> Vec<usize> {
    let mut enabled_pos: Vec<usize> = Vec::new();
    enabled_pos.reserve(features.len());
    for (pos, f) in features.iter().enumerate() {
        if f.enabled {
            enabled_pos.push(pos);
        }
    }
    if enabled_pos.is_empty() {
        return Vec::new();
    }

    if !cfg.require_forager {
        return enabled_pos.iter().map(|&p| features[p].index).collect();
    }

    let slots_to_fill = cfg.slots_to_fill.max(1);
    prune_low_volume_tail(features, &mut enabled_pos, cfg.volume_drop(), slots_to_fill);
    score_forager_candidates(features, &enabled_pos, cfg, slots_to_fill)
}

fn prune_low_volume_tail(
    features: &[CoinFeature],
    positions: &mut Vec<usize>,
    volume_drop_pct: f64,
    slots_to_fill: usize,
) {
    if positions.is_empty() {
        return;
    }

    let mut keep = ((positions.len() as f64) * (1.0 - volume_drop_pct)).round() as usize;
    if keep == 0 {
        keep = 1;
    }
    keep = keep.max(slots_to_fill).min(positions.len());
    if keep >= positions.len() {
        return;
    }

    let cmp_volume = |pa: &usize, pb: &usize| {
        let a = &features[*pa];
        let b = &features[*pb];
        compare_desc(a.volume_score, b.volume_score, a.index, b.index)
    };
    positions.select_nth_unstable_by(keep.saturating_sub(1), cmp_volume);
    positions.truncate(keep);
}

fn score_forager_candidates(
    features: &[CoinFeature],
    positions: &[usize],
    cfg: &SelectionConfig,
    slots_to_fill: usize,
) -> Vec<usize> {
    if positions.is_empty() {
        return Vec::new();
    }

    let volume_scores = normalize_higher_is_better(
        &positions
            .iter()
            .map(|&pos| features[pos].volume_score)
            .collect::<Vec<f64>>(),
    );
    let ema_readiness_scores = normalize_lower_is_better(
        &positions
            .iter()
            .map(|&pos| features[pos].ema_readiness_score)
            .collect::<Vec<f64>>(),
    );
    let volatility_scores = normalize_higher_is_better(
        &positions
            .iter()
            .map(|&pos| features[pos].volatility_score)
            .collect::<Vec<f64>>(),
    );

    let mut scored: Vec<(usize, f64)> = positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let score = cfg.weights.volume * volume_scores[i]
                + cfg.weights.ema_readiness * ema_readiness_scores[i]
                + cfg.weights.volatility * volatility_scores[i];
            (pos, score)
        })
        .collect();

    scored.sort_unstable_by(|a, b| {
        let fa = &features[a.0];
        let fb = &features[b.0];
        compare_desc(a.1, b.1, fa.index, fb.index)
    });

    scored
        .into_iter()
        .take(slots_to_fill)
        .map(|(pos, _)| features[pos].index)
        .collect()
}

fn normalize_higher_is_better(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return vec![1.0; values.len()];
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() <= f64::EPSILON {
        return values
            .iter()
            .map(|value| if value.is_finite() { 1.0 } else { 0.0 })
            .collect();
    }
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                (value - min) / (max - min)
            } else {
                0.0
            }
        })
        .collect()
}

fn normalize_lower_is_better(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return vec![1.0; values.len()];
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() <= f64::EPSILON {
        return values
            .iter()
            .map(|value| if value.is_finite() { 1.0 } else { 0.0 })
            .collect();
    }
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                (max - value) / (max - min)
            } else {
                0.0
            }
        })
        .collect()
}

fn compare_desc(value_a: f64, value_b: f64, idx_a: usize, idx_b: usize) -> Ordering {
    match value_b.partial_cmp(&value_a).unwrap_or(Ordering::Equal) {
        Ordering::Equal => idx_a.cmp(&idx_b),
        ordering => ordering,
    }
}

pub struct CoinFeatureInput {
    pub index: usize,
    pub enabled: bool,
    pub volume_score: f64,
    pub volatility_score: f64,
    pub ema_readiness_score: f64,
}

impl<'source> FromPyObject<'source> for CoinFeatureInput {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index = ob.get_item("index")?.extract::<usize>()?;
        let enabled = ob.get_item("enabled")?.extract::<bool>()?;
        let volume_score = ob.get_item("volume_score")?.extract::<f64>()?;
        let volatility_score = ob.get_item("volatility_score")?.extract::<f64>()?;
        let ema_readiness_score = ob.get_item("ema_readiness_score")?.extract::<f64>()?;
        Ok(CoinFeatureInput {
            index,
            enabled,
            volume_score,
            volatility_score,
            ema_readiness_score,
        })
    }
}

impl From<CoinFeatureInput> for CoinFeature {
    fn from(value: CoinFeatureInput) -> Self {
        CoinFeature {
            index: value.index,
            enabled: value.enabled,
            volume_score: value.volume_score,
            volatility_score: value.volatility_score,
            ema_readiness_score: value.ema_readiness_score,
        }
    }
}

impl<'source> FromPyObject<'source> for ForagerScoreWeights {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(Self {
            volume: ob.get_item("volume")?.extract::<f64>()?,
            ema_readiness: ob.get_item("ema_readiness")?.extract::<f64>()?,
            volatility: ob.get_item("volatility")?.extract::<f64>()?,
        })
    }
}

#[pyfunction]
pub fn select_coin_indices_py(
    py_features: Vec<CoinFeatureInput>,
    slots_to_fill: usize,
    volume_drop_pct: f64,
    weights: ForagerScoreWeights,
    require_forager: bool,
) -> PyResult<Vec<usize>> {
    let total_weight = weights.volume + weights.ema_readiness + weights.volatility;
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "forager_score_weights must contain at least one positive finite weight",
        ));
    }
    let features: Vec<CoinFeature> = py_features.into_iter().map(Into::into).collect();
    let cfg = SelectionConfig {
        slots_to_fill,
        volume_drop_pct,
        weights,
        require_forager,
    };
    Ok(select_coins(&features, &cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature(index: usize, volume: f64, volatility: f64, ema_readiness: f64) -> CoinFeature {
        CoinFeature {
            index,
            enabled: true,
            volume_score: volume,
            volatility_score: volatility,
            ema_readiness_score: ema_readiness,
        }
    }

    fn default_config() -> SelectionConfig {
        SelectionConfig {
            slots_to_fill: 3,
            volume_drop_pct: 0.0,
            weights: ForagerScoreWeights::default(),
            require_forager: true,
        }
    }

    #[test]
    fn returns_enabled_indices_when_not_forager() {
        let features = vec![
            CoinFeature {
                index: 0,
                enabled: true,
                volume_score: 0.1,
                volatility_score: 0.1,
                ema_readiness_score: 0.1,
            },
            CoinFeature {
                index: 1,
                enabled: false,
                volume_score: 1.0,
                volatility_score: 1.0,
                ema_readiness_score: 1.0,
            },
            CoinFeature {
                index: 2,
                enabled: true,
                volume_score: 0.2,
                volatility_score: 0.2,
                ema_readiness_score: 0.2,
            },
        ];
        let cfg = SelectionConfig {
            require_forager: false,
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![0, 2]);
    }

    #[test]
    fn respects_volume_drop_percentage_but_keeps_enough_candidates() {
        let features = vec![
            make_feature(0, 0.1, 0.1, 0.1),
            make_feature(1, 0.2, 0.2, 0.2),
            make_feature(2, 0.3, 0.3, 0.3),
            make_feature(3, 0.4, 0.4, 0.4),
        ];
        let cfg = SelectionConfig {
            slots_to_fill: 3,
            volume_drop_pct: 0.9,
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![3, 2, 1]);
    }

    #[test]
    fn ema_readiness_can_override_volatility_when_weighted() {
        let features = vec![
            make_feature(0, 1.0, 1.0, -0.01),
            make_feature(1, 1.0, 1.0, 0.20),
        ];
        let cfg = SelectionConfig {
            slots_to_fill: 1,
            weights: ForagerScoreWeights {
                volume: 0.0,
                ema_readiness: 1.0,
                volatility: 0.0,
            },
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![0]);
    }

    #[test]
    fn legacy_like_default_prefers_highest_volatility() {
        let features = vec![
            make_feature(0, 1.0, 0.2, 0.0),
            make_feature(1, 1.0, 0.8, 0.0),
            make_feature(2, 1.0, 0.5, 0.0),
        ];
        assert_eq!(select_coins(&features, &default_config()), vec![1, 2, 0]);
    }

    #[test]
    fn ties_are_deterministic_by_index() {
        let features = vec![
            make_feature(2, 1.0, 1.0, 0.0),
            make_feature(1, 1.0, 1.0, 0.0),
            make_feature(0, 1.0, 1.0, 0.0),
        ];
        let cfg = SelectionConfig {
            slots_to_fill: 2,
            weights: ForagerScoreWeights {
                volume: 1.0,
                ema_readiness: 0.0,
                volatility: 0.0,
            },
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![0, 1]);
    }
}
