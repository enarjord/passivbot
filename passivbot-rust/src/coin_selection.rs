use crate::types::ForagerScoreWeights;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::FromPyObject;
use std::cmp::Ordering;
use std::collections::HashSet;

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
    pub score_hysteresis_pct: f64,
    pub incumbent_indices: HashSet<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForagerPositionSide {
    Long,
    Short,
}

impl ForagerPositionSide {
    fn from_str(value: &str) -> Result<Self, ForagerSelectionError> {
        match value {
            "long" => Ok(Self::Long),
            "short" => Ok(Self::Short),
            other => Err(ForagerSelectionError::InvalidPositionSide(
                other.to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForagerCandidate {
    pub index: usize,
    pub enabled: bool,
    pub volume_score: f64,
    pub volatility_score: f64,
    pub bid: f64,
    pub ask: f64,
    pub ema_lower: f64,
    pub ema_upper: f64,
    pub entry_initial_ema_dist: f64,
}

#[derive(Debug, Clone)]
pub struct ForagerSelectionConfig {
    pub slots_to_fill: usize,
    pub volume_drop_pct: f64,
    pub weights: ForagerScoreWeights,
    pub require_forager: bool,
    pub position_side: ForagerPositionSide,
    pub score_hysteresis_pct: f64,
    pub incumbent_indices: HashSet<usize>,
}

#[derive(Debug, Clone)]
pub struct ForagerScoredCandidate {
    pub index: usize,
    pub score: f64,
    pub volume_component: f64,
    pub ema_readiness_component: f64,
    pub volatility_component: f64,
    pub selected: bool,
    pub incumbent: bool,
    pub rank: usize,
}

#[derive(Debug, Clone)]
pub struct ForagerHysteresisEvent {
    pub incumbent_index: usize,
    pub incumbent_score: f64,
    pub challenger_index: usize,
    pub challenger_score: f64,
    pub score_gap: f64,
    pub kept_incumbent: bool,
}

#[derive(Debug, Clone)]
pub struct ForagerSelectionResult {
    pub selected_indices: Vec<usize>,
    pub scored: Vec<ForagerScoredCandidate>,
    pub hysteresis_events: Vec<ForagerHysteresisEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForagerSelectionError {
    InvalidPositionSide(String),
    NonFiniteInput { field: &'static str, index: usize },
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

impl ForagerSelectionConfig {
    fn volume_drop(&self) -> f64 {
        SelectionConfig::clamp_pct(self.volume_drop_pct)
    }

    fn volume_required(&self) -> bool {
        self.volume_drop() > 0.0 || self.weights.volume != 0.0
    }

    fn volatility_required(&self) -> bool {
        self.weights.volatility != 0.0
    }

    fn ema_readiness_required(&self) -> bool {
        self.weights.ema_readiness != 0.0
    }

    fn score_hysteresis(&self) -> f64 {
        if self.score_hysteresis_pct.is_finite() {
            self.score_hysteresis_pct.max(0.0)
        } else {
            0.0
        }
    }
}

fn canonicalize_forager_weights(
    weights: &ForagerScoreWeights,
) -> Result<ForagerScoreWeights, ForagerSelectionError> {
    weights
        .canonicalize()
        .map_err(|_| ForagerSelectionError::NonFiniteInput {
            field: "forager_score_weights",
            index: 0,
        })
}

fn validate_forager_weights(
    weights: &ForagerScoreWeights,
) -> Result<ForagerScoreWeights, ForagerSelectionError> {
    let normalized = canonicalize_forager_weights(weights)?;
    if !(normalized.volume.is_finite()
        && normalized.ema_readiness.is_finite()
        && normalized.volatility.is_finite())
    {
        return Err(ForagerSelectionError::NonFiniteInput {
            field: "forager_score_weights",
            index: 0,
        });
    }
    Ok(normalized)
}

fn validate_required_score(
    value: f64,
    field: &'static str,
    index: usize,
) -> Result<f64, ForagerSelectionError> {
    if !value.is_finite() {
        return Err(ForagerSelectionError::NonFiniteInput { field, index });
    }
    Ok(value)
}

fn compute_ema_readiness_score(
    candidate: &ForagerCandidate,
    cfg: &ForagerSelectionConfig,
) -> Result<f64, ForagerSelectionError> {
    let entry_initial_ema_dist = validate_required_score(
        candidate.entry_initial_ema_dist,
        "entry_initial_ema_dist",
        candidate.index,
    )?;
    match cfg.position_side {
        ForagerPositionSide::Long => {
            let market_price =
                validate_required_score(candidate.bid, "forager_market_bid", candidate.index)?;
            let ema_lower =
                validate_required_score(candidate.ema_lower, "forager_ema_lower", candidate.index)?;
            let entry_threshold = ema_lower * (1.0 - entry_initial_ema_dist);
            if !(market_price > 0.0 && entry_threshold.is_finite() && entry_threshold > 0.0) {
                return Err(ForagerSelectionError::NonFiniteInput {
                    field: "forager_ema_readiness",
                    index: candidate.index,
                });
            }
            Ok(market_price / entry_threshold - 1.0)
        }
        ForagerPositionSide::Short => {
            let market_price =
                validate_required_score(candidate.ask, "forager_market_ask", candidate.index)?;
            let ema_upper =
                validate_required_score(candidate.ema_upper, "forager_ema_upper", candidate.index)?;
            let entry_threshold = ema_upper * (1.0 + entry_initial_ema_dist);
            if !(market_price > 0.0 && entry_threshold.is_finite() && entry_threshold > 0.0) {
                return Err(ForagerSelectionError::NonFiniteInput {
                    field: "forager_ema_readiness",
                    index: candidate.index,
                });
            }
            Ok(1.0 - market_price / entry_threshold)
        }
    }
}

fn build_coin_features(
    candidates: &[ForagerCandidate],
    cfg: &ForagerSelectionConfig,
) -> Result<Vec<CoinFeature>, ForagerSelectionError> {
    let require_volume = cfg.volume_required();
    let require_volatility = cfg.volatility_required();
    let require_ema_readiness = cfg.ema_readiness_required();

    candidates
        .iter()
        .map(|candidate| {
            if !candidate.enabled {
                return Ok(CoinFeature {
                    index: candidate.index,
                    enabled: false,
                    volume_score: 0.0,
                    volatility_score: 0.0,
                    ema_readiness_score: 0.0,
                });
            }
            let volume_score = if require_volume {
                validate_required_score(
                    candidate.volume_score,
                    "forager_volume_score",
                    candidate.index,
                )?
            } else {
                candidate.volume_score
            };
            let volatility_score = if require_volatility {
                validate_required_score(
                    candidate.volatility_score,
                    "forager_volatility_score",
                    candidate.index,
                )?
            } else {
                candidate.volatility_score
            };
            let ema_readiness_score = if require_ema_readiness {
                compute_ema_readiness_score(candidate, cfg)?
            } else {
                0.0
            };
            Ok(CoinFeature {
                index: candidate.index,
                enabled: candidate.enabled,
                volume_score,
                volatility_score,
                ema_readiness_score,
            })
        })
        .collect()
}

pub fn select_forager_candidates(
    candidates: &[ForagerCandidate],
    cfg: &ForagerSelectionConfig,
) -> Result<Vec<usize>, ForagerSelectionError> {
    Ok(select_forager_candidates_with_diagnostics(candidates, cfg)?.selected_indices)
}

pub fn select_forager_candidates_with_diagnostics(
    candidates: &[ForagerCandidate],
    cfg: &ForagerSelectionConfig,
) -> Result<ForagerSelectionResult, ForagerSelectionError> {
    let normalized_weights = validate_forager_weights(&cfg.weights)?;
    let normalized_cfg = ForagerSelectionConfig {
        slots_to_fill: cfg.slots_to_fill,
        volume_drop_pct: cfg.volume_drop_pct,
        weights: normalized_weights,
        require_forager: cfg.require_forager,
        position_side: cfg.position_side,
        score_hysteresis_pct: cfg.score_hysteresis(),
        incumbent_indices: cfg.incumbent_indices.clone(),
    };
    let features = build_coin_features(candidates, &normalized_cfg)?;
    let selection_cfg = SelectionConfig {
        slots_to_fill: normalized_cfg.slots_to_fill,
        volume_drop_pct: normalized_cfg.volume_drop_pct,
        weights: normalized_cfg.weights,
        require_forager: normalized_cfg.require_forager,
        score_hysteresis_pct: normalized_cfg.score_hysteresis_pct,
        incumbent_indices: normalized_cfg.incumbent_indices,
    };
    Ok(select_coins_with_diagnostics(&features, &selection_cfg))
}

pub fn select_coins(features: &[CoinFeature], cfg: &SelectionConfig) -> Vec<usize> {
    select_coins_with_diagnostics(features, cfg).selected_indices
}

pub fn select_coins_with_diagnostics(
    features: &[CoinFeature],
    cfg: &SelectionConfig,
) -> ForagerSelectionResult {
    let mut enabled_pos: Vec<usize> = Vec::new();
    enabled_pos.reserve(features.len());
    for (pos, f) in features.iter().enumerate() {
        if f.enabled {
            enabled_pos.push(pos);
        }
    }
    if enabled_pos.is_empty() {
        return ForagerSelectionResult {
            selected_indices: Vec::new(),
            scored: Vec::new(),
            hysteresis_events: Vec::new(),
        };
    }

    if !cfg.require_forager {
        return ForagerSelectionResult {
            selected_indices: enabled_pos.iter().map(|&p| features[p].index).collect(),
            scored: Vec::new(),
            hysteresis_events: Vec::new(),
        };
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
) -> ForagerSelectionResult {
    if positions.is_empty() {
        return ForagerSelectionResult {
            selected_indices: Vec::new(),
            scored: Vec::new(),
            hysteresis_events: Vec::new(),
        };
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

    let mut scored: Vec<ScoredPosition> = positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let score = cfg.weights.volume * volume_scores[i]
                + cfg.weights.ema_readiness * ema_readiness_scores[i]
                + cfg.weights.volatility * volatility_scores[i];
            ScoredPosition {
                pos,
                score,
                volume_component: volume_scores[i],
                ema_readiness_component: ema_readiness_scores[i],
                volatility_component: volatility_scores[i],
            }
        })
        .collect();

    scored.sort_unstable_by(|a, b| {
        let fa = &features[a.pos];
        let fb = &features[b.pos];
        compare_desc(a.score, b.score, fa.index, fb.index)
    });

    let selection_size = slots_to_fill.min(scored.len());
    let mut selected: Vec<ScoredPosition> = scored.iter().take(selection_size).copied().collect();
    let mut hysteresis_events: Vec<ForagerHysteresisEvent> = Vec::new();
    apply_score_hysteresis(
        features,
        &scored,
        cfg,
        &mut selected,
        &mut hysteresis_events,
    );

    selected.sort_unstable_by(|a, b| {
        let fa = &features[a.pos];
        let fb = &features[b.pos];
        compare_desc(a.score, b.score, fa.index, fb.index)
    });
    let selected_indices: Vec<usize> = selected
        .iter()
        .map(|item| features[item.pos].index)
        .collect();
    let selected_set: HashSet<usize> = selected_indices.iter().copied().collect();
    let scored = scored
        .iter()
        .enumerate()
        .map(|(rank, item)| {
            let index = features[item.pos].index;
            ForagerScoredCandidate {
                index,
                score: item.score,
                volume_component: item.volume_component,
                ema_readiness_component: item.ema_readiness_component,
                volatility_component: item.volatility_component,
                selected: selected_set.contains(&index),
                incumbent: cfg.incumbent_indices.contains(&index),
                rank: rank + 1,
            }
        })
        .collect();
    ForagerSelectionResult {
        selected_indices,
        scored,
        hysteresis_events,
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredPosition {
    pos: usize,
    score: f64,
    volume_component: f64,
    ema_readiness_component: f64,
    volatility_component: f64,
}

fn apply_score_hysteresis(
    features: &[CoinFeature],
    scored: &[ScoredPosition],
    cfg: &SelectionConfig,
    selected: &mut Vec<ScoredPosition>,
    events: &mut Vec<ForagerHysteresisEvent>,
) {
    let hysteresis = if cfg.score_hysteresis_pct.is_finite() {
        cfg.score_hysteresis_pct.max(0.0)
    } else {
        0.0
    };
    if hysteresis <= 0.0 || cfg.incumbent_indices.is_empty() || selected.is_empty() {
        return;
    }

    for incumbent in scored {
        let incumbent_pos = incumbent.pos;
        let incumbent_score = incumbent.score;
        let incumbent_idx = features[incumbent_pos].index;
        if !cfg.incumbent_indices.contains(&incumbent_idx) {
            continue;
        }
        if selected.iter().any(|item| item.pos == incumbent_pos) {
            continue;
        }

        let replace_at = selected
            .iter()
            .enumerate()
            .filter(|(_, item)| !cfg.incumbent_indices.contains(&features[item.pos].index))
            .min_by(|(_, a), (_, b)| {
                compare_desc(
                    b.score,
                    a.score,
                    features[b.pos].index,
                    features[a.pos].index,
                )
            })
            .map(|(i, _)| i);

        let Some(replace_at) = replace_at else {
            continue;
        };
        let challenger = selected[replace_at];
        let challenger_idx = features[challenger.pos].index;
        let challenger_score = challenger.score;
        let score_gap = challenger_score - incumbent_score;
        let kept_incumbent = score_gap <= hysteresis;
        events.push(ForagerHysteresisEvent {
            incumbent_index: incumbent_idx,
            incumbent_score,
            challenger_index: challenger_idx,
            challenger_score,
            score_gap,
            kept_incumbent,
        });
        if kept_incumbent {
            selected[replace_at] = *incumbent;
        }
    }
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

impl<'source> FromPyObject<'source> for ForagerCandidate {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(Self {
            index: ob.get_item("index")?.extract::<usize>()?,
            enabled: ob.get_item("enabled")?.extract::<bool>()?,
            volume_score: ob.get_item("volume_score")?.extract::<f64>()?,
            volatility_score: ob.get_item("volatility_score")?.extract::<f64>()?,
            bid: ob.get_item("bid")?.extract::<f64>()?,
            ask: ob.get_item("ask")?.extract::<f64>()?,
            ema_lower: ob.get_item("ema_lower")?.extract::<f64>()?,
            ema_upper: ob.get_item("ema_upper")?.extract::<f64>()?,
            entry_initial_ema_dist: ob.get_item("entry_initial_ema_dist")?.extract::<f64>()?,
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
    let weights = validate_forager_weights(&weights).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "forager_score_weights must be finite and non-negative",
        )
    })?;
    let features: Vec<CoinFeature> = py_features.into_iter().map(Into::into).collect();
    let cfg = SelectionConfig {
        slots_to_fill,
        volume_drop_pct,
        weights,
        require_forager,
        score_hysteresis_pct: 0.0,
        incumbent_indices: HashSet::new(),
    };
    Ok(select_coins(&features, &cfg))
}

#[pyfunction]
pub fn select_forager_candidates_py(
    py_candidates: Vec<ForagerCandidate>,
    pside: &str,
    slots_to_fill: usize,
    volume_drop_pct: f64,
    weights: ForagerScoreWeights,
    require_forager: bool,
) -> PyResult<Vec<usize>> {
    let position_side = ForagerPositionSide::from_str(pside).map_err(|err| match err {
        ForagerSelectionError::InvalidPositionSide(value) => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid forager position side: {value}"
            ))
        }
        _ => pyo3::exceptions::PyValueError::new_err("invalid forager position side"),
    })?;
    let cfg = ForagerSelectionConfig {
        slots_to_fill,
        volume_drop_pct,
        weights: validate_forager_weights(&weights).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "forager_score_weights must be finite and non-negative",
            )
        })?,
        require_forager,
        position_side,
        score_hysteresis_pct: 0.0,
        incumbent_indices: HashSet::new(),
    };
    select_forager_candidates(&py_candidates, &cfg).map_err(|err| match err {
        ForagerSelectionError::InvalidPositionSide(value) => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid forager position side: {value}"
            ))
        }
        ForagerSelectionError::NonFiniteInput { field, index } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid forager candidate input '{field}' at index {index}"
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(
        index: usize,
        volume: f64,
        volatility: f64,
        bid: f64,
        ask: f64,
    ) -> ForagerCandidate {
        ForagerCandidate {
            index,
            enabled: true,
            volume_score: volume,
            volatility_score: volatility,
            bid,
            ask,
            ema_lower: 100.0,
            ema_upper: 100.0,
            entry_initial_ema_dist: 0.1,
        }
    }

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
            score_hysteresis_pct: 0.0,
            incumbent_indices: HashSet::new(),
        }
    }

    fn default_forager_config(pside: ForagerPositionSide) -> ForagerSelectionConfig {
        ForagerSelectionConfig {
            slots_to_fill: 3,
            volume_drop_pct: 0.0,
            weights: ForagerScoreWeights::default(),
            require_forager: true,
            position_side: pside,
            score_hysteresis_pct: 0.0,
            incumbent_indices: HashSet::new(),
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
    fn zero_weights_fall_back_to_ema_readiness_only_ranking() {
        let features = vec![
            make_feature(2, 10.0, 0.9, 0.3),
            make_feature(0, 1.0, 0.2, -0.1),
            make_feature(1, 5.0, 0.5, 0.2),
        ];
        let cfg = SelectionConfig {
            slots_to_fill: 2,
            weights: ForagerScoreWeights {
                volume: 0.0,
                ema_readiness: 0.0,
                volatility: 0.0,
            },
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![0, 1]);
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

    #[test]
    fn score_hysteresis_keeps_incumbent_within_tolerance() {
        let features = vec![
            make_feature(0, 1.0, 0.0, 0.0),
            make_feature(1, 0.99, 0.0, 0.0),
            make_feature(2, 0.1, 0.0, 0.0),
        ];
        let cfg = SelectionConfig {
            slots_to_fill: 1,
            weights: ForagerScoreWeights {
                volume: 1.0,
                ema_readiness: 0.0,
                volatility: 0.0,
            },
            score_hysteresis_pct: 0.02,
            incumbent_indices: HashSet::from([1]),
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![1]);
        let result = select_coins_with_diagnostics(&features, &cfg);
        assert_eq!(result.selected_indices, vec![1]);
        assert_eq!(result.hysteresis_events.len(), 1);
        assert!(result.hysteresis_events[0].kept_incumbent);
        assert_eq!(result.hysteresis_events[0].incumbent_index, 1);
        assert_eq!(result.hysteresis_events[0].challenger_index, 0);
        let incumbent = result.scored.iter().find(|item| item.index == 1).unwrap();
        assert!(incumbent.selected);
        assert!(incumbent.incumbent);
    }

    #[test]
    fn score_hysteresis_replaces_incumbent_outside_tolerance() {
        let features = vec![
            make_feature(0, 1.0, 0.0, 0.0),
            make_feature(1, 0.8, 0.0, 0.0),
            make_feature(2, 0.1, 0.0, 0.0),
        ];
        let cfg = SelectionConfig {
            slots_to_fill: 1,
            weights: ForagerScoreWeights {
                volume: 1.0,
                ema_readiness: 0.0,
                volatility: 0.0,
            },
            score_hysteresis_pct: 0.02,
            incumbent_indices: HashSet::from([1]),
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![0]);
        let result = select_coins_with_diagnostics(&features, &cfg);
        assert_eq!(result.selected_indices, vec![0]);
        assert_eq!(result.hysteresis_events.len(), 1);
        assert!(!result.hysteresis_events[0].kept_incumbent);
        assert_eq!(result.hysteresis_events[0].incumbent_index, 1);
        assert_eq!(result.hysteresis_events[0].challenger_index, 0);
    }

    #[test]
    fn select_forager_candidates_uses_bid_for_long_readiness() {
        let candidates = vec![
            make_candidate(0, 1.0, 1.0, 88.2, 88.2),
            make_candidate(1, 1.0, 1.0, 90.9, 90.9),
        ];
        let cfg = ForagerSelectionConfig {
            slots_to_fill: 1,
            weights: ForagerScoreWeights {
                volume: 0.0,
                ema_readiness: 1.0,
                volatility: 0.0,
            },
            ..default_forager_config(ForagerPositionSide::Long)
        };
        assert_eq!(
            select_forager_candidates(&candidates, &cfg).unwrap(),
            vec![0]
        );
    }

    #[test]
    fn select_forager_candidates_uses_ask_for_short_readiness() {
        let candidates = vec![
            make_candidate(0, 1.0, 1.0, 109.0, 109.0),
            make_candidate(1, 1.0, 1.0, 111.0, 111.0),
        ];
        let cfg = ForagerSelectionConfig {
            slots_to_fill: 1,
            weights: ForagerScoreWeights {
                volume: 0.0,
                ema_readiness: 1.0,
                volatility: 0.0,
            },
            ..default_forager_config(ForagerPositionSide::Short)
        };
        assert_eq!(
            select_forager_candidates(&candidates, &cfg).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn select_forager_candidates_rejects_missing_required_input() {
        let candidates = vec![ForagerCandidate {
            ema_lower: f64::NAN,
            ..make_candidate(0, 1.0, 1.0, 90.0, 90.0)
        }];
        let cfg = ForagerSelectionConfig {
            slots_to_fill: 1,
            weights: ForagerScoreWeights {
                volume: 0.0,
                ema_readiness: 1.0,
                volatility: 0.0,
            },
            ..default_forager_config(ForagerPositionSide::Long)
        };
        assert_eq!(
            select_forager_candidates(&candidates, &cfg),
            Err(ForagerSelectionError::NonFiniteInput {
                field: "forager_ema_lower",
                index: 0,
            })
        );
    }
}
