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
}

#[derive(Debug, Clone)]
pub struct SelectionConfig {
    pub max_positions: usize,
    pub volume_drop_pct: f64,
    pub volatility_drop_pct: f64,
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

    fn volatility_drop(&self) -> f64 {
        Self::clamp_pct(self.volatility_drop_pct)
    }
}

pub fn select_coins(features: &[CoinFeature], cfg: &SelectionConfig) -> Vec<usize> {
    // Work on positions within the `features` slice to avoid assuming contiguity of `CoinFeature.index`.
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
        // Preserve deterministic ordering matching the input slice.
        return enabled_pos.iter().map(|&p| features[p].index).collect();
    }

    let max_positions = cfg.max_positions.max(1);
    select_by_volume(features, &mut enabled_pos, cfg, max_positions);
    select_by_volatility(features, &mut enabled_pos, cfg, max_positions)
}

fn select_by_volume(
    features: &[CoinFeature],
    positions: &mut Vec<usize>,
    cfg: &SelectionConfig,
    max_positions: usize,
) {
    if positions.is_empty() {
        return;
    }

    let drop = cfg.volume_drop();
    let mut keep = ((positions.len() as f64) * (1.0 - drop)).round() as usize;
    if keep == 0 {
        keep = 1;
    }
    keep = keep.max(max_positions).min(positions.len());
    if keep >= positions.len() {
        return;
    }

    // Select the top `keep` by volume score using a deterministic comparator (score desc, idx asc).
    let cmp_volume = |pa: &usize, pb: &usize| {
        let a = &features[*pa];
        let b = &features[*pb];
        compare_desc(a.volume_score, b.volume_score, a.index, b.index)
    };
    positions.select_nth_unstable_by(keep.saturating_sub(1), cmp_volume);
    positions.truncate(keep);
}

fn select_by_volatility(
    features: &[CoinFeature],
    positions: &mut Vec<usize>,
    cfg: &SelectionConfig,
    max_positions: usize,
) -> Vec<usize> {
    if positions.is_empty() {
        return Vec::new();
    }

    let drop = cfg.volatility_drop();
    let mut keep = ((positions.len() as f64) * (1.0 - drop)).round() as usize;
    if keep == 0 {
        keep = 1;
    }
    keep = keep.max(max_positions).min(positions.len());

    let cmp_vol = |pa: &usize, pb: &usize| {
        let a = &features[*pa];
        let b = &features[*pb];
        compare_desc(a.volatility_score, b.volatility_score, a.index, b.index)
    };

    // Match legacy behavior:
    // 1) sort by volatility (desc)
    // 2) drop the first `drop_count` entries
    // 3) return the next `max_positions`
    //
    // We avoid a full sort by:
    // - partitioning at `drop_count` to discard the top segment
    // - selecting the top `max_positions` within the retained tail
    // - sorting only that small subset for deterministic order
    let drop_count = positions.len().saturating_sub(keep);
    if drop_count > 0 && drop_count < positions.len() {
        positions.select_nth_unstable_by(drop_count, cmp_vol);
    }
    let tail_start = drop_count.min(positions.len());
    let tail_len = positions.len().saturating_sub(tail_start);
    if tail_len == 0 {
        return Vec::new();
    }

    let take = max_positions.min(tail_len);
    {
        let tail = &mut positions[tail_start..];
        if take < tail.len() {
            tail.select_nth_unstable_by(take.saturating_sub(1), cmp_vol);
        }
        tail[..take].sort_unstable_by(cmp_vol);
    }
    positions[tail_start..tail_start + take]
        .iter()
        .map(|&p| features[p].index)
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
}

impl<'source> FromPyObject<'source> for CoinFeatureInput {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index = ob.get_item("index")?.extract::<usize>()?;
        let enabled = ob.get_item("enabled")?.extract::<bool>()?;
        let volume_score = ob.get_item("volume_score")?.extract::<f64>()?;
        let volatility_score = ob.get_item("volatility_score")?.extract::<f64>()?;
        Ok(CoinFeatureInput {
            index,
            enabled,
            volume_score,
            volatility_score,
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
        }
    }
}

#[pyfunction]
pub fn select_coin_indices_py(
    py_features: Vec<CoinFeatureInput>,
    max_positions: usize,
    volume_drop_pct: f64,
    volatility_drop_pct: f64,
    require_forager: bool,
) -> PyResult<Vec<usize>> {
    let features: Vec<CoinFeature> = py_features.into_iter().map(Into::into).collect();
    let cfg = SelectionConfig {
        max_positions,
        volume_drop_pct,
        volatility_drop_pct,
        require_forager,
    };
    Ok(select_coins(&features, &cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature(index: usize, volume: f64, volatility: f64) -> CoinFeature {
        CoinFeature {
            index,
            enabled: true,
            volume_score: volume,
            volatility_score: volatility,
        }
    }

    fn default_config() -> SelectionConfig {
        SelectionConfig {
            max_positions: 3,
            volume_drop_pct: 0.0,
            volatility_drop_pct: 0.0,
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
            },
            CoinFeature {
                index: 1,
                enabled: false,
                volume_score: 1.0,
                volatility_score: 1.0,
            },
            CoinFeature {
                index: 2,
                enabled: true,
                volume_score: 0.2,
                volatility_score: 0.2,
            },
        ];
        let cfg = SelectionConfig {
            require_forager: false,
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![0, 2]);
    }

    #[test]
    fn respects_volume_drop_percentage() {
        let features = vec![
            make_feature(0, 0.1, 0.1),
            make_feature(1, 0.3, 0.3),
            make_feature(2, 0.2, 0.2),
            make_feature(3, 0.4, 0.4),
            make_feature(4, 0.5, 0.5),
        ];
        let cfg = SelectionConfig {
            max_positions: 2,
            volume_drop_pct: 0.4,
            ..default_config()
        };
        // After dropping top 40% by volume we keep 3 coins -> indices [4,3,1]
        // Volatility drop default 0.0, so pick highest two -> [4,3]
        assert_eq!(select_coins(&features, &cfg), vec![4, 3]);
    }

    #[test]
    fn applies_volatility_drop_percentage() {
        let features = vec![
            make_feature(0, 1.0, 0.9),
            make_feature(1, 1.0, 0.8),
            make_feature(2, 1.0, 0.7),
            make_feature(3, 1.0, 0.6),
        ];
        let cfg = SelectionConfig {
            max_positions: 2,
            volatility_drop_pct: 0.5,
            ..default_config()
        };
        // Drop top 50% volatility (indices 0 & 1), select from remaining highest -> [2,3]
        assert_eq!(select_coins(&features, &cfg), vec![2, 3]);
    }

    #[test]
    fn ensures_min_positions_even_with_high_drop() {
        let features = vec![make_feature(0, 0.1, 0.1), make_feature(1, 0.2, 0.2)];
        let cfg = SelectionConfig {
            max_positions: 2,
            volume_drop_pct: 0.9,
            volatility_drop_pct: 0.9,
            ..default_config()
        };
        assert_eq!(select_coins(&features, &cfg), vec![1, 0]);
    }
}
