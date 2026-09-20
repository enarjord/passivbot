//! Pure replay of revised scoped permission from a reconstructed episode trace.
//! No persisted or previous controller state is accepted as authority.

use crate::hsl_revised::{signal, Observation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub timestamp: i64,
    pub pnl: f64,
    pub upnl: f64,
    pub exposed: bool,
    /// A supported scope flatten after this risk observation, never an
    /// artificial flat from an ambiguous quantity estimate.
    pub flatten: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Episode {
    pub points: Vec<Point>,
    pub entry_reference: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Restart {
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Intervention {
    Panic,
    Normal,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub episodes: Vec<Episode>,
    pub now: i64,
    pub start: i64,
    pub budget: f64,
    pub span: f64,
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub restart: Restart,
    pub intervention: Intervention,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Normal,
    Panic,
    Halted,
}

#[derive(Debug, Serialize)]
pub struct Decision {
    pub timestamp: i64,
    pub action: Action,
    pub red_at: Option<i64>,
    pub flat_at: Option<i64>,
    pub reason: &'static str,
    pub raw: f64,
    pub ema: f64,
    pub numeric_range_approximation: bool,
}

fn cooldown_finished(flat: Option<i64>, timestamp: i64, input: &Input) -> bool {
    input.restart == Restart::Always
        && flat
            .and_then(|f| f.checked_add(input.cooldown_ms))
            .is_some_and(|deadline| timestamp >= deadline)
}

pub fn replay(input: &Input) -> Result<Vec<Decision>, String> {
    if input.start > input.now
        || input.cooldown_ms < 0
        || !input.budget.is_finite()
        || input.budget <= 0.0
        || !input.span.is_finite()
        || input.span < 1.0
        || !input.threshold.is_finite()
        || !(0.0..=1.0).contains(&input.threshold)
    {
        return Err("invalid revised HSL controller configuration".into());
    }
    let mut previous_time = None;
    let mut previous_flat = false;
    for episode in &input.episodes {
        if episode.points.is_empty() {
            return Err("empty HSL episode".into());
        }
        if previous_time.is_some() && !previous_flat {
            return Err("episode reset without supported flatten".into());
        }
        for (index, point) in episode.points.iter().enumerate() {
            if previous_time.is_some_and(|t| point.timestamp < t) {
                return Err("unordered HSL trace".into());
            }
            if point.flatten && (point.exposed || index + 1 != episode.points.len()) {
                return Err("invalid HSL flatten point".into());
            }
            previous_time = Some(point.timestamp);
        }
        previous_flat = episode.points.last().unwrap().flatten;
    }
    let mut decisions = Vec::new();
    let mut red_at = None;
    let mut flat_at = None;
    for episode in &input.episodes {
        let points: Vec<_> = episode
            .points
            .iter()
            .enumerate()
            .filter(|(_, p)| input.start <= p.timestamp && p.timestamp <= input.now)
            .collect();
        if points.is_empty() {
            continue;
        }
        let reference = if points[0].0 == 0 {
            episode.entry_reference
        } else {
            None
        };
        let rows: Vec<_> = points
            .iter()
            .map(|(_, p)| Observation {
                timestamp_ms: p.timestamp,
                realized: p.pnl,
                unrealized: p.upnl,
            })
            .collect();
        let risk = signal(&rows, input.budget, input.span, input.threshold, reference)?;
        for (index, (_, point)) in points.iter().enumerate() {
            let t = point.timestamp;
            let mut reason = "green";
            if cooldown_finished(flat_at, t, input) {
                red_at = None;
                flat_at = None;
                reason = "cooldown_complete";
            }
            if flat_at.is_some() && point.exposed {
                if input.intervention == Intervention::Normal {
                    red_at = None;
                    flat_at = None;
                    reason = "normal_intervention";
                } else {
                    red_at = Some(t);
                    flat_at = None;
                    reason = "panic_intervention";
                }
            }
            if risk.panic[index] && (point.exposed || point.flatten) && red_at.is_none() {
                red_at = Some(t);
                flat_at = None;
                reason = "drawdown";
            }
            if point.flatten && red_at.is_some() {
                flat_at = Some(t);
                reason = "stop_flattened";
            }
            if cooldown_finished(flat_at, t, input) {
                red_at = None;
                flat_at = None;
                reason = "cooldown_complete";
            }
            let action = if red_at.is_none() {
                Action::Normal
            } else if point.exposed {
                Action::Panic
            } else {
                Action::Halted
            };
            decisions.push(Decision {
                timestamp: t,
                action,
                red_at,
                flat_at,
                reason,
                raw: risk.raw[index],
                ema: risk.ema[index],
                numeric_range_approximation: risk.numeric_range_approximation,
            });
        }
    }
    if decisions.is_empty() {
        return Err("no in-window current trace".into());
    }
    if decisions.last().unwrap().timestamp != input.now {
        return Err("HSL trace must end at the current observation".into());
    }
    Ok(decisions)
}

#[pyfunction]
pub fn hsl_revised_controller(input_json: &str) -> PyResult<String> {
    let input =
        serde_json::from_str(input_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = replay(&input).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_flat_evidence_does_not_invent_cooldown() {
        let input = Input {
            now: 120_000,
            start: 0,
            budget: 1000.0,
            span: 1.0,
            threshold: 0.05,
            cooldown_ms: 60_000,
            restart: Restart::Always,
            intervention: Intervention::Normal,
            episodes: vec![Episode {
                entry_reference: None,
                points: vec![
                    Point {
                        timestamp: 0,
                        pnl: 0.0,
                        upnl: 0.0,
                        exposed: true,
                        flatten: false,
                    },
                    Point {
                        timestamp: 60_000,
                        pnl: 0.0,
                        upnl: -100.0,
                        exposed: true,
                        flatten: false,
                    },
                    Point {
                        timestamp: 120_000,
                        pnl: -100.0,
                        upnl: 0.0,
                        exposed: false,
                        flatten: false,
                    },
                ],
            }],
        };
        let result = replay(&input).unwrap();
        assert_eq!(result.last().unwrap().action, Action::Halted);
        assert_eq!(result.last().unwrap().flat_at, None);
        assert_eq!(result.last().unwrap().red_at, Some(60_000));
    }
}
