//! Pure replay of revised scoped permission from a reconstructed episode trace.
//! No persisted or previous controller state is accepted as authority.

use crate::hsl_revised::{signal_with_references, Observation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub timestamp: i64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub pnl: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub upnl: f64,
    pub exposed: bool,
    /// A scope flatten selected by the best-effort reconciler after this risk
    /// observation. Its disclosed estimates are consumed without a second veto.
    pub flatten: bool,
    /// Candle-free cashflow peak known at this observation, relative to budget.
    /// A reference updates the peak, without inserting a past EMA sample.
    #[serde(
        default,
        deserialize_with = "crate::hsl_revised_json::optional",
        skip_serializing_if = "Option::is_none"
    )]
    pub cashflow_reference_delta: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Episode {
    pub points: Vec<Point>,
    /// First supported increase after this episode's initial scope-flat seed.
    /// A lifecycle event only: it does not add a price or EMA observation.
    #[serde(default)]
    pub opened_at: Option<i64>,
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub entry_reference: Option<f64>,
    /// Estimated entry value relative to the shared current budget. Only the
    /// incomplete initial episode may carry this reference; it adds no EMA row.
    #[serde(default, deserialize_with = "crate::hsl_revised_json::optional")]
    pub entry_reference_delta: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Restart {
    Always,
    Never,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub episodes: Vec<Episode>,
    pub now: i64,
    pub start: i64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub budget: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub span: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub restart: Restart,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Normal,
    Panic,
    Halted,
}

#[derive(Clone, Debug, Serialize)]
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

/// Reconstructed lifecycle evidence for diagnostics, never prior trading authority.
#[derive(Clone, Debug, Serialize)]
pub struct LifecycleEvent {
    pub timestamp: i64,
    pub kind: &'static str,
    pub red_at: i64,
    pub flat_at: Option<i64>,
    pub reason: &'static str,
    pub raw: Option<f64>,
    pub ema: Option<f64>,
}

pub struct Replay {
    pub decisions: Vec<Decision>,
    pub events: Vec<LifecycleEvent>,
    pub observations: usize,
    pub numeric_range_approximation: bool,
    pub(crate) cursor: Option<Cursor>,
}

#[derive(Clone, Debug)]
pub(crate) struct Cursor {
    pub peak_delta: f64,
    pub first_required: i64,
    pub exposed: bool,
    pub seed: Option<Vec<Episode>>,
    pub threshold_sensitive: bool,
}

fn cooldown_finished(flat: Option<i64>, timestamp: i64, input: &Input) -> bool {
    input.restart == Restart::Always
        && flat
            .and_then(|f| f.checked_add(input.cooldown_ms))
            .is_some_and(|deadline| timestamp >= deadline)
}

pub fn replay(input: &Input) -> Result<Vec<Decision>, String> {
    Ok(replay_with_events(input)?.decisions)
}

pub fn replay_with_events(input: &Input) -> Result<Replay, String> {
    replay_collected::<true, false>(input)
}

/// Evaluate every historical transition, but retain only the current permission.
/// Runtime callers need the count and diagnostics, not a full decision allocation.
pub(crate) fn replay_latest_with_events(input: &Input) -> Result<Replay, String> {
    replay_collected::<false, false>(input)
}

pub(crate) fn replay_latest_with_seed(input: &Input) -> Result<Replay, String> {
    replay_collected::<false, true>(input)
}

fn replay_collected<const KEEP_HISTORY: bool, const SEED: bool>(
    input: &Input,
) -> Result<Replay, String> {
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
        if episode.entry_reference_delta.is_some()
            && (previous_time.is_some() || episode.entry_reference.is_some())
        {
            return Err("entry delta requires only the initial incomplete episode".into());
        }
        if episode.points.is_empty() {
            return Err("empty HSL episode".into());
        }
        if previous_time.is_some() && !previous_flat {
            return Err("episode reset without supported flatten".into());
        }
        if let Some(opened) = episode.opened_at {
            let seed = &episode.points[0];
            if !previous_flat
                || previous_time != Some(seed.timestamp)
                || episode.points.len() < 2
                || seed.exposed
                || seed.flatten
                || opened < seed.timestamp
                || opened > episode.points.last().unwrap().timestamp
                || episode
                    .points
                    .iter()
                    .any(|p| p.exposed && p.timestamp < opened)
            {
                return Err("invalid HSL opening event".into());
            }
        }
        for (index, point) in episode.points.iter().enumerate() {
            if point.timestamp > input.now {
                return Err("future HSL trace observation".into());
            }
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
    let current = input
        .episodes
        .last()
        .and_then(|e| e.points.last())
        .ok_or("no in-window current trace")?;
    let anchor = Observation {
        timestamp_ms: current.timestamp,
        realized: current.pnl,
        unrealized: current.upnl,
    };
    let mut decisions = Vec::new();
    let mut observations = 0;
    let mut numeric_range_approximation = false;
    let mut events = Vec::new();
    let mut red_at = None;
    let mut flat_at = None;
    let mut cursor_peak = None;
    let mut threshold_sensitive = false;
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
        if reference.is_some()
            && (points.len() != 1
                || !points[0].1.exposed
                || points[0].1.flatten
                || points[0].1.timestamp != input.now)
        {
            return Err("entry reference requires the current exposed singleton".into());
        }
        let rows: Vec<_> = points
            .iter()
            .map(|(_, p)| Observation {
                timestamp_ms: p.timestamp,
                realized: p.pnl,
                unrealized: p.upnl,
            })
            .collect();
        let relative_reference = if points[0].0 == 0 {
            episode.entry_reference_delta
        } else {
            None
        };
        let risk = signal_with_references(
            &rows,
            input.budget,
            input.span,
            input.threshold,
            reference,
            relative_reference,
            &points
                .iter()
                .map(|(_, p)| p.cashflow_reference_delta)
                .collect::<Vec<_>>(),
            &anchor,
        )?;
        cursor_peak = risk.last_peak_delta;
        let mut opening = episode.opened_at.filter(|t| *t >= input.start);
        for (index, (original_index, point)) in points.iter().enumerate() {
            let score = risk.raw[index].min(risk.ema[index]);
            threshold_sensitive |=
                (score - input.threshold).abs() <= 1e-12 * score.abs().max(1.0);
            let t = point.timestamp;
            let mut reason = "green";
            // Exposure, including a round trip between price observations, ends
            // the previous cooldown. Historical RED is never a trading latch.
            let reopened = *original_index > 0 && opening.is_some_and(|opened| opened <= t);
            if reopened || point.exposed {
                if flat_at.is_some() {
                    events.push(LifecycleEvent {
                        timestamp: opening.unwrap_or(t),
                        kind: "restart",
                        red_at: red_at.unwrap(),
                        flat_at,
                        reason: "exposure_resumed",
                        raw: None,
                        ema: None,
                    });
                    reason = "exposure_resumed";
                }
                if flat_at.is_some() {
                    red_at = None;
                }
                flat_at = None;
                if reopened {
                    red_at = None;
                    opening = None;
                }
            }
            if point.exposed || point.flatten {
                // Evaluate the terminal accounting sample BEFORE resetting the
                // episode. The closing order's type is deliberately irrelevant.
                if risk.panic[index] {
                    if red_at.is_none() {
                        red_at = Some(t);
                        events.push(LifecycleEvent {
                            timestamp: t,
                            kind: "red",
                            red_at: t,
                            flat_at: None,
                            reason: "drawdown",
                            raw: Some(risk.raw[index]),
                            ema: Some(risk.ema[index]),
                        });
                    }
                    reason = "drawdown";
                } else {
                    red_at = None;
                    reason = "green";
                }
                if point.flatten {
                    flat_at = red_at.map(|_| t);
                    if let Some(red) = red_at {
                        reason = "stop_flattened";
                        events.push(LifecycleEvent {
                            timestamp: t,
                            kind: "flat",
                            red_at: red,
                            flat_at,
                            reason,
                            raw: Some(risk.raw[index]),
                            ema: Some(risk.ema[index]),
                        });
                    }
                }
            }
            if cooldown_finished(flat_at, t, input) {
                events.push(LifecycleEvent {
                    timestamp: t,
                    kind: "restart",
                    red_at: red_at.unwrap(),
                    flat_at,
                    reason: "cooldown_complete",
                    raw: None,
                    ema: None,
                });
                red_at = None;
                flat_at = None;
                reason = "cooldown_complete";
            }
            if !point.exposed && flat_at.is_none() {
                red_at = None;
            }
            let action = if point.exposed && risk.panic[index] {
                Action::Panic
            } else if !point.exposed && flat_at.is_some() {
                Action::Halted
            } else {
                Action::Normal
            };
            observations += 1;
            numeric_range_approximation |= risk.numeric_range_approximation;
            if !KEEP_HISTORY {
                decisions.clear();
            }
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
    let relevant = if current.exposed {
        input.episodes.last().unwrap()
    } else {
        input
            .episodes
            .iter()
            .rev()
            .find(|e| e.points.last().unwrap().flatten)
            .unwrap_or(input.episodes.last().unwrap())
    };
    let first_required = if !current.exposed
        && !relevant
            .points
            .iter()
            .any(|p| p.exposed || p.flatten || p.pnl != current.pnl || p.upnl != 0.0)
    {
        i64::MAX
    } else {
        relevant
            .points
            .iter()
            .find(|p| p.timestamp >= input.start)
            .map_or(input.now, |p| p.timestamp)
    };
    Ok(Replay {
        decisions,
        events,
        observations,
        numeric_range_approximation,
        cursor: cursor_peak.map(|peak_delta| Cursor {
            peak_delta,
            first_required,
            exposed: current.exposed,
            threshold_sensitive,
            seed: if SEED {
                let mut first = relevant.clone();
                // The omitted preceding episode cannot authorize cooldown for
                // current exposure; this event no longer has a predecessor here.
                first.opened_at = None;
                if first
                    .points
                    .iter()
                    .all(|p| !p.exposed && !p.flatten && p.pnl == current.pnl && p.upnl == 0.0)
                {
                    first.points = vec![current.clone()];
                }
                let mut seed = vec![first];
                if !current.exposed && relevant.points.last().unwrap().flatten {
                    let terminal = relevant.points.last().unwrap();
                    let mut flat = terminal.clone();
                    flat.flatten = false;
                    seed.push(Episode {
                        points: vec![flat, current.clone()],
                        opened_at: None,
                        entry_reference: None,
                        entry_reference_delta: None,
                    });
                }
                Some(seed)
            } else {
                None
            },
        }),
    })
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
    fn latest_replay_preserves_events_counts_and_earlier_numeric_diagnostics() {
        for budget in [1.0, 1000.0, f64::MAX] {
            for span in [1.0, 2.5, 10000.0] {
                let mut points: Vec<_> = (0..100)
                    .map(|i| Point {
                        timestamp: i * 60_000,
                        pnl: if i == 0 { f64::MAX } else { 0.0 },
                        upnl: ((i * 79) % 401 - 200) as f64,
                        exposed: true,
                        flatten: false,
                        cashflow_reference_delta: None,
                    })
                    .collect();
                points.last_mut().unwrap().exposed = false;
                points.last_mut().unwrap().flatten = true;
                let input = Input {
                    episodes: vec![Episode {
                        points,
                        opened_at: None,
                        entry_reference: None,
                        entry_reference_delta: None,
                    }],
                    now: 99 * 60_000,
                    start: 0,
                    budget,
                    span,
                    threshold: 0.1,
                    cooldown_ms: 0,
                    restart: Restart::Always,
                };
                let full = replay_with_events(&input).unwrap();
                let latest = replay_latest_with_events(&input).unwrap();
                assert_eq!(latest.decisions.len(), 1);
                assert_eq!(latest.observations, full.decisions.len());
                assert_eq!(
                    latest.numeric_range_approximation,
                    full.decisions.iter().any(|d| d.numeric_range_approximation)
                );
                assert_eq!(
                    serde_json::to_value(latest.decisions.last()).unwrap(),
                    serde_json::to_value(full.decisions.last()).unwrap()
                );
                assert_eq!(
                    serde_json::to_value(&latest.events).unwrap(),
                    serde_json::to_value(&full.events).unwrap()
                );
            }
        }
    }

    #[test]
    fn reporting_keeps_zero_cooldown_stop_without_changing_permission() {
        let input: Input = serde_json::from_value(serde_json::json!({
            "now": 120000, "start": 0, "budget": 1000.0, "span": 1.0,
            "threshold": 0.05, "cooldown_ms": 0, "restart": "always",
            "episodes": [
                {"points": [
                    {"timestamp": 0, "pnl": 0.0, "upnl": 0.0, "exposed": true, "flatten": false},
                    {"timestamp": 60000, "pnl": -100.0, "upnl": 0.0, "exposed": false, "flatten": true}
                ]},
                {"points": [
                    {"timestamp": 60000, "pnl": -100.0, "upnl": 0.0, "exposed": false, "flatten": false},
                    {"timestamp": 120000, "pnl": -100.0, "upnl": 0.0, "exposed": false, "flatten": false}
                ]}
            ]
        })).unwrap();
        let result = replay_with_events(&input).unwrap();
        assert_eq!(result.decisions.last().unwrap().action, Action::Normal);
        assert_eq!(
            result
                .events
                .iter()
                .map(|e| (e.kind, e.timestamp))
                .collect::<Vec<_>>(),
            vec![("red", 60000), ("flat", 60000), ("restart", 60000)]
        );
        let plain = replay(&input).unwrap();
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            serde_json::to_string(&result.decisions).unwrap()
        );
        assert!(result.events[0].raw.unwrap() > 0.05);
        assert!(result.events[2].raw.is_none());
    }

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
            episodes: vec![Episode {
                entry_reference: None,
                entry_reference_delta: None,
                opened_at: None,
                points: vec![
                    Point {
                        timestamp: 0,
                        pnl: 0.0,
                        upnl: 0.0,
                        exposed: true,
                        flatten: false,
                        cashflow_reference_delta: None,
                    },
                    Point {
                        timestamp: 60_000,
                        pnl: 0.0,
                        upnl: -100.0,
                        exposed: true,
                        flatten: false,
                        cashflow_reference_delta: None,
                    },
                    Point {
                        timestamp: 120_000,
                        pnl: -100.0,
                        upnl: 0.0,
                        exposed: false,
                        flatten: false,
                        cashflow_reference_delta: None,
                    },
                ],
            }],
        };
        let result = replay(&input).unwrap();
        assert_eq!(result.last().unwrap().action, Action::Normal);
        assert_eq!(result.last().unwrap().flat_at, None);
        assert_eq!(result.last().unwrap().red_at, None);
    }
}
