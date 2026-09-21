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
    /// A supported scope flatten after this risk observation, never an
    /// artificial flat from an ambiguous quantity estimate.
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
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub budget: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
    pub span: f64,
    #[serde(deserialize_with = "crate::hsl_revised_json::number")]
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
}

fn cooldown_finished(flat: Option<i64>, timestamp: i64, input: &Input) -> bool {
    input.restart == Restart::Always
        && flat
            .and_then(|f| f.checked_add(input.cooldown_ms))
            .is_some_and(|deadline| timestamp >= deadline)
}

fn advance_permissions(
    timestamp: i64,
    exposed: bool,
    input: &Input,
    red_at: &mut Option<i64>,
    flat_at: &mut Option<i64>,
    events: &mut Vec<LifecycleEvent>,
) -> Option<&'static str> {
    if cooldown_finished(*flat_at, timestamp, input) {
        events.push(LifecycleEvent {
            timestamp,
            kind: "restart",
            red_at: red_at.expect("flat stop has RED origin"),
            flat_at: *flat_at,
            reason: "cooldown_complete",
            raw: None,
            ema: None,
        });
        *red_at = None;
        *flat_at = None;
        return Some("cooldown_complete");
    }
    if flat_at.is_some() && exposed {
        let previous_flat = *flat_at;
        *flat_at = None;
        if input.intervention == Intervention::Normal {
            events.push(LifecycleEvent {
                timestamp,
                kind: "restart",
                red_at: red_at.expect("flat stop has RED origin"),
                flat_at: previous_flat,
                reason: "normal_intervention",
                raw: None,
                ema: None,
            });
            *red_at = None;
            return Some("normal_intervention");
        }
        *red_at = Some(timestamp);
        events.push(LifecycleEvent {
            timestamp,
            kind: "red",
            red_at: timestamp,
            flat_at: None,
            reason: "panic_intervention",
            raw: None,
            ema: None,
        });
        return Some("panic_intervention");
    }
    None
}

pub fn replay(input: &Input) -> Result<Vec<Decision>, String> {
    Ok(replay_with_events(input)?.decisions)
}

pub fn replay_with_events(input: &Input) -> Result<Replay, String> {
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
    let mut events = Vec::new();
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
        let mut opening = episode.opened_at.filter(|t| *t >= input.start);
        for (index, (original_index, point)) in points.iter().enumerate() {
            let t = point.timestamp;
            let mut reason = "green";
            // The seed precedes the opening even when timestamps tie. Process
            // the event at its real time, before cooldown expiry at a later bar.
            if *original_index > 0 {
                if let Some(opened) = opening.filter(|opened| *opened <= t) {
                    opening = None;
                    reason = advance_permissions(
                        opened,
                        true,
                        input,
                        &mut red_at,
                        &mut flat_at,
                        &mut events,
                    )
                    .unwrap_or(reason);
                }
            }
            reason = advance_permissions(
                t,
                point.exposed,
                input,
                &mut red_at,
                &mut flat_at,
                &mut events,
            )
            .unwrap_or(reason);
            if risk.panic[index] && (point.exposed || point.flatten) && red_at.is_none() {
                red_at = Some(t);
                flat_at = None;
                reason = "drawdown";
                events.push(LifecycleEvent {
                    timestamp: t,
                    kind: "red",
                    red_at: t,
                    flat_at: None,
                    reason,
                    raw: Some(risk.raw[index]),
                    ema: Some(risk.ema[index]),
                });
            }
            if point.flatten && red_at.is_some() {
                flat_at = Some(t);
                reason = "stop_flattened";
                events.push(LifecycleEvent {
                    timestamp: t,
                    kind: "flat",
                    red_at: red_at.unwrap(),
                    flat_at,
                    reason,
                    raw: Some(risk.raw[index]),
                    ema: Some(risk.ema[index]),
                });
            }
            if cooldown_finished(flat_at, t, input) {
                reason =
                    advance_permissions(t, false, input, &mut red_at, &mut flat_at, &mut events)
                        .expect("completed cooldown emits restart");
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
    Ok(Replay { decisions, events })
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
    fn reporting_keeps_zero_cooldown_stop_without_changing_permission() {
        let input: Input = serde_json::from_value(serde_json::json!({
            "now": 120000, "start": 0, "budget": 1000.0, "span": 1.0,
            "threshold": 0.05, "cooldown_ms": 0, "restart": "always", "intervention": "panic",
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
            intervention: Intervention::Normal,
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
        assert_eq!(result.last().unwrap().action, Action::Halted);
        assert_eq!(result.last().unwrap().flat_at, None);
        assert_eq!(result.last().unwrap().red_at, Some(60_000));
    }
}
