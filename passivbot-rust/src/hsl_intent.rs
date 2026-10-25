//! Shared v9 intent and lifecycle diagnostics. No replay policy lives here.
use serde::Serialize;

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
    pub signal_raw_fraction: f64,
    pub signal_ema_fraction: f64,
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
    pub signal_raw_fraction: Option<f64>,
    pub signal_ema_fraction: Option<f64>,
}
