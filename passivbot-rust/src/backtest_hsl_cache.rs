//! Disposable simulator-only shortcuts over immutable, ordered execution facts.
//! A miss returns to the shared best-effort reconstruction; it never reuses
//! permission after an input change.
use super::hsl_runtime::{Scope, SignalSettings};
use super::*;
use crate::hsl_controller::Action;

pub(super) type Cutoffs =
    std::collections::BTreeMap<(Option<usize>, Option<usize>), (usize, Option<(i64, usize)>)>;

pub(super) type Traces = std::collections::BTreeMap<(Option<usize>, Option<usize>), Trace>;
pub(super) struct Trace {
    episodes: Vec<crate::hsl_controller::Episode>,
    realized: crate::hsl_sum::CurrencySum,
    cashflow: f64,
    fill_count: usize,
    reference: Option<f64>,
    rolling: Option<RollingWindow>,
    green: Option<GreenWindow>,
    flat: Option<FlatWindow>,
    estimated: Option<EstimatedWindow>,
}
/// A fresh singleton is reconstructed at each capture; only the completed
/// GREEN facts are reused. No historical duration is assigned to the position.
struct EstimatedWindow {
    budget: f64,
    span: f64,
    threshold: f64,
}
/// A closed execution tape has stable reconstructed quantities/bases until
/// its first execution expires. A whole-window equity bound proves every suffix
/// GREEN, so no hidden flatten/cooldown transition can appear as prices expire.
struct FlatWindow {
    first_fill: i64,
    budget: f64,
    span: f64,
    threshold: f64,
    cooldown_ms: i64,
    restart: crate::hsl_controller::Restart,
}
/// Proof that shifting this unchanged execution window preserves its reconstructed
/// inventory and basis. It expires before any retained fill leaves the window.
/// The current endpoint is authoritative; earlier samples use reconstructed basis.
pub(super) struct RollingWindow {
    coin: usize,
    side: usize,
    first_fill: i64,
    position: crate::hsl_history::Position,
    historical_size: f64,
    historical_basis: f64,
    seed: Option<Vec<crate::hsl_controller::Episode>>,
}

/// Disposable prefix state. Unchanged running peaks and comparison margins
/// prove surviving historical decisions. The authoritative endpoint is excluded until
/// the next capture replaces it with its historical valuation.
struct GreenWindow {
    budget: f64,
    span: f64,
    anchor: f64,
    reference: Option<f64>,
    peak: f64,
    peak_time: i64,
    ema: f64,
    first_raw: f64,
    first_delta: f64,
    updates: usize,
    threshold: f64,
    tolerance: f64,
    runs: Vec<RedRun>,
}
struct RedRun {
    start: i64,
    end: i64,
    raw: f64,
    ema: f64,
}
type RollingSignal = (
    f64,
    f64,
    f64,
    Option<i64>,
    Vec<crate::hsl_controller::LifecycleEvent>,
);
impl GreenWindow {
    fn raw(budget: f64, peak: f64, delta: f64) -> Option<f64> {
        let denominator = budget + peak;
        (denominator.is_finite() && denominator > 0.0)
            .then(|| (peak - delta) / denominator)
            .filter(|v| v.is_finite())
    }
    fn observe(&mut self, timestamp: i64, raw: f64, ema: f64, index: usize, beta: f64) {
        if raw > self.threshold {
            let weight = beta.powf(index as f64);
            if weight > 0.0 {
                self.tolerance = self.tolerance.min((ema - self.threshold).abs() / weight);
            }
        }
        if raw.min(ema) > self.threshold {
            if let Some(run) = self.runs.last_mut().filter(|r| r.end + 60_000 == timestamp) {
                run.end = timestamp;
            } else {
                self.runs.push(RedRun {
                    start: timestamp,
                    end: timestamp,
                    raw,
                    ema,
                });
            }
        }
    }
    fn seed(
        episode: &crate::hsl_controller::Episode,
        budget: f64,
        settings: SignalSettings,
    ) -> Option<Self> {
        let points = &episode.points;
        if points.len() < 3 || !settings.span.is_finite() || settings.span < 1.0 {
            return None;
        }
        let anchor = points.last()?.pnl;
        let reference = episode.entry_reference_delta;
        let first = (points[0].pnl - anchor) + points[0].upnl;
        let mut state = Self {
            budget,
            span: settings.span,
            anchor,
            reference,
            peak: reference.map_or(first, |v| v.max(first)),
            peak_time: points[0].timestamp,
            ema: 0.0,
            first_raw: 0.0,
            first_delta: first,
            updates: 0,
            threshold: settings.threshold,
            tolerance: f64::INFINITY,
            runs: Vec::new(),
        };
        let alpha = 2.0 / (settings.span + 1.0);
        for (i, point) in points[..points.len() - 1].iter().enumerate() {
            if point.flatten
                || point.cashflow_reference_delta.is_some()
                || !point.pnl.is_finite()
                || !point.upnl.is_finite()
                || point.pnl.abs() > 1e100
                || point.upnl.abs() > 1e100
                || (i > 0 && point.timestamp != points[i - 1].timestamp + 60_000)
            {
                return None;
            }
            let delta = (point.pnl - anchor) + point.upnl;
            if delta >= state.peak {
                state.peak = delta;
                state.peak_time = point.timestamp;
            }
            let raw = Self::raw(budget, state.peak, delta)?;
            // A reconstructed flat prefix must be identically GREEN before
            // the observed opening. It contributes only zero EMA samples.
            if !point.exposed && (raw != 0.0 || state.ema != 0.0) {
                return None;
            }
            if i == 0 {
                state.ema = raw;
                state.first_raw = raw;
            } else {
                state.ema = alpha * raw + (1.0 - alpha) * state.ema;
            }
            state.observe(point.timestamp, raw, state.ema, i, 1.0 - alpha);
        }
        let last = points.last()?;
        if last.timestamp != points[points.len() - 2].timestamp + 60_000 || state.tolerance <= 1e-10
        {
            return None;
        }
        Some(state)
    }
    /// Extend the historical prefix, then remove at most one expired sample.
    /// Removing its EMA seed is a geometric correction, valid only when every
    /// surviving running peak is unchanged. Otherwise the reference reseeds us.
    fn advance(
        &mut self,
        episode: &crate::hsl_controller::Episode,
        historical: f64,
        expired: usize,
        old_len: usize,
        budget: f64,
        settings: SignalSettings,
    ) -> Option<RollingSignal> {
        let points = &episode.points;
        if self.threshold.to_bits() != settings.threshold.to_bits()
            || self.span.to_bits() != settings.span.to_bits()
            || self.budget.to_bits() != budget.to_bits()
            || self.updates >= 1024
            || expired > 1
            || points.len() < 3
            || self.anchor.to_bits() != points.last()?.pnl.to_bits()
            || self.reference != episode.entry_reference_delta
        {
            return None;
        }
        let prior = &points[points.len() - 2];
        let delta = (prior.pnl - self.anchor) + historical;
        if delta >= self.peak {
            self.peak = delta;
            self.peak_time = prior.timestamp;
        }
        let alpha = 2.0 / (settings.span + 1.0);
        let beta = 1.0 - alpha;
        let historical_raw = Self::raw(budget, self.peak, delta)?;
        self.ema = alpha * historical_raw + beta * self.ema;
        self.observe(prior.timestamp, historical_raw, self.ema, old_len - 1, beta);
        if expired == 1 {
            let first = &points[0];
            let first_delta = (first.pnl - self.anchor) + first.upnl;
            let initial = self.reference.map_or(first_delta, |v| v.max(first_delta));
            // A lost maximum would alter subsequent raw samples. A vanished
            // first peak may only be replaced by this new initial floor.
            if self.peak_time < first.timestamp && self.peak > initial {
                return None;
            }
            if self.first_delta > initial {
                return None;
            }
            self.first_delta = first_delta;
            let new_raw = Self::raw(budget, initial, first_delta)?;
            let correction = new_raw - self.first_raw;
            // Normalizing every comparison margin by beta^index bounds the
            // effect of a changed EMA seed without revisiting distant samples.
            // beta=0 makes all surviving non-seed EMAs independent of that seed.
            let remaining = if beta == 0.0 {
                f64::INFINITY
            } else {
                beta * self.tolerance
            };
            if remaining <= beta * correction.abs() + 1e-10 {
                return None;
            }
            self.tolerance = remaining - beta * correction.abs();
            self.ema += beta.powf((old_len - 1) as f64) * correction;
            self.runs.retain(|r| r.end >= first.timestamp);
            for run in &mut self.runs {
                if run.start >= first.timestamp {
                    run.ema += correction
                        * beta.powf((((run.start - first.timestamp) / 60_000) + 1) as f64);
                } else {
                    run.start = first.timestamp;
                    run.raw = new_raw;
                    run.ema = new_raw;
                }
            }
            let first_red = new_raw > self.threshold;
            let starts_here = self
                .runs
                .first()
                .is_some_and(|r| r.start == first.timestamp);
            if !first_red && starts_here {
                if self.runs[0].end == first.timestamp {
                    self.runs.remove(0);
                } else {
                    let second = &points[1];
                    let second_delta = (second.pnl - self.anchor) + second.upnl;
                    let second_raw = Self::raw(budget, initial.max(second_delta), second_delta)?;
                    let run = &mut self.runs[0];
                    run.start = second.timestamp;
                    run.raw = second_raw;
                    run.ema = alpha * second_raw + beta * new_raw;
                }
            } else if first_red {
                if starts_here
                    || self
                        .runs
                        .first()
                        .is_some_and(|r| r.start == first.timestamp + 60_000)
                {
                    let run = &mut self.runs[0];
                    run.start = first.timestamp;
                    run.raw = new_raw;
                    run.ema = new_raw;
                } else {
                    self.runs.insert(
                        0,
                        RedRun {
                            start: first.timestamp,
                            end: first.timestamp,
                            raw: new_raw,
                            ema: new_raw,
                        },
                    );
                }
                self.tolerance = self.tolerance.min((new_raw - self.threshold).abs());
            }
            self.first_raw = new_raw;
            if self.peak_time < first.timestamp {
                self.peak = initial;
                self.peak_time = first.timestamp;
            }
        }
        let current = points.last()?.upnl;
        let peak = self.peak.max(current);
        // The current authoritative endpoint is the final sample only: its
        // peak must not be projected backward onto historical drawdowns.
        let raw = Self::raw(budget, peak, current)?;
        let ema = alpha * raw + beta * self.ema;
        let score = raw.min(ema);
        if !ema.is_finite()
            || ema < -1e-12
            || (score - settings.threshold).abs() <= 1e-12 * score.abs().max(1.0)
        {
            return None;
        }
        self.updates += 1;
        let red = score > self.threshold;
        let prefix_red = self
            .runs
            .last()
            .filter(|r| r.end == prior.timestamp)
            .map(|r| r.start);
        let red_at = red.then_some(prefix_red.unwrap_or(points.last()?.timestamp));
        let mut events: Vec<_> = self
            .runs
            .iter()
            .map(|r| crate::hsl_controller::LifecycleEvent {
                timestamp: r.start,
                kind: "red",
                red_at: r.start,
                flat_at: None,
                reason: "drawdown",
                raw: Some(r.raw),
                ema: Some(r.ema),
            })
            .collect();
        if red && prefix_red.is_none() {
            events.push(crate::hsl_controller::LifecycleEvent {
                timestamp: points.last()?.timestamp,
                kind: "red",
                red_at: points.last()?.timestamp,
                flat_at: None,
                reason: "drawdown",
                raw: Some(raw),
                ema: Some(ema),
            });
        }
        Some((raw, ema, peak, red_at, events))
    }
}

impl Backtest<'_> {
    pub(super) fn rolling_hsl_window(
        &self,
        snapshot: &crate::hsl_snapshot::Input,
    ) -> Option<RollingWindow> {
        use crate::hsl_history::PositionSide;
        // Broader multi-pair reconciliation is deliberately not cached here.
        let selected = crate::hsl_snapshot::select(snapshot).ok()?;
        if selected.len() != 1 {
            return None;
        }
        let pair = selected[0];
        let side = if pair.position.pside == PositionSide::Long {
            LONG
        } else {
            SHORT
        };
        let direction = if side == LONG { 1.0 } else { -1.0 };
        if pair.position.size == 0.0 || pair.fills.is_empty()
            || pair.fills.iter().any(|f| {
                !f.delta.is_some_and(|q| q.is_finite() && q != 0.0)
                    || !f.price.is_some_and(|p| p.is_finite() && p > 0.0)
                    || !f.realized.is_some_and(f64::is_finite)
                    || !f.fee.is_some_and(f64::is_finite)
            })
            // A current fill changes phase on the following observation. Seed
            // only after all retained executions have settled into history.
            || pair.fills.last()?.timestamp >= snapshot.now - 60_000
        {
            return None;
        }
        let prepared = crate::hsl_snapshot::prepare(snapshot).ok()?;
        if prepared.pairs.len() != 1 {
            return None;
        }
        let history = &prepared.pairs[0].history;
        let last = history.events.last()?;
        let coin = self
            .backtest_params
            .coins
            .iter()
            .position(|c| c == &pair.symbol)?;
        if last.after < 0.0
            || (last.after == 0.0 && !history.reasons.contains("estimated_current_opening"))
            || history
                .reconciliation
                .as_ref()
                .is_some_and(|r| r.applied_at != snapshot.now)
        {
            return None;
        }
        let historical_size = direction * last.after;
        let historical_basis = last.basis;
        // The normal controller cursor intentionally keeps only its latest
        // relevant episode. A suffix proof needs all completed facts, so retain
        // the already reconstructed full tape once, at this cache boundary.
        let seed = crate::hsl_trace::compose_prepared(snapshot, prepared, false).ok()?.episodes;
        Some(RollingWindow {
            coin,
            side,
            first_fill: pair.fills.first()?.timestamp,
            position: pair.position.clone(),
            historical_size,
            historical_basis,
            seed: Some(seed),
        })
    }
}

impl Trace {
    fn append(&mut self, now: i64, upnl: f64, exposed: bool) {
        let episode = self.episodes.last_mut().unwrap();
        if !exposed {
            episode.points.truncate(1);
        }
        episode.points.push(crate::hsl_controller::Point {
            timestamp: now,
            pnl: self.cashflow,
            upnl,
            exposed,
            flatten: false,
            cashflow_reference_delta: None,
        });
    }
}

impl Backtest<'_> {
    /// Find the flat prefix immediately before the only episode that can
    /// authorize current panic/cooldown. Simulator executions are globally ordered
    /// and carry the authoritative post-fill position, so no episode inference is
    /// needed here. Cache loss just repeats this factual reverse walk.
    pub(super) fn hsl_history_cutoff(
        &mut self,
        side: Option<usize>,
        coin: Option<usize>,
    ) -> Option<(i64, usize)> {
        let key = (side, coin);
        if let Some((count, cutoff)) = self.hsl_cutoffs.get(&key) {
            // Only executions belonging to this scope can change its flat
            // boundary. Other sides/coins leave the selected positions intact.
            if *count <= self.fills.len()
                && self.fills[*count..].iter().all(|fill| {
                    let s = if fill.order_type.is_long() {
                        LONG
                    } else {
                        SHORT
                    };
                    side.is_some_and(|v| v != s)
                        || coin.is_some_and(|c| self.backtest_params.coins[c] != fill.coin)
                })
            {
                let cutoff = *cutoff;
                let cutoff = cutoff.map(|(timestamp, consumed)| {
                    // No selected execution has ever occurred: advance the
                    // fill-only prefix without changing the lookback timestamp.
                    if timestamp == i64::MIN {
                        (timestamp, self.fills.len())
                    } else {
                        (timestamp, consumed)
                    }
                });
                self.hsl_cutoffs.insert(key, (self.fills.len(), cutoff));
                return cutoff;
            }
        }
        let mut sizes = [
            self.positions
                .long
                .iter()
                .map(|p| p.size)
                .collect::<Vec<_>>(),
            self.positions
                .short
                .iter()
                .map(|p| p.size)
                .collect::<Vec<_>>(),
        ];
        let selected =
            |s: usize, c: usize| side.is_none_or(|v| v == s) && coin.is_none_or(|v| v == c);
        let mut exposed = (0..self.n_coins)
            .map(|c| {
                [LONG, SHORT]
                    .into_iter()
                    .filter(|&s| selected(s, c) && sizes[s][c] != 0.0)
                    .count()
            })
            .sum::<usize>();
        // Flat now: keep its just-completed episode, starting at the preceding
        // flatten. Exposed now: start at the most recent flatten.
        let needed = if exposed == 0 { 2 } else { 1 };
        let mut boundaries = 0;
        let mut cutoff = None;
        for (index, fill) in self.fills.iter().enumerate().rev() {
            let s = if fill.order_type.is_long() {
                LONG
            } else {
                SHORT
            };
            if side.is_some_and(|v| v != s) {
                continue;
            }
            let c = match coin {
                Some(c) if fill.coin != self.backtest_params.coins[c] => continue,
                Some(c) => c,
                None => self
                    .backtest_params
                    .coins
                    .iter()
                    .position(|c| *c == fill.coin)
                    .expect("simulator fill names its own dataset coin"),
            };
            if !selected(s, c) {
                continue;
            }
            let before = round_(
                fill.position_size - fill.fill_qty,
                self.exchange_params_list[c].qty_step,
            );
            if exposed == 0 && before != 0.0 {
                boundaries += 1;
                if boundaries == needed {
                    cutoff = Some((fill.timestamp_ms as i64, index + 1));
                    break;
                }
            }
            exposed = exposed - usize::from(sizes[s][c] != 0.0) + usize::from(before != 0.0);
            sizes[s][c] = before;
        }
        if cutoff.is_none() {
            // Before the first selected simulator entry the scope is known flat.
            // One preceding zero sample preserves the EMA seed; replaying days
            // of identical zero observations adds no information.
            cutoff = self
                .fills
                .iter()
                .find(|fill| {
                    let s = if fill.order_type.is_long() {
                        LONG
                    } else {
                        SHORT
                    };
                    side.is_none_or(|v| v == s)
                        && coin.is_none_or(|c| self.backtest_params.coins[c] == fill.coin)
                })
                .map(|fill| ((fill.timestamp_ms as i64).saturating_sub(60_000), 0))
                // This scope has no execution history. Skip the unrelated tape
                // but retain the configured valuation/diagnostic window.
                .or(Some((i64::MIN, self.fills.len())));
        }
        self.hsl_cutoffs.insert(key, (self.fills.len(), cutoff));
        cutoff
    }

    fn flat_hsl_window(
        &self,
        key: (Option<usize>, Option<usize>),
        episodes: &[crate::hsl_controller::Episode],
        output: &crate::hsl_evaluator::Output,
        budget: f64,
        settings: SignalSettings,
    ) -> Option<FlatWindow> {
        let decision = output.decision.as_ref()?;
        if output.cursor.as_ref()?.exposed
            || decision.action != Action::Normal
            || decision.raw != 0.0
            || decision.ema != 0.0
            || decision.red_at.is_some()
            || decision.flat_at.is_some()
            || output.reasons.iter().any(|r| {
                !matches!(
                    r.as_str(),
                    "estimated_opening_quantity"
                        | "estimated_opening_basis"
                        | "estimated_entry_peak"
                        | "local_quantity_reconciliation"
                        | "quantity_roundoff"
                )
            })
        {
            return None;
        }
        let start = decision.timestamp
            - (self.backtest_params.pnls_max_lookback_days * 86_400_000.0).round() as i64;
        let in_scope = |f: &&Fill| {
            let side = if f.order_type.is_long() { LONG } else { SHORT };
            f.timestamp_ms as i64 >= start
                && key.0.is_none_or(|s| s == side)
                && key
                    .1
                    .is_none_or(|c| self.backtest_params.coins[c] == f.coin)
        };
        // Most captures follow a fresh execution and cannot prove an idle
        // window. Check the same retained tail before scanning the full tape.
        // Reverse selection preserves the original iterator order even if a
        // caller's timestamps are not sorted.
        let last = self.fills.iter().rfind(&in_scope)?;
        if last.timestamp_ms as i64 >= decision.timestamp - 60_000 || last.position_size != 0.0 {
            return None;
        }
        let mut fills = self.fills.iter().filter(in_scope);
        let first = fills.next()?;
        if first.timestamp_ms as i64 <= start + 60_000
            || std::iter::once(first).chain(fills).any(|f| {
                f.coin != first.coin
                    || f.order_type.is_long() != first.order_type.is_long()
                    || !f.fill_qty.is_finite()
                    || f.fill_qty == 0.0
                    || !f.fill_price.is_finite()
                    || f.fill_price <= 0.0
                    || !f.pnl.is_finite()
                    || !f.fee_paid.is_finite()
            })
        {
            return None;
        }
        let anchor = episodes.last()?.points.last()?.pnl;
        // Every possible reconstructed flatten precedes the last retained
        // execution. With automatic restart, its cooldown has therefore ended
        // by this bound, even if clipping changes earlier hypothetical RED.
        // The ordinary scalar path reports current transitions only; older
        // lifecycle events have already passed the report's watermark.
        let settled = settings.restart == crate::hsl_controller::Restart::Always
            && (last.timestamp_ms as i64)
                .checked_add(settings.cooldown_ms)
                .is_some_and(|end| decision.timestamp > end)
            && output
                .events
                .iter()
                .all(|e| e.timestamp < decision.timestamp);
        if !settled
            && (!output.events.is_empty()
                || !Self::hsl_suffixes_are_green(episodes, budget, settings, anchor))
        {
            return None;
        }
        Some(FlatWindow {
            first_fill: first.timestamp_ms as i64,
            budget,
            span: settings.span,
            threshold: settings.threshold,
            cooldown_ms: settings.cooldown_ms,
            restart: settings.restart,
        })
    }

    /// Append a factual mark to an unchanged reconstructed episode. Any fill,
    /// budget/slot change, window clipping or numeric concern leaves this scalar
    /// shortcut. Changed inputs always receive a fresh controller evaluation.
    pub(super) fn advance_hsl_scope(
        &mut self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        settings: SignalSettings,
    ) -> Option<Scope> {
        let previous = self
            .hsl_scopes
            .iter()
            .find(|s| s.side == side && s.coin == coin)?;
        let cursor = previous.result.cursor.as_ref()?;
        let now = (self.first_timestamp_ms + (k as u64 + 1) * self.interval_ms) as i64;
        let start =
            now - (self.backtest_params.pnls_max_lookback_days * 86_400_000.0).round() as i64;
        let slots = side.map_or(1, |s| self.hard_stop_coin_slot_n_positions(s)) as u64;
        let budget = self
            .hsl_balance_budget(
                side,
                coin,
                slots,
                settings.scale_budget_with_excess_allowance,
            )
            .ok()?;
        let prior = previous.result.decision.as_ref()?;
        let flat = self
            .hsl_traces
            .get(&(side, coin))
            .and_then(|t| t.flat.as_ref())
            .is_some_and(|f| {
                start + 60_000 < f.first_fill
                    && f.span.to_bits() == settings.span.to_bits()
                    && f.threshold.to_bits() == settings.threshold.to_bits()
                    && f.budget.to_bits() == budget.to_bits()
                    && f.cooldown_ms == settings.cooldown_ms
                    && f.restart == settings.restart
            });
        if previous.fill_count != self.fills.len()
            || previous.slots != slots
            || previous.budget.to_bits() != budget.to_bits()
            || !budget.is_finite()
            || budget <= 0.0
            || budget.abs() > 1e100
            || prior.numeric_range_approximation
            // A reconciled current endpoint changes when it becomes history;
            // its EMA cannot be extended as though that endpoint were fixed.
            || previous.result.reasons.contains("current_basis_reconciliation")
            || previous.result.reasons.contains("current_quantity_reconciliation")
            || now != previous.timestamp + 60_000
            || (start > cursor.first_required && !flat)
        {
            return None;
        }
        let (upnl, exposed) = self.hsl_scope_upnl(k, side, coin)?;
        if exposed != cursor.exposed {
            return None;
        }
        let mut decision = prior.clone();
        let mut events = Vec::new();
        let mut next_peak = cursor.peak_delta;
        let d = &mut decision;
        d.timestamp = now;
        if cursor.exposed {
            let peak_delta = cursor.peak_delta.max(upnl);
            let denominator = budget + peak_delta;
            if !denominator.is_finite() || denominator <= 0.0 {
                return None;
            }
            let raw = (peak_delta - upnl) / denominator;
            let alpha = 2.0 / (settings.span + 1.0);
            let ema = alpha * raw + (1.0 - alpha) * prior.ema;
            let score = raw.min(ema);
            // Let the batch reference adjudicate comparisons near RED, including
            // very small configured thresholds and cancellation-sensitive values.
            if !raw.is_finite()
                || !ema.is_finite()
                || (score - settings.threshold).abs() <= 1e-12 * score.abs().max(1.0)
            {
                return None;
            }
            d.raw = raw;
            d.ema = ema;
            d.flat_at = None;
            if score > settings.threshold {
                d.action = Action::Panic;
                d.reason = "drawdown";
                if d.red_at.is_none() {
                    d.red_at = Some(now);
                    events.push(crate::hsl_controller::LifecycleEvent {
                        timestamp: now,
                        kind: "red",
                        red_at: now,
                        flat_at: None,
                        reason: "drawdown",
                        raw: Some(raw),
                        ema: Some(ema),
                    });
                }
            } else {
                d.action = Action::Normal;
                d.reason = "green";
                d.red_at = None;
            }
            next_peak = peak_delta;
        } else {
            d.reason = "green";
            if d.action == Action::Halted
                && settings.restart == crate::hsl_controller::Restart::Always
                && d.flat_at
                    .is_some_and(|flat| now >= flat + settings.cooldown_ms)
            {
                events.push(crate::hsl_controller::LifecycleEvent {
                    timestamp: now,
                    kind: "restart",
                    red_at: d.red_at?,
                    flat_at: d.flat_at,
                    reason: "cooldown_complete",
                    raw: None,
                    ema: None,
                });
                d.action = Action::Normal;
                d.reason = "cooldown_complete";
                d.red_at = None;
                d.flat_at = None;
            }
        }
        let trace = self.hsl_traces.get_mut(&(side, coin))?;
        let mut expired = 0;
        let mut first_required = cursor.first_required;
        if flat {
            for episode in &mut trace.episodes {
                if episode.points.last()?.flatten {
                    let count = episode.points.partition_point(|p| p.timestamp < start);
                    episode.points.drain(..count);
                    expired += count;
                    first_required = episode.points.first()?.timestamp;
                }
            }
        }
        trace.append(now, upnl, exposed);
        // All fallible checks precede moving the old result. The outer loop
        // replaces the complete scope batch atomically after successful evaluation.
        let index = self
            .hsl_scopes
            .iter()
            .position(|s| s.side == side && s.coin == coin)
            .unwrap();
        let mut next = self.hsl_scopes.swap_remove(index);
        next.timestamp = now;
        next.result.decision = Some(decision);
        next.result.events = events;
        next.result.cursor.as_mut().unwrap().peak_delta = next_peak;
        next.result.observations = next.result.observations + 1 - expired;
        next.result.cursor.as_mut().unwrap().first_required = first_required;
        Some(next)
    }
    fn hsl_scope_upnl(
        &self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
    ) -> Option<(f64, bool)> {
        let mut total = crate::hsl_sum::CurrencySum::new();
        let mut first = None;
        let mut multiple = false;
        for c in 0..self.n_coins {
            if coin.is_some_and(|v| v != c) {
                continue;
            }
            for s in [LONG, SHORT] {
                if side.is_some_and(|v| v != s) {
                    continue;
                }
                let position = if s == LONG {
                    self.positions.long[c]
                } else {
                    self.positions.short[c]
                };
                if position.size == 0.0 {
                    continue;
                }
                if !self.coin_is_valid_at(c, k) {
                    return None;
                }
                let mark = self.hlcvs_value(k, c, CLOSE);
                if !mark.is_finite() || mark <= 0.0 {
                    return None;
                }
                let value = if s == LONG {
                    calc_pnl_long(
                        position.price,
                        mark,
                        position.size,
                        self.exchange_params_list[c].c_mult,
                    )
                } else {
                    calc_pnl_short(
                        position.price,
                        mark,
                        position.size,
                        self.exchange_params_list[c].c_mult,
                    )
                };
                if !value.is_finite() || value.abs() > 1e100 {
                    return None;
                }
                if let Some(initial) = first {
                    if !multiple {
                        total.add(initial);
                        multiple = true;
                    }
                    total.add(value);
                } else {
                    first = Some(value);
                }
            }
        }
        let exposed = first.is_some();
        let upnl = if !multiple {
            first.unwrap_or(0.0) // Zero only after proving this scope flat.
        } else {
            let mut reasons = std::collections::BTreeSet::new();
            let value = total.value(&mut reasons);
            if !reasons.is_empty() {
                return None;
            }
            value
        };
        Some((upnl, exposed))
    }

    /// Fixed chronological peaks dominate the peaks of every clipped suffix.
    /// With positive equity, lowering a peak can only lower raw drawdown. If
    /// the original raw/EMA pair is strictly GREEN, a clipped initial raw
    /// seed is zero and both raw and EMA can only decrease. Unlike a global range,
    /// this proof never pairs a later high with an earlier low.
    fn hsl_suffixes_are_green(
        episodes: &[crate::hsl_controller::Episode],
        budget: f64,
        settings: SignalSettings,
        anchor: f64,
    ) -> bool {
        for episode in episodes {
            let Some(first) = episode.points.first() else { return false; };
            if episode.entry_reference.is_some() { return false; }
            let initial = (first.pnl - anchor) + first.upnl;
            let mut peak = episode.entry_reference_delta.map_or(initial, |v| v.max(initial));
            let mut ema = None;
            let alpha = 2.0 / (settings.span + 1.0);
            if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 { return false; }
            for point in &episode.points {
                let delta = (point.pnl - anchor) + point.upnl;
                if !delta.is_finite() || delta.abs() > 1e100
                    || !peak.is_finite() || peak.abs() > 1e100
                    || point.cashflow_reference_delta.is_some()
                    || budget + delta <= 0.0 {
                    return false;
                }
                peak = peak.max(delta);
                let Some(raw) = GreenWindow::raw(budget, peak, delta) else { return false; };
                let next = ema.map_or(raw, |v| alpha * raw + (1.0 - alpha) * v);
                // Fresh reconstruction can retain an estimated opening peak
                // even after its oldest price expires. Its new EMA seed is
                // not necessarily zero: require every raw sample GREEN in
                // that episode. Observed-opening episodes lose their reference
                // on clipping and retain the monotonic raw/EMA proof.
                let score = if episode.entry_reference_delta.is_some() { raw } else { raw.min(next) };
                if !next.is_finite() || score >= settings.threshold - 1e-10 {
                    return false;
                }
                ema = Some(next);
            }
        }
        true
    }

    pub(super) fn seed_hsl_trace(
        &mut self,
        key: (Option<usize>, Option<usize>),
        output: &mut crate::hsl_evaluator::Output,
        mut rolling: Option<RollingWindow>,
        budget: f64,
        settings: SignalSettings,
    ) {
        self.hsl_traces.remove(&key);
        // With no observed opening, each capture is a fresh current-position
        // estimate. Extending yesterday's estimate would invent episode age.
        if output.reasons.contains("estimated_current_opening")
            && !rolling.as_ref().is_some_and(|p| p.historical_size == 0.0) {
            return;
        }
        if let Some(episodes) = output.cursor.as_mut().and_then(|c| c.seed.take()) {
            let episodes = rolling.as_mut().and_then(|p| p.seed.take()).unwrap_or(episodes);
            if episodes.iter().any(|e| {
                e.entry_reference.is_some()
                    || e.points
                        .iter()
                        .any(|p| p.cashflow_reference_delta.is_some())
            }) {
                return;
            }
            let rolling = rolling.filter(|_| {
                // Completed episodes may precede the current open episode,
                // but only a strict whole-window bound can prove that none of
                // their surviving suffixes starts RED or a cooldown lifecycle.
                episodes.last().is_some_and(|e| e.points.iter().all(|p| !p.flatten))
                    && episodes[..episodes.len() - 1].iter().all(|e| e.points.last().is_some_and(|p| p.flatten))
                    && Self::hsl_suffixes_are_green(
                        &episodes[..episodes.len() - 1], budget, settings,
                        episodes.last().and_then(|e| e.points.last()).map_or(f64::NAN, |p| p.pnl),
                    )
                    && output.reasons.iter().all(|r| {
                        matches!(
                            r.as_str(),
                            "estimated_opening_quantity"
                                | "estimated_opening_basis"
                                | "estimated_entry_peak"
                                | "current_basis_reconciliation"
                                | "current_quantity_reconciliation"
                                | "local_quantity_reconciliation"
                                | "estimated_current_opening"
                                | "quantity_roundoff"
                        )
                    })
            });
            if rolling.is_none()
                && (output.reasons.contains("current_basis_reconciliation")
                    || output.reasons.contains("current_quantity_reconciliation"))
            {
                return;
            }
            let flat = self.flat_hsl_window(key, &episodes, output, budget, settings);
            let estimated = rolling.as_ref().filter(|p| p.historical_size == 0.0)
                .filter(|_| episodes.last().is_some_and(|e| e.points.len() == 1))
                .map(|_| EstimatedWindow { budget, span: settings.span, threshold: settings.threshold });
            if output.reasons.contains("estimated_current_opening") && estimated.is_none() { return; }
            let reference = episodes[0].entry_reference_delta;
            self.hsl_traces.insert(
                key,
                Trace {
                    episodes,
                    reference,
                    rolling,
                    green: None,
                    flat,
                    estimated,
                    realized: crate::hsl_sum::CurrencySum::new(),
                    cashflow: 0.0,
                    fill_count: self.fills.len(),
                },
            );
        }
    }

    /// Reuse reconstructed facts after cashflow/budget changes, evaluating the
    /// unchanged shared controller again rather than translating its permission.
    pub(super) fn replay_hsl_trace(
        &mut self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        settings: SignalSettings,
    ) -> Option<Scope> {
        use crate::hsl_controller as controller;
        let previous = self
            .hsl_scopes
            .iter()
            .find(|s| s.side == side && s.coin == coin)?;
        let cursor = previous.result.cursor.as_ref()?;
        let now = (self.first_timestamp_ms + (k as u64 + 1) * self.interval_ms) as i64;
        let start =
            now - (self.backtest_params.pnls_max_lookback_days * 86_400_000.0).round() as i64;
        let slots = side.map_or(1, |s| self.hard_stop_coin_slot_n_positions(s)) as u64;
        if coin.is_some() && slots == 0 {
            return None;
        }
        let budget = self
            .hsl_balance_budget(
                side,
                coin,
                slots,
                settings.scale_budget_with_excess_allowance,
            )
            .ok()?;
        let (upnl, exposed) = self.hsl_scope_upnl(k, side, coin)?;
        let rolling = self
            .hsl_traces
            .get(&(side, coin))
            .and_then(|t| {
                let proof = t.rolling.as_ref()?;
                let position = if proof.side == LONG {
                    self.positions.long[proof.coin]
                } else {
                    self.positions.short[proof.coin]
                };
                (t.fill_count == self.fills.len()
                    && start <= proof.first_fill
                    && position.size.to_bits() == proof.position.size.to_bits()
                    && position.price.to_bits() == proof.position.basis.to_bits())
                .then_some(())
            })
            .is_some();
        if (self.hsl_traces.get(&(side, coin)).is_some_and(|t| t.rolling.is_some()) && !rolling)
            || !budget.is_finite()
            || budget <= 0.0
            || budget > 1e100
            || now != previous.timestamp + 60_000
            || (start > cursor.first_required && !rolling)
            || cursor.exposed != exposed
            || previous
                .result
                .decision
                .as_ref()?
                .numeric_range_approximation
        {
            return None;
        }
        // Retained fills may reconstruct a flat tail while the current
        // exchange-equivalent position remains exposed. Recreate its current
        // singleton through the shared controller, never append an estimated
        // exposure to yesterday's historical facts.
        if rolling && self.hsl_traces.get(&(side, coin))?.estimated.is_some() {
            let trace = self.hsl_traces.get_mut(&(side, coin))?;
            let proof = trace.estimated.as_ref()?;
            if proof.budget.to_bits() != budget.to_bits()
                || proof.span.to_bits() != settings.span.to_bits()
                || proof.threshold.to_bits() != settings.threshold.to_bits() { return None; }
            let prior_count = trace.episodes.len() - 1;
            for episode in &mut trace.episodes[..prior_count] {
                let expired = episode.points.partition_point(|p| p.timestamp < start);
                episode.points.drain(..expired);
                if episode.points.is_empty() { return None; }
            }
            let current = trace.episodes.last_mut()?;
            if current.points.len() != 1 { return None; }
            current.points[0].timestamp = now;
            current.points[0].upnl = upnl;
            // The current cashflow is fixed between fills. The estimate has
            // no observed opening and receives a new one-sample EMA each time.
            current.entry_reference_delta = Some(current.points[0].pnl);
            let replay = controller::replay_latest_with_events(&controller::Input {
                episodes: vec![current.clone()], now, start, budget,
                span: settings.span, threshold: settings.threshold,
                cooldown_ms: settings.cooldown_ms, restart: settings.restart,
            }).ok()?;
            if replay.numeric_range_approximation
                || replay.cursor.as_ref()?.threshold_sensitive { return None; }
            let mut next = previous.clone();
            next.timestamp = now;
            next.result.decision = replay.decisions.into_iter().last();
            next.result.events = replay.events;
            next.result.cursor = replay.cursor;
            next.result.observations = trace.episodes.iter().map(|e| e.points.len()).sum();
            let index = self.hsl_scopes.iter().position(|s| s.side == side && s.coin == coin)?;
            self.hsl_scopes.swap_remove(index);
            return Some(next);
        }
        let rolling_mark = if rolling {
            let proof = self.hsl_traces.get(&(side, coin))?.rolling.as_ref()?;
            Some(self.hlcvs_value(k, proof.coin, CLOSE))
        } else {
            None
        };
        let trace = self.hsl_traces.get_mut(&(side, coin))?;
        // Estimated missing-opening tapes can change when new fills arrive.
        // Reconstruct those instead of extending an estimate with new evidence.
        if trace.reference.is_some() && trace.fill_count != self.fills.len() {
            return None;
        }
        for fill in &self.fills[trace.fill_count..] {
            let s = if fill.order_type.is_long() {
                LONG
            } else {
                SHORT
            };
            if side.is_some_and(|v| v != s)
                || coin.is_some_and(|c| self.backtest_params.coins[c] != fill.coin)
            {
                continue;
            }
            trace.realized.add(fill.pnl);
            trace.realized.add(fill.fee_paid);
        }
        let mut reasons = std::collections::BTreeSet::new();
        trace.cashflow = trace.realized.value(&mut reasons);
        if !reasons.is_empty() || trace.cashflow.abs() > budget * 100.0 {
            return None;
        }
        trace.fill_count = self.fills.len();
        let old_len = trace.episodes.last()?.points.len();
        let mut rolling_update = None;
        if rolling {
            if trace.green.is_none() {
                trace.green = GreenWindow::seed(trace.episodes.last()?, budget, settings);
            }
            let prior_count = trace.episodes.len() - 1;
            for episode in &mut trace.episodes[..prior_count] {
                let expired = episode.points.partition_point(|p| p.timestamp < start);
                episode.points.drain(..expired);
                // The first retained execution has not expired, so completed
                // episodes retain their terminal flatten and cannot disappear.
                if episode.points.is_empty() {
                    return None;
                }
            }
            let proof = trace.rolling.as_mut()?;
            let episode = trace.episodes.last_mut()?;
            // Yesterday's authoritative endpoint becomes a historical mark.
            // A missing entry can give it a different reconstructed basis.
            let historical = proof.position.pnl(
                proof.historical_size,
                proof.historical_basis,
                proof.position.mark,
            );
            if !historical.is_finite() || historical.abs() > 1e100 {
                return None;
            }
            episode.points.last_mut()?.upnl = historical;
            let expired = episode.points.partition_point(|p| p.timestamp < start);
            episode.points.drain(..expired);
            rolling_update = Some((historical, expired));
            proof.position.mark = rolling_mark?;
        }
        trace.append(now, upnl, exposed);
        let green_result = rolling_update.and_then(|(historical, expired)| {
            trace.green.as_mut()?.advance(
                trace.episodes.last()?,
                historical,
                expired,
                old_len,
                budget,
                settings,
            )
        });
        if green_result.is_none() {
            trace.green = None;
        }
        if let Some((raw, ema, peak, red_at, events)) = green_result {
            let mut next = previous.clone();
            let d = next.result.decision.as_mut()?;
            d.timestamp = now;
            d.raw = raw;
            d.ema = ema;
            let red = raw.min(ema) > settings.threshold;
            d.action = if red { Action::Panic } else { Action::Normal };
            d.reason = if red { "drawdown" } else { "green" };
            d.red_at = red_at;
            d.flat_at = None;
            next.timestamp = now;
            next.budget = budget;
            next.slots = slots;
            next.result.events = events;
            next.result.observations = trace.episodes.iter().map(|e| e.points.len()).sum();
            let cursor = next.result.cursor.as_mut()?;
            cursor.peak_delta = peak;
            cursor.first_required = trace.episodes.last()?.points[0].timestamp;
            let index = self
                .hsl_scopes
                .iter()
                .position(|s| s.side == side && s.coin == coin)?;
            self.hsl_scopes.swap_remove(index);
            return Some(next);
        }
        let mut episodes = std::mem::take(&mut trace.episodes);
        if let Some(reference) = trace.reference {
            episodes[0].entry_reference_delta = Some(reference - trace.cashflow);
        }
        let input = controller::Input {
            episodes,
            now,
            start,
            budget,
            span: settings.span,
            threshold: settings.threshold,
            cooldown_ms: settings.cooldown_ms,
            restart: settings.restart,
        };
        let replay = controller::replay_latest_with_events(&input);
        // Restore facts even if reference evaluation fails; no decision is cached.
        trace.episodes = input.episodes;
        let replay = replay.ok()?;
        // Numerical reuse must not decide a threshold comparison at rounding
        // distance, including the terminal sample of a now-flat episode.
        if replay.numeric_range_approximation || replay.cursor.as_ref()?.threshold_sensitive {
            return None;
        }
        let decision = replay.decisions.into_iter().last()?;
        let index = self
            .hsl_scopes
            .iter()
            .position(|s| s.side == side && s.coin == coin)
            .unwrap();
        let mut result = self.hsl_scopes.swap_remove(index).result;
        result.decision = Some(decision);
        result.events = replay.events;
        result.observations = replay.observations;
        result.cursor = replay.cursor;
        Some(Scope {
            timestamp: now,
            fill_count: self.fills.len(),
            budget,
            slots,
            side,
            coin,
            result,
        })
    }
}

#[cfg(test)]
mod green_window_tests {
    use super::*;

    #[test]
    fn chronological_green_proof_matches_every_reference_suffix() {
        use crate::hsl_controller::{self as controller, Episode, Point, Restart};
        let mut points = Vec::new();
        for i in 0..21 {
            let pnl = if i % 2 == 0 { (i / 2) as f64 * 1000.0 }
                else { ((i + 1) / 2) as f64 * 1000.0 - 20.0 };
            points.push(Point {
                timestamp: i * 60_000, pnl, upnl: 0.0,
                exposed: i < 20, flatten: i == 20,
                cashflow_reference_delta: None,
            });
        }
        let episode = Episode { points, entry_reference: None,
            entry_reference_delta: None, opened_at: None };
        let settings = SignalSettings { scale_budget_with_excess_allowance: false,
            span: 17.5, threshold: 0.01, cooldown_ms: 300_000, restart: Restart::Always };
        // A global min/max pairs later profit with an earlier low and rejects
        // this tape. Chronological peaks prove its actual suffixes GREEN.
        assert!(GreenWindow::raw(20_000.0, 0.0, -10_000.0).unwrap() > settings.threshold);
        assert!(Backtest::hsl_suffixes_are_green(&[episode.clone()], 20_000.0, settings, 10_000.0));
        for i in 0..21 {
            let reference = controller::replay_latest_with_events(&controller::Input {
                episodes: vec![episode.clone()], now: 20 * 60_000,
                start: i * 60_000, budget: 20_000.0, span: settings.span,
                threshold: settings.threshold, cooldown_ms: settings.cooldown_ms,
                restart: settings.restart,
            }).unwrap();
            assert!(reference.events.is_empty(), "suffix {i}");
            assert!(reference.decisions.iter().all(|d| d.action == Action::Normal));
        }
        let mut bad = episode;
        bad.points[10].upnl = -5000.0;
        assert!(!Backtest::hsl_suffixes_are_green(&[bad], 20_000.0, settings, 10_000.0));
        // Raw may exceed the threshold while a slow EMA stays GREEN. A
        // clipped suffix starts at raw zero, so its EMA cannot exceed the
        // corresponding original EMA either.
        let smooth = SignalSettings { span: 2333.0, ..settings };
        let mut spike = Episode { points: Vec::new(), entry_reference: None,
            entry_reference_delta: None, opened_at: None };
        for i in 0..21 {
            spike.points.push(Point { timestamp: i * 60_000, pnl: 0.0,
                upnl: if i % 2 == 0 && i < 20 { 1000.0 } else { 0.0 },
                exposed: i < 20, flatten: i == 20, cashflow_reference_delta: None });
        }
        assert!(Backtest::hsl_suffixes_are_green(&[spike.clone()], 10_000.0, smooth, 0.0));
        let mut estimated = spike.clone();
        estimated.entry_reference_delta = Some(1000.0);
        assert!(!Backtest::hsl_suffixes_are_green(&[estimated.clone()], 10_000.0, smooth, 0.0));
        // A fresh estimated opening keeps its peak when the first price is
        // removed; a one-sample seed can become RED despite the old slow EMA.
        estimated.points.drain(..1);
        let reconstructed = controller::replay_latest_with_events(&controller::Input {
            episodes: vec![estimated], now: 20 * 60_000, start: 60_000,
            budget: 10_000.0, span: smooth.span, threshold: smooth.threshold,
            cooldown_ms: smooth.cooldown_ms, restart: smooth.restart,
        }).unwrap();
        assert!(!reconstructed.events.is_empty());
        for i in 0..21 {
            let reference = controller::replay_latest_with_events(&controller::Input {
                episodes: vec![spike.clone()], now: 20 * 60_000, start: i * 60_000,
                budget: 10_000.0, span: smooth.span, threshold: smooth.threshold,
                cooldown_ms: smooth.cooldown_ms, restart: smooth.restart,
            }).unwrap();
            assert!(reference.events.is_empty(), "smoothed suffix {i}");
        }
    }

    use crate::hsl_controller::{self as controller, Episode, Point, Restart};

    fn point(timestamp: i64, upnl: f64) -> Point {
        Point {
            timestamp,
            pnl: 0.0,
            upnl,
            exposed: true,
            flatten: false,
            cashflow_reference_delta: None,
        }
    }

    #[test]
    fn rolling_prefix_matches_reference_with_reconciled_endpoints() {
        for threshold in [0.025, 0.9] {
            for offset in [-90.0, 10.0] {
                for span in [1.0, 17.5, 2754.0] {
                    for reference in [Some(0.0), None] {
                        let settings = SignalSettings {
                            span,
                            threshold,
                            cooldown_ms: 0,
                            restart: Restart::Always,
                            scale_budget_with_excess_allowance: false,
                        };
                        let historical = |i: i64| offset + ((i * 17) % 53) as f64;
                        let mut episode = Episode {
                            opened_at: None,
                            entry_reference: None,
                            entry_reference_delta: reference,
                            points: (0..64).map(|i| point(i * 60_000, historical(i))).collect(),
                        };
                        episode.points.last_mut().unwrap().upnl = 100.0;
                        let mut state = None;
                        let mut hits = 0;
                        for i in 64..2300 {
                            if state.is_none() {
                                state = GreenWindow::seed(&episode, 1000.0, settings);
                            }
                            let old_len = episode.points.len();
                            let old = historical(i - 1);
                            episode.points.last_mut().unwrap().upnl = old;
                            let expired = usize::from(i >= 80);
                            if expired == 1 {
                                episode.points.remove(0);
                            }
                            episode.points.push(point(
                                i * 60_000,
                                if i % 17 == 0 {
                                    -950.0
                                } else {
                                    100.0 + (i % 13) as f64
                                },
                            ));
                            let fast = state.as_mut().and_then(|s| {
                                s.advance(&episode, old, expired, old_len, 1000.0, settings)
                            });
                            let input = controller::Input {
                                episodes: vec![episode.clone()],
                                now: i * 60_000,
                                start: episode.points[0].timestamp,
                                budget: 1000.0,
                                span,
                                threshold: settings.threshold,
                                cooldown_ms: 0,
                                restart: Restart::Always,
                            };
                            let expected = controller::replay_latest_with_events(&input).unwrap();
                            if let Some((raw, ema, peak, red_at, events)) = fast {
                                hits += 1;
                                let d = expected.decisions.last().unwrap();
                                let red = raw.min(ema) > settings.threshold;
                                assert_eq!(
                                    d.action,
                                    if red { Action::Panic } else { Action::Normal }
                                );
                                assert_eq!(d.red_at, red_at);
                                assert_eq!(expected.events.len(), events.len());
                                for (a, b) in events.iter().zip(&expected.events) {
                                    assert_eq!(a.timestamp, b.timestamp);
                                    assert!((a.raw.unwrap() - b.raw.unwrap()).abs() < 1e-12);
                                    assert!((a.ema.unwrap() - b.ema.unwrap()).abs() < 1e-12);
                                }
                                assert!((raw - d.raw).abs() < 1e-12, "raw {raw} != {}", d.raw);
                                assert!(
                                    (ema - d.ema).abs() < 1e-12,
                                    "span={span} i={i} ema {ema} != {}",
                                    d.ema
                                );
                                assert_eq!(peak, expected.cursor.unwrap().peak_delta);
                            } else {
                                state = None;
                            }
                        }
                        assert!(
                            hits > if threshold == 0.9 { 1000 } else { 50 },
                            "span={span}, reference={reference:?}, hits={hits}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rolling_prefix_rejects_risk_changes_and_unsupported_inputs() {
        let settings = SignalSettings {
            span: 17.5,
            threshold: 0.9,
            cooldown_ms: 0,
            restart: Restart::Always,
            scale_budget_with_excess_allowance: false,
        };
        let mut episode = Episode {
            opened_at: None,
            entry_reference: None,
            entry_reference_delta: Some(0.0),
            points: (0..8).map(|i| point(i * 60_000, -20.0)).collect(),
        };
        let mut cache = GreenWindow::seed(&episode, 1000.0, settings).unwrap();
        episode.points.push(point(8 * 60_000, -20.0));
        assert!(cache
            .advance(&episode, -20.0, 0, 8, 999.0, settings)
            .is_none());
        let mut cache = GreenWindow::seed(&episode, 1000.0, settings).unwrap();
        assert!(cache
            .advance(&episode, -20.0, 2, 9, 1000.0, settings)
            .is_none());
        for point in &mut episode.points {
            point.upnl = -900.00000001;
        }
        assert!(GreenWindow::seed(&episode, 1000.0, settings).is_none());
        for point in &mut episode.points {
            point.upnl = -20.0;
        }
        episode.points.last_mut().unwrap().upnl = -20.0;
        episode.points[1].timestamp += 1;
        assert!(GreenWindow::seed(&episode, 1000.0, settings).is_none());
    }
}
