//! Disposable simulator-only shortcuts over immutable, ordered execution facts.
//! A miss returns to the shared best-effort reconstruction; it never reuses
//! permission after an input change.
use super::revised_runtime::{Scope, SignalSettings};
use super::*;
use crate::hsl_revised_controller::Action;

pub(super) type Cutoffs =
    std::collections::BTreeMap<(Option<usize>, Option<usize>), (usize, Option<(i64, usize)>)>;

pub(super) type Traces = std::collections::BTreeMap<(Option<usize>, Option<usize>), Trace>;
pub(super) struct Trace {
    episodes: Vec<crate::hsl_revised_controller::Episode>,
    realized: crate::hsl_revised_sum::CurrencySum,
    cashflow: f64,
    fill_count: usize,
    reference: Option<f64>,
}
impl Trace {
    fn append(&mut self, now: i64, upnl: f64, exposed: bool) {
        let episode = self.episodes.last_mut().unwrap();
        if !exposed {
            episode.points.truncate(1);
        }
        episode.points.push(crate::hsl_revised_controller::Point {
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
    pub(super) fn revised_history_cutoff(
        &mut self,
        side: Option<usize>,
        coin: Option<usize>,
    ) -> Option<(i64, usize)> {
        let key = (side, coin);
        if let Some((count, cutoff)) = self.revised_hsl_cutoffs.get(&key) {
            if *count == self.fills.len() {
                return *cutoff;
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
                .map(|fill| ((fill.timestamp_ms as i64).saturating_sub(60_000), 0));
        }
        self.revised_hsl_cutoffs
            .insert(key, (self.fills.len(), cutoff));
        cutoff
    }

    /// Append a factual mark to an unchanged reconstructed episode. Any fill,
    /// budget/slot change, window clipping or numeric concern leaves this scalar
    /// shortcut. Changed inputs always receive a fresh controller evaluation.
    pub(super) fn advance_revised_scope(
        &mut self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        settings: SignalSettings,
    ) -> Option<Scope> {
        let previous = self
            .revised_hsl_scopes
            .iter()
            .find(|s| s.side == side && s.coin == coin)?;
        let cursor = previous.result.cursor.as_ref()?;
        let now = (self.first_timestamp_ms + (k as u64 + 1) * self.interval_ms) as i64;
        let start =
            now - (self.backtest_params.pnls_max_lookback_days * 86_400_000.0).round() as i64;
        let slots = side.map_or(1, |s| self.hard_stop_coin_slot_n_positions(s)) as u64;
        let budget = self.balance.usd_total_balance
            / if coin.is_some() {
                slots.max(1) as f64
            } else {
                1.0
            };
        let prior = previous.result.decision.as_ref()?;
        if previous.fill_count != self.fills.len()
            || previous.slots != slots
            || previous.budget.to_bits() != budget.to_bits()
            || !budget.is_finite()
            || budget <= 0.0
            || budget.abs() > 1e100
            || prior.numeric_range_approximation
            || now != previous.timestamp + 60_000
            || start > cursor.first_required
        {
            return None;
        }
        let (upnl, exposed) = self.revised_scope_upnl(k, side, coin)?;
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
                    events.push(crate::hsl_revised_controller::LifecycleEvent {
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
                && settings.restart == crate::hsl_revised_controller::Restart::Always
                && d.flat_at
                    .is_some_and(|flat| now >= flat + settings.cooldown_ms)
            {
                events.push(crate::hsl_revised_controller::LifecycleEvent {
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
        self.revised_hsl_traces
            .get_mut(&(side, coin))?
            .append(now, upnl, exposed);
        // All fallible checks precede moving the old result. The outer loop
        // replaces the complete scope batch atomically after successful evaluation.
        let index = self
            .revised_hsl_scopes
            .iter()
            .position(|s| s.side == side && s.coin == coin)
            .unwrap();
        let mut next = self.revised_hsl_scopes.swap_remove(index);
        next.timestamp = now;
        next.result.decision = Some(decision);
        next.result.events = events;
        next.result.cursor.as_mut().unwrap().peak_delta = next_peak;
        next.result.observations += 1;
        Some(next)
    }
    fn revised_scope_upnl(
        &self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
    ) -> Option<(f64, bool)> {
        let mut total = crate::hsl_revised_sum::CurrencySum::new();
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

    pub(super) fn seed_revised_trace(
        &mut self,
        key: (Option<usize>, Option<usize>),
        output: &mut crate::hsl_revised_evaluator::Output,
    ) {
        self.revised_hsl_traces.remove(&key);
        // With no observed opening, each capture is a fresh current-position
        // estimate. Extending yesterday's estimate would invent episode age.
        if output.reasons.contains("estimated_current_opening") {
            return;
        }
        if let Some(episodes) = output.cursor.as_mut().and_then(|c| c.seed.take()) {
            if episodes.iter().any(|e| {
                e.entry_reference.is_some()
                    || e.points
                        .iter()
                        .any(|p| p.cashflow_reference_delta.is_some())
            }) {
                return;
            }
            let reference = episodes[0].entry_reference_delta;
            self.revised_hsl_traces.insert(
                key,
                Trace {
                    episodes,
                    reference,
                    realized: crate::hsl_revised_sum::CurrencySum::new(),
                    cashflow: 0.0,
                    fill_count: self.fills.len(),
                },
            );
        }
    }

    /// Reuse reconstructed facts after cashflow/budget changes, evaluating the
    /// unchanged shared controller again rather than translating its permission.
    pub(super) fn replay_revised_trace(
        &mut self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        settings: SignalSettings,
    ) -> Option<Scope> {
        use crate::hsl_revised_controller as controller;
        let previous = self
            .revised_hsl_scopes
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
        let budget =
            self.balance.usd_total_balance / if coin.is_some() { slots as f64 } else { 1.0 };
        let (upnl, exposed) = self.revised_scope_upnl(k, side, coin)?;
        if !budget.is_finite()
            || budget <= 0.0
            || budget > 1e100
            || now != previous.timestamp + 60_000
            || start > cursor.first_required
            || cursor.exposed != exposed
            || previous
                .result
                .decision
                .as_ref()?
                .numeric_range_approximation
        {
            return None;
        }
        let trace = self.revised_hsl_traces.get_mut(&(side, coin))?;
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
        trace.append(now, upnl, exposed);
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
            .revised_hsl_scopes
            .iter()
            .position(|s| s.side == side && s.coin == coin)
            .unwrap();
        let mut result = self.revised_hsl_scopes.swap_remove(index).result;
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
