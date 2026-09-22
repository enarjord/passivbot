//! Disposable simulator-only shortcuts over immutable, ordered execution facts.
//! A miss returns to the shared best-effort reconstruction; it never reuses
//! permission after an input change.
use super::revised_runtime::{Policy, Scope};
use super::*;
use crate::hsl_revised_controller::Action;

pub(super) type Cutoffs =
    std::collections::BTreeMap<(Option<usize>, Option<usize>), (usize, Option<(i64, usize)>)>;

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
    /// budget/slot change, window clipping or numeric concern rebuilds the same
    /// full reference. Cached results never survive changed trading inputs.
    pub(super) fn advance_revised_scope(
        &self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        policy: &Policy,
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
        let mut total = crate::hsl_revised_sum::CurrencySum::new();
        let mut values = Vec::new();
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
                values.push(value);
            }
        }
        if cursor.exposed != !values.is_empty() {
            return None;
        }
        let upnl = if values.len() == 1 {
            values[0]
        } else {
            for value in values {
                total.add(value);
            }
            let mut reasons = std::collections::BTreeSet::new();
            let value = total.value(&mut reasons);
            if !reasons.is_empty() {
                return None;
            }
            value
        };
        let mut next = previous.clone();
        next.timestamp = now;
        next.result.events.clear();
        let d = next.result.decision.as_mut()?;
        d.timestamp = now;
        if cursor.exposed {
            let peak_delta = cursor.peak_delta.max(upnl);
            let denominator = budget + peak_delta;
            if !denominator.is_finite() || denominator <= 0.0 {
                return None;
            }
            let raw = (peak_delta - upnl) / denominator;
            let alpha = 2.0 / (policy.ema_span_minutes + 1.0);
            let ema = alpha * raw + (1.0 - alpha) * prior.ema;
            let score = raw.min(ema);
            // Let the batch reference adjudicate comparisons near RED, including
            // very small configured thresholds and cancellation-sensitive values.
            if !raw.is_finite()
                || !ema.is_finite()
                || (score - policy.red_threshold).abs() <= 1e-12 * score.abs().max(1.0)
            {
                return None;
            }
            d.raw = raw;
            d.ema = ema;
            d.flat_at = None;
            if score > policy.red_threshold {
                d.action = Action::Panic;
                d.reason = "drawdown";
                if d.red_at.is_none() {
                    d.red_at = Some(now);
                    next.result
                        .events
                        .push(crate::hsl_revised_controller::LifecycleEvent {
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
            next.result.cursor.as_mut()?.peak_delta = peak_delta;
        } else {
            d.reason = "green";
            if d.action == Action::Halted
                && policy.restart_after_red_policy.as_deref() == Some("always")
                && d.flat_at.is_some_and(|flat| {
                    now >= flat + (policy.cooldown_minutes_after_red * 60_000.0).round() as i64
                })
            {
                next.result
                    .events
                    .push(crate::hsl_revised_controller::LifecycleEvent {
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
        next.result.observations += 1;
        Some(next)
    }
}
