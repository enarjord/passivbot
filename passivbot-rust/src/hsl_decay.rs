//! Forward-only v9 risk memory. Inventory and exchange accounting stay outside
//! this module; only the explicitly authored risk graph uses binary32.
use crate::hsl_intent::Action;
use serde::Serialize;

const MINUTE_MS: i64 = 60_000;

fn number(value: f64, name: &str) -> Result<f32, String> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() {
        return Err(format!("{name} is outside the finite v9 risk domain"));
    }
    Ok(narrowed)
}

fn coefficients(span: f64) -> Result<(f32, f32), String> {
    let span = number(span, "EMA span")?;
    if span < 1.0 {
        return Err("v9 risk EMA spans must be at least one minute".into());
    }
    let alpha = 2.0_f32 / (span + 1.0_f32);
    let beta = 1.0_f32 - alpha;
    if !(0.0..1.0).contains(&beta) || !alpha.is_finite() {
        return Err("v9 risk EMA span rounds to a non-decaying binary32 clock".into());
    }
    Ok((alpha, beta))
}

#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub slow_alpha: f32,
    pub slow_beta: f32,
    pub fast_alpha: f32,
    pub fast_beta: f32,
    pub threshold: f32,
    pub cooldown_ms: i64,
}

impl Policy {
    pub fn new(
        memory_ema_span_minutes: f64,
        drawdown_ratio_ema_span_minutes: f64,
        red_threshold: f64,
        cooldown_ms: i64,
    ) -> Result<Self, String> {
        let (slow_alpha, slow_beta) = coefficients(memory_ema_span_minutes)?;
        let (fast_alpha, fast_beta) = coefficients(drawdown_ratio_ema_span_minutes)?;
        let threshold = number(red_threshold, "HSL threshold")?;
        if !(0.0..=1.0).contains(&threshold) || cooldown_ms < 0 {
            return Err("invalid v9 HSL threshold or cooldown".into());
        }
        Ok(Self {
            slow_alpha,
            slow_beta,
            fast_alpha,
            fast_beta,
            threshold,
            cooldown_ms,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub timestamp_ms: i64,
    pub budget: f64,
    pub upnl: f64,
    pub net_realized_pnl: f64,
    /// Gross scope exposure, never signed/net inventory.
    pub exposed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    pub timestamp_ms: i64,
    pub action: Action,
    pub exposed: bool,
    pub red: bool,
    pub raw: f64,
    pub ema: f64,
    pub drawdown: f64,
    pub cooldown_until: Option<i64>,
    pub flatten_qualified: bool,
    pub nonpositive_equity: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ScopeState {
    minute: Option<i64>,
    last_timestamp_ms: Option<i64>,
    d: f32,
    m: f32,
    minute_prior_m: f32,
    previous_upnl: f32,
    previous_budget: f32,
    exposed: bool,
    cooldown_until: Option<i64>,
    last: Option<Decision>,
}

fn ratio(budget: f32, upnl: f32, d: f32) -> Result<(f32, bool), String> {
    let equity = budget + upnl;
    let peak = equity + d;
    if !equity.is_finite() || !peak.is_finite() {
        return Err("v9 HSL equity or denominator exceeds the finite risk domain".into());
    }
    if equity <= 0.0 || peak <= 0.0 {
        // Known insolvency is a loss signal. A finite diagnostic accompanies a
        // direct RED decision; it is not a fabricated normal ratio.
        return Ok((1.0, true));
    }
    let result = d / peak;
    if !result.is_finite() {
        return Err("non-finite v9 HSL ratio".into());
    }
    Ok((result, false))
}

impl ScopeState {
    fn signal(
        &mut self,
        policy: &Policy,
        budget: f32,
        upnl: f32,
    ) -> Result<(f32, bool, bool), String> {
        let (raw, nonpositive) = ratio(budget, upnl, self.d)?;
        self.m = policy
            .fast_beta
            .mul_add(self.minute_prior_m, policy.fast_alpha * raw);
        if !self.m.is_finite() {
            return Err("non-finite v9 HSL ratio EMA".into());
        }
        Ok((
            raw,
            nonpositive || raw.min(self.m) > policy.threshold,
            nonpositive,
        ))
    }

    fn observe_inner(&mut self, policy: &Policy, event: Event) -> Result<Decision, String> {
        let budget = number(event.budget, "HSL budget")?;
        let upnl = number(event.upnl, "HSL UPNL")?;
        let net = number(event.net_realized_pnl, "HSL realized PnL")?;
        if event.timestamp_ms < 0
            || self
                .last_timestamp_ms
                .is_some_and(|t| event.timestamp_ms < t)
            || (!event.exposed && upnl != 0.0)
        {
            return Err("invalid v9 HSL event ordering, budget or flat UPNL".into());
        }
        let minute = event.timestamp_ms.div_euclid(MINUTE_MS);
        let was_exposed = self.exposed;
        let pre_red = was_exposed && self.last.as_ref().is_some_and(|d| d.red);
        if !was_exposed && !event.exposed {
            // No economics belong to an HSL episode while flat. The continuous
            // account realized memory consumes such independently attributed PnL.
            if net != 0.0 {
                return Err("flat v9 HSL event has cashflow without an episode".into());
            }
            self.last_timestamp_ms = Some(event.timestamp_ms);
            let result = self.current(event.timestamp_ms);
            self.last = Some(result.clone());
            return Ok(result);
        }
        if !was_exposed {
            // True new exposure: reset before consuming entry fees or mark loss.
            self.d = 0.0;
            self.m = 0.0;
            self.minute_prior_m = 0.0;
            self.previous_upnl = 0.0;
            self.minute = None;
            self.cooldown_until = None;
        }
        let new_minute = self.minute != Some(minute);
        if let Some(previous) = self.minute {
            // Bootstrap may bridge elapsed empty minutes using its explicit
            // carried-mark policy. Ordinary replay supplies each minute itself.
            for intermediate in (previous + 1)..minute {
                self.minute_prior_m = self.m;
                self.d = policy.slow_beta.mul_add(self.d, 0.0);
                self.signal(policy, self.previous_budget, self.previous_upnl)?;
                self.minute = Some(intermediate);
            }
        }
        if new_minute {
            self.minute_prior_m = self.m;
        }
        let loss = (self.previous_upnl - upnl) - net;
        if !loss.is_finite() {
            return Err("v9 HSL economic increment exceeds the finite risk domain".into());
        }
        self.d = if new_minute {
            policy.slow_beta.mul_add(self.d, loss).max(0.0)
        } else {
            (self.d + loss).max(0.0)
        };
        if !self.d.is_finite() {
            return Err("v9 HSL drawdown exceeds the finite risk domain".into());
        }
        let (raw, red, nonpositive_equity) = self.signal(policy, budget, upnl)?;
        let flatten_qualified = was_exposed && !event.exposed && (pre_red || red);
        if was_exposed && !event.exposed {
            self.cooldown_until = if flatten_qualified {
                Some(
                    event
                        .timestamp_ms
                        .checked_add(policy.cooldown_ms)
                        .ok_or("v9 HSL cooldown timestamp overflow")?,
                )
            } else {
                None
            };
        }
        self.minute = Some(minute);
        self.last_timestamp_ms = Some(event.timestamp_ms);
        self.previous_upnl = upnl;
        self.previous_budget = budget;
        self.exposed = event.exposed;
        let action = if event.exposed && red {
            Action::Panic
        } else if !event.exposed && self.cooldown_until.is_some_and(|t| event.timestamp_ms < t) {
            Action::Halted
        } else {
            Action::Normal
        };
        let result = Decision {
            timestamp_ms: event.timestamp_ms,
            action,
            exposed: event.exposed,
            red,
            raw: raw as f64,
            ema: self.m as f64,
            drawdown: self.d as f64,
            cooldown_until: self.cooldown_until,
            flatten_qualified,
            nonpositive_equity,
        };
        self.last = Some(result.clone());
        Ok(result)
    }

    /// The caller supplies each economic event once, already canonicalized.
    /// Failed events leave state unchanged, including clocks and cooldown.
    pub fn observe(&mut self, policy: &Policy, event: Event) -> Result<Decision, String> {
        let mut next = self.clone();
        let result = next.observe_inner(policy, event)?;
        *self = next;
        Ok(result)
    }

    /// Current quote assessment is disposable. It never becomes a later
    /// completed-minute/fill sample or retains an unrecorded tick-price peak.
    pub fn preview(&self, policy: &Policy, event: Event) -> Result<Decision, String> {
        if event.exposed != self.exposed || event.net_realized_pnl != 0.0 {
            return Err("v9 HSL preview cannot consume inventory or cashflow facts".into());
        }
        let mut temporary = self.clone();
        temporary.observe(policy, event)
    }

    /// A documented current-basis approximation for an unknown exposed prefix.
    /// This is not an actual episode opening and does not claim exact recovery.
    pub fn seed_unknown(
        &mut self,
        policy: &Policy,
        timestamp_ms: i64,
        budget: f64,
        upnl: f64,
    ) -> Result<Decision, String> {
        let mut next = Self::default();
        next.observe(
            policy,
            Event {
                timestamp_ms,
                budget,
                upnl,
                net_realized_pnl: 0.0,
                exposed: true,
            },
        )?;
        let (raw, nonpositive) = ratio(
            number(budget, "HSL budget")?,
            number(upnl, "HSL UPNL")?,
            next.d,
        )?;
        next.m = raw;
        next.minute_prior_m = raw;
        let mut result = next.last.clone().unwrap();
        result.ema = raw as f64;
        result.red = nonpositive || raw > policy.threshold;
        result.action = if result.red {
            Action::Panic
        } else {
            Action::Normal
        };
        next.last = Some(result.clone());
        *self = next;
        Ok(result)
    }

    /// Permission/expiry query. It consumes neither a minute nor a cashflow.
    pub fn current(&self, now_ms: i64) -> Decision {
        let mut result = self.last.clone().unwrap_or(Decision {
            timestamp_ms: now_ms,
            action: Action::Normal,
            exposed: false,
            red: false,
            raw: 0.0,
            ema: 0.0,
            drawdown: 0.0,
            cooldown_until: self.cooldown_until,
            flatten_qualified: false,
            nonpositive_equity: false,
        });
        result.timestamp_ms = now_ms;
        result.flatten_qualified = false;
        if !self.exposed {
            result.action = if self.cooldown_until.is_some_and(|t| now_ms < t) {
                Action::Halted
            } else {
                Action::Normal
            };
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct RealizedMemory {
    beta: f32,
    minute: i64,
    last_timestamp_ms: i64,
    d: f32,
}

impl RealizedMemory {
    pub fn new(span_minutes: f64, start_ms: i64) -> Result<Self, String> {
        let (_, beta) = coefficients(span_minutes)?;
        if start_ms < 0 {
            return Err("negative v9 realized-memory start".into());
        }
        Ok(Self {
            beta,
            minute: start_ms.div_euclid(MINUTE_MS),
            last_timestamp_ms: start_ms,
            d: 0.0,
        })
    }

    pub fn advance_to(&mut self, timestamp_ms: i64) -> Result<(), String> {
        if timestamp_ms < self.last_timestamp_ms {
            return Err("unordered v9 realized-memory event".into());
        }
        let minute = timestamp_ms.div_euclid(MINUTE_MS);
        for _ in self.minute..minute {
            self.d = self.beta.mul_add(self.d, 0.0);
        }
        self.minute = minute;
        self.last_timestamp_ms = timestamp_ms;
        Ok(())
    }

    pub fn fill(&mut self, timestamp_ms: i64, net_realized_pnl: f64) -> Result<(), String> {
        let net = number(net_realized_pnl, "net realized PnL")?;
        let mut next = self.clone();
        next.advance_to(timestamp_ms)?;
        next.d = (next.d - net).max(0.0);
        if !next.d.is_finite() {
            return Err("v9 realized drawdown exceeds the finite risk domain".into());
        }
        *self = next;
        Ok(())
    }

    pub fn drawdown(&self) -> f64 {
        self.d as f64
    }

    pub fn allowance(&self, balance: f64, effective_pct: f64) -> Result<f64, String> {
        allowance(balance, effective_pct, self.drawdown())
    }
}

pub fn allowance(balance: f64, effective_pct: f64, drawdown: f64) -> Result<f64, String> {
    let balance = number(balance, "realized-loss allowance balance")?;
    let pct = number(effective_pct, "realized-loss allowance percentage")?;
    let d = number(drawdown, "realized-loss allowance drawdown")?;
    if balance <= 0.0 || pct < 0.0 || d < 0.0 {
        return Err("invalid v9 realized-loss allowance input".into());
    }
    let peak = balance + d;
    let value = pct.mul_add(peak, -d).max(0.0);
    if !peak.is_finite() || !value.is_finite() {
        return Err("v9 realized-loss allowance exceeds the finite risk domain".into());
    }
    Ok(value as f64)
}

/// Preserve the authored binary32 percentage graph across CPU and devices.
pub fn allowance_for(
    balance: f64,
    loss_allowance_pct: f64,
    total_wallet_exposure_limit: f64,
    drawdown: f64,
) -> Result<f64, String> {
    let loss_pct = number(loss_allowance_pct, "unstuck loss allowance percentage")?;
    let twel = number(total_wallet_exposure_limit, "unstuck total exposure limit")?;
    if loss_pct < 0.0 || twel < 0.0 {
        return Err("negative v9 unstuck allowance percentage or exposure limit".into());
    }
    let effective = loss_pct * twel;
    if !effective.is_finite() {
        return Err("v9 unstuck effective percentage exceeds the finite risk domain".into());
    }
    allowance(balance, effective as f64, drawdown)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unstuck_allowance_rejects_finite_operand_product_overflow() {
        assert!(allowance_for(1000.0, 1e20, 1e20, 0.0).is_err());
    }

    #[test]
    fn unstuck_allowance_rejects_peak_overflow() {
        assert!(allowance_for(2e38, 0.1, 1.0, 2e38).is_err());
    }

    #[test]
    fn unstuck_allowance_rejects_finite_operand_fma_overflow() {
        assert!(allowance_for(1000.0, 1e37, 1.0, 0.0).is_err());
    }

    #[test]
    fn unstuck_allowance_preserves_supported_finite_graph() {
        assert_eq!(
            allowance_for(1000.0, 0.01, 2.0, 10.0).unwrap(),
            10.199999809265137
        );
    }

    #[test]
    fn shared_realized_memory_preserves_full_fill_order_and_independent_allowances() {
        let mut gain_then_loss = RealizedMemory::new(3.0, 0).unwrap();
        gain_then_loss.fill(1, 50.0).unwrap();
        gain_then_loss.fill(1, -40.0).unwrap();
        assert_eq!(gain_then_loss.drawdown(), 40.0);
        let mut loss_then_gain = RealizedMemory::new(3.0, 0).unwrap();
        loss_then_gain.fill(1, -40.0).unwrap();
        loss_then_gain.fill(1, 50.0).unwrap();
        assert_eq!(loss_then_gain.drawdown(), 0.0);

        // Neither consumer mutates D or contributes a second spendable budget.
        let gate = gain_then_loss.allowance(1000.0, 0.05).unwrap();
        let unstuck = allowance_for(1000.0, 0.01, 2.0, 40.0).unwrap();
        assert_eq!(gate, 0.05_f32.mul_add(1040.0, -40.0) as f64);
        assert_eq!(unstuck, 0.0);
        assert_eq!(gain_then_loss.drawdown(), 40.0);
        gain_then_loss.advance_to(60_000).unwrap();
        assert_eq!(gain_then_loss.drawdown(), 20.0);
        gain_then_loss.advance_to(120_000).unwrap();
        assert_eq!(gain_then_loss.drawdown(), 10.0);
        gain_then_loss.fill(120_000, -0.25).unwrap();
        assert_eq!(gain_then_loss.drawdown(), 10.25);
        gain_then_loss.fill(120_000, 0.125).unwrap();
        assert_eq!(gain_then_loss.drawdown(), 10.125);
    }

    fn policy() -> Policy {
        Policy::new(3.0, 3.0, 0.04, 60_000).unwrap()
    }
    fn event(timestamp_ms: i64, upnl: f64, net_realized_pnl: f64, exposed: bool) -> Event {
        Event {
            timestamp_ms,
            budget: 100.0,
            upnl,
            net_realized_pnl,
            exposed,
        }
    }

    #[test]
    fn currency_decay_precedes_ratio_smoothing() {
        let p = policy();
        let mut state = ScopeState::default();
        state.observe(&p, event(0, 0.0, 0.0, true)).unwrap();
        let first = state.observe(&p, event(60_000, -10.0, 0.0, true)).unwrap();
        assert_eq!(first.drawdown, 10.0);
        assert_eq!(first.raw as f32, 0.1_f32);
        assert_eq!(first.ema as f32, 0.05_f32);
        let next = state.observe(&p, event(120_000, -10.0, 0.0, true)).unwrap();
        assert_eq!(next.drawdown, 5.0);
        assert_eq!(next.raw as f32, 5.0_f32 / 95.0);
        assert_eq!(next.ema as f32, 0.5_f32.mul_add(0.05, 0.5 * (5.0 / 95.0)));
    }

    #[test]
    fn same_minute_events_clip_drawdown_but_replace_ema() {
        let p = policy();
        let mut state = ScopeState::default();
        state.observe(&p, event(0, 0.0, 0.0, true)).unwrap();
        state.observe(&p, event(1, 0.0, 20.0, true)).unwrap();
        let loss = state.observe(&p, event(2, 0.0, -20.0, true)).unwrap();
        assert_eq!(loss.drawdown, 20.0);
        assert_eq!(loss.ema as f32, 0.5 * (20.0_f32 / 120.0));
        let before = loss.ema;
        for _ in 0..10 {
            assert_eq!(state.current(59_999).ema, before);
        }
    }

    #[test]
    fn entry_fees_count_and_pre_red_post_green_still_latches() {
        let p = policy();
        let mut state = ScopeState::default();
        let entry = state.observe(&p, event(0, 0.0, -1.0, true)).unwrap();
        assert_eq!(entry.drawdown, 1.0);
        let red = state.observe(&p, event(60_000, -20.0, 0.0, true)).unwrap();
        assert_eq!(red.action, Action::Panic);
        let flat = state.observe(&p, event(60_001, 0.0, 30.0, false)).unwrap();
        assert!(!flat.red);
        assert!(flat.flatten_qualified);
        assert_eq!(flat.cooldown_until, Some(120_001));
        assert_eq!(flat.action, Action::Halted);
        let later = state.observe(&p, event(90_000, 0.0, 0.0, false)).unwrap();
        assert_eq!(later.cooldown_until, flat.cooldown_until);
        assert_eq!(state.current(120_001).action, Action::Normal);
        let reopen = state.observe(&p, event(120_001, 0.0, -2.0, true)).unwrap();
        assert_eq!(reopen.drawdown, 2.0);
        assert_eq!(reopen.cooldown_until, None);
    }

    #[test]
    fn older_red_does_not_qualify_green_flatten() {
        let p = policy();
        let mut state = ScopeState::default();
        state.observe(&p, event(0, -20.0, 0.0, true)).unwrap();
        state.observe(&p, event(1, 10.0, 0.0, true)).unwrap();
        let flat = state.observe(&p, event(2, 0.0, 10.0, false)).unwrap();
        assert!(!flat.flatten_qualified);
        assert_eq!(flat.cooldown_until, None);
    }

    #[test]
    fn budget_change_does_not_become_trading_profit() {
        let p = policy();
        let mut state = ScopeState::default();
        state.observe(&p, event(0, -10.0, 0.0, true)).unwrap();
        let mut changed = event(1, -10.0, 0.0, true);
        changed.budget = 200.0;
        let result = state.observe(&p, changed).unwrap();
        assert_eq!(result.drawdown, 10.0);
        assert_eq!(result.raw as f32, 0.05);
        assert_eq!(result.ema as f32, 0.025);
    }

    #[test]
    fn current_insolvency_is_red_without_fast_warmup() {
        let p = Policy::new(100.0, 100.0, 1.0, 0).unwrap();
        let mut state = ScopeState::default();
        let result = state.observe(&p, event(0, -101.0, 0.0, true)).unwrap();
        assert!(result.nonpositive_equity && result.red);
        assert_eq!(result.action, Action::Panic);
    }

    #[test]
    fn unknown_prefix_uses_explicit_current_basis_estimate() {
        let p = policy();
        let mut state = ScopeState::default();
        let result = state.seed_unknown(&p, 0, 100.0, -20.0).unwrap();
        assert_eq!(result.drawdown, 20.0);
        assert_eq!(result.raw, result.ema);
        assert_eq!(result.action, Action::Panic);
    }

    #[test]
    fn realized_memory_clips_each_fill_and_decays_without_fills() {
        let mut a = RealizedMemory::new(3.0, 0).unwrap();
        a.fill(1, -10.0).unwrap();
        a.fill(2, 5.0).unwrap();
        assert_eq!(a.drawdown(), 5.0);
        let mut b = RealizedMemory::new(3.0, 0).unwrap();
        b.fill(1, 5.0).unwrap();
        b.fill(2, -10.0).unwrap();
        assert_eq!(b.drawdown(), 10.0);
        b.advance_to(60_000).unwrap();
        assert_eq!(b.drawdown(), 5.0);
        assert_eq!(
            b.allowance(95.0, 0.1).unwrap() as f32,
            0.1_f32.mul_add(100.0, -5.0)
        );
    }

    #[test]
    fn invalid_event_is_atomic_and_impossible_spans_reject() {
        assert!(Policy::new(1e30, 1.0, 0.1, 0).is_err());
        let p = policy();
        let mut state = ScopeState::default();
        let initial = state.observe(&p, event(0, -10.0, 0.0, true)).unwrap();
        let mut invalid = event(60_000, -10.0, 0.0, true);
        invalid.budget = f64::MAX;
        assert!(state.observe(&p, invalid).is_err());
        assert_eq!(state.current(0).drawdown, initial.drawdown);
        assert_eq!(state.current(0).ema, initial.ema);
    }

    #[test]
    fn quote_previews_never_commit_extrema_or_smoothing() {
        let p = policy();
        let mut with_previews = ScopeState::default();
        with_previews.observe(&p, event(0, 0.0, 0.0, true)).unwrap();
        let mut without_previews = with_previews.clone();
        assert!(
            with_previews
                .preview(&p, event(10_000, -20.0, 0.0, true))
                .unwrap()
                .red
        );
        assert!(
            !with_previews
                .preview(&p, event(20_000, 20.0, 0.0, true))
                .unwrap()
                .red
        );
        assert_eq!(with_previews.current(20_000).drawdown, 0.0);
        let actual = with_previews
            .observe(&p, event(60_000, -5.0, 0.0, true))
            .unwrap();
        let expected = without_previews
            .observe(&p, event(60_000, -5.0, 0.0, true))
            .unwrap();
        assert_eq!(actual.drawdown, expected.drawdown);
        assert_eq!(actual.raw, expected.raw);
        assert_eq!(actual.ema, expected.ema);
        assert_eq!(actual.action, expected.action);
        assert!(with_previews
            .preview(&p, event(60_001, 0.0, 1.0, false))
            .is_err());
    }
}
