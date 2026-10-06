//! HSL backtest execution integration. Permission is rebuilt from simulator facts.
use super::*;
use crate::hsl_controller::{Action, Restart};
use crate::hsl_evaluator as evaluator;
use crate::hsl_history::PositionSide;
use crate::hsl_snapshot::Mode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub enabled: bool,
    #[serde(default)]
    pub scale_budget_with_excess_allowance: bool,
    pub red_threshold: f64,
    pub ema_span_minutes: f64,
    pub cooldown_minutes_after_red: f64,
    pub restart_after_red_policy: Option<String>,
    pub panic_close_order_type: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub mode: String,
    pub sides: [Policy; 2],
    pub portfolio: Option<Policy>,
    pub coins: BTreeMap<String, [Policy; 2]>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            enabled: false,
            scale_budget_with_excess_allowance: false,
            red_threshold: 0.25,
            ema_span_minutes: 60.0,
            cooldown_minutes_after_red: 0.0,
            restart_after_red_policy: None,
            panic_close_order_type: "limit".into(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: "coin".into(),
            sides: [Policy::default(), Policy::default()],
            portfolio: None,
            coins: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn default_has_no_active_or_inherited_policy() {
        let config = Config::default();
        assert_eq!(config.mode, "coin");
        assert!(config.portfolio.is_none());
        assert!(config.coins.is_empty());
        assert!(config
            .sides
            .iter()
            .all(|p| !p.enabled && p.restart_after_red_policy.is_none()));
        config.validate(&[], -1.0, 5).unwrap();
        let mut enabled = config;
        enabled.sides[0].enabled = true;
        assert!(enabled.validate(&[], 1.0, 1).is_err());
        enabled.sides[0].restart_after_red_policy = Some("always".into());
        enabled.validate(&[], 1.0, 1).unwrap();
    }
}

impl Config {
    /// Validate before simulation, even when no candle ever causes an evaluation.
    pub fn validate(&self, coins: &[String], lookback: f64, interval: u64) -> Result<(), String> {
        if !["coin", "pside", "unified"].contains(&self.mode.as_str()) {
            return Err("invalid HSL signal mode".into());
        }
        if self.mode != "coin" && !self.coins.is_empty() {
            return Err("HSL coin policies require coin mode".into());
        }
        if self.coins.keys().any(|coin| !coins.contains(coin)) {
            return Err("HSL policy names a coin outside the dataset".into());
        }
        let policies: Vec<&Policy> = if self.mode == "unified" {
            vec![self
                .portfolio
                .as_ref()
                .ok_or("missing explicit hsl portfolio policy")?]
        } else {
            if self.portfolio.is_some() {
                return Err("hsl portfolio policy requires unified mode".into());
            }
            self.sides
                .iter()
                .chain(self.coins.values().flatten())
                .collect()
        };
        for policy in &policies {
            if self.mode != "coin" && policy.scale_budget_with_excess_allowance {
                return Err("HSL excess allowance budget scaling requires coin mode".into());
            }
            if !policy.red_threshold.is_finite()
                || !(0.0 < policy.red_threshold && policy.red_threshold <= 1.0)
                || !policy.ema_span_minutes.is_finite()
                || policy.ema_span_minutes < 1.0
                || !policy.cooldown_minutes_after_red.is_finite()
                || policy.cooldown_minutes_after_red < 0.0
                || policy.cooldown_minutes_after_red * 60_000.0 >= i64::MAX as f64
            {
                return Err("invalid HSL numeric policy".into());
            }
            if !["limit", "market"].contains(&policy.panic_close_order_type.as_str()) {
                return Err("invalid HSL panic close order type".into());
            }
            if policy.enabled
                && !matches!(
                    policy.restart_after_red_policy.as_deref(),
                    Some("always" | "never")
                )
            {
                return Err("enabled HSL requires explicit always or never restart policy".into());
            }
        }
        if policies.iter().any(|p| p.enabled) {
            if !lookback.is_finite() || !(1.0..=90.0).contains(&lookback) {
                return Err("enabled HSL requires finite lookback in [1,90] days".into());
            }
            if interval != 1 {
                return Err("enabled HSL backtest requires 1m candles".into());
            }
        }
        Ok(())
    }
}

/// Borrow-free numeric view of the already validated immutable policy.
#[derive(Clone, Copy)]
pub(super) struct SignalSettings {
    pub scale_budget_with_excess_allowance: bool,
    pub span: f64,
    pub threshold: f64,
    pub cooldown_ms: i64,
    pub restart: Restart,
}

#[derive(Clone, Debug)]
pub(super) struct Scope {
    pub(super) timestamp: i64,
    pub(super) fill_count: usize,
    pub(super) budget: f64,
    pub(super) slots: u64,
    pub(super) side: Option<usize>,
    pub(super) coin: Option<usize>,
    pub result: evaluator::Output,
}

impl Scope {
    fn contains(&self, side: usize, coin: usize) -> bool {
        self.side.is_none_or(|s| s == side) && self.coin.is_none_or(|c| c == coin)
    }
}

impl Backtest<'_> {
    pub(super) fn hsl_exposure_budget(
        &self,
        side: Option<usize>,
        coin: Option<usize>,
    ) -> Option<evaluator::ExposureBudget> {
        let (side, coin) = (side?, coin?);
        let params = if side == LONG {
            &self.bot_params_original[coin].long
        } else {
            &self.bot_params_original[coin].short
        };
        Some(evaluator::ExposureBudget {
            wallet_exposure_limit: params.wallet_exposure_limit,
            total_wallet_exposure_limit: params.total_wallet_exposure_limit,
            we_excess_allowance_pct: params.risk_we_excess_allowance_pct,
        })
    }
    pub(super) fn hsl_balance_budget(
        &self,
        side: Option<usize>,
        coin: Option<usize>,
        slots: u64,
        scale: bool,
    ) -> Result<f64, String> {
        let mode = if coin.is_some() {
            Mode::Coin
        } else if side.is_some() {
            Mode::Pside
        } else {
            Mode::Unified
        };
        evaluator::budget(
            self.balance.usd_total_balance,
            &mode,
            slots,
            scale,
            self.hsl_exposure_budget(side, coin).as_ref(),
        )
    }

    /// Current post-fill account value, before ordinary collateral revaluation.
    pub(super) fn hsl_fill_is_terminal(&self, k: usize) -> bool {
        let balance = if self.balance.use_btc_collateral {
            self.balance.btc_cash_wallet * self.btc_usd_prices[k] + self.balance.usd_cash_wallet
        } else {
            self.balance.usd_total_balance
        };
        let equity =
            balance + self.unrealized_pnl_pside(LONG, k) + self.unrealized_pnl_pside(SHORT, k);
        balance <= 0.0 || equity <= self.liquidation_equity_floor_usd()
    }

    pub(super) fn hsl_enabled(&self) -> bool {
        self.backtest_params.equity_hard_stop_loss.hsl.is_some()
    }

    pub(super) fn hsl_policy(
        &self,
        side: Option<usize>,
        coin: Option<usize>,
    ) -> Result<&Policy, String> {
        let cfg = self
            .backtest_params
            .equity_hard_stop_loss
            .hsl
            .as_ref()
            .ok_or("missing HSL config")?;
        match side {
            None => cfg
                .portfolio
                .as_ref()
                .ok_or("missing hsl portfolio HSL policy".into()),
            Some(side) => Ok(coin
                .and_then(|c| cfg.coins.get(&self.backtest_params.coins[c]))
                .map_or(&cfg.sides[side], |p| &p[side])),
        }
    }

    fn hsl_scopes(&self) -> Result<Vec<(Option<usize>, Option<usize>)>, String> {
        let cfg = self
            .backtest_params
            .equity_hard_stop_loss
            .hsl
            .as_ref()
            .ok_or("missing HSL config")?;
        Ok(match cfg.mode.as_str() {
            "unified" => vec![(None, None)],
            "pside" => vec![(Some(LONG), None), (Some(SHORT), None)],
            "coin" => [LONG, SHORT]
                .into_iter()
                .flat_map(|s| (0..self.n_coins).map(move |c| (Some(s), Some(c))))
                .collect(),
            _ => return Err("invalid HSL signal mode".into()),
        })
    }

    fn evaluate_hsl_scope(
        &mut self,
        k: usize,
        side: Option<usize>,
        coin: Option<usize>,
        boundary: bool,
    ) -> Result<Scope, String> {
        let policy = self.hsl_policy(side, coin)?;
        let restart = match policy.restart_after_red_policy.as_deref() {
            Some("always") => Restart::Always,
            Some("never") => Restart::Never,
            _ => return Err("invalid HSL restart policy".into()),
        };
        if !["limit", "market"].contains(&policy.panic_close_order_type.as_str()) {
            return Err("invalid HSL panic close order type".into());
        }
        let cooldown = policy.cooldown_minutes_after_red * 60_000.0;
        if !cooldown.is_finite() || cooldown < 0.0 || cooldown >= i64::MAX as f64 {
            return Err("invalid HSL cooldown".into());
        }
        let settings = SignalSettings {
            scale_budget_with_excess_allowance: policy.scale_budget_with_excess_allowance,
            span: policy.ema_span_minutes,
            threshold: policy.red_threshold,
            cooldown_ms: cooldown.round() as i64,
            restart,
        };
        if !boundary {
            if let Some(scope) = self.advance_hsl_scope(k, side, coin, settings) {
                return Ok(scope);
            }
            if let Some(scope) = self.replay_hsl_trace(k, side, coin, settings) {
                return Ok(scope);
            }
        }
        let mode = if coin.is_some() {
            Mode::Coin
        } else if side.is_some() {
            Mode::Pside
        } else {
            Mode::Unified
        };
        let side_name = side.map(|s| {
            if s == LONG {
                PositionSide::Long
            } else {
                PositionSide::Short
            }
        });
        let cutoff = self.hsl_history_cutoff(side, coin);
        let symbol = coin.map(|c| self.backtest_params.coins[c].as_str());
        let observed = self.hsl_inputs_at_clipped(k, mode, side_name, symbol, boundary, cutoff)?;
        let timestamp = observed.snapshot.now;
        let slots = side.map_or(1, |s| observed.slots[s]) as u64;
        let exposure_budget = self.hsl_exposure_budget(side, coin);
        let budget = if coin.is_some() && slots == 0 {
            0.0
        } else {
            evaluator::budget(
                observed.snapshot.balance,
                &observed.snapshot.mode,
                slots,
                settings.scale_budget_with_excess_allowance,
                exposure_budget.as_ref(),
            )?
        };
        let mut result = evaluator::evaluate_for_simulator(evaluator::Input {
            snapshot: observed.snapshot,
            slots,
            scale_budget_with_excess_allowance: settings.scale_budget_with_excess_allowance,
            exposure_budget,
            span: settings.span,
            threshold: settings.threshold,
            cooldown_ms: settings.cooldown_ms,
            restart: settings.restart,
        })?;
        result.reasons.extend(observed.reasons);
        self.seed_hsl_trace((side, coin), &mut result);
        Ok(Scope {
            timestamp,
            fill_count: self.fills.len(),
            budget,
            slots,
            side,
            coin,
            result,
        })
    }

    pub(super) fn update_hsl(&mut self, k: usize) -> Result<(), String> {
        let mut results = Vec::new();
        for (side, coin) in self.hsl_scopes()? {
            if self.hsl_policy(side, coin)?.enabled {
                results.push(self.evaluate_hsl_scope(k, side, coin, false)?);
            }
        }
        // Replace atomically only after every selected current evaluation succeeds.
        self.hsl_scopes = results;
        for scope in &self.hsl_scopes {
            self.hsl_report.observe(
                (scope.side, scope.coin),
                scope.timestamp,
                "bar_close",
                &scope.result,
            );
        }
        self.clear_hsl_entries();
        Ok(())
    }

    pub(super) fn hsl_order_params(&self, side: usize, coin: usize) -> BotParams {
        let mut params = if side == LONG {
            self.bot_params[coin].long.clone()
        } else {
            self.bot_params[coin].short.clone()
        };
        if let Some(cfg) = &self.backtest_params.equity_hard_stop_loss.hsl {
            let scope = match cfg.mode.as_str() {
                "unified" => (None, None),
                "pside" => (Some(side), None),
                "coin" => (Some(side), Some(coin)),
                _ => panic!("invalid HSL mode before order construction"),
            };
            let policy = self
                .hsl_policy(scope.0, scope.1)
                .expect("validated HSL execution policy");
            params.hsl_enabled = policy.enabled;
            params.hsl_panic_close_order_type = policy.panic_close_order_type.clone();
        }
        params
    }

    pub(super) fn hsl_protective_orders(
        &self,
        k: usize,
    ) -> Result<Vec<orchestrator::ExecutableOrder>, String> {
        let mut inputs = Vec::new();
        for side in [LONG, SHORT] {
            for coin in 0..self.n_coins {
                if self.hsl_action(side, coin) != Action::Panic {
                    continue;
                }
                let size = if side == LONG {
                    self.positions.long[coin].size
                } else {
                    self.positions.short[coin].size
                };
                if size == 0.0 {
                    continue;
                }
                let params = self.hsl_order_params(side, coin);
                let price = self.hlcvs_value(k, coin, CLOSE);
                inputs.push(orchestrator::ProtectiveCloseInput {
                    symbol_idx: coin,
                    pside: if side == LONG {
                        orchestrator::PositionSide::Long
                    } else {
                        orchestrator::PositionSide::Short
                    },
                    position_size: size,
                    order_book: OrderBook {
                        bid: price,
                        ask: price,
                    },
                    price_step: self.exchange_params_list[coin].price_step,
                    execution_type: if params.hsl_panic_close_order_type == "market" {
                        orchestrator::ExecutionType::Market
                    } else {
                        orchestrator::ExecutionType::Limit
                    },
                });
            }
        }
        orchestrator::compute_protective_closes(&inputs).map_err(str::to_owned)
    }

    pub(super) fn hsl_action(&self, side: usize, coin: usize) -> Action {
        self.hsl_scopes
            .iter()
            .find(|s| s.contains(side, coin))
            .and_then(|s| s.result.decision.as_ref())
            .map_or(Action::Normal, |d| d.action)
    }

    fn clear_hsl_entries(&mut self) {
        for side in [LONG, SHORT] {
            for coin in 0..self.n_coins {
                if self.hsl_action(side, coin) != Action::Normal {
                    if side == LONG {
                        self.open_orders.long[coin].entries.clear();
                    } else {
                        self.open_orders.short[coin].entries.clear();
                    }
                }
            }
        }
    }

    pub(super) fn finish_hsl_flat(
        &mut self,
        k: usize,
        coin: usize,
        side: usize,
    ) -> Result<(), String> {
        for (scope_side, scope_coin) in self.hsl_scopes()? {
            if scope_side.is_some_and(|s| s != side)
                || scope_coin.is_some_and(|c| c != coin)
                || !self.hsl_policy(scope_side, scope_coin)?.enabled
            {
                continue;
            }
            let exposed = [LONG, SHORT].into_iter().any(|s| {
                (0..self.n_coins).any(|c| {
                    scope_side.is_none_or(|v| v == s)
                        && scope_coin.is_none_or(|v| v == c)
                        && (if s == LONG {
                            self.positions.long[c].size
                        } else {
                            self.positions.short[c].size
                        }) != 0.0
                })
            });
            if exposed {
                continue;
            }
            if self.hsl_fill_is_terminal(k) {
                // The fill proves this scope flat even though a liquidating account
                // cannot enter normal risk evaluation. Finalize observation only;
                // liquidation owns the terminal trading outcome.
                let now = (self.first_timestamp_ms + k as u64 * self.interval_ms) as i64;
                self.hsl_report.observed_flat((scope_side, scope_coin), now);
                continue;
            }
            let updated = self.evaluate_hsl_scope(k, scope_side, scope_coin, true)?;
            self.hsl_report.observe(
                (scope_side, scope_coin),
                updated.timestamp,
                "scope_flat",
                &updated.result,
            );
            self.hsl_scopes
                .retain(|s| s.side != scope_side || s.coin != scope_coin);
            self.hsl_scopes.push(updated);
        }
        self.clear_hsl_entries();
        Ok(())
    }

    pub(super) fn apply_hsl_modes(
        &self,
        idx: usize,
        long: &mut Option<orchestrator::TradingMode>,
        short: &mut Option<orchestrator::TradingMode>,
    ) {
        for (side, mode) in [(LONG, long), (SHORT, short)] {
            match self.hsl_action(side, idx) {
                Action::Normal => {}
                Action::Panic => *mode = Some(orchestrator::TradingMode::Panic),
                Action::Halted => *mode = Some(orchestrator::TradingMode::GracefulStop),
            }
        }
    }
}
