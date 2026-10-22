//! Forward-only v9 CPU adapter. No history reconstruction enters ordinary replay.
use super::*;
use crate::hsl_decay::{self, Event, ScopeState};
use crate::hsl_intent::{Action, Decision, LifecycleEvent};
use crate::hsl_support as evaluator;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub enabled: bool,
    #[serde(default)]
    pub scale_budget_with_excess_allowance: bool,
    pub red_threshold: f64,
    pub memory_ema_span_minutes: f64,
    pub drawdown_ratio_ema_span_minutes: f64,
    pub cooldown_minutes_after_red: f64,
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
            memory_ema_span_minutes: 43_200.0,
            drawdown_ratio_ema_span_minutes: 60.0,
            cooldown_minutes_after_red: 0.0,
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
impl Config {
    pub fn validate(&self, coins: &[String], interval: u64) -> Result<(), String> {
        if !["coin", "pside", "portfolio"].contains(&self.mode.as_str()) {
            return Err("invalid v9 HSL signal mode".into());
        }
        if self.mode != "coin" && !self.coins.is_empty() {
            return Err("HSL coin policies require coin mode".into());
        }
        if self.coins.keys().any(|coin| !coins.contains(coin)) {
            return Err("HSL policy names a coin outside the dataset".into());
        }
        let policies: Vec<&Policy> = if self.mode == "portfolio" {
            vec![self
                .portfolio
                .as_ref()
                .ok_or("missing explicit portfolio HSL policy")?]
        } else {
            if self.portfolio.is_some() {
                return Err("portfolio policy requires portfolio mode".into());
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
            if !["limit", "market"].contains(&policy.panic_close_order_type.as_str()) {
                return Err("invalid HSL panic close order type".into());
            }
            let cooldown = policy.cooldown_minutes_after_red * 60_000.0;
            if !cooldown.is_finite() || cooldown < 0.0 || cooldown >= i64::MAX as f64 {
                return Err("invalid HSL cooldown".into());
            }
            hsl_decay::Policy::new(
                policy.memory_ema_span_minutes,
                policy.drawdown_ratio_ema_span_minutes,
                policy.red_threshold,
                cooldown.round() as i64,
            )?;
        }
        if policies.iter().any(|p| p.enabled) && interval != 1 {
            return Err("enabled HSL backtest requires 1m candles".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct Scope {
    pub(super) side: Option<usize>,
    pub(super) coin: Option<usize>,
    policy: hsl_decay::Policy,
    state: ScopeState,
    pub result: evaluator::Output,
}
impl Scope {
    fn contains(&self, side: usize, coin: usize) -> bool {
        self.side.is_none_or(|s| s == side) && self.coin.is_none_or(|c| c == coin)
    }
}

impl<const ORGANILLO: bool> super::BacktestEngine<'_, ORGANILLO> {
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
        self.backtest_params
            .equity_hard_stop_loss
            .hsl
            .as_ref()
            .is_some_and(|c| {
                c.portfolio.as_ref().is_some_and(|p| p.enabled)
                    || c.sides
                        .iter()
                        .chain(c.coins.values().flatten())
                        .any(|p| p.enabled)
            })
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
                .ok_or_else(|| "missing portfolio HSL policy".into()),
            Some(side) => Ok(coin
                .and_then(|c| cfg.coins.get(&self.backtest_params.coins[c]))
                .map_or(&cfg.sides[side], |p| &p[side])),
        }
    }

    pub(super) fn initialize_hsl(&mut self) -> Result<(), String> {
        if self.hsl_initialized {
            return Ok(());
        }
        let Some(cfg) = &self.backtest_params.equity_hard_stop_loss.hsl else {
            self.hsl_initialized = true;
            return Ok(());
        };
        cfg.validate(
            &self.backtest_params.coins,
            self.backtest_params.candle_interval_minutes,
        )?;
        let keys: Vec<_> = match cfg.mode.as_str() {
            "portfolio" => vec![(None, None)],
            "pside" => vec![(Some(LONG), None), (Some(SHORT), None)],
            "coin" => [LONG, SHORT]
                .into_iter()
                .flat_map(|s| (0..self.n_coins).map(move |c| (Some(s), Some(c))))
                .collect(),
            _ => return Err("invalid HSL mode".into()),
        };
        let mut scopes = Vec::new();
        for (side, coin) in keys {
            let p = self.hsl_policy(side, coin)?;
            if !p.enabled {
                continue;
            }
            scopes.push(Scope {
                side,
                coin,
                policy: hsl_decay::Policy::new(
                    p.memory_ema_span_minutes,
                    p.drawdown_ratio_ema_span_minutes,
                    p.red_threshold,
                    (p.cooldown_minutes_after_red * 60_000.0).round() as i64,
                )?,
                state: ScopeState::default(),
                result: evaluator::Output {
                    decision: None,
                    events: Vec::new(),
                    reasons: BTreeSet::new(),
                    observations: 0,
                    episodes: 0,
                },
            });
        }
        self.hsl_scopes = scopes;
        self.hsl_initialized = true;
        Ok(())
    }

    fn scope_facts(
        &self,
        side: Option<usize>,
        coin: Option<usize>,
        _k: usize,
        _boundary_price: Option<(usize, f64)>,
    ) -> Result<(f64, f64, bool), String> {
        let mut upnl = 0.0;
        let mut exposed = false;
        for s in [LONG, SHORT] {
            if side.is_some_and(|v| v != s) {
                continue;
            }
            let positions = if s == LONG {
                &self.positions.long
            } else {
                &self.positions.short
            };
            let first = coin.unwrap_or(0);
            let end = coin.map_or(self.n_coins, |c| c + 1);
            for c in first..end {
                let position = positions[c];
                if position.size == 0.0 {
                    continue;
                }
                exposed = true;
                let mark = self.hsl_marks[c];
                if !mark.is_finite() || mark <= 0.0 {
                    return Err("invalid causal HSL mark".into());
                }
                upnl += if s == LONG {
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
            }
        }
        let mut budget = self.balance.usd_total_balance;
        if let (Some(s), Some(c)) = (side, coin) {
            let slots = self.hard_stop_coin_slot_n_positions(s);
            if slots == 0 {
                if exposed {
                    return Err("held HSL coin has zero configured slots".into());
                }
                return Ok((0.0, 0.0, false));
            }
            budget /= slots as f64;
            if self
                .hsl_policy(side, coin)?
                .scale_budget_with_excess_allowance
            {
                let authored = if s == LONG {
                    &self.bot_params_original[c].long
                } else {
                    &self.bot_params_original[c].short
                };
                let base = if authored.wallet_exposure_limit < 0.0 {
                    authored.total_wallet_exposure_limit / slots as f64
                } else {
                    authored.wallet_exposure_limit
                };
                budget *= 1.0 + effective_we_excess_allowance_pct(authored, base);
            }
        }
        Ok((budget, upnl, exposed))
    }

    fn observe_hsl(
        &mut self,
        k: usize,
        timestamp: i64,
        boundary_price: Option<(usize, f64)>,
        cashflow: Option<(usize, usize, f64)>,
        phase: &'static str,
    ) -> Result<(), String> {
        self.initialize_hsl()?;
        if self.hsl_scopes.is_empty() {
            return Ok(());
        }
        if let Some((coin, price)) = boundary_price {
            self.hsl_marks[coin] = price;
            for c in 0..self.n_coins {
                if !self.hsl_marks[c].is_finite() && k > 0 {
                    self.hsl_marks[c] = self.hlcvs_value(k - 1, c, CLOSE);
                }
            }
        } else {
            for c in 0..self.n_coins {
                if self.coin_is_valid_at(c, k) {
                    self.hsl_marks[c] = self.hlcvs_value(k, c, CLOSE);
                }
            }
        }
        for i in 0..self.hsl_scopes.len() {
            let (side, coin) = (self.hsl_scopes[i].side, self.hsl_scopes[i].coin);
            let (budget, upnl, exposed) = self.scope_facts(side, coin, k, boundary_price)?;
            if coin.is_some() && side.is_some_and(|s| self.hard_stop_coin_slot_n_positions(s) == 0)
            {
                // Zero configured slots are an inactive scope, not an observed
                // zero-equity GREEN. scope_facts still rejects held exposure.
                let scope = &mut self.hsl_scopes[i];
                scope.result = evaluator::Output {
                    decision: None,
                    events: Vec::new(),
                    reasons: BTreeSet::from(["inactive_scope".into()]),
                    observations: 0,
                    episodes: 0,
                };
                self.hsl_report
                    .observe((side, coin), timestamp, phase, &scope.result);
                continue;
            }
            let net = cashflow
                .filter(|(s, c, _)| side.is_none_or(|v| v == *s) && coin.is_none_or(|v| v == *c))
                .map_or(0.0, |(_, _, net)| net);
            let scope = &mut self.hsl_scopes[i];
            let previous = scope.result.decision.clone();
            let decision = scope.state.observe(
                &scope.policy,
                Event {
                    timestamp_ms: timestamp,
                    budget,
                    upnl,
                    net_realized_pnl: net,
                    exposed,
                },
            )?;
            let prior_action = previous.as_ref().map_or(Action::Normal, |d| d.action);
            let red_at = if decision.action == Action::Normal {
                None
            } else {
                previous.as_ref().and_then(|d| d.red_at).or(Some(timestamp))
            };
            let flat_at = decision
                .cooldown_until
                .map(|until| until - scope.policy.cooldown_ms);
            let mut events = Vec::new();
            let mut event = |kind, reason| {
                events.push(LifecycleEvent {
                    timestamp,
                    kind,
                    red_at: red_at.unwrap_or(timestamp),
                    flat_at,
                    reason,
                    raw: Some(decision.raw),
                    ema: Some(decision.ema),
                })
            };
            if decision.action == Action::Panic && prior_action != Action::Panic {
                event("red", "drawdown");
            }
            if decision.flatten_qualified {
                if prior_action == Action::Normal {
                    event("red", "final_accounting");
                }
                event("flat", "stop_flattened");
                if decision.action == Action::Normal {
                    event("restart", "cooldown_complete");
                }
            } else if prior_action != Action::Normal && decision.action == Action::Normal {
                event(
                    "restart",
                    if exposed {
                        "current_green"
                    } else {
                        "cooldown_complete"
                    },
                );
            }
            scope.result = evaluator::Output {
                decision: Some(Decision {
                    timestamp,
                    action: decision.action,
                    red_at,
                    flat_at,
                    reason: if decision.action == Action::Panic {
                        "drawdown"
                    } else if decision.action == Action::Halted {
                        "cooldown"
                    } else {
                        "green"
                    },
                    raw: decision.raw,
                    ema: decision.ema,
                    numeric_range_approximation: false,
                }),
                events,
                reasons: BTreeSet::new(),
                observations: 1,
                episodes: 0,
            };
            self.hsl_report
                .observe((side, coin), timestamp, phase, &scope.result);
        }
        self.clear_hsl_entries();
        Ok(())
    }

    pub(super) fn update_hsl(&mut self, k: usize) -> Result<(), String> {
        let timestamp = self
            .first_timestamp_ms
            .checked_add((k as u64 + 1) * self.interval_ms)
            .and_then(|t| i64::try_from(t).ok())
            .ok_or("HSL close timestamp overflow")?;
        self.observe_hsl(k, timestamp, None, None, "bar_close")
    }

    pub(super) fn hsl_before_fill(
        &mut self,
        k: usize,
        coin: usize,
        exec: OrderFillExecution,
    ) -> Result<(), String> {
        if !self.hsl_enabled() {
            return Ok(());
        }
        let timestamp = i64::try_from(self.fill_timestamp_ms(k, exec)?)
            .map_err(|_| "risk fill timestamp overflow")?;
        self.observe_hsl(k, timestamp, Some((coin, exec.price)), None, "pre_fill")
    }

    pub(super) fn hsl_after_fill(
        &mut self,
        k: usize,
        coin: usize,
        side: usize,
        exec: OrderFillExecution,
        net: f64,
    ) -> Result<(), String> {
        let timestamp = i64::try_from(self.fill_timestamp_ms(k, exec)?)
            .map_err(|_| "risk fill timestamp overflow")?;
        if self.unstuck_memory_enabled {
            self.unstuck_memory.fill(timestamp, net)?;
        }
        if self.loss_gate_memory_enabled {
            self.loss_gate_memory.fill(timestamp, net)?;
        }
        if !self.hsl_enabled() {
            return Ok(());
        }
        self.observe_hsl(
            k,
            timestamp,
            Some((coin, exec.price)),
            Some((side, coin, net)),
            "post_fill",
        )
    }

    pub(super) fn hsl_order_params(&self, side: usize, coin: usize) -> BotParams {
        let mut params = if side == LONG {
            self.bot_params[coin].long.clone()
        } else {
            self.bot_params[coin].short.clone()
        };
        if let Some(cfg) = &self.backtest_params.equity_hard_stop_loss.hsl {
            let scope = match cfg.mode.as_str() {
                "portfolio" => (None, None),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    fn params(mode: &str, coins: usize) -> BacktestParams {
        let policy = Policy {
            enabled: true,
            red_threshold: 0.005,
            memory_ema_span_minutes: 3.0,
            drawdown_ratio_ema_span_minutes: 1.0,
            cooldown_minutes_after_red: 2.0,
            ..Policy::default()
        };
        let config = Config {
            mode: mode.into(),
            sides: [policy.clone(), policy.clone()],
            portfolio: (mode == "portfolio").then_some(policy),
            coins: BTreeMap::new(),
        };
        BacktestParams {
            side_activation: crate::types::SideActivation::uniform(true),
            minimum_coin_age_days: [0.0; 2],
            inception_timestamps_ms: vec![],
            starting_balance: 1000.0,
            maker_fee: 0.001,
            taker_fee: 0.002,
            coins: (0..coins).map(|i| format!("COIN{i}")).collect(),
            active_coin_indices: None,
            first_timestamp_ms: 0,
            requested_start_timestamp_ms: 0,
            first_valid_indices: vec![0; coins],
            last_valid_indices: vec![5; coins],
            warmup_minutes: vec![0; coins],
            trade_start_indices: vec![0; coins],
            global_warmup_bars: 1,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: false,
            ew_half_life_days: None,
            hsl_detailed_report: true,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: crate::types::MinimumCostFilter::uniform(false),
            dynamic_wel_by_tradability: false,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            unstuck_memory_ema_span_minutes: 3.0,
            loss_gate_memory_ema_span_minutes: 3.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: crate::types::EquityHardStopLossConfig { hsl: Some(config) },
            market_orders_allowed: true,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0,
            limit_order_fill_buffer_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        }
    }

    fn bots(coins: usize) -> Vec<BotParamsPair> {
        (0..coins)
            .map(|_| {
                let mut pair = BotParamsPair::default();
                for side in [&mut pair.long, &mut pair.short] {
                    side.n_positions = coins;
                    side.total_wallet_exposure_limit = 1.0;
                    side.unstuck_loss_allowance_pct = 0.01;
                    side.unstuck_close_pct = 0.1;
                    side.unstuck_threshold = 0.95;
                    side.ema_span_0 = 1.0;
                    side.ema_span_1 = 2.0;
                }
                pair
            })
            .collect()
    }

    fn candles(coins: usize, close: f64) -> Array3<f64> {
        Array3::from_shape_fn(
            (6, coins, 4),
            |(_, _, column)| if column == VOLUME { 1.0 } else { close },
        )
    }

    fn execution(price: f64) -> OrderFillExecution {
        OrderFillExecution {
            at_bar_close: false,
            price,
            fee_rate: 0.001,
            liquidity: "maker",
        }
    }

    fn order(qty: f64, order_type: OrderType) -> Order {
        Order {
            qty,
            price: 100.0,
            order_type,
        }
    }

    #[test]
    fn v9_fill_uses_execution_fact_before_future_candle_close() {
        let mut values = candles(1, 100.0);
        values[[1, 0, CLOSE]] = 1.0;
        let btc = Array1::from_elem(6, 20_000.0);
        let p = params("coin", 1);
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(1),
            vec![ExchangeParams::default()],
            &p,
        );
        bt.process_entry_fill_long(
            1,
            0,
            &order(1.0, OrderType::EntryInitialNormalLong),
            execution(100.0),
        )
        .unwrap();
        let d = bt.hsl_scopes[0].state.current(60_000);
        assert_eq!(d.drawdown as f32, 0.1_f32);
        assert_eq!(d.action, Action::Normal);
        assert_eq!(bt.unstuck_memory.drawdown() as f32, 0.1_f32);
        bt.update_hsl(1).unwrap();
        assert_eq!(bt.hsl_action(LONG, 0), Action::Panic);
        assert!(bt.hsl_scopes[0].state.current(120_000).drawdown > 99.0);
    }

    #[test]
    fn side_activation_preserves_hsl_protection_for_existing_positions() {
        for side in [LONG, SHORT] {
            let mut values = candles(1, 100.0);
            values[[1, 0, CLOSE]] = if side == LONG { 1.0 } else { 199.0 };
            let btc = Array1::from_elem(6, 20_000.0);
            let p = params("coin", 1);
            let mut bt = Backtest::new(
                values.view(),
                btc.view(),
                bots(1),
                vec![ExchangeParams::default()],
                &p,
            );
            if side == LONG {
                bt.process_entry_fill_long(
                    1,
                    0,
                    &order(1.0, OrderType::EntryInitialNormalLong),
                    execution(100.0),
                )
                .unwrap();
            } else {
                bt.process_entry_fill_short(
                    1,
                    0,
                    &order(-1.0, OrderType::EntryInitialNormalShort),
                    execution(100.0),
                )
                .unwrap();
            }
            bt.backtest_params.side_activation = crate::types::SideActivation::uniform(false);
            bt.update_hsl(1).unwrap();
            assert_eq!(bt.hsl_action(side, 0), Action::Panic);
            let orders = bt.hsl_protective_orders(1).unwrap();
            assert!(orders.iter().any(|o| o.pside.to_pside_int() == side
                && matches!(
                    o.order_type,
                    OrderType::ClosePanicLong | OrderType::ClosePanicShort
                )));
            assert!(!orders.iter().any(|o| o.order_type.is_entry()));
        }
    }

    #[test]
    fn v9_terminal_bar_commits_completed_mark_before_liquidation() {
        let mut values = candles(1, 100.0);
        values[[1, 0, CLOSE]] = 1.0;
        let btc = Array1::from_elem(6, 20_000.0);
        let p = params("coin", 1);
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(1),
            vec![ExchangeParams::default()],
            &p,
        );
        bt.process_entry_fill_long(
            1,
            0,
            &order(20.0, OrderType::EntryInitialNormalLong),
            execution(100.0),
        )
        .unwrap();
        assert_eq!(bt.hsl_action(LONG, 0), Action::Normal);
        bt.equity_tracking_active = true;

        // Exercise run's early terminal branch: the fill at the bar start was
        // GREEN, but the now-completed close must be observed before stopping.
        bt.run().unwrap();

        assert!(bt.liquidated);
        let terminal = bt.hsl_scopes[0].state.current(120_000);
        assert_eq!(terminal.action, Action::Panic);
        assert!(terminal.drawdown > 1900.0);
        assert!(terminal.nonpositive_equity);
    }

    #[test]
    fn v9_inactive_coin_never_fabricates_a_green_signal() {
        let values = candles(1, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let p = params("coin", 1);
        let mut policies = bots(1);
        policies[0].long.n_positions = 0;
        policies[0].short.n_positions = 0;
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            policies,
            vec![ExchangeParams::default()],
            &p,
        );
        bt.update_hsl(1).unwrap();
        for scope in &bt.hsl_scopes {
            assert!(scope.result.decision.is_none());
            assert!(scope.result.reasons.contains("inactive_scope"));
        }
        for sample in &bt.hsl_report.samples {
            assert!(sample.action.is_none() && sample.raw.is_none() && sample.ema.is_none());
            assert_eq!(sample.reasons, vec!["inactive_scope"]);
        }
    }

    #[test]
    fn v9_partial_realization_counts_only_new_fee_and_preserves_episode() {
        let values = candles(1, 90.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let p = params("coin", 1);
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(1),
            vec![ExchangeParams::default()],
            &p,
        );
        bt.process_entry_fill_long(
            1,
            0,
            &order(1.0, OrderType::EntryInitialNormalLong),
            execution(100.0),
        )
        .unwrap();
        bt.update_hsl(1).unwrap();
        let before = bt.hsl_scopes[0].state.current(120_000).drawdown as f32;
        bt.process_close_fill_long(
            2,
            0,
            &order(-0.5, OrderType::CloseGridLong),
            execution(90.0),
        )
        .unwrap();
        let after = bt.hsl_scopes[0].state.current(120_000);
        assert_eq!(after.drawdown as f32, before + 0.045_f32);
        assert!(after.exposed);
        assert_eq!(after.cooldown_until, None);
        assert!((bt.balance.usd_total_balance - 994.855).abs() < 1e-10);
        assert_eq!(bt.unstuck_memory.drawdown() as f32, 0.05_f32 - (-5.045_f32));
    }

    #[test]
    fn v9_aggregate_constituent_changes_do_not_reset_exposed_scope() {
        for mode in ["pside", "portfolio"] {
            let values = candles(2, 90.0);
            let btc = Array1::from_elem(6, 20_000.0);
            let p = params(mode, 2);
            let mut bt = Backtest::new(
                values.view(),
                btc.view(),
                bots(2),
                vec![ExchangeParams::default(); 2],
                &p,
            );
            bt.process_entry_fill_long(
                1,
                0,
                &order(1.0, OrderType::EntryInitialNormalLong),
                execution(100.0),
            )
            .unwrap();
            bt.update_hsl(1).unwrap();
            let before = bt.hsl_scopes[0].state.current(120_000).drawdown;
            bt.process_entry_fill_long(
                2,
                1,
                &order(1.0, OrderType::EntryInitialNormalLong),
                execution(100.0),
            )
            .unwrap();
            let after = bt.hsl_scopes[0].state.current(120_000);
            assert!(
                after.drawdown >= before,
                "{mode} reset an aggregate episode"
            );
            bt.process_close_fill_long(
                2,
                0,
                &order(-1.0, OrderType::CloseGridLong),
                execution(90.0),
            )
            .unwrap();
            assert!(bt.hsl_scopes[0].state.current(120_000).exposed);
            assert_eq!(bt.hsl_scopes[0].state.current(120_000).cooldown_until, None);
        }
    }

    #[test]
    fn v9_hedged_net_zero_is_gross_exposed_and_realized_memory_is_shared() {
        let values = candles(1, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let p = params("portfolio", 1);
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(1),
            vec![ExchangeParams::default()],
            &p,
        );
        bt.process_entry_fill_long(
            1,
            0,
            &order(1.0, OrderType::EntryInitialNormalLong),
            execution(100.0),
        )
        .unwrap();
        bt.process_entry_fill_short(
            1,
            0,
            &order(-1.0, OrderType::EntryInitialNormalShort),
            execution(100.0),
        )
        .unwrap();
        assert!(bt.hsl_scopes[0].state.current(60_000).exposed);
        assert_eq!(bt.unstuck_memory.drawdown() as f32, 0.2_f32);
        bt.process_close_fill_long(
            2,
            0,
            &order(-1.0, OrderType::CloseGridLong),
            execution(80.0),
        )
        .unwrap();
        assert!(bt.hsl_scopes[0].state.current(120_000).exposed);
        assert!(bt.unstuck_memory.drawdown() > 20.0);
        bt.process_close_fill_short(
            2,
            0,
            &order(1.0, OrderType::CloseGridShort),
            execution(80.0),
        )
        .unwrap();
        assert!(!bt.hsl_scopes[0].state.current(120_000).exposed);
        assert!(bt.unstuck_memory.drawdown() > 0.0);
    }

    #[test]
    fn v9_flat_queries_and_budget_changes_do_not_renew_cooldown() {
        let values = candles(1, 80.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let p = params("coin", 1);
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(1),
            vec![ExchangeParams::default()],
            &p,
        );
        bt.process_entry_fill_long(
            1,
            0,
            &order(1.0, OrderType::EntryInitialNormalLong),
            execution(100.0),
        )
        .unwrap();
        bt.update_hsl(1).unwrap();
        bt.process_close_fill_long(
            2,
            0,
            &order(-1.0, OrderType::CloseGridLong),
            execution(80.0),
        )
        .unwrap();
        assert_eq!(bt.hsl_action(LONG, 0), Action::Halted);
        let deadline = bt.hsl_scopes[0].state.current(120_000).cooldown_until;
        bt.balance.usd_total_balance = 1_000_000.0;
        bt.update_hsl(2).unwrap();
        assert_eq!(
            bt.hsl_scopes[0].state.current(180_000).cooldown_until,
            deadline
        );
        assert_eq!(bt.hsl_action(LONG, 0), Action::Halted);
        bt.update_hsl(3).unwrap();
        assert_eq!(bt.hsl_action(LONG, 0), Action::Normal);
    }

    #[test]
    fn realized_gate_only_consumes_actual_net_fills_once_across_sides_and_flatness() {
        let values = candles(2, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let mut p = params("coin", 2);
        p.max_realized_loss_pct = 0.05;
        p.equity_hard_stop_loss = crate::types::EquityHardStopLossConfig::default();
        let mut bot_params = bots(2);
        for pair in &mut bot_params {
            pair.long.unstuck_enabled = false;
            pair.short.unstuck_enabled = false;
        }
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bot_params,
            vec![ExchangeParams::default(); 2],
            &p,
        );
        assert!(bt.loss_gate_memory_enabled);
        assert!(!bt.unstuck_memory_enabled);
        assert!(!bt.hsl_enabled());
        bt.process_entry_fill_long(
            0,
            0,
            &order(1.0, OrderType::EntryInitialNormalLong),
            execution(100.0),
        )
        .unwrap();
        bt.process_entry_fill_short(
            0,
            1,
            &order(-1.0, OrderType::EntryInitialNormalShort),
            execution(100.0),
        )
        .unwrap();
        assert_eq!(bt.loss_gate_memory.drawdown(), 0.2_f32 as f64);
        // Profit clips only the already-realized loss. The later loss is fully
        // debited; netting this same-time pair would incorrectly hide it.
        bt.process_close_fill_short(
            0,
            1,
            &order(1.0, OrderType::CloseGridShort),
            execution(80.0),
        )
        .unwrap();
        bt.process_close_fill_long(
            0,
            0,
            &order(-1.0, OrderType::CloseGridLong),
            execution(80.0),
        )
        .unwrap();
        assert_eq!(bt.loss_gate_memory.drawdown(), 20.08_f32 as f64);
        assert_eq!(bt.unstuck_memory.drawdown(), 0.0);
        assert_eq!(bt.positions.long[0].size, 0.0);
        assert_eq!(bt.positions.short[1].size, 0.0);
        let net_ledger: f64 = bt.fills.iter().map(|fill| fill.pnl + fill.fee_paid).sum();
        assert_eq!(bt.pnl_cumsum_running_net, net_ledger);
        assert_eq!(bt.pnl_cumsum_running_net_pside[LONG], -20.2 + 0.02);
        assert_eq!(bt.pnl_cumsum_running_net_pside[SHORT], 19.8 + 0.02);
        let before = bt.get_orchestrator_input_cached(0, None, None);
        assert_eq!(
            before.global.loss_gate_realized_pnl_drawdown,
            20.08_f32 as f64
        );
        bt.orchestrator_input_cache = Some(before);

        // Cash balance and HSL episode state never mutate the realized memory.
        bt.balance.usd_total_balance += 100.0;
        assert_eq!(bt.loss_gate_memory.drawdown(), 20.08_f32 as f64);
        bt.loss_gate_memory.advance_to(60_000).unwrap();
        let after = bt.get_orchestrator_input_cached(1, None, None);
        assert_eq!(
            after.global.loss_gate_realized_pnl_drawdown,
            10.04_f32 as f64
        );
        assert_eq!(after.balance_raw, bt.balance.usd_total_balance);
        // A signed entry rebate reduces the same D, without resetting a flat
        // account or a newly opened position to zero.
        let mut rebate = execution(100.0);
        rebate.fee_rate = -0.001;
        bt.process_entry_fill_long(1, 0, &order(1.0, OrderType::EntryInitialNormalLong), rebate)
            .unwrap();
        assert_eq!(bt.loss_gate_memory.drawdown(), (10.04_f32 - 0.1_f32) as f64);
    }

    #[test]
    fn actual_loss_restricts_unstuck_and_gate_from_equal_span_memories() {
        let values = candles(1, 90.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let mut p = params("coin", 1);
        p.max_realized_loss_pct = 0.15;
        p.equity_hard_stop_loss = crate::types::EquityHardStopLossConfig::default();
        let mut bot_params = bots(1);
        bot_params[0].short.unstuck_enabled = false;
        let long = &mut bot_params[0].long;
        long.unstuck_enabled = true;
        long.unstuck_ema_gating_enabled = false;
        long.unstuck_loss_allowance_pct = 0.02;
        long.unstuck_close_pct = 0.5;
        long.unstuck_threshold = 0.5;
        long.wallet_exposure_limit = 0.5;
        long.risk_wel_enforcer_enabled = false;
        long.risk_twel_enforcer_enabled = false;
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bot_params,
            vec![ExchangeParams::default()],
            &p,
        );
        let mut entry = execution(100.0);
        entry.fee_rate = 0.0;
        bt.process_entry_fill_long(0, 0, &order(10.0, OrderType::EntryInitialNormalLong), entry)
            .unwrap();
        let mut exit = execution(50.0);
        exit.fee_rate = 0.0;
        bt.process_close_fill_long(0, 0, &order(-1.0, OrderType::CloseGridLong), exit)
            .unwrap();
        assert_eq!(bt.loss_gate_memory.drawdown(), 50.0);
        let input = bt.get_orchestrator_input_cached(0, None, None);
        assert_eq!(input.balance_raw, 950.0);
        assert_eq!(input.global.loss_gate_realized_pnl_drawdown, 50.0);
        assert_eq!(input.global.unstuck_realized_pnl_drawdown, 50.0);
        assert_eq!(bt.unstuck_allowance_for_position(0, LONG).unwrap(), 0.0);
        assert_eq!(
            crate::hsl_decay::allowance(input.balance_raw, p.max_realized_loss_pct, 50.0).unwrap(),
            0.15_f32.mul_add(1000.0, -50.0) as f64,
        );
        let actual = orchestrator::compute_ideal_orders(&input).unwrap();
        assert!(actual
            .orders
            .iter()
            .all(|order| order.order_type != OrderType::CloseUnstuckLong));
        let mut without_loss_memory = input.clone();
        without_loss_memory.global.unstuck_realized_pnl_drawdown = 0.0;
        without_loss_memory.global.loss_gate_realized_pnl_drawdown = 0.0;
        let reference = orchestrator::compute_ideal_orders(&without_loss_memory).unwrap();
        assert!(reference
            .orders
            .iter()
            .any(|order| order.order_type == OrderType::CloseUnstuckLong));
        assert_eq!(bt.loss_gate_memory.drawdown(), 50.0);
    }

    #[test]
    fn gate_only_replay_decays_without_fills_at_every_completed_minute() {
        let values = candles(1, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        for interval in [1, 5] {
            let mut p = params("coin", 1);
            p.max_realized_loss_pct = 0.0;
            p.candle_interval_minutes = interval;
            p.equity_hard_stop_loss = crate::types::EquityHardStopLossConfig::default();
            let mut bot_params = bots(1);
            let pair = &mut bot_params[0];
            for side in [&mut pair.long, &mut pair.short] {
                side.unstuck_enabled = false;
                side.entry_eligible = false;
                side.n_positions = 0;
            }
            let mut bt = Backtest::new(
                values.view(),
                btc.view(),
                bot_params,
                vec![ExchangeParams::default()],
                &p,
            );
            let mut entry = execution(100.0);
            entry.fee_rate = 0.0;
            bt.process_entry_fill_long(0, 0, &order(1.0, OrderType::EntryInitialNormalLong), entry)
                .unwrap();
            let mut exit = execution(36.0);
            exit.fee_rate = 0.0;
            bt.process_close_fill_long(0, 0, &order(-1.0, OrderType::CloseGridLong), exit)
                .unwrap();
            assert_eq!(bt.loss_gate_memory.drawdown(), 64.0);
            let (fills, _) = bt.run().unwrap();
            assert_eq!(fills.len(), 2);
            let mut expected = 64.0_f32;
            // The last row is next-candle lookahead; k=4 is the last
            // completed row in this six-row ordinary backtest fixture.
            for _ in 0..5 * interval {
                expected = 0.5_f32.mul_add(expected, 0.0);
            }
            assert_eq!(bt.loss_gate_memory.drawdown(), expected as f64);
        }
    }

    #[test]
    fn realized_gate_zero_activates_memory_and_disabled_gate_does_not() {
        let values = candles(1, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        for pct in [0.0, 1.0 - f64::EPSILON, 1.0, 2.0] {
            let mut p = params("coin", 1);
            p.max_realized_loss_pct = pct;
            p.equity_hard_stop_loss = crate::types::EquityHardStopLossConfig::default();
            let mut bot_params = bots(1);
            bot_params[0].long.unstuck_enabled = false;
            bot_params[0].short.unstuck_enabled = false;
            let bt = Backtest::new(
                values.view(),
                btc.view(),
                bot_params,
                vec![ExchangeParams::default()],
                &p,
            );
            assert_eq!(bt.loss_gate_memory_enabled, pct < 1.0);
        }
    }

    #[test]
    fn v9_disabled_risk_does_not_narrow_unconsumed_economics() {
        let values = candles(1, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let mut p = params("coin", 1);
        p.equity_hard_stop_loss = crate::types::EquityHardStopLossConfig::default();
        let mut bot_params = bots(1);
        bot_params[0].long.unstuck_enabled = false;
        bot_params[0].short.unstuck_enabled = false;
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bot_params,
            vec![ExchangeParams::default()],
            &p,
        );
        assert!(!bt.loss_gate_memory_enabled);
        assert!(!bt.unstuck_memory_enabled);
        bt.hsl_after_fill(1, 0, LONG, execution(100.0), f64::MAX)
            .unwrap();
        assert_eq!(bt.loss_gate_memory.drawdown(), 0.0);
    }

    #[test]
    fn independent_realized_spans_share_actual_fills_but_not_filter_state() {
        let values = candles(1, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let mut p = params("coin", 1);
        p.max_realized_loss_pct = 0.15;
        p.unstuck_memory_ema_span_minutes = 3.0;
        p.loss_gate_memory_ema_span_minutes = 7.0;
        p.equity_hard_stop_loss = crate::types::EquityHardStopLossConfig::default();
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(1),
            vec![ExchangeParams::default()],
            &p,
        );
        assert!(bt.unstuck_memory_enabled && bt.loss_gate_memory_enabled);
        let mut entry = execution(100.0);
        entry.fee_rate = 0.0;
        bt.process_entry_fill_long(0, 0, &order(10.0, OrderType::EntryInitialNormalLong), entry)
            .unwrap();
        let mut exit = execution(50.0);
        exit.fee_rate = 0.0;
        bt.process_close_fill_long(0, 0, &order(-1.0, OrderType::CloseGridLong), exit)
            .unwrap();
        // The ordinary ledger has one loss, while each filter remembers all of it.
        assert_eq!(bt.fills.len(), 2);
        assert_eq!(bt.pnl_cumsum_running_net, -50.0);
        assert_eq!(bt.balance.usd_total_balance, 950.0);
        assert_eq!(bt.unstuck_memory.drawdown(), 50.0);
        assert_eq!(bt.loss_gate_memory.drawdown(), 50.0);
        bt.unstuck_memory.advance_to(60_000).unwrap();
        bt.loss_gate_memory.advance_to(60_000).unwrap();
        let input = bt.get_orchestrator_input_cached(1, None, None);
        assert_eq!(input.global.unstuck_realized_pnl_drawdown, 25.0);
        assert_eq!(input.global.loss_gate_realized_pnl_drawdown, 37.5);
        // A later actual profit repays each existing memory independently.
        let mut profit = execution(110.0);
        profit.fee_rate = 0.0;
        bt.process_close_fill_long(1, 0, &order(-1.0, OrderType::CloseGridLong), profit)
            .unwrap();
        assert_eq!(bt.unstuck_memory.drawdown(), 15.0);
        assert_eq!(bt.loss_gate_memory.drawdown(), 27.5);
        assert_eq!(bt.pnl_cumsum_running_net, -40.0);
    }

    #[test]
    fn coin_hsl_memory_override_is_independent_of_ratio_and_account_memories() {
        let values = candles(2, 100.0);
        let btc = Array1::from_elem(6, 20_000.0);
        let mut p = params("coin", 2);
        p.unstuck_memory_ema_span_minutes = 15.0;
        p.loss_gate_memory_ema_span_minutes = 31.0;
        let config = p.equity_hard_stop_loss.hsl.as_mut().unwrap();
        let mut local = config.sides.clone();
        local[LONG].memory_ema_span_minutes = 7.0;
        local[LONG].drawdown_ratio_ema_span_minutes = 3.0;
        local[LONG].scale_budget_with_excess_allowance = true;
        config.coins.insert("COIN1".into(), local);
        let mut bt = Backtest::new(
            values.view(),
            btc.view(),
            bots(2),
            vec![ExchangeParams::default(); 2],
            &p,
        );
        bt.initialize_hsl().unwrap();
        let base = bt
            .hsl_scopes
            .iter()
            .find(|s| s.side == Some(LONG) && s.coin == Some(0))
            .unwrap();
        let local = bt
            .hsl_scopes
            .iter()
            .find(|s| s.side == Some(LONG) && s.coin == Some(1))
            .unwrap();
        assert_eq!((base.policy.slow_beta, base.policy.fast_beta), (0.5, 0.0));
        assert_eq!(
            (local.policy.slow_beta, local.policy.fast_beta),
            (0.75, 0.5)
        );
        assert_eq!(
            bt.hsl_policy(Some(LONG), Some(1))
                .unwrap()
                .memory_ema_span_minutes,
            7.0
        );
        assert!(
            bt.hsl_policy(Some(LONG), Some(1))
                .unwrap()
                .scale_budget_with_excess_allowance
        );
    }
}
