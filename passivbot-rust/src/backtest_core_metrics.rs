//! Causal observational hooks for the bounded core; never supplies trading input.
use super::*;
use crate::core_metrics::{
    CoreMetrics, CoreMetricsResult, EvaluationWindow, ExposureObservation, FillObservation,
    Finalization, Metric, MetricRequest, PositionSide, DAY_MS, MINUTE_MS,
};

impl<const ORGANILLO: bool> BacktestEngine<'_, ORGANILLO> {
    /// Resolve a fixed evaluation boundary before replay, independent of capacity.
    pub fn configure_core_metrics(
        &mut self,
        request: MetricRequest,
        ew_half_life_days: f64,
    ) -> Result<(), String> {
        if self.current_step != 0 || self.core_request.is_some() {
            return Err("core metric request must be configured once before replay".into());
        }
        if !ew_half_life_days.is_finite() || ew_half_life_days <= 0.0 {
            return Err("core metric EW half-life must be finite and positive".into());
        }
        if self.interval_ms == 0 {
            return Err("metric observation interval must be positive".into());
        }
        let requested_offset = self
            .backtest_params
            .requested_start_timestamp_ms
            .saturating_sub(self.first_timestamp_ms);
        let requested_index = requested_offset / self.interval_ms
            + u64::from(requested_offset % self.interval_ms != 0);
        let first_index = (self.warmup_bars.max(1) as u64)
            .checked_add(1)
            .ok_or("core metric warmup clock overflow")?
            .max(requested_index);
        let final_index = self.hlcvs.shape()[0]
            .checked_sub(1)
            .ok_or("empty replay has no evaluated interval")? as u64;
        if first_index >= final_index {
            return Err("replay has no positive evaluated interval after declared warmup".into());
        }
        let timestamp = |index: u64| {
            self.first_timestamp_ms
                .checked_add(
                    index
                        .checked_mul(self.interval_ms)
                        .ok_or("core metric clock overflow")?,
                )
                .ok_or("core metric clock overflow")
        };
        self.core_window = Some(EvaluationWindow {
            start_timestamp_ms: timestamp(first_index)?,
            intended_end_timestamp_ms: timestamp(final_index)?,
        });
        self.core_request = Some(request);
        self.core_half_life_days = ew_half_life_days;
        Ok(())
    }

    pub fn core_metrics_result(&self) -> Result<&CoreMetricsResult, String> {
        self.final_core_metrics
            .as_ref()
            .ok_or_else(|| "core metrics are unavailable before completed replay".into())
    }

    /// BTC inputs are candle closes. At the opening/fill boundary only the
    /// preceding close is complete; a completed-bar observation admits close k.
    /// This quote is observational and never replaces the trading wallet quote.
    fn core_btc_price(&self, k: usize, at_bar_close: bool) -> Result<f64, String> {
        let completed_index = if at_bar_close {
            k
        } else {
            k.checked_sub(1)
                .ok_or("opening metric observation has no completed BTC close")?
        };
        Ok(self.btc_usd_prices[completed_index])
    }

    pub(super) fn core_raw_account_balance(
        &self,
        k: usize,
        at_bar_close: bool,
    ) -> Result<f64, String> {
        if self.balance.use_btc_collateral {
            Ok(
                self.balance.btc_cash_wallet * self.core_btc_price(k, at_bar_close)?
                    + self.balance.usd_cash_wallet,
            )
        } else {
            Ok(self.balance.usd_total_balance)
        }
    }

    fn core_position_facts(&self) -> Result<([f64; 2], f64), String> {
        let mut upnl = [0.0; 2];
        let mut gross = 0.0;
        for (side, positions) in [&self.positions.long, &self.positions.short]
            .into_iter()
            .enumerate()
        {
            for (coin, position) in positions.iter().enumerate() {
                if position.size == 0.0 {
                    continue;
                }
                let mark = self.core_marks[coin];
                let multiplier = self.exchange_params_list[coin].c_mult;
                if !mark.is_finite() || mark <= 0.0 || !multiplier.is_finite() || multiplier <= 0.0
                {
                    return Err(format!(
                        "missing causal metric mark or multiplier for held coin {coin}"
                    ));
                }
                upnl[side] += if side == LONG {
                    calc_pnl_long(position.price, mark, position.size, multiplier)
                } else {
                    calc_pnl_short(position.price, mark, position.size, multiplier)
                };
                gross += position.size.abs() * mark * multiplier;
            }
        }
        Ok((upnl, gross))
    }

    fn record_core_artifact(
        &mut self,
        timestamp: u64,
        account_equity: f64,
        strategy_equity: f64,
        k: usize,
        at_bar_close: bool,
    ) -> Result<(), String> {
        if self.backtest_params.metrics_only {
            return Ok(());
        }
        let btc = self.core_btc_price(k, at_bar_close)?;
        if !btc.is_finite() || btc <= 0.0 {
            return Err("invalid BTC conversion for equity artifact".into());
        }
        if self.equities.timestamps_ms.last() == Some(&timestamp) {
            *self.equities.usd_total_equity.last_mut().unwrap() = account_equity;
            *self.equities.btc_total_equity.last_mut().unwrap() = account_equity / btc;
            *self.strategy_equity_series.last_mut().unwrap() = strategy_equity;
        } else {
            self.equities.timestamps_ms.push(timestamp);
            self.equities.usd_total_equity.push(account_equity);
            self.equities.btc_total_equity.push(account_equity / btc);
            self.strategy_equity_series.push(strategy_equity);
        }
        Ok(())
    }

    pub(super) fn begin_core_metrics_if_due(&mut self, k: usize) -> Result<(), String> {
        let Some(window) = self.core_window else {
            return Ok(());
        };
        let timestamp = self.first_timestamp_ms + k as u64 * self.interval_ms;
        if timestamp != window.start_timestamp_ms || self.core_observer.is_some() {
            return Ok(());
        }
        for coin in 0..self.n_coins {
            if self.coin_is_valid_at(coin, k - 1) {
                self.core_marks[coin] = self.hlcvs_value(k - 1, coin, CLOSE);
            }
        }
        let (upnl, _) = self.core_position_facts()?;
        self.core_side_start = [
            self.pnl_cumsum_running_net_pside[LONG] + upnl[LONG],
            self.pnl_cumsum_running_net_pside[SHORT] + upnl[SHORT],
        ];
        let balance = self.core_raw_account_balance(k, false)?;
        let account = balance + upnl[LONG] + upnl[SHORT];
        // One authoritative USD economic series prevents accumulation-order drift.
        let strategy = if self.balance.use_btc_collateral {
            self.backtest_params.starting_balance
                + self.pnl_cumsum_running_net
                + upnl[LONG]
                + upnl[SHORT]
        } else {
            account
        };
        let mut observer = CoreMetrics::new_with_interval(
            self.core_request.unwrap(),
            window,
            strategy,
            Some(self.core_half_life_days),
            self.interval_ms,
        )?;
        for (side, positions) in [&self.positions.long, &self.positions.short]
            .into_iter()
            .enumerate()
        {
            for (coin, position) in positions.iter().enumerate() {
                if position.size != 0.0 {
                    observer.seed_open_position(
                        coin,
                        if side == LONG {
                            PositionSide::Long
                        } else {
                            PositionSide::Short
                        },
                    );
                }
            }
        }
        if timestamp % DAY_MS == 0 {
            observer.observe_daily_close(timestamp, strategy);
        }
        self.core_observer = Some(observer);
        self.core_actual_end_timestamp_ms = Some(timestamp);
        self.hsl_report
            .advance(i64::try_from(timestamp).map_err(|_| "HSL metric clock exceeds i64")?);
        self.record_core_exposure(timestamp, strategy, balance)?;
        self.record_core_artifact(timestamp, account, strategy, k, false)
    }

    fn record_core_exposure(
        &mut self,
        timestamp: u64,
        strategy: f64,
        balance: f64,
    ) -> Result<(), String> {
        let request = self.core_request.unwrap();
        if !request.needs_exposure() || self.interval_ms != MINUTE_MS {
            return Ok(());
        }
        let (_, gross) = self.core_position_facts()?;
        let limits = [
            self.bot_params_master
                .long
                .total_wallet_exposure_limit_ratio,
            self.bot_params_master
                .short
                .total_wallet_exposure_limit_ratio,
        ];
        let mut utilization = [0.0; 2];
        for (side, positions) in [&self.positions.long, &self.positions.short]
            .into_iter()
            .enumerate()
        {
            let metric = if side == LONG {
                Metric::PositionUtilizationLong
            } else {
                Metric::PositionUtilizationShort
            };
            if !request.contains(metric) {
                continue;
            }
            for (coin, position) in positions.iter().enumerate() {
                if position.size == 0.0 {
                    continue;
                }
                let e =
                    position.size.abs() * position.price * self.exchange_params_list[coin].c_mult
                        / balance;
                let wel = self
                    .runtime_budget(coin, side)
                    .effective_wallet_exposure_limit_ratio;
                utilization[side] += crate::exposure_rms::position_moment(e, wel, limits[side]);
            }
        }
        self.core_observer
            .as_mut()
            .unwrap()
            .observe_exposure(ExposureObservation {
                timestamp_ms: timestamp,
                strategy_equity: strategy,
                gross_marked_notional_usd: gross,
                position_utilization_squared: utilization,
            });
        Ok(())
    }

    pub(super) fn record_core_fill(
        &mut self,
        k: usize,
        coin: usize,
        side: usize,
        exec: OrderFillExecution,
        before: f64,
        after: f64,
        qty: f64,
        pre_balance: f64,
        net: f64,
    ) -> Result<(), String> {
        self.last_fill_index = Some(k);
        if self.core_request.is_none() {
            return Ok(());
        }
        self.core_marks[coin] = exec.price;
        let timestamp = self.fill_timestamp_ms(k, exec)?;
        if let Some(observer) = self.core_observer.as_mut() {
            observer.observe_fill(FillObservation {
                timestamp_ms: timestamp,
                coin_index: coin,
                position_side: if side == LONG {
                    PositionSide::Long
                } else {
                    PositionSide::Short
                },
                position_size_before: before,
                position_size_after: after,
                notional_usd: qty_to_cost(qty, exec.price, self.exchange_params_list[coin].c_mult),
                pre_fill_balance_usd: pre_balance,
                net_realized_pnl_usd: net,
            });
        }
        Ok(())
    }

    pub(super) fn record_core_observation(
        &mut self,
        k: usize,
        at_fill_boundary: bool,
    ) -> Result<(), String> {
        if self.core_observer.is_none() {
            return Ok(());
        }
        if !at_fill_boundary {
            for coin in 0..self.n_coins {
                if self.coin_is_valid_at(coin, k) {
                    self.core_marks[coin] = self.hlcvs_value(k, coin, CLOSE);
                }
            }
        }
        let timestamp =
            self.first_timestamp_ms + (k as u64 + u64::from(!at_fill_boundary)) * self.interval_ms;
        let (upnl, _) = self.core_position_facts()?;
        let balance = self.core_raw_account_balance(k, !at_fill_boundary)?;
        let account = balance + upnl[LONG] + upnl[SHORT];
        // One authoritative USD economic series prevents accumulation-order drift.
        let strategy = if self.balance.use_btc_collateral {
            self.backtest_params.starting_balance
                + self.pnl_cumsum_running_net
                + upnl[LONG]
                + upnl[SHORT]
        } else {
            account
        };
        let observer = self.core_observer.as_mut().unwrap();
        if at_fill_boundary {
            observer.observe_terminal(timestamp, strategy);
        } else {
            observer.observe_minute(timestamp, strategy);
        }
        if timestamp % DAY_MS == 0 && !at_fill_boundary {
            observer.observe_daily_close(timestamp, strategy);
        }
        self.core_actual_end_timestamp_ms = Some(timestamp);
        self.hsl_report
            .advance(i64::try_from(timestamp).map_err(|_| "HSL metric clock exceeds i64")?);
        self.record_core_exposure(timestamp, strategy, balance)?;
        self.record_core_artifact(timestamp, account, strategy, k, !at_fill_boundary)
    }

    /// Observation uses resolved policies for admitted coins and active sides.
    /// Capacity and current exposure never disable an otherwise enabled scope.
    fn observed_hsl_enabled(&self) -> Result<bool, String> {
        let Some(config) = &self.backtest_params.equity_hard_stop_loss.hsl else {
            return Ok(false);
        };
        match config.mode.as_str() {
            "portfolio" => Ok(config
                .portfolio
                .as_ref()
                .ok_or("missing explicit portfolio HSL policy")?
                .enabled),
            "pside" => Ok([LONG, SHORT].into_iter().any(|side| {
                self.backtest_params.side_activation.for_side(side) && config.sides[side].enabled
            })),
            "coin" => {
                for side in [LONG, SHORT] {
                    if !self.backtest_params.side_activation.for_side(side) {
                        continue;
                    }
                    for coin in 0..self.n_coins {
                        if self.hsl_policy(Some(side), Some(coin))?.enabled {
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            }
            _ => Err("invalid HSL observation mode".into()),
        }
    }

    pub(super) fn finish_core_metrics(&mut self) -> Result<(), String> {
        let observer = self
            .core_observer
            .as_ref()
            .ok_or("replay did not reach the declared metric window")?;
        let end = self
            .core_actual_end_timestamp_ms
            .ok_or("missing metric final clock")?;
        let (upnl, _) = self.core_position_facts()?;
        let profits = [
            self.pnl_cumsum_running_net_pside[LONG] + upnl[LONG] - self.core_side_start[LONG],
            self.pnl_cumsum_running_net_pside[SHORT] + upnl[SHORT] - self.core_side_start[SHORT],
        ];
        let hsl = self.hsl_report.core_summary(self.observed_hsl_enabled()?);
        self.final_core_metrics = Some(observer.finish(Finalization {
            actual_end_timestamp_ms: end,
            net_side_profit_usd: profits,
            hsl,
        }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_metrics::{MetricStatus, DEFAULT_EW_HALF_LIFE_DAYS};
    use ndarray::{Array1, Array3};

    fn data(n: usize) -> Array3<f64> {
        let mut data = Array3::from_elem((n, 1, 4), 1.0);
        for k in 0..n {
            data[[k, 0, HIGH]] = 120.0;
            data[[k, 0, LOW]] = 80.0;
            data[[k, 0, CLOSE]] = 100.0;
        }
        data
    }
    fn engine<'a>(
        data: &'a Array3<f64>,
        btc: &'a Array1<f64>,
        metrics_only: bool,
        interval: u64,
    ) -> Backtest<'a> {
        let mut params =
            super::super::tests::ordinary_ema_fixture_params(data.shape()[0], interval);
        params.metrics_only = metrics_only;
        params.equity_hard_stop_loss.hsl = None;
        let mut exchange = ExchangeParams::default();
        exchange.c_mult = 2.0;
        Backtest::new(
            data.view(),
            btc.view(),
            vec![BotParamsPair::default()],
            vec![exchange],
            &params,
        )
    }
    fn request(metrics: &[Metric]) -> MetricRequest {
        MetricRequest::from_metrics(metrics.iter().copied())
    }
    fn number(bt: &Backtest<'_>, metric: Metric) -> f64 {
        let value = bt.core_metrics_result().unwrap().metrics[metric.definition().name];
        assert_eq!(value.status, MetricStatus::Finite, "{metric:?}: {value:?}");
        value.value.unwrap()
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10_f64.max(b.abs() * 1e-10), "{a} != {b}");
    }
    fn execution(price: f64) -> OrderFillExecution {
        OrderFillExecution {
            at_bar_close: false,
            price,
            fee_fraction: 0.001,
            liquidity: "maker",
        }
    }
    fn controlled_roundtrip(bt: &mut Backtest<'_>) {
        bt.begin_core_metrics_if_due(2).unwrap();
        bt.process_entry_fill_long(
            2,
            0,
            &Order {
                qty: 1.0,
                price: 100.0,
                order_type: OrderType::EntryInitialNormalLong,
            },
            execution(100.0),
        )
        .unwrap();
        bt.record_core_observation(2, false).unwrap();
        bt.process_close_fill_long(
            3,
            0,
            &Order {
                qty: -1.0,
                price: 110.0,
                order_type: OrderType::CloseGridLong,
            },
            execution(110.0),
        )
        .unwrap();
        bt.record_core_observation(3, false).unwrap();
        for k in 4..bt.hlcvs.shape()[0] - 1 {
            bt.record_core_observation(k, false).unwrap();
        }
        if bt.core_request.is_some() {
            bt.finish_core_metrics().unwrap();
        }
    }

    #[test]
    fn zero_capacity_does_not_delay_evaluation_or_allocate_metric_histories() {
        let data = data(20);
        let btc = Array1::from_elem(20, 20_000.0);
        let mut bt = engine(&data, &btc, true, 1);
        bt.configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        let (fills, equities) = bt.run().unwrap();
        assert!(!bt.equity_tracking_active);
        assert!(fills.is_empty() && equities.timestamps_ms.is_empty());
        let meta = &bt.core_metrics_result().unwrap().metadata;
        assert_eq!(meta.start_timestamp_ms, 2 * MINUTE_MS);
        assert_eq!(meta.actual_end_timestamp_ms, 19 * MINUTE_MS);
        assert_eq!(meta.equity_observation_count, 18);
        close(number(&bt, Metric::Completion), 1.0);
        close(number(&bt, Metric::Pain), 0.0);
        close(number(&bt, Metric::Cagr), 0.0);
        close(number(&bt, Metric::Turnover), 0.0);
        close(number(&bt, Metric::FillGapMax), 17.0 / 1440.0);
    }

    #[test]
    fn hsl_metadata_uses_effective_admitted_policy_without_capacity_or_entry_gates() {
        use super::super::hsl_runtime::{Config, Policy};
        let data = data(8);
        let btc = Array1::from_elem(8, 20_000.0);
        for overridden_off in [false, true] {
            let mut bt = engine(&data, &btc, true, 1);
            bt.backtest_params.side_activation = crate::types::SideActivation {
                long: true,
                short: false,
            };
            bt.backtest_params.trade_start_indices = vec![100];
            let mut config = Config::default();
            config.sides[LONG].enabled = true;
            if overridden_off {
                config.coins.insert(
                    bt.backtest_params.coins[0].clone(),
                    [Policy::default(), Policy::default()],
                );
            }
            bt.backtest_params.equity_hard_stop_loss.hsl = Some(config);
            // Preserve the trading helper's existing authority; observation
            // intentionally resolves overrides before declaring enablement.
            assert!(bt.hsl_enabled());
            assert_eq!(bt.observed_hsl_enabled().unwrap(), !overridden_off);
            bt.configure_core_metrics(
                request(&[Metric::HslTriggersPerYear, Metric::HslRedTime]),
                DEFAULT_EW_HALF_LIFE_DAYS,
            )
            .unwrap();
            bt.run().unwrap();
            assert_eq!(
                bt.core_metrics_result().unwrap().metadata.hsl_enabled,
                !overridden_off
            );
            close(number(&bt, Metric::HslTriggersPerYear), 0.0);
            close(number(&bt, Metric::HslRedTime), 0.0);
        }
    }

    #[test]
    fn hsl_observation_respects_active_sides_and_explicit_portfolio_policy() {
        use super::super::hsl_runtime::{Config, Policy};
        let data = data(8);
        let btc = Array1::from_elem(8, 20_000.0);
        let mut bt = engine(&data, &btc, true, 1);
        bt.backtest_params.side_activation = crate::types::SideActivation {
            long: true,
            short: false,
        };
        for mode in ["coin", "pside"] {
            let mut config = Config::default();
            config.mode = mode.into();
            config.sides[SHORT].enabled = true;
            bt.backtest_params.equity_hard_stop_loss.hsl = Some(config);
            assert!(!bt.observed_hsl_enabled().unwrap());
        }
        let mut config = Config::default();
        config.mode = "portfolio".into();
        config.sides[LONG].enabled = true;
        config.portfolio = Some(Policy::default());
        bt.backtest_params.equity_hard_stop_loss.hsl = Some(config);
        assert!(!bt.observed_hsl_enabled().unwrap());
        bt.backtest_params
            .equity_hard_stop_loss
            .hsl
            .as_mut()
            .unwrap()
            .portfolio
            .as_mut()
            .unwrap()
            .enabled = true;
        assert!(bt.observed_hsl_enabled().unwrap());
    }

    #[test]
    fn full_flat_artifacts_include_baseline_and_causal_close_clocks() {
        let data = data(8);
        let btc = Array1::from_elem(8, 20_000.0);
        let mut bt = engine(&data, &btc, false, 1);
        bt.configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        let (_, equities) = bt.run().unwrap();
        assert_eq!(
            equities.timestamps_ms,
            (2..=7).map(|k| k * MINUTE_MS).collect::<Vec<_>>()
        );
        assert_eq!(equities.usd_total_equity, vec![1000.0; 6]);
        assert_eq!(bt.strategy_equity_series, vec![1000.0; 6]);
    }

    #[test]
    fn explicit_requested_start_is_rounded_up_once_and_short_windows_fail() {
        let data = data(8);
        let btc = Array1::from_elem(8, 20_000.0);
        let mut bt = engine(&data, &btc, true, 1);
        bt.backtest_params.requested_start_timestamp_ms = 3 * MINUTE_MS + 1;
        bt.configure_core_metrics(request(&[Metric::Completion]), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        bt.run().unwrap();
        assert_eq!(
            bt.core_metrics_result()
                .unwrap()
                .metadata
                .start_timestamp_ms,
            4 * MINUTE_MS
        );
        close(number(&bt, Metric::Completion), 1.0);
        let mut short = engine(&data, &btc, true, 1);
        short.backtest_params.requested_start_timestamp_ms = 7 * MINUTE_MS;
        assert!(short
            .configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap_err()
            .contains("no positive evaluated interval"));
        let mut overflow = engine(&data, &btc, true, 1);
        overflow.warmup_bars = usize::MAX;
        assert!(overflow
            .configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap_err()
            .contains("warmup clock overflow"));
        let mut zero = engine(&data, &btc, true, 1);
        zero.interval_ms = 0;
        assert!(zero
            .configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap_err()
            .contains("must be positive"));
    }

    #[test]
    fn selected_vs_all_core_and_report_vs_metrics_only_preserve_fill_accounting() {
        let data = data(10);
        let btc = Array1::from_elem(10, 20_000.0);
        let mut legacy = engine(&data, &btc, false, 1);
        controlled_roundtrip(&mut legacy);
        let mut full = engine(&data, &btc, false, 1);
        full.configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        controlled_roundtrip(&mut full);
        let selected = [
            Metric::Turnover,
            Metric::LossProfit,
            Metric::HoldingMax,
            Metric::ShortContribution,
        ];
        let mut minimal = engine(&data, &btc, true, 1);
        minimal
            .configure_core_metrics(request(&selected), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        controlled_roundtrip(&mut minimal);
        assert_eq!(legacy.fills.len(), 2);
        assert_eq!(full.fills.len(), 2);
        assert!(minimal.fills.is_empty());
        for (a, b) in legacy.fills.iter().zip(&full.fills) {
            assert_eq!(
                (
                    a.pnl,
                    a.fee_paid,
                    a.fill_qty,
                    a.fill_price,
                    a.usd_total_balance,
                    a.timestamp_ms
                ),
                (
                    b.pnl,
                    b.fee_paid,
                    b.fill_qty,
                    b.fill_price,
                    b.usd_total_balance,
                    b.timestamp_ms
                )
            );
        }
        assert_eq!(
            legacy.balance.usd_total_balance,
            full.balance.usd_total_balance
        );
        assert_eq!(
            full.balance.usd_total_balance,
            minimal.balance.usd_total_balance
        );
        assert_eq!(minimal.last_fill_index, Some(3));
        assert_eq!(
            minimal.core_metrics_result().unwrap().metadata.fill_count,
            2
        );
        for metric in selected {
            assert_eq!(
                full.core_metrics_result().unwrap().metrics[metric.definition().name],
                minimal.core_metrics_result().unwrap().metrics[metric.definition().name]
            );
        }
        close(
            number(&full, Metric::Turnover),
            (200.0 / 1000.0 + 220.0 / 999.8) / (7.0 / 1440.0),
        );
        close(number(&full, Metric::LossProfit), 0.2 / 19.78);
        close(number(&full, Metric::HoldingMax), 1.0 / 1440.0);
        close(number(&full, Metric::ShortContribution), 0.0);
        assert!(minimal.equities.timestamps_ms.is_empty());
    }

    #[test]
    fn early_fill_terminal_uses_previous_completed_marks_and_factual_account_equity() {
        let mut data = data(8);
        data[[3, 0, CLOSE]] = 9999.0;
        let btc = Array1::from_elem(8, 20_000.0);
        let mut bt = engine(&data, &btc, false, 1);
        bt.configure_core_metrics(
            request(&[Metric::Pain, Metric::Completion]),
            DEFAULT_EW_HALF_LIFE_DAYS,
        )
        .unwrap();
        bt.begin_core_metrics_if_due(2).unwrap();
        bt.process_entry_fill_long(
            2,
            0,
            &Order {
                qty: 1.0,
                price: 100.0,
                order_type: OrderType::EntryInitialNormalLong,
            },
            execution(100.0),
        )
        .unwrap();
        bt.record_core_observation(2, false).unwrap();
        bt.balance.usd_total_balance = -10.0;
        bt.balance.usd_cash_wallet = -10.0;
        bt.update_equities(3);
        assert!(bt.check_and_apply_liquidation(3));
        bt.record_core_observation(3, true).unwrap();
        bt.finish_core_metrics().unwrap();
        assert_eq!(
            bt.equities.timestamps_ms,
            vec![2 * MINUTE_MS, 3 * MINUTE_MS]
        );
        assert_eq!(bt.equities.usd_total_equity.last(), Some(&-10.0));
        // Future close 9999 must not become the terminal mark of this still-open position.
        close(*bt.strategy_equity_series.last().unwrap(), -10.0);
        assert!(bt.liquidated());
        close(number(&bt, Metric::Completion), 1.0 / 5.0);
    }

    #[test]
    fn coarser_replay_remains_supported_without_inventing_minute_metrics() {
        let data = data(10);
        let btc = Array1::from_elem(10, 20_000.0);
        let mut bt = engine(&data, &btc, true, 5);
        bt.configure_core_metrics(MetricRequest::all(), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        bt.run().unwrap();
        let result = bt.core_metrics_result().unwrap();
        assert_eq!(result.metadata.observation_interval_ms, 5 * MINUTE_MS);
        close(number(&bt, Metric::Cagr), 0.0);
        close(number(&bt, Metric::Completion), 1.0);
        let value = result.metrics[Metric::Pain.definition().name];
        assert_eq!(value.status, MetricStatus::Invalid);
        assert_eq!(value.reason, Some("unsupported_observation_interval"));
    }
    #[test]
    fn usd_and_strategy_artifacts_share_one_authoritative_economic_series() {
        let data = data(10);
        let btc = Array1::from_elem(10, 20_000.0);
        let mut bt = engine(&data, &btc, false, 1);
        bt.configure_core_metrics(request(&[Metric::Pain]), DEFAULT_EW_HALF_LIFE_DAYS)
            .unwrap();
        bt.begin_core_metrics_if_due(2).unwrap();
        for _ in 0..100 {
            bt.process_entry_fill_long(
                2,
                0,
                &Order {
                    qty: 1.0,
                    price: 100.0,
                    order_type: OrderType::EntryInitialNormalLong,
                },
                execution(100.0),
            )
            .unwrap();
        }
        for k in 2..9 {
            bt.record_core_observation(k, false).unwrap();
        }
        bt.finish_core_metrics().unwrap();
        assert_eq!(bt.equities.usd_total_equity, bt.strategy_equity_series);
    }

    // A fixed wallet isolates observation valuation from the trading owner's
    // separately modeled collateral conversion/revaluation operations.
    fn fixed_btc_wallet(bt: &mut Backtest<'_>) {
        bt.balance.use_btc_collateral = true;
        bt.balance.btc_collateral_target_ratio = 0.0;
        bt.balance.btc_cash_wallet = 0.02;
        bt.balance.usd_cash_wallet = 600.0;
        bt.balance.usd_total_balance = 1000.0;
    }

    #[test]
    fn btc_baseline_uses_previous_close_and_completed_path_admits_current_close() {
        let data = data(8);
        for current_btc_close in [20_000.0, 80_000.0] {
            let mut btc = Array1::from_elem(8, 20_000.0);
            btc[2] = current_btc_close;
            let mut bt = engine(&data, &btc, false, 1);
            fixed_btc_wallet(&mut bt);
            bt.positions.long[0] = Position {
                size: 1.0,
                price: 100.0,
            };
            bt.bot_params_master.long.total_wallet_exposure_limit_ratio = 1.0;
            bt.runtime_budget[0]
                .long
                .effective_wallet_exposure_limit_ratio = 0.5;
            bt.configure_core_metrics(
                request(&[Metric::PositionUtilizationLong]),
                DEFAULT_EW_HALF_LIFE_DAYS,
            )
            .unwrap();
            bt.begin_core_metrics_if_due(2).unwrap();
            assert_eq!(bt.equities.timestamps_ms, vec![2 * MINUTE_MS]);
            close(bt.equities.usd_total_equity[0], 1000.0);
            close(bt.equities.btc_total_equity[0], 1000.0 / 20_000.0);
            bt.record_core_observation(2, false).unwrap();
            bt.finish_core_metrics().unwrap();
            let completed_balance = 0.02 * current_btc_close + 600.0;
            close(bt.equities.usd_total_equity[1], completed_balance);
            close(
                bt.equities.btc_total_equity[1],
                completed_balance / current_btc_close,
            );
            // Baseline and completed close each contribute one factual moment.
            let initial_moment = (200.0_f64 / 1000.0).powi(2) / 0.5;
            let final_moment = (200.0 / completed_balance).powi(2) / 0.5;
            close(
                number(&bt, Metric::PositionUtilizationLong),
                ((initial_moment + final_moment) / 2.0).sqrt(),
            );
            // Collateral price changes never become strategy profit.
            assert_eq!(bt.strategy_equity_series, vec![1000.0, 1000.0]);
        }
    }

    #[test]
    fn btc_fill_turnover_uses_execution_boundary_for_all_four_fill_paths() {
        let data = data(8);
        for side in [LONG, SHORT] {
            for closing in [false, true] {
                for at_bar_close in [false, true] {
                    for current_btc_close in [40_000.0, 80_000.0] {
                        let mut btc = Array1::from_elem(8, 20_000.0);
                        btc[2] = current_btc_close;
                        let mut bt = engine(&data, &btc, false, 1);
                        fixed_btc_wallet(&mut bt);
                        if closing {
                            let position = Position {
                                size: if side == LONG { 1.0 } else { -1.0 },
                                price: if side == LONG { 110.0 } else { 90.0 },
                            };
                            if side == LONG {
                                bt.positions.long[0] = position;
                            } else {
                                bt.positions.short[0] = position;
                            }
                        }
                        bt.configure_core_metrics(
                            request(&[Metric::Turnover]),
                            DEFAULT_EW_HALF_LIFE_DAYS,
                        )
                        .unwrap();
                        bt.begin_core_metrics_if_due(2).unwrap();
                        let exec = OrderFillExecution {
                            at_bar_close,
                            fee_fraction: 0.0,
                            ..execution(100.0)
                        };
                        let order = Order {
                            qty: if (side == LONG) != closing { 1.0 } else { -1.0 },
                            price: 100.0,
                            order_type: match (side, closing) {
                                (LONG, false) => OrderType::EntryInitialNormalLong,
                                (SHORT, false) => OrderType::EntryInitialNormalShort,
                                (LONG, true) => OrderType::CloseGridLong,
                                _ => OrderType::CloseGridShort,
                            },
                        };
                        match (side, closing) {
                            (LONG, false) => bt.process_entry_fill_long(2, 0, &order, exec),
                            (SHORT, false) => bt.process_entry_fill_short(2, 0, &order, exec),
                            (LONG, true) => bt.process_close_fill_long(2, 0, &order, exec),
                            _ => bt.process_close_fill_short(2, 0, &order, exec),
                        }
                        .unwrap();
                        bt.record_core_observation(2, false).unwrap();
                        bt.finish_core_metrics().unwrap();
                        let pre_balance = if at_bar_close {
                            0.02 * current_btc_close + 600.0
                        } else {
                            1000.0
                        };
                        close(number(&bt, Metric::Turnover), 200.0 / pre_balance * 1440.0);
                        assert_eq!(
                            bt.fills[0].timestamp_ms,
                            (2 + u64::from(at_bar_close)) * MINUTE_MS
                        );
                        // Actual trading wallet/revaluation still uses its original quote.
                        let net_pnl = if closing { -20.0 } else { 0.0 };
                        close(bt.balance.btc_cash_wallet, 0.02);
                        close(bt.balance.usd_cash_wallet, 600.0 + net_pnl);
                        close(
                            bt.balance.usd_total_balance,
                            0.02 * current_btc_close + 600.0 + net_pnl,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn btc_hsl_episode_anchor_shares_causal_pre_fill_balance_on_both_sides() {
        use crate::hsl_intent::{Action, Decision};
        use crate::hsl_support::Output;
        let data = data(8);
        for side in [LONG, SHORT] {
            for at_bar_close in [false, true] {
                for current_btc_close in [40_000.0, 80_000.0] {
                    let mut btc = Array1::from_elem(8, 20_000.0);
                    btc[2] = current_btc_close;
                    let mut bt = engine(&data, &btc, false, 1);
                    fixed_btc_wallet(&mut bt);
                    // Drive the observational lifecycle explicitly. Disabled controller
                    // policies prevent unrelated risk replay from replacing this fixture.
                    bt.backtest_params.equity_hard_stop_loss.hsl =
                        Some(super::super::hsl_runtime::Config::default());
                    let now = (2 + u64::from(at_bar_close)) * MINUTE_MS;
                    bt.hsl_report.observe(
                        (Some(side), Some(0)),
                        now as i64,
                        "pre_fill",
                        &Output {
                            decision: Some(Decision {
                                timestamp: now as i64,
                                action: Action::Panic,
                                red_at: Some(now as i64),
                                flat_at: None,
                                reason: "fixture",
                                signal_raw_fraction: 0.2,
                                signal_ema_fraction: 0.1,
                                numeric_range_approximation: false,
                            }),
                            events: vec![],
                            reasons: Default::default(),
                            observations: 1,
                            episodes: 1,
                        },
                    );
                    let position = Position {
                        size: if side == LONG { 1.0 } else { -1.0 },
                        price: if side == LONG { 110.0 } else { 90.0 },
                    };
                    if side == LONG {
                        bt.positions.long[0] = position;
                    } else {
                        bt.positions.short[0] = position;
                    }
                    let exec = OrderFillExecution {
                        at_bar_close,
                        fee_fraction: 0.0,
                        ..execution(100.0)
                    };
                    let order = Order {
                        qty: if side == LONG { -1.0 } else { 1.0 },
                        price: 100.0,
                        order_type: if side == LONG {
                            OrderType::ClosePanicLong
                        } else {
                            OrderType::ClosePanicShort
                        },
                    };
                    if side == LONG {
                        bt.process_close_fill_long(2, 0, &order, exec)
                    } else {
                        bt.process_close_fill_short(2, 0, &order, exec)
                    }
                    .unwrap();
                    let summary = bt.hsl_report.core_summary(true);
                    let pre_balance = if at_bar_close {
                        0.02 * current_btc_close + 600.0
                    } else {
                        1000.0
                    };
                    assert_eq!(summary.execution_episode_count, u64::from(!at_bar_close));
                    assert!(summary.episode_loss_valid);
                    let expected = if at_bar_close {
                        0.0
                    } else {
                        20.0 / pre_balance
                    };
                    close(summary.episode_loss_fraction_sum, expected);
                    close(summary.episode_loss_fraction_max, expected);
                }
            }
        }
    }

    #[test]
    fn btc_partial_terminal_artifacts_ignore_uncompleted_quote_with_or_without_collateral() {
        let mut data = data(8);
        data[[3, 0, CLOSE]] = 9999.0;
        for collateral in [false, true] {
            for future_btc_close in [20_000.0, 80_000.0] {
                let mut btc = Array1::from_elem(8, 20_000.0);
                btc[2] = 25_000.0;
                btc[3] = future_btc_close;
                let mut bt = engine(&data, &btc, false, 1);
                if collateral {
                    fixed_btc_wallet(&mut bt);
                }
                bt.positions.long[0] = Position {
                    size: 1.0,
                    price: 100.0,
                };
                bt.configure_core_metrics(
                    request(&[Metric::Pain, Metric::Completion]),
                    DEFAULT_EW_HALF_LIFE_DAYS,
                )
                .unwrap();
                bt.begin_core_metrics_if_due(2).unwrap();
                bt.record_core_observation(2, false).unwrap();
                bt.record_core_observation(3, true).unwrap();
                bt.finish_core_metrics().unwrap();
                assert_eq!(
                    bt.equities.timestamps_ms,
                    vec![2 * MINUTE_MS, 3 * MINUTE_MS]
                );
                let terminal_account = if collateral { 1100.0 } else { 1000.0 };
                close(bt.equities.usd_total_equity[1], terminal_account);
                close(bt.equities.btc_total_equity[1], terminal_account / 25_000.0);
                assert_eq!(bt.strategy_equity_series, vec![1000.0, 1000.0]);
                assert_eq!(
                    bt.core_metrics_result()
                        .unwrap()
                        .metadata
                        .equity_observation_count,
                    2
                );
                close(number(&bt, Metric::Pain), 0.0);
                close(number(&bt, Metric::Completion), 1.0 / 5.0);
            }
        }
    }
}
