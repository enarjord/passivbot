//! Factual simulator observations for the shared revised HSL reconstruction.
//! No revised trading path is activated by this adapter.
use super::*;
use crate::hsl_revised_history::{self as history, PositionSide};
use crate::hsl_revised_prices as prices;
use crate::hsl_revised_snapshot as snapshot;
use std::collections::{BTreeMap, BTreeSet};

#[allow(dead_code)] // Runtime consumers land in the next integration step.
pub(super) struct Inputs {
    pub snapshot: snapshot::Input,
    pub slots: [usize; 2],
    pub reasons: BTreeSet<String>,
}

impl Backtest<'_> {
    /// Observe the simulator after all fills in bar `k`. Candle timestamps name
    /// bar opens; valuation is available at the bar end, just as in the live
    /// candle projector. Fill timestamps and simulator execution order are kept.
    #[allow(dead_code)] // Connected by the subsequent revised runtime dispatch.
    pub(super) fn revised_hsl_inputs(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
    ) -> Result<Inputs, String> {
        if self.interval_ms != 60_000 || k >= self.hlcvs.shape()[0] {
            return Err("revised HSL backtest inputs require a valid 1m bar".into());
        }
        let days = self.backtest_params.pnls_max_lookback_days;
        if !days.is_finite() || !(1.0..=90.0).contains(&days) {
            return Err("revised HSL requires a finite lookback in [1,90] days".into());
        }
        match mode {
            snapshot::Mode::Coin if pside.is_none() || symbol.is_none() => {
                return Err("coin requires side and symbol".into());
            }
            snapshot::Mode::Pside if pside.is_none() || symbol.is_some() => {
                return Err("pside requires only side".into());
            }
            snapshot::Mode::Unified if pside.is_some() || symbol.is_some() => {
                return Err("unified has no side/symbol selector".into());
            }
            _ => {}
        }
        let close_time = |index: usize| -> Result<i64, String> {
            self.first_timestamp_ms
                .checked_add(
                    u64::try_from(index)
                        .ok()
                        .and_then(|i| i.checked_add(1))
                        .and_then(|i| i.checked_mul(self.interval_ms))
                        .ok_or("backtest HSL timestamp overflow")?,
                )
                .and_then(|t| i64::try_from(t).ok())
                .ok_or_else(|| "backtest HSL timestamp overflow".into())
        };
        let now = close_time(k)?;
        let start = now
            .checked_sub((days * 86_400_000.0).round() as i64)
            .ok_or("backtest HSL lookback overflow")?;
        // Simulator fills are appended in event-time order. Binary clipping and
        // row bounds keep each capture limited to the configured lookback.
        let fill_start = self
            .fills
            .partition_point(|f| (f.timestamp_ms as i128) < start as i128);
        let fill_end = self
            .fills
            .partition_point(|f| (f.timestamp_ms as i128) <= now as i128);
        let first_row =
            ((start as i128 - self.first_timestamp_ms as i128 + 59_999).max(0) / 60_000) as usize;
        let mut pairs = Vec::new();
        let mut reasons = BTreeSet::new();
        for idx in 0..self.n_coins {
            let coin = &self.backtest_params.coins[idx];
            if symbol.is_some_and(|s| s != coin) {
                continue;
            }
            // Select the requested side before requiring unrelated valuation.
            for (side, position) in [
                (PositionSide::Long, self.positions.long[idx]),
                (PositionSide::Short, self.positions.short[idx]),
            ] {
                if pside.is_some_and(|s| s != side) {
                    continue;
                }
                let mut fills = Vec::new();
                for (offset, fill) in self.fills[fill_start..fill_end].iter().enumerate() {
                    let sequence = fill_start + offset;
                    if fill.coin != *coin
                        || fill.order_type.is_long() != (side == PositionSide::Long)
                    {
                        continue;
                    }
                    if !fill.order_type.is_entry() && !fill.order_type.is_close() {
                        return Err("unattributed backtest HSL execution".into());
                    }
                    let timestamp = i64::try_from(fill.timestamp_ms)
                        .map_err(|_| "backtest HSL fill timestamp overflow")?;
                    if timestamp < start || timestamp > now {
                        continue;
                    }
                    if fill.index > k {
                        return Err("future backtest HSL execution".into());
                    }
                    let sequence =
                        i64::try_from(sequence).map_err(|_| "backtest HSL sequence overflow")?;
                    fills.push(history::Fill {
                        identity: sequence.to_string(),
                        timestamp,
                        delta: Some(fill.fill_qty),
                        price: Some(fill.fill_price),
                        realized: Some(fill.pnl),
                        fee: Some(fill.fee_paid),
                        sequence: Some(sequence),
                        revision: 0,
                    });
                }
                if position.size == 0.0 && fills.is_empty() && !self.coin_is_valid_at(idx, k) {
                    // A pre-listing/expired flat pair has no scope contribution.
                    continue;
                }
                let current_valid = self.coin_is_valid_at(idx, k);
                if position.size != 0.0 && !current_valid {
                    return Err("missing held-position backtest HSL valuation".into());
                }
                let latest = self
                    .coin_valid_range(idx)
                    .and_then(|(first, last)| {
                        let last = last.min(k);
                        (first.max(first_row)..=last)
                            .rev()
                            .find(|&j| self.coin_is_valid_at(idx, j))
                    })
                    .ok_or("missing backtest HSL valuation history")?;
                let mark = self.hlcvs_value(latest, idx, CLOSE);
                if !mark.is_finite() || mark <= 0.0 {
                    return Err("invalid current backtest HSL valuation".into());
                }
                let mut candles = Vec::new();
                for j in first_row..=k {
                    let end = close_time(j)?;
                    let candle_start = end - 60_000;
                    if candle_start < start || !self.coin_is_valid_at(idx, j) {
                        continue;
                    }
                    candles.push(prices::Candle {
                        start: candle_start,
                        minutes: 1,
                        open: None,
                        high: None,
                        low: None,
                        close: Some(self.hlcvs_value(j, idx, CLOSE)),
                        available_at: Some(end),
                    });
                }
                let projected = prices::minute_prices(&prices::Input {
                    start,
                    end: now,
                    candles,
                })?;
                reasons.extend(projected.reasons);
                let projected: BTreeMap<_, _> = projected
                    .rows
                    .into_iter()
                    .map(|p| (p.timestamp, p.close))
                    .collect();
                let market = &self.exchange_params_list[idx];
                let revision = u64::try_from(k).map_err(|_| "backtest HSL revision overflow")?;
                pairs.push(snapshot::Pair {
                    symbol: coin.clone(),
                    position: history::Position {
                        size: position.size,
                        basis: position.price,
                        mark,
                        multiplier: market.c_mult,
                        quantity_step: Some(market.qty_step),
                        inverse: false,
                        pside: side,
                    },
                    position_at: now,
                    mark_at: close_time(latest)?,
                    fills_started_at: Some(now),
                    fills_at: Some(now),
                    prices_at: now,
                    fills,
                    prices: projected,
                    revisions: [revision; 4],
                    fills_position_anchor: Some(snapshot::PositionAnchor {
                        position_at: now,
                        size: position.size,
                        basis: position.price,
                        multiplier: market.c_mult,
                        inverse: false,
                        pside: side,
                        revision,
                    }),
                });
            }
        }
        Ok(Inputs {
            snapshot: snapshot::Input {
                now,
                start,
                balance: self.balance.usd_total_balance,
                balance_at: now,
                config_at: now,
                max_current_age_ms: 0,
                mode,
                pside,
                symbol: symbol.map(str::to_owned),
                pairs,
            },
            slots: [
                self.hard_stop_coin_slot_n_positions(LONG),
                self.hard_stop_coin_slot_n_positions(SHORT),
            ],
            reasons,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    const DAY: u64 = 86_400_000;
    const FIRST: u64 = 10 * DAY;

    fn params(rows: usize, coins: usize) -> BacktestParams {
        BacktestParams {
            starting_balance: 1000.0,
            maker_fee: 0.0,
            taker_fee: 0.0,
            coins: (0..coins).map(|i| format!("C{i}")).collect(),
            active_coin_indices: None,
            first_timestamp_ms: FIRST,
            requested_start_timestamp_ms: FIRST,
            first_valid_indices: vec![0; coins],
            last_valid_indices: vec![rows - 1; coins],
            warmup_minutes: vec![0; coins],
            trade_start_indices: vec![0; coins],
            global_warmup_bars: 0,
            btc_collateral_cap: 0.0,
            btc_collateral_ltv_cap: None,
            metrics_only: false,
            skip_btc_analysis: false,
            filter_by_min_effective_cost: false,
            dynamic_wel_by_tradability: true,
            hedge_mode: true,
            max_realized_loss_pct: 1.0,
            pnls_max_lookback_days: 1.0,
            liquidation_threshold: 0.05,
            equity_hard_stop_loss: EquityHardStopLossConfig::default(),
            market_orders_allowed: false,
            market_order_near_touch_threshold: 0.001,
            market_order_slippage_pct: 0.0,
            forager_score_hysteresis_pct: 0.0,
            candle_interval_minutes: 1,
        }
    }

    fn candles(rows: usize, coins: usize) -> Array3<f64> {
        Array3::from_shape_fn((rows, coins, 4), |(_, coin, feature)| {
            if feature == VOLUME {
                1.0
            } else {
                100.0 + coin as f64 * 100.0
            }
        })
    }

    fn make<'a>(candles: &'a Array3<f64>, btc: &'a Array1<f64>) -> Backtest<'a> {
        let coins = candles.shape()[1];
        let mut bot = BotParamsPair::default();
        bot.long.n_positions = 4;
        bot.short.n_positions = 2;
        Backtest::new(
            candles.view(),
            btc.view(),
            vec![bot; coins],
            vec![
                ExchangeParams {
                    c_mult: 2.0,
                    qty_step: 0.1,
                    price_step: 0.01,
                    ..Default::default()
                };
                coins
            ],
            &params(candles.shape()[0], coins),
        )
    }

    fn fill(
        bt: &mut Backtest<'_>,
        k: usize,
        coin: usize,
        side: PositionSide,
        qty: f64,
        price: f64,
    ) {
        let entry =
            qty * if side == PositionSide::Long {
                1.0
            } else {
                -1.0
            } > 0.0;
        let order_type = match (side, entry) {
            (PositionSide::Long, true) => OrderType::EntryInitialNormalLong,
            (PositionSide::Long, false) => OrderType::CloseGridLong,
            (PositionSide::Short, true) => OrderType::EntryInitialNormalShort,
            (PositionSide::Short, false) => OrderType::CloseGridShort,
        };
        let order = Order {
            qty,
            price,
            order_type,
        };
        let exec = OrderFillExecution {
            price,
            fee_rate: 0.001,
            liquidity: "maker",
        };
        match (side, entry) {
            (PositionSide::Long, true) => bt.process_entry_fill_long(k, coin, &order, exec),
            (PositionSide::Short, true) => bt.process_entry_fill_short(k, coin, &order, exec),
            (PositionSide::Long, false) => {
                bt.process_close_fill_long(k, coin, &order, exec).unwrap()
            }
            (PositionSide::Short, false) => {
                bt.process_close_fill_short(k, coin, &order, exec).unwrap()
            }
        }
    }

    #[test]
    fn simulator_native_quantities_cashflows_and_close_clock_reach_shared_replay() {
        for side in [PositionSide::Long, PositionSide::Short] {
            let data = candles(6, 1);
            let btc = Array1::from_elem(6, 20_000.0);
            let mut bt = make(&data, &btc);
            let d = if side == PositionSide::Long {
                1.0
            } else {
                -1.0
            };
            fill(&mut bt, 1, 0, side, 3.0 * d, 100.0);
            fill(&mut bt, 2, 0, side, -d, 100.0 - 10.0 * d);
            // Derived after-states must never become the reconstructed position.
            bt.fills[0].position_size = 987.0;
            bt.fills[0].position_price = 123.0;
            let input = bt
                .revised_hsl_inputs(3, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert_eq!(input.snapshot.now, (FIRST + 4 * 60_000) as i64);
            assert_eq!(input.snapshot.balance, bt.balance.usd_total_balance);
            assert_eq!(input.snapshot.pairs.len(), 1);
            let pair = &input.snapshot.pairs[0];
            assert_eq!(pair.position.size, 2.0 * d);
            assert_eq!(pair.position.basis, 100.0);
            assert_eq!(pair.fills[0].delta, Some(3.0 * d));
            assert_eq!(pair.fills[1].realized, Some(-20.0));
            assert_eq!(pair.fills[0].fee, Some(-0.6));
            assert_eq!(pair.fills[1].sequence, Some(1));
            assert!(pair.fills.iter().all(|f| f.timestamp < pair.position_at));
            assert_eq!(pair.prices[&((FIRST + 60_000) as i64)], 100.0);
            let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
            assert!(
                trace
                    .episodes
                    .last()
                    .unwrap()
                    .points
                    .last()
                    .unwrap()
                    .exposed
            );
            assert!(!trace.reasons.contains("position_fill_timestamp_tie"));
            assert!(input.reasons.contains("backfilled_price"));
        }
    }

    #[test]
    fn same_bar_flats_and_reopens_keep_simulator_order() {
        let data = candles(5, 1);
        let btc = Array1::from_elem(5, 20_000.0);
        let mut bt = make(&data, &btc);
        fill(&mut bt, 1, 0, PositionSide::Long, 1.0, 100.0);
        fill(&mut bt, 2, 0, PositionSide::Long, -1.0, 80.0);
        fill(&mut bt, 2, 0, PositionSide::Long, 1.0, 80.0);
        fill(&mut bt, 2, 0, PositionSide::Long, -1.0, 80.0);
        let input = bt
            .revised_hsl_inputs(2, snapshot::Mode::Unified, None, None)
            .unwrap();
        let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
        assert_eq!(trace.episodes.len(), 3);
        assert_eq!(
            trace.episodes[1].opened_at,
            Some((FIRST + 2 * 60_000) as i64)
        );
        assert!(trace.episodes[..2]
            .iter()
            .all(|e| e.points.last().unwrap().flatten));
    }

    #[test]
    fn unified_includes_disabled_side_exposure_and_raw_balance() {
        let data = candles(5, 2);
        let btc = Array1::from_elem(5, 20_000.0);
        let mut bt = make(&data, &btc);
        bt.bot_params_master.short.n_positions = 0;
        bt.bot_params_master.short.total_wallet_exposure_limit = 0.0;
        fill(&mut bt, 1, 1, PositionSide::Short, -2.0, 200.0);
        bt.balance.usd_total_balance = 995.123;
        bt.balance.usd_total_balance_rounded = 1000.0;
        let unified = bt
            .revised_hsl_inputs(2, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(unified.snapshot.balance, 995.123);
        assert_eq!(unified.snapshot.pairs.len(), 4);
        assert!(unified
            .snapshot
            .pairs
            .iter()
            .any(|p| p.symbol == "C1" && p.position.size == -2.0));
        let long = bt
            .revised_hsl_inputs(2, snapshot::Mode::Pside, Some(PositionSide::Long), None)
            .unwrap();
        assert_eq!(long.snapshot.pairs.len(), 2);
        assert!(long
            .snapshot
            .pairs
            .iter()
            .all(|p| p.position.pside == PositionSide::Long));
    }

    #[test]
    fn dynamic_slot_policy_preserves_configured_or_effective_divisor() {
        let data = candles(3, 1);
        let btc = Array1::from_elem(3, 20_000.0);
        let mut bt = make(&data, &btc);
        bt.effective_n_positions.long = 1;
        bt.effective_n_positions.short = 0;
        let input = bt
            .revised_hsl_inputs(1, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(input.slots, [1, 0]);
        bt.backtest_params.dynamic_wel_by_tradability = false;
        let input = bt
            .revised_hsl_inputs(1, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(input.slots, [4, 2]);
    }

    #[test]
    fn retained_flat_history_survives_delisting_without_fresh_irrelevant_mark() {
        let data = candles(8, 1);
        let btc = Array1::from_elem(8, 20_000.0);
        let mut bt = make(&data, &btc);
        fill(&mut bt, 1, 0, PositionSide::Long, 1.0, 100.0);
        fill(&mut bt, 2, 0, PositionSide::Long, -1.0, 80.0);
        bt.coin_last_valid_idx[0] = 2;
        let input = bt
            .revised_hsl_inputs(6, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(input.snapshot.pairs.len(), 1);
        assert_eq!(input.snapshot.pairs[0].mark_at, (FIRST + 3 * 60_000) as i64);
        let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
        assert!(trace.reasons.contains("stale_flat_mark"));
        assert_eq!(trace.episodes.len(), 2);
        // No held exposure may use that stale valuation.
        bt.positions.long[0] = Position {
            size: 1.0,
            price: 100.0,
        };
        assert!(bt
            .revised_hsl_inputs(6, snapshot::Mode::Unified, None, None)
            .is_err());
    }

    #[test]
    fn bounded_window_ignores_old_fills_and_candles_at_inclusive_edge() {
        let data = candles(1443, 1);
        let btc = Array1::from_elem(1443, 20_000.0);
        let mut bt = make(&data, &btc);
        fill(&mut bt, 0, 0, PositionSide::Long, 1.0, 100.0);
        fill(&mut bt, 1, 0, PositionSide::Long, -1.0, 80.0);
        fill(&mut bt, 2, 0, PositionSide::Long, 1.0, 100.0);
        let input = bt
            .revised_hsl_inputs(
                1441,
                snapshot::Mode::Coin,
                Some(PositionSide::Long),
                Some("C0"),
            )
            .unwrap();
        assert_eq!(input.snapshot.start, (FIRST + 2 * 60_000) as i64);
        assert_eq!(input.snapshot.pairs[0].fills.len(), 1);
        assert_eq!(
            input.snapshot.pairs[0].fills[0].timestamp,
            input.snapshot.start
        );
        assert_eq!(input.snapshot.pairs[0].prices.len(), 1441);
        assert!(input.snapshot.pairs[0]
            .prices
            .keys()
            .all(|t| *t >= input.snapshot.start));
    }

    #[test]
    fn source_column_mapping_and_future_prices_cannot_leak() {
        let mut data = candles(5, 2);
        data[[4, 1, CLOSE]] = 9999.0;
        let btc = Array1::from_elem(5, 20_000.0);
        let mut bt = make(&data, &btc);
        bt.active_coin_indices.swap(0, 1);
        let input = bt
            .revised_hsl_inputs(
                2,
                snapshot::Mode::Coin,
                Some(PositionSide::Long),
                Some("C0"),
            )
            .unwrap();
        assert_eq!(input.snapshot.pairs[0].position.mark, 200.0);
        assert!(input.snapshot.pairs[0].prices.values().all(|p| *p == 200.0));
        bt.interval_ms = 5 * 60_000;
        assert!(bt
            .revised_hsl_inputs(2, snapshot::Mode::Unified, None, None)
            .is_err());
    }
}
