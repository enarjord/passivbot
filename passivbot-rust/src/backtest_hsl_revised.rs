//! Factual simulator observations for the shared revised HSL reconstruction.
//! Shared by the staged Rust simulator execution path; public runtime gates stay closed.
use super::*;
use crate::hsl_revised_history::{self as history, PositionSide};
use crate::hsl_revised_prices as prices;
use crate::hsl_revised_snapshot as snapshot;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Inputs {
    pub snapshot: snapshot::Input,
    pub slots: [usize; 2],
    pub reasons: BTreeSet<String>,
}

impl Backtest<'_> {
    /// Observe the simulator after all fills in bar `k`. Candle timestamps name
    /// bar opens; valuation is available at the bar end, just as in the live
    /// candle projector. Fill timestamps and simulator execution order are kept.
    #[cfg(test)]
    pub(super) fn revised_hsl_inputs(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
    ) -> Result<Inputs, String> {
        self.revised_hsl_inputs_at(k, mode, pside, symbol, false)
    }

    /// A scope-flat observation uses only preceding completed candles and the
    /// just-observed execution. No current bar close is available at its open.
    pub(super) fn revised_hsl_inputs_at(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
        at_flat_boundary: bool,
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
        let now = close_time(k)? - if at_flat_boundary { 60_000 } else { 0 };
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
        // The source candle names its open, but its 1m close is an
        // instantaneous observation. Retain a close exactly at the inclusive
        // left edge without retaining any earlier price sample or fill.
        let first_row =
            (((start as i128 - self.first_timestamp_ms as i128 + 59_999).max(0) / 60_000) - 1)
                .max(0) as usize;
        let mut pairs = Vec::new();
        let mut flat_coin = None;
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
                if at_flat_boundary && position.size != 0.0 {
                    return Err(
                        "revised HSL boundary observation requires a flat selected scope".into(),
                    );
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
                    // Preserve explicit flat proof for a requested coin without
                    // manufacturing a mark for a market that has no quote.
                    if matches!(mode, snapshot::Mode::Coin) {
                        flat_coin = Some(snapshot::FlatCoin {
                            symbol: coin.clone(),
                            pside: side,
                            position_at: now,
                            fills_at: now,
                            history_start: start,
                        });
                    }
                    continue;
                }
                let current_valid = self.coin_is_valid_at(idx, k);
                if position.size != 0.0 && !current_valid {
                    return Err("missing held-position backtest HSL valuation".into());
                }
                let latest = self.coin_valid_range(idx).and_then(|(first, last)| {
                    let last = if at_flat_boundary {
                        k.checked_sub(1)?
                    } else {
                        k
                    }
                    .min(last);
                    (first.max(first_row)..=last)
                        .rev()
                        .find(|&j| self.coin_is_valid_at(idx, j))
                });
                if at_flat_boundary && fills.is_empty() && latest.is_none() {
                    // A freshly observed flat pair without any retained activity
                    // contributes no risk. Its first candle has not closed yet;
                    // do not borrow that future quote for another pair's boundary.
                    if matches!(mode, snapshot::Mode::Coin) {
                        flat_coin = Some(snapshot::FlatCoin {
                            symbol: coin.clone(),
                            pside: side,
                            position_at: now,
                            fills_at: now,
                            history_start: start,
                        });
                    }
                    continue;
                }
                let (mark, mark_at) = if let Some(latest) = latest {
                    (self.hlcvs_value(latest, idx, CLOSE), close_time(latest)?)
                } else if at_flat_boundary {
                    let fill = fills
                        .last()
                        .ok_or("missing flat-boundary valuation evidence")?;
                    (
                        fill.price.ok_or("missing flat-boundary fill price")?,
                        fill.timestamp,
                    )
                } else {
                    return Err("missing backtest HSL valuation history".into());
                };
                if !mark.is_finite() || mark <= 0.0 {
                    return Err("invalid current backtest HSL valuation".into());
                }
                let mut candles = Vec::new();
                for j in first_row..=k {
                    let end = close_time(j)?;
                    let candle_start = end - 60_000;
                    if end < start || end > now || !self.coin_is_valid_at(idx, j) {
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
                    mark_at,
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
                flat_coin,
                global_fill_sequence: true,
                fills_before_same_time_price: false,
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
    #[test]
    fn intrabar_entry_never_acquires_the_preceding_close_profit() {
        for side in [PositionSide::Long, PositionSide::Short] {
            let mut data = candles(4, 1);
            let entry = if side == PositionSide::Long {
                80.0
            } else {
                120.0
            };
            for k in 1..4 {
                data[[k, 0, CLOSE]] = entry;
            }
            let btc = Array1::from_elem(4, 20_000.0);
            let mut bt = make(&data, &btc);
            let qty = if side == PositionSide::Long {
                1.0
            } else {
                -1.0
            };
            fill(&mut bt, 1, 0, side, qty, entry);
            let input = bt
                .revised_hsl_inputs(2, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
            let points = &trace.episodes[0].points;
            let preceding_close = points
                .iter()
                .find(|p| p.timestamp == (FIRST + 60_000) as i64)
                .unwrap();
            assert!(!preceding_close.exposed);
            assert_eq!(preceding_close.upnl, 0.0);
            assert!(points.iter().all(|p| p.upnl == 0.0));
            assert!(points.last().unwrap().exposed);
        }
    }

    #[test]
    fn globally_sequenced_cross_pair_flat_and_reopen_survive_scope_composition() {
        for mode in [snapshot::Mode::Pside, snapshot::Mode::Unified] {
            let data = candles(4, 2);
            let btc = Array1::from_elem(4, 20_000.0);
            let mut bt = make(&data, &btc);
            fill(&mut bt, 0, 0, PositionSide::Long, 1.0, 100.0);
            fill(&mut bt, 0, 1, PositionSide::Long, 1.0, 200.0);
            fill(&mut bt, 1, 0, PositionSide::Long, -1.0, 90.0);
            fill(&mut bt, 1, 1, PositionSide::Long, -1.0, 190.0);
            fill(&mut bt, 1, 0, PositionSide::Long, 1.0, 90.0);
            let side = if matches!(mode, snapshot::Mode::Pside) {
                Some(PositionSide::Long)
            } else {
                None
            };
            let mut input = bt.revised_hsl_inputs(2, mode, side, None).unwrap();
            let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
            assert_eq!(trace.episodes.len(), 2);
            assert_eq!(
                trace.episodes[0].points.last().unwrap().timestamp,
                (FIRST + 60_000) as i64
            );
            assert!(trace.episodes[0].points.last().unwrap().flatten);
            assert_eq!(trace.episodes[1].opened_at, Some((FIRST + 60_000) as i64));
            // Same integer sequence values from separate exchange streams do
            // not establish global ordering without the explicit contract.
            input.snapshot.global_fill_sequence = false;
            assert!(snapshot::prepare(&input.snapshot)
                .unwrap()
                .boundaries
                .is_empty());
        }
    }

    #[test]
    fn explicitly_selected_flat_coin_needs_no_invented_prelisting_or_expired_quote() {
        for before_listing in [true, false] {
            let data = candles(6, 1);
            let btc = Array1::from_elem(6, 20_000.0);
            let mut bt = make(&data, &btc);
            if before_listing {
                bt.coin_first_valid_idx[0] = 4;
            } else {
                bt.coin_last_valid_idx[0] = 1;
            }
            let input = bt
                .revised_hsl_inputs(
                    2,
                    snapshot::Mode::Coin,
                    Some(PositionSide::Long),
                    Some("C0"),
                )
                .unwrap();
            assert!(input.snapshot.pairs.is_empty());
            assert!(input.snapshot.flat_coin.is_some());
            let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
            let point = &trace.episodes[0].points[0];
            assert!(!point.exposed);
            assert_eq!(point.pnl, 0.0);
            assert_eq!(point.upnl, 0.0);
            assert_eq!(point.timestamp, input.snapshot.now);
            assert!(bt
                .revised_hsl_inputs(
                    2,
                    snapshot::Mode::Coin,
                    Some(PositionSide::Long),
                    Some("UNKNOWN")
                )
                .and_then(|i| snapshot::prepare(&i.snapshot))
                .is_err());
        }
    }
    #[test]
    fn close_at_lookback_edge_uses_the_actual_preceding_close() {
        for side in [PositionSide::Long, PositionSide::Short] {
            let mut data = candles(1443, 1);
            let d = if side == PositionSide::Long {
                1.0
            } else {
                -1.0
            };
            for k in 2..1443 {
                data[[k, 0, CLOSE]] = 100.0 + 20.0 * d;
            }
            let btc = Array1::from_elem(1443, 20_000.0);
            let mut bt = make(&data, &btc);
            fill(&mut bt, 1, 0, side, d, 100.0);
            fill(&mut bt, 2, 0, side, -d, 100.0);
            let input = bt
                .revised_hsl_inputs(1441, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert_eq!(input.snapshot.start, (FIRST + 2 * 60_000) as i64);
            assert_eq!(input.snapshot.pairs[0].fills.len(), 1);
            assert_eq!(input.snapshot.pairs[0].prices[&input.snapshot.start], 100.0);
            let trace = crate::hsl_revised_trace::compose(&input.snapshot).unwrap();
            let first = &trace.episodes[0].points[0];
            assert_eq!(first.timestamp, input.snapshot.start);
            assert!(first.exposed);
            assert!(trace
                .episodes
                .iter()
                .flat_map(|e| &e.points)
                .all(|p| p.upnl == 0.0));
            assert!(trace.episodes[0].points.last().unwrap().flatten);
        }
    }
    fn enable_revised(bt: &mut Backtest<'_>, mode: &str, order_type: &str) {
        use super::super::revised_runtime::{Config, Policy};
        let policy = Policy {
            enabled: true,
            red_threshold: 0.05,
            ema_span_minutes: 1.0,
            cooldown_minutes_after_red: 10.0,
            restart_after_red_policy: "always".into(),
            panic_close_order_type: order_type.into(),
        };
        bt.backtest_params.equity_hard_stop_loss.signal_mode = mode.into();
        bt.backtest_params.equity_hard_stop_loss.revised = Some(Config {
            mode: mode.into(),
            intervention: "panic".into(),
            sides: [policy.clone(), policy.clone()],
            portfolio: Some(policy),
            coins: BTreeMap::new(),
        });
    }

    #[test]
    fn revised_runtime_drives_real_panic_orders_for_all_scopes() {
        for mode in ["coin", "pside", "unified"] {
            for order_type in ["market", "limit"] {
                let mut c = candles(5, 1);
                for k in 1..5 {
                    for f in [HIGH, LOW, CLOSE] {
                        c[[k, 0, f]] = 80.0;
                    }
                }
                let btc = Array1::from_elem(5, 1.0);
                let mut bt = make(&c, &btc);
                enable_revised(&mut bt, mode, order_type);
                fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
                bt.update_hard_stop_state(1).unwrap();
                assert_eq!(
                    bt.revised_action(LONG, 0),
                    crate::hsl_revised_controller::Action::Panic
                );
                bt.update_open_orders_all(1).unwrap();
                assert!(bt.open_orders.long[0].entries.is_empty());
                let close = &bt.open_orders.long[0].closes[0];
                assert_eq!(close.order.order_type, OrderType::ClosePanicLong);
                assert_eq!(close.order.qty, -10.0);
                assert_eq!(
                    close.execution_type,
                    if order_type == "market" {
                        orchestrator::ExecutionType::Market
                    } else {
                        orchestrator::ExecutionType::Limit
                    }
                );
            }
        }
    }

    #[test]
    fn revised_flat_boundary_cancels_reentry_without_future_candle_close() {
        for future_close in [10.0, 200.0] {
            let mut c = candles(4, 1);
            c[[1, 0, CLOSE]] = future_close;
            let btc = Array1::from_elem(4, 1.0);
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, "coin", "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
            fill(&mut bt, 1, 0, PositionSide::Long, -10.0, 80.0);
            assert_eq!(
                bt.revised_action(LONG, 0),
                crate::hsl_revised_controller::Action::Halted
            );
            let decision = bt
                .revised_hsl_scopes
                .iter()
                .find_map(|s| s.result.decision.as_ref())
                .unwrap();
            assert_eq!(decision.flat_at, Some((FIRST + 60_000) as i64));
            assert_eq!(decision.timestamp, (FIRST + 60_000) as i64);
            assert!(bt.open_orders.long[0].entries.is_empty());
        }
    }

    #[test]
    fn revised_unified_uses_one_controller_and_explicit_portfolio_execution_policy() {
        let c = candles(4, 1);
        let btc = Array1::from_elem(4, 1.0);
        let mut bt = make(&c, &btc);
        enable_revised(&mut bt, "unified", "limit");
        let cfg = bt
            .backtest_params
            .equity_hard_stop_loss
            .revised
            .as_mut()
            .unwrap();
        cfg.sides[LONG].panic_close_order_type = "market".into();
        cfg.sides[SHORT].panic_close_order_type = "market".into();
        fill(&mut bt, 0, 0, PositionSide::Short, -10.0, 80.0);
        bt.trading_enabled.short = false;
        bt.update_revised_hsl(1).unwrap();
        assert_eq!(bt.revised_hsl_scopes.len(), 1);
        assert_eq!(
            bt.revised_action(SHORT, 0),
            crate::hsl_revised_controller::Action::Panic
        );
        bt.update_open_orders_all(1).unwrap();
        let close = &bt.open_orders.short[0].closes[0];
        assert_eq!(close.execution_type, orchestrator::ExecutionType::Limit);
        bt.check_for_fills(2).unwrap();
        assert_eq!(bt.positions.short[0].size, 0.0);
    }
    #[test]
    fn revised_full_run_executes_red_before_next_bar_and_keeps_cooldown() {
        for mode in ["coin", "pside", "unified"] {
            let mut c = candles(8, 1);
            for k in 2..8 {
                for f in [HIGH, LOW, CLOSE] {
                    c[[k, 0, f]] = 80.0;
                }
            }
            let btc = Array1::from_elem(8, 1.0);
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, mode, "market");
            bt.bot_params_master.long.total_wallet_exposure_limit = 4.0;
            bt.bot_params_original[0].long.wallet_exposure_limit = 4.0;
            bt.bot_params_original[0].long.total_wallet_exposure_limit = 4.0;
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
            let (fills, _) = bt.run().unwrap();
            let panics: Vec<_> = fills
                .iter()
                .filter(|f| f.order_type == OrderType::ClosePanicLong)
                .collect();
            assert_eq!(panics.len(), 1, "{mode}");
            assert_eq!(panics[0].index, 3, "{mode}");
            assert_eq!(panics[0].fill_qty, -10.0);
            assert_eq!(bt.positions.long[0].size, 0.0);
            assert_eq!(
                bt.revised_action(LONG, 0),
                crate::hsl_revised_controller::Action::Halted
            );
            assert!(bt.open_orders.long[0].entries.is_empty());
        }
    }

    #[test]
    fn revised_cooldown_rebuilds_from_fills_and_expires_without_prior_runtime() {
        use crate::hsl_revised_controller::Action;
        let c = candles(1445, 1);
        let btc = Array1::from_elem(1445, 1.0);
        for mode in ["coin", "pside", "unified"] {
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, mode, "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
            fill(&mut bt, 1, 0, PositionSide::Long, -10.0, 80.0);
            for (restart, k, expected) in [
                ("always", 9, Action::Halted),
                ("always", 10, Action::Normal),
                ("never", 10, Action::Halted),
                ("never", 1442, Action::Normal),
            ] {
                let mut fresh = make(&c, &btc);
                enable_revised(&mut fresh, mode, "market");
                let cfg = fresh
                    .backtest_params
                    .equity_hard_stop_loss
                    .revised
                    .as_mut()
                    .unwrap();
                for p in &mut cfg.sides {
                    p.restart_after_red_policy = restart.into();
                }
                cfg.portfolio.as_mut().unwrap().restart_after_red_policy = restart.into();
                fresh.fills = bt.fills.clone();
                fresh.balance.usd_total_balance = bt.balance.usd_total_balance;
                fresh.balance.usd_cash_wallet = bt.balance.usd_cash_wallet;
                assert!(fresh.revised_hsl_scopes.is_empty());
                fresh.update_revised_hsl(k).unwrap();
                assert_eq!(
                    fresh.revised_action(LONG, 0),
                    expected,
                    "{mode} {restart} {k}"
                );
            }
        }
    }

    #[test]
    fn revised_partial_close_retains_protection_until_scope_flat() {
        use crate::hsl_revised_controller::Action;
        let c = candles(5, 2);
        let btc = Array1::from_elem(5, 1.0);
        for mode in ["pside", "unified"] {
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, mode, "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 120.0);
            fill(&mut bt, 0, 1, PositionSide::Long, 1.0, 200.0);
            bt.update_revised_hsl(1).unwrap();
            assert_eq!(bt.revised_action(LONG, 0), Action::Panic);
            fill(&mut bt, 2, 0, PositionSide::Long, -4.0, 100.0);
            bt.update_revised_hsl(2).unwrap();
            let orders = bt.revised_protective_orders(2).unwrap();
            assert_eq!(orders.len(), 2);
            assert_eq!(orders.iter().find(|o| o.symbol_idx == 0).unwrap().qty, -6.0);
            fill(&mut bt, 3, 0, PositionSide::Long, -6.0, 100.0);
            assert_eq!(bt.revised_action(LONG, 1), Action::Panic);
            fill(&mut bt, 3, 1, PositionSide::Long, -1.0, 200.0);
            assert_eq!(bt.revised_action(LONG, 0), Action::Halted);
            assert_eq!(bt.revised_action(LONG, 1), Action::Halted);
        }
    }

    #[test]
    fn revised_simulator_fill_tie_requires_sequence_and_matching_anchor() {
        let c = candles(3, 1);
        let btc = Array1::from_elem(3, 1.0);
        let mut bt = make(&c, &btc);
        fill(&mut bt, 0, 0, PositionSide::Long, 1.0, 100.0);
        fill(&mut bt, 1, 0, PositionSide::Long, -1.0, 80.0);
        for (global, anchored, eligible) in [
            (true, true, true),
            (false, true, false),
            (true, false, false),
        ] {
            let mut inputs = bt
                .revised_hsl_inputs_at(
                    1,
                    snapshot::Mode::Coin,
                    Some(PositionSide::Long),
                    Some("C0"),
                    true,
                )
                .unwrap();
            inputs.snapshot.global_fill_sequence = global;
            if !anchored {
                inputs.snapshot.pairs[0].fills_position_anchor = None;
            }
            let result = snapshot::prepare(&inputs.snapshot).unwrap();
            assert_eq!(
                result.reasons.contains("position_fill_timestamp_tie"),
                !eligible
            );
            assert_eq!(
                result
                    .boundaries
                    .last()
                    .is_some_and(|b| b.lifecycle_eligible),
                eligible
            );
        }
    }

    #[test]
    fn revised_first_bar_flat_uses_execution_without_unrelated_future_quotes() {
        let c = candles(3, 2);
        let btc = Array1::from_elem(3, 1.0);
        let mut bt = make(&c, &btc);
        enable_revised(&mut bt, "unified", "market");
        fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
        fill(&mut bt, 0, 0, PositionSide::Long, -10.0, 80.0);
        assert_eq!(
            bt.revised_action(LONG, 0),
            crate::hsl_revised_controller::Action::Halted
        );
    }
    #[test]
    fn revised_residual_close_gate_does_not_reenable_disabled_side_entries() {
        let c = candles(3, 1);
        let btc = Array1::from_elem(3, 1.0);
        for (side, direction) in [(LONG, 1.0), (SHORT, -1.0)] {
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, "unified", "market");
            let pside = if side == LONG {
                PositionSide::Long
            } else {
                PositionSide::Short
            };
            fill(&mut bt, 0, 0, pside, direction, 100.0);
            bt.update_revised_hsl(0).unwrap();
            assert_eq!(
                bt.revised_action(side, 0),
                crate::hsl_revised_controller::Action::Normal
            );
            let (enabled, orders) = if side == LONG {
                (&mut bt.trading_enabled.long, &mut bt.open_orders.long[0])
            } else {
                (&mut bt.trading_enabled.short, &mut bt.open_orders.short[0])
            };
            *enabled = false;
            orders.entries.push(BacktestOrder {
                order: Order {
                    qty: direction,
                    price: 100.0,
                    order_type: if side == LONG {
                        OrderType::EntryInitialNormalLong
                    } else {
                        OrderType::EntryInitialNormalShort
                    },
                },
                execution_type: orchestrator::ExecutionType::Market,
            });
            bt.check_for_fills(1).unwrap();
            assert_eq!(bt.fills.len(), 1);
        }
    }
}
