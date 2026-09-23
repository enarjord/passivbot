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
        self.revised_hsl_inputs_at_clipped(k, mode, pside, symbol, at_flat_boundary, None)
    }

    pub(super) fn revised_hsl_inputs_at_clipped(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
        at_flat_boundary: bool,
        cutoff: Option<(i64, usize)>,
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
        // Earlier completed episodes cannot change current permission or the
        // latest completed episode's cooldown. This is a suffix of the configured
        // lookback, never an extension beyond it.
        let start = cutoff.map_or(start, |(timestamp, _)| start.max(timestamp));
        // Simulator fills are appended in event-time order. Binary clipping and
        // row bounds keep each capture limited to the configured lookback.
        let fill_start = self
            .fills
            .partition_point(|f| (f.timestamp_ms as i128) < start as i128)
            .max(cutoff.map_or(0, |(_, consumed)| consumed));
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
            restart_after_red_policy: Some("always".into()),
            panic_close_order_type: order_type.into(),
        };
        bt.backtest_params.equity_hard_stop_loss.signal_mode = mode.into();
        bt.backtest_params.equity_hard_stop_loss.revised = Some(Config {
            mode: mode.into(),
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
        let mut c = candles(4, 1);
        c[[2, 0, LOW]] = 99.0;
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
            let report = bt.revised_hsl_report_value().unwrap().unwrap();
            // Native Python transport must preserve the complete serde schema,
            // including optional values and normal/panic/halted transitions.
            pyo3::prepare_freethreaded_python();
            pyo3::Python::with_gil(|py| {
                use pyo3::prelude::*;
                let actual = crate::python::revised_hsl_report_to_py(py, &bt).unwrap().unwrap();
                let encoded: String = py.import_bound("json").unwrap()
                    .call_method1("dumps", (actual,)).unwrap().extract().unwrap();
                let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
                assert_eq!(decoded, report, "direct report transport: {mode}");
            });

            assert_eq!(report["summary"]["triggers"], 1, "{mode}: {report}");
            assert_eq!(report["summary"]["restarts"], 0);
            assert_eq!(report["summary"]["panic_close_fills"], 1);
            assert!(report["summary"]["panic_close_loss"].as_f64().unwrap() > 0.0);
            assert!(report["samples"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["phase"] == "scope_flat"));
            if mode == "unified" {
                assert_eq!(report["summary"]["triggers_long"], 0);
                assert_eq!(report["summary"]["triggers_short"], 0);
            }

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
                    p.restart_after_red_policy = Some(restart.into());
                }
                cfg.portfolio.as_mut().unwrap().restart_after_red_policy = Some(restart.into());
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
    fn revised_simulator_fill_tie_distinguishes_exact_and_current_flat_timing() {
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
            assert!(!result.boundaries.is_empty());
            assert_eq!(
                result.reasons.contains("current_flat_timestamp_estimate"),
                false
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
    #[test]
    fn revised_panic_fill_uses_explicit_type_fees_slippage_and_close_eligibility() {
        for mode in ["coin", "pside", "unified"] {
            for side in [LONG, SHORT] {
                for execution in ["market", "limit"] {
                    for before_entry_start in [false, true] {
                        let mut c = candles(5, 1);
                        // Panic limits sit one tick through the previous touch.
                        // Move the next whole candle away so only market execution fills.
                        for f in [HIGH, LOW, CLOSE] {
                            c[[2, 0, f]] = if side == LONG { 99.0 } else { 101.0 };
                        }
                        c[[3, 0, LOW]] = 99.0;
                        c[[3, 0, HIGH]] = 101.0;
                        let btc = Array1::from_elem(5, 1.0);
                        let mut bt = make(&c, &btc);
                        enable_revised(&mut bt, mode, execution);
                        let opposite = if execution == "market" {
                            "limit"
                        } else {
                            "market"
                        };
                        bt.bot_params[0].long.hsl_panic_close_order_type = opposite.into();
                        bt.bot_params[0].short.hsl_panic_close_order_type = opposite.into();
                        bt.bot_params_master.long.hsl_panic_close_order_type = opposite.into();
                        bt.bot_params_master.short.hsl_panic_close_order_type = opposite.into();
                        bt.exchange_params_list[0].maker_fee = 0.001;
                        bt.exchange_params_list[0].taker_fee = 0.003;
                        bt.backtest_params.market_order_slippage_pct = 0.01;
                        let (pside, qty, basis) = if side == LONG {
                            (PositionSide::Long, 10.0, 120.0)
                        } else {
                            (PositionSide::Short, -10.0, 80.0)
                        };
                        fill(&mut bt, 0, 0, pside, qty, basis);
                        if before_entry_start {
                            bt.coin_trade_start_idx[0] = 4;
                        }
                        bt.trading_enabled.long = false;
                        bt.trading_enabled.short = false;
                        bt.update_revised_hsl(1).unwrap();
                        bt.update_open_orders_all(1).unwrap();
                        // No limit-price crossing; explicit market policy still fills.
                        bt.check_for_fills(2).unwrap();
                        if execution == "limit" {
                            assert_eq!(bt.fills.len(), 1, "{mode} {side} {execution}");
                            bt.check_for_fills(3).unwrap();
                        }
                        assert_eq!(
                            bt.fills.len(),
                            2,
                            "{mode} {side} {execution} warmup={before_entry_start}"
                        );
                        let close = bt.fills.last().unwrap();
                        let expected_price = if execution == "limit" {
                            if side == LONG {
                                99.99
                            } else {
                                100.01
                            }
                        } else if side == LONG {
                            98.01
                        } else {
                            102.01
                        };
                        let fee = if execution == "limit" { 0.001 } else { 0.003 };
                        assert_eq!(close.fill_qty, -qty);
                        assert_eq!(close.fill_price, expected_price);
                        assert!(
                            (close.fee_paid + qty.abs() * expected_price * 2.0 * fee).abs() < 1e-10
                        );
                        assert_eq!(
                            close.liquidity,
                            if execution == "limit" {
                                "maker"
                            } else {
                                "taker"
                            }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn revised_reporting_reset_cannot_change_panic_execution() {
        let c = candles(5, 1);
        let btc = Array1::from_elem(5, 1.0);
        let mut outcomes = Vec::new();
        for reset_report in [false, true] {
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, "unified", "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 120.0);
            bt.update_revised_hsl(1).unwrap();
            if reset_report {
                bt.revised_hsl_report = super::super::revised_report::Report::default();
            }
            bt.update_open_orders_all(1).unwrap();
            bt.check_for_fills(2).unwrap();
            outcomes.push((
                bt.positions.long[0].size,
                bt.revised_action(LONG, 0),
                bt.fills
                    .iter()
                    .map(|f| (f.timestamp_ms, f.fill_qty, f.fill_price, f.order_type))
                    .collect::<Vec<_>>(),
            ));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
    #[test]
    fn revised_panic_loss_ratio_revalues_collateral_before_fill() {
        for side in [PositionSide::Long, PositionSide::Short] {
            let long = side == PositionSide::Long;
            let mark = if long { 80.0 } else { 120.0 };
            let qty = if long { 10.0 } else { -10.0 };
            let mut c = candles(5, 1);
            for k in 1..5 {
                for f in [HIGH, LOW, CLOSE] {
                    c[[k, 0, f]] = mark;
                }
            }
            let btc = Array1::from_vec(vec![1.0, 1.0, 1.0, 2.0, 2.0]);
            let mut bt = make(&c, &btc);
            enable_revised(&mut bt, "coin", "market");
            fill(&mut bt, 0, 0, side, qty, 100.0);
            bt.balance.use_btc_collateral = true;
            bt.balance.btc_cash_wallet = 500.0;
            bt.balance.usd_cash_wallet = bt.balance.usd_total_balance - 500.0;
            bt.update_revised_hsl(2).unwrap();
            let pside = if long { LONG } else { SHORT };
            assert_eq!(
                bt.revised_action(pside, 0),
                crate::hsl_revised_controller::Action::Panic
            );
            let expected_equity = bt.balance.usd_cash_wallet + 500.0 * 2.0 - 400.0;
            assert!((bt.current_usd_equity_at(3) - expected_equity).abs() > 400.0);
            let order = Order {
                qty: -qty,
                price: mark,
                order_type: if long {
                    OrderType::ClosePanicLong
                } else {
                    OrderType::ClosePanicShort
                },
            };
            let exec = OrderFillExecution {
                price: mark,
                fee_rate: 0.0,
                liquidity: "taker",
            };
            if long {
                bt.process_close_fill_long(3, 0, &order, exec).unwrap();
            } else {
                bt.process_close_fill_short(3, 0, &order, exec).unwrap();
            }
            let metrics = bt.revised_hsl_report.metrics(1000.0, 3.0);
            assert_eq!(metrics.panic_close_loss_sum, 400.0);
            assert!(
                (metrics.panic_close_loss_drawdown_pct_mean - 400.0 / expected_equity).abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn revised_terminal_panic_fill_finishes_flat_latency_at_execution() {
        for terminal_price in [1.0, 52.0] {
            for mode in ["coin", "pside", "unified"] {
                let mut c = candles(8, 1);
                for k in 2..8 {
                    for f in [HIGH, LOW, CLOSE] {
                        c[[k, 0, f]] = if k == 2 { 80.0 } else { terminal_price };
                    }
                }
                let btc = Array1::from_elem(8, 1.0);
                let mut bt = make(&c, &btc);
                enable_revised(&mut bt, mode, "market");
                let config = bt
                    .backtest_params
                    .equity_hard_stop_loss
                    .revised
                    .as_mut()
                    .unwrap();
                for policy in config.sides.iter_mut().chain(config.portfolio.iter_mut()) {
                    policy.cooldown_minutes_after_red = 0.0;
                }
                bt.bot_params_master.long.total_wallet_exposure_limit = 4.0;
                bt.bot_params_original[0].long.wallet_exposure_limit = 4.0;
                bt.bot_params_original[0].long.total_wallet_exposure_limit = 4.0;
                fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
                let (fills, equities) = bt.run().unwrap();
                assert!(
                    bt.liquidated
                        && bt.balance.usd_total_balance <= bt.liquidation_equity_floor_usd(),
                    "{mode}"
                );
                assert_eq!(bt.balance.usd_total_balance > 0.0, terminal_price == 52.0);
                assert_eq!(bt.positions.long[0].size, 0.0);
                let panic = fills
                    .iter()
                    .find(|f| f.order_type == OrderType::ClosePanicLong)
                    .unwrap();
                assert_eq!(panic.index, 3);
                assert_eq!(
                    equities.timestamps_ms.last().copied(),
                    Some(panic.timestamp_ms)
                );
                assert_eq!(
                    bt.strategy_equity_series.len(),
                    equities.timestamps_ms.len()
                );
                for emas in &bt.revised_hsl_report.signal_emas {
                    assert_eq!(
                        emas.len() + 1,
                        bt.strategy_equity_series.len(),
                        "terminal strategy sample must not duplicate a stale EMA"
                    );
                }
                let metrics = bt.revised_hsl_report.metrics(1000.0, 3.0);
                // RED at bar 2 close and the real flatten at bar 3 open coincide.
                assert_eq!(metrics.flatten_time_minutes_mean, 0.0, "{mode}");
                assert_eq!(metrics.duration_minutes_mean, 0.0, "{mode}");
                assert_eq!(metrics.time_in_red_pct, 0.0, "{mode}");
                assert_eq!(
                    bt.revised_hsl_report.summary.observed_minutes, 0.0,
                    "{mode}"
                );
                assert_eq!(metrics.restarts, 0);
                let report = bt.revised_hsl_report_value().unwrap().unwrap();
                assert!(report["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e["kind"] == "flat" && e["observed_at"] == panic.timestamp_ms));
            }
        }
    }

    #[test]
    fn latest_episode_suffix_preserves_current_decision() {
        use crate::hsl_revised_controller::Restart;
        use crate::hsl_revised_evaluator::{evaluate, Input};
        let data = candles(50, 2);
        let btc = Array1::from_elem(50, 50000.);
        for (_mode, side, coin) in [
            (snapshot::Mode::Coin, Some(LONG), Some(0)),
            (snapshot::Mode::Pside, Some(LONG), None),
            (snapshot::Mode::Unified, None, None),
        ] {
            let mode = || match (side, coin) {
                (_, Some(_)) => snapshot::Mode::Coin,
                (Some(_), _) => snapshot::Mode::Pside,
                _ => snapshot::Mode::Unified,
            };
            let mut bt = make(&data, &btc);
            for (k, qty) in [(3, 1.), (8, -1.), (9, 1.), (15, -1.), (18, 1.)] {
                fill(
                    &mut bt,
                    k,
                    0,
                    PositionSide::Long,
                    qty,
                    if qty > 0. { 100. } else { 70. },
                );
                let cutoff = bt.revised_history_cutoff(side, coin);
                let side_name = side.map(|_| PositionSide::Long);
                let symbol = coin.map(|c| bt.backtest_params.coins[c].as_str());
                let full = bt
                    .revised_hsl_inputs_at(k, mode(), side_name, symbol, false)
                    .unwrap();
                let trimmed = bt
                    .revised_hsl_inputs_at_clipped(k, mode(), side_name, symbol, false, cutoff)
                    .unwrap();
                assert!(trimmed.snapshot.start >= full.snapshot.start);
                let config = |snapshot| Input {
                    snapshot,
                    slots: 1,
                    span: 3.5,
                    threshold: 0.06,
                    cooldown_ms: 600_000,
                    restart: Restart::Always,
                };
                let expected = evaluate(config(full.snapshot)).unwrap().decision.unwrap();
                let actual = evaluate(config(trimmed.snapshot))
                    .unwrap()
                    .decision
                    .unwrap();
                assert_eq!(actual.action, expected.action);
                assert_eq!(actual.flat_at, expected.flat_at);
                assert!((actual.raw - expected.raw).abs() < 1e-12);
                assert!((actual.ema - expected.ema).abs() < 1e-12);
                // Discarding this optimization cannot change the selected prefix.
                bt.revised_hsl_cutoffs.clear();
                assert_eq!(bt.revised_history_cutoff(side, coin), cutoff);
            }
        }
    }
    #[test]
    fn incremental_marks_match_fresh_replay_through_fills_recovery_and_budget_changes() {
        use crate::backtest::revised_runtime::{Config, Policy};
        use crate::hsl_revised_controller::Restart;
        use crate::hsl_revised_evaluator::{evaluate, Input};
        let mut data = candles(75, 2);
        for k in 0..75 {
            let price = match k {
                0..=5 => 100.,
                6..=16 => 70.,
                17..=27 => 120.,
                28..=43 => 80.,
                _ => 110.,
            };
            for c in 0..2 {
                for f in [HIGH, LOW, CLOSE] {
                    data[[k, c, f]] = price + c as f64;
                }
            }
        }
        let btc = Array1::from_elem(75, 50000.);
        for name in ["coin", "pside", "unified"] {
            for restart in ["always", "never"] {
                let mode = || match name {
                    "coin" => snapshot::Mode::Coin,
                    "pside" => snapshot::Mode::Pside,
                    _ => snapshot::Mode::Unified,
                };
                let mut bt = make(&data, &btc);
                let policy = Policy {
                    enabled: true,
                    red_threshold: 0.06,
                    ema_span_minutes: 3.5,
                    cooldown_minutes_after_red: 5.,
                    restart_after_red_policy: Some(restart.into()),
                    panic_close_order_type: "limit".into(),
                };
                bt.backtest_params.equity_hard_stop_loss.revised = Some(Config {
                    mode: name.into(),
                    sides: [policy.clone(), policy.clone()],
                    portfolio: (name == "unified").then_some(policy.clone()),
                    coins: Default::default(),
                });
                for k in 2..70 {
                    bt.current_step = k;
                    for (when, coin, side, qty) in [
                        (3, 0, PositionSide::Long, 2.),
                        (5, 1, PositionSide::Short, -1.),
                        (7, 0, PositionSide::Long, 1.),
                        (10, 0, PositionSide::Long, 0.5),
                        (10, 0, PositionSide::Long, -0.5),
                        (12, 0, PositionSide::Long, -1.),
                        (15, 0, PositionSide::Long, -2.),
                        (15, 0, PositionSide::Long, 1.),
                        (20, 0, PositionSide::Long, -1.),
                        (30, 1, PositionSide::Short, 1.),
                        (32, 0, PositionSide::Long, 1.),
                        (34, 0, PositionSide::Long, -0.5),
                        (40, 0, PositionSide::Long, -0.5),
                        (60, 0, PositionSide::Long, 1.),
                    ] {
                        if k == when {
                            fill(&mut bt, k, coin, side, qty, data[[k, coin, CLOSE]]);
                        }
                    }
                    if k == 25 || k == 45 {
                        let delta = if k == 25 { 500. } else { -400. };
                        bt.balance.usd_total_balance += delta;
                        bt.balance.usd_cash_wallet += delta;
                    }
                    if k == 47 {
                        bt.effective_n_positions.long = 1;
                    }
                    if k % 13 == 0 {
                        bt.revised_hsl_traces.clear();
                    }
                    if k % 11 == 0 {
                        bt.revised_hsl_cutoffs.clear();
                        bt.revised_hsl_scopes.clear();
                    }
                    bt.update_revised_hsl(k).unwrap();
                    for scope in &bt.revised_hsl_scopes {
                        let actual = scope.result.decision.as_ref().unwrap();
                        let side = scope.side.map(|s| {
                            if s == LONG {
                                PositionSide::Long
                            } else {
                                PositionSide::Short
                            }
                        });
                        let symbol = scope.coin.map(|c| bt.backtest_params.coins[c].as_str());
                        let full = bt
                            .revised_hsl_inputs_at(k, mode(), side, symbol, false)
                            .unwrap();
                        let expected = evaluate(Input {
                            snapshot: full.snapshot,
                            slots: if name == "coin" {
                                full.slots[scope.side.unwrap()] as u64
                            } else {
                                1
                            },
                            span: policy.ema_span_minutes,
                            threshold: policy.red_threshold,
                            cooldown_ms: 300_000,
                            restart: if restart == "always" {
                                Restart::Always
                            } else {
                                Restart::Never
                            },
                        })
                        .unwrap()
                        .decision
                        .unwrap();
                        assert_eq!(
                            actual.action, expected.action,
                            "mode={name} restart={restart} k={k}"
                        );
                        assert_eq!(
                            actual.flat_at, expected.flat_at,
                            "mode={name} restart={restart} k={k}"
                        );
                        assert_eq!(
                            actual.red_at, expected.red_at,
                            "mode={name} restart={restart} k={k}"
                        );
                        assert!(
                            (actual.raw - expected.raw).abs() < 1e-12,
                            "mode={name} k={k}"
                        );
                        assert!(
                            (actual.ema - expected.ema).abs() < 1e-12,
                            "mode={name} k={k} actual={} expected={}",
                            actual.ema,
                            expected.ema
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn lookback_crossing_rebuilds_incomplete_current_episode() {
        use crate::backtest::revised_runtime::{Config, Policy};
        use crate::hsl_revised_controller::Restart;
        use crate::hsl_revised_evaluator::{evaluate, Input};
        let mut data = candles(1450, 1);
        for k in 0..1450 {
            for field in [HIGH, LOW, CLOSE] {
                data[[k, 0, field]] = if k < 800 { 110. } else { 80. };
            }
        }
        let btc = Array1::from_elem(1450, 50000.);
        let mut bt = make(&data, &btc);
        let policy = Policy {
            enabled: true,
            red_threshold: 0.06,
            ema_span_minutes: 308.5,
            cooldown_minutes_after_red: 5.,
            restart_after_red_policy: Some("always".into()),
            panic_close_order_type: "limit".into(),
        };
        bt.backtest_params.equity_hard_stop_loss.revised = Some(Config {
            mode: "coin".into(),
            sides: [policy.clone(), policy.clone()],
            portfolio: None,
            coins: Default::default(),
        });
        for k in [2, 3, 4, 1441, 1442, 1443, 1444, 1445] {
            bt.current_step = k;
            if k == 3 {
                fill(&mut bt, k, 0, PositionSide::Long, 1., 100.);
            }
            bt.update_revised_hsl(k).unwrap();
            let actual = bt.revised_hsl_scopes[0].result.decision.as_ref().unwrap();
            let full = bt
                .revised_hsl_inputs_at(
                    k,
                    snapshot::Mode::Coin,
                    Some(PositionSide::Long),
                    Some("C0"),
                    false,
                )
                .unwrap();
            let expected = evaluate(Input {
                snapshot: full.snapshot,
                slots: full.slots[LONG] as u64,
                span: policy.ema_span_minutes,
                threshold: policy.red_threshold,
                cooldown_ms: 300_000,
                restart: Restart::Always,
            })
            .unwrap()
            .decision
            .unwrap();
            assert_eq!(actual.action, expected.action, "k={k}");
            assert_eq!(actual.red_at, expected.red_at, "k={k}");
            assert!((actual.raw - expected.raw).abs() < 1e-12, "k={k}");
            assert!((actual.ema - expected.ema).abs() < 1e-12, "k={k}");
        }
    }
}
