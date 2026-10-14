//! Factual simulator observations for the shared HSL reconstruction.
//! Simulator observations feed the same evaluator used by live HSL.
use super::*;
use crate::hsl_history::{self as history, PositionSide};
use crate::hsl_prices as prices;
use crate::hsl_snapshot as snapshot;
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
    pub(super) fn hsl_inputs(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
    ) -> Result<Inputs, String> {
        self.hsl_inputs_at(k, mode, pside, symbol, false)
    }

    /// A scope-flat observation uses only preceding completed candles and the
    /// just-observed execution. No current bar close is available at its open.
    #[cfg(test)]
    pub(super) fn hsl_inputs_at(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
        at_flat_boundary: bool,
    ) -> Result<Inputs, String> {
        self.capture_hsl_inputs(k, mode, pside, symbol, at_flat_boundary, None, false)
    }

    pub(super) fn hsl_inputs_at_clipped(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
        at_flat_boundary: bool,
        cutoff: Option<(i64, usize)>,
    ) -> Result<Inputs, String> {
        self.capture_hsl_inputs(k, mode, pside, symbol, at_flat_boundary, cutoff, true)
    }

    /// Compact only proven-flat price prefixes. Missing current openings are
    /// reconciled at now by the shared evaluator, never projected into the past.
    fn capture_hsl_inputs(
        &self,
        k: usize,
        mode: snapshot::Mode,
        pside: Option<PositionSide>,
        symbol: Option<&str>,
        at_flat_boundary: bool,
        cutoff: Option<(i64, usize)>,
        compact: bool,
    ) -> Result<Inputs, String> {
        if self.interval_ms != 60_000 || k >= self.hlcvs.shape()[0] {
            return Err("HSL backtest inputs require a valid 1m bar".into());
        }
        let days = self.backtest_params.pnls_max_lookback_days;
        if !days.is_finite() || !(1.0..=90.0).contains(&days) {
            return Err("HSL requires a finite lookback in [1,90] days".into());
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
        let selected = |side: PositionSide, coin: &str| {
            pside.is_none_or(|s| s == side) && symbol.is_none_or(|s| s == coin)
        };
        // The reconciler has zero historical inventory before a pair's first
        // retained entry. Keep one flat EMA seed, every retained execution and
        // every subsequent minute; reductions with missing inventory need the
        // full lookback. Check the complete scope to preserve aligned grids.
        let mut compact_prefix = compact
            && !at_flat_boundary
            && (self.backtest_params.metrics_only || !self.backtest_params.hsl_detailed_report);
        let mut first_pairs = BTreeSet::new();
        let mut first_entry = None;
        for fill in &self.fills[fill_start..fill_end] {
            let side = if fill.order_type.is_long() {
                PositionSide::Long
            } else {
                PositionSide::Short
            };
            if selected(side, &fill.coin)
                && first_pairs.insert((&fill.coin, side == PositionSide::Long))
            {
                compact_prefix &= fill.order_type.is_entry();
                first_entry = Some(first_entry.map_or(fill.timestamp_ms as i64, |t: i64| {
                    t.min(fill.timestamp_ms as i64)
                }));
            }
        }
        let price_start = if compact_prefix {
            first_entry
                .map_or(now, |t| t.saturating_sub(60_000))
                .max(start)
        } else {
            start
        };
        // Preserve the original treatment of wholly flat empty scopes.
        let compact_current = compact_prefix
            && first_entry.is_none()
            && self.positions.long.iter().enumerate().any(|(c, long)| {
                (long.size != 0.0 && selected(PositionSide::Long, &self.backtest_params.coins[c]))
                    || (self.positions.short[c].size != 0.0
                        && selected(PositionSide::Short, &self.backtest_params.coins[c]))
            });
        let price_start = if first_entry.is_none() && !compact_current {
            start
        } else {
            price_start
        };
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
                    return Err("HSL boundary observation requires a flat selected scope".into());
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
                // An observed flat coin with no retained executions has no
                // historical exposure to value. Feed the shared evaluator its
                // explicit flat proof instead of building a minute price tape.
                // Detailed exports keep that tape's historical diagnostics.
                let compact_flat = matches!(mode, snapshot::Mode::Coin)
                    && (self.backtest_params.metrics_only
                        || !self.backtest_params.hsl_detailed_report);
                if position.size == 0.0
                    && fills.is_empty()
                    && (compact_flat || !self.coin_is_valid_at(idx, k))
                {
                    // Preserve explicit flat proof for a requested coin without
                    // manufacturing a mark for a market that has no quote.
                    if matches!(mode, snapshot::Mode::Coin) {
                        flat_coin = Some(snapshot::FlatCoin {
                            symbol: coin.clone(),
                            pside: side,
                            position_at: now,
                            fills_at: Some(now),
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
                            fills_at: Some(now),
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
                // Full exports retain historical price diagnostics. Compact
                // execution-free scopes need only the factual endpoint mark.
                let projected: BTreeMap<_, _> = if compact_current {
                    BTreeMap::from([(now, mark)])
                } else {
                    let mut closes = Vec::new();
                    let compact_first_row =
                        (((price_start as i128 - self.first_timestamp_ms as i128 + 59_999).max(0)
                            / 60_000)
                            - 1)
                        .max(0) as usize;
                    for j in first_row.max(compact_first_row)..=k {
                        let end = close_time(j)?;
                        if end < price_start || end > now || !self.coin_is_valid_at(idx, j) {
                            continue;
                        }
                        closes.push((end, self.hlcvs_value(j, idx, CLOSE)));
                    }
                    let projected =
                        prices::minute_prices_from_ordered_closes(price_start, now, &closes)?;
                    reasons.extend(projected.reasons);
                    projected
                        .rows
                        .into_iter()
                        .map(|p| (p.timestamp, p.close))
                        .collect()
                };
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
    use crate::types::EquityHardStopLossConfig;
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
            ew_half_life_days: None,
            calculate_legacy_w: true,
            hsl_detailed_report: true,
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
            limit_order_fill_buffer_pct: 0.0,
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
    fn compact_flat_coin_omits_only_irrelevant_valuation_history() {
        let data = candles(1500, 2);
        let btc = Array1::from_elem(1500, 20_000.0);
        for side in [PositionSide::Long, PositionSide::Short] {
            let mut bt = make(&data, &btc);
            bt.backtest_params.hsl_detailed_report = false;
            // An execution on another coin/side is not this scope's history.
            fill(&mut bt, 0, 1, PositionSide::Long, 1.0, 200.0);
            let compact = bt
                .hsl_inputs(1499, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert!(compact.snapshot.pairs.is_empty());
            let flat = compact.snapshot.flat_coin.as_ref().unwrap();
            assert_eq!(flat.pside, side);
            assert_eq!(flat.position_at, compact.snapshot.now);
            assert_eq!(flat.fills_at, Some(compact.snapshot.now));
            assert_eq!(flat.history_start, compact.snapshot.start);
            bt.backtest_params.hsl_detailed_report = true;
            let detailed = bt
                .hsl_inputs(1499, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert!(detailed.snapshot.flat_coin.is_none());
            assert!(detailed.snapshot.pairs[0].prices.len() > 1000);
            bt.backtest_params.metrics_only = true;
            assert!(bt
                .hsl_inputs(1499, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap()
                .snapshot
                .pairs
                .is_empty());
        }
    }

    #[test]
    fn compact_flat_coin_preserves_retained_fills_and_expires_old_episodes() {
        let data = candles(1500, 1);
        let btc = Array1::from_elem(1500, 20_000.0);
        for side in [PositionSide::Long, PositionSide::Short] {
            let mut bt = make(&data, &btc);
            bt.backtest_params.hsl_detailed_report = false;
            let direction = if side == PositionSide::Long {
                1.0
            } else {
                -1.0
            };
            fill(&mut bt, 0, 0, side, direction, 100.0);
            fill(&mut bt, 1, 0, side, -direction, 100.0 - 20.0 * direction);
            let retained = bt
                .hsl_inputs(100, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert!(retained.snapshot.flat_coin.is_none());
            assert_eq!(retained.snapshot.pairs[0].fills.len(), 2);
            assert!(!retained.snapshot.pairs[0].prices.is_empty());
            let expired = bt
                .hsl_inputs(1499, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert!(expired.snapshot.flat_coin.is_some());
            let evaluate = |input: Inputs| {
                crate::hsl_evaluator::evaluate_for_simulator(crate::hsl_evaluator::Input {
                    snapshot: input.snapshot,
                    slots: 1,
                    scale_budget_with_excess_allowance: false,
                    exposure_budget: None,
                    span: 1.5,
                    threshold: 0.01,
                    cooldown_ms: 60_000,
                    restart: crate::hsl_controller::Restart::Never,
                })
                .unwrap()
            };
            let compact = evaluate(expired);
            bt.backtest_params.hsl_detailed_report = true;
            let detailed = evaluate(
                bt.hsl_inputs(1499, snapshot::Mode::Coin, Some(side), Some("C0"))
                    .unwrap(),
            );
            assert_eq!(
                serde_json::to_value(&compact.decision).unwrap(),
                serde_json::to_value(&detailed.decision).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&compact.events).unwrap(),
                serde_json::to_value(&detailed.events).unwrap()
            );
        }
    }

    #[test]
    fn compact_flat_proof_never_hides_current_exposure_or_invalid_balance() {
        let data = candles(10, 1);
        let btc = Array1::from_elem(10, 20_000.0);
        let mut bt = make(&data, &btc);
        bt.backtest_params.hsl_detailed_report = false;
        // Missing retained executions cannot turn observed exposure into flatness.
        bt.positions.short[0].size = -1.0;
        bt.positions.short[0].price = 110.0;
        let held = bt
            .hsl_inputs(
                8,
                snapshot::Mode::Coin,
                Some(PositionSide::Short),
                Some("C0"),
            )
            .unwrap();
        assert!(held.snapshot.flat_coin.is_none());
        assert_eq!(held.snapshot.pairs[0].position.size, -1.0);
        assert!(!held.snapshot.pairs[0].prices.is_empty());
        bt.coin_last_valid_idx[0] = 1;
        assert!(bt
            .hsl_inputs(
                8,
                snapshot::Mode::Coin,
                Some(PositionSide::Short),
                Some("C0")
            )
            .is_err());
        bt.positions.short[0].size = 0.0;
        bt.balance.usd_total_balance = f64::NAN;
        let flat = bt
            .hsl_inputs(
                8,
                snapshot::Mode::Coin,
                Some(PositionSide::Short),
                Some("C0"),
            )
            .unwrap();
        assert!(
            crate::hsl_evaluator::evaluate_for_simulator(crate::hsl_evaluator::Input {
                snapshot: flat.snapshot,
                slots: 0,
                scale_budget_with_excess_allowance: false,
                exposure_budget: None,
                span: 1.5,
                threshold: 0.01,
                cooldown_ms: 0,
                restart: crate::hsl_controller::Restart::Always,
            })
            .is_err()
        );
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
                .hsl_inputs(3, snapshot::Mode::Coin, Some(side), Some("C0"))
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
            let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
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
            .hsl_inputs(2, snapshot::Mode::Unified, None, None)
            .unwrap();
        let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
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
            .hsl_inputs(2, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(unified.snapshot.balance, 995.123);
        assert_eq!(unified.snapshot.pairs.len(), 4);
        assert!(unified
            .snapshot
            .pairs
            .iter()
            .any(|p| p.symbol == "C1" && p.position.size == -2.0));
        let long = bt
            .hsl_inputs(2, snapshot::Mode::Pside, Some(PositionSide::Long), None)
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
            .hsl_inputs(1, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(input.slots, [1, 0]);
        bt.backtest_params.dynamic_wel_by_tradability = false;
        let input = bt
            .hsl_inputs(1, snapshot::Mode::Unified, None, None)
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
            .hsl_inputs(6, snapshot::Mode::Unified, None, None)
            .unwrap();
        assert_eq!(input.snapshot.pairs.len(), 1);
        assert_eq!(input.snapshot.pairs[0].mark_at, (FIRST + 3 * 60_000) as i64);
        let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
        assert!(trace.reasons.contains("stale_flat_mark"));
        assert_eq!(trace.episodes.len(), 2);
        // No held exposure may use that stale valuation.
        bt.positions.long[0] = Position {
            size: 1.0,
            price: 100.0,
        };
        assert!(bt
            .hsl_inputs(6, snapshot::Mode::Unified, None, None)
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
            .hsl_inputs(
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
            .hsl_inputs(
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
            .hsl_inputs(2, snapshot::Mode::Unified, None, None)
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
                .hsl_inputs(2, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
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
            let mut input = bt.hsl_inputs(2, mode, side, None).unwrap();
            let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
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
                .hsl_inputs(
                    2,
                    snapshot::Mode::Coin,
                    Some(PositionSide::Long),
                    Some("C0"),
                )
                .unwrap();
            assert!(input.snapshot.pairs.is_empty());
            assert!(input.snapshot.flat_coin.is_some());
            let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
            let point = &trace.episodes[0].points[0];
            assert!(!point.exposed);
            assert_eq!(point.pnl, 0.0);
            assert_eq!(point.upnl, 0.0);
            assert_eq!(point.timestamp, input.snapshot.now);
            assert!(bt
                .hsl_inputs(
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
                .hsl_inputs(1441, snapshot::Mode::Coin, Some(side), Some("C0"))
                .unwrap();
            assert_eq!(input.snapshot.start, (FIRST + 2 * 60_000) as i64);
            assert_eq!(input.snapshot.pairs[0].fills.len(), 1);
            assert_eq!(input.snapshot.pairs[0].prices[&input.snapshot.start], 100.0);
            let trace = crate::hsl_trace::compose(&input.snapshot).unwrap();
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
    fn enable_hsl(bt: &mut Backtest<'_>, mode: &str, order_type: &str) {
        use super::super::hsl_runtime::{Config, Policy};
        let policy = Policy {
            enabled: true,
            scale_budget_with_excess_allowance: false,
            red_threshold: 0.05,
            ema_span_minutes: 1.0,
            cooldown_minutes_after_red: 10.0,
            restart_after_red_policy: Some("always".into()),
            panic_close_order_type: order_type.into(),
        };
        bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
            mode: mode.into(),
            sides: [policy.clone(), policy.clone()],
            portfolio: Some(policy),
            coins: BTreeMap::new(),
        });
    }

    #[test]
    fn cached_orchestrator_input_reuses_configured_hsl_policies() {
        use super::super::hsl_runtime::{Config, Policy};
        for mode in ["coin", "pside", "unified"] {
            for enabled in [false, true] {
                let data = candles(5, 2);
                let btc = Array1::from_elem(5, 1.0);
                let mut bt = make(&data, &btc);
                let policy = Policy {
                    enabled,
                    panic_close_order_type: "limit".into(),
                    restart_after_red_policy: Some("always".into()),
                    ..Policy::default()
                };
                let mut config = Config {
                    mode: mode.into(),
                    sides: [policy.clone(), policy.clone()],
                    portfolio: (mode == "unified").then(|| Policy {
                        panic_close_order_type: "market".into(),
                        ..policy.clone()
                    }),
                    coins: BTreeMap::new(),
                };
                if mode == "coin" {
                    let override_policy = Policy {
                        enabled: !enabled,
                        panic_close_order_type: "market".into(),
                        ..policy.clone()
                    };
                    config.coins.insert(
                        "C0".into(),
                        [override_policy.clone(), override_policy],
                    );
                }
                config.validate(&bt.backtest_params.coins, 1.0, 1).unwrap();
                bt.backtest_params.equity_hard_stop_loss.hsl = Some(config);
                for pair in &mut bt.bot_params {
                    pair.long.hsl_enabled = !enabled;
                    pair.short.hsl_enabled = !enabled;
                    pair.long.hsl_panic_close_order_type = "stale".into();
                    pair.short.hsl_panic_close_order_type = "stale".into();
                }
                let expected = |coin: usize| {
                    if mode == "coin" && coin == 0 {
                        (!enabled, "market")
                    } else if mode == "unified" {
                        (enabled, "market")
                    } else {
                        (enabled, "limit")
                    }
                };
                let initial = bt.get_orchestrator_input_cached(1, None, None);
                let pointers: Vec<_> = initial
                    .symbols
                    .iter()
                    .map(|symbol| {
                        (
                            symbol.long.bot_params.hsl_panic_close_order_type.as_ptr(),
                            symbol.short.bot_params.hsl_panic_close_order_type.as_ptr(),
                        )
                    })
                    .collect();
                for symbol in &initial.symbols {
                    let (flag, order_type) = expected(symbol.symbol_idx);
                    for params in [&symbol.long.bot_params, &symbol.short.bot_params] {
                        assert_eq!(params.hsl_enabled, flag);
                        assert_eq!(params.hsl_panic_close_order_type, order_type);
                    }
                }
                bt.orchestrator_input_cache = Some(initial);
                let refreshed = bt.get_orchestrator_input_cached(2, None, None);
                for symbol in &refreshed.symbols {
                    let (flag, order_type) = expected(symbol.symbol_idx);
                    for params in [&symbol.long.bot_params, &symbol.short.bot_params] {
                        assert_eq!(params.hsl_enabled, flag);
                        assert_eq!(params.hsl_panic_close_order_type, order_type);
                    }
                    // Immutable cached strings must retain their owned buffers during refresh.
                    assert_eq!(
                        symbol.long.bot_params.hsl_panic_close_order_type.as_ptr(),
                        pointers[symbol.symbol_idx].0,
                    );
                    assert_eq!(
                        symbol.short.bot_params.hsl_panic_close_order_type.as_ptr(),
                        pointers[symbol.symbol_idx].1,
                    );
                }
            }
        }
    }

    #[test]
    fn hsl_runtime_drives_real_panic_orders_for_all_scopes() {
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
                enable_hsl(&mut bt, mode, order_type);
                fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
                bt.update_hsl(1).unwrap();
                assert_eq!(bt.hsl_action(LONG, 0), crate::hsl_controller::Action::Panic);
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
    fn hsl_flat_boundary_cancels_reentry_without_future_candle_close() {
        for future_close in [10.0, 200.0] {
            let mut c = candles(4, 1);
            c[[1, 0, CLOSE]] = future_close;
            let btc = Array1::from_elem(4, 1.0);
            let mut bt = make(&c, &btc);
            enable_hsl(&mut bt, "coin", "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
            fill(&mut bt, 1, 0, PositionSide::Long, -10.0, 80.0);
            assert_eq!(
                bt.hsl_action(LONG, 0),
                crate::hsl_controller::Action::Halted
            );
            let decision = bt
                .hsl_scopes
                .iter()
                .find_map(|s| s.result.decision.as_ref())
                .unwrap();
            assert_eq!(decision.flat_at, Some((FIRST + 60_000) as i64));
            assert_eq!(decision.timestamp, (FIRST + 60_000) as i64);
            assert!(bt.open_orders.long[0].entries.is_empty());
        }
    }

    #[test]
    fn hsl_unified_uses_one_controller_and_explicit_portfolio_execution_policy() {
        let mut c = candles(4, 1);
        c[[2, 0, LOW]] = 99.0;
        let btc = Array1::from_elem(4, 1.0);
        let mut bt = make(&c, &btc);
        enable_hsl(&mut bt, "unified", "limit");
        let cfg = bt
            .backtest_params
            .equity_hard_stop_loss
            .hsl
            .as_mut()
            .unwrap();
        cfg.sides[LONG].panic_close_order_type = "market".into();
        cfg.sides[SHORT].panic_close_order_type = "market".into();
        fill(&mut bt, 0, 0, PositionSide::Short, -10.0, 80.0);
        bt.trading_enabled.short = false;
        bt.update_hsl(1).unwrap();
        assert_eq!(bt.hsl_scopes.len(), 1);
        assert_eq!(
            bt.hsl_action(SHORT, 0),
            crate::hsl_controller::Action::Panic
        );
        bt.update_open_orders_all(1).unwrap();
        let close = &bt.open_orders.short[0].closes[0];
        assert_eq!(close.execution_type, orchestrator::ExecutionType::Limit);
        bt.check_for_fills(2).unwrap();
        assert_eq!(bt.positions.short[0].size, 0.0);
    }
    #[test]
    fn hsl_full_run_executes_red_before_next_bar_and_keeps_cooldown() {
        for mode in ["coin", "pside", "unified"] {
            let mut c = candles(8, 1);
            for k in 2..8 {
                for f in [HIGH, LOW, CLOSE] {
                    c[[k, 0, f]] = 80.0;
                }
            }
            let btc = Array1::from_elem(8, 1.0);
            let mut bt = make(&c, &btc);
            enable_hsl(&mut bt, mode, "market");
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
            let report = bt.hsl_report_value().unwrap().unwrap();
            // Native Python transport must preserve the complete serde schema,
            // including optional values and normal/panic/halted transitions.
            pyo3::prepare_freethreaded_python();
            pyo3::Python::with_gil(|py| {
                use pyo3::prelude::*;
                let actual = crate::python::hsl_report_to_py(py, &bt).unwrap().unwrap();
                let encoded: String = py
                    .import_bound("json")
                    .unwrap()
                    .call_method1("dumps", (actual,))
                    .unwrap()
                    .extract()
                    .unwrap();
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
                bt.hsl_action(LONG, 0),
                crate::hsl_controller::Action::Halted
            );
            assert!(bt.open_orders.long[0].entries.is_empty());
        }
    }

    #[test]
    fn hsl_cooldown_rebuilds_from_fills_and_expires_without_prior_runtime() {
        use crate::hsl_controller::Action;
        let c = candles(1445, 1);
        let btc = Array1::from_elem(1445, 1.0);
        for mode in ["coin", "pside", "unified"] {
            let mut bt = make(&c, &btc);
            enable_hsl(&mut bt, mode, "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
            fill(&mut bt, 1, 0, PositionSide::Long, -10.0, 80.0);
            for (restart, k, expected) in [
                ("always", 9, Action::Halted),
                ("always", 10, Action::Normal),
                ("never", 10, Action::Halted),
                ("never", 1442, Action::Normal),
            ] {
                let mut fresh = make(&c, &btc);
                enable_hsl(&mut fresh, mode, "market");
                let cfg = fresh
                    .backtest_params
                    .equity_hard_stop_loss
                    .hsl
                    .as_mut()
                    .unwrap();
                for p in &mut cfg.sides {
                    p.restart_after_red_policy = Some(restart.into());
                }
                cfg.portfolio.as_mut().unwrap().restart_after_red_policy = Some(restart.into());
                fresh.fills = bt.fills.clone();
                fresh.balance.usd_total_balance = bt.balance.usd_total_balance;
                fresh.balance.usd_cash_wallet = bt.balance.usd_cash_wallet;
                assert!(fresh.hsl_scopes.is_empty());
                fresh.update_hsl(k).unwrap();
                assert_eq!(fresh.hsl_action(LONG, 0), expected, "{mode} {restart} {k}");
            }
        }
    }

    #[test]
    fn hsl_partial_close_retains_protection_until_scope_flat() {
        use crate::hsl_controller::Action;
        let c = candles(5, 2);
        let btc = Array1::from_elem(5, 1.0);
        for mode in ["pside", "unified"] {
            let mut bt = make(&c, &btc);
            enable_hsl(&mut bt, mode, "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 120.0);
            fill(&mut bt, 0, 1, PositionSide::Long, 1.0, 200.0);
            bt.update_hsl(1).unwrap();
            assert_eq!(bt.hsl_action(LONG, 0), Action::Panic);
            fill(&mut bt, 2, 0, PositionSide::Long, -4.0, 100.0);
            bt.update_hsl(2).unwrap();
            let orders = bt.hsl_protective_orders(2).unwrap();
            assert_eq!(orders.len(), 2);
            assert_eq!(orders.iter().find(|o| o.symbol_idx == 0).unwrap().qty, -6.0);
            fill(&mut bt, 3, 0, PositionSide::Long, -6.0, 100.0);
            assert_eq!(bt.hsl_action(LONG, 1), Action::Panic);
            fill(&mut bt, 3, 1, PositionSide::Long, -1.0, 200.0);
            assert_eq!(bt.hsl_action(LONG, 0), Action::Halted);
            assert_eq!(bt.hsl_action(LONG, 1), Action::Halted);
        }
    }

    #[test]
    fn hsl_simulator_fill_tie_distinguishes_exact_and_current_flat_timing() {
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
                .hsl_inputs_at(
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
    fn hsl_first_bar_flat_uses_execution_without_unrelated_future_quotes() {
        let c = candles(3, 2);
        let btc = Array1::from_elem(3, 1.0);
        let mut bt = make(&c, &btc);
        enable_hsl(&mut bt, "unified", "market");
        fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 100.0);
        fill(&mut bt, 0, 0, PositionSide::Long, -10.0, 80.0);
        assert_eq!(
            bt.hsl_action(LONG, 0),
            crate::hsl_controller::Action::Halted
        );
    }
    #[test]
    fn hsl_residual_close_gate_does_not_reenable_disabled_side_entries() {
        let c = candles(3, 1);
        let btc = Array1::from_elem(3, 1.0);
        for (side, direction) in [(LONG, 1.0), (SHORT, -1.0)] {
            let mut bt = make(&c, &btc);
            enable_hsl(&mut bt, "unified", "market");
            let pside = if side == LONG {
                PositionSide::Long
            } else {
                PositionSide::Short
            };
            fill(&mut bt, 0, 0, pside, direction, 100.0);
            bt.update_hsl(0).unwrap();
            assert_eq!(
                bt.hsl_action(side, 0),
                crate::hsl_controller::Action::Normal
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
    fn hsl_panic_fill_uses_explicit_type_fees_slippage_and_close_eligibility() {
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
                        enable_hsl(&mut bt, mode, execution);
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
                        bt.update_hsl(1).unwrap();
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
    fn hsl_reporting_reset_cannot_change_panic_execution() {
        let c = candles(5, 1);
        let btc = Array1::from_elem(5, 1.0);
        let mut outcomes = Vec::new();
        for reset_report in [false, true] {
            let mut bt = make(&c, &btc);
            enable_hsl(&mut bt, "unified", "market");
            fill(&mut bt, 0, 0, PositionSide::Long, 10.0, 120.0);
            bt.update_hsl(1).unwrap();
            if reset_report {
                bt.hsl_report = super::super::hsl_report::Report::default();
            }
            bt.update_open_orders_all(1).unwrap();
            bt.check_for_fills(2).unwrap();
            outcomes.push((
                bt.positions.long[0].size,
                bt.hsl_action(LONG, 0),
                bt.fills
                    .iter()
                    .map(|f| (f.timestamp_ms, f.fill_qty, f.fill_price, f.order_type))
                    .collect::<Vec<_>>(),
            ));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
    #[test]
    fn hsl_panic_loss_ratio_revalues_collateral_before_fill() {
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
            enable_hsl(&mut bt, "coin", "market");
            fill(&mut bt, 0, 0, side, qty, 100.0);
            bt.balance.use_btc_collateral = true;
            bt.balance.btc_cash_wallet = 500.0;
            bt.balance.usd_cash_wallet = bt.balance.usd_total_balance - 500.0;
            bt.update_hsl(2).unwrap();
            let pside = if long { LONG } else { SHORT };
            assert_eq!(
                bt.hsl_action(pside, 0),
                crate::hsl_controller::Action::Panic
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
            let metrics = bt.hsl_report.metrics(1000.0, 3.0);
            assert_eq!(metrics.panic_close_loss_sum, 400.0);
            assert!(
                (metrics.panic_close_loss_drawdown_pct_mean - 400.0 / expected_equity).abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn hsl_terminal_panic_fill_finishes_flat_latency_at_execution() {
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
                enable_hsl(&mut bt, mode, "market");
                let config = bt
                    .backtest_params
                    .equity_hard_stop_loss
                    .hsl
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
                for emas in &bt.hsl_report.signal_emas {
                    assert_eq!(
                        emas.len() + 1,
                        bt.strategy_equity_series.len(),
                        "terminal strategy sample must not duplicate a stale EMA"
                    );
                }
                let metrics = bt.hsl_report.metrics(1000.0, 3.0);
                // RED at bar 2 close and the real flatten at bar 3 open coincide.
                assert_eq!(metrics.flatten_time_minutes_mean, 0.0, "{mode}");
                assert_eq!(metrics.duration_minutes_mean, 0.0, "{mode}");
                assert_eq!(metrics.time_in_red_pct, 0.0, "{mode}");
                assert_eq!(bt.hsl_report.summary.observed_minutes, 0.0, "{mode}");
                assert_eq!(metrics.restarts, 0);
                let report = bt.hsl_report_value().unwrap().unwrap();
                assert!(report["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e["kind"] == "flat" && e["observed_at"] == panic.timestamp_ms));
            }
        }
    }

    #[test]
    fn unrelated_executions_preserve_scope_cutoffs_and_empty_history() {
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        let data = candles(100, 2);
        let btc = Array1::from_elem(100, 50000.);
        let mut bt = make(&data, &btc);
        bt.backtest_params.metrics_only = true;
        for k in 0..60 {
            fill(
                &mut bt,
                k,
                0,
                PositionSide::Long,
                if k % 2 == 0 { 1.0 } else { -1.0 },
                100.0,
            );
            for (side, coin, _mode, symbol) in [
                (Some(SHORT), Some(0), snapshot::Mode::Coin, Some("C0")),
                (Some(SHORT), None, snapshot::Mode::Pside, None),
                (Some(LONG), Some(1), snapshot::Mode::Coin, Some("C1")),
            ] {
                let cutoff = bt.hsl_history_cutoff(side, coin);
                assert_eq!(cutoff, Some((i64::MIN, bt.fills.len())));
                let pside = side.map(|s| {
                    if s == LONG {
                        PositionSide::Long
                    } else {
                        PositionSide::Short
                    }
                });
                let mode = || {
                    if coin.is_some() {
                        snapshot::Mode::Coin
                    } else {
                        snapshot::Mode::Pside
                    }
                };
                let full = bt.hsl_inputs_at(k, mode(), pside, symbol, false).unwrap();
                let clipped = bt
                    .hsl_inputs_at_clipped(k, mode(), pside, symbol, false, cutoff)
                    .unwrap();
                assert_eq!(full.snapshot.start, clipped.snapshot.start);
                let input = |snapshot| Input {
                    snapshot,
                    slots: 1,
                    scale_budget_with_excess_allowance: false,
                    exposure_budget: None,
                    span: 3.5,
                    threshold: 0.06,
                    cooldown_ms: 600_000,
                    restart: Restart::Always,
                };
                assert_eq!(
                    serde_json::to_value(evaluate(input(full.snapshot)).unwrap()).unwrap(),
                    serde_json::to_value(evaluate(input(clipped.snapshot)).unwrap()).unwrap()
                );
                if k % 10 == 0 {
                    bt.hsl_cutoffs.clear();
                    assert_eq!(bt.hsl_history_cutoff(side, coin), cutoff);
                }
            }
        }
        // The first relevant fill invalidates an empty-history prefix.
        fill(&mut bt, 60, 0, PositionSide::Short, -1.0, 100.0);
        let cutoff = bt.hsl_history_cutoff(Some(SHORT), Some(0));
        assert_ne!(cutoff.unwrap().0, i64::MIN);
        // Unrelated fills retain the selected episode, including its fill index.
        fill(&mut bt, 61, 1, PositionSide::Long, 1.0, 200.0);
        assert_eq!(bt.hsl_history_cutoff(Some(SHORT), Some(0)), cutoff);
        bt.hsl_cutoffs.clear();
        assert_eq!(bt.hsl_history_cutoff(Some(SHORT), Some(0)), cutoff);
    }

    #[test]
    fn latest_episode_suffix_preserves_current_decision() {
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
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
                let cutoff = bt.hsl_history_cutoff(side, coin);
                let side_name = side.map(|_| PositionSide::Long);
                let symbol = coin.map(|c| bt.backtest_params.coins[c].as_str());
                let full = bt
                    .hsl_inputs_at(k, mode(), side_name, symbol, false)
                    .unwrap();
                let trimmed = bt
                    .hsl_inputs_at_clipped(k, mode(), side_name, symbol, false, cutoff)
                    .unwrap();
                assert!(trimmed.snapshot.start >= full.snapshot.start);
                let config = |snapshot| Input {
                    snapshot,
                    slots: 1,
                    scale_budget_with_excess_allowance: false,
                    exposure_budget: None,
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
                bt.hsl_cutoffs.clear();
                assert_eq!(bt.hsl_history_cutoff(side, coin), cutoff);
            }
        }
    }
    #[test]
    fn incremental_marks_match_fresh_replay_through_fills_recovery_and_budget_changes() {
        use crate::backtest::hsl_runtime::{Config, Policy};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
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
        for name in ["coin", "coin_scaled", "pside", "unified"] {
            for restart in ["always", "never"] {
                let mode = || match name {
                    "coin" | "coin_scaled" => snapshot::Mode::Coin,
                    "pside" => snapshot::Mode::Pside,
                    _ => snapshot::Mode::Unified,
                };
                let mut bt = make(&data, &btc);
                for params in &mut bt.bot_params_original {
                    params.long.risk_we_excess_allowance_pct = 0.44;
                    params.short.risk_we_excess_allowance_pct = 0.25;
                }
                let policy = Policy {
                    enabled: true,
                    scale_budget_with_excess_allowance: name == "coin_scaled",
                    red_threshold: 0.06,
                    ema_span_minutes: 3.5,
                    cooldown_minutes_after_red: 5.,
                    restart_after_red_policy: Some(restart.into()),
                    panic_close_order_type: "limit".into(),
                };
                bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                    mode: if name == "coin_scaled" { "coin" } else { name }.into(),
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
                        bt.hsl_traces.clear();
                    }
                    if k % 11 == 0 {
                        bt.hsl_cutoffs.clear();
                        bt.hsl_scopes.clear();
                    }
                    bt.update_hsl(k).unwrap();
                    for scope in &bt.hsl_scopes {
                        let actual = scope.result.decision.as_ref().unwrap();
                        let side = scope.side.map(|s| {
                            if s == LONG {
                                PositionSide::Long
                            } else {
                                PositionSide::Short
                            }
                        });
                        let symbol = scope.coin.map(|c| bt.backtest_params.coins[c].as_str());
                        let full = bt.hsl_inputs_at(k, mode(), side, symbol, false).unwrap();
                        let expected = evaluate(Input {
                            scale_budget_with_excess_allowance: name == "coin_scaled",
                            exposure_budget: bt.hsl_exposure_budget(scope.side, scope.coin),
                            snapshot: full.snapshot,
                            slots: if name.starts_with("coin") {
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
        use crate::backtest::hsl_runtime::{Config, Policy};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
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
            scale_budget_with_excess_allowance: false,
            red_threshold: 0.06,
            ema_span_minutes: 308.5,
            cooldown_minutes_after_red: 5.,
            restart_after_red_policy: Some("always".into()),
            panic_close_order_type: "limit".into(),
        };
        bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
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
            bt.update_hsl(k).unwrap();
            let actual = bt.hsl_scopes[0].result.decision.as_ref().unwrap();
            let full = bt
                .hsl_inputs_at(
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
                scale_budget_with_excess_allowance: false,
                exposure_budget: None,
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
    /// Omitted idle price rows cannot change the shared controller, including
    /// current RED; retained executions and detailed exports keep the full tape.
    #[test]
    fn flat_prefix_capture_is_exact_and_scope_bounded() {
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, evaluate_for_simulator, Input};
        let mut data = candles(1460, 2);
        for k in 0..1460 {
            for c in 0..2 {
                for f in [HIGH, LOW, CLOSE] {
                    data[[k, c, f]] = if k < 1400 { 100. } else { 80. };
                }
            }
        }
        let btc = Array1::from_elem(1460, 50000.);
        let mut bt = make(&data, &btc);
        bt.backtest_params.hsl_detailed_report = false;
        fill(&mut bt, 3, 0, PositionSide::Long, 1., 100.);
        for later_entry in [false, true] {
            if later_entry {
                fill(&mut bt, 1400, 0, PositionSide::Long, 0.5, 80.);
            }
            for name in ["coin", "pside", "unified"] {
                let mode = || match name {
                    "coin" => snapshot::Mode::Coin,
                    "pside" => snapshot::Mode::Pside,
                    _ => snapshot::Mode::Unified,
                };
                let side = if name == "unified" {
                    None
                } else {
                    Some(PositionSide::Long)
                };
                let symbol = if name == "coin" { Some("C0") } else { None };
                let full = bt
                    .capture_hsl_inputs(1450, mode(), side, symbol, false, None, false)
                    .unwrap();
                let compact = bt
                    .capture_hsl_inputs(1450, mode(), side, symbol, false, None, true)
                    .unwrap();
                assert_eq!(compact.snapshot.start, full.snapshot.start);
                assert!(full.snapshot.pairs.iter().any(|p| p.prices.len() > 1000));
                assert!(compact.snapshot.pairs.iter().all(|p| if later_entry {
                    (2..100).contains(&p.prices.len())
                } else {
                    p.prices.len() == 1
                }));
                for threshold in [0.00001, 0.99] {
                    let eval = |snapshot, compact| {
                        let evaluator = if compact {
                            evaluate_for_simulator
                        } else {
                            evaluate
                        };
                        evaluator(Input {
                            snapshot,
                            slots: 1,
                            scale_budget_with_excess_allowance: false,
                            exposure_budget: None,
                            span: 627.5,
                            threshold,
                            cooldown_ms: 300000,
                            restart: Restart::Always,
                        })
                        .unwrap()
                    };
                    let full = bt
                        .capture_hsl_inputs(1450, mode(), side, symbol, false, None, false)
                        .unwrap();
                    let compact = bt
                        .capture_hsl_inputs(1450, mode(), side, symbol, false, None, true)
                        .unwrap();
                    assert_eq!(
                        serde_json::to_value(eval(full.snapshot, false)).unwrap(),
                        serde_json::to_value(eval(compact.snapshot, true)).unwrap()
                    );
                }
            }
        }
        bt.backtest_params.hsl_detailed_report = true;
        let detailed = bt
            .hsl_inputs_at_clipped(
                1450,
                snapshot::Mode::Coin,
                Some(PositionSide::Long),
                Some("C0"),
                false,
                None,
            )
            .unwrap();
        assert!(detailed.snapshot.pairs[0].prices.len() > 1000);
        let mut bt = make(&data, &btc);
        bt.backtest_params.hsl_detailed_report = false;
        fill(&mut bt, 3, 0, PositionSide::Long, 1., 100.);
        fill(&mut bt, 3, 1, PositionSide::Long, 1., 100.);
        fill(&mut bt, 1449, 1, PositionSide::Long, -0.5, 80.);
        let coin = bt
            .hsl_inputs_at_clipped(
                1450,
                snapshot::Mode::Coin,
                Some(PositionSide::Long),
                Some("C0"),
                false,
                None,
            )
            .unwrap();
        assert_eq!(coin.snapshot.pairs[0].prices.len(), 1);
        let aggregate = bt
            .hsl_inputs_at_clipped(1450, snapshot::Mode::Unified, None, None, false, None)
            .unwrap();
        assert!(aggregate
            .snapshot
            .pairs
            .iter()
            .all(|p| p.prices.len() > 1000));
    }
    /// Compact captures remain rederivable through clipping, fees,
    /// budget/slot changes, GREEN recovery, terminal closes and cache loss.
    #[test]
    fn compact_current_matches_fresh_controller_across_runtime_changes() {
        use crate::backtest::hsl_runtime::{Config, Policy};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        let n = 1700;
        let mut data = candles(n, 2);
        for k in 0..n {
            let p = if (1530..1536).contains(&k) {
                45.
            } else {
                100. + 12. * (k as f64 * 0.013).sin()
            };
            for c in 0..2 {
                for f in [HIGH, LOW, CLOSE] {
                    data[[k, c, f]] = p + c as f64;
                }
            }
        }
        let btc = Array1::from_elem(n, 50000.);
        for mode in ["coin", "coin_scaled", "pside", "unified"] {
            for restart in [Restart::Always, Restart::Never] {
                let mut bt = make(&data, &btc);
                bt.backtest_params.hsl_detailed_report = false;
                let policy = Policy {
                    enabled: true,
                    scale_budget_with_excess_allowance: mode == "coin_scaled",
                    red_threshold: 0.06,
                    ema_span_minutes: 308.5,
                    cooldown_minutes_after_red: 5.,
                    restart_after_red_policy: Some(
                        if restart == Restart::Always {
                            "always"
                        } else {
                            "never"
                        }
                        .into(),
                    ),
                    panic_close_order_type: "limit".into(),
                };
                bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                    mode: if mode == "coin_scaled" { "coin" } else { mode }.into(),
                    sides: [policy.clone(), policy.clone()],
                    portfolio: (mode == "unified").then_some(policy.clone()),
                    coins: Default::default(),
                });
                for k in 2..n {
                    bt.current_step = k;
                    for (when, c, side, qty) in [
                        (3, 0, PositionSide::Long, 2.),
                        (5, 1, PositionSide::Short, -1.),
                        (1400, 0, PositionSide::Long, 0.5),
                        (1510, 0, PositionSide::Long, -1.0),
                        (1550, 0, PositionSide::Long, -1.5),
                        (1552, 0, PositionSide::Long, 1.),
                        (1570, 1, PositionSide::Short, 1.),
                    ] {
                        if k == when {
                            fill(&mut bt, k, c, side, qty, data[[k, c, CLOSE]]);
                        }
                    }
                    if k == 1490 || k == 1520 {
                        let delta = if k == 1490 { 100. } else { -50. };
                        bt.balance.usd_total_balance += delta;
                        bt.balance.usd_cash_wallet += delta;
                    }
                    if k == 1500 {
                        bt.effective_n_positions.long = 1;
                    }
                    if k == 1540 || k == 1600 {
                        bt.hsl_scopes.clear();
                        bt.hsl_traces.clear();
                    }
                    bt.update_hsl(k).unwrap();
                    if k < 1435 || (k > 1610 && k % 137 != 0) {
                        continue;
                    }
                    for scope in &bt.hsl_scopes {
                        let side = scope.side.map(|s| {
                            if s == LONG {
                                PositionSide::Long
                            } else {
                                PositionSide::Short
                            }
                        });
                        let symbol = scope.coin.map(|c| bt.backtest_params.coins[c].as_str());
                        let selected = match mode {
                            "coin" | "coin_scaled" => snapshot::Mode::Coin,
                            "pside" => snapshot::Mode::Pside,
                            _ => snapshot::Mode::Unified,
                        };
                        let full = bt.hsl_inputs_at(k, selected, side, symbol, false).unwrap();
                        let expected = evaluate(Input {
                            snapshot: full.snapshot,
                            slots: scope
                                .side
                                .map_or(1, |s| bt.hard_stop_coin_slot_n_positions(s))
                                as u64,
                            scale_budget_with_excess_allowance: mode == "coin_scaled",
                            exposure_budget: bt.hsl_exposure_budget(scope.side, scope.coin),
                            span: policy.ema_span_minutes,
                            threshold: policy.red_threshold,
                            cooldown_ms: 300000,
                            restart,
                        })
                        .unwrap()
                        .decision
                        .unwrap();
                        let actual = scope.result.decision.as_ref().unwrap();
                        assert_eq!(actual.action, expected.action, "mode={mode} k={k}");
                        assert_eq!(actual.red_at, expected.red_at, "red mode={mode} k={k}");
                        assert_eq!(actual.flat_at, expected.flat_at, "flat mode={mode} k={k}");
                        assert!(
                            (actual.raw - expected.raw).abs() < 1e-12,
                            "raw mode={mode} k={k}: {} vs {}",
                            actual.raw,
                            expected.raw
                        );
                        assert!(
                            (actual.ema - expected.ema).abs() < 1e-12,
                            "ema mode={mode} k={k}: {} vs {}",
                            actual.ema,
                            expected.ema
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn hsl_metrics_survive_draining_equities_and_disabled_policy() {
        for enabled in [false, true] {
            let c = candles(8, 1);
            let btc = Array1::from_elem(8, 1.0);
            let mut bot = BotParamsPair::default();
            for side in [&mut bot.long, &mut bot.short] {
                side.n_positions = 1;
                side.total_wallet_exposure_limit = 1.0;
                side.ema_span_0 = 1.0;
                side.ema_span_1 = 2.0;
            }
            let mut bt = Backtest::new(
                c.view(),
                btc.view(),
                vec![bot],
                vec![ExchangeParams::default()],
                &params(8, 1),
            );
            if enabled {
                enable_hsl(&mut bt, "coin", "limit");
            }
            bt.bot_params_master.long.total_wallet_exposure_limit = 4.0;
            bt.bot_params_original[0].long.wallet_exposure_limit = 4.0;
            bt.bot_params_original[0].long.total_wallet_exposure_limit = 4.0;
            fill(&mut bt, 0, 0, PositionSide::Long, 1.0, 100.0);
            bt.equity_tracking_active = true;
            let (_, equities) = bt.run().unwrap();
            assert!(!equities.timestamps_ms.is_empty());
            assert!(bt.equities.timestamps_ms.is_empty());
            let cached = bt.final_hard_stop_metrics.unwrap();
            let actual = bt.hard_stop_metrics();
            assert_eq!(actual.triggers, cached.triggers);
            assert_eq!(actual.panic_close_loss_sum, cached.panic_close_loss_sum);
            assert_eq!(
                bt.strategy_equity_metrics_for_analysis()
                    .overall
                    .gain_strategy_eq,
                bt.final_strategy_equity_metrics
                    .unwrap()
                    .overall
                    .gain_strategy_eq
            );
        }
    }
    #[test]
    fn reduction_only_rolling_cache_matches_fresh_reconstruction() {
        use crate::backtest::hsl_runtime::{Config, Policy, SignalSettings};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        for reopen in [false, true] {
            for side in [LONG, SHORT] {
                for mode in ["coin", "pside"] {
                    for restart in [Restart::Always, Restart::Never] {
                        let mut data = candles(3000, 1);
                        for k in 0..3000 {
                            for field in [HIGH, LOW, CLOSE] {
                                data[[k, 0, field]] = 85.0 + ((k * 17) % 53) as f64;
                            }
                        }
                        let btc = Array1::from_elem(3000, 50000.);
                        let mut bt = make(&data, &btc);
                        bt.exchange_params_list[0].qty_step = 0.01;
                        let policy = Policy {
                            enabled: true,
                            scale_budget_with_excess_allowance: false,
                            red_threshold: 0.005,
                            ema_span_minutes: 17.5,
                            cooldown_minutes_after_red: 5.,
                            restart_after_red_policy: Some(if restart == Restart::Always {
                                "always".into()
                            } else {
                                "never".into()
                            }),
                            panic_close_order_type: "limit".into(),
                        };
                        bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                            mode: mode.into(),
                            sides: [policy.clone(), policy.clone()],
                            portfolio: None,
                            coins: Default::default(),
                        });
                        let pside = if side == LONG {
                            PositionSide::Long
                        } else {
                            PositionSide::Short
                        };
                        let sign = if side == LONG { 1.0 } else { -1.0 };
                        fill(&mut bt, 1, 0, pside, sign * 17.13, 93.2);
                        let coin = (mode == "coin").then_some(0);
                        let settings = SignalSettings {
                            scale_budget_with_excess_allowance: false,
                            span: policy.ema_span_minutes,
                            threshold: policy.red_threshold,
                            cooldown_ms: 300_000,
                            restart,
                        };
                        let mut hits = 0;
                        for k in (1448..1480).chain(2888..2930) {
                            bt.current_step = k;
                            if k == 1450 || k == 1460 {
                                fill(&mut bt, k, 0, pside, -sign * 6.27, 107.4);
                            }
                            if k == 1478 && reopen {
                                fill(&mut bt, k, 0, pside, sign * 0.83, 91.2);
                            }
                            if k == 2920 {
                                let size = if side == LONG {
                                    bt.positions.long[0].size
                                } else {
                                    bt.positions.short[0].size
                                };
                                fill(&mut bt, k, 0, pside, -size, 99.3);
                            }
                            if k == 2922 {
                                fill(&mut bt, k, 0, pside, sign * 0.64, 98.1);
                            }
                            if k == 1468 {
                                bt.balance.usd_total_balance += 13.0;
                            }
                            let reduction_only = (!reopen || k < 1478) && k < 2920;
                            if k == 1478 && reopen {
                                assert!(bt
                                    .replay_hsl_trace(k, Some(side), coin, settings)
                                    .is_none());
                            }
                            let replay = if reduction_only {
                                bt.replay_hsl_trace(k, Some(side), coin, settings)
                            } else {
                                None
                            };
                            if let Some(scope) = replay {
                                hits += 1;
                                bt.hsl_scopes.push(scope);
                            } else {
                                bt.update_hsl(k).unwrap();
                            }
                            let cutoff = bt.hsl_history_cutoff(Some(side), coin);
                            let actual = &bt
                                .hsl_scopes
                                .iter()
                                .find(|s| s.side == Some(side) && s.coin == coin)
                                .unwrap()
                                .result;
                            let full = bt
                                .hsl_inputs_at_clipped(
                                    k,
                                    if mode == "coin" {
                                        snapshot::Mode::Coin
                                    } else {
                                        snapshot::Mode::Pside
                                    },
                                    Some(pside),
                                    coin.map(|_| "C0"),
                                    false,
                                    cutoff,
                                )
                                .unwrap();
                            let expected = evaluate(Input {
                                snapshot: full.snapshot,
                                slots: if coin.is_some() {
                                    full.slots[side] as u64
                                } else {
                                    1
                                },
                                scale_budget_with_excess_allowance: false,
                                exposure_budget: None,
                                span: settings.span,
                                threshold: settings.threshold,
                                cooldown_ms: settings.cooldown_ms,
                                restart,
                            })
                            .unwrap();
                            let a = actual.decision.as_ref().unwrap();
                            let b = expected.decision.as_ref().unwrap();
                            assert_eq!(a.action, b.action, "{mode} side={side} k={k}");
                            assert_eq!(a.red_at, b.red_at, "{mode} side={side} k={k}");
                            assert_eq!(a.flat_at, b.flat_at, "{mode} side={side} k={k}");
                            // The existing non-rolling cache retains only the
                            // latest episode after flatten/reopen, so its total
                            // historical diagnostics intentionally differ.
                            if reduction_only {
                                assert_eq!(actual.observations, expected.observations, "k={k}");
                                assert_eq!(actual.reasons, expected.reasons, "k={k}");
                            }
                            assert!(
                                (a.raw - b.raw).abs() < 1e-12,
                                "raw k={k}: {} != {}",
                                a.raw,
                                b.raw
                            );
                            assert!(
                                (a.ema - b.ema).abs() < 1e-12,
                                "ema k={k}: {} != {}",
                                a.ema,
                                b.ema
                            );
                            if reduction_only {
                                assert_eq!(actual.events.len(), expected.events.len());
                                for (a, b) in actual.events.iter().zip(&expected.events) {
                                    // Incremental EMA seed correction changes floating-point
                                    // operation order, just as the terminal EMA above does.
                                    assert!((a.raw.unwrap() - b.raw.unwrap()).abs() < 1e-12);
                                    assert!((a.ema.unwrap() - b.ema.unwrap()).abs() < 1e-12);
                                    let mut a = serde_json::to_value(a).unwrap();
                                    let mut b = serde_json::to_value(b).unwrap();
                                    for key in ["raw", "ema"] {
                                        a.as_object_mut().unwrap().remove(key);
                                        b.as_object_mut().unwrap().remove(key);
                                    }
                                    assert_eq!(a, b, "event identity k={k}");
                                }
                            }
                        }
                        assert!(hits >= 20, "rolling cache must be exercised: {hits}");
                    }
                }
            }
        }
    }
    #[test]
    /// Current estimates receive fresh GREEN/RED signals without invented age.
    fn estimated_current_singleton_reuses_only_completed_green_history() {
        use crate::backtest::hsl_runtime::{Config, Policy, SignalSettings};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        for side in [LONG, SHORT] {
            for mode in ["coin", "pside"] {
                let mut data = candles(1600, 1);
                for k in 0..1600 {
                    for field in [HIGH, LOW, CLOSE] {
                        data[[k, 0, field]] = if k >= 1455 { if side == LONG { 90.0 } else { 110.0 } } else { 95.0 + ((k * 17) % 13) as f64 };
                    }
                }
                let btc = Array1::from_elem(1600, 50_000.0);
                let mut bt = make(&data, &btc);
                let policy = Policy { enabled: true, scale_budget_with_excess_allowance: false,
                    red_threshold: 0.4, ema_span_minutes: 2103.0,
                    cooldown_minutes_after_red: 5.0, restart_after_red_policy: Some("always".into()),
                    panic_close_order_type: "limit".into() };
                bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                    mode: mode.into(), sides: [policy.clone(), policy],
                    portfolio: None, coins: Default::default() });
                let pside = if side == LONG { PositionSide::Long } else { PositionSide::Short };
                let sign = if side == LONG { 1.0 } else { -1.0 };
                fill(&mut bt, 1, 0, pside, sign, 100.0);
                fill(&mut bt, 500, 0, pside, -sign * 0.8, 101.0);
                fill(&mut bt, 1000, 0, pside, sign, 99.0);
                fill(&mut bt, 1100, 0, pside, -sign * 1.1, 101.0);
                let coin = (mode == "coin").then_some(0);
                let settings = SignalSettings { scale_budget_with_excess_allowance: false,
                    span: 2103.0, threshold: 0.4, cooldown_ms: 300_000, restart: Restart::Always };
                let mut hits = 0;
                for k in 1448..1480 {
                    bt.current_step = k;
                    if k == 1455 {
                        // A newly reported authoritative size invalidates the
                        // cache; its fresh singleton can legitimately be RED.
                        if side == LONG { bt.positions.long[0].size = 50.0; }
                        else { bt.positions.short[0].size = -50.0; }
                    }
                    if let Some(scope) = bt.replay_hsl_trace(k, Some(side), coin, settings) {
                        hits += 1;
                        bt.hsl_scopes.push(scope);
                    } else { bt.update_hsl(k).unwrap(); }
                    let cutoff = bt.hsl_history_cutoff(Some(side), coin);
                    let full = bt.hsl_inputs_at_clipped(k,
                        if coin.is_some() { snapshot::Mode::Coin } else { snapshot::Mode::Pside },
                        Some(pside), coin.map(|_| "C0"), false, cutoff).unwrap();
                    let expected = evaluate(Input { snapshot: full.snapshot,
                        slots: if coin.is_some() { full.slots[side] as u64 } else { 1 },
                        scale_budget_with_excess_allowance: false, exposure_budget: None,
                        span: settings.span, threshold: settings.threshold,
                        cooldown_ms: settings.cooldown_ms, restart: settings.restart }).unwrap();
                    let actual = &bt.hsl_scopes.iter().find(|s| s.side == Some(side) && s.coin == coin).unwrap().result;
                    if k >= 1456 {
                        assert_eq!(actual.decision.as_ref().unwrap().action, crate::hsl_controller::Action::Panic);
                        assert_eq!(actual.decision.as_ref().unwrap().red_at, Some(actual.decision.as_ref().unwrap().timestamp));
                    }
                    assert!(actual.reasons.contains("estimated_current_opening"), "{mode} {side} {k}");
                    assert_eq!(actual.reasons, expected.reasons);
                    assert_eq!(actual.observations, expected.observations);
                    assert_eq!(serde_json::to_value(&actual.decision).unwrap(), serde_json::to_value(&expected.decision).unwrap());
                    assert_eq!(serde_json::to_value(&actual.events).unwrap(), serde_json::to_value(&expected.events).unwrap());
                }
                assert!(hits >= 20, "estimated singleton cache hits={hits}");
            }
        }
    }
    #[test]
    /// Mixed partial closes and additions preserve authoritative HSL decisions.
    fn mixed_partial_close_window_matches_fresh_reconstruction() {
        use crate::backtest::hsl_runtime::{Config, Policy, SignalSettings};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        for reopen in [true] {
            for side in [LONG, SHORT] {
                for mode in ["coin", "pside"] {
                    for restart in [Restart::Always, Restart::Never] {
                        let mut data = candles(3000, 1);
                        for k in 0..3000 {
                            for field in [HIGH, LOW, CLOSE] {
                                data[[k, 0, field]] = 85.0 + ((k * 17) % 53) as f64;
                            }
                        }
                        let btc = Array1::from_elem(3000, 50000.);
                        let mut bt = make(&data, &btc);
                        bt.exchange_params_list[0].qty_step = 0.00001;
                        let policy = Policy {
                            enabled: true,
                            scale_budget_with_excess_allowance: false,
                            red_threshold: 0.4,
                            ema_span_minutes: 17.5,
                            cooldown_minutes_after_red: 5.,
                            restart_after_red_policy: Some(if restart == Restart::Always {
                                "always".into()
                            } else {
                                "never".into()
                            }),
                            panic_close_order_type: "limit".into(),
                        };
                        bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                            mode: mode.into(),
                            sides: [policy.clone(), policy.clone()],
                            portfolio: None,
                            coins: Default::default(),
                        });
                        let pside = if side == LONG {
                            PositionSide::Long
                        } else {
                            PositionSide::Short
                        };
                        let sign = if side == LONG { 1.0 } else { -1.0 };
                        fill(&mut bt, 1, 0, pside, sign * 0.1713, 93.2);
                        let coin = (mode == "coin").then_some(0);
                        let settings = SignalSettings {
                            scale_budget_with_excess_allowance: false,
                            span: policy.ema_span_minutes,
                            threshold: policy.red_threshold,
                            cooldown_ms: 300_000,
                            restart,
                        };
                        let mut hits = 0;
                        let mut mixed_hits = 0;
                        let mut closed_hits = 0;
                        for k in (1448..1540).chain(2888..2930) {
                            bt.current_step = k;
                            if k == 1450 || k == 1460 {
                                fill(&mut bt, k, 0, pside, -sign * 0.0627, 107.4);
                            }
                            if k == 1478 && reopen {
                                fill(&mut bt, k, 0, pside, sign * 0.0083, 91.2);
                            }
                            if k == 1480 {
                                fill(&mut bt, k, 0, pside, -sign * 0.004, 108.2);
                            }
                            if k == 1510 {
                                let size = if side == LONG {
                                    bt.positions.long[0].size
                                } else {
                                    bt.positions.short[0].size
                                };
                                fill(&mut bt, k, 0, pside, -size, 99.3);
                            }
                            if k == 2922 {
                                fill(&mut bt, k, 0, pside, sign * 0.0064, 98.1);
                            }
                            if k == 1468 {
                                bt.balance.usd_total_balance += 13.0;
                            }
                            if k == 1482 {
                                bt.hsl_scopes.clear();
                                bt.hsl_traces.clear();
                            }
                            let reduction_only = k < 2920;
                            if k == 1478 && reopen {
                                assert!(bt
                                    .replay_hsl_trace(k, Some(side), coin, settings)
                                    .is_none());
                            }
                            let replay = if reduction_only {
                                let fast = bt.advance_hsl_scope(k, Some(side), coin, settings);
                                if fast.is_some() && (1513..1540).contains(&k) { closed_hits += 1; }
                                fast.or_else(|| bt.replay_hsl_trace(k, Some(side), coin, settings))
                            } else {
                                None
                            };
                            if let Some(scope) = replay {
                                hits += 1;
                                if k > 1482 && k < 1510 { mixed_hits += 1; }
                                bt.hsl_scopes.push(scope);
                            } else {
                                bt.update_hsl(k).unwrap();
                            }
                            let cutoff = bt.hsl_history_cutoff(Some(side), coin);
                            let actual = &bt
                                .hsl_scopes
                                .iter()
                                .find(|s| s.side == Some(side) && s.coin == coin)
                                .unwrap()
                                .result;
                            let full = bt
                                .hsl_inputs_at_clipped(
                                    k,
                                    if mode == "coin" {
                                        snapshot::Mode::Coin
                                    } else {
                                        snapshot::Mode::Pside
                                    },
                                    Some(pside),
                                    coin.map(|_| "C0"),
                                    false,
                                    cutoff,
                                )
                                .unwrap();
                            let expected = evaluate(Input {
                                snapshot: full.snapshot,
                                slots: if coin.is_some() {
                                    full.slots[side] as u64
                                } else {
                                    1
                                },
                                scale_budget_with_excess_allowance: false,
                                exposure_budget: None,
                                span: settings.span,
                                threshold: settings.threshold,
                                cooldown_ms: settings.cooldown_ms,
                                restart,
                            })
                            .unwrap();
                            let a = actual.decision.as_ref().unwrap();
                            let b = expected.decision.as_ref().unwrap();
                            assert_eq!(a.action, b.action, "{mode} side={side} k={k}");
                            assert_eq!(a.red_at, b.red_at, "{mode} side={side} k={k}");
                            assert_eq!(a.flat_at, b.flat_at, "{mode} side={side} k={k}");
                            // The existing non-rolling cache retains only the
                            // latest episode after flatten/reopen, so its total
                            // historical diagnostics intentionally differ.
                            if reduction_only && k < 1478 {
                                assert_eq!(actual.observations, expected.observations, "k={k}");
                                assert_eq!(actual.reasons, expected.reasons, "k={k}");
                            }
                            assert!(
                                (a.raw - b.raw).abs() < 1e-12,
                                "raw k={k}: {} != {}",
                                a.raw,
                                b.raw
                            );
                            assert!(
                                (a.ema - b.ema).abs() < 1e-12,
                                "ema k={k}: {} != {}",
                                a.ema,
                                b.ema
                            );
                            if reduction_only && (k < 1478 || (1482..1540).contains(&k)) {
                                assert_eq!(actual.events.len(), expected.events.len(), "events mode={mode} side={side} k={k}");
                                for (a, b) in actual.events.iter().zip(&expected.events) {
                                    // Incremental EMA seed correction changes floating-point
                                    // operation order, just as the terminal EMA above does.
                                    for (x, y) in [(a.raw, b.raw), (a.ema, b.ema)] {
                                        match (x, y) {
                                            (Some(x), Some(y)) => assert!((x - y).abs() < 1e-12),
                                            (None, None) => (),
                                            _ => panic!("event numeric fields differ k={k}"),
                                        }
                                    }
                                    let mut a = serde_json::to_value(a).unwrap();
                                    let mut b = serde_json::to_value(b).unwrap();
                                    for key in ["raw", "ema"] {
                                        a.as_object_mut().unwrap().remove(key);
                                        b.as_object_mut().unwrap().remove(key);
                                    }
                                    assert_eq!(a, b, "event identity k={k}");
                                }
                            }
                        }
                        assert!(hits >= 20, "rolling cache must be exercised: {hits}");
                        assert!(mixed_hits >= 20, "mixed cache hits={mixed_hits}");
                        assert!(closed_hits >= 20, "closed reconstructed cache hits={closed_hits}");
                    }
                }
            }
        }
    }
    #[test]
    fn closed_reduction_window_matches_fresh_reference_until_execution_expiry() {
        use crate::backtest::hsl_runtime::{Config, Policy, SignalSettings};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        for side in [LONG, SHORT] {
            for reopen in [false, true] {
                let mut data = candles(3000, 1);
                for k in 0..3000 {
                    for field in [HIGH, LOW, CLOSE] {
                        data[[k, 0, field]] = 85.0 + ((k * 17) % 53) as f64;
                    }
                }
                let btc = Array1::from_elem(3000, 50000.);
                let mut bt = make(&data, &btc);
                let policy = Policy {
                    enabled: true,
                    scale_budget_with_excess_allowance: false,
                    red_threshold: 0.9,
                    ema_span_minutes: 17.5,
                    cooldown_minutes_after_red: 5.,
                    restart_after_red_policy: Some("always".into()),
                    panic_close_order_type: "limit".into(),
                };
                bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                    mode: "coin".into(),
                    sides: [policy.clone(), policy.clone()],
                    portfolio: None,
                    coins: Default::default(),
                });
                let pside = if side == LONG {
                    PositionSide::Long
                } else {
                    PositionSide::Short
                };
                let sign = if side == LONG { 1.0 } else { -1.0 };
                fill(&mut bt, 1, 0, pside, sign, 93.2);
                fill(&mut bt, 400, 0, pside, -sign, 107.4);
                let settings = SignalSettings {
                    scale_budget_with_excess_allowance: false,
                    span: 17.5,
                    threshold: 0.9,
                    cooldown_ms: 300_000,
                    restart: Restart::Always,
                };
                let mut clipped_hits = 0;
                for k in (1438..1470).chain(1810..1860) {
                    bt.current_step = k;
                    if reopen && k == 1455 {
                        fill(&mut bt, k, 0, pside, sign, 91.2);
                    }
                    if reopen && k == 1462 {
                        fill(&mut bt, k, 0, pside, -sign, 99.3);
                    }
                    if let Some(scope) = bt.advance_hsl_scope(k, Some(side), Some(0), settings) {
                        if k >= 1445 && k < 1470 && !reopen {
                            clipped_hits += 1;
                        }
                        bt.hsl_scopes.push(scope);
                    } else {
                        bt.update_hsl(k).unwrap();
                    }
                    let cutoff = bt.hsl_history_cutoff(Some(side), Some(0));
                    let full = bt
                        .hsl_inputs_at_clipped(
                            k,
                            snapshot::Mode::Coin,
                            Some(pside),
                            Some("C0"),
                            false,
                            cutoff,
                        )
                        .unwrap();
                    let source_reasons = full.reasons.clone();
                    let mut expected = evaluate(Input {
                        snapshot: full.snapshot,
                        slots: full.slots[side] as u64,
                        scale_budget_with_excess_allowance: false,
                        exposure_budget: None,
                        span: settings.span,
                        threshold: settings.threshold,
                        cooldown_ms: settings.cooldown_ms,
                        restart: settings.restart,
                    })
                    .unwrap();
                    expected.reasons.extend(source_reasons);
                    let actual = &bt
                        .hsl_scopes
                        .iter()
                        .find(|s| s.side == Some(side) && s.coin == Some(0))
                        .unwrap()
                        .result;
                    let a = actual.decision.as_ref().unwrap();
                    let b = expected.decision.as_ref().unwrap();
                    assert_eq!(a.action, b.action, "side={side} k={k}");
                    assert_eq!(a.red_at, b.red_at);
                    assert_eq!(a.flat_at, b.flat_at);
                    assert!((a.raw - b.raw).abs() < 1e-12);
                    assert!((a.ema - b.ema).abs() < 1e-12);
                    if reopen && k >= 1464 {
                        assert_eq!(
                            actual.observations, expected.observations,
                            "reopened observations k={k}"
                        );
                        assert_eq!(actual.reasons, expected.reasons);
                        assert_eq!(
                            serde_json::to_value(&actual.events).unwrap(),
                            serde_json::to_value(&expected.events).unwrap()
                        );
                    }
                    if !reopen {
                        // Once the last execution expires, the pre-existing
                        // all-flat scalar cache has separate count semantics.
                        // Require fresh counts throughout this closed-tape proof.
                        if k < 1841 {
                            assert_eq!(
                                actual.observations, expected.observations,
                                "observations k={k}"
                            );
                        }
                        assert_eq!(actual.reasons, expected.reasons, "reasons k={k}");
                        assert_eq!(
                            serde_json::to_value(&actual.events).unwrap(),
                            serde_json::to_value(&expected.events).unwrap()
                        );
                    }
                }
                if !reopen {
                    assert!(clipped_hits >= 20, "clipped flat cache hits={clipped_hits}");
                }
            }
        }
    }    #[test]
    fn settled_red_history_matches_current_reference_until_execution_expiry() {
        use crate::backtest::hsl_runtime::{Config, Policy, SignalSettings};
        use crate::hsl_controller::Restart;
        use crate::hsl_evaluator::{evaluate, Input};
        for (threshold, restart) in [(0.005, Restart::Always), (0.005, Restart::Never), (0.9, Restart::Always), (0.9, Restart::Never)] {
        for side in [LONG, SHORT] {
            for reopen in [false, true] {
                let mut data = candles(3000, 1);
                for k in 0..3000 {
                    for field in [HIGH, LOW, CLOSE] {
                        data[[k, 0, field]] = 85.0 + ((k * 17) % 53) as f64;
                    }
                }
                let btc = Array1::from_elem(3000, 50000.);
                let mut bt = make(&data, &btc);
                let policy = Policy {
                    enabled: true,
                    scale_budget_with_excess_allowance: false,
                    red_threshold: threshold,
                    ema_span_minutes: 17.5,
                    cooldown_minutes_after_red: 5.,
                    restart_after_red_policy: Some(if restart == Restart::Always { "always" } else { "never" }.into()),
                    panic_close_order_type: "limit".into(),
                };
                bt.backtest_params.equity_hard_stop_loss.hsl = Some(Config {
                    mode: "coin".into(),
                    sides: [policy.clone(), policy.clone()],
                    portfolio: None,
                    coins: Default::default(),
                });
                let pside = if side == LONG {
                    PositionSide::Long
                } else {
                    PositionSide::Short
                };
                let sign = if side == LONG { 1.0 } else { -1.0 };
                fill(&mut bt, 1, 0, pside, sign, 93.2);
                fill(&mut bt, 400, 0, pside, -sign, 107.4);
                let settings = SignalSettings {
                    scale_budget_with_excess_allowance: false,
                    span: 17.5,
                    threshold,
                    cooldown_ms: 300_000,
                    restart,
                };
                let mut actual_report = crate::backtest::hsl_report::Report::new(false, true);
                let mut reference_report = crate::backtest::hsl_report::Report::new(false, true);
                let mut clipped_hits = 0;
                for k in (1438..1470).chain(1810..1860) {
                    bt.current_step = k;
                    if k == 1815 {
                        bt.balance.usd_total_balance = 1.0;
                        bt.balance.usd_cash_wallet = 1.0;
                    }
                    if reopen && k == 1455 {
                        fill(&mut bt, k, 0, pside, sign, 91.2);
                    }
                    if reopen && k == 1462 {
                        fill(&mut bt, k, 0, pside, -sign, 99.3);
                    }
                    if let Some(scope) = bt.advance_hsl_scope(k, Some(side), Some(0), settings) {
                        if k >= 1445 && k < 1470 && !reopen {
                            clipped_hits += 1;
                        }
                        bt.hsl_scopes.push(scope);
                    } else {
                        bt.update_hsl(k).unwrap();
                    }
                    let cutoff = bt.hsl_history_cutoff(Some(side), Some(0));
                    let full = bt
                        .hsl_inputs_at_clipped(
                            k,
                            snapshot::Mode::Coin,
                            Some(pside),
                            Some("C0"),
                            false,
                            cutoff,
                        )
                        .unwrap();
                    let source_reasons = full.reasons.clone();
                    let mut expected = evaluate(Input {
                        snapshot: full.snapshot,
                        slots: full.slots[side] as u64,
                        scale_budget_with_excess_allowance: false,
                        exposure_budget: None,
                        span: settings.span,
                        threshold: settings.threshold,
                        cooldown_ms: settings.cooldown_ms,
                        restart: settings.restart,
                    })
                    .unwrap();
                    expected.reasons.extend(source_reasons);
                    let actual = &bt
                        .hsl_scopes
                        .iter()
                        .find(|s| s.side == Some(side) && s.coin == Some(0))
                        .unwrap()
                        .result;
                    let a = actual.decision.as_ref().unwrap();
                    let b = expected.decision.as_ref().unwrap();
                    assert_eq!(a.action, b.action, "side={side} k={k}");
                    assert_eq!(a.red_at, b.red_at);
                    assert_eq!(a.flat_at, b.flat_at);
                    assert!((a.raw - b.raw).abs() < 1e-12);
                    assert!((a.ema - b.ema).abs() < 1e-12);
                    if !reopen {
                    actual_report.observe((Some(side), Some(0)), a.timestamp, "bar_close", actual);
                    reference_report.observe((Some(side), Some(0)), b.timestamp, "bar_close", &expected);
                    assert_eq!(serde_json::to_value(&actual_report.summary).unwrap(), serde_json::to_value(&reference_report.summary).unwrap(), "summary {restart:?} threshold={threshold} side={side} k={k}");
                    assert_eq!(serde_json::to_value(&actual_report.events).unwrap(), serde_json::to_value(&reference_report.events).unwrap(), "reported events k={k}");
                    }
                    if reopen && k >= 1464 && k < 1815 {
                        assert_eq!(
                            actual.observations, expected.observations,
                            "reopened observations k={k}"
                        );
                        assert_eq!(actual.reasons, expected.reasons);
                        assert!(actual.events.iter().all(|e| e.timestamp <= a.timestamp));
                    }
                    if !reopen {
                        // Once the last execution expires, the pre-existing
                        // all-flat scalar cache has separate count semantics.
                        // Require fresh counts throughout this closed-tape proof.
                        if k < 1815 {
                            assert_eq!(
                                actual.observations, expected.observations,
                                "observations k={k}"
                            );
                        }
                        assert_eq!(actual.reasons, expected.reasons, "reasons k={k}");
                        assert!(actual.events.iter().all(|e| e.timestamp <= a.timestamp));
                    }
                }
                if !reopen && restart == Restart::Always {
                    assert!(clipped_hits >= 20, "clipped flat cache hits={clipped_hits}");
                }
            }
        }
        }
    }
}
