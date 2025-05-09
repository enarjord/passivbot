use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::constants::{CLOSE, HIGH, LONG, LOW, NO_POS, SHORT, VOLUME};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_min_entry_qty, calc_next_entry_long,
    calc_next_entry_short,
};
use crate::types::{
    Analysis, BacktestParams, Balance, BotParams, BotParamsPair, EMABands, Equities,
    ExchangeParams, Fill, Order, OrderBook, OrderType, Position, Positions, StateParams,
    TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, hysteresis_rounding, qty_to_cost,
    round_, round_dn, round_up,
};
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView3, Axis, Dim, ViewRepr};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Default, Copy, Debug)]
pub struct EmaAlphas {
    pub long: Alphas,
    pub short: Alphas,
}

#[derive(Clone, Default, Copy, Debug)]
pub struct Alphas {
    pub alphas: [f64; 3],
    pub alphas_inv: [f64; 3],
}

#[derive(Debug)]
pub struct EMAs {
    pub long: [f64; 3],
    pub short: [f64; 3],
}
impl EMAs {
    pub fn compute_bands(&self, pside: usize) -> EMABands {
        let (upper, lower) = match pside {
            LONG => (
                *self
                    .long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                *self
                    .long
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            ),
            SHORT => (
                *self
                    .short
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                *self
                    .short
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            ),
            _ => panic!("Invalid pside"),
        };
        EMABands { upper, lower }
    }
}

#[derive(Debug, Default)]
pub struct OpenOrdersNew {
    pub long: HashMap<usize, OpenOrderBundleNew>,
    pub short: HashMap<usize, OpenOrderBundleNew>,
}

#[derive(Debug, Default)]
pub struct OpenOrderBundleNew {
    pub entries: Vec<Order>,
    pub closes: Vec<Order>,
}

#[derive(Default, Debug)]
pub struct Actives {
    long: HashSet<usize>,
    short: HashSet<usize>,
}

#[derive(Default, Debug)]
pub struct IsStuck {
    long: HashSet<usize>,
    short: HashSet<usize>,
}

#[derive(Default, Debug)]
pub struct TrailingPrices {
    pub long: HashMap<usize, TrailingPriceBundle>,
    pub short: HashMap<usize, TrailingPriceBundle>,
}

pub struct TrailingEnabled {
    long: bool,
    short: bool,
}

pub struct TradingEnabled {
    long: bool,
    short: bool,
}

pub struct RollingVolumeSum {
    long: Vec<f64>,
    short: Vec<f64>,
    prev_k_long: usize,
    prev_k_short: usize,
}

pub struct Backtest<'a> {
    hlcvs: &'a ArrayView3<'a, f64>,
    btc_usd_prices: &'a ArrayView1<'a, f64>, // Change to ArrayView1 (1D view)
    bot_params_pair: BotParamsPair,
    exchange_params_list: Vec<ExchangeParams>,
    backtest_params: BacktestParams,
    pub balance: Balance,
    n_coins: usize,
    ema_alphas: EmaAlphas,
    emas: Vec<EMAs>,
    positions: Positions,
    open_orders: OpenOrdersNew,
    trailing_prices: TrailingPrices,
    actives: Actives,
    pnl_cumsum_running: f64,
    pnl_cumsum_max: f64,
    fills: Vec<Fill>,
    is_stuck: IsStuck,
    trading_enabled: TradingEnabled,
    trailing_enabled: TrailingEnabled,
    equities: Equities,
    last_valid_timestamps: HashMap<usize, usize>,
    first_valid_timestamps: HashMap<usize, usize>,
    did_fill_long: HashSet<usize>,
    did_fill_short: HashSet<usize>,
    n_eligible_long: usize,
    n_eligible_short: usize,
    rolling_volume_sum: RollingVolumeSum,
    volume_indices_buffer: Option<Vec<(f64, usize)>>,
}

impl<'a> Backtest<'a> {
    pub fn new(
        hlcvs: &'a ArrayView3<'a, f64>,
        btc_usd_prices: &'a ArrayView1<'a, f64>, // Updated parameter type
        bot_params_pair: BotParamsPair,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
        // Determine if BTC collateral is used
        let mut balance = Balance::default();
        balance.use_btc_collateral = btc_usd_prices.iter().any(|&p| p != 1.0);

        // Initialize balances
        balance.btc = if balance.use_btc_collateral {
            backtest_params.starting_balance / btc_usd_prices[0]
        } else {
            0.0
        };
        balance.usd = if balance.use_btc_collateral {
            0.0
        } else {
            backtest_params.starting_balance
        };
        balance.usd_total = backtest_params.starting_balance;
        balance.usd_total_rounded_last = balance.usd_total;
        balance.usd_total_rounded = balance.usd_total;

        let n_timesteps = hlcvs.shape()[0];
        let n_coins = hlcvs.shape()[1];
        let initial_emas = (0..n_coins)
            .map(|i| {
                let close_price = hlcvs[[0, i, CLOSE]];
                EMAs {
                    long: [close_price; 3],
                    short: [close_price; 3],
                }
            })
            .collect();
        let mut equities = Equities::default();
        equities.usd.push(backtest_params.starting_balance);
        equities.btc.push(balance.btc); // Initial BTC equity
        let mut bot_params_pair_cloned = bot_params_pair.clone();
        bot_params_pair_cloned.long.n_positions = n_coins.min(bot_params_pair.long.n_positions);
        bot_params_pair_cloned.short.n_positions = n_coins.min(bot_params_pair.short.n_positions);
        let n_eligible_long = bot_params_pair_cloned.long.n_positions.max(
            (n_coins as f64 * (1.0 - bot_params_pair.long.filter_volume_drop_pct)).round() as usize,
        );
        let n_eligible_short = bot_params_pair_cloned.short.n_positions.max(
            (n_coins as f64 * (1.0 - bot_params_pair.short.filter_volume_drop_pct)).round()
                as usize,
        );
        Backtest {
            hlcvs,
            btc_usd_prices,
            bot_params_pair: bot_params_pair_cloned,
            exchange_params_list,
            backtest_params: backtest_params.clone(),
            balance,
            n_coins,
            ema_alphas: calc_ema_alphas(&bot_params_pair),
            emas: initial_emas,
            positions: Positions::default(),
            open_orders: OpenOrdersNew::default(),
            trailing_prices: TrailingPrices::default(),
            actives: Actives::default(),
            pnl_cumsum_running: 0.0,
            pnl_cumsum_max: 0.0,
            fills: Vec::new(),
            is_stuck: IsStuck::default(),
            trading_enabled: TradingEnabled {
                long: bot_params_pair.long.wallet_exposure_limit != 0.0
                    && bot_params_pair.long.n_positions > 0,
                short: bot_params_pair.short.wallet_exposure_limit != 0.0
                    && bot_params_pair.short.n_positions > 0,
            },
            trailing_enabled: TrailingEnabled {
                long: bot_params_pair.long.close_trailing_grid_ratio != 0.0
                    || bot_params_pair.long.entry_trailing_grid_ratio != 0.0,
                short: bot_params_pair.short.close_trailing_grid_ratio != 0.0
                    || bot_params_pair.short.entry_trailing_grid_ratio != 0.0,
            },
            equities: equities,
            last_valid_timestamps: HashMap::new(),
            first_valid_timestamps: HashMap::new(),
            did_fill_long: HashSet::new(),
            did_fill_short: HashSet::new(),
            n_eligible_long,
            n_eligible_short,
            rolling_volume_sum: RollingVolumeSum {
                long: vec![0.0; n_coins],
                short: vec![0.0; n_coins],
                prev_k_long: 0,
                prev_k_short: 0,
            },
            volume_indices_buffer: Some(vec![(0.0, 0); n_coins]), // Initialize here
        }
    }

    pub fn calc_preferred_coins(&mut self, k: usize, pside: usize) -> Vec<usize> {
        let (bot_params, n_positions) = match pside {
            LONG => (
                &self.bot_params_pair.long,
                self.bot_params_pair.long.n_positions,
            ),
            SHORT => (
                &self.bot_params_pair.short,
                self.bot_params_pair.short.n_positions,
            ),
            _ => panic!("Invalid pside"),
        };

        if self.n_coins <= n_positions {
            return (0..self.n_coins).collect();
        }
        let volume_filtered = self.filter_by_relative_volume(k, pside);
        self.rank_by_noisiness(k, &volume_filtered, pside)
    }

    fn filter_by_relative_volume(&mut self, k: usize, pside: usize) -> Vec<usize> {
        let bot_params = match pside {
            LONG => &self.bot_params_pair.long,
            SHORT => &self.bot_params_pair.short,
            _ => panic!("Invalid pside"),
        };
        let window = bot_params.filter_volume_rolling_window;
        let start_k = k.saturating_sub(window);

        let (rolling_volume_sum, prev_k) = match pside {
            LONG => (
                &mut self.rolling_volume_sum.long,
                &mut self.rolling_volume_sum.prev_k_long,
            ),
            SHORT => (
                &mut self.rolling_volume_sum.short,
                &mut self.rolling_volume_sum.prev_k_short,
            ),
            _ => panic!("Invalid pside"),
        };

        let volume_indices = self.volume_indices_buffer.as_mut().unwrap();

        if k > window && k - *prev_k < window {
            let safe_start = (*prev_k).saturating_sub(window);
            for idx in 0..self.n_coins {
                rolling_volume_sum[idx] -=
                    self.hlcvs.slice(s![safe_start..start_k, idx, VOLUME]).sum();
                rolling_volume_sum[idx] += self.hlcvs.slice(s![*prev_k..k, idx, VOLUME]).sum();
                volume_indices[idx] = (rolling_volume_sum[idx], idx);
            }
        } else {
            for idx in 0..self.n_coins {
                rolling_volume_sum[idx] = self.hlcvs.slice(s![start_k..k, idx, VOLUME]).sum();
                volume_indices[idx] = (rolling_volume_sum[idx], idx);
            }
        }
        *prev_k = k;

        volume_indices.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let n_eligible = match pside {
            LONG => self.n_eligible_long,
            SHORT => self.n_eligible_short,
            _ => panic!("Invalid pside"),
        };
        volume_indices
            .iter()
            .take(n_eligible.min(self.n_coins))
            .map(|&(_, idx)| idx)
            .collect()
    }

    fn rank_by_noisiness(&self, k: usize, candidates: &[usize], pside: usize) -> Vec<usize> {
        let bot_params = match pside {
            LONG => &self.bot_params_pair.long,
            SHORT => &self.bot_params_pair.short,
            _ => panic!("Invalid pside"),
        };
        let start_k = k.saturating_sub(bot_params.filter_noisiness_rolling_window);

        let mut noisinesses: Vec<(f64, usize)> = candidates
            .iter()
            .map(|&idx| {
                let noisiness: f64 = self
                    .hlcvs
                    .slice(s![start_k..k, idx, ..])
                    .axis_iter(Axis(0))
                    .map(|row| (row[HIGH] - row[LOW]) / row[CLOSE])
                    .sum();
                (noisiness, idx)
            })
            .collect();

        noisinesses.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        noisinesses.into_iter().map(|(_, idx)| idx).collect()
    }

    pub fn run(&mut self) -> (Vec<Fill>, Equities) {
        let n_timesteps = self.hlcvs.shape()[0];
        for idx in 0..self.n_coins {
            self.trailing_prices
                .long
                .insert(idx, TrailingPriceBundle::default());
            self.trailing_prices
                .short
                .insert(idx, TrailingPriceBundle::default());
        }

        // --- find first & last valid candle for every coin (binary-search) ---
        let (first_valid, last_valid) = find_valid_timestamp_bounds(&self.hlcvs);
        for idx in 0..self.n_coins {
            self.first_valid_timestamps.insert(idx, first_valid[idx]);
            if n_timesteps - last_valid[idx] > 1400 {
                // add only if delisted more than one day before last timestamp
                self.last_valid_timestamps.insert(idx, last_valid[idx]); // keep same name for callers
            }
        }

        for k in 1..(n_timesteps - 1) {
            self.check_for_fills(k);
            self.update_emas(k);
            let mut balance_changed = false;
            if self.balance.use_btc_collateral {
                self.balance.usd_total =
                    (self.balance.btc * self.btc_usd_prices[k]) + self.balance.usd;
                self.balance.btc_total = self.balance.usd_total / self.btc_usd_prices[k];
                let new_usd_total_rounded = hysteresis_rounding(
                    self.balance.usd_total,
                    self.balance.usd_total_rounded_last,
                    0.02,
                    0.5,
                );
                if new_usd_total_rounded != self.balance.usd_total_rounded {
                    self.balance.usd_total_rounded_last = self.balance.usd_total_rounded;
                    self.balance.usd_total_rounded = new_usd_total_rounded;
                    balance_changed = true;
                }
            }
            if balance_changed || !self.did_fill_long.is_empty() || !self.did_fill_short.is_empty()
            {
                self.update_open_orders_any_fill(k);
            } else {
                self.update_open_orders_no_fill(k);
            }
            self.update_equities(k);
        }
        (self.fills.clone(), self.equities.clone())
    }

    fn create_state_params(&self, k: usize, idx: usize, pside: usize) -> StateParams {
        let close_price = self.hlcvs[[k, idx, CLOSE]];
        StateParams {
            balance: self.balance.usd_total_rounded,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: self.emas[idx].compute_bands(pside),
        }
    }

    fn get_position(&self, idx: usize, pside: usize) -> Position {
        match pside {
            LONG => self.positions.long.get(&idx).cloned().unwrap_or_default(),
            SHORT => self.positions.short.get(&idx).cloned().unwrap_or_default(),
            _ => panic!("Invalid pside"),
        }
    }

    fn update_balance(&mut self, k: usize, mut pnl: f64, fee_paid: f64) {
        if self.balance.use_btc_collateral {
            // Fees reduce USD portion
            self.balance.usd += fee_paid;

            if pnl > 0.0 {
                // If USD balance is negative, offset it with realized PNL first
                if self.balance.usd < 0.0 {
                    let offset_amount = pnl.min(-self.balance.usd);
                    self.balance.usd += offset_amount;
                    pnl -= offset_amount;
                }
                // Any remaining positive PNL is converted to BTC
                if pnl > 0.0 {
                    let btc_to_add = (pnl / self.btc_usd_prices[k]);
                    self.balance.btc += btc_to_add * 0.999; // apply 0.1% spot trading fee
                }
            } else if pnl < 0.0 {
                // Negative PNL directly reduces USD
                self.balance.usd += pnl;
            }

            // Now recalc totals
            self.balance.usd_total = (self.balance.btc * self.btc_usd_prices[k]) + self.balance.usd;
            self.balance.btc_total = self.balance.usd_total / self.btc_usd_prices[k];
        } else {
            // Simple USD-only logic
            self.balance.usd += pnl + fee_paid;

            // Keep total fields consistent
            self.balance.usd_total = self.balance.usd;
            self.balance.usd_total_rounded = self.balance.usd;
            self.balance.usd_total_rounded_last = self.balance.usd;
            self.balance.btc_total = 0.0;
        }
    }

    fn update_equities(&mut self, k: usize) {
        // Start with the “running totals” in our Balance struct
        let mut equity_usd = self.balance.usd_total;
        let mut equity_btc = self.balance.btc_total;

        // Add the unrealized PNL of all positions
        let mut long_keys: Vec<usize> = self.positions.long.keys().cloned().collect();
        long_keys.sort();
        for idx in long_keys {
            let position = &self.positions.long[&idx];
            let current_price = self.hlcvs[[k, idx, CLOSE]];
            let upnl = calc_pnl_long(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity_usd += upnl;
            equity_btc += upnl / self.btc_usd_prices[k];
        }

        let mut short_keys: Vec<usize> = self.positions.short.keys().cloned().collect();
        short_keys.sort();
        for idx in short_keys {
            let position = &self.positions.short[&idx];
            let current_price = self.hlcvs[[k, idx, CLOSE]];
            let upnl = calc_pnl_short(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity_usd += upnl;
            equity_btc += upnl / self.btc_usd_prices[k];
        }

        // Finally push the results into the Equities struct
        self.equities.usd.push(equity_usd);
        self.equities.btc.push(equity_btc);
    }

    fn update_actives(&mut self, k: usize, pside: usize) -> Vec<usize> {
        // Calculate all the information we need before borrowing
        let (positions, n_positions) = match pside {
            LONG => (&self.positions.long, self.bot_params_pair.long.n_positions),
            SHORT => (
                &self.positions.short,
                self.bot_params_pair.short.n_positions,
            ),
            _ => panic!("Invalid pside"),
        };

        // Sort positions to ensure stable iteration
        let mut current_positions: Vec<usize> = positions.keys().cloned().collect();
        current_positions.sort();
        let mut preferred_coins = Vec::new();

        // Only calculate preferred coins if there are open slots
        if current_positions.len() < n_positions {
            preferred_coins = self.calc_preferred_coins(k, pside);
        }

        // Now we can mutably borrow self.actives
        let actives = match pside {
            LONG => &mut self.actives.long,
            SHORT => &mut self.actives.short,
            _ => unreachable!(),
        };

        actives.clear();

        // Add all markets with existing positions
        for &market_idx in &current_positions {
            actives.insert(market_idx);
        }

        let mut actives_without_pos = Vec::new();

        // Add additional markets based on preferred_coins
        for &market_idx in &preferred_coins {
            if actives.len() < n_positions {
                if actives.insert(market_idx) {
                    actives_without_pos.push(market_idx);
                }
            } else {
                break;
            }
        }

        actives_without_pos
    }

    fn check_for_fills(&mut self, k: usize) {
        self.did_fill_long.clear();
        self.did_fill_short.clear();
        if self.trading_enabled.long {
            let mut open_orders_keys_long: Vec<usize> =
                self.open_orders.long.keys().cloned().collect();
            open_orders_keys_long.sort();
            for idx in open_orders_keys_long {
                // Process close fills long
                if !self.open_orders.long[&idx].closes.is_empty() {
                    let mut closes_to_process = Vec::new();
                    {
                        for close_order in &self.open_orders.long[&idx].closes {
                            if self.order_filled(k, idx, close_order) {
                                closes_to_process.push(close_order.clone());
                            }
                        }
                    }
                    for order in closes_to_process {
                        //if order.qty != 0.0 && self.positions.long.contains_key(&idx) && self.positions.long.contains_key(&idx)
                        //if order.qty != 0.0 && self.get_position
                        if self.positions.long.contains_key(&idx) {
                            self.did_fill_long.insert(idx);
                            self.reset_trailing_prices(idx, LONG);
                            self.process_close_fill_long(k, idx, &order);
                        }
                    }
                }
                // Process entry fills long
                if !self.open_orders.long[&idx].entries.is_empty() {
                    let mut entries_to_process = Vec::new();
                    {
                        for entry_order in &self.open_orders.long[&idx].entries {
                            if self.order_filled(k, idx, entry_order) {
                                entries_to_process.push(entry_order.clone());
                            }
                        }
                    }
                    for order in entries_to_process {
                        self.did_fill_long.insert(idx);
                        self.reset_trailing_prices(idx, LONG);
                        self.process_entry_fill_long(k, idx, &order);
                    }
                }
            }
        }
        if self.trading_enabled.short {
            let mut open_orders_keys_short: Vec<usize> =
                self.open_orders.short.keys().cloned().collect();
            open_orders_keys_short.sort();
            for idx in open_orders_keys_short {
                // Process close fills short
                if !self.open_orders.short[&idx].closes.is_empty() {
                    let mut closes_to_process = Vec::new();
                    {
                        for close_order in &self.open_orders.short[&idx].closes {
                            if self.order_filled(k, idx, close_order) {
                                closes_to_process.push(close_order.clone());
                            }
                        }
                    }
                    for order in closes_to_process {
                        if self.positions.short.contains_key(&idx) {
                            self.did_fill_short.insert(idx);
                            self.reset_trailing_prices(idx, SHORT);
                            self.process_close_fill_short(k, idx, &order);
                        }
                    }
                }
                // Process entry fills short
                if !self.open_orders.short[&idx].entries.is_empty() {
                    let mut entries_to_process = Vec::new();
                    {
                        for entry_order in &self.open_orders.short[&idx].entries {
                            if self.order_filled(k, idx, entry_order) {
                                entries_to_process.push(entry_order.clone());
                            }
                        }
                    }
                    for order in entries_to_process {
                        self.did_fill_short.insert(idx);
                        self.reset_trailing_prices(idx, SHORT);
                        self.process_entry_fill_short(k, idx, &order);
                    }
                }
            }
        }
    }

    fn update_stuck_status(&mut self, idx: usize, pside: usize) {
        match pside {
            LONG => {
                if self.positions.long.contains_key(&idx) {
                    let wallet_exposure = calc_wallet_exposure(
                        self.exchange_params_list[idx].c_mult,
                        self.balance.usd_total_rounded,
                        self.positions.long[&idx].size,
                        self.positions.long[&idx].price,
                    );
                    if wallet_exposure / self.bot_params_pair.long.wallet_exposure_limit
                        > self.bot_params_pair.long.unstuck_threshold
                    {
                        self.is_stuck.long.insert(idx);
                    } else {
                        self.is_stuck.long.remove(&idx);
                    }
                } else {
                    self.is_stuck.long.remove(&idx);
                }
            }
            SHORT => {
                if self.positions.short.contains_key(&idx) {
                    let wallet_exposure = calc_wallet_exposure(
                        self.exchange_params_list[idx].c_mult,
                        self.balance.usd_total_rounded,
                        self.positions.short[&idx].size.abs(),
                        self.positions.short[&idx].price,
                    );
                    if wallet_exposure / self.bot_params_pair.short.wallet_exposure_limit
                        > self.bot_params_pair.short.unstuck_threshold
                    {
                        self.is_stuck.short.insert(idx);
                    } else {
                        self.is_stuck.short.remove(&idx);
                    }
                } else {
                    self.is_stuck.short.remove(&idx);
                }
            }
            _ => panic!("Invalid pside in update_stuck_status"),
        }
    }

    fn process_close_fill_long(&mut self, k: usize, idx: usize, close_fill: &Order) {
        let mut new_psize = round_(
            self.positions.long[&idx].size + close_fill.qty,
            self.exchange_params_list[idx].qty_step,
        );
        let mut adjusted_close_qty = close_fill.qty;
        if new_psize < 0.0 {
            println!("warning: close qty greater than psize long");
            println!("coin: {}", self.backtest_params.coins[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", close_fill);
            println!("bot config: {:?}", self.bot_params_pair.long);
            new_psize = 0.0;
            adjusted_close_qty = -self.positions.long[&idx].size;
        }
        let fee_paid = -qty_to_cost(
            adjusted_close_qty,
            close_fill.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let pnl = calc_pnl_long(
            self.positions.long[&idx].price,
            close_fill.price,
            adjusted_close_qty,
            self.exchange_params_list[idx].c_mult,
        );
        self.pnl_cumsum_running += pnl;
        self.pnl_cumsum_max = self.pnl_cumsum_max.max(self.pnl_cumsum_running);
        self.update_balance(k, pnl, fee_paid);

        let current_pprice = self.positions.long[&idx].price;
        if new_psize == 0.0 {
            self.positions.long.remove(&idx);
        } else {
            self.positions.long.get_mut(&idx).unwrap().size = new_psize;
        }
        self.fills.push(Fill {
            index: k,                                      // index minute
            coin: self.backtest_params.coins[idx].clone(), // coin
            pnl,                                           // realized pnl
            fee_paid,                                      // fee paid
            balance_usd_total: self.balance.usd_total,     // balance after fill
            balance_btc: self.balance.btc,                 // Added
            balance_usd: self.balance.usd,                 // Added
            btc_price: self.btc_usd_prices[k],             // Added
            fill_qty: adjusted_close_qty,                  // fill qty
            fill_price: close_fill.price,                  // fill price
            position_size: new_psize,                      // psize after fill
            position_price: current_pprice,                // pprice after fill
            order_type: close_fill.order_type.clone(),     // fill type
        });
    }

    fn process_close_fill_short(&mut self, k: usize, idx: usize, order: &Order) {
        let mut new_psize = round_(
            self.positions.short[&idx].size + order.qty,
            self.exchange_params_list[idx].qty_step,
        );
        let mut adjusted_close_qty = order.qty;
        if new_psize > 0.0 {
            println!("warning: close qty greater than psize short");
            println!("coin: {}", self.backtest_params.coins[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", order);
            new_psize = 0.0;
            adjusted_close_qty = self.positions.short[&idx].size.abs();
        }
        let fee_paid = -qty_to_cost(
            adjusted_close_qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let pnl = calc_pnl_short(
            self.positions.short[&idx].price,
            order.price,
            adjusted_close_qty,
            self.exchange_params_list[idx].c_mult,
        );
        self.pnl_cumsum_running += pnl;
        self.pnl_cumsum_max = self.pnl_cumsum_max.max(self.pnl_cumsum_running);
        self.update_balance(k, pnl, fee_paid);

        let current_pprice = self.positions.short[&idx].price;
        if new_psize == 0.0 {
            self.positions.short.remove(&idx);
        } else {
            self.positions.short.get_mut(&idx).unwrap().size = new_psize;
        }
        self.fills.push(Fill {
            index: k,                                      // index minute
            coin: self.backtest_params.coins[idx].clone(), // coin
            pnl,                                           // realized pnl
            fee_paid,                                      // fee paid
            balance_usd_total: self.balance.usd_total,     // balance after fill
            balance_btc: self.balance.btc,                 // Added
            balance_usd: self.balance.usd,                 // Added
            btc_price: self.btc_usd_prices[k],             // Added
            fill_qty: adjusted_close_qty,                  // fill qty
            fill_price: order.price,                       // fill price
            position_size: new_psize,                      // psize after fill
            position_price: current_pprice,                // pprice after fill
            order_type: order.order_type.clone(),          // fill type
        });
    }

    fn process_entry_fill_long(&mut self, k: usize, idx: usize, order: &Order) {
        // long entry fill
        let fee_paid = -qty_to_cost(
            order.qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        self.update_balance(k, 0.0, fee_paid);

        let position_entry = self
            .positions
            .long
            .entry(idx)
            .or_insert(Position::default());
        let (new_psize, new_pprice) = calc_new_psize_pprice(
            position_entry.size,
            position_entry.price,
            order.qty,
            order.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.long.get_mut(&idx).unwrap().size = new_psize;
        self.positions.long.get_mut(&idx).unwrap().price = new_pprice;
        self.fills.push(Fill {
            index: k,                                        // index minute
            coin: self.backtest_params.coins[idx].clone(),   // coin
            pnl: 0.0,                                        // realized pnl
            fee_paid,                                        // fee paid
            balance_usd_total: self.balance.usd_total,       // balance after fill
            balance_btc: self.balance.btc,                   // Added
            balance_usd: self.balance.usd,                   // Added
            btc_price: self.btc_usd_prices[k],               // Added
            fill_qty: order.qty,                             // fill qty
            fill_price: order.price,                         // fill price
            position_size: self.positions.long[&idx].size,   // psize after fill
            position_price: self.positions.long[&idx].price, // pprice after fill
            order_type: order.order_type.clone(),            // fill type
        });
    }

    fn process_entry_fill_short(&mut self, k: usize, idx: usize, order: &Order) {
        // short entry fill
        let fee_paid = -qty_to_cost(
            order.qty,
            order.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        self.update_balance(k, 0.0, fee_paid);
        let position_entry = self
            .positions
            .short
            .entry(idx)
            .or_insert(Position::default());
        let (new_psize, new_pprice) = calc_new_psize_pprice(
            position_entry.size,
            position_entry.price,
            order.qty,
            order.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.short.get_mut(&idx).unwrap().size = new_psize;
        self.positions.short.get_mut(&idx).unwrap().price = new_pprice;
        self.fills.push(Fill {
            index: k,                                         // index minute
            coin: self.backtest_params.coins[idx].clone(),    // coin
            pnl: 0.0,                                         // realized pnl
            fee_paid,                                         // fee paid
            balance_usd_total: self.balance.usd_total,        // balance after fill
            balance_btc: self.balance.btc,                    // Added
            balance_usd: self.balance.usd,                    // Added
            btc_price: self.btc_usd_prices[k],                // Added
            fill_qty: order.qty,                              // fill qty
            fill_price: order.price,                          // fill price
            position_size: self.positions.short[&idx].size,   // psize after fill
            position_price: self.positions.short[&idx].price, // pprice after fill
            order_type: order.order_type.clone(),             // fill type
        });
    }

    fn calc_next_grid_entry_long(&self, k: usize, idx: usize) -> Order {
        let state_params = self.create_state_params(k, idx, LONG);
        let binding = Position::default();
        let position = self.positions.long.get(&idx).unwrap_or(&binding);
        calc_next_entry_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            position,
            &self.trailing_prices.long[&idx],
        )
    }

    fn calc_next_grid_entry_short(&self, k: usize, idx: usize) -> Order {
        let state_params = self.create_state_params(k, idx, SHORT);
        let binding = Position::default();
        let position = self.positions.short.get(&idx).unwrap_or(&binding);
        calc_next_entry_short(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.short,
            position,
            &self.trailing_prices.short[&idx],
        )
    }

    fn calc_grid_close_long(&self, k: usize, idx: usize) -> Order {
        let state_params = self.create_state_params(k, idx, LONG);
        let binding = Position::default();
        let position = self.positions.long.get(&idx).unwrap_or(&binding);
        calc_next_close_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            &position,
            &self.trailing_prices.long[&idx],
        )
    }

    fn calc_grid_close_short(&self, k: usize, idx: usize) -> Order {
        let state_params = self.create_state_params(k, idx, SHORT);
        let binding = Position::default();
        let position = self.positions.short.get(&idx).unwrap_or(&binding);
        calc_next_close_short(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.short,
            &position,
            &self.trailing_prices.short[&idx],
        )
    }

    fn reset_trailing_prices(&mut self, idx: usize, pside: usize) {
        let trailing_price_bundle = if pside == LONG {
            self.trailing_prices.long.entry(idx).or_default()
        } else {
            self.trailing_prices.short.entry(idx).or_default()
        };
        *trailing_price_bundle = TrailingPriceBundle::default();
    }

    fn update_trailing_prices(&mut self, k: usize, idx: usize, pside: usize) {
        let trailing_price_bundle = if pside == LONG {
            self.trailing_prices.long.entry(idx).or_default()
        } else {
            self.trailing_prices.short.entry(idx).or_default()
        };
        if self.hlcvs[[k, idx, LOW]] < trailing_price_bundle.min_since_open {
            trailing_price_bundle.min_since_open = self.hlcvs[[k, idx, LOW]];
            trailing_price_bundle.max_since_min = self.hlcvs[[k, idx, CLOSE]];
        } else {
            trailing_price_bundle.max_since_min = trailing_price_bundle
                .max_since_min
                .max(self.hlcvs[[k, idx, HIGH]]);
        }
        if self.hlcvs[[k, idx, HIGH]] > trailing_price_bundle.max_since_open {
            trailing_price_bundle.max_since_open = self.hlcvs[[k, idx, HIGH]];
            trailing_price_bundle.min_since_max = self.hlcvs[[k, idx, CLOSE]];
        } else {
            trailing_price_bundle.min_since_max = trailing_price_bundle
                .min_since_max
                .min(self.hlcvs[[k, idx, LOW]]);
        }
    }

    fn has_next_grid_order(&mut self, order: &Order, pside: usize) -> bool {
        match pside {
            LONG => {
                if order.qty == 0.0 {
                    false
                } else if order.qty > 0.0 {
                    order.order_type == OrderType::EntryGridNormalLong
                        || order.order_type == OrderType::EntryInitialNormalLong
                        || order.order_type == OrderType::EntryInitialPartialLong
                } else {
                    order.order_type == OrderType::CloseGridLong
                }
            }
            SHORT => {
                if order.qty == 0.0 {
                    false
                } else if order.qty < 0.0 {
                    order.order_type == OrderType::EntryGridNormalShort
                        || order.order_type == OrderType::EntryInitialNormalShort
                        || order.order_type == OrderType::EntryInitialPartialShort
                } else {
                    order.order_type == OrderType::CloseGridShort
                }
            }
            _ => panic!("Invalid pside"),
        }
    }

    fn update_open_orders_long_single(&mut self, k: usize, idx: usize) {
        let state_params = self.create_state_params(k, idx, LONG);
        let position = self
            .positions
            .long
            .get(&idx)
            .cloned()
            .unwrap_or(Position::default());

        // check if coin is delisted; if so, close pos as unstuck close
        if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
            if k >= delist_timestamp && self.positions.long.contains_key(&idx) {
                self.open_orders.long.entry(idx).or_default().closes = vec![Order {
                    qty: -self.positions.long[&idx].size,
                    price: round_(
                        f64::min(
                            self.hlcvs[[k, idx, HIGH]] - self.exchange_params_list[idx].price_step,
                            self.positions.long[&idx].price,
                        ),
                        self.exchange_params_list[idx].price_step,
                    ),
                    order_type: OrderType::CloseUnstuckLong,
                }];
                self.open_orders
                    .long
                    .entry(idx)
                    .or_default()
                    .entries
                    .clear();
                return;
            }
        }
        let next_entry_order = calc_next_entry_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            &position,
            &self.trailing_prices.long[&idx],
        );
        // if initial entry or grid, peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_entry_order)
            && self.has_next_grid_order(&next_entry_order, LONG)
        {
            self.open_orders.long.entry(idx).or_default().entries = calc_entries_long(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.long,
                &position,
                &self.trailing_prices.long[&idx],
            );
        } else {
            self.open_orders.long.entry(idx).or_default().entries = [next_entry_order].to_vec();
        }
        let next_close_order = calc_next_close_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            &position,
            &self.trailing_prices.long[&idx],
        );
        // if initial entry or grid, peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_close_order)
            && self.has_next_grid_order(&next_close_order, LONG)
        {
            self.open_orders.long.entry(idx).or_default().closes = calc_closes_long(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.long,
                &position,
                &self.trailing_prices.long[&idx],
            );
        } else {
            self.open_orders.long.entry(idx).or_default().closes = [next_close_order].to_vec();
        }
    }

    fn update_open_orders_short_single(&mut self, k: usize, idx: usize) {
        let state_params = self.create_state_params(k, idx, SHORT);
        let position = self
            .positions
            .short
            .get(&idx)
            .cloned()
            .unwrap_or(Position::default());

        // check if coin is delisted; if so, close pos as unstuck close
        if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
            if k >= delist_timestamp && self.positions.short.contains_key(&idx) {
                self.open_orders.short.entry(idx).or_default().closes = vec![Order {
                    qty: self.positions.short[&idx].size.abs(),
                    price: round_(
                        f64::max(
                            self.hlcvs[[k, idx, LOW]] + self.exchange_params_list[idx].price_step,
                            self.positions.short[&idx].price,
                        ),
                        self.exchange_params_list[idx].price_step,
                    ),
                    order_type: OrderType::CloseUnstuckShort,
                }];
                self.open_orders
                    .short
                    .entry(idx)
                    .or_default()
                    .entries
                    .clear();
                return;
            }
        }
        let next_entry_order = calc_next_entry_short(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.short,
            &position,
            &self.trailing_prices.short[&idx],
        );
        // if initial entry or grid, peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_entry_order)
            && self.has_next_grid_order(&next_entry_order, SHORT)
        {
            self.open_orders.short.entry(idx).or_default().entries = calc_entries_short(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.short,
                &position,
                &self.trailing_prices.short[&idx],
            );
        } else {
            self.open_orders.short.entry(idx).or_default().entries = [next_entry_order].to_vec();
        }

        let next_close_order = calc_next_close_short(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.short,
            &position,
            &self.trailing_prices.short[&idx],
        );
        // if initial entry or grid, peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_close_order)
            && self.has_next_grid_order(&next_close_order, SHORT)
        {
            self.open_orders.short.entry(idx).or_default().closes = calc_closes_short(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.short,
                &position,
                &self.trailing_prices.short[&idx],
            );
        } else {
            self.open_orders.short.entry(idx).or_default().closes = [next_close_order].to_vec()
        }
    }

    fn order_filled(&self, k: usize, idx: usize, order: &Order) -> bool {
        // check if will fill in next candle
        if order.qty > 0.0 {
            self.hlcvs[[k, idx, LOW]] < order.price
        } else if order.qty < 0.0 {
            self.hlcvs[[k, idx, HIGH]] > order.price
        } else {
            false
        }
    }

    fn calc_unstucking_close(&mut self, k: usize) -> (usize, usize, Order) {
        let mut stuck_positions = Vec::new();
        let mut unstuck_allowances = (0.0, 0.0);

        if self.bot_params_pair.long.unstuck_loss_allowance_pct > 0.0 {
            unstuck_allowances.0 = calc_auto_unstuck_allowance(
                self.balance.usd_total_rounded,
                self.bot_params_pair.long.unstuck_loss_allowance_pct
                    * self.bot_params_pair.long.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            );
            if unstuck_allowances.0 > 0.0 {
                // Check long positions
                // Sort the keys for long
                let mut long_keys: Vec<usize> = self.positions.long.keys().cloned().collect();
                long_keys.sort();
                for idx in long_keys {
                    let position = &self.positions.long[&idx];
                    let wallet_exposure = calc_wallet_exposure(
                        self.exchange_params_list[idx].c_mult,
                        self.balance.usd_total_rounded,
                        position.size,
                        position.price,
                    );
                    if wallet_exposure / self.bot_params_pair.long.wallet_exposure_limit
                        > self.bot_params_pair.long.unstuck_threshold
                    {
                        let pprice_diff =
                            calc_pprice_diff_int(LONG, position.price, self.hlcvs[[k, idx, CLOSE]]);
                        stuck_positions.push((idx, LONG, pprice_diff));
                    }
                }
            }
        }

        if self.bot_params_pair.short.unstuck_loss_allowance_pct > 0.0 {
            unstuck_allowances.1 = calc_auto_unstuck_allowance(
                self.balance.usd_total_rounded,
                self.bot_params_pair.short.unstuck_loss_allowance_pct
                    * self.bot_params_pair.short.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            );
            if unstuck_allowances.1 > 0.0 {
                // Check short positions
                // Sort the keys for short
                let mut short_keys: Vec<usize> = self.positions.short.keys().cloned().collect();
                short_keys.sort();

                for idx in short_keys {
                    let position = &self.positions.short[&idx];
                    let wallet_exposure = calc_wallet_exposure(
                        self.exchange_params_list[idx].c_mult,
                        self.balance.usd_total_rounded,
                        position.size,
                        position.price,
                    );
                    if wallet_exposure / self.bot_params_pair.short.wallet_exposure_limit
                        > self.bot_params_pair.short.unstuck_threshold
                    {
                        let pprice_diff = calc_pprice_diff_int(
                            SHORT,
                            position.price,
                            self.hlcvs[[k, idx, CLOSE]],
                        );
                        stuck_positions.push((idx, SHORT, pprice_diff));
                    }
                }
            }
        }
        if stuck_positions.is_empty() {
            return (NO_POS, NO_POS, Order::default());
        }
        // Sort with tie-breaker: first by diff, then by idx
        stuck_positions.sort_by(|(i1, side1, d1), (i2, side2, d2)| {
            match d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => i1.cmp(i2),
                other => other,
            }
        });
        for (idx, pside, _) in stuck_positions {
            match pside {
                LONG => {
                    let close_price = f64::max(
                        self.hlcvs[[k, idx, CLOSE]],
                        round_up(
                            self.emas[idx].compute_bands(LONG).upper
                                * (1.0 + self.bot_params_pair.long.unstuck_ema_dist),
                            self.exchange_params_list[idx].price_step,
                        ),
                    );
                    if self.open_orders.long[&idx].closes.is_empty()
                        || self.open_orders.long[&idx].closes[0].qty == 0.0
                        || close_price < self.open_orders.long[&idx].closes[0].price
                    {
                        let min_entry_qty =
                            calc_min_entry_qty(close_price, &self.exchange_params_list[idx]);
                        let mut close_qty = -f64::min(
                            self.positions.long[&idx].size,
                            f64::max(
                                min_entry_qty,
                                round_dn(
                                    cost_to_qty(
                                        self.balance.usd_total_rounded
                                            * self.bot_params_pair.long.wallet_exposure_limit
                                            * self.bot_params_pair.long.unstuck_close_pct,
                                        close_price,
                                        self.exchange_params_list[idx].c_mult,
                                    ),
                                    self.exchange_params_list[idx].qty_step,
                                ),
                            ),
                        );
                        if close_qty != 0.0 {
                            let pnl_if_closed = calc_pnl_long(
                                self.positions.long[&idx].price,
                                close_price,
                                close_qty,
                                self.exchange_params_list[idx].c_mult,
                            );
                            let pnl_if_closed_abs = pnl_if_closed.abs();
                            if pnl_if_closed < 0.0 && pnl_if_closed_abs > unstuck_allowances.0 {
                                // means unstuck allowance would be exceeded
                                // reduce qty
                                close_qty = -f64::min(
                                    self.positions.long[&idx].size,
                                    f64::max(
                                        min_entry_qty,
                                        round_dn(
                                            close_qty.abs()
                                                * (unstuck_allowances.0 / pnl_if_closed_abs),
                                            self.exchange_params_list[idx].qty_step,
                                        ),
                                    ),
                                );
                            }
                            return (
                                idx,
                                LONG,
                                Order {
                                    qty: close_qty,
                                    price: close_price,
                                    order_type: OrderType::CloseUnstuckLong,
                                },
                            );
                        }
                    }
                }
                SHORT => {
                    let close_price = f64::min(
                        self.hlcvs[[k, idx, CLOSE]],
                        round_dn(
                            self.emas[idx].compute_bands(SHORT).lower
                                * (1.0 - self.bot_params_pair.short.unstuck_ema_dist),
                            self.exchange_params_list[idx].price_step,
                        ),
                    );
                    if self.open_orders.short[&idx].closes.is_empty()
                        || self.open_orders.short[&idx].closes[0].qty == 0.0
                        || close_price > self.open_orders.short[&idx].closes[0].price
                    {
                        let min_entry_qty =
                            calc_min_entry_qty(close_price, &self.exchange_params_list[idx]);
                        let mut close_qty = f64::min(
                            self.positions.short[&idx].size.abs(),
                            f64::max(
                                min_entry_qty,
                                round_dn(
                                    cost_to_qty(
                                        self.balance.usd_total_rounded
                                            * self.bot_params_pair.short.wallet_exposure_limit
                                            * self.bot_params_pair.short.unstuck_close_pct,
                                        close_price,
                                        self.exchange_params_list[idx].c_mult,
                                    ),
                                    self.exchange_params_list[idx].qty_step,
                                ),
                            ),
                        );
                        if close_qty != 0.0 {
                            let pnl_if_closed = calc_pnl_short(
                                self.positions.short[&idx].price,
                                close_price,
                                close_qty,
                                self.exchange_params_list[idx].c_mult,
                            );
                            let pnl_if_closed_abs = pnl_if_closed.abs();
                            if pnl_if_closed < 0.0 && pnl_if_closed_abs > unstuck_allowances.1 {
                                // means unstuck allowance would be exceeded
                                // reduce qty
                                close_qty = f64::min(
                                    self.positions.short[&idx].size.abs(),
                                    f64::max(
                                        min_entry_qty,
                                        round_dn(
                                            close_qty * (unstuck_allowances.1 / pnl_if_closed_abs),
                                            self.exchange_params_list[idx].qty_step,
                                        ),
                                    ),
                                );
                            }
                            return (
                                idx,
                                SHORT,
                                Order {
                                    qty: close_qty,
                                    price: close_price,
                                    order_type: OrderType::CloseUnstuckShort,
                                },
                            );
                        }
                    }
                }
                _ => panic!("Invalid pside"),
            };
        }

        (NO_POS, NO_POS, Order::default())
    }

    fn update_open_orders_any_fill(&mut self, k: usize) {
        if self.trading_enabled.long {
            if self.trailing_enabled.long {
                let mut positions_long_indices: Vec<usize> =
                    self.positions.long.keys().cloned().collect();
                positions_long_indices.sort();
                for idx in &positions_long_indices {
                    if !self.did_fill_long.contains(&idx) {
                        self.update_trailing_prices(k, *idx, LONG);
                    }
                }
            }
            self.update_actives(k, LONG);
            self.open_orders
                .long
                .retain(|&idx, _| self.actives.long.contains(&idx));
            let mut active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
            active_long_indices.sort(); // Ensure deterministic order
            for &idx in &active_long_indices {
                self.update_stuck_status(idx, LONG);
                self.update_open_orders_long_single(k, idx);
            }
        }
        if self.trading_enabled.short {
            if self.trailing_enabled.short {
                let mut positions_short_indices: Vec<usize> =
                    self.positions.short.keys().cloned().collect();
                positions_short_indices.sort();
                for idx in &positions_short_indices {
                    if !self.did_fill_short.contains(&idx) {
                        self.update_trailing_prices(k, *idx, SHORT);
                    }
                }
            }
            self.update_actives(k, SHORT);
            self.open_orders
                .short
                .retain(|&idx, _| self.actives.short.contains(&idx));
            let mut active_short_indices: Vec<usize> = self.actives.short.iter().cloned().collect();
            active_short_indices.sort(); // Ensure deterministic order
            for &idx in &active_short_indices {
                self.update_stuck_status(idx, SHORT);
                self.update_open_orders_short_single(k, idx);
            }
        }
        let (unstucking_idx, unstucking_pside, unstucking_close) = self.calc_unstucking_close(k);
        if unstucking_pside != NO_POS {
            match unstucking_pside {
                LONG => {
                    self.open_orders
                        .long
                        .entry(unstucking_idx)
                        .or_default()
                        .closes = vec![unstucking_close];
                }
                SHORT => {
                    self.open_orders
                        .short
                        .entry(unstucking_idx)
                        .or_default()
                        .closes = vec![unstucking_close];
                }
                _ => unreachable!(),
            }
        }
    }

    fn update_open_orders_no_fill(&mut self, k: usize) {
        // Update selectively:
        // - actives if len(positions) < n_positions
        // - unstuck close if any stuck
        // - entries for coins with open trailing entries
        // - closes for coins with open trailing closes
        if self.trading_enabled.long {
            let mut positions_long_indices: Vec<usize> =
                self.positions.long.keys().cloned().collect();
            positions_long_indices.sort();
            if self.trailing_enabled.long {
                for idx in &positions_long_indices {
                    if !self.did_fill_long.contains(idx) {
                        self.update_trailing_prices(k, *idx, LONG);
                    }
                }
            }
            let mut actives_without_pos = Vec::<usize>::new();
            if positions_long_indices.len() < self.bot_params_pair.long.n_positions {
                actives_without_pos = self.update_actives(k, LONG);
                self.open_orders
                    .long
                    .retain(|&idx, _| self.actives.long.contains(&idx));
            }
            let mut active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
            active_long_indices.sort();

            for idx in active_long_indices {
                if actives_without_pos.contains(&idx)
                    || self.open_orders.long.get(&idx).map_or(false, |orders| {
                        orders.closes.iter().any(|order| {
                            order.order_type == OrderType::CloseUnstuckLong
                                || order.order_type == OrderType::CloseTrailingLong
                        }) || orders.entries.iter().any(|order| {
                            order.order_type == OrderType::EntryTrailingNormalLong
                                || order.order_type == OrderType::EntryTrailingCroppedLong
                        })
                    })
                {
                    self.update_open_orders_long_single(k, idx);
                }
            }
        }

        if self.trading_enabled.short {
            let mut positions_short_indices: Vec<usize> =
                self.positions.short.keys().cloned().collect();
            positions_short_indices.sort();
            if self.trailing_enabled.short {
                for idx in &positions_short_indices {
                    if !self.did_fill_short.contains(idx) {
                        self.update_trailing_prices(k, *idx, SHORT);
                    }
                }
            }
            let mut actives_without_pos = Vec::<usize>::new();
            if positions_short_indices.len() < self.bot_params_pair.short.n_positions {
                actives_without_pos = self.update_actives(k, SHORT);
                self.open_orders
                    .short
                    .retain(|&idx, _| self.actives.short.contains(&idx));
            }
            let mut active_short_indices: Vec<usize> = self.actives.short.iter().cloned().collect();
            active_short_indices.sort();
            for idx in active_short_indices {
                if actives_without_pos.contains(&idx)
                    || self.open_orders.short.get(&idx).map_or(false, |orders| {
                        orders.closes.iter().any(|order| {
                            order.order_type == OrderType::CloseUnstuckShort
                                || order.order_type == OrderType::CloseTrailingShort
                        }) || orders.entries.iter().any(|order| {
                            order.order_type == OrderType::EntryTrailingNormalShort
                                || order.order_type == OrderType::EntryTrailingCroppedShort
                        })
                    })
                {
                    self.update_open_orders_short_single(k, idx);
                }
            }
        }

        if !self.is_stuck.long.is_empty() || !self.is_stuck.short.is_empty() {
            let (unstucking_idx, unstucking_pside, unstucking_close) =
                self.calc_unstucking_close(k);
            if unstucking_pside != NO_POS {
                match unstucking_pside {
                    LONG => {
                        if let Some(orders) = self.open_orders.long.get_mut(&unstucking_idx) {
                            orders.closes = vec![unstucking_close];
                        }
                    }
                    SHORT => {
                        if let Some(orders) = self.open_orders.short.get_mut(&unstucking_idx) {
                            orders.closes = vec![unstucking_close];
                        }
                    }
                    _ => panic!("Invalid unstucking_pside"),
                }
            }
        }
    }

    #[inline]
    fn update_emas(&mut self, k: usize) {
        for i in 0..self.n_coins {
            let close_price = self.hlcvs[[k, i, CLOSE]];

            let long_alphas = &self.ema_alphas.long.alphas;
            let long_alphas_inv = &self.ema_alphas.long.alphas_inv;
            let short_alphas = &self.ema_alphas.short.alphas;
            let short_alphas_inv = &self.ema_alphas.short.alphas_inv;

            let emas = &mut self.emas[i];

            for z in 0..3 {
                emas.long[z] = close_price * long_alphas[z] + emas.long[z] * long_alphas_inv[z];
                emas.short[z] = close_price * short_alphas[z] + emas.short[z] * short_alphas_inv[z];
            }
        }
    }
}

/// Binary-search the **first** and **last** valid candle index for every coin.
/// A candle is *invalid* when `high == low == close` **and** `volume <= 0.0`
/// (volume is -1.0 in new data, 0.0 in older back/front-filled data).
fn find_valid_timestamp_bounds(hlcvs: &ArrayView3<f64>) -> (Vec<usize>, Vec<usize>) {
    let n_ts = hlcvs.shape()[0];
    let n_coins = hlcvs.shape()[1];
    let mut firsts = vec![0; n_coins];
    let mut lasts = vec![0; n_coins];

    for idx in 0..n_coins {
        // helper closure to keep the predicate in one place
        let is_invalid = |k: usize| {
            let row = hlcvs.slice(s![k, idx, ..]);
            row[HIGH] == row[LOW] && row[HIGH] == row[CLOSE] && row[VOLUME] <= 0.0
        };

        /* ---------- first valid ---------- */
        let (mut lo, mut hi) = (0usize, n_ts - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if is_invalid(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // market never became valid
        if is_invalid(lo) {
            firsts[idx] = n_ts; // or usize::MAX – choose a sentinel
            lasts[idx] = n_ts;
            continue; // next coin
        }
        firsts[idx] = lo;

        /* ---------- last valid ---------- */
        let (mut lo2, mut hi2) = (lo, n_ts - 1); // <-- start at first_valid
        while lo2 < hi2 {
            let mid = (lo2 + hi2 + 1) / 2; // bias to upper half
            if is_invalid(mid) {
                hi2 = mid - 1;
            } else {
                lo2 = mid;
            }
        }
        lasts[idx] = lo2;
    }
    (firsts, lasts)
}

fn calc_ema_alphas(bot_params_pair: &BotParamsPair) -> EmaAlphas {
    let mut ema_spans_long = [
        bot_params_pair.long.ema_span_0,
        bot_params_pair.long.ema_span_1,
        (bot_params_pair.long.ema_span_0 * bot_params_pair.long.ema_span_1).sqrt(),
    ];
    ema_spans_long.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ema_spans_short = [
        bot_params_pair.short.ema_span_0,
        bot_params_pair.short.ema_span_1,
        (bot_params_pair.short.ema_span_0 * bot_params_pair.short.ema_span_1).sqrt(),
    ];
    ema_spans_short.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ema_alphas_long = ema_spans_long.map(|x| 2.0 / (x + 1.0));
    let ema_alphas_long_inv = ema_alphas_long.map(|x| 1.0 - x);

    let ema_alphas_short = ema_spans_short.map(|x| 2.0 / (x + 1.0));
    let ema_alphas_short_inv = ema_alphas_short.map(|x| 1.0 - x);

    EmaAlphas {
        long: Alphas {
            alphas: ema_alphas_long,
            alphas_inv: ema_alphas_long_inv,
        },
        short: Alphas {
            alphas: ema_alphas_short,
            alphas_inv: ema_alphas_short_inv,
        },
    }
}

fn analyze_backtest_basic(fills: &[Fill], equities: &Vec<f64>) -> Analysis {
    if fills.len() <= 1 {
        return Analysis::default();
    }
    // Calculate daily equities
    let mut daily_eqs = Vec::new(); // stores last equity of each day
    let mut daily_eqs_mins = Vec::new(); // stores min equity of each day

    let mut current_day = 0;
    let mut current_min = equities[0];
    let mut last_equity = equities[0];

    for (i, &equity) in equities.iter().enumerate() {
        let day = i / 1440;
        if day > current_day {
            daily_eqs.push(last_equity);
            daily_eqs_mins.push(current_min);
            current_day = day;
            current_min = equity;
        } else {
            current_min = current_min.min(equity);
        }
        last_equity = equity;
    }

    // Push final day’s values
    if !equities.is_empty() {
        daily_eqs.push(last_equity);
        daily_eqs_mins.push(current_min);
    }

    // Calculate daily percentage changes
    let daily_eqs_pct_change: Vec<f64> =
        daily_eqs.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
    let daily_eqs_mins_pct_change: Vec<f64> = daily_eqs_mins
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Calculate ADG and standard metrics
    let (gain, adg) = smoothed_terminal_geometric_gain_and_adg(&daily_eqs);
    let mdg = {
        let mut sorted_pct_change = daily_eqs_pct_change.clone();
        sorted_pct_change.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or_else(|| {
                if a.is_nan() && b.is_nan() {
                    Ordering::Equal
                } else if a.is_nan() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
        });
        if sorted_pct_change.len() % 2 == 0 {
            (sorted_pct_change[sorted_pct_change.len() / 2 - 1]
                + sorted_pct_change[sorted_pct_change.len() / 2])
                / 2.0
        } else {
            sorted_pct_change[sorted_pct_change.len() / 2]
        }
    };

    // Calculate variance and standard deviation
    let variance = daily_eqs_mins_pct_change
        .iter()
        .map(|&x| (x - adg).powi(2))
        .sum::<f64>()
        / daily_eqs_mins_pct_change.len() as f64;
    let std_dev = variance.sqrt();

    // Calculate Sharpe Ratio
    let sharpe_ratio = if std_dev != 0.0 { adg / std_dev } else { 0.0 };

    // Calculate Sortino Ratio (using downside deviation)
    let downside_returns: Vec<f64> = daily_eqs_mins_pct_change
        .iter()
        .filter(|&&x| x < 0.0)
        .cloned()
        .collect();
    let downside_deviation = if !downside_returns.is_empty() {
        (downside_returns.iter().map(|x| x.powi(2)).sum::<f64>() / downside_returns.len() as f64)
            .sqrt()
    } else {
        0.0
    };
    let sortino_ratio = if downside_deviation != 0.0 {
        adg / downside_deviation
    } else {
        0.0
    };

    // Calculate Omega Ratio (threshold = 0)
    let (gains_sum, losses_sum) =
        daily_eqs_pct_change
            .iter()
            .fold((0.0, 0.0), |(gains, losses), &ret| {
                if ret >= 0.0 {
                    (gains + ret, losses)
                } else {
                    (gains, losses + ret.abs())
                }
            });
    let omega_ratio = if losses_sum != 0.0 {
        gains_sum / losses_sum
    } else {
        f64::INFINITY
    };

    // Calculate Expected Shortfall (99%)
    let expected_shortfall_1pct = {
        let mut sorted_returns = daily_eqs_mins_pct_change.clone();
        sorted_returns.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or_else(|| {
                if a.is_nan() && b.is_nan() {
                    Ordering::Equal
                } else if a.is_nan() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
        });
        let cutoff_index = (daily_eqs_mins_pct_change.len() as f64 * 0.01) as usize;
        if cutoff_index > 0 {
            sorted_returns[..cutoff_index]
                .iter()
                .map(|x| x.abs())
                .sum::<f64>()
                / cutoff_index as f64
        } else {
            sorted_returns[0].abs()
        }
    };

    // Calculate drawdowns
    let drawdowns = calc_drawdowns(&daily_eqs_mins);
    let drawdown_worst_mean_1pct = {
        let mut sorted_drawdowns = drawdowns.clone();
        sorted_drawdowns.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or_else(|| {
                if a.is_nan() && b.is_nan() {
                    Ordering::Equal
                } else if a.is_nan() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
        });
        let cutoff_index = std::cmp::max(1, (sorted_drawdowns.len() as f64 * 0.01) as usize);
        let worst_n = std::cmp::min(cutoff_index, sorted_drawdowns.len());
        sorted_drawdowns[..worst_n]
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            / worst_n as f64
    };
    let drawdown_worst = drawdowns
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b.abs()));

    // Calculate Sterling Ratio (using average of worst 1% drawdowns)
    let sterling_ratio = {
        if drawdown_worst_mean_1pct != 0.0 {
            adg / drawdown_worst_mean_1pct
        } else {
            0.0
        }
    };

    let calmar_ratio = if drawdown_worst != 0.0 {
        adg / drawdown_worst
    } else {
        0.0
    };

    // Calculate equity-balance differences
    let mut bal_eq = Vec::with_capacity(equities.len());
    let mut fill_iter = fills.iter().peekable();
    let mut last_balance = fills[0].balance_usd_total;

    for (i, &equity) in equities.iter().enumerate() {
        while let Some(fill) = fill_iter.peek() {
            if fill.index <= i {
                last_balance = fill.balance_usd_total;
                fill_iter.next();
            } else {
                break;
            }
        }
        bal_eq.push((last_balance, equity));
    }

    // Calculate equity-balance differences with separate positive and negative tracking
    let mut ebds_pos = Vec::new();
    let mut ebds_neg = Vec::new();

    for &(balance, equity) in bal_eq.iter() {
        let ebd = (equity - balance) / balance;
        if ebd > 0.0 {
            ebds_pos.push(ebd);
        } else if ebd < 0.0 {
            ebds_neg.push(ebd);
        }
    }

    let equity_balance_diff_pos_max = ebds_pos.iter().fold(0.0, |max, &x| f64::max(max, x));
    let equity_balance_diff_pos_mean = if !ebds_pos.is_empty() {
        ebds_pos.iter().sum::<f64>() / ebds_pos.len() as f64
    } else {
        0.0
    };

    let equity_balance_diff_neg_max = ebds_neg.iter().fold(0.0, |max, &x| f64::max(max, x.abs()));
    let equity_balance_diff_neg_mean = if !ebds_neg.is_empty() {
        ebds_neg.iter().map(|x| x.abs()).sum::<f64>() / ebds_neg.len() as f64
    } else {
        0.0
    };

    // Calculate profit factor
    let (total_profit, total_loss) = fills.iter().fold((0.0, 0.0), |(profit, loss), fill| {
        if fill.pnl > 0.0 {
            (profit + fill.pnl, loss)
        } else {
            (profit, loss + fill.pnl.abs())
        }
    });
    let loss_profit_ratio = if total_profit == 0.0 {
        f64::INFINITY
    } else {
        total_loss / total_profit
    };

    // Calculate position durations and position_unchanged_hours_max
    let mut positions_opened: HashMap<String, usize> = HashMap::new(); // Tracks position open time
    let mut durations: Vec<usize> = Vec::new(); // Total position durations
    let mut last_fill_time: HashMap<String, usize> = HashMap::new(); // Last fill time per position
    let mut unchanged_durations: Vec<usize> = Vec::new(); // Durations of unchanged periods

    for fill in fills {
        let side = if fill.order_type.to_string().contains("long") {
            "long"
        } else {
            "short"
        };
        let key = format!("{}_{}", fill.coin, side);

        // Record the opening time if the position is new
        if !positions_opened.contains_key(&key) {
            positions_opened.insert(key.clone(), fill.index);
            last_fill_time.insert(key.clone(), fill.index); // Initialize last fill time
        }

        // Calculate unchanged duration since the last fill
        if let Some(&last_time) = last_fill_time.get(&key) {
            let unchanged_duration = fill.index - last_time;
            unchanged_durations.push(unchanged_duration);
        }
        // Update the last fill time
        last_fill_time.insert(key.clone(), fill.index);

        // If the position is fully closed, calculate total duration and reset
        if fill.position_size == 0.0 {
            if let Some(&start_idx) = positions_opened.get(&key) {
                durations.push(fill.index - start_idx);
                positions_opened.remove(&key);
                last_fill_time.remove(&key); // Reset tracking
            }
        }
    }

    // Add unchanged durations and total durations for remaining open positions
    let last_index = fills.last().map_or(0, |f| f.index);
    for (key, &start_idx) in positions_opened.iter() {
        durations.push(last_index - start_idx); // Total duration for open positions
        if let Some(&last_time) = last_fill_time.get(key) {
            unchanged_durations.push(last_index - last_time); // Unchanged duration till end
        }
    }

    // Calculate duration statistics
    let n_days = (equities.len() as f64) / 1440.0; // Convert minutes to days
    let positions_held_per_day = durations.len() as f64 / n_days;

    let position_held_hours_mean = if !durations.is_empty() {
        durations.iter().sum::<usize>() as f64 / (durations.len() as f64 * 60.0)
    } else {
        0.0
    };

    let position_held_hours_max = if !durations.is_empty() {
        *durations.iter().max().unwrap() as f64 / 60.0
    } else {
        0.0
    };

    let position_held_hours_median = if !durations.is_empty() {
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_unstable();
        let mid = sorted_durations.len() / 2;
        if sorted_durations.len() % 2 == 0 {
            (sorted_durations[mid - 1] + sorted_durations[mid]) as f64 / (2.0 * 60.0)
        } else {
            sorted_durations[mid] as f64 / 60.0
        }
    } else {
        0.0
    };

    let position_unchanged_hours_max = if !unchanged_durations.is_empty() {
        *unchanged_durations.iter().max().unwrap() as f64 / 60.0
    } else {
        0.0
    };
    let equity_choppiness = calc_equity_choppiness(&daily_eqs);
    let equity_jerkiness = calc_equity_jerkiness(&daily_eqs);
    let exponential_fit_error = calc_exponential_fit_error(&daily_eqs);

    let volume_pct_per_day_avg = calc_avg_volume_pct_per_day(fills);

    let mut analysis = Analysis::default();
    analysis.adg = adg;
    analysis.mdg = mdg;
    analysis.gain = gain;
    analysis.sharpe_ratio = sharpe_ratio;
    analysis.sortino_ratio = sortino_ratio;
    analysis.omega_ratio = omega_ratio;
    analysis.expected_shortfall_1pct = expected_shortfall_1pct;
    analysis.calmar_ratio = calmar_ratio;
    analysis.sterling_ratio = sterling_ratio;
    analysis.drawdown_worst = drawdown_worst;
    analysis.drawdown_worst_mean_1pct = drawdown_worst_mean_1pct;
    analysis.equity_balance_diff_neg_max = equity_balance_diff_neg_max;
    analysis.equity_balance_diff_neg_mean = equity_balance_diff_neg_mean;
    analysis.equity_balance_diff_pos_max = equity_balance_diff_pos_max;
    analysis.equity_balance_diff_pos_mean = equity_balance_diff_pos_mean;
    analysis.loss_profit_ratio = loss_profit_ratio;
    analysis.positions_held_per_day = positions_held_per_day;
    analysis.position_held_hours_mean = position_held_hours_mean;
    analysis.position_held_hours_max = position_held_hours_max;
    analysis.position_held_hours_median = position_held_hours_median;
    analysis.position_unchanged_hours_max = position_unchanged_hours_max;
    analysis.equity_choppiness = equity_choppiness;
    analysis.equity_jerkiness = equity_jerkiness;
    analysis.exponential_fit_error = exponential_fit_error;
    analysis.volume_pct_per_day_avg = volume_pct_per_day_avg;

    analysis
}

pub fn analyze_backtest(fills: &[Fill], equities: &Vec<f64>) -> Analysis {
    let mut analysis = analyze_backtest_basic(fills, equities);

    if fills.len() <= 1 {
        return analysis;
    }

    let n = equities.len();
    let mut subset_analyses = Vec::with_capacity(10);
    subset_analyses.push(analysis.clone());

    for i in 1..10 {
        // fraction of the data we want to keep:
        //  i=1 => fraction = 0.5       => last half
        //  i=2 => fraction = 0.3333    => last third
        //  i=3 => fraction = 0.25      => last quarter
        //  etc.
        let fraction = 1.0 / (1.0 + i as f64);

        // start index for slicing the 'last' fraction
        let start_idx = (n as f64 - fraction * (n as f64)).round() as usize;

        // slice from start_idx to the end
        let subset_equities = &equities[start_idx..];
        if subset_equities.len() == 0 {
            break;
        }

        // filter fills that happened after or at start_idx
        let subset_fills: Vec<Fill> = fills
            .iter()
            .filter(|fill| fill.index >= start_idx)
            .cloned()
            .collect();
        if subset_fills.len() == 0 {
            break;
        }

        let subset_analysis = analyze_backtest_basic(&subset_fills, &subset_equities.to_vec());
        subset_analyses.push(subset_analysis);
    }

    // Compute weighted metrics as the mean of subset analyses
    analysis.adg_w = subset_analyses.iter().map(|a| a.adg).sum::<f64>() / 10.0;
    analysis.mdg_w = subset_analyses.iter().map(|a| a.mdg).sum::<f64>() / 10.0;
    analysis.sharpe_ratio_w = subset_analyses.iter().map(|a| a.sharpe_ratio).sum::<f64>() / 10.0;
    analysis.sortino_ratio_w = subset_analyses.iter().map(|a| a.sortino_ratio).sum::<f64>() / 10.0;
    analysis.omega_ratio_w = subset_analyses.iter().map(|a| a.omega_ratio).sum::<f64>() / 10.0;
    analysis.calmar_ratio_w = subset_analyses.iter().map(|a| a.calmar_ratio).sum::<f64>() / 10.0;
    analysis.sterling_ratio_w = subset_analyses
        .iter()
        .map(|a| a.sterling_ratio)
        .sum::<f64>()
        / 10.0;
    analysis.loss_profit_ratio_w = subset_analyses
        .iter()
        .map(|a| a.loss_profit_ratio)
        .sum::<f64>()
        / 10.0;
    analysis.equity_choppiness_w = subset_analyses
        .iter()
        .map(|a| a.equity_choppiness)
        .sum::<f64>()
        / 10.0;
    analysis.equity_jerkiness_w = subset_analyses
        .iter()
        .map(|a| a.equity_jerkiness)
        .sum::<f64>()
        / 10.0;
    analysis.exponential_fit_error_w = subset_analyses
        .iter()
        .map(|a| a.exponential_fit_error)
        .sum::<f64>()
        / 10.0;
    analysis.volume_pct_per_day_avg_w = subset_analyses
        .iter()
        .map(|a| a.volume_pct_per_day_avg)
        .sum::<f64>()
        / 10.0;
    analysis
}

/// Returns (Analysis in USD, Analysis in BTC).
/// If `balance.use_btc_collateral == false`, both are identical.
pub fn analyze_backtest_pair(
    fills: &[Fill],
    equities: &Equities,
    use_btc_collateral: bool,
) -> (Analysis, Analysis) {
    let analysis_usd = analyze_backtest(fills, &equities.usd);
    if !use_btc_collateral {
        return (analysis_usd.clone(), analysis_usd);
    }
    let mut btc_fills = fills.to_vec();
    for fill in btc_fills.iter_mut() {
        fill.balance_usd_total /= fill.btc_price; // Use actual BTC balance if available
        fill.pnl = fill.pnl / fill.btc_price; // Convert PNL to BTC
    }
    let analysis_btc = analyze_backtest(&btc_fills, &equities.btc);
    (analysis_usd, analysis_btc)
}

fn calc_drawdowns(equity_series: &[f64]) -> Vec<f64> {
    let mut cumulative_returns = vec![1.0];
    let mut cumulative_max = vec![1.0];

    for window in equity_series.windows(2) {
        let pct_change = (window[1] - window[0]) / window[0];
        let new_return = cumulative_returns.last().unwrap() * (1.0 + pct_change);
        cumulative_returns.push(new_return);
        cumulative_max.push(f64::max(*cumulative_max.last().unwrap(), new_return));
    }

    cumulative_returns
        .iter()
        .zip(cumulative_max.iter())
        .map(|(&ret, &max)| (ret - max) / max)
        .collect()
}

/// Calculates the normalized total variation (sum of absolute first differences divided by net equity gain)
pub fn calc_equity_choppiness(equity: &[f64]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let variation: f64 = equity.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let net_gain = equity.last().unwrap() - equity[0];
    if net_gain.abs() < f64::EPSILON {
        return f64::INFINITY; // Prevent division by near-zero
    }
    variation / net_gain.abs()
}

/// Calculates the normalized mean absolute second derivative
/// (each second difference is divided by the mean of the 3 equity points)
pub fn calc_equity_jerkiness(equity: &[f64]) -> f64 {
    if equity.len() < 3 {
        return 0.0;
    }
    equity
        .windows(3)
        .map(|w| {
            let numerator = (w[2] - 2.0 * w[1] + w[0]).abs();
            let denom = (w[0] + w[1] + w[2]) / 3.0;
            if denom.abs() < f64::EPSILON {
                0.0
            } else {
                numerator / denom.abs()
            }
        })
        .sum::<f64>()
        / (equity.len() - 2) as f64
}

/// Calculates the mean squared error from a log-linear (exponential) fit
pub fn calc_exponential_fit_error(equity: &[f64]) -> f64 {
    if equity.len() < 2 || equity.iter().any(|&x| x <= 0.0) {
        return f64::INFINITY;
    }

    let n = equity.len();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let log_y: Vec<f64> = equity.iter().map(|&y| y.ln()).collect();

    let sum_x = x.iter().sum::<f64>();
    let sum_y = log_y.iter().sum::<f64>();
    let sum_xx = x.iter().map(|v| v * v).sum::<f64>();
    let sum_xy = x.iter().zip(log_y.iter()).map(|(x, y)| x * y).sum::<f64>();

    let denom = (n as f64 * sum_xx - sum_x * sum_x);
    if denom == 0.0 {
        return f64::INFINITY;
    }

    let slope = (n as f64 * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n as f64;

    let mse = x
        .iter()
        .zip(log_y.iter())
        .map(|(x_i, y_i)| {
            let y_hat = slope * x_i + intercept;
            (y_hat - y_i).powi(2)
        })
        .sum::<f64>()
        / n as f64;

    mse
}

/// Applies EMA smoothing (span=3) to daily equity values and computes geometric mean growth rate
pub fn smoothed_terminal_geometric_gain_and_adg(daily_eqs: &[f64]) -> (f64, f64) {
    if daily_eqs.len() < 2 {
        return (0.0, 0.0);
    }
    if daily_eqs[0] <= 0.0 {
        return (f64::INFINITY, f64::INFINITY);
    }
    let alpha = 2.0 / (3.0 + 1.0); // span = 3 → alpha = 0.5
    let mut smoothed = Vec::with_capacity(daily_eqs.len());
    smoothed.push(daily_eqs[0]);
    for i in 1..daily_eqs.len() {
        let prev = *smoothed.last().unwrap();
        let current = alpha * daily_eqs[i] + (1.0 - alpha) * prev;
        smoothed.push(current);
    }

    let start = smoothed[0];
    let end = *smoothed.last().unwrap();
    if end <= 0.0 {
        return (-1.0, -1.0);
    }
    let n_days = daily_eqs.len() as f64;
    let gain = end / start;
    (gain, gain.powf(1.0 / n_days) - 1.0)
}

/// Calculates average volume per day as a percentage of balance.
/// For each fill: abs(qty) * price / balance_at_fill
pub fn calc_avg_volume_pct_per_day(fills: &[Fill]) -> f64 {
    if fills.is_empty() {
        return 0.0;
    }

    // Use a HashMap to sum cost_pct per day
    use std::collections::HashMap;
    let mut daily_totals: HashMap<usize, f64> = HashMap::new();

    for fill in fills {
        let day = fill.index / 1440;
        let cost_pct = (fill.fill_qty.abs() * fill.fill_price) / fill.balance_usd_total;
        *daily_totals.entry(day).or_insert(0.0) += cost_pct;
    }

    let total_days = daily_totals.len() as f64;
    if total_days == 0.0 {
        0.0
    } else {
        daily_totals.values().sum::<f64>() / total_days
    }
}
