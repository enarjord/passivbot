use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::constants::{CLOSE, HIGH, LONG, LOW, NO_POS, SHORT, VOLUME};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_min_entry_qty, calc_next_entry_long,
    calc_next_entry_short,
};
use crate::types::{
    Analysis, BacktestParams, BotParams, BotParamsPair, EMABands, ExchangeParams, Fill, Order,
    OrderBook, OrderType, Position, Positions, StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, qty_to_cost, round_, round_dn,
    round_up,
};
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView3, Axis, Dim, ViewRepr};
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
    bot_params_pair: BotParamsPair,
    exchange_params_list: Vec<ExchangeParams>,
    backtest_params: BacktestParams,
    balance: f64,
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
    equities: Vec<f64>,
    delist_timestamps: HashMap<usize, usize>,
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
        bot_params_pair: BotParamsPair,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
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
        let mut equities = Vec::<f64>::new();
        equities.push(backtest_params.starting_balance);
        let mut bot_params_pair_cloned = bot_params_pair.clone();
        bot_params_pair_cloned.long.n_positions = n_coins.min(bot_params_pair.long.n_positions);
        bot_params_pair_cloned.short.n_positions = n_coins.min(bot_params_pair.short.n_positions);
        let n_eligible_long = bot_params_pair_cloned.long.n_positions.max(
            (n_coins as f64 * (1.0 - bot_params_pair.long.filter_relative_volume_clip_pct)).round()
                as usize,
        );
        let n_eligible_short = bot_params_pair_cloned.short.n_positions.max(
            (n_coins as f64 * (1.0 - bot_params_pair.short.filter_relative_volume_clip_pct)).round()
                as usize,
        );
        Backtest {
            hlcvs,
            bot_params_pair: bot_params_pair_cloned,
            exchange_params_list,
            backtest_params: backtest_params.clone(),
            balance: backtest_params.starting_balance,
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
            delist_timestamps: HashMap::new(),
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

        // Early return if all coins are already eligible
        if self.n_coins <= n_positions {
            return (0..self.n_coins).collect();
        }

        let n_eligible = match pside {
            LONG => self.n_eligible_long,
            SHORT => self.n_eligible_short,
            _ => panic!("Invalid pside"),
        };

        let window = bot_params.filter_rolling_window;
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

        // Use the pre-allocated buffer for volume indices
        let volume_indices = self.volume_indices_buffer.as_mut().unwrap();

        // Update rolling volume sums
        if k > window && k - *prev_k < window {
            // Rolling calculation
            let safe_start = (*prev_k).saturating_sub(window);
            for idx in 0..self.n_coins {
                rolling_volume_sum[idx] -=
                    self.hlcvs.slice(s![safe_start..start_k, idx, VOLUME]).sum();
                rolling_volume_sum[idx] += self.hlcvs.slice(s![*prev_k..k, idx, VOLUME]).sum();
                volume_indices[idx] = (rolling_volume_sum[idx], idx);
            }
        } else {
            // Full calculation
            for idx in 0..self.n_coins {
                rolling_volume_sum[idx] = self.hlcvs.slice(s![start_k..k, idx, VOLUME]).sum();
                volume_indices[idx] = (rolling_volume_sum[idx], idx);
            }
        }
        *prev_k = k;

        // Sort by volume in descending order
        volume_indices.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Calculate noisiness for top n_eligible coins
        let actual_n_eligible = n_eligible.min(self.n_coins);
        let mut noisinesses = Vec::with_capacity(actual_n_eligible);

        for &(_, idx) in volume_indices.iter().take(actual_n_eligible) {
            let noisiness: f64 = self
                .hlcvs
                .slice(s![start_k..k, idx, ..])
                .axis_iter(Axis(0))
                .map(|row| (row[HIGH] - row[LOW]) / row[CLOSE])
                .sum();
            noisinesses.push((noisiness, idx));
        }

        // Sort by noisiness in descending order
        noisinesses.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Return indices sorted by noisiness
        noisinesses.into_iter().map(|(_, idx)| idx).collect()
    }
    pub fn run(&mut self) -> (Vec<Fill>, Vec<f64>) {
        let check_points: Vec<usize> = (0..7).map(|i| i * 60 * 24).collect();
        let n_timesteps = self.hlcvs.shape()[0];

        for idx in 0..self.n_coins {
            self.trailing_prices
                .long
                .insert(idx, TrailingPriceBundle::default());
            self.trailing_prices
                .short
                .insert(idx, TrailingPriceBundle::default());

            // check if the coin was delisted at any point
            if n_timesteps > *check_points.last().unwrap() {
                let last_hlc_close = self.hlcvs[[n_timesteps - 1, idx, CLOSE]];
                if check_points.iter().all(|&point| {
                    self.hlcvs[[n_timesteps - 1 - point, idx, HIGH]] == last_hlc_close
                        && self.hlcvs[[n_timesteps - 1 - point, idx, LOW]] == last_hlc_close
                        && self.hlcvs[[n_timesteps - 1 - point, idx, CLOSE]] == last_hlc_close
                }) {
                    // was delisted. Find timestamp of delisting
                    let mut i = n_timesteps - check_points.last().unwrap();
                    while i > 0
                        && self.hlcvs[[i, idx, HIGH]] == last_hlc_close
                        && self.hlcvs[[i, idx, LOW]] == last_hlc_close
                        && self.hlcvs[[i, idx, CLOSE]] == last_hlc_close
                    {
                        i -= 1;
                    }
                    if i > 1 {
                        self.delist_timestamps.insert(idx, i);
                    }
                }
            }
        }
        for k in 1..(n_timesteps - 1) {
            self.check_for_fills(k);
            self.update_emas(k);
            self.update_open_orders(k);
            self.update_equities(k);
        }
        (self.fills.clone(), self.equities.clone())
    }

    fn create_state_params(&self, k: usize, idx: usize, pside: usize) -> StateParams {
        let close_price = self.hlcvs[[k, idx, CLOSE]];
        StateParams {
            balance: self.balance,
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

    fn update_equities(&mut self, k: usize) {
        let mut equity = self.balance;

        // Sort long keys
        let mut long_keys: Vec<usize> = self.positions.long.keys().cloned().collect();
        long_keys.sort();
        // Calculate unrealized PnL for each long position in sorted order
        for idx in long_keys {
            let position = &self.positions.long[&idx];
            let current_price = self.hlcvs[[k, idx, CLOSE]];
            let upnl = calc_pnl_long(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity += upnl;
        }

        // Sort short keys
        let mut short_keys: Vec<usize> = self.positions.short.keys().cloned().collect();
        short_keys.sort();
        // Calculate unrealized PnL for each short position in sorted order
        for idx in short_keys {
            let position = &self.positions.short[&idx];
            let current_price = self.hlcvs[[k, idx, CLOSE]];
            let upnl = calc_pnl_short(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity += upnl;
        }

        self.equities.push(equity);
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
                        self.balance,
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
                        self.balance,
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
        self.balance += pnl + fee_paid;

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
            balance: self.balance,                         // balance after fill
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
        self.balance += pnl + fee_paid;

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
            balance: self.balance,                         // balance after fill
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
        self.balance += fee_paid;
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
            balance: self.balance,                           // balance after fill
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
        self.balance += fee_paid;
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
            balance: self.balance,                            // balance after fill
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
        if let Some(&delist_timestamp) = self.delist_timestamps.get(&idx) {
            if k >= delist_timestamp && self.positions.long.contains_key(&idx) {
                self.open_orders.long.get_mut(&idx).unwrap().closes = [Order {
                    qty: -self.positions.long[&idx].size,
                    price: round_(
                        f64::min(
                            self.hlcvs[[k, idx, HIGH]] - self.exchange_params_list[idx].price_step,
                            self.positions.long[&idx].price,
                        ),
                        self.exchange_params_list[idx].price_step,
                    ),
                    order_type: OrderType::CloseUnstuckLong,
                }]
                .to_vec();
                self.open_orders.long.entry(idx).or_default().entries = Vec::new();
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
        if let Some(&delist_timestamp) = self.delist_timestamps.get(&idx) {
            if k >= delist_timestamp && self.positions.short.contains_key(&idx) {
                self.open_orders.short.get_mut(&idx).unwrap().closes = [Order {
                    qty: self.positions.short[&idx].size.abs(),
                    price: round_(
                        f64::max(
                            self.hlcvs[[k, idx, LOW]] + self.exchange_params_list[idx].price_step,
                            self.positions.short[&idx].price,
                        ),
                        self.exchange_params_list[idx].price_step,
                    ),
                    order_type: OrderType::CloseUnstuckLong,
                }]
                .to_vec();
                self.open_orders.short.entry(idx).or_default().entries = Vec::new();
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
                self.balance,
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
                        self.balance,
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
                self.balance,
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
                        self.balance,
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
                                        self.balance
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
                                        self.balance
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
                        .get_mut(&unstucking_idx)
                        .unwrap()
                        .closes = [unstucking_close].to_vec();
                }
                SHORT => {
                    self.open_orders
                        .short
                        .get_mut(&unstucking_idx)
                        .unwrap()
                        .closes = [unstucking_close].to_vec();
                }
                _ => panic!("Invalid unstucking_pside"),
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

    fn update_open_orders(&mut self, k: usize) {
        if (!self.did_fill_long.is_empty() || !self.did_fill_short.is_empty()) {
            self.update_open_orders_any_fill(k);
        } else {
            self.update_open_orders_no_fill(k);
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
    let mut daily_eqs = Vec::new();
    let mut current_day = 0;
    let mut current_min = equities[0];

    for (i, &equity) in equities.iter().enumerate() {
        let day = i / 1440;
        if day > current_day {
            daily_eqs.push(current_min);
            current_day = day;
            current_min = equity;
        } else {
            current_min = current_min.min(equity);
        }
    }
    if current_min != f64::INFINITY {
        daily_eqs.push(current_min);
    }

    // Calculate daily percentage changes
    let daily_eqs_pct_change: Vec<f64> =
        daily_eqs.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    // Calculate ADG and standard metrics
    let adg = daily_eqs_pct_change.iter().sum::<f64>() / daily_eqs_pct_change.len() as f64;
    let mdg = {
        let mut sorted_pct_change = daily_eqs_pct_change.clone();
        sorted_pct_change.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if sorted_pct_change.len() % 2 == 0 {
            (sorted_pct_change[sorted_pct_change.len() / 2 - 1]
                + sorted_pct_change[sorted_pct_change.len() / 2])
                / 2.0
        } else {
            sorted_pct_change[sorted_pct_change.len() / 2]
        }
    };

    // Calculate variance and standard deviation
    let variance = daily_eqs_pct_change
        .iter()
        .map(|&x| (x - adg).powi(2))
        .sum::<f64>()
        / daily_eqs_pct_change.len() as f64;
    let std_dev = variance.sqrt();

    // Calculate Sharpe Ratio
    let sharpe_ratio = if std_dev != 0.0 { adg / std_dev } else { 0.0 };

    // Calculate Sortino Ratio (using downside deviation)
    let downside_returns: Vec<f64> = daily_eqs_pct_change
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
        let mut sorted_returns = daily_eqs_pct_change.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let cutoff_index = (daily_eqs_pct_change.len() as f64 * 0.01) as usize;
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
    let drawdowns = calc_drawdowns(&daily_eqs);
    let drawdown_worst_mean_1pct = {
        let mut sorted_drawdowns = drawdowns.clone();
        sorted_drawdowns.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(Ordering::Equal));
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
    let mut last_balance = fills[0].balance;

    for (i, &equity) in equities.iter().enumerate() {
        while let Some(fill) = fill_iter.peek() {
            if fill.index <= i {
                last_balance = fill.balance;
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

    let gain = fills[fills.len() - 1].balance / fills[0].balance;

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

    // Calculate position durations
    let mut positions_opened: HashMap<String, usize> = HashMap::new();
    let mut durations: Vec<usize> = Vec::new();

    for fill in fills {
        let key = format!(
            "{}_{}",
            fill.coin,
            if fill.order_type.to_string().contains("long") {
                "long"
            } else {
                "short"
            }
        );

        if !positions_opened.contains_key(&key) {
            positions_opened.insert(key.clone(), fill.index);
        }

        if fill.position_size == 0.0 {
            if let Some(&start_idx) = positions_opened.get(&key) {
                durations.push(fill.index - start_idx);
                positions_opened.remove(&key);
            }
        }
    }

    // Add remaining open positions
    let last_index = fills.last().map_or(0, |f| f.index);
    for (_key, &start_idx) in positions_opened.iter() {
        durations.push(last_index - start_idx);
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

    analysis
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
