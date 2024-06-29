use crate::closes::{
    calc_next_close_long, calc_unstuck_close_long, calc_unstuck_close_short,
    determine_position_for_unstucking,
};
use crate::constants::{CLOSE, HIGH, LONG, LOW, NO_POS, SHORT};
use crate::entries::calc_next_entry_long;
use crate::types::{
    BacktestParams, BotParams, BotParamsPair, EMABands, ExchangeParams, Fill, Order, OrderBook,
    OrderType, Position, Positions, StateParams,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, qty_to_cost, round_, round_up,
};
use ndarray::s;
use ndarray::{Array1, Array2, Array3};
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

#[derive(Debug, Default)]
pub struct OpenOrders {
    pub long: HashMap<usize, OpenOrderBundle>,
    pub short: HashMap<usize, OpenOrderBundle>,
}

#[derive(Debug, Default)]
pub struct OpenOrderBundle {
    pub entry: Order,
    pub close: Order,
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

#[derive(Debug)]
pub struct TrailingPriceBundle {
    min_price_since_open: f64,
    max_price_since_min: f64,
    max_price_since_open: f64,
    min_price_since_max: f64,
}
impl Default for TrailingPriceBundle {
    fn default() -> Self {
        TrailingPriceBundle {
            min_price_since_open: f64::MAX,
            max_price_since_min: 0.0,
            max_price_since_open: 0.0,
            min_price_since_max: f64::MAX,
        }
    }
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

pub struct Backtest {
    hlcs: Array3<f64>,              // 3D array: (n_timesteps, n_markets, 3)
    noisiness_indices: Array2<i32>, // 2D array: (n_timesteps, n_markets)
    bot_params_pair: BotParamsPair,
    exchange_params_list: Vec<ExchangeParams>,
    backtest_params: BacktestParams,
    balance: f64,
    n_markets: usize,
    ema_alphas: EmaAlphas,
    emas: Vec<EMAs>,
    positions: Positions,
    open_orders: OpenOrders, // keys are symbol indices
    trailing_prices: TrailingPrices,
    actives: Actives,
    pnl_cumsum_running: f64,
    pnl_cumsum_max: f64,
    fills: Vec<Fill>,
    is_stuck: IsStuck,
    trailing_enabled: TrailingEnabled,
}

impl Backtest {
    pub fn new(
        hlcs: Array3<f64>,
        noisiness_indices: Array2<i32>,
        bot_params_pair: BotParamsPair,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
        let n_markets = hlcs.shape()[1];
        let initial_emas = (0..n_markets)
            .map(|i| {
                let close_price = hlcs[[0, i, CLOSE]];
                EMAs {
                    long: [close_price; 3],
                    short: [close_price; 3],
                }
            })
            .collect();
        Backtest {
            hlcs,
            noisiness_indices,
            bot_params_pair: bot_params_pair.clone(),
            exchange_params_list,
            backtest_params: backtest_params.clone(),
            balance: backtest_params.starting_balance,
            n_markets,
            ema_alphas: calc_ema_alphas(&bot_params_pair),
            emas: initial_emas,
            positions: Positions::default(),
            open_orders: OpenOrders::default(),
            trailing_prices: TrailingPrices::default(),
            actives: Actives::default(),
            pnl_cumsum_running: 0.0,
            pnl_cumsum_max: 0.0,
            fills: Vec::new(),
            is_stuck: IsStuck::default(),
            trailing_enabled: TrailingEnabled {
                long: bot_params_pair.long.close_trailing_grid_ratio != 0.0
                    || bot_params_pair.long.entry_trailing_grid_ratio != 0.0,
                short: bot_params_pair.short.close_trailing_grid_ratio != 0.0
                    || bot_params_pair.short.entry_trailing_grid_ratio != 0.0,
            },
        }
    }

    pub fn run(&mut self) -> Vec<Fill> {
        for idx in 0..self.n_markets {
            self.trailing_prices
                .long
                .insert(idx, TrailingPriceBundle::default());
            self.trailing_prices
                .short
                .insert(idx, TrailingPriceBundle::default());
        }
        for k in 1..self.hlcs.shape()[0] {
            //if k % 100000 == 0 {
            //    println!("k           {:?}", k);
            //    println!("hlcs        {:?}", self.hlcs.slice(s![k, .., ..]));
            //    println!("noise       {:?}", self.noisiness_indices.slice(s![k, ..]));
            //    println!("emas        {:?}", self.emas);
            //    println!("actvs       {:?}", self.actives);
            //    println!("oos         {:?}", self.open_orders);
            //    if let Some(last_fill) = self.fills.last() {
            //        println!("Last fill: {:?}", last_fill);
            //    } else {
            //        println!("No fills available.");
            //    }
            //}
            let any_fill = self.check_for_fills(k);
            self.update_emas(k);
            self.update_open_orders(k, any_fill);
        }
        self.fills.clone()
    }

    fn prepare_emas(&self) {
        let mut ema_spans_long = [
            self.bot_params_pair.long.ema_span_0,
            self.bot_params_pair.long.ema_span_1,
            (self.bot_params_pair.long.ema_span_0 * self.bot_params_pair.long.ema_span_1).sqrt(),
        ];
        ema_spans_long.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    fn update_actives_long(&mut self, k: usize) -> Vec<usize> {
        // Preallocate the vector with the maximum possible capacity
        let mut actives_without_pos = Vec::with_capacity(self.bot_params_pair.long.n_positions);
        // Clear the existing actives
        self.actives.long.clear();
        // Add markets with existing positions
        for &market_idx in self.positions.long.keys() {
            self.actives.long.insert(market_idx);
        }
        // Add additional markets based on noisiness_indices
        for &market_idx in self.noisiness_indices.row(k).iter() {
            let market_idx = market_idx as usize;
            if self.actives.long.len() < self.bot_params_pair.long.n_positions {
                if self.actives.long.insert(market_idx) {
                    // Only add to actives_without_pos if it's a new insertion
                    actives_without_pos.push(market_idx);
                }
            } else {
                break;
            }
        }
        actives_without_pos
    }

    fn update_actives_short(&mut self, k: usize) {
        // there are free slots
        if !self.actives.short.is_empty() {
            self.actives.short.clear();
        }
        for &market_idx in self.positions.short.keys() {
            self.actives.short.insert(market_idx);
        }
        // Adding additional markets based on noisiness_indices until reaching the limit
        for &market_idx in self.noisiness_indices.row(k).iter() {
            if self.actives.short.len() < self.bot_params_pair.short.n_positions {
                self.actives.short.insert(market_idx as usize);
            } else {
                break;
            }
        }
    }

    fn check_for_fills(&mut self, k: usize) -> bool {
        let mut any_fill = false;
        let mut open_orders_keys_long: Vec<usize> = self.open_orders.long.keys().cloned().collect();
        open_orders_keys_long.sort();
        for idx in open_orders_keys_long {
            // Process close fills
            while self.open_orders.long[&idx].close.qty != 0.0
                && self.hlcs[[k, idx, HIGH]] > self.open_orders.long[&idx].close.price
            {
                any_fill = true;
                self.reset_trailing_prices_long(idx);
                self.process_close_fill_long(k, idx);

                if !self.positions.long.contains_key(&idx) {
                    break;
                } else if self.open_orders.long[&idx].close.order_type == OrderType::CloseGridLong {
                    let next_grid_close = self.calc_grid_close_long(k, idx);
                    if next_grid_close.order_type == OrderType::CloseTrailingLong {
                        break;
                    }
                    self.open_orders.long.get_mut(&idx).unwrap().close = next_grid_close;
                } else {
                    break;
                }
            }

            // Process entry fills
            while self.open_orders.long[&idx].entry.qty != 0.0
                && self.hlcs[[k, idx, LOW]] < self.open_orders.long[&idx].entry.price
            {
                any_fill = true;
                self.reset_trailing_prices_long(idx);
                self.process_entry_fill_long(k, idx);

                if self.open_orders.long[&idx].entry.order_type == OrderType::EntryGridNormalLong
                    || self.open_orders.long[&idx].entry.order_type
                        == OrderType::EntryInitialNormalLong
                {
                    let next_grid_entry = self.calc_next_grid_entry_long(k, idx);
                    if next_grid_entry.qty == 0.0
                        || next_grid_entry.order_type == OrderType::EntryTrailingNormalLong
                        || next_grid_entry.order_type == OrderType::EntryTrailingCroppedLong
                    {
                        break;
                    }
                    self.open_orders.long.get_mut(&idx).unwrap().entry = next_grid_entry;
                } else {
                    break;
                }
            }
        }
        any_fill
    }

    fn update_stuck_status(&mut self, idx: usize) {
        if !self.positions.long.contains_key(&idx) {
            if self.is_stuck.long.contains(&idx) {
                self.is_stuck.long.remove(&idx);
            }
            return;
        }
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
        } else if self.is_stuck.long.contains(&idx) {
            self.is_stuck.long.remove(&idx);
        }
    }

    fn process_close_fill_long(&mut self, k: usize, idx: usize) {
        let mut new_psize = round_(
            self.positions.long[&idx].size + self.open_orders.long[&idx].close.qty,
            self.exchange_params_list[idx].qty_step,
        );
        if new_psize < 0.0 {
            println!("warning: close qty greater than psize long");
            println!("symbol: {}", self.backtest_params.symbols[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", self.open_orders.long[&idx].close);
            new_psize = 0.0;
            self.open_orders.long.get_mut(&idx).unwrap().close = Order::new(
                -self.positions.long[&idx].size,
                self.open_orders.long[&idx].close.price,
                self.open_orders.long[&idx].close.order_type.clone(),
            );
        }
        let fee_paid = -qty_to_cost(
            self.open_orders.long[&idx].close.qty,
            self.open_orders.long[&idx].close.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let pnl = calc_pnl_long(
            self.positions.long[&idx].price,
            self.open_orders.long[&idx].close.price,
            self.open_orders.long[&idx].close.qty,
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
            index: k,                                                         // index minute
            symbol: self.backtest_params.symbols[idx].clone(),                // symbol
            pnl,                                                              // realized pnl
            fee_paid,                                                         // fee paid
            balance: self.balance,                                            // balance after fill
            fill_qty: self.open_orders.long[&idx].close.qty,                  // fill qty
            fill_price: self.open_orders.long[&idx].close.price,              // fill price
            position_size: new_psize,                                         // psize after fill
            position_price: current_pprice,                                   // pprice after fill
            order_type: self.open_orders.long[&idx].close.order_type.clone(), // fill type
        });
    }

    fn process_entry_fill_long(&mut self, k: usize, idx: usize) {
        // long entry fill
        let fee_paid = -qty_to_cost(
            self.open_orders.long[&idx].entry.qty,
            self.open_orders.long[&idx].entry.price,
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
            self.open_orders.long[&idx].entry.qty,
            self.open_orders.long[&idx].entry.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.long.get_mut(&idx).unwrap().size = new_psize;
        self.positions.long.get_mut(&idx).unwrap().price = new_pprice;
        self.fills.push(Fill {
            index: k,                                                         // index minute
            symbol: self.backtest_params.symbols[idx].clone(),                // symbol
            pnl: 0.0,                                                         // realized pnl
            fee_paid,                                                         // fee paid
            balance: self.balance,                                            // balance after fill
            fill_qty: self.open_orders.long[&idx].entry.qty,                  // fill qty
            fill_price: self.open_orders.long[&idx].entry.price,              // fill price
            position_size: self.positions.long[&idx].size,                    // psize after fill
            position_price: self.positions.long[&idx].price,                  // pprice after fill
            order_type: self.open_orders.long[&idx].entry.order_type.clone(), // fill type
        });
    }

    fn calc_next_grid_entry_long(&self, k: usize, idx: usize) -> Order {
        let close_price = self.hlcs[[k, idx, CLOSE]];
        let state_params = StateParams {
            balance: self.balance,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: EMABands {
                upper: *self.emas[idx]
                    .long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                lower: *self.emas[idx]
                    .long
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            },
        };
        let binding = Position::default();
        let position = self.positions.long.get(&idx).unwrap_or(&binding);
        calc_next_entry_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            position,
            self.trailing_prices.long[&idx].min_price_since_open,
            self.trailing_prices.long[&idx].max_price_since_min,
        )
    }

    fn calc_grid_close_long(&self, k: usize, idx: usize) -> Order {
        let close_price = self.hlcs[[k, idx, CLOSE]];
        let state_params = StateParams {
            balance: self.balance,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: EMABands {
                upper: *self.emas[idx]
                    .long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                lower: *self.emas[idx]
                    .long
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            },
        };
        let binding = Position::default();
        let position = self.positions.long.get(&idx).unwrap_or(&binding);
        calc_next_close_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            &position,
            self.trailing_prices.long[&idx].max_price_since_open,
            self.trailing_prices.long[&idx].min_price_since_max,
        )
    }

    fn reset_trailing_prices_long(&mut self, idx: usize) {
        let trailing_price_bundle = self.trailing_prices.long.entry(idx).or_default();
        trailing_price_bundle.min_price_since_open = f64::MAX;
        trailing_price_bundle.max_price_since_min = 0.0;
        trailing_price_bundle.max_price_since_open = 0.0;
        trailing_price_bundle.min_price_since_max = f64::MAX;
    }

    fn update_trailing_prices_long(&mut self, k: usize, idx: usize) {
        let trailing_price_bundle = self.trailing_prices.long.entry(idx).or_default();
        if self.hlcs[[k, idx, LOW]] < trailing_price_bundle.min_price_since_open {
            trailing_price_bundle.min_price_since_open = self.hlcs[[k, idx, LOW]];
            trailing_price_bundle.max_price_since_min = self.hlcs[[k, idx, CLOSE]];
        } else {
            trailing_price_bundle.max_price_since_min = self.hlcs[[k, idx, HIGH]];
        }
        if self.hlcs[[k, idx, HIGH]] > trailing_price_bundle.max_price_since_open {
            trailing_price_bundle.max_price_since_open = self.hlcs[[k, idx, HIGH]];
            trailing_price_bundle.min_price_since_max = self.hlcs[[k, idx, CLOSE]];
        } else {
            trailing_price_bundle.min_price_since_max = self.hlcs[[k, idx, LOW]];
        }
    }

    fn calc_unstucking_close(&mut self, k: usize) -> (usize, usize, Order) {
        let (unstucking_idx, unstucking_pside) = determine_position_for_unstucking(
            &self.positions,
            &self.exchange_params_list,
            self.balance,
            &self.bot_params_pair,
            &self.hlcs.slice(s![k, .., ..]).to_owned(),
        );
        let mut unstucking_close = Order::default();
        if unstucking_pside == LONG {
            unstucking_close = calc_unstuck_close_long(
                &self.exchange_params_list[unstucking_idx],
                &self.bot_params_pair.long,
                &self.hlcs.slice(s![k, unstucking_idx, ..]).to_owned(),
                self.balance,
                *self.emas[unstucking_idx]
                    .long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                &self.positions.long[&(unstucking_idx)],
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            );
        } else if unstucking_pside == SHORT {
            // TODO impl short
            unstucking_close = Order::default();
        }
        (unstucking_idx, unstucking_pside, unstucking_close)
    }

    fn update_open_orders_long_single(
        &mut self,
        k: usize,
        idx: usize,
        unstucking_idx: usize,
        unstucking_pside: usize,
        unstucking_close: &Order,
    ) {
        let default_position = Position::default();
        let close_price = self.hlcs[[k, idx, CLOSE]];
        let state_params = StateParams {
            balance: self.balance,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: EMABands {
                upper: *self.emas[idx]
                    .long
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MIN),
                lower: *self.emas[idx]
                    .long
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&f64::MAX),
            },
        };
        let position = self
            .positions
            .long
            .get(&idx)
            .cloned()
            .unwrap_or(default_position);

        let order_bundle = self
            .open_orders
            .long
            .entry(idx)
            .or_insert_with(OpenOrderBundle::default);
        order_bundle.entry = calc_next_entry_long(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.long,
            &position,
            self.trailing_prices.long[&idx].min_price_since_open,
            self.trailing_prices.long[&idx].max_price_since_min,
        );
        if unstucking_idx == idx && unstucking_pside == LONG && unstucking_close.qty != 0.0 {
            order_bundle.close = unstucking_close.clone();
        } else {
            order_bundle.close = calc_next_close_long(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.long,
                &position,
                self.trailing_prices.long[&idx].max_price_since_open,
                self.trailing_prices.long[&idx].min_price_since_max,
            );
        }
    }

    #[inline]
    fn should_update_orders(
        &self,
        idx: usize,
        actives_without_pos: &[usize],
        unstucking_idx: usize,
        unstucking_pside: usize,
    ) -> bool {
        actives_without_pos.contains(&idx)
            || (unstucking_pside != NO_POS && idx == unstucking_idx)
            || self.open_orders.long[&idx].close.order_type == OrderType::CloseUnstuckLong
            || self.open_orders.long[&idx].entry.order_type == OrderType::EntryTrailingNormalLong
            || self.open_orders.long[&idx].entry.order_type == OrderType::EntryTrailingCroppedLong
            || self.open_orders.long[&idx].close.order_type == OrderType::CloseTrailingLong
    }

    fn update_open_orders(&mut self, k: usize, any_fill: bool) {
        let positions_long_indices: Vec<usize> = self.positions.long.keys().cloned().collect();
        if self.trailing_enabled.long {
            for idx in &positions_long_indices {
                self.update_trailing_prices_long(k, *idx);
            }
        }
        if any_fill {
            // update everything for all symbols
            // - update actives
            // - remove open orders from inactive symbols
            // - unstuck close
            // - entries
            // - closes
            self.update_actives_long(k);
            // remove open orders from inactive symbols
            self.open_orders
                .long
                .retain(|&idx, _| self.actives.long.contains(&idx));
            let active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
            let (unstucking_idx, unstucking_pside, unstucking_close) =
                self.calc_unstucking_close(k);
            for &idx in &active_long_indices {
                self.update_stuck_status(idx);
                self.update_open_orders_long_single(
                    k,
                    idx,
                    unstucking_idx,
                    unstucking_pside,
                    &unstucking_close,
                );
            }
        } else {
            // update selectively:
            // - actives if len(positions) < n_positions
            // - unstuck close if any stuck
            // - entries for symbols with open trailing entries
            // - closes for symbols with open trailing closes
            let mut actives_without_pos = Vec::<usize>::new();
            if positions_long_indices.len() < self.bot_params_pair.long.n_positions {
                actives_without_pos = self.update_actives_long(k);
                self.open_orders
                    .long
                    .retain(|&idx, _| self.actives.long.contains(&idx));
            }
            let active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
            //let (mut unstucking_idx, mut unstucking_pside, mut unstucking_close) =
            //    (NO_POS, NO_POS, Order::default());

            let (unstucking_idx, unstucking_pside, unstucking_close) =
                if !self.is_stuck.long.is_empty() {
                    self.calc_unstucking_close(k)
                } else {
                    (NO_POS, NO_POS, Order::default())
                };

            for idx in active_long_indices {
                if self.should_update_orders(
                    idx,
                    &actives_without_pos,
                    unstucking_idx,
                    unstucking_pside,
                ) {
                    self.update_open_orders_long_single(
                        k,
                        idx,
                        unstucking_idx,
                        unstucking_pside,
                        &unstucking_close,
                    );
                }
            }
        }
        /*
        let active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
        if k > 143 && k < 160 {
            for idx in active_long_indices {
                let binding = Position::default();
                let position = self.positions.long.get(&idx).unwrap_or(&binding);
                println!(
                    "{:?} {:?} {:?} {:?}",
                    k, position, self.backtest_params.symbols[idx], self.open_orders.long[&idx],
                );
            }
            println!(" ");
        }
        */
    }

    #[inline]
    fn update_emas(&mut self, k: usize) {
        for i in 0..self.n_markets {
            let close_price = self.hlcs[[k, i, CLOSE]];

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
