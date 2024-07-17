use crate::closes::{calc_next_close_long, calc_next_close_short, calc_unstucking_close};
use crate::constants::{CLOSE, HIGH, LONG, LOW, NO_POS, SHORT};
use crate::entries::{calc_next_entry_long, calc_next_entry_short};
use crate::types::{
    Analysis, BacktestParams, BotParams, BotParamsPair, EMABands, ExchangeParams, Fill, Order,
    OrderBook, OrderType, Position, Positions, StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, qty_to_cost, round_, round_up,
};
use ndarray::s;
use ndarray::{Array1, Array2, Array3};
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

pub struct Backtest {
    hlcs: Array3<f64>,            // 3D array: (n_timesteps, n_markets, 3)
    preferred_coins: Array2<i32>, // 2D array: (n_timesteps, n_markets)
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
    trading_enabled: TradingEnabled,
    trailing_enabled: TrailingEnabled,
    equities: Vec<f64>,
}

impl Backtest {
    pub fn new(
        hlcs: Array3<f64>,
        preferred_coins: Array2<i32>,
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
        let mut equities = Vec::<f64>::new();
        equities.push(backtest_params.starting_balance);
        let mut bot_params_pair_cloned = bot_params_pair.clone();
        bot_params_pair_cloned.long.n_positions = n_markets.min(bot_params_pair.long.n_positions);
        bot_params_pair_cloned.short.n_positions = n_markets.min(bot_params_pair.short.n_positions);
        Backtest {
            hlcs,
            preferred_coins,
            bot_params_pair: bot_params_pair_cloned,
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
            trading_enabled: TradingEnabled {
                long: bot_params_pair.long.wallet_exposure_limit != 0.0,
                short: bot_params_pair.short.wallet_exposure_limit != 0.0,
            },
            trailing_enabled: TrailingEnabled {
                long: bot_params_pair.long.close_trailing_grid_ratio != 0.0
                    || bot_params_pair.long.entry_trailing_grid_ratio != 0.0,
                short: bot_params_pair.short.close_trailing_grid_ratio != 0.0
                    || bot_params_pair.short.entry_trailing_grid_ratio != 0.0,
            },
            equities: equities,
        }
    }

    pub fn run(&mut self) -> (Vec<Fill>, Vec<f64>) {
        for idx in 0..self.n_markets {
            self.trailing_prices
                .long
                .insert(idx, TrailingPriceBundle::default());
            self.trailing_prices
                .short
                .insert(idx, TrailingPriceBundle::default());
        }
        for k in 1..self.hlcs.shape()[0] {
            let any_fill = self.check_for_fills(k);
            self.update_emas(k);
            self.update_open_orders(k, any_fill);
            self.update_equities(k);
        }
        (self.fills.clone(), self.equities.clone())
    }

    fn update_equities(&mut self, k: usize) {
        let mut equity = self.balance;
        // Calculate unrealized PnL for long positions
        for (&idx, position) in &self.positions.long {
            let current_price = self.hlcs[[k, idx, CLOSE]];
            let upnl = calc_pnl_long(
                position.price,
                current_price,
                position.size,
                self.exchange_params_list[idx].c_mult,
            );
            equity += upnl;
        }
        // Calculate unrealized PnL for short positions
        for (&idx, position) in &self.positions.short {
            let current_price = self.hlcs[[k, idx, CLOSE]];
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
        let (actives, positions, n_positions) = match pside {
            LONG => (
                &mut self.actives.long,
                &self.positions.long,
                self.bot_params_pair.long.n_positions,
            ),
            SHORT => (
                &mut self.actives.short,
                &self.positions.short,
                self.bot_params_pair.short.n_positions,
            ),
            _ => panic!("Invalid pside"),
        };
        let mut actives_without_pos = Vec::with_capacity(n_positions);
        actives.clear();
        for &market_idx in positions.keys() {
            actives.insert(market_idx);
        }
        // Add additional markets based on preferred_coins
        for &market_idx in self.preferred_coins.row(k).iter() {
            let market_idx = market_idx as usize;
            if actives.len() < n_positions {
                if actives.insert(market_idx) {
                    // Only add to actives_without_pos if it's a new insertion
                    actives_without_pos.push(market_idx);
                }
            } else {
                break;
            }
        }
        actives_without_pos
    }

    fn check_for_fills(&mut self, k: usize) -> bool {
        let mut any_fill = false;
        if self.trading_enabled.long {
            let mut open_orders_keys_long: Vec<usize> =
                self.open_orders.long.keys().cloned().collect();
            open_orders_keys_long.sort();
            for idx in open_orders_keys_long {
                // Process close fills
                while self.open_orders.long[&idx].close.qty != 0.0
                    && self.hlcs[[k, idx, HIGH]] > self.open_orders.long[&idx].close.price
                {
                    any_fill = true;
                    self.reset_trailing_prices(idx, LONG);
                    self.process_close_fill_long(k, idx);

                    if !self.positions.long.contains_key(&idx) {
                        break;
                    } else if self.open_orders.long[&idx].close.order_type
                        == OrderType::CloseGridLong
                    {
                        let next_long_close = self.calc_grid_close_long(k, idx);
                        if next_long_close.order_type == OrderType::CloseTrailingLong {
                            break;
                        }
                        self.open_orders.long.get_mut(&idx).unwrap().close = next_long_close;
                    } else {
                        break;
                    }
                }

                // Process entry fills
                while self.open_orders.long[&idx].entry.qty != 0.0
                    && self.hlcs[[k, idx, LOW]] < self.open_orders.long[&idx].entry.price
                {
                    any_fill = true;
                    self.reset_trailing_prices(idx, LONG);
                    self.process_entry_fill_long(k, idx);

                    if self.open_orders.long[&idx].entry.order_type
                        == OrderType::EntryGridNormalLong
                        || self.open_orders.long[&idx].entry.order_type
                            == OrderType::EntryInitialNormalLong
                    {
                        let next_grid_entry = self.calc_next_grid_entry_long(k, idx);
                        if next_grid_entry.qty == 0.0
                            || next_grid_entry.price >= self.open_orders.long[&idx].entry.price
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
        }
        if self.trading_enabled.short {
            let mut open_orders_keys_short: Vec<usize> =
                self.open_orders.short.keys().cloned().collect();
            open_orders_keys_short.sort();
            for idx in open_orders_keys_short {
                // Process close fills
                while self.open_orders.short[&idx].close.qty != 0.0
                    && self.hlcs[[k, idx, LOW]] < self.open_orders.short[&idx].close.price
                {
                    any_fill = true;
                    self.reset_trailing_prices(idx, SHORT);
                    self.process_close_fill_short(k, idx);

                    if !self.positions.short.contains_key(&idx) {
                        break;
                    } else if self.open_orders.short[&idx].close.order_type
                        == OrderType::CloseGridShort
                    {
                        let next_short_close = self.calc_grid_close_short(k, idx);
                        if next_short_close.order_type == OrderType::CloseTrailingShort {
                            break;
                        }
                        self.open_orders.short.get_mut(&idx).unwrap().close = next_short_close;
                    } else {
                        break;
                    }
                }

                // Process entry fills
                while self.open_orders.short[&idx].entry.qty != 0.0
                    && self.hlcs[[k, idx, HIGH]] > self.open_orders.short[&idx].entry.price
                {
                    any_fill = true;
                    self.reset_trailing_prices(idx, SHORT);
                    self.process_entry_fill_short(k, idx);

                    if self.open_orders.short[&idx].entry.order_type
                        == OrderType::EntryGridNormalShort
                        || self.open_orders.short[&idx].entry.order_type
                            == OrderType::EntryInitialNormalShort
                    {
                        let next_grid_entry = self.calc_next_grid_entry_short(k, idx);
                        if next_grid_entry.qty == 0.0
                            || next_grid_entry.price <= self.open_orders.short[&idx].entry.price
                            || next_grid_entry.order_type == OrderType::EntryTrailingNormalShort
                            || next_grid_entry.order_type == OrderType::EntryTrailingCroppedShort
                        {
                            break;
                        }
                        self.open_orders.short.get_mut(&idx).unwrap().entry = next_grid_entry;
                    } else {
                        break;
                    }
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

    fn process_close_fill_short(&mut self, k: usize, idx: usize) {
        let mut new_psize = round_(
            self.positions.short[&idx].size + self.open_orders.short[&idx].close.qty,
            self.exchange_params_list[idx].qty_step,
        );
        if new_psize > 0.0 {
            println!("warning: close qty greater than psize short");
            println!("symbol: {}", self.backtest_params.symbols[idx]);
            println!("new_psize: {}", new_psize);
            println!("close order: {:?}", self.open_orders.short[&idx].close);
            new_psize = 0.0;
            self.open_orders.short.get_mut(&idx).unwrap().close = Order::new(
                self.positions.short[&idx].size.abs(),
                self.open_orders.short[&idx].close.price,
                self.open_orders.short[&idx].close.order_type.clone(),
            );
        }
        let fee_paid = -qty_to_cost(
            self.open_orders.short[&idx].close.qty,
            self.open_orders.short[&idx].close.price,
            self.exchange_params_list[idx].c_mult,
        ) * self.backtest_params.maker_fee;
        let pnl = calc_pnl_short(
            self.positions.short[&idx].price,
            self.open_orders.short[&idx].close.price,
            self.open_orders.short[&idx].close.qty,
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
            index: k,                                                          // index minute
            symbol: self.backtest_params.symbols[idx].clone(),                 // symbol
            pnl,                                                               // realized pnl
            fee_paid,                                                          // fee paid
            balance: self.balance,                                             // balance after fill
            fill_qty: self.open_orders.short[&idx].close.qty,                  // fill qty
            fill_price: self.open_orders.short[&idx].close.price,              // fill price
            position_size: new_psize,                                          // psize after fill
            position_price: current_pprice,                                    // pprice after fill
            order_type: self.open_orders.short[&idx].close.order_type.clone(), // fill type
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

    fn process_entry_fill_short(&mut self, k: usize, idx: usize) {
        // short entry fill
        let fee_paid = -qty_to_cost(
            self.open_orders.short[&idx].entry.qty,
            self.open_orders.short[&idx].entry.price,
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
            self.open_orders.short[&idx].entry.qty,
            self.open_orders.short[&idx].entry.price,
            self.exchange_params_list[idx].qty_step,
        );
        self.positions.short.get_mut(&idx).unwrap().size = new_psize;
        self.positions.short.get_mut(&idx).unwrap().price = new_pprice;
        self.fills.push(Fill {
            index: k,                                                          // index minute
            symbol: self.backtest_params.symbols[idx].clone(),                 // symbol
            pnl: 0.0,                                                          // realized pnl
            fee_paid,                                                          // fee paid
            balance: self.balance,                                             // balance after fill
            fill_qty: self.open_orders.short[&idx].entry.qty,                  // fill qty
            fill_price: self.open_orders.short[&idx].entry.price,              // fill price
            position_size: self.positions.short[&idx].size,                    // psize after fill
            position_price: self.positions.short[&idx].price,                  // pprice after fill
            order_type: self.open_orders.short[&idx].entry.order_type.clone(), // fill type
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
            ema_bands: self.emas[idx].compute_bands(LONG),
        };
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
        let close_price = self.hlcs[[k, idx, CLOSE]];
        let state_params = StateParams {
            balance: self.balance,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: self.emas[idx].compute_bands(SHORT),
        };
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
        let close_price = self.hlcs[[k, idx, CLOSE]];
        let state_params = StateParams {
            balance: self.balance,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: self.emas[idx].compute_bands(LONG),
        };
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
        let close_price = self.hlcs[[k, idx, CLOSE]];
        let state_params = StateParams {
            balance: self.balance,
            order_book: OrderBook {
                bid: close_price,
                ask: close_price,
            },
            ema_bands: self.emas[idx].compute_bands(SHORT),
        };
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
        if self.hlcs[[k, idx, LOW]] < trailing_price_bundle.min_price_since_open {
            trailing_price_bundle.min_price_since_open = self.hlcs[[k, idx, LOW]];
            trailing_price_bundle.max_price_since_min = self.hlcs[[k, idx, CLOSE]];
        } else {
            trailing_price_bundle.max_price_since_min = trailing_price_bundle
                .max_price_since_min
                .max(self.hlcs[[k, idx, HIGH]]);
        }
        if self.hlcs[[k, idx, HIGH]] > trailing_price_bundle.max_price_since_open {
            trailing_price_bundle.max_price_since_open = self.hlcs[[k, idx, HIGH]];
            trailing_price_bundle.min_price_since_max = self.hlcs[[k, idx, CLOSE]];
        } else {
            trailing_price_bundle.min_price_since_max = trailing_price_bundle
                .min_price_since_max
                .min(self.hlcs[[k, idx, LOW]]);
        }
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
            ema_bands: self.emas[idx].compute_bands(LONG),
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
            &self.trailing_prices.long[&idx],
        );
        if unstucking_idx == idx && unstucking_pside == LONG && unstucking_close.qty != 0.0 {
            order_bundle.close = unstucking_close.clone();
        } else {
            order_bundle.close = calc_next_close_long(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.long,
                &position,
                &self.trailing_prices.long[&idx],
            );
        }
    }

    fn update_open_orders_short_single(
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
            ema_bands: self.emas[idx].compute_bands(SHORT),
        };
        let position = self
            .positions
            .short
            .get(&idx)
            .cloned()
            .unwrap_or(default_position);
        let order_bundle = self
            .open_orders
            .short
            .entry(idx)
            .or_insert_with(OpenOrderBundle::default);
        order_bundle.entry = calc_next_entry_short(
            &self.exchange_params_list[idx],
            &state_params,
            &self.bot_params_pair.short,
            &position,
            &self.trailing_prices.short[&idx],
        );
        if unstucking_idx == idx && unstucking_pside == SHORT && unstucking_close.qty != 0.0 {
            order_bundle.close = unstucking_close.clone();
        } else {
            order_bundle.close = calc_next_close_short(
                &self.exchange_params_list[idx],
                &state_params,
                &self.bot_params_pair.short,
                &position,
                &self.trailing_prices.short[&idx],
            );
        }
    }

    #[inline]
    fn should_update_orders(
        &self,
        idx: usize,
        pside: usize,
        actives_without_pos: &[usize],
        unstucking_idx: usize,
        unstucking_pside: usize,
    ) -> bool {
        if pside == LONG {
            actives_without_pos.contains(&idx)
                || (unstucking_pside == pside && idx == unstucking_idx)
                || self.open_orders.long[&idx].close.order_type == OrderType::CloseUnstuckLong
                || self.open_orders.long[&idx].entry.order_type
                    == OrderType::EntryTrailingNormalLong
                || self.open_orders.long[&idx].entry.order_type
                    == OrderType::EntryTrailingCroppedLong
                || self.open_orders.long[&idx].close.order_type == OrderType::CloseTrailingLong
        } else {
            actives_without_pos.contains(&idx)
                || (unstucking_pside == pside && idx == unstucking_idx)
                || self.open_orders.short[&idx].close.order_type == OrderType::CloseUnstuckShort
                || self.open_orders.short[&idx].entry.order_type
                    == OrderType::EntryTrailingNormalShort
                || self.open_orders.short[&idx].entry.order_type
                    == OrderType::EntryTrailingCroppedShort
                || self.open_orders.short[&idx].close.order_type == OrderType::CloseTrailingShort
        }
    }

    fn update_open_orders_any_fill(&mut self, k: usize) {
        let (unstucking_idx, unstucking_pside, unstucking_close) = calc_unstucking_close(
            &self.positions,
            &self.exchange_params_list,
            &self.bot_params_pair,
            &self.hlcs.slice(s![k, .., ..]).to_owned(),
            self.balance,
            &self
                .emas
                .iter()
                .map(|ema| ema.compute_bands(LONG))
                .collect::<Vec<EMABands>>(),
            &self
                .emas
                .iter()
                .map(|ema| ema.compute_bands(SHORT))
                .collect::<Vec<EMABands>>(),
            self.pnl_cumsum_max,
            self.pnl_cumsum_running,
        );
        if self.trading_enabled.long {
            if self.trailing_enabled.long {
                let positions_long_indices: Vec<usize> =
                    self.positions.long.keys().cloned().collect();
                for idx in &positions_long_indices {
                    self.update_trailing_prices(k, *idx, LONG);
                }
            }
            self.update_actives(k, LONG);
            self.open_orders
                .long
                .retain(|&idx, _| self.actives.long.contains(&idx));
            let active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
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
        }
        if self.trading_enabled.short {
            if self.trailing_enabled.short {
                let positions_short_indices: Vec<usize> =
                    self.positions.short.keys().cloned().collect();
                for idx in &positions_short_indices {
                    self.update_trailing_prices(k, *idx, SHORT);
                }
            }
            self.update_actives(k, SHORT);
            self.open_orders
                .short
                .retain(|&idx, _| self.actives.short.contains(&idx));
            let active_short_indices: Vec<usize> = self.actives.short.iter().cloned().collect();
            for &idx in &active_short_indices {
                self.update_stuck_status(idx);
                self.update_open_orders_short_single(
                    k,
                    idx,
                    unstucking_idx,
                    unstucking_pside,
                    &unstucking_close,
                );
            }
        }
    }

    fn update_open_orders_no_fill(&mut self, k: usize) {
        // update selectively:
        // - actives if len(positions) < n_positions
        // - unstuck close if any stuck
        // - entries for symbols with open trailing entries
        // - closes for symbols with open trailing closes
        let (unstucking_idx, unstucking_pside, unstucking_close) =
            if !(self.is_stuck.long.is_empty() && self.is_stuck.short.is_empty()) {
                calc_unstucking_close(
                    &self.positions,
                    &self.exchange_params_list,
                    &self.bot_params_pair,
                    &self.hlcs.slice(s![k, .., ..]).to_owned(),
                    self.balance,
                    &self
                        .emas
                        .iter()
                        .map(|ema| ema.compute_bands(LONG))
                        .collect::<Vec<EMABands>>(),
                    &self
                        .emas
                        .iter()
                        .map(|ema| ema.compute_bands(SHORT))
                        .collect::<Vec<EMABands>>(),
                    self.pnl_cumsum_max,
                    self.pnl_cumsum_running,
                )
            } else {
                (NO_POS, NO_POS, Order::default())
            };
        if self.trading_enabled.long {
            let positions_long_indices: Vec<usize> = self.positions.long.keys().cloned().collect();
            if self.trailing_enabled.long {
                for idx in &positions_long_indices {
                    self.update_trailing_prices(k, *idx, LONG);
                }
            }
            let mut actives_without_pos = Vec::<usize>::new();
            if positions_long_indices.len() < self.bot_params_pair.long.n_positions {
                actives_without_pos = self.update_actives(k, LONG);
                self.open_orders
                    .long
                    .retain(|&idx, _| self.actives.long.contains(&idx));
            }
            let active_long_indices: Vec<usize> = self.actives.long.iter().cloned().collect();
            for idx in active_long_indices {
                if self.should_update_orders(
                    idx,
                    LONG,
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
        if self.trading_enabled.short {
            let positions_short_indices: Vec<usize> =
                self.positions.short.keys().cloned().collect();
            if self.trailing_enabled.short {
                for idx in &positions_short_indices {
                    self.update_trailing_prices(k, *idx, SHORT);
                }
            }
            let mut actives_without_pos = Vec::<usize>::new();
            if positions_short_indices.len() < self.bot_params_pair.short.n_positions {
                actives_without_pos = self.update_actives(k, SHORT);
                self.open_orders
                    .short
                    .retain(|&idx, _| self.actives.short.contains(&idx));
            }
            let active_short_indices: Vec<usize> = self.actives.short.iter().cloned().collect();
            for idx in active_short_indices {
                if self.should_update_orders(
                    idx,
                    SHORT,
                    &actives_without_pos,
                    unstucking_idx,
                    unstucking_pside,
                ) {
                    self.update_open_orders_short_single(
                        k,
                        idx,
                        unstucking_idx,
                        unstucking_pside,
                        &unstucking_close,
                    );
                }
            }
        }
    }

    fn update_open_orders(&mut self, k: usize, any_fill: bool) {
        if any_fill {
            self.update_open_orders_any_fill(k);
        } else {
            self.update_open_orders_no_fill(k);
        }
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

pub fn analyze_backtest(fills: &[Fill], equities: &Vec<f64>) -> Analysis {
    let n_days = (fills.last().unwrap().index as f64 - fills[0].index as f64) / 1440.0;

    // Calculate daily equities
    let mut daily_eqs = Vec::new();
    let mut current_day = 0;
    let mut sum = 0.0;
    let mut count = 0;
    for (i, &equity) in equities.iter().enumerate() {
        let day = i / 1440;
        if day > current_day {
            daily_eqs.push(sum / count as f64);
            current_day = day;
            sum = equity;
            count = 1;
        } else {
            sum += equity;
            count += 1;
        }
    }
    if count > 0 {
        daily_eqs.push(sum / count as f64);
    }

    // Calculate daily percentage changes
    let daily_eqs_pct_change: Vec<f64> =
        daily_eqs.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

    // Calculate ADG and Sharpe ratio
    let adg = daily_eqs_pct_change.iter().sum::<f64>() / daily_eqs_pct_change.len() as f64;
    let variance = daily_eqs_pct_change
        .iter()
        .map(|&x| (x - adg).powi(2))
        .sum::<f64>()
        / daily_eqs_pct_change.len() as f64;
    let sharpe_ratio = adg / variance.sqrt();

    // Calculate drawdowns
    let drawdowns = calc_drawdowns(&equities);
    let drawdown_worst = drawdowns
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b.abs()));

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
        bal_eq.push((equity, last_balance));
    }

    let (equity_balance_diff_sum, equity_balance_diff_max) =
        bal_eq
            .iter()
            .fold((0.0, 0.0), |(sum, max), &(equity, balance)| {
                let diff = (balance - equity).abs() / ((equity + balance) / 2.0);
                (sum + diff, f64::max(max, diff))
            });
    let equity_balance_diff_mean = equity_balance_diff_sum / bal_eq.len() as f64;

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

    Analysis {
        adg,
        sharpe_ratio,
        drawdown_worst,
        equity_balance_diff_mean,
        equity_balance_diff_max,
        loss_profit_ratio,
    }
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

fn precalc_preferred_coins_noisisness() {}
