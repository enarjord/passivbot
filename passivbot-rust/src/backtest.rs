use crate::types::{BotParams, BotParamsAll, ExchangeParams, Order, Position};
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
pub struct Positions {
    pub long: HashMap<usize, Position>,
    pub short: HashMap<usize, Position>,
}

#[derive(Debug, Default)]
pub struct OpenOrders {
    pub entry: Order,
    pub close: Order,
    pub unstuck: Order,
}

#[derive(Default, Debug)]
pub struct Actives {
    long: HashSet<usize>,
    short: HashSet<usize>,
}

pub struct Backtest {
    hlcs: Array3<f64>,              // 3D array: (n_timesteps, n_markets, 3)
    noisiness_indices: Array2<i32>, // 2D array: (n_timesteps, n_markets)
    bot_params_all: BotParamsAll,
    exchange_params_list: Vec<ExchangeParams>,
    n_markets: usize,
    ema_alphas: EmaAlphas,
    emas: Vec<EMAs>,
    positions: Positions,
    open_orders: HashMap<usize, OpenOrders>, // keys are market indices
    actives: Actives,
}

impl Backtest {
    pub fn new(
        hlcs: Array3<f64>,
        noisiness_indices: Array2<i32>,
        bot_params_all: BotParamsAll,
        exchange_params_list: Vec<ExchangeParams>,
    ) -> Self {
        let n_markets = hlcs.shape()[1];
        let initial_emas = (0..n_markets)
            .map(|i| {
                let close_price = hlcs[[0, i, 2]];
                EMAs {
                    long: [close_price; 3],
                    short: [close_price; 3],
                }
            })
            .collect();
        Backtest {
            hlcs,
            noisiness_indices,
            bot_params_all: bot_params_all.clone(),
            exchange_params_list,
            n_markets,
            ema_alphas: calc_ema_alphas(&bot_params_all),
            emas: initial_emas,
            positions: Positions::default(),
            open_orders: HashMap::new(),
            actives: Actives::default(),
        }
    }

    pub fn run(&mut self) {
        for k in 1..self.hlcs.shape()[0] {
            if k == 500001 {
                println!(
                    "{} {:?} {:?} {:?} {:?} {:?} {:?}",
                    k,
                    self.hlcs.slice(s![k, .., ..]),
                    self.noisiness_indices.slice(s![k, ..]),
                    self.ema_alphas,
                    self.exchange_params_list,
                    self.bot_params_all,
                    self.actives,
                );
            }
            if k % 100000 == 0 {
                println!(
                    "{} {:?} {:?} {:?} {:?}",
                    k,
                    self.hlcs.slice(s![k, .., ..]),
                    self.noisiness_indices.slice(s![k, ..]),
                    self.emas,
                    self.actives,
                );
            }
            let any_fill = false;
            self.update_emas(k);
            self.update_actives(k);
            self.update_open_orders(k, any_fill);
        }
    }

    fn prepare_emas(&self) {
        let mut ema_spans_long = [
            self.bot_params_all.long.ema_span0,
            self.bot_params_all.long.ema_span1,
            (self.bot_params_all.long.ema_span0 * self.bot_params_all.long.ema_span1).sqrt(),
        ];
        ema_spans_long.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    fn update_actives(&mut self, k: usize) {
        if self.positions.long.len() < self.bot_params_all.long.n_positions {
            // there are free slots
            if !self.actives.long.is_empty() {
                self.actives.long.clear();
            }
            for &market_idx in self.positions.long.keys() {
                self.actives.long.insert(market_idx);
            }
            // Adding additional markets based on noisiness_indices until reaching the limit
            for &market_idx in self.noisiness_indices.row(k).iter() {
                if self.actives.long.len() < self.bot_params_all.long.n_positions {
                    self.actives.long.insert(market_idx as usize);
                } else {
                    break;
                }
            }
        }

        if self.positions.short.len() < self.bot_params_all.short.n_positions {
            // there are free slots
            if !self.actives.short.is_empty() {
                self.actives.short.clear();
            }
            for &market_idx in self.positions.short.keys() {
                self.actives.short.insert(market_idx);
            }
            // Adding additional markets based on noisiness_indices until reaching the limit
            for &market_idx in self.noisiness_indices.row(k).iter() {
                if self.actives.short.len() < self.bot_params_all.short.n_positions {
                    self.actives.short.insert(market_idx as usize);
                } else {
                    break;
                }
            }
        }
    }

    fn update_open_orders(&mut self, k: usize, any_fill: bool) {
        if any_fill {
            // update all open orders
        } else {
            // update only EMA based orders and trailing orders
        }
    }

    #[inline]
    fn update_emas(&mut self, k: usize) {
        for i in 0..self.n_markets {
            let close_price = self.hlcs[[k, i, 2]];

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

fn calc_ema_alphas(bot_params_all: &BotParamsAll) -> EmaAlphas {
    let mut ema_spans_long = [
        bot_params_all.long.ema_span0,
        bot_params_all.long.ema_span1,
        (bot_params_all.long.ema_span0 * bot_params_all.long.ema_span1).sqrt(),
    ];
    ema_spans_long.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ema_spans_short = [
        bot_params_all.short.ema_span0,
        bot_params_all.short.ema_span1,
        (bot_params_all.short.ema_span0 * bot_params_all.short.ema_span1).sqrt(),
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
