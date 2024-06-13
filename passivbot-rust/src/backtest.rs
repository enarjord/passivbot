use crate::types::{BotParams, BotParamsAll, ExchangeParams};
use ndarray::s;
use ndarray::{Array1, Array2, Array3};

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

pub struct Backtest {
    hlcs: Array3<f64>,              // 3D array: (n_timesteps, n_markets, 3)
    noisiness_indices: Array2<i32>, // 2D array: (n_timesteps, n_markets)
    bot_params_all: BotParamsAll,
    n_markets: usize,
    ema_alphas: EmaAlphas,
    emas: Vec<EMAs>,
}

impl Backtest {
    pub fn new(
        hlcs: Array3<f64>,
        noisiness_indices: Array2<i32>,
        bot_params_all: BotParamsAll,
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
            n_markets,
            ema_alphas: calc_ema_alphas(&bot_params_all),
            emas: initial_emas,
        }
    }

    pub fn run(&mut self) {
        for k in 1..self.hlcs.shape()[0] {
            if k == 15 {
                println!(
                    "{} {:?} {:?} {:?}",
                    k,
                    self.hlcs.slice(s![k, .., ..]),
                    self.noisiness_indices.slice(s![k, ..]),
                    self.ema_alphas
                );
            }
            if k % 100000 == 0 {
                println!(
                    "{} {:?} {:?} {:?}",
                    k,
                    self.hlcs.slice(s![k, .., ..]),
                    self.noisiness_indices.slice(s![k, ..]),
                    self.emas
                );
            }
            self.update_emas(k);
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

    fn update_emas(&mut self, k: usize) {
        for i in 0..self.n_markets {
            let close_price = self.hlcs[[k, i, 2]];

            let long_alphas = &self.ema_alphas.long.alphas;
            let long_alphas_inv = &self.ema_alphas.long.alphas_inv;
            let short_alphas = &self.ema_alphas.short.alphas;
            let short_alphas_inv = &self.ema_alphas.short.alphas_inv;

            self.emas[i].long[0] =
                close_price * long_alphas[0] + self.emas[i].long[0] * long_alphas_inv[0];
            self.emas[i].long[1] =
                close_price * long_alphas[1] + self.emas[i].long[1] * long_alphas_inv[1];
            self.emas[i].long[2] =
                close_price * long_alphas[2] + self.emas[i].long[2] * long_alphas_inv[2];

            self.emas[i].short[0] =
                close_price * short_alphas[0] + self.emas[i].short[0] * short_alphas_inv[0];
            self.emas[i].short[1] =
                close_price * short_alphas[1] + self.emas[i].short[1] * short_alphas_inv[1];
            self.emas[i].short[2] =
                close_price * short_alphas[2] + self.emas[i].short[2] * short_alphas_inv[2];
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
