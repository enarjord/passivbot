use crate::types::{BotParams, BotParamsLongShort, ExchangeParams};
use ndarray::s;
use ndarray::{Array2, Array3};

pub struct Backtest {
    hlcs: Array3<f64>,              // 3D array: (n_timesteps, n_markets, 3)
    noisiness_indices: Array2<i32>, // 2D array: (n_timesteps, n_markets)
    bot_params_long_short: BotParamsLongShort,
}

impl Backtest {
    pub fn new(
        hlcs: Array3<f64>,
        noisiness_indices: Array2<i32>,
        bot_params_long_short: BotParamsLongShort,
    ) -> Self {
        Backtest {
            hlcs,
            noisiness_indices,
            bot_params_long_short,
        }
    }

    pub fn run(&self) {
        for k in 1..self.hlcs.shape()[0] {
            if k == 15 {
                println!(
                    "{} {:?} {:?}",
                    k,
                    self.hlcs.slice(s![k, .., ..]),
                    self.noisiness_indices.slice(s![k, ..])
                );
            } else if k == 5433 {
                println!(
                    "{} {:?} {:?}",
                    k,
                    self.hlcs.slice(s![k, .., ..]),
                    self.noisiness_indices.slice(s![k, ..])
                );
            }
        }
    }
}
