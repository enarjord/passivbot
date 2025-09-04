use crate::closes::{
    calc_closes_long, calc_closes_short, calc_next_close_long, calc_next_close_short,
};
use crate::constants::{CLOSE, HIGH, LONG, LOW, NO_POS, SHORT, VOLUME};
use crate::entries::{
    calc_entries_long, calc_entries_short, calc_min_entry_qty, calc_next_entry_long,
    calc_next_entry_short,
};
use crate::types::{
    BacktestParams, Balance, BotParams, BotParamsPair, EMABands, Equities, ExchangeParams, Fill,
    Order, OrderBook, OrderType, Position, Positions, StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_auto_unstuck_allowance, calc_new_psize_pprice, calc_pnl_long, calc_pnl_short,
    calc_pprice_diff_int, calc_wallet_exposure, cost_to_qty, hysteresis_rounding, qty_to_cost,
    round_, round_dn, round_up,
};
use ndarray::{s, ArrayView1, ArrayView3, Axis};
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

#[derive(Debug, Clone)]
pub struct EffectiveNPositions {
    pub long: usize,
    pub short: usize,
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
    pub entries: Vec<Order>,
    pub closes: Vec<Order>,
}

#[derive(Default, Debug)]
pub struct Actives {
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

#[derive(Debug)]
pub struct TradingEnabled {
    long: bool,
    short: bool,
}

struct RollingSum {
    long: Vec<f64>,
    short: Vec<f64>,
    prev_k_long: usize,
    prev_k_short: usize,
}

pub struct Backtest<'a> {
    hlcvs: &'a ArrayView3<'a, f64>,
    btc_usd_prices: &'a ArrayView1<'a, f64>, // Change to ArrayView1 (1D view)
    bot_params_master: BotParamsPair,
    bot_params: Vec<BotParamsPair>,
    bot_params_original: Vec<BotParamsPair>,
    effective_n_positions: EffectiveNPositions,
    exchange_params_list: Vec<ExchangeParams>,
    backtest_params: BacktestParams,
    pub balance: Balance,
    n_coins: usize,
    ema_alphas: Vec<EmaAlphas>,
    emas: Vec<EMAs>,
    positions: Positions,
    open_orders: OpenOrders,
    trailing_prices: TrailingPrices,
    actives: Actives,
    pnl_cumsum_running: f64,
    pnl_cumsum_max: f64,
    fills: Vec<Fill>,
    trading_enabled: TradingEnabled,
    trailing_enabled: Vec<TrailingEnabled>,
    any_trailing_long: bool,
    any_trailing_short: bool,
    equities: Equities,
    last_valid_timestamps: HashMap<usize, usize>,
    first_valid_timestamps: HashMap<usize, usize>,
    did_fill_long: HashSet<usize>,
    did_fill_short: HashSet<usize>,
    n_eligible_long: usize,
    n_eligible_short: usize,
    rolling_volume_sum: RollingSum,
    volume_indices_buffer: Option<Vec<(f64, usize)>>,
}

impl<'a> Backtest<'a> {
    pub fn new(
        hlcvs: &'a ArrayView3<'a, f64>,
        btc_usd_prices: &'a ArrayView1<'a, f64>,
        bot_params: Vec<BotParamsPair>,
        exchange_params_list: Vec<ExchangeParams>,
        backtest_params: &BacktestParams,
    ) -> Self {
        // Determine if BTC collateral is used
        let mut balance = Balance::default();
        balance.use_btc_collateral = btc_usd_prices.iter().any(|&p| p != 1.0);

        // Initialize balances
        if balance.use_btc_collateral {
            balance.btc = backtest_params.starting_balance / btc_usd_prices[0];
        } else {
            balance.usd = backtest_params.starting_balance;
        }
        balance.usd_total = backtest_params.starting_balance;
        balance.usd_total_rounded = balance.usd_total;

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
        equities.btc.push(balance.btc);

        // init bot params
        let mut bot_params_master = bot_params[0].clone();
        bot_params_master.long.n_positions = n_coins.min(bot_params_master.long.n_positions);
        bot_params_master.short.n_positions = n_coins.min(bot_params_master.short.n_positions);

        // Store original bot params to preserve dynamic WEL indicators
        let bot_params_original = bot_params.clone();

        let n_eligible_long = bot_params_master.long.n_positions.max(
            (n_coins as f64 * (1.0 - bot_params_master.long.filter_volume_drop_pct)).round()
                as usize,
        );
        let n_eligible_short = bot_params_master.short.n_positions.max(
            (n_coins as f64 * (1.0 - bot_params_master.short.filter_volume_drop_pct)).round()
                as usize,
        );
        let effective_n_positions = EffectiveNPositions {
            long: n_eligible_long,
            short: n_eligible_short,
        };

        // Calculate EMA alphas for each coin
        let ema_alphas: Vec<EmaAlphas> = bot_params.iter().map(|bp| calc_ema_alphas(bp)).collect();

        let trailing_enabled: Vec<TrailingEnabled> = bot_params
            .iter()
            .map(|bp| TrailingEnabled {
                long: bp.long.close_trailing_grid_ratio != 0.0
                    || bp.long.entry_trailing_grid_ratio != 0.0,
                short: bp.short.close_trailing_grid_ratio != 0.0
                    || bp.short.entry_trailing_grid_ratio != 0.0,
            })
            .collect();
        let any_trailing_long = trailing_enabled.iter().any(|te| te.long);
        let any_trailing_short = trailing_enabled.iter().any(|te| te.short);

        Backtest {
            hlcvs,
            btc_usd_prices,
            bot_params_master: bot_params_master.clone(),
            bot_params: bot_params.clone(),
            bot_params_original,
            effective_n_positions,
            exchange_params_list,
            backtest_params: backtest_params.clone(),
            balance,
            n_coins,
            ema_alphas,
            emas: initial_emas,
            positions: Positions::default(),
            open_orders: OpenOrders::default(),
            trailing_prices: TrailingPrices::default(),
            actives: Actives::default(),
            pnl_cumsum_running: 0.0,
            pnl_cumsum_max: 0.0,
            fills: Vec::new(),
            trading_enabled: TradingEnabled {
                long: bot_params
                    .iter()
                    .any(|bp| bp.long.wallet_exposure_limit != 0.0)
                    && bot_params_master.long.n_positions > 0,
                short: bot_params
                    .iter()
                    .any(|bp| bp.short.wallet_exposure_limit != 0.0)
                    && bot_params_master.short.n_positions > 0,
            },
            trailing_enabled,
            any_trailing_long,
            any_trailing_short,
            equities: equities,
            last_valid_timestamps: HashMap::new(),
            first_valid_timestamps: HashMap::new(),
            did_fill_long: HashSet::new(),
            did_fill_short: HashSet::new(),
            n_eligible_long,
            n_eligible_short,
            rolling_volume_sum: RollingSum {
                long: vec![0.0; n_coins],
                short: vec![0.0; n_coins],
                prev_k_long: 0,
                prev_k_short: 0,
            },
            volume_indices_buffer: Some(vec![(0.0, 0); n_coins]), // Initialize here
        }
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
            self.update_rounded_balance(k);
            self.update_trailing_prices(k);
            self.update_n_positions_and_wallet_exposure_limits(k);
            self.update_open_orders_all(k);
            self.update_equities(k);
        }
        (self.fills.clone(), self.equities.clone())
    }

    fn update_n_positions_and_wallet_exposure_limits(&mut self, k: usize) {
        let last_ts = self.hlcvs.shape()[0] - 1;
        let eligible: Vec<usize> = (0..self.n_coins)
            .filter(|&idx| {
                let first = *self.first_valid_timestamps.get(&idx).unwrap_or(&0);
                let last = *self.last_valid_timestamps.get(&idx).unwrap_or(&last_ts);
                k >= first && k <= last
            })
            .collect();

        if eligible.is_empty() {
            return; // nothing tradable right now
        }

        // ---------- 2. effective position counts ----------
        self.effective_n_positions.long =
            self.bot_params_master.long.n_positions.min(eligible.len());
        self.effective_n_positions.short =
            self.bot_params_master.short.n_positions.min(eligible.len());

        // avoid division by zero (possible directly after a delisting)
        if self.effective_n_positions.long == 0 && self.effective_n_positions.short == 0 {
            return;
        }

        // ---------- 3. dynamic WELs ----------
        let dyn_wel_long = if self.effective_n_positions.long > 0 {
            self.bot_params_master.long.total_wallet_exposure_limit
                / self.effective_n_positions.long as f64
        } else {
            0.0
        };
        let dyn_wel_short = if self.effective_n_positions.short > 0 {
            self.bot_params_master.short.total_wallet_exposure_limit
                / self.effective_n_positions.short as f64
        } else {
            0.0
        };

        // ---------- 4. apply to every eligible coin ----------
        for &idx in &eligible {
            // long side
            if self.bot_params_original[idx].long.wallet_exposure_limit < 0.0 {
                self.bot_params[idx].long.wallet_exposure_limit = dyn_wel_long;
            }
            // short side
            if self.bot_params_original[idx].short.wallet_exposure_limit < 0.0 {
                self.bot_params[idx].short.wallet_exposure_limit = dyn_wel_short;
            }
        }
    }

    #[inline(always)]
    fn update_rounded_balance(&mut self, k: usize) {
        if self.balance.use_btc_collateral {
            // 1. raw, unrounded totals
            self.balance.usd_total = (self.balance.btc * self.btc_usd_prices[k]) + self.balance.usd;
            self.balance.btc_total = self.balance.usd_total / self.btc_usd_prices[k];

            // 2. apply hysteresis rounding
            self.balance.usd_total_rounded = hysteresis_rounding(
                self.balance.usd_total,
                self.balance.usd_total_rounded,
                0.02, // round size
                0.5,  // stickiness
            );
        }
    }

    #[inline(always)]
    fn bp(&self, coin_idx: usize, pside: usize) -> &BotParams {
        match pside {
            0 => &self.bot_params[coin_idx].long,
            1 => &self.bot_params[coin_idx].short,
            _ => unreachable!("invalid pside"),
        }
    }

    pub fn calc_preferred_coins(&mut self, k: usize, pside: usize) -> Vec<usize> {
        let n_positions = match pside {
            LONG => self.effective_n_positions.long,
            SHORT => self.effective_n_positions.short,
            _ => panic!("Invalid pside"),
        };

        if self.n_coins <= n_positions {
            return (0..self.n_coins).collect();
        }
        let volume_filtered = self.filter_by_relative_volume(k, pside);
        self.rank_by_noisiness(k, &volume_filtered, pside)
    }

    fn filter_by_relative_volume(&mut self, k: usize, pside: usize) -> Vec<usize> {
        let window = match pside {
            LONG => self.bot_params_master.long.filter_volume_rolling_window,
            SHORT => self.bot_params_master.short.filter_volume_rolling_window,
            _ => panic!("Invalid pside"),
        };
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
            LONG => &self.bot_params_master.long,
            SHORT => &self.bot_params_master.short,
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
                    let btc_to_add = pnl / self.btc_usd_prices[k];
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

    fn update_actives_long(&mut self, k: usize) -> Vec<usize> {
        let n_positions = self.effective_n_positions.long;

        let mut current_positions: Vec<usize> = self.positions.long.keys().cloned().collect();
        current_positions.sort();
        let preferred_coins = if current_positions.len() < n_positions {
            self.calc_preferred_coins(k, LONG)
        } else {
            Vec::new()
        };

        let actives = &mut self.actives.long;
        actives.clear();

        for &idx in &current_positions {
            actives.insert(idx);
        }

        let mut actives_without_pos = Vec::new();
        for &idx in &preferred_coins {
            if actives.len() >= n_positions {
                break;
            }
            if actives.insert(idx) {
                actives_without_pos.push(idx);
            }
        }

        actives_without_pos
    }

    fn update_actives_short(&mut self, k: usize) -> Vec<usize> {
        let n_positions = self.effective_n_positions.short;

        let mut current_positions: Vec<usize> = self.positions.short.keys().cloned().collect();
        current_positions.sort();

        let preferred_coins = if current_positions.len() < n_positions {
            self.calc_preferred_coins(k, SHORT)
        } else {
            Vec::new()
        };

        let actives = &mut self.actives.short;
        actives.clear();

        for &idx in &current_positions {
            actives.insert(idx);
        }

        let mut actives_without_pos = Vec::new();
        for &idx in &preferred_coins {
            if actives.len() >= n_positions {
                break;
            }
            if actives.insert(idx) {
                actives_without_pos.push(idx);
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
                        self.process_entry_fill_short(k, idx, &order);
                    }
                }
            }
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
            println!("bot config: {:?}", self.bp(idx, LONG));
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

    fn update_trailing_prices(&mut self, k: usize) {
        // ----- LONG side -----
        if self.trading_enabled.long && self.any_trailing_long {
            for (&idx, _) in &self.positions.long {
                if !self.trailing_enabled[idx].long {
                    continue;
                }
                let bundle = self.trailing_prices.long.entry(idx).or_default();
                if self.did_fill_long.contains(&idx) {
                    *bundle = TrailingPriceBundle::default();
                } else {
                    let low = self.hlcvs[[k, idx, LOW]];
                    let high = self.hlcvs[[k, idx, HIGH]];
                    let close = self.hlcvs[[k, idx, CLOSE]];

                    if low < bundle.min_since_open {
                        bundle.min_since_open = low;
                        bundle.max_since_min = close;
                    } else {
                        bundle.max_since_min = bundle.max_since_min.max(high);
                    }

                    if high > bundle.max_since_open {
                        bundle.max_since_open = high;
                        bundle.min_since_max = close;
                    } else {
                        bundle.min_since_max = bundle.min_since_max.min(low);
                    }
                }
            }
        }

        // ----- SHORT side -----
        if self.trading_enabled.short && self.any_trailing_short {
            for (&idx, _) in &self.positions.short {
                if !self.trailing_enabled[idx].short {
                    continue;
                }
                let bundle = self.trailing_prices.short.entry(idx).or_default();
                if self.did_fill_short.contains(&idx) {
                    *bundle = TrailingPriceBundle::default();
                } else {
                    let low = self.hlcvs[[k, idx, LOW]];
                    let high = self.hlcvs[[k, idx, HIGH]];
                    let close = self.hlcvs[[k, idx, CLOSE]];

                    if low < bundle.min_since_open {
                        bundle.min_since_open = low;
                        bundle.max_since_min = close;
                    } else {
                        bundle.max_since_min = bundle.max_since_min.max(high);
                    }

                    if high > bundle.max_since_open {
                        bundle.max_since_open = high;
                        bundle.min_since_max = close;
                    } else {
                        bundle.min_since_max = bundle.min_since_max.min(low);
                    }
                }
            }
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
        if self.positions.long.contains_key(&idx) {
            if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
                if k >= delist_timestamp {
                    self.open_orders.long.entry(idx).or_default().closes = vec![Order {
                        qty: -self.positions.long[&idx].size,
                        price: round_(
                            f64::min(
                                self.hlcvs[[k, idx, HIGH]]
                                    - self.exchange_params_list[idx].price_step,
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
        }

        let next_entry_order = calc_next_entry_long(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, LONG),
            &position,
            &self.trailing_prices.long[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_entry_order) {
            self.open_orders.long.entry(idx).or_default().entries = calc_entries_long(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, LONG),
                &position,
                &self.trailing_prices.long[&idx],
            );
        } else {
            self.open_orders.long.entry(idx).or_default().entries = [next_entry_order].to_vec();
        }
        let next_close_order = calc_next_close_long(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, LONG),
            &position,
            &self.trailing_prices.long[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_close_order) {
            // calc all orders
            self.open_orders.long.entry(idx).or_default().closes = calc_closes_long(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, LONG),
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
        if self.positions.short.contains_key(&idx) {
            if let Some(&delist_timestamp) = self.last_valid_timestamps.get(&idx) {
                if k >= delist_timestamp {
                    self.open_orders.short.entry(idx).or_default().closes = vec![Order {
                        qty: self.positions.short[&idx].size.abs(),
                        price: round_(
                            f64::max(
                                self.hlcvs[[k, idx, LOW]]
                                    + self.exchange_params_list[idx].price_step,
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
        }
        let next_entry_order = calc_next_entry_short(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, SHORT),
            &position,
            &self.trailing_prices.short[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_entry_order) {
            self.open_orders.short.entry(idx).or_default().entries = calc_entries_short(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, SHORT),
                &position,
                &self.trailing_prices.short[&idx],
            );
        } else {
            self.open_orders.short.entry(idx).or_default().entries = [next_entry_order].to_vec();
        }

        let next_close_order = calc_next_close_short(
            &self.exchange_params_list[idx],
            &state_params,
            self.bp(idx, SHORT),
            &position,
            &self.trailing_prices.short[&idx],
        );
        // peek next candle to see if order will fill
        if self.order_filled(k + 1, idx, &next_close_order) {
            self.open_orders.short.entry(idx).or_default().closes = calc_closes_short(
                &self.exchange_params_list[idx],
                &state_params,
                self.bp(idx, SHORT),
                &position,
                &self.trailing_prices.short[&idx],
            );
        } else {
            self.open_orders.short.entry(idx).or_default().closes = [next_close_order].to_vec()
        }
    }

    fn order_filled(&self, k: usize, idx: usize, order: &Order) -> bool {
        // check if filled in current candle (pass k+1 to check if will fill in next candle)
        if order.qty > 0.0 {
            self.hlcvs[[k, idx, LOW]] < order.price
        } else if order.qty < 0.0 {
            self.hlcvs[[k, idx, HIGH]] > order.price
        } else {
            false
        }
    }

    fn calc_unstucking_close(&mut self, k: usize) -> (usize, usize, Order) {
        let mut stuck_positions: Vec<(usize, usize, f64)> = Vec::new(); // (idx, pside, pprice_diff)

        // Calculate long unstuck allowance and check long positions
        let long_allowance = if self.bot_params_master.long.unstuck_loss_allowance_pct > 0.0 {
            calc_auto_unstuck_allowance(
                self.balance.usd_total_rounded,
                self.bot_params_master.long.unstuck_loss_allowance_pct
                    * self.bot_params_master.long.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            )
        } else {
            0.0
        };

        if long_allowance > 0.0 {
            for (&idx, position) in &self.positions.long {
                let wallet_exposure = calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_rounded,
                    position.size,
                    position.price,
                );

                if self.bp(idx, LONG).wallet_exposure_limit == 0.0
                    || wallet_exposure / self.bp(idx, LONG).wallet_exposure_limit
                        > self.bp(idx, LONG).unstuck_threshold
                {
                    let ema_bands = self.emas[idx].compute_bands(LONG);
                    let ema_price = round_up(
                        ema_bands.upper * (1.0 + self.bp(idx, LONG).unstuck_ema_dist),
                        self.exchange_params_list[idx].price_step,
                    );

                    let current_price = self.hlcvs[[k, idx, CLOSE]];
                    if current_price >= ema_price {
                        let pprice_diff = calc_pprice_diff_int(LONG, position.price, current_price);
                        stuck_positions.push((idx, LONG, pprice_diff));
                    }
                }
            }
        }

        // Calculate short unstuck allowance and check short positions
        let short_allowance = if self.bot_params_master.short.unstuck_loss_allowance_pct > 0.0 {
            calc_auto_unstuck_allowance(
                self.balance.usd_total_rounded,
                self.bot_params_master.short.unstuck_loss_allowance_pct
                    * self.bot_params_master.short.total_wallet_exposure_limit,
                self.pnl_cumsum_max,
                self.pnl_cumsum_running,
            )
        } else {
            0.0
        };

        if short_allowance > 0.0 {
            for (&idx, position) in &self.positions.short {
                let wallet_exposure = calc_wallet_exposure(
                    self.exchange_params_list[idx].c_mult,
                    self.balance.usd_total_rounded,
                    position.size.abs(),
                    position.price,
                );

                if self.bp(idx, SHORT).wallet_exposure_limit == 0.0
                    || wallet_exposure / self.bp(idx, SHORT).wallet_exposure_limit
                        > self.bp(idx, SHORT).unstuck_threshold
                {
                    let ema_bands = self.emas[idx].compute_bands(SHORT);
                    let ema_price = round_dn(
                        ema_bands.lower * (1.0 - self.bp(idx, SHORT).unstuck_ema_dist),
                        self.exchange_params_list[idx].price_step,
                    );

                    let current_price = self.hlcvs[[k, idx, CLOSE]];
                    if current_price <= ema_price {
                        let pprice_diff =
                            calc_pprice_diff_int(SHORT, position.price, current_price);
                        stuck_positions.push((idx, SHORT, pprice_diff));
                    }
                }
            }
        }

        if stuck_positions.is_empty() {
            return (NO_POS, NO_POS, Order::default());
        }

        // Sort by pprice_diff, then by idx for deterministic ordering
        stuck_positions.sort_by(|(i1, _, d1), (i2, _, d2)| {
            match d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => i1.cmp(i2),
                other => other,
            }
        });

        // Process stuck positions
        for (idx, pside, _pprice_diff) in stuck_positions {
            let close_price = self.hlcvs[[k, idx, CLOSE]];

            if pside == LONG {
                let min_entry_qty =
                    calc_min_entry_qty(close_price, &self.exchange_params_list[idx]);
                let mut close_qty = -f64::min(
                    self.positions.long[&idx].size,
                    f64::max(
                        min_entry_qty,
                        round_dn(
                            cost_to_qty(
                                self.balance.usd_total_rounded
                                    * self.bp(idx, LONG).wallet_exposure_limit
                                    * self.bp(idx, LONG).unstuck_close_pct,
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

                    if pnl_if_closed < 0.0 && pnl_if_closed_abs > long_allowance {
                        close_qty = -f64::min(
                            self.positions.long[&idx].size,
                            f64::max(
                                min_entry_qty,
                                round_dn(
                                    close_qty.abs() * (long_allowance / pnl_if_closed_abs),
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
            } else {
                // SHORT
                let min_entry_qty =
                    calc_min_entry_qty(close_price, &self.exchange_params_list[idx]);
                let mut close_qty = f64::min(
                    self.positions.short[&idx].size.abs(),
                    f64::max(
                        min_entry_qty,
                        round_dn(
                            cost_to_qty(
                                self.balance.usd_total_rounded
                                    * self.bp(idx, SHORT).wallet_exposure_limit
                                    * self.bp(idx, SHORT).unstuck_close_pct,
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

                    if pnl_if_closed < 0.0 && pnl_if_closed_abs > short_allowance {
                        close_qty = f64::min(
                            self.positions.short[&idx].size.abs(),
                            f64::max(
                                min_entry_qty,
                                round_dn(
                                    close_qty * (short_allowance / pnl_if_closed_abs),
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

        (NO_POS, NO_POS, Order::default())
    }

    fn update_open_orders_all(&mut self, k: usize) {
        self.open_orders = OpenOrders::default();
        if self.trading_enabled.long {
            let mut active_long_indices: Vec<usize> = self.positions.long.keys().cloned().collect();
            if self.positions.long.len() != self.effective_n_positions.long {
                self.update_actives_long(k);
                active_long_indices = self.actives.long.iter().cloned().collect();
            }
            active_long_indices.sort();
            for &idx in &active_long_indices {
                self.update_open_orders_long_single(k, idx);
            }
        }
        if self.trading_enabled.short {
            let mut active_short_indices: Vec<usize> =
                self.positions.short.keys().cloned().collect();
            if self.positions.short.len() != self.effective_n_positions.short {
                self.update_actives_short(k);
                active_short_indices = self.actives.short.iter().cloned().collect();
            }
            active_short_indices.sort();
            for &idx in &active_short_indices {
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

    #[inline]
    fn update_emas(&mut self, k: usize) {
        for i in 0..self.n_coins {
            let close_price = self.hlcvs[[k, i, CLOSE]];

            let long_alphas = &self.ema_alphas[i].long.alphas;
            let long_alphas_inv = &self.ema_alphas[i].long.alphas_inv;
            let short_alphas = &self.ema_alphas[i].short.alphas;
            let short_alphas_inv = &self.ema_alphas[i].short.alphas_inv;

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
        let is_invalid = |k: usize| hlcvs[[k, idx, VOLUME]] < 0.0;

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
