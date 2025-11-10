use crate::types::{Analysis, Equities, Fill};
use std::cmp::Ordering;
use std::collections::HashMap;

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
        let side = if fill.order_type.is_long() {
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

pub fn analyze_backtest(fills: &[Fill], equities: &Vec<f64>, exposures_series: &[f64]) -> Analysis {
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

    let exposures: Vec<f64> = if !exposures_series.is_empty() {
        exposures_series
            .iter()
            .cloned()
            .filter(|value| value.is_finite())
            .collect()
    } else {
        fills
            .iter()
            .map(|fill| fill.total_wallet_exposure)
            .filter(|value| value.is_finite())
            .collect()
    };
    if !exposures.is_empty() {
        if let Some(max_val) = exposures
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
        {
            analysis.total_wallet_exposure_max = max_val;
        }
        analysis.total_wallet_exposure_mean =
            exposures.iter().sum::<f64>() / exposures.len() as f64;
        let mut sorted = exposures.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        analysis.total_wallet_exposure_median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
    }
    analysis
}

/// Returns (Analysis in USD, Analysis in BTC).
/// If `balance.use_btc_collateral == false`, both are identical.
pub fn analyze_backtest_pair(
    fills: &[Fill],
    equities: &Equities,
    use_btc_collateral: bool,
    total_wallet_exposures: &[f64],
) -> (Analysis, Analysis) {
    let analysis_usd = analyze_backtest(fills, &equities.usd, total_wallet_exposures);
    if !use_btc_collateral {
        return (analysis_usd.clone(), analysis_usd);
    }
    let mut btc_fills = fills.to_vec();
    for fill in btc_fills.iter_mut() {
        fill.balance_usd_total /= fill.btc_price; // Use actual BTC balance if available
        fill.pnl = fill.pnl / fill.btc_price; // Convert PNL to BTC
    }
    let analysis_btc = analyze_backtest(&btc_fills, &equities.btc, total_wallet_exposures);
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

    let denom = n as f64 * sum_xx - sum_x * sum_x;
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
