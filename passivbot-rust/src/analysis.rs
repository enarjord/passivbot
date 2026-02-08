use crate::types::{Analysis, Equities, Fill};
use std::cmp::Ordering;
use std::collections::HashMap;

const MS_PER_DAY: u64 = 86_400_000;
const MS_PER_HOUR: u64 = 3_600_000;

fn fallback_timestamp_ms(index: usize) -> u64 {
    (index as u64) * 60_000
}

fn analyze_backtest_basic(fills: &[Fill], equities: &Vec<f64>, timestamps_ms: &[u64]) -> Analysis {
    if fills.len() <= 1 {
        return Analysis::default();
    }
    // Calculate daily equities
    let mut daily_eqs = Vec::new(); // stores last equity of each day
    let mut daily_eqs_mins = Vec::new(); // stores min equity of each day

    let use_timestamps = !timestamps_ms.is_empty() && timestamps_ms.len() == equities.len();
    let mut current_day = if use_timestamps {
        (timestamps_ms[0] / MS_PER_DAY) as usize
    } else {
        0
    };
    let mut current_min = equities[0];
    let mut last_equity = equities[0];
    for (i, &equity) in equities.iter().enumerate() {
        let day = if use_timestamps {
            (timestamps_ms[i] / MS_PER_DAY) as usize
        } else {
            i / 1440
        };
        if day > current_day {
            if daily_eqs.is_empty() {
                daily_eqs.push(last_equity);
                daily_eqs_mins.push(current_min);
            } else {
                daily_eqs.push(last_equity);
                daily_eqs_mins.push(current_min);
            }
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
    let daily_eqs_pct_change: Vec<f64> = daily_eqs
        .windows(2)
        .map(|w| {
            let denom = w[0].abs().max(1e-12);
            (w[1] - w[0]) / denom
        })
        .collect();
    let daily_eqs_mins_pct_change: Vec<f64> = daily_eqs_mins
        .windows(2)
        .map(|w| {
            let denom = w[0].abs().max(1e-12);
            (w[1] - w[0]) / denom
        })
        .collect();

    // Calculate ADG and standard metrics
    let (gain, adg) = smoothed_terminal_geometric_gain_and_adg(&daily_eqs);
    let daily_pnl_ratios = calc_daily_pnl_ratios(fills);
    let adg_pnl = mean(&daily_pnl_ratios);
    let mdg_pnl = median(&daily_pnl_ratios);
    let (sharpe_ratio_pnl, sortino_ratio_pnl) = calc_sharpe_and_sortino(&daily_pnl_ratios, adg_pnl);
    let mdg = {
        if daily_eqs_pct_change.is_empty() {
            0.0
        } else {
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
        }
    };

    // Calculate variance and standard deviation
    let std_dev = if daily_eqs_mins_pct_change.is_empty() {
        0.0
    } else {
        let var = daily_eqs_mins_pct_change
            .iter()
            .map(|&x| (x - adg).powi(2))
            .sum::<f64>()
            / daily_eqs_mins_pct_change.len() as f64;
        var.sqrt()
    };

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
    let expected_shortfall_1pct = if daily_eqs_mins_pct_change.is_empty() {
        0.0
    } else {
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
        let cutoff_index = (sorted_returns.len() as f64 * 0.01).max(1.0) as usize;
        let worst_n = cutoff_index.min(sorted_returns.len());
        sorted_returns[..worst_n]
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            / worst_n as f64
    };

    // Calculate drawdowns
    let drawdowns_daily = calc_drawdowns(&daily_eqs_mins);
    let drawdown_worst_mean_1pct = if drawdowns_daily.is_empty() {
        0.0
    } else {
        let mut sorted_drawdowns = drawdowns_daily.clone();
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
        if worst_n == 0 {
            0.0
        } else {
            sorted_drawdowns[..worst_n]
                .iter()
                .map(|x| x.abs())
                .sum::<f64>()
                / worst_n as f64
        }
    };
    let drawdowns_full = calc_drawdowns(equities);
    let drawdown_worst = if drawdowns_full.is_empty() {
        0.0
    } else {
        drawdowns_full
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b.abs()))
    };

    // Calculate Sterling Ratio (using average of worst 1% drawdowns)
    let sterling_ratio = {
        let denom = drawdown_worst_mean_1pct.abs().max(1e-12);
        adg / denom
    };

    let calmar_ratio = {
        let denom = drawdown_worst.abs().max(1e-12);
        adg / denom
    };

    // Calculate equity-balance differences
    let mut bal_eq = Vec::with_capacity(equities.len());
    let mut fill_iter = fills.iter().peekable();
    let mut last_balance = fills[0].usd_total_balance;

    for (i, &equity) in equities.iter().enumerate() {
        while let Some(fill) = fill_iter.peek() {
            if fill.index <= i {
                last_balance = fill.usd_total_balance;
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
    let mut positions_opened: HashMap<String, u64> = HashMap::new(); // Tracks position open time
    let mut durations_ms: Vec<u64> = Vec::new(); // Total position durations (ms)
    let mut last_fill_time: HashMap<String, u64> = HashMap::new(); // Last fill time per position (ms)
    let mut unchanged_durations_ms: Vec<u64> = Vec::new(); // Durations of unchanged periods (ms)

    for fill in fills {
        let side = if fill.order_type.is_long() {
            "long"
        } else {
            "short"
        };
        let key = format!("{}_{}", fill.coin, side);
        let fill_ts = if fill.timestamp_ms > 0 {
            fill.timestamp_ms
        } else {
            fallback_timestamp_ms(fill.index)
        };

        // Record the opening time if the position is new
        if !positions_opened.contains_key(&key) {
            positions_opened.insert(key.clone(), fill_ts);
            last_fill_time.insert(key.clone(), fill_ts); // Initialize last fill time
        }

        // Calculate unchanged duration since the last fill
        if let Some(&last_time) = last_fill_time.get(&key) {
            let unchanged_duration = fill_ts.saturating_sub(last_time);
            unchanged_durations_ms.push(unchanged_duration);
        }
        // Update the last fill time
        last_fill_time.insert(key.clone(), fill_ts);

        // If the position is fully closed, calculate total duration and reset
        if fill.position_size == 0.0 {
            if let Some(&start_idx) = positions_opened.get(&key) {
                durations_ms.push(fill_ts.saturating_sub(start_idx));
                positions_opened.remove(&key);
                last_fill_time.remove(&key); // Reset tracking
            }
        }
    }

    // Add unchanged durations and total durations for remaining open positions
    let last_ts = fills.last().map_or(0u64, |f| {
        if f.timestamp_ms > 0 {
            f.timestamp_ms
        } else {
            fallback_timestamp_ms(f.index)
        }
    });
    for (key, &start_idx) in positions_opened.iter() {
        durations_ms.push(last_ts.saturating_sub(start_idx)); // Total duration for open positions
        if let Some(&last_time) = last_fill_time.get(key) {
            unchanged_durations_ms.push(last_ts.saturating_sub(last_time)); // Unchanged duration till end
        }
    }

    // Calculate duration statistics
    let n_days = if use_timestamps && timestamps_ms.len() >= 2 {
        let start_ts = timestamps_ms[0];
        let end_ts = *timestamps_ms.last().unwrap_or(&start_ts);
        let range_ms = end_ts.saturating_sub(start_ts) as f64;
        if range_ms > 0.0 {
            range_ms / MS_PER_DAY as f64
        } else {
            (equities.len() as f64) / 1440.0
        }
    } else {
        (equities.len() as f64) / 1440.0
    };
    let n_days = if n_days <= 0.0 { 1e-9 } else { n_days };
    let positions_held_per_day = durations_ms.len() as f64 / n_days;

    let position_held_hours_mean = if !durations_ms.is_empty() {
        durations_ms.iter().sum::<u64>() as f64
            / (durations_ms.len() as f64 * MS_PER_HOUR as f64)
    } else {
        0.0
    };

    let position_held_hours_max = if !durations_ms.is_empty() {
        *durations_ms.iter().max().unwrap() as f64 / MS_PER_HOUR as f64
    } else {
        0.0
    };

    let position_held_hours_median = if !durations_ms.is_empty() {
        let mut sorted_durations = durations_ms.clone();
        sorted_durations.sort_unstable();
        let mid = sorted_durations.len() / 2;
        if sorted_durations.len() % 2 == 0 {
            (sorted_durations[mid - 1] + sorted_durations[mid]) as f64
                / (2.0 * MS_PER_HOUR as f64)
        } else {
            sorted_durations[mid] as f64 / MS_PER_HOUR as f64
        }
    } else {
        0.0
    };

    let position_unchanged_hours_max = if !unchanged_durations_ms.is_empty() {
        *unchanged_durations_ms.iter().max().unwrap() as f64 / MS_PER_HOUR as f64
    } else {
        0.0
    };
    let equity_choppiness = calc_equity_choppiness(&daily_eqs);
    let equity_jerkiness = calc_equity_jerkiness(&daily_eqs);
    let exponential_fit_error = calc_exponential_fit_error(&daily_eqs);

    let volume_pct_per_day_avg = calc_avg_volume_pct_per_day(fills);
    let peak_recovery_hours_equity = calc_peak_recovery_hours(
        equities,
        if use_timestamps { Some(timestamps_ms) } else { None },
    );
    let peak_recovery_hours_pnl = if equities.is_empty() {
        0.0
    } else {
        let mut deltas = vec![0.0f64; equities.len()];
        for fill in fills {
            if fill.index < deltas.len() {
                deltas[fill.index] += fill.pnl + fill.fee_paid;
            }
        }
        let mut running = 0.0;
        for value in deltas.iter_mut() {
            running += *value;
            *value = running;
        }
        calc_peak_recovery_hours(&deltas, if use_timestamps { Some(timestamps_ms) } else { None })
    };

    let mut analysis = Analysis::default();
    analysis.adg = adg;
    analysis.mdg = mdg;
    analysis.gain = gain;
    analysis.adg_pnl = adg_pnl;
    analysis.mdg_pnl = mdg_pnl;
    analysis.sharpe_ratio_pnl = sharpe_ratio_pnl;
    analysis.sortino_ratio_pnl = sortino_ratio_pnl;
    analysis.mdg_pnl = mdg_pnl;
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
    analysis.peak_recovery_hours_equity = peak_recovery_hours_equity;
    analysis.peak_recovery_hours_pnl = peak_recovery_hours_pnl;

    analysis
}

pub fn analyze_backtest(
    fills: &[Fill],
    equities: &Vec<f64>,
    timestamps_ms: &[u64],
    exposures_series: &[f64],
) -> Analysis {
    let mut analysis = analyze_backtest_basic(fills, equities, timestamps_ms);

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
        let subset_start_ts = if timestamps_ms.len() == equities.len() && !timestamps_ms.is_empty() {
            timestamps_ms[start_idx]
        } else {
            fallback_timestamp_ms(start_idx)
        };
        let subset_fills: Vec<Fill> = fills
            .iter()
            .filter(|fill| {
                if fill.timestamp_ms > 0 {
                    fill.timestamp_ms >= subset_start_ts
                } else {
                    fill.index >= start_idx
                }
            })
            .cloned()
            .collect();
        if subset_fills.len() == 0 {
            break;
        }

        let subset_timestamps = if timestamps_ms.len() == equities.len() {
            &timestamps_ms[start_idx..]
        } else {
            &[]
        };
        let subset_analysis =
            analyze_backtest_basic(&subset_fills, &subset_equities.to_vec(), subset_timestamps);
        subset_analyses.push(subset_analysis);
    }

    // Compute weighted metrics as the mean of subset analyses
    analysis.adg_w = subset_analyses.iter().map(|a| a.adg).sum::<f64>() / 10.0;
    analysis.adg_pnl_w = subset_analyses.iter().map(|a| a.adg_pnl).sum::<f64>() / 10.0;
    analysis.mdg_pnl_w = subset_analyses.iter().map(|a| a.mdg_pnl).sum::<f64>() / 10.0;
    analysis.mdg_w = subset_analyses.iter().map(|a| a.mdg).sum::<f64>() / 10.0;
    analysis.sharpe_ratio_w = subset_analyses.iter().map(|a| a.sharpe_ratio).sum::<f64>() / 10.0;
    analysis.sortino_ratio_w = subset_analyses.iter().map(|a| a.sortino_ratio).sum::<f64>() / 10.0;
    analysis.sharpe_ratio_pnl_w = subset_analyses
        .iter()
        .map(|a| a.sharpe_ratio_pnl)
        .sum::<f64>()
        / 10.0;
    analysis.sortino_ratio_pnl_w = subset_analyses
        .iter()
        .map(|a| a.sortino_ratio_pnl)
        .sum::<f64>()
        / 10.0;
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

    analysis.positions_held_per_day_w = subset_analyses
        .iter()
        .map(|a| a.positions_held_per_day)
        .sum::<f64>()
        / 10.0;

    // Use absolute values for exposure metrics since short positions have negative twe_net.
    // The metric represents "how much exposure" regardless of direction.
    let exposures: Vec<f64> = if !exposures_series.is_empty() {
        exposures_series
            .iter()
            .cloned()
            .filter(|value| value.is_finite())
            .map(|v| v.abs())
            .collect()
    } else {
        fills
            .iter()
            .map(|fill| fill.twe_net.abs())
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

    // Compute high-exposure duration metrics per side:
    // Mean and max continuous duration (hours) where twe exceeded the
    // mean of daily-resampled twe averages.
    {
        use std::collections::BTreeMap;

        // twe_short is stored as signed negative; use abs for magnitude
        let twe_extractors: [(fn(&Fill) -> f64, &str); 2] = [
            (|f| f.twe_long, "long"),
            (|f| f.twe_short.abs(), "short"),
        ];

        for (extract_twe, side) in &twe_extractors {
            let mut daily_twe: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
            for fill in fills {
                let ts = if fill.timestamp_ms > 0 {
                    fill.timestamp_ms
                } else {
                    fallback_timestamp_ms(fill.index)
                };
                let day = (ts / MS_PER_DAY) as usize;
                let entry = daily_twe.entry(day).or_insert((0.0, 0));
                entry.0 += extract_twe(fill);
                entry.1 += 1;
            }

            if daily_twe.is_empty() {
                continue;
            }

            let first_day = *daily_twe.keys().next().unwrap();
            let last_day = *daily_twe.keys().next_back().unwrap();
            let total_days = last_day - first_day + 1;

            let daily_means_sum: f64 = daily_twe
                .values()
                .map(|(sum, count)| sum / *count as f64)
                .sum();
            let daily_twe_mean = daily_means_sum / total_days as f64;

            let mut start_ts: Option<u64> = None;
            let mut durations_ms: Vec<u64> = Vec::new();

            for fill in fills {
                let ts = if fill.timestamp_ms > 0 {
                    fill.timestamp_ms
                } else {
                    fallback_timestamp_ms(fill.index)
                };
                if extract_twe(fill) > daily_twe_mean {
                    if start_ts.is_none() {
                        start_ts = Some(ts);
                    }
                } else if let Some(start) = start_ts {
                    durations_ms.push(ts.saturating_sub(start));
                    start_ts = None;
                }
            }
            if let Some(start) = start_ts {
                if let Some(last_fill) = fills.last() {
                    let last_ts = if last_fill.timestamp_ms > 0 {
                        last_fill.timestamp_ms
                    } else {
                        fallback_timestamp_ms(last_fill.index)
                    };
                    durations_ms.push(last_ts.saturating_sub(start));
                }
            }

            if !durations_ms.is_empty() {
                let hrs_mean = durations_ms.iter().sum::<u64>() as f64
                    / (durations_ms.len() as f64 * MS_PER_HOUR as f64);
                let hrs_max = *durations_ms.iter().max().unwrap() as f64 / MS_PER_HOUR as f64;

                match *side {
                    "long" => {
                        analysis.high_exposure_hours_mean_long = hrs_mean;
                        analysis.high_exposure_hours_max_long = hrs_max;
                    }
                    "short" => {
                        analysis.high_exposure_hours_mean_short = hrs_mean;
                        analysis.high_exposure_hours_max_short = hrs_max;
                    }
                    _ => {}
                }
            }
        }
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
    let analysis_usd = analyze_backtest(
        fills,
        &equities.usd_total_equity,
        &equities.timestamps_ms,
        total_wallet_exposures,
    );
    if !use_btc_collateral {
        return (analysis_usd.clone(), analysis_usd);
    }
    let mut btc_fills = fills.to_vec();
    for fill in btc_fills.iter_mut() {
        let price = if fill.btc_price > 0.0 {
            fill.btc_price
        } else {
            1.0
        };
        fill.usd_total_balance /= price; // balance expressed in BTC
        fill.pnl /= price;
        fill.fee_paid /= price;
        fill.fill_price /= price;
        fill.position_price /= price;
    }
    let analysis_btc = analyze_backtest(
        &btc_fills,
        &equities.btc_total_equity,
        &equities.timestamps_ms,
        total_wallet_exposures,
    );
    (analysis_usd, analysis_btc)
}

fn calc_daily_pnl_ratios(fills: &[Fill]) -> Vec<f64> {
    if fills.is_empty() {
        return Vec::new();
    }
    use std::collections::BTreeMap;
    let mut daily_totals: BTreeMap<usize, (f64, f64)> = BTreeMap::new(); // day -> (pnl_sum_with_fees, last_balance)

    for fill in fills {
        let day = if fill.timestamp_ms > 0 {
            (fill.timestamp_ms / MS_PER_DAY) as usize
        } else {
            fill.index / 1440
        };
        let entry = daily_totals
            .entry(day)
            .or_insert((0.0, fill.usd_total_balance));
        // include fees to get net daily pnl
        entry.0 += fill.pnl + fill.fee_paid;
        entry.1 = fill.usd_total_balance;
    }

    let mut daily_pct = Vec::with_capacity(daily_totals.len());
    for (_day, (pnl_sum, last_balance)) in daily_totals {
        if !pnl_sum.is_finite() || !last_balance.is_finite() {
            continue;
        }
        let denom = last_balance.abs().max(1e-12);
        daily_pct.push(pnl_sum / denom);
    }
    daily_pct
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn calc_sharpe_and_sortino(values: &[f64], mean_val: f64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let std_dev = {
        let var = values.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / values.len() as f64;
        var.sqrt()
    };
    let sharpe = if std_dev != 0.0 {
        mean_val / std_dev
    } else {
        0.0
    };

    let downside: Vec<f64> = values.iter().copied().filter(|x| *x < 0.0).collect();
    let downside_dev = if downside.is_empty() {
        0.0
    } else {
        (downside.iter().map(|x| x.powi(2)).sum::<f64>() / downside.len() as f64).sqrt()
    };
    let sortino = if downside_dev != 0.0 {
        mean_val / downside_dev
    } else {
        0.0
    };
    (sharpe, sortino)
}

fn calc_drawdowns(equity_series: &[f64]) -> Vec<f64> {
    if equity_series.is_empty() {
        return Vec::new();
    }

    let mut drawdowns = Vec::with_capacity(equity_series.len());
    let mut peak = equity_series[0];
    if peak.abs() < 1e-12 {
        peak = 1e-12;
    }

    for &value in equity_series.iter() {
        if value > peak {
            peak = value.max(1e-12);
        }
        let denom = peak.abs().max(1e-12);
        drawdowns.push((value - peak) / denom);
    }

    drawdowns
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

fn calc_peak_recovery_hours(series: &[f64], timestamps_ms: Option<&[u64]>) -> f64 {
    if series.is_empty() {
        return 0.0;
    }
    let use_timestamps = match timestamps_ms {
        Some(ts) => !ts.is_empty() && ts.len() == series.len(),
        None => false,
    };
    let mut peak = f64::NEG_INFINITY;
    let mut peak_ts: u64 = if use_timestamps {
        timestamps_ms.unwrap()[0]
    } else {
        0
    };
    let mut max_duration_ms: u64 = 0;
    for (i, &value) in series.iter().enumerate() {
        let ts = if use_timestamps {
            timestamps_ms.unwrap()[i]
        } else {
            fallback_timestamp_ms(i)
        };
        if value >= peak {
            let duration_ms = ts.saturating_sub(peak_ts);
            if duration_ms > max_duration_ms {
                max_duration_ms = duration_ms;
            }
            peak = value;
            peak_ts = ts;
        }
    }
    (max_duration_ms as f64) / MS_PER_HOUR as f64
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

    // Use BTreeMap to enforce deterministic iteration order
    use std::collections::BTreeMap;
    let mut daily_totals: BTreeMap<usize, f64> = BTreeMap::new();

    for fill in fills {
        let day = if fill.timestamp_ms > 0 {
            (fill.timestamp_ms / MS_PER_DAY) as usize
        } else {
            fill.index / 1440
        };
        let cost_pct = (fill.fill_qty.abs() * fill.fill_price) / fill.usd_total_balance;
        *daily_totals.entry(day).or_insert(0.0) += cost_pct;
    }

    let total_days = daily_totals.len() as f64;
    if total_days == 0.0 {
        0.0
    } else {
        daily_totals.values().sum::<f64>() / total_days
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrderType;

    fn make_fill(index: usize, twe_net: f64) -> Fill {
        Fill {
            index,
            timestamp_ms: (index as u64) * 60_000,
            coin: "TEST".to_string(),
            pnl: 0.0,
            fee_paid: 0.1,
            usd_total_balance: 10000.0,
            btc_cash_wallet: 0.0,
            usd_cash_wallet: 10000.0,
            btc_price: 50000.0,
            fill_qty: -0.1,
            fill_price: 50000.0,
            position_size: -0.1,
            position_price: 50000.0,
            order_type: OrderType::EntryInitialNormalShort,
            wallet_exposure: twe_net.abs(),
            twe_long: 0.0,
            twe_short: twe_net,
            twe_net,
        }
    }

    #[test]
    fn test_total_wallet_exposure_max_short_only() {
        // For short-only configs, twe_net is always negative.
        // total_wallet_exposure_max should report the maximum absolute exposure.
        let fills = vec![
            make_fill(100, -0.5), // 50% short exposure
            make_fill(200, -1.0), // 100% short exposure
            make_fill(300, -0.3), // 30% short exposure
        ];

        let equities: Vec<f64> = vec![10000.0; 400];
        let timestamps: Vec<u64> = (0..equities.len()).map(|i| (i as u64) * 60_000).collect();
        // Empty exposures_series forces analysis to use fill.twe_net
        let exposures_series: Vec<f64> = vec![];

        let analysis = analyze_backtest(&fills, &equities, &timestamps, &exposures_series);

        // With abs() fix: max(0.5, 1.0, 0.3) = 1.0
        assert!(
            (analysis.total_wallet_exposure_max - 1.0).abs() < 0.01,
            "Expected total_wallet_exposure_max=1.0 for short-only, got {}",
            analysis.total_wallet_exposure_max
        );
    }

    #[test]
    fn test_total_wallet_exposure_from_exposures_series() {
        // When exposures_series is provided, it should use that instead of fill.twe_net
        let fills = vec![make_fill(100, -0.5)];
        let equities: Vec<f64> = vec![10000.0; 200];
        let timestamps: Vec<u64> = (0..equities.len()).map(|i| (i as u64) * 60_000).collect();
        // Provide negative exposure values (short-only pattern)
        let exposures_series: Vec<f64> = vec![-0.2, -0.5, -0.8, -0.3];

        let analysis = analyze_backtest(&fills, &equities, &timestamps, &exposures_series);

        // With abs() fix: max(0.2, 0.5, 0.8, 0.3) = 0.8
        assert!(
            (analysis.total_wallet_exposure_max - 0.8).abs() < 0.01,
            "Expected total_wallet_exposure_max=0.8, got {}",
            analysis.total_wallet_exposure_max
        );
    }
}
