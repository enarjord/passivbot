//! Shared controlled equity references for Rust producers and GPU reductions.
//! This runs analysis only; it does not replay candles or exercise HSL.

use super::*;
use crate::analysis::analyze_backtest;
use crate::types::EquityHardStopLossConfig;
use ndarray::{Array1, Array3};

fn assert_metrics(case: &str, actual: &[(&str, f64)], expected: &serde_json::Value) {
    for &(name, value) in actual {
        if let Some(reference) = expected[name].as_f64() {
            assert!(
                value.is_finite()
                    && (value - reference).abs() <= 1e-8_f64.max(reference.abs() * 1e-12),
                "{case} {name}: Rust {value}, reference {reference}"
            );
        } else {
            assert!(
                expected[name].is_null() && !value.is_finite(),
                "{case} {name}: expected nonfinite, Rust {value}"
            );
        }
    }
}

#[test]
fn weighted_equity_shared_references_match_current_producers() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/gpu_weighted_equity.json"
    ))
    .unwrap();
    let hlcvs = Array3::from_shape_vec((4, 1, 4), vec![1.0; 4 * 1 * 4]).unwrap();
    let btc_usd_prices = Array1::from_vec(vec![20_000.0; 4]);

    let mut bp_pair = BotParamsPair::default();
    bp_pair.long.n_positions = 1;
    bp_pair.short.n_positions = 1;
    bp_pair.long.total_wallet_exposure_limit = 1.0;
    bp_pair.short.total_wallet_exposure_limit = 1.0;
    bp_pair.long.ema_span_0 = 10.0;
    bp_pair.long.ema_span_1 = 20.0;
    bp_pair.short.ema_span_0 = 10.0;
    bp_pair.short.ema_span_1 = 20.0;
    bp_pair.long.hsl_enabled = true;
    bp_pair.short.hsl_enabled = true;

    let hs = EquityHardStopLossConfig::default();

    let backtest_params = BacktestParams {
        starting_balance: 100.0,
        maker_fee: 0.0,
        taker_fee: 0.0,
        coins: vec!["TEST".to_string()],
        active_coin_indices: None,
        first_timestamp_ms: 0,
        requested_start_timestamp_ms: 0,
        first_valid_indices: vec![0],
        last_valid_indices: vec![3],
        warmup_minutes: vec![0],
        trade_start_indices: vec![0],
        global_warmup_bars: 0,
        btc_collateral_cap: 0.0,
        btc_collateral_ltv_cap: None,
        metrics_only: true,
        hsl_detailed_report: false,
        skip_btc_analysis: false,
        filter_by_min_effective_cost: false,
        dynamic_wel_by_tradability: true,
        hedge_mode: true,
        max_realized_loss_pct: 1.0,
        pnls_max_lookback_days: 365.0,
        liquidation_threshold: 0.05,
        market_orders_allowed: false,
        market_order_near_touch_threshold: 0.001,
        market_order_slippage_pct: 0.0,
        limit_order_fill_buffer_pct: 0.0,
        forager_score_hysteresis_pct: 0.0,
        equity_hard_stop_loss: hs,
        candle_interval_minutes: 1,
    };

    let mut bt = Backtest::new(
        hlcvs.view(),
        btc_usd_prices.view(),
        vec![bp_pair],
        vec![ExchangeParams::default()],
        &backtest_params,
    );
    for case in fixture["cases"].as_array().unwrap() {
        let first = case["first_timestamp_ms"].as_u64().unwrap();
        let interval = case["interval_ms"].as_u64().unwrap();
        for quantized in [false, true] {
            let values: Vec<f64> = case["equities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| {
                    let value = value.as_f64().unwrap();
                    if quantized {
                        value as f32 as f64
                    } else {
                        value
                    }
                })
                .collect();
            let timestamps: Vec<u64> = (0..values.len())
                .map(|i| first + i as u64 * interval)
                .collect();
            bt.equities.timestamps_ms = timestamps.clone();
            let raw = bt.strategy_equity_metrics_from_series(&values, None, Some(&timestamps));
            let name = case["case"].as_str().unwrap();
            let expected_raw = if quantized {
                "expected_raw_f32"
            } else {
                "expected_raw"
            };
            assert_metrics(
                name,
                &[
                    ("adg_strategy_eq_w", raw.adg_strategy_eq_w),
                    ("mdg_strategy_eq_w", raw.mdg_strategy_eq_w),
                    ("sharpe_ratio_strategy_eq_w", raw.sharpe_ratio_strategy_eq_w),
                    (
                        "sortino_ratio_strategy_eq_w",
                        raw.sortino_ratio_strategy_eq_w,
                    ),
                    ("omega_ratio_strategy_eq_w", raw.omega_ratio_strategy_eq_w),
                    ("calmar_ratio_strategy_eq_w", raw.calmar_ratio_strategy_eq_w),
                    (
                        "sterling_ratio_strategy_eq_w",
                        raw.sterling_ratio_strategy_eq_w,
                    ),
                ],
                &case[expected_raw],
            );
            let expected_full = if quantized { "expected_raw_full_f32" } else { "expected_raw_full" };
            assert_metrics(name, &[
                ("adg_rolling_hmean_strategy_eq", raw.adg_rolling_hmean_strategy_eq),
                ("adg_strategy_eq", raw.adg_strategy_eq),
                ("adg_time_integrated_strategy_eq", raw.adg_time_integrated_strategy_eq),
                ("calmar_ratio_strategy_eq", raw.calmar_ratio_strategy_eq),
                ("expected_shortfall_1pct_strategy_eq", raw.expected_shortfall_1pct_strategy_eq),
                ("mdg_strategy_eq", raw.mdg_strategy_eq),
                ("omega_ratio_strategy_eq", raw.omega_ratio_strategy_eq),
                ("positive_gain_participation_strategy_eq", raw.positive_gain_participation_strategy_eq),
                ("sharpe_ratio_strategy_eq", raw.sharpe_ratio_strategy_eq),
                ("sortino_ratio_strategy_eq", raw.sortino_ratio_strategy_eq),
                ("sterling_ratio_strategy_eq", raw.sterling_ratio_strategy_eq),
            ], &case[expected_full]);
            for (variant, fill_indices) in case["fill_indices"].as_object().unwrap() {
                let fills: Vec<Fill> = fill_indices
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|index| {
                        let index = index.as_u64().unwrap() as usize;
                        Fill {
                            index,
                            timestamp_ms: timestamps[index],
                            coin: "TEST".to_string(),
                            pnl: 0.0,
                            fee_paid: 0.1,
                            usd_total_balance: 100.0,
                            btc_cash_wallet: 0.0,
                            usd_cash_wallet: 100.0,
                            btc_price: 50000.0,
                            fill_qty: -0.1,
                            fill_price: 50000.0,
                            position_size: -0.1,
                            position_price: 50000.0,
                            order_type: OrderType::EntryInitialNormalShort,
                            liquidity: "maker".to_string(),
                            wallet_exposure: 0.0,
                            twe_long: 0.0,
                            twe_short: 0.0,
                            twe_net: 0.0,
                        }
                    })
                    .collect();
                let account = analyze_backtest(&fills, &values, &timestamps, &[]);
                let expected_account = if quantized {
                    "expected_account_f32"
                } else {
                    "expected_account"
                };
                assert_metrics(
                    name,
                    &[
                        ("adg_w_usd", account.adg_w),
                        ("mdg_w_usd", account.mdg_w),
                        ("sharpe_ratio_w_usd", account.sharpe_ratio_w),
                        ("sortino_ratio_w_usd", account.sortino_ratio_w),
                        ("omega_ratio_w_usd", account.omega_ratio_w),
                        ("calmar_ratio_w_usd", account.calmar_ratio_w),
                        ("sterling_ratio_w_usd", account.sterling_ratio_w),
                    ],
                    &case[expected_account][variant],
                );
            }
        }
    }
}
