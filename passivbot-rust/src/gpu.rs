//! Rust-owned GPU screening assets.
//!
//! GPU screening is intentionally approximate, but strategy-adjacent programs
//! still live behind the Rust ownership boundary. Python only compiles and
//! dispatches this source through the optional Apple MPS backend.

use std::sync::LazyLock;

/// Private replay ABI: sorted strategy band, authored unstuck band (0, 1, derived),
/// minute range, completed-hour range, forager quote volume, forager log range.
pub const MARKET_EMA_BOOTSTRAP_COUNT_COLS: usize = 10;

pub fn market_ema_bootstrap_counts(
    spans: [f64; 8],
    interval_minutes: f64,
    available: u64,
) -> Result<[i32; MARKET_EMA_BOOTSTRAP_COUNT_COLS], &'static str> {
    if !interval_minutes.is_finite() || interval_minutes < 1.0 {
        return Err("GPU EMA bootstrap interval must be finite and at least one minute");
    }
    if available == 0 || available > i32::MAX as u64 {
        return Err("GPU EMA bootstrap history must fit a positive int32 sample count");
    }
    if spans.iter().any(|span| !span.is_finite() || *span < 0.0) {
        return Err("GPU EMA bootstrap spans must be finite and nonnegative");
    }
    // Match CPU derivation order: derive in original minutes, sort, then scale.
    // Do not recover counts from an alpha or a float32 transport value.
    let mut price = [spans[0], spans[1], (spans[0] * spans[1]).sqrt()];
    price.sort_by(f64::total_cmp);
    let sources = [
        price[0],
        price[1],
        price[2],
        spans[2],
        spans[3],
        (spans[2] * spans[3]).sqrt(),
        spans[4],
        spans[5],
        spans[6],
        spans[7],
    ];
    Ok(std::array::from_fn(|index| {
        let span_samples = if index == 7 {
            sources[index]
        } else {
            sources[index] / interval_minutes
        };
        crate::utils::ema_bootstrap_count(span_samples, available) as i32
    }))
}

const MPS_DECIMAL_ROUNDING_SOURCE: &str = include_str!("gpu/mps_decimal_rounding.metal");

const MPS_ADAPTIVE_TIMING_SOURCE: &str = include_str!("gpu/mps_adaptive_timing.metal");
const MPS_UNSTUCK_EMA_MARKER: &str = "// PASSIVBOT_UNSTUCK_EMA_COMMON";
const MPS_UNSTUCK_EMA_COMMON_SOURCE: &str = include_str!("gpu/mps_unstuck_ema_common.metal");

const MPS_HSL_MARKER: &str = "// PASSIVBOT_HSL_COMMON";
const MPS_HSL_DECAY_SOURCE: &str = include_str!("gpu/mps_hsl_decay.metal");
const MPS_HSL_COMMON_SOURCE: &str = include_str!("gpu/mps_hsl_common.metal");
const MPS_HSL_SOURCE: &str = include_str!("gpu/mps_hsl.metal");
const MPS_BTC_RISK_MARKER: &str = "// PASSIVBOT_BTC_RISK_COMMON";
const MPS_BTC_RISK_COMMON_SOURCE: &str = include_str!("gpu/mps_btc_risk_common.metal");
const MPS_EQUITY_BALANCE_DIFF_MARKER: &str = "// PASSIVBOT_EQUITY_BALANCE_DIFF_COMMON";
const MPS_EQUITY_BALANCE_DIFF_COMMON_SOURCE: &str =
    include_str!("gpu/mps_equity_balance_diff_common.metal");
const MPS_ENTRY_INTERVAL_MARKER: &str = "// PASSIVBOT_ENTRY_INTERVAL_COMMON";
const MPS_ENTRY_INTERVAL_COMMON_SOURCE: &str = include_str!("gpu/mps_entry_interval_common.metal");
const MPS_MULTICOIN_MARKER: &str = "// PASSIVBOT_MULTICOIN_COMMON";
const MPS_MULTICOIN_COMMON_SOURCE: &str = include_str!("gpu/mps_multicoin_common.metal");
const MPS_EMA_ANCHOR_MULTICOIN_BODY: &str = include_str!("gpu/mps_ema_anchor_multicoin_long.metal");
const MPS_TRAILING_MARTINGALE_MULTICOIN_BODY: &str =
    include_str!("gpu/mps_trailing_martingale_multicoin.metal");
const MPS_STRATEGY_EQ_RECOVERY_DISTRIBUTION_SOURCE: &str =
    include_str!("gpu/mps_strategy_eq_recovery_distribution.metal");

fn compose_hsl_source(body: &str) -> String {
    assert_eq!(body.matches(MPS_UNSTUCK_EMA_MARKER).count(), 1);
    assert_eq!(
        body.matches(MPS_HSL_MARKER).count(),
        1,
        "MPS source must contain exactly one shared-HSL marker"
    );
    assert_eq!(
        body.matches(MPS_BTC_RISK_MARKER).count(),
        1,
        "MPS source must contain exactly one shared BTC-risk marker"
    );
    assert_eq!(
        body.matches(MPS_EQUITY_BALANCE_DIFF_MARKER).count(),
        1,
        "MPS source must contain exactly one shared equity-balance-diff marker"
    );
    assert_eq!(
        body.matches(MPS_ENTRY_INTERVAL_MARKER).count(),
        1,
        "MPS source must contain exactly one shared entry-interval marker"
    );
    let source = body
        .replacen(
            "using namespace metal;",
            &format!("using namespace metal;\n{MPS_HSL_DECAY_SOURCE}"),
            1,
        )
        .replacen(
            "// PASSIVBOT_DECIMAL_ROUNDING\n",
            MPS_DECIMAL_ROUNDING_SOURCE,
            1,
        )
        .replacen(
            "// PASSIVBOT_ADAPTIVE_TIMING",
            MPS_ADAPTIVE_TIMING_SOURCE,
            1,
        )
        .replacen(MPS_UNSTUCK_EMA_MARKER, MPS_UNSTUCK_EMA_COMMON_SOURCE, 1)
        .replacen(
            MPS_HSL_MARKER,
            &format!("{}\n{}", MPS_HSL_SOURCE, MPS_HSL_COMMON_SOURCE),
            1,
        )
        .replacen(MPS_BTC_RISK_MARKER, MPS_BTC_RISK_COMMON_SOURCE, 1)
        .replacen(
            MPS_EQUITY_BALANCE_DIFF_MARKER,
            MPS_EQUITY_BALANCE_DIFF_COMMON_SOURCE,
            1,
        )
        .replacen(
            MPS_ENTRY_INTERVAL_MARKER,
            MPS_ENTRY_INTERVAL_COMMON_SOURCE,
            1,
        );
    // Authored risk helpers fence their own arithmetic, including inlining.
    // Preserve ordinary strategy contraction outside those definitions.
    source
}

fn compose_multicoin_source(body: &str) -> String {
    assert_eq!(
        body.matches(MPS_MULTICOIN_MARKER).count(),
        1,
        "MPS multi-coin source must contain exactly one shared-common marker"
    );
    let body = body.replacen(
        MPS_MULTICOIN_MARKER,
        &format!(
            "{}\n{}",
            MPS_DECIMAL_ROUNDING_SOURCE, MPS_MULTICOIN_COMMON_SOURCE
        ),
        1,
    );
    compose_hsl_source(&format!(
        "{}{}",
        include_str!("gpu/mps_organillo.metal"),
        body
    ))
}

pub static MPS_EMA_ANCHOR_MULTICOIN_SOURCE: LazyLock<String> =
    LazyLock::new(|| compose_multicoin_source(MPS_EMA_ANCHOR_MULTICOIN_BODY));
pub static MPS_TRAILING_MARTINGALE_MULTICOIN_SOURCE: LazyLock<String> =
    LazyLock::new(|| compose_multicoin_source(MPS_TRAILING_MARTINGALE_MULTICOIN_BODY));

pub fn mps_ema_anchor_multicoin_source() -> &'static str {
    MPS_EMA_ANCHOR_MULTICOIN_SOURCE.as_str()
}

pub fn mps_ema_anchor_multicoin_long_source() -> &'static str {
    MPS_EMA_ANCHOR_MULTICOIN_SOURCE.as_str()
}

pub fn mps_trailing_martingale_multicoin_source() -> &'static str {
    MPS_TRAILING_MARTINGALE_MULTICOIN_SOURCE.as_str()
}

pub fn mps_strategy_eq_recovery_distribution_source() -> &'static str {
    MPS_STRATEGY_EQ_RECOVERY_DISTRIBUTION_SOURCE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_counts_preserve_original_f64_precision_and_derived_order() {
        let near = f64::from_bits(3.0_f64.to_bits() + 1);
        let spans = [near, near, near, near, near, near, near, near];
        assert_eq!(
            market_ema_bootstrap_counts(spans, 1.0, 100).unwrap(),
            [7; 10]
        );
        assert_eq!(
            market_ema_bootstrap_counts([3.0; 8], 1.0, 100).unwrap(),
            [6; 10]
        );
        let a = f64::from_bits(1.0_f64.to_bits() + 1);
        let counts = market_ema_bootstrap_counts(
            [a, 9.0, a, 9.0, near * 3.0, near, near * 3.0, near * 3.0],
            3.0,
            100,
        )
        .unwrap();
        // sqrt(a*b)/3 is just above 1; sqrt((a/3)*(b/3)) rounds to 1.
        assert_eq!(&counts[..3], &[1, 3, 6]);
        assert_eq!(&counts[3..], &[1, 6, 3, 7, 7, 7, 7]);
        let values = [1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert!((crate::utils::ema_last_f64(&values, near) - 8.866141732283465).abs() < 1e-12);
        assert_eq!(crate::utils::ema_last_f64(&values, 3.0), 1.0);
    }

    #[test]
    fn bootstrap_counts_clip_huge_spans_without_integer_wrap_or_span_rounding() {
        assert_eq!(
            market_ema_bootstrap_counts([f64::MAX; 8], 60.0, 73).unwrap(),
            [73; 10]
        );
        assert_eq!(
            market_ema_bootstrap_counts([1e-100; 8], 60.0, 73).unwrap(),
            [1; 10]
        );
        assert_eq!(
            market_ema_bootstrap_counts([0.0; 8], 1.0, 73).unwrap(),
            [1; 10]
        );
        assert!(market_ema_bootstrap_counts([3.0; 8], 0.0, 73).is_err());
        assert!(market_ema_bootstrap_counts([f64::NAN; 8], 1.0, 73).is_err());
        assert!(market_ema_bootstrap_counts([3.0; 8], 1.0, 0).is_err());
        assert!(market_ema_bootstrap_counts([3.0; 8], 1.0, i32::MAX as u64 + 1).is_err());
    }

    #[test]
    fn replay_exports_share_v9_risk_arithmetic_and_explicit_contraction() {
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            let off = source.matches("#pragma STDC FP_CONTRACT OFF").count();
            assert!(off >= 2);
            assert_eq!(off, source.matches("#pragma STDC FP_CONTRACT ON").count());
            assert!(!source.starts_with("#pragma STDC FP_CONTRACT OFF"));
            let contract_at = |definition: &str| {
                let prefix = &source[..source.find(definition).unwrap()];
                match prefix.rfind("#pragma STDC FP_CONTRACT OFF") {
                    None => true,
                    Some(off) => prefix
                        .rfind("#pragma STDC FP_CONTRACT ON")
                        .is_some_and(|on| on > off),
                }
            };
            for definition in [
                "inline bool hsl_decay_policy(",
                "inline bool hsl_decay_observe(",
                "inline bool hsl_decay_auto_allowance(",
                "inline bool hsl_current_scope_facts(",
                "inline void observe_hsl_event(",
            ] {
                assert!(
                    !contract_at(definition),
                    "risk definition must not contract: {definition}"
                );
            }
            for definition in [
                "inline void advance_joint_realized_memory(",
                "inline float allowed_wallet_exposure_limit(",
                "inline float coin_hsl_budget_multiplier(",
            ] {
                if source.contains(definition) {
                    assert!(
                        !contract_at(definition),
                        "risk definition must not contract: {definition}"
                    );
                }
            }
            if source.contains("inline void record_joint_portfolio_fill(") {
                assert!(contract_at("inline void record_joint_portfolio_fill("));
            }
            assert_eq!(source.matches("struct HslDecayPolicy {").count(), 1);
            assert_eq!(source.matches("inline bool hsl_decay_observe(").count(), 1);
            assert!(source.contains(MPS_HSL_DECAY_SOURCE));
            assert!(
                source.find("struct HslDecayPolicy {").unwrap()
                    < source
                        .find("struct JointPortfolioAccount {")
                        .unwrap_or(usize::MAX)
            );
        }
    }

    fn assert_shared_hsl_contract(source: &str) {
        assert!(!source.contains(MPS_HSL_MARKER));
        assert!(source.contains(MPS_HSL_COMMON_SOURCE));
        assert!(source.contains(MPS_HSL_DECAY_SOURCE));
        assert!(!source.contains("replay_factual_hsl("));
        assert!(!source.contains("hsl_compose_scope("));
        assert!(!source.contains("never_restart"));
        assert_eq!(source.matches("inline void hsl_before_fill(").count(), 1);
        assert_eq!(source.matches("inline void hsl_after_fill(").count(), 1);
        assert_eq!(source.matches("struct HslState").count(), 1);
        assert_eq!(source.matches("inline HslState load_hsl(").count(), 1);
        assert_eq!(source.matches("inline void observe_hsl(").count(), 1);
        assert_eq!(source.matches("inline bool hsl_decay_observe(").count(), 1);
        assert_eq!(source.matches("inline void update_hsl(").count(), 1);
        assert_eq!(
            source.matches("inline bool update_dual_side_hsl(").count(),
            1
        );
        assert_eq!(source.matches("struct HslOutputAggregate").count(), 1);
        for signature in [
            "inline HslOutputAggregate init_hsl_output_aggregate(",
            "inline void accumulate_hsl_output(",
            "inline void write_hsl_output_aggregate(",
            "inline void write_one_side_hsl_outputs(",
            "inline void write_one_side_coin_hsl_outputs(",
            "inline void write_dual_side_hsl_outputs(",
            "inline void write_dual_side_coin_hsl_outputs(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert!(source.contains("constant int HSL_SIGNAL_UNIFIED = 0"));
        assert!(source.contains("constant int HSL_SIGNAL_PSIDE = 1"));
        assert!(source.contains("constant int HSL_SIGNAL_COIN = 2"));
        assert!(source.contains("int ho = po + hsl_param_offset"));
        assert!(source.contains("long_hsl.signal_mode != short_hsl.signal_mode"));
        assert!(source.contains("h.drawdown_ema_max = fmax("));
        assert!(source.contains("output.drawdown_ema_max_long"));
        assert!(source.contains("output.drawdown_ema_max_short"));
        let panic_loss_aggregation = source
            .find("output.panic_close_loss_sum += h.panic_close_loss_sum")
            .unwrap();
        let enabled_gate = source.find("if (!h.enabled) return;").unwrap();
        assert!(panic_loss_aggregation < enabled_gate);
        assert_eq!(source.matches("struct HslStrategyEquityStats").count(), 1);
        assert_eq!(source.matches("struct HslDrawdownEmaTailStats").count(), 1);
        assert_eq!(
            source
                .matches("inline HslStrategyEquityStats init_hsl_strategy_equity_stats(")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("inline void update_hsl_strategy_equity_stats(")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("inline float hsl_strategy_equity_recovery_max_steps(")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("inline float hsl_strategy_equity_drawdown_mean_worst_1pct(")
                .count(),
            1
        );
        for signature in [
            "inline HslDrawdownEmaTailStats init_hsl_drawdown_ema_tail_stats(",
            "inline int hsl_drawdown_ema_tail_bin(",
            "inline void update_hsl_drawdown_ema_tail_stats(",
            "inline float hsl_drawdown_ema_mean_worst_1pct(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert!(source.contains("#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 0"));
        assert!(source.contains("#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 0"));
        assert!(source.contains("#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 0"));
        assert_eq!(
            source
                .matches("inline float hsl_strategy_equity_drawdown_max(")
                .count(),
            1
        );
        assert!(source
            .contains("const bool shared_has_position = has_position_long || has_position_short"));
        assert!(source.contains("const bool shared_has_blocking_orders = has_blocking_orders_long"));
    }

    fn assert_shared_btc_risk_contract(source: &str) {
        assert!(!source.contains(MPS_BTC_RISK_MARKER));
        assert!(source.contains(MPS_BTC_RISK_COMMON_SOURCE));
        for signature in [
            "struct BtcRiskState",
            "inline BtcRiskState init_btc_risk_state()",
            "inline void reset_btc_risk_day(",
            "inline void update_btc_risk_state(",
            "inline void write_btc_risk_day(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert!(source.contains("#if PASSIVBOT_BTC_RISK_ENABLED"));
        assert!(source.contains("float btc_equity = usd_equity / btc_price"));
    }

    fn assert_shared_equity_balance_diff_contract(source: &str) {
        assert!(!source.contains(MPS_EQUITY_BALANCE_DIFF_MARKER));
        assert!(source.contains(MPS_EQUITY_BALANCE_DIFF_COMMON_SOURCE));
        for signature in [
            "struct EquityBalanceDiffState",
            "inline EquityBalanceDiffState init_equity_balance_diff_state()",
            "inline void update_equity_balance_diff_state(",
            "inline void write_equity_balance_diff_state(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert!(source.contains("#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED"));
        assert!(source.contains("const float diff = (equity - balance) / balance"));
    }

    fn assert_shared_entry_interval_contract(source: &str) {
        assert!(!source.contains(MPS_ENTRY_INTERVAL_MARKER));
        assert!(source.contains(MPS_ENTRY_INTERVAL_COMMON_SOURCE));
        for signature in [
            "inline void init_entry_interval_output(",
            "inline void record_initial_entry_interval(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert!(source.contains("#if PASSIVBOT_ENTRY_INTERVAL_ENABLED"));
        assert!(source.contains("constant int ENTRY_INTERVAL_BINS = 128"));
        assert!(source.contains("constant int ENTRY_INTERVAL_STAT_COLS = 2"));
        assert!(source.contains("constant int ENTRY_INTERVAL_COUNT_COLS = 129"));
        assert!(source.contains("device int* counts"));
    }

    #[test]
    fn strategy_eq_recovery_distribution_source_preserves_strict_rust_contract() {
        let source = mps_strategy_eq_recovery_distribution_source();

        assert!(source.contains("kernel void passivbot_strategy_eq_recovery_distribution("));
        assert!(source.contains("if (!(value > strategy_equity_samples[offset + pending_slot]))"));
        assert!(source.contains("constant int RECOVERY_METRIC_COLS = 7"));
        assert!(source.contains("recovery_histogram_percentile("));
        assert!(source.contains("recovery_histogram_mean_worst_pct("));
        assert!(source.contains("== RECOVERY_FAIL_CLOSED_SENTINEL"));
        assert!(source.contains("output[output_offset + metric] = full_horizon"));
    }

    fn assert_shared_multicoin_contract(source: &str) {
        assert!(!source.contains(MPS_MULTICOIN_MARKER));
        assert!(
            source.contains(
                &MPS_MULTICOIN_COMMON_SOURCE
                    .replace(" ORGANILLO_PASS", "")
                    .replace(" ORGANILLO_ARGS", "")
            ) || source.contains(MPS_MULTICOIN_COMMON_SOURCE)
        );
        assert_eq!(source.matches("struct GpuDecimal {").count(), 1);
        for signature in [
            "inline float round_step(",
            "inline float ceil_step(",
            "inline float floor_step(",
            "inline float min_entry_qty(",
            "inline bool finite_positive(",
            "inline float float32_floor_nonnegative(",
            "inline void record_realized_net(",
            "inline void record_gross_pnl(",
            "inline int wallet_exposure_denominator_n_positions(",
            "inline float coin_override_or(",
            "inline float allowed_wallet_exposure_limit(",
            "inline float clamped_market_price(",
            "inline JointPortfolioAccount init_joint_portfolio_account(",
            "inline void record_joint_portfolio_fill(",
            "inline float joint_portfolio_equity(",
            "inline bool joint_portfolio_can_generate(",
            "inline bool update_joint_pside_hsl(",
            "inline int joint_hsl_report_tier(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert_eq!(source.matches("struct JointPortfolioAccount").count(), 1);
        assert!(source.contains("#ifndef PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY"));
        assert!(source.contains("#define PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY 1"));
        assert!(source.contains("#if PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY"));
        assert!(source.contains("return min(configured_n_positions, observed_tradable_count)"));
        assert!(source.contains("return configured_n_positions"));
        assert!(source.contains("if (is_long) account.realized_pnl_long += net_pnl"));
        assert!(source.contains("else account.realized_pnl_short += net_pnl"));
        assert!(source.contains("long_hsl.signal_mode != short_hsl.signal_mode"));
        assert!(source.contains("#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED"));
        assert!(source.contains("const int recovery_stride = sizes[8]"));
        assert!(source.contains("const int recovery_sample_count = sizes[9]"));
        assert!(source.contains("const bool recovery_terminal = liquidated || k == stop_k - 1"));
        assert!(source.contains("int(b) * recovery_sample_count + sample_index"));
        assert!(source.contains("return update_dual_side_hsl("));
        assert!(source.contains("account.realized_pnl_total"));
        assert!(source.contains("account.realized_pnl_long, account.realized_pnl_short"));
    }

    #[test]
    fn all_strategy_sources_compose_one_shared_btc_risk_controller() {
        for body in [
            MPS_EMA_ANCHOR_MULTICOIN_BODY,
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY,
        ] {
            assert_eq!(body.matches(MPS_BTC_RISK_MARKER).count(), 1);
            assert!(!body.contains("struct BtcRiskState"));
        }
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            assert_shared_btc_risk_contract(source);
            assert!(source.contains("constant float* btc_prices"));
            assert!(source.contains("update_btc_risk_state("));
            assert!(source.contains("write_btc_risk_day("));
        }
    }

    #[test]
    fn all_strategy_sources_compose_one_shared_equity_balance_diff_controller() {
        for body in [
            MPS_EMA_ANCHOR_MULTICOIN_BODY,
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY,
        ] {
            assert_eq!(body.matches(MPS_EQUITY_BALANCE_DIFF_MARKER).count(), 1);
            assert!(!body.contains("struct EquityBalanceDiffState"));
        }
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            assert_shared_equity_balance_diff_contract(source);
            assert!(source.contains("update_equity_balance_diff_state("));
            assert!(source.contains("write_equity_balance_diff_state("));
        }
    }

    #[test]
    fn all_strategy_sources_compose_one_shared_entry_interval_accumulator() {
        for body in [
            MPS_EMA_ANCHOR_MULTICOIN_BODY,
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY,
        ] {
            assert_eq!(body.matches(MPS_ENTRY_INTERVAL_MARKER).count(), 1);
            assert!(!body.contains("inline void record_initial_entry_interval("));
        }
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            assert_shared_entry_interval_contract(source);
        }
    }

    #[test]
    fn strategy_sources_do_not_include_high_exposure_replay() {
        for source in [
            MPS_EMA_ANCHOR_MULTICOIN_BODY,
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY,
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            assert!(!source.contains("PASSIVBOT_HIGH_EXPOSURE"));
            assert!(!source.contains("HighExposureState"));
            assert!(!source.contains("high_exposure_output"));
        }
    }

    #[test]
    fn multicoin_kernels_route_every_fill_through_joint_account_state() {
        for (body, expected_account_record_sites, helper_prefix) in [
            (
                MPS_EMA_ANCHOR_MULTICOIN_BODY,
                2,
                "force_close_ema_multicoin_delisted",
            ),
            (
                MPS_TRAILING_MARTINGALE_MULTICOIN_BODY,
                2,
                "force_close_tm_multicoin_delisted",
            ),
        ] {
            assert!(body.contains(
                "JointPortfolioAccount account = init_joint_portfolio_account(starting_balance)"
            ));
            assert!(body.contains("thread float& balance = account.balance"));
            assert!(body
                .contains("thread float& realized_pnl_cumsum_last = account.realized_pnl_total"));
            // Both independent consumers retain the actual-fill producer;
            // neither policy takes its rate from a side's HSL controller.
            assert!(body.contains("joint_realized_loss_allowance("));
            assert_eq!(
                body.matches("configure_joint_realized_memory(account, params[po +")
                    .count(),
                2
            );
            assert!(!body.contains("account.realized_decay_beta ="));
            assert!(!body.contains("volume_drop"));
            assert!(!body.contains("unstuck_pnl_values"));
            assert_eq!(
                body.matches("record_realized_net(").count(),
                expected_account_record_sites
            );
            assert!(!body.contains("float balance = starting_balance"));
            assert!(!body.contains("balance += net_pnl"));
            assert!(!body.contains("balance -= fee"));
            assert!(body.contains(&format!("{helper_prefix}_one_side(")));
            assert!(body.contains(&format!("{helper_prefix}_fused(")));
            assert!(body.contains("last_valid + 1400 >= timestep_count"));
            assert!(body.contains("ordinary_market_fill_price("));
            assert!(body.contains("record_hsl_panic_fill("));
        }
    }

    #[test]
    fn market_emas_bootstrap_at_the_first_causal_usable_boundary() {
        assert!(!MPS_MULTICOIN_COMMON_SOURCE.contains("ceil(2.0f * span)"));
        assert!(MPS_MULTICOIN_COMMON_SOURCE.contains("min(requested_count, k - first_valid + 1)"));
        assert!(MPS_MULTICOIN_COMMON_SOURCE.contains("numerator = fma(beta, numerator, sample)"));
        assert!(MPS_MULTICOIN_COMMON_SOURCE.contains("denominator = fma(beta, denominator, 1.0f)"));
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            assert_eq!(
                source.matches("inline float bootstrap_market_ema(").count(),
                1
            );
            assert!(source.contains("planning_ready && k >= int(coin_settings[coin_offset + 8])"));
            assert!(source.contains("if (can_bootstrap && !side.market_ema_bootstrapped[c])"));
            assert!(source.contains("side.market_ema_bootstrapped[c] = true"));
            assert!(source.contains("!side.hourly_ema_bootstrapped[c]"));
            assert_eq!(source.matches("constant int* bootstrap_counts,").count(), 6);
            assert_eq!(source.matches("bootstrap_counts[c * 10 +").count(), 10);
            for offset in [
                "int(b) * C * 10",
                "int(b) * 2 * C * 10",
                "(int(b) * 2 + 1) * C * 10",
            ] {
                assert!(source.contains(&format!("bootstrap_counts + {offset}")));
            }
            assert!(!source.contains("PASSIVBOT_WEIGHTED"));
        }
    }

    #[test]
    fn realized_memory_consumers_have_separate_rates_and_one_economic_ledger() {
        let source = MPS_MULTICOIN_COMMON_SOURCE;
        assert!(source.contains("RealizedDecayMemory unstuck_memory"));
        assert!(source.contains("RealizedDecayMemory loss_gate_memory"));
        assert!(source.contains("if (account.unstuck_memory.enabled)"));
        assert!(source.contains("if (account.loss_gate_memory.enabled)"));
        assert_eq!(
            source.matches("hsl_decay_realized_fill(net_pnl,").count(),
            2
        );
        assert_eq!(source.matches("account.balance = next;").count(), 1);
        assert_eq!(
            source
                .matches("account.realized_pnl_total += net_pnl;")
                .count(),
            1
        );
        assert!(source.contains("account.balance, fraction, account.loss_gate_memory.drawdown"));
    }

    #[test]
    fn multicoin_sources_compose_hsl_before_joint_controller_helpers() {
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            let hsl = source.find("struct HslState").unwrap();
            let joint = source.find("inline bool update_joint_pside_hsl(").unwrap();
            assert!(hsl < joint);
            assert_shared_hsl_contract(source);
            assert_shared_multicoin_contract(source);
        }
    }

    #[test]
    fn multicoin_temporal_kernel_buffers_use_untyped_state_transport() {
        for source in [
            mps_ema_anchor_multicoin_source(),
            mps_trailing_martingale_multicoin_source(),
        ] {
            for kernel in source.split("kernel void ").skip(1) {
                let signature = kernel.split('{').next().unwrap();
                assert!(!signature.contains("ReplayState* replay_states"));
            }
            assert_eq!(
                source.matches("device char* replay_state_bytes,").count(),
                3
            );
            assert!(source.contains("reinterpret_cast<device "));
        }
    }

    #[test]
    fn ema_anchor_multicoin_mps_source_exposes_expected_kernel_contract() {
        let normalized = mps_ema_anchor_multicoin_source()
            .replace(" ORGANILLO_PASS", "")
            .replace(" ORGANILLO_ARGS", "");
        let source = normalized.as_str();
        assert_shared_hsl_contract(source);
        assert_shared_multicoin_contract(source);
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin"));
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin_fused"));
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin_long"));
        assert!(source.contains("const bool short_side"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 48"));
        assert!(source.contains("constant int OVERRIDE_COLS = 34"));
        assert!(source.contains("constant int HSL_OVERRIDE_START = 19"));
        assert!(source.contains("constant int FORCED_ACTIVE_OVERRIDE_COL = 27"));
        assert!(source.contains("apply_coin_hsl_overrides("));
        assert!(source.contains("coin_override_or"));
        assert!(source.contains("constant int COIN_COLS = 13"));
        assert!(source.contains("passes_multicoin_min_effective_cost"));
        assert!(source.contains("filter_by_min_effective_cost, account.balance"));
        assert!(!source.contains("min_cost_exact_open_uncertain"));
        assert!(!source.contains("multicoin_min_cost_rejection_possible"));
        assert!(source.contains("constant int DAILY_COLS = 9"));
        assert!(source.contains("day_min_balance"));
        assert!(source.contains("constant int SCALAR_COLS = 62"));
        assert!(source.contains("constant int SCALAR_COLS = 64"));
        assert!(source.contains("constant int SCALAR_COLS = 68"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 67"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 69"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 73"));
        assert_eq!(source.matches("struct EmaMulticoinSideState").count(), 1);
        assert_eq!(source.matches("struct EmaMulticoinSideConfig").count(), 1);
        assert_eq!(source.matches("struct EmaMulticoinFillState").count(), 1);
        assert!(source.contains("load_ema_multicoin_side_config("));
        assert!(source.contains("init_ema_multicoin_side_state("));
        assert!(source.contains("init_ema_multicoin_fill_state("));
        assert!(source.contains("process_ema_multicoin_side_fills("));
        assert!(source.contains("accumulate_ema_multicoin_side_unrealized_pnl("));
        assert!(source.contains("update_ema_multicoin_side_indicators("));
        assert!(source.contains("count_ema_multicoin_tradable_coins("));
        assert!(source.contains("ema_multicoin_side_has_position("));
        assert!(source.contains("ema_multicoin_side_held_marks_are_valid("));
        assert!(source.contains("ema_multicoin_side_has_blocking_orders("));
        assert!(source.contains("if (k > int(coin_settings[coin_offset + 7])) continue;"));
        assert!(source.contains("bool long_coin_has_blocking_orders = valid &&"));
        assert!(source.contains("bool short_coin_has_blocking_orders = valid &&"));
        assert!(source.contains("if (mark_valid) {\n                    unrealized +="));
        assert!(source.contains("forced_normal_count"));
        assert!(source.contains("retains the separate dynamic-WEL denominator"));
        assert!(source.contains("if (!survivor[c] || !forced_normal) continue;"));
        assert_eq!(
            source.matches("if (!managed_candidate) continue;").count(),
            3
        );
        assert!(!source.contains("any_valid"));
        assert!(source.contains("held_positions_have_missing_prices("));
        assert!(source.contains("scalars[int(b) * FUSED_SCALAR_COLS + 9] = -2.0f;"));
        assert!(source.contains("update_ema_multicoin_dual_side_hsl("));
        assert!(source.contains("if (long_side.hsl.signal_mode != short_side.hsl.signal_mode)"));
        assert!(source.contains("update_joint_pside_hsl("));
        assert!(source.contains("const bool long_active = long_effective_n_positions > 0"));
        assert!(source.contains("const bool short_active = short_effective_n_positions > 0"));
        assert!(source.contains("update_ema_multicoin_side_selection("));
        assert!(source.contains("generate_ema_multicoin_side_orders("));
        assert!(source.contains("passivbot_ema_anchor_multicoin_fused_impl("));
        assert!(source.contains("update_ema_multicoin_dual_side_hsl("));
        assert!(source.contains("write_dual_side_hsl_outputs("));
        assert!(source.contains("write_dual_side_coin_hsl_outputs("));
        assert!(source.contains("hsl_strategy_equity_recovery_max_steps("));
        assert!(source.contains("hsl_drawdown_ema_mean_worst_1pct("));
        assert!(source.contains("hsl_strategy_equity_drawdown_max("));
        assert!(source.contains("long_coin_overrides, short_coin_overrides"));
        assert!(source.contains("net_position_cost -= short_side.psize[c]"));
        assert!(source.contains("float twe_abs = fabs(net_position_cost / account.balance)"));
        assert!(
            source.contains("hsl_multicoin_failure_status(long_side.hsl, long_side.coin_hsl, C)")
        );
        assert!(
            source.contains("hsl_multicoin_failure_status(short_side.hsl, short_side.coin_hsl, C)")
        );
        assert!(source.contains("select_ema_multicoin_unstuck_coin("));
        assert!(source.contains("int forced_unstuck_coin"));
        assert!(source.contains("long_unstuck_diff < short_unstuck_diff"));
        assert!(source.contains("long_unstuck_candidate <= short_unstuck_candidate"));
        assert!(source.contains(
            "const EmaMulticoinSideConfig config = load_ema_multicoin_side_config(params, po)"
        ));
        assert!(source.contains(
            "init_ema_multicoin_side_state(\n        side, config, coin_settings, coin_overrides, C"
        ));
        assert!(source.contains(
            "update_ema_multicoin_side_indicators(\n            side, config, bars, hour_log_ranges,"
        ));
        assert!(source.contains("multicoin_utc_day_index("));
        assert!(source.contains("multicoin_active_fill_day("));
        assert!(!MPS_EMA_ANCHOR_MULTICOIN_BODY.contains("start_hour_minute"));
        assert!(source.contains("bool any_fill = process_ema_multicoin_side_fills("));
        assert!(source.contains(
            "update_ema_multicoin_side_selection(\n                side, config, bars, coin_settings, coin_overrides"
        ));
        assert!(source.contains(
            "generate_ema_multicoin_side_orders(\n                side, config, account"
        ));
        assert!(source.contains("side.max_tradable_seen = max("));
        assert_eq!(
            source.matches("const bool past_activation_guard").count(),
            2
        );
        assert_eq!(
            source
                .matches("!post_fill_balance_depleted && past_activation_guard")
                .count(),
            4
        );
        assert!(source.contains("EmaMulticoinSideState side"));
        assert!(source.contains("thread HslState& hsl = side.hsl"));
        assert!(source.contains("thread HslState* coin_hsl = side.coin_hsl"));
        assert!(source.contains("thread float* psize = side.psize"));
        assert!(source.contains("thread int* entry_tick = side.entry_tick"));
        assert!(source.contains("thread bool* selected = side.selected"));
        assert!(source.contains("load_hsl(params, po, 30)"));
        assert!(source.contains("write_one_side_hsl_outputs("));
        assert!(source.contains("record_hsl_panic_fill("));
        assert_eq!(
            MPS_EMA_ANCHOR_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_close_fill(")
                .count(),
            2
        );
        assert_eq!(
            MPS_EMA_ANCHOR_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_entry_fill(")
                .count(),
            1
        );
        assert!(source.contains("coin_hsl_entry_blocked_mask"));
        assert!(source.contains("market_execution ? taker_fee : maker_fee"));
        assert!(source.contains("const bool market_orders_allowed = run_settings[9] > 0.5f"));
        assert!(source.contains("const bool market_orders_allowed = run_settings[11] > 0.5f"));
        assert!(source.contains("should_use_ordinary_market_execution("));
        assert!(source.contains("ordinary_market_fill_price("));
        assert!(source.contains("resize_market_close_qty("));
        assert!(source.contains("bool entry_market[MAX_COINS]"));
        assert!(source.contains("bool close_market[MAX_COINS]"));
        assert!(source.contains("bool secondary_close_market[MAX_COINS]"));
        assert!(source.contains("contribution[c] = quantity * entry_exposure_price"));
        assert!(source.contains("float price = entry_market[best]"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[scalar_offset + 19] = loss_sum"));
        assert!(source
            .contains("scalars[scalar_offset + 20] = position_unchanged_max_min * interval_ms"));
        assert!(source.contains("scalars[scalar_offset + 21] = allowed_wallet_exposure_limit"));
        assert!(source.contains("scalars[scalar_offset + 22] = total_wallet_exposure_max"));
        assert!(source.contains("scalars[scalar_offset + 23] = total_wallet_exposure_mean"));
        assert!(source.contains("scalars[scalar_offset + 24] = fill_count"));
        assert!(source.contains("scalars[scalar_offset + 25] = fill_count_entry"));
        assert!(source.contains("scalars[scalar_offset + 26] = fill_count_long"));
        assert!(source.contains("scalars[scalar_offset + 27] = fills_active_days_count"));
        assert!(source.contains("scalars[scalar_offset + 28] = pnl_recovery_max_min * interval_ms"));
        assert!(source.contains("scalars[scalar_offset + 29] = held_sum_min * interval_ms"));
        assert!(source.contains("scalars[scalar_offset + 30] = held_count"));
        assert!(
            source.contains("scalars[scalar_offset + 31] = account_recovery_max_min * interval_ms")
        );
        assert!(source.contains("if (effective_equity >= account_peak)"));
        assert!(source.contains("const bool collect_coin_fill_counts = run_settings[6] > 0.5f"));
        assert!(source.contains("const bool hedge_mode = run_settings[10] > 0.5f"));
        assert!(source.contains("compute_ema_multicoin_one_way_initial_blocks"));
        assert!(source.contains("device float* coin_fill_counts"));
        assert_eq!(
            source
                .matches("coin_fill_counts[candidate_index * coin_count + c] += 1.0f")
                .count()
                + source
                    .matches("coin_fill_counts[candidate_index * coin_count + coin] += 1.0f",)
                    .count(),
            2
        );
        assert!(source.contains("close_tick[c] <= fill_ticks[tick_offset + 0]"));
        assert!(source.contains("entry_tick[c] > fill_ticks[tick_offset + 1]"));
        assert!(!source.contains("volume_drop"));
        assert!(source.contains("config.forager_volume_span = params[po + 12]"));
        assert!(source.contains("config.w_volume = params[po + 14]"));
        assert!(source.contains("effective_n_positions"));
        assert!(source.contains(
            "const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f"
        ));
        assert!(
            source.contains("if (alive && !post_fill_balance_depleted && past_activation_guard)")
        );
        assert!(source.contains("const float score_hysteresis = fmax(run_settings[4], 0.0f)"));
        assert!(source.contains("incumbent[c] = side.entry_qty[c] > 0.0f && psize[c] <= 0.0f"));
        assert!(source.contains("if (!selected[c] || incumbent[c] || !survivor[c]) continue"));
        assert!(source.contains("score[challenger] - score[incumbent_coin]"));
        assert!(source.contains("allowed_wallet_exposure_limit"));
        assert!(source.contains("twel_entry_gate_enabled"));
        assert!(source.contains("twel_enforcer_enabled"));
        assert!(source.contains("twel_enforcer_reduce_portfolio"));
        assert!(source.contains("clamped_market_price"));
        assert!(source.contains("secondary_close_qty"));
        assert!(source.contains("finalize_ema_multicoin_reducers_one_side("));
        assert!(source.contains("finalize_ema_multicoin_reducers_fused("));
        assert!(source.contains("gate_ema_multicoin_close("));
        assert!(source.contains("candidate.qty > best_candidate.qty"));
        assert!(source.contains("close_is_protective_reducer[MAX_COINS]"));
        assert!(!source.contains("realized_loss_proxy_allows_close"));
        assert!(source.contains("current_effective_n_positions"));
        assert!(source.contains("wallet_exposure_denominator_n_positions("));
        assert!(source.contains("coin_hsl[c].slot_count = float(effective_n_positions)"));
        assert!(source.contains("distance == best_distance && c > best"));
        assert!(source.contains("= fma("));
        assert!(source.contains("one global auto-unstuck intent"));
        assert!(source.contains("unstuck_loss_allowance_pct"));
        assert!(!source.contains("hard_stop"));
        assert_eq!(
            mps_ema_anchor_multicoin_source(),
            mps_ema_anchor_multicoin_long_source()
        );
    }

    #[test]
    fn trailing_martingale_multicoin_mps_source_exposes_expected_kernel_contract() {
        let normalized = mps_trailing_martingale_multicoin_source()
            .replace(" ORGANILLO_PASS", "")
            .replace(" ORGANILLO_ARGS", "");
        let source = normalized.as_str();
        assert_shared_hsl_contract(source);
        assert_shared_multicoin_contract(source);
        assert_shared_entry_interval_contract(source);
        assert!(source.contains("kernel void passivbot_trailing_martingale_multicoin"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 89"));
        assert!(source.contains("constant int OVERRIDE_COLS = 75"));
        assert!(source.contains("constant int HSL_OVERRIDE_START = 34"));
        assert!(source.contains("constant int GATE_INITIAL_OVERRIDE_COL = 42"));
        assert!(source.contains("constant int GATE_REENTRY_OVERRIDE_COL = 43"));
        assert!(source.contains("constant int FORCED_ACTIVE_OVERRIDE_COL = 44"));
        assert!(source.contains("apply_coin_hsl_overrides("));
        assert!(source.contains("constant int COIN_COLS = 13"));
        assert!(source.contains("passes_multicoin_min_effective_cost"));
        assert!(source.contains("filter_by_min_effective_cost, account.balance"));
        assert!(!source.contains("min_cost_exact_open_uncertain"));
        assert!(!source.contains("multicoin_min_cost_rejection_possible"));
        assert!(source.contains("constant int SCALAR_COLS = 62"));
        assert!(source.contains("constant int SCALAR_COLS = 64"));
        assert!(source.contains("constant int SCALAR_COLS = 68"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 67"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 69"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 73"));
        assert_eq!(
            source
                .matches("struct TrailingMartingaleMulticoinSideState")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("struct TrailingMartingaleMulticoinSideConfig")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("struct TrailingMartingaleMulticoinFillState")
                .count(),
            1
        );
        assert!(source.contains("load_trailing_martingale_multicoin_side_config("));
        assert!(source.contains("init_trailing_martingale_multicoin_side_state("));
        assert!(source.contains("init_trailing_martingale_multicoin_fill_state("));
        assert!(source.contains("record_tm_multicoin_gross_pnl("));
        assert!(source.contains("record_tm_multicoin_close_fill("));
        assert!(source.contains("record_tm_multicoin_entry_fill("));
        assert!(source.contains("finalize_tm_multicoin_close_position("));
        assert!(source.contains("apply_tm_multicoin_entry_position("));
        assert!(source.contains("update_tm_multicoin_position_fill_timestamp("));
        assert!(source.contains("process_tm_multicoin_side_fills("));
        assert_eq!(source.matches("record_initial_entry_interval(").count(), 4);
        assert!(source.contains("float last_initial_entry_k[MAX_COINS]"));
        assert!(source.contains("candidate.order_type == 0"));
        assert!(source.contains("candidate.order_type == 11"));
        assert!(source.contains("apply_tm_multicoin_recursive_entry_twel_gate("));
        assert!(source.contains("if (k < first_valid || k > last_valid) continue;"));
        assert!(source.contains("!finite_positive(bars[bar_offset + 2])"));
        assert!(source.contains("entry_gate_suffix_keep_count"));
        assert!(source.contains("entry_gate_suffix_partial_qty"));
        assert!(source.contains("accumulate_tm_multicoin_side_unrealized_pnl("));
        assert!(source.contains("update_tm_multicoin_side_indicators("));
        assert!(source.contains("count_tm_multicoin_tradable_coins("));
        assert!(source.contains("tm_multicoin_side_has_position("));
        assert!(source.contains("tm_multicoin_side_held_marks_are_valid("));
        assert!(source.contains("tm_multicoin_side_has_blocking_orders("));
        assert!(source.contains("if (k > int(coin_settings[coin_offset + 7])) continue;"));
        assert!(source.contains("bool long_coin_has_blocking_orders = valid &&"));
        assert!(source.contains("bool short_coin_has_blocking_orders = valid &&"));
        assert!(source.contains("if (mark_valid) {\n                    unrealized +="));
        assert!(source.contains("forced_normal_count"));
        assert!(source.contains("retains the separate dynamic-WEL denominator"));
        assert!(source.contains("if (!survivor[c] || !forced_normal) continue;"));
        assert_eq!(
            source.matches("if (!managed_candidate) continue;").count(),
            3
        );
        assert!(!source.contains("any_valid"));
        assert!(source.contains("held_positions_have_missing_prices("));
        assert!(source.contains("scalars[int(b) * FUSED_SCALAR_COLS + 9] = -2.0f;"));
        assert!(source.contains("update_tm_multicoin_dual_side_hsl("));
        assert!(source.contains("update_tm_multicoin_side_selection("));
        assert!(source.contains("generate_tm_multicoin_side_orders("));
        assert!(source.contains("tm_multicoin_entry_initial_balance_pct("));
        assert!(source.contains("wallet_exposure_denominator_n_positions("));
        assert!(source.contains("coin_hsl[c].slot_count = float(effective_n_positions)"));
        assert!(source.contains("passivbot_trailing_martingale_multicoin_fused_impl("));
        assert!(source.contains("kernel void passivbot_trailing_martingale_multicoin_fused("));
        assert!(source.contains("const TrailingMartingaleMulticoinSideConfig long_config ="));
        assert!(source.contains("long_coin_overrides, short_coin_overrides"));
        assert!(source.contains("net_position_cost -= short_side.psize[c]"));
        assert!(
            source.contains("hsl_multicoin_failure_status(long_side.hsl, long_side.coin_hsl, C)")
        );
        assert!(
            source.contains("hsl_multicoin_failure_status(short_side.hsl, short_side.coin_hsl, C)")
        );
        assert!(source.contains("select_tm_multicoin_unstuck_coin("));
        assert!(source.contains("int forced_unstuck_coin"));
        assert!(source.contains("long_unstuck_diff < short_unstuck_diff"));
        assert!(source.contains("long_unstuck_candidate <= short_unstuck_candidate"));
        assert!(source.contains("const TrailingMartingaleMulticoinSideConfig config ="));
        assert!(source.contains(
            "init_trailing_martingale_multicoin_side_state(\n        side, config, coin_settings, coin_overrides, C"
        ));
        assert!(source.contains(
            "update_tm_multicoin_side_indicators(\n            side, config, bars, hour_log_ranges, coin_settings, k, C"
        ));
        assert!(source.contains("multicoin_utc_day_index("));
        assert!(source.contains("multicoin_active_fill_day("));
        assert!(!MPS_TRAILING_MARTINGALE_MULTICOIN_BODY.contains("start_hour_minute"));
        assert!(source.contains(
            "bool any_fill = process_tm_multicoin_side_fills(\n            market_execution_ticks,\n            side, config, account, fills"
        ));
        assert!(source.contains(
            "update_tm_multicoin_side_selection(\n                side, config, bars, coin_settings, coin_overrides"
        ));
        assert!(source
            .contains("generate_tm_multicoin_side_orders(\n                side, config, account"));
        assert!(source.contains("TrailingMartingaleMulticoinSideState side"));
        assert!(source.contains("TrailingMartingaleMulticoinFillState fills"));
        assert!(source.contains("thread HslState& hsl = side.hsl"));
        assert!(source.contains("thread HslState* coin_hsl = side.coin_hsl"));
        assert!(source.contains("thread float* psize = side.psize"));
        assert!(source.contains("thread int* entry_tick = side.entry_tick"));
        assert!(source.contains("thread bool* selected = side.selected"));
        assert!(source.contains("thread int& max_tradable_seen = side.max_tradable_seen"));
        assert_eq!(
            source.matches("const bool past_activation_guard").count(),
            2
        );
        assert_eq!(
            source
                .matches("!post_fill_balance_depleted && past_activation_guard")
                .count(),
            4
        );
        assert!(source.contains("load_hsl(params, po, 47)"));
        assert!(source.contains("write_one_side_hsl_outputs("));
        assert!(source.contains("record_hsl_panic_fill("));
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("record_tm_multicoin_close_fill(")
                .count(),
            5
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("record_tm_multicoin_entry_fill(")
                .count(),
            4
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("finalize_tm_multicoin_close_position(")
                .count(),
            3
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("apply_tm_multicoin_entry_position(")
                .count(),
            4
        );
        assert!(source.contains("next_recursive_grid_entry("));
        assert!(source.contains("entry_recursive_mode"));
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("update_tm_multicoin_position_fill_timestamp(")
                .count(),
            3
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_close_fill(")
                .count(),
            2
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_entry_fill(")
                .count(),
            1
        );
        assert!(source.contains("coin_hsl_entry_blocked_mask"));
        assert!(source.contains("market_execution ? taker_fee : maker_fee"));
        assert!(source.contains("bool entry_market[MAX_COINS]"));
        assert!(source.contains("bool close_market[MAX_COINS]"));
        assert!(source.contains("bool secondary_close_market[MAX_COINS]"));
        assert!(source.contains("should_use_ordinary_market_execution("));
        assert!(source.contains("ordinary_market_fill_price("));
        assert!(source.contains("resize_market_close_qty("));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[scalar_offset + 19] = loss_sum"));
        assert!(source
            .contains("scalars[scalar_offset + 20] = position_unchanged_max_min * interval_ms"));
        assert!(source.contains("scalars[scalar_offset + 21] = allowed_wallet_exposure_limit"));
        assert!(source.contains("scalars[scalar_offset + 22] = total_wallet_exposure_max"));
        assert!(source.contains("scalars[scalar_offset + 23] = total_wallet_exposure_mean"));
        assert!(source.contains("scalars[scalar_offset + 24] = fill_count"));
        assert!(source.contains("scalars[scalar_offset + 25] = fill_count_entry"));
        assert!(source.contains("scalars[scalar_offset + 26] = fill_count_long"));
        assert!(source.contains("scalars[scalar_offset + 27] = fills_active_days_count"));
        assert!(source.contains("scalars[scalar_offset + 28] = pnl_recovery_max_min * interval_ms"));
        assert!(source.contains("scalars[scalar_offset + 29] = held_sum_min * interval_ms"));
        assert!(source.contains("scalars[scalar_offset + 30] = held_count"));
        assert!(
            source.contains("scalars[scalar_offset + 31] = account_recovery_max_min * interval_ms")
        );
        assert!(source.contains("if (effective_equity >= account_peak)"));
        assert!(source.contains("const bool collect_coin_fill_counts = run_settings[6] > 0.5f"));
        assert!(source.contains("const bool hedge_mode = run_settings[10] > 0.5f"));
        assert!(source.contains("const bool market_orders_allowed = run_settings[9] > 0.5f"));
        assert!(source.contains("const bool market_orders_allowed = run_settings[11] > 0.5f"));
        assert!(source.contains("compute_tm_multicoin_one_way_initial_blocks"));
        assert!(source.contains("device float* coin_fill_counts"));
        assert_eq!(
            source
                .matches("coin_fill_counts[int(b) * C + c] += 1.0f")
                .count(),
            0
        );
        assert_eq!(
            source
                .matches("coin_fill_counts[candidate_index * coin_count + coin] += 1.0f")
                .count(),
            2
        );
        assert!(source.contains("coin_wel_enforcer_enabled"));
        assert!(source.contains("coin_wel_enforcer_threshold"));
        assert!(source.contains("twel_enforcer_enabled"));
        assert!(source.contains("twel_enforcer_reduce_portfolio"));
        assert!(source.contains("twel_close_qty"));
        assert!(!source.contains("realized_loss_proxy_allows_close"));
        assert!(source.contains("const bool loss_gate_enabled = !PASSIVBOT_TM_LOSS_GATE_DISABLED"));
        assert!(source.contains("prepare_tm_close_admission"));
        assert!(source.contains("tm_projected_close_pnl"));
        assert!(source.contains("source.finalized_reducer_qty"));
        assert!(source.contains("source.admitted_groups[group / 32]"));
        assert_eq!(
            source
                .matches("&& !apply_tm_multicoin_close_admission(")
                .count(),
            2
        );
        assert!(source.contains(
            "const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f"
        ));
        assert!(
            source.contains("if (alive && !post_fill_balance_depleted && past_activation_guard)")
        );
        assert!(source.contains("market_price * 0.9995f / price_step"));
        assert!(source.matches("clamped_market_price(").count() >= 4);
        assert!(source.contains("finalized_twel_reducer_qty = ordinary_can_accompany_reducer"));
        assert!(source.contains("finalized_reducer_qty_with_ordinary"));
        assert!(source.contains("finalized_wel_reducer_qty = finalized_reducer_qty"));
        assert!(source.contains("finalized_wel_reducer_qty"));
        assert!(source.contains("finalized_unstuck_reducer_qty"));
        assert!(source.contains("reducer_candidate_preferred"));
        assert!(source.contains("? raw_twel_reducer_qty : wel_reducer_qty"));
        assert!(source.contains("? raw_unstuck_reducer_qty : exposure_reducer_qty"));
        assert!(source.contains("secondary_close_qty"));
        assert!(source.contains("recursive_grid_close_groups_after_reducer"));
        assert!(source.contains("for (int rung = 0; rung < max_rungs"));
        assert!(source.contains("bool reducer_before_group"));
        assert!(source.contains("close_reconstruct_after_reducer"));
        assert!(source.contains("close_gen_allowed_wel"));
        assert!(source.contains("current_effective_n_positions"));
        assert!(source.contains("close_grid_gen_psize"));
        assert!(source.contains("(use_twel || use_unstuck) && wel_reducer_qty <= 0.0f"));
        assert!(source.contains("float reserved_grid_qty = strategy_wel_reducer_qty"));
        assert!(!source.contains("strategy_unstuck_reducer_qty"));
        assert!(source.contains("psize[c] - reserved_grid_qty"));
        assert!(source.contains("dust_remainder"));
        assert!(!source.contains("float primary_diff = fabs"));
        assert!(source.contains("coin_override_or"));
        assert!(source.contains("min_since_open"));
        assert!(source.contains("max_since_min"));
        assert!(source.contains("max_since_open"));
        assert!(source.contains("min_since_max"));
        assert!(source.contains("effective_n_positions"));
        assert!(source.contains("allowed_wallet_exposure_limit"));
        assert!(source.contains("twel_entry_gate_enabled"));
        assert!(source.contains("distance == best_distance && c > best"));
        assert!(source.contains("entry_retracement_base"));
        assert!(source.contains("close_retracement_base"));
        assert!(source.contains("touch_nearest_ticks[k * C + c]"));
        assert!(source.contains("as_type<float>(touch_min_qty_bits[k * C + c])"));
        assert!(source.contains("minimum_close_relation > 0"));
        assert!(source.contains("one-global-intent least-stuck selector"));
        assert!(source.contains("unstuck_loss_allowance_pct"));
        assert!(!source.contains("hard_stop"));
    }
}
