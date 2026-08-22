//! Rust-owned GPU screening assets.
//!
//! GPU screening is intentionally approximate, but strategy-adjacent programs
//! still live behind the Rust ownership boundary. Python only compiles and
//! dispatches this source through the optional Apple MPS backend.

use std::sync::LazyLock;

const MPS_HSL_MARKER: &str = "// PASSIVBOT_HSL_COMMON";
const MPS_HSL_COMMON_SOURCE: &str = include_str!("gpu/mps_hsl_common.metal");
const MPS_MULTICOIN_MARKER: &str = "// PASSIVBOT_MULTICOIN_COMMON";
const MPS_MULTICOIN_COMMON_SOURCE: &str = include_str!("gpu/mps_multicoin_common.metal");
const MPS_EMA_ANCHOR_BODY: &str = include_str!("gpu/mps_ema_anchor_directional.metal");
const MPS_TRAILING_MARTINGALE_BODY: &str =
    include_str!("gpu/mps_trailing_martingale_directional.metal");
const MPS_EMA_ANCHOR_MULTICOIN_BODY: &str = include_str!("gpu/mps_ema_anchor_multicoin_long.metal");
const MPS_TRAILING_MARTINGALE_MULTICOIN_BODY: &str =
    include_str!("gpu/mps_trailing_martingale_multicoin.metal");

fn compose_hsl_source(body: &str) -> String {
    assert_eq!(
        body.matches(MPS_HSL_MARKER).count(),
        1,
        "MPS source must contain exactly one shared-HSL marker"
    );
    body.replacen(MPS_HSL_MARKER, MPS_HSL_COMMON_SOURCE, 1)
}

fn compose_multicoin_source(body: &str) -> String {
    assert_eq!(
        body.matches(MPS_MULTICOIN_MARKER).count(),
        1,
        "MPS multi-coin source must contain exactly one shared-common marker"
    );
    compose_hsl_source(&body.replacen(MPS_MULTICOIN_MARKER, MPS_MULTICOIN_COMMON_SOURCE, 1))
}

pub static MPS_EMA_ANCHOR_SOURCE: LazyLock<String> =
    LazyLock::new(|| compose_hsl_source(MPS_EMA_ANCHOR_BODY));
pub static MPS_EMA_ANCHOR_MULTICOIN_SOURCE: LazyLock<String> =
    LazyLock::new(|| compose_multicoin_source(MPS_EMA_ANCHOR_MULTICOIN_BODY));
pub static MPS_TRAILING_MARTINGALE_SOURCE: LazyLock<String> =
    LazyLock::new(|| compose_hsl_source(MPS_TRAILING_MARTINGALE_BODY));
pub static MPS_TRAILING_MARTINGALE_MULTICOIN_SOURCE: LazyLock<String> =
    LazyLock::new(|| compose_multicoin_source(MPS_TRAILING_MARTINGALE_MULTICOIN_BODY));

pub fn mps_ema_anchor_source() -> &'static str {
    MPS_EMA_ANCHOR_SOURCE.as_str()
}

pub fn mps_ema_anchor_multicoin_source() -> &'static str {
    MPS_EMA_ANCHOR_MULTICOIN_SOURCE.as_str()
}

pub fn mps_ema_anchor_multicoin_long_source() -> &'static str {
    MPS_EMA_ANCHOR_MULTICOIN_SOURCE.as_str()
}

pub fn mps_trailing_martingale_source() -> &'static str {
    MPS_TRAILING_MARTINGALE_SOURCE.as_str()
}

pub fn mps_trailing_martingale_multicoin_source() -> &'static str {
    MPS_TRAILING_MARTINGALE_MULTICOIN_SOURCE.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_shared_hsl_contract(source: &str) {
        assert!(!source.contains(MPS_HSL_MARKER));
        assert!(source.contains(MPS_HSL_COMMON_SOURCE));
        assert_eq!(source.matches("struct HslState").count(), 1);
        assert_eq!(source.matches("inline HslState load_hsl(").count(), 1);
        assert_eq!(source.matches("inline bool derive_hsl_signal(").count(), 1);
        assert_eq!(
            source
                .matches("inline void update_hsl_from_signal(")
                .count(),
            1
        );
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
        assert!(source
            .contains("const bool shared_has_position = has_position_long || has_position_short"));
        assert!(source.contains("const bool shared_has_blocking_orders = has_blocking_orders_long"));
    }

    fn assert_shared_multicoin_contract(source: &str) {
        assert!(!source.contains(MPS_MULTICOIN_MARKER));
        assert!(source.contains(MPS_MULTICOIN_COMMON_SOURCE));
        for signature in [
            "inline float round_step(",
            "inline float ceil_step(",
            "inline float floor_step(",
            "inline float min_entry_qty(",
            "inline bool finite_positive(",
            "inline float float32_floor_nonnegative(",
            "inline void record_realized_net(",
            "inline void record_gross_pnl(",
            "inline float coin_override_or(",
            "inline float allowed_wallet_exposure_limit(",
            "inline float clamped_market_price(",
            "inline JointPortfolioAccount init_joint_portfolio_account(",
            "inline void record_joint_portfolio_fill(",
            "inline float joint_portfolio_equity(",
            "inline float joint_hsl_realized_pnl(",
            "inline float joint_hsl_unrealized_pnl(",
            "inline bool joint_portfolio_can_generate(",
            "inline bool update_joint_pside_hsl(",
            "inline void try_restart_joint_pside_hsl(",
            "inline int joint_pside_hsl_global_tier(",
        ] {
            assert_eq!(source.matches(signature).count(), 1, "{signature}");
        }
        assert_eq!(source.matches("struct JointPortfolioAccount").count(), 1);
        assert!(source.contains("if (is_long) account.realized_pnl_long += net_pnl"));
        assert!(source.contains("else account.realized_pnl_short += net_pnl"));
        assert!(source.contains("long_hsl.signal_mode != short_hsl.signal_mode"));
        assert!(source.contains("return update_dual_side_hsl("));
        assert!(source.contains("account.realized_pnl_total"));
        assert!(source.contains("account.realized_pnl_long, account.realized_pnl_short"));
        assert!(source.contains("joint_hsl_realized_pnl(account, unified, true)"));
        assert!(source.contains("joint_hsl_realized_pnl(account, unified, false)"));
    }

    fn assert_directional_hsl_accounting_contract(source: &str) {
        assert!(source.contains("thread float& realized_pnl_cumsum_long"));
        assert!(source.contains("thread float& realized_pnl_cumsum_short"));
        assert!(source.contains("if (is_long) realized_pnl_cumsum_long += net_pnl"));
        assert!(source.contains("else realized_pnl_cumsum_short += net_pnl"));
        assert_eq!(source.matches("update_dual_side_hsl(").count(), 2);
        assert!(source.contains("realized_pnl_cumsum_long, realized_pnl_cumsum_short"));
        assert!(source.contains("long_unreal, short_unreal"));
        assert!(source.contains("long_side.psize > 0.0f, short_side.psize > 0.0f"));
        assert!(source.contains("long_blocking_orders, short_blocking_orders"));
        assert!(source.contains(
            "const bool hsl_modes_valid = long_hsl.signal_mode == short_hsl.signal_mode"
        ));
        assert!(source.contains("bool hsl_update_valid = update_dual_side_hsl("));
        assert!(source.contains("if (!hsl_update_valid)"));
        assert!(source.contains("alive = false"));
        assert!(source.contains("liq_day = di"));
    }

    #[test]
    fn directional_sources_compose_one_shared_hsl_controller() {
        assert_eq!(MPS_EMA_ANCHOR_BODY.matches(MPS_HSL_MARKER).count(), 1);
        assert_eq!(
            MPS_TRAILING_MARTINGALE_BODY.matches(MPS_HSL_MARKER).count(),
            1
        );
        assert!(!MPS_EMA_ANCHOR_BODY.contains("struct HslState"));
        assert!(!MPS_TRAILING_MARTINGALE_BODY.contains("struct HslState"));
        assert_shared_hsl_contract(mps_ema_anchor_source());
        assert_shared_hsl_contract(mps_trailing_martingale_source());
        assert_directional_hsl_accounting_contract(mps_ema_anchor_source());
        assert_directional_hsl_accounting_contract(mps_trailing_martingale_source());
    }

    #[test]
    fn multicoin_kernels_route_every_fill_through_joint_account_state() {
        for (body, expected_account_record_sites) in [
            (MPS_EMA_ANCHOR_MULTICOIN_BODY, 2),
            (MPS_TRAILING_MARTINGALE_MULTICOIN_BODY, 2),
        ] {
            assert!(body.contains(
                "JointPortfolioAccount account = init_joint_portfolio_account(starting_balance)"
            ));
            assert!(body.contains("thread float& balance = account.balance"));
            assert!(body
                .contains("thread float& realized_pnl_cumsum_last = account.realized_pnl_total"));
            assert!(
                body.contains("thread float& realized_pnl_cumsum_max = account.realized_pnl_peak")
            );
            assert_eq!(
                body.matches("record_realized_net(").count(),
                expected_account_record_sites
            );
            assert!(!body.contains("float balance = starting_balance"));
            assert!(!body.contains("balance += net_pnl"));
            assert!(!body.contains("balance -= fee"));
        }
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
    fn ema_anchor_mps_source_exposes_expected_kernel_contract() {
        let source = mps_ema_anchor_source();
        assert_shared_hsl_contract(source);
        assert_directional_hsl_accounting_contract(source);
        assert!(source.contains("kernel void passivbot_ema_anchor"));
        assert!(source.contains("constant int DAILY_COLS = 8"));
        assert!(source.contains("constant int SCALAR_COLS = 62"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[so + 44] = loss_sum"));
        assert!(source.contains("scalars[so + 45] = position_unchanged_max_min * interval_ms"));
        assert!(source.contains("scalars[so + 46] = long_enabled"));
        assert!(source.contains("scalars[so + 47] = short_enabled"));
        assert!(source.contains("scalars[so + 48] = total_wallet_exposure_max"));
        assert!(source.contains("scalars[so + 49] = total_wallet_exposure_mean"));
        assert!(source.contains("scalars[so + 50] = fill_count"));
        assert!(source.contains("scalars[so + 51] = fill_count_entry"));
        assert!(source.contains("scalars[so + 52] = fill_count_long"));
        assert!(source.contains("scalars[so + 53] = fills_active_days_count"));
        assert!(source.contains("scalars[so + 54] = pnl_recovery_max_min * interval_ms"));
        assert!(source.contains("scalars[so + 55] = held_sum_min * interval_ms"));
        assert!(source.contains("scalars[so + 56] = held_count"));
        assert!(source.contains("scalars[so + 57] = account_recovery_max_min * interval_ms"));
        assert!(source.contains("scalars[so + 58] = profit_sum_long"));
        assert!(source.contains("scalars[so + 61] = loss_sum_short"));
        assert!(source.contains("if (eqf >= account_peak)"));
        assert!(source.contains("constant int SIDE_PARAMS = 34"));
        assert!(source.contains("struct HslState"));
        assert!(source.contains("update_hsl("));
        assert!(source.contains("try_restart_hsl("));
        assert!(source.contains("hsl_tier_samples_total"));
        assert!(source.contains("h.restart_retrigger_count"));
        assert!(source.contains("h.halt_duration_sum_steps"));
        assert!(source.contains("record_hsl_panic_fill("));
        assert!(source.contains("h.panic_loss_drawdown_sum"));
        assert!(source.contains("h.slot_count"));
        assert!(source.contains("h.no_restart_peak_strategy_equity"));
        assert!(source.contains("terminal || h.cooldown_minutes <= 0.0f"));
        assert!(source.contains("h.pending_stop_k + h.cooldown_minutes"));
        assert!(source.contains("load_hsl(params, po, 23)"));
        assert!(source.contains("load_hsl(params, po + SIDE_PARAMS, 23)"));
        assert!(source.contains("const float cmp_eps = 1.0e-12f"));
        assert_eq!(source.matches("0.9999999403953552f").count(), 3);
        assert!(source.contains("total_exposure_reducer_qty"));
        assert!(source.contains("unstuck_reducer_variant"));
        assert!(source.contains("unstuck_ema_gating_enabled"));
        assert!(source.contains("unstuck_loss_allowance_pct"));
        assert!(source.contains("Exact Rust emits at most one global unstuck intent"));
        assert!(source.contains("secondary_close_qty"));
        assert!(source.contains("generate_short_orders"));
        assert!(source.contains("const bool hedge_mode"));
        assert!(source.contains("const bool filter_by_min_effective_cost"));
        assert!(source.contains("passes_min_effective_cost"));
        assert!(source.contains("projected_cost_lower"));
        assert!(source.contains("float guaranteed_balance_lower"));
        assert!(source.contains("realized_loss_gate_allows"));
        assert!(source.contains("float32_floor_nonnegative"));
        assert!(source.contains("record_realized_net"));
        assert!(source.contains("const float max_realized_loss_pct = settings[14]"));
        assert!(source.contains("const float taker_fee = settings[15]"));
        assert!(source.contains("const float market_order_slippage_pct = fmax(settings[16], 0.0f)"));
        assert!(source.contains("const bool long_hsl_panic_market = settings[17] > 0.5f"));
        assert!(source.contains("const bool short_hsl_panic_market = settings[18] > 0.5f"));
        assert!(source.contains("market_panic ? taker_fee : maker_fee"));
        assert!(source.contains("close * (1.0f - market_order_slippage_pct)"));
        assert!(source.contains("close * (1.0f + market_order_slippage_pct)"));
        assert!(!source.contains("accumulate_min_cost_balance_error"));
        assert_eq!(source.matches("= fma(").count(), 6);
        assert!(!source.contains("alpha0 * close +"));
        assert_eq!(source.matches("const int fo = k * 11").count(), 1);
        assert_eq!(source.matches("const int touch_down_tick").count(), 1);
        assert_eq!(source.matches("const int touch_up_tick").count(), 1);
        assert_eq!(source.matches("high_fill_max_tick").count(), 4);
        assert_eq!(source.matches("low_nonfill_max_tick").count(), 4);
        assert!(!source.contains("nextafter("));
    }

    #[test]
    fn ema_anchor_multicoin_mps_source_exposes_expected_kernel_contract() {
        let source = mps_ema_anchor_multicoin_source();
        assert_shared_hsl_contract(source);
        assert_shared_multicoin_contract(source);
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin"));
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin_fused"));
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin_long"));
        assert!(source.contains("const bool short_side"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 42"));
        assert!(source.contains("constant int OVERRIDE_COLS = 29"));
        assert!(source.contains("constant int HSL_OVERRIDE_START = 19"));
        assert!(source.contains("apply_coin_hsl_overrides("));
        assert!(source.contains("coin_override_or"));
        assert!(source.contains("constant int COIN_COLS = 12"));
        assert!(source.contains("constant int DAILY_COLS = 9"));
        assert!(source.contains("day_min_balance"));
        assert!(source.contains("constant int SCALAR_COLS = 57"));
        assert!(source.contains("constant int FUSED_SCALAR_COLS = 62"));
        assert_eq!(source.matches("struct EmaMulticoinSideState").count(), 1);
        assert_eq!(source.matches("struct EmaMulticoinSideConfig").count(), 1);
        assert_eq!(source.matches("struct EmaMulticoinFillState").count(), 1);
        assert!(source.contains("load_ema_multicoin_side_config("));
        assert!(source.contains("init_ema_multicoin_side_state("));
        assert!(source.contains("init_ema_multicoin_fill_state("));
        assert!(source.contains("process_ema_multicoin_side_fills("));
        assert!(source.contains(
            "accumulate_ema_multicoin_side_unrealized_pnl("
        ));
        assert!(source.contains("update_ema_multicoin_side_indicators("));
        assert!(source.contains("count_ema_multicoin_tradable_coins("));
        assert!(source.contains("ema_multicoin_side_has_position("));
        assert!(source.contains("ema_multicoin_side_held_marks_are_valid("));
        assert!(source.contains("ema_multicoin_side_has_blocking_orders("));
        assert!(source.contains("update_ema_multicoin_dual_side_hsl("));
        assert!(source.contains("if (long_side.hsl.signal_mode != short_side.hsl.signal_mode)"));
        assert!(source.contains("update_joint_pside_hsl("));
        assert!(source.contains(
            "const bool long_active = long_effective_n_positions > 0"
        ));
        assert!(source.contains(
            "const bool short_active = short_effective_n_positions > 0"
        ));
        assert!(source.contains("update_ema_multicoin_side_selection("));
        assert!(source.contains("generate_ema_multicoin_side_orders("));
        assert!(source.contains("passivbot_ema_anchor_multicoin_fused_impl("));
        assert!(source.contains("update_ema_multicoin_dual_side_hsl("));
        assert!(source.contains("write_dual_side_hsl_outputs("));
        assert!(source.contains("write_dual_side_coin_hsl_outputs("));
        assert!(source.contains("long_coin_overrides, short_coin_overrides"));
        assert!(source.contains("net_position_cost -= short_side.psize[c]"));
        assert!(source.contains(
            "float twe_abs = fabs(net_position_cost / account.balance)"
        ));
        assert!(source.contains("account.balance = 0.0f"));
        assert!(source.contains("alive || hsl_validation_failed"));
        assert!(source.contains("bool any_unstuck_enabled = false"));
        assert!(source.contains("&& !any_unstuck_enabled"));
        assert!(source.contains(
            "const EmaMulticoinSideConfig config = load_ema_multicoin_side_config(params, po)"
        ));
        assert!(source.contains(
            "init_ema_multicoin_side_state(\n        side, config, bars, coin_settings, coin_overrides, C"
        ));
        assert!(source.contains(
            "update_ema_multicoin_side_indicators(\n            side, config, bars, coin_settings, k, C, start_hour_minute"
        ));
        assert!(source.contains(
            "bool any_fill = process_ema_multicoin_side_fills("
        ));
        assert!(source.contains(
            "update_ema_multicoin_side_selection(\n                side, config, bars, coin_settings, coin_overrides"
        ));
        assert!(source.contains(
            "generate_ema_multicoin_side_orders(\n                side, config, account"
        ));
        assert!(source.contains("side.max_tradable_seen = max("));
        assert!(source.contains(
            "side.previous_effective_n_positions = effective_n_positions"
        ));
        assert!(source.contains("EmaMulticoinSideState side"));
        assert!(source.contains("thread HslState& hsl = side.hsl"));
        assert!(source.contains("thread HslState* coin_hsl = side.coin_hsl"));
        assert!(source.contains("thread float* psize = side.psize"));
        assert!(source.contains("thread int* entry_tick = side.entry_tick"));
        assert!(source.contains("thread bool* selected = side.selected"));
        assert!(source.contains("load_hsl(params, po, 31)"));
        assert!(source.contains("write_one_side_hsl_outputs("));
        assert!(source.contains("record_hsl_panic_fill("));
        assert_eq!(
            MPS_EMA_ANCHOR_MULTICOIN_BODY
                .matches("record_coin_hsl_realized_fill(")
                .count(),
            2
        );
        assert_eq!(
            MPS_EMA_ANCHOR_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_close_fill(")
                .count(),
            1
        );
        assert_eq!(
            MPS_EMA_ANCHOR_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_entry_fill(")
                .count(),
            1
        );
        assert!(source.contains("coin_hsl_eligibility_changed"));
        assert!(source.contains("coin_hsl_entry_blocked_mask"));
        assert!(source.contains("market_panic ? taker_fee : maker_fee"));
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
        assert!(source.contains("device float* coin_fill_counts"));
        assert_eq!(
            source
                .matches("coin_fill_counts[candidate_index * coin_count + c] += 1.0f")
                .count(),
            2
        );
        assert!(source.contains("close_tick[c] <= fill_ticks[tick_offset + 0]"));
        assert!(source.contains("entry_tick[c] > fill_ticks[tick_offset + 1]"));
        assert!(source.contains("config.volume_drop = clamp(params[po + 14]"));
        assert!(source.contains("effective_n_positions"));
        assert!(source.contains(
            "const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f"
        ));
        assert!(source.contains("if (alive && !post_fill_balance_depleted)"));
        assert!(source.contains("const float score_hysteresis = fmax(run_settings[4], 0.0f)"));
        assert!(source.contains("incumbent[c] = selected[c] && psize[c] <= 0.0f"));
        assert!(source.contains("if (!selected[c] || incumbent[c] || !survivor[c]) continue"));
        assert!(source.contains("score[challenger] - score[incumbent_coin]"));
        assert!(source.contains("allowed_wallet_exposure_limit"));
        assert!(source.contains("twel_entry_gate_enabled"));
        assert!(source.contains("twel_enforcer_enabled"));
        assert!(source.contains("twel_enforcer_reduce_portfolio"));
        assert!(source.contains("clamped_market_price"));
        assert!(source.contains("secondary_close_qty"));
        assert!(source.contains("realized_loss_proxy_allows_close"));
        assert!(source.contains("const bool loss_gate_enabled = run_settings[5] < 1.0f"));
        assert!(source.contains("current_effective_n_positions"));
        assert!(source.contains("distance == best_distance && c > best"));
        assert!(source.contains("= fma("));
        assert!(source.contains("one global auto-unstuck intent"));
        assert!(source.contains("unstuck_loss_allowance_pct"));
        assert!(!source.contains("hard_stop"));
        assert_eq!(source, mps_ema_anchor_multicoin_long_source());
    }

    #[test]
    fn trailing_martingale_mps_source_exposes_expected_kernel_contract() {
        let source = mps_trailing_martingale_source();
        assert_shared_hsl_contract(source);
        assert_directional_hsl_accounting_contract(source);
        assert!(source.contains("kernel void passivbot_trailing_martingale"));
        assert!(source.contains("constant int SIDE_PARAMS = 51"));
        assert!(source.contains("struct HslState"));
        assert!(source.contains("update_hsl("));
        assert!(source.contains("try_restart_hsl("));
        assert!(source.contains("constant int SCALAR_COLS = 62"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[so + 44] = loss_sum"));
        assert!(source.contains("scalars[so + 45] = position_unchanged_max_min * interval_ms"));
        assert!(source.contains("scalars[so + 46] = long_enabled"));
        assert!(source.contains("scalars[so + 47] = short_enabled"));
        assert!(source.contains("scalars[so + 48] = total_wallet_exposure_max"));
        assert!(source.contains("scalars[so + 49] = total_wallet_exposure_mean"));
        assert!(source.contains("scalars[so + 50] = fill_count"));
        assert!(source.contains("scalars[so + 51] = fill_count_entry"));
        assert!(source.contains("scalars[so + 52] = fill_count_long"));
        assert!(source.contains("scalars[so + 53] = fills_active_days_count"));
        assert!(source.contains("scalars[so + 54] = pnl_recovery_max_min * interval_ms"));
        assert!(source.contains("scalars[so + 55] = held_sum_min * interval_ms"));
        assert!(source.contains("scalars[so + 56] = held_count"));
        assert!(source.contains("scalars[so + 57] = account_recovery_max_min * interval_ms"));
        assert!(source.contains("scalars[so + 58] = profit_sum_long"));
        assert!(source.contains("scalars[so + 61] = loss_sum_short"));
        assert!(source.contains("if (eqf >= account_peak)"));
        assert!(source.contains("hsl_tier_samples_total"));
        assert!(source.contains("h.restart_retrigger_count"));
        assert!(source.contains("h.halt_duration_sum_steps"));
        assert!(source.contains("record_hsl_panic_fill("));
        assert!(source.contains("market_panic ? taker_fee : maker_fee"));
        assert!(source.contains("h.panic_loss_drawdown_sum"));
        assert!(source.contains("&& !long_side.close_is_panic"));
        assert!(source.contains("&& !short_side.close_is_panic"));
        assert!(source.contains("if (!long_side.close_is_panic"));
        assert!(source.contains("if (!short_side.close_is_panic"));
        assert!(source.contains("h.slot_count"));
        assert!(source.contains("h.no_restart_peak_strategy_equity"));
        assert!(source.contains("terminal || h.cooldown_minutes <= 0.0f"));
        assert!(source.contains("h.pending_stop_k + h.cooldown_minutes"));
        assert!(source.contains("load_hsl(params, po, 40)"));
        assert!(source.contains("load_hsl(params, po + SIDE_PARAMS, 40)"));
        assert!(source.contains("const float cmp_eps = 1.0e-12f"));
        assert_eq!(source.matches("0.9999999403953552f").count(), 3);
        assert!(source.contains("unstuck_reducer_qty"));
        assert!(source.contains("unstuck_ema_gating_enabled"));
        assert!(source.contains("unstuck_loss_allowance_pct"));
        assert!(source.contains("grid_source.unstuck_enabled = false"));
        assert!(source.contains("min_since_open"));
        assert!(source.contains("max_since_min"));
        assert!(source.contains("max_since_open"));
        assert!(source.contains("min_since_max"));
        assert!(source.contains("if (we_if <= s.entry_cap * 1.01f) return qty"));
        assert!(source.contains("s.entry_cap * balance - cost"));
        assert!(source.contains("s.allowed_wel"));
        assert!(source.contains("s.wel_enforcer_enabled"));
        assert!(source.contains("wel_target"));
        assert!(source.contains("s.twel_enforcer_enabled"));
        assert!(source.contains("twel_target"));
        assert!(source.contains("price_now * 0.9995f / price_step"));
        assert!(source.contains("finalized_wel_qty = finalized_reducer_qty"));
        assert!(source.contains("finalized_reducer_qty_with_ordinary"));
        assert!(source.contains("float finalized_twel_qty = ordinary_can_accompany_reducer"));
        assert!(source.contains("finalized_twel_qty > finalized_wel_qty"));
        assert!(source.contains("float reducer_qty = use_unstuck"));
        assert!(source.contains("? unstuck_qty : (use_twel ? twel_qty : wel_qty)"));
        assert!(source.contains("secondary_close_qty"));
        assert!(source.contains("twel_entry_gate_enabled"));
        assert_eq!(source.matches("= fma(").count(), 6);
        assert!(!source.contains("alpha0 * close +"));
        assert_eq!(source.matches("const int fo = k * 11").count(), 1);
        assert_eq!(source.matches("const int touch_down_tick").count(), 1);
        assert_eq!(source.matches("const int touch_up_tick").count(), 1);
        assert_eq!(source.matches("high_fill_max_tick").count(), 9);
        assert_eq!(source.matches("low_nonfill_max_tick").count(), 9);
        assert!(!source.contains("nextafter("));
        assert!(source.contains("int cticks = touch_controls ? touch_nearest_ticks : target_ticks"));
        assert!(source.contains("remainder == mq && mq_relation > 0"));
        assert!(!source.contains("entry_raw_touch"));
        assert!(!source.contains("close_raw_touch"));
        assert!(source.contains("close_touch > target_ticks"));
        assert!(source.contains("close_touch < target_ticks"));
        assert!(source.contains("touch_down_ticks >= band_ticks"));
        assert!(source.contains("touch_up_ticks <= band_ticks"));
        assert!(source.contains("touch_down_ticks >= raw_reentry_ticks"));
        assert!(source.contains("touch_up_ticks <= raw_reentry_ticks"));
        assert!(source.contains("entry_gen_balance"));
        assert!(source.contains("close_gen_balance"));
        assert!(source.contains("recursive_close_groups"));
        assert!(source.contains("realized_loss_proxy_allows_close"));
        assert!(source.contains("const float max_realized_loss_pct = settings[14]"));
        assert!(source.contains("const bool loss_gate_enabled = max_realized_loss_pct < 1.0f"));
        assert!(source.contains("const float taker_fee = settings[15]"));
        assert!(source.contains("const float market_order_slippage_pct = fmax(settings[16], 0.0f)"));
        assert!(source.contains("const bool long_hsl_panic_market = settings[17] > 0.5f"));
        assert!(source.contains("const bool short_hsl_panic_market = settings[18] > 0.5f"));
        assert!(source.contains("market_panic ? taker_fee : maker_fee"));
        assert!(source.contains("close * (1.0f - market_order_slippage_pct)"));
        assert!(source.contains("close * (1.0f + market_order_slippage_pct)"));
        assert!(source.contains("the proxy uses a zero-loss envelope"));
        assert!(source.contains("int grid_rung_limit = strategy_wel_qty > 0.0f ? 499 : 500"));
        assert!(source.contains("long_side.close_gen_psize - strategy_wel_qty"));
        assert!(source.contains("short_side.close_gen_psize - strategy_wel_qty"));
        assert!(source.contains("(use_twel || use_unstuck) && wel_qty <= 0.0f"));
        assert!(source.contains("long_side.secondary_close_price"));
        assert!(source.contains("short_side.secondary_close_price"));
        assert!(source.contains("dust_remainder"));
        assert!(!source.contains("float primary_diff = fabs"));
        assert_eq!(source.matches("bool reducer_before_group").count(), 2);
        assert!(source.contains("long_scan_close_grid = long_scan_close_grid"));
        assert!(source.contains("short_scan_close_grid = short_scan_close_grid"));
        assert!(source.contains("const bool filter_by_min_effective_cost"));
        assert!(source.contains("passes_min_effective_cost"));
        assert!(source.contains("projected_cost_lower"));
        assert!(source.contains("float guaranteed_balance_lower"));
        assert!(!source.contains("accumulate_min_cost_balance_error"));
        assert!(source.contains("for (int rung = 0; rung < 500; ++rung)"));
        assert!(source.contains("cooldown_min != 0.0f"));
        assert!(!source.contains("price_now > rounded_target"));
        assert!(!source.contains("price_now < rounded_target"));
        assert!(!source.contains("nearest_ticks("));
    }

    #[test]
    fn trailing_martingale_multicoin_mps_source_exposes_expected_kernel_contract() {
        let source = mps_trailing_martingale_multicoin_source();
        assert_shared_hsl_contract(source);
        assert_shared_multicoin_contract(source);
        assert!(source.contains("kernel void passivbot_trailing_martingale_multicoin"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 59"));
        assert!(source.contains("constant int OVERRIDE_COLS = 44"));
        assert!(source.contains("constant int HSL_OVERRIDE_START = 34"));
        assert!(source.contains("apply_coin_hsl_overrides("));
        assert!(source.contains("constant int COIN_COLS = 12"));
        assert!(source.contains("constant int SCALAR_COLS = 57"));
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
        assert!(source.contains("accumulate_tm_multicoin_side_unrealized_pnl("));
        assert!(source.contains("update_tm_multicoin_side_indicators("));
        assert!(source.contains("count_tm_multicoin_tradable_coins("));
        assert!(source.contains("tm_multicoin_side_has_position("));
        assert!(source.contains("tm_multicoin_side_held_marks_are_valid("));
        assert!(source.contains("tm_multicoin_side_has_blocking_orders("));
        assert!(source.contains("update_tm_multicoin_dual_side_hsl("));
        assert!(source.contains("update_tm_multicoin_side_selection("));
        assert!(source.contains("generate_tm_multicoin_side_orders("));
        assert!(source.contains(
            "const TrailingMartingaleMulticoinSideConfig config ="
        ));
        assert!(source.contains(
            "init_trailing_martingale_multicoin_side_state(\n        side, config, bars, coin_settings, coin_overrides, C"
        ));
        assert!(source.contains(
            "update_tm_multicoin_side_indicators(\n            side, config, bars, coin_settings, k, C, start_hour_minute"
        ));
        assert!(source.contains(
            "bool any_fill = process_tm_multicoin_side_fills(\n            side, config, account, fills"
        ));
        assert!(source.contains(
            "update_tm_multicoin_side_selection(\n                side, config, bars, coin_settings, coin_overrides"
        ));
        assert!(source.contains(
            "generate_tm_multicoin_side_orders(\n                side, config, account"
        ));
        assert!(source.contains("TrailingMartingaleMulticoinSideState side"));
        assert!(source.contains("TrailingMartingaleMulticoinFillState fills"));
        assert!(source.contains("thread HslState& hsl = side.hsl"));
        assert!(source.contains("thread HslState* coin_hsl = side.coin_hsl"));
        assert!(source.contains("thread float* psize = side.psize"));
        assert!(source.contains("thread int* entry_tick = side.entry_tick"));
        assert!(source.contains("thread bool* selected = side.selected"));
        assert!(source.contains("thread int& max_tradable_seen = side.max_tradable_seen"));
        assert!(source.contains("side.previous_effective_n_positions"));
        assert!(source.contains("load_hsl(params, po, 48)"));
        assert!(source.contains("write_one_side_hsl_outputs("));
        assert!(source.contains("record_hsl_panic_fill("));
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("record_tm_multicoin_close_fill(")
                .count(),
            4
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("record_tm_multicoin_entry_fill(")
                .count(),
            2
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("finalize_tm_multicoin_close_position(")
                .count(),
            2
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("apply_tm_multicoin_entry_position(")
                .count(),
            2
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("update_tm_multicoin_position_fill_timestamp(")
                .count(),
            2
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("record_coin_hsl_realized_fill(")
                .count(),
            2
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_close_fill(")
                .count(),
            1
        );
        assert_eq!(
            MPS_TRAILING_MARTINGALE_MULTICOIN_BODY
                .matches("advance_coin_hsl_equity_after_entry_fill(")
                .count(),
            1
        );
        assert!(source.contains("coin_hsl_eligibility_changed"));
        assert!(source.contains("coin_hsl_entry_blocked_mask"));
        assert!(source.contains("market_panic ? taker_fee : maker_fee"));
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
        assert!(source.contains("device float* coin_fill_counts"));
        assert_eq!(
            source
                .matches("coin_fill_counts[int(b) * C + c] += 1.0f")
                .count(),
            0
        );
        assert_eq!(
            source
                .matches(
                    "coin_fill_counts[candidate_index * coin_count + coin] += 1.0f"
                )
                .count(),
            2
        );
        assert!(source.contains("coin_wel_enforcer_enabled"));
        assert!(source.contains("coin_wel_enforcer_threshold"));
        assert!(source.contains("twel_enforcer_enabled"));
        assert!(source.contains("twel_enforcer_reduce_portfolio"));
        assert!(source.contains("twel_close_qty"));
        assert!(source.contains("realized_loss_proxy_allows_close"));
        assert!(source.contains("const bool loss_gate_enabled = run_settings[5] < 1.0f"));
        assert!(source.contains(
            "const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f"
        ));
        assert!(source.contains("if (alive && !post_fill_balance_depleted)"));
        assert!(source.contains("market_price * 0.9995f / price_step"));
        assert_eq!(source.matches("clamped_market_price(").count(), 3);
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
        assert!(source.contains("float reserved_grid_qty = use_unstuck"));
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
