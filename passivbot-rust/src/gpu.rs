//! Rust-owned GPU screening assets.
//!
//! GPU screening is intentionally approximate, but strategy-adjacent programs
//! still live behind the Rust ownership boundary. Python only compiles and
//! dispatches this source through the optional Apple MPS backend.

pub const MPS_EMA_ANCHOR_SOURCE: &str = include_str!("gpu/mps_ema_anchor_directional.metal");
pub const MPS_EMA_ANCHOR_MULTICOIN_SOURCE: &str =
    include_str!("gpu/mps_ema_anchor_multicoin_long.metal");
pub const MPS_EMA_ANCHOR_MULTICOIN_LONG_SOURCE: &str = MPS_EMA_ANCHOR_MULTICOIN_SOURCE;
pub const MPS_TRAILING_MARTINGALE_SOURCE: &str =
    include_str!("gpu/mps_trailing_martingale_directional.metal");
pub const MPS_TRAILING_MARTINGALE_MULTICOIN_SOURCE: &str =
    include_str!("gpu/mps_trailing_martingale_multicoin.metal");

pub fn mps_ema_anchor_source() -> &'static str {
    MPS_EMA_ANCHOR_SOURCE
}

pub fn mps_ema_anchor_multicoin_source() -> &'static str {
    MPS_EMA_ANCHOR_MULTICOIN_SOURCE
}

pub fn mps_ema_anchor_multicoin_long_source() -> &'static str {
    MPS_EMA_ANCHOR_MULTICOIN_LONG_SOURCE
}

pub fn mps_trailing_martingale_source() -> &'static str {
    MPS_TRAILING_MARTINGALE_SOURCE
}

pub fn mps_trailing_martingale_multicoin_source() -> &'static str {
    MPS_TRAILING_MARTINGALE_MULTICOIN_SOURCE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_anchor_mps_source_exposes_expected_kernel_contract() {
        let source = mps_ema_anchor_source();
        assert!(source.contains("kernel void passivbot_ema_anchor"));
        assert!(source.contains("constant int DAILY_COLS = 5"));
        assert!(source.contains("constant int SCALAR_COLS = 45"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[so + 44] = loss_sum"));
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
        assert!(source.contains("fmax(params[po + 25], 1.0f)"));
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
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin"));
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin_long"));
        assert!(source.contains("const bool short_side"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 31"));
        assert!(source.contains("constant int OVERRIDE_COLS = 19"));
        assert!(source.contains("coin_override_or"));
        assert!(source.contains("constant int COIN_COLS = 11"));
        assert!(source.contains("constant int DAILY_COLS = 6"));
        assert!(source.contains("day_min_balance"));
        assert!(source.contains("constant int SCALAR_COLS = 20"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[scalar_offset + 19] = loss_sum"));
        assert!(source.contains("close_tick[c] <= fill_ticks[tick_offset + 0]"));
        assert!(source.contains("entry_tick[c] > fill_ticks[tick_offset + 1]"));
        assert!(source.contains("const float volume_drop = clamp(params[po + 14]"));
        assert!(source.contains("effective_n_positions"));
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
        assert!(source.contains("kernel void passivbot_trailing_martingale"));
        assert!(source.contains("constant int SIDE_PARAMS = 51"));
        assert!(source.contains("struct HslState"));
        assert!(source.contains("update_hsl("));
        assert!(source.contains("try_restart_hsl("));
        assert!(source.contains("constant int SCALAR_COLS = 45"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[so + 44] = loss_sum"));
        assert!(source.contains("hsl_tier_samples_total"));
        assert!(source.contains("h.restart_retrigger_count"));
        assert!(source.contains("h.halt_duration_sum_steps"));
        assert!(source.contains("record_hsl_panic_fill("));
        assert!(source.contains("h.panic_loss_drawdown_sum"));
        assert!(source.contains("&& !long_side.close_is_panic"));
        assert!(source.contains("&& !short_side.close_is_panic"));
        assert!(source.contains("if (!long_side.close_is_panic"));
        assert!(source.contains("if (!short_side.close_is_panic"));
        assert!(source.contains("h.slot_count"));
        assert!(source.contains("h.no_restart_peak_strategy_equity"));
        assert!(source.contains("terminal || h.cooldown_minutes <= 0.0f"));
        assert!(source.contains("h.pending_stop_k + h.cooldown_minutes"));
        assert!(source.contains("fmax(params[po + 42], 1.0f)"));
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
        assert!(source.contains("kernel void passivbot_trailing_martingale_multicoin"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 48"));
        assert!(source.contains("constant int OVERRIDE_COLS = 34"));
        assert!(source.contains("constant int SCALAR_COLS = 20"));
        assert!(source.contains("record_gross_pnl"));
        assert!(source.contains("scalars[scalar_offset + 19] = loss_sum"));
        assert!(source.contains("coin_wel_enforcer_enabled"));
        assert!(source.contains("coin_wel_enforcer_threshold"));
        assert!(source.contains("twel_enforcer_enabled"));
        assert!(source.contains("twel_enforcer_reduce_portfolio"));
        assert!(source.contains("twel_close_qty"));
        assert!(source.contains("realized_loss_proxy_allows_close"));
        assert!(source.contains("const bool loss_gate_enabled = run_settings[5] < 1.0f"));
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
