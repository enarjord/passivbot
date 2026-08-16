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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_anchor_mps_source_exposes_expected_kernel_contract() {
        let source = mps_ema_anchor_source();
        assert!(source.contains("kernel void passivbot_ema_anchor"));
        assert!(source.contains("constant int DAILY_COLS = 5"));
        assert!(source.contains("constant int SCALAR_COLS = 18"));
        assert!(source.contains("generate_short_orders"));
        assert!(source.contains("const bool hedge_mode"));
        assert_eq!(source.matches("= fma(").count(), 5);
        assert!(!source.contains("alpha0 * close +"));
        assert_eq!(source.matches("const int fo = k * 11").count(), 1);
        assert_eq!(source.matches("const int touch_down_tick").count(), 1);
        assert_eq!(source.matches("const int touch_up_tick").count(), 1);
        assert_eq!(source.matches("high_fill_max_tick").count(), 3);
        assert_eq!(source.matches("low_nonfill_max_tick").count(), 3);
        assert!(!source.contains("nextafter("));
    }

    #[test]
    fn ema_anchor_multicoin_mps_source_exposes_expected_kernel_contract() {
        let source = mps_ema_anchor_multicoin_source();
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin"));
        assert!(source.contains("kernel void passivbot_ema_anchor_multicoin_long"));
        assert!(source.contains("const bool short_side"));
        assert!(source.contains("constant int MAX_COINS = 64"));
        assert!(source.contains("constant int PARAM_COLS = 19"));
        assert!(source.contains("constant int COIN_COLS = 11"));
        assert!(source.contains("constant int DAILY_COLS = 5"));
        assert!(source.contains("constant int SCALAR_COLS = 18"));
        assert!(source.contains("close_tick[c] <= fill_ticks[tick_offset + 0]"));
        assert!(source.contains("entry_tick[c] > fill_ticks[tick_offset + 1]"));
        assert!(source.contains("const float volume_drop = clamp(params[po + 14]"));
        assert!(source.contains("effective_n_positions"));
        assert!(source.contains("float total_cap = twel - 1.0e-7f"));
        assert!(source.contains("= fma("));
        assert!(!source.contains("unstuck"));
        assert!(!source.contains("hard_stop"));
        assert_eq!(source, mps_ema_anchor_multicoin_long_source());
    }

    #[test]
    fn trailing_martingale_mps_source_exposes_expected_kernel_contract() {
        let source = mps_trailing_martingale_source();
        assert!(source.contains("kernel void passivbot_trailing_martingale"));
        assert!(source.contains("constant int SIDE_PARAMS = 27"));
        assert!(source.contains("min_since_open"));
        assert!(source.contains("max_since_min"));
        assert!(source.contains("max_since_open"));
        assert!(source.contains("min_since_max"));
        assert!(source.contains("if (we_if <= s.twel * 1.01f) return qty"));
        assert!(source.contains("s.twel * balance - cost"));
        assert_eq!(source.matches("= fma(").count(), 5);
        assert!(!source.contains("alpha0 * close +"));
        assert_eq!(source.matches("const int fo = k * 11").count(), 1);
        assert_eq!(source.matches("const int touch_down_tick").count(), 1);
        assert_eq!(source.matches("const int touch_up_tick").count(), 1);
        assert_eq!(source.matches("high_fill_max_tick").count(), 6);
        assert_eq!(source.matches("low_nonfill_max_tick").count(), 6);
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
        assert!(source.contains("for (int rung = 0; rung < 500; ++rung)"));
        assert!(source.contains("cooldown_min != 0.0f"));
        assert!(!source.contains("price_now > rounded_target"));
        assert!(!source.contains("price_now < rounded_target"));
        assert!(!source.contains("nearest_ticks("));
    }
}
