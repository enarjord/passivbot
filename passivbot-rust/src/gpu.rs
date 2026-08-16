//! Rust-owned GPU screening assets.
//!
//! GPU screening is intentionally approximate, but strategy-adjacent programs
//! still live behind the Rust ownership boundary. Python only compiles and
//! dispatches this source through the optional Apple MPS backend.

pub const MPS_EMA_ANCHOR_SOURCE: &str = include_str!("gpu/mps_ema_anchor_directional.metal");
pub const MPS_TRAILING_MARTINGALE_SOURCE: &str =
    include_str!("gpu/mps_trailing_martingale_directional.metal");

pub fn mps_ema_anchor_source() -> &'static str {
    MPS_EMA_ANCHOR_SOURCE
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
        assert_eq!(source.matches("const int fo = k * 6").count(), 1);
        assert_eq!(source.matches("high_fill_max_tick").count(), 3);
        assert_eq!(source.matches("low_nonfill_max_tick").count(), 3);
        assert_eq!(source.matches("nextafter(tick_value, INFINITY)").count(), 1);
        assert_eq!(source.matches("nextafter(tick_value, -INFINITY)").count(), 1);
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
        assert_eq!(source.matches("const int fo = k * 6").count(), 1);
        assert_eq!(source.matches("high_fill_max_tick").count(), 3);
        assert_eq!(source.matches("low_nonfill_max_tick").count(), 3);
        assert_eq!(source.matches("nextafter(tick_value, INFINITY)").count(), 1);
        assert_eq!(source.matches("nextafter(tick_value, -INFINITY)").count(), 1);
        assert_eq!(source.matches("int entry_touch = nearest_ticks").count(), 1);
    }
}
