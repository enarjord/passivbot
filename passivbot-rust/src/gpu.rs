//! Rust-owned GPU screening assets.
//!
//! GPU screening is intentionally approximate, but strategy-adjacent programs
//! still live behind the Rust ownership boundary. Python only compiles and
//! dispatches this source through the optional Apple MPS backend.

pub const MPS_EMA_ANCHOR_SOURCE: &str = include_str!("gpu/mps_ema_anchor.metal");

pub fn mps_ema_anchor_source() -> &'static str {
    MPS_EMA_ANCHOR_SOURCE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_anchor_mps_source_exposes_expected_kernel_contract() {
        let source = mps_ema_anchor_source();
        assert!(source.contains("kernel void passivbot_ema_anchor"));
        assert!(source.contains("constant int DAILY_COLS = 5"));
        assert!(source.contains("constant int SCALAR_COLS = 15"));
    }
}
