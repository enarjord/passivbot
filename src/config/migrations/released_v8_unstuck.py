"""Frozen released-v8 unstuck paths, used only by explicit version conversion.

Current native protocol identifiers must not reinterpret released authored inputs.
The target paths are public config fields; all values remain unchanged.
"""

from types import MappingProxyType


RELEASED_V8_UNSTUCK_FLAT_PATHS = MappingProxyType({
    "unstuck_close_pct": ("unstuck", "close_fraction"),
    "unstuck_ema_dist": ("unstuck", "ema_distance_fraction"),
    "unstuck_ema_span_0": ("unstuck", "ema_span_0_minutes"),
    "unstuck_ema_span_1": ("unstuck", "ema_span_1_minutes"),
    "unstuck_loss_allowance_pct": ("unstuck", "loss_allowance_fraction"),
    "unstuck_threshold": ("unstuck", "exposure_trigger_ratio"),
})
