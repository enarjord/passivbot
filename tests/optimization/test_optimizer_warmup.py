"""Regression tests for optimizer warmup stamping.

Covers the bug where prepare_hlcvs_mss stamped mss[coin]["warmup_minutes"]
from the template bot's decorative values (e.g. entry_volatility_ema_span_hours)
instead of the worst case the optimizer's search space can actually produce.
"""

import numpy as np

from backtest import ensure_valid_index_metadata
from config_utils import get_template_config
from warmup_utils import compute_per_coin_warmup_minutes


def _make_optimizer_config() -> dict:
    """Build a config where template bot and bounds disagree on warmup.

    Template bot has entry_volatility_ema_span_hours = 1690, which would
    produce warmup = 1690 * 60 * 0.3 = 30420 min under
    compute_per_coin_warmup_minutes.

    Bounds pin that field to [0, 0] and cap ema_span_0 at 100, so the
    largest warmup the optimizer can actually produce is 100 * 0.3 = 30 min.
    """
    config = get_template_config()
    config["live"]["warmup_ratio"] = 0.3
    config["live"]["max_warmup_minutes"] = 0
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["backtest"]["exchanges"] = ["combined"]
    config["backtest"]["coins"] = {"combined": ["HYPE"]}

    long_bot = config["bot"]["long"]
    long_bot["ema_span_0"] = 770.0
    long_bot["ema_span_1"] = 210.0
    long_bot["forager_volume_ema_span"] = 520.0
    long_bot["forager_volatility_ema_span"] = 225.0
    long_bot["entry_volatility_ema_span_hours"] = 1690.0

    short_bot = config["bot"]["short"]
    short_bot["ema_span_0"] = 1.0
    short_bot["ema_span_1"] = 1.0
    short_bot["forager_volume_ema_span"] = 0.0
    short_bot["forager_volatility_ema_span"] = 0.0
    short_bot["entry_volatility_ema_span_hours"] = 0.0

    bounds = config["optimize"]["bounds"]
    bounds["long_ema_span_0"] = [1, 100, 1]
    bounds["long_ema_span_1"] = [1, 100, 1]
    bounds["long_forager_volatility_ema_span"] = [0, 0]
    bounds["long_forager_volume_ema_span"] = [0, 0]
    bounds["long_entry_volatility_ema_span_hours"] = [0, 0]
    bounds["short_ema_span_0"] = [1, 100, 1]
    bounds["short_ema_span_1"] = [1, 100, 1]
    bounds["short_forager_volatility_ema_span"] = [0, 0]
    bounds["short_forager_volume_ema_span"] = [0, 0]
    bounds["short_entry_volatility_ema_span_hours"] = [0, 0]
    return config


def _dummy_hlcvs(shape: tuple[int, int, int]) -> np.ndarray:
    """Return a finite OHLCV-shaped array so ensure_valid_index_metadata
    treats every bar as valid."""
    return np.ones(shape, dtype=np.float64)


def test_stamp_optimizer_warmup_uses_bounds_when_template_bot_exceeds_them():
    """Bug regression: the optimizer must size warmup from bounds, not the
    decorative template bot values."""
    from optimize import _stamp_optimizer_warmup  # noqa: PLC0415 — fails loudly if missing

    config = _make_optimizer_config()
    mss = {"HYPE": {"first_valid_index": 0, "last_valid_index": 180000}}

    # Simulate prepare_hlcvs_mss's template-derived stamping to establish
    # the buggy baseline we're repairing.
    template_warmup_map = compute_per_coin_warmup_minutes(config)
    ensure_valid_index_metadata(mss, _dummy_hlcvs((180001, 1, 4)), ["HYPE"], template_warmup_map)
    assert mss["HYPE"]["warmup_minutes"] == 30420, (
        "baseline sanity check: compute_per_coin_warmup_minutes on the "
        "template config should produce the buggy 30420-minute warmup "
        "(1690h * 60 * 0.3). If this assertion fails, the test's "
        "assumptions about the template/bounds disagreement no longer hold."
    )

    # The fix: after _stamp_optimizer_warmup runs, the stamped warmup
    # reflects the max the optimizer's search space can produce
    # (ema_span_0 ∈ [1, 100] ⇒ 100 * 0.3 = 30 min).
    _stamp_optimizer_warmup(config, mss, ["HYPE"])

    assert mss["HYPE"]["warmup_minutes"] == 30
    assert mss["HYPE"]["trade_start_index"] == 30


def test_stamp_optimizer_warmup_respects_last_valid_index_cap():
    """trade_start_index must never exceed last_valid_index, even if the
    bounds-derived warmup would push it past the end of the data."""
    from optimize import _stamp_optimizer_warmup  # noqa: PLC0415

    config = _make_optimizer_config()
    config["optimize"]["bounds"]["long_ema_span_0"] = [1, 100000, 1]
    mss = {"HYPE": {"first_valid_index": 0, "last_valid_index": 10}}

    _stamp_optimizer_warmup(config, mss, ["HYPE"])

    # Warmup = 100000 * 0.3 = 30000 min. Clamp to last_valid_index = 10.
    assert mss["HYPE"]["warmup_minutes"] == 30000
    assert mss["HYPE"]["trade_start_index"] == 10


def test_stamp_optimizer_warmup_is_wired_into_register_exchange_data():
    """Future-proof the fix: if a refactor silently drops the call from
    _register_exchange_data, the regression tests above still pass because
    they exercise _stamp_optimizer_warmup directly. This test pins the
    wiring so a missing call site fails a test rather than re-introducing
    the bug."""
    import inspect

    from optimize import _register_exchange_data

    source = inspect.getsource(_register_exchange_data)
    assert "_stamp_optimizer_warmup(" in source, (
        "_register_exchange_data must call _stamp_optimizer_warmup. "
        "If this test fails, the warmup-from-bounds fix has regressed: "
        "the optimizer will size per-coin warmup from template bot values "
        "again instead of from optimize.bounds."
    )
