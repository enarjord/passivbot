"""Offline end-to-end replay through the canonical config loader and Rust backtester."""

from copy import deepcopy
import numpy as np
from config import get_template_config, prepare_config
from config.overrides import parse_overrides
from backtest import run_backtest


def _legacy_replay_config():
    c = get_template_config()
    c["config_version"] = "v8.2.0"
    for root in (c["bot"], c["optimize"]["bounds"]):
        for side in ("long", "short"):
            strategy = root[side]["strategy"]["trailing_martingale"]
            for key in ("ema_span_0", "ema_span_1"):
                strategy[key] = strategy["entry"].pop(key)
    for side in ("long", "short"):
        for key in ("ema_span_0", "ema_span_1"):
            c["bot"][side]["unstuck"].pop(key)
            c["optimize"]["bounds"][side]["unstuck"].pop(key)
    c["backtest"].update(
        start_date="2024-01-01",
        end_date="2024-01-15",
        starting_balance=1000,
        exchanges=["binance"],
        coins={"binance": ["BTC", "ETH"]},
        suite_enabled=False,
        candle_interval_minutes=1,
    )
    c["live"].update(
        approved_coins={"long": ["BTC", "ETH"], "short": []},
        ignored_coins={"long": [], "short": []},
        warmup_ratio=1,
        max_warmup_minutes=100,
    )
    for side in ("long", "short"):
        b = c["bot"][side]
        b["risk"].update(
            n_positions=2,
            total_wallet_exposure_limit=1.0 if side == "long" else 0.0,
            entry_cooldown_minutes=0,
            we_excess_allowance_pct=0,
            position_exposure_enforcer_enabled=False,
            total_exposure_enforcer_enabled=False,
        )
        b["hsl"]["enabled"] = False
        b["forager"].update(volume_ema_span_1m=10, volatility_ema_span_1m=10)
        t = b["strategy"]["trailing_martingale"]
        t.update(
            ema_span_0=10.5,
            ema_span_1=40.5,
            volatility_ema_span_1m=1,
            volatility_ema_span_1h=1,
        )
        t["entry"].update(
            initial_qty_pct=0.25,
            initial_ema_dist=-0.001,
            double_down_factor=1,
            ema_gate_mode="all",
            threshold_base_pct=0.015,
            retracement_base_pct=0,
        )
        t["close"].update(
            qty_pct=0.2,
            threshold_base_pct=0.01,
            retracement_base_pct=0,
            threshold_we_weight=0,
        )
        for group in ("entry", "close"):
            for key in t[group]:
                if key.endswith("_weight"):
                    t[group][key] = 0
        b["unstuck"].update(
            enabled=True,
            ema_gating_enabled=True,
            close_pct=0.1,
            ema_dist=-0.005,
            loss_allowance_pct=0.02,
            threshold=0.3,
        )
    c["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {
                            "ema_span_0": 21.5,
                            "ema_span_1": 110.25,
                        }
                    }
                }
            }
        }
    }
    return c


def _replay(config):
    n = 20_000
    t = np.arange(n)
    close = 100 + 4 * np.sin(t / 23) + 8 * np.sin(t / 510) - 12 * (t / n)
    hlcvs = np.empty((n, 2, 4))
    for i in range(2):
        prices = close * (1 + 0.05 * i) + np.sin(t / 41 + i)
        hlcvs[:, i, 0] = prices + 0.3
        hlcvs[:, i, 1] = prices - 0.3
        hlcvs[:, i, 2] = prices
        hlcvs[:, i, 3] = 1000
    start = 1704067200000
    ts = start + t * 60000
    mss = {
        coin: dict(
            maker=0.0002,
            taker=0.0005,
            qty_step=0.001,
            price_step=0.01,
            min_qty=0.001,
            min_cost=1,
            c_mult=1,
            first_valid_index=0,
            last_valid_index=n - 1,
            warmup_minutes=100,
        )
        for coin in ("BTC", "ETH")
    }
    mss["__meta__"] = {
        "requested_start_ts": start + 100 * 60000,
        "requested_start_date": "2024-01-01",
        "warmup_minutes_requested": 100,
    }
    config["backtest"]["coins"] = {"binance": ["BTC", "ETH"]}
    return run_backtest(hlcvs, mss, config, "binance", np.full(n, 50000.0), ts)


def test_migrated_replay_matches_explicit_legacy_pairs_and_independent_spans_change_fills():
    import passivbot_rust as pbr

    assert not getattr(pbr, "__is_stub__", False)
    raw = _legacy_replay_config()
    migrated = parse_overrides(prepare_config(raw, verbose=False), verbose=False)
    explicit = deepcopy(raw)
    for side in ("long", "short"):
        strategy = explicit["bot"][side]["strategy"]["trailing_martingale"]
        for key in ("ema_span_0", "ema_span_1"):
            explicit["bot"][side]["unstuck"][key] = strategy[key]
    explicit["coin_overrides"]["ETH"]["bot"]["long"]["unstuck"] = {
        "ema_span_0": 21.5,
        "ema_span_1": 110.25,
    }
    explicit = parse_overrides(prepare_config(explicit, verbose=False), verbose=False)
    old_fills, old_equity, _ = _replay(migrated)
    expected_fills, expected_equity, _ = _replay(explicit)
    np.testing.assert_array_equal(old_fills, expected_fills)
    np.testing.assert_array_equal(old_equity, expected_equity)
    assert sum(row[13] == "close_unstuck_long" for row in old_fills) > 100
    independent = deepcopy(migrated)
    independent["bot"]["long"]["unstuck"].update(ema_span_0=3.5, ema_span_1=5.5)
    independent["coin_overrides"]["ETH"]["bot"]["long"].pop("unstuck")
    new_fills, _, _ = _replay(independent)
    assert not np.array_equal(new_fills, old_fills)
