"""Real-extension causal, risk, config and live/backtest feature parity tests."""

import copy
import math

import numpy as np
import pytest

from directional_efficiency import (
    load_symbol_values,
    required_windows,
    values_from_candles,
    reject_gpu_directional_efficiency,
)
from test_orchestrator_json_api import make_input, make_symbol, bot_params_pair, compute


@pytest.fixture(scope="module", autouse=True)
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    assert not getattr(pbr, "__is_stub__", False)
    assert callable(pbr.calc_directional_efficiency)
    return pbr


@pytest.mark.parametrize(
    "closes",
    [
        [100, 101, 105],
        [105, 101, 100],
        [100, 110, 100],
        [100, 100, 100],
        [10, 9, 12, 11],
        [1e-20, 1e-10, 1],
    ],
)
def test_rust_matches_independent_reference(require_real_passivbot_rust_module, closes):
    steps = np.diff(np.log(closes))
    expected = steps.sum() / np.abs(steps).sum() if np.abs(steps).sum() else 0
    assert require_real_passivbot_rust_module.calc_directional_efficiency(
        closes
    ) == pytest.approx(expected)


def _position_input(pside="long", efficiency=-1.0):
    kwargs = {
        f"{pside}_pos_size": 1.0 if pside == "long" else -1.0,
        f"{pside}_pos_price": 100.0,
        f"{pside}_bp": {
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
            "risk_directional_efficiency_cooldown_minutes": 20.0,
        },
    }
    symbol = make_symbol(
        0,
        bid=90.0 if pside == "long" else 110.0,
        ask=90.01 if pside == "long" else 110.01,
        **kwargs,
    )
    symbol["directional_efficiency"] = [[60.0, efficiency]]
    symbol[pside]["last_increase_fill_timestamp_ms"] = 1_000_000
    inp = make_input(
        balance=1_000.0,
        symbols=[symbol],
        global_bp=bot_params_pair(
            **{
                f"{pside}_overrides": {
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 1.0,
                }
            }
        ),
    )
    inp["timestamp_ms"] = 1_000_000 + 5 * 60_000
    return inp


def _orders(out, prefix, side="long"):
    return [
        o
        for o in out["orders"]
        if o["pside"] == side and o["order_type"].startswith(prefix)
    ]


@pytest.mark.parametrize("side,adverse", [("long", -1.0), ("short", 1.0)])
def test_adverse_pacing_keeps_closes_and_restarts_deterministically(
    require_real_passivbot_rust_module, side, adverse
):
    pbr = require_real_passivbot_rust_module
    inp = _position_input(side, adverse)
    blocked = compute(pbr, inp)
    assert not _orders(blocked, "entry_", side)
    assert _orders(blocked, "close_", side)
    assert compute(pbr, copy.deepcopy(inp)) == blocked  # No RAM-only decision state.
    inp["symbols"][0]["directional_efficiency"] = [[60.0, -adverse]]
    favorable = compute(pbr, inp)
    assert _orders(favorable, "entry_", side)
    assert _orders(favorable, "close_", side) == _orders(blocked, "close_", side)
    inp["symbols"][0]["directional_efficiency"] = [[60.0, adverse]]
    inp["timestamp_ms"] = 1_000_000 + 20 * 60_000
    assert _orders(compute(pbr, inp), "entry_", side)


def test_missing_pacing_input_blocks_only_additions(require_real_passivbot_rust_module):
    pbr = require_real_passivbot_rust_module
    inp = _position_input()
    symbol = inp["symbols"][0]
    symbol["directional_efficiency"] = []
    with pytest.raises(ValueError, match="MissingDirectionalEfficiency"):
        compute(pbr, inp)
    symbol["allow_missing_directional_efficiency"] = True
    out = compute(pbr, inp)
    assert not _orders(out, "entry_")
    assert _orders(out, "close_")
    assert out["diagnostics"]["warnings"]
    # A bounded stale ranking snapshot must never authorize DCA.
    symbol["forager_directional_efficiency"] = [[60.0, 1.0]]
    assert not _orders(compute(pbr, inp), "entry_")


def test_ranking_penalty_changes_selection_not_exposure(
    require_real_passivbot_rust_module,
):
    pbr = require_real_passivbot_rust_module
    symbols = [make_symbol(i, bid=100.0, ask=100.01) for i in range(2)]
    for i, symbol in enumerate(symbols):
        symbol["long"]["bot_params"]["forager_directional_efficiency_penalty"] = 0.75
        symbol["forager_directional_efficiency"] = [[60.0, 1.0 if i == 0 else 0.0]]
    inp = make_input(balance=1_000.0, symbols=symbols)
    selected = {o["symbol_idx"] for o in _orders(compute(pbr, inp), "entry_")}
    assert selected == {1}
    for symbol in symbols:
        symbol["long"]["bot_params"]["forager_directional_efficiency_penalty"] = 0.0
    baseline = compute(pbr, inp)
    assert {o["symbol_idx"] for o in _orders(baseline, "entry_")} == {0}
    for symbol in symbols:
        symbol.pop("forager_directional_efficiency")
    assert compute(pbr, inp) == baseline


@pytest.mark.parametrize("value", [float("nan"), 1.1, -1.1])
def test_invalid_metric_is_fatal_even_with_live_unavailability(
    require_real_passivbot_rust_module, value
):
    inp = _position_input(efficiency=value)
    inp["symbols"][0]["allow_missing_directional_efficiency"] = True
    with pytest.raises(ValueError):
        compute(require_real_passivbot_rust_module, inp)


def _candles(closes):
    result = np.zeros(len(closes), dtype=[("ts", "i8"), ("c", "f8")])
    result["ts"] = np.arange(len(closes)) * 60_000
    result["c"] = closes
    return result


def test_completed_window_ignores_future_and_rejects_gaps():
    candles = _candles([100, 110, 105, 1_000])
    actual = values_from_candles(candles, {2}, 120_000)
    candles["c"][3] = 0.01
    assert values_from_candles(candles, {2}, 120_000) == actual
    assert values_from_candles(candles[[0, 2, 3]], {2}, 120_000) == []
    assert values_from_candles(candles, {3}, 120_000) == []


@pytest.mark.asyncio
async def test_live_carry_forward_is_ranking_only_and_bounded():
    class CM:
        async def get_candles(self, symbol, **kwargs):
            assert kwargs["allow_remote_fetch"] is False
            assert kwargs["fill_trailing_gaps"] is False
            return _candles([100, 105, 110])

    current, ranking = await load_symbol_values(
        CM(),
        "BTC",
        {2},
        now_ms=240_000,
        allow_remote_fetch=False,
        ranking_max_age_ms=60_000,
    )
    assert current == []
    assert ranking == [[2.0, 1.0]]
    assert await load_symbol_values(
        CM(),
        "BTC",
        {2},
        now_ms=300_000,
        allow_remote_fetch=False,
        ranking_max_age_ms=60_000,
    ) == ([], [])


def test_config_hydration_roundtrip_bounds_and_warmup():
    from config_utils import format_config, get_template_config
    from config.shared_bot import flatten_shared_bot_side
    from config.optimize_bounds import flatten_optimize_bounds
    from warmup_utils import (
        compute_backtest_warmup_minutes,
        compute_per_coin_warmup_minutes,
    )

    cfg = get_template_config()
    assert not required_windows(flatten_shared_bot_side(cfg["bot"]["long"]))
    risk = cfg["bot"]["long"]["risk"]
    risk.update(
        directional_efficiency_cooldown_minutes=30,
        directional_efficiency_lookback_minutes=120,
    )
    cfg["live"].update(warmup_ratio=0, max_warmup_minutes=1)
    cfg = format_config(cfg, verbose=False)
    assert cfg["bot"]["long"]["risk"]["directional_efficiency_cooldown_minutes"] == 30
    assert format_config(cfg, verbose=False)["bot"] == cfg["bot"]
    assert compute_backtest_warmup_minutes(cfg) >= 120
    assert compute_per_coin_warmup_minutes(cfg)["__default__"] >= 120
    bounds = flatten_optimize_bounds(
        cfg["optimize"]["bounds"], strategy_kind=cfg["live"]["strategy_kind"]
    )
    assert "long_forager_directional_efficiency_penalty" in bounds
    cfg["backtest"]["candle_interval_minutes"] = 5
    with pytest.raises(ValueError, match="1 minute"):
        format_config(cfg, verbose=False)


@pytest.mark.parametrize(
    "key,value",
    [
        ("directional_efficiency_penalty", -0.1),
        ("directional_efficiency_penalty", 1.1),
        ("directional_efficiency_lookback_minutes", 1.5),
    ],
)
def test_invalid_config_is_rejected(key, value):
    from config_utils import format_config, get_template_config

    cfg = get_template_config()
    cfg["bot"]["long"]["forager"][key] = value
    with pytest.raises(ValueError, match="directional_efficiency"):
        format_config(cfg, verbose=False)


def test_gpu_rejects_enabled_feature_and_tunable_bounds():
    reject_gpu_directional_efficiency({"directional_efficiency_penalty": 0.0})
    for payload in [
        {"directional_efficiency_penalty": 0.5},
        {"risk_directional_efficiency_cooldown_minutes": 20},
        {"directional_efficiency_penalty": [0, 1, 0.1]},
    ]:
        with pytest.raises(ValueError, match="CPU"):
            reject_gpu_directional_efficiency({"nested": payload})


def test_dca_opt_in_stages_only_one_initial_order(require_real_passivbot_rust_module):
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.01,
        long_bp={"risk_directional_efficiency_cooldown_minutes": 30},
    )
    out = compute(
        require_real_passivbot_rust_module, make_input(balance=1000, symbols=[symbol])
    )
    assert len(_orders(out, "entry_")) == 1


def test_missing_ranking_input_does_not_block_held_position_exits(
    require_real_passivbot_rust_module,
):
    inp = _position_input(efficiency=1.0)
    symbol = inp["symbols"][0]
    symbol["long"]["bot_params"]["forager_directional_efficiency_penalty"] = 0.5
    # No ranking is needed to manage an already-held symbol.
    out = compute(require_real_passivbot_rust_module, inp)
    assert _orders(out, "close_")


def test_optimizer_rejects_fractional_window_search():
    from config_utils import format_config, get_template_config

    cfg = get_template_config()
    cfg["optimize"]["bounds"]["long"]["forager"][
        "directional_efficiency_lookback_minutes"
    ] = [15, 120]
    with pytest.raises(ValueError, match="integer step"):
        format_config(cfg, verbose=False)


def test_real_backtest_pacing_and_future_candle_invariance():
    from backtest import run_backtest
    from test_backtest_directional_eligibility import (
        _ema_anchor_config,
        _synthetic_inputs,
    )

    config = _ema_anchor_config(True)
    h, markets, btc, timestamps = _synthetic_inputs()
    for k in range(len(h)):
        close = 100.0 * math.exp(-0.005 * k)
        h[k, 0, :3] = [close * 1.002, close * 0.998, close]
    config["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 0
    base = run_backtest(
        h, copy.deepcopy(markets), copy.deepcopy(config), "binance", btc, timestamps
    )
    config["bot"]["long"]["risk"].update(
        directional_efficiency_cooldown_minutes=15,
        directional_efficiency_lookback_minutes=3,
    )
    paced = run_backtest(
        h, copy.deepcopy(markets), copy.deepcopy(config), "binance", btc, timestamps
    )

    def adds(result):
        return [
            r
            for r in result[0]
            if r[2] == "LONGCOIN" and str(r[13]).startswith("entry_")
        ]

    assert 0 < len(adds(paced)) < len(adds(base))
    assert all(
        float(b[1]) - float(a[1]) >= 15 * 60_000
        for a, b in zip(adds(paced), adds(paced)[1:])
    )
    changed = h.copy()
    changed[40:, 0, :3] *= 0.5
    future = run_backtest(
        changed,
        copy.deepcopy(markets),
        copy.deepcopy(config),
        "binance",
        btc,
        timestamps,
    )

    def prefix(result):
        return [list(r) for r in result[0] if float(r[1]) < timestamps[40]]

    assert prefix(future) == prefix(paced)


@pytest.mark.asyncio
async def test_live_fetch_failure_is_explicit_unavailability_not_neutral(caplog):
    class CM:
        async def get_candles(self, *args, **kwargs):
            raise TimeoutError("private transport details")

    assert await load_symbol_values(
        CM(), "BTC", {3}, now_ms=1_000_000, allow_remote_fetch=True
    ) == ([], [])
    assert "input unavailable" in caplog.text
    assert "private transport details" not in caplog.text


@pytest.mark.asyncio
async def test_live_malformed_prices_are_not_swallowed():
    class CM:
        async def get_candles(self, *args, **kwargs):
            return _candles([100, 0, 105])

    with pytest.raises(ValueError, match="positive closes"):
        await load_symbol_values(
            CM(), "BTC", {2}, now_ms=180_000, allow_remote_fetch=False
        )
