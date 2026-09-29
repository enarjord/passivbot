"""Regression tests for optimizer warmup stamping.

Covers the bug where prepare_hlcvs_mss stamped mss[coin]["warmup_minutes"]
from the template bot's decorative values (e.g. volatility_ema_span_1h)
instead of the worst case the optimizer's search space can actually produce.
"""

from pathlib import Path

import pytest

from backtest_universe import effective_backtest_data_coins
from config_utils import get_template_config
from optimization.warmup import (
    _apply_config_overrides,
    build_optimizer_data_config,
    compute_optimizer_backtest_warmup_minutes,
    compute_optimizer_per_coin_warmup_minutes,
    stamp_warmup_metadata,
    validate_optimizer_effective_configs,
)
from warmup_utils import compute_per_coin_warmup_minutes


def _make_optimizer_config() -> dict:
    """Build a config where template bot and bounds disagree on warmup.

    Template bot has volatility_ema_span_1h = 1690, which would
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
    long_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] = 770.0
    long_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_1"] = 210.0
    long_bot["forager"]["volume_ema_span_1m"] = 520.0
    long_bot["forager"]["volatility_ema_span_1m"] = 225.0
    long_bot["strategy"]["trailing_martingale"]["volatility_ema_span_1h"] = 1690.0
    long_bot["strategy"]["trailing_martingale"]["volatility_ema_span_1m"] = 60.0

    short_bot = config["bot"]["short"]
    short_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] = 1.0
    short_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_1"] = 1.0
    short_bot["forager"]["volume_ema_span_1m"] = 0.0
    short_bot["forager"]["volatility_ema_span_1m"] = 0.0
    short_bot["strategy"]["trailing_martingale"]["volatility_ema_span_1h"] = 0.0
    short_bot["strategy"]["trailing_martingale"]["volatility_ema_span_1m"] = 0.0

    bounds = config["optimize"]["bounds"]
    bounds["long_ema_span_0"] = [1, 100, 1]
    bounds["long_ema_span_1"] = [1, 100, 1]
    bounds["long_forager_volatility_ema_span_1m"] = [0, 0]
    bounds["long_forager_volume_ema_span_1m"] = [0, 0]
    bounds["long_volatility_ema_span_1h"] = [0, 0]
    bounds["long_volatility_ema_span_1m"] = [0, 0]
    bounds["short_ema_span_0"] = [1, 100, 1]
    bounds["short_ema_span_1"] = [1, 100, 1]
    bounds["short_forager_volatility_ema_span_1m"] = [0, 0]
    bounds["short_forager_volume_ema_span_1m"] = [0, 0]
    bounds["short_volatility_ema_span_1h"] = [0, 0]
    bounds["short_volatility_ema_span_1m"] = [0, 0]
    for side in ("long", "short"):
        for i in (0, 1):
            bounds[f"{side}_unstuck_ema_span_{i}"] = [1, 100, 1]
    return config


def test_stamp_optimizer_warmup_uses_bounds_when_template_bot_exceeds_them():
    """Bug regression: optimizer stamping must use bounds, not decorative bot values."""
    config = _make_optimizer_config()
    mss = {"HYPE": {"first_valid_index": 0, "last_valid_index": 180000}}

    # Simulate the buggy template-derived stamping to establish the baseline
    # we're repairing.
    template_warmup_map = compute_per_coin_warmup_minutes(config)
    stamp_warmup_metadata(mss, ["HYPE"], template_warmup_map)
    assert mss["HYPE"]["warmup_minutes"] == 30420, (
        "baseline sanity check: compute_per_coin_warmup_minutes on the "
        "template config should produce the buggy 30420-minute warmup "
        "(1690h * 60 * 0.3). If this assertion fails, the test's "
        "assumptions about the template/bounds disagreement no longer hold."
    )

    # The fix: optimizer stamping reflects the max the optimizer's search
    # space can produce (ema_span_0 ∈ [1, 100] ⇒ 100 * 0.3 = 30 min).
    warmup_map = compute_optimizer_per_coin_warmup_minutes(config)
    stamp_warmup_metadata(mss, ["HYPE"], warmup_map)

    assert mss["HYPE"]["warmup_minutes"] == 30
    assert mss["HYPE"]["trade_start_index"] == 30


def test_shared_optimizer_warmup_helper_uses_bounds_when_template_bot_exceeds_them():
    config = _make_optimizer_config()

    warmup_map = compute_optimizer_per_coin_warmup_minutes(config)

    assert warmup_map["__default__"] == 30
    assert "HYPE" not in warmup_map
    assert compute_optimizer_backtest_warmup_minutes(config) == 30


def test_optimizer_data_config_uses_reachable_side_gates_from_bounds():
    config = get_template_config()
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": ["ETH"]}
    config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.0
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 1.0
    config["optimize"]["bounds"]["long_n_positions"] = [1, 3, 1]
    config["optimize"]["bounds"]["long_total_wallet_exposure_limit"] = [0.5, 1.5, 0.01]
    config["optimize"]["bounds"]["short_n_positions"] = [1, 3, 1]
    config["optimize"]["bounds"]["short_total_wallet_exposure_limit"] = [0.0]

    data_config = build_optimizer_data_config(config)

    assert data_config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] > 0.0
    assert data_config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert effective_backtest_data_coins(data_config) == ["BTC"]


def test_optimizer_data_config_enables_template_disabled_side_when_bounds_reach_it():
    config = get_template_config()
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0.0
    config["optimize"]["bounds"]["short_n_positions"] = [1, 3, 1]
    config["optimize"]["bounds"]["short_total_wallet_exposure_limit"] = [0.5, 1.5, 0.01]

    data_config = build_optimizer_data_config(config)

    assert data_config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 1.5


def test_optimizer_warmup_fixed_runtime_override_rejects_unknown_path():
    config = get_template_config()

    with pytest.raises(KeyError, match="n_positons"):
        _apply_config_overrides(config, {"bot.long.risk.n_positons": 7})

    assert "n_positons" not in config["bot"]["long"]["risk"]


def test_optimizer_rejects_invalid_effective_fixed_runtime_value_before_backend():
    config = get_template_config()
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.forager.score_weights.volatility": -1.0
    }

    with pytest.raises(ValueError, match="score_weights.*non-negative"):
        validate_optimizer_effective_configs(config)


def test_stamp_optimizer_warmup_respects_last_valid_index_cap():
    """trade_start_index must never exceed last_valid_index, even if the
    bounds-derived warmup would push it past the end of the data."""

    config = _make_optimizer_config()
    config["optimize"]["bounds"]["long_ema_span_0"] = [1, 100000, 1]
    mss = {"HYPE": {"first_valid_index": 0, "last_valid_index": 10}}

    stamp_warmup_metadata(mss, ["HYPE"], compute_optimizer_per_coin_warmup_minutes(config))

    # Warmup = 100000 * 0.3 = 30000 min. Clamp to last_valid_index = 10.
    assert mss["HYPE"]["warmup_minutes"] == 30000
    assert mss["HYPE"]["trade_start_index"] == 10


def test_stamp_warmup_metadata_respects_last_valid_index_cap():
    warmup_map = {"__default__": 30000}
    mss = {"HYPE": {"first_valid_index": 0, "last_valid_index": 10}}

    stamp_warmup_metadata(mss, ["HYPE"], warmup_map)

    assert mss["HYPE"]["warmup_minutes"] == 30000
    assert mss["HYPE"]["trade_start_index"] == 10


def test_stamp_optimizer_warmup_is_wired_into_register_exchange_data():
    """Future-proof the fix: if a refactor silently drops the call from
    _register_exchange_data, the regression tests above still pass because
    they exercise _stamp_optimizer_warmup directly. This test pins the
    wiring so a missing call site fails a test rather than re-introducing
    the bug."""
    source = Path("src/optimize.py").read_text(encoding="utf-8")
    assert "_stamp_optimizer_warmup(" in source, (
        "_register_exchange_data must call _stamp_optimizer_warmup. "
        "If this test fails, the warmup-from-bounds fix has regressed: "
        "the optimizer will size per-coin warmup from template bot values "
        "again instead of from optimize.bounds."
    )


def test_prepare_suite_contexts_uses_shared_optimizer_warmup_helper():
    source = Path("src/optimize_suite.py").read_text(encoding="utf-8")
    assert "compute_optimizer_per_coin_warmup_minutes(" in source
    assert "stamp_warmup_metadata(" in source


@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
def test_optimizer_rms_history_is_not_a_shared_activation_delay(consumer):
    from copy import deepcopy
    import numpy as np
    from backtest import run_backtest
    from config import prepare_config
    from optimize import _stamp_optimizer_warmup
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["live"]["max_warmup_minutes"] = 3
    cfg["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = 60.0
    bounds = cfg["optimize"]["bounds"]["long"]
    bounds["forager"]["unilateralness_ema_span_1m"] = [1.0, 60.0]
    if consumer == "forager":
        bounds["forager"]["score_weights"] = {"unilateralness": [0.0, 1.0]}
    else:
        bounds["entry_cooldown"]["weights_minutes"] = {"adverse_directionality": [0.0, 10.0]}
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    assert compute_optimizer_per_coin_warmup_minutes(cfg)["__default__"] == 1201
    activation = compute_optimizer_per_coin_warmup_minutes(cfg, for_trade_activation=True)
    assert activation["__default__"] == 3
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    unstamped_markets = deepcopy(markets)
    baseline = run_backtest(hlcvs, unstamped_markets, cfg, "binance", btc, timestamps)
    _stamp_optimizer_warmup(cfg, markets, ["LONGCOIN", "SHORTCOIN"])
    assert markets["LONGCOIN"]["trade_start_index"] == 3
    result = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert len(result[0]) > 0
    np.testing.assert_array_equal(result[0], baseline[0])
    np.testing.assert_array_equal(result[1], baseline[1])
    assert result[2] == baseline[2]
    if consumer == "forager":
        # This candidate fits available slots and does not consume a ranking score.
        cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1.0
        result = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
        np.testing.assert_array_equal(result[0], baseline[0])
        np.testing.assert_array_equal(result[1], baseline[1])
        assert result[2] == baseline[2]
    else:
        # The enabled candidate uses its own short window, not the search maximum.
        cfg["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 1.0
        cfg["bot"]["long"]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 1.0
        fills, _, _, payload = run_backtest(
            hlcvs, markets, cfg, "binance", btc, timestamps, return_payload=True
        )
        assert payload.backtest_params["trade_start_indices"][0] == 3
        long_entries = [row for row in fills if str(row[13]).startswith("entry_") and "long" in str(row[13])]
        assert min(int(row[0]) for row in long_entries) == 21
        standalone = run_backtest(hlcvs, unstamped_markets, cfg, "binance", btc, timestamps)
        np.testing.assert_array_equal(fills, standalone[0])
        assert len(fills) > 0


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("consumer", ["forager", "adverse"])
@pytest.mark.parametrize("pin", ["weight", "span"])
@pytest.mark.parametrize("pin_source", ["runtime", "anchor"])
def test_rms_dataset_history_respects_pins(side, consumer, pin, pin_source):
    from copy import deepcopy
    from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY
    from optimization.warmup import _build_optimizer_boundary_configs
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["warmup_ratio"] = 0.0
    cfg["optimize"]["bounds"] = {}
    cfg["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0)
    cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    weight_path = (["forager", "score_weights", "unilateralness"] if consumer == "forager"
                   else ["entry_cooldown", "weights_minutes", "adverse_directionality"])
    weight_key = ("forager_score_weights_unilateralness" if consumer == "forager"
                  else "entry_cooldown_weights_minutes_adverse_directionality")
    span_path = ["forager", "unilateralness_ema_span_1m"]
    bounds = cfg["optimize"]["bounds"]
    for pside in ("long", "short"):
        bounds[f"{pside}_n_positions"] = [1.0]
        bounds[f"{pside}_total_wallet_exposure_limit"] = [float(pside == side)]
    bounds[f"{side}_{weight_key}"] = [0.0, 1.0]
    bounds[f"{side}_unilateralness_ema_span_1m"] = [1.0, 100000.0]
    key = f"{side}_{weight_key}" if pin == "weight" else f"{side}_unilateralness_ema_span_1m"
    path = ["bot", side, *(weight_path if pin == "weight" else span_path)]
    value = 0.0 if pin == "weight" else 2.5
    expected = 0 if pin == "weight" else 51
    if pin_source == "runtime":
        cfg["optimize"]["fixed_runtime_overrides"] = {".".join(path): value}
    else:
        other_key = f"{side}_unilateralness_ema_span_1m" if pin == "weight" else f"{side}_{weight_key}"
        other_path = ["bot", side, *(span_path if pin == "weight" else weight_path)]
        cfg[ANCHOR_PLAN_KEY] = {
            "anchors": [{"source": "anchor.json", "fixed_values": [{"key": key, "path": path, "value": value}]}],
            "fixed_keys": [key], "tunable_keys": [other_key], "key_paths": [other_path],
        }
    before = deepcopy(cfg)
    # Exercise the config passed to dataset preparation as well as finalized candidates.
    data_cfg = build_optimizer_data_config(cfg)
    assert compute_backtest_warmup_minutes(data_cfg) == expected
    assert compute_optimizer_backtest_warmup_minutes(cfg) == expected
    for candidate in _build_optimizer_boundary_configs(cfg, rms_consumer_corner=True):
        assert compute_backtest_warmup_minutes(candidate) == expected
    assert cfg == before
    if pin_source == "runtime":
        cfg["optimize"]["fixed_runtime_overrides"] = {}
    else:
        cfg.pop(ANCHOR_PLAN_KEY)
    assert compute_backtest_warmup_minutes(build_optimizer_data_config(cfg)) == 2000001


def test_standalone_rms_history_does_not_apply_optimizer_only_pin():
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["warmup_ratio"] = 0.0
    cfg["optimize"]["bounds"] = {}
    cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1.0
    cfg["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 2.5
    cfg["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.forager.score_weights.unilateralness": 0.0,
        "bot.long.forager.unilateralness_ema_span_1m": 1.0,
    }
    assert compute_backtest_warmup_minutes(cfg) == 51


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("searched", [False, True])
def test_rms_history_uses_selected_coin_overrides_not_unused_global(side, searched):
    from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["warmup_ratio"] = 0.0
    cfg["live"]["approved_coins"] = {s: ["BTC", "ETH"] for s in ("long", "short")}
    cfg["backtest"]["coins"] = {"binance": ["BTC", "ETH"]}
    cfg["optimize"]["bounds"] = {}
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].update(n_positions=1, total_wallet_exposure_limit=float(s == side))
        cfg["optimize"]["bounds"].update({f"{s}_n_positions": [1.0], f"{s}_total_wallet_exposure_limit": [float(s == side)]})
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 100000.0
    cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = float(not searched)
    if searched:
        cfg["optimize"]["bounds"][f"{side}_entry_cooldown_weights_minutes_adverse_directionality"] = [0.0, 1.0]
    cfg["coin_overrides"] = {coin: {"bot": {side: {"entry_cooldown": {
        "weights_minutes": {"adverse_directionality": 0.0}, "max_duration_minutes": None,
    }}}} for coin in ("BTC", "ETH")}
    # An unselected override with an active inherited policy must not inflate history either.
    cfg["coin_overrides"]["SOL"] = {"bot": {side: {"entry_cooldown": {"base_duration_minutes": 0.0}}}}
    assert compute_backtest_warmup_minutes(cfg) == 0
    assert max(compute_per_coin_warmup_minutes(cfg).values()) == 0
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 0
    # An inheriting selected coin makes the global policy reachable again.
    cfg["backtest"]["coins"]["binance"].append("XRP")
    assert compute_backtest_warmup_minutes(cfg) == 0
    cfg["live"]["approved_coins"][side].append("XRP")
    assert compute_backtest_warmup_minutes(cfg) == 2000001
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 2000001
    # Without a resolved selection, retain the conservative global history.
    cfg["backtest"]["coins"] = {}
    cfg["live"]["approved_coins"] = {"long": [], "short": []}
    assert compute_backtest_warmup_minutes(cfg) == 2000001


@pytest.mark.parametrize("searched", [False, True])
@pytest.mark.parametrize("dataset_known", [False, True])
def test_rms_history_respects_side_specific_coin_eligibility(searched, dataset_known):
    from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["warmup_ratio"] = 0.0
    cfg["live"]["approved_coins"] = {"long": ["BTC"], "short": ["ETH"]}
    cfg["backtest"]["coins"] = {"binance": ["BTC", "ETH"]} if dataset_known else {}
    cfg["optimize"]["bounds"] = {}
    cfg["coin_overrides"] = {}
    for side, coin in (("long", "BTC"), ("short", "ETH")):
        cfg["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0)
        cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 100000.0
        cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
        cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = float(not searched)
        cfg["optimize"]["bounds"].update({f"{side}_n_positions": [1.0], f"{side}_total_wallet_exposure_limit": [1.0]})
        if searched:
            cfg["optimize"]["bounds"][f"{side}_entry_cooldown_weights_minutes_adverse_directionality"] = [0.0, 1.0]
        cfg["coin_overrides"][coin] = {"bot": {side: {"entry_cooldown": {
            "weights_minutes": {"adverse_directionality": 0.0}, "max_duration_minutes": None,
        }}}}
    assert compute_backtest_warmup_minutes(cfg) == 0
    assert max(compute_per_coin_warmup_minutes(cfg).values()) == 0
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 0
    # Opening the opposite side makes that inherited policy a real consumer.
    cfg["live"]["approved_coins"]["long"].append("ETH")
    assert compute_backtest_warmup_minutes(cfg) == 2000001
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 2000001
    # A zero per-coin exposure pin still makes the newly approved side ineligible.
    cfg["coin_overrides"]["ETH"]["bot"]["long"] = {"wallet_exposure_limit": 0.0}
    assert compute_backtest_warmup_minutes(cfg) == 0
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 0


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("searched", [False, True])
@pytest.mark.parametrize("override_enabled", [False, True])
@pytest.mark.parametrize("exact_dataset", [False, True])
def test_rms_history_resolves_market_alias_policies(monkeypatch, side, searched, override_enabled, exact_dataset):
    from backtest import _get_backtest_coin_override
    from warmup_utils import compute_backtest_warmup_minutes

    symbol = "BTC/USDT:USDT"
    monkeypatch.setattr("utils._load_coin_to_symbol_map", lambda exchange: {
        "BTC": [symbol], symbol: [symbol], "BTCUSDT": [symbol],
    })
    coin, override_key = (symbol, "BTC") if exact_dataset else ("BTC", "binance::BTCUSDT")
    cfg = get_template_config()
    cfg["live"].update(warmup_ratio=0.0, approved_coins={"long": [coin], "short": [coin]})
    cfg["backtest"]["coins"] = {"binance": [coin]}
    cfg["optimize"]["bounds"] = {}
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].update(n_positions=1, total_wallet_exposure_limit=float(s == side))
        cfg["optimize"]["bounds"].update({f"{s}_n_positions": [1.0], f"{s}_total_wallet_exposure_limit": [float(s == side)]})
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 2.5
    cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = float(not override_enabled)
    if searched:
        cfg["optimize"]["bounds"][f"{side}_entry_cooldown_weights_minutes_adverse_directionality"] = [0.0, 1.0]
    patch = {"bot": {side: {"entry_cooldown": {
        "weights_minutes": {"adverse_directionality": float(override_enabled)},
        "max_duration_minutes": 60.0 if override_enabled else None,
    }}}}
    cfg["coin_overrides"] = {override_key: patch}
    # The history budget must use the same policy as the actual payload resolver.
    assert _get_backtest_coin_override(cfg, {}, "binance", coin) == patch
    expected = 51 if override_enabled else 0
    assert compute_backtest_warmup_minutes(cfg) == expected
    assert compute_optimizer_backtest_warmup_minutes(cfg) == expected
    assert compute_per_coin_warmup_minutes(cfg)[override_key] == expected
    # An exact identifier on another venue must not shadow the inherited policy.
    cfg["coin_overrides"] = {"bybit::BTCUSDT": patch}
    assert _get_backtest_coin_override(cfg, {}, "binance", coin) == {}
    assert compute_backtest_warmup_minutes(cfg) == (51 if searched or not override_enabled else 0)
    # Sizing may precede metadata loading: retain both possible policies offline.
    cfg["coin_overrides"] = {override_key: patch}
    monkeypatch.setattr("utils._load_coin_to_symbol_map", lambda exchange: {})
    assert compute_backtest_warmup_minutes(cfg) == 51
