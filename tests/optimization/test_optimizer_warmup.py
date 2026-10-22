"""Regression tests for optimizer warmup stamping.

Covers the bug where prepare_hlcvs_mss stamped mss[coin]["warmup_minutes"]
from the template bot's decorative values (e.g. volatility_ema_span_1h)
instead of the worst case the optimizer's search space can actually produce.
"""
from config.universe import set_side_universe_sources

from pathlib import Path

import pytest

from backtest_universe import effective_backtest_data_coins
from config_utils import get_template_config
from config.optimize_bounds import set_flat_optimize_bound
from optimization.warmup import (
    _apply_config_overrides,
    build_optimizer_data_config,
    compute_optimizer_backtest_warmup_minutes,
    compute_optimizer_per_coin_warmup_minutes,
    stamp_warmup_metadata,
    validate_optimizer_effective_configs,
)
from warmup_utils import compute_per_coin_warmup_minutes


def _set_bounds(config, entries):
    """Author nested config bounds using the optimizer gene-path writer."""
    for key, value in entries.items():
        set_flat_optimize_bound(config["optimize"]["bounds"], config['bot']["portfolio"]['strategy_kind'], key, value)


def _make_optimizer_config() -> dict:
    """Build a config where template bot and bounds disagree on warmup.

    Template bot has volatility_ema_span_1h = 1690, which would
    produce warmup = 1690 * 60 * 0.3 = 30420 min under
    compute_per_coin_warmup_minutes.

    Bounds pin that field to [0, 0] and cap ema_span_0 at 100, so the
    largest warmup the optimizer can actually produce is 100 * 0.3 = 30 min.
    """
    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    config['bot']["portfolio"]['warmup']['span_multiplier'] = 0.3
    config['bot']["portfolio"]['warmup']['max_minutes'] = None
    set_side_universe_sources(config, 'approved_coins', {"long": ["HYPE"], "short": ["HYPE"]})
    config["backtest"]["exchanges"] = ["combined"]
    config["backtest"]["coins"] = {"combined": ["HYPE"]}

    long_bot = config["bot"]["long"]
    long_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_0_minutes"] = 770.0
    long_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_1_minutes"] = 210.0
    long_bot["forager"]["volume_ema_span_minutes"] = 520.0
    long_bot["forager"]["volatility_ema_span_minutes"] = 225.0
    long_bot["strategy"]["trailing_martingale"]["volatility_ema_span_1h_candles"] = 1690.0
    long_bot["strategy"]["trailing_martingale"]["volatility_ema_span_minutes"] = 60.0

    short_bot = config["bot"]["short"]
    short_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_0_minutes"] = 1.0
    short_bot["strategy"]["trailing_martingale"]["entry"]["ema_span_1_minutes"] = 1.0
    short_bot["forager"]["volume_ema_span_minutes"] = 0.0
    short_bot["forager"]["volatility_ema_span_minutes"] = 0.0
    short_bot["strategy"]["trailing_martingale"]["volatility_ema_span_1h_candles"] = 0.0
    short_bot["strategy"]["trailing_martingale"]["volatility_ema_span_minutes"] = 0.0

    bounds = config["optimize"]["bounds"]
    _set_bounds(config, {"long_ema_span_0": [1, 100, 1]})
    _set_bounds(config, {"long_ema_span_1": [1, 100, 1]})
    _set_bounds(config, {"long_forager_volatility_ema_span_1m": [0, 0]})
    _set_bounds(config, {"long_forager_volume_ema_span_1m": [0, 0]})
    _set_bounds(config, {"long_volatility_ema_span_1h": [0, 0]})
    _set_bounds(config, {"long_volatility_ema_span_1m": [0, 0]})
    _set_bounds(config, {"short_ema_span_0": [1, 100, 1]})
    _set_bounds(config, {"short_ema_span_1": [1, 100, 1]})
    _set_bounds(config, {"short_forager_volatility_ema_span_1m": [0, 0]})
    _set_bounds(config, {"short_forager_volume_ema_span_1m": [0, 0]})
    _set_bounds(config, {"short_volatility_ema_span_1h": [0, 0]})
    _set_bounds(config, {"short_volatility_ema_span_1m": [0, 0]})
    for side in ("long", "short"):
        for i in (0, 1):
            _set_bounds(config, {f"{side}_unstuck_ema_span_{i}": [1, 100, 1]})
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
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    set_side_universe_sources(config, 'approved_coins', {"long": ["BTC"], "short": ["ETH"]})
    config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.0
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 1.0
    _set_bounds(config, {"long_n_positions": [1, 3, 1]})
    _set_bounds(config, {"long_total_wallet_exposure_limit": [0.5, 1.5, 0.01]})
    _set_bounds(config, {"short_n_positions": [1, 3, 1]})
    _set_bounds(config, {"short_total_wallet_exposure_limit": [0.0]})

    data_config = build_optimizer_data_config(config)

    assert data_config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] > 0.0
    assert data_config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert effective_backtest_data_coins(data_config) == ["BTC"]


def test_optimizer_data_config_reaches_zero_capacity_when_explicitly_activated():
    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0.0
    _set_bounds(config, {"short_n_positions": [1, 3, 1]})
    _set_bounds(config, {"short_total_wallet_exposure_limit": [0.5, 1.5, 0.01]})

    data_config = build_optimizer_data_config(config)

    assert data_config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 1.5


def test_optimizer_warmup_fixed_runtime_override_rejects_unknown_path():
    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True

    with pytest.raises(KeyError, match="n_positons"):
        _apply_config_overrides(config, {"bot.long.risk.n_positons": 7})

    assert "n_positons" not in config["bot"]["long"]["risk"]


def test_optimizer_rejects_invalid_effective_fixed_runtime_value_before_backend():
    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.forager.score_weights.volatility": -1.0
    }

    with pytest.raises(ValueError, match=r"score_weights\.volatility pin.*nonnegative"):
        validate_optimizer_effective_configs(config)


def test_stamp_optimizer_warmup_respects_last_valid_index_cap():
    """trade_start_index must never exceed last_valid_index, even if the
    bounds-derived warmup would push it past the end of the data."""

    config = _make_optimizer_config()
    _set_bounds(config, {"long_ema_span_0": [1, 100000, 1]})
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

    cfg = _ema_anchor_config(True, include_dataset=False)
    cfg['bot']["portfolio"]['warmup']['max_minutes'] = 3
    cfg["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = 60.0
    bounds = cfg["optimize"]["bounds"]["long"]
    bounds["forager"]["unilateralness_ema_span_minutes"] = [1.0, 60.0]
    if consumer == "forager":
        bounds["forager"]["score_weights"]["unilateralness"] = [0.0, 1.0]
    else:
        bounds["entry_cooldown"]["weights_minutes"] = {"adverse_directionality": [0.0, 10.0]}
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    # Only one selected coin is long-eligible, so score-only history is dormant.
    expected_history = 3 if consumer == "forager" else 1201
    assert compute_optimizer_per_coin_warmup_minutes(cfg)["__default__"] == expected_history
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
        cfg["bot"]["long"]["forager"]["unilateralness_ema_span_minutes"] = 1.0
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
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    cfg["optimize"]["bounds"] = {}
    cfg["bot"][side]["risk"].update(max_positions=1, total_wallet_exposure_limit=1.0)
    cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    if consumer == "forager":
        # Only the searched coordinate participates in normalization. Its positive
        # lower bound avoids activating an omitted weight at the all-zero fallback.
        cfg["bot"][side]["forager"]["score_weights"] = dict(
            volume=0.0, volatility=0.0, ema_readiness=0.0, unilateralness=1.0,
        )
    weight_path = (["forager", "score_weights", "unilateralness"] if consumer == "forager"
                   else ["entry_cooldown", "weights_minutes", "adverse_directionality"])
    weight_key = ("forager_score_weights_unilateralness" if consumer == "forager"
                  else "entry_cooldown_weights_minutes_adverse_directionality")
    span_path = ["forager", "unilateralness_ema_span_minutes"]
    bounds = cfg["optimize"]["bounds"]
    for pside in ("long", "short"):
        _set_bounds(cfg, {f"{pside}_n_positions": [1.0]})
        _set_bounds(cfg, {f"{pside}_total_wallet_exposure_limit": [float(pside == side)]})
    _set_bounds(cfg, {f"{side}_{weight_key}": [0.1 if consumer == "forager" else 0.0, 1.0]})
    _set_bounds(cfg, {f"{side}_unilateralness_ema_span_1m": [1.0, 100000.0]})
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
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    cfg["optimize"]["bounds"] = {}
    cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1.0
    cfg["bot"]["long"]["forager"]["unilateralness_ema_span_minutes"] = 2.5
    cfg["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.forager.score_weights.unilateralness": 0.0,
        "bot.long.forager.unilateralness_ema_span_minutes": 1.0,
    }
    assert compute_backtest_warmup_minutes(cfg) == 51


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("searched", [False, True])
def test_rms_history_uses_selected_coin_overrides_not_unused_global(side, searched):
    from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes

    cfg = get_template_config()
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    set_side_universe_sources(cfg, 'approved_coins', {s: ["BTC", "ETH"] for s in ("long", "short")})
    cfg["backtest"]["coins"] = {"binance": ["BTC", "ETH"]}
    cfg["optimize"]["bounds"] = {}
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].update(max_positions=1, total_wallet_exposure_limit=float(s == side))
        _set_bounds(cfg, {f"{s}_n_positions": [1.0], f"{s}_total_wallet_exposure_limit": [float(s == side)]})
    cfg["bot"][side]["forager"]["unilateralness_ema_span_minutes"] = 100000.0
    cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = float(not searched)
    if searched:
        _set_bounds(cfg, {f"{side}_entry_cooldown_weights_minutes_adverse_directionality": [0.0, 1.0]})
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
    cfg['bot'][side]['universe']['approved_coins'].append("XRP")
    assert compute_backtest_warmup_minutes(cfg) == 2000001
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 2000001
    # Without a resolved selection, retain the conservative global history.
    cfg["backtest"]["coins"] = {}
    set_side_universe_sources(cfg, 'approved_coins', {"long": [], "short": []})
    assert compute_backtest_warmup_minutes(cfg) == 2000001


@pytest.mark.parametrize("searched", [False, True])
@pytest.mark.parametrize("dataset_known", [False, True])
def test_rms_history_respects_side_specific_coin_eligibility(searched, dataset_known):
    from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes

    cfg = get_template_config()
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    set_side_universe_sources(cfg, 'approved_coins', {"long": ["BTC"], "short": ["ETH"]})
    cfg["backtest"]["coins"] = {"binance": ["BTC", "ETH"]} if dataset_known else {}
    cfg["optimize"]["bounds"] = {}
    cfg["coin_overrides"] = {}
    for side, coin in (("long", "BTC"), ("short", "ETH")):
        cfg["bot"][side]["risk"].update(max_positions=1, total_wallet_exposure_limit=1.0)
        cfg["bot"][side]["forager"]["unilateralness_ema_span_minutes"] = 100000.0
        cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
        cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = float(not searched)
        _set_bounds(cfg, {f"{side}_n_positions": [1.0], f"{side}_total_wallet_exposure_limit": [1.0]})
        if searched:
            _set_bounds(cfg, {f"{side}_entry_cooldown_weights_minutes_adverse_directionality": [0.0, 1.0]})
        cfg["coin_overrides"][coin] = {"bot": {side: {"entry_cooldown": {
            "weights_minutes": {"adverse_directionality": 0.0}, "max_duration_minutes": None,
        }}}}
    assert compute_backtest_warmup_minutes(cfg) == 0
    assert max(compute_per_coin_warmup_minutes(cfg).values()) == 0
    assert compute_optimizer_backtest_warmup_minutes(cfg) == 0
    # Opening the opposite side makes that inherited policy a real consumer.
    cfg['bot']["long"]['universe']['approved_coins'].append("ETH")
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
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    set_side_universe_sources(cfg, 'approved_coins', {'long': [coin], 'short': [coin]})
    cfg["backtest"]["coins"] = {"binance": [coin]}
    cfg["optimize"]["bounds"] = {}
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].update(max_positions=1, total_wallet_exposure_limit=float(s == side))
        _set_bounds(cfg, {f"{s}_n_positions": [1.0], f"{s}_total_wallet_exposure_limit": [float(s == side)]})
    cfg["bot"][side]["forager"]["unilateralness_ema_span_minutes"] = 2.5
    cfg["bot"][side]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = float(not override_enabled)
    if searched:
        _set_bounds(cfg, {f"{side}_entry_cooldown_weights_minutes_adverse_directionality": [0.0, 1.0]})
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


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("searched", [False, True])
def test_score_only_rms_history_requires_reachable_ranking(side, dynamic, slots, searched):
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    set_side_universe_sources(cfg, 'approved_coins', {'long': ['BTC', 'ETH'], 'short': ['BTC', 'ETH']})
    cfg["backtest"].update(coins={"binance": ["BTC", "ETH"]}, dynamic_wel_by_tradability=dynamic)
    cfg["optimize"]["bounds"] = {}
    _set_bounds(cfg, {f"{side}_n_positions": [slots]})
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].update(max_positions=slots, total_wallet_exposure_limit=float(s == side))
        _set_bounds(cfg, {
            f"{s}_n_positions": [slots],
            f"{s}_total_wallet_exposure_limit": [float(s == side)],
        })
    cfg["bot"][side]["forager"].update(
        unilateralness_ema_span_minutes=100000,
        score_weights=dict(volume=0, volatility=0, ema_readiness=0, unilateralness=float(not searched)),
    )
    if searched:
        _set_bounds(cfg, {
            # Keep omitted zero weights zero through normalization, including at
            # the lower corner, while leaving RMS history fully reachable.
            f"{side}_forager_score_weights_unilateralness": [0.1, 1.0],
            f"{side}_unilateralness_ema_span_1m": [1.0, 100000.0],
            f"{side}_n_positions": [slots],
        })
    expected = 2000001 if dynamic or slots < 2 else 0
    assert compute_backtest_warmup_minutes(cfg) == expected
    assert compute_optimizer_backtest_warmup_minutes(cfg) == expected
    if not searched:
        assert max(compute_per_coin_warmup_minutes(cfg).values()) == expected
    # Adverse cooldown independently consumes RMS even when ranking cannot run.
    cfg["bot"][side]["entry_cooldown"].update(
        base_duration_minutes=0, max_duration_minutes=60,
        weights_minutes={"exposure_ratio": 0.0, "adverse_directionality": 1.0},
    )
    assert compute_backtest_warmup_minutes(cfg) == 2000001


def test_scoring_history_keeps_reachable_slot_bounds_and_single_coin_dormancy():
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    for activation_side in ("long", "short"):
        cfg["bot"][activation_side]["enabled"] = True
    cfg['bot']["portfolio"]['warmup']['span_multiplier'] = 0.0
    set_side_universe_sources(cfg, 'approved_coins', {'long': ['BTC', 'ETH'], 'short': []})
    cfg["backtest"].update(coins={"binance": ["BTC", "ETH"]}, dynamic_wel_by_tradability=False)
    cfg["optimize"]["bounds"] = {}
    cfg["bot"]["long"]["risk"].update(max_positions=2, total_wallet_exposure_limit=1)
    cfg["bot"]["long"]["forager"].update(unilateralness_ema_span_minutes=100000)
    cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1
    _set_bounds(cfg, {"long_n_positions": [1, 2]})
    assert compute_backtest_warmup_minutes(cfg) == 2000001
    cfg['bot']["long"]['universe']['approved_coins'] = ["BTC"]
    assert compute_backtest_warmup_minutes(cfg) == 0
    cfg["backtest"]["dynamic_wel_by_tradability"] = True
    assert compute_backtest_warmup_minutes(cfg) == 0


def _mirrored_weight_config(strategy, short_bounds):
    from config import prepare_config

    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    config["bot"]["portfolio"]["strategy_kind"] = strategy
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    config["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        config["bot"][side]["risk"].update(
            max_positions=1,
            total_wallet_exposure_limit=float(side == "long" or short_bounds == "explicit_zero"),
        )
        config["bot"][side]["forager"]["score_weights"] = dict(
            volume=0.0, volatility=0.0, ema_readiness=0.0, unilateralness=0.0,
        )
    for index, name in enumerate(("volume", "volatility", "ema_readiness", "unilateralness")):
        _set_bounds(config, {f"long_forager_score_weights_{name}": [0.0, float(index + 1)]})
        if short_bounds == "explicit_zero":
            _set_bounds(config, {f"short_forager_score_weights_{name}": [0.0]})
    return prepare_config(config, verbose=False)


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("short_bounds", ["omitted", "explicit_zero"])
@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
def test_mirrored_candidate_weights_follow_source_normalization_at_every_corner(
    strategy, short_bounds, fraction,
):
    from optimize import individual_to_config
    from optimization.shape import build_optimization_shape
    from optimizer_overrides import optimizer_overrides

    config = _mirrored_weight_config(strategy, short_bounds)
    shape = build_optimization_shape(config)
    vector = [bound.low + fraction * (bound.high - bound.low) for bound in shape.bounds]
    candidate = individual_to_config(
        vector, optimizer_overrides, config["optimize"]["enable_overrides"], config,
        key_paths=shape.key_paths,
    )
    expected = (dict(volume=0.0, volatility=0.0, ema_readiness=1.0, unilateralness=0.0)
                if fraction == 0.0 else
                dict(volume=0.1, volatility=0.2, ema_readiness=0.3, unilateralness=0.4))
    for side in ("long", "short"):
        assert candidate["bot"][side]["forager"]["score_weights"] == pytest.approx(expected)
        assert candidate["bot"][side]["risk"]["total_wallet_exposure_limit"] == 1.0


def test_disabling_mirror_keeps_independent_fixed_zero_weights_zero():
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config

    config = _mirrored_weight_config("trailing_martingale", "explicit_zero")
    shape = build_optimization_shape(config)
    candidate = build_optimizer_vector_config(
        [bound.low for bound in shape.bounds], config, key_paths=shape.key_paths,
        overrides_list=[],
    )
    assert candidate["bot"]["long"]["forager"]["score_weights"]["ema_readiness"] == 1.0
    assert candidate["bot"]["short"]["forager"]["score_weights"] == dict(
        volume=0.0, volatility=0.0, ema_readiness=0.0, unilateralness=0.0,
    )


def test_mirrored_fixed_zero_source_group_does_not_enable_fallback():
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config

    config = _mirrored_weight_config("trailing_martingale", "explicit_zero")
    for name in ("volume", "volatility", "ema_readiness", "unilateralness"):
        _set_bounds(config, {f"long_forager_score_weights_{name}": [0.0]})
    shape = build_optimization_shape(config)
    candidate = build_optimizer_vector_config(
        [bound.low for bound in shape.bounds], config, key_paths=shape.key_paths,
    )
    for side in ("long", "short"):
        assert candidate["bot"][side]["forager"]["score_weights"] == dict(
            volume=0.0, volatility=0.0, ema_readiness=0.0, unilateralness=0.0,
        )


@pytest.mark.parametrize("pin_side", ["long", "short"])
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_runtime_zero_weight_pins_keep_effective_source_authority(pin_side, mirror, fraction):
    from config import prepare_config
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config

    config = _mirrored_weight_config("trailing_martingale", "explicit_zero")
    names = ("volume", "volatility", "ema_readiness", "unilateralness")
    config["optimize"]["fixed_runtime_overrides"] = {
        f"bot.{pin_side}.forager.score_weights.{name}": 0.0 for name in names
    }
    config = prepare_config(config, verbose=False)
    shape = build_optimization_shape(config)
    candidate = build_optimizer_vector_config(
        [bound.low + fraction * (bound.high - bound.low) for bound in shape.bounds],
        config, key_paths=shape.key_paths,
        overrides_list=["mirror_short_from_long"] if mirror else [],
    )
    zero = dict.fromkeys(names, 0.0)
    if pin_side == "long":
        assert candidate["bot"]["long"]["forager"]["score_weights"] == zero
        assert candidate["bot"]["short"]["forager"]["score_weights"] == zero
    elif mirror:
        # Short runtime pins precede, and are replaced by, the numeric mirror.
        assert candidate["bot"]["short"]["forager"]["score_weights"] == (
            candidate["bot"]["long"]["forager"]["score_weights"]
        )
        assert sum(candidate["bot"]["short"]["forager"]["score_weights"].values()) == pytest.approx(1.0)
    else:
        assert candidate["bot"]["short"]["forager"]["score_weights"] == zero


@pytest.mark.parametrize("mirror", [False, True])
def test_runtime_zero_weight_object_keeps_leaf_only_permission_boundary(mirror):
    from config import prepare_config

    config = _mirrored_weight_config("trailing_martingale", "explicit_zero")
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long"] if mirror else []
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.forager.score_weights": dict(
            volume=0.0, volatility=0.0, ema_readiness=0.0, unilateralness=0.0,
        ),
    }
    with pytest.raises(TypeError, match="must target leaf settings"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("risk_value", [0.0, 1.0])
@pytest.mark.parametrize("searched_weights", [False, True])
def test_selected_anchor_owns_fixed_zero_weights_when_risk_tuning_activates_side(
    mirror, risk_value, searched_weights,
):
    from copy import deepcopy

    from config import prepare_config
    from optimize import install_anchored_fine_tune_plan
    from optimization.fine_tune_anchors import ANCHOR_GENE_KEY
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config

    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    config["optimize"]["bounds"] = {}
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    names = ("volume", "volatility", "ema_readiness", "unilateralness")
    zero = dict.fromkeys(names, 0.0)
    positive = dict.fromkeys(names, 0.25)
    for side in ("long", "short"):
        config["bot"][side]["risk"].update(max_positions=1, total_wallet_exposure_limit=0.0)
        config["bot"][side]["forager"]["score_weights"] = deepcopy(zero)
    _set_bounds(config, {"long_total_wallet_exposure_limit": [0.0, 1.0]})
    for name in names:
        _set_bounds(config, {f"long_forager_score_weights_{name}": [0.0, 1.0]})
        _set_bounds(config, {f"short_forager_score_weights_{name}": [0.0]})
    config = prepare_config(config, verbose=False)
    anchors = [deepcopy(config), deepcopy(config)]
    anchors[1]["bot"]["long"]["forager"]["score_weights"] = deepcopy(positive)
    selectors = ["long.risk.total_wallet_exposure_limit"]
    if searched_weights:
        selectors.append("long.forager.score_weights")
    install_anchored_fine_tune_plan(
        config, selectors, "unused", starting_configs_override=anchors,
    )
    shape = build_optimization_shape(config)
    for anchor_index, anchor_weights in enumerate((zero, positive)):
        vector = [float(anchor_index) if key == ANCHOR_GENE_KEY else
                  risk_value if key == "long_total_wallet_exposure_limit" else bound.low
                  for (key, _), bound in zip(shape.key_paths, shape.bounds, strict=True)]
        candidate = build_optimizer_vector_config(
            vector, config, overrides_list=["mirror_short_from_long"] if mirror else [],
        )
        expected = (dict(volume=0.0, volatility=0.0, ema_readiness=1.0, unilateralness=0.0)
                    if searched_weights else anchor_weights)
        assert candidate["_optimizer_anchor"]["id"] == anchor_index
        assert candidate["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == risk_value
        assert candidate["bot"]["long"]["forager"]["score_weights"] == expected
        assert candidate["bot"]["short"]["forager"]["score_weights"] == (expected if mirror else zero)


@pytest.mark.parametrize("authority", ["dormant", "runtime", "anchor"])
@pytest.mark.parametrize("mirror", [False, True])
def test_preflight_validates_effective_zero_weight_activation_without_mutating_genes(
    monkeypatch, authority, mirror,
):
    from copy import deepcopy

    from config import prepare_config
    from optimize import install_anchored_fine_tune_plan
    from optimization import warmup

    config = get_template_config()
    for activation_side in ("long", "short"):
        config["bot"][activation_side]["enabled"] = True
    config["optimize"]["bounds"] = {}
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long"] if mirror else []
    names = ("volume", "volatility", "ema_readiness", "unilateralness")
    zero = dict.fromkeys(names, 0.0)
    for side in ("long", "short"):
        config["bot"][side]["risk"].update(max_positions=1, total_wallet_exposure_limit=0.0)
        config["bot"][side]["forager"]["score_weights"] = deepcopy(zero)
    _set_bounds(config, {"long_total_wallet_exposure_limit": [0.0, 1.0]})
    if authority != "dormant":
        for name in names:
            _set_bounds(config, {f"long_forager_score_weights_{name}": [0.0, 1.0]})
    if authority == "runtime":
        config["optimize"]["fixed_runtime_overrides"] = {
            f"bot.long.forager.score_weights.{name}": 0.0 for name in names
        }
    config = prepare_config(config, verbose=False)
    if authority == "anchor":
        install_anchored_fine_tune_plan(
            config, ["long.risk.total_wallet_exposure_limit"], "unused",
            starting_configs_override=[deepcopy(config)],
        )
    before = deepcopy(config)
    boundary_builder = warmup._build_optimizer_boundary_configs
    captured = []

    def capture(source):
        candidates = boundary_builder(source)
        captured.extend((candidate, deepcopy(candidate)) for candidate in candidates)
        return candidates

    monkeypatch.setattr(warmup, "_build_optimizer_boundary_configs", capture)
    validate_optimizer_effective_configs(config)
    assert config == before
    assert captured
    for candidate, original in captured:
        assert candidate == original
        assert candidate["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.0
        assert candidate["bot"]["long"]["forager"]["score_weights"] == zero


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), "invalid"])
def test_effective_preflight_keeps_forager_normalization_owner_errors(monkeypatch, invalid):
    from config import prepare_config
    from optimization import warmup

    config = prepare_config(get_template_config(), verbose=False)
    candidate = warmup.build_optimizer_max_config(config)
    candidate["bot"]["long"]["forager"]["score_weights"]["volume"] = invalid
    monkeypatch.setattr(warmup, "_build_optimizer_boundary_configs", lambda _config: [candidate])
    with pytest.raises(
        (TypeError, ValueError),
        match=r"bot\.long\.forager\.score_weights\.volume.*(numeric|finite and non-negative)",
    ):
        validate_optimizer_effective_configs(config)
