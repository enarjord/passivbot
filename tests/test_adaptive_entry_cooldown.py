"""RMS, dynamic duration, side scoping and disabled-policy regressions."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from config import get_template_config, prepare_config
from config.entry_cooldown import maximum_duration
from test_orchestrator_json_api import make_input, make_symbol, bot_params_pair, compute
from live import reconciler


def adaptive_input(*, score=-0.75, exposure=0.0, base=0.0):
    bp = {
        "risk_entry_cooldown_minutes": base,
        "entry_cooldown_min_duration_minutes": 0.0,
        "entry_cooldown_max_duration_minutes": 30.0,
        "entry_cooldown_weights_minutes": {
            "exposure_ratio": exposure,
            "adverse_directionality": 20.0,
        },
        "unilateralness_ema_span_1m": 60.0,
    }
    inp = make_input(
        balance=1000.0,
        global_bp=bot_params_pair(long_overrides=bp),
        symbols=[make_symbol(0, bid=100.0, ask=100.0, long_bp=bp)],
    )
    inp["symbols"][0]["emas"]["m1"]["signed_unilateralness"] = [[60.0, score]]
    inp["symbols"][0]["long"]["last_increase_fill_timestamp_ms"] = 60000
    inp["timestamp_ms"] = 60000 + 10 * 60000
    return inp


def entries(result):
    return [
        o for o in result["orders"] if o["pside"] == "long" and o["order_type"].startswith("entry_")
    ]


def test_zero_base_adverse_delay_expires_and_recomputes():
    import passivbot_rust as pbr

    inp = adaptive_input()
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] == 15
    assert not entries(compute(pbr, inp))
    # Restart is a fresh input, with no hidden countdown state.
    replay = json.loads(json.dumps(inp))
    replay["timestamp_ms"] = 16 * 60000
    assert entries(compute(pbr, replay))
    # Improving conditions can also reopen the gate without elapsed-time changes.
    inp["symbols"][0]["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.75]]
    assert entries(compute(pbr, inp))


def test_disabled_input_and_missing_enabled_input_are_distinct():
    import passivbot_rust as pbr

    inp = adaptive_input()
    inp["symbols"][0]["emas"]["m1"]["signed_unilateralness"] = []
    inp["symbols"][0]["allow_missing_strategy_inputs"] = True
    out = compute(pbr, inp)
    assert not entries(out)
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] is None
    inp["symbols"][0]["long"]["bot_params"]["entry_cooldown_weights_minutes"][
        "adverse_directionality"
    ] = 0
    assert entries(compute(pbr, inp))


def test_rms_real_replay_and_flat_tail():
    import passivbot_rust as pbr

    span = 12.5
    n = math.ceil(20 * span)
    rng = np.random.default_rng(71)
    closes = np.exp(np.cumsum(rng.normal(0.001, 0.003, n + 40)))

    def reference(xs):
        m = q = 0.0
        a = 2 / (span + 1)
        for r in np.diff(np.log(xs[-n - 1 :])):
            m += a * (r - m)
            q += a * (r * r - q)
        return m / math.sqrt(q) if q else 0.0

    assert pbr.calc_signed_unilateralness(closes.tolist(), span) == pytest.approx(
        reference(closes), abs=2e-13
    )
    assert pbr.calc_signed_unilateralness(
        closes[-n - 1 :].tolist(), span
    ) == pbr.calc_signed_unilateralness(closes.tolist(), span)
    flat = np.r_[closes, np.full(20, closes[-1])]
    assert abs(pbr.calc_signed_unilateralness(flat.tolist(), span)) < abs(reference(closes))
    with pytest.raises(ValueError):
        pbr.calc_signed_unilateralness([1.0, float("nan")] * (n + 1), span)


def test_config_limits_zero_weights():
    cfg = prepare_config(get_template_config(), verbose=False)
    ec = cfg["bot"]["long"]["entry_cooldown"]
    assert maximum_duration(ec) == 24.1
    ec["base_duration_minutes"] = 0
    ec["weights_minutes"]["exposure_ratio"] = 10
    with pytest.raises(ValueError, match="finite"):
        maximum_duration(ec)
    ec["max_duration_minutes"] = 30
    assert maximum_duration(ec) == 30


def test_adaptive_output_validator_checks_current_duration():
    import passivbot_rust as pbr

    inp = adaptive_input()
    raw = pbr.compute_ideal_orders_json(json.dumps(inp))
    reconciler.parse_and_validate_rust_orchestrator_output(raw, {0: "BTC/USDT:USDT"}, inp)
    # An otherwise valid early entry is invalid under the submitted adverse score.
    relaxed = copy.deepcopy(inp)
    relaxed["symbols"][0]["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.75]]
    out = compute(pbr, relaxed)
    from passivbot_exceptions import FatalBotException

    with pytest.raises(FatalBotException, match="cooldown"):
        reconciler.validate_rust_orchestrator_output(out, {0: "BTC/USDT:USDT"}, inp)


def test_rms_forager_prefers_lower_penalty_and_ignores_direction_sign():
    import passivbot_rust as pbr

    bp = {
        "n_positions": 1,
        "forager_score_weights": {
            "volume": 0.0,
            "volatility": 0.0,
            "ema_readiness": 0.0,
            "unilateralness": 1.0,
        },
    }
    symbols = [make_symbol(i, bid=100.0, ask=100.0, long_bp=bp) for i in range(3)]
    for symbol, score in zip(symbols, [-0.8, 0.2, 0.9]):
        symbol["long"]["mode"] = None
        symbol["emas"]["m1"]["signed_unilateralness"] = [[60.0, score]]
    inp = make_input(balance=1000.0, global_bp=bot_params_pair(long_overrides=bp), symbols=symbols)
    result = compute(pbr, inp)
    selection = next(s for s in result["diagnostics"]["forager_selections"] if s["pside"] == "long")
    assert selection["selected_symbol_indices"] == [1]
    reconciler.validate_rust_orchestrator_output(result, {i: str(i) for i in range(3)}, inp)


@pytest.mark.asyncio
@pytest.mark.parametrize("scoring_weight", [0.0, 1.0])
async def test_live_rms_uses_completed_contiguous_rows_and_replays(monkeypatch, scoring_weight):
    from live.unilateralness import load
    from candlestick_manager import CANDLE_DTYPE

    span = 2.5
    n = math.ceil(20 * span) + 1
    rows = np.zeros(n, dtype=CANDLE_DTYPE)
    rows["ts"] = np.arange(n) * 60000
    rows["c"] = np.exp(np.arange(n) * 0.001)

    class CM:
        async def get_candles(self, *args, **kwargs):
            assert kwargs["standardize"] is False
            assert kwargs["allow_remote_fetch"] is False
            assert kwargs["end_ts"] in ((n - 1) * 60000, (n + 1) * 60000)
            return rows.copy()

    bot = SimpleNamespace(
        cm=CM(),
        is_forager_mode=lambda side: True,
        is_approved=lambda side, symbol: True,
        is_pside_enabled=lambda side: True,
        get_exchange_time=lambda: n * 60000 + 30000,
        bp=lambda side, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
        bot_value=lambda side, key: (
            {"unilateralness": scoring_weight} if key == "forager_score_weights" else span
        ),
    )
    first, ranking, missing = await load(bot, ["BTC"], {"BTC"})
    assert not missing
    assert first["BTC"][span] > 0.99
    assert await load(bot, ["BTC"], {"BTC"}) == (first, ranking, missing)
    bot.get_exchange_time = lambda: (n + 2) * 60000 + 30000
    current, rank, missing = await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000})
    assert current["BTC"] == {} and rank == first and missing == {"BTC": {"current": [span], "forager": []}}
    bot.get_exchange_time = lambda: n * 60000 + 30000
    rows = rows[np.arange(n) != n // 2]
    second, ranking, missing = await load(bot, ["BTC"], {"BTC"})
    assert missing == {"BTC": {"current": [span], "forager": [span] if scoring_weight else []}}
    assert second["BTC"] == {}


def test_cooldown_cli_and_partial_override_keep_effective_weights():
    import argparse
    from config.overrides import parse_overrides
    from config_utils import add_config_arguments, update_config_with_args
    from passivbot import Passivbot
    from config.strategy import merge_runtime_bot_side

    cfg = get_template_config()
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, cfg, command="backtest", help_all=True, group_map={})
    args = parser.parse_args(
        [
            "--bot.long.entry_cooldown.max_duration_minutes",
            "30",
            "--bot.long.entry_cooldown.weights_minutes.exposure_ratio",
            "10",
            "--bot.long.entry_cooldown.weights_minutes.adverse_directionality",
            "20",
        ]
    )
    update_config_with_args(cfg, args, verbose=False, allowed_keys=keys)
    cfg["coin_overrides"] = {
        "BTC": {"bot": {"long": {"entry_cooldown": {"weights_minutes": {"exposure_ratio": 0.0}}}}}
    }
    cfg = parse_overrides(prepare_config(cfg, verbose=False), verbose=False)
    side = cfg["bot"]["long"]
    bot = SimpleNamespace(config=cfg, coin_overrides=cfg["coin_overrides"])
    bot.bot_value = lambda side, key: Passivbot.bot_value(bot, side, key)
    merged = Passivbot.config_get(bot, ["bot", "long", "entry_cooldown_weights_minutes"], "BTC")
    assert merged == {"exposure_ratio": 0.0, "adverse_directionality": 20.0}
    runtime = merge_runtime_bot_side(
        side, pside="long", override_side=cfg["coin_overrides"]["BTC"]["bot"]["long"]
    )
    assert runtime["entry_cooldown_weights_minutes"] == merged
    assert runtime["entry_cooldown_max_duration_minutes"] == 30.0


def test_zero_base_adaptive_fill_horizon_uses_ceiling():
    from passivbot import Passivbot

    cfg = prepare_config(get_template_config(), verbose=False)
    ec = cfg["bot"]["long"]["entry_cooldown"]
    ec.update(base_duration_minutes=0.0, max_duration_minutes=45.0)
    ec["weights_minutes"]["adverse_directionality"] = 20.0
    bot = SimpleNamespace(config=cfg, coin_overrides={})
    bot.bot_value = lambda side, key: Passivbot.bot_value(bot, side, key)
    bot.is_pside_enabled = lambda side: Passivbot.is_pside_enabled(bot, side)
    bot.bp = lambda side, key, symbol=None: Passivbot.bot_value(bot, side, key)
    assert Passivbot._max_configured_entry_cooldown_minutes(bot) == 45.0
    assert Passivbot._entry_cooldown_enabled_pairs(bot, ["BTC"]) == {("BTC", "long")}


def test_optimizer_paths_and_warmup_follow_enabled_bounds():
    from optimization.config_adapter import get_optimization_key_paths
    from optimization.warmup import compute_optimizer_per_coin_warmup_minutes

    cfg = get_template_config()
    cfg["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = 90.0
    cfg["live"]["max_warmup_minutes"] = 1
    cfg["optimize"]["bounds"]["long"]["entry_cooldown"]["weights_minutes"] = {
        "adverse_directionality": [0.0, 20.0]
    }
    cfg["optimize"]["bounds"]["long"]["forager"]["unilateralness_ema_span_1m"] = [60.0, 240.5]
    cfg = prepare_config(cfg, verbose=False)
    paths = dict(get_optimization_key_paths(cfg))
    assert paths["long_entry_cooldown_weights_minutes_adverse_directionality"] == (
        "bot",
        "long",
        "entry_cooldown",
        "weights_minutes",
        "adverse_directionality",
    )
    assert (
        compute_optimizer_per_coin_warmup_minutes(cfg)["__default__"] == math.ceil(20 * 240.5) + 1
    )


def test_duration_floor_ceiling_and_uncapped_exposure():
    import passivbot_rust as pbr

    inp = adaptive_input(exposure=10.0, base=5.0)
    side = inp["symbols"][0]["long"]
    side["position"] = {"size": 20.0, "price": 100.0}
    side["bot_params"]["wallet_exposure_limit"] = 1.0
    side["bot_params"]["entry_cooldown_max_duration_minutes"] = 100.0
    # WE=2, ratio=2, base 5 + adverse .75*20 + exposure 2*10 = 40.
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] == 40.0
    side["bot_params"]["entry_cooldown_max_duration_minutes"] = 30.0
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] == 30.0
    side["bot_params"]["entry_cooldown_min_duration_minutes"] = 12.0
    side["position"]["size"] = 0.0
    inp["symbols"][0]["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.8]]
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] == 12.0


def test_zero_exposure_allowance_saturates_ceiling():
    import passivbot_rust as pbr

    inp = adaptive_input(exposure=10.0)
    side = inp["symbols"][0]["long"]
    side["position"] = {"size": 20.0, "price": 100.0}
    side["bot_params"]["wallet_exposure_limit"] = 0.0
    side["runtime_budget"] = {
        "effective_wallet_exposure_limit": 0.0,
        "effective_n_positions": 1,
        "configured_wallet_exposure_limit": 0.0,
        "configured_n_positions": 1,
    }
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] == 30.0


def test_optional_bounds_survive_export_without_changing_defaults():
    from config_utils import clean_config
    from config.optimize_bounds import flatten_optimize_bounds

    cfg = get_template_config()
    original = flatten_optimize_bounds(
        cfg["optimize"]["bounds"], strategy_kind=cfg["live"]["strategy_kind"]
    )
    assert not any("unilateralness" in key or "weights_minutes" in key for key in original)
    cfg["optimize"]["bounds"]["long"]["forager"]["score_weights"] = {"unilateralness": [0.0, 1.0]}
    cfg["optimize"]["bounds"]["long"]["forager"]["unilateralness_ema_span_1m"] = [20.0, 80.5]
    cfg = clean_config(prepare_config(cfg, verbose=False))
    bounds = flatten_optimize_bounds(
        cfg["optimize"]["bounds"], strategy_kind=cfg["live"]["strategy_kind"]
    )
    assert bounds["long_forager_score_weights_unilateralness"] == [0.0, 1.0]
    assert bounds["long_unilateralness_ema_span_1m"] == [20.0, 80.5]


def test_forager_can_rank_carried_score_while_current_cooldown_is_unavailable():
    import passivbot_rust as pbr

    inp = adaptive_input()
    symbol = inp["symbols"][0]
    symbol["emas"]["m1"]["signed_unilateralness"] = []
    symbol["forager_m1"] = copy.deepcopy(symbol["emas"]["m1"])
    symbol["forager_m1"]["signed_unilateralness"] = [[60.0, 0.3]]
    symbol["unilateralness_unavailable"] = {"current": [60.0]}
    result = compute(pbr, inp)
    assert not entries(result)
    # Ranking observations must never satisfy a current cooldown input.
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))["0:long"] is None


@pytest.mark.parametrize("ranking_required", [False, True])
@pytest.mark.parametrize("metadata_history", [3, 1201])
@pytest.mark.parametrize("span", [1.0, 60.0])
def test_cpu_score_only_rms_warmup_blocks_only_required_ranking(ranking_required, metadata_history, span):
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    if ranking_required:
        cfg["live"]["approved_coins"] = {
            side: ["LONGCOIN", "SHORTCOIN"] for side in ("long", "short")
        }
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    baseline, _, _ = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert len(baseline) > 0
    for side in ("long", "short"):
        cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
        cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = span
    for coin in ("LONGCOIN", "SHORTCOIN"):
        markets[coin]["warmup_minutes"] = metadata_history
        markets[coin]["warmup_minutes_source"] = "history"
    fills, _, _ = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    if ranking_required and span == 60.0:
        assert len(fills) == 0
    elif ranking_required:
        assert len(fills) > 0
        assert min(int(row[0]) for row in fills) >= math.ceil(20 * span)
    else:
        assert fills.tolist() == baseline.tolist()


def test_cpu_adverse_cooldown_still_waits_for_full_rms_history():
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    for side in ("long", "short"):
        cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 1.0
        cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    fills, _, _ = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert len(fills) == 0


@pytest.mark.parametrize("side,score", [("long", -0.6), ("short", 0.6)])
def test_adverse_sign_and_missing_indicator_keep_independent_closes(side, score):
    import passivbot_rust as pbr

    inp = adaptive_input(score=score)
    symbol = inp["symbols"][0]
    if side == "short":
        inp["global"]["global_bot_params"]["short"] = copy.deepcopy(
            inp["global"]["global_bot_params"]["long"]
        )
        symbol["short"]["bot_params"] = copy.deepcopy(symbol["long"]["bot_params"])
        symbol["short"]["last_increase_fill_timestamp_ms"] = 60000
    symbol[side]["position"] = {"size": 1.0 if side == "long" else -1.0, "price": 100.0}
    assert json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))[f"0:{side}"] == 12.0
    symbol["emas"]["m1"]["signed_unilateralness"] = []
    symbol["allow_missing_strategy_inputs"] = True
    result = compute(pbr, inp)
    assert not any(
        o["pside"] == side and o["order_type"].startswith("entry_") for o in result["orders"]
    )
    assert any(
        o["pside"] == side and o["order_type"].startswith("close_") for o in result["orders"]
    )


def test_invalid_submitted_adaptive_inputs_are_fatal():
    import passivbot_rust as pbr
    from passivbot_exceptions import FatalBotException

    inp = adaptive_input()
    out = compute(pbr, inp)
    inp["symbols"][0]["long"]["bot_params"]["entry_cooldown_max_duration_minutes"] = None
    with pytest.raises(FatalBotException, match="entry cooldown inputs"):
        reconciler.validate_rust_orchestrator_output(out, {0: "BTC"}, inp)


@pytest.mark.parametrize("pending_span,score,raises", [(60.0, None, False), (30.0, None, True), (60.0, 1.5, True)])
def test_cpu_rms_warmup_defer_is_span_specific_and_never_masks_invalid_scores(pending_span, score, raises):
    import passivbot_rust as pbr

    bp = {"n_positions": 1, "forager_score_weights": {
        "volume": 0.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": 1.0,
    }}
    symbols = [make_symbol(i, bid=100.0, ask=100.0, long_bp=bp) for i in range(2)]
    for symbol in symbols:
        symbol["long"]["mode"] = None
        symbol["unilateralness_warmup_spans"] = [pending_span]
        symbol["emas"]["m1"]["signed_unilateralness"] = [] if score is None else [[60.0, score]]
    inp = make_input(balance=1000.0, global_bp=bot_params_pair(long_overrides=bp), symbols=symbols)
    if raises:
        with pytest.raises(ValueError):
            compute(pbr, inp)
    else:
        result = compute(pbr, inp)
        selection = next(s for s in result["diagnostics"]["forager_selections"] if s["pside"] == "long")
        assert selection["ranking_required"]
        assert selection["selected_symbol_indices"] == []
        assert result["diagnostics"]["warnings"]
        # Once only one eligible candidate remains, no score is consumed.
        inp["symbols"].pop()
        result = compute(pbr, inp)
        selection = next(s for s in result["diagnostics"]["forager_selections"] if s["pside"] == "long")
        assert not selection["ranking_required"]
        assert selection["selected_symbol_indices"] == [0]


@pytest.mark.parametrize("pending_span,score,raises", [(60.0, None, False), (30.0, None, True), (60.0, 1.5, True)])
@pytest.mark.parametrize("side", ["long", "short"])
def test_adverse_warmup_marker_is_scoped_and_keeps_closes(side, pending_span, score, raises):
    import passivbot_rust as pbr

    inp = adaptive_input()
    symbol = inp["symbols"][0]
    if side == "short":
        inp["global"]["global_bot_params"]["short"] = copy.deepcopy(inp["global"]["global_bot_params"]["long"])
        symbol["short"]["bot_params"] = copy.deepcopy(symbol["long"]["bot_params"])
        symbol["long"]["bot_params"]["entry_cooldown_weights_minutes"]["adverse_directionality"] = 0.0
    symbol[side]["position"] = {"size": 1.0 if side == "long" else -1.0, "price": 100.0}
    symbol["emas"]["m1"]["signed_unilateralness"] = [] if score is None else [[60.0, score]]
    symbol["unilateralness_warmup_spans"] = [pending_span]
    if raises:
        with pytest.raises(ValueError):
            compute(pbr, inp)
    else:
        result = compute(pbr, inp)
        assert not any(o["pside"] == side and o["order_type"].startswith("entry_") for o in result["orders"])
        assert any(o["pside"] == side and o["order_type"].startswith("close_") for o in result["orders"])
        assert result["diagnostics"]["warnings"]
        # The marker expires; missing required inputs are strict again.
        symbol["unilateralness_warmup_spans"] = []
        with pytest.raises(ValueError):
            compute(pbr, inp)


@pytest.mark.parametrize("bad_close", [0.0, -1.0])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("bad_row", [0, 25])
def test_cpu_rms_rejects_nonpositive_close_with_normal_backtest_error(bad_close, consumer, side, bad_row):
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 1.0
    if consumer == "forager":
        cfg["live"]["approved_coins"][side] = ["LONGCOIN", "SHORTCOIN"]
        cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
    else:
        cfg["bot"][side]["entry_cooldown"].update(
            max_duration_minutes=60.0,
            weights_minutes={"exposure_ratio": 0.0, "adverse_directionality": 10.0},
        )
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    coin_index, coin = (0, "LONGCOIN") if side == "long" else (1, "SHORTCOIN")
    hlcvs[bad_row, coin_index, 2] = bad_close
    with pytest.raises(ValueError, match=f"RMS requires positive closes: coin {coin} index {coin_index} candle {bad_row}"):
        run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    # Unavailable history outside the declared listing range is not consumed.
    markets[coin]["first_valid_index"] = bad_row + 1
    run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)


@pytest.mark.asyncio
@pytest.mark.parametrize("active_side", [None, "long", "short"])
async def test_live_rms_skips_dormant_side_spans(active_side):
    from live.unilateralness import load
    from candlestick_manager import CANDLE_DTYPE

    calls = []
    async def candles(*args, **kwargs):
        calls.append(kwargs)
        # The disabled side's 100000-minute span must not enlarge this request.
        assert kwargs["end_ts"] - kwargs["start_ts"] == 20 * 60000
        rows = np.zeros(21, dtype=CANDLE_DTYPE)
        rows["ts"] = np.arange(21) * 60000
        rows["c"] = 100.0
        return rows

    bot = SimpleNamespace(
        cm=SimpleNamespace(get_candles=candles),
        get_exchange_time=lambda: 21 * 60000,
        is_forager_mode=lambda side: True,
        is_approved=lambda side, symbol: True,
        is_pside_enabled=lambda side: side == active_side,
        bp=lambda side, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 10.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
        bot_value=lambda side, key: (
            {"unilateralness": 1.0} if key == "forager_score_weights"
            else (1.0 if side == active_side else 100000.0)
        ),
    )
    result, ranking, missing = await load(bot, ["BTC"], {"BTC"})
    expected = {"BTC": {1.0: 0.0} if active_side else {}}
    assert result == ranking == expected
    assert not missing
    assert len(calls) == (1 if active_side else 0)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("gate", ["n_positions", "total_wallet_exposure_limit"])
@pytest.mark.parametrize("interval", [1, 5])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
def test_cpu_dormant_rms_matches_disabled_policy(side, gate, interval, consumer):
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["bot"][side]["risk"][gate] = 0
    cfg["bot"][side]["risk"]["total_wallet_exposure_limit"] = 0.0
    cfg["optimize"]["bounds"][side]["risk"][gate] = [0, 0]
    cfg["backtest"]["candle_interval_minutes"] = interval
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    timestamps = timestamps[0] + np.arange(len(timestamps)) * interval * 60000
    baseline = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 100000.0
    if consumer == "forager":
        cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
    else:
        cfg["bot"][side]["entry_cooldown"].update(
            max_duration_minutes=60.0,
            weights_minutes={"exposure_ratio": 0.0, "adverse_directionality": 10.0},
        )
    result = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert len(result[0]) > 0
    np.testing.assert_array_equal(result[0], baseline[0])
    np.testing.assert_array_equal(result[1], baseline[1])
    assert result[2] == baseline[2]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("interval", [1, 5])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
def test_cpu_entry_ineligible_rms_policy_is_inert(side, interval, consumer):
    from config_utils import clean_config
    from backtest import build_backtest_payload, execute_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["backtest"]["candle_interval_minutes"] = interval
    # Keep the side globally enabled, but exclude every loaded coin on that side.
    cfg["live"]["approved_coins"][side] = ["UNLOADEDCOIN"]
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    payload = build_backtest_payload(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert all(not pair[side]["entry_eligible"] for pair in payload.bot_params_list)
    baseline = execute_backtest(payload, cfg)
    # Load a normal user config before dataset selection finalizes eligibility.
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 100000.0
    if consumer == "forager":
        cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
    else:
        cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
        cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 10.0
    cfg = prepare_config(clean_config(cfg), verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    payload = build_backtest_payload(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert all(not pair[side]["entry_eligible"] for pair in payload.bot_params_list)
    result = execute_backtest(payload, cfg)
    assert len(result[0]) > 0
    np.testing.assert_array_equal(result[0], baseline[0])
    np.testing.assert_array_equal(result[1], baseline[1])
    assert result[2] == baseline[2]
    if interval == 5:
        for pair in payload.bot_params_list:
            pair[side]["entry_eligible"] = True
        with pytest.raises(ValueError, match="RMS unilateralness requires.*one-minute"):
            execute_backtest(payload, cfg)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("override", [False, True])
def test_disabled_side_cooldown_does_not_extend_fill_coverage(side, override):
    from passivbot import Passivbot

    bot = Passivbot.__new__(Passivbot)
    bot.config = prepare_config(get_template_config(), verbose=False)
    other = "short" if side == "long" else "long"
    for pside in (side, other):
        bot.config["bot"][pside]["entry_cooldown"]["base_duration_minutes"] = 0.0
        bot.config["bot"][pside]["risk"].update(n_positions=1, total_wallet_exposure_limit=0.0)
    bot.config["bot"][other]["risk"]["total_wallet_exposure_limit"] = 1.0
    bot.config["bot"][other]["entry_cooldown"]["base_duration_minutes"] = 5.0
    dormant = bot.config["bot"][side]["entry_cooldown"]
    dormant["max_duration_minutes"] = 7 * 24 * 60
    dormant["weights_minutes"]["exposure_ratio"] = 10.0
    bot.coin_overrides = {}
    bot.is_approved = lambda pside, symbol: True
    if override:
        bot.coin_overrides = {
            "BTC": {"bot": {side: {"entry_cooldown": {"max_duration_minutes": 14 * 24 * 60}}}}
        }
    from config.runtime_compile import compile_runtime_config
    bot.config = compile_runtime_config(bot.config, runtime="live")
    bot._equity_hard_stop_enabled = lambda: False
    bot._live_risk_uses_authoritative_pnl = lambda: False
    now = 30 * 24 * 60 * 60000
    assert bot._max_configured_entry_cooldown_minutes() == 5.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (True, now - 6 * 60000)
    # With both sides disabled there is no cooldown-only history requirement.
    bot.config["bot"][other]["risk"]["total_wallet_exposure_limit"] = 0.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (False, None)
    # Reactivation restores the configured side/coin horizon.
    bot.config["bot"][side]["risk"]["total_wallet_exposure_limit"] = 1.0
    expected = (14 if override else 7) * 24 * 60
    assert bot._max_configured_entry_cooldown_minutes() == expected
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (True, now - (expected + 1) * 60000)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("span", [1.0, 1.01, 2.25])
def test_adverse_rms_activates_at_first_complete_return_window(side, span):
    import passivbot_rust as pbr
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs
    from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes

    cfg = _ema_anchor_config(True)
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = span
    cfg["bot"][side]["entry_cooldown"].update(
        base_duration_minutes=0.0, max_duration_minutes=60.0,
        weights_minutes={"exposure_ratio": 0.0, "adverse_directionality": 1.0},
    )
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    n_returns = math.ceil(20 * span)
    # History is a close count; activation is the elapsed offset from the first close.
    assert compute_backtest_warmup_minutes(cfg) == n_returns + 1
    assert compute_per_coin_warmup_minutes(cfg)["__default__"] == n_returns + 1
    assert compute_backtest_warmup_minutes(cfg, for_trade_activation=True) == n_returns
    assert compute_per_coin_warmup_minutes(cfg, for_trade_activation=True)["__default__"] == n_returns
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    first = 3
    for coin in ("LONGCOIN", "SHORTCOIN"):
        markets[coin]["first_valid_index"] = first
        markets[coin]["warmup_minutes"] = n_returns + 1
        markets[coin]["warmup_minutes_source"] = "history"
    coin_index, coin = (0, "LONGCOIN") if side == "long" else (1, "SHORTCOIN")
    closes = hlcvs[first:first + n_returns + 1, coin_index, 2].tolist()
    with pytest.raises(ValueError, match="incomplete unilateralness warmup"):
        pbr.calc_signed_unilateralness(closes[:-1], span)
    assert pbr.calc_signed_unilateralness(closes, span) == 0.0
    fills, _, _, payload = run_backtest(
        hlcvs, markets, cfg, "binance", btc, timestamps, return_payload=True,
    )
    assert payload.backtest_params["trade_start_indices"][coin_index] < first + n_returns
    entries = [row for row in fills if str(row[2]) == coin and str(row[13]).startswith("entry_")]
    # Orders planned with the first complete window fill on the next candle.
    assert min(int(row[0]) for row in entries) == first + n_returns + 1


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("span", [1.0, 60.0])
@pytest.mark.parametrize("target", ["LONGCOIN", "SHORTCOIN"])
def test_adverse_rms_warmup_does_not_delay_other_sides_or_coins(side, span, target):
    from backtest import run_backtest
    from config.overrides import parse_overrides
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    coins = ["LONGCOIN", "SHORTCOIN"]
    cfg = _ema_anchor_config(True)
    cfg["live"]["approved_coins"] = {s: coins[:] for s in ("long", "short")}
    cfg["backtest"]["dynamic_wel_by_tradability"] = False
    for s in ("long", "short"):
        cfg["bot"][s]["risk"]["n_positions"] = 2
        cfg["bot"][s]["forager"]["unilateralness_ema_span_1m"] = span
        cfg["bot"][s]["entry_cooldown"]["base_duration_minutes"] = 0.0
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": coins}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    baseline, _, _ = run_backtest(hlcvs, copy.deepcopy(markets), cfg, "binance", btc, timestamps)
    cfg["coin_overrides"] = {target: {"bot": {side: {"entry_cooldown": {
        "max_duration_minutes": 60.0,
        "weights_minutes": {"adverse_directionality": 1.0},
    }}}}}
    cfg = parse_overrides(cfg, verbose=False)
    # Cached history represents the long RMS window, not universal trade readiness.
    for coin in coins:
        markets[coin]["warmup_minutes"] = math.ceil(20 * span) + 1
        markets[coin]["warmup_minutes_source"] = "history"
    fills, _, _, payload = run_backtest(
        hlcvs, markets, cfg, "binance", btc, timestamps, return_payload=True,
    )
    assert payload.backtest_params["global_warmup_bars"] == 10
    assert payload.backtest_params["trade_start_indices"] == [10, 10]

    def entry_times(rows, coin, pside):
        return [int(row[0]) for row in rows
                if str(row[2]) == coin and str(row[13]).startswith("entry_") and pside in str(row[13])]

    for coin in coins:
        for s in ("long", "short"):
            times = entry_times(fills, coin, s)
            if (coin, s) == (target, side):
                if span == 60.0:
                    assert not times
                else:
                    assert min(times) == 21  # First full window plans a next-candle fill.
            else:
                assert times and min(times) == min(entry_times(baseline, coin, s))
                assert min(times) < math.ceil(20 * span)


def test_adverse_warmup_marker_does_not_hide_other_missing_emas():
    import passivbot_rust as pbr

    inp = adaptive_input()
    symbol = inp["symbols"][0]
    symbol["unilateralness_warmup_spans"] = [60.0]
    symbol["emas"]["m1"]["signed_unilateralness"] = []
    symbol["emas"]["m1"]["close"] = []
    with pytest.raises(ValueError, match="MissingEma"):
        compute(pbr, inp)


@pytest.mark.parametrize("side", ["long", "short"])
def test_zero_non_rms_budget_does_not_restore_automatic_shared_delay(side):
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["live"]["warmup_ratio"] = 0.0
    # This legacy span would impose a shared delay beyond the entire test.
    cfg["bot"][side]["forager"]["volume_ema_span_1m"] = 5000.0
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 1.0
    cfg["bot"][side]["entry_cooldown"].update(
        base_duration_minutes=0.0, max_duration_minutes=60.0,
        weights_minutes={"exposure_ratio": 0.0, "adverse_directionality": 1.0},
    )
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    for coin in ("LONGCOIN", "SHORTCOIN"):
        markets[coin]["warmup_minutes"] = 21
        markets[coin]["warmup_minutes_source"] = "history"
    fills, _, _, payload = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps, return_payload=True)
    assert payload.backtest_params["global_warmup_bars"] == 1
    first_entries = {
        s: min(int(row[0]) for row in fills if str(row[13]).startswith("entry_") and s in str(row[13]))
        for s in ("long", "short")
    }
    assert first_entries[side] == 21
    assert first_entries["short" if side == "long" else "long"] < 20


def test_score_only_rms_keeps_legacy_zero_budget_when_ranking_is_unused():
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["live"]["warmup_ratio"] = 0.0
    cfg["bot"]["long"]["forager"]["volume_ema_span_1m"] = 5000.0
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    baseline = run_backtest(hlcvs, copy.deepcopy(markets), cfg, "binance", btc, timestamps)
    cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1.0
    actual = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    np.testing.assert_array_equal(actual[0], baseline[0])
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert actual[2] == baseline[2]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("interval", [1, 5])
@pytest.mark.parametrize("universe,slots,dynamic", [(1, 1, False), (1, 1, True), (2, 2, False)])
def test_cpu_unused_forager_rms_preserves_aggregated_candle_results(side, interval, universe, slots, dynamic):
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["backtest"].update(candle_interval_minutes=interval, dynamic_wel_by_tradability=dynamic)
    cfg["bot"][side]["risk"]["n_positions"] = slots
    if universe == 2:
        cfg["live"]["approved_coins"][side] = ["LONGCOIN", "SHORTCOIN"]
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    timestamps = timestamps[0] + np.arange(len(timestamps)) * interval * 60000
    baseline = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 100000.0
    actual = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert len(actual[0]) > 0
    np.testing.assert_array_equal(actual[0], baseline[0])
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert actual[2] == baseline[2]
    if interval > 1:
        # The same no-ranking universe still needs 1m data for adverse cooldown.
        cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
        cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 10.0
        with pytest.raises(ValueError, match="RMS unilateralness requires.*one-minute"):
            run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("missing_idx", [0, 1])
@pytest.mark.parametrize("marker", ["warmup", "live"])
def test_mixed_rms_readiness_defers_entire_ranking(side, missing_idx, marker):
    import passivbot_rust as pbr

    bp = {"n_positions": 1, "total_wallet_exposure_limit": 1.0, "forager_score_weights": {
        "volume": 0.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": 1.0,
    }}
    symbols = [make_symbol(i, bid=100.0, ask=100.0, **{f"{side}_bp": bp}) for i in range(2)]
    for i, symbol in enumerate(symbols):
        symbol[side]["mode"] = None
        symbol["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.8]] if i != missing_idx else []
    missing = symbols[missing_idx]
    if marker == "warmup":
        missing["unilateralness_warmup_spans"] = [60.0]
    else:
        missing["unilateralness_unavailable"] = {"forager": [60.0]}
    inp = make_input(balance=1000.0, global_bp=bot_params_pair(**{f"{side}_overrides": bp}), symbols=symbols)
    result = compute(pbr, inp)
    selection = next(s for s in result["diagnostics"]["forager_selections"] if s["pside"] == side)
    assert selection["ranking_required"]
    assert selection["selected_symbol_indices"] == []
    missing["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.0]]
    result = compute(pbr, inp)
    selection = next(s for s in result["diagnostics"]["forager_selections"] if s["pside"] == side)
    assert selection["selected_symbol_indices"] == [missing_idx]
    # Invalid ready data remains fatal even alongside an explicitly missing score.
    missing["emas"]["m1"]["signed_unilateralness"] = []
    symbols[1-missing_idx]["emas"]["m1"]["signed_unilateralness"] = [[60.0, 1.5]]
    with pytest.raises(ValueError):
        compute(pbr, inp)


@pytest.mark.parametrize("side", ["long", "short"])
def test_cpu_staggered_listing_waits_for_every_compared_rms_window(side):
    from backtest import build_backtest_payload, execute_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["live"]["approved_coins"][side] = ["LONGCOIN", "SHORTCOIN"]
    cfg["bot"][side]["forager"].update(unilateralness_ema_span_1m=1.0)
    cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    markets["SHORTCOIN"]["first_valid_index"] = 10
    payload = build_backtest_payload(hlcvs, markets, cfg, "binance", btc, timestamps)
    # Both coins join the compared universe together, after one has enough
    # history and while the later listing is still building its RMS window.
    payload.backtest_params["trade_start_indices"] = [20, 20]
    fills, _, _ = execute_backtest(payload, cfg)
    times = [int(row[0]) for row in fills if str(row[13]).startswith("entry_") and side in str(row[13])]
    assert times and min(times) == 31


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("base,floor,ceiling", [(0.0, 0.0, 0.0), (0.0, 5.0, 5.0), (10.0, 0.0, 5.0)])
def test_constant_cooldown_does_not_require_directional_inputs(side, base, floor, ceiling):
    import passivbot_rust as pbr
    inp = adaptive_input(base=base, exposure=10.0)
    symbol = inp["symbols"][0]
    if side == "short":
        symbol["short"]["bot_params"] = copy.deepcopy(symbol["long"]["bot_params"])
    bp = symbol[side]["bot_params"]
    bp["entry_cooldown_min_duration_minutes"] = floor
    bp["entry_cooldown_max_duration_minutes"] = ceiling
    symbol["emas"]["m1"]["signed_unilateralness"] = []
    # Inspect just this policy; the opposite side has no active modifier.
    symbol["short" if side == "long" else "long"]["bot_params"]["entry_cooldown_weights_minutes"] = {
        "exposure_ratio": 0.0, "adverse_directionality": 0.0,
    }
    durations = json.loads(pbr.entry_cooldown_durations_json(json.dumps(inp)))
    assert durations[f"0:{side}"] == ceiling
    compute(pbr, inp)


@pytest.mark.parametrize("marker", [{"current": [60.0]}, {"current": [30.0]}, {"forager": [60.0]}])
def test_live_rms_unavailability_is_consumer_and_span_specific(marker):
    import passivbot_rust as pbr
    inp = adaptive_input()
    symbol = inp["symbols"][0]
    symbol["emas"]["m1"]["signed_unilateralness"] = []
    symbol["unilateralness_unavailable"] = marker
    if marker != {"current": [60.0]}:
        with pytest.raises(ValueError, match="MissingEma"):
            compute(pbr, inp)
        return
    symbol["long"]["position"] = {"size": 1.0, "price": 100.0}
    result = compute(pbr, inp)
    assert not entries(result)
    assert any(o["order_type"].startswith("close_") for o in result["orders"])
    symbol["emas"]["m1"]["close"] = []
    with pytest.raises(ValueError, match="MissingEma"):
        compute(pbr, inp)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("interval", [1, 5])
@pytest.mark.parametrize("base,floor,ceiling", [(0, 0, 0), (0, 5, 5), (10, 0, 5)])
def test_constant_clamp_cpu_needs_no_rms_history(side, interval, base, floor, ceiling):
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = _ema_anchor_config(True)
    cfg["backtest"]["candle_interval_minutes"] = interval
    cfg["bot"][side]["entry_cooldown"].update(
        base_duration_minutes=base, min_duration_minutes=floor, max_duration_minutes=ceiling,
    )
    cfg["bot"][side]["forager"]["unilateralness_ema_span_1m"] = 100000.0
    cfg["optimize"]["bounds"][side]["entry_cooldown"]["base_duration_minutes"] = [base, base]
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    baseline = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    warmup = compute_backtest_warmup_minutes(cfg)
    cfg["bot"][side]["entry_cooldown"]["weights_minutes"] = {
        "adverse_directionality": 20.0, "exposure_ratio": 10.0,
    }
    assert compute_backtest_warmup_minutes(cfg) == warmup
    actual = run_backtest(hlcvs, markets, cfg, "binance", btc, timestamps)
    np.testing.assert_array_equal(actual[0], baseline[0])
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert actual[2] == baseline[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("base,floor,ceiling", [(0, 0, 0), (0, 5, 5), (10, 0, 5)])
async def test_live_constant_clamp_skips_rms_fetch(base, floor, ceiling):
    from live.unilateralness import load
    async def candles(*args, **kwargs):
        pytest.fail("constant cooldown must not request RMS candles")
    params = {
        "entry_cooldown_weights_minutes": {"adverse_directionality": 20.0},
        "risk_entry_cooldown_minutes": base,
        "entry_cooldown_min_duration_minutes": floor,
        "entry_cooldown_max_duration_minutes": ceiling,
    }
    bot = SimpleNamespace(
        cm=SimpleNamespace(get_candles=candles), get_exchange_time=lambda: 60000,
        is_approved=lambda side, symbol: True,
        is_pside_enabled=lambda side: True, bp=lambda side,key,symbol: params[key],
        bot_value=lambda side,key: {"unilateralness": 0.0} if key=="forager_score_weights" else 60.0,
    )
    assert await load(bot, ["BTC"], set()) == ({"BTC": {}}, {"BTC": {}}, {})


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("metadata", [[], [0, 0]])
def test_direct_rust_zero_warmup_keeps_automatic_non_rms_budget(side, metadata):
    from backtest import build_backtest_payload, execute_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    for pside in ("long", "short"):
        cfg["bot"][pside]["forager"].update(volume_ema_span_1m=30.0, unilateralness_ema_span_1m=1.0)
        cfg["bot"][pside]["strategy"]["ema_anchor"]["offset_volatility_ema_span_1h"] = 0.1
        cfg["bot"][pside]["entry_cooldown"].update(base_duration_minutes=0.0, max_duration_minutes=60.0)
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    payload = build_backtest_payload(hlcvs, markets, cfg, "binance", btc, timestamps)
    # Direct Rust callers use zero to request automatic EMA warmup, even when
    # Python normally provides an explicitly calculated activation budget.
    payload.backtest_params["global_warmup_bars"] = 0
    payload.backtest_params["warmup_minutes"] = metadata
    payload.backtest_params["trade_start_indices"] = metadata
    baseline = execute_backtest(payload, cfg)
    for pair in payload.bot_params_list:
        pair[side]["entry_cooldown_weights_minutes"]["adverse_directionality"] = 10.0
    actual = execute_backtest(payload, cfg)
    assert len(actual[0]) > 0
    assert min(int(row[0]) for row in actual[0]) == 32
    np.testing.assert_array_equal(actual[0], baseline[0])
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert actual[2] == baseline[2]


@pytest.mark.parametrize("source,expected", [("activation", 1000), (None, 1000), ("history", 10)])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
def test_rms_history_never_erases_stamped_or_untyped_activation(source, expected, consumer):
    from backtest import build_backtest_payload
    from optimization.warmup import stamp_warmup_metadata
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    cfg["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 60.0
    if consumer == "forager":
        cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1.0
    else:
        cfg["bot"]["long"]["entry_cooldown"].update(max_duration_minutes=60.0)
        cfg["bot"]["long"]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 10.0
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    h, markets, b, timestamps = _synthetic_inputs()
    hlcvs = np.tile(h, (25, 1, 1))
    btc = np.full(len(hlcvs), b[0])
    timestamps = timestamps[0] + np.arange(len(hlcvs)) * 60000
    for coin in ("LONGCOIN", "SHORTCOIN"):
        markets[coin]["last_valid_index"] = len(hlcvs)-1
    stamp_warmup_metadata(markets, ["LONGCOIN", "SHORTCOIN"], {"__default__": 1000})
    for coin in ("LONGCOIN", "SHORTCOIN"):
        if source is None:
            del markets[coin]["warmup_minutes_source"]
        elif source == "history":
            markets[coin].update(warmup_minutes_source="history", warmup_minutes=1201)
    payload = build_backtest_payload(hlcvs, markets, cfg, "binance", btc, timestamps)
    assert payload.backtest_params["warmup_minutes"] == [expected, expected]
    assert payload.backtest_params["trade_start_indices"] == [expected, expected]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("marker", ["live", "warmup", "unmarked"])
@pytest.mark.parametrize("missing_idx", [1, 2])
def test_volume_pruning_limits_required_rms_scoring_set(side, marker, missing_idx):
    import passivbot_rust as pbr

    bp = {"n_positions": 1, "total_wallet_exposure_limit": 1.0,
          "filter_volume_drop_pct": 1.0 / 3.0,
          "filter_volume_ema_span_1m": 10.0,
          "unilateralness_ema_span_1m": 60.0,
          "forager_score_weights": {
              "volume": 0.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": 1.0}}
    symbols = [make_symbol(i, bid=100.0, ask=100.0, **{f"{side}_bp": bp}) for i in range(3)]
    for i, symbol in enumerate(symbols):
        symbol[side]["mode"] = None
        symbol["emas"]["m1"]["volume"] = [[10.0, 1.0 if i == 2 else 100.0]]
        symbol["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.9 if i == 0 else 0.1]]
    symbols[missing_idx]["emas"]["m1"]["signed_unilateralness"] = []
    if marker == "live":
        symbols[missing_idx]["unilateralness_unavailable"] = {"forager": [60.0]}
    elif marker == "warmup":
        symbols[missing_idx]["unilateralness_warmup_spans"] = [60.0]
    inp = make_input(balance=1000, global_bp=bot_params_pair(**{f"{side}_overrides": bp}), symbols=symbols)
    if missing_idx == 1 and marker == "unmarked":
        with pytest.raises(ValueError, match="MissingEma"):
            compute(pbr, inp)
        return
    out = compute(pbr, inp)
    selection = next(s for s in out["diagnostics"]["forager_selections"] if s["pside"] == side)
    assert selection["selected_symbol_indices"] == ([1] if missing_idx == 2 else [])
    # Supplying the dropped score cannot change the winner; restoring a retained
    # score restores the complete ranking, and malformed retained scores fail.
    symbols[missing_idx]["emas"]["m1"]["signed_unilateralness"] = [[60.0, 0.0]]
    ready = compute(pbr, inp)
    assert next(s for s in ready["diagnostics"]["forager_selections"] if s["pside"] == side)["selected_symbol_indices"] == [1]
    symbols[0]["emas"]["m1"]["signed_unilateralness"] = [[60.0, 1.5]]
    with pytest.raises(ValueError, match="signed_unilateralness"):
        compute(pbr, inp)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("score_weight", [0.0, 1.0])
def test_late_validity_end_rejects_held_valuation_before_optional_rms_ranking(side, score_weight):
    from backtest import build_backtest_payload, execute_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    cfg = _ema_anchor_config(True)
    coins = ["LONGCOIN", "SHORTCOIN", "THIRDCOIN"]
    cfg["live"]["approved_coins"] = {"long": [], "short": []}
    cfg["live"]["approved_coins"][side] = coins
    cfg["bot"][side]["risk"]["n_positions"] = 3
    cfg["bot"][side]["forager"].update(unilateralness_ema_span_1m=1.0,
        score_weights={"volume": 0.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": score_weight})
    cfg["backtest"].update(dynamic_wel_by_tradability=False, candle_interval_minutes=1)
    cfg = prepare_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": coins}
    hlcvs, markets, btc, timestamps = _synthetic_inputs()
    hlcvs = np.concatenate([hlcvs, hlcvs[:, 1:2].copy()], axis=1)
    markets["THIRDCOIN"] = copy.deepcopy(markets["SHORTCOIN"])
    # The first coin fills early, then moves adversely and stays held through its
    # late validity end. Two candidates would compete if planning could proceed.
    held_price = 80.0 if side == "long" else 120.0
    hlcvs[12:, 0, :3] = [held_price + 0.01, held_price - 0.01, held_price]
    payload = build_backtest_payload(hlcvs, markets, cfg, "binance", btc, timestamps)
    payload.backtest_params["trade_start_indices"] = [10, 31, 31]
    # Keep actual candles beyond the tradable range for held-position valuation.
    payload.backtest_params["last_valid_indices"] = [30, 59, 59]
    assert all(pair[side]["entry_eligible"] for pair in payload.bot_params_list)
    assert all(pair[side]["n_positions"] == 3 for pair in payload.bot_params_list)
    assert all(pair[side]["forager_score_weights"]["unilateralness"] == score_weight
               for pair in payload.bot_params_list)
    # An out-of-range held position cannot be valued, even when finite prices
    # exist outside its declared valid range. This guard precedes ranking and
    # must remain fatal with both disabled and enabled RMS scoring.
    with pytest.raises(ValueError, match="missing held-position valuation candle: coin LONGCOIN index 0 candle 31"):
        execute_backtest(payload, cfg)


@pytest.mark.parametrize("side", ["long", "short"])
def test_unapproved_override_cooldown_does_not_extend_fill_coverage(side):
    from passivbot import Passivbot
    from config.runtime_compile import compile_runtime_config

    bot = Passivbot.__new__(Passivbot)
    cfg = prepare_config(get_template_config(), verbose=False)
    for pside in ("long", "short"):
        cfg["bot"][pside]["risk"].update(n_positions=1, total_wallet_exposure_limit=float(pside == side))
        cfg["bot"][pside]["entry_cooldown"]["base_duration_minutes"] = 5.0
    bot.config = compile_runtime_config(cfg, runtime="live")
    bot.coin_overrides = {"BTC": {"bot": {side: {"entry_cooldown": {
        "max_duration_minutes": 14400.0,
        "weights_minutes": {"exposure_ratio": 10.0},
    }}}}}
    approved = set()
    bot.is_approved = lambda pside, symbol: (pside, symbol) in approved
    bot._equity_hard_stop_enabled = lambda: False
    bot._live_risk_uses_authoritative_pnl = lambda: False
    now = 30 * 24 * 60 * 60000
    assert bot._max_configured_entry_cooldown_minutes() == 5.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (True, now - 6 * 60000)
    approved.add((side, "BTC"))
    assert bot._max_configured_entry_cooldown_minutes() == 14400.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (True, now - 14401 * 60000)
    approved.clear()
    assert bot._max_configured_entry_cooldown_minutes() == 5.0
    # A removed held coin still permits DCA under auto graceful-stop. Include its
    # override even on a cold restart before the per-cycle mode map is available.
    bot.config["_coins_sources"] = {"approved_coins": {side: ["ETH"]}}
    bot.approved_coins_minus_ignored_coins = {side: {"ETH"}}
    approved.add((side, "ETH"))
    bot.positions = {"BTC": {side: {"size": 1.0 if side == "long" else -1.0}}}
    assert bot._max_configured_entry_cooldown_minutes() == 14400.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (True, now - 14401 * 60000)
    # Closing the unapproved position releases the extra history requirement.
    bot.positions["BTC"][side]["size"] = 0.0
    assert bot._max_configured_entry_cooldown_minutes() == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("age_minutes", [0, 1, 2])
async def test_rms_load_rollover_rechecks_all_symbols_and_recovers(side, age_minutes):
    from live.unilateralness import load
    from candlestick_manager import CANDLE_DTYPE

    now = 30 * 60000 + 59999
    roll = True
    calls = []

    async def candles(symbol, **kwargs):
        nonlocal now
        calls.append(kwargs["end_ts"])
        rows = np.zeros(21, dtype=CANDLE_DTYPE)
        rows["ts"] = kwargs["end_ts"] - np.arange(20, -1, -1) * 60000
        rows["c"] = np.exp(np.arange(21) * 0.001)
        if roll:
            now += 60000
        return rows

    bot = SimpleNamespace(
        cm=SimpleNamespace(get_candles=candles),
        get_exchange_time=lambda: now,
        is_approved=lambda side, symbol: True,
        is_pside_enabled=lambda pside: pside == side,
        is_forager_mode=lambda pside: True,
        bot_value=lambda pside, key: {"unilateralness": 1.0} if key == "forager_score_weights" else 1.0,
        bp=lambda pside, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
    )
    symbols = ["BTC", "ETH"]
    ages = dict.fromkeys(symbols, age_minutes * 60000)
    current, ranking, missing = await load(bot, symbols, set(), ages)
    assert len(set(calls)) == 1
    assert current == {symbol: {} for symbol in symbols}
    for symbol in symbols:
        assert bool(ranking[symbol]) == (age_minutes == 2)
        assert missing[symbol] == {"current": [1.0], "forager": [] if age_minutes == 2 else [1.0]}
    roll = False
    current, ranking, missing = await load(bot, symbols, set(), ages)
    assert not missing
    assert current == ranking
    assert all(current[symbol][1.0] > 0.99 for symbol in symbols)


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["long", "short"])
async def test_live_rms_replay_cache_reuses_repairs_and_recovers(tmp_path, monkeypatch, side):
    from live.unilateralness import load
    from candlestick_manager import CandlestickManager, CANDLE_DTYPE
    import passivbot_rust as pbr

    cm = CandlestickManager(exchange=None, exchange_name="fake", cache_dir=str(tmp_path))
    now = 100 * 60000 + 30000
    span = 2.5
    enabled = True
    rows = np.zeros(51, dtype=CANDLE_DTYPE)
    rows["ts"] = (np.arange(51) + 49) * 60000
    rows["c"] = np.exp(np.arange(51) * 0.001)
    for field in ("o", "h", "l"):
        rows[field] = rows["c"]
    rows["bv"] = 1.0
    cm._persist_batch("BTC", rows, skip_memory_retention=True)
    fetches, replays = [], []
    missing = False

    async def candles(symbol, **kwargs):
        fetches.append(symbol)
        return np.empty(0, dtype=CANDLE_DTYPE) if missing else cm._cache[symbol].copy()

    original = pbr.calc_signed_unilateralness

    def replay(closes, requested_span):
        replays.append(requested_span)
        return original(closes, requested_span)

    monkeypatch.setattr(cm, "get_candles", candles)
    monkeypatch.setattr(pbr, "calc_signed_unilateralness", replay)
    bot = SimpleNamespace(
        cm=cm, get_exchange_time=lambda: now,
        is_approved=lambda side, symbol: True,
        is_pside_enabled=lambda pside: pside == side and enabled,
        is_forager_mode=lambda pside: True,
        bot_value=lambda pside, key: {"unilateralness": 1.0} if key == "forager_score_weights" else span,
        bp=lambda pside, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
    )
    first = await load(bot, ["BTC"], set())
    for _ in range(10):
        assert await load(bot, ["BTC"], {"BTC"}) == first
    assert len(fetches) == len(replays) == 1
    # Real canonical ingestion invalidates the cached value at the same cutoff.
    repaired = rows[-1:].copy()
    for field in ("o", "h", "l", "c"):
        repaired[field] *= 0.9
    cm._persist_batch("BTC", repaired, skip_memory_retention=True)
    changed = await load(bot, ["BTC"], set())
    assert changed != first
    assert len(fetches) == len(replays) == 2
    # Losing cache is restart-equivalent, including the repaired candle.
    cm._ema_cache.clear()
    assert await load(bot, ["BTC"], set()) == changed
    assert len(replays) == 3
    # Gap invalidation and failed reads never acquire a reusable neutral value.
    cm._invalidate_ema_cache("BTC", timeframe="1m")
    missing = True
    for _ in range(2):
        current, ranking, unavailable = await load(bot, ["BTC"], set())
        assert current == ranking == {"BTC": {}}
        assert unavailable == {"BTC": {"current": [span], "forager": [span]}}
    assert len(fetches) == 5 and len(replays) == 3
    missing = False
    assert await load(bot, ["BTC"], set()) == changed
    # A new cutoff is never served by the previous minute's cache.
    now += 60000
    current, ranking, unavailable = await load(bot, ["BTC"], set())
    assert current == ranking == {"BTC": {}}
    assert unavailable
    appended = repaired.copy()
    appended["ts"] += 60000
    cm._persist_batch("BTC", appended, skip_memory_retention=True)
    assert not (await load(bot, ["BTC"], set()))[2]
    # Config/universe changes bound the namespace; unrelated EMA cache survives.
    cm._ema_cache["BTC"][("close", 3.0, "60000")] = (1.0, 0, 0)
    span = 1.0
    assert not (await load(bot, ["BTC"], set()))[2]
    assert [key[1] for key in cm._ema_cache["BTC"] if key[0] == "signed_unilateralness"] == [1.0]
    enabled = False
    assert await load(bot, ["BTC"], set()) == ({"BTC": {}}, {"BTC": {}}, {})
    assert list(cm._ema_cache["BTC"]) == [("close", 3.0, "60000")]
    enabled = True
    await load(bot, ["BTC"], set())
    await load(bot, [], set())
    assert list(cm._ema_cache["BTC"]) == [("close", 3.0, "60000")]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("modifier", ["exposure_ratio", "adverse_directionality"])
def test_coin_zero_modifier_shadows_positive_fixed_runtime_pin(side, modifier):
    from config.overrides import parse_overrides

    cfg = get_template_config()
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
    path = f"bot.{side}.entry_cooldown.weights_minutes.{modifier}"
    cfg["optimize"]["fixed_runtime_overrides"] = {path: 10.0}
    cfg["optimize"]["bounds"][f"{side}_entry_cooldown_weights_minutes_{modifier}"] = [0.0, 10.0]
    cfg = prepare_config(cfg, verbose=False)
    cfg["coin_overrides"] = {"BTC": {"bot": {side: {"entry_cooldown": {
        "max_duration_minutes": None, "weights_minutes": {modifier: 0.0},
    }}}}}
    parsed = parse_overrides(cfg, verbose=False)
    assert parsed["optimize"]["fixed_runtime_overrides"][path] == 10.0
    assert parsed["coin_overrides"]["BTC"]["bot"][side]["entry_cooldown"]["max_duration_minutes"] is None
    cfg["coin_overrides"]["BTC"]["bot"][side]["entry_cooldown"]["weights_minutes"][modifier] = 1.0
    with pytest.raises(ValueError, match="ceiling|finite|max_duration"):
        parse_overrides(cfg, verbose=False)


@pytest.mark.asyncio
async def test_stale_rms_ranking_cache_keeps_actual_age_and_refresh_policy():
    from live.unilateralness import load
    from candlestick_manager import CANDLE_DTYPE, CandlestickManager

    rows = np.zeros(21, dtype=CANDLE_DTYPE)
    rows["ts"] = np.arange(21) * 60000
    rows["c"] = np.exp(np.arange(21) * .001)
    calls = 0
    now = 22 * 60000

    async def candles(*args, **kwargs):
        nonlocal calls
        calls += 1
        return rows.copy()

    cm = SimpleNamespace(_ema_cache={}, get_candles=candles)
    bot = SimpleNamespace(
        cm=cm, get_exchange_time=lambda: now,
        is_approved=lambda side, symbol: True,
        is_pside_enabled=lambda side: side == "long", is_forager_mode=lambda side: True,
        bot_value=lambda side, key: {"unilateralness": 1.0} if key == "forager_score_weights" else 1.0,
        bp=lambda side, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
    )
    first = await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000})
    assert first[0] == {"BTC": {}} and first[1]["BTC"][1.0] > .99
    assert first[2] == {"BTC": {"current": [1.0], "forager": []}}
    for _ in range(10):
        assert await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000}) == first
    assert calls == 1
    # Remote-enabled consumers still get a chance to refresh their stale history.
    assert await load(bot, ["BTC"], set(), {"BTC": 120000}) == first
    assert calls == 2
    CandlestickManager._invalidate_ema_cache(cm, "BTC", timeframe="1m")
    rows["c"][-1] *= .9
    changed = await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000})
    assert changed != first and calls == 3
    now += 60000
    assert await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000}) == changed
    assert calls == 3
    # Both elapsed time and a tightened age budget reject the cached observation.
    assert (await load(bot, ["BTC"], {"BTC"}, {"BTC": 60000}))[1] == {"BTC": {}}
    now += 60000
    assert (await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000}))[1] == {"BTC": {}}
    rows["ts"] += 3 * 60000
    recovered = await load(bot, ["BTC"], {"BTC"}, {"BTC": 120000})
    assert recovered[0] == recovered[1] and not recovered[2]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("consumer", ["forager", "adverse"])
def test_direct_rust_orders_enforce_supported_rms_span(side, consumer):
    import passivbot_rust as pbr

    bp = {"unilateralness_ema_span_1m": 100000.0, "n_positions": 1,
          "entry_cooldown_max_duration_minutes": 60.0,
          "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": float(consumer == "adverse")}}
    if consumer == "forager":
        bp["forager_score_weights"] = {"volume": 0.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": 1.0}
    inp = make_input(balance=1000.0, global_bp=bot_params_pair(**{f"{side}_overrides": bp}),
                     symbols=[make_symbol(i, bid=100.0, ask=100.0, **{f"{side}_bp": bp}) for i in range(2)])
    for symbol in inp["symbols"]:
        symbol["emas"]["m1"]["signed_unilateralness"] = [[100000.0, 0.5]]
    compute(pbr, inp)
    inp["global"]["global_bot_params"][side]["unilateralness_ema_span_1m"] = 100000.01
    for symbol in inp["symbols"]:
        symbol[side]["bot_params"]["unilateralness_ema_span_1m"] = 100000.01
        symbol["emas"]["m1"]["signed_unilateralness"] = [[100000.01, 0.5]]
    with pytest.raises(ValueError, match="unilateralness.*100000"):
        compute(pbr, inp)


@pytest.mark.parametrize("side", ["long", "short"])
def test_explicit_live_universe_uses_effective_coin_cooldowns(side):
    from passivbot import Passivbot
    from config.runtime_compile import compile_runtime_config

    bot = Passivbot.__new__(Passivbot)
    cfg = prepare_config(get_template_config(), verbose=False)
    for pside in ("long", "short"):
        cfg["bot"][pside]["risk"].update(n_positions=1, total_wallet_exposure_limit=float(pside == side))
    cfg["live"]["approved_coins"] = {pside: ["BTC", "ETH"] for pside in ("long", "short")}
    cfg.pop("_coins_sources", None)
    cfg["bot"][side]["entry_cooldown"].update(
        base_duration_minutes=0.0, max_duration_minutes=14400.0,
        weights_minutes={"exposure_ratio": 10.0, "adverse_directionality": 0.0},
    )
    bot.config = compile_runtime_config(cfg, runtime="live")
    bot.approved_coins_minus_ignored_coins = {side: {"BTC", "ETH"}}
    bot.is_approved = lambda pside, symbol: symbol in bot.approved_coins_minus_ignored_coins.get(pside, set())
    bot._equity_hard_stop_enabled = lambda: False
    bot._live_risk_uses_authoritative_pnl = lambda: False
    bot.coin_overrides = {coin: {"bot": {side: {"entry_cooldown": {
        "max_duration_minutes": None, "weights_minutes": {"exposure_ratio": 0.0},
    }}}} for coin in ("BTC", "ETH")}
    now = 30 * 24 * 60 * 60000
    assert bot._max_configured_entry_cooldown_minutes() == 0.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (False, None)
    # An inheriting approved symbol restores the global adaptive horizon.
    bot.approved_coins_minus_ignored_coins[side].add("XRP")
    assert bot._max_configured_entry_cooldown_minutes() == 14400.0
    bot.approved_coins_minus_ignored_coins[side].remove("XRP")
    # A partial override still inherits its unspecified modifier.
    bot.coin_overrides["BTC"]["bot"][side]["entry_cooldown"] = {"max_duration_minutes": 60.0}
    assert bot._max_configured_entry_cooldown_minutes() == 60.0
    assert bot._required_fill_history_start_ms(now, pnl_start_ms=None) == (True, now - 61 * 60000)
    # All and unresolved sources remain conservative even with a resolved current set.
    bot.config["_coins_sources"] = {"approved_coins": "all"}
    assert bot._max_configured_entry_cooldown_minutes() == 14400.0
    bot.config["_coins_sources"] = {"approved_coins": {side: ["all"]}}
    assert bot._max_configured_entry_cooldown_minutes() == 14400.0
    bot.config.pop("_coins_sources")
    bot.approved_coins_minus_ignored_coins = {}
    assert bot._max_configured_entry_cooldown_minutes() == 14400.0


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["long", "short"])
async def test_live_adverse_rms_respects_side_approval_and_transitions(side):
    from live.unilateralness import load, adverse_enabled
    from candlestick_manager import CANDLE_DTYPE

    other = "short" if side == "long" else "long"
    spans = {side: 1.0, other: 100000.0}
    approved = {side: {"BTC"}, other: {"ETH"}}
    scoring = {"long": 0.0, "short": 0.0}
    requests = []
    rows = np.zeros(21, dtype=CANDLE_DTYPE)
    rows["ts"] = np.arange(21) * 60000
    rows["c"] = np.exp(np.arange(21) * .001)

    async def candles(symbol, **kwargs):
        requests.append((symbol, kwargs["start_ts"]))
        return rows.copy()

    bot = SimpleNamespace(
        cm=SimpleNamespace(get_candles=candles, _ema_cache={}),
        get_exchange_time=lambda: 21 * 60000,
        is_pside_enabled=lambda s: True, is_forager_mode=lambda s: True,
        is_approved=lambda s, symbol: symbol in approved[s],
        bot_value=lambda s, key: {"unilateralness": scoring[s]} if key == "forager_score_weights" else spans[s],
        bp=lambda s, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
    )
    assert adverse_enabled(bot, side, "BTC")
    assert not adverse_enabled(bot, other, "BTC")
    current, _, missing = await load(bot, ["BTC"], set())
    assert requests == [("BTC", 0)] and not missing
    assert current["BTC"][1.0] > .99
    approved[other].add("BTC")
    _, _, missing = await load(bot, ["BTC"], set())
    assert requests[-1][1] == (20 - 2000000) * 60000
    assert missing["BTC"]["current"] == [100000.0]
    approved[other].remove("BTC")
    # Independent Forager demand must still retain the otherwise ineligible span.
    scoring[other] = 1.0
    _, _, missing = await load(bot, ["BTC"], set())
    assert missing["BTC"] == {"current": [], "forager": [100000.0]}


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("available", [False, True])
async def test_unapproved_held_graceful_stop_keeps_adverse_inputs_and_closes(side, available):
    import passivbot_rust as pbr
    from live.unilateralness import load
    from candlestick_manager import CANDLE_DTYPE

    rows = np.zeros(21 if available else 0, dtype=CANDLE_DTYPE)
    rows["ts"] = np.arange(len(rows)) * 60000
    rows["c"] = 100.0
    async def candles(*args, **kwargs):
        return rows.copy()
    bp = {"n_positions": 1, "total_wallet_exposure_limit": 1.0,
          "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 10.0},
          "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
          "entry_cooldown_max_duration_minutes": 30.0, "unilateralness_ema_span_1m": 1.0}
    position = {"size": 1.0 if side == "long" else -1.0, "price": 100.0}
    bot = SimpleNamespace(
        cm=SimpleNamespace(get_candles=candles), get_exchange_time=lambda: 21 * 60000,
        positions={"BTC": {side: position}}, PB_modes={side: {"BTC": "graceful_stop"}},
        is_approved=lambda s, symbol: False, is_pside_enabled=lambda s: s == side,
        is_forager_mode=lambda s: False,
        bot_value=lambda s, key: {"unilateralness": 0.0} if key == "forager_score_weights" else bp[key],
        bp=lambda s, key, symbol: bp[key],
    )
    current, _, missing = await load(bot, ["BTC"], set())
    assert current["BTC"] == ({1.0: 0.0} if available else {})
    assert missing == ({} if available else {"BTC": {"current": [1.0], "forager": []}})
    symbol = make_symbol(0, bid=100.0, ask=100.0, **{f"{side}_bp": bp})
    symbol[side].update(mode="graceful_stop", position=position)
    symbol["emas"]["m1"]["signed_unilateralness"] = sorted(current["BTC"].items())
    symbol["unilateralness_unavailable"] = missing.get("BTC", {})
    inp = make_input(balance=1000, global_bp=bot_params_pair(**{f"{side}_overrides": bp}), symbols=[symbol])
    result = compute(pbr, inp)
    assert any(o["pside"] == side and o["order_type"].startswith("close_") for o in result["orders"])
    # A missing score must defer only this entry consumer, never fail the batch.
    if not available:
        assert not any(o["pside"] == side and o["order_type"].startswith("entry_") for o in result["orders"])


@pytest.mark.asyncio
@pytest.mark.parametrize("long_span", [1.0, 2.5])
@pytest.mark.parametrize("age_minutes", [29, 30, 31])
@pytest.mark.parametrize("rollover", [False, True])
async def test_live_rms_uses_latest_complete_pre_gap_window(long_span, age_minutes, rollover):
    import passivbot_rust as pbr
    from live.unilateralness import load
    from candlestick_manager import CANDLE_DTYPE

    # The larger span is complete only before the gap; the smaller one is current.
    minutes = np.r_[np.arange(51), np.arange(52, 81)]
    rows = np.zeros(len(minutes), dtype=CANDLE_DTYPE)
    rows["ts"] = minutes * 60000
    rows["c"] = np.exp(minutes * .001)
    now = 81 * 60000
    calls = 0
    async def candles(*args, **kwargs):
        nonlocal now, calls
        calls += 1
        if rollover:
            now += 60000
        return rows.copy()
    spans = {"long": long_span, "short": 3.5 - long_span}
    bot = SimpleNamespace(
        cm=SimpleNamespace(get_candles=candles, _ema_cache={}),
        get_exchange_time=lambda: now,
        is_pside_enabled=lambda side: True, is_forager_mode=lambda side: True,
        is_approved=lambda side, symbol: True,
        bot_value=lambda side, key: {"unilateralness": 1.0} if key == "forager_score_weights" else spans[side],
        bp=lambda side, key, symbol: {
            "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0, "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }[key],
    )
    result, rank, missing = await load(bot, ["BTC"], {"BTC"}, {"BTC": age_minutes * 60000})
    assert set(result["BTC"]) == (set() if rollover else {1.0})
    expected = {1.0, 2.5} if age_minutes >= 30 + int(rollover) else {1.0}
    assert set(rank["BTC"]) == expected
    assert missing["BTC"] == {"current": [1.0, 2.5] if rollover else [2.5],
                              "forager": [] if 2.5 in expected else [2.5]}
    if 2.5 in expected:
        assert rank["BTC"][2.5] == pbr.calc_signed_unilateralness(rows["c"][:51].tolist(), 2.5)
        assert bot.cm._ema_cache["BTC"][("signed_unilateralness", 2.5, "60000")][1] == 50 * 60000
    assert bot.cm._ema_cache["BTC"][("signed_unilateralness", 1.0, "60000")][1] == 80 * 60000
    if not rollover and age_minutes >= 30:
        assert await load(bot, ["BTC"], {"BTC"}, {"BTC": age_minutes * 60000}) == (result, rank, missing)
        assert calls == 1
        # The cache path also rechecks each span's own source age on rollover.
        clock = iter([81 * 60000, 82 * 60000])
        bot.get_exchange_time = lambda: next(clock)
        cached_rollover = await load(bot, ["BTC"], {"BTC"}, {"BTC": age_minutes * 60000})
        assert cached_rollover[0] == {"BTC": {}}
        assert set(cached_rollover[1]["BTC"]) == ({1.0, 2.5} if age_minutes >= 31 else {1.0})
        assert calls == 1
        bot.get_exchange_time = lambda: now
        # Losing the memo reconstructs exactly the same bounded-age observation.
        bot.cm._ema_cache.clear()
        assert await load(bot, ["BTC"], {"BTC"}, {"BTC": age_minutes * 60000}) == (result, rank, missing)
        assert calls == 2
