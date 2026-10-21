"""Side-owned age admission uses factual time in the actual trading engine."""

from copy import deepcopy
import json

import numpy as np
import pytest

from config import prepare_config
from config.age_policy import inception_timestamp_ms
from config.field_registry import describe_field
from config.schema import get_template_config

DAY = 86_400_000
BASE = 1_704_067_200_000


def ages(config, long, short):
    for side, value in (("long", long), ("short", short)):
        config["bot"][side]["universe"]["minimum_coin_age_days"] = value
    return config


@pytest.fixture(autouse=True)
def real_native():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension
    assert not getattr(pbr, "__is_stub__", False)
    verify_loaded_runtime_extension()


@pytest.mark.parametrize("days", [0.0, 0.5, 7.25, 60.0])
def test_side_defaults_types_fixed_authority_and_float_values(days):
    source = ages(get_template_config(), days, 60.0)
    output = prepare_config(source, verbose=False)
    assert output["bot"]["long"]["universe"]["minimum_coin_age_days"] == days
    assert "minimum_coin_age_days" not in output["bot"]["global"]["universe"]
    for side in ("long", "short"):
        description = describe_field(f"bot.{side}.universe.minimum_coin_age_days")
        assert description["types"] == ["float"]
        assert description["unit"] == "days"
        assert not description["optimizer"]["eligible"]
        assert description["coin_override"] == "forbidden"


@pytest.mark.parametrize("value", [True, False, -0.1, float("nan"), float("inf"), "7", None])
@pytest.mark.parametrize("layer", ["root", "fixed", "scenario"])
def test_invalid_age_fails_at_authored_boundary(value, layer):
    source = get_template_config()
    path = "bot.short.universe.minimum_coin_age_days"
    if layer == "root":
        source["bot"]["short"]["universe"]["minimum_coin_age_days"] = value
    elif layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: value}
    else:
        source["backtest"]["scenarios"] = [{"label": "age", "overrides": {path: value}}]
    with pytest.raises((ValueError, TypeError), match="minimum_coin_age_days"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("layer", ["root", "fixed", "scenario", "coin"])
def test_global_age_is_not_an_unpublished_v9_alias(layer):
    source = get_template_config()
    path = "bot.global.universe.minimum_coin_age_days"
    if layer == "root":
        source["bot"]["global"]["universe"]["minimum_coin_age_days"] = 7.0
    elif layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: 7.0}
    elif layer == "scenario":
        source["backtest"]["scenarios"] = [{"label": "old", "overrides": {path: 7.0}}]
    else:
        source["coin_overrides"] = {"AAA": {"bot": {"long": {"universe": {"minimum_coin_age_days": 7.0}}}}}
    with pytest.raises((ValueError, KeyError), match="minimum_coin_age_days|universe"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("now,inception,days,expected", [
    (BASE, None, 0.0, True), (BASE, 0, 0.0, True),
    (BASE, None, 0.5, False), (BASE, 0, 0.5, False),
    (BASE, BASE + 1, 0.5, False), (BASE, BASE + 1, 0.0, True),
    (BASE + DAY // 2 - 1, BASE, 0.5, False),
    (BASE + DAY // 2, BASE, 0.5, True),
    (BASE + DAY // 2 + 1, BASE, 0.5, True),
])
def test_native_age_predicate_has_exact_factual_boundary(now, inception, days, expected):
    import passivbot_rust as pbr
    assert pbr.coin_age_eligible(now, inception, days) is expected


def test_native_replay_index_uses_same_predicate_on_nonuniform_completed_times():
    import passivbot_rust as pbr
    times = [BASE, BASE + 30_000, BASE + 60_000, BASE + 180_000]
    for days, inception in [(1 / 1440, BASE), (0.0, None), (1.0, None), (0.5, BASE + DAY)]:
        expected = next((i for i, t in enumerate(times) if pbr.coin_age_eligible(t, inception, days)), len(times))
        assert pbr.coin_age_first_eligible_index(times, inception, days) == expected
    with pytest.raises(ValueError, match="increasing"):
        pbr.coin_age_first_eligible_index([BASE, BASE], BASE, 1.0)


@pytest.mark.parametrize("bad", [True, False, -1, 1.5, float("nan"), float("inf"), "1", 2**64])
def test_inception_transport_never_fabricates_a_required_fact(bad):
    with pytest.raises(ValueError, match="inception_timestamp_ms"):
        inception_timestamp_ms(bad)


@pytest.mark.parametrize("forager", [False, True])
def test_live_admission_applies_to_all_selection_modes_but_not_held_management(forager):
    from passivbot import Passivbot
    symbol = "AAA/USDT:USDT"
    bot = object.__new__(Passivbot)
    bot.config = ages(get_template_config(), 60.0, 7.0)
    bot.positions = {symbol: {side: {"size": 0.0, "price": 0.0} for side in ("long", "short")}}
    bot.first_timestamps = {symbol: BASE}
    bot.get_exchange_time = lambda: BASE + 10 * DAY
    bot.is_forager_mode = lambda _side: forager
    bot.approved_coins_minus_ignored_coins = {side: {symbol} for side in ("long", "short")}
    bot.ignored_coins = {side: set() for side in ("long", "short")}
    assert not bot.is_approved("long", symbol)
    assert bot.is_approved("short", symbol)
    bot.positions[symbol]["long"]["size"] = 1.0
    bot.first_timestamps = {}  # Unknown inception does not change held management.
    assert bot.is_approved("long", symbol)
    assert not bot.is_approved("short", symbol)


def _native_payload(long_days, short_days, *, inception=BASE, interval=1):
    from test_hsl_backtest_config import inputs, payload
    config, markets, _ = inputs()
    ages(config, long_days, short_days)
    for side in ("long", "short"):
        config["bot"][side]["universe"]["approved_coins"] = ["AAA"]
        config["bot"][side]["risk"].update(max_positions=1, total_wallet_exposure_limit=1.0)
        config["bot"][side]["hsl"]["enabled"] = False
    config["bot"]["global"]["hedge_mode"] = True
    config["bot"]["short"]["strategy"] = deepcopy(config["bot"]["long"]["strategy"])
    markets["AAA"]["inception_timestamp_ms"] = inception
    args = list(payload(cfg=config, mss=markets))
    n = 32
    args[0] = np.array([[[101.0, 99.0, 100.0, 1000.0]]] * n, dtype=np.float64)
    args[1] = np.full(n, 50_000.0)
    args[-1].update(first_timestamp_ms=BASE, requested_start_timestamp_ms=BASE,
                    last_valid_indices=[n - 1], candle_interval_minutes=interval)
    return config, markets, args


@pytest.mark.parametrize("interval", [1, 5])
def test_actual_native_replay_has_independent_side_entry_boundaries(interval):
    import passivbot_rust as pbr
    # A 4.5-bar and an 8-bar threshold use already-warmed history; no extra age+warmup delay.
    _, _, args = _native_payload(4.5 * interval / 1440, 8 * interval / 1440, interval=interval)
    fills = pbr.run_backtest(*args)[0]
    first = {}
    for fill in fills:
        if "entry_initial" in str(fill[13]):
            first.setdefault("long" if str(fill[13]).endswith("long") else "short", int(fill[1]))
    assert first == {"long": BASE + 6 * interval * 60_000, "short": BASE + 9 * interval * 60_000}


@pytest.mark.parametrize("inception,days,has_entries", [(None, 1.0, False), (BASE + DAY, 1.0, False), (None, 0.0, True)])
def test_actual_native_unknown_future_and_zero_bypass(inception, days, has_entries):
    import passivbot_rust as pbr
    _, _, args = _native_payload(days, days, inception=inception)
    fills = pbr.run_backtest(*args)[0]
    assert any("entry_initial" in str(fill[13]) for fill in fills) is has_entries


def test_effective_fixed_scenario_transport_and_evaluation_identities():
    from backtest import prep_backtest_args
    from optimization.warmup import _apply_config_overrides
    from optimization.evaluation_contract import build_evaluation_contract
    from optimization.prepared_dataset_identity import materialized_dataset_identity
    from suite_runner import apply_scenario_overrides
    config, markets, _ = _native_payload(60.0, 7.25)
    original = deepcopy(config)
    identity = build_evaluation_contract(config)
    config["optimize"]["fixed_runtime_overrides"] = {"bot.long.universe.minimum_coin_age_days": 1.5}
    assert build_evaluation_contract(config) != identity
    _apply_config_overrides(config, config["optimize"]["fixed_runtime_overrides"])
    apply_scenario_overrides(config, {"bot.short.universe.minimum_coin_age_days": 2.25})
    assert prep_backtest_args(config, markets, "binance")[-1]["minimum_coin_age_days"] == [1.5, 2.25]
    values = np.ones((3, 1, 4)); times = np.arange(3) * 60_000 + BASE
    before = materialized_dataset_identity(["AAA"], values, np.ones(3), times, markets)
    markets["AAA"]["inception_timestamp_ms"] += 1
    assert materialized_dataset_identity(["AAA"], values, np.ones(3), times, markets) != before
    assert original["bot"]["long"]["universe"]["minimum_coin_age_days"] == 60.0


def test_gpu_indices_share_native_boundary_without_cropping_readiness():
    from optimization.gpu.model import native_age_admission_indices, validate_age_eligible_indices
    times = BASE + np.arange(10, dtype=np.int64) * 60_000
    params = {"first_timestamp_ms": BASE, "candle_interval_minutes": 1,
              "minimum_coin_age_days": [4.5 / 1440, 8 / 1440],
              "inception_timestamps_ms": [BASE, None]}
    assert native_age_admission_indices(times, params, 2) == [(5, 8), (10, 10)]
    params["minimum_coin_age_days"][1] = 0.0
    assert native_age_admission_indices(times, params, 2) == [(5, 0), (10, 0)]
    with pytest.raises(ValueError, match="decision clock"):
        native_age_admission_indices(times + np.array([0] * 9 + [1]), params, 2)
    for pair in [(16_777_217, 0), (-1, 0), (True, 0), (0,), (2**31, 0)]:
        with pytest.raises(ValueError, match="index|indices"):
            validate_age_eligible_indices(pair, 2**31)
    validate_age_eligible_indices((0, 16_777_216), 16_777_216)


@pytest.mark.parametrize("layer", ["root", "fixed", "scenario", "nested_scenario"])
def test_released_v8_age_fans_out_reports_new_semantics_without_mutation(layer):
    from released_v8_fixtures import released_v8_policy
    from tools.migrate_config_v9 import migrate
    source = released_v8_policy()
    source["live"]["minimum_coin_age_days"] = 7.25
    patch = {"live.minimum_coin_age_days": 2.5}
    if layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    elif layer in {"scenario", "nested_scenario"}:
        source["backtest"]["scenarios"] = [{"label": "age", "overrides": patch if layer == "scenario" else {"live": {"minimum_coin_age_days": 2.5}}}]
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == original
    for side in ("long", "short"):
        assert result.config["bot"][side]["universe"]["minimum_coin_age_days"] == 7.25
    if layer != "root":
        converted = (result.config["optimize"]["fixed_runtime_overrides"] if layer == "fixed"
                     else result.config["backtest"]["scenarios"][0]["overrides"])
        assert converted == {f"bot.{side}.universe.minimum_coin_age_days": 2.5 for side in ("long", "short")}
    assert any("every" in change["reason"] and "age" in change["reason"] for change in result.changes)


@pytest.mark.parametrize("days", [True, False, -1, float("nan"), float("inf")])
def test_native_age_endpoints_reject_invalid_policy_instead_of_coercing_bool(days):
    import passivbot_rust as pbr
    with pytest.raises((ValueError, TypeError), match="minimum_coin_age_days"):
        pbr.coin_age_eligible(BASE, BASE, days)
    with pytest.raises((ValueError, TypeError), match="minimum_coin_age_days"):
        pbr.coin_age_first_eligible_index([BASE, BASE + 60_000], BASE, days)


@pytest.mark.parametrize("times", [[True], [1.5], [-1], [2**64], [BASE + 1, BASE], [BASE, BASE]])
def test_native_replay_index_rejects_malformed_or_unsorted_facts(times):
    import passivbot_rust as pbr
    with pytest.raises((ValueError, TypeError, OverflowError)):
        pbr.coin_age_first_eligible_index(times, BASE, 1.0)


@pytest.mark.parametrize("pair", [None, [], [0.0], [0.0, 0.0, 0.0], [True, 0.0], [0.0, float("nan")]])
def test_native_backtest_requires_complete_valid_age_policy(pair):
    import passivbot_rust as pbr
    _, _, args = _native_payload(0.0, 0.0)
    if pair is None:
        args[-1].pop("minimum_coin_age_days")
    else:
        args[-1]["minimum_coin_age_days"] = pair
    with pytest.raises((ValueError, TypeError, KeyError), match="minimum_coin_age_days"):
        pbr.run_backtest(*args)


@pytest.mark.parametrize("mode", [None, "normal", "graceful_stop"])
def test_actual_native_held_dca_and_close_intent_does_not_change_with_unknown_age(mode):
    import passivbot_rust as pbr
    from test_orchestrator_json_api import make_symbol, make_input, compute
    symbol = make_symbol(0, bid=100.0, ask=100.0, long_pos_size=1.0,
                         long_pos_price=100.0, long_mode=mode)
    packet = make_input(balance=1000.0, symbols=[symbol])
    packet["timestamp_ms"] = BASE
    before = compute(pbr, packet)["orders"]
    assert any(order["order_type"].startswith("close_") for order in before)
    assert any(order["order_type"].startswith("entry_grid") for order in before)
    packet["symbols"][0]["long"]["minimum_coin_age_days"] = 60.0
    packet["symbols"][0]["inception_timestamp_ms"] = None
    assert compute(pbr, packet)["orders"] == before
    # Age does not override ordinary market tradability either.
    packet["symbols"][0]["tradable"] = False
    unavailable = compute(pbr, packet)["orders"]
    packet["symbols"][0]["long"]["minimum_coin_age_days"] = 0.0
    assert compute(pbr, packet)["orders"] == unavailable


def test_native_live_packet_missing_age_is_malformed():
    import passivbot_rust as pbr
    from test_orchestrator_json_api import make_symbol, make_input, compute
    packet = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    packet["symbols"][0]["long"].pop("minimum_coin_age_days")
    with pytest.raises((ValueError, TypeError), match="minimum_coin_age_days"):
        compute(pbr, packet)


def test_released_v8_coin_file_age_does_not_gain_universe_override_permission(tmp_path):
    from released_v8_fixtures import released_v8_policy
    from tools.migrate_config_v9 import migrate
    source = released_v8_policy()
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"live": {"minimum_coin_age_days": 7.25}}))
    source["coin_overrides"] = {"AAA": {"override_config_path": file.name}}
    before, contents = deepcopy(source), file.read_bytes()
    result = migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "input.json"))
    assert result.config["coin_overrides"] == {"AAA": {}}
    assert all(result.config["bot"][side]["universe"]["minimum_coin_age_days"] == 60.0 for side in ("long", "short"))
    assert any("ignored" in item["reason"] and "minimum_coin_age_days" in item["path"] for item in result.changes)
    assert source == before and file.read_bytes() == contents


@pytest.mark.parametrize("young_side", ["long", "short"])
@pytest.mark.parametrize("selection", ["static", "forager", "always_include", "forced_normal"])
def test_actual_native_one_way_arbitrates_only_age_eligible_side(young_side, selection):
    import passivbot_rust as pbr
    config, markets, args = _native_payload(7.0 if young_side == "long" else 60.0,
                                           7.0 if young_side == "short" else 60.0)
    from backtest import prep_backtest_args
    config["backtest"]["coins"]["binance"] = ["AAA", "BBB"]
    markets["BBB"] = deepcopy(markets["AAA"])
    for side in ("long", "short"):
        bot = config["bot"][side]
        bot["universe"]["approved_coins"] = ["AAA", "BBB"]
        bot["risk"]["max_positions"] = 2 if selection == "static" else 1
        bot["forager"].update(volume_ema_span_minutes=1.0, volatility_ema_span_minutes=1.0,
                              score_weights={"volume": 1.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": 0.0})
        if selection == "always_include":
            bot["forager"]["always_include"] = ["AAA", "BBB"]
    if selection == "forced_normal":
        config["coin_overrides"] = {coin: {"live": {"modes": {"long": "normal", "short": "normal"}}} for coin in ("AAA", "BBB")}
    args[0] = np.repeat(args[0], 2, axis=1)
    args[2], args[3], args[4], _ = prep_backtest_args(config, markets, "binance")
    assert all(pair[young_side]["is_forced_active"] is (selection in {"always_include", "forced_normal"}) for pair in args[2])
    params = args[-1]
    params.update(hedge_mode=False, coins=["AAA", "BBB"], first_timestamp_ms=BASE + 10 * DAY,
                  requested_start_timestamp_ms=BASE + 10 * DAY, first_valid_indices=[0, 0],
                  last_valid_indices=[31, 31], warmup_minutes=[1, 1], trade_start_indices=[1, 1],
                  inception_timestamps_ms=[BASE, BASE])
    entries = [row for row in pbr.run_backtest(*args)[0] if str(row[13]).startswith("entry_")]
    assert entries and all(str(row[13]).endswith("_" + young_side) for row in entries)


@pytest.mark.parametrize("facts", [[True], [1.5], [-1], [None, None]])
def test_native_backtest_rejects_malformed_inception_transport(facts):
    import passivbot_rust as pbr
    _, _, args = _native_payload(0.0, 0.0)
    args[-1]["inception_timestamps_ms"] = facts
    with pytest.raises((ValueError, TypeError, OverflowError)):
        pbr.run_backtest(*args)


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_cli_side_age_values_keep_fractional_days(command):
    from test_shared_bot_cli_policy import parser_for
    from config_utils import update_config_with_args
    parser, keys = parser_for(command)
    config = get_template_config()
    parsed = parser.parse_args(["--minimum-coin-age-days-long", "0.5",
                                "--bot.short.universe.minimum_coin_age_days", "7.25"])
    update_config_with_args(config, parsed, allowed_keys=keys)
    assert [config["bot"][side]["universe"]["minimum_coin_age_days"] for side in ("long", "short")] == [0.5, 7.25]
    with pytest.raises(SystemExit):
        parser.parse_args(["--bot.global.universe.minimum_coin_age_days", "1"])


def test_editor_schema_has_same_fixed_age_boundary_and_no_coin_permission():
    from jsonschema import Draft202012Validator
    from config.json_schema import generate_schema
    validator = Draft202012Validator(generate_schema())
    source = ages(get_template_config(), 0.5, 7.25)
    validator.validate(source)
    source["optimize"]["fixed_runtime_overrides"] = {"bot.short.universe.minimum_coin_age_days": True}
    assert list(validator.iter_errors(source))
    source["optimize"]["fixed_runtime_overrides"] = {}
    source["coin_overrides"] = {"AAA": {"bot": {"long": {"universe": {"minimum_coin_age_days": 0.0}}}}}
    assert list(validator.iter_errors(source))


def test_cached_native_packet_rejects_changed_age_or_facts():
    from backtest import prep_backtest_args
    config, markets, args = _native_payload(60.0, 7.25)
    cached = deepcopy(args[-1])
    config["bot"]["long"]["universe"]["minimum_coin_age_days"] = 2.5
    with pytest.raises(ValueError, match="cached age admission"):
        prep_backtest_args(config, markets, "binance", backtest_params=cached)
