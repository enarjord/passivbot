"""Candidate transport excludes recording snapshots, retaining canonical lowering."""

from copy import deepcopy

import pytest

from config.activation import ActivationSource, PreparedActivationBot
from config.inherited_bounds import prepare_bound_plan
from optimization.gpu.parameters import prepare_candidate_parameters
from tools.gpu_parity import build_parser, fixture_inputs


def inputs(strategy="trailing_martingale", sides="both", scope="coin"):
    config, _candles, markets, _btc, _timestamps = fixture_inputs(
        build_parser().parse_args([
            "--fixture", strategy, "--sides", sides, "--hsl", scope,
            "--coins", "2", "--bars", "128",
        ])
    )
    prepare_bound_plan(config)
    # Match the retained activation wrapper found in finalized CLI snapshots.
    config["_raw_effective"] = {
        "bot": PreparedActivationBot(
            deepcopy(dict(config["bot"])), ActivationSource(True, sides == "both")
        )
    }
    return config, markets


def legacy_parameters(config, markets, monkeypatch):
    """Use the former full input at the unchanged canonical lowering boundary."""
    import config.runtime_compile as runtime

    compiled = runtime.compile_runtime_config(config, runtime="backtest", record_step=False)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "compile_runtime_config", lambda *_args, **_kwargs: compiled)
        return prepare_candidate_parameters(config, markets, "binance")


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_candidate_runtime_projection_matches_full_canonical_payload(
    strategy, sides, scope, monkeypatch,
):
    config, markets = inputs(strategy, sides, scope)
    from optimization.warmup import _apply_config_overrides

    # Canonical fixed/scenario changes have already been materialized before
    # candidate transport. Coin pins remain dataset-owned and are not globals.
    entry = ("entry." if strategy == "trailing_martingale" else "")
    _apply_config_overrides(config, {
        f"bot.long.strategy.{strategy}.{entry}ema_span_0_minutes": 37.25,
        "bot.long.entry_cooldown.min_duration_minutes": 3.5,
        "bot.portfolio.unstuck.memory_ema_span_minutes": 1777.25,
        "bot.portfolio.loss_gate.memory_ema_span_minutes": 2888.5,
    })
    config["coin_overrides"] = {"COIN00": {"bot": {"long": {
        "strategy": {strategy: ({"entry": {"initial_qty_fraction": 0.4}}
                                if strategy == "trailing_martingale"
                                else {"base_qty_fraction": 0.4})},
    }}}}
    if scope == "coin":
        config["coin_overrides"]["COIN00"]["bot"]["long"]["hsl"] = {"enabled": False}
    original = deepcopy(config)
    bot = config["bot"]
    owners = dict(vars(bot))
    bounds = config["optimize"]["bounds"]
    bound_owners = dict(vars(bounds))
    snapshot = config["_raw_effective"]["bot"]
    activation = snapshot._activation_source

    expected = legacy_parameters(config, markets, monkeypatch)
    assert prepare_candidate_parameters(config, markets, "binance") == expected
    assert config == original
    assert config["bot"] is bot
    assert all(getattr(bot, key) is value for key, value in owners.items())
    assert config["optimize"]["bounds"] is bounds
    assert all(getattr(bounds, key) is value for key, value in bound_owners.items())
    assert config["_raw_effective"]["bot"] is snapshot
    assert snapshot._activation_source is activation


@pytest.mark.parametrize("decoration", [None, 17, "not a config", object(), "cycle"])
def test_ignored_snapshot_shapes_retain_full_lowering_result(decoration, monkeypatch):
    config, markets = inputs()
    if decoration == "cycle":
        decoration = {}
        decoration["self"] = decoration
    for key in ("_raw", "_raw_effective", "_transform_log"):
        config[key] = decoration
    expected = legacy_parameters(config, markets, monkeypatch)
    assert prepare_candidate_parameters(config, markets, "binance") == expected
    assert all(config[key] is decoration for key in ("_raw", "_raw_effective", "_transform_log"))


def test_only_known_recording_decorations_avoid_copying(monkeypatch):
    config, markets = inputs()
    copied = []

    class CopyObservation:
        def __init__(self, name):
            self.name = name

        def __deepcopy__(self, memo):
            copied.append(self.name)
            return self

    for key in ("_raw", "_raw_effective", "_transform_log", "_other_runtime_owner", "_coins_sources"):
        config[key] = CopyObservation(key)
    prepare_candidate_parameters(config, markets, "binance")
    assert set(copied) == {"_other_runtime_owner", "_coins_sources"}


@pytest.mark.parametrize("foreign", ["deepcopy", "items", "key"])
def test_foreign_roots_and_keys_retain_legacy_compilation(foreign, monkeypatch):
    config, markets = inputs()
    visits = []

    class Root(dict):
        def __deepcopy__(self, memo):
            visits.append("root_deepcopy")
            copied = deepcopy(dict(self), memo)
            copied["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] = 2345.5
            return copied

        def items(self):
            if foreign == "items":
                pytest.fail("candidate transport inspected a foreign root's items")
            return super().items()

    class Key(str):
        pass

    if foreign == "key":
        class Decoration:
            def __deepcopy__(self, memo):
                visits.append("foreign_key_decoration")
                return self

        del config["_raw_effective"]
        config[Key("_raw_effective")] = Decoration()
    else:
        config = Root(config)
    expected = legacy_parameters(config, markets, monkeypatch)
    assert visits
    visits.clear()
    assert prepare_candidate_parameters(config, markets, "binance") == expected
    assert visits
    if foreign != "key":
        assert expected["unstuck_memory_ema_span_minutes"] == 2345.5


@pytest.mark.parametrize("malformed", ["version", "retired", "number", "source", "market", "inactive_pin"])
def test_projection_preserves_required_input_failures(malformed, monkeypatch):
    config, markets = inputs()
    if malformed == "version":
        config["config_version"] = "v8.0.0"
    elif malformed == "retired":
        config["bot"]["long"]["hsl"]["restart_after_red_policy"] = "never"
    elif malformed == "number":
        config["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = float("nan")
    elif malformed == "source":
        config["bot"]._bot_policy = object()
    elif malformed == "market":
        del markets["COIN00"]["qty_step"]
    else:
        config, markets = inputs(scope="portfolio")
        config["coin_overrides"] = {"COIN00": {"bot": {"long": {"hsl": {"enabled": False}}}}}

    with pytest.raises((ValueError, TypeError, KeyError)) as expected:
        legacy_parameters(config, markets, monkeypatch)
    with pytest.raises(type(expected.value)) as actual:
        prepare_candidate_parameters(config, markets, "binance")
    assert str(actual.value) == str(expected.value)
