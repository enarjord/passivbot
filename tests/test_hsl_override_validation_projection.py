"""Read-only HSL validation retains selector and foreign-copy behavior."""

from copy import deepcopy

import pytest

import config.hsl as hsl


def config(mode="coin"):
    return {"bot": {"portfolio": {"hsl": {"scope": mode}}}, "live": {}}


def patch(path, value, nested):
    if not nested:
        return {path: value}
    result = {}
    node = result
    *parents, leaf = path.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value
    return result


def legacy_validation(config, overrides, *, allow_engine=False):
    """The former copying boundary, using the unchanged validation consumers."""
    leaves = dict(hsl._walk_paths(overrides))
    if "live.hsl_engine" in leaves and not allow_engine:
        raise ValueError(
            "live.hsl_engine is a startup selector, not a scenario or optimizer override"
        )
    selected = deepcopy(config)
    if "bot.portfolio.hsl.scope" in leaves:
        selected.setdefault("bot", {}).setdefault("portfolio", {}).setdefault("hsl", {})["scope"] = leaves["bot.portfolio.hsl.scope"]
    if "live.hsl_engine" in leaves:
        selected["live"]["hsl_engine"] = leaves["live.hsl_engine"]
    if not allow_engine:
        hsl.engine(selected)
    for path in leaves:
        hsl.validate_parameter_path(path, hsl._mode(selected))
    hsl.validate_hsl_paths({**selected, "optimize": {"scoring": overrides}})


def outcome(function, config, overrides, *, allow_engine=False):
    try:
        result = function(config, overrides, allow_engine=allow_engine)
    except Exception as error:
        return type(error), str(error)
    return result


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["empty", "hsl", "other"])
def test_ordinary_validation_preserves_results_without_copying(mode, nested, kind, monkeypatch):
    value = config(mode)
    if kind == "empty":
        overrides = {}
    else:
        owner = "portfolio" if mode == "portfolio" else "long"
        path = f"bot.{owner}.hsl.drawdown_threshold_fraction" if kind == "hsl" else "bot.long.entry_cooldown.min_duration_minutes"
        overrides = patch(path, 0.25, nested)
    before = deepcopy(value)
    expected = outcome(legacy_validation, value, overrides)
    assert expected is None
    monkeypatch.setattr(hsl, "deepcopy", lambda *_args: pytest.fail("ordinary validation copied config"))
    assert outcome(hsl.validate_override_paths, value, overrides) == expected
    assert value == before


@pytest.mark.parametrize("original", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("selected", ["coin", "position_side", "portfolio", "invalid"])
@pytest.mark.parametrize("nested", [False, True])
def test_scope_selectors_still_copy_and_validate_final_scope(original, selected, nested, monkeypatch):
    value = config(original)
    overrides = patch("bot.portfolio.hsl.scope", selected, nested)
    overrides["bot.long.hsl.enabled"] = False
    expected = outcome(legacy_validation, value, overrides)
    copied = []

    def copy_selected(source):
        copied.append(source)
        return deepcopy(source)

    monkeypatch.setattr(hsl, "deepcopy", copy_selected)
    assert outcome(hsl.validate_override_paths, value, overrides) == expected
    assert copied == [value]
    assert value["bot"]["portfolio"]["hsl"]["scope"] == original


@pytest.mark.parametrize("allow_engine", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_engine_selector_preserves_pre_copy_rejection_and_detached_mutation(allow_engine, nested, monkeypatch):
    value = config()
    overrides = patch("live.hsl_engine", "legacy", nested)
    expected = outcome(legacy_validation, value, overrides, allow_engine=allow_engine)
    calls = []

    def copy_selected(source):
        calls.append(source)
        return deepcopy(source)

    monkeypatch.setattr(hsl, "deepcopy", copy_selected)
    assert outcome(hsl.validate_override_paths, value, overrides, allow_engine=allow_engine) == expected
    assert len(calls) == int(allow_engine)
    assert value["live"] == {}


@pytest.mark.parametrize("malformed", [
    "scope", "scope_type", "bot", "portfolio", "hsl", "live", "engine",
    "removed_path", "removed_metric", "inactive_path", "competing_errors",
])
def test_required_failures_and_error_order_match_copying_validation(malformed):
    value = config()
    overrides = {"bot.long.hsl.enabled": True}
    if malformed == "scope":
        value["bot"]["portfolio"]["hsl"]["scope"] = "invalid"
    elif malformed == "scope_type":
        value["bot"]["portfolio"]["hsl"]["scope"] = None
    elif malformed in {"bot", "portfolio", "hsl", "live"}:
        if malformed == "live":
            value["live"] = None
        elif malformed == "bot":
            value["bot"] = None
        elif malformed == "portfolio":
            value["bot"]["portfolio"] = None
        else:
            value["bot"]["portfolio"]["hsl"] = None
    elif malformed == "engine":
        value["live"]["hsl_engine"] = "legacy"
    elif malformed == "removed_path":
        overrides = {"bot.long.hsl.restart_after_red_policy": "never"}
    elif malformed == "removed_metric":
        overrides = {"arbitrary_metric": "hard_stop_time_in_yellow"}
    elif malformed == "inactive_path":
        value["bot"]["portfolio"]["hsl"]["scope"] = "portfolio"
    else:
        value["bot"]["portfolio"]["hsl"]["scope"] = "invalid"
        overrides = {"bot.long.hsl.restart_after_red_policy": "never"}
    expected = outcome(legacy_validation, value, overrides)
    assert expected is not None
    assert outcome(hsl.validate_override_paths, value, overrides) == expected


@pytest.mark.parametrize("where", [
    "root", "live", "bot", "portfolio", "hsl", "scope", "key",
    "live_key", "bot_key", "portfolio_key", "hsl_key",
])
def test_foreign_consumed_paths_retain_copy_protocol(where):
    value = config()
    calls = []

    class Foreign(dict):
        def __deepcopy__(self, memo):
            calls.append(where)
            raise RuntimeError("foreign selector copy")

    class ForeignString(str):
        def __deepcopy__(self, memo):
            calls.append(where)
            raise RuntimeError("foreign selector copy")

    if where == "root":
        value = Foreign(value)
    elif where == "key":
        value[ForeignString("extra")] = 1
    elif where.endswith("_key"):
        target = value["live"] if where == "live_key" else value["bot"]
        if where in {"portfolio_key", "hsl_key"}:
            target = target["portfolio"]
        if where == "hsl_key":
            target = target["hsl"]
        target[ForeignString("extra")] = 1
    elif where == "scope":
        value["bot"]["portfolio"]["hsl"]["scope"] = ForeignString("coin")
    elif where == "live":
        value["live"] = Foreign()
    elif where == "bot":
        value["bot"] = Foreign(value["bot"])
    elif where == "portfolio":
        value["bot"]["portfolio"] = Foreign(value["bot"]["portfolio"])
    else:
        value["bot"]["portfolio"]["hsl"] = Foreign(value["bot"]["portfolio"]["hsl"])
    expected = outcome(legacy_validation, value, {})
    assert expected == (RuntimeError, "foreign selector copy")
    calls.clear()
    assert outcome(hsl.validate_override_paths, value, {}) == expected
    assert calls == [where]


def test_unused_recording_metadata_does_not_enter_validation_copy_boundary():
    class RecordingOnly:
        def __deepcopy__(self, memo):
            pytest.fail("unused recording metadata was copied")

    value = config()
    recording = {"owner": RecordingOnly()}
    recording["self"] = recording
    value["_raw_effective"] = recording
    assert hsl.validate_override_paths(value, {}) is None
    assert value["_raw_effective"] is recording and recording["self"] is recording


def prepared_bot(kind):
    from config.activation import ActivationSource, PreparedActivationBot
    if kind == "activation":
        return PreparedActivationBot(config()["bot"], ActivationSource(True, False))
    from config import prepare_config
    from config.schema import get_template_config
    from config.bot_inheritance import PreparedBotPolicy
    value = prepare_config(get_template_config(), verbose=False, target="canonical", runtime=None)["bot"]
    assert type(value) is PreparedBotPolicy
    return value


@pytest.mark.parametrize("kind", ["activation", "policy"])
def test_known_prepared_bot_values_and_source_owners_remain_read_only(kind, monkeypatch):
    bot = prepared_bot(kind)
    value = {"bot": bot, "live": {}}
    before = deepcopy(bot)
    owners = {key: item for key, item in getattr(bot, "__dict__", {}).items()}
    activation = getattr(bot, "_activation_source", None)
    monkeypatch.setattr(hsl, "deepcopy", lambda *_args: pytest.fail("prepared selector view was copied"))
    assert hsl.validate_override_paths(value, {"bot.long.hsl.enabled": False}) is None
    assert bot == before and value["bot"] is bot
    assert all(getattr(bot, key) is item for key, item in owners.items())
    assert getattr(bot, "_activation_source", None) is activation


@pytest.mark.parametrize("kind,protocol", [
    (kind, protocol)
    for kind in ("activation", "policy")
    for protocol in ("class_deepcopy", "class_reduce", "instance_deepcopy", "registered")
    if (kind, protocol) != ("activation", "instance_deepcopy")
])
def test_known_wrappers_with_custom_copy_protocol_retain_fallback(kind, protocol, monkeypatch):
    from copyreg import dispatch_table
    bot = prepared_bot(kind)
    value = {"bot": bot, "live": {}}

    def changed(*_args):
        return {"portfolio": {"hsl": {"scope": "portfolio"}}}

    if protocol == "class_deepcopy":
        monkeypatch.setattr(type(bot), "__deepcopy__", changed, raising=False)
    elif protocol == "class_reduce":
        monkeypatch.setattr(type(bot), "__reduce_ex__", lambda *_args: (dict, (changed(),)), raising=False)
    elif protocol == "instance_deepcopy":
        monkeypatch.setattr(bot, "__deepcopy__", changed, raising=False)
    else:
        monkeypatch.setitem(dispatch_table, type(bot), lambda *_args: (dict, (changed(),)))
    overrides = {"bot.long.hsl.enabled": False}
    expected = outcome(legacy_validation, value, overrides)
    assert expected and "inactive in portfolio mode" in expected[1]
    assert outcome(hsl.validate_override_paths, value, overrides) == expected
