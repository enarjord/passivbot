"""Released strategy conversion is independent of current native vocabulary."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest


# This vocabulary is stdlib-only: these regressions can run before a native
# rebuild without accidentally exercising a stale strategy implementation.
FROZEN = runpy.run_path(str(
    Path(__file__).parents[1] / "src/config/migrations/released_v8_strategy.py"
))


def test_frozen_v8_strategy_vocabulary_has_33_immutable_paths():
    mappings = FROZEN["RELEASED_V8_STRATEGY_PATH_RENAMES"]
    assert {kind: len(fields) for kind, fields in mappings.items()} == {
        "trailing_martingale": 23, "ema_anchor": 10,
    }
    with pytest.raises(TypeError):
        mappings["another"] = {}
    with pytest.raises(TypeError):
        mappings["ema_anchor"][("offset",)] = ("different",)
    ids = FROZEN["RELEASED_V8_STRATEGY_OPTIMIZER_PATHS"]
    assert ids["trailing_martingale"]["ema_span_0"] == ("entry", "ema_span_0_minutes")
    assert "entry_ema_span_0" not in ids["trailing_martingale"]
    assert ids["ema_anchor"]["ema_span_0"] == ("ema_span_0_minutes",)


@pytest.mark.parametrize("kind,expected", [
    ("trailing_martingale", ("entry", "reentry_qty_multiplier")),
    ("ema_anchor", ("entry_qty_exposure_coefficient",)),
])
@pytest.mark.parametrize("prefix", ["long", "bot.long", "coin_overrides.BTC.bot.long"])
def test_frozen_v8_dotted_native_id_uses_authored_strategy(kind, expected, prefix):
    expected_prefix = ("bot", "long") if prefix == "long" else tuple(prefix.split("."))
    assert FROZEN["released_strategy_selector"](
        (*prefix.split("."), "entry_double_down_factor"), strategy_kind=kind,
    ) == (*expected_prefix, "strategy", kind, *expected)


@pytest.mark.parametrize("selector", [
    "long_entry_double_down_factor", "entry_double_down_factor",
    "bot.*.entry_double_down_factor",
    "bot.long.strategy.trailing_martingale.entry_double_down_factor",
])
def test_frozen_v8_selector_does_not_guess_new_native_id_aliases(selector):
    parts = tuple(selector.split("."))
    assert FROZEN["released_strategy_selector"](
        parts, strategy_kind="trailing_martingale",
    ) == parts


def test_frozen_v8_flat_bounds_preserve_values_and_report_final_paths():
    config = {"optimize": {"bounds": {
        "long_entry_double_down_factor": [1.125, 2.375, 0.125],
        "short_ema_span_0": [12.5, 72.5],
    }}}
    original = deepcopy(config)
    changes = []
    FROZEN["rewrite_released_strategy_bounds"](
        config, strategy_kind="trailing_martingale",
        record=lambda *args: changes.append(args),
    )
    assert config == {"optimize": {"bounds": {
        "long": {"strategy": {"trailing_martingale": {
            "entry": {"reentry_qty_multiplier": [1.125, 2.375, 0.125]},
        }}},
        "short": {"strategy": {"trailing_martingale": {
            "entry": {"ema_span_0_minutes": [12.5, 72.5]},
        }}},
    }}}
    assert original["optimize"]["bounds"]["short_ema_span_0"] == [12.5, 72.5]
    assert len(changes) == 2
    assert changes[0][2] == {
        "optimize.bounds.long.strategy.trailing_martingale.entry.reentry_qty_multiplier":
        [1.125, 2.375, 0.125],
    }


def test_frozen_v8_flat_bound_collision_rejects_conflicting_authority():
    config = {"optimize": {"bounds": {
        "long_entry_double_down_factor": [1, 2],
        "long": {"strategy": {"trailing_martingale": {
            "entry": {"reentry_qty_multiplier": [3, 4]},
        }}},
    }}}
    with pytest.raises(ValueError, match="conflicts"):
        FROZEN["rewrite_released_strategy_bounds"](
            config, strategy_kind="trailing_martingale",
        )


@pytest.mark.parametrize("kind,old,new", [
    ("trailing_martingale", "entry.double_down_factor", "entry.reentry_qty_multiplier"),
    ("ema_anchor", "entry_double_down_factor", "entry_qty_exposure_coefficient"),
])
def test_released_public_names_do_not_depend_on_runtime_projection(monkeypatch, kind, old, new):
    from config import strategy_spec
    from config.public_names import released_public_path

    def unexpected_current_metadata(*args, **kwargs):
        raise AssertionError("released spelling conversion must not read current native metadata")
    monkeypatch.setattr(strategy_spec, "get_strategy_spec", unexpected_current_metadata)
    assert released_public_path(("bot", "long", "strategy", kind, *old.split("."))) == (
        "bot", "long", "strategy", kind, *new.split("."),
    )


@pytest.mark.parametrize("kind,local", [
    ("trailing_martingale", ("entry", "reentry_qty_multiplier")),
    ("ema_anchor", ("entry_qty_exposure_coefficient",)),
])
def test_full_v8_flat_bounds_and_dotted_fixed_selector_migrate_without_mutation(kind, local):
    from config.migrations.released_v8 import get_released_v8_defaults
    from tools.migrate_config_v9 import migrate

    config = get_released_v8_defaults(strategy_kind=kind)
    config["backtest"]["visible_metrics"] = [
        name for name in config["backtest"]["visible_metrics"]
        if not name.endswith("_w") and not name.startswith("strategy_eq_underwater_pct_")
    ]
    config["optimize"]["scoring"] = [
        row for row in config["optimize"]["scoring"]
        if not row["metric"].endswith("_w")
        and not row["metric"].startswith("strategy_eq_underwater_pct_")
    ]
    config["optimize"]["bounds"] = {"long_entry_double_down_factor": [1.125, 2.375]}
    config["optimize"]["fixed_params"] = ["bot.long.entry_double_down_factor"]
    config["backtest"]["scenarios"] = [{"label": "fractional", "overrides": {
        "bot.short.entry_double_down_factor": 1.875,
    }}]
    original = deepcopy(config)
    converted = migrate(config, memory_span_minutes=123.5)
    assert config == original
    bound = converted.config["optimize"]["bounds"]["long"]["strategy"][kind]
    for part in local:
        bound = bound[part]
    assert bound == [1.125, 2.375]
    target = ".".join(("bot", "long", "strategy", kind, *local))
    assert converted.config["optimize"]["fixed_params"] == [target]
    scenario_target = ".".join(("bot", "short", "strategy", kind, *local))
    assert converted.config["backtest"]["scenarios"][0]["overrides"][scenario_target] == 1.875
    bound_target = ".".join(("optimize", "bounds", "long", "strategy", kind, *local))
    assert any(isinstance(change["new"], dict) and bound_target in change["new"]
               for change in converted.changes)


def _released_flat_bound_source(kind, missing_spans=("ema_span_0", "ema_span_1")):
    from config.migrations.released_v8 import get_released_v8_defaults

    config = get_released_v8_defaults(strategy_kind=kind)
    config["backtest"]["visible_metrics"] = [
        name for name in config["backtest"]["visible_metrics"]
        if not name.endswith("_w") and not name.startswith("strategy_eq_underwater_pct_")
    ]
    config["optimize"]["scoring"] = [
        row for row in config["optimize"]["scoring"]
        if not row["metric"].endswith("_w")
        and not row["metric"].startswith("strategy_eq_underwater_pct_")
    ]
    for side in ("long", "short"):
        for span in missing_spans:
            config["bot"][side]["unstuck"].pop(span)
    return config


def _at(node, path):
    for part in path:
        node = node[part]
    return node


@pytest.mark.parametrize("kind,local", [
    ("trailing_martingale", ("entry", "reentry_qty_multiplier")),
    ("ema_anchor", ("entry_qty_exposure_coefficient",)),
])
def test_single_released_flat_quantity_bound_with_omitted_unstuck_spans(kind, local):
    from tools.migrate_config_v9 import migrate

    config = _released_flat_bound_source(kind)
    config["optimize"]["bounds"] = {"long_entry_double_down_factor": [1.125, 2.375]}
    original = deepcopy(config)
    converted = migrate(config, memory_span_minutes=123.5)
    assert _at(converted.config["optimize"]["bounds"], ("long", "strategy", kind, *local)) == [1.125, 2.375]
    assert config == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("missing_spans", [
    ("ema_span_0", "ema_span_1"), ("ema_span_0",), ("ema_span_1",),
])
def test_all_released_flat_strategy_bounds_precede_missing_unstuck_inference(kind, missing_spans):
    from tools.migrate_config_v9 import migrate

    config = _released_flat_bound_source(kind, missing_spans)
    fields = FROZEN["RELEASED_V8_STRATEGY_PATH_RENAMES"][kind]
    native_ids = FROZEN["RELEASED_V8_STRATEGY_OPTIMIZER_PATHS"][kind]
    expected = {}
    for side in ("long", "short"):
        for old, new in fields.items():
            old_id = next(key for key, path in native_ids.items() if path == new)
            expected[f"{side}_{old_id}"] = (
                (side, "strategy", kind, *new),
                deepcopy(_at(config["optimize"]["bounds"], (side, "strategy", kind, *old))),
            )
        strategy = config["bot"][side]["strategy"][kind]
        spans = strategy["entry"] if kind == "trailing_martingale" else strategy
        spans.update(ema_span_0=101.25, ema_span_1=902.75)
        for span in ("ema_span_0", "ema_span_1"):
            if span not in missing_spans:
                config["bot"][side]["unstuck"][span] = 237.5
    config["optimize"]["bounds"] = {
        key: deepcopy(value) for key, (_, value) in expected.items()
    }
    for side in ("long", "short"):
        explicit = {
            span: [211.25, 277.75, 0.25]
            for span in ("ema_span_0", "ema_span_1") if span not in missing_spans
        }
        if explicit:
            config["optimize"]["bounds"][side] = {"unstuck": explicit}
    original = deepcopy(config)
    converted = migrate(config, memory_span_minutes=123.5)
    assert config == original
    bounds = converted.config["optimize"]["bounds"]
    for old_id, (path, value) in expected.items():
        assert _at(bounds, path) == value
        target = ".".join(("optimize", "bounds", *path))
        assert any(
            change["path"] == f"optimize.bounds.{old_id}"
            and change["old"] == value and change["new"] == {target: value}
            for change in converted.changes
        )
    for side in ("long", "short"):
        for span, inherited in (("ema_span_0", 101.25), ("ema_span_1", 902.75)):
            value = inherited if span in missing_spans else 237.5
            assert converted.config["bot"][side]["unstuck"][f"{span}_minutes"] == value
            expected_bound = [value, value] if span in missing_spans else [211.25, 277.75, 0.25]
            assert bounds[side]["unstuck"][f"{span}_minutes"] == expected_bound


@pytest.mark.parametrize("kind,old,local", [
    ("trailing_martingale", ("entry", "double_down_factor"), ("entry", "reentry_qty_multiplier")),
    ("ema_anchor", ("entry_double_down_factor",), ("entry_qty_exposure_coefficient",)),
])
@pytest.mark.parametrize("conflict", [False, True])
def test_missing_unstuck_flat_bound_keeps_equal_and_conflicting_authority(kind, old, local, conflict):
    from tools.migrate_config_v9 import migrate

    config = _released_flat_bound_source(kind)
    value = [1.125, 2.375]
    config["optimize"]["bounds"] = {
        "long_entry_double_down_factor": value,
        "long": {"strategy": {kind: {}}},
    }
    target = config["optimize"]["bounds"]["long"]["strategy"][kind]
    for part in old[:-1]:
        target = target.setdefault(part, {})
    target[old[-1]] = [3.125, 4.375] if conflict else deepcopy(value)
    original = deepcopy(config)
    if conflict:
        with pytest.raises(ValueError, match="conflicts"):
            migrate(config, memory_span_minutes=123.5)
    else:
        converted = migrate(config, memory_span_minutes=123.5)
        assert _at(converted.config["optimize"]["bounds"], ("long", "strategy", kind, *local)) == value
    assert config == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_flat_bound_conversion_does_not_admit_old_ids_in_v9(kind):
    from config import get_template_config, prepare_config

    config = get_template_config()
    config["bot"]["portfolio"]["strategy_kind"] = kind
    config["optimize"]["bounds"] = {"long_entry_double_down_factor": [1.125, 2.375]}
    original = deepcopy(config)
    with pytest.raises(ValueError, match=r"long_entry_double_down_factor is an unknown configuration field"):
        prepare_config(config, verbose=False)
    assert config == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("bound", [0.0, [0.0, 0.0]])
def test_released_flat_zero_span_preserves_disabled_strategy_and_unstuck_fallback(kind, bound):
    from tools.migrate_config_v9 import migrate

    config = _released_flat_bound_source(kind)
    strategy = config["bot"]["short"]["strategy"][kind]
    spans = strategy["entry"] if kind == "trailing_martingale" else strategy
    fallback = spans["ema_span_0"]
    spans["ema_span_0"] = 0.0
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0.0
    config["optimize"]["bounds"] = {"short_ema_span_0": deepcopy(bound)}
    original = deepcopy(config)
    converted = migrate(config, memory_span_minutes=123.5)
    public_span = ("entry", "ema_span_0_minutes") if kind == "trailing_martingale" else ("ema_span_0_minutes",)
    assert _at(converted.config, ("bot", "short", "strategy", kind, *public_span)) == 0.0
    assert _at(converted.config["optimize"]["bounds"], ("short", "strategy", kind, *public_span)) == bound
    assert converted.config["bot"]["short"]["unstuck"]["ema_span_0_minutes"] == fallback
    assert converted.config["optimize"]["bounds"]["short"]["unstuck"]["ema_span_0_minutes"] == [fallback, fallback]
    assert config == original
