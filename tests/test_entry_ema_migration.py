from copy import deepcopy
import json

import pytest

from config import get_template_config, prepare_config, compile_runtime_config
from config.overrides import parse_overrides
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_bound_selectors, require_existing_config_path
from config_utils import clean_config
from optimizer_overrides import apply_coupled_unstuck_ema_spans
from warmup_utils import compute_per_coin_warmup_minutes

KIND = "trailing_martingale"
SPANS = ("ema_span_0", "ema_span_1")


def legacy_config():
    config = clean_config(get_template_config())
    config["config_version"] = "v8.3.0"
    for root in (config["bot"], config["optimize"]["bounds"]):
        for side in ("long", "short"):
            strategy = root[side]["strategy"][KIND]
            for key in SPANS:
                strategy[key] = strategy["entry"].pop(key)
    return config


def test_lossless_migration_runtime_warmup_and_roundtrip():
    old = legacy_config()
    old["bot"]["long"]["strategy"][KIND].update(ema_span_0=123.75, ema_span_1=998.125)
    old["optimize"]["bounds"]["long"]["strategy"][KIND]["ema_span_0"] = [
        12.5,
        321.75,
        0.25,
    ]
    prepared = prepare_config(old, verbose=False)
    strategy = prepared["bot"]["long"]["strategy"][KIND]
    assert strategy["entry"]["ema_span_0"] == 123.75
    assert strategy["entry"]["ema_span_1"] == 998.125
    assert not any(key in strategy for key in SPANS)
    assert prepared["bot"]["long"]["unstuck"] == old["bot"]["long"]["unstuck"]
    bounds = flatten_optimize_bounds(prepared["optimize"]["bounds"], strategy_kind=KIND)
    assert bounds["long_ema_span_0"] == [12.5, 321.75, 0.25]
    saved = clean_config(prepared)
    reloaded = prepare_config(saved, verbose=False)
    assert clean_config(reloaded) == saved
    assert (
        compile_runtime_config(reloaded)["bot"]
        == compile_runtime_config(prepared)["bot"]
    )
    assert compute_per_coin_warmup_minutes(reloaded) == compute_per_coin_warmup_minutes(
        prepared
    )


@pytest.mark.parametrize(
    "file_old,inline_old", [(True, False), (False, True), (True, True)]
)
def test_external_inline_precedence_and_partial_coupling(
    tmp_path, file_old, inline_old
):
    config = get_template_config()

    def patch(old, **values):
        return {
            "bot": {"long": {"strategy": {KIND: values if old else {"entry": values}}}}
        }

    source = tmp_path / "coin.json"
    source.write_text(json.dumps(patch(file_old, ema_span_0=41.25, ema_span_1=81.5)))
    config["coin_overrides"] = {
        "BTC": {
            "override_config_path": str(source),
            **patch(inline_old, ema_span_0=17.125),
        }
    }
    config = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    entry = config["coin_overrides"]["BTC"]["bot"]["long"]["strategy"][KIND]["entry"]
    assert entry == {"ema_span_0": 17.125, "ema_span_1": 81.5}
    config["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
    apply_coupled_unstuck_ema_spans(config)
    assert config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"] == entry


def test_conflicting_paths_warn_and_new_value_wins(caplog):
    config = get_template_config()
    strategy = config["bot"]["long"]["strategy"][KIND]
    strategy["ema_span_0"] = 17.5
    strategy["entry"]["ema_span_0"] = 123.5
    prepared = prepare_config(config, verbose=False)
    assert prepared["bot"]["long"]["strategy"][KIND]["entry"]["ema_span_0"] == 123.5
    assert "Cannot preserve conflicting EMA values" in caplog.text
    assert "explicit" in caplog.text and "wins" in caplog.text


def test_old_selectors_and_entry_group_select_the_same_genes():
    config = prepare_config(get_template_config(), verbose=False)
    bounds = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=KIND)
    for selector in [
        "long.strategy.ema_span_0",
        "bot.long.strategy.trailing_martingale.ema_span_0",
    ]:
        assert require_existing_config_path(config, selector)[-2:] == (
            "entry",
            "ema_span_0",
        )
        assert set(resolve_bound_selectors(config, [selector], bounds)) == {
            "long_ema_span_0"
        }
    selected = resolve_bound_selectors(config, ["long.strategy.entry"], bounds)
    assert {"long_ema_span_0", "long_ema_span_1"} <= set(selected)
    assert not any(key.startswith("long_close") for key in selected)


def test_nested_and_dotted_scenario_conflict_is_deterministic(caplog):
    config = legacy_config()
    config["backtest"]["scenarios"] = [
        {
            "label": "mixed",
            "overrides": {
                "bot": {"long": {"strategy": {KIND: {"ema_span_0": 17.5}}}},
                "bot.long.strategy.trailing_martingale.entry.ema_span_0": 51.25,
            },
        }
    ]
    prepared = prepare_config(config, verbose=False)
    overrides = prepared["backtest"]["scenarios"][0]["overrides"]
    assert overrides["bot.long.strategy.trailing_martingale.entry.ema_span_0"] == 51.25
    assert "bot.long.strategy.trailing_martingale.ema_span_0" not in overrides
    assert "conflicting scenario" in caplog.text


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("suffix", ["", "_underscores", "_short"])
def test_legacy_cli_paths_apply_to_canonical_entry(side, suffix):
    import argparse
    from config_utils import (
        add_config_arguments,
        create_acronym,
        update_config_with_args,
    )

    config = get_template_config()
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(
        parser, config, command="backtest", help_all=True, group_map={}
    )
    old = f"bot.{side}.strategy.trailing_martingale.ema_span_0"
    flag = "--" + old
    if suffix == "_underscores":
        flag = "--" + old.replace(".", "_")
    elif suffix == "_short":
        flag = "-" + create_acronym(old, set())
    args = parser.parse_args([flag, "123.75"])
    update_config_with_args(config, args, verbose=False, allowed_keys=allowed)
    prepared = prepare_config(config, verbose=False)
    assert prepared["bot"][side]["strategy"][KIND]["entry"]["ema_span_0"] == 123.75


@pytest.mark.parametrize("reverse", [False, True])
def test_new_scenario_alias_wins_over_old_fully_qualified_path(reverse, caplog):
    config = legacy_config()
    items = [
        ("long.strategy.entry.ema_span_0", 91.5),
        ("bot.long.strategy.trailing_martingale.ema_span_0", 17.5),
    ]
    if reverse:
        items.reverse()
    config["backtest"]["scenarios"] = [{"label": "alias", "overrides": dict(items)}]
    prepared = prepare_config(config, verbose=False)
    assert (
        prepared["backtest"]["scenarios"][0]["overrides"][
            "bot.long.strategy.trailing_martingale.entry.ema_span_0"
        ]
        == 91.5
    )
    assert "conflicting scenario" in caplog.text


def test_scenario_strategy_replacement_and_atomic_coin_patch_migrate():
    from config.migrations.entry_ema import migrate_entry_ema_spans

    config = legacy_config()
    config["backtest"]["scenarios"] = [
        {
            "label": "replace",
            "overrides": {
                "bot.long.strategy.trailing_martingale": {"ema_span_0": 71.25},
                "coin_overrides": {
                    "BTC": {
                        "bot": {"long": {"strategy": {KIND: {"ema_span_1": 33.125}}}}
                    }
                },
            },
        }
    ]
    migrate_entry_ema_spans(config)
    overrides = config["backtest"]["scenarios"][0]["overrides"]
    assert overrides["bot.long.strategy.trailing_martingale"] == {
        "entry": {"ema_span_0": 71.25}
    }
    assert overrides["coin_overrides"]["BTC"]["bot"]["long"]["strategy"][KIND] == {
        "entry": {"ema_span_1": 33.125}
    }


@pytest.mark.parametrize("kind", [KIND, "ema_anchor", "trailing_grid_v7"])
def test_legacy_wildcard_leaf_selector_preserves_active_strategy_genes(kind):
    config = get_template_config()
    config["live"]["strategy_kind"] = kind
    bounds = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=kind)
    selected = resolve_bound_selectors(config, ["*.strategy.*.ema_span_0"], bounds)
    assert set(selected) == {"long_ema_span_0", "short_ema_span_0"}
    for path in selected.values():
        assert ("entry" in path) == (kind == KIND)
