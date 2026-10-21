from config.universe import set_side_universe_sources
import json
import pytest
from copy import deepcopy

from config_utils import (
    clean_config,
    dump_config,
    get_template_config,
    load_config,
    sanitize_prepared_config_for_dump,
    strip_config_metadata,
)


def test_clean_config_removes_internal_sections_and_keeps_user_values():
    template = get_template_config()
    config = {
        "bot": {
            "long": {"risk": {"max_positions": 5}, "_note": "keep me?"},
            "short": {"risk": {"max_positions": 3}},
        },
        "coin_overrides": {
            "BTC": {
                "_meta": "do not keep",
                "live": {"modes": {"long": "manual"}},
            }
        },
        "_transform_log": ["noise"],
        "_raw": {"bot": {}},
        "_raw_effective": {"bot": {"long": {"n_positions": 7}}},
    }

    cleaned = clean_config(deepcopy(config))

    assert "_transform_log" not in cleaned
    assert "_raw" not in cleaned
    assert "_raw_effective" not in cleaned
    assert cleaned["bot"]["long"]["risk"]["max_positions"] == 5
    assert cleaned["bot"]["short"]["risk"]["max_positions"] == 3
    assert (
        cleaned["bot"]["long"]["forager"]["volume_ema_span_minutes"]
        == template["bot"]["long"]["forager"]["volume_ema_span_minutes"]
    )
    assert (
        cleaned["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0_minutes"]
        == template["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
            "ema_span_0_minutes"
        ]
    )
    assert "BTC" in cleaned["coin_overrides"]
    assert "_meta" not in cleaned["coin_overrides"]["BTC"]
    assert (
        cleaned["coin_overrides"]["BTC"]["live"]["modes"]["long"]
        == config["coin_overrides"]["BTC"]["live"]["modes"]["long"]
    )


def test_clean_config_fills_missing_values_from_template():
    config = {}
    cleaned = clean_config(config)
    template = get_template_config()
    assert cleaned['bot']['global']['strategy_kind'] == template['bot']['global']['strategy_kind']
    assert cleaned["bot"]["long"]["strategy"] == {
        "trailing_martingale": template["bot"]["long"]["strategy"][
            "trailing_martingale"
        ]
    }
    assert cleaned["bot"]["short"]["strategy"] == {
        "trailing_martingale": template["bot"]["short"]["strategy"][
            "trailing_martingale"
        ]
    }
    from config.optimize_bounds import flatten_optimize_bounds
    from config.param_paths import iter_optimizer_key_paths
    from config import prepare_config

    bounds = flatten_optimize_bounds(cleaned["optimize"]["bounds"], strategy_kind="trailing_martingale")
    prepared = prepare_config(cleaned, verbose=False)
    for key, path in iter_optimizer_key_paths(prepared, bounds):
        effective = prepared
        for part in path:
            effective = effective[part]
        assert bounds[key] == [effective, effective]


def _mutable_container_ids(node):
    if isinstance(node, dict):
        return {id(node)}.union(*(_mutable_container_ids(value) for value in node.values()))
    if isinstance(node, list):
        return {id(node)}.union(*(_mutable_container_ids(value) for value in node))
    return set()


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("sparse", [False, True])
def test_clean_config_keeps_input_unchanged_and_detaches_all_output_containers(
    strategy, mode, sparse
):
    from config.hsl import generated_template

    source = generated_template(get_template_config(), mode)
    source['bot']['global']['strategy_kind'] = strategy
    set_side_universe_sources(source, 'approved_coins', {"long": ["BTC"], "short": ["ETH"]})
    source["live"]["startup"]["startup_phase_budgets"] = {"candles": 15.0}
    source["backtest"]["reducer"] = {"adg_strategy_eq": "max"}
    source["backtest"]["market_settings"]["overrides"] = {"BTC": {"min_qty": 0.001}}
    source["backtest"]["scenarios"] = [
        {"label": "stress", "overrides": {"bot.long.risk.max_positions": 2.0}}
    ]
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"risk": {"max_positions": 1.0}}}, "_note": ["private"]}
    }
    source["optimize"]["bounds"]["long"]["forager"]["score_weights"] = {
        "unilateralness": [0.0, 0.5]
    }
    if sparse:
        source["bot"]["global"]["hsl"] = {"scope": mode}
        source["bot"]["long"]["entry_cooldown"] = {"max_duration_minutes": 12.5}
        source["optimize"]["bounds"] = {
            "long": {"forager": {"score_weights": {"unilateralness": [0.0, 0.5]}}}
        }
    original = deepcopy(source)

    cleaned = clean_config(source)
    repeated = clean_config(source)

    assert source == original
    assert cleaned == repeated
    assert not (_mutable_container_ids(source) & _mutable_container_ids(cleaned))
    assert not (_mutable_container_ids(cleaned) & _mutable_container_ids(repeated))
    assert cleaned["backtest"]["reducer"] == {"adg_strategy_eq": "max", "default": "mean"}
    assert cleaned["live"]["startup"]["startup_phase_budgets"] == {"candles": 15.0}
    assert "_note" not in cleaned["coin_overrides"]["BTC"]
    assert cleaned["optimize"]["bounds"]["long"]["forager"]["score_weights"][
        "unilateralness"
    ] == [0.0, 0.5]
    assert set(cleaned["bot"]["long"]["strategy"]) == {strategy}
    if sparse and mode == "portfolio":
        assert cleaned["bot"]["global"]["hsl"] == {"scope": "portfolio"}

    saved_output = deepcopy(cleaned)
    source['bot']["long"]['universe']['approved_coins'].append("SOL")
    source["backtest"]["scenarios"][0]["overrides"]["bot.long.risk.max_positions"] = 3.0
    assert cleaned == saved_output

    changed_source = deepcopy(source)
    cleaned['bot']["short"]['universe']['approved_coins'].append("SOL")
    cleaned["optimize"]["bounds"]["long"]["forager"]["score_weights"][
        "unilateralness"
    ][0] = 0.1
    assert source == changed_source
    assert repeated == saved_output


@pytest.mark.parametrize(
    "path,value,error,match",
    [
        (("config_version",), "v99.0.0", ValueError, "newer than supported"),
        (
            ("optimize", "fixed_runtime_overrides"),
            {"live.hsl_engine": "legacy"},
            ValueError,
            "removed in v9",
        ),
        (
            ("coin_overrides",),
            {"BTC": {"bot": {"long": {"n_positions": 1}}}},
            ValueError,
            "retired",
        ),
        (("bot", "global", "hsl", "scope"), "unsupported", ValueError, "scope must be"),
        (
            ("optimize", "bounds", "global", "hsl"),
            [],
            TypeError,
            "optimize.bounds.global.hsl must be a mapping",
        ),
        (
            ("bot", "long", "forager", "score_weights", "volume"),
            -1.0,
            ValueError,
            "score_weights",
        ),
        (
            ("bot", "long", "entry_cooldown", "max_duration_minutes"),
            float("inf"),
            ValueError,
            "must be finite",
        ),
        (("bot", "global", "strategy_kind"), "unsupported", ValueError, "unsupported strategy kind"),
    ],
)
def test_clean_config_rejects_invalid_inputs_without_mutating_them(path, value, error, match):
    source = get_template_config()
    # Exercise validation after sparse adaptive bounds have been hydrated too.
    source["optimize"]["bounds"] = {}
    node = source
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value
    original = deepcopy(source)

    with pytest.raises(error, match=match):
        clean_config(source)

    assert source == original


def test_clean_config_prunes_inactive_strategy_subtrees_for_selected_kind():
    template = get_template_config()
    config = deepcopy(template)
    config['bot']['global']['strategy_kind'] = "ema_anchor"

    cleaned = clean_config(config)

    assert cleaned["bot"]["long"]["strategy"] == {
        "ema_anchor": template["bot"]["long"]["strategy"]["ema_anchor"]
    }
    assert cleaned["bot"]["short"]["strategy"] == {
        "ema_anchor": template["bot"]["short"]["strategy"]["ema_anchor"]
    }
    assert cleaned["optimize"]["bounds"]["long"]["strategy"] == {
        "ema_anchor": template["optimize"]["bounds"]["long"]["strategy"]["ema_anchor"]
    }
    assert cleaned["optimize"]["bounds"]["short"]["strategy"] == {
        "ema_anchor": template["optimize"]["bounds"]["short"]["strategy"]["ema_anchor"]
    }


def test_strip_config_metadata_removes_known_keys_recursively():
    config = {
        "bot": {
            "long": {"n_positions": 3, "_raw": {"ignore": True}},
            "_coins_sources": {"BTC": "binance"},
        },
        "_raw": {"bot": {}},
        "_raw_effective": {"bot": {"long": {"n_positions": 5}}},
        "_transform_log": ["load"],
        "nested": {"_raw": 123, "value": 5},
        "_coins_sources": {"ADA": "bybit"},
    }
    stripped = strip_config_metadata(config)
    assert "_raw" not in stripped
    assert "_raw_effective" not in stripped
    assert "_transform_log" not in stripped
    assert "_coins_sources" not in stripped
    assert "_coins_sources" not in stripped["bot"]
    assert "_raw" not in stripped["bot"]["long"]
    assert stripped["nested"]["value"] == 5


def test_sanitize_prepared_config_for_dump_removes_analysis_and_metadata():
    config = {
        "analysis": {"adg": 0.1},
        "_raw": {"bot": {}},
        "_raw_effective": {"bot": {}},
        "_transform_log": ["normalize"],
        "_coins_sources": {"approved_coins": "all"},
        "disable_plotting": "all",
        "backtest": {
            "cache_dir": "tmp/cache",
            "coins": {"binance": ["BTC"]},
        },
        "bot": {"long": {"risk": {"max_positions": 3}}, "short": {"risk": {"max_positions": 0}}},
    }

    sanitized = sanitize_prepared_config_for_dump(config)

    assert "analysis" not in sanitized
    assert "_raw" not in sanitized
    assert "_raw_effective" not in sanitized
    assert "_transform_log" not in sanitized
    assert "_coins_sources" not in sanitized
    assert "disable_plotting" not in sanitized
    assert "cache_dir" not in sanitized["backtest"]
    assert "coins" not in sanitized["backtest"]



def test_dump_config_clean_preserves_explicit_v8_reducer_conversion(tmp_path):
    from config import prepare_config
    from config.migrations.v9 import migrate_v8_to_v9
    source = get_template_config()
    source["config_version"] = "v8.6.0"
    source["backtest"].pop("reducer")
    source["backtest"]["aggregate"] = {"adg_pnl": "max"}
    before = deepcopy(source)
    cfg_path, out_path = tmp_path / "in.json", tmp_path / "out.json"
    cfg_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="explicit migration"):
        load_config(str(cfg_path), verbose=False)
    migrated = migrate_v8_to_v9(source, memory_span_minutes=43200.0)
    dump_config(prepare_config(migrated.config, verbose=False), str(out_path), clean=True)
    dumped = json.loads(out_path.read_text())
    assert source == before
    assert json.loads(cfg_path.read_text()) == before
    assert dumped["backtest"]["reducer"]["adg_pnl"] == "max"
    assert dumped["backtest"]["reducer"]["default"] == "mean"
    assert "aggregate" not in dumped["backtest"]
    assert "drawdown_worst_strategy_eq" not in dumped["backtest"]["reducer"]
    assert "position_held_days_max" not in dumped["backtest"]["reducer"]


def test_clean_config_preserves_backtest_reducer_overrides():
    config = {
        "backtest": {"reducer": {"adg_pnl": "max", "default": "mean"}},
    }

    cleaned = clean_config(config)

    assert cleaned["backtest"]["reducer"]["adg_pnl"] == "max"
    assert cleaned["backtest"]["reducer"]["default"] == "mean"
    assert "drawdown_worst_strategy_eq" not in cleaned["backtest"]["reducer"]
    assert "position_held_days_max" not in cleaned["backtest"]["reducer"]


def test_clean_config_preserves_sparse_backtest_reducer():
    config = {"backtest": {"reducer": {"default": "mean"}}}

    cleaned = clean_config(config)

    assert cleaned["backtest"]["reducer"] == {"default": "mean"}


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_clean_hsl_config_preserves_engine_policy_and_bounds(mode):
    from config.hsl import generated_template, REMOVED_FIELDS

    cfg = generated_template(get_template_config(), mode)
    cfg["bot"]["long"]["hsl"]["enabled"] = True
    if mode == "portfolio":
        cfg["bot"]["global"]["hsl"].update(enabled=True)
    original = deepcopy(cfg)
    cleaned = clean_config(cfg)
    assert cfg == original
    assert "hsl_engine" not in cleaned["live"]
    for side in ["long", "short"]:
        assert not REMOVED_FIELDS.intersection(cleaned["bot"][side]["hsl"])
        assert cleaned["bot"][side]["hsl"] == cfg["bot"][side]["hsl"]
    if mode == "portfolio":
        assert cleaned["bot"]["global"]["hsl"] == cfg["bot"]["global"]["hsl"]
        expected_bounds = {**cfg["optimize"]["bounds"]["global"]["hsl"],
                           "memory_ema_span_minutes": [cfg["bot"]["global"]["hsl"]["memory_ema_span_minutes"]] * 2}
        assert cleaned["optimize"]["bounds"]["global"]["hsl"] == expected_bounds
    else:
        assert "hsl" not in cleaned["bot"]
    assert clean_config(cleaned) == cleaned
    assert sanitize_prepared_config_for_dump(cfg) == cleaned



def test_clean_hsl_does_not_author_missing_portfolio_or_restart():
    from config.hsl import generated_template
    cfg = generated_template(get_template_config(), "portfolio")
    assert "restart_after_red_policy" not in clean_config(cfg)["bot"]["global"]["hsl"]
    cfg["bot"]["global"]["hsl"] = {"scope": "portfolio"}
    assert clean_config(cfg)["bot"]["global"]["hsl"] == {"scope": "portfolio"}


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_hsl_saved_fitness_contract_keeps_fixed_scope_policy(mode):
    from config.hsl import generated_template
    from optimization.evaluation_contract import build_evaluation_contract
    from optimize import _resume_config_mismatches
    from optimization.evaluation_contract import CONTRACT_KEY

    cfg = generated_template(get_template_config(), mode)
    contract = build_evaluation_contract(cfg)
    assert "hsl_engine" not in contract.get("live", {})
    block = (
        contract["bot"]["global"]["hsl"] if mode == "portfolio" else contract["bot"]["long"]["hsl"]
    )
    assert "restart_after_red_policy" not in block
    assert contract["version"] == 6
    assert "independent_loss_memories" in contract["risk_model"]
    assert contract["risk_model"].startswith("v9_")
    assert "drawdown_ratio_threshold" not in block  # candidate value, not fixed policy
    record = {**deepcopy(cfg), CONTRACT_KEY: contract}
    assert _resume_config_mismatches(record, cfg) == []
    candidate_block = (
        cfg["bot"]["global"]["hsl"] if mode == "portfolio" else cfg["bot"]["long"]["hsl"]
    )
    candidate_block["drawdown_ratio_threshold"] = 0.031
    assert _resume_config_mismatches(record, cfg) == []
    path = (
        "bot.global.hsl.panic_close_order_type"
        if mode == "portfolio"
        else "bot.long.hsl.panic_close_order_type"
    )
    cfg["optimize"]["fixed_runtime_overrides"][path] = "market"
    assert any(
        "evaluation.bot" in diff for diff in _resume_config_mismatches(record, cfg)
    )


def test_hsl_exported_config_can_be_prepared_without_legacy_hydration():
    from config.hsl import generated_template, REMOVED_FIELDS
    from config_utils import format_config

    cfg = generated_template(get_template_config(), "portfolio")
    cfg["bot"]["global"]["hsl"].update(
        enabled=True,
        drawdown_ratio_ema_span_minutes=12.5,
        drawdown_ratio_threshold=0.07,
    )
    cfg["optimize"]["bounds"]["global"]["hsl"] = {"drawdown_ratio_threshold": [0.02, 0.2]}
    exported = sanitize_prepared_config_for_dump(cfg)
    loaded = format_config(json.loads(json.dumps(exported)), verbose=False)
    assert loaded["bot"]["global"]["hsl"] == cfg["bot"]["global"]["hsl"]
    assert loaded["optimize"]["bounds"]["global"]["hsl"]["drawdown_ratio_threshold"] == [0.02, 0.2]
    assert not REMOVED_FIELDS.intersection(loaded["bot"]["long"]["hsl"])
