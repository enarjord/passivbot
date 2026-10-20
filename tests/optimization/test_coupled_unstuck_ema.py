from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from config_utils import clean_config, get_template_config
from config.load import prepare_config
from config.overrides import parse_overrides
from optimization.config_adapter import get_optimization_key_paths
from optimization.warmup import (
    _apply_config_overrides,
    _finalize_optimizer_vector_config,
    build_optimizer_vector_config,
    compute_optimizer_per_coin_warmup_minutes,
)
from optimizer_overrides import apply_coupled_unstuck_ema_spans


def config_for(kind="trailing_martingale"):
    config = get_template_config()
    config["live"]["strategy_kind"] = kind
    config["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
    config = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    for side in ("long", "short"):
        (
            config["bot"][side]["strategy"][kind]["entry"]
            if kind == "trailing_martingale"
            else config["bot"][side]["strategy"][kind]
        ).update(ema_span_0=17.25, ema_span_1=211.75)
        config["bot"][side]["unstuck"].update(ema_span_0=9999.5, ema_span_1=8888.5)
    return config


@pytest.mark.parametrize(
    "kind", ["trailing_martingale", "ema_anchor"]
)
def test_coupled_candidates_drop_redundant_genes_and_materialize_effective_coin_spans(
    kind,
):
    config = config_for(kind)
    config["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "strategy": {
                        kind: (
                            {"entry": {"ema_span_0": 71.5}}
                            if kind == "trailing_martingale"
                            else {"ema_span_0": 71.5}
                        )
                    },
                    "unstuck": {"ema_span_0": 1.0, "ema_span_1": 2.0},
                }
            }
        },
        "ETH": {"bot": {"short": {"unstuck": {"ema_span_0": 3.0}}}},
    }
    original = deepcopy(config)
    paths = get_optimization_key_paths(config)
    assert not any("unstuck_ema_span" in key for key, _ in paths)
    vector = []
    for key, path in paths:
        value = config
        for part in path:
            value = value[part]
        vector.append(101.25 if key == "long_ema_span_1" else value)
    candidate = build_optimizer_vector_config(vector, config, key_paths=paths)
    assert config == original
    assert candidate["bot"]["long"]["unstuck"]["ema_span_1"] == 101.25
    btc = candidate["coin_overrides"]["BTC"]["bot"]
    assert btc["long"]["unstuck"]["ema_span_0"] == 71.5
    assert btc["long"]["unstuck"]["ema_span_1"] == 101.25
    assert btc["short"]["unstuck"]["ema_span_0"] == 17.25
    # Reopening a saved candidate with optimizer overrides disabled preserves behavior.
    saved = clean_config(candidate)
    saved["optimize"]["enable_overrides"] = []
    loaded = parse_overrides(prepare_config(saved, verbose=False), verbose=False)
    assert (
        loaded["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]
        == btc["long"]["unstuck"]
    )


def test_independent_default_and_mirror_fixed_override_order():
    config = config_for()
    config["optimize"]["enable_overrides"] = []
    assert any(
        "unstuck_ema_span" in key for key, _ in get_optimization_key_paths(config)
    )
    independent = _finalize_optimizer_vector_config(deepcopy(config))
    assert independent["bot"]["long"]["unstuck"]["ema_span_0"] == 9999.5
    config["optimize"]["enable_overrides"] = [
        "couple_unstuck_ema_spans",
        "mirror_short_from_long",
    ]
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.strategy.trailing_martingale.entry.ema_span_0": 61.25,
        "bot.short.unstuck.ema_span_0": 1.5,
    }
    candidate = _finalize_optimizer_vector_config(deepcopy(config))
    for side in ("long", "short"):
        assert candidate["bot"][side]["unstuck"]["ema_span_0"] == 61.25
        assert "unstuck_ema_span_0" not in candidate["bot"][side]


def test_coupling_materializes_scenario_dependencies_for_plain_saved_suite_replay():
    config = config_for()
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 4.5}}}}
    }
    config["backtest"]["scenarios"] = [
        {
            "label": "alternate",
            "overrides": {
                "bot.long.strategy.trailing_martingale.entry.ema_span_0": 301.25,
                "bot.long.unstuck.ema_span_0": 3.0,
                "coin_overrides.BTC.bot.long.unstuck.ema_span_0": 5.0,
            },
        }
    ]
    finalized = _finalize_optimizer_vector_config(deepcopy(config))
    replay = clean_config(finalized)
    replay["optimize"]["enable_overrides"] = []
    _apply_config_overrides(replay, replay["backtest"]["scenarios"][0]["overrides"])
    assert replay["bot"]["long"]["unstuck"]["ema_span_0"] == 301.25
    assert (
        replay["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_0"]
        == 301.25
    )
    assert config["bot"]["long"]["unstuck"]["ema_span_0"] == 9999.5


def test_coupling_contract_ignores_derived_coin_values_but_tracks_mode_and_sources():
    from optimization.evaluation_contract import build_evaluation_contract

    config = config_for()
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 4.5}}}}
    }
    first = build_evaluation_contract(config)
    config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0"
    ] = 501.25
    assert build_evaluation_contract(config) == first
    config["coin_overrides"]["BTC"]["bot"]["long"]["strategy"] = {
        "trailing_martingale": {"entry": {"ema_span_0": 71.5}}
    }
    assert build_evaluation_contract(config) != first
    config["optimize"]["enable_overrides"] = []
    assert build_evaluation_contract(config)["coupled_unstuck_ema_spans"] is False


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
def test_coupled_suite_resume_compares_saved_candidate_dependencies(kind):
    from optimize import _resume_config_mismatches
    from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract

    config = config_for(kind)
    strategy_path = f"strategy.{kind}" + (".entry" if kind == "trailing_martingale" else "")
    strategy = {"ema_span_0": 71.5}
    if kind == "trailing_martingale":
        strategy = {"entry": strategy}
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"strategy": {kind: strategy}}}},
        "ETH": {"bot": {"long": {"unstuck": {"close_pct": 0.05}}}},
    }
    config["backtest"].update(suite_enabled=True, scenarios=[
        {"label": "base", "coins": ["BTC", "ETH"], "start_date": "2024-01-01"},
        {"label": "alternate", "overrides": {
            f"bot.long.{strategy_path}.ema_span_1": 301.25,
            "coin_overrides.ETH.bot.long.unstuck.close_pct": 0.075,
        }},
    ])
    saved = deepcopy(config)
    spans = saved["bot"]["long"]["strategy"][kind]
    if kind == "trailing_martingale":
        spans = spans["entry"]
    spans["ema_span_0"] = 20.0
    entry = clean_config(_finalize_optimizer_vector_config(saved))
    entry["suite_metrics"] = {"scenario_labels": ["base", "alternate"]}
    entry[CONTRACT_KEY] = build_evaluation_contract(config)
    before = deepcopy(config)
    assert entry["backtest"]["scenarios"] != config["backtest"]["scenarios"]
    assert _resume_config_mismatches(entry, config) == []
    assert config == before

    changes = [
        (0, "coins", ["BTC"]),
        (0, "start_date", "2024-01-02"),
        (1, "overrides", {f"bot.long.{strategy_path}.ema_span_1": 401.25}),
        (1, "overrides", {f"coin_overrides.BTC.bot.long.{strategy_path}.ema_span_0": 80.5}),
        (1, "overrides", {"coin_overrides.ETH.bot.long.unstuck.close_pct": 0.08}),
    ]
    for index, key, value in changes:
        changed = deepcopy(config)
        if key == "overrides":
            changed["backtest"]["scenarios"][index][key].update(value)
        else:
            changed["backtest"]["scenarios"][index][key] = value
        assert any("backtest.scenarios" in item
                   for item in _resume_config_mismatches(entry, changed))

    corrupted = deepcopy(entry)
    corrupted["backtest"]["scenarios"][0]["overrides"]["bot.long.unstuck.ema_span_0"] = 99.0
    assert any("backtest.scenarios" in item
               for item in _resume_config_mismatches(corrupted, config))
    changed = deepcopy(config)
    changed["optimize"]["enable_overrides"] = []
    assert any("enable_overrides" in item
               for item in _resume_config_mismatches(entry, changed))


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
def test_gpu_candidate_packing_couples_after_candidate_and_exact_coin_values(
    kind
):
    from optimization.gpu import model, service

    cls = service.SharedAccountReplay
    proxy = cls.__new__(cls)
    proxy.hsl_signal_mode = "coin"

    prefix = "EMA_ANCHOR" if kind == "ema_anchor" else "TRAILING_MARTINGALE"
    keys = getattr(
        model,
        prefix + "_MULTICOIN_PARAM_KEYS",
    )
    proxy.param_keys = keys
    proxy.sides = ["long", "short"]
    proxy.couple_unstuck_emas = True
    proxy.base_params = {side: {key: 1.0 for key in keys} for side in proxy.sides}
    proxy.static_coin_override_params = {
        "long": {"ema_span_0": 71.5, "unstuck_ema_span_0": 4.0}
    }
    candidates = [
        {
            "long_ema_span_0": 17.25,
            "long_ema_span_1": 211.75,
            "long_unstuck_ema_span_1": 9.0,
        }
    ]
    matrix = proxy._parameter_matrix(candidates, side="long")
    assert matrix[0, keys.index("unstuck_ema_span_0")] == 17.25
    assert matrix[0, keys.index("unstuck_ema_span_1")] == 211.75


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_coin_packing_preserves_dependency_when_only_one_strategy_span_is_pinned(
    kind, side
):
    from optimization.gpu import model

    prefix = "EMA_ANCHOR" if kind == "ema_anchor" else "TRAILING_MARTINGALE"
    strategy_keys = (
        model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS
        if kind == "ema_anchor"
        else tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
    )
    config = config_for(kind)
    config["coin_overrides"] = {
        "BTC": {
            "bot": {
                side: {
                    "strategy": {
                        kind: (
                            {"entry": {"ema_span_0": 71.5}}
                            if kind == "trailing_martingale"
                            else {"ema_span_0": 71.5}
                        )
                    },
                    "unstuck": {"ema_span_0": 4.0, "ema_span_1": 5.0},
                }
            }
        }
    }
    strategy = deepcopy(config["bot"][side]["strategy"][kind])
    (strategy["entry"] if kind == "trailing_martingale" else strategy)[
        "ema_span_0"
    ] = 71.5
    payload = SimpleNamespace(
        backtest_params={
            "coins": ["BTC"],
            "dynamic_wel_by_tradability": True,
            "unstuck_memory_ema_span_minutes": config["bot"]["global"]["unstuck"]["memory_ema_span_minutes"],
            "loss_gate_memory_ema_span_minutes": config["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"],
            "equity_hard_stop_loss": {
                "engine": "hsl",
                "mode": "coin",
                "coins": {"BTC": [config["bot"][s]["hsl"] for s in ("long", "short")]},
            },
        },
        strategy_params_list=[{side: strategy}],
        bot_params_list=[
            {
                side: {
                    "risk_entry_cooldown_minutes": 0,
                    "total_wallet_exposure_limit": 1,
                    "n_positions": 1,
                }
            }
        ],
    )
    from optimization.gpu.coin_parameters import build_coin_override_parameters

    matrix, _ = build_coin_override_parameters(
        strategy_kind=kind,
        config=config,
        mss={"BTC": {}},
        exchange="bybit",
        coins=["BTC"],
        payload=payload,
        side=side,
        resolve_override=lambda config, mss, exchange, coin: config["coin_overrides"][
            coin
        ],
    )
    start = getattr(model, f"{prefix}_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN")
    assert matrix[0, start] == 71.5
    assert np.isnan(matrix[0, start + 1])
    assert np.isnan(matrix[0, start + 2 :]).all()


def test_coupled_warmup_tracks_strategy_bounds_and_ignores_migrated_unstuck_pins():
    config = config_for()
    config["live"].update(warmup_ratio=1.0, max_warmup_minutes=1_000_000)
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 900_000.5}}}}
    }
    config["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0"
    ] = [17.25, 80_000.5]
    coupled = compute_optimizer_per_coin_warmup_minutes(config)
    assert coupled["BTC"] >= 80_001
    assert coupled["BTC"] < 900_000


def test_anchored_search_derives_spans_from_selected_anchor_and_tunable_strategy():
    from optimize import install_anchored_fine_tune_plan
    from optimization.shape import build_optimization_shape

    config = config_for()
    config["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0"
    ] = [1.0, 1000.0]
    config["optimize"]["round_to_n_significant_digits"] = 6
    anchors = [deepcopy(config), deepcopy(config)]
    anchors[0]["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0"
    ] = 101.25
    anchors[1]["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0"
    ] = 401.75
    install_anchored_fine_tune_plan(
        config, ["bot.long.strategy.trailing_martingale.entry.ema_span_1"],
        "<memory>", starting_configs_override=anchors
    )
    shape = build_optimization_shape(config)
    assert [key for key, _ in shape.key_paths] == ["__anchor_id__", "long_ema_span_1"]
    candidate = build_optimizer_vector_config([1.0, 311.5], config)
    assert candidate["bot"]["long"]["unstuck"]["ema_span_0"] == 401.75
    assert candidate["bot"]["long"]["unstuck"]["ema_span_1"] == 311.5


def test_exact_suite_finalization_couples_after_context_overrides_without_mutating_candidate():
    from optimize import SuiteEvaluator

    config = config_for()
    config["backtest"].update(coins={"bybit": ["BTC"]}, cache_dir={})
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 3.0}}}}
    }
    before = deepcopy(config)
    ctx = SimpleNamespace(
        config=deepcopy(config),
        overrides={
            "bot.long.strategy.trailing_martingale.entry.ema_span_0": 401.5,
            "bot.long.unstuck.ema_span_0": 4.5,
        },
    )
    evaluator = SuiteEvaluator.__new__(SuiteEvaluator)
    evaluator.base = SimpleNamespace(scoring_specs=[], limit_checks=[])
    evaluator.objective_bases = []
    ctx.msss = {}
    actual = evaluator.build_scenario_candidate_config(config, ctx)
    assert actual["bot"]["long"]["unstuck"]["ema_span_0"] == 401.5
    assert (
        actual["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_0"] == 401.5
    )
    assert config == before


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_coupling_fails_explicitly_for_invalid_strategy_spans(bad):
    config = config_for()
    config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "ema_span_0"
    ] = bad
    with pytest.raises(
        ValueError, match="couple_unstuck_ema_spans requires positive finite"
    ):
        _finalize_optimizer_vector_config(config)


def test_scenario_override_file_is_resolved_before_materializing_coupled_spans(
    tmp_path,
):
    import json

    config = config_for()
    file_config = {"bot": {"long": {"strategy": {"trailing_martingale": {
        "entry": {"ema_span_0": 71.25, "ema_span_1": 99.5},
    }}}}}
    path = tmp_path / "coin.json"
    path.write_text(json.dumps(file_config))
    config["backtest"]["scenarios"] = [
        {
            "label": "file",
            "overrides": {
                "coin_overrides": {
                    "BTC": {
                        "override_config_path": str(path),
                        "bot": {
                            "long": {
                                "strategy": {
                                    "trailing_martingale": {"entry": {"ema_span_1": 211.5}}
                                }
                            }
                        },
                    }
                }
            },
        }
    ]
    candidate = _finalize_optimizer_vector_config(config)
    saved = candidate["backtest"]["scenarios"][0]["overrides"]["coin_overrides"]["BTC"]
    assert "override_config_path" not in saved
    assert saved["bot"]["long"]["unstuck"]["ema_span_0"] == 71.25
    assert saved["bot"]["long"]["unstuck"]["ema_span_1"] == 211.5


def test_coupled_saved_scenario_preserves_explicit_coin_override_clear():
    config = config_for()
    config["coin_overrides"] = {
        "BTC": {
            "bot": {"long": {"strategy": {"trailing_martingale": {"entry": {"ema_span_0": 71.5}}}}}
        }
    }
    config["backtest"]["scenarios"] = [
        {"label": "clear", "overrides": {"coin_overrides": {}}}
    ]
    candidate = _finalize_optimizer_vector_config(config)
    overrides = candidate["backtest"]["scenarios"][0]["overrides"]
    assert overrides["coin_overrides"] == {}
    _apply_config_overrides(candidate, overrides)
    assert candidate["coin_overrides"] == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "bot": {
                "long": {
                    "strategy": {"trailing_martingale": {"entry": {"ema_span_0": 71.25}}},
                    "unstuck": {"ema_span_0": 3.0},
                }
            }
        },
        {"bot.long.strategy.trailing_martingale.entry.ema_span_0": 71.25,
         "bot.long.unstuck.ema_span_0": 3.0},
    ],
)
def test_coupled_saved_scenarios_preserve_nested_and_dotted_canonical_paths(overrides):
    import json

    config = config_for()
    config["backtest"]["scenarios"] = [{"label": "canonical", "overrides": overrides}]
    candidate = _finalize_optimizer_vector_config(config)
    saved = json.loads(json.dumps(clean_config(candidate), sort_keys=True))
    saved["optimize"]["enable_overrides"] = []
    _apply_config_overrides(saved, saved["backtest"]["scenarios"][0]["overrides"])
    assert (
        saved["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"]
        == 71.25
    )
    assert saved["bot"]["long"]["unstuck"]["ema_span_0"] == 71.25


@pytest.mark.parametrize("path", [
    "long.ema_span_0",
    "bot.long.ema_span_0",
    "bot.long.strategy.trailing_martingale.ema_span_0",
])
def test_coupled_scenario_rejects_retired_authored_strategy_paths(path):
    config = clean_config(config_for())
    config["backtest"]["scenarios"] = [{"label": "retired", "overrides": {path: 71.25}}]
    with pytest.raises(ValueError, match="unknown|canonical"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_finalization_keeps_file_coupling_and_repeated_candidates_canonical(tmp_path, kind):
    import json
    from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH

    config = config_for(kind)
    strategy_suffix = f"strategy.{kind}" + (".entry" if kind == "trailing_martingale" else "")
    file_strategy = {"ema_span_0": 71.25, "ema_span_1": 99.5}
    inline_strategy = {"ema_span_1": 211.5}
    if kind == "trailing_martingale":
        file_strategy = {"entry": file_strategy}
        inline_strategy = {"entry": inline_strategy}
    path = tmp_path / "canonical-coin.json"
    path.write_text(json.dumps({"bot": {"long": {
        "strategy": {kind: file_strategy},
        "entry_cooldown": {"base_duration_minutes": 3.5},
        "unstuck": {"ema_span_0": 4.5},
    }}}))
    source_bytes = path.read_bytes()
    config["coin_overrides"] = {"BTC": {
        "override_config_path": str(path),
        "bot": {"long": {"strategy": {kind: inline_strategy}}},
    }}
    config["bot"]["long"]["forager"]["score_weights"] = {
        "volume": 1, "ema_readiness": 3, "volatility": 0, "unilateralness": 0,
    }
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long", "couple_unstuck_ema_spans"]
    config["optimize"]["fixed_runtime_overrides"][f"bot.long.{strategy_suffix}.ema_span_0"] = 31.25
    config["backtest"]["scenarios"] = [{"label": "canonical", "overrides": {
        f"bot.long.{strategy_suffix}.ema_span_1": 301.25,
        f"coin_overrides.BTC.bot.long.{strategy_suffix}.ema_span_0": 81.25,
    }}]
    before = deepcopy(config)
    candidate = _finalize_optimizer_vector_config(deepcopy(config))
    assert config == before and path.read_bytes() == source_bytes
    for side in ("long", "short"):
        assert candidate["bot"][side]["unstuck"]["ema_span_0"] == 31.25
        assert not (set(candidate["bot"][side]) & set(FLAT_BOT_KEY_TO_GROUP_PATH))
        assert candidate["bot"][side]["forager"]["score_weights"] == {
            "volume": .25, "ema_readiness": .75, "volatility": 0.0, "unilateralness": 0.0,
        }
    patch = candidate["coin_overrides"]["BTC"]
    assert "override_config_path" not in patch
    assert patch["bot"]["long"]["entry_cooldown"]["base_duration_minutes"] == 3.5
    assert patch["bot"]["long"]["unstuck"]["ema_span_0"] == 71.25
    assert patch["bot"]["long"]["unstuck"]["ema_span_1"] == 211.5
    scenario = candidate["backtest"]["scenarios"][0]["overrides"]
    assert scenario["bot.long.unstuck.ema_span_1"] == 301.25
    assert scenario["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_0"] == 81.25
    assert scenario["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_1"] == 211.5
    saved = clean_config(candidate)
    repeated = _finalize_optimizer_vector_config(deepcopy(candidate))
    assert clean_config(repeated) == saved
    for side in ("long", "short"):
        assert not (set(repeated["bot"][side]) & set(FLAT_BOT_KEY_TO_GROUP_PATH))
    assert path.read_bytes() == source_bytes


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("anchored", [False, True])
def test_candidate_genes_pins_and_mirroring_win_over_stale_runtime_aliases(kind, anchored):
    from config.shared_bot import FLAT_BOT_KEY_TO_GROUP_PATH
    from optimize import install_anchored_fine_tune_plan

    config = config_for(kind)
    config["optimize"]["enable_overrides"] = [
        "mirror_short_from_long", "couple_unstuck_ema_spans",
    ]
    config["optimize"]["bounds"]["long"]["hsl"]["red_threshold"] = [0.1, 0.5, 0.01]
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.hsl.cooldown_minutes_after_red": 1.0,
    }
    config["bot"]["long"]["forager"]["score_weights"] = {
        "volume": 1.0, "ema_readiness": 3.0, "volatility": 0.0, "unilateralness": 0.0,
    }
    vector = [0.35]
    if anchored:
        install_anchored_fine_tune_plan(
            config, ["bot.long.hsl.red_threshold"], "<memory>",
            starting_configs_override=[deepcopy(config)],
        )
        vector = [0.0, *vector]
    # Derived aliases can survive in internal templates after authored ingestion.
    for side in ("long", "short"):
        config["bot"][side]["hsl_red_threshold"] = 0.15
        config["bot"][side]["hsl_cooldown_minutes_after_red"] = 0.3
        config["bot"][side]["forager_score_weights"] = {
            "volume": 0.9, "ema_readiness": 0.05, "volatility": 0.05,
        }
    before = deepcopy(config)

    candidate = build_optimizer_vector_config(
        vector, config,
        key_paths=[("long_hsl_red_threshold", ("bot", "long", "hsl", "red_threshold"))],
    )

    assert config == before
    for side in ("long", "short"):
        assert candidate["bot"][side]["hsl"]["red_threshold"] == pytest.approx(0.35)
        assert candidate["bot"][side]["hsl"]["cooldown_minutes_after_red"] == 1.0
        assert candidate["bot"][side]["forager"]["score_weights"] == {
            "volume": 0.25, "ema_readiness": 0.75, "volatility": 0.0, "unilateralness": 0.0,
        }
        assert not (set(candidate["bot"][side]) & set(FLAT_BOT_KEY_TO_GROUP_PATH))
    assert _finalize_optimizer_vector_config(deepcopy(candidate)) == candidate


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("origin", ["root", "external"])
def test_authored_shared_aliases_remain_rejected_before_optimizer_finalization(tmp_path, kind, origin):
    import json

    config = clean_config(config_for(kind))
    if origin == "root":
        config["bot"]["long"]["n_positions"] = 2
        error = r"bot\.long\.n_positions is retired"
    else:
        path = tmp_path / "retired-coin.json"
        path.write_text(json.dumps({"bot": {"long": {"n_positions": 2}}}))
        source_bytes = path.read_bytes()
        config["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
        error = r"coin_overrides\.BTC\.override_config_path.*bot\.long\.n_positions is retired"
    before = deepcopy(config)
    with pytest.raises(ValueError, match=error):
        parse_overrides(prepare_config(config, verbose=False), verbose=False)
    assert config == before
    if origin == "external":
        assert path.read_bytes() == source_bytes
