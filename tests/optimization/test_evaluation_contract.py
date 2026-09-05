from copy import deepcopy
import json

import pytest

from config_utils import clean_config, get_template_config
from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY


def _config():
    config = clean_config(get_template_config())
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-10"
    return config


def _record(config):
    return {**deepcopy(config), CONTRACT_KEY: build_evaluation_contract(config)}


@pytest.mark.parametrize(
    "key,value",
    [
        ("hsl_signal_mode", "pside"),
        ("hedge_mode", True),
        ("market_orders_allowed", True),
        ("market_order_near_touch_threshold", 0.03),
        ("max_realized_loss_pct", 0.17),
        ("pnls_max_lookback_days", 17),
        ("warmup_ratio", 0.7),
        ("max_warmup_minutes", 200),
        ("minimum_coin_age_days", 17),
        ("forager_score_hysteresis_pct", 0.07),
    ],
)
def test_resume_rejects_changed_fixed_live_simulation_inputs(key, value):
    from optimize import _resume_config_mismatches

    config = _config()
    old = _record(config)
    assert config["live"][key] != value
    config["live"][key] = value
    assert any(
        "evaluation.live" in item for item in _resume_config_mismatches(old, config)
    )


def test_resume_ignores_candidate_values_but_rejects_fixed_bot_policy():
    from optimize import _resume_config_mismatches

    config = _config()
    old = deepcopy(config)  # Legacy ordinary artifacts remain supported.
    old["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 37.0
    assert _resume_config_mismatches(old, config) == []
    config["bot"]["long"]["hsl"]["enabled"] = not config["bot"]["long"]["hsl"][
        "enabled"
    ]
    assert any(
        "evaluation.bot" in item for item in _resume_config_mismatches(old, config)
    )


def test_resume_compares_resolved_coin_override_values_not_file_paths(tmp_path):
    from optimize import _resume_config_mismatches

    config = _config()
    config["coin_overrides"] = {
        "BTC": {"bot": {"long": {"risk": {"entry_cooldown_minutes": 5.0}}}}
    }
    old = _record(config)
    path = tmp_path / "override.json"
    path.write_text(
        json.dumps({"bot": {"long": {"risk": {"entry_cooldown_minutes": 5.0}}}})
    )
    config["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
    assert _resume_config_mismatches(old, config) == []
    path.write_text(
        json.dumps({"bot": {"long": {"risk": {"entry_cooldown_minutes": 9.0}}}})
    )
    assert any(
        "evaluation.coin_overrides" in item
        for item in _resume_config_mismatches(old, config)
    )


def test_resume_ignores_machine_settings_and_overridden_source_values():
    from optimize import _resume_config_mismatches

    config = _config()
    config["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": False}
    old = _record(config)
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["live"]["execution_delay_seconds"] = 17
    config["optimize"]["n_cpus"] += 1
    assert _resume_config_mismatches(old, config) == []


def test_resume_validates_all_anchor_fixed_values_and_rejects_unprovable_legacy():
    from optimize import _resume_config_mismatches

    config = _config()
    path = ["bot", "long", "strategy", "trailing_martingale", "entry_initial_ema_dist"]
    config[ANCHOR_PLAN_KEY] = {
        "anchors": [
            {"source": "first.json", "fixed_values": [{"path": path, "value": 0.01}]},
            {"source": "second.json", "fixed_values": [{"path": path, "value": 0.02}]},
        ],
        "key_paths": [["bot", "long", "risk", "entry_cooldown_minutes"]],
    }
    old = _record(config)
    old["optimizer_anchor"] = {"id": 0}
    old.pop(ANCHOR_PLAN_KEY)
    assert _resume_config_mismatches(old, config) == []
    config[ANCHOR_PLAN_KEY]["anchors"][1]["source"] = "moved.json"
    assert _resume_config_mismatches(old, config) == []
    config[ANCHOR_PLAN_KEY]["anchors"][1]["fixed_values"][0]["value"] = 0.03
    assert any(
        "evaluation.anchors" in item for item in _resume_config_mismatches(old, config)
    )
    old.pop(CONTRACT_KEY)
    assert any(
        "legacy anchored" in item for item in _resume_config_mismatches(old, config)
    )


def test_changed_contract_blocks_checkpoint_fitness_reuse(tmp_path):
    import msgpack
    from optimize import _validate_resume_results

    config = _config()
    old = _record(config)
    old["metrics"] = {"objectives": {"w_0": -1.0}}
    (tmp_path / "all_results.bin").write_bytes(msgpack.packb(old, use_bin_type=True))
    config["live"]["hsl_signal_mode"] = "pside"
    with pytest.raises(ValueError, match="critical parameters have changed"):
        _validate_resume_results(str(tmp_path), config)


@pytest.mark.parametrize("backend", ["deap", "pymoo"])
def test_result_writers_persist_contract_before_candidate_projection(
    backend, monkeypatch
):
    from types import SimpleNamespace
    import optimize
    from optimization.callback import build_pymoo_record_entry

    config = _config()
    candidate = deepcopy(config)
    candidate["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 37.0
    build = lambda *args, **kwargs: deepcopy(candidate)
    if backend == "pymoo":
        entry = build_pymoo_record_entry(
            vector=[37.0],
            metrics={},
            template=config,
            build_config_fn=build,
            overrides_fn=None,
        )
    else:
        monkeypatch.setattr(optimize, "individual_to_config", build)
        entries = []
        optimize._record_individual_result(
            SimpleNamespace(evaluation_metrics={}),
            config,
            [],
            SimpleNamespace(record=entries.append),
        )
        entry = entries[0]
    assert entry[CONTRACT_KEY] == build_evaluation_contract(config)
    assert optimize._resume_config_mismatches(entry, config) == []


def test_added_fixed_backtest_settings_do_not_evade_legacy_comparison():
    from optimize import _resume_config_mismatches

    old = _config()
    old["backtest"].pop("market_settings")
    new = deepcopy(old)
    new["backtest"]["market_settings"] = {"overrides": {"BTC": {"price_step": 0.1}}}
    assert any(
        "backtest.market_settings" in item
        for item in _resume_config_mismatches(old, new)
    )


@pytest.mark.parametrize("snapshot", ["invalid", {"version": 0}])
def test_resume_rejects_unrecognized_contract_snapshot(snapshot):
    from optimize import _resume_config_mismatches

    config = _config()
    old = {**deepcopy(config), CONTRACT_KEY: snapshot}
    assert any(
        "unsupported or malformed" in item
        for item in _resume_config_mismatches(old, config)
    )


def test_new_fixed_runtime_override_cannot_evade_legacy_key_comparison():
    from optimize import _resume_config_mismatches

    old = _config()
    old["optimize"].pop("fixed_runtime_overrides", None)
    new = deepcopy(old)
    new["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.risk.entry_cooldown_minutes": 17.0,
    }
    assert any(
        "optimize.fixed_runtime_overrides" in item
        for item in _resume_config_mismatches(old, new)
    )


def test_real_anchored_candidate_record_preserves_other_anchor_policy():
    from optimization.callback import build_pymoo_record_entry
    from optimize import _resume_config_mismatches, individual_to_config
    from optimizer_overrides import optimizer_overrides

    config = _config()
    fixed_path = [
        "bot",
        "long",
        "strategy",
        "trailing_martingale",
        "entry",
        "threshold_base_pct",
    ]
    config[ANCHOR_PLAN_KEY] = {
        "anchors": [
            {"fixed_values": [{"path": fixed_path, "value": 0.01}]},
            {"fixed_values": [{"path": fixed_path, "value": 0.02}]},
        ],
        "key_paths": [["bot", "long", "risk", "entry_cooldown_minutes"]],
        "tunable_keys": ["long_entry_cooldown_minutes"],
    }
    entry = build_pymoo_record_entry(
        vector=[1.0, 37.0],
        metrics={},
        template=config,
        build_config_fn=individual_to_config,
        overrides_fn=optimizer_overrides,
    )
    assert entry["optimizer_anchor"]["id"] == 1
    assert (
        entry["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
            "threshold_base_pct"
        ]
        == 0.02
    )
    assert entry["bot"]["long"]["risk"]["entry_cooldown_minutes"] == 37.0
    assert _resume_config_mismatches(entry, config) == []
    config[ANCHOR_PLAN_KEY]["anchors"][0]["fixed_values"][0]["value"] = 0.03
    assert any(
        "evaluation.anchors" in item
        for item in _resume_config_mismatches(entry, config)
    )


@pytest.mark.parametrize(
    "key,old_value,new_value",
    [
        ("market_settings_sources", {"BTC": "binance"}, {"BTC": "bybit"}),
        ("ohlcv_source_dir", "dataset-a", "dataset-b"),
        ("hlcvs_data_dir", "prepared-a", "prepared-b"),
    ],
)
@pytest.mark.parametrize("record_snapshot", [False, True])
def test_resume_rejects_changed_data_source_selectors(
    tmp_path, key, old_value, new_value, record_snapshot
):
    import msgpack
    from optimize import _resume_config_mismatches, _validate_resume_results

    config = _config()
    config["backtest"][key] = old_value
    old = _record(config) if record_snapshot else deepcopy(config)
    assert _resume_config_mismatches(old, config) == []
    old["metrics"] = {"objectives": {"w_0": -1.0}}
    (tmp_path / "all_results.bin").write_bytes(msgpack.packb(old, use_bin_type=True))
    config["backtest"][key] = new_value
    assert any(key in item for item in _resume_config_mismatches(old, config))
    with pytest.raises(ValueError, match="critical parameters have changed"):
        _validate_resume_results(str(tmp_path), config)


def test_gpu_runtime_settings_retain_documented_strict_resume_comparison():
    from optimize import _resume_config_mismatches

    config = _config()
    config["optimize"]["backend"] = "gpu"
    old = _record(config)
    config["optimize"]["gpu"]["exact_workers"] = 987
    assert any("gpu" in item for item in _resume_config_mismatches(old, config))


@pytest.mark.parametrize("backend", ["deap", "pymoo"])
@pytest.mark.parametrize("token", ["now", "today", "", None])
def test_prepared_dynamic_end_date_rollover_rejects_saved_fitness(
    monkeypatch, backend, token
):
    import utils
    from config import prepare_config
    from optimize import _resume_config_mismatches

    config = _config()
    config["optimize"]["backend"] = backend
    config["backtest"]["end_date"] = token
    monkeypatch.setattr(
        utils, "utc_ms", lambda: utils.date_to_ts("2026-08-21T12:00:00")
    )
    previous = prepare_config(deepcopy(config), verbose=False)
    old = _record(previous)
    assert previous["backtest"]["end_date"] == "2026-08-19"
    assert (
        _resume_config_mismatches(old, prepare_config(deepcopy(config), verbose=False))
        == []
    )
    monkeypatch.setattr(
        utils, "utc_ms", lambda: utils.date_to_ts("2026-08-22T12:00:00")
    )
    current = prepare_config(deepcopy(config), verbose=False)
    assert current["backtest"]["end_date"] == "2026-08-20"
    assert any(
        "backtest.end_date" in item for item in _resume_config_mismatches(old, current)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["deap", "pymoo"])
async def test_main_resolves_override_file_before_cpu_candidate_and_rust_payload(
    tmp_path, monkeypatch, backend
):
    import numpy as np
    import optimize

    config = _config()
    config["optimize"].update(backend=backend, limits=[])
    config["backtest"].update(exchanges=["binance"], filter_by_min_effective_cost=False)
    config["live"].update(approved_coins={"long": ["BTC"], "short": []})
    config["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    (tmp_path / "coin.json").write_text(
        json.dumps(
            {
                "bot": {"long": {"risk": {"entry_cooldown_minutes": 37.0}}},
            }
        )
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(optimize.sys, "argv", ["passivbot optimize", str(path)])
    real_execute = optimize.execute_backtest
    observed = []

    class EvaluatedOffline(BaseException):
        pass

    def execute_and_capture(payload, candidate):
        assert payload.bot_params_list[0]["long"]["risk_entry_cooldown_minutes"] == 37.0
        assert (
            candidate["coin_overrides"]["BTC"]["bot"]["long"]["risk"][
                "entry_cooldown_minutes"
            ]
            == 37.0
        )
        observed.append(real_execute(payload, candidate))
        raise EvaluatedOffline

    async def evaluate_before_data_download(run_config, *args, **kwargs):
        # Enter the real CPU candidate/evaluation path using only synthetic data,
        # stopping before the CLI attempts market discovery or downloads.
        run_config["backtest"]["coins"] = {"binance": ["BTC"]}
        timestamps = 1704067200000 + np.arange(4, dtype=np.int64) * 60_000
        candles = np.tile([101.0, 99.0, 100.0, 1.0], (4, 1, 1))
        market = {
            "BTC": {
                "qty_step": 0.001,
                "price_step": 0.01,
                "min_qty": 0.001,
                "min_cost": 1.0,
                "c_mult": 1.0,
                "maker": 0.0,
                "taker": 0.0,
                "exchange": "binance",
                "first_valid_index": 0,
                "last_valid_index": 3,
            }
        }
        evaluator = optimize.Evaluator(
            hlcvs_specs={"binance": None},
            btc_usd_specs={},
            msss={"binance": market},
            config=run_config,
            timestamps={"binance": timestamps},
        )
        evaluator.shared_hlcvs_np["binance"] = candles
        evaluator.shared_btc_np["binance"] = np.full(4, 50_000.0)
        vector = optimize.config_to_individual(run_config, evaluator.bounds)
        evaluator.evaluate(vector, [])

    monkeypatch.setattr(optimize, "execute_backtest", execute_and_capture)
    monkeypatch.setattr(
        optimize, "format_approved_ignored_coins", evaluate_before_data_download
    )
    with pytest.raises(EvaluatedOffline):
        await optimize.main()
    assert len(observed) == 1


@pytest.mark.parametrize("backend", ["deap", "pymoo"])
@pytest.mark.parametrize("token", ["now", "today"])
def test_prepared_cpu_suite_date_rollover_rejects_saved_fitness(
    monkeypatch, backend, token
):
    import utils
    from types import SimpleNamespace
    from config import prepare_config
    from optimize import (
        _materialize_suite_run_contract,
        _materialize_resolved_suite_dates,
        _resume_config_mismatches,
    )
    from suite_runner import apply_scenario, build_scenarios

    config = _config()
    config["optimize"]["backend"] = backend
    suite = {
        "enabled": True,
        "exchanges": ["binance"],
        "scenarios": [
            {"label": "rolling", "start_date": "2026-08-01", "end_date": token},
        ],
    }

    def prepared_on(day):
        monkeypatch.setattr(utils, "utc_ms", lambda: utils.date_to_ts(day))
        run = prepare_config(deepcopy(config), verbose=False)
        _materialize_suite_run_contract(run, suite)
        scenarios, _ = build_scenarios(suite, base_exchanges=["binance"])
        raw, _ = apply_scenario(
            run, scenarios[0], ["BTC"], [], ["binance"], {"BTC"}, quiet=True
        )
        prepared = prepare_config(raw, verbose=False)
        metadata = {
            f"requested_{key}": prepared["backtest"][key]
            for key in ("start_date", "end_date")
        }
        _materialize_resolved_suite_dates(
            run,
            [
                SimpleNamespace(
                    label="rolling", msss={"binance": {"__meta__": metadata}}
                )
            ],
        )
        return run

    old = _record(prepared_on("2026-08-21T12:00:00"))
    assert _resume_config_mismatches(old, prepared_on("2026-08-21T18:00:00")) == []
    current = prepared_on("2026-08-22T12:00:00")
    assert old["backtest"]["end_date"] == current["backtest"]["end_date"]
    assert any(
        "backtest.scenarios" in item for item in _resume_config_mismatches(old, current)
    )
