from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import optimize
from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY
from optimization.native_checkpoint import load_checkpoint
from optimization.native_datasets import NativeDatasetRegistry
from test_native_backend import execute, guard_cpu, inputs, managed_arrays
from test_native_datasets import contexts


def variable_base(manager, *, anchor=False, disabled=False, mirror=False):
    original = inputs(manager)
    config = deepcopy(original.config)
    config["optimize"]["bounds"]["short_total_wallet_exposure_limit"] = [0, 1]
    if disabled:
        config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0
    if mirror:
        config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    if anchor:
        seeds = [deepcopy(config), deepcopy(config)]
        seeds[0]["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0
        seeds[1]["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 1
        optimize.install_anchored_fine_tune_plan(
            config, ["long.strategy.entry.initial_qty_pct"], None, starting_configs_override=seeds,
        )
    return optimize.Evaluator(original.hlcvs_specs, original.btc_usd_specs, original.msss,
                              config, timestamps=original.timestamps)


def vector(base, *, disabled=False, anchor_id=0):
    config = deepcopy(base.config)
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0 if disabled else 1
    return optimize.config_to_individual(config, base.bounds, optimization_shape=base.optimization_shape,
                                         anchor_id=anchor_id)


@pytest.mark.parametrize("anchor,disabled,mirror", [(False, False, False), (False, True, False),
                                                    (True, False, False), (False, False, True)])
def test_finite_variants_select_execution_views_without_copying_histories(monkeypatch, anchor, disabled, mirror):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = variable_base(manager, anchor=anchor, disabled=disabled, mirror=mirror)
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance":("COIN00", "COIN01", "COIN02")}) as registry:
            assert len(registry.bindings) == (1 if mirror else 2)
            datasets = [binding.dataset for binding in registry.bindings]
            assert len({dataset.hlcvs for dataset in datasets}) == 1
            assert len({dataset.btc for dataset in datasets}) == 1
            assert len({dataset.timestamps for dataset in datasets}) == 1
            off = registry.planner.prepare("off", vector(base, disabled=True, anchor_id=0))
            on = registry.planner.prepare("on", vector(base, anchor_id=1 if anchor else 0))
            assert (off.requests[0].dataset_id == on.requests[0].dataset_id) == mirror
            assert (off.effective_key == on.effective_key) == mirror
            assert off.requests[0].parameters["short_total_wallet_exposure_limit"] == (1 if mirror else 0)
            assert on.requests[0].parameters["short_total_wallet_exposure_limit"] == 1


def test_exact_last_scenario_policy_deduplicates_fixed_views_and_preserves_slots(monkeypatch):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = variable_base(manager)
        prepared = contexts(base, base.timestamps["binance"], lazy=True)
        prepared[1].overrides = {"bot.short.risk.total_wallet_exposure_limit":0}
        suite = optimize.SuiteEvaluator(base, prepared, {"default":"mean"})
        with NativeDatasetRegistry(suite) as registry:
            assert len(registry.bindings) == 3
            off = registry.planner.prepare("off", vector(base, disabled=True))
            on = registry.planner.prepare("on", vector(base))
            assert off.requests[0].dataset_id != on.requests[0].dataset_id
            assert off.requests[1].dataset_id == on.requests[1].dataset_id
            screen = registry.planner.prepare("on", vector(base), scenarios=["stress"])
            assert screen.slots == (on.slots[1],)
            assert screen.effective_key == on.effective_key != off.effective_key


def test_unprepared_continuous_static_changes_still_fail_before_submission(monkeypatch):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = variable_base(manager)
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance":("COIN00", "COIN01", "COIN02")}) as registry:
            base.config["optimize"]["fixed_runtime_overrides"] = {"backtest.starting_balance":1234}
            with pytest.raises(ValueError, match="dataset-owned execution inputs"):
                registry.planner.prepare("changed", vector(base))


@pytest.mark.parametrize("bounds,expected", [([0.4, 1.0], 2), ([0.5, 1.0], 2),
                                            ([0.4, 1.0, 0.6], 2), ([0.4, 0.59, 0.1], 1),
                                            ([0.6, 1.0], 1)])
def test_position_rounding_and_stepped_endpoints_determine_finite_views(monkeypatch, bounds, expected):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        original = inputs(manager)
        config = deepcopy(original.config)
        config["optimize"]["bounds"]["short_n_positions"] = bounds
        base = optimize.Evaluator(original.hlcvs_specs, original.btc_usd_specs, original.msss,
                                  config, timestamps=original.timestamps)
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance":("COIN00", "COIN01", "COIN02")}) as registry:
            assert len(registry.bindings) == expected
            requests = []
            for index, endpoint in enumerate((bounds[0], bounds[1])):
                candidate = deepcopy(config)
                candidate["bot"]["short"]["risk"]["n_positions"] = endpoint
                values = optimize.config_to_individual(candidate, base.bounds,
                                                        optimization_shape=base.optimization_shape)
                requests.append(registry.planner.prepare(f"endpoint:{index}", values).requests[0])
            assert len({request.dataset_id for request in requests}) == expected


def test_two_side_choices_prepare_only_supported_topologies_and_reject_zero_sides(monkeypatch):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = variable_base(manager)
        config = deepcopy(base.config)
        config["optimize"]["bounds"]["long_total_wallet_exposure_limit"] = [0, 1]
        base = optimize.Evaluator(base.hlcvs_specs, base.btc_usd_specs, base.msss,
                                  config, timestamps=base.timestamps)
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance":("COIN00", "COIN01", "COIN02")}) as registry:
            assert len(registry.bindings) == 3
            choices = {}
            for long, short in ((1, 0), (0, 1), (1, 1), (0, 0)):
                candidate = deepcopy(config)
                for side, value in (("long", long), ("short", short)):
                    candidate["bot"][side]["risk"]["total_wallet_exposure_limit"] = value
                values = optimize.config_to_individual(candidate, base.bounds,
                                                        optimization_shape=base.optimization_shape)
                if not long and not short:
                    with pytest.raises(ValueError, match="dataset-owned execution inputs"):
                        registry.planner.prepare("zero", values)
                else:
                    choices[long, short] = registry.planner.prepare(f"{long}:{short}", values).requests[0].dataset_id
            assert len(set(choices.values())) == 3


def test_native_anchors_survive_interrupt_and_resume_without_seed_files(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = variable_base(manager, anchor=True)
        records, path = [], tmp_path / "checkpoint.pkl"
        def interrupt():
            if records:
                raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            execute(base, SimpleNamespace(record=records.append), path, interrupt_check=interrupt)
        state = load_checkpoint(path, base.config)
        assert state["anchor_plan"] == base.config[ANCHOR_PLAN_KEY]
        restored = deepcopy(base.config)
        restored.pop(ANCHOR_PLAN_KEY)
        assert optimize._restore_gpu_resume_anchor_plan(restored, str(path))
        assert restored[ANCHOR_PLAN_KEY] == state["anchor_plan"]
        restored["optimize"]["iters"] = 12
        resumed = optimize.Evaluator(base.hlcvs_specs, base.btc_usd_specs, base.msss,
                                     restored, timestamps=base.timestamps)
        assert len(resumed.bounds) == 2
        execute(resumed, SimpleNamespace(record=records.append), path, resume=True)
        done = load_checkpoint(path, resumed.config)
        assert done["phase"] == "idle" and done["completed"] == len(records) == 12
        restored[ANCHOR_PLAN_KEY]["anchors"][0]["fixed_values"][0]["value"] += 0.01
        with pytest.raises(ValueError, match="evaluation contract changed"):
            load_checkpoint(path, restored)


@pytest.mark.parametrize("strategy,anchor,disabled", [("trailing_martingale", False, False),
                                                     ("trailing_martingale", True, True),
                                                     ("ema_anchor", False, False),
                                                     ("ema_anchor", False, True)])
def test_cuda_variants_match_independent_replays_and_reuse_one_market_pack(monkeypatch, strategy, anchor, disabled):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.service import MpsMulticoinProxy
    import optimization.gpu.native as native
    guard_cpu(monkeypatch)
    monkeypatch.setattr(native, "CudaBacktestService", CudaBacktestService)
    with managed_arrays() as manager:
        base = variable_base(manager, anchor=anchor, disabled=disabled)
        if strategy == "ema_anchor":
            from tools.gpu_parity import build_parser, fixture_inputs
            fixture, *_ = fixture_inputs(build_parser().parse_args([
                "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
            ]))
            config = deepcopy(base.config)
            config["live"]["strategy_kind"] = strategy
            for side in ("long", "short"):
                config["bot"][side]["strategy"] = deepcopy(fixture["bot"][side]["strategy"])
            config["optimize"]["bounds"] = {key: value for key, value in config["optimize"]["bounds"].items()
                                           if "initial_qty_pct" not in key}
            config["optimize"]["bounds"]["long_base_qty_pct"] = [0.01, 0.05]
            base = optimize.Evaluator(base.hlcvs_specs, base.btc_usd_specs, base.msss,
                                      config, timestamps=base.timestamps)
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance":("COIN00", "COIN01", "COIN02")}) as registry:
            plans = [registry.planner.prepare(f"candidate:{index}",
                     vector(base, disabled=index % 2 == 0, anchor_id=index % 2 if anchor else 0))
                     for index in range(4)]
            expected = []
            for plan in plans:
                binding = next(row for row in registry.bindings if row.dataset_id == plan.requests[0].dataset_id)
                dataset = binding.dataset
                with dataset.attach() as (candles, btc, timestamps):
                    replay = MpsMulticoinProxy(config=json.loads(dataset.config_json),
                                               mss=json.loads(dataset.markets_json), exchange=dataset.exchange,
                                               hlcvs=candles, btc=btc, timestamps=timestamps,
                                               needed_metrics=dataset.metrics, batch_size=1)
                    expected.append(replay.evaluate_results([dict(plan.requests[0].parameters)])[0])
                    del replay
            with CudaBacktestService(batch_size=2) as service:
                registry.register(service)
                for plan, reference in zip(plans, expected, strict=True):
                    result = service.submit(plan.requests[0]).result(timeout=60)
                    assert result.metrics == reference.metrics
                    assert result.liquidated == reference.liquidated
                    assert len(service._prepared_cache) == 1
                    owners = [row.evaluate.__self__ for row in service._executor._replays.values()]
                    assert sum(bool(owner.runners or owner.fused_runner is not None) for owner in owners) == 1
