from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.population import Population

import optimize
from optimization.backends.gpu_backend import NativeSearchProblem, _Search
from optimization.callback import build_pymoo_record_entry
from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY
from optimization.gpu.executor import BacktestResult
from optimization.native_datasets import NativeDatasetRegistry
from optimization.native_session import NativeEvaluationSession
from test_native_backend import execute, guard_cpu, inputs, managed_arrays
from test_native_coupled_spans import coupled_inputs
from test_native_datasets import contexts
from test_native_variants import variable_base


COLUMNS = {"binance": ("COIN00", "COIN01", "COIN02")}


def record_kwargs(base, vector, *, suite=False):
    return dict(
        vector=list(vector), metrics={"suite_metrics": {"base": {}}} if suite else {},
        template=base.config, build_config_fn=optimize.individual_to_config,
        overrides_fn=optimize.optimizer_overrides,
        overrides_list=base.config["optimize"].get("enable_overrides", []),
    )


@pytest.mark.parametrize("policy", ["fixed", "mirror", "coupled_tm", "coupled_ema", "anchor", "weights"])
@pytest.mark.parametrize("suite", [False, True])
def test_prepared_record_matches_rebuilt_effective_candidate(monkeypatch, policy, suite):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        if policy.startswith("coupled_"):
            kind = "ema_anchor" if policy == "coupled_ema" else "trailing_martingale"
            base = coupled_inputs(manager, kind)
        elif policy == "anchor":
            base = variable_base(manager, anchor=True)
        else:
            previous = inputs(manager)
            config = deepcopy(previous.config)
            if policy == "fixed":
                config["optimize"]["fixed_runtime_overrides"] = {
                    "bot.long.strategy.trailing_martingale.entry.initial_qty_pct": 0.025,
                }
            elif policy == "mirror":
                config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
            else:
                config["optimize"]["round_to_n_significant_digits"] = 2
                config["optimize"]["bounds"]["long"]["forager"] = {
                    "score_weights": {name: [0.0, 1.0] for name in ("volume", "ema_readiness", "volatility")},
                }
            base = optimize.Evaluator(previous.hlcvs_specs, previous.btc_usd_specs, previous.msss,
                                      config, timestamps=previous.timestamps)
        evaluator = base
        if suite:
            prepared = contexts(base, base.timestamps["binance"], lazy=True)
            kind = base.config['bot']['global']['strategy_kind']
            entry = "entry." if kind == "trailing_martingale" else ""
            prepared[1].overrides = {f"bot.long.strategy.{kind}.{entry}ema_span_1": 3.5}
            evaluator = optimize.SuiteEvaluator(base, prepared, {"default": "mean"})
        vector = [(bound.low + bound.high) / 2 for bound in base.bounds]
        if policy == "anchor":
            vector[0] = 1.0
        elif policy == "weights":
            weights = dict(volume=0.24, ema_readiness=0.55, volatility=0.53)
            for index, (name, _path) in enumerate(base.key_paths):
                for key, value in weights.items():
                    if name == f"long_forager_score_weights_{key}":
                        vector[index] = value
        supplied = list(vector)
        with NativeDatasetRegistry(evaluator, standalone_candle_coins=COLUMNS) as registry:
            plan = registry.planner.prepare("record", vector)
            assert plan.record_config is not None
            before = deepcopy(plan.record_config)
            kwargs = record_kwargs(base, plan.vector, suite=suite)
            expected = build_pymoo_record_entry(**kwargs)
            def forbidden(*_args, **_kwargs):
                pytest.fail("prepared config must not be materialized again")
            kwargs["build_config_fn"] = forbidden
            actual = build_pymoo_record_entry(**kwargs, prepared_config=plan.record_config)
            assert actual == expected
            assert plan.record_config == before
            if policy == "anchor":
                actual["optimizer_anchor"]["tunable_keys"].append("changed")
                assert plan.record_config == before
            if policy == "weights":
                assert tuple(supplied) != plan.vector
            if suite:
                screen = registry.planner.prepare("screen", vector, scenarios=["base"])
                assert screen.record_config is None


def search_for(base, registry, session, vectors, records, *, builder=None):
    problem = NativeSearchProblem(base)
    state = dict(
        algorithm=SimpleNamespace(problem=problem, n_iter=1, evaluator=SimpleNamespace(n_eval=0)),
        population=Population.new("X", np.asarray(vectors)), phase="generation", sequence=0,
        completed=0, screened=0,
    )
    search = _Search(
        state=state, session=session, registry=registry,
        recorder=SimpleNamespace(record=records.append), template=base.config,
        build_config_fn=optimize.individual_to_config if builder is None else builder,
        overrides_fn=optimize.optimizer_overrides,
        overrides_list=base.config["optimize"].get("enable_overrides", []),
        checkpoint_path=None, checkpoint_interval=0, interrupt_check=lambda: None,
        screening_policy={"scenarios": []},
    )
    search.checkpoint = lambda **_kwargs: None
    return search


def test_out_of_order_and_cached_anchor_duplicates_record_each_owned_config(monkeypatch):
    import optimization.backends.gpu_backend as backend

    guard_cpu(monkeypatch)
    build_entry = backend.build_pymoo_record_entry
    def require_prepared(**kwargs):
        assert kwargs["prepared_config"] is not None
        return build_entry(**kwargs)
    monkeypatch.setattr(backend, "build_pymoo_record_entry", require_prepared)
    with managed_arrays() as manager:
        previous = inputs(manager)
        config = deepcopy(previous.config)
        path = ["bot", "long", "strategy", "trailing_martingale", "entry", "initial_qty_pct"]
        config[ANCHOR_PLAN_KEY] = dict(
            anchors=[dict(fixed_values=[], source=name) for name in ("first", "second")],
            key_paths=[path], tunable_keys=["long_entry_initial_qty_pct"], fixed_keys=[],
        )
        base = optimize.Evaluator(previous.hlcvs_specs, previous.btc_usd_specs, previous.msss,
                                  config, timestamps=previous.timestamps)
        vectors = [[0.0, 0.02], [1.0, 0.04], [1.0, 0.02], [0.0, 0.04], [0.0, 0.02]]
        with NativeDatasetRegistry(base, standalone_candle_coins=COLUMNS) as registry:
            class Service:
                def __init__(self):
                    self.submitted = []
                def submit(self, request):
                    assert len(self.submitted) < 2, "effective duplicates should reuse simulator evidence"
                    owner = self
                    class ReadyAfterCallbacks(Future):
                        def add_done_callback(self, callback):
                            super().add_done_callback(callback)
                            if len(owner.submitted) == 2:
                                for row, ready in reversed(owner.submitted):
                                    ready.set_result(BacktestResult(
                                        row.request_id, row.dataset_id,
                                        {metric: 0.1 for metric in registry.bindings[0].dataset.metrics}, False,
                                    ))
                    future = ReadyAfterCallbacks()
                    self.submitted.append((request, future))
                    return future
            service = Service()
            session = NativeEvaluationSession(service, registry.scorer, max_candidates=2)
            records = []
            search = search_for(base, registry, session, vectors, records)
            admit = session.admit
            def observe_admission(plan):
                assert plan.record_config is None
                assert len(search._record_configs) < session.max_candidates
                return admit(plan)
            monkeypatch.setattr(session, "admit", observe_admission)
            search.evaluate_population()
            assert search._record_configs == {}
            assert session.active_candidate_ids == ()
            actual = [(row["optimizer_anchor"]["id"],
                       row["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"])
                      for row in records]
            assert actual[:2] == [(1, 0.04), (0, 0.02)]
            assert sorted(actual) == sorted((int(anchor), value) for anchor, value in vectors)
            assert search.state["completed"] == len(vectors)
            for row in records:
                vector = [float(row["optimizer_anchor"]["id"]),
                          row["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"]]
                expected = build_pymoo_record_entry(**{**record_kwargs(base, vector), "metrics": row["metrics"]})
                assert row == expected


@pytest.mark.parametrize("custom_builder", [False, True])
def test_changed_completion_vector_or_custom_builder_uses_existing_record_path(monkeypatch, custom_builder):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        vector = optimize.config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
        with NativeDatasetRegistry(base, standalone_candle_coins=COLUMNS) as registry:
            plan = registry.planner.prepare("record", vector)
            changed = list(plan.vector)
            index = [name for name, _path in base.key_paths].index("long_entry_initial_qty_pct")
            changed[index] = 0.035
            calls = []
            def builder(*args):
                calls.append(args[0])
                result = optimize.individual_to_config(*args)
                result["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.033
                return result
            records = []
            search = search_for(base, registry, SimpleNamespace(max_candidates=1), [plan.vector], records,
                                builder=builder if custom_builder else None)
            payload = dict(fitness=[0.1, 0.2], constraint_violation=0,
                           evaluation_vector=list(plan.vector) if custom_builder else changed, metrics={})
            search.apply_terminal(0, payload, prepared=(plan.vector, plan.record_config))
            assert records[0]["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] == (
                0.033 if custom_builder else 0.035
            )
            assert len(calls) == int(custom_builder)
            assert search.state["population"][0].X.tolist() == payload["evaluation_vector"]


@pytest.mark.parametrize("failure", ["interrupt", "producer", "record"])
def test_shutdown_releases_pending_record_configs_and_resume_rebuilds(monkeypatch, tmp_path, failure):
    from test_native_backend import FakeService

    guard_cpu(monkeypatch)
    instances = []
    init = _Search.__init__
    def observe(search, **kwargs):
        init(search, **kwargs)
        instances.append(search)
    monkeypatch.setattr(_Search, "__init__", observe)
    if failure == "producer":
        FakeService.failure = RuntimeError("producer failed")
    records = []
    def record(row):
        if failure == "record":
            raise RuntimeError("record failed")
        records.append(row)
    def interrupt():
        if failure == "interrupt" and instances and instances[-1]._record_configs:
            raise KeyboardInterrupt
    with managed_arrays() as manager:
        base = inputs(manager)
        checkpoint = tmp_path / "checkpoint.pkl"
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
            execute(base, SimpleNamespace(record=record), checkpoint, interrupt_check=interrupt)
        assert instances and instances[-1]._record_configs == {}
        assert "_record_configs" not in instances[-1].state
        FakeService.failure = None
        execute(base, SimpleNamespace(record=records.append), checkpoint, resume=True)
        assert len(records) == 8
        assert instances[-1]._record_configs == {}


def test_prefilter_rejection_reuses_config_without_admission_or_retention(monkeypatch, tmp_path):
    import optimization.native_planning as planning
    from test_native_backend import FakeService

    guard_cpu(monkeypatch)
    monkeypatch.setattr(planning, "prefilter_candidate", lambda *_args: (1.0, []))
    seen = []
    original = _Search.apply_terminal
    def terminal(search, index, payload, *, prepared=None):
        assert search._record_configs == {}
        assert prepared is not None and prepared[1] is not None
        seen.append(prepared[0])
        return original(search, index, payload, prepared=prepared)
    monkeypatch.setattr(_Search, "apply_terminal", terminal)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["prefilter"] = {
            "checks": [{"metric": "entry_close_span", "enabled": True, "max": 0.0, "sides": ["long"]}],
        }
        records = []
        execute(base, SimpleNamespace(record=records.append), tmp_path / "checkpoint.pkl")
        assert len(records) == len(seen) == 8
        assert FakeService.rows == []
        assert all(row["metrics"]["evaluation_origin"] == "config_prefilter" for row in records)
