"""GPU orchestration and durable recovery with controlled offline completion timing.

Only the GPU proxy and process-pool scheduling are replaced. Candidate evaluation,
NSGA ask/tell, result recording, checkpointing and CLI resume use production code.
"""

import copy
import json
import pickle
import sys
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")  # Proxy metric helpers require torch, but no GPU is used.

from test_hsl_revised_offline_runtime import deny_network, offline_cli_config


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_kind", ["interrupt", "recorder_error", "seeded", "seeded_screened"])
async def test_exact_collection_durable_tail_and_cli_resume(tmp_path, monkeypatch, stop_kind, capsys):
    import optimize
    from config.optimize_bounds import set_flat_optimize_bound
    from opt_utils import load_results
    from optimization import problem
    from optimization.backends import gpu_backend
    from optimization.gpu import runtime, service
    from optimization.progress import work_scope
    from optimization.shape import build_optimization_shape

    cfg = offline_cli_config(tmp_path, monkeypatch, "coin")
    cfg["live"]["strategy_kind"] = "trailing_martingale"
    cfg["live"]["approved_coins"]["short"] = ["BTC"]
    cfg["bot"]["short"]["risk"].update(n_positions=0, total_wallet_exposure_limit=0.0)
    cfg["optimize"].update(
        backend="gpu", iters=32, n_cpus=1, population_size=8, seed=42,
        scoring=[{"metric": "adg_usd", "goal": "max"}], limits=[],
        enable_overrides=[], compress_results_file=True, write_all_results=True,
    )
    cfg["optimize"]["gpu"].update(
        population_size=8, batch_size=8, exact_workers=1, max_pending_exact=4,
        validate_per_generation=2, drift_probes=1, auto_lean_parallelism=False,
    )
    seeded = stop_kind.startswith("seeded")
    if stop_kind == "seeded_screened":
        cfg["optimize"]["gpu"]["seed_bootstrap"]["mode"] = "screened"
    shape = build_optimization_shape(cfg)
    for key, path in shape.key_paths:
        value = cfg
        for part in path:
            value = value[part]
        set_flat_optimize_bound(cfg["optimize"]["bounds"], "trailing_martingale", key, [value, value])
    cfg["optimize"]["bounds"]["long"]["hsl"]["red_threshold"] = [0.01, 0.2]

    state = SimpleNamespace(resuming=seeded, proxy_calls=0, pools=[], submissions=[], records=[], scopes=[])
    oldest_ready = threading.Event()
    recorded = threading.Event()
    original_record = optimize.ResultRecorder.record

    def record(recorder, entry):
        original_record(recorder, entry)
        state.records.append(copy.deepcopy(entry))
        if not state.resuming and len(state.records) == 2:
            recorded.set()
            if stop_kind == "recorder_error":
                raise RuntimeError("injected recorder failure after durable flush")

    class Result:
        def __init__(self, payload, index):
            self.payload, self.index = payload, index

        def ready(self):
            # The second job finishes first. Nothing may be persisted ahead of
            # the oldest submission, even though that second payload is ready.
            return state.resuming or self.index > 0 or oldest_ready.is_set()

        def get(self):
            assert self.ready()
            return copy.deepcopy(self.payload)

    class Pool:
        def __init__(self, *, processes, initializer):
            self.evaluator, self.overrides, self.n_obj, self.has_constraints = initializer.args[:4]
            self._pool = ()
            self.terminated = False
            state.pools.append(self)

        def evaluate(self, vector):
            return problem._evaluate_pymoo_worker(
                self.evaluator, vector, self.overrides, self.n_obj, self.has_constraints
            )

        def apply_async(self, function, args):
            assert function is gpu_backend._evaluate_pymoo_worker_from_globals
            vector, = args
            result = Result(self.evaluate(vector), len(state.submissions))
            state.submissions.append(tuple(vector))
            return result

        def terminate(self):
            self.terminated = True

        def close(self):
            pass

        def join(self):
            pass

    class Proxy:
        def __init__(self, **kwargs):
            pass

        def evaluate(self, candidates):
            state.scopes.append(work_scope())
            state.proxy_calls += 1
            if not state.resuming and state.proxy_calls == 2:
                assert len(state.submissions) == 2
                assert state.records == []
                checkpoint, = (tmp_path / "optimize_results").rglob("checkpoint.pkl")
                before = checkpoint.read_bytes()
                assert pickle.loads(before)["generation"] == 1
                assert pickle.loads(before)["exact_done"] == 0
                oldest_ready.set()
                assert recorded.wait(10), "exact results did not persist during proxy evaluation"
                assert checkpoint.read_bytes() == before, "incomplete ask/tell was checkpointed"
                artifact = checkpoint.parent / "all_results.bin"
                assert len(list(load_results(str(artifact)))) == 2
                if stop_kind == "interrupt":
                    raise KeyboardInterrupt
            pool = state.pools[-1]
            rows = []
            for candidate in candidates:
                vector = [
                    candidate["long_hsl_red_threshold"] if key == "long_hsl_red_threshold" else bound.low
                    for (key, _), bound in zip(pool.evaluator.optimization_shape.key_paths, pool.evaluator.bounds)
                ]
                payload = pool.evaluate(vector)
                rows.append({"adg_usd": -float(payload["F"][0]), "backtest_completion_ratio": 1.0})
            return rows

    monkeypatch.setattr(optimize.ResultRecorder, "record", record)
    monkeypatch.setattr(gpu_backend.multiprocessing, "Pool", Pool)
    monkeypatch.setattr(service, "MpsSingleCoinProxy", Proxy)
    monkeypatch.setattr(runtime, "gpu_device", lambda *args: "mps")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    argv = ["optimize", str(path), "--suite", "n"]
    if seeded:
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        for index, threshold in enumerate([0.05, 0.15]):
            seed_config = copy.deepcopy(cfg)
            seed_config["bot"]["long"]["hsl"]["red_threshold"] = threshold
            (seeds / f"seed_{index}.json").write_text(json.dumps(seed_config))
        argv += ["-t", str(seeds)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        await optimize.main()
    if seeded:
        assert stopped.value.code == 0
        artifact, = (tmp_path / "optimize_results").rglob("all_results.bin")
        records = list(load_results(str(artifact)))
        final = pickle.loads((artifact.parent / "checkpoint.pkl").read_bytes())
        assert final["seed_exact_done"] == 2
        assert final["exact_done"] == cfg["optimize"]["iters"]
        assert len(records) == cfg["optimize"]["iters"] + 2
        output = capsys.readouterr().err
        assert "GPU seed exact start | completed=0/2 workers=1" in output
        assert "GPU seed exact complete | completed=2/2 inflight=0 queued=0" in output
        assert "phase=seed_exact" in output and "phase=generation_complete" in output
        assert "phase=complete" in output and "evolution_exact=32/32" in output
        if stop_kind == "seeded_screened":
            assert state.scopes[:2] == ["gen=0 phase=seed_proxy", "gen=1 phase=gpu_proxy"]
            assert "phase=seed_proxy" in output and "seed_proxy=2" in output
        else:
            assert state.scopes[0] == "gen=1 phase=gpu_proxy"
        assert work_scope() == "phase=gpu_proxy"
        return
    assert stopped.value.code == (130 if stop_kind == "interrupt" else 1)
    assert state.pools[-1].terminated
    assert state.proxy_calls == 2
    assert state.scopes[:2] == ["gen=1 phase=gpu_proxy", "gen=2 phase=gpu_proxy"]
    assert work_scope() == "phase=gpu_proxy"
    assert len(state.submissions) == len(state.records) == 2
    assert not any(t.name == "gpu-exact-collector" for t in threading.enumerate())

    artifact, = (tmp_path / "optimize_results").rglob("all_results.bin")
    checkpoint = artifact.parent / "checkpoint.pkl"
    assert pickle.loads(checkpoint.read_bytes())["exact_done"] == 0
    first_records = list(load_results(str(artifact)))
    thresholds = [entry["bot"]["long"]["hsl"]["red_threshold"] for entry in first_records]
    shape = state.pools[-1].evaluator.optimization_shape
    index = next(i for i, (key, _) in enumerate(shape.key_paths) if key == "long_hsl_red_threshold")
    assert thresholds == [vector[index] for vector in state.submissions]

    state.resuming = True
    monkeypatch.setattr(sys, "argv", ["optimize", str(path), "--suite", "n", "--resume", str(artifact.parent)])
    with pytest.raises(SystemExit) as finished:
        await optimize.main()
    assert finished.value.code == 0
    records = list(load_results(str(artifact)))
    final = pickle.loads(checkpoint.read_bytes())
    assert final["exact_done"] == len(records) == cfg["optimize"]["iters"]
    assert len(state.submissions) == len(set(state.submissions)) == len(records)
    assert records[:2] == first_records
    assert len(final["completed_hashes"]) == len(records)
    assert len(final["drift_pairs"]) == len(records)
    assert not any(t.name == "gpu-exact-collector" for t in threading.enumerate())
