"""CPU-only observations of replay orchestration, without a device context."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu import mps_kernel, service


class Clock:
    def __init__(self):
        self.seconds = 0.0
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.seconds


@pytest.mark.parametrize("profile", [False, True])
def test_temporal_host_split_keeps_dispatches_waits_and_tuner_samples(monkeypatch, profile):
    clock, launches, ranges, waits, samples, metadata_reads = Clock(), [], [], [], [], []
    batch_size = 65

    class Kernel:
        def __call__(self, *args, **kwargs):
            launches.append((args, kwargs))
            clock.seconds += 2.0

        @property
        def max_threads_per_threadgroup(self):
            metadata_reads.append("maximum")
            return 256

        @property
        def thread_execution_width(self):
            metadata_reads.append("width")
            return 32

        @property
        def static_thread_group_memory_length(self):
            metadata_reads.append("memory")
            return 64

    def make_range(values, **kwargs):
        assert kwargs == {"dtype": "int32", "device": "mps"}
        ranges.append(values)
        return tuple(values)

    def wait():
        waits.append(True)
        clock.seconds += 3.0

    runner = mps_kernel._MulticoinReplayRunner.__new__(mps_kernel._MulticoinReplayRunner)
    runner.replay_kernel_name = "replay"
    runner._profile_pipeline = profile
    runner._pipeline_seconds = 0.0
    runner._temporal_replay_enabled = lambda: True
    runner.native_factual_hsl = False
    runner.bars = SimpleNamespace(device=SimpleNamespace(type="mps"))
    runner.temporal_chunk_bar_limit = 3
    runner.max_dispatch_candidate_bars = 100_000
    runner.n_coins = runner.replay_sides = 1
    runner._history_batch_bytes = lambda _: 0
    runner.hsl_scratch_budget_bytes = 1024
    runner._replay_state_bytes = 32
    runner._replay_states = {(batch_size, 32): "state"}
    runner.interrupt_check = lambda: None
    monkeypatch.setattr(mps_kernel.time, "perf_counter", clock)
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "mps")
    monkeypatch.setattr(mps_kernel, "torch", SimpleNamespace(tensor=make_range, int32="int32"))
    monkeypatch.setattr(mps_kernel, "synchronize", wait)
    monkeypatch.setattr(mps_kernel, "record_replay_chunk", lambda *a, **kw: samples.append((a, kw)))
    monkeypatch.setattr(mps_kernel, "_require_available_held_valuation", lambda _: None)
    monkeypatch.setattr(mps_kernel, "TemporalReplayProgress", lambda *a, **kw: SimpleNamespace(log=lambda *a, **kw: None))
    scalars = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: None))
    mps_kernel._MulticoinReplayRunner._dispatch_replay(
        runner, SimpleNamespace(replay=Kernel()), ("input",),
        torch.full((batch_size,), 7, dtype=torch.int32), batch_size=batch_size, scalars=scalars,
    )

    assert ranges == [[1, 4], [4, 7]]
    assert len(waits) == len(launches) == 2
    assert [call[1] for call in launches] == [
        {"threads": (65, 1, 1), "group_size": (32, 1, 1)},
    ] * 2
    assert samples == [
        ((65, 3, 6, 5.0), {"eligible": False}),
        ((65, 3, 6, 5.0), {"eligible": True}),
    ]
    assert runner._last_temporal_dispatch["dispatch_count"] == 2
    assert runner._last_temporal_dispatch["max_dispatch_seconds"] == 5.0
    # Existing replay/progress/tuner clocks remain; only profiling adds one
    # clock read per chunk plus the two existing pipeline-lookup reads.
    assert clock.reads == (12 if profile else 8)
    if profile:
        assert metadata_reads == ["maximum", "width", "memory"]
        assert runner._last_temporal_dispatch["dispatch_host_timings_seconds"] == {
            "submission": 4.0, "synchronization_wait": 6.0,
        }
        assert runner._profile_dispatch_launches == [{
            "kernel": "replay", "threads": [65, 1, 1], "group_size": [32, 1, 1],
            "threadgroup_count": [3, 1, 1], "dispatch_count": 2,
            "pipeline_capabilities": {
                "max_threads_per_threadgroup": 256, "thread_execution_width": 32,
                "static_thread_group_memory_length": 64,
            },
        }]
    else:
        assert metadata_reads == []
        assert "dispatch_host_timings_seconds" not in runner._last_temporal_dispatch
        assert not hasattr(runner, "_profile_dispatch_launches")


def test_launch_profile_distinguishes_unavailable_caps_and_implicit_groups():
    profile = mps_kernel._replay_launch_profile(object(), "replay", (3, 1, 1), None)
    assert profile["group_size"] is profile["threadgroup_count"] is None
    assert set(profile["pipeline_capabilities"].values()) == {None}


def test_unchunked_cuda_profile_normalizes_scalar_group_and_clears_temporal_state(monkeypatch):
    calls = []
    runner = mps_kernel._MulticoinReplayRunner.__new__(mps_kernel._MulticoinReplayRunner)
    runner.replay_kernel_name = "replay"
    runner._profile_pipeline = True
    runner._pipeline_seconds = 0.0
    runner._temporal_replay_enabled = lambda: False
    runner.replay_sides = 1
    runner.cuda_whole_group_size = 64
    runner._last_temporal_dispatch = {"dispatch_count": 99, "temporal_chunk_bars": 3}
    monkeypatch.setattr(mps_kernel.time, "perf_counter", Clock())
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "cuda")
    monkeypatch.setattr(mps_kernel, "synchronize", lambda: pytest.fail("must not add synchronization"))
    runner._dispatch_replay(
        SimpleNamespace(replay=lambda *a, **kw: calls.append((a, kw))), ("input",), None,
        batch_size=65, scalars=None,
    )
    assert calls == [(("input",), {"threads": (65, 1, 1), "group_size": 64})]
    assert runner._last_temporal_dispatch is None
    assert runner._profile_dispatch_launches[0]["group_size"] == [64, 1, 1]
    assert runner._profile_dispatch_launches[0]["threadgroup_count"] == [2, 1, 1]


@pytest.mark.parametrize("profile", [False, True])
def test_non_temporal_profile_preserves_launch_and_reuses_existing_caps(monkeypatch, profile):
    reads, calls = [], []

    class Kernel:
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))

        @property
        def max_threads_per_threadgroup(self):
            reads.append("maximum")
            return 1024

        @property
        def thread_execution_width(self):
            reads.append("width")
            return 32

        @property
        def static_thread_group_memory_length(self):
            reads.append("memory")
            return 0

    runner = mps_kernel._MulticoinReplayRunner.__new__(mps_kernel._MulticoinReplayRunner)
    runner.replay_kernel_name = "replay"
    runner._profile_pipeline = profile
    runner._pipeline_seconds = 0.0
    runner._temporal_replay_enabled = lambda: False
    monkeypatch.setattr(mps_kernel.time, "perf_counter", Clock())
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "mps")
    monkeypatch.setattr(mps_kernel, "synchronize", lambda: pytest.fail("must not add synchronization"))
    runner._dispatch_replay(SimpleNamespace(replay=Kernel()), ("input",), None, batch_size=2, scalars=None)
    assert calls == [(("input",), {"threads": (2, 1, 1), "group_size": (2, 1, 1)})]
    assert reads == ["maximum", "width"] + (["memory"] if profile else [])


def test_history_profiles_keep_physical_temporal_children(monkeypatch):
    runner = mps_kernel._MulticoinReplayRunner.__new__(mps_kernel._MulticoinReplayRunner)
    runner._history_batch_limit = lambda: 2
    physical = []

    def run(params, *, profile, end_steps):
        assert profile and end_steps is None
        width = len(params)
        temporal = dict(dispatch_count=2, temporal_chunk_bars=3, max_dispatch_seconds=float(width),
                        dispatch_host_timings_seconds={"submission": float(width), "synchronization_wait": 1.0})
        physical.append(temporal)
        runner.last_profile = dict(temporal, batch_size=width, kernel_candidate_steps=width * 6,
                                   kernel_seconds=float(width + 2), cold=width == 2,
                                   temporal_dispatches=[temporal], dispatch_launches=[{"threads": [width, 1, 1]}])
        return {"value": torch.zeros(width)}

    runner.run = run
    result = runner._run_history_batches(np.zeros((3, 1)), profile=True, end_steps=None)
    assert result["value"].shape == (3,)
    assert runner.last_profile["kernel_seconds"] == 7.0
    assert runner.last_profile["max_dispatch_seconds"] == 2.0
    assert runner.last_profile["dispatch_count"] == 4
    assert runner.last_profile["temporal_dispatches"] == physical
    assert runner.last_profile["dispatch_launches"] == [{"threads": [2, 1, 1]}, {"threads": [1, 1, 1]}]
    assert "dispatch_host_timings_seconds" not in runner.last_profile
    runner.n = 100
    runner.n_coins = 2
    profile = service._new_gpu_proxy_profile(
        SimpleNamespace(batch_size=3, dispatch_batch_size=3), [{}, {}, {}], (runner,), coin_count=2, side_count=2,
    )
    service._add_gpu_runner_profile(profile, runner, side_count=2)
    assert profile["dispatch_count_scope"] == "kernel_dispatches"
    assert profile["actual_dispatch_batch_sizes_scope"] == "physical_candidate_batches"
    assert profile["dispatch_count"] == 4
    assert profile["cold_dispatch_count"] == 1
    assert profile["warm_dispatch_count"] == 3
    assert profile["actual_dispatch_batch_sizes"] == [2, 1]
    assert profile["kernel_candidate_bars"] == 18 * 2 * 2
    assert [p["dispatch_count"] for p in profile["temporal_dispatches"]] == [2, 2]
    assert profile["kernel_execution_host_subtimings_seconds"] == {
        "temporal_submission": 3.0, "temporal_synchronization_wait": 2.0,
        "other_replay_host_wall": 2.0,
    }


def test_service_keeps_host_children_outside_additive_phase_totals(monkeypatch):
    runner = SimpleNamespace(n=100, n_coins=1, last_profile={
        "batch_size": 2, "dispatch_count": 2, "kernel_seconds": 10.0, "cold": False,
        "kernel_candidate_steps": 200, "temporal_chunk_bars": 50,
        "dispatch_host_timings_seconds": {"submission": 2.0, "synchronization_wait": 3.0},
        "dispatch_launches": [{"threads": [2, 1, 1]}],
    })
    profile = service._new_gpu_proxy_profile(
        SimpleNamespace(batch_size=2, dispatch_batch_size=2), [{}, {}], (runner,), coin_count=1, side_count=1,
    )
    service._add_gpu_runner_profile(profile, runner)
    # A legacy profile without children remains wholly attributed to other host work.
    runner.last_profile = {"batch_size": 1, "kernel_seconds": 5.0, "cold": False}
    service._add_gpu_runner_profile(profile, runner)
    monkeypatch.setattr(service.time, "perf_counter", lambda: 20.0)
    service._finish_gpu_proxy_profile(profile, 0.0)
    assert profile["timing_measurement"]["scope"] == "host_wall"
    assert profile["timing_measurement"]["device_timestamps_measured"] is False
    assert profile["kernel_execution_host_subtimings_seconds"] == {
        "temporal_submission": 2.0, "temporal_synchronization_wait": 3.0,
        "other_replay_host_wall": 10.0,
    }
    assert profile["timings_seconds"]["kernel_execution"] == 15.0
    assert service._gpu_profile_runner_seconds(profile["timings_seconds"]) == 15.0
    assert profile["timings_seconds"]["host_overhead"] == 5.0
    assert sum(profile["timings_seconds"].values()) == 20.0
    assert profile["dispatch_launches"] == [{"threads": [2, 1, 1]}]
