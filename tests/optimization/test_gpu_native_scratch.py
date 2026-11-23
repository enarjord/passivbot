"""Native CUDA capacity uses resident data, replay state and metric workspace."""

from types import SimpleNamespace

import pytest

from optimization.gpu.autotune import cuda_history_budget, history_dispatch_ceiling

GIB = 1024**3
MIB = 1024**2


def runner(*, row=64 * MIB, workspace=256 * MIB, device="cuda"):
    value = SimpleNamespace(
        bars=SimpleNamespace(device=SimpleNamespace(type=device)),
        native_factual_hsl=True,
        hsl_fact_capacity=0,
        hsl_scratch_budget_bytes=512 * MIB,
    )
    value._history_batch_bytes = lambda width: width * row + workspace
    value._history_batch_limit = lambda: (value.hsl_scratch_budget_bytes - workspace) // max(1, row)
    return value


@pytest.mark.parametrize("free_gib", [10, 5, 1])
@pytest.mark.parametrize("owners", [1, 2])
def test_budget_grows_and_shrinks_with_free_vram_and_shared_owners(free_gib, owners):
    value = runner()
    budget = cuda_history_budget(value, free_gib * GIB, 12 * GIB, owners=owners)
    allowance = max(int(free_gib * GIB * .35), free_gib * GIB - int(12 * GIB * .2)) // owners
    assert budget == max(value._history_batch_bytes(1), allowance)
    assert budget * owners <= free_gib * GIB
    if free_gib >= 5:
        assert free_gib * GIB - budget * owners >= int(12 * GIB * .2)


def test_workspace_is_reserved_once_and_capacity_can_exceed_legacy_budget():
    value = runner()
    replay = SimpleNamespace(fused_runner=value)
    assert history_dispatch_ceiling(replay, 128) == 4
    value.hsl_scratch_budget_bytes = cuda_history_budget(value, 10 * GIB, 12 * GIB)
    assert value.hsl_scratch_budget_bytes > 2 * GIB
    width = history_dispatch_ceiling(replay, 128)
    assert width == 117
    assert value._history_batch_bytes(width) <= value.hsl_scratch_budget_bytes
    assert value._history_batch_bytes(width + 1) > value.hsl_scratch_budget_bytes


def test_allowance_reserves_headroom_but_dispatch_still_caps_width():
    value = runner()
    value.hsl_scratch_budget_bytes = cuda_history_budget(value, 10 * GIB, 12 * GIB)
    assert value.hsl_scratch_budget_bytes > value._history_batch_bytes(3)
    assert history_dispatch_ceiling(SimpleNamespace(fused_runner=value), 3) == 3


def test_small_devices_reserve_at_least_one_gib_when_feasible():
    value = runner(row=MIB, workspace=0)
    assert cuda_history_budget(value, 2 * GIB, 4 * GIB) == GIB


def test_headroom_is_preserved_even_when_pressure_fallback_would_be_larger():
    value = runner()
    assert cuda_history_budget(value, 3 * GIB, 12 * GIB) == 3 * GIB - int(12 * GIB * .2)


def test_pressure_keeps_one_candidate_but_never_admits_unfit_workspace():
    value = runner(row=500 * MIB, workspace=100 * MIB)
    assert cuda_history_budget(value, GIB, 12 * GIB) == 600 * MIB
    with pytest.raises(MemoryError, match="history and workspace"):
        cuda_history_budget(value, 599 * MIB, 12 * GIB)
    with pytest.raises(MemoryError, match="history and workspace"):
        cuda_history_budget(value, GIB, 12 * GIB, owners=2)


def test_empty_history_does_not_reduce_dispatch_width():
    value = runner(row=0, workspace=0)
    assert cuda_history_budget(value, GIB, 12 * GIB) == 512 * MIB


def test_max_can_use_more_scratch_but_keeps_its_reserve_under_pressure():
    value = runner(row=2**20, workspace=0)
    assert cuda_history_budget(value, 2 * GIB, 8 * GIB, max_throughput=True) == (
        2 * GIB - int(8 * GIB * 0.10)
    )
    with pytest.raises(MemoryError, match="max-mode reserve"):
        cuda_history_budget(value, 512 * MIB, 8 * GIB, max_throughput=True)


@pytest.mark.parametrize("automatic,device", [(True, "cuda"), (False, "cuda"), (True, "mps")])
@pytest.mark.parametrize("fused", [False, True])
def test_new_owner_policy_preserves_fixed_and_metal_and_ignores_retired_facts(automatic, device, fused):
    from optimization.gpu.service import SharedAccountReplay

    proxy = object.__new__(SharedAccountReplay)
    proxy.run = proxy.data = None
    proxy._automatic_scratch_budget = automatic
    proxy._factual_capacity_hints = {"long": 9}
    proxy.shared_account_fused = fused
    proxy.sides = ["long", "short"]
    proxy.dispatch_batch_size = 128
    calls = []
    def memory():
        calls.append(True)
        return 10 * GIB, 12 * GIB
    proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=memory))
    value = runner(device=device)
    result = proxy._construct_runner("long", lambda *_args, **_kwargs: value, {})
    assert result is value
    assert getattr(value, "automatic_scratch_budget", False) == (automatic and device == "cuda")
    assert value.hsl_fact_capacity == 0
    assert not hasattr(value, "hsl_fact_capacity_learned")
    assert not hasattr(value, "initial_factual_capacity_batch_width")
    assert len(calls) == int(automatic and device == "cuda")
    if automatic and device == "cuda":
        expected = cuda_history_budget(value, 10 * GIB, 12 * GIB, owners=1 if fused else 2)
        assert value.hsl_scratch_budget_bytes == expected
    else:
        assert value.hsl_scratch_budget_bytes == 512 * MIB


def test_retired_fact_hints_do_not_inflate_forward_scratch():
    from optimization.gpu.service import SharedAccountReplay

    proxy = object.__new__(SharedAccountReplay)
    proxy.run = proxy.data = None
    proxy._automatic_scratch_budget = True
    proxy._factual_capacity_hints = {"long": 1024}
    proxy.shared_account_fused = False
    proxy.sides = ["long"]
    proxy.dispatch_batch_size = 4
    proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (10 * GIB, 12 * GIB)))
    value = runner()
    proxy._construct_runner("long", lambda *_args, **_kwargs: value, {})
    assert value.hsl_scratch_budget_bytes == 10 * GIB - int(12 * GIB * .2)


def test_memory_query_failure_propagates_without_fallback():
    from optimization.gpu.service import SharedAccountReplay

    proxy = object.__new__(SharedAccountReplay)
    proxy.run = proxy.data = None
    proxy._automatic_scratch_budget = True
    proxy._factual_capacity_hints = None
    def failed():
        raise RuntimeError("CUDA memory query failed")
    proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=failed))
    with pytest.raises(RuntimeError, match="memory query failed"):
        proxy._construct_runner("long", lambda *_args, **_kwargs: runner(), {})


@pytest.mark.parametrize("existing_stack,expected", [
    (1024, (78_144 - 1024) * 46 * 1536),
    (55_552, (78_144 - 55_552) * 46 * 1536),
    (78_144, 0),
    (78_160, 0),
])
def test_pending_stack_uses_selected_symbol_and_current_device_context(existing_stack, expected):
    from contextlib import contextmanager
    from optimization.gpu.cuda_kernel import CudaShaderLibrary

    active_devices = []

    @contextmanager
    def device(index):
        assert index == 1
        active_devices.append(index)
        try:
            yield
        finally:
            active_devices.pop()

    class Device:
        def __init__(self, index):
            self.context = device(index)

        def __enter__(self):
            self.context.__enter__()

        def __exit__(self, *args):
            self.context.__exit__(*args)

        @property
        def attributes(self):
            assert active_devices == [1]
            assert expected
            return {"MultiProcessorCount": 46, "MaxThreadsPerMultiProcessor": 1536}

    def get_limit(limit):
        assert limit == 0 and active_devices == [1]
        return existing_stack

    def get_function(name):
        assert name == "fused_replay" and active_devices == [1]
        return SimpleNamespace(attributes={"local_size_bytes": 78_144})

    library = object.__new__(CudaShaderLibrary)
    library._module = SimpleNamespace(get_function=get_function)
    library._cupy = SimpleNamespace(cuda=SimpleNamespace(
        Device=Device, runtime=SimpleNamespace(deviceGetLimit=get_limit),
    ))
    assert library.pending_stack_bytes("fused_replay", 1) == expected
    assert not active_devices


@pytest.fixture
def torch_runtime():
    return pytest.importorskip("torch")


SCRATCH_CACHES = (
    "_buffers", "_hsl_scratch_buffers", "_replay_states", "_sizes", "_full_end_steps",
)


def scratch_runner():
    from optimization.gpu.mps_kernel import _MulticoinReplayRunner

    value = object.__new__(_MulticoinReplayRunner)
    value.automatic_scratch_budget = True
    value.hsl_fact_capacity = 1
    value._replay_state_bytes = 16
    value._hsl_native_cache_bytes_per_candidate = lambda: 0
    for name in SCRATCH_CACHES:
        setattr(value, name, {})
    return value


@pytest.mark.usefixtures("torch_runtime")
def test_live_max_pressure_counts_reclaimable_scratch_and_preserves_saved_ceiling(monkeypatch):
    from optimization.gpu import mps_kernel

    value = scratch_runner()
    value._core_buffers = {5: SimpleNamespace(nbytes=64 * MIB)}
    value._buffers = {5: (SimpleNamespace(nbytes=64 * MIB),)}
    value._max_mode_scratch_ceiling_bytes = value.hsl_scratch_budget_bytes = 2 * GIB
    value._history_batch_bytes = lambda width: width * 64 * MIB + 64 * MIB
    value.bars = SimpleNamespace(device="cuda")
    free_bytes = GIB
    unused = 0
    queries = []
    def mem_get_info(device):
        assert device == "cuda"
        return free_bytes, 8 * GIB
    def empty_cache():
        queries.append(True)
    monkeypatch.setattr(mps_kernel.torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(mps_kernel.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_reserved", lambda _device: unused)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_allocated", lambda _device: 0)
    value._refresh_max_mode_scratch_budget(7)
    assert value.hsl_scratch_budget_bytes == GIB - int(8 * GIB * .10) + 128 * MIB
    free_bytes = 7 * GIB
    value._refresh_max_mode_scratch_budget(7)
    assert value.hsl_scratch_budget_bytes == 2 * GIB
    unused = 64 * MIB
    free_bytes = 256 * MIB
    with pytest.raises(MemoryError, match="fell below the max-mode replay reserve"):
        value._refresh_max_mode_scratch_budget(1)
    assert queries == [True]


@pytest.mark.usefixtures("torch_runtime")
def test_live_max_pressure_reclaims_allocator_reservations_before_reducing_width(monkeypatch):
    from optimization.gpu import mps_kernel

    value = scratch_runner()
    value._core_buffers = {}
    value._max_mode_scratch_ceiling_bytes = value.hsl_scratch_budget_bytes = 2 * GIB
    value._history_batch_bytes = lambda width: width * 64 * MIB + 64 * MIB
    value.bars = SimpleNamespace(device="cuda")
    free_bytes = GIB
    calls = []
    def mem_get_info(_device):
        calls.append(free_bytes)
        return free_bytes, 8 * GIB
    def empty_cache():
        nonlocal free_bytes
        free_bytes += 512 * MIB
    monkeypatch.setattr(mps_kernel.torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(mps_kernel.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_reserved", lambda _device: 768 * MIB)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_allocated", lambda _device: 256 * MIB)
    value._refresh_max_mode_scratch_budget(7)
    assert calls == [GIB, GIB + 512 * MIB]
    assert value.hsl_scratch_budget_bytes >= value._history_batch_bytes(7)


@pytest.mark.usefixtures("torch_runtime")
def test_pending_stack_is_subtracted_after_allocator_reclaim_and_can_reduce_width(monkeypatch):
    from optimization.gpu import mps_kernel

    value = scratch_runner()
    value._core_buffers = {}
    value._max_mode_scratch_ceiling_bytes = value.hsl_scratch_budget_bytes = 2 * GIB
    value._history_batch_bytes = lambda width: width * 64 * MIB + 64 * MIB
    value.bars = SimpleNamespace(device="cuda")
    free = 6 * GIB
    calls = []

    def mem_get_info(_device):
        calls.append(free)
        return free, 8 * GIB

    def empty_cache():
        nonlocal free
        free += 256 * MIB

    monkeypatch.setattr(mps_kernel.torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(mps_kernel.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_reserved", lambda _device: 256 * MIB)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_allocated", lambda _device: 0)
    pending = (78_144 - 1024) * 46 * 1536
    value._refresh_max_mode_scratch_budget(7, pending_stack_bytes=pending)
    assert calls == [6 * GIB, 6 * GIB + 256 * MIB]
    assert value.hsl_scratch_budget_bytes == free - int(8 * GIB * .1) - pending
    assert value._history_batch_bytes(5) <= value.hsl_scratch_budget_bytes < value._history_batch_bytes(6)


@pytest.mark.usefixtures("torch_runtime")
def test_pending_stack_does_not_credit_torch_scratch_that_cannot_be_reclaimed(monkeypatch):
    from optimization.gpu import mps_kernel
    from optimization.gpu.residency import CudaReplayMemoryPressure

    value = scratch_runner()
    value.cuda_max_scratch_policy = True
    value._core_buffers = {}
    value._replay_states = {5: SimpleNamespace(nbytes=256 * MIB)}
    value._max_mode_scratch_ceiling_bytes = value.hsl_scratch_budget_bytes = 2 * GIB
    value._history_batch_bytes = lambda width: width * 128 * MIB
    value.bars = SimpleNamespace(device="cuda")
    free = 2 * GIB
    calls = []

    def empty_cache():
        assert not value._replay_states
        calls.append(free)

    monkeypatch.setattr(mps_kernel.torch.cuda, "mem_get_info", lambda _: (free, 8 * GIB))
    monkeypatch.setattr(mps_kernel.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_reserved", lambda _: 256 * MIB)
    monkeypatch.setattr(mps_kernel.torch.cuda, "memory_allocated", lambda _: 256 * MIB)
    with pytest.raises(CudaReplayMemoryPressure, match="additional driver stack"):
        value._refresh_max_mode_scratch_budget(1, pending_stack_bytes=1152 * MIB)
    assert calls == [free]


@pytest.mark.usefixtures("torch_runtime")
def test_live_max_pressure_drops_old_scratch_if_needed_to_restore_reserve(monkeypatch):
    from optimization.gpu import mps_kernel

    value = scratch_runner()
    old_output = SimpleNamespace(nbytes=128 * MIB)
    value._core_output = old_output
    value._core_buffers = {5: old_output}
    value._replay_states = {5: SimpleNamespace(nbytes=128 * MIB)}
    value._scratch_shape = (5, 0, 0, 16)
    value.cuda_max_scratch_policy = True
    value._max_mode_scratch_ceiling_bytes = value.hsl_scratch_budget_bytes = 2 * GIB
    value._history_batch_bytes = lambda width: width * 64 * MIB
    value.bars = SimpleNamespace(device="cuda")
    free_bytes = 768 * MIB
    def empty_cache():
        nonlocal free_bytes
        assert not value._replay_states and not value._core_buffers
        assert value._scratch_shape is value._core_output is None
        free_bytes += 256 * MIB
    monkeypatch.setattr(mps_kernel.torch.cuda, "mem_get_info", lambda _device: (free_bytes, 8 * GIB))
    monkeypatch.setattr(mps_kernel.torch.cuda, "empty_cache", empty_cache)
    for name in ("memory_reserved", "memory_allocated"):
        monkeypatch.setattr(mps_kernel.torch.cuda, name, lambda _device: 256 * MIB)
    value._refresh_max_mode_scratch_budget(5)
    assert free_bytes == GIB
    assert value.hsl_scratch_budget_bytes == GIB - int(8 * GIB * .10)


@pytest.mark.parametrize("change", ["grow", "shrink", "state"])
@pytest.mark.usefixtures("torch_runtime")
def test_shape_changes_release_all_retained_scratch_and_reuse_unchanged_shapes(change):
    import weakref

    class Buffer:
        pass

    value = scratch_runner()
    value._prepare_scratch_shape(64)
    refs = []
    for name in SCRATCH_CACHES:
        buffers = getattr(value, name)
        buffers[64] = Buffer()
        refs.append(weakref.ref(buffers[64]))
    value.bars = Buffer()
    market = weakref.ref(value.bars)
    value._prepare_scratch_shape(64)
    assert all(ref() is not None for ref in refs)
    width = 64
    if change == "grow":
        width = 117
    elif change == "shrink":
        width = 32
    elif change == "facts":
        value.hsl_fact_capacity = 2
    elif change == "cache":
        value._hsl_native_cache_bytes_per_candidate = lambda: 32
    else:
        value._replay_state_bytes = 32
    value._prepare_scratch_shape(width)
    assert all(ref() is None for ref in refs)
    assert market() is value.bars


@pytest.mark.parametrize("policy", ["conservative", "cold_max"])
@pytest.mark.usefixtures("torch_runtime")
def test_physical_attempt_checks_scratch_before_first_device_allocation(monkeypatch, policy):
    import numpy as np
    import weakref
    from optimization.gpu import mps_kernel as kernel
    from optimization.gpu.residency import CudaReplayMemoryPressure

    class Buffer:
        nbytes = 0

    class AllocationReached(Exception):
        pass

    value = scratch_runner()
    value._prepare_scratch_shape(64)
    refs = []
    for name in SCRATCH_CACHES:
        buffer = Buffer()
        buffers = getattr(value, name)
        buffers[64] = (buffer,) if name in {"_buffers", "_hsl_scratch_buffers"} else buffer
        refs.append(weakref.ref(buffer))
    del buffer
    value._pack_params = np.asarray
    value._bootstrap_coin_overrides = ()
    value.interval_minutes = 1
    value.n = 73
    value._hourly_first_observed = value._hourly_coin_ranges = None
    value.requested_start_idx = 0
    value.run_config = SimpleNamespace(warmup_bars=1)
    monkeypatch.setattr(kernel, "pack_bootstrap_counts", lambda *_, **__: np.ones((117, 1, 2, 10), dtype=np.int32))
    # This unit test isolates scratch lifetime at the allocation boundary;
    # strategy/hourly admission has its own complete-layout regression tests.
    monkeypatch.setattr(kernel, "validate_hourly_bootstrap_inputs", lambda *_, **__: None)
    monkeypatch.setattr(kernel, "unstuck_required", lambda *_, **__: False)
    monkeypatch.setattr(kernel, "gpu_close_sizing_vendor", lambda: "nvidia")
    value._unstuck_ema_overrides = ()
    value.parameter_keys = ("realized_loss_memory_ema_span_minutes", "loss_gate_max_drawdown_fraction")
    value.loss_gate_enabled = True
    value.replay_sides = 1
    value.replay_kernel_name = "actual_fused_kernel"
    value._use_disabled_hsl_specialization = lambda _: False
    value._unstuck_ema_required = value._unstuck_required = lambda _: False
    value._library_cache_call = lambda: (None, ())
    value._prepare_replay_state_layout = lambda *_: None
    value._disable_native_cache_over_budget = lambda: False
    value._history_bytes_per_candidate = lambda: 0
    value.hsl_capacity = 0
    value.settings = [0.0] * 16
    value.settings[14] = 1.0  # Gate flag survives factual control writes.
    value.cuda_max_scratch_policy = policy == "cold_max"
    value.bars = SimpleNamespace(device=SimpleNamespace(index=1))
    value._core_buffers = {}
    value._max_mode_scratch_ceiling_bytes = 2 * GIB
    value._history_batch_bytes = lambda width: (width + 1) * 32 * MIB
    called = []

    def pending_stack(name, device):
        called.append((name, device))
        return (78_128 - 1024) * 46 * 1536

    library = SimpleNamespace(pending_stack_bytes=pending_stack)
    monkeypatch.setattr(kernel, "_cached_library_with_miss", lambda *_: (library, False))
    monkeypatch.setattr(kernel, "gpu_device", lambda: "cuda")
    monkeypatch.setattr(kernel.torch.cuda, "mem_get_info", lambda _: (4941 * MIB, 8 * GIB))
    monkeypatch.setattr(kernel.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(kernel.torch.cuda, "memory_allocated", lambda _: 0)
    def allocate(*_, **__):
        assert policy == "conservative", "replay allocations reached with insufficient driver stack backing"
        assert all(ref() is None for ref in refs)
        assert value.settings[14] == 1
        raise AllocationReached

    monkeypatch.setattr(kernel.torch, "as_tensor", allocate)
    candidates = np.tile([1440.0, 0.0], (117, 1))
    if policy == "cold_max":
        with pytest.raises(CudaReplayMemoryPressure, match="additional driver stack"):
            value._run_factual_attempt(candidates)
        assert called == [("actual_fused_kernel", 1)]
        assert all(ref() is not None for ref in refs)
    else:
        with pytest.raises(AllocationReached):
            value._run_factual_attempt(candidates)
        assert not called


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.usefixtures("torch_runtime")
def test_automatic_scratch_replacement_releases_old_binding_before_allocation(monkeypatch, automatic):
    import math
    from optimization.gpu import mps_kernel as kernel

    value = scratch_runner()
    value.automatic_scratch_budget = automatic
    value.native_factual_hsl = True
    value.hsl_capacity = value.hsl_scopes = 1
    value._history_batch_bytes = lambda width: width * 64 * MIB + 256 * MIB
    value.hsl_scratch_budget_bytes = cuda_history_budget(runner(), 10 * GIB, 12 * GIB)
    resident = [256 * MIB]  # Shared reduction workspace.
    class Tensor:
        def __init__(self, shape, dtype, **_):
            self.nbytes = math.prod(shape) * dtype.itemsize
            if resident[0] + self.nbytes > 10 * GIB:
                self.nbytes = 0
                raise MemoryError("retained and replacement histories overlap")
            resident[0] += self.nbytes
        def __del__(self):
            resident[0] -= self.nbytes
    monkeypatch.setattr(kernel, "_hsl_layout", lambda *_args, **_kwargs: (0, 2 * MIB))
    monkeypatch.setattr(kernel, "gpu_device", lambda: "cuda")
    monkeypatch.setattr(kernel.torch, "empty", Tensor)
    value._prepare_scratch_shape(64)
    value._hsl_buffers(64)
    assert resident[0] == 256 * MIB + 4 * GIB
    value._prepare_scratch_shape(117)
    if automatic:
        value._hsl_buffers(117)
        assert resident[0] == 256 * MIB + 117 * 64 * MIB
    else:
        with pytest.raises(MemoryError, match="overlap"):
            value._hsl_buffers(117)


@pytest.mark.usefixtures("torch_runtime")
def test_split_replay_drops_previous_raw_result_before_next_dispatch():
    import numpy as np
    import torch
    import weakref

    value = scratch_runner()
    value._history_batch_limit = lambda: 1
    previous = []
    def replay(*_args, **_kwargs):
        assert all(ref() is None for ref in previous)
        result = torch.ones(1)
        previous.append(weakref.ref(result))
        value.last_profile = {}
        return {"metric": result}
    value.run = replay
    output = value._run_history_batches(np.zeros((2, 1)), profile=False, end_steps=None)
    torch.testing.assert_close(output["metric"], torch.ones(2))
    assert all(ref() is None for ref in previous)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
@pytest.mark.usefixtures("torch_runtime")
def test_device_shape_release_preserves_native_replay(strategy, sides):
    import numpy as np
    import torch
    import weakref
    from test_gpu_hsl_multicoin import make_proxy, raw

    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU unavailable")
    proxy = make_proxy("portfolio", strategy, sides, minutes=128, factual_hsl=True)
    value = proxy.fused_runner or proxy.runners["long"]
    # Exercise the CUDA scratch-lifetime helper with the available device's
    # real allocator. Production only enables this flag for automatic CUDA.
    value.automatic_scratch_budget = True
    _, output = raw(proxy, [{}])
    expected = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in output.items()}
    del output
    for width in (3, 1):
        refs = []
        for name in SCRATCH_CACHES:
            for buffers in getattr(value, name).values():
                buffers = buffers if isinstance(buffers, tuple) else (buffers,)
                refs.extend(weakref.ref(buffer) for buffer in buffers)
        del buffers
        _, output = raw(proxy, [{}] * width)
        assert all(ref() is None for ref in refs)
        for key, reference in expected.items():
            actual = output[key]
            if isinstance(reference, torch.Tensor):
                torch.testing.assert_close(actual.cpu(), reference.expand_as(actual),
                                           rtol=0, atol=0, equal_nan=True)
            else:
                assert actual == reference
        del output


@pytest.mark.usefixtures("torch_runtime")
@pytest.mark.parametrize("same_shape,allowance,expected_width", [
    (True, 20, 4),
    (False, 30, 2),
    (False, 19, 0),
])
def test_metal_max_scratch_reuses_exact_shapes_or_limits_before_replay(
    monkeypatch, same_shape, allowance, expected_width,
):
    from optimization.gpu import mps_kernel, residency
    from optimization.gpu.autotune import ReplayScratchLimit

    value = object.__new__(mps_kernel._MulticoinReplayRunner)
    value.mps_max_scratch_policy = True
    value._replay_state_bytes = 10
    value._max_mode_parameter_cols = 12
    value._max_mode_scratch_ceiling_bytes = 100
    value._scratch_shape = (4 if same_shape else 5, 12, 10)
    value._replay_states = {value._scratch_shape: SimpleNamespace(nbytes=30)}
    value._buffers = {}
    value._hsl_scratch_buffers = {}
    value._sizes = {}
    value._full_end_steps = {}
    value._core_buffers = {}
    value._core_output = None
    value._history_bytes_per_candidate = lambda: 10
    value._history_workspace_bytes = lambda: 10
    samples, reclaims = [], []

    def available(_torch, *, pending_host_bytes):
        assert pending_host_bytes == 0
        samples.append(bool(value._replay_states))
        return allowance

    monkeypatch.setattr(residency, "mps_max_available_bytes", available)
    monkeypatch.setattr(mps_kernel.torch.mps, "empty_cache", lambda: reclaims.append(True))
    if expected_width == 4:
        value._admit_mps_max_scratch(4)
        assert samples == [True]
        assert value._scratch_shape == (4, 12, 10)
        assert value._replay_states
        assert not reclaims
    else:
        error = ReplayScratchLimit if expected_width else residency.MetalReplayMemoryPressure
        with pytest.raises(error) as caught:
            value._admit_mps_max_scratch(4)
        assert samples == [False, False]
        assert value._scratch_shape is None
        assert not value._replay_states
        assert len(reclaims) == 1
        if expected_width:
            assert caught.value.limit == expected_width
