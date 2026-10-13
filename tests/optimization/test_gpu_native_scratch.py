"""Native CUDA capacity uses resident data, factual history and metric workspace."""

from types import SimpleNamespace

import pytest

from optimization.gpu.autotune import cuda_history_budget, history_dispatch_ceiling

GIB = 1024**3
MIB = 1024**2


def runner(*, row=64 * MIB, workspace=256 * MIB, device="cuda"):
    value = SimpleNamespace(
        bars=SimpleNamespace(device=SimpleNamespace(type=device)),
        native_factual_hsl=True,
        hsl_fact_capacity_learned=1,
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


def test_allowance_leaves_room_for_fact_growth_but_dispatch_still_caps_width():
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


@pytest.mark.parametrize("automatic,device", [(True, "cuda"), (False, "cuda"), (True, "mps")])
@pytest.mark.parametrize("fused", [False, True])
def test_new_owner_policy_preserves_fixed_and_metal_and_learned_facts(automatic, device, fused):
    from optimization.gpu.service import MpsMulticoinProxy

    proxy = object.__new__(MpsMulticoinProxy)
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
    assert value.hsl_fact_capacity == value.hsl_fact_capacity_learned == 9
    assert len(calls) == int(automatic and device == "cuda")
    if automatic and device == "cuda":
        expected = cuda_history_budget(value, 10 * GIB, 12 * GIB, owners=1 if fused else 2)
        assert value.hsl_scratch_budget_bytes == expected
    else:
        assert value.hsl_scratch_budget_bytes == 512 * MIB


def test_learned_fact_capacity_is_sized_before_legacy_limit_check():
    from optimization.gpu.service import MpsMulticoinProxy

    proxy = object.__new__(MpsMulticoinProxy)
    proxy.run = proxy.data = None
    proxy._automatic_scratch_budget = True
    proxy._factual_capacity_hints = {"long": 1024}
    proxy.shared_account_fused = False
    proxy.sides = ["long"]
    proxy.dispatch_batch_size = 4
    proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (10 * GIB, 12 * GIB)))
    value = runner()
    value._history_batch_bytes = lambda width: width * value.hsl_fact_capacity * MIB
    proxy._construct_runner("long", lambda *_args, **_kwargs: value, {})
    assert value.hsl_scratch_budget_bytes == 10 * GIB - int(12 * GIB * .2)


def test_memory_query_failure_propagates_without_fallback():
    from optimization.gpu.service import MpsMulticoinProxy

    proxy = object.__new__(MpsMulticoinProxy)
    proxy.run = proxy.data = None
    proxy._automatic_scratch_budget = True
    proxy._factual_capacity_hints = None
    def failed():
        raise RuntimeError("CUDA memory query failed")
    proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=failed))
    with pytest.raises(RuntimeError, match="memory query failed"):
        proxy._construct_runner("long", lambda *_args, **_kwargs: runner(), {})


SCRATCH_CACHES = (
    "_buffers", "_hsl_scratch_buffers", "_unstuck_pnl_buffers", "_recovery_buffers",
    "_volume_buffers", "_weighted_equity_buffers", "_hsl_ema_tail_buffers",
    "_equity_balance_diff_buffers", "_entry_interval_stat_buffers",
    "_entry_interval_count_buffers", "_replay_states", "_sizes", "_full_end_steps",
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


@pytest.mark.parametrize("change", ["grow", "shrink", "facts", "cache", "state"])
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


def test_physical_attempt_releases_scratch_before_first_device_allocation(monkeypatch):
    import numpy as np
    import weakref
    from optimization.gpu import mps_kernel as kernel

    class Buffer:
        pass

    class AllocationReached(Exception):
        pass

    value = scratch_runner()
    value._prepare_scratch_shape(64)
    refs = []
    for name in SCRATCH_CACHES:
        buffers = getattr(value, name)
        buffers[64] = Buffer()
        refs.append(weakref.ref(buffers[64]))
    value._pack_params = np.asarray
    value._use_disabled_hsl_specialization = lambda _: False
    value._unstuck_ema_required = value._unstuck_required = lambda _: False
    value._library_cache_call = lambda: (None, ())
    value._prepare_replay_state_layout = lambda *_: None
    value._disable_native_cache_over_budget = lambda: False
    value._history_bytes_per_candidate = lambda: 0
    value.hsl_capacity = 0
    value.settings = [0, 0]
    monkeypatch.setattr(kernel, "_cached_library_with_miss", lambda *_: (None, False))
    monkeypatch.setattr(kernel, "gpu_device", lambda: "cuda")
    def allocate(*_, **__):
        assert all(ref() is None for ref in refs)
        raise AllocationReached
    monkeypatch.setattr(kernel.torch, "as_tensor", allocate)
    with pytest.raises(AllocationReached):
        value._run_factual_attempt(np.zeros((117, 1)))


@pytest.mark.parametrize("automatic", [False, True])
def test_real_hsl_allocator_does_not_overlap_64_and_117_candidate_histories(monkeypatch, automatic):
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
def test_device_shape_release_preserves_native_replay(strategy, sides):
    import numpy as np
    import torch
    import weakref
    from test_gpu_hsl_multicoin import make_proxy, raw

    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU unavailable")
    proxy = make_proxy("unified", strategy, sides, minutes=128, factual_hsl=True)
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
