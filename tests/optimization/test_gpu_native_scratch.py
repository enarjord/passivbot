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
