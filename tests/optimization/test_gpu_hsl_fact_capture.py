"""Actual native fill paths own finite HSL facts; no controller replacement yet."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU required"
)


@pytest.fixture(scope="module")
def reference():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return passivbot_rust


def _captured(runner, batch_size, *, with_positions=False):
    from optimization.gpu.mps_kernel import _hsl_layout
    assert runner.hsl_fact_capacity > 0
    _, offset = _hsl_layout(runner.hsl_capacity)
    # Full scratch readback is exclusively a test inspector. Normal runs return
    # the existing modest metric payload; facts remain resident on the worker.
    scratch = runner._hsl_buffers(batch_size)[0].cpu().numpy()
    dtype = np.dtype([
        ("delta", "<f4"), ("price", "<f4"), ("realized", "<f4"), ("fee", "<f4"),
        ("minute", "<i4"), ("first_sequence", "<i4"), ("last_sequence", "<i4"),
        ("actual_size_after", "<f4"),
    ])
    result = []
    positions = []
    for candidate in scratch:
        scopes = []
        endpoints = []
        for raw in candidate:
            metadata = np.frombuffer(raw[offset:offset+2].tobytes(), dtype="<i4", count=6)
            head, count, _, failure, _, _ = metadata
            assert failure == 0
            records = np.frombuffer(raw[offset+2:].tobytes(), dtype=dtype,
                                    count=runner.hsl_fact_capacity)
            scopes.append(records[(head+np.arange(count)) % runner.hsl_fact_capacity].copy())
            endpoints.append(np.frombuffer(raw[offset:offset+2].tobytes(), dtype="<f4")[12:14].copy())
        result.append(scopes)
        positions.append(endpoints)
    return (result, np.asarray(positions)) if with_positions else result


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_actual_fill_capture_preserves_outputs_and_complete_global_chronology(
    reference, strategy, sides, mode,
):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy(mode, strategy, sides, minutes=256)
    runner, baseline = raw(proxy, [{}, {}])
    assert float(baseline["fill_count"].min()) > 0
    runner.hsl_fact_capacity = 64
    _, captured = raw(proxy, [{}, {}])
    for key, expected in baseline.items():
        if isinstance(expected, torch.Tensor):
            np.testing.assert_array_equal(captured[key].cpu().numpy(), expected.cpu().numpy(), err_msg=key)
        else:
            assert captured[key] == expected, key
    rows, positions = _captured(runner, 2, with_positions=True)
    for candidate, scopes in enumerate(rows):
        count = 0
        chronology = []
        gross = 0.
        for scope, records in enumerate(scopes):
            if scope % (runner.n_coins+1) == 0:
                assert len(records) == 0  # Aggregate controllers borrow pair facts.
                np.testing.assert_array_equal(positions[candidate, scope], 0.)
                continue
            if not len(records):
                np.testing.assert_array_equal(positions[candidate, scope], 0.)
                continue
            assert np.isfinite(records["price"]).all()
            assert (records["price"] > 0).all()
            assert np.isfinite(records["delta"]).all()
            assert (records["delta"] != 0).all()
            assert np.all(np.diff(records["minute"]) >= 0)
            assert np.all(records["first_sequence"][1:] > records["last_sequence"][:-1])
            side = (sides[0] if len(sides) == 1 else
                    "long" if scope < runner.n_coins+1 else "short")
            assert np.all(records["actual_size_after"] >= 0 if side == "long"
                          else records["actual_size_after"] <= 0)
            assert positions[candidate, scope, 0] == records[-1]["actual_size_after"]
            direction = 1 if side == "long" else -1
            basis = 0.
            for record in records:
                delta = direction * float(record["delta"])
                after = abs(float(record["actual_size_after"]))
                if after == 0.:
                    basis = 0.
                elif delta > 0.:
                    before = after - delta
                    basis = basis * (before / after) + float(record["price"]) * (delta / after)
            np.testing.assert_allclose(positions[candidate, scope, 1], basis,
                                       rtol=2e-5, atol=2e-4)
            count += int(np.sum(records["last_sequence"]-records["first_sequence"]+1))
            chronology.extend(range(int(first), int(last)+1) for first, last in
                              zip(records["first_sequence"], records["last_sequence"]))
            gross += float(records["realized"].sum(dtype=np.float64))
        expected_count = int(captured["fill_count"][candidate])
        assert count == expected_count
        assert sorted(sequence for run in chronology for sequence in run) == list(range(expected_count))
        expected_gross = float(captured["profit_sum"][candidate]-captured["loss_sum"][candidate])
        np.testing.assert_allclose(gross, expected_gross, rtol=2e-5, atol=2e-4)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_capture_overflow_rejects_result_without_truncating_history(reference, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("coin", strategy, ("long",), minutes=512)
    runner, baseline = raw(proxy, [{}])
    assert float(baseline["fill_count"][0]) > 2
    runner.hsl_fact_capacity = 1
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()
    with pytest.raises(RuntimeError, match="GPU HSL factual history overflow"):
        raw(proxy, [{}])


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_factual_storage_is_in_physical_dispatch_budget(reference, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("unified", strategy, ("long", "short"), minutes=256)
    runner, baseline = raw(proxy, [{}, {}])
    original_cost = runner._history_bytes_per_candidate()
    runner.hsl_fact_capacity = 1024
    assert runner._history_bytes_per_candidate() == original_cost + runner.hsl_scopes*1538*32
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()
    _, captured = raw(proxy, [{}, {}])
    for key, value in baseline.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(captured[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert captured[key] == value, key
    assert list(runner._hsl_scratch_buffers) == [(1, runner.hsl_fact_capacity)]
    records = _captured(runner, 1)
    assert any(len(scope) for scope in records[0])


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_temporal_rebinding_preserves_complete_factual_order(reference, sides, mode):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy(mode, "trailing_martingale", sides, minutes=256)
    runner, _ = raw(proxy, [{}, {}])
    runner.hsl_fact_capacity = 64
    _, reference_outputs = raw(proxy, [{}, {}])
    records, endpoints = _captured(runner, 2, with_positions=True)
    runner.max_dispatch_candidate_bars = 100
    _, chunked = raw(proxy, [{}, {}])
    for key, value in reference_outputs.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(chunked[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert chunked[key] == value, key
    chunked_records, chunked_endpoints = _captured(runner, 2, with_positions=True)
    np.testing.assert_array_equal(chunked_endpoints, endpoints)
    for original, changed in zip(records, chunked_records):
        for original_scope, changed_scope in zip(original, changed):
            np.testing.assert_array_equal(original_scope, changed_scope)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_gpu_capacity_growth_preserves_results_and_is_reused(reference, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("unified", strategy, ("long", "short"), minutes=256)
    runner, baseline = raw(proxy, [{}, {}])
    runner.hsl_fact_capacity = 1
    initial_library = runner._library_cache_call()
    _, grown = raw(proxy, [{}, {}], profile=True)
    assert runner.hsl_fact_capacity > 1
    assert runner.last_hsl_fact_retries > 0
    assert runner.last_profile["hsl_fact_retry_count"] == runner.last_hsl_fact_retries
    assert runner.last_profile["hsl_fact_retry_seconds"] > 0
    learned = runner.hsl_fact_capacity
    assert runner._library_cache_call() == initial_library
    for key, value in baseline.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(grown[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert grown[key] == value, key
    _, repeated = raw(proxy, [{}, {}])
    assert runner.hsl_fact_capacity == learned
    assert runner.last_hsl_fact_retries == 0
    for key, value in grown.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(repeated[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert repeated[key] == value, key


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
def test_actual_overflow_checks_interrupt_before_retry(reference, strategy, sides):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("unified", strategy, sides, minutes=256)
    runner, _ = raw(proxy, [{}])
    runner.hsl_fact_capacity = 1
    calls = []
    def interrupt():
        calls.append(runner.hsl_fact_capacity)
        raise KeyboardInterrupt
    runner.interrupt_check = interrupt
    with pytest.raises(KeyboardInterrupt):
        raw(proxy, [{}])
    assert calls and calls[-1] == 1
    assert runner.hsl_fact_capacity == 1
    assert not getattr(runner, "hsl_fact_retry_count_total", 0)
    # A later request replays from factual initial state after cancellation.
    runner.interrupt_check = lambda: None
    _, recovered = raw(proxy, [{}])
    assert float(recovered["fill_count"][0]) > 2
    assert runner.hsl_fact_capacity > 1


@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
def test_temporal_overflow_marker_survives_later_dispatches(reference, sides):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("unified", "trailing_martingale", sides, minutes=512)
    runner, baseline = raw(proxy, [{}])
    assert float(baseline["fill_count"][0]) > 2
    runner.hsl_fact_capacity = 1
    runner.max_dispatch_candidate_bars = 64
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()
    with pytest.raises(RuntimeError, match="single-candidate scratch budget"):
        raw(proxy, [{}])
    assert runner._last_temporal_dispatch["dispatch_count"] > 1


def test_temporal_capacity_growth_restarts_factual_state_and_preserves_outputs(reference):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("unified", "trailing_martingale", ("long", "short"), minutes=256)
    runner, baseline = raw(proxy, [{}])
    runner.hsl_fact_capacity = 1
    runner.max_dispatch_candidate_bars = 128
    _, grown = raw(proxy, [{}])
    assert runner.hsl_fact_capacity > 1
    assert runner.last_hsl_fact_retries > 0
    for key, value in baseline.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(grown[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert grown[key] == value, key
    records = _captured(runner, 1)[0]
    ordinal_ranges = [(int(record["first_sequence"]), int(record["last_sequence"]))
                      for scope in records for record in scope]
    assert sorted(sequence for first, last in ordinal_ranges for sequence in range(first, last+1)) == list(range(int(grown["fill_count"][0])))
