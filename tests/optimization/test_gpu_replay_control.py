"""Packed readonly control input preserves per-attempt stops and kernel ABI."""
from pathlib import Path
from types import SimpleNamespace
import re

import pytest
torch = pytest.importorskip("torch")

from optimization.gpu import mps_kernel
from optimization.gpu.cuda_kernel import cuda_source


@pytest.mark.parametrize("recovery", [False, True])
def test_replay_control_is_fresh_with_changed_stops_at_identical_batch_size(recovery):
    prefix = torch.tensor([3, 20, 2, 1, 0, 0, 0, 0] + ([1, 21] if recovery else []), dtype=torch.int32)
    ends = torch.tensor([3, 7, 5], dtype=torch.int32)
    first = mps_kernel._pack_replay_control(prefix, ends, recovery)
    ends.copy_(torch.tensor([8, 4, 6], dtype=torch.int32))
    second = mps_kernel._pack_replay_control(prefix, ends, recovery)
    assert first.tolist() == prefix.tolist() + [3, 7, 5]
    assert second.tolist() == prefix.tolist() + [8, 4, 6]
    assert first.data_ptr() != second.data_ptr()
    assert second.is_contiguous()
    assert second.numel() * second.element_size() == 4 * (len(prefix) + len(ends))
    prefix[1] = 99
    assert first[1].item() == second[1].item() == 20


@pytest.mark.parametrize("kind", ["prefix", "dtype", "end_shape"])
def test_replay_control_rejects_inconsistent_layout(kind):
    sizes = torch.zeros(8, dtype=torch.int32)
    ends = torch.ones(2, dtype=torch.int32)
    if kind == "prefix": sizes = sizes[:7]
    if kind == "dtype": ends = ends.to(torch.float32)
    if kind == "end_shape": ends = ends.reshape(1, 2)
    with pytest.raises(ValueError, match="replay control"):
        mps_kernel._pack_replay_control(sizes, ends, False)


RUNNERS = [
    mps_kernel.MpsEmaAnchorMulticoinRunner,
    mps_kernel.MpsEmaAnchorMulticoinFusedRunner,
    mps_kernel.MpsTrailingMartingaleMulticoinRunner,
    mps_kernel.MpsTrailingMartingaleMulticoinFusedRunner,
]


@pytest.mark.parametrize("runner_cls", RUNNERS)
@pytest.mark.parametrize("recovery", [False, True])
def test_maximum_feature_temporal_dispatch_repackages_stops_and_preserves_argument_order(monkeypatch, runner_cls, recovery):
    # Exercise the real Python temporal loop on CPU tensors with a recording
    # producer; this verifies the control ABI, not financial/device parity.
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "cpu")
    monkeypatch.setattr(mps_kernel, "synchronize", lambda: None)
    monkeypatch.setattr(mps_kernel, "record_replay_chunk", lambda *a, **k: None)
    monkeypatch.setattr(mps_kernel, "TemporalReplayProgress", lambda *a, **k: SimpleNamespace(log=lambda *a, **k: None))
    runner = runner_cls.__new__(runner_cls)
    tm = "TrailingMartingale" in runner_cls.__name__
    fused = "Fused" in runner_cls.__name__
    names = ["market_execution_ticks", "bars", "fill_ticks", "touch_ticks",
             "touch_nearest_ticks", "touch_min_qty_bits", "touch_min_qty_relation",
             "hour_log_ranges", "coin_settings", "coin_overrides", "short_coin_overrides", "settings", "btc_prices"]
    runner.__dict__.update({n: object() for n in names})
    runner.bars = SimpleNamespace(device=SimpleNamespace(type="mps"))
    runner.organillo_buffers = (object(), object())
    runner.btc_prices_enabled = runner.equity_balance_diff_enabled = True
    runner.entry_interval_enabled = tm
    runner.recovery_distribution_enabled = recovery
    runner.hsl_capacity = 2
    runner.native_factual_hsl = False  # Fixed temporal bounds in this ABI-only test.
    runner.max_dispatch_candidate_bars = 10000
    runner.temporal_chunk_bar_limit = 2
    runner.n_coins = 2
    runner.hsl_scratch_budget_bytes = 10000
    runner._replay_state_bytes, runner._replay_states = 4, {}
    runner.interrupt_check = lambda: None
    runner._temporal_replay_enabled = lambda: True
    runner._history_batch_bytes = lambda batch: 0
    runner._hsl_ema_tail_samples_enabled = lambda: True
    hsl = (object(), object())
    runner._hsl_buffers = lambda batch: hsl
    params, bootstrap = object(), object()
    daily, gaps, counts, equity, intervals, interval_counts, recovery_samples, tail = [object() for _ in range(8)]
    scalars = torch.zeros((2, 64))
    sizes = torch.tensor([2, 20, 2, 1, 0, 0, 0, 0] + ([1, 21] if recovery else []), dtype=torch.int32)
    input_order = [runner.market_execution_ticks, runner.bars, runner.fill_ticks, runner.touch_ticks]
    if tm: input_order += [runner.touch_nearest_ticks, runner.touch_min_qty_bits, runner.touch_min_qty_relation]
    input_order += [runner.hour_log_ranges, runner.coin_settings, *runner.organillo_buffers, runner.coin_overrides]
    if fused: input_order += [runner.short_coin_overrides]
    input_order += [params, bootstrap, runner.settings]
    after_control = [runner.btc_prices, equity]
    if tm: after_control += [intervals, interval_counts]
    after_control += [daily, scalars, gaps, counts]
    if recovery: after_control += [recovery_samples]
    after_control += [tail, *hsl]
    calls = []
    def kernel(*args, **kwargs):
        control_index = len(input_order)
        assert all(a is b for a, b in zip(args[:control_index], input_order))
        assert all(a is b for a, b in zip(args[control_index + 1:-2], after_control))
        calls.append((args[control_index].clone(), args[-1].tolist(), len(args)))
    library = SimpleNamespace(**{runner.replay_kernel_name: kernel})
    stops = ([3, 5], [4, 6])
    offset = len(sizes)
    for ends in stops:
        start = len(calls)
        runner._dispatch(library, params, bootstrap, sizes, torch.tensor(ends, dtype=torch.int32),
                         daily, scalars, gaps, counts, equity, intervals, interval_counts,
                         recovery_samples, tail, batch_size=2)
        attempt = calls[start:]
        assert [row[1] for row in attempt] == ([[1, 3], [3, 5]] if ends[-1] == 5 else [[1, 3], [3, 5], [5, 6]])
        assert all(row[0][:offset].tolist() == sizes.tolist() and row[0][offset:].tolist() == ends for row in attempt)
        assert runner._last_temporal_dispatch["kernel_candidate_steps"] == sum(k - 1 for k in ends)
    expected_count = (31 if tm and fused else 30 if tm else 26 if fused else 25) - (not recovery)
    assert all(row[2] == expected_count <= 31 for row in calls)


def _maximum_signature(strategy, suffix, recovery):
    root = Path(__file__).resolve().parents[2]
    filename = "mps_ema_anchor_multicoin_long.metal" if strategy == "ema_anchor" else "mps_trailing_martingale_multicoin.metal"
    source = (root / "passivbot-rust/src/gpu" / filename).read_text()
    name = "passivbot_" + strategy + "_multicoin" + suffix
    signature = re.search(r"kernel void " + name + r"\(([\s\S]*?)\)\s*\{", source)[1]
    lines = [line for line in signature.splitlines() if not line.lstrip().startswith("#")]
    if not recovery: lines = [line for line in lines if "recovery_samples" not in line]
    signature = "\n".join(lines).replace("ORGANILLO_ARGS", ", constant uint* organillo_values, constant int* organillo_rows")
    return name, signature


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suffix", ["", "_long", "_fused"])
@pytest.mark.parametrize("recovery", [False, True])
def test_cuda_lowered_maximum_feature_control_argument_order(strategy, suffix, recovery):
    name, signature = _maximum_signature(strategy, suffix, recovery)
    source = "kernel void " + name + "(" + signature + ") {\nconstant int* end_steps = sizes + " + str(10 if recovery else 8) + ";\n}"
    lowered = cuda_source(source)
    actual = re.search(r'extern "C" __global__ void ' + name + r"\(([\s\S]*?)\)\s*\{", lowered)[1]
    formal = [x.strip().split()[-1].lstrip("*&") for x in actual.split(",")]
    assert "end_steps" not in formal
    assert formal[formal.index("sizes") + 1] == "btc_prices"
    assert formal[-1] == "passivbot_dispatch_count"
    assert len(formal) - 1 <= 31
    assert "const int* end_steps = sizes + " + str(10 if recovery else 8) in lowered


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_exposure_rms_budget_calls_match_ungated_helper_contract(strategy):
    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    common = (root / "mps_multicoin_common.metal").read_text()
    declaration = re.search(r"inline void update_exposure_rms_budgets\(([\s\S]*?)\)\s*\{", common)[1]
    # RMS budget observation follows tradability, independently of entry
    # selection/Organillo. Passing eligibility buffers changes its call ABI.
    assert len(declaration.split(",")) == 10 and "ORGANILLO_ARGS" not in declaration
    name = "mps_ema_anchor_multicoin_long.metal" if strategy == "ema_anchor" else "mps_trailing_martingale_multicoin.metal"
    calls = re.findall(r"update_exposure_rms_budgets\(([^;]*?)\);", (root / name).read_text())
    assert len(calls) == 3  # two fused sides and the directional implementation
    assert all("ORGANILLO_PASS" not in call and len(call.split(",")) == 10 for call in calls)
    assert all(call.split(",")[2].strip() == "coin_settings" for call in calls)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires the real Apple Metal compiler")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("recovery", [False, True])
def test_real_mps_compiles_maximum_feature_fused_temporal_control(strategy, recovery):
    # Bypass the process library cache and resolve the kernel: Metal creates
    # the compute pipeline lazily at lookup, not merely at source assembly.
    features = dict(
        hsl_ema_tail_enabled=True, hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True, recovery_distribution_enabled=recovery,
        dynamic_wel_by_tradability=True, btc_risk_enabled=True,
        equity_balance_diff_enabled=True,
        temporal_chunking=True, hsl_capacity=2, hsl_lookback=0,
        mps_coin_capacity=2, raw_strategy_risk_enabled=True,
        raw_strategy_growth_enabled=True, unstuck_ema_enabled=True,
        hsl_factual_only=True, unstuck_enabled=True,
        exposure_rms_enabled=True, organillo_enabled=True,
    )
    if strategy == "trailing_martingale":
        features.update(entry_interval_enabled=True, coin_overrides_absent=False)
        loader = mps_kernel._trailing_martingale_multicoin_shader_library
    else:
        loader = mps_kernel._ema_anchor_multicoin_shader_library
    library = loader.__wrapped__(**features)
    kernel = getattr(library, "passivbot_" + strategy + "_multicoin_fused")
    assert kernel.max_threads_per_threadgroup > 0
    assert kernel.thread_execution_width > 0
