from config.universe import set_side_universe_sources
import ast
from functools import lru_cache
import inspect
import textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.runtime import (
    gpu_device,
    compile_shader as _compile_shader,
    synchronize,
)


def compile_shader(source):
    # Standalone math probes share the full kernel source but do not allocate
    # an HSL history window. Production runners specialize these dimensions.
    if "struct HslState" in source and "#define PASSIVBOT_HSL_CAPACITY" not in source:
        source = (
            "#define PASSIVBOT_HSL_CAPACITY 1\n#define PASSIVBOT_HSL_TREE_SIZE 1\n#define PASSIVBOT_HSL_LOOKBACK 0\n"
            + source
        )
    return _compile_shader(source)


GPU_AVAILABLE = torch.backends.mps.is_available() or torch.cuda.is_available()

from optimization.gpu.model import (
    GAP_BINS,
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    _build_hourly_log_range,
    _directional_touch_ticks,
    _maximum_effective_min_cost,
    _minimum_entry_qty_encoding,
    _strict_fill_tick_boundaries,
    build_mps_multicoin_data,
)
from optimization.gpu.mps_kernel import (
    _pack_replay_control,
    MPS_MULTICOIN_FUSED_SCALAR_COLS,
    MpsEmaAnchorMulticoinRunner,
    MpsEmaAnchorMulticoinFusedRunner,
    MpsTrailingMartingaleMulticoinFusedRunner,
    _decode_entry_interval_outputs,
    _cached_library_with_miss,
    _decode_multicoin_fused_outputs,
    _encode_max_realized_loss_pct,
    _pack_tm_parameter_matrix,
    _TM_DECIMAL_WORDS,
    _scale_directional_minute_parameters,
    _scale_ema_multicoin_coin_overrides,
    _scale_tm_multicoin_coin_overrides,
    _with_btc_risk,
    _with_dynamic_wel_by_tradability,
    _with_entry_interval,
    _hsl_layout,
    _with_hsl,
    _with_hsl_ema_tail,
    _with_hsl_features,
    _with_recovery_distribution,
    _with_tm_loss_gate_disabled,
    strategy_eq_recovery_distribution_from_samples,
)
from optimization.gpu.metrics import (
    _entry_interval_metrics,
    _fill_gap_metrics,
    compute_objectives,
)
from optimization.gpu.service import SharedAccountReplay


def _tm_probe_tensor(values, **kwargs):
    """Direct shader probes use the same private parameter packing as replay."""
    raw = np.asarray(values, dtype=np.float64)
    width = raw.shape[-1]
    if width in (len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS),
                 2 * len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)):
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    elif width in (len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS),
                   2 * len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)):
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    else:
        return torch.tensor(values, **kwargs)
    packed = _pack_tm_parameter_matrix(raw.reshape(-1, width), keys, sides=width // len(keys))
    return torch.tensor(packed, **kwargs)


def _tm_probe_full(shape, value, **kwargs):
    from types import SimpleNamespace
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner

    raw = np.full(shape, value, dtype=np.float64)
    packed = MpsTrailingMartingaleMulticoinRunner._prepare_coin_overrides(
        SimpleNamespace(interval_minutes=1.0), raw)
    return torch.tensor(packed, **kwargs)


@pytest.mark.parametrize(
    ("runner_cls", "dispatch_name"),
    [
        (MpsEmaAnchorMulticoinRunner, "_dispatch"),
    ],
)
def test_strategy_runner_valid_evaluation_has_one_kernel_dispatch(
    runner_cls, dispatch_name
):
    method = runner_cls._run_factual_attempt
    source = inspect.getsource(method)
    tree = ast.parse(textwrap.dedent(source))
    dispatch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == dispatch_name)
            or (
                isinstance(node.func, ast.Attribute) and node.func.attr == dispatch_name
            )
        )
    ]

    assert len(dispatch_calls) == 1
    assert "high_exposure" not in source


def test_profiled_library_classifies_shared_cache_miss_not_runner_age():
    @lru_cache(maxsize=1)
    def loader(value):
        return object()

    first, first_missed = _cached_library_with_miss(loader, 1)
    second, second_missed = _cached_library_with_miss(loader, 1)

    assert first is second
    assert first_missed is True
    assert second_missed is False


def _assert_fill_scalar_contract(output):
    fill_count = output["fill_count"]
    assert torch.equal(fill_count, fill_count.round())
    assert (fill_count >= 0.0).all()
    assert (fill_count >= output["fill_count_entry"]).all()
    assert (fill_count >= output["fill_count_long"]).all()
    assert torch.equal(fill_count, output["day_fill_count"].sum(dim=1))
    active_days = output["fills_active_days_count"]
    assert torch.equal(active_days, active_days.round())
    assert (active_days >= 0.0).all()
    duration_days = (output["last_eq_ts"] - output["first_eq_ts"]).clamp(
        min=0.0
    ) / 86_400_000.0
    assert (active_days <= duration_days.ceil().clamp(min=1.0)).all()
    pnl_recovery = output["pnl_recovery_max_ms"]
    assert (pnl_recovery >= 0.0).all()
    no_fills = fill_count == 0.0
    assert (pnl_recovery[no_fills] == 0.0).all()
    held_count = output["held_count"]
    held_sum = output["held_sum_ms"]
    assert torch.equal(held_count, held_count.round())
    assert (held_count >= 0.0).all()
    assert (held_sum >= 0.0).all()
    assert (held_sum[held_count == 0.0] == 0.0).all()
    if (held_count > 0.0).any():
        assert (
            held_sum[held_count > 0.0]
            <= output["held_max_ms"][held_count > 0.0] * held_count[held_count > 0.0]
        ).all()
    has_fills = ~no_fills
    if has_fills.any():
        first_fill_step = torch.round(output["first_fill_ts"] / 60_000.0)
        last_eq_step = torch.round(output["last_eq_ts"] / 60_000.0)
        recovery_steps = torch.round(pnl_recovery / 60_000.0)
        assert (
            recovery_steps[has_fills]
            <= (last_eq_step - first_fill_step).clamp(min=0.0)[has_fills]
        ).all()
    account_recovery = output["account_recovery_max_ms"]
    assert (account_recovery >= 0.0).all()
    equity_span = output["last_eq_ts"] - output["first_eq_ts"]
    has_equity = torch.isfinite(equity_span) & (equity_span >= 0.0)
    assert (account_recovery[has_equity] <= equity_span[has_equity] + 1.0e-6).all()
    assert (account_recovery[~has_equity] == 0.0).all()


def test_tm_parameter_packing_preserves_positive_underflow_mode_per_side():
    width = len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
    params = np.zeros((1, width * 2), dtype=np.float64)
    entry_column = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "entry_retracement_base_pct"
    )
    close_column = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "close_retracement_base_pct"
    )
    params[0, entry_column] = 1.0e-50
    params[0, width + close_column] = 1.0e-50

    packed = _pack_tm_parameter_matrix(
        params, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, sides=2
    )

    assert packed[0, entry_column] == np.finfo(np.float32).tiny
    packed_width = packed.shape[1] // 2
    assert packed[0, packed_width + close_column] == np.finfo(np.float32).tiny
    assert packed[0, close_column] == 0.0
    assert packed[0, packed_width + entry_column] == 0.0








@pytest.mark.parametrize(
    "value", [0.0, 0.05, 0.999999999, float(np.nextafter(1.0, 0.0))]
)
def test_mps_realized_loss_fraction_matches_nearest_f32_cast(value):
    encoded = _encode_max_realized_loss_pct(value)

    assert encoded == float(np.float32(value))


@pytest.mark.parametrize("value", [1.0, 2.0, 1.0e100])
def test_mps_disabled_realized_loss_limit_has_finite_float32_encoding(value):
    assert _encode_max_realized_loss_pct(value) == 1.0


def test_decode_multicoin_fused_outputs_maps_directional_reductions():
    daily = torch.zeros((1, 1, 9), dtype=torch.float32)
    daily[:, :, 1].fill_(float("inf"))
    daily[:, :, 5].fill_(float("inf"))
    scalars = torch.arange(MPS_MULTICOIN_FUSED_SCALAR_COLS, dtype=torch.float32).reshape(
        1, MPS_MULTICOIN_FUSED_SCALAR_COLS
    )
    gaps = torch.zeros((1, GAP_BINS), dtype=torch.int32)

    output = _decode_multicoin_fused_outputs(daily, scalars, gaps)

    assert "entry_initial_balance_pct" not in output
    assert torch.equal(output["entry_initial_balance_pct_long"], scalars[:, 21])
    assert torch.equal(output["hsl_drawdown_ema_max_long"], scalars[:, 55])
    assert torch.equal(output["hsl_drawdown_ema_max_short"], scalars[:, 56])
    assert torch.equal(output["entry_initial_balance_pct_short"], scalars[:, 57])
    assert torch.equal(output["profit_sum_long"], scalars[:, 58])
    assert torch.equal(output["loss_sum_long"], scalars[:, 59])
    assert torch.equal(output["profit_sum_short"], scalars[:, 60])
    assert torch.equal(output["loss_sum_short"], scalars[:, 61])
    assert torch.equal(output["hsl_strategy_eq_recovery_max_ms_long"], scalars[:, 62])
    assert torch.equal(output["hsl_strategy_eq_recovery_max_ms_short"], scalars[:, 63])
    assert torch.equal(output["hsl_drawdown_ema_mean_worst_1pct_long"], scalars[:, 64])
    assert torch.equal(output["hsl_drawdown_ema_mean_worst_1pct_short"], scalars[:, 65])
    assert torch.equal(output["hsl_drawdown_raw_max_long"], scalars[:, 66])
    assert torch.equal(output["hsl_drawdown_raw_max_short"], scalars[:, 67])
    assert torch.equal(output["hsl_drawdown_raw_mean_worst_1pct_long"], scalars[:, 68])
    assert torch.equal(output["hsl_drawdown_raw_mean_worst_1pct_short"], scalars[:, 69])
    assert torch.equal(
        output["hsl_drawdown_ema_mean_worst_1pct_portfolio"], scalars[:, 70]
    )


def test_hsl_ema_tail_source_variant_is_opt_in_and_guarded():
    source = "#ifndef PASSIVBOT_HSL_EMA_TAIL_ENABLED\nbody"

    assert _with_hsl_ema_tail(source, False) is source
    assert _with_hsl_ema_tail(source, True) == (
        "#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 1\n" + source
    )
    with pytest.raises(RuntimeError, match="feature guard"):
        _with_hsl_ema_tail("body", True)


def test_hsl_raw_drawdown_source_variant_is_opt_in_and_guarded():
    source = "#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED\nbody"

    assert (
        _with_hsl_features(
            source,
            ema_tail_enabled=False,
            raw_drawdown_enabled=False,
            raw_tail_enabled=False,
        )
        is source
    )
    assert _with_hsl_features(
        source,
        ema_tail_enabled=False,
        raw_drawdown_enabled=True,
        raw_tail_enabled=False,
    ) == ("#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n" + source)
    with pytest.raises(RuntimeError, match="raw-drawdown feature guard"):
        _with_hsl_features(
            "body",
            ema_tail_enabled=False,
            raw_drawdown_enabled=True,
            raw_tail_enabled=False,
        )
    with pytest.raises(ValueError, match="require raw-drawdown"):
        _with_hsl_features(
            source,
            ema_tail_enabled=False,
            raw_drawdown_enabled=False,
            raw_tail_enabled=True,
        )


def test_hsl_raw_tail_source_variant_is_separately_opt_in_and_guarded():
    source = (
        "#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED\n"
        "#ifndef PASSIVBOT_HSL_RAW_TAIL_ENABLED\nbody"
    )

    assert _with_hsl_features(
        source,
        ema_tail_enabled=False,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    ) == (
        "#define PASSIVBOT_HSL_RAW_TAIL_CAPACITY 1\n"
        "#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 1\n"
        "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n" + source
    )
    with pytest.raises(RuntimeError, match="raw-tail feature guard"):
        _with_hsl_features(
            "#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED\nbody",
            ema_tail_enabled=False,
            raw_drawdown_enabled=True,
            raw_tail_enabled=True,
        )


def test_hsl_diagnostics_source_variant_is_opt_out_and_fail_closed():
    source = "#ifndef PASSIVBOT_HSL_DIAGNOSTICS_ENABLED\nbody"

    assert _with_hsl_features(
        source,
        ema_tail_enabled=False,
        raw_drawdown_enabled=False,
        raw_tail_enabled=False,
        diagnostics_enabled=False,
    ) == ("#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 0\n" + source)
    with pytest.raises(ValueError, match="require diagnostics"):
        _with_hsl_features(
            source,
            ema_tail_enabled=True,
            raw_drawdown_enabled=False,
            raw_tail_enabled=False,
            diagnostics_enabled=False,
        )
    with pytest.raises(RuntimeError, match="diagnostics feature guard"):
        _with_hsl_features(
            "body",
            ema_tail_enabled=False,
            raw_drawdown_enabled=False,
            raw_tail_enabled=False,
            diagnostics_enabled=False,
        )


def test_recovery_distribution_source_variant_is_opt_in_and_guarded():
    source = "#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED\nbody"

    assert _with_recovery_distribution(source, False) is source
    assert _with_recovery_distribution(source, True) == (
        "#define PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED 1\n" + source
    )
    with pytest.raises(RuntimeError, match="recovery-distribution feature guard"):
        _with_recovery_distribution("body", True)


def test_entry_interval_source_variant_is_opt_in_and_guarded():
    source = "inline void record_initial_entry_interval(\nbody"

    assert _with_entry_interval(source, False) is source
    assert _with_entry_interval(source, True) == (
        "#define PASSIVBOT_ENTRY_INTERVAL_ENABLED 1\n" + source
    )
    with pytest.raises(RuntimeError, match="entry-interval contract"):
        _with_entry_interval("body", True)


def test_decode_entry_interval_outputs_preserves_accumulator_columns():
    stats = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    counts = torch.arange(258, dtype=torch.int32).reshape(2, 129)
    counts[0, 0] = 16_777_217
    counts[0, 1] = 16_777_217

    output = _decode_entry_interval_outputs(stats, counts)

    assert torch.equal(output["entry_interval_sum_steps"], stats[:, 0])
    assert torch.equal(output["entry_interval_count"], counts[:, 0])
    assert torch.equal(output["entry_interval_max_steps"], stats[:, 1])
    assert torch.equal(output["entry_interval_hist"], counts[:, 1:])
    assert output["entry_interval_count"][0].item() == 16_777_217
    assert output["entry_interval_hist"][0, 0].item() == 16_777_217

    with pytest.raises(RuntimeError, match="partially present"):
        _decode_entry_interval_outputs(stats, None)
    with pytest.raises(RuntimeError, match="stats have an invalid shape"):
        _decode_entry_interval_outputs(torch.zeros((2, 1)), counts)
    with pytest.raises(RuntimeError, match="counts have an invalid shape"):
        _decode_entry_interval_outputs(stats, torch.zeros((2, 128)))


def test_fixed_wel_denominator_source_variant_is_opt_in_and_guarded():
    source = "#ifndef PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY\nbody"

    assert _with_dynamic_wel_by_tradability(source, True) is source
    assert _with_dynamic_wel_by_tradability(source, False) == (
        "#define PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY 0\n" + source
    )
    with pytest.raises(RuntimeError, match="dynamic-WEL feature guard"):
        _with_dynamic_wel_by_tradability("body", False)


def test_btc_risk_source_variant_is_opt_in_and_guarded():
    source = "struct BtcRiskState { float peak; };\nbody"

    assert _with_btc_risk(source, False) is source
    assert _with_btc_risk(source, True) == (
        "#define PASSIVBOT_BTC_RISK_ENABLED 1\n" + source
    )
    with pytest.raises(RuntimeError, match="shared BTC-risk contract"):
        _with_btc_risk("body", True)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_strategy_eq_recovery_distribution_matches_strict_rust_contract():
    matrix = torch.tensor(
        [
            [100.0, 90.0, 95.0, 101.0, 100.0, 102.0],
            [100.0, 100.0, 101.0, float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )

    actual = (
        strategy_eq_recovery_distribution_from_samples(
            matrix, sample_interval_days=1.0 / 24.0
        )
        .cpu()
        .numpy()
    )

    assert actual[0].tolist() == pytest.approx(
        [value / 24.0 for value in [8.0 / 6.0, 1.0, 2.75, 2.95, 3.0, 3.0, 3.0]]
    )
    assert actual[1].tolist() == pytest.approx(
        [value / 24.0 for value in [1.0, 1.0, 1.9, 1.98, 2.0, 2.0, 2.0]]
    )






@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_hsl_strategy_equity_recovery_matches_exact_rust_recurrence():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_hsl_strategy_equity_recovery_probe(
    constant float* samples,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    HslStrategyEquityStats stats = init_hsl_strategy_equity_stats();
    for (int k = 0; k < 7; ++k) {
        update_hsl_strategy_equity_stats(stats, samples[k], 0);
    }
    output[0] = hsl_strategy_equity_recovery_max_steps(stats);
    HslStrategyEquityStats resumed = init_hsl_strategy_equity_stats();
    update_hsl_strategy_equity_stats(resumed, samples[0], 0);
    update_hsl_strategy_equity_stats(resumed, samples[2], 0);
    update_hsl_strategy_equity_stats(resumed, samples[3], 0);
    update_hsl_strategy_equity_stats(resumed, samples[4], 0);
    output[1] = hsl_strategy_equity_recovery_max_steps(resumed);
}
"""
    samples = torch.tensor(
        [100.0, 100.0, 90.0, 95.0, 101.0, 101.0, 99.0],
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(2, dtype=torch.float32, device=gpu_device())
    library = compile_shader(passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel)

    library.passivbot_hsl_strategy_equity_recovery_probe(
        samples, output, threads=(1, 1, 1)
    )
    synchronize()

    # Exact Rust waits for a strictly greater sample, so the equal sample at
    # k=1 remains underwater until 101 at k=4.
    assert output[0].item() == 4.0
    # Exact Rust compacts away halted cooldown bars before recovery analysis.
    # The resumed sequence has four eligible samples, so recovery is three
    # samples even though its source candle indices span five steps.
    assert output[1].item() == 3.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_hsl_drawdown_ema_tail_reducer_matches_exact_unambiguous_bins():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_hsl_drawdown_ema_tail_probe(
    constant float* samples,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    HslDrawdownEmaTailStats stats = init_hsl_drawdown_ema_tail_stats();
    for (int k = 0; k < 200; ++k) {
        update_hsl_drawdown_ema_tail_stats(stats, samples[k]);
    }
    output[0] = hsl_drawdown_ema_mean_worst_1pct(stats);
}
"""
    samples = torch.tensor(
        [0.01] * 196 + [0.2, 0.2, 0.4, 0.4],
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(1, dtype=torch.float32, device=gpu_device())
    source = _with_hsl_ema_tail(passivbot_rust.mps_ema_anchor_multicoin_source_py(), True)
    library = compile_shader(source + probe_kernel)
    library.passivbot_hsl_drawdown_ema_tail_probe(samples, output, threads=(1, 1, 1))
    synchronize()

    # floor(200 * 1%) is two, and the two worst values occupy the highest bin.
    assert output.item() == pytest.approx(0.4, abs=1.0e-6)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_hsl_strategy_equity_raw_drawdown_matches_exact_peak_contract():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_hsl_strategy_equity_raw_drawdown_probe(
    constant float* samples,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    HslStrategyEquityStats stats = init_hsl_strategy_equity_stats();
    for (int k = 0; k < 5; ++k) {
        update_hsl_strategy_equity_stats(stats, samples[k], k / 2);
    }
    output[0] = hsl_strategy_equity_drawdown_max(stats);
    output[1] = hsl_strategy_equity_drawdown_mean_worst_1pct(stats);
}
"""
    samples = torch.tensor(
        [100.0, 90.0, 110.0, 88.0, 120.0],
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(2, dtype=torch.float32, device=gpu_device())
    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_multicoin_source_py(),
        ema_tail_enabled=False,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    library = compile_shader(source + probe_kernel)
    library.passivbot_hsl_strategy_equity_raw_drawdown_probe(
        samples, output, threads=(1, 1, 1)
    )
    synchronize()

    assert output[0].item() == pytest.approx(0.2, abs=1.0e-6)
    # The two intraday lows reduce to daily worst values of 0.1 and 0.2;
    # the open terminal day contributes zero. With three observed days, Rust's
    # floor(1%) contract retains the single worst day.
    assert output[1].item() == pytest.approx(0.2, abs=1.0e-6)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_side_states_are_isolated_for_fused_execution():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_side_state_isolation_probe(
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    long_side.psize[0] = 1.0f;
    short_side.psize[0] = 2.0f;
    long_side.entry_tick[0] = 3;
    short_side.entry_tick[0] = 4;
    long_side.selected[0] = true;
    short_side.selected[0] = false;
    long_side.hsl.enabled = true;
    short_side.hsl.enabled = false;
    long_side.coin_hsl[0].triggers = 5.0f;
    short_side.coin_hsl[0].triggers = 6.0f;
    long_side.entry_qty[0] = 7.0f;
    short_side.entry_qty[0] = 8.0f;
    output[0] = long_side.psize[0];
    output[1] = short_side.psize[0];
    output[2] = float(long_side.entry_tick[0]);
    output[3] = float(short_side.entry_tick[0]);
    output[4] = long_side.selected[0] && !short_side.selected[0] ? 1.0f : 0.0f;
    output[5] = long_side.hsl.enabled && !short_side.hsl.enabled ? 1.0f : 0.0f;
    output[6] = long_side.coin_hsl[0].triggers
        + short_side.coin_hsl[0].triggers;
    output[7] = float(
        long_side.entry_qty[0]
            + short_side.entry_qty[0]
    );
}
"""

    output = torch.zeros(8, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_side_state_isolation_probe(
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    assert output.cpu().tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 11.0, 15.0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_fused_reducers_share_loss_budget_and_fallback():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_fused_reducer_budget_probe(
    constant int* market_execution_ticks,
    constant float* bars,
    constant float* coin_settings,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    for (int c = 0; c < MAX_COINS; ++c) {
        long_side.psize[c] = 0.0f;
        long_side.pprice[c] = 0.0f;
        long_side.close_qty[c] = 0.0f;
        long_side.close_tick[c] = 0;
        long_side.close_market[c] = false;
        long_side.secondary_close_qty[c] = 0.0f;
        long_side.secondary_close_tick[c] = 0;
        long_side.secondary_close_market[c] = false;
        long_side.twel_close_qty[c] = 0.0f;
        long_side.twel_close_tick[c] = 0;
        long_side.unstuck_close_qty[c] = 0.0f;
        long_side.unstuck_close_tick[c] = 0;
        long_side.close_is_protective_reducer[c] = false;
        long_side.close_is_unstuck_reducer[c] = false;
        long_side.close_is_hsl_panic[c] = false;

        short_side.psize[c] = 0.0f;
        short_side.pprice[c] = 0.0f;
        short_side.close_qty[c] = 0.0f;
        short_side.close_tick[c] = 0;
        short_side.close_market[c] = false;
        short_side.secondary_close_qty[c] = 0.0f;
        short_side.secondary_close_tick[c] = 0;
        short_side.secondary_close_market[c] = false;
        short_side.twel_close_qty[c] = 0.0f;
        short_side.twel_close_tick[c] = 0;
        short_side.unstuck_close_qty[c] = 0.0f;
        short_side.unstuck_close_tick[c] = 0;
        short_side.close_is_protective_reducer[c] = false;
        short_side.close_is_unstuck_reducer[c] = false;
        short_side.close_is_hsl_panic[c] = false;
    }

    // The long preferred TWEL close loses 15 and cannot spend the 10 budget.
    // Its unstuck fallback loses 5. The short TWEL close loses 6, wins the
    // next global quantity comparison, and leaves only 4 for that fallback.
    long_side.psize[0] = 5.0f;
    long_side.pprice[0] = 105.0f;
    long_side.twel_close_qty[0] = 3.0f;
    long_side.twel_close_tick[0] = 100;
    long_side.unstuck_close_qty[0] = 1.0f;
    long_side.unstuck_close_tick[0] = 100;

    short_side.psize[1] = 5.0f;
    short_side.pprice[1] = 97.0f;
    short_side.twel_close_qty[1] = 2.0f;
    short_side.twel_close_tick[1] = 100;

    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    finalize_ema_multicoin_reducers_fused(
        market_execution_ticks,
        long_side, short_side, account,
        bars, coin_settings, 1, 2,
        true, true, true, 0.0f, 0.0f, 0.01f
    );

    output[0] = long_side.close_qty[0];
    output[1] = long_side.close_market[0] ? 1.0f : 0.0f;
    output[2] = long_side.close_is_protective_reducer[0] ? 1.0f : 0.0f;
    output[3] = long_side.close_is_unstuck_reducer[0] ? 1.0f : 0.0f;
    output[4] = short_side.close_qty[1];
    output[5] = short_side.close_market[1] ? 1.0f : 0.0f;
    output[6] = short_side.close_is_protective_reducer[1] ? 1.0f : 0.0f;
    output[7] = short_side.close_is_unstuck_reducer[1] ? 1.0f : 0.0f;
}
"""
    bars = torch.zeros((2, 2, 4), dtype=torch.float32, device=gpu_device())
    bars[:, :, :3] = 100.0
    bars[:, :, 3] = 1.0
    coin_settings = torch.tensor(
        [
            [
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                60_000.0,
                100.0,
                1.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                60_000.0,
                100.0,
                1.0,
                0.0,
                0.0,
            ],
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(8, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_fused_reducer_budget_probe(
        _probe_market_execution_ticks(bars, [1.0, 1.0]),
        bars,
        coin_settings,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    assert output.cpu().tolist() == [0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_side_states_are_isolated_for_fused_execution():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_side_state_isolation_probe(
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    long_side.psize[0] = 1.0f;
    short_side.psize[0] = 2.0f;
    long_side.entry_tick[0] = 3;
    short_side.entry_tick[0] = 4;
    long_side.selected[0] = true;
    short_side.selected[0] = false;
    long_side.close_grid_gen_psize[0] = 5.0f;
    short_side.close_grid_gen_psize[0] = 6.0f;
    long_side.hsl.enabled = true;
    short_side.hsl.enabled = false;
    long_side.coin_hsl[0].triggers = 7.0f;
    short_side.coin_hsl[0].triggers = 8.0f;
    long_side.entry_qty[0] = 9.0f;
    short_side.entry_qty[0] = 10.0f;
    long_side.incumbent[0] = true;
    short_side.incumbent[0] = false;
    long_side.max_tradable_seen = 11;
    short_side.max_tradable_seen = 12;
    long_side.score[0] = 13;
    short_side.score[0] = 14;
    output[0] = long_side.psize[0];
    output[1] = short_side.psize[0];
    output[2] = float(long_side.entry_tick[0]);
    output[3] = float(short_side.entry_tick[0]);
    output[4] = long_side.selected[0] && !short_side.selected[0] ? 1.0f : 0.0f;
    output[5] = long_side.close_grid_gen_psize[0]
        + short_side.close_grid_gen_psize[0];
    output[6] = long_side.hsl.enabled && !short_side.hsl.enabled ? 1.0f : 0.0f;
    output[7] = long_side.coin_hsl[0].triggers
        + short_side.coin_hsl[0].triggers;
    output[8] = float(
        long_side.entry_qty[0]
            + short_side.entry_qty[0]
    );
    output[9] = long_side.incumbent[0]
        && !short_side.incumbent[0] ? 1.0f : 0.0f;
    output[10] = float(long_side.max_tradable_seen);
    output[11] = float(short_side.max_tradable_seen);
    output[12] = float(long_side.score[0]);
    output[13] = float(short_side.score[0]);
}
"""

    output = torch.zeros(14, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_side_state_isolation_probe(
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    assert output.cpu().tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        1.0,
        11.0,
        1.0,
        15.0,
        19.0,
        1.0,
        11.0,
        12.0,
        13.0,
        14.0,
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_fill_state_shares_directional_accounting():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_fill_state_probe(
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    record_tm_multicoin_gross_pnl(4.0f, fills, false);
    record_tm_multicoin_gross_pnl(-2.0f, fills, false);
    record_tm_multicoin_gross_pnl(3.0f, fills, true);
    record_tm_multicoin_gross_pnl(-5.0f, fills, true);
    record_realized_net(
        -1.0f, account,
        fills.day_fill_count, fills.fill_count,
        fills.fill_count_entry, fills.fill_count_long,
        fills.pnl_recovery_peak, fills.pnl_recovery_peak_k,
        fills.pnl_recovery_max_min, 10.0f, true, true
    );
    record_realized_net(
        2.0f, account,
        fills.day_fill_count, fills.fill_count,
        fills.fill_count_entry, fills.fill_count_long,
        fills.pnl_recovery_peak, fills.pnl_recovery_peak_k,
        fills.pnl_recovery_max_min, 20.0f, false, false
    );
    output[0] = account.balance;
    output[1] = account.realized_pnl_total;
    output[2] = account.realized_pnl_long;
    output[3] = account.realized_pnl_short;
    output[4] = fills.profit_sum;
    output[5] = fills.loss_sum;
    output[6] = fills.profit_sum_long;
    output[7] = fills.loss_sum_long;
    output[8] = fills.profit_sum_short;
    output[9] = fills.loss_sum_short;
    output[10] = fills.fill_count;
    output[11] = fills.fill_count_entry;
    output[12] = fills.fill_count_long;
    output[13] = fills.day_fill_count;
    output[14] = fills.pnl_recovery_peak;
    output[15] = fills.pnl_recovery_peak_k;
    output[16] = fills.pnl_recovery_max_min;
    output[17] = fills.held_max_min + fills.held_sum_min
        + fills.held_count + fills.position_unchanged_max_min
        + fills.day_volume;
}
"""

    output = torch.zeros(18, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_fill_state_probe(output, threads=(1, 1, 1))
    synchronize()

    assert output.cpu().tolist() == [
        1001.0,
        1.0,
        -1.0,
        2.0,
        7.0,
        7.0,
        4.0,
        2.0,
        3.0,
        5.0,
        2.0,
        1.0,
        1.0,
        2.0,
        1.0,
        20.0,
        10.0,
        0.0,
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_close_fill_centralizes_hsl_accounting():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_close_fill_probe(
    constant float* hsl_params,
    device float* coin_fill_counts,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    long_side.hsl = load_hsl(hsl_params, 0, 0);
    long_side.coin_hsl[0] = load_hsl(hsl_params, 0, 0);
    short_side.hsl = load_hsl(hsl_params, 7, 0);
    short_side.coin_hsl[0] = load_hsl(hsl_params, 7, 0);
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    float long_equity = 1000.0f;
    float short_equity = 989.0f;
    coin_fill_counts[0] = 0.0f;
    record_tm_multicoin_close_fill(
        long_side, account, fills, coin_fill_counts,
        0, 1, 0, 10, -10.0f, -11.0f, 1.0f,
        110.0f, 100.0f, 1.0f, false, true, true, long_equity
    );
    record_tm_multicoin_close_fill(
        short_side, account, fills, coin_fill_counts,
        0, 1, 0, 11, -10.0f, -11.0f, 1.0f,
        90.0f, 100.0f, 1.0f, true, true, true, short_equity
    );
    output[0] = account.balance;
    output[1] = account.realized_pnl_total;
    output[2] = account.realized_pnl_long;
    output[3] = account.realized_pnl_short;
    output[4] = fills.profit_sum;
    output[5] = fills.loss_sum;
    output[6] = fills.fill_count;
    output[7] = fills.fill_count_entry;
    output[8] = fills.fill_count_long;
    output[9] = fills.day_fill_count;
    output[10] = long_equity;
    output[11] = short_equity;
    output[12] = long_side.coin_realized_pnl[0];
    output[13] = short_side.coin_realized_pnl[0];
    output[14] = long_side.coin_realized_pnl[0];
    output[15] = long_side.coin_hsl[0].panic_event_start_equity;
    output[16] = long_side.coin_hsl[0].panic_event_loss;
    output[17] = short_side.hsl.panic_event_start_equity;
    output[18] = short_side.hsl.panic_event_loss;
    output[19] = coin_fill_counts[0];
}
"""

    hsl_params = torch.tensor(
        [
            1.0,
            0.2,
            60.0,
            0.0,
            0.0,
            2.0,
            1.0,
            1.0,
            0.2,
            60.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device=gpu_device())
    output = torch.zeros(20, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_close_fill_probe(
        hsl_params, coin_fill_counts, output, threads=(1, 1, 1)
    )
    synchronize()

    assert output.cpu().tolist() == [
        978.0,
        -22.0,
        -11.0,
        -11.0,
        0.0,
        20.0,
        2.0,
        0.0,
        1.0,
        2.0,
        999.0,
        989.0,
        -11.0,
        -11.0,
        -11.0,
        1000.0,
        11.0,
        989.0,
        11.0,
        2.0,
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_entry_fill_centralizes_hsl_accounting():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_entry_fill_probe(
    constant float* hsl_params,
    device float* coin_fill_counts,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    long_side.hsl = load_hsl(hsl_params, 0, 0);
    long_side.coin_hsl[0] = load_hsl(hsl_params, 0, 0);
    short_side.hsl = load_hsl(hsl_params, 7, 0);
    short_side.coin_hsl[0] = load_hsl(hsl_params, 7, 0);
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    float long_equity = 1000.0f;
    float short_equity = 990.0f;
    coin_fill_counts[0] = 0.0f;
    record_tm_multicoin_entry_fill(
        long_side, account, fills, coin_fill_counts,
        0, 1, 0, 10, 10.0f, 1.0f,
        110.0f, 100.0f, 1.0f, false, true, long_equity
    );
    record_tm_multicoin_entry_fill(
        short_side, account, fills, coin_fill_counts,
        0, 1, 0, 11, 10.0f, 1.0f,
        90.0f, 100.0f, 1.0f, true, true, short_equity
    );
    output[0] = account.balance;
    output[1] = account.realized_pnl_total;
    output[2] = account.realized_pnl_long;
    output[3] = account.realized_pnl_short;
    output[4] = fills.fill_count;
    output[5] = fills.fill_count_entry;
    output[6] = fills.fill_count_long;
    output[7] = fills.day_fill_count;
    output[8] = long_equity;
    output[9] = short_equity;
    output[10] = long_side.coin_realized_pnl[0];
    output[11] = short_side.coin_realized_pnl[0];
    output[12] = long_side.coin_realized_pnl[0];
    output[13] = fills.profit_sum;
    output[14] = fills.loss_sum;
    output[15] = coin_fill_counts[0];
}
"""

    hsl_params = torch.tensor(
        [
            1.0,
            0.2,
            60.0,
            0.0,
            0.0,
            2.0,
            1.0,
            1.0,
            0.2,
            60.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device=gpu_device())
    output = torch.zeros(16, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_entry_fill_probe(
        hsl_params, coin_fill_counts, output, threads=(1, 1, 1)
    )
    synchronize()

    assert output.cpu().tolist() == [
        980.0,
        -20.0,
        -10.0,
        -10.0,
        2.0,
        2.0,
        1.0,
        2.0,
        980.0,
        990.0,
        -10.0,
        -10.0,
        -10.0,
        0.0,
        0.0,
        2.0,
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_fill_position_helpers_preserve_chronology():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_fill_position_probe(
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState close_side;
    TrailingMartingaleMulticoinSideState entry_side;
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    for (int c = 0; c < 2; ++c) {
        close_side.psize[c] = c == 0 ? 0.0f : 1.0f;
        close_side.pprice[c] = 100.0f;
        close_side.position_open_k[c] = 5.0f;
        close_side.close_qty[c] = 1.0f;
        close_side.secondary_close_qty[c] = 2.0f;
        close_side.close_reconstruct_after_reducer[c] = true;
        close_side.close_is_unstuck_reducer[c] = true;
        close_side.close_is_hsl_panic[c] = true;
        close_side.filled_coin[c] = false;
    }
    close_side.position_last_fill_k[0] = 10.0f;
    finalize_tm_multicoin_close_position(close_side, fills, 0, 20);
    finalize_tm_multicoin_close_position(close_side, fills, 1, 20);
    update_tm_multicoin_position_fill_timestamp(close_side, fills, 0, 20);

    entry_side.psize[0] = 0.0f;
    entry_side.pprice[0] = 0.0f;
    entry_side.position_open_k[0] = -1.0f;
    entry_side.position_last_fill_k[0] = 12.0f;
    entry_side.entry_qty[0] = 1.0f;
    entry_side.filled_coin[0] = false;
    apply_tm_multicoin_entry_position(
        entry_side, fills, 0, 20, 1.0f, 120.0f, 1.0f, 1.0f, 1000.0f
    );
    entry_side.entry_qty[0] = 1.0f;
    apply_tm_multicoin_entry_position(
        entry_side, fills, 0, 21, 1.0f, 100.0f, 1.0f, 1.0f, 1000.0f
    );
    update_tm_multicoin_position_fill_timestamp(entry_side, fills, 0, 21);

    output[0] = close_side.pprice[0];
    output[1] = close_side.position_open_k[0];
    output[2] = close_side.close_qty[0];
    output[3] = close_side.secondary_close_qty[0];
    output[4] = close_side.close_reconstruct_after_reducer[0] ? 1.0f : 0.0f;
    output[5] = close_side.close_is_unstuck_reducer[0] ? 1.0f : 0.0f;
    output[6] = close_side.close_is_hsl_panic[0] ? 1.0f : 0.0f;
    output[7] = close_side.filled_coin[0] ? 1.0f : 0.0f;
    output[8] = close_side.position_last_fill_k[0];
    output[9] = close_side.pprice[1];
    output[10] = close_side.position_open_k[1];
    output[11] = close_side.filled_coin[1] ? 1.0f : 0.0f;
    output[12] = fills.held_max_min;
    output[13] = fills.held_sum_min;
    output[14] = fills.held_count;
    output[15] = fills.position_unchanged_max_min;
    output[16] = entry_side.psize[0];
    output[17] = entry_side.pprice[0];
    output[18] = entry_side.position_open_k[0];
    output[19] = entry_side.last_increase_k[0];
    output[20] = entry_side.entry_qty[0];
    output[21] = entry_side.filled_coin[0] ? 1.0f : 0.0f;
    output[22] = entry_side.position_last_fill_k[0];
    output[23] = fills.day_volume;
}
"""

    output = torch.zeros(24, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_fill_position_probe(output, threads=(1, 1, 1))
    synchronize()

    np.testing.assert_allclose(
        output.cpu().numpy(),
        [
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            100.0,
            5.0,
            1.0,
            15.0,
            15.0,
            1.0,
            10.0,
            2.0,
            110.0,
            20.0,
            21.0,
            0.0,
            1.0,
            21.0,
            0.22,
        ],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_side_fill_pass_shares_account_chronology():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_side_fill_pass_probe(
    constant int* market_execution_ticks,
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* hsl_params,
    device float* coin_fill_counts,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    TrailingMartingaleMulticoinSideConfig long_config = {};
    TrailingMartingaleMulticoinSideConfig short_config = {};
    long_config.coin_hsl_mode = false;
    short_config.coin_hsl_mode = false;
    long_side.hsl = load_hsl(hsl_params, 0, 0);
    short_side.hsl = load_hsl(hsl_params, 0, 0);
    long_side.entry_deferred_twel_gate = false;
    short_side.entry_deferred_twel_gate = false;
    for (int c = 0; c < 1; ++c) {
        long_side.psize[c] = 0.0f;
        short_side.psize[c] = 0.0f;
        long_side.pprice[c] = 0.0f;
        short_side.pprice[c] = 0.0f;
        long_side.position_open_k[c] = -1.0f;
        short_side.position_open_k[c] = -1.0f;
        long_side.position_last_fill_k[c] = -1.0f;
        short_side.position_last_fill_k[c] = -1.0f;
        long_side.entry_qty[c] = 1.0f;
        short_side.entry_qty[c] = 1.0f;
        long_side.entry_tick[c] = 100;
        short_side.entry_tick[c] = 100;
        long_side.entry_recursive_mode[c] = false;
        short_side.entry_recursive_mode[c] = false;
        long_side.close_qty[c] = 0.0f;
        short_side.close_qty[c] = 0.0f;
        long_side.secondary_close_qty[c] = 0.0f;
        short_side.secondary_close_qty[c] = 0.0f;
        long_side.close_reconstruct_after_reducer[c] = false;
        short_side.close_reconstruct_after_reducer[c] = false;
        long_side.close_recursive_mode[c] = false;
        short_side.close_recursive_mode[c] = false;
        long_side.close_is_unstuck_reducer[c] = false;
        short_side.close_is_unstuck_reducer[c] = false;
        long_side.close_is_hsl_panic[c] = false;
        short_side.close_is_hsl_panic[c] = false;
        long_side.coin_realized_pnl[c] = 0.0f;
        short_side.coin_realized_pnl[c] = 0.0f;
    }
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    coin_fill_counts[0] = 0.0f;
    float long_equity = 1000.0f;
    float short_equity = 1000.0f;
    bool long_entry = process_tm_multicoin_side_fills(
        market_execution_ticks,
        long_side, long_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 1, 1, false, true, true, false,
        1.0f, false, 0.0f, 0.0f, false, long_equity
    );
    bool short_entry = process_tm_multicoin_side_fills(
        market_execution_ticks,
        short_side, short_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 1, 1, true, true, true, false,
        1.0f, false, 0.0f, 0.0f, false, short_equity
    );
    long_side.close_qty[0] = 1.0f;
    short_side.close_qty[0] = 1.0f;
    long_side.close_tick[0] = 100;
    short_side.close_tick[0] = 100;
    long_equity = 1000.0f;
    short_equity = 1000.0f;
    bool long_close = process_tm_multicoin_side_fills(
        market_execution_ticks,
        long_side, long_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 2, 1, false, true, true, false,
        1.0f, false, 0.0f, 0.0f, false, long_equity
    );
    bool short_close = process_tm_multicoin_side_fills(
        market_execution_ticks,
        short_side, short_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 2, 1, true, true, true, false,
        1.0f, false, 0.0f, 0.0f, false, short_equity
    );
    output[0] = long_entry && short_entry && long_close && short_close
        ? 1.0f : 0.0f;
    output[1] = account.balance;
    output[2] = account.realized_pnl_total;
    output[3] = fills.fill_count;
    output[4] = fills.fill_count_entry;
    output[5] = fills.fill_count_long;
    output[6] = long_side.psize[0];
    output[7] = short_side.psize[0];
    output[8] = long_side.pprice[0];
    output[9] = short_side.pprice[0];
    output[10] = fills.held_max_min;
    output[11] = fills.held_sum_min;
    output[12] = fills.held_count;
    output[13] = fills.position_unchanged_max_min;
    output[14] = fills.day_volume;
    output[15] = coin_fill_counts[0];
}
"""

    bars = torch.tensor(
        [
            [[100.0, 100.0, 100.0, 1.0]],
            [[100.0, 100.0, 100.0, 1.0]],
            [[100.0, 100.0, 100.0, 1.0]],
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    fill_ticks = torch.tensor(
        [[0, 0], [100, 99], [100, 99]], dtype=torch.int32, device=gpu_device()
    )
    touch_ticks = torch.zeros((3, 2), dtype=torch.int32, device=gpu_device())
    touch_nearest_ticks = torch.zeros(3, dtype=torch.int32, device=gpu_device())
    touch_min_qty_bits = torch.zeros(3, dtype=torch.int32, device=gpu_device())
    touch_min_qty_relation = torch.zeros(
        3,
        dtype=torch.int8 if gpu_device() == "cuda" else torch.int32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 1.0
    coin_settings[0, 1] = 1.0
    coin_settings[0, 2] = 1.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = _tm_probe_full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    hsl_params = torch.tensor(
        [1.0, 0.2, 60.0, 0.0, 0.0, 1.0, 1.0],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device=gpu_device())
    output = torch.zeros(16, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_side_fill_pass_probe(
        _probe_market_execution_ticks(bars, [1.0]),
        bars,
        fill_ticks,
        touch_ticks,
        touch_nearest_ticks,
        touch_min_qty_bits,
        touch_min_qty_relation,
        coin_settings,
        coin_overrides,
        hsl_params,
        coin_fill_counts,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    np.testing.assert_allclose(
        output.cpu().numpy(),
        [
            1.0,
            1000.0,
            0.0,
            4.0,
            2.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            2.0,
            1.0,
            0.4,
            4.0,
        ],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_selection_phase_keeps_side_local_hsl_rankings():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_selection_phase_probe(
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState long_side = {};
    TrailingMartingaleMulticoinSideState short_side = {};
    TrailingMartingaleMulticoinSideConfig long_config = {};
    TrailingMartingaleMulticoinSideConfig short_config = {};
    long_config.coin_hsl_mode = true;
    short_config.coin_hsl_mode = false;
    long_config.w_volume = 1.0f;
    short_config.w_volume = 1.0f;
    long_config.w_ready = 0.0f;
    short_config.w_ready = 0.0f;
    long_config.w_volatility = 0.0f;
    short_config.w_volatility = 0.0f;
    long_config.initial_ema_dist = 0.0f;
    short_config.initial_ema_dist = 0.0f;
    long_config.n_positions = 1;
    short_config.n_positions = 1;
    for (int c = 0; c < 3; ++c) {
        long_side.psize[c] = 0.0f;
        short_side.psize[c] = 0.0f;
        long_side.entry_qty[c] = 0.0f;
        short_side.entry_qty[c] = 0.0f;
        long_side.selected[c] = false;
        short_side.selected[c] = false;
        long_side.incumbent[c] = false;
        short_side.incumbent[c] = false;
        long_side.survivor[c] = false;
        short_side.survivor[c] = false;
        long_side.ema0[c] = 100.0f;
        long_side.ema1[c] = 100.0f;
        long_side.ema2[c] = 100.0f;
        short_side.ema0[c] = 100.0f;
        short_side.ema1[c] = 100.0f;
        short_side.ema2[c] = 100.0f;
        long_side.forager_volume[c] = float(3 - c) * 100.0f;
        short_side.forager_volume[c] = float(c + 1) * 100.0f;
        long_side.forager_volatility[c] = 0.0f;
        short_side.forager_volatility[c] = 0.0f;
        long_side.coin_hsl[c].enabled = false;
        long_side.coin_hsl[c].halted = false;
        long_side.coin_hsl[c].tier = 0;
        long_side.coin_hsl[c].red_active_now = false;
    }
    ulong long_selection_blocked_mask = 0ul;
    ulong short_selection_blocked_mask = 0ul;
    ulong long_order_blocked_mask = 0ul;
    ulong short_order_blocked_mask = 0ul;
    compute_tm_multicoin_one_way_initial_blocks(
        long_side, long_config, coin_overrides,
        short_side, short_config, coin_overrides,
        bars, coin_settings, 1, 3, 0, 0, 1, 1, true, true, false, 0.0f,
        long_selection_blocked_mask, short_selection_blocked_mask,
        long_order_blocked_mask, short_order_blocked_mask
    );
    // Block the long side's highest-volume coin without affecting short.
    long_side.coin_hsl[0].enabled = true;
    long_side.coin_hsl[0].hsl.action = 1; // Flat scope in cooldown.
    update_tm_multicoin_side_selection(
        long_side, long_config, bars, coin_settings, coin_overrides,
        1, 3, false, true, 1, 0.0f, 0ul, false, 0.0f
    );
    update_tm_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        1, 3, true, true, 1, 0.0f, 0ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[c] = long_side.selected[c] ? 1.0f : 0.0f;
        output[3 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
    // Mimic a new opposite-held block on short's highest-ranked coin.
    // The changed mask must force reselection onto the next candidate.
    update_tm_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        1, 3, true, false, 1, 0.0f, 4ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[6 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
    output[9] = float(long_selection_blocked_mask);
    output[10] = float(short_selection_blocked_mask);
    output[11] = float(long_order_blocked_mask);
    output[12] = float(short_order_blocked_mask);
    // Coin one is the current flat incumbent, but its validity window ended
    // at k=1. With no fill or mask change, k=2 must still evict it and select
    // the only surviving unblocked candidate.
    update_tm_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        2, 3, true, false, 1, 0.0f, 4ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[13 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
}
"""

    bars = torch.tensor(
        [
            [[100.0, 100.0, 100.0, 1.0]] * 3,
            [[100.0, 100.0, 100.0, 1.0]] * 3,
            [[100.0, 100.0, 100.0, 1.0]] * 3,
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((3, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[:, 7] = 10.0
    coin_settings[1, 7] = 1.0
    coin_overrides = _tm_probe_full(
        (3, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(16, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py().replace(
            "constant int MAX_COINS = 64;", "constant int MAX_COINS = 3;"
        )
        + probe_kernel
    )
    library.passivbot_tm_multicoin_selection_phase_probe(
        bars, coin_settings, coin_overrides, output, threads=(1, 1, 1)
    )
    synchronize()

    assert output.cpu().tolist() == [
        0.0,
        1.0,
        0.0,  # Long excludes its HSL-blocked top-ranked coin.
        0.0,
        0.0,
        1.0,  # Short ranks independently.
        0.0,
        1.0,
        0.0,  # Opposite-held block changes short selection.
        0.0,
        0.0,
        0.0,
        7.0,  # Selection and order one-way masks.
        1.0,
        0.0,
        0.0,  # Expired flat incumbent is evicted without a fill.
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_order_phase_builds_both_sides_on_shared_account():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_order_phase_probe(
    constant float* bars,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideConfig long_config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideConfig short_config =
        load_trailing_martingale_multicoin_side_config(params, PARAM_COLS);
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    init_trailing_martingale_multicoin_side_state(
        long_side, long_config, coin_settings, coin_overrides, 1
    );
    init_trailing_martingale_multicoin_side_state(
        short_side, short_config, coin_settings, coin_overrides, 1
    );
    long_side.selected[0] = true;
    short_side.selected[0] = true;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    generate_tm_multicoin_side_orders(
        long_side, long_config, account,
        bars, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides,
        1, 1, false, 1, 1, 0, false, 1.0f,
        false, 0.001f, 0.0f, -2, 0ul
    );
    generate_tm_multicoin_side_orders(
        short_side, short_config, account,
        bars, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides,
        1, 1, true, 1, 1, 0, false, 1.0f,
        false, 0.001f, 0.0f, -2, 0ul
    );
    output[0] = long_side.entry_qty[0];
    output[1] = short_side.entry_qty[0];
    output[2] = float(long_side.entry_tick[0]);
    output[3] = float(short_side.entry_tick[0]);
    output[4] = long_side.close_qty[0];
    output[5] = short_side.close_qty[0];
    output[6] = account.balance;
    output[7] = account.realized_pnl_total;
    generate_tm_multicoin_side_orders(
        long_side, long_config, account,
        bars, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides,
        1, 1, false, 1, 1, 0, false, 1.0f,
        false, 0.001f, 0.0f, -2, 1ul
    );
    output[8] = long_side.entry_qty[0];
    output[9] = long_side.entry_candidate[0] ? 1.0f : 0.0f;
    output[10] = long_side.contribution[0];
}
"""

    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", "long", count=3, closes=(100.0, 100.0)
    )
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_ema_dist")] = 0.01
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("gate_initial")] = 1.0
    short_row = list(row)
    short_row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_qty_pct")
    ] = 0.5
    params = _tm_probe_tensor([row, short_row], dtype=torch.float32, device=gpu_device())
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[101.0, 99.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device=gpu_device(),
    )
    touch_ticks = torch.tensor(
        [[[10_000, 10_000]], [[10_000, 10_000]]],
        dtype=torch.int32,
        device=gpu_device(),
    )
    touch_nearest_ticks = torch.full(
        (2, 1), 10_000, dtype=torch.int32, device=gpu_device()
    )
    touch_min_qty_bits = torch.zeros((2, 1), dtype=torch.int32, device=gpu_device())
    touch_min_qty_relation = torch.zeros(
        (2, 1),
        dtype=torch.int8 if gpu_device() == "cuda" else torch.int32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_settings[0, 9] = 100.0
    coin_settings[0, 10] = 1.0
    coin_overrides = _tm_probe_full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(11, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_order_phase_probe(
        bars,
        touch_ticks,
        touch_nearest_ticks,
        touch_min_qty_bits,
        touch_min_qty_relation,
        coin_settings,
        coin_overrides,
        params,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    values = output.cpu().numpy()
    assert values[0] > values[1] > 0.0
    assert values[2] < values[3]
    assert values[4] == 0.0
    assert values[5] == 0.0
    assert values[6] == 1_000.0
    assert values[7] == 0.0
    assert values[8:].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_candle_helpers_advance_independent_sides():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_candle_helpers_probe(
    constant float* bars,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    EmaMulticoinSideConfig long_config = {};
    EmaMulticoinSideConfig short_config = {};
    long_config.alpha_forager_volume = 1.0f;
    long_config.alpha_forager_volatility = 1.0f;
    short_config.alpha_forager_volume = 0.5f;
    short_config.alpha_forager_volatility = 0.5f;
    for (int c = 0; c < 2; ++c) {
        float seed = c == 0 ? 90.0f : 190.0f;
        long_side.ema0[c] = seed;
        long_side.ema1[c] = seed;
        long_side.ema2[c] = seed;
        short_side.ema0[c] = seed + 20.0f;
        short_side.ema1[c] = seed + 20.0f;
        short_side.ema2[c] = seed + 20.0f;
        long_side.alpha0_coin[c] = 1.0f;
        long_side.alpha1_coin[c] = 0.5f;
        long_side.alpha2_coin[c] = 0.25f;
        short_side.alpha0_coin[c] = 0.5f;
        short_side.alpha1_coin[c] = 0.25f;
        short_side.alpha2_coin[c] = 0.125f;
        long_side.alpha_1m_coin[c] = 0.5f;
        short_side.alpha_1m_coin[c] = 0.25f;
        long_side.alpha_1h_coin[c] = 1.0f;
        short_side.alpha_1h_coin[c] = 0.5f;
        long_side.volatility_1m[c] = 0.0f;
        short_side.volatility_1m[c] = 0.0f;
        long_side.volatility_1h[c] = 0.0f;
        short_side.volatility_1h[c] = 0.0f;
        long_side.forager_volume[c] = 0.0f;
        short_side.forager_volume[c] = 0.0f;
        long_side.forager_volatility[c] = 0.0f;
        short_side.forager_volatility[c] = 0.0f;
        long_side.psize[c] = c == 0 ? 2.0f : 0.2f;
        long_side.pprice[c] = c == 0 ? 90.0f : 100.0f;
        short_side.psize[c] = c == 0 ? 3.0f : 0.2f;
        short_side.pprice[c] = c == 0 ? 110.0f : 300.0f;
    }
    output[0] = accumulate_ema_multicoin_side_unrealized_pnl(
        long_side, bars, coin_settings, 1, 2, false, 1.0e9f
    );
    output[1] = accumulate_ema_multicoin_side_unrealized_pnl(
        short_side, bars, coin_settings, 1, 2, true, 1.0e9f
    );
    update_ema_multicoin_side_indicators(
        long_side, long_config, bars, hour_log_ranges,
        coin_settings, 1, 2
    );
    update_ema_multicoin_side_indicators(
        short_side, short_config, bars, hour_log_ranges,
        coin_settings, 1, 2
    );
    output[2] = float(count_ema_multicoin_tradable_coins(
        bars, coin_settings, coin_overrides, 1, 2
    ));
    output[3] = ema_multicoin_side_has_position(long_side, 2) ? 1.0f : 0.0f;
    output[4] = ema_multicoin_side_has_position(short_side, 2) ? 1.0f : 0.0f;
    output[5] = long_side.ema0[0];
    output[6] = short_side.ema0[0];
    output[7] = long_side.forager_volatility[0];
    output[8] = short_side.forager_volatility[0];
    output[9] = long_side.forager_volume[0];
    output[10] = short_side.forager_volume[0];
    output[11] = long_side.volatility_1h[0];
}
"""

    bars = torch.tensor(
        [
            [[101.0, 99.0, 100.0, 10.0], [202.0, 198.0, 200.0, 20.0]],
            [[110.0, 90.0, 100.0, 10.0], [220.0, 180.0, 200.0, 20.0]],
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((2, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[:, 4] = 1.0
    coin_settings[:, 7] = 10.0
    coin_overrides = torch.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_overrides[1, 11] = 0.0
    hour_log_ranges = torch.tensor(
        [[-1.0, -1.0], [np.log(105.0 / 95.0), np.log(205.0 / 195.0)]],
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(12, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_candle_helpers_probe(
        bars,
        hour_log_ranges,
        coin_settings,
        coin_overrides,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    expected_range = np.log(110.0 / 90.0)
    expected_hour = np.log(105.0 / 95.0)
    np.testing.assert_allclose(
        output.cpu().numpy(),
        [
            1.0e9,
            1.0e9,
            1.0,
            1.0,
            1.0,
            100.0,
            105.0,
            expected_range,
            expected_range * 0.5,
            1_000.0,
            500.0,
            expected_hour,
        ],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_candle_helpers_advance_independent_sides():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_candle_helpers_probe(
    constant float* bars,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    TrailingMartingaleMulticoinSideConfig long_config = {};
    TrailingMartingaleMulticoinSideConfig short_config = {};
    long_config.alpha_forager_volume = 1.0f;
    long_config.alpha_forager_volatility = 1.0f;
    short_config.alpha_forager_volume = 0.5f;
    short_config.alpha_forager_volatility = 0.5f;
    for (int c = 0; c < 2; ++c) {
        float seed = c == 0 ? 90.0f : 190.0f;
        long_side.ema0[c] = seed;
        long_side.ema1[c] = seed;
        long_side.ema2[c] = seed;
        short_side.ema0[c] = seed + 20.0f;
        short_side.ema1[c] = seed + 20.0f;
        short_side.ema2[c] = seed + 20.0f;
        long_side.alpha0_coin[c] = 1.0f;
        long_side.alpha1_coin[c] = 0.5f;
        long_side.alpha2_coin[c] = 0.25f;
        short_side.alpha0_coin[c] = 0.5f;
        short_side.alpha1_coin[c] = 0.25f;
        short_side.alpha2_coin[c] = 0.125f;
        long_side.alpha_1m_coin[c] = 0.5f;
        short_side.alpha_1m_coin[c] = 0.25f;
        long_side.alpha_1h_coin[c] = 1.0f;
        short_side.alpha_1h_coin[c] = 0.5f;
        long_side.volatility_1m[c] = 0.0f;
        short_side.volatility_1m[c] = 0.0f;
        long_side.volatility_1h[c] = 0.0f;
        short_side.volatility_1h[c] = 0.0f;
        long_side.forager_volume[c] = 0.0f;
        short_side.forager_volume[c] = 0.0f;
        long_side.forager_volatility[c] = 0.0f;
        short_side.forager_volatility[c] = 0.0f;
        long_side.psize[c] = c == 0 ? 2.0f : 0.0f;
        long_side.pprice[c] = 90.0f;
        short_side.psize[c] = c == 0 ? 3.0f : 0.0f;
        short_side.pprice[c] = 110.0f;
        long_side.filled_coin[c] = false;
        short_side.filled_coin[c] = c == 0;
        long_side.min_since_open[c] = c == 0 ? 95.0f : INFINITY;
        long_side.max_since_min[c] = c == 0 ? 96.0f : 0.0f;
        long_side.max_since_open[c] = c == 0 ? 105.0f : 0.0f;
        long_side.min_since_max[c] = c == 0 ? 104.0f : INFINITY;
        short_side.min_since_open[c] = c == 0 ? 95.0f : INFINITY;
        short_side.max_since_min[c] = c == 0 ? 96.0f : 0.0f;
        short_side.max_since_open[c] = c == 0 ? 105.0f : 0.0f;
        short_side.min_since_max[c] = c == 0 ? 104.0f : INFINITY;
    }
    output[0] = accumulate_tm_multicoin_side_unrealized_pnl(
        long_side, bars, coin_settings, 1, 2, false, 1000.0f
    );
    output[1] = accumulate_tm_multicoin_side_unrealized_pnl(
        short_side, bars, coin_settings, 1, 2, true, 1000.0f
    );
    update_tm_multicoin_side_indicators(
        long_side, long_config, bars, hour_log_ranges, coin_settings, 1, 2
    );
    update_tm_multicoin_side_indicators(
        short_side, short_config, bars, hour_log_ranges, coin_settings, 1, 2
    );
    output[2] = float(count_tm_multicoin_tradable_coins(
        bars, coin_settings, coin_overrides, 1, 2
    ));
    output[3] = tm_multicoin_side_has_position(long_side, 2) ? 1.0f : 0.0f;
    output[4] = tm_multicoin_side_has_position(short_side, 2) ? 1.0f : 0.0f;
    output[5] = long_side.ema0[0];
    output[6] = short_side.ema0[0];
    output[7] = long_side.forager_volatility[0];
    output[8] = short_side.forager_volatility[0];
    output[9] = long_side.forager_volume[0];
    output[10] = short_side.forager_volume[0];
    output[11] = long_side.volatility_1h[0];
    output[12] = long_side.min_since_open[0];
    output[13] = long_side.max_since_min[0];
    output[14] = long_side.max_since_open[0];
    output[15] = long_side.min_since_max[0];
    output[16] = short_side.max_since_min[0];
    output[17] = short_side.max_since_open[0];
    output[18] = short_side.min_since_open[0];
    output[19] = short_side.min_since_max[0];
}
"""

    bars = torch.tensor(
        [
            [[101.0, 99.0, 100.0, 10.0], [202.0, 198.0, 200.0, 20.0]],
            [[110.0, 90.0, 100.0, 10.0], [220.0, 180.0, 200.0, 20.0]],
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((2, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[:, 4] = 1.0
    coin_settings[:, 7] = 10.0
    coin_overrides = _tm_probe_full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_overrides[1, 24] = 0.0
    hour_log_ranges = torch.full((2, 2), -1.0, dtype=torch.float32, device=gpu_device())
    hour_log_ranges[1, 0] = np.log(105.0 / 95.0)
    output = torch.zeros(20, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_candle_helpers_probe(
        bars,
        hour_log_ranges,
        coin_settings,
        coin_overrides,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    values = output.cpu().numpy()
    expected_range = np.log(110.0 / 90.0)
    expected_hour = np.log(105.0 / 95.0)
    np.testing.assert_allclose(
        values[:18],
        [
            1020.0,
            1030.0,
            1.0,
            1.0,
            1.0,
            100.0,
            105.0,
            expected_range,
            expected_range * 0.5,
            1000.0,
            500.0,
            expected_hour,
            90.0,
            100.0,
            110.0,
            100.0,
            0.0,
            0.0,
        ],
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.isinf(values[18])
    assert np.isinf(values[19])


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_fill_phase_shares_account_across_sides():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_shared_fill_phase_probe(
    constant int* market_execution_ticks,
    constant float* bars,
    constant int* fill_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* coin_fill_counts,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideState long_side = {};
    EmaMulticoinSideState short_side = {};
    long_side.hsl.signal_mode = HSL_SIGNAL_UNIFIED;
    short_side.hsl.signal_mode = HSL_SIGNAL_UNIFIED;
    long_side.psize[0] = 0.0f;
    short_side.psize[0] = 0.0f;
    long_side.pprice[0] = 0.0f;
    short_side.pprice[0] = 0.0f;
    long_side.entry_qty[0] = 1.0f;
    short_side.entry_qty[0] = 2.0f;
    long_side.entry_tick[0] = 95;
    short_side.entry_tick[0] = 105;
    long_side.close_qty[0] = 0.0f;
    short_side.close_qty[0] = 0.0f;
    long_side.secondary_close_qty[0] = 0.0f;
    short_side.secondary_close_qty[0] = 0.0f;
    long_side.close_is_unstuck_reducer[0] = false;
    short_side.close_is_unstuck_reducer[0] = false;
    long_side.close_is_hsl_panic[0] = false;
    short_side.close_is_hsl_panic[0] = false;
    long_side.position_open_k[0] = -1.0f;
    short_side.position_open_k[0] = -1.0f;
    long_side.position_last_fill_k[0] = -1.0f;
    short_side.position_last_fill_k[0] = -1.0f;
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;

    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    EmaMulticoinFillState fills = init_ema_multicoin_fill_state();
    bool long_filled = process_ema_multicoin_side_fills(
        market_execution_ticks,
        long_side, account, fills,
        bars, fill_ticks, coin_settings, coin_overrides,
        coin_fill_counts, 0, 1, 1, false, true,
        false, 0.0f, false, 1000.0f
    );
    float balance_after_long = account.balance;
    bool short_filled = process_ema_multicoin_side_fills(
        market_execution_ticks,
        short_side, account, fills,
        bars, fill_ticks, coin_settings, coin_overrides,
        coin_fill_counts, 0, 1, 1, true, true,
        false, 0.0f, false, balance_after_long
    );

    output[0] = long_filled ? 1.0f : 0.0f;
    output[1] = short_filled ? 1.0f : 0.0f;
    output[2] = balance_after_long;
    output[3] = account.balance;
    output[4] = account.realized_pnl_total;
    output[5] = account.realized_pnl_long;
    output[6] = account.realized_pnl_short;
    output[7] = fills.fill_count;
    output[8] = fills.fill_count_entry;
    output[9] = fills.fill_count_long;
    output[10] = fills.day_fill_count;
    output[11] = long_side.psize[0];
    output[12] = short_side.psize[0];
    output[13] = long_side.pprice[0];
    output[14] = short_side.pprice[0];
}
"""

    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[110.0, 90.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device=gpu_device(),
    )
    fill_ticks = torch.tensor(
        [[[100, 100]], [[110, 90]]], dtype=torch.int32, device=gpu_device()
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 1.0
    coin_settings[0, 1] = 1.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 5] = 0.01
    coin_settings[0, 7] = 10.0
    coin_overrides = torch.full(
        (1, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device=gpu_device())
    output = torch.zeros(15, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_shared_fill_phase_probe(
        _probe_market_execution_ticks(bars, [1.0]),
        bars,
        fill_ticks,
        coin_settings,
        coin_overrides,
        coin_fill_counts,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    np.testing.assert_allclose(
        output.cpu().numpy(),
        [
            1.0,
            1.0,
            999.05,
            996.95,
            -3.05,
            -0.95,
            -2.10,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            95.0,
            105.0,
        ],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_selection_phase_keeps_side_local_rankings():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_selection_phase_probe(
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    EmaMulticoinSideConfig long_config = {};
    EmaMulticoinSideConfig short_config = {};
    long_config.coin_hsl_mode = false;
    short_config.coin_hsl_mode = false;
    long_config.w_volume = 1.0f;
    short_config.w_volume = 1.0f;
    long_config.w_ready = 0.0f;
    short_config.w_ready = 0.0f;
    long_config.w_volatility = 0.0f;
    short_config.w_volatility = 0.0f;
    long_config.offset = 0.0f;
    short_config.offset = 0.0f;
    long_config.n_positions = 1;
    short_config.n_positions = 1;
    for (int c = 0; c < 3; ++c) {
        long_side.psize[c] = 0.0f;
        short_side.psize[c] = 0.0f;
        long_side.entry_qty[c] = 0.0f;
        short_side.entry_qty[c] = 0.0f;
        long_side.selected[c] = false;
        short_side.selected[c] = false;
        long_side.incumbent[c] = false;
        short_side.incumbent[c] = false;
        long_side.survivor[c] = false;
        short_side.survivor[c] = false;
        long_side.ema0[c] = 100.0f;
        long_side.ema1[c] = 100.0f;
        long_side.ema2[c] = 100.0f;
        short_side.ema0[c] = 100.0f;
        short_side.ema1[c] = 100.0f;
        short_side.ema2[c] = 100.0f;
        long_side.forager_volume[c] = float(3 - c) * 100.0f;
        short_side.forager_volume[c] = float(c + 1) * 100.0f;
        long_side.forager_volatility[c] = 0.0f;
        short_side.forager_volatility[c] = 0.0f;
    }
    ulong long_selection_blocked_mask = 0ul;
    ulong short_selection_blocked_mask = 0ul;
    ulong long_order_blocked_mask = 0ul;
    ulong short_order_blocked_mask = 0ul;
    compute_ema_multicoin_one_way_initial_blocks(
        long_side, long_config, coin_overrides,
        short_side, short_config, coin_overrides,
        bars, coin_settings, 1, 3, 0, 0, 1, 1, true, true, false, 0.0f,
        long_selection_blocked_mask, short_selection_blocked_mask,
        long_order_blocked_mask, short_order_blocked_mask
    );
    update_ema_multicoin_side_selection(
        long_side, long_config, bars, coin_settings, coin_overrides,
        1, 3, false, true, 1, 0.0f, 0ul, false, 0.0f
    );
    update_ema_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        1, 3, true, true, 1, 0.0f, 0ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[c] = long_side.selected[c] ? 1.0f : 0.0f;
        output[3 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
    // Opposite-held eligibility changes outside the side's own fills.
    // The mask transition must evict the blocked incumbent and promote next.
    update_ema_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        1, 3, true, false, 1, 0.0f, 4ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[6 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
    output[9] = float(long_selection_blocked_mask);
    output[10] = float(short_selection_blocked_mask);
    output[11] = float(long_order_blocked_mask);
    output[12] = float(short_order_blocked_mask);

    // No fill or mask transition occurs at k=2, but coin zero has reached
    // its tail. The validity transition must evict the flat incumbent.
    update_ema_multicoin_side_selection(
        long_side, long_config, bars, coin_settings, coin_overrides,
        2, 3, false, false, 1, 0.0f, 0ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[14 + c] = long_side.selected[c] ? 1.0f : 0.0f;
    }

    // With an existing position, the caller cannot prove a positive exact
    // cash-balance floor. A zero lower bound must keep every other flat slot
    // out of the Forager selection even when its exchange minimum is small.
    long_config.twel = 1.0f;
    long_config.allowance_pct = 0.0f;
    long_config.scale_hsl_budget = false;
    long_config.base_qty_pct = 1.0f;
    for (int c = 0; c < 3; ++c) {
        long_side.psize[c] = c == 0 ? 1.0f : 0.0f;
        long_side.selected[c] = false;
        long_side.incumbent[c] = false;
        long_side.survivor[c] = false;
    }
    update_ema_multicoin_side_selection(
        long_side, long_config, bars, coin_settings, coin_overrides,
        1, 3, false, true, 1, 0.0f, 0ul, true, 0.0f
    );
    output[13] = long_side.selected[1] || long_side.selected[2] ? 1.0f : 0.0f;
}
"""

    bars = torch.tensor(
        [
            [[100.0, 100.0, 100.0, 1.0]] * 3,
            [[100.0, 100.0, 100.0, 1.0]] * 3,
            [[100.0, 100.0, 100.0, 1.0]] * 3,
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((3, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[:, 7] = 10.0
    coin_settings[0, 7] = 1.0
    coin_settings[:, 12] = 1.0
    coin_overrides = torch.full(
        (3, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(17, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_selection_phase_probe(
        bars, coin_settings, coin_overrides, output, threads=(1, 1, 1)
    )
    synchronize()

    assert output.cpu().tolist() == [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 7.0,
        0.0,
        0.0, 1.0, 0.0,
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_order_phase_builds_both_sides_on_shared_account():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_order_phase_probe(
    constant float* bars,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideConfig long_config =
        load_ema_multicoin_side_config(params, 0);
    EmaMulticoinSideConfig short_config =
        load_ema_multicoin_side_config(params, PARAM_COLS);
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    init_ema_multicoin_side_state(
        long_side, long_config, coin_settings, coin_overrides, 1
    );
    init_ema_multicoin_side_state(
        short_side, short_config, coin_settings, coin_overrides, 1
    );
    long_side.selected[0] = true;
    short_side.selected[0] = true;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    generate_ema_multicoin_side_orders(
        long_side, long_config, account,
        bars, touch_ticks, coin_settings, coin_overrides,
        1, 1, false, 1, 1, 0,
        false, 0.001f, -2, 0ul
    );
    generate_ema_multicoin_side_orders(
        short_side, short_config, account,
        bars, touch_ticks, coin_settings, coin_overrides,
        1, 1, true, 1, 1, 0,
        false, 0.001f, -2, 0ul
    );
    output[0] = long_side.entry_qty[0];
    output[1] = short_side.entry_qty[0];
    output[2] = float(long_side.entry_tick[0]);
    output[3] = float(short_side.entry_tick[0]);
    output[4] = long_side.close_qty[0];
    output[5] = short_side.close_qty[0];
    output[6] = account.balance;
    output[7] = account.realized_pnl_total;
    generate_ema_multicoin_side_orders(
        long_side, long_config, account,
        bars, touch_ticks, coin_settings, coin_overrides,
        1, 1, false, 1, 1, 0,
        false, 0.001f, -2, 1ul
    );
    output[8] = long_side.entry_qty[0];
    output[9] = long_side.entry_candidate[0] ? 1.0f : 0.0f;
    output[10] = long_side.contribution[0];
}
"""

    _, row = _multicoin_exposure_fixture(
        "ema_anchor", "long", count=3, closes=(100.0, 100.0)
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.01
    short_row = list(row)
    short_row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("base_qty_pct")] = 0.5
    params = torch.tensor([row, short_row], dtype=torch.float32, device=gpu_device())
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[101.0, 99.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device=gpu_device(),
    )
    touch_ticks = torch.tensor(
        [[[10_000, 10_000]], [[10_000, 10_000]]],
        dtype=torch.int32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_settings[0, 9] = 100.0
    coin_settings[0, 10] = 1.0
    coin_overrides = torch.full(
        (1, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(11, dtype=torch.float32, device=gpu_device())

    library = compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_order_phase_probe(
        bars,
        touch_ticks,
        coin_settings,
        coin_overrides,
        params,
        output,
        threads=(1, 1, 1),
    )
    synchronize()

    values = output.cpu().numpy()
    assert values[0] > values[1] > 0.0
    assert values[2] < values[3]
    assert values[4] == 0.0
    assert values[5] == 0.0
    assert values[6] == 1_000.0
    assert values[7] == 0.0
    assert values[8:].tolist() == [0.0, 0.0, 0.0]


def _single_coin_exposure_fields(
    *, allowance_pct=0.0, scale_budget=False, entry_gate=True, threshold=1.0
):
    return [
        allowance_pct,
        float(scale_budget),
        float(entry_gate),
        threshold,
    ]


def _tm_wel_enforcer_fields(*, enabled=False, threshold=1.0):
    return [float(enabled), threshold]


def _tm_twel_enforcer_fields(
    *,
    enabled=False,
    unstuck_enabled=False,
    unstuck_ema_gating_enabled=True,
    unstuck_close_pct=0.1,
    unstuck_ema_dist=0.0,
    unstuck_loss_allowance_pct=0.1,
    unstuck_threshold=0.5,
):
    return [
        float(enabled),
        float(unstuck_enabled),
        float(unstuck_ema_gating_enabled),
        unstuck_close_pct,
        unstuck_ema_dist,
        unstuck_loss_allowance_pct,
        unstuck_threshold,
    ] + list(_HSL_DISABLED_VALUES.values())


_ADAPTIVE_DISABLED_VALUES = dict(
    zip(
        (
            "entry_cooldown_min_duration_minutes",
            "entry_cooldown_max_duration_minutes",
            "entry_cooldown_exposure_weight",
            "entry_cooldown_adverse_weight",
            "unilateralness_ema_span_1m",
            "forager_score_weights_unilateralness",
            "unilateralness_window",
        ),
        (0.0, -1.0, 0.0, 0.0, 60.0, 0.0, 1200.0),
    )
)

_UNSTUCK_DISABLED_VALUES = {
    "unstuck_enabled": 0.0,
    "unstuck_ema_gating_enabled": 1.0,
    "unstuck_close_pct": 0.1,
    "unstuck_ema_dist": 0.0,
    "unstuck_loss_allowance_pct": 0.1,
    "unstuck_threshold": 0.5,
}

_HSL_DISABLED_VALUES = {
    "hsl_enabled": 0.0,
    "hsl_red_threshold": 0.2,
    "hsl_drawdown_ratio_ema_span_minutes": 60.0,
    'hsl_cooldown_ms_lo': 0.0,
    'hsl_memory_ema_span_minutes': 43200.0,
    "hsl_signal_mode": 0.0,
    'hsl_cooldown_ms_hi': 0.0,
}




@pytest.mark.parametrize(
    ("keys", "minute_span_key", "hour_span_key"),
    [
        (
            EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
            "offset_volatility_ema_span_1m",
            "offset_volatility_ema_span_1h",
        ),
        (
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
            "volatility_ema_span_1m",
            "volatility_ema_span_1h",
        ),
    ],
)
def test_single_coin_interval_packing_scales_only_elapsed_minute_inputs(
    keys, minute_span_key, hour_span_key
):
    side = {key: float(index + 1) for index, key in enumerate(keys)}
    side.update(
        {
            "ema_span_0": 30.0,
            "ema_span_1": 90.0,
            minute_span_key: 120.0,
            hour_span_key: 24.0,
            "entry_cooldown_minutes": 2.0,
            "hsl_drawdown_ratio_ema_span_minutes": 60.0,
            'hsl_cooldown_ms_lo': 450000.0,
        }
    )
    side.update(_ADAPTIVE_DISABLED_VALUES)
    original = np.asarray(
        [[side[key] for key in keys] * 2], dtype=np.float64
    )

    scaled = _scale_directional_minute_parameters(
        original, keys, sides=2, interval_minutes=5.0
    )

    assert np.array_equal(
        original,
        np.asarray([[side[key] for key in keys] * 2], dtype=np.float64),
    )
    for side_index in range(2):
        offset = side_index * len(keys)
        assert scaled[0, offset + keys.index("ema_span_0")] == 6.0
        assert scaled[0, offset + keys.index("ema_span_1")] == 18.0
        assert scaled[0, offset + keys.index(minute_span_key)] == 24.0
        assert scaled[0, offset + keys.index("entry_cooldown_minutes")] == 0.4
        assert (
            scaled[
                0,
                offset + keys.index("hsl_cooldown_ms_lo"),
            ]
            == 450000.0
        )
        assert scaled[0, offset + keys.index(hour_span_key)] == 24.0
        hsl_span = scaled[0, offset + keys.index("hsl_drawdown_ratio_ema_span_minutes")]
        alpha_1m = 2.0 / 61.0
        alpha_5m = 2.0 / (hsl_span + 1.0)
        assert alpha_5m == pytest.approx(1.0 - (1.0 - alpha_1m) ** 5)


@pytest.mark.parametrize("span", [1.0, 5.5, 60.0, 720.0])
def test_single_coin_interval_packing_compounds_hsl_elapsed_minute_decay(span):
    side = {
        key: float(index + 1)
        for index, key in enumerate(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
    }
    side.update(_ADAPTIVE_DISABLED_VALUES)
    side["hsl_drawdown_ratio_ema_span_minutes"] = span
    original = np.asarray(
        [[side[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS] * 2],
        dtype=np.float64,
    )

    scaled = _scale_directional_minute_parameters(
        original,
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
        sides=2,
        interval_minutes=5.0,
    )

    alpha_1m = 2.0 / (span + 1.0)
    for side_index in range(2):
        offset = side_index * len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        effective_span = scaled[
            0,
            offset + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("hsl_drawdown_ratio_ema_span_minutes"),
        ]
        alpha_5m = 2.0 / (effective_span + 1.0)
        assert alpha_5m == pytest.approx(1.0 - (1.0 - alpha_1m) ** 5)


def test_single_coin_interval_packing_preserves_one_minute_hsl_span_exactly():
    values = np.arange(
        1,
        len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS) * 2 + 1,
        dtype=np.float64,
    ).reshape(1, -1)
    for side_index in range(2):
        for key, value in _ADAPTIVE_DISABLED_VALUES.items():
            values[
                0,
                side_index * len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
                + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(key),
            ] = value
        values[
            0,
            side_index * len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
            + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("hsl_drawdown_ratio_ema_span_minutes"),
        ] = 5.5

    scaled = _scale_directional_minute_parameters(
        values,
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
        sides=2,
        interval_minutes=1.0,
    )

    assert np.array_equal(scaled, values)


def test_single_coin_interval_packing_rejects_invalid_interval():
    values = np.zeros((1, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS) * 2), dtype=np.float64)

    with pytest.raises(ValueError, match="at least one minute"):
        _scale_directional_minute_parameters(
            values,
            EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
            sides=2,
            interval_minutes=0.0,
        )


def test_multicoin_ema_interval_packing_scales_forager_and_directional_spans():
    values = {
        key: float(index + 1)
        for index, key in enumerate(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
    }
    values.update(
        {
            "ema_span_0": 30.0,
            "ema_span_1": 90.0,
            "offset_volatility_ema_span_1h": 24.0,
            "offset_volatility_ema_span_1m": 120.0,
            "forager_volume_ema_span_1m": 180.0,
            "forager_volatility_ema_span_1m": 240.0,
            "entry_cooldown_minutes": 15.0,
            "hsl_drawdown_ratio_ema_span_minutes": 60.0,
            'hsl_cooldown_ms_lo': 2100000.0,
        }
    )
    values.update(_ADAPTIVE_DISABLED_VALUES)
    original = np.asarray(
        [[values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS] * 2],
        dtype=np.float64,
    )

    scaled = _scale_directional_minute_parameters(
        original,
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
        sides=2,
        interval_minutes=5.0,
    )

    for side_index in range(2):
        offset = side_index * len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        for key, expected in (
            ("ema_span_0", 6.0),
            ("ema_span_1", 18.0),
            ("offset_volatility_ema_span_1m", 24.0),
            ("forager_volume_ema_span_1m", 36.0),
            ("forager_volatility_ema_span_1m", 48.0),
            ("entry_cooldown_minutes", 3.0),
            ("hsl_cooldown_ms_lo", 2100000.0),
        ):
            assert (
                scaled[0, offset + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(key)]
                == expected
            )
        assert (
            scaled[
                0,
                offset
                + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(
                    "offset_volatility_ema_span_1h"
                ),
            ]
            == 24.0
        )


def test_multicoin_ema_interval_packing_scales_only_finite_coin_overrides():
    overrides = np.full((2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float64)
    ema0_column = EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS.index("ema_span_0")
    ema1_column = EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS.index("ema_span_1")
    volatility_1m_column = EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS.index(
        "offset_volatility_ema_span_1m"
    )
    hsl_span_column = EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN + 2
    hsl_cooldown_column = EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN + 3
    overrides[0, ema0_column] = 35.0
    overrides[0, ema1_column] = 70.0
    overrides[0, volatility_1m_column] = 140.0
    overrides[0, EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN] = 21.0
    overrides[0, hsl_span_column] = 60.0
    overrides[0, hsl_cooldown_column] = 28.0

    scaled = _scale_ema_multicoin_coin_overrides(overrides, 7.0)

    assert scaled[0, ema0_column] == 5.0
    assert scaled[0, ema1_column] == 10.0
    assert scaled[0, volatility_1m_column] == 20.0
    assert scaled[0, EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN] == 3.0
    assert scaled[0, hsl_cooldown_column] == 28.0
    alpha_7m = 2.0 / (float(scaled[0, hsl_span_column]) + 1.0)
    assert alpha_7m == pytest.approx(1.0 - (1.0 - 2.0 / 61.0) ** 7)
    assert np.isnan(scaled[1]).all()


def test_multicoin_tm_interval_packing_scales_forager_and_directional_spans():
    values = {
        key: float(index + 1)
        for index, key in enumerate(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
    }
    values.update(
        {
            "ema_span_0": 30.0,
            "ema_span_1": 90.0,
            "volatility_ema_span_1h": 24.0,
            "volatility_ema_span_1m": 120.0,
            "forager_volume_ema_span_1m": 180.0,
            "forager_volatility_ema_span_1m": 240.0,
            "entry_cooldown_minutes": 15.0,
            "hsl_drawdown_ratio_ema_span_minutes": 60.0,
            'hsl_cooldown_ms_lo': 2100000.0,
        }
    )
    values.update(_ADAPTIVE_DISABLED_VALUES)
    original = np.asarray(
        [[values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS] * 2],
        dtype=np.float64,
    )

    scaled = _scale_directional_minute_parameters(
        original,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
        sides=2,
        interval_minutes=5.0,
    )

    for side_index in range(2):
        offset = side_index * len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
        for key, expected in (
            ("ema_span_0", 6.0),
            ("ema_span_1", 18.0),
            ("volatility_ema_span_1m", 24.0),
            ("forager_volume_ema_span_1m", 36.0),
            ("forager_volatility_ema_span_1m", 48.0),
            ("entry_cooldown_minutes", 3.0),
            ("hsl_cooldown_ms_lo", 2100000.0),
        ):
            assert (
                scaled[
                    0,
                    offset + TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key),
                ]
                == expected
            )
        assert (
            scaled[
                0,
                offset
                + TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                    "volatility_ema_span_1h"
                ),
            ]
            == 24.0
        )


def test_multicoin_tm_interval_packing_scales_only_finite_coin_overrides():
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float64
    )
    override_keys = tuple(key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
    ema0_column = override_keys.index("ema_span_0")
    ema1_column = override_keys.index("ema_span_1")
    volatility_1m_column = override_keys.index("volatility_ema_span_1m")
    volatility_1h_column = override_keys.index("volatility_ema_span_1h")
    hsl_span_column = TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN + 2
    hsl_cooldown_column = TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN + 3
    overrides[0, ema0_column] = 35.0
    overrides[0, ema1_column] = 70.0
    overrides[0, volatility_1m_column] = 140.0
    overrides[0, volatility_1h_column] = 24.0
    overrides[0, TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN] = 21.0
    overrides[0, hsl_span_column] = 60.0
    overrides[0, hsl_cooldown_column] = 28.0

    scaled = _scale_tm_multicoin_coin_overrides(overrides, 7.0)

    assert scaled[0, ema0_column] == 5.0
    assert scaled[0, ema1_column] == 10.0
    assert scaled[0, volatility_1m_column] == 20.0
    assert scaled[0, volatility_1h_column] == 24.0
    assert scaled[0, TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN] == 3.0
    assert scaled[0, hsl_cooldown_column] == 28.0
    alpha_7m = 2.0 / (float(scaled[0, hsl_span_column]) + 1.0)
    assert alpha_7m == pytest.approx(1.0 - (1.0 - 2.0 / 61.0) ** 7)
    assert np.isnan(scaled[1]).all()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_service_dispatches_forced_delist_tail(strategy_kind, topology):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import SharedAccountReplay

    entry_k = 7
    last_valid = 10
    count = last_valid + 1401
    hlcvs = np.full((count, 2, 4), np.nan, dtype=np.float64)
    hlcvs[: last_valid + 1, 0] = [100.0, 100.0, 100.0, 1.0]
    hlcvs[:, 1] = [200.0, 200.0, 200.0, 1.0]
    if topology in {"long", "fused"}:
        hlcvs[entry_k, 0, 1] = 80.0
    if topology in {"short", "fused"}:
        hlcvs[entry_k, 0, 0] = 120.0
    delist_close = {
        "long": 50.0,
        "short": 150.0,
        "fused": 100.0,
    }[topology]
    hlcvs[last_valid, 0] = [
        delist_close,
        delist_close,
        delist_close,
        1.0,
    ]
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market_settings = {
        coin: {
            "qty_step": 0.001,
            "price_step": 0.01,
            "min_qty": 0.001,
            "min_cost": 5.0,
            "c_mult": 1.0,
            "maker": 0.0,
            "taker": 0.001,
            "exchange": "bybit",
            "first_valid_index": 0,
            "last_valid_index": last,
            "warmup_minutes": 1,
        }
        for coin, last in (("BTC", last_valid), ("ETH", count - 1))
    }
    market_settings["__meta__"] = {
        "requested_start_ts": int(timestamps[0]),
        "requested_start_date": "2023-11-14",
        "warmup_minutes_requested": 1,
    }
    config = get_template_config()
    config['bot']['global']['strategy_kind'] = strategy_kind
    config['bot']['global']['warmup']['max_minutes'] = 1
    enabled_sides = ("long", "short") if topology == "fused" else (topology,)
    set_side_universe_sources(config, 'approved_coins', {
        side: ["BTC"] if side in enabled_sides else [] for side in ("long", "short")
    })
    config['bot']['global']['hedge_mode'] = topology == "fused"
    config["backtest"]["coins"] = {"bybit": ["BTC", "ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.1 if enabled else 0.0
        risk["n_positions"] = 1 if enabled else 0
        config["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 100.0
        risk["we_excess_allowance_pct"] = 0.0
        risk['position_exposure_enforcer']['enabled'] = False
        risk['total_exposure_enforcer']['enabled'] = False
        config["bot"][side]["hsl"]["enabled"] = False
        config["bot"][side]["unstuck"]["enabled"] = False
    for side in enabled_sides:
        if strategy_kind == "ema_anchor":
            config["bot"][side]["strategy"]["ema_anchor"].update(
                {
                    "base_qty_pct": 1.0,
                    "ema_span_0": 2.0,
                    "ema_span_1": 3.0,
                    "offset": 0.1,
                    "offset_psize_weight": 0.0,
                }
            )
        else:
            strategy = config["bot"][side]["strategy"]["trailing_martingale"]
            strategy["entry"]["ema_span_0"] = 2.0
            strategy["entry"]["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 1.0
            strategy["entry"]["initial_ema_dist"] = 0.1
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.5

    # Exercise the canonical pre-fill BTC replay before the delayed entry.
    btc_prices = np.full(count, 40_000.0)
    btc_prices[:entry_k] = 60_000.0
    proxy = SharedAccountReplay(
        config=config,
        hlcvs=hlcvs,
        mss=market_settings,
        btc=btc_prices,
        timestamps=timestamps,
        exchange="bybit",
        batch_size=1,
        needed_metrics={
            "backtest_completion_ratio",
            "fills_per_day",
            "adg_btc",
            "drawdown_worst_btc",
            "equity_balance_diff_neg_max_btc",
            "equity_balance_diff_neg_max_usd",
            "equity_balance_diff_neg_mean_btc",
            "equity_balance_diff_neg_mean_usd",
            "expected_shortfall_1pct_btc",
            "entry_interval_hours_max",
            "paper_loss_mean_ratio_btc",
            "paper_loss_mean_ratio_usd",
            "paper_loss_ratio_btc",
            "paper_loss_ratio_usd",
            "strategy_eq_recovery_days_p99",
        },
    )
    result = proxy.evaluate([{}])[0]
    exact_fills, _, exact_analysis = run_backtest(
        hlcvs,
        market_settings,
        config,
        "bybit",
        btc_prices,
        timestamps,
    )
    expected_fills = 4.0 if topology == "fused" else 2.0
    assert exact_analysis["fills_count"] == expected_fills
    assert len(exact_fills) == exact_analysis["fills_count"]
    assert result["fills_per_day"] == pytest.approx(
        exact_analysis["fills_per_day"], rel=1.0e-5
    )
    assert np.isfinite(result["entry_interval_hours_max"])
    assert np.isfinite(result["strategy_eq_recovery_days_p99"])
    exact_order_types = {row[13] for row in exact_fills}
    expected_panic_types = {f"close_panic_{side}" for side in enabled_sides}
    assert expected_panic_types <= exact_order_types
    assert result["adg_btc"] == pytest.approx(exact_analysis["adg_btc"], rel=2.0e-3)
    assert result["drawdown_worst_btc"] == pytest.approx(
        exact_analysis["drawdown_worst_btc"], rel=2.0e-3
    )
    assert result["expected_shortfall_1pct_btc"] == pytest.approx(
        exact_analysis["expected_shortfall_1pct_btc"], rel=2.0e-3
    )
    for metric in (
        "equity_balance_diff_neg_max_btc",
        "equity_balance_diff_neg_max_usd",
        "equity_balance_diff_neg_mean_btc",
        "equity_balance_diff_neg_mean_usd",
        "paper_loss_mean_ratio_btc",
        "paper_loss_mean_ratio_usd",
        "paper_loss_ratio_btc",
        "paper_loss_ratio_usd",
    ):
        assert result[metric] == pytest.approx(
            exact_analysis[metric], rel=3.0e-3, abs=1.0e-8
        ), metric
    assert proxy.btc_risk_enabled is True
    assert proxy.equity_balance_diff_enabled is True
    assert result["backtest_completion_ratio"] > 0.99

    if topology == "long":
        ebd_only_proxy = SharedAccountReplay(
            config=config,
            hlcvs=hlcvs,
            mss=market_settings,
            btc=btc_prices,
            timestamps=timestamps,
            exchange="bybit",
            batch_size=1,
            needed_metrics={"paper_loss_ratio_btc"},
        )
        ebd_only_result = ebd_only_proxy.evaluate([{}])[0]
        assert ebd_only_proxy.btc_risk_enabled is False
        assert ebd_only_proxy.equity_balance_diff_enabled is True
        assert ebd_only_result["paper_loss_ratio_btc"] == pytest.approx(
            exact_analysis["paper_loss_ratio_btc"],
            rel=3.0e-3,
            abs=1.0e-8,
        )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused-hedge", "fused-one-way"])
@pytest.mark.parametrize(
    ("count", "first_valid_indices", "last_valid_indices"),
    [
        (12, (0, 0), (6, 7)),
        (12, (0, 9), (6, 11)),
        (1408, (0, 0), (6, 7)),
    ],
    ids=[
        "all-coins-ended-tail",
        "gap-between-coin-histories",
        "all-coins-forced-delisted",
    ],
)
def test_mps_multicoin_service_matches_exact_declared_all_invalid_time(
    strategy_kind, topology, count, first_valid_indices, last_valid_indices
):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import SharedAccountReplay

    hlcvs = np.full((count, 2, 4), 100.0, dtype=np.float64)
    hlcvs[:, :, 3] = 1.0
    if topology in {"long", "fused-hedge", "fused-one-way"}:
        hlcvs[3, 0, 1] = 80.0
    if topology in {"short", "fused-hedge", "fused-one-way"}:
        hlcvs[3, 0, 0] = 120.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market_settings = {
        coin: {
            "qty_step": 0.001,
            "price_step": 0.01,
            "min_qty": 0.001,
            "min_cost": 5.0,
            "c_mult": 1.0,
            "maker": 0.0,
            "taker": 0.001,
            "exchange": "bybit",
            "first_valid_index": first,
            "last_valid_index": last,
            "warmup_minutes": 1,
        }
        for coin, first, last in zip(
            ("BTC", "ETH"), first_valid_indices, last_valid_indices
        )
    }
    market_settings["__meta__"] = {
        "requested_start_ts": int(timestamps[0]),
        "requested_start_date": "2023-11-14",
        "warmup_minutes_requested": 1,
    }
    config = get_template_config()
    config['bot']['global']['strategy_kind'] = strategy_kind
    config['bot']['global']['warmup']['max_minutes'] = 1
    enabled_sides = ("long", "short") if topology.startswith("fused-") else (topology,)
    set_side_universe_sources(config, 'approved_coins', {
        side: ["BTC"] if side in enabled_sides else [] for side in ("long", "short")
    })
    config['bot']['global']['hedge_mode'] = topology == "fused-hedge"
    config["backtest"]["coins"] = {"bybit": ["BTC", "ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    config["backtest"]["dynamic_wel_by_tradability"] = False
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.1 if enabled else 0.0
        risk["n_positions"] = 2 if enabled else 0
        config["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 100.0
        risk["we_excess_allowance_pct"] = 0.0
        risk['position_exposure_enforcer']['enabled'] = False
        risk['total_exposure_enforcer']['enabled'] = False
        config["bot"][side]["hsl"]["enabled"] = enabled
        config["bot"][side]["hsl"]["red_threshold"] = 0.9
        config["bot"][side]["hsl"]["ema_span_minutes"] = 1.0
        config["bot"][side]["unstuck"]["enabled"] = False
    for side in enabled_sides:
        if strategy_kind == "ema_anchor":
            config["bot"][side]["strategy"]["ema_anchor"].update(
                {
                    "base_qty_pct": 1.0,
                    "ema_span_0": 2.0,
                    "ema_span_1": 3.0,
                    "offset": 0.1,
                    "offset_psize_weight": 0.0,
                }
            )
        else:
            strategy = config["bot"][side]["strategy"]["trailing_martingale"]
            strategy["entry"]["ema_span_0"] = 2.0
            strategy["entry"]["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 1.0
            strategy["entry"]["initial_ema_dist"] = 0.1
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.5

    proxy = SharedAccountReplay(
        config=config,
        hlcvs=hlcvs,
        mss=market_settings,
        btc=np.full(count, 50_000.0),
        timestamps=timestamps,
        exchange="bybit",
        batch_size=1,
        needed_metrics={
            "backtest_completion_ratio",
            "entry_initial_balance_pct_long",
            "entry_initial_balance_pct_short",
            "fills_per_day",
            "position_held_days_max",
            "position_held_time_weighted_mean_hours",
        },
    )
    if count < 1400:
        with pytest.raises(ValueError, match="held-position valuation"):
            proxy.evaluate([{}])
        with pytest.raises(ValueError, match="held-position valuation"):
            run_backtest(
                hlcvs,
                market_settings,
                config,
                "bybit",
                np.full(count, 50_000.0),
                timestamps,
            )
        return
    result = proxy.evaluate([{}])[0]
    exact_fills, _, exact_analysis = run_backtest(
        hlcvs,
        market_settings,
        config,
        "bybit",
        np.full(count, 50_000.0),
        timestamps,
    )

    assert len(exact_fills) == exact_analysis["fills_count"]
    assert result["fills_per_day"] == pytest.approx(
        exact_analysis["fills_per_day"], rel=1.0e-5
    )
    for side in enabled_sides:
        metric = f"entry_initial_balance_pct_{side}"
        assert result[metric] == pytest.approx(exact_analysis[metric], rel=1.0e-5)
    # The synthetic exact helper retains the template's calendar date span,
    # whereas the proxy receives this prepared span directly. The
    # proxy must nevertheless cover the all-invalid tail through its endpoint.
    assert result["backtest_completion_ratio"] > 0.9
    assert result["position_held_time_weighted_mean_hours"] == pytest.approx(
        exact_analysis["position_held_time_weighted_mean_hours"], rel=1.0e-5
    )
    assert result["position_held_days_max"] == pytest.approx(
        exact_analysis["position_held_days_max"], rel=1.0e-5
    )




def _probe_market_execution_ticks(bars, price_steps):
    import passivbot_rust
    closes = bars.detach().cpu().numpy().reshape(-1, len(price_steps), 4)[:, :, 2].astype(np.float64)
    ticks = passivbot_rust.prepare_gpu_market_execution_ticks(
        closes, price_steps, 0.0, [0] * len(price_steps),
        [len(closes) - 1] * len(price_steps),
    )
    return torch.as_tensor(ticks, dtype=torch.int32, device=bars.device)


def _multicoin_exposure_fixture(
    strategy_kind,
    side,
    coin_overrides=None,
    *,
    count=64,
    markets=None,
    closes=(100.0, 120.0),
    highs=None,
    lows=None,
    max_realized_loss_pct=1.0,
    first_valid_indices=(0, 0),
    last_valid_indices=None,
    liquidation_threshold=0.05,
    collect_coin_fill_counts=False,
    filter_by_min_effective_cost=False,
    market_order_slippage_pct=0.0,
    market_orders_allowed=False,
    market_order_near_touch_threshold=0.001,
    hsl_panic_market=False,
    recovery_distribution_enabled=False,
    entry_interval_enabled=False,
    dynamic_wel_by_tradability=True,
    requested_start_index=0,
    return_context=False,
    interval_minutes=1,
):
    coin_count = 2
    if last_valid_indices is None:
        last_valid_indices = (count - 1,) * coin_count
    interval_ms = int(interval_minutes) * 60_000
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * interval_ms
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    close_matrix = np.asarray(closes, dtype=np.float64)
    for coin in range(coin_count):
        close = (
            np.full(count, close_matrix[coin])
            if close_matrix.ndim == 1
            else close_matrix[:, coin]
        )
        hlcvs[:, coin, 0] = (
            close * 1.01 if highs is None else np.asarray(highs)[:, coin]
        )
        hlcvs[:, coin, 1] = close * 0.99 if lows is None else np.asarray(lows)[:, coin]
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    if markets is None:
        markets = [
            ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0) for _ in range(coin_count)
        ]
    runs = [
        ProxyRun(
            1_000.0,
            1,
            max(1, first_valid_indices[coin]),
            int(timestamps[requested_start_index]),
            int(timestamps[requested_start_index]),
            int(timestamps[0]),
            interval_ms,
            liquidation_threshold,
            first_valid_indices[coin],
            last_valid_indices[coin],
        )
        for coin in range(coin_count)
    ]
    data = build_mps_multicoin_data(
        hlcvs,
        timestamps,
        runs,
        markets,
        include_hourly_ranges=True,
        market_order_slippage_pct=market_order_slippage_pct,
    )
    if strategy_kind == "ema_anchor":
        values = {
            "base_qty_pct": 1.0,
            "ema_span_0": 2.0,
            "ema_span_1": 3.0,
            "entry_double_down_factor": 0.0,
            "offset": 0.0,
            "offset_psize_weight": 0.0,
            "offset_volatility_1h_weight": 0.0,
            "offset_volatility_1m_weight": 0.0,
            "offset_volatility_ema_span_1h": 2.0,
            "offset_volatility_ema_span_1m": 2.0,
            "entry_cooldown_minutes": 0.0,
            "total_wallet_exposure_limit": 1.0,
            "forager_volume_ema_span_1m": 2.0,
            "forager_volatility_ema_span_1m": 2.0,
            "forager_score_weights_volume": 1.0,
            "forager_score_weights_ema_readiness": 0.0,
            "forager_score_weights_volatility": 0.0,
            "n_positions": 2.0,
            "we_excess_allowance_pct": 0.0,
            "hsl_scale_budget_with_excess_allowance": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
            "twel_enforcer_reduce_portfolio": 0.0,
            **_UNSTUCK_DISABLED_VALUES,
            **_ADAPTIVE_DISABLED_VALUES,
            "unstuck_ema_span_0": 2.0,
            "unstuck_ema_span_1": 3.0,
            **_HSL_DISABLED_VALUES,
        "unstuck_memory_ema_span_minutes": 43200.0,
        "loss_gate_memory_ema_span_minutes": 43200.0,
        }
        row = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]
        runner = MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            pnl_lookback_bars=1440 if interval_minutes == 1 else 0,
            side=side,
            coin_overrides=coin_overrides,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            filter_by_min_effective_cost=filter_by_min_effective_cost,
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market,
            recovery_distribution_enabled=recovery_distribution_enabled,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
        )
    else:
        values = {
            "ema_span_0": 2.0,
            "ema_span_1": 3.0,
            "volatility_ema_span_1h": 2.0,
            "volatility_ema_span_1m": 2.0,
            "entry_double_down_factor": 0.0,
            "entry_initial_ema_dist": 0.0,
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_threshold_we_weight": 0.0,
            "entry_threshold_volatility_1h_weight": 0.0,
            "entry_threshold_volatility_1m_weight": 0.0,
            "entry_retracement_base_pct": 0.0,
            "entry_retracement_we_weight": 0.0,
            "entry_retracement_volatility_1h_weight": 0.0,
            "entry_retracement_volatility_1m_weight": 0.0,
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 10.0,
            "close_threshold_we_weight": 0.0,
            "close_threshold_volatility_1h_weight": 0.0,
            "close_threshold_volatility_1m_weight": 0.0,
            "close_retracement_base_pct": 0.0,
            "close_retracement_volatility_1h_weight": 0.0,
            "close_retracement_volatility_1m_weight": 0.0,
            "entry_cooldown_minutes": 0.0,
            "total_wallet_exposure_limit": 1.0,
            "gate_initial": 0.0,
            "gate_reentry": 0.0,
            "forager_volume_ema_span_1m": 2.0,
            "forager_volatility_ema_span_1m": 2.0,
            "forager_score_weights_volume": 1.0,
            "forager_score_weights_ema_readiness": 0.0,
            "forager_score_weights_volatility": 0.0,
            "n_positions": 2.0,
            "we_excess_allowance_pct": 0.0,
            "hsl_scale_budget_with_excess_allowance": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "wel_enforcer_enabled": 0.0,
            "wel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
            "twel_enforcer_reduce_portfolio": 0.0,
            **_UNSTUCK_DISABLED_VALUES,
            **_ADAPTIVE_DISABLED_VALUES,
            "unstuck_ema_span_0": 2.0,
            "unstuck_ema_span_1": 3.0,
            **_HSL_DISABLED_VALUES,
        "unstuck_memory_ema_span_minutes": 43200.0,
        "loss_gate_memory_ema_span_minutes": 43200.0,
        }
        row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
        runner = MpsTrailingMartingaleMulticoinRunner(
            runs[0],
            data,
            pnl_lookback_bars=1440 if interval_minutes == 1 else 0,
            side=side,
            coin_overrides=coin_overrides,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            filter_by_min_effective_cost=filter_by_min_effective_cost,
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market,
            recovery_distribution_enabled=recovery_distribution_enabled,
            entry_interval_enabled=entry_interval_enabled,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
        )
    if return_context:
        return runner, row, runs[0], data
    return runner, row


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("forced_delist", [False, True])
@pytest.mark.parametrize("collect_counts", [False, True])
@pytest.mark.parametrize("signal_mode", [0, 1, 2])
def test_ema_disabled_hsl_preserves_optional_fill_counts(
    side, forced_delist, collect_counts, signal_mode
):
    from optimization.gpu.metrics import _fill_activity_metrics

    count = 1509 if forced_delist else 313
    steps = np.arange(count)
    closes = np.column_stack(
        [
            base * (1.0 + 0.09 * np.sin(steps / 17.0 + coin))
            for coin, base in enumerate((100.0, 120.0))
        ]
    )
    runner, row, run, _ = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=count,
        closes=closes,
        last_valid_indices=(89, 97) if forced_delist else None,
        markets=[ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001, 0.02)] * 2,
        collect_coin_fill_counts=collect_counts,
        return_context=True,
    )
    params = np.asarray([row] * 3, dtype=np.float64)
    params[:, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("hsl_signal_mode")] = signal_mode

    output = runner.run(params)
    synchronize()
    compact = {key: value.cpu().clone() for key, value in output.items()}
    assert runner.dispatch_hsl_disabled is (signal_mode != 2)
    runner.hsl_disabled_specialization = False
    baseline = runner.run(params)
    assert runner.dispatch_hsl_disabled is False
    assert compact.keys() == baseline.keys()
    for key, expected in compact.items():
        torch.testing.assert_close(
            baseline[key].cpu(), expected, rtol=0, atol=0, equal_nan=True,
            msg=lambda message: f"{key}: {message}",
        )
    assert torch.all(compact["fill_count_entry"] > 0)
    assert torch.all(compact["fill_count"] > compact["fill_count_entry"])
    if forced_delist:
        assert torch.all(compact["hsl_panic_close_loss_sum"] > 0)
    assert ("coin_fill_counts" in compact) is collect_counts
    if collect_counts:
        torch.testing.assert_close(
            compact["coin_fill_counts"].sum(dim=1),
            compact["fill_count"],
            rtol=0,
            atol=0,
        )
        requested = {"fills_top_symbol_share", "fills_active_symbols_count"}
        metrics = _fill_activity_metrics(compact, run, requested)
        assert torch.all(metrics["fills_active_symbols_count"] == 2)
        assert torch.all(metrics["fills_top_symbol_share"] >= 0.5)
        assert torch.all(metrics["fills_top_symbol_share"] <= 1.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_ema_disabled_hsl_mixed_batch_preserves_each_candidate(side):
    count = 1509
    steps = np.arange(count)
    closes = np.column_stack(
        [base * (1.0 + 0.09 * np.sin(steps / 17.0 + coin))
         for coin, base in enumerate((100.0, 120.0))]
    )
    runner, row = _multicoin_exposure_fixture(
        "ema_anchor", side, count=count, closes=closes,
        last_valid_indices=(89, 97), collect_coin_fill_counts=True,
        markets=[ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001, 0.02)] * 2,
    )
    params = np.asarray([row] * 3, dtype=np.float64)
    params[:, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("hsl_signal_mode")] = [0, 1, 2]
    mixed = {key: value.cpu().clone() for key, value in runner.run(params).items()}
    assert runner.dispatch_hsl_disabled is False
    assert torch.all(mixed["hsl_panic_close_loss_sum"] > 0)
    for index, mode in enumerate((0, 1, 2)):
        individual = runner.run(params[index:index + 1])
        assert runner.dispatch_hsl_disabled is (mode != 2)
        for key, expected in mixed.items():
            torch.testing.assert_close(
                individual[key].cpu(), expected[index:index + 1],
                rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{key}: {message}",
            )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("batch_size", [3, 35])
def test_tm_multicoin_temporal_replay_preserves_every_output(
    side, features, batch_size
):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner

    count = 1513
    steps = np.arange(count)
    closes = np.column_stack(
        [
            base * (1 + 0.12 * np.sin(steps / 37.0 + coin))
            for coin, base in enumerate((100.0, 120.0))
        ]
    )
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        requested_start_index=31,
        return_context=True,
    )
    if features:
        for key, value in (
            ("unstuck_enabled", 1.0),
            ("unstuck_ema_gating_enabled", 1.0),
            ("unstuck_ema_span_0", 7.25),
            ("unstuck_ema_span_1", 133.5),
            ("hsl_enabled", 1.0),
            ("hsl_signal_mode", 2.0),
            ("hsl_red_threshold", 0.02),
            ("hsl_drawdown_ratio_ema_span_minutes", 1.0),
            ("hsl_cooldown_ms_lo", 3.0),
        ):
            row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    kwargs = dict(
        side=side,
        collect_coin_fill_counts=True,
        recovery_distribution_enabled=features,
        raw_strategy_risk_enabled=True,
        entry_interval_enabled=features,
        hsl_ema_tail_enabled=features,
        hsl_raw_drawdown_enabled=features,
        hsl_raw_tail_enabled=features,
        btc_risk_enabled=features,
        equity_balance_diff_enabled=features,
        btc_prices=30_000.0 + steps if features else None,
    )
    generic = MpsTrailingMartingaleMulticoinRunner(
        run, data, pnl_lookback_bars=1440, **kwargs
    )
    # Include empty and early-finished candidates alongside a full replay. The
    # boundary crosses activation, hours, UTC days, and HSL episodes.
    params = np.asarray([row] * batch_size, dtype=np.float64)
    ends = np.resize(np.asarray([1, 123, count - 1], dtype=np.int32), batch_size)
    expected = generic.run(params, end_steps=ends)
    expected = {
        key: value.cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in expected.items()
    }
    chunked = MpsTrailingMartingaleMulticoinRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        max_dispatch_candidate_bars=batch_size * 2 * 47,
        **kwargs,
    )
    for _ in range(2):
        actual = chunked.run(params, profile=True, end_steps=ends)
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value.cpu(), expected[key], rtol=0, atol=0, equal_nan=True
                )
            else:
                assert value == expected[key]
        assert chunked.last_profile["dispatch_count"] == 33
        assert chunked.last_profile["kernel_candidate_steps"] == int((ends - 1).sum())
        assert chunked.last_profile["temporal_chunk_bars"] == 47
        assert chunked.last_profile["threads_per_threadgroup"] == min(batch_size, 32)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_tm_multicoin_temporal_replay_interrupt_does_not_leak_partial_state():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner

    generic, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale", "long", return_context=True
    )
    matrix = np.asarray([row], dtype=np.float64)
    expected = {key: value.cpu().clone() for key, value in generic.run(matrix).items()}
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("test interruption")

    chunked = MpsTrailingMartingaleMulticoinRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        side="long",
        max_dispatch_candidate_bars=2 * 7,
        interrupt_check=interrupt,
    )
    with pytest.raises(InterruptedError, match="test interruption"):
        chunked.run(matrix)
    chunked.interrupt_check = lambda: None
    actual = chunked.run(matrix)
    for key, value in actual.items():
        torch.testing.assert_close(
            value.cpu(), expected[key], rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_tm_multicoin_temporal_replay_preserves_unavailable_valuation():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner

    generic, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=64,
        last_valid_indices=(24, 63),
        return_context=True,
    )
    chunked = MpsTrailingMartingaleMulticoinRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        side="long",
        max_dispatch_candidate_bars=2 * 7,
    )
    for runner in (generic, chunked):
        with pytest.raises(ValueError, match="unavailable held-position valuation"):
            runner.run(np.asarray([row], dtype=np.float64))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "marker,error,message",
    [
        (-2, ValueError, "held-position valuation"),
        (-3, RuntimeError, "PnL history overflow"),
        (-4, ValueError, "HSL controller inputs"),
    ],
)
def test_tm_multicoin_temporal_replay_stops_on_fatal_marker(
    side, marker, error, message
):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner

    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale", side, return_context=True
    )
    calls = 0

    def inject_failure():
        nonlocal calls
        calls += 1
        if calls == 2:
            # A completed early dispatch reported fatal input/history failure.
            # The next compact status check must stop replay immediately.
            runner._buffers[1][1][0, 9] = marker

    runner = MpsTrailingMartingaleMulticoinRunner(
        run,
        data,
        side=side,
        pnl_lookback_bars=1440,
        max_dispatch_candidate_bars=2 * 7,
        interrupt_check=inject_failure,
    )
    matrix = np.asarray([row], dtype=np.float64)
    with pytest.raises(error, match=message):
        runner.run(matrix)
    # Fatal statuses are checked after every completed chunk. Waiting for a
    # third interrupt callback would require continuing work after that failure.
    assert calls == 2
    assert runner._buffers[1][1][0, 9].item() == marker
    runner.interrupt_check = lambda: None
    assert runner.run(matrix)["alive"].item()  # A new replay resets fatal state.


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_equity_recovery_includes_unrecovered_final_tail(strategy_kind, side):
    count = 32
    final_multiplier = 0.7 if side == "long" else 1.3
    closes = np.column_stack(
        [
            np.linspace(initial, initial * final_multiplier, count)
            for initial in (100.0, 120.0)
        ]
    )
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        highs=closes if side == "long" else closes * 1.01,
        lows=closes * 0.99 if side == "long" else closes,
    )
    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["fill_count"].item() > 0.0
    # Every mark after the first is below the account's initial equity peak.
    # The still-open drawdown must span the complete recorded equity interval.
    duration = output["last_eq_ts"] - output["first_eq_ts"]
    assert duration.item() > 0.0
    assert output["account_recovery_max_ms"].item() == duration.item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_runner_profile_distinguishes_cold_warm_and_disabled(strategy_kind):
    runner, row = _multicoin_exposure_fixture(strategy_kind, "long")
    second_runner, _row = _multicoin_exposure_fixture(strategy_kind, "long")
    matrix = np.asarray([row], dtype=np.float64)
    loader, _args = runner._library_cache_call()
    loader.cache_clear()

    runner.run(matrix, profile=True)
    cold = dict(runner.last_profile)
    second_runner.run(matrix, profile=True)
    warm = dict(second_runner.last_profile)
    second_runner.run(matrix)

    assert cold["cold"] is True
    assert warm["cold"] is False
    assert cold["batch_size"] == 1
    assert cold["dispatch_count"] == 1
    for key in (
        "cpu_pack_seconds",
        "upload_and_zero_seconds",
        "compile_seconds",
        "pre_dispatch_sync_seconds",
        "kernel_seconds",
        "metric_decode_seconds",
    ):
        assert cold[key] >= 0.0
        assert warm[key] >= 0.0
    assert second_runner.last_profile == {}


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_tm_multicoin_entry_intervals_remain_coin_and_side_local(topology):
    count = 45
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    entry_steps_by_coin = ((5, 15, 35), (7, 27))
    close_steps_by_coin = ((6, 16, 36), (8, 28))
    for coin in range(2):
        entries = list(entry_steps_by_coin[coin])
        closes_for_coin = list(close_steps_by_coin[coin])
        if topology in {"long", "fused"}:
            lows[entries, coin] = 98.0
            highs[closes_for_coin, coin] = 102.0
        else:
            highs[entries, coin] = 102.0
            lows[closes_for_coin, coin] = 98.0

    runner, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "short" if topology == "short" else "long",
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        entry_interval_enabled=True,
        return_context=True,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_initial_qty_pct": 0.1,
            "entry_retracement_base_pct": 0.0,
            "close_threshold_base_pct": 0.01,
            "close_retracement_base_pct": 0.0,
            "gate_initial": 1.0,
            "n_positions": 2.0,
        }
    )
    row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    candidates = np.asarray([row], dtype=np.float64)
    if topology == "fused":
        runner = MpsTrailingMartingaleMulticoinFusedRunner(
            run,
            data,
            pnl_lookback_bars=1440,
            entry_interval_enabled=True,
        )
        candidates = np.asarray([row + row], dtype=np.float64)

    output = runner.run(candidates)
    synchronize()
    output = {key: value.cpu() for key, value in output.items()}
    metrics = _entry_interval_metrics(output, run, "trailing_martingale")

    expected_count = 6.0 if topology == "fused" else 3.0
    assert output["entry_interval_count"].item() == expected_count
    assert output["entry_interval_hist"].sum().item() == expected_count
    assert metrics["entry_interval_hours_mean"].item() == pytest.approx(
        50.0 / 3.0 / 60.0
    )
    assert metrics["entry_interval_hours_max"].item() == pytest.approx(1.0 / 3.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["single_side", "fused"])
def test_mps_multicoin_retains_synchronized_btc_risk_surface(strategy_kind, topology):
    _, row, run, data = _multicoin_exposure_fixture(
        strategy_kind, "long", count=5, return_context=True
    )
    btc_prices = np.array([100.0, 100.0, 50.0, 200.0, 100.0])
    if strategy_kind == "ema_anchor":
        row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("base_qty_pct")] = 0.0
        runner_cls = (
            MpsEmaAnchorMulticoinFusedRunner
            if topology == "fused"
            else MpsEmaAnchorMulticoinRunner
        )
    else:
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_qty_pct")] = (
            0.0
        )
        runner_cls = (
            MpsTrailingMartingaleMulticoinFusedRunner
            if topology == "fused"
            else MpsTrailingMartingaleMulticoinRunner
        )
    if topology == "fused":
        runner = runner_cls(run, data, pnl_lookback_bars=1440, btc_prices=btc_prices)
        candidates = np.asarray([row + row], dtype=np.float64)
    else:
        runner = runner_cls(
            run, data, pnl_lookback_bars=1440, side="long", btc_prices=btc_prices
        )
        candidates = np.asarray([row], dtype=np.float64)

    output = runner.run(candidates)
    synchronize()

    assert output["btc_day_end_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_min_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_max_dd"][0, 0].item() == pytest.approx(0.75)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_forced_normal_reserves_active_slots(strategy_kind, topology):
    if strategy_kind == "ema_anchor":
        override_cols = EMA_ANCHOR_COIN_OVERRIDE_COLS
        forced_column = EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        fused_runner_cls = MpsEmaAnchorMulticoinFusedRunner
    else:
        override_cols = TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
        forced_column = TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        fused_runner_cls = MpsTrailingMartingaleMulticoinFusedRunner
    forced = np.full((2, override_cols), np.nan, dtype=np.float32)
    forced[:, forced_column] = 1.0
    fixture_side = topology if topology != "fused" else "long"
    baseline_runner, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        fixture_side,
        collect_coin_fill_counts=True,
        return_context=True,
    )
    row[keys.index("n_positions")] = 1.0
    # Forced-normal expands only active selection; exact runtime budgeting
    # deliberately retains the configured one-slot WEL denominator. Disable
    # the separate TWEL entry gate so this regression observes selection
    # without asking the second forced coin to share that budget.
    row[keys.index("twel_entry_gate_enabled")] = 0.0

    if topology == "fused":
        baseline_runner = fused_runner_cls(
            run,
            data,
            pnl_lookback_bars=1440,
            collect_coin_fill_counts=True,
            hedge_mode=True,
        )
        forced_runner = fused_runner_cls(
            run,
            data,
            pnl_lookback_bars=1440,
            long_coin_overrides=forced,
            short_coin_overrides=forced,
            collect_coin_fill_counts=True,
            hedge_mode=True,
        )
        parameters = np.asarray([row + row], dtype=np.float64)
    else:
        forced_runner, _ = _multicoin_exposure_fixture(
            strategy_kind,
            topology,
            coin_overrides=forced,
            collect_coin_fill_counts=True,
        )
        parameters = np.asarray([row], dtype=np.float64)

    baseline = baseline_runner.run(parameters)
    expanded = forced_runner.run(parameters)
    synchronize()

    assert (baseline["coin_fill_counts"][0] > 0.0).sum().item() == 1
    assert (expanded["coin_fill_counts"][0] > 0.0).sum().item() == 2


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_forced_normal_service_matches_exact_active_symbols(
    strategy_kind, topology
):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import SharedAccountReplay

    count = 10
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.empty((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 120.0
    hlcvs[:, :, 1] = 80.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, 0, 3] = 1.0
    hlcvs[:, 1, 3] = 2.0
    market_settings = {
        coin: {
            "qty_step": 0.001,
            "price_step": 0.01,
            "min_qty": 0.001,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "maker": 0.0,
            "taker": 0.0,
            "exchange": "bybit",
            "first_valid_index": 0,
            "last_valid_index": count - 1,
            "warmup_minutes": 1,
        }
        for coin in ("BTC", "ETH")
    }
    market_settings["__meta__"] = {
        "requested_start_ts": int(timestamps[0]),
        "requested_start_date": "2023-11-14",
        "warmup_minutes_requested": 1,
    }
    config = get_template_config()
    config['bot']['global']['strategy_kind'] = strategy_kind
    config['bot']['global']['warmup']['max_minutes'] = 1
    enabled_sides = ("long", "short") if topology == "fused" else (topology,)
    set_side_universe_sources(config, 'approved_coins', {
        side: ["BTC", "ETH"] if side in enabled_sides else []
        for side in ("long", "short")
    })
    config['bot']['global']['hedge_mode'] = topology == "fused"
    config["backtest"]["coins"] = {"bybit": ["BTC", "ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    config["backtest"]["dynamic_wel_by_tradability"] = True
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.2 if enabled else 0.0
        risk["n_positions"] = 1 if enabled else 0
        config["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 100.0
        risk["we_excess_allowance_pct"] = 0.0
        risk["total_exposure_entry_gate_enabled"] = False
        risk['position_exposure_enforcer']['enabled'] = False
        risk['total_exposure_enforcer']['enabled'] = False
        config["bot"][side]["hsl"]["enabled"] = False
        config["bot"][side]["unstuck"]["enabled"] = False
    for side in enabled_sides:
        if strategy_kind == "ema_anchor":
            config["bot"][side]["strategy"]["ema_anchor"].update(
                {
                    "base_qty_pct": 1.0,
                    "ema_span_0": 2.0,
                    "ema_span_1": 3.0,
                    "entry_double_down_factor": 0.0,
                    "offset": 0.1,
                    "offset_psize_weight": 0.0,
                }
            )
        else:
            strategy = config["bot"][side]["strategy"]["trailing_martingale"]
            strategy["entry"]["ema_span_0"] = 2.0
            strategy["entry"]["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 1.0
            strategy["entry"]["initial_ema_dist"] = 0.1
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.5
    config["coin_overrides"] = {
        coin: {"live": {f"forced_mode_{side}": "normal" for side in enabled_sides}}
        for coin in ("BTC", "ETH")
    }

    proxy = SharedAccountReplay(
        config=config,
        hlcvs=hlcvs,
        mss=market_settings,
        btc=np.full(count, 50_000.0),
        timestamps=timestamps,
        exchange="bybit",
        batch_size=1,
        needed_metrics={"fills_active_symbols_count"},
    )
    for side in enabled_sides:
        assert proxy.base_params[side]["entry_cooldown_minutes"] == 100.0
    result = proxy.evaluate([{}])[0]
    _, _, exact_analysis = run_backtest(
        hlcvs,
        market_settings,
        config,
        "bybit",
        np.full(count, 50_000.0),
        timestamps,
    )

    assert result["fills_active_symbols_count"] == 2.0
    assert (
        result["fills_active_symbols_count"]
        == exact_analysis["fills_active_symbols_count"]
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("interval_minutes", [5, 7])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_aggregated_interval_directional_smoke(
    interval_minutes, side
):
    runner, row = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=420,
        interval_minutes=interval_minutes,
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("interval_minutes", [5, 7])
def test_mps_ema_multicoin_aggregated_interval_fused_smoke(interval_minutes):
    _, row, run, data = _multicoin_exposure_fixture(
        "ema_anchor",
        "long",
        count=420,
        interval_minutes=interval_minutes,
        return_context=True,
    )
    runner = MpsEmaAnchorMulticoinFusedRunner(run, data, pnl_lookback_bars=0)

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("interval_minutes", [5, 7])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_aggregated_interval_directional_smoke(interval_minutes, side):
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=420,
        interval_minutes=interval_minutes,
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("interval_minutes", [5, 7])
def test_mps_tm_multicoin_aggregated_interval_fused_smoke(interval_minutes):
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=420,
        interval_minutes=interval_minutes,
        return_context=True,
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(run, data, pnl_lookback_bars=0)

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
@pytest.mark.parametrize("forced_delist", [False, True])
def test_mps_multicoin_staggered_tail_keeps_balance_only_equity_and_hsl(
    strategy_kind, topology, forced_delist
):
    last_valid = 6
    count = last_valid + 1401 if forced_delist else 12
    closes = np.full((count, 2), 100.0)
    # Keep a positive packed mark after the explicit validity window so this
    # exercises the index boundary rather than relying on NaNs becoming zero.
    closes[last_valid + 1 :, 0] = 123.0
    highs = closes.copy()
    lows = closes.copy()
    highs[3, :] = 101.0
    lows[3, :] = 99.0
    if strategy_kind == "ema_anchor":
        override_cols = EMA_ANCHOR_COIN_OVERRIDE_COLS
        wallet_exposure_column = 11
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        fused_runner_cls = MpsEmaAnchorMulticoinFusedRunner
    else:
        override_cols = TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
        wallet_exposure_column = 24
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        fused_runner_cls = MpsTrailingMartingaleMulticoinFusedRunner
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[1, wallet_exposure_column] = 0.0
    markets = [ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0, 0.02) for _ in range(2)]
    fixture_side = topology if topology != "fused" else "long"
    runner, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        fixture_side,
        coin_overrides=overrides,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        last_valid_indices=(last_valid, count - 1),
        return_context=True,
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.9,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 0.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    if topology == "fused":
        runner = fused_runner_cls(
            run,
            data,
            pnl_lookback_bars=1440,
            long_coin_overrides=overrides,
            short_coin_overrides=overrides,
        )
        short_row = list(row)
        qty_key = (
            "base_qty_pct" if strategy_kind == "ema_anchor" else "entry_initial_qty_pct"
        )
        short_row[keys.index(qty_key)] *= 0.5
        params = np.asarray([row + short_row], dtype=np.float64)
    else:
        params = np.asarray([row], dtype=np.float64)
    if not forced_delist:
        with pytest.raises(ValueError, match="held-position valuation"):
            runner.run(params)
        return
    output = runner.run(params)
    synchronize()

    assert output["balance"].item() > 0.0
    if topology in {"long", "fused"}:
        assert (output["psize"].item() == 0.0) is forced_delist
    if topology in {"short", "fused"}:
        assert (output["short_psize"].item() == 0.0) is forced_delist
    expected_min_fills = 4.0 if topology == "fused" else 2.0
    if forced_delist:
        assert output["fill_count"].item() >= expected_min_fills
        assert output["hsl_panic_close_loss_sum"].item() > 0.0
    else:
        assert output["hsl_panic_close_loss_sum"].item() == 0.0
    assert output["day_end_eq"][0, 0].item() == pytest.approx(output["balance"].item())
    assert output["last_eq_ts"].item() > last_valid * run.interval_ms
    assert output["hsl_tier_samples_total"].item() > last_valid


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
@pytest.mark.parametrize(
    (
        "first_valid_indices",
        "last_valid_indices",
        "invalid_start",
        "invalid_end",
        "requested_start_index",
        "expected_first_eq_index",
    ),
    [
        ((0, 0), (6, 7), 8, 10, 0, 2),
        ((0, 9), (6, 11), 7, 8, 0, 2),
        ((0, 9), (6, 11), 7, 8, 7, 9),
    ],
    ids=[
        "all-coins-ended-tail",
        "gap-after-tracking-start",
        "gap-crosses-requested-start",
    ],
)
def test_mps_multicoin_all_invalid_time_keeps_equity_and_hsl_clock(
    strategy_kind,
    topology,
    first_valid_indices,
    last_valid_indices,
    invalid_start,
    invalid_end,
    requested_start_index,
    expected_first_eq_index,
):
    count = 12
    assert all(
        not any(
            first <= k <= last
            for first, last in zip(first_valid_indices, last_valid_indices)
        )
        for k in range(invalid_start, invalid_end + 1)
    )
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    # Keep positions flat: this test owns the equity/HSL clock across unavailable
    # listing windows. Held invalid tails are rejected in the boundary tests.
    fixture_side = topology if topology != "fused" else "long"
    runner, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        fixture_side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=first_valid_indices,
        last_valid_indices=last_valid_indices,
        requested_start_index=requested_start_index,
        return_context=True,
    )
    if strategy_kind == "ema_anchor":
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        fused_runner_cls = MpsEmaAnchorMulticoinFusedRunner
    else:
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        fused_runner_cls = MpsTrailingMartingaleMulticoinFusedRunner
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.9,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 0.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    if topology == "fused":
        runner = fused_runner_cls(run, data, pnl_lookback_bars=1440)
        params = np.asarray([row + row], dtype=np.float64)
    else:
        params = np.asarray([row], dtype=np.float64)
    output = runner.run(params)
    synchronize()

    assert output["balance"].item() > 0.0
    assert output["fill_count"].item() == 0.0
    assert output["first_eq_ts"].item() == pytest.approx(
        expected_first_eq_index * run.interval_ms
    )
    assert output["last_eq_ts"].item() >= invalid_end * run.interval_ms
    # Shared replay reports elapsed steps, with no initial sample duration.
    expected_hsl_samples = (
        output["last_eq_ts"].item() - output["first_eq_ts"].item()
    ) / run.interval_ms
    assert output["hsl_tier_samples_total"].item() == pytest.approx(
        expected_hsl_samples
    )
    day_samples = output["day_end_eq"][0]
    assert torch.isfinite(day_samples).any().item()
    assert day_samples[torch.isfinite(day_samples)][-1].item() == pytest.approx(
        output["balance"].item()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("adverse_side", ["long", "short"])
def test_mps_multicoin_fused_hsl_restarts_during_all_coins_ended_tail(
    strategy_kind, adverse_side
):
    count = 15
    last_valid = 10
    adverse_close = 70.0 if adverse_side == "long" else 130.0
    closes = np.full((count, 2), 100.0)
    closes[8:, 0] = adverse_close
    highs = closes * 1.02
    lows = closes * 0.98
    if strategy_kind == "ema_anchor":
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        override_cols = EMA_ANCHOR_COIN_OVERRIDE_COLS
        wallet_exposure_column = 11
        fused_runner_cls = MpsEmaAnchorMulticoinFusedRunner
    else:
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        override_cols = TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
        wallet_exposure_column = 24
        fused_runner_cls = MpsTrailingMartingaleMulticoinFusedRunner
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[1, wallet_exposure_column] = 0.0
    _, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        last_valid_indices=(last_valid, last_valid),
        return_context=True,
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.01,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 120000.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 1.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    # Exercise restart with the entire tail flat: the opposite side must not
    # retain an unrelated profitable trailing position past its valid range.
    long_row, short_row = list(row), list(row)
    inactive_row = short_row if adverse_side == "long" else long_row
    inactive_row[keys.index("total_wallet_exposure_limit")] = 0.0
    output = fused_runner_cls(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
    ).run(np.asarray([long_row + short_row], dtype=np.float64))
    synchronize()

    assert output[f"hsl_triggers_{adverse_side}"].item() == 1.0
    assert output[f"hsl_restarts_{adverse_side}"].item() == 1.0
    assert output["last_eq_ts"].item() > last_valid * run.interval_ms


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize(
    "strategy_kind, change",
    [
        ("ema_anchor", "candle"),
        ("ema_anchor", "ranking"),
        ("trailing_martingale", "candle"),
        ("trailing_martingale", "ranking"),
    ],
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_reselects_flat_candidate(strategy_kind, change, side):
    import passivbot_rust

    closes = np.full((4, 2), 100.0)
    highs = np.full((4, 2), 101.0)
    lows = np.full((4, 2), 99.0)
    runner, row, _run, data = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=4,
        closes=closes,
        highs=highs,
        lows=lows,
        return_context=True,
    )
    # Inject only into the isolated selection probe; public builders reject internal gaps.
    if change == "candle":
        data["bars"][2, 0, :3] = 0.0
    if strategy_kind == "ema_anchor":
        state_type = "EmaMulticoinSideState"
        config_type = "EmaMulticoinSideConfig"
        load_config = "load_ema_multicoin_side_config"
        init_state = "init_ema_multicoin_side_state"
        update_selection = "update_ema_multicoin_side_selection"
        source = passivbot_rust.mps_ema_anchor_multicoin_source_py()
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    else:
        state_type = "TrailingMartingaleMulticoinSideState"
        config_type = "TrailingMartingaleMulticoinSideConfig"
        load_config = "load_trailing_martingale_multicoin_side_config"
        init_state = "init_trailing_martingale_multicoin_side_state"
        update_selection = "update_tm_multicoin_side_selection"
        source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    row[keys.index("n_positions")] = 1.0
    probe_kernel = f"""
kernel void passivbot_multicoin_candidate_recovery_probe(
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {{
    if (b > 0) return;
    const int C = 2;
    const bool short_side = short_side_raw != 0;
    {config_type} config = {load_config}(params, 0);
    {state_type} candidate_side;
    {init_state}(
        candidate_side, config, coin_settings, coin_overrides, C
    );
    candidate_side.forager_volume[0] = 200.0f;
    candidate_side.forager_volume[1] = 100.0f;
    {update_selection}(
        candidate_side, config, bars, coin_settings, coin_overrides,
        1, C, short_side, false, 1, 0.0f, 0ul, false, 0.0f
    );
    output[0] = candidate_side.selected[0] ? 1.0f : 0.0f;
    output[1] = candidate_side.selected[1] ? 1.0f : 0.0f;
    {"candidate_side.forager_volume[1] = 300.0f;" if change == "ranking" else ""}
    {update_selection}(
        candidate_side, config, bars, coin_settings, coin_overrides,
        2, C, short_side, false, 1, 0.0f, 0ul, false, 0.0f
    );
    output[2] = candidate_side.selected[0] ? 1.0f : 0.0f;
    output[3] = candidate_side.selected[1] ? 1.0f : 0.0f;
    {"candidate_side.forager_volume[1] = 100.0f;" if change == "ranking" else ""}
    {update_selection}(
        candidate_side, config, bars, coin_settings, coin_overrides,
        3, C, short_side, false, 1, 0.0f, 0ul, false, 0.0f
    );
    output[4] = candidate_side.selected[0] ? 1.0f : 0.0f;
    output[5] = candidate_side.selected[1] ? 1.0f : 0.0f;
}}
"""
    params = _tm_probe_tensor(row, dtype=torch.float32, device=gpu_device())
    output = torch.zeros(6, dtype=torch.float32, device=gpu_device())
    library = compile_shader(source + probe_kernel)
    library.passivbot_multicoin_candidate_recovery_probe(
        data["bars"],
        data["coin_settings"],
        runner.coin_overrides,
        params,
        output,
        int(side == "short"),
        threads=(1, 1, 1),
    )
    synchronize()

    assert output.cpu().tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("prior_entry", [False, True])
def test_mps_multicoin_hysteresis_requires_prior_entry_order(strategy_kind, side, prior_entry):
    import passivbot_rust

    runner, row, _, data = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=4,
        return_context=True,
    )
    keys = (EMA_ANCHOR_MULTICOIN_PARAM_KEYS if strategy_kind == "ema_anchor"
            else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
    row[keys.index("n_positions")] = 1.0
    probe = r"""
kernel void passivbot_tm_incumbent_probe(
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    constant int& prior_entry,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    config.w_volume = 0.01f;
    config.w_ready = 0.0f;
    config.w_volatility = 0.99f;
    TrailingMartingaleMulticoinSideState state;
    init_trailing_martingale_multicoin_side_state(
        state, config, coin_settings, coin_overrides, 2
    );
    state.forager_volume[0] = 100.0f;
    state.forager_volume[1] = 200.0f;
    // A held coin stays selected regardless of its score.
    state.psize[0] = 1.0f;
    update_tm_multicoin_side_selection(
        state, config, bars, coin_settings, coin_overrides,
        1, 2, short_side_raw != 0, false, 1, 0.02f, 0ul, false, 0.0f
    );
    output[0] = state.selected[0];
    output[1] = state.selected[1];
    // Closing the position grants no incumbent priority on its own. An
    // outstanding entry does grant priority within the score hysteresis.
    state.psize[0] = 0.0f;
    state.entry_qty[0] = prior_entry ? 1.0f : 0.0f;
    update_tm_multicoin_side_selection(
        state, config, bars, coin_settings, coin_overrides,
        2, 2, short_side_raw != 0, true, 1, 0.02f, 0ul, false, 0.0f
    );
    output[2] = state.selected[0];
    output[3] = state.selected[1];
}
"""
    if strategy_kind == "ema_anchor":
        probe = (probe.replace("load_trailing_martingale_", "load_ema_")
                 .replace("init_trailing_martingale_", "init_ema_")
                 .replace("TrailingMartingale", "Ema")
                 .replace("trailing_martingale", "ema_anchor")
                 .replace("update_tm_", "update_ema_"))
    output = torch.zeros(4, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        getattr(passivbot_rust, f"mps_{strategy_kind}_multicoin_source_py")() + probe
    )
    library.passivbot_tm_incumbent_probe(
        data["bars"],
        data["coin_settings"],
        runner.coin_overrides,
        torch.tensor(row, dtype=torch.float32, device=gpu_device()),
        output,
        int(side == "short"),
        int(prior_entry),
        threads=(1, 1, 1),
    )
    synchronize()
    assert output.cpu().tolist() == [
        1.0,
        0.0,
        float(prior_entry),
        float(not prior_entry),
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("invalid_mode", ["tail", "internal"])
def test_mps_tm_recursive_entry_twel_gate_does_not_expand_invalid_coin(
    side, invalid_mode
):
    import passivbot_rust

    closes = np.full((3, 2), 100.0)
    highs = closes * 1.01
    lows = closes * 0.99
    last_valid_indices = (0, 2)
    if invalid_mode == "internal":
        last_valid_indices = (2, 2)
    runner, row, _run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=3,
        closes=closes,
        highs=highs,
        lows=lows,
        last_valid_indices=last_valid_indices,
        market_orders_allowed=True,
        return_context=True,
    )
    if invalid_mode == "internal":
        # The isolated guard probe owns this malformed state, not the public builder.
        data["bars"][1, 0, 1] = 0.0
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_double_down_factor": 1.0,
            "entry_initial_qty_pct": 0.1,
            "entry_threshold_base_pct": 0.01,
            "total_wallet_exposure_limit": 1.0,
            "n_positions": 1.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
        }
    )
    params = _tm_probe_tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(2, dtype=torch.float32, device=gpu_device())
    probe_kernel = r"""
kernel void passivbot_tm_multicoin_tail_recursive_gate_probe(
    constant float* bars,
    constant int* fill_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    const int C = 2;
    const int k = 1;
    const bool short_side = short_side_raw != 0;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideState side;
    init_trailing_martingale_multicoin_side_state(
        side, config, coin_settings, coin_overrides, C
    );
    const int tick_offset = (k * C) * 2;
    const float price_step = coin_settings[1];
    side.entry_deferred_twel_gate = true;
    side.psize[0] = 1.0f;
    side.pprice[0] = 100.0f;
    side.entry_tick[0] = short_side
        ? fill_ticks[tick_offset]
        : fill_ticks[tick_offset + 1] + 1;
    side.entry_order_type[0] = short_side ? 15 : 4;
    side.entry_strategy_qty[0] = 1.0f;
    side.entry_qty[0] = 1.0f;
    side.entry_gen_balance[0] = 1000.0f;
    side.entry_gen_allowed_wel[0] = 1.0f;
    side.entry_gen_market_price[0] = 100.0f;
    side.entry_gen_psize[0] = 1.0f;
    side.entry_gen_pprice[0] = 100.0f;
    side.entry_gen_initial_price[0] = 100.0f;
    side.entry_gen_touch_tick[0] = short_side ? 0 : 100000000;
    side.entry_recursive_mode[0] = true;
    apply_tm_multicoin_recursive_entry_twel_gate(
        side, config, bars, fill_ticks, coin_settings, coin_overrides,
        k, C, short_side, true, 0.001f
    );
    output[0] = side.entry_qty[0];
    output[1] = float(side.entry_gate_suffix_keep_count[0]);
}
"""
    source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    library = compile_shader(source + probe_kernel)
    library.passivbot_tm_multicoin_tail_recursive_gate_probe(
        data["bars"],
        data["touch_ticks"],
        data["coin_settings"],
        runner.coin_overrides,
        params,
        output,
        int(side == "short"),
        threads=(1, 1, 1),
    )
    synchronize()

    # Exact Rust retains the already generated next entry but does not expand
    # its recursive market suffix once NextCandle.tradable becomes false.
    assert output[0].item() > 0.0
    assert output[1].item() == 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mark_case", ["tail", "in_range_zero"])
def test_mps_multicoin_risk_reducers_exclude_tail_but_keep_its_exposure(
    strategy_kind, side, mark_case
):
    import passivbot_rust

    if strategy_kind == "ema_anchor":
        probe_kernel = r"""
kernel void passivbot_ema_multicoin_tail_twel_probe(
    constant float* bars,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool short_side = short_side_raw != 0;
    EmaMulticoinSideConfig config =
        load_ema_multicoin_side_config(params, 0);
    EmaMulticoinSideState side;
    init_ema_multicoin_side_state(
        side, config, coin_settings, coin_overrides, 2
    );
    side.selected[0] = true;
    side.selected[1] = true;
    side.psize[0] = 8.0f;
    side.psize[1] = 3.0f;
    side.pprice[0] = 100.0f;
    side.pprice[1] = 100.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    float selected_diff = INFINITY;
    int selected_unstuck = select_ema_multicoin_unstuck_coin(
        side, config, account, bars, touch_ticks, coin_settings,
        coin_overrides, 1, 2, short_side, 1, selected_diff
    );
    generate_ema_multicoin_side_orders(
        side, config, account,
        bars, touch_ticks, coin_settings, coin_overrides,
        1, 2, short_side, 1, 1, 0,
        false, 0.001f, -2, 0ul
    );
    output[0] = side.twel_close_qty[0];
    output[1] = side.twel_close_qty[1];
    output[2] = side.unstuck_close_qty[0];
    output[3] = side.unstuck_close_qty[1];
    output[4] = float(selected_unstuck);
    output[5] = accumulate_ema_multicoin_side_unrealized_pnl(
        side, bars, coin_settings, 1, 2, short_side, 0.0f
    );
    side.psize[1] = 0.0f;
    side.selected[0] = false;
    side.selected[1] = false;
    update_ema_multicoin_side_selection(
        side, config, bars, coin_settings, coin_overrides,
        1, 2, short_side, true, 2, 0.0f, 0ul, false, 0.0f
    );
    output[6] = side.selected[0] ? 1.0f : 0.0f;
    output[7] = side.selected[1] ? 1.0f : 0.0f;
}
"""
        source = passivbot_rust.mps_ema_anchor_multicoin_source_py()
        kernel_name = "passivbot_ema_multicoin_tail_twel_probe"
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    else:
        probe_kernel = r"""
kernel void passivbot_tm_multicoin_tail_twel_probe(
    constant float* bars,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool short_side = short_side_raw != 0;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideState side;
    init_trailing_martingale_multicoin_side_state(
        side, config, coin_settings, coin_overrides, 2
    );
    side.selected[0] = true;
    side.selected[1] = true;
    side.psize[0] = 8.0f;
    side.psize[1] = 3.0f;
    side.pprice[0] = 100.0f;
    side.pprice[1] = 100.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    float selected_diff = INFINITY;
    int selected_unstuck = select_tm_multicoin_unstuck_coin(
        side, config, account, bars, touch_ticks, coin_settings,
        coin_overrides, 1, 2, short_side, 1, selected_diff
    );
    generate_tm_multicoin_side_orders(
        side, config, account,
        bars, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides,
        1, 2, short_side, 1, 1, 0, false, 1.0f,
        false, 0.001f, 0.0f, -2, 0ul
    );
    output[0] = side.twel_close_qty[0];
    output[1] = side.twel_close_qty[1];
    output[2] = side.unstuck_close_qty[0];
    output[3] = side.unstuck_close_qty[1];
    output[4] = float(selected_unstuck);
    output[5] = accumulate_tm_multicoin_side_unrealized_pnl(
        side, bars, coin_settings, 1, 2, short_side, 0.0f
    );
    side.psize[1] = 0.0f;
    side.selected[0] = false;
    side.selected[1] = false;
    update_tm_multicoin_side_selection(
        side, config, bars, coin_settings, coin_overrides,
        1, 2, short_side, true, 2, 0.0f, 0ul, false, 0.0f
    );
    output[6] = side.selected[0] ? 1.0f : 0.0f;
    output[7] = side.selected[1] ? 1.0f : 0.0f;
}
"""
        source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        kernel_name = "passivbot_tm_multicoin_tail_twel_probe"
        keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS

    closes = np.full((3, 2), 100.0)
    last_valid_indices = (0, 2)
    if mark_case == "in_range_zero":
        last_valid_indices = (2, 2)
    runner, row, _run, data = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=3,
        closes=closes,
        last_valid_indices=last_valid_indices,
        return_context=True,
    )
    if mark_case == "in_range_zero":
        # Keep the narrow reducer test independent of input preflight.
        data["bars"][1, 0, :3] = 0.0
    values = dict(zip(keys, row))
    values.update(
        {
            "n_positions": 2.0,
            "total_wallet_exposure_limit": 1.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
            "unstuck_enabled": 1.0,
            "unstuck_ema_gating_enabled": 0.0,
            "unstuck_close_pct": 0.1,
            "unstuck_loss_allowance_pct": 1.0,
            "unstuck_threshold": 0.1,
        }
    )
    params = _tm_probe_tensor(
        [values[key] for key in keys], dtype=torch.float32, device=gpu_device()
    )
    output = torch.zeros(8, dtype=torch.float32, device=gpu_device())
    library = compile_shader(source + probe_kernel)
    kernel = getattr(library, kernel_name)
    args = [data["bars"], data["touch_ticks"]]
    if strategy_kind == "trailing_martingale":
        args.extend(
            [
                data["touch_nearest_ticks"],
                data["touch_min_qty_bits"],
                data["touch_min_qty_relation"],
            ]
        )
    args.extend(
        [
            data["coin_settings"],
            runner.coin_overrides,
            params,
            output,
            int(side == "short"),
        ]
    )
    kernel(*args, threads=(1, 1, 1))
    synchronize()

    if mark_case == "tail":
        # Coin zero contributes 0.8 TWE, but it is beyond last_valid at k=1
        # and cannot be managed or marked. Coin one remains the only eligible
        # risk-reducer candidate.
        assert output[0].item() == 0.0
        assert output[1].item() > 0.0
        assert output[2].item() == 0.0
        assert output[3].item() > 0.0
        assert output[4].item() == 1.0
        assert output[5].item() == 0.0
    else:
        # A finite zero mark is not entry-eligible, but exact Rust still marks
        # and manages an existing in-range position at that price.
        assert output[0].item() > 0.0
        expected_upnl = 800.0 if side == "short" else -800.0
        assert output[5].item() == pytest.approx(expected_upnl)

    # Current entry capacity is one coin even though the grow-only WEL
    # denominator remains two. The held ineligible coin consumes that slot.
    assert output[6].item() == 1.0
    assert output[7].item() == 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_min_cost_admits_funded_concurrent_flat_coins(
    strategy_kind, side
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0) for min_cost in (30.0, 5.0)
    ]
    unfiltered, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=6,
        markets=markets,
        collect_coin_fill_counts=True,
    )
    filtered, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=6,
        markets=markets,
        collect_coin_fill_counts=True,
        filter_by_min_effective_cost=True,
    )

    parameters = np.asarray([row], dtype=np.float64)
    unfiltered_output = unfiltered.run(parameters)
    filtered_output = filtered.run(parameters)
    synchronize()

    assert unfiltered_output["coin_fill_counts"].cpu().tolist()[0][0] > 0.0
    assert unfiltered_output["coin_fill_counts"].cpu().tolist()[0][1] > 0.0
    assert filtered.settings.cpu()[-1].item() == 1.0
    filtered_counts = filtered_output["coin_fill_counts"].cpu().tolist()[0]
    assert filtered_counts == unfiltered_output["coin_fill_counts"].cpu().tolist()[0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_min_cost_rejected_coin_does_not_block_later_funded_coin(
    strategy_kind, side
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0) for min_cost in (1500.0, 5.0)
    ]
    filtered, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=10,
        markets=markets,
        first_valid_indices=(0, 5),
        collect_coin_fill_counts=True,
        filter_by_min_effective_cost=True,
    )

    output = filtered.run(np.asarray([row], dtype=np.float64))
    synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    assert counts[0] == 0.0
    assert counts[1] > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_min_cost_keeps_funded_slots_eligible_after_selection(
    strategy_kind, side
):
    count = 10
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[6:, 1] = 108.0
    else:
        highs[6:, 1] = 132.0
    markets = [ProxyMarket(0.001, 0.01, 0.001, 1.0, 1.0, 0.0) for _ in range(2)]
    unfiltered, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(0, 5),
        collect_coin_fill_counts=True,
    )
    filtered, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(0, 5),
        collect_coin_fill_counts=True,
        filter_by_min_effective_cost=True,
    )

    parameters = np.asarray([row], dtype=np.float64)
    unfiltered_output = unfiltered.run(parameters)
    filtered_output = filtered.run(parameters)
    synchronize()

    assert unfiltered_output["coin_fill_counts"].cpu().tolist()[0][1] > 0.0
    assert filtered_output["coin_fill_counts"].cpu().tolist()[0] == (
        unfiltered_output["coin_fill_counts"].cpu().tolist()[0]
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_mps_fused_min_cost_admits_funded_coins_with_both_account_modes(
    strategy_kind, hedge_mode
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0) for min_cost in (60.0, 5.0)
    ]
    _, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        count=10,
        markets=markets,
        first_valid_indices=(0, 5),
        return_context=True,
    )
    runner_cls = (
        MpsEmaAnchorMulticoinFusedRunner
        if strategy_kind == "ema_anchor"
        else MpsTrailingMartingaleMulticoinFusedRunner
    )
    runner = runner_cls(
        run,
        data,
        pnl_lookback_bars=1440,
        hedge_mode=hedge_mode,
        filter_by_min_effective_cost=True,
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    synchronize()

    assert output["fill_count"].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("override_kind", ["initial_qty", "wel_allowance"])
def test_mps_multicoin_min_effective_cost_uses_per_coin_overrides(
    strategy_kind, override_kind
):
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    if override_kind == "initial_qty":
        qty_column = 0 if strategy_kind == "ema_anchor" else 6
        overrides[0, qty_column] = 0.1
        passing_overrides = overrides.copy()
        passing_overrides[0, qty_column] = 1.0
        min_cost = 60.0
    else:
        wel_column = (
            11
            if strategy_kind == "ema_anchor"
            else TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
        )
        allowance_column = (
            12
            if strategy_kind == "ema_anchor"
            else TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN
        )
        overrides[:, wel_column] = 0.1
        overrides[1, allowance_column] = 1.0
        passing_overrides = overrides.copy()
        passing_overrides[0, allowance_column] = 1.0
        min_cost = 150.0
    markets = [ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0) for _ in range(2)]
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        overrides,
        count=6,
        markets=markets,
        collect_coin_fill_counts=True,
        filter_by_min_effective_cost=True,
    )
    passing_runner, _ = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        passing_overrides,
        count=6,
        markets=markets,
        collect_coin_fill_counts=True,
        filter_by_min_effective_cost=True,
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    passing_output = passing_runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    passing_counts = passing_output["coin_fill_counts"].cpu().tolist()[0]
    assert counts[0] == 0.0
    assert counts[1] > 0.0
    assert passing_counts[0] > 0.0
    assert passing_counts[1] > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_min_cost_admits_funded_slot_while_portfolio_is_open():
    strategy_kind = "trailing_martingale"
    highs = np.tile(np.asarray([100.0, 120.0]), (10, 1))
    lows = highs * 0.99
    markets = [ProxyMarket(0.001, 0.01, 0.001, 1.0, 1.0, 0.0) for _ in range(2)]
    unfiltered, row = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        count=10,
        markets=markets,
        highs=highs,
        lows=lows,
        first_valid_indices=(0, 4),
        collect_coin_fill_counts=True,
    )
    filtered, _ = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        count=10,
        markets=markets,
        highs=highs,
        lows=lows,
        first_valid_indices=(0, 4),
        collect_coin_fill_counts=True,
        filter_by_min_effective_cost=True,
    )

    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_qty_pct")] = 0.5
    parameters = np.asarray([row], dtype=np.float64)
    unfiltered_output = unfiltered.run(parameters)
    filtered_output = filtered.run(parameters)
    synchronize()

    unfiltered_counts = unfiltered_output["coin_fill_counts"].cpu().tolist()[0]
    filtered_counts = filtered_output["coin_fill_counts"].cpu().tolist()[0]
    assert unfiltered_counts[0] > 0.0
    assert unfiltered_counts[1] > 0.0
    assert filtered_counts[0] > 0.0
    assert filtered_counts[1] > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_fused_one_way_min_cost_rejection_blocks_side_arbitration(strategy_kind):
    markets = [ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0) for _ in range(2)]
    _, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        count=6,
        markets=markets,
        return_context=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    qty_key = (
        "base_qty_pct" if strategy_kind == "ema_anchor" else "entry_initial_qty_pct"
    )
    long_row = list(row)
    short_row = list(row)
    long_row[keys.index(qty_key)] = 0.001
    short_row[keys.index(qty_key)] = 1.0
    runner_cls = (
        MpsEmaAnchorMulticoinFusedRunner
        if strategy_kind == "ema_anchor"
        else MpsTrailingMartingaleMulticoinFusedRunner
    )
    runner = runner_cls(
        run,
        data,
        pnl_lookback_bars=1440,
        hedge_mode=False,
        filter_by_min_effective_cost=True,
    )

    output = runner.run(np.asarray([long_row + short_row], dtype=np.float64))
    synchronize()

    assert output["fill_count_long"].item() == 0.0
    assert output["fill_count"].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("signal_mode", ["portfolio", "pside"])
def test_mps_one_sided_multicoin_hsl_panics_the_portfolio(
    strategy_kind, side, signal_mode
):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    shock = 0.7 if side == "long" else 1.3
    closes[20:] *= shock
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        collect_coin_fill_counts=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.05,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 0.0 if signal_mode == "portfolio" else 1.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    enabled_key = f"hsl_{side}_enabled"
    other_side = "short" if side == "long" else "long"
    assert output[enabled_key].item()
    assert not output[f"hsl_{other_side}_enabled"].item()
    assert output[f"hsl_triggers_{side}"].item() == 1.0
    assert output[f"hsl_triggers_{other_side}"].item() == 0.0
    assert output["hsl_trigger_drawdown_count"].item() == 1.0
    assert output["hsl_panic_loss_drawdown_count"].item() == 1.0
    assert output["hsl_panic_close_loss_sum"].item() > 0.0
    assert (output["coin_fill_counts"][0] >= 2.0).all().item()
    assert output["open_positions"].item() == 0.0




@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("shock_coin", [0, 1, "both"])
def test_mps_one_sided_multicoin_coin_hsl_isolates_each_coin_episode(
    strategy_kind, side, shock_coin
):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    shock = 0.7 if side == "long" else 1.3
    if shock_coin == "both":
        closes[20:] *= shock
        markets = [
            ProxyMarket(
                0.001,
                0.01,
                0.001,
                0.0,
                1.0,
                maker_fee=0.0,
                taker_fee=0.01,
            )
            for _ in range(2)
        ]
    else:
        closes[20:, shock_coin] *= shock
        markets = None
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_order_slippage_pct=0.02 if shock_coin == "both" else 0.0,
        hsl_panic_market=shock_coin == "both",
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.05,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 2.0,
        # The kernel must derive the effective value of two; this packed
        # single-coin default is deliberately not pre-adjusted by Python.
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    packed_slot_variant = list(row)
    packed_slot_variant[keys.index("n_positions")] = 64.0
    output = runner.run(np.asarray([row, packed_slot_variant], dtype=np.float64))
    synchronize()

    assert output[f"hsl_{side}_enabled"].all().item()
    expected_episode_count = 2.0 if shock_coin == "both" else 1.0
    assert (output[f"hsl_triggers_{side}"] == expected_episode_count).all().item()
    assert (output["hsl_trigger_drawdown_count"] == expected_episode_count).all().item()
    assert (
        (output["hsl_panic_loss_drawdown_count"] == expected_episode_count).all().item()
    )
    assert (output["hsl_panic_close_loss_sum"] > 0.0).all().item()
    assert (output[f"hsl_strategy_eq_recovery_max_ms_{side}"] > 0.0).all().item()
    if shock_coin == "both":
        assert (output["coin_fill_counts"] >= 2.0).all().item()
        assert (
            (
                output["hsl_panic_loss_drawdown_max"]
                > output["hsl_panic_loss_drawdown_min"]
            )
            .all()
            .item()
        )
        assert output["hsl_panic_loss_drawdown_sum"][0].item() == pytest.approx(
            output["hsl_panic_loss_drawdown_min"][0].item()
            + output["hsl_panic_loss_drawdown_max"][0].item()
        )
        expected_open_positions = 0.0
    else:
        assert (output["coin_fill_counts"][:, shock_coin] >= 2.0).all().item()
        # A stopped coin cannot suppress the healthy flat coin's next entry.
        # CPU replay and both GPU strategies leave that unaffected coin open.
        assert (output["coin_fill_counts"][:, 1 - shock_coin] >= 1.0).all().item()
        expected_open_positions = 1.0
    assert (output["open_positions"] == expected_open_positions).all().item()
    assert output["hsl_trigger_drawdown_sum"][0].item() == pytest.approx(
        output["hsl_trigger_drawdown_sum"][1].item()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    ("global_enabled", "coin_enabled", "expected_triggers"),
    [(False, True, 1.0), (True, False, 0.0)],
)
def test_mps_multicoin_coin_hsl_override_controls_enablement(
    strategy_kind, side, global_enabled, coin_enabled, expected_triggers
):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[20:, 1] *= 0.7 if side == "long" else 1.3
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    hsl_start = 19 if strategy_kind == "ema_anchor" else 34
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    if coin_enabled:
        overrides[1, hsl_start : hsl_start + 6] = [
            1.0,
            0.05,
            1.0,
            0.0,
            2.0,
            0.0,
        ]
    else:
        overrides[1, hsl_start] = 0.0
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        collect_coin_fill_counts=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "hsl_enabled": float(global_enabled),
        "hsl_red_threshold": 0.05,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 2.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output[f"hsl_triggers_{side}"].item() == expected_triggers
    if coin_enabled:
        assert output[f"hsl_{side}_enabled"].item()
        assert output["hsl_tier_samples_total"].item() > 0.0
        assert output["coin_fill_counts"][0, 1].item() >= 2.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_coin_hsl_recovery_uses_total_exposure_contract(
    strategy_kind, side
):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[20:, 1] *= 0.7 if side == "long" else 1.3
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    wel_index = 11 if strategy_kind == "ema_anchor" else 24
    hsl_start = 19 if strategy_kind == "ema_anchor" else 34
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    # A fixed per-coin WEL of zero disables entries, but does not change that
    # coin's positive side total_wallet_exposure_limit. Exact Rust's HSL
    # reporting contract checks the latter, so the enabled controller still
    # samples aggregate side equity while coin one trades.
    overrides[0, wel_index] = 0.0
    overrides[1, hsl_start] = 0.0
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        collect_coin_fill_counts=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.05,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 2.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["coin_fill_counts"][0, 1].item() >= 1.0
    assert output[f"hsl_strategy_eq_recovery_max_ms_{side}"].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_coin_hsl_overrides_can_disable_every_controller(
    strategy_kind, side
):
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    hsl_start = 19 if strategy_kind == "ema_anchor" else 34
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[:, hsl_start] = 0.0
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, coin_overrides=overrides, count=32
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    row[keys.index("hsl_enabled")] = 1.0
    row[keys.index("hsl_signal_mode")] = 2.0

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert not output[f"hsl_{side}_enabled"].item()
    assert output["hsl_tier_samples_total"].item() == 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_coin_hsl_override_selects_market_panic(strategy_kind, side):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[20:, 1] *= 0.7 if side == "long" else 1.3
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    hsl_start = 19 if strategy_kind == "ema_anchor" else 34
    outputs = []
    for market_panic in (0.0, 1.0):
        overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
        overrides[1, hsl_start : hsl_start + 6] = [
            1.0,
            0.05,
            1.0,
            0.0,
            2.0,
            market_panic,
        ]
        runner, row = _multicoin_exposure_fixture(
            strategy_kind,
            side,
            coin_overrides=overrides,
            count=count,
            closes=closes,
            markets=markets,
            market_order_slippage_pct=0.02,
            hsl_panic_market=False,
        )
        keys = (
            EMA_ANCHOR_MULTICOIN_PARAM_KEYS
            if strategy_kind == "ema_anchor"
            else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        )
        row[keys.index("hsl_signal_mode")] = 2.0
        outputs.append(runner.run(np.asarray([row], dtype=np.float64)))
        synchronize()

    limit_output, market_output = outputs
    assert limit_output[f"hsl_triggers_{side}"].item() == 1.0
    assert market_output[f"hsl_triggers_{side}"].item() == 1.0
    assert (
        market_output["hsl_panic_close_loss_sum"].item()
        > limit_output["hsl_panic_close_loss_sum"].item()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_coin_hsl_reselects_around_blocked_forager_coin(
    strategy_kind, side
):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[20:, 1] *= 0.7 if side == "long" else 1.3
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        collect_coin_fill_counts=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "n_positions": 1.0,
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.05,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 2.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output[f"hsl_triggers_{side}"].item() == 1.0
    assert output["coin_fill_counts"][0, 1].item() >= 2.0
    assert output["coin_fill_counts"][0, 0].item() >= 1.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_one_sided_multicoin_hsl_market_panic_applies_taker_costs(
    strategy_kind, side
):
    count = 64
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[20:] *= 0.7 if side == "long" else 1.3
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    limit_runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, count=count, closes=closes, markets=markets
    )
    market_runner, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        markets=markets,
        market_order_slippage_pct=0.02,
        hsl_panic_market=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.05,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 1.0,
        'hsl_cooldown_ms_hi': 0.0,
    }.items():
        row[keys.index(key)] = value
    params = np.asarray([row], dtype=np.float64)

    assert market_runner.settings[7].item() == pytest.approx(0.02)
    assert market_runner.settings[8].item() == 1.0
    assert torch.allclose(
        market_runner.coin_settings[:, 11],
        torch.full((2,), 0.01, device=gpu_device()),
    )
    limit_output = limit_runner.run(params)
    market_output = market_runner.run(params)
    synchronize()

    assert limit_output[f"hsl_triggers_{side}"].item() == 1.0
    assert market_output[f"hsl_triggers_{side}"].item() == 1.0
    assert limit_output["open_positions"].item() == 0.0
    assert market_output["open_positions"].item() == 0.0
    assert (
        market_output["hsl_panic_close_loss_sum"].item()
        > limit_output["hsl_panic_close_loss_sum"].item()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_tracks_position_unchanged_max(strategy_kind, side):
    runner, row = _multicoin_exposure_fixture(strategy_kind, side, count=16)

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert "coin_fill_counts" not in output
    unchanged_ms = output["position_unchanged_max_ms"].item()
    assert unchanged_ms > 0.0
    assert unchanged_ms <= output["held_max_ms"].item()
    assert output["entry_initial_balance_pct"].item() == pytest.approx(0.5)
    assert output["total_wallet_exposure_max"].item() > 0.0
    assert (
        output["total_wallet_exposure_max"].item()
        >= output["total_wallet_exposure_mean"].item()
    )
    fill_days = output["day_has_fill"].bool()
    assert torch.isfinite(output["day_net_pnl"][fill_days]).all()
    assert (output["day_last_fill_balance"][fill_days] > 0.0).all()
    assert (output["day_fill_count"][fill_days] >= 1.0).all()
    assert torch.equal(output["day_fill_count"], output["day_fill_count"].round())
    _assert_fill_scalar_contract(output)
    if side == "long":
        assert torch.equal(output["fill_count_long"], output["fill_count"])
    else:
        assert (output["fill_count_long"] == 0.0).all()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_fixed_wel_uses_configured_position_denominator(
    strategy_kind, side
):
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    wallet_exposure_column = 11 if strategy_kind == "ema_anchor" else 24
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[1, wallet_exposure_column] = 0.0
    fixed_runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        coin_overrides=overrides,
        dynamic_wel_by_tradability=False,
    )
    dynamic_runner, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        coin_overrides=overrides,
        dynamic_wel_by_tradability=True,
    )

    parameters = np.asarray([row], dtype=np.float64)
    fixed = fixed_runner.run(parameters)
    dynamic = dynamic_runner.run(parameters)
    synchronize()

    # Fixed WEL allocates TWEL 1.0 across both configured slots even though
    # the second prepared coin is explicitly ineligible for this side. Dynamic
    # mode allocates the same TWEL to its one observed eligible slot.
    assert fixed["entry_initial_balance_pct"].item() == pytest.approx(0.5)
    assert dynamic["entry_initial_balance_pct"].item() == pytest.approx(1.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hedge_mode", [True, False], ids=["hedge", "one-way"])
def test_mps_fused_multicoin_fixed_wel_uses_each_configured_denominator(
    strategy_kind, hedge_mode
):
    _, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        return_context=True,
    )
    runner_cls = (
        MpsEmaAnchorMulticoinFusedRunner
        if strategy_kind == "ema_anchor"
        else MpsTrailingMartingaleMulticoinFusedRunner
    )
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    wallet_exposure_column = 11 if strategy_kind == "ema_anchor" else 24
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[1, wallet_exposure_column] = 0.0
    runner = runner_cls(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        hedge_mode=hedge_mode,
        dynamic_wel_by_tradability=False,
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    synchronize()

    assert output["entry_initial_balance_pct_long"].item() == pytest.approx(0.5)
    assert output["entry_initial_balance_pct_short"].item() == pytest.approx(0.5)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_active_fill_days_use_equity_start_buckets(strategy_kind, side):
    runner, row = _multicoin_exposure_fixture(strategy_kind, side, count=3 * 1440 + 32)
    if strategy_kind == "trailing_martingale":
        row[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_threshold_base_pct")
        ] = 0.0
        row[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_base_pct")
        ] = 0.0

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    duration_days = (output["last_eq_ts"] - output["first_eq_ts"]).item() / 86_400_000.0
    assert output["fills_active_days_count"].item() == int(np.ceil(duration_days))
    assert output["fills_active_days_count"].item() >= 3.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_tracks_fill_counts_by_symbol(strategy_kind, side):
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, count=32, collect_coin_fill_counts=True
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    coin_counts = output["coin_fill_counts"]
    assert coin_counts.shape == (1, 2)
    assert torch.equal(coin_counts, coin_counts.round())
    assert (coin_counts > 0.0).all()
    assert coin_counts.sum().item() == output["fill_count"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_uses_first_coin_override(strategy_kind, side):
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    initial_qty_column = 0 if strategy_kind == "ema_anchor" else 6
    allowance_column = 12 if strategy_kind == "ema_anchor" else 25
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[0, initial_qty_column] = 0.25
    overrides[0, allowance_column] = 0.5
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, coin_overrides=overrides, count=16
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    # TWEL 1.0 / two effective positions, with a bounded 50% allowance.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(0.1875)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_freezes_denominator_at_liquidation(
    strategy_kind, side
):
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=20,
        first_valid_indices=(0, 12),
        liquidation_threshold=1.0,
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert not output["alive"].item()
    # The liquidation floor terminates before the second coin becomes
    # tradable, so exact Rust retains a one-position divisor.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(1.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_freezes_before_post_fill_balance_depletion(
    strategy_kind, side
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 2.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 2.0),
    ]
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=12,
        markets=markets,
        first_valid_indices=(0, 3),
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert not output["alive"].item()
    # Coin one fills and its fee depletes balance on the same candle that coin
    # two first becomes tradable. Exact Rust liquidates before growing the
    # effective-position denominator.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(1.0)


def test_directional_touch_ticks_preserve_alignment_and_round_non_aligned_prices():
    down, up, nearest = _directional_touch_ticks(
        np.array([100.0, 100.006, 0.1 + 0.2]), 0.01
    )

    assert down.tolist() == [10_000, 10_000, 30]
    assert up.tolist() == [10_000, 10_001, 30]
    assert nearest.tolist() == [10_000, 10_001, 30]


def test_minimum_entry_qty_encoding_uses_original_float64_touch_price():
    market = ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0)

    bits, relation = _minimum_entry_qty_encoding(np.array([100.004, 100.0]), market)
    rounded = np.ascontiguousarray(bits).view(np.float32)
    assert rounded.tolist() == pytest.approx([0.1, 0.101])
    assert relation.tolist() == [-1, -1]


def test_minimum_entry_qty_encoding_preserves_just_above_aligned_minimum():
    market = ProxyMarket(1.0, 0.001, 0.0, 11.0, 1.0, 0.0)

    bits, relation = _minimum_entry_qty_encoding(np.array([0.088]), market)
    rounded = np.ascontiguousarray(bits).view(np.float32)
    assert rounded.tolist() == [125.0]
    assert relation.tolist() == [1]


def test_maximum_effective_min_cost_preserves_float64_executable_threshold():
    market = ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0)

    threshold = _maximum_effective_min_cost(np.array([100.004, 100.0]), market)

    assert threshold == pytest.approx(0.101 * 100.0)


from optimization.gpu.mps_kernel import (
    MpsEmaAnchorMulticoinRunner,
    MpsTrailingMartingaleMulticoinRunner,
)


def test_strict_fill_tick_boundaries_preserve_float32_candle_crossing():
    step = 0.01
    order_tick = 371_177
    exact_order_price = order_tick * step
    represented_above = float(np.float32(exact_order_price))
    assert represented_above > exact_order_price

    high_fill_max, low_nonfill_max = _strict_fill_tick_boundaries(
        np.array([represented_above, exact_order_price]),
        np.array([exact_order_price, represented_above]),
        step,
    )

    assert high_fill_max.tolist() == [order_tick, order_tick - 1]
    assert low_nonfill_max.tolist() == [order_tick, order_tick]


def test_strict_fill_tick_boundaries_preserve_rust_step_decimal_rounding():
    step = 0.1
    order_tick = 371_177
    rust_order_price = 37_117.7
    assert order_tick * step > rust_order_price

    high_fill_max, low_nonfill_max = _strict_fill_tick_boundaries(
        np.array([rust_order_price]),
        np.array([rust_order_price]),
        step,
    )

    assert high_fill_max.tolist() == [order_tick - 1]
    assert low_nonfill_max.tolist() == [order_tick]


def test_initial_single_candle_hour_bucket_matches_rust_skip_contract():
    timestamps = 3_540_000 + np.arange(62, dtype=np.int64) * 60_000
    high = np.full(62, 105.0)
    low = np.full(62, 95.0)
    high[0] = 110.0
    low[0] = 90.0
    high[30] = 106.0
    low[30] = 94.0
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=1,
        trade_start_idx=1,
        requested_start_ts_ms=int(timestamps[0]),
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=len(timestamps) - 1,
    )

    hour_log_range, hour_valid = _build_hourly_log_range(high, low, timestamps, run)

    assert not hour_valid[1]
    assert hour_valid[61]
    assert hour_log_range[61] == pytest.approx(np.log(106.0 / 94.0))


def test_nondivisor_interval_hour_bucket_matches_rust_boundary_overlap():
    interval_ms = 7 * 60_000
    timestamps = np.arange(20, dtype=np.int64) * interval_ms
    high = np.full(20, 101.0)
    low = np.full(20, 99.0)
    # The 00:56 candle crosses the 01:00 boundary. Exact Rust includes it in
    # both the hour ending at 01:03 and the next window starting from floor(
    # 01:00 / seven minutes), because an aggregated candle cannot be split.
    high[8] = 200.0
    low[8] = 50.0
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=1,
        trade_start_idx=1,
        requested_start_ts_ms=0,
        guard_ts_ms=0,
        first_ts_ms=0,
        interval_ms=interval_ms,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=len(timestamps) - 1,
    )

    hour_log_range, hour_valid = _build_hourly_log_range(high, low, timestamps, run)

    assert hour_valid[9]
    assert hour_valid[18]
    assert hour_log_range[9] == pytest.approx(np.log(200.0 / 50.0))
    assert hour_log_range[18] == pytest.approx(np.log(200.0 / 50.0))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_anchor_multicoin_directional_shader_smoke(side):
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_multicoin_source_py(),
        ema_tail_enabled=True,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    assert "kernel void passivbot_ema_anchor_multicoin" in source
    assert "kernel void passivbot_ema_anchor_multicoin_long" in source
    assert f"constant int PARAM_COLS = {len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)}" in source
    assert f"constant int OVERRIDE_COLS = {EMA_ANCHOR_COIN_OVERRIDE_COLS}" in source
    assert "allowed_wallet_exposure_limit" in source
    assert "twel_entry_gate_enabled" in source
    assert "twel_enforcer_enabled" in source
    assert "twel_enforcer_reduce_portfolio" in source
    assert "clamped_market_price" in source
    assert "secondary_close_qty" in source
    assert "finalize_ema_multicoin_reducers_one_side(" in source
    assert "finalize_ema_multicoin_reducers_fused(" in source
    assert "gate_ema_multicoin_close(" in source
    assert "realized_loss_proxy_allows_close" not in source
    assert "constant int DAILY_COLS = 9" in source
    assert "day_min_balance" in source
    assert "coin_override_or" in source
    assert "const float score_hysteresis = fmax(run_settings[4], 0.0f)" in source
    assert "incumbent[c] = side.entry_qty[c] > 0.0f && psize[c] <= 0.0f" in source
    assert "if (!selected[c] || incumbent[c] || !survivor[c]) continue" in source
    assert "score[challenger] - score[incumbent_coin]" in source
    count = 512
    coin_count = 3
    phase = np.linspace(0.0, 12.0 * np.pi, count)
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    for coin in range(coin_count):
        close = 100.0 + coin * 20.0 + np.sin(phase + coin) * (4.0 + coin)
        hlcvs[:, coin, 0] = close * 1.01
        hlcvs[:, coin, 1] = close * 0.99
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002) for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            10,
            10,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    row = [
        0.1,
        10.0,
        30.0,
        1.5,
        0.01,
        0.0,
        0.0,
        0.0,
        60.0,
        60.0,
        0.0,
        1.0,
        60.0,
        60.0,
        0.0,
        1.0,
        0.0,
        0.0,
        2.0,
    ]
    row += _single_coin_exposure_fields() + [0.0, 0.0]
    row += list(_UNSTUCK_DISABLED_VALUES.values())
    row += list(_HSL_DISABLED_VALUES.values()) + [2.0, 3.0]
    row += list(_ADAPTIVE_DISABLED_VALUES.values()) + [43200.0, 43200.0]

    runner = MpsEmaAnchorMulticoinRunner(
        runs[0],
        data,
        pnl_lookback_bars=1440,
        side=side,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=0.1,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
        hsl_raw_drawdown_enabled=True,
        recovery_distribution_enabled=True,
    )
    assert runner.settings.cpu()[4].item() == pytest.approx(0.02)
    assert runner.settings.cpu()[5].item() <= 0.1
    assert runner.settings.cpu()[9].item() == 1.0
    assert runner.settings.cpu()[10].item() == pytest.approx(0.003)
    output = runner.run(np.array([row, row], dtype=np.float64))
    synchronize()

    assert runner._buffers[2][1].shape == (2, 65)
    assert (output["hsl_drawdown_ema_mean_worst_1pct_long"] == 0.0).all()
    assert (output["hsl_drawdown_ema_mean_worst_1pct_short"] == 0.0).all()
    assert (output["hsl_drawdown_raw_max_long"] == 0.0).all()
    assert (output["hsl_drawdown_raw_max_short"] == 0.0).all()
    assert output["balance"].device.type == gpu_device()
    assert output["balance"].shape == (2,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_min_balance"].shape == output["day_min_eq"].shape
    assert torch.isfinite(
        output["day_min_balance"][output["day_min_eq"].isfinite()]
    ).all()
    assert output["day_has_fill"].sum().item() > 0
    assert (output["open_positions"] <= 2.0).all()
    _assert_fill_scalar_contract(output)
    if side == "long":
        assert torch.equal(output["fill_count_long"], output["fill_count"])
    else:
        assert (output["fill_count_long"] == 0.0).all()
    recovery = strategy_eq_recovery_distribution_from_samples(
        output["strategy_eq_recovery_samples"],
        sample_interval_days=output["strategy_eq_recovery_sample_interval_days"],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[:, 3] > 0.0).all().item()

    with pytest.raises(ValueError, match="finite and non-negative"):
        MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            pnl_lookback_bars=1440,
            side=side,
            forager_score_hysteresis_pct=-0.01,
        )
    with pytest.raises(ValueError, match="max_realized_loss_pct"):
        MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            pnl_lookback_bars=1440,
            side=side,
            max_realized_loss_pct=float("nan"),
        )
    with pytest.raises(ValueError, match="market_order_near_touch_threshold"):
        MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            pnl_lookback_bars=1440,
            side=side,
            market_order_near_touch_threshold=float("nan"),
        )


    disabled = np.full(
        (coin_count, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    disabled[:, 11] = 0.0
    disabled_output = MpsEmaAnchorMulticoinRunner(
        runs[0], data, pnl_lookback_bars=1440, side=side, coin_overrides=disabled
    ).run(np.array([row], dtype=np.float64))
    synchronize()
    assert disabled_output["day_has_fill"].sum().item() == 0
    assert disabled_output["open_positions"].item() == 0.0

    exact_last = np.full(
        (coin_count, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    exact_last[:, :11] = np.asarray(row[:11], dtype=np.float32)
    changed_candidate = list(row)
    changed_candidate[:11] = [
        0.01,
        200.0,
        400.0,
        0.0,
        0.05,
        2.0,
        5.0,
        5.0,
        10.0,
        10.0,
        30.0,
    ]
    exact_last_output = MpsEmaAnchorMulticoinRunner(
        runs[0], data, pnl_lookback_bars=1440, side=side, coin_overrides=exact_last
    ).run(np.array([row, changed_candidate], dtype=np.float64))
    synchronize()
    assert torch.equal(
        exact_last_output["day_has_fill"][0], exact_last_output["day_has_fill"][1]
    )
    assert torch.equal(
        exact_last_output["day_end_eq"][0], exact_last_output["day_end_eq"][1]
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_market_entry_uses_stored_next_close_intent(side):
    count = 5
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    closes[-1] *= 1.1 if side == "long" else 0.9
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    resting_runner, row = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
    )
    market_runner, _ = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.0005
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = 100.0
    params = np.asarray([row], dtype=np.float64)

    resting = resting_runner.run(params)
    promoted = market_runner.run(params)
    synchronize()

    assert market_runner.settings[9].item() == 1.0
    assert market_runner.settings[10].item() == pytest.approx(0.001)
    assert resting["fill_count"].item() == 0.0
    assert promoted["fill_count"].item() == 2.0
    assert promoted["open_positions"].item() == 2.0
    assert promoted["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]
    assert promoted["balance"].item() < 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_trailing_market_entry_uses_stored_next_close_intent(side):
    count = 5
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    closes[-1] *= 1.1 if side == "long" else 0.9
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    resting_runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
    )
    market_runner, _ = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_retracement_base_pct")
    ] = 0.01
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_retracement_base_pct")
    ] = 0.01
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_ema_dist")] = (
        0.0005
    )
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = (
        100.0
    )
    params = np.asarray([row], dtype=np.float64)

    resting = resting_runner.run(params)
    promoted = market_runner.run(params)
    synchronize()

    assert market_runner.settings[9].item() == 1.0
    assert market_runner.settings[10].item() == pytest.approx(0.001)
    assert resting["fill_count"].item() == 0.0
    assert promoted["fill_count"].item() == 2.0
    assert promoted["open_positions"].item() == 2.0
    assert promoted["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]
    assert promoted["balance"].item() < 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_market_entry_does_not_expose_suffix(side):
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "twel_entry_gate_enabled": 0.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    # The promoted first order executes, but its passive limit did not cross.
    # Market policy therefore cannot reveal the recursive suffix.
    assert output["fill_count_entry"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_batch_selects_market_mode_per_candidate(side):
    count = 6
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    runner, recursive = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(2, 2),
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "twel_entry_gate_enabled": 0.0,
        "entry_cooldown_minutes": 0.0,
    }.items():
        recursive[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    trailing = recursive.copy()
    trailing[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_retracement_base_pct")
    ] = 0.01

    output = runner.run(np.asarray([recursive, trailing], dtype=np.float64))
    synchronize()

    assert output["fill_count_entry"].cpu().tolist()[0] > 2.0
    assert output["fill_count_entry"].cpu().tolist()[1] == 2.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_coin_override_controls_initial_ema_gate(side):
    count = 6
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[2:] = 99.5
    else:
        highs[2:] = 100.5
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    overrides[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN] = 0.0
    overrides[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN] = 1.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(1, 1),
        collect_coin_fill_counts=True,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.1,
        "entry_initial_qty_pct": 0.1,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["coin_fill_counts"].cpu().tolist() == [[0.0, 1.0]]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_coin_override_controls_reentry_ema_gate(side):
    count = 7
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3:] = 98.5
    else:
        highs[3:] = 101.5
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    overrides[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN] = 0.0
    overrides[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(1, 1),
        collect_coin_fill_counts=True,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_ema_dist": 0.1,
        "entry_initial_qty_pct": 0.1,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "gate_initial": 0.0,
        "gate_reentry": 1.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    coin_fills = output["coin_fill_counts"].cpu().tolist()[0]
    assert coin_fills[0] == 1.0
    assert coin_fills[1] > coin_fills[0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_fused_coin_overrides_resolve_ema_gates_per_side():
    count = 6
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    lows[2:] = 99.5
    highs[2:] = 100.5
    empty = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    long_overrides = empty.copy()
    short_overrides = empty.copy()
    long_overrides[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN] = 0.0
    short_overrides[0, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN] = 0.0
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(1, 1),
        collect_coin_fill_counts=True,
        return_context=True,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.1,
        "entry_initial_qty_pct": 0.1,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=long_overrides,
        short_coin_overrides=short_overrides,
        collect_coin_fill_counts=True,
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    synchronize()

    assert output["fill_count_long"].item() == 1.0
    assert output["fill_count"].item() - output["fill_count_long"].item() == 1.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("cooldown, expands", [(0.0, True), (100.0, False)])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_recursive_market_entry_passive_suffix(
    side, cooldown, expands,
    market_orders_allowed,
):
    count = 6
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(2, 2),
        collect_coin_fill_counts=True,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "twel_entry_gate_enabled": 0.0,
        "entry_cooldown_minutes": cooldown,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    if expands:
        assert output["fill_count_entry"].item() > 2.0
        assert all(value > 1.0 for value in counts)
    else:
        assert output["fill_count_entry"].item() == 2.0
        assert counts == [1.0, 1.0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_market_entry_stops_duplicate_tick_suffix(
    side,
):
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(2, 2),
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.0,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "twel_entry_gate_enabled": 0.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["fill_count_entry"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "threshold, expected_counts",
    [(0.06, [0.0, 1.0]), (0.16, [1.0, 2.0])],
)
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_recursive_entry_gate_orders_cross_coin_ladders(
    side,
    threshold,
    expected_counts,
    market_orders_allowed,
):
    count = 5
    closes = np.tile(np.asarray([100.0, 100.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    markets = [ProxyMarket(0.001, 0.01, 0.001, 20.0, 1.0, 0.0) for _ in range(2)]
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        first_valid_indices=(2, 2),
        collect_coin_fill_counts=True,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 2.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": threshold,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["fill_count_entry"].item() == sum(expected_counts)
    assert output["coin_fill_counts"].cpu().tolist() == [expected_counts]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_recursive_entry_gate_keeps_partial_boundary(side, market_orders_allowed):
    count = 5
    closes = np.tile(np.asarray([100.0, 100.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(2, 2),
        collect_coin_fill_counts=True,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 2.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 0.18,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["fill_count_entry"].item() == 4.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 2.0]]
    exposure = output["total_wallet_exposure_max"].item()
    assert 0.17 < exposure < 0.18


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_coin_override_selects_recursive_market_entry(side):
    count = 6
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 11] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(2, 2),
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.01,
        "close_retracement_base_pct": 0.01,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    assert counts[0] == 1.0
    assert counts[1] > 1.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_trailing_market_close_uses_stored_next_close_intent(side):
    count = 8
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        closes[5] = base * 1.08
        highs[5] = base * 1.10
        lows[5] = base * 1.08
        closes[6] = base * 0.90
        highs[6] = closes[6]
        lows[6] = closes[6]
    else:
        closes[5] = base * 0.92
        highs[5] = base * 0.92
        lows[5] = base * 0.90
        closes[6] = base * 1.10
        highs[6] = closes[6]
        lows[6] = closes[6]
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 10.0,
        "entry_retracement_base_pct": 0.01,
        "close_qty_pct": 1.0,
        "close_threshold_base_pct": 0.02,
        "close_retracement_base_pct": 0.01,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    params = np.asarray([row, row], dtype=np.float64)

    output = runner.run(
        params,
        end_steps=np.asarray([count - 2, count - 1], dtype=np.int32),
    )
    synchronize()

    assert output["fill_count"].cpu().tolist() == [2.0, 4.0]
    assert output["fill_count_entry"].cpu().tolist() == [2.0, 2.0]
    assert output["coin_fill_counts"].cpu().tolist() == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() > 0.0
    assert output[size_key][1].item() == 0.0
    assert output["balance"][1].item() < output["balance"][0].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_next_close_promotes_without_suffix(side):
    count = 7
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.001,
        "close_qty_pct": 1.0,
        "close_threshold_base_pct": 0.0005,
        "close_retracement_base_pct": 0.0,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # No passive recursive close crosses. Exact Rust emits and promotes only
    # calc_next_close for each coin, rather than exposing the suffix.
    assert output["fill_count"].item() == 4.0
    assert output["fill_count_entry"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 2.0]]
    assert output[size_key].item() == 0.0
    assert output["balance"].item() < 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_expanded_recursive_close_promotes_each_group(side, market_orders_allowed):
    count = 6
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    trigger_distance = 0.06
    if side == "long":
        highs[4] = base * (1.0 + trigger_distance)
    else:
        lows[4] = base * (1.0 - trigger_distance)
    if not market_orders_allowed:
        highs[3] = base * 1.01
        lows[3] = base * 0.99
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.1,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_initial_qty_pct": 1.0,
        "entry_retracement_base_pct": 0.001,
        "close_qty_pct": 0.25,
        "close_threshold_base_pct": 0.005,
        "close_threshold_we_weight": 0.01,
        "close_retracement_base_pct": 0.0,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # A passive group reaches on the one post-entry execution candle. More
    # than one close per coin proves the immutable suffix was emitted and its
    # groups were independently promoted against the generation snapshot.
    assert output["fill_count_entry"].item() == 2.0
    assert output["fill_count"].item() > 4.0, (
        output["coin_fill_counts"].cpu().tolist(),
        output[size_key].item(),
        output["first_fill_ts"].item(),
        output["last_fill_ts"].item(),
    )
    assert all(count > 2.0 for count in output["coin_fill_counts"][0].cpu())
    assert output[size_key].item() == 0.0
    if market_orders_allowed:
        assert output["balance"].item() < 1_000.0
    else:
        # No taker fees or market promotion: filled passive close groups profit.
        assert output["balance"].item() > 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_coin_override_selects_recursive_market_close(side):
    count = 7
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 20] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.001,
        "close_qty_pct": 1.0,
        "close_threshold_base_pct": 0.0005,
        "close_retracement_base_pct": 0.01,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The base trailing coin never triggers on flat candles. The overridden
    # recursive coin emits and promotes its ordinary next close.
    assert output["fill_count_entry"].item() == 2.0
    assert output["fill_count"].item() == 3.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 2.0]]
    assert output[size_key].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_market_close_uses_executable_minimum(side):
    count = 7
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 100.04, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.02,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.001,
        "close_qty_pct": 0.1,
        "close_threshold_base_pct": 0.0005,
        "close_threshold_we_weight": 0.01,
        "close_retracement_base_pct": 0.0,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(
        np.asarray([row], dtype=np.float64),
        end_steps=np.asarray([5], dtype=np.int32),
    )
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output["fill_count_entry"].item() == 1.0
    assert output["fill_count"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 0.0]]
    # The requested 10% close is below the executable market minimum; the
    # promoted order is raised to that minimum without closing the position.
    expected_remaining = 8.998 if side == "long" else 8.987
    assert output[size_key].item() == pytest.approx(expected_remaining, abs=0.002)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_recursive_market_close_preserves_below_min_position():
    count = 7
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[3:, 0] = 50.0
    highs = closes.copy()
    lows = closes.copy()
    highs[4, 0] = 100.0
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 600.0, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        "short",
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.02,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.001,
        "close_qty_pct": 0.1,
        "close_threshold_base_pct": 0.0005,
        "close_threshold_we_weight": 0.01,
        "close_retracement_base_pct": 0.0,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(
        np.asarray([row], dtype=np.float64),
        end_steps=np.asarray([5], dtype=np.int32),
    )
    synchronize()

    # The short opens while the executable minimum is six units. After price
    # halves, the minimum rises to twelve units, above the whole remaining
    # position. Exact Rust preserves one recursive market close at the full
    # position size instead of filtering every group and leaving dust open.
    assert output["fill_count_entry"].item() == 1.0
    assert output["fill_count"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 0.0]]
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_short_market_entry_uses_executable_minimum():
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 100.04, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full((2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 11] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "ema_anchor",
        "short",
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("base_qty_pct")] = 0.0001
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.0005
    params = np.asarray([row], dtype=np.float64)

    output = runner.run(params)
    synchronize()

    assert output["fill_count"].item() == 1.0
    assert output["fill_count_entry"].item() == 1.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 0.0]]
    assert output["short_psize"].item() == pytest.approx(1.001)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_short_market_entry_uses_executable_minimum():
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 100.04, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        "short",
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )
    for key, value in {
        "entry_initial_qty_pct": 0.0001,
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.01,
        "close_retracement_base_pct": 0.01,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    assert output["fill_count"].item() == 1.0
    assert output["fill_count_entry"].item() == 1.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 0.0]]
    assert output["short_psize"].item() == pytest.approx(1.001)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_market_entry_cap_uses_market_touch(side):
    count = 5
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[-1] = 99.0
    else:
        highs[-1] = 101.0
    markets = [ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0) for _ in range(2)]
    market_runner, row = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("base_qty_pct")] = 1.0
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.002
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("total_wallet_exposure_limit")] = 0.2
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("n_positions")] = 2.0
    params = np.asarray([row], dtype=np.float64)

    promoted = market_runner.run(params)
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert promoted["fill_count"].item() == 2.0
    # Rust retains the full 1.002 long clip on one coin and portfolio-clips
    # the other to .997 at the market touch. Short clips are .998 each.
    expected_size = 1.999 if side == "long" else 1.996
    assert promoted[size_key].item() == pytest.approx(expected_size)
    assert promoted["total_wallet_exposure_max"].item() == pytest.approx(
        expected_size * 100.0 / 1_000.0
    )
    assert promoted["total_wallet_exposure_max"].item() < 0.2


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_market_entry_cap_uses_market_touch(side):
    count = 5
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[-1] = 99.0
    else:
        highs[-1] = 101.0
    markets = [ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0) for _ in range(2)]
    market_runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
    )
    for key, value in {
        "entry_initial_qty_pct": 1.0,
        "entry_initial_ema_dist": 0.002,
        "entry_retracement_base_pct": 0.01,
        "close_retracement_base_pct": 0.01,
        "total_wallet_exposure_limit": 0.2,
        "n_positions": 2.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    promoted = market_runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert promoted["fill_count"].item() == 2.0
    expected_size = 1.999
    assert promoted[size_key].item() == pytest.approx(expected_size)
    passive_limit_size = 2.003 if side == "long" else 1.995
    assert promoted[size_key].item() != pytest.approx(passive_limit_size)
    assert promoted["total_wallet_exposure_max"].item() == pytest.approx(
        expected_size * 100.0 / 1_000.0
    )
    assert promoted["total_wallet_exposure_max"].item() < 0.2


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_market_close_uses_stored_next_close_intent(side):
    count = 6
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    closes[-1] *= 0.9 if side == "long" else 1.1
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    resting_runner, row = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
    )
    market_runner, _ = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        collect_coin_fill_counts=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.0005
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = 100.0
    params = np.asarray([row], dtype=np.float64)

    resting = resting_runner.run(params)
    promoted = market_runner.run(
        np.asarray([row, row], dtype=np.float64),
        end_steps=np.asarray([count - 2, count - 1], dtype=np.int32),
    )
    synchronize()

    assert resting["fill_count"].item() == 0.0
    assert promoted["fill_count"].cpu().tolist() == [2.0, 4.0]
    assert promoted["fill_count_entry"].cpu().tolist() == [2.0, 2.0]
    assert promoted["coin_fill_counts"].cpu().tolist() == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    size_key = "psize" if side == "long" else "short_psize"
    assert promoted[size_key][1].item() < promoted[size_key][0].item()
    assert promoted["balance"][1].item() < promoted["balance"][0].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_multicoin_fused_market_execution_covers_both_sides():
    count = 6
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    _, row, run, data = _multicoin_exposure_fixture(
        "ema_anchor",
        "long",
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        return_context=True,
        market_order_slippage_pct=0.01,
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.0005
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = 100.0
    params = np.asarray([row + row, row + row], dtype=np.float64)
    overrides = np.full((2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    resting_runner = MpsEmaAnchorMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=True,
        market_order_slippage_pct=0.01,
    )
    market_runner = MpsEmaAnchorMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )

    end_steps = np.asarray([count - 2, count - 1], dtype=np.int32)
    resting = resting_runner.run(params, end_steps=end_steps)
    promoted = market_runner.run(params, end_steps=end_steps)
    synchronize()

    assert market_runner.settings[11].item() == 1.0
    assert market_runner.settings[12].item() == pytest.approx(0.001)
    assert resting["fill_count"].cpu().tolist() == [0.0, 0.0]
    assert promoted["fill_count"].cpu().tolist() == [4.0, 8.0]
    assert promoted["fill_count_long"].cpu().tolist() == [2.0, 4.0]
    assert promoted["coin_fill_counts"].cpu().tolist() == [
        [2.0, 2.0],
        [4.0, 4.0],
    ]
    assert promoted["psize"][1].item() < promoted["psize"][0].item()
    assert promoted["short_psize"][1].item() < promoted["short_psize"][0].item()
    assert promoted["balance"][1].item() < promoted["balance"][0].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("hedge_mode", [True, False])
def test_mps_tm_multicoin_fused_trailing_market_entries_respect_position_mode(
    hedge_mode,
):
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=count,
        closes=closes,
        highs=closes,
        lows=closes,
        markets=markets,
        return_context=True,
        market_order_slippage_pct=0.01,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.01,
        "close_retracement_base_pct": 0.01,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    params = np.asarray([row + row], dtype=np.float64)
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    resting_runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
    )
    market_runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )

    resting = resting_runner.run(params)
    promoted = market_runner.run(params)
    synchronize()

    assert market_runner.settings[11].item() == 1.0
    assert market_runner.settings[12].item() == pytest.approx(0.001)
    assert resting["fill_count"].item() == 0.0
    expected_fills = 4.0 if hedge_mode else 2.0
    assert promoted["fill_count"].item() == expected_fills
    assert promoted["fill_count_entry"].item() == expected_fills
    expected_coin_fills = [2.0, 2.0] if hedge_mode else [1.0, 1.0]
    assert promoted["coin_fill_counts"].cpu().tolist() == [expected_coin_fills]
    if hedge_mode:
        assert promoted["fill_count_long"].item() == 2.0
        assert promoted["psize"].item() > 0.0
        assert promoted["short_psize"].item() > 0.0
    else:
        assert promoted["fill_count_long"].item() in (0.0, 2.0)
        assert (promoted["psize"].item() > 0.0) != (
            promoted["short_psize"].item() > 0.0
        )
    assert promoted["balance"].item() < 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("hedge_mode", [True, False])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_fused_recursive_market_entries_respect_position_mode(
    hedge_mode,
    market_orders_allowed,
):
    count = 6
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    highs[3] = closes[3] * 1.2
    lows[3] = closes[3] * 0.8
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        first_valid_indices=(2, 2),
        return_context=True,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": 0.0005,
        "entry_threshold_base_pct": 0.01,
        "entry_retracement_base_pct": 0.0,
        "close_retracement_base_pct": 0.01,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    params = np.asarray([row + row], dtype=np.float64)
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    )

    output = runner.run(params)
    synchronize()

    assert output["fill_count_entry"].item() > (4.0 if hedge_mode else 2.0)
    assert all(
        value > (2.0 if hedge_mode else 1.0)
        for value in output["coin_fill_counts"].cpu().tolist()[0]
    )
    if hedge_mode:
        assert output["psize"].item() > 0.0
        assert output["short_psize"].item() > 0.0
    else:
        assert (output["psize"].item() > 0.0) != (output["short_psize"].item() > 0.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("hedge_mode", [True, False])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_fused_recursive_market_closes_respect_position_mode(
    hedge_mode,
    market_orders_allowed,
):
    count = 6
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    highs[4] = base * 1.06
    lows[4] = base * 0.94
    if not market_orders_allowed:
        highs[3] = base * 1.01
        lows[3] = base * 0.99
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            maker_fee=0.0,
            taker_fee=0.01,
        )
        for _ in range(2)
    ]
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        return_context=True,
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_initial_qty_pct": 1.0,
        "entry_retracement_base_pct": 0.001,
        "close_qty_pct": 0.25,
        "close_threshold_base_pct": 0.005,
        "close_threshold_we_weight": 0.01,
        "close_retracement_base_pct": 0.0,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    params = np.asarray([row + row], dtype=np.float64)
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.1,
    )

    output = runner.run(params)
    synchronize()

    expected_entries = 4.0 if hedge_mode else 2.0
    assert output["fill_count_entry"].item() == expected_entries
    assert output["fill_count"].item() > expected_entries * 2.0
    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() == 0.0
    expected_min_coin_fills = 4.0 if hedge_mode else 2.0
    assert all(
        count > expected_min_coin_fills for count in output["coin_fill_counts"][0].cpu()
    )
    if market_orders_allowed:
        assert output["balance"].item() < 1_000.0
    else:
        # No taker fees or market promotion: filled passive close groups profit.
        assert output["balance"].item() > 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_ema_anchor_multicoin_fused_kernel_smoke_all_hsl_modes():
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_multicoin_source_py(),
        ema_tail_enabled=True,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    assert "kernel void passivbot_ema_anchor_multicoin_fused" in source
    assert "constant int FUSED_SCALAR_COLS = 73" in source

    count = 512
    coin_count = 3
    phase = np.linspace(0.0, 12.0 * np.pi, count)
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    for coin in range(coin_count):
        close = 100.0 + coin * 20.0 + np.sin(phase + coin) * (4.0 + coin)
        hlcvs[:, coin, 0] = close * 1.01
        hlcvs[:, coin, 1] = close * 0.99
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002) for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            10,
            10,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)

    base = [
        0.1,
        10.0,
        30.0,
        1.5,
        0.01,
        0.0,
        0.0,
        0.0,
        60.0,
        60.0,
        0.0,
        1.0,
        60.0,
        60.0,
        0.0,
        1.0,
        0.0,
        0.0,
        2.0,
    ]
    base += _single_coin_exposure_fields() + [0.0, 0.0]
    base += list(_UNSTUCK_DISABLED_VALUES.values())

    def side_row(signal_mode):
        hsl = dict(_HSL_DISABLED_VALUES)
        hsl["hsl_enabled"] = 1.0
        hsl["hsl_red_threshold"] = 0.9
        hsl["hsl_signal_mode"] = float(signal_mode)
        row = base + list(hsl.values()) + [2.0, 3.0] + list(_ADAPTIVE_DISABLED_VALUES.values()) + [43200.0, 43200.0]
        assert len(row) == len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        return row

    unstuck_long = side_row(0)
    unstuck_long[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("unstuck_enabled")] = 1.0
    unstuck_short = side_row(0)
    unstuck_short[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("unstuck_enabled")] = 1.0

    rows = [
        side_row(0) + side_row(0),
        side_row(1) + side_row(1),
        side_row(2) + side_row(2),
        side_row(0) + side_row(1),
        side_row(3) + side_row(3),
        unstuck_long + side_row(0),
        side_row(0) + unstuck_short,
        unstuck_long + unstuck_short,
    ]
    batch_size = len(rows)
    params = torch.as_tensor(
        np.asarray(rows, dtype=np.float32), device=gpu_device()
    ).contiguous()
    overrides = torch.full(
        (coin_count, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    run_settings = torch.tensor(
        [
            1_000.0,
            50.0,
            60_000.0,
            0.0,
            0.02,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.001,
            0.0,
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    sizes = torch.tensor(
        [
            batch_size,
            data["n"],
            coin_count,
            data["n_days"],
            0,
            10,
            data["start_minute_of_day"],
            data["start_minute_of_hour"],
        ],
        dtype=torch.int32,
        device=gpu_device(),
    )
    end_steps = torch.full(
        (batch_size,), data["n"] - 1, dtype=torch.int32, device=gpu_device()
    )
    daily = torch.zeros(
        (batch_size, data["n_days"], 9), dtype=torch.float32, device=gpu_device()
    )
    daily[:, :, 1].fill_(float("inf"))
    daily[:, :, 5].fill_(float("inf"))
    scalars = torch.zeros(
        (batch_size, MPS_MULTICOIN_FUSED_SCALAR_COLS),
        dtype=torch.float32,
        device=gpu_device(),
    )
    gaps = torch.zeros((batch_size, GAP_BINS), dtype=torch.int32, device=gpu_device())
    coin_fill_counts = torch.zeros(
        (batch_size, coin_count), dtype=torch.float32, device=gpu_device()
    )

    # Match the production history specialization and scratch ABI.
    capacity = min(count + 2, 1440 + 2)
    _, nodes = _hsl_layout(capacity)
    hsl_buffers = (
        torch.empty(
            (batch_size, 2 * (coin_count + 1), nodes, 32),
            dtype=torch.uint8,
            device=gpu_device(),
        ),
        torch.empty(
            (batch_size, 2 * (coin_count + 1), capacity * 2),
            dtype=torch.int32,
            device=gpu_device(),
        ),
    )
    source = "#define PASSIVBOT_HSL_LOOKBACK 1440\n" + _with_hsl(source, capacity)
    source = source.replace(
        "constant int MAX_COINS = 64;", f"constant int MAX_COINS = {coin_count};"
    )
    library = compile_shader(source)
    library.passivbot_ema_anchor_multicoin_fused(
        data["market_execution_ticks"],
        data["bars"],
        data["fill_ticks"],
        data["touch_ticks"],
        data["hour_log_ranges"],
        data["coin_settings"],
        overrides,
        overrides,
        params,
        run_settings,
        _pack_replay_control(sizes, end_steps, sizes.numel() == 10),
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        *hsl_buffers,
        threads=(batch_size, 1, 1),
    )
    synchronize()
    values = scalars.cpu().numpy()

    assert values[:3, 13].tolist() == [1.0, 1.0, 1.0]
    assert (values[:3, 24] > 0.0).all()
    assert (values[:3, 26] > 0.0).all()
    assert (values[:3, 24] - values[:3, 26] > 0.0).all()
    assert values[:3, 32:34].tolist() == [[1.0, 1.0]] * 3
    assert (values[:3, 38] > 0.0).all()
    assert (values[:3, 64] > 0.0).all()
    assert (values[:3, 65] > 0.0).all()
    assert (values[:3, 66] > 0.0).all()
    assert (values[:3, 67] > 0.0).all()
    np.testing.assert_allclose(values[:3, 18], values[:3, 58] + values[:3, 60])
    np.testing.assert_allclose(values[:3, 19], values[:3, 59] + values[:3, 61])
    np.testing.assert_allclose(values[:3, 24], daily[:3, :, 8].sum(dim=1).cpu().numpy())
    # Rust analyzes abs(long TWE + signed short TWE), not gross exposure.
    assert (values[:3, 22] <= 1.01).all()
    assert (coin_fill_counts[:3].sum(dim=1).cpu().numpy() > 0.0).all()
    # Mixed/unknown signal modes remain fail closed. Auto-unstuck is valid on
    # either or both directional surfaces; the fused kernel chooses only one
    # global least-stuck candidate before generating side orders.
    assert values[3:5, 9].tolist() == [0.0, 0.0]
    assert values[3:5, 13].tolist() == [0.0, 0.0]
    assert values[3:5, 24].tolist() == [0.0, 0.0]
    assert values[5:, 9].tolist() == [-1.0, -1.0, -1.0]
    assert values[5:, 13].tolist() == [1.0, 1.0, 1.0]
    assert (values[5:, 24] > 0.0).all()
    assert values[5, 24] > values[0, 24]

    override_matrix = np.full(
        (coin_count, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    runner = MpsEmaAnchorMulticoinFusedRunner(
        runs[0],
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=override_matrix,
        short_coin_overrides=override_matrix,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=1.0,
        collect_coin_fill_counts=True,
        hsl_ema_tail_enabled=True,
        hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True,
        recovery_distribution_enabled=True,
    )
    torch.testing.assert_close(runner.settings, run_settings)
    for invalid_row in rows[3:5]:
        with pytest.raises(ValueError, match="HSL"):
            runner.run(np.asarray([invalid_row], dtype=np.float64))
    valid_indices = [0, 1, 2, 5, 6, 7]
    runner_output = runner.run(
        np.asarray(rows, dtype=np.float64)[valid_indices], profile=True
    )
    synchronize()
    torch.testing.assert_close(runner_output["day_end_eq"], daily[valid_indices, :, 0])
    torch.testing.assert_close(runner_output["day_min_eq"], daily[valid_indices, :, 1])
    torch.testing.assert_close(runner_output["fill_count"], scalars[valid_indices, 24])
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_long"], scalars[valid_indices, 21]
    )
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_short"], scalars[valid_indices, 57]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_long"], scalars[valid_indices, 55]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_short"], scalars[valid_indices, 56]
    )
    torch.testing.assert_close(
        runner_output["profit_sum_long"], scalars[valid_indices, 58]
    )
    torch.testing.assert_close(
        runner_output["loss_sum_long"], scalars[valid_indices, 59]
    )
    torch.testing.assert_close(
        runner_output["profit_sum_short"], scalars[valid_indices, 60]
    )
    torch.testing.assert_close(
        runner_output["loss_sum_short"], scalars[valid_indices, 61]
    )
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_long"],
        scalars[valid_indices, 62],
    )
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_short"],
        scalars[valid_indices, 63],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_long"],
        scalars[valid_indices, 64],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_short"],
        scalars[valid_indices, 65],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_long"], scalars[valid_indices, 66]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_short"], scalars[valid_indices, 67]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_mean_worst_1pct_long"],
        scalars[valid_indices, 68],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_mean_worst_1pct_short"],
        scalars[valid_indices, 69],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_portfolio"],
        scalars[valid_indices, 70],
    )
    assert runner_output["alive"].cpu().tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert torch.equal(
        runner_output["coin_fill_counts"], coin_fill_counts[valid_indices]
    )
    recovery = strategy_eq_recovery_distribution_from_samples(
        runner_output["strategy_eq_recovery_samples"],
        sample_interval_days=runner_output["strategy_eq_recovery_sample_interval_days"],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[:, 3] > 0.0).all().item()
    assert runner.last_profile["kernel_seconds"] >= 0.0

    metric_rows = []
    red_threshold_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("hsl_red_threshold")
    slow_span_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("hsl_memory_ema_span_minutes")
    for signal_mode in range(3):
        row = side_row(signal_mode) + side_row(signal_mode)
        for side_offset in (0, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)):
            row[side_offset + red_threshold_index] = 0.001
            row[side_offset + slow_span_index] = 43200.0
        metric_rows.append(row)

    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"
    proxy.batch_size = 3
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": data["ts0"], "n": data["n"]}
    proxy.run = runs[0]
    proxy.sides = ["long", "short"]
    proxy.needed_metrics = {
        "hard_stop_panic_close_loss_drawdown_pct_mean",
        "hard_stop_time_in_red_pct",
        "hard_stop_triggers",
        "hard_stop_triggers_long",
        "hard_stop_triggers_short",
        "drawdown_worst_ema_strategy_eq",
        "drawdown_worst_ema_strategy_eq_long",
        "drawdown_worst_ema_strategy_eq_short",
        "drawdown_worst_mean_1pct_ema_strategy_eq",
        "drawdown_worst_mean_1pct_ema_strategy_eq_long",
        "drawdown_worst_mean_1pct_ema_strategy_eq_short",
        "drawdown_worst_strategy_eq_long",
        "drawdown_worst_strategy_eq_short",
        "peak_recovery_hours_strategy_eq_long",
        "peak_recovery_hours_strategy_eq_short",
        "peak_recovery_days_strategy_eq_long",
        "peak_recovery_days_strategy_eq_short",
    }
    proxy.param_keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    width = len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
    proxy.base_params = {
        "long": dict(zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, metric_rows[0][:width])),
        "short": dict(zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, metric_rows[0][width:])),
    }
    proxy.weighted_equity_metrics = ()
    proxy.fused_runner = runner
    proxy.runners = {}
    candidates = [
        {
            **{
                f"long_{key}": row[index]
                for index, key in enumerate(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
            },
            **{
                f"short_{key}": row[width + index]
                for index, key in enumerate(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
            },
        }
        for row in metric_rows
    ]
    np.testing.assert_allclose(
        proxy._parameter_matrix(candidates, "long"),
        np.asarray(metric_rows, dtype=np.float64)[:, :width],
    )
    np.testing.assert_allclose(
        proxy._parameter_matrix(candidates, "short"),
        np.asarray(metric_rows, dtype=np.float64)[:, width:],
    )
    seen_hsl_triggers = {}

    def reduce_service_output(output, *args, **kwargs):
        seen_hsl_triggers["long"] = output["hsl_triggers_long"].clone()
        seen_hsl_triggers["short"] = output["hsl_triggers_short"].clone()
        seen_hsl_triggers["raw_long"] = output["hsl_drawdown_raw_max_long"].clone()
        seen_hsl_triggers["raw_short"] = output["hsl_drawdown_raw_max_short"].clone()
        seen_hsl_triggers["portfolio_ema"] = output[
            "hsl_drawdown_ema_mean_worst_1pct_portfolio"
        ].clone()
        return compute_objectives(output, *args, **kwargs)

    proxy._compute_objectives = reduce_service_output
    service_results = proxy.evaluate(candidates)
    assert (seen_hsl_triggers["long"] + seen_hsl_triggers["short"] > 0.0).all()
    assert all(item["hard_stop_triggers"] > 0.0 for item in service_results)
    # Unified mode reports portfolio events without attributing them to a side.
    assert [item["hard_stop_triggers"] for item in service_results] == pytest.approx(
        (seen_hsl_triggers["long"] + seen_hsl_triggers["short"]).tolist()
    )
    assert service_results[0]["hard_stop_triggers_long"] == 0
    assert service_results[0]["hard_stop_triggers_short"] == 0
    assert all(
        item["hard_stop_triggers"]
        == item["hard_stop_triggers_long"] + item["hard_stop_triggers_short"]
        for item in service_results[1:]
    )
    assert all(item["hard_stop_time_in_red_pct"] > 0.0 for item in service_results)
    assert all(
        item["hard_stop_panic_close_loss_drawdown_pct_mean"] > 0.0
        for item in service_results
    )
    assert all(
        item["drawdown_worst_ema_strategy_eq"]
        == max(
            item["drawdown_worst_ema_strategy_eq_long"],
            item["drawdown_worst_ema_strategy_eq_short"],
        )
        for item in service_results
    )
    assert [
        item["drawdown_worst_mean_1pct_ema_strategy_eq"] for item in service_results
    ] == pytest.approx(seen_hsl_triggers["portfolio_ema"].tolist())
    assert (seen_hsl_triggers["portfolio_ema"] > 0.0).all()
    for side in ("long", "short"):
        # A profitable side can have zero raw drawdown when portfolio HSL closes
        # both sides early. Verify the factual replay summary, not positivity.
        assert [
            item[f"drawdown_worst_strategy_eq_{side}"] for item in service_results
        ] == pytest.approx(seen_hsl_triggers[f"raw_{side}"].tolist())
        assert (seen_hsl_triggers[f"raw_{side}"] >= 0.0).all()
    assert all(
        item["drawdown_worst_strategy_eq_long"] > 0.0
        and item["drawdown_worst_strategy_eq_short"] > 0.0
        for item in service_results[1:]
    )
    assert all(
        item["peak_recovery_hours_strategy_eq_long"] >= 0.0
        and item["peak_recovery_hours_strategy_eq_short"] >= 0.0
        and item["peak_recovery_days_strategy_eq_long"]
        == pytest.approx(item["peak_recovery_hours_strategy_eq_long"] / 24.0)
        and item["peak_recovery_days_strategy_eq_short"]
        == pytest.approx(item["peak_recovery_hours_strategy_eq_short"] / 24.0)
        for item in service_results
    )

    with pytest.raises(ValueError, match=f"{2 * len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)} columns"):
        runner.run(np.asarray([side_row(0)], dtype=np.float64))
    truncated = runner.run(
        np.asarray([rows[0]], dtype=np.float64),
        end_steps=np.asarray([60], dtype=np.int32),
    )
    synchronize()
    assert truncated["last_eq_ts"].item() <= 59 * 60_000.0
    assert truncated["fill_count"].item() <= values[0, 24]

    # Coin overrides can enable auto-unstuck even when the side template does
    # not, so the kernel must inspect the effective per-coin setting too.
    override_unstuck = overrides.clone()
    override_unstuck[0, 13] = 1.0
    override_sizes = sizes.clone()
    override_sizes[0] = 1
    override_daily = torch.zeros(
        (1, data["n_days"], 9), dtype=torch.float32, device=gpu_device()
    )
    override_daily[:, :, 1].fill_(float("inf"))
    override_daily[:, :, 5].fill_(float("inf"))
    override_scalars = torch.zeros(
        (1, MPS_MULTICOIN_FUSED_SCALAR_COLS),
        dtype=torch.float32,
        device=gpu_device(),
    )
    override_gaps = torch.zeros((1, GAP_BINS), dtype=torch.int32, device=gpu_device())
    override_coin_fills = torch.zeros(
        (1, coin_count), dtype=torch.float32, device=gpu_device()
    )
    library.passivbot_ema_anchor_multicoin_fused(
        data["market_execution_ticks"],
        data["bars"],
        data["fill_ticks"],
        data["touch_ticks"],
        data["hour_log_ranges"],
        data["coin_settings"],
        override_unstuck,
        overrides,
        params[:1],
        run_settings,
        _pack_replay_control(override_sizes, end_steps[:1], override_sizes.numel() == 10),
        override_daily,
        override_scalars,
        override_gaps,
        override_coin_fills,
        *hsl_buffers,
        threads=(1, 1, 1),
    )
    synchronize()
    assert override_scalars[0, 9].item() == -1.0
    assert override_scalars[0, 13].item() == 1.0
    assert override_scalars[0, 24].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize(
    ("runner_cls", "param_keys"),
    [
        (MpsEmaAnchorMulticoinFusedRunner, EMA_ANCHOR_MULTICOIN_PARAM_KEYS),
        (
            MpsTrailingMartingaleMulticoinFusedRunner,
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
        ),
    ],
)
def test_mps_fused_multicoin_one_way_arbitrates_each_flat_symbol(
    runner_cls, param_keys
):
    count = 16
    coin_count = 2
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 102.0
    hlcvs[:, :, 1] = 98.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002) for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    values = {key: 0.0 for key in param_keys}
    values.update(
        {
            "n_positions": 2.0,
            "total_wallet_exposure_limit": 1.0,
            "forager_score_weights_ema_readiness": 1.0,
        }
    )
    if runner_cls is MpsEmaAnchorMulticoinFusedRunner:
        values.update(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "offset": 0.01,
            }
        )
    else:
        values.update(
            {
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_initial_ema_dist": 0.01,
                "entry_initial_qty_pct": 0.1,
                "gate_initial": 1.0,
                "gate_reentry": 1.0,
            }
        )
    values.update(unstuck_ema_span_0=17.25, unstuck_ema_span_1=211.75)
    side_row = [values[key] for key in param_keys]
    params = np.asarray([side_row + side_row], dtype=np.float64)

    hedge_output = runner_cls(
        runs[0], data, pnl_lookback_bars=1440, hedge_mode=True
    ).run(params)
    one_way_output = runner_cls(
        runs[0], data, pnl_lookback_bars=1440, hedge_mode=False
    ).run(params)
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if runner_cls is MpsEmaAnchorMulticoinFusedRunner
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    wel_col = 11 if runner_cls is MpsEmaAnchorMulticoinFusedRunner else 24
    long_overrides = np.full((coin_count, override_cols), np.nan, dtype=np.float32)
    short_overrides = np.full((coin_count, override_cols), np.nan, dtype=np.float32)
    long_overrides[0, wel_col] = 0.0
    short_overrides[1, wel_col] = 0.0
    mixed_output = runner_cls(
        runs[0],
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=long_overrides,
        short_coin_overrides=short_overrides,
        hedge_mode=False,
    ).run(params)
    synchronize()

    assert hedge_output["fill_count"].item() > hedge_output["fill_count_long"].item()
    assert one_way_output["fill_count"].item() > 0.0
    assert (
        one_way_output["fill_count"].item() == one_way_output["fill_count_long"].item()
    )
    assert (
        0.0 < mixed_output["fill_count_long"].item() < mixed_output["fill_count"].item()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_trailing_martingale_multicoin_fused_kernel_smoke_all_hsl_modes():
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py(),
        ema_tail_enabled=True,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    assert "kernel void passivbot_trailing_martingale_multicoin_fused" in source
    assert "constant int FUSED_SCALAR_COLS = 73" in source

    count = 512
    coin_count = 3
    phase = np.linspace(0.0, 12.0 * np.pi, count)
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    for coin in range(coin_count):
        close = 100.0 + coin * 20.0 + np.sin(phase + coin) * (6.0 + coin)
        hlcvs[:, coin, 0] = close * 1.015
        hlcvs[:, coin, 1] = close * 0.985
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002) for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            10,
            10,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)

    values = {
        "ema_span_0": 10.0,
        "ema_span_1": 30.0,
        "volatility_ema_span_1h": 60.0,
        "volatility_ema_span_1m": 60.0,
        "entry_double_down_factor": 1.5,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.1,
        "entry_threshold_base_pct": 0.02,
        "entry_threshold_we_weight": 0.0,
        "entry_threshold_volatility_1h_weight": 0.0,
        "entry_threshold_volatility_1m_weight": 0.0,
        "entry_retracement_base_pct": 0.0,
        "entry_retracement_we_weight": 0.0,
        "entry_retracement_volatility_1h_weight": 0.0,
        "entry_retracement_volatility_1m_weight": 0.0,
        "close_qty_pct": 0.2,
        "close_threshold_base_pct": 0.01,
        "close_threshold_we_weight": 0.0,
        "close_threshold_volatility_1h_weight": 0.0,
        "close_threshold_volatility_1m_weight": 0.0,
        "close_retracement_base_pct": 0.0,
        "close_retracement_volatility_1h_weight": 0.0,
        "close_retracement_volatility_1m_weight": 0.0,
        "entry_cooldown_minutes": 0.0,
        "total_wallet_exposure_limit": 1.0,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
        "forager_volume_ema_span_1m": 60.0,
        "forager_volatility_ema_span_1m": 60.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 2.0,
        "we_excess_allowance_pct": 0.0,
        "hsl_scale_budget_with_excess_allowance": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        **_UNSTUCK_DISABLED_VALUES,
        **_ADAPTIVE_DISABLED_VALUES,
        "unstuck_ema_span_0": 2.0,
        "unstuck_ema_span_1": 3.0,
        **_HSL_DISABLED_VALUES,
        "unstuck_memory_ema_span_minutes": 43200.0,
        "loss_gate_memory_ema_span_minutes": 43200.0,
    }

    def side_row(signal_mode):
        side_values = dict(values)
        side_values["hsl_enabled"] = 1.0
        side_values["hsl_red_threshold"] = 0.9
        side_values["hsl_signal_mode"] = float(signal_mode)
        return [side_values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    unstuck_long = side_row(0)
    unstuck_long[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("unstuck_enabled")] = (
        1.0
    )
    unstuck_short = side_row(0)
    unstuck_short[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("unstuck_enabled")] = (
        1.0
    )
    rows = [
        side_row(0) + side_row(0),
        side_row(1) + side_row(1),
        side_row(2) + side_row(2),
        side_row(0) + side_row(1),
        side_row(3) + side_row(3),
        unstuck_long + side_row(0),
        side_row(0) + unstuck_short,
        unstuck_long + unstuck_short,
    ]
    batch_size = len(rows)
    params = torch.as_tensor(
        np.asarray(rows, dtype=np.float32), device=gpu_device()
    ).contiguous()
    overrides = torch.full(
        (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    run_settings = torch.tensor(
        [
            1_000.0,
            50.0,
            60_000.0,
            0.0,
            0.02,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.001,
            0.0,
        ],
        dtype=torch.float32,
        device=gpu_device(),
    )
    sizes = torch.tensor(
        [
            batch_size,
            data["n"],
            coin_count,
            data["n_days"],
            0,
            10,
            data["start_minute_of_day"],
            data["start_minute_of_hour"],
        ],
        dtype=torch.int32,
        device=gpu_device(),
    )
    end_steps = torch.full(
        (batch_size,), data["n"] - 1, dtype=torch.int32, device=gpu_device()
    )
    daily = torch.zeros(
        (batch_size, data["n_days"], 9), dtype=torch.float32, device=gpu_device()
    )
    daily[:, :, 1].fill_(float("inf"))
    daily[:, :, 5].fill_(float("inf"))
    scalars = torch.zeros(
        (batch_size, MPS_MULTICOIN_FUSED_SCALAR_COLS),
        dtype=torch.float32,
        device=gpu_device(),
    )
    gaps = torch.zeros((batch_size, GAP_BINS), dtype=torch.int32, device=gpu_device())
    coin_fill_counts = torch.zeros(
        (batch_size, coin_count), dtype=torch.float32, device=gpu_device()
    )

    # Match the production history specialization and scratch ABI.
    capacity = min(count + 2, 1440 + 2)
    _, nodes = _hsl_layout(capacity)
    hsl_buffers = (
        torch.empty(
            (batch_size, 2 * (coin_count + 1), nodes, 32),
            dtype=torch.uint8,
            device=gpu_device(),
        ),
        torch.empty(
            (batch_size, 2 * (coin_count + 1), capacity * 2),
            dtype=torch.int32,
            device=gpu_device(),
        ),
    )
    source = "#define PASSIVBOT_HSL_LOOKBACK 1440\n" + _with_hsl(source, capacity)
    source = source.replace(
        "constant int MAX_COINS = 64;", f"constant int MAX_COINS = {coin_count};"
    )
    library = compile_shader(source)
    library.passivbot_trailing_martingale_multicoin_fused(
        data["market_execution_ticks"],
        data["bars"],
        data["fill_ticks"],
        data["touch_ticks"],
        data["touch_nearest_ticks"],
        data["touch_min_qty_bits"],
        data["touch_min_qty_relation"],
        data["hour_log_ranges"],
        data["coin_settings"],
        overrides,
        overrides,
        params,
        run_settings,
        _pack_replay_control(sizes, end_steps, sizes.numel() == 10),
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        *hsl_buffers,
        threads=(batch_size, 1, 1),
    )
    synchronize()
    output = scalars.cpu().numpy()

    assert output[:3, 13].tolist() == [1.0, 1.0, 1.0]
    assert (output[:3, 24] > 0.0).all()
    assert (output[:3, 26] > 0.0).all()
    assert (output[:3, 24] - output[:3, 26] > 0.0).all()
    assert output[:3, 32:34].tolist() == [[1.0, 1.0]] * 3
    assert (output[:3, 38] > 0.0).all()
    assert (output[:3, 64] > 0.0).all()
    assert (output[:3, 65] > 0.0).all()
    assert (output[:3, 66] > 0.0).all()
    assert (output[:3, 67] > 0.0).all()
    np.testing.assert_allclose(output[:3, 18], output[:3, 58] + output[:3, 60])
    np.testing.assert_allclose(output[:3, 19], output[:3, 59] + output[:3, 61])
    np.testing.assert_allclose(output[:3, 24], daily[:3, :, 8].sum(dim=1).cpu().numpy())
    assert (output[:3, 22] <= 1.01).all()
    assert (coin_fill_counts[:3].sum(dim=1).cpu().numpy() > 0.0).all()
    assert output[3:5, 9].tolist() == [0.0, 0.0]
    assert output[3:5, 13].tolist() == [0.0, 0.0]
    assert output[3:5, 24].tolist() == [0.0, 0.0]
    assert output[5:, 9].tolist() == [-1.0, -1.0, -1.0]
    assert output[5:, 13].tolist() == [1.0, 1.0, 1.0]
    assert (output[5:, 24] > 0.0).all()

    override_matrix = np.full(
        (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        runs[0],
        data,
        pnl_lookback_bars=1440,
        long_coin_overrides=override_matrix,
        short_coin_overrides=override_matrix,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=1.0,
        collect_coin_fill_counts=True,
        hsl_ema_tail_enabled=True,
        hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True,
        recovery_distribution_enabled=True,
    )
    torch.testing.assert_close(runner.settings, run_settings)
    for invalid_row in rows[3:5]:
        with pytest.raises(ValueError, match="HSL"):
            runner.run(np.asarray([invalid_row], dtype=np.float64))
    valid_indices = [0, 1, 2, 5, 6, 7]
    runner_output = runner.run(
        np.asarray(rows, dtype=np.float64)[valid_indices], profile=True
    )
    synchronize()
    torch.testing.assert_close(runner_output["day_end_eq"], daily[valid_indices, :, 0])
    torch.testing.assert_close(runner_output["day_min_eq"], daily[valid_indices, :, 1])
    torch.testing.assert_close(runner_output["fill_count"], scalars[valid_indices, 24])
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_long"], scalars[valid_indices, 21]
    )
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_short"], scalars[valid_indices, 57]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_long"], scalars[valid_indices, 55]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_short"], scalars[valid_indices, 56]
    )
    torch.testing.assert_close(
        runner_output["profit_sum_long"], scalars[valid_indices, 58]
    )
    torch.testing.assert_close(
        runner_output["loss_sum_long"], scalars[valid_indices, 59]
    )
    torch.testing.assert_close(
        runner_output["profit_sum_short"], scalars[valid_indices, 60]
    )
    torch.testing.assert_close(
        runner_output["loss_sum_short"], scalars[valid_indices, 61]
    )
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_long"],
        scalars[valid_indices, 62],
    )
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_short"],
        scalars[valid_indices, 63],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_long"],
        scalars[valid_indices, 64],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_short"],
        scalars[valid_indices, 65],
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_long"], scalars[valid_indices, 66]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_short"], scalars[valid_indices, 67]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_portfolio"],
        scalars[valid_indices, 70],
    )
    assert runner_output["alive"].cpu().tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert torch.equal(
        runner_output["coin_fill_counts"], coin_fill_counts[valid_indices]
    )
    recovery = strategy_eq_recovery_distribution_from_samples(
        runner_output["strategy_eq_recovery_samples"],
        sample_interval_days=runner_output["strategy_eq_recovery_sample_interval_days"],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[:, 3] > 0.0).all().item()
    assert runner.last_profile["kernel_seconds"] >= 0.0

    metric_rows = []
    red_threshold_index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "hsl_red_threshold"
    )
    slow_span_index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "hsl_memory_ema_span_minutes"
    )
    for signal_mode in range(3):
        row = side_row(signal_mode) + side_row(signal_mode)
        for side_offset in (
            0,
            len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS),
        ):
            row[side_offset + red_threshold_index] = 0.001
            row[side_offset + slow_span_index] = 43200.0
        metric_rows.append(row)

    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.batch_size = 3
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": data["ts0"], "n": data["n"]}
    proxy.run = runs[0]
    proxy.sides = ["long", "short"]
    proxy.needed_metrics = {
        "hard_stop_panic_close_loss_drawdown_pct_mean",
        "hard_stop_time_in_red_pct",
        "hard_stop_triggers_per_year",
        "fills_per_day",
        "drawdown_worst_ema_strategy_eq",
        "drawdown_worst_ema_strategy_eq_long",
        "drawdown_worst_ema_strategy_eq_short",
        "drawdown_worst_mean_1pct_ema_strategy_eq",
        "drawdown_worst_mean_1pct_ema_strategy_eq_long",
        "drawdown_worst_mean_1pct_ema_strategy_eq_short",
        "drawdown_worst_strategy_eq_long",
        "drawdown_worst_strategy_eq_short",
        "peak_recovery_hours_strategy_eq_long",
        "peak_recovery_hours_strategy_eq_short",
        "peak_recovery_days_strategy_eq_long",
        "peak_recovery_days_strategy_eq_short",
    }
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    width = len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
    proxy.base_params = {
        "long": dict(
            zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, metric_rows[0][:width])
        ),
        "short": dict(
            zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, metric_rows[0][width:])
        ),
    }
    proxy.weighted_equity_metrics = ()
    proxy.fused_runner = runner
    proxy.runners = {}
    candidates = [
        {
            **{
                f"long_{key}": row[index]
                for index, key in enumerate(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
            },
            **{
                f"short_{key}": row[width + index]
                for index, key in enumerate(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
            },
        }
        for row in metric_rows
    ]
    np.testing.assert_allclose(
        proxy._parameter_matrix(candidates, "long"),
        np.asarray(metric_rows, dtype=np.float64)[:, :width],
    )
    np.testing.assert_allclose(
        proxy._parameter_matrix(candidates, "short"),
        np.asarray(metric_rows, dtype=np.float64)[:, width:],
    )
    seen_hsl_triggers = {}

    def reduce_service_output(output, *args, **kwargs):
        seen_hsl_triggers["long"] = output["hsl_triggers_long"].clone()
        seen_hsl_triggers["short"] = output["hsl_triggers_short"].clone()
        seen_hsl_triggers["portfolio_ema"] = output[
            "hsl_drawdown_ema_mean_worst_1pct_portfolio"
        ].clone()
        seen_hsl_triggers["samples"] = output["hsl_tier_samples_total"].clone()
        return compute_objectives(output, *args, **kwargs)

    proxy._compute_objectives = reduce_service_output
    service_results = proxy.evaluate(candidates)
    assert (seen_hsl_triggers["samples"] > 0.0).all()
    assert (seen_hsl_triggers["long"] >= 0.0).all()
    assert (seen_hsl_triggers["short"] >= 0.0).all()
    assert all(item["fills_per_day"] > 0.0 for item in service_results)
    assert all(item["hard_stop_triggers_per_year"] >= 0.0 for item in service_results)
    assert all(item["hard_stop_time_in_red_pct"] >= 0.0 for item in service_results)
    assert all(
        item["hard_stop_panic_close_loss_drawdown_pct_mean"] >= 0.0
        for item in service_results
    )
    assert all(
        item["drawdown_worst_ema_strategy_eq"]
        == max(
            item["drawdown_worst_ema_strategy_eq_long"],
            item["drawdown_worst_ema_strategy_eq_short"],
        )
        for item in service_results
    )
    assert [
        item["drawdown_worst_mean_1pct_ema_strategy_eq"] for item in service_results
    ] == pytest.approx(seen_hsl_triggers["portfolio_ema"].tolist())
    assert (seen_hsl_triggers["portfolio_ema"] > 0.0).all()
    assert all(
        item["drawdown_worst_strategy_eq_long"] > 0.0
        and item["drawdown_worst_strategy_eq_short"] > 0.0
        for item in service_results
    )
    assert all(
        item["peak_recovery_hours_strategy_eq_long"] >= 0.0
        and item["peak_recovery_hours_strategy_eq_short"] >= 0.0
        and item["peak_recovery_days_strategy_eq_long"]
        == pytest.approx(item["peak_recovery_hours_strategy_eq_long"] / 24.0)
        and item["peak_recovery_days_strategy_eq_short"]
        == pytest.approx(item["peak_recovery_hours_strategy_eq_short"] / 24.0)
        for item in service_results
    )

    with pytest.raises(ValueError, match=f"{2 * len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)} columns"):
        runner.run(np.asarray([side_row(0)], dtype=np.float64))
    with pytest.raises(ValueError, match="short override matrix shaped"):
        MpsTrailingMartingaleMulticoinFusedRunner(
            runs[0],
            data,
            pnl_lookback_bars=1440,
            short_coin_overrides=np.empty(
                (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS - 1),
                dtype=np.float32,
            ),
        )
    truncated = runner.run(
        np.asarray([rows[0]], dtype=np.float64),
        end_steps=np.asarray([60], dtype=np.int32),
    )
    synchronize()
    assert truncated["last_eq_ts"].item() <= 59 * 60_000.0
    assert truncated["fill_count"].item() <= output[0, 24]

    override_unstuck = overrides.clone()
    override_unstuck[0, 28] = 1.0
    override_sizes = sizes.clone()
    override_sizes[0] = 1
    override_daily = torch.zeros(
        (1, data["n_days"], 9), dtype=torch.float32, device=gpu_device()
    )
    override_daily[:, :, 1].fill_(float("inf"))
    override_daily[:, :, 5].fill_(float("inf"))
    override_scalars = torch.zeros(
        (1, MPS_MULTICOIN_FUSED_SCALAR_COLS),
        dtype=torch.float32,
        device=gpu_device(),
    )
    override_gaps = torch.zeros((1, GAP_BINS), dtype=torch.int32, device=gpu_device())
    override_coin_fills = torch.zeros(
        (1, coin_count), dtype=torch.float32, device=gpu_device()
    )
    library.passivbot_trailing_martingale_multicoin_fused(
        data["market_execution_ticks"],
        data["bars"],
        data["fill_ticks"],
        data["touch_ticks"],
        data["touch_nearest_ticks"],
        data["touch_min_qty_bits"],
        data["touch_min_qty_relation"],
        data["hour_log_ranges"],
        data["coin_settings"],
        override_unstuck,
        overrides,
        params[:1],
        run_settings,
        _pack_replay_control(override_sizes, end_steps[:1], override_sizes.numel() == 10),
        override_daily,
        override_scalars,
        override_gaps,
        override_coin_fills,
        *hsl_buffers,
        threads=(1, 1, 1),
    )
    synchronize()
    assert override_scalars[0, 9].item() == -1.0
    assert override_scalars[0, 13].item() == 1.0
    assert override_scalars[0, 24].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_trailing_martingale_multicoin_directional_shader_smoke(side):
    import passivbot_rust

    source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    assert "kernel void passivbot_trailing_martingale_multicoin" in source
    assert f"constant int PARAM_COLS = {len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS) + _TM_DECIMAL_WORDS}" in source
    assert "effective_n_positions" in source
    assert "min_since_open" in source
    assert "entry_retracement_base" in source
    assert "close_retracement_base" in source
    assert "as_type<float>(touch_min_qty_bits[k * C + c])" in source
    assert (
        f"constant int OVERRIDE_COLS = {TRAILING_MARTINGALE_COIN_OVERRIDE_COLS + _TM_DECIMAL_WORDS}"
        in source
    )
    assert "allowed_wallet_exposure_limit" in source
    assert "twel_entry_gate_enabled" in source
    assert "coin_override_or" in source
    assert "merge_reducer" not in source
    assert "finalized_reducer_qty" in source
    assert "apply_tm_multicoin_close_admission" in source
    assert "realized_loss_proxy_allows_close" not in source
    assert "recursive_grid_close_would_expand" in source
    assert "close_recursive_mode" in source
    assert "close_gen_market_price" in source
    assert "group.market = should_use_ordinary_market_execution" in source
    assert "const bool loss_gate_enabled = !PASSIVBOT_TM_LOSS_GATE_DISABLED" in source

    count = 512
    coin_count = 3
    phase = np.linspace(0.0, 12.0 * np.pi, count)
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    for coin in range(coin_count):
        close = 100.0 + coin * 20.0 + np.sin(phase + coin) * (6.0 + coin)
        hlcvs[:, coin, 0] = close * 1.015
        hlcvs[:, coin, 1] = close * 0.985
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002) for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            10,
            10,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    assert data["touch_nearest_ticks"].shape == (count, coin_count)
    assert data["touch_nearest_ticks"].dtype == torch.int32
    assert data["touch_min_qty_bits"].shape == (count, coin_count)
    assert data["touch_min_qty_relation"].shape == (count, coin_count)
    values = {
        "ema_span_0": 10.0,
        "ema_span_1": 30.0,
        "volatility_ema_span_1h": 60.0,
        "volatility_ema_span_1m": 60.0,
        "entry_double_down_factor": 1.5,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.1,
        "entry_threshold_base_pct": 0.02,
        "entry_threshold_we_weight": 0.0,
        "entry_threshold_volatility_1h_weight": 0.0,
        "entry_threshold_volatility_1m_weight": 0.0,
        "entry_retracement_base_pct": 0.0,
        "entry_retracement_we_weight": 0.0,
        "entry_retracement_volatility_1h_weight": 0.0,
        "entry_retracement_volatility_1m_weight": 0.0,
        "close_qty_pct": 0.2,
        "close_threshold_base_pct": 0.01,
        "close_threshold_we_weight": 0.0,
        "close_threshold_volatility_1h_weight": 0.0,
        "close_threshold_volatility_1m_weight": 0.0,
        "close_retracement_base_pct": 0.0,
        "close_retracement_volatility_1h_weight": 0.0,
        "close_retracement_volatility_1m_weight": 0.0,
        "entry_cooldown_minutes": 0.0,
        "total_wallet_exposure_limit": 1.0,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
        "forager_volume_ema_span_1m": 60.0,
        "forager_volatility_ema_span_1m": 60.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 2.0,
        "we_excess_allowance_pct": 0.0,
        "hsl_scale_budget_with_excess_allowance": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        **_UNSTUCK_DISABLED_VALUES,
        **_ADAPTIVE_DISABLED_VALUES,
        "unstuck_ema_span_0": 2.0,
        "unstuck_ema_span_1": 3.0,
        **_HSL_DISABLED_VALUES,
        "unstuck_memory_ema_span_minutes": 43200.0,
        "loss_gate_memory_ema_span_minutes": 43200.0,
    }
    row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    runner = MpsTrailingMartingaleMulticoinRunner(
        runs[0],
        data,
        pnl_lookback_bars=1440,
        side=side,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=0.1,
        hsl_raw_drawdown_enabled=True,
        recovery_distribution_enabled=True,
    )
    assert runner.settings.cpu()[5].item() <= 0.1
    output = runner.run(np.array([row, row], dtype=np.float64))
    synchronize()

    assert output["balance"].device.type == gpu_device()
    assert output["balance"].shape == (2,)
    assert runner._buffers[2][1].shape == (2, 65)
    assert (output["hsl_drawdown_raw_max_long"] == 0.0).all()
    assert (output["hsl_drawdown_raw_max_short"] == 0.0).all()
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0
    assert (output["open_positions"] <= 2.0).all()
    recovery = strategy_eq_recovery_distribution_from_samples(
        output["strategy_eq_recovery_samples"],
        sample_interval_days=output["strategy_eq_recovery_sample_interval_days"],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[:, 3] > 0.0).all().item()

    disabled = np.full(
        (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    disabled[:, 24] = 0.0
    disabled_output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, pnl_lookback_bars=1440, side=side, coin_overrides=disabled
    ).run(np.array([row], dtype=np.float64))
    synchronize()
    assert disabled_output["day_has_fill"].sum().item() == 0
    assert disabled_output["open_positions"].item() == 0.0

    exact_last = np.full(
        (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    exact_last[:, :25] = np.asarray(row[:25], dtype=np.float32)
    changed_candidate = list(row)
    changed_candidate[:25] = [
        200.0,
        400.0,
        240.0,
        180.0,
        0.0,
        0.05,
        0.01,
        0.1,
        2.0,
        5.0,
        5.0,
        0.02,
        3.0,
        5.0,
        5.0,
        0.8,
        0.05,
        2.0,
        5.0,
        5.0,
        0.02,
        5.0,
        5.0,
        120.0,
        1.0,
    ]
    exact_last_output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, pnl_lookback_bars=1440, side=side, coin_overrides=exact_last
    ).run(np.array([row, changed_candidate], dtype=np.float64))
    synchronize()
    assert torch.equal(
        exact_last_output["day_has_fill"][0],
        exact_last_output["day_has_fill"][1],
    )
    assert torch.equal(
        exact_last_output["day_end_eq"][0],
        exact_last_output["day_end_eq"][1],
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_bounded_allowance_with_gate_disabled_expands_volume(
    strategy_kind, side
):
    runner, baseline = _multicoin_exposure_fixture(strategy_kind, side)
    expanded = list(baseline)
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    expanded[keys.index("we_excess_allowance_pct")] = 0.5
    expanded[keys.index("hsl_scale_budget_with_excess_allowance")] = 1.0
    expanded[keys.index("twel_entry_gate_enabled")] = 0.0

    output = runner.run(np.asarray([baseline, expanded], dtype=np.float64))
    synchronize()
    volume = output["day_volume"].sum(dim=1).cpu().numpy()

    assert volume[0] > 0.0
    assert volume[1] > volume[0] * 1.1


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_multicoin_coin_override_allowance_expands_one_symbol(strategy_kind):
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    allowance_column = 12 if strategy_kind == "ema_anchor" else 25
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[0, allowance_column] = 1.0
    baseline_runner, baseline = _multicoin_exposure_fixture(strategy_kind, "long")
    override_runner, overridden = _multicoin_exposure_fixture(
        strategy_kind, "long", overrides
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    baseline[keys.index("twel_entry_gate_enabled")] = 0.0
    overridden[keys.index("twel_entry_gate_enabled")] = 0.0

    baseline_output = baseline_runner.run(np.asarray([baseline], dtype=np.float64))
    override_output = override_runner.run(np.asarray([overridden], dtype=np.float64))
    synchronize()

    baseline_volume = baseline_output["day_volume"].sum().item()
    override_volume = override_output["day_volume"].sum().item()
    assert baseline_volume > 0.0
    assert override_volume > baseline_volume * 1.1


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_multicoin_twel_threshold_reduces_entry_volume(strategy_kind):
    runner, full_cap = _multicoin_exposure_fixture(strategy_kind, "long")
    reduced_cap = list(full_cap)
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    reduced_cap[keys.index("twel_enforcer_threshold")] = 0.5

    output = runner.run(np.asarray([full_cap, reduced_cap], dtype=np.float64))
    synchronize()
    volume = output["day_volume"].sum(dim=1).cpu().numpy()

    assert volume[0] > 0.0
    assert 0.0 < volume[1] < volume[0] * 0.75


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_multicoin_equal_distance_twel_tie_keeps_higher_coin_index(
    strategy_kind,
):
    override_cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    wallet_exposure_column = 11 if strategy_kind == "ema_anchor" else 24
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[0, wallet_exposure_column] = 0.4
    overrides[1, wallet_exposure_column] = 0.5
    markets = [ProxyMarket(0.001, 0.01, 0.001, 300.0, 1.0, 0.0) for _ in range(2)]
    runner, candidate = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        overrides,
        count=6,
        markets=markets,
        closes=(100.0, 100.0),
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    candidate[keys.index("twel_enforcer_threshold")] = 0.5

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()
    volume = output["day_volume"].sum().item()

    # Exact Rust removes equal-distance entries by ascending symbol index.
    # That leaves the higher-index coin's 0.5-WEL order. Keeping index zero
    # instead would admit only its 0.4-WEL order because the residual is dust.
    # EMA Anchor also closes the selected flat-price order in this short fixture,
    # so it reports the matching entry and exit volume.
    expected_volume = 1.0 if strategy_kind == "ema_anchor" else 0.5
    assert expected_volume - 0.05 < volume < expected_volume + 0.05


def _tm_row(*, initial_ema_dist=0.01, gate_initial=1.0, gate_reentry=1.0):
    return [
        2.0,  # ema_span_0
        3.0,  # ema_span_1
        2.0,  # volatility_ema_span_1h
        2.0,  # volatility_ema_span_1m
        1.0,  # entry_double_down_factor
        initial_ema_dist,
        0.1,  # entry_initial_qty_pct
        0.01,  # entry_threshold_base_pct
        0.0,  # entry_threshold_we_weight
        0.0,  # entry_threshold_volatility_1h_weight
        0.0,  # entry_threshold_volatility_1m_weight
        0.001,  # entry_retracement_base_pct
        0.0,  # entry_retracement_we_weight
        0.0,  # entry_retracement_volatility_1h_weight
        0.0,  # entry_retracement_volatility_1m_weight
        1.0,  # close_qty_pct
        0.01,  # close_threshold_base_pct
        0.0,  # close_threshold_we_weight
        0.0,  # close_threshold_volatility_1h_weight
        0.0,  # close_threshold_volatility_1m_weight
        0.001,  # close_retracement_base_pct
        0.0,  # close_retracement_volatility_1h_weight
        0.0,  # close_retracement_volatility_1m_weight
        0.0,  # entry_cooldown_minutes
        1.0,  # total_wallet_exposure_limit
        gate_initial,
        gate_reentry,
    ]


def _tm_parameter_row():
    values = dict.fromkeys(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, 0.0)
    values.update(dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, _tm_row())))
    values.update(_UNSTUCK_DISABLED_VALUES)
    values.update(_HSL_DISABLED_VALUES)
    values.update(_ADAPTIVE_DISABLED_VALUES)
    values.update(unstuck_ema_span_0=2.0, unstuck_ema_span_1=3.0,
                  unstuck_memory_ema_span_minutes=43200.0,
                  loss_gate_memory_ema_span_minutes=43200.0)
    return [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]


def test_tm_loss_gate_specialization_is_explicit_and_guarded():
    source = (ROOT / "passivbot-rust/src/gpu/mps_trailing_martingale_multicoin.metal").read_text()
    assert _with_tm_loss_gate_disabled(source, False) == source
    assert _with_tm_loss_gate_disabled(source, True).startswith("#define PASSIVBOT_TM_LOSS_GATE_DISABLED 1\n")
    with pytest.raises(RuntimeError, match="loss-gate specialization"):
        _with_tm_loss_gate_disabled("", True)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize(
    "btc_prices",
    [
        np.array([100.0, 101.0]),
        np.array([100.0, 0.0, 102.0]),
        np.array([100.0, np.nan, 102.0]),
        np.array([100.0, np.inf, 102.0]),
        np.array([100.0, 1.0e-50, 102.0]),
    ],
)
def test_mps_btc_risk_prices_fail_closed_after_float32_packing(btc_prices):
    from optimization.gpu.mps_kernel import _btc_risk_price_tensor

    with pytest.raises(ValueError, match="MPS BTC-risk prices"):
        _btc_risk_price_tensor(btc_prices, expected_count=3)
























@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_side_multicoin_auto_unstuck_selects_only_one_coin(
    strategy_kind, side
):
    count = 6
    closes = np.full((count, 2), 100.0)
    if side == "long":
        closes[3:, 0] = 97.5
        closes[3:, 1] = 98.5
        highs = closes + 0.01
        lows = closes.copy()
    else:
        closes[3:, 0] = 102.5
        closes[3:, 1] = 101.5
        highs = closes.copy()
        lows = closes - 0.01
    markets = [
        ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0),
        ProxyMarket(1.0, 0.01, 2.0, 0.0, 1.0, 0.0),
    ]
    runner, disabled = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.05,
        market_orders_allowed=strategy_kind == "ema_anchor",
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    disabled[keys.index("entry_cooldown_minutes")] = 100.0
    disabled[keys.index("twel_entry_gate_enabled")] = 0.0
    if strategy_kind == "ema_anchor":
        disabled[keys.index("offset")] = 0.01
    else:
        disabled[keys.index("entry_initial_ema_dist")] = 0.01
        disabled[keys.index("entry_initial_qty_pct")] = 1.0
        disabled[keys.index("entry_threshold_base_pct")] = 10.0
        disabled[keys.index("entry_retracement_base_pct")] = 0.0
        disabled[keys.index("close_threshold_base_pct")] = 0.5
        disabled[keys.index("close_retracement_base_pct")] = 0.0
        disabled[keys.index("gate_initial")] = 1.0
    enabled = list(disabled)
    enabled[keys.index("unstuck_enabled")] = 1.0
    enabled[keys.index("unstuck_ema_gating_enabled")] = 0.0
    enabled[keys.index("unstuck_close_pct")] = 0.1
    enabled[keys.index("unstuck_loss_allowance_pct")] = 0.2
    enabled[keys.index("unstuck_threshold")] = 0.5
    gated = list(enabled)
    gated[keys.index("unstuck_ema_gating_enabled")] = 1.0
    gated[keys.index("unstuck_ema_dist")] = 0.1
    allowance_disabled = list(disabled)
    allowance_disabled[keys.index("we_excess_allowance_pct")] = 1.0
    allowance_disabled[keys.index("twel_entry_gate_enabled")] = 1.0
    allowance_disabled[keys.index("twel_enforcer_threshold")] = 0.5
    allowance_enabled = list(enabled)
    allowance_enabled[keys.index("we_excess_allowance_pct")] = 1.0
    allowance_enabled[keys.index("twel_entry_gate_enabled")] = 1.0
    allowance_enabled[keys.index("twel_enforcer_threshold")] = 0.5

    output = runner.run(
        np.asarray(
            [
                disabled,
                enabled,
                gated,
                allowance_disabled,
                allowance_enabled,
            ],
            dtype=np.float64,
        )
    )
    synchronize()
    key = "psize" if side == "long" else "short_psize"
    remaining = output[key].cpu().numpy()
    open_positions = output["open_positions"].cpu().tolist()
    assert open_positions[:3] == [2.0] * 3
    assert remaining[0] > 1.0
    # Coin one is less stuck and has a two-unit minimum. Selecting coin zero,
    # or selecting both coins, would produce a different aggregate remainder.
    assert remaining[1] == pytest.approx(remaining[0] - 2.0)
    assert remaining[2] == pytest.approx(remaining[0])
    # The bounded allowance doubles each 0.5 per-coin WEL to 1.0, while the
    # side-wide entry gate caps exposure at 0.5. Exact Rust evaluates unstuck's
    # threshold against the allowed limit, so the position does not exceed it.
    assert open_positions[4] == open_positions[3]
    assert remaining[4] == pytest.approx(remaining[3])

    strict_runner, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.0005,
        market_orders_allowed=strategy_kind == "ema_anchor",
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
    )
    strict_output = strict_runner.run(np.asarray([enabled], dtype=np.float64))
    synchronize()
    assert strict_output[key].item() == pytest.approx(remaining[0])


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_unstuck_finalization_keeps_ordinary_close_remainder():
    count = 7
    closes = np.full((count, 2), 100.0)
    highs = np.full((count, 2), 100.0)
    lows = np.full((count, 2), 100.0)
    lows[3, 0] = 98.0
    highs[4, 0] = 201.0
    markets = [
        ProxyMarket(0.5, 0.01, 0.5, 500.0, 1.0, 0.0),
        ProxyMarket(0.5, 0.01, 0.5, 0.0, 1.0, 0.0),
    ]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        coin_overrides=overrides,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.05,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_retracement_base_pct": 0.0,
            "close_qty_pct": 0.3,
            "close_threshold_base_pct": 1.0,
            "close_retracement_base_pct": 0.01,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.2,
            "unstuck_enabled": 1.0,
            "unstuck_ema_gating_enabled": 0.0,
            "unstuck_close_pct": 0.7,
            "unstuck_loss_allowance_pct": 0.2,
            "unstuck_threshold": 0.5,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    # As in the directional kernel regression above, the valid ordinary close
    # keeps unstuck's finalized selection key below the larger WEL reducer.
    assert output["psize"].item() < 3.0


def _tm_multicoin_off_tick_reducer_case(
    side,
    *,
    maker_fee,
    taker_fee=0.0,
    price_step=1.0,
    close_qty_pct=1.0,
    close_threshold_base=0.0,
    close_threshold_we=0.0,
    next_high=101.5,
    next_low=99.5,
    market_orders_allowed=False,
    market_order_near_touch_threshold=0.02,
    market_order_slippage_pct=0.0,
    generation_touch=None,
    next_close=100.0,
    min_cost=0.0,
    collect_coin_fill_counts=False,
):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    if generation_touch is None:
        generation_touch = 100.4 if side == "long" else 100.6
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 102.0, next_high, 100.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 98.0, next_low, 100.0]
    hlcvs[:, 0, 2] = [
        100.0,
        100.0,
        100.0,
        generation_touch,
        next_close,
        100.0,
    ]
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(0.001, price_step, 0.001, min_cost, 1.0, maker_fee, taker_fee)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(hlcvs, timestamps, [run, run], [market, market],
                                    market_order_slippage_pct=market_order_slippage_pct)
    _, row = _multicoin_exposure_fixture("trailing_martingale", side, count=count)
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_retracement_base_pct": 0.0,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
            "close_qty_pct": close_qty_pct,
            "close_threshold_base_pct": close_threshold_base,
            "close_threshold_we_weight": close_threshold_we,
            "close_retracement_base_pct": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    runner = MpsTrailingMartingaleMulticoinRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        side=side,
        coin_overrides=overrides,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=market_order_near_touch_threshold,
        market_order_slippage_pct=market_order_slippage_pct,
        collect_coin_fill_counts=collect_coin_fill_counts,
    )
    return runner, candidate, market


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_market_wel_keeps_passive_grid_reservation():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_market_wel_reservation_probe(
    constant float* bars,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool short_side = short_side_raw != 0;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideState side;
    init_trailing_martingale_multicoin_side_state(
        side, config, coin_settings, coin_overrides, 1
    );
    side.selected[0] = true;
    side.psize[0] = 10.0f;
    side.pprice[0] = 100.0f;
    side.position_open_k[0] = 0.0f;
    side.position_last_fill_k[0] = 0.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    generate_tm_multicoin_side_orders(
        side, config, account,
        bars, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides,
        1, 1, short_side, 1, 1, 0, false, 1.0f,
        true, 1.1f, 0.0f, -2, 0ul
    );
    output[0] = side.close_qty[0];
    output[1] = side.close_grid_gen_psize[0];
    output[2] = side.close_market[0] ? 1.0f : 0.0f;
    output[3] = side.close_is_exposure_reducer[0] ? 1.0f : 0.0f;
    output[4] = float(side.close_grid_max_rungs[0]);
}
"""
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", "long", count=3, closes=(100.0, 100.0)
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_qty_pct": 0.0,
            "close_threshold_base_pct": 10.0,
            "close_threshold_we_weight": 0.0,
            "close_retracement_base_pct": 0.0,
            "n_positions": 1.0,
            "total_wallet_exposure_limit": 1.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.99,
        }
    )
    params = _tm_probe_tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device=gpu_device(),
    )
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[100.0, 100.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device=gpu_device(),
    )
    touch_ticks = torch.full((2, 1, 2), 20_000, dtype=torch.int32, device=gpu_device())
    touch_nearest_ticks = torch.full(
        (2, 1), 20_000, dtype=torch.int32, device=gpu_device()
    )
    touch_min_qty_bits = torch.zeros((2, 1), dtype=torch.int32, device=gpu_device())
    touch_min_qty_relation = torch.zeros(
        (2, 1),
        dtype=torch.int8 if gpu_device() == "cuda" else torch.int32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 3] = 100.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = _tm_probe_full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(5, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_market_wel_reservation_probe(
        bars,
        touch_ticks,
        touch_nearest_ticks,
        touch_min_qty_bits,
        touch_min_qty_relation,
        coin_settings,
        coin_overrides,
        params,
        output,
        0,
        threads=(1, 1, 1),
    )
    synchronize()

    # The passive strategy request is 0.5 at price 200, while executable-touch
    # min-cost sizing at market price 100 promotes the emitted reducer to 1.0.
    # The immutable ladder must still start from 10.0 - 0.5, not 10.0 - 1.0.
    assert output.cpu().tolist() == pytest.approx(
        [1.0, 9.5, 1.0, 1.0, 499.0], abs=2.0e-4
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_market_unstuck_sizes_at_touch_without_reseeding_grid(
    side,
):
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_market_unstuck_reservation_probe(
    constant float* bars,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* output,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool short_side = short_side_raw != 0;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideState state;
    init_trailing_martingale_multicoin_side_state(
        state, config, coin_settings, coin_overrides, 1
    );
    state.selected[0] = true;
    state.psize[0] = 10.0f;
    state.pprice[0] = 100.0f;
    state.position_open_k[0] = 0.0f;
    state.position_last_fill_k[0] = 0.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    generate_tm_multicoin_side_orders(
        state, config, account,
        bars, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides,
        1, 1, short_side, 1, 1, 0, false, 1.0f,
        true, 1.1f, 0.0f, 0, 0ul
    );
    output[0] = state.close_qty[0];
    output[1] = state.close_grid_gen_psize[0];
    output[2] = state.close_market[0] ? 1.0f : 0.0f;
    output[3] = state.close_is_exposure_reducer[0] ? 1.0f : 0.0f;
    output[4] = state.close_is_unstuck_reducer[0] ? 1.0f : 0.0f;
    output[5] = state.close_gen_market_price[0];
    output[6] = float(state.close_grid_max_rungs[0]);
}
"""
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=3, closes=(100.0, 100.0)
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_qty_pct": 0.0,
            "close_threshold_base_pct": 10.0,
            "close_threshold_we_weight": 0.0,
            "close_retracement_base_pct": 0.0,
            "n_positions": 1.0,
            "total_wallet_exposure_limit": 1.0,
            "wel_enforcer_enabled": 0.0,
            "twel_enforcer_enabled": 0.0,
            "unstuck_enabled": 1.0,
            "unstuck_ema_gating_enabled": 0.0,
            "unstuck_close_pct": 0.1,
            "unstuck_loss_allowance_pct": 0.2,
            "unstuck_threshold": 0.5,
        }
    )
    params = _tm_probe_tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device=gpu_device(),
    )
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[100.0, 100.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device=gpu_device(),
    )
    touch_ticks = torch.full((2, 1, 2), 20_000, dtype=torch.int32, device=gpu_device())
    touch_nearest_ticks = torch.full(
        (2, 1), 20_000, dtype=torch.int32, device=gpu_device()
    )
    touch_min_qty_bits = torch.zeros((2, 1), dtype=torch.int32, device=gpu_device())
    touch_min_qty_relation = torch.zeros(
        (2, 1),
        dtype=torch.int8 if gpu_device() == "cuda" else torch.int32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 3] = 100.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = _tm_probe_full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    output = torch.zeros(7, dtype=torch.float32, device=gpu_device())
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_market_unstuck_reservation_probe(
        bars,
        touch_ticks,
        touch_nearest_ticks,
        touch_min_qty_bits,
        touch_min_qty_relation,
        coin_settings,
        coin_overrides,
        params,
        output,
        int(side == "short"),
        threads=(1, 1, 1),
    )
    synchronize()

    # The passive unstuck request is 0.5 at price 200. Market policy enlarges
    # the emitted reducer to the executable-touch minimum of 1.0 at 100, but
    # external unstuck does not reseed Rust's full-size strategy ladder.
    assert output.cpu().tolist() == pytest.approx(
        [1.0, 10.0, 1.0, 1.0, 1.0, 100.0, 500.0], abs=2.0e-4
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_market_unstuck_override_pays_slippage_and_taker_fee(
    side,
):
    count = 6
    closes = np.full((count, 2), 100.0)
    if side == "long":
        closes[3:, 0] = 97.5
        closes[3:, 1] = 98.5
        highs = closes + 0.01
        lows = closes.copy()
    else:
        closes[3:, 0] = 102.5
        closes[3:, 1] = 101.5
        highs = closes.copy()
        lows = closes - 0.01

    def run_case(*, slippage, taker_fee, max_realized_loss_pct=1.0):
        markets = [
            ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0, taker_fee),
            ProxyMarket(1.0, 0.01, 2.0, 0.0, 1.0, 0.0, taker_fee),
        ]
        overrides = np.full(
            (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
        )
        # Exercise the effective per-coin runtime: the side template remains
        # disabled while coin one alone enables auto-unstuck.
        overrides[1, 28] = 1.0
        runner, candidate = _multicoin_exposure_fixture(
            "trailing_martingale",
            side,
            coin_overrides=overrides,
            count=count,
            markets=markets,
            closes=closes,
            highs=highs,
            lows=lows,
            market_orders_allowed=True,
            market_order_near_touch_threshold=0.001,
            market_order_slippage_pct=slippage,
            max_realized_loss_pct=max_realized_loss_pct,
        )
        values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
        values.update(
            {
                "entry_initial_ema_dist": 0.01,
                "entry_initial_qty_pct": 1.0,
                "entry_threshold_base_pct": 10.0,
                "entry_retracement_base_pct": 0.0,
                "close_threshold_base_pct": 0.5,
                "close_retracement_base_pct": 0.0,
                "entry_cooldown_minutes": 100.0,
                "gate_initial": 1.0,
                "unstuck_enabled": 0.0,
                "unstuck_ema_gating_enabled": 0.0,
                "unstuck_close_pct": 0.1,
                "unstuck_loss_allowance_pct": 0.2,
                "unstuck_threshold": 0.5,
            }
        )
        output = runner.run(
            np.asarray(
                [[values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]],
                dtype=np.float64,
            )
        )
        synchronize()
        return output

    no_cost = run_case(slippage=0.0, taker_fee=0.0)
    slipped = run_case(slippage=0.01, taker_fee=0.0)
    charged = run_case(slippage=0.01, taker_fee=0.002)
    strict = run_case(slippage=0.01, taker_fee=0.002, max_realized_loss_pct=0.0)
    position_key = "psize" if side == "long" else "short_psize"

    assert no_cost["fill_count"].item() > 2.0
    assert slipped["fill_count"].item() == no_cost["fill_count"].item()
    assert charged["fill_count"].item() == slipped["fill_count"].item()
    assert slipped[position_key].item() == pytest.approx(
        no_cost[position_key].item(), abs=2.0e-4
    )
    assert charged[position_key].item() == pytest.approx(
        slipped[position_key].item(), abs=2.0e-4
    )
    assert strict[position_key].item() > charged[position_key].item()
    assert slipped["balance"].item() < no_cost["balance"].item()
    assert charged["balance"].item() < slipped["balance"].item()
    assert strict["balance"].item() >= charged["balance"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_market_reducer_keeps_generation_dust_allocation():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_market_reducer_dust_probe(
    constant int* market_execution_ticks,
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    device float* coin_fill_counts,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideState side;
    init_trailing_martingale_multicoin_side_state(
        side, config, coin_settings, coin_overrides, 1
    );
    side.psize[0] = 1.6f;
    side.pprice[0] = 40.0f;
    side.position_open_k[0] = 0.0f;
    side.position_last_fill_k[0] = 0.0f;
    side.entry_qty[0] = 0.0f;
    side.close_qty[0] = 1.0f;
    side.close_tick[0] = 1000;
    side.close_market[0] = true;
    side.close_is_exposure_reducer[0] = true;
    side.close_reconstruct_after_reducer[0] = true;
    side.close_recursive_mode[0] = true;
    side.close_gen_balance[0] = 1000.0f;
    side.close_gen_allowed_wel[0] = 0.016f;
    side.close_gen_market_price[0] = 100.0f;
    side.close_grid_gen_psize[0] = 0.4f;
    side.close_grid_max_rungs[0] = 1;
    CloseGroup probe_group;
    TmCloseGridContext probe_context = {
        false, 0.4f, 40.0f, 1000.0f, 0.016f,
        390, 400, 400, 0.0f, 0,
        config.close_qty_pct, config.close_threshold_base,
        config.close_threshold_we, config.close_threshold_v1h,
        config.close_threshold_v1m, 0.0f, 0.0f,
        0.1f, 0.1f, 0.1f, 40.0f, 1.0f, 0, 0.0f, 1,
        100.0f, true, 0.001f, 0.4f
    };
    int probe_group_count = recursive_grid_close_groups_after_reducer(
        probe_context, 0, probe_group
    );
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    float hsl_equity_before_fills = 1000.0f;
    process_tm_multicoin_side_fills(
        market_execution_ticks,
        side, config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 1, 1, false, true, false, false,
        1.0f, true, 0.0f, 0.001f, false, hsl_equity_before_fills
    );
    output[0] = side.psize[0];
    output[1] = fills.fill_count;
    output[2] = float(probe_group_count);
    output[3] = probe_group.qty;
    output[4] = probe_group.price;
    output[5] = min_entry_qty(
        probe_group.price, 0.1f, 0.1f, 40.0f, 1.0f
    );
    output[6] = probe_group.market ? 1.0f : 0.0f;
}
"""
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", "long", count=3, closes=(100.0, 100.0)
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 2.75,
            "close_threshold_we_weight": 0.0,
            "close_retracement_base_pct": 0.0,
        }
    )
    params = _tm_probe_tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device=gpu_device(),
    )
    fill_ticks = torch.tensor(
        [[[0, 0]], [[2200, 0]]], dtype=torch.int32, device=gpu_device()
    )
    touch_ticks = torch.tensor(
        [[[390, 400]], [[390, 400]]], dtype=torch.int32, device=gpu_device()
    )
    touch_nearest_ticks = torch.full(
        (2, 1), 400, dtype=torch.int32, device=gpu_device()
    )
    touch_min_qty_bits = torch.zeros((2, 1), dtype=torch.int32, device=gpu_device())
    touch_min_qty_relation = torch.zeros(
        (2, 1),
        dtype=torch.int8 if gpu_device() == "cuda" else torch.int32,
        device=gpu_device(),
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device=gpu_device())
    coin_settings[0, 0] = 0.1
    coin_settings[0, 1] = 0.1
    coin_settings[0, 2] = 0.1
    coin_settings[0, 3] = 40.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = _tm_probe_full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device=gpu_device(),
    )
    library = compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )

    remaining_sizes = []
    for next_close in (100.0, 200.0):
        bars = torch.tensor(
            [
                [[100.0, 40.0, 100.0, 1.0]],
                [[220.0, 30.0, next_close, 1.0]],
            ],
            dtype=torch.float32,
            device=gpu_device(),
        )
        coin_fill_counts = torch.zeros(1, dtype=torch.float32, device=gpu_device())
        output = torch.zeros(7, dtype=torch.float32, device=gpu_device())
        library.passivbot_tm_multicoin_market_reducer_dust_probe(
            _probe_market_execution_ticks(bars, [0.1]),
            bars,
            fill_ticks,
            touch_ticks,
            touch_nearest_ticks,
            touch_min_qty_bits,
            touch_min_qty_relation,
            coin_settings,
            coin_overrides,
            params,
            coin_fill_counts,
            output,
            threads=(1, 1, 1),
        )
        synchronize()
        remaining_sizes.append(output.cpu().tolist())

    # The passive group keeps 0.4 at price 150 and leaves a 0.2 remainder.
    # At the generation touch of 100 the minimum across reducer and group is
    # 0.3, so the remainder is allocated to that group. A later market fill at
    # 200 lowers the reducer minimum to 0.2, but cannot retroactively revoke
    # the already emitted dust allocation.
    np.testing.assert_allclose(
        np.asarray(remaining_sizes)[:, :2],
        [[0.0, 2.0], [0.0, 2.0]],
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        np.asarray(remaining_sizes)[:, 2:],
        [[1.0, 0.4, 150.0, 0.3, 0.0]] * 2,
        atol=2.0e-4,
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_off_tick_grid_precedes_reducer_for_volume(side):
    runner, candidate, market = _tm_multicoin_off_tick_reducer_case(
        side, maker_fee=0.001
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    # Multicoin's shared total-entry headroom floors the short by one step.
    entry_qty = 10.101 if side == "long" else 9.9
    grid_price = 100.0 if side == "long" else 101.0
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_price = 101.0 if side == "long" else 100.0
    reducer_qty = entry_qty - grid_qty
    balance = 1_000.0 - entry_qty * entry_price * market.maker_fee
    expected_volume = entry_qty * entry_price / balance
    grid_pnl = grid_qty * (
        grid_price - entry_price if side == "long" else entry_price - grid_price
    )
    balance += grid_pnl - grid_qty * grid_price * market.maker_fee
    expected_volume += grid_qty * grid_price / balance
    reducer_pnl = reducer_qty * (
        reducer_price - entry_price if side == "long" else entry_price - reducer_price
    )
    balance += reducer_pnl - reducer_qty * reducer_price * market.maker_fee
    expected_volume += reducer_qty * reducer_price / balance

    assert output["balance"].item() == pytest.approx(balance, abs=3.0e-4)
    assert output["profit_sum"].item() == pytest.approx(
        grid_pnl + reducer_pnl, abs=3.0e-4
    )
    assert output["loss_sum"].item() == 0.0
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_market_grid_precedes_farther_reducer_for_volume(
    side,
):
    runner, candidate, market = _tm_multicoin_off_tick_reducer_case(
        side,
        maker_fee=0.001,
        taker_fee=0.002,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.006,
        market_order_slippage_pct=0.01,
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.9
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_qty = entry_qty - grid_qty
    market_fill_price = 99.0 if side == "long" else 101.0
    balance = 1_000.0 - entry_qty * entry_price * market.maker_fee
    expected_volume = entry_qty * entry_price / balance
    grid_pnl = grid_qty * (
        market_fill_price - entry_price
        if side == "long"
        else entry_price - market_fill_price
    )
    balance += grid_pnl - grid_qty * market_fill_price * market.taker_fee
    expected_volume += grid_qty * market_fill_price / balance
    reducer_pnl = reducer_qty * (
        market_fill_price - entry_price
        if side == "long"
        else entry_price - market_fill_price
    )
    balance += reducer_pnl - reducer_qty * market_fill_price * market.taker_fee
    expected_volume += reducer_qty * market_fill_price / balance

    assert output["balance"].item() == pytest.approx(balance, abs=3.0e-4)
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_same_tick_wel_resizes_merged_market_group_once(side):
    runner, candidate, _ = _tm_multicoin_off_tick_reducer_case(
        side,
        maker_fee=0.0,
        price_step=1.0,
        close_qty_pct=0.1,
        close_threshold_base=0.01,
        close_threshold_we=-0.02,
        next_high=99.0 if side == "long" else 101.0,
        next_low=99.0 if side == "long" else 101.0,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.005,
        generation_touch=99.0 if side == "long" else 101.0,
        next_close=99.0 if side == "long" else 101.0,
        min_cost=50.0,
        collect_coin_fill_counts=True,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values["wel_enforcer_threshold"] = 0.99
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Exact calc_closes_* merges the below-executable-minimum WEL prefix with
    # the first same-tick recursive grid group before market sizing. The
    # completed group therefore executes once, after the initial entry.
    assert output["coin_fill_counts"][0, 0].item() == 2.0
    assert output["fill_count"].item() == 2.0
    assert output[size_key].item() > 0.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_early_market_reducer_charges_taker_fee(side):
    common = {
        "side": side,
        "maker_fee": 0.0,
        "price_step": 0.1,
        "close_threshold_base": 0.02,
        "next_high": 102.5,
        "next_low": 97.5,
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.005,
    }
    free_runner, candidate, _ = _tm_multicoin_off_tick_reducer_case(
        taker_fee=0.0, **common
    )
    charged_runner, _, _ = _tm_multicoin_off_tick_reducer_case(
        taker_fee=0.002, **common
    )

    free = free_runner.run(np.asarray([candidate], dtype=np.float64))
    charged = charged_runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert charged[size_key].item() == pytest.approx(free[size_key].item(), abs=2.0e-4)
    # Only the near-touch WEL reducer is promoted; the farther recursive
    # groups stay passive.  The reducer sorts before the first group, so a
    # lower balance here specifically proves that early path used taker_fee.
    assert charged["balance"].item() < free["balance"].item() - 0.01


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_reducer_consumes_one_recursive_close_slot(side):
    runner, candidate, _ = _tm_multicoin_off_tick_reducer_case(
        side,
        maker_fee=0.0,
        close_qty_pct=0.0001,
        close_threshold_we=1.0e-6,
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    expected_size = 4.052 if side == "long" else 4.451
    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(expected_size, abs=2.0e-4)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_global_position_exposure_repair(side):
    runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=10
    )
    repaired = list(baseline)
    repaired[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")] = (
        1.0
    )
    repaired[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
    ] = 0.5
    output = runner.run(np.asarray([baseline, repaired], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert 0.0 < sizes[1] < sizes[0]
    assert output["day_volume"][1].sum().item() > output["day_volume"][0].sum().item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("repair_kind", ["position", "total"])
def test_mps_tm_multicoin_market_exposure_repair_uses_slippage_and_taker_fee(
    side, repair_kind
):
    market_no_fee = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0, 0.0) for _ in range(2)
    ]
    market_with_taker_fee = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0, 0.002) for _ in range(2)
    ]
    passive_runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=10,
        markets=market_no_fee,
    )
    promoted_runner, _ = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=10,
        markets=market_no_fee,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
        market_order_slippage_pct=0.01,
    )
    fee_runner, _ = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=10,
        markets=market_with_taker_fee,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
        market_order_slippage_pct=0.01,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "wel_enforcer_threshold": 0.5,
            "twel_enforcer_threshold": 0.5,
            "wel_enforcer_enabled": float(repair_kind == "position"),
            "twel_enforcer_enabled": float(repair_kind == "total"),
            "twel_enforcer_reduce_portfolio": float(repair_kind == "total"),
        }
    )
    candidate = np.asarray(
        [[values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]],
        dtype=np.float64,
    )

    passive = passive_runner.run(candidate)
    promoted = promoted_runner.run(candidate)
    charged = fee_runner.run(candidate)
    synchronize()

    # The promoted protective reducer crosses immediately at an adverse
    # directional price; adding a nonzero taker fee then lowers balance again.
    assert promoted["loss_sum"].item() > passive["loss_sum"].item() + 1.0
    assert promoted["balance"].item() < passive["balance"].item() - 1.0
    assert charged["balance"].item() < promoted["balance"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_trailing_market_exposure_repair_stays_partial(side):
    baseline_runner, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=10
    )
    promoted_runner, _ = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=10,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "close_threshold_base_pct": 10.0,
            "close_retracement_base_pct": 0.01,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = np.asarray(
        [[values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]],
        dtype=np.float64,
    )

    baseline = baseline_runner.run(candidate)
    promoted = promoted_runner.run(candidate)
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # A trailing strategy does not otherwise populate the recursive close
    # snapshot. The promoted WEL reducer must retain its generation touch so
    # the next-candle allocator cannot interpret zero as an infinite minimum
    # and absorb the whole remaining position.
    assert baseline[size_key].item() > 0.0
    assert promoted[size_key].item() == pytest.approx(
        baseline[size_key].item(), abs=2.0e-4
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize(
    ("strategy_kind", "repair_kind"),
    [
        ("ema_anchor", "total"),
        ("trailing_martingale", "total"),
        ("trailing_martingale", "position"),
    ],
)
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_fused_dual_multicoin_exposure_repair(
    strategy_kind, repair_kind, market_orders_allowed
):
    fixture_kwargs = {"count": 10}
    if strategy_kind == "ema_anchor":
        count = 70
        highs = np.full((count, 2), 100.5)
        lows = np.full((count, 2), 99.5)
        highs[62, :] = 102.0
        lows[62, :] = 98.0
        fixture_kwargs = {
            "count": count,
            "closes": (100.0, 100.0),
            "highs": highs,
            "lows": lows,
        }
    _, baseline, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        return_context=True,
        market_order_slippage_pct=0.01,
        **fixture_kwargs,
    )
    if strategy_kind == "ema_anchor":
        runner = MpsEmaAnchorMulticoinFusedRunner(
            run,
            data,
            pnl_lookback_bars=1440,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=0.002,
            market_order_slippage_pct=0.01,
        )
        param_keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    else:
        runner = MpsTrailingMartingaleMulticoinFusedRunner(
            run,
            data,
            pnl_lookback_bars=1440,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=0.002,
            market_order_slippage_pct=0.01,
        )
        param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS

    baseline = list(baseline)
    baseline[param_keys.index("entry_cooldown_minutes")] = 100.0
    if strategy_kind == "ema_anchor":
        baseline[param_keys.index("offset")] = 0.01
    else:
        baseline[param_keys.index("entry_initial_ema_dist")] = 0.01
    repaired = list(baseline)
    if repair_kind == "total":
        repaired[param_keys.index("twel_entry_gate_enabled")] = 0.0
        repaired[param_keys.index("twel_enforcer_threshold")] = 0.5
        repaired[param_keys.index("twel_enforcer_enabled")] = 1.0
        repaired[param_keys.index("twel_enforcer_reduce_portfolio")] = 1.0
    else:
        repaired[param_keys.index("wel_enforcer_enabled")] = 1.0
        repaired[param_keys.index("wel_enforcer_threshold")] = 0.5

    output = runner.run(
        np.asarray([baseline + baseline, repaired + repaired], dtype=np.float64)
    )
    synchronize()

    assert output["alive"].cpu().tolist() == [True, True]
    assert output["psize"][0].item() > 0.0
    assert output["short_psize"][0].item() > 0.0
    assert output["psize"][1].item() < output["psize"][0].item()
    assert output["short_psize"][1].item() < output["short_psize"][0].item()
    assert output["fill_count"][1].item() > output["fill_count"][0].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("repair_kind", ["position", "total"])
def test_mps_tm_fused_one_way_multicoin_market_exposure_repair(repair_kind):
    _, baseline, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=10,
        return_context=True,
        market_order_slippage_pct=0.01,
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        pnl_lookback_bars=1440,
        hedge_mode=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
        market_order_slippage_pct=0.01,
    )
    baseline = list(baseline)
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_ema_dist")
    ] = 0.01
    repaired = list(baseline)
    if repair_kind == "position":
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")
        ] = 1.0
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
        ] = 0.5
    else:
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_entry_gate_enabled")
        ] = 0.0
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
        ] = 0.5
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
        ] = 1.0
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "twel_enforcer_reduce_portfolio"
            )
        ] = 1.0

    output = runner.run(
        np.asarray([baseline + baseline, repaired + repaired], dtype=np.float64)
    )
    synchronize()

    baseline_size = output["psize"][0].item() + output["short_psize"][0].item()
    repaired_size = output["psize"][1].item() + output["short_psize"][1].item()
    assert (output["psize"][0].item() > 0.0) != (output["short_psize"][0].item() > 0.0)
    assert 0.0 < repaired_size < baseline_size
    assert output["fill_count"][1].item() > output["fill_count"][0].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_total_exposure_repair_policy(side):
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[0, 24] = 0.1
    runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        overrides,
        count=10,
        closes=(100.0, 100.0),
    )
    overweight = list(baseline)
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_entry_gate_enabled")
    ] = 0.0
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
    ] = 0.5
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
    ] = 1.0
    portfolio = list(overweight)
    portfolio[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_reduce_portfolio")
    ] = 1.0

    output = runner.run(np.asarray([overweight, portfolio], dtype=np.float64))
    synchronize()

    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 1.0])
    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = (output[pprice_key] / output["balance"]).cpu()
    assert (total_twe < 0.5).all()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_loss_gate_blocks_lossy_total_exposure_repair(
    side, market_orders_allowed
):
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[0, 24] = 0.1
    runner_kwargs = {
        "strategy_kind": "trailing_martingale",
        "side": side,
        "coin_overrides": overrides,
        "count": 10,
        "closes": (100.0, 100.0),
        "market_orders_allowed": market_orders_allowed,
        "market_order_near_touch_threshold": 1.1,
        "market_order_slippage_pct": 0.01,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.0, **runner_kwargs
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() < gated[size_key].item()
    assert gated["open_positions"].item() == pytest.approx(2.0)
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
@pytest.mark.parametrize("max_loss_pct", [0.0, 0.1])
def test_mps_tm_multicoin_fee_only_close_uses_loss_allowance(
    side, market_orders_allowed, max_loss_pct
):
    markets = [
        ProxyMarket(
            0.001,
            0.01,
            0.001,
            0.0,
            1.0,
            0.0 if market_orders_allowed else 0.001,
            0.001,
        )
        for _ in range(2)
    ]
    runner_kwargs = {
        "strategy_kind": "trailing_martingale",
        "side": side,
        "count": 10,
        "closes": (100.0, 100.0),
        "markets": markets,
        "market_orders_allowed": market_orders_allowed,
        "market_order_near_touch_threshold": 1.1,
        "market_order_slippage_pct": 0.001,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=max_loss_pct, **runner_kwargs
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 0.0,
            "close_threshold_we_weight": 0.0,
            "entry_cooldown_minutes": 100.0,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() == pytest.approx(0.0)
    if max_loss_pct == 0.0:
        assert gated[size_key].item() > 0.0
        assert gated["balance"].item() >= ungated["balance"].item()
    else:
        assert gated[size_key].item() == pytest.approx(0.0)
        assert gated["balance"].item() == pytest.approx(ungated["balance"].item(), abs=0.002)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_preserves_profitable_close_after_repair(side):
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[0, 24] = 0.1
    count = 10
    blocked_highs = np.full((count, 2), 100.5)
    blocked_lows = np.full((count, 2), 99.5)
    recovery_highs = blocked_highs.copy()
    recovery_lows = blocked_lows.copy()
    if side == "long":
        recovery_highs[7:, :] = 102.0
    else:
        recovery_lows[7:, :] = 98.0
    common = {
        "strategy_kind": "trailing_martingale",
        "side": side,
        "coin_overrides": overrides,
        "count": count,
        "closes": (100.0, 100.0),
        "max_realized_loss_pct": 0.0,
    }
    blocked_runner, candidate = _multicoin_exposure_fixture(
        highs=blocked_highs, lows=blocked_lows, **common
    )
    recovery_runner, _ = _multicoin_exposure_fixture(
        highs=recovery_highs, lows=recovery_lows, **common
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 0.01,
            "close_threshold_we_weight": 0.0,
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    blocked = blocked_runner.run(np.asarray([candidate], dtype=np.float64))
    recovered = recovery_runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert blocked[size_key].item() > 0.0
    assert recovered[size_key].item() == pytest.approx(0.0)
    assert recovered["balance"].item() > blocked["balance"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_scans_past_blocked_recursive_rung(side):
    count = 10
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        highs[5:, 0] = 102.0
    else:
        lows[5:, 0] = 98.0
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=(100.0, 100.0),
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.0,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "close_qty_pct": 0.25,
            "close_threshold_base_pct": 0.015,
            "close_threshold_we_weight": -0.025,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The first immutable recursive rungs are below break-even, but later
    # rungs become profitable as their simulated exposure falls. Exact Rust
    # filters the early rungs independently and retains those later closes.
    assert output[size_key].item() < 5.0
    assert output["balance"].item() > 1_000.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_does_not_reallocate_blocked_twel(side):
    count = 10
    closes = np.full((count, 2), 100.0)
    if side == "long":
        closes[5:, 1] = 102.0
    else:
        closes[5:, 1] = 98.0
    highs = closes * 1.01
    lows = closes * 0.99
    markets = [ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001) for _ in range(2)]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[0, 24] = 0.6
    overrides[1, 24] = 0.4
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        max_realized_loss_pct=0.0,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    baseline = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    repaired_values = dict(values, twel_enforcer_enabled=1.0)
    repaired = [
        repaired_values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([baseline, repaired], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Coin zero wins TWEL's equal-adverse tie and alone reaches the target, but
    # its fee-only reducer is loss-gated. Exact Rust does not then reallocate
    # that action to profitable coin one.
    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 2.0])
    assert output[size_key][1].item() == pytest.approx(
        output[size_key][0].item(), abs=2.0e-4
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_total_exposure_repair_policy(side):
    overrides = np.full((2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[0, 11] = 0.1
    count = 70
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        lows[62, :] = 98.0
    else:
        highs[62, :] = 102.0
    runner, baseline = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        overrides,
        count=count,
        closes=(100.0, 100.0),
        highs=highs,
        lows=lows,
    )
    overweight = list(baseline)
    overweight[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.01
    overweight[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = 100.0
    overweight[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_entry_gate_enabled")] = 0.0
    overweight[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")] = 0.5
    overweight[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")] = 1.0
    portfolio = list(overweight)
    portfolio[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_reduce_portfolio")
    ] = 1.0

    output = runner.run(np.asarray([overweight, portfolio], dtype=np.float64))
    synchronize()

    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 1.0])
    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = (output[pprice_key] / output["balance"]).cpu()
    assert (total_twe < 0.5).all()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_loss_gate_blocks_lossy_total_exposure_repair(side):
    overrides = np.full((2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[0, 11] = 0.1
    count = 70
    closes = np.full((count, 2), 100.0)
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        lows[62, :] = 98.0
        closes[63:, :] = 98.0
        highs[63:, :] = 98.0
        lows[63:, :] = 97.9
    else:
        highs[62, :] = 102.0
        closes[63:, :] = 102.0
        highs[63:, :] = 102.1
        lows[63:, :] = 102.0

    runner_kwargs = {
        "strategy_kind": "ema_anchor",
        "side": side,
        "coin_overrides": overrides,
        "count": count,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.001,
        "market_order_slippage_pct": 0.01,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.001, **runner_kwargs
    )
    values = dict(zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "offset": 0.01,
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    candidate = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() < gated[size_key].item()
    # Both paths keep the same symbol open, but the shared loss allowance
    # blocks part of the lossy repair and therefore retains more quantity.
    assert ungated["open_positions"].item() == pytest.approx(1.0)
    assert gated["open_positions"].item() == pytest.approx(1.0)
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_loss_gate_blocks_fee_only_ordinary_close(side):
    markets = [ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001) for _ in range(2)]
    runner_kwargs = {
        "strategy_kind": "ema_anchor",
        "side": side,
        "count": 70,
        "closes": (100.0, 100.0),
        "markets": markets,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.0, **runner_kwargs
    )
    candidate[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = 100.0

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() == pytest.approx(0.0)
    assert gated[size_key].item() > 0.0
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_rejects_held_tails_before_twel_repair(side):
    count = 7
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 101.0
    hlcvs[:, :, 1] = 99.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    # A still-held coin tail cannot be valued or hidden by a TWEL repair.
    hlcvs[3, 0, :3] = [111.0, 89.0, 110.0]
    hlcvs[4:, 0, :] = 0.0
    # The still-tradable coin is adverse for long and favorable for short.
    hlcvs[4:, 1, :3] = [91.0, 89.0, 90.0]
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            last_valid,
        )
        for last_valid in (3, count - 1)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, [market, market])
    _, row = _multicoin_exposure_fixture("trailing_martingale", side, count=count)
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.75,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    repaired = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    baseline = list(repaired)
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
    ] = 0.0
    with pytest.raises(ValueError, match="held-position valuation"):
        MpsTrailingMartingaleMulticoinRunner(
            runs[1], data, pnl_lookback_bars=1440, side=side
        ).run(np.asarray([baseline, repaired], dtype=np.float64))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_rejects_held_tails_before_twel_repair(side):
    count = 70
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 101.0
    hlcvs[:, :, 1] = 99.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    # Fill wide-offset entries, then lose coin zero coverage while coin one
    # remains tradable. Both baseline and repair must reject unavailable valuation.
    hlcvs[62, 0, :3] = [140.0, 70.0, 130.0]
    hlcvs[62, 1, :3] = [130.0, 70.0, 90.0]
    hlcvs[63:, 0, :] = 0.0
    hlcvs[63:, 1, :3] = [101.0, 89.0, 90.0]
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            last_valid,
        )
        for last_valid in (62, count - 1)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, [market, market])
    _, row = _multicoin_exposure_fixture("ema_anchor", side, count=count)
    values = dict(zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "offset": 0.2,
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.75,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    repaired = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]
    baseline = list(repaired)
    baseline[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")] = 0.0
    with pytest.raises(ValueError, match="held-position valuation"):
        MpsEmaAnchorMulticoinRunner(
            runs[1], data, pnl_lookback_bars=1440, side=side
        ).run(np.asarray([baseline, repaired], dtype=np.float64))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_twel_repair_preserves_trailing_closes(side):
    highs = np.full((7, 2), 100.0)
    lows = np.full((7, 2), 100.0)
    highs[3, :] = 105.0
    lows[3, :] = 95.0
    highs[5, :] = 105.0
    lows[5, :] = 95.0
    if side == "long":
        highs[4, :] = 99.9
        lows[4, :] = 98.0
        closes = np.full((7, 2), 100.0)
        closes[4, :] = 99.0
    else:
        highs[4, :] = 102.0
        lows[4, :] = 100.1
        closes = np.full((7, 2), 100.0)
        closes[4, :] = 101.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=7,
        closes=closes,
        highs=highs,
        lows=lows,
    )
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_qty_pct")] = 0.2
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_base_pct")] = (
        0.0
    )
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_retracement_base_pct")
    ] = 0.001
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")] = (
        100.0
    )
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_entry_gate_enabled")] = 0.0
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")] = 0.8
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")] = 1.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_reduce_portfolio")
    ] = 1.0

    output = runner.run(np.asarray([row], dtype=np.float64))
    synchronize()

    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = output[pprice_key][0].item() / output["balance"][0].item()
    # Two ordinary trailing closes remove 0.2 TWE in total and TWEL repair
    # independently removes another 0.2.  Suppressing one trailing close
    # leaves approximately 0.7 instead of 0.6.
    assert total_twe < 0.65


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_reducer_retains_reachable_post_repair_grid(side):
    runner, row = _multicoin_exposure_fixture("trailing_martingale", side, count=6)
    enabled = list(row)
    enabled[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")] = (
        1.0
    )
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
    ] = 0.5
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    reducer_only = list(enabled)
    reducer_only[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_retracement_base_pct")
    ] = 0.001
    repaired_grid = list(enabled)
    repaired_grid[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_base_pct")
    ] = 0.0
    repaired_grid[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_we_weight")
    ] = 0.0
    output = runner.run(np.asarray([reducer_only, repaired_grid], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    volumes = output["day_volume"].sum(dim=1).cpu().numpy()
    assert sizes[0] > 0.0
    assert sizes[1] == pytest.approx(0.0)
    assert volumes[1] > volumes[0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_grid_can_fill_without_off_tick_reducer(side):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    generation_touch = 100.4 if side == "long" else 100.6
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 102.0, 100.5, 100.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 98.0, 100.5, 100.0]
    hlcvs[:, 0, 2] = [100.0, 100.0, 100.0, generation_touch, 100.0, 100.0]
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(hlcvs, timestamps, [run, run], [market, market])
    _, row = _multicoin_exposure_fixture("trailing_martingale", side, count=count)
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
            "close_threshold_base_pct": 0.0,
            "close_threshold_we_weight": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, pnl_lookback_bars=1440, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(5.0, abs=0.1)
    assert 1.0 < output["day_volume"].sum().item() < 2.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_small_excess_reducer_crosses_strict_target(side):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 101.0, 101.0, 101.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 99.0, 99.0, 99.0]
    hlcvs[:, 0, 2] = 100.0
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(1.0e-8, 1.0, 1.0e-8, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(hlcvs, timestamps, [run, run], [market, market])
    _, row = _multicoin_exposure_fixture("trailing_martingale", side, count=count)
    target = 0.99999
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "n_positions": 1.0,
            "close_retracement_base_pct": 0.001,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": target,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, pnl_lookback_bars=1440, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    exposure = output[size_key].item() * 100.0 / output["balance"].item()
    assert output[size_key].item() > 0.0
    assert exposure < target


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_tm_multicoin_short_post_repair_grid_uses_negative_target_tick():
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 102.5, 103.0, 103.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 99.0, 101.5, 101.5]
    hlcvs[:, 0, 2] = [100.0, 100.0, 100.0, 102.0, 102.0, 102.0]
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(hlcvs, timestamps, [run, run], [market, market])
    _, row = _multicoin_exposure_fixture("trailing_martingale", "short", count=count)
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "n_positions": 1.0,
            "close_threshold_base_pct": -0.01,
            "close_threshold_we_weight": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, pnl_lookback_bars=1440, side="short", coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    # The reducer at the 102 bid fills, while Rust's 101 target grid order
    # remains below the next candle's 101.5 low and must not fill.
    assert output["short_psize"].item() == pytest.approx(4.999, abs=0.002)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_post_repair_grid_volume_uses_per_fill_balance(side):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=6,
        markets=markets,
        closes=(100.0, 100.0),
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "close_threshold_base_pct": 0.009,
            "close_threshold_we_weight": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    first_reducer_qty = 2.501
    second_reducer_qty = 2.5
    grid_qty = 2.499
    reducer_price = 100.0
    grid_price = 100.9 if side == "long" else 99.1
    grid_pnl = grid_qty * abs(grid_price - 100.0)
    balance_0 = 1_000.0
    balance_1 = balance_0 + grid_pnl
    balance_2 = balance_1 + grid_pnl
    # Total-entry headroom clips the second exact-0.5-WEL entry by one step.
    expected_volume = 0.5 + 0.4999
    expected_volume += first_reducer_qty * reducer_price / balance_0
    expected_volume += grid_qty * grid_price / balance_1
    expected_volume += second_reducer_qty * reducer_price / balance_1
    expected_volume += grid_qty * grid_price / balance_2

    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_same_tick_twel_and_grid_use_separate_denominators(side):
    runner, candidate, market = _tm_multicoin_off_tick_reducer_case(
        side, maker_fee=0.001
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "wel_enforcer_enabled": 0.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.901
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_qty = entry_qty - grid_qty
    close_price = 100.0 if side == "long" else 101.0
    balance_after_entry = 1_000.0 - entry_qty * entry_price * market.maker_fee
    expected_volume = entry_qty * entry_price / balance_after_entry
    grid_pnl = grid_qty * (
        close_price - entry_price if side == "long" else entry_price - close_price
    )
    balance_after_grid = (
        balance_after_entry + grid_pnl - grid_qty * close_price * market.maker_fee
    )
    expected_volume += grid_qty * close_price / balance_after_grid
    reducer_pnl = reducer_qty * (
        close_price - entry_price if side == "long" else entry_price - close_price
    )
    balance_after_close = (
        balance_after_grid + reducer_pnl - reducer_qty * close_price * market.maker_fee
    )
    expected_volume += reducer_qty * close_price / balance_after_close
    assert output["balance"].item() == pytest.approx(balance_after_close, abs=3.0e-4)
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_finalized_reducer_tie_keeps_nearer_wel(side):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.empty((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 105.0, 100.0, 100.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 95.0, 100.0, 100.0]
    hlcvs[:, 0, 2] = 100.0
    hlcvs[:, 0, 3] = 200.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    markets = [
        ProxyMarket(0.001, 0.01, 9.0, 0.0, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 9.0, 0.0, 1.0, 0.0),
    ]
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(hlcvs, timestamps, [run, run], markets)
    _, row = _multicoin_exposure_fixture("trailing_martingale", side, count=count)
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_retracement_base_pct": 0.0,
            "entry_cooldown_minutes": 100.0,
            "close_threshold_base_pct": 10.0,
            "close_retracement_base_pct": 0.0,
            "n_positions": 1.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.05,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.1,
            "twel_enforcer_enabled": 1.0,
        }
    )
    candidate = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
    twel_only_values = dict(values)
    twel_only_values["wel_enforcer_enabled"] = 0.0
    twel_only = [
        twel_only_values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, pnl_lookback_bars=1440, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate, twel_only], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    assert sizes[1] == pytest.approx(0.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_static_override_repairs_only_selected_symbol(
    side, market_orders_allowed
):
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    overrides[0, 26] = 1.0
    overrides[0, 27] = 0.5
    baseline_runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=10,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.002,
    )
    override_runner, overridden = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        overrides,
        count=10,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.002,
    )
    baseline_output = baseline_runner.run(np.asarray([baseline], dtype=np.float64))
    override_output = override_runner.run(np.asarray([overridden], dtype=np.float64))
    synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert 0.0 < override_output[size_key].item() < baseline_output[size_key].item()
    assert (
        override_output["day_volume"].sum().item()
        > baseline_output["day_volume"].sum().item()
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_trailing_martingale_multicoin_drops_subminimum_final_touch_close(
    side,
):
    count = 6
    coin_count = 2
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = 100.004
    hlcvs[:, 0, 1] = 100.004
    if side == "long":
        hlcvs[3, 0, 1] = 99.0
        hlcvs[4, 0, 0] = 101.0
    else:
        hlcvs[3, 0, 0] = 101.0
        hlcvs[4, 0, 1] = 99.0
    hlcvs[:, 0, 2] = 100.004
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0:3] = 200.0
    hlcvs[:, 1, 3] = 1.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0),
    ]
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    row = _tm_row(gate_initial=0.0, gate_reentry=0.0)
    row[15] = 0.01
    row[16] = 0.199
    row[17] = -2.0
    row[20] = 0.0
    row.extend(
        [
            60.0,  # forager volume span
            60.0,  # forager volatility span
            1.0,  # volume score weight
            0.0,  # EMA readiness score weight
            0.0,  # volatility score weight
            1.0,  # n_positions
        ]
    )
    row.extend(_single_coin_exposure_fields())
    row.extend(_tm_wel_enforcer_fields())
    row.append(0.0)  # TWEL enforcer disabled
    row.append(0.0)  # TWEL reduce_overweight policy
    row.extend(_UNSTUCK_DISABLED_VALUES.values())
    row.extend(_HSL_DISABLED_VALUES.values())
    row.extend([2.0, 3.0])
    row.extend(_ADAPTIVE_DISABLED_VALUES.values())
    row.extend([43200.0, 43200.0])

    output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, pnl_lookback_bars=1440, side=side
    ).run(np.array([row], dtype=np.float64))
    synchronize()

    # Raw-touch sizing proposes 0.100, but its finalized limit price is 100.00:
    # notional 10.0 is below 10.0004. Rust's final close normalization drops
    # this leg; the position is not itself dust, so the full-position exception
    # does not apply. Sizing at the raw touch does not bypass final admission.
    position_key = "psize" if side == "long" else "short_psize"
    assert 0.1 * 100.0 < markets[0].min_cost
    assert output[position_key].item() == pytest.approx(1.0, abs=1e-6)
    assert output["fill_count"].item() == 1


@pytest.mark.parametrize("topology", ["multicoin", "fused"])
@pytest.mark.parametrize("tail_columns", [0, 2, 4, 6])
def test_holding_duration_scalar_does_not_alias_optional_hsl_outputs(
    topology, tail_columns
):
    from optimization.gpu.mps_kernel import (
        _decode_outputs,
        _decode_multicoin_fused_outputs,
    )

    base_columns = 62 if topology == "multicoin" else 67
    daily_columns = 9
    daily = torch.zeros((1, 1, daily_columns), dtype=torch.float32)
    scalars = torch.zeros((1, base_columns + tail_columns), dtype=torch.float32)
    scalars[:, -2] = 123.0
    scalars[:, -1] = 456.0
    decoder = {
        "multicoin": _decode_outputs,
        "fused": _decode_multicoin_fused_outputs,
    }[topology]
    output = decoder(daily, scalars, torch.zeros((1, 128), dtype=torch.int32))
    assert output["held_sum_squared_hours"].item() == 123.0
    assert output["gap_sum_squared_hours"].item() == 456.0
    for key in output:
        if key.startswith("hsl_drawdown_"):
            assert output[key].item() == 0.0, key


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["multicoin", "fused"])
@pytest.mark.parametrize("interval_minutes", [1, 5])
def test_streamed_gap_moment_matches_distinct_fill_timestamps(
    strategy_kind, topology, interval_minutes
):
    count = 20
    multi, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        count=count,
        return_context=True,
        interval_minutes=interval_minutes,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    values = dict(zip(keys, row))
    if strategy_kind == "trailing_martingale":
        values.update(close_threshold_base_pct=0.0, close_retracement_base_pct=0.0)
    values["total_wallet_exposure_limit"] = 0.5
    row = [values[key] for key in keys]
    if topology == "multicoin":
        runner, matrix = multi, np.asarray([row])
    elif topology == "fused":
        cls = (
            MpsEmaAnchorMulticoinFusedRunner
            if strategy_kind == "ema_anchor"
            else MpsTrailingMartingaleMulticoinFusedRunner
        )
        runner, matrix = cls(run, data, pnl_lookback_bars=0), np.asarray([row + row])
    fill_timestamps = set()
    for end in range(3, count):
        output = runner.run(matrix, end_steps=np.array([end], dtype=np.int32))
        timestamp = output["last_fill_ts"].item()
        if np.isfinite(timestamp):
            fill_timestamps.add(timestamp)
    gaps = np.diff(sorted(fill_timestamps)) / 3_600_000.0
    assert len(gaps) >= 2, "fixture must include multiple coalesced fill gaps"
    assert output["gap_hist"].sum().item() == len(gaps)
    assert output["gap_sum_squared_hours"].item() == pytest.approx(
        float(np.square(gaps).sum()), rel=2e-6
    )
    boundaries = [
        output["first_eq_ts"].item(),
        *sorted(fill_timestamps),
        output["last_eq_ts"].item(),
    ]
    all_gaps = np.diff(boundaries) / 3_600_000.0
    expected = np.square(all_gaps).sum() / all_gaps.sum()
    metrics = _fill_gap_metrics(
        {key: value.cpu() for key, value in output.items()}, run
    )
    assert metrics["fills_gap_time_weighted_mean_hours"].item() == pytest.approx(
        expected, rel=2e-6
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize(
    "source_name",
    [
        "mps_ema_anchor_multicoin_source_py",
        "mps_trailing_martingale_multicoin_source_py",
    ],
)
@pytest.mark.parametrize("spans", [(2.5, 91.75), (0.1, 0.9), (100000.5, 2.25)])
def test_mps_independent_unstuck_band_matches_adjusted_recurrence(source_name, spans):
    import passivbot_rust

    closes = np.array([100.0, 97.0, 93.0, 98.0, 104.0, 91.0], dtype=np.float32)
    probe = r"""
kernel void unstuck_band_probe(constant float* inputs, device float* output,
                              uint b [[thread_position_in_grid]]) {
    if (b > 0) return;
    UnstuckEmaBand band = init_unstuck_ema_band(inputs[0], inputs[1], inputs[2]);
    for (int i = 0; i < 6; ++i) {
        if (i > 0) update_unstuck_ema_band(band, inputs[2 + i]);
        output[i * 2] = unstuck_ema_lower(band);
        output[i * 2 + 1] = unstuck_ema_upper(band);
    }
}
"""
    inputs = torch.tensor([*spans, *closes], dtype=torch.float32, device=gpu_device())
    output = torch.zeros(12, dtype=torch.float32, device=gpu_device())
    library = compile_shader(getattr(passivbot_rust, source_name)() + probe)
    library.unstuck_band_probe(inputs, output, threads=(1, 1, 1))
    expected = []
    # Use the Rust price EMA reference, including its price/1 initialization.
    # Backtests clamp sub-candle spans to alpha=1 (equivalent to span=1).
    for i in range(len(closes)):
        values = [
            passivbot_rust.ema_last(closes[: i + 1].astype(np.float64), max(1.0, span))
            for span in (*spans, np.sqrt(spans[0] * spans[1]))
        ]
        expected.extend([min(values), max(values)])
    np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=2e-6, atol=2e-5)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("interval_minutes", [1, 5])
def test_mps_independent_unstuck_horizons_change_reducer_only(
    strategy_kind, side, interval_minutes
):
    count = 6
    closes = np.full((count, 2), 100.0)
    closes[3:] = 98.0 if side == "long" else 102.0
    runner, row, run, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        highs=closes + 0.01,
        lows=closes - 0.01,
        interval_minutes=interval_minutes,
        return_context=True,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    values = dict(zip(keys, row))
    values.update(
        entry_cooldown_minutes=1000.0,
        unstuck_enabled=1.0,
        unstuck_ema_gating_enabled=1.0,
        unstuck_close_pct=0.1,
        unstuck_loss_allowance_pct=0.2,
        unstuck_threshold=0.5,
        unstuck_ema_dist=-0.01,
        twel_entry_gate_enabled=0.0,
    )
    if strategy_kind == "ema_anchor":
        values["offset"] = 0.01
    else:
        values.update(
            entry_initial_ema_dist=0.01, gate_initial=1.0, close_threshold_base_pct=0.5
        )
    rows = []
    for enabled, span in [
        (True, 0.5),
        (True, 100000.5),
        (False, 0.5),
        (False, 100000.5),
    ]:
        values.update(
            unstuck_enabled=float(enabled),
            unstuck_ema_span_0=span,
            unstuck_ema_span_1=span,
        )
        packed = [values[key] for key in keys]
        rows.append(packed)
    output = runner.run(np.asarray(rows, dtype=np.float64))
    positions = output["psize" if side == "long" else "short_psize"].cpu().numpy()
    assert abs(positions[0]) < abs(positions[1])
    assert positions[1] == positions[2] == positions[3]
    # Disabled unstuck horizons cannot change strategy entries, closes or metrics.
    for result in output.values():
        np.testing.assert_array_equal(result[2].cpu().numpy(), result[3].cpu().numpy())
    cols = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    overrides = np.full((2, cols), np.nan)
    ema_start = (EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN
                 if strategy_kind == "ema_anchor"
                 else TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN)
    overrides[:, ema_start:ema_start + 2] = 0.5  # Pin both unstuck spans.
    pinned_runner, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        closes=closes,
        highs=closes + 0.01,
        lows=closes - 0.01,
        interval_minutes=interval_minutes,
        coin_overrides=overrides,
    )
    pinned = pinned_runner.run(np.asarray([rows[1]], dtype=np.float64))
    assert pinned[
        "psize" if side == "long" else "short_psize"
    ].item() == pytest.approx(positions[0])


@pytest.mark.parametrize("span", [0.0, -1.0, float("nan"), float("inf"), 1e100, 1e-100])
def test_unstuck_spans_fail_closed_if_not_representable_in_packed_candles(span):
    keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    row = _tm_parameter_row()
    row[keys.index("unstuck_ema_span_0")] = span
    with pytest.raises(ValueError, match="unstuck EMA spans"):
        _scale_directional_minute_parameters(
            np.asarray([row * 2]), keys, sides=2, interval_minutes=5
        )


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_independent_unstuck_coin_spans_scale_without_changing_strategy(strategy):
    from optimization.gpu import model, mps_kernel

    if strategy == "ema_anchor":
        cols = model.EMA_ANCHOR_COIN_OVERRIDE_COLS
        start = model.EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN
        scale = mps_kernel._scale_ema_multicoin_coin_overrides
    else:
        cols = model.TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
        start = model.TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN
        scale = mps_kernel._scale_tm_multicoin_coin_overrides
    matrix = np.full((2, cols), np.nan)
    matrix[0, start : start + 2] = [17.25, 211.75]
    actual = scale(matrix, 5)
    np.testing.assert_allclose(actual[0, start : start + 2], [3.45, 42.35])
    assert np.isnan(actual[0, :start]).all()
    assert np.isnan(actual[1]).all()
    matrix[0, start] = float("inf")
    with pytest.raises(ValueError, match="unstuck EMA spans"):
        scale(matrix, 5)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("side", ["long", "short"])
def test_tm_multicoin_scenario_batch_matches_separate_candidate_batches(side):
    count = 503
    steps = np.arange(count)
    closes = np.column_stack(
        [100 * (1 + 0.1 * np.sin(steps / 17 + c)) for c in range(2)]
    )
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count, closes=closes
    )
    position_column = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("n_positions")
    qty_column = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_qty_pct")
    matrices, separate = [], []
    for slots in (1, 2):
        matrix = np.asarray([row] * 3, dtype=np.float64)
        matrix[:, position_column] = slots
        matrix[:, qty_column] = [0.1, 0.5, 1.0]
        matrices.append(matrix)
        separate.append(
            {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in runner.run(matrix).items()
            }
        )
    together = runner.run(np.concatenate(matrices))
    assert together["fill_count"].sum().item() > 0
    for key, value in together.items():
        if isinstance(value, torch.Tensor):
            expected = torch.cat([output[key] for output in separate])
            torch.testing.assert_close(
                value.cpu(), expected, rtol=0, atol=0, equal_nan=True
            )
        else:
            assert value == separate[0][key] == separate[1][key]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_mps_gate_only_realized_memory_clips_fills_and_queries_are_pure():
    import passivbot_rust

    probe = r"""
kernel void shared_realized_memory_probe(device float* output,
    uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    configure_joint_realized_memory(account, 3.0f, 3.0f, false, true, 1);
    record_joint_portfolio_fill(account, -80.0f, true);
    record_joint_portfolio_fill(account, 100.0f, false);
    record_joint_portfolio_fill(account, -20.0f, true);
    output[0] = account.loss_gate_memory.drawdown;
    advance_joint_realized_memory(account, 1);
    output[1] = account.loss_gate_memory.drawdown;
    float allowance = 0.0f;
    output[2] = float(joint_realized_loss_allowance(account, 0.1f, allowance));
    output[3] = allowance;
    joint_realized_loss_allowance(account, 0.1f, allowance);
    output[4] = account.loss_gate_memory.drawdown;
    account.loss_gate_memory.enabled = false;
    record_joint_portfolio_fill(account, -5.0f, false);
    output[5] = account.loss_gate_memory.drawdown;
    output[6] = account.balance;
}
"""
    output = torch.zeros(7, dtype=torch.float32, device=gpu_device())
    source = "#define PASSIVBOT_UNSTUCK_ENABLED 0\n" + passivbot_rust.mps_ema_anchor_multicoin_source_py()
    library = compile_shader(source + probe)
    library.shared_realized_memory_probe(output, threads=(1, 1, 1))
    synchronize()
    # The first loss is fully repaid before the final loss: summing this batch
    # before clipping would incorrectly report zero drawdown.
    assert output.cpu().tolist() == [20.0, 10.0, 1.0, 91.0, 10.0, 10.0, 995.0]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_multicoin_excess_allowance_changes_coin_hsl_signal(strategy_kind, side):
    closes = np.tile(np.asarray([100.0, 120.0]), (64, 1))
    closes[20:, 0] *= 0.7 if side == "long" else 1.3
    columns = (
        EMA_ANCHOR_COIN_OVERRIDE_COLS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    )
    overrides = np.full((2, columns), np.nan)
    wel_column, allowance_column = (
        (11, 12) if strategy_kind == "ema_anchor" else (24, 25)
    )
    overrides[:, wel_column] = 0.2
    overrides[:, allowance_column] = 1.0
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, overrides, closes=closes, collect_coin_fill_counts=True
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.2,
        "hsl_drawdown_ratio_ema_span_minutes": 1.0,
        'hsl_cooldown_ms_lo': 0.0,
        'hsl_memory_ema_span_minutes': 43200.0,
        "hsl_signal_mode": 2.0,
        "we_excess_allowance_pct": 1.0,
    }.items():
        row[keys.index(key)] = value
    scaled = list(row)
    scaled[keys.index("hsl_scale_budget_with_excess_allowance")] = 1.0
    out = runner.run(np.asarray([row, scaled], dtype=np.float64))
    synchronize()
    assert out[f"hsl_triggers_{side}"][0].item() > 0
    assert out[f"hsl_triggers_{side}"][1].item() == 0
    assert (out["coin_fill_counts"] >= 1).all().item()


@pytest.mark.parametrize("profile", [False, True])
def test_multicoin_pipeline_lookup_is_profiled_only_when_enabled(monkeypatch, profile):
    from optimization.gpu import mps_kernel

    runner = mps_kernel._MulticoinReplayRunner.__new__(mps_kernel._MulticoinReplayRunner)
    runner.replay_kernel_name = "replay"
    runner._profile_pipeline = profile
    runner._pipeline_seconds = 0.0
    runner._temporal_replay_enabled = lambda: False
    calls = []
    clock = iter([10.0, 13.0])
    def tick():
        assert profile, "disabled profiling must not read the clock"
        return next(clock)
    class Library:
        @property
        def replay(self):
            def kernel(*args, **kwargs):
                calls.append((args, kwargs))
            kernel.max_threads_per_threadgroup = 1024
            kernel.thread_execution_width = 32
            return kernel
    monkeypatch.setattr(mps_kernel.time, "perf_counter", tick)
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "mps")
    runner._dispatch_replay(Library(), ("input",), None, batch_size=2, scalars=None)
    assert runner._pipeline_seconds == (3.0 if profile else 0.0)
    assert calls == [(("input",), {"threads": (2, 1, 1), "group_size": (2, 1, 1)})]


@pytest.mark.parametrize("profile", [False, True])
def test_all_replay_kernel_lookup_profiles_only_when_enabled(monkeypatch, profile):
    from optimization.gpu import mps_kernel
    clock = iter([10.0, 13.0])
    def tick():
        assert profile, "disabled profiling must not read the clock"
        return next(clock)
    kernel = object()
    class Library:
        @property
        def replay(self):
            return kernel
    monkeypatch.setattr(mps_kernel.time, "perf_counter", tick)
    actual, seconds, prepared = mps_kernel._lookup_replay_kernel(Library(), "replay", profile=profile)
    assert actual is kernel
    assert seconds == (3.0 if profile else 0.0)
    assert prepared == (13.0 if profile else 0.0)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_gpu_strict_entry_gate_decrements_representable_quantity_steps():
    import passivbot_rust

    # Exercise the production arithmetic directly on the device. Venue counts
    # specify the expected decrement independently of floating physical units.
    source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    start = source.index("inline float decrement_aligned_qty_step(")
    stop = source.index("\n}", start) + 2
    probe = """#include <metal_stdlib>
using namespace metal;
""" + source[start:stop] + """
kernel void passivbot_quantity_decrement_probe(
    constant float* quantity, constant float* steps, device float* result,
    uint b [[thread_position_in_grid]]
) {
    result[b] = decrement_aligned_qty_step(quantity[b], steps[b]);
}
"""
    counts = np.asarray([0, 1, 2, 64, 66, 70, 78, 81, 93, 65537, 16777216, 16777218, 20000000, 33554432, 67108864, 1e10], dtype=np.float32)
    steps = np.repeat(np.asarray([0.001, 0.1, 0.25, 1e-8], dtype=np.float32), len(counts))
    counts = np.tile(counts, 4)
    quantity = torch.as_tensor(counts * steps, device=gpu_device())
    packed_steps = torch.as_tensor(steps, device=gpu_device())
    output = torch.empty_like(quantity)
    library = compile_shader(probe)
    library.passivbot_quantity_decrement_probe(quantity, packed_steps, output,
        threads=(len(counts), 1, 1))
    synchronize()
    returned_counts = torch.round(output / packed_steps).cpu().numpy()
    exact = counts < 2**24
    assert returned_counts[exact].tolist() == np.maximum(counts[exact] - 1, 0).tolist()
    large = ~exact
    assert np.all(returned_counts[large] < counts[large])
    assert bool(torch.all(output[quantity > 0] < quantity[quantity > 0]).item())
    assert bool(torch.all(output >= 0).item())


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU required for production fragment")
def test_gpu_strict_entry_gate_rechecks_the_recomputed_total():
    import torch
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    source=passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    def function(name):
        start=source.index("inline float "+name+"(")
        end=source.index("\n}",start)+2
        return source[start:end]
    start=source.index("inline void apply_tm_multicoin_recursive_entry_twel_gate(")
    first=source.index("            float room_cost =",start)
    last=source.index("            float executable_min =",first)
    arithmetic=source[first:last]
    helpers=function("floor_step")+"\n"+function("decrement_aligned_qty_step")
    helpers+="\n"+function("strict_cap_quantity")
    probe=("#include <metal_stdlib>\nusing namespace metal;\n"+helpers+"""
kernel void strict_cap_probe(constant float* inputs, device float* output,
 uint b [[thread_position_in_grid]]) {
    float retained_cost=inputs[b*5],cap_cost=inputs[b*5+1],gate_price=inputs[b*5+2];
    float c_mult=inputs[b*5+3],qty_step=inputs[b*5+4],kept_qty=0;
"""+arithmetic+"""
    output[b*2]=kept_qty;
    output[b*2+1]=retained_cost+kept_qty*gate_price*c_mult;
}
""")
    cap=np.nextafter(np.float32(1e6),np.float32(np.inf))
    rows=np.asarray([
        [384.41486,438.61206,.01,1,.001],
        [1e6,cap,1,1,1e-10],
        [.25,.5,1,1,.25],
        [0,100,1,1,1],
    ],dtype=np.float32)
    device=gpu_device()
    inputs=torch.as_tensor(rows,device=device)
    outputs=torch.empty((len(rows),2),dtype=torch.float32,device=device)
    compile_shader(probe).strict_cap_probe(inputs,outputs,threads=len(rows))
    synchronize()
    result=outputs.cpu().numpy()
    assert np.all(np.isfinite(result))
    assert np.all(result[:,0]>=0)
    assert np.all(result[:,1]<rows[:,1])


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_market_entries_use_original_precision_execution_tick(strategy_kind, side):
    import passivbot_rust
    # BUY is nextafter(167.93 / 1.0005, +inf); SELL is the observed downward
    # boundary. Each legacy float32 path loses one different execution tick.
    close = 167.8460769615193 if side == "long" else 167.84391568013112
    step, slip = .01, .0005
    native_price = (passivbot_rust.round_dn(close * (1.0 - slip), step) if side == "short"
                    else passivbot_rust.round_up(close * (1.0 + slip), step))
    native_tick = round(native_price / step)
    slipped32 = np.float32(close) * (np.float32(1.0) + np.float32(slip) if side == "long"
                                   else np.float32(1.0) - np.float32(slip))
    units32 = np.float32(slipped32 / np.float32(step))
    legacy_tick = int(np.ceil(np.float32(units32 - np.float32(1e-6))) if side == "long"
                      else np.floor(np.float32(units32 + np.float32(1e-6))))
    assert native_tick == (16794 if side == "long" else 16775)
    assert legacy_tick == native_tick + (-1 if side == "long" else 1)
    closes = np.full((6, 2), close, dtype=np.float64)
    markets = [ProxyMarket(.001, step, .001, 0.0, 1.0, 0.0, .0001)] * 2
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, count=6, closes=closes, highs=closes, lows=closes,
        markets=markets, market_orders_allowed=True,
        market_order_near_touch_threshold=.001, market_order_slippage_pct=slip,
    )
    keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS if strategy_kind == "ema_anchor" else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    row[keys.index("entry_cooldown_minutes")] = 100.0
    if strategy_kind == "ema_anchor": row[keys.index("offset")] = 0.0
    else: row[keys.index("entry_initial_ema_dist")] = 0.0
    # First planning is k=2; stored intent fills at k=3. Stops are exclusive.
    output = runner.run(np.asarray([row], dtype=np.float64), end_steps=np.asarray([4], dtype=np.int32))
    synchronize()
    assert output["fill_count_entry"].item() == 2.0
    assert output["open_positions"].item() == 2.0
    assert output["fill_count"].item() == output["fill_count_entry"].item()
    size = output["short_psize" if side == "short" else "psize"].item()
    cost = output["short_pprice" if side == "short" else "pprice"].item()
    assert size > 0.0
    assert cost / size == pytest.approx(native_price, abs=5e-5)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Apple MPS and NVIDIA CUDA unavailable")
def test_market_execution_decode_retains_native_effective_step_clamp():
    import passivbot_rust
    probe = r"""
kernel void market_effective_step_probe(
    constant int* ticks, constant float* steps, device float* output,
    uint b [[thread_position_in_grid]]
) {
    output[b] = ordinary_market_fill_price(ticks[b % 2], steps[b / 2]);
}
"""
    steps64 = np.asarray([0.0, 2.0**-53, 2.0**-52, 2.0**-51])
    ticks = torch.tensor([1, 3], dtype=torch.int32, device=gpu_device())
    steps = torch.as_tensor(steps64.astype(np.float32), device=gpu_device())
    output = torch.empty(8, dtype=torch.float32, device=gpu_device())
    library = compile_shader(passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe)
    library.market_effective_step_probe(ticks, steps, output, threads=(8, 1, 1))
    synchronize()
    expected = np.asarray([[1.0, 3.0]] * 4) * np.maximum(steps64[:, None], 2.0**-52)
    np.testing.assert_array_equal(output.cpu().numpy(), expected.astype(np.float32).reshape(-1))
