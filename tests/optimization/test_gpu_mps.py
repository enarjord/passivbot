import ast
from functools import lru_cache
import inspect
import textwrap

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.model import (
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
    _build_hourly_log_range,
    _directional_touch_ticks,
    _maximum_effective_min_cost,
    _minimum_entry_qty_encoding,
    _strict_fill_tick_boundaries,
    build_mps_data,
    build_mps_multicoin_data,
)
from optimization.gpu.mps_kernel import (
    MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY,
    MpsEmaAnchorMulticoinRunner,
    MpsEmaAnchorRunner,
    MpsEmaAnchorMulticoinFusedRunner,
    MpsTrailingMartingaleRunner,
    MpsTrailingMartingaleMulticoinFusedRunner,
    _decode_entry_interval_outputs,
    _cached_library_with_miss,
    _decode_multicoin_fused_outputs,
    _encode_max_realized_loss_pct,
    _pack_tm_parameter_matrix,
    _tm_dispatch_specialization,
    _scale_directional_minute_parameters,
    _scale_ema_multicoin_coin_overrides,
    _scale_tm_multicoin_coin_overrides,
    _scale_single_coin_minute_parameters,
    _upgrade_legacy_single_coin_wel_params,
    _with_btc_risk,
    _with_dynamic_wel_by_tradability,
    _with_entry_interval,
    _with_hsl_ema_tail,
    _with_hsl_features,
    _with_recovery_distribution,
    _with_tm_dispatch_features,
    strategy_eq_recovery_distribution_from_samples,
)
from optimization.gpu.metrics import (
    _entry_interval_metrics,
    _fill_gap_metrics,
    compute_objectives,
)
from optimization.gpu.service import MpsMulticoinEmaProxy


@pytest.mark.parametrize(
    ("runner_cls", "dispatch_name"),
    [
        (MpsEmaAnchorRunner, "dispatch_once"),
        (MpsTrailingMartingaleRunner, "dispatch_once"),
        (MpsEmaAnchorMulticoinRunner, "_dispatch"),
    ],
)
def test_strategy_runner_valid_evaluation_has_one_kernel_dispatch(
    runner_cls, dispatch_name
):
    source = inspect.getsource(runner_cls.run)
    tree = ast.parse(textwrap.dedent(source))
    dispatch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == dispatch_name
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == dispatch_name
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
    assert torch.equal(
        fill_count, output["day_fill_count"].sum(dim=1)
    )
    active_days = output["fills_active_days_count"]
    assert torch.equal(active_days, active_days.round())
    assert (active_days >= 0.0).all()
    duration_days = (
        output["last_eq_ts"] - output["first_eq_ts"]
    ).clamp(min=0.0) / 86_400_000.0
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
            <= output["held_max_ms"][held_count > 0.0]
            * held_count[held_count > 0.0]
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
    assert (
        account_recovery[has_equity] <= equity_span[has_equity] + 1.0e-6
    ).all()
    assert (account_recovery[~has_equity] == 0.0).all()


def test_tm_parameter_packing_preserves_positive_underflow_mode_per_side():
    width = len(TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS)
    params = np.zeros((1, width * 2), dtype=np.float64)
    entry_column = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_retracement_base_pct"
    )
    close_column = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "close_retracement_base_pct"
    )
    params[0, entry_column] = 1.0e-50
    params[0, width + close_column] = 1.0e-50

    packed = _pack_tm_parameter_matrix(
        params, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS, sides=2
    )

    assert packed[0, entry_column] == np.finfo(np.float32).tiny
    assert packed[0, width + close_column] == np.finfo(np.float32).tiny
    assert packed[0, close_column] == 0.0
    assert packed[0, width + entry_column] == 0.0


@pytest.mark.parametrize("side_width", [35, 52])
def test_legacy_single_coin_rows_gain_per_side_wallet_exposure_sentinel(
    side_width,
):
    legacy_width = side_width - 1
    params = np.arange(legacy_width * 2, dtype=np.float64).reshape(1, -1)

    upgraded = _upgrade_legacy_single_coin_wel_params(
        params, side_width=side_width
    )

    assert upgraded.shape == (1, side_width * 2)
    assert upgraded[0, :legacy_width].tolist() == params[
        0, :legacy_width
    ].tolist()
    assert upgraded[0, legacy_width] == -1.0
    assert upgraded[0, side_width:-1].tolist() == params[
        0, legacy_width:
    ].tolist()
    assert upgraded[0, -1] == -1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize(
    ("base_wel", "allowance_pct", "legacy_raw", "expected_allowed_wel"),
    [
        (0.4, 0.25, False, 0.5),
        (0.4, 2.0, False, 0.9),
        (0.4, 2.0, True, 1.2),
        (-1.0, 0.25, False, 0.9),
    ],
)
def test_mps_single_coin_wallet_exposure_override_keeps_twel_separate(
    strategy_kind,
    base_wel,
    allowance_pct,
    legacy_raw,
    expected_allowed_wel,
):
    import passivbot_rust

    if strategy_kind == "ema_anchor":
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
        source = passivbot_rust.mps_ema_anchor_source_py()
        side_type = "EmaSide"
    else:
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        source = passivbot_rust.mps_trailing_martingale_source_py()
        side_type = "TmSide"
    values = {key: float(index + 1) for index, key in enumerate(keys)}
    values.update(
        {
            "total_wallet_exposure_limit": 0.9,
            "wallet_exposure_limit": base_wel,
            "we_excess_allowance_pct": allowance_pct,
            "we_excess_allowance_legacy_raw": float(legacy_raw),
        }
    )
    params = torch.tensor(
        [values[key] for key in keys], dtype=torch.float32, device="mps"
    )
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    probe_kernel = f"""
kernel void passivbot_single_coin_wel_probe(
    constant float* packed,
    device float* result,
    uint b [[thread_position_in_grid]]
) {{
    if (b > 0) return;
    {side_type} side = load_side(packed, 0, 100.0f);
    result[0] = side.twel;
    result[1] = side.allowed_wel;
}}
"""
    library = torch.mps.compile_shader(source + probe_kernel)
    library.passivbot_single_coin_wel_probe(
        params, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    assert output[0].item() == pytest.approx(0.9)
    assert output[1].item() == pytest.approx(expected_allowed_wel)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("side", "expected_entry_ticks"),
    [("long", 97.0), ("short", 103.0)],
)
def test_mps_ema_inventory_ratio_uses_single_coin_base_wel(
    side, expected_entry_ticks
):
    import passivbot_rust

    values = {key: 0.0 for key in EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS}
    values.update(
        {
            "base_qty_pct": 0.1,
            "ema_span_0": 2.0,
            "ema_span_1": 3.0,
            "offset_psize_weight": 0.1,
            "total_wallet_exposure_limit": 0.9,
            "twel_enforcer_threshold": 1.0,
            "wallet_exposure_limit": 0.4,
        }
    )
    params = torch.tensor(
        [values[key] for key in EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS],
        dtype=torch.float32,
        device="mps",
    )
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    generator = (
        "generate_long_orders" if side == "long" else "generate_short_orders"
    )
    probe_kernel = f"""
kernel void passivbot_ema_base_wel_inventory_probe(
    constant float* packed,
    device float* result,
    uint b [[thread_position_in_grid]]
) {{
    if (b > 0) return;
    EmaSide state = load_side(packed, 0, 100.0f);
    state.psize = 1.0f;
    state.pprice = 100.0f;
    {generator}(
        state,
        1000.0f,
        100.0f,
        100,
        100,
        0.001f,
        1.0f,
        0.001f,
        0.0f,
        1.0f,
        0.0f,
        false,
        false,
        0.0f
    );
    result[0] = float(state.entry_ticks);
    result[1] = state.base_wel;
}}
"""
    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_source_py() + probe_kernel
    )
    library.passivbot_ema_base_wel_inventory_probe(
        params, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    # With current WE=0.1, exact Rust's base-WEL ratio is 0.1 / 0.4 = 0.25.
    # Dividing by TWEL=0.9 instead would produce ticks 98/102.
    assert output[0].item() == expected_entry_ticks
    assert output[1].item() == pytest.approx(0.4)


@pytest.mark.parametrize(
    "value", [0.0, 0.05, 0.999999999, float(np.nextafter(1.0, 0.0))]
)
def test_mps_realized_loss_limit_float32_encoding_never_loosens(value):
    encoded = _encode_max_realized_loss_pct(value)

    assert encoded <= value
    assert encoded < 1.0


@pytest.mark.parametrize("value", [1.0, 2.0, 1.0e100])
def test_mps_disabled_realized_loss_limit_has_finite_float32_encoding(value):
    assert _encode_max_realized_loss_pct(value) == 1.0


def test_decode_multicoin_fused_outputs_maps_directional_reductions():
    daily = torch.zeros((1, 1, 9), dtype=torch.float32)
    daily[:, :, 1].fill_(float("inf"))
    daily[:, :, 5].fill_(float("inf"))
    scalars = torch.arange(72, dtype=torch.float32).reshape(1, 72)
    gaps = torch.zeros((1, 128), dtype=torch.int32)

    output = _decode_multicoin_fused_outputs(daily, scalars, gaps)

    assert "entry_initial_balance_pct" not in output
    assert torch.equal(output["entry_initial_balance_pct_long"], scalars[:, 21])
    assert torch.equal(output["hsl_drawdown_ema_max_long"], scalars[:, 57])
    assert torch.equal(output["hsl_drawdown_ema_max_short"], scalars[:, 58])
    assert torch.equal(output["entry_initial_balance_pct_short"], scalars[:, 59])
    assert torch.equal(output["profit_sum_long"], scalars[:, 60])
    assert torch.equal(output["loss_sum_long"], scalars[:, 61])
    assert torch.equal(output["profit_sum_short"], scalars[:, 62])
    assert torch.equal(output["loss_sum_short"], scalars[:, 63])
    assert torch.equal(
        output["hsl_strategy_eq_recovery_max_ms_long"], scalars[:, 64]
    )
    assert torch.equal(
        output["hsl_strategy_eq_recovery_max_ms_short"], scalars[:, 65]
    )
    assert torch.equal(
        output["hsl_drawdown_ema_mean_worst_1pct_long"], scalars[:, 66]
    )
    assert torch.equal(
        output["hsl_drawdown_ema_mean_worst_1pct_short"], scalars[:, 67]
    )
    assert torch.equal(output["hsl_drawdown_raw_max_long"], scalars[:, 68])
    assert torch.equal(output["hsl_drawdown_raw_max_short"], scalars[:, 69])
    assert torch.equal(
        output["hsl_drawdown_raw_mean_worst_1pct_long"], scalars[:, 70]
    )
    assert torch.equal(
        output["hsl_drawdown_raw_mean_worst_1pct_short"], scalars[:, 71]
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

    assert _with_hsl_features(
        source,
        ema_tail_enabled=False,
        raw_drawdown_enabled=False,
        raw_tail_enabled=False,
    ) is source
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
        "#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 1\n"
        "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n"
        + source
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
        "#define PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED 1\n"
        + source
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_strategy_eq_recovery_distribution_matches_strict_rust_contract():
    matrix = torch.tensor(
        [
            [100.0, 90.0, 95.0, 101.0, 100.0, 102.0],
            [100.0, 100.0, 101.0, float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
        device="mps",
    )

    actual = strategy_eq_recovery_distribution_from_samples(
        matrix, sample_interval_days=1.0 / 24.0
    ).cpu().numpy()

    assert actual[0].tolist() == pytest.approx(
        [value / 24.0 for value in [8.0 / 6.0, 1.0, 2.75, 2.95, 3.0, 3.0, 3.0]]
    )
    assert actual[1].tolist() == pytest.approx(
        [value / 24.0 for value in [1.0, 1.0, 1.9, 1.98, 2.0, 2.0, 2.0]]
    )


def test_trailing_martingale_no_hsl_specialization_keeps_base_scalar_abi(
    monkeypatch,
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    def fake_base_init(self, *args, **kwargs):
        self.long_enabled = True
        self.short_enabled = False
        self.hsl_ema_tail_enabled = bool(kwargs["hsl_ema_tail_enabled"])
        self.hsl_raw_drawdown_enabled = bool(
            kwargs["hsl_raw_drawdown_enabled"]
        )
        self.hsl_raw_tail_enabled = bool(kwargs["hsl_raw_tail_enabled"])

    monkeypatch.setattr(MpsEmaAnchorRunner, "__init__", fake_base_init)
    monkeypatch.setattr(
        MpsTrailingMartingaleRunner,
        "_encode_hour_boundary_flags",
        lambda self: None,
    )
    runner = MpsTrailingMartingaleRunner(
        None,
        None,
        None,
        hsl_enabled=False,
        hsl_ema_tail_enabled=True,
        hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True,
    )

    assert runner.shader_topology == "long_no_hsl"
    assert runner.hsl_ema_tail_enabled is False
    assert runner.hsl_raw_drawdown_enabled is False
    assert runner.hsl_raw_tail_enabled is False


def test_trailing_martingale_hsl_specialization_keeps_requested_features(
    monkeypatch,
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    def fake_base_init(self, *args, **kwargs):
        self.long_enabled = False
        self.short_enabled = True
        self.hsl_ema_tail_enabled = bool(kwargs["hsl_ema_tail_enabled"])
        self.hsl_raw_drawdown_enabled = bool(
            kwargs["hsl_raw_drawdown_enabled"]
        )
        self.hsl_raw_tail_enabled = bool(kwargs["hsl_raw_tail_enabled"])

    monkeypatch.setattr(MpsEmaAnchorRunner, "__init__", fake_base_init)
    monkeypatch.setattr(
        MpsTrailingMartingaleRunner,
        "_encode_hour_boundary_flags",
        lambda self: None,
    )
    runner = MpsTrailingMartingaleRunner(
        None,
        None,
        None,
        hsl_enabled=True,
        hsl_ema_tail_enabled=True,
        hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True,
    )

    assert runner.shader_topology == "short_hsl"
    assert runner.hsl_ema_tail_enabled is True
    assert runner.hsl_raw_drawdown_enabled is True
    assert runner.hsl_raw_tail_enabled is True


def test_trailing_martingale_hsl_specialization_disables_unrequested_diagnostics(
    monkeypatch,
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    def fake_base_init(self, *args, **kwargs):
        self.long_enabled = True
        self.short_enabled = False
        self.hsl_ema_tail_enabled = False
        self.hsl_raw_drawdown_enabled = False
        self.hsl_raw_tail_enabled = False
        self.recovery_distribution_enabled = False
        self.btc_risk_enabled = False
        self.equity_balance_diff_enabled = False

    monkeypatch.setattr(MpsEmaAnchorRunner, "__init__", fake_base_init)
    monkeypatch.setattr(
        MpsTrailingMartingaleRunner,
        "_encode_hour_boundary_flags",
        lambda self: None,
    )
    runner = MpsTrailingMartingaleRunner(
        None,
        None,
        None,
        hsl_enabled=True,
        hsl_diagnostics_enabled=False,
    )

    assert runner.shader_topology == "long_hsl"
    assert runner.hsl_diagnostics_enabled is False
    assert runner._shader_library_cache_call()[1][-1] is False


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_hsl_diagnostic_opt_out_preserves_strategy_outputs():
    count = 30
    close = np.full(count, 100.0)
    close[8:] = 70.0
    high = close * 1.02
    low = close * 0.98
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0)
    row[6] = 0.5
    row[7] = 0.5
    row[16] = 0.5
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.01,
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(key)] = value
    matrix = np.asarray([row + row], dtype=np.float64)

    full = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        hsl_enabled=True,
        hsl_diagnostics_enabled=True,
    ).run(matrix)
    lean = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        hsl_enabled=True,
        hsl_diagnostics_enabled=False,
    ).run(matrix)
    torch.mps.synchronize()

    for key in (
        "day_end_eq",
        "day_min_eq",
        "day_max_dd",
        "fill_count",
        "psize",
        "pprice",
        "liq_step",
    ):
        torch.testing.assert_close(lean[key], full[key], rtol=0.0, atol=0.0)
    assert full["hsl_triggers_long"].item() > 0.0
    assert lean["hsl_triggers_long"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("interval_ms", "history_start_step", "trade_start_step"),
    [
        (60_000, 59, 80),
        (45 * 60_000, 1, 4),
    ],
)
def test_mps_tm_history_window_preserves_full_run_and_selects_recent_suffix(
    interval_ms, history_start_step, trade_start_step
):
    count = 120
    steps = np.arange(count, dtype=np.float64)
    close = 100.0 + np.sin(steps / 7.0)
    high = close + 0.2
    low = close - 0.2
    first_ts_ms = (1_700_000_000_000 // 3_600_000) * 3_600_000
    timestamps = first_ts_ms + steps.astype(np.int64) * interval_ms
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        interval_ms,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0)
    row_two = _tm_single_row(initial_ema_dist=0.002)
    matrix = np.asarray([row + row, row_two + row_two], dtype=np.float64)
    volatility_column = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_threshold_volatility_1h_weight"
    )
    matrix[:, volatility_column] = [0.25, 0.5]

    def run_once(end_step=None, history_start_step=None, trade_start_step=None):
        runner = MpsTrailingMartingaleRunner(
            market,
            run,
            data,
            long_enabled=True,
            short_enabled=False,
            hsl_enabled=False,
            recovery_distribution_enabled=True,
        )
        return runner.run(
            matrix,
            end_step=end_step,
            history_start_step=history_start_step,
            trade_start_step=trade_start_step,
        )

    ordinary = run_once()
    explicit_full = run_once(count)
    truncated = run_once(60)
    recent = run_once(
        history_start_step=history_start_step,
        trade_start_step=trade_start_step,
    )
    sliced_run = ProxyRun(
        1_000.0,
        trade_start_step - history_start_step - 1,
        trade_start_step - history_start_step,
        int(timestamps[trade_start_step]),
        int(timestamps[trade_start_step]),
        int(timestamps[history_start_step]),
        interval_ms,
        0.05,
        0,
        count - history_start_step - 1,
    )
    sliced_data = build_mps_data(
        high[history_start_step:],
        low[history_start_step:],
        close[history_start_step:],
        timestamps[history_start_step:],
        sliced_run,
        market,
    )
    sliced = MpsTrailingMartingaleRunner(
        market,
        sliced_run,
        sliced_data,
        long_enabled=True,
        short_enabled=False,
        hsl_enabled=False,
        recovery_distribution_enabled=True,
    ).run(matrix)
    torch.mps.synchronize()

    for key in ordinary:
        if isinstance(ordinary[key], torch.Tensor):
            torch.testing.assert_close(
                ordinary[key], explicit_full[key], rtol=0.0, atol=0.0,
                equal_nan=True,
            )
    assert truncated["last_eq_ts"][0].item() < ordinary["last_eq_ts"][0].item()
    assert recent["first_eq_ts"][0].item() == trade_start_step * interval_ms
    assert recent["last_eq_ts"][0].item() == ordinary["last_eq_ts"][0].item()
    for key in (
        "balance",
        "psize",
        "pprice",
        "fill_count",
        "fill_count_entry",
        "max_dd",
        "held_max_ms",
        "held_sum_ms",
    ):
        torch.testing.assert_close(recent[key], sliced[key], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        recent["strategy_eq_recovery_samples"],
        sliced["strategy_eq_recovery_samples"][
            :, : recent["strategy_eq_recovery_samples"].shape[1]
        ],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert recent["first_eq_ts"][0].item() == (
        sliced["first_eq_ts"][0].item() + history_start_step * interval_ms
    )
    assert recent["last_eq_ts"][0].item() == (
        sliced["last_eq_ts"][0].item() + history_start_step * interval_ms
    )


@pytest.mark.parametrize("recovery_enabled", [False, True])
@pytest.mark.parametrize("runner_name", ["ema_anchor", "trailing_martingale"])
def test_single_coin_size_buffer_packs_last_valid_before_recovery_fields(
    runner_name, recovery_enabled
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    runner_cls = (
        MpsEmaAnchorRunner
        if runner_name == "ema_anchor"
        else MpsTrailingMartingaleRunner
    )
    runner = object.__new__(runner_cls)
    runner.n = 17
    runner.n_days = 2
    runner.run_config = ProxyRun(
        1_000.0, 1, 1, 0, 0, 0, 60_000, 0.05, 3, 11
    )
    runner.rolling_capacity = 9
    runner.pnl_lookback_bars = 7
    runner.recovery_distribution_enabled = recovery_enabled
    runner.recovery_stride = 60
    runner.n_recovery_samples = 4

    actual = runner._single_coin_size_values(5, 13)

    assert actual == [5, 17, 2, 13, 3, 9, 7, 11] + (
        [60, 4] if recovery_enabled else []
    )

    truncated = runner._single_coin_size_values(5, 13, end_step=12)
    assert truncated == [5, 12, 2, 13, 3, 9, 7, 11] + (
        [60, 4] if recovery_enabled else []
    )

    with pytest.raises(ValueError, match="end_step"):
        runner._single_coin_size_values(5, 13, end_step=18)


@pytest.mark.parametrize("recovery_enabled", [False, True])
def test_tm_size_buffer_appends_recent_window_after_recovery_fields(
    recovery_enabled,
):
    runner = object.__new__(MpsTrailingMartingaleRunner)
    runner.n = 17
    runner.n_days = 2
    runner.run_config = ProxyRun(
        1_000.0, 1, 1, 0, 0, 0, 60_000, 0.05, 3, 11
    )
    runner.rolling_capacity = 9
    runner.pnl_lookback_bars = 7
    runner.recovery_distribution_enabled = recovery_enabled
    runner.recovery_stride = 2
    runner.n_recovery_samples = 10

    ordinary = runner._trailing_single_coin_size_values(5, 13)
    recent = runner._trailing_single_coin_size_values(
        5,
        13,
        history_start_step=5,
        trade_start_step=10,
    )

    recovery = [2, 10] if recovery_enabled else [0, 0]
    assert ordinary == [5, 17, 2, 13, 3, 9, 7, 11] + recovery + [
        -1,
        -1,
        10 if recovery_enabled else 0,
        -1,
        0,
        -1,
    ]
    assert recent == [5, 17, 2, 13, 3, 9, 7, 11] + recovery + [
        5,
        10,
        5 if recovery_enabled else 0,
        17,
        0,
        5,
    ]
    with pytest.raises(ValueError, match="requires both"):
        runner._trailing_single_coin_size_values(
            5, 13, history_start_step=5
        )


def test_trailing_martingale_runner_accepts_ordinary_market_execution(monkeypatch):
    from optimization.gpu.mps_kernel import MpsEmaAnchorRunner
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    def fake_base_init(self, *args, **kwargs):
        self.long_enabled = True
        self.short_enabled = False
        self.hsl_ema_tail_enabled = False
        self.hsl_raw_drawdown_enabled = False
        self.hsl_raw_tail_enabled = False
        self.recovery_distribution_enabled = False

    monkeypatch.setattr(MpsEmaAnchorRunner, "__init__", fake_base_init)
    monkeypatch.setattr(
        MpsTrailingMartingaleRunner,
        "_encode_hour_boundary_flags",
        lambda self: None,
    )
    runner = MpsTrailingMartingaleRunner(
        None, None, None, market_orders_allowed=True, hsl_enabled=False
    )

    assert runner.shader_topology == "long_no_hsl"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_coin_hsl_rolling_pnl_window_expires_and_resets_fill_events():
    import passivbot_rust

    dense_round_trip_count = 2_096
    fills_per_round_trip = 4
    assert MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY >= dense_round_trip_count
    probe_kernel = r"""
kernel void passivbot_hsl_rolling_pnl_probe(
    device float2* values,
    device int2* indices,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    HslRollingPnlWindow window = init_hsl_rolling_pnl_window();
    const int capacity = 4;
    const int lookback_bars = 2;
    record_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 0, lookback_bars, true, 50.0f
    );
    HslRollingPnlSignal first = effective_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 0, lookback_bars
    );
    record_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 1, lookback_bars, true, -80.0f
    );
    HslRollingPnlSignal second = effective_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 1, lookback_bars
    );
    HslRollingPnlSignal expired = effective_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 3, lookback_bars
    );
    record_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 4, lookback_bars, true, 55.0f
    );
    HslRollingPnlSignal final_signal = effective_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 4, lookback_bars
    );
    reset_hsl_rolling_pnl_window(window);
    HslRollingPnlSignal reset_signal = effective_hsl_rolling_pnl(
        window, values, indices, 0, capacity, 4, lookback_bars
    );

    output[0] = first.peak;
    output[1] = first.current;
    output[2] = second.peak;
    output[3] = second.current;
    output[4] = expired.peak;
    output[5] = expired.current;
    output[6] = final_signal.peak;
    output[7] = final_signal.current;
    output[8] = reset_signal.peak;
    output[9] = reset_signal.current;

    HslRollingPnlWindow inactive = init_hsl_rolling_pnl_window();
    record_hsl_rolling_pnl(
        inactive, values, indices, 4, 2, 0, 10, false, 1.0f
    );
    record_hsl_rolling_pnl(
        inactive, values, indices, 4, 2, 1, 10, false, 1.0f
    );
    record_hsl_rolling_pnl(
        inactive, values, indices, 4, 2, 2, 10, false, 1.0f
    );
    output[10] = inactive.overflowed ? 1.0f : 0.0f;

    HslRollingPnlWindow overflow = init_hsl_rolling_pnl_window();
    record_hsl_rolling_pnl(
        overflow, values, indices, 4, 2, 0, 10, true, 1.0f
    );
    record_hsl_rolling_pnl(
        overflow, values, indices, 4, 2, 1, 10, true, 1.0f
    );
    record_hsl_rolling_pnl(
        overflow, values, indices, 4, 2, 2, 10, true, 1.0f
    );
    output[11] = overflow.overflowed ? 1.0f : 0.0f;

    HslRollingPnlWindow dense = init_hsl_rolling_pnl_window();
    for (int k = 0; k < __DENSE_ROUND_TRIP_COUNT__; ++k) {
        for (int fill = 0; fill < __FILLS_PER_ROUND_TRIP__; ++fill) {
            record_hsl_rolling_pnl(
                dense, values, indices, 6, __DENSE_CAPACITY__, k,
                __DENSE_ROUND_TRIP_COUNT__ + 1, true, -0.1f
            );
        }
        record_hsl_rolling_pnl(
            dense, values, indices, 6, __DENSE_CAPACITY__, k,
            __DENSE_ROUND_TRIP_COUNT__ + 1, true, 1.0f
        );
    }
    output[12] = dense.overflowed ? 1.0f : 0.0f;
    output[13] = float(dense.event_count);

    HslRollingPnlWindow coalesced = init_hsl_rolling_pnl_window();
    record_hsl_rolling_pnl(
        coalesced, values, indices, 6, __DENSE_CAPACITY__, 0, 2, true, 50.0f
    );
    record_hsl_rolling_pnl(
        coalesced, values, indices, 6, __DENSE_CAPACITY__, 0, 2, true, -80.0f
    );
    HslRollingPnlSignal coalesced_signal = effective_hsl_rolling_pnl(
        coalesced, values, indices, 6, __DENSE_CAPACITY__, 0, 2
    );
    output[14] = coalesced_signal.peak;
    output[15] = coalesced_signal.current;
    output[16] = float(coalesced.event_count);
}
""".replace(
        "__DENSE_ROUND_TRIP_COUNT__", str(dense_round_trip_count)
    ).replace("__FILLS_PER_ROUND_TRIP__", str(fills_per_round_trip)).replace(
        "__DENSE_CAPACITY__", str(MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY)
    )
    buffer_size = 6 + MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY
    values = torch.empty((buffer_size, 2), dtype=torch.float32, device="mps")
    indices = torch.empty((buffer_size, 2), dtype=torch.int32, device="mps")
    output = torch.zeros(17, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_source_py() + probe_kernel
    )

    library.passivbot_hsl_rolling_pnl_probe(
        values, indices, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [
        50.0,
        50.0,
        50.0,
        -30.0,
        0.0,
        -80.0,
        55.0,
        55.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        float(dense_round_trip_count),
        50.0,
        -30.0,
        1.0,
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        device="mps",
    )
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_source_py() + probe_kernel
    )

    library.passivbot_hsl_strategy_equity_recovery_probe(
        samples, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    # Exact Rust waits for a strictly greater sample, so the equal sample at
    # k=1 remains underwater until 101 at k=4.
    assert output[0].item() == 4.0
    # Exact Rust compacts away halted cooldown bars before recovery analysis.
    # The resumed sequence has four eligible samples, so recovery is three
    # samples even though its source candle indices span five steps.
    assert output[1].item() == 3.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        device="mps",
    )
    output = torch.zeros(1, dtype=torch.float32, device="mps")
    source = _with_hsl_ema_tail(
        passivbot_rust.mps_ema_anchor_source_py(), True
    )
    library = torch.mps.compile_shader(source + probe_kernel)
    library.passivbot_hsl_drawdown_ema_tail_probe(
        samples, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    # floor(200 * 1%) is two, and the two worst values occupy the highest bin.
    assert output.item() == pytest.approx(0.4, abs=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        device="mps",
    )
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_source_py(),
        ema_tail_enabled=False,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    library = torch.mps.compile_shader(source + probe_kernel)
    library.passivbot_hsl_strategy_equity_raw_drawdown_probe(
        samples, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    assert output[0].item() == pytest.approx(0.2, abs=1.0e-6)
    # The two intraday lows reduce to daily worst values of 0.1 and 0.2;
    # the open terminal day contributes zero. With three observed days, Rust's
    # floor(1%) contract retains the single worst day.
    assert output[1].item() == pytest.approx(0.2, abs=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_joint_pside_hsl_contract_routes_unified_and_directional_signals():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_joint_pside_hsl_contract_probe(
    constant float* hsl_params,
    constant float* samples,
    constant float* settings,
    constant int* sizes,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    const int B = sizes[0];
    const int T = sizes[1];
    if (b >= uint(B)) return;
    const int po = int(b) * 22;
    HslState long_hsl = load_hsl(hsl_params, po, 0);
    HslState short_hsl = load_hsl(hsl_params, po, 11);
    const float starting_balance = settings[0];
    const float interval_ms = settings[1];
    JointPortfolioAccount account = init_joint_portfolio_account(starting_balance);
    float unrealized_long = 0.0f;
    float unrealized_short = 0.0f;
    bool has_position_long = false;
    bool has_position_short = false;
    for (int k = 0; k < T; ++k) {
        int so = (int(b) * T + k) * 8;
        account.realized_pnl_long = samples[so + 0];
        account.realized_pnl_short = samples[so + 1];
        account.realized_pnl_total = account.realized_pnl_long
            + account.realized_pnl_short;
        account.realized_pnl_peak = fmax(
            account.realized_pnl_peak, account.realized_pnl_total
        );
        account.balance = starting_balance + account.realized_pnl_total;
        unrealized_long = samples[so + 2];
        unrealized_short = samples[so + 3];
        has_position_long = samples[so + 4] > 0.5f;
        has_position_short = samples[so + 5] > 0.5f;
        bool blocking_long = samples[so + 6] > 0.5f;
        bool blocking_short = samples[so + 7] > 0.5f;
        update_joint_pside_hsl(
            long_hsl, short_hsl, account, starting_balance,
            unrealized_long, unrealized_short,
            has_position_long, has_position_short,
            blocking_long, blocking_short, float(k), interval_ms
        );
        try_restart_joint_pside_hsl(
            long_hsl, short_hsl, account, starting_balance,
            unrealized_long, unrealized_short, float(k)
        );
    }
    int oo = int(b) * 62;
    output[oo + 0] = float(long_hsl.tier);
    output[oo + 1] = float(short_hsl.tier);
    output[oo + 2] = long_hsl.triggers;
    output[oo + 3] = short_hsl.triggers;
    output[oo + 4] = float(hsl_mode(long_hsl, has_position_long));
    output[oo + 5] = float(hsl_mode(short_hsl, has_position_short));
    output[oo + 6] = joint_portfolio_equity(
        account, unrealized_long, unrealized_short
    );
    output[oo + 7] = float(joint_pside_hsl_global_tier(long_hsl, short_hsl));
    long_hsl.halt_duration_sum_steps = 2.0f;
    long_hsl.halt_duration_max_steps = 2.0f;
    long_hsl.halt_duration_count = 1.0f;
    short_hsl.halt_duration_sum_steps = 3.0f;
    short_hsl.halt_duration_max_steps = 3.0f;
    short_hsl.halt_duration_count = 1.0f;
    long_hsl.panic_loss_drawdown_min = 0.2f;
    long_hsl.panic_loss_drawdown_sum = 0.2f;
    long_hsl.panic_loss_drawdown_max = 0.2f;
    long_hsl.panic_loss_drawdown_count = 1.0f;
    short_hsl.panic_loss_drawdown_min = 0.1f;
    short_hsl.panic_loss_drawdown_sum = 0.1f;
    short_hsl.panic_loss_drawdown_max = 0.1f;
    short_hsl.panic_loss_drawdown_count = 1.0f;
    write_dual_side_hsl_outputs(
        long_hsl, short_hsl, 4.0f, 1.0f, 2.0f, 3.0f, -1.0f,
        output, oo + 8
    );
    HslState long_coin_hsl[2];
    HslState short_coin_hsl[2];
    long_coin_hsl[0] = long_hsl;
    long_coin_hsl[1] = long_hsl;
    short_coin_hsl[0] = short_hsl;
    short_coin_hsl[1] = short_hsl;
    write_dual_side_coin_hsl_outputs(
        long_coin_hsl, short_coin_hsl, 2,
        4.0f, 1.0f, 2.0f, 3.0f, -1.0f,
        output, oo + 35
    );
}
"""

    def controller(mode):
        return [
            1.0,
            0.05,
            1.0,
            0.0,
            1.0,
            2.0,
            0.5,
            0.75,
            0.0,
            float(mode),
            1.0,
        ]

    params = torch.tensor(
        [
            controller(0) + controller(0),
            controller(1) + controller(1),
            controller(2) + controller(2),
            controller(0) + controller(1),
            controller(0) + controller(0),
        ],
        dtype=torch.float32,
        device="mps",
    )
    samples = torch.zeros((5, 4, 8), dtype=torch.float32, device="mps")
    samples[:, :2, 4] = 1.0
    samples[:, :2, 5] = 1.0
    samples[:, 1:, 3] = -10.0
    samples[4, :, 4] = 1.0
    settings = torch.tensor([100.0, 60_000.0], device="mps")
    sizes = torch.tensor([5, 4], dtype=torch.int32, device="mps")
    output = torch.zeros((5, 62), dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_joint_pside_hsl_contract_probe(
        params,
        samples,
        settings,
        sizes,
        output,
        threads=(5, 1, 1),
    )
    torch.mps.synchronize()
    values = output.cpu().numpy()

    assert values[0, :4].tolist() == [3.0, 3.0, 1.0, 1.0]
    assert values[1, :4].tolist() == [0.0, 3.0, 0.0, 1.0]
    assert values[2, :4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert values[3, :4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert values[4, :4].tolist() == [3.0, 3.0, 0.0, 0.0]
    assert values[:, 6].tolist() == [90.0, 90.0, 90.0, 90.0, 90.0]
    assert values[:, 7].tolist() == [3.0, 3.0, 0.0, 0.0, 3.0]
    assert values[0, 8:14].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert values[1, 8:14].tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert values[:, 14:18].tolist() == [[4.0, 1.0, 2.0, 3.0]] * 5
    assert values[:, 18:21].tolist() == [[5.0, 3.0, 2.0]] * 5
    np.testing.assert_allclose(values[:, 29:33], [[0.1, 0.3, 0.2, 2.0]] * 5)
    assert (values[:, 33:35] >= 0.0).all()
    assert values[0, 35:41].tolist() == [1.0, 1.0, 2.0, 2.0, 0.0, 0.0]
    assert values[1, 35:41].tolist() == [1.0, 1.0, 0.0, 2.0, 0.0, 0.0]
    assert values[:, 41:45].tolist() == [[4.0, 1.0, 2.0, 3.0]] * 5
    assert values[:, 45:48].tolist() == [[10.0, 3.0, 4.0]] * 5
    np.testing.assert_allclose(values[:, 56:60], [[0.1, 0.6, 0.2, 4.0]] * 5)
    np.testing.assert_allclose(values[:, 60:62], values[:, 33:35])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    long_side.coin_hsl_entry_blocked_mask = 7ul;
    short_side.coin_hsl_entry_blocked_mask = 8ul;
    output[0] = long_side.psize[0];
    output[1] = short_side.psize[0];
    output[2] = float(long_side.entry_tick[0]);
    output[3] = float(short_side.entry_tick[0]);
    output[4] = long_side.selected[0] && !short_side.selected[0] ? 1.0f : 0.0f;
    output[5] = long_side.hsl.enabled && !short_side.hsl.enabled ? 1.0f : 0.0f;
    output[6] = long_side.coin_hsl[0].triggers
        + short_side.coin_hsl[0].triggers;
    output[7] = float(
        long_side.coin_hsl_entry_blocked_mask
            + short_side.coin_hsl_entry_blocked_mask
    );
}
"""

    output = torch.zeros(8, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_side_state_isolation_probe(
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 11.0, 15.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_multicoin_fused_reducers_share_loss_budget_and_fallback():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_fused_reducer_budget_probe(
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
    bars = torch.zeros((2, 2, 4), dtype=torch.float32, device="mps")
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
        device="mps",
    )
    output = torch.zeros(8, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_fused_reducer_budget_probe(
        bars,
        coin_settings,
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    long_side.coin_hsl_entry_blocked_mask = 9ul;
    short_side.coin_hsl_entry_blocked_mask = 10ul;
    long_side.selection_initialized = true;
    short_side.selection_initialized = false;
    long_side.max_tradable_seen = 11;
    short_side.max_tradable_seen = 12;
    long_side.previous_effective_n_positions = 13;
    short_side.previous_effective_n_positions = 14;
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
        long_side.coin_hsl_entry_blocked_mask
            + short_side.coin_hsl_entry_blocked_mask
    );
    output[9] = long_side.selection_initialized
        && !short_side.selection_initialized ? 1.0f : 0.0f;
    output[10] = float(long_side.max_tradable_seen);
    output[11] = float(short_side.max_tradable_seen);
    output[12] = float(long_side.previous_effective_n_positions);
    output[13] = float(short_side.previous_effective_n_positions);
}
"""

    output = torch.zeros(14, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_side_state_isolation_probe(
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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

    output = torch.zeros(18, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_fill_state_probe(
        output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    short_side.hsl = load_hsl(hsl_params, 11, 0);
    short_side.coin_hsl[0] = load_hsl(hsl_params, 11, 0);
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;
    long_side.coin_hsl[0].coin_realized_baseline = -20.0f;
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
    output[14] = long_side.coin_hsl[0].coin_realized_peak;
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
            1.0,
            1.0,
            0.5,
            0.75,
            0.0,
            2.0,
            1.0,
            1.0,
            0.2,
            60.0,
            0.0,
            1.0,
            1.0,
            0.5,
            0.75,
            0.0,
            1.0,
            1.0,
        ],
        dtype=torch.float32,
        device="mps",
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device="mps")
    output = torch.zeros(20, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_close_fill_probe(
        hsl_params, coin_fill_counts, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

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
        9.0,
        1000.0,
        11.0,
        989.0,
        11.0,
        2.0,
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    short_side.hsl = load_hsl(hsl_params, 11, 0);
    short_side.coin_hsl[0] = load_hsl(hsl_params, 11, 0);
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;
    long_side.coin_hsl[0].coin_realized_baseline = -20.0f;
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
    output[12] = long_side.coin_hsl[0].coin_realized_peak;
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
            1.0,
            1.0,
            0.5,
            0.75,
            0.0,
            2.0,
            1.0,
            1.0,
            0.2,
            60.0,
            0.0,
            1.0,
            1.0,
            0.5,
            0.75,
            0.0,
            1.0,
            1.0,
        ],
        dtype=torch.float32,
        device="mps",
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device="mps")
    output = torch.zeros(16, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_entry_fill_probe(
        hsl_params, coin_fill_counts, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

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
        10.0,
        0.0,
        0.0,
        2.0,
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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

    output = torch.zeros(24, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_fill_position_probe(
        output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_side_fill_pass_shares_account_chronology():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_side_fill_pass_probe(
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
    TrailingMartingaleMulticoinSideConfig long_config;
    TrailingMartingaleMulticoinSideConfig short_config;
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
        long_side.entry_recursive_market_mode[c] = false;
        short_side.entry_recursive_market_mode[c] = false;
        long_side.close_qty[c] = 0.0f;
        short_side.close_qty[c] = 0.0f;
        long_side.secondary_close_qty[c] = 0.0f;
        short_side.secondary_close_qty[c] = 0.0f;
        long_side.close_reconstruct_after_reducer[c] = false;
        short_side.close_reconstruct_after_reducer[c] = false;
        long_side.close_recursive_market_mode[c] = false;
        short_side.close_recursive_market_mode[c] = false;
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
        long_side, long_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 1, 1, false, true, true, false,
        1.0f, 0.0f, 0.0f, false, long_equity
    );
    bool short_entry = process_tm_multicoin_side_fills(
        short_side, short_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 1, 1, true, true, true, false,
        1.0f, 0.0f, 0.0f, false, short_equity
    );
    long_side.close_qty[0] = 1.0f;
    short_side.close_qty[0] = 1.0f;
    long_side.close_tick[0] = 100;
    short_side.close_tick[0] = 100;
    long_equity = 1000.0f;
    short_equity = 1000.0f;
    bool long_close = process_tm_multicoin_side_fills(
        long_side, long_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 2, 1, false, true, true, false,
        1.0f, 0.0f, 0.0f, false, long_equity
    );
    bool short_close = process_tm_multicoin_side_fills(
        short_side, short_config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 2, 1, true, true, true, false,
        1.0f, 0.0f, 0.0f, false, short_equity
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
        device="mps",
    )
    fill_ticks = torch.tensor(
        [[0, 0], [100, 99], [100, 99]], dtype=torch.int32, device="mps"
    )
    touch_ticks = torch.zeros((3, 2), dtype=torch.int32, device="mps")
    touch_nearest_ticks = torch.zeros(3, dtype=torch.int32, device="mps")
    touch_min_qty_bits = torch.zeros(3, dtype=torch.int32, device="mps")
    touch_min_qty_relation = torch.zeros(3, dtype=torch.int32, device="mps")
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 0] = 1.0
    coin_settings[0, 1] = 1.0
    coin_settings[0, 2] = 1.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = torch.full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    hsl_params = torch.tensor(
        [1.0, 0.2, 60.0, 0.0, 1.0, 1.0, 0.5, 0.75, 0.0, 1.0, 1.0],
        dtype=torch.float32,
        device="mps",
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device="mps")
    output = torch.zeros(16, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_side_fill_pass_probe(
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
    torch.mps.synchronize()

    np.testing.assert_allclose(
        output.cpu().numpy(),
        [1.0, 1000.0, 0.0, 4.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 2.0, 2.0, 1.0, 0.4, 4.0],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    TrailingMartingaleMulticoinSideConfig long_config;
    TrailingMartingaleMulticoinSideConfig short_config;
    long_config.coin_hsl_mode = true;
    short_config.coin_hsl_mode = false;
    long_config.volume_drop = 0.0f;
    short_config.volume_drop = 0.0f;
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
    long_side.selection_initialized = false;
    short_side.selection_initialized = false;
    long_side.previous_effective_n_positions = 0;
    short_side.previous_effective_n_positions = 0;
    long_side.coin_hsl_entry_blocked_mask = 0ul;
    short_side.coin_hsl_entry_blocked_mask = 0ul;
    long_side.one_way_initial_blocked_mask = 0ul;
    short_side.one_way_initial_blocked_mask = 0ul;
    for (int c = 0; c < 3; ++c) {
        long_side.psize[c] = 0.0f;
        short_side.psize[c] = 0.0f;
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
        long_side.coin_hsl[c].orange_graceful_stop = false;
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
    long_side.coin_hsl[0].tier = 2;
    long_side.coin_hsl[0].orange_graceful_stop = true;
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
    output[6] = long_side.selection_initialized ? 1.0f : 0.0f;
    output[7] = short_side.selection_initialized ? 1.0f : 0.0f;
    output[8] = float(long_side.previous_effective_n_positions);
    output[9] = float(short_side.previous_effective_n_positions);
    output[10] = float(long_side.coin_hsl_entry_blocked_mask);
    output[11] = float(short_side.coin_hsl_entry_blocked_mask);
    // Mimic a new opposite-held block on short's highest-ranked coin.
    // The changed mask must force reselection onto the next candidate.
    update_tm_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        1, 3, true, false, 1, 0.0f, 4ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[12 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
    output[15] = float(short_side.one_way_initial_blocked_mask);
    output[16] = float(long_selection_blocked_mask);
    output[17] = float(short_selection_blocked_mask);
    output[18] = float(long_order_blocked_mask);
    output[19] = float(short_order_blocked_mask);
    // Coin one is the current flat incumbent, but its validity window ended
    // at k=1. With no fill or mask change, k=2 must still evict it and select
    // the only surviving unblocked candidate.
    update_tm_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        2, 3, true, false, 1, 0.0f, 4ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[20 + c] = short_side.selected[c] ? 1.0f : 0.0f;
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
        device="mps",
    )
    coin_settings = torch.zeros((3, 13), dtype=torch.float32, device="mps")
    coin_settings[:, 7] = 10.0
    coin_settings[1, 7] = 1.0
    coin_overrides = torch.full(
        (3, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    output = torch.zeros(23, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_selection_phase_probe(
        bars, coin_settings, coin_overrides, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        7.0,
        1.0,
        0.0,
        0.0,
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "entry_initial_ema_dist"
    )] = 0.01
    row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("gate_initial")] = 1.0
    short_row = list(row)
    short_row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "entry_initial_qty_pct"
    )] = 0.5
    params = torch.tensor([row, short_row], dtype=torch.float32, device="mps")
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[101.0, 99.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device="mps",
    )
    touch_ticks = torch.tensor(
        [[[10_000, 10_000]], [[10_000, 10_000]]],
        dtype=torch.int32,
        device="mps",
    )
    touch_nearest_ticks = torch.full(
        (2, 1), 10_000, dtype=torch.int32, device="mps"
    )
    touch_min_qty_bits = torch.zeros((2, 1), dtype=torch.int32, device="mps")
    touch_min_qty_relation = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_settings[0, 9] = 100.0
    coin_settings[0, 10] = 1.0
    coin_overrides = torch.full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    output = torch.zeros(11, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
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
    torch.mps.synchronize()

    values = output.cpu().numpy()
    assert values[0] > values[1] > 0.0
    assert values[2] < values[3]
    assert values[4] == 0.0
    assert values[5] == 0.0
    assert values[6] == 1_000.0
    assert values[7] == 0.0
    assert values[8:].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_dual_hsl_phase_covers_all_signal_topologies():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_dual_hsl_phase_probe(
    constant float* params,
    constant float* bars,
    constant float* invalid_bars,
    constant float* coin_settings,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b >= 8) return;
    int po = int(b) * 22;
    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    TrailingMartingaleMulticoinSideConfig long_config;
    TrailingMartingaleMulticoinSideConfig short_config;
    long_side.hsl = load_hsl(params, po, 0);
    short_side.hsl = load_hsl(params, po + 11, 0);
    long_config.coin_hsl_mode =
        long_side.hsl.signal_mode == HSL_SIGNAL_COIN;
    short_config.coin_hsl_mode =
        short_side.hsl.signal_mode == HSL_SIGNAL_COIN;
    long_side.coin_hsl[0] = long_side.hsl;
    short_side.coin_hsl[0] = short_side.hsl;
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;
    long_side.psize[0] = (b == 4 || b == 5) ? 1.0f : 0.0f;
    short_side.psize[0] = 0.0f;
    long_side.pprice[0] = 100.0f;
    short_side.pprice[0] = 100.0f;
    long_side.entry_qty[0] = 0.0f;
    short_side.entry_qty[0] = 0.0f;
    long_side.close_qty[0] = 0.0f;
    short_side.close_qty[0] = 0.0f;
    long_side.secondary_close_qty[0] = 0.0f;
    short_side.secondary_close_qty[0] = 0.0f;

    JointPortfolioAccount account = init_joint_portfolio_account(100.0f);
    account.realized_pnl_total = 0.0f;
    account.realized_pnl_long = 0.0f;
    account.realized_pnl_short = 0.0f;
    account.balance = 100.0f;
    bool all_valid = true;
    bool sample_enabled = false;
    int sampled_tier = 0;
    float triggers_while_long_open = -1.0f;
    for (int k = 1; k <= 5; ++k) {
        if (k == 2) {
            account.realized_pnl_total = -10.0f;
            account.realized_pnl_long = b == 7 ? 0.0f : -10.0f;
            account.realized_pnl_short = b == 7 ? -10.0f : 0.0f;
            account.balance = 90.0f;
            long_side.coin_realized_pnl[0] = b == 7 ? 0.0f : -10.0f;
            short_side.coin_realized_pnl[0] = b == 7 ? -10.0f : 0.0f;
        }
        if (b == 4 && k == 4) long_side.psize[0] = 0.0f;
        constant float* selected_bars = b == 5 ? invalid_bars : bars;
        int long_slots = b == 7 ? 0 : 1;
        int short_slots = b == 6 ? 0 : 1;
        bool valid = update_tm_multicoin_dual_side_hsl(
            long_side, long_config, long_slots,
            short_side, short_config, short_slots,
                account, selected_bars, coin_settings, k, 0, 1,
            100.0f, 60000.0f, sample_enabled, sampled_tier
        );
        all_valid = all_valid && valid;
        if (b == 4 && k == 3) {
            triggers_while_long_open = long_side.hsl.triggers;
        }
    }
    int oo = int(b) * 12;
    output[oo + 0] = all_valid ? 1.0f : 0.0f;
    output[oo + 1] = sample_enabled ? 1.0f : 0.0f;
    output[oo + 2] = float(sampled_tier);
    output[oo + 3] = float(long_side.hsl.tier);
    output[oo + 4] = float(short_side.hsl.tier);
    output[oo + 5] = long_side.hsl.triggers;
    output[oo + 6] = short_side.hsl.triggers;
    output[oo + 7] = float(long_side.coin_hsl[0].tier);
    output[oo + 8] = float(short_side.coin_hsl[0].tier);
    output[oo + 9] = long_side.coin_hsl[0].triggers;
    output[oo + 10] = short_side.coin_hsl[0].triggers;
    output[oo + 11] = triggers_while_long_open;
}
"""

    def controller(mode):
        return [
            1.0,
            0.05,
            1.0,
            0.0,
            1.0,
            2.0,
            0.5,
            0.75,
            0.0,
            float(mode),
            1.0,
        ]

    params = torch.tensor(
        [
            controller(0) + controller(0),
            controller(1) + controller(1),
            controller(2) + controller(2),
            controller(0) + controller(1),
            controller(0) + controller(0),
            controller(0) + controller(0),
            controller(2) + controller(2),
            controller(2) + controller(2),
        ],
        dtype=torch.float32,
        device="mps",
    )
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 0.0]]] * 6,
        dtype=torch.float32,
        device="mps",
    )
    invalid_bars = bars.clone()
    invalid_bars[:, 0, 2] = float("nan")
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 4] = 1.0
    coin_settings[0, 6] = 0.0
    coin_settings[0, 7] = 5.0
    output = torch.zeros((8, 12), dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py() + probe_kernel
    )
    library.passivbot_tm_multicoin_dual_hsl_phase_probe(
        params, bars, invalid_bars, coin_settings, output, threads=(8, 1, 1)
    )
    torch.mps.synchronize()
    values = output.cpu().numpy()

    # Unified mode consumes the shared account signal and requires portfolio
    # flatness before either controller may halt.
    assert values[0, :7].tolist() == [1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0]
    # Pside mode isolates directional strategy PnL.
    assert values[1, :7].tolist() == [1.0, 1.0, 3.0, 3.0, 0.0, 1.0, 0.0]
    # Coin mode advances the per-coin controllers, not the pside templates.
    assert values[2, :7].tolist() == [1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    assert values[2, 7:11].tolist() == [3.0, 0.0, 1.0, 0.0]
    # Mixed topologies fail closed without advancing state.
    assert values[3, :7].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # One open long position blocks both unified controllers until the whole
    # account is flat for two consecutive samples.
    assert values[4, 5:7].tolist() == [1.0, 1.0]
    assert values[4, 11] == 0.0
    # A held coin without a valid mark rejects the sample before mutating any
    # controller instead of fabricating neutral unrealized PnL.
    assert values[5, :11].tolist() == [0.0] * 11
    # Coin HSL skips a side without an effective slot budget while retaining
    # the active side's controller, matching exact Rust asymmetric tradability.
    assert values[6, :11].tolist() == [
        1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0
    ]
    assert values[7, :11].tolist() == [
        1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0
    ]
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_multicoin_dual_hsl_phase_covers_all_signal_topologies():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_dual_hsl_phase_probe(
    constant float* params,
    constant float* bars,
    constant float* invalid_bars,
    constant float* coin_settings,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b >= 8) return;
    int po = int(b) * 22;
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    EmaMulticoinSideConfig long_config;
    EmaMulticoinSideConfig short_config;
    long_side.hsl = load_hsl(params, po, 0);
    short_side.hsl = load_hsl(params, po + 11, 0);
    long_config.coin_hsl_mode =
        long_side.hsl.signal_mode == HSL_SIGNAL_COIN;
    short_config.coin_hsl_mode =
        short_side.hsl.signal_mode == HSL_SIGNAL_COIN;
    long_side.coin_hsl[0] = long_side.hsl;
    short_side.coin_hsl[0] = short_side.hsl;
    long_side.coin_realized_pnl[0] = 0.0f;
    short_side.coin_realized_pnl[0] = 0.0f;
    long_side.psize[0] = (b == 4 || b == 5) ? 1.0f : 0.0f;
    short_side.psize[0] = 0.0f;
    long_side.pprice[0] = 100.0f;
    short_side.pprice[0] = 100.0f;
    long_side.entry_qty[0] = 0.0f;
    short_side.entry_qty[0] = 0.0f;
    long_side.close_qty[0] = 0.0f;
    short_side.close_qty[0] = 0.0f;
    long_side.secondary_close_qty[0] = 0.0f;
    short_side.secondary_close_qty[0] = 0.0f;

    JointPortfolioAccount account = init_joint_portfolio_account(100.0f);
    account.realized_pnl_total = 0.0f;
    account.realized_pnl_long = 0.0f;
    account.realized_pnl_short = 0.0f;
    account.balance = 100.0f;
    bool all_valid = true;
    bool sample_enabled = false;
    int sampled_tier = 0;
    float triggers_while_long_open = -1.0f;
    for (int k = 1; k <= 5; ++k) {
        if (k == 2) {
            account.realized_pnl_total = -10.0f;
            account.realized_pnl_long = b == 7 ? 0.0f : -10.0f;
            account.realized_pnl_short = b == 7 ? -10.0f : 0.0f;
            account.balance = 90.0f;
            long_side.coin_realized_pnl[0] = b == 7 ? 0.0f : -10.0f;
            short_side.coin_realized_pnl[0] = b == 7 ? -10.0f : 0.0f;
        }
        if (b == 4 && k == 4) long_side.psize[0] = 0.0f;
        constant float* selected_bars = b == 5 ? invalid_bars : bars;
        int long_slots = b == 7 ? 0 : 1;
        int short_slots = b == 6 ? 0 : 1;
        bool valid = update_ema_multicoin_dual_side_hsl(
            long_side, long_config, long_slots,
            short_side, short_config, short_slots,
                account, selected_bars, coin_settings, k, 0, 1,
            100.0f, 60000.0f, sample_enabled, sampled_tier
        );
        all_valid = all_valid && valid;
        if (b == 4 && k == 3) {
            triggers_while_long_open = long_side.hsl.triggers;
        }
    }
    int oo = int(b) * 12;
    output[oo + 0] = all_valid ? 1.0f : 0.0f;
    output[oo + 1] = sample_enabled ? 1.0f : 0.0f;
    output[oo + 2] = float(sampled_tier);
    output[oo + 3] = float(long_side.hsl.tier);
    output[oo + 4] = float(short_side.hsl.tier);
    output[oo + 5] = long_side.hsl.triggers;
    output[oo + 6] = short_side.hsl.triggers;
    output[oo + 7] = float(long_side.coin_hsl[0].tier);
    output[oo + 8] = float(short_side.coin_hsl[0].tier);
    output[oo + 9] = long_side.coin_hsl[0].triggers;
    output[oo + 10] = short_side.coin_hsl[0].triggers;
    output[oo + 11] = triggers_while_long_open;
}
"""

    def controller(mode):
        return [
            1.0,
            0.05,
            1.0,
            0.0,
            1.0,
            2.0,
            0.5,
            0.75,
            0.0,
            float(mode),
            1.0,
        ]

    params = torch.tensor(
        [
            controller(0) + controller(0),
            controller(1) + controller(1),
            controller(2) + controller(2),
            controller(0) + controller(1),
            controller(0) + controller(0),
            controller(0) + controller(0),
            controller(2) + controller(2),
            controller(2) + controller(2),
        ],
        dtype=torch.float32,
        device="mps",
    )
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 0.0]]] * 6,
        dtype=torch.float32,
        device="mps",
    )
    invalid_bars = bars.clone()
    invalid_bars[:, 0, 2] = float("nan")
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 4] = 1.0
    coin_settings[0, 6] = 0.0
    coin_settings[0, 7] = 5.0
    output = torch.zeros((8, 12), dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_dual_hsl_phase_probe(
        params, bars, invalid_bars, coin_settings, output, threads=(8, 1, 1)
    )
    torch.mps.synchronize()
    values = output.cpu().numpy()

    # Unified mode consumes the shared account signal and requires portfolio
    # flatness before either controller may halt.
    assert values[0, :7].tolist() == [1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0]
    # Pside mode isolates directional strategy PnL.
    assert values[1, :7].tolist() == [1.0, 1.0, 3.0, 3.0, 0.0, 1.0, 0.0]
    # Coin mode advances the per-coin controllers, not the pside templates.
    assert values[2, :7].tolist() == [1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    assert values[2, 7:11].tolist() == [3.0, 0.0, 1.0, 0.0]
    # Mixed topologies fail closed without advancing state.
    assert values[3, :7].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # One open long position blocks both unified controllers until the whole
    # account is flat for two consecutive samples.
    assert values[4, 5:7].tolist() == [1.0, 1.0]
    assert values[4, 11] == 0.0
    # A held coin without a valid mark rejects the sample before mutating any
    # controller instead of fabricating neutral unrealized PnL.
    assert values[5, :11].tolist() == [0.0] * 11
    # Coin HSL skips a side without an effective slot budget while retaining
    # the active side's controller, matching exact Rust asymmetric tradability.
    assert values[6, :11].tolist() == [
        1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0
    ]
    assert values[7, :11].tolist() == [
        1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    EmaMulticoinSideConfig long_config;
    EmaMulticoinSideConfig short_config;
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
        device="mps",
    )
    coin_settings = torch.zeros((2, 13), dtype=torch.float32, device="mps")
    coin_settings[:, 4] = 1.0
    coin_settings[:, 7] = 10.0
    coin_overrides = torch.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device="mps",
    )
    coin_overrides[1, 11] = 0.0
    hour_log_ranges = torch.tensor(
        [[-1.0, -1.0], [np.log(105.0 / 95.0), np.log(205.0 / 195.0)]],
        dtype=torch.float32,
        device="mps",
    )
    output = torch.zeros(12, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
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
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    TrailingMartingaleMulticoinSideConfig long_config;
    TrailingMartingaleMulticoinSideConfig short_config;
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
        device="mps",
    )
    coin_settings = torch.zeros((2, 13), dtype=torch.float32, device="mps")
    coin_settings[:, 4] = 1.0
    coin_settings[:, 7] = 10.0
    coin_overrides = torch.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    coin_overrides[1, 24] = 0.0
    hour_log_ranges = torch.full(
        (2, 2), -1.0, dtype=torch.float32, device="mps"
    )
    hour_log_ranges[1, 0] = np.log(105.0 / 95.0)
    output = torch.zeros(20, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )
    library.passivbot_tm_multicoin_candle_helpers_probe(
        bars,
        hour_log_ranges,
        coin_settings,
        coin_overrides,
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_multicoin_fill_phase_shares_account_across_sides():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_ema_multicoin_shared_fill_phase_probe(
    constant float* bars,
    constant int* fill_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* coin_fill_counts,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
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
        long_side, account, fills,
        bars, fill_ticks, coin_settings, coin_overrides,
        coin_fill_counts, 0, 1, 1, false, true,
        false, 0.0f, false, 1000.0f
    );
    float balance_after_long = account.balance;
    bool short_filled = process_ema_multicoin_side_fills(
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
        device="mps",
    )
    fill_ticks = torch.tensor(
        [[[100, 100]], [[110, 90]]], dtype=torch.int32, device="mps"
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 0] = 1.0
    coin_settings[0, 1] = 1.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 5] = 0.01
    coin_settings[0, 7] = 10.0
    coin_overrides = torch.full(
        (1, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device="mps",
    )
    coin_fill_counts = torch.zeros(1, dtype=torch.float32, device="mps")
    output = torch.zeros(15, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_shared_fill_phase_probe(
        bars,
        fill_ticks,
        coin_settings,
        coin_overrides,
        coin_fill_counts,
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    EmaMulticoinSideConfig long_config;
    EmaMulticoinSideConfig short_config;
    long_config.coin_hsl_mode = false;
    short_config.coin_hsl_mode = false;
    long_config.volume_drop = 0.0f;
    short_config.volume_drop = 0.0f;
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
    long_side.selection_initialized = false;
    short_side.selection_initialized = false;
    long_side.previous_effective_n_positions = 0;
    short_side.previous_effective_n_positions = 0;
    long_side.one_way_initial_blocked_mask = 0ul;
    short_side.one_way_initial_blocked_mask = 0ul;
    for (int c = 0; c < 3; ++c) {
        long_side.psize[c] = 0.0f;
        short_side.psize[c] = 0.0f;
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
    output[6] = long_side.selection_initialized ? 1.0f : 0.0f;
    output[7] = short_side.selection_initialized ? 1.0f : 0.0f;
    output[8] = float(long_side.previous_effective_n_positions);
    output[9] = float(short_side.previous_effective_n_positions);
    // Opposite-held eligibility changes outside the side's own fills.
    // The mask transition must evict the blocked incumbent and promote next.
    update_ema_multicoin_side_selection(
        short_side, short_config, bars, coin_settings, coin_overrides,
        1, 3, true, false, 1, 0.0f, 4ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[10 + c] = short_side.selected[c] ? 1.0f : 0.0f;
    }
    output[13] = float(short_side.one_way_initial_blocked_mask);
    output[14] = float(long_selection_blocked_mask);
    output[15] = float(short_selection_blocked_mask);
    output[16] = float(long_order_blocked_mask);
    output[17] = float(short_order_blocked_mask);

    // No fill or mask transition occurs at k=2, but coin zero has reached
    // its tail. The validity transition must evict the flat incumbent.
    update_ema_multicoin_side_selection(
        long_side, long_config, bars, coin_settings, coin_overrides,
        2, 3, false, false, 1, 0.0f, 0ul, false, 0.0f
    );
    for (int c = 0; c < 3; ++c) {
        output[19 + c] = long_side.selected[c] ? 1.0f : 0.0f;
    }

    // With an existing position, the caller cannot prove a positive exact
    // cash-balance floor. A zero lower bound must keep every other flat slot
    // out of the Forager selection even when its exchange minimum is small.
    long_config.twel = 1.0f;
    long_config.allowance_pct = 0.0f;
    long_config.legacy_raw_allowance = false;
    long_config.base_qty_pct = 1.0f;
    long_side.selection_initialized = false;
    long_side.previous_effective_n_positions = 0;
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
    output[18] = long_side.selected[1] || long_side.selected[2] ? 1.0f : 0.0f;
}
"""

    bars = torch.tensor(
        [
            [[100.0, 100.0, 100.0, 1.0]] * 3,
            [[100.0, 100.0, 100.0, 1.0]] * 3,
            [[100.0, 100.0, 100.0, 1.0]] * 3,
        ],
        dtype=torch.float32,
        device="mps",
    )
    coin_settings = torch.zeros((3, 13), dtype=torch.float32, device="mps")
    coin_settings[:, 7] = 10.0
    coin_settings[0, 7] = 1.0
    coin_settings[:, 12] = 1.0
    coin_overrides = torch.full(
        (3, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device="mps",
    )
    output = torch.zeros(22, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
        passivbot_rust.mps_ema_anchor_multicoin_source_py() + probe_kernel
    )
    library.passivbot_ema_multicoin_selection_phase_probe(
        bars, coin_settings, coin_overrides, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        7.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    params = torch.tensor([row, short_row], dtype=torch.float32, device="mps")
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[101.0, 99.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device="mps",
    )
    touch_ticks = torch.tensor(
        [[[10_000, 10_000]], [[10_000, 10_000]]],
        dtype=torch.int32,
        device="mps",
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
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
        device="mps",
    )
    output = torch.zeros(11, dtype=torch.float32, device="mps")

    library = torch.mps.compile_shader(
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
    torch.mps.synchronize()

    values = output.cpu().numpy()
    assert values[0] > values[1] > 0.0
    assert values[2] < values[3]
    assert values[4] == 0.0
    assert values[5] == 0.0
    assert values[6] == 1_000.0
    assert values[7] == 0.0
    assert values[8:].tolist() == [0.0, 0.0, 0.0]


def _single_coin_exposure_fields(
    *, allowance_pct=0.0, legacy_raw=False, entry_gate=True, threshold=1.0
):
    return [
        allowance_pct,
        float(legacy_raw),
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
    "hsl_ema_span_minutes": 60.0,
    "hsl_cooldown_minutes_after_red": 0.0,
    "hsl_no_restart_drawdown_threshold": 1.0,
    "hsl_restart_policy": 1.0,
    "hsl_tier_ratio_yellow": 0.5,
    "hsl_tier_ratio_orange": 0.75,
    "hsl_orange_graceful_stop": 0.0,
    "hsl_signal_mode": 0.0,
    "hsl_slot_count": 1.0,
}


def _single_coin_param_row(values, keys):
    merged = {
        **_UNSTUCK_DISABLED_VALUES,
        **_HSL_DISABLED_VALUES,
        "wallet_exposure_limit": -1.0,
        **values,
    }
    return [merged[key] for key in keys]


@pytest.mark.parametrize(
    ("keys", "minute_span_key", "hour_span_key"),
    [
        (
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
            "offset_volatility_ema_span_1m",
            "offset_volatility_ema_span_1h",
        ),
        (
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
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
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 7.5,
        }
    )
    original = np.asarray(
        [[side[key] for key in keys] * 2], dtype=np.float64
    )

    scaled = _scale_single_coin_minute_parameters(
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
                offset + keys.index("hsl_cooldown_minutes_after_red"),
            ]
            == 1.5
        )
        assert scaled[0, offset + keys.index(hour_span_key)] == 24.0
        hsl_span = scaled[0, offset + keys.index("hsl_ema_span_minutes")]
        alpha_1m = 2.0 / 61.0
        alpha_5m = 2.0 / (hsl_span + 1.0)
        assert alpha_5m == pytest.approx(1.0 - (1.0 - alpha_1m) ** 5)


@pytest.mark.parametrize("span", [1.0, 5.5, 60.0, 720.0])
def test_single_coin_interval_packing_compounds_hsl_elapsed_minute_decay(span):
    side = {
        key: float(index + 1)
        for index, key in enumerate(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    }
    side["hsl_ema_span_minutes"] = span
    original = np.asarray(
        [[side[key] for key in EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS] * 2],
        dtype=np.float64,
    )

    scaled = _scale_single_coin_minute_parameters(
        original,
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        sides=2,
        interval_minutes=5.0,
    )

    alpha_1m = 2.0 / (span + 1.0)
    for side_index in range(2):
        offset = side_index * len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
        effective_span = scaled[
            0,
            offset + EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS.index(
                "hsl_ema_span_minutes"
            ),
        ]
        alpha_5m = 2.0 / (effective_span + 1.0)
        assert alpha_5m == pytest.approx(1.0 - (1.0 - alpha_1m) ** 5)


def test_single_coin_interval_packing_preserves_one_minute_hsl_span_exactly():
    values = np.arange(
        1,
        len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS) * 2 + 1,
        dtype=np.float64,
    ).reshape(1, -1)
    for side_index in range(2):
        values[
            0,
            side_index * len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
            + EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS.index("hsl_ema_span_minutes"),
        ] = 5.5

    scaled = _scale_single_coin_minute_parameters(
        values,
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        sides=2,
        interval_minutes=1.0,
    )

    assert np.array_equal(scaled, values)


def test_single_coin_interval_packing_matches_exact_rust_hsl_elapsed_decay():
    import passivbot_rust

    side = {
        key: float(index + 1)
        for index, key in enumerate(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    }
    side["hsl_ema_span_minutes"] = 60.0
    original = np.asarray(
        [[side[key] for key in EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS] * 2],
        dtype=np.float64,
    )
    scaled = _scale_single_coin_minute_parameters(
        original,
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        sides=2,
        interval_minutes=5.0,
    )
    effective_span = scaled[
        0,
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS.index("hsl_ema_span_minutes"),
    ]
    exact = passivbot_rust.EquityHardStopRuntime()
    packed = passivbot_rust.EquityHardStopRuntime()
    common = {
        "red_threshold": 0.25,
        "tier_ratio_yellow": 0.5,
        "tier_ratio_orange": 0.75,
    }
    exact.apply_sample(
        timestamp_ms=0,
        equity=100.0,
        peak_strategy_equity=100.0,
        ema_span_minutes=60.0,
        **common,
    )
    packed.apply_sample(
        timestamp_ms=0,
        equity=100.0,
        peak_strategy_equity=100.0,
        ema_span_minutes=effective_span,
        **common,
    )

    exact_step = exact.apply_sample(
        timestamp_ms=5 * 60_000,
        equity=90.0,
        peak_strategy_equity=100.0,
        ema_span_minutes=60.0,
        **common,
    )
    packed_step = packed.apply_sample(
        timestamp_ms=60_000,
        equity=90.0,
        peak_strategy_equity=100.0,
        ema_span_minutes=effective_span,
        **common,
    )

    assert exact_step["drawdown_ema"] == pytest.approx(
        packed_step["drawdown_ema"]
    )


def test_single_coin_interval_packing_rejects_invalid_interval():
    values = np.zeros(
        (1, len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS) * 2), dtype=np.float64
    )

    with pytest.raises(ValueError, match="at least one minute"):
        _scale_single_coin_minute_parameters(
            values,
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
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
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 35.0,
        }
    )
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
            ("hsl_cooldown_minutes_after_red", 7.0),
        ):
            assert scaled[0, offset + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(key)] == expected
        assert scaled[
            0,
            offset
            + EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(
                "offset_volatility_ema_span_1h"
            ),
        ] == 24.0


def test_multicoin_ema_interval_packing_scales_only_finite_coin_overrides():
    overrides = np.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float64
    )
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
    assert scaled[0, hsl_cooldown_column] == 4.0
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
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 35.0,
        }
    )
    original = np.asarray(
        [
            [
                values[key]
                for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
            ]
            * 2
        ],
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
            ("hsl_cooldown_minutes_after_red", 7.0),
        ):
            assert scaled[
                0,
                offset + TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key),
            ] == expected
        assert scaled[
            0,
            offset
            + TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "volatility_ema_span_1h"
            ),
        ] == 24.0


def test_multicoin_tm_interval_packing_scales_only_finite_coin_overrides():
    overrides = np.full(
        (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float64
    )
    override_keys = tuple(
        key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
    )
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
    assert (
        scaled[0, TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN]
        == 3.0
    )
    assert scaled[0, hsl_cooldown_column] == 4.0
    alpha_7m = 2.0 / (float(scaled[0, hsl_span_column]) + 1.0)
    assert alpha_7m == pytest.approx(1.0 - (1.0 - 2.0 / 61.0) ** 7)
    assert np.isnan(scaled[1]).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_five_minute_shader_smoke(strategy_kind):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    count = 32
    interval_ms = 5 * 60_000
    phase = np.linspace(0.0, 4.0 * np.pi, count)
    close = 100.0 + np.sin(phase)
    high = close * 1.01
    low = close * 0.99
    timestamps = (
        1_700_000_000_000
        + np.arange(count, dtype=np.int64) * interval_ms
    )
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        2,
        2,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        interval_ms,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "ema_anchor":
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 10.0,
                "ema_span_1": 30.0,
                "entry_double_down_factor": 1.5,
                "offset": 0.01,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 60.0,
                "offset_volatility_ema_span_1m": 60.0,
                "entry_cooldown_minutes": 2.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
    else:
        row = _tm_single_row(initial_ema_dist=0.01)
        row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("ema_span_0")] = 10.0
        row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("ema_span_1")] = 30.0
        row[
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "entry_cooldown_minutes"
            )
        ] = 2.0
        runner_cls = MpsTrailingMartingaleRunner
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
    params = np.asarray([row + row], dtype=np.float64)

    runner = runner_cls(
        market, run, data, long_enabled=True, short_enabled=False
    )
    packed = runner._pack_params(params)
    output = runner.run(params)
    torch.mps.synchronize()

    assert runner.interval_minutes == 5.0
    assert packed[0, keys.index("ema_span_0")] == pytest.approx(2.0)
    assert packed[0, keys.index("entry_cooldown_minutes")] == pytest.approx(
        0.4
    )
    packed_hsl_span = packed[0, keys.index("hsl_ema_span_minutes")]
    assert 2.0 / (packed_hsl_span + 1.0) == pytest.approx(
        1.0 - (1.0 - 2.0 / 61.0) ** 5
    )
    assert output["balance"].device.type == "mps"
    assert torch.isfinite(output["balance"]).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("hsl_enabled", [False, True])
@pytest.mark.parametrize("forced_delist", [False, True])
def test_mps_single_coin_invalid_tail_matches_forced_delist_boundary(
    strategy_kind, side, hsl_enabled, forced_delist
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    last_valid = 6
    count = last_valid + 1401 if forced_delist else 10
    close = np.full(count, np.nan)
    high = np.full(count, np.nan)
    low = np.full(count, np.nan)
    close[: last_valid + 1] = 100.0
    high[: last_valid + 1] = 100.0
    low[: last_valid + 1] = 100.0
    if side == "long":
        low[3] = 98.0
        high[last_valid] = 80.0
        low[last_valid] = 80.0
        close[last_valid] = 80.0
    else:
        high[3] = 102.0
        high[last_valid] = 120.0
        low[last_valid] = 120.0
        close[last_valid] = 120.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
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
        last_valid,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "ema_anchor":
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 0.0,
                "offset": 0.01,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 100.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
    else:
        row = _tm_single_row(initial_ema_dist=0.01)
        row[
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "entry_cooldown_minutes"
            )
        ] = 100.0
        runner_cls = MpsTrailingMartingaleRunner
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS

    if hsl_enabled:
        for key, value in {
            "hsl_enabled": 1.0,
            "hsl_red_threshold": 0.005,
            "hsl_ema_span_minutes": 1.0,
            "hsl_cooldown_minutes_after_red": 0.0,
            "hsl_no_restart_drawdown_threshold": 1.0,
            "hsl_restart_policy": 2.0,
            "hsl_tier_ratio_yellow": 0.5,
            "hsl_tier_ratio_orange": 0.75,
            "hsl_orange_graceful_stop": 0.0,
            "hsl_signal_mode": 0.0,
            "hsl_slot_count": 1.0,
        }.items():
            row[keys.index(key)] = value

    runner_kwargs = {}
    if strategy_kind == "trailing_martingale":
        runner_kwargs["hsl_enabled"] = hsl_enabled

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        **runner_kwargs,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    if forced_delist:
        assert output[size_key].item() == 0.0
        assert output["fill_count"].item() >= 2.0
        assert output["hsl_panic_close_loss_sum"].item() > 0.0
    else:
        assert output[size_key].item() > 0.0
    assert output["day_end_eq"][0, 0].item() == pytest.approx(
        output["balance"].item()
    )
    if hsl_enabled:
        expected_hsl_samples = (
            (output["last_eq_ts"] - output["first_eq_ts"]) / 60_000.0 + 1.0
        )
        assert output["hsl_tier_samples_total"].item() == pytest.approx(
            expected_hsl_samples.item()
        )
        assert output["hsl_tier_samples_red"].item() > 0.0
        assert (
            output["hsl_tier_samples_red"].item()
            < output["hsl_tier_samples_total"].item()
        )
        if not forced_delist:
            assert output[f"hsl_triggers_{side}"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_forced_delist_closes_both_hedged_sides(strategy_kind):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    last_valid = 6
    count = last_valid + 1401
    close = np.full(count, np.nan)
    high = np.full(count, np.nan)
    low = np.full(count, np.nan)
    close[: last_valid + 1] = 100.0
    high[: last_valid + 1] = 100.0
    low[: last_valid + 1] = 100.0
    high[3] = 102.0
    low[3] = 98.0
    close[last_valid] = 90.0
    high[last_valid] = 90.0
    low[last_valid] = 90.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
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
        last_valid,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "ema_anchor":
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 0.0,
                "offset": 0.01,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 100.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner
        runner_kwargs = {}
    else:
        row = _tm_single_row(initial_ema_dist=0.01)
        row[
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "entry_cooldown_minutes"
            )
        ] = 100.0
        runner_cls = MpsTrailingMartingaleRunner
        runner_kwargs = {"hsl_enabled": False}

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
        **runner_kwargs,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() == 0.0
    assert output["fill_count"].item() >= 4.0
    assert output["fill_count_long"].item() >= 2.0
    assert output["fill_count"].item() - output["fill_count_long"].item() >= 2.0
    assert output["loss_sum_long"].item() > 0.0
    assert output["profit_sum_short"].item() > 0.0
    assert output["hsl_panic_close_loss_sum"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_single_coin_recovery_samples_internal_nan_candle(
    strategy_kind, topology
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    count = 123
    invalid_index = 61
    close = np.full(count, 100.0)
    high = close.copy()
    low = close.copy()
    high[invalid_index] = np.nan
    low[invalid_index] = np.nan
    close[invalid_index] = np.nan
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "ema_anchor":
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 0.0,
                "offset": 0.5,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner
        runner_kwargs = {}
    else:
        row = _tm_single_row(initial_ema_dist=0.5)
        runner_cls = MpsTrailingMartingaleRunner
        runner_kwargs = {"hsl_enabled": False}

    sides = ("long", "short") if topology == "fused" else (topology,)
    output = runner_cls(
        market,
        run,
        data,
        long_enabled="long" in sides,
        short_enabled="short" in sides,
        hedge_mode=topology == "fused",
        recovery_distribution_enabled=True,
        **runner_kwargs,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    samples = output["strategy_eq_recovery_samples"][0].cpu().numpy()
    assert samples[:3].tolist() == pytest.approx([1_000.0, 1_000.0, 1_000.0])
    assert np.isnan(samples[3:]).all()
    assert output["last_eq_ts"].item() == pytest.approx(121 * run.interval_ms)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_internal_nan_candle_clears_pending_entry(strategy_kind):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    count = 5
    close = np.full(count, 100.0)
    high = close.copy()
    low = close.copy()
    high[2] = np.nan
    low[2] = np.nan
    close[2] = np.nan
    low[3] = 50.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "ema_anchor":
        side = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 0.0,
                "offset": 0.1,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner
        runner_kwargs = {}
    else:
        side = _tm_single_row(initial_ema_dist=0.1)
        runner_cls = MpsTrailingMartingaleRunner
        runner_kwargs = {"hsl_enabled": False}

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        hedge_mode=False,
        **runner_kwargs,
    ).run(np.asarray([side + side], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_recovery_samples_all_internal_nan_candle(
    strategy_kind, topology
):
    count = 123
    invalid_index = 61
    closes = np.full((count, 2), 100.0)
    highs = closes.copy()
    lows = closes.copy()
    if topology != "fused":
        highs[3, :] = 101.0
        lows[3, :] = 99.0
    highs[invalid_index, :] = np.nan
    lows[invalid_index, :] = np.nan
    closes[invalid_index, :] = np.nan
    fixture_side = topology if topology != "fused" else "long"
    runner, row, run, data = _multicoin_exposure_fixture(
        strategy_kind,
        fixture_side,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        recovery_distribution_enabled=True,
        return_context=True,
    )
    if topology == "fused":
        runner_cls = (
            MpsEmaAnchorMulticoinFusedRunner
            if strategy_kind == "ema_anchor"
            else MpsTrailingMartingaleMulticoinFusedRunner
        )
        runner = runner_cls(
            run,
            data,
            recovery_distribution_enabled=True,
        )
        params = row + row
    else:
        params = row

    assert torch.isnan(data["bars"][invalid_index, :, 2]).all().item()
    output = runner.run(np.asarray([params], dtype=np.float64))
    torch.mps.synchronize()

    samples = output["strategy_eq_recovery_samples"][0].cpu().numpy()
    if topology == "fused":
        assert output["fill_count"].item() == 0.0
    else:
        assert output["fill_count"].item() > 0.0
        assert (
            output["psize"].item() > 0.0
            or output.get("short_psize", torch.zeros(1)).item() > 0.0
        )
    assert np.isfinite(samples[:3]).all()
    assert samples[1] == pytest.approx(output["balance"].item())
    assert np.isnan(samples[3:]).all()
    assert output["last_eq_ts"].item() == pytest.approx(121 * run.interval_ms)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("inactive_hsl_enabled", [False, True])
def test_mps_single_coin_service_dispatches_forced_delist_tail(
    strategy_kind, inactive_hsl_enabled
):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import MpsSingleCoinProxy

    entry_k = 7
    last_valid = 10
    count = last_valid + 1401
    hlcvs = np.full((count, 1, 4), np.nan, dtype=np.float64)
    hlcvs[: last_valid + 1, 0] = [100.0, 100.0, 100.0, 1.0]
    hlcvs[entry_k, 0, 1] = 98.0
    hlcvs[last_valid, 0] = [80.0, 80.0, 80.0, 1.0]
    timestamps = (
        1_700_000_000_000
        + np.arange(count, dtype=np.int64) * 60_000
    )
    market_settings = {
        "BTC": {
            "qty_step": 0.001,
            "price_step": 0.01,
            "min_qty": 0.001,
            "min_cost": 5.0,
            "c_mult": 1.0,
            "maker": 0.0,
            "taker": 0.001,
            "exchange": "bybit",
            "first_valid_index": 0,
            "last_valid_index": last_valid,
            "warmup_minutes": 1,
        },
        "__meta__": {
            "requested_start_ts": int(timestamps[0]),
            "requested_start_date": "2023-11-14",
            "warmup_minutes_requested": 1,
        },
    }
    config = get_template_config()
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["max_warmup_minutes"] = 1
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
    config["backtest"]["coins"] = {"bybit": ["BTC"]}
    config["backtest"]["exchanges"] = ["bybit"]
    for side, enabled in (("long", True), ("short", False)):
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 1.0 if enabled else 0.0
        risk["n_positions"] = 1 if enabled else 0
        risk["we_excess_allowance_pct"] = 0.0
        risk["position_exposure_enforcer_enabled"] = False
        risk["total_exposure_enforcer_enabled"] = False
        config["bot"][side]["hsl"]["enabled"] = (
            inactive_hsl_enabled if not enabled else False
        )
        config["bot"][side]["unstuck"]["enabled"] = False
    if strategy_kind == "ema_anchor":
        config["bot"]["long"]["strategy"]["ema_anchor"].update(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "offset": 0.01,
                "offset_psize_weight": 0.0,
            }
        )
    else:
        strategy = config["bot"]["long"]["strategy"]["trailing_martingale"]
        strategy["ema_span_0"] = 2.0
        strategy["ema_span_1"] = 3.0
        strategy["entry"]["initial_qty_pct"] = 0.1
        strategy["entry"]["initial_ema_dist"] = 0.01

    # The first tracked no-position sample sees the higher BTC price, then the
    # first entry fills after the drop. Exact Rust seeds its BTC balance from
    # that first fill and applies it to the earlier sample too.
    btc_prices = np.full(count, 40_000.0)
    btc_prices[:entry_k] = 60_000.0
    proxy = MpsSingleCoinProxy(
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
    assert proxy.runner.shader_topology == "long_no_hsl"
    result = proxy.evaluate([{}])[0]
    exact_fills, _, exact_analysis = run_backtest(
        hlcvs,
        market_settings,
        config,
        "bybit",
        btc_prices,
        timestamps,
    )

    assert len(exact_fills) == exact_analysis["fills_count"]
    assert result["fills_per_day"] == pytest.approx(
        exact_analysis["fills_per_day"], rel=1.0e-5
    )
    assert np.isfinite(result["entry_interval_hours_max"])
    assert np.isfinite(result["strategy_eq_recovery_days_p99"])
    assert result["adg_btc"] == pytest.approx(
        exact_analysis["adg_btc"], rel=2.0e-3
    )
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

    ebd_only_proxy = MpsSingleCoinProxy(
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
        exact_analysis["paper_loss_ratio_btc"], rel=3.0e-3, abs=1.0e-8
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize(
    ("topology", "hedge_mode"),
    [
        ("long", False),
        ("short", False),
        ("fused", True),
        ("fused", False),
    ],
)
def test_mps_single_coin_overrides_shadow_candidates_and_track_exact(
    strategy_kind, topology, hedge_mode
):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import MpsSingleCoinProxy

    count = 80
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.empty((count, 1, 4), dtype=np.float64)
    hlcvs[:, 0] = [110.0, 90.0, 100.0, 1.0]
    market_settings = {
        "ETH": {
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
        },
        "__meta__": {
            "requested_start_ts": int(timestamps[0]),
            "requested_start_date": "2023-11-14",
            "warmup_minutes_requested": 1,
        },
    }
    enabled_sides = (
        ("long", "short") if topology == "fused" else (topology,)
    )
    config = get_template_config()
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["max_warmup_minutes"] = 1
    config["live"]["approved_coins"] = {
        side: ["ETH"] if side in enabled_sides else []
        for side in ("long", "short")
    }
    config["live"]["hedge_mode"] = hedge_mode
    config["backtest"]["coins"] = {"bybit": ["ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.9 if enabled else 0.0
        risk["n_positions"] = 1 if enabled else 0
        risk["entry_cooldown_minutes"] = 0.0
        risk["we_excess_allowance_pct"] = 0.0
        risk["total_exposure_entry_gate_enabled"] = False
        risk["position_exposure_enforcer_enabled"] = False
        risk["total_exposure_enforcer_enabled"] = False
        config["bot"][side]["hsl"]["enabled"] = False
        config["bot"][side]["unstuck"]["enabled"] = False

    side_overrides = {}
    candidate = {}
    for side in enabled_sides:
        if strategy_kind == "ema_anchor":
            strategy = config["bot"][side]["strategy"]["ema_anchor"]
            strategy.update(
                {
                    "base_qty_pct": 0.1,
                    "ema_span_0": 2.0,
                    "ema_span_1": 3.0,
                    "entry_double_down_factor": 0.0,
                    "offset": 0.4,
                    "offset_psize_weight": 0.0,
                }
            )
            strategy_patch = {"offset": 0.01}
            candidate[f"{side}_offset"] = 0.4
            locked_key = "offset"
            locked_value = 0.01
        else:
            strategy = config["bot"][side]["strategy"][
                "trailing_martingale"
            ]
            strategy["ema_span_0"] = 2.0
            strategy["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 0.1
            strategy["entry"]["initial_ema_dist"] = 0.4
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.01
            strategy_patch = {
                "entry": {"initial_ema_dist": 0.01},
            }
            candidate[f"{side}_entry_initial_ema_dist"] = 0.4
            locked_key = "entry_initial_ema_dist"
            locked_value = 0.01
        candidate[f"{side}_entry_cooldown_minutes"] = 0.0
        candidate[f"{side}_total_wallet_exposure_limit"] = 0.9
        candidate[f"{side}_we_excess_allowance_pct"] = 0.0
        candidate[f"{side}_unstuck_close_pct"] = 0.9
        candidate[f"{side}_hsl_red_threshold"] = 0.8
        risk_patch = {
            "entry_cooldown_minutes": 1_000.0,
            "we_excess_allowance_pct": 0.25,
        }
        if strategy_kind == "trailing_martingale":
            risk_patch.update(
                {
                    "position_exposure_enforcer_enabled": False,
                    "position_exposure_enforcer_threshold": 0.7,
                }
            )
            candidate[f"{side}_wel_enforcer_enabled"] = 1.0
            candidate[f"{side}_wel_enforcer_threshold"] = 0.9
        side_overrides[side] = {
            "strategy": {strategy_kind: strategy_patch},
            "risk": risk_patch,
            "wallet_exposure_limit": 0.4,
            "unstuck": {"close_pct": 0.2},
            "hsl": {"red_threshold": 0.2},
        }
    config["coin_overrides"] = {"ETH": {"bot": side_overrides}}

    proxy = MpsSingleCoinProxy(
        config=config,
        hlcvs=hlcvs,
        mss=market_settings,
        btc=np.full(count, 50_000.0),
        timestamps=timestamps,
        exchange="bybit",
        batch_size=1,
        needed_metrics={"fills_per_day"},
    )
    parameter_matrix = proxy._parameter_matrix([candidate])
    side_width = len(proxy.param_keys)
    for side in enabled_sides:
        side_offset = 0 if side == "long" else side_width
        assert parameter_matrix[
            0, side_offset + proxy.param_keys.index(locked_key)
        ] == pytest.approx(locked_value)
        assert parameter_matrix[
            0, side_offset + proxy.param_keys.index("entry_cooldown_minutes")
        ] == pytest.approx(1_000.0)
        assert parameter_matrix[
            0,
            side_offset
            + proxy.param_keys.index("total_wallet_exposure_limit"),
        ] == pytest.approx(0.9)
        assert parameter_matrix[
            0,
            side_offset
            + proxy.param_keys.index("wallet_exposure_limit"),
        ] == pytest.approx(0.4)
        assert parameter_matrix[
            0, side_offset + proxy.param_keys.index("we_excess_allowance_pct")
        ] == pytest.approx(0.25)
        assert parameter_matrix[
            0, side_offset + proxy.param_keys.index("unstuck_close_pct")
        ] == pytest.approx(0.2)
        assert parameter_matrix[
            0, side_offset + proxy.param_keys.index("hsl_red_threshold")
        ] == pytest.approx(0.2)
        if strategy_kind == "trailing_martingale":
            assert parameter_matrix[
                0, side_offset + proxy.param_keys.index("wel_enforcer_enabled")
            ] == pytest.approx(0.0)
            assert parameter_matrix[
                0, side_offset + proxy.param_keys.index("wel_enforcer_threshold")
            ] == pytest.approx(0.7)

    result = proxy.evaluate([candidate])[0]
    _, _, exact_analysis = run_backtest(
        hlcvs,
        market_settings,
        config,
        "bybit",
        np.full(count, 50_000.0),
        timestamps,
    )

    assert result["fills_per_day"] > 0.0
    fill_rate_drift = abs(
        result["fills_per_day"] - exact_analysis["fills_per_day"]
    )
    # TM's float32 screening path may cross one recursive fill boundary that
    # exact Rust's float64 path does not. The exact optimizer validation remains
    # authoritative; this matrix is proving override dispatch and bounded drift.
    fill_count_equivalent_drift = (
        fill_rate_drift * exact_analysis["fills_analysis_duration_days"]
    )
    allowed_fill_drift = 0.0 if strategy_kind == "ema_anchor" else 1.0
    assert fill_count_equivalent_drift <= allowed_fill_drift + 1.0e-9


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_service_dispatches_forced_delist_tail(
    strategy_kind, topology
):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import MpsMulticoinProxy

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
    timestamps = (
        1_700_000_000_000
        + np.arange(count, dtype=np.int64) * 60_000
    )
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
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["max_warmup_minutes"] = 1
    enabled_sides = (
        ("long", "short") if topology == "fused" else (topology,)
    )
    config["live"]["approved_coins"] = {
        side: ["BTC"] if side in enabled_sides else []
        for side in ("long", "short")
    }
    config["live"]["hedge_mode"] = topology == "fused"
    config["backtest"]["coins"] = {"bybit": ["BTC", "ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.1 if enabled else 0.0
        risk["n_positions"] = 1 if enabled else 0
        risk["entry_cooldown_minutes"] = 100.0
        risk["we_excess_allowance_pct"] = 0.0
        risk["position_exposure_enforcer_enabled"] = False
        risk["total_exposure_enforcer_enabled"] = False
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
            strategy = config["bot"][side]["strategy"][
                "trailing_martingale"
            ]
            strategy["ema_span_0"] = 2.0
            strategy["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 1.0
            strategy["entry"]["initial_ema_dist"] = 0.1
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.5

    # Exercise the canonical pre-fill BTC replay before the delayed entry.
    btc_prices = np.full(count, 40_000.0)
    btc_prices[:entry_k] = 60_000.0
    proxy = MpsMulticoinProxy(
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
    expected_panic_types = {
        f"close_panic_{side}" for side in enabled_sides
    }
    assert expected_panic_types <= exact_order_types
    assert result["adg_btc"] == pytest.approx(
        exact_analysis["adg_btc"], rel=2.0e-3
    )
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
        ebd_only_proxy = MpsMulticoinProxy(
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize(
    "topology", ["long", "short", "fused-hedge", "fused-one-way"]
)
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
    from optimization.gpu.service import MpsMulticoinProxy

    hlcvs = np.full((count, 2, 4), 100.0, dtype=np.float64)
    hlcvs[:, :, 3] = 1.0
    if topology in {"long", "fused-hedge", "fused-one-way"}:
        hlcvs[3, 0, 1] = 80.0
    if topology in {"short", "fused-hedge", "fused-one-way"}:
        hlcvs[3, 0, 0] = 120.0
    timestamps = (
        1_700_000_000_000
        + np.arange(count, dtype=np.int64) * 60_000
    )
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
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["max_warmup_minutes"] = 1
    enabled_sides = (
        ("long", "short") if topology.startswith("fused-") else (topology,)
    )
    config["live"]["approved_coins"] = {
        side: ["BTC"] if side in enabled_sides else []
        for side in ("long", "short")
    }
    config["live"]["hedge_mode"] = topology == "fused-hedge"
    config["backtest"]["coins"] = {"bybit": ["BTC", "ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    config["backtest"]["dynamic_wel_by_tradability"] = False
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.1 if enabled else 0.0
        risk["n_positions"] = 2 if enabled else 0
        risk["entry_cooldown_minutes"] = 100.0
        risk["we_excess_allowance_pct"] = 0.0
        risk["position_exposure_enforcer_enabled"] = False
        risk["total_exposure_enforcer_enabled"] = False
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
            strategy = config["bot"][side]["strategy"][
                "trailing_martingale"
            ]
            strategy["ema_span_0"] = 2.0
            strategy["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 1.0
            strategy["entry"]["initial_ema_dist"] = 0.1
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.5

    proxy = MpsMulticoinProxy(
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
        },
    )
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
    assert result["position_held_days_max"] == pytest.approx(
        exact_analysis["position_held_days_max"], rel=1.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_hsl_can_restart_during_invalid_tail(
    strategy_kind, side
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    count = 15
    last_valid = 10
    close = np.full(count, np.nan)
    close[:8] = 100.0
    close[8 : last_valid + 1] = 70.0 if side == "long" else 130.0
    high = np.where(np.isfinite(close), close * 1.02, np.nan)
    low = np.where(np.isfinite(close), close * 0.98, np.nan)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
        last_valid,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(initial_ema_dist=0.0)
        row[6] = 0.5
        row[7] = 0.5
        row[16] = 0.5
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsTrailingMartingaleRunner
    else:
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.5,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 1.0,
                "offset": 0.0,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsEmaAnchorRunner
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.01,
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 2.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 0.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 0.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    runner_kwargs = {}
    if strategy_kind == "trailing_martingale":
        runner_kwargs["hsl_enabled"] = True
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        **runner_kwargs,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == 0.0
    assert output[f"hsl_triggers_{side}"].item() == 1.0
    assert output[f"hsl_restarts_{side}"].item() == 1.0
    assert output["last_eq_ts"].item() > last_valid * run.interval_ms


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_active_fill_day_bucket_uses_elapsed_time(
    strategy_kind,
):
    import passivbot_rust

    cases = np.asarray(
        [
            [1_439.0, 0.0, 60_000.0],
            [1_440.0, 0.0, 60_000.0],
            [297.0, 10.0, 5.0 * 60_000.0],
            [298.0, 10.0, 5.0 * 60_000.0],
            [215.0, 10.0, 7.0 * 60_000.0],
            [216.0, 10.0, 7.0 * 60_000.0],
        ],
        dtype=np.float32,
    )
    expected = [0, 1, 0, 1, 0, 1]
    probe_kernel = r"""
kernel void passivbot_elapsed_fill_day_bucket_probe(
    constant float* cases [[buffer(0)]],
    device int* output [[buffer(1)]],
    uint b [[thread_position_in_grid]]
) {
    int offset = int(b) * 3;
    output[b] = elapsed_fill_day_bucket(
        cases[offset], cases[offset + 1], cases[offset + 2]
    );
}
"""
    source = (
        passivbot_rust.mps_ema_anchor_source_py()
        if strategy_kind == "ema_anchor"
        else passivbot_rust.mps_trailing_martingale_source_py()
    )
    library = torch.mps.compile_shader(source + probe_kernel)
    inputs = torch.tensor(cases, dtype=torch.float32, device="mps").contiguous()
    output = torch.zeros(len(cases), dtype=torch.int32, device="mps")

    library.passivbot_elapsed_fill_day_bucket_probe(
        inputs,
        output,
        threads=(len(cases), 1, 1),
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == expected


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
    timestamps = (
        1_700_000_000_000
        + np.arange(count, dtype=np.int64) * interval_ms
    )
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
        hlcvs[:, coin, 1] = (
            close * 0.99 if lows is None else np.asarray(lows)[:, coin]
        )
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    if markets is None:
        markets = [
            ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
            for _ in range(coin_count)
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
            "forager_volume_drop_pct": 0.0,
            "forager_score_weights_volume": 1.0,
            "forager_score_weights_ema_readiness": 0.0,
            "forager_score_weights_volatility": 0.0,
            "n_positions": 2.0,
            "we_excess_allowance_pct": 0.0,
            "we_excess_allowance_legacy_raw": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
            "twel_enforcer_reduce_portfolio": 0.0,
            **_UNSTUCK_DISABLED_VALUES,
            **_HSL_DISABLED_VALUES,
        }
        row = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]
        runner = MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
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
            "forager_volume_drop_pct": 0.0,
            "forager_score_weights_volume": 1.0,
            "forager_score_weights_ema_readiness": 0.0,
            "forager_score_weights_volatility": 0.0,
            "n_positions": 2.0,
            "we_excess_allowance_pct": 0.0,
            "we_excess_allowance_legacy_raw": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "wel_enforcer_enabled": 0.0,
            "wel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
            "twel_enforcer_reduce_portfolio": 0.0,
            **_UNSTUCK_DISABLED_VALUES,
            **_HSL_DISABLED_VALUES,
        }
        row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
        runner = MpsTrailingMartingaleMulticoinRunner(
            runs[0],
            data,
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
            entry_interval_enabled=True,
        )
        candidates = np.asarray([row + row], dtype=np.float64)

    output = runner.run(candidates)
    torch.mps.synchronize()
    output = {key: value.cpu() for key, value in output.items()}
    metrics = _entry_interval_metrics(output, run, "trailing_martingale")

    expected_count = 6.0 if topology == "fused" else 3.0
    assert output["entry_interval_count"].item() == expected_count
    assert output["entry_interval_hist"].sum().item() == expected_count
    assert metrics["entry_interval_hours_mean"].item() == pytest.approx(
        50.0 / 3.0 / 60.0
    )
    assert metrics["entry_interval_hours_max"].item() == pytest.approx(1.0 / 3.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["single_side", "fused"])
def test_mps_multicoin_retains_synchronized_btc_risk_surface(
    strategy_kind, topology
):
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
        row[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "entry_initial_qty_pct"
            )
        ] = 0.0
        runner_cls = (
            MpsTrailingMartingaleMulticoinFusedRunner
            if topology == "fused"
            else MpsTrailingMartingaleMulticoinRunner
        )
    if topology == "fused":
        runner = runner_cls(run, data, btc_prices=btc_prices)
        candidates = np.asarray([row + row], dtype=np.float64)
    else:
        runner = runner_cls(run, data, side="long", btc_prices=btc_prices)
        candidates = np.asarray([row], dtype=np.float64)

    output = runner.run(candidates)
    torch.mps.synchronize()

    assert output["btc_day_end_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_min_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_max_dd"][0, 0].item() == pytest.approx(0.75)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_forced_normal_reserves_active_slots(
    strategy_kind, topology
):
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
            collect_coin_fill_counts=True,
            hedge_mode=True,
        )
        forced_runner = fused_runner_cls(
            run,
            data,
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
    torch.mps.synchronize()

    assert (baseline["coin_fill_counts"][0] > 0.0).sum().item() == 1
    assert (expanded["coin_fill_counts"][0] > 0.0).sum().item() == 2


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_multicoin_forced_normal_service_matches_exact_active_symbols(
    strategy_kind, topology
):
    from backtest import run_backtest
    from config.schema import get_template_config
    from optimization.gpu.service import MpsMulticoinProxy

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
    config["live"]["strategy_kind"] = strategy_kind
    config["live"]["max_warmup_minutes"] = 1
    enabled_sides = (
        ("long", "short") if topology == "fused" else (topology,)
    )
    config["live"]["approved_coins"] = {
        side: ["BTC", "ETH"] if side in enabled_sides else []
        for side in ("long", "short")
    }
    config["live"]["hedge_mode"] = topology == "fused"
    config["backtest"]["coins"] = {"bybit": ["BTC", "ETH"]}
    config["backtest"]["exchanges"] = ["bybit"]
    config["backtest"]["dynamic_wel_by_tradability"] = True
    for side in ("long", "short"):
        enabled = side in enabled_sides
        risk = config["bot"][side]["risk"]
        risk["total_wallet_exposure_limit"] = 0.2 if enabled else 0.0
        risk["n_positions"] = 1 if enabled else 0
        risk["entry_cooldown_minutes"] = 100.0
        risk["we_excess_allowance_pct"] = 0.0
        risk["total_exposure_entry_gate_enabled"] = False
        risk["position_exposure_enforcer_enabled"] = False
        risk["total_exposure_enforcer_enabled"] = False
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
            strategy = config["bot"][side]["strategy"][
                "trailing_martingale"
            ]
            strategy["ema_span_0"] = 2.0
            strategy["ema_span_1"] = 3.0
            strategy["entry"]["initial_qty_pct"] = 1.0
            strategy["entry"]["initial_ema_dist"] = 0.1
            strategy["entry"]["double_down_factor"] = 0.0
            strategy["close"]["threshold_base_pct"] = 0.5
    config["coin_overrides"] = {
        coin: {
            "live": {
                f"forced_mode_{side}": "normal" for side in enabled_sides
            }
        }
        for coin in ("BTC", "ETH")
    }

    proxy = MpsMulticoinProxy(
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
    assert result["fills_active_symbols_count"] == exact_analysis[
        "fills_active_symbols_count"
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("interval_minutes", [5, 7])
def test_mps_ema_multicoin_aggregated_interval_fused_smoke(interval_minutes):
    _, row, run, data = _multicoin_exposure_fixture(
        "ema_anchor",
        "long",
        count=420,
        interval_minutes=interval_minutes,
        return_context=True,
    )
    runner = MpsEmaAnchorMulticoinFusedRunner(run, data)

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("interval_minutes", [5, 7])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_aggregated_interval_directional_smoke(
    interval_minutes, side
):
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=420,
        interval_minutes=interval_minutes,
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("interval_minutes", [5, 7])
def test_mps_tm_multicoin_aggregated_interval_fused_smoke(interval_minutes):
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=420,
        interval_minutes=interval_minutes,
        return_context=True,
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(run, data)

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert runner.interval_minutes == interval_minutes
    assert torch.isfinite(output["day_end_eq"]).any().item()
    assert torch.isfinite(output["last_eq_ts"]).all().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0, 0.02)
        for _ in range(2)
    ]
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 1.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 0.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    if topology == "fused":
        runner = fused_runner_cls(
            run,
            data,
            long_coin_overrides=overrides,
            short_coin_overrides=overrides,
        )
        short_row = list(row)
        qty_key = (
            "base_qty_pct"
            if strategy_kind == "ema_anchor"
            else "entry_initial_qty_pct"
        )
        short_row[keys.index(qty_key)] *= 0.5
        params = np.asarray([row + short_row], dtype=np.float64)
    else:
        params = np.asarray([row], dtype=np.float64)
    output = runner.run(params)
    torch.mps.synchronize()

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
    assert output["day_end_eq"][0, 0].item() == pytest.approx(
        output["balance"].item()
    )
    assert output["last_eq_ts"].item() > last_valid * run.interval_ms
    assert output["hsl_tier_samples_total"].item() > last_valid


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    highs[3, :] = 101.0
    lows[3, :] = 99.0
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 1.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 0.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    if topology == "fused":
        runner = fused_runner_cls(run, data)
        params = np.asarray([row + row], dtype=np.float64)
    else:
        params = np.asarray([row], dtype=np.float64)
    output = runner.run(params)
    torch.mps.synchronize()

    assert output["balance"].item() > 0.0
    assert output["first_eq_ts"].item() == pytest.approx(
        expected_first_eq_index * run.interval_ms
    )
    assert output["last_eq_ts"].item() >= invalid_end * run.interval_ms
    expected_hsl_samples = (
        output["last_eq_ts"].item() - output["first_eq_ts"].item()
    ) / run.interval_ms + 1.0
    assert output["hsl_tier_samples_total"].item() == pytest.approx(
        expected_hsl_samples
    )
    day_samples = output["day_end_eq"][0]
    assert torch.isfinite(day_samples).any().item()
    assert day_samples[torch.isfinite(day_samples)][-1].item() == pytest.approx(
        output["balance"].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 2.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 0.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 1.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    output = fused_runner_cls(
        run,
        data,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output[f"hsl_triggers_{adverse_side}"].item() == 1.0
    assert output[f"hsl_restarts_{adverse_side}"].item() == 1.0
    assert output["last_eq_ts"].item() > last_valid * run.interval_ms


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_reselects_recovered_flat_candidate(
    strategy_kind, side
):
    import passivbot_rust

    closes = np.full((4, 2), 100.0)
    closes[2, 0] = 0.0
    highs = np.full((4, 2), 101.0)
    lows = np.full((4, 2), 99.0)
    highs[2, 0] = 0.0
    lows[2, 0] = 0.0
    runner, row, _run, data = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=4,
        closes=closes,
        highs=highs,
        lows=lows,
        return_context=True,
    )
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
    {update_selection}(
        candidate_side, config, bars, coin_settings, coin_overrides,
        2, C, short_side, false, 1, 0.0f, 0ul, false, 0.0f
    );
    output[2] = candidate_side.selected[0] ? 1.0f : 0.0f;
    output[3] = candidate_side.selected[1] ? 1.0f : 0.0f;
    {update_selection}(
        candidate_side, config, bars, coin_settings, coin_overrides,
        3, C, short_side, false, 1, 0.0f, 0ul, false, 0.0f
    );
    output[4] = candidate_side.selected[0] ? 1.0f : 0.0f;
    output[5] = candidate_side.selected[1] ? 1.0f : 0.0f;
}}
"""
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    output = torch.zeros(6, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(source + probe_kernel)
    library.passivbot_multicoin_candidate_recovery_probe(
        data["bars"],
        data["coin_settings"],
        runner.coin_overrides,
        params,
        output,
        int(side == "short"),
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        lows[1, 0] = 0.0
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
    params = torch.tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device="mps",
    )
    output = torch.zeros(2, dtype=torch.float32, device="mps")
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
    side.entry_gen_initial_tick[0] = int(rint(100.0f / price_step));
    side.entry_gen_touch_tick[0] = short_side ? 0 : 100000000;
    side.entry_recursive_market_mode[0] = true;
    apply_tm_multicoin_recursive_entry_twel_gate(
        side, config, bars, fill_ticks, coin_settings, coin_overrides,
        k, C, short_side, 0.001f
    );
    output[0] = side.entry_qty[0];
    output[1] = float(side.entry_gate_suffix_keep_count[0]);
}
"""
    source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    library = torch.mps.compile_shader(source + probe_kernel)
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
    torch.mps.synchronize()

    # Exact Rust retains the already generated next entry but does not expand
    # its recursive market suffix once NextCandle.tradable becomes false.
    assert output[0].item() > 0.0
    assert output[1].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    side.selection_initialized = false;
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
    side.selection_initialized = false;
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
        closes[1, 0] = 0.0
        last_valid_indices = (2, 2)
    runner, row, _run, data = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=3,
        closes=closes,
        last_valid_indices=last_valid_indices,
        return_context=True,
    )
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
    params = torch.tensor(
        [values[key] for key in keys], dtype=torch.float32, device="mps"
    )
    output = torch.zeros(8, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(source + probe_kernel)
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
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_min_cost_rejection_blocks_concurrent_flat_coins(
    strategy_kind, side
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0)
        for min_cost in (30.0, 5.0)
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
    torch.mps.synchronize()

    assert unfiltered_output["coin_fill_counts"].cpu().tolist()[0][0] > 0.0
    assert unfiltered_output["coin_fill_counts"].cpu().tolist()[0][1] > 0.0
    assert filtered.settings.cpu()[-1].item() == 1.0
    filtered_counts = filtered_output["coin_fill_counts"].cpu().tolist()[0]
    assert filtered_counts[0] == 0.0
    assert filtered_counts[1] == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_min_cost_rejection_never_reuses_flat_cash_floor(
    strategy_kind, side
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0)
        for min_cost in (60.0, 5.0)
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
    torch.mps.synchronize()

    # Coin zero is screened while the proxy is flat. Exact Rust may admit that
    # false negative using its higher current cash balance, so Metal must not
    # reuse the liquidation floor when coin one becomes tradable later.
    assert output["coin_fill_counts"].cpu().tolist()[0] == [0.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_min_cost_floor_expires_after_first_selection(
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
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 1.0, 1.0, 0.0)
        for _ in range(2)
    ]
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
    torch.mps.synchronize()

    assert unfiltered_output["coin_fill_counts"].cpu().tolist()[0][1] > 0.0
    assert filtered_output["coin_fill_counts"].cpu().tolist()[0] == [0.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_mps_fused_min_cost_rejection_never_reuses_flat_cash_floor(
    strategy_kind, hedge_mode
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0)
        for min_cost in (60.0, 5.0)
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
        hedge_mode=hedge_mode,
        filter_by_min_effective_cost=True,
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        min_cost = 5.0
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
        min_cost = 7.0
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, min_cost, 1.0, 0.0)
        for _ in range(2)
    ]
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
    torch.mps.synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    passing_counts = passing_output["coin_fill_counts"].cpu().tolist()[0]
    assert counts == [0.0, 0.0]
    assert passing_counts[0] > 0.0
    assert passing_counts[1] > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_min_cost_blocks_new_flat_slot_while_portfolio_is_open():
    strategy_kind = "trailing_martingale"
    highs = np.tile(np.asarray([100.0, 120.0]), (10, 1))
    lows = highs * 0.99
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 1.0, 1.0, 0.0)
        for _ in range(2)
    ]
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

    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_initial_qty_pct")
    ] = 0.5
    parameters = np.asarray([row], dtype=np.float64)
    unfiltered_output = unfiltered.run(parameters)
    filtered_output = filtered.run(parameters)
    torch.mps.synchronize()

    unfiltered_counts = unfiltered_output["coin_fill_counts"].cpu().tolist()[0]
    filtered_counts = filtered_output["coin_fill_counts"].cpu().tolist()[0]
    assert unfiltered_counts[0] > 0.0
    assert unfiltered_counts[1] > 0.0
    assert filtered_counts[0] > 0.0
    assert filtered_counts[1] == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_fused_one_way_min_cost_rejection_blocks_side_arbitration(strategy_kind):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
        for _ in range(2)
    ]
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
        "base_qty_pct"
        if strategy_kind == "ema_anchor"
        else "entry_initial_qty_pct"
    )
    long_row = list(row)
    short_row = list(row)
    long_row[keys.index(qty_key)] = 0.01
    short_row[keys.index(qty_key)] = 1.0
    runner_cls = (
        MpsEmaAnchorMulticoinFusedRunner
        if strategy_kind == "ema_anchor"
        else MpsTrailingMartingaleMulticoinFusedRunner
    )
    runner = runner_cls(
        run,
        data,
        hedge_mode=False,
        filter_by_min_effective_cost=True,
    )

    output = runner.run(np.asarray([long_row + short_row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count_long"].item() == 0.0
    assert output["fill_count"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("signal_mode", ["unified", "pside"])
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 0.0 if signal_mode == "unified" else 1.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_dual_multicoin_pside_hsl_runs_both_directional_controllers(
    strategy_kind,
):
    count = 128
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    closes[20:60] *= 1.3
    closes[60:] *= 0.65
    outputs = {}
    runners = {}
    rows = {}
    for side in ("long", "short"):
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
            "hsl_ema_span_minutes": 1.0,
            "hsl_cooldown_minutes_after_red": 0.0,
            "hsl_no_restart_drawdown_threshold": 1.0,
            "hsl_restart_policy": 2.0,
            "hsl_tier_ratio_yellow": 0.5,
            "hsl_tier_ratio_orange": 0.75,
            "hsl_orange_graceful_stop": 0.0,
            "hsl_signal_mode": 1.0,
            "hsl_slot_count": 1.0,
        }.items():
            row[keys.index(key)] = value
        runners[side] = runner
        rows[side] = row
        outputs[side] = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    for side in ("long", "short"):
        other_side = "short" if side == "long" else "long"
        output = outputs[side]
        assert output[f"hsl_{side}_enabled"].item()
        assert not output[f"hsl_{other_side}_enabled"].item()
        assert output[f"hsl_triggers_{side}"].item() == 1.0
        assert output[f"hsl_triggers_{other_side}"].item() == 0.0
        assert output["hsl_trigger_drawdown_count"].item() == 1.0
        assert output["hsl_panic_loss_drawdown_count"].item() == 1.0
        assert output[f"hsl_strategy_eq_recovery_max_ms_{side}"].item() > 0.0
        assert output[
            f"hsl_strategy_eq_recovery_max_ms_{other_side}"
        ].item() == 0.0
        assert output["open_positions"].item() == 0.0

    from optimization.gpu.metrics import (
        _hard_stop_lifecycle_metrics,
        _hard_stop_panic_loss_metrics,
    )
    from optimization.gpu.service import (
        CORE_OUTPUT_KEYS,
        DIRECTIONAL_HSL_OUTPUT_KEYS,
        _combine_hedged_multicoin_outputs,
    )

    side_outputs = {
        side: {
            key: value.cpu()
            for key, value in outputs[side].items()
            if key in CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS
        }
        for side in ("long", "short")
    }
    run = runners["long"].run_config
    combined = _combine_hedged_multicoin_outputs(
        side_outputs["long"],
        side_outputs["short"],
        run.starting_balance,
        run.liquidation_threshold,
        runners["long"].start_minute_of_day,
        run.interval_ms,
    )
    lifecycle = _hard_stop_lifecycle_metrics(combined, run)
    panic = _hard_stop_panic_loss_metrics(combined, run)
    assert lifecycle["hard_stop_triggers"].item() == 2.0
    assert lifecycle["hard_stop_triggers_long"].item() == 1.0
    assert lifecycle["hard_stop_triggers_short"].item() == 1.0
    assert panic["hard_stop_panic_close_loss_sum"].item() > 0.0
    assert panic["hard_stop_panic_close_loss_drawdown_pct_max"].item() > 0.0

    truncated = {
        side: runners[side].run(
            np.asarray([rows[side]], dtype=np.float64),
            end_steps=np.asarray([60], dtype=np.int32),
        )
        for side in ("long", "short")
    }
    torch.mps.synchronize()
    assert truncated["long"]["hsl_triggers_long"].item() == 0.0
    assert truncated["short"]["hsl_triggers_short"].item() == 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        # The kernel must derive the effective value of two; this packed
        # single-coin default is deliberately not pre-adjusted by Python.
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    packed_slot_variant = list(row)
    packed_slot_variant[keys.index("hsl_slot_count")] = 64.0
    output = runner.run(
        np.asarray([row, packed_slot_variant], dtype=np.float64)
    )
    torch.mps.synchronize()

    assert output[f"hsl_{side}_enabled"].all().item()
    expected_episode_count = 2.0 if shock_coin == "both" else 1.0
    assert (
        output[f"hsl_triggers_{side}"] == expected_episode_count
    ).all().item()
    assert (
        output["hsl_trigger_drawdown_count"] == expected_episode_count
    ).all().item()
    assert (
        output["hsl_panic_loss_drawdown_count"] == expected_episode_count
    ).all().item()
    assert (output["hsl_panic_close_loss_sum"] > 0.0).all().item()
    assert (output[f"hsl_strategy_eq_recovery_max_ms_{side}"] > 0.0).all().item()
    if shock_coin == "both":
        assert (output["coin_fill_counts"] >= 2.0).all().item()
        assert (
            output["hsl_panic_loss_drawdown_max"]
            > output["hsl_panic_loss_drawdown_min"]
        ).all().item()
        assert output["hsl_panic_loss_drawdown_sum"][0].item() == pytest.approx(
            output["hsl_panic_loss_drawdown_min"][0].item()
            + output["hsl_panic_loss_drawdown_max"][0].item()
        )
        expected_open_positions = 0.0
    else:
        assert (output["coin_fill_counts"][:, shock_coin] >= 2.0).all().item()
        expected_open_positions = (
            1.0 if strategy_kind == "trailing_martingale" else 0.0
        )
    assert (output["open_positions"] == expected_open_positions).all().item()
    assert output["hsl_trigger_drawdown_sum"][0].item() == pytest.approx(
        output["hsl_trigger_drawdown_sum"][1].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        overrides[1, hsl_start : hsl_start + 10] = [
            1.0,
            0.05,
            1.0,
            0.0,
            1.0,
            2.0,
            0.5,
            0.75,
            0.0,
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert output[f"hsl_triggers_{side}"].item() == expected_triggers
    if coin_enabled:
        assert output[f"hsl_{side}_enabled"].item()
        assert output["hsl_tier_samples_total"].item() > 0.0
        assert output["coin_fill_counts"][0, 1].item() >= 2.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["coin_fill_counts"][0, 1].item() >= 1.0
    assert output[f"hsl_strategy_eq_recovery_max_ms_{side}"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    assert not output[f"hsl_{side}_enabled"].item()
    assert output["hsl_tier_samples_total"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        overrides[1, hsl_start : hsl_start + 10] = [
            1.0,
            0.05,
            1.0,
            0.0,
            1.0,
            2.0,
            0.5,
            0.75,
            0.0,
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
        torch.mps.synchronize()

    limit_output, market_output = outputs
    assert limit_output[f"hsl_triggers_{side}"].item() == 1.0
    assert market_output[f"hsl_triggers_{side}"].item() == 1.0
    assert (
        market_output["hsl_panic_close_loss_sum"].item()
        > limit_output["hsl_panic_close_loss_sum"].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert output[f"hsl_triggers_{side}"].item() == 1.0
    assert output["coin_fill_counts"][0, 1].item() >= 2.0
    assert output["coin_fill_counts"][0, 0].item() >= 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 1.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value
    params = np.asarray([row], dtype=np.float64)

    assert market_runner.settings[7].item() == pytest.approx(0.02)
    assert market_runner.settings[8].item() == 1.0
    assert torch.allclose(
        market_runner.coin_settings[:, 11],
        torch.full((2,), 0.01, device="mps"),
    )
    limit_output = limit_runner.run(params)
    market_output = market_runner.run(params)
    torch.mps.synchronize()

    assert limit_output[f"hsl_triggers_{side}"].item() == 1.0
    assert market_output[f"hsl_triggers_{side}"].item() == 1.0
    assert limit_output["open_positions"].item() == 0.0
    assert market_output["open_positions"].item() == 0.0
    assert (
        market_output["hsl_panic_close_loss_sum"].item()
        > limit_output["hsl_panic_close_loss_sum"].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_tracks_position_unchanged_max(strategy_kind, side):
    runner, row = _multicoin_exposure_fixture(strategy_kind, side, count=16)

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

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
    assert torch.equal(
        output["day_fill_count"], output["day_fill_count"].round()
    )
    _assert_fill_scalar_contract(output)
    if side == "long":
        assert torch.equal(output["fill_count_long"], output["fill_count"])
    else:
        assert (output["fill_count_long"] == 0.0).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    # Fixed WEL allocates TWEL 1.0 across both configured slots even though
    # the second prepared coin is explicitly ineligible for this side. Dynamic
    # mode allocates the same TWEL to its one observed eligible slot.
    assert fixed["entry_initial_balance_pct"].item() == pytest.approx(0.5)
    assert dynamic["entry_initial_balance_pct"].item() == pytest.approx(1.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        hedge_mode=hedge_mode,
        dynamic_wel_by_tradability=False,
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["entry_initial_balance_pct_long"].item() == pytest.approx(0.5)
    assert output["entry_initial_balance_pct_short"].item() == pytest.approx(0.5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_active_fill_days_use_equity_start_buckets(
    strategy_kind, side
):
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, count=3 * 1440 + 32
    )
    if strategy_kind == "trailing_martingale":
        row[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "entry_threshold_base_pct"
            )
        ] = 0.0
        row[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "close_threshold_base_pct"
            )
        ] = 0.0

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    duration_days = (
        output["last_eq_ts"] - output["first_eq_ts"]
    ).item() / 86_400_000.0
    assert output["fills_active_days_count"].item() == int(np.ceil(duration_days))
    assert output["fills_active_days_count"].item() >= 3.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_tracks_fill_counts_by_symbol(strategy_kind, side):
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, count=32, collect_coin_fill_counts=True
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    coin_counts = output["coin_fill_counts"]
    assert coin_counts.shape == (1, 2)
    assert torch.equal(coin_counts, coin_counts.round())
    assert (coin_counts > 0.0).all()
    assert coin_counts.sum().item() == output["fill_count"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_uses_first_coin_override(
    strategy_kind, side
):
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
    torch.mps.synchronize()

    # TWEL 1.0 / two effective positions, with a bounded 50% allowance.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(0.1875)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    assert not output["alive"].item()
    # The liquidation floor terminates before the second coin becomes
    # tradable, so exact Rust retains a one-position divisor.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(1.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

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

    bits, relation = _minimum_entry_qty_encoding(
        np.array([100.004, 100.0]), market
    )
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

    threshold = _maximum_effective_min_cost(
        np.array([100.004, 100.0]), market
    )

    assert threshold == pytest.approx(0.101 * 100.0)


from optimization.gpu.mps_kernel import (
    MpsEmaAnchorMulticoinRunner,
    MpsEmaAnchorMulticoinLongRunner,
    MpsEmaAnchorRunner,
    MpsTrailingMartingaleMulticoinRunner,
    MpsTrailingMartingaleRunner,
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

    hour_log_range, hour_valid = _build_hourly_log_range(
        high, low, timestamps, run
    )

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

    hour_log_range, hour_valid = _build_hourly_log_range(
        high, low, timestamps, run
    )

    assert hour_valid[9]
    assert hour_valid[18]
    assert hour_log_range[9] == pytest.approx(np.log(200.0 / 50.0))
    assert hour_log_range[18] == pytest.approx(np.log(200.0 / 50.0))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_coin_hsl_preserves_intraminute_realized_peak():
    import passivbot_rust

    trace_kernel = r"""
kernel void passivbot_hsl_coin_fill_peak(
    constant float* params [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    HslState h = load_hsl(params, 0, 0);
    record_coin_hsl_realized_fill(h, 100.0f);
    record_coin_hsl_realized_fill(h, 90.0f);
    HslSignal signal;
    bool valid = derive_hsl_signal(
        h, 1000.0f, 1000.0f, 90.0f, 0.0f, signal
    );
    output[0] = valid ? h.coin_realized_peak : -1.0f;
    output[1] = valid ? signal.drawdown_raw : -1.0f;
}
"""
    source = passivbot_rust.mps_ema_anchor_source_py() + trace_kernel
    library = torch.mps.compile_shader(source)
    params = torch.tensor(
        [1.0, 0.1, 1.0, 0.0, 1.0, 2.0, 0.5, 0.75, 0.0, 2.0, 2.0],
        dtype=torch.float32,
        device="mps",
    )
    output = torch.zeros(2, dtype=torch.float32, device="mps")

    library.passivbot_hsl_coin_fill_peak(params, output, threads=(1, 1, 1))
    actual = output.cpu().numpy()

    assert actual[0] == pytest.approx(100.0)
    assert actual[1] == pytest.approx(0.02)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_joint_multicoin_account_feeds_exact_pside_hsl_scopes():
    import passivbot_rust

    trace_kernel = r"""
kernel void passivbot_joint_multicoin_account_trace(
    device const float* samples [[buffer(0)]],
    constant float* params [[buffer(1)]],
    constant float* settings [[buffer(2)]],
    constant int* sizes [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint b [[thread_position_in_grid]]
) {
    const int N = sizes[0];
    const int SAMPLE_COLS = 8;
    const int HSL_PARAM_COLS = 11;
    const int OUTPUT_COLS = 13;
    if (b > 0) return;
    JointPortfolioAccount account = init_joint_portfolio_account(settings[0]);
    HslState long_hsl = load_hsl(params, 0, 0);
    HslState short_hsl = load_hsl(params, HSL_PARAM_COLS, 0);
    for (int k = 0; k < N; ++k) {
        const int si = k * SAMPLE_COLS;
        // Exact Rust check_for_fills() completes the long-side pass before
        // the short-side pass for every candle.
        record_joint_portfolio_fill(account, samples[si + 0], true);
        record_joint_portfolio_fill(account, samples[si + 1], false);
        float long_unreal = samples[si + 2];
        float short_unreal = samples[si + 3];
        float equity = joint_portfolio_equity(
            account, long_unreal, short_unreal
        );
        bool can_generate = joint_portfolio_can_generate(
            account, equity, settings[1]
        );
        bool long_unified = long_hsl.signal_mode == HSL_SIGNAL_UNIFIED;
        bool short_unified = short_hsl.signal_mode == HSL_SIGNAL_UNIFIED;
        HslSignal long_signal;
        HslSignal short_signal;
        bool long_valid = can_generate && derive_hsl_signal(
            long_hsl,
            account.balance,
            settings[0],
            joint_hsl_realized_pnl(account, long_unified, true),
            joint_hsl_unrealized_pnl(
                long_unreal, short_unreal, long_unified, true
            ),
            long_signal
        );
        bool short_valid = can_generate && derive_hsl_signal(
            short_hsl,
            account.balance,
            settings[0],
            joint_hsl_realized_pnl(account, short_unified, false),
            joint_hsl_unrealized_pnl(
                long_unreal, short_unreal, short_unified, false
            ),
            short_signal
        );
        if (long_valid) {
            update_hsl_from_signal(
                long_hsl,
                long_signal,
                account.realized_pnl_long,
                samples[si + 4] > 0.5f,
                samples[si + 6] > 0.5f,
                float(k),
                settings[2]
            );
        }
        if (short_valid) {
            update_hsl_from_signal(
                short_hsl,
                short_signal,
                account.realized_pnl_short,
                samples[si + 5] > 0.5f,
                samples[si + 7] > 0.5f,
                float(k),
                settings[2]
            );
        }
        const int oi = k * OUTPUT_COLS;
        output[oi + 0] = account.balance;
        output[oi + 1] = account.realized_pnl_total;
        output[oi + 2] = account.realized_pnl_peak;
        output[oi + 3] = account.realized_pnl_long;
        output[oi + 4] = account.realized_pnl_short;
        output[oi + 5] = equity;
        output[oi + 6] = can_generate ? 1.0f : 0.0f;
        output[oi + 7] = long_valid ? long_signal.drawdown_raw : -1.0f;
        output[oi + 8] = long_hsl.drawdown_ema;
        output[oi + 9] = float(long_hsl.tier);
        output[oi + 10] = short_valid ? short_signal.drawdown_raw : -1.0f;
        output[oi + 11] = short_hsl.drawdown_ema;
        output[oi + 12] = float(short_hsl.tier);
    }
}
"""
    source = passivbot_rust.mps_ema_anchor_multicoin_source_py() + trace_kernel
    library = torch.mps.compile_shader(source)

    starting_balance = 1_000.0
    samples = np.array(
        [
            [-2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [12.0, -5.0, -20.0, -10.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -80.0, -40.0, 1.0, 1.0, 0.0, 0.0],
            [-30.0, 20.0, -120.0, -5.0, 1.0, 1.0, 0.0, 0.0],
            [5.0, 5.0, -10.0, -80.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -950.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    hsl_params = np.array(
        [1.0, 0.1, 2.0, 0.0, 1.0, 0.0, 0.5, 0.75, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    output = torch.zeros(
        (len(samples), 13), dtype=torch.float32, device="mps"
    )
    library.passivbot_joint_multicoin_account_trace(
        torch.as_tensor(samples, dtype=torch.float32, device="mps").contiguous(),
        torch.as_tensor(
            np.concatenate([hsl_params, hsl_params]),
            dtype=torch.float32,
            device="mps",
        ).contiguous(),
        torch.tensor(
            [starting_balance, 100.0, 60_000.0],
            dtype=torch.float32,
            device="mps",
        ),
        torch.tensor([len(samples)], dtype=torch.int32, device="mps"),
        output,
        threads=(1, 1, 1),
    )
    actual = output.cpu().numpy()

    realized_long = np.cumsum(samples[:, 0], dtype=np.float64)
    realized_short = np.cumsum(samples[:, 1], dtype=np.float64)
    realized_total = realized_long + realized_short
    balance = starting_balance + realized_total
    equity = balance + samples[:, 2] + samples[:, 3]
    np.testing.assert_allclose(actual[:, 0], balance, atol=2.0e-5)
    np.testing.assert_allclose(actual[:, 1], realized_total, atol=2.0e-5)
    intraminute_realized_peak = []
    running_realized = 0.0
    running_peak = 0.0
    for long_fill, short_fill in samples[:, :2]:
        # Rust completes all long fills before starting the short pass. The
        # loss budget retains a peak reached between those two side passes.
        running_realized += float(long_fill)
        running_peak = max(running_peak, running_realized)
        running_realized += float(short_fill)
        running_peak = max(running_peak, running_realized)
        intraminute_realized_peak.append(running_peak)
    np.testing.assert_allclose(
        actual[:, 2], intraminute_realized_peak,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(actual[:, 3], realized_long, atol=2.0e-5)
    np.testing.assert_allclose(actual[:, 4], realized_short, atol=2.0e-5)
    np.testing.assert_allclose(actual[:, 5], equity, atol=2.0e-5)
    assert actual[:, 6].tolist() == [1.0] * (len(samples) - 1) + [0.0]

    tier_ids = {"green": 0.0, "yellow": 1.0, "orange": 2.0, "red": 3.0}
    for side_index, (realized, unrealized) in enumerate(
        (
            (realized_long, samples[:, 2]),
            (realized_short, samples[:, 3]),
        )
    ):
        runtime = passivbot_rust.EquityHardStopRuntime()
        peak_strategy_pnl = float("-inf")
        base_column = 7 if side_index == 0 else 10
        for k, (realized_pnl, unrealized_pnl) in enumerate(
            zip(realized, unrealized)
        ):
            if not bool(actual[k, 6]):
                assert actual[k, base_column] == -1.0
                continue
            strategy_pnl = float(realized_pnl + unrealized_pnl)
            peak_strategy_pnl = max(peak_strategy_pnl, strategy_pnl)
            strategy_equity = starting_balance + strategy_pnl
            peak_strategy_equity = max(
                starting_balance + peak_strategy_pnl, strategy_equity
            )
            exact = runtime.apply_sample(
                timestamp_ms=k * 60_000,
                equity=strategy_equity,
                peak_strategy_equity=peak_strategy_equity,
                red_threshold=0.1,
                ema_span_minutes=2.0,
                tier_ratio_yellow=0.5,
                tier_ratio_orange=0.75,
            )
            assert actual[k, base_column] == pytest.approx(
                exact["drawdown_raw"], abs=2.0e-6
            )
            assert actual[k, base_column + 1] == pytest.approx(
                exact["drawdown_ema"], abs=2.0e-6
            )
            assert actual[k, base_column + 2] == tier_ids[exact["tier"]]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_shared_hsl_trace_matches_exact_rust():
    import passivbot_rust

    trace_kernel = r"""
kernel void passivbot_hsl_trace(
    device const float* samples [[buffer(0)]],
    constant float* params [[buffer(1)]],
    constant float* settings [[buffer(2)]],
    constant int* sizes [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint b [[thread_position_in_grid]]
) {
    const int B = sizes[0];
    const int N = sizes[1];
    const int PARAM_COLS = 11;
    const int SAMPLE_COLS = 5;
    const int OUTPUT_COLS = 11;
    if (b >= uint(B)) return;
    HslState h = load_hsl(params, int(b) * PARAM_COLS, 0);
    for (int k = 0; k < N; k++) {
        const int si = k * SAMPLE_COLS;
        HslSignal signal;
        bool valid = derive_hsl_signal(
            h,
            samples[si + 0],
            settings[0],
            samples[si + 1],
            samples[si + 2],
            signal
        );
        if (valid) {
            update_hsl_from_signal(
                h,
                signal,
                samples[si + 1],
                samples[si + 3] > 0.5f,
                samples[si + 4] > 0.5f,
                float(k),
                settings[1]
            );
        }
        const int oi = (int(b) * N + k) * OUTPUT_COLS;
        output[oi + 0] = valid ? signal.drawdown_raw : -1.0f;
        output[oi + 1] = h.drawdown_ema;
        output[oi + 2] = float(h.tier);
        output[oi + 3] = h.red_active_now ? 1.0f : 0.0f;
        output[oi + 4] = h.red_latched ? 1.0f : 0.0f;
        output[oi + 5] = h.halted ? 1.0f : 0.0f;
        output[oi + 6] = h.no_restart_latched ? 1.0f : 0.0f;
        output[oi + 7] = h.cooldown_until_k;
        output[oi + 8] = h.triggers;
        output[oi + 9] = h.pending_stop_k;
        output[oi + 10] = h.drawdown_ema_max;
    }
}
"""
    source = passivbot_rust.mps_ema_anchor_source_py() + trace_kernel
    library = torch.mps.compile_shader(source)

    starting_balance = 1_000.0
    realized = np.zeros(7, dtype=np.float32)
    unrealized = np.array(
        [0.0, -20.0, -60.0, -120.0, -160.0, -40.0, 0.0],
        dtype=np.float32,
    )
    samples = np.column_stack(
        [
            np.full(len(realized), starting_balance, dtype=np.float32),
            realized,
            unrealized,
            np.ones(len(realized), dtype=np.float32),
            np.zeros(len(realized), dtype=np.float32),
        ]
    )
    signal_modes = ("unified", "pside", "coin")
    params = np.array(
        [
            [1.0, 0.1, 2.0, 0.0, 0.3, 1.0, 0.5, 0.75, 0.0, mode, 2.0]
            for mode in range(len(signal_modes))
        ],
        dtype=np.float32,
    )
    output = torch.zeros(
        (len(signal_modes), len(realized), 11),
        dtype=torch.float32,
        device="mps",
    )
    library.passivbot_hsl_trace(
        torch.as_tensor(samples, dtype=torch.float32, device="mps").contiguous(),
        torch.as_tensor(params, dtype=torch.float32, device="mps").contiguous(),
        torch.tensor(
            [starting_balance, 60_000.0], dtype=torch.float32, device="mps"
        ),
        torch.tensor(
            [len(signal_modes), len(realized)], dtype=torch.int32, device="mps"
        ),
        output,
        threads=(len(signal_modes), 1, 1),
    )
    actual = output.cpu().numpy()

    tier_ids = {"green": 0.0, "yellow": 1.0, "orange": 2.0, "red": 3.0}
    for mode_index, signal_mode in enumerate(signal_modes):
        runtime = passivbot_rust.EquityHardStopRuntime()
        peak_strategy_pnl = float("-inf")
        peak_coin_realized = 0.0
        exact_drawdown_ema_max = 0.0
        for k, (realized_pnl, unrealized_pnl) in enumerate(
            zip(realized, unrealized)
        ):
            if signal_mode == "coin":
                peak_coin_realized = max(peak_coin_realized, float(realized_pnl))
                coin_signal = passivbot_rust.hsl_coin_drawdown_signal(
                    balance=starting_balance,
                    n_positions=2,
                    peak_realized=peak_coin_realized,
                    last_realized=float(realized_pnl),
                    current_upnl=float(unrealized_pnl),
                )
                drawdown_raw = coin_signal["drawdown_raw"]
                equity = max(1.0 - drawdown_raw, 1.0e-12)
                peak_equity = 1.0
            else:
                strategy_pnl = float(realized_pnl + unrealized_pnl)
                peak_strategy_pnl = max(peak_strategy_pnl, strategy_pnl)
                equity = starting_balance + strategy_pnl
                peak_equity = max(starting_balance + peak_strategy_pnl, equity)
            exact = runtime.apply_sample(
                timestamp_ms=k * 60_000,
                equity=equity,
                peak_strategy_equity=peak_equity,
                red_threshold=0.1,
                ema_span_minutes=2.0,
                tier_ratio_yellow=0.5,
                tier_ratio_orange=0.75,
            )
            assert actual[mode_index, k, 0] == pytest.approx(
                exact["drawdown_raw"], abs=2.0e-6
            )
            assert actual[mode_index, k, 1] == pytest.approx(
                exact["drawdown_ema"], abs=2.0e-6
            )
            exact_drawdown_ema_max = max(
                exact_drawdown_ema_max, abs(exact["drawdown_ema"])
            )
            assert actual[mode_index, k, 10] == pytest.approx(
                exact_drawdown_ema_max, abs=2.0e-6
            )
            assert actual[mode_index, k, 2] == tier_ids[exact["tier"]]
            assert bool(actual[mode_index, k, 3]) == exact["red_active_now"]
            assert bool(actual[mode_index, k, 4]) == exact["red_latched"]
            assert not bool(actual[mode_index, k, 5])

    restart_policies = ("always", "threshold", "never")
    lifecycle_unrealized = np.array(
        [0.0, -200.0, -200.0, -200.0, -200.0], dtype=np.float32
    )
    lifecycle_samples = np.column_stack(
        [
            np.full(
                len(lifecycle_unrealized), starting_balance, dtype=np.float32
            ),
            np.zeros(len(lifecycle_unrealized), dtype=np.float32),
            lifecycle_unrealized,
            np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(len(lifecycle_unrealized), dtype=np.float32),
        ]
    )
    lifecycle_params = np.array(
        [
            [1.0, 0.1, 1.0, 5.0, 0.3, policy, 0.5, 0.75, 0.0, 0.0, 1.0]
            for policy in range(len(restart_policies))
        ],
        dtype=np.float32,
    )
    lifecycle_output = torch.zeros(
        (len(restart_policies), len(lifecycle_unrealized), 11),
        dtype=torch.float32,
        device="mps",
    )
    library.passivbot_hsl_trace(
        torch.as_tensor(
            lifecycle_samples, dtype=torch.float32, device="mps"
        ).contiguous(),
        torch.as_tensor(
            lifecycle_params, dtype=torch.float32, device="mps"
        ).contiguous(),
        torch.tensor(
            [starting_balance, 60_000.0], dtype=torch.float32, device="mps"
        ),
        torch.tensor(
            [len(restart_policies), len(lifecycle_unrealized)],
            dtype=torch.int32,
            device="mps",
        ),
        lifecycle_output,
        threads=(len(restart_policies), 1, 1),
    )
    lifecycle_actual = lifecycle_output.cpu().numpy()
    for policy_index, restart_policy in enumerate(restart_policies):
        exact = passivbot_rust.hsl_red_episode_finalization(
            restart_after_red_policy=restart_policy,
            stop_timestamp_ms=2 * 60_000,
            stop_equity=800.0,
            stop_peak_strategy_equity=1_000.0,
            previous_no_restart_peak_strategy_equity=0.0,
            drawdown_ema=0.2,
            red_threshold=0.1,
            no_restart_drawdown_threshold=0.3,
            cooldown_minutes_after_red=5.0,
        )
        final = lifecycle_actual[policy_index, -1]
        assert bool(final[5])
        assert bool(final[6]) == exact["no_restart_latched"]
        expected_cooldown_step = (
            -1.0
            if exact["cooldown_until_ms"] is None
            else exact["cooldown_until_ms"] / 60_000.0
        )
        assert final[7] == pytest.approx(expected_cooldown_step)
        assert final[8] == 1.0
        assert final[9] == 2.0
        assert final[10] == pytest.approx(0.2, abs=2.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_shader_smoke():
    import passivbot_rust

    source = passivbot_rust.mps_ema_anchor_source_py()
    assert "kernel void passivbot_ema_anchor" in source
    assert "constant int SIDE_PARAMS = 35" in source
    assert "total_exposure_reducer_qty" in source
    assert "secondary_close_qty" in source
    assert "realized_loss_gate_allows" in source
    assert "float32_floor_nonnegative" in source
    assert "record_realized_net" in source
    assert "const float max_realized_loss_pct = settings[14]" in source
    assert "const float taker_fee = settings[15]" in source
    assert "const float market_order_slippage_pct = fmax(settings[16], 0.0f)" in source
    assert "const bool long_hsl_panic_market = settings[17] > 0.5f" in source
    assert "const bool short_hsl_panic_market = settings[18] > 0.5f" in source
    assert "const bool market_orders_allowed = settings[19] > 0.5f" in source
    assert "const float market_order_near_touch_threshold = fmax(settings[20], 0.0f)" in source
    assert "market_execution ? taker_fee : maker_fee" in source
    assert "if (variant.is_panic) return true" in source
    ordering_start = source.index("restore_ordinary_close(long_side)")
    assert source.index("prepare_ordinary_market_close(", ordering_start) < source.index(
        "gate_reducer_variant(", ordering_start
    )
    assert "constant int SCALAR_COLS = 68" in source
    assert "scalars[so + 50] = fill_count" in source
    assert "scalars[so + 51] = fill_count_entry" in source
    assert "scalars[so + 52] = fill_count_long" in source
    assert "scalars[so + 53] = fills_active_days_count" in source
    assert "scalars[so + 54] = pnl_recovery_max_min * interval_ms" in source
    assert "scalars[so + 55] = held_sum_min * interval_ms" in source
    assert "scalars[so + 56] = held_count" in source
    assert "scalars[so + 57] = account_recovery_max_min * interval_ms" in source
    assert "scalars[so + 58] = profit_sum_long" in source
    assert "scalars[so + 61] = loss_sum_short" in source
    assert "scalars[so + 62] = long_hsl.enabled" in source
    assert "long_hsl.drawdown_ema_max" in source
    assert "scalars[so + 63] = short_hsl.enabled" in source
    assert "record_gross_pnl" in source
    assert "hsl_tier_samples_total" in source
    assert "h.restart_retrigger_count" in source
    assert "record_hsl_panic_fill(" in source
    assert "side.psize * price_now * c_mult / balance" in source
    assert "short_side.psize * c_mult * (short_side.pprice - close)" in source
    assert "long_close_fill || long_entry_fill" in source
    assert "short_close_fill || short_entry_fill" in source
    assert "fabs(adj) * cp * c_mult /" not in source
    assert "fabs(eq) * ep * c_mult /" not in source
    assert "fabs(adj) * cp / balance" in source
    assert "fabs(eq) * ep / balance" in source
    assert "floor(value / step + 0.5f) * step" in source
    assert "rint(value / step)" not in source
    assert source.count("touch_down_ticks") >= 3
    assert source.count("touch_up_ticks") >= 3
    assert "nearest_ticks(price_now, price_step)" not in source
    assert "nextafter(" not in source
    assert "(run_peak - eqf) / fmax(fabs(run_peak)" in source
    assert "fabs(raw_steps - nearest_count) <= 1.0e-8f" in source
    assert "if (current_cost_we >= cap" in source
    assert "const bool filter_by_min_effective_cost" in source
    assert "passes_min_effective_cost" in source
    assert "projected_cost_lower" in source
    assert "float guaranteed_balance_lower" in source
    assert "accumulate_min_cost_balance_error" not in source
    assert source.index("float eqf = liq ? liq_floor : equity") < source.index(
        "(run_peak - eqf) / fmax(fabs(run_peak)"
    )

    count = 512
    phase = np.linspace(0.0, 10.0 * np.pi, count)
    close = 100.0 + np.sin(phase) * 5.0
    high = close * 1.01
    low = close * 0.99
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=0.0002,
    )
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=10,
        trade_start_idx=10,
        requested_start_ts_ms=int(timestamps[0]),
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
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
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    parameters = np.array([row + row, row + row], dtype=np.float64)

    runner = MpsEmaAnchorRunner(market, run, data)
    output = runner.run(parameters)
    torch.mps.synchronize()

    assert runner._buffers[2][1].shape == (2, 66)
    assert (output["hsl_drawdown_ema_mean_worst_1pct_long"] == 0.0).all()
    assert (output["hsl_drawdown_ema_mean_worst_1pct_short"] == 0.0).all()
    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (2,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0
    fill_days = output["day_has_fill"]
    assert torch.isfinite(output["day_net_pnl"][fill_days]).all()
    assert (output["day_last_fill_balance"][fill_days] > 0.0).all()
    assert (output["day_fill_count"][fill_days] >= 1.0).all()
    assert torch.equal(
        output["day_fill_count"], output["day_fill_count"].round()
    )
    _assert_fill_scalar_contract(output)
    assert (output["total_wallet_exposure_max"] > 0.0).all()
    assert (
        output["total_wallet_exposure_max"]
        >= output["total_wallet_exposure_mean"]
    ).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_anchor_near_touch_market_entry_uses_next_close_and_taker_fee(side):
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=0.0,
        c_mult=1.0,
        maker_fee=0.0,
        taker_fee=0.01,
    )
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
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [
        0.1,
        2.0,
        3.0,
        0.0,
        0.0005,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    parameters = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hedge_mode": True,
        "hsl_enabled": False,
    }

    resting = MpsEmaAnchorRunner(market, run, data, **common).run(parameters)
    promoted_runner = MpsEmaAnchorRunner(
        market,
        run,
        data,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
        taker_fee=market.taker_fee,
        **common,
    )
    promoted = promoted_runner.run(parameters)
    generic = MpsEmaAnchorRunner(
        market,
        run,
        data,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        market_order_slippage_pct=0.01,
        taker_fee=market.taker_fee,
        **{**common, "hsl_enabled": True},
    ).run(parameters)
    torch.mps.synchronize()

    assert promoted_runner.settings[19].item() == 1.0
    assert promoted_runner.settings[20].item() == pytest.approx(0.001)
    assert promoted_runner.shader_topology == f"{side}_no_hsl"
    assert resting["fill_count"].item() == 0.0
    assert promoted["fill_count"].item() == 1.0
    assert promoted.keys() == generic.keys()
    for key in promoted:
        torch.testing.assert_close(
            promoted[key].cpu(),
            generic[key].cpu(),
            rtol=1.0e-6,
            atol=1.0e-6,
            equal_nan=True,
        )
    if side == "long":
        expected_qty = 1.001
        expected_price = 101.0
        assert promoted["psize"].item() == pytest.approx(expected_qty)
        assert promoted["pprice"].item() == pytest.approx(expected_price)
    else:
        expected_qty = 1.0
        expected_price = 99.0
        assert promoted["short_psize"].item() == pytest.approx(expected_qty)
        assert promoted["short_pprice"].item() == pytest.approx(expected_price)
    expected_balance = 1_000.0 - expected_qty * expected_price * market.taker_fee
    assert promoted["balance"].item() == pytest.approx(expected_balance, abs=2.0e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    ("taker_fee", "market_order_slippage_pct"),
    [(0.01, 0.0), (0.0, 0.01)],
)
def test_mps_ema_market_close_loss_gate_projects_execution_cost(
    side, taker_fee, market_order_slippage_pct
):
    count = 6
    close = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=0.0,
        c_mult=1.0,
        maker_fee=0.0,
        taker_fee=taker_fee,
    )
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
        last_valid_idx=count - 1,
    )
    data = build_mps_data(close, close, close, timestamps, run, market)
    row = _single_coin_param_row(
        {
            "base_qty_pct": 0.1,
            "ema_span_0": 2.0,
            "ema_span_1": 3.0,
            "entry_double_down_factor": 0.0,
            # Both the initial entry and subsequent ordinary close cross touch.
            "offset": -0.005,
            "offset_psize_weight": 0.0,
            "offset_volatility_1h_weight": 0.0,
            "offset_volatility_1m_weight": 0.0,
            "offset_volatility_ema_span_1h": 2.0,
            "offset_volatility_ema_span_1m": 2.0,
            "entry_cooldown_minutes": 100.0,
            "total_wallet_exposure_limit": 1.0,
            "we_excess_allowance_pct": 0.0,
            "we_excess_allowance_legacy_raw": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
        },
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    )
    parameters = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.0,
        "taker_fee": market.taker_fee,
        "market_order_slippage_pct": market_order_slippage_pct,
    }

    ungated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=1.0, **common
    ).run(parameters)
    gated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=0.0, **common
    ).run(parameters)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated["fill_count"].item() == 2.0
    assert ungated[size_key].item() == pytest.approx(0.0)
    assert gated["fill_count"].item() == 1.0
    assert gated[size_key].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    assert "constant int PARAM_COLS = 42" in source
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
    assert "incumbent[c] = selected[c] && psize[c] <= 0.0f" in source
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
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
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
    row += list(_HSL_DISABLED_VALUES.values())

    runner = MpsEmaAnchorMulticoinRunner(
        runs[0],
        data,
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
    torch.mps.synchronize()

    assert runner._buffers[2][1].shape == (2, 65)
    assert (output["hsl_drawdown_ema_mean_worst_1pct_long"] == 0.0).all()
    assert (output["hsl_drawdown_ema_mean_worst_1pct_short"] == 0.0).all()
    assert (output["hsl_drawdown_raw_max_long"] == 0.0).all()
    assert (output["hsl_drawdown_raw_max_short"] == 0.0).all()
    assert output["balance"].device.type == "mps"
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
        sample_interval_days=output[
            "strategy_eq_recovery_sample_interval_days"
        ],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[:, 3] > 0.0).all().item()

    with pytest.raises(ValueError, match="finite and non-negative"):
        MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            side=side,
            forager_score_hysteresis_pct=-0.01,
        )
    with pytest.raises(ValueError, match="max_realized_loss_pct"):
        MpsEmaAnchorMulticoinRunner(
            runs[0], data, side=side, max_realized_loss_pct=float("nan")
        )
    with pytest.raises(ValueError, match="market_order_near_touch_threshold"):
        MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            side=side,
            market_order_near_touch_threshold=float("nan"),
        )

    legacy_long = MpsEmaAnchorMulticoinLongRunner(runs[0], data)
    assert legacy_long.side == "long"

    disabled = np.full(
        (coin_count, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    disabled[:, 11] = 0.0
    disabled_output = MpsEmaAnchorMulticoinRunner(
        runs[0], data, side=side, coin_overrides=disabled
    ).run(np.array([row], dtype=np.float64))
    torch.mps.synchronize()
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
        runs[0], data, side=side, coin_overrides=exact_last
    ).run(np.array([row, changed_candidate], dtype=np.float64))
    torch.mps.synchronize()
    assert torch.equal(
        exact_last_output["day_has_fill"][0], exact_last_output["day_has_fill"][1]
    )
    assert torch.equal(
        exact_last_output["day_end_eq"][0], exact_last_output["day_end_eq"][1]
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    row[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    params = np.asarray([row], dtype=np.float64)

    resting = resting_runner.run(params)
    promoted = market_runner.run(params)
    torch.mps.synchronize()

    assert market_runner.settings[9].item() == 1.0
    assert market_runner.settings[10].item() == pytest.approx(0.001)
    assert resting["fill_count"].item() == 0.0
    assert promoted["fill_count"].item() == 2.0
    assert promoted["open_positions"].item() == 2.0
    assert promoted["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]
    assert promoted["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_retracement_base_pct"
        )
    ] = 0.01
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_retracement_base_pct"
        )
    ] = 0.01
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_initial_ema_dist"
        )
    ] = 0.0005
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_cooldown_minutes"
        )
    ] = 100.0
    params = np.asarray([row], dtype=np.float64)

    resting = resting_runner.run(params)
    promoted = market_runner.run(params)
    torch.mps.synchronize()

    assert market_runner.settings[9].item() == 1.0
    assert market_runner.settings[10].item() == pytest.approx(0.001)
    assert resting["fill_count"].item() == 0.0
    assert promoted["fill_count"].item() == 2.0
    assert promoted["open_positions"].item() == 2.0
    assert promoted["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]
    assert promoted["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    # The promoted first order executes, but its passive limit did not cross.
    # Market policy therefore cannot reveal the recursive suffix.
    assert output["fill_count_entry"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_retracement_base_pct"
        )
    ] = 0.01

    output = runner.run(np.asarray([recursive, trailing], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count_entry"].cpu().tolist()[0] > 2.0
    assert output["fill_count_entry"].cpu().tolist()[1] == 2.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    assert output["coin_fill_counts"].cpu().tolist() == [[0.0, 1.0]]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    coin_fills = output["coin_fill_counts"].cpu().tolist()[0]
    assert coin_fills[0] == 1.0
    assert coin_fills[1] > coin_fills[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    long_overrides[
        1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN
    ] = 0.0
    short_overrides[
        0, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN
    ] = 0.0
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
        long_coin_overrides=long_overrides,
        short_coin_overrides=short_overrides,
        collect_coin_fill_counts=True,
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count_long"].item() == 1.0
    assert output["fill_count"].item() - output["fill_count_long"].item() == 1.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("cooldown, expands", [(0.0, True), (100.0, False)])
def test_mps_tm_multicoin_recursive_market_entry_passive_suffix(
    side, cooldown, expands
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
        "entry_cooldown_minutes": cooldown,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    if expands:
        assert output["fill_count_entry"].item() > 2.0
        assert all(value > 1.0 for value in counts)
    else:
        assert output["fill_count_entry"].item() == 2.0
        assert counts == [1.0, 1.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    assert output["fill_count_entry"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 1.0]]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "threshold, expected_counts",
    [(0.06, [0.0, 1.0]), (0.16, [1.0, 2.0])],
)
def test_mps_tm_multicoin_recursive_entry_gate_orders_cross_coin_ladders(
    side, threshold, expected_counts,
):
    count = 5
    closes = np.tile(np.asarray([100.0, 100.0]), (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    if side == "long":
        lows[3] = closes[3] * 0.8
    else:
        highs[3] = closes[3] * 1.2
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 20.0, 1.0, 0.0)
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
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 2.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": threshold,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count_entry"].item() == sum(expected_counts)
    assert output["coin_fill_counts"].cpu().tolist() == [expected_counts]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_entry_gate_keeps_partial_boundary(side):
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
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 2.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 0.18,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count_entry"].item() == 4.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 2.0]]
    exposure = output["total_wallet_exposure_max"].item()
    assert 0.17 < exposure < 0.18


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    torch.mps.synchronize()

    counts = output["coin_fill_counts"].cpu().tolist()[0]
    assert counts[0] == 1.0
    assert counts[1] > 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_next_close_promotes_without_suffix(side):
    count = 7
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    markets = [
        ProxyMarket(
            0.001, 0.01, 0.001, 0.0, 1.0,
            maker_fee=0.0, taker_fee=0.01,
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
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # No passive recursive close crosses. Exact Rust emits and promotes only
    # calc_next_close for each coin, rather than exposing the suffix.
    assert output["fill_count"].item() == 4.0
    assert output["fill_count_entry"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 2.0]]
    assert output[size_key].item() == 0.0
    assert output["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_expanded_recursive_close_promotes_each_group(side):
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
    markets = [
        ProxyMarket(
            0.001, 0.01, 0.001, 0.0, 1.0,
            maker_fee=0.0, taker_fee=0.01,
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
    torch.mps.synchronize()

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
    assert output["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_coin_override_selects_recursive_market_close(side):
    count = 7
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The base trailing coin never triggers on flat candles. The overridden
    # recursive coin emits and promotes its ordinary next close.
    assert output["fill_count_entry"].item() == 2.0
    assert output["fill_count"].item() == 3.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 2.0]]
    assert output[size_key].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_recursive_market_close_uses_executable_minimum(side):
    count = 7
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 100.04, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output["fill_count_entry"].item() == 1.0
    assert output["fill_count"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 0.0]]
    # The requested 10% close is below the executable market minimum; the
    # promoted order is raised to that minimum without closing the position.
    expected_remaining = 8.998 if side == "long" else 8.987
    assert output[size_key].item() == pytest.approx(
        expected_remaining, abs=0.002
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    torch.mps.synchronize()

    # The short opens while the executable minimum is six units. After price
    # halves, the minimum rises to twelve units, above the whole remaining
    # position. Exact Rust preserves one recursive market close at the full
    # position size instead of filtering every group and leaving dust open.
    assert output["fill_count_entry"].item() == 1.0
    assert output["fill_count"].item() == 2.0
    assert output["coin_fill_counts"].cpu().tolist() == [[2.0, 0.0]]
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_multicoin_short_market_entry_uses_executable_minimum():
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 100.04, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
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
    torch.mps.synchronize()

    assert output["fill_count"].item() == 1.0
    assert output["fill_count_entry"].item() == 1.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 0.0]]
    assert output["short_psize"].item() == pytest.approx(1.001)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_short_market_entry_uses_executable_minimum():
    count = 5
    closes = np.tile(np.asarray([100.0, 120.0]), (count, 1))
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 100.04, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    torch.mps.synchronize()

    assert output["fill_count"].item() == 1.0
    assert output["fill_count_entry"].item() == 1.0
    assert output["coin_fill_counts"].cpu().tolist() == [[1.0, 0.0]]
    assert output["short_psize"].item() == pytest.approx(1.001)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
        for _ in range(2)
    ]
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
    row[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("total_wallet_exposure_limit")
    ] = 0.2
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("n_positions")] = 2.0
    params = np.asarray([row], dtype=np.float64)

    promoted = market_runner.run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert promoted["fill_count"].item() == 2.0
    expected_size = 1.998 if side == "long" else 1.996
    assert promoted[size_key].item() == pytest.approx(expected_size)
    assert promoted["total_wallet_exposure_max"].item() == pytest.approx(
        expected_size * 100.0 / 1_000.0
    )
    assert promoted["total_wallet_exposure_max"].item() < 0.2


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
        for _ in range(2)
    ]
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
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    row[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    params = np.asarray([row], dtype=np.float64)

    resting = resting_runner.run(params)
    promoted = market_runner.run(
        np.asarray([row, row], dtype=np.float64),
        end_steps=np.asarray([count - 2, count - 1], dtype=np.int32),
    )
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    )
    row[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.0005
    row[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    params = np.asarray([row + row, row + row], dtype=np.float64)
    overrides = np.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
    resting_runner = MpsEmaAnchorMulticoinFusedRunner(
        run,
        data,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=True,
    )
    market_runner = MpsEmaAnchorMulticoinFusedRunner(
        run,
        data,
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
    torch.mps.synchronize()

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
    assert (
        promoted["short_psize"][1].item()
        < promoted["short_psize"][0].item()
    )
    assert promoted["balance"][1].item() < promoted["balance"][0].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    )
    for key, value in {
        "entry_initial_ema_dist": 0.0005,
        "entry_retracement_base_pct": 0.01,
        "close_retracement_base_pct": 0.01,
        "entry_cooldown_minutes": 100.0,
    }.items():
        row[TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(key)] = value
    params = np.asarray([row + row], dtype=np.float64)
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    resting_runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
    )
    market_runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
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
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("hedge_mode", [True, False])
def test_mps_tm_multicoin_fused_recursive_market_entries_respect_position_mode(
    hedge_mode,
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
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    )

    output = runner.run(params)
    torch.mps.synchronize()

    assert output["fill_count_entry"].item() > (4.0 if hedge_mode else 2.0)
    assert all(
        value > (2.0 if hedge_mode else 1.0)
        for value in output["coin_fill_counts"].cpu().tolist()[0]
    )
    if hedge_mode:
        assert output["psize"].item() > 0.0
        assert output["short_psize"].item() > 0.0
    else:
        assert (output["psize"].item() > 0.0) != (
            output["short_psize"].item() > 0.0
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("hedge_mode", [True, False])
def test_mps_tm_multicoin_fused_recursive_market_closes_respect_position_mode(
    hedge_mode,
):
    count = 6
    base = np.asarray([100.0, 120.0])
    closes = np.tile(base, (count, 1))
    highs = closes.copy()
    lows = closes.copy()
    highs[4] = base * 1.06
    lows[4] = base * 0.94
    markets = [
        ProxyMarket(
            0.001, 0.01, 0.001, 0.0, 1.0,
            maker_fee=0.0, taker_fee=0.01,
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
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        long_coin_overrides=overrides,
        short_coin_overrides=overrides,
        collect_coin_fill_counts=True,
        hedge_mode=hedge_mode,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.1,
    )

    output = runner.run(params)
    torch.mps.synchronize()

    expected_entries = 4.0 if hedge_mode else 2.0
    assert output["fill_count_entry"].item() == expected_entries
    assert output["fill_count"].item() > expected_entries * 2.0
    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() == 0.0
    expected_min_coin_fills = 4.0 if hedge_mode else 2.0
    assert all(
        count > expected_min_coin_fills
        for count in output["coin_fill_counts"][0].cpu()
    )
    assert output["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_multicoin_fused_kernel_smoke_all_hsl_modes():
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_multicoin_source_py(),
        ema_tail_enabled=True,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    assert "kernel void passivbot_ema_anchor_multicoin_fused" in source
    assert "constant int FUSED_SCALAR_COLS = 72" in source

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
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
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
        row = base + list(hsl.values())
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
        np.asarray(rows, dtype=np.float32), device="mps"
    ).contiguous()
    overrides = torch.full(
        (coin_count, EMA_ANCHOR_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device="mps",
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
        device="mps",
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
        device="mps",
    )
    end_steps = torch.full(
        (batch_size,), data["n"] - 1, dtype=torch.int32, device="mps"
    )
    daily = torch.zeros(
        (batch_size, data["n_days"], 9), dtype=torch.float32, device="mps"
    )
    daily[:, :, 1].fill_(float("inf"))
    daily[:, :, 5].fill_(float("inf"))
    scalars = torch.zeros((batch_size, 72), dtype=torch.float32, device="mps")
    gaps = torch.zeros((batch_size, 128), dtype=torch.int32, device="mps")
    coin_fill_counts = torch.zeros(
        (batch_size, coin_count), dtype=torch.float32, device="mps"
    )

    library = torch.mps.compile_shader(source)
    library.passivbot_ema_anchor_multicoin_fused(
        data["bars"],
        data["fill_ticks"],
        data["touch_ticks"],
        data["hour_log_ranges"],
        data["coin_settings"],
        overrides,
        overrides,
        params,
        run_settings,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        threads=(batch_size, 1, 1),
    )
    torch.mps.synchronize()
    values = scalars.cpu().numpy()

    assert values[:3, 13].tolist() == [1.0, 1.0, 1.0]
    assert (values[:3, 24] > 0.0).all()
    assert (values[:3, 26] > 0.0).all()
    assert (values[:3, 24] - values[:3, 26] > 0.0).all()
    assert values[:3, 32:34].tolist() == [[1.0, 1.0]] * 3
    assert (values[:3, 38] > 0.0).all()
    assert (values[:3, 66] > 0.0).all()
    assert (values[:3, 67] > 0.0).all()
    assert (values[:3, 68] > 0.0).all()
    assert (values[:3, 69] > 0.0).all()
    np.testing.assert_allclose(
        values[:3, 18], values[:3, 60] + values[:3, 62]
    )
    np.testing.assert_allclose(
        values[:3, 19], values[:3, 61] + values[:3, 63]
    )
    np.testing.assert_allclose(
        values[:3, 24], daily[:3, :, 8].sum(dim=1).cpu().numpy()
    )
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
    runner_output = runner.run(
        np.asarray(rows, dtype=np.float64), profile=True
    )
    torch.mps.synchronize()
    torch.testing.assert_close(runner_output["day_end_eq"], daily[:, :, 0])
    torch.testing.assert_close(runner_output["day_min_eq"], daily[:, :, 1])
    torch.testing.assert_close(runner_output["fill_count"], scalars[:, 24])
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_long"], scalars[:, 21]
    )
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_short"], scalars[:, 59]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_long"], scalars[:, 57]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_short"], scalars[:, 58]
    )
    torch.testing.assert_close(runner_output["profit_sum_long"], scalars[:, 60])
    torch.testing.assert_close(runner_output["loss_sum_long"], scalars[:, 61])
    torch.testing.assert_close(runner_output["profit_sum_short"], scalars[:, 62])
    torch.testing.assert_close(runner_output["loss_sum_short"], scalars[:, 63])
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_long"], scalars[:, 64]
    )
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_short"], scalars[:, 65]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_long"], scalars[:, 66]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_short"], scalars[:, 67]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_long"], scalars[:, 68]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_short"], scalars[:, 69]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_mean_worst_1pct_long"], scalars[:, 70]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_mean_worst_1pct_short"], scalars[:, 71]
    )
    assert runner_output["alive"].cpu().tolist() == [
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    assert torch.equal(runner_output["coin_fill_counts"], coin_fill_counts)
    recovery = strategy_eq_recovery_distribution_from_samples(
        runner_output["strategy_eq_recovery_samples"],
        sample_interval_days=runner_output[
            "strategy_eq_recovery_sample_interval_days"
        ],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[[0, 1, 2, 5, 6, 7], 3] > 0.0).all().item()
    assert (
        runner_output["strategy_eq_recovery_samples"][3:5, 0] < 0.0
    ).all().item()
    assert runner.last_profile["kernel_seconds"] >= 0.0

    metric_rows = []
    red_threshold_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(
        "hsl_red_threshold"
    )
    restart_policy_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(
        "hsl_restart_policy"
    )
    for signal_mode in range(3):
        row = side_row(signal_mode) + side_row(signal_mode)
        for side_offset in (0, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)):
            row[side_offset + red_threshold_index] = 0.001
            row[side_offset + restart_policy_index] = 2.0
        metric_rows.append(row)

    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
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
        "long": dict(
            zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, metric_rows[0][:width])
        ),
        "short": dict(
            zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, metric_rows[0][width:])
        ),
    }
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
        return compute_objectives(output, *args, **kwargs)

    proxy._compute_objectives = reduce_service_output
    service_results = proxy.evaluate(candidates)
    assert (seen_hsl_triggers["long"] + seen_hsl_triggers["short"] > 0.0).all()
    assert all(item["hard_stop_triggers"] > 0.0 for item in service_results)
    assert all(
        item["hard_stop_triggers"]
        == item["hard_stop_triggers_long"] + item["hard_stop_triggers_short"]
        for item in service_results
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
    assert all(
        item["drawdown_worst_mean_1pct_ema_strategy_eq"]
        == max(
            item["drawdown_worst_mean_1pct_ema_strategy_eq_long"],
            item["drawdown_worst_mean_1pct_ema_strategy_eq_short"],
        )
        and item["drawdown_worst_mean_1pct_ema_strategy_eq"] > 0.0
        for item in service_results
    )
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

    with pytest.raises(ValueError, match="84 columns"):
        runner.run(np.asarray([side_row(0)], dtype=np.float64))
    truncated = runner.run(
        np.asarray([rows[0]], dtype=np.float64),
        end_steps=np.asarray([60], dtype=np.int32),
    )
    torch.mps.synchronize()
    assert truncated["last_eq_ts"].item() <= 59 * 60_000.0
    assert truncated["fill_count"].item() <= values[0, 24]

    # Coin overrides can enable auto-unstuck even when the side template does
    # not, so the kernel must inspect the effective per-coin setting too.
    override_unstuck = overrides.clone()
    override_unstuck[0, 13] = 1.0
    override_sizes = sizes.clone()
    override_sizes[0] = 1
    override_daily = torch.zeros(
        (1, data["n_days"], 9), dtype=torch.float32, device="mps"
    )
    override_daily[:, :, 1].fill_(float("inf"))
    override_daily[:, :, 5].fill_(float("inf"))
    override_scalars = torch.zeros((1, 72), dtype=torch.float32, device="mps")
    override_gaps = torch.zeros((1, 128), dtype=torch.int32, device="mps")
    override_coin_fills = torch.zeros(
        (1, coin_count), dtype=torch.float32, device="mps"
    )
    library.passivbot_ema_anchor_multicoin_fused(
        data["bars"],
        data["fill_ticks"],
        data["touch_ticks"],
        data["hour_log_ranges"],
        data["coin_settings"],
        override_unstuck,
        overrides,
        params[:1],
        run_settings,
        override_sizes,
        end_steps[:1],
        override_daily,
        override_scalars,
        override_gaps,
        override_coin_fills,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()
    assert override_scalars[0, 9].item() == -1.0
    assert override_scalars[0, 13].item() == 1.0
    assert override_scalars[0, 24].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
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
    side_row = [values[key] for key in param_keys]
    params = np.asarray([side_row + side_row], dtype=np.float64)

    hedge_output = runner_cls(runs[0], data, hedge_mode=True).run(params)
    one_way_output = runner_cls(runs[0], data, hedge_mode=False).run(params)
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
        long_coin_overrides=long_overrides,
        short_coin_overrides=short_overrides,
        hedge_mode=False,
    ).run(params)
    torch.mps.synchronize()

    assert hedge_output["fill_count"].item() > hedge_output[
        "fill_count_long"
    ].item()
    assert one_way_output["fill_count"].item() > 0.0
    assert one_way_output["fill_count"].item() == one_way_output[
        "fill_count_long"
    ].item()
    assert 0.0 < mixed_output["fill_count_long"].item() < mixed_output[
        "fill_count"
    ].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_multicoin_fused_kernel_smoke_all_hsl_modes():
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py(),
        ema_tail_enabled=True,
        raw_drawdown_enabled=True,
        raw_tail_enabled=True,
    )
    assert "kernel void passivbot_trailing_martingale_multicoin_fused" in source
    assert "constant int FUSED_SCALAR_COLS = 72" in source

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
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
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
        "forager_volume_drop_pct": 0.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 2.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        **_UNSTUCK_DISABLED_VALUES,
        **_HSL_DISABLED_VALUES,
    }

    def side_row(signal_mode):
        side_values = dict(values)
        side_values["hsl_enabled"] = 1.0
        side_values["hsl_red_threshold"] = 0.9
        side_values["hsl_signal_mode"] = float(signal_mode)
        return [
            side_values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        ]

    unstuck_long = side_row(0)
    unstuck_long[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("unstuck_enabled")
    ] = 1.0
    unstuck_short = side_row(0)
    unstuck_short[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("unstuck_enabled")
    ] = 1.0
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
        np.asarray(rows, dtype=np.float32), device="mps"
    ).contiguous()
    overrides = torch.full(
        (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        float("nan"),
        dtype=torch.float32,
        device="mps",
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
        device="mps",
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
        device="mps",
    )
    end_steps = torch.full(
        (batch_size,), data["n"] - 1, dtype=torch.int32, device="mps"
    )
    daily = torch.zeros(
        (batch_size, data["n_days"], 9), dtype=torch.float32, device="mps"
    )
    daily[:, :, 1].fill_(float("inf"))
    daily[:, :, 5].fill_(float("inf"))
    scalars = torch.zeros((batch_size, 72), dtype=torch.float32, device="mps")
    gaps = torch.zeros((batch_size, 128), dtype=torch.int32, device="mps")
    coin_fill_counts = torch.zeros(
        (batch_size, coin_count), dtype=torch.float32, device="mps"
    )

    library = torch.mps.compile_shader(source)
    library.passivbot_trailing_martingale_multicoin_fused(
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
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        threads=(batch_size, 1, 1),
    )
    torch.mps.synchronize()
    output = scalars.cpu().numpy()

    assert output[:3, 13].tolist() == [1.0, 1.0, 1.0]
    assert (output[:3, 24] > 0.0).all()
    assert (output[:3, 26] > 0.0).all()
    assert (output[:3, 24] - output[:3, 26] > 0.0).all()
    assert output[:3, 32:34].tolist() == [[1.0, 1.0]] * 3
    assert (output[:3, 38] > 0.0).all()
    assert (output[:3, 66] > 0.0).all()
    assert (output[:3, 67] > 0.0).all()
    assert (output[:3, 68] > 0.0).all()
    assert (output[:3, 69] > 0.0).all()
    np.testing.assert_allclose(output[:3, 18], output[:3, 60] + output[:3, 62])
    np.testing.assert_allclose(output[:3, 19], output[:3, 61] + output[:3, 63])
    np.testing.assert_allclose(
        output[:3, 24], daily[:3, :, 8].sum(dim=1).cpu().numpy()
    )
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
    runner_output = runner.run(np.asarray(rows, dtype=np.float64), profile=True)
    torch.mps.synchronize()
    torch.testing.assert_close(runner_output["day_end_eq"], daily[:, :, 0])
    torch.testing.assert_close(runner_output["day_min_eq"], daily[:, :, 1])
    torch.testing.assert_close(runner_output["fill_count"], scalars[:, 24])
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_long"], scalars[:, 21]
    )
    torch.testing.assert_close(
        runner_output["entry_initial_balance_pct_short"], scalars[:, 59]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_long"], scalars[:, 57]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_max_short"], scalars[:, 58]
    )
    torch.testing.assert_close(runner_output["profit_sum_long"], scalars[:, 60])
    torch.testing.assert_close(runner_output["loss_sum_long"], scalars[:, 61])
    torch.testing.assert_close(runner_output["profit_sum_short"], scalars[:, 62])
    torch.testing.assert_close(runner_output["loss_sum_short"], scalars[:, 63])
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_long"], scalars[:, 64]
    )
    torch.testing.assert_close(
        runner_output["hsl_strategy_eq_recovery_max_ms_short"], scalars[:, 65]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_long"], scalars[:, 66]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_ema_mean_worst_1pct_short"], scalars[:, 67]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_long"], scalars[:, 68]
    )
    torch.testing.assert_close(
        runner_output["hsl_drawdown_raw_max_short"], scalars[:, 69]
    )
    assert runner_output["alive"].cpu().tolist() == [
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    assert torch.equal(runner_output["coin_fill_counts"], coin_fill_counts)
    recovery = strategy_eq_recovery_distribution_from_samples(
        runner_output["strategy_eq_recovery_samples"],
        sample_interval_days=runner_output[
            "strategy_eq_recovery_sample_interval_days"
        ],
    )
    assert torch.isfinite(recovery).all().item()
    assert (recovery[[0, 1, 2, 5, 6, 7], 3] > 0.0).all().item()
    assert (
        runner_output["strategy_eq_recovery_samples"][3:5, 0] < 0.0
    ).all().item()
    assert runner.last_profile["kernel_seconds"] >= 0.0

    metric_rows = []
    red_threshold_index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "hsl_red_threshold"
    )
    restart_policy_index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "hsl_restart_policy"
    )
    for signal_mode in range(3):
        row = side_row(signal_mode) + side_row(signal_mode)
        for side_offset in (
            0,
            len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS),
        ):
            row[side_offset + red_threshold_index] = 0.001
            row[side_offset + restart_policy_index] = 2.0
        metric_rows.append(row)

    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
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
    proxy.fused_runner = runner
    proxy.runners = {}
    candidates = [
        {
            **{
                f"long_{key}": row[index]
                for index, key in enumerate(
                    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
                )
            },
            **{
                f"short_{key}": row[width + index]
                for index, key in enumerate(
                    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
                )
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
    assert all(
        item["drawdown_worst_mean_1pct_ema_strategy_eq"]
        == max(
            item["drawdown_worst_mean_1pct_ema_strategy_eq_long"],
            item["drawdown_worst_mean_1pct_ema_strategy_eq_short"],
        )
        and item["drawdown_worst_mean_1pct_ema_strategy_eq"] > 0.0
        for item in service_results
    )
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

    with pytest.raises(ValueError, match="118 columns"):
        runner.run(np.asarray([side_row(0)], dtype=np.float64))
    with pytest.raises(ValueError, match="short override matrix shaped"):
        MpsTrailingMartingaleMulticoinFusedRunner(
            runs[0],
            data,
            short_coin_overrides=np.empty(
                (coin_count, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS - 1),
                dtype=np.float32,
            ),
        )
    truncated = runner.run(
        np.asarray([rows[0]], dtype=np.float64),
        end_steps=np.asarray([60], dtype=np.int32),
    )
    torch.mps.synchronize()
    assert truncated["last_eq_ts"].item() <= 59 * 60_000.0
    assert truncated["fill_count"].item() <= output[0, 24]

    override_unstuck = overrides.clone()
    override_unstuck[0, 28] = 1.0
    override_sizes = sizes.clone()
    override_sizes[0] = 1
    override_daily = torch.zeros(
        (1, data["n_days"], 9), dtype=torch.float32, device="mps"
    )
    override_daily[:, :, 1].fill_(float("inf"))
    override_daily[:, :, 5].fill_(float("inf"))
    override_scalars = torch.zeros((1, 72), dtype=torch.float32, device="mps")
    override_gaps = torch.zeros((1, 128), dtype=torch.int32, device="mps")
    override_coin_fills = torch.zeros(
        (1, coin_count), dtype=torch.float32, device="mps"
    )
    library.passivbot_trailing_martingale_multicoin_fused(
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
        override_sizes,
        end_steps[:1],
        override_daily,
        override_scalars,
        override_gaps,
        override_coin_fills,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()
    assert override_scalars[0, 9].item() == -1.0
    assert override_scalars[0, 13].item() == 1.0
    assert override_scalars[0, 24].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_trailing_martingale_multicoin_directional_shader_smoke(side):
    import passivbot_rust

    source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    assert "kernel void passivbot_trailing_martingale_multicoin" in source
    assert "constant int PARAM_COLS = 59" in source
    assert "effective_n_positions" in source
    assert "min_since_open" in source
    assert "entry_retracement_base" in source
    assert "close_retracement_base" in source
    assert "as_type<float>(touch_min_qty_bits[k * C + c])" in source
    assert (
        f"constant int OVERRIDE_COLS = {TRAILING_MARTINGALE_COIN_OVERRIDE_COLS}"
        in source
    )
    assert "allowed_wallet_exposure_limit" in source
    assert "twel_entry_gate_enabled" in source
    assert "coin_override_or" in source
    assert "merge_reducer" not in source
    assert "finalized_reducer_qty" in source
    assert "realized_loss_proxy_allows_close" in source
    assert "recursive_grid_close_would_expand" in source
    assert "close_recursive_market_mode" in source
    assert "close_gen_market_price" in source
    assert "selected.market = should_use_ordinary_market_execution" in source
    assert "const bool loss_gate_enabled = run_settings[5] < 1.0f" in source

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
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
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
        "forager_volume_drop_pct": 0.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 2.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        **_UNSTUCK_DISABLED_VALUES,
        **_HSL_DISABLED_VALUES,
    }
    row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    runner = MpsTrailingMartingaleMulticoinRunner(
        runs[0],
        data,
        side=side,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=0.1,
        hsl_raw_drawdown_enabled=True,
        recovery_distribution_enabled=True,
    )
    assert runner.settings.cpu()[5].item() <= 0.1
    output = runner.run(np.array([row, row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (2,)
    assert runner._buffers[2][1].shape == (2, 65)
    assert (output["hsl_drawdown_raw_max_long"] == 0.0).all()
    assert (output["hsl_drawdown_raw_max_short"] == 0.0).all()
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0
    assert (output["open_positions"] <= 2.0).all()
    recovery = strategy_eq_recovery_distribution_from_samples(
        output["strategy_eq_recovery_samples"],
        sample_interval_days=output[
            "strategy_eq_recovery_sample_interval_days"
        ],
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
        runs[0], data, side=side, coin_overrides=disabled
    ).run(np.array([row], dtype=np.float64))
    torch.mps.synchronize()
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
        runs[0], data, side=side, coin_overrides=exact_last
    ).run(np.array([row, changed_candidate], dtype=np.float64))
    torch.mps.synchronize()
    assert torch.equal(
        exact_last_output["day_has_fill"][0],
        exact_last_output["day_has_fill"][1],
    )
    assert torch.equal(
        exact_last_output["day_end_eq"][0],
        exact_last_output["day_end_eq"][1],
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_legacy_raw_allowance_with_gate_disabled_expands_volume(
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
    expanded[keys.index("we_excess_allowance_legacy_raw")] = 1.0
    expanded[keys.index("twel_entry_gate_enabled")] = 0.0

    output = runner.run(np.asarray([baseline, expanded], dtype=np.float64))
    torch.mps.synchronize()
    volume = output["day_volume"].sum(dim=1).cpu().numpy()

    assert volume[0] > 0.0
    assert volume[1] > volume[0] * 1.1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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

    baseline_output = baseline_runner.run(
        np.asarray([baseline], dtype=np.float64)
    )
    override_output = override_runner.run(
        np.asarray([overridden], dtype=np.float64)
    )
    torch.mps.synchronize()

    baseline_volume = baseline_output["day_volume"].sum().item()
    override_volume = override_output["day_volume"].sum().item()
    assert baseline_volume > 0.0
    assert override_volume > baseline_volume * 1.1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()
    volume = output["day_volume"].sum(dim=1).cpu().numpy()

    assert volume[0] > 0.0
    assert 0.0 < volume[1] < volume[0] * 0.75


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 300.0, 1.0, 0.0)
        for _ in range(2)
    ]
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
    torch.mps.synchronize()
    volume = output["day_volume"].sum().item()

    # Exact Rust removes equal-distance entries by ascending symbol index.
    # That leaves the higher-index coin's 0.5-WEL order. Keeping index zero
    # instead would admit only its 0.4-WEL order because the residual is dust.
    # EMA Anchor also closes the selected flat-price order in this short fixture,
    # so it reports the matching entry and exit volume.
    expected_volume = 1.0 if strategy_kind == "ema_anchor" else 0.5
    assert expected_volume - 0.05 < volume < expected_volume + 0.05


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_preserves_tick_aligned_computed_target():
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 99.99, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # A 100.00 bid must fill when the following candle trades one tick below.
    # Unconditionally nudging the aligned target down to 99.99 misses this fill.
    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_directionally_rounds_non_aligned_candle_touch():
    count = 5
    close = np.full(count, 100.006)
    high = np.full(count, 100.006)
    low = np.array([100.006, 100.006, 100.006, 100.005, 100.006])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, -0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Rust rounds a bid clamped to 100.006 down to 100.00. Nearest snapping
    # would place 100.01 and incorrectly fill on the 100.005 low.
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_recovery_captures_early_post_fill_liquidation_endpoint():
    count = 64
    close = np.full(count, 100.0)
    high = np.full(count, 101.0)
    low = np.full(count, 99.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=2.0,
    )
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
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [1.0, 2.0, 3.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    safe_row = list(row)
    safe_row[0] = 0.0
    parameters = np.array(
        [row + row, safe_row + safe_row],
        dtype=np.float64,
    )
    btc_prices = np.full(count, 50_000.0)

    runner = MpsEmaAnchorRunner(
        market,
        run,
        data,
        recovery_distribution_enabled=True,
        btc_prices=btc_prices,
        btc_risk_enabled=False,
        equity_balance_diff_enabled=True,
    )
    output = runner.run(parameters)
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert output["day_volume"][0].sum().item() < 0.0
    assert not output["alive"][0].item()
    assert output["alive"][1].item()
    for name in (
        "equity_balance_diff_neg_max",
        "equity_balance_diff_neg_mean",
        "equity_balance_diff_pos_max",
        "equity_balance_diff_pos_mean",
        "equity_balance_diff_neg_max_btc",
        "equity_balance_diff_neg_mean_btc",
        "equity_balance_diff_pos_max_btc",
        "equity_balance_diff_pos_mean_btc",
    ):
        assert torch.isfinite(output[name]).all().item(), name
    assert output["equity_balance_diff_neg_max"][0].item() > 1.9
    assert output["equity_balance_diff_neg_max_btc"][0].item() == pytest.approx(
        output["equity_balance_diff_neg_max"][0].item(), rel=1.0e-6
    )
    samples = output["strategy_eq_recovery_samples"][0]
    finite_samples = samples[torch.isfinite(samples)]
    assert finite_samples.numel() >= 2
    assert finite_samples[-1].item() == pytest.approx(
        run.starting_balance * run.liquidation_threshold
    )
    recovery = strategy_eq_recovery_distribution_from_samples(
        output["strategy_eq_recovery_samples"],
        sample_interval_days=output[
            "strategy_eq_recovery_sample_interval_days"
        ],
    )
    assert recovery[0, 3].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_recovery_fails_closed_on_coin_hsl_rolling_overflow(strategy_kind):
    count = 12
    close = np.full(count, 100.0)
    high = np.full(count, 101.0)
    low = np.full(count, 99.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(initial_ema_dist=0.0)
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsTrailingMartingaleRunner
    else:
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.5,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 1.0,
                "offset": 0.0,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsEmaAnchorRunner
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 1.0e-8,
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }.items():
        row[keys.index(key)] = value

    runner = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        pnl_lookback_bars=count,
        hsl_panic_market_long=True,
        recovery_distribution_enabled=True,
    )
    # Exercise the production overflow path without needing 2,049 fills.
    runner.rolling_capacity = 1
    output = runner.run(np.asarray([row + row], dtype=np.float64))
    recovery = strategy_eq_recovery_distribution_from_samples(
        output["strategy_eq_recovery_samples"],
        sample_interval_days=output[
            "strategy_eq_recovery_sample_interval_days"
        ],
    )
    torch.mps.synchronize()

    assert not output["alive"].item()
    assert output["balance"].item() == 0.0
    samples = output["strategy_eq_recovery_samples"][0]
    assert torch.isfinite(samples[0]).item()
    assert samples[0].item() < 0.0
    expected_full_horizon_days = (
        runner.n_recovery_samples - 1
    ) * runner.recovery_stride * run.interval_ms / 86_400_000.0
    assert torch.allclose(
        recovery[0],
        torch.full_like(recovery[0], expected_full_horizon_days),
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("hedge_mode", [False, True])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_dual_side_respects_one_way_initial_arbitration(
    hedge_mode, market_orders_allowed
):
    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=0.0002,
    )
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
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=hedge_mode,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.02,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0
    assert (output["short_psize"].item() > 0.0) is hedge_mode


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_short_only_opens_short_position():
    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market, run, data, long_enabled=False, short_enabled=True
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() > 0.0


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


def _tm_single_row(
    *,
    initial_ema_dist=0.01,
    gate_initial=1.0,
    gate_reentry=1.0,
    allowance_pct=0.0,
    legacy_raw=False,
    entry_gate=True,
    threshold=1.0,
    wel_enforcer_enabled=False,
    wel_enforcer_threshold=1.0,
    twel_enforcer_enabled=False,
    unstuck_enabled=False,
    unstuck_ema_gating_enabled=True,
    unstuck_close_pct=0.1,
    unstuck_ema_dist=0.0,
    unstuck_loss_allowance_pct=0.1,
    unstuck_threshold=0.5,
):
    return _tm_row(
        initial_ema_dist=initial_ema_dist,
        gate_initial=gate_initial,
        gate_reentry=gate_reentry,
    ) + _single_coin_exposure_fields(
        allowance_pct=allowance_pct,
        legacy_raw=legacy_raw,
        entry_gate=entry_gate,
        threshold=threshold,
    ) + _tm_wel_enforcer_fields(
        enabled=wel_enforcer_enabled,
        threshold=wel_enforcer_threshold,
    ) + _tm_twel_enforcer_fields(
        enabled=twel_enforcer_enabled,
        unstuck_enabled=unstuck_enabled,
        unstuck_ema_gating_enabled=unstuck_ema_gating_enabled,
        unstuck_close_pct=unstuck_close_pct,
        unstuck_ema_dist=unstuck_ema_dist,
        unstuck_loss_allowance_pct=unstuck_loss_allowance_pct,
        unstuck_threshold=unstuck_threshold,
    ) + [-1.0]


def test_tm_dispatch_specialization_requires_every_enabled_side_and_row():
    row = _tm_single_row()
    matrix = np.asarray([row + row, row + row], dtype=np.float32)

    assert _tm_dispatch_specialization(
        matrix,
        long_enabled=True,
        short_enabled=False,
        market_orders_allowed=False,
        loss_gate_enabled=False,
    ) == (True, False, True, True, True, True, True)

    entry_column = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_retracement_base_pct"
    )
    matrix[1, entry_column] = 0.0
    assert _tm_dispatch_specialization(
        matrix,
        long_enabled=True,
        short_enabled=False,
        market_orders_allowed=True,
        loss_gate_enabled=True,
    ) == (False, False, True, True, False, False, True)

    matrix[:, entry_column] = 0.0
    assert _tm_dispatch_specialization(
        matrix,
        long_enabled=True,
        short_enabled=False,
        market_orders_allowed=False,
        loss_gate_enabled=False,
    )[:2] == (False, True)

    volatility_column = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_threshold_volatility_1m_weight"
    )
    matrix[1, volatility_column] = 0.1
    assert _tm_dispatch_specialization(
        matrix,
        long_enabled=True,
        short_enabled=False,
        market_orders_allowed=False,
        loss_gate_enabled=False,
    )[6] is False
    matrix[1, volatility_column] = 0.0

    matrix[:, entry_column] = 0.001
    side_width = len(TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS)
    short_wel_column = side_width + TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "wel_enforcer_enabled"
    )
    matrix[:, short_wel_column] = 1.0
    assert _tm_dispatch_specialization(
        matrix,
        long_enabled=True,
        short_enabled=False,
        market_orders_allowed=False,
        loss_gate_enabled=False,
    )[3] is True
    assert _tm_dispatch_specialization(
        matrix,
        long_enabled=True,
        short_enabled=True,
        market_orders_allowed=False,
        loss_gate_enabled=False,
    )[3] is False


def test_tm_dispatch_feature_defines_are_opt_in_and_guarded():
    import passivbot_rust

    source = passivbot_rust.mps_trailing_martingale_long_no_hsl_source_py()
    transformed = _with_tm_dispatch_features(
        source,
        trailing_entry_only=True,
        recursive_entry_only=False,
        trailing_close_only=True,
        reducers_disabled=True,
        market_orders_disabled=True,
        loss_gate_disabled=True,
        volatility_disabled=True,
    )

    for name in (
        "PASSIVBOT_TM_TRAILING_ENTRY_ONLY",
        "PASSIVBOT_TM_TRAILING_CLOSE_ONLY",
        "PASSIVBOT_TM_REDUCERS_DISABLED",
        "PASSIVBOT_TM_MARKET_ORDERS_DISABLED",
        "PASSIVBOT_TM_LOSS_GATE_DISABLED",
        "PASSIVBOT_TM_VOLATILITY_DISABLED",
    ):
        assert f"#define {name} 1" in transformed
        assert f"#ifndef {name}" in transformed

    assert _with_tm_dispatch_features(
        source,
        trailing_entry_only=False,
        recursive_entry_only=False,
        trailing_close_only=False,
        reducers_disabled=False,
        market_orders_disabled=False,
        loss_gate_disabled=False,
        volatility_disabled=False,
    ) == source

    recursive = _with_tm_dispatch_features(
        source,
        trailing_entry_only=False,
        recursive_entry_only=True,
        trailing_close_only=False,
        reducers_disabled=False,
        market_orders_disabled=False,
        loss_gate_disabled=False,
        volatility_disabled=False,
    )
    assert "#define PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY 1" in recursive
    assert "#ifndef PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY" in recursive

    with pytest.raises(ValueError, match="mutually exclusive"):
        _with_tm_dispatch_features(
            source,
            trailing_entry_only=True,
            recursive_entry_only=True,
            trailing_close_only=False,
            reducers_disabled=False,
            market_orders_disabled=False,
            loss_gate_disabled=False,
            volatility_disabled=False,
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_entry_specialization_matches_generic(
    monkeypatch, side
):
    import optimization.gpu.mps_kernel as mps_kernel

    count = 512
    steps = np.arange(count, dtype=np.float64)
    close = 100.0 + np.sin(steps / 17.0) * 6.0
    high = close + 0.75
    low = close - 0.75
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_double_down_factor"
    )] = 1.5
    row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_initial_qty_pct"
    )] = 0.05
    row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
        "entry_retracement_base_pct"
    )] = 0.0
    parameters = np.asarray([row + row], dtype=np.float64)

    specialized_runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
    )
    specialized = specialized_runner.run(parameters, profile=True)
    original_specialization = mps_kernel._tm_dispatch_specialization

    def force_generic_recursive_entry(*args, **kwargs):
        features = original_specialization(*args, **kwargs)
        return (features[0], False, *features[2:])

    monkeypatch.setattr(
        mps_kernel,
        "_tm_dispatch_specialization",
        force_generic_recursive_entry,
    )
    generic_runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
    )
    generic = generic_runner.run(parameters, profile=True)
    torch.mps.synchronize()

    assert specialized_runner.last_profile["dispatch_specialization"][
        "recursive_entry_only"
    ] is True
    assert generic_runner.last_profile["dispatch_specialization"][
        "recursive_entry_only"
    ] is False
    assert specialized.keys() == generic.keys()
    for key in specialized:
        if isinstance(specialized[key], torch.Tensor):
            torch.testing.assert_close(
                specialized[key].cpu(),
                generic[key].cpu(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            assert specialized[key] == generic[key]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("topology", "hsl_enabled"),
    [
        ("long", True),
        ("short", True),
        ("long", False),
        ("short", False),
        ("dual", True),
    ],
)
def test_mps_tm_zero_volatility_specialization_matches_generic(
    monkeypatch, topology, hsl_enabled
):
    import optimization.gpu.mps_kernel as mps_kernel

    count = 512
    steps = np.arange(count, dtype=np.float64)
    close = 100.0 + np.sin(steps / 17.0) * 3.0
    high = close + 0.5
    low = close - 0.5
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    rows = []
    for initial_dist, threshold in ((0.005, 0.03), (0.01, 0.08)):
        row = _tm_single_row(initial_ema_dist=initial_dist)
        row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("hsl_enabled")] = (
            float(hsl_enabled)
        )
        row[
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("hsl_red_threshold")
        ] = threshold
        rows.append(row + row)
    parameters = np.asarray(rows, dtype=np.float64)

    specialized_runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=topology != "short",
        short_enabled=topology != "long",
        hsl_enabled=hsl_enabled,
    )
    specialized = specialized_runner.run(parameters, profile=True)
    original_specialization = mps_kernel._tm_dispatch_specialization

    def force_generic_volatility(*args, **kwargs):
        features = original_specialization(*args, **kwargs)
        return (*features[:6], False)

    monkeypatch.setattr(
        mps_kernel, "_tm_dispatch_specialization", force_generic_volatility
    )
    generic_runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=topology != "short",
        short_enabled=topology != "long",
        hsl_enabled=hsl_enabled,
    )
    generic = generic_runner.run(parameters, profile=True)
    torch.mps.synchronize()

    assert specialized_runner.last_profile["dispatch_specialization"][
        "volatility_disabled"
    ] is True
    assert generic_runner.last_profile["dispatch_specialization"][
        "volatility_disabled"
    ] is False
    assert specialized.keys() == generic.keys()
    for key in specialized:
        if isinstance(specialized[key], torch.Tensor):
            torch.testing.assert_close(
                specialized[key].cpu(),
                generic[key].cpu(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            assert specialized[key] == generic[key]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_reducer_free_recursive_close_matches_generic(monkeypatch, side):
    import optimization.gpu.mps_kernel as mps_kernel

    count = 512
    steps = np.arange(count, dtype=np.float64)
    close = 100.0 + np.sin(steps / 17.0) * 3.0
    high = close + 0.5
    low = close - 0.5
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row()
    for key, value in (
        ("close_qty_pct", 0.25),
        ("close_threshold_we_weight", 0.02),
        ("close_retracement_base_pct", 0.0),
    ):
        row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(key)] = value
    parameters = np.asarray([row + row], dtype=np.float64)

    specialized_runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
    )
    specialized = specialized_runner.run(parameters, profile=True)
    original_specialization = mps_kernel._tm_dispatch_specialization

    def force_generic_reducers(*args, **kwargs):
        features = original_specialization(*args, **kwargs)
        return (*features[:3], False, *features[4:])

    monkeypatch.setattr(
        mps_kernel, "_tm_dispatch_specialization", force_generic_reducers
    )
    generic_runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
    )
    generic = generic_runner.run(parameters, profile=True)
    torch.mps.synchronize()

    assert specialized_runner.last_profile["dispatch_specialization"][
        "reducers_disabled"
    ] is True
    assert generic_runner.last_profile["dispatch_specialization"][
        "reducers_disabled"
    ] is False
    assert specialized.keys() == generic.keys()
    for key in specialized:
        if isinstance(specialized[key], torch.Tensor):
            torch.testing.assert_close(
                specialized[key].cpu(),
                generic[key].cpu(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            assert specialized[key] == generic[key]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("topology", ["long", "short", "fused"])
def test_mps_tm_single_coin_entry_intervals_track_only_fresh_positions(topology):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 45
    close = np.full(count, 100.0)
    high = close.copy()
    low = close.copy()
    entry_steps = (5, 15, 35)
    close_steps = (6, 16, 36)
    if topology in {"long", "fused"}:
        low[list(entry_steps)] = 98.0
        high[list(close_steps)] = 102.0
    else:
        high[list(entry_steps)] = 102.0
        low[list(close_steps)] = 98.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.01)
    row[11] = 0.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=topology in {"long", "fused"},
        short_enabled=topology in {"short", "fused"},
        hsl_enabled=False,
        entry_interval_enabled=True,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()
    output = {key: value.cpu() for key, value in output.items()}
    metrics = _entry_interval_metrics(output, run, "trailing_martingale")

    expected_count = 4.0 if topology == "fused" else 2.0
    assert output["entry_interval_count"].item() == expected_count
    assert output["entry_interval_hist"].sum().item() == expected_count
    assert metrics["entry_interval_hours_mean"].item() == pytest.approx(0.25)
    assert metrics["entry_interval_hours_max"].item() == pytest.approx(1.0 / 3.0)
    assert metrics["entry_interval_hours_median"].item() >= 0.25


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_same_candle_close_then_reentry_is_not_an_initial_entry(side):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = close.copy()
    low = close.copy()
    if side == "long":
        low[2] = 98.0
        high[3], low[3] = 102.0, 97.0
    else:
        high[2] = 102.0
        high[3], low[3] = 103.0, 98.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.01)
    row[11] = 0.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        entry_interval_enabled=True,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count"].item() >= 3.0
    assert output["entry_interval_count"].item() == 0
    assert output["entry_interval_hist"].sum().item() == 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_retains_synchronized_btc_risk_surface(strategy_kind):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )

    count = 5
    prices = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(prices, prices, prices, timestamps, run, market)
    btc_prices = np.array([100.0, 100.0, 50.0, 200.0, 100.0])
    if strategy_kind == "trailing_martingale":
        row = _tm_single_row()
        row[6] = 0.0
        runner_cls = MpsTrailingMartingaleRunner
    else:
        row = [0.0, 2.0, 3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
        row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
        runner_cls = MpsEmaAnchorRunner

    output = runner_cls(
        market,
        run,
        data,
        btc_prices=btc_prices,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["btc_day_end_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_min_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_max_dd"][0, 0].item() == pytest.approx(0.75)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_no_hsl_specializations_retain_synchronized_btc_risk_surface(
    side,
):
    count = 5
    prices = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(prices, prices, prices, timestamps, run, market)
    row = _tm_single_row()
    row[6] = 0.0
    runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        btc_prices=np.array([100.0, 100.0, 50.0, 200.0, 100.0]),
    )

    output = runner.run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert runner.shader_topology == f"{side}_no_hsl"
    assert output["btc_day_end_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_min_eq"][0, 0].item() == pytest.approx(5.0)
    assert output["btc_day_max_dd"][0, 0].item() == pytest.approx(0.75)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("recursive_entry", [False, True])
def test_mps_tm_near_touch_market_entry_uses_taker_fill(
    side, recursive_entry
):
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.001)
    row[11] = 0.0 if recursive_entry else 0.001
    row[16] = 10.0
    row[20] = 0.001
    params = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hsl_enabled": False,
    }

    resting = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(params)
    market_output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
        **common,
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    price_key = "pprice" if side == "long" else "short_pprice"
    assert resting[size_key].item() == 0.0
    assert market_output[size_key].item() > 0.0
    assert market_output[price_key].item() == pytest.approx(
        101.0 if side == "long" else 99.0, abs=1.0e-4
    )
    assert market_output["balance"].item() < 1_000.0
    # Market promotion guarantees rung zero, but exact Rust does not expand a
    # recursive ladder unless the original passive order strictly crosses.
    assert market_output["fill_count_entry"].item() == 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_batch_selects_recursive_or_trailing_market_mode_per_candidate(side):
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3] = 99.89
    high[3] = 100.11
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    recursive = _tm_single_row(
        initial_ema_dist=0.001,
        gate_initial=False,
        gate_reentry=False,
        entry_gate=True,
        threshold=0.1202,
    )
    recursive[4] = 1.0
    recursive[6] = 0.05
    recursive[7] = 0.001
    recursive[11] = 0.0
    recursive[16] = 10.0
    recursive[20] = 0.001
    trailing = recursive.copy()
    trailing[11] = 0.001
    params = np.asarray(
        [recursive + recursive, trailing + trailing], dtype=np.float64
    )

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
    ).run(params)
    torch.mps.synchronize()

    assert output["fill_count_entry"].cpu().tolist() == [3.0, 1.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("side", "threshold", "expected_size"),
    [("long", 0.1202, 1.201), ("short", 0.15, 1.499)],
)
def test_mps_tm_recursive_market_entry_retains_exact_twel_prefix(
    side, threshold, expected_size
):
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    # Strictly cross the near passive prefix on the next candle. Exact Rust
    # then expands the immutable ladder; near-touch promotion makes one later
    # rung executable without requiring its passive price to cross.
    low[3] = 99.89
    high[3] = 100.11
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.001,
        gate_initial=False,
        gate_reentry=False,
        entry_gate=True,
        threshold=threshold,
    )
    row[4] = 1.0
    row[6] = 0.05
    row[7] = 0.001
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.001
    params = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hsl_enabled": False,
    }

    resting = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(params)
    promoted = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
        **common,
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert resting["fill_count_entry"].item() >= 1.0
    # Exact Rust retains the nearest full rungs plus at most one partially
    # TWEL-cropped boundary rung.  It never lets a farther order reappear.
    assert promoted["fill_count_entry"].item() == 3.0
    assert promoted["fill_count_entry"].item() > resting[
        "fill_count_entry"
    ].item()
    assert promoted[size_key].item() == pytest.approx(expected_size, abs=1.0e-4)
    # Rust's entry gate measures wallet exposure at the executable market
    # snapshot, before backtest-only adverse slippage is applied to the fill.
    assert promoted[size_key].item() * 100.0 / 1_000.0 < threshold
    assert promoted["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_coin_hsl_coalesces_recursive_same_candle_entry_fees(side):
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3] = 99.89
    high[3] = 100.11
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.001,
        gate_initial=False,
        gate_reentry=False,
        entry_gate=True,
        threshold=0.15,
    )
    for key, value in {
        "entry_double_down_factor": 1.0,
        "entry_initial_qty_pct": 0.05,
        "entry_threshold_base_pct": 0.001,
        "entry_retracement_base_pct": 0.0,
        "close_threshold_base_pct": 10.0,
        "close_retracement_base_pct": 0.001,
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 1.0,
        "hsl_signal_mode": 2.0,
    }.items():
        row[TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(key)] = value
    params = np.asarray([row + row], dtype=np.float64)
    runner = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        pnl_lookback_bars=count,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
    )
    # Three production ladder fills land on one candle. A one-slot ring proves
    # their fees are coalesced rather than overflowing per fill.
    runner.rolling_capacity = 1

    output = runner.run(params)
    torch.mps.synchronize()

    assert output["fill_count_entry"].item() == 3.0
    assert output["alive"].item()
    assert output["balance"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_market_entry_applies_twel_at_executable_touch(side):
    count = 5
    close = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(close, close, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.001,
        entry_gate=True,
        threshold=0.12,
    )
    row[6] = 1.0
    row[11] = 0.0
    row[16] = 10.0
    params = np.asarray([row + row], dtype=np.float64)

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.003,
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output["fill_count_entry"].item() == 1.0
    assert output[size_key].item() == pytest.approx(1.199, abs=1.0e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_long_market_entry_crop_requires_total_exposure_gate():
    count = 5
    close = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(close, close, close, timestamps, run, market)
    disabled = _tm_single_row(initial_ema_dist=0.02, entry_gate=False)
    enabled = _tm_single_row(initial_ema_dist=0.02, entry_gate=True)
    disabled[6] = enabled[6] = 1.0
    disabled[16] = enabled[16] = 10.0
    disabled[20] = enabled[20] = 0.001
    common = {
        "long_enabled": True,
        "short_enabled": False,
        "hsl_enabled": False,
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.03,
    }

    without_gate = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(np.asarray([disabled + disabled], dtype=np.float64))
    with_gate = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(np.asarray([enabled + enabled], dtype=np.float64))
    torch.mps.synchronize()

    assert without_gate["psize"].item() > with_gate["psize"].item()
    assert without_gate["psize"].item() == pytest.approx(10.204, abs=1.0e-3)
    assert 9.997 <= with_gate["psize"].item() < 10.0
    assert with_gate["psize"].item() * 100.0 / run.starting_balance < 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_market_entry_gate_drops_unexecutable_cap_remainder(side):
    count = 5
    close = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(close, close, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.02, entry_gate=True)
    row[6] = 1.0
    row[20] = 0.001
    row[24] = 0.05

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.03,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count_entry"].item() == 0.0
    assert output["psize" if side == "long" else "short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_market_entry_gate_handles_qty_step_below_float32_ulp(side):
    count = 5
    close = np.full(count, 1.0e-7)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 1.0e-9, 1.0, 5.0, 1.0, 0.0)
    run = ProxyRun(
        10.0,
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
    data = build_mps_data(close, close, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.02, entry_gate=True)
    row[6] = 1.0
    row[20] = 0.001
    row[24] = 0.5

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.03,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert np.spacing(np.float32(50_000_000.0)) > market.qty_step
    assert output["fill_count_entry"].item() == 0.0
    assert output["psize" if side == "long" else "short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_next_close_promotes_without_expanding_ladder(side):
    count = 7
    close = np.full(count, 99.0 if side == "long" else 101.0)
    close[:3] = 100.0
    high = close.copy()
    low = close.copy()
    if side == "long":
        high[3] = 100.0
        low[3] = 98.0
    else:
        high[3] = 102.0
        low[3] = 100.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.01)
    row[11] = 0.001
    row[16] = 0.005
    row[20] = 0.0
    params = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hsl_enabled": False,
    }

    resting = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(params)
    zero_cost_market = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.006,
        **common,
    ).run(params)
    costly_market = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.006,
        **common,
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert resting[size_key].item() > 0.0
    assert resting["fill_count"].item() == 1.0
    # No passive recursive close crosses. Exact Rust emits only calc_next_close,
    # then promotes that one order; market policy must not expose farther rungs.
    assert zero_cost_market["fill_count"].item() == 2.0
    assert zero_cost_market[size_key].item() == 0.0
    assert costly_market["fill_count"].item() == 2.0
    assert costly_market[size_key].item() == 0.0
    assert costly_market["balance"].item() < zero_cost_market["balance"].item()
    assert costly_market["loss_sum"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("risk_reducer", [False, True])
def test_mps_tm_expanded_recursive_close_promotes_each_emitted_group(
    side, risk_reducer
):
    # One post-entry execution candle: multiple close fills therefore prove
    # that the emitted recursive grid, rather than repeated calc_next_close
    # generations, was promoted group by group.
    count = 6
    close = np.full(count, 99.0 if side == "long" else 101.0)
    close[:3] = 100.0
    high = close.copy()
    low = close.copy()
    trigger_distance = 0.0054
    if side == "long":
        high[3], low[3] = 100.0, 98.0
        high[4] = 99.0 * (1.0 + trigger_distance)
    else:
        high[3], low[3] = 102.0, 100.0
        low[4] = 101.0 * (1.0 - trigger_distance)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        wel_enforcer_enabled=risk_reducer,
        wel_enforcer_threshold=0.5,
    )
    row[6] = 1.0
    row[11] = 0.001
    row[15] = 0.25
    row[16] = 0.005
    row[17] = 0.001
    row[20] = 0.0
    params = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hsl_enabled": False,
    }

    resting = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(params)
    promoted = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.008,
        **common,
    ).run(params)
    gated = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        taker_fee=0.01,
        market_order_slippage_pct=0.01,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.008,
        max_realized_loss_pct=0.0,
        **common,
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert resting["fill_count"].item() == (3.0 if risk_reducer else 2.0)
    assert resting[size_key].item() > 0.0
    # One passive group expands the immutable ladder. Market policy promotes
    # every emitted group against the generation snapshot. With WEL enabled,
    # the independently promoted reducer executes in canonical price order and
    # the ordinary ladder is trimmed around its quantity before all closes fill.
    assert promoted["fill_count"].item() == (4.0 if risk_reducer else 5.0)
    assert promoted[size_key].item() == 0.0
    assert promoted["balance"].item() < resting["balance"].item()
    assert gated["fill_count"].item() == 1.0
    assert gated[size_key].item() > resting[size_key].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_market_loss_gate_uses_generation_projection(side):
    count = 6
    entry_market = 99.0 if side == "long" else 101.0
    favorable_next = 110.0 if side == "long" else 90.0
    close = np.asarray(
        [100.0, 100.0, 100.0, entry_market, favorable_next, favorable_next]
    )
    high = close.copy()
    low = close.copy()
    if side == "long":
        high[3], low[3] = 100.0, 98.0
        high[4], low[4] = 110.0, 99.0
    else:
        high[3], low[3] = 102.0, 100.0
        high[4], low[4] = 101.0, 90.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.01)
    row[6] = 1.0
    row[11] = 0.001
    row[15] = 0.25
    row[16] = 0.005
    row[17] = 0.001
    row[20] = 0.0
    row[23] = 100.0
    params = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hsl_enabled": False,
        "taker_fee": 0.01,
        "market_order_slippage_pct": 0.01,
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.02,
    }

    ungated = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(params)
    gated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=0.0, **common
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The next candle gaps far enough to make its actual market fills
    # profitable. Exact Rust nevertheless removed these recursive market
    # groups when their generation-time projected slippage and taker fee were
    # lossy; the favorable next-candle price must not re-admit them.
    assert ungated["fill_count"].item() > 1.0
    assert ungated[size_key].item() == 0.0
    assert gated["fill_count"].item() == 1.0
    assert gated[size_key].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_market_reducer_is_not_regated_at_next_candle_price(side):
    count = 7
    entry_price = 99.0 if side == "long" else 101.0
    generation_price = 102.0 if side == "long" else 98.0
    adverse_fill_price = 90.0 if side == "long" else 110.0
    close = np.asarray(
        [
            100.0,
            100.0,
            100.0,
            entry_price,
            generation_price,
            adverse_fill_price,
            adverse_fill_price,
        ]
    )
    high = close.copy()
    low = close.copy()
    if side == "long":
        high[3], low[3] = 100.0, 98.0
        high[5], low[5] = 105.0, adverse_fill_price
    else:
        high[3], low[3] = 102.0, 100.0
        high[5], low[5] = adverse_fill_price, 95.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        entry_gate=False,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.1,
    )
    row[6] = 1.0
    row[11] = 0.001
    row[15] = 0.25
    row[16] = 0.005
    row[17] = 0.001
    row[20] = 0.0
    row[23] = 100.0
    params = np.asarray([row + row], dtype=np.float64)
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hsl_enabled": False,
        "taker_fee": 0.001,
        "market_order_slippage_pct": 0.01,
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.02,
        "max_realized_loss_pct": 0.0,
    }

    output = MpsTrailingMartingaleRunner(
        market, run, data, **common
    ).run(params)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The reducer is profitable after projected slippage at generation and is
    # therefore emitted. Its actual next-candle market fill gaps through the
    # position price and realizes a loss, but that later price cannot revoke
    # an order which exact Rust already admitted.
    assert output["fill_count"].item() > 1.0
    assert output[size_key].item() < 10.0
    assert output["balance"].item() < 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("winning_reducer", ["wel", "twel"])
def test_mps_tm_same_tick_wel_merges_into_recursive_market_group(
    side, winning_reducer
):
    count = 6
    close = np.full(count, 99.0 if side == "long" else 101.0)
    close[:3] = 100.0
    high = close.copy()
    low = close.copy()
    if side == "long":
        high[3], low[3] = 100.0, 98.0
        high[4] = 100.0
    else:
        high[3], low[3] = 102.0, 100.0
        low[4] = 100.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        entry_gate=False,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.5,
        twel_enforcer_enabled=winning_reducer == "twel",
        threshold=0.25 if winning_reducer == "twel" else 1.0,
    )
    row[6] = 1.0
    row[15] = 0.25
    row[16] = -0.01
    row[17] = 0.0
    row[20] = 0.0
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hsl_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.02,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # calc_closes_* emits WEL first and then an ordinary close at the same
    # quantized touch. Rust merges them before orchestration chooses the
    # protective reducer, retaining the later ordinary type. The merged WEL
    # quantity therefore remains in the ordinary group even when larger TWEL
    # wins; market execution and loss gating still close the full position.
    assert output["fill_count"].item() == (
        2.0 if winning_reducer == "wel" else 3.0
    )
    assert output[size_key].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_close_sizing_uses_generation_market(side):
    import passivbot_rust

    row = _tm_single_row(initial_ema_dist=0.0)
    row[15] = 0.5
    row[16] = 0.1 if side == "long" else -0.1
    row[17] = 0.01 if side == "long" else -0.01
    row[20] = 0.0
    row[24] = 0.01
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_close_generation_market_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_psize = 0.1f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = is_long ? 80.0f : 140.0f;
    source.close_gen_touch_down_ticks = is_long ? 7999 : 13999;
    source.close_gen_touch_up_ticks = is_long ? 8001 : 14001;
    source.close_gen_touch_nearest_ticks = is_long ? 8000 : 14000;
    source.close_gen_touch_min_qty = is_long ? 0.063f : 0.036f;
    source.close_gen_touch_min_qty_relation = 0;
    source.market_order_near_touch_threshold = 0.12f;

    CloseGroup group;
    source.market_orders_allowed = true;
    output[0] = float(recursive_close_groups(
        source, is_long, 0, 0.001f, 0.01f, 0.001f, 5.005f, 1.0f,
        0.1f, 0, 0.0f, 500, group
    ));
    output[1] = group.qty;
    recursive_close_groups(
        source, is_long, 1, 0.001f, 0.01f, 0.001f, 5.005f, 1.0f,
        0.1f, 0, 0.0f, 500, group
    );
    output[2] = group.qty;

    source.market_orders_allowed = false;
    output[3] = float(recursive_close_groups(
        source, is_long, 0, 0.001f, 0.01f, 0.001f, 5.005f, 1.0f,
        0.1f, 0, 0.0f, 500, group
    ));
    output[4] = group.qty;
    recursive_close_groups(
        source, is_long, 1, 0.001f, 0.01f, 0.001f, 5.005f, 1.0f,
        0.1f, 0, 0.0f, 500, group
    );
    output[5] = group.qty;
}
"""
    output = torch.zeros(6, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_close_generation_market_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    values = output.cpu().numpy()
    assert values[0] >= 2.0
    np.testing.assert_allclose(values[:3], values[3:], atol=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_close_group_uses_market_touch_minimum(side):
    import passivbot_rust

    row = _tm_single_row(initial_ema_dist=0.0)
    row[15] = 0.01
    row[16] = 0.001
    row[17] = 0.1
    row[20] = 0.0
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_close_market_minimum_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_psize = 1.0f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = 100.0f;
    source.close_gen_touch_down_ticks = 9999;
    source.close_gen_touch_up_ticks = 10001;
    source.close_gen_touch_nearest_ticks = 10000;
    source.close_gen_touch_min_qty = 0.201f;
    source.close_gen_touch_min_qty_relation = 0;
    source.market_order_near_touch_threshold = 0.02f;

    CloseGroup group;
    source.market_orders_allowed = true;
    recursive_close_groups(
        source, is_long, 0, 0.001f, 0.01f, 0.001f, 20.01f, 1.0f,
        1.0f, 0, 0.0f, 500, group
    );
    output[0] = group.qty;
    output[1] = group.market ? 1.0f : 0.0f;

    source.market_orders_allowed = false;
    recursive_close_groups(
        source, is_long, 0, 0.001f, 0.01f, 0.001f, 20.01f, 1.0f,
        1.0f, 0, 0.0f, 500, group
    );
    output[2] = group.qty;
    output[3] = group.market ? 1.0f : 0.0f;
}
"""
    output = torch.zeros(4, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_close_market_minimum_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    values = output.cpu().numpy()
    assert values[1] == 1.0
    assert values[3] == 0.0
    if side == "long":
        # The sell limit sits above the market snapshot, so its passive
        # minimum is smaller. Promotion raises it at executable bid touch.
        assert values[0] == pytest.approx(0.201, abs=1.0e-6)
        assert values[2] == pytest.approx(0.198, abs=1.0e-6)
    else:
        # The buy limit sits below the market snapshot and already satisfies
        # the (smaller) executable-ask minimum, so sizing remains unchanged.
        assert values[0] == pytest.approx(0.203, abs=1.0e-6)
        assert values[2] == pytest.approx(0.203, abs=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_recursive_wel_seed_precedes_market_resize():
    import passivbot_rust

    row = _tm_single_row(
        initial_ema_dist=0.0,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.049,
    )
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_wel_seed_probe(
    constant float* params,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TmSide source = load_side(params, 0, 100.0f);
    float psize = 0.5f;
    float passive_price = 99.01f;
    float market_price = 99.004f;
    float passive_wel_qty = exposure_reducer_qty(
        psize, 100.0f, 1000.0f,
        source.allowed_wel * source.wel_enforcer_threshold,
        passive_price, 0.001f, 0.001f, 19.801f, 1.0f
    );
    float executable_wel_qty = resize_market_close_qty(
        passive_wel_qty, psize, market_price,
        0.001f, 0.001f, 19.801f, 1.0f
    );
    output[0] = passive_wel_qty;
    output[1] = executable_wel_qty;
    output[2] = round_step(psize - passive_wel_qty, 0.001f);
    output[3] = round_step(psize - executable_wel_qty, 0.001f);
}
"""
    output = torch.zeros(4, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_wel_seed_probe(
        params,
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    values = output.cpu().tolist()
    assert values[0] == pytest.approx(0.2, abs=1.0e-6)
    assert values[1] == pytest.approx(0.201, abs=1.0e-6)
    assert values[2] == pytest.approx(0.3, abs=1.0e-6)
    assert values[3] == pytest.approx(0.299, abs=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_expansion_keeps_pregate_wel_reachability(side):
    import passivbot_rust

    row = _tm_single_row(
        initial_ema_dist=0.0,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.05,
    )
    row[15] = 0.25
    row[16] = 0.1
    row[17] = 0.0
    row[20] = 0.0
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_close_pregate_wel_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_psize = 1.0f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = 100.0f;
    source.close_gen_touch_down_ticks = 10000;
    source.close_gen_touch_up_ticks = 10000;
    source.close_gen_touch_nearest_ticks = 10000;
    source.close_gen_touch_min_qty = 0.001f;
    source.close_gen_touch_min_qty_relation = 0;

    source.close_loss_gate_disabled_reducers = false;
    output[0] = recursive_strategy_close_would_expand(
        source, is_long, 10000, 9999,
        0.001f, 0.01f, 0.001f, 0.0f, 1.0f
    ) ? 1.0f : 0.0f;
    source.close_loss_gate_disabled_reducers = true;
    output[1] = recursive_strategy_close_would_expand(
        source, is_long, 10000, 9999,
        0.001f, 0.01f, 0.001f, 0.0f, 1.0f
    ) ? 1.0f : 0.0f;
}
"""
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_close_pregate_wel_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    assert output.cpu().tolist() == [1.0, 1.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_recursive_close_quantity_tolerance_scales_with_magnitude():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_recursive_close_quantity_tolerance_probe(
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    output[0] = quantity_is_meaningfully_below(9.5e-6f, 1.0e-5f)
        ? 1.0f : 0.0f;
    output[1] = quantity_is_meaningfully_below(9.999995e-6f, 1.0e-5f)
        ? 1.0f : 0.0f;
    output[2] = quantity_is_meaningfully_below(1.0e-5f, 1.0e-5f)
        ? 1.0f : 0.0f;
    output[3] = quantity_is_meaningfully_below(0.95f, 1.0f)
        ? 1.0f : 0.0f;
    output[4] = quantity_is_meaningfully_below(0.9999995f, 1.0f)
        ? 1.0f : 0.0f;
    output[5] = quantity_is_meaningfully_below(1.0f, 1.0f)
        ? 1.0f : 0.0f;
}
"""
    output = torch.zeros(6, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_close_quantity_tolerance_probe(
        output,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    # The first sub-micro-step case was admitted by the former fixed +1e-6
    # epsilon. The near-boundary cases remain equal within the same 1e-6
    # relative tolerance at both tested quantity magnitudes.
    assert output.cpu().tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_recursive_market_groups_trim_to_position_before_fill():
    count = 8
    close = np.asarray(
        [100.0, 100.0, 100.0, 100.0, 90.0, 90.0, 90.0, 90.0]
    )
    high = close.copy()
    low = close.copy()
    low[3] = 98.0
    high[5] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 10.01, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0)
    row[6] = 0.041
    row[15] = 0.01
    row[16] = 0.005
    row[17] = -0.01
    row[20] = 0.0
    row[23] = 100.0
    params = np.asarray([row + row], dtype=np.float64)

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.2,
    ).run(params)
    torch.mps.synchronize()

    # Multiple recursive groups resize against the same generation position.
    # Rust trims their aggregate first, so no final below-minimum fragment is
    # executed as a separate fill.
    assert output["fill_count"].item() == 4.0
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_recursive_ladder_refinalizes_requested_reducer_qty():
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3] = 98.0
    high[4] = 201.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.5, 0.01, 0.5, 500.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        gate_initial=1.0,
        gate_reentry=0.0,
        entry_gate=False,
        threshold=0.3,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.3
    row[16] = 1.0
    row[17] = 0.0
    row[20] = 0.0
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        hsl_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # The near-touch TWEL request leaves a remainder below its own executable
    # minimum, so singular finalization expands it to the whole position. The
    # immutable recursive ladder also has a farther passive rung whose lower
    # price-specific minimum admits that remainder. Exact Rust re-finalizes
    # the original TWEL request with the expanded set, yielding two closes.
    assert output["fill_count"].item() == 3.0
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_recursive_reselects_after_candidate_finalization():
    import passivbot_rust

    row = _tm_single_row(
        initial_ema_dist=0.01,
        gate_initial=1.0,
        gate_reentry=0.0,
        entry_gate=False,
        threshold=0.3,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.3
    row[16] = 1.0
    row[17] = 0.0
    row[20] = 0.0
    row[23] = 100.0
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_candidate_reselection_probe(
    constant float* params,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_psize = 10.0f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = 100.0f;
    source.close_gen_touch_down_ticks = 9999;
    source.close_gen_touch_up_ticks = 10001;
    source.close_gen_touch_nearest_ticks = 10000;
    source.close_gen_touch_min_qty = 5.0f;
    source.close_gen_touch_min_qty_relation = 0;
    source.wel_enforcer_enabled = false;
    source.twel_enforcer_enabled = false;
    source.unstuck_enabled = false;
    source.market_orders_allowed = true;
    source.market_order_near_touch_threshold = 0.001f;

    CloseGroup group;
    int group_count = recursive_close_groups(
        source, true, -1, 0.5f, 0.01f, 0.5f, 500.0f, 1.0f,
        10.0f, 0, 0.0f, 500, group
    );
    float requested_a = 5.5f;
    float requested_b = 6.0f;
    float singular_a = finalized_reducer_qty(
        10.0f, requested_a, 100.0f,
        0.5f, 0.5f, 500.0f, 1.0f
    );
    float singular_b = finalized_reducer_qty(
        10.0f, requested_b, 200.0f,
        0.5f, 0.5f, 500.0f, 1.0f
    );
    RecursiveCloseAllocation allocation_a = recursive_close_allocation(
        source, true, group_count, 10.0f, requested_a, 100.0f, true,
        0.5f, 0.01f, 0.5f, 500.0f, 1.0f,
        0, 0.0f, 500
    );
    RecursiveCloseAllocation allocation_b = recursive_close_allocation(
        source, true, group_count, 10.0f, requested_b, 200.0f, false,
        0.5f, 0.01f, 0.5f, 500.0f, 1.0f,
        0, 0.0f, 500
    );
    output[0] = float(group_count);
    output[1] = singular_a;
    output[2] = singular_b;
    output[3] = allocation_a.reducer_qty;
    output[4] = allocation_b.reducer_qty;
    output[5] = reducer_candidate_preferred(
        allocation_b.reducer_qty, 20000, 10,
        allocation_a.reducer_qty, 10000, 24, true
    ) ? 1.0f : 0.0f;
}
"""
    output = torch.zeros(6, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_candidate_reselection_probe(
        params, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    values = output.cpu().tolist()
    assert values[0] >= 1.0
    # Singular finalization makes A appear larger by absorbing its otherwise
    # uncloseable remainder. The expanded ordinary ladder can carry that
    # remainder, so A returns to 5.5 and B's independently finalized 6.0 must
    # become the preferred reducer.
    assert values[1:] == [10.0, 6.0, 5.5, 6.0, 1.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_recursive_finalization_drops_mixed_minimum_reducer():
    import passivbot_rust

    row = _tm_single_row(initial_ema_dist=0.01)
    row[15] = 0.3
    row[16] = 1.0
    row[17] = 0.0
    row[20] = 0.0
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_mixed_minimum_reducer_probe(
    constant float* params,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_psize = 4.0f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = 100.0f;
    source.close_gen_touch_down_ticks = 9999;
    source.close_gen_touch_up_ticks = 10001;
    source.close_gen_touch_nearest_ticks = 10000;
    source.close_gen_touch_min_qty = 5.0f;
    source.close_gen_touch_min_qty_relation = 0;
    source.wel_enforcer_enabled = false;
    source.twel_enforcer_enabled = false;
    source.unstuck_enabled = false;
    source.market_orders_allowed = true;
    source.market_order_near_touch_threshold = 0.001f;

    CloseGroup group;
    int group_count = recursive_close_groups(
        source, true, -1, 0.5f, 0.01f, 0.5f, 500.0f, 1.0f,
        4.0f, 0, 0.0f, 500, group
    );
    RecursiveCloseAllocation allocation = recursive_close_allocation(
        source, true, group_count, 4.0f, 4.0f, 100.0f, true,
        0.5f, 0.01f, 0.5f, 500.0f, 1.0f,
        0, 0.0f, 500
    );
    output[0] = float(group_count);
    output[1] = allocation.reducer_qty;
    output[2] = allocation.ordinary_budget;
    output[3] = allocation.normalize_close_groups ? 1.0f : 0.0f;
}
"""
    output = torch.zeros(4, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_mixed_minimum_reducer_probe(
        params, output, threads=(1, 1, 1)
    )
    torch.mps.synchronize()

    values = output.cpu().tolist()
    assert values[0] >= 1.0
    # The full position is below the reducer's market-touch minimum (5.0),
    # but the farther 200-price ordinary group has a 2.5 minimum. Exact Rust
    # therefore filters the reducer and reallocates all 4.0 units to ordinary
    # closes instead of collapsing to the protective order.
    assert values[1:] == [0.0, 4.0, 1.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_reducer_gate_falls_back_to_next_candidate(side):
    import passivbot_rust

    row = _tm_single_row()
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_reducer_gate_fallback_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = 100.0f;
    ReducerCandidate candidates[3];
    for (int i = 0; i < 3; ++i) {
        candidates[i] = empty_reducer_candidate();
    }
    candidates[0].finalized_qty = 6.0f;
    candidates[0].ticks = is_long ? 9000 : 11000;
    candidates[0].price = is_long ? 90.0f : 110.0f;
    candidates[0].order_type_id = is_long ? 24 : 25;
    candidates[1].finalized_qty = 5.0f;
    candidates[1].ticks = is_long ? 11000 : 9000;
    candidates[1].price = is_long ? 110.0f : 90.0f;
    candidates[1].order_type_id = is_long ? 10 : 21;
    int selected = select_recursive_close_reducer(
        candidates, is_long, source, 0.01f, 0.0f,
        0.0f, 0.0f, 1.0f, true, 0.0f
    );
    output[0] = float(selected);
}
"""
    output = torch.zeros(1, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_reducer_gate_fallback_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    assert output.item() == 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_reducer_gate_uses_generation_market_snapshot(side):
    import passivbot_rust

    row = _tm_single_row()
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_reducer_generation_gate_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_pprice = is_long ? 99.0f : 101.0f;
    source.close_gen_market_price = is_long ? 102.0f : 98.0f;
    ReducerCandidate candidates[3];
    for (int i = 0; i < 3; ++i) {
        candidates[i] = empty_reducer_candidate();
    }
    candidates[0].finalized_qty = 1.0f;
    candidates[0].ticks = is_long ? 10200 : 9800;
    candidates[0].price = source.close_gen_market_price;
    candidates[0].order_type_id = is_long ? 24 : 25;
    candidates[0].market = true;
    int selected = select_recursive_close_reducer(
        candidates, is_long, source, 0.01f, 0.01f,
        0.0f, 0.001f, 1.0f, true, 0.0f
    );
    float adverse_next_fill = is_long ? 89.1f : 111.1f;
    bool adverse_next_allowed = realized_loss_proxy_allows_reducer(
        1.0f, adverse_next_fill, source.close_gen_pprice, is_long,
        false, 1.0f, 0.001f, true, source.close_gen_balance,
        0.0f, 0.0f, 0.0f
    );
    output[0] = float(selected);
    output[1] = adverse_next_allowed ? 1.0f : 0.0f;
}
"""
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_reducer_generation_gate_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    # Generation-time projection admits candidate zero. The adverse next-bar
    # fill would fail the same gate, proving that execution must not consult it
    # again after the order has already been emitted.
    assert output.cpu().tolist() == [0.0, 0.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_recursive_reducer_gate_uses_generation_pnl_snapshot(side):
    import passivbot_rust

    row = _tm_single_row()
    params = torch.tensor(row, dtype=torch.float32, device="mps")
    probe_kernel = r"""
kernel void passivbot_tm_recursive_reducer_generation_pnl_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide source = load_side(params, 0, 100.0f);
    source.close_gen_balance = 1000.0f;
    source.close_gen_pprice = 100.0f;
    source.close_gen_market_price = 100.0f;
    ReducerCandidate candidates[3];
    for (int i = 0; i < 3; ++i) {
        candidates[i] = empty_reducer_candidate();
    }
    candidates[0].finalized_qty = 1.0f;
    candidates[0].ticks = is_long ? 9900 : 10100;
    candidates[0].price = is_long ? 99.0f : 101.0f;
    candidates[0].order_type_id = is_long ? 9 : 20;
    candidates[0].is_unstuck = true;

    source.close_gen_realized_pnl_cumsum_last = 0.0f;
    source.close_gen_realized_pnl_cumsum_max = 0.0f;
    output[0] = float(select_recursive_close_reducer(
        candidates, is_long, source, 0.01f, 0.0f,
        0.0f, 0.0f, 1.0f, true, 0.01f
    ));
    source.close_gen_realized_pnl_cumsum_last = -10.0f;
    source.close_gen_realized_pnl_cumsum_max = 0.0f;
    output[1] = float(select_recursive_close_reducer(
        candidates, is_long, source, 0.01f, 0.0f,
        0.0f, 0.0f, 1.0f, true, 0.01f
    ));
}
"""
    output = torch.zeros(2, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_recursive_reducer_generation_pnl_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    # A fresh 1% budget admits the one-unit losing unstuck close. A generation
    # snapshot already 10 units below its realized-PnL peak has no remaining
    # room and rejects it, regardless of fills which occur on the next candle.
    assert output.cpu().tolist() == [0.0, -1.0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_hsl_panic_replacement_clears_ordinary_market_state(side):
    import passivbot_rust

    params = torch.tensor(
        _tm_single_row(), dtype=torch.float32, device="mps"
    )
    probe_kernel = r"""
kernel void passivbot_tm_hsl_panic_state_probe(
    constant float* params,
    device float* output,
    constant int& is_long_raw,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool is_long = is_long_raw != 0;
    TmSide side = load_side(params, 0, 100.0f);
    side.psize = 0.2f;
    side.close_market = true;
    side.secondary_close_market = true;
    side.secondary_close_qty = 0.1f;
    side.close_is_exposure_reducer = true;
    side.close_is_twel_reducer = true;
    side.close_is_unstuck_reducer = true;
    install_hsl_panic_close(side, is_long, 9999, 10001, 0.01f);

    output[0] = float(side.close_ticks);
    output[1] = side.close_price;
    output[2] = side.close_qty;
    output[3] = side.secondary_close_qty;
    output[4] = float(side.close_market);
    output[5] = float(side.secondary_close_market);
    output[6] = float(side.close_is_exposure_reducer);
    output[7] = float(side.close_is_twel_reducer);
    output[8] = float(side.close_is_unstuck_reducer);
    output[9] = float(side.close_is_panic);
}
"""
    output = torch.zeros(10, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_source_py() + probe_kernel
    )
    library.passivbot_tm_hsl_panic_state_probe(
        params,
        output,
        1 if side == "long" else 0,
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()

    values = output.cpu().numpy()
    expected_ticks = 9998.0 if side == "long" else 10002.0
    assert values[0] == expected_ticks
    assert values[1] == pytest.approx(expected_ticks * 0.01)
    assert values[2] == pytest.approx(0.2)
    assert values[3:9].tolist() == [0.0] * 6
    assert values[9] == 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_mps_tm_dual_side_market_entries_respect_position_mode(hedge_mode):
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.001)
    row[16] = 10.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=hedge_mode,
        hsl_enabled=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0
    assert (output["short_psize"].item() > 0.0) is hedge_mode


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_position_unchanged_includes_open_tail(strategy_kind, side):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
    else:
        high[3] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(initial_ema_dist=0.01)
        runner_cls = MpsTrailingMartingaleRunner
    else:
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 0.0,
                "offset": 0.01,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    position_size = output["psize" if side == "long" else "short_psize"]
    assert position_size.item() > 0.0
    expected_open_tail_ms = output["last_eq_ts"] - output["last_fill_ts"]
    assert expected_open_tail_ms.item() > 0.0
    assert output["position_unchanged_max_ms"].item() == pytest.approx(
        expected_open_tail_ms.item()
    )
    expected_held_ms = output["last_eq_ts"] - output["first_fill_ts"]
    assert output["held_count"].item() == 1.0
    assert output["held_sum_ms"].item() == pytest.approx(expected_held_ms.item())
    assert output["held_max_ms"].item() == pytest.approx(expected_held_ms.item())
    expected_long = 0.1 if side == "long" else 0.0
    expected_short = 0.1 if side == "short" else 0.0
    assert output["entry_initial_balance_pct_long"].item() == pytest.approx(
        expected_long
    )
    assert output["entry_initial_balance_pct_short"].item() == pytest.approx(
        expected_short
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_hsl_panics_and_permanently_halts(strategy_kind, side):
    count = 30
    close = np.full(count, 100.0)
    close[8:] = 70.0 if side == "long" else 130.0
    high = close * 1.02
    low = close * 0.98
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        baseline = _tm_single_row(initial_ema_dist=0.0)
        baseline[6] = 0.5
        baseline[7] = 0.5
        baseline[16] = 0.5
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsTrailingMartingaleRunner
    else:
        baseline = _single_coin_param_row(
            {
                "base_qty_pct": 0.5,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 1.0,
                "offset": 0.0,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsEmaAnchorRunner
    baseline[keys.index("hsl_signal_mode")] = 2.0
    hsl = list(baseline)
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.01,
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }.items():
        hsl[keys.index(key)] = value
    restarting_hsl = list(hsl)
    restarting_hsl[keys.index("hsl_restart_policy")] = 0.0
    restarting_hsl[keys.index("hsl_cooldown_minutes_after_red")] = 2.0
    zero_cooldown_hsl = list(hsl)
    zero_cooldown_hsl[keys.index("hsl_restart_policy")] = 0.0
    unscaled_coin_hsl = list(hsl)
    unscaled_coin_hsl[keys.index("hsl_red_threshold")] = 0.6
    scaled_coin_hsl = list(unscaled_coin_hsl)
    scaled_coin_hsl[keys.index("hsl_slot_count")] = 4.0
    capped_coin_hsl = list(hsl)
    capped_coin_hsl[keys.index("hsl_restart_policy")] = 1.0
    capped_coin_hsl[keys.index("hsl_cooldown_minutes_after_red")] = 2.0
    capped_coin_hsl[keys.index("hsl_slot_count")] = 4.0
    tiny_threshold_hsl = list(hsl)
    tiny_threshold_hsl[keys.index("hsl_red_threshold")] = 1.0e-8
    negative_span_hsl = list(hsl)
    negative_span_hsl[keys.index("hsl_ema_span_minutes")] = -2.0
    recursive_close_hsl = list(hsl)
    if strategy_kind == "trailing_martingale":
        recursive_close_hsl[keys.index("close_retracement_base_pct")] = 0.0
    inactive = list(baseline)
    rows = (
        [
            baseline + inactive,
            hsl + inactive,
            restarting_hsl + inactive,
            zero_cooldown_hsl + inactive,
            unscaled_coin_hsl + inactive,
            scaled_coin_hsl + inactive,
            capped_coin_hsl + inactive,
            tiny_threshold_hsl + inactive,
            negative_span_hsl + inactive,
            recursive_close_hsl + inactive,
        ]
        if side == "long"
        else [
            inactive + baseline,
            inactive + hsl,
            inactive + restarting_hsl,
            inactive + zero_cooldown_hsl,
            inactive + unscaled_coin_hsl,
            inactive + scaled_coin_hsl,
            inactive + capped_coin_hsl,
            inactive + tiny_threshold_hsl,
            inactive + negative_span_hsl,
            inactive + recursive_close_hsl,
        ]
    )
    runner = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        pnl_lookback_bars=5,
        hsl_ema_tail_enabled=True,
        hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True,
    )
    output = runner.run(np.asarray(rows, dtype=np.float64))
    generic_output = None
    if strategy_kind == "trailing_martingale":
        assert runner.shader_topology == f"{side}_hsl"
        generic_runner = runner_cls(
            market,
            run,
            data,
            long_enabled=side == "long",
            short_enabled=side == "short",
            pnl_lookback_bars=5,
            hsl_ema_tail_enabled=True,
            hsl_raw_drawdown_enabled=True,
            hsl_raw_tail_enabled=True,
        )
        generic_runner.shader_topology = "generic"
        generic_output = generic_runner.run(np.asarray(rows, dtype=np.float64))
    market_runner = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        taker_fee=0.01,
        market_order_slippage_pct=0.02,
        hsl_panic_market_long=side == "long",
        hsl_panic_market_short=side == "short",
    )
    market_output = market_runner.run(
        np.asarray([rows[1]], dtype=np.float64)
    )
    ordinary_market_runner = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        taker_fee=0.01,
        market_order_slippage_pct=0.02,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.001,
        hsl_panic_market_long=side == "long",
        hsl_panic_market_short=side == "short",
    )
    ordinary_market_output = ordinary_market_runner.run(
        np.asarray([rows[1]], dtype=np.float64)
    )
    gated_output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.0,
    ).run(np.asarray([rows[9]], dtype=np.float64))
    torch.mps.synchronize()

    if generic_output is not None:
        assert output.keys() == generic_output.keys()
        for key in output:
            torch.testing.assert_close(
                output[key].cpu(),
                generic_output[key].cpu(),
                rtol=1.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() > 0.0
    assert output[size_key][1].item() == 0.0
    assert output[size_key][2].item() > 0.0
    assert output[size_key][3].item() == 0.0
    assert output[size_key][4].item() > 0.0
    assert output[size_key][5].item() == 0.0
    assert output[size_key][6].item() > 0.0
    assert output[size_key][7].item() == 0.0
    assert output["balance"][7].item() < 990.0
    assert output[size_key][8].item() == 0.0
    assert output["balance"][8].item() == pytest.approx(
        output["balance"][1].item(), abs=1.0e-4
    )
    assert output["day_volume"][1].sum().item() > 1.0
    trigger_key = f"hsl_triggers_{side}"
    other_trigger_key = (
        "hsl_triggers_short" if side == "long" else "hsl_triggers_long"
    )
    restart_key = f"hsl_restarts_{side}"
    assert output[trigger_key][0].item() == 0.0
    assert output[trigger_key][1].item() == 1.0
    assert output[other_trigger_key][1].item() == 0.0
    assert output[restart_key][1].item() == 0.0
    assert output[restart_key][2].item() >= 1.0
    assert output["hsl_tier_samples_total"][1].item() > 0.0
    assert output["hsl_tier_samples_red"][1].item() > 0.0
    assert output["hsl_duration_count"][1].item() == 1.0
    assert output["hsl_duration_max_steps"][1].item() > 0.0
    assert output["hsl_trigger_drawdown_sum"][1].item() > 0.0
    assert output["hsl_trigger_drawdown_count"][1].item() == 1.0
    assert output["hsl_flatten_time_count"][1].item() == 1.0
    assert output["hsl_panic_close_loss_sum"][1].item() > 0.0
    assert output["hsl_panic_close_loss_max"][1].item() > 0.0
    assert output["hsl_panic_loss_drawdown_count"][1].item() == 1.0
    assert output["hsl_panic_loss_drawdown_min"][1].item() > 0.0
    assert output["hsl_panic_loss_drawdown_sum"][1].item() > 0.0
    assert output["hsl_panic_loss_drawdown_max"][1].item() > 0.0
    assert output[f"hsl_strategy_eq_recovery_max_ms_{side}"][1].item() > 0.0
    assert output[f"hsl_drawdown_ema_mean_worst_1pct_{side}"][1].item() > 0.0
    assert output[f"hsl_drawdown_raw_max_{side}"][1].item() > 0.0
    assert output[
        f"hsl_drawdown_raw_mean_worst_1pct_{side}"
    ][1].item() > 0.0
    assert output[
        f"hsl_drawdown_ema_mean_worst_1pct_{'short' if side == 'long' else 'long'}"
    ][1].item() == 0.0
    assert output[size_key][9].item() == 0.0
    assert output["hsl_panic_close_loss_sum"][9].item() > 0.0
    assert output["hsl_panic_loss_drawdown_count"][9].item() == 1.0
    assert gated_output[size_key].item() == 0.0
    assert gated_output["hsl_panic_close_loss_sum"].item() > 0.0
    assert market_runner.settings[15].item() == pytest.approx(0.01)
    assert market_runner.settings[16].item() == pytest.approx(0.02)
    assert market_runner.settings[17].item() == float(side == "long")
    assert market_runner.settings[18].item() == float(side == "short")
    assert market_output[size_key].item() == 0.0
    assert market_output["balance"].item() < output["balance"][1].item()
    assert (
        market_output["hsl_panic_close_loss_sum"].item()
        > output["hsl_panic_close_loss_sum"][1].item()
    )
    assert ordinary_market_runner.settings[19].item() == 1.0
    assert ordinary_market_output[size_key].item() == 0.0
    assert ordinary_market_output[trigger_key].item() == 1.0
    assert ordinary_market_output["hsl_flatten_time_count"].item() == 1.0
    assert ordinary_market_output["hsl_panic_close_loss_sum"].item() > 0.0
    assert torch.isfinite(ordinary_market_output["balance"]).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("signal_mode", ["unified", "pside", "coin"])
def test_mps_dual_side_single_coin_hsl_respects_signal_scope(
    strategy_kind, signal_mode
):
    count = 48
    close = np.full(count, 100.0)
    close[8:20] = 70.0
    close[20:32] = 100.0
    close[32:] = 130.0
    high = close * 1.02
    low = close * 0.98
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        baseline = _tm_single_row(initial_ema_dist=0.0)
        baseline[6] = 0.5
        baseline[7] = 0.5
        baseline[16] = 0.5
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsTrailingMartingaleRunner
    else:
        baseline = _single_coin_param_row(
            {
                "base_qty_pct": 0.5,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 1.0,
                "offset": 0.0,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsEmaAnchorRunner
    hsl = list(baseline)
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.01,
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": {"unified": 0.0, "pside": 1.0, "coin": 2.0}[
            signal_mode
        ],
        "hsl_slot_count": 1.0,
    }.items():
        hsl[keys.index(key)] = value
    inactive = list(baseline)
    inactive[keys.index("hsl_signal_mode")] = hsl[
        keys.index("hsl_signal_mode")
    ]
    long_hsl = list(hsl)
    short_hsl = list(hsl)
    if signal_mode == "unified":
        qty_key = (
            "entry_initial_qty_pct"
            if strategy_kind == "trailing_martingale"
            else "base_qty_pct"
        )
        short_hsl[keys.index(qty_key)] = 0.1
    mismatched_short_hsl = list(short_hsl)
    mismatched_short_hsl[keys.index("hsl_signal_mode")] = (
        hsl[keys.index("hsl_signal_mode")] + 1.0
    ) % 3.0
    rows = np.asarray(
        [
            long_hsl + inactive,
            inactive + short_hsl,
            long_hsl + short_hsl,
            long_hsl + mismatched_short_hsl,
        ],
        dtype=np.float64,
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hsl_ema_tail_enabled=True,
        hsl_raw_drawdown_enabled=True,
        hsl_raw_tail_enabled=True,
    ).run(rows)
    torch.mps.synchronize()

    assert output["hsl_triggers_long"][0].item() == (
        0.0 if signal_mode == "unified" else 1.0
    )
    assert output["hsl_triggers_short"][0].item() == 0.0
    assert output["hsl_triggers_long"][1].item() == 0.0
    assert output["hsl_triggers_short"][1].item() == (
        0.0 if signal_mode == "unified" else 1.0
    )
    assert output["hsl_triggers_long"][2].item() == 1.0
    assert output["hsl_triggers_short"][2].item() == 1.0
    assert output["hsl_trigger_drawdown_count"][2].item() == 2.0
    assert output["hsl_strategy_eq_recovery_max_ms_long"][2].item() > 0.0
    assert output["hsl_strategy_eq_recovery_max_ms_short"][2].item() > 0.0
    assert output["hsl_drawdown_ema_mean_worst_1pct_long"][2].item() > 0.0
    assert output["hsl_drawdown_ema_mean_worst_1pct_short"][2].item() > 0.0
    assert output["hsl_drawdown_raw_max_long"][2].item() > 0.0
    assert output["hsl_drawdown_raw_max_short"][2].item() > 0.0
    assert output["hsl_drawdown_raw_mean_worst_1pct_long"][2].item() > 0.0
    assert output["hsl_drawdown_raw_mean_worst_1pct_short"][2].item() > 0.0
    if signal_mode == "unified":
        assert output["hsl_panic_loss_drawdown_count"][2].item() >= 1.0
    else:
        assert output["hsl_panic_loss_drawdown_count"][2].item() == 2.0
    if signal_mode == "unified":
        assert output["short_psize"][0].item() > 0.0
        assert output["psize"][1].item() > 0.0
    assert not output["alive"][3].item()
    assert output["liq_step"][3].item() == 0
    assert output["balance"][3].item() == 0.0
    assert output["fill_count"][3].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("strategy_kind", "market_orders_allowed"),
    [
        ("ema_anchor", False),
        ("ema_anchor", True),
        ("trailing_martingale", False),
        ("trailing_martingale", True),
    ],
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_auto_unstuck_reduces_eligible_position(
    strategy_kind, market_orders_allowed, side
):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 100.01)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
    else:
        high[:] = 100.0
        high[3] = 102.0
        low[:] = 99.99
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)

    def candidate(unstuck_enabled, *, ema_gating=False, ema_dist=0.0):
        if strategy_kind == "trailing_martingale":
            row = _tm_single_row(
                initial_ema_dist=0.01,
                gate_initial=1.0,
                gate_reentry=0.0,
                entry_gate=True,
                unstuck_enabled=unstuck_enabled,
                unstuck_ema_gating_enabled=ema_gating,
                unstuck_close_pct=0.1,
                unstuck_ema_dist=ema_dist,
                unstuck_loss_allowance_pct=0.2,
                unstuck_threshold=0.5,
            )
            row[6] = 1.0
            row[7] = 10.0
            row[11] = 0.001 if market_orders_allowed else 0.0
            row[16] = 0.5
            row[20] = 0.001 if market_orders_allowed else 0.0
            row[23] = 100.0
            return row + row
        row = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            100.0,
            1.0,
        ]
        row += _single_coin_exposure_fields(entry_gate=True)
        row += _tm_twel_enforcer_fields(
            unstuck_enabled=unstuck_enabled,
            unstuck_ema_gating_enabled=ema_gating,
            unstuck_close_pct=0.1,
            unstuck_ema_dist=ema_dist,
            unstuck_loss_allowance_pct=0.2,
            unstuck_threshold=0.5,
        )
        return row + row

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.05,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    ).run(
        np.asarray(
            [
                candidate(False),
                candidate(True),
                candidate(True, ema_gating=True, ema_dist=0.1),
            ],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()

    key = "psize" if side == "long" else "short_psize"
    remaining = output[key].cpu().numpy()
    # At the 101 short-entry limit, exact Rust's strict TWEL gate retains 9
    # whole contracts; both strategy kernels must share that result.
    initial_size = 9.0 if side == "short" else 10.0
    assert remaining[0] == pytest.approx(initial_size)
    assert remaining[1] == pytest.approx(initial_size - 1.0)
    assert remaining[2] == pytest.approx(initial_size)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_auto_unstuck_scales_loss_to_own_allowance(
    strategy_kind, side
):
    count = 6
    close = np.full(count, 100.0)
    if side == "long":
        close[3:] = 98.0
        high = close + 0.01
        low = close.copy()
    else:
        close[3:] = 102.0
        high = close.copy()
        low = close - 0.01
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)

    def candidate(*, unstuck_enabled, loss_allowance_pct):
        if strategy_kind == "trailing_martingale":
            row = _tm_single_row(
                initial_ema_dist=0.01,
                gate_initial=1.0,
                gate_reentry=0.0,
                entry_gate=True,
                unstuck_enabled=unstuck_enabled,
                unstuck_ema_gating_enabled=False,
                unstuck_close_pct=1.0,
                unstuck_loss_allowance_pct=loss_allowance_pct,
                unstuck_threshold=0.5,
            )
            row[6] = 1.0
            row[7] = 10.0
            row[11] = 0.0
            row[16] = 0.5
            row[20] = 0.0
            row[23] = 100.0
            return row + row
        row = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            100.0,
            1.0,
        ]
        row += _single_coin_exposure_fields(entry_gate=True)
        row += _tm_twel_enforcer_fields(
            unstuck_enabled=unstuck_enabled,
            unstuck_ema_gating_enabled=False,
            unstuck_close_pct=1.0,
            unstuck_loss_allowance_pct=loss_allowance_pct,
            unstuck_threshold=0.5,
        )
        return row + row

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.05,
    ).run(
        np.asarray(
            [
                candidate(unstuck_enabled=False, loss_allowance_pct=0.0005),
                candidate(unstuck_enabled=True, loss_allowance_pct=0.0005),
                candidate(unstuck_enabled=True, loss_allowance_pct=0.02),
            ],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()

    key = "psize" if side == "long" else "short_psize"
    remaining = output[key].cpu().numpy()
    initial_size = 9.0 if side == "short" else 10.0
    assert remaining[0] == pytest.approx(initial_size)
    assert remaining[1] == pytest.approx(initial_size - 1.0)
    assert remaining[2] == pytest.approx(0.0)

    strict_output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.0005,
    ).run(
        np.asarray(
            [candidate(unstuck_enabled=True, loss_allowance_pct=0.02)],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()
    assert strict_output[key].item() == pytest.approx(initial_size)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    disabled[keys.index("forager_volume_drop_pct")] = 1.0
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
    torch.mps.synchronize()
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
    strict_output = strict_runner.run(
        np.asarray([enabled], dtype=np.float64)
    )
    torch.mps.synchronize()
    assert strict_output[key].item() == pytest.approx(remaining[0])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_auto_unstuck_selects_one_least_stuck_side(strategy_kind):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 100.01)
    low = np.full(count, 99.99)
    high[3] = 102.0
    low[3] = 98.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)

    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(
            initial_ema_dist=0.01,
            gate_initial=1.0,
            gate_reentry=0.0,
            entry_gate=True,
            unstuck_enabled=True,
            unstuck_ema_gating_enabled=False,
            unstuck_close_pct=0.1,
            unstuck_loss_allowance_pct=0.2,
            unstuck_threshold=0.5,
        )
        row[6] = 1.0
        row[7] = 10.0
        row[11] = 0.0
        row[16] = 0.5
        row[20] = 0.0
        row[23] = 100.0
    else:
        row = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            100.0,
            1.0,
        ]
        row += _single_coin_exposure_fields(entry_gate=True)
        row += _tm_twel_enforcer_fields(
            unstuck_enabled=True,
            unstuck_ema_gating_enabled=False,
            unstuck_close_pct=0.1,
            unstuck_loss_allowance_pct=0.2,
            unstuck_threshold=0.5,
        )

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == pytest.approx(9.0)
    assert output["short_psize"].item() == pytest.approx(9.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_equal_unstuck_twel_reducers_keep_nearer_twel(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
    else:
        high[3] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        gate_initial=1.0,
        gate_reentry=0.0,
        entry_gate=True,
        twel_enforcer_enabled=True,
        threshold=0.5,
        unstuck_enabled=True,
        unstuck_ema_gating_enabled=False,
        unstuck_close_pct=0.5,
        unstuck_loss_allowance_pct=0.2,
        unstuck_threshold=0.5,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.05,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The equal-size TWEL reducer is one price band more executable than the
    # auto-unstuck reducer. The next candle crosses TWEL but only touches
    # unstuck, proving selection follows Rust's reachability tie-break.
    # The strict entry gate starts the short fixture at 9 rather than 10, so
    # its equal-size selected reducer leaves 4 contracts instead of 5.
    expected_size = 5.0 if side == "long" else 4.0
    assert output[size_key].item() == pytest.approx(expected_size)
    assert output["total_wallet_exposure_max"].item() > 0.0
    fill_days = output["day_has_fill"]
    assert torch.isfinite(output["day_net_pnl"][fill_days]).all()
    assert (output["day_last_fill_balance"][fill_days] > 0.0).all()
    assert (output["day_fill_count"][fill_days] >= 1.0).all()
    assert torch.equal(
        output["day_fill_count"], output["day_fill_count"].round()
    )
    assert (
        output["total_wallet_exposure_max"].item()
        >= output["total_wallet_exposure_mean"].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_unstuck_finalization_keeps_valid_ordinary_close_remainder():
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3] = 98.0
    high[4] = 201.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.5, 0.01, 0.5, 500.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        gate_initial=1.0,
        gate_reentry=0.0,
        entry_gate=True,
        threshold=1.0,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.2,
        unstuck_enabled=True,
        unstuck_ema_gating_enabled=False,
        unstuck_close_pct=0.7,
        unstuck_loss_allowance_pct=0.2,
        unstuck_threshold=0.5,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.3
    row[16] = 1.0
    row[20] = 0.01
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        max_realized_loss_pct=0.05,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # At the 100 touch, the requested unstuck close is 7 and its minimum is 5.
    # The ordinary close near 198 has a 3-unit quantity above its ~2.5 minimum,
    # so exact finalization keeps the unstuck selection key at 7 instead of
    # promoting it to the full 10. The genuinely larger WEL reducer therefore
    # wins and leaves less than the unstuck request would have left.
    assert output["psize"].item() < 3.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    # As in the directional kernel regression above, the valid ordinary close
    # keeps unstuck's finalized selection key below the larger WEL reducer.
    assert output["psize"].item() < 3.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_exposure_headroom_and_entry_gate(
    strategy_kind, side
):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)

    def row(*, legacy_raw, entry_gate):
        exposure = _single_coin_exposure_fields(
            allowance_pct=0.5,
            legacy_raw=legacy_raw,
            entry_gate=entry_gate,
            threshold=0.5,
        )
        if strategy_kind == "trailing_martingale":
            values = _tm_row(
                initial_ema_dist=0.01,
                gate_initial=1.0,
                gate_reentry=1.0,
            )
            values[6] = 1.0
            return (
                values
                + exposure
                + _tm_wel_enforcer_fields()
                + _tm_twel_enforcer_fields()
            )
        values = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            0.0,
            1.0,
        ]
        return values + exposure + _tm_twel_enforcer_fields()

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    runner = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    )
    rows = [
        row(legacy_raw=False, entry_gate=False),
        row(legacy_raw=True, entry_gate=False),
        row(legacy_raw=True, entry_gate=True),
    ]
    output = runner.run(
        np.asarray([values + values for values in rows], dtype=np.float64)
    )
    torch.mps.synchronize()
    sizes = (
        output["psize"] if side == "long" else output["short_psize"]
    ).cpu().numpy()

    assert sizes[0] > 0.0
    assert sizes[1] > sizes[0] * 1.4
    assert sizes[2] < sizes[0] * 0.6


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_position_exposure_repair_reduces_strictly_below_target(
    side, market_orders_allowed
):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    baseline = _tm_single_row()
    baseline[6] = 1.0
    baseline[7] = 10.0
    baseline[11] = 0.001 if market_orders_allowed else 0.0
    baseline[16] = 10.0
    baseline[20] = 0.001 if market_orders_allowed else 0.0
    repaired = list(baseline)
    repaired[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "wel_enforcer_enabled"
        )
    ] = 1.0
    repaired[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "wel_enforcer_threshold"
        )
    ] = 0.5
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    ).run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    repaired_we = (
        sizes[1]
        * output["pprice" if side == "long" else "short_pprice"][1].item()
        / output["balance"][1].item()
    )
    assert 0.0 < sizes[1] < sizes[0]
    assert 0.45 < repaired_we < 0.5
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )
    other_side = "short" if side == "long" else "long"
    assert torch.allclose(output[f"profit_sum_{side}"], output["profit_sum"])
    assert torch.allclose(output[f"loss_sum_{side}"], output["loss_sum"])
    assert (output[f"profit_sum_{other_side}"] == 0.0).all()
    assert (output[f"loss_sum_{other_side}"] == 0.0).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_single_coin_total_exposure_repair(side, market_orders_allowed):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    # A coarse quantity step makes the semantic distinction observable:
    # TWEL repair rounds up to the target, while WEL repair intentionally
    # takes one additional step to finish strictly below its target.
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    baseline = _tm_single_row(entry_gate=False, threshold=0.5)
    baseline[6] = 1.0
    baseline[7] = 10.0
    baseline[11] = 0.001 if market_orders_allowed else 0.0
    baseline[16] = 10.0
    baseline[20] = 0.001 if market_orders_allowed else 0.0
    repaired = list(baseline)
    repaired[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 1.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    ).run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    assert sizes[1] == pytest.approx(5.0)
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_ema_single_coin_total_exposure_repair(side, market_orders_allowed):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    if side == "long":
        close[2:] = 99.9
        high[2:] = 99.9
        low[3] = 98.0
        low[4:] = 99.9
    else:
        close[2:] = 100.1
        high[3] = 102.0
        high[4:] = 100.1
        low[2:] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 1.0,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        # Keep the ordinary close passive in the market-enabled case so this
        # fixture isolates the immediately executable TWEL reducer.
        "offset": 0.01 if market_orders_allowed else 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 0.0,
    }
    baseline = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    repaired_values = dict(values, twel_enforcer_enabled=1.0)
    repaired = _single_coin_param_row(
        repaired_values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
    )
    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=0.001,
    ).run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 8.0
    pprice_key = "pprice" if side == "long" else "short_pprice"
    repaired_we = (
        sizes[1]
        * output[pprice_key][1].item()
        / output["balance"][1].item()
    )
    assert 0.0 < sizes[1] < sizes[0]
    assert 0.35 < repaired_we <= 0.51
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )
    other_side = "short" if side == "long" else "long"
    assert torch.allclose(output[f"profit_sum_{side}"], output["profit_sum"])
    assert torch.allclose(output[f"loss_sum_{side}"], output["loss_sum"])
    assert (output[f"profit_sum_{other_side}"] == 0.0).all()
    assert (output[f"loss_sum_{other_side}"] == 0.0).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_ema_realized_loss_gate_blocks_lossy_total_exposure_repair(
    side, market_orders_allowed
):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    if side == "long":
        close[2:] = 99.9
        high[2:] = 99.9
        low[3] = 98.0
        low[4:] = 99.9
    else:
        close[2:] = 100.1
        high[3] = 102.0
        high[4:] = 100.1
        low[2:] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
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
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 1.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    kwargs = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "market_orders_allowed": market_orders_allowed,
        "market_order_near_touch_threshold": 0.001,
        "market_order_slippage_pct": 0.01,
        "taker_fee": 0.01,
    }

    ungated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=1.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=0.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key][0].item() < gated[size_key][0].item()
    assert gated[size_key][0].item() > 8.0
    assert gated["balance"][0].item() >= ungated["balance"][0].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_realized_loss_gate_shares_budget_between_sides():
    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
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
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 1.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
        max_realized_loss_pct=0.0031,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    sizes = sorted([output["psize"][0].item(), output["short_psize"][0].item()])
    assert sizes[0] < 8.0
    assert sizes[1] > 8.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_realized_loss_gate_reserves_unfilled_batch_loss():
    count = 6
    close = np.array([100.0, 100.0, 100.0, 100.0, 99.9, 99.9])
    high = np.array([100.0, 100.0, 100.0, 100.01, 99.9, 99.9])
    low = np.array([100.0, 100.0, 100.0, 99.99, 99.9, 99.9])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
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
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 1.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    common = {
        "long_enabled": True,
        "short_enabled": True,
        "hedge_mode": True,
    }

    ungated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=1.0, **common
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=0.0031, **common
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    sizes = {
        "ungated_long": ungated["psize"][0].item(),
        "ungated_short": ungated["short_psize"][0].item(),
        "gated_long": gated["psize"][0].item(),
        "gated_short": gated["short_psize"][0].item(),
    }
    assert sizes["ungated_long"] > 8.0, sizes
    assert sizes["ungated_short"] < 8.0, sizes
    assert sizes["gated_long"] > 8.0, sizes
    assert sizes["gated_short"] > 8.0, sizes


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_zero_loss_budget_blocks_loss_below_balance_ulp():
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 100.01, 100.01, 100.0])
    low = np.array([100.0, 100.0, 100.0, 99.99, 99.99, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.000001, 0.01, 0.000001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        100_000.0,
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 0.000001,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)

    ungated = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        max_realized_loss_pct=1.0,
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        max_realized_loss_pct=0.0,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert ungated["psize"][0].item() == pytest.approx(0.0)
    assert gated["psize"][0].item() > 0.0009


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_realized_loss_gate_blocks_lossy_total_exposure_repair(
    side, market_orders_allowed
):
    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    high[3], low[3] = 105.0, 95.0
    if side == "long":
        close[4:] = 98.0
        high[4:] = 98.0
        low[4:] = 97.9
    else:
        close[4:] = 102.0
        high[4:] = 102.1
        low[4:] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.001 if market_orders_allowed else 0.0
    row[16] = 10.0
    row[20] = 0.001 if market_orders_allowed else 0.0
    kwargs = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "market_orders_allowed": market_orders_allowed,
        "market_order_near_touch_threshold": 0.001,
        "market_order_slippage_pct": 0.01,
        "taker_fee": 0.01,
    }

    ungated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=1.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=0.1, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key][0].item() < gated[size_key][0].item()
    assert gated[size_key][0].item() > 9.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_loss_gate_rebuilds_profitable_ordinary_close_after_reducer(side):
    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    high[3], low[3] = 105.0, 95.0
    if side == "long":
        close[4] = high[4] = 98.0
        low[4] = 97.9
        high[5] = 100.1
        low[5] = 98.0
    else:
        close[4] = low[4] = 102.0
        high[4] = 102.1
        high[5] = 102.0
        low[5] = 99.9
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 0.01
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.1,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_realized_loss_gate_blocks_fee_only_ordinary_close(
    side, market_orders_allowed
):
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[2] = 99.99
        close[3:] = 99.0
        high[3] = 99.0
        low[3] = 98.9
        high[4] = 99.6
        low[4] = 99.0
    else:
        high[2] = 100.01
        close[3:] = 101.0
        high[3] = 101.1
        low[3] = 101.0
        high[4] = 101.0
        low[4] = 100.4
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.0,
        gate_initial=False,
        gate_reentry=False,
        entry_gate=False,
    )
    row[6] = 0.1
    row[7] = 10.0
    row[11] = 0.001 if market_orders_allowed else 0.0
    row[16] = -0.005
    row[17] = 0.0
    row[20] = 0.001 if market_orders_allowed else 0.0
    kwargs = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "market_orders_allowed": market_orders_allowed,
        "market_order_near_touch_threshold": 0.001,
        "market_order_slippage_pct": 0.01,
        "taker_fee": 0.01,
    }

    ungated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=1.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=0.1, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key][0].item() == pytest.approx(0.0)
    assert gated[size_key][0].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_equal_wel_twel_reducers_keep_nearer_wel(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The WEL reducer rests at 100.00 and is not strictly touched by the next
    # candle.  Choosing the farther TWEL reducer would close half the position.
    assert output[size_key][0].item() == pytest.approx(10.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_finalized_reducer_tie_keeps_nearer_wel(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 9.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.05,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.1,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0

    twel_only = list(row)
    twel_only[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "wel_enforcer_enabled"
        )
    ] = 0.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(
        np.asarray(
            [row + row, twel_only + twel_only], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Raw TWEL is larger, but both candidates absorb their sub-minimum tail and
    # finalize to the full position. Exact Rust then selects the nearer WEL,
    # which is only touched (not strictly crossed) by the next candle.
    assert output[size_key][0].item() > 9.0
    assert output[size_key][1].item() == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_offsets_raw_close_before_tick_quantization(side):
    count = 6
    close = np.full(count, 100.019)
    high = np.full(count, 100.019)
    low = np.full(count, 100.019)
    high[3] = 105.0
    low[3] = 95.0
    if side == "long":
        close[4] = 99.95
        high[4] = 99.955
        low[4] = 99.90
    else:
        close[4] = 100.08
        high[4] = 100.10
        low[4] = 100.075
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Exact raw-first quantization emits 99.96 long / 100.07 short, neither of
    # which is strictly touched.  Rounding the close first moves the order one
    # tick farther and incorrectly fills it.
    assert output[size_key][0].item() == pytest.approx(10.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_repair_preserves_triggered_trailing_close(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 99.9, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 98.0, 95.0, 100.0])
    if side == "long":
        close[4] = 99.0
    else:
        close[4] = 101.0
        high[4] = 102.0
        low[4] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.2
    row[16] = 0.0
    row[20] = 0.001
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # TWEL repair alone leaves five contracts.  The independently triggered
    # trailing close must therefore reduce the position further in the same
    # candle without exceeding the original position.
    assert 0.0 < output[size_key][0].item() < 5.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_wel_repair_suppresses_triggered_trailing_close(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 99.9, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 98.0, 95.0, 100.0])
    if side == "long":
        close[4] = 99.0
    else:
        close[4] = 101.0
        high[4] = 102.0
        low[4] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    trailing = _tm_single_row(
        entry_gate=False,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.5,
    )
    trailing[6] = 1.0
    trailing[7] = 10.0
    trailing[11] = 0.0
    trailing[15] = 0.2
    trailing[16] = 0.0
    trailing[20] = 0.001
    trailing[23] = 100.0
    nontrailing = list(trailing)
    nontrailing[16] = 10.0
    nontrailing[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([trailing + trailing, nontrailing + nontrailing]))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Strategy WEL takes precedence inside calc_closes_*; unlike an appended
    # TWEL reducer, it must not retain a triggered trailing close.
    assert output[size_key][0].item() == pytest.approx(
        output[size_key][1].item()
    )
    assert output[size_key][0].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_and_trailing_close_absorb_dust_in_signed_order(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 99.9, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 98.0, 95.0, 100.0])
    if side == "long":
        close[4] = 99.0
        threshold = 0.4
        entry_price = 99.0
    else:
        close[4] = 101.0
        high[4] = 102.0
        low[4] = 100.1
        threshold = 0.5
        entry_price = 101.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 2.0, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=threshold,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.3
    row[16] = 0.0
    row[20] = 0.001
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() == pytest.approx(0.0)

    # Both orders are generated on the retracement candle (99 long / 101
    # short), not on the following fill candle.
    reducer_price = 98.95 if side == "long" else 101.06
    ordinary_price = 99.0 if side == "long" else 101.0
    balance = 1_000.0 - 10.0 * entry_price * market.maker_fee
    expected_volume = 10.0 * entry_price / balance
    reducer_pnl = 6.0 * (
        reducer_price - entry_price
        if side == "long"
        else entry_price - reducer_price
    )
    balance += reducer_pnl - 6.0 * reducer_price * market.maker_fee
    expected_volume += 6.0 * reducer_price / balance
    ordinary_pnl = 4.0 * (
        ordinary_price - entry_price
        if side == "long"
        else entry_price - ordinary_price
    )
    balance += ordinary_pnl - 4.0 * ordinary_price * market.maker_fee
    expected_volume += 4.0 * ordinary_price / balance
    # The TWEL order is marketable and therefore has a negative signed
    # distance; exact canonical ordering executes it before the touch close.
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_grid_uses_pre_repair_strategy_state(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 95.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 1.0
    row[16] = 0.0
    row[17] = 0.08
    row[20] = 0.0
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The original-position grid is outside this candle while the TWEL order
    # fills. Rebuilding from the post-TWEL position moves the grid inward and
    # incorrectly closes the remainder.
    assert output[size_key][0].item() > 3.5


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_position_reducer_reachability_survives_grid_pruning(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.10
    candidate[17] = 0.10
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    pprice_key = "pprice" if side == "long" else "short_pprice"
    size = output[size_key].item()
    repaired_we = size * output[pprice_key].item() / output["balance"].item()
    assert 0.0 < repaired_we < 0.5
    assert output["day_volume"].sum().item() > 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_grid_can_fill_without_off_tick_reducer(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.0
    candidate[17] = 0.0
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Long: the 100.4 touch makes a reducer at 101 and rebuilt grid at 100;
    # high=100.5 reaches only the grid.  Short is the mirror image: reducer
    # 100, grid 101, low=100.5.  The residual grid quantity therefore closes
    # while the reducer-sized half remains open.
    assert output[size_key].item() == pytest.approx(5.05, abs=0.1)
    assert 1.0 < output["day_volume"].sum().item() < 2.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_off_tick_grid_precedes_reducer_for_volume(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 101.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 99.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.0
    candidate[17] = 0.0
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    # Exact Rust's strict TWEL gate floors the short entry from the strategy's
    # nearest-rounded 9.901 to 9.900 at the 101 limit price.
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
        reducer_price - entry_price
        if side == "long"
        else entry_price - reducer_price
    )
    balance += reducer_pnl - reducer_qty * reducer_price * market.maker_fee
    expected_volume += reducer_qty * reducer_price / balance

    assert output["balance"].item() == pytest.approx(balance, abs=3.0e-4)
    assert output["profit_sum"].item() == pytest.approx(
        grid_pnl + reducer_pnl, abs=3.0e-4
    )
    assert output["loss_sum"].item() == 0.0
    other_side = "short" if side == "long" else "long"
    assert output[f"profit_sum_{side}"].item() == pytest.approx(
        grid_pnl + reducer_pnl, abs=3.0e-4
    )
    assert output[f"loss_sum_{side}"].item() == 0.0
    assert output[f"profit_sum_{other_side}"].item() == 0.0
    assert output[f"loss_sum_{other_side}"].item() == 0.0
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_reducer_consumes_one_recursive_close_slot(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 101.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 99.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[15] = 0.0001
    candidate[16] = 0.0
    candidate[17] = 1.0e-6
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    # One reducer plus 499 ordinary closes exhausts Rust's shared 500-order
    # generation loop.  A fresh 500-rung proxy loop would remove one extra
    # quantity step here.
    expected_size = 4.052 if side == "long" else 4.451
    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(expected_size, abs=2.0e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_small_excess_reducer_crosses_strict_target(side):
    count = 6
    entry_price = 99.0 if side == "long" else 101.0
    close = np.array([100.0, 100.0, 100.0, entry_price, entry_price, entry_price])
    high = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 102.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 98.0, 98.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
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
    target = 0.99999
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=target
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[20] = 0.001
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    pprice_key = "pprice" if side == "long" else "short_pprice"
    exposure = (
        output[size_key].item()
        * output[pprice_key].item()
        / output["balance"].item()
    )
    assert output[size_key].item() > 0.0
    assert exposure < target


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_same_tick_twel_and_grid_use_separate_volume_denominators(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 101.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 99.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.001)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        entry_gate=False, threshold=0.5, twel_enforcer_enabled=True
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.0
    candidate[17] = 0.0
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    # The coarse price step sends both the raw-touch grid and the offset TWEL
    # reducer to the same tick. Exact Rust nevertheless keeps them as separate
    # orders, executing the grid first by order-type ID.
    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.901
    grid_qty = 5.045
    reducer_qty = entry_qty - grid_qty
    close_price = 100.0 if side == "long" else 101.0
    balance_after_entry = 1_000.0 - entry_qty * entry_price * 0.001
    grid_pnl = grid_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_grid = (
        balance_after_entry + grid_pnl - grid_qty * close_price * 0.001
    )
    reducer_pnl = reducer_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_close = (
        balance_after_grid
        + reducer_pnl
        - reducer_qty * close_price * 0.001
    )
    expected_volume = entry_qty * entry_price / balance_after_entry
    expected_volume += grid_qty * close_price / balance_after_grid
    expected_volume += reducer_qty * close_price / balance_after_close
    assert output["balance"].item() == pytest.approx(
        balance_after_close, abs=2.0e-4
    )
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


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
    market = ProxyMarket(
        0.001, price_step, 0.001, min_cost, 1.0, maker_fee, taker_fee
    )
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    runner = MpsTrailingMartingaleMulticoinRunner(
        run,
        data,
        side=side,
        coin_overrides=overrides,
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=market_order_near_touch_threshold,
        market_order_slippage_pct=market_order_slippage_pct,
        collect_coin_fill_counts=collect_coin_fill_counts,
    )
    return runner, candidate, market


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    params = torch.tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device="mps",
    )
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[100.0, 100.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device="mps",
    )
    touch_ticks = torch.full(
        (2, 1, 2), 20_000, dtype=torch.int32, device="mps"
    )
    touch_nearest_ticks = torch.full(
        (2, 1), 20_000, dtype=torch.int32, device="mps"
    )
    touch_min_qty_bits = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    touch_min_qty_relation = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 3] = 100.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = torch.full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    output = torch.zeros(5, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
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
    torch.mps.synchronize()

    # The passive strategy request is 0.5 at price 200, while executable-touch
    # min-cost sizing at market price 100 promotes the emitted reducer to 1.0.
    # The immutable ladder must still start from 10.0 - 0.5, not 10.0 - 1.0.
    assert output.cpu().tolist() == pytest.approx(
        [1.0, 9.5, 1.0, 1.0, 499.0], abs=2.0e-4
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    params = torch.tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device="mps",
    )
    bars = torch.tensor(
        [[[100.0, 100.0, 100.0, 1.0]], [[100.0, 100.0, 100.0, 1.0]]],
        dtype=torch.float32,
        device="mps",
    )
    touch_ticks = torch.full(
        (2, 1, 2), 20_000, dtype=torch.int32, device="mps"
    )
    touch_nearest_ticks = torch.full(
        (2, 1), 20_000, dtype=torch.int32, device="mps"
    )
    touch_min_qty_bits = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    touch_min_qty_relation = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 0] = 0.001
    coin_settings[0, 1] = 0.01
    coin_settings[0, 2] = 0.001
    coin_settings[0, 3] = 100.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = torch.full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    output = torch.zeros(7, dtype=torch.float32, device="mps")
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
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
    torch.mps.synchronize()

    # The passive unstuck request is 0.5 at price 200. Market policy enlarges
    # the emitted reducer to the executable-touch minimum of 1.0 at 100, but
    # external unstuck does not reseed Rust's full-size strategy ladder.
    assert output.cpu().tolist() == pytest.approx(
        [1.0, 10.0, 1.0, 1.0, 1.0, 100.0, 500.0], abs=2.0e-4
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
        values = dict(
            zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate)
        )
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
                [
                    [
                        values[key]
                        for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
                    ]
                ],
                dtype=np.float64,
            )
        )
        torch.mps.synchronize()
        return output

    no_cost = run_case(slippage=0.0, taker_fee=0.0)
    slipped = run_case(slippage=0.01, taker_fee=0.0)
    charged = run_case(slippage=0.01, taker_fee=0.002)
    strict = run_case(
        slippage=0.01, taker_fee=0.002, max_realized_loss_pct=0.0
    )
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_market_reducer_keeps_generation_dust_allocation():
    import passivbot_rust

    probe_kernel = r"""
kernel void passivbot_tm_multicoin_market_reducer_dust_probe(
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
    side.close_recursive_market_mode[0] = true;
    side.close_gen_balance[0] = 1000.0f;
    side.close_gen_allowed_wel[0] = 0.016f;
    side.close_gen_market_price[0] = 100.0f;
    side.close_grid_gen_psize[0] = 0.4f;
    side.close_grid_max_rungs[0] = 1;
    CloseGroup probe_group;
    int probe_group_count = recursive_grid_close_groups_after_reducer(
        false, 0.4f, 40.0f, 1000.0f, 0.016f,
        390, 400, 400, 0.0f, 0,
        config.close_qty_pct, config.close_threshold_base,
        config.close_threshold_we, config.close_threshold_v1h,
        config.close_threshold_v1m, 0.0f, 0.0f,
        0.1f, 0.1f, 0.1f, 40.0f, 1.0f, 0, 0.0f, 1, 0,
        100.0f, true, 0.001f, 0.4f, probe_group
    );
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    float hsl_equity_before_fills = 1000.0f;
    process_tm_multicoin_side_fills(
        side, config, account, fills,
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        coin_settings, coin_overrides, coin_fill_counts,
        0, 1, 1, false, true, false, false,
        1.0f, 0.0f, 0.001f, false, hsl_equity_before_fills
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
    params = torch.tensor(
        [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],
        dtype=torch.float32,
        device="mps",
    )
    fill_ticks = torch.tensor(
        [[[0, 0]], [[2200, 0]]], dtype=torch.int32, device="mps"
    )
    touch_ticks = torch.tensor(
        [[[390, 400]], [[390, 400]]], dtype=torch.int32, device="mps"
    )
    touch_nearest_ticks = torch.full(
        (2, 1), 400, dtype=torch.int32, device="mps"
    )
    touch_min_qty_bits = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    touch_min_qty_relation = torch.zeros(
        (2, 1), dtype=torch.int32, device="mps"
    )
    coin_settings = torch.zeros((1, 13), dtype=torch.float32, device="mps")
    coin_settings[0, 0] = 0.1
    coin_settings[0, 1] = 0.1
    coin_settings[0, 2] = 0.1
    coin_settings[0, 3] = 40.0
    coin_settings[0, 4] = 1.0
    coin_settings[0, 7] = 10.0
    coin_overrides = torch.full(
        (1, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), float("nan"), dtype=torch.float32, device="mps"
    )
    library = torch.mps.compile_shader(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + probe_kernel
    )

    remaining_sizes = []
    for next_close in (100.0, 200.0):
        bars = torch.tensor(
            [
                [[100.0, 40.0, 100.0, 1.0]],
                    [[220.0, 30.0, next_close, 1.0]],
            ],
            dtype=torch.float32,
            device="mps",
        )
        coin_fill_counts = torch.zeros(1, dtype=torch.float32, device="mps")
        output = torch.zeros(7, dtype=torch.float32, device="mps")
        library.passivbot_tm_multicoin_market_reducer_dust_probe(
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
        torch.mps.synchronize()
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_off_tick_grid_precedes_reducer_for_volume(side):
    runner, candidate, market = _tm_multicoin_off_tick_reducer_case(
        side, maker_fee=0.001
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

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
        reducer_price - entry_price
        if side == "long"
        else entry_price - reducer_price
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

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
    balance += (
        grid_pnl
        - grid_qty * market_fill_price * market.taker_fee
    )
    expected_volume += grid_qty * market_fill_price / balance
    reducer_pnl = reducer_qty * (
        market_fill_price - entry_price
        if side == "long"
        else entry_price - market_fill_price
    )
    balance += (
        reducer_pnl
        - reducer_qty * market_fill_price * market.taker_fee
    )
    expected_volume += reducer_qty * market_fill_price / balance

    assert output["balance"].item() == pytest.approx(balance, abs=3.0e-4)
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Exact calc_closes_* merges the below-executable-minimum WEL prefix with
    # the first same-tick recursive grid group before market sizing. The
    # completed group therefore executes once, after the initial entry.
    assert output["coin_fill_counts"][0, 0].item() == 2.0
    assert output["fill_count"].item() == 2.0
    assert output[size_key].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert charged[size_key].item() == pytest.approx(
        free[size_key].item(), abs=2.0e-4
    )
    # Only the near-touch WEL reducer is promoted; the farther recursive
    # groups stay passive.  The reducer sorts before the first group, so a
    # lower balance here specifically proves that early path used taker_fee.
    assert charged["balance"].item() < free["balance"].item() - 0.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_reducer_consumes_one_recursive_close_slot(side):
    runner, candidate, _ = _tm_multicoin_off_tick_reducer_case(
        side,
        maker_fee=0.0,
        close_qty_pct=0.0001,
        close_threshold_we=1.0e-6,
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    expected_size = 4.052 if side == "long" else 4.451
    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(expected_size, abs=2.0e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_global_position_exposure_repair(side):
    runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=10
    )
    repaired = list(baseline)
    repaired[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")
    ] = 1.0
    repaired[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
    ] = 0.5
    output = runner.run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert 0.0 < sizes[1] < sizes[0]
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("repair_kind", ["position", "total"])
def test_mps_tm_multicoin_market_exposure_repair_uses_slippage_and_taker_fee(
    side, repair_kind
):
    market_no_fee = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0, 0.0)
        for _ in range(2)
    ]
    market_with_taker_fee = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0, 0.002)
        for _ in range(2)
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
            "twel_enforcer_reduce_portfolio": float(
                repair_kind == "total"
            ),
        }
    )
    candidate = np.asarray(
        [
            [
                values[key]
                for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
            ]
        ],
        dtype=np.float64,
    )

    passive = passive_runner.run(candidate)
    promoted = promoted_runner.run(candidate)
    charged = fee_runner.run(candidate)
    torch.mps.synchronize()

    # The promoted protective reducer crosses immediately at an adverse
    # directional price; adding a nonzero taker fee then lowers balance again.
    assert promoted["loss_sum"].item() > passive["loss_sum"].item() + 1.0
    assert promoted["balance"].item() < passive["balance"].item() - 1.0
    assert charged["balance"].item() < promoted["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        [
            [
                values[key]
                for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
            ]
        ],
        dtype=np.float64,
    )

    baseline = baseline_runner.run(candidate)
    promoted = promoted_runner.run(candidate)
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # A trailing strategy does not otherwise populate the recursive close
    # snapshot. The promoted WEL reducer must retain its generation touch so
    # the next-candle allocator cannot interpret zero as an infinite minimum
    # and absorb the whole remaining position.
    assert baseline[size_key].item() > 0.0
    assert promoted[size_key].item() == pytest.approx(
        baseline[size_key].item(), abs=2.0e-4
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
        **fixture_kwargs,
    )
    if strategy_kind == "ema_anchor":
        runner = MpsEmaAnchorMulticoinFusedRunner(
            run,
            data,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=0.002,
            market_order_slippage_pct=0.01,
        )
        param_keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    else:
        runner = MpsTrailingMartingaleMulticoinFusedRunner(
            run,
            data,
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
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    assert output["alive"].cpu().tolist() == [True, True]
    assert output["psize"][0].item() > 0.0
    assert output["short_psize"][0].item() > 0.0
    assert output["psize"][1].item() < output["psize"][0].item()
    assert output["short_psize"][1].item() < output["short_psize"][0].item()
    assert output["fill_count"][1].item() > output["fill_count"][0].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("repair_kind", ["position", "total"])
def test_mps_tm_fused_one_way_multicoin_market_exposure_repair(repair_kind):
    _, baseline, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        count=10,
        return_context=True,
    )
    runner = MpsTrailingMartingaleMulticoinFusedRunner(
        run,
        data,
        hedge_mode=False,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
        market_order_slippage_pct=0.01,
    )
    baseline = list(baseline)
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_cooldown_minutes"
        )
    ] = 100.0
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_initial_ema_dist"
        )
    ] = 0.01
    repaired = list(baseline)
    if repair_kind == "position":
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "wel_enforcer_enabled"
            )
        ] = 1.0
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "wel_enforcer_threshold"
            )
        ] = 0.5
    else:
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "twel_entry_gate_enabled"
            )
        ] = 0.0
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "twel_enforcer_threshold"
            )
        ] = 0.5
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "twel_enforcer_enabled"
            )
        ] = 1.0
        repaired[
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
                "twel_enforcer_reduce_portfolio"
            )
        ] = 1.0

    output = runner.run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    baseline_size = (
        output["psize"][0].item() + output["short_psize"][0].item()
    )
    repaired_size = (
        output["psize"][1].item() + output["short_psize"][1].item()
    )
    assert (output["psize"][0].item() > 0.0) != (
        output["short_psize"][0].item() > 0.0
    )
    assert 0.0 < repaired_size < baseline_size
    assert output["fill_count"][1].item() > output["fill_count"][0].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_total_exposure_repair_policy(side):
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_entry_gate_enabled"
        )
    ] = 0.0
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_threshold"
        )
    ] = 0.5
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 1.0
    portfolio = list(overweight)
    portfolio[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_reduce_portfolio"
        )
    ] = 1.0

    output = runner.run(np.asarray([overweight, portfolio], dtype=np.float64))
    torch.mps.synchronize()

    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 1.0])
    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = (output[pprice_key] / output["balance"]).cpu()
    assert (total_twe < 0.5).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_loss_gate_blocks_lossy_total_exposure_repair(
    side, market_orders_allowed
):
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
        max_realized_loss_pct=0.1, **runner_kwargs
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() < gated[size_key].item()
    assert gated["open_positions"].item() == pytest.approx(2.0)
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_loss_gate_blocks_fee_only_ordinary_close(
    side, market_orders_allowed
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
        max_realized_loss_pct=0.1, **runner_kwargs
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() == pytest.approx(0.0)
    assert gated[size_key].item() > 0.0
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_preserves_profitable_close_after_repair(side):
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
        "max_realized_loss_pct": 0.1,
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    blocked = blocked_runner.run(np.asarray([candidate], dtype=np.float64))
    recovered = recovery_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert blocked[size_key].item() > 0.0
    assert recovered[size_key].item() == pytest.approx(0.0)
    assert recovered["balance"].item() > blocked["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_scans_past_blocked_recursive_rung(side):
    count = 10
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        highs[5:, 0] = 102.0
    else:
        lows[5:, 0] = 98.0
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=(100.0, 100.0),
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.1,
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The first immutable recursive rungs are below break-even, but later
    # rungs become profitable as their simulated exposure falls. Exact Rust
    # filters the early rungs independently and retains those later closes.
    assert output[size_key].item() < 5.0
    assert output["balance"].item() > 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
        for _ in range(2)
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
        max_realized_loss_pct=0.1,
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
    baseline = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    repaired_values = dict(values, twel_enforcer_enabled=1.0)
    repaired = [
        repaired_values[key]
        for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Coin zero wins TWEL's equal-adverse tie and alone reaches the target, but
    # its fee-only reducer is loss-gated. Exact Rust does not then reallocate
    # that action to profitable coin one.
    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 2.0])
    assert output[size_key][1].item() == pytest.approx(
        output[size_key][0].item(), abs=2.0e-4
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_total_exposure_repair_policy(side):
    overrides = np.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
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
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_entry_gate_enabled")
    ] = 0.0
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
    ] = 0.5
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
    ] = 1.0
    portfolio = list(overweight)
    portfolio[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_reduce_portfolio"
        )
    ] = 1.0

    output = runner.run(np.asarray([overweight, portfolio], dtype=np.float64))
    torch.mps.synchronize()

    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 1.0])
    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = (output[pprice_key] / output["balance"]).cpu()
    assert (total_twe < 0.5).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_loss_gate_blocks_lossy_total_exposure_repair(side):
    overrides = np.full(
        (2, EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32
    )
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
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() < gated[size_key].item()
    # Both paths keep the same symbol open, but the shared loss allowance
    # blocks part of the lossy repair and therefore retains more quantity.
    assert ungated["open_positions"].item() == pytest.approx(1.0)
    assert gated["open_positions"].item() == pytest.approx(1.0)
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_loss_gate_blocks_fee_only_ordinary_close(side):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
        for _ in range(2)
    ]
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
    candidate[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() == pytest.approx(0.0)
    assert gated[size_key].item() > 0.0
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_twel_repair_excludes_tailed_positions(side):
    count = 7
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 101.0
    hlcvs[:, :, 1] = 99.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    # Coin zero tails after this close. It remains part of total exposure but
    # is no longer a managed TWEL reducer candidate.
    hlcvs[3, 0, :3] = [111.0, 109.0, 110.0]
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, runs, [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
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
    repaired = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    baseline = list(repaired)
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        runs[1], data, side=side
    ).run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    if side == "long":
        # Only the tailed coin holds a long position, so no reachable repair
        # exists and the total size stays unchanged.
        assert sizes[1] == pytest.approx(sizes[0])
    else:
        # The surviving short position receives the reachable reducer.
        assert sizes[1] < sizes[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_twel_repair_excludes_tailed_positions(side):
    count = 70
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 101.0
    hlcvs[:, :, 1] = 99.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    # Fill the wide-offset entries, then tail coin zero while coin one remains
    # tradable. The tailed position stays in total exposure but cannot receive
    # a reducer.
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, runs, [market, market]
    )
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
    baseline[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
    ] = 0.0
    output = MpsEmaAnchorMulticoinRunner(
        runs[1], data, side=side
    ).run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[1] < sizes[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_qty_pct")
    ] = 0.2
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_threshold_base_pct"
        )
    ] = 0.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_retracement_base_pct"
        )
    ] = 0.001
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_cooldown_minutes"
        )
    ] = 100.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_entry_gate_enabled"
        )
    ] = 0.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_threshold"
        )
    ] = 0.8
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 1.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_reduce_portfolio"
        )
    ] = 1.0

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = output[pprice_key][0].item() / output["balance"][0].item()
    # Two ordinary trailing closes remove 0.2 TWE in total and TWEL repair
    # independently removes another 0.2.  Suppressing one trailing close
    # leaves approximately 0.7 instead of 0.6.
    assert total_twe < 0.65


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_reducer_retains_reachable_post_repair_grid(side):
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=6
    )
    enabled = list(row)
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")
    ] = 1.0
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
    ] = 0.5
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    reducer_only = list(enabled)
    reducer_only[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_retracement_base_pct"
        )
    ] = 0.001
    repaired_grid = list(enabled)
    repaired_grid[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_base_pct")
    ] = 0.0
    repaired_grid[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_we_weight")
    ] = 0.0
    output = runner.run(
        np.asarray([reducer_only, repaired_grid], dtype=np.float64)
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    volumes = output["day_volume"].sum(dim=1).cpu().numpy()
    assert sizes[0] > 0.0
    assert sizes[1] == pytest.approx(0.0)
    assert volumes[1] > volumes[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(5.0, abs=0.1)
    assert 1.0 < output["day_volume"].sum().item() < 2.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    exposure = output[size_key].item() * 100.0 / output["balance"].item()
    assert output[size_key].item() > 0.0
    assert exposure < target


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", "short", count=count
    )
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side="short", coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    # The reducer at the 102 bid fills, while Rust's 101 target grid order
    # remains below the next candle's 101.5 low and must not fill.
    assert output["short_psize"].item() == pytest.approx(4.999, abs=0.002)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.901
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_qty = entry_qty - grid_qty
    close_price = 100.0 if side == "long" else 101.0
    balance_after_entry = (
        1_000.0 - entry_qty * entry_price * market.maker_fee
    )
    expected_volume = entry_qty * entry_price / balance_after_entry
    grid_pnl = grid_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_grid = (
        balance_after_entry
        + grid_pnl
        - grid_qty * close_price * market.maker_fee
    )
    expected_volume += grid_qty * close_price / balance_after_grid
    reducer_pnl = reducer_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_close = (
        balance_after_grid
        + reducer_pnl
        - reducer_qty * close_price * market.maker_fee
    )
    expected_volume += reducer_qty * close_price / balance_after_close
    assert output["balance"].item() == pytest.approx(
        balance_after_close, abs=3.0e-4
    )
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
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
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], markets
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
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
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    twel_only_values = dict(values)
    twel_only_values["wel_enforcer_enabled"] = 0.0
    twel_only = [
        twel_only_values[key]
        for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate, twel_only], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    assert sizes[1] == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_mps_tm_multicoin_static_override_repairs_only_selected_symbol(
    side, market_orders_allowed
):
    overrides = np.full((2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS), np.nan, dtype=np.float32)
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
    baseline_output = baseline_runner.run(
        np.asarray([baseline], dtype=np.float64)
    )
    override_output = override_runner.run(
        np.asarray([overridden], dtype=np.float64)
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert (
        0.0
        < override_output[size_key].item()
        < baseline_output[size_key].item()
    )
    assert (
        override_output["day_volume"].sum().item()
        > baseline_output["day_volume"].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_min_effective_cost_filter_blocks_only_flat_entries(
    strategy_kind, side
):
    count = 8
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 103.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 98.0, 97.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 200.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    ema_row = [
        0.1,
        2.0,
        3.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    ema_row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    row = _tm_single_row() if strategy_kind == "trailing_martingale" else ema_row
    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hedge_mode": True,
    }

    promoted = runner_cls(market, run, data, **common).run(
        np.array([row + row], dtype=np.float64)
    )
    filtered_runner = runner_cls(
        market,
        run,
        data,
        filter_by_min_effective_cost=True,
        **common,
    )
    filtered = filtered_runner.run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert filtered_runner.settings.cpu()[12].item() == 1.0
    assert promoted["day_has_fill"].sum().item() > 0
    assert filtered["day_has_fill"].sum().item() == 0
    assert filtered["psize"].item() == 0.0
    assert filtered["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_min_effective_cost_uses_downward_projected_cost_bound():
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0e-9, 0.01, 1.0e-9, 10.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    base_qty_pct = 0.199999999
    row = [
        base_qty_pct,
        2.0,
        3.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    guaranteed_balance_lower = run.starting_balance * run.liquidation_threshold
    rounded_projection = np.float32(guaranteed_balance_lower) * np.float32(
        base_qty_pct
    )

    assert guaranteed_balance_lower * base_qty_pct < 10.0
    assert rounded_projection >= np.float32(data["max_effective_min_cost"])

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        filter_by_min_effective_cost=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() == 0
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_short_market_entry_respects_eligible_min_effective_cost_filter(
    strategy_kind,
):
    count = 5
    close = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 4.0, 1.0, 0.0)
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
    data = build_mps_data(close, close, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(initial_ema_dist=0.001)
        row[6] = 0.2
        runner_cls = MpsTrailingMartingaleRunner
        expected_psize = 1.998
    else:
        row = [
            0.1,
            2.0,
            3.0,
            0.0,
            0.0005,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            0.0,
            1.0,
        ]
        row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
        runner_cls = MpsEmaAnchorRunner
        expected_psize = 1.0

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=False,
        short_enabled=True,
        filter_by_min_effective_cost=True,
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.002,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["fill_count"].item() == 1.0
    assert output["short_psize"].item() == pytest.approx(expected_psize)
    assert output["short_pprice"].item() == pytest.approx(100.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_mps_dual_single_coin_min_cost_rejection_blocks_other_side(
    strategy_kind, hedge_mode
):
    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 97.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 4.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        long_row = _tm_single_row(initial_ema_dist=0.02)
        short_row = _tm_single_row(initial_ema_dist=0.01)
        short_row[6] = 0.01
        runner_cls = MpsTrailingMartingaleRunner
    else:
        long_row = [
            0.1,
            2.0,
            3.0,
            0.0,
            0.02,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            0.0,
            1.0,
        ]
        long_row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
        short_row = list(long_row)
        short_row[0] = 0.01
        short_row[4] = 0.01
        runner_cls = MpsEmaAnchorRunner

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=hedge_mode,
        filter_by_min_effective_cost=True,
    ).run(np.array([long_row + short_row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() == 0
    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_min_effective_cost_filter_keeps_managing_an_open_position(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
        high[4] = 102.0
    else:
        high[3] = 102.0
        low[4] = 98.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    # Leave enough headroom above the liquidation-floor projection to open the
    # position; this test targets management after opening, not equality.
    market = ProxyMarket(0.001, 0.01, 0.001, 4.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [
        0.1,
        2.0,
        3.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hedge_mode=True,
        filter_by_min_effective_cost=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert output["gap_hist"].sum().item() >= 1
    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() == 0.0
    gap_metrics = _fill_gap_metrics(
        {
            key: output[key].cpu()
            for key in (
                "gap_hist",
                "first_fill_ts",
                "last_fill_ts",
                "first_eq_ts",
                "last_eq_ts",
            )
        },
        run,
    )
    assert all(
        torch.isfinite(value).all().item()
        for value in gap_metrics.values()
    )
    assert gap_metrics["fills_gap_p99_hours"].item() >= gap_metrics[
        "fills_gap_p95_hours"
    ].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_trailing_martingale_multicoin_sizes_raw_touch_close_before_price_finalization(
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
            0.0,  # volume drop
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

    output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, side=side
    ).run(np.array([row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust sizes against the raw 100.004 touch before finalizing the
    # executable close to 100.00. The raw minimum is 0.100; recomputing it
    # after finalization would yield 0.101 and leave the wrong remainder.
    position_key = "psize" if side == "long" else "short_psize"
    assert output[position_key].item() == pytest.approx(0.9, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
def test_mps_trailing_martingale_shader_contract_and_directional_smoke(
    long_enabled, short_enabled
):
    import passivbot_rust
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    source = passivbot_rust.mps_trailing_martingale_source_py()
    assert "kernel void passivbot_trailing_martingale" in source
    assert "constant int SIDE_PARAMS = 52" in source
    assert "s.allowed_wel" in source
    assert "s.entry_cap" in source
    assert "min_since_open" in source
    assert "max_since_min" in source
    assert "max_since_open" in source
    assert "min_since_max" in source
    assert "s.entry_retracement_base > 0.0f" in source
    assert "s.close_retracement_base > 0.0f" in source
    assert "entry_gen_balance" in source
    assert "close_gen_balance" in source
    assert "merge_reducer" not in source
    assert "finalized_reducer_qty" in source
    assert "finalized_reducer_qty_with_ordinary" in source
    assert "reducer_candidate_preferred" in source
    assert "strict distance to the executable" in source
    assert "for (int rung = 0; rung < 500; ++rung)" in source
    assert "ladder_side, ladder_balance" in source
    assert "recursive_close_groups" in source
    assert "recursive_strategy_close_would_expand" in source
    assert "selected.market = should_use_ordinary_market_execution" in source
    assert source.count("bool all_below_min") == 1
    assert source.count("bool normalize_close_groups") == 3
    assert source.count("int collapse_ordinary_rank") == 3
    assert source.count("recursive_close_allocation(") == 7
    assert source.count("select_recursive_close_reducer(") == 3
    assert source.count("#if PASSIVBOT_TM_REDUCERS_DISABLED") == 2
    assert source.count("ReducerCandidate candidates[3]") == 2
    assert source.count("candidate_allocations[candidate_idx]") == 4
    assert source.count("candidates[candidate_idx].finalized_qty") == 4
    assert source.count("close_gen_psize - strategy_wel_qty") == 2
    assert "strategy_wel_qty = reducer_qty" not in source
    assert source.count("group.market ?") >= 5
    assert source.count("float reducer_fee_rate = reducer_market") == 4
    assert "realized_loss_proxy_allows_close" in source
    assert "const float max_realized_loss_pct = settings[14]" in source
    assert "const bool loss_gate_enabled = !PASSIVBOT_TM_LOSS_GATE_DISABLED" in source
    assert "const float taker_fee = settings[15]" in source
    assert "const float market_order_slippage_pct = fmax(settings[16], 0.0f)" in source
    assert "const bool long_hsl_panic_market = settings[17] > 0.5f" in source
    assert "const bool short_hsl_panic_market = settings[18] > 0.5f" in source
    assert "market_execution ? taker_fee : maker_fee" in source
    assert (
        "const bool market_orders_allowed = !PASSIVBOT_TM_MARKET_ORDERS_DISABLED"
        in source
    )
    assert "#ifndef PASSIVBOT_TM_TRAILING_ENTRY_ONLY" in source
    assert "#ifndef PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY" in source
    assert "#ifndef PASSIVBOT_TM_TRAILING_CLOSE_ONLY" in source
    assert "#ifndef PASSIVBOT_TM_REDUCERS_DISABLED" in source
    assert (
        "const float market_order_near_touch_threshold = fmax(settings[20], 0.0f)"
        in source
    )
    assert "should_use_ordinary_market_execution(" in source
    assert "ordinary_market_fill_price(" in source
    assert "resize_market_close_qty(" in source
    assert (
        "float reducer_exec_price = reducer_market ? price_now : reducer_price"
        in source
    )
    assert "s.secondary_close_market = s.close_market" in source
    assert "s.close_market = reducer_market" in source
    assert "entry_gen_market_price" in source
    assert "close_gen_market_price" in source
    assert "preferred.finalized_qty, gate_price" in source
    assert "source.close_gen_market_price, !is_long" in source
    assert "close_gen_realized_pnl_cumsum_last" in source
    assert "close_gen_realized_pnl_cumsum_max" in source
    assert source.count("for (int allocation_pass = 0;") == 1
    assert "reducer_below_min && !all_below_min" in source
    assert "include_reducer = false" in source
    assert "allocation.normalize_close_groups = allocation_pass > 0" in source
    assert "reducer_qty, reducer_fill_price" not in source
    assert "adj, reducer_fill_price" not in source
    assert "sim.market_orders_allowed = false" in source
    assert source.count("ladder_side.market_orders_allowed = false") == 2
    assert source.count("ladder_side.twel_entry_gate_enabled = false") == 2
    assert source.count("ladder_side.psize + strategy_eq") == 2
    assert "bool entry_market = rung == 0" not in source
    assert "entry_ticks, true, ladder_market_price" in source
    assert "entry_ticks, false, ladder_market_price" in source
    assert source.count("entry_strategy_qty") == 5
    assert source.count("if (!(eq > 0.0f)) break;") == 2
    assert source.count("if (twel_boundary_partial) break;") == 2
    assert source.count("entry_passive_reachable") >= 6
    assert "gate_market_entry_by_twel_strict" not in source
    assert source.count("gate_entry_by_twel_strict(") == 4
    assert "s.entry_retracement_base > 0.0f" in source
    assert "#if PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY" in source
    assert "s.close_retracement_base > 0.0f" in source
    assert "the proxy uses a zero-loss envelope" in source
    assert "const bool filter_by_min_effective_cost" in source
    assert "passes_min_effective_cost" in source
    assert "projected_cost_lower" in source
    assert "float guaranteed_balance_lower" in source
    assert "accumulate_min_cost_balance_error" not in source
    assert "cooldown_min != 0.0f" in source
    assert "int entry_touch = is_long ? touch_down_ticks : touch_up_ticks" in source
    assert "int raw_reentry_ticks = reentry_target_is_touch" in source
    assert "int cticks = touch_controls ? touch_nearest_ticks : target_ticks" in source
    assert "entry_raw_touch" not in source
    assert "close_raw_touch" not in source
    assert "nextafter(" not in source

    count = 8
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 103.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 98.0, 97.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row()

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=long_enabled,
        short_enabled=short_enabled,
        hedge_mode=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert (output["psize"].item() > 0.0) is long_enabled
    assert (output["short_psize"].item() > 0.0) is short_enabled
    _assert_fill_scalar_contract(output)
    if long_enabled and not short_enabled:
        assert torch.equal(output["fill_count_long"], output["fill_count"])
    if short_enabled and not long_enabled:
        assert output["fill_count_long"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("is_long", [True, False])
def test_mps_trailing_martingale_fills_recursive_entry_ladder(is_long):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if is_long:
        low[2] = 99.99  # Fill the initial entry generated on the prior bar.
        low[3] = 90.0  # Strictly cross several pre-generated recursive rungs.
    else:
        high[2] = 100.01
        high[3] = 110.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[4] = 1.5  # entry double-down factor
    row[6] = 0.05  # initial entry uses 5% of the exposure budget
    row[11] = 0.0  # recursive entry mode
    row[16] = 1.0  # keep trailing closes unreachable in this fixture

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=is_long,
        short_enabled=not is_long,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    position = output["psize" if is_long else "short_psize"].item()
    # Initial qty is 0.5 and the first reentry is 0.75. Crossing more than
    # those two orders proves the recursive ladder, not merely one reentry,
    # was filled in the same candle.
    assert position > 1.25


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_fills_recursive_entry_ladders_in_hedge_mode():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[2], high[2] = 99.99, 100.01
    low[3], high[3] = 90.0, 110.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[4], row[6], row[11], row[16] = 1.5, 0.05, 0.0, 1.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 1.25
    assert output["short_psize"].item() > 1.25


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("is_long", [True, False])
@pytest.mark.parametrize("close_threshold_we", [-0.005, 0.005])
def test_mps_trailing_martingale_fills_sorted_recursive_close_grid(
    is_long, close_threshold_we
):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if is_long:
        low[3] = 99.0  # Fill the initial entry generated on the prior bar.
        high[4] = 100.85 if close_threshold_we > 0.0 else 100.6
    else:
        high[3] = 101.0
        low[4] = 99.15 if close_threshold_we > 0.0 else 99.4
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[6] = 0.1  # Open a 1-unit position.
    row[7] = 0.01  # Exposure cap prevents reentries in this fixture.
    row[15] = 0.05  # Two recursive close rungs at the standard lower bound.
    row[16] = 0.005 if close_threshold_we > 0.0 else 0.01
    row[17] = close_threshold_we * 10.0
    row[20] = 0.0  # Recursive close mode.

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=is_long,
        short_enabled=not is_long,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    position = output["psize" if is_long else "short_psize"].item()
    # The candle strictly crosses one of two sorted 0.5-unit closes. With
    # positive WE weight, the generated grid must be reversed before filling;
    # its first generated (farthest) order is deliberately not crossed.
    assert position == pytest.approx(0.5, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_fills_recursive_close_grids_in_hedge_mode():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3], high[3] = 99.0, 101.0
    low[4], high[4] = 99.15, 100.85
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[6], row[7], row[15] = 1.0, 1.0, 0.2
    row[16], row[17], row[20] = 0.005, 0.005, 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == pytest.approx(4.0, abs=1e-6)
    assert output["short_psize"].item() == pytest.approx(4.0, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_preserves_tick_aligned_computed_target():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 99.99, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_quantizes_non_aligned_entry_down():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.006)
    high = np.full(count, 100.006)
    low = np.array([100.006, 100.006, 100.006, 100.005, 100.006])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust selects the raw 100.006 bid, then finalize_next_entry floors
    # the executable long order to 100.00. The 100.005 low must not fill it.
    assert not output["day_has_fill"].any().item()
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_quantizes_non_aligned_short_entry_up():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.006, dtype=np.float64)
    high = np.array([100.006, 100.006, 100.006, 100.007, 100.006])
    low = close.copy()
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=False, short_enabled=True
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # finalize_next_entry ceils the selected 100.006 short touch to 100.01.
    # A 100.007 high remains below the executable order and must not fill it.
    assert not output["day_has_fill"].any().item()
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_entry_quantizes_before_strict_fill():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.000003, dtype=np.float64)
    high = close.copy()
    low = np.array(
        [100.000003, 100.000003, 100.000003, 100.000001, 100.000003],
        dtype=np.float64,
    )
    assert np.float32(close[2]) == np.float32(low[3])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust first chooses the raw touch, then floors the executable long
    # entry to 100.00. A low above 100.00 must not fill it.
    assert not output["day_has_fill"].any().item()
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_initial_gate_chooses_tick_before_float32_collapse():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.000003, dtype=np.float64)
    high = close.copy()
    low = np.array(
        [100.000003, 100.000003, 100.000003, 100.000001, 100.000003],
        dtype=np.float64,
    )
    assert np.float32(close[2]) == np.float32(low[3])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=1.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust's min(raw bid, EMA target) chooses the 100.00 tick. The
    # following 100.000001 low must not fill it. A float32 pre-comparison
    # incorrectly retained the raw 100.000003 bid and filled.
    assert not output["day_has_fill"].any().item()
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_quantizes_selected_raw_close_to_nearest_tick():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 100.000003, dtype=np.float64)
    high = np.array(
        [100.000003, 100.000003, 100.000003, 100.000003, 100.000002, 100.000003],
        dtype=np.float64,
    )
    low = np.array(
        [100.000003, 100.000003, 100.000003, 100.000001, 100.000003, 100.000003],
        dtype=np.float64,
    )
    assert np.float32(close[3]) == np.float32(high[4])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[16] = 0.0  # tick-aligned close target at the float32 position price
    row[20] = 0.0  # disable trailing-close touch override

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust chooses the raw 100.000003 ask over the 100.00 target, then
    # calc_closes_long quantizes it to the nearest tick, 100.00. The following
    # 100.000002 high must therefore close it.
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_sizes_raw_touch_close_before_price_finalization():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 100.004, dtype=np.float64)
    high = np.array([100.004, 100.004, 100.004, 100.004, 101.0, 100.004])
    low = np.array([100.004, 100.004, 100.004, 99.0, 100.004, 100.004])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[15] = 0.01
    row[16] = 0.199
    row[17] = -2.0  # keep later grid closes above this fixture's high
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # The raw 100.004 ask implies a 0.100 minimum close. Only after sizing does
    # Rust finalize the executable close to 100.00, whose minimum would be
    # 0.101 if it were incorrectly used for sizing. The 1.000 position must
    # therefore retain 0.900 after the close.
    assert output["psize"].item() == pytest.approx(0.9, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_preserves_just_above_aligned_raw_touch_minimum():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 0.088, dtype=np.float64)
    high = np.array([0.088, 0.088, 0.088, 0.088, 0.1, 0.088])
    low = np.array([0.088, 0.088, 0.088, 0.07, 0.088, 0.088])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.0, 11.0, 1.0, 0.0)
    run = ProxyRun(
        200.0,
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[15] = 0.05
    row[16] = 0.299
    row[17] = -3.0  # keep a possible second grid close above the high
    row[20] = 0.0

    rounded_only_data = dict(data)
    rounded_only_data["touch_min_qty_relation"] = torch.zeros_like(
        data["touch_min_qty_relation"]
    )
    rounded_only = MpsTrailingMartingaleRunner(
        market,
        run,
        rounded_only_data,
        long_enabled=True,
        short_enabled=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()
    assert rounded_only["psize"].item() == 125.0

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Rust's raw-touch minimum is 125.00000000000001. A nominal 125-unit
    # remainder is therefore below the effective minimum, so the 250-unit
    # position must be closed in full rather than leaving half open.
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_quantizes_selected_short_close_to_nearest_tick():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 100.004, dtype=np.float64)
    high = np.array([100.004, 100.004, 100.004, 101.0, 100.004, 100.004])
    low = np.array([100.004, 100.004, 100.004, 100.004, 100.001, 100.004])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[16] = 0.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=False, short_enabled=True
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # The short opens at the ceiled 100.01 entry. Rust then selects the raw
    # 100.004 bid for its close and rounds it to 100.00. A 100.001 low is not
    # strictly below that executable close, so the position must remain open.
    assert output["short_psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_one_way_arbitrates_initial_entry():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row()

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_entry_cap_uses_rust_nearest_step_rounding():
    """Guard the one-quantity divergence that can split a later trailing path."""

    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.0,
        gate_initial=0.0,
        gate_reentry=0.0,
        entry_gate=False,
    )
    row[6] = 2.0  # Oversize the initial order so the exposure cap crops it.
    row[7] = 0.5  # Keep any subsequent reentry far below the fixture market.
    row[16] = 0.5  # Keep the close above the fixture market.
    row[20] = 0.0
    row[24] = 0.24036  # Exact cap quantity is 2.4036 at price 100.

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Rust finalization rounds the cap to the nearest quantity step. Flooring
    # this to 2.403 was enough to change a later close/reentry decision.
    assert output["psize"].item() == pytest.approx(2.404, abs=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_touch_close_preserves_raw_price():
    """Match Rust finalization when an off-tick market touch controls a close."""

    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 9
    close = np.array(
        [
            100.0,
            100.0,
            100.004,
            100.004,
            100.004,
            100.004,
            100.004,
            100.004,
            100.004,
        ]
    )
    high = np.array(
        [
            100.0,
            100.0,
            100.004,
            100.005,
            100.005,
            100.005,
            100.005,
            100.005,
            100.004,
        ]
    )
    low = np.array(
        [
            100.0,
            100.0,
            99.0,
            100.003,
            100.003,
            100.003,
            100.004,
            100.004,
            100.004,
        ]
    )
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
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
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[7] = 0.5  # Prevent another entry in this fixture.
    row[16] = 0.0  # Use the current touch after the close trail retraces.
    row[20] = 0.000001

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust keeps the raw 100.004 ask, which fills at the next 100.005
    # high. Rounding the touch up to 100.01 would leave the position open.
    assert output["psize"].item() == 0.0
