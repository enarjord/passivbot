from config.universe import set_side_universe_sources
import sys
from functools import partial
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from config.shared_bot import flatten_shared_bot_side
from config.schema import get_template_config
from optimization.gpu.model import (
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_PARAM_KEYS,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_PARAM_KEYS,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
    validate_hsl_signal_topology,
)
from optimization.gpu.coin_parameters import build_coin_override_parameters

from optimization.gpu.service import (
    CORE_OUTPUT_KEYS,
    DIRECTIONAL_HSL_OUTPUT_KEYS,
    MPS_MAX_DISPATCH_CANDIDATE_BARS,
    SharedAccountReplay,
    _btc_daily_price_context,
    _candidate_wallet_exposure_limit_outputs,
    _candidate_position_slot_outputs,
    _hsl_lookback_bars,
    _directional_entry_initial_metrics,
    _directional_gross_pnl_outputs,
    _gpu_proxy_execution_checkpoint_contract,
    _gpu_profile_unattributed_seconds,
    _add_gpu_terminal_profile,
    _hsl_params,
    _hsl_diagnostics_needed,
    _mps_dispatch_batch_size,
    _mps_multicoin_dispatch_plan,
    _mps_strategy_eq_recovery_distribution,
    _new_gpu_dispatch_progress,
    _new_gpu_proxy_profile,
    _update_gpu_dispatch_progress,
    _add_gpu_runner_profile,
    _multicoin_exposure_eligible_coins,
    _position_exposure_enforcer_params,
    _require_exact_safe_proxy_candles,
    _require_no_internal_invalid_hsl_candles,
    _require_no_internal_invalid_multicoin_hsl_candles,
    _require_supported_multicoin_valid_tails,
    _candle_interval_minutes,
    _single_coin_exposure_params,
    _total_exposure_enforcer_params,
    _unstuck_params,
)


build_ema_coin_parameters = partial(build_coin_override_parameters, strategy_kind="ema_anchor")
build_tm_coin_parameters = partial(build_coin_override_parameters, strategy_kind="trailing_martingale")


def _complete_hsl_payload(payload):
    from copy import deepcopy

    defaults = get_template_config()["bot"]["long"]["hsl"]
    params = getattr(payload, "backtest_params", {})
    payload.backtest_params = params
    params.setdefault("coins", ["BTC", "ETH"][: len(payload.bot_params_list)])
    params.setdefault("dynamic_wel_by_tradability", True)
    params.setdefault("unstuck_memory_ema_span_minutes", 1440.0)
    params.setdefault("loss_gate_memory_ema_span_minutes", 1440.0)
    old = params.get("equity_hard_stop_loss", {})
    if "mode" in old:
        return
    mode = old.get("signal_mode", "coin")
    policies = {}
    for coin, sides in zip(params["coins"], payload.bot_params_list):
        pair = []
        for side in ("long", "short"):
            bot = sides.get(side, {})
            policy = deepcopy(defaults)
            policy.update(
                {key: bot[f"hsl_{key}"] for key in defaults if f"hsl_{key}" in bot}
            )
            bot.setdefault("n_positions", 1)
            pair.append(policy)
        policies[coin] = pair
    params["equity_hard_stop_loss"] = {
        "engine": "hsl",
        "mode": mode,
        "coins": policies,
        "sides": next(iter(policies.values())),
        "portfolio": deepcopy(defaults),
    }


@pytest.mark.parametrize("proxy_cls", [SharedAccountReplay])
def test_proxy_constructors_reject_exact_only_metrics_before_setup(
    monkeypatch, proxy_cls
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    with pytest.raises(ValueError, match="exact Rust backtests and analysis"):
        proxy_cls(
            config=get_template_config(),
            hlcvs=None,
            mss={},
            btc=None,
            timestamps=None,
            exchange="bybit",
            batch_size=1,
            needed_metrics={"fills_count"},
        )


@pytest.mark.parametrize("raw_interval", [5, 5.0, "5"])
def test_single_coin_candle_interval_accepts_positive_integers(raw_interval):
    assert (
        _candle_interval_minutes({"candle_interval_minutes": raw_interval})
        == 5
    )


@pytest.mark.parametrize("raw_interval", [0, -1, 1.5, float("nan"), None, "bad"])
def test_single_coin_candle_interval_rejects_invalid_values(raw_interval):
    with pytest.raises(ValueError, match="integer >= 1"):
        _candle_interval_minutes({"candle_interval_minutes": raw_interval})


def test_gpu_proxy_execution_checkpoint_contract_tracks_effective_inputs():
    backtest_params = {
        "coins": ["BTC"],
        "starting_balance": 10_000.0,
        "candle_interval_minutes": 1,
        "requested_start_timestamp_ms": 1_000,
        "first_timestamp_ms": 1_000,
        "first_valid_indices": [0],
        "last_valid_indices": [2],
        "trade_start_indices": [1],
        "global_warmup_bars": 1,
        "liquidation_threshold": 0.05,
        "filter_by_min_effective_cost": False,
        "dynamic_wel_by_tradability": True,
        "hedge_mode": False,
        "max_realized_loss_pct": 1.0,
        "unstuck_memory_ema_span_minutes": 43200.0,
        "loss_gate_memory_ema_span_minutes": 43200.0,
        "market_order_slippage_pct": 0.0005,
        "market_orders_allowed": False,
        "market_order_near_touch_threshold": 0.001,
        "forager_score_hysteresis_pct": 0.02,
    }
    market = {
        "qty_step": 0.001,
        "price_step": 0.1,
        "min_qty": 0.001,
        "min_cost": 5.0,
        "c_mult": 1.0,
        "maker_fee": 0.0004,
        "taker_fee": 0.00055,
    }
    original = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
    )
    changed_fee = dict(market, maker_fee=0.0005)
    changed = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[changed_fee],
        base_params={"long": {"offset": 0.01}},
    )
    changed_timestamps = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_100, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
    )
    changed_hlcvs_values = np.arange(12, dtype=np.float64).reshape(3, 1, 4)
    changed_hlcvs_values[1, 0, 2] += 0.5
    changed_hlcvs = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=changed_hlcvs_values,
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
    )
    changed_base_params = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.02}},
    )
    btc_prices = np.array([50_000.0, 51_000.0, 49_000.0])
    with_btc = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
        btc_prices=btc_prices,
    )
    changed_btc = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
        btc_prices=btc_prices + np.array([0.0, 1.0, 0.0]),
    )

    assert changed != original
    assert changed_timestamps != original
    assert changed_hlcvs != original
    assert changed_base_params != original
    assert "directional_hsl_rolling_capacity" not in original
    assert "btc_analysis" not in original
    assert changed_btc != with_btc
    assert original["timestamps"]["count"] == 3


def test_btc_daily_price_context_matches_proxy_utc_day_grid():
    day_ms = 86_400_000
    context = _btc_daily_price_context(
        [10.0, 12.0, 11.0, 20.0],
        [0, day_ms - 1, day_ms, 2 * day_ms - 1],
        expected_count=4,
        expected_days=2,
    )

    assert context["btc_day_end_price"].tolist() == [12.0, 20.0]


@pytest.mark.parametrize(
    "btc_prices",
    ([10.0, float("nan")], [10.0, 0.0], [10.0]),
)
def test_btc_daily_price_context_rejects_unsafe_series(btc_prices):
    with pytest.raises(ValueError, match="MPS BTC analysis"):
        _btc_daily_price_context(
            btc_prices,
            [0, 60_000],
            expected_count=2,
            expected_days=1,
        )


def test_btc_daily_price_context_rejects_skipped_utc_days():
    day_ms = 86_400_000
    with pytest.raises(ValueError, match="missing prepared prices"):
        _btc_daily_price_context(
            [10.0, 20.0],
            [0, 2 * day_ms],
            expected_count=2,
            expected_days=3,
        )


def test_gpu_proxy_execution_checkpoint_contract_rejects_timestamp_shape_mismatch():
    with pytest.raises(ValueError, match="timestamp identity disagrees"):
        _gpu_proxy_execution_checkpoint_contract(
            strategy_kind="ema_anchor",
            exchange="bybit",
            enabled_sides=["long"],
            hlcvs=np.zeros((3, 1, 4), dtype=np.float64),
            timestamps=np.array([1_000, 2_000], dtype=np.int64),
            backtest_params={"coins": ["BTC"]},
            exchange_params=[],
            base_params={},
        )


@pytest.mark.parametrize("enabled,expected", [(True, 1), (False, 0)])
def test_forward_hsl_keeps_only_an_inert_binding(enabled, expected):
    assert (
        _hsl_lookback_bars({}, hsl_enabled=enabled)
        == expected
    )




@pytest.mark.parametrize("malformed", [None, "dtype", "device", "columns", "rows", "bounds", "ambiguous", "non_tensor"])
def test_compact_recovery_output_remains_on_device_and_rejects_bad_transport(monkeypatch, malformed):
    class Device:
        type = "cuda"
    class Tensor:
        device = Device()
        dtype = "float32"
        shape = (2, 7)
        ndim = 2
        def cpu(self):
            pytest.fail("compact recovery transport must not copy to CPU")
    fake = ModuleType("optimization.gpu.mps_kernel")
    fake.torch = SimpleNamespace(Tensor=Tensor, float32="float32")
    fake.MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS = 7
    fake.strategy_eq_recovery_distribution_from_samples = lambda *a, **k: pytest.fail("compact recovery must not reduce twice")
    monkeypatch.setitem(sys.modules, "optimization.gpu.mps_kernel", fake)
    stats, bounds = Tensor(), Tensor()
    bounds.shape = (2,)
    output = {"strategy_eq_recovery_distribution": stats, "last_eq_ts": bounds}
    if malformed == "dtype": stats.dtype = "float64"
    elif malformed == "device": stats.device = SimpleNamespace(type="cpu")
    elif malformed == "columns": stats.shape = (2, 6)
    elif malformed == "rows": stats.shape = (1, 7)
    elif malformed == "bounds": output.pop("last_eq_ts")
    elif malformed == "ambiguous": output["strategy_eq_recovery_samples"] = object()
    elif malformed == "non_tensor": output["strategy_eq_recovery_distribution"] = object()
    if malformed:
        with pytest.raises(RuntimeError, match="(Invalid compact GPU|Ambiguous GPU) recovery output"):
            _mps_strategy_eq_recovery_distribution(output, {"strategy_eq_recovery_days_mean"})
    else:
        assert _mps_strategy_eq_recovery_distribution(output, {"strategy_eq_recovery_days_mean"}) is stats
        assert _mps_strategy_eq_recovery_distribution(output, {"adg_strategy_eq"}) is None


def test_recovery_distribution_postprocessor_is_opt_in_and_fail_closed(monkeypatch):
    needed = {"strategy_eq_recovery_days_p99"}

    assert _mps_strategy_eq_recovery_distribution({}, {"adg_strategy_eq"}) is None
    with pytest.raises(RuntimeError, match="recovery sampling output is missing"):
        _mps_strategy_eq_recovery_distribution({}, needed)

    samples = object()
    expected = SimpleNamespace()
    expected.cpu = lambda: (_ for _ in ()).throw(
        AssertionError("recovery distribution left MPS during reduction")
    )
    called = {}

    def fake_postprocessor(values, *, sample_interval_days):
        called["values"] = values
        called["sample_interval_days"] = sample_interval_days
        return expected

    fake_mps_kernel = ModuleType("optimization.gpu.mps_kernel")
    fake_mps_kernel.strategy_eq_recovery_distribution_from_samples = fake_postprocessor
    monkeypatch.setitem(sys.modules, "optimization.gpu.mps_kernel", fake_mps_kernel)
    actual = _mps_strategy_eq_recovery_distribution(
        {
            "strategy_eq_recovery_samples": samples,
            "strategy_eq_recovery_sample_interval_days": 1.0 / 24.0,
        },
        needed,
    )

    assert called == {
        "values": samples,
        "sample_interval_days": 1.0 / 24.0,
    }
    assert actual is expected




@pytest.mark.parametrize(
    "strategy,batch,bars,sides,cap,expected",
    [
        ("trailing_martingale", 4096, 900000, 1, 1_000_000_000, (True, 512, 4096)),
        ("trailing_martingale", 4096, 60000, 1, 1_000_000_000, (False, 574, 60000)),
        ("trailing_martingale", 4096, 4320, 1, 1_000_000_000, (False, 4096, 4320)),
        ("trailing_martingale", 16, 900000, 1, 1_000_000_000, (False, 16, 900000)),
        ("trailing_martingale", 4096, 900000, 2, 1_000_000_000, (False, 19, 900000)),
        ("ema_anchor", 4096, 900000, 1, 1_000_000_000, (False, 38, 900000)),
        ("trailing_martingale", 4096, 900000, 1, 100, (True, 1, 3)),
    ],
)
@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_mps_multicoin_temporal_plan_preserves_work_limit(
    strategy, batch, bars, sides, cap, expected, device
):
    expected = ((True, 1, 3) if cap == 100 else
                (True, min(batch, 1024 if device == "cuda" else 512), 128))
    result = _mps_multicoin_dispatch_plan(
        strategy,
        batch,
        n_bars=bars,
        n_coins=29,
        n_sides=sides,
        max_candidate_bars=cap,
        device=device,
    )
    assert result == expected
    _, candidate_batch, history_chunk = result
    assert candidate_batch * history_chunk * 29 * sides <= cap


@pytest.mark.parametrize("sides", [1, 2])
@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("bars,expected", [(8192, (True, 128)), (8193, (True, 128))])
def test_cuda_tm_long_history_has_interrupt_boundaries_below_work_cap(sides, batch, bars, expected):
    temporal, candidates, history = _mps_multicoin_dispatch_plan(
        "trailing_martingale", batch, n_bars=bars, n_coins=16,
        n_sides=sides, max_candidate_bars=500_000_000, device="cuda",
    )
    assert (temporal, history) == expected
    assert candidates == batch
    assert candidates * history * 16 * sides <= 500_000_000


@pytest.mark.parametrize("batch", [1, 17, 512, 513, 1024, 1025, 4096])
@pytest.mark.parametrize("device,limit", [("mps", 512), ("cuda", 1024)])
def test_multicoin_temporal_plan_respects_requested_batch_and_state_cap(
    batch, device, limit
):
    temporal, candidates, history = _mps_multicoin_dispatch_plan(
        "trailing_martingale",
        batch,
        n_bars=2_500_000,
        n_coins=28,
        n_sides=1,
        max_candidate_bars=1_000_000_000,
        device=device,
    )
    assert candidates <= min(batch, limit)
    if temporal:
        assert candidates * history <= 2_097_152
    assert candidates * history * 28 <= 1_000_000_000


def test_mps_dispatch_batch_size_bounds_single_and_multicoin_work():
    n_bars = 1_923_175

    assert _mps_dispatch_batch_size(8192, n_bars=n_bars) == 519
    assert _mps_dispatch_batch_size(8192, n_bars=n_bars, n_coins=4) == 129
    assert _mps_dispatch_batch_size(8192, n_bars=n_bars, n_coins=4, n_sides=2) == 64
    assert (
        _mps_dispatch_batch_size(
            8192,
            n_bars=n_bars,
            max_candidate_bars=500_000_000,
        )
        == 259
    )
    assert _mps_dispatch_batch_size(32, n_bars=1000, n_coins=4) == 32
    with pytest.raises(ValueError, match="one GPU candidate exceeds"):
        _mps_dispatch_batch_size(8192, n_bars=3 * 10**9)
    with pytest.raises(ValueError, match="max_candidate_bars"):
        _mps_dispatch_batch_size(32, n_bars=1000, max_candidate_bars=0)
    assert MPS_MAX_DISPATCH_CANDIDATE_BARS == 1_000_000_000








def test_gpu_terminal_profile_rebases_recent_window_steps():
    torch = pytest.importorskip("torch")
    profile = {
        "terminal_candidate_count": 0,
        "terminal_without_equity_count": 0,
        "estimated_post_terminal_candidate_bars": 0,
        "_terminal_step_fractions": [],
        "side_count": 1,
    }

    _add_gpu_terminal_profile(
        profile,
        {
            "alive": torch.as_tensor([False]),
            "last_eq_ts": torch.as_tensor([75 * 60_000.0]),
        },
        interval_ms=60_000,
        effective_start_step=50,
        effective_end_step=100,
    )

    assert profile["terminal_candidate_count"] == 1
    assert profile["estimated_post_terminal_candidate_bars"] == 23
    assert profile["_terminal_step_fractions"] == [pytest.approx(25.0 / 48.0)]


def test_gpu_dispatch_progress_is_rate_limited_and_reports_eta(
    monkeypatch, caplog
):
    readings = iter((100.0, 120.0, 161.0, 222.0))
    monkeypatch.setattr(
        "optimization.gpu.service.time.monotonic", lambda: next(readings)
    )
    progress = _new_gpu_dispatch_progress(8, 2)

    _update_gpu_dispatch_progress(
        progress, completed_candidates=2, strategy="trailing_martingale"
    )
    assert not caplog.records

    with caplog.at_level("INFO"):
        _update_gpu_dispatch_progress(
            progress, completed_candidates=4, strategy="trailing_martingale"
        )
        _update_gpu_dispatch_progress(
            progress, completed_candidates=8, strategy="trailing_martingale"
        )

    assert "batches_done=2/4" in caplog.records[0].message
    assert "scenario_evals=4/8" in caplog.records[0].message
    assert "eta_group=" in caplog.records[0].message
    assert "batches_done=4/4" in caplog.records[1].message




def test_gpu_profile_candidate_bars_include_coin_and_side_topology():
    proxy = SimpleNamespace(
        batch_size=4,
        dispatch_batch_size=4,
        strategy_kind="ema_anchor",
    )
    runner = SimpleNamespace(
        n=100,
        n_coins=8,
        last_profile={
            "batch_size": 1,
            "dispatch_count": 1,
            "cold": False,
        },
    )
    profile = _new_gpu_proxy_profile(proxy, [{}], (runner,), coin_count=8, side_count=2)

    _add_gpu_runner_profile(profile, runner, side_count=2)

    assert profile["candidate_bars"] == 1_600
    assert profile["kernel_candidate_bars"] == 1_600


def test_gpu_profile_temporal_dispatches_count_replayed_steps_once():
    proxy = SimpleNamespace(
        batch_size=256, dispatch_batch_size=256, strategy_kind="trailing_martingale"
    )
    runner = SimpleNamespace(
        n=60_000,
        n_coins=29,
        last_profile={
            "batch_size": 256,
            "dispatch_count": 8,
            "cold": True,
            "kernel_candidate_steps": 256 * 59_999,
            "temporal_chunk_bars": 8192,
            "max_dispatch_seconds": 4.0,
            "replay_state_bytes_per_candidate": 29_296,
            "threads_per_threadgroup": 32,
        },
    )
    profile = _new_gpu_proxy_profile(
        proxy, [{}] * 256, (runner,), coin_count=29, side_count=1
    )
    _add_gpu_runner_profile(profile, runner)
    assert profile["kernel_candidate_bars"] == 256 * 59_999 * 29
    assert profile["dispatch_count"] == 8
    assert profile["cold_dispatch_count"] == 1
    assert profile["warm_dispatch_count"] == 7
    assert profile["temporal_dispatches"][0]["max_dispatch_seconds"] == 4.0
    assert profile["temporal_dispatches"][0]["threads_per_threadgroup"] == 32


def test_gpu_profile_candidate_bars_use_truncated_effective_steps():
    proxy = SimpleNamespace(
        batch_size=4,
        dispatch_batch_size=4,
        strategy_kind="ema_anchor",
    )
    runner = SimpleNamespace(
        n=100,
        n_coins=8,
        last_profile={
            "batch_size": 2,
            "dispatch_count": 1,
            "cold": False,
        },
    )
    profile = _new_gpu_proxy_profile(
        proxy, [{}, {}], (runner,), coin_count=8, side_count=2
    )

    _add_gpu_runner_profile(
        profile,
        runner,
        side_count=2,
        effective_candidate_steps=np.asarray([10, 20]),
    )

    assert profile["candidate_bars"] == 3_200
    assert profile["kernel_candidate_bars"] == 480


def test_gpu_profile_unattributed_seconds_excludes_nested_runner_and_transfer():
    timings = {
        "device_to_host": 3.0,
        "candidate_packing": 1.0,
        "upload_and_buffer_clear": 1.0,
        "cold_compilation": 1.0,
        "warm_library_lookup": 0.0,
        "kernel_execution": 2.0,
        "metric_reduction": 2.0,
    }

    assert _gpu_profile_unattributed_seconds(
        timings,
        12.0,
        device_to_host_before=1.0,
        runner_seconds_before=2.0,
    ) == pytest.approx(5.0)




def test_core_output_contract_retains_gross_pnl_aggregates():
    assert {
        "profit_sum",
        "loss_sum",
        "position_unchanged_max_ms",
        "entry_initial_balance_pct",
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "day_fill_count",
        "fill_count",
        "fill_count_entry",
        "fill_count_long",
        "fills_active_days_count",
        "coin_fill_counts",
        "pnl_recovery_max_ms",
        "account_recovery_max_ms",
    } <= CORE_OUTPUT_KEYS




def test_multicoin_exposure_eligibility_unions_fused_sides():
    long = np.zeros((3, 2), dtype=np.float32)
    short = np.zeros((3, 2), dtype=np.float32)
    short[1, 1] = 0.5
    short[2, 1] = np.nan

    eligible = _multicoin_exposure_eligible_coins(
        {"long": long, "short": short},
        ["long", "short"],
        1,
    )

    assert eligible.tolist() == [False, True, True]


@pytest.mark.parametrize("side", ["long", "short"])
def test_directional_entry_initial_metrics_preserve_candidate_batch_shape(side):
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.1, 0.2, 0.3])

    metrics = _directional_entry_initial_metrics(side, values)

    assert metrics[f"entry_initial_balance_pct_{side}"].tolist() == values.tolist()
    other = "short" if side == "long" else "long"
    assert metrics[f"entry_initial_balance_pct_{other}"].shape == values.shape
    assert metrics[f"entry_initial_balance_pct_{other}"].tolist() == [0.0, 0.0, 0.0]


def test_candidate_wallet_exposure_limits_preserve_sides_and_base_fallback():
    torch = pytest.importorskip("torch")

    outputs = _candidate_wallet_exposure_limit_outputs(
        [
            {"long_total_wallet_exposure_limit": 1.25},
            {
                "long_total_wallet_exposure_limit": 1.5,
                "short_total_wallet_exposure_limit": 0.75,
            },
        ],
        {"long": 1.0, "short": 0.5},
        torch=torch,
    )

    assert outputs["candidate_total_wallet_exposure_limit_long"].tolist() == [
        1.25,
        1.5,
    ]
    assert outputs["candidate_total_wallet_exposure_limit_short"].tolist() == [
        0.5,
        0.75,
    ]


@pytest.mark.parametrize("side", ["long", "short"])
def test_directional_gross_pnl_outputs_zero_the_inactive_side(side):
    torch = pytest.importorskip("torch")
    profit = torch.tensor([7.0])
    loss = torch.tensor([2.0])

    outputs = _directional_gross_pnl_outputs(side, profit, loss)
    other_side = "short" if side == "long" else "long"

    assert outputs[f"profit_sum_{side}"] is profit
    assert outputs[f"loss_sum_{side}"] is loss
    assert outputs[f"profit_sum_{other_side}"].item() == 0.0
    assert outputs[f"loss_sum_{other_side}"].item() == 0.0


def test_candidate_position_slots_follow_candidate_positions_and_enabledness():
    torch = pytest.importorskip("torch")

    outputs = _candidate_position_slot_outputs(
        [
            {"long_n_positions": 3.0, "short_total_wallet_exposure_limit": 0.0},
            {"short_n_positions": 2.0},
        ],
        {"long": 2.0, "short": 1.0},
        {"long": 1.0, "short": 0.5},
        torch=torch,
    )

    assert outputs["position_slots_long"].tolist() == [3.0, 2.0]
    assert outputs["position_slots_short"].tolist() == [0.0, 2.0]






def test_directional_hsl_output_contract_retains_lifecycle_and_panic_scalars():
    assert {
        "hsl_triggers_long",
        "hsl_duration_sum_steps",
        "hsl_restart_retrigger_count",
        "hsl_halt_to_restart_equity_loss",
        "hsl_panic_close_loss_sum",
        "hsl_panic_loss_drawdown_count",
        "hsl_strategy_eq_recovery_max_ms_long",
        "hsl_strategy_eq_recovery_max_ms_short",
        "hsl_drawdown_ema_mean_worst_1pct_long",
        "hsl_drawdown_ema_mean_worst_1pct_short",
        "hsl_drawdown_raw_mean_worst_1pct_long",
        "hsl_drawdown_raw_mean_worst_1pct_short",
    } <= DIRECTIONAL_HSL_OUTPUT_KEYS


def test_hsl_diagnostics_are_requested_only_for_hsl_metric_families():
    assert not _hsl_diagnostics_needed(
        {
            "adg_strategy_eq",
            "drawdown_worst_strategy_eq",
            "fills_per_day",
        }
    )
    assert _hsl_diagnostics_needed({"hard_stop_triggers_per_year"})
    assert _hsl_diagnostics_needed({"drawdown_worst_ema_strategy_eq_long"})
    assert _hsl_diagnostics_needed({"peak_recovery_days_strategy_eq_long"})












def test_multicoin_proxy_preserves_directional_hsl_outputs_for_reduction():
    torch = pytest.importorskip("torch")
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.batch_size = 1
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 0.0}
    proxy.run = SimpleNamespace()
    proxy.sides = ["long"]
    proxy.needed_metrics = {"hard_stop_panic_close_loss_sum"}
    proxy.weighted_equity_metrics = ()
    proxy._parameter_matrix = lambda candidates, side=None: np.zeros(
        (len(candidates), 0)
    )
    raw = {
        key: torch.zeros(1)
        for key in (
            "first_fill_ts",
            "last_fill_ts",
            "last_high_ts",
            "first_eq_ts",
            "last_eq_ts",
            "profit_sum",
            "loss_sum",
            "entry_initial_balance_pct",
        )
    }
    raw["hsl_triggers_long"] = torch.tensor([2.0])
    raw["hsl_panic_close_loss_sum"] = torch.tensor([37.5])
    proxy.runners = {"long": SimpleNamespace(run=lambda *args, **kwargs: raw)}

    def reduce(output, *args, **kwargs):
        assert output["hsl_triggers_long"].item() == 2.0
        assert output["hsl_panic_close_loss_sum"].item() == 37.5
        return {"hard_stop_panic_close_loss_sum": output["hsl_panic_close_loss_sum"]}

    proxy._compute_objectives = reduce

    assert proxy.evaluate([{}]) == [{"hard_stop_panic_close_loss_sum": 37.5}]


@pytest.mark.parametrize(
    ("param_keys", "candidate_key"),
    [
        (EMA_ANCHOR_MULTICOIN_PARAM_KEYS, "offset"),
        (
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
            "entry_initial_qty_pct",
        ),
    ],
)
@pytest.mark.parametrize("missing_summary", ["none", "raw", "growth", "calmar", "portfolio_ema"])
def test_multicoin_proxy_routes_dual_side_batch_through_fused_runner(
    param_keys, candidate_key, missing_summary
):
    torch = pytest.importorskip("torch")
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.batch_size = 2
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 1_000.0}
    proxy.run = SimpleNamespace()
    proxy.sides = ["long", "short"]
    proxy.needed_metrics = {
        "none": {"hard_stop_triggers"},
        "raw": {"drawdown_worst_strategy_eq"},
        "growth": {"adg_strategy_eq"},
        "calmar": {"calmar_ratio_strategy_eq"},
        "portfolio_ema": {"drawdown_worst_mean_1pct_ema_strategy_eq"},
    }[missing_summary]
    proxy.param_keys = param_keys
    proxy.base_params = {
        side: {
            key: float(index + (100 if side == "short" else 0))
            for index, key in enumerate(param_keys)
        }
        for side in ("long", "short")
    }
    seen = {}

    def run(params, **kwargs):
        seen["params"] = params.copy()
        seen["kwargs"] = kwargs
        batch = len(params)
        raw = {
            key: torch.zeros(batch)
            for key in (
                "first_fill_ts",
                "last_fill_ts",
                "last_high_ts",
                "first_eq_ts",
                "last_eq_ts",
            )
        }
        raw["hsl_triggers_long"] = torch.tensor([1.0, 2.0])
        raw["hsl_triggers_short"] = torch.tensor([3.0, 4.0])
        return raw

    proxy.fused_runner = SimpleNamespace(run=run)
    proxy.runners = {}

    def reduce(output, *args, **kwargs):
        assert output["hsl_unified_scope"].tolist() == [True, False]
        return {
            "hard_stop_triggers": output["hsl_triggers_long"]
            + output["hsl_triggers_short"]
        }

    proxy._compute_objectives = reduce
    candidates = [
        {f"long_{candidate_key}": 0.25, f"short_{candidate_key}": 0.5,
         "long_hsl_signal_mode": 0, "short_hsl_signal_mode": 0},
        {f"long_{candidate_key}": 0.75, f"short_{candidate_key}": 1.0,
         "long_hsl_signal_mode": 2, "short_hsl_signal_mode": 2},
    ]

    if missing_summary in {"raw", "calmar"}:
        with pytest.raises(RuntimeError, match="raw strategy-risk summary is missing"):
            proxy.evaluate(candidates)
        return

    if missing_summary == "growth":
        with pytest.raises(RuntimeError, match="raw strategy-equity daily summaries are missing"):
            proxy.evaluate(candidates)
        return

    if missing_summary == "portfolio_ema":
        with pytest.raises(RuntimeError, match="portfolio HSL EMA-tail summary is missing"):
            proxy.evaluate(candidates)
        return

    if missing_summary == "weighted":
        with pytest.raises(RuntimeError, match="weighted equity summary is missing"):
            proxy.evaluate(candidates)
        return

    assert proxy.evaluate(candidates) == [
        {"hard_stop_triggers": 4.0},
        {"hard_stop_triggers": 6.0},
    ]
    assert seen["params"].shape == (
        2,
        len(param_keys) * 2,
    )
    parameter_index = param_keys.index(candidate_key)
    side_width = len(param_keys)
    assert seen["params"][:, parameter_index].tolist() == [0.25, 0.75]
    assert seen["params"][:, side_width + parameter_index].tolist() == [0.5, 1.0]
    assert seen["kwargs"] == {"profile": False}


@pytest.mark.parametrize(
    (
        "strategy_kind",
        "runner_name",
        "override_cols",
        "proxy_mode",
        "interval_minutes",
    ),
    [
        (
            "ema_anchor",
            "MpsEmaAnchorMulticoinFusedRunner",
            EMA_ANCHOR_COIN_OVERRIDE_COLS,
            "shared-account-fused-ema-v1",
            1,
        ),
        (
            "trailing_martingale",
            "MpsTrailingMartingaleMulticoinFusedRunner",
            TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
            "shared-account-fused-tm-v1",
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    (
        "needed_metrics",
        "tail_enabled",
        "raw_drawdown_enabled",
        "raw_tail_enabled",
        "recovery_distribution_enabled",
    ),
    [
        ({"hard_stop_time_in_red_pct"}, False, False, False, False),
        ({"drawdown_worst_mean_1pct_ema_strategy_eq"}, True, False, False, False),
        ({"drawdown_worst_strategy_eq_long"}, False, True, False, False),
        ({"drawdown_worst_mean_1pct_strategy_eq_long"}, False, True, True, False),
        ({"strategy_eq_recovery_days_p99"}, False, False, False, True),
        ({"entry_interval_hours_p95"}, False, False, False, False),
        ({"fills_top_symbol_share"}, False, False, False, False),
        ({"fills_active_symbols_count"}, False, False, False, False),
    ],
)
@pytest.mark.parametrize("dynamic_wel_by_tradability", [True, False])
def test_multicoin_proxy_constructs_fused_shared_account_runner(
    monkeypatch,
    strategy_kind,
    runner_name,
    override_cols,
    proxy_mode,
    interval_minutes,
    needed_metrics,
    tail_enabled,
    raw_drawdown_enabled,
    raw_tail_enabled,
    recovery_distribution_enabled,
    dynamic_wel_by_tradability,
):
    torch = pytest.importorskip("torch")

    import optimization.gpu.mps_kernel as mps_kernel
    import optimization.gpu.service as gpu_service

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    backtest = ModuleType("backtest")
    backtest._get_backtest_coin_override = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "backtest", backtest)
    config = get_template_config()
    config['bot']['global']['strategy_kind'] = strategy_kind
    set_side_universe_sources(config, 'approved_coins', {'long': ['BTC'], 'short': ['ETH']})
    set_side_universe_sources(config, 'ignored_coins', {'long': ['ETH'], 'short': ['BTC']})
    config['bot']['global']['hedge_mode'] = False
    config["backtest"]["dynamic_wel_by_tradability"] = dynamic_wel_by_tradability
    for side in ("long", "short"):
        config['bot'][side]['risk'].update({'max_positions': 2, 'total_wallet_exposure_limit': 1.0, 'position_exposure_enforcer': {'enabled': strategy_kind == 'trailing_martingale', 'threshold': 0.7}, 'total_exposure_enforcer': {'enabled': True, 'policy': 'reduce_portfolio', 'threshold': 0.8}})
        # Construction must retain the fused shared-account path when both
        # directional surfaces participate in global auto-unstuck selection.
        config["bot"][side]["unstuck"]["enabled"] = True
        config["bot"][side]["hsl"]["enabled"] = True
        for key in ("ema_span_0", "ema_span_1"):
            strategy = config["bot"][side]["strategy"][strategy_kind]
            span = (
                strategy["entry"]
                if strategy_kind == "trailing_martingale"
                else strategy
            )[key]
            config["bot"][side]["unstuck"][key] = span
            strategy_bounds = config["optimize"]["bounds"][side]["strategy"][
                strategy_kind
            ]
            (
                strategy_bounds["entry"]
                if strategy_kind == "trailing_martingale"
                else strategy_bounds
            )[key] = [
                span,
                span,
            ]
            config["optimize"]["bounds"][side]["unstuck"][key] = [span, span]
        flat = flatten_shared_bot_side(config["bot"][side])

    def bot_payload(side, coin):
        flat = flatten_shared_bot_side(config["bot"][side])
        entry_eligible = (side, coin) in {("long", "BTC"), ("short", "ETH")}
        return {
            "entry_eligible": entry_eligible,
            "wallet_exposure_limit": -1.0 if entry_eligible else 0.0,
            "entry_cooldown_minutes": flat["risk_entry_cooldown_minutes"],
            "filter_volume_ema_span_1m": flat["forager_volume_ema_span_1m"],
            "filter_volatility_ema_span_1m": flat["forager_volatility_ema_span_1m"],
            "forager_score_weights": flat["forager_score_weights"],
            "n_positions": flat["n_positions"],
            "total_wallet_exposure_limit": flat["total_wallet_exposure_limit"],
            "risk_twel_entry_gate_enabled": True,
            "risk_twel_enforcer_enabled": flat["risk_twel_enforcer_enabled"],
            "risk_twel_enforcer_policy": flat["risk_twel_enforcer_policy"],
            "risk_twel_enforcer_threshold": flat["risk_twel_enforcer_threshold"],
            "risk_wel_enforcer_enabled": flat["risk_wel_enforcer_enabled"],
            "risk_wel_enforcer_threshold": flat["risk_wel_enforcer_threshold"],
            "unstuck_enabled": flat["unstuck_enabled"],
            "hsl_enabled": flat["hsl_enabled"],
        }

    payload = SimpleNamespace(
        bot_params_list=[
            {side: bot_payload(side, coin) for side in ("long", "short")}
            for coin in ("BTC", "ETH")
        ],
        strategy_params_list=[
            {
                side: dict(config["bot"][side]["strategy"][strategy_kind])
                for side in ("long", "short")
            }
            for _ in range(2)
        ],
        exchange_params=[
            {
                "qty_step": 0.001,
                "price_step": 0.01,
                "min_qty": 0.001,
                "min_cost": 5.0,
                "c_mult": 1.0,
                "maker_fee": 0.0002,
                "taker_fee": 0.0006,
            }
            for _ in range(2)
        ],
        backtest_params={
            "candle_interval_minutes": interval_minutes,
            "dynamic_wel_by_tradability": dynamic_wel_by_tradability,
            "forager_score_hysteresis_pct": 0.0,
            "last_valid_indices": [2, 3],
            "first_valid_indices": [0, 0],
            "equity_hard_stop_loss": {"signal_mode": "coin"},
            "coins": ["BTC", "ETH"],
            "starting_balance": 1_000.0,
            "global_warmup_bars": 1,
            "trade_start_indices": [1, 1],
            "requested_start_timestamp_ms": 0,
            "first_timestamp_ms": 0,
            "liquidation_threshold": 0.05,
            "filter_by_min_effective_cost": True,
            "max_realized_loss_pct": 1.0,
            "market_order_slippage_pct": 0.0,
            "hedge_mode": config['bot']['global']['hedge_mode'],
        },
    )
    _complete_hsl_payload(payload)
    backtest.build_backtest_payload = lambda *args, **kwargs: payload
    backtest.prep_backtest_args = lambda *args, **kwargs: (
        payload.bot_params_list, payload.strategy_params_list,
        payload.exchange_params, payload.backtest_params,
    )
    built_data_kwargs = {}

    def fake_build_mps_multicoin_data(*args, **kwargs):
        built_data_kwargs.update(kwargs)
        return {
            "n": 4,
            "n_coins": 2,
            "n_days": 1,
            "ts0": 0,
        }

    monkeypatch.setattr(
        gpu_service,
        "build_mps_multicoin_data",
        fake_build_mps_multicoin_data,
    )
    constructed = {}

    class FakeFusedRunner:
        def __init__(self, run, data, **kwargs):
            constructed.update({"run": run, "data": data, "kwargs": kwargs})

    monkeypatch.setattr(mps_kernel, runner_name, FakeFusedRunner)
    values = np.ones((4, 2, 4), dtype=np.float64)
    timestamps = np.arange(4, dtype=np.int64) * interval_minutes * 60_000

    proxy = SharedAccountReplay(
        config=config,
        hlcvs=values,
        mss={"BTC": {}, "ETH": {}},
        btc=np.ones(4, dtype=np.float64),
        timestamps=timestamps,
        exchange="bybit",
        batch_size=8,
        needed_metrics=needed_metrics,
    )

    assert isinstance(proxy.fused_runner, FakeFusedRunner)
    from config.metrics import EXPOSURE_RMS_METRICS
    assert constructed["kwargs"]["exposure_rms_enabled"] == bool(
        needed_metrics & EXPOSURE_RMS_METRICS
    )
    assert built_data_kwargs["include_hourly_ranges"] is True
    assert constructed["run"].interval_ms == interval_minutes * 60_000
    assert proxy.runners == {}
    assert proxy.coin_override_contract["proxy_mode"] == proxy_mode
    assert proxy.coin_override_contract["hsl_proxy_mode"] == proxy_mode
    assert constructed["kwargs"]["long_coin_overrides"].shape == (
        2,
        override_cols,
    )
    assert constructed["kwargs"]["short_coin_overrides"].shape == (
        2,
        override_cols,
    )
    wallet_exposure_column = 11 if strategy_kind == "ema_anchor" else 24
    long_overrides = constructed["kwargs"]["long_coin_overrides"]
    short_overrides = constructed["kwargs"]["short_coin_overrides"]
    assert np.isnan(long_overrides[0, wallet_exposure_column])
    assert long_overrides[1, wallet_exposure_column] == 0.0
    assert short_overrides[0, wallet_exposure_column] == 0.0
    assert np.isnan(short_overrides[1, wallet_exposure_column])
    assert constructed["kwargs"]["hsl_ema_tail_enabled"] is tail_enabled
    assert constructed["kwargs"]["collect_coin_fill_counts"] is bool(
        needed_metrics & {"fills_top_symbol_share", "fills_active_symbols_count"}
    )
    assert constructed["kwargs"]["hsl_raw_drawdown_enabled"] is raw_drawdown_enabled
    assert constructed["kwargs"]["hsl_raw_tail_enabled"] is raw_tail_enabled
    assert (
        constructed["kwargs"]["recovery_distribution_enabled"]
        is recovery_distribution_enabled
    )
    assert "high_exposure_enabled" not in constructed["kwargs"]
    assert constructed["kwargs"]["hedge_mode"] is False
    assert constructed["kwargs"]["filter_by_min_effective_cost"] is True
    assert (
        constructed["kwargs"]["dynamic_wel_by_tradability"]
        is dynamic_wel_by_tradability
    )
    assert constructed["kwargs"]["entry_interval_enabled"] is (
        strategy_kind == "trailing_martingale"
        and "entry_interval_hours_p95" in needed_metrics
    )


def test_gpu_multicoin_proxy_accepts_staggered_valid_gaps_and_ended_tails():
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [99, 98])
    _require_supported_multicoin_valid_tails(hlcvs, [0, 50], [49, 99])
    _require_supported_multicoin_valid_tails(hlcvs, [100, 0], [99, 99])
    _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [98, 97])
    _require_supported_multicoin_valid_tails(hlcvs, [0, 60], [39, 99])
    _require_supported_multicoin_valid_tails(
        np.ones((1401, 2, 4), dtype=np.float64),
        [0, 0],
        [1400, 0],
    )
    with pytest.raises(ValueError, match="at least one prepared coin"):
        _require_supported_multicoin_valid_tails(
            np.ones((100, 0, 4), dtype=np.float64), [], []
        )
    with pytest.raises(ValueError, match="matching first/last"):
        _require_supported_multicoin_valid_tails(hlcvs, [0], [99, 98])
    with pytest.raises(ValueError, match="non-empty prepared valid range"):
        _require_supported_multicoin_valid_tails(hlcvs, [100, 100], [99, 99])
    with pytest.raises(ValueError, match="first_valid_idx within"):
        _require_supported_multicoin_valid_tails(hlcvs, [99, 0], [98, 99])


def test_gpu_multicoin_proxy_rejects_all_nan_candle_in_windows():
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    hlcvs[40, :, :3] = np.nan

    with pytest.raises(ValueError, match="contiguous.*candle_index=40"):
        _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [59, 99])


def test_gpu_multicoin_proxy_rejects_all_nan_first_valid_candle():
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    hlcvs[10, 1, :3] = np.nan

    with pytest.raises(ValueError, match="first-valid candle"):
        _require_supported_multicoin_valid_tails(hlcvs, [0, 10], [99, 99])


@pytest.mark.parametrize(
    "unsupported_value",
    [
        0.0,
        -1.0,
        np.inf,
        float(np.nextafter(0.0, 1.0)),
        float(np.finfo(np.float32).max) * 2.0,
    ],
)
def test_gpu_multicoin_proxy_rejects_finite_unmodeled_all_invalid_candle(
    unsupported_value,
):
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    hlcvs[40, :, :3] = unsupported_value

    with pytest.raises(ValueError, match=r"finite but.*candle_index=40"):
        _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [59, 99])


def test_gpu_multicoin_valid_tail_gate_rejects_out_of_range_last_index():
    with pytest.raises(ValueError, match="prepared valid range"):
        _require_supported_multicoin_valid_tails(
            np.ones((100, 2, 4), dtype=np.float64),
            [0, 0],
            [100, 99],
        )


@pytest.mark.parametrize(
    "unrepresentable_close",
    [np.nextafter(0.0, 1.0), np.finfo(np.float64).max],
)
def test_gpu_multicoin_forced_delist_requires_representable_final_candle(
    unrepresentable_close,
):
    hlcvs = np.ones((1401, 2, 4), dtype=np.float64)
    hlcvs[0, 0, 2] = unrepresentable_close

    with pytest.raises(ValueError, match="forced-delist final candle"):
        _require_supported_multicoin_valid_tails(
            hlcvs,
            [0, 0],
            [0, 1400],
        )


def test_gpu_hsl_requires_contiguous_valid_candles():
    high = np.array([100.0, np.nan, 100.0])
    low = np.array([99.0, np.nan, 99.0])
    close = np.array([99.5, np.nan, 99.5])

    with pytest.raises(ValueError, match="invalid candle at 1"):
        _require_no_internal_invalid_hsl_candles(
            high, low, close, first_valid_idx=0, last_valid_idx=2
        )
    _require_no_internal_invalid_hsl_candles(
        high, low, close, first_valid_idx=2, last_valid_idx=2
    )


def test_gpu_multicoin_hsl_requires_each_coin_to_have_contiguous_valid_candles():
    hlcvs = np.ones((3, 2, 4), dtype=np.float64)
    hlcvs[1, 1, :3] = np.nan

    with pytest.raises(ValueError, match="invalid candle at 1"):
        _require_no_internal_invalid_multicoin_hsl_candles(
            hlcvs,
            hsl_enabled_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[2, 2],
        )
    _require_no_internal_invalid_multicoin_hsl_candles(
        hlcvs,
        hsl_enabled_coins=[True, False],
        first_valid_indices=[0, 0],
        last_valid_indices=[2, 2],
    )


def test_gpu_proxy_rejects_all_internal_gaps():
    hlcvs = np.ones((4, 2, 4), dtype=np.float64)
    hlcvs[2, 1, :3] = np.nan

    with pytest.raises(ValueError, match="all-NaN.*invalid candle at 2"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )

    hlcvs[2, 1, :3] = 0.0
    with pytest.raises(ValueError, match="coin index 1, invalid candle at 2"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )

    hlcvs[2, 1, :3] = [np.nan, np.nan, 1.0]
    with pytest.raises(ValueError, match="partially invalid"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )

    hlcvs[2, 1, :3] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )








@pytest.mark.parametrize(
    ("mode", "scale_budget"), [("unscaled", 0.0), ("scaled", 1.0)]
)
def test_single_coin_exposure_policy_packs_rust_inputs(mode, scale_budget):
    packed = _single_coin_exposure_params({"we_excess_allowance_pct": 0.25, "total_exposure_entry_gate_enabled": False, "total_exposure_enforcer": {"threshold": 0.8}}, side="long", scale_hsl_budget=bool(scale_budget))
    assert packed == {"we_excess_allowance_pct": 0.25, "hsl_scale_budget_with_excess_allowance": scale_budget, "twel_entry_gate_enabled": 1.0, "twel_enforcer_threshold": 0.8}



def test_single_coin_hsl_budget_scaling_defaults_off():
    assert _single_coin_exposure_params({}, side="short")["hsl_scale_budget_with_excess_allowance"] == 0.0



def test_tm_position_exposure_repair_packs_exact_rust_inputs():
    assert _position_exposure_enforcer_params(
        {
            "position_exposure_enforcer": {"enabled": True, "threshold": 0.8},
        },
        side="short",
    ) == {
        "wel_enforcer_enabled": 1.0,
        "wel_enforcer_threshold": 0.8,
    }

    with pytest.raises(ValueError, match="finite positive"):
        _position_exposure_enforcer_params(
            {
                "position_exposure_enforcer": {"enabled": True, "threshold": 0.0},
            },
            side="long",
        )


@pytest.mark.parametrize(
    ("policy", "reduce_portfolio"),
    [("reduce_overweight", 0.0), ("reduce_portfolio", 1.0)],
)
def test_tm_total_exposure_repair_packs_exact_rust_inputs(policy, reduce_portfolio):
    assert _total_exposure_enforcer_params(
        {
            "total_exposure_enforcer": {"enabled": True, "policy": policy},
        },
        side="long",
    ) == {
        "twel_enforcer_enabled": 1.0,
        "twel_enforcer_reduce_portfolio": reduce_portfolio,
    }

    with pytest.raises(ValueError, match=r"risk\.total_exposure_enforcer\.policy"):
        _total_exposure_enforcer_params(
            {"total_exposure_enforcer": {"policy": "largest_loss"}},
            side="short",
        )


def test_single_coin_unstuck_packs_exact_rust_inputs():
    assert _unstuck_params(
        {
            "unstuck_ema_span_0": 17.25,
            "unstuck_ema_span_1": 211.75,
            "unstuck_enabled": True,
            "unstuck_ema_gating_enabled": False,
            "unstuck_close_pct": 0.125,
            "unstuck_ema_dist": -0.01,
            "unstuck_loss_allowance_pct": 0.02,
            "unstuck_threshold": 0.85,
        }
    ) == {
        "unstuck_ema_span_0": 17.25,
        "unstuck_ema_span_1": 211.75,
        "unstuck_enabled": 1.0,
        "unstuck_ema_gating_enabled": 0.0,
        "unstuck_close_pct": 0.125,
        "unstuck_ema_dist": -0.01,
        "unstuck_loss_allowance_pct": 0.02,
        "unstuck_threshold": 0.85,
    }


def test_single_coin_hsl_packs_state_machine_inputs():
    packed = _hsl_params(
        {
            "hsl_enabled": True,
            "hsl_scale_budget_with_excess_allowance": False,
            "hsl_red_threshold": 0.2,
            "hsl_drawdown_ratio_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 120.0,
            "hsl_memory_ema_span_minutes": 43200.5,
            "n_positions": 4,
            "hsl_slot_count": 1,
        },
        signal_mode="coin",
    )

    assert packed == {
        "hsl_enabled": 1.0,
        "hsl_scale_budget_with_excess_allowance": 0.0,
        "hsl_red_threshold": 0.2,
        "hsl_drawdown_ratio_ema_span_minutes": 60.0,
        "hsl_cooldown_ms_lo": 7200000.0,
        "hsl_memory_ema_span_minutes": 43200.5,
        "hsl_signal_mode": 2.0,
        "hsl_cooldown_ms_hi": 0.0,
    }


@pytest.mark.parametrize(
    ("signal_mode", "expected_id"),
    [("portfolio", 0.0), ("pside", 1.0), ("coin", 2.0)],
)
def test_single_coin_hsl_packs_explicit_signal_mode_ids(signal_mode, expected_id):
    packed = _hsl_params(
        {
            "hsl_enabled": False,
            "hsl_scale_budget_with_excess_allowance": False,
            "hsl_memory_ema_span_minutes": 43200.0,
            "hsl_red_threshold": 0.1,
            "hsl_drawdown_ratio_ema_span_minutes": 1.0,
            "hsl_cooldown_minutes_after_red": 0.0,
            "hsl_slot_count": 1,
        },
        signal_mode=signal_mode,
    )

    assert packed["hsl_signal_mode"] == expected_id


@pytest.mark.parametrize("signal_mode", ["portfolio", "coin", "pside"])
def test_dual_side_single_coin_hsl_accepts_all_signal_modes(signal_mode):
    validate_hsl_signal_topology(signal_mode, coin_count=1, enabled_side_count=2)


def test_one_sided_single_coin_hsl_accepts_portfolio_signal_mode():
    validate_hsl_signal_topology("portfolio", coin_count=1, enabled_side_count=1)


def test_single_coin_hsl_rejects_unknown_signal_mode():
    with pytest.raises(ValueError, match="coin, position_side, or portfolio"):
        validate_hsl_signal_topology("unified", coin_count=1, enabled_side_count=1)


@pytest.mark.parametrize("signal_mode", ["portfolio", "pside", "coin"])
def test_one_sided_multicoin_hsl_accepts_all_signal_modes(signal_mode):
    validate_hsl_signal_topology(signal_mode, coin_count=3, enabled_side_count=1)


def test_dual_side_multicoin_hsl_accepts_decomposable_pside_mode():
    validate_hsl_signal_topology("pside", coin_count=3, enabled_side_count=2)


@pytest.mark.parametrize("signal_mode", ["portfolio", "pside", "coin"])
def test_dual_side_multicoin_hsl_accepts_shared_account_controller(signal_mode):
    validate_hsl_signal_topology(
        signal_mode,
        coin_count=3,
        enabled_side_count=2,
        shared_account_controller=True,
    )


@pytest.mark.parametrize("signal_mode", ["coin", "portfolio"])
def test_dual_side_multicoin_hsl_rejects_joint_account_modes(signal_mode):
    with pytest.raises(ValueError, match="supports only position_side"):
        validate_hsl_signal_topology(signal_mode, coin_count=3, enabled_side_count=2)


def test_directional_parameter_matrix_keeps_side_values_separate():
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_PARAM_KEYS},
    }
    proxy.param_keys = EMA_ANCHOR_PARAM_KEYS

    matrix = np.concatenate([proxy._parameter_matrix([{"long_offset": 0.125, "short_offset": 0.25}], side=side) for side in ("long", "short")], axis=1)

    assert matrix.shape == (1, 2 * len(EMA_ANCHOR_PARAM_KEYS))
    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125
    assert matrix[0, len(EMA_ANCHOR_PARAM_KEYS) + offset_index] == 0.25




@pytest.mark.parametrize(("side", "base"), [("long", 1.0), ("short", 2.0)])
def test_multicoin_parameter_matrix_uses_only_enabled_side(side, base):
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.sides = [side]
    proxy.base_params = {side: {key: base for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS}}

    other_side = "short" if side == "long" else "long"
    matrix = proxy._parameter_matrix(
        [
            {
                f"{side}_offset": 0.125,
                f"{side}_we_excess_allowance_pct": 0.25,
                f"{side}_twel_enforcer_threshold": 0.8,
                f"{other_side}_offset": 9.0,
            }
        ]
    )

    assert matrix.shape == (1, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS))
    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125
    assert matrix[
        0, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("we_excess_allowance_pct")
    ] == pytest.approx(0.25)
    assert matrix[
        0, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
    ] == pytest.approx(0.8)


def test_multicoin_parameter_matrix_keeps_dual_side_values_separate():
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.sides = ["long", "short"]
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS},
    }
    candidate = {"long_offset": 0.125, "short_offset": 0.25}

    long_matrix = proxy._parameter_matrix([candidate], "long")
    short_matrix = proxy._parameter_matrix([candidate], "short")

    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert long_matrix[0, offset_index] == 0.125
    assert short_matrix[0, offset_index] == 0.25


@pytest.mark.parametrize("side", ["long", "short"])
def test_multicoin_tm_parameter_matrix_keeps_forager_and_strategy_values(side):
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.sides = [side]
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    proxy.base_params = {
        side: {key: 1.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS}
    }

    matrix = proxy._parameter_matrix(
        [
            {
                f"{side}_entry_threshold_base_pct": 0.125,
                f"{side}_forager_score_weights_volume": 0.25,
                f"{side}_we_excess_allowance_pct": 0.4,
                f"{side}_twel_enforcer_threshold": 0.75,
            }
        ]
    )

    assert matrix.shape == (1, len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS))
    assert (
        matrix[
            0,
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_threshold_base_pct"),
        ]
        == 0.125
    )
    assert (
        matrix[
            0,
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("forager_score_weights_volume"),
        ]
        == 0.25
    )
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("we_excess_allowance_pct"),
    ] == pytest.approx(0.4)
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold"),
    ] == pytest.approx(0.75)


def test_multicoin_tm_parameter_matrix_keeps_dual_side_values_separate():
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.sides = ["long", "short"]
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    proxy.base_params = {
        "long": {key: 1.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS},
        "short": {key: 2.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS},
    }
    candidate = {
        "long_entry_threshold_base_pct": 0.125,
        "short_entry_threshold_base_pct": 0.25,
    }

    long_matrix = proxy._parameter_matrix([candidate], "long")
    short_matrix = proxy._parameter_matrix([candidate], "short")

    index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_threshold_base_pct")
    assert long_matrix[0, index] == 0.125
    assert short_matrix[0, index] == 0.25








def test_multicoin_coin_overrides_pack_only_explicit_exact_values():
    strategy_base = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    strategy_override = dict(strategy_base, offset=0.25, ema_span_0=90.0)
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base},
            {"long": strategy_override},
        ],
        bot_params_list=[
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "risk_entry_cooldown_minutes": 15.0,
                    "wallet_exposure_limit": 0.4,
                    "risk_we_excess_allowance_pct": 0.25,
                    "unstuck_ema_span_0": 17.25,
                    "unstuck_ema_span_1": 211.75,
                    "unstuck_enabled": True,
                    "unstuck_ema_gating_enabled": False,
                    "unstuck_close_pct": 0.125,
                    "unstuck_ema_dist": -0.01,
                    "unstuck_loss_allowance_pct": 0.02,
                    "unstuck_threshold": 0.85,
                }
            },
        ],
    )
    _complete_hsl_payload(payload)
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "ema_anchor": {"offset": 0.25, "ema_span_0": 90.0}
                        },
                        "entry_cooldown": {"base_duration_minutes": 15.0},
                        "risk": {
                            "we_excess_allowance_pct": 0.25,
                        },
                        "wallet_exposure_limit": 0.4,
                        "unstuck": {
                            "ema_span_0": 17.25,
                            "ema_span_1": 211.75,
                            "enabled": True,
                            "ema_gating_enabled": False,
                            "close_pct": 0.125,
                            "ema_distance_fraction": -0.01,
                            "loss_allowance_pct": 0.02,
                            "exposure_trigger_ratio": 0.85,
                        },
                    }
                }
            }
        }
    }

    matrix, contract = build_ema_coin_parameters(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, EMA_ANCHOR_COIN_OVERRIDE_COLS)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("offset")] == pytest.approx(0.25)
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("ema_span_0")] == pytest.approx(90.0)
    assert matrix[1, 10] == pytest.approx(15.0)
    assert matrix[1, 11] == pytest.approx(0.4)
    assert matrix[1, 12] == pytest.approx(0.25)
    assert matrix[1, 13:19].tolist() == pytest.approx(
        [1.0, 0.0, 0.125, -0.01, 0.02, 0.85]
    )
    start = EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN
    assert np.isnan(matrix[1, 19:start]).all()
    assert matrix[1, start : start + 2].tolist() == pytest.approx([17.25, 211.75])
    assert np.isnan(matrix[1, start + 2 :]).all()
    assert contract["coins"] == ["BTC", "ETH"]
    assert contract["values"][0] == [None] * EMA_ANCHOR_COIN_OVERRIDE_COLS
    assert contract["exact_overrides"] == [
        {},
        config["coin_overrides"]["ETH"],
    ]


def test_coin_override_contract_keeps_exact_values_beyond_float32_precision():
    first = 0.4
    second = float(np.nextafter(first, 1.0))

    def build(value):
        config = {
            "coin_overrides": {
                "ETH": {
                    "bot": {"long": {"strategy": {"ema_anchor": {"offset": value}}}}
                }
            }
        }
        payload = SimpleNamespace(
            strategy_params_list=[{"long": {"offset": value}}],
            bot_params_list=[
                {
                    "long": {
                        "entry_eligible": True,
                        "wallet_exposure_limit": -1.0,
                    }
                }
            ],
        )
        _complete_hsl_payload(payload)
        return build_ema_coin_parameters(
            config=config,
            mss={"ETH": {}},
            exchange="bybit",
            coins=["ETH"],
            payload=payload,
            side="long",
            resolve_override=lambda config, _mss, _exchange, coin: config[
                "coin_overrides"
            ].get(coin, {}),
        )

    first_matrix, first_contract = build(first)
    second_matrix, second_contract = build(second)

    assert first_matrix.dtype == second_matrix.dtype == np.float64
    assert not np.array_equal(first_matrix, second_matrix, equal_nan=True)
    assert np.array_equal(first_matrix.astype(np.float32), second_matrix.astype(np.float32), equal_nan=True)
    assert first_contract["values"] != second_contract["values"]
    assert first_contract["exact_overrides"] != second_contract["exact_overrides"]
    first_exact = first_contract["exact_overrides"][0]["bot"]["long"]
    second_exact = second_contract["exact_overrides"][0]["bot"]["long"]
    assert first_exact["strategy"]["ema_anchor"]["offset"] == first
    assert second_exact["strategy"]["ema_anchor"]["offset"] == second


def test_coin_override_contract_keeps_backtest_inert_live_values():
    exact_patch = {
        "live": {
            "modes": {"long": "graceful_stop", "short": "panic"},
            "account": {"leverage": 3},
        }
    }
    config = {"coin_overrides": {"ETH": exact_patch}}
    payload = SimpleNamespace(
        strategy_params_list=[{"long": {}}],
        bot_params_list=[
            {
                "long": {
                    "entry_eligible": True,
                    "is_forced_active": False,
                    "wallet_exposure_limit": -1.0,
                }
            }
        ],
    )
    _complete_hsl_payload(payload)

    matrix, contract = build_ema_coin_parameters(
        config=config,
        mss={"ETH": {}},
        exchange="bybit",
        coins=["ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert np.isnan(matrix).all()
    assert contract["values"] == [[None] * EMA_ANCHOR_COIN_OVERRIDE_COLS]
    assert contract["exact_overrides"] == [exact_patch]


@pytest.mark.parametrize(
    ("builder", "strategy", "forced_column"),
    [
        (
            build_ema_coin_parameters,
            {},
            EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
        ),
        (
            build_tm_coin_parameters,
            {"entry": {}, "close": {}},
            TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
        ),
    ],
)
def test_multicoin_coin_overrides_pack_forced_normal_active_slot(
    builder, strategy, forced_column
):
    payload = SimpleNamespace(
        strategy_params_list=[{"long": strategy}, {"long": strategy}],
        bot_params_list=[
            {
                "long": {
                    "entry_eligible": True,
                    "is_forced_active": False,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "entry_eligible": True,
                    "is_forced_active": True,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
        ],
    )
    _complete_hsl_payload(payload)
    config = {"coin_overrides": {"ETH": {"live": {"modes": {"long": "normal"}}}}}

    matrix, contract = builder(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert np.isnan(matrix[0, forced_column])
    assert matrix[1, forced_column] == 1.0
    assert contract["values"][0][forced_column] is None
    assert contract["values"][1][forced_column] == 1.0


def test_multicoin_coin_overrides_pack_dual_sides_independently():
    strategy_base = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base, "short": strategy_base},
            {
                "long": dict(strategy_base, offset=0.25),
                "short": dict(strategy_base, offset=0.5),
            },
        ],
        bot_params_list=[
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": -1.0,
                },
                "short": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": -1.0,
                },
            },
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": 0.4,
                },
                "short": {
                    "risk_entry_cooldown_minutes": 30.0,
                    "wallet_exposure_limit": -1.0,
                },
            },
        ],
    )
    _complete_hsl_payload(payload)
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {"ema_anchor": {"offset": 0.25}},
                        "wallet_exposure_limit": 0.4,
                    },
                    "short": {
                        "strategy": {"ema_anchor": {"offset": 0.5}},
                        "entry_cooldown": {"base_duration_minutes": 30.0},
                    },
                }
            }
        }
    }

    def resolver(config, _mss, _exchange, coin):
        return config["coin_overrides"].get(coin, {})

    long_matrix, _ = build_ema_coin_parameters(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=resolver,
    )
    short_matrix, _ = build_ema_coin_parameters(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="short",
        resolve_override=resolver,
    )

    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    assert long_matrix[1, offset_index] == pytest.approx(0.25)
    assert long_matrix[1, 11] == pytest.approx(0.4)
    assert np.isnan(long_matrix[1, 10])
    assert short_matrix[1, offset_index] == pytest.approx(0.5)
    assert short_matrix[1, 10] == pytest.approx(30.0)
    assert np.isnan(short_matrix[1, 11])


@pytest.mark.parametrize(
    ("builder", "wallet_exposure_column"),
    [
        (build_ema_coin_parameters, 11),
        (build_tm_coin_parameters, 24),
    ],
)
def test_multicoin_coin_overrides_preserve_side_entry_eligibility(
    builder, wallet_exposure_column
):
    strategy = {
        "entry": {},
        "close": {},
    }
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy},
            {"long": strategy},
        ],
        bot_params_list=[
            {
                "long": {
                    "entry_eligible": False,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.0,
                }
            },
            {
                "long": {
                    "entry_eligible": True,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
        ],
    )
    _complete_hsl_payload(payload)

    matrix, contract = builder(
        config={"coin_overrides": {}},
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda *_args: {},
    )

    assert matrix[0, wallet_exposure_column] == 0.0
    assert np.isnan(matrix[1, wallet_exposure_column])
    assert contract["values"][0][wallet_exposure_column] == 0.0
    assert contract["values"][1][wallet_exposure_column] is None


def test_multicoin_coin_overrides_pack_complete_hsl_group():
    strategy = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    effective_hsl = {
        "hsl_enabled": False,
        "hsl_red_threshold": 0.2,
        "hsl_drawdown_ratio_ema_span_minutes": 5.5,
        "hsl_memory_ema_span_minutes": 7.5,
        "hsl_scale_budget_with_excess_allowance": True,
        "hsl_cooldown_minutes_after_red": 12.5,
        "hsl_panic_close_order_type": "market",
    }
    payload = SimpleNamespace(
        strategy_params_list=[{"long": strategy}, {"long": strategy}],
        bot_params_list=[
            {"long": {}},
            {"long": effective_hsl},
        ],
    )
    _complete_hsl_payload(payload)
    hsl_patch = {
        "enabled": False,
        "drawdown_ratio_threshold": 0.2,
        "drawdown_ratio_ema_span_minutes": 5.5,
        "memory_ema_span_minutes": 7.5,
        "scale_budget_with_excess_allowance": True,
        "cooldown_minutes_after_red": 12.5,
        "panic_close_order_type": "market",
    }
    config = {"coin_overrides": {"ETH": {"bot": {"long": {"hsl": hsl_patch}}}}}

    matrix, contract = build_ema_coin_parameters(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, EMA_ANCHOR_COIN_OVERRIDE_COLS)
    assert np.isnan(matrix[0]).all()
    assert matrix[
        1, 19:EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
    ].tolist() == pytest.approx([0.0, 0.2, 5.5, 750000.0, 0.0, 1.0, 7.5, 1.0])
    assert contract["values"][1][
        19:EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
    ] == pytest.approx(
        matrix[1, 19:EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN].tolist()
    )
    assert np.isnan(matrix[1, EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN])


def test_multicoin_coin_overrides_reject_invalid_hsl_panic_order_type():
    strategy = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    payload = SimpleNamespace(
        strategy_params_list=[{"long": strategy}, {"long": strategy}],
        bot_params_list=[
            {"long": {}},
            {
                "long": {
                    "hsl_enabled": True,
            "hsl_scale_budget_with_excess_allowance": False,
                    "hsl_panic_close_order_type": "makret",
                }
            },
        ],
    )
    _complete_hsl_payload(payload)
    config = {
        "coin_overrides": {
            "ETH": {"bot": {"long": {"hsl": {"panic_close_order_type": "makret"}}}}
        }
    }

    with pytest.raises(ValueError, match="to be limit or market"):
        build_ema_coin_parameters(
            config=config,
            mss={"BTC": {}, "ETH": {}},
            exchange="bybit",
            coins=["BTC", "ETH"],
            payload=payload,
            side="long",
            resolve_override=lambda config, _mss, _exchange, coin: config[
                "coin_overrides"
            ].get(coin, {}),
        )


def test_multicoin_tm_coin_overrides_pack_only_explicit_exact_values():
    assert (
        tuple(key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
        == TRAILING_MARTINGALE_PARAM_KEYS[:23]
    )
    strategy_base = {
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {
            "ema_span_0": 10.0,
            "ema_span_1": 20.0,
            "ema_gate_mode": "all",
            "double_down_factor": 1.1,
            "initial_ema_dist": 0.01,
            "initial_qty_pct": 0.02,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_we_weight": 0.08,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
        "close": {
            "qty_pct": 0.2,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
    }
    strategy_override = {
        **strategy_base,
        "entry": {
            **strategy_base["entry"],
            "ema_gate_mode": "reentry",
            "threshold_base_pct": 0.25,
            "retracement_base_pct": 1.0e-50,
        },
        "close": {
            **strategy_base["close"],
            "qty_pct": 0.5,
            "retracement_base_pct": 1.0e-50,
        },
    }
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base},
            {"long": strategy_override},
        ],
        bot_params_list=[
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "risk_entry_cooldown_minutes": 15.0,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.4,
                    "risk_we_excess_allowance_pct": 0.25,
                    "risk_wel_enforcer_enabled": True,
                    "risk_wel_enforcer_threshold": 0.8,
                    "unstuck_enabled": True,
                    "unstuck_ema_gating_enabled": False,
                    "unstuck_close_pct": 0.125,
                    "unstuck_ema_dist": -0.01,
                    "unstuck_loss_allowance_pct": 0.02,
                    "unstuck_threshold": 0.85,
                }
            },
        ],
    )
    _complete_hsl_payload(payload)
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "trailing_martingale": {
                                "entry": {
                                    "ema_gate_mode": "reentry",
                                    "threshold_base_pct": 0.25,
                                    "retracement_base_pct": 1.0e-50,
                                },
                                "close": {
                                    "qty_pct": 0.5,
                                    "retracement_base_pct": 1.0e-50,
                                },
                            }
                        },
                        "entry_cooldown": {"base_duration_minutes": 15.0},
                        "risk": {
                            "we_excess_allowance_pct": 0.25,
                            "position_exposure_enforcer": {
                                "enabled": True,
                                "threshold": 0.8,
                            },
                        },
                        "wallet_exposure_limit": 0.4,
                        "unstuck": {
                            "enabled": True,
                            "ema_gating_enabled": False,
                            "close_pct": 0.125,
                            "ema_distance_fraction": -0.01,
                            "loss_allowance_pct": 0.02,
                            "exposure_trigger_ratio": 0.85,
                        },
                    }
                }
            }
        }
    }

    matrix, contract = build_tm_coin_parameters(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, 7] == pytest.approx(0.25)
    assert matrix[1, 11] == 1.0e-50  # Preserve authored precision until shader packing.
    assert matrix[1, 15] == pytest.approx(0.5)
    assert matrix[1, 20] == 1.0e-50
    assert matrix[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN] == 0.0
    assert matrix[1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN] == 1.0
    assert matrix[1, 23] == pytest.approx(15.0)
    assert matrix[1, 24] == pytest.approx(0.4)
    assert matrix[1, 25] == pytest.approx(0.25)
    assert matrix[1, 26] == pytest.approx(1.0)
    assert matrix[1, 27] == pytest.approx(0.8)
    assert matrix[1, 28:34].tolist() == pytest.approx(
        [1.0, 0.0, 0.125, -0.01, 0.02, 0.85]
    )
    assert np.isnan(
        matrix[
            1,
            34:TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
        ]
    ).all()
    assert contract["values"][0] == [None] * TRAILING_MARTINGALE_COIN_OVERRIDE_COLS


def test_trailing_parameter_matrix_keeps_nested_flattened_sides_separate():
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"
    proxy.param_keys = TRAILING_MARTINGALE_PARAM_KEYS
    proxy.base_params = {
        "long": {key: 1.0 for key in TRAILING_MARTINGALE_PARAM_KEYS},
        "short": {key: 2.0 for key in TRAILING_MARTINGALE_PARAM_KEYS},
    }

    candidate = {"long_entry_threshold_base_pct": 0.125, "short_close_qty_pct": 0.25}
    matrix = np.concatenate([proxy._parameter_matrix([candidate], side=side)
                             for side in ("long", "short")], axis=1)

    assert matrix.shape == (1, 2 * len(TRAILING_MARTINGALE_PARAM_KEYS))
    entry_index = TRAILING_MARTINGALE_PARAM_KEYS.index("entry_threshold_base_pct")
    close_index = TRAILING_MARTINGALE_PARAM_KEYS.index("close_qty_pct")
    assert matrix[0, entry_index] == 0.125
    assert matrix[0, len(TRAILING_MARTINGALE_PARAM_KEYS) + close_index] == 0.25


def test_gpu_side_enablement_uses_config_risk_not_per_coin_sentinel():
    config = {
        "bot": {
            "long": {
                "risk": {"total_wallet_exposure_limit": 1.0, "max_positions": 1},
                "universe": {"approved_coins": ["BTC"]},
            },
            "short": {
                "risk": {"total_wallet_exposure_limit": 0.0, "max_positions": 0},
                "universe": {"approved_coins": ["BTC"]},
            },
        },
        "live": {},
    }

    assert gpu_side_enabled(config, "long")
    assert not gpu_side_enabled(config, "short")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("disabled", (0.0, 0.0)),
        ("all", (1.0, 1.0)),
        ("initial", (1.0, 0.0)),
        ("reentry", (0.0, 1.0)),
    ],
)
def test_trailing_martingale_flattening_preserves_nested_params_and_gates(
    mode, expected
):
    strategy = {
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {
            "ema_span_0": 10.0,
            "ema_span_1": 20.0,
            "ema_gate_mode": mode,
            "double_down_factor": 1.1,
            "initial_ema_dist": 0.01,
            "initial_qty_pct": 0.02,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_we_weight": 0.08,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
        "close": {
            "qty_pct": 0.2,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
    }
    flattened = flatten_trailing_martingale_params(
        strategy,
        {"entry_cooldown_minutes": 7.0, "total_wallet_exposure_limit": 1.5},
    )

    assert tuple(flattened) == TRAILING_MARTINGALE_PARAM_KEYS
    assert flattened["entry_double_down_factor"] == 1.1
    assert flattened["close_qty_pct"] == 0.2
    assert (flattened["gate_initial"], flattened["gate_reentry"]) == expected


def test_trailing_martingale_flattening_rejects_unknown_gate_mode():
    with pytest.raises(ValueError, match="ema_gate_mode"):
        flatten_trailing_martingale_params(
            {
                "entry": {"ema_gate_mode": "mystery"},
                "close": {},
            },
            {"total_wallet_exposure_limit": 1.0},
        )


def test_trailing_martingale_flattening_reads_canonical_payload_cooldown():
    strategy = {
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {"ema_span_0": 10.0, "ema_span_1": 20.0, "ema_gate_mode": "all"},
        "close": {},
    }
    for key in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS:
        _name, path = key
        target = strategy
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target.setdefault(path[-1], 0.1)

    flattened = flatten_trailing_martingale_params(
        strategy,
        {
            "risk_entry_cooldown_minutes": 37.0,
            "total_wallet_exposure_limit": 1.5,
        },
    )

    assert flattened["entry_cooldown_minutes"] == 37.0






def test_parameter_columns_keep_candidate_fallback_and_missing_key_failure():
    proxy = SharedAccountReplay.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy.param_keys = ("offset",)
    proxy.sides = ["short"]
    proxy.base_params = {"short": {}}
    np.testing.assert_array_equal(
        proxy._parameter_matrix([{"short_offset": 0.125}]), [[0.125]]
    )
    with pytest.raises(KeyError, match="offset"):
        proxy._parameter_matrix([{}])
    with pytest.raises((TypeError, ValueError)):
        proxy._parameter_matrix([{"short_offset": None}])


def _suite_batch_proxy():
    proxy = object.__new__(SharedAccountReplay)
    proxy.hsl_signal_mode = "coin"

    proxy._torch = object()
    proxy.strategy_kind = "trailing_martingale"
    proxy.sides = ["long"]
    proxy.data = object()
    proxy.run = ProxyRun(1000, 5, 10, 600000, 600000, 0, 60000, 0.05, 0, 999)
    proxy._runner_specs = {
        "long": (
            SimpleNamespace,
            dict(
                side="long",
                coin_overrides=np.array([[np.nan, 1.0]]),
                btc_prices=np.array([1.0, 2.0]),
            ),
        )
    }
    proxy.checkpoint_contract = dict(
        base_params={"long": {"n_positions": 4}}, backtest={"max_realized_loss_pct": 1}
    )
    proxy.coin_override_contract = {"coins": ["A", "B"]}
    proxy.needed_metrics = {"adg_strategy_eq"}
    proxy.weighted_equity_metrics = ()
    proxy.couple_unstuck_emas = False
    proxy.batch_size = 1024
    proxy.max_dispatch_candidate_bars = 500000000
    return proxy


@pytest.mark.parametrize(
    "changed",
    [
        "data",
        "run",
        "runtime",
        "overrides",
        "btc",
        "metrics",
        "coupling",
        "batch",
        "work_limit",
        "interrupt",
    ],
)
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_suite_batch_key_rejects_changed_execution_inputs(monkeypatch, changed, device):
    import copy
    from dataclasses import replace
    from optimization.gpu import service

    monkeypatch.setattr(service, "gpu_device", lambda _: device)
    first = _suite_batch_proxy()
    second = copy.copy(first)
    original = first.suite_batch_key()
    assert original is not None
    second.checkpoint_contract = copy.deepcopy(first.checkpoint_contract)
    second.checkpoint_contract["base_params"]["long"]["n_positions"] = 20
    assert second.suite_batch_key() == original
    if changed == "data":
        second.data = object()
    elif changed == "run":
        second.run = replace(first.run, starting_balance=2000)
    elif changed == "runtime":
        second.checkpoint_contract["backtest"]["max_realized_loss_pct"] = 0.5
    elif changed in ("overrides", "btc"):
        second._runner_specs = copy.deepcopy(first._runner_specs)
        second._runner_specs["long"][1][
            "coin_overrides" if changed == "overrides" else "btc_prices"
        ].flat[-1] = 9
    elif changed == "metrics":
        second.needed_metrics = {"sortino_ratio_strategy_eq"}
    elif changed == "coupling":
        second.couple_unstuck_emas = True
    elif changed == "batch":
        second.batch_size = 512
    elif changed == "interrupt":
        second._runner_specs = copy.deepcopy(first._runner_specs)
        second._runner_specs["long"][1]["interrupt_check"] = lambda: None
    else:
        second.max_dispatch_candidate_bars = 1000
    assert second.suite_batch_key() != original


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_suite_batching_rejects_unsupported_topologies(monkeypatch, device):
    from optimization.gpu import service

    proxy = _suite_batch_proxy()
    monkeypatch.setattr(service, "gpu_device", lambda _: device)
    proxy.sides = ["long", "short"]
    assert proxy.suite_batch_key() is None
    proxy.sides = ["long"]
    proxy.strategy_kind = "ema_anchor"
    assert proxy.suite_batch_key() is None


def test_suite_materialization_preserves_parameter_and_metric_defaults():
    proxy = _suite_batch_proxy()
    proxy.param_keys = ["n_positions", "total_wallet_exposure_limit", "ema_span_0"]
    proxy.base_params = {"long": dict(zip(proxy.param_keys, [4, 1.5, 32]))}
    proxy.base_total_wallet_exposure_limits = {"long": 1.5, "short": 0}
    proxy.base_n_positions = {"long": 4, "short": 0}
    original = [{"long_ema_span_0": 64}]
    result = proxy.materialize_suite_candidates(original)
    assert result == [
        dict(
            long_n_positions=4.0,
            long_total_wallet_exposure_limit=1.5,
            long_ema_span_0=64.0,
            short_n_positions=0,
            short_total_wallet_exposure_limit=0,
        )
    ]
    np.testing.assert_array_equal(
        proxy._parameter_matrix(result), proxy._parameter_matrix(original)
    )
    assert original == [{"long_ema_span_0": 64}]


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.99999999, 1.0, 1e308])
def test_realized_gate_fraction_uses_shared_nearest_f32_graph(fraction):
    pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _encode_max_realized_loss_pct

    assert _encode_max_realized_loss_pct(fraction) == float(np.float32(min(fraction, 1.0)))
    if fraction == 0.99999999:
        # The separately packed configured flag must preserve this active gate.
        assert fraction < 1.0 and _encode_max_realized_loss_pct(fraction) == 1.0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("device,factual,expected", [
    ("cuda", True, (True, 64, 128)),
    ("mps", True, (True, 64, 128)),
])
def test_native_plan_has_interrupt_boundaries_on_both_devices(strategy, device, factual, expected):
    assert _mps_multicoin_dispatch_plan(
        strategy, 64, n_bars=11520, n_coins=25, n_sides=2,
        max_candidate_bars=1_000_000_000, device=device, factual_hsl=factual,
    ) == expected


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("bars", [128, 512, 8192])
def test_native_short_replays_have_interrupt_boundaries(strategy, bars):
    assert _mps_multicoin_dispatch_plan(
        strategy, 64, n_bars=bars, n_coins=25, n_sides=2,
        max_candidate_bars=1_000_000_000, device="cuda", factual_hsl=True,
    ) == (True, 64, 128)
