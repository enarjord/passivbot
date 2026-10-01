"""GPU finite RMS, timing-policy and exact-Rust parity regressions."""

from pathlib import Path
from functools import lru_cache
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
from optimization.gpu.model import ADAPTIVE_PARAM_KEYS, adaptive_params
from optimization.gpu.service import _candidate_parameter_matrix

GPU = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def test_candidate_fractional_span_window_and_cooldown_pins():
    base = adaptive_params({"unilateralness_ema_span_1m": 4.25})
    rows = _candidate_parameter_matrix(
        [
            {
                "long_unilateralness_ema_span_1m": 7.00000001,
                "long_entry_cooldown_exposure_weight": 9.0,
            }
        ],
        ADAPTIVE_PARAM_KEYS,
        {"long": base},
        static_overrides={"long": {"entry_cooldown_exposure_weight": 3.0}},
    )
    assert rows[0, ADAPTIVE_PARAM_KEYS.index("unilateralness_window")] == 141
    assert rows[0, ADAPTIVE_PARAM_KEYS.index("entry_cooldown_exposure_weight")] == 3
    assert (
        rows[0, ADAPTIVE_PARAM_KEYS.index("entry_cooldown_max_duration_minutes")] == -1
    )


@lru_cache(maxsize=1)
def _library():
    common = (
        Path(__file__).resolve().parents[2]
        / "passivbot-rust/src/gpu/mps_adaptive_timing.metal"
    ).read_text()
    return compile_shader(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + common
        + r"""
    kernel void probe(constant float* bars, constant float* params,
                     constant int* sizes, device float* scores, device float* durations, uint b [[thread_position_in_grid]]) {
        if (b > 0) return;
        AdaptiveTiming a = load_adaptive_timing(params, 0);
        for (int k = sizes[1]; k < sizes[0]; ++k) {
            update_adaptive_rms(a, bars, k, 0, 1, 0);
            scores[k] = a.score;
            durations[k] = adaptive_duration(a, params[7], params[8], sizes[2] != 0);
        }
    }
    """
    )


@GPU
@pytest.mark.parametrize("span", [1.0, 3.25, 60.0])
@pytest.mark.parametrize(
    "kind", ["trend", "bouncy", "shock_flat", "trend_flat", "tiny_tail"]
)
def test_gpu_rms_matches_finite_rust_window(span, kind):
    import passivbot_rust as pbr

    n = math.ceil(20 * span)
    t = np.arange(3 * n + 43)
    if kind == "trend":
        close = 100 * np.exp(0.001 * t)
    elif kind == "bouncy":
        close = 100 * np.exp(0.02 * np.sin(t / 7) + 0.003 * np.cos(t / 2))
    elif kind == "shock_flat":
        close = np.where(t < n // 2 + 1, 100.0, 120.0)
    elif kind == "trend_flat":
        close = 100 * np.exp(0.001 * np.minimum(t, n + 3))
    else:
        close = np.where(t < n, 100.0, 150.0) + 0.0002 * np.sin(t / 3)
    close = close.astype(np.float32)
    params = np.array([0.0, 90.0, 2.0, 13.0, span, 1.0, n, 5.25, 0.7], np.float32)
    tensors = [
        torch.as_tensor(x, device=gpu_device())
        for x in [close, params, np.array([len(t), 1, 0], np.int32)]
    ]
    scores = torch.full((len(t),), float("nan"), device=gpu_device())
    durations = torch.empty_like(scores)
    _library().probe(*tensors, scores, durations, threads=1)
    synchronize()
    got = scores.cpu().numpy()
    duration = durations.cpu().numpy()
    assert np.isnan(got[1:n]).all()
    indices = sorted(set([n, n + 1, 2 * n, 3 * n, *range(n, len(t), max(1, n // 9))]))
    expected = np.array(
        [
            pbr.calc_signed_unilateralness(
                close[k - n : k + 1].astype(float).tolist(), span
            )
            for k in indices
        ]
    )
    np.testing.assert_allclose(got[indices], expected, atol=3e-4, rtol=3e-4)
    # The long side penalizes negative directionality only; clamp before rounding.
    wanted = np.ceil(np.clip(5.25 + 2 * 0.7 + 13 * np.maximum(-got[indices], 0), 0, 90))
    np.testing.assert_array_equal(duration[indices], wanted)
    if kind in ("shock_flat", "trend_flat"):
        assert got[-1] == 0.0


@GPU
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
@pytest.mark.parametrize("modifier", ["exposure_ratio", "adverse_directionality"])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_gpu_adaptive_cooldown_changes_fills_with_exact_parity(
    side, coin_count, modifier, strategy
):
    from test_gpu_entry_sizing_parity import _fixture, _evaluate

    cfg, _, mss, _, _ = _fixture(side, coin_count, "reentry")
    cfg["live"]["strategy_kind"] = strategy
    for s in ("long", "short"):
        cfg["bot"][s]["strategy"]["ema_anchor"].update(
            base_qty_pct=0.1,
            ema_span_0=2,
            ema_span_1=3,
            offset=0.01,
            offset_psize_weight=0,
            entry_double_down_factor=0.1,
            offset_volatility_1h_weight=0,
            offset_volatility_1m_weight=0,
        )
    count = 301
    t = np.arange(count)
    direction = -1 if side == "long" else 1
    prices = 100 * np.exp(direction * 0.0007 * t + 0.005 * np.sin(t / 3))
    candles = np.zeros((count, coin_count, 4))
    candles[:, :, 2] = prices[:, None]
    candles[:, :, 0] = prices[:, None] * 1.01
    candles[:, :, 1] = prices[:, None] * 0.99
    candles[:, :, 3] = 10.0
    ts = 1_700_000_000_000 + t.astype(np.int64) * 60000
    btc = np.full(count, 50000.0)
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].pop("entry_cooldown_minutes", None)
        cfg["bot"][s]["entry_cooldown"]["base_duration_minutes"] = 0.0
        cfg["bot"][s]["forager"]["unilateralness_ema_span_1m"] = 3.25
    for key in mss:
        if not key.startswith("__"):
            mss[key]["last_valid_index"] = count - 1
    ec = cfg["bot"][side]["entry_cooldown"]
    ec["max_duration_minutes"] = 30.0
    ec["weights_minutes"][modifier] = 20.0
    out, fills = _evaluate(side, (cfg, candles, mss, btc, ts))
    assert len(fills) > 0
    assert out["fill_count"].item() == len(fills)
    ec["weights_minutes"][modifier] = 0.0
    _, baseline = _evaluate(side, (cfg, candles, mss, btc, ts))
    assert fills[:, 0].tolist() != baseline[:, 0].tolist()
    size_key = "short_psize" if side == "short" else "psize"
    assert out[size_key].item() == pytest.approx(
        abs(sum(float(f[9]) for f in fills)), abs=1e-5
    )


@GPU
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_unilateralness_ranking_selects_bouncy_coin(strategy, side):
    from test_gpu_entry_sizing_parity import _fixture, _evaluate

    cfg, _, mss, _, _ = _fixture(side, 2, "initial")
    cfg["live"]["strategy_kind"] = strategy
    count = 101
    t = np.arange(count)
    candles = np.zeros((count, 2, 4))
    candles[:, 0, 2] = 100 * np.exp(0.001 * t)
    candles[:, 1, 2] = 200 * np.exp(0.01 * np.sin(t / 3))
    candles[:, :, 0] = candles[:, :, 2] * 1.002
    candles[:, :, 1] = candles[:, :, 2] * 0.998
    candles[:, :, 3] = 10
    ts = 1_700_000_000_000 + t.astype(np.int64) * 60000
    for s in ("long", "short"):
        bot = cfg["bot"][s]
        bot["risk"].pop("entry_cooldown_minutes", None)
        bot["entry_cooldown"]["base_duration_minutes"] = 0
        bot["forager"].update(
            unilateralness_ema_span_1m=3.25,
            volume_drop_pct=0,
            score_weights=dict(
                volume=0, volatility=0, ema_readiness=0, unilateralness=1
            ),
        )
        if strategy == "ema_anchor":
            bot["strategy"]["ema_anchor"].update(
                base_qty_pct=0.1,
                offset=0,
                ema_span_0=2,
                ema_span_1=3,
                entry_double_down_factor=0,
            )
    cfg["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1)
    for coin in ("BTC", "ETH"):
        mss[coin].update(last_valid_index=count - 1, price_step=0.001)
    out, fills = _evaluate(side, (cfg, candles, mss, np.full(count, 50000.0), ts))
    assert len(fills) > 0, fills
    assert all(f[2] == "ETH" for f in fills)
    assert fills[0][2] == "ETH"
    assert fills[0][0] >= 66
    assert out["fill_count"].item() == len(fills)
    key = "short_psize" if side == "short" else "psize"
    assert out[key].item() == pytest.approx(
        abs(sum(float(f[9]) for f in fills)), abs=1e-5
    )


@GPU
@pytest.mark.parametrize("recent", [False, True])
def test_adaptive_temporal_chunks_preserve_all_outputs(recent):
    from test_gpu_mps import _tm_directional_temporal_fixture
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner
    from optimization.gpu.model import (
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS as keys,
    )

    market, run, data, row, kwargs = _tm_directional_temporal_fixture()
    for key, value in dict(
        entry_cooldown_max_duration_minutes=30,
        entry_cooldown_exposure_weight=7,
        entry_cooldown_adverse_weight=11,
        unilateralness_ema_span_1m=3.25,
        unilateralness_window=65,
    ).items():
        row[keys.index(key)] = value
    matrix = np.asarray([row + row], dtype=float)
    history = dict(history_start_step=73, trade_start_step=113) if recent else {}
    plain = MpsTrailingMartingaleRunner(market, run, data, **kwargs)
    expected = {
        k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in plain.run(matrix, **history).items()
    }
    chunked = MpsTrailingMartingaleRunner(
        market, run, data, max_dispatch_candidate_bars=94, **kwargs
    )
    for key, value in chunked.run(matrix, **history).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                value.cpu(), expected[key], rtol=0, atol=0, equal_nan=True
            )
        else:
            assert value == expected[key]


def test_new_optimizer_dimensions_map_for_both_strategies_and_sides():
    from optimization.backends.gpu_backend import (
        EMA_STRATEGY_BOUND_MAP,
        TRAILING_MARTINGALE_STRATEGY_BOUND_MAP,
    )

    expected = {
        "entry_cooldown_min_duration_minutes": "entry_cooldown_min_duration_minutes",
        "entry_cooldown_max_duration_minutes": "entry_cooldown_max_duration_minutes",
        "entry_cooldown_weights_minutes_exposure_ratio": "entry_cooldown_exposure_weight",
        "entry_cooldown_weights_minutes_adverse_directionality": "entry_cooldown_adverse_weight",
        "unilateralness_ema_span_1m": "unilateralness_ema_span_1m",
        "forager_score_weights_unilateralness": "forager_score_weights_unilateralness",
    }
    for mapping in (EMA_STRATEGY_BOUND_MAP, TRAILING_MARTINGALE_STRATEGY_BOUND_MAP):
        for side in ("long", "short"):
            for source, target in expected.items():
                assert mapping[f"{side}_{source}"] == f"{side}_{target}"


@GPU
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize(
    "base,floor,ceiling,exposure,adverse",
    [(0, 0, 30, 7, 11), (2.1, 4.2, 8.9, 0, 0), (10, 0, 5, 7, 11), (0, 0, -1, 0, 0)],
)
def test_policy_floor_ceiling_and_additive_weights(
    short, base, floor, ceiling, exposure, adverse
):
    close = (100 * np.exp(np.arange(84) * (-0.002 if not short else 0.002))).astype(
        np.float32
    )
    params = np.array(
        [floor, ceiling, exposure, adverse, 3.25, 1, 65, base, 1.7], np.float32
    )
    tensors = [
        torch.as_tensor(x, device=gpu_device())
        for x in (close, params, np.array([len(close), 1, int(short)], np.int32))
    ]
    scores = torch.empty(len(close), device=gpu_device())
    durations = torch.empty_like(scores)
    _library().probe(*tensors, scores, durations, threads=1)
    synchronize()
    signed = scores.cpu().numpy()[65:]
    expected = np.maximum(
        base + exposure * 1.7 + adverse * np.maximum(signed if short else -signed, 0),
        floor,
    )
    if ceiling >= 0:
        expected = np.minimum(expected, ceiling)
    np.testing.assert_array_equal(durations.cpu().numpy()[65:], np.ceil(expected))


@GPU
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cooldown_coin_override_pins_and_null_ceiling(strategy):
    from test_gpu_entry_sizing_parity import _fixture
    from optimization.gpu.service import MpsMulticoinProxy
    from optimization.gpu.model import (
        EMA_ANCHOR_COIN_OVERRIDE_ADAPTIVE_START,
        TRAILING_MARTINGALE_COIN_OVERRIDE_ADAPTIVE_START,
    )

    cfg, candles, mss, btc, ts = _fixture("long", 2, "initial")
    cfg["live"]["strategy_kind"] = strategy
    cfg["bot"]["long"]["entry_cooldown"].update(
        min_duration_minutes=1, max_duration_minutes=30
    )
    cfg["bot"]["long"]["entry_cooldown"]["weights_minutes"]["exposure_ratio"] = 10
    cfg["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "entry_cooldown": dict(
                        min_duration_minutes=0,
                        max_duration_minutes=None,
                        weights_minutes=dict(
                            exposure_ratio=0, adverse_directionality=0
                        ),
                    )
                }
            }
        }
    }
    proxy = MpsMulticoinProxy(
        config=cfg,
        hlcvs=candles,
        mss=mss,
        btc=btc,
        timestamps=ts,
        exchange="bybit",
        batch_size=1,
        needed_metrics={"adg_strategy_eq"},
    )
    start = (
        EMA_ANCHOR_COIN_OVERRIDE_ADAPTIVE_START
        if strategy == "ema_anchor"
        else TRAILING_MARTINGALE_COIN_OVERRIDE_ADAPTIVE_START
    )
    matrix = proxy.runners["long"].coin_overrides.cpu().numpy()
    np.testing.assert_array_equal(matrix[1, start : start + 4], [0, -1, 0, 0])
    assert np.isnan(matrix[0, start : start + 4]).all()


@pytest.mark.parametrize(
    "name",
    [
        "mps_ema_anchor_source_py",
        "mps_trailing_martingale_source_py",
        "mps_ema_anchor_multicoin_source_py",
        "mps_trailing_martingale_multicoin_source_py",
    ],
)
def test_adaptive_source_translates_to_cuda(name):
    import passivbot_rust as pbr
    from optimization.gpu.cuda_kernel import cuda_source

    source = cuda_source(getattr(pbr, name)())
    assert "struct AdaptiveTiming" in source
    assert "adaptive_log_return" in source
    assert "update_adaptive_rms" in source
    assert "PASSIVBOT_ADAPTIVE_TIMING" not in source


@GPU
@pytest.mark.parametrize("side", ["long", "short"])
def test_multicoin_adaptive_checkpoint_preserves_every_output(side):
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner
    from optimization.gpu.model import TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS as keys

    steps = np.arange(1513)
    closes = np.column_stack(
        [100 * (1 + 0.12 * np.sin(steps / 37)), 120 * (1 + 0.08 * np.sin(steps / 11))]
    )
    _, row, run, data = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=len(steps),
        closes=closes,
        requested_start_index=31,
        return_context=True,
    )
    for key, value in dict(
        entry_cooldown_max_duration_minutes=30,
        entry_cooldown_exposure_weight=7,
        entry_cooldown_adverse_weight=11,
        unilateralness_ema_span_1m=3.25,
        unilateralness_window=65,
        forager_score_weights_unilateralness=1,
        n_positions=1,
    ).items():
        row[keys.index(key)] = value
    matrix = np.asarray([row], dtype=float)
    plain = MpsTrailingMartingaleMulticoinRunner(run, data, side=side)
    expected = {
        key: value.cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in plain.run(matrix).items()
    }
    chunked = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, max_dispatch_candidate_bars=94
    )
    for key, value in chunked.run(matrix).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                value.cpu(), expected[key], rtol=0, atol=0, equal_nan=True
            )
        else:
            assert value == expected[key]


@GPU
@pytest.mark.parametrize("strategy,chunked", [("ema_anchor", False), ("trailing_martingale", False), ("trailing_martingale", True)])
@pytest.mark.parametrize("side", ["long", "short"])
def test_ranking_retries_after_new_listing_rms_warmup(strategy, side, chunked, monkeypatch):
    from test_gpu_entry_sizing_parity import _fixture, _evaluate
    if chunked:
        from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner
        original_init = MpsTrailingMartingaleMulticoinRunner.__init__
        def bounded_init(self, *args, **kwargs):
            kwargs["max_dispatch_candidate_bars"] = 180
            original_init(self, *args, **kwargs)
        monkeypatch.setattr(MpsTrailingMartingaleMulticoinRunner, "__init__", bounded_init)

    cfg, _, mss, _, _ = _fixture(side, 2, "initial")
    cfg["live"]["strategy_kind"] = strategy
    coins = ["BTC", "ETH", "SOL"]
    cfg["live"]["approved_coins"] = {s: coins for s in ("long", "short")}
    cfg["backtest"]["coins"] = {"bybit": coins}
    mss["SOL"] = dict(mss["ETH"])
    cfg["backtest"]["dynamic_wel_by_tradability"] = False
    count = 101
    candles = np.full((count, 3, 4), 100.0)
    candles[:, :, 3] = [1.0, 2.0, 10.0]
    # Initialize ranking with BTC/ETH; SOL becomes eligible while its RMS is pending.
    # No price touch/fill changes selection until after SOL's complete window.
    candles[90:, :, 0] = 110.0
    candles[90:, :, 1] = 90.0
    for s in ("long", "short"):
        bot = cfg["bot"][s]
        bot["risk"].pop("entry_cooldown_minutes", None)
        bot["entry_cooldown"]["base_duration_minutes"] = 0.0
        bot["forager"].update(
            unilateralness_ema_span_1m=1.0, volume_drop_pct=0,
            score_weights=dict(volume=1, volatility=0, ema_readiness=0, unilateralness=1),
        )
        if strategy == "ema_anchor":
            bot["strategy"][strategy].update(
                base_qty_pct=0.1, offset=0.05, ema_span_0=2, ema_span_1=3,
                entry_double_down_factor=0,
            )
        else:
            bot["strategy"][strategy]["entry"].update(
                initial_ema_dist=0.05, ema_gate_mode="all",
            )
    cfg["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1)
    for coin in coins:
        mss[coin].update(last_valid_index=count - 1, price_step=0.001)
    mss["SOL"].update(first_valid_index=60, warmup_minutes=1)
    ts = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60000
    out, fills = _evaluate(side, (cfg, candles, mss, np.full(count, 50000.0), ts))
    assert len(fills) > 0
    assert fills[0][2] == "SOL"
    assert int(fills[0][0]) == 90
    assert out["fill_count"].item() == len(fills)


@GPU
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_gpu_dormant_ranking_accepts_five_minute_candles(strategy, side):
    from test_gpu_entry_sizing_parity import _fixture, _evaluate

    cfg, candles, mss, btc, ts = _fixture(side, 2, "initial")
    cfg["live"]["strategy_kind"] = strategy
    cfg["backtest"].update(dynamic_wel_by_tradability=False, candle_interval_minutes=5)
    ts = (ts[0] // 300000) * 300000 + np.arange(len(ts), dtype=np.int64) * 300000
    for s in ("long", "short"):
        cfg["bot"][s]["risk"].pop("entry_cooldown_minutes", None)
        cfg["bot"][s]["entry_cooldown"]["base_duration_minutes"] = 0.0
    cfg["bot"][side]["forager"].update(
        unilateralness_ema_span_1m=100000,
        score_weights=dict(volume=0, volatility=0, ema_readiness=0, unilateralness=1),
    )
    active, fills = _evaluate(side, (cfg, candles, mss, btc, ts))
    cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 0
    disabled, baseline = _evaluate(side, (cfg, candles, mss, btc, ts))
    np.testing.assert_array_equal(fills, baseline)
    for key, value in active.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, disabled[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("count,slots,dynamic,adverse,accept", [
    (2, 2, False, 0, True), (1, 1, True, 0, True),
    (2, 1, False, 0, False), (2, 2, True, 0, False),
    (1, 1, False, 1, False),
])
def test_gpu_rms_interval_demand_keeps_real_consumers(count, slots, dynamic, adverse, accept):
    from optimization.gpu.mps_kernel import _scale_directional_minute_parameters
    from optimization.gpu.model import EMA_ANCHOR_MULTICOIN_PARAM_KEYS as keys
    from tools.gpu_proxy_benchmark import _base_parameter_values

    values = _base_parameter_values()
    values.update(n_positions=slots, forager_score_weights_unilateralness=1,
                  entry_cooldown_adverse_weight=adverse,
                  entry_cooldown_max_duration_minutes=30)
    matrix = np.asarray([[values[key] for key in keys]])
    kwargs = dict(sides=1, interval_minutes=5, ranking_coin_counts=(count,),
                  dynamic_wel_by_tradability=dynamic)
    if accept:
        packed = _scale_directional_minute_parameters(matrix, keys, **kwargs)
        assert packed[0, keys.index("forager_score_weights_unilateralness")] == 0
    else:
        with pytest.raises(ValueError, match="requires one-minute candles"):
            _scale_directional_minute_parameters(matrix, keys, **kwargs)


@GPU
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_fused_dormant_scoring_respects_each_sides_eligible_count(strategy):
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu import model, mps_kernel

    _, row, run, data = _multicoin_exposure_fixture(
        strategy, "long", count=100, interval_minutes=5, return_context=True,
    )
    prefix = "EMA_ANCHOR" if strategy == "ema_anchor" else "TRAILING_MARTINGALE"
    keys = getattr(model, prefix + "_MULTICOIN_PARAM_KEYS")
    cls = (mps_kernel.MpsEmaAnchorMulticoinFusedRunner if strategy == "ema_anchor"
           else mps_kernel.MpsTrailingMartingaleMulticoinFusedRunner)
    short_overrides = np.full((2, getattr(model, prefix + "_COIN_OVERRIDE_COLS")), np.nan)
    short_overrides[1, getattr(model, prefix + "_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN")] = 0
    runner = cls(run, data, short_coin_overrides=short_overrides, dynamic_wel_by_tradability=False)
    assert runner.rms_ranking_coin_counts == (2, 1)
    row[keys.index("n_positions")] = 2
    row[keys.index("forager_score_weights_unilateralness")] = 1
    row[keys.index("unilateralness_ema_span_1m")] = 100000
    matrix = np.asarray([row + row], dtype=float)
    active = {key: value.cpu().clone() for key, value in runner.run(matrix).items()
              if isinstance(value, torch.Tensor)}
    matrix[0, keys.index("forager_score_weights_unilateralness")] = 0
    matrix[0, len(keys) + keys.index("forager_score_weights_unilateralness")] = 0
    disabled = runner.run(matrix)
    for key, value in active.items():
        torch.testing.assert_close(value, disabled[key].cpu(), rtol=0, atol=0, equal_nan=True)


@GPU
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_constant_coin_cooldown_accepts_aggregated_candles_with_exact_parity(strategy, side):
    import copy
    from test_gpu_entry_sizing_parity import _fixture, _evaluate

    cfg, candles, mss, btc, ts = _fixture(side, 2, "reentry")
    cfg["live"]["strategy_kind"] = strategy
    cfg["backtest"].update(dynamic_wel_by_tradability=False, candle_interval_minutes=5)
    ts = (ts[0] // 300000) * 300000 + np.arange(len(ts), dtype=np.int64) * 300000
    # Supply enough trading history for both strategy families after activation.
    candles = np.concatenate((candles, np.repeat(candles[-1:], 94, axis=0)))
    btc = np.full(len(candles), 50000.0)
    ts = ts[0] + np.arange(len(candles), dtype=np.int64) * 300000
    mss["__meta__"].update(data_interval_minutes=5, requested_start_ts=int(ts[0]))
    for coin in ("BTC", "ETH"):
        mss[coin]["last_valid_index"] = len(candles) * 5 - 1
    candles[:, :, 0] = candles[:, :, 2] * 1.001
    candles[:, :, 1] = candles[:, :, 2] * 0.999
    cfg["bot"][side]["strategy"]["ema_anchor"].update(
        ema_span_0=2, ema_span_1=3, offset=0.01, offset_psize_weight=0,
        offset_volatility_1h_weight=0, offset_volatility_1m_weight=0,
    )
    cfg["coin_overrides"] = {coin: {"bot": {side: {"entry_cooldown": {
        "min_duration_minutes": 10.0, "max_duration_minutes": 10.0,
        "weights_minutes": {"adverse_directionality": 7.0},
    }}}} for coin in ("BTC", "ETH")}
    active, fills = _evaluate(side, (cfg, candles, mss, btc, ts))
    baseline_cfg = copy.deepcopy(cfg)
    for override in baseline_cfg["coin_overrides"].values():
        override["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 0.0
    disabled, baseline = _evaluate(side, (baseline_cfg, candles, mss, btc, ts))
    assert len(fills) > 0
    np.testing.assert_array_equal(fills, baseline)
    assert active["fill_count"].item() == len(fills)
    for key, value in active.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, disabled[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("pins,base,floor,ceiling,adverse,accept", [
    ([np.nan, 10, 10, 7, np.nan], 0, 0, -1, 0, True),
    ([np.nan, np.nan, 10, 7, np.nan], 10, 0, -1, 0, True),
    ([np.nan, np.nan, 10, 7, np.nan], 0, 10, -1, 0, True),
    ([np.nan, np.nan, 10, 7, np.nan], 0, 0, -1, 0, False),
    ([0, 0, np.nan, np.nan, np.nan], 10, 10, 10, 7, False),
    ([np.nan, 10, 10, np.nan, np.nan], 0, 0, 30, 7, True),
    ([np.nan, 10, 10, 7, 0], 0, 0, -1, 0, True),
    ([np.nan, 0, 10, 7, 0], 0, 0, -1, 0, True),
])
def test_effective_coin_cooldown_interval_demand(pins, base, floor, ceiling, adverse, accept):
    from optimization.gpu.mps_kernel import _scale_directional_minute_parameters
    from optimization.gpu.model import EMA_ANCHOR_MULTICOIN_PARAM_KEYS as keys
    from tools.gpu_proxy_benchmark import _base_parameter_values

    values = _base_parameter_values()
    values.update(entry_cooldown_minutes=base, entry_cooldown_min_duration_minutes=floor,
                  entry_cooldown_max_duration_minutes=ceiling, entry_cooldown_adverse_weight=adverse,
                  forager_score_weights_unilateralness=0)
    matrix = np.asarray([[values[key] for key in keys]])
    kwargs = dict(sides=1, interval_minutes=5, cooldown_coin_overrides=(np.asarray([pins]),))
    if accept:
        _scale_directional_minute_parameters(matrix, keys, **kwargs)
    else:
        with pytest.raises(ValueError, match="requires one-minute candles"):
            _scale_directional_minute_parameters(matrix, keys, **kwargs)


def test_inherited_coin_cooldown_clamp_validates_every_candidate_and_coin():
    from optimization.gpu.mps_kernel import _scale_directional_minute_parameters
    from optimization.gpu.model import EMA_ANCHOR_MULTICOIN_PARAM_KEYS as keys
    from tools.gpu_proxy_benchmark import _base_parameter_values

    values = _base_parameter_values()
    values.update(entry_cooldown_minutes=10, entry_cooldown_adverse_weight=0,
                  forager_score_weights_unilateralness=0)
    matrix = np.asarray([[values[key] for key in keys]] * 2)
    pins = np.asarray([[np.nan, np.nan, 10, 7, np.nan]])
    kwargs = dict(sides=1, interval_minutes=5, cooldown_coin_overrides=(pins,))
    _scale_directional_minute_parameters(matrix, keys, **kwargs)
    matrix[1, keys.index("entry_cooldown_minutes")] = 0
    with pytest.raises(ValueError, match="requires one-minute candles"):
        _scale_directional_minute_parameters(matrix, keys, **kwargs)
    pins = np.vstack((pins, [np.nan, 0, 30, 7, np.nan]))
    kwargs["cooldown_coin_overrides"] = (pins,)
    with pytest.raises(ValueError, match="requires one-minute candles"):
        _scale_directional_minute_parameters(matrix[:1], keys, **kwargs)


@GPU
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("pin_side", ["long", "short"])
def test_fused_constant_override_cooldown_accepts_aggregated_candles(strategy, pin_side):
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu import model, mps_kernel

    _, row, run, data = _multicoin_exposure_fixture(
        strategy, "long", count=100, interval_minutes=5, return_context=True,
    )
    prefix = "EMA_ANCHOR" if strategy == "ema_anchor" else "TRAILING_MARTINGALE"
    keys = getattr(model, prefix + "_MULTICOIN_PARAM_KEYS")
    cols = getattr(model, prefix + "_COIN_OVERRIDE_COLS")
    start = getattr(model, prefix + "_COIN_OVERRIDE_ADAPTIVE_START")
    overrides = np.full((2, cols), np.nan)
    overrides[:, start:start + 4] = [10, 10, 0, 7]
    cls = (mps_kernel.MpsEmaAnchorMulticoinFusedRunner if strategy == "ema_anchor"
           else mps_kernel.MpsTrailingMartingaleMulticoinFusedRunner)
    kwargs = {pin_side + "_coin_overrides": overrides}
    runner = cls(run, data, dynamic_wel_by_tradability=False, **kwargs)
    matrix = np.asarray([row + row], dtype=float)
    active = {key: value.cpu().clone() for key, value in runner.run(matrix).items()
              if isinstance(value, torch.Tensor)}
    overrides[:, start + 3] = 0
    baseline = cls(run, data, dynamic_wel_by_tradability=False, **kwargs).run(matrix)
    for key, value in active.items():
        torch.testing.assert_close(value, baseline[key].cpu(), rtol=0, atol=0, equal_nan=True)
    # One nonconstant override on either fused side must still require 1m input.
    overrides[1, start] = 0
    overrides[1, start + 3] = 7
    runner = cls(run, data, dynamic_wel_by_tradability=False, **kwargs)
    with pytest.raises(ValueError, match="requires one-minute candles"):
        runner.run(matrix)
