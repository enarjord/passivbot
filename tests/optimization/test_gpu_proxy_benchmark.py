import sys

import numpy as np
import pytest

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
)
from tools.gpu_proxy_benchmark import (
    HSL_PNL_LOOKBACK_BARS,
    HSL_SIGNAL_MODE_COIN,
    _bounded_positive,
    _fixture_sha256,
    _parameter_matrix,
    _recursive_close_ladder_candidate_count,
    _recursive_entry_ladder_candidate_count,
    _require_mps_torch,
    _single_coin_value_overrides,
    build_parser,
    main,
    run_benchmark_case,
)


@pytest.mark.parametrize(
    "keys",
    (
        EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    ),
)
def test_gpu_proxy_benchmark_candidate_matrix_is_fixed_and_finite(keys):
    first = _parameter_matrix(keys, 8, 7)
    second = _parameter_matrix(keys, 8, 7)
    changed = _parameter_matrix(keys, 8, 8)

    assert first.shape == (8, len(keys))
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert not np.array_equal(first, changed)


def test_gpu_proxy_benchmark_rejects_non_positive_sizes():
    parser = build_parser()

    with pytest.raises(SystemExit):
        _bounded_positive(parser, "--candidates", 0, 10)
    with pytest.raises(SystemExit):
        _bounded_positive(parser, "--candidates", 11, 10)


def test_gpu_proxy_benchmark_accepts_independent_dispatch_batch_size():
    args = build_parser().parse_args(
        ["--candidates", "4096", "--dispatch-batch-size", "512"]
    )

    assert args.candidates == 4096
    assert args.dispatch_batch_size == 512


def test_gpu_proxy_benchmark_exposes_single_side_tm_hsl_case():
    args = build_parser().parse_args(["--case", "tm-single-long-hsl"])
    overrides = _single_coin_value_overrides(args.case)
    matrix = _parameter_matrix(
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
        2,
        7,
        value_overrides=overrides,
    )

    assert args.case == "tm-single-long-hsl"
    assert np.all(
        matrix[:, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("hsl_enabled")]
        == 1.0
    )
    assert np.all(
        matrix[
            :,
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "hsl_red_threshold"
            ),
        ]
        == 0.02
    )
    assert np.all(
        matrix[
            :,
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("hsl_signal_mode"),
        ]
        == HSL_SIGNAL_MODE_COIN
    )
    assert HSL_PNL_LOOKBACK_BARS == 43_200


def test_gpu_proxy_benchmark_exposes_recursive_entry_comparison_case():
    args = build_parser().parse_args(["--case", "tm-single-long-entry-ladder"])
    matrix = _parameter_matrix(
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
        2,
        7,
        value_overrides=_single_coin_value_overrides(args.case),
    )

    assert np.all(
        matrix[
            :,
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "entry_retracement_base_pct"
            ),
        ]
        == 0.0
    )


def test_gpu_proxy_benchmark_counts_only_recursive_entry_ladder_candidates():
    base = {
        "long_entry_retracement_base_pct": 0.0,
        "long_entry_cooldown_minutes": 0.0,
    }

    assert _recursive_entry_ladder_candidate_count(
        [
            base,
            {**base, "long_entry_retracement_base_pct": 0.001},
            {**base, "long_entry_cooldown_minutes": 1.0},
        ]
    ) == 1


@pytest.mark.parametrize(
    ("case", "expected_weight"),
    (
        ("tm-single-long-static-close", 0.0),
        ("tm-single-long-close-ladder", 0.02),
    ),
)
def test_gpu_proxy_benchmark_exposes_recursive_close_comparison_cases(
    case, expected_weight
):
    args = build_parser().parse_args(["--case", case])
    matrix = _parameter_matrix(
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
        2,
        7,
        value_overrides=_single_coin_value_overrides(args.case),
    )

    assert np.all(
        matrix[:, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index("close_qty_pct")]
        == 0.25
    )
    assert np.all(
        matrix[
            :,
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "close_retracement_base_pct"
            ),
        ]
        == 0.0
    )
    assert np.all(
        matrix[
            :,
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
                "close_threshold_we_weight"
            ),
        ]
        == expected_weight
    )


def test_gpu_proxy_benchmark_counts_only_recursive_close_ladder_candidates():
    base = {
        "long_close_qty_pct": 0.25,
        "long_close_retracement_base_pct": 0.0,
        "long_close_threshold_we_weight": 0.02,
    }

    assert _recursive_close_ladder_candidate_count(
        [
            base,
            {**base, "long_close_qty_pct": 1.0},
            {**base, "long_close_retracement_base_pct": 0.001},
            {**base, "long_close_threshold_we_weight": 0.0},
        ]
    ) == 1


@pytest.mark.parametrize(
    "case, extra_args",
    (
        ("ema-single-long", ["--single-bars", "525600"]),
        (
            "ema-multicoin-overhead",
            ["--multicoin-bars", "100000", "--coins", "32"],
        ),
    ),
)
def test_gpu_proxy_benchmark_applies_safety_limit_per_dispatch(
    monkeypatch, case, extra_args
):
    class FakeTorch:
        __version__ = "test"

    monkeypatch.setattr(
        "tools.gpu_proxy_benchmark._require_mps_torch", lambda _parser: FakeTorch()
    )
    monkeypatch.setattr(
        "tools.gpu_proxy_benchmark.run_benchmark_case",
        lambda *_args, **_kwargs: {},
    )

    assert (
        main(
            [
                "--case",
                case,
                "--candidates",
                "4096",
                "--dispatch-batch-size",
                "1",
                "--warm-runs",
                "1",
                *extra_args,
                "--compact",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "case, extra_args",
    (
        ("ema-single-long", ["--single-bars", "525600"]),
        (
            "ema-multicoin-overhead",
            ["--multicoin-bars", "100000", "--coins", "32"],
        ),
    ),
)
def test_gpu_proxy_benchmark_rejects_oversized_dispatch(case, extra_args):
    with pytest.raises(SystemExit):
        main(
            [
                "--case",
                case,
                "--candidates",
                "4096",
                "--dispatch-batch-size",
                "4096",
                "--warm-runs",
                "1",
                *extra_args,
                "--compact",
            ]
        )


def test_gpu_proxy_benchmark_reports_profiled_dispatch_chunks(monkeypatch):
    class FakeProxy:
        runner = type("Runner", (), {"pnl_lookback_bars": 43_200})()

        def evaluate(self, _candidates):
            self.last_profile = {
                "actual_dispatch_batch_sizes": [2, 2, 2, 2],
                "cold_dispatch_count": 0,
                "dispatch_chunk_count": 4,
                "dispatch_count": 4,
                "dispatch_chunk_wall_seconds": [0.4, 0.3, 0.2, 0.1],
                "kernel_candidate_bars": 80,
                "timings_seconds": {
                    "cold_compilation": 0.0,
                    "device_to_host": 0.1,
                    "host_overhead": 0.2,
                    "kernel_execution": 0.3,
                    "warm_library_lookup": 0.0,
                },
            }

    monkeypatch.setattr(
        "tools.gpu_proxy_benchmark._build_case",
        lambda *_args, **_kwargs: (FakeProxy(), [{}] * 8, 10, 1, 1, "fixture"),
    )

    report = run_benchmark_case(
        "ema-single-long",
        candidates=8,
        dispatch_batch_size=2,
        warm_runs=1,
        single_bars=10,
        multicoin_bars=10,
        coins=2,
        seed=7,
    )

    assert report["actual_dispatch_batch_size"] == 2
    assert report["dispatch_chunk_count"] == 4
    assert report["dispatch_count_per_run"] == 4
    assert report["dispatch_chunk_wall_seconds_max"] == pytest.approx(0.4)
    assert report["warm"]["dispatch_chunk_wall_seconds_max_p50"] == pytest.approx(
        0.4
    )
    assert report["kernel_candidate_bars"] == 80
    assert report["hsl_pnl_lookback_bars"] == 43_200
    assert report["hsl_signal_mode"] == 0.0
    assert report["recursive_close_ladder_candidate_count"] == 0


def test_gpu_proxy_benchmark_reports_missing_optional_gpu_dependencies(
    monkeypatch, capsys
):
    parser = build_parser()
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(SystemExit) as exc:
        _require_mps_torch(parser)

    assert exc.value.code == 2
    assert ".[full,gpu-mps]" in capsys.readouterr().err


def test_gpu_proxy_benchmark_fixture_hash_covers_shape_dtype_and_values():
    baseline = np.asarray([[1.0, 2.0]], dtype=np.float64)

    assert _fixture_sha256(baseline) == _fixture_sha256(baseline.copy())
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline.astype(np.float32))
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline.reshape(2, 1))
    assert _fixture_sha256(baseline) != _fixture_sha256(baseline + 1.0)
