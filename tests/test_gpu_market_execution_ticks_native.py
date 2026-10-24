"""The native market-price transport follows existing CPU directional rounding."""

from decimal import Decimal

import numpy as np
import pytest

import passivbot_rust as pbr


SELL = 0
BUY = 1
EPSILON = np.finfo(np.float64).eps


def prepare(closes, steps, slippage=0.0, first=None, last=None, offset=0):
    coins = closes.shape[1]
    if first is None:
        first = [0] * coins
    if last is None:
        last = [offset + closes.shape[0] - 1] * coins
    return pbr.prepare_gpu_market_execution_ticks(
        closes, steps, slippage, first, last, row_offset_int=offset
    )


def assert_native_prices(closes, steps, slippage, ticks):
    """Compare reconstruction with the existing Rust CPU rounding exports."""
    assert ticks.dtype == np.int32
    assert ticks.shape == (*closes.shape, 2)
    assert ticks.flags.c_contiguous
    for row in range(closes.shape[0]):
        for coin, authored_step in enumerate(steps):
            close = max(float(closes[row, coin]), EPSILON)
            step = max(authored_step, EPSILON)
            expected = (
                max(pbr.round_dn(close * (1.0 - slippage), step), step),
                max(pbr.round_up(close * (1.0 + slippage), step), step),
            )
            for lane in (SELL, BUY):
                count = int(ticks[row, coin, lane])
                assert 1 <= count <= np.iinfo(np.int32).max
                # Canonical decimal multiplication independently reconstructs
                # the existing native price, without duplicating tick recovery.
                restored = float(Decimal(count) * Decimal(str(step)))
                assert restored == expected[lane]


def test_observed_f64_close_preserves_native_sell_tick_instead_of_f32_boundary():
    closes = np.array([[167.84391568013112]], dtype=np.float64)
    ticks = prepare(closes, [0.01], 0.0005)
    assert ticks.tolist() == [[[16775, 16793]]]
    assert pbr.GPU_MARKET_EXECUTION_SELL_LANE == SELL
    assert pbr.GPU_MARKET_EXECUTION_BUY_LANE == BUY
    old_sell = np.floor(
        np.float32(closes[0, 0])
        * (np.float32(1.0) - np.float32(0.0005))
        / np.float32(0.01)
    )
    assert old_sell == 16776
    assert_native_prices(closes, [0.01], 0.0005, ticks)


@pytest.mark.parametrize("slippage", [0.0, 0.0005, 1.0, 2.0])
def test_both_market_directions_match_existing_native_rounding(slippage):
    closes = np.array([[0.3, 167.84391568013112], [0.1, 1.25]], dtype=np.float64)
    steps = [0.1, 0.01]
    assert_native_prices(closes, steps, slippage, prepare(closes, steps, slippage))


def test_adjacent_float_boundary_values_do_not_use_epsilon_snapping():
    closes = np.array(
        [[np.nextafter(0.3, 0.0)], [0.3], [np.nextafter(0.3, 1.0)]],
        dtype=np.float64,
    )
    ticks = prepare(closes, [0.1])
    assert ticks[:, 0].tolist() == [[2, 3], [3, 3], [3, 4]]
    assert_native_prices(closes, [0.1], 0.0, ticks)


@pytest.mark.parametrize(
    "close, step",
    [(1e-30, 0.0), (1e-30, 1e-300), (1e-10, 3e-12), (3e200, 1e200), (1e300, 1e300)],
)
def test_zero_tiny_and_large_steps_keep_the_cpu_effective_step(close, step):
    closes = np.array([[close]], dtype=np.float64)
    assert_native_prices(closes, [step], 0.0, prepare(closes, [step]))


def test_strided_float64_view_and_chunks_match_full_preparation_without_mutation():
    source = np.linspace(0.1, 4.0, 60, dtype=np.float64).reshape(10, 6)
    closes = source[::-2, ::2]
    assert not closes.flags.c_contiguous
    before = source.copy()
    steps = [0.01, 0.02, 0.03]
    expected = prepare(closes, steps, 0.0005)
    chunks = [
        prepare(closes[:2], steps, 0.0005, [0, 0, 0], [4, 4, 4], 0),
        prepare(closes[2:], steps, 0.0005, [0, 0, 0], [4, 4, 4], 2),
    ]
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    np.testing.assert_array_equal(source, before)
    assert_native_prices(closes, steps, 0.0005, expected)


def test_empty_row_chunk_has_the_declared_contiguous_output_shape():
    ticks = prepare(np.empty((0, 2), dtype=np.float64), [0.01, 0.02])
    assert ticks.shape == (0, 2, 2)
    assert ticks.dtype == np.int32
    assert ticks.flags.c_contiguous


def test_only_inactive_global_window_cells_may_have_nan_or_zero_sentinels():
    closes = np.array([[np.nan, 1.0, np.nan], [2.0, 2.0, np.inf], [3.0, np.nan, 0.0]])
    first, last = [11, 10, 0], [12, 11, -1]
    ticks = prepare(closes, [0.01] * 3, first=first, last=last, offset=10)
    assert ticks[0, 0].tolist() == [0, 0]
    assert ticks[2, 1].tolist() == [0, 0]
    assert ticks[:, 2].tolist() == [[0, 0]] * 3
    assert (ticks[1:, 0] > 0).all()
    assert (ticks[:2, 1] > 0).all()
    pieces = [
        prepare(closes[:1], [0.01] * 3, first=first, last=last, offset=10),
        prepare(closes[1:], [0.01] * 3, first=first, last=last, offset=11),
    ]
    np.testing.assert_array_equal(np.concatenate(pieces), ticks)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_active_closes_fail_instead_of_becoming_zero_sentinels(value):
    with pytest.raises(ValueError, match="Active GPU market closes"):
        prepare(np.array([[value]], dtype=np.float64), [0.01])


@pytest.mark.parametrize("step", [-0.01, np.nan, np.inf, -np.inf])
def test_invalid_steps_fail_even_for_an_empty_active_window(step):
    with pytest.raises(ValueError, match="price step"):
        prepare(np.array([[np.nan]], dtype=np.float64), [step], first=[0], last=[-1])


@pytest.mark.parametrize("slippage", [-0.001, np.nan, np.inf, -np.inf])
def test_invalid_slippage_fails_before_cell_preparation(slippage):
    with pytest.raises(ValueError, match="slippage"):
        prepare(np.array([[1.0]], dtype=np.float64), [0.01], slippage)


def test_int32_count_limit_is_exact_and_overflow_never_saturates():
    maximum = int(np.iinfo(np.int32).max)
    close = float(Decimal(maximum) * Decimal("0.01"))
    assert prepare(np.array([[close]], dtype=np.float64), [0.01]).tolist() == [
        [[maximum, maximum]]
    ]
    for close in (float(Decimal(maximum + 1) * Decimal("0.01")), np.finfo(np.float64).max):
        with pytest.raises(ValueError):
            prepare(np.array([[close]], dtype=np.float64), [0.01])
    with pytest.raises(ValueError):
        prepare(np.array([[np.finfo(np.float64).max]], dtype=np.float64), [1.0], 1.0)


def test_vector_shape_dtype_windows_and_chunk_offset_are_validated():
    closes = np.ones((2, 2), dtype=np.float64)
    for steps, first, last in [
        ([0.01], [0, 0], [1, 1]),
        ([0.01] * 2, [0], [1, 1]),
        ([0.01] * 2, [0, 0], [1]),
    ]:
        with pytest.raises(ValueError, match="one step/window per coin"):
            prepare(closes, steps, first=first, last=last)
    with pytest.raises(ValueError, match="nonnegative indices"):
        prepare(closes, [0.01] * 2, first=[-1, 0], last=[1, 1])
    for offset in (-1, np.iinfo(np.int64).max):
        with pytest.raises((ValueError, OverflowError)):
            prepare(closes, [0.01] * 2, offset=int(offset))
    with pytest.raises(OverflowError):
        prepare(closes, [0.01] * 2, offset=1 << 64)
    with pytest.raises(ValueError):
        prepare(np.empty((2, 0), dtype=np.float64), [])
    for malformed in (
        np.ones(2), np.ones((1, 1, 1)), np.ones((1, 1), dtype=np.float32)
    ):
        with pytest.raises((TypeError, ValueError)):
            pbr.prepare_gpu_market_execution_ticks(malformed, [0.01], 0.0, [0], [1])
