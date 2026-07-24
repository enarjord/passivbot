from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from outcome.candles import (
    VerifiedCoverage,
    trades_to_1s_candles_by_native_book,
    trades_to_canonical_signal_1s_candles,
)
from outcome.models import NormalizedOutcomeTrade, OutcomePriceGridChange


def _signal_candle_payload(candle: Any) -> dict[str, Any]:
    return {
        "timestamp_ms": candle.timestamp_ms,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


def build_trade_derived_ema_anchor_input(
    *,
    market_spec: Mapping[str, Any],
    trades: Iterable[NormalizedOutcomeTrade],
    verified_coverage: Sequence[VerifiedCoverage],
    fee_schedule: Mapping[str, Any],
    starting_collateral: float,
    strategy_params: Mapping[str, Any],
    settlement_time_ms: int,
    yes_fraction: float,
    price_grid_changes: Iterable[OutcomePriceGridChange] = (),
) -> dict[str, Any]:
    """Build a Rust strategy input from verified actual fills.

    Signal and execution candles share the same one-second source contract. Execution candles
    retain their native YES/NO book and omit carried zero-volume seconds because those cannot
    fill an order.
    """

    trade_list = list(trades)
    if not trade_list:
        raise ValueError("outcome EMA-anchor backtest input requires actual fills")
    if not verified_coverage:
        raise ValueError("outcome EMA-anchor backtest input requires verified coverage")
    venue_market_keys = {(trade.venue.value, trade.market_id) for trade in trade_list}
    if len(venue_market_keys) != 1:
        raise ValueError("outcome EMA-anchor backtest input requires one venue and market")
    if any(
        not any(
            interval.start_ms <= trade.exchange_time_ms < interval.end_ms
            for interval in verified_coverage
        )
        for trade in trade_list
    ):
        raise ValueError("outcome trade falls outside verified coverage")

    signal_candles = trades_to_canonical_signal_1s_candles(
        trade_list,
        verified_coverage=verified_coverage,
    )
    if not signal_candles:
        raise ValueError("outcome fills produced no canonical signal candles")
    execution_by_book = trades_to_1s_candles_by_native_book(
        trade_list,
        verified_coverage=verified_coverage,
    )
    execution_candles = [
        {
            **_signal_candle_payload(candle),
            "outcome": outcome.value,
        }
        for outcome, candles in execution_by_book.items()
        for candle in candles
        if candle.volume > 0.0
    ]
    execution_candles.sort(key=lambda candle: (candle["timestamp_ms"], candle["outcome"]))
    signal_times = {candle.timestamp_ms for candle in signal_candles}
    if any(candle["timestamp_ms"] not in signal_times for candle in execution_candles):
        raise ValueError("execution candle has no corresponding canonical signal second")

    grid_changes = [
        {
            "timestamp_ms": change.timestamp_ms,
            "old_grid": {
                "kind": change.old_grid.kind,
                **(
                    {"step": change.old_grid.fixed_step}
                    if change.old_grid.kind == "fixed_step"
                    else {
                        "max_significant_figures": change.old_grid.max_significant_figures,
                        "max_decimal_places": change.old_grid.max_decimal_places,
                    }
                ),
            },
            "new_grid": {
                "kind": change.new_grid.kind,
                **(
                    {"step": change.new_grid.fixed_step}
                    if change.new_grid.kind == "fixed_step"
                    else {
                        "max_significant_figures": change.new_grid.max_significant_figures,
                        "max_decimal_places": change.new_grid.max_decimal_places,
                    }
                ),
            },
        }
        for change in sorted(price_grid_changes, key=lambda item: item.timestamp_ms)
    ]
    return {
        "market": dict(market_spec),
        "fee_schedule": dict(fee_schedule),
        "starting_collateral": float(starting_collateral),
        "strategy_params": dict(strategy_params),
        "signal_candles": [
            _signal_candle_payload(candle) for candle in signal_candles
        ],
        "execution_candles": execution_candles,
        "price_grid_changes": grid_changes,
        "settlement_time_ms": int(settlement_time_ms),
        "yes_fraction": float(yes_fraction),
    }
