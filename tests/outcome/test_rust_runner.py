from __future__ import annotations

import pytest

from outcome.orchestrator import run_outcome_portfolio_backtest
from outcome.rust_runner import (
    make_rust_ema_anchor_outcome_job,
    make_rust_outcome_job,
    plan_outcome_ema_anchor,
    run_outcome_ema_anchor_backtest,
    run_single_outcome_backtest,
)

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover
    pbr = None


pytestmark = pytest.mark.skipif(
    pbr is None or bool(getattr(pbr, "__is_stub__", False)),
    reason="real passivbot_rust extension required",
)


def payload(market_id: str = "binary-1", start_ms: int = 1_000, settle_ms: int = 5_000) -> dict:
    return {
        "market": {
            "venue": "fixture",
            "market_id": market_id,
            "yes_asset_id": "yes",
            "no_asset_id": "no",
            "payout_unit": 1.0,
            "min_price": 0.001,
            "max_price": 0.999,
            "price_grid": {"kind": "fixed_step", "step": 0.001},
            "qty_step": 0.1,
            "min_qty": 0.1,
            "min_notional": 0.0,
            "trading_opens_ms": start_ms,
            "trading_closes_ms": settle_ms,
            "scheduled_resolution_ms": settle_ms,
            "capabilities": {
                "complementary_books_merged": False,
                "supports_split": True,
                "supports_merge": True,
                "supports_redeem": True,
                "supports_post_only": True,
                "supports_gtd": True,
                "sell_requires_inventory": True,
            },
        },
        "fee_schedule": {
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "formula": "notional",
        },
        "starting_collateral": 10.0,
        "actions": [
            {
                "kind": "place_order",
                "timestamp_ms": start_ms + 500,
                "order": {
                    "order_id": f"{market_id}-yes",
                    "outcome": "yes",
                    "side": "buy",
                    "price": 0.4,
                    "qty": 1.0,
                    "post_only": True,
                    "expires_at_ms": None,
                },
            }
        ],
        "candles": [
            {
                "timestamp_ms": start_ms + 1_000,
                "outcome": "yes",
                "open": 0.42,
                "high": 0.45,
                "low": 0.399,
                "close": 0.42,
                "volume": 0.01,
            }
        ],
        "settlement_time_ms": settle_ms,
        "yes_fraction": 1.0,
    }


def ema_payload(
    market_id: str = "binary-1",
    start_ms: int = 1_000,
    settle_ms: int = 5_000,
) -> dict:
    base = payload(market_id, start_ms, settle_ms)
    return {
        "market": base["market"],
        "fee_schedule": base["fee_schedule"],
        "starting_collateral": 10.0,
        "strategy_params": {
            "ema_span_fast_seconds": 2.0,
            "ema_span_slow_seconds": 4.0,
            "ema_warmup_seconds": 0,
            "quote_offset": 0.01,
            "inventory_skew": 0.0,
            "clip_qty": 1.0,
            "max_total_inventory_qty": 10.0,
            "max_abs_residual_qty": 5.0,
            "min_locked_pair_edge": 0.005,
            "estimated_fee_per_share": 0.0,
            "risk_reduction_only_ms_before_close": 0,
            "entry_cutoff_ms_before_close": 0,
            "execution_mode": "accumulate_pairs",
        },
        "signal_candles": [
            {
                "timestamp_ms": timestamp,
                "open": 0.5,
                "high": 0.5,
                "low": 0.5,
                "close": 0.5,
                "volume": 1.0 if timestamp == start_ms else 0.0,
            }
            for timestamp in range(start_ms, settle_ms, 1_000)
        ],
        "execution_candles": [
            {
                "timestamp_ms": start_ms + 1_000,
                "outcome": "yes",
                "open": 0.5,
                "high": 0.5,
                "low": 0.489,
                "close": 0.5,
                "volume": 0.01,
            },
            {
                "timestamp_ms": start_ms + 1_000,
                "outcome": "no",
                "open": 0.5,
                "high": 0.511,
                "low": 0.5,
                "close": 0.5,
                "volume": 0.01,
            },
        ],
        "settlement_time_ms": settle_ms,
        "yes_fraction": 1.0,
    }


def test_real_rust_binding_runs_strict_cross_and_settlement():
    output = run_single_outcome_backtest(payload())

    assert output["fills_count"] == 1
    assert output["fills"][0]["fill"]["price"] == pytest.approx(0.4)
    assert output["ending_collateral"] == pytest.approx(10.6)
    assert output["fill_model"] == "trade_derived_1s_strict_cross_no_volume_cap"


def test_rust_jobs_compose_through_shared_wallet_orchestrator():
    first = payload("first", 1_000, 5_000)
    second = payload("second", 5_000, 9_000)
    portfolio = run_outcome_portfolio_backtest(
        [make_rust_outcome_job(first), make_rust_outcome_job(second)],
        starting_collateral=10.0,
    )

    assert portfolio.ending_collateral == pytest.approx(11.2)
    assert portfolio.fills_count == 2


def test_real_rust_ema_anchor_outcome_binding_uses_dense_signals_and_trade_candles():
    strategy_payload = ema_payload()
    output = run_outcome_ema_anchor_backtest(strategy_payload)

    assert output["strategy_kind"] == "ema_anchor_outcome"
    assert output["fills_count"] == 2
    assert output["gross_spread_pnl"] == pytest.approx(0.02)
    assert output["settlement_pnl"] == pytest.approx(0.0)
    assert output["pair_completion_ratio"] == pytest.approx(1.0)
    assert output["time_weighted_abs_residual_qty"] == pytest.approx(0.0)
    assert output["time_weighted_total_inventory_qty"] == pytest.approx(1.5)


def test_rust_ema_anchor_jobs_compose_through_shared_wallet_orchestrator():
    portfolio = run_outcome_portfolio_backtest(
        [
            make_rust_ema_anchor_outcome_job(
                ema_payload("first", 1_000, 5_000)
            ),
            make_rust_ema_anchor_outcome_job(
                ema_payload("second", 5_000, 9_000)
            ),
        ],
        starting_collateral=10.0,
    )

    assert [result.market_id for result in portfolio.market_results] == [
        "first",
        "second",
    ]
    assert portfolio.ending_collateral == pytest.approx(10.04)
    assert portfolio.fills_count == 4
    assert portfolio.pair_completion_ratio == pytest.approx(1.0)
    assert portfolio.time_weighted_abs_residual_qty == pytest.approx(0.0)
    assert portfolio.time_weighted_total_inventory_qty == pytest.approx(1.5)


def test_real_rust_live_planner_reconstructs_dense_ema_state():
    base = payload()
    plan = plan_outcome_ema_anchor(
        {
            "market": base["market"],
            "strategy_params": {
                "ema_span_fast_seconds": 2.0,
                "ema_span_slow_seconds": 4.0,
                "ema_warmup_seconds": 3,
                "quote_offset": 0.01,
                "inventory_skew": 0.0,
                "clip_qty": 1.0,
                "max_total_inventory_qty": 10.0,
                "max_abs_residual_qty": 5.0,
                "min_locked_pair_edge": 0.005,
                "estimated_fee_per_share": 0.0,
                "risk_reduction_only_ms_before_close": 0,
                "entry_cutoff_ms_before_close": 0,
                "execution_mode": "accumulate_pairs",
            },
            "observations": [
                {"timestamp_ms": 1_000, "close": 0.50},
                {"timestamp_ms": 2_000, "close": 0.51},
                {"timestamp_ms": 3_000, "close": 0.52},
            ],
            "inventory": {
                "yes_qty": 0.0,
                "no_qty": 0.0,
                "yes_average_cost": 0.0,
                "no_average_cost": 0.0,
                "free_collateral": 10.0,
            },
        }
    )

    assert plan["strategy_kind"] == "ema_anchor_outcome"
    assert plan["observation_count"] == 3
    assert plan["quotes"]["canonical_bid"] is not None
    assert plan["quotes"]["canonical_ask"] is not None
