from __future__ import annotations

import pytest

from outcome.evaluation import (
    evaluate_ema_anchor_outcome_modes,
    summarize_outcome_strategy_modes,
)

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover
    pbr = None


pytestmark = pytest.mark.skipif(
    pbr is None or bool(getattr(pbr, "__is_stub__", False)),
    reason="real passivbot_rust extension required",
)


def test_evaluation_runs_modes_and_both_settlement_outcomes():
    strategy_payload = {
        "market": {
            "venue": "fixture",
            "market_id": "binary-1",
            "yes_asset_id": "yes",
            "no_asset_id": "no",
            "payout_unit": 1.0,
            "min_price": 0.001,
            "max_price": 0.999,
            "price_grid": {"kind": "fixed_step", "step": 0.001},
            "qty_step": 0.1,
            "min_qty": 0.1,
            "min_notional": 0.0,
            "trading_opens_ms": 1_000,
            "order_entry_opens_ms": 1_000,
            "trading_closes_ms": 5_000,
            "scheduled_event_ms": 5_000,
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
                "volume": 1.0 if timestamp == 1_000 else 0.0,
            }
            for timestamp in range(1_000, 5_000, 1_000)
        ],
        "execution_candles": [
            {
                "timestamp_ms": 2_000,
                "outcome": "yes",
                "open": 0.5,
                "high": 0.5,
                "low": 0.489,
                "close": 0.5,
                "volume": 0.01,
            },
            {
                "timestamp_ms": 2_000,
                "outcome": "no",
                "open": 0.5,
                "high": 0.511,
                "low": 0.5,
                "close": 0.5,
                "volume": 0.01,
            },
        ],
        "settlement_time_ms": 5_000,
        "yes_fraction": 0.0,
    }
    evaluations = evaluate_ema_anchor_outcome_modes(
        strategy_payload,
        execution_modes=["accumulate_pairs"],
    )

    assert len(evaluations) == 2
    assert {evaluation.yes_fraction for evaluation in evaluations} == {0.0, 1.0}
    assert all(evaluation.net_pnl == pytest.approx(0.02) for evaluation in evaluations)

    summaries = summarize_outcome_strategy_modes(evaluations)
    assert len(summaries) == 1
    assert summaries[0].settlement_cases == 2
    assert summaries[0].settlement_sensitivity == pytest.approx(0.0)
    assert summaries[0].pre_settlement_paired_qty == pytest.approx(1.0)
    assert summaries[0].cumulative_yes_buy_qty == pytest.approx(1.0)
    assert summaries[0].cumulative_no_buy_qty == pytest.approx(1.0)
    assert summaries[0].complementary_buy_qty == pytest.approx(1.0)
    assert summaries[0].pair_completion_ratio == pytest.approx(1.0)
    assert summaries[0].max_paired_qty == pytest.approx(1.0)
    assert summaries[0].max_abs_residual_qty == pytest.approx(0.0)
    assert summaries[0].time_weighted_abs_residual_qty == pytest.approx(0.0)
    assert summaries[0].time_weighted_total_inventory_qty == pytest.approx(1.5)
    assert summaries[0].trading_fees_paid == pytest.approx(0.0)
    assert summaries[0].min_settlement_fees_paid == pytest.approx(0.0)
    assert summaries[0].max_settlement_fees_paid == pytest.approx(0.0)
    assert summaries[0].min_total_fees_paid == pytest.approx(0.0)
    assert summaries[0].max_total_fees_paid == pytest.approx(0.0)
    assert len(summaries[0].post_fill_adverse_selection) == 1
    assert summaries[0].post_fill_adverse_selection[0].horizon_ms == 1_000
    assert (
        summaries[0]
        .post_fill_adverse_selection[0]
        .mean_adverse_selection_per_share
        == pytest.approx(-0.01)
    )

    strategy_payload["fee_schedule"]["maker_rate"] = -0.01
    rebate_evaluations = evaluate_ema_anchor_outcome_modes(
        strategy_payload,
        execution_modes=["accumulate_pairs"],
        settlement_fractions=[0.0],
    )
    assert rebate_evaluations[0].rebates_earned > 0.0
    rebate_summary = summarize_outcome_strategy_modes(rebate_evaluations)[0]
    assert rebate_summary.min_rebates_earned == pytest.approx(
        rebate_evaluations[0].rebates_earned
    )
    assert rebate_summary.max_rebates_earned == pytest.approx(
        rebate_evaluations[0].rebates_earned
    )


def test_mode_summary_exposes_unpaired_inventory_as_settlement_sensitivity():
    strategy_payload = {
        "market": {
            "venue": "fixture",
            "market_id": "binary-1",
            "yes_asset_id": "yes",
            "no_asset_id": "no",
            "payout_unit": 1.0,
            "min_price": 0.001,
            "max_price": 0.999,
            "price_grid": {"kind": "fixed_step", "step": 0.001},
            "qty_step": 0.1,
            "min_qty": 0.1,
            "min_notional": 0.0,
            "trading_opens_ms": 1_000,
            "order_entry_opens_ms": 1_000,
            "trading_closes_ms": 5_000,
            "scheduled_event_ms": 5_000,
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
            "settlement_rate": 0.1,
        },
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
            "execution_mode": "yes_only",
        },
        "signal_candles": [
            {
                "timestamp_ms": timestamp,
                "open": 0.5,
                "high": 0.5,
                "low": 0.5,
                "close": 0.5,
                "volume": 1.0 if timestamp == 1_000 else 0.0,
            }
            for timestamp in range(1_000, 5_000, 1_000)
        ],
        "execution_candles": [
            {
                "timestamp_ms": 2_000,
                "outcome": "yes",
                "open": 0.5,
                "high": 0.5,
                "low": 0.489,
                "close": 0.5,
                "volume": 0.01,
            }
        ],
        "settlement_time_ms": 5_000,
        "yes_fraction": 0.0,
    }

    summary = summarize_outcome_strategy_modes(
        evaluate_ema_anchor_outcome_modes(
            strategy_payload,
            execution_modes=["yes_only"],
        )
    )[0]

    assert summary.pre_settlement_yes_qty == pytest.approx(1.0)
    assert summary.pre_settlement_no_qty == pytest.approx(0.0)
    assert summary.pre_settlement_paired_qty == pytest.approx(0.0)
    assert summary.pre_settlement_net_yes_exposure == pytest.approx(1.0)
    assert summary.cumulative_yes_buy_qty == pytest.approx(1.0)
    assert summary.cumulative_no_buy_qty == pytest.approx(0.0)
    assert summary.complementary_buy_qty == pytest.approx(0.0)
    assert summary.pair_completion_ratio == pytest.approx(0.0)
    assert summary.settlement_sensitivity == pytest.approx(0.9)
    assert summary.max_paired_qty == pytest.approx(0.0)
    assert summary.max_abs_residual_qty == pytest.approx(1.0)
    assert summary.time_weighted_abs_residual_qty == pytest.approx(0.75)
    assert summary.time_weighted_total_inventory_qty == pytest.approx(0.75)
    assert summary.trading_fees_paid == pytest.approx(0.0)
    assert summary.min_settlement_fees_paid == pytest.approx(0.0)
    assert summary.max_settlement_fees_paid == pytest.approx(0.1)
    assert summary.min_total_fees_paid == pytest.approx(0.0)
    assert summary.max_total_fees_paid == pytest.approx(0.1)
