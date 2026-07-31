from __future__ import annotations

from dataclasses import replace

import pytest

from outcome.orchestrator import (
    InsufficientCapitalPolicy,
    OutcomeBacktestJob,
    OutcomePostFillAdverseSelection,
    SingleOutcomeBacktestResult,
    run_outcome_portfolio_backtest,
)


def result(
    market_id: str,
    start_ms: int,
    settle_ms: int,
    allocation: float,
    pnl: float,
) -> SingleOutcomeBacktestResult:
    return SingleOutcomeBacktestResult(
        market_id=market_id,
        trading_open_time_ms=start_ms,
        settlement_time_ms=settle_ms,
        starting_collateral=allocation,
        ending_collateral=allocation + pnl,
        orders_placed_count=20,
        fills_count=10,
        maker_fills_count=8,
        traded_notional=20.0,
        trading_fees_paid=0.08,
        settlement_fees_paid=0.02,
        fees_paid=0.1,
        rebates_earned=0.02,
        gross_spread_pnl=pnl + 0.1,
        settlement_pnl=0.0,
        pre_settlement_yes_qty=5.0,
        pre_settlement_no_qty=4.0,
        pre_settlement_paired_qty=4.0,
        pre_settlement_net_yes_exposure=1.0,
        max_paired_qty=5.0,
        max_abs_residual_qty=1.0,
        cumulative_yes_buy_qty=10.0,
        cumulative_no_buy_qty=8.0,
        pair_completion_ratio=0.8,
        time_weighted_abs_residual_qty=0.5,
        time_weighted_total_inventory_qty=6.0,
        worst_case_settlement_equity_min=max(0.0, allocation - 2.0),
        post_fill_adverse_selection=(
            OutcomePostFillAdverseSelection(
                horizon_ms=1_000,
                total_fills_count=10,
                observed_fills_count=10,
                total_fill_qty=10.0,
                observed_fill_qty=10.0,
                fill_qty_coverage_ratio=1.0,
                total_adverse_selection_quote=0.5,
                mean_adverse_selection_per_share=0.05,
            ),
        ),
    )


def job(
    market_id: str,
    start_ms: int,
    settle_ms: int,
    allocation: float,
    pnl: float,
    *,
    capital_release_ms: int | None = None,
) -> OutcomeBacktestJob:
    return OutcomeBacktestJob(
        market_id=market_id,
        trading_open_time_ms=start_ms,
        settlement_time_ms=settle_ms,
        requested_collateral=allocation,
        runner=lambda allocated: result(market_id, start_ms, settle_ms, allocated, pnl),
        capital_release_time_ms=capital_release_ms,
    )


def test_overlapping_markets_cannot_each_receive_the_full_wallet():
    jobs = [
        job("first", 0, 10_000, 100.0, 1.0),
        job("second", 1_000, 9_000, 100.0, 1.0),
    ]
    with pytest.raises(ValueError, match="insufficient shared collateral"):
        run_outcome_portfolio_backtest(jobs, starting_collateral=100.0)


def test_settled_capital_and_profit_are_reused_only_after_settlement():
    portfolio = run_outcome_portfolio_backtest(
        [
            job("first", 0, 5_000, 60.0, 3.0),
            job("overlap-skipped", 1_000, 4_000, 60.0, 100.0),
            job("after-settlement", 5_000, 10_000, 60.0, 2.0),
        ],
        starting_collateral=100.0,
        insufficient_capital_policy=InsufficientCapitalPolicy.SKIP,
    )

    assert [result.market_id for result in portfolio.market_results] == [
        "first",
        "after-settlement",
    ]
    assert [skipped.market_id for skipped in portfolio.skipped_markets] == ["overlap-skipped"]
    assert portfolio.ending_collateral == pytest.approx(105.0)
    assert portfolio.net_pnl == pytest.approx(5.0)
    assert portfolio.max_concurrent_allocated_collateral == pytest.approx(60.0)
    assert portfolio.allocated_collateral_time_ratio == pytest.approx(0.6)
    assert portfolio.fills_count == 20
    assert portfolio.orders_placed_count == 40
    assert portfolio.order_fill_ratio == pytest.approx(0.5)
    assert portfolio.maker_fill_ratio == pytest.approx(0.8)
    assert portfolio.trading_fees_paid == pytest.approx(0.16)
    assert portfolio.settlement_fees_paid == pytest.approx(0.04)
    assert portfolio.fees_paid == pytest.approx(0.2)
    assert portfolio.cumulative_yes_buy_qty == pytest.approx(20.0)
    assert portfolio.cumulative_no_buy_qty == pytest.approx(16.0)
    assert portfolio.pair_completion_ratio == pytest.approx(0.8)
    assert portfolio.max_abs_residual_qty == pytest.approx(1.0)
    assert portfolio.time_weighted_abs_residual_qty == pytest.approx(0.5)
    assert portfolio.time_weighted_total_inventory_qty == pytest.approx(6.0)
    assert len(portfolio.post_fill_adverse_selection) == 1
    markout = portfolio.post_fill_adverse_selection[0]
    assert markout.horizon_ms == 1_000
    assert markout.total_fills_count == 20
    assert markout.total_fill_qty == pytest.approx(20.0)
    assert markout.total_adverse_selection_quote == pytest.approx(1.0)
    assert markout.mean_adverse_selection_per_share == pytest.approx(0.05)


def test_pair_completion_does_not_pair_opposite_sides_across_markets():
    yes_only = replace(
        result("yes-only", 0, 5_000, 50.0, 0.0),
        cumulative_yes_buy_qty=10.0,
        cumulative_no_buy_qty=0.0,
        pair_completion_ratio=0.0,
    )
    no_only = replace(
        result("no-only", 0, 5_000, 50.0, 0.0),
        cumulative_yes_buy_qty=0.0,
        cumulative_no_buy_qty=10.0,
        pair_completion_ratio=0.0,
    )
    portfolio = run_outcome_portfolio_backtest(
        [
            OutcomeBacktestJob(
                market_id=yes_only.market_id,
                trading_open_time_ms=yes_only.trading_open_time_ms,
                settlement_time_ms=yes_only.settlement_time_ms,
                requested_collateral=yes_only.starting_collateral,
                runner=lambda _allocated: yes_only,
            ),
            OutcomeBacktestJob(
                market_id=no_only.market_id,
                trading_open_time_ms=no_only.trading_open_time_ms,
                settlement_time_ms=no_only.settlement_time_ms,
                requested_collateral=no_only.starting_collateral,
                runner=lambda _allocated: no_only,
            ),
        ],
        starting_collateral=100.0,
    )

    assert portfolio.cumulative_yes_buy_qty == pytest.approx(10.0)
    assert portfolio.cumulative_no_buy_qty == pytest.approx(10.0)
    assert portfolio.pair_completion_ratio == pytest.approx(0.0)


def test_same_timestamp_settlement_is_released_before_new_market_allocation():
    portfolio = run_outcome_portfolio_backtest(
        [
            job("first", 0, 5_000, 100.0, 1.0),
            job("second", 5_000, 10_000, 100.0, 1.0),
        ],
        starting_collateral=100.0,
    )
    assert portfolio.ending_collateral == pytest.approx(102.0)


def test_resolved_capital_is_not_reused_before_authoritative_release():
    portfolio = run_outcome_portfolio_backtest(
        [
            job(
                "first",
                0,
                5_000,
                100.0,
                1.0,
                capital_release_ms=6_000,
            ),
            job("before-release", 5_000, 9_000, 100.0, 100.0),
            job("at-release", 6_000, 10_000, 100.0, 1.0),
        ],
        starting_collateral=100.0,
        insufficient_capital_policy=InsufficientCapitalPolicy.SKIP,
    )

    assert [item.market_id for item in portfolio.market_results] == [
        "first",
        "at-release",
    ]
    assert [item.market_id for item in portfolio.skipped_markets] == ["before-release"]
    assert portfolio.ending_collateral == pytest.approx(102.0)


def test_redemption_lag_does_not_dilute_inventory_time_metrics():
    portfolio = run_outcome_portfolio_backtest(
        [
            job(
                "delayed-redemption",
                0,
                5_000,
                100.0,
                1.0,
                capital_release_ms=10_000,
            ),
        ],
        starting_collateral=100.0,
    )

    assert portfolio.allocated_collateral_time_ratio == pytest.approx(1.0)
    assert portfolio.time_weighted_abs_residual_qty == pytest.approx(0.5)
    assert portfolio.time_weighted_total_inventory_qty == pytest.approx(6.0)


def test_overlapping_market_residual_peaks_are_aggregated_chronologically():
    first = replace(
        result("first", 0, 10_000, 40.0, 0.0),
        max_abs_residual_qty=5.0,
        residual_qty_timeline=((1_000, 5.0),),
    )
    second = replace(
        result("second", 500, 9_000, 40.0, 0.0),
        max_abs_residual_qty=5.0,
        residual_qty_timeline=((2_000, -5.0),),
    )
    portfolio = run_outcome_portfolio_backtest(
        [
            OutcomeBacktestJob("first", 0, 10_000, 40.0, lambda _: first),
            OutcomeBacktestJob("second", 500, 9_000, 40.0, lambda _: second),
        ],
        starting_collateral=100.0,
    )

    assert portfolio.max_abs_residual_qty == pytest.approx(10.0)


def test_same_timestamp_residual_changes_are_applied_atomically():
    decreasing = replace(
        result("z-decreasing", 0, 10_000, 40.0, 0.0),
        max_abs_residual_qty=5.0,
        residual_qty_timeline=((1_000, 5.0), (2_000, 0.0)),
    )
    increasing = replace(
        result("a-increasing", 500, 9_000, 40.0, 0.0),
        max_abs_residual_qty=5.0,
        residual_qty_timeline=((2_000, 5.0),),
    )

    portfolio = run_outcome_portfolio_backtest(
        [
            OutcomeBacktestJob(
                "z-decreasing",
                0,
                10_000,
                40.0,
                lambda _: decreasing,
            ),
            OutcomeBacktestJob(
                "a-increasing",
                500,
                9_000,
                40.0,
                lambda _: increasing,
            ),
        ],
        starting_collateral=100.0,
    )

    assert portfolio.max_abs_residual_qty == pytest.approx(5.0)


def test_same_market_intra_timestamp_residual_peak_is_preserved():
    paired_within_second = replace(
        result("paired", 0, 10_000, 40.0, 0.0),
        max_abs_residual_qty=5.0,
        residual_qty_timeline=((2_000, 5.0), (2_000, 0.0)),
    )
    portfolio = run_outcome_portfolio_backtest(
        [
            OutcomeBacktestJob(
                "paired",
                0,
                10_000,
                40.0,
                lambda _: paired_within_second,
            ),
        ],
        starting_collateral=100.0,
    )

    assert portfolio.max_abs_residual_qty == pytest.approx(5.0)


def test_skipped_far_future_market_does_not_extend_portfolio_horizon():
    portfolio = run_outcome_portfolio_backtest(
        [
            job("executed", 0, 10_000, 60.0, 0.0),
            job("skipped", 1_000, 100_000, 60.0, 0.0),
        ],
        starting_collateral=100.0,
        insufficient_capital_policy=InsufficientCapitalPolicy.SKIP,
    )

    assert [item.market_id for item in portfolio.market_results] == ["executed"]
    assert [item.market_id for item in portfolio.skipped_markets] == ["skipped"]
    assert portfolio.allocated_collateral_time_ratio == pytest.approx(0.6)
    assert portfolio.time_weighted_abs_residual_qty == pytest.approx(0.5)
    assert portfolio.time_weighted_total_inventory_qty == pytest.approx(6.0)


def test_all_skipped_portfolio_has_zero_duration_metrics():
    portfolio = run_outcome_portfolio_backtest(
        [job("skipped", 1_000, 100_000, 200.0, 0.0)],
        starting_collateral=100.0,
        insufficient_capital_policy=InsufficientCapitalPolicy.SKIP,
    )

    assert portfolio.market_results == ()
    assert [item.market_id for item in portfolio.skipped_markets] == ["skipped"]
    assert portfolio.ending_collateral == pytest.approx(100.0)
    assert portfolio.allocated_collateral_time_ratio == pytest.approx(0.0)
    assert portfolio.time_weighted_abs_residual_qty == pytest.approx(0.0)
    assert portfolio.time_weighted_total_inventory_qty == pytest.approx(0.0)
    assert portfolio.post_fill_adverse_selection == ()


def test_empty_portfolio_preserves_wallet():
    portfolio = run_outcome_portfolio_backtest([], starting_collateral=100.0)
    assert portfolio.ending_collateral == 100.0
    assert portfolio.market_results == ()
    assert portfolio.pair_completion_ratio == 0.0
