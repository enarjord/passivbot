from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
from itertools import groupby
import math
from typing import Callable, Iterable


class InsufficientCapitalPolicy(str, Enum):
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class OutcomePostFillAdverseSelection:
    horizon_ms: int
    total_fills_count: int
    observed_fills_count: int
    total_fill_qty: float
    observed_fill_qty: float
    fill_qty_coverage_ratio: float | None
    total_adverse_selection_quote: float | None
    mean_adverse_selection_per_share: float | None

    def __post_init__(self) -> None:
        if self.horizon_ms <= 0:
            raise ValueError("post-fill adverse-selection horizon must be positive")
        if (
            self.total_fills_count < 0
            or not 0 <= self.observed_fills_count <= self.total_fills_count
        ):
            raise ValueError("invalid post-fill adverse-selection fill counts")
        if (
            not math.isfinite(self.total_fill_qty)
            or self.total_fill_qty < 0.0
            or not math.isfinite(self.observed_fill_qty)
            or self.observed_fill_qty < 0.0
            or self.observed_fill_qty > self.total_fill_qty + 1e-9
        ):
            raise ValueError("invalid post-fill adverse-selection quantities")
        if (self.total_fills_count == 0) != (self.total_fill_qty == 0.0):
            raise ValueError("post-fill adverse-selection fill count and quantity disagree")
        if self.total_fill_qty > 0.0:
            if (
                self.fill_qty_coverage_ratio is None
                or not math.isfinite(self.fill_qty_coverage_ratio)
                or abs(
                    self.fill_qty_coverage_ratio
                    - self.observed_fill_qty / self.total_fill_qty
                )
                > 1e-9
            ):
                raise ValueError("post-fill adverse-selection coverage ratio disagrees")
        elif self.fill_qty_coverage_ratio is not None:
            raise ValueError("empty post-fill adverse selection must not invent coverage")
        if self.observed_fill_qty > 0.0:
            if (
                self.total_adverse_selection_quote is None
                or self.mean_adverse_selection_per_share is None
                or not math.isfinite(self.total_adverse_selection_quote)
                or not math.isfinite(self.mean_adverse_selection_per_share)
                or abs(
                    self.mean_adverse_selection_per_share
                    - self.total_adverse_selection_quote / self.observed_fill_qty
                )
                > 1e-9
            ):
                raise ValueError("post-fill adverse-selection markout values disagree")
        elif (
            self.total_adverse_selection_quote is not None
            or self.mean_adverse_selection_per_share is not None
        ):
            raise ValueError("unobserved post-fill adverse selection must remain unavailable")


@dataclass(frozen=True)
class SingleOutcomeBacktestResult:
    market_id: str
    trading_open_time_ms: int
    settlement_time_ms: int
    starting_collateral: float
    ending_collateral: float
    orders_placed_count: int
    fills_count: int
    maker_fills_count: int
    traded_notional: float
    trading_fees_paid: float
    settlement_fees_paid: float
    fees_paid: float
    rebates_earned: float
    gross_spread_pnl: float
    settlement_pnl: float
    pre_settlement_yes_qty: float
    pre_settlement_no_qty: float
    pre_settlement_paired_qty: float
    pre_settlement_net_yes_exposure: float
    max_paired_qty: float
    max_abs_residual_qty: float
    cumulative_yes_buy_qty: float
    cumulative_no_buy_qty: float
    pair_completion_ratio: float
    time_weighted_abs_residual_qty: float
    time_weighted_total_inventory_qty: float
    worst_case_settlement_equity_min: float
    post_fill_adverse_selection: tuple[OutcomePostFillAdverseSelection, ...]
    residual_qty_timeline: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError("single outcome result market_id must not be empty")
        if self.trading_open_time_ms < 0 or self.settlement_time_ms <= self.trading_open_time_ms:
            raise ValueError("single outcome result must have an ordered non-empty lifecycle")
        for name in (
            "starting_collateral",
            "ending_collateral",
            "traded_notional",
            "trading_fees_paid",
            "settlement_fees_paid",
            "fees_paid",
            "rebates_earned",
            "pre_settlement_yes_qty",
            "pre_settlement_no_qty",
            "pre_settlement_paired_qty",
            "max_paired_qty",
            "max_abs_residual_qty",
            "cumulative_yes_buy_qty",
            "cumulative_no_buy_qty",
            "pair_completion_ratio",
            "time_weighted_abs_residual_qty",
            "time_weighted_total_inventory_qty",
            "worst_case_settlement_equity_min",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if abs(
            self.fees_paid - self.trading_fees_paid - self.settlement_fees_paid
        ) > 1e-9:
            raise ValueError("total fees disagree with trading plus settlement fees")
        for name in (
            "gross_spread_pnl",
            "settlement_pnl",
            "pre_settlement_net_yes_exposure",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.pair_completion_ratio > 1.0:
            raise ValueError("pair_completion_ratio must be in [0, 1]")
        larger_buy_qty = max(self.cumulative_yes_buy_qty, self.cumulative_no_buy_qty)
        expected_completion = (
            min(self.cumulative_yes_buy_qty, self.cumulative_no_buy_qty)
            / larger_buy_qty
            if larger_buy_qty > 0.0
            else 0.0
        )
        if abs(self.pair_completion_ratio - expected_completion) > 1e-9:
            raise ValueError("pair_completion_ratio disagrees with cumulative buys")
        if self.pre_settlement_paired_qty > min(
            self.pre_settlement_yes_qty,
            self.pre_settlement_no_qty,
        ) + 1e-9:
            raise ValueError("pre-settlement paired quantity exceeds inventory")
        if self.orders_placed_count < 0:
            raise ValueError("orders_placed_count must be non-negative")
        if self.fills_count < 0 or not 0 <= self.maker_fills_count <= self.fills_count:
            raise ValueError("invalid fill counts")
        horizons = [metric.horizon_ms for metric in self.post_fill_adverse_selection]
        if len(horizons) != len(set(horizons)):
            raise ValueError("post-fill adverse-selection horizons must be unique")
        if any(
            metric.total_fills_count != self.fills_count
            for metric in self.post_fill_adverse_selection
        ):
            raise ValueError("post-fill adverse-selection fill count disagrees with result")
        previous_time = self.trading_open_time_ms
        for timestamp_ms, residual_qty in self.residual_qty_timeline:
            if (
                timestamp_ms < previous_time
                or timestamp_ms >= self.settlement_time_ms
                or not math.isfinite(residual_qty)
            ):
                raise ValueError("invalid single-outcome residual quantity timeline")
            previous_time = timestamp_ms

    @property
    def net_pnl(self) -> float:
        return self.ending_collateral - self.starting_collateral


OutcomeRunner = Callable[[float], SingleOutcomeBacktestResult]


def _portfolio_max_abs_residual_qty(
    results: Iterable[SingleOutcomeBacktestResult],
) -> float:
    """Sweep per-market residual states on one chronological portfolio timeline."""

    events: list[tuple[int, int, str, int, float]] = []
    for result in results:
        timeline = result.residual_qty_timeline
        if not timeline and result.max_abs_residual_qty > 0.0:
            # Hand-written or legacy runners without emitted fill history remain conservative:
            # treat their reported peak as live for the whole allocation interval.
            timeline = ((result.trading_open_time_ms, result.max_abs_residual_qty),)
        for sequence, (timestamp_ms, residual_qty) in enumerate(timeline):
            events.append((timestamp_ms, 1, result.market_id, sequence, residual_qty))
        # Release settled inventory before any market opening at the same timestamp.
        events.append((result.settlement_time_ms, 0, result.market_id, 0, 0.0))

    current: dict[str, float] = {}
    peak = 0.0
    for _timestamp_ms, timestamp_events in groupby(
        sorted(events),
        key=lambda event: event[0],
    ):
        grouped_events = tuple(timestamp_events)
        final_state = dict(current)
        residual_events: dict[str, list[float]] = {}
        for _event_time, priority, market_id, _sequence, residual_qty in grouped_events:
            final_state[market_id] = residual_qty
            if priority != 0:
                residual_events.setdefault(market_id, []).append(residual_qty)
        # Cross-market changes within one timestamp remain atomic because their relative
        # ordering is unknowable. Within each market, however, Rust emits a deterministic fill
        # order whose intermediate residual exposure must contribute to the portfolio peak.
        final_abs_total = sum(abs(value) for value in final_state.values())
        peak = max(peak, final_abs_total)
        for market_id, market_residuals in residual_events.items():
            other_final_abs = final_abs_total - abs(final_state[market_id])
            peak = max(
                peak,
                *(other_final_abs + abs(residual) for residual in market_residuals),
            )
        current = final_state
    return peak


def _portfolio_post_fill_adverse_selection(
    results: Iterable[SingleOutcomeBacktestResult],
) -> tuple[OutcomePostFillAdverseSelection, ...]:
    result_list = tuple(results)
    if not result_list:
        return ()
    expected_horizons = {
        metric.horizon_ms for metric in result_list[0].post_fill_adverse_selection
    }
    for result in result_list[1:]:
        if {
            metric.horizon_ms for metric in result.post_fill_adverse_selection
        } != expected_horizons:
            raise ValueError(
                "single-outcome results use different adverse-selection horizons"
            )
    aggregated = []
    for horizon_ms in sorted(expected_horizons):
        metrics = [
            next(
                metric
                for metric in result.post_fill_adverse_selection
                if metric.horizon_ms == horizon_ms
            )
            for result in result_list
        ]
        total_fills_count = sum(metric.total_fills_count for metric in metrics)
        observed_fills_count = sum(metric.observed_fills_count for metric in metrics)
        total_fill_qty = sum(metric.total_fill_qty for metric in metrics)
        observed_fill_qty = sum(metric.observed_fill_qty for metric in metrics)
        total_adverse_selection_quote = (
            sum(
                metric.total_adverse_selection_quote
                for metric in metrics
                if metric.total_adverse_selection_quote is not None
            )
            if observed_fill_qty > 0.0
            else None
        )
        aggregated.append(
            OutcomePostFillAdverseSelection(
                horizon_ms=horizon_ms,
                total_fills_count=total_fills_count,
                observed_fills_count=observed_fills_count,
                total_fill_qty=total_fill_qty,
                observed_fill_qty=observed_fill_qty,
                fill_qty_coverage_ratio=(
                    observed_fill_qty / total_fill_qty
                    if total_fill_qty > 0.0
                    else None
                ),
                total_adverse_selection_quote=total_adverse_selection_quote,
                mean_adverse_selection_per_share=(
                    total_adverse_selection_quote / observed_fill_qty
                    if total_adverse_selection_quote is not None
                    else None
                ),
            )
        )
    return tuple(aggregated)


@dataclass(frozen=True)
class OutcomeBacktestJob:
    market_id: str
    trading_open_time_ms: int
    settlement_time_ms: int
    requested_collateral: float
    runner: OutcomeRunner
    capital_release_time_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError("outcome job market_id must not be empty")
        if self.trading_open_time_ms < 0 or self.settlement_time_ms <= self.trading_open_time_ms:
            raise ValueError("outcome job must have an ordered non-empty lifecycle")
        if (
            self.capital_release_time_ms is not None
            and self.capital_release_time_ms < self.settlement_time_ms
        ):
            raise ValueError("outcome job capital release cannot precede settlement")
        if not math.isfinite(self.requested_collateral) or self.requested_collateral <= 0.0:
            raise ValueError("requested_collateral must be finite and positive")

    @property
    def effective_capital_release_time_ms(self) -> int:
        if self.capital_release_time_ms is None:
            return self.settlement_time_ms
        return self.capital_release_time_ms


@dataclass(frozen=True)
class SkippedOutcomeBacktest:
    market_id: str
    trading_open_time_ms: int
    requested_collateral: float
    available_collateral: float
    reason: str


@dataclass(frozen=True)
class OutcomePortfolioBacktestResult:
    starting_collateral: float
    ending_collateral: float
    market_results: tuple[SingleOutcomeBacktestResult, ...]
    skipped_markets: tuple[SkippedOutcomeBacktest, ...]
    fills_count: int
    orders_placed_count: int
    maker_fills_count: int
    traded_notional: float
    trading_fees_paid: float
    settlement_fees_paid: float
    fees_paid: float
    rebates_earned: float
    gross_spread_pnl: float
    settlement_pnl: float
    max_concurrent_allocated_collateral: float
    allocated_collateral_time_ratio: float
    cumulative_yes_buy_qty: float
    cumulative_no_buy_qty: float
    pair_completion_ratio: float
    max_abs_residual_qty: float
    time_weighted_abs_residual_qty: float
    time_weighted_total_inventory_qty: float
    worst_case_settlement_equity_min: float
    post_fill_adverse_selection: tuple[OutcomePostFillAdverseSelection, ...]

    @property
    def net_pnl(self) -> float:
        return self.ending_collateral - self.starting_collateral

    @property
    def maker_fill_ratio(self) -> float:
        return self.maker_fills_count / self.fills_count if self.fills_count else 0.0

    @property
    def order_fill_ratio(self) -> float:
        return self.fills_count / self.orders_placed_count if self.orders_placed_count else 0.0


def run_outcome_portfolio_backtest(
    jobs: Iterable[OutcomeBacktestJob],
    *,
    starting_collateral: float,
    insufficient_capital_policy: InsufficientCapitalPolicy = InsufficientCapitalPolicy.FAIL,
) -> OutcomePortfolioBacktestResult:
    """Compose settled single-market runs against one conservative shared wallet.

    Each accepted job receives a fixed allocation that stays unavailable until its authoritative
    capital-release time. The orchestrator never lends another market unused intra-run collateral.
    """

    if not math.isfinite(starting_collateral) or starting_collateral <= 0.0:
        raise ValueError("portfolio starting_collateral must be finite and positive")
    ordered_jobs = sorted(
        jobs,
        key=lambda job: (
            job.trading_open_time_ms,
            job.effective_capital_release_time_ms,
            job.market_id,
        ),
    )
    if len({job.market_id for job in ordered_jobs}) != len(ordered_jobs):
        raise ValueError("outcome portfolio jobs must have unique market IDs")
    if not ordered_jobs:
        return OutcomePortfolioBacktestResult(
            starting_collateral=starting_collateral,
            ending_collateral=starting_collateral,
            market_results=(),
            skipped_markets=(),
            orders_placed_count=0,
            fills_count=0,
            maker_fills_count=0,
            traded_notional=0.0,
            trading_fees_paid=0.0,
            settlement_fees_paid=0.0,
            fees_paid=0.0,
            rebates_earned=0.0,
            gross_spread_pnl=0.0,
            settlement_pnl=0.0,
            max_concurrent_allocated_collateral=0.0,
            allocated_collateral_time_ratio=0.0,
            cumulative_yes_buy_qty=0.0,
            cumulative_no_buy_qty=0.0,
            pair_completion_ratio=0.0,
            max_abs_residual_qty=0.0,
            time_weighted_abs_residual_qty=0.0,
            time_weighted_total_inventory_qty=0.0,
            worst_case_settlement_equity_min=starting_collateral,
            post_fill_adverse_selection=(),
        )

    free_collateral = starting_collateral
    allocated_collateral = 0.0
    max_allocated = 0.0
    allocation_time_area = 0.0
    last_event_time_ms = ordered_jobs[0].trading_open_time_ms
    pending_releases: list[tuple[int, str]] = []
    active_results: dict[str, SingleOutcomeBacktestResult] = {}
    accepted_release_times: dict[str, int] = {}
    results: list[SingleOutcomeBacktestResult] = []
    skipped: list[SkippedOutcomeBacktest] = []
    worst_case_equity_min = starting_collateral

    def advance_time(timestamp_ms: int) -> None:
        nonlocal allocation_time_area, last_event_time_ms
        nonlocal allocated_collateral, free_collateral, worst_case_equity_min
        if timestamp_ms < last_event_time_ms:
            raise AssertionError("portfolio event time moved backwards")
        while pending_releases and pending_releases[0][0] <= timestamp_ms:
            release_time_ms, market_id = heapq.heappop(pending_releases)
            allocation_time_area += allocated_collateral * (
                release_time_ms - last_event_time_ms
            )
            last_event_time_ms = release_time_ms
            settled = active_results.pop(market_id)
            allocated_collateral -= settled.starting_collateral
            free_collateral += settled.ending_collateral
            worst_case_equity_min = min(
                worst_case_equity_min,
                free_collateral
                + sum(
                    result.worst_case_settlement_equity_min
                    for result in active_results.values()
                ),
            )
        allocation_time_area += allocated_collateral * (timestamp_ms - last_event_time_ms)
        last_event_time_ms = timestamp_ms

    for job in ordered_jobs:
        advance_time(job.trading_open_time_ms)
        if job.requested_collateral > free_collateral + 1e-12:
            if insufficient_capital_policy is InsufficientCapitalPolicy.FAIL:
                raise ValueError(
                    f"insufficient shared collateral for {job.market_id}: "
                    f"requested {job.requested_collateral}, available {free_collateral}"
                )
            skipped.append(
                SkippedOutcomeBacktest(
                    market_id=job.market_id,
                    trading_open_time_ms=job.trading_open_time_ms,
                    requested_collateral=job.requested_collateral,
                    available_collateral=free_collateral,
                    reason="insufficient_shared_collateral",
                )
            )
            continue

        result = job.runner(job.requested_collateral)
        if result.market_id != job.market_id:
            raise ValueError("single-market runner returned the wrong market_id")
        if (
            result.trading_open_time_ms != job.trading_open_time_ms
            or result.settlement_time_ms != job.settlement_time_ms
        ):
            raise ValueError(
                "single-market runner returned lifecycle timestamps that differ from its job"
            )
        if abs(result.starting_collateral - job.requested_collateral) > 1e-9:
            raise ValueError("single-market runner changed its allocated starting collateral")

        free_collateral -= result.starting_collateral
        allocated_collateral += result.starting_collateral
        max_allocated = max(max_allocated, allocated_collateral)
        results.append(result)
        active_results[result.market_id] = result
        accepted_release_times[result.market_id] = (
            job.effective_capital_release_time_ms
        )
        heapq.heappush(
            pending_releases,
            (job.effective_capital_release_time_ms, result.market_id),
        )
        worst_case_equity_min = min(
            worst_case_equity_min,
            free_collateral
            + sum(
                active.worst_case_settlement_equity_min
                for active in active_results.values()
            ),
        )

    if not results:
        return OutcomePortfolioBacktestResult(
            starting_collateral=starting_collateral,
            ending_collateral=starting_collateral,
            market_results=(),
            skipped_markets=tuple(skipped),
            orders_placed_count=0,
            fills_count=0,
            maker_fills_count=0,
            traded_notional=0.0,
            trading_fees_paid=0.0,
            settlement_fees_paid=0.0,
            fees_paid=0.0,
            rebates_earned=0.0,
            gross_spread_pnl=0.0,
            settlement_pnl=0.0,
            max_concurrent_allocated_collateral=0.0,
            allocated_collateral_time_ratio=0.0,
            cumulative_yes_buy_qty=0.0,
            cumulative_no_buy_qty=0.0,
            pair_completion_ratio=0.0,
            max_abs_residual_qty=0.0,
            time_weighted_abs_residual_qty=0.0,
            time_weighted_total_inventory_qty=0.0,
            worst_case_settlement_equity_min=starting_collateral,
            post_fill_adverse_selection=(),
        )

    capital_release_horizon_ms = max(accepted_release_times.values())
    if last_event_time_ms < capital_release_horizon_ms:
        advance_time(capital_release_horizon_ms)
    if pending_releases:
        raise AssertionError("portfolio capital-release queue was not fully released")
    if abs(allocated_collateral) > 1e-9:
        raise AssertionError("portfolio retained allocated collateral after final settlement")

    portfolio_start_time_ms = min(result.trading_open_time_ms for result in results)
    capital_release_duration_ms = capital_release_horizon_ms - portfolio_start_time_ms
    inventory_horizon_ms = max(result.settlement_time_ms for result in results)
    inventory_duration_ms = inventory_horizon_ms - portfolio_start_time_ms
    utilization = (
        allocation_time_area / (starting_collateral * capital_release_duration_ms)
        if capital_release_duration_ms > 0
        else 0.0
    )
    cumulative_yes_buy_qty = sum(result.cumulative_yes_buy_qty for result in results)
    cumulative_no_buy_qty = sum(result.cumulative_no_buy_qty for result in results)
    larger_buy_qty = max(cumulative_yes_buy_qty, cumulative_no_buy_qty)
    pair_completion_ratio = (
        min(cumulative_yes_buy_qty, cumulative_no_buy_qty) / larger_buy_qty
        if larger_buy_qty > 0.0
        else 0.0
    )
    residual_qty_time_area_ms = sum(
        result.time_weighted_abs_residual_qty
        * (result.settlement_time_ms - result.trading_open_time_ms)
        for result in results
    )
    total_inventory_time_area_ms = sum(
        result.time_weighted_total_inventory_qty
        * (result.settlement_time_ms - result.trading_open_time_ms)
        for result in results
    )
    return OutcomePortfolioBacktestResult(
        starting_collateral=starting_collateral,
        ending_collateral=free_collateral,
        market_results=tuple(results),
        skipped_markets=tuple(skipped),
        orders_placed_count=sum(result.orders_placed_count for result in results),
        fills_count=sum(result.fills_count for result in results),
        maker_fills_count=sum(result.maker_fills_count for result in results),
        traded_notional=sum(result.traded_notional for result in results),
        trading_fees_paid=sum(result.trading_fees_paid for result in results),
        settlement_fees_paid=sum(result.settlement_fees_paid for result in results),
        fees_paid=sum(result.fees_paid for result in results),
        rebates_earned=sum(result.rebates_earned for result in results),
        gross_spread_pnl=sum(result.gross_spread_pnl for result in results),
        settlement_pnl=sum(result.settlement_pnl for result in results),
        max_concurrent_allocated_collateral=max_allocated,
        allocated_collateral_time_ratio=utilization,
        cumulative_yes_buy_qty=cumulative_yes_buy_qty,
        cumulative_no_buy_qty=cumulative_no_buy_qty,
        pair_completion_ratio=pair_completion_ratio,
        max_abs_residual_qty=_portfolio_max_abs_residual_qty(results),
        time_weighted_abs_residual_qty=(
            residual_qty_time_area_ms / inventory_duration_ms
            if inventory_duration_ms > 0
            else 0.0
        ),
        time_weighted_total_inventory_qty=(
            total_inventory_time_area_ms / inventory_duration_ms
            if inventory_duration_ms > 0
            else 0.0
        ),
        worst_case_settlement_equity_min=worst_case_equity_min,
        post_fill_adverse_selection=_portfolio_post_fill_adverse_selection(results),
    )
