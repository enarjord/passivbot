from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from outcome.orchestrator import OutcomePostFillAdverseSelection
from outcome.rust_runner import (
    post_fill_adverse_selection_from_rust_output,
    run_outcome_ema_anchor_backtest,
)


@dataclass(frozen=True)
class OutcomeStrategyEvaluation:
    strategy_kind: str
    execution_mode: str
    yes_fraction: float
    ending_collateral: float
    net_pnl: float
    gross_spread_pnl: float
    settlement_pnl: float
    trading_fees_paid: float
    settlement_fees_paid: float
    fees_paid: float
    rebates_earned: float
    pre_settlement_yes_qty: float
    pre_settlement_no_qty: float
    pre_settlement_yes_cost: float
    pre_settlement_no_cost: float
    pre_settlement_paired_qty: float
    pre_settlement_net_yes_exposure: float
    pre_settlement_worst_case_equity: float
    orders_placed_count: int
    fills_count: int
    order_fill_ratio: float
    cumulative_yes_buy_qty: float
    cumulative_no_buy_qty: float
    max_paired_qty: float
    max_abs_residual_qty: float
    time_weighted_abs_residual_qty: float
    time_weighted_total_inventory_qty: float
    worst_case_settlement_equity_min: float
    post_fill_adverse_selection: tuple[OutcomePostFillAdverseSelection, ...]


@dataclass(frozen=True)
class OutcomeStrategyModeSummary:
    strategy_kind: str
    execution_mode: str
    settlement_cases: int
    gross_spread_pnl: float
    trading_fees_paid: float
    min_settlement_fees_paid: float
    max_settlement_fees_paid: float
    min_total_fees_paid: float
    max_total_fees_paid: float
    min_rebates_earned: float
    max_rebates_earned: float
    worst_net_pnl: float
    best_net_pnl: float
    settlement_sensitivity: float
    pre_settlement_yes_qty: float
    pre_settlement_no_qty: float
    pre_settlement_paired_qty: float
    pre_settlement_net_yes_exposure: float
    cumulative_yes_buy_qty: float
    cumulative_no_buy_qty: float
    complementary_buy_qty: float
    pair_completion_ratio: float
    pre_settlement_worst_case_equity: float
    worst_case_settlement_equity_min: float
    orders_placed_count: int
    fills_count: int
    order_fill_ratio: float
    max_paired_qty: float
    max_abs_residual_qty: float
    time_weighted_abs_residual_qty: float
    time_weighted_total_inventory_qty: float
    post_fill_adverse_selection: tuple[OutcomePostFillAdverseSelection, ...]


def evaluate_ema_anchor_outcome_modes(
    payload: Mapping[str, Any],
    *,
    execution_modes: Iterable[str] = (
        "accumulate_pairs",
        "inventory_aware",
        "yes_only",
    ),
    settlement_fractions: Iterable[float] = (0.0, 1.0),
) -> list[OutcomeStrategyEvaluation]:
    """Run strategy modes on identical candles and both ordinary settlement outcomes."""

    evaluations = []
    for execution_mode in execution_modes:
        for yes_fraction in settlement_fractions:
            run_payload = deepcopy(dict(payload))
            params = dict(run_payload["strategy_params"])
            params["execution_mode"] = execution_mode
            run_payload["strategy_params"] = params
            run_payload["yes_fraction"] = float(yes_fraction)
            output = run_outcome_ema_anchor_backtest(run_payload)
            fill_yes_buy_qty, fill_no_buy_qty = _cumulative_buy_quantities(
                output.get("fills")
            )
            cumulative_yes_buy_qty = float(output["cumulative_yes_buy_qty"])
            cumulative_no_buy_qty = float(output["cumulative_no_buy_qty"])
            if (
                abs(cumulative_yes_buy_qty - fill_yes_buy_qty) > 1e-9
                or abs(cumulative_no_buy_qty - fill_no_buy_qty) > 1e-9
            ):
                raise ValueError("Rust cumulative outcome buys disagree with emitted fills")
            orders = int(output["orders_placed_count"])
            fills = int(output["fills_count"])
            starting = float(output["starting_collateral"])
            ending = float(output["ending_collateral"])
            evaluations.append(
                OutcomeStrategyEvaluation(
                    strategy_kind=str(output["strategy_kind"]),
                    execution_mode=execution_mode,
                    yes_fraction=float(yes_fraction),
                    ending_collateral=ending,
                    net_pnl=ending - starting,
                    gross_spread_pnl=float(output["gross_spread_pnl"]),
                    settlement_pnl=float(output["settlement_pnl"]),
                    trading_fees_paid=float(output["trading_fees_paid"]),
                    settlement_fees_paid=float(output["settlement_fees_paid"]),
                    fees_paid=float(output["fees_paid"]),
                    rebates_earned=float(output["rebates_earned"]),
                    pre_settlement_yes_qty=float(output["pre_settlement_yes_qty"]),
                    pre_settlement_no_qty=float(output["pre_settlement_no_qty"]),
                    pre_settlement_yes_cost=float(output["pre_settlement_yes_cost"]),
                    pre_settlement_no_cost=float(output["pre_settlement_no_cost"]),
                    pre_settlement_paired_qty=float(output["pre_settlement_paired_qty"]),
                    pre_settlement_net_yes_exposure=float(
                        output["pre_settlement_net_yes_exposure"]
                    ),
                    pre_settlement_worst_case_equity=float(
                        output["pre_settlement_worst_case_equity"]
                    ),
                    orders_placed_count=orders,
                    fills_count=fills,
                    order_fill_ratio=fills / orders if orders else 0.0,
                    cumulative_yes_buy_qty=cumulative_yes_buy_qty,
                    cumulative_no_buy_qty=cumulative_no_buy_qty,
                    max_paired_qty=float(output["max_paired_qty"]),
                    max_abs_residual_qty=float(output["max_abs_residual_qty"]),
                    time_weighted_abs_residual_qty=float(
                        output["time_weighted_abs_residual_qty"]
                    ),
                    time_weighted_total_inventory_qty=float(
                        output["time_weighted_total_inventory_qty"]
                    ),
                    worst_case_settlement_equity_min=float(
                        output["worst_case_settlement_equity_min"]
                    ),
                    post_fill_adverse_selection=(
                        post_fill_adverse_selection_from_rust_output(
                            output["post_fill_adverse_selection"]
                        )
                    ),
                )
            )
    return evaluations


def summarize_outcome_strategy_modes(
    evaluations: Iterable[OutcomeStrategyEvaluation],
) -> list[OutcomeStrategyModeSummary]:
    """Collapse settlement scenarios into one risk-aware row per execution mode."""

    grouped: dict[tuple[str, str], list[OutcomeStrategyEvaluation]] = {}
    for evaluation in evaluations:
        grouped.setdefault(
            (evaluation.strategy_kind, evaluation.execution_mode),
            [],
        ).append(evaluation)

    summaries = []
    for (strategy_kind, execution_mode), cases in sorted(grouped.items()):
        baseline = cases[0]
        path_fields = (
            "gross_spread_pnl",
            "trading_fees_paid",
            "pre_settlement_yes_qty",
            "pre_settlement_no_qty",
            "pre_settlement_paired_qty",
            "pre_settlement_net_yes_exposure",
            "pre_settlement_worst_case_equity",
            "orders_placed_count",
            "fills_count",
            "order_fill_ratio",
            "cumulative_yes_buy_qty",
            "cumulative_no_buy_qty",
            "max_paired_qty",
            "max_abs_residual_qty",
            "time_weighted_abs_residual_qty",
            "time_weighted_total_inventory_qty",
            "worst_case_settlement_equity_min",
            "post_fill_adverse_selection",
        )
        for case in cases[1:]:
            for field in path_fields:
                left = getattr(baseline, field)
                right = getattr(case, field)
                if isinstance(left, float):
                    if abs(left - right) > 1e-9:
                        raise ValueError(
                            f"settlement scenarios changed pre-settlement field {field}"
                        )
                elif left != right:
                    raise ValueError(
                        f"settlement scenarios changed pre-settlement field {field}"
                    )

        net_pnls = [case.net_pnl for case in cases]
        settlement_fees = [case.settlement_fees_paid for case in cases]
        total_fees = [case.fees_paid for case in cases]
        rebates = [case.rebates_earned for case in cases]
        larger_buy_qty = max(
            baseline.cumulative_yes_buy_qty,
            baseline.cumulative_no_buy_qty,
        )
        complementary_buy_qty = min(
            baseline.cumulative_yes_buy_qty,
            baseline.cumulative_no_buy_qty,
        )
        pair_completion_ratio = (
            complementary_buy_qty / larger_buy_qty if larger_buy_qty > 0.0 else 0.0
        )
        summaries.append(
            OutcomeStrategyModeSummary(
                strategy_kind=strategy_kind,
                execution_mode=execution_mode,
                settlement_cases=len(cases),
                gross_spread_pnl=baseline.gross_spread_pnl,
                trading_fees_paid=baseline.trading_fees_paid,
                min_settlement_fees_paid=min(settlement_fees),
                max_settlement_fees_paid=max(settlement_fees),
                min_total_fees_paid=min(total_fees),
                max_total_fees_paid=max(total_fees),
                min_rebates_earned=min(rebates),
                max_rebates_earned=max(rebates),
                worst_net_pnl=min(net_pnls),
                best_net_pnl=max(net_pnls),
                settlement_sensitivity=max(net_pnls) - min(net_pnls),
                pre_settlement_yes_qty=baseline.pre_settlement_yes_qty,
                pre_settlement_no_qty=baseline.pre_settlement_no_qty,
                pre_settlement_paired_qty=baseline.pre_settlement_paired_qty,
                pre_settlement_net_yes_exposure=baseline.pre_settlement_net_yes_exposure,
                cumulative_yes_buy_qty=baseline.cumulative_yes_buy_qty,
                cumulative_no_buy_qty=baseline.cumulative_no_buy_qty,
                complementary_buy_qty=complementary_buy_qty,
                pair_completion_ratio=pair_completion_ratio,
                pre_settlement_worst_case_equity=baseline.pre_settlement_worst_case_equity,
                worst_case_settlement_equity_min=baseline.worst_case_settlement_equity_min,
                orders_placed_count=baseline.orders_placed_count,
                fills_count=baseline.fills_count,
                order_fill_ratio=baseline.order_fill_ratio,
                max_paired_qty=baseline.max_paired_qty,
                max_abs_residual_qty=baseline.max_abs_residual_qty,
                time_weighted_abs_residual_qty=baseline.time_weighted_abs_residual_qty,
                time_weighted_total_inventory_qty=(
                    baseline.time_weighted_total_inventory_qty
                ),
                post_fill_adverse_selection=baseline.post_fill_adverse_selection,
            )
        )
    return summaries


def _cumulative_buy_quantities(raw_fills: Any) -> tuple[float, float]:
    if not isinstance(raw_fills, list):
        raise TypeError("Rust outcome backtest omitted fills")
    totals = {"yes": 0.0, "no": 0.0}
    for raw in raw_fills:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("fill"), Mapping):
            raise TypeError("Rust outcome backtest returned a malformed fill")
        fill = raw["fill"]
        if str(fill.get("side")) != "buy":
            continue
        outcome = str(fill.get("outcome"))
        if outcome not in totals:
            raise ValueError(f"Rust outcome backtest returned unknown outcome {outcome!r}")
        qty = float(fill["qty"])
        if qty <= 0.0:
            raise ValueError("Rust outcome backtest returned non-positive fill quantity")
        totals[outcome] += qty
    return totals["yes"], totals["no"]
