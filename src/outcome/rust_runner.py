from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Callable, Mapping

from outcome.orchestrator import (
    OutcomeBacktestJob,
    OutcomePostFillAdverseSelection,
    SingleOutcomeBacktestResult,
)
from outcome.models import NormalizedOutcomeMarket


def _executable_price_bounds(market: NormalizedOutcomeMarket) -> tuple[float, float]:
    """Return the first and last executable prices on the venue-reported grid."""

    payout_unit = market.payout_unit
    grid = market.price_grid
    upper_probe = math.nextafter(payout_unit, -math.inf)
    if grid.kind == "fixed_step":
        minimum_increment = grid.fixed_step
        assert minimum_increment is not None
        upper_increment = minimum_increment
    elif grid.kind == "significant_figures":
        assert grid.max_significant_figures is not None
        assert grid.max_decimal_places is not None
        minimum_increment = 10.0 ** -grid.max_decimal_places
        if upper_probe <= 0.0:
            raise ValueError("outcome price grid has no executable prices inside payout bounds")
        magnitude = math.floor(math.log10(upper_probe))
        upper_increment = max(
            minimum_increment,
            10.0 ** (magnitude - grid.max_significant_figures + 1),
        )
    else:  # pragma: no cover - model validation rejects unsupported kinds
        raise ValueError(f"unsupported outcome price grid {grid.kind!r}")

    min_price = minimum_increment
    max_price = math.floor(upper_probe / upper_increment) * upper_increment
    if not (
        math.isfinite(min_price)
        and math.isfinite(max_price)
        and 0.0 < min_price < max_price < payout_unit
    ):
        raise ValueError("outcome price grid has no executable prices inside payout bounds")
    return min_price, max_price


def normalized_market_to_rust_spec(
    market: NormalizedOutcomeMarket,
    *,
    qty_step: float | None = None,
    min_order_qty: float | None = None,
) -> dict[str, Any]:
    """Translate exchange-neutral metadata into the authoritative Rust market contract."""

    opens_ms = market.lifecycle.trading_open_time_ms
    closes_ms = market.lifecycle.trading_close_time_ms
    scheduled_event_ms = market.lifecycle.scheduled_event_time_ms
    if opens_ms is None or closes_ms is None or scheduled_event_ms is None:
        raise ValueError("outcome market lifecycle is incomplete for Rust planning")
    acceptance_ms = market.lifecycle.order_acceptance_time_ms
    order_entry_opens_ms = max(
        opens_ms,
        opens_ms if acceptance_ms is None else acceptance_ms,
    )
    effective_qty_step = market.qty_step if qty_step is None else qty_step
    effective_min_order_qty = (
        market.min_order_qty if min_order_qty is None else min_order_qty
    )
    if (
        effective_qty_step is None
        or not math.isfinite(effective_qty_step)
        or effective_qty_step <= 0.0
        or effective_min_order_qty is None
        or not math.isfinite(effective_min_order_qty)
        or effective_min_order_qty <= 0.0
    ):
        raise ValueError("outcome market quantity constraints are incomplete for Rust planning")
    if (
        market.qty_step is not None
        and qty_step is not None
        and not math.isclose(market.qty_step, qty_step, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("explicit outcome qty_step disagrees with venue metadata")
    if (
        market.min_order_qty is not None
        and min_order_qty is not None
        and not math.isclose(
            market.min_order_qty,
            min_order_qty,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("explicit outcome min_order_qty disagrees with venue metadata")
    price_grid: dict[str, Any] = {"kind": market.price_grid.kind}
    if market.price_grid.kind == "fixed_step":
        price_grid["step"] = market.price_grid.fixed_step
    elif market.price_grid.kind == "significant_figures":
        price_grid.update(
            {
                "max_significant_figures": market.price_grid.max_significant_figures,
                "max_decimal_places": market.price_grid.max_decimal_places,
            }
        )
    else:  # pragma: no cover - model validation rejects unsupported kinds
        raise ValueError(f"unsupported outcome price grid {market.price_grid.kind!r}")
    min_price, max_price = _executable_price_bounds(market)
    return {
        "venue": market.venue.value,
        "market_id": market.market_id,
        "yes_asset_id": market.yes_asset.asset_id,
        "no_asset_id": market.no_asset.asset_id,
        "payout_unit": market.payout_unit,
        "min_price": min_price,
        "max_price": max_price,
        "price_grid": price_grid,
        "qty_step": effective_qty_step,
        "min_qty": effective_min_order_qty,
        "min_notional": market.min_order_notional or 0.0,
        "trading_opens_ms": opens_ms,
        "order_entry_opens_ms": order_entry_opens_ms,
        "trading_closes_ms": closes_ms,
        "scheduled_event_ms": scheduled_event_ms,
        "capabilities": {
            "complementary_books_merged": market.capabilities.complementary_books_merged,
            "supports_split": market.capabilities.supports_split,
            "supports_merge": market.capabilities.supports_merge,
            "supports_redeem": market.capabilities.supports_redeem,
            "supports_post_only": market.capabilities.supports_post_only,
            "supports_gtd": market.capabilities.supports_gtd,
            "sell_requires_inventory": market.capabilities.sell_requires_inventory,
        },
    }


def run_single_outcome_backtest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the authoritative Rust single-market outcome simulator."""

    import passivbot_rust

    encoded = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True, allow_nan=False)
    result_json = passivbot_rust.run_single_outcome_backtest_json(encoded)
    result = json.loads(result_json)
    if not isinstance(result, dict):
        raise TypeError("Rust single outcome backtest returned a non-object")
    return result


def summary_from_rust_output(output: Mapping[str, Any]) -> SingleOutcomeBacktestResult:
    residual_qty = 0.0
    residual_qty_timeline = []
    for raw_fill in output["fills"]:
        fill = raw_fill["fill"]
        accounting = raw_fill["accounting"]
        residual_qty += float(accounting["canonical_yes_exposure_delta"])
        residual_qty_timeline.append((int(fill["timestamp_ms"]), residual_qty))
    return SingleOutcomeBacktestResult(
        market_id=str(output["market_id"]),
        trading_open_time_ms=int(output["trading_open_time_ms"]),
        settlement_time_ms=int(output["settlement_time_ms"]),
        starting_collateral=float(output["starting_collateral"]),
        ending_collateral=float(output["ending_collateral"]),
        orders_placed_count=int(output["orders_placed_count"]),
        fills_count=int(output["fills_count"]),
        maker_fills_count=int(output["maker_fills_count"]),
        traded_notional=float(output["traded_notional"]),
        trading_fees_paid=float(output["trading_fees_paid"]),
        settlement_fees_paid=float(output["settlement_fees_paid"]),
        fees_paid=float(output["fees_paid"]),
        rebates_earned=float(output["rebates_earned"]),
        gross_spread_pnl=float(output["gross_spread_pnl"]),
        settlement_pnl=float(output["settlement_pnl"]),
        pre_settlement_yes_qty=float(output["pre_settlement_yes_qty"]),
        pre_settlement_no_qty=float(output["pre_settlement_no_qty"]),
        pre_settlement_paired_qty=float(output["pre_settlement_paired_qty"]),
        pre_settlement_net_yes_exposure=float(
            output["pre_settlement_net_yes_exposure"]
        ),
        max_paired_qty=float(output["max_paired_qty"]),
        max_abs_residual_qty=float(output["max_abs_residual_qty"]),
        cumulative_yes_buy_qty=float(output["cumulative_yes_buy_qty"]),
        cumulative_no_buy_qty=float(output["cumulative_no_buy_qty"]),
        pair_completion_ratio=float(output["pair_completion_ratio"]),
        time_weighted_abs_residual_qty=float(
            output["time_weighted_abs_residual_qty"]
        ),
        time_weighted_total_inventory_qty=float(
            output["time_weighted_total_inventory_qty"]
        ),
        worst_case_settlement_equity_min=float(
            output["worst_case_settlement_equity_min"]
        ),
        post_fill_adverse_selection=post_fill_adverse_selection_from_rust_output(
            output["post_fill_adverse_selection"]
        ),
        residual_qty_timeline=tuple(residual_qty_timeline),
    )


def post_fill_adverse_selection_from_rust_output(
    raw_metrics: Any,
) -> tuple[OutcomePostFillAdverseSelection, ...]:
    if not isinstance(raw_metrics, list):
        raise TypeError("Rust outcome backtest omitted post-fill adverse selection")
    metrics = []
    for raw in raw_metrics:
        if not isinstance(raw, Mapping):
            raise TypeError("Rust outcome backtest returned a malformed markout metric")
        metrics.append(
            OutcomePostFillAdverseSelection(
                horizon_ms=int(raw["horizon_ms"]),
                total_fills_count=int(raw["total_fills_count"]),
                observed_fills_count=int(raw["observed_fills_count"]),
                total_fill_qty=float(raw["total_fill_qty"]),
                observed_fill_qty=float(raw["observed_fill_qty"]),
                fill_qty_coverage_ratio=(
                    None
                    if raw["fill_qty_coverage_ratio"] is None
                    else float(raw["fill_qty_coverage_ratio"])
                ),
                total_adverse_selection_quote=(
                    None
                    if raw["total_adverse_selection_quote"] is None
                    else float(raw["total_adverse_selection_quote"])
                ),
                mean_adverse_selection_per_share=(
                    None
                    if raw["mean_adverse_selection_per_share"] is None
                    else float(raw["mean_adverse_selection_per_share"])
                ),
            )
        )
    return tuple(metrics)


def make_rust_outcome_job(payload: Mapping[str, Any]) -> OutcomeBacktestJob:
    return _make_rust_outcome_job(payload, run_single_outcome_backtest)


def make_rust_ema_anchor_outcome_job(
    payload: Mapping[str, Any],
) -> OutcomeBacktestJob:
    """Wrap one EMA-anchor outcome payload for shared-wallet orchestration."""

    return _make_rust_outcome_job(payload, run_outcome_ema_anchor_backtest)


def _make_rust_outcome_job(
    payload: Mapping[str, Any],
    run_backtest: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> OutcomeBacktestJob:
    template = deepcopy(dict(payload))
    market = template.get("market")
    if not isinstance(market, Mapping):
        raise ValueError("Rust outcome payload is missing market")
    market_id = str(market.get("market_id", ""))
    trading_open_time_ms = int(market["trading_opens_ms"])
    settlement_time_ms = int(template["settlement_time_ms"])
    requested_collateral = float(template["starting_collateral"])

    def runner(allocated_collateral: float) -> SingleOutcomeBacktestResult:
        run_payload = deepcopy(template)
        run_payload["starting_collateral"] = allocated_collateral
        return summary_from_rust_output(run_backtest(run_payload))

    return OutcomeBacktestJob(
        market_id=market_id,
        trading_open_time_ms=trading_open_time_ms,
        settlement_time_ms=settlement_time_ms,
        requested_collateral=requested_collateral,
        runner=runner,
    )


def run_outcome_ema_anchor_backtest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the Rust EMA-anchor outcome strategy on canonical signal and execution candles."""

    import passivbot_rust

    encoded = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True, allow_nan=False)
    result_json = passivbot_rust.run_outcome_ema_anchor_backtest_json(encoded)
    result = json.loads(result_json)
    if not isinstance(result, dict):
        raise TypeError("Rust EMA-anchor outcome backtest returned a non-object")
    return result


def plan_outcome_ema_anchor(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct Rust EMA state from dense closes and return current native intents."""

    import passivbot_rust

    encoded = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True, allow_nan=False)
    result_json = passivbot_rust.plan_outcome_ema_anchor_json(encoded)
    result = json.loads(result_json)
    if not isinstance(result, dict):
        raise TypeError("Rust EMA-anchor outcome planner returned a non-object")
    return result
