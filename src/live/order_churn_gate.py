from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


ORDER_CHURN_GATE_SUPPORTED_EXCHANGES = frozenset(
    {
        "binance",
        "bitget",
        "bitunix",
        "bybit",
        "fake",
        "gateio",
        "hyperliquid",
        "kucoin",
        "okx",
        "weex",
    }
)

ORDER_CHURN_CONSOLE_REPEAT_SECONDS = 5.0 * 60.0
ORDER_CHURN_ADMISSION_SUMMARY_SECONDS = 10.0 * 60.0


def connector_supports_order_churn_gate(bot) -> bool:
    """Return the explicit setup-time rollout decision for this connector."""
    marker = getattr(bot, "_order_churn_gate_enabled_for_connector", None)
    if marker is not None:
        return bool(marker)
    exchange = str(getattr(bot, "exchange", "") or "").lower()
    if exchange:
        return exchange in ORDER_CHURN_GATE_SUPPORTED_EXCHANGES
    return True


@dataclass(frozen=True, order=True)
class OrderCohort:
    symbol: str
    position_side: str
    order_side: str
    reduce_only: bool
    execution_type: str
    pb_order_type: str


@dataclass(frozen=True)
class IdealObservation:
    cohort: OrderCohort
    qty: float
    price: float
    source_index: int

    @property
    def stable_key(self) -> tuple:
        return (self.cohort, self.price, self.qty)


@dataclass(frozen=True)
class IdealSnapshot:
    monotonic_seconds: float
    observations: tuple[IdealObservation, ...]


@dataclass(frozen=True)
class ChurnDecision:
    churn_evidenced: bool
    reason: str
    tight_prefix_count: int = 0
    tight_prefix_seconds: float = 0.0


def order_cohort(order: Mapping[str, object]) -> OrderCohort:
    symbol = str(order.get("symbol") or "")
    position_side = str(order.get("position_side") or "").lower()
    order_side = str(order.get("side") or "").lower()
    execution_type = str(
        order.get("type") or order.get("execution_type") or ""
    ).lower()
    pb_order_type = str(order.get("pb_order_type") or "").lower()
    reduce_only = order.get("reduce_only")
    if not symbol:
        raise ValueError("order cohort requires symbol")
    if position_side not in {"long", "short"}:
        raise ValueError("order cohort requires position_side long or short")
    if order_side not in {"buy", "sell"}:
        raise ValueError("order cohort requires side buy or sell")
    if not isinstance(reduce_only, bool):
        raise ValueError("order cohort requires authoritative reduce_only bool")
    if execution_type not in {"limit", "market"}:
        raise ValueError("order cohort requires execution type limit or market")
    if not pb_order_type or pb_order_type == "unknown":
        raise ValueError("order cohort requires normalized pb_order_type")
    return OrderCohort(
        symbol=symbol,
        position_side=position_side,
        order_side=order_side,
        reduce_only=reduce_only,
        execution_type=execution_type,
        pb_order_type=pb_order_type,
    )


def normalize_ideal_orders(
    orders: Sequence[Mapping[str, object]],
) -> list[IdealObservation]:
    out: list[IdealObservation] = []
    for source_index, order in enumerate(orders):
        qty = abs(float(order["qty"]))
        price = float(order["price"])
        if not math.isfinite(qty) or qty <= 0.0:
            raise ValueError("ideal order requires finite positive qty")
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("ideal order requires finite positive price")
        out.append(
            IdealObservation(
                cohort=order_cohort(order),
                qty=qty,
                price=price,
                source_index=source_index,
            )
        )
    return sorted(out, key=lambda observation: observation.stable_key)


def _relative_diff(current: float, previous: float) -> float:
    if not math.isfinite(current) or current <= 0.0:
        return math.inf
    return abs(previous - current) / current


def deterministic_one_to_one_matches(
    current: Sequence[IdealObservation],
    previous: Sequence[IdealObservation],
    tolerance: float,
) -> dict[int, int]:
    """Return deterministic maximum-cardinality matches within one tolerance.

    A small augmenting-path matcher preserves the most resting orders possible
    without the flow network and composite integer costs formerly used here.
    Candidate ordering is stable, but no economic meaning is assigned to the
    total matching cost.
    """
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("matching tolerance must be finite and non-negative")
    candidates: list[tuple[float, tuple, tuple, int, int]] = []
    for current_idx, current_observation in enumerate(current):
        for previous_idx, previous_observation in enumerate(previous):
            if current_observation.cohort != previous_observation.cohort:
                continue
            price_diff = _relative_diff(
                current_observation.price, previous_observation.price
            )
            qty_diff = _relative_diff(
                current_observation.qty, previous_observation.qty
            )
            if price_diff <= tolerance and qty_diff <= tolerance:
                candidates.append(
                    (
                        price_diff + qty_diff,
                        current_observation.stable_key,
                        previous_observation.stable_key,
                        current_idx,
                        previous_idx,
                    )
                )
    candidates_by_current: dict[int, list[int]] = defaultdict(list)
    for _cost, _current_key, _previous_key, current_idx, previous_idx in sorted(
        candidates
    ):
        candidates_by_current[current_idx].append(previous_idx)

    previous_owner: dict[int, int] = {}

    def assign(current_idx: int, visited_previous: set[int]) -> bool:
        for previous_idx in candidates_by_current.get(current_idx, []):
            if previous_idx in visited_previous:
                continue
            visited_previous.add(previous_idx)
            owner = previous_owner.get(previous_idx)
            if owner is None or assign(owner, visited_previous):
                previous_owner[previous_idx] = current_idx
                return True
        return False

    for current_idx in range(len(current)):
        assign(current_idx, set())
    return {
        current_idx: previous_idx
        for previous_idx, current_idx in sorted(previous_owner.items())
    }


def _group_by_cohort(
    observations: Sequence[IdealObservation],
) -> dict[OrderCohort, tuple[IdealObservation, ...]]:
    grouped: dict[OrderCohort, list[IdealObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.cohort].append(observation)
    return {
        cohort: tuple(sorted(rows, key=lambda row: (row.price, row.qty)))
        for cohort, rows in grouped.items()
    }


def _continuous_drift_start_index(
    values: Sequence[float], tolerance: float
) -> int | None:
    """Return the first changed observation in the current directional run."""
    if len(values) < 3:
        return None
    changed = [
        (index, right - left)
        for index, (left, right) in enumerate(zip(values, values[1:]), start=1)
        if right != left
    ]
    if len(changed) < 2:
        return None
    direction = 1 if changed[-1][1] > 0.0 else -1
    run_start_index = changed[-1][0]
    run_move_count = 0
    for target_index, delta in reversed(changed):
        if (delta > 0.0) != (direction > 0):
            break
        run_start_index = target_index
        run_move_count += 1
    if run_move_count < 2:
        return None
    baseline_index = run_start_index - 1
    if _relative_diff(values[-1], values[baseline_index]) <= tolerance:
        return None
    return run_start_index


def _exclusive_cohort_reappearance_seconds(
    cohort: OrderCohort,
    current_group: Sequence[IdealObservation],
    current_groups: Mapping[OrderCohort, Sequence[IdealObservation]],
    historical_groups: Sequence[
        tuple[float, Mapping[OrderCohort, Sequence[IdealObservation]]]
    ],
    *,
    now_monotonic: float,
    max_sample_gap_seconds: float,
) -> float | None:
    """Return the fixed span between the last two current-cohort runs.

    This intentionally recognizes only a narrow shape: each contiguous
    snapshot contains exactly one semantic cohort, and the current cohort has
    appeared in at least three distinct runs with unchanged cardinality. The
    first appearance and first reappearance therefore remain fail-open. Empty
    snapshots, coexisting cohorts, cadence gaps, and ladder-shape changes break
    the proof instead of broadening it.
    """
    if len(current_groups) != 1:
        return None
    cardinality_by_cohort = {cohort: len(current_group)}
    samples_newest_first: list[tuple[float, OrderCohort]] = [
        (now_monotonic, cohort)
    ]
    previous_time = now_monotonic
    for snapshot_time, groups in historical_groups:
        time_gap = previous_time - snapshot_time
        if (
            time_gap < 0.0
            or time_gap > max_sample_gap_seconds
            or len(groups) != 1
        ):
            break
        historical_cohort, historical_group = next(iter(groups.items()))
        historical_cardinality = len(historical_group)
        expected_cardinality = cardinality_by_cohort.setdefault(
            historical_cohort, historical_cardinality
        )
        if historical_cardinality != expected_cardinality:
            break
        samples_newest_first.append((snapshot_time, historical_cohort))
        previous_time = snapshot_time

    runs: list[tuple[OrderCohort, float]] = []
    for sample_time, sample_cohort in reversed(samples_newest_first):
        if not runs or runs[-1][0] != sample_cohort:
            runs.append((sample_cohort, sample_time))
    appearances = [
        run_start for run_cohort, run_start in runs if run_cohort == cohort
    ]
    if len(appearances) < 3:
        return None
    return max(0.0, appearances[-1] - appearances[-2])


class OrderChurnGateState:
    def __init__(self) -> None:
        self.history_by_symbol: dict[str, deque[IdealSnapshot]] = {}
        self.action_attempt_timestamps: deque[float] = deque()
        self.generation = 0
        self.reset_count = 0
        self.history_started = False
        self.console_log_state: dict[str, dict[str, object]] = {}
        self.admission_reason_counts: Counter[str] = Counter()
        self.admission_summary_started_at: float | None = None
        self.history_reset_during_evaluation = False

    def should_log_console_event(
        self,
        family: str,
        signature: object,
        *,
        now_monotonic: float,
        repeat_seconds: float = ORDER_CHURN_CONSOLE_REPEAT_SECONDS,
    ) -> tuple[bool, int]:
        """Throttle repeated INFO projections without affecting decisions."""
        if not family:
            raise ValueError("console log family must be non-empty")
        if not math.isfinite(now_monotonic):
            raise ValueError("console log timestamp must be finite")
        if not math.isfinite(repeat_seconds) or repeat_seconds <= 0.0:
            raise ValueError("console log repeat interval must be finite and positive")
        previous = self.console_log_state.get(family)
        if previous is None or previous.get("signature") != signature:
            self.console_log_state[family] = {
                "signature": signature,
                "last_log_monotonic": now_monotonic,
                "suppressed_count": 0,
            }
            return True, 0
        last_log = float(previous["last_log_monotonic"])
        if now_monotonic - last_log >= repeat_seconds:
            suppressed = int(previous.get("suppressed_count", 0) or 0)
            previous["last_log_monotonic"] = now_monotonic
            previous["suppressed_count"] = 0
            return True, suppressed
        previous["suppressed_count"] = int(previous.get("suppressed_count", 0) or 0) + 1
        return False, 0

    def begin_generation(self) -> int:
        self.generation += 1
        return self.generation

    def record_admission_reasons(
        self,
        reasons: Mapping[str, int],
        *,
        now_monotonic: float,
        interval_seconds: float = ORDER_CHURN_ADMISSION_SUMMARY_SECONDS,
    ) -> dict[str, int] | None:
        """Aggregate admission outcomes for a low-rate account-wide summary."""
        if not math.isfinite(now_monotonic):
            raise ValueError("admission summary timestamp must be finite")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("admission summary interval must be finite and positive")
        for reason, count in reasons.items():
            count_i = int(count)
            if count_i > 0:
                self.admission_reason_counts[str(reason)] += count_i
        if self.admission_summary_started_at is None:
            self.admission_summary_started_at = now_monotonic
            return None
        if now_monotonic - self.admission_summary_started_at < interval_seconds:
            return None
        summary = dict(sorted(self.admission_reason_counts.items()))
        self.admission_reason_counts.clear()
        self.admission_summary_started_at = now_monotonic
        return summary

    def clear_history(self) -> bool:
        had_history = bool(self.history_by_symbol)
        self.history_by_symbol.clear()
        if had_history:
            self.reset_count += 1
        return had_history

    def clear_symbol_history(self, symbol: str) -> bool:
        if self.history_by_symbol.pop(str(symbol), None) is None:
            return False
        self.reset_count += 1
        return True

    def symbols_with_history(self) -> set[str]:
        return set(self.history_by_symbol)

    @staticmethod
    def _prune_snapshots(
        snapshots: deque[IdealSnapshot], now: float, window_seconds: float
    ) -> None:
        cutoff = now - window_seconds
        while snapshots and snapshots[0].monotonic_seconds < cutoff:
            snapshots.popleft()

    def evaluate_and_record(
        self,
        ideal_orders_by_symbol: Mapping[str, Sequence[Mapping[str, object]]],
        *,
        now_monotonic: float,
        tolerance: float,
        stability_seconds: float,
        window_seconds: float,
        max_sample_gap_seconds: float,
    ) -> dict[int, ChurnDecision]:
        self.history_reset_during_evaluation = False
        current_by_symbol = {
            str(symbol): normalize_ideal_orders(list(orders))
            for symbol, orders in ideal_orders_by_symbol.items()
        }
        decisions: dict[int, ChurnDecision] = {}
        for symbol, current in current_by_symbol.items():
            snapshots = self.history_by_symbol.setdefault(symbol, deque())
            self._prune_snapshots(snapshots, now_monotonic, window_seconds)
            current_groups = _group_by_cohort(current)
            historical_groups = [
                (snapshot.monotonic_seconds, _group_by_cohort(snapshot.observations))
                for snapshot in reversed(snapshots)
            ]
            by_source_index: dict[int, ChurnDecision] = {}
            for cohort, current_group in current_groups.items():
                reappearance_seconds = _exclusive_cohort_reappearance_seconds(
                    cohort,
                    current_group,
                    current_groups,
                    historical_groups,
                    now_monotonic=now_monotonic,
                    max_sample_gap_seconds=max_sample_gap_seconds,
                )
                tracks = [[observation] for observation in current_group]
                track_times = [now_monotonic]
                previous_time = now_monotonic
                for snapshot_time, groups in historical_groups:
                    time_gap = previous_time - snapshot_time
                    previous_group = groups.get(cohort)
                    if (
                        time_gap < 0.0
                        or time_gap > max_sample_gap_seconds
                        or previous_group is None
                        or len(previous_group) != len(current_group)
                    ):
                        break
                    for index, observation in enumerate(previous_group):
                        tracks[index].append(observation)
                    track_times.append(snapshot_time)
                    previous_time = snapshot_time

                for observation, track in zip(current_group, tracks):
                    current_observation = track[0]
                    tight_count = 0
                    oldest_tight_time = now_monotonic
                    for index, historical in enumerate(track[1:], start=1):
                        if (
                            _relative_diff(current_observation.price, historical.price)
                            > tolerance
                            or _relative_diff(current_observation.qty, historical.qty)
                            > tolerance
                        ):
                            break
                        tight_count += 1
                        oldest_tight_time = track_times[index]
                    tight_seconds = max(0.0, now_monotonic - oldest_tight_time)
                    if tight_count >= 2 and tight_seconds >= stability_seconds:
                        decision = ChurnDecision(
                            False,
                            "stable_tight_prefix",
                            tight_prefix_count=tight_count,
                            tight_prefix_seconds=tight_seconds,
                        )
                    elif (
                        reappearance_seconds is not None
                        and reappearance_seconds >= stability_seconds
                    ):
                        decision = ChurnDecision(
                            True,
                            "intermittent_cohort_reappearance",
                            tight_prefix_count=tight_count,
                            tight_prefix_seconds=tight_seconds,
                        )
                    elif len(track) == 1:
                        decision = ChurnDecision(
                            False,
                            (
                                "intermittent_run_short"
                                if reappearance_seconds is not None
                                else "no_history"
                            ),
                        )
                    elif now_monotonic - track_times[-1] < stability_seconds:
                        decision = ChurnDecision(
                            False,
                            (
                                "intermittent_run_short"
                                if reappearance_seconds is not None
                                else "history_short"
                            ),
                            tight_prefix_count=tight_count,
                            tight_prefix_seconds=tight_seconds,
                        )
                    else:
                        chronological = list(reversed(track))
                        chronological_times = list(reversed(track_times))
                        price_drift_start = _continuous_drift_start_index(
                            [row.price for row in chronological], tolerance
                        )
                        qty_drift_start = _continuous_drift_start_index(
                            [row.qty for row in chronological], tolerance
                        )
                        price_drift = (
                            price_drift_start is not None
                            and now_monotonic
                            - chronological_times[price_drift_start]
                            >= stability_seconds
                        )
                        qty_drift = (
                            qty_drift_start is not None
                            and now_monotonic - chronological_times[qty_drift_start]
                            >= stability_seconds
                        )
                        if price_drift and qty_drift:
                            reason = "continuous_price_qty_drift"
                        elif price_drift:
                            reason = "continuous_price_drift"
                        elif qty_drift:
                            reason = "continuous_qty_drift"
                        elif (
                            price_drift_start is not None
                            or qty_drift_start is not None
                        ):
                            reason = "drift_run_short"
                        elif reappearance_seconds is not None:
                            reason = "intermittent_run_short"
                        else:
                            reason = "no_continuous_drift"
                        decision = ChurnDecision(
                            price_drift or qty_drift,
                            reason,
                            tight_prefix_count=tight_count,
                            tight_prefix_seconds=tight_seconds,
                        )
                    by_source_index[observation.source_index] = decision

            stable_decisions = list(by_source_index.values())
            if (
                len(current_groups) == 1
                and stable_decisions
                and all(
                    decision.reason == "stable_tight_prefix"
                    for decision in stable_decisions
                )
            ):
                # A completed stable exclusive run ends the preceding switching
                # episode for the whole symbol. Retain the proven stable prefix
                # as fresh history so one later switch cannot resurrect older
                # intermittent evidence.
                stable_prefix_seconds = min(
                    decision.tight_prefix_seconds for decision in stable_decisions
                )
                stable_prefix_start = now_monotonic - stable_prefix_seconds
                discarded_history = False
                while (
                    snapshots
                    and snapshots[0].monotonic_seconds < stable_prefix_start
                ):
                    snapshots.popleft()
                    discarded_history = True
                if discarded_history:
                    self.reset_count += 1
                    self.history_reset_during_evaluation = True

            for observation in current:
                source_order = ideal_orders_by_symbol[symbol][observation.source_index]
                decisions[id(source_order)] = by_source_index[observation.source_index]
            snapshots.append(
                IdealSnapshot(
                    monotonic_seconds=now_monotonic,
                    observations=tuple(current),
                )
            )

        for symbol, snapshots in list(self.history_by_symbol.items()):
            self._prune_snapshots(snapshots, now_monotonic, window_seconds)
            if not snapshots and symbol not in current_by_symbol:
                self.history_by_symbol.pop(symbol, None)
        return decisions

    def prune_action_attempts(
        self, *, now_monotonic: float, window_seconds: float
    ) -> None:
        cutoff = now_monotonic - window_seconds
        while (
            self.action_attempt_timestamps
            and self.action_attempt_timestamps[0] < cutoff
        ):
            self.action_attempt_timestamps.popleft()

    def action_attempt_count(
        self, *, now_monotonic: float, window_seconds: float
    ) -> int:
        self.prune_action_attempts(
            now_monotonic=now_monotonic, window_seconds=window_seconds
        )
        return len(self.action_attempt_timestamps)

    def record_action_attempts(self, count: int, *, now_monotonic: float) -> None:
        if count < 0:
            raise ValueError("action attempt count must be non-negative")
        self.action_attempt_timestamps.extend(now_monotonic for _ in range(count))
