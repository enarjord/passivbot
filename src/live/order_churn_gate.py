from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


ORDER_CHURN_GATE_SUPPORTED_EXCHANGES = frozenset(
    {
        "binance",
        "bitget",
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


def connector_supports_order_churn_gate(bot) -> bool:
    """Return the explicit setup-time rollout decision for this connector.

    Test doubles and direct unit constructions predate the setup marker and remain
    enabled by default. Production ``setup_bot()`` always installs the marker.
    """
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
        # Exact duplicates are intentionally indistinguishable. Their source
        # list position may select which identical object receives a match,
        # but it must not affect the causal multiset of admission decisions.
        return (self.cohort, self.price, self.qty)


@dataclass(frozen=True)
class IdealSnapshot:
    generation: int
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


def normalize_ideal_orders(orders: Sequence[Mapping[str, object]]) -> list[IdealObservation]:
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


def _match_cost(current: IdealObservation, previous: IdealObservation) -> float:
    return _relative_diff(current.price, previous.price) + _relative_diff(
        current.qty, previous.qty
    )


@dataclass
class _FlowEdge:
    to: int
    reverse_index: int
    capacity: int
    cost: int


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, cost: int) -> int:
    forward_index = len(graph[source])
    reverse_index = len(graph[target])
    graph[source].append(_FlowEdge(target, reverse_index, 1, cost))
    graph[target].append(_FlowEdge(source, forward_index, 0, -cost))
    return forward_index


def deterministic_one_to_one_matches(
    current: Sequence[IdealObservation],
    previous: Sequence[IdealObservation],
    tolerance: float,
) -> dict[int, int]:
    """Return maximum-cardinality, minimum-cost deterministic matches.

    The returned keys and values are indices into the supplied sequences. Cohort equality is exact;
    price and quantity must both be within ``tolerance`` relative to the current observation.
    """
    if not current or not previous:
        return {}
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("matching tolerance must be finite and non-negative")

    current_order = sorted(range(len(current)), key=lambda idx: current[idx].stable_key)
    previous_order = sorted(range(len(previous)), key=lambda idx: previous[idx].stable_key)
    n_current = len(current_order)
    n_previous = len(previous_order)
    source = 0
    current_offset = 1
    previous_offset = current_offset + n_current
    sink = previous_offset + n_previous
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for local_idx in range(n_current):
        _add_flow_edge(graph, source, current_offset + local_idx, 0)
    for local_idx in range(n_previous):
        _add_flow_edge(graph, previous_offset + local_idx, sink, 0)

    tracked_edges: list[tuple[int, int, int, int]] = []
    # One unit of quantized distance must dominate the complete tie-break sum
    # across a maximum-cardinality matching, not merely one edge.
    tie_scale = max(1, min(n_current, n_previous) * n_current * n_previous + 1)
    for current_local, current_idx in enumerate(current_order):
        current_observation = current[current_idx]
        for previous_local, previous_idx in enumerate(previous_order):
            previous_observation = previous[previous_idx]
            if current_observation.cohort != previous_observation.cohort:
                continue
            price_diff = _relative_diff(current_observation.price, previous_observation.price)
            qty_diff = _relative_diff(current_observation.qty, previous_observation.qty)
            if price_diff > tolerance or qty_diff > tolerance:
                continue
            base_cost = int(round((price_diff + qty_diff) * 1_000_000_000_000))
            tie_cost = current_local * n_previous + previous_local
            edge_index = _add_flow_edge(
                graph,
                current_offset + current_local,
                previous_offset + previous_local,
                base_cost * tie_scale + tie_cost,
            )
            tracked_edges.append((current_local, previous_local, edge_index, current_idx))

    node_count = len(graph)
    while True:
        infinity = 10**60
        distances = [infinity] * node_count
        predecessors: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == infinity:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distances[node] + edge.cost
                    predecessor_key = (node, edge_index)
                    if candidate < distances[edge.to] or (
                        candidate == distances[edge.to]
                        and (
                            predecessors[edge.to] is None
                            or predecessor_key < predecessors[edge.to]
                        )
                    ):
                        distances[edge.to] = candidate
                        predecessors[edge.to] = predecessor_key
                        changed = True
            if not changed:
                break
        if predecessors[sink] is None:
            break
        node = sink
        while node != source:
            predecessor = predecessors[node]
            if predecessor is None:
                raise RuntimeError("broken matching augmenting path")
            previous_node, edge_index = predecessor
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = previous_node

    matches: dict[int, int] = {}
    for current_local, previous_local, edge_index, current_idx in tracked_edges:
        edge = graph[current_offset + current_local][edge_index]
        if edge.capacity == 0:
            matches[current_idx] = previous_order[previous_local]
    return matches


def _snapshot_associations(
    current: Sequence[IdealObservation],
    previous: Sequence[IdealObservation],
    tight_tolerance: float,
    wider_tolerance: float,
) -> dict[int, str]:
    tight = deterministic_one_to_one_matches(current, previous, tight_tolerance)
    remaining_current_indices = [idx for idx in range(len(current)) if idx not in tight]
    used_previous = set(tight.values())
    remaining_previous_indices = [
        idx for idx in range(len(previous)) if idx not in used_previous
    ]
    remaining_current = [current[idx] for idx in remaining_current_indices]
    remaining_previous = [previous[idx] for idx in remaining_previous_indices]
    wider_local = deterministic_one_to_one_matches(
        remaining_current, remaining_previous, wider_tolerance
    )
    result = {idx: "tight" for idx in tight}
    for current_local in wider_local:
        result[remaining_current_indices[current_local]] = "wider"
    return result


class OrderChurnGateState:
    def __init__(self) -> None:
        self.history_by_symbol: dict[str, deque[IdealSnapshot]] = {}
        self.compatibility_epoch_by_symbol: dict[str, object] = {}
        self.action_attempt_timestamps: deque[float] = deque()
        self.generation = 0
        self.account_epoch: object | None = None
        self.reset_count = 0
        self.console_log_state: dict[str, dict[str, object]] = {}

    def should_log_console_event(
        self,
        family: str,
        signature: object,
        *,
        now_monotonic: float,
        repeat_seconds: float = ORDER_CHURN_CONSOLE_REPEAT_SECONDS,
    ) -> tuple[bool, int]:
        """Throttle repeated INFO projections without suppressing events or decisions."""
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

    def reset_history_for_epoch(self, account_epoch: object) -> bool:
        if self.account_epoch is None:
            self.account_epoch = account_epoch
            return False
        if self.account_epoch == account_epoch:
            return False
        self.account_epoch = account_epoch
        self.history_by_symbol.clear()
        self.compatibility_epoch_by_symbol.clear()
        self.reset_count += 1
        return True

    def reset_history_for_symbol_epochs(
        self, compatibility_epochs: Mapping[str, object]
    ) -> set[str]:
        """Clear only histories normalized under changed symbol metadata."""
        changed: set[str] = set()
        for symbol, epoch in compatibility_epochs.items():
            symbol = str(symbol)
            previous = self.compatibility_epoch_by_symbol.get(symbol)
            if previous is not None and previous != epoch:
                self.history_by_symbol.pop(symbol, None)
                changed.add(symbol)
            self.compatibility_epoch_by_symbol[symbol] = epoch
        if changed:
            self.reset_count += 1
        return changed

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
        generation: int,
        now_monotonic: float,
        tight_tolerance: float,
        wider_tolerance: float,
        stability_seconds: float,
        window_seconds: float,
        max_generation_gap_seconds: float,
    ) -> dict[int, ChurnDecision]:
        if generation != self.generation:
            raise ValueError("generation must be the current planning generation")
        current_by_symbol = {
            str(symbol): normalize_ideal_orders(list(orders))
            for symbol, orders in ideal_orders_by_symbol.items()
        }
        decisions: dict[int, ChurnDecision] = {}
        for symbol, current in current_by_symbol.items():
            snapshots = self.history_by_symbol.setdefault(symbol, deque())
            self._prune_snapshots(snapshots, now_monotonic, window_seconds)
            decisions_by_index: dict[int, ChurnDecision] = {}
            unresolved = set(range(len(current)))
            tight_counts = {idx: 0 for idx in unresolved}
            oldest_tight_times = {idx: now_monotonic for idx in unresolved}
            expected_generation = generation - 1
            previous_time = now_monotonic
            for snapshot in reversed(snapshots):
                if not unresolved:
                    break
                if snapshot.generation != expected_generation:
                    for current_idx in unresolved:
                        decisions_by_index[current_idx] = ChurnDecision(
                            False, "generation_gap"
                        )
                    unresolved.clear()
                    break
                time_gap = previous_time - snapshot.monotonic_seconds
                if time_gap < 0.0 or time_gap > max_generation_gap_seconds:
                    for current_idx in unresolved:
                        decisions_by_index[current_idx] = ChurnDecision(
                            False, "time_gap"
                        )
                    unresolved.clear()
                    break
                # Match the complete current cohort group, including candidates
                # already resolved by newer evidence. Excluding them could let an
                # older observation be reused for another current order.
                association = _snapshot_associations(
                    current,
                    snapshot.observations,
                    tight_tolerance,
                    wider_tolerance,
                )
                resolved_now: set[int] = set()
                for current_idx in sorted(unresolved):
                    relation = association.get(current_idx)
                    if relation == "tight":
                        tight_counts[current_idx] += 1
                        oldest_tight_times[current_idx] = snapshot.monotonic_seconds
                        tight_seconds = max(
                            0.0,
                            now_monotonic - oldest_tight_times[current_idx],
                        )
                        if (
                            tight_counts[current_idx] >= 2
                            and tight_seconds >= stability_seconds
                        ):
                            decisions_by_index[current_idx] = ChurnDecision(
                                False,
                                "stable_tight_prefix",
                                tight_prefix_count=tight_counts[current_idx],
                                tight_prefix_seconds=tight_seconds,
                            )
                            resolved_now.add(current_idx)
                        continue
                    if relation == "wider":
                        decisions_by_index[current_idx] = ChurnDecision(
                            True,
                            "wider_but_not_tight",
                            tight_prefix_count=tight_counts[current_idx],
                            tight_prefix_seconds=max(
                                0.0,
                                now_monotonic - oldest_tight_times[current_idx],
                            ),
                        )
                    else:
                        decisions_by_index[current_idx] = ChurnDecision(
                            False, "uncertain_no_association"
                        )
                    resolved_now.add(current_idx)
                unresolved.difference_update(resolved_now)
                expected_generation -= 1
                previous_time = snapshot.monotonic_seconds

            for current_idx in unresolved:
                tight_count = tight_counts[current_idx]
                tight_seconds = max(
                    0.0, now_monotonic - oldest_tight_times[current_idx]
                )
                decisions_by_index[current_idx] = (
                    ChurnDecision(
                        False,
                        "tight_history_short",
                        tight_prefix_count=tight_count,
                        tight_prefix_seconds=tight_seconds,
                    )
                    if tight_count
                    else ChurnDecision(False, "no_history")
                )

            for current_idx, observation in enumerate(current):
                decision = decisions_by_index[current_idx]
                source_order = ideal_orders_by_symbol[symbol][observation.source_index]
                decisions[id(source_order)] = decision
            snapshots.append(
                IdealSnapshot(
                    generation=generation,
                    monotonic_seconds=now_monotonic,
                    observations=tuple(current),
                )
            )
        for symbol, snapshots in list(self.history_by_symbol.items()):
            self._prune_snapshots(snapshots, now_monotonic, window_seconds)
            if not snapshots and symbol not in current_by_symbol:
                self.history_by_symbol.pop(symbol, None)
                self.compatibility_epoch_by_symbol.pop(symbol, None)
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
