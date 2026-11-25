from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from live import executor as executor_module
from live.executor import (
    _apply_creation_batch_capacity,
    _apply_order_churn_admission,
)
from live.freshness import FreshnessLedger
from live.order_churn_gate import OrderChurnGateState
from live.planning_snapshot import PlanningSnapshot


def _order(
    name: str,
    *,
    distance: float | None = 0.01,
    execution_type: str = "limit",
    priority: str = "ordinary",
    churn: bool = True,
) -> dict:
    order = {
        "name": name,
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": "buy",
        "reduce_only": False,
        "type": execution_type,
        "pb_order_type": "entry_ema_anchor_long",
        "execution_priority": priority,
        "qty": 1.0,
        "price": 99.0,
        "_churn_evidence": churn,
    }
    if distance is not None:
        order["_churn_gate_market_distance"] = distance
    return order


class _Bot:
    def __init__(self):
        self._order_churn_gate_state = OrderChurnGateState()
        self.values = {
            "order_replacement_churn_gate_activation_count": 10,
            "order_replacement_churn_gate_window_minutes": 10.0,
            "order_replacement_churn_gate_market_distance_fraction": 0.005,
            "max_n_creations_per_batch": 20,
        }

    def live_value(self, key: str):
        return self.values[key]

    @staticmethod
    def _loss_gate_enabled() -> bool:
        return False

    @staticmethod
    def _is_market_execution_order(order: dict) -> bool:
        return order.get("type") == "market"


def _fill_allowance(bot: _Bot) -> None:
    bot._order_churn_gate_state.record_action_attempts(
        bot.values["order_replacement_churn_gate_activation_count"],
        now_monotonic=time.monotonic(),
    )


def test_allowance_blocks_only_far_churn_evidenced_ordinary_orders():
    bot = _Bot()
    _fill_allowance(bot)
    far = _order("far", distance=0.01)
    near = _order("near", distance=0.002)
    market = _order("market", execution_type="market", distance=None)
    critical = _order("critical", priority="risk_critical", distance=None)
    stable = _order("stable", churn=False, distance=None)

    admitted = _apply_order_churn_admission(
        bot, [far, near, market, stable, critical]
    )

    assert [order["name"] for order in admitted] == [
        "near",
        "market",
        "stable",
        "critical",
    ]
    assert far["_churn_gate_reason"] == "allowance_exhausted"


def _stamp_account(bot: _Bot) -> PlanningSnapshot:
    ledger = FreshnessLedger()
    bot._ensure_freshness_ledger = lambda: ledger
    ledger.begin_epoch()
    surfaces = {"positions", "open_orders"}
    for surface in surfaces:
        ledger.stamp(surface, signature=(), now_ms=1000)
    return PlanningSnapshot.capture(
        ts_ms=1000,
        exchange=bot.exchange,
        user="test",
        ledger=ledger,
        required_surfaces=surfaces,
        min_epochs={surface: ledger.epoch for surface in surfaces},
        symbols=(),
        market_snapshots={},
        market_snapshot_max_age_ms=1000,
        account_invalidation_generation=bot._account_invalidation_generation,
    )


@pytest.mark.parametrize(
    "state, exempt",
    [
        ("confirmed_empty", True),
        ("opposite_side_resting", True),
        ("partially_covered", False),
        ("unknown_resting_side", False),
        ("flat", False),
        ("awaiting_confirmation", False),
        ("cohort_changed", False),
        ("account_invalidated", False),
        ("multiple_close_candidates", False),
        ("missing_market_distance", False),
        ("one_way_empty", True),
        ("one_way_opposite_side_resting", False),
        ("one_way_opposing_close_candidate", False),
        ("venue_one_way", False),
        ("other_connector", False),
    ],
)
def test_bitunix_churn_allowance_for_sole_uncovered_close(state, exempt):
    bot = _Bot()
    bot.exchange = "weex" if state == "other_connector" else "bitunix"
    bot.hedge_mode = state != "venue_one_way"
    bot._config_hedge_mode = not state.startswith("one_way_")
    bot._account_invalidation_generation = 3
    bot._authoritative_pending_confirmations = {}
    bot.values["max_n_creations_per_batch"] = 1
    _fill_allowance(bot)
    entry = _order("unrelated", churn=False, distance=None)
    close = dict(
        _order("replacement"),
        side="sell",
        reduce_only=True,
        pb_order_type="close_grid_long",
    )
    symbol = close["symbol"]
    bot.positions = {symbol: {"long": {"size": 0.0 if state == "flat" else 2.0}}}
    bot.fetched_open_orders = []
    if state in {"opposite_side_resting", "one_way_opposite_side_resting"}:
        bot.fetched_open_orders = [
            {"symbol": symbol, "position_side": "short", "side": "buy", "qty": 0.25}
        ]
    elif state in {"partially_covered", "unknown_resting_side"}:
        bot.fetched_open_orders = [
            {
                "symbol": symbol,
                "position_side": "long",
                "side": "sell" if state == "partially_covered" else None,
                "qty": 0.25,
            }
        ]
    snapshot = _stamp_account(bot)
    ledger = bot._ensure_freshness_ledger()
    orders = [entry, close]
    if state == "awaiting_confirmation":
        bot._authoritative_pending_confirmations["open_orders"] = ledger.epoch + 1
    elif state == "cohort_changed":
        ledger.stamp("open_orders", signature=("changed",), now_ms=1001)
    elif state == "account_invalidated":
        bot._account_invalidation_generation += 1
    elif state == "multiple_close_candidates":
        orders.append(dict(close, name="other_close", price=102.0))
    elif state == "one_way_opposing_close_candidate":
        orders.append(
            dict(close, name="short_close", position_side="short", side="buy")
        )
    elif state == "missing_market_distance":
        del close["_churn_gate_market_distance"]

    admitted = _apply_order_churn_admission(bot, orders, planning_snapshot=snapshot)
    selected = _apply_creation_batch_capacity(bot, admitted)

    assert selected == ([close] if exempt else [entry])
    expected_reason = {
        "confirmed_empty": "uncovered_close_exempt",
        "opposite_side_resting": "uncovered_close_exempt",
        "one_way_empty": "uncovered_close_exempt",
        "missing_market_distance": "market_distance_unavailable",
    }.get(state, "allowance_exhausted")
    assert close["_churn_gate_reason"] == expected_reason


def test_unsupported_connector_bypasses_policy():
    bot = _Bot()
    bot._order_churn_gate_enabled_for_connector = False

    def fail_live_value(_key):
        raise AssertionError("unsupported connector must not read churn config")

    bot.live_value = fail_live_value
    orders = [_order("far"), _order("near", distance=0.002)]

    assert _apply_order_churn_admission(bot, orders) == orders


def test_exempt_order_counts_against_later_far_candidate():
    bot = _Bot()
    bot.values["order_replacement_churn_gate_activation_count"] = 1
    near = _order("near", distance=0.002)
    far = _order("far", distance=0.01)

    admitted = _apply_order_churn_admission(bot, [near, far])

    assert admitted == [near]
    assert far["_churn_gate_reason"] == "allowance_exhausted"


def test_missing_market_distance_defers_only_churn_evidenced_candidate():
    bot = _Bot()
    churn = _order("churn", distance=None)
    stable = _order("stable", churn=False, distance=None)

    admitted = _apply_order_churn_admission(bot, [churn, stable])

    assert admitted == [stable]
    assert churn["_churn_gate_reason"] == "market_distance_unavailable"


def test_create_capacity_preserves_risk_priority():
    bot = _Bot()
    bot.values["max_n_creations_per_batch"] = 2
    ordinary = _order("ordinary")
    critical = _order("critical", priority="risk_critical")
    another = _order("another")

    admitted = _apply_creation_batch_capacity(
        bot, [ordinary, another, critical]
    )

    assert [order["name"] for order in admitted] == ["critical", "ordinary"]
    assert another["_churn_gate_reason"] == "batch_capacity"


def test_capacity_preserves_native_close_priority_when_loss_gate_enabled():
    bot = _Bot()
    bot.values["max_n_creations_per_batch"] = 1
    bot._loss_gate_enabled = lambda: True
    preferred = dict(
        _order("native_priority"), side="sell", reduce_only=True, pb_order_type="close_grid_long"
    )
    uncovered = dict(
        preferred,
        name="uncovered",
        symbol="ETH/USDT:USDT",
        _churn_gate_reason="uncovered_close_exempt",
    )

    # The caller has already sorted the orders by native close-allocation priority.
    assert _apply_creation_batch_capacity(bot, [preferred, uncovered]) == [preferred]


@pytest.mark.parametrize("state", ["uncovered", "resting", "invalidated", "loss_gate"])
def test_binance_capacity_for_uncovered_two_child_close(state):
    bot = _Bot()
    bot.exchange = "binance"
    bot.hedge_mode = bot._config_hedge_mode = True
    bot._account_invalidation_generation = 1
    bot.values["max_n_creations_per_batch"] = 3
    symbol = "ETH/USDT:USDT"
    bot.positions = {symbol: {"long": {"size": 2.0}}}
    bot.fetched_open_orders = []
    if state == "resting":
        bot.fetched_open_orders.append({
            "symbol": symbol, "position_side": "long", "side": "sell", "qty": 0.25,
        })
    snapshot = _stamp_account(bot)
    if state == "invalidated":
        bot._account_invalidation_generation += 1
    if state == "loss_gate":
        bot._loss_gate_enabled = lambda: True

    critical = _order("critical", priority="risk_critical")
    entries = [_order(f"entry_{index}") for index in range(3)]
    close_a = dict(
        _order("close_a"), symbol=symbol, side="sell", reduce_only=True,
        pb_order_type="close_grid_long", price=101.0,
    )
    close_b = dict(close_a, name="close_b", price=102.0)
    # The caller has already honored native close allocation on the loss-gate path.
    plan = [close_a, *entries, close_b] if state == "loss_gate" else [*entries, close_a, close_b]

    admitted = _apply_creation_batch_capacity(
        bot, [*plan, critical], planning_snapshot=snapshot
    )

    expected = {
        "uncovered": [critical, close_a, close_b],
        "resting": [critical, *entries[:2]],
        "invalidated": [critical, *entries[:2]],
        "loss_gate": [critical, close_a, entries[0]],
    }
    assert admitted == expected[state]


def test_capacity_after_admission_does_not_starve_later_admissible_order():
    bot = _Bot()
    bot.values["max_n_creations_per_batch"] = 1
    bot.values["order_replacement_churn_gate_activation_count"] = 1
    _fill_allowance(bot)
    far_churn = _order("far_churn", distance=0.01)
    stable = _order("stable", distance=None, churn=False)

    churn_admitted = _apply_order_churn_admission(bot, [far_churn, stable])
    admitted = _apply_creation_batch_capacity(bot, churn_admitted)

    assert admitted == [stable]
    assert far_churn["_churn_gate_reason"] == "allowance_exhausted"


def test_disabled_gate_preserves_preselected_orders():
    bot = _Bot()
    bot.values["order_replacement_churn_gate_activation_count"] = 0
    orders = [_order("ordinary"), _order("critical", priority="risk_critical")]

    assert _apply_order_churn_admission(bot, orders) == orders
    assert all(order["_churn_gate_reason"] == "disabled" for order in orders)


def test_diagnostic_emitter_failure_cannot_change_admission(caplog):
    bot = _Bot()

    def fail_event(**_kwargs):
        raise RuntimeError("diagnostic sink failed")

    bot._emit_order_churn_admission_event = fail_event
    stable = _order("stable", churn=False, distance=None)

    with caplog.at_level(logging.DEBUG):
        admitted = _apply_order_churn_admission(bot, [stable])

    assert admitted == [stable]
    assert "error_type=RuntimeError" in caplog.text
    assert "diagnostic sink failed" not in caplog.text


def test_repeated_churn_deferral_keeps_events_but_throttles_info(
    monkeypatch, caplog
):
    bot = _Bot()
    now = [100.0]
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: now[0])
    structured_emitter = MagicMock()
    monkeypatch.setattr(
        executor_module._pb_attr("Passivbot"),
        "_emit_execution_create_filter_event",
        structured_emitter,
    )
    _fill_allowance(bot)
    far = _order("far")

    with caplog.at_level(logging.DEBUG):
        assert _apply_order_churn_admission(bot, [far]) == []
        now[0] = 101.0
        assert _apply_order_churn_admission(bot, [far]) == []
        now[0] = 400.0
        assert _apply_order_churn_admission(bot, [far]) == []

    records = [
        record
        for record in caplog.records
        if "churn gate deferred" in record.getMessage()
    ]
    assert [record.levelno for record in records] == [
        logging.INFO,
        logging.DEBUG,
        logging.INFO,
    ]
    assert "suppressed_repeats=1" in records[-1].getMessage()
    assert structured_emitter.call_count == 3


def test_churn_admission_logs_bounded_account_summary(monkeypatch, caplog):
    bot = _Bot()
    now = [100.0]
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: now[0])

    near = _order("near", distance=0.002)
    stable = _order("stable", churn=False, distance=None)
    with caplog.at_level(logging.INFO):
        assert _apply_order_churn_admission(bot, [near, stable]) == [near, stable]
        assert "churn gate admission summary" not in caplog.text
        now[0] += 601.0
        assert _apply_order_churn_admission(bot, [near]) == [near]

    summaries = [
        record.getMessage()
        for record in caplog.records
        if "churn gate admission summary" in record.getMessage()
    ]
    assert len(summaries) == 1
    assert "candidates=3" in summaries[0]
    assert "market_distance_exempt:2" in summaries[0]
    assert "no_churn_evidence:1" in summaries[0]


@pytest.mark.asyncio
async def test_cancellation_capacity_diagnostics_remain_isolated(
    monkeypatch, caplog
):
    class FakePassivbot:
        @staticmethod
        def _emit_execution_order_event(*_args, **_kwargs):
            return None

        @staticmethod
        def _live_event_console_available(_bot):
            return False

        @staticmethod
        def _log_symbol(symbol):
            return str(symbol).split("/")[0]

    class FakeBot:
        debug_mode = False

        def __init__(self):
            self._health_orders_cancelled = 0
            self._order_wave_in_progress = None
            self.state_change_detected_by_symbol = set()
            self.submitted_orders = []

        @staticmethod
        def live_value(key):
            assert key == "max_n_cancellations_per_batch"
            return 1

        @staticmethod
        def add_to_recent_order_cancellations(_order):
            return None

        @staticmethod
        def log_order_action(*_args, **_kwargs):
            return None

        @staticmethod
        def _log_order_action_summary(*_args, **_kwargs):
            return None

        async def execute_cancellations(self, orders):
            self.submitted_orders = list(orders)
            return []

        @staticmethod
        def _resolve_pb_order_type(order):
            return str(order["pb_order_type"])

    monkeypatch.setattr(executor_module, "_pb_attr", lambda _name: FakePassivbot)
    bot = FakeBot()
    orders = [
        {
            "symbol": "BTC/USDT:USDT",
            "pb_order_type": "close_grid_long",
        },
        {
            "symbol": "ETH/USDT:USDT",
            "pb_order_type": "close_grid_short",
        },
    ]

    with caplog.at_level(logging.DEBUG):
        result = await executor_module.execute_cancellations_parent(bot, orders)

    assert result == []
    assert bot.submitted_orders == [orders[0]]
