from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock

import pytest

from live import executor as executor_module
from live.executor import (
    _apply_order_churn_final_admission,
    _complete_terminal_signed_action_attempts,
)
from live.order_churn_gate import OrderChurnGateState


def _order(
    name: str,
    *,
    price: float,
    side: str = "buy",
    execution_type: str = "limit",
    priority: str = "ordinary",
    churn: bool = True,
) -> dict:
    return {
        "name": name,
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": side,
        "reduce_only": False,
        "type": execution_type,
        "pb_order_type": "entry_ema_anchor_long",
        "execution_priority": priority,
        "qty": 1.0,
        "price": price,
        "_churn_evidence": churn,
    }


class _Bot:
    def __init__(self):
        self._order_churn_gate_state = OrderChurnGateState()
        self.values = {
            "order_replacement_churn_gate_activation_count": 10,
            "order_replacement_churn_gate_window_minutes": 10.0,
            "order_replacement_churn_gate_market_dist_pct": 0.005,
            "max_n_creations_per_batch": 20,
        }
        self.prices = {"BTC/USDT:USDT": 100.0}
        self.action_headroom: float | int | None = math.inf

    def live_value(self, key: str):
        return self.values[key]

    def _is_market_execution_order(self, order: dict) -> bool:
        return order.get("type") == "market"

    async def _fetch_fresh_order_churn_market_prices(
        self, symbols: set[str], *, max_age_ms: int = 10_000
    ):
        return {symbol: self.prices.get(symbol) for symbol in symbols}

    async def _order_churn_far_create_headroom(self):
        return self.action_headroom


@pytest.mark.asyncio
async def test_allowance_blocks_only_far_churn_evidenced_ordinary_orders():
    bot = _Bot()
    bot._order_churn_gate_state.record_action_attempts(10, now_monotonic=0.0)
    # Keep the attempts in whatever monotonic window the implementation observes.
    import time

    bot._order_churn_gate_state.action_attempt_timestamps.clear()
    bot._order_churn_gate_state.record_action_attempts(
        10, now_monotonic=time.monotonic()
    )
    far = _order("far", price=99.0)
    near = _order("near", price=99.8)
    market = _order("market", price=90.0, execution_type="market")
    critical = _order("critical", price=90.0, priority="risk_critical")
    stable = _order("stable", price=90.0, churn=False)

    admitted = await _apply_order_churn_final_admission(
        bot, [far, near, market, stable, critical]
    )

    assert [order["name"] for order in admitted] == [
        "critical",
        "near",
        "market",
        "stable",
    ]
    assert far["_churn_gate_reason"] == "allowance_exhausted"


@pytest.mark.asyncio
async def test_unsupported_generic_connector_bypasses_churn_admission_policy():
    bot = _Bot()
    bot._order_churn_gate_enabled_for_connector = False

    def fail_live_value(_key):
        raise AssertionError("generic fallback must not read churn configuration")

    bot.live_value = fail_live_value
    orders = [_order("far", price=90.0), _order("near", price=99.8)]

    admitted = await _apply_order_churn_final_admission(bot, orders)

    assert admitted == orders


@pytest.mark.asyncio
async def test_exempt_orders_count_against_later_far_candidate_in_same_batch():
    bot = _Bot()
    bot.values["order_replacement_churn_gate_activation_count"] = 1
    near = _order("near", price=99.8)
    far = _order("far", price=99.0)

    admitted = await _apply_order_churn_final_admission(bot, [near, far])

    assert [order["name"] for order in admitted] == ["near"]
    assert far["_churn_gate_reason"] == "allowance_exhausted"


@pytest.mark.asyncio
async def test_missing_final_market_data_defers_only_churn_evidenced_candidate():
    bot = _Bot()
    bot.prices = {}
    churn = _order("churn", price=99.0)
    stable = _order("stable", price=99.0, churn=False)

    admitted = await _apply_order_churn_final_admission(bot, [churn, stable])

    assert [order["name"] for order in admitted] == ["stable"]
    assert churn["_churn_gate_reason"] == "market_price_unavailable"


@pytest.mark.asyncio
async def test_hyperliquid_style_headroom_only_blocks_far_churn_candidate():
    bot = _Bot()
    bot.action_headroom = 0
    far = _order("far", price=99.0)
    near = _order("near", price=99.8)
    critical = _order("critical", price=90.0, priority="risk_critical")

    admitted = await _apply_order_churn_final_admission(bot, [far, near, critical])

    assert [order["name"] for order in admitted] == ["critical", "near"]
    assert far["_churn_gate_reason"] == "action_headroom_exhausted"


@pytest.mark.asyncio
async def test_far_candidate_cannot_consume_headroom_needed_by_later_exempt_order():
    bot = _Bot()
    bot.action_headroom = 1
    far = _order("far", price=99.0)
    near = _order("near", price=99.8)

    admitted = await _apply_order_churn_final_admission(bot, [far, near])

    assert [order["name"] for order in admitted] == ["near"]
    assert far["_churn_gate_reason"] == "action_headroom_exhausted"
    assert near["_churn_gate_reason"] == "market_distance_exempt"


@pytest.mark.asyncio
async def test_far_candidate_reserves_required_config_action_with_create():
    bot = _Bot()
    bot.action_headroom = 1
    far = _order("far", price=99.0)

    admitted = await _apply_order_churn_final_admission(
        bot,
        [far],
        config_action_costs_by_symbol={far["symbol"]: 1},
    )

    assert admitted == []
    assert far["_churn_gate_reason"] == "action_headroom_exhausted"

    bot.action_headroom = 2
    admitted = await _apply_order_churn_final_admission(
        bot,
        [far],
        config_action_costs_by_symbol={far["symbol"]: 1},
    )
    assert admitted == [far]


def test_signed_action_tokens_complete_only_for_acknowledged_members():
    bot = _Bot()
    completed = []
    bot._complete_order_churn_signed_action_attempts = lambda tokens: completed.append(
        tokens
    )
    bot.did_create_order = lambda result: bool(result.get("acknowledged"))
    orders = [
        _order("ack", price=99.0),
        _order("rejected", price=99.0),
        _order("ambiguous", price=99.0),
    ]

    _complete_terminal_signed_action_attempts(
        bot,
        ("token-ack", "token-rejected", "token-ambiguous"),
        ({"acknowledged": True}, {"status": "rejected"}, {}),
        orders,
        action="create",
    )

    assert completed == [("token-ack", "token-rejected")]


def test_cancel_signed_action_tokens_distinguish_terminal_and_ambiguous_members():
    bot = _Bot()
    completed = []
    bot._complete_order_churn_signed_action_attempts = lambda tokens: completed.append(
        tokens
    )
    bot.did_cancel_order = lambda result, _order: result.get("status") == "success"
    orders = [_order("rejected", price=99.0), _order("ambiguous", price=99.0)]

    _complete_terminal_signed_action_attempts(
        bot,
        ("token-rejected", "token-ambiguous"),
        ({"status": "rejected"}, {}),
        orders,
        action="cancel",
    )

    assert completed == [("token-rejected",)]


@pytest.mark.asyncio
async def test_disabled_gate_preserves_priority_and_batch_cap_without_distance_fetch():
    bot = _Bot()
    bot.values["order_replacement_churn_gate_activation_count"] = 0
    bot.values["max_n_creations_per_batch"] = 2
    ordinary = _order("ordinary", price=90.0)
    critical = _order("critical", price=90.0, priority="risk_critical")
    another = _order("another", price=90.0)

    admitted = await _apply_order_churn_final_admission(
        bot, [ordinary, another, critical]
    )

    assert [order["name"] for order in admitted] == ["critical", "ordinary"]


@pytest.mark.asyncio
async def test_diagnostic_emitter_failure_cannot_change_admission():
    bot = _Bot()

    def fail_event(**_kwargs):
        raise RuntimeError("diagnostic sink failed")

    bot._emit_order_churn_admission_event = fail_event
    stable = _order("stable", price=90.0, churn=False)

    admitted = await _apply_order_churn_final_admission(bot, [stable])

    assert admitted == [stable]


@pytest.mark.asyncio
async def test_churn_admission_emitter_diagnostics_bound_hostile_exception_types(caplog):
    hostile_name = "AuthorizationChurnAdmissionEmitterFailure"
    secret = "https://hostile.example.invalid/churn?token=emitter-secret"
    HostileError = type(hostile_name, (RuntimeError,), {})

    def fail_event(**_kwargs):
        raise HostileError(secret)

    bot = _Bot()
    bot._emit_order_churn_admission_event = fail_event
    stable = _order("stable", price=90.0, churn=False)

    with caplog.at_level(logging.DEBUG):
        bot.values["order_replacement_churn_gate_activation_count"] = 0
        assert await _apply_order_churn_final_admission(bot, [stable]) == [stable]
        assert stable["_churn_gate_reason"] == "disabled"

        bot.values["order_replacement_churn_gate_activation_count"] = 10
        assert await _apply_order_churn_final_admission(bot, [stable]) == [stable]
        assert stable["_churn_gate_reason"] == "no_churn_evidence"

    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("[event] order churn admission emitter failed")
    ]
    assert [(record.levelno, record.getMessage()) for record in records] == [
        (
            logging.DEBUG,
            "[event] order churn admission emitter failed | error_type=RuntimeError",
        ),
        (
            logging.DEBUG,
            "[event] order churn admission emitter failed | error_type=RuntimeError",
        ),
    ]
    rendered = "\n".join(record.getMessage() for record in records)
    assert hostile_name not in rendered
    assert secret not in rendered
    assert "hostile.example.invalid" not in rendered


@pytest.mark.asyncio
async def test_churn_market_data_diagnostic_bounds_hostile_exception_type(caplog):
    hostile_name = "PrivateKeyChurnMarketDataFailure"
    secret = "https://hostile.example.invalid/churn?token=market-secret"
    HostileError = type(hostile_name, (RuntimeError,), {})

    async def fail_market_data(*_args, **_kwargs):
        raise HostileError(secret)

    bot = _Bot()
    bot._fetch_fresh_order_churn_market_prices = fail_market_data
    churn = _order("churn", price=99.0)
    stable = _order("stable", price=90.0, churn=False)

    with caplog.at_level(logging.DEBUG):
        admitted = await _apply_order_churn_final_admission(bot, [churn, stable])

    assert admitted == [stable]
    assert churn["_churn_gate_reason"] == "market_price_unavailable"
    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("[order] fresh market data unavailable")
    ]
    assert [(record.levelno, record.getMessage()) for record in records] == [
        (
            logging.WARNING,
            "[order] fresh market data unavailable for churn-gate final admission | "
            "error_type=RuntimeError",
        )
    ]
    rendered = "\n".join(record.getMessage() for record in records)
    assert hostile_name not in rendered
    assert secret not in rendered
    assert "hostile.example.invalid" not in rendered


@pytest.mark.asyncio
async def test_churn_headroom_diagnostic_bounds_hostile_exception_type(caplog):
    hostile_name = "SecretChurnHeadroomFailure"
    secret = "https://hostile.example.invalid/churn?token=headroom-secret"
    HostileError = type(hostile_name, (RuntimeError,), {})

    async def fail_headroom():
        raise HostileError(secret)

    bot = _Bot()
    bot._order_churn_far_create_headroom = fail_headroom
    churn = _order("churn", price=99.0)
    stable = _order("stable", price=90.0, churn=False)

    with caplog.at_level(logging.DEBUG):
        admitted = await _apply_order_churn_final_admission(bot, [churn, stable])

    assert admitted == [stable]
    assert churn["_churn_gate_reason"] == "action_headroom_unavailable"
    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("[order] connector churn headroom unavailable")
    ]
    assert [(record.levelno, record.getMessage()) for record in records] == [
        (
            logging.WARNING,
            "[order] connector churn headroom unavailable | error_type=RuntimeError",
        )
    ]
    rendered = "\n".join(record.getMessage() for record in records)
    assert hostile_name not in rendered
    assert secret not in rendered
    assert "hostile.example.invalid" not in rendered


@pytest.mark.asyncio
async def test_cancellation_capacity_diagnostics_bound_hostile_exception_types(
    monkeypatch, caplog
):
    hostile_name = "ApiKeyCancellationCapacityFailure"
    secret = "https://hostile.example.invalid/cancel?token=capacity-secret"
    HostileError = type(hostile_name, (RuntimeError,), {})

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

        def live_value(self, key):
            assert key == "max_n_cancellations_per_batch"
            return 1

        def _emit_execution_cancel_filter_event(self, **_kwargs):
            raise HostileError(secret)

        def add_to_recent_order_cancellations(self, _order):
            return None

        def log_order_action(self, *_args, **_kwargs):
            return None

        def _log_order_action_summary(self, *_args, **_kwargs):
            return None

        async def execute_cancellations(self, orders):
            self.submitted_orders = list(orders)
            return []

        def _resolve_pb_order_type(self, order):
            return str(order["pb_order_type"])

    def fail_priority_filter(_order):
        raise HostileError(secret)

    monkeypatch.setattr(executor_module, "_pb_attr", lambda _name: FakePassivbot)
    monkeypatch.setattr(executor_module, "_order_is_reduce_only", fail_priority_filter)
    bot = FakeBot()
    orders = [
        {"symbol": "BTC/USDT:USDT", "pb_order_type": "close_grid_long"},
        {"symbol": "ETH/USDT:USDT", "pb_order_type": "close_grid_short"},
    ]

    with caplog.at_level(logging.DEBUG):
        result = await executor_module.execute_cancellations_parent(bot, orders)

    assert result == []
    assert bot.submitted_orders == [orders[0]]
    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith(
            (
                "[order] cancellation priority filtering failed",
                "[event] cancellation-capacity emitter failed",
            )
        )
    ]
    assert [(record.levelno, record.getMessage()) for record in records] == [
        (
            logging.ERROR,
            "[order] cancellation priority filtering failed; using input order | "
            "error_type=RuntimeError",
        ),
        (
            logging.DEBUG,
            "[event] cancellation-capacity emitter failed | error_type=RuntimeError",
        ),
    ]
    rendered = "\n".join(record.getMessage() for record in records)
    assert hostile_name not in rendered
    assert secret not in rendered
    assert "hostile.example.invalid" not in rendered


@pytest.mark.asyncio
async def test_repeated_churn_deferral_keeps_structured_path_but_throttles_info(
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
    bot._order_churn_gate_state.record_action_attempts(10, now_monotonic=now[0])
    far = _order("far", price=99.0)

    with caplog.at_level(logging.DEBUG):
        assert await _apply_order_churn_final_admission(bot, [far]) == []
        now[0] = 101.0
        assert await _apply_order_churn_final_admission(bot, [far]) == []
        now[0] = 400.0
        assert await _apply_order_churn_final_admission(bot, [far]) == []

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
