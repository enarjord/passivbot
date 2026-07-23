import pytest

from live import event_emitters, executor, market_data, reconciler
from live.event_bus import EventTypes, ListEventSink, LiveEvent, LiveEventPipeline
from live.fresh_entry_eligibility import FreshEntryEligibilityTrace
from live.market_snapshot import MarketSnapshot
from live.order_churn_gate import OrderChurnGateState
from passivbot import Passivbot


def _initial(symbol: str, pside: str = "long", *, price: float = 100.0) -> dict:
    return {
        "symbol": symbol,
        "side": "buy" if pside == "long" else "sell",
        "position_side": pside,
        "qty": 1.0,
        "price": price,
        "reduce_only": False,
        "custom_id": f"entry-initial-{pside}",
        "pb_order_type": f"entry_initial_normal_{pside}",
        "type": "limit",
    }


def _protective(symbol: str, pside: str = "long") -> dict:
    return {
        "symbol": symbol,
        "side": "sell" if pside == "long" else "buy",
        "position_side": pside,
        "qty": 1.0,
        "price": 101.0,
        "reduce_only": True,
        "custom_id": f"close-panic-{pside}",
        "pb_order_type": f"close_panic_{pside}",
        "type": "limit",
    }


def test_eligibility_emitter_builds_schema_valid_correlated_event():
    sink = ListEventSink()

    class EventBot:
        _live_event_current_cycle_id = "cy_1"

        def __init__(self):
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink], monitor_sinks=[]
            )

        def _emit_live_event(self, event_type, **kwargs):
            return self._live_event_pipeline.emit(LiveEvent(event_type, **kwargs))

    bot = EventBot()
    event_emitters.emit_initial_entry_eligibility_event(
        bot,
        data={"records_total": 0, "records": []},
        wave={"event_id": "ow_1"},
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == EventTypes.ENTRY_INITIAL_ELIGIBILITY
    assert event.status == "succeeded"
    assert event.cycle_id == "cy_1"
    assert event.order_wave_id == "ow_1"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_reconciliation_trace_diagnostics_redact_hostile_failure_metadata(caplog):
    secret = "trace-secret https://example.invalid/trace"
    hostile_error = type("ApiKeySecretError", (RuntimeError,), {})

    class Trace:
        def record_evaluated(self, *_args, **_kwargs):
            raise hostile_error(secret)

    with caplog.at_level("DEBUG"):
        assert (
            reconciler._trace_record(
                Trace(), "record_evaluated", "BTC/USDT:USDT", "long"
            )
            is None
        )

    class Bot:
        @property
        def active_symbols(self):
            raise hostile_error(secret)

    with caplog.at_level("DEBUG"):
        assert reconciler._initialize_fresh_entry_trace(Bot(), {}, {}) is None

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "error_type=RuntimeError" in rendered
    assert hostile_error.__name__ not in rendered
    assert secret not in rendered
    assert "example.invalid" not in rendered


def _reconciliation_bot(symbol: str, actual_orders: list[dict]) -> Passivbot:
    bot = Passivbot.__new__(Passivbot)
    bot.active_symbols = [symbol]
    bot.PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "normal"}}
    bot._last_plan_detail = {}
    bot._malformed_actual_order_symbols = set()
    bot._orchestrator_trailing_unavailable_symbols = set()
    bot._malformed_actual_order_counts = {}
    bot._fresh_entry_conversion_blocked_counts = {}
    bot._snapshot_actual_orders = lambda *args, **kwargs: {symbol: list(actual_orders)}
    bot.is_pside_enabled = lambda pside: pside == "long"
    bot._annotate_order_deltas = lambda cancel, create: (cancel, create)
    bot._apply_order_match_tolerance = lambda cancel, create: (cancel, create, 0)
    bot._apply_freshness_creation_guardrails = lambda create: (create, 0)
    bot._order_plan_summary_is_interesting = lambda **kwargs: False

    async def keep_sort(self, orders, _label):
        return orders

    bot._sort_orders_by_market_diff = keep_sort.__get__(bot, Passivbot)
    return bot


@pytest.mark.asyncio
async def test_reconciliation_trace_distinguishes_satisfied_and_no_candidate():
    symbol = "BTC/USDT:USDT"
    matching = _initial(symbol)
    bot = _reconciliation_bot(symbol, [matching])

    to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        bot, {symbol: [_initial(symbol)]}
    )

    assert to_cancel == []
    assert to_create == []
    data = bot._fresh_entry_eligibility_trace.to_event_data()
    assert data["records"][0]["outcome"] == "already_satisfied"
    assert data["records"][0]["reason_counts"] == {
        "exact_reconciliation_match": 1
    }

    empty_bot = _reconciliation_bot(symbol, [])
    await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        empty_bot, {symbol: []}
    )
    empty_data = empty_bot._fresh_entry_eligibility_trace.to_event_data()
    assert empty_data["records"][0]["outcome"] == "no_candidate"
    assert empty_data["records"][0]["reason_counts"] == {
        "rust_no_initial_candidate": 1
    }


@pytest.mark.asyncio
async def test_reconciliation_trace_observes_protective_only():
    symbol = "BTC/USDT:USDT"
    protective_bot = _reconciliation_bot(symbol, [])
    await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        protective_bot, {symbol: [_protective(symbol)]}
    )
    protective_data = protective_bot._fresh_entry_eligibility_trace.to_event_data()
    assert protective_data["records"][0]["outcome"] == "protective_only"
    assert protective_data["records"][0]["reason_counts"] == {
        "protective_actions_only": 1
    }


@pytest.mark.asyncio
async def test_malformed_open_order_snapshot_blocks_every_account_action():
    malformed_symbol = "BTC/USDT:USDT"
    healthy_symbol = "ETH/USDT:USDT"
    stale_actual = _initial(healthy_symbol, price=90.0)
    bot = _reconciliation_bot(healthy_symbol, [stale_actual])
    bot.active_symbols = [malformed_symbol, healthy_symbol]
    bot.PB_modes["long"][malformed_symbol] = "normal"
    bot.PB_modes["short"][malformed_symbol] = "normal"

    def snapshot(*_args, **_kwargs):
        bot._malformed_actual_order_symbols = {malformed_symbol}
        bot._malformed_actual_order_counts = {malformed_symbol: 1}
        return {malformed_symbol: [], healthy_symbol: [stale_actual]}

    bot._snapshot_actual_orders = snapshot
    to_cancel, to_create = await reconciler.calc_orders_to_cancel_and_create_from_ideal(
        bot,
        {
            malformed_symbol: [_protective(malformed_symbol)],
            healthy_symbol: [_initial(healthy_symbol, price=100.0)],
        },
        apply_mode_filters=False,
    )

    assert to_cancel == []
    assert to_create == []


class _CreateBot:
    def __init__(self, trace: FreshEntryEligibilityTrace):
        self._fresh_entry_eligibility_trace = trace
        self._order_wave_in_progress = {"event_id": "ow_1"}
        self._health_orders_placed = 0
        self.debug_mode = False
        self.submitted: list[dict] = []

    def live_value(self, key):
        return {
            "max_n_creations_per_batch": 1,
            "order_replacement_churn_gate_activation_count": 10,
            "order_replacement_churn_gate_window_minutes": 10.0,
        }[key]

    def get_exchange_time(self):
        return 123456

    def log_order_action(self, *args, **kwargs):
        return None

    def _log_order_action_summary(self, *args, **kwargs):
        return None

    def _is_market_execution_order(self, _order):
        return False

    def _resolve_pb_order_type(self, order):
        return str(order.get("pb_order_type") or "unknown")

    async def execute_orders(self, orders):
        self.submitted = list(orders)
        return [{"id": "created", **order} for order in orders]

    def did_create_order(self, _result):
        return True

    def add_to_recent_order_executions(self, _order):
        return None

    def add_new_order(self, _order, source="POST"):
        return None

    def _monitor_record_event(self, *args, **kwargs):
        return None

    def _monitor_order_payload(self, order, source="POST"):
        return {"source": source, "symbol": order["symbol"]}


@pytest.mark.asyncio
async def test_final_batch_cap_classifies_eligible_and_blocked_without_changing_submission(
    monkeypatch,
):
    first = _initial("ADA/USDT:USDT")
    second = _initial("BTC/USDT:USDT")
    trace = FreshEntryEligibilityTrace()
    trace.record_ideal_orders([first, second])
    trace.record_evaluated(first["symbol"], "long")
    trace.record_evaluated(second["symbol"], "long")
    bot = _CreateBot(trace)
    emitted = []

    monkeypatch.setattr(Passivbot, "_record_emitted_order_custom_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(Passivbot, "_emit_execution_order_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Passivbot,
        "_emit_initial_entry_eligibility_event",
        lambda _bot, *, data, wave=None: emitted.append((data, wave)),
    )

    result = await executor.execute_orders_parent(bot, [first, second])

    assert bot.submitted == [first]
    assert len(result) == 1
    assert len(emitted) == 1
    records = {
        item["symbol"]: item for item in emitted[0][0]["records"]
    }
    assert records[first["symbol"]]["outcome"] == "eligible"
    assert records[second["symbol"]]["outcome"] == "blocked_candidate"
    assert records[second["symbol"]]["reason_counts"] == {"batch_capacity": 1}
    assert bot._fresh_entry_eligibility_trace is None


@pytest.mark.asyncio
async def test_completed_plan_emits_low_balance_block_from_existing_filter(monkeypatch):
    order = _initial("BTC/USDT:USDT")
    trace = FreshEntryEligibilityTrace()
    trace.record_ideal_orders([order])
    trace.record_evaluated(order["symbol"], "long")
    emitted = []

    class PlanBot:
        debug_mode = False
        balance_threshold = 1.0
        quote = "USDT"
        state_change_detected_by_symbol = set()
        _equity_hard_stop_coin_replay_pending_pairs = set()

        def __init__(self):
            self._fresh_entry_eligibility_trace = trace
            self.execution_scheduled = False

        def get_raw_balance(self):
            return 0.0

        async def execute_cancellations_parent(self, _orders):
            return []

    bot = PlanBot()
    wave = {
        "skipped_create": 0,
        "deferred_create": 0,
    }
    monkeypatch.setattr(Passivbot, "_begin_order_wave", lambda *args, **kwargs: wave)
    monkeypatch.setattr(
        Passivbot, "_emit_execution_create_filter_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        Passivbot,
        "_emit_initial_entry_eligibility_event",
        lambda _bot, *, data, wave=None: emitted.append(data),
    )
    monkeypatch.setattr(Passivbot, "_live_event_console_available", lambda *args: False)
    monkeypatch.setattr(Passivbot, "_shutdown_requested", lambda *args: True)
    monkeypatch.setattr(Passivbot, "_track_order_wave_confirmation", lambda *args: None)
    monkeypatch.setattr(Passivbot, "_log_order_wave_summary", lambda *args: None)

    await executor.execute_order_plan(bot, [], [order])

    assert len(emitted) == 1
    assert emitted[0]["records"][0]["outcome"] == "blocked_candidate"
    assert emitted[0]["records"][0]["reason_counts"] == {"low_balance": 1}
    assert bot._fresh_entry_eligibility_trace is None


@pytest.mark.asyncio
async def test_pre_create_market_filter_records_exact_existing_gate_reasons():
    order = _initial("BTC/USDT:USDT", price=1.0)

    class MarketBot:
        config = {"live": {"limit_order_create_max_market_dist_pct": 0.1}}

        def __init__(self, invalid=None, snapshot_error=None):
            self._fresh_entry_eligibility_trace = FreshEntryEligibilityTrace()
            self._fresh_entry_eligibility_trace.record_ideal_orders([order])
            self._invalid = invalid or []
            self._snapshot_error = snapshot_error

        def _current_planning_snapshot_invalid_for_creations(self, _symbols):
            return self._invalid

        async def _get_live_market_snapshots(self, symbols, **kwargs):
            if self._snapshot_error is not None:
                raise self._snapshot_error
            return {
                symbol: MarketSnapshot(
                    symbol=symbol,
                    bid=99.0,
                    ask=101.0,
                    last=100.0,
                    fetched_ms=1,
                    source="test",
                )
                for symbol in symbols
            }

        def _live_market_snapshot_max_age_ms(self):
            return 10_000

        def _record_market_snapshot_surface(self, _symbols, _snapshots):
            return None

        def _market_snapshot_signature_invalid(self, _symbols):
            return []

        def _emit_execution_create_filter_event(self, **kwargs):
            return None

        def _log_symbols(self, symbols, limit=12):
            return ",".join(symbols[:limit])

        def _log_compact_symbol_payload(self, _details):
            return "details"

        def _log_symbol(self, symbol):
            return str(symbol)

    planning_bot = MarketBot(
        invalid=[{"surface": "positions", "reason": "epoch_too_old"}]
    )
    assert await market_data.filter_fresh_market_snapshot_creations(
        planning_bot, [order]
    ) == []
    planning_record = planning_bot._fresh_entry_eligibility_trace.to_event_data()[
        "records"
    ][0]
    assert planning_record["reason_counts"] == {
        "pre_create_planning_snapshot_invalid": 1
    }

    unavailable_bot = MarketBot(snapshot_error=RuntimeError("unavailable"))
    assert await market_data.filter_fresh_market_snapshot_creations(
        unavailable_bot, [order]
    ) == []
    unavailable_record = unavailable_bot._fresh_entry_eligibility_trace.to_event_data()[
        "records"
    ][0]
    assert unavailable_record["reason_counts"] == {
        "pre_create_market_snapshot_unavailable": 1
    }

    distance_bot = MarketBot()
    assert await market_data.filter_fresh_market_snapshot_creations(
        distance_bot, [order]
    ) == []
    distance_record = distance_bot._fresh_entry_eligibility_trace.to_event_data()[
        "records"
    ][0]
    assert distance_record["reason_counts"] == {
        "limit_order_create_market_distance": 1
    }


@pytest.mark.asyncio
async def test_eligibility_emitter_failure_cannot_change_connector_batch(monkeypatch):
    order = _initial("BTC/USDT:USDT")
    trace = FreshEntryEligibilityTrace()
    trace.record_ideal_orders([order])
    trace.record_evaluated(order["symbol"], "long")
    bot = _CreateBot(trace)

    monkeypatch.setattr(Passivbot, "_record_emitted_order_custom_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(Passivbot, "_emit_execution_order_event", lambda *args, **kwargs: None)

    def fail_emission(*args, **kwargs):
        raise RuntimeError("diagnostic sink failed")

    monkeypatch.setattr(Passivbot, "_emit_initial_entry_eligibility_event", fail_emission)

    result = await executor.execute_orders_parent(bot, [order])

    assert bot.submitted == [order]
    assert len(result) == 1
    assert bot._fresh_entry_eligibility_trace is None


@pytest.mark.asyncio
async def test_connector_bound_create_attempt_is_counted_once_even_when_ambiguous(
    monkeypatch,
):
    order = _initial("BTC/USDT:USDT")
    bot = _CreateBot(FreshEntryEligibilityTrace())
    bot._order_churn_gate_state = OrderChurnGateState()

    async def fail_batch(_orders):
        raise RuntimeError("ambiguous connector failure")

    bot.execute_orders = fail_batch
    bot.add_to_recent_order_executions = lambda _order: None
    bot._record_order_churn_signed_action_attempts = lambda count: (
        "create-action",
    )
    completed_signed_actions = []
    bot._complete_order_churn_signed_action_attempts = (
        lambda tokens: completed_signed_actions.append(tokens)
    )
    emitted = []
    bot._emit_order_churn_actions_accounted_event = lambda **kwargs: emitted.append(
        kwargs
    )
    monkeypatch.setattr(Passivbot, "_record_emitted_order_custom_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(Passivbot, "_emit_execution_order_event", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="ambiguous connector failure"):
        await executor.execute_orders_parent(bot, [order])

    assert len(bot._order_churn_gate_state.action_attempt_timestamps) == 1
    assert completed_signed_actions == []
    assert emitted == [
        {
            "action_count": 1,
            "action_kind": "create",
            "rolling_count": 1,
            "wave": {"event_id": "ow_1"},
        }
    ]
