import pytest
import passivbot_rust as pbr


def _pb_custom_id(order_type: str, suffix: str) -> str:
    return f"pb-0x{pbr.order_type_snake_to_id(order_type):04x}-{suffix}"


@pytest.mark.asyncio
async def test_execute_orders_parent_tracks_acknowledged_custom_id():
    import passivbot as pb_mod

    class FakeBot:
        execute_orders_parent = pb_mod.Passivbot.execute_orders_parent
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event
        _monitor_order_payload = pb_mod.Passivbot._monitor_order_payload
        _record_emitted_order_custom_id = pb_mod.Passivbot._record_emitted_order_custom_id
        _extract_order_custom_id = pb_mod.Passivbot._extract_order_custom_id
        _extract_order_exchange_id = pb_mod.Passivbot._extract_order_exchange_id
        _extract_order_reduce_only = pb_mod.Passivbot._extract_order_reduce_only
        _extract_order_float = pb_mod.Passivbot._extract_order_float
        _canonical_passivbot_custom_id = pb_mod.Passivbot._canonical_passivbot_custom_id
        _order_identity_fingerprint = pb_mod.Passivbot._order_identity_fingerprint
        _build_emitted_order_record = pb_mod.Passivbot._build_emitted_order_record
        _emitted_order_records = pb_mod.Passivbot._emitted_order_records

        def __init__(self):
            self.monitor_publisher = None
            self._health_orders_placed = 0
            self.debug_mode = False
            self.orders_emitted_to_exchange = []

        def live_value(self, key):
            assert key == "max_n_creations_per_batch"
            return 5

        def get_exchange_time(self):
            return 123456

        def add_to_recent_order_executions(self, order):
            return None

        def log_order_action(self, *args, **kwargs):
            return None

        def _log_order_action_summary(self, *args, **kwargs):
            return None

        async def execute_orders(self, orders):
            return [{"id": "abc123", **orders[0]}]

        def did_create_order(self, executed):
            return True

        def add_new_order(self, order, source="POST"):
            return None

        def _resolve_pb_order_type(self, order):
            return "entry_grid_normal_long"

    bot = FakeBot()
    custom_id = _pb_custom_id("entry_grid_normal_long", "aaaa")
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "price": 100000.0,
        "reduce_only": False,
        "custom_id": custom_id,
    }

    res = await pb_mod.Passivbot.execute_orders_parent(bot, [order])

    assert len(res) == 1
    assert len(bot.orders_emitted_to_exchange) == 1
    record = bot.orders_emitted_to_exchange[0]
    assert record["timestamp"] == 123456
    assert record["exchange_id"] == "abc123"
    assert record["custom_id"] == custom_id
    assert record["canonical_custom_id"] == "0x0004-aaaa"
    assert record["pb_type"] == "entry_grid_normal_long"
    assert record["fingerprint"] == {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "reduce_only": False,
        "pb_type": "entry_grid_normal_long",
        "qty": 0.01,
        "price": 100000.0,
    }


def _make_detection_bot(now_ts: int, start_ts: int):
    import passivbot as pb_mod

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.orders_emitted_to_exchange = []
    bot.foreign_passivbot_seen = {}
    bot._foreign_passivbot_stop_requested = False
    bot.stop_signal_received = False
    bot.bot_start_exchange_ts = start_ts
    bot.get_exchange_time = lambda: now_ts
    return bot


@pytest.mark.asyncio
async def test_apply_open_orders_snapshot_runs_foreign_writer_detector():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=2_000_000, start_ts=1_000_000)
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.state_change_detected_by_symbol = set()
    bot._record_authoritative_surface = lambda *_args, **_kwargs: None
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: False
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.log_order_action = lambda *_args, **_kwargs: None
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    foreign_order = {
        "id": "foreign-1",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "amount": 0.01,
        "price": 99_000.0,
        "reduce_only": False,
        "timestamp": 1_020_000,
        "custom_id": _pb_custom_id("entry_grid_normal_long", "foreign"),
    }

    ok = await pb_mod.Passivbot._apply_open_orders_snapshot(bot, [foreign_order])

    assert ok is True
    assert len(bot.foreign_passivbot_seen) == 1
    assert bot.open_orders["BTC/USDT:USDT"] == [foreign_order]
    assert bot.stop_signal_received is False


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_ignores_manual_and_prestart_orders():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=1_100_000, start_ts=1_000_000)
    stale_custom_id = _pb_custom_id("entry_grid_normal_long", "stale")
    bot.orders_emitted_to_exchange = [
        {
            "timestamp": 1_100_000 - pb_mod.FOREIGN_PASSIVBOT_LOOKBACK_MS - 1,
            "exchange_id": "",
            "custom_id": stale_custom_id,
            "canonical_custom_id": stale_custom_id,
            "pb_type": "entry_grid_normal_long",
            "fingerprint": None,
        }
    ]
    orders = [
        {
            "id": "1",
            "symbol": "BTC/USDT:USDT",
            "timestamp": 1_000_000 - 1,
            "custom_id": _pb_custom_id("entry_grid_normal_long", "old"),
        },
        {
            "id": "2",
            "symbol": "BTC/USDT:USDT",
            "timestamp": 1_020_000,
            "custom_id": "manual-order-123",
        },
        {
            "id": "3",
            "symbol": "BTC/USDT:USDT",
            "timestamp": 1_020_000,
            "custom_id": "0004-manual",
        },
        {
            "id": "4",
            "symbol": "BTC/USDT:USDT",
            "timestamp": 1_020_000,
        },
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, orders)

    assert bot.foreign_passivbot_seen == {}
    assert bot.orders_emitted_to_exchange == []
    assert bot.stop_signal_received is False


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_accepts_gateio_prefixed_self_order():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=2_000_000, start_ts=1_000_000)
    custom_id = _pb_custom_id("entry_grid_normal_long", "gateio")
    emitted = {
        "id": "gateio-order-1",
        "symbol": "SUI/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 8.0,
        "price": 0.928,
        "reduce_only": False,
        "custom_id": custom_id,
    }
    pb_mod.Passivbot._record_emitted_order_custom_id(bot, emitted, emitted_ts=1_990_000)

    fetched_open = [
        {
            "id": "gateio-order-1",
            "symbol": "SUI/USDT:USDT",
            "side": "buy",
            "position_side": "long",
            "qty": 8.0,
            "price": 0.928,
            "reduceOnly": False,
            "timestamp": 1_990_500,
            "custom_id": f"t-{custom_id}",
            "info": {"text": f"t-{custom_id}"},
        }
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, fetched_open)

    assert bot.foreign_passivbot_seen == {}
    assert bot.stop_signal_received is False


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_accepts_canonical_custom_id_match():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=2_000_000, start_ts=1_000_000)
    custom_id = _pb_custom_id("entry_grid_normal_long", "canon")
    emitted = {
        "id": "ack-id",
        "symbol": "DOT/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 7.0,
        "price": 1.233,
        "reduce_only": False,
        "custom_id": custom_id,
    }
    pb_mod.Passivbot._record_emitted_order_custom_id(bot, emitted, emitted_ts=1_990_000)
    fetched_open = [
        {
            "id": "different-fetch-id",
            "symbol": "DOT/USDT:USDT",
            "side": "buy",
            "position_side": "long",
            "qty": 7.0,
            "price": 1.233,
            "reduceOnly": False,
            "timestamp": 1_990_500,
            "custom_id": f"t-{custom_id}",
        }
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, fetched_open)

    assert bot.foreign_passivbot_seen == {}
    assert bot.stop_signal_received is False


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_accepts_recent_fingerprint_when_ids_missing():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=2_000_000, start_ts=1_000_000)
    pb_type = "entry_grid_normal_long"
    emitted = {
        "symbol": "DOT/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 17.0,
        "price": 1.214,
        "reduce_only": False,
        "pb_order_type": pb_type,
    }
    pb_mod.Passivbot._record_emitted_order_custom_id(bot, emitted, emitted_ts=1_990_000)
    fetched_open = [
        {
            "symbol": "DOT/USDT:USDT",
            "side": "buy",
            "position_side": "long",
            "qty": 17.0,
            "price": 1.214,
            "reduceOnly": False,
            "timestamp": 1_990_500,
            "custom_id": _pb_custom_id(pb_type, "open-only"),
        }
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, fetched_open)

    assert bot.foreign_passivbot_seen == {}
    assert bot.stop_signal_received is False


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_rejects_conflicting_custom_id_despite_fingerprint():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=2_000_000, start_ts=1_000_000)
    emitted = {
        "symbol": "DOT/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 17.0,
        "price": 1.214,
        "reduce_only": False,
        "custom_id": _pb_custom_id("entry_grid_normal_long", "ours"),
    }
    pb_mod.Passivbot._record_emitted_order_custom_id(bot, emitted, emitted_ts=1_990_000)
    fetched_open = [
        {
            "symbol": "DOT/USDT:USDT",
            "side": "buy",
            "position_side": "long",
            "qty": 17.0,
            "price": 1.214,
            "reduceOnly": False,
            "timestamp": 1_990_500,
            "custom_id": _pb_custom_id("entry_grid_normal_long", "foreign"),
        }
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, fetched_open)

    assert len(bot.foreign_passivbot_seen) == 1
    assert bot.stop_signal_received is False


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_stops_after_unique_threshold():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=2_000_000, start_ts=1_000_000)
    first_two = [
        {
            "id": "1",
            "symbol": "BTC/USDT:USDT",
            "timestamp": 1_020_000,
            "custom_id": _pb_custom_id("entry_grid_normal_long", "1111"),
        },
        {
            "id": "2",
            "symbol": "ETH/USDT:USDT",
            "timestamp": 1_030_000,
            "custom_id": _pb_custom_id("close_grid_long", "2222"),
        },
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, first_two)

    assert len(bot.foreign_passivbot_seen) == 2
    assert bot.stop_signal_received is False

    third = [
        {
            "id": "3",
            "symbol": "SOL/USDT:USDT",
            "timestamp": 1_040_000,
            "custom_id": _pb_custom_id("entry_initial_normal_long", "3333"),
        }
    ]

    with pytest.raises(Exception, match="foreign Passivbot writer detected"):
        await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, third)

    assert len(bot.foreign_passivbot_seen) == pb_mod.FOREIGN_PASSIVBOT_MAX_UNIQUE_PER_WINDOW
    assert bot.stop_signal_received is True
