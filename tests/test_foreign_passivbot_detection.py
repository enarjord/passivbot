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

        def __init__(self):
            self.monitor_publisher = None
            self._health_orders_placed = 0
            self.debug_mode = False
            self.orders_emitted_to_exchange = {}

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
    assert bot.orders_emitted_to_exchange == {custom_id: 123456}


def _make_detection_bot(now_ts: int, start_ts: int):
    import passivbot as pb_mod

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.orders_emitted_to_exchange = {}
    bot.foreign_passivbot_seen = {}
    bot._foreign_passivbot_stop_requested = False
    bot.stop_signal_received = False
    bot.bot_start_exchange_ts = start_ts
    bot.get_exchange_time = lambda: now_ts
    return bot


@pytest.mark.asyncio
async def test_detect_foreign_passivbot_orders_ignores_manual_and_prestart_orders():
    import passivbot as pb_mod

    bot = _make_detection_bot(now_ts=1_100_000, start_ts=1_000_000)
    stale_custom_id = _pb_custom_id("entry_grid_normal_long", "stale")
    bot.orders_emitted_to_exchange = {
        stale_custom_id: 1_100_000 - pb_mod.FOREIGN_PASSIVBOT_LOOKBACK_MS - 1
    }
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
        },
    ]

    await pb_mod.Passivbot._detect_foreign_passivbot_orders(bot, orders)

    assert bot.foreign_passivbot_seen == {}
    assert bot.orders_emitted_to_exchange == {}
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
