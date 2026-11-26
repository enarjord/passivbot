from types import SimpleNamespace

from live import forager_health
from monitor_tui import _render_forager_panel
from passivbot import Passivbot


def test_startup_refresh_accumulates_attempts_per_side_and_latches():
    bot = SimpleNamespace()
    btc_tail = ("BTC/USDT:USDT", "1m")
    eth_tail = ("ETH/USDT:USDT", "1m")
    eth_history = ("ETH/USDT:USDT", "1m_history")
    requirements = {"long": {btc_tail, eth_tail, eth_history}, "short": {btc_tail}}
    Passivbot._record_forager_startup_refresh(bot, requirements, {btc_tail})
    assert bot._forager_startup_ready_sides == {"short"}
    Passivbot._record_forager_startup_refresh(bot, requirements, {eth_tail})
    assert bot._forager_startup_ready_sides == {"short"}
    # The first repair attempt counts even when the full window needs more pages.
    Passivbot._record_forager_startup_refresh(bot, requirements, {eth_history})
    assert bot._forager_startup_ready_sides == {"long", "short"}
    Passivbot._record_forager_startup_refresh(
        bot, {"long": {("SOL/USDT:USDT", "1m")}}, set()
    )
    assert bot._forager_startup_ready_sides == {"long", "short"}


def test_persistent_exclusion_warns_once_per_streak():
    bot = SimpleNamespace()
    exclusion = {"symbol": "BTC/USDT:USDT", "feature": "rms_directionality", "reason": "missing_rms"}
    selection = {
        "excluded_candidates": [exclusion],
        "forager_mode": True,
        "slots_to_fill": 1,
        "candidate_count": 3,
        "eligible_count": 2,
        "selected_symbol_indices": [0],
        "deferral_reason": None,
    }
    count = forager_health.EXCLUSION_WARNING_CYCLES
    for cycle in range(1, count + 2):
        state = forager_health.observe(bot, {"long": selection}, now_ms=cycle)["long"]
        assert state["persistent_exclusions"] == ([exclusion] if cycle == count else [])
    forager_health.observe(bot, {}, now_ms=count + 2)
    state = forager_health.observe(bot, {"long": selection}, now_ms=count + 3)["long"]
    assert state["persistent_exclusions"] == []


def test_entry_stall_warns_and_clears_on_a_working_initial_order():
    symbol = "BTC/USDT:USDT"
    bot = SimpleNamespace(
        config={"logging": {"forager_no_entries_warning_minutes": 1.0}},
        open_orders={},
    )
    selection = {
        "excluded_candidates": [],
        "forager_mode": True,
        "slots_to_fill": 1,
        "candidate_count": 3,
        "eligible_count": 2,
        "selected_symbol_indices": [0],
        "deferral_reason": None,
    }
    for now_ms in (0, 59_999, 60_000, 60_001):
        state = forager_health.observe(bot, {"long": selection}, now_ms=now_ms)["long"]
        assert state["stall_warning"] == (now_ms >= 60_000)
        assert state["stall_warning_due"] == (now_ms == 60_000)
    bot.open_orders[symbol] = [{"position_side": "long", "side": "buy"}]
    state = forager_health.observe(bot, {"long": selection}, now_ms=60_002)["long"]
    assert state["stall_ms"] == 0
    assert not state["stall_warning"]
    bot.open_orders.clear()
    state = forager_health.observe(bot, {"long": selection}, now_ms=120_003)["long"]
    assert state["stall_ms"] == 0
    selection["forager_mode"] = False
    state = forager_health.observe(bot, {"long": selection}, now_ms=180_004)["long"]
    assert not state["stall_warning"]


def test_entry_stall_repeats_when_all_candidates_are_excluded():
    bot = SimpleNamespace(config={"logging": {"forager_no_entries_warning_minutes": 1.0}})
    selection = {
        "excluded_candidates": [{
            "symbol": "BTC/USDT:USDT", "feature": "rms_directionality", "reason": "missing_rms",
        }],
        "forager_mode": True,
        "slots_to_fill": 1,
        "candidate_count": 1,
        "eligible_count": 0,
        "selected_symbol_indices": [],
        "deferral_reason": None,
    }
    for now_ms in (0, 59_999, 60_000, 60_001, 120_000):
        state = forager_health.observe(bot, {"long": selection}, now_ms=now_ms)["long"]
        assert state["stall_warning_due"] == (now_ms in (60_000, 120_000))


def test_entry_stall_counts_from_end_of_intentional_cooldown():
    bot = SimpleNamespace(config={"logging": {"forager_no_entries_warning_minutes": 1.0}})
    selection = {
        "excluded_candidates": [],
        "forager_mode": True,
        "slots_to_fill": 2,
        "candidate_count": 2,
        "eligible_count": 2,
        "selected_symbol_indices": [0, 1],
        "deferral_reason": None,
    }
    for now_ms in (0, 60_000, 120_000):
        state = forager_health.observe(
            bot, {"long": selection}, now_ms=now_ms, active_entry_cooldowns={(0, "long"), (1, "long")}
        )["long"]
        assert not state["stall_warning"]
    for now_ms in (120_001, 180_000, 180_001):
        state = forager_health.observe(
            bot, {"long": selection}, now_ms=now_ms, active_entry_cooldowns={(1, "long")}
        )["long"]
        assert state["stall_warning_due"] == (now_ms == 180_001)


def test_forager_monitor_renders_entry_stall():
    lines = _render_forager_panel({"forager": {"long": {
        "enabled": True,
        "entry_stall": {"duration_ms": 120_000, "reason": "no_entries_with_free_slots"},
    }}})
    assert any("free slots with no working entry for 2.0m" in line for line in lines)
