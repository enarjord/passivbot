from freshness_ledger import ACCOUNT_SURFACES, FreshnessLedger


def test_freshness_ledger_tracks_surface_generations():
    ledger = FreshnessLedger(now_ms=1000)

    ledger.begin_epoch(now_ms=1100)
    changed = ledger.stamp("positions", (("BTC", "long", 0.1),), now_ms=1200)
    unchanged = ledger.stamp("positions", (("BTC", "long", 0.1),), now_ms=1300)
    changed_again = ledger.stamp("positions", (("BTC", "long", 0.2),), now_ms=1400)

    state = ledger.surfaces["positions"]
    assert changed is True
    assert unchanged is False
    assert changed_again is True
    assert state.generation == 2
    assert state.updated_ms == 1400
    assert state.epoch == 1


def test_symbol_block_clears_only_after_required_surfaces_reach_min_epoch():
    ledger = FreshnessLedger(now_ms=1000)
    ledger.begin_epoch(now_ms=1100)
    ledger.stamp("positions", ("old",), now_ms=1200)

    ledger.flag_symbol_block(
        "BTC/USDT:USDT",
        reason="self_order_disappeared_position_may_be_stale",
        required_surfaces=ACCOUNT_SURFACES,
        min_epoch=2,
        detected_ms=1300,
    )

    assert set(ledger.blocked_symbols()) == {"BTC/USDT:USDT"}
    assert ledger.surfaces_missing_after(ACCOUNT_SURFACES, 2) == [
        "balance",
        "fills",
        "open_orders",
        "positions",
    ]

    ledger.begin_epoch(now_ms=1400)
    for surface in ("balance", "positions", "open_orders"):
        ledger.stamp(surface, surface, now_ms=1500)

    assert set(ledger.blocked_symbols()) == {"BTC/USDT:USDT"}
    assert ledger.surfaces_missing_after(ACCOUNT_SURFACES, 2) == ["fills"]

    ledger.stamp("fills", "fills", now_ms=1600)

    assert ledger.blocked_symbols() == {}
