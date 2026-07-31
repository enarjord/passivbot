from freshness_ledger import FreshnessLedger


def test_freshness_ledger_tracks_current_epoch_surface_changes():
    ledger = FreshnessLedger(now_ms=1000)

    ledger.begin_epoch(now_ms=1100)
    changed = ledger.stamp("positions", (("BTC", "long", 0.1),), now_ms=1200)
    unchanged = ledger.stamp("positions", (("BTC", "long", 0.1),), now_ms=1300)
    changed_again = ledger.stamp("positions", (("BTC", "long", 0.2),), now_ms=1400)

    state = ledger.surfaces["positions"]
    assert changed is True
    assert unchanged is False
    assert changed_again is True
    assert state.updated_ms == 1400
    assert state.epoch == 1
    assert ledger.surfaces_at_epoch() == {"positions"}
    assert ledger.changed_surfaces_at_epoch() == {"positions"}

    ledger.begin_epoch(now_ms=1500)
    ledger.stamp("positions", (("BTC", "long", 0.2),), now_ms=1600)
    ledger.stamp("open_orders", (), now_ms=1700)

    assert ledger.surfaces_at_epoch() == {"positions", "open_orders"}
    assert ledger.changed_surfaces_at_epoch() == {"open_orders"}
    assert ledger.changed_surfaces_at_epoch(1) == {"positions"}

    ledger.begin_epoch(now_ms=1800)
    ledger.stamp("positions", (("BTC", "long", 0.3),), now_ms=1900)
    ledger.stamp("positions", (("BTC", "long", 0.3),), now_ms=2000)

    assert ledger.surfaces_at_epoch() == {"positions"}
    assert ledger.changed_surfaces_at_epoch() == {"positions"}
