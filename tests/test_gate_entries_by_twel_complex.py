import math
from copy import deepcopy

import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover - exercised when the extension is unavailable
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


def _apply_entries_and_calc_twe(balance, positions, original_entries, gated_entries):
    """
    Helper that mirrors the Rust logic to evaluate the resulting total wallet exposure.
    """

    # Track current absolute position size/price per idx
    pos_state = {
        pos["idx"]: {
            "size": abs(pos["position_size"]),
            "price": pos["position_price"],
            "c_mult": pos["c_mult"],
        }
        for pos in positions
    }

    # Allow matching of multiple entries with identical idx/price
    entries_by_idx = {}
    for entry in original_entries:
        entries_by_idx.setdefault(entry["idx"], []).append(deepcopy(entry))

    for idx, qty, price, _order_type_id in gated_entries:
        candidates = entries_by_idx.get(idx, [])
        if not candidates:
            raise AssertionError(f"Missing original entry metadata for idx {idx}")

        # Match the first candidate with the same price (within rounding tolerance)
        chosen_idx = None
        for i, candidate in enumerate(candidates):
            if math.isclose(candidate["price"], price, rel_tol=0.0, abs_tol=1e-9):
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = 0  # fallback; shouldn't happen but keeps tests resilient
        candidate = candidates.pop(chosen_idx)

        state = pos_state.setdefault(idx, {"size": 0.0, "price": 0.0, "c_mult": candidate["c_mult"]})
        new_size, new_price = pbr.calc_new_psize_pprice(
            state["size"], state["price"], qty, price, candidate["qty_step"]
        )
        state["size"] = new_size
        state["price"] = new_price
        state["c_mult"] = candidate["c_mult"]

    total_we = 0.0
    for state in pos_state.values():
        if state["price"] > 0.0 and state["size"] > 0.0:
            total_we += pbr.calc_wallet_exposure(
                state["c_mult"], balance, state["size"], state["price"]
            )
    return total_we


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available; build and install in venv first",
)
def test_single_entry_is_trimmed_to_respect_twel():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 1.0, "position_price": 100.0, "c_mult": 1.0},  # TWE = 0.1
    ]
    entries = [
        {
            "idx": 0,
            "qty": 150.0,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        }
    ]

    result = pbr.gate_entries_by_twel_py(side, balance, twel, positions, entries)
    assert result, "Expected the entry to be trimmed rather than dropped entirely"
    trimmed_qty = result[0][1]
    assert trimmed_qty < entries[0]["qty"]

    twe_after = _apply_entries_and_calc_twe(balance, positions, entries, result)
    assert twe_after < twel - 1e-12


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available; build and install in venv first",
)
def test_blocks_when_current_twe_at_or_above_limit():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 50.0, "position_price": 100.0, "c_mult": 1.0},  # TWE = 5.0
        {"idx": 1, "position_size": 75.0, "position_price": 100.0, "c_mult": 1.0},  # TWE = 7.5
    ]
    entries = [
        {
            "idx": 0,
            "qty": 50.0,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        },
        {
            "idx": 1,
            "qty": 50.0,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        },
    ]

    result = pbr.gate_entries_by_twel_py(side, balance, twel, positions, entries)
    assert result == [], "Current TWE already exceeds the limit, so all entries should be blocked"


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available; build and install in venv first",
)
def test_prunes_multiple_entries_until_exposure_safe():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 1.0, "position_price": 100.0, "c_mult": 1.0},  # TWE = 0.1
    ]
    entries = [
        {
            "idx": 0,
            "qty": 50.0,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        },
        {
            "idx": 0,
            "qty": 25.0,
            "price": 200.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 200.0,
            "order_type_id": order_type_id,
        },
        {
            "idx": 0,
            "qty": 100.0,
            "price": 50.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 50.0,
            "order_type_id": order_type_id,
        },
    ]

    result = pbr.gate_entries_by_twel_py(side, balance, twel, positions, entries)
    assert len(result) <= len(entries)

    twe_after = _apply_entries_and_calc_twe(balance, positions, entries, result)
    assert twe_after < twel - 1e-12

    total_requested = sum(entry["qty"] for entry in entries)
    total_granted = sum(qty for _, qty, _, _ in result)
    assert total_granted <= total_requested


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available; build and install in venv first",
)
def test_rebalances_quantities_but_keeps_eligible_orders():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 6.0, "position_price": 100.0, "c_mult": 1.0},  # TWE = 0.6
    ]
    entries = [
        {
            "idx": 0,
            "qty": 1.5,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        },
        {
            "idx": 0,
            "qty": 1.5,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        },
        {
            "idx": 0,
            "qty": 1.5,
            "price": 100.0,
            "qty_step": 0.01,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "market_price": 100.0,
            "order_type_id": order_type_id,
        },
    ]

    result = pbr.gate_entries_by_twel_py(side, balance, twel, positions, entries)
    assert len(result) == len(entries), "All three entries remain but one should be trimmed"
    twe_after = _apply_entries_and_calc_twe(balance, positions, entries, result)
    assert twe_after < twel - 1e-12

    trimmed = False
    remaining_pool = {i: [] for i in range(len(entries))}
    for idx, qty, price, _ in result:
        # identify matching original entry
        for entry in entries:
            if entry["idx"] == idx and math.isclose(entry["price"], price, abs_tol=1e-9):
                if qty < entry["qty"] - 1e-9:
                    trimmed = True
                break
    assert trimmed, "At least one of the entries should have a reduced quantity to stay below TWEL"
