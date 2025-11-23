import math
from copy import deepcopy

import pytest

import passivbot_rust as pbr


def _apply_entries_and_calc_twe(balance, positions, original_entries, gated_entries):
    pos_state = {
        pos["idx"]: {
            "size": abs(pos["position_size"]),
            "price": pos["position_price"],
            "c_mult": pos["c_mult"],
        }
        for pos in positions
    }

    entries_by_idx = {}
    for entry in original_entries:
        entries_by_idx.setdefault(entry["idx"], []).append(deepcopy(entry))

    for idx, qty, price, _order_type_id in gated_entries:
        candidates = entries_by_idx.get(idx, [])
        if not candidates:
            raise AssertionError(f"Missing original entry metadata for idx {idx}")
        match_idx = None
        for i, candidate in enumerate(candidates):
            if math.isclose(candidate["price"], price, rel_tol=0.0, abs_tol=1e-9):
                match_idx = i
                break
        if match_idx is None:
            match_idx = 0
        candidate = candidates.pop(match_idx)

        state = pos_state.setdefault(
            idx, {"size": 0.0, "price": 0.0, "c_mult": candidate["c_mult"]}
        )
        new_size, new_price = pbr.calc_new_psize_pprice(
            state["size"], state["price"], qty, price, candidate["qty_step"]
        )
        state["size"] = new_size
        state["price"] = new_price
        state["c_mult"] = candidate["c_mult"]

    total = 0.0
    for state in pos_state.values():
        if state["price"] > 0.0 and state["size"] > 0.0:
            total += pbr.calc_wallet_exposure(
                state["c_mult"], balance, state["size"], state["price"]
            )
    return total


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available; build and install in venv first")
def test_single_entry_is_trimmed_to_respect_twel():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 1.0, "position_price": 100.0, "c_mult": 1.0},
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
    assert result
    trimmed_qty = result[0][1]
    assert trimmed_qty < entries[0]["qty"]

    twe_after = _apply_entries_and_calc_twe(balance, positions, entries, result)
    assert twe_after < twel - 1e-12


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available; build and install in venv first")
def test_blocks_when_current_twe_at_or_above_limit():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 50.0, "position_price": 100.0, "c_mult": 1.0},
        {"idx": 1, "position_size": 75.0, "position_price": 100.0, "c_mult": 1.0},
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
    assert result == []


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available; build and install in venv first")
def test_prunes_multiple_entries_until_exposure_safe():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 1.0, "position_price": 100.0, "c_mult": 1.0},
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
    twe_after = _apply_entries_and_calc_twe(balance, positions, entries, result)
    assert twe_after < twel - 1e-12

    assert len(result) <= len(entries)
    total_requested = sum(entry["qty"] for entry in entries)
    total_granted = sum(qty for _, qty, _, _ in result)
    assert total_granted <= total_requested


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available; build and install in venv first")
def test_rebalances_quantities_but_keeps_eligible_orders():
    side = "long"
    balance = 1000.0
    twel = 1.0
    order_type_id = pbr.order_type_snake_to_id("entry_grid_normal_long")

    positions = [
        {"idx": 0, "position_size": 6.0, "position_price": 100.0, "c_mult": 1.0},
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
    twe_after = _apply_entries_and_calc_twe(balance, positions, entries, result)
    assert twe_after < twel - 1e-12

    trimmed = False
    for idx, qty, price, _ in result:
        for entry in entries:
            if entry["idx"] == idx and math.isclose(entry["price"], price, abs_tol=1e-9):
                if qty < entry["qty"] - 1e-9:
                    trimmed = True
                break
    assert trimmed
