"""Rust pair reconstruction parity against independent Decimal examples."""

from dataclasses import asdict, replace
import json
import math
import random

import pytest

from hsl_reference import Fill, Position, reconstruct


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    assert not getattr(pbr, "__is_stub__", False)
    assert hasattr(pbr, "hsl_revised_history"), "rebuild the source-matched Rust extension"
    return pbr


def optional(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def run(pbr, position, fills, prices, start=0, end=300_000):
    encoded = []
    for f in fills:
        row = asdict(f)
        for key in ("delta", "price", "realized", "fee"):
            row[key] = optional(row[key])
        encoded.append(row)
    data = {"start": start, "end": end, "position": asdict(position),
            "fills": encoded, "prices": prices}
    return json.loads(pbr.hsl_revised_history(json.dumps(data, allow_nan=False)))


def compare(pbr, position, fills, prices, start=0, end=300_000):
    expected = reconstruct(position, fills, prices, start, end)
    actual = run(pbr, position, fills, prices, start, end)
    for i, row in enumerate(expected.rows):
        sample = actual["samples"][i]
        assert sample["timestamp"] == row.timestamp
        for key, value in (("pnl", row.pnl), ("upnl", row.upnl),
                           ("size", expected.sizes[i]), ("basis", expected.bases[i])):
            assert sample[key] == pytest.approx(float(value), rel=2e-12, abs=2e-12)
    assert len(actual["samples"]) == len(expected.rows)
    assert [e["realized_cumsum"] for e in actual["events"]] == pytest.approx(
        [float(v) for _, v in expected.cashflows], rel=2e-12, abs=2e-12)
    assert [e["fill"]["identity"] for e in actual["events"]] == [f.identity for f, _ in expected.cashflows]
    # Basis equality may differ by an ulp; it is diagnostic, not a risk veto.
    assert set(actual["reasons"]) - {"current_basis_reconciliation"} == set(expected.reasons) - {"current_basis_reconciliation"}
    return actual


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("inverse", [False, True])
def test_partial_close_missing_opening_and_empty_prices(require_real_passivbot_rust_module, side, inverse):
    d = 1 if side == "long" else -1
    p = Position(d, 100, 80, 10, inverse, side)
    fills = [Fill("add", 60_000, d, 100, 0, -1),
             Fill("partial", 120_000, -d, 90, -10, -1)]
    for tape in (fills, fills[1:], []):
        for prices in ({60_000: 100, 120_000: 90, 180_000: 85}, {}):
            actual = compare(require_real_passivbot_rust_module, p, tape, prices)
            assert actual["samples"][-1]["size"] == d
            assert actual["samples"][-1]["basis"] == 100


@pytest.mark.parametrize("field,value,reason", [
    ("delta", "bad", "invalid_quantity"), ("delta", 0, "invalid_quantity"),
    ("price", None, "estimated_fill_price"), ("price", -1, "estimated_fill_price"),
    ("realized", None, "estimated_realized_pnl"), ("fee", None, "unknown_fee"),
])
def test_invalid_history_preserves_independent_cashflow(require_real_passivbot_rust_module, field, value, reason):
    p = Position(1, 100, 80)
    fills = [Fill("open", 60_000, 2, 100, 0),
             replace(Fill("close", 120_000, -1, 80, -20, -1), **{field: value})]
    actual = compare(require_real_passivbot_rust_module, p, fills, {60_000: 100, 120_000: 80})
    assert reason in actual["reasons"]


def test_clamp_bad_old_episode_and_later_clean_tape(require_real_passivbot_rust_module):
    p = Position(1, 90, 80)
    fills = [Fill("old_add", 1, 2, 100, 0), Fill("old_close", 2, -1, 90, -10),
             Fill("current_open", 60_000, 1, 90, 0)]
    result = compare(require_real_passivbot_rust_module, p, fills, {1: 100, 2: 90, 60_000: 90})
    assert "clamped_quantity" in result["reasons"]
    assert result["events"][-1]["before"] == 0
    assert not result["events"][-1]["quantity_estimated"]


@pytest.mark.parametrize("sequences", [(1, 2), (None, 2), (1, None), (None, None), (1, 1)])
def test_ties_partial_sequences_and_arrival_order(require_real_passivbot_rust_module, sequences):
    p = Position(1, 100, 80)
    fills = [Fill("b", 60_000, 2, 100, 0, sequence=sequences[0]),
             Fill("a", 60_000, -1, 90, -10, sequence=sequences[1])]
    expected = compare(require_real_passivbot_rust_module, p, fills, {60_000: 90})
    assert run(require_real_passivbot_rust_module, p, list(reversed(fills)), {60_000: 90}) == expected


def test_revision_before_clip_and_conflicting_identity(require_real_passivbot_rust_module):
    p = Position(1, 100, 80)
    old = Fill("partial", 60_000, -1, 90, -10)
    fix = replace(old, timestamp=-1, revision=1)
    result = compare(require_real_passivbot_rust_module, p, [old, fix], {})
    assert not result["events"]
    fix = replace(old, revision=1)
    conflict = replace(fix, realized=-20)
    result = compare(require_real_passivbot_rust_module, p, [old, fix, conflict], {})
    assert "conflicting_identity" in result["reasons"]
    repaired = replace(old, realized=-5, revision=2)
    result = compare(require_real_passivbot_rust_module, p, [old, fix, conflict, repaired], {})
    assert result["samples"][-1]["pnl"] == -5


@pytest.mark.parametrize("seed", range(30))
def test_generated_tapes_and_damage_converge_on_repair(require_real_passivbot_rust_module, seed):
    rng = random.Random(seed)
    side = "long" if seed % 2 else "short"
    d = 1 if side == "long" else -1
    quantity = 0
    fills = []
    prices = {}
    for i in range(1, 30):
        delta = rng.randint(1, 5) if quantity == 0 or rng.random() > .5 else -rng.randint(1, quantity)
        quantity += delta
        price = rng.randint(50, 150)
        fills.append(Fill(str(i), i * 60_000, d * delta, price,
                          rng.randint(-20, 20) if delta < 0 else 0, -.1,
                          sequence=i))
        prices[i * 60_000] = price
    p = Position(d * quantity, 100, 90, multiplier=2, inverse=seed % 3 == 0, pside=side)
    full = compare(require_real_passivbot_rust_module, p, fills, prices, end=1_800_000)
    compare(require_real_passivbot_rust_module, p, fills[3:], prices, end=1_800_000)
    damaged = [f for i, f in enumerate(fills) if i % 4]
    compare(require_real_passivbot_rust_module, p, damaged, prices, end=1_800_000)
    assert run(require_real_passivbot_rust_module, p, fills, prices, end=1_800_000) == full


def test_invalid_historical_price_is_omitted_but_endpoint_remains(require_real_passivbot_rust_module):
    result = run(require_real_passivbot_rust_module, Position(1, 100, 80), [], {60_000: -1, 120_000: 0})
    assert "invalid_historical_price" in result["reasons"]
    assert len(result["samples"]) == 1
    assert result["samples"][0]["upnl"] == -20


@pytest.mark.parametrize("patch", [{"mark": 0}, {"basis": 0}, {"multiplier": 0}, {"size": -1}])
def test_invalid_current_inputs_are_errors(require_real_passivbot_rust_module, patch):
    with pytest.raises(ValueError, match="invalid current"):
        run(require_real_passivbot_rust_module, replace(Position(1, 100, 80), **patch), [], {})


@pytest.mark.parametrize("inverse", [False, True])
def test_extreme_breakeven_stays_zero(require_real_passivbot_rust_module, inverse):
    p = Position(1e300, 1e-300, 1e-300, multiplier=1e300, inverse=inverse)
    assert run(require_real_passivbot_rust_module, p, [], {})["samples"][-1]["upnl"] == 0


def test_tiny_inverse_prices_keep_known_loss_direction(require_real_passivbot_rust_module):
    p = Position(1, float.fromhex("0x0.0000000000002p-1022"),
                 float.fromhex("0x0.0000000000001p-1022"), inverse=True)
    result = run(require_real_passivbot_rust_module, p, [], {})
    assert result["samples"][-1]["upnl"] < 0
    assert math.isfinite(result["samples"][-1]["upnl"])
    assert "numeric_range_approximation" in result["reasons"]
