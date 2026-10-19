from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from live.realized_facts import RealizedFactsError, canonical_realized_facts


def fill(identity, net=0.0, *, timestamp=60_000, symbol="A/USDT:USDT", pside="long",
         side="buy", qty=1.0, price=100.0, start=None):
    result = dict(id=identity, symbol=symbol, position_side=pside, timestamp=timestamp,
                  pnl=net, fee_paid=0.0, side=side, qty=qty, price=price, raw=[])
    if start is not None:
        result["raw"] = [{"source": "fetch_my_trades", "data": {
            "id": identity, "side": side, "amount": abs(qty), "price": price,
            "info": {"startPosition": str(start)},
        }}]
    return result


def ids(result):
    return [json.loads(row["identity"])[-1] for row in result.fills]


def drawdown(rows):
    result = 0.0
    for row in rows:
        result = max(0.0, result - row["net"])
    return result


def test_permutation_stable_account_order_and_individual_clipping():
    rows = [fill("a-profit", 5.0), fill("z-loss", -10.0),
            fill("earlier", -2.0, timestamp=0)]
    expected = canonical_realized_facts(rows)
    for order in permutations(rows):
        assert canonical_realized_facts(order) == expected
    assert ids(expected) == ["earlier", "a-profit", "z-loss"]
    assert expected.reasons == ("ambiguous_same_time_realized_order",)
    assert expected.ambiguous_cohort_count == 1
    assert len(expected.fills) == 3
    assert drawdown(expected.fills) == 10.0
    # Netting the same economic cashflows would erase the profit-side peak.
    assert max(0.0, -sum(row["net"] for row in expected.fills)) == 7.0


@pytest.mark.parametrize("pside,first_side,next_side", [
    ("long", "sell", "buy"), ("short", "buy", "sell"),
])
def test_proven_raw_pair_chain_wins_over_identity_and_input_order(pside, first_side, next_side):
    rows = [fill("z-first", -10.0, pside=pside, side=first_side, qty=1.0, start=3.0),
            fill("a-next", 5.0, pside=pside, side=next_side, qty=2.0, start=2.0)]
    for order in permutations(rows):
        result = canonical_realized_facts(SimpleNamespace(**row) for row in order)
        assert ids(result) == ["z-first", "a-next"]
        assert result.reasons == ()
        assert result.proven_pair_chain_count == 1
        assert drawdown(result.fills) == 5.0


def test_multiple_pair_ties_disclose_unknown_interleaving_and_preserve_chain():
    rows = [fill("z-first", -10.0, side="sell", qty=1.0, start=3.0),
            fill("a-next", 5.0, side="buy", qty=2.0, start=2.0),
            fill("b-other", 2.0, symbol="B/USDT:USDT"),
            fill("s-other", -1.0, pside="short", side="sell")]
    expected = canonical_realized_facts(rows)
    for order in permutations(rows):
        assert canonical_realized_facts(order) == expected
    assert ids(expected) == ["z-first", "a-next", "s-other", "b-other"]
    assert expected.proven_pair_chain_count == 1
    assert expected.ambiguous_cohort_count == 1
    assert expected.reasons == ("ambiguous_same_time_realized_order",)


def test_ambiguous_raw_cycle_uses_disclosed_stable_fallback():
    rows = [fill("z", -10.0, side="sell", qty=1.0, start=2.0),
            fill("a", 5.0, side="buy", qty=1.0, start=1.0)]
    a = canonical_realized_facts(rows)
    b = canonical_realized_facts(reversed(rows))
    assert a == b
    assert ids(a) == ["a", "z"]
    assert a.proven_pair_chain_count == 0
    assert a.reasons == ("ambiguous_same_time_realized_order",)


def test_derived_after_state_never_proves_order_and_inputs_are_untouched():
    rows = [fill("z", -10.0), fill("a", 5.0)]
    rows[0].update(psize=1.0, pprice=100.0)
    rows[1].update(psize=2.0, pprice=100.0)
    original = deepcopy(rows)
    result = canonical_realized_facts(rows)
    assert rows == original
    assert ids(result) == ["a", "z"]
    assert result.proven_pair_chain_count == 0
    assert result.ambiguous_cohort_count == 1


def test_duplicate_risk_equivalent_identity_collapses_once():
    one = fill("one", -3.0)
    duplicate = deepcopy(one)
    duplicate.update(psize=999.0, provenance={"unrelated": "metadata"})
    result = canonical_realized_facts([one, duplicate])
    assert ids(result) == ["one"]
    assert result.duplicate_count == 1
    assert result.reasons == ("duplicate_realized_fill_collapsed",)
    assert drawdown(result.fills) == 3.0


@pytest.mark.parametrize("change", [
    {"pnl": -4.0}, {"pnl": -4.0, "fee_paid": 1.0}, {"fee_paid": -1.0},
    {"timestamp": 60_001}, {"qty": 2.0}, {"price": 101.0}, {"side": "sell"},
])
def test_conflicting_duplicate_identity_rejects_every_input_order(change):
    one = fill("one", -3.0)
    changed = dict(one, **change)
    for rows in ([one, changed], [changed, one]):
        with pytest.raises(RealizedFactsError, match="conflicting realized fill identity"):
            canonical_realized_facts(rows)


def test_conflicting_independent_chain_evidence_is_not_merged():
    one = fill("one", -3.0, start=1.0)
    changed = fill("one", -3.0, start=2.0)
    with pytest.raises(RealizedFactsError, match="conflicting"):
        canonical_realized_facts([one, changed])


def test_exchange_id_can_recur_on_different_pairs_without_deduplication():
    result = canonical_realized_facts([fill("same", -1.0), fill("same", -2.0, pside="short")])
    assert len(result.fills) == 2
    assert result.duplicate_count == 0
    assert len({row["identity"] for row in result.fills}) == 2
    assert result.ambiguous_cohort_count == 1


@pytest.mark.parametrize("field,value", [
    ("pnl", True), ("pnl", float("nan")), ("fee_paid", float("inf")),
    ("timestamp", True), ("timestamp", 1.5), ("timestamp", -1),
    ("position_side", "buy"), ("id", ""), ("symbol", " A "),
])
def test_malformed_required_facts_reject(field, value):
    event = fill("one")
    event[field] = value
    with pytest.raises(RealizedFactsError):
        canonical_realized_facts([event])


def test_missing_fee_is_not_a_neutral_fee_and_net_overflow_rejects():
    event = fill("one")
    del event["fee_paid"]
    with pytest.raises(RealizedFactsError, match="missing fee_paid"):
        canonical_realized_facts([event])
    event.update(pnl=sys.float_info.max, fee_paid=sys.float_info.max)
    with pytest.raises(RealizedFactsError, match="net PnL"):
        canonical_realized_facts([event])


def test_empty_and_no_chain_paths_need_no_native_or_exchange_modules():
    src = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import importlib.abc, sys
sys.path.insert(0,sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fill_events_manager','passivbot_rust','ccxt','numpy'}:
            raise AssertionError('unexpected dependency '+fullname)
sys.meta_path.insert(0,Block())
from live.realized_facts import canonical_realized_facts
assert not canonical_realized_facts([]).fills
rows=[dict(id='b',symbol='A',position_side='long',timestamp=1,pnl=-1,fee_paid=0),
      dict(id='a',symbol='A',position_side='long',timestamp=1,pnl=1,fee_paid=0)]
assert canonical_realized_facts(rows).ambiguous_cohort_count==1
"""
    subprocess.run([sys.executable, "-I", "-c", script, src], check=True, capture_output=True, text=True)
