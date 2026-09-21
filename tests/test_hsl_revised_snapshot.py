"""Real-extension parity for immutable snapshot and scope-boundary reconstruction."""
from dataclasses import asdict, replace
import json
from itertools import permutations

import pytest

import test_hsl_reference_replay as cases
import test_hsl_reference_replay_fake_exchange as fake_cases
from hsl_reference import Fill, dec, ordered_fills, reconstruct
from hsl_reference_replay import scope_boundaries as reference_boundaries, causal_fills, selected_pairs


def number(value):
    if value is None:
        return None
    try:
        return float(dec(value))
    except (ValueError, ArithmeticError):
        return None


def fill_payload(fill):
    row = asdict(fill)
    for key in ("delta", "price", "realized", "fee"):
        row[key] = number(row[key])
    return row


def payload(snapshot, mode="unified", *, quantity_step=None, **kwargs):
    pairs = []
    for p in snapshot.pairs:
        position = asdict(p.position)
        position["quantity_step"] = quantity_step
        for k in ("size", "basis", "mark", "multiplier"):
            position[k] = number(position[k])
        anchor = None
        if p.fills_position_anchor is not None:
            anchor = dict(zip(("position_at", "size", "basis", "multiplier", "inverse", "pside", "revision"), p.fills_position_anchor))
            for k in ("size", "basis", "multiplier"):
                anchor[k] = number(anchor[k])
        pairs.append(dict(symbol=p.symbol, position=position, position_at=p.position_at, mark_at=p.mark_at,
                          fills_started_at=p.fills_started_at, fills_at=p.fills_at, prices_at=p.prices_at,
                          fills=[fill_payload(f) for f in p.fills], prices={str(t): float(x) for t,x in p.prices},
                          revisions=p.revisions, fills_position_anchor=anchor))
    return dict(now=snapshot.now, start=snapshot.start, balance=float(snapshot.balance),
                balance_at=snapshot.balance_at, config_at=snapshot.config_at, max_current_age_ms=120_000,
                mode=mode, pside=kwargs.get("pside"), symbol=kwargs.get("symbol"), pairs=pairs)


def rust(value):
    import passivbot_rust as pbr
    assert hasattr(pbr, "hsl_revised_snapshot"), "rebuild the source-matched extension"
    return json.loads(pbr.hsl_revised_snapshot(json.dumps(value, allow_nan=False)))


def compare(snapshot, mode, *, quantity_step=None, **kwargs):
    request = payload(snapshot, mode, quantity_step=quantity_step, **kwargs)
    try:
        expected = reference_boundaries(snapshot, mode, **kwargs)
    except ValueError:
        with pytest.raises(ValueError):
            rust(request)
        raise
    actual = rust(request)
    assert len(actual["boundaries"]) == len(expected.boundaries)
    assert expected.reasons <= set(actual["reasons"])
    by_key = {(p["symbol"], p["pside"]): p for p in actual["pairs"]}
    for pair in selected_pairs(snapshot, mode, **kwargs):
        direction = 1 if pair.position.pside == "long" else -1
        fills, _ = ordered_fills(causal_fills(pair, snapshot.now), snapshot.start, snapshot.now, direction)
        limit = min(pair.position_at, pair.fills_at if pair.fills_at is not None else snapshot.now)
        prices = {t:p for t,p in pair.prices if snapshot.start <= t <= min(snapshot.now, pair.prices_at)}
        expected_history = reconstruct(pair.position, [f for f in fills if f.timestamp <= limit],
                                       prices, snapshot.start, snapshot.now)
        got_history = by_key[pair.key]["history"]
        assert len(got_history["samples"]) == len(expected_history.rows)
        for got, row, size, basis in zip(got_history["samples"], expected_history.rows,
                                         expected_history.sizes, expected_history.bases):
            assert got["timestamp"] == row.timestamp
            for field, value in (("pnl", row.pnl), ("upnl", row.upnl), ("size", size), ("basis", basis)):
                assert got[field] == pytest.approx(float(value))
    for a, e in zip(actual["boundaries"], expected.boundaries):
        assert a["timestamp"] == e.timestamp
        assert a["lifecycle_eligible"] == e.lifecycle_eligible
        assert a["pnl"] == pytest.approx(float(e.observation.pnl))
        assert a["upnl"] == 0
        for consumed, (key, fills) in zip(a["consumed"], e.consumed):
            assert (consumed["symbol"], consumed["pside"]) == key
            assert consumed["count"] == len(fills)
            for got, original in zip(by_key[key]["fills"][:consumed["count"]], fills):
                for field, value in fill_payload(original).items():
                    if field in ("delta", "price", "realized", "fee") and value is not None:
                        assert got[field] == pytest.approx(value, rel=1e-14, abs=1e-14)
                    else:
                        assert got[field] == value
    return expected


SIMPLE_CASES = [
    "flatten_signal_drives_real_cooldown_anchor_before_reopening",
    "sequenced_same_millisecond_flats_keep_distinct_cashflow_prefixes",
    "opposite_signed_exposures_are_not_a_flat_portfolio",
    "roundtrip_wholly_within_unsequenced_cohort_has_final_flat_only",
    "unsequenced_cohort_cannot_prove_an_internal_flat_then_reopen",
    "old_missing_opening_does_not_poison_later_flat",
    "clamp_cannot_create_boundary_but_does_not_poison_clean_suffix",
    "unknown_quantity_after_candidate_does_not_certify_flat",
    "conflicting_prefix_is_local_and_repair_rebuilds_boundaries",
    "missing_flat_fill_does_not_invent_timestamp_from_current_flat",
    "missing_final_reduction_cannot_promote_an_earlier_partial_close_to_flat",
    "exact_window_edge_and_empty_restart_reproduce_same_boundary",
    "cross_pair_boundary_cannot_postdate_an_older_position_anchor",
    "close_after_position_anchor_is_disclosed_until_positions_catch_up",
    "older_fill_capture_cannot_certify_flat_against_newer_position",
    "conflicting_post_position_variants_cannot_certify_an_older_flat",
    "same_timestamp_tail_execution_is_uncertain_until_later_position_observation",
    "boundary_selector_requires_explicit_current_coin_observation",
    "equal_clock_causal_fill_proof_is_bound_to_the_observed_position",
]


@pytest.mark.parametrize("name", SIMPLE_CASES)
def test_reference_boundary_cases_against_rust(name, monkeypatch):
    monkeypatch.setattr(cases, "scope_boundaries", compare)
    getattr(cases, "test_" + name)()


@pytest.mark.parametrize("mode,kwargs", [("coin", dict(pside="long", symbol="A")), ("pside", dict(pside="long")), ("unified", {})])
def test_scope_mapping(mode, kwargs, monkeypatch):
    monkeypatch.setattr(cases, "scope_boundaries", compare)
    cases.test_partial_close_is_not_flat_and_flat_cashflow_includes_fees(mode, kwargs)


@pytest.mark.parametrize("other_side", ["long", "short"])
def test_cross_pair_cohorts_and_scope_isolation(other_side, monkeypatch):
    monkeypatch.setattr(cases, "scope_boundaries", compare)
    cases.test_cross_pair_close_open_cohort_does_not_invent_portfolio_flat(other_side)
    cases.test_degraded_other_pair_does_not_poison_coin_snapshot_quality(other_side)


@pytest.mark.parametrize("mixed", [False, True])
def test_future_revision_cannot_erase_causal_evidence(mixed, monkeypatch):
    monkeypatch.setattr(cases, "scope_boundaries", compare)
    cases.test_impossible_correction_preserves_earlier_causal_financial_and_boundary_evidence(mixed)


@pytest.mark.parametrize("field,value", [("price", 101), ("fee", -1), ("revision", 1)])
def test_boundary_prefix_corrections(field, value, monkeypatch):
    monkeypatch.setattr(cases, "scope_boundaries", compare)
    cases.test_consumed_boundary_records_corrections_even_with_same_flat_and_realized_pnl(field, value)


@pytest.mark.fake_live
@pytest.mark.parametrize("pside", ["long", "short"])
def test_fake_exchange_boundary_reconstruction(pside, monkeypatch):
    monkeypatch.setattr(fake_cases, "scope_boundaries", compare)
    fake_cases.test_fake_exchange_partial_final_delayed_fill_and_cache_free_replay(pside)


@pytest.mark.parametrize("field,value", [("balance", 0), ("balance_at", -1_000_000), ("config_at", 999_999), ("max_current_age_ms", -1)])
def test_invalid_minimum_snapshot(field, value):
    request = payload(cases.frame(cases.closed_tape()))
    request[field] = value
    with pytest.raises(ValueError):
        rust(request)


def test_post_position_cashflow_is_isolated_then_counted_once():
    p = cases.pair(size=1, basis=100, position_at=2*cases.M,
                   fills=[Fill("open", cases.M, 1, 100, 0), Fill("close", 3*cases.M, -1, 80, -20)])
    old = rust(payload(cases.frame(p)))
    assert old["pairs"][0]["history"]["samples"][-1]["pnl"] == 0
    assert "post_position_fill" in old["reasons"]
    fresh = cases.after_position(replace(p, position_at=4*cases.M, position=replace(p.position, size=0, basis=0)))
    result = rust(payload(cases.frame(fresh)))
    assert result["pairs"][0]["history"]["samples"][-1]["pnl"] == -20
    assert len(result["boundaries"]) == 1


@pytest.mark.parametrize("seed", range(25))
def test_generated_damage_retains_numeric_history_independently_of_boundaries(seed):
    import random
    rng = random.Random(seed)
    tape = [Fill("open", cases.M, 2, 100, 0, -1),
            Fill("partial", 2*cases.M, -1, 80, -20, -1),
            Fill("flat", 3*cases.M, -1, 70, -30, -1)]
    for i in range(len(tape)):
        if rng.random() < .65:
            field = rng.choice(["delta", "price", "realized", "fee"])
            tape[i] = replace(tape[i], **{field: None})
    compare(cases.frame(cases.pair(fills=tape, prices={0:100, cases.M:100, 2*cases.M:80})), "unified")


def test_scope_sum_preserves_large_cancelling_cashflows():
    pairs = [cases.pair(symbol, fills=[Fill("close", cases.M, -1, 100, value)], mark=100)
             for symbol, value in [("A", 1e308), ("B", 1e308), ("C", -1e308)]]
    result = rust(payload(cases.frame(*pairs)))
    assert result["boundaries"][0]["pnl"] == pytest.approx(1e308)
    assert "numeric_range_approximation" in result["reasons"]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("opening,first,last", [("0.8", "0.1", "0.7"), ("0.3", "0.1", "0.2")])
def test_quantity_roundoff_cannot_hide_a_real_flat(side, opening, first, last):
    d = 1 if side == "long" else -1
    fills = [Fill("open", cases.M, dec(opening)*d, 100, 0),
             Fill("partial", 2*cases.M, -dec(first)*d, 90, -1),
             Fill("flat", 3*cases.M, -dec(last)*d, 80, -2)]
    p = cases.pair(pside=side, fills=fills)
    result = rust(payload(cases.frame(p), quantity_step=.1))
    assert len(result["boundaries"]) == 1
    assert result["boundaries"][0]["lifecycle_eligible"]
    assert "uncertain_episode_flat" not in result["reasons"]
    compare(cases.frame(p), "unified", quantity_step=.1)


def test_quantity_tolerance_does_not_hide_a_material_missing_close():
    fills = [Fill("open", cases.M, ".8000001", 100, 0),
             Fill("partial", 2*cases.M, "-.1", 90, -1),
             Fill("last_known", 3*cases.M, "-.7", 80, -2)]
    p = cases.pair(fills=fills)
    result = rust(payload(cases.frame(p)))
    assert not result["boundaries"]
    assert "clamped_quantity" in result["reasons"]
    assert "uncertain_episode_flat" in result["reasons"]
    # Actual residual current exposure cannot be snapped away by historical rounding.
    p = cases.pair(size=1e-8, basis=100, fills=[])
    result = rust(payload(cases.frame(p)))
    assert result["pairs"][0]["history"]["samples"][-1]["size"] == 1e-8


def test_many_decimal_partial_fills_do_not_accumulate_false_missing_quantity():
    fills = [Fill("open", 1, "200", 100, 0)]
    fills += [Fill(f"close-{i}", i+2, "-.1", 90, -1) for i in range(2000)]
    result = rust(payload(cases.frame(cases.pair(fills=fills)), quantity_step=.1))
    assert len(result["boundaries"]) == 1
    assert result["boundaries"][0]["lifecycle_eligible"]
    assert result["boundaries"][0]["pnl"] == -2000


@pytest.mark.parametrize("side", ["long", "short"])
def test_larger_later_position_does_not_hide_earlier_small_flat(side):
    d = 1 if side == "long" else -1
    quantities = [".1", "-.1", ".3", ".3", "2", "2", ".3"]
    fills = [Fill(str(i), i + 1, dec(q) * d, 100, 0) for i, q in enumerate(quantities)]
    p = cases.pair(pside=side, size=4.9*d, basis=100, fills=fills)
    result = rust(payload(cases.frame(p), quantity_step=.1))
    assert [b["timestamp"] for b in result["boundaries"]] == [2]
    assert result["boundaries"][0]["lifecycle_eligible"]
    compare(cases.frame(p), "unified", quantity_step=.1)


@pytest.mark.parametrize("values", permutations([-1e16, -1, 1e16]))
def test_small_scope_cashflow_survives_large_finite_cancellation(values):
    pairs = [cases.pair(str(i), fills=[Fill("close", cases.M, -1, 100, value)], mark=100)
             for i, value in enumerate(values)]
    result = rust(payload(cases.frame(*pairs)))
    assert result["boundaries"][0]["pnl"] == -1


@pytest.mark.parametrize("field,value", [
    ("position_at", 0), ("mark_at", 0), ("position_at", 300_001),
    ("mark_at", 300_001), ("basis", 0), ("mark", 0), ("multiplier", 0),
    ("size", -1), ("quantity_step", 0), ("quantity_step", -1), ("start", 300_001), ("balance_at", 0),
    ("balance_at", 300_001), ("config_at", 300_001),
])
def test_inactive_candle_free_scope_still_validates_current_snapshot(field, value):
    import passivbot_rust as pbr
    p = cases.pair("TEST", size=1, basis=100, now=300_000)
    snapshot = payload(cases.frame(p, now=300_000), "coin", symbol="TEST", pside="long")
    if field in ("basis", "mark", "multiplier", "size", "quantity_step"):
        snapshot["pairs"][0]["position"][field] = value
    elif field in ("position_at", "mark_at"):
        snapshot["pairs"][0][field] = value
    else:
        snapshot[field] = value
    request = dict(snapshot=snapshot, slots=0, span=1, threshold=.05)
    with pytest.raises(ValueError):
        pbr.hsl_revised_candle_free(json.dumps(request))


@pytest.mark.parametrize("values", permutations([-1e16, -1, 1e16]))
@pytest.mark.parametrize("component", ["upnl", "realized"])
def test_small_scope_loss_still_panics_after_large_finite_cancellation(values, component):
    import passivbot_rust as pbr
    pairs = []
    for i, value in enumerate(values):
        if component == "upnl":
            p = cases.pair(str(i), size=abs(value), basis=2, mark=1 if value < 0 else 3)
        else:
            p = cases.pair(str(i), fills=[Fill("close", cases.M, -1, 100, value)], mark=100)
        pairs.append(p)
    snapshot = payload(cases.frame(*pairs, balance=1000))
    request = dict(snapshot=snapshot, slots=1, span=10000, threshold=.0005)
    result = json.loads(pbr.hsl_revised_candle_free(json.dumps(request)))
    assert result[component] == -1
    assert result["signal"]["raw"] == pytest.approx([1 / 1001])
    assert result["signal"]["panic"] == [True]


def test_quantity_roundoff_scale_resets_after_later_episode_opening():
    fills = [Fill("old_open", 1, ".8000001", 100, 0),
             Fill("old_close", 2, "-.8", 90, -1),
             Fill("new_open", 3, "1000000000000", 100, 0)]
    p = cases.pair(size=1e12, basis=100, fills=fills)
    result = rust(payload(cases.frame(p)))
    assert not result["boundaries"]
    assert "clamped_quantity" in result["reasons"]


@pytest.mark.parametrize("step", [None, .0001220703125])
@pytest.mark.parametrize("side", ["long", "short"])
def test_scale_only_rounding_does_not_certify_a_real_residual_flat(step, side):
    d = 1 if side == "long" else -1
    residual = .0001220703125
    fills = [Fill("old_close", 1, -residual*d, 100, -1),
             Fill("later_add", 2, 1e12*d, 100, 0)]
    p = cases.pair(size=(1e12+residual)*d, basis=100, pside=side, fills=fills)
    result = rust(payload(cases.frame(p), quantity_step=step))
    assert not result["boundaries"]
    assert "quantity_precision_unavailable" in result["reasons"]
    assert result["pairs"][0]["history"]["samples"][-1]["size"] == float(p.position.size)


@pytest.mark.parametrize("values", list(permutations([-1e50, -1e30, -1e10, 1e30, 1e50])))
def test_nested_scope_cancellation_keeps_the_net_loss(values):
    import passivbot_rust as pbr
    pairs = [cases.pair(str(i), fills=[Fill("close", 1, -1, 100, value)], mark=100)
             for i, value in enumerate(values)]
    request = payload(cases.frame(*pairs, balance=1e11))
    boundary = rust(request)["boundaries"][0]
    assert boundary["pnl"] == -1e10
    result = json.loads(pbr.hsl_revised_candle_free(json.dumps(dict(snapshot=request, slots=1, span=10000, threshold=.05))))
    assert result["realized"] == -1e10
    assert result["signal"]["panic"] == [True]


@pytest.mark.parametrize("field", ["realized", "fee"])
@pytest.mark.parametrize("values", list(permutations([-1e16, -1, 1e16])))
def test_pair_cashflows_keep_small_losses_before_scope_aggregation(field, values):
    import passivbot_rust as pbr
    fills = [Fill("open", 1, 3, 100, 0)]
    fills += [Fill(str(i), i+2, -1, 100, value if field == "realized" else 0,
                   value if field == "fee" else 0) for i, value in enumerate(values)]
    request = payload(cases.frame(cases.pair(fills=fills), balance=1000))
    prepared = rust(request)
    assert prepared["pairs"][0]["history"]["events"][-1]["realized_cumsum"] == -1
    assert prepared["boundaries"][0]["pnl"] == -1
    result = json.loads(pbr.hsl_revised_candle_free(json.dumps(dict(snapshot=request, slots=1, span=10000, threshold=.0005))))
    assert result["realized"] == -1
    assert result["signal"]["panic"] == [True]


def test_scope_retains_fee_below_one_pairs_rounded_cumulative_precision():
    import passivbot_rust as pbr
    pairs = [cases.pair("A", fills=[Fill("close", 1, -1, 100, 1e16, -1)]),
             cases.pair("B", fills=[Fill("close", 1, -1, 100, -1e16)])]
    request = payload(cases.frame(*pairs, balance=1000))
    assert rust(request)["boundaries"][0]["pnl"] == -1
    result = json.loads(pbr.hsl_revised_candle_free(json.dumps(dict(snapshot=request, slots=1, span=10000, threshold=.0005))))
    assert result["realized"] == -1
    assert result["signal"]["panic"] == [True]


def test_currency_sum_matches_exact_fraction_oracle_for_generated_scopes():
    from fractions import Fraction
    import random
    rng = random.Random(9173)
    for _ in range(100):
        levels = [rng.uniform(-1, 1) * 2**rng.randint(-900, 900) for _ in range(8)]
        residual = rng.uniform(-1, 1) * 2**rng.randint(-900, 900)
        values = levels + [-x for x in levels] + [residual]
        rng.shuffle(values)
        expected = float(sum((Fraction(x) for x in values), Fraction()))
        pairs = [cases.pair(str(i), fills=[Fill("close", 1, -1, 100, value)])
                 for i, value in enumerate(values)]
        assert rust(payload(cases.frame(*pairs)))["boundaries"][0]["pnl"] == expected


@pytest.mark.parametrize("offset_upnl", [False, True])
def test_small_loss_survives_large_realized_peak_or_offsetting_current_upnl(offset_upnl):
    import passivbot_rust as pbr
    if offset_upnl:
        p = cases.pair(size=1e16, basis=2, mark=3,
                       fills=[Fill("loss", 1, 1e16, 2, -1e16, -1)])
    else:
        p = cases.pair(fills=[Fill("open", 1, 2, 100, 0),
                             Fill("profit", 2, -1, 100, 1e16),
                             Fill("fee", 3, -1, 100, 0, -1)])
    request = payload(cases.frame(p, balance=1000))
    result = json.loads(pbr.hsl_revised_candle_free(json.dumps(dict(snapshot=request, slots=1, span=10000, threshold=.0005))))
    assert result["signal"]["raw"] == pytest.approx([1/1001])
    assert result["signal"]["panic"] == [True]


@pytest.mark.parametrize("field,value", [("span", 0), ("span", .5), ("threshold", -1),
                                        ("threshold", 2)])
def test_inactive_scope_rejects_invalid_signal_configuration(field, value):
    import passivbot_rust as pbr
    p = cases.pair("TEST", size=1, basis=100, now=300_000)
    snapshot = payload(cases.frame(p, now=300_000), "coin", symbol="TEST", pside="long")
    request = dict(snapshot=snapshot, slots=0, span=1, threshold=.05)
    request[field] = value
    with pytest.raises(ValueError, match="invalid revised HSL signal inputs"):
        pbr.hsl_revised_candle_free(json.dumps(request))


@pytest.mark.parametrize("budget,loss", [(1e16, 1.), (1e300, 1e280), (1e-200, 1e-220),
                                        (1e308, 1e308)])
def test_candle_free_loss_survives_absolute_peak_rounding(budget, loss):
    from fractions import Fraction
    import passivbot_rust as pbr
    expected = float(Fraction(loss) / (Fraction(budget) + Fraction(loss)))
    p = cases.pair(fills=[Fill("close", 1, -1, 100, -loss)], mark=100)
    snapshot = payload(cases.frame(p, balance=budget))
    request = dict(snapshot=snapshot, slots=1, span=1e9, threshold=expected * .5)
    result = json.loads(pbr.hsl_revised_candle_free(json.dumps(request)))
    assert result["signal"]["raw"][0] == pytest.approx(expected, rel=1e-14, abs=0)
    assert result["signal"]["ema"] == result["signal"]["raw"]
    assert result["signal"]["panic"] == [True]


@pytest.mark.parametrize("pside", ["long", "short"])
def test_confirmed_flat_can_replay_cooldown_with_old_factual_mark(pside):
    M = cases.M
    direction = 1 if pside == "long" else -1
    pair = cases.pair(pside=pside, fills=[Fill("open", M, direction, 100, 0),
                                        Fill("flat", 2*M, -direction, 80, -20)],
                      mark_at=M, now=8*M)
    frame = cases.frame(pair, now=8*M)
    result = rust(payload(frame, quantity_step=.1))
    assert "stale_flat_mark" in result["reasons"]
    assert len(result["boundaries"]) == 1
    assert result["boundaries"][0]["lifecycle_eligible"]
    compare(frame, "unified", quantity_step=.1)
    request = payload(frame)
    request["pairs"][0]["position"].update(size=direction, basis=100.)
    with pytest.raises(ValueError, match="position/mark"):
        rust(request)
    request = payload(frame)
    request["pairs"][0]["mark_at"] = 9*M
    with pytest.raises(ValueError, match="position/mark"):
        rust(request)


def test_explicit_flat_coin_proof_is_neutral_but_absence_is_not_flat():
    request = payload(cases.frame(), "coin", pside="long", symbol="A")
    with pytest.raises(ValueError, match="absent is not flat"):
        rust(request)
    proof = dict(symbol="A", pside="long", position_at=request["now"],
                 fills_at=request["now"], history_start=request["start"])
    request["flat_coin"] = proof
    assert rust(request)["pairs"] == []
    import passivbot_rust as pbr
    trace = json.loads(pbr.hsl_revised_trace(json.dumps(request)))
    assert trace["episodes"][0]["points"] == [dict(
        timestamp=request["now"], pnl=0, upnl=0, exposed=False, flatten=False)]
    for changes in [dict(symbol="B"), dict(pside="short"),
                    dict(position_at=request["now"]-120_001),
                    dict(position_at=request["now"]+1),
                    dict(fills_at=request["now"]-1), dict(history_start=-1)]:
        request["flat_coin"] = {**proof, **changes}
        with pytest.raises(ValueError, match="explicit flat"):
            rust(request)
    request["flat_coin"] = proof
    request["pairs"] = payload(cases.frame(cases.pair()))["pairs"]
    with pytest.raises(ValueError, match="explicit flat"):
        rust(request)
