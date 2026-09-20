"""Real-extension parity for immutable snapshot and scope-boundary reconstruction."""
from dataclasses import asdict, replace
import json

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


def payload(snapshot, mode="unified", **kwargs):
    pairs = []
    for p in snapshot.pairs:
        position = asdict(p.position)
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


def compare(snapshot, mode, **kwargs):
    request = payload(snapshot, mode, **kwargs)
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
