"""Source-verified native sparse-account flattening contract regressions."""

import json

import passivbot_rust as pbr
import pytest


@pytest.fixture(scope="module", autouse=True)
def verified_native_runtime():
    assert not getattr(pbr, "__is_stub__", False), "requires a real rebuilt native runtime"
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()


def facts(now, size, mode, side, *, slow=1.0):
    return {
        "snapshot": {
            "now": now, "start": 0, "balance": 100.0, "balance_at": now,
            "config_at": now, "max_current_age_ms": 10_000, "mode": mode,
            "pside": None if mode == "portfolio" else side,
            "symbol": "A" if mode == "coin" else None,
            "pairs": [{
                "symbol": "A", "position": {
                    "size": -size if side == "short" else size,
                    "basis": 100.0 if size else 0.0,
                    "mark": 110.0 if side == "short" else 90.0,
                    "multiplier": 1.0, "inverse": False, "pside": side,
                },
                "position_at": now, "mark_at": now, "fills_started_at": None,
                "fills_at": None, "prices_at": now, "prices": {}, "fills": [],
                "revisions": [0, 0, 0, 0], "fills_position_anchor": None,
            }],
        },
        "slots": 1, "scale_budget_with_excess_allowance": False,
        "exposure_budget": None, "memory_ema_span_minutes": slow,
        "drawdown_ratio_ema_span_minutes": 1.0, "threshold": 0.05,
        "cooldown_ms": 60_000,
        "budget_observations": [{
            "timestamp": 0, "order": -1, "balance": 100.0, "slots": 1,
            "scale_budget_with_excess_allowance": False,
            "exposure_budget": None, "estimated": True,
        }],
    }


def evaluate(replay, value):
    return json.loads(replay.evaluate(json.dumps(value, allow_nan=False)))


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_carried_green_retires_old_red_before_sparse_account_flatten(mode, side):
    replay = pbr.HslV9Replay()
    initial = evaluate(replay, facts(1, 1.0, mode, side))
    assert initial["risk"]["drawdown"] == 10.0
    assert initial["risk"]["action"] == "panic"
    advances = replay.cache_info()["advanced_events"]
    preview = evaluate(replay, facts(60_001, 1.0, mode, side))
    assert preview["risk"]["action"] == "normal"
    assert replay.cache_info()["advanced_events"] == advances
    flat = facts(120_001, 0.0, mode, side)
    output = evaluate(replay, flat)
    assert output["risk"]["action"] == "normal"
    assert output["risk"]["cooldown_until"] is None
    assert not output["risk"]["flatten_qualified"]
    assert evaluate(replay, flat) == output
    replay.clear_derived_cache()
    assert evaluate(replay, flat) == output


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_genuine_sparse_pre_red_post_green_keeps_deadline_without_renewal(mode, side):
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1, 1.0, mode, side, slow=43_200.0))
    flat = facts(60_001, 0.0, mode, side, slow=43_200.0)
    stopped = evaluate(replay, flat)
    assert not stopped["risk"]["red"]
    assert stopped["risk"]["action"] == "halted"
    assert stopped["risk"]["cooldown_until"] == 120_001
    assert evaluate(replay, flat) == stopped
    replay.clear_derived_cache()
    later = evaluate(replay, facts(90_001, 0.0, mode, side, slow=43_200.0))
    assert later["risk"]["cooldown_until"] == 120_001
    assert later["risk"]["action"] == "halted"
    expired = evaluate(replay, facts(120_001, 0.0, mode, side, slow=43_200.0))
    assert expired["risk"]["cooldown_until"] == 120_001
    assert expired["risk"]["action"] == "normal"


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_sparse_account_pre_assessment_never_backdates_recovered_quote(mode, side):
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1, 1.0, mode, side))
    flat = facts(60_001, 0.0, mode, side)
    flat["snapshot"]["pairs"][0]["position_at"] = 59_999
    flat["snapshot"]["pairs"][0]["position"]["mark"] = 100.0
    output = evaluate(replay, flat)
    assert output["risk"]["action"] == "halted"
    assert output["risk"]["cooldown_until"] == 119_999


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_sparse_account_pre_assessment_excludes_same_time_post_account_deposit(mode, side):
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1, 1.0, mode, side, slow=43_200.0))
    flat = facts(60_001, 0.0, mode, side, slow=43_200.0)
    flat["snapshot"]["balance"] = 10_000.0
    flat["budget_observations"].append({
        "timestamp": 60_001, "observed_at": 60_001, "order": 2**63 - 1,
        "balance": 10_000.0, "slots": 1,
        "scale_budget_with_excess_allowance": False,
        "exposure_budget": None, "estimated": False,
    })
    output = evaluate(replay, flat)
    assert output["risk"]["action"] == "halted"
    assert output["risk"]["cooldown_until"] == 120_001


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_completed_recovery_mark_precedes_equal_time_flat_account_fact(mode, side):
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1, 1.0, mode, side, slow=43_200.0))
    flat = facts(60_001, 0.0, mode, side, slow=43_200.0)
    flat["snapshot"]["pairs"][0]["position_at"] = 60_000
    flat["snapshot"]["pairs"][0]["prices"] = {"60000": 100.0}
    output = evaluate(replay, flat)
    assert output["risk"]["action"] == "normal"
    assert output["risk"]["cooldown_until"] is None


@pytest.mark.parametrize("mode", ["pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_sparse_aggregate_account_cohort_requires_zero_gross_inventory(mode, side):
    def cohort(now, first_size, second_size):
        value = facts(now, first_size, mode, side)
        peer = json.loads(json.dumps(value["snapshot"]["pairs"][0]))
        peer["symbol"] = "B"
        peer["position"]["size"] = -second_size if side == "short" else second_size
        peer["position"]["basis"] = 100.0 if second_size else 0.0
        value["snapshot"]["pairs"].append(peer)
        return value

    replay = pbr.HslV9Replay()
    assert evaluate(replay, cohort(1, 1.0, 1.0))["risk"]["drawdown"] == 20.0
    assert evaluate(replay, cohort(60_001, 1.0, 1.0))["risk"]["action"] == "normal"
    flat = evaluate(replay, cohort(120_001, 0.0, 0.0))
    assert not flat["risk"]["exposed"]
    assert flat["risk"]["cooldown_until"] is None
    assert flat["risk"]["action"] == "normal"

    partial = pbr.HslV9Replay()
    evaluate(partial, cohort(1, 1.0, 1.0))
    remaining = evaluate(partial, cohort(60_001, 0.0, 1.0))
    assert remaining["risk"]["exposed"]
    assert not remaining["risk"]["flatten_qualified"]
    assert remaining["risk"]["cooldown_until"] is None


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_phase3_partial_fill_flat_account_keeps_deadline_before_max_deposit(mode, side):
    def flat_at(now):
        value = facts(now, 0.0, mode, side, slow=43_200.0)
        value["snapshot"]["balance"] = 10_000.0
        value["snapshot"]["pairs"][0]["fills"] = [{
            "identity": "partial", "timestamp": 60_001,
            "delta": 0.5 if side == "short" else -0.5,
            "price": 110.0 if side == "short" else 90.0,
            "realized": -5.0, "fee": 0.0, "sequence": 0, "revision": 0,
        }]
        value["budget_observations"].extend([
            {"timestamp": 60_001, "order": 0, "balance": 100.0, "slots": 1,
             "scale_budget_with_excess_allowance": False,
             "exposure_budget": None, "estimated": False},
            {"timestamp": 60_001, "observed_at": 60_001, "order": 2**63 - 1,
             "balance": 10_000.0, "slots": 1,
             "scale_budget_with_excess_allowance": False,
             "exposure_budget": None, "estimated": False},
        ])
        return value

    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1, 1.0, mode, side, slow=43_200.0))
    # The partial fill is already assessed with post-fill budget100. Its same-
    # clock flat account fact is phase3; MAX denotes a later account deposit.
    flat = flat_at(60_001)
    stopped = evaluate(replay, flat)
    assert not stopped["risk"]["red"]
    assert not stopped["risk"]["exposed"]
    assert stopped["risk"]["action"] == "halted"
    assert stopped["risk"]["cooldown_until"] == 120_001
    assert evaluate(replay, flat) == stopped
    replay.clear_derived_cache()
    assert evaluate(replay, flat) == stopped
    later = evaluate(replay, flat_at(90_001))
    assert later["risk"]["action"] == "halted"
    assert later["risk"]["cooldown_until"] == 120_001
    expired = evaluate(replay, flat_at(120_001))
    assert expired["risk"]["action"] == "normal"
    assert expired["risk"]["cooldown_until"] == 120_001
