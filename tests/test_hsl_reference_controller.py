"""Controller traces settled before connecting revised scope execution."""

from dataclasses import replace
import json
import sys

import pytest

from hsl_reference import Observation, dec
from hsl_reference_controller import Decision, Episode, Point, replay

REFERENCE_REPLAY = replay


@pytest.fixture(params=["reference", "rust"], autouse=True)
def controller_backend(request, monkeypatch):
    if request.param == "reference":
        return
    import passivbot_rust as pbr
    assert not getattr(pbr, "__is_stub__", False)
    assert hasattr(pbr, "hsl_revised_controller"), "rebuild the source-matched extension"

    def rust_replay(episodes, **kwargs):
        payload = dict(kwargs)
        payload["budget"], payload["span"], payload["threshold"] = map(float,
            (kwargs["budget"], kwargs["span"], kwargs["threshold"]))
        payload["cooldown_ms"] = payload.pop("cooldown")
        payload["episodes"] = [{"entry_reference": None if e.entry_reference is None else float(e.entry_reference),
                                "points": [{"timestamp": p.observation.timestamp,
                                            "pnl": float(p.observation.pnl), "upnl": float(p.observation.upnl),
                                            "exposed": p.exposed, "flatten": p.flatten} for p in e.points]}
                               for e in episodes]
        try:
            expected = REFERENCE_REPLAY(episodes, **kwargs)
        except ValueError:
            with pytest.raises(ValueError):
                pbr.hsl_revised_controller(json.dumps(payload))
            raise
        actual = json.loads(pbr.hsl_revised_controller(json.dumps(payload)))
        assert len(actual) == len(expected)
        for r, e in zip(actual, expected):
            for key in ("timestamp", "action", "red_at", "flat_at", "reason"):
                assert r[key] == getattr(e, key)
            assert r["raw"] == pytest.approx(float(e.raw))
            assert r["ema"] == pytest.approx(float(e.ema))
            assert r["numeric_range_approximation"] == e.numeric_range_approximation
        return tuple(Decision(**r) for r in actual)
    monkeypatch.setattr(sys.modules[__name__], "replay", rust_replay)


def point(t, pnl=0, upnl=0, *, exposed=True, flatten=False):
    return Point(Observation(t, dec(pnl), dec(upnl)), exposed, flatten)


def run(*episodes, now=None, start=0, cooldown=200, restart="always", intervention="panic", span=1):
    if now is None:
        now = max((p.observation.timestamp for e in episodes for p in e.points), default=1000)
    return replay(episodes, now=now, start=start, budget=1000, span=span,
                  threshold=".05", cooldown=cooldown, restart=restart, intervention=intervention)


def stopped():
    return Episode((point(0, exposed=False), point(100, upnl=-100),
                    point(150, -50, -50), point(200, -100, exposed=False, flatten=True)))


def test_actual_flat_anchors_cooldown_partial_does_not():
    trace = run(stopped(), Episode((point(250, exposed=False), point(399, exposed=False), point(400, exposed=False))))
    assert [d.action for d in trace] == ["normal", "panic", "panic", "halted", "halted", "halted", "normal"]
    assert trace[2].red_at == 100 and trace[2].flat_at is None
    assert trace[3].flat_at == 200


@pytest.mark.parametrize("restart", ["always", "never"])
def test_normal_intervention_clears_halt_but_fresh_loss_still_panics(restart):
    second = Episode((point(220, -100, exposed=False), point(230, -101), point(240, -101, -100)))
    trace = run(stopped(), second, restart=restart, intervention="normal")
    assert trace[-2].action == "normal" and trace[-2].reason == "normal_intervention"
    assert trace[-1].action == "panic" and trace[-1].red_at == 240


@pytest.mark.parametrize("restart", ["always", "never"])
def test_repanic_during_cooldown_anchors_new_flat(restart):
    second = Episode((point(230, -101), point(240, -101, exposed=False, flatten=True)))
    third = Episode((point(300, -101, exposed=False), point(439, -101, exposed=False), point(440, -101, exposed=False)))
    trace = run(stopped(), second, third, restart=restart)
    assert trace[4].reason == "panic_intervention" and trace[4].red_at == 230
    assert trace[5].flat_at == 240
    assert trace[-2].action == "halted"
    assert trace[-1].action == ("normal" if restart == "always" else "halted")


def test_residual_exposure_is_not_a_normal_intervention():
    unfinished = Episode((point(0, exposed=False), point(100, upnl=-100), point(150, -50, -50),
                          point(200, -50, 0)))
    trace = run(unfinished, intervention="normal")
    assert trace[-1].action == "panic"
    assert trace[-1].red_at == 100 and trace[-1].flat_at is None


@pytest.mark.parametrize("restart", ["always", "never"])
def test_window_expiry_forgets_stop_and_current_risk_can_create_new_one(restart):
    active = Episode((point(100, upnl=-100), point(200, upnl=0), point(300, upnl=0)))
    # A recovered live mark that left no historical evidence cannot be supplied
    # as a remembered stop to this pure replay.
    assert run(active, start=101, restart=restart)[-1].action == "normal"
    damaged_now = replace(active, points=active.points[:-1] + (point(300, upnl=-100),))
    assert run(damaged_now, start=101, restart=restart)[-1].action == "panic"
    # Closed stops expire too. Remaining ordinary flat points cannot invent them.
    assert run(stopped(), Episode((point(300, exposed=False),)), start=201,
               restart=restart)[-1].action == "normal"


@pytest.mark.parametrize("cooldown,expected", [(0, "normal"), (1, "halted")])
def test_zero_cooldown_is_no_wait(cooldown, expected):
    assert run(stopped(), cooldown=cooldown)[-1].action == expected
    assert run(stopped(), cooldown=cooldown, restart="never")[-1].action == "halted"


def test_flatten_risk_before_same_timestamp_reopen():
    first = Episode((point(0, exposed=False), point(100, -100, exposed=False, flatten=True)))
    second = Episode((point(100, -101), point(150, -101, 200)))
    result = run(first, second)
    assert result[1].action == "halted"
    assert result[2].action == "panic" and result[2].reason == "panic_intervention"


def test_poll_and_cache_free_copy_have_identical_output():
    episodes = [stopped(), Episode((point(250, exposed=False),))]
    expected = run(*episodes)
    fresh = [replace(e, points=tuple(replace(p, observation=replace(p.observation)) for p in e.points)) for e in episodes]
    assert run(*fresh) == expected == run(*episodes)


def test_no_history_singleton_uses_reference_and_flat_scope_is_normal():
    current = Episode((point(1000, upnl=-100),), entry_reference=1100)
    assert run(current, span=1_000_000)[-1].action == "panic"
    assert run(Episode((point(1000, exposed=False),)))[-1].action == "normal"


def test_flat_observation_without_completion_evidence_cannot_start_cooldown():
    episode = Episode((point(0), point(100, upnl=-100), point(200, -100, exposed=False)))
    last = run(episode)[-1]
    assert last.action == "halted" and last.flat_at is None


def test_reset_requires_actual_boundary_and_ordered_trace():
    with pytest.raises(ValueError, match="reset"):
        run(Episode((point(0),)), Episode((point(100),)))
    with pytest.raises(ValueError, match="unordered"):
        run(Episode((point(100), point(0))))


def test_long_ema_prevents_a_brief_excursion_and_final_recovery():
    episode = Episode((point(0), point(60_000, upnl=-100), point(120_000, upnl=0)))
    assert all(d.action == "normal" for d in run(episode, now=120_000, span=1_000_000))
    assert run(episode, now=120_000, span=1)[-1].action == "panic"
    # The latter historical crossing remains reconstructible from the trace.
    # If that point was an unrecorded transient mark, a fresh trace lacks it.
    without_transient = replace(episode, points=(episode.points[0], episode.points[-1]))
    assert run(without_transient, now=120_000, span=1)[-1].action == "normal"


def test_corrected_historical_loss_can_void_a_previous_stop_without_local_latch():
    original = stopped()
    assert run(original)[-1].action == "halted"
    corrected = Episode(tuple(replace(p, observation=replace(p.observation, pnl=dec(0), upnl=dec(0))) for p in original.points))
    assert run(corrected)[-1].action == "normal"
    assert run(original)[-1].action == "halted"


def test_ordinary_flat_resets_peak_without_creating_cooldown():
    first = Episode((point(0), point(100, 100, exposed=False, flatten=True)))
    next_episode = Episode((point(100, 100, exposed=False), point(200, 100)))
    assert all(d.action == "normal" for d in run(first, next_episode))


@pytest.mark.parametrize("policy", ["always", "never"])
def test_all_expired_evidence_requires_current_sample(policy):
    with pytest.raises(ValueError, match="no in-window"):
        run(stopped(), start=201, now=1000, restart=policy)
    with pytest.raises(ValueError, match="no in-window"):
        run(restart=policy)


def test_same_minute_partial_samples_do_not_advance_ema_clock():
    episode = Episode((point(0), point(60_000, upnl=-100),
                       point(60_001, -50, -150), point(120_000, -50, 50)))
    trace = run(episode, now=120_000, span=3)
    assert float(trace[1].ema) == pytest.approx(.05)
    assert float(trace[2].ema) == pytest.approx(.1)


def test_partial_exit_does_not_extend_original_red_time():
    episode = Episode((point(0), point(100, upnl=-100), point(200, -25, -75),
                       point(300, -50, -50), point(400, -75, -25)))
    trace = run(episode)
    assert all(d.red_at == 100 and d.flat_at is None for d in trace[1:])


def test_current_endpoint_is_required_to_evaluate_expiry():
    # A stale last trace point must not preserve a cooldown that has now expired.
    with pytest.raises(ValueError, match="current observation"):
        run(stopped(), now=1000)
    trace = run(stopped(), Episode((point(1000, exposed=False),)), now=1000)
    assert trace[-1].action == "normal"


def test_expired_unreconstructible_crossing_does_not_turn_an_ordinary_flat_into_stop():
    # The remaining flatten is not proof of an expired numerical RED decision.
    # Exchange-derived stop provenance, if supported later, must be explicit.
    trace = run(stopped(), Episode((point(300, exposed=False),)), start=151, restart="never")
    assert trace[-1].action == "normal"


def test_red_recovered_before_flat_is_still_a_reconstructible_stop():
    episode = Episode((point(0), point(60_000, upnl=-100), point(120_000, upnl=0),
                       point(180_000, exposed=False, flatten=True)))
    trace = run(episode, cooldown=0, restart="never")
    assert trace[-1].action == "halted"
    assert trace[-1].red_at == 60_000
    assert trace[-1].flat_at == 180_000


@pytest.mark.parametrize("seed", range(12))
def test_generated_multi_episode_replay_matches_reference(seed):
    import random
    rng = random.Random(seed)
    episodes = []
    clock = 0
    for _ in range(5):
        points = [point(clock, exposed=False)]
        realized = 0
        for _ in range(8):
            clock += rng.choice([1, 60_000, 120_000])
            realized -= rng.randrange(0, 6)
            points.append(point(clock, realized, rng.randrange(-180, 181)))
        clock += 1
        points.append(point(clock, realized - rng.randrange(0, 60), exposed=False, flatten=True))
        episodes.append(Episode(tuple(points)))
        clock += rng.randrange(1, 200_000)
    episodes.append(Episode((point(clock, exposed=False),)))
    run(*episodes, start=seed * 60_000, cooldown=180_000,
        span=rng.choice([1, 2.5, 30.5]),
        restart=rng.choice(["always", "never"]), intervention=rng.choice(["panic", "normal"]))


def test_completed_episode_rebases_to_common_current_endpoint():
    loss = Episode((point(0), point(60_000, -100, exposed=False, flatten=True)))
    recovered = Episode((point(120_000, -100, exposed=False), point(180_000, 0, exposed=False)))
    trace = replay((loss, recovered), now=180_000, start=0, budget=1000, span=1,
                   threshold=".095", cooldown=0, restart="never", intervention="panic")
    assert float(trace[1].raw) == pytest.approx(.1)
    assert trace[-1].action == "halted"


def test_completed_episode_can_end_at_nonpositive_historical_equity():
    loss = Episode((point(0), point(60_000, -2000, exposed=False, flatten=True)))
    recovered = Episode((point(120_000, 0, exposed=False),))
    trace = run(loss, recovered, cooldown=0, restart="never")
    assert float(trace[1].raw) == pytest.approx(2)
    assert trace[-1].action == "halted"


def test_full_history_cannot_accept_minimal_history_entry_reference():
    episode = Episode((point(0), point(100)), entry_reference=2000)
    with pytest.raises(ValueError, match="singleton"):
        run(episode)


def test_future_tail_cannot_be_silently_clipped():
    episode = Episode((point(0), point(100), point(200, exposed=False, flatten=True)))
    with pytest.raises(ValueError, match="future"):
        run(episode, now=100)
