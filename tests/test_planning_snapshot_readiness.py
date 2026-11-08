"""Planning-input absence defers without changing snapshot authority."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from live import hsl_live, planning_gates
from live.freshness import FreshnessLedger
from live.market_snapshot import MarketSnapshot
from live.planning_snapshot import PlanningSnapshot, PlanningSnapshotUnavailable

NOW = 1_800_000_000_000
SYMBOL = "BTC/USDT:USDT"


def captured_snapshot(kind="fresh", *, symbols=(SYMBOL,)):
    ledger = FreshnessLedger(now_ms=NOW)
    ledger.begin_epoch(now_ms=NOW)
    for surface in ("positions", "open_orders", "market_snapshot"):
        ledger.stamp(surface, now_ms=NOW)
    return PlanningSnapshot.capture(
        ts_ms=NOW, exchange="fake", user="synthetic", ledger=ledger,
        required_surfaces=("positions", "open_orders", "market_snapshot"),
        min_epochs={surface: (2 if kind == "epoch" and surface == "positions" else 1)
                    for surface in ("positions", "open_orders", "market_snapshot")},
        symbols=symbols,
        market_snapshots={} if kind == "missing" else {
            symbol: MarketSnapshot(symbol, 100.0, 101.0, 100.5,
                                   NOW - (10_001 if kind == "stale" else 0), "fixture")
            for symbol in symbols
        },
        market_snapshot_max_age_ms=10_000,
    )


@pytest.mark.parametrize("kind,reason,surface", [
    ("epoch", "surface_epoch_too_old", "positions"),
    ("missing", "missing_symbols", "market_snapshot"),
    ("stale", "snapshot_too_old", "market_snapshot"),
])
def test_typed_failure_retains_exact_captured_readiness_without_provider_refresh(kind, reason, surface):
    snapshot = captured_snapshot(kind)
    provider = SimpleNamespace(_staged_execution_ready_state=Mock(return_value=(True, {})))
    with pytest.raises(PlanningSnapshotUnavailable) as raised:
        snapshot.raise_if_invalid(now_ms=NOW, context="rust order calculation")
    error = raised.value
    assert error.invalid_details == tuple(snapshot.invalid_details(now_ms=NOW))
    assert error.snapshot_ts_ms == NOW and error.checked_at_ms == NOW
    assert not hasattr(error, "snapshot")

    handled, details = planning_gates.handle_staged_execution_precondition_error(provider, error)

    assert handled and details["missing"] == [surface]
    assert details["invalid"][surface][0]["reason"] == reason
    assert details["context"] == "rust order calculation"
    assert details["epoch"] == snapshot.epoch
    assert details["min_epochs"] == {stamp.name: stamp.min_epoch for stamp in snapshot.surfaces}
    assert details["invalid_counts"] == {surface: 1}
    assert details["defer_reason"] == "captured_planning_snapshot_unavailable"
    provider._staged_execution_ready_state.assert_not_called()


@pytest.mark.parametrize("kind", ["stale", "missing"])
def test_snapshot_failure_projection_is_bounded_and_reports_full_counts(kind):
    symbols = tuple(f"COIN{i}/USDT:USDT" for i in range(100))
    snapshot = captured_snapshot(kind, symbols=symbols)
    with pytest.raises(PlanningSnapshotUnavailable) as raised:
        snapshot.raise_if_invalid(now_ms=NOW, context="rust order calculation")
    handled, details = planning_gates.handle_staged_execution_precondition_error(object(), raised.value)
    assert handled
    rows = details["invalid"]["market_snapshot"]
    if kind == "stale":
        assert len(rows) == 8
        assert details["invalid_counts"] == {"market_snapshot": 100}
        assert len(raised.value.invalid_details) == 100
    else:
        assert len(rows) == 1 and len(rows[0]["symbols"]) == 12
        assert rows[0]["symbols_count"] == 100
        assert len(raised.value.invalid_details[0]["symbols"]) == 100


@pytest.mark.parametrize("message", [
    "unrelated structural failure",
    "planning snapshot invalid before rust order calculation: synthetic invariant",
])
def test_unknown_runtime_failure_is_not_classified_as_readiness(message):
    bot = SimpleNamespace(_staged_execution_ready_state=Mock())
    assert planning_gates.handle_staged_execution_precondition_error(bot, RuntimeError(message)) == (False, {})
    bot._staged_execution_ready_state.assert_not_called()


@pytest.mark.parametrize("message", [
    "staged planner precondition failed: synthetic",
    "planning snapshot invalid before capture: synthetic",
])
def test_existing_precondition_compatibility_keeps_current_retry_policy(message):
    bot = SimpleNamespace(_staged_execution_ready_state=Mock(return_value=(True, {"missing": ["positions"]})))
    handled, details = planning_gates.handle_staged_execution_precondition_error(bot, RuntimeError(message))
    assert handled and details["missing"] == []
    bot._staged_execution_ready_state.assert_called_once()


def test_captured_failure_diagnostic_retains_bounded_details_and_counts():
    from test_passivbot_balance_split import _unit_bot

    snapshot = captured_snapshot("stale")
    with pytest.raises(PlanningSnapshotUnavailable) as raised:
        snapshot.raise_if_invalid(now_ms=NOW, context="rust order calculation")
    _, details = planning_gates.handle_staged_execution_precondition_error(object(), raised.value)
    bot = _unit_bot()
    events = []
    bot._monitor_record_event = lambda kind, tags, payload=None, **kwargs: events.append((kind, payload))

    bot._log_staged_execution_defer(details)

    assert len(events) == 1
    assert events[0][0] == "planning_unavailable"
    assert events[0][1]["invalid"] == details["invalid"]
    assert events[0][1]["invalid_counts"] == {"market_snapshot": 1}
    assert events[0][1]["defer_reason"] == "captured_planning_snapshot_unavailable"


async def run_fake(tmp_path, monkeypatch, exercise):
    import argparse
    import json
    import hjson
    from config import prepare_config
    from config.hsl import generated_template
    from hsl_fixture import load_fake_hsl_config, mutable_runtime_fixture
    from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
    import tools.run_fake_live as runner

    user = f"fake_snapshot_readiness_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    canonical = prepare_config(load_fake_hsl_config(), target="canonical", runtime=None, verbose=False)
    cfg = generated_template(mutable_runtime_fixture(canonical), "coin")
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["enabled"] = False
        cfg["bot"][side]["unstuck"]["enabled"] = False
    config = tmp_path / "config.json"
    config.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
    scenario = hjson.loads((REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text())
    scenario.pop("assertions", None)
    scenario["run_initial_cycle"] = True
    path = tmp_path / "scenario.hjson"
    path.write_text(hjson.dumps(scenario))
    completed = []

    async def run(bot):
        import passivbot
        import utils
        from live import market_snapshot
        clock = [int(utils.utc_ms())]
        monkeypatch.setattr(utils, "utc_ms", lambda: clock[0])
        monkeypatch.setattr(passivbot, "utc_ms", lambda: clock[0])
        monkeypatch.setattr(market_snapshot, "utc_ms", lambda: clock[0])
        await bot.refresh_protective_authoritative_state(require_balance=True)
        await exercise(bot, clock)
        completed.append(True)
        return {"snapshot_readiness_checked": True}

    monkeypatch.setattr(runner, "_run_fake_cycle", run)
    args = argparse.Namespace(config=str(config), scenario=str(path), user=user, max_steps=1,
                              output_dir=str(tmp_path / "run"), log_level=1, snapshot_each_step=False)
    try:
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expiry", "pre_capture_expiry", "pre_capture_epoch", "unknown", "malformed_native"])
async def test_actual_owner_defers_only_snapshot_readiness_then_recovers(tmp_path, monkeypatch, failure):
    async def exercise(bot, clock):
        import passivbot_rust as pbr
        from passivbot_exceptions import FatalBotException
        instance = hsl_live.owner(bot)
        instance.poll_inputs()
        original_packets = bot._planning_data_packets
        native = Mock(wraps=pbr.compute_ideal_orders_json)
        monkeypatch.setattr(pbr, "compute_ideal_orders_json", native)
        deferred = []
        original_defer = bot._log_staged_execution_defer

        def defer(details):
            deferred.append(details)
            original_defer(details)

        monkeypatch.setattr(bot, "_log_staged_execution_defer", defer)
        expiring = [True]

        def packets(required):
            result = original_packets(required)
            if failure == "expiry" and expiring[0]:
                expiring[0] = False
                clock[0] += 10_001
            return result

        monkeypatch.setattr(bot, "_planning_data_packets", packets)
        if failure.startswith("pre_capture_"):
            original_quotes = bot._get_orchestrator_market_snapshots
            requested = []

            async def quotes(symbols, *, record_surface=True):
                result = await original_quotes(symbols, record_surface=record_surface)
                if record_surface and expiring[0]:
                    expiring[0] = False
                    requested.extend(symbols)
                    if failure == "pre_capture_expiry":
                        clock[0] += 10_001
                    else:
                        bot._ensure_freshness_ledger().begin_epoch()
                return result

            monkeypatch.setattr(bot, "_get_orchestrator_market_snapshots", quotes)
        if failure == "unknown":
            def broken_capture(**kwargs):
                raise RuntimeError("synthetic structural snapshot failure")
            monkeypatch.setattr(PlanningSnapshot, "capture", broken_capture)
        elif failure == "malformed_native":
            native.side_effect = lambda _payload: "["
        task = asyncio.create_task(instance._ordinary_plan())
        instance._ordinary = task
        await asyncio.wait_for(asyncio.wait({task}), 5.0)
        assert task.done()
        expected = {
            "expiry": PlanningSnapshotUnavailable,
            "pre_capture_expiry": planning_gates.StagedPlannerInputsUnavailable,
            "pre_capture_epoch": planning_gates.StagedPlannerInputsUnavailable,
            "unknown": RuntimeError,
            "malformed_native": FatalBotException,
        }[failure]
        assert type(task.exception()) is expected
        requests = len(bot.cca.export_request_log())
        if failure in {"unknown", "malformed_native"}:
            with pytest.raises(expected):
                await instance.cycle()
            assert not deferred
            assert native.called is (failure == "malformed_native")
            assert not [row for row in bot.cca.export_request_log()[requests:]
                        if row["method"] in {"create_order", "cancel_order"}]
            return
        if failure.startswith("pre_capture_"):
            # Market data may already have recovered when the owner observes
            # the failed background task. Report the original rejection.
            await original_quotes(requested)
            assert bot._staged_planner_precondition_state(symbols=requested)[0]
        result = await instance.cycle()
        assert result["ordinary_completed"] and not result["ordinary_executed"]
        detail = deferred[-1]["invalid"]["market_snapshot"][0]
        assert deferred[-1]["missing"] == ["market_snapshot"]
        if failure == "pre_capture_epoch":
            assert detail["reason"] == "surface_epoch_too_old"
            assert detail["epoch"] < detail["min_epoch"] == deferred[-1]["epoch"]
        else:
            assert detail["reason"] == ("stale" if failure == "pre_capture_expiry" else "snapshot_too_old")
            assert detail["age_ms"] > 10_000
        native.assert_not_called()
        assert not [row for row in bot.cca.export_request_log()[requests:]
                    if row["method"] in {"create_order", "cancel_order"}]
        assert instance._ordinary is not None
        plan = await asyncio.wait_for(asyncio.shield(instance._ordinary), 5.0)
        assert plan is not None
        assert plan[2].invalid_details(now_ms=clock[0]) == []
        recovered = await instance.cycle()
        assert recovered["ordinary_completed"] and recovered["ordinary_executed"]
        assert native.called

    await run_fake(tmp_path, monkeypatch, exercise)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expiry", "expiry_sink_failure", "legacy_pre_capture", "unknown"])
async def test_actual_protective_builder_defers_only_known_snapshot_readiness(tmp_path, monkeypatch, failure):
    async def exercise(bot, clock):
        import passivbot_rust as pbr
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        quotes = await bot._get_orchestrator_market_snapshots([SYMBOL])
        original_packets = bot._planning_data_packets
        native = Mock(wraps=pbr.compute_protective_closes_json)
        monkeypatch.setattr(pbr, "compute_protective_closes_json", native)
        deferred = []
        monkeypatch.setattr(bot, "_log_staged_execution_defer", deferred.append)
        if failure.startswith("expiry"):
            def packets(required):
                result = original_packets(required)
                clock[0] += 10_001
                return result
            monkeypatch.setattr(bot, "_planning_data_packets", packets)
            if failure == "expiry_sink_failure":
                def broken_sink(_details):
                    raise RuntimeError("synthetic observer failure")
                monkeypatch.setattr(bot, "_log_staged_execution_defer", broken_sink)
            error_type = AuthoritativeSurfaceUnavailable
        elif failure == "legacy_pre_capture":
            monkeypatch.setattr(bot, "_market_snapshot_signature_invalid", lambda _symbols: [
                {"reason": "synthetic_input_unavailable", "symbol": SYMBOL}
            ])
            error_type = AuthoritativeSurfaceUnavailable
        else:
            def broken_capture(**kwargs):
                raise RuntimeError("synthetic structural snapshot failure")
            monkeypatch.setattr(PlanningSnapshot, "capture", broken_capture)
            error_type = RuntimeError
        requests = len(bot.cca.export_request_log())
        with pytest.raises(error_type) as raised:
            await bot.calc_protective_panic_ideal_orders_orchestrator(
                target_psides_by_symbol={SYMBOL: {"long"}}, market_snapshots=quotes,
            )
        if failure.startswith("expiry"):
            assert isinstance(raised.value.__cause__, PlanningSnapshotUnavailable)
            assert raised.value.surface == "protective_planning_inputs"
            if failure == "expiry":
                assert deferred[-1]["context"] == "protective panic order calculation"
                assert deferred[-1]["invalid"]["market_snapshot"][0]["reason"] == "snapshot_too_old"
        elif failure == "legacy_pre_capture":
            assert type(raised.value.__cause__) is RuntimeError
            assert raised.value.surface == "protective_planning_inputs"
            assert not deferred
        else:
            assert type(raised.value) is RuntimeError
            assert not deferred
        native.assert_not_called()
        assert not [row for row in bot.cca.export_request_log()[requests:]
                    if row["method"] in {"create_order", "cancel_order"}]

    await run_fake(tmp_path, monkeypatch, exercise)
