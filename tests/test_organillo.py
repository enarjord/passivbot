from datetime import datetime, timezone

import numpy as np
import pytest

from organillo import DAY_MS, freeze_config, prepare_selection, read_csv, validate_buffers


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()


def test_strict_csv_and_coverage(tmp_path):
    path = tmp_path / "selection.csv"
    path.write_text("date,B,A\n2025-01-01,1,0\n2025-01-02,0,1\n")
    config = {"backtest": {"organillo_csv": str(path), "start_date": "2025-01-01"}}
    freeze_config(config)
    start = 1_735_689_600_000
    stamps = start + np.arange(48) * 3_600_000
    selection = prepare_selection(config, ["A", "B"], stamps)
    assert selection.values.dtype == np.uint8
    assert selection.rows.dtype == np.int32
    assert selection.values[selection.rows[23]].tolist() == [0, 1]
    assert selection.values[selection.rows[24]].tolist() == [1, 0]
    assert prepare_selection(config, ["A", "B"], stamps) is selection
    assert not selection.values.flags.writeable
    with pytest.raises(ValueError, match="missing required UTC date 2025-01-03"):
        prepare_selection(config, ["A", "B"], start + np.arange(72) * 3_600_000)
    with pytest.raises(ValueError, match="missing coin columns C"):
        prepare_selection(config, ["C"], stamps)
    path.write_text("date,B,A\n2025-01-01,0,0\n2025-01-02,0,1\n")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        freeze_config(config)


@pytest.mark.parametrize("text, message", [
    ("date,A,A\n2025-01-01,0,1\n", "unique"),
    ("date,A\n2025-01-01,1.0\n", "column A: expected 0 or 1"),
    ("date,A\n2025-01-01,\n", "expected 0 or 1"),
    ("date,A\n2025-01-01,1,0\n", "expected 2 cells"),
    ("date,A\n2025-1-01,1\n", "YYYY-MM-DD"),
    ("date,A\n2025-01-01,1\n2025-01-01,0\n", "increase uniquely"),
    ("date,A\n", "at least one date"),
])
def test_invalid_csv(tmp_path, text, message):
    path = tmp_path / "selection.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_csv(path)


def test_validator_checks_fresh_contents_when_invoked_again(tmp_path, monkeypatch, capsys):
    import sys
    from organillo import main
    path = tmp_path / "selection.csv"
    path.write_text("date,A\n2025-01-01,1\n2025-01-02,1\n")
    monkeypatch.setattr(sys, "argv", ["organillo-validate", str(path),
                                     "--start-date", "2025-01-01", "--end-date", "2025-01-03"])
    main()
    path.write_text("date,A\n2025-01-01,1\n")
    with pytest.raises(SystemExit) as failed:
        main()
    assert failed.value.code == 2
    assert "missing required daily coverage" in capsys.readouterr().err


def test_warmup_is_inert_and_operational_coverage_is_strict(tmp_path):
    path = tmp_path / "selection.csv"
    path.write_text("date,A\n2025-01-02,1\n")
    start = 1_735_689_600_000
    stamps = start + np.arange(48) * 3_600_000
    config = {"backtest": {"organillo_csv": str(path), "start_date": "2025-01-02"}}
    selection = prepare_selection(config, ["A"], stamps)
    assert np.all(selection.values[selection.rows[:24]] == 0)
    assert np.all(selection.values[selection.rows[24:-1]] == 1)
    assert selection.values[selection.rows[-1]].tolist() == [0]
    with pytest.raises(ValueError, match="row indices"):
        validate_buffers(selection.values, np.full(48, -1, dtype=np.int32), 48, 1)


def test_alignment_cache_distinguishes_operational_starts(tmp_path):
    from copy import deepcopy
    path = tmp_path / "selection.csv"
    path.write_text("date,A\n2025-01-01,1\n2025-01-02,1\n")
    stamps = 1_735_689_600_000 + np.arange(48) * 3_600_000
    config = {"backtest": {"organillo_csv": str(path), "start_date": "2025-01-01"}}
    first = prepare_selection(config, ["A"], stamps)
    later = deepcopy(config)
    later["backtest"]["start_date"] = "2025-01-02"
    second = prepare_selection(later, ["A"], stamps)
    assert first.identity != second.identity
    assert np.all(first.values[first.rows[:-1]] == 1)
    assert first.values[first.rows[-1]].tolist() == [0]
    assert np.all(second.values[second.rows[:24]] == 0)


@pytest.mark.parametrize("with_end", [False, True])
def test_unused_lookahead_needs_no_additional_csv_date(tmp_path, with_end):
    from organillo import period_coins
    path = tmp_path / "selection.csv"
    path.write_text("date,A\n2025-01-01,1\n")
    config = {"backtest": {"organillo_csv": str(path), "start_date": "2025-01-01"}}
    if with_end:
        config["backtest"]["end_date"] = "2025-01-02"
        assert period_coins(config) == {"A"}
    stamps = 1_735_689_600_000 + np.arange(25) * 3_600_000
    selection = prepare_selection(config, ["A"], stamps)
    assert np.all(selection.values[selection.rows[:-1]] == 1)
    assert selection.values[selection.rows[-1]].tolist() == [0]
    assert prepare_selection(config, ["A"], stamps) is selection


def test_consumed_post_end_candles_still_require_csv_coverage(tmp_path):
    path = tmp_path / "selection.csv"
    path.write_text("date,A\n2025-01-01,1\n")
    config = {"backtest": {"organillo_csv": str(path), "start_date": "2025-01-01",
                           "end_date": "2025-01-02"}}
    # The engine consumes the penultimate candle even when externally prepared
    # data extends beyond the configured period. Only the final row is unused.
    stamps = 1_735_689_600_000 + np.arange(26) * 3_600_000
    with pytest.raises(ValueError, match="missing required UTC date 2025-01-02"):
        prepare_selection(config, ["A"], stamps)


@pytest.mark.asyncio
async def test_suite_master_preload_defers_each_scenario_schedule(tmp_path, monkeypatch):
    from copy import deepcopy
    from suite_runner import prepare_master_datasets
    import backtest
    config = fixture_inputs(bars=512)[0]
    config["backtest"].update(organillo_csv=str(tmp_path / "scenario.csv"), organillo_sha256="a" * 64)
    original = deepcopy(config)
    calls = []
    async def prepare(master, exchange, **kwargs):
        calls.append(master)
        assert master["backtest"]["organillo_csv"] is None
        assert master["backtest"]["organillo_sha256"] is None
        return (["COIN00", "COIN01"], np.ones((2, 2, 4)),
                {"COIN00": {}, "COIN01": {}, "__meta__": {}}, "", str(tmp_path),
                np.ones(2), np.array([0, 60000], dtype=np.int64))
    monkeypatch.setattr(backtest, "prepare_hlcvs_mss", prepare)
    datasets = await prepare_master_datasets(config, ["binance"])
    assert datasets["binance"].coins == ["COIN00", "COIN01"]
    assert len(calls) == 1 and config == original


def fixture_inputs(strategy="trailing_martingale", sides="long", bars=5760):
    from tools.gpu_parity import build_parser, fixture_inputs as prepare
    args = build_parser().parse_args(["--fixture", strategy, "--sides", sides,
                                      "--bars", str(bars), "--coins", "2"])
    return prepare(args)


def write_schedule(path, inputs, daily_values):
    config, _candles, _markets, _btc, stamps = inputs
    days = np.unique(stamps // DAY_MS * DAY_MS)
    coins = config["backtest"]["coins"]["binance"]
    rows = ["date," + ",".join(coins)]
    for i, day in enumerate(days):
        values = daily_values[min(i, len(daily_values) - 1)]
        date = datetime.fromtimestamp(int(day) / 1000, timezone.utc).strftime("%Y-%m-%d")
        rows.append(date + "," + ",".join(map(str, values)))
    path.write_text("\n".join(rows) + "\n")
    config["backtest"]["organillo_csv"] = str(path)
    freeze_config(config)
    return path


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_cpu_all_ones_preserves_results_and_all_zero_blocks_entries(tmp_path, strategy, dynamic,
                                                                  require_real_passivbot_rust_module):
    from backtest import build_backtest_payload, execute_backtest
    inputs = fixture_inputs(strategy)
    config, candles, markets, btc, stamps = inputs
    config["backtest"]["dynamic_wel_by_tradability"] = dynamic
    base = build_backtest_payload(candles, markets, config, "binance", btc, stamps, metrics_only=False)
    fills0, equities0, analysis0 = execute_backtest(base, config)
    write_schedule(tmp_path / "ones.csv", inputs, [[1, 1]])
    ones = build_backtest_payload(candles, markets, config, "binance", btc, stamps, metrics_only=False)
    fills1, equities1, analysis1 = execute_backtest(ones, config)
    np.testing.assert_array_equal(fills0, fills1)
    np.testing.assert_array_equal(equities0, equities1)
    assert analysis0 == analysis1
    config["backtest"]["organillo_sha256"] = None
    write_schedule(tmp_path / "zero.csv", inputs, [[0, 0]])
    zero = build_backtest_payload(candles, markets, config, "binance", btc, stamps, metrics_only=False)
    fills, _equities, _analysis = execute_backtest(zero, config)
    assert len(fills) == 0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("backend", ["cpu", "native"])
def test_exclusive_end_lookahead_preserves_disabled_results(tmp_path, strategy, backend,
        monkeypatch, require_real_passivbot_rust_module):
    from copy import deepcopy
    from backtest import build_backtest_payload, execute_backtest
    from organillo import period_coins
    if backend == "native":
        torch = pytest.importorskip("torch")
        if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
            pytest.skip("native GPU runtime unavailable")
    config, candles, markets, btc, stamps = fixture_inputs(strategy, sides="both", bars=512)
    # Supply the execution lookahead at the existing exclusive endpoint, without
    # adding that next UTC day's unused eligibility to the CSV.
    candles = np.concatenate((candles, candles[-1:]), axis=0)
    btc = np.concatenate((btc, btc[-1:]))
    stamps = np.concatenate((stamps, stamps[-1:] + 60_000))
    for coin in config["backtest"]["coins"]["binance"]:
        markets[coin]["last_valid_index"] = len(stamps) - 1
    ones = deepcopy(config)
    write_schedule(tmp_path / "ones.csv", (ones, candles, markets, btc, stamps[:-1]), [[1, 1]])
    assert period_coins(ones) == {"COIN00", "COIN01"}
    selection = prepare_selection(ones, ["COIN00", "COIN01"], stamps)
    assert selection.values[selection.rows[-1]].tolist() == [0, 0]
    if backend == "cpu":
        baseline = execute_backtest(build_backtest_payload(
            candles, markets, config, "binance", btc, stamps), config)
        enabled = execute_backtest(build_backtest_payload(
            candles, markets, ones, "binance", btc, stamps), ones)
        np.testing.assert_array_equal(baseline[0], enabled[0])
        np.testing.assert_array_equal(baseline[1], enabled[1])
        assert baseline[2] == enabled[2]
        assert enabled[2]["backtest_completion_ratio"] == 1.0
    else:
        from contextlib import ExitStack
        from tools.gpu_parity import _native_dataset
        from optimization.gpu.native import NativeBacktestService
        from optimization.gpu.executor import BacktestRequest
        def forbidden(*_args, **_kwargs):
            pytest.fail("native eligibility cannot invoke CPU simulation")
        monkeypatch.setattr("backtest.execute_backtest", forbidden)
        metrics = ("adg_strategy_eq", "fills_per_day", "backtest_completion_ratio")
        with ExitStack() as resources:
            with NativeBacktestService(batch_size=1, tuning_mode="off") as service:
                for key, current in (("disabled", config), ("ones", ones)):
                    dataset = resources.enter_context(_native_dataset(
                        (current, candles, markets, btc, stamps), "binance", metrics))
                    service.register_dataset(key, dataset)
                baseline = service.submit(BacktestRequest("disabled", "disabled", {})).result(timeout=180)
                enabled = service.submit(BacktestRequest("ones", "ones", {})).result(timeout=180)
                assert baseline.metrics == enabled.metrics
                assert enabled.metrics["backtest_completion_ratio"] == 1.0


def idle_clock_inputs(strategy, sides="both"):
    """A short replay spanning four UTC dates and an explicit lookahead."""
    config, candles, markets, btc, stamps = fixture_inputs(strategy, sides, bars=512)
    endpoint = int(stamps[-1]) + 60_000
    candles = np.concatenate((candles, candles[-1:]), axis=0)
    btc = np.concatenate((btc, btc[-1:]))
    stamps = endpoint - (512 - np.arange(513)) * 600_000
    config["backtest"].update(candle_interval_minutes=10,
        dynamic_wel_by_tradability=True,
        start_date=datetime.fromtimestamp(int(stamps[0])/1000, timezone.utc).isoformat(),
        end_date=datetime.fromtimestamp(endpoint/1000, timezone.utc).isoformat())
    markets["__meta__"].update(requested_start_ts=int(stamps[0]), data_interval_minutes=10)
    for coin in config["backtest"]["coins"]["binance"]:
        markets[coin]["last_valid_index"] = 5120
    return config, candles, markets, btc, stamps


def native_clock_metrics(inputs, metrics, monkeypatch):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("native GPU runtime unavailable")
    from tools.gpu_parity import _native_dataset
    from optimization.gpu.native import NativeBacktestService
    from optimization.gpu.executor import BacktestRequest
    def forbidden(*_args, **_kwargs):
        pytest.fail("native eligibility cannot invoke CPU simulation")
    with monkeypatch.context() as isolated:
        isolated.setattr("backtest.execute_backtest", forbidden)
        with _native_dataset(inputs, "binance", metrics) as dataset:
            with NativeBacktestService(batch_size=1, tuning_mode="off") as service:
                service.register_dataset("clock", dataset)
                result = service.submit(BacktestRequest("clock", "clock", {})).result(timeout=180)
                assert not result.liquidated
                return result.metrics


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("policy", ["zero", "delayed"])
@pytest.mark.parametrize("backend", ["cpu", "native"])
def test_initial_idle_schedule_keeps_full_accounting_period(tmp_path, strategy, sides, policy,
        backend, monkeypatch, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload, execute_backtest
    inputs = idle_clock_inputs(strategy, sides)
    config, candles, markets, btc, stamps = inputs
    write_schedule(tmp_path / "clock.csv", inputs,
                   [[0, 0]] if policy == "zero" else [[0, 0], [1, 1]])
    payload = build_backtest_payload(candles, markets, config, "binance", btc, stamps)
    fills, equities, analysis = execute_backtest(payload, config)
    # Mask-aware preparation omits future-only admissions from global warmup.
    # Accounting still waits for ordinary raw candle/warmup readiness, which can
    # precede the disabled global guard by one candle in this fixture.
    params = payload.backtest_params
    ready = min((markets[coin]["first_valid_index"] + markets[coin]["warmup_minutes"]) // 10
                for coin in config["backtest"]["coins"]["binance"])
    assert min(params["trade_start_indices"]) == ready
    first = max(max(params["global_warmup_bars"], 1) + 1, ready)
    expected_stamps = stamps[first:-1]
    expected_stamps = expected_stamps[expected_stamps >= params["requested_start_timestamp_ms"]]
    expected_duration = (expected_stamps[-1] - expected_stamps[0]) / DAY_MS
    if backend == "cpu":
        np.testing.assert_array_equal(equities[:, 0], expected_stamps)
        first_admission = (int(stamps[0]) // DAY_MS + 1) * DAY_MS
        idle = equities if policy == "zero" else equities[equities[:, 0] < first_admission]
        assert len(idle) > 0
        np.testing.assert_array_equal(idle[:, 1], np.full(len(idle), 1000.0))
        if policy == "zero":
            assert len(fills) == 0
        else:
            assert len(fills) > 0
            assert min(float(fill[1]) for fill in fills) >= first_admission
        daily = list({int(row[0]) // DAY_MS: float(row[1]) for row in equities}.values())
        expected_adg = (np.mean(daily[-3:]) / daily[0]) ** (1.0 / len(daily)) - 1.0
        assert analysis["adg_strategy_eq"] == pytest.approx(expected_adg, abs=1e-12)
        observed = analysis
    else:
        metrics = ("adg_strategy_eq", "fills_per_day", "backtest_completion_ratio")
        observed = native_clock_metrics(inputs, metrics, monkeypatch)
        for name, atol in (("adg_strategy_eq", 1e-7), ("fills_per_day", 1e-6)):
            assert observed[name] == pytest.approx(analysis[name], abs=atol, rel=1e-4)
    if backend == "cpu":
        assert observed["fills_analysis_duration_days"] == pytest.approx(expected_duration)
    assert observed["fills_per_day"] == pytest.approx(len(fills) / expected_duration, abs=1e-6)
    assert observed["backtest_completion_ratio"] == 1.0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("condition", ["zero_slots", "zero_wel", "delayed_readiness"])
@pytest.mark.parametrize("backend", ["cpu", "native"])
def test_idle_clock_preserves_unmasked_capacity_and_readiness(tmp_path, strategy, condition,
        backend, monkeypatch, require_real_passivbot_rust_module):
    from copy import deepcopy
    from backtest import build_backtest_payload, execute_backtest
    inputs = idle_clock_inputs(strategy)
    config, candles, markets, btc, stamps = inputs
    if condition == "delayed_readiness":
        for coin in config["backtest"]["coins"]["binance"]:
            markets[coin]["first_valid_index"] = 120
    else:
        key = "max_positions" if condition == "zero_slots" else "total_wallet_exposure_limit"
        for side in ("long", "short"):
            config["bot"][side]["risk"][key] = 0
    ones = deepcopy(config)
    write_schedule(tmp_path / "ones-clock.csv", (ones, candles, markets, btc, stamps), [[1, 1]])
    if backend == "cpu":
        baseline = execute_backtest(build_backtest_payload(
            candles, markets, config, "binance", btc, stamps), config)
        enabled = execute_backtest(build_backtest_payload(
            candles, markets, ones, "binance", btc, stamps), ones)
        np.testing.assert_array_equal(baseline[0], enabled[0])
        np.testing.assert_array_equal(baseline[1], enabled[1])
        assert baseline[2] == enabled[2]
        if condition == "delayed_readiness":
            assert baseline[1][0, 0] > stamps[0] + 120 * 60_000
    else:
        metrics = ("adg_strategy_eq", "fills_per_day", "backtest_completion_ratio")
        if condition != "delayed_readiness":
            # The native backend rejects a configuration with no enabled side.
            for current in (inputs, (ones, candles, markets, btc, stamps)):
                with pytest.raises(ValueError, match="requires one or two enabled sides"):
                    native_clock_metrics(current, metrics, monkeypatch)
        else:
            assert native_clock_metrics(inputs, metrics, monkeypatch) == native_clock_metrics(
                (ones, candles, markets, btc, stamps), metrics, monkeypatch)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("hsl", ["coin", "portfolio"])
@pytest.mark.parametrize("policy", ["zero", "delayed"])
def test_idle_prefix_hsl_observations_follow_accounting_clock(tmp_path, strategy, sides,
        hsl, policy, monkeypatch, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload, execute_backtest
    from tools.gpu_parity import build_parser, fixture_inputs as prepare
    config, candles, markets, btc, stamps = prepare(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2", "--bars", "2880",
        "--hsl", hsl, "--hsl-red-threshold", "0.002"]))
    candles = np.concatenate((candles, candles[-1:]), axis=0)
    btc = np.concatenate((btc, btc[-1:]))
    stamps = np.concatenate((stamps, stamps[-1:] + 60_000))
    for coin in config["backtest"]["coins"]["binance"]:
        markets[coin]["last_valid_index"] = len(stamps) - 1
    inputs = config, candles, markets, btc, stamps
    write_schedule(tmp_path / "hsl-clock.csv", inputs,
                   [[0, 0]] if policy == "zero" else [[0, 0], [1, 1]])
    _fills, _equities, analysis = execute_backtest(build_backtest_payload(
        candles, markets, config, "binance", btc, stamps), config)
    if policy == "zero":
        assert len(_fills) == 0
        assert analysis["hard_stop_triggers_per_year"] == 0.0
        assert analysis["hard_stop_time_in_red_pct"] == 0.0
    else:
        assert len(_fills) > 0
        first_admission = (int(stamps[0]) // DAY_MS + 1) * DAY_MS
        assert min(float(fill[1]) for fill in _fills) >= first_admission
    metrics = ("adg_strategy_eq", "hard_stop_time_in_red_pct",
               "hard_stop_triggers_per_year", "drawdown_worst_mean_1pct_ema_strategy_eq",
               "backtest_completion_ratio")
    observed = native_clock_metrics(inputs, metrics, monkeypatch)
    assert observed["backtest_completion_ratio"] == 1.0
    for name in metrics:
        assert observed[name] == pytest.approx(analysis[name], abs=1e-5, rel=1e-4), name

def test_payload_subset_preserves_physical_column_mapping(tmp_path, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload, subset_backtest_payload
    inputs = fixture_inputs()
    write_schedule(tmp_path / "selection.csv", inputs, [[0, 1], [1, 0]])
    config, candles, markets, btc, stamps = inputs
    payload = build_backtest_payload(candles, markets, config, "binance", btc, stamps)
    subset = subset_backtest_payload(payload, coin_indices=[1])
    np.testing.assert_array_equal(subset.bundle.eligibility, np.asarray(payload.bundle.eligibility)[:, [1]])
    np.testing.assert_array_equal(subset.bundle.eligibility_rows, payload.bundle.eligibility_rows)


def test_disabled_preparation_never_reads_csv(monkeypatch):
    monkeypatch.setattr("organillo.read_csv", lambda _p: pytest.fail("disabled path read a CSV"))
    config = {"backtest": {"organillo_csv": None}}
    freeze_config(config)
    assert prepare_selection(config, [], None) is None


def test_disabled_cpu_and_native_preparation_allocate_no_schedule(
        monkeypatch, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload
    from optimization.native_datasets import NativeDatasetRegistry
    from shared_arrays import SharedArrayManager
    from optimize import Evaluator

    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled preparation accessed an eligibility schedule")
    for name in ("read_csv", "freeze_config", "prepare_selection", "share_selection"):
        monkeypatch.setattr("organillo." + name, unexpected)
    config, candles, markets, btc, timestamps = fixture_inputs(bars=512)
    config["optimize"].update(scoring=[{"metric": "adg_strategy_eq", "goal": "max"}], limits=[])
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    assert payload.bundle.eligibility is None
    manager = SharedArrayManager()
    base = None
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc)]
        base = Evaluator({"binance": specs[0]}, {"binance": specs[1]}, {"binance": markets},
                         config, timestamps={"binance": timestamps}, shared_array_manager=manager)
        assert not base.selection_specs
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance": ("COIN00", "COIN01")}) as registry:
            assert not registry._selections
            for binding in registry.bindings:
                assert binding.dataset.eligibility is None
                assert binding.dataset.eligibility_rows is None
                with binding.dataset.attach_selection() as selection:
                    assert selection is None
    finally:
        if base is not None:
            for group in base._attachments.values():
                for attachment in group.values():
                    attachment.close()
            base._attachments.clear()
        manager.cleanup()


def test_canonical_config_roundtrip_preserves_optional_input(tmp_path):
    import json
    from config import get_template_config, prepare_config
    source = get_template_config()
    assert source["backtest"]["organillo_csv"] is None
    assert source["backtest"]["organillo_sha256"] is None
    path = tmp_path / "selection.csv"
    path.write_text("date,BTC\n2025-01-01,1\n")
    source["backtest"]["organillo_csv"] = str(path)
    freeze_config(source)
    prepared = prepare_config(json.loads(json.dumps(source)), verbose=False,
                              target="canonical", runtime=None)
    for key in ("organillo_csv", "organillo_sha256"):
        assert prepared["backtest"][key] == source["backtest"][key]


@pytest.mark.parametrize("field", ["organillo_csv", "organillo_sha256"])
def test_optimizer_runtime_pins_cannot_replace_prepared_schedule(field):
    from config import get_template_config, prepare_config
    source = get_template_config()
    source["optimize"]["fixed_runtime_overrides"] = {f"backtest.{field}": None}
    with pytest.raises(ValueError, match="prepared dataset input"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepared_buffer_hash_must_match_config(tmp_path, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload
    from organillo import Selection
    inputs = fixture_inputs(bars=512)
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 1]])
    config, candles, markets, btc, stamps = inputs
    selected = prepare_selection(config, config["backtest"]["coins"]["binance"], stamps)
    wrong = Selection(selected.values, selected.rows, "0" * 64)
    with pytest.raises(ValueError, match="prepared eligibility SHA-256"):
        build_backtest_payload(candles, markets, config, "binance", btc, stamps, selection=wrong)
    config["backtest"]["organillo_csv"] = None
    with pytest.raises(ValueError, match="requires backtest.organillo_csv"):
        build_backtest_payload(candles, markets, config, "binance", btc, stamps, selection=selected)
    config["backtest"]["organillo_csv"] = str(tmp_path / "selection.csv")
    config["backtest"]["organillo_sha256"] = None
    from tools.gpu_parity import _native_dataset
    with pytest.raises(ValueError, match="validated SHA-256"):
        with _native_dataset(inputs, "binance", ["adg_strategy_eq"]):
            pytest.fail("unidentified prepared schedule accepted")


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_enabled_shader_lowers_byte_eligibility_to_cuda(strategy, require_real_passivbot_rust_module):
    import passivbot_rust
    from optimization.gpu.cuda_kernel import cuda_source
    source = getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
    lowered = cuda_source("#define PASSIVBOT_ORGANILLO 1\n" + source, coin_capacity=2)
    assert "using uchar = unsigned char;" in lowered
    assert "const uchar* organillo_values" in lowered
    assert "const int* organillo_rows" in lowered
    if strategy == "trailing_martingale":
        assert "const signed char* touch_min_qty_relation" in lowered


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_eligibility_macro_preserves_current_launch_argument_classification(
        strategy, enabled, monkeypatch, require_real_passivbot_rust_module):
    import re
    import sys
    from types import SimpleNamespace
    import passivbot_rust
    from optimization.gpu.cuda_kernel import CudaShaderLibrary

    class RawModule:
        def __init__(self, **kwargs):
            self.code = kwargs["code"]
        def compile(self):
            pass

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(RawModule=RawModule))
    source = getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
    assert "#define ORGANILLO_ARGS , constant uchar* organillo_values, constant int* organillo_rows" in source
    if enabled:
        source = "#define PASSIVBOT_ORGANILLO 1\n" + source
    library = CudaShaderLibrary(source, coin_capacity=2)
    for name, signature in re.findall(r"kernel\s+void\s+(\w+)\((.*?)\)\s*\{", source, re.S):
        # Eligibility adds two compiler-side pointer arguments. Current kernels
        # have no scalar-reference conversion after them, and the one byte input
        # precedes them, so host launch classifications must remain valid.
        expanded = signature.replace("ORGANILLO_ARGS", (
            ", constant uchar* organillo_values, constant int* organillo_rows" if enabled else ""
        ))
        arguments = expanded.split(",")
        expected = {
            index: match[1]
            for index, argument in enumerate(arguments)
            if (match := re.search(r"\bconstant\s+(\w+)\s*&", argument))
        }
        assert not expected, name
        assert library._scalar_parameters[name] == expected, name
        byte_indices = {
            index for index, argument in enumerate(arguments)
            if re.search(r"\bconstant\s+int\s*\*\s*touch_min_qty_relation\b", argument)
        }
        assert library._byte_parameters[name] == byte_indices, name
        if enabled and "ORGANILLO_ARGS" in signature:
            coin = next(i for i, argument in enumerate(arguments) if "coin_settings" in argument)
            assert "organillo_values" in arguments[coin + 1]
            assert "organillo_rows" in arguments[coin + 2]
            assert all(index < coin for index in byte_indices)
        # Keep the compiler source macro-based; only host argument inspection expands it.
        assert "ORGANILLO_ARGS" in library._module.code


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("hedge", [False, True])
def test_native_gpu_rotating_schedule_parity(tmp_path, strategy, sides, dynamic, hedge,
                                             require_real_passivbot_rust_module):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("native GPU runtime unavailable")
    from tools.gpu_parity import run_comparison, DEFAULT_TOLERANCES
    inputs = fixture_inputs(strategy, sides, bars=512)
    # A short, already aggregated replay crosses daily boundaries without accumulating
    # float32 compounding drift. Producer validity indices remain in one-minute units.
    config, _candles, markets, _btc, stamps = inputs
    endpoint = int(stamps[-1]) + 60_000
    stamps[:] = endpoint - (len(stamps) - np.arange(len(stamps))) * 600_000
    config["backtest"].update(candle_interval_minutes=10,
        start_date=datetime.fromtimestamp(int(stamps[0])/1000, timezone.utc).isoformat(),
        end_date=datetime.fromtimestamp(endpoint/1000, timezone.utc).isoformat())
    markets["__meta__"].update(requested_start_ts=int(stamps[0]), data_interval_minutes=10)
    for coin in config["backtest"]["coins"]["binance"]:
        markets[coin]["last_valid_index"] = len(stamps) * 10 - 1
    inputs[0]["backtest"]["dynamic_wel_by_tradability"] = dynamic
    inputs[0]["backtest"]["hedge_mode"] = hedge
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 0], [0, 1], [1, 1]])
    report = run_comparison(inputs, "binance", ["adg_strategy_eq", "drawdown_worst_strategy_eq",
                            "backtest_completion_ratio", "fills_per_day"], DEFAULT_TOLERANCES)
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_gpu_zero_schedule_and_hsl_replay(tmp_path, strategy,
                                                require_real_passivbot_rust_module):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("native GPU runtime unavailable")
    from tools.gpu_parity import build_parser, fixture_inputs as prepare, run_comparison, DEFAULT_TOLERANCES
    for policy in ([[0, 0]], [[1, 0]]):
        args = build_parser().parse_args(["--fixture", strategy, "--sides", "both", "--coins", "2",
                                         "--bars", "512", "--hsl", "portfolio",
                                         "--hsl-red-threshold", "0.002"])
        inputs = prepare(args)
        write_schedule(tmp_path / f"selection_{len(policy)}.csv", inputs, policy)
        report = run_comparison(inputs, "binance", ["adg_strategy_eq", "drawdown_worst_strategy_eq",
                                "backtest_completion_ratio", "fills_per_day"], DEFAULT_TOLERANCES)
        assert report["passed"], report["metrics"]


def test_period_universe_is_exact_union_with_approved_side_restrictions(tmp_path):
    from backtest_universe import effective_backtest_data_coins
    inputs = fixture_inputs(bars=2880)
    config = inputs[0]
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 0], [0, 1]])
    assert effective_backtest_data_coins(config) == ["COIN00", "COIN01"]
    config["backtest"]["organillo_sha256"] = None
    write_schedule(tmp_path / "one.csv", inputs, [[1, 0]])
    assert effective_backtest_data_coins(config) == ["COIN00"]
    config['bot']["long"]['universe']['approved_coins'] = ["COIN00"]
    with pytest.raises(ValueError, match="outside resolved approved universe"):
        effective_backtest_data_coins(config)


def test_saved_contract_rejects_changed_schedule(tmp_path, monkeypatch):
    from copy import deepcopy
    from optimization import evaluation_contract as contracts
    monkeypatch.setattr(contracts, "evaluation_implementation_identity", lambda: {"test": 1})
    inputs = fixture_inputs(bars=512)
    config = inputs[0]
    before = contracts.build_evaluation_contract(config)
    assert "organillo" not in before
    write_schedule(tmp_path / "ones.csv", inputs, [[1, 1]])
    enabled = contracts.build_evaluation_contract(config)
    changed = deepcopy(config)
    changed["backtest"]["organillo_sha256"] = "0" * 64
    assert enabled != contracts.build_evaluation_contract(changed)
    changed["backtest"]["organillo_sha256"] = None
    with pytest.raises(ValueError, match="validated SHA-256"):
        contracts.build_evaluation_contract(changed)


def test_registry_reuses_shared_selection_and_fallback_handles_sliced_timestamps(tmp_path):
    from optimization.native_datasets import NativeDatasetRegistry
    from shared_arrays import SharedArrayManager
    from optimize import Evaluator, SuiteEvaluator
    from types import SimpleNamespace
    from copy import deepcopy
    inputs = fixture_inputs(bars=512)
    config, candles, markets, btc, timestamps = inputs
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 0]])
    config["optimize"].update(scoring=[{"metric": "adg_strategy_eq", "goal": "max"}], limits=[])
    manager = SharedArrayManager()
    base = None
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc)]
        base = Evaluator({"binance": specs[0]}, {"binance": specs[1]}, {"binance": markets},
                         config, timestamps={"binance": timestamps}, shared_array_manager=manager)
        with NativeDatasetRegistry(base, standalone_candle_coins={"binance": ("COIN00", "COIN01")}) as registry:
            dataset = registry.bindings[0].dataset
            assert dataset.eligibility == base.selection_specs["binance"][0]
            with dataset.attach_selection() as selected:
                np.testing.assert_array_equal(selected.values, base.prepared_selection("binance").values)
            altered = deepcopy(config)
            altered["backtest"]["organillo_sha256"] = "0" * 64
            with pytest.raises(ValueError, match="newly prepared dataset"):
                dataset.with_config(altered)
        # Exercise the registry-owned fallback and physical/time slice separately.
        base.shared_array_manager = None
        subset_config = deepcopy(config)
        subset_config["backtest"]["coins"]["binance"] = ["COIN01"]
        ctx = SimpleNamespace(label="slice", config=subset_config, exchanges=["binance"],
            msss={"binance": markets}, timestamps={"binance": timestamps[32:96]}, overrides={},
            hlcvs_specs=base.hlcvs_specs, btc_usd_specs=base.btc_usd_specs,
            master_hlcvs_specs=base.hlcvs_specs, master_btc_specs=base.btc_usd_specs,
            time_slice={"binance": (32,96)}, coin_slice_indices={"binance": [1]},
            coin_indices={"binance": None}, candle_coins={"binance": ("COIN00", "COIN01")})
        with NativeDatasetRegistry(SuiteEvaluator(base, [ctx], {"default": "mean"})) as registry:
            dataset = registry.bindings[0].dataset
            with dataset.attach_selection() as selected:
                assert selected.values.shape[1] == 1 and len(selected.rows) == 64
    finally:
        # Attached worker views must close before their parent-owned segments.
        if base is not None:
            for group in base._attachments.values():
                for attachment in group.values():
                    attachment.close()
            base._attachments.clear()
        manager.cleanup()


def test_enabled_invalid_transport_is_rejected_before_simulation(tmp_path, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload
    from organillo import Selection
    inputs = fixture_inputs(bars=512)
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 1]])
    config, candles, markets, btc, timestamps = inputs
    wrong = Selection(np.full((1, 2), 2, dtype=np.uint8), np.zeros(512, dtype=np.int32), "0" * 64)
    with pytest.raises(ValueError, match="binary uint8"):
        build_backtest_payload(candles, markets, config, "binance", btc, timestamps, selection=wrong)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_service_disabled_ones_zero_eviction_and_no_cpu_replay(tmp_path, strategy,
        coin_count, monkeypatch, require_real_passivbot_rust_module):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("native GPU runtime unavailable")
    from tools.gpu_parity import build_parser, fixture_inputs as prepare, _native_dataset
    from optimization.gpu.native import NativeBacktestService
    from optimization.gpu.executor import BacktestRequest
    from contextlib import ExitStack
    from copy import deepcopy
    inputs = prepare(build_parser().parse_args(["--fixture", strategy, "--sides", "both",
                                               "--coins", str(coin_count), "--bars", "512"]))
    ones = (deepcopy(inputs[0]), *inputs[1:])
    zero = (deepcopy(inputs[0]), *inputs[1:])
    write_schedule(tmp_path / "ones.csv", ones, [[1] * coin_count])
    write_schedule(tmp_path / "zero.csv", zero, [[0] * coin_count])
    def forbidden(*_args, **_kwargs):
        pytest.fail("native GPU eligibility cannot invoke CPU simulation")
    monkeypatch.setattr("backtest.execute_backtest", forbidden)
    metrics = ("adg_strategy_eq", "fills_per_day", "backtest_completion_ratio")
    with ExitStack() as resources:
        datasets = {key: resources.enter_context(_native_dataset(value, "binance", metrics))
                    for key, value in (("disabled", inputs), ("ones", ones), ("zero", zero))}
        with NativeBacktestService(batch_size=1, tuning_mode="off") as service:
            for key, dataset in datasets.items():
                service.register_dataset(key, dataset)
            actual = {}
            for index, key in enumerate(("disabled", "ones", "zero", "disabled", "ones", "zero")):
                result = service.submit(BacktestRequest(str(index), key, {})).result(timeout=180)
                if key in actual:
                    assert actual[key] == result.metrics
                actual[key] = result.metrics
            assert actual["disabled"] == actual["ones"]
            assert actual["zero"]["fills_per_day"] == actual["zero"]["adg_strategy_eq"] == 0.0
        assert not service._prepared_cache and not service._subset_cache


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_gpu_chunk_and_factual_hsl_preserve_eligibility_schedule(tmp_path, strategy,
        require_real_passivbot_rust_module):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("native GPU runtime unavailable")
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import build_parser, fixture_inputs as prepare
    inputs = prepare(build_parser().parse_args(["--fixture", strategy, "--sides", "both",
        "--coins", "2", "--bars", "1600", "--hsl", "portfolio", "--hsl-red-threshold", ".002"]))
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 0], [0, 0]])
    config, candles, markets, btc, timestamps = inputs
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="binance", batch_size=1,
        needed_metrics={"adg_strategy_eq", "fills_per_day", "backtest_completion_ratio"}, factual_hsl=True)
    runner = proxy.fused_runner
    parameters = np.concatenate([proxy._parameter_matrix([{}], side) for side in proxy.sides], axis=1)
    runner.max_dispatch_candidate_bars = None
    full = runner.run(parameters)
    full = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in full.items()}
    runner.max_dispatch_candidate_bars = 128
    chunked = runner.run(parameters)
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    for key, expected in full.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(chunked[key], expected, rtol=0, atol=0, equal_nan=True, msg=key)
        else:
            assert chunked[key] == expected


def test_future_only_warmup_does_not_delay_initial_coin(tmp_path, require_real_passivbot_rust_module):
    from backtest import build_backtest_payload
    from warmup_utils import compute_per_coin_warmup_minutes
    inputs = fixture_inputs(strategy="ema_anchor", bars=2880)
    config, candles, markets, btc, timestamps = inputs
    config['bot']['global']['warmup']['max_minutes'] = 2000
    config["coin_overrides"] = {"COIN01": {"bot": {"long": {"strategy": {
        "ema_anchor": {"ema_span_0": 20000.0, "ema_span_1": 40000.0}}}}}}
    for coin in ("COIN00", "COIN01"):
        markets[coin]["warmup_minutes_source"] = "history"
    write_schedule(tmp_path / "selection.csv", inputs, [[1, 0], [0, 1]])
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    warm = compute_per_coin_warmup_minutes(config, for_trade_activation=True, include_rms=False)
    assert payload.backtest_params["trade_start_indices"][0] == warm.get("COIN00", warm["__default__"])
    assert payload.backtest_params["trade_start_indices"][1] > payload.backtest_params["trade_start_indices"][0]


def test_suite_schedule_override_is_frozen_scoped_and_preserves_expected_hash(tmp_path):
    from copy import deepcopy
    from suite_runner import SuiteScenario, apply_scenario
    inputs = fixture_inputs(sides="both", bars=2880)
    config = inputs[0]
    write_schedule(tmp_path / "base.csv", inputs, [[1, 0]])
    other = deepcopy(config)
    other["backtest"]["organillo_sha256"] = None
    write_schedule(tmp_path / "other.csv", (other, *inputs[1:]), [[0, 1]])
    scenario = SuiteScenario("other", None, None, None, None,
                            overrides={"backtest.organillo_csv": str(tmp_path / "other.csv")})
    kwargs = dict(master_coins=["COIN00", "COIN01"], master_ignored=[],
                  available_exchanges=["binance"], available_coins={"COIN00", "COIN01"})
    with pytest.raises(ValueError, match="SHA-256 differs"):
        apply_scenario(config, scenario, **kwargs)
    scenario.overrides["backtest.organillo_sha256"] = None
    resolved, coins = apply_scenario(config, scenario, **kwargs)
    assert coins == ["COIN01"]
    assert resolved["backtest"]["organillo_sha256"] == other["backtest"]["organillo_sha256"]
    assert resolved["backtest"]["organillo_sha256"] != config["backtest"]["organillo_sha256"]
    # A scenario subset may use a CSV over the full master universe.
    from dataclasses import replace
    scenario = replace(scenario, coins=["COIN01"])
    assert apply_scenario(config, scenario, **kwargs)[1] == ["COIN01"]
