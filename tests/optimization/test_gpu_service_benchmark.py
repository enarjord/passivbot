import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tools import gpu_service_benchmark as benchmark


@pytest.mark.parametrize('flags', [
    ('--coins', '2'), ('--bars', '2879'), ('--candidates', '129'),
    ('--rounds', '1'), ('--tuning-windows', '9'), ('--max-rounds', '257'),
    ('--rounds', '4', '--max-rounds', '3'),
    ('--accumulation-delay', 'nan'), ('--accumulation-delay', '-1'),
    ('--accumulation-delay', '.2'),
])
def test_invalid_workload_fails_before_runtime_import(flags):
    with pytest.raises(SystemExit) as error:
        benchmark.main(['--strategy', 'ema_anchor', '--report', 'unused.json', *flags])
    assert error.value.code == 2


def test_tuning_evidence_requires_every_dataset_and_completed_windows():
    evidence = dict(samples=1000, seconds=1000, completed_windows=[])
    assert not benchmark.enough_windows(evidence, ['base', 'early', 'late'], 1)
    evidence['completed_windows'] = [{'dataset': 'base'}, {'dataset': 'base'}, {'dataset': 'early'}]
    assert not benchmark.enough_windows(evidence, ['base', 'early', 'late'], 1)
    evidence['completed_windows'].append({'dataset': 'late'})
    assert benchmark.enough_windows(evidence, ['base', 'early', 'late'], 1)
    assert not benchmark.enough_windows(evidence, ['base', 'early', 'late'], 2)


def test_source_digest_preserves_arrays_and_matches_full_byte_reference():
    arrays = (np.arange(24, dtype=np.float32).reshape(3, 2, 4),
              np.arange(3, dtype=np.float64), np.arange(3, dtype=np.int64))
    copies = [array.copy() for array in arrays]
    assert benchmark.fixture_digest(arrays) == hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
    for array, copy in zip(arrays, copies):
        np.testing.assert_array_equal(array, copy)
    arrays[0][0, 0, 0] = 99
    assert benchmark.fixture_digest(arrays) != benchmark.fixture_digest(copies)


def test_cpu_simulation_guard_restores_all_targets_after_failure():
    functions = [lambda: 1, lambda: 2, lambda: 3]
    backtest = SimpleNamespace(execute_backtest=functions[0], run_backtest=functions[1],
                               pbr=SimpleNamespace(run_backtest_bundle=functions[2]))
    with pytest.raises(RuntimeError, match='must not invoke CPU simulation'):
        with benchmark.forbid_cpu_simulations(backtest):
            for call in (backtest.execute_backtest, backtest.run_backtest, backtest.pbr.run_backtest_bundle):
                with pytest.raises(RuntimeError, match='must not invoke CPU simulation'):
                    call()
            backtest.pbr.run_backtest_bundle()
    assert [backtest.execute_backtest, backtest.run_backtest, backtest.pbr.run_backtest_bundle] == functions


def test_cli_routes_help_without_cuda_or_full_install_gate(monkeypatch):
    import sys
    from passivbot_cli import main as cli
    seen = []
    def invoke(module):
        seen.append((module, list(sys.argv)))
        return True, 0
    def forbidden():
        pytest.fail('help must not inspect full runtime dependencies')
    monkeypatch.setattr(cli, '_invoke_module_main', invoke)
    monkeypatch.setattr(cli, '_missing_full_install_markers', forbidden)
    assert cli.main(['tool', 'gpu-service-benchmark', '--help']) == 0
    assert seen == [('tools.gpu_service_benchmark', ['passivbot tool gpu-service-benchmark', '--help'])]


def test_scenario_preparation_aligns_dates_market_indices_and_requested_start():
    config = {"backtest": {"coins": {"binance": ["A", "B"]}, "start_date": "old", "end_date": "old"}}
    markets = {"A": {"first_valid_index": 0, "last_valid_index": 9, "warmup_minutes": 60},
               "B": {"first_valid_index": 0, "last_valid_index": 9},
               "__meta__": {"requested_start_ts": 0}}
    timestamps = np.arange(10, dtype=np.int64) * 60_000
    selected, settings = benchmark.scenario_metadata(config, markets, ["B"], timestamps, (5, 10))
    assert selected["backtest"]["coins"]["binance"] == ["B"]
    assert selected["backtest"]["start_date"] == "1970-01-01T00:05:00+00:00"
    assert selected["backtest"]["end_date"] == "1970-01-01T00:10:00+00:00"
    assert settings == {"B": {"first_valid_index": 0, "last_valid_index": 4},
                        "__meta__": {"requested_start_ts": 300_000}}
    assert config["backtest"]["start_date"] == "old" and markets["B"]["last_valid_index"] == 9
