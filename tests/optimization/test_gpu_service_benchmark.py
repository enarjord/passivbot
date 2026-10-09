import hashlib
import subprocess
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


@pytest.mark.parametrize('changes,message', [
    ({'dataset_id': 'other'}, 'identity'),
    ({'request_id': 'other'}, 'identity'),
    ({'metrics': {}}, 'metric set'),
    ({'metrics': {'x': float('nan')}}, 'Nonfinite'),
    ({'metrics': {'x': 2.0}}, 'Metrics differ'),
    ({'liquidated': True}, 'Liquidation differs'),
])
def test_result_validation_rejects_corrupt_or_inconsistent_results(changes, message):
    reference = SimpleNamespace(dataset_id='a', request_id='r', metrics={'x': 1.0}, liquidated=False)
    row = SimpleNamespace(**(vars(reference) | changes))
    with pytest.raises(RuntimeError, match=message):
        benchmark.validated_rounding(reference, row, 'a', 'r', ['x'])


def test_result_validation_remains_enabled_under_optimized_python():
    import subprocess
    import sys

    completed = subprocess.run([sys.executable, '-O', '-c', '''
from types import SimpleNamespace
from tools.gpu_service_benchmark import validated_rounding
row = SimpleNamespace(dataset_id='wrong', request_id='r', metrics={'x': 1.0}, liquidated=False)
try:
    validated_rounding(None, row, 'a', 'r', ['x'])
except RuntimeError as error:
    if 'identity' not in str(error):
        raise
else:
    raise SystemExit('Optimized Python skipped result validation')
'''], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_rss_tree_includes_children_of_every_thread_without_double_counting(tmp_path, monkeypatch):
    for pid, kib, threads in [(10, 100, {10: '20', 11: '30 20'}),
                              (20, 200, {20: '40'}), (30, 300, {30: ''}), (40, 400, {40: ''})]:
        root = tmp_path / str(pid)
        root.mkdir()
        (root / 'status').write_text(f'VmRSS:\t{kib} kB\n')
        for tid, children in threads.items():
            task = root / 'task' / str(tid)
            task.mkdir(parents=True)
            (task / 'children').write_text(children)
    def proc_path(value):
        return tmp_path / str(value).removeprefix('/proc/')
    monkeypatch.setattr(benchmark, 'Path', proc_path)
    assert benchmark.rss_tree(10) == 1000 * 1024


def test_rss_tree_counts_a_real_linux_worker_spawned_child():
    import subprocess
    import sys
    from pathlib import Path

    if sys.platform != 'linux' or not Path('/proc/self/task').is_dir():
        pytest.skip('Linux proc thread children are required')
    script = '''
import os, subprocess, sys
from pathlib import Path
from queue import Queue
from threading import Event, Thread, get_native_id
sys.path.insert(0, sys.argv[1])
from tools.gpu_service_benchmark import rss_tree
ready, done = Queue(), Event()
def worker():
    child = subprocess.Popen([sys.executable, '-c',
        "import sys; data=bytearray(64*1024**2); data[::4096]=b'x'*(len(data)//4096); print('ready',flush=True); sys.stdin.read(1)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        ready.put((child, get_native_id()))
        done.wait(20)
        child.communicate('x', timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
thread = Thread(target=worker)
thread.start()
try:
    child, tid = ready.get(timeout=15)
    if child.stdout.readline().strip() != 'ready':
        raise RuntimeError('Child did not initialize its resident allocation')
    root = Path(f'/proc/{os.getpid()}')
    children_file = root/'task'/str(tid)/'children'
    if not children_file.is_file():
        raise SystemExit(77)  # This kernel/mount lacks optional task child lists.
    if str(child.pid) not in children_file.read_text().split():
        raise RuntimeError('Child is not owned by the worker thread')
    if str(child.pid) in (root/'task'/str(os.getpid())/'children').read_text().split():
        raise RuntimeError('Fixture child unexpectedly belongs to the leader')
    def resident(pid):
        line = next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
        return int(line.split()[1]) * 1024
    child_rss, parent_rss = resident(child.pid), resident(os.getpid())
    measured = rss_tree(os.getpid())
    if child_rss < 32*1024**2 or measured < child_rss + parent_rss - 1024**2:
        raise RuntimeError(f'Worker child omitted: measured={measured}, parent={parent_rss}, child={child_rss}')
finally:
    done.set()
    thread.join(timeout=20)
    if thread.is_alive():
        raise RuntimeError('Worker child cleanup did not complete')
'''
    completed = subprocess.run([sys.executable, '-c', script, str(Path(benchmark.__file__).parents[1])],
                               capture_output=True, text=True, timeout=45)
    if completed.returncode == 77:
        pytest.skip('Linux worker child lists are unavailable')
    assert completed.returncode == 0, completed.stderr


def test_missing_proc_child_lists_do_not_report_root_only_rss(tmp_path, monkeypatch):
    root = tmp_path / '10'
    root.mkdir()
    (root / 'status').write_text('VmRSS:\t100 kB\n')
    (root / 'task' / '10').mkdir(parents=True)
    monkeypatch.setattr(benchmark, 'Path', lambda value: tmp_path / str(value).removeprefix('/proc/'))
    assert benchmark.rss_tree(10) is None


@pytest.mark.parametrize('metadata', [
    {'skipped': 'stub_module', 'runtime_compiled_source_stamp': None, 'expected_source_fingerprint': 'current'},
    {'runtime_compiled_source_stamp': None, 'expected_source_fingerprint': 'current'},
    {'runtime_compiled_source_stamp': 'old', 'expected_source_fingerprint': 'current'},
    {'runtime_compiled_source_stamp': None, 'expected_source_fingerprint': None},
])
def test_unverified_runtime_fails_before_fixture_or_device_preparation(monkeypatch, metadata):
    import rust_utils
    from tools import gpu_parity
    metadata = dict(runtime_compiled_sha256='binary', **metadata)
    def forbidden(*args):
        pytest.fail('unverified runtime reached fixture preparation')
    monkeypatch.setattr(gpu_parity, 'fixture_inputs', forbidden)
    monkeypatch.setattr(rust_utils, 'verify_loaded_runtime_extension', lambda: metadata)
    with pytest.raises(RuntimeError, match='source-fingerprint-verified'):
        benchmark.main(['--strategy', 'ema_anchor', '--report', 'unused.json'])


@pytest.mark.parametrize('response', [
    '', 'malformed', 'N/A, N/A',
    subprocess.TimeoutExpired('nvidia-smi', 5),
    subprocess.CalledProcessError(1, 'nvidia-smi'),
])
def test_failed_device_samples_do_not_claim_availability(monkeypatch, response):
    sampler = benchmark.Sampler()
    sampler.smi = 'installed-nvidia-smi'
    monkeypatch.setattr(benchmark, 'rss_tree', lambda pid: 100)
    def sample(*args, **kwargs):
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(benchmark.subprocess, 'check_output', sample)
    monkeypatch.setattr(sampler.stop, 'wait', lambda seconds: sampler.stop.set())
    sampler._sample()
    assert sampler.errors
    assert not benchmark.sampling_availability(sampler.rows)['global_device']


def test_successful_device_samples_support_availability_with_transient_gaps():
    assert not benchmark.sampling_availability([])['global_device']
    rows = [{'process_tree_rss_bytes': 100},
            {'process_tree_rss_bytes': None, 'global_device_used_bytes': 42, 'global_gpu_utilization_max_pct': 0}]
    assert benchmark.sampling_availability(rows) == {'process_tree_rss': False, 'global_device': True}


def test_real_worker_rss_control_skips_unsupported_child_lists(monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(Path, 'is_dir', lambda path: True)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=77, stderr=''))
    with pytest.raises(pytest.skip.Exception, match='worker child lists are unavailable'):
        test_rss_tree_counts_a_real_linux_worker_spawned_child()
