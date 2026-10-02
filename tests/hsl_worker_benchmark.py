"""Offline CPU worker tuning benchmark using complete native HSL results.

PYTHONPATH=src:tests python tests/hsl_worker_benchmark.py --minutes 4000 \
    --lookback-days 1 --seconds 240 --workers 1 --max-workers 3

Use --fixed for a fixed-pool comparison. No timing thresholds or decisions are
mocked. The controller consumes actual completed-job intervals; all results must
have the same complete-result digest. Run comparisons without competing work.
"""

import json
import logging
import multiprocessing
import tempfile
import time

import passivbot_rust

from hsl_backtest_benchmark import build_fixture, fixture_parser, result_digest
from optimization.gpu.exact_autotune import ExactWorkerController, MIB
from rust_utils import verify_loaded_runtime_extension


def initialize(options):
    global fixture
    verify_loaded_runtime_extension()
    fixture = build_fixture(options)


def evaluate(_):
    began = time.perf_counter()
    result = passivbot_rust.run_backtest(*fixture)
    finished = time.perf_counter()
    return began, finished, result_digest(result)


def main():
    parser = fixture_parser()
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--fixed", action="store_true")
    options = parser.parse_args()
    if (
        options.minutes < 41
        or options.seconds <= 0
        or not 1 <= options.workers <= options.max_workers
    ):
        parser.error(
            "minutes >= 41, seconds > 0 and 1 <= workers <= max-workers required"
        )
    artifact = verify_loaded_runtime_extension()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    context = multiprocessing.get_context("spawn")
    records = []
    digest = None
    completed = 0
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as cache:
        controller = ExactWorkerController(
            options.workers,
            options.max_workers,
            [],
            per_worker=512 * MIB,
            mode="refresh",
            cache_dir=cache,
            hardware={"benchmark": artifact},
        )
        pool = context.Pool(
            controller.workers, initializer=initialize, initargs=(options,)
        )
        try:
            while time.perf_counter() - start < options.seconds:
                epoch = controller.epoch
                workers = controller.workers
                before = time.perf_counter()
                results = pool.map(evaluate, range(max(8, workers * 4)))
                wall = time.perf_counter() - before
                for began, finished, value in results:
                    if digest is not None and digest != value:
                        raise RuntimeError(
                            "Complete native result changed across worker counts"
                        )
                    digest = value
                    completed += 1
                    controller.record(finished - began, 0, began, finished, epoch=epoch)
                records.append(
                    dict(
                        workers=workers,
                        completed=len(results),
                        seconds=wall,
                        validations_per_second=len(results) / wall,
                    )
                )
                if not options.fixed:
                    controller.update()
                if controller.target != workers:
                    # Every submitted job above completed before replacing its pool.
                    pool.close()
                    pool.join()
                    pool = context.Pool(
                        controller.target, initializer=initialize, initargs=(options,)
                    )
                    controller.applied()
        finally:
            pool.close()
            pool.join()
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                dict(
                    fixture=vars(options),
                    artifact=artifact,
                    completed=completed,
                    seconds=elapsed,
                    workers=controller.workers,
                    cached_workers=controller.cache.read(
                        controller.key, "exact_workers", 1, options.max_workers
                    ),
                    result_sha256=digest,
                    waves=records,
                ),
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
