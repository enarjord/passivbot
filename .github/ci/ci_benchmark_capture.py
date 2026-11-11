"""Capture completed node IDs for the temporary CI matrix experiment."""

from collections import Counter
import json
import os
from pathlib import Path
import sys


class Capture:
    def __init__(self):
        self.completed = []

    def pytest_runtest_logreport(self, report):
        if report.when == "teardown":
            self.completed.append(report.nodeid)

    def pytest_sessionfinish(self):
        Path(os.environ["CI_BENCH_EXECUTED"]).write_text(json.dumps(self.completed))


def pytest_configure(config):
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(Capture())


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    for path in sorted(directory.glob("shard-*.json")):
        expected = Counter(json.loads(path.read_text())["selected"])
        completed = Counter(json.loads(path.with_suffix(".executed").read_text()))
        if completed != expected:
            raise SystemExit(
                f"{path.name}: {(expected - completed).total()} missing, "
                f"{(completed - expected).total()} extra executions"
            )
        print(f"{path.name}: {completed.total()} tests completed exactly once")
