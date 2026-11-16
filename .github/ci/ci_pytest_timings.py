"""Report aggregate test-file durations in CI logs."""

from collections import Counter
import json


def pytest_configure(config):
    # xdist forwards worker reports to the controller's normal pytest hooks.
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(FileTimings())


class FileTimings:
    def __init__(self):
        self.seconds = Counter()
        self.tests = Counter()

    def pytest_runtest_logreport(self, report):
        if report.when in ("setup", "call", "teardown"):
            path = report.nodeid.split("::", 1)[0]
            self.seconds[path] += report.duration
            if report.when == "teardown":
                self.tests[path] += 1

    def pytest_terminal_summary(self, terminalreporter):
        for path, seconds in sorted(self.seconds.items(), key=lambda entry: (-entry[1], entry[0])):
            row = {"file": path, "seconds": round(seconds, 3), "tests": self.tests[path]}
            terminalreporter.write_line("CI_TEST_FILE " + json.dumps(row))
