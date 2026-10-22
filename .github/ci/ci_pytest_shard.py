"""Opt-in CI sharding after normal pytest collection and marker filtering."""

import json
from pathlib import Path
import sys

import pytest

from ci_sharding import FORMAT_VERSION, shard_for_path, validate_shard


def pytest_addoption(parser):
    group = parser.getgroup("CI sharding")
    group.addoption("--ci-shard-index", type=int, default=None)
    group.addoption("--ci-shard-count", type=int, default=None)
    group.addoption("--ci-shard-manifest", type=Path, default=None)


def pytest_configure(config):
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    if index is None and count is None and config.getoption("ci_shard_manifest") is None:
        return
    if index is None or count is None:
        raise pytest.UsageError("--ci-shard-index and --ci-shard-count must be supplied together")
    try:
        validate_shard(index, count)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    index = config.getoption("ci_shard_index")
    if index is None:
        return
    count = config.getoption("ci_shard_count")
    collected = [item.nodeid for item in items]
    selected, deselected = [], []
    for item in items:
        path = item.path.relative_to(config.rootpath).as_posix()
        if item.nodeid.split("::", 1)[0] != path:
            raise pytest.UsageError("collected node ID does not use its repository-relative file path")
        target = selected if shard_for_path(path, count) == index else deselected
        target.append(item)
    if not selected:
        raise pytest.UsageError(f"CI shard {index}/{count} selected no tests")
    manifest_path = config.getoption("ci_shard_manifest")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "format_version": FORMAT_VERSION,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "shard_index": index,
            "shard_count": count,
            "collected": collected,
            "selected": [item.nodeid for item in selected],
        }) + "\n", encoding="utf-8")
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
