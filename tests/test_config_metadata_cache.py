"""Derived defaults remain owned and follow the current Rust metadata snapshot."""

from copy import deepcopy

import pytest

from config import schema, strategy_spec


def test_template_and_path_caches_do_not_leak_caller_mutations():
    config = schema.get_template_config()
    expected = deepcopy(config)
    paths = strategy_spec.strategy_optimize_key_path_map("trailing_martingale")
    expected_paths = dict(paths)
    config["bot"]["long"]["strategy"].clear()
    config["optimize"]["bounds"].clear()
    config["backtest"]["scenarios"].clear()
    paths.clear()
    assert schema.get_template_config() == expected
    assert strategy_spec.strategy_optimize_key_path_map("trailing_martingale") == expected_paths
    schema._template_cache = None
    strategy_spec._optimize_path_cache.clear()
    assert schema.get_template_config() == expected
    assert strategy_spec.strategy_optimize_key_path_map("trailing_martingale") == expected_paths


def test_replaced_rust_metadata_rebuilds_defaults_and_paths(monkeypatch):
    if getattr(strategy_spec.pbr, "__is_stub__", False):
        pytest.skip("real Rust metadata required")
    original = strategy_spec.pbr.get_strategy_spec
    schema.get_template_config()
    old_paths = strategy_spec.strategy_optimize_key_path_map("trailing_martingale")
    selected = next(row for row in original("trailing_martingale")["parameters"]
                    if row["config_path"] == ["strategy", "long", "entry", "initial_qty_pct"])
    key = selected["optimize_key"]
    def changed(kind):
        spec = deepcopy(original(kind))
        if kind == "trailing_martingale":
            row = next(row for row in spec["parameters"] if row["optimize_key"] == key)
            row["default"] = 0.123
            row["optimize_key"] = "long_replacement_initial_qty"
        return spec
    try:
        with monkeypatch.context() as patch:
            patch.setattr(strategy_spec.pbr, "get_strategy_spec", changed)
            strategy_spec.get_strategy_spec.cache_clear()
            assert schema.get_template_config()["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] == 0.123
            paths = strategy_spec.strategy_optimize_key_path_map("trailing_martingale")
            assert key not in paths
            assert paths["long_replacement_initial_qty"] == old_paths[key]
    finally:
        strategy_spec.get_strategy_spec.cache_clear()
    assert strategy_spec.strategy_optimize_key_path_map("trailing_martingale") == old_paths
    assert schema.get_template_config()["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] == selected["default"]
