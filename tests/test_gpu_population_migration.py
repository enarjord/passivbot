"""Legacy population choices survive canonical loading without GPU dependencies."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from config import get_template_config, load_prepared_config, prepare_config
from config_utils import add_config_arguments, clean_config, format_config, update_config_with_args


def legacy_config(population=64, backend="gpu"):
    config = get_template_config()
    config["optimize"]["backend"] = backend
    config["optimize"]["gpu"]["population_size"] = population
    config["optimize"]["iters"] = 64
    return config


@pytest.mark.parametrize("normalize", [prepare_config, format_config, clean_config])
@pytest.mark.parametrize("current", [None, "auto", "none", "null", ""])
def test_legacy_population_survives_canonical_surfaces(normalize, current, caplog):
    source = legacy_config()
    source["optimize"]["population_size"] = current
    original = copy.deepcopy(source)
    prepared = normalize(source)
    assert prepared["optimize"]["population_size"] == 64
    assert prepared["optimize"]["iters"] == 64
    assert "population_size" not in prepared["optimize"]["gpu"]
    assert "Migrated optimize.gpu.population_size" in caplog.text
    # The raw cleaner preserves authored dates; canonical preparation resolves them.
    cleaned = clean_config(format_config(prepared, verbose=False))
    caplog.clear()
    assert clean_config(format_config(cleaned, verbose=False)) == cleaned
    assert "legacy" not in caplog.text
    assert source == original


@pytest.mark.parametrize("wrapper", [False, True])
def test_raw_file_missing_general_population_migrates(tmp_path, wrapper):
    source = legacy_config(backend=" GPU ")
    del source["optimize"]["population_size"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"config": source} if wrapper else source))
    prepared = load_prepared_config(str(path), verbose=False)
    assert prepared["optimize"]["backend"] == "gpu"
    assert prepared["optimize"]["population_size"] == 64


@pytest.mark.parametrize("legacy,expected", [("64", 64), (64.0, 64), (1, 8)])
def test_legacy_integer_choices_preserve_old_effective_minimum(legacy, expected):
    assert prepare_config(legacy_config(legacy), verbose=False)["optimize"]["population_size"] == expected


@pytest.mark.parametrize("legacy", [None, "auto"])
def test_legacy_auto_does_not_invent_an_explicit_population(legacy):
    prepared = prepare_config(legacy_config(legacy), verbose=False)
    assert prepared["optimize"]["population_size"] is None
    assert "population_size" not in prepared["optimize"]["gpu"]


def test_explicit_canonical_population_wins_with_warning(caplog):
    source = legacy_config()
    source["optimize"]["population_size"] = "32"
    prepared = prepare_config(source, verbose=False)
    assert prepared["optimize"]["population_size"] == 32
    assert "remains authoritative" in caplog.text
    assert "population_size" not in prepared["optimize"]["gpu"]


@pytest.mark.parametrize("legacy", [0, -1, True, 64.5, "64.5", "invalid", float("nan"), float("inf"), {}, []])
@pytest.mark.parametrize("normalize", [prepare_config, format_config, clean_config])
def test_invalid_legacy_population_rejected_before_pruning_even_with_new_value(legacy, normalize):
    source = legacy_config(legacy)
    source["optimize"]["population_size"] = 32
    with pytest.raises(ValueError, match=r"optimize\.gpu\.population_size.*positive integer"):
        normalize(source)


@pytest.mark.parametrize("backend", ["deap", "pymoo"])
@pytest.mark.parametrize("current", [None, 32])
def test_cpu_backend_preserves_general_population_and_ignores_unused_legacy(backend, current, caplog):
    source = legacy_config("invalid", backend=backend)
    source["optimize"]["population_size"] = current
    prepared = prepare_config(source, verbose=False)
    assert prepared["optimize"]["population_size"] == current
    assert "legacy optimize.gpu.population_size" not in caplog.text


def test_invalid_canonical_population_is_not_replaced_by_legacy_value():
    source = legacy_config()
    source["optimize"]["population_size"] = 0
    with pytest.raises(ValueError, match=r"optimize\.population_size must be > 0"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("selected,expected", [("gpu", 64), ("pymoo", None)])
def test_raw_cli_backend_override_precedes_population_migration(selected, expected):
    source = legacy_config(backend="deap")
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(parser, source, command="optimize", help_all=True)
    args = parser.parse_args(["--optimizer-backend", selected])
    update_config_with_args(source, args, allowed_keys=allowed)
    assert prepare_config(source, verbose=False)["optimize"]["population_size"] == expected


def test_canonical_migration_and_actual_population_plan_without_gpu_imports_or_simulation():
    # A fresh interpreter proves optional imports cannot be hidden by another test.
    script = textwrap.dedent("""
        import builtins
        import sys
        blocked = ("torch", "cupy", "optimization.gpu", "optimization.backends.gpu_backend",
                   "optimize", "backtest")
        original_import = builtins.__import__
        def host_only(name, *args, **kwargs):
            if any(name == p or name.startswith(p + ".") for p in blocked):
                raise AssertionError("population migration imported " + name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = host_only
        import passivbot_rust
        assert not getattr(passivbot_rust, "__is_stub__", False)
        def forbidden(*args, **kwargs):
            raise AssertionError("population migration ran a CPU backtest")
        passivbot_rust.run_backtest_bundle = forbidden
        from config import get_template_config, prepare_config
        from config_utils import clean_config, format_config
        from optimization.backends.pymoo_backend import _resolve_pymoo_population_plan
        source = get_template_config()
        source["optimize"].update(backend="gpu", population_size=None, iters=64)
        source["optimize"]["gpu"]["population_size"] = 64
        config = prepare_config(source, verbose=False)
        config = clean_config(format_config(config, verbose=False))
        for algorithm, n_obj in (("nsga2", 2), ("nsga3", 4)):
            config["optimize"]["pymoo"]["algorithm"] = algorithm
            plan = _resolve_pymoo_population_plan(config, n_obj=n_obj)
            assert plan["requested_population_size"] == 64
            assert plan["actual_population_size"] == 64
            assert max(1, config["optimize"]["iters"] // plan["actual_population_size"]) == 1
        config["optimize"]["pymoo"]["algorithms"]["nsga3"]["ref_dirs"]["n_partitions"] = 8
        plan = _resolve_pymoo_population_plan(config, n_obj=4)
        assert plan["requested_population_size"] == 64
        assert plan["actual_population_size"] == len(plan["ref_dirs"]) == 165
        assert not any(name == p or name.startswith(p + ".")
                       for name in sys.modules for p in blocked)
    """)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
