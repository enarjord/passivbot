"""Released public v8.6 GPU policy migration; no account or private fixtures."""
from copy import deepcopy
import pytest

from test_v9_config_migration import legacy
from tools.migrate_config_v9 import migrate


# Released v8.6 defaults in public master, including the queue/drift controls
# consumed by its GPU backend. This synthetic input does not load old v9 files.
RELEASED_GPU = {
    "tuning_mode": "auto", "auto_lean_parallelism": True, "batch_size": None,
    "max_dispatch_candidate_bars": None, "checkpoint_interval_seconds": 5.0,
    "drift_halt": 0.6, "drift_rank_halt": None, "drift_objective_tolerance": 1e-6,
    "drift_min_samples": 32, "drift_probes": 4, "drift_window": 128,
    "exact_workers": None, "max_pending_exact": None, "population_size": None,
    "seed_bootstrap": {"max_exact": 128, "mode": "auto"},
    "screening": {"scenarios": [], "survival_fraction": 0.1, "min_survivors": 64},
    "validate_per_generation": 8,
}
RETIRED = {
    "auto_lean_parallelism", "drift_halt", "drift_rank_halt", "drift_objective_tolerance",
    "drift_min_samples", "drift_probes", "drift_window", "exact_workers",
    "max_pending_exact", "validate_per_generation",
}


def released(backend):
    config=legacy()
    config["optimize"].update(backend=backend, gpu=deepcopy(RELEASED_GPU))
    return config


@pytest.mark.parametrize("backend", ["gpu", "pymoo"])
def test_public_released_gpu_controls_are_reported_and_current_knobs_preserved(backend):
    source=released(backend);before=deepcopy(source)
    result=migrate(source,memory_span_minutes=43200.0)
    gpu=result.config["optimize"]["gpu"]
    assert set(gpu)=={"tuning_mode","batch_size","max_dispatch_candidate_bars",
                     "checkpoint_interval_seconds","screening"}
    assert gpu=={key:value for key,value in RELEASED_GPU.items()if key in gpu}
    assert result.config["optimize"]["backend"]==backend
    expected={"optimize.gpu."+key for key in RETIRED}
    expected|={"optimize.gpu.seed_bootstrap.max_exact","optimize.gpu.seed_bootstrap.mode"}
    reports={row["path"]:row for row in result.changes if row["path"]in expected}
    assert set(reports)==expected
    for key in RETIRED:
        assert reports["optimize.gpu."+key]["old"]==RELEASED_GPU[key]
    assert all(row["new"]is None and "retired released-v8 GPU"in row["reason"]for row in reports.values())
    assert source==before
    assert migrate(result.config).config==result.config


@pytest.mark.parametrize("backend,legacy_population,current,expected", [
    ("gpu",3,None,8),("gpu",257,None,257),("gpu",257,128,128),
    ("pymoo",257,None,None),("pymoo",257,128,128),
])
def test_population_alias_keeps_its_existing_backend_precedence(backend,legacy_population,current,expected):
    source=released(backend);source["optimize"]["gpu"]["population_size"]=legacy_population
    source["optimize"]["population_size"]=current
    result=migrate(source,memory_span_minutes=43200.0)
    assert result.config["optimize"]["population_size"]==expected
    assert "population_size"not in result.config["optimize"]["gpu"]


@pytest.mark.parametrize("backend", ["gpu","pymoo"])
@pytest.mark.parametrize("extra", [{"unrecognized_control":1},
    {"seed_bootstrap":{"mode":"auto","max_exact":128,"unrecognized_control":1}}])
def test_unknown_gpu_or_bootstrap_keys_are_not_erased(backend,extra):
    source=released(backend);source["optimize"]["gpu"].update(deepcopy(extra));before=deepcopy(source)
    with pytest.raises(ValueError,match="unknown configuration field"):
        migrate(source,memory_span_minutes=43200.0)
    assert source==before


@pytest.mark.parametrize("key",sorted(RETIRED)+["seed_bootstrap","population_size"])
@pytest.mark.parametrize("owner",["fixed_params","fixed_runtime_overrides"])
def test_retired_gpu_selectors_require_an_explicit_new_policy(owner,key):
    source=released("gpu");path="optimize.gpu."+key
    source["optimize"][owner]=[path]if owner=="fixed_params"else{path:1}
    before=deepcopy(source)
    with pytest.raises(ValueError,match="selects a retired GPU control"):
        migrate(source,memory_span_minutes=43200.0)
    assert source==before


@pytest.mark.parametrize("legacy_suite",[False,True])
def test_scenarios_still_cannot_author_optimizer_policy(legacy_suite):
    source=released("pymoo")
    scenario={"label":"search-policy","overrides":{"optimize.gpu.auto_lean_parallelism":False}}
    if legacy_suite:
        source["backtest"]["suite"]={"enabled":False,"scenarios":[scenario]}
    else:source["backtest"]["scenarios"]=[scenario]
    with pytest.raises(ValueError,match="optimizer controls must be top-level"):
        migrate(source,memory_span_minutes=43200.0)


@pytest.mark.parametrize("backend",["proxy","cpu","unsupported"])
def test_unsupported_backends_are_not_mapped(backend):
    source=released(backend)
    with pytest.raises(ValueError):migrate(source,memory_span_minutes=43200.0)


def test_deap_still_requires_search_backend_acknowledgment():
    source=released("deap")
    with pytest.raises(ValueError,match="acknowledge-search-backend-change"):
        migrate(source,memory_span_minutes=43200.0)


def test_deap_explicit_ack_keeps_authored_pymoo_policy():
    source=released("deap");policy=deepcopy(source["optimize"]["pymoo"]["shared"])
    result=migrate(source,memory_span_minutes=43200.0,acknowledge_search_backend_change=True)
    assert result.config["optimize"]["backend"]=="pymoo"
    assert result.config["optimize"]["pymoo"]["shared"]==policy


def test_deap_ack_still_requires_an_explicit_pymoo_policy():
    source=released("deap");source["optimize"].pop("pymoo")
    with pytest.raises(ValueError,match="explicit optimize.pymoo.shared policy"):
        migrate(source,memory_span_minutes=43200.0,acknowledge_search_backend_change=True)


@pytest.mark.parametrize("patch", [
    {"optimize":{"gpu":{"auto_lean_parallelism":False}}},
    {"optimize.gpu":{"seed_bootstrap":{"mode":"auto"}}},
])
def test_object_fixed_override_cannot_hide_a_retired_gpu_selector(patch):
    source=released("pymoo");source["optimize"]["fixed_runtime_overrides"]=patch
    with pytest.raises(ValueError,match="selects a retired GPU control"):
        migrate(source,memory_span_minutes=43200.0)


def test_empty_released_bootstrap_object_has_an_explicit_report():
    source=released("pymoo");source["optimize"]["gpu"]["seed_bootstrap"]={}
    result=migrate(source,memory_span_minutes=43200.0)
    assert "seed_bootstrap"not in result.config["optimize"]["gpu"]
    assert any(row["path"]=="optimize.gpu.seed_bootstrap"and row["old"]=={}and row["new"]is None
               for row in result.changes)


@pytest.mark.parametrize("value",[{"unknown":1},[1],float("nan")])
def test_retired_scalar_does_not_hide_an_unreleased_subtree(value):
    source=released("gpu");source["optimize"]["gpu"]["drift_halt"]=value
    with pytest.raises(ValueError,match="released scalar control"):
        migrate(source,memory_span_minutes=43200.0)
