import math
import os
import random
import json

import pytest

from pareto_store import ParetoStore
from pure_funcs import calc_hash
from opt_utils import round_floats


def _make_candidate(w_values, violation=0.0):
    """Build a minimal candidate dict compatible with ParetoStore expectations."""
    return {
        "metrics": {
            "definition_version": 2,
            "objectives": dict(w_values),
            "constraint_violation": violation,
        },
        "optimize": {"scoring": sorted(w_values.keys())},
    }


def _make_adg_candidate(value, *, include_scoring=True):
    candidate = {
        "metrics": {
            "definition_version": 2,
            "objectives": {"cagr_fraction_strategy_equity": value},
            "constraint_violation": 0.0,
        },
    }
    if include_scoring:
        candidate["optimize"] = {
            "scoring": [{"metric": "cagr_fraction_strategy_equity", "goal": "max"}],
        }
    return candidate


@pytest.mark.parametrize("sig_digits", [6])
def test_pareto_front_retains_per_metric_extremes(sig_digits, tmp_path):
    random.seed(42)

    # Build a synthetic candidate set with 3 objectives and some violating entries.
    candidates = []
    for i in range(300):
        obj = {
            "volatility_annual_fraction_strategy_equity": round(random.uniform(-0.002, 0.0), 8),
            "ulcer_index_fraction_strategy_equity": round(random.uniform(-0.001, 0.0), 8),
            "pain_index_fraction_strategy_equity": round(random.uniform(0.0, 1.0), 8),
        }
        violation = 0.0 if i % 5 else 1.0  # every 5th candidate violates constraints
        candidates.append(_make_candidate(obj, violation=violation))

    # Compute best (minimum) per-objective among non-violating candidates after rounding.
    non_violating = [
        round_floats(c, sig_digits) for c in candidates if c["metrics"]["constraint_violation"] <= 0
    ]
    assert non_violating, "Expected some non-violating candidates"

    best_per_metric = {}
    for cand in non_violating:
        for k, v in cand["metrics"]["objectives"].items():
            if k not in best_per_metric or v < best_per_metric[k]:
                best_per_metric[k] = v

    # Build Pareto front with pruning (max_size << total candidates).
    store = ParetoStore(
        directory=str(tmp_path),
        sig_digits=sig_digits,
        flush_interval=10_000,
        max_size=50,  # force pruning to check extreme preservation
    )
    for cand in candidates:
        store.add_entry(cand)

    front = store.get_front()
    front_hashes = {calc_hash(round_floats(cfg, sig_digits)) for cfg in front}

    # Assert each metric's minimum value is represented in the pruned front.
    for metric, min_val in best_per_metric.items():
        found = False
        for cfg in front:
            cfg_val = cfg["metrics"]["objectives"].get(metric)
            if cfg_val is None:
                continue
            if math.isclose(cfg_val, min_val, rel_tol=0.0, abs_tol=1e-12):
                found = True
                break
        assert found, f"Front missing extreme for {metric}: expected {min_val}"


def test_pareto_store_persists_candidate_without_extra_rounding(tmp_path):
    entry = {
        "bot": {
            "long": {
                "entry_exposure_ratio_coefficient": 1.384,
            },
            "short": {},
        },
        "metrics": {
            "definition_version": 2,
            "objectives": {"ulcer_index_fraction_strategy_equity": 0.5},
            "constraint_violation": 0.0,
        },
        "optimize": {
            "scoring": ["ulcer_index_fraction_strategy_equity"],
            "round_to_n_significant_digits": 3,
        },
    }

    store = ParetoStore(
        directory=str(tmp_path),
        sig_digits=3,
        flush_interval=10_000,
        max_size=50,
    )
    assert store.add_entry(entry)

    saved_files = list((tmp_path / "pareto").glob("*.json"))
    assert len(saved_files) == 1
    saved = json.loads(saved_files[0].read_text())

    assert saved["bot"]["long"]["entry_exposure_ratio_coefficient"] == 1.384


def test_pareto_store_single_pass_drops_and_removes_dominated_entries(tmp_path):
    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)

    assert store.add_entry(_make_candidate({"ulcer_index_fraction_strategy_equity": 1.0}))
    assert not store.add_entry(_make_candidate({"ulcer_index_fraction_strategy_equity": 2.0}))
    assert store.add_entry(_make_candidate({"ulcer_index_fraction_strategy_equity": 0.5}))

    front = store.get_front()
    assert len(front) == 1
    assert front[0]["metrics"]["objectives"]["ulcer_index_fraction_strategy_equity"] == 0.5


def test_pareto_store_bootstrap_failure_raises_and_preserves_files(tmp_path):
    pareto_dir = tmp_path / "pareto"
    pareto_dir.mkdir()
    corrupt = pareto_dir / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to load 1 existing Pareto file"):
        ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)

    assert corrupt.exists()


def test_pareto_store_flush_does_not_delete_files_that_were_never_loaded(tmp_path):
    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)
    unknown = tmp_path / "pareto" / "unknown.json"
    unknown.write_text(json.dumps(_make_candidate({"ulcer_index_fraction_strategy_equity": 1.0})), encoding="utf-8")

    store.flush_now()

    assert unknown.exists()


def test_pareto_store_bootstrap_uses_nonempty_scoring_from_later_file(tmp_path):
    pareto_dir = tmp_path / "pareto"
    pareto_dir.mkdir()
    first_without_scoring = pareto_dir / "000_without_scoring.json"
    later_with_scoring = pareto_dir / "001_with_scoring.json"
    first_without_scoring.write_text(
        json.dumps(_make_adg_candidate(1.0, include_scoring=False)),
        encoding="utf-8",
    )
    later_with_scoring.write_text(
        json.dumps(_make_adg_candidate(2.0, include_scoring=True)),
        encoding="utf-8",
    )

    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)

    front = store.get_front()
    assert store.scoring_keys == ["cagr_fraction_strategy_equity"]
    assert len(front) == 1
    assert front[0]["metrics"]["objectives"]["cagr_fraction_strategy_equity"] == 2.0


def test_pareto_store_rejects_scored_entry_after_unscored_front(tmp_path):
    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)
    assert store.add_entry(_make_adg_candidate(1.0, include_scoring=False))

    with pytest.raises(ValueError, match="after unscored entries"):
        store.add_entry(_make_adg_candidate(2.0, include_scoring=True))


def test_pareto_store_rejects_scoring_basis_mismatch(tmp_path):
    first = _make_adg_candidate(1.0)
    first["optimize"]["scoring"][0]["scenario"] = "base"
    second = _make_adg_candidate(2.0)
    second["optimize"]["scoring"][0]["scenario"] = None
    second["optimize"]["scoring"][0]["aggregate"] = "mean"
    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)

    assert store.add_entry(first)
    with pytest.raises(ValueError, match="optimize.scoring differs"):
        store.add_entry(second)


def test_pareto_store_rejects_missing_objective_values(tmp_path):
    entry = {
        "metrics": {
            "definition_version": 2,
            "objectives": {"ulcer_index_fraction_strategy_equity": None},
            "constraint_violation": 0.0,
        },
        "optimize": {"scoring": ["ulcer_index_fraction_strategy_equity"]},
    }
    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)

    with pytest.raises(ValueError, match="unavailable"):
        store.add_entry(entry)


def test_pareto_store_rejects_nan_objective_values(tmp_path):
    entry = {
        "metrics": {
            "definition_version": 2,
            "objectives": {"ulcer_index_fraction_strategy_equity": math.nan},
            "constraint_violation": 0.0,
        },
        "optimize": {"scoring": ["ulcer_index_fraction_strategy_equity"]},
    }
    store = ParetoStore(directory=str(tmp_path), sig_digits=6, flush_interval=10_000, max_size=50)

    with pytest.raises(ValueError, match="malformed"):
        store.add_entry(entry)


def test_pareto_store_keeps_legacy_filename_and_full_entry_identity(tmp_path):
    pareto_dir = tmp_path / "pareto"
    pareto_dir.mkdir()
    legacy = _make_candidate({
        "cagr_fraction_strategy_equity": 1.0,
        "ulcer_index_fraction_strategy_equity": 0.5,
    })
    new = _make_candidate({
        "cagr_fraction_strategy_equity": 2.0,
        "ulcer_index_fraction_strategy_equity": 1.0,
    })
    legacy_hash = calc_hash(legacy)
    new_hash = calc_hash(new)
    legacy_path = pareto_dir / f"{legacy_hash}.json"
    legacy_bytes = json.dumps(legacy, indent=2).encode()
    legacy_path.write_bytes(legacy_bytes)

    store = ParetoStore(str(tmp_path))
    assert store.add_entry(new)
    new_path = pareto_dir / f"{new_hash[:24]}.json"
    assert json.loads(new_path.read_bytes()) == new
    assert legacy_path.read_bytes() == legacy_bytes

    resumed = ParetoStore(str(tmp_path))
    assert not resumed.add_entry(legacy)
    assert not resumed.add_entry(new)
    assert set(resumed._entries) == {legacy_hash, new_hash}
    assert set(pareto_dir.iterdir()) == {legacy_path, new_path}


def test_pareto_store_extends_colliding_name_without_overwriting_an_untracked_file(
    tmp_path, monkeypatch
):
    store = ParetoStore(str(tmp_path))
    original = _make_candidate({
        "cagr_fraction_strategy_equity": 1.0,
        "ulcer_index_fraction_strategy_equity": 0.5,
    })
    new = _make_candidate({
        "cagr_fraction_strategy_equity": 2.0,
        "ulcer_index_fraction_strategy_equity": 1.0,
    })
    original_hash = calc_hash(original)
    new_hash = original_hash[:24] + calc_hash(new)[24:]

    def collision_hash(entry):
        return new_hash if entry == new else calc_hash(entry)

    monkeypatch.setattr("pareto_store.calc_hash", collision_hash)
    pareto_dir = tmp_path / "pareto"
    original_path = pareto_dir / f"{original_hash[:24]}.json"
    original_bytes = json.dumps(original, indent=2).encode()
    link = os.link

    def publish_after_competitor(source, destination):
        if destination == str(original_path):
            original_path.write_bytes(original_bytes)
        link(source, destination)

    # The short path is claimed by another writer just before publication.
    with monkeypatch.context() as m:
        m.setattr("pareto_store.os.link", publish_after_competitor)
        assert store.add_entry(new)
    new_path = pareto_dir / f"{new_hash[:32]}.json"
    assert json.loads(new_path.read_bytes()) == new
    assert original_path.read_bytes() == original_bytes
    store.flush_now()
    assert set(pareto_dir.iterdir()) == {original_path, new_path}

    resumed = ParetoStore(str(tmp_path))
    assert set(resumed._entries) == {original_hash, new_hash}
    assert not resumed.add_entry(new)
    assert resumed.add_entry(_make_candidate({
        "cagr_fraction_strategy_equity": 3.0,
        "ulcer_index_fraction_strategy_equity": 1.0,
    }))
    assert not new_path.exists()
    assert original_path.read_bytes() == original_bytes


def test_pareto_store_reuses_existing_file_with_same_full_hash(tmp_path):
    store = ParetoStore(str(tmp_path))
    entry = _make_adg_candidate(1.0)
    path = tmp_path / "pareto" / f"{calc_hash(entry)[:24]}.json"
    original_bytes = json.dumps(entry).encode()
    path.write_bytes(original_bytes)

    assert store.add_entry(entry)
    assert store.get_front() == [entry]
    assert path.read_bytes() == original_bytes
    assert list(path.parent.iterdir()) == [path]


def test_pareto_store_keeps_previous_front_when_publication_fails(tmp_path, monkeypatch):
    store = ParetoStore(str(tmp_path))
    objective = {"ulcer_index_fraction_strategy_equity": 0.5}
    original = _make_candidate(objective, violation=1.0)
    better = _make_candidate(objective, violation=0.0)
    assert store.add_entry(original)
    pareto_dir = tmp_path / "pareto"
    original_path = pareto_dir / f"{calc_hash(original)[:24]}.json"

    def reject_link(*args, **kwargs):
        raise PermissionError("publication denied")

    with monkeypatch.context() as m:
        m.setattr("pareto_store.os.link", reject_link)
        with pytest.raises(PermissionError, match="publication denied"):
            store.add_entry(better)

    assert store.get_front() == [original]
    assert list(pareto_dir.iterdir()) == [original_path]
    assert store.add_entry(better)
    assert store.get_front() == [better]
    assert not original_path.exists()


def test_pareto_store_skips_occupied_prefix_with_unreadable_json(tmp_path):
    store = ParetoStore(str(tmp_path))
    entry = _make_adg_candidate(1.0)
    digest = calc_hash(entry)
    path = tmp_path / "pareto" / f"{digest[:24]}.json"
    occupant = '{"int":' + "9" * 5000 + "}"
    path.write_text(occupant, encoding="utf-8")

    assert store.add_entry(entry)
    assert path.read_text(encoding="utf-8") == occupant
    assert json.loads(path.with_name(f"{digest[:32]}.json").read_bytes()) == entry
