from tools.pareto_transform import convert_entry, select_prune_indices


def test_convert_entry_creates_metrics():
    entry = {
        "analyses_combined": {
            "adg_btc_mean": 0.1,
            "adg_btc_min": 0.05,
            "adg_btc_max": 0.2,
            "adg_btc_std": 0.01,
            "w_0": -0.001,
        }
    }
    changed = convert_entry(entry)
    assert changed is True
    metrics = entry["metrics"]
    assert metrics["stats"]["adg_btc"]["mean"] == 0.1
    assert metrics["stats"]["adg_btc"]["max"] == 0.2
    assert metrics["objectives"]["w_0"] == -0.001
    assert "analyses_combined" not in entry


def test_convert_entry_no_stats_returns_false():
    entry = {"analyses_combined": {"foo": 1}}
    assert convert_entry(entry) is False
    assert "metrics" not in entry


def test_convert_entry_handles_objectives_only():
    entry = {"analyses_combined": {"w_0": -0.1}}
    assert convert_entry(entry) is True
    assert entry["metrics"]["objectives"]["w_0"] == -0.1
    assert "stats" in entry["metrics"]


def test_select_prune_indices_prefers_diverse_points():
    entries = [
        {"metrics": {"objectives": {"w_0": 0.0, "w_1": 0.0}}},
        {"metrics": {"objectives": {"w_0": 0.1, "w_1": 0.1}}},
        {"metrics": {"objectives": {"w_0": 1.0, "w_1": 1.0}}},
        {"metrics": {"objectives": {"w_0": 0.0, "w_1": 1.0}}},
    ]
    keep = select_prune_indices(entries, target=2, seed=0)
    assert len(keep) == 2
    assert 2 in keep


def test_select_prune_indices_preserves_entries_without_objectives():
    entries = [
        {"extra": "no-objective"},
        {"metrics": {"objectives": {"w_0": 0.0}}},
        {"metrics": {"objectives": {"w_0": 1.0}}},
    ]
    keep = select_prune_indices(entries, target=2, seed=1)
    assert 0 in keep
    assert len(keep) == 2
