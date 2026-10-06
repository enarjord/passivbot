from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import session_artifacts as artifacts
from config_utils import get_template_config
from optimization.prepared_dataset_identity import materialized_dataset_identity


def test_coin_labels_are_sorted_bounded_and_preserve_market_identity():
    assert artifacts.coins_label(["XMR"]) == "XMR"
    assert artifacts.coins_label(["XMR", "BTC", "XMR"]) == "BTC_XMR"
    assert artifacts.coins_label([f"COIN{i}" for i in range(7)]) == "7_coins"
    assert artifacts.coins_label(["X" * 100, "Y" * 100]) == "2_coins"
    assert artifacts.coins_label([]) == "0_coins"
    assert artifacts.safe_component("../coin") != artifacts.safe_component("coin")
    assert artifacts.safe_component("a/b") != artifacts.safe_component("a:b")
    assert "/" not in artifacts.safe_component("../coin")
    assert len(artifacts.safe_component("X" * 200)) <= 80


def test_timestamp_is_utc_and_sorts_across_dates():
    assert artifacts.utc_timestamp(0) == "1970-01-01T00_00_00Z"
    assert artifacts.utc_timestamp(86_399_000) < artifacts.utc_timestamp(86_400_000)
    assert artifacts.utc_datetime(1) == "1970-01-01T00:00:00.001Z"


@pytest.mark.parametrize(
    "label", ["CON", "AUX", "NUL", "PRN", "COM1", "LPT9", "con", "aux"]
)
def test_reserved_windows_components_are_escaped(label):
    escaped = artifacts.safe_component(label)
    assert escaped.casefold() != label.casefold()
    assert escaped.startswith(label + "-")
    assert artifacts.safe_component(escaped) == escaped


def test_numpy_setup_inputs_are_saved_with_the_same_identity(tmp_path):
    directory, manifest = artifacts.create_session_dir(
        tmp_path,
        coins=["XMR"],
        source="binance",
        span="1days",
        setup={"seed": np.int64(42)},
    )
    saved = json.loads((directory / artifacts.SESSION_MANIFEST).read_text())
    assert saved["setup"] == {"seed": 42}
    assert saved["setup_sha256"] == artifacts.setup_hash({"seed": 42})


def test_optimizer_records_actual_seed_and_restores_it_on_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts.secrets, "randbits", lambda n: 123)
    options = {"seed": None}
    artifacts.resolve_optimizer_seed(options)
    assert options["seed"] == 123
    (tmp_path / artifacts.SESSION_MANIFEST).write_text('{"seed":123}')
    resumed = {"seed": None}
    artifacts.resolve_optimizer_seed(resumed, tmp_path)
    assert resumed == options
    explicit = {"seed": 456}
    artifacts.resolve_optimizer_seed(explicit, tmp_path)
    assert explicit["seed"] == 456


def test_legacy_resume_preserves_unspecified_seed(tmp_path):
    options = {"seed": None}
    artifacts.resolve_optimizer_seed(options, tmp_path)
    assert options["seed"] is None


def test_setup_projection_ignores_paths_logging_and_output_settings():
    config = get_template_config()
    config["optimize"]["seed"] = 42
    other = deepcopy(config)
    other["backtest"].update(
        base_dir="another",
        cache_dir={"binance": "elsewhere"},
        visible_metrics=["nothing"],
        offline=True,
    )
    other["live"].update(user="some_account", base_config_path="other.json")
    other["logging"] = {"level": 4}
    other["results_dir"] = "generated/path"
    other["optimize"].update(compress_results_file=False, write_all_results=False)
    assert artifacts.setup_hash(
        artifacts.effective_setup_config(config, optimize=True)
    ) == artifacts.setup_hash(artifacts.effective_setup_config(other, optimize=True))
    other["optimize"]["seed"] = 43
    assert artifacts.setup_hash(
        artifacts.effective_setup_config(config, optimize=True)
    ) != artifacts.setup_hash(artifacts.effective_setup_config(other, optimize=True))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("backtest", "starting_balance", 1234),
        ("optimize", "iters", 20),
        ("optimize", "population_size", 8),
    ],
)
def test_effective_execution_settings_change_setup(section, key, value):
    config = get_template_config()
    before = artifacts.setup_hash(
        artifacts.effective_setup_config(config, optimize=True)
    )
    config[section][key] = value
    assert (
        artifacts.setup_hash(artifacts.effective_setup_config(config, optimize=True))
        != before
    )


def test_setup_serialization_keeps_meaningful_list_order_and_rejects_nonfinite():
    assert artifacts.setup_hash({"a": 1, "b": 2}) == artifacts.setup_hash(
        {"b": 2, "a": 1}
    )
    assert artifacts.setup_hash([1, 2]) != artifacts.setup_hash([2, 1])
    with pytest.raises(ValueError):
        artifacts.setup_hash({"value": float("nan")})


def test_directory_collision_retries_without_reusing_existing_output(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(artifacts, "utc_timestamp", lambda *a: "2026-10-06T14_50_56Z")
    ids = iter(["a" * 32, "a" * 32, "b" * 32])
    monkeypatch.setattr(artifacts, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    kwargs = dict(
        coins=["XMR"],
        source="combined",
        span="1737days",
        setup={"seed": 42},
        scenarios=12,
    )
    first, first_meta = artifacts.create_session_dir(tmp_path, **kwargs)
    (first / "keep.txt").write_text("original")
    second, second_meta = artifacts.create_session_dir(tmp_path, **kwargs)
    assert first != second
    assert first.name.startswith(
        "2026-10-06T14_50_56Z_XMR_combined_1737days_suite-12sc_setup-"
    )
    assert first_meta["setup_sha256"] == second_meta["setup_sha256"]
    assert first_meta["run_id"] != second_meta["run_id"]
    assert (first / "keep.txt").read_text() == "original"
    assert len(first_meta["setup_sha256"]) == 64
    assert json.loads((second / artifacts.SESSION_MANIFEST).read_text()) == second_meta


def test_selected_starting_configs_are_frozen_and_hashed_without_metrics():
    seed = {"bot": {"long": {"value": 1}}, "metrics": {"old": 1}}
    stream, identity = artifacts.snapshot_starting_configs([seed])
    try:
        seed["bot"]["long"]["value"] = 2
        assert (
            list(artifacts.iter_starting_snapshot(stream))[0]["bot"]["long"]["value"]
            == 1
        )
        assert (
            list(artifacts.iter_starting_snapshot(stream))[0]["bot"]["long"]["value"]
            == 1
        )
        same, same_identity = artifacts.snapshot_starting_configs(
            [
                {"bot": {"long": {"value": 1}}, "metrics": {"old": 999}},
            ]
        )
        same.close()
        assert identity == same_identity
        different, different_identity = artifacts.snapshot_starting_configs([seed])
        different.close()
        assert identity != different_identity
    finally:
        stream.close()


def test_prepared_data_values_and_market_settings_change_identity():
    hlcvs = np.ones((3, 1, 4))
    btc, ts = np.ones(3), np.arange(3, dtype=np.int64)
    mss = {"XMR": {"qty_step": 1}}
    before = materialized_dataset_identity(["XMR"], hlcvs, btc, ts, mss)
    hlcvs[1, 0, 2] = 2
    assert materialized_dataset_identity(["XMR"], hlcvs, btc, ts, mss) != before
    hlcvs[1, 0, 2] = 1
    mss["XMR"]["qty_step"] = 2
    assert materialized_dataset_identity(["XMR"], hlcvs, btc, ts, mss) != before


def test_suite_span_is_date_envelope():
    label, metadata = artifacts.date_span(
        [
            {"backtest": {"start_date": "2026-01-01", "end_date": "2026-01-10"}},
            {"backtest": {"start_date": "2026-01-20", "end_date": "2026-01-25"}},
        ]
    )
    assert label == "24days"
    assert len(metadata["windows"]) == 2


def test_suite_span_preserves_resolved_time_of_day():
    config = {
        "backtest": {
            "start_date": "2026-01-01T12:00:00Z",
            "end_date": "2026-01-02T12:00:00Z",
        }
    }
    label, metadata = artifacts.date_span([config])
    assert label == "1days"
    assert metadata["windows"] == [config["backtest"]]


def test_artifact_manifest_uses_relative_paths(tmp_path):
    directory = tmp_path / "scenario"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    assert artifacts.artifact_paths(directory, tmp_path) == {
        "config.json": "scenario/config.json"
    }
