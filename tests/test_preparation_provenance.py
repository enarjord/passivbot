from copy import deepcopy
import hashlib
import json

import pytest

from preparation_provenance import load_preparation_reference, preparation_metadata


def report():
    return {
        "schema_version": 1, "kind": "prepared-inputs", "consumer": "optimize",
        "suite": {"enabled": True, "reducer": {"default": "mean"}},
        "acquisition_mode": "auto",
        "datasets": [{"exchange": "combined", "cache_dir": "caches/hlcvs_data/example",
                      "ordered_coins": ["ETH", "BTC"], "rows": 20,
                      "manifest_present": True, "manifest_sha256": "a" * 64}],
        "cases": [{"label": "early", "start_date_literal": "2021-04",
                   "end_date_literal": "2023-01-01", "exchanges": ["combined"],
                   "ordered_coins": {"combined": ["ETH", "BTC"]},
                   "rows": {"combined": 10}}],
        "scope": "Owner-validated prepared inputs only; no strategy replay or fitness.",
    }


def reference(tmp_path, value):
    path = tmp_path / "prepared.json"
    raw = json.dumps(value, indent=2).encode()
    path.write_bytes(raw)
    loaded = load_preparation_reference(path)
    assert loaded.sha256 == hashlib.sha256(raw).hexdigest()
    return loaded


def test_reference_allows_only_storage_and_acquisition_relocation(tmp_path):
    original = report()
    expected = reference(tmp_path, original)
    actual = deepcopy(original)
    actual["datasets"][0]["cache_dir"] = "other-root/caches/hlcvs_data/example"
    actual["acquisition_mode"] = "rest"
    result = preparation_metadata(actual, expected)
    assert result["reference_verified"] is True
    assert result["requested_manifest_sha256"] == expected.sha256
    actual["cases"].clear()
    assert result["report"]["cases"] == original["cases"]


@pytest.mark.parametrize("change", [
    lambda value: value["cases"][0].update(start_date_literal="2021-04-01"),
    lambda value: value["datasets"][0]["ordered_coins"].reverse(),
    lambda value: value["cases"][0]["ordered_coins"]["combined"].reverse(),
    lambda value: value["datasets"][0].update(rows=20.0),
    lambda value: value["cases"][0]["rows"].update(combined=True),
    lambda value: value["datasets"][0].update(manifest_sha256="b" * 64),
    lambda value: value["suite"]["reducer"].update(default="max"),
    lambda value: value["cases"][0].update(exchanges=["bybit"]),
    lambda value: value["cases"].append(deepcopy(value["cases"][0])),
])
def test_reference_rejects_changed_order_literal_type_or_input_fact(tmp_path, change):
    expected = reference(tmp_path, report())
    actual = report()
    change(actual)
    with pytest.raises(ValueError, match="Prepared inputs differ"):
        preparation_metadata(actual, expected)


@pytest.mark.parametrize("change", [
    lambda value: value.update(kind="preparation-plan"),
    lambda value: value.update(schema_version=True),
    lambda value: value.update(schema_version=2),
    lambda value: value["datasets"][0].update(manifest_present=False),
    lambda value: value["datasets"][0].update(manifest_sha256=None),
    lambda value: value.update(cases=[]),
])
def test_reference_requires_completed_verified_owner_report(tmp_path, change):
    value = report()
    change(value)
    with pytest.raises(ValueError):
        reference(tmp_path, value)


@pytest.mark.parametrize("raw", ['{"kind":1,"kind":2}', '{"schema_version":NaN}'])
def test_reference_rejects_ambiguous_json(tmp_path, raw):
    path = tmp_path / "prepared.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_preparation_reference(path)


def test_no_reference_records_honest_unverified_summary():
    value = report()
    value["datasets"][0].update(manifest_present=False, manifest_sha256=None)
    result = preparation_metadata(value, None)
    assert result["reference_verified"] is False
    assert result["requested_manifest_sha256"] is None


@pytest.mark.asyncio
async def test_invalid_reference_fails_cli_before_data_or_network(tmp_path, monkeypatch):
    import optimize
    path = tmp_path / "plan.json"
    path.write_text('{"schema_version":1,"kind":"preparation-plan"}')

    def forbidden(*args, **kwargs):
        pytest.fail("invalid preparation reference reached configuration/data loading")

    monkeypatch.setattr(optimize, "load_input_config", forbidden)
    monkeypatch.setattr(optimize.sys, "argv", ["optimize", "--preparation-manifest", str(path)])
    with pytest.raises(ValueError, match="completed optimizer preparation"):
        await optimize.main()
