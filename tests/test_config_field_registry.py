"""Explicit field types cover defaults, sparse fields, and Rust metadata."""

from copy import deepcopy

import pytest

from config.field_registry import get_field_registry
from config.field_spec import CoinOverridePolicy, FieldDomain, FieldRegistry, FieldSpec, FieldType, NO_DEFAULT
from config.hsl import FIELDS as HSL_FIELDS, generated_template
from config.schema import get_template_config
from config.strategy_spec import get_strategy_spec, get_supported_strategy_kinds
from tools.clean_config import normalize_export_numbers
from tools.generate_config_field_reference import DEFAULT_OUTPUT, main, render_reference


def _settings(value, path=()):
    # Containers that own dynamic members or records have descriptors themselves.
    # Their numeric member contracts are checked separately below.
    if isinstance(value, dict) and value:
        for name, child in value.items():
            if (*path, name) != ("optimize", "bounds"):
                yield from _settings(child, (*path, name))
    else:
        yield path, value
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _settings(item, (*path, index))


def test_every_template_setting_has_an_explicit_descriptor():
    registry = get_field_registry()
    missing = [".".join(map(str, path)) for path, _ in _settings(get_template_config())
               if registry.find(path) is None]
    assert missing == []


def test_only_documented_optional_authority_lacks_exact_path_defaults():
    absent = {field.path for field in get_field_registry()
              if field.default is NO_DEFAULT and "*" not in field.path}
    assert absent == {("bot", "global", "hsl", field) for field in HSL_FIELDS}


def test_numeric_type_does_not_depend_on_default_spelling(monkeypatch):
    import config.schema

    template = get_template_config()
    template["bot"]["long"]["risk"]["max_positions"] = 7.0
    template["bot"]["long"]["unstuck"]["ema_span_0"] = 60
    monkeypatch.setattr(config.schema, "get_template_config", lambda: deepcopy(template))
    registry = get_field_registry()
    assert registry.find(("bot", "long", "risk", "max_positions")).numeric_type is int
    assert registry.find(("bot", "long", "unstuck", "ema_span_0")).numeric_type is float


@pytest.mark.parametrize("path", [
    "backtest.starting_balance", "backtest.gap_tolerance_ohlcvs_minutes",
    "bot.global.universe.minimum_coin_age_days", "bot.global.warmup.max_minutes", "live.candle_lock_timeout_seconds",
    "live.execution_delay_seconds", "live.fills_confirmation_overlap_minutes",
    "live.fills_recent_overlap_minutes", "live.forager_ws_candle_rest_audit_minutes",
    "live.inactive_coin_candle_ttl_minutes", "live.max_active_candle_tail_gap_minutes",
    "live.max_forager_candle_refresh_seconds", "live.warmup_jitter_seconds",
    "logging.memory_snapshot_interval_minutes", "logging.volume_refresh_info_threshold_seconds",
])
def test_fractional_consumer_fields_are_declared_as_float(path):
    parts = tuple(path.split("."))
    field = get_field_registry().find(parts)
    assert field.numeric_type is float
    for value in (2, 2.0, 2.25):
        payload = {}
        target = payload
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        exported = normalize_export_numbers(payload)
        target = exported
        for part in parts:
            target = target[part]
        assert type(target) is float
        assert target == value


@pytest.mark.parametrize("path,numeric_type", [
    ("live.balance_override", float),
    ("live.max_forager_candle_staleness_minutes", float),
    ("live.max_concurrent_api_requests", int),
    ("optimize.seed", int),
    ("optimize.population_size", int),
    ("optimize.gpu.batch_size", int),
    ("backtest.ew_half_life_days", float),
    ("bot.long.entry_cooldown.max_duration_minutes", float),
])
def test_nullable_fields_retain_declared_numeric_types(path, numeric_type):
    field = get_field_registry().find(tuple(path.split(".")))
    assert field.numeric_type is numeric_type
    assert field.nullable
    assert field.default is None


def test_registry_defaults_are_isolated_and_optional_authority_is_absent():
    template = get_template_config()
    registry = get_field_registry()
    field = registry.find(("backtest", "exchanges"))
    field.default.append("extra")
    assert get_template_config() == template
    assert get_field_registry().find(field.path).default == template["backtest"]["exchanges"]
    authority = registry.find(("bot", "global", "hsl", "enabled"))
    assert authority.default is NO_DEFAULT
    assert "enabled" not in template["bot"]["global"]["hsl"]


def test_all_rust_strategy_fields_are_typed_with_identical_paths_and_defaults():
    registry = get_field_registry()
    for kind in get_supported_strategy_kinds():
        spec = get_strategy_spec(kind)
        for parameter in spec["parameters"] + spec["fixed_parameters"]:
            native_path = parameter["config_path"]
            path = ("bot", native_path[1], "strategy", kind, *native_path[2:])
            field = registry.find(path)
            assert field.default == parameter["default"]
            assert field.source == "rust strategy metadata"
            if parameter in spec["parameters"]:
                assert field.numeric_type is float
                assert field.optimize_key == parameter["optimize_key"]
                assert field.recommended_bounds == tuple(parameter["bounds"])
            else:
                assert field.types == (FieldType.STRING,)
                assert field.optimize_key is None
                assert field.domain.choices == tuple(parameter["allowed_values"])


def test_optional_metadata_is_explicit_without_granting_runtime_authority():
    field = FieldSpec(("bot", "long", "test"), (FieldType.FLOAT,))
    assert field.scope is None
    assert field.consumers == ()
    assert field.domain is None
    assert field.coin_override is None
    assert field.order == 0
    assert field.recommended_bounds is None
    annotated = FieldSpec(
        field.path, field.types, scope="long", consumers=("live", "backtest"),
        domain=FieldDomain(minimum=0.0, finite=True),
        coin_override=CoinOverridePolicy.FORBIDDEN, order=3,
        recommended_bounds=(0.0, 1.0, 0.01), unit="fraction", denominator="balance",
    )
    assert annotated.domain.minimum == 0.0
    assert annotated.coin_override is CoinOverridePolicy.FORBIDDEN


def test_sparse_dynamic_fields_and_all_bound_types_are_declared():
    registry = get_field_registry()
    assert registry.find(("backtest", "market_settings", "overrides", "BTC", "min_cost")).numeric_type is float
    assert registry.find(("optimize", "limits", 17, "range", 1)).numeric_type is float
    assert registry.find(("optimize", "prefilter", "checks", 9, "weight")).numeric_type is float
    for scope in ("coin", "pside", "portfolio"):
        config = generated_template(get_template_config(), scope)
        exported = normalize_export_numbers(config)

        def check_bounds(bounds, path=()):
            if isinstance(bounds, dict):
                for name, child in bounds.items():
                    check_bounds(child, (*path, name))
            else:
                assert isinstance(bounds, list)
                assert all(type(value) is (int if path[-1] == "max_positions" else float)
                           for value in bounds), path

        check_bounds(exported["optimize"]["bounds"])


def test_coin_only_wallet_exposure_type_does_not_create_a_root_setting():
    registry = get_field_registry()
    field = registry.find(("coin_overrides", "BTC", "bot", "long", "wallet_exposure_limit"))
    assert field.numeric_type is float
    assert field.coin_override is CoinOverridePolicy.ALLOWED
    assert field.default is NO_DEFAULT
    assert registry.find(("bot", "long", "wallet_exposure_limit")) is None
    source = {"coin_overrides": {"BTC": {"bot": {"long": {"wallet_exposure_limit": 1}}}}}
    exported = normalize_export_numbers(source)
    assert type(exported["coin_overrides"]["BTC"]["bot"]["long"]["wallet_exposure_limit"]) is float


def test_serialization_metadata_does_not_round_or_validate_integer_inputs():
    source = {"live": {"max_n_creations_per_batch": 2.25, "warmup_ratio": None}}
    assert normalize_export_numbers(source) == source


def test_exporter_uses_registry_types_instead_of_template_numeric_values(monkeypatch):
    import config.field_registry

    registry = FieldRegistry([
        FieldSpec(("live", "leverage"), (FieldType.FLOAT,)),
        FieldSpec(("live", "warmup_ratio"), (FieldType.INTEGER,)),
    ])
    monkeypatch.setattr(config.field_registry, "get_field_registry", lambda: registry)
    payload = {"live": {"leverage": 3, "warmup_ratio": 1.0, "unknown": 2}}
    original = deepcopy(payload)
    exported = normalize_export_numbers(payload)
    assert payload == original
    assert type(exported["live"]["leverage"]) is float
    assert type(exported["live"]["warmup_ratio"]) is int
    assert type(exported["live"]["unknown"]) is int


def test_registry_rejects_duplicate_or_ambiguous_declarations():
    field = FieldSpec(("live", "leverage"), (FieldType.INTEGER,))
    with pytest.raises(ValueError, match="duplicate field path"):
        FieldRegistry([field, field])
    with pytest.raises(TypeError, match="explicitly declared"):
        FieldSpec(("x",), ("integer",))
    with pytest.raises(ValueError, match="choose one numeric"):
        FieldSpec(("x",), (FieldType.INTEGER, FieldType.FLOAT))
    registry = FieldRegistry([
        FieldSpec(("a", "*", "c"), (FieldType.STRING,)),
        FieldSpec(("a", "b", "*"), (FieldType.STRING,)),
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        registry.find(("a", "b", "c"))


def test_generated_reference_is_current_and_check_does_not_write(tmp_path):
    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == render_reference()
    path = tmp_path / "reference.md"
    assert main(["--check", "--output", str(path)]) == 1
    assert not path.exists()
    assert main(["--output", str(path)]) == 0
    before = path.read_bytes()
    assert main(["--check", "--output", str(path)]) == 0
    assert path.read_bytes() == before
