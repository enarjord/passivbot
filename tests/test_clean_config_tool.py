"""Offline cleanup contracts and safe file/directory publication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import sys

import pytest

from config.hsl import generated_template
from config.load import prepare_config
from config.project import project_config
from config.schema import CONFIG_SCHEMA_VERSION, get_template_config
from config_utils import sanitize_prepared_config_for_dump
from json_utils import reformat_json_text
from passivbot_cli.main import main as cli_main
from tools.clean_config import (
    atomic_write,
    cleanup_config,
    discover_files,
    main,
    render_source,
)

EXAMPLE_CONFIGS = sorted(
    (Path(__file__).resolve().parents[1] / "configs" / "examples").glob("*.json")
)


def write_config(path, cfg=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(get_template_config() if cfg is None else cfg), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("source", EXAMPLE_CONFIGS, ids=lambda path: path.name)
@pytest.mark.parametrize("mode", ["full", "live"])
def test_public_example_exports_are_reusable_and_idempotent(source, mode, tmp_path):
    original = source.read_bytes()
    destination = tmp_path / source.name
    assert main([str(source), str(destination), "--mode", mode]) == 0
    output = destination.read_text()
    assert render_source(output, source=destination, mode=mode) == output
    assert source.read_bytes() == original
    prepared = prepare_config(json.loads(output), live_only=True, verbose=False)
    assert (
        prepared["live"]["strategy_kind"]
        == json.loads(original)["live"]["strategy_kind"]
    )
    if mode == "live":
        assert "optimize" not in json.loads(output)
        assert "backtest" not in json.loads(output)


def test_full_cleanup_normalizes_result_envelope_and_preserves_authored_sources(
    tmp_path,
):
    cfg = get_template_config()
    cfg["live"]["approved_coins"] = "missing-coin-list.json"
    cfg["live"]["ignored_coins"] = "missing-ignored-list.json"
    cfg["backtest"]["end_date"] = "now"
    cfg["backtest"]["filter_by_min_effective_cost"] = None
    cfg["bot"]["long"]["risk"]["n_positions"] = 3.5
    cfg["bot"]["long"]["risk"]["dead"] = 777
    cfg["backtest"]["metrics"] = {"adg": 1.0}
    cfg["live"]["base_config_path"] = "old-runtime-path"
    cfg["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.risk.total_wallet_exposure_limit": 2.5
    }
    cfg["coin_overrides"] = {"BTC": {"live": {"leverage": 3.0}, "_note": "private"}}
    cfg["_raw"] = {"bad": "metadata"}
    wrapper = {"config": cfg, "analysis": {"adg": 123}, "timestamp": 456}
    original = deepcopy(wrapper)
    result = cleanup_config(wrapper, base_config_path=str(tmp_path / "input.json"))
    assert wrapper == original
    assert result["config_version"] == CONFIG_SCHEMA_VERSION
    assert set(result) == {
        "config_version",
        "bot",
        "live",
        "coin_overrides",
        "backtest",
        "optimize",
        "logging",
        "monitor",
    }
    assert result["live"]["approved_coins"] == "missing-coin-list.json"
    assert result["live"]["ignored_coins"] == "missing-ignored-list.json"
    assert result["backtest"]["end_date"] == "now"
    assert result["backtest"]["filter_by_min_effective_cost"] is None
    assert result["bot"]["long"]["risk"]["n_positions"] == 4
    assert "dead" not in result["bot"]["long"]["risk"]
    assert "metrics" not in result["backtest"]
    assert "base_config_path" not in result["live"]
    assert set(result["bot"]["long"]["strategy"]) == {"trailing_martingale"}
    assert result["coin_overrides"]["BTC"] == {"live": {"leverage": 3.0}}
    assert (
        result["bot"]["long"]["risk"]["total_wallet_exposure_limit"]
        == cfg["bot"]["long"]["risk"]["total_wallet_exposure_limit"]
    )
    assert (
        result["optimize"]["fixed_runtime_overrides"]
        == cfg["optimize"]["fixed_runtime_overrides"]
    )
    assert cleanup_config(result) == result


@pytest.mark.parametrize("mode", ["full", "live", "backtest", "optimize"])
@pytest.mark.parametrize("hsl_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize(
    "strategy", ["trailing_martingale", "ema_anchor", "trailing_grid_v7"]
)
def test_varied_configs_match_canonical_pipeline_and_roundtrip(
    mode, hsl_mode, strategy
):
    rng = random.Random(711)
    cfg = generated_template(get_template_config(), hsl_mode)
    cfg["live"]["strategy_kind"] = strategy
    cfg["live"]["approved_coins"] = {"long": ["BTC", "ETH"], "short": []}
    cfg["backtest"]["scenarios"] = [{"label": "base"}]
    cfg["backtest"]["coin_sources"] = {"BTC": "binance"}
    cfg["backtest"]["reducer"] = {"default": "mean", "adg_pnl": "min"}
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["risk"]["total_wallet_exposure_limit"] = rng.uniform(0.1, 2.0)
        cfg["bot"][side]["risk"]["n_positions"] = rng.randint(1, 10)
        cfg["bot"][side]["hsl"].update(enabled=True, restart_after_red_policy="never")
        cfg["bot"][side]["unstuck"]["ema_span_0"] = rng.uniform(5.1, 1000.9)
    if hsl_mode == "unified":
        cfg["bot"]["hsl"].update(enabled=True, restart_after_red_policy="never")
    target = "canonical" if mode == "full" else mode
    projected = project_config(cfg, target, record_step=False)
    expected = project_config(
        sanitize_prepared_config_for_dump(
            prepare_config(projected, live_only=True, verbose=False)
        ),
        target,
        record_step=False,
    )
    result = cleanup_config(cfg, mode=mode)
    assert result == expected
    assert cleanup_config(result, mode=mode) == result
    assert (
        result["bot"]["long"]["unstuck"]["ema_span_0"]
        == cfg["bot"]["long"]["unstuck"]["ema_span_0"]
    )
    if mode == "live":
        assert set(result) == {
            "config_version",
            "bot",
            "live",
            "logging",
            "monitor",
            "coin_overrides",
        }


def test_live_projection_discards_invalid_unused_blocks_and_omitted_side_stays_disabled():
    cfg = get_template_config()
    cfg["backtest"] = "unused garbage"
    cfg["optimize"] = {"bounds": "unused garbage"}
    del cfg["bot"]["long"]
    result = cleanup_config(cfg, mode="live")
    assert "backtest" not in result and "optimize" not in result
    assert result["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.0


@pytest.mark.parametrize(
    "source",
    [
        None,
        [],
        {},
        {"bot": {}},
        {"bot": "bad", "live": {}},
        {"universal_live_config": {}},
    ],
)
def test_non_configs_fail_instead_of_becoming_defaults(source):
    with pytest.raises((ValueError, TypeError)):
        cleanup_config(source)


def test_enabled_legacy_hsl_requires_explicit_migration():
    cfg = get_template_config()
    cfg["config_version"] = "v8.5.0"
    cfg["bot"]["long"]["hsl"]["enabled"] = True
    with pytest.raises(ValueError, match="migrate-hsl"):
        cleanup_config(cfg)


def test_supported_schema_renames_use_shared_migration():
    cfg = get_template_config()
    cfg["config_version"] = "v8.5.0"
    result = cleanup_config(cfg)
    assert result["config_version"] == CONFIG_SCHEMA_VERSION
    assert result["bot"]["long"]["hsl"]["restart_after_red_policy"] is None


def test_full_cleanup_normalizes_optimizer_policy_without_resolving_inputs():
    cfg = get_template_config()
    cfg["live"]["approved_coins"] = ["BTC", "ETH"]
    cfg["live"]["ignored_coins"] = "unreadable-coins.json"
    cfg["optimize"]["backend"] = " DEAP "
    cfg["optimize"]["population_size"] = "42"
    cfg["optimize"]["seed"] = "123"
    cfg["optimize"]["limits"] = []
    result = cleanup_config(cfg)
    assert result["live"]["approved_coins"] == ["BTC", "ETH"]
    assert result["live"]["ignored_coins"] == "unreadable-coins.json"
    assert result["optimize"]["backend"] == "deap"
    assert result["optimize"]["population_size"] == 42
    assert result["optimize"]["seed"] == 123
    assert result["optimize"]["limits"] == []
    assert cleanup_config(result) == result


@pytest.mark.parametrize("mode", ["full", "backtest", "optimize"])
@pytest.mark.parametrize("value", [[], ["BTC"], "binance", 17, True])
def test_invalid_backtest_coin_sources_fail_before_batch_publication(
    tmp_path, mode, value
):
    source = tmp_path / "src"
    valid = write_config(source / "a.json")
    cfg = get_template_config()
    cfg["backtest"]["coin_sources"] = value
    malformed = write_config(source / "z.json", cfg)
    original_valid, original_malformed = valid.read_bytes(), malformed.read_bytes()
    destination = tmp_path / "out"
    for args in [[str(source), str(destination)], [str(source), "--in-place"]]:
        with pytest.raises(SystemExit) as error:
            main([*args, "--mode", mode])
        assert error.value.code == 2
        assert valid.read_bytes() == original_valid
        assert malformed.read_bytes() == original_malformed
        assert not destination.exists()


@pytest.mark.parametrize("mode", ["full", "backtest", "optimize"])
def test_backtest_coin_sources_follow_shared_routing_validation(mode):
    cfg = get_template_config()
    cfg["backtest"]["coin_sources"] = {" BTC ": "binance", "ETH": "bybit"}
    result = cleanup_config(cfg, mode=mode)
    assert result["backtest"]["coin_sources"] == {"BTC": "binance", "ETH": "bybit"}
    cfg["backtest"]["coin_sources"] = {"BTC": "binance", " BTC ": "bybit"}
    with pytest.raises(ValueError, match="conflicting exchanges"):
        cleanup_config(cfg, mode=mode)
    cfg["backtest"]["coin_sources"] = None
    assert cleanup_config(cfg, mode=mode)["backtest"]["coin_sources"] == {}
    cfg["backtest"]["coin_sources"] = []
    assert "backtest" not in cleanup_config(cfg, mode="live")


@pytest.mark.parametrize(
    "section", ["backtest", "optimize", "logging", "monitor", "coin_overrides"]
)
def test_invalid_selected_section_reports_its_path(section):
    cfg = get_template_config()
    cfg[section] = "malformed"
    with pytest.raises(TypeError, match=f"config.{section} must be an object"):
        cleanup_config(cfg)


@pytest.mark.parametrize("max_inline", [0, 20, 10000])
def test_format_only_preserves_numbers_duplicate_members_and_arbitrary_payload(
    max_inline,
):
    text = '{"z": [1.234567890123456789, -0, 1e400, 1E-9999], "z": null, "s": "☃", "_raw": {"analysis": 3}, "yes": true}'
    formatted = render_source(
        text, source=Path("result.json"), mode="format", max_inline=max_inline
    )

    def decode(value):
        return json.loads(value, parse_int=str, parse_float=str, object_pairs_hook=list)

    assert decode(formatted) == decode(text)
    assert "1.234567890123456789" in formatted
    assert formatted.count('"z"') == 2
    assert formatted.endswith("\n")
    assert (
        render_source(
            formatted, source=Path("result.json"), mode="format", max_inline=max_inline
        )
        == formatted
    )


def test_formatter_sorts_only_when_requested():
    text = '{"z": 0, "a": {"y": 1, "x": 2}, "a": 3}'
    assert reformat_json_text(text).startswith('{"z":')
    assert (
        reformat_json_text(text, sort_keys=True)
        == '{"a": {"x": 2, "y": 1}, "a": 3, "z": 0}'
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"bot": {}, "bot": {}, "live": {}}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": 1,}',
    ],
)
def test_invalid_json_cleanup_does_not_write(tmp_path, text):
    source = tmp_path / "source.json"
    source.write_text(text)
    destination = tmp_path / "out.json"
    with pytest.raises(SystemExit) as error:
        main([str(source), str(destination)])
    assert error.value.code == 2
    assert source.read_text() == text
    assert not destination.exists()


@pytest.mark.parametrize("text", ['{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'])
def test_format_rejects_non_json_constants(text):
    with pytest.raises(ValueError, match="valid JSON"):
        reformat_json_text(text)


def test_cli_full_default_and_explicit_in_place(tmp_path):
    source = write_config(tmp_path / "source.json")
    original = source.read_bytes()
    destination = tmp_path / "nested" / "clean.json"
    assert cli_main(["tool", "clean-config", str(source), str(destination)]) == 0
    assert source.read_bytes() == original
    output = json.loads(destination.read_text())
    assert output == cleanup_config(json.loads(original))
    assert main([str(destination), "--check"]) == 0
    source.chmod(0o640)
    before_stat = source.stat()
    assert main([str(source), "--in-place"]) == 0
    after_stat = source.stat()
    assert stat.S_IMODE(after_stat.st_mode) == 0o640
    assert (after_stat.st_uid, after_stat.st_gid) == (
        before_stat.st_uid,
        before_stat.st_gid,
    )
    assert source.read_bytes() == destination.read_bytes()
    before_stat = source.stat()
    assert main([str(source), "--in-place"]) == 0
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


@pytest.mark.parametrize(
    "tail",
    [
        [],
        ["--overwrite"],
        ["--in-place", "destination.json"],
        ["--check", "destination.json"],
        ["--max-depth", "0", "--dry-run"],
        ["--indent", "-1", "--dry-run"],
    ],
)
def test_invalid_cli_combinations(tmp_path, tail):
    source = write_config(tmp_path / "source.json")
    original = source.read_bytes()
    with pytest.raises(SystemExit) as error:
        main([str(source), *tail])
    assert error.value.code == 2
    assert source.read_bytes() == original


def test_destination_protection_and_explicit_overwrite(tmp_path):
    source = write_config(tmp_path / "source.json")
    destination = tmp_path / "out.json"
    destination.write_text("old")
    destination.chmod(0o600)
    with pytest.raises(SystemExit):
        main([str(source), str(destination)])
    assert destination.read_text() == "old"
    assert main([str(source), str(destination), "--overwrite"]) == 0
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        main([str(source), str(source), "--overwrite"])


def test_directory_depth_structure_and_hjson(tmp_path):
    source = tmp_path / "src"
    one = write_config(source / "a.json")
    two = write_config(source / "sub" / "b.JSON")
    three = write_config(source / "sub" / "deeper" / "c.json")
    hjson_path = source / "sparse.hjson"
    hjson_path.write_text("{bot: {long: {}, short: {}}, live: {}}")
    assert discover_files(source) == [one]
    assert discover_files(source, max_depth=2) == [one, two]
    assert discover_files(source, max_depth=3) == [one, two, three]
    destination = tmp_path / "out"
    assert (
        main(
            [
                str(source),
                str(destination),
                "--max_depth",
                "2",
                "--include-hjson",
                "--mode",
                "live",
            ]
        )
        == 0
    )
    assert (destination / "a.json").exists()
    assert (destination / "sub" / "b.JSON").exists()
    assert not (destination / "sub" / "deeper").exists()
    assert (
        json.loads((destination / "sparse.hjson").read_text())["config_version"]
        == CONFIG_SCHEMA_VERSION
    )
    assert json.loads(one.read_text()) == get_template_config()


def test_bulk_preflight_failure_leaves_every_input_and_destination_untouched(tmp_path):
    source = tmp_path / "src"
    valid = write_config(source / "a.json")
    broken = source / "z.json"
    broken.write_text("broken JSON")
    before = valid.read_bytes()
    destination = tmp_path / "out"
    for args in [[str(source), str(destination)], [str(source), "--in-place"]]:
        with pytest.raises(SystemExit):
            main(args)
        assert valid.read_bytes() == before
        assert broken.read_text() == "broken JSON"
        assert not destination.exists()


def test_bulk_existing_output_preflight_does_not_create_other_outputs(tmp_path):
    source = tmp_path / "src"
    write_config(source / "a.json")
    write_config(source / "b.json")
    destination = tmp_path / "out"
    write_config(destination / "b.json", {"existing": True})
    with pytest.raises(SystemExit):
        main([str(source), str(destination)])
    assert not (destination / "a.json").exists()
    assert json.loads((destination / "b.json").read_text()) == {"existing": True}


def test_readonly_modes_and_empty_directory(tmp_path):
    source = write_config(tmp_path / "source.json")
    original = source.read_bytes()
    destination = tmp_path / "out" / "result.json"
    assert main([str(source), str(destination), "--dry-run"]) == 0
    assert main([str(source), "--dry-run"]) == 0
    assert main([str(source), "--check"]) == 1
    assert not destination.parent.exists()
    assert source.read_bytes() == original
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        main([str(empty), "--check"])


def test_symlinks_hardlinks_and_directory_overlap(tmp_path):
    source = tmp_path / "src"
    actual = write_config(source / "a.json")
    link = source / "link.json"
    link.symlink_to(actual)
    (source / "loop").symlink_to(source, target_is_directory=True)
    assert discover_files(source, max_depth=20) == [actual]
    assert main([str(link), "--in-place"]) == 0
    assert link.is_symlink()
    hardlink = tmp_path / "hard.json"
    os.link(actual, hardlink)
    with pytest.raises(SystemExit):
        main([str(actual), str(hardlink), "--overwrite"])
    with pytest.raises(SystemExit):
        main([str(actual), str(link), "--overwrite"])
    for destination in (source, source / "out", tmp_path):
        with pytest.raises(SystemExit):
            main([str(source), str(destination), "--dry-run"])


def test_atomic_failure_preserves_original_and_removes_temporary(tmp_path, monkeypatch):
    target = tmp_path / "source.json"
    target.write_text("original")

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write(target, b"new", overwrite=True)
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_exclusive_atomic_publish_keeps_existing_target(tmp_path):
    target = tmp_path / "source.json"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        atomic_write(target, b"new", overwrite=False)
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_hjson_selection_and_json_override(tmp_path):
    source = write_config(tmp_path / "source.hjson")
    destination = tmp_path / "result.json"
    assert (
        main(
            [
                str(source),
                str(destination),
                "--mode",
                "format",
                "--input-format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(destination.read_text()) == get_template_config()
    with pytest.raises(SystemExit):
        main([str(source), "--mode", "format", "--dry-run"])
    source.write_text("{bot: {}, live: {}, bot: {}}")
    with pytest.raises(SystemExit):
        main([str(source), "--dry-run"])


def test_public_cli_without_optional_dependencies_or_network(tmp_path):
    source = write_config(tmp_path / "source.json")
    cfg = json.loads(source.read_text())
    cfg["optimize"]["backend"] = "gpu"
    write_config(source, cfg)
    destination = tmp_path / "out.json"
    root = Path(__file__).resolve().parents[1]
    code = r"""
import importlib.abc
import socket
import sys
blocked = {"optimization.backends", "deap", "pymoo", "torch", "matplotlib", "plotly", "dash", "dictdiffer"}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
def deny_network(*args, **kwargs):
    raise AssertionError("cleanup attempted network access")
socket.socket.connect = deny_network
socket.getaddrinfo = deny_network
from passivbot_cli.main import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "tool",
            "clean-config",
            str(source),
            str(destination),
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(destination.read_text())["optimize"]["backend"] == "gpu"


def test_direct_script_help_and_dependency_free_format_mode(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "tools" / "clean_config.py"
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-depth" in result.stdout
    source = tmp_path / "source.json"
    source.write_text('{"a": 1e400}')
    destination = tmp_path / "out.json"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            str(source),
            str(destination),
            "--mode",
            "format",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert destination.read_text() == '{"a": 1e400}\n'
