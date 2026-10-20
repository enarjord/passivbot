"""Authoring exports retain selected path context and reject unresolved sources."""

from copy import deepcopy
import json
import os
from pathlib import Path
import socket

import pytest

from config import load_prepared_config
from config.authoring import compact_config, explain_config, resolved_config
from config.overrides import parse_overrides
from config.schema import get_template_config
from tools.clean_config import main as clean_main


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*_args, **_kwargs):
        pytest.fail("authoring path checks must remain offline")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, denied)
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        monkeypatch.setattr(socket, name, denied)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def policy():
    source = get_template_config()
    source["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        source["bot"][side]["universe"] = {
            "approved_coins": ["BTC"], "ignored_coins": [],
        }
    return source


def coin_patch(span):
    return {"bot": {"long": {"strategy": {"trailing_martingale": {
        "entry": {"ema_span_0": span},
    }}}}}


def coin_span(path):
    loaded = load_prepared_config(str(path), verbose=False)
    parsed = parse_overrides(loaded, verbose=False)
    return parsed["coin_overrides"]["BTC"]["bot"]["long"]["strategy"][
        "trailing_martingale"
    ]["entry"]["ema_span_0"]


def symlink_source(tmp_path, monkeypatch, *, decoy):
    actual = tmp_path / "physical" / "deep" / "source.json"
    selected = tmp_path / "selected" / "link.json"
    source = policy()
    source["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    write_json(actual, source)
    selected.parent.mkdir()
    selected.symlink_to(actual)
    selected_policy = write_json(selected.parent / "coin.json", coin_patch(141.5))
    protected = {selected_policy: selected_policy.read_bytes()}
    if decoy:
        target_decoy = write_json(actual.parent / "coin.json", coin_patch(777.5))
        protected[target_decoy] = target_decoy.read_bytes()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert coin_span(selected) == 141.5
    return source, actual, selected, protected


@pytest.mark.parametrize("decoy", [False, True], ids=["only-beside-link", "target-decoy"])
@pytest.mark.parametrize("entrypoint", ["programmatic", "cli"])
def test_compact_selected_symlink_rebases_original_policy_when_output_moves(
    tmp_path, monkeypatch, decoy, entrypoint,
):
    source, actual, selected, protected = symlink_source(tmp_path, monkeypatch, decoy=decoy)
    before = deepcopy(source)
    original = actual.read_bytes()
    link_identity = (selected.lstat().st_ino, os.readlink(selected))
    destination = tmp_path / "output" / "search.json"
    write_json(destination.parent / "coin.json", coin_patch(888.5))
    if entrypoint == "programmatic":
        result = compact_config(source, base_config_path=str(selected), destination_path=str(destination))
        write_json(destination, result)
    else:
        assert clean_main([str(selected), str(destination), "--mode", "compact"]) == 0
        result = json.loads(destination.read_text())
    assert coin_span(destination) == 141.5
    # Ordinary moved outputs keep relative references to their selected files.
    assert not Path(result["coin_overrides"]["BTC"]["override_config_path"]).is_absolute()
    assert source == before
    assert actual.read_bytes() == original
    assert selected.is_symlink()
    assert (selected.lstat().st_ino, os.readlink(selected)) == link_identity
    assert {path: path.read_bytes() for path in protected} == protected


@pytest.mark.parametrize("decoy", [False, True], ids=["only-beside-link", "target-decoy"])
def test_compact_symlink_in_place_preserves_both_reload_aliases_and_atomic_target(
    tmp_path, monkeypatch, decoy,
):
    _source, actual, selected, protected = symlink_source(tmp_path, monkeypatch, decoy=decoy)
    original = actual.read_bytes()
    info = actual.stat()
    link_identity = (selected.lstat().st_ino, os.readlink(selected))
    assert clean_main([str(selected), "--mode", "compact", "--in-place"]) == 0
    assert actual.read_bytes() != original
    assert actual.stat().st_ino != info.st_ino
    assert actual.stat().st_mode == info.st_mode
    assert selected.is_symlink()
    assert (selected.lstat().st_ino, os.readlink(selected)) == link_identity
    assert coin_span(selected) == coin_span(actual) == 141.5
    assert {path: path.read_bytes() for path in protected} == protected
    assert not list(actual.parent.glob(f".{actual.name}.*.tmp"))
    after = actual.read_bytes()
    assert clean_main([str(selected), "--mode", "compact", "--in-place"]) == 0
    assert actual.read_bytes() == after
    assert (selected.lstat().st_ino, os.readlink(selected)) == link_identity


@pytest.mark.parametrize("entrypoint", ["programmatic", "cli"])
def test_compact_parent_symlink_dotdot_uses_filesystem_directory_resolution(
    tmp_path, monkeypatch, entrypoint,
):
    source_dir = tmp_path / "physical" / "source"
    (source_dir / "deep").mkdir(parents=True)
    source_link = tmp_path / "source-shortcut"
    source_link.symlink_to(source_dir / "deep", target_is_directory=True)
    source_path = source_link / ".." / "source.json"
    source = policy()
    source["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    actual_source = write_json(source_dir / "source.json", source)
    actual_policy = write_json(source_dir / "coin.json", coin_patch(141.5))
    decoy = write_json(tmp_path / "coin.json", coin_patch(777.5))
    output_dir = tmp_path / "physical" / "output"
    (output_dir / "deep").mkdir(parents=True)
    output_link = tmp_path / "output-shortcut"
    output_link.symlink_to(output_dir / "deep", target_is_directory=True)
    destination = output_link / ".." / "search.json"
    protected = {path: path.read_bytes() for path in (actual_source, actual_policy, decoy)}
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert coin_span(source_path) == 141.5
    if entrypoint == "programmatic":
        result = compact_config(source, base_config_path=str(source_path), destination_path=str(destination))
        write_json(destination, result)
    else:
        assert clean_main([str(source_path), str(destination), "--mode", "compact"]) == 0
    assert coin_span(destination) == coin_span(output_dir / "search.json") == 141.5
    assert not (tmp_path / "search.json").exists()
    assert {path: path.read_bytes() for path in protected} == protected


def set_source(source, side, field, value, context):
    selector = f"bot.{side}.universe.{field}"
    if context == "root":
        source["bot"][side]["universe"][field] = value
    elif context == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {selector: value}
    else:
        if context == "scenario-object":
            universe = {"approved_coins": [], "ignored_coins": [], field: value}
            overrides = {f"bot.{side}.universe": universe}
        else:
            overrides = {selector: value}
        source["backtest"]["scenarios"] = [{"label": "mixed", "overrides": overrides}]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
@pytest.mark.parametrize("context", ["root", "fixed", "scenario", "scenario-object"])
@pytest.mark.parametrize("missing", ["missing-coins-authoring-path-regression.json", "./missing-coins-authoring-path-regression"])
def test_mixed_missing_file_sources_fail_without_outputs_or_source_changes(
    tmp_path, monkeypatch, side, field, context, missing,
):
    monkeypatch.chdir(tmp_path)
    source = policy()
    set_source(source, side, field, ["BTC", missing], context)
    before = deepcopy(source)
    source_path = write_json(tmp_path / "source.json", source)
    source_bytes = source_path.read_bytes()
    with pytest.raises(FileNotFoundError, match="coin-list file not found"):
        resolved_config(source, base_config_path=str(source_path))
    flags = ({"scenario": "mixed"} if context.startswith("scenario") else
             {"optimize": True} if context == "fixed" else {})
    with pytest.raises(FileNotFoundError, match="coin-list file not found"):
        explain_config(source, "bot.global.hedge_mode", base_config_path=str(source_path), **flags)
    output = tmp_path / "resolved.json"
    with pytest.raises(SystemExit) as error:
        clean_main([str(source_path), str(output), "--mode", "resolved"])
    assert error.value.code == 2
    assert not output.exists()
    assert source_path.read_bytes() == source_bytes
    assert source == before


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
@pytest.mark.parametrize("context", ["root", "fixed", "scenario"])
def test_existing_coin_file_inside_mixed_source_stays_rejected(
    tmp_path, monkeypatch, field, context,
):
    monkeypatch.chdir(tmp_path)
    coins = write_json(tmp_path / "coins.json", ["ETH"])
    source = policy()
    set_source(source, "long", field, ["BTC", str(coins)], context)
    original = coins.read_bytes()
    with pytest.raises(ValueError, match="coin-list file remains unresolved inside a multi-item source"):
        resolved_config(source)
    assert coins.read_bytes() == original


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_literal_coin_tokens_in_multi_item_sources_remain_identifiers(tmp_path, monkeypatch, field):
    monkeypatch.chdir(tmp_path)
    source = policy()
    tokens = ["BTC", "bybit::BTCUSDT", "BTC/USDT:USDT", "BTC-USDT-SWAP"]
    source["bot"]["long"]["universe"][field] = tokens
    assert resolved_config(source)["bot"]["long"]["universe"][field] == tokens


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_single_coin_file_source_still_resolves(tmp_path, monkeypatch, field):
    monkeypatch.chdir(tmp_path)
    source = policy()
    coins = write_json(tmp_path / "coins.json", ["BTC", "bybit::BTCUSDT"])
    original = coins.read_bytes()
    source["bot"]["long"]["universe"][field] = [str(coins)]
    assert resolved_config(source)["bot"]["long"]["universe"][field] == ["BTC", "bybit::BTCUSDT"]
    assert coins.read_bytes() == original
