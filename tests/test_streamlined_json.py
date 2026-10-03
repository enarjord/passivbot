from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

import pytest

from json_utils import dump_json_streamlined, json_dumps_streamlined
from tools.streamline_json import parse_separators


def test_json_dumps_streamlined_defaults_to_spaced_inline_arrays():
    assert (
        json_dumps_streamlined({"score_weights_volume": [0, 1, 0.01]})
        == '{"score_weights_volume": [0, 1, 0.01]}'
    )


def test_streamline_json_separator_parser_keeps_compact_option():
    assert parse_separators(", :") == (", ", ": ")
    assert parse_separators(",:") == (",", ":")


@pytest.mark.parametrize("indent", [0, 2, 4, None])
@pytest.mark.parametrize("max_inline", [0, 60, 10000])
@pytest.mark.parametrize("sort_keys", [False, True])
def test_streamlined_json_roundtrip_matches_standard_encoder(indent, max_inline, sort_keys):
    data = {
        "scenarios": [{"label": "base"}, {"label": "recent", "start_date": "2025-10-02"}],
        "keys": {10: "ten", 2: "two"},
        "escaped": 'quote" slash\\ newline\n braces{} brackets[] café',
        "tuple": (1, None, True, False, -0.0),
        "empty": [{}, []],
    }
    expected = json.loads(json.dumps(data, sort_keys=sort_keys))
    rendered = json_dumps_streamlined(data, indent=indent, max_inline=max_inline, sort_keys=sort_keys)
    assert json.loads(rendered) == expected
    assert list(json.loads(rendered)["keys"]) == list(expected["keys"])
    if indent is None:
        assert rendered == json.dumps(data, sort_keys=sort_keys)


@pytest.mark.parametrize("max_inline", [0, 60, 10000])
def test_custom_serialization_runs_once_and_preserves_unicode(max_inline):
    calls = []

    def encode(value):
        calls.append(value)
        return {"path": str(value), "labels": ["café", "東京"]}

    value = Path("example.json")
    rendered = json_dumps_streamlined({"value": value}, max_inline=max_inline, default=encode, ensure_ascii=False)
    assert calls == [value]
    assert "café" in rendered and "東京" in rendered
    assert json.loads(rendered) == {"value": {"path": "example.json", "labels": ["café", "東京"]}}


@pytest.mark.parametrize("max_inline", [0, 60, 10000])
@pytest.mark.parametrize("separators", [(", ", ": "), (",", ":")])
def test_encoded_key_collisions_preserve_every_object_member(max_inline, separators):
    data = {
        "nested": [{1: "numeric key", "1": "string key"}],
        "other": {None: "null key", "null": "string key"},
        "padding": "x" * 80,
        "empty": [{}, []],
    }
    expected = json.dumps(data, separators=separators)
    rendered = json_dumps_streamlined(data, max_inline=max_inline, separators=separators)
    # A normal dict decoder would hide the lost member this regression guards.
    assert json.loads(rendered, object_pairs_hook=list) == json.loads(
        expected, object_pairs_hook=list
    )
    assert rendered.count('"1"') == 2
    assert rendered.count('"null"') == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("max_inline", [0, 10000])
def test_strict_numbers_fail_before_writing(value, max_inline):
    stream = StringIO()
    with pytest.raises(ValueError, match="Out of range float"):
        dump_json_streamlined({"nested": [{"value": value}]}, stream, max_inline=max_inline, allow_nan=False)
    assert stream.getvalue() == ""
    assert json_dumps_streamlined(value) == json.dumps(value)


@pytest.mark.parametrize("max_inline", [0, 10000])
def test_unsupported_values_and_cycles_keep_standard_errors(max_inline):
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_dumps_streamlined({"value": object()}, max_inline=max_inline)
    with pytest.raises(TypeError, match="keys must be"):
        json_dumps_streamlined({(1, 2): "bad key"}, max_inline=max_inline)
    circular = []
    circular.append(circular)
    with pytest.raises(ValueError, match="Circular reference"):
        json_dumps_streamlined(circular, max_inline=max_inline)


def test_file_writer_and_utils_exports_use_same_formatter():
    from utils import dump_json_streamlined as legacy_dump, json_dumps_streamlined as legacy_dumps

    assert legacy_dump is dump_json_streamlined
    assert legacy_dumps is json_dumps_streamlined
    data = {"scenarios": [{"label": "base"}, {"label": "recent", "start_date": "2025-10-02"}]}
    stream = StringIO()
    dump_json_streamlined(data, stream, indent=2, separators=(",", ":"), sort_keys=True)
    assert stream.getvalue() == json_dumps_streamlined(data, indent=2, separators=(",", ":"), sort_keys=True)
    assert '{"label":"base"}' in stream.getvalue()
    assert json.loads(stream.getvalue()) == data


def test_formatter_import_uses_only_standard_library():
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r}); "
        "from json_utils import json_dumps_streamlined; "
        "assert 'utils' not in sys.modules and 'ccxt' not in sys.modules; "
        "assert json_dumps_streamlined({'x': [1, 2]}) == '{\"x\": [1, 2]}'"
    )
    subprocess.run([sys.executable, "-S", "-c", code], check=True)
