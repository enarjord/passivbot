from tools.streamline_json import parse_separators
from utils import json_dumps_streamlined


def test_json_dumps_streamlined_defaults_to_spaced_inline_arrays():
    assert (
        json_dumps_streamlined({"score_weights_volume": [0, 1, 0.01]})
        == '{"score_weights_volume": [0, 1, 0.01]}'
    )


def test_streamline_json_separator_parser_keeps_compact_option():
    assert parse_separators(", :") == (", ", ": ")
    assert parse_separators(",:") == (",", ":")
