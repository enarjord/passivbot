"""
Tests for handling mixed-type (non-string) keys in config dictionaries.
Addresses the fix in PR #537 for config key sorting when keys are not all strings.
"""

import pytest
from config_utils import format_config, get_template_config, remove_unused_keys_recursively


def test_remove_unused_keys_handles_integer_keys():
    """Test that remove_unused_keys_recursively handles integer keys without crashing."""
    template = {"a": 1, "b": 2}
    config = {"a": 1, "b": 2, 123: "invalid_int_key", 456: "another_int"}

    # Should not crash, and should remove integer keys
    remove_unused_keys_recursively(template, config, verbose=False)

    assert 123 not in config
    assert 456 not in config
    assert "a" in config
    assert "b" in config


def test_remove_unused_keys_handles_mixed_type_keys():
    """Test handling of various non-string key types."""
    template = {"valid_key": {"nested": 1}}
    config = {
        "valid_key": {"nested": 1},
        123: "int_key",
        45.6: "float_key",
        (1, 2): "tuple_key",
        True: "bool_key",
    }

    # Should remove all non-string keys
    remove_unused_keys_recursively(template, config, verbose=False)

    assert "valid_key" in config
    assert 123 not in config
    assert 45.6 not in config
    assert (1, 2) not in config
    assert True not in config


def test_remove_unused_keys_preserves_underscore_string_keys():
    """Test that string keys starting with underscore are preserved."""
    template = {"a": 1}
    config = {"a": 1, "_internal": "should_stay", "_raw": "metadata"}

    remove_unused_keys_recursively(template, config, verbose=False)

    assert "_internal" in config
    assert "_raw" in config
    assert "a" in config


def test_remove_unused_keys_sorting_stability():
    """Test that mixed-type key sorting doesn't cause crashes."""
    template = {"x": 1, "y": 2}
    config = {
        "x": 1,
        "y": 2,
        "z": 3,  # Will be removed
        100: "int",
        200: "int2",
        "a": 4,  # Will be removed
    }

    # Should process without crashes despite mixed types
    remove_unused_keys_recursively(template, config, verbose=False)

    # Template keys preserved
    assert "x" in config
    assert "y" in config
    # Non-template string keys removed
    assert "z" not in config
    assert "a" not in config
    # Integer keys removed
    assert 100 not in config
    assert 200 not in config


def test_remove_unused_keys_in_nested_structure():
    """
    Test that remove_unused_keys_recursively properly handles integer keys
    in nested structures during config cleanup (the core fix in PR #537).
    """
    template = {"section": {"valid_key": 1, "another_key": 2}}
    config = {
        "section": {
            "valid_key": 1,
            "another_key": 2,
            123: "invalid_int_key",
            "extra_string": "remove_me",
        },
        456: "root_level_int",
    }

    # This is what format_config does - should not crash
    remove_unused_keys_recursively(template, config, verbose=False)

    # Integer keys should be removed
    assert 456 not in config
    assert 123 not in config["section"]
    # Extra string keys should be removed
    assert "extra_string" not in config["section"]
    # Valid structure preserved
    assert "valid_key" in config["section"]
    assert "another_key" in config["section"]


def test_remove_unused_keys_with_numeric_string_keys():
    """Test that string representations of numbers are handled as strings."""
    template = {"1": "one", "2": "two"}
    config = {"1": "one", "2": "two", 1: "int_one", 2: "int_two"}

    remove_unused_keys_recursively(template, config, verbose=False)

    # String keys preserved
    assert "1" in config
    assert "2" in config
    # Integer keys removed
    assert 1 not in config
    assert 2 not in config


def test_json_coercion_scenario():
    """
    Test scenario where JSON parsing might coerce numeric strings to integers.
    This simulates real-world cases where configs might have mixed key types.
    """
    import json

    # This can happen when JSON is parsed in certain ways
    template = {"setting_1": True, "setting_2": False}
    config_with_int_keys = {
        "setting_1": True,
        "setting_2": False,
        0: "zero",  # Could come from JSON coercion
        1: "one",
    }

    # Should handle gracefully
    remove_unused_keys_recursively(template, config_with_int_keys, verbose=False)

    assert "setting_1" in config_with_int_keys
    assert "setting_2" in config_with_int_keys
    assert 0 not in config_with_int_keys
    assert 1 not in config_with_int_keys
