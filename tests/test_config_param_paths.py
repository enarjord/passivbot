from copy import deepcopy

import pytest

from config import param_paths
from config.schema import get_template_config
from config.strategy_spec import get_supported_strategy_kinds, strategy_optimize_key_path_map


@pytest.mark.parametrize("kind", get_supported_strategy_kinds())
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("nested", [False, True])
def test_group_resolution_matches_single_keys_with_one_metadata_projection(
    monkeypatch, kind, mode, nested
):
    config = get_template_config()
    config["live"].update(strategy_kind=kind, hsl_signal_mode=mode)
    if not nested:
        config["bot"]["long"] = {"n_positions": 3.0}
    keys = [
        *strategy_optimize_key_path_map(kind),
        *param_paths.OPTIMIZABLE_BOT_KEY_PATHS,
        "long_n_positions", "short_unstuck_ema_span_0",
        "hsl_red_threshold", "hsl_ema_span_minutes", "hsl_unknown",
        "unknown", "other_n_positions", "long_unknown", "long_n_positions",
    ]
    before = deepcopy(config)
    expected = [(key, param_paths.resolve_optimizer_key_path(config, key)) for key in keys]
    original = param_paths._strategy_path_map_for_config
    calls = []

    def project(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(param_paths, "_strategy_path_map_for_config", project)
    assert list(param_paths.iter_optimizer_key_paths(config, iter(keys))) == expected
    assert len(calls) == 1
    assert config == before


def test_later_group_observes_current_strategy_mode_and_bot_shape():
    config = get_template_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    keys = ["long_ema_span_0", "long_n_positions", "hsl_red_threshold"]
    first = dict(param_paths.iter_optimizer_key_paths(config, keys))
    config["live"].update(strategy_kind="ema_anchor", hsl_signal_mode="unified")
    config["bot"]["long"] = {"n_positions": 2.0}
    second = dict(param_paths.iter_optimizer_key_paths(config, keys))
    assert first["long_ema_span_0"] == (
        "bot", "long", "strategy", "trailing_martingale", "entry", "ema_span_0"
    )
    assert second["long_ema_span_0"] == (
        "bot", "long", "strategy", "ema_anchor", "ema_span_0"
    )
    assert first["long_n_positions"] == ("bot", "long", "risk", "n_positions")
    assert second["long_n_positions"] == ("bot", "long", "n_positions")
    assert second["hsl_red_threshold"] == ("bot", "hsl", "red_threshold")


def test_empty_and_hsl_only_groups_preserve_lazy_metadata_errors(monkeypatch):
    def unavailable(_config):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(param_paths, "_strategy_path_map_for_config", unavailable)
    config = {"live": {"hsl_signal_mode": "unified"}}
    assert list(param_paths.iter_optimizer_key_paths(config, [])) == []
    assert list(param_paths.iter_optimizer_key_paths(config, ["hsl_red_threshold"])) == [
        ("hsl_red_threshold", ("bot", "hsl", "red_threshold"))
    ]
    paths = param_paths.iter_optimizer_key_paths(config, ["hsl_red_threshold", "long_n_positions"])
    assert next(paths) == ("hsl_red_threshold", ("bot", "hsl", "red_threshold"))
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        next(paths)
