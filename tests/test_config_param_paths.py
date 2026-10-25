from copy import deepcopy

import pytest

from config import param_paths
from config.schema import get_template_config
from config.strategy_spec import get_supported_strategy_kinds, strategy_optimize_key_path_map


@pytest.mark.parametrize("kind", get_supported_strategy_kinds())
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("nested", [False, True])
def test_group_resolution_matches_single_keys_with_one_metadata_projection(
    monkeypatch, kind, mode, nested
):
    config = get_template_config()
    config['bot']["portfolio"]['strategy_kind'] = kind
    config["bot"]["portfolio"]["hsl"]["scope"] = mode
    if not nested:
        config["bot"]["long"] = {"risk": {"max_positions": 3.0}}
    keys = [
        *strategy_optimize_key_path_map(kind),
        *param_paths.OPTIMIZABLE_BOT_KEY_PATHS,
        "long_n_positions", "short_unstuck_ema_span_0",
        "hsl_drawdown_threshold_fraction", "hsl_drawdown_fraction_ema_span_minutes", "hsl_unknown",
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
    config['bot']["portfolio"]['strategy_kind'] = "trailing_martingale"
    keys = ["long_entry_ema_span_0_minutes", "long_ema_span_0_minutes",
            "long_n_positions", "hsl_drawdown_threshold_fraction"]
    first = dict(param_paths.iter_optimizer_key_paths(config, keys))
    config['bot']["portfolio"]['strategy_kind'] = "ema_anchor"
    config["bot"]["portfolio"]["hsl"]["scope"] = "portfolio"
    config["bot"]["long"] = {"risk": {"max_positions": 2.0}}
    second = dict(param_paths.iter_optimizer_key_paths(config, keys))
    assert first["long_entry_ema_span_0_minutes"] == (
        "bot", "long", "strategy", "trailing_martingale", "entry", "ema_span_0_minutes"
    )
    assert second["long_ema_span_0_minutes"] == (
        "bot", "long", "strategy", "ema_anchor", "ema_span_0_minutes"
    )
    assert first["long_n_positions"] == ("bot", "long", "risk", "max_positions")
    assert second["long_n_positions"] == ("bot", "long", "risk", "max_positions")
    assert second["hsl_drawdown_threshold_fraction"] == ("bot", "portfolio", "hsl", "drawdown_threshold_fraction")


def test_empty_and_hsl_only_groups_preserve_lazy_metadata_errors(monkeypatch):
    def unavailable(_config):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(param_paths, "_strategy_path_map_for_config", unavailable)
    config = {"bot": {"portfolio": {'hsl': {'scope': 'portfolio'}}}}
    assert list(param_paths.iter_optimizer_key_paths(config, [])) == []
    assert list(param_paths.iter_optimizer_key_paths(config, ["hsl_drawdown_threshold_fraction"])) == [
        ("hsl_drawdown_threshold_fraction", ("bot", "portfolio", "hsl", "drawdown_threshold_fraction"))
    ]
    paths = param_paths.iter_optimizer_key_paths(config, ["hsl_drawdown_threshold_fraction", "long_n_positions"])
    assert next(paths) == ("hsl_drawdown_threshold_fraction", ("bot", "portfolio", "hsl", "drawdown_threshold_fraction"))
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        next(paths)
