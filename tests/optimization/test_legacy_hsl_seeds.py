"""Legacy HSL seeds are parameter suggestions, never runtime or fitness authority."""
from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.hsl import generated_template
from config.schema import CONFIG_SCHEMA_VERSION, get_template_config
from optimization.bounds import Bound
from optimize import _build_starting_seed_config, configs_to_individuals_streaming, extract_configs


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("policy", ["threshold", "always", "never"])
@pytest.mark.parametrize("artifact", ["json", "pareto"])
def test_enabled_legacy_seed_is_loaded_under_current_schema(mode, policy, artifact, tmp_path):
    cfg = generated_template(get_template_config(), mode)
    cfg["config_version"] = "v8.4.0"
    cfg["live"]["hsl_engine"] = "legacy"
    block = cfg["bot"]["hsl"] if mode == "unified" else cfg["bot"]["long"]["hsl"]
    block.update(enabled=True, restart_after_red_policy=policy, red_threshold=0.234,
                 no_restart_drawdown_threshold=1.0, tier_ratios={"yellow": 0.5, "orange": 0.75})
    original = deepcopy(cfg)
    path = tmp_path / ("seed.json" if artifact == "json" else "seed_pareto.txt")
    path.write_text(json.dumps({"config": cfg, "fitness": 123}))
    before = path.read_bytes()
    seeds = extract_configs(str(path))
    assert len(seeds) == 1
    seed = _build_starting_seed_config(seeds[0])
    assert seed["config_version"] == CONFIG_SCHEMA_VERSION
    if mode == "unified":
        assert seed["live"]["hsl_signal_mode"] == "unified"
    hsl_path = ("bot", "hsl", "red_threshold") if mode == "unified" else ("bot", "long", "hsl", "red_threshold")
    bounds = [Bound.from_config("threshold", [0.1, 0.4])]
    individuals, count = configs_to_individuals_streaming(seeds, bounds, sig_digits=6, key_paths=[("threshold", hsl_path)])
    assert count == 1
    assert individuals == [(0.234,)]
    assert path.read_bytes() == before
    assert cfg == original
    assert "fitness" not in seed
    # The exception never applies to the main runtime/optimization config.
    with pytest.raises(ValueError, match="migrate-hsl"):
        prepare_config(cfg, verbose=False)


def test_bot_only_flat_legacy_seed_loads_best_effort():
    bot = deepcopy(get_template_config()["bot"])
    for side in ("long", "short"):
        bot[side].pop("hsl")
        bot[side].update(hsl_enabled=True, hsl_restart_after_red_policy="threshold", hsl_red_threshold=0.321)
    original = deepcopy(bot)
    result = _build_starting_seed_config(bot)
    assert result["bot"]["long"]["hsl"]["red_threshold"] == 0.321
    assert bot == original



def test_legacy_seed_parameters_are_evaluated_under_main_hsl_policy(tmp_path):
    from optimize import individual_to_config
    cfg = generated_template(get_template_config())
    cfg["bot"]["long"]["hsl"].update(enabled=True, restart_after_red_policy="never")
    cfg["optimize"]["fixed_runtime_overrides"]["bot.long.hsl.restart_after_red_policy"] = "never"
    base = prepare_config(cfg, verbose=False)
    seed = deepcopy(cfg)
    seed["config_version"] = "v8.4.0"
    seed["live"]["hsl_engine"] = "legacy"
    seed["bot"]["long"]["hsl"].update(restart_after_red_policy="threshold", red_threshold=0.234)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed))
    seeds = extract_configs(str(path))
    key_paths = [("long_hsl_red_threshold", ("bot", "long", "hsl", "red_threshold"))]
    bounds = [Bound.from_config("long_hsl_red_threshold", [0.1, 0.4])]
    individuals, _ = configs_to_individuals_streaming(seeds, bounds, sig_digits=6, key_paths=key_paths)
    result = individual_to_config(individuals[0], {}, [], base, key_paths=key_paths)
    policy = result["bot"]["long"]["hsl"]
    assert policy["enabled"] is True
    assert policy["restart_after_red_policy"] == "never"
    assert policy["red_threshold"] == 0.234



def test_legacy_hsl_numeric_seed_is_clamped_instead_of_rejected(tmp_path):
    cfg = get_template_config()
    cfg["config_version"] = "v8.4.0"
    cfg["live"]["hsl_engine"] = "legacy"
    cfg["bot"]["long"]["hsl"].update(enabled=True, restart_after_red_policy="threshold", red_threshold=2.0)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(cfg))
    seeds = extract_configs(str(path))
    assert len(seeds) == 1
    key_paths = [("long_hsl_red_threshold", ("bot", "long", "hsl", "red_threshold"))]
    bounds = [Bound.from_config("long_hsl_red_threshold", [0.1, 0.4])]
    individuals, _ = configs_to_individuals_streaming(seeds, bounds, sig_digits=6, key_paths=key_paths)
    assert individuals == [(0.4,)]
