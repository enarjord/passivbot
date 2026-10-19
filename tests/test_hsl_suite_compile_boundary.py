"""Suite overrides remain canonical until the single runtime lowering boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from config.hsl import generated_template
from config.migrations.v9 import require_v9_runtime_config
from config.runtime_compile import compile_runtime_config
from config.schema import get_template_config
from optimize import SuiteEvaluator, _apply_config_overrides


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
def test_scenario_override_stays_canonical_and_lowers_effective_policy_once(mode):
    config = generated_template(get_template_config(), mode)
    scope = ("bot", "risk", "hsl") if mode == "portfolio" else ("bot", "long", "hsl")
    policy = config
    for key in scope:
        policy = policy[key]
    policy.update(enabled=True, red_threshold=0.2)
    original = deepcopy(config)
    context = SimpleNamespace(
        label="scenario", config=deepcopy(config), msss={},
        overrides={"bot.long.risk.max_positions": 2.0,
                   ".".join((*scope, "red_threshold")): 0.3},
    )
    suite = SuiteEvaluator.__new__(SuiteEvaluator)
    suite.base = SimpleNamespace(scoring_specs=[], limit_checks=[])
    suite.objective_bases = []
    candidate = suite.build_scenario_candidate_config(config, context)
    require_v9_runtime_config(candidate)
    assert "n_positions" not in candidate["bot"]["long"]
    assert candidate["bot"]["long"]["risk"]["max_positions"] == 2.0
    runtime = compile_runtime_config(candidate, runtime="backtest", record_step=False)
    assert runtime["bot"]["long"]["n_positions"] == 2.0
    assert runtime["bot"]["long"]["risk"]["max_positions"] == 2.0
    effective = runtime
    for key in scope:
        effective = effective[key]
    assert effective["red_threshold"] == 0.3
    assert config == original
    # Lowered transport never becomes an accepted alternate public layout.
    with pytest.raises(ValueError, match="n_positions.*retired"):
        compile_runtime_config(runtime, runtime="backtest", record_step=False)


def test_fixed_override_does_not_inject_runtime_aliases_into_authored_config():
    config = get_template_config()
    _apply_config_overrides(config, {"bot.long.risk.max_positions": 3.0})
    require_v9_runtime_config(config)
    assert config["bot"]["long"]["risk"]["max_positions"] == 3.0
    assert "n_positions" not in config["bot"]["long"]
