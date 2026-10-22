"""Logical common coordinates become existing native parameters, not new IDs."""
from copy import deepcopy

import pytest

from config.load import prepare_config
from config.universe import universe_source_view
from config_utils import clean_config
from optimization.native_datasets import NativeDatasetRegistry
from optimize import Evaluator, config_to_individual
from shared_arrays import SharedArrayManager
from tools.gpu_parity import build_parser, fixture_inputs


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["global", "long", "short"])
def test_projection_declines_fractional_authored_position_counts(kind, scope):
    from optimization.native_planning import _ExecutionProjection, execution_key

    config, *_ = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--bars", "128", "--coins", "1", "--sides", "both"]))
    key = execution_key(config)
    projection = _ExecutionProjection(config, key)
    for value in (1e-300, 4.5):
        changed = deepcopy(config)
        changed["bot"][scope]["risk"]["max_positions"] = value
        assert not projection.matches(changed)
        with pytest.raises(TypeError, match="max_positions.*serialization type"):
            execution_key(changed)
    for value in (4, 4.0):
        changed = deepcopy(config)
        changed["bot"][scope]["risk"]["max_positions"] = value
        assert projection.matches(changed)
        assert execution_key(changed) == key


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("independent_short,topology_search", [(False, False), (True, False), (False, True)])
def test_actual_registry_and_fast_planner_fan_out_global_gene(kind, independent_short, topology_search):
    original, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--bars", "128", "--coins", "1", "--sides", "both"]))
    raw = clean_config(original)
    raw["bot"] = {"portfolio": deepcopy(raw["bot"]["portfolio"]),
                  "global": deepcopy(universe_source_view(original)["bot"]["long"])}
    raw["optimize"]["bounds"] = {"global": {"unstuck": {"close_fraction": [0.02, 0.08]}}}
    if independent_short:
        raw["optimize"]["bounds"]["short"] = {"unstuck": {"close_fraction": [0.04, 0.09]}}
    if topology_search:
        raw["bot"]["short"] = {"risk": {"total_wallet_exposure_limit": 1.0}}
        raw["optimize"]["bounds"]["global"]["risk"] = {"total_wallet_exposure_limit": [0.0, 1.0]}
    config = prepare_config(raw, verbose=False)
    config["backtest"]["coins"] = deepcopy(original["backtest"]["coins"])
    manager = SharedArrayManager()
    registry = None
    try:
        candle_spec, _ = manager.create_from(candles)
        btc_spec, _ = manager.create_from(btc)
        evaluator = Evaluator({"binance": candle_spec}, {"binance": btc_spec},
                              {"binance": markets}, config)
        evaluator.timestamps = {"binance": timestamps}
        registry = NativeDatasetRegistry(evaluator,
            standalone_candle_coins={"binance": config["backtest"]["coins"]["binance"]})
        planner = registry.planner
        vector = config_to_individual(evaluator.config, evaluator.bounds,
                                      optimization_shape=evaluator.optimization_shape)
        keys = [key for key, _ in evaluator.key_paths]
        vector[keys.index("bot.global.unstuck.close_fraction")] = 0.07
        if independent_short:
            vector[keys.index("bot.short.unstuck.close_fraction")] = 0.055
        if topology_search:
            vector[keys.index("bot.global.risk.total_wallet_exposure_limit")] = 1.0
        first = planner.prepare("first", vector)
        assert len(first.requests) == 1
        parameters = first.requests[0].parameters
        assert parameters["long_unstuck_close_pct"] == 0.07
        assert parameters["short_unstuck_close_pct"] == (0.055 if independent_short else 0.07)
        assert not any(key.startswith("bot.global") for key in parameters)
        # The second call exercises the exact owned-class projection cache.
        vector[keys.index("bot.global.unstuck.close_fraction")] = 0.06
        if topology_search:
            vector[keys.index("bot.global.risk.total_wallet_exposure_limit")] = 0.0
        second = planner.prepare("second", vector)
        assert second.requests[0].parameters["long_unstuck_close_pct"] == 0.06
        assert len(registry.bindings) == (2 if topology_search else 1)
        assert "unstuck" not in universe_source_view(second.record_config)["bot"].get("long", {})
    finally:
        if registry is not None:
            registry.close()
        manager.cleanup()
