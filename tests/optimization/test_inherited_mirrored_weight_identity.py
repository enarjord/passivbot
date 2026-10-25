"""Derived weight formatting cannot change a prepared policy's fixed inputs."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from config import prepare_config
from config.bot import normalize_forager_score_weights
from config.bot_inheritance import bot_policy_source_view, get_bot_policy
from config.inherited_bounds import get_bound_plan
from config_utils import clean_config
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
from optimization.warmup import build_optimizer_vector_config
from optimize import Evaluator, SuiteEvaluator, config_to_individual
from shared_arrays import SharedArraySpec
from tools.gpu_parity import DEFAULT_METRICS, build_parser, fixture_inputs


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_mirrored_searched_weights_keep_logical_fixed_identity_and_static_pin_gate(kind):
    original, candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    columns = deepcopy(original["backtest"]["coins"])
    raw = clean_config(original)
    names = ("volume", "ema_readiness", "volatility", "unilateralness")
    weights = dict(zip(names, (2.0, 5.0, 3.0, 0.0)))
    for side in ("long", "short"):
        raw["bot"][side]["forager"]["score_weights"] = deepcopy(weights)
    raw["optimize"]["bounds"] = {"long": {"forager": {"score_weights": {
        name: [0.0, 1.0] for name in names
    }}}}
    raw["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    raw["optimize"]["scoring"] = [{"metric": "adg_fraction_strategy_equity", "goal": "max"}]
    raw["optimize"]["limits"] = []
    config = prepare_config(raw, verbose=False)
    config["backtest"]["coins"] = columns
    original_source = bot_policy_source_view(config)
    spec = SharedArraySpec("mirrored-candles", candles.shape, candles.dtype.str)
    base = Evaluator({"binance": spec}, {}, {"binance": markets}, config)
    entry = "entry." if kind == "trailing_martingale" else ""
    pin = f"bot.long.strategy.{kind}.{entry}ema_span_1_minutes"
    contexts = [SimpleNamespace(
        label=label, exchanges=["binance"], config=deepcopy(config),
        msss={"binance": markets}, overrides={pin: 3.5} if label == "stress" else {},
    ) for label in ("base", "stress")]
    suite = SuiteEvaluator(base, contexts, {"default": "mean"})
    bindings = []
    for context in contexts:
        effective = suite.build_scenario_candidate_config(config, context)
        dataset = PreparedGpuDataset(
            config=effective, markets=markets, exchange="binance", hlcvs=spec,
            btc=SharedArraySpec("mirrored-btc", (len(candles),), "<f8"),
            timestamps=SharedArraySpec("mirrored-timestamps", (len(candles),), "<i8"),
            candle_coins=columns["binance"], metrics=DEFAULT_METRICS,
        )
        bindings.append(ScenarioBinding(context.label, context.label, dataset))
    planner = NativeCandidatePlanner(suite, bindings)
    vector = config_to_individual(config, base.bounds, optimization_shape=base.optimization_shape)
    positions = {tuple(path): index for index, (_name, path) in enumerate(base.key_paths)}
    expected_fixed = normalize_forager_score_weights(weights, path="short")
    identities, keys = [], []
    for values in ((0.1, 0.3, 0.2, 0.0), (0.7, 0.1, 0.2, 0.0), (0.1, 0.3, 0.2, 0.0)):
        varied = list(vector)
        for name, value in zip(names, values):
            varied[positions[("bot", "long", "forager", "score_weights", name)]] = value
        candidate = build_optimizer_vector_config(varied, config)
        effective = suite.build_scenario_candidate_config(candidate, contexts[1])
        plan = get_bound_plan(effective)
        identities.append(plan.identity())
        fixed = json.loads(plan.source_signature)["omitted_fixed_values"]
        assert {path[-1]: value for path, value in fixed
                if path[1:4] == ["short", "forager", "score_weights"]} == expected_fixed
        assert bot_policy_source_view(effective)["bot"]["short"]["forager"]["score_weights"] == weights
        origins = {origin.target: origin for origin in get_bot_policy(effective).resolve()[1]}
        assert origins[("bot", "short", "forager", "score_weights", "volume")].kind == "derived"
        prepared = planner.prepare(str(values), varied)
        assert len(prepared.requests) == 2
        keys.append(prepared.effective_key)
    assert len(set(identities)) == 1
    assert keys[0] != keys[1] and keys[0] == keys[2]
    assert bot_policy_source_view(config) == original_source
    contexts[1].overrides[pin] = 4.5
    with pytest.raises(ValueError, match="dataset-owned execution inputs"):
        planner.prepare("changed-static-pin", varied)
