"""Canonical CPU candidate preparation for a registered backtest service.

No device, replay, selection or simulation work belongs here. Fixed optimizer
policies and exact-last scenario overrides are materialized by the existing CPU
helpers before the transport adapter encodes effective scalar values.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from config.param_paths import iter_optimizer_key_paths
from config.strategy_spec import get_strategy_param_keys
from config_utils import clean_config
from optimization.bounds import enforce_bounds
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import BacktestRequest
from optimization.gpu.model import gpu_side_enabled
from optimization.gpu.parameters import prepare_candidate_parameters
from optimization.native_results import CandidateEvaluation, CanonicalResultScorer, ResultSlot
from optimizer_overrides import unstuck_ema_spans_coupled


def _remove_path(config, path):
    node = config
    for part in path[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def _static_contract(config):
    """Leave every execution value not represented by a dynamic request in place.

    Feature enablement, execution modes and coin overrides stay dataset-owned.
    Changing one requires another prepared dataset, never silently ignoring it.
    """
    cleaned = clean_config(config)
    result = {key: cleaned[key] for key in ("bot", "live", "backtest", "coin_overrides") if key in cleaned}
    # Numeric exposure/position values are dynamic, but their side enablement
    # chooses the registered replay's directional/fused kernel topology.
    result["enabled_sides"] = tuple(side for side in ("long", "short") if gpu_side_enabled(config, side))
    result["backtest"]["coins"] = config["backtest"]["coins"]
    # The standalone simulation consumes the materialized scenario, not the
    # optimizer's saved suite recipe or its candidate-dependent derived patches.
    for key in ("scenarios", "suite_enabled", "suite_reducers", "cache_dir", "base_dir"):
        result["backtest"].pop(key, None)
    kind = config["live"]["strategy_kind"]
    dynamic_bound_keys = []
    for side in ("long", "short"):
        for key in get_strategy_param_keys(kind):
            _remove_path(result, ("bot", side, "strategy", kind, *key.split(".")))
        dynamic_keys = (
            "n_positions", "total_wallet_exposure_limit", "risk_entry_cooldown_minutes",
            "risk_we_excess_allowance_pct", "risk_twel_enforcer_threshold", "risk_wel_enforcer_threshold",
            "forager_volume_ema_span_1m", "forager_volatility_ema_span_1m", "forager_volume_drop_pct",
            "unilateralness_ema_span_1m", "entry_cooldown_min_duration_minutes", "entry_cooldown_max_duration_minutes",
            "entry_cooldown_weights_minutes_exposure_ratio", "entry_cooldown_weights_minutes_adverse_directionality",
            "unstuck_close_pct", "unstuck_ema_dist", "unstuck_loss_allowance_pct", "unstuck_threshold",
            "unstuck_ema_span_0", "unstuck_ema_span_1", "hsl_red_threshold", "hsl_ema_span_minutes",
            "hsl_cooldown_minutes_after_red",
            *(f"forager_score_weights_{name}" for name in ("volume", "ema_readiness", "volatility", "unilateralness")),
        )
        dynamic_bound_keys.extend(f"{side}_{key}" for key in dynamic_keys)
    for _, path in iter_optimizer_key_paths(config, dynamic_bound_keys):
        if path is not None:
            _remove_path(result, path)
    for key in ("red_threshold", "ema_span_minutes", "cooldown_minutes_after_red"):
        _remove_path(result, ("bot", "hsl", key))
    coupled = unstuck_ema_spans_coupled(config)
    # CPU worker views lower coupled unstuck spans to ordinary request inheritance.
    # Strategy coin pins remain static; their derived unstuck copies must not
    # split otherwise identical execution views.
    result["coupled_unstuck_emas"] = coupled
    if coupled:
        for patch in result.get("coin_overrides", {}).values():
            for side in ("long", "short"):
                for name in ("ema_span_0", "ema_span_1"):
                    _remove_path(patch, ("bot", side, "unstuck", name))
                    _remove_path(patch, ("bot", side, f"unstuck_{name}"))
    return result


@dataclass(frozen=True)
class ScenarioBinding:
    scenario: str
    dataset_id: str
    dataset: PreparedGpuDataset


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    vector: tuple[float, ...]
    effective_key: str
    stage: str
    requests: tuple[BacktestRequest, ...]
    slots: tuple[ResultSlot, ...]

    def collector(self, scorer):
        return CandidateEvaluation(self.candidate_id, self.vector, self.slots, scorer, stage=self.stage)


class NativeCandidatePlanner:
    def __init__(self, evaluator, bindings, *, overrides_list=None):
        self.scorer = CanonicalResultScorer(evaluator)
        self.base = self.scorer.base
        self.overrides = tuple(overrides_list if overrides_list is not None else
                               self.base.config["optimize"].get("enable_overrides", []))
        self.bindings = tuple(bindings)
        self._variants = {}
        for binding in self.bindings:
            pair = (binding.scenario, binding.dataset.exchange)
            variants = self._variants.setdefault(pair, {})
            key = execution_key(json.loads(binding.dataset.config_json))
            if key in variants:
                raise ValueError("scenario/exchange execution variants must be unique")
            variants[key] = binding
        self.scorer.validate_coverage(list(self._variants), "full")
        if len({binding.dataset_id for binding in self.bindings}) != len(self.bindings):
            raise ValueError("prepared dataset identities must be unique")

    def prepare(self, candidate_id, vector, *, scenarios=None):
        from optimize import _canonicalize_optimizer_individual

        values = [float(value) for value in vector]
        if len(values) != len(self.base.bounds) or any(not math.isfinite(value) for value in values):
            raise ValueError("candidate must have a complete finite optimizer vector")
        values = enforce_bounds(values, self.base.bounds, self.base.sig_digits)
        config = _canonicalize_optimizer_individual(
            values, self.base.config, self.base.bounds, self.base.sig_digits,
            self.base.key_paths, self.overrides,
        )
        configs = {"base": config}
        if self.scorer.suite:
            configs = {ctx.label: self.scorer.evaluator.build_scenario_candidate_config(config, ctx)
                       for ctx in self.scorer.evaluator.contexts}
        selected = set(configs) if scenarios is None else set(scenarios)
        if not selected or selected - configs.keys():
            raise ValueError("screening must select nonempty known scenario labels")
        stage = "full" if selected == set(configs) else "screening"
        requests, slots, identity = [], [], []
        for index, ((scenario, exchange), variants) in enumerate(self._variants.items()):
            effective = configs[scenario]
            binding = variants.get(execution_key(effective))
            if binding is None:
                raise ValueError(f"candidate changes dataset-owned execution inputs for scenario {scenario!r}; "
                                 "prepare a compatible dataset before submitting it")
            parameters = prepare_candidate_parameters(
                effective, json.loads(binding.dataset.markets_json), binding.dataset.exchange,
            )
            identity.append((binding.dataset_id, parameters))
            if binding.scenario in selected:
                slot = ResultSlot(f"{candidate_id}:{index}", binding.dataset_id, binding.scenario,
                                  binding.dataset.exchange, binding.dataset.metrics)
                slots.append(slot)
                requests.append(BacktestRequest(slot.request_id, slot.dataset_id, MappingProxyType(parameters)))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        plan = CandidatePlan(candidate_id, tuple(values), key, stage, tuple(requests), tuple(slots))
        # Validate the selected stage and requested metric surface before device work.
        plan.collector(self.scorer)
        return plan


def execution_key(config):
    return json.dumps(_static_contract(config), sort_keys=True, allow_nan=False)
