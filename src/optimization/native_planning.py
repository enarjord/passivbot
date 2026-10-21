"""Canonical CPU candidate preparation for a registered backtest service.

No device, replay, selection or simulation work belongs here. Fixed optimizer
policies and exact-last scenario overrides are materialized by the existing CPU
helpers before the transport adapter encodes effective scalar values.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from types import MappingProxyType

from config.param_paths import iter_optimizer_key_paths
from config.strategy_spec import get_strategy_param_keys, get_strategy_spec, get_supported_strategy_kinds
from config_utils import clean_config
from optimization.bounds import enforce_bounds
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import BacktestRequest
from optimization.gpu.model import gpu_side_enabled
from optimization.gpu.parameters import prepare_candidate_parameters
from optimization.native_results import CandidateEvaluation, CanonicalResultScorer, ResultSlot
from optimizer_overrides import unstuck_ema_spans_coupled
from config.prefilter import prefilter_enabled
from optimization.prefilter import prefilter_candidate, build_prefilter_payload, prefilter_config_for_exchange


def _remove_path(config, path):
    node = config
    for part in path[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def _request_owned_paths(config):
    """Paths removed from execution identity, independent of searched bounds."""
    kind = config['bot']['global']['strategy_kind']
    dynamic_bound_keys = ["unstuck_memory_ema_span_minutes", "loss_gate_memory_ema_span_minutes"]
    for side in ("long", "short"):
        for key in get_strategy_param_keys(kind):
            yield ("bot", side, "strategy", kind, *key.split("."))
        dynamic_keys = (
            "n_positions", "total_wallet_exposure_limit", "risk_entry_cooldown_minutes",
            "risk_we_excess_allowance_pct", "risk_twel_enforcer_threshold", "risk_wel_enforcer_threshold",
            "forager_volume_ema_span_1m", "forager_volatility_ema_span_1m",
            "unilateralness_ema_span_1m", "entry_cooldown_min_duration_minutes", "entry_cooldown_max_duration_minutes",
            "entry_cooldown_weights_minutes_exposure_ratio", "entry_cooldown_weights_minutes_adverse_directionality",
            "unstuck_close_pct", "unstuck_ema_dist", "unstuck_loss_allowance_pct", "unstuck_threshold",
            "unstuck_ema_span_0", "unstuck_ema_span_1", "hsl_red_threshold", "hsl_memory_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes",
            "hsl_cooldown_minutes_after_red",
            *(f"forager_score_weights_{name}" for name in ("volume", "ema_readiness", "volatility", "unilateralness")),
        )
        dynamic_bound_keys.extend(f"{side}_{key}" for key in dynamic_keys)
    for _, path in iter_optimizer_key_paths(config, dynamic_bound_keys):
        if path is not None:
            yield path
    for key in ("drawdown_ratio_threshold", "memory_ema_span_minutes", "drawdown_ratio_ema_span_minutes", "cooldown_after_red_minutes"):
        yield ("bot", "global", "hsl", key)


def _static_contract(config):
    """Leave every execution value not represented by a dynamic request in place.

    Feature enablement, execution modes and coin overrides stay dataset-owned.
    Changing one requires another prepared dataset, never silently ignoring it.
    """
    from config.universe import anchored_universe_source_view, runtime_universe_snapshot
    cleaned = clean_config(anchored_universe_source_view(config))
    result = {key: cleaned[key] for key in ("bot", "live", "backtest", "coin_overrides") if key in cleaned}
    # Numeric exposure/position values are dynamic, but their side enablement
    # chooses the registered replay's directional/fused kernel topology.
    result["enabled_sides"] = tuple(side for side in ("long", "short") if gpu_side_enabled(config, side))
    available = sorted({coin for coins in config["backtest"]["coins"].values() for coin in coins})
    result["prepared_universe"] = runtime_universe_snapshot(config, available_coins=available)
    result["backtest"]["coins"] = config["backtest"]["coins"]
    # The standalone simulation consumes the materialized scenario, not the
    # optimizer's saved suite recipe or its candidate-dependent derived patches.
    for key in ("scenarios", "suite_enabled", "suite_reducers", "cache_dir", "base_dir"):
        result["backtest"].pop(key, None)
    kind = config['bot']['global']['strategy_kind']
    dynamic_bound_keys = ["unstuck_memory_ema_span_minutes", "loss_gate_memory_ema_span_minutes"]
    for side in ("long", "short"):
        for key in get_strategy_param_keys(kind):
            _remove_path(result, ("bot", side, "strategy", kind, *key.split(".")))
        dynamic_keys = (
            "n_positions", "total_wallet_exposure_limit", "risk_entry_cooldown_minutes",
            "risk_we_excess_allowance_pct", "risk_twel_enforcer_threshold", "risk_wel_enforcer_threshold",
            "forager_volume_ema_span_1m", "forager_volatility_ema_span_1m",
            "unilateralness_ema_span_1m", "entry_cooldown_min_duration_minutes", "entry_cooldown_max_duration_minutes",
            "entry_cooldown_weights_minutes_exposure_ratio", "entry_cooldown_weights_minutes_adverse_directionality",
            "unstuck_close_pct", "unstuck_ema_dist", "unstuck_loss_allowance_pct", "unstuck_threshold",
            "unstuck_ema_span_0", "unstuck_ema_span_1", "hsl_red_threshold", "hsl_memory_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes",
            "hsl_cooldown_minutes_after_red",
            *(f"forager_score_weights_{name}" for name in ("volume", "ema_readiness", "volatility", "unilateralness")),
        )
        dynamic_bound_keys.extend(f"{side}_{key}" for key in dynamic_keys)
    for _, path in iter_optimizer_key_paths(config, dynamic_bound_keys):
        if path is not None:
            _remove_path(result, path)
    for key in ("drawdown_ratio_threshold", "memory_ema_span_minutes", "drawdown_ratio_ema_span_minutes", "cooldown_after_red_minutes"):
        _remove_path(result, ("bot", "global", "hsl", key))
    coupled = unstuck_ema_spans_coupled(config)
    # CPU worker views lower coupled unstuck spans to ordinary request inheritance.
    # Strategy coin pins remain static; their derived unstuck copies must not
    # split otherwise identical execution views.
    result["coupled_unstuck_emas"] = coupled
    if coupled:
        for patch in result.get("coin_overrides", {}).values():
            for side in ("long", "short"):
                for name in ("ema_span_0_minutes", "ema_span_1_minutes"):
                    _remove_path(patch, ("bot", side, "unstuck", name))
                    _remove_path(patch, ("bot", side, f"unstuck_{name.removesuffix('_minutes')}"))
    return result


def _finite_projection_number(value):
    # Plain finite transport values only; bool and numeric subclasses must use
    # the original cleaner. Comparing huge integers avoids float overflow here.
    return ((type(value) is float and math.isfinite(value)) or
            (type(value) is int and -sys.float_info.max <= value <= sys.float_info.max))


def _compile_projection_guard(value, numeric_paths, path=(), ancestors=()):
    """Compile an exact input comparison, permitting selected numeric leaves."""
    if path in numeric_paths and value is not None:
        if not _finite_projection_number(value):
            return None
        if "score_weights" in path:
            # The cleaner normalizes a copy only to hydrate discarded bounds.
            # Finite nonnegative inputs, including all-zero weights, cannot fail
            # that normalization. Negative/malformed values retain full cleaning.
            if value < 0.0:
                return None
            return lambda current: _finite_projection_number(current) and current >= 0.0
        return _finite_projection_number
    kind = type(value)
    if kind in (dict, list, tuple):
        # Ignored metadata can contain cycles or arbitrarily deep containers.
        # It remains valid for the legacy cleaner; decline this optional guard.
        if len(path) > 32 or id(value) in ancestors:
            return None
        ancestors = (*ancestors, id(value))
    if kind is dict:
        if any(type(key) is not str for key in value):
            return None
        children = tuple((key, _compile_projection_guard(child, numeric_paths, (*path, key), ancestors))
                         for key, child in value.items())
        if any(guard is None for _key, guard in children):
            return None
        return lambda current: (type(current) is dict and len(current) == len(children)
                                and all(type(key) is str for key in current)
                                and all(key in current and guard(current[key]) for key, guard in children))
    if kind in (list, tuple):
        children = tuple(_compile_projection_guard(child, numeric_paths, (*path, index), ancestors)
                         for index, child in enumerate(value))
        if any(guard is None for guard in children):
            return None
        return lambda current: (type(current) is kind and len(current) == len(children)
                                and all(guard(child) for guard, child in zip(children, current)))
    if kind is float:
        if not math.isfinite(value):
            return None
        encoded = value.hex()  # Preserve signed zero and exact JSON float values.
        return lambda current: type(current) is float and current.hex() == encoded
    if kind in (str, int, bool, type(None)):
        return lambda current: type(current) is kind and current == value
    return None


class _ExecutionProjection:
    """One validated input shape per planner scenario/exchange, never persisted.

    Every other input value and key is compared, including optimizer policies and
    metadata which the cleaner may inspect. Changed/unsupported inputs fall back
    to execution_key. Finite request-owned numbers cannot change cleaned static
    content; weights additionally require nonnegative values, so their joint
    normalization succeeds and changes only discarded optimizer bounds.
    Template selection depends on unchanged shape/scope/bounds. Adaptive bound
    hydration changes only discarded optimizer content; its duration check still
    requires finite values. Retired-key/string checks see unchanged keys/strings.
    """

    def __init__(self, config, key):
        self.key = key
        self.cleaner = clean_config
        self.guard = None
        # Decline unsupported inputs before metadata helpers inspect containers
        # with potentially overridden methods. This check runs only at creation.
        if _compile_projection_guard(config, ()) is None:
            return
        # The request-owned gate selector is a string, so compare it exactly.
        numeric_paths = {path for path in _request_owned_paths(config) if path[-1] != "ema_gate_mode"}
        if unstuck_ema_spans_coupled(config):
            for coin in config.get("coin_overrides", {}):
                for side in ("long", "short"):
                    for name in ("ema_span_0_minutes", "ema_span_1_minutes"):
                        numeric_paths.add(("coin_overrides", coin, "bot", side, "unstuck", name))
                        numeric_paths.add(("coin_overrides", coin, "bot", side, f"unstuck_{name.removesuffix('_minutes')}"))
        self.guard = _compile_projection_guard(config, numeric_paths)
        self.specs = tuple(get_strategy_spec(kind) for kind in get_supported_strategy_kinds())
        self.sides = tuple(side for side in ("long", "short") if gpu_side_enabled(config, side))

    def matches(self, config):
        if self.cleaner is not clean_config or self.guard is None or not self.guard(config):
            return False
        specs = tuple(get_strategy_spec(kind) for kind in get_supported_strategy_kinds())
        return (len(self.specs) == len(specs) and all(a is b for a, b in zip(self.specs, specs)) and
                self.sides == tuple(side for side in ("long", "short") if gpu_side_enabled(config, side)))


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
    prefilter_payload: dict | None = None
    record_config: dict | None = None

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
        self._execution_projections = {}
        self._projection_eligible = {}
        for binding in self.bindings:
            pair = (binding.scenario, binding.dataset.exchange)
            variants = self._variants.setdefault(pair, {})
            config = binding.dataset.policy_config()
            key = execution_key(config)
            if key in variants:
                raise ValueError("scenario/exchange execution variants must be unique")
            variants[key] = binding
            # Preparation cadence affects request grouping. Limit reuse to
            # single-coin views; source arrays may contain more coins.
            # Every variant in a pair must qualify independently.
            self._projection_eligible[pair] = (
                self._projection_eligible.get(pair, True)
                and len(binding.dataset.coin_indices) == 1
            )
        self.scorer.validate_coverage(list(self._variants), "full")
        if len({binding.dataset_id for binding in self.bindings}) != len(self.bindings):
            raise ValueError("prepared dataset identities must be unique")

    def _execution_key(self, pair, config):
        if not self._projection_eligible.get(pair, False):
            return execution_key(config)
        projection = self._execution_projections.get(pair)
        if projection is not None and projection.matches(config):
            return projection.key
        key = execution_key(config)
        if pair not in self._execution_projections:
            # Only configs freshly owned by prepare reach this boundary. Public
            # execution_key callers always retain the full cleaner/validation.
            self._execution_projections[pair] = _ExecutionProjection(config, key)
        return key

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
        policy = self.base.config["optimize"].get("prefilter")
        probe_configs = {}
        for index, (pair, variants) in enumerate(self._variants.items()):
            scenario, exchange = pair
            effective = configs[scenario]
            if self._projection_eligible[pair]:
                key = self._execution_key(pair, effective)
            else:
                key = execution_key(effective)
            binding = variants.get(key)
            if binding is None:
                raise ValueError(f"candidate changes dataset-owned execution inputs for scenario {scenario!r}; "
                                 "prepare a compatible dataset before submitting it")
            parameters = prepare_candidate_parameters(
                effective, json.loads(binding.dataset.markets_json), binding.dataset.exchange,
            )
            if prefilter_enabled(policy):
                probe_configs.setdefault(scenario, []).append(prefilter_config_for_exchange(
                    effective, json.loads(binding.dataset.markets_json), exchange,
                ))
            identity.append((binding.dataset_id, parameters))
            if binding.scenario in selected:
                slot = ResultSlot(f"{candidate_id}:{index}", binding.dataset_id, binding.scenario,
                                  binding.dataset.exchange, binding.dataset.metrics)
                slots.append(slot)
                requests.append(BacktestRequest(slot.request_id, slot.dataset_id, MappingProxyType(parameters)))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        plan = CandidatePlan(candidate_id, tuple(values), key, stage, tuple(requests), tuple(slots),
                             record_config=config if stage == "full" else None)
        # Validate the selected stage and requested metric surface before device work.
        plan.collector(self.scorer)
        if prefilter_enabled(policy):
            constraint, findings = prefilter_candidate(probe_configs, policy)
            if constraint > 0:
                payload = build_prefilter_payload(self.base, values, constraint, findings,
                                                 suite=self.scorer.suite,
                                                 scenario_labels=tuple(configs) if self.scorer.suite else ())
                plan = CandidatePlan(candidate_id, tuple(values), key, stage,
                                     tuple(requests), tuple(slots), payload, config)
        return plan


def execution_key(config):
    return json.dumps(_static_contract(config), sort_keys=True, allow_nan=False)
