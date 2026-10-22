"""CPU-confirmed useful-discovery accounting for an explicitly frozen protocol.

This observes an existing search; it does not choose economic cutoffs or run one.
All times use one measured origin before preparation. A persisted/GPU-observed
candidate is not confirmed evidence. Held-out results must not feed search fitness.
"""

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
import hashlib
from itertools import groupby
import json
import math

from .account_metrics import METRIC_DEFINITION, SCORABLE_METRICS


COLD_PHASES = frozenset({"preparation", "startup", "compilation", "gpu_tuning", "search",
                         "cpu_confirmation", "heldout", "export"})
_ACTIVITY = frozenset({"fill_count", "close_fill_count", "active_days", "traded_notional",
                       "exposed_fraction"})
_OPERATIONAL = {
    "live": frozenset({"user", "base_dir", "base_config_path", "session_name"}),
    "backtest": frozenset({"base_dir", "hlcvs_data_dir", "ohlcv_source_dir", "organillo_csv",
                            "compress_cache", "offline", "hsl_detailed_report"}),
}


def _finite(value, path, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be finite") from error
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{path} must be finite and >= {minimum}")
    return value


def _sha(value, path):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _canonical(value, path=()):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical configuration keys must be strings")
            if key == "override_config_path":
                if item not in (None, ""):
                    raise ValueError("materialize override files before identifying an effective configuration")
                continue
            if key.startswith("_"):
                continue
            if len(path) == 1 and key in _OPERATIONAL.get(path[0], ()):
                continue
            result[key] = _canonical(item, (*path, key))
        return ["mapping", sorted(result.items())]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_canonical(item, path) for item in value]]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["boolean", value]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, (int, float)):
        _finite(value, path)
        # Exact decimal spelling removes 1 versus 1.0 and negative-zero noise;
        # it does not round distinct genes into one configuration.
        number = Decimal(str(value))
        sign, digits, exponent = number.as_tuple()
        digits = list(digits)
        while digits and digits[-1] == 0:
            digits.pop()
            exponent += 1
        spelling = ("-" if sign else "") + "".join(map(str, digits)) + f"e{exponent}"
        return ["number", "0" if number == 0 else spelling]
    raise ValueError(f"unsupported canonical configuration value at {path}")


def canonical_configuration_identity(effective_config, *, model):
    """Hash complete materialized economic policy, excluding I/O/fitness metadata.

    Root bot/live/coin_overrides are required, and must contain the effective
    values after optimizer/scenario application. Search settings and source file
    paths are not policy. Dataset/execution identities additionally bind every
    scenario's CPU confirmation through the protocol.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be explicit")
    if not isinstance(effective_config, Mapping) or {"bot", "live", "coin_overrides"} - effective_config.keys():
        raise ValueError("effective config requires bot, live and materialized coin_overrides")
    for name in ("bot", "live", "coin_overrides"):
        if not isinstance(effective_config[name], Mapping):
            raise ValueError(f"effective config {name} must be a mapping")
    policy = {name: effective_config[name] for name in ("bot", "live", "coin_overrides")}
    for name in ("backtest", "scenario_overrides"):
        if name in effective_config:
            policy[name] = effective_config[name]
    payload = {"identity_version": 1, "model": model, "policy": _canonical(policy)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def compound_candidate_identity(scenario_configuration_sha256, *, model):
    """Bind a candidate to every exact scenario policy without dropping dates.

    Each value is canonical_configuration_identity() of the full materialized
    configuration for that label. A worker must verify its actual selected
    configuration against that label before reporting this shared candidate ID.
    Dataset, runtime and full native-input identities remain separate checks.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be explicit")
    if not isinstance(scenario_configuration_sha256, Mapping) or not scenario_configuration_sha256:
        raise ValueError("compound candidate requires a nonempty scenario configuration mapping")
    for label, identity in scenario_configuration_sha256.items():
        if not isinstance(label, str) or not label:
            raise ValueError("scenario configuration labels must be nonempty strings")
        _sha(identity, f"scenario_configuration_sha256.{label}")
    payload = {"identity_version": 1, "identity_kind": "compound_scenario_policy",
               "model": model, "scenario_configuration_sha256": dict(scenario_configuration_sha256)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_protocol(protocol):
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    required = {"version", "scenarios", "heldout_scenarios", "runtime_identities", "axes",
                "constraints", "minimum_activity", "target_k", "time_budget_seconds",
                "timer_origin", "timing_phases", "target_measure"}
    if required - protocol.keys():
        raise ValueError(f"protocol missing {', '.join(sorted(required - protocol.keys()))}")
    if protocol["version"] != 1 or type(protocol["version"]) is not int:
        raise ValueError("unsupported discovery protocol version")
    for name in ("scenarios", "heldout_scenarios", "runtime_identities"):
        mapping = protocol[name]
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"{name} must be a nonempty identity mapping")
        for label, identity in mapping.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"{name} labels must be nonempty strings")
            _sha(identity, f"{name}.{label}")
    if set(protocol["scenarios"]) & set(protocol["heldout_scenarios"]):
        raise ValueError("search and held-out scenarios must be disjoint")
    if set(protocol["scenarios"].values()) & set(protocol["heldout_scenarios"].values()):
        raise ValueError("held-out inputs must be distinct from search inputs")
    if type(protocol["target_k"]) is not int or protocol["target_k"] < 1:
        raise ValueError("target_k must be an explicitly chosen positive integer")
    if protocol["target_measure"] not in ("retained_frontier", "ever_frontier_additions"):
        raise ValueError("target_measure must explicitly choose retained_frontier or ever_frontier_additions")
    if _finite(protocol["time_budget_seconds"], "time_budget_seconds", minimum=0) <= 0:
        raise ValueError("time budget must be positive")
    phases = protocol["timing_phases"]
    if (protocol["timer_origin"] != "before_preparation" or not isinstance(phases, (list, tuple))
            or any(not isinstance(phase, str) for phase in phases)
            or len(phases) != len(COLD_PHASES) or set(phases) != COLD_PHASES):
        raise ValueError("cold timing must include preparation/startup/compile/tuning/search/CPU confirmation/heldout/export")
    axes = protocol["axes"]
    if not isinstance(axes, list) or not axes:
        raise ValueError("axes must be an explicitly chosen nonempty list")
    seen = set()
    for axis in axes:
        if not isinstance(axis, Mapping) or set(axis) != {"metric", "goal", "reducer", "novelty_tolerance"}:
            raise ValueError("each axis requires metric, goal, reducer and novelty_tolerance")
        metric = axis["metric"]
        if not isinstance(metric, str) or metric not in SCORABLE_METRICS or metric in seen:
            raise ValueError("objective metrics must be unique ordinary-account metrics")
        seen.add(metric)
        if axis["goal"] not in ("min", "max") or axis["reducer"] not in ("min", "max", "mean"):
            raise ValueError("objective goal/reducer must be explicit")
        _finite(axis["novelty_tolerance"], f"axes.{metric}.novelty_tolerance", minimum=0)
    constraints = protocol["constraints"]
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("explicit economic constraints are required")
    for constraint in constraints:
        if not isinstance(constraint, Mapping) or set(constraint) != {"metric", "op", "value"}:
            raise ValueError("constraints require metric, op and value")
        if (not isinstance(constraint["metric"], str) or constraint["metric"] not in SCORABLE_METRICS
                or constraint["op"] not in ("<=", ">=")):
            raise ValueError("constraint metric/operator is invalid")
        _finite(constraint["value"], "constraint.value")
    activity = protocol["minimum_activity"]
    if not isinstance(activity, Mapping) or not activity or set(activity) - _ACTIVITY:
        raise ValueError("minimum_activity requires explicit supported account activity criteria")
    if not any(_finite(value, f"minimum_activity.{key}", minimum=0) > 0 for key, value in activity.items()):
        raise ValueError("at least one positive activity requirement is needed")
    # Validate every threshold even after any() encounters an earlier positive one.
    for key, value in activity.items():
        _finite(value, f"minimum_activity.{key}", minimum=0)
    # A protocol must also be persistable verbatim as immutable JSON evidence.
    try:
        json.dumps(protocol, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("protocol must be finite JSON data") from error
    return deepcopy(dict(protocol))


class DiscoveryCounter:
    """Recompute confirmation-time discoveries independently of search arrival order."""
    def __init__(self, protocol, *, starting_pool):
        self._protocol = validate_protocol(protocol)
        self.protocol_sha256 = hashlib.sha256(json.dumps(
            self._protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.records = {}
        self.starting = set()
        for record in starting_pool:
            identity = self.add(record)
            self.starting.add(identity)
        if not self.starting:
            raise ValueError("the shared starting reference pool must be explicit and nonempty")

    @property
    def protocol(self):
        """Return a copy: caller edits cannot change the frozen counting rules."""
        return deepcopy(self._protocol)

    def add(self, record):
        required = {"model", "observed_elapsed_seconds", "confirmations",
                    "export_completed_elapsed_seconds"}
        if not isinstance(record, Mapping) or required - record.keys():
            raise ValueError("candidate evidence is incomplete")
        model = record["model"]
        if not isinstance(model, str) or model not in self._protocol["runtime_identities"]:
            raise ValueError("candidate model has no frozen CPU runtime identity")
        expected = {**self._protocol["scenarios"], **self._protocol["heldout_scenarios"]}
        if ("effective_config" in record) == ("scenario_effective_configs" in record):
            raise ValueError("candidate requires exactly one effective_config or scenario_effective_configs")
        scenario_policies = None
        if "scenario_effective_configs" in record:
            configs = record["scenario_effective_configs"]
            if not isinstance(configs, Mapping) or set(configs) != set(expected):
                raise ValueError("scenario effective configs must cover exactly the frozen search and held-out labels")
            scenario_policies = {
                label: canonical_configuration_identity(config, model=model)
                for label, config in configs.items()
            }
            identity = compound_candidate_identity(scenario_policies, model=model)
        else:
            identity = canonical_configuration_identity(record["effective_config"], model=model)
        observed = _finite(record["observed_elapsed_seconds"], "observed time", minimum=0)
        exported = record["export_completed_elapsed_seconds"]
        if exported is not None:
            _finite(exported, "export time", minimum=observed)
        if not isinstance(record["confirmations"], Mapping):
            raise ValueError("confirmations must be an explicit scenario mapping")
        if set(record["confirmations"]) - expected.keys():
            raise ValueError("confirmation has an unregistered scenario")
        for label, confirmation in record["confirmations"].items():
            needed = {"candidate_sha256", "input_sha256", "runtime_sha256", "engine",
                      "completed_elapsed_seconds", "metrics"}
            if not isinstance(confirmation, Mapping) or needed - confirmation.keys():
                raise ValueError("scenario confirmation is missing its lineage, metrics or clock")
            if confirmation["candidate_sha256"] != identity:
                raise ValueError("CPU confirmation belongs to a different effective configuration")
            if (scenario_policies is not None and
                    confirmation.get("scenario_configuration_sha256") != scenario_policies[label]):
                raise ValueError("CPU confirmation scenario configuration differs from the compound candidate")
            if confirmation["input_sha256"] != expected[label]:
                raise ValueError("CPU confirmation input differs from the frozen scenario")
            if confirmation["engine"] not in ("cpu", "gpu"):
                raise ValueError("confirmation engine must explicitly be cpu or gpu")
            _sha(confirmation["runtime_sha256"], "confirmation.runtime_sha256")
            if (confirmation["engine"] == "cpu" and
                    confirmation["runtime_sha256"] != self._protocol["runtime_identities"][model]):
                raise ValueError("CPU confirmation runtime differs from the frozen model")
            _finite(confirmation["completed_elapsed_seconds"], "confirmation time", minimum=observed)
            if not isinstance(confirmation["metrics"], Mapping):
                raise ValueError("confirmation metrics must be explicit")
        incoming = deepcopy(dict(record))
        previous = self.records.get(identity)
        if previous is not None:
            for label, confirmation in previous["confirmations"].items():
                if label in incoming["confirmations"]:
                    new_confirmation = incoming["confirmations"][label]
                    # A GPU observation is never authoritative CPU evidence.
                    # Permit subsequent CPU confirmation, and never downgrade it.
                    if confirmation["engine"] != new_confirmation["engine"]:
                        if confirmation["engine"] == "cpu":
                            incoming["confirmations"][label] = confirmation
                        continue
                    old = {k: v for k, v in confirmation.items() if k != "completed_elapsed_seconds"}
                    new = {k: v for k, v in incoming["confirmations"][label].items() if k != "completed_elapsed_seconds"}
                    if old != new:
                        raise ValueError("conflicting confirmation facts for one canonical candidate")
                    incoming["confirmations"][label]["completed_elapsed_seconds"] = min(
                        confirmation["completed_elapsed_seconds"],
                        incoming["confirmations"][label]["completed_elapsed_seconds"])
                else:
                    incoming["confirmations"][label] = confirmation
            incoming["observed_elapsed_seconds"] = min(observed, previous["observed_elapsed_seconds"])
            exports = [value for value in (exported, previous["export_completed_elapsed_seconds"]) if value is not None]
            incoming["export_completed_elapsed_seconds"] = min(exports) if exports else None
        self.records[identity] = incoming
        return identity

    def _assess(self, record):
        protocol = self._protocol
        required = set(protocol["scenarios"]) | set(protocol["heldout_scenarios"])
        confirmations = record["confirmations"]
        if (required - confirmations.keys() or record["export_completed_elapsed_seconds"] is None
                or any(row["engine"] != "cpu" for row in confirmations.values())):
            return dict(status="pending")
        finished = max(record["export_completed_elapsed_seconds"],
                       *(row["completed_elapsed_seconds"] for row in confirmations.values()))
        reasons = []
        for label, confirmation in confirmations.items():
            metrics = confirmation["metrics"]
            if metrics.get("metric_definition") != METRIC_DEFINITION:
                raise ValueError(f"{label} requires version-neutral ordinary-account metrics")
            for key in ("completed", "liquidated", "nonpositive_equity_observed"):
                if type(metrics.get(key)) is not bool:
                    raise ValueError(f"{label}.{key} requires an authoritative boolean")
            if not metrics["completed"] or metrics["liquidated"] or metrics["nonpositive_equity_observed"]:
                reasons.append(f"{label}:incomplete_or_liquidated_or_insolvent")
                continue
            for constraint in protocol["constraints"]:
                key, target = constraint["metric"], constraint["value"]
                if key not in metrics:
                    raise ValueError(f"missing required account metric {label}.{key}")
                value = _finite(metrics[key], f"{label}.{key}")
                if not (value <= target if constraint["op"] == "<=" else value >= target):
                    reasons.append(f"{label}:constraint:{key}")
            for key, minimum in protocol["minimum_activity"].items():
                if key not in metrics:
                    raise ValueError(f"missing required account activity {label}.{key}")
                if _finite(metrics[key], f"{label}.{key}") < minimum:
                    reasons.append(f"{label}:activity:{key}")
        if reasons:
            return dict(status="infeasible", finished=finished, reasons=reasons)
        vector = []
        for axis in protocol["axes"]:
            values = []
            for label in protocol["scenarios"]:
                metric = axis["metric"]
                if metric not in confirmations[label]["metrics"]:
                    raise ValueError(f"missing required account objective {label}.{metric}")
                values.append(_finite(confirmations[label]["metrics"][metric], f"{label}.{metric}"))
            reducer = axis["reducer"]
            value = min(values) if reducer == "min" else max(values) if reducer == "max" else math.fsum(values) / len(values)
            vector.append(_finite(value if axis["goal"] == "min" else -value, "objective reducer"))
        return dict(status="feasible", finished=finished, vector=vector)

    def report(self):
        assessed = {identity: self._assess(row) for identity, row in self.records.items()}
        if any(assessed[identity]["status"] == "pending" for identity in self.starting):
            raise ValueError("starting reference pool is not fully CPU confirmed and exported")
        baseline_ready = max(assessed[identity]["finished"] for identity in self.starting)
        budget = self._protocol["time_budget_seconds"]
        tolerances = [axis["novelty_tolerance"] for axis in self._protocol["axes"]]
        frontier = {}
        def dominates(left, right):
            return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))
        def near(left, right):
            return all(abs(a - b) <= tolerance for a, b, tolerance in zip(left, right, tolerances))
        def insert(identity, vector):
            for old in [key for key, value in frontier.items() if dominates(vector, value)]:
                del frontier[old]
            frontier[identity] = vector
        starting_vectors = {identity: assessed[identity]["vector"] for identity in self.starting
                            if assessed[identity]["status"] == "feasible"}
        if baseline_ready <= budget:
            for identity, vector in sorted(starting_vectors.items()):
                if not any(dominates(old, vector) for old in frontier.values()):
                    insert(identity, vector)
        # Novelty is measured against every feasible starting policy and every
        # prior useful addition, even after a later policy dominates it. This
        # prevents near-copy cycles from turning into fresh discoveries.
        novelty_anchors = list(starting_vectors.values())
        feasible, useful, late = [], [], []
        pending = sorted(identity for identity, row in assessed.items() if row["status"] == "pending")
        ordered = sorted((max(row["finished"], baseline_ready), identity, row)
                         for identity, row in assessed.items()
                         if identity not in self.starting and row["status"] == "feasible")
        target_time = None
        for finished, batch in groupby(ordered, key=lambda item: item[0]):
            batch = list(batch)
            events = [dict(candidate_sha256=identity, confirmed_exported_seconds=finished)
                      for _, identity, _ in batch]
            if finished > budget:
                late.extend(events)
                continue
            feasible.extend(events)
            # Simultaneous evidence is one event: an inferior member of a batch
            # must not count merely because its hash sorts before its dominator.
            for _, identity, row in batch:
                vector = row["vector"]
                if any(dominates(other["vector"], vector) for _, _, other in batch):
                    continue
                if (any(dominates(old, vector) for old in frontier.values()) or
                        any(near(old, vector) for old in novelty_anchors)):
                    continue
                insert(identity, vector)
                novelty_anchors.append(vector)
                useful.append(dict(candidate_sha256=identity, confirmed_exported_seconds=finished))
            measure = (len(useful) if self._protocol["target_measure"] == "ever_frontier_additions"
                       else len(frontier.keys() - self.starting))
            if target_time is None and measure >= self._protocol["target_k"]:
                target_time = finished
        k = self._protocol["target_k"]
        reached = target_time is not None
        return dict(protocol_sha256=self.protocol_sha256,
                    unique_observed=len(self.records), starting_reference_count=len(self.starting),
                    baseline_ready_seconds=baseline_ready, pending=pending,
                    baseline_ready_within_budget=baseline_ready <= budget,
                    infeasible={identity: row["reasons"] for identity, row in sorted(assessed.items())
                                if row["status"] == "infeasible"},
                    broad_feasible_discoveries=feasible, useful_frontier_discoveries=useful,
                    current_frontier=frontier, retained_useful_count=len(frontier.keys() - self.starting),
                    late_confirmed_feasible=late, target_measure=self._protocol["target_measure"],
                    target_k=k, target_reached=reached, time_to_target_seconds=target_time,
                    right_censored=not reached, censor_seconds=budget if not reached else None,
                    required_timing_scope="cold end-to-end; confirmation/export and starting reference included")
