"""Machine-independent fixed simulation inputs used to validate saved fitness."""

from copy import deepcopy
import platform

from config.overrides import parse_overrides
from config.policy import effective_minimum_cost_filter
from config.param_paths import resolve_optimizer_key_path
from config_utils import clean_config
from optimization.config_adapter import _flatten_bounds_for_config
from optimization.fine_tune_anchors import get_anchor_plan
from optimization.evaluation_implementation import evaluation_implementation_identity
from optimization.warmup import _apply_config_overrides
from optimizer_overrides import optimizer_overrides, unstuck_ema_spans_coupled
from config.prefilter import PREFILTER_VERSION, prefilter_enabled, resolve_prefilter

CONTRACT_VERSION = 7
CONTRACT_KEY = "optimizer_evaluation_contract"
CONTRACT_CACHE_KEY = "_optimizer_evaluation_contract"



def _remove_path(config, path):
    node = config
    for part in path[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict) and path:
        node.pop(path[-1], None)


def has_unresolved_override_files(config: dict) -> bool:
    """Legacy file references cannot establish the policy used for saved fitness."""

    def contains_file(value):
        if isinstance(value, dict):
            return any(
                (str(key).split(".")[-1] == "override_config_path" and bool(item))
                or contains_file(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_file(item) for item in value)
        return False

    return contains_file(config.get("coin_overrides", {})) or contains_file(
        config.get("backtest", {}).get("scenarios", [])
    )


def build_evaluation_contract(config: dict) -> dict:
    """Retain fixed policy while excluding values owned by the candidate vector."""
    effective = clean_config(config)
    _apply_config_overrides(
        effective, effective.get("optimize", {}).get("fixed_runtime_overrides", {})
    )
    overrides = effective.get("optimize", {}).get("enable_overrides", [])
    effective = optimizer_overrides(overrides, effective, None)
    for side in ("long", "short"):
        effective = optimizer_overrides(overrides, effective, side)
    effective = clean_config(effective)
    base_config_path = config.get("live", {}).get("base_config_path")
    if base_config_path is not None:
        effective["live"]["base_config_path"] = base_config_path
    effective = parse_overrides(effective, verbose=False)

    # Resolve before removing the base values required to validate coin patches.
    coin_overrides = deepcopy(effective.get("coin_overrides", {}))
    coupled = unstuck_ema_spans_coupled(config)
    if coupled:
        # These leaves derive from the strategy policy already recorded below;
        # candidate-dependent materialized coin values are not fixed policy.
        for bot in [
            effective.get("bot", {}),
            *(patch.get("bot", {}) for patch in coin_overrides.values()),
        ]:
            for side in ("long", "short"):
                for key in ("ema_span_0_minutes", "ema_span_1_minutes"):
                    bot.get(side, {}).get("unstuck", {}).pop(key, None)
    bounds = _flatten_bounds_for_config(effective, effective["optimize"]["bounds"])
    for key in bounds:
        path = resolve_optimizer_key_path(effective, key)
        if path is not None:
            _remove_path(effective, path)

    plan = get_anchor_plan(config)
    anchor_contract = None
    if plan is not None:
        anchors = []
        for anchor in plan["anchors"]:
            values = sorted(
                (
                    {"path": list(item["path"]), "value": deepcopy(item["value"])}
                    for item in anchor.get("fixed_values", [])
                ),
                key=lambda item: item["path"],
            )
            for item in values:
                _remove_path(effective, item["path"])
            anchors.append(values)
        # Anchor order is meaningful to the anchor gene; filenames and labels are not.
        anchor_contract = {
            "key_paths": [list(path) for path in plan.get("key_paths", [])],
            "anchors": anchors,
        }
    contract = {
        "version": CONTRACT_VERSION,
        "age_admission": "side_initial_entry_v1",
        "risk_model": "v9_independent_loss_memories_currency_drawdown_ratio_ema_finite_cooldown",
        "coupled_unstuck_ema_spans": coupled,
        "implementation": deepcopy(evaluation_implementation_identity()),
        "prepared_data": deepcopy(config.get("_optimizer_prepared_dataset_identity")),
        "bot": effective.get("bot", {}),
        "backtest_policy": {"filter_by_min_effective_cost": effective_minimum_cost_filter(effective)},
        "coin_overrides": coin_overrides,
        "anchors": anchor_contract,
    }
    if effective.get("backtest", {}).get("organillo_csv") is not None:
        digest = effective["backtest"].get("organillo_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Organillo requires a validated SHA-256 before saving evaluation fitness")
        contract["organillo"] = {"sha256": digest, "contract": 1}
    policy = resolve_prefilter(config.get("optimize", {}).get("prefilter"))
    if prefilter_enabled(policy):
        contract["prefilter"] = {"version": PREFILTER_VERSION, "policy": policy}
    half_life = effective.get("backtest", {}).get("ew_half_life_days")
    if half_life is not None:
        from config.ew import EW_DEFINITION_VERSION
        contract["exponential_weighting"] = {
            "half_life_days": half_life,
            "definition_version": EW_DEFINITION_VERSION,
        }
    if config.get("optimize", {}).get("backend") == "gpu":
        # Authoritative GPU fitness must never inherit CPU/proxy-validation
        # compatibility. Source and dependency identities above include the
        # replay shader, CUDA lowering and host metric implementation.
        metal = platform.system() == "Darwin" and platform.machine() == "arm64"
        contract["execution"] = {
            "engine": "metal_native" if metal else "cuda_native",
            "version": 2,
            "precision": ("f32_replay_integer_tick_boundaries_f32_ew_f64_compact_host"
                          if metal else "f32_replay_integer_tick_boundaries_f64_host"),
        }
    return contract


def recorded_evaluation_contract(template: dict) -> dict:
    contract = template.get(CONTRACT_CACHE_KEY)
    return deepcopy(
        contract if isinstance(contract, dict) else build_evaluation_contract(template)
    )
