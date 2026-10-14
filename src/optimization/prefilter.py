"""Pure config constraints and the common optimizer result for skipped replay.

Rust owns every strategy/risk measurement. Python combines declared rule violations
and adapts them to the same F/G and persistence surfaces as replay evaluations.
"""

from copy import deepcopy
import json
import math

from backtest_universe import (
    effective_backtest_approved_coins_by_side,
    normalize_backtest_coin,
)
from config.prefilter import PREFILTER_VERSION, prefilter_enabled, resolve_prefilter
from config.metrics import CURRENCY_METRICS
from config.scoring import default_objective_goal, from_engine_value
from config.shared_bot import require_grouped_bot_value
from config.strategy import build_runtime_strategy_side, merge_runtime_bot_side
from metrics_schema import KNOWN_ANALYSIS_METRICS
from optimization.evaluation_payload import build_evaluation_payload
from utils import heuristic_symbol_to_coin, looks_like_exact_market_identifier


def prefilter_config_for_exchange(config, markets, exchange):
    """Resolve venue aliases outside the pure scalar API using prepared metadata."""
    from backtest import _get_backtest_coin_override

    coins = config["backtest"]["coins"][exchange]
    return {
        **config,
        "backtest": {**config["backtest"], "coins": {exchange: coins}},
        "coin_overrides": {
            coin: _get_backtest_coin_override(config, markets, exchange, coin)
            for coin in coins
        },
    }


def _probe_inputs(config, params, checks):
    kind = config["live"]["strategy_kind"]
    if kind != "trailing_martingale":
        raise ValueError(
            "enabled config prefilters currently require trailing_martingale"
        )
    approved = effective_backtest_approved_coins_by_side(config)
    data = config["backtest"].get("coins", {})
    if not isinstance(data, dict):
        raise ValueError("prefilter requires resolved backtest coin membership")
    coins = {
        normalize_backtest_coin(coin) for values in data.values() for coin in values
    }
    overrides = config.get("coin_overrides", {})
    for side in ("long", "short"):
        if not any(side in check["sides"] for check in checks):
            continue
        selected = (
            {
                coin
                for coin in coins
                if coin in approved[side]
                or any(
                    not looks_like_exact_market_identifier(identifier)
                    and heuristic_symbol_to_coin(identifier)
                    == heuristic_symbol_to_coin(coin)
                    for identifier in approved[side]
                )
            }
            if data
            else set(approved[side])
        )
        for coin in sorted(selected):
            patch = overrides.get(coin, {})
            if (
                not isinstance(patch, dict)
                or "override_config_path" in patch
                or patch.get("live", {}).get("override_config_path")
            ):
                raise ValueError("prefilter requires resolved inline coin overrides")
            override_side = patch.get("bot", {}).get(side, {})
            base = config["bot"][side]
            strategy = build_runtime_strategy_side(
                base, strategy_kind=kind, pside=side, override_side=override_side
            )
            bot = merge_runtime_bot_side(
                base, pside=side, strategy_kind=kind, override_side=override_side
            )

            def required(key):
                return require_grouped_bot_value(bot, side, key, prefer_flat=True)

            slots = int(round(float(required("n_positions"))))
            total = float(required("total_wallet_exposure_limit"))
            if slots <= 0 or total <= 0:
                continue
            # prep_backtest_args derives global WEL from total/slots; only an
            # explicit per-coin WEL overrides that runtime allocation.
            base_limit = float(override_side.get("wallet_exposure_limit", -1))
            if base_limit == 0:
                continue
            if base_limit <= 0:
                base_limit = total / slots
            weights = required("entry_cooldown_weights_minutes")
            yield coin, side, {
                "side": side,
                "strategy": strategy,
                "wallet_exposure_limit": base_limit,
                "total_wallet_exposure_limit": total,
                "risk_we_excess_allowance_pct": float(
                    required("risk_we_excess_allowance_pct")
                ),
                "cooldown_base": float(required("risk_entry_cooldown_minutes")),
                "cooldown_min": float(required("entry_cooldown_min_duration_minutes")),
                "cooldown_max": required("entry_cooldown_max_duration_minutes"),
                "cooldown_exposure_weight": weights["exposure_ratio"],
                "cooldown_adverse_weight": weights["adverse_directionality"],
                **{
                    key: params[key]
                    for key in (
                        "exposure_ratios",
                        "volatility_ema_1m",
                        "volatility_ema_1h",
                        "adverse_directionality",
                        "grid_max_entries",
                    )
                },
                "calculate_grid": any(
                    check["metric"] == "grid_span" and side in check["sides"]
                    for check in checks
                ),
            }


def inspect_config_prefilter(config, prefilter_params):
    """Return constraint and explanatory contributions without mutating inputs."""
    params = resolve_prefilter(prefilter_params)
    checks = [check for check in params["checks"] if check["enabled"]]
    if not checks:
        return 0.0, []
    import passivbot_rust as pbr

    worst = [None] * len(checks)
    for coin, side, probe in _probe_inputs(config, params, checks):
        measurements = json.loads(
            pbr.config_prefilter_metrics_json(json.dumps(probe, allow_nan=False))
        )
        for index, check in enumerate(checks):
            if side not in check["sides"]:
                continue
            rows = (
                [{"grid_span": measurements["grid_span"]}]
                if check["metric"] == "grid_span"
                else measurements["observations"]
            )
            for row in rows:
                value = float(row[check["metric"]])
                if not math.isfinite(value):
                    raise ValueError("non-finite config prefilter measurement")
                violation = (
                    max(
                        0.0,
                        0.0 if check["min"] is None else check["min"] - value,
                        0.0 if check["max"] is None else value - check["max"],
                    )
                    / check["scale"]
                )
                incomplete = (
                    check["metric"] == "grid_span" and not measurements["grid_complete"]
                )
                if incomplete:
                    # A truncated or unreachable ladder cannot establish a valid
                    # initial-to-last-fill span, even if its partial span fits.
                    violation = max(1.0, violation)
                item = {
                    "metric": check["metric"],
                    "coin": coin,
                    "side": side,
                    "value": value,
                    "exposure_ratio": row.get("exposure_ratio"),
                    "violation": violation,
                    "grid_incomplete": incomplete,
                    "weight": check["weight"],
                }
                if worst[index] is None or violation > worst[index]["violation"]:
                    worst[index] = item
    contributions = [
        item for item in worst if item is not None and item["violation"] > 0
    ]
    constraint = params["penalty_weight"] * math.fsum(
        item["weight"] * item["violation"] for item in contributions
    )
    if not math.isfinite(constraint) or constraint < 0:
        raise ValueError("config prefilter constraint must be finite and nonnegative")
    # DEAP's existing comparator treats differences <= 1e-12 as ties.
    if contributions:
        constraint = max(constraint, 1e-9)
    return constraint, contributions


def config_prefilter(config, prefilter_params) -> float:
    """Pure, opt-in scalar API: zero passes; a positive constraint skips replay."""
    return inspect_config_prefilter(config, prefilter_params)[0]


def prefilter_candidate(configs, params):
    """Use the worst complete scenario constraint, independent of screen subset."""
    if not prefilter_enabled(params):
        return 0.0, []
    worst, findings = 0.0, []
    for label, config in configs.items():
        value, rows = inspect_config_prefilter(config, params)
        if value > worst:
            worst = value
            findings = [dict(row, scenario=label) for row in rows]
    return worst, findings


def build_prefilter_payload(base, vector, constraint, findings, *, suite=False):
    """Make penalty defaults directly; never rescore them as simulator output."""
    if not math.isfinite(constraint) or constraint <= 0:
        raise ValueError("prefilter rejection requires a finite positive constraint")
    specs = base.scoring_specs
    names = (
        KNOWN_ANALYSIS_METRICS
        | {
            f"{name}_{currency}"
            for name in CURRENCY_METRICS
            for currency in ("usd", "btc")
        }
        | {spec.metric for spec in specs}
        | {check["metric"] for check in base.limit_checks}
    )
    defaults = {
        name: (
            -constraint
            if (
                default_objective_goal(name) == "max"
                or name in {"win_rate", "win_rate_w"}
            )
            else constraint
        )
        for name in names
    }
    raw = {spec.metric: from_engine_value(spec, constraint) for spec in specs}
    defaults.update(raw)
    metrics = {
        "objectives": raw,
        "unpenalized_objectives": tuple(constraint for _ in specs),
        "stats": {
            name: {stat: value for stat in ("mean", "min", "max", "median", "std")}
            for name, value in defaults.items()
        },
        "constraint_violation": constraint,
        "evaluation_origin": "config_prefilter",
        "backtest_performed": False,
        "prefilter": {"version": PREFILTER_VERSION, "findings": deepcopy(findings)},
    }
    if suite:
        metrics["suite_metrics"] = {
            "evaluation_origin": "config_prefilter",
            "metrics": deepcopy(metrics["stats"]),
        }
    return build_evaluation_payload(
        tuple(constraint for _ in specs), constraint, metrics, vector
    )
