"""Shared field descriptions grounded in the existing validators and policy maps.

Domains are accepted numeric domains, never recommended optimizer ranges. Rules
depending on another field, exchange facts or native f32 arithmetic remain with
the owning validator. Unknown metadata stays unset.
"""

from dataclasses import replace

from .field_spec import CoinOverridePolicy, FieldDomain, FieldType


def enrich_fields(fields):
    from .field_registry import get_shared_optimizer_bindings
    from .field_policies import CONDITIONAL_HSL_OVERRIDE_PATHS, get_live_coin_override_paths
    from .field_registry import get_coin_override_paths
    from .optimize_bounds import flatten_optimize_bounds, get_optimize_bounds_defaults
    from .strategy_spec import get_strategy_param_keys

    bindings = {path: key for key, path in get_shared_optimizer_bindings().items()}
    recommendations = flatten_optimize_bounds(
        get_optimize_bounds_defaults(), strategy_kind="trailing_martingale"
    )
    allowed = get_coin_override_paths()
    result = []
    for field in fields:
        path = field.path
        data = {}
        if path[:1] == ("bot",):
            local = ".".join(path[2:])
            key = bindings.get(path, field.optimize_key)
            data.update(optimize_key=key, optimizable=key is not None,
                        coin_override=CoinOverridePolicy.FORBIDDEN)
            if path[2] != "strategy" and key in recommendations:
                data["recommended_bounds"] = tuple(recommendations[key])
            elif path[:3] == ("bot", "global", "hsl") and "long_" + str(key) in recommendations:
                data["recommended_bounds"] = tuple(recommendations["long_" + key])
            if path[1] in {"long", "short"}:
                if local in allowed:
                    data["coin_override"] = (CoinOverridePolicy.COIN_HSL_ONLY
                                             if local in CONDITIONAL_HSL_OVERRIDE_PATHS
                                             else CoinOverridePolicy.ALLOWED)
                if path[2] == "strategy" and ".".join(path[4:]) in get_strategy_param_keys(path[3]):
                    data["coin_override"] = CoinOverridePolicy.ALLOWED
            if "hsl" in path and key:
                data["optimize_when"] = ("portfolio HSL scope" if path[1] == "global"
                                         else "coin or position_side HSL scope")
            elif path[2] == "strategy" and key:
                data["optimize_when"] = f"{path[3]} is the active strategy"
            data.update(_bot_metadata(path))
        elif path[:1] == ("coin_overrides",):
            data["optimizable"] = False
            if path[-1] == "wallet_exposure_limit":
                data.update(unit="ratio", denominator="raw balance")
        elif path[:1] == ("backtest",):
            data["consumers"] = ("backtest", "optimize")
            data.update(_operational_fraction_metadata(path))
        elif path[:1] == ("optimize",):
            data["consumers"] = ("optimize",)
        elif path[:1] == ("live",):
            # Reuse the command/runtime owner's declarations without copying
            # their shared-field classification into a second static list.
            from config_utils import get_field_runtime_rule

            rule = get_field_runtime_rule(".".join(path))
            if rule.get("consumed_by"):
                data["consumers"] = tuple(command for command in ("live", "backtest", "optimize")
                                          if command in rule["consumed_by"])
            data["coin_override"] = (CoinOverridePolicy.ALLOWED if path in get_live_coin_override_paths()
                                     else CoinOverridePolicy.FORBIDDEN)
            if path in {("live", "modes", side) for side in ("long", "short")}:
                from .control_sentinels import FORCED_MODES

                data.update(domain=FieldDomain(choices=FORCED_MODES),
                            description="Null leaves the mode unforced; per-coin null clears a root forced mode.")
            elif path == ("live", "startup", "warmup_concurrency"):
                data.update(domain=FieldDomain(minimum=1, choices=("auto",)),
                            description="Auto uses the existing context/exchange concurrency policy; a positive integer is explicit.")
            data.update(_operational_fraction_metadata(path))
        if path == ("optimize", "scoring", "*", "goal"):
            from .scoring import OBJECTIVE_GOALS

            data["domain"] = FieldDomain(choices=tuple(sorted(OBJECTIVE_GOALS)))
        if path in {("optimize", "scoring", "*", "reducer"),
                    ("optimize", "limits", "*", "reducer"), ("backtest", "reducer", "*")}:
            from .reducers import SUPPORTED_REDUCERS

            data["domain"] = FieldDomain(choices=tuple(sorted(SUPPORTED_REDUCERS)))
        result.append(replace(field, **data))
    return result


def _operational_fraction_metadata(path):
    denominators = {
        ("backtest", "market_order_slippage_fraction"): "candle close price",
        ("backtest", "limit_order_fill_buffer_fraction"): "limit order price",
        ("live", "account", "balance_hysteresis_snap_fraction"): "absolute previous hysteresis balance",
        ("live", "fills", "fallback_fee_fraction"): "absolute fill quote notional",
        ("live", "fills", "max_abs_fee_fraction"): "absolute fill quote notional",
        ("live", "execution", "limit_order_create_max_market_distance_fraction"): "market price",
        ("live", "execution", "order_match_tolerance_fraction"): "proposed order price or quantity",
        ("live", "execution", "order_replacement_churn_gate_market_distance_fraction"): "market price",
    }
    if path in denominators:
        return dict(unit="fraction", denominator=denominators[path])
    return {}


def _bot_metadata(path):
    local = ".".join(path[2:])
    name = path[-1]
    data = {}
    if local.startswith("strategy."):
        if path[3] == "trailing_martingale":
            denominators = {
                ("entry", "initial_qty_fraction"): "raw balance times effective per-position wallet exposure limit, converted at entry price",
                ("entry", "initial_ema_distance_fraction"): "entry EMA band reference price",
                ("entry", "threshold_base_fraction"): "position entry price",
                ("entry", "retracement_base_fraction"): "trailing price extremum",
                ("close", "qty_fraction"): "budget-derived full-position quantity at position entry price",
                ("close", "threshold_base_fraction"): "position entry price",
                ("close", "retracement_base_fraction"): "trailing price extremum",
            }
            if tuple(path[4:]) in denominators:
                data.update(unit="fraction", denominator=denominators[tuple(path[4:])])
                if tuple(path[4:]) == ("close", "qty_fraction"):
                    data["description"] = "Close slice uses the budget-derived full position plus any over-budget remainder, rounded and capped by the actual position; static nontrailing full-close mode retains its existing behavior."
        elif path[3] == "ema_anchor" and name in {"base_qty_fraction", "offset_fraction"}:
            data.update(unit="fraction", denominator=(
                "raw balance times effective per-position wallet exposure limit, converted at order price"
                if name == "base_qty_fraction" else "EMA anchor reference price"))
        return data  # Native metadata still owns defaults, bounds and choices.
    if local == "strategy_kind":
        from .strategy_spec import get_supported_strategy_kinds

        data["domain"] = FieldDomain(choices=get_supported_strategy_kinds())
    elif local == "universe.minimum_coin_age_days":
        data.update(unit="days", domain=FieldDomain(minimum=0.0, finite=True))
    elif local == "forager.score_hysteresis_margin":
        data.update(unit="normalized score gap", domain=FieldDomain(minimum=0.0, finite=True),
                    description="Absolute challenger-minus-incumbent ranking score margin required for replacement.")
    elif local in {"warmup.span_multiplier", "execution.market_order_near_touch_fraction"}:
        denominators = {"warmup.span_multiplier": "indicator span",
                        "execution.market_order_near_touch_fraction": "market price"}
        data.update(unit="multiplier" if local == "warmup.span_multiplier" else "fraction", denominator=denominators[local],
                    domain=FieldDomain(minimum=0.0, finite=True))
    elif local == "warmup.max_minutes":
        data.update(unit="minutes", domain=FieldDomain(minimum=0.0, exclusive_minimum=True, finite=True),
                    description="Null leaves history warmup uncapped; positive finite values cap its existing calculation.")
    elif local == "forager.always_include":
        data["description"] = "Persistent side selection within the eligible universe; operational modes and risk gates retain priority."
    elif local in {"universe.approved_coins", "universe.ignored_coins"}:
        data["description"] = "Side-owned coin identifiers or a reloadable local coin-list source."
    elif name in {"memory_ema_span_minutes", "drawdown_ratio_ema_span_minutes"}:
        data.update(unit="minutes", domain=FieldDomain(minimum=1.0, finite=True),
                    description="Floating-point EMA span; native f32 decay representability is validated at load time.")
    elif local.startswith("entry_cooldown."):
        data.update(unit="minutes", domain=FieldDomain(minimum=0.0, finite=True))
        if name == "max_duration_minutes":
            data["description"] = "Null means no ceiling; enabled modifier weights require a finite ceiling."
        elif "weights_minutes" in local:
            data["description"] = "Additional cooldown minutes per unit modifier."
            data["denominator"] = ("position wallet exposure / effective wallet exposure limit"
                                   if name == "exposure_ratio" else "adverse directionality score")
    elif local.startswith("forager.score_weights."):
        data.update(unit="relative weight", domain=FieldDomain(minimum=0.0, finite=True),
                    denominator="sum of authored forager weights",
                    description="Nonnegative relative ranking weight, normalized by the loader; all-zero vectors use the documented fallback.")
    elif local.startswith("forager.") and name.endswith("ema_span_1m"):
        data["unit"] = "one-minute candles"
        if name == "unilateralness_ema_span_1m":
            data["domain"] = FieldDomain(minimum=1.0, maximum=100_000.0, finite=True)
        else:
            data["description"] = "Positive finite span required when this ranking component is active."
    elif local == "hsl.drawdown_ratio_threshold":
        data.update(unit="fraction", denominator="scope equity plus remembered drawdown",
                    domain=FieldDomain(minimum=0.0, maximum=1.0, exclusive_minimum=True, finite=True),
                    description="RED requires both current and smoothed drawdown ratios to exceed this threshold.")
    elif local == "hsl.cooldown_minutes_after_red":
        data.update(unit="minutes", domain=FieldDomain(minimum=0.0, finite=True))
    elif local == "hsl.panic_close_order_type":
        data["domain"] = FieldDomain(choices=("limit", "market"))
    elif local == "hsl.scope":
        data["domain"] = FieldDomain(choices=("coin", "position_side", "portfolio"))
    elif local == "loss_gate.max_drawdown_ratio":
        data.update(unit="fraction", denominator="raw balance plus loss-gate remembered drawdown",
                    domain=FieldDomain(minimum=0.0, finite=True),
                    description="Values at least 1 disable the gate; zero blocks lossy non-panic closes.")
    elif local == "risk.max_positions":
        data.update(unit="positions", domain=FieldDomain(minimum=0, finite=True))
    elif local == "risk.total_wallet_exposure_limit":
        data.update(unit="ratio", denominator="raw balance",
                    domain=FieldDomain(minimum=0.0, finite=True))
    elif local == "risk.we_excess_allowance_fraction":
        data.update(unit="fraction", denominator="effective per-position wallet exposure limit",
                    domain=FieldDomain(minimum=0.0, finite=True))
    elif local in {"risk.position_exposure_enforcer.threshold_ratio", "risk.total_exposure_enforcer.threshold_ratio"}:
        data.update(unit="ratio", domain=FieldDomain(finite=True),
                    denominator=("effective per-position wallet exposure limit" if "position_exposure_enforcer" in path
                                 else "side total wallet exposure limit"),
                    description="A positive threshold is required when its exposure enforcer is enabled.")
        if local == "risk.total_exposure_enforcer.threshold_ratio":
            data.update(domain=FieldDomain(minimum=0.0, exclusive_minimum=True, finite=True),
                        description="Positive threshold used by the unconditional total-exposure entry cap and optional total-exposure reducer.")
    elif local == "risk.total_exposure_enforcer.policy":
        data["domain"] = FieldDomain(choices=("reduce_overweight", "reduce_portfolio"))
    elif local == "unstuck.close_fraction":
        data.update(unit="fraction", denominator="raw balance times effective per-position wallet exposure limit, converted at close price",
                    domain=FieldDomain(minimum=0.0, maximum=1.0, finite=True))
    elif local == "unstuck.loss_allowance_fraction":
        data.update(unit="fraction", denominator="applicable total wallet exposure limit",
                    domain=FieldDomain(minimum=0.0, finite=True),
                    description="Multiplies applicable TWEL in the realized-loss allowance formula.")
    elif local == "unstuck.exposure_trigger_ratio":
        data.update(unit="ratio", denominator="effective per-position wallet exposure limit",
                    domain=FieldDomain(finite=True),
                    description="Eligibility requires position wallet exposure / effective WEL strictly above this value.")
    elif local == "unstuck.ema_distance_fraction":
        data.update(unit="fraction", denominator="unstuck EMA band reference",
                    domain=FieldDomain(minimum=-1.0, exclusive_minimum=True, finite=True) if path[1] == "long"
                    else FieldDomain(maximum=1.0, exclusive_maximum=True, finite=True))
    elif local in {"unstuck.ema_span_0", "unstuck.ema_span_1"}:
        data.update(unit="minutes", domain=FieldDomain(minimum=0.0, exclusive_minimum=True, finite=True))
    return data
