"""CPU-only transport of canonical configuration values into GPU request parameters."""

from types import SimpleNamespace
import numpy as np

from config.shared_bot import flatten_shared_bot_side
from optimization.gpu.hsl import project_bot
from optimization.gpu.model import (
    adaptive_params, EMA_ANCHOR_MULTICOIN_PARAM_KEYS, GLOBAL_RISK_PARAM_KEYS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, UNSTUCK_EMA_PARAM_KEYS,
    flatten_trailing_martingale_params,
)


def _single_coin_exposure_params(
    risk: dict, *, side: str, scale_hsl_budget: bool = False
) -> dict[str, float]:
    return {
        "we_excess_allowance_fraction": float(
            risk.get("we_excess_allowance_fraction", 0.0) or 0.0
        ),
        "hsl_scale_budget_with_excess_allowance": float(scale_hsl_budget),
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold_ratio": float(
            risk.get('total_exposure_enforcer', {}).get('threshold_ratio', 1.0) or 0.0
        ),
    }


def _position_exposure_enforcer_params(risk: dict, *, side: str) -> dict[str, float]:
    enabled = bool(
        risk.get('position_exposure_enforcer', {}).get('enabled', risk.get('risk_wel_enforcer_enabled', False))
    )
    threshold = float(
        risk.get('position_exposure_enforcer', {}).get('threshold_ratio', risk.get('risk_wel_enforcer_threshold_ratio', 0.0))
        or 0.0
    )
    if enabled and (not np.isfinite(threshold) or threshold <= 0.0):
        raise ValueError(
            "MPS proxy requires a finite positive "
            f"bot.{side}.risk.position_exposure_enforcer.threshold_ratio when the "
            "position exposure enforcer is enabled"
        )
    return {
        "wel_enforcer_enabled": float(enabled),
        "wel_enforcer_threshold_ratio": threshold,
    }


def _total_exposure_enforcer_params(risk: dict, *, side: str) -> dict[str, float]:
    policy = (
        str(risk.get('total_exposure_enforcer', {}).get('policy', 'reduce_overweight'))
        .strip()
        .lower()
    )
    if policy not in {"reduce_overweight", "reduce_portfolio"}:
        raise ValueError(
            "MPS proxy requires "
            f"bot.{side}.risk.total_exposure_enforcer.policy to be "
            f"reduce_overweight or reduce_portfolio, got {policy!r}"
        )
    return {
        "twel_enforcer_enabled": float(
            bool(risk.get('total_exposure_enforcer', {}).get('enabled', False))
        ),
        "twel_enforcer_reduce_portfolio": float(policy == "reduce_portfolio"),
    }


def _unstuck_params(bot: dict) -> dict[str, float]:
    return {
        "unstuck_enabled": float(bool(bot["unstuck_enabled"])),
        "unstuck_ema_gating_enabled": float(bool(bot["unstuck_ema_gating_enabled"])),
        "unstuck_close_fraction": float(bot["unstuck_close_fraction"]),
        "unstuck_ema_distance_fraction": float(bot["unstuck_ema_distance_fraction"]),
        "unstuck_loss_allowance_fraction": float(bot["unstuck_loss_allowance_fraction"]),
        "unstuck_exposure_trigger_ratio": float(bot["unstuck_exposure_trigger_ratio"]),
        **{key: float(bot[key]) for key in UNSTUCK_EMA_PARAM_KEYS},
    }


def _hsl_params(bot: dict, *, signal_mode: str) -> dict[str, float]:
    from optimization.gpu.hsl import pack_params

    return pack_params(bot, signal_mode)


def multicoin_parameters_from_payload(payload, config, *, sides=("long", "short")):
    """Encode resolved canonical payload values, without simulation or device work.

    Static coin patches belong to the registered dataset's override arrays. Callers
    preparing global candidate values must supply a payload without those patches.
    """
    kind = config['bot']["portfolio"]['strategy_kind']
    keys = {"ema_anchor": EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
            "trailing_martingale": TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS}[kind]
    from config.hsl import native_hsl_mode

    mode = native_hsl_mode(config["bot"]["portfolio"]["hsl"]["scope"])
    result = {}
    for side in sides:
        bot = project_bot(payload, 0, side, config)
        strategy = dict(payload.strategy_params_list[0][side])
        if kind == "trailing_martingale":
            strategy = flatten_trailing_martingale_params(strategy, bot)
        weights = bot.get("forager_score_weights", {}) or {}
        strategy.update({
            "entry_cooldown_minutes": float(bot.get("risk_entry_cooldown_minutes", 0.0) or 0.0),
            "total_wallet_exposure_limit_ratio": float(bot["total_wallet_exposure_limit_ratio"]),
            "forager_volume_ema_span_minutes": float(bot.get("forager_volume_ema_span_minutes", 0.0) or 0.0),
            "forager_log_range_ema_span_minutes": float(bot.get("forager_log_range_ema_span_minutes", 0.0) or 0.0),
            "forager_score_weights_volume": float(weights.get("volume", 0.0)),
            "forager_score_weights_ema_readiness": float(weights.get("ema_readiness", 0.0)),
            "forager_score_weights_log_range": float(weights.get("log_range", 0.0)),
            "n_positions": float(bot["n_positions"]),
        })
        risk = config["bot"][side]["risk"]
        strategy.update(_single_coin_exposure_params(
            risk, side=side,
            scale_hsl_budget=config["bot"][side]["hsl"]["scale_budget_with_excess_allowance"],
        ))
        if kind == "trailing_martingale":
            strategy.update(_position_exposure_enforcer_params(risk, side=side))
        strategy.update(_total_exposure_enforcer_params(risk, side=side))
        flat_bot = flatten_shared_bot_side(config["bot"][side])
        strategy.update(_unstuck_params(flat_bot))
        strategy.update(_hsl_params(project_bot(payload, 0, side, config, base=True), signal_mode=mode))
        strategy.update(adaptive_params(flat_bot))
        strategy.update({key: float(payload.backtest_params[key]) for key in GLOBAL_RISK_PARAM_KEYS})
        missing = set(keys) - strategy.keys()
        if missing:
            raise ValueError(f"GPU {kind} {side} payload is missing parameters: {sorted(missing)}")
        result[side] = {key: float(strategy[key]) for key in keys}
    return result


def prepare_candidate_parameters(config, markets, exchange):
    """Prepare compact effective values on the CPU, never run a CPU backtest.

    The optimizer has already materialized fixed/runtime/scenario overrides. Coin
    patches stay dataset-owned and must not leak the first coin's policy into the
    unpatched global values used by every other coin.
    """
    from backtest import prep_backtest_args
    from config.gpu import validate_gpu_backtest_config
    from config.runtime_compile import compile_runtime_config
    validate_gpu_backtest_config(config)
    effective = compile_runtime_config(config, runtime="backtest", record_step=False)
    effective["coin_overrides"] = {}
    bots, strategies, _markets, backtest = prep_backtest_args(
        effective, markets, exchange, is_runtime_compiled=True, metrics_only=True,
    )
    payload = SimpleNamespace(bot_params_list=bots, strategy_params_list=strategies,
                              backtest_params=backtest)
    params = multicoin_parameters_from_payload(payload, effective)
    result = {f"{side}_{key}": value for side, values in params.items()
              for key, value in values.items() if key not in GLOBAL_RISK_PARAM_KEYS}
    result.update({key: float(backtest[key]) for key in GLOBAL_RISK_PARAM_KEYS})
    # Reporting observes configured policies even when trading has zero slots.
    # Keep this transport separate from the deliberately capacity-gated shader bit.
    hsl = backtest["equity_hard_stop_loss"]
    for index, side in enumerate(("long", "short")):
        if hsl["mode"] == "coin":
            policy = config["bot"][side]["hsl"]
        else:
            policy = hsl["portfolio"] if hsl["mode"] == "portfolio" else hsl["sides"][index]
        result[f"_core_hsl_policy_enabled_{side}"] = float(bool(policy["enabled"]))
    return result
