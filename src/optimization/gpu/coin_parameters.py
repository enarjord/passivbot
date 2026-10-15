"""CPU encoding of resolved coin policies; no device or simulation dependency.

The canonical backtest payload owns effective values and the override resolver
owns precedence. This module only selects explicitly pinned values and encodes
the replay ABI; NaN means inherit the candidate's global parameter.
"""

from copy import deepcopy

import numpy as np

from optimization.gpu import model
from optimization.gpu.hsl import pack_params, project_bot
from optimizer_overrides import unstuck_ema_spans_coupled


_MISSING = object()


def _lookup(patch, path):
    for part in path:
        if not isinstance(patch, dict) or part not in patch:
            return _MISSING
        patch = patch[part]
    return patch


def build_coin_override_parameters(
    *, config, mss, exchange, coins, payload, side, strategy_kind, resolve_override=None,
):
    """Encode ordered coin pins from an already resolved canonical payload.

    Preserve exact patches separately from float32 transport values for execution
    identity. Eligibility and forced-active sentinels are payload-owned even when
    no authored pin is present. No configuration or payload is mutated.
    """
    if strategy_kind == "ema_anchor":
        prefix = "EMA_ANCHOR"
        strategy_paths = tuple((key, (key,)) for key in model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS)
    elif strategy_kind == "trailing_martingale":
        prefix = "TRAILING_MARTINGALE"
        strategy_paths = model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
    else:
        raise ValueError(f"unsupported GPU coin parameter strategy: {strategy_kind!r}")

    layout_prefix = f"{prefix}_COIN_OVERRIDE_"
    columns = {name.removeprefix(layout_prefix): value for name, value in vars(model).items()
               if name.startswith(layout_prefix)}

    def column(name):
        return columns[name]

    if resolve_override is None:
        from backtest import _get_backtest_coin_override

        resolve_override = _get_backtest_coin_override

    matrix = np.full((len(coins), column("COLS")), np.nan, dtype=np.float32)
    exact_overrides = []
    coupled = unstuck_ema_spans_coupled(config)
    for row, coin in enumerate(coins):
        patch = resolve_override(config, mss, exchange, coin) or {}
        exact_overrides.append(deepcopy(patch))
        side_patch = patch.get("bot", {}).get(side, {})
        strategy_patch = side_patch.get("strategy", {}).get(strategy_kind, {}) or {}
        effective_bot = project_bot(payload, row, side, config)
        effective_strategy = payload.strategy_params_list[row][side]
        if strategy_kind == "trailing_martingale":
            effective_strategy = model.flatten_trailing_martingale_params(
                effective_strategy, payload.bot_params_list[row][side],
            )
        for index, (key, path) in enumerate(strategy_paths):
            if _lookup(strategy_patch, path) is not _MISSING:
                value = float(effective_strategy[key])
                if key in {"entry_retracement_base_pct", "close_retracement_base_pct"}:
                    value = model.encode_tm_retracement_base_pct(value)
                matrix[row, index] = value
        if (strategy_kind == "trailing_martingale"
                and _lookup(strategy_patch, ("entry", "ema_gate_mode")) is not _MISSING):
            matrix[row, column("GATE_INITIAL_COLUMN")] = float(effective_strategy["gate_initial"])
            matrix[row, column("GATE_REENTRY_COLUMN")] = float(effective_strategy["gate_reentry"])

        cooldown_patch = side_patch.get("entry_cooldown", {}) or {}
        encoded = model.adaptive_params(effective_bot)
        adaptive_paths = (
            ("min_duration_minutes",), ("max_duration_minutes",),
            ("weights_minutes", "exposure_ratio"), ("weights_minutes", "adverse_directionality"),
        )
        for offset, (key, path) in enumerate(zip(model.ADAPTIVE_OVERRIDE_KEYS, adaptive_paths, strict=True)):
            if _lookup(cooldown_patch, path) is not _MISSING:
                matrix[row, column("ADAPTIVE_START") + offset] = encoded[key]
        if "base_duration_minutes" in cooldown_patch:
            matrix[row, column("COOLDOWN_COLUMN")] = float(
                effective_bot.get("risk_entry_cooldown_minutes", 0.0) or 0.0,
            )
        if not bool(effective_bot.get("entry_eligible", True)) or "wallet_exposure_limit" in side_patch:
            matrix[row, column("WALLET_EXPOSURE_COLUMN")] = float(effective_bot["wallet_exposure_limit"])
        risk_patch = side_patch.get("risk", {}) or {}
        if "we_excess_allowance_pct" in risk_patch:
            matrix[row, column("ALLOWANCE_PCT_COLUMN")] = float(
                effective_bot.get("risk_we_excess_allowance_pct", 0.0) or 0.0,
            )
        if strategy_kind == "trailing_martingale":
            if "position_exposure_enforcer_enabled" in risk_patch:
                matrix[row, column("WEL_ENFORCER_ENABLED_COLUMN")] = float(
                    bool(effective_bot.get("risk_wel_enforcer_enabled", False)),
                )
            if "position_exposure_enforcer_threshold" in risk_patch:
                matrix[row, column("WEL_ENFORCER_THRESHOLD_COLUMN")] = float(
                    effective_bot.get("risk_wel_enforcer_threshold", 0.0) or 0.0,
                )

        unstuck_patch = side_patch.get("unstuck", {}) or {}
        unstuck_keys = ("enabled", "ema_gating_enabled", "close_pct", "ema_dist", "loss_allowance_pct", "threshold")
        for offset, key in enumerate(unstuck_keys):
            if key in unstuck_patch:
                matrix[row, column("UNSTUCK_START_COLUMN") + offset] = float(effective_bot[f"unstuck_{key}"])
        for offset, key in enumerate(model.UNSTUCK_EMA_PARAM_KEYS):
            strategy_key = key.removeprefix("unstuck_")
            if coupled:
                path = ("entry", strategy_key) if strategy_kind == "trailing_martingale" else (strategy_key,)
                # Only genuine strategy pins override a candidate's coupled spans.
                matrix[row, column("UNSTUCK_EMA_START_COLUMN") + offset] = matrix[
                    row, strategy_paths.index((strategy_key, path)),
                ]
            elif strategy_key in unstuck_patch:
                matrix[row, column("UNSTUCK_EMA_START_COLUMN") + offset] = float(effective_bot[key])

        hsl_patch = side_patch.get("hsl", {}) or {}
        if hsl_patch:
            packed = pack_params(effective_bot, signal_mode="coin")
            for offset, (key, path) in enumerate(model.HSL_COIN_OVERRIDE_PATHS):
                value = _lookup(hsl_patch, path)
                if value is _MISSING:
                    continue
                encoded = (model.encode_hsl_panic_order_type(
                    value, field_name="coin override hsl.panic_close_order_type",
                ) if key == "hsl_panic_market" else float(packed[key]))
                matrix[row, column("HSL_START_COLUMN") + offset] = encoded
        if bool(effective_bot.get("is_forced_active", False)):
            matrix[row, column("FORCED_ACTIVE_COLUMN")] = 1.0

    contract = {
        "exchange": exchange, "coins": coins, "side": side, "exact_overrides": exact_overrides,
        "values": [[None if not np.isfinite(value) else float(value) for value in row] for row in matrix],
    }
    return matrix, contract
