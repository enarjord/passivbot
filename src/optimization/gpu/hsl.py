"""Canonical HSL policy transport for offline GPU screening.

The shared backtest payload owns effective policies, including coin overrides.
"""


def project_bot(payload, coin_index, side, config, *, base=False):
    bot = payload.bot_params_list[coin_index][side]
    hsl = payload.backtest_params["equity_hard_stop_loss"]
    if hsl["engine"] != "hsl":
        raise ValueError("GPU HSL payload engine differs from selected config")
    index = ("long", "short").index(side)
    mode = hsl["mode"]
    if mode == "unified":
        policy = hsl["portfolio"]
    elif mode == "coin":
        coin = payload.backtest_params["coins"][coin_index]
        policy = config["bot"][side]["hsl"] if base else hsl["coins"][coin][index]
    else:
        policy = hsl["sides"][index]
    result = dict(bot)
    result.update({f"hsl_{key}": value for key, value in policy.items()})
    if mode == "coin":
        configured = int(bot["n_positions"])
        result["hsl_enabled"] = bool(policy["enabled"]) and configured > 0
        # One tradable coin has exactly one effective slot.
        result["hsl_slot_count"] = (
            1 if payload.backtest_params["dynamic_wel_by_tradability"] else configured
        )
    return result


def pack_params(bot, signal_mode):
    enabled = bool(bot["hsl_enabled"])
    policy = bot["hsl_restart_after_red_policy"]
    if enabled and policy not in ("always", "never"):
        raise ValueError("GPU HSL requires an explicit restart policy")
    return {
        "hsl_enabled": float(enabled),
        "hsl_red_threshold": float(bot["hsl_red_threshold"]),
        "hsl_ema_span_minutes": float(bot["hsl_ema_span_minutes"]),
        "hsl_cooldown_minutes_after_red": float(bot["hsl_cooldown_minutes_after_red"]),
        "hsl_restart_policy": 2.0 if policy == "never" else 0.0,
        "hsl_signal_mode": {"unified": 0.0, "pside": 1.0, "coin": 2.0}[signal_mode],
        "hsl_slot_count": (
            float(bot["hsl_slot_count"]) if signal_mode == "coin" else 1.0
        ),
    }
