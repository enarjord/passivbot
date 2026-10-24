"""Runnable-universe checks owned by simulation entry points, never live loading."""

from backtest_universe import backtest_side_enabled
from config.activation import side_enabled
from config.universe import require_side_universe_sources
from passivbot_exceptions import SimulationConfigError


POSITION_SIDES = ("long", "short")


def validate_simulation_config(config, *, resolved=False, context=None):
    """Reject known empty simulations without acquiring or rereading coin sources.

    Before resolution, file paths and 'all' are potentially nonempty sources,
    not fabricated memberships. Afterwards inspect the acquired execution view.
    """
    enabled = [side for side in POSITION_SIDES if side_enabled(config, side)]
    active = [side for side in enabled if backtest_side_enabled(config, side)]
    reasons = []
    if not enabled:
        reasons.append("Both position sides are disabled. Enable bot.long.enabled and/or bot.short.enabled.")
    elif not active:
        reasons.append(
            "Enabled position sides have no trading capacity. Set a positive "
            "risk.total_wallet_exposure_limit and risk.max_positions on at least one enabled side."
        )

    approved = require_side_universe_sources(config, "approved_coins")
    if not any(approved[side] for side in (active or POSITION_SIDES)):
        if resolved:
            reasons.append(
                "No approved coins remain for an active position side after coin resolution and "
                "ignored-coin filtering. Check approved_coins, ignored_coins, and the selected exchanges."
            )
        else:
            reasons.append(
                "No coins are approved for an active position side. Set "
                "bot.global.universe.approved_coins, a side approval source, or --symbols."
            )
    if reasons:
        raise SimulationConfigError("\n".join(reasons), context=context)


def simulation_config_error_message(error, prog):
    """Keep expected user-input failures concise at executable boundaries."""
    detail = f" (scenario {error.context!r})" if error.context else ""
    return (
        f"Cannot start {prog.split()[-1]}{detail}: no active trading universe.\n\n"
        f"{error}\n\n"
        "Supply a runnable config, for example:\n"
        f"  {prog} configs/examples/btc_long.json\n\n"
        "Or configure a side and coins explicitly:\n"
        f"  {prog} --bot.long.enabled y -s BTC,ETH\n"
        "See configs/examples/ for more starting points."
    )
