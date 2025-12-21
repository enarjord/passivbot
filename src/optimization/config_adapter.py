"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

from typing import List

from config_utils import get_template_config
from optimization.bounds import Bound


def extract_bounds_tuple_list_from_config(config) -> List[Bound]:
    """
    Extracts list of Bound instances for bot parameters.
    Also sets all bounds to (low, low, step) if pside is not enabled.

    Supported formats:
        - [low, high]: continuous optimization (step=None)
        - [low, high, step]: discrete optimization with given step
        - [low, high, 0] or [low, high, null]: treated as continuous
        - single value: fixed parameter (low=high, step=None)
    """
    template_config = get_template_config()
    bounds = []
    for pside in sorted(template_config["bot"]):
        is_enabled = all(
            [
                Bound.from_config(k, config["optimize"]["bounds"][k]).high > 0.0
                for k in [f"{pside}_n_positions", f"{pside}_total_wallet_exposure_limit"]
            ]
        )
        for key in sorted(template_config["bot"][pside]):
            bound_key = f"{pside}_{key}"
            assert (
                bound_key in config["optimize"]["bounds"]
            ), f"bound {bound_key} missing from optimize.bounds"
            bound_vals = Bound.from_config(bound_key, config["optimize"]["bounds"][bound_key])
            if is_enabled:
                bounds.append(bound_vals)
            else:
                # Disabled: fix to low value, preserve step for consistency
                bounds.append(Bound(bound_vals.low, bound_vals.low, bound_vals.step))
    return bounds
