from copy import deepcopy

from .bot import apply_forager_internal_aliases
from .optimize_bounds import prune_inactive_optimize_strategy_bounds
from .shared_bot import BOT_POSITION_SIDES, canonicalize_shared_bot_side
from .strategy import prune_inactive_strategy_subtrees, sync_canonical_strategy_config
from .transform_log import record_transform


def compile_runtime_config(config: dict, runtime: str = "generic", *, record_step: bool = True) -> dict:
    normalized_runtime = str(runtime).strip().lower()
    result = deepcopy(config)
    for pside in BOT_POSITION_SIDES:
        canonicalize_shared_bot_side(
            result.get("bot", {}).get(pside),
            path_prefix=("bot", pside),
            seed_missing_groups=False,
        )
    for override in (result.get("coin_overrides") or {}).values():
        if not isinstance(override, dict):
            continue
        override_bot = override.get("bot", {})
        if not isinstance(override_bot, dict):
            continue
        for pside in BOT_POSITION_SIDES:
            canonicalize_shared_bot_side(
                override_bot.get(pside),
                path_prefix=("coin_overrides", pside),
                seed_missing_groups=False,
            )
    sync_canonical_strategy_config(result)
    prune_inactive_strategy_subtrees(result)
    prune_inactive_optimize_strategy_bounds(result)
    apply_forager_internal_aliases(result)
    if record_step:
        record_transform(result, "compile_runtime_config", {"runtime": normalized_runtime})
    return result
