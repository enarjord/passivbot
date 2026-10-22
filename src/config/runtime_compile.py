from copy import deepcopy

from .bot import apply_forager_internal_aliases
from .optimize_bounds import prune_inactive_optimize_strategy_bounds
from .shared_bot import BOT_POSITION_SIDES, canonicalize_shared_bot_side
from .strategy import prune_inactive_strategy_subtrees, sync_canonical_strategy_config
from .transform_log import record_transform


def compile_runtime_config(
    config: dict, runtime: str = "generic", *, record_step: bool = True
) -> dict:
    normalized_runtime = str(runtime).strip().lower()
    result = deepcopy(config)
    from .bot_inheritance import reconciled_bot_policy
    policy = reconciled_bot_policy(result)
    previous_aliases = result["bot"]._runtime_aliases if policy is not None else frozenset()
    from .migrations.v9 import require_v9_runtime_config

    if policy is not None:
        from .bot_inheritance import bot_policy_source_view
        require_v9_runtime_config(bot_policy_source_view(result))
    else:
        require_v9_runtime_config(result)
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
    from .hsl import normalize_hsl
    from .schema import get_expanded_config_prototype

    normalize_hsl(result, get_expanded_config_prototype(), verbose=False)
    if normalized_runtime == "live":
        from .hsl import require_live_runtime_support

        require_live_runtime_support(result)
    sync_canonical_strategy_config(result)
    prune_inactive_strategy_subtrees(result)
    prune_inactive_optimize_strategy_bounds(result)
    from .bot_inheritance import get_bot_policy, install_bot_policy
    policy = policy if policy is not None else get_bot_policy(result)
    original_keys = {side: set(result["bot"][side]) for side in BOT_POSITION_SIDES}
    apply_forager_internal_aliases(result)
    if policy is not None:
        aliases = previous_aliases | frozenset((side, key) for side in BOT_POSITION_SIDES
                            for key in set(result["bot"][side]) - original_keys[side])
        install_bot_policy(result, policy, effective_bot=result["bot"], runtime_aliases=aliases)
    if record_step:
        record_transform(
            result, "compile_runtime_config", {"runtime": normalized_runtime}
        )
    return result
