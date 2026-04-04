from .detect import build_base_config_from_flavor, detect_flavor
from .legacy_v7 import migrate_btc_collateral_settings, migrate_suite_to_scenarios
from .renames import apply_backward_compatibility_renames, rename_config_keys


def apply_migrations(result: dict, *, verbose: bool = True, tracker=None) -> None:
    apply_backward_compatibility_renames(result, verbose=verbose, tracker=tracker)
    migrate_btc_collateral_settings(result, verbose=verbose, tracker=tracker)
    migrate_suite_to_scenarios(result, verbose=verbose, tracker=tracker)
    rename_config_keys(result, verbose=verbose, tracker=tracker)


__all__ = [
    "apply_backward_compatibility_renames",
    "apply_migrations",
    "build_base_config_from_flavor",
    "detect_flavor",
    "migrate_btc_collateral_settings",
    "migrate_suite_to_scenarios",
    "rename_config_keys",
]
