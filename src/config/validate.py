def validate_config(config: dict, *, raw_optimize=None, verbose: bool = True, tracker=None) -> None:
    import config_utils as legacy

    legacy.require_config_dict(config, "monitor")
    config["live"]["hsl_signal_mode"] = legacy.normalize_hsl_signal_mode(
        config["live"]["hsl_signal_mode"]
    )
    config["live"]["hsl_position_during_cooldown_policy"] = (
        legacy.normalize_hsl_cooldown_position_policy(
            config["live"]["hsl_position_during_cooldown_policy"]
        )
    )
    legacy._normalize_monitor_config(config)
    legacy._normalize_pymoo_config(config, raw_optimize=raw_optimize)
    legacy._validate_forager_config(config, verbose=verbose, tracker=tracker)
