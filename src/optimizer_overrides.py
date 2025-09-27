from config_utils import require_config_value


def optimizer_overrides(overrides_list, config, pside):
    if not overrides_list:
        # No overrides to apply
        return config

    for override in overrides_list:
        if override == "lossless_close_trailing":

            # Logic for lossless close
            threshold = require_config_value(config, f"bot.{pside}.close_trailing_threshold_pct")
            retracement = require_config_value(config, f"bot.{pside}.close_trailing_retracement_pct")
            config["bot"][pside]["close_trailing_threshold_pct"] = max(threshold, retracement)

        elif override == "example":
            # Logic for override 'example'
            pass

        else:
            print(f"Unknown override: {override}")
            return config

    return config
