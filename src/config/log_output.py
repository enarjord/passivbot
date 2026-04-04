import logging


def log_config_message(verbose: bool, level: int, message: str, *args) -> None:
    prefixed_message = "[config] " + message
    noisy_info_prefixes = (
        "Added missing ",
        "Removed unused key",
        "adding missing ",
        "renaming parameter ",
        "dropping obsolete parameter ",
        "Skipping template subtree ",
    )
    if level == logging.INFO and any(message.startswith(prefix) for prefix in noisy_info_prefixes):
        logging.debug(prefixed_message, *args)
    elif verbose or level >= logging.WARNING:
        logging.log(level, prefixed_message, *args)
    else:
        logging.debug(prefixed_message, *args)
