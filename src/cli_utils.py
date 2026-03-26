import os


def get_cli_prog(default: str) -> str:
    override = os.environ.get("PASSIVBOT_CLI_PROG")
    if not override:
        return default
    override = override.strip()
    return override or default
