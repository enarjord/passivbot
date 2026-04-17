class RestartBotException(Exception):
    """Raised to trigger a clean bot restart without incrementing error counts."""

    pass


class FatalBotException(Exception):
    """Raised to stop the bot cleanly without entering the auto-restart loop."""

    pass
