class RestartBotException(Exception):
    """Raised to trigger a clean bot restart without incrementing error counts."""

    pass


class FatalBotException(Exception):
    """Raised to stop the bot cleanly without entering the auto-restart loop."""

    pass


class FillEventDataError(ValueError):
    """Fetched fill values cannot be decoded; unrelated parser bugs are not covered."""


class SimulationConfigError(ValueError):
    """A simulation has no runnable universe; ordinary configuration may still be valid."""

    def __init__(self, message, *, context=None):
        super().__init__(message)
        self.context = context


class GPUScreeningMigrationError(ValueError):
    """An enabled legacy GPU search policy requires explicit scenario migration."""
