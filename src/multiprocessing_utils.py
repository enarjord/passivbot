import signal


def ignore_sigint_in_worker() -> None:
    """Ensure worker processes ignore SIGINT so the parent controls shutdown."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
