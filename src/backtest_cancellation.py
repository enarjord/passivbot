from __future__ import annotations

import asyncio
import logging


def backtest_cancel_requested() -> bool:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return False
    return task is not None and task.cancelling() > 0


def raise_if_backtest_cancel_requested(stage: str) -> None:
    if not backtest_cancel_requested():
        return
    logging.info("[backtest] interrupt requested; aborting %s", stage)
    raise asyncio.CancelledError(f"backtest interrupted during {stage}")
