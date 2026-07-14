from __future__ import annotations

import logging
import multiprocessing
import random
from typing import Any

import numpy as np

MAX_NUMPY_SEED = 2**32 - 1


def normalize_optional_seed(value: Any, *, path: str = "optimize.seed") -> int | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "auto", "random"}:
            return None
        value = int(value)
    elif value is None:
        return None
    else:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{path} must be null or an integer from 0 to {MAX_NUMPY_SEED}")
        value = int(value)
    if value < 0 or value > MAX_NUMPY_SEED:
        raise ValueError(f"{path} must be null or an integer from 0 to {MAX_NUMPY_SEED}")
    return value


def seed_rngs(seed: int | None, *, context: str) -> None:
    if seed is None:
        logging.info("%s RNG seed: random", context)
        return
    random.seed(seed)
    np.random.seed(seed)
    logging.info("%s RNG seed: %d", context, seed)


def seed_worker_rngs(base_seed: int | None, *, context: str) -> int:
    if base_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        identity = multiprocessing.current_process()._identity
        worker_index = int(identity[0]) if identity else 0
        seed_sequence = np.random.SeedSequence([int(base_seed), worker_index])
    worker_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    logging.debug("%s RNG worker seed: %d", context, worker_seed)
    return worker_seed
