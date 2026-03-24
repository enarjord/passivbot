"""Custom pymoo Output for per-generation logging."""

import time
from typing import Sequence

import numpy as np
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class OptimizeOutput(Output):

    def __init__(self, scoring_keys: Sequence[str]):
        super().__init__()
        self.scoring_keys = list(scoring_keys)
        self.front = Column("front", width=8)
        self.obj_columns = []
        for key in self.scoring_keys:
            col = Column(key, width=max(len(key) + 2, 13))
            self.obj_columns.append(col)
        self.elapsed = Column("elapsed", width=10)
        self.columns += [self.front] + self.obj_columns + [self.elapsed]
        self._start_time = time.time()

    def update(self, algorithm):
        super().update(algorithm)
        opt = algorithm.opt
        front_size = len(opt) if opt is not None else 0
        self.front.set(front_size)

        if opt is not None:
            F = opt.get("F")
            if F is not None and len(F) > 0:
                best = F.min(axis=0)
                for i, col in enumerate(self.obj_columns):
                    # abs() because pymoo minimizes negated objectives
                    col.set(f"{abs(float(best[i])):.4f}")
            else:
                for col in self.obj_columns:
                    col.set("-")
        else:
            for col in self.obj_columns:
                col.set("-")

        self.elapsed.set(f"{time.time() - self._start_time:.1f}s")
