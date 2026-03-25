from __future__ import annotations

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
            self.obj_columns.append(Column(key, width=max(len(key) + 2, 13)))
        self.elapsed = Column("elapsed", width=10)
        self.columns += [self.front] + self.obj_columns + [self.elapsed]
        self._start_time = time.time()

    def update(self, algorithm):
        super().update(algorithm)
        opt = algorithm.opt
        front_size = len(opt) if opt is not None else 0
        self.front.set(front_size)
        if opt is not None:
            objectives = opt.get("F")
            if objectives is not None and len(objectives) > 0:
                best = np.min(objectives, axis=0)
                for idx, col in enumerate(self.obj_columns):
                    col.set(f"{abs(float(best[idx])):.4f}")
            else:
                for col in self.obj_columns:
                    col.set("-")
        else:
            for col in self.obj_columns:
                col.set("-")
        self.elapsed.set(f"{time.time() - self._start_time:.1f}s")
