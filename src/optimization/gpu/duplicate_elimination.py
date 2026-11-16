"""Bound distance scratch without changing pymoo's duplicate ordering."""

import numpy as np
from pymoo.core.duplicate import DefaultDuplicateElimination
from scipy.spatial.distance import cdist


class BoundedDuplicateElimination(DefaultDuplicateElimination):
    """Retain pymoo's Euclidean tolerance using blocks of distance rows."""

    _DISTANCE_SCRATCH_BYTES = 8 * 2**20

    def _do(self, pop, other, is_duplicate):
        rows = self.func(pop).astype(float, copy=False)
        references = rows if other is None else self.func(other).astype(float, copy=False)
        if not len(references):
            return is_duplicate
        max_distances = self._DISTANCE_SCRATCH_BYTES // np.dtype(float).itemsize
        block_columns = min(len(references), max_distances)
        block_rows = max_distances // block_columns
        for start in range(0, len(rows), block_rows):
            stop = min(start + block_rows, len(rows))
            reference_stop = stop if other is None else len(references)
            for left in range(0, reference_stop, block_columns):
                right = min(left + block_columns, reference_stop)
                distances = cdist(rows[start:stop], references[left:right])
                # Cross-population comparisons and complete earlier blocks
                # can compare every column. NaN distances never match.
                if other is not None or right <= start:
                    is_duplicate[start:stop] |= np.any(distances <= self.epsilon, axis=1)
                else:
                    # Pymoo compares earlier rows even when they were already
                    # marked duplicate. Exclude the diagonal and later rows.
                    for index, position in enumerate(range(start, stop)):
                        earlier = max(0, position - left)
                        is_duplicate[position] |= np.any(distances[index, :earlier] <= self.epsilon)
                del distances
        return is_duplicate
