import numpy as np

from optimization.bounds import Bound
from optimization.repair import BoundsRepair


def test_bounds_repair_quantizes_stepped_and_continuous_values():
    repair = BoundsRepair(
        bounds=[Bound(0.0, 1.0, 0.25), Bound(0.0, 10.0)],
        sig_digits=3,
    )

    repaired = repair._do(
        None,
        np.asarray([[0.62, 9.876], [-0.2, 12.3]], dtype=np.float64),
    )

    assert repaired.tolist() == [[0.5, 9.88], [0.0, 10.0]]
