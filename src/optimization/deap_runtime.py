import math

from deap import base


class ConstraintAwareFitness(base.Fitness):
    constraint_violation: float = 0.0

    def dominates(self, other, obj=slice(None)):
        self_violation = getattr(self, "constraint_violation", 0.0)
        other_violation = getattr(other, "constraint_violation", 0.0)
        if math.isclose(self_violation, other_violation, rel_tol=0.0, abs_tol=1e-12):
            return super().dominates(other, obj)
        return self_violation < other_violation
