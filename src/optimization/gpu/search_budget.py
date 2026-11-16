"""Nominal search slots for GPU max cohorts and their checkpoints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaMaxSearchBudget:
    """Freeze cohort sizing independently of mutable device admission limits.

    The backend owns allocation accounting and persists this plan with its
    search state. The iteration budget may be extended on resume. The class
    name preserves the serialized type used by the original CUDA search.
    """

    population_size: int
    target_width: int
    cohort_limit: int | None = None

    def __post_init__(self):
        if type(self.population_size) is not int or self.population_size < 1:
            raise ValueError("GPU max population size must be a positive integer")
        if type(self.target_width) is not int or self.target_width < 1:
            raise ValueError("GPU max cohort target width must be a positive integer")
        if self.cohort_limit is not None and (type(self.cohort_limit) is not int or self.cohort_limit < 1):
            raise ValueError("GPU max search cohort limit must be a positive integer")

    @property
    def cohort_width(self) -> int:
        target = self.target_width
        if self.cohort_limit is not None:
            target = min(target, self.cohort_limit)
        return max(self.population_size, target)

    @staticmethod
    def remaining(*, iters: int, allocated: int) -> int:
        if type(iters) is not int or iters < 1:
            raise ValueError("optimize.iters must be a positive integer in GPU max mode")
        if type(allocated) is not int or allocated < 0:
            raise ValueError("allocated GPU max search slots must be a nonnegative integer")
        return max(0, iters - allocated)

    def bootstrap_size(self, *, iters: int, authored_seed_count: int, allocated: int = 0) -> int:
        """Return additional generated slots, keeping all authored seeds free."""
        if type(authored_seed_count) is not int or authored_seed_count < 0:
            raise ValueError("authored GPU max seed count must be a nonnegative integer")
        return min(self.remaining(iters=iters, allocated=allocated),
                   max(0, self.cohort_width - authored_seed_count))

    def offspring_size(self, *, iters: int, allocated: int) -> int:
        """Reserve attempts before duplicate elimination and scenario screening."""
        return min(self.cohort_width, self.remaining(iters=iters, allocated=allocated))
