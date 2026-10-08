"""Service-owned batch adaptation using successful production work only.

The existing evidence controller supplies smoothing, cold-shape rejection,
cooldowns and rollback. Dataset classes do not share timing evidence; no search
policy, simulations, device imports or checkpoint state live here.
"""

from optimization.gpu.autotune import BatchController


class ExecutionBatchTuner:
    def __init__(self, *, initial=64, headroom=lambda: True, enabled=True):
        self.initial = initial
        self.headroom = headroom
        self.enabled = enabled
        self.controllers = {}
        self._ceilings = {}
        self._demand = 0
        self._allow_trial = False

    def constrain(self, dataset_id, ceiling):
        """Apply the prepared replay's dispatch bound on its owning worker."""
        self._ceilings[dataset_id] = ceiling

    def width(self, dataset_id, ceiling):
        # Claim one initial request, prepare on its owner, then discover the
        # physical limit. Never hide multiple serial replays behind a cold batch.
        if dataset_id not in self._ceilings:
            return 1
        limit = min(ceiling, self._ceilings.get(dataset_id, ceiling))
        if not self.enabled:
            return min(self.initial, limit)
        controller = self.controllers.get(dataset_id)
        if controller is None or controller.ceiling != limit:
            controller = BatchController(
                limit, min(self.initial, limit),
                allow_partial_batches=True,
                can_grow=lambda: (
                    self._demand >= min(controller.ceiling, controller.width * 2)
                    and self.headroom()
                ),
                can_trial=lambda: self._allow_trial,
            )
            self.controllers[dataset_id] = controller
        return controller.width

    def observe(self, dataset_id, count, seconds, *, backlog, closing):
        if not self.enabled:
            return
        controller = self.controllers[dataset_id]
        self._demand = count + backlog
        self._allow_trial = not closing
        controller.observe(count, seconds)
