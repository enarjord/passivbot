"""Observed budget facts, with causal source clocks distinct from capture clocks."""

from bisect import bisect_left, bisect_right
from copy import deepcopy
from operator import itemgetter


CURRENT_ORDER = 2**63 - 1
_FACT_ORDER = itemgetter("timestamp", "order")
_TIMESTAMP = itemgetter("timestamp")


class BudgetObservations:
    def __init__(self):
        self._scopes = {}
        self._policies = {}
        self._captured = {}

    def retain_scopes(self, scopes):
        """Release dormant owners' duplicate budgets without declaring flatness."""
        for mapping in (self._scopes, self._policies, self._captured):
            for key in tuple(mapping):
                if key not in scopes:
                    del mapping[key]

    def capture(self, scope, *, start, now, balance, slots, scale, exposure,
                balance_at=None):
        if balance_at is None:
            balance_at = now
        if type(balance_at) is not int or not 0 <= balance_at <= now:
            raise ValueError("invalid HSL balance source clock")
        if now < self._captured.get(scope, now):
            raise ValueError("HSL budget capture clock moved backwards")
        self._captured[scope] = now
        policy = dict(slots=slots, scale_budget_with_excess_allowance=scale,
                      exposure_budget=deepcopy(exposure))
        previous_policy = self._policies.get(scope)
        if previous_policy is None or previous_policy[0] != policy:
            self._policies[scope] = (deepcopy(policy), now)
        policy_at = self._policies[scope][1]
        # A balance receipt cannot backdate a changed configuration component.
        source_at = max(balance_at, policy_at)
        values = dict(balance=balance, **policy)
        history = self._scopes.setdefault(scope, [])
        if not history:
            history.append(dict(timestamp=start, order=-1, estimated=True, **values))

        def record(at, components):
            current = dict(timestamp=at, observed_at=now, order=CURRENT_ORDER,
                           estimated=False, **components)
            key = (at, CURRENT_ORDER)
            index = bisect_left(history, key, key=_FACT_ORDER)
            if index < len(history) and _FACT_ORDER(history[index]) == key:
                history[index] = current
            else:
                history.insert(index, current)

        if balance_at < policy_at:
            # A new account balance can predate a changed configuration. Apply
            # that component with the policy already observed at its clock;
            # the later full budget uses the new policy at its own clock.
            prior = next((fact for fact in reversed(history)
                          if not fact["estimated"] and fact["timestamp"] <= balance_at), None)
            if prior is not None and prior["balance"] != balance:
                record(balance_at, dict(balance=balance, **{
                    key: deepcopy(prior[key]) for key in policy
                }))
        latest = history[-1]
        if latest["estimated"] or any(latest[key] != value for key, value in values.items()):
            record(source_at, values)
        left = bisect_right(history, start, key=_TIMESTAMP) - 1
        if left > 0:
            del history[:left]
        # Only the optional exposure policy needs a recursive copy. The
        # remaining observation fields are scalars.
        snapshot = []
        memo = {}
        for fact in history:
            snapshot_fact = fact.copy()
            if fact["exposure_budget"] is not None:
                snapshot_fact["exposure_budget"] = deepcopy(fact["exposure_budget"], memo)
            snapshot.append(snapshot_fact)
        return snapshot
