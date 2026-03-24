"""
pymoo Problem definition for passivbot optimization.
"""

import logging
from itertools import permutations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from optimization.bounds import extract_bounds_arrays
from limit_utils import expand_limit_checks, compute_limit_violation
from metrics_schema import flatten_metric_stats

# Metrics where higher values are better (will be negated for pymoo minimization).
# All others default to minimize (lower is better).
MAXIMIZE_METRICS = {
    "adg", "adg_w", "adg_pnl", "adg_pnl_w",
    "adg_per_exposure_long", "adg_per_exposure_short",
    "adg_w_per_exposure_long", "adg_w_per_exposure_short",
    "calmar_ratio", "calmar_ratio_w",
    "entry_initial_balance_pct_long", "entry_initial_balance_pct_short",
    "gain", "gain_per_exposure_long", "gain_per_exposure_short",
    "mdg", "mdg_w", "mdg_pnl", "mdg_pnl_w",
    "mdg_per_exposure_long", "mdg_per_exposure_short",
    "mdg_w_per_exposure_long", "mdg_w_per_exposure_short",
    "omega_ratio", "omega_ratio_w",
    "sharpe_ratio", "sharpe_ratio_w",
    "sharpe_ratio_pnl", "sharpe_ratio_pnl_w",
    "sortino_ratio", "sortino_ratio_w",
    "sortino_ratio_pnl", "sortino_ratio_pnl_w",
    "sterling_ratio", "sterling_ratio_w",
    "volume_pct_per_day_avg", "volume_pct_per_day_avg_w",
}


def _resolve_scoring_key(scoring_key):
    """
    Generate candidate flat_stats keys for a scoring key.

    For "adg", produces: adg_mean, adg_usd_mean, adg_btc_mean, etc.
    Returns (candidates_list, sign) where sign is -1.0 for maximize metrics.
    """
    candidates = []
    seen = set()

    parts = scoring_key.split("_")
    if len(parts) <= 1:
        base_candidates = [scoring_key]
    else:
        base_candidates = []
        for perm in permutations(parts):
            c = "_".join(perm)
            if c not in seen:
                base_candidates.append(c)
                seen.add(c)

    for candidate in base_candidates:
        if candidate not in seen:
            seen.add(candidate)
        candidates.append(f"{candidate}_mean")
        for suffix in ("usd", "btc"):
            candidates.append(f"{candidate}_{suffix}_mean")
            cparts = candidate.split("_")
            if len(cparts) >= 2:
                inserted = "_".join(cparts[:-1] + [suffix, cparts[-1]])
                candidates.append(f"{inserted}_mean")

    # Determine sign: check if any known maximize metric matches
    sign = 1.0
    base = scoring_key
    for m in MAXIMIZE_METRICS:
        if base == m or base.startswith(m + "_") or base.endswith("_" + m):
            sign = -1.0
            break
    # Also check with currency prefixes stripped
    for prefix in ("usd_", "btc_"):
        if base.startswith(prefix):
            stripped = base[len(prefix):]
            if stripped in MAXIMIZE_METRICS:
                sign = -1.0
                break

    return candidates, sign


class PassivbotProblem(ElementwiseProblem):
    """
    Elementwise pymoo Problem for passivbot parameter optimization.

    Evaluates a single individual in _evaluate(). Parallelization is handled
    by pymoo via the elementwise_runner passed at construction time
    (e.g. StarmapParallelization wrapping a multiprocessing.Pool).
    """

    def __init__(self, config, evaluator, **kwargs):
        xl, xu, keys = extract_bounds_arrays(config)

        scoring_keys = config["optimize"]["scoring"]
        n_obj = len(scoring_keys)

        # Build limit checks for constraint evaluation
        objective_index_map = {}
        for idx, metric in enumerate(scoring_keys):
            objective_index_map.setdefault(metric, []).append(idx)

        limits = config["optimize"].get("limits", [])
        limit_checks = expand_limit_checks(
            limits,
            _build_scoring_weights(scoring_keys),
            penalty_weight=1.0,
            objective_index_map=objective_index_map,
        )

        super().__init__(
            n_var=len(keys),
            n_obj=n_obj,
            n_ieq_constr=len(limit_checks),
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self.keys = keys
        self.config = config
        self.evaluator = evaluator
        self.scoring_keys = scoring_keys
        self.limit_checks = limit_checks
        self.overrides_list = config.get("optimize", {}).get("enable_overrides", [])
        self.log = logging.getLogger(__name__)

        # Pre-resolve scoring keys to flat_stats lookup candidates + signs
        self._resolved_keys = []
        for sk in scoring_keys:
            candidates, sign = _resolve_scoring_key(sk)
            self._resolved_keys.append((candidates, sign))

    def _score_metrics(self, flat_stats):
        """Look up each scoring key in flat_stats and apply sign for minimization."""
        scores = []
        for candidates, sign in self._resolved_keys:
            val = None
            for key in candidates:
                if key in flat_stats:
                    val = flat_stats[key]
                    break
            if val is None:
                val = 0.0
            scores.append(float(val) * sign)
        return scores

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a single individual.

        x: 1D numpy array (n_var,)
        out["F"]: 1D array (n_obj,) objectives (minimize)
        out["G"]: 1D array (n_ieq_constr,) constraints (g <= 0 feasible)
        out["metrics"]: dict passed through to ParetoWriterCallback
        """
        try:
            metrics = self.evaluator.evaluate(list(x), self.overrides_list)
        except Exception as exc:
            self.log.warning("Evaluation failed: %s", exc)
            out["F"] = [1e18] * self.n_obj
            if self.limit_checks:
                out["G"] = [1e18] * len(self.limit_checks)
            out["metrics"] = {}
            return

        stats = metrics.get("stats", {}) if metrics else {}
        flat_stats = flatten_metric_stats(stats) if stats else {}

        overrides = metrics.get("flat_stats_override", {}) if metrics else {}
        if overrides:
            flat_stats.update(overrides)

        out["F"] = self._score_metrics(flat_stats)

        if self.limit_checks:
            G = []
            for check in self.limit_checks:
                val = flat_stats.get(check["metric_key"])
                violation = compute_limit_violation(check, val)
                G.append(violation if violation > 0 else -1.0)
            out["G"] = G

        out["metrics"] = metrics or {}


def _build_scoring_weights(scoring_keys):
    """Build a scoring_weights dict for expand_limit_checks compatibility."""
    weights = {}
    for sk in scoring_keys:
        _, sign = _resolve_scoring_key(sk)
        weights[sk] = sign
    return weights
