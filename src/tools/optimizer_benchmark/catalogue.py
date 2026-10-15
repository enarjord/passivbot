"""Declared workload coverage. Configuration coverage is not transition proof."""

from dataclasses import asdict, dataclass
from itertools import combinations, product


STRATEGIES = ("trailing_martingale", "ema_anchor")
FEATURES = (
    "hsl_coin", "hsl_pside", "hsl_unified", "hsl_market", "hsl_never",
    "hsl_threshold", "hsl_scaled", "unstuck", "unstuck_no_ema", "cooldown",
    "adaptive_cooldown", "overrides", "excess", "position_enforcer", "total_enforcer",
    "entry_gate_off", "volatility", "exposure_weight", "passive", "entry_trailing",
    "close_trailing", "gate_disabled", "gate_all", "gate_reentry", "one_way",
    "loss_gate", "min_cost", "fill_buffer", "slippage", "dynamic_wel_off",
    "eligibility", "weighted_metrics", "forager_unilateral", "forager_hysteresis",
    "btc_collateral", "total_enforcer_portfolio", "coarse_candles",
)
TM_FEATURES = {"exposure_weight", "passive", "entry_trailing", "close_trailing",
               "gate_disabled", "gate_all", "gate_reentry"}
PAIRWISE_FLAGS = ("unstuck", "cooldown", "overrides", "excess", "total_enforcer",
                  "entry_gate_off", "volatility", "hsl_coin")


@dataclass(frozen=True)
class Case:
    strategy: str
    sides: str = "both"
    topology: str = "forager"
    features: tuple[str, ...] = ()
    regime: str = "oscillating"

    @property
    def name(self):
        suffix = "+".join(self.features) or "base"
        return f"{self.strategy}/{self.sides}/{self.topology}/{suffix}/{self.regime}"

    def record(self):
        return dict(asdict(self), name=self.name)


def pairwise_rows(flags=PAIRWISE_FLAGS):
    """Deterministic greedy binary covering array; every pair has all four states."""
    pairs = list(combinations(range(len(flags)), 2))
    uncovered = {(i, j, a, b) for i, j in pairs for a, b in product((0, 1), repeat=2)}
    candidates = list(product((0, 1), repeat=len(flags)))
    result = []
    while uncovered:
        best = max(candidates, key=lambda row: sum((i, j, row[i], row[j]) in uncovered
                                                 for i, j in pairs))
        result.append(tuple(flag for flag, enabled in zip(flags, best) if enabled))
        uncovered.difference_update((i, j, best[i], best[j]) for i, j in pairs)
        candidates.remove(best)
    return result


def catalogue():
    cases = [Case(strategy, side, topology) for strategy, side, topology in
             product(STRATEGIES, ("long", "short", "both"), ("single", "fixed", "forager"))]
    for strategy in STRATEGIES:
        cases.extend(Case(strategy, features=(feature,), regime="shock" if feature.startswith("hsl")
                          else "oscillating") for feature in FEATURES
                     if strategy == "trailing_martingale" or feature not in TM_FEATURES)
        cases.extend(Case(strategy, features=features, regime="shock") for features in pairwise_rows())
        cases.extend(Case(strategy, features=("hsl_coin", "unstuck", "total_enforcer", "overrides"),
                          regime=regime) for regime in ("trend", "shock"))
    cases.append(Case("trailing_grid_v7", "long", "single"))
    return list({case.name: case for case in cases}.values())


SMOKE = (
    Case("trailing_martingale", "long", "single"),
    Case("trailing_martingale", "short", "fixed"),
    Case("trailing_martingale"),
    Case("ema_anchor", "long", "single"),
    Case("ema_anchor", "short", "fixed"),
    Case("ema_anchor"),
    Case("trailing_martingale", features=("hsl_coin",), regime="shock"),
    Case("trailing_martingale", features=("unstuck", "overrides", "excess", "cooldown")),
)


def select_cases(profile, patterns=()):
    import fnmatch
    cases = list(SMOKE) if profile == "smoke" else catalogue()
    if patterns:
        cases = [case for case in cases if any(fnmatch.fnmatchcase(case.name, p) for p in patterns)]
    if not cases:
        raise ValueError("case selection is empty")
    return cases


def declared_gpu_support(case):
    if case.strategy == "trailing_grid_v7":
        return "CPU-only strategy"
    if "btc_collateral" in case.features:
        return "GPU collateral simulation unsupported"
    if case.strategy == "ema_anchor" and "position_enforcer" in case.features:
        return "GPU EMA Anchor position exposure enforcement unsupported"
    return None
