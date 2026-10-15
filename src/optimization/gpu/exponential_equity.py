"""Daily exponential analysis matching Rust's explicit EW definition.

The native replay supplies factual daily equity and first/last observation
clocks. Missing active observations are producer failures. Short histories and
nonpositive equity are explicitly unavailable, represented by -inf scalars so
the canonical optimizer scorer rejects them rather than choosing a subset.
"""

from __future__ import annotations

import torch

from config.ew import EW_DEFINITION_VERSION, validate_ew_half_life_days
from optimization.gpu.metric_registry import EXPONENTIAL_EQUITY_METRICS

DAY_MS = 86_400_000


def exponential_equity_metrics(out, run, data, requested, *, relative_timestamps=False):
    """Return requested scalar tensors and a separate batched metadata sidecar.

    Status codes: 0 available, 1 insufficient complete days, 2 nonpositive equity,
    3 liquidated.
    Daily end timestamps are UTC midnight boundaries (exclusive), with equity
    sampled at the preceding candle. Partial boundary days never become returns.
    """
    requested = set(requested) & EXPONENTIAL_EQUITY_METRICS
    if not requested:
        return {}, {}
    half_life = validate_ew_half_life_days(data.get("ew_half_life_days"))
    if half_life is None:
        raise ValueError("Exponential metrics require backtest.ew_half_life_days")
    interval = int(run.interval_ms)
    origin = int(data["ts0"])
    if interval <= 0 or DAY_MS % interval != 0 or origin % interval != 0:
        raise ValueError("Exponential metrics require an aligned UTC candle grid")
    day_end = out["day_end_eq"]
    if day_end.ndim != 2 or day_end.shape[1] == 0:
        raise RuntimeError("EW daily equity must be a nonempty batch-by-day tensor")
    batch, days = day_end.shape
    device = day_end.device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    tiny = torch.finfo(dtype).tiny if device.type == "mps" else 1e-300
    candle_count = int(data["n"])
    expected_days = (origin + (candle_count - 1) * interval) // DAY_MS - origin // DAY_MS + 1
    if candle_count < 1 or days != expected_days:
        raise RuntimeError("EW daily equity shape disagrees with the evaluation candle grid")

    def daily(name):
        value = out[name]
        if value.shape != day_end.shape or value.device != device:
            raise RuntimeError(f"EW daily output {name} has inconsistent shape/device")
        return value.to(dtype)

    def clock(name):
        value = out[name]
        if value.shape != (batch,) or value.device != device:
            raise RuntimeError(f"EW observation clock {name} has inconsistent shape/device")
        if device.type == "mps" and not relative_timestamps:
            raise ValueError("Metal EW requires relative observation clocks")
        value = value.to(dtype)
        # Native f32 relative clocks can round at long spans. Recover candle
        # indices before adding the UTC origin, just as the replay decoder does.
        relative = value if relative_timestamps else value - origin
        rounded = torch.round(relative / interval).to(dtype) * interval
        if not relative_timestamps and bool((torch.isfinite(relative) & (torch.abs(relative - rounded) > 1e-3)).any()):
            raise RuntimeError("EW absolute observation clock is off the candle grid")
        if device.type == "mps":
            steps = torch.round(torch.where(torch.isfinite(relative), relative, 0) / interval)
            return steps.to(torch.int64) * interval + origin
        return rounded + origin

    first = clock("first_eq_ts")
    last = clock("last_eq_ts")
    first_valid = torch.isfinite(out["first_eq_ts"])
    last_valid = torch.isfinite(out["last_eq_ts"])
    has_clock = first_valid & last_valid
    if bool((first_valid != last_valid).any()):
        raise RuntimeError("EW first/last observation clocks disagree")
    if bool((has_clock & ((first < origin) | (last < first))).any()):
        raise RuntimeError("EW observation clock is outside the candle grid")
    evaluation_end = origin + candle_count * interval
    if bool((has_clock & (last >= evaluation_end)).any()):
        raise RuntimeError("EW observation clock exceeds evaluation end")
    starts = (torch.arange(days, device=device, dtype=torch.int64) + origin // DAY_MS) * DAY_MS
    ends = starts + DAY_MS
    occupied = (has_clock[:, None] & (starts[None, :] <= last[:, None])
                & (ends[None, :] > first[:, None]))
    # Missing a day inside a factual continuous replay is fatal. Never bridge it
    # by pct_change over the remaining days.
    account_min = daily("day_min_eq")
    if bool((occupied & ~torch.isfinite(account_min)).any()):
        raise RuntimeError("EW active daily samples are missing/nonfinite")
    complete = (occupied & (starts[None, :] >= first[:, None])
                & ((ends - interval)[None, :] <= last[:, None]))
    returns_mask = complete[:, 1:] & occupied[:, :-1]
    returns_mask &= (ends[:-1] - interval)[None, :] >= first[:, None]
    return_ends = ends[1:]
    observation_count = returns_mask.sum(dim=1)
    # Shift to the newest admitted return before exponentiation. The common
    # factor cancels after normalization and avoids underflow on old scenarios.
    if days > 1:
        latest = torch.where(returns_mask, return_ends[None, :], 0).max(dim=1).values
        earliest = torch.where(returns_mask, return_ends[None, :], evaluation_end).min(dim=1).values
    else:
        latest = torch.zeros(batch, device=device, dtype=torch.int64)
        earliest = torch.full((batch,), evaluation_end, device=device, dtype=torch.int64)
    ages = (latest[:, None] - return_ends[None, :]).to(dtype) / DAY_MS
    weights = torch.where(returns_mask, torch.exp2(-ages / half_life), torch.zeros_like(ages))
    sums = weights.sum(dim=1)
    probabilities = weights / sums.clamp(min=tiny)[:, None]
    ess = sums.square() / weights.square().sum(dim=1).clamp(min=tiny)
    metadata = {
        "definition_version": EW_DEFINITION_VERSION,
        "half_life_days": half_life,
        "evaluation_end_ms": evaluation_end,
        "observation_count": observation_count,
        "effective_sample_size": ess,
        "first_return_end_ms": torch.where(observation_count > 0, earliest, -1),
        "last_return_end_ms": torch.where(observation_count > 0, latest, -1),
        "curves": {},
    }
    metrics = {}
    curves = {
        "strategy_eq": ("raw_strategy_day_end_eq", "raw_strategy_day_min_eq", "_strategy_eq_ew"),
        "usd": ("day_end_eq", "day_min_eq", "_ew_usd"),
        "btc": ("btc_day_end_eq", "btc_day_min_eq", "_ew_btc"),
    }
    for curve, (end_name, min_name, suffix) in curves.items():
        names = {name for name in requested if name.endswith(suffix)}
        if not names:
            continue
        missing = {end_name, min_name} - set(out)
        if missing:
            raise RuntimeError("EW factual curve output is missing " + ", ".join(sorted(missing)))
        equity, minimum = daily(end_name), daily(min_name)
        if bool((occupied & (~torch.isfinite(equity) | ~torch.isfinite(minimum))).any()):
            raise RuntimeError(f"EW {curve} active daily samples are nonfinite")
        ruined = (occupied & ((equity <= 0) | (minimum <= 0))).any(dim=1)
        status = torch.where(ruined, 2, torch.where(observation_count < 2, 1, 0))
        if "liq_step" in out:
            liquidation = out["liq_step"]
            if liquidation.shape != (batch,) or not bool(torch.isfinite(liquidation).all()):
                raise RuntimeError("EW liquidation status is malformed")
            status = torch.where(liquidation.to(device=device) >= 0, 3, status)
        available = status == 0
        # Safe temporary values are only for tensor arithmetic; the explicit
        # unavailable status replaces every result for a ruined candidate.
        safe = torch.where(occupied & (equity > 0), equity, torch.ones_like(equity))
        returns = torch.where(returns_mask, (safe[:, 1:] - safe[:, :-1]) / safe[:, :-1], 0.0)
        if bool((available[:, None] & returns_mask & (~torch.isfinite(returns) | (returns <= -1))).any()):
            raise RuntimeError(f"EW {curve} daily returns are not finite or representable")
        mean = (probabilities * returns).sum(dim=1)
        variance = (probabilities * (returns - mean[:, None]).square()).sum(dim=1)
        downside = (probabilities * returns.clamp(max=0).square()).sum(dim=1)
        positive = (probabilities * returns.clamp(min=0)).sum(dim=1)
        negative = (probabilities * (-returns).clamp(min=0)).sum(dim=1)
        log_return = torch.log1p(returns)
        values = {
            "adg": torch.expm1((probabilities * log_return).sum(dim=1)),
            "sharpe_ratio": (mean / variance.sqrt().clamp(min=1e-12)).clamp(-1000, 1000),
            "sortino_ratio": (mean / downside.sqrt().clamp(min=1e-12)).clamp(-1000, 1000),
            "omega_ratio": (positive / negative.clamp(min=1e-12)).clamp(0, 1000),
        }
        for stem, value in values.items():
            if bool((available & ~torch.isfinite(value)).any()):
                raise RuntimeError(f"EW {curve} weighted moments are nonfinite")
            name = stem + suffix
            if name in names:
                metrics[name] = torch.where(available, value, -torch.inf)
        metadata["curves"][curve] = {"status": status, "available": available}
    return metrics, metadata
