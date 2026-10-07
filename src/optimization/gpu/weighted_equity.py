"""Requested weighted equity reductions over resident factual sample histories."""

import math

import torch

from optimization.gpu.metric_registry import (
    WEIGHTED_EQUITY_STEMS,
    WEIGHTED_RAW_EQUITY_METRICS,
    WEIGHTED_ACCOUNT_EQUITY_METRICS,
)

from optimization.gpu.metrics import (
    _masked_median,
    _mean_worst_one_pct_largest,
    _omega_ratio,
    _pct_change,
    _sharpe_sortino,
    _smoothed_adg,
)


def weighted_equity_history_bytes(sample_capacity, n_days, requested):
    """Reserve resident f32 curves and sequential f64 reduction working space.

    Controlled CUDA peak measurements use under 67 scratch bytes per observation.
    Reserve 80 plus daily work and compact results; allocator/driver overhead is
    covered by the service's separate device headroom. Curves reduce sequentially.
    """
    requested = set(requested)
    curves = int(bool(requested & WEIGHTED_RAW_EQUITY_METRICS)) + int(
        bool(requested & WEIGHTED_ACCOUNT_EQUITY_METRICS)
    )
    if not curves:
        return 0
    return int(sample_capacity) * (curves * 4 + 80) + int(n_days) * 256 + len(requested) * 8


def weighted_equity_from_samples(
    samples, *, first_timestamps_ms, sample_counts, interval_ms,
    n_days, requested, curve, fill_counts=None,
):
    """Reduce true suffixes without transferring histories to the caller.

    Raw strategy suffixes retain full-curve peaks and divide by ten. Account
    suffixes reset peaks and average the nonempty suffixes; a no-fill account
    analysis retains Rust's defaults. The caller owns the calendar dimension.
    Samples are packed from each candidate's first actual equity observation.
    """
    if curve not in ("raw_strategy", "account"):
        raise ValueError("weighted equity requires an explicit raw_strategy/account curve")
    names = {
        (f"{stem}_strategy_eq_w" if curve == "raw_strategy" else f"{stem}_w_usd"): stem
        for stem in WEIGHTED_EQUITY_STEMS
    }
    requested = set(requested)
    if unsupported := requested.difference(names):
        raise ValueError("unsupported weighted equity metrics: " + ", ".join(sorted(unsupported)))
    if not requested:
        return {}
    if (samples.ndim != 2 or not samples.is_floating_point()
            or not isinstance(n_days, int) or isinstance(n_days, bool) or n_days < 1
            or not math.isfinite(interval_ms) or interval_ms <= 0):
        raise ValueError("invalid weighted equity history shape or calendar metadata")
    batch, capacity = samples.shape
    for metadata in (first_timestamps_ms, sample_counts):
        if metadata.shape != (batch,) or metadata.device != samples.device:
            raise ValueError("weighted equity metadata must align with resident samples")
    if sample_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("weighted equity sample counts must be integers")
    if curve == "account" and (
        fill_counts is None or fill_counts.shape != (batch,)
        or fill_counts.device != samples.device
    ):
        raise ValueError("account weighted equity requires aligned actual fill counts")

    indices = torch.arange(capacity, device=samples.device)[None, :].expand(batch, -1)
    valid = indices < sample_counts[:, None]
    first = first_timestamps_ms.to(torch.float64)
    invalid = ((sample_counts < 0) | (sample_counts > capacity)
               | ((sample_counts > 0) & (~torch.isfinite(first) | (first < 0))))
    if curve == "account":
        invalid |= ~torch.isfinite(fill_counts) | (fill_counts < 0)
    # Empty histories have no clock. This unused shape origin is not a sample.
    first = torch.where(sample_counts > 0, first, torch.zeros_like(first))
    day_ids = torch.div(
        first[:, None] + indices * interval_ms, 86_400_000, rounding_mode="floor"
    ) - torch.div(first[:, None], 86_400_000, rounding_mode="floor")
    invalid |= (valid & ((day_ids < 0) | (day_ids >= n_days)
                        | ~torch.isfinite(samples))).any(dim=1)
    if bool(invalid.any()):
        raise ValueError("weighted equity history has invalid samples, counts or timestamps")
    totals = {
        name: torch.zeros(batch, dtype=torch.float64, device=samples.device)
        for name in requested
    }
    if capacity == 0:
        return totals
    # Padding after a late first observation can extend beyond the dataset's
    # calendar. It contributes neither indices nor values to a daily reduction.
    day_ids = torch.where(valid, day_ids, torch.zeros_like(day_ids)).to(torch.long)
    stems = {names[name] for name in requested}
    need_returns = bool(stems & {"mdg", "omega_ratio"})
    need_adg = bool(stems - {"mdg", "omega_ratio"})
    need_minima = bool(stems & {"sharpe_ratio", "sortino_ratio"})
    need_drawdowns = bool(stems & {"calmar_ratio", "sterling_ratio"})
    drawdown_samples = samples.to(torch.float64) if need_drawdowns else None
    full_peaks = (
        torch.cummax(torch.where(valid, drawdown_samples, float("-inf")), dim=1).values
        if need_drawdowns and curve == "raw_strategy" else None
    )
    included = torch.zeros(batch, dtype=torch.float64, device=samples.device)
    counts = sample_counts.to(torch.float64)
    for suffix in range(10):
        start = torch.floor(counts - counts / (1.0 + suffix) + 0.5).to(torch.long)
        mask = valid & (indices >= start[:, None])
        last = torch.full((batch, n_days), -1, dtype=torch.long, device=samples.device)
        last.scatter_reduce_(
            1, day_ids, torch.where(mask, indices, -1), reduce="amax", include_self=True
        )
        active = last >= 0
        ends = samples.gather(1, last.clamp(min=0)).to(torch.float64)
        values = {}
        if need_adg:
            adg = _smoothed_adg(ends, active)
            values["adg"] = adg
        if need_returns:
            returns, return_mask = _pct_change(ends, active)
            if "mdg" in stems:
                values["mdg"] = _masked_median(returns, return_mask)
            if "omega_ratio" in stems:
                values["omega_ratio"] = _omega_ratio(returns, return_mask)
        if need_minima:
            minima = torch.full(
                (batch, n_days), float("inf"), dtype=samples.dtype, device=samples.device
            )
            minima.scatter_reduce_(
                1, day_ids, torch.where(mask, samples, float("inf")),
                reduce="amin", include_self=True,
            )
            changes, change_mask = _pct_change(minima.to(torch.float64), active)
            values["sharpe_ratio"], values["sortino_ratio"] = _sharpe_sortino(
                changes, change_mask, adg
            )
        if need_drawdowns:
            peaks = full_peaks if curve == "raw_strategy" else torch.cummax(
                torch.where(mask, drawdown_samples, float("-inf")), dim=1
            ).values
            if curve == "account":
                # Ordinary Rust account drawdowns seed near-zero peaks at 1e-12
                # and clamp every subsequent record peak to that positive floor.
                # Raw strategy drawdowns deliberately preserve signed peaks.
                initial = drawdown_samples.gather(
                    1, start.clamp(max=capacity - 1)[:, None]
                )
                peaks = torch.where(
                    (initial.abs() < 1e-12) | (peaks > initial),
                    peaks.clamp(min=1e-12), peaks,
                )
            drawdowns = torch.where(
                mask, (peaks - drawdown_samples) / peaks.abs().clamp(min=1e-12), 0
            )
            daily_dd = torch.zeros(
                (batch, n_days), dtype=torch.float64, device=samples.device
            )
            daily_dd.scatter_reduce_(1, day_ids, drawdowns, reduce="amax", include_self=True)
            if "calmar_ratio" in stems:
                values["calmar_ratio"] = adg / daily_dd.max(dim=1).values.clamp(min=1e-12)
            if "sterling_ratio" in stems:
                values["sterling_ratio"] = adg / _mean_worst_one_pct_largest(
                    daily_dd, active
                ).clamp(min=1e-12)
        include = active.any(dim=1)
        included += include.to(torch.float64)
        for name in requested:
            totals[name] += torch.where(include, values[names[name]], 0)
    denominator = 10.0 if curve == "raw_strategy" else included.clamp(min=1)
    eligible = sample_counts >= 2 if curve == "raw_strategy" else fill_counts > 0
    return {
        name: torch.where(eligible, value / denominator, torch.zeros_like(value))
        for name, value in totals.items()
    }
