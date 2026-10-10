"""Exact native initial-entry intervals, reduced before device histories escape."""

import torch

ENTRY_INTERVAL_NAMES = (
    "entry_interval_hours_mean", "entry_interval_hours_median",
    "entry_interval_hours_p95", "entry_interval_hours_p99", "entry_interval_hours_max",
)
MAX_EXACT_ENTRY_BARS = 1 << 24


def entry_interval_history_bytes(n_bars):
    """Histogram, int64 CDF/conversion workspace, rank results and streamed stats.

    The allocation is four bytes per count; admission also reserves conservative
    device reduction scratch. This is not a bound on total GPU memory or retained
    allocator caches. No full history is transferred to the CPU.
    """
    if type(n_bars) is not int or not 3 <= n_bars <= MAX_EXACT_ENTRY_BARS:
        raise ValueError("exact native entry intervals require 3..2**24 bars")
    return 32 * (n_bars + 2) + 512


def entry_intervals_from_counts(counts, interval_ms):
    """Return mean, median, p95, p99, max hours from ordered integer-step bins.

    Column zero holds the total; remaining columns represent gaps 0..T. These
    are consecutive normal initial entries per coin/side, without leading or
    trailing intervals. Percentiles interpolate ranks like Rust's sorted fills.
    """
    if counts.dtype != torch.int32 or counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError("native entry intervals require int32 candidate-by-gap counts")
    if not isinstance(interval_ms, int) or isinstance(interval_ms, bool) or interval_ms < 1:
        raise ValueError("entry interval duration must be positive integer milliseconds")
    if bool((counts < 0).any()):
        raise RuntimeError("native entry interval count overflow or invalid gap")
    total = counts[:, 0].to(torch.int64)
    cumulative = counts[:, 1:].cumsum(dim=1, dtype=torch.int64)
    if bool((cumulative[:, -1] != total).any()):
        raise RuntimeError("native entry interval histogram count disagrees with total")
    ranks = (total - 1).clamp(min=0).to(torch.float64)[:, None] * torch.tensor(
        [.5, .95, .99], dtype=torch.float64, device=counts.device,
    )
    lower, upper = ranks.floor().to(torch.int64), ranks.ceil().to(torch.int64)
    queries = torch.cat((lower + 1, upper + 1, total[:, None]), dim=1)
    indices = torch.searchsorted(cumulative, queries).clamp(max=cumulative.shape[1] - 1)
    lo, hi = indices[:, :3].double(), indices[:, 3:6].double()
    quantiles = lo + (hi - lo) * (ranks - lower.double())
    # Sum of gaps from their CDF avoids an additional candidate-by-bar product.
    sum_steps = (cumulative.shape[1] - 1) * total - cumulative[:, :-1].sum(dim=1)
    mean = sum_steps.double() / total.clamp(min=1).double()
    metrics = torch.cat((mean[:, None], quantiles, indices[:, -1:].double()), dim=1)
    metrics *= interval_ms / 3_600_000.0
    return torch.where(total[:, None] > 0, metrics, 0.0)
