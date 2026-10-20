"""Requested HSL EMA observations reduced without transferring their histories."""

import torch


def drawdown_ema_tail_history_bytes(sample_capacity: int, scope_count: int) -> int:
    """Admission allowance for active-side/portfolio histories and their device reduction.

    Channel-major capture needs four bytes per bar and scope. Scope reductions run
    separately, so their masks, absolute values, top-k values/indices and reducer
    workspace need one scope's additional allowance. Replay capture has no autograd
    history; only compact scope results survive each reduction.
    This is part of replay scratch admission, not a bound on total device memory
    or on the allocator's retained cache.
    """
    return scope_count * (4 * sample_capacity + 64) + 44 * sample_capacity


def _drawdown_ema_tail_scope(matrix):
    """Reduce one scope and release its full-width intermediates on return."""
    capacity = matrix.shape[1]
    finite = torch.isfinite(matrix)
    counts = finite.sum(dim=1)
    values = torch.where(finite, matrix.abs(), 0.0)
    # The capacity bounds every row's requested tail; row counts stay on-device.
    top = torch.topk(values, max(capacity // 100, 1), dim=1, sorted=True).values
    wanted = (counts // 100).clamp(min=1)
    selected = torch.arange(top.shape[1], device=matrix.device)[None, :] < wanted[:, None]
    means = torch.where(selected, top, 0.0).sum(dim=1) / wanted
    return torch.where(counts > 0, means, 0.0)


def drawdown_ema_tail_from_samples(samples):
    """Mean of the largest max(floor(valid_count / 100), 1) per scope.

    Shape is (candidate, scope, bar), with active-side/portfolio channels.
    NaNs mark unobserved bars;
    they do not enlarge the tail denominator. The portfolio channel contains
    per-bar scope maxima, rather than a maximum of separately reduced tails.
    All intermediate tensors and the compact (candidate, scope) result stay on the
    input device. Float32 trajectory differences from CPU replay remain possible.
    """
    if samples.dtype != torch.float32 or samples.ndim != 3 or samples.shape[1] not in (2, 3):
        raise ValueError("HSL EMA tail requires float32 candidate-by-scope samples")
    batch_size, scope_count, capacity = samples.shape
    if not batch_size or not capacity:
        return torch.zeros((batch_size, scope_count), dtype=samples.dtype, device=samples.device)
    scope_means = []
    for scope in range(scope_count):
        if scope and samples.device.type == "mps":
            # MPS command buffers may retain PyTorch-freed intermediates until
            # completion. Finish the previous helper's work before allocating
            # another scope's workspace. CUDA's current-stream allocator can
            # reuse freed buffers in stream order without a host wait.
            torch.mps.synchronize()
        scope_means.append(_drawdown_ema_tail_scope(samples[:, scope, :]))
    return torch.stack(scope_means, dim=1)
