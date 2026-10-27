"""GPU metric selection; all production arithmetic belongs to Rust."""

from config.metrics import CORE_METRIC_CATALOG, metric_base_name

SUPPORTED_METRICS = tuple(CORE_METRIC_CATALOG)


def validate_gpu_metric_names(metric_names, *, native_replay=True, device=None):
    if native_replay is not True:
        raise ValueError("GPU core metrics require native factual replay")
    if device not in {None, "mps", "cuda"}:
        raise ValueError("GPU core metrics require Metal or CUDA")
    requested = frozenset(metric_base_name(name) for name in metric_names)
    unsupported = sorted(requested - CORE_METRIC_CATALOG.keys())
    if unsupported:
        raise ValueError(f"GPU core metrics are not implemented: {unsupported}")
    return requested
