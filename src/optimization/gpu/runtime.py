"""Optional GPU runtime selection; CPU entry points do not import torch."""

from __future__ import annotations

import time


def gpu_device(torch_module=None) -> str:
    if torch_module is None:
        import torch as torch_module
    if torch_module.backends.mps.is_available():
        return "mps"
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "GPU optimization requires available Apple MPS or NVIDIA CUDA; "
        "use optimize.backend='pymoo' or 'deap' for CPU optimization"
    )


def wait_for_cuda_stream() -> None:
    """Yield host CPU time during long waits for the current CUDA stream."""
    import torch

    if gpu_device(torch) != "cuda":
        return
    completed = torch.cuda.Event()
    completed.record()
    # Keep short kernels on the active-wait path to avoid host wake-up latency.
    # Longer waits yield so CUDA screening does not occupy an exact-worker core.
    active_until = time.perf_counter() + 0.5
    while not completed.query():
        if time.perf_counter() >= active_until:
            time.sleep(0.001)


def synchronize() -> None:
    import torch

    if gpu_device(torch) == "mps":
        torch.mps.synchronize()
    else:
        wait_for_cuda_stream()
        # Preserve this helper's device-wide contract, including other streams.
        torch.cuda.synchronize()


def compile_shader(source: str, *, cuda_coin_capacity: int | None = None):
    import torch

    if gpu_device(torch) == "mps":
        return torch.mps.compile_shader(source)
    from optimization.gpu.cuda_kernel import CudaShaderLibrary

    return CudaShaderLibrary(source, coin_capacity=cuda_coin_capacity)


def checkpoint_runtime(torch_module) -> dict:
    """Keep existing MPS checkpoints stable and distinguish CUDA numerical state."""
    if gpu_device(torch_module) == "mps":
        return {}
    import cupy

    return {
        "cuda_runtime": {
            "kernel_contract": 1,
            "torch": str(torch_module.__version__),
            "cupy": str(cupy.__version__),
            "compute_capability": list(torch_module.cuda.get_device_capability()),
        }
    }
