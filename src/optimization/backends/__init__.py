from optimization.backends.deap_backend import run_backend as run_deap_backend
from optimization.backends.pymoo_backend import run_backend as run_pymoo_backend


def run_gpu_backend(**kwargs):
    # Keep optional GPU runtime imports entirely off the normal CPU path.
    from optimization.backends.gpu_backend import run_backend

    return run_backend(**kwargs)


BACKEND_RUNNERS = {
    "deap": run_deap_backend,
    "gpu": run_gpu_backend,
    "pymoo": run_pymoo_backend,
}


def get_backend_runner(name: str):
    backend = str(name or "deap").strip().lower()
    if backend not in BACKEND_RUNNERS:
        raise ValueError(f"unsupported optimizer backend {name!r}")
    return BACKEND_RUNNERS[backend]
