from optimization.backends.deap_backend import run_backend as run_deap_backend
from optimization.backends.pymoo_backend import run_backend as run_pymoo_backend


BACKEND_RUNNERS = {
    "deap": run_deap_backend,
    "pymoo": run_pymoo_backend,
}


def get_backend_runner(name: str):
    backend = str(name or "deap").strip().lower()
    if backend not in BACKEND_RUNNERS:
        raise ValueError(f"unsupported optimizer backend {name!r}")
    return BACKEND_RUNNERS[backend]
