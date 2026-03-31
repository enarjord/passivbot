from copy import deepcopy

from .transform_log import record_transform


def compile_runtime_config(config: dict, runtime: str = "generic", *, record_step: bool = True) -> dict:
    import config_utils as legacy

    normalized_runtime = str(runtime).strip().lower()
    result = deepcopy(config)
    legacy._apply_forager_internal_aliases(result)
    if record_step:
        record_transform(result, "compile_runtime_config", {"runtime": normalized_runtime})
    return result
