import logging
from copy import deepcopy

from .normalize import normalize_config
from .parse import load_raw_config
from .project import project_config
from .runtime_compile import compile_runtime_config
from .schema import get_template_config


def load_input_config(config_path: str | None, *, log_info: bool = True) -> tuple[dict, str]:
    if config_path:
        if log_info:
            logging.info("loading config %s", config_path)
        return load_raw_config(config_path), config_path
    if log_info:
        logging.info("loading schema defaults from src/config/schema.py")
    return get_template_config(), ""


def prepare_config(
    config: dict,
    *,
    base_config_path: str = "",
    live_only: bool = False,
    verbose: bool = True,
    target: str = "canonical",
    runtime: str | None = None,
) -> dict:
    source = deepcopy(config)
    source.setdefault("_raw", deepcopy(source))
    result = normalize_config(
        source,
        base_config_path=base_config_path,
        live_only=live_only,
        verbose=verbose,
        record_step=True,
    )
    if target != "canonical":
        result = project_config(result, target, record_step=True)
    if runtime is not None:
        result = compile_runtime_config(result, runtime=runtime, record_step=True)
    return result


def load_prepared_config(
    config_path: str | None,
    *,
    live_only: bool = False,
    verbose: bool = True,
    target: str = "canonical",
    runtime: str | None = None,
    log_info: bool = True,
) -> dict:
    source, base_config_path = load_input_config(config_path, log_info=log_info)
    return prepare_config(
        source,
        base_config_path=base_config_path,
        live_only=live_only,
        verbose=verbose,
        target=target,
        runtime=runtime,
    )
