import logging
from copy import deepcopy

from .normalize import normalize_config
from .parse import load_raw_config
from .project import project_config
from .runtime_compile import compile_runtime_config
from .schema import get_template_config


def strip_persisted_hsl_incomplete_history_override(source: dict, config_path: str) -> None:
    """live.hsl_accept_incomplete_history waives the HSL fail-closed coverage
    contract and must only ever be granted by the CLI flag of the current
    invocation; a value persisted in a config file is stripped here, before
    CLI overrides are applied, so it can never survive a restart."""
    containers = [source]
    live = source.get("live")
    if isinstance(live, dict):
        containers.append(live)
    for container in containers:
        if container.get("hsl_accept_incomplete_history"):
            logging.critical(
                "[risk] ignoring hsl_accept_incomplete_history=true persisted in %s: "
                "this override is per-run CLI-only (--hsl-accept-incomplete-history) "
                "and persisted values are stripped to keep HSL coverage fail-closed",
                config_path or "<config>",
            )
            container.pop("hsl_accept_incomplete_history", None)


def load_input_config(
    config_path: str | None, *, log_info: bool = True
) -> tuple[dict, str, dict]:
    if config_path:
        if log_info:
            logging.info("loading config %s", config_path)
        source = load_raw_config(config_path)
        strip_persisted_hsl_incomplete_history_override(source, config_path)
        return source, config_path, deepcopy(source)
    if log_info:
        logging.info("loading schema defaults from src/config/schema.py")
    source = get_template_config()
    return source, "", deepcopy(source)


def prepare_config(
    config: dict,
    *,
    base_config_path: str = "",
    live_only: bool = False,
    verbose: bool = True,
    log_config_transforms: bool | None = None,
    target: str = "canonical",
    runtime: str | None = None,
    raw_snapshot: dict | None = None,
    effective_snapshot: dict | None = None,
) -> dict:
    source = deepcopy(config)
    if raw_snapshot is None:
        raw_snapshot = deepcopy(source.get("_raw", source))
    if effective_snapshot is None:
        effective_snapshot = deepcopy(source.get("_raw_effective", source))
    source["_raw"] = deepcopy(raw_snapshot)
    source["_raw_effective"] = deepcopy(effective_snapshot)
    normalize_verbose = verbose if log_config_transforms is None else log_config_transforms
    result = normalize_config(
        source,
        base_config_path=base_config_path,
        live_only=live_only,
        verbose=normalize_verbose,
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
    log_config_transforms: bool | None = None,
    target: str = "canonical",
    runtime: str | None = None,
    log_info: bool = True,
) -> dict:
    source, base_config_path, raw_snapshot = load_input_config(config_path, log_info=log_info)
    return prepare_config(
        source,
        base_config_path=base_config_path,
        live_only=live_only,
        verbose=verbose,
        log_config_transforms=log_config_transforms,
        target=target,
        runtime=runtime,
        raw_snapshot=raw_snapshot,
    )
