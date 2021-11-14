import json
import pathlib
from typing import Any
from typing import Dict
from typing import List

from pydantic import BaseModel
from pydantic import validator


class NonMutatingMixin(BaseModel):
    """
    Base class for non mutating configurations
    """

    class Config:

        allow_mutation = False


class ApiKey(NonMutatingMixin):
    exchange: str
    key: str
    secret: str


class LongConfig(NonMutatingMixin):
    enabled: bool
    eprice_exp_base: float
    eprice_pprice_diff: float
    grid_span: float
    initial_qty_pct: float
    markup_range: float
    max_n_entry_orders: float
    min_markup: float
    n_close_orders: float
    wallet_exposure_limit: float
    secondary_allocation: float
    secondary_pprice_diff: float


class ShortConfig(NonMutatingMixin):
    enabled: bool
    eprice_exp_base: float
    eprice_pprice_diff: float
    grid_span: float
    initial_qty_pct: float
    markup_range: float
    max_n_entry_orders: float
    min_markup: float
    n_close_orders: float
    wallet_exposure_limit: float
    secondary_allocation: float
    secondary_pprice_diff: float


class NamedConfig(NonMutatingMixin):
    long: LongConfig
    short: ShortConfig


class SymbolConfig(NonMutatingMixin):
    key_name: str
    config_name: str


class LoggingCliConfig(NonMutatingMixin):
    level: str = "warning"
    datefmt: str = "%H:%M:%S"
    fmt: str = "[%(asctime)s][%(levelname)-7s] - %(message)s"


class LoggingFileConfig(NonMutatingMixin):
    level: str = "warning"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    fmt: str = "%(asctime)s,%(msecs)03d [%(name)-17s:%(lineno)-4d][%(levelname)-7s] %(message)s"
    path: pathlib.Path = pathlib.Path("logs/passivbot.log")


class LoggingConfig(NonMutatingMixin):
    cli: LoggingCliConfig = LoggingCliConfig()
    file: LoggingFileConfig = LoggingFileConfig()


class PassivBotConfig(NonMutatingMixin):
    api_keys: Dict[str, ApiKey]
    configs: Dict[str, NamedConfig]
    symbols: Dict[str, SymbolConfig]

    # Optional Configs
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def parse_files(cls, *files: pathlib.Path) -> "PassivBotConfig":
        """
        Helper class method to load the configuration from multiple files
        """
        config_dicts: List[Dict[str, Any]] = []
        for file in files:
            config_dicts.append(json.loads(file.read_text()))
        config = config_dicts.pop(0)
        if config_dicts:
            merge_dictionaries(config, *config_dicts)
        return cls.parse_raw(json.dumps(config))

    @validator("symbols", each_item=True)
    @classmethod
    def _validate_symbols_mapping(cls, value, values, field, **kwargs):
        api_keys = values["api_keys"]
        configs = values["configs"]
        if value.key_name not in api_keys:
            raise ValueError(f"The {value.key_name!r} key name is not defined under 'api_keys'.")
        if value.config_name not in configs:
            raise ValueError(
                f"The {value.config_name!r} configuration name is not defined under 'configs'."
            )
        return value


def merge_dictionaries(target_dict: Dict[Any, Any], *source_dicts: Dict[Any, Any]) -> None:
    """
    Recursively merge each of the ``source_dicts`` into ``target_dict`` in-place
    """
    for source_dict in source_dicts:
        for key, value in source_dict.items():
            if isinstance(value, dict):
                target_dict_value = target_dict.setdefault(key, {})
                merge_dictionaries(target_dict_value, value)
            else:
                target_dict[key] = value
