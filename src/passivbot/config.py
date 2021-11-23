from __future__ import annotations

import json
import pathlib
from typing import Any
from typing import Dict

from pydantic import BaseModel
from pydantic import PrivateAttr
from pydantic import root_validator
from pydantic import validator
from pydantic.generics import GenericModel

from passivbot.utils.logs import SORTED_LEVEL_NAMES


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

    # Private attributes
    _name: str = PrivateAttr()

    @validator("exchange")
    @classmethod
    def _validate_exchange(cls, value: str) -> str:
        supported_exchanges: tuple[str, str] = ("binance", "bybit")
        if value not in supported_exchanges:
            raise ValueError(
                f"The exchange {value!r} is not supported. Choose one of {', '.join(supported_exchanges)}"
            )
        return value

    @property
    def name(self) -> str:
        return self._name


class SymbolConfig(NonMutatingMixin):
    key_name: str
    config_name: str

    # Private attributes
    _name: str = PrivateAttr()

    @property
    def name(self) -> str:
        return self._name


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
    assigned_balance: float | None
    long: LongConfig
    short: ShortConfig

    # Private attributes
    _name: str = PrivateAttr()
    _parent: BaseConfig = PrivateAttr()
    _market_type: str = PrivateAttr()
    _key: ApiKey = PrivateAttr()
    _symbol: SymbolConfig = PrivateAttr()

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> BaseConfig:
        return self._parent

    @property
    def market_type(self) -> str:
        return self._market_type

    @property
    def key(self) -> ApiKey:
        return self._key

    @property
    def symbol(self) -> SymbolConfig:
        return self._symbol


class LoggingCliConfig(NonMutatingMixin):
    level: str = "info"
    datefmt: str = "%H:%M:%S"
    fmt: str = "[%(asctime)s] [%(levelname)-7s] %(message)s"

    @validator("level")
    @classmethod
    def _validate_level(cls, value):
        value = value.lower()
        if value.lower() not in SORTED_LEVEL_NAMES:
            raise ValueError(
                f"The log level {value!r} is not value. Available levels: {', '.join(SORTED_LEVEL_NAMES)}"
            )
        return value


class LoggingFileConfig(NonMutatingMixin):
    level: str = "info"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    fmt: str = "%(asctime)s,%(msecs)03d [%(name)-17s:%(lineno)-4d][%(levelname)-7s] %(message)s"
    path: pathlib.Path | None

    @validator("level")
    @classmethod
    def _validate_level(cls, value):
        value = value.lower()
        if value.lower() not in SORTED_LEVEL_NAMES:
            raise ValueError(
                f"The log level {value!r} is not value. Available levels: {', '.join(SORTED_LEVEL_NAMES)}"
            )
        return value


class LoggingConfig(NonMutatingMixin):
    cli: LoggingCliConfig = LoggingCliConfig()
    file: LoggingFileConfig = LoggingFileConfig()


class BaseConfig(NonMutatingMixin):

    # Optional Configs
    logging: LoggingConfig = LoggingConfig()

    # Private attributes
    _basedir: pathlib.Path = PrivateAttr()

    @classmethod
    def parse_files(cls, *files: pathlib.Path) -> BaseConfig:
        """
        Helper class method to load the configuration from multiple files
        """
        config_dicts: list[dict[str, Any]] = []
        for file in files:
            config_dicts.append(json.loads(file.read_text()))
        config = config_dicts.pop(0)
        if config_dicts:
            merge_dictionaries(config, *config_dicts)
        return cls.parse_raw(json.dumps(config))

    @property
    def basedir(self) -> pathlib.Path:
        return self._basedir


class ApiKeysMixin(GenericModel):
    api_keys: Dict[str, ApiKey]

    @root_validator
    @classmethod
    def _set_api_keys_name_attribute(cls, values):
        for name, config in values["api_keys"].items():
            config._name = name
        return values


class ConfigsMixin(GenericModel):
    configs: Dict[str, NamedConfig]

    @root_validator
    @classmethod
    def _set_configs_name_attribute(cls, values):
        for name, config in values["configs"].items():
            config._name = name
        return values


class SymbolsMixin(ApiKeysMixin, ConfigsMixin):
    symbols: Dict[str, SymbolConfig]

    @root_validator
    @classmethod
    def _set_symbols_name_attribute(cls, values):
        for name, config in values["symbols"].items():
            config._name = name
        return values

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


class LiveConfig(BaseConfig, SymbolsMixin):

    # Private attributes
    _symbol: SymbolConfig = PrivateAttr()
    _active_config: NamedConfig = PrivateAttr()

    @property
    def symbol(self) -> SymbolConfig:
        return self._symbol

    @property
    def active_config(self) -> NamedConfig:
        return self._active_config


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
