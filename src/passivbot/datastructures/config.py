from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import root_validator
from pydantic import validator
from pydantic.generics import GenericModel

from passivbot.datastructures import StopMode
from passivbot.utils.logs import SORTED_LEVEL_NAMES


BaseConfigType = TypeVar("BaseConfigType", bound="BaseConfig")


class PassivbotBaseModel(BaseModel):
    """
    Base class for configurations
    """


class LongConfig(PassivbotBaseModel):
    enabled: bool
    eprice_exp_base: float
    eprice_pprice_diff: float
    grid_span: float
    initial_qty_pct: float
    markup_range: float
    max_n_entry_orders: int
    min_markup: float
    n_close_orders: int
    wallet_exposure_limit: float
    secondary_allocation: float
    secondary_pprice_diff: float

    @validator("max_n_entry_orders", pre=True)
    @classmethod
    def _cast_max_n_entry_orders(cls, value) -> int:
        return int(round(float(value)))

    @validator("n_close_orders", pre=True)
    @classmethod
    def _cast_n_close_orders(cls, value) -> int:
        return int(round(float(value)))

    @root_validator(pre=True)
    @classmethod
    def _migrate_pre_v6_configs(cls, fields):
        replacements = (
            ("secondary_pprice_diff", "secondary_grid_spacing"),
            ("secondary_allocation", "secondary_pbr_allocation"),
            ("wallet_exposure_limit", "pbr_limit"),
        )
        for target, previous in replacements:
            if previous in fields:
                fields[target] = fields.pop(previous)
        return fields


class ApiKey(PassivbotBaseModel):
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


class SymbolConfig(PassivbotBaseModel):
    key_name: str
    config_name: str

    # Private attributes
    _name: str = PrivateAttr()

    def __repr__(self):
        """
        String representation of the config class
        """
        reprstr = "Symbol("
        if self.name:
            reprstr += f"name={self.name!r}, "
        reprstr += f"key_name={self.key_name!r}, config_name={self.config_name!r})"
        return reprstr

    @property
    def name(self) -> str:
        return self._name


class ShortConfig(PassivbotBaseModel):
    enabled: bool
    eprice_exp_base: float
    eprice_pprice_diff: float
    grid_span: float
    initial_qty_pct: float
    markup_range: float
    max_n_entry_orders: int
    min_markup: float
    n_close_orders: int
    wallet_exposure_limit: float
    secondary_allocation: float
    secondary_pprice_diff: float

    @validator("max_n_entry_orders", pre=True)
    @classmethod
    def _cast_max_n_entry_orders(cls, value) -> int:
        return int(round(float(value)))

    @validator("n_close_orders", pre=True)
    @classmethod
    def _cast_n_close_orders(cls, value) -> int:
        return int(round(float(value)))

    @root_validator(pre=True)
    @classmethod
    def _migrate_pre_v6_configs(cls, fields):
        replacements = (
            ("secondary_pprice_diff", "secondary_grid_spacing"),
            ("secondary_allocation", "secondary_pbr_allocation"),
            ("wallet_exposure_limit", "pbr_limit"),
        )
        for target, previous in replacements:
            if previous in fields:
                fields[target] = fields.pop(previous)
        return fields


class NamedConfig(PassivbotBaseModel):
    assigned_balance: Optional[float] = None
    cross_wallet_pct: float = Field(1.0, ge=0.0, le=1.0)
    last_price_diff_limit: float = 0.3
    max_leverage: int = Field(25, ge=1)
    profit_trans_pct: float = 0.0
    stop_mode: StopMode = StopMode.NORMAL
    long: LongConfig
    short: ShortConfig

    # Private attributes
    _name: str = PrivateAttr()
    _parent: BaseConfig = PrivateAttr()
    _market_type: str = PrivateAttr()
    _api_key: ApiKey = PrivateAttr()
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
    def api_key(self) -> ApiKey:
        return self._api_key

    @property
    def symbol(self) -> SymbolConfig:
        return self._symbol

    @root_validator(pre=True)
    @classmethod
    def _migrate_pre_v6_configs(cls, fields):
        if "shrt" in fields:
            fields["short"] = fields.pop("shrt")
        return fields


class DownloaderNamedConfig(NamedConfig):

    _parent: DownloaderConfig = PrivateAttr()

    @property
    def parent(self) -> DownloaderConfig:
        return self._parent


class BacktestNamedConfig(NamedConfig):

    _parent: BacktestConfig = PrivateAttr()

    @property
    def parent(self) -> BacktestConfig:
        return self._parent


class LoggingCliConfig(PassivbotBaseModel):
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


class LoggingFileConfig(PassivbotBaseModel):
    level: str = "info"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    fmt: str = "%(asctime)s,%(msecs)03d [%(name)-17s:%(lineno)-4d][%(levelname)-7s] %(message)s"
    path: Optional[pathlib.Path] = None

    @validator("level")
    @classmethod
    def _validate_level(cls, value):
        value = value.lower()
        if value.lower() not in SORTED_LEVEL_NAMES:
            raise ValueError(
                f"The log level {value!r} is not value. Available levels: {', '.join(SORTED_LEVEL_NAMES)}"
            )
        return value


class LoggingConfig(PassivbotBaseModel):
    cli: LoggingCliConfig = LoggingCliConfig()
    file: LoggingFileConfig = LoggingFileConfig()


class BaseConfig(PassivbotBaseModel):

    # Optional Configs
    logging: LoggingConfig = LoggingConfig()

    # Private attributes
    _basedir: pathlib.Path = PrivateAttr()

    @classmethod
    def parse_files(cls: type[BaseConfigType], *files: pathlib.Path) -> BaseConfigType:
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
    _active_config: NamedConfig = PrivateAttr()

    @property
    def active_config(self) -> NamedConfig:
        return self._active_config


class BaseBacktestConfig(BaseConfig, SymbolsMixin):
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None
    data_dir: Optional[pathlib.Path] = None
    backtests_dir: Optional[pathlib.Path] = None

    # Private attributes
    _active_config: NamedConfig = PrivateAttr()

    @property
    def active_config(self) -> NamedConfig:
        return self._active_config


class DownloaderConfig(BaseBacktestConfig):
    download_only: bool = False

    # Private attributes
    _active_config: DownloaderNamedConfig = PrivateAttr()

    @property
    def active_config(self) -> DownloaderNamedConfig:
        return self._active_config


class BacktestConfig(BaseBacktestConfig):

    starting_balance: float = Field(10000.0, ge=1.0)
    latency_simulation_ms: int = Field(1000, ge=0)

    # Private attributes
    _active_config: BacktestNamedConfig = PrivateAttr()

    @property
    def active_config(self) -> BacktestNamedConfig:
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
