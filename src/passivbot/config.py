import json
import pathlib
from typing import Any
from typing import Dict
from typing import List

from pydantic import BaseModel


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


class PassivBotConfig(NonMutatingMixin):
    api_keys: Dict[str, ApiKey]
    configs: Dict[str, NamedConfig]
    symbols: Dict[str, SymbolConfig]

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
