from constants import MANAGER_CONFIG_PATH, CONFIG_FIELDS_ALIASES, CONFIGS_PATH
from typing import Dict, List, Union
from yaml import load, FullLoader
from instance import Instance
from os import path
from logging import error
from sys import exit


class ConfigParser:
    def __init__(self) -> None:
        self.config_file = {}
        self.existing_config_paths = {}

    def get_config(self) -> dict:
        if not path.exists(MANAGER_CONFIG_PATH):
            error("No such file: {}".format(MANAGER_CONFIG_PATH))
            exit(1)

        with open(MANAGER_CONFIG_PATH, "r") as f:
            config = load(f, Loader=FullLoader)

        if config is not None:
            self.config_file = config
            self.config_file["parsed"] = True

        return config

    def get_defaults(self) -> Dict:
        if not self.config_file.get("parsed"):
            self.get_config()

        return self.parse_scope_with_legacy_support(self.config_file.get("defaults"))

    def get_narrow_config(self, scoped_config: Dict, source=None) -> Dict:
        if type(source) is not dict:
            source = self.get_defaults()
        return dict(source, **self.parse_scope_with_legacy_support(scoped_config))

    def get_instances(self) -> Dict[str, Instance]:
        result = {}
        for user in self.config_file.get("instances", []):
            result.update(self.parse_user(user))

        return result

    def parse_user(self, user_config: Dict) -> Dict[str, Instance]:
        if type(user_config) is not dict:
            return {}

        config = self.get_narrow_config(user_config)
        config['user'] = user_config.get("user")

        instances = {}
        symbols = user_config.get("symbols")
        instances.update(self.parse_symbols(symbols, config))
        return instances

    def parse_symbols(self, symbols: List, user_config: Dict) -> Dict[str, Instance]:
        symbols_type = type(symbols)
        if symbols_type is list:
            return self.parse_symbols_list(symbols, user_config)

        if symbols_type is dict:
            return self.parse_config_symbols_dict(symbols, user_config)

        return {}

    def parse_symbols_list(self, symbols: List, user_config: Dict) -> Dict[str, Instance]:
        instances = {}
        for symbol in symbols:
            config = user_config.copy()
            config["symbol"] = symbol
            instance = self.generate_instance(config)
            instances[instance.get_id()] = instance

        return instances

    def parse_config_symbols_dict(self, symbols: Dict, user_config: Dict) -> Dict[str, Instance]:
        instances = {}
        for symbol, scoped_config in symbols.items():
            config = user_config.copy()

            scoped_config_type = type(scoped_config)
            if scoped_config_type is str:
                config["config"] = scoped_config

            elif scoped_config_type is dict:
                config.update(self.get_narrow_config(scoped_config, config))

            config["symbol"] = symbol
            instance = self.generate_instance(config)
            instances[instance.get_id()] = instance

        return instances

    def parse_scope_with_legacy_support(self, scoped_config: Dict) -> Dict:
        config = {}
        for field, aliases in CONFIG_FIELDS_ALIASES.items():
            for alias in aliases:
                value = scoped_config.get(alias)
                if (value is not None) and (value != ""):
                    config[field] = value

        return config

    def validate_config_path(self, config_path: str) -> Union[str, None]:
        exising = self.existing_config_paths.get(config_path)
        if exising is not None:
            return exising

        full_config_path = path.join(CONFIGS_PATH, config_path)
        check_in = {
            full_config_path: config_path,
            config_path: config_path
        }

        for full_path, partial_path in check_in.items():
            if path.exists(full_path):
                self.existing_config_paths[full_path] = full_path
                self.existing_config_paths[partial_path] = full_path
                return full_config_path

        return None

    def generate_instance(self, config: Dict) -> Instance:
        full_config_path = self.validate_config_path(config.get("config"))
        if full_config_path is None:
            error(
                "{}-{}: config does not exist".format(config.get("user"), config.get("symbol")))
        else:
            config["config"] = full_config_path

        return Instance(config)
