from constants import MANAGER_CONFIG_PATH, CONFIGS_PATH, PASSIVBOT_PATH, MANAGER_CONFIG_SETTINGS_PATH
from typing import Dict, List, Union
from yaml import load, FullLoader
from logging import error, info
from instance import Instance
from os import path
from sys import exit


class ConfigParser:
    def __init__(self) -> None:
        self.config_file = None
        self.config_settings = None
        self.defaults = None
        self.existing_config_paths = {}

    def get_config(self) -> Dict:
        if self.config_file is not None:
            return self.config_file

        if not path.exists(MANAGER_CONFIG_PATH):
            error("Could not load config. No such file: {}".format(
                MANAGER_CONFIG_PATH))
            exit(1)

        with open(MANAGER_CONFIG_PATH, "r") as f:
            self.config_file = load(f, Loader=FullLoader)

        return self.config_file

    def get_config_settings(self) -> Dict:
        if self.config_settings is not None:
            return self.config_settings

        if not path.exists(MANAGER_CONFIG_SETTINGS_PATH):
            info("Could not load config fields. No such file: {}".format(
                MANAGER_CONFIG_SETTINGS_PATH))
            return {}

        with open(MANAGER_CONFIG_SETTINGS_PATH, "r") as f:
            self.config_settings = load(f, Loader=FullLoader)

        return self.config_settings

    def get_config_available_arguments(self) -> List[Dict]:
        return self.get_config_settings().get("arguments")

    def get_config_available_flags(self) -> List[Dict]:
        return self.get_config_settings().get("flags")

    def get_defaults(self) -> Dict:
        if self.defaults is None:
            self.defaults = self.parse_scoped_settings(
                self.get_config().get("defaults"))

        return self.defaults.copy()

    def get_narrow_config(self, scoped_config: Dict, source=None) -> Dict:
        if type(source) is not dict:
            source = self.get_defaults()

        scoped_settings = self.parse_scoped_settings(scoped_config)
        scoped_settings["flags"] = dict(source.get(
            "flags"), **scoped_settings.get("flags"))
        return dict(source, **scoped_settings)

    def get_instances(self) -> Dict[str, Instance]:
        result = {}
        for user in self.get_config().get("instances", []):
            result.update(self.parse_user(user))

        return result

    def parse_user(self, user_config: Dict) -> Dict[str, Instance]:
        if type(user_config) is not dict:
            return {}

        config = self.get_narrow_config(user_config)
        config["user"] = user_config.get("user")

        instances = {}
        symbols = user_config.get("symbols")
        instances.update(self.parse_symbols(symbols, config))
        return instances

    def parse_symbols(self, symbols: List, user_config: Dict) -> Dict[str, Instance]:
        symbols_type = type(symbols)
        if symbols_type is list:
            return self.parse_symbols_list(symbols, user_config)

        if symbols_type is dict:
            return self.parse_symbols_dict(symbols, user_config)

        return {}

    def parse_symbols_list(self, symbols: List, user_config: Dict) -> Dict[str, Instance]:
        instances = {}
        for symbol in symbols:
            config = user_config.copy()
            config["symbol"] = symbol
            instance = self.generate_instance(config)
            instances[instance.get_id()] = instance

        return instances

    def parse_symbols_dict(self, symbols: Dict, user_config: Dict) -> Dict[str, Instance]:
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

    def parse_scoped_settings(self, scoped_config: Dict) -> Dict:
        config = {
            "flags": {}
        }

        for arg in self.get_config_available_arguments():
            value = self.try_aliases(scoped_config, arg.get("aliases"))
            if value is not None:
                config[arg.get("name")] = value

        for flag in self.get_config_available_flags():
            value = self.try_aliases(scoped_config, flag.get("aliases"))
            if value is not None:
                config["flags"][flag.get("flag")] = value

        return config

    def try_aliases(self, config: Dict, aliases: List[str]):
        if type(aliases) is not list:
            return None

        for alias in aliases:
            if config.get(alias) is not None:
                return config.get(alias)

        return None

    def validate_path(self, file_path: str, absolute_prepend: str) -> Union[str, None]:
        if file_path is None or file_path == "":
            return None

        existing = self.existing_config_paths.get(file_path)
        if existing is not None:
            return existing

        full_config_path = path.join(absolute_prepend, file_path)
        check_in = {
            full_config_path: file_path,
            file_path: file_path
        }

        for full_path, partial_path in check_in.items():
            if path.exists(full_path):
                self.existing_config_paths[full_path] = full_path
                self.existing_config_paths[partial_path] = full_path
                return full_config_path

        return None

    def generate_instance(self, config: Dict) -> Instance:
        config = config.copy()

        full_config_path = self.validate_path(
            config.get("config"), CONFIGS_PATH)
        if full_config_path is None:
            error(
                "{}-{}: config does not exist".format(config.get("user"), config.get("symbol")))
        else:
            config["config"] = full_config_path

        full_api_keys_path = self.validate_path(
            config.get("flags").get("api_keys"), PASSIVBOT_PATH)
        if full_config_path is not None:
            config["api_keys"] = full_api_keys_path

        return Instance(config)
