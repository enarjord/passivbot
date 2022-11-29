from constants import CONFIGS_PATH, PASSIVBOT_PATH, logger
from typing import Dict, List, Union
from instance import Instance
from os import path


class ConfigParserVersion:
    def __init__(self, config: Dict, settings: Dict) -> None:
        self.config = config
        self.config_settings = settings
        self.defaults = None
        self.system_paths_cache = {}

    def get_config(self) -> Dict:
        return self.config

    def get_config_settings(self) -> Dict:
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

    def get_instances(self) -> Dict[str, Instance]:
        return {}

    def narrow_config(self, scoped_config: Dict, source=None) -> Dict:
        if type(source) is not dict:
            source = self.get_defaults()

        scoped_settings = self.parse_scoped_settings(scoped_config)
        scoped_settings["flags"] = dict(source.get(
            "flags"), **scoped_settings.get("flags"))
        return dict(source, **scoped_settings)

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

        existing = self.system_paths_cache.get(file_path)
        if existing is not None:
            return existing

        full_config_path = path.join(absolute_prepend, file_path)
        check_in = {
            full_config_path: file_path,
            file_path: file_path
        }

        for full_path, partial_path in check_in.items():
            if path.exists(full_path):
                self.system_paths_cache[full_path] = full_path
                self.system_paths_cache[partial_path] = full_path
                return full_config_path

        return None

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

    def generate_instance(self, config: Dict) -> Instance:
        config = config.copy()

        full_config_path = self.validate_path(
            config.get("config"), CONFIGS_PATH)
        if full_config_path is None:
            logger.error(
                f"{config.get('user')}-{config.get('symbol')}: config does not exist")
        else:
            config["config"] = full_config_path

        full_api_keys_path = self.validate_path(
            config.get("flags").get("api_keys"), PASSIVBOT_PATH)
        if full_config_path is not None:
            config["api_keys"] = full_api_keys_path

        return Instance(config)
