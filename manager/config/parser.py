from constants import MANAGER_CONFIG_PATH, MANAGER_CONFIG_SETTINGS_PATH, logger
from yaml import load, FullLoader, YAMLError
from .v2 import ConfigParserV2
from .v1 import ConfigParserV1
from instance import Instance
from typing import Dict
from sys import exit
from os import path


class ConfigParser:
    def __init__(self, config_path: str = None) -> None:
        self.config = None
        self.config_settings = None
        self.config_path = MANAGER_CONFIG_PATH
        if config_path is not None:
            self.config_path = config_path

    def preload(self):
        self.get_config()
        self.get_config_settings()

    def load_yaml(self, filepath: str) -> Dict:
        if not path.exists(filepath):
            logger.error(f"Path does not exist: {filepath}")
            exit(1)

        try:
            with open(filepath, "r") as f:
                yaml = load(f, Loader=FullLoader)
        except YAMLError as error:
            logger.error("Error while parsing YAML file:")

            if not hasattr(error, 'problem_mark'):
                exit(1)

            logger.error(f"  {str(error.problem_mark)}")

            if error.context is not None:
                logger.error(f"  {str(error.problem)} {str(error.context)}")

            logger.error(f"  {str(error.problem)}")
            exit(1)

        return yaml

    def get_config(self) -> Dict:
        if self.config is None:
            self.config = self.load_yaml(self.config_path)

        return self.config

    def get_config_settings(self) -> Dict:
        if self.config_settings is None:
            self.config_settings = self.load_yaml(MANAGER_CONFIG_SETTINGS_PATH)

        return self.config_settings

    def get_instances(self) -> Dict[str, Instance]:
        config = self.get_config()
        settings = self.get_config_settings()
        version = config.get("version")

        parser = ConfigParserV1
        if version == 2:
            parser = ConfigParserV2

        return parser(config, settings).get_instances()
