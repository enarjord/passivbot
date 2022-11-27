from constants import MANAGER_CONFIG_PATH, logger
from yaml import load, FullLoader, YAMLError
from .v2 import ConfigParserV2
from .v1 import ConfigParserV1
from instance import Instance
from logging import error
from typing import Dict
from sys import exit
from os import path


class ConfigParser:
    def __init__(self) -> None:
        self.config = None

    def get_config(self) -> Dict:
        if self.config is not None:
            return self.config

        if not path.exists(MANAGER_CONFIG_PATH):
            error("Could not load config. No such file: {}".format(
                MANAGER_CONFIG_PATH))
            exit(1)

        try:
            with open(MANAGER_CONFIG_PATH, "r") as f:
                self.config = load(f, Loader=FullLoader)
        except YAMLError as error:
            logger.error("Error while parsing YAML file:")
            logger.error("  {}".format(str(error.problem_mark)))

            if not hasattr(error, 'problem_mark'):
                exit(1)

            context = ""
            if error.context is not None:
                context = error.context
            logger.error("  {} {}".format(str(error.problem), context))
            exit(1)

        return self.config

    def get_instances(self) -> Dict[str, Instance]:
        config = self.get_config()
        version = config.get("version")
        parser = None
        if version == 2:
            parser = ConfigParserV2(config)
        else:
            parser = ConfigParserV1(config)

        return parser.get_instances()
