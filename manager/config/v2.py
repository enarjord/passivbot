from .v import ConfigParserVersion
from typing import Dict, List
from instance import Instance


class ConfigParserV2(ConfigParserVersion):
    def __init__(self, config: Dict, settings: Dict) -> None:
        super().__init__(config, settings)

    def get_instances(self) -> Dict[str, Instance]:
        result = {}
        for user in self.get_config().get("instances", []):
            result.update(self.parse_user(user))

        return result

    def parse_user(self, user_config: Dict) -> Dict[str, Instance]:
        if type(user_config) is not dict:
            return {}

        config = self.narrow_config(user_config)
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
                config.update(self.narrow_config(scoped_config, config))

            config["symbol"] = symbol
            instance = self.generate_instance(config)
            instances[instance.get_id()] = instance

        return instances
