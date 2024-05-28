from .v import ConfigParserVersion
from instance import Instance
from typing import Dict, List


class ConfigParserV1(ConfigParserVersion):
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
        config = self.make_backwards_compatible(config)
        config["user"] = user_config.get("user")

        instances = {}
        symbols = user_config.get("symbols")
        instances.update(self.parse_symbols(symbols, config))
        return instances

    def parse_symbols(self, symbols: List, user_config: Dict) -> Dict[str, Instance]:
        instances = {}
        for symbol in symbols:
            config = user_config.copy()
            config["symbol"] = symbol
            instance = self.generate_instance(config)
            instances[instance.get_id()] = instance

        return instances

    def make_backwards_compatible(self, config: Dict) -> Dict:
        def not_zero(v): return v > 0.0
        rules = {
            "-m": lambda v:  v != "futures",
            "-lm": lambda v: v != "n",
            "-sm": lambda v: v != "m",
            "-lw": not_zero,
            "-sw": not_zero,
            "-ab": not_zero,
            "-lmm": not_zero,
            "-lmr": not_zero,
            "-smm": not_zero,
            "-smr": not_zero,
        }

        flags = {}
        for key, value in config.get("flags").items():
            rule = rules.get(key)
            if rule is not None and rule(value):
                flags[key] = value

        config["flags"] = flags

        return config
