import datetime
import json
import sys
import types

if "ccxt" not in sys.modules:
    ccxt_module = types.ModuleType("ccxt")
    async_support = types.ModuleType("ccxt.async_support")
    async_support.exchanges = []
    setattr(ccxt_module, "async_support", async_support)
    sys.modules["ccxt"] = ccxt_module
    sys.modules["ccxt.async_support"] = async_support

if "dateutil" not in sys.modules:
    dateutil_module = types.ModuleType("dateutil")
    parser_module = types.ModuleType("dateutil.parser")

    def _simple_parse(value, default=None):
        value = value.strip()
        if not value:
            raise ValueError("empty date string")

        default = default or datetime.datetime(1900, 1, 1)
        try:
            dt = datetime.datetime.fromisoformat(value)
        except ValueError:
            if len(value) == 4 and value.isdigit():
                dt = default.replace(year=int(value))
            elif len(value) == 7 and value[4] == "-":
                dt = default.replace(year=int(value[:4]), month=int(value[5:7]))
            elif len(value) == 10 and value[4] == "-" and value[7] == "-":
                dt = default.replace(
                    year=int(value[:4]), month=int(value[5:7]), day=int(value[8:10])
                )
            else:
                raise
        if default.tzinfo is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=default.tzinfo)
        return dt

    parser_module.parse = _simple_parse
    dateutil_module.parser = parser_module
    sys.modules["dateutil"] = dateutil_module
    sys.modules["dateutil.parser"] = parser_module

if "hjson" not in sys.modules:
    hjson_module = types.ModuleType("hjson")
    hjson_module.load = lambda fp: json.load(fp)
    hjson_module.dumps = lambda obj: json.dumps(obj)
    sys.modules["hjson"] = hjson_module

import utils


def test_normalize_coins_source_splits_strings():
    result = utils.normalize_coins_source(" BTC ,ETH ")
    assert result == {"long": ["BTC", "ETH"], "short": ["BTC", "ETH"]}


def test_normalize_coins_source_loads_files(tmp_path):
    coins_file = tmp_path / "coins.hjson"
    coins_file.write_text('{"long": [" BTC ", "ETH"], "short": ["XRP"]}')

    source = {"long": str(coins_file), "short": ["ADA", ["DOT", "XLM"]]}
    result = utils.normalize_coins_source(source)

    assert result["long"] == ["BTC", "ETH"]
    assert result["short"] == ["ADA", "DOT", "XLM"]


def test_read_external_coins_lists_text_formats(tmp_path):
    text_file = tmp_path / "coins.txt"
    text_file.write_text('"BTC"\n"ETH"\n')

    result = utils.read_external_coins_lists(text_file)
    assert result == {"long": ["BTC", "ETH"], "short": ["BTC", "ETH"]}
