import json
import sys
from importlib.util import spec_from_loader, module_from_spec
from pathlib import Path

import hjson

from bots.configs import LiveConfig
from helpers.print_functions import print_


def remove_numba_decorators(text: str) -> str:
    """
    Removes the decorators associated with numba so that a clean file remains.
    :param text: The text (script) to clean.
    :return: Cleaned text.
    """
    match = '@jitclass('
    index = text.find(match)
    while index != -1:
        open_brackets = 1
        start_index = index + len(match)
        end_index = 0
        for i in range(len(text[start_index:])):
            if text[start_index:][i] == '(':
                open_brackets += 1
            if text[start_index:][i] == ')':
                open_brackets -= 1
            if open_brackets == 0:
                end_index = i
                break
        text = text.replace(text[index:index + len(match) + end_index + 1], '')
        index = text.find(match)
    text = text.replace('@njit', '')
    return text


def load_module_from_file(file_name: str, module_name: str, to_be_replaced: tuple = (None, None),
                          insert_at_start: str = ''):
    """
    Loads a module directly from a file. In the case of a script with numba compatibility, it strips it first, imports
    the clean module, and then imports the module with numba enabled.
    :param file_name: The filename to import.
    :param module_name: The name of the new module.
    :return: A module.
    """
    text = Path(file_name).read_text()
    original_text = text
    if insert_at_start:
        original_text = insert_at_start + '\n' + original_text
    if to_be_replaced[0]:
        original_text = original_text.replace(to_be_replaced[0], to_be_replaced[1])
    numba_free_text = remove_numba_decorators(text)
    module_spec = spec_from_loader(module_name, loader=None)
    module = module_from_spec(module_spec)
    exec(numba_free_text, module.__dict__)
    sys.modules[module_name] = module
    exec(original_text, module.__dict__)
    sys.modules[module_name] = module
    return module


def get_strategy_definition(filename: str) -> str:
    """
    Finds the strategy definition inside a strategy script and extracts it so that it can be inserted in the backtest
    bot file before loading it.
    :param filename: The filename to check.
    :return: A string containing the strategy definition.
    """
    text = Path(filename).read_text()
    match = 'strategy_definition'
    index = text.find(match)
    index = index + len(match)
    start_index = 0
    for i in range(len(text[index:])):
        if text[index:][i].isalnum():
            start_index = i
            break
    index = index + start_index
    open_brackets = 0
    first_bracket = False
    end_index = 0
    for i in range(len(text[index:])):
        if text[index:][i] == '(':
            open_brackets += 1
            if not first_bracket:
                first_bracket = True
        if text[index:][i] == ')':
            open_brackets -= 1
        if open_brackets == 0 and first_bracket:
            end_index = i
            break
    return text[index:index + end_index + 1]


def load_key_secret(exchange: str, user: str) -> (str, str):
    """
    Loads the key and secret from the API key file.
    :param exchange: The exchange to use as a key.
    :param user: The user to use as a key.
    :return: The key and secret.
    """
    try:
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:
            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]
            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print_(["Looks like the keys aren't configured yet, or you entered the wrong username!"])
        raise Exception('API KeyFile Missing!')
    except FileNotFoundError:
        print_(["File not found!"])
        raise Exception('API KeyFile Missing!')


def load_base_config(path: str) -> dict:
    """
    Loads the base config from an hjson file.
    :param path: The path to the config.
    :return: The config as a dictionary.
    """
    try:
        config = hjson.load(open(path))
        return config
    except Exception as e:
        print_(["Could not read config", e])
        return {}


async def load_exchange_settings(exchange: str, symbol: str, user: str, market_type: str) -> dict:
    """
    Loads exchange several exchange settings.
    :param exchange: The exchange to fetch for.
    :param symbol: The symbol to fetch for.
    :param user: The user to use.
    :param market_type: The market type to use.
    :return: A dictionary with exchange settings.
    """
    settings = {}
    config = LiveConfig(symbol.upper(), user, exchange, market_type, 1, 1.0, 0.0, 0.0)
    if exchange == 'binance':
        from bots.binance import BinanceBot
        bot = BinanceBot(config, None)
        # if 'spot' in config['market_type']:
        #     settings['maker_fee'] = 0.001
        #     settings['taker_fee'] = 0.001
        #     settings['spot'] = True
        #     settings['hedge_mode'] = False
        # else:
        settings['maker_fee'] = 0.0002
        settings['taker_fee'] = 0.0004
        # settings['spot'] = False
    # elif exchange == 'bybit':
    #     # from bots.bybit import BybitBot
    #     # bot = BybitBot(config, None)
    #     # if 'spot' in config['market_type']:
    #     #     raise Exception('spot not implemented on bybit')
    #     settings['maker_fee'] = -0.00025
    #     settings['taker_fee'] = 0.00075
    else:
        raise Exception(f'Unknown exchange {exchange}')
    await bot.market_type_init()
    await bot.fetch_exchange_info()
    if 'inverse' in bot.market_type:
        settings['inverse'] = True
    elif any(x in bot.market_type for x in ['linear', 'spot']):
        settings['inverse'] = False
    else:
        raise Exception('Unknown market type')
    for key in ['max_leverage', 'minimal_quantity', 'minimal_cost', 'quantity_step', 'price_step', 'max_leverage',
                'contract_multiplier']:  # , 'hedge_mode']:
        settings[key] = getattr(bot, key)
    try:
        await bot.session.close()
    except:
        pass
    return settings
