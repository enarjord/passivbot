import datetime
import json
from time import time

import hjson


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def print_(args, r=False, n=False):
    line = ts_to_date(time())[:19] + '  '
    str_args = '{} ' * len(args)
    line += str_args.format(*args)
    if n:
        print('\n' + line, end=' ')
    elif r:
        print('\r' + line, end=' ')
    else:
        print(line)
    return line


def load_key_secret(exchange: str, user: str) -> (str, str):
    try:
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:
            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]
            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
        raise Exception('API KeyFile Missing!')
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception('API KeyFile Missing!')


def load_config(path: str) -> dict:
    try:
        config = hjson.load(open(path))
        return config
    except Exception as e:
        print('Could not read config')
        print(e)
        return {}

def add_or_append(dict, key, value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]
    return dict
