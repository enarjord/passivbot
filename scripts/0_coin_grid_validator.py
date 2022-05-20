
import argparse
import os
from ast import arg
import hjson
import json
import sys
from pybit import HTTP
#binance
import requests

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("tmp/" + __file__ +".log", "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

def arguments_management():
    ### Parameters management
    parser = argparse.ArgumentParser( description="This script will list the coins working with the grid settings",
    usage="python3 " + __file__ + " ../configs/live/a_tedy.json ../configs/backtest/default.hjson -mv24 0 -mt24 0",
    epilog="This script will use starting_balance, initial_qty_pct, wallet_exposure_limit to find coin working with the grid."
    )
    parser.add_argument("live_config_filepath", type=str, help="file path to live config")
    parser.add_argument("backtest_config_filepath", type=str, help="file path to backtest")

    parser.add_argument("-mv24","--min-volume-24h",
                        type=int,required=False,dest="min_volume_24h",default=0,
                        help="specify minimum volume 24h wanted",
    )

    parser.add_argument("-mt24","--min-turnover-24h",
                        type=int,required=False,dest="min_turnover_24h",default=0,
                        help="specify minimum turnover 24h wanted",
    )

    parser.add_argument("-we","--force-wallet-exposure",
                        type=float,required=False,dest="force_wallet_exposure",default=0,
                        help="Force the wallet exposure ratio",
    )

    args = parser.parse_args()

    if not os.path.exists(args.live_config_filepath) :
        print("live_config_path doesn't exist")
        exit()

    if not os.path.exists(args.backtest_config_filepath) :
        print("backtest_config_path doesn't exist")
        exit()

    return args


def get_config_data(args):

    backtest_config = hjson.load(open(args.backtest_config_filepath, encoding="utf-8"))
    live_config = hjson.load(open(args.live_config_filepath, encoding="utf-8"))
    api_key = hjson.load(open("./../api-keys.json", encoding="utf-8"))

    platform = ''
    try:
        platform = api_key[backtest_config['user']]['exchange']
    except Exception as e:
            print('Problem reading the user', backtest_config['user'] ,'in the api-keys.json')
            print(f'Error: {e}')
            exit()

    input_datas = {
                'min_volume_24h'        : args.min_volume_24h,
                'min_turnover_24h'      : args.min_turnover_24h,
                'starting_balance'      : backtest_config['starting_balance'],
                'wallet_exposure_limit' : args.force_wallet_exposure if not (args.force_wallet_exposure == 0) else live_config['long']['wallet_exposure_limit'],
                'initial_qty_pct'       : live_config['long']['initial_qty_pct'],
                'user'                  : backtest_config['user'],
                'platform'              : platform,
    }

    print("-----------------")
    print("- Settings used -")
    print("-----------------")
    for key, value in input_datas.items():
        print(key.ljust(25), ' : ', value)

    return input_datas


def binance_find_frid_ok(input_datas):
    #dont touch these
    iec = input_datas['starting_balance'] * input_datas['wallet_exposure_limit'] * input_datas['initial_qty_pct']

    data_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    data = requests.get(data_url).json()
    prices_url = "https://fapi.binance.com/fapi/v1/ticker/price"
    pricedata = requests.get(prices_url).json()
    
    bash_symbols=[]
    try: 
        for datas in data["symbols"]:
            if "USDT" in datas["pair"]:
                # print(datas)
                symbol = datas["pair"]
                min_qty = float(datas["filters"][1]["minQty"] )     
                min_notional_fixed =  float(datas["filters"][5]["notional"])

                for i in pricedata:
                    if i["symbol"] == symbol:
                        price = i["price"]
                        # print(i)

                min_notional_calc = min_qty * float(price)

                if min_notional_calc <= min_notional_fixed:
                    min_notional = min_notional_fixed
                else:
                    min_notional = min_notional_calc

                if iec >= min_notional:
                    print(f"{symbol.ljust(13)}\t min_notional: ${min_notional:.2f}\t GRID OK")
                    bash_symbols.append(symbol)
                else:
                    print(f"{symbol.ljust(13)}\t min_notional: ${min_notional:.2f}\t GRID --KO")
                    
    except Exception as e:
        print(f'Error: {e}')
        

    return bash_symbols


def bybit_find_grid_ok(input_datas):
    #Initial Entry Cost:
    session = HTTP("https://api.bybit.com")
    res = session.query_symbol() #gives all symbols, cannot filter
    initial_entry_cost = input_datas['starting_balance'] * input_datas['wallet_exposure_limit'] * input_datas['initial_qty_pct']
    config_symbol = ''
    bash_symbols=[]

    for coin in res['result']:
        if 'USDT' in coin['name']:
            config_symbol = coin['name']

        min_qty = float(coin['lot_size_filter']['min_trading_qty'])
        symboldata = session.latest_information_for_symbol(symbol=config_symbol)
        try:
            min_notional = min_qty * float(symboldata['result'][0]['last_price'])
        except:
            continue
        volume_24h = float(symboldata['result'][0]['volume_24h'])
        turnover_24h = float(symboldata['result'][0]['turnover_24h'])

        if (
            (turnover_24h >= input_datas['min_turnover_24h']) and 
            (volume_24h >= input_datas['min_volume_24h']) and
            (initial_entry_cost >= min_notional)
            ) :
            print (f"{config_symbol.ljust(15)} min_notional: {min_notional:.2f} GRID OK [volume_24h : {volume_24h}, turnover_24 : {turnover_24h}]")
            bash_symbols.append(config_symbol)

    return bash_symbols

args = arguments_management()
input_datas = get_config_data(args)

print("-----------------------------------------")
print("- Loading Coins informations From " + input_datas['platform'] + " -")
print("-----------------------------------------")

bash_symbols = []

if (input_datas['platform'] == 'bybit'):
    bash_symbols = bybit_find_grid_ok(input_datas)

if (input_datas['platform'] == 'binance'):
    if input_datas['min_turnover_24h'] > 0:
        print('Sorry the turnover 24h filter is not available on Binance')
        exit()
    if input_datas['min_volume_24h'] > 0:
        print('Sorry the volume 24h filter is not available on Binance')
        exit()
    bash_symbols = binance_find_frid_ok(input_datas)

print ("found ", len(bash_symbols), " symbols OK with the grid.")
print ("Full Wallet exposure with all this symbols is : ", len(bash_symbols) * input_datas['wallet_exposure_limit'])

saving_data = "./tmp/grid_ok_coins.json"
print ("Saving list to ", saving_data)
with open(saving_data, 'w') as outfile:
    json.dump(bash_symbols, outfile)