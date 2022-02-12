
import argparse
import os
from ast import arg
import hjson
import json
from pybit import HTTP


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

    input_datas = {
                'min_volume_24h'        : args.min_volume_24h,
                'min_turnover_24h'      : args.min_turnover_24h,
                'starting_balance'      : backtest_config['starting_balance'],
                'wallet_exposure_limit' : live_config['long']['wallet_exposure_limit'],
                'initial_qty_pct'       : live_config['long']['initial_qty_pct'],
    }

    print("-----------------")
    print("- Settings used -")
    print("-----------------")
    for key, value in input_datas.items():
        print(key.ljust(25), ' : ', value)

    return input_datas

def find_grid_ok(input_datas):
    print("-----------------------------------------")
    print("- Loading Coins informations From ByBit -")
    print("-----------------------------------------")
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
bash_symbols = find_grid_ok(input_datas)

print ("found ", len(bash_symbols), " symbols.")
print ("Full Wallet exposure with all this symbols is : ", len(bash_symbols) * input_datas['wallet_exposure_limit'])

saving_data = "./tmp/grid_ok_coins.json"
print ("Saving list to ", saving_data)
with open(saving_data, 'w') as outfile:
    json.dump(bash_symbols, outfile)