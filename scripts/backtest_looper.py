import argparse
import os
from pickle import FALSE, TRUE
import json
import hjson
import subprocess


def arguments_management():
    ### Parameters management
    parser = argparse.ArgumentParser( description="This script will loop and generate backtests on a list of coins",
    epilog=""
    )
    parser.add_argument("backtest_config_filepath", type=str, help="file path to backtest")

    parser.add_argument("-jf","--json-file-coin-list",
                        type=str,required=False,dest="json_file_coin_list",default="",
                        help="A json file containing a coin list array, ex : tmp/grid_ok_coins.json",
    )

    parser.add_argument("-cl","--coin_list",
                        type=str,required=False,dest="coin_list",default="",
                        help="A list of coin separated by space, ex : 'ONEUSDT XLMUSDT'",
    )

    args = parser.parse_args()

    if not os.path.exists(args.backtest_config_filepath) :
        print("backtest_config_path doesn't exist")
        exit()

    args.builded_coin_list = []
    if (len(args.coin_list.strip().split(' ')) > 0) :
       args.builded_coin_list = args.coin_list.strip().split(' ')
       if args.builded_coin_list[0] == '' :
           args.builded_coin_list.pop(0)

    if os.path.exists(args.json_file_coin_list) :
        args.builded_coin_list = hjson.load(open(args.json_file_coin_list, encoding="utf-8"))

    if len(args.builded_coin_list) == 0 :
        print('No coin finded with the program arguments. See arguments -jf or -cl')
        exit()

    return args

def backtest_looping(args) :
    nb_coin=len(args.builded_coin_list)
    backtest_command_line = [
                                "python3", "backtest.py", "-s", "#SYMBOL_NAME#",  args.backtest_config_filepath,
                                "-bd", "./tedy_scripts/backtests/"
    ]

    # @TODO : idealement bosser dans un repertoire pour ce script 
    # @TODO : supprimer les plots en amont 

    print("Coin list : ", args.builded_coin_list)
    print("Number of coins : ", nb_coin)

    # backtest_command_line = [
    #     "python3", "backtest.py", "-u", user_connexion, "-s", "#SYMBOL_NAME#", "--starting_balance="+str(starting_balance), "-sd"
    #     , start_date, "-ed", end_date, config_file
    #     ]
    print("Command executed is : ", print(' '.join(backtest_command_line)))
    current_i = 0
    for current_symbol in args.builded_coin_list:
        print('BackTesting ', current_symbol, ' (', (current_i+1),'/', nb_coin,')')
        freqtrade_timeout_seconds = 10 * 60 * 60 # set timeout of backtesting to 10 minutes maximum 10 * 60 * 60
        final_command_line = []
        for element in backtest_command_line:
            final_command_line.append(element.replace("#SYMBOL_NAME#", current_symbol))

        # run FreqTrade
        try:
            subprocess.run(final_command_line, cwd="..")
        except subprocess.TimeoutExpired:
            print('Timeout Reached (', freqtrade_timeout_seconds, ' seconds)')
        current_i = current_i + 1

args = arguments_management()
backtest_looping(args)
