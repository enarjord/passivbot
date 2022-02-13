import argparse
import os
from pickle import FALSE, TRUE
import hjson
import subprocess
import glob
import shutil

def arguments_management():
    ### Parameters management
    parser = argparse.ArgumentParser( description="This script will loop and generate backtests on a list of coins",
    epilog="",
    usage="python3 " + __file__ + " -jf tmp/grid_ok_coins.json  ../configs/live/a_tedy.json ../configs/backtest/default.hjson"
    )

    parser.add_argument("live_config_filepath", type=str, help="file path to live config")
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

    if not os.path.exists(args.live_config_filepath) :
        print("live_config_path doesn't exist")
        exit()

    if not os.path.exists(args.backtest_config_filepath) :
        print("backtest_config_path doesn't exist")
        exit()


    args.live_config_filepath       = os.path.realpath(args.live_config_filepath)
    args.backtest_config_filepath   = os.path.realpath(args.backtest_config_filepath)

    if not os.path.exists(args.live_config_filepath) :
        print("live_config_path doesn't exist")
        exit()

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

def backtest_looping(args, backtest_directory) :
    nb_coin=len(args.builded_coin_list)   
    backtest_command_line = [
                                "python3", "backtest.py", 
                                "-s", "#SYMBOL_NAME#",
                                "-bd", backtest_directory,  
                                "-b", args.backtest_config_filepath,
                                args.live_config_filepath
    ]

    if not os.path.exists("../"+backtest_directory) :
        print("Sorry, you must create this directory : ", os.path.realpath("../"+backtest_directory))
        exit()



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

        try:
            subprocess.run(final_command_line, cwd="..")
        except subprocess.TimeoutExpired:
            print('Timeout Reached (', freqtrade_timeout_seconds, ' seconds)')
        current_i = current_i + 1

def clean_backtest_directories(backtest_directory) :
    glob_delete = "../" + backtest_directory + "/*/*/plots"
    list = glob.glob(glob_delete, recursive=True)
    for to_delete in list :
        print('Cleaning directory : ', os.path.realpath(to_delete))
        shutil.rmtree(to_delete)

backtest_directory = "./scripts/backtests/"
args = arguments_management()
clean_backtest_directories(backtest_directory)
backtest_looping(args, backtest_directory)
