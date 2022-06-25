
import glob
import os
import subprocess
import json
import pandas as pd
from tabulate import tabulate
import argparse
import os
import hjson
import requests


import sys

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
    parser = argparse.ArgumentParser( description="This script will read all the 'plots' folders from backtests and create a summary sorted by adg",
    epilog="",
    usage="python3 " + __file__ + " 11 ../configs/live/a_tedy.json -max-stuck-avg 7 -max-stuck 200  -min-gain 10"
    )
    
    parser.add_argument("nb_best_coins", type=int, help="Number of coin wanted")
    parser.add_argument("live_config_filepath", type=str, help="file path to live config")
    parser.add_argument("backtest_config_filepath", type=str, help="file path to backtest")

    parser.add_argument("-min-bkr","--min-closest-bkr",
                        type=float,required=False,dest="min_closest_bkr",default=0,
                        help="Show only result upper than min_closest_bkr",
    )

    parser.add_argument("-max-stuck-avg","--max-hrs-stuck-avg-long",
                        type=float,required=False,dest="max_stuck_avg",default=999999,
                        help="Show only result lower than max_stuck_avg",
    )

    parser.add_argument("-max-stuck","--max-hrs-stuck-long",
                        type=float,required=False,dest="max_stuck",default=999999,
                        help="Show only result lower than max_stuck",
    )

    parser.add_argument("-min-gain","--min-gain",
                        type=float,required=False,dest="min_gain",default=0,
                        help="Show only result lower than max_stuck",
    )

    parser.add_argument("-min-days","--min-days",
                        type=float,required=False,dest="min_days",default=0,
                        help="Show only result upper than min-days",
    )

    parser.add_argument("-max-marketcap-pos","--max-marketcap-pos",
                        type=float,required=False,dest="max_marketcap_pos",default=0,
                        help="Max marketcap position",
    )


    args = parser.parse_args()

    args.live_config_filepath       = os.path.realpath(args.live_config_filepath)


    if not os.path.exists(args.live_config_filepath) :
        print("live_config_path doesn't exist")
        exit()

    live_config = hjson.load(open(args.live_config_filepath, encoding="utf-8"))
    args.wallet_exposure_limit = live_config['long']['wallet_exposure_limit']

    args.backtest_config_filepath   = os.path.realpath(args.backtest_config_filepath)

    if not os.path.exists(args.backtest_config_filepath) :
        print("backtest_config_path doesn't exist")
        exit()

    backtest_config = hjson.load(open(args.backtest_config_filepath, encoding="utf-8"))

    args.starting_balance = backtest_config['starting_balance']

    return args

args = arguments_management()
number_coin_wanted = args.nb_best_coins

print('python3 ' + __file__ + (" ").join(sys.argv[1:]))

# find the market cap
API_KEY = open('config/api.coinmarketcap.com.conf', 'r').read()

headers = {
    'X-CMC_PRO_API_KEY': API_KEY
}

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
r = requests.get(url, headers=headers)
find_ranks = {}
if r.status_code == 200:
    data = r.json()
    print(data)
    for d in data['data']:
        symbol = d['symbol']
        find_ranks[symbol] = d['cmc_rank']




# Grab all files availables
files = glob.glob('backtests/*/*/plots/*/result.json')

if len(files) == 0:
    print('No files found')
    exit()
else:
    print('Reading ', len(files), ' backtests')

datas_list = []
for file in files:
    # print(file)
    f = open(file)
    bt = json.load(f)
    f.close()

    symbol              = bt['result']['symbol']
    n_days              = bt['result']['n_days']
    hrs_stuck_avg_long  = bt['result']['hrs_stuck_avg_long']
    hrs_stuck_max_long  = bt['result']['hrs_stuck_max_long']
    adg_perct           = (bt['result']['average_daily_gain']*100) if ('average_daily_gain' in bt['result']) else bt['result']['adg_long']*100
    gain_pct            = (bt['result']['gain']*100)  if ('gain' in bt['result']) else  bt['result']['gain_long']*100
    starting_balance    = bt['result']['starting_balance']
    closest_bkr         = (bt['result']['closest_bkr']) if ('closest_bkr' in bt['result']) else (bt['result']['closest_bkr_long'])

    gain_dollard = bt['result']['final_balance_long'] -  bt['result']['starting_balance']

    marketcap_symbol = symbol.upper().replace('USDT', '')


    marketcapPosition = find_ranks[marketcap_symbol] if (marketcap_symbol in find_ranks) else 999

    if (closest_bkr < args.min_closest_bkr) :
        continue
    
    if (hrs_stuck_avg_long > args.max_stuck_avg) :
        continue
    
    if (hrs_stuck_max_long > args.max_stuck) :
        continue
    
    if (gain_pct < args.min_gain) :
        continue

    if (n_days < args.min_days) :
        continue

    if (marketcapPosition > args.max_marketcap_pos) :
        continue


 
    # print('BTC', find_ranks.get('BTC'))
    # print('TRX', find_ranks.get('TRX'))






    datas = {}
    datas['symbol']                 = symbol
    datas['n_days']                 = n_days
    datas['hrs_stuck_avg_long']     = hrs_stuck_avg_long
    datas['hrs_stuck_max_long']     = hrs_stuck_max_long
    datas['adg %']                  = adg_perct
    datas['gain %']                 = gain_pct
    datas['total gain $']                 = gain_dollard
    datas['starting balance']       = starting_balance
    datas['closest bkr']            = closest_bkr
    datas['marketcapPosition']            = marketcapPosition
    
    

    # print(datas)
    datas_list.append(datas)

if len(datas_list) == 0:
    print("No results finded")
    exit()
else:
    print(len(datas_list), " coins after filtering")

df = pd.DataFrame(datas_list)
df.sort_values(by=['marketcapPosition', 'adg %', 'gain %'], ascending=[True, False, False], inplace=True)
best_coin = df['symbol'].values[0:number_coin_wanted].tolist()
total_wallet_exposure = args.wallet_exposure_limit * len(best_coin)
print(tabulate(df, headers='keys', tablefmt='psql'))
print('')
print('')
print("--------------------------------------------------------------")
print("Limited to the first ", number_coin_wanted, " coins found : ")

print("- Total wallet_exposure_limit : ", total_wallet_exposure)

print("- coin list : ", best_coin)

# adg_pct                 = (df['adg %'].values[0:number_coin_wanted].mean() * total_wallet_exposure)
adg_pct                 = (df['adg %'].values[0:number_coin_wanted].sum())
print("- global adg % : ", (adg_pct), "%")

adg_dollard             = adg_pct * args.starting_balance / 100
print("- global adg $ : ", (adg_dollard), "$" )

print("- global adg 1 month (30 days) : ", int(adg_dollard * 30) , "$" )

# global_gain_pct         = (df['gain %'].values[0:number_coin_wanted].mean() * total_wallet_exposure)
global_gain_pct         = (df['gain %'].values[0:number_coin_wanted].sum())
print("- global gain % : ", int(global_gain_pct), "%")

global_gain_dollard     = global_gain_pct * args.starting_balance / 100
print("- global gain $ : ", int(global_gain_dollard), "$")

print("--------------------------------------------------------------")
saving_data = "./tmp/best_coins.json"
print ("Saving the coin list to ", saving_data)
with open(saving_data, 'w') as outfile:
    json.dump(best_coin, outfile)


print('python3 ' + __file__ + (" ").join(sys.argv[1:]))
