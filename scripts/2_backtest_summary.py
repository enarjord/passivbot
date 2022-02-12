
import glob
import os
import subprocess
import json
import pandas as pd
from tabulate import tabulate
import argparse
import os

def arguments_management():
    ### Parameters management
    parser = argparse.ArgumentParser( description="This script will read all the 'plots' folders from backtests and create a summary sorted by adg",
    epilog="",
    usage="python3 " + __file__ + " 11"
    )
    
    parser.add_argument("nb_best_coins", type=int, help="Number of coin wanted")

    parser.add_argument("-mbkr","--min-closest-bkr",
                        type=float,required=False,dest="min_closest_bkr",default=0,
                        help="Show only result upper than min_closest_bkr",
    )

    args = parser.parse_args()

    return args

args = arguments_management()
number_coin_wanted = args.nb_best_coins

print("List all files availables")
files = glob.glob('backtests/*/*/plots/*/result.json')

if len(files) == 0:
    print('No files found')
    exit()

datas_list = []
for file in files:
    print(file)
    f = open(file)
    bt = json.load(f)
    f.close()

    datas = {}

    closest_bkr = 0
    if ('closest_bkr' in bt['result']) :
        closest_bkr            = bt['result']['closest_bkr']
    if ('closest_bkr_long' in bt['result']) :
        closest_bkr            = bt['result']['closest_bkr_long']

    if (closest_bkr < args.min_closest_bkr) :
        continue
    

    datas['symbol']                 = bt['result']['symbol']
    datas['n_days']                 = bt['result']['n_days']
    datas['hrs_stuck_avg_long']     = bt['result']['hrs_stuck_avg_long']
    datas['hrs_stuck_max_long']     = bt['result']['hrs_stuck_max_long']

    if ('average_daily_gain' in bt['result']) :
        datas['adg %']                  = bt['result']['average_daily_gain']*100
    if ('adg_long' in bt['result']) :
        datas['adg %']                  = bt['result']['adg_long']*100
    
    if ('gain' in bt['result']) :
        datas['gain %']                 = bt['result']['gain']*100
    if ('gain_long' in bt['result']) :
        datas['gain %']                 = bt['result']['gain_long']*100

    datas['starting balance']       = bt['result']['starting_balance']
    
    datas['closest bkr']            = closest_bkr
    

    # print(datas)
    datas_list.append(datas)

if len(datas_list) == 0:
    print("No results finded")
    exit()
    
df = pd.DataFrame(datas_list)
df.sort_values(by=['adg %', 'gain %'], ascending=False, inplace=True)

print(tabulate(df, headers='keys', tablefmt='psql'))
print("The best ", number_coin_wanted, " coins found ")
best_coin = df['symbol'].values[0:number_coin_wanted].tolist()
print(best_coin)

saving_data = "./tmp/best_coins.json"
print ("Saving list to ", saving_data)
with open(saving_data, 'w') as outfile:
    json.dump(best_coin, outfile)