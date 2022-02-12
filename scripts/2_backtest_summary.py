
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
    args = parser.parse_args()

    return args

args = arguments_management()
number_coin_wanted = args.nb_best_coins

print("List all files availables")
files = glob.glob('backtests/*/*/plots/*/result.json')

if len(files) == 0:
    print('No files finded')
    exit()

datas_list = []
for file in files:
    print(file)
    f = open(file)
    bt = json.load(f)
    f.close()

    datas = {}
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



    # print(datas)
    datas_list.append(datas)


df = pd.DataFrame(datas_list)
df.sort_values(by=['adg %', 'gain %'], ascending=False, inplace=True)

print(tabulate(df, headers='keys', tablefmt='psql'))
print("The best ", number_coin_wanted, " coins finded ")
best_coin = df['symbol'].values[0:number_coin_wanted].tolist()
print(best_coin)

saving_data = "./tmp/best_coins.json"
print ("Saving list to ", saving_data)
with open(saving_data, 'w') as outfile:
    json.dump(best_coin, outfile)