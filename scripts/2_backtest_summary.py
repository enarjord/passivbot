
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


    args = parser.parse_args()

    return args

args = arguments_management()
number_coin_wanted = args.nb_best_coins

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

    if (closest_bkr < args.min_closest_bkr) :
        continue
    
    if (hrs_stuck_avg_long > args.max_stuck_avg) :
        continue
    
    if (hrs_stuck_max_long > args.max_stuck) :
        continue
    
    if (gain_pct < args.min_gain) :
        continue


    datas = {}
    datas['symbol']                 = symbol
    datas['n_days']                 = n_days
    datas['hrs_stuck_avg_long']     = hrs_stuck_avg_long
    datas['hrs_stuck_max_long']     = hrs_stuck_max_long
    datas['adg %']                  = adg_perct
    datas['gain %']                 = gain_pct
    datas['starting balance']       = starting_balance
    datas['closest bkr']            = closest_bkr
    

    # print(datas)
    datas_list.append(datas)

if len(datas_list) == 0:
    print("No results finded")
    exit()
else:
    print(len(datas_list), " coins after filtering")

df = pd.DataFrame(datas_list)
df.sort_values(by=['adg %', 'gain %'], ascending=False, inplace=True)
print(tabulate(df, headers='keys', tablefmt='psql'))
print('')
print('')
print("--------------------------------------------------------------")
print("Limited to the first ", number_coin_wanted, " coins found : ")
best_coin = df['symbol'].values[0:number_coin_wanted].tolist()
print("- coin list : ", best_coin)
print("- Sum adg % : ", df['adg %'].values[0:number_coin_wanted].sum())
print("- Sum gain % : ", df['gain %'].values[0:number_coin_wanted].sum())
print("--------------------------------------------------------------")
saving_data = "./tmp/best_coins.json"
print ("Saving the coin list to ", saving_data)
with open(saving_data, 'w') as outfile:
    json.dump(best_coin, outfile)