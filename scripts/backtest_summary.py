
import glob
import os
import subprocess
import json
import pandas as pd
from tabulate import tabulate

number_coin_wanted = 11

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
    datas['adg %']                  = bt['result']['adg_long']*100
    datas['hrs_stuck_avg_long']     = bt['result']['hrs_stuck_avg_long']
    datas['hrs_stuck_max_long']     = bt['result']['hrs_stuck_max_long']
    datas['gain %']                 = bt['result']['gain_long']*100

    # print(datas)
    datas_list.append(datas)


df = pd.DataFrame(datas_list)
df.sort_values(by=['adg %', 'gain %'], ascending=False, inplace=True)

print(tabulate(df, headers='keys', tablefmt='psql'))
print("The best ", number_coin_wanted, " coins finded ")
print(   df['symbol'].values[0:number_coin_wanted] )