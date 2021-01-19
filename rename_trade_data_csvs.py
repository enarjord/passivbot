import shutil
import os
import pandas as pd

exs = ['binance', 'bybit']
for ex in exs:
    base_path = f'historical_data/{ex}/agg_trades_futures/'
    dirs = os.listdir(base_path)
    for s in dirs:
        if '_cache' in s:
            print('removing', s)
            shutil.rmtree(base_path + s)
        elif '.' not in s:
            fnames = os.listdir(base_path + s)
            for f in fnames:
                if '_' not in f:
                    fff = f'{base_path}{s}/{f}'
                    df = pd.read_csv(fff).set_index('trade_id')
                    if (kd := 'Unnamed: 0') in df.columns:
                        df = df.drop(kd, axis=1)
                    if len(df) == 100000:
                        new_name = f'{df.index[0]}_{df.index[-1]}.csv'
                        fffn = f'{base_path}{s}/{new_name}'
                        print('renaming', fff, fffn)
                        os.rename(fff, fffn)
                    else:
                        print('removing', fff)
                        os.remove(fff)
