
# run "python backtest_all_strat.py"
# change "source_path" & "command_line" to be correct for you
from datetime import datetime
import os
import subprocess
import json
import pandas as pd
from tabulate import tabulate


###################################################""
# @DONE : Limiter chaque backtest à un temps maximum => codé mais pas testé :(
###################################################""

timerange = "--timerange=20210101-"
#timeframe = "5m" <= une stratégie doit avoir son timeframe

source_path = "./user_data/strategies"
backtest_path = "./user_data/backtest_results"
command_line = ["freqtrade", "backtesting", 
            #"--timeframe", timeframe, 
            #"--strategy-list", "{strategy_name}", 
            "-s", "{strategy_name}", 
             timerange]
max_strategy_to_check = -1
freqtrade_timeout_seconds = 10 * 60 * 60 # set timeout of backtesting to 10 minutes maximum 10 * 60 * 60

# Download candles needed
#download_command = ['freqtrade', 'download-data', '--timeframes', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '12h', timerange]
#subprocess.run(download_command)


# function to get the strategy list
def getStrategiesNames(source_path):
    strats_files=os.listdir(source_path)
    strats_name=[]
    for filename in strats_files:
        strats_name.append(filename.replace(".py", ""))
    return strats_name

# format for human readable dataframe
def formatDataFrame(df):
    # rebuil the dataframe and print it
    #df.drop(columns=['profit_mean_pct', 'profit_sum',  'profit_total_pct'], inplace=True)
    df.rename(columns={   'key' : 'Strategy',
                        'trades' : 'Buys',
                        'profit_mean' : 'Avg Profit %',
                        'profit_sum_pct' : 'Cum Profit %',
                        'profit_total_abs' : 'Tot Profit COIN',
                        'profit_total' : 'Tot Profit %',
                        'duration_avg' : 'Avg Duration',
                        'wins' : 'Win',
                        'draws' : 'Draw',
                        'losses' : 'Loss',
                        'max_drawdown_abs' : 'DD amount',
                        'max_drawdown_per' : 'DrawDown %'}, inplace=True)

    # calculate and adjust dataframe
    df['Tot Profit COIN'] = df['Tot Profit COIN'].round(2)
    df['Avg Profit %'] = df['Avg Profit %'].multiply(100).round(2)
    df['Tot Profit %'] = df['Tot Profit %'].multiply(100).round(2)
    df['Win %'] = (100 * df['Win'] / (df['Win'] + df['Loss'])).round(2)

    # Reorder dataframe
    df = df[['Strategy', 'Buys', 'Avg Profit %', 'Cum Profit %', 'Tot Profit COIN', 'Tot Profit %', 'Avg Duration', 'Win', 'Draw',
                        'Loss', 'Win %', 'DD amount', 'DrawDown %']]

    return df







#####################
# Start the main part
#####################

# Check if directory exist
if not os.path.isdir(source_path):
    print("Directory not exist ", source_path)
    exit()
else:
    print("OK, Directory exist ", source_path)
    
print("Command executed is : ", print(' '.join(command_line)))
print("-----------")
print("Find all strategies on the directory ", source_path)
a_strategies_name = getStrategiesNames(source_path)
nb_strats = len(a_strategies_name)

if (max_strategy_to_check == -1):
    max_strategy_to_check = nb_strats

print("Find ", nb_strats, ' Strat.')
print('User in script limit to ', max_strategy_to_check, ' Strat.')

df = pd.DataFrame()
limit = 0
start_time = datetime.now()
nb_strat_to_run = min(max_strategy_to_check, nb_strats)
a_failed_starts = []
# part to launch the command line
for current_strat in a_strategies_name: 
    print('------------------------')
    print('BackTesting strategy ', current_strat, ' (', (limit+1),'/', nb_strat_to_run,')')
    final_command_line = []
    for element in command_line:
        final_command_line.append(element.replace("{strategy_name}", current_strat))

    # run FreqTrade
    try:
        subprocess.run(final_command_line, capture_output=True, text=True, timeout=freqtrade_timeout_seconds)
    except subprocess.TimeoutExpired:
        print('Timeout Reached (', freqtrade_timeout_seconds, ' seconds)')


    # find the last backtesting result
    f = open(backtest_path + "/.last_result.json")
    last_result = json.load(f)
    f.close()
    bt_file = last_result['latest_backtest']

    # read the last backtest result
    f = open(backtest_path + "/" + bt_file)
    bt = json.load(f)
    f.close()

    # beacareful, strategy can be in error (no result fund)
    if (bt['strategy_comparison'][0]['key'] != current_strat):
        a_failed_starts.append(current_strat)
        print('Sorry, Strat in Error')
        limit = limit + 1
        continue

    # append the backtest result to dataframe
    df_new = pd.DataFrame(bt['strategy_comparison'])


    df_new = formatDataFrame(df = df_new)
    print('Partial Result :')
    print(tabulate(df_new, headers='keys', tablefmt='psql'))

    if (len(df) == 0):  # j'ai pas trouvé comment faire autrement... tristesse...
        df = df_new
    else:
        df = df.append(df_new)


    eta = ((datetime.now() - start_time) / (limit+1)) * (nb_strat_to_run - (limit+1)) 
    print('ETA : {}'.format(eta))
    print('Partial : Find ', len(a_failed_starts), ' Strategie(s) in Error : ', a_failed_starts)

    limit = limit + 1
    if limit >= max_strategy_to_check:
        break

print("")
print("------------------------------------------------------------")
print("FINAL SUMMARY, sorted by 'DrawDown %' than by 'Tot Profit %'")
print("------------------------------------------------------------")

df.sort_values(by=['DrawDown %', 'Tot Profit %'], ascending=True, inplace=True)

# Print beautifully the dataframe
print(tabulate(df, headers='keys', tablefmt='psql'))


print("")
print('Find ', len(a_failed_starts), ' Strategie(s) in Error : ', a_failed_starts)
print("Command executed is : ", print(' '.join(command_line)))


