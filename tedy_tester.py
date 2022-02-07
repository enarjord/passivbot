#print only "grid ok" coins
from pybit import HTTP
import subprocess

# @TODO : faire un script de lecture des fichiers et de sort

# @TODO : tedy_grid_ok => qui écrit dans un fichier le résultat ?
# @TODO : tedy_bach_backtest
# @TODO : tedy_resultat_manager
# @TODO : séparer la generation du script shell


# SETTINGS PART START =============>
starting_balance = 100
wallet_exposure_limit = 0.15
min_volume_24h = 0 # 10000000 # the min volume you want, to avoid low volume coin 
min_turnover_24h = 0 # 3000000
start_date = "2021-01-01"
end_date = "2022-02-01"
user_connexion = "bybit_tedy"
config_file = "configs/live/auto_unstuck_enabled.example.json"
initial_qty_pct =  0.016877534968530956
# <======================== SETTINGS PART END

config = {'starting_balance': starting_balance, # your starting balance in $
          'wallet_exposure_limit': wallet_exposure_limit, # your wallet exposure
          'initial_qty_pct': initial_qty_pct} # the initial_qty_pct picked from the config you will run

backtest_command_line = [
    "python3", "backtest.py", "-u", user_connexion, "-s", "#SYMBOL_NAME#", "--starting_balance="+str(starting_balance), "-sd"
    , start_date, "-ed", end_date, config_file
    ]

print('----------------------------------------------------------------------------------------')
print("Reading informations from ByBit to find :")
print("1/ coin with grid OK")
print("2/ turnover_24h > ", min_turnover_24h)
print("3/ volums_24h > ", min_volume_24h)
print('')

#Initial Entry Cost:
session = HTTP("https://api.bybit.com")
res = session.query_symbol() #gives all symbols, cannot filter
initial_entry_cost = config['starting_balance'] * config['wallet_exposure_limit'] * config['initial_qty_pct']
config['symbol'] = ''
bash_symbols=[]

for coin in res['result']:
    if 'USDT' in coin['name']:
        config['symbol'] = coin['name']

    min_qty = float(coin['lot_size_filter']['min_trading_qty'])
    symboldata = session.latest_information_for_symbol(symbol=config['symbol'])
    try:
        min_notional = min_qty * float(symboldata['result'][0]['last_price'])
    except:
        continue
    volume_24h = float(symboldata['result'][0]['volume_24h'])
    turnover_24h = float(symboldata['result'][0]['turnover_24h'])

    #print(symboldata['result'][0]['turnover_24h'])
    if turnover_24h >= min_turnover_24h:
        if volume_24h >= min_volume_24h:
            if initial_entry_cost >= min_notional:
                print (f"{config['symbol']} \t min_notional: {min_notional:.2f} GRID OK [volume_24h : {volume_24h}, turnover_24 : {turnover_24h}]")
                bash_symbols.append(config['symbol'])

# bash_symbols = ["DOGEUSDT"]

print ("found ", len(bash_symbols), " symbols.")
print ("Wallet exposure id ", len(bash_symbols) * config['wallet_exposure_limit'])
print('----------------------------------------------------------------------------------------')
print("Starting Backtesting")
print("Command executed is : ", print(' '.join(backtest_command_line)))
# python3 backtest.py -u bybit_tedy -s XLMUSDT --starting_balance=100 -sd 2021-01-01 -ed 2022-02-01 configs/live/auto_unstuck_enabled.example.json
nb_coin=len(bash_symbols)
current_i = 0
for current_symbol in bash_symbols:
    print('BackTesting ', current_symbol, ' (', (current_i+1),'/', nb_coin,')')
    freqtrade_timeout_seconds = 10 * 60 * 60 # set timeout of backtesting to 10 minutes maximum 10 * 60 * 60
    final_command_line = []
    for element in backtest_command_line:
        final_command_line.append(element.replace("#SYMBOL_NAME#", current_symbol))

    # run FreqTrade
    try:
        # subprocess.run(final_command_line, capture_output=True, text=True, timeout=freqtrade_timeout_seconds)
        subprocess.run(final_command_line)
    except subprocess.TimeoutExpired:
        print('Timeout Reached (', freqtrade_timeout_seconds, ' seconds)')
    current_i = current_i + 1

print ("found ", len(bash_symbols), " symbols.")
print ("Wallet exposure id ", len(bash_symbols) * config['wallet_exposure_limit'])

################### shell script generating ##############
print('----------------------------------------------------------------------------------------')
print ("Linux Bash to create the screens commands (run_server_live.sh) :")
print("#!/bin/bash")
print('symbols=(', end='')
for symbol in bash_symbols:
    print (symbol, end=' ')
print(')')
print ('for i in "${symbols[@]}"')
print ('do')
print ('    :')
print ('    echo "Running screen on $i"')
print ('    screen -S "tedy_$i" -dm bash -c "cd /home/tedy/Documents/passivbot5.3/passivbot;python3 passivbot.py bybit_tedy $i  configs/live/auto_unstuck_enabled.example.json"')
print ('done')
print ('')

print('----------------------------------------------------------------------------------------')
print ("Linux Bash to kill all screens (stop_server_live.sh) :")
print("#!/bin/bash")
print('symbols=(', end='')
for symbol in bash_symbols:
    print (symbol, end=' ')
print(')')
print ('for i in "${symbols[@]}"')
print ('do')
print ('    :')
print ('    echo "Kill screen for $i"')
print ('    screen -S "tedy_$i" -X quit')
print ('done')
print ('')



