
# live_config="../configs/live/a_tedy.json"
live_config="../configs/live/static_grid_mode_auto_unstuck_enabled.example.json"
backtest_config="../configs/backtest/default.hjson"
nb_best_coin=20

echo "Using live config => ${live_config}"
echo "Using backtest config => ${backtest_config}"
echo "Searching nb coins => ${nb_best_coin}"
echo ''
sleep 2

python3 0_coin_grid_validator.py ${live_config} ${backtest_config} -mv24 0 -mt24 0
#python3 1_backtest_looper.py ${live_config} ${backtest_config} -jf tmp/grid_ok_coins.json -oh
#python3 2_backtest_summary.py ${nb_best_coin} ${live_config} ${backtest_config} -max-stuck-avg 7 -max-stuck 200  -min-gain 10
#python3 2_backtest_summary.py ${nb_best_coin} ${live_config} ${backtest_config} 

# #python3 3_server_script_generator.py  bybit_tedy ${live_config} -jf tmp/best_coins.json