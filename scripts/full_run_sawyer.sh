live_config="../configs/live/a_sawyer.json"
backtest_config="../configs/backtest/default_sawyer.hjson"
nb_best_coin=3

python3 0_coin_grid_validator.py ${live_config} ${backtest_config} -mv24 0 -mt24 0
python3 1_backtest_looper.py ${live_config} ${backtest_config} -jf tmp/grid_ok_coins.json
python3 2_backtest_summary.py ${nb_best_coin} ${live_config} -max-stuck-avg 7 -max-stuck 200  -min-gain 10

#python3 3_server_script_generator.py  adefinir ${live_config} -jf tmp/best_coins.json