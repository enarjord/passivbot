
python3 0_bybit_coin_grid_validator.py ../configs/live/a_tedy.json ../configs/backtest/default.hjson -mv24 0 -mt24 0
python3 1_backtest_looper.py -jf tmp/grid_ok_coins.json  ../configs/live/a_tedy.json ../configs/backtest/default.hjson
python3 2_backtest_summary.py 11

#python3 3_server_script_generator.py  -jf tmp/grid_ok_coins.json