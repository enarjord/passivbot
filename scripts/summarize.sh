

# live_config="../configs/live/a_tedy.json"
# live_config="../configs/live/a_sawyer.json"
live_config="../configs/live/a_pro.json"


backtest_config="../configs/backtest/default.hjson"
nb_best_coin="5"


python3 2_backtest_summary.py ${nb_best_coin} ${live_config} ${backtest_config} \
-min-days 400 \
-min-gain 100 \
-max-marketcap-pos 20  
#-max-stuck-avg 7
# -max-stuck 200

#python3 3_server_script_generator.py  ${user} ${live_config} -jf tmp/best_coins.json