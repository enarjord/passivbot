
live_config="../configs/live/a_pro.json"
backtest_config="../configs/backtest/default_pro.hjson"
nb_best_coin=20

python3 0_coin_grid_validator.py ${live_config} ${backtest_config} -mv24 0 -mt24 0
python3 1_backtest_looper.py ${live_config} ${backtest_config} -jf tmp/grid_ok_coins.json
python3 2_backtest_summary.py ${nb_best_coin} ${live_config} ${backtest_config} -max-stuck-avg 7 -max-stuck 200  -min-gain 10

# ATTENTION pour la génération de script faut les passer en spot le live ajouter : -m spot
# Fuck en fait on peut pas passivbot ne supporte pas le spot...