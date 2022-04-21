#!/bin/bash
current_pwd=`pwd`
gs=" -gs "
gs=""

symbols=(SXPUSDT CELRUSDT CHRUSDT DOGEUSDT CHZUSDT ONEUSDT 1INCHUSDT ENJUSDT RSRUSDT GRTUSDT VETUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Running screen on $i"
    screen -S "bybit_tedy_$i" -dm bash -c "cd ${current_pwd}/;python3 passivbot.py $gs bybit_tedy $i  configs/live/static_grid_mode_auto_unstuck_enabled.example.json"
done

