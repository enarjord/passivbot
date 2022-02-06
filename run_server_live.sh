#!/bin/bash
symbols=(DOGEUSDT XLMUSDT VETUSDT CHZUSDT SHIB1000USDT IOSTUSDT BITUSDT ONEUSDT SLPUSDT SPELLUSDT PEOPLEUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Running screen on $i"
    screen -S "tedy_$i" -dm bash -c "cd /home/tedy/Documents/passivbot5.3/passivbot;python3 passivbot.py bybit_tedy $i  configs/live/auto_unstuck_enabled.example.json"
done
