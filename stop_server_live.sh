#!/bin/bash
symbols=(DOGEUSDT XLMUSDT VETUSDT CHZUSDT SHIB1000USDT IOSTUSDT BITUSDT ONEUSDT SLPUSDT SPELLUSDT PEOPLEUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Kill screen for $i"
    screen -S "tedy_$i" -X quit
done
