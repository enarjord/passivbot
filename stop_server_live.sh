#!/bin/bash
symbols=(SXPUSDT CELRUSDT CHRUSDT DOGEUSDT CHZUSDT ONEUSDT 1INCHUSDT ENJUSDT RSRUSDT GRTUSDT VETUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Kill screen for $i"
    screen -S "tedy_$i" -X quit
done

