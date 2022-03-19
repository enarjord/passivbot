#!/bin/bash
symbols=(ALICEUSDT CHZUSDT 1INCHUSDT MATICUSDT RUNEUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Kill screen for $i"
    screen -S "sawyer_$i" -X quit
done

