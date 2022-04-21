#!/bin/bash
symbols=(CHZUSDT ALICEUSDT RUNEUSDT MATICUSDT 1INCHUSDT CRVUSDT SXPUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Kill screen for $i"
    screen -S "sawyer_$i" -X quit
done

