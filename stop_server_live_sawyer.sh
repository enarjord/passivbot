#!/bin/bash
symbols=(BNBUSDT MATICUSDT ALICEUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Kill screen for $i"
    screen -S "sawyer_$i" -X quit
done

