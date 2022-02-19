#!/bin/bash
current_pwd=`pwd`
symbols=(BNBUSDT MATICUSDT ALICEUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Running screen on $i"
    screen -S "sawyer_$i" -dm bash -c "cd ${current_pwd}/;python3 passivbot.py sawyer $i  configs/live/a_sawyer.json"
done

