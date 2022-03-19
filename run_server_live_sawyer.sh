#!/bin/bash
current_pwd=`pwd`
gs=" -gs "
gs=""
symbols=(ALICEUSDT CHZUSDT 1INCHUSDT MATICUSDT RUNEUSDT )

for i in "${symbols[@]}"
do
    :
    echo "Running screen on $i"
    screen -S "sawyer_$i" -dm bash -c "cd ${current_pwd}/;python3 passivbot.py $gs sawyer $i  configs/live/a_sawyer.json"
done

