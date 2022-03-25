#!/bin/bash
current_pwd=`pwd`
gs=' -gs '
gs=''
symbols=(ALICEUSDT CELRUSDT RUNEUSDT FTMUSDT CTKUSDT SXPUSDT MATICUSDT COTIUSDT ONEUSDT YFIUSDT ENJUSDT EGLDUSDT BANDUSDT VETUSDT GRTUSDT OCEANUSDT CVCUSDT ATOMUSDT CRVUSDT IOSTUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Running screen on $i"
    screen -S "bybit_pro_$i" -dm bash -c "cd ${current_pwd}/;python3 passivbot.py $gs bybit_pro $i  configs/live/a_pro.json  -m spot "
done

