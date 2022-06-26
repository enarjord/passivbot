#!/bin/bash
symbols=(MATICUSDT DOGEUSDT AVAXUSDT SOLUSDT DOTUSDT UNIUSDT XRPUSDT ADAUSDT TRXUSDT LTCUSDT )
for i in "${symbols[@]}"
do
    :
    echo "Kill screen for $i"
    screen -S "bybit_pro_$i" -X quit
done

