#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 BNBUSDT /opt/passivbot/configs/live/BNBUSDT.json --leverage 20
