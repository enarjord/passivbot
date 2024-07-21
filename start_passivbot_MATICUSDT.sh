#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 MATICUSDT /opt/passivbot/configs/live/MATICUSDT.json --leverage 20
