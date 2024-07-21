#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 WIFUSDT /opt/passivbot/configs/live/WIFUSDT.json --leverage 20
