#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 NOTUSDT /opt/passivbot/configs/live/NOTUSDT.json --leverage 20
