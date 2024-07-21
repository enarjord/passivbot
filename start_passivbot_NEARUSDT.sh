#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 NEARUSDT /opt/passivbot/configs/live/NEARUSDT.json --leverage 20
