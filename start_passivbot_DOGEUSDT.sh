#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 DOGEUSDT /opt/passivbot/configs/live/DOGEUSDT.json --leverage 20
