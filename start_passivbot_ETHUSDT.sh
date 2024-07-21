#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 ETHUSDT /opt/passivbot/configs/live/ETHUSDT.json --leverage 20
