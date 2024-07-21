#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 TONUSDT /opt/passivbot/configs/live/TONUSDT.json --leverage 20
