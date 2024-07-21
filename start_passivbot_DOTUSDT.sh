#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 DOTUSDT /opt/passivbot/configs/live/DOTUSDT.json --leverage 20
