#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 FTMUSDT /opt/passivbot/configs/live/FTMUSDT.json --leverage 20
