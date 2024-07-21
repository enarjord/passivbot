#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 PIXELUSDT /opt/passivbot/configs/live/PIXELUSDT.json --leverage 20
