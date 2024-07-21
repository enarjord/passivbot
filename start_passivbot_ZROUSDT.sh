#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 ZROUSDT /opt/passivbot/configs/live/ZROUSDT.json --leverage 20
