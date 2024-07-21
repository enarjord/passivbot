#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 AGIUSDT /opt/passivbot/configs/live/AGIUSDT.json --leverage 20
