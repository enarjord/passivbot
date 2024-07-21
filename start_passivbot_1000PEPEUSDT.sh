#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 1000PEPEUSDT /opt/passivbot/configs/live/1000PEPEUSDT.json --leverage 20
