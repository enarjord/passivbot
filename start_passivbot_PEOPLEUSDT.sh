#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 PEOPLEUSDT /opt/passivbot/configs/live/PEOPLEUSDT.json --leverage 20
