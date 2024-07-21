#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate passivbot-env
python /opt/passivbot/passivbot.py binance_01 XRPUSDT /opt/passivbot/configs/live/XRPUSDT.json --leverage 20
