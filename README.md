# passivbot_futures
trading bot running on binance usdt futures

use at own risk


requires python >= 3.8


dependencies, install with pip:


`python3.8 -m pip install matplotlib pandas websockets ccxt`


------------------------------------------------------------------

released freely -- anybody may copy, redistribute, modify, use for commercial, non-commercial, educational or non-educational purposes, censor, claim as one's own or otherwise do or not do whatever without permission from anybody

------------------------------------------------------------------

usage:

binance account needs futures enabled

add api key and secret as json file in dir `api_key_secret/binance/your_user_name.json`

formatted like this: `["KEY", "SECRET"]`


make a copy of `settings/binance_futures/default.json`

rename the copy `your_user_name.json` and make desired changes

run in terminal: `python3 passivbot_futures.py your_user_name`
