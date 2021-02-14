FROM python:3.8.7-buster
ENV PYTHONUNBUFFERED 1

RUN mkdir /passivbot_futures && \
    mkdir /passivbot_futures/api_key_secrets && mkdir /passivbot_futures/api_key_secrets/binance && mkdir /passivbot_futures/api_key_secrets/bybit && \
    mkdir /passivbot_futures/live_settings && mkdir /passivbot_futures/live_settings/binance && mkdir /passivbot_futures/live_settings/bybit

WORKDIR /passivbot_futures

COPY start_bot.py start_bot_docker.py passivbot.py binance.py bybit.py  ./

RUN pip3 install --upgrade pip

RUN python3 -m pip install matplotlib pandas websockets ccxt==1.41.63
