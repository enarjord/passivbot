FROM python:3.8-slim-buster

WORKDIR /passivbot

# Telegram implementation require git to determine the version
RUN apt-get update && apt-get install git -y

ADD ./* /passivbot/

RUN pip install -r requirements.txt
