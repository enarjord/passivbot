FROM python:3.8-slim-buster

WORKDIR /app

# Telegram implementation require git to determine the version
RUN apt-get update && apt-get install git -y

COPY requirements.txt /app

RUN pip install -r requirements.txt
