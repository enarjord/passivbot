#!/usr/bin/env bash

if [ "$#" -ne 2 ]
then
  echo "Usage: sh pusher.sh -t latest"
  exit 1
fi

if [ "$1" -ne '-t' ]
then
  echo "Usage: sh pusher.sh -t latest"
  exit 1
fi

# web
docker build -t jalanga/passivbot_futures:$2 .
docker push jalanga/passivbot_futures:$2