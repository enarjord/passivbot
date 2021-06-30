# Installing passivbot

Passivbot can be installed and used either by installing it manually, or by starting it using Docker.
This page contains a description of both installation methods, and a step-by-step walkthrough.

## Manual install

Since Passivbot is writting in Python, the easiest way to start it is by simply running it as a Python program.
The next sections describe what you will need to install & be able to run Passivbot, and how to get it installed.

### Requirements

The following requirements are applicable for installing Passivbot:

- Git 2.17 or above
- Python 3.8.x (newer versions aren't supported yet)
- Supported OS:
  - Mac
  - Linux
  - Windows
    
#### Hardware requirements

Passivbot is a very lightweight bot, and can easily be run on a single-core machine with 1GB of RAM.
While the hardware requirements are very low, you may run into issues when running it on things like a Raspberry Pi.
In case you do, please let us know so we can help out & improve the bot!

### Iinitial install

When installing the bot for the first time, you can simply clone the repository using the following command from a terminal:

```shell
    git clone https://github.com/enarjord/passivbot.git
```

This will create a folder `passivbot`, with the code checkout on the master branch (this is the default where all stable & verified updates are pushed to).
After the code has been checked out, you will need to install the python dependencies using the following commands:

```shell
cd passivbot
pip install -r requirements.txt
```

### Upgrading

Updating the bot is as straightforward as getting the latest version, and stopping and starting the bot.
Before starting an upgrade, please make sure your bot is in an acceptable state to upgrade (e.g. does it have a position open that will be time-critical).


If you've installed the bot using the git clone instructions above, please follow these steps to upgrade:
1) Shutdown the bot via one of the two options described below.
   * via telegram by using the `/stop`
   * from the terminal using `ctrl+c`
2) Pull the latest version of the bot by issueing `git pull` from the terminal.
3) Start the bot again using the command described in [live.md](Live)
    
!!! Info
    **Please note that shutting down the bot properly may take a couple of seconds, as it needs time to properly detach from the websocket and shutdown the bot. When the bot is completely shut down, you will see a message that Telegram has been shut down, and another message that Passivbot has been shut down.

## Docker

At the moment, the Docker image is not built automatically yet. For now you can build the Docker image locally and push
it into your local Docker image repository using the following command from the root folder: 

```shell
docker build -t passivbot .
```

After building the docker image and having added it to your local container registry, you can use this image to
create a container to run commands in. These instructions assume you have a recent working version of Docker installed
on the machine where this is executed from.

### Starting live mode

In order to start passivbot live with Docker, you'll need to mount the config folder in order to make the required information available.
To start passivbot in live mode, you can issue the following command from the root folder of passivbot:

```bash
docker run --name passivbot -d -v $PASSIVBOT_ROOT/config:/passivbot/config passivbot
```

!!! Info
    **The $PASSIVBOT_ROOT should be replaced with the location of your passivbot-folder (e.g. /home/passivbot)

For more detailed information on how to run a docker container, please check the [Docker documentation](https://docs.docker.com/engine/reference/commandline/run/).