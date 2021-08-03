# Installing passivbot

Passivbot can be installed and used either by installing it manually, or by starting it using Docker.
This page contains a description of both installation methods, and a step-by-step walkthrough.

## Manual install

Since Passivbot is written in Python, the easiest way to start it is by simply running it as a Python program.
The next sections describe what you will need to install & be able to run Passivbot, and how to get it installed.

### Initial install

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
1) Shutdown the bot as described in [Running live](live.md)
2) Pull the latest version of the bot by issueing `git pull` from the terminal.
3) Run `pip install -r requirements.txt` from the passivbot root directory to update the python requirements.
4) Start the bot again using the command described in [Running live](live.md) *

!!! Warning
	Take care when upgrading, as configs from different bot versions are highly likely to be incompatible and can result in unexpected results or severe losses

### Different versions

When you are working with the bot for a longer period of time, there is a chance you may want to stick to a specific
version for a while, even though the `master` branch may have some breaking changes since you last upgraded.

In cases this happens, you can simply switch to the appropriate branch using git. If you are not familiar with using git,
you can learn more about it on [this website](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging).

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
    The $PASSIVBOT_ROOT should be replaced with the location of your passivbot-folder (e.g. /home/passivbot)

For more detailed information on how to run a docker container, please check the [Docker documentation](https://docs.docker.com/engine/reference/commandline/run/).

## Installing on a Raspberry Pi

!!! Warning
	This process currently only installs the requirements to run the bot live. You will need to do the backtesting and optimising on different (more powerful) hardware.

!!! Info	
	Tested on a Raspberry Pi 3b with Raspberry Pi OS Kernel version 5.10.

Raspberry Pi OS currently uses Python2.7 and Python3.7, we need Python3.8.x. If you already have python3.8 installed, you can skip ahead to `Installing the requirements`.

### Installing python3.8

To install the dependencies for Python3.8 run the following:

```shell
sudo apt-get update
sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
```

To download, unpack and install Python3.8 run the following (this might take a long time):

```shell
wget https://www.python.org/ftp/python/3.8.11/Python-3.8.11.tgz
sudo tar zxf Python-3.8.11.tgz
cd Python-3.8.11
sudo ./configure --enable-optimizations
sudo make -j 4
sudo make altinstall
```

Upgrade pip and setuptools:

```shell
sudo pip3.8 install --upgrade pip
pip3.8 install setuptools --upgrade
```

### Installing the requirements

Clone the repository:

```shell
git clone https://github.com/enarjord/passivbot.git
cd passivbot
```

Install llvmlite:

```shell
sudo apt install llvm-9
LLVM_CONFIG=llvm-config-9 pip3.8 install llvmlite
```

Install the requirements:

```shell
python3.8 -m pip install -r requirements_liveonly.txt
```

That's it, you're ready to run the bot live. 

!!! Info 
	When [Running live](live.md) you will need to start the bot with `python3.8`.
