# Installing on a Raspberry Pi

!!! Warning
	This process currently only installs the requirements to run the bot live. You will need to do the backtesting and optimising on different (more powerful) hardware.

!!! Info	
	Tested on a Raspberry Pi 3b with the latest Raspberry Pi OS.

Raspberry Pi OS currently uses Python2.7 and Python3.7, we need Python3.8.x. If you already have python3.8 installed, you can skip ahead to `Installing the requirements`.

## Installing python3.8

To install the dependencies for Python3.8 run the following:

```shell
sudo apt-get update
sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
```

To download, unpack and install Python3.8 run the following (this might take a long time):

```shell
wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
sudo tar zxf Python-3.8.0.tgz
cd Python-3.8.0
sudo ./configure --enable-optimizations
sudo make -j 4
sudo make altinstall
```

Upgrade pip and setuptools:

```shell
sudo pip3.8 install --upgrade pip
pip3.8 install setuptools --upgrade
```

## Installing the requirements

Clone the repository:

```shell
git clone https://github.com/enarjord/passivbot.git
cd passivbot
```

Install llvmlite:

```shell
sudo apt install llvm-9
LLVM_CONFIG=llvm-config-9 pip3.8 install llvmlite
````

Install the requirements:

```shell
python3.8 -m pip install -r requirements_liveonly.txt
```

That's it, you're ready to run the bot live. 

## Running the bot on your Pi

Setup your api keys and telegram token as detailed in [Running live](live.md) and start the bot in the same way but with python3.8 ie:

```shell
python3.8 start_bot.py {account_name} {symbol} {path/to/config.json}
```

An actual command with the values filled in could look like this for example:

```shell
python3.8 start_bot.py binance_01 XMRUSDT configs/live/binance_xmrusdt.json
```
