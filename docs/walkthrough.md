# Example walkthrough of using the bot

The documentation provides detailed descriptions on each individual functionality the bot provides, like
installing, backtesting, optimizing and running the bot live. Because these scattered sections can be
overwhelming to somebody that is new to using Passivbot, this chapter provides a general walkthrough of
getting the bot set up, and the steps after that to using it.

This page is aimed at first-time users to help them get started, without diving into all the details straight away.

!!! Info
    These instructions are merely an example to help inexperienced users get started. Feel free to set things up the way you
    want if you're an experienced user!

## Prerequisites

In order to get Passivbot setup, you'll need a server to install it on. You can either host a server
at home, or rent a server on a VPS-provider like digitalocean, vultr or a similar provider. For setting
up a server, please refer to the instructions of the specific VPS you want to run the bot on. The smallest server
available is expected to be enough to run the bot.

The instructions in this tutorial will assume you are running an Ubuntu-server and are logged in to the terminal.

## Installing

Run the following commands to install git, python and Passivbot:

```shell
sudo apt install tmux
sudo apt install python
sudo apt install git python
git clone https://github.com/enarjord/passivbot.git
cd passivbot
pip install -r requirements.txt
```

## Configuration

After running the installation commands above, you need to create an API key on the exchange you want to run.
This will provide you with a key and a secret, that you need to fill out in the `api-keys.json`. You first need to
create this file by copying the template file `api-keys.example.json`. 
The instructions below assume your key is `{X}` and your secret is `{Y}`.

!!! Warning
    Make sure you enable `futures` on the API key. Also, be aware that on Binance you have to complete a quiz before you
    can trade futures. Apart from that, you need to make sure there are funds present in the futures wallet. 

## Run the optimizer for a config

!!! Warning
    For running the optimizer, a minimum of 8gb is recommended, and 16gb is be advisable. 

Once you've set up your account, you can try to find a good config using [the optimizer](optimize.md). If you want, you can limit the
search space by modifying by creating your own optimize configuration. You can do so by copying and modifying the default
optimize configuration file:

```shell
cp configs/optimize/default.hjson configs/optimize/myconfig.hjson
vi configs/optimize/myconfig.hjson
<make the changes you like using the editor>
:wq
```

!!! Info
    To learn about the different search space parameters, please refer to the [Configuration](configuration.md) page.

After this, you can start an optimize run on a symbol (XLMUSDT in this example):

```shell
python3 harmony_search.py -u binance_01 -s XLMUSDT -o configs/optimize/myconfig.hjson --start_date 2021-06-01 --end_date 2021-07-01
```

## Run the backtest for a config

After the optimizer finishes, you can backtest the best config produced:

```shell
python3 backtest.py results_harmony_search/{date}XXX/xxxxxx_best_config_{long/short}.json -u binance_01 -s XLMUSDT --start_date 2021-06-01 --end_date 2021-07-01
```

If you're happy with the config, copy it over to your live config file:

```shell
cp backtests/binance/XLMUSDT/plots/{date}/live_config.json configs/live/myconfig.json
```

## Starting the bot

To allow the bot to keep on running after you've disconnected, start a new tmux session:

```shell
tmux new -s mypassivbot
```

Once you're satisfied with the configuration, start the bot:

```shell
python3 passivbot.py binance_01 XLMUSDT configs/live/myconfig.json
```

If the bot started succesfully, you should receive a message **Passivbot started!**. After this, you can disconnect
from the tmux session by pressing `ctrl+b`, following by pressing `d`.

!!! Info
    To stop the bot, hit `ctrl+c`
