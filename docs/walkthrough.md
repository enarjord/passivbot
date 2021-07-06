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
sudo apt-get install screen
sudo apt-get install python
sudo apt-get install git python
git clone https://github.com/enarjord/passivbot.git
cd passivbot
pip install -r requirements.txt
```

## Configuration

After running the installation commands above, you need to create an API key on the exchange you want to run.
This will provide you with a key and a secret, that you need to fill in in the `api-keys.json`. The instructions below
assume your key is `{X}` and your secret is `{Y}`.

Also, you will need to get a Telegram token & chat-id to fill into the `api-keys.json`. Please refer to the [Telegram](telegram.md)
section for instructions on how to set this up.

These instructions show using [vi](http://www.atmos.albany.edu/daes/atmclasses/atm350/vi_cheat_sheet.pdf) to edit the `api-keys.json`, but you can use any command-line editor you like.

```shell
vi api-keys.json
{navigate to the key item in the api-keys file under binance_01 using your arrow keys}
{copy the key from the exchange into your clipboard, and paste it in the editor window}
{navigate to the secret item in the api-keys file under binance_01 using your arrow keys}
{copy the secret from the exchange into your clipboard, and paste it in the editor window}
{navigate to the token item in the telegram subkey in the api-keys file under binance_01 using your arrow keys}
{copy the telegram token into your clipboard, and paste it in the editor window}
{navigate to the chat-id item in the telegram subkey in the api-keys file under binance_01 using your arrow keys}
{copy the telegram chatid into your clipboard, and paste it in the editor window}
:wq
```

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
python3 optimize.py -u binance_01 -s XLMUSDT -o configs/optimize/myconfig.hjson --start_date 2021-06-01T00:00 --end_date 2021-07-01T00:00 
```

## Run the backtest for a config

After the optimizer finishes, it will run a backtest for you. You can also manually trigger the same result the optimize produced:

```shell
python3 backtest.py -u binance_01 -s XLMUSDT --start_date 2021-06-01T00:00 --end_date 2021-07-01T00:00 backtest/binance/XLMUSDT/plots/{date}/live_config.json
```

If you're happy with the config, copy it over to your live config file:

```shell
cp backtest/binance/XLMUSDT/plots/{date}/live_config.json myconfig.json
```

## Starting the bot

To allow the bot to keep on running after you've disconnected, start a new screen session:

```shell
screen -S mypassivbot
```

Once you're satisfied with the configuration, start the bot:

```shell
python3 start.py binance_01 XLMUSDT configs/mycnfig.json
```

If the bot started succesfully, you should receive a message **Passivbot started!**. After this, you can disconnect
from the screen session by pressing `ctrl+a`, following by pressing `d`.

!!! Info
    To stop the bot, simply use the `stop` button on the second page of buttons in Telegram
