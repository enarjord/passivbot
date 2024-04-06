# Passivbot Quick Start

Here follows a quick start guide for passivbot installation

## Exchange account and API access

Passivbot works on Binance, Binance.US, Bybit, Bitget, OKX, Kucoin, BingX and Hyperliquid.  
US residents are advised to use Hyperliquid with a VPN because, at the time of writing, Hyperliquid does not require KYC verification for derivatives trading.

Create an API key with trading permissions. Some exchanges will use only a key and secret, some will use a key, secret and passphrase.  

Deposit some USDT and make an internal transfer to the trading or derivatives account.

## VPS

The bot can run on a local computer, but the general recommendation is to use a VPS (Virtual Private Server).  

Digital Ocean is one of many usable VPSs.  
Referral for 60 days free usage:  
https://m.do.co/c/10996bead65a  

Passivbot is lightweight, and 50+ bot instances can be run on the cheapest droplet ($4 per month as of this writing), as long as swap space is added (see below).  

Most exchanges have servers located in Asia, so choosing Singapore as droplet server location will probably give the lowest latency, but low latency is not critical for passivbot performance.

The bot can be installed on a Windows server, but in this guide it is assumed that a Ubuntu 22.04 x64 droplet is used.  

Access the VPS via SSH, web browser console, PuTTY or whatever method is preferred (if in doubt, the simplest is to click on "Access Console" -> "Launch Droplet Console" and a pop-up window will appear).

If running more than one or two bot instances on the cheapest droplet with 500mb RAM, adding swap space is needed. See this guide:  
https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04  

4GB swap is sufficient for most cases; add more if needed.  

## Step by step passivbot installation instructions

* Log into the VPS (see above)
* Perform system update and upgrade: `sudo apt update; sudo apt upgrade` and choose default values if prompted.
* Install python package manager: `sudo apt install python3-pip`
* Clone the github repository: `git clone https://github.com/enarjord/passivbot.git`
* Navigate into the passivbot directory: `cd passivbot`
* Install passivbot's dependencies: `pip install -r requirements_liveonly.txt`

The bot needs access to the exchange account, which it gets via the API key and secret (and passphrase if required).  
There needs to be added a new file with the filename `api-keys.json` into which the key and secret is pasted. A template file `api-keys.example.json` is included.  
A simple way to achieve this is the following:
* Copy the contents of this file `https://github.com/enarjord/passivbot/blob/master/api-keys.example.json` into a simple text editor of choice.
* Edit the relevant part of the file like this:
```
    "{exchange_name}_01" : {
        "exchange": "{exchange_name}",
        "key": "1234abc",
        "secret": "4567xyz",
        "passphrase": "9876abc"
    },
 ```
...and copy the entire file to the clipboard.
* Type this command: `cat > api-keys.json` and hit enter.
* Now paste the clipboard content (right click and 'paste' might be necessary).
* Hit enter again, then control+d.
* Verify the file was written correctly: `cat api-keys.json`, which should output the content which was pasted.

Now the installation is complete and bots may be launched.

A bot needs a live config whose parameters tell the bot how to behave. There are a few example configs included in the dir `configs/live/`.  
More live configs may be found here: https://github.com/JohnKearney1/PassivBot-Configurations or here https://pbconfigdb.scud.dedyn.io/ or elsewhere.  

Live configs are typically optimized on a single token's price history. To copy a config into the VPS, use the same procedure as when pasting the API key/secret:  
`cat > configs/live/{config_name}.json`, paste, hit enter, then control+d.  

A passivbot instance must be started in a way which allows it to run in the background after the console is closed.  
There are multiple options: the included passivbot manager, screen or tmux.  
See here for instructions to use the manager: `https://github.com/enarjord/passivbot/blob/master/docs/manager.md`  

Alternatively, use screen or tmux, which work similarly, but with different commands. Here follow instructions for tmux:
* Start a new tmux session: `tmux new -s {name}` where `{name}` is an arbitrary name chosen, for example `bot1` or `xrp`.
A green line at the bottom indicates one is inside a tmux session. Tmux sessions can be compared to browser tabs.
* Now a passivbot instance may be started: `python3 passivbot.py {exchange_name}_01 SYMUSDT configs/live/{config_name}.json`,  
where `{exchange_name}_01` must match the entry inside `api-keys.json` and `configs/live/{config_name}.json` must match the live config name file.  
If started without any additional arguments, default settings are used. Run `python3 passivbot.py --help` to see all command options.  
For example, to start a bot with shorts disabled and long wallet exposure limit set to 0.5, run  
`python3 passivbot.py {exchange_name}_01 SYMUSDT configs/live/{config_name}.json -sm gs -lw 0.5`

Now the console may be closed without stopping the bot.  

See the tmux documentation (or google "tmux cheatsheet") to learn how to attach and detach from sessions, create new sessions, navigate between sessions, create multiple windows inside the same session and so on.




