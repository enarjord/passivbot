# Telegram integration

!!! Warning
    Using one Telegram bot on multiple passivbot instances simultaneously will result in errors, so you need to generate
    use a separate telegram bot for each passivbot instance.

## Telegram support

Passivbot provides interfacing with the bot via Telegram via a telegram bot. There are a number of commands available to control the bot.

## Setup

In order to set up telegram, you'll need a telegram bot token and a chat-id. Once you have those, you can enable telegram
for each individual account that is specified in the `api-keys.json` file. There is an example telegram configuration in
the file to get started. If a telegram configuration is not present for an account, Telegram is disabled at startup.

For setup instructions, see https://docs.microsoft.com/en-us/azure/bot-service/bot-service-channel-connect-telegram?view=azure-bot-service-4.0

Start a chat with @getmyid_bot in telegram to get chat id.

There are several commands & messages provided via Telegram, please issue a `/help` command in the telegram chat to see
all the options.

## Available commands

The following commands are available via telegram |

| Command               | Explanation |
| --------------------- | ----------- |
| `/balance`            | the equity & wallet balance in the configured account
| `/open_orders`        | a list of all buy & sell orders currently open
| `/position`           | information about the current position(s)
| `/show_config`        | the active configuration used
| `/reload_config`      | reload the configuration from disk, based on the file initially used
| `/closed_trades`      | a brief overview of bot's last 10 closed trades
| `/daily`              | an overview of daily profit
| `/help`               | The help page explaining all the commands
| `/set_leverage`       | initiates a conversation via which the user can modify the active leverage
| `/set_short`          | initiates a conversation via which the user can enable/disable shorting
| `/set_long`           | initiates a conversation via which the user can enable/disable long
| `/set_config`         | initiates a conversation via which the user can switch to a different configuration file  
| `/transfer`           | initiates a conversation via which the user can transfer USDT funds between the spot and futures wallet<br/>**Note:** To enable this functionality, make sure you enable `Universal Transfer` on the API key  
| `/set_profit_transfer`| initiates a conversation via which the user can specify the percentage of profit that will automatically transferred to thee spot wallet<br/>**Note:** To enable this functionality, make sure you enable `Universal Transfer` on the API key
| `/stop`               | initiates a conversation via which the user can activate a stop-mode
| `/force_open`         | initiates a conversion via which the user can actively force (re)entry of a position based on the calculated grid

## Stop-mode

By using the `/stop` command in telegram, you can activate one of the available stop-modes. Each stop-mode has a different purpose
and effect, so please read carefully through the following sections to understand each stop-mode. The Telegram conversation when
you activate a stop-mode also has a general overview of what happens on each stop-mode.

### Graceful

When you activate the stop-mode `graceful`, the bot will continue to support the current open positions. This means it will keep
creating reentries as normal, until the position is closed. When there is no open position, it will **not** enter a new position.
This stop-mode can be useful when you want to stop the bot, but have time available to just the bot roll out until it's no longer
in a position.

### Freeze

After activating the stop-mode `freeze`, the bot will cancel all open orders. Any open position is left alone, but the bot
will no longer reenter, nor will it create initial entry orders. This stop-mode can be useful if you don't want your position to
increase (for example because you fear your position might get dangerous if it reenters).

!!! Info
    If you create an order on the exchange when stop-modee `freeze` is enabled, your manual orders will be cancelled by the bot.
    If you want to manually manage your position, please see stop-mode `manual`.

### Manual

Upon activating the stop-mode `manual`, the bot will not perform any more actions on the exchange. Any orders already created
will be left as-is, and no orders will be cancelled. This stop-mode can bee useful for example when you have a position open
that is in a dangerous state, and you want to manually manage the position (for example by opening a position on the opposite
side to drive away the liquidation price).

### Shutdown

Upon activating stop-mode `shutdown`, the but will completely shut down, leaving no process open on the server.

!!! Warning
    After activating this stop-mode, you will **NOT** be able to start the bot again via Telegram. You'll need to manually
    start the bot again.

### Panic

When the stop-mode `panic` is activated, the bot will immediately sell all open positions against market price and stop
further trading. This stop-mode can be useful when you're in a position and want to get out immediately, regardless of the
current profit or loss on the open position(s).

!!! Warning
    Activating this stop-mode will immediately sell all open positions at market price. It does **NOT** take the amount of
    loss into account, it simply sells everything. MAKE SURE THIS IS WHAT YOU WANT TO DO!

### Resume

When a stop-mode is activated (you can check the parameter `stop_mode` in the `/show_config` command), you can use this
mode to resume normal operation.

## Force open

By using the `/force_open` command, the user can actively have a market-order placed on the exchange to force a (re)entry of a position.
Upon initiating this command, the bot will ask the user if it should place a market-order for a `long` or `short` position side.
The quantity of the order will be based on the nearest order currently available in the grid.
