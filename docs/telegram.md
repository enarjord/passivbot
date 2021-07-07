# Telegram integration

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
| `/set_leverage`       | initiates a conversion via which the user can modify the active leverage
| `/set_short`          | initiates a conversion via which the user can enable/disable shorting
| `/set_long`           | initiates a conversion via which the user can enable/disable long
| `/set_config`         | initiates a conversion via which the user can switch to a different configuration file  
| `/transfer`           | initiates a conversion via which the user can transfer USDT funds between the spot and futures wallet<br/>**Note:** To enable this functionality, make sure you enable `Universal Transfer` on the API key  
| `/set_profit_transfer`| initiates a conversion via which the user can specify the percentage of profit that will automatically transferred to thee spot wallet<br/>**Note:** To enable this functionality, make sure you enable `Universal Transfer` on the API key
| `/stop`               | initiates a conversation via which the user can activate a stop mode
