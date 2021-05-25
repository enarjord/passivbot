# Telegram integration

## Telegram support

The bot provides interfacing with the bot via Telegram via a telegram bot. There are a number of commands available to control the bot.

## Setup

In order to set up telegram, you'll need a telegram bot token and a chat-id. Once you have those, you can enable teleegram for each individual account that is specified in
the api-keys.json file. There is an example telegram configuration in that file to get started. If a telegram configuration
is not present for an account, telegram is disabled at startup.

For setup instructions, see https |//docs.microsoft.com/en-us/azure/bot-service/bot-service-channel-connect-telegram?view=azure-bot-service-4.0

Start a chat with @getmyid_bot in telegram to get chat id.

There are several commands & messages provided via Telegram, please issue a `/help` command in the telegram chat to see
all the options.

## Available commands

The following commands are available via telegram |

| Command          | Explanation |
| ---------------- | ----------- |
| `/balance`       | the equity & wallet balance in the configured account
| `/open_orders`   | a list of all buy & sell orders currently open
| `/stop`          | initiates a conversation via which the user can activate a stop mode
| `/position`      | information about the current position(s)
| `/show_config`   | the active configuration used
| `/reload_config` | reload the configuration from disk, based on the file initially used
| `/closed_trades` | a brief overview of bot's last 10 closed trades
| `/daily`         | an overview of daily profit
| `/set_leverage`  | initiates a conversion via which the user can modify the active leverage
| `/set_short`     | initiates a conversion via which the user can enable/disable shorting
| `/set_long`      | initiates a conversion via which the user can enable/disable long
| `/set_config`    | initiates a conversion via which the user can switch to a different configuration file  
| `/help`          | The help page explaining all the commands