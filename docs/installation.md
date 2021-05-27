# Welcome to MkDocs

### Requirements

- Python >= 3.8
- [requirements.txt](requirements.txt) dependencies

### Setup dependencies

```bash
    pip install -r requirements.txt
```

### Usage:

#### Supports exchanges Bybit futures and Binance futures, using hedge mode when supported

1. Add your api key and secret in file [api-keys.json](api-keys.json)
2. ```bash
   python3 start_bot.py {account_name} {symbol} {path/to/config.json}
   ```

Example:

```bash
python3 start_bot.py binance_01 XMRUSDT configs/live/binance_xmrusdt.json
```

#### Run with docker

Modify command with exchange and user_name in [docker-compose.yml](docker-compose.yml) and start
with `docker-compose up -d` (-d for background run). All code and files generated are in original git folder.

## Updating

If you want to update the bot, simply pull the latest version of the desired branch using ``git pull``