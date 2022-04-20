# Instances manager

* [Usage](#usage)
* [Commands](#commands)
* [Configuration](#configuration)
* [Examples](#examples)

It is a module that allows you to easily configure and run multiple instances of the bot in parallel, without requiring you to install any additional tools or libraries.

## Usage

These are the steps you need to follow to use the instances manager:

1. Copy or rename the `config.example.yaml` file to `config.yaml` in the manager directory.
2. Fill in the `config.yaml` file with your settings. Refer to the [Configuration](#configuration) section for details.
3. Launch configured instances by running `python3 manager start -a`

## Commands

* `sync` - sync instances with config
* `list` - list instances
* `info` - show info about an instance
* `start` - start instances
* `stop` - stop instances
* `restart` - restart instances
* `help` - show help


All operations, that take instances as arguments, allow you to query instances by these parameters:

* `id` - instance id
* `user` - user name
* `symbol` - symbol of an instance
* `pid` - process id of an instance
* `status` - status of an instance
    * It can be `running` or `stopped`

Look through this [section](#how-to-write-queries), to get an idea of how to write queries.

If you want to learn more about the commands, and how to use them, run this:
```shell
python3 manager help
```

## Configuration

This section explains how to configure the instances manager.

There are two main entries in the configuration file: `defaults` and `instances`.

### `defaults`

These are the default settings for all instances. You can override them in the `instances` section.

* `live_config_name` - name of the live config file, within a `configs/live` directory
* `live_config_path` - path to the live config file
    * this field is optional, and empty by default. You may put any absolute path to a config file, or leave it empty.
* `market_type` - `futures` or `spot`

---

### WARNING

Following instances will be treated as the same instance,
because `market_type` is not used to identify instances:

```yaml
...
instances:
    - user: main
      symbols:
        - BTCUSDT
      market_type: spot

    - user: main
      symbols:
        - BTCUSDT
      market_type: futures
```

But there is a workaround. Create a separate user in the `api-keys.json` for the `spot` instances, and give it the same API keys as the `futures` instances.

---

Explanation of the rest of `defaults` fields can be found in the [Controlling the bot](https://www.passivbot.com/en/latest/live/#controlling-the-bot) section. These fields have no effect by default, but once you change them to a non-default value, the manager will start using them.


### `instances`

In this entry, you configure your instances. `defaults` can be overridden in this section.

* `user` - name of an entry in the `api-keys.json` file
* `symbols` - list of symbols to trade
* any filed from the `defaults` section

How to override `defaults`:
```yaml
...
instances:
    - user: main
      symbols:
        - BTCUSDT
      live_config_name: my_config.yaml
      live_config_path: /path/to/my_config.yaml
      short_mode: 'n'
```

## Examples

### How to quickly start/stop/restart instances
```shell
python3 manager start -a
python3 manager stop -a
python3 manager restart -a
```

---

### How to write queries

Imagine we have the following configuration:
```yaml
defaults:
    ...
instances:
    - user: main
      symbols:
        - BTCUSDT
        - ETHUSDT
        - XMRUSDT
    - user: test-btc
      symbols:
        - BTCUSDT
```
Now we want to stop all the instances with the `BTC` symbol, it can be done by running:
```shell
python3 manager stop symbol=BTC
```

Or if we want to start all instances for the `main` user:
```shell
python3 manager start user=main
# starts all instances for the `main` user
```

Search parameters can be combined:
```shell
python3 manager start user=main symbol=BTC
# stops all instances for the `main` user with the `BTC` symbol
```

Queries do not require parameter identifiers, but it is recommended to use them. Otherwise, "collisions" can occur:
```shell
python3 manager start btc
# starts all instances with the `BTC` symbol
# AND
# all instances with "btc" is user name
```