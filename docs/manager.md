# Passivbot instances manager

* [Getting started](#getting-started)
  * [Create config file](#create-config-file)
  * [Configure instances](#configure-your-instances)
  * [Start instances](#start-instances)
* [Available commands](#available-commands)
* [Configuration](#configuration)
  * [Sections of a config file](#sections-of-a-config-file)
  * [Overrides](#overrides)
* [FAQ](#faq)
* [Examples](#examples)

Manager is a tool that allows you to easily configure and run multiple passivbot instances in parallel without installing any additional tools or libraries



## Getting started

### Create config file
Following command will create a `config.yaml` file where you can configure your instances
```sh
manager init
```
### Configure your instances
Go to the newly created `config.yaml` to manage your instances. Read the [configuration](#configuration) section for details
```sh
nano config.yaml
# or
vim config.yaml
# or any other editor of your choice
```
### Start instances
Following command will launch all the instances from your config
```sh
manager start -a
```



## Available Commands

### `manager sync`
Syncs all the instances with the config

**WARNING:** all passivbot instances that are launched on your machine in the moment when you run this command will be stopped.The only instances that will be started again, are the onces that you have in your config

Supports queries and following flags: `-s`, `-y`

### `manager start`
Starts specified instances. To see the instance logs go to `[passovbot path]/logs/[user]/[symbol].log`. The file will contain passivbot output logs and errors, unless you start the instance with the `-s` flag, in which case logs file will not be created.

Supports queries and following flags: `-a`, `-s`, `-y`
### `manager stop`
Stops specified instances

Supports queries and following flags: `-a`, `-u`, `-f`, `-y`

### `manager restart`
Restarts specified instances

Supports queries and following flags: `-a`, `-u`, `-f`, `-y`

### `manager list`
List in the terminal all the instances that run on your machine at this moment

Supports queries

### `manager info`
Get the detailed info about a passivbot instance. Unsynced instances are not fully supported yet, thus this command will not list the `long_exposure`, `long_mode` and all other flags for unsynced instances

Requires a query. If a query matches multiple instances, the first matched instance will be displayed

### `manager init`
Create a config file for the manager

### `manager help`
Show a list of all available commands. Will show info about a single command, if command's name will be passed as an argument


Refer to this [section](#how-to-write-queries), to get an idea on how to write queries



## Flags

### `-a` (aliased as `--all`)
Perform an action on all running instances

### `-u` (aliased as `--unsynced`)
Perform an action on unsynced instances

### `-f` (aliased as `--force`)
Only supported by commands that stop system processes. Makes a supported command to force stop a process

### `-y` (aliased as `--yes`)
Skip the confirmation step for an action and perform it immediately

### `-s` (aliased as `--silent`)
Start the affected instances without the output logs

### `-h` (aliased as `--help`)
Get help for a command



## Configuration
With the new version of a config parser there's now a support for symbol scoped configuration

Any instance in the config file will be assigned with a nearest scope of settings. It means that if there's an instance in the config with the specified `config` setting, then the `config` field of the `user` and `default` scope will be ignored for that instance. See the [overrides](#overrides) section for a better understanding

### Sections of a config file

```yaml
defaults:
  # This section contains the default
  #settings for your instances. If an
  # instance will not have closer scoped
  # settings, it will use these

instances:
  # your instances go here

  # name of an entry in the api-keys.json
  - user: binance_01
    # defaults overrides

    symbols:
      BTCUSDT:
        # user and defaults overrides
        
```


### Overrides

Imagine you have the following config:

```yaml
defaults:
  long_mode: gs
  short_mode: m
  ...

instances:
  - user: binance_01
    symbols:
      BTCUSDT:

      SOLUSDT: sol.json

      ETHUSDT:
        lm: p
        sm: gs

    config: override.json
```

This is what's going on up there:
* `BTC`'s settings:
  * `override.json` as a config from user scope
  * `m` for shorts from default scope
  * `gs` for longs from default scope

* `SOL`'s settings:
  * `sol.json` as a config from symbol scope
  * `m` for shorts from default scope
  * `gs` for longs from default scope

* `ETH`'s settings:
  * `override.json` as a config from user scope
  * `gs` for shorts from symbol scope
  * `p` for longs from symbol scope

In the example with `SOL` and `ETH`, you can see that symbol scoped settings support two types of data: strings and objects.
* If you pass a string to a symbol, then it will be considered as a name of a config or an absolute path to it

* If you pass an object to a symbol, then you can override any user and default scoped settings in that object

## FAQ

---

### I am confused with the new config syntax! Will my configs get invalid if I update the manager?
No, you don't have to update your config file if you update the manager, unless you want to. New config parser is backwards compatible, so your configs are still valid.

Here are all the ways you can list symbols now:
```yaml
- user: symbols_as_array
  symbols:
    - BTCUSDT
    - ETHUSDT

- user: symbols_as_object
  symbols:
    # yes, it is valid yaml
    BTCUSDT:
    ETHUSDT: eth.usdt
    BNBUSDT:
      lm: gs
      sm: gs
      cfg: panic.json
```

---

### How to write queries?

Imagine we have the following configuration:
```yaml
defaults:
    ...
instances:
    - user: binance_01
      symbols:
        - BTCUSDT
        - ETHUSDT
        - XMRUSDT

    - user: test-btc
      symbols:
        - BTCUSDT
```
Now we want to stop all the instances that trade BTC, and for that we run:
```sh
python3 manager stop symbol=BTC
```

Or if we want to start all instances for the `binance_01` user:
```sh
python3 manager start user=binance_01
```

Search parameters can be combined:
```sh
python3 manager start user=binance_01 symbol=BTCUSDT
```

Queries do not require parameter identifiers, but it is recommended to use them. Otherwise, "collisions" can occur:
```shell
python3 manager start btc
# starts all instances with the `BTC` symbol
# AND
# all instances with "btc" is user name
```