# Passivbot instances manager

* [Getting started](#getting-started)
  * [Create config file](#create-config-file)
  * [Configure instances](#configure-your-instances)
  * [Start instances](#start-instances)
* [Available commands](#available-commands)
* [Configuration](#configuration)
  * [Sections of a config file](#sections-of-a-config-file)
  * [Overrides](#overrides)
  * [Listing symbols](#listing-symbols)
* [FAQ](#faq)

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

**WARNING:** all passivbot instances that are running on your machine in the moment of running of this command will be stopped.The only instances that will be started again, are the onces that you have in your config

Supports queries and following flags: `-s`, `-y`

### `manager start`
Starts specified instances. To see the instance logs go to `[passovbot path]/logs/[user]/[symbol].log`. The file will contain passivbot output logs and errors, unless you start the instance with the `-s` flag, in which case logs file will not be created.

Supports queries and following flags: `-a`, `-s`, `-y`, `-m`
### `manager stop`
Stops specified instances

Supports queries and following flags: `-a`, `-u`, `-f`, `-y`

### `manager restart`
Restarts specified instances

Supports queries and following flags: `-a`, `-u`, `-f`, `-y`, `-m`

### `manager list`
List in the terminal all the instances that run on your machine at this moment

Supports queries

### `manager info`
Get the detailed info about a passivbot instance

Requires a query. If a query matches multiple instances, the first matched instance will be displayed

### `manager init`
Create a config file for the manager. Provide an argument to create a config file with a non-default name

Supports filename argument

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

### `-m` (aliased as `--modify`)
**EXPERIMENTAL** API will probably change in the future

Override flags of starting instances, hypothetically allows you to set any flags that passivbot supports. Again: it *overrides* the values that you set in you config. There is currently no way to track the consequences of this flag within the manager cli, so use with caution

### `-c` (aliased as `--config`)
Specify a path to a config file that you want to use with the manager. Should be absolute

### `-h` (aliased as `--help`)
Get help for a command



## Configuration
When you just start to use the manager, your config will look very similar to this:
```yaml
version: 2

defaults:
  config: config.yaml
  # ... other settings

instances:
  - user: binance_01
    symbols:
      - BTCUSDT
      - ETHUSDT
```

Such config will generate two instances for the `binance_01` user, one for each symbol under the `symbols:` section. This is how they will look in the processes tree:
```sh
pid  cmd
 1   python passivbot.py binance_01 BTCUSDT /config.json
 2   python passivbot.py binance_01 ETHUSDT /config.json
```

You can have as many users and symbols as you want:
```yaml
instances:
  - user: binance_01
    symbols:
     - BTCUSDT
     - ETHUSDT
     - SOLUSDT

  - user: bybit_01
    symbols:
     - ...

  - user: binance_spot_01
    symbols:
     - ...
```

---

### Sections of a config file
By the time you read this, you might already be familiar with the sections of a manager config file, but in case you are not there are two main section in a manager config file:
* `defaults` - serves as a top level settings scope, in other words: contains settings that all instances fall back to, when don't have a closer scoped settings
* `instances` - contains a list of instances groups

Every instance entry should have a user and a list of symbols, following instances won't work:
```yaml
instances:
  # does not have a user
  - symbols:
    - BTCUSDT

  - user: binance_01
    # does not have symbols
  
  # does not specify a user and a list of symbols
  - BTCUSDT
```

---

### Listing symbols
There are two ways to provide a list of symbols for a group of instances:

```yaml
# as a list
symbols:
  - BTCUSDT
  - ETHUSDT

# as an object
symbols:
  BTCUSDT:
  ETHUSDT:
```

These syntaxes can not be intermixed, as it won't be a valid yaml anymore. You can not list symbols as follows:

```yaml
symbols:
  # not valid: mixed items types
  - BTCUSDT
  ETHUSDT:
  - NANOUSDT
```

So what's the difference between the two of those syntaxes, you might be asking?

The difference is that with the objects syntax you can define the symbol-specific settings:

```yaml
symbols:
  BTCUSDT:
    cfg: slow.json
    lw: 0.5
    
  ETHUSDT:
    cfg: eth.json
    lw: 0.15
```

In-depth explanation of the symbol-specific settings is covered in the [overrides](#overrides) section

With symbols listed as objects you can also use a shorthand to provide symbol-specific config files:

```yaml
symbols:
  BTCUSDT: slow.json
  ETHUSDT: eth.json
```

But do not mix the settings overrides and the shorthand config file definition, as it will lead to an invalid yaml syntax:

```yaml
syymbols:
  # not valid yaml: lw does not belong to any object
  BTCUSDT: slow.json
    lw: 0.3
```

If you want to provide both scoped settings and a scoped config, then you should use the `cfg` setting:

```yaml
symbols:
  # now yaml parser is happy
  BTCUSDT:
    cfg: slow.json
    lw: 0.3
```

---

### Overrides

Every instance always falls back to the closest scope of settings

`default` section of a config file, being the top level settings scope, defines the default settings for all instances

The next level of scope is the `user` level instance the `instances` section

And the closest scope that an instance can have is the symbol scoped settings

You can visualize the settings lookup logic as follows:

```
symbol -> user -> defaults
```

When an instance gets created, and it looks for, let's take, the `long_exposure` setting, it first looks at the symbol scope, then it goes to the user scope and to the defaults scope at last. At every step an instance checks if the `long_exposure` setting is defined inside a scope. Once the setting definition was found, instance stores the value and starts looking for the next setting. If none of the available scopes contained the setting, then it will remain blank for that instance

Let's look at an example. Imagine you have the following config:

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

## FAQ

---

### Where are instance logs located?

Instance logs are located inside the `logs` folder in the root of a passivbot directory

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

### How to use modifiers?

Config modifiers is an experimental flag that allows you to override all settings of any instance via the cli. 

To apply a modifier, you should use the `-m` flag when using the start or restart command:
```sh
manager start user=binance_01 -m "-lw 0.3 -lm n"
```

Use this flag with caution and at your own risk as at the moment manager will not let you know if there are any modifiers applied to an instance, which can lead to an unexpected outcome.

Modifiers certainly support key-value pairs, such as `-lm n` or `-ab 100`. Flags that do not require a value are not yet supported, those are the `-gs` and `-tm` flag