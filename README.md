# Refactoring Information

Below is some general information about a re-structure of the codebase.
The idea is to arrange files in a way that allows for the easiest development, usage, and expansion over time.
Since this is such a broad and somewhat ambiguous process, there is no single correct way to arrange the files, although some arrangements may have benefits over others.
Thus, it's advantageous to get as much developer feedback as possible to ensure that the updated structure is easy to change, compartmentalize, test etc.
NOTHING here is final.

**Proposed Structure:** This is just temporary and will be updated as progress is made.

> `passivbot_futures/`  
> │  
> ├── `Refactor.md`    
> ├── `README.MD`  
> ├── `main.py` &/or `start_bot.py`  
> ├── `setup.py`  
> ├── `.gitignore`  
> ├── `requirements.txt`
> │── `changelog.txt`  
> ├── `infrastructure/`  
> │    ├── `Dockerfile`  
> │    ├── `docker-compose.yml`  
> ├── `data/`  
> │    └── `historical_data/`  
> │    └── `logs/`
> └── `bot/`  
>    ├── `passivbot.py`  
>    ├── `exchanges/`  
>    │    ├── `bybit.py`  
>    │    └── `binance.py`  
>    ├── `backtester/`  
>    │    ├── `backtest.py`  
>    │    ├── `backtest_notes.ipynb`  
>    │    ├── `binanace_notes.ipynb`  
>    │    ├── `bybit_notes.ipynb`  
>    │    └── `backtest_configs/`  
>    ├── `live_settings/`  
>    │    ├── `binance/`  
>    │    └── `bybit/`  
>    ├── `API_KEYS/`  
>    │     ├── `binance/`  
>    │     └── `bybit/`


### `README.MD`  

Abstract for the program, general usage, description, version info, author information and contribution guidelines.

### `main.py` &/or `start_bot.py`  

start_bot.py should stay in the root folder for the time being for development usage.
main.py is NOT written yet, as there should be some discussion on how it is implemented.
From my somewhat novice understanding of setuptools/pypi distribution, having a main.py that acts as a central entry point for the script is best practice.
This means conceiving of *some* kind of menu or UI that allows end-users to change and use all parts of the bot from a central place.
I had experimented with command line interface menus, but given the complexity of the bot, it seems the best approach may be to run a local webapp, similar to how jupyter-lab works.
In a distribution environment, it's best to assume the end user knows absolutely nothing about how to configure or read complex json files, edit code, or otherwise.
Therefore, said webapp should include functionality to edit and add users (API keys), edit configuration files (for backtester and live bot), copy or upload an external configuration file,
and most importantly, facilitate backtesting. My inexperience may be failing me here, but it seems that the single most complex part of packaging passivbot will stem from finding a good way to implement
backtesting *and* the data analytics (currently done in jupyter) within whatever UI is selected for use. That said, this isn't a pre-requisite for a general restructure,
I just say this as it effects how the files should be laid out to make compartmental development, testing, and end usage as easy as possible.  

Martijn397 shared some work he had done in the realm of web UI on discord:  

![@martijn397](Data/img/passivUI.png)

### `setup.py`  

Contains setup instructions for producing a distribution file accepted by pypi. There's a lot of ambiguity in how this file can be written, but from my understanding, it requires a main point of entry for the final script.
i.e. Our end user opens a command line, runs "passivbot", and the UI for the bot opens, with the command line remaining in the background to keep a log.
Therefore, setup.py needs to know information about the author, command alias, descriptions etc, but most importantly *what* to run when the user types that command.

### `.gitignore`  

Contains files git will ignore. Needs no changes.  

### `Data/`  

The data directory holds all the associated development files, and data that is shared between instances of passivbot such as API keys and historical data.

### `Dockerfile`

Unsure whether docker components need any changes, I am not well versed with docker, so I'm leaving in data for time being.

### `docker-compose.yml`

Unsure whether docker components need any changes, I am not well versed with docker, so I'm leaving in data for time being.

### `changelog.txt`  

Running list of all changes, notes by dev. Needs no changes.  

### `requirements.txt`  

Lists dependencies for the package. To my knowledge, this is only used in development environments, and should list ALL deps, including ones only used for testing.
Dependencies for the distribution version of the project are defined in setyp.py and only include what's necessary.

### `API_KEYS/`  

Moves the API keys up to a directory in the root folder. This is mostly a preference thing, and could be re-located to the
`bot/` directory depending on what's easier / more convenient.
The inner files don't need any changes for the refactor, although it may be worthwhile to eventually consider an updated naming scheme / subdirectories for users, assuming the UI is responsible for tracking, accessing, and editing multiple different users for different exchanges.

### `historical_data/`

Historical data has been placed in the data folder so that if there were ever multiple / different versions of the bot being distributed together, they can share some information.


### `bot/`  

The bot folder contains a given version of passivbot, the backtester, the configurations, logs, and historical data.
`API_KEYS/` could be moved inside of this folder, but I've left them in the root contingent on more feedback. If passiv ever includes multiple versions that operate differently, the `bot/` folder could be duplicated and named using version / purpose:

> `bot{ver}` --> `botV2.1` & `botV3.2`   

In this case, v2.1 and v3.2 can coexist, pull from the same API keys and historical data, but have separate configuration pools and different operating logic.
The end user then gets to select which version to use via the UI, and all the rest is done in the background.

Live settings, backtesting settings, backtesting info, and all of the major components of the bot are located here.

## Changelog  

- 3/30/21 - Created `Refactor` branch and made initial commit.
- 3/31/21 - Updated README.md (Changed proposed structure by moving `API_KEYS`)
