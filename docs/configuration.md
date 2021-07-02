# Configuration

In order to configure Passivbot, you will need to provide a json file when starting the bot.
These config files are typically stored in the `config/live`.

!!! Info
    The configuration of telegram is not covered in this chapter. Please refer to [Telegram](telegram.md) for configuring Telegram.

##Configuration options

In the configuration file provided to run live, the parameters mentioned in the table can be set.
The configuration is split in 4 categories for clarity (user, initial entry, reentry, taking profit, stoploss).
A more detailed explanation on each configuration option is given later on in this chapter.

It's important to realize that most of the configuration is split across a configuration for short positions and long positions.
Configuration parameters that this applies to, are prepended with `short§long:` in the configuration table below.

!!! Info
    The config file can be quite overwhelming when looking at it for the first time. You can have a look at
    the configuration files provided in the repository by default to get a feel for config files.

| Parameter               | Description
| ---------------------   | ------------- |
| `config_name`           | A user-defined identification of the file<br/>**Category:** User<br/>**Datatype:** String 
| `logging_level`         | Indication if logging is to be written to file<br/>**0**: Don't write to logfile<br/>**1**: Write to logfile<br/>**Category:** User<br/>**Datatype:** Integer
| `short§long:enabled`    | Enables/disables the applicable position side<br/>**Category:** User<br/>**Datatype:** Boolean
| `allow_sharing_wallet`  | Indicates if the bot is allowed to start when a position already exists on another symbol<br/>**Category:** User<br/>**Datatype:** Boolean
| `n_spans`               | Number of spans used to determine initial entry<br/>**Category:** Initial entry<br/>**Datatype:** Integer
| `max_spans`             | Maximum number of ticks used in MA spans<br/>**Category:** Initial entry<br/>**Datatype:** Float
| `min_spans`             | Minimum number of ticks used in MA spans<br/>**Category:** Initial entry<br/>**Datatype:** Float
| `short§long:pbr_limit`  | Position cost to balance ratio limit<br/>**Category:** Initial entry<br/>**Datatype:** Float
| `short§long:iprc_MAr_coeffs` | Initial price Mean Average coefficients<br/>**Category:** Initial entry<br/>**Datatype:** [[Float,Float]..]
| `short§long:iprc_const` | Initial price constant<br/>**Category:** Initial entry<br/>**Datatype:** Float
| `short§long:iqty_const` | Initial quantity percentage of balance<br/>**Category:** Initial entry<br/>**Datatype:** Float
| `short§long:iqty_MAr_coeffs` | Initial quantity Mean Average coefficients<br/>**Category:** Initial entry<br/>**Datatype:** [[Float, Float]..]
| `short§long:rprc_const` | Reentry price constant<br/>**Category:** Reentry<br/>**Datatype:** Float
| `short§long:rqty_const` | Reentry quantity constant<br/>**Category:** Reentry<br/>**Datatype:** Float
| `short§long:rprc_PBr_coeffs` | Reentry position cost to balance ratio coefficients<br/>**Category:** Reentry<br/>**Datatype:** [[Float, Float]..]
| `short§long:rqty_MAr_coeffs` | Reentry quantity Mean Average coefficients<br/>**Category:** Reentry<br/>**Datatype:** [[Float, Float]..]
| `short§long:rprc_MAr_coeffs` | Reentry price Mean Average coefficients<br/>**Category:** Reentry<br/>**Datatype:** [[Float, Float]..]
| `short§long:markup_const` | Profit markup constant<br/>**Category:** Taking profit<br/>**Datatype:** <br/>**Datatype:** Float
| `short§long:markup_MAr_coeffs` | Markup Mean Average coefficients<br/>**Category:** Taking profit<br/>**Datatype:** [[Float, Float]..]
| `short§long:pbr_stop_loss` | Position cost to balance ratio stoploss<br/>**Category:** Stoploss<br/>**Datatype:** <br/>**Datatype:** Float
| `profit_trans_pct`      | Percentage indicating how much profit should be transferred to Spot wallet on each order filled<br/>**Category:** Closing<br/>**Datatype:** Float

## Initial trade entry

## Reentry

## Closing trades

### Taking profit

### Limiting loss

## Logging

When running, Passivbot writes information to the console by default. It is possible to have the bot write this information
to a log file for easier access and to allow looking up information historically. This information includes orders being created, cancelled, positions and open orders. 

If the configuration parameter `logging_level` is set to `0`, the aforementioned information will **not** be written to a logfile.
If the parameter is set to `1`, the information **will** be written to a logfile.

If writing the logs to a file is enabled, the information is stored in a file at `logs/{exchange}/{config_name}.log`.
The config_name is the value that is specified in the `config_name` parameter in the configuration file.

!!! Warning
    Be aware that Windows can have a limitation on the maximum path length of a file. If you run into a problem with this,
    you can try moving Passivbot to a different location that results in a shorter pathname, or a shorter value in the `config_name` parameter.
