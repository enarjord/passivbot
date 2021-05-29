# Configuration

##Live configuration

In the configuration file provided to run live, the following parameters can be set |

| Parameter               | Category      | Explanation                              |
| ---------------------   | ------------- | ---------------------------------------- |
| `config_name`           | User          | A user-defined identification of the file
| `logging_level`         | User          |
| `ddown_factor`          | Reentry       |
| `qty_pct`               | Initial entry | The percentage of the equity that is used for initial entry
| `leverage`              | Initial entry | The leverage that is applied when a position is opened
| `n_close_orders`        | Closing       |
| `grid_spacing`          | Reentry       |
| `pos_margin_grid_coeff` | Reentry       |
| `volatility_grid_coeff` | Reentry       |
| `volatility_qty_coeff`  | Reentry       |
| `min_markup`            | Reentry       |
| `markup_range`          | Reentry       |
| `do_long`               | General       | Boolean indicating if the bot trades in long positions
| `do_shrt`               | General       | Boolean indicating if the bot trades in short positions
| `ema_span`              | Initial entry |
| `ema_spread`            | Initial entry |
| `stop_loss_liq_diff`    | Closing       |
| `stop_loss_pos_pct`     | Closing       |
| `entry_liq_diff_thr`    | Reentry       | The closest liquidation difference that an order is allowed to bring the position