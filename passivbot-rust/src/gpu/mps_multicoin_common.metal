// Shared Apple Metal multi-coin screening primitives.
//
// Exact Rust backtests remain authoritative. EMA Anchor, Trailing Martingale,
// and the future joint-side portfolio kernel compose this module so rounding,
// fill accounting, override lookup, and exposure allowance cannot drift apart.

#ifndef PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS
#define PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS 0
#endif

inline float round_step(float value, float step) {
    return floor(value / step + 0.5f) * step;
}

inline float ceil_step(float value, float step) {
    return ceil(value / step - 1.0e-6f) * step;
}

inline float floor_step(float value, float step) {
    return floor(value / step + 1.0e-6f) * step;
}

inline float min_entry_qty(
    float price, float qty_step, float min_qty, float min_cost, float c_mult
) {
    float raw_min = fmax(min_qty, min_cost / fmax(price, 1.0e-12f) / c_mult);
    float raw_steps = raw_min / qty_step;
    float nearest_count = floor(raw_steps + 0.5f);
    float nearest = nearest_count * qty_step;
    float representation_tolerance = 1.1920928955078125e-7f
        * fmax(fabs(raw_min), fabs(nearest)) * 4.0f;
    bool aligned = nearest_count > 0.0f && fabs(raw_steps - nearest_count) <= 1.0e-8f
        && (nearest >= raw_min || raw_min - nearest <= representation_tolerance);
    return aligned ? fmax(nearest, raw_min) : ceil(raw_steps) * qty_step;
}

inline bool should_use_ordinary_market_execution(
    int order_ticks,
    bool buy_order,
    float market_price,
    float price_step,
    bool market_orders_allowed,
    float near_touch_threshold
) {
    if (!market_orders_allowed || order_ticks <= 0
        || !(market_price > 0.0f) || !isfinite(market_price)) {
        return false;
    }
    float order_price = float(order_ticks) * price_step;
    if (buy_order ? order_price >= market_price : order_price <= market_price) {
        return true;
    }
    return fabs(order_price / market_price - 1.0f)
        <= fmax(near_touch_threshold, 0.0f);
}

inline float ordinary_market_fill_price(
    float close,
    bool buy_order,
    float market_order_slippage_pct,
    float price_step
) {
    float slipped = close * (
        buy_order
            ? 1.0f + market_order_slippage_pct
            : 1.0f - market_order_slippage_pct
    );
    return fmax(
        buy_order ? ceil_step(slipped, price_step)
                  : floor_step(slipped, price_step),
        price_step
    );
}

inline float resize_market_close_qty(
    float requested_qty,
    float position_size,
    float executable_touch,
    float qty_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    if (!(requested_qty > 0.0f) || position_size <= requested_qty) {
        return requested_qty;
    }
    float minimum_qty = min_entry_qty(
        executable_touch, qty_step, min_qty, min_cost, c_mult
    );
    float tolerance = 1.0e-12f
        * fmax(requested_qty, minimum_qty) * 4.0f;
    if (requested_qty + tolerance >= minimum_qty) return requested_qty;
    float resized = fmin(minimum_qty, position_size);
    float remainder = position_size - resized;
    if (remainder > 0.0f && remainder + tolerance < minimum_qty) {
        resized = position_size;
    }
    return resized;
}

inline bool finite_positive(float value) {
    return isfinite(value) && value > 0.0f;
}

inline int multicoin_interval_minutes(float interval_ms) {
    return max(1, int(interval_ms / 60000.0f + 0.5f));
}

inline int multicoin_utc_day_index(
    int start_day_minute, int k, float interval_ms
) {
    return (start_day_minute + k * multicoin_interval_minutes(interval_ms))
        / 1440;
}

inline int multicoin_active_fill_day(
    int k, int first_eq_k, float interval_ms
) {
    return ((k - first_eq_k) * multicoin_interval_minutes(interval_ms)) / 1440;
}

inline float float32_floor_nonnegative(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return fmax(value, 0.0f);
    return as_type<float>(as_type<uint>(value) - 1u);
}

inline bool realized_loss_gate_allows(
    float net_pnl, float remaining_loss_budget, bool gate_enabled
) {
    return !gate_enabled || net_pnl >= 0.0f
        || -net_pnl <= remaining_loss_budget;
}

// Finite realized-PnL window used by auto-unstuck, independent of HSL.
struct RollingPnlWindow {
    int event_head;
    int event_count;
    int peak_head;
    int peak_count;
    float absolute_cumulative;
    bool overflowed;
};

struct RollingPnlSignal {
    float peak;
    float current;
};

inline RollingPnlWindow init_rolling_pnl_window() {
    RollingPnlWindow window;
    window.event_head = 0;
    window.event_count = 0;
    window.peak_head = 0;
    window.peak_count = 0;
    window.absolute_cumulative = 0.0f;
    window.overflowed = false;
    return window;
}

inline void reset_rolling_pnl_window(
    thread RollingPnlWindow& window
) {
    window.event_head = 0;
    window.event_count = 0;
    window.peak_head = 0;
    window.peak_count = 0;
}

inline void prune_rolling_pnl_window(
    thread RollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars
) {
    if (lookback_bars <= 0 || window.overflowed) return;
    while (window.event_count > 0) {
        int slot = window.event_head;
        if (k - indices[base + slot].x <= lookback_bars) break;
        if (window.peak_count > 0
            && indices[base + window.peak_head].y == slot) {
            window.peak_head = (window.peak_head + 1) % capacity;
            window.peak_count -= 1;
        }
        window.event_head = (window.event_head + 1) % capacity;
        window.event_count -= 1;
    }
}

inline void record_rolling_pnl(
    thread RollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars,
    bool active,
    float pnl
) {
    if (!active || lookback_bars <= 0 || window.overflowed) return;
    window.absolute_cumulative += pnl;
    prune_rolling_pnl_window(
        window, values, indices, base, capacity, k, lookback_bars
    );
    if (window.event_count > 0) {
        int slot = (window.event_head + window.event_count - 1) % capacity;
        if (indices[base + slot].x == k) {
            values[base + slot].y = fmax(
                values[base + slot].y, window.absolute_cumulative
            );
            if (window.peak_count > 0) {
                int peak_tail = (
                    window.peak_head + window.peak_count - 1
                ) % capacity;
                if (indices[base + peak_tail].y == slot) {
                    window.peak_count -= 1;
                }
            }
            while (window.peak_count > 0) {
                int back = (
                    window.peak_head + window.peak_count - 1
                ) % capacity;
                int peak_slot = indices[base + back].y;
                if (values[base + peak_slot].y
                    > values[base + slot].y) break;
                window.peak_count -= 1;
            }
            int peak_tail = (
                window.peak_head + window.peak_count
            ) % capacity;
            indices[base + peak_tail].y = slot;
            window.peak_count += 1;
            return;
        }
    }
    if (window.event_count >= capacity || window.peak_count >= capacity) {
        window.overflowed = true;
        return;
    }
    int slot = (window.event_head + window.event_count) % capacity;
    values[base + slot] = float2(
        window.absolute_cumulative - pnl, window.absolute_cumulative
    );
    indices[base + slot].x = k;
    window.event_count += 1;

    while (window.peak_count > 0) {
        int back = (window.peak_head + window.peak_count - 1) % capacity;
        int peak_slot = indices[base + back].y;
        if (values[base + peak_slot].y > window.absolute_cumulative) break;
        window.peak_count -= 1;
    }
    int peak_tail = (window.peak_head + window.peak_count) % capacity;
    indices[base + peak_tail].y = slot;
    window.peak_count += 1;
}

inline RollingPnlSignal effective_rolling_pnl(
    thread RollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars
) {
    RollingPnlSignal signal;
    signal.peak = 0.0f;
    signal.current = 0.0f;
    if (lookback_bars <= 0 || window.overflowed) return signal;
    prune_rolling_pnl_window(
        window, values, indices, base, capacity, k, lookback_bars
    );
    if (window.event_count == 0) return signal;
    float base_cumulative = values[base + window.event_head].x;
    int peak_slot = indices[base + window.peak_head].y;
    signal.current = window.absolute_cumulative - base_cumulative;
    signal.peak = fmax(
        values[base + peak_slot].y - base_cumulative,
        fmax(signal.current, 0.0f)
    );
    return signal;
}

// Joint-side account state for the fused multi-coin portfolio path. Exact
// Rust processes every long fill before every short fill for a candle; callers
// preserve that ordering while this state owns the one shared cash balance and
// the realized-PnL scopes consumed by liquidation, loss gates, and HSL.
struct JointPortfolioAccount {
    float balance;
    float realized_pnl_total;
    float realized_pnl_peak;
    float realized_pnl_long;
    float realized_pnl_short;
#if PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS > 0
    RollingPnlWindow unstuck_pnl;
    device float2* unstuck_pnl_values;
    device int2* unstuck_pnl_indices;
    int unstuck_pnl_base;
    int unstuck_pnl_k;
    float unstuck_pnl_drawdown;
#endif
};

inline JointPortfolioAccount init_joint_portfolio_account(
    float starting_balance
) {
    JointPortfolioAccount account;
    account.balance = starting_balance;
    account.realized_pnl_total = 0.0f;
    account.realized_pnl_peak = 0.0f;
    account.realized_pnl_long = 0.0f;
    account.realized_pnl_short = 0.0f;
#if PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS > 0
    account.unstuck_pnl = init_rolling_pnl_window();
    account.unstuck_pnl_values = nullptr;
    account.unstuck_pnl_indices = nullptr;
    account.unstuck_pnl_base = 0;
    account.unstuck_pnl_k = 0;
    account.unstuck_pnl_drawdown = 0.0f;
#endif
    return account;
}

inline void record_joint_portfolio_fill(
    thread JointPortfolioAccount& account,
    float net_pnl,
    bool is_long
) {
#if PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS > 0
    record_rolling_pnl(
        account.unstuck_pnl, account.unstuck_pnl_values,
        account.unstuck_pnl_indices, account.unstuck_pnl_base,
        PASSIVBOT_UNSTUCK_PNL_CAPACITY, account.unstuck_pnl_k,
        PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS, true, net_pnl
    );
#endif
    account.balance += net_pnl;
    account.realized_pnl_total += net_pnl;
    account.realized_pnl_peak = fmax(
        account.realized_pnl_peak, account.realized_pnl_total
    );
    if (is_long) account.realized_pnl_long += net_pnl;
    else account.realized_pnl_short += net_pnl;
}

// Auto-unstuck uses the configured fill-PnL window; HSL and the conservative
// realized-loss gate retain their independent accounting contracts.
inline float unstuck_pnl_drawdown(thread const JointPortfolioAccount& account) {
#if PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS > 0
    return account.unstuck_pnl_drawdown;
#else
    return account.realized_pnl_peak - account.realized_pnl_total;
#endif
}

#if PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS > 0
inline void bind_unstuck_pnl_window(
    thread JointPortfolioAccount& account,
    device float2* values,
    device int2* indices,
    int candidate
) {
    account.unstuck_pnl_values = values;
    account.unstuck_pnl_indices = indices;
    account.unstuck_pnl_base = candidate * PASSIVBOT_UNSTUCK_PNL_CAPACITY;
}

inline bool refresh_unstuck_pnl_window(thread JointPortfolioAccount& account) {
    RollingPnlSignal signal = effective_rolling_pnl(
        account.unstuck_pnl, account.unstuck_pnl_values,
        account.unstuck_pnl_indices, account.unstuck_pnl_base,
        PASSIVBOT_UNSTUCK_PNL_CAPACITY, account.unstuck_pnl_k,
        PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS
    );
    account.unstuck_pnl_drawdown = fmax(signal.peak - signal.current, 0.0f);
    return !account.unstuck_pnl.overflowed;
}
#endif

inline void record_realized_net(
    float net_pnl,
    thread JointPortfolioAccount& account,
    thread float& day_fill_count,
    thread float& fill_count,
    thread float& fill_count_entry,
    thread float& fill_count_long,
    thread float& pnl_recovery_peak,
    thread float& pnl_recovery_peak_k,
    thread float& pnl_recovery_max_min,
    float fill_k,
    bool is_entry,
    bool is_long
) {
    record_joint_portfolio_fill(account, net_pnl, is_long);
    day_fill_count += 1.0f;
    fill_count += 1.0f;
    if (is_entry) fill_count_entry += 1.0f;
    if (is_long) fill_count_long += 1.0f;
    if (account.realized_pnl_total > pnl_recovery_peak) {
        if (pnl_recovery_peak_k >= 0.0f) {
            pnl_recovery_max_min = fmax(
                pnl_recovery_max_min, fill_k - pnl_recovery_peak_k
            );
        }
        pnl_recovery_peak = account.realized_pnl_total;
        pnl_recovery_peak_k = fill_k;
    }
}

inline void record_gross_pnl(
    float pnl, thread float& profit_sum, thread float& loss_sum
) {
    if (pnl > 0.0f) profit_sum += pnl;
    else loss_sum += fabs(pnl);
}

// Backtest WEL denominator mode is fixed for a compiled proxy instance.
#ifndef PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY
#define PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY 1
#endif

inline int wallet_exposure_denominator_n_positions(
    int configured_n_positions, int observed_tradable_count
) {
#if PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY
    return min(configured_n_positions, observed_tradable_count);
#else
    return configured_n_positions;
#endif
}

inline float coin_override_or(
    constant float* coin_overrides, int coin, int column, float fallback
) {
    float value = coin_overrides[coin * OVERRIDE_COLS + column];
    return isfinite(value) ? value : fallback;
}

inline float allowed_wallet_exposure_limit(
    float base_limit, float total_limit, float allowance_pct, bool legacy_raw
) {
    if (!(isfinite(base_limit) && base_limit > 0.0f)) return 0.0f;
    float raw = fmax(allowance_pct, 0.0f);
    float effective = raw;
    if (!legacy_raw) {
        float max_effective = (
            isfinite(total_limit) && total_limit > 0.0f
        ) ? fmax(total_limit / base_limit - 1.0f, 0.0f) : 0.0f;
        effective = fmin(raw, max_effective);
    }
    return base_limit * (1.0f + effective);
}

inline bool passes_multicoin_min_effective_cost(
    bool enabled, float balance, float wel,
    float initial_qty_pct, float price,
    constant float* coin_settings, int coin_offset
) {
    if (!enabled) return true;
    if (!finite_positive(price)) return false;
    const float c_mult = coin_settings[coin_offset + 4];
    const float minimum_cost = min_entry_qty(
        price, coin_settings[coin_offset + 0], coin_settings[coin_offset + 2],
        coin_settings[coin_offset + 3], c_mult
    ) * price * c_mult;
    const float projected_cost = balance * wel * initial_qty_pct;
    return isfinite(projected_cost) && projected_cost > 0.0f
        && projected_cost >= minimum_cost;
}

inline float clamped_market_price(
    constant float* bars, constant float* coin_settings,
    int k, int coin, int coin_count
) {
    int coin_offset = coin * COIN_COLS;
    int first_valid = int(coin_settings[coin_offset + 6]);
    int last_valid = int(coin_settings[coin_offset + 7]);
    int market_k = clamp(k, first_valid, last_valid);
    return bars[(market_k * coin_count + coin) * 4 + 2];
}

inline float joint_portfolio_equity(
    thread const JointPortfolioAccount& account,
    float unrealized_pnl_long,
    float unrealized_pnl_short
) {
    return account.balance + unrealized_pnl_long + unrealized_pnl_short;
}

inline bool joint_portfolio_can_generate(
    thread const JointPortfolioAccount& account,
    float equity,
    float liquidation_floor
) {
    return isfinite(account.balance) && account.balance > 0.0f
        && isfinite(equity) && equity > liquidation_floor;
}

// Fused long+short multi-coin kernels own two independent pside controllers.
// Unified mode feeds the same shared account signal to both controllers while
// pside mode feeds directional realized and unrealized PnL. Coin mode has a
// separate per-coin controller topology and is deliberately rejected here.
inline bool update_joint_pside_hsl(
    thread HslState& long_hsl,
    thread HslState& short_hsl,
    thread const JointPortfolioAccount& account,
    float starting_balance,
    float unrealized_pnl_long,
    float unrealized_pnl_short,
    bool has_position_long,
    bool has_position_short,
    bool has_blocking_orders_long,
    bool has_blocking_orders_short,
    float kf,
    float interval_ms
) {
    if (long_hsl.signal_mode == HSL_SIGNAL_COIN
        || short_hsl.signal_mode == HSL_SIGNAL_COIN
        || long_hsl.signal_mode != short_hsl.signal_mode) return false;
    return update_dual_side_hsl(
        long_hsl, short_hsl, account.balance, starting_balance,
        account.realized_pnl_total,
        account.realized_pnl_long, account.realized_pnl_short,
        unrealized_pnl_long, unrealized_pnl_short,
        has_position_long, has_position_short,
        has_blocking_orders_long, has_blocking_orders_short,
        kf, interval_ms
    );
}

inline int joint_pside_hsl_global_tier(
    thread const HslState& long_hsl,
    thread const HslState& short_hsl
) {
    return max(long_hsl.tier, short_hsl.tier);
}

// Unheld unavailable coins need no valuation. Held positions must remain
// inside their declared range with finite positive H/L/C.
inline bool held_positions_have_missing_prices(
    thread const float* psize,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count
) {
    for (int coin = 0; coin < coin_count; ++coin) {
        if (!(psize[coin] > 0.0f)) continue;
        int settings = coin * COIN_COLS;
        if (k < int(coin_settings[settings + 6]) || k > int(coin_settings[settings + 7])) return true;
        int offset = (k * coin_count + coin) * 4;
        for (int field = 0; field < 3; ++field) {
            float value = bars[offset + field];
            if (!(isfinite(value) && value > 0.0f)) return true;
        }
    }
    return false;
}
