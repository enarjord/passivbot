// Shared Apple Metal multi-coin screening primitives.
//
// Exact Rust backtests remain authoritative. EMA Anchor, Trailing Martingale,
// and the future joint-side portfolio kernel compose this module so rounding,
// fill accounting, override lookup, and exposure allowance cannot drift apart.

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
    return account;
}

inline void record_joint_portfolio_fill(
    thread JointPortfolioAccount& account,
    float net_pnl,
    bool is_long
) {
    account.balance += net_pnl;
    account.realized_pnl_total += net_pnl;
    account.realized_pnl_peak = fmax(
        account.realized_pnl_peak, account.realized_pnl_total
    );
    if (is_long) account.realized_pnl_long += net_pnl;
    else account.realized_pnl_short += net_pnl;
}

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
    bool enabled, float guaranteed_balance_lower, float wel,
    float initial_qty_pct, float max_effective_min_cost
) {
    if (!enabled) return true;
    float rounded_projected_cost = guaranteed_balance_lower * wel * initial_qty_pct;
    // Discount by 16 float32 unit roundoffs. This covers upward encoding and
    // multiply rounding of all three operands before the conservative compare.
    float projected_cost_lower = rounded_projected_cost
        * (1.0f - 9.5367431640625e-7f);
    return isfinite(rounded_projected_cost) && rounded_projected_cost > 0.0f
        && projected_cost_lower >= max_effective_min_cost;
}

inline bool multicoin_min_cost_rejection_possible(
    thread const float* psize,
    thread HslState* coin_hsl,
    bool coin_hsl_mode,
    int current_hsl_mode,
    float twel,
    float allowance_pct,
    bool legacy_raw_allowance,
    float initial_qty_pct,
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    int wel_override_col,
    int allowance_override_col,
    int initial_qty_override_col,
    int k,
    int coin_count,
    int effective_n_positions,
    float guaranteed_balance_lower
) {
    if (effective_n_positions <= 0 || !(guaranteed_balance_lower > 0.0f)) {
        return false;
    }
    for (int c = 0; c < coin_count; ++c) {
        if (psize[c] > 0.0f) continue;
        const int coin_offset = c * COIN_COLS;
        const int bar_offset = (k * coin_count + c) * 4;
        const float coin_wel = coin_override_or(
            coin_overrides, c, wel_override_col, -1.0f
        );
        const int coin_mode = coin_hsl_mode
            ? hsl_mode(coin_hsl[c], false) : current_hsl_mode;
        if (k < int(coin_settings[coin_offset + 8])
            || k > int(coin_settings[coin_offset + 7])
            || !finite_positive(bars[bar_offset + 2])
            || coin_wel == 0.0f
            || coin_mode != 0) {
            continue;
        }
        const float base_limit = coin_wel >= 0.0f
            ? coin_wel : twel / fmax(float(effective_n_positions), 1.0f);
        const float allowed_wel = allowed_wallet_exposure_limit(
            base_limit, twel,
            coin_override_or(
                coin_overrides, c, allowance_override_col, allowance_pct
            ),
            legacy_raw_allowance
        );
        if (!passes_multicoin_min_effective_cost(
            true, guaranteed_balance_lower, allowed_wel,
            coin_override_or(
                coin_overrides, c, initial_qty_override_col,
                initial_qty_pct
            ),
            coin_settings[coin_offset + 12]
        )) {
            return true;
        }
    }
    return false;
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

inline float joint_hsl_realized_pnl(
    thread const JointPortfolioAccount& account,
    bool unified,
    bool is_long
) {
    if (unified) return account.realized_pnl_total;
    return is_long
        ? account.realized_pnl_long : account.realized_pnl_short;
}

inline float joint_hsl_unrealized_pnl(
    float unrealized_pnl_long,
    float unrealized_pnl_short,
    bool unified,
    bool is_long
) {
    if (unified) return unrealized_pnl_long + unrealized_pnl_short;
    return is_long ? unrealized_pnl_long : unrealized_pnl_short;
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

inline void try_restart_joint_pside_hsl(
    thread HslState& long_hsl,
    thread HslState& short_hsl,
    thread const JointPortfolioAccount& account,
    float starting_balance,
    float unrealized_pnl_long,
    float unrealized_pnl_short,
    float kf
) {
    if (long_hsl.signal_mode == HSL_SIGNAL_COIN
        || short_hsl.signal_mode == HSL_SIGNAL_COIN
        || long_hsl.signal_mode != short_hsl.signal_mode) return;
    const bool unified = long_hsl.signal_mode == HSL_SIGNAL_UNIFIED;
    float long_equity = starting_balance
        + joint_hsl_realized_pnl(account, unified, true)
        + joint_hsl_unrealized_pnl(
            unrealized_pnl_long, unrealized_pnl_short, unified, true
        );
    float short_equity = starting_balance
        + joint_hsl_realized_pnl(account, unified, false)
        + joint_hsl_unrealized_pnl(
            unrealized_pnl_long, unrealized_pnl_short, unified, false
        );
    try_restart_hsl(long_hsl, kf, long_equity);
    try_restart_hsl(short_hsl, kf, short_equity);
}

inline int joint_pside_hsl_global_tier(
    thread const HslState& long_hsl,
    thread const HslState& short_hsl
) {
    return max(long_hsl.tier, short_hsl.tier);
}
