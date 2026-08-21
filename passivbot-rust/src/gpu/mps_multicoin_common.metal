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

inline bool finite_positive(float value) {
    return isfinite(value) && value > 0.0f;
}

inline float float32_floor_nonnegative(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return fmax(value, 0.0f);
    return as_type<float>(as_type<uint>(value) - 1u);
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
