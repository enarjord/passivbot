// Shared Apple Metal multi-coin screening primitives.
//
// Exact Rust backtests remain authoritative. EMA Anchor, Trailing Martingale,
// and the future joint-side portfolio kernel compose this module so rounding,
// fill accounting, override lookup, and exposure allowance cannot drift apart.



// Ordinary market EMAs finalize one normalized, causal warmup scalar.
// source_kind: 0 close, 1 minute log range, 2 quote volume, 3 completed hour.
// Risk decay and directionality RMS use their separate initialization contracts.
inline float bootstrap_market_ema(
    constant float* bars, constant float* hour_log_ranges,
    int k, int first_valid, int coin, int coin_count, float span, int requested_count, int source_kind,
    thread bool& observed
) {
    observed = false;
    if (!(isfinite(span) && span > 0.0f) || k < first_valid) return 0.0f;
    const float alpha = clamp(2.0f / (span + 1.0f), 0.0f, 1.0f);
    const float beta = 1.0f - alpha;
    const int capacity = min(requested_count, k - first_valid + 1);
    int start = k;
    int count = 0;
    for (int j = k; j >= first_valid && count < capacity; --j) {
        if (source_kind == 3 && hour_log_ranges[j * coin_count + coin] < 0.0f) continue;
        start = j;
        ++count;
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int j = start; j <= k && count > 0; ++j) {
        float sample;
        if (source_kind == 3) {
            sample = hour_log_ranges[j * coin_count + coin];
            if (sample < 0.0f) continue;
        } else {
            const int offset = (j * coin_count + coin) * 4;
            const float high = bars[offset], low = bars[offset + 1], close = bars[offset + 2];
            sample = source_kind == 0 ? close : (source_kind == 1
                ? log(high / low) : fmax(bars[offset + 3], 0.0f) * ((high + low + close) / 3.0f));
        }
        numerator = fma(beta, numerator, sample);
        denominator = fma(beta, denominator, 1.0f);
    }
    observed = denominator > 0.0f;
    return observed ? numerator / denominator : 0.0f;
}

#if PASSIVBOT_RAW_STRATEGY_RISK_ENABLED
// Observational raw portfolio risk; account liquidation values remain separate.
inline void update_raw_strategy_risk(
    float equity, thread float& peak, thread float& daily_worst
) {
    peak = fmax(peak, equity);
    daily_worst = fmax(daily_worst,
        (peak - equity) / fmax(fabs(peak), 1.0e-12f));
}
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

// The caller already quantized quantity. Subtract in venue units so binary
// subtraction followed by floor cannot accidentally remove a second increment.
// Above float32's exact integer range, retain a representably smaller quantity;
// the conservative reduction can span several venue steps.
inline float decrement_aligned_qty_step(float quantity, float step) {
    float units = rint(quantity / step);
    float previous_units = fmax(units - 1.0f, 0.0f);
    if (previous_units >= units && units > 0.0f) {
        previous_units = nextafter(units, 0.0f);
    }
    float result = previous_units * step;
    // The multiply can round back up even when the unit count decreased.
    return fmax(fmin(result, nextafter(quantity, 0.0f)), 0.0f);
}

inline float strict_cap_quantity(
    float quantity, float step, float retained_cost, float cap_cost,
    float gate_price, float c_mult
) {
    if (retained_cost + quantity * gate_price * c_mult < cap_cost)
        return quantity;
    quantity = decrement_aligned_qty_step(quantity, step);
    if (retained_cost + quantity * gate_price * c_mult < cap_cost)
        return quantity;
    // A retained-cost ULP can cover millions of venue increments. Search the
    // ordered positive float encodings of unit counts, rather than decrementing
    // each increment. Floor maps fractional count encodings to whole units.
    // Positive multiplication/addition and this map are monotone. At most 31
    // bisections cover every nonnegative finite float count (including an
    // infinite upper sentinel); every accepted candidate satisfies the cap.
    uint lower = 0;
    uint upper = as_type<uint>(fmax(rint(quantity / step), 0.0f));
    while (lower < upper) {
        uint middle = lower + (upper - lower + 1) / 2;
        float candidate = floor(as_type<float>(middle)) * step;
        bool accepted = candidate <= quantity
            && retained_cost + candidate * gate_price * c_mult < cap_cost;
        if (accepted) lower = middle;
        else upper = middle - 1;
    }
    return floor(as_type<float>(lower)) * step;
}

// Quantities produced by venue rounding can recover their integer counts.
// Require a unique round-trip before subtracting counts. Leave the caller's
// physical difference unchanged when the proof is unavailable.
inline bool try_aligned_qty_difference(
    float total, float held, float step, thread float& difference
) {
    float total_count = rint(total / step);
    float held_count = rint(held / step);
    bool counts_resolvable = total_count >= 0.0f && total_count < 0x1p24f
        && held_count >= 0.0f && held_count < 0x1p24f
        && total - nextafter(total, -INFINITY) <= step
        && nextafter(held, INFINITY) - held <= step;
    bool on_grid = total_count * step == total && held_count * step == held;
    if (counts_resolvable && on_grid) {
        difference = fmax(total_count - held_count, 0.0f) * step;
        return true;
    }
    return false;
}

inline float aligned_qty_difference(float total, float held, float step) {
    float difference;
    if (try_aligned_qty_difference(total, held, step, difference)) return difference;
    return fmax(floor_step(total - held, step), 0.0f);
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

inline int market_execution_tick_at(
    constant int* market_execution_ticks,
    int k, int coin, int coin_count, bool buy_order
) {
    return market_execution_ticks[(k * coin_count + coin) * 2 + int(buy_order)];
}

inline int clamped_market_execution_tick(
    constant int* market_execution_ticks, constant float* coin_settings,
    int k, int coin, int coin_count, bool buy_order
) {
    int offset = coin * COIN_COLS;
    int market_k = clamp(k, int(coin_settings[offset + 6]), int(coin_settings[offset + 7]));
    return market_execution_tick_at(market_execution_ticks, market_k, coin, coin_count, buy_order);
}

inline float ordinary_market_fill_price(int execution_tick, float price_step) {
    // Native market execution clamps only its effective step to f64::EPSILON.
    // max after f32 encoding is identical to encoding max(original f64 step, 2^-52).
    const float effective_step = fmax(price_step, 2.2204460492503131e-16f);
    return float(execution_tick) * effective_step;
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
struct RealizedDecayMemory {
    float beta;
    float drawdown;
    long minute;
    bool enabled;
};

struct JointPortfolioAccount {
    float balance;
    float balance_correction;
    float realized_pnl_total;
    float realized_pnl_long;
    float realized_pnl_short;
    RealizedDecayMemory unstuck_memory;
    RealizedDecayMemory loss_gate_memory;
    int realized_decay_interval_minutes;
    bool realized_decay_valid;
};

inline JointPortfolioAccount init_joint_portfolio_account(
    float starting_balance
) {
    JointPortfolioAccount account;
    account.balance = starting_balance;
    account.balance_correction = 0.0f;
    account.realized_pnl_total = 0.0f;
    account.realized_pnl_long = 0.0f;
    account.realized_pnl_short = 0.0f;
    account.unstuck_memory.beta = account.loss_gate_memory.beta = 0.0f;
    account.unstuck_memory.drawdown = account.loss_gate_memory.drawdown = 0.0f;
    account.unstuck_memory.minute = account.loss_gate_memory.minute = 0;
    account.unstuck_memory.enabled = account.loss_gate_memory.enabled = false;
    account.realized_decay_interval_minutes = 1;
    account.realized_decay_valid = true;
    return account;
}

// Each policy consumes every actual account net fill once. Independent rates
// own independent state; projected losses reserve allowance without changing it.
inline bool joint_realized_loss_allowance(
    thread JointPortfolioAccount& account, float fraction, thread float& allowance
) {
    if (!hsl_decay_realized_allowance(
            account.balance, fraction, account.loss_gate_memory.drawdown, allowance)) {
        account.realized_decay_valid = false;
        return false;
    }
    return true;
}

inline void configure_joint_realized_memory(
    thread JointPortfolioAccount& account, float unstuck_span, float loss_gate_span,
    bool unstuck_enabled, bool loss_gate_enabled, int interval_minutes
) {
    float alpha;
    account.realized_decay_valid = account.realized_decay_valid
        && hsl_decay_rate(unstuck_span, alpha, account.unstuck_memory.beta)
        && hsl_decay_rate(loss_gate_span, alpha, account.loss_gate_memory.beta);
    account.unstuck_memory.enabled = unstuck_enabled;
    account.loss_gate_memory.enabled = loss_gate_enabled;
    account.realized_decay_interval_minutes = interval_minutes;
}

#pragma STDC FP_CONTRACT OFF
inline void advance_realized_memory(
    thread RealizedDecayMemory& memory, long minute,
    thread bool& valid
) {
    if (!memory.enabled) return;
    if (minute < memory.minute) { valid = false; return; }
    for (long m = memory.minute; m < minute; ++m)
        memory.drawdown = fma(memory.beta, memory.drawdown, 0.0f);
    memory.minute = minute;
}

inline void advance_joint_realized_memory(thread JointPortfolioAccount& account, int bar_index) {
    const long minute = long(bar_index) * long(account.realized_decay_interval_minutes);
    advance_realized_memory(account.unstuck_memory, minute, account.realized_decay_valid);
    advance_realized_memory(account.loss_gate_memory, minute, account.realized_decay_valid);
}
#pragma STDC FP_CONTRACT ON

inline void record_joint_portfolio_fill(
    thread JointPortfolioAccount& account,
    float net_pnl,
    bool is_long
) {
    if (account.unstuck_memory.enabled) {
        account.realized_decay_valid = account.realized_decay_valid
            && hsl_decay_realized_fill(net_pnl, account.unstuck_memory.drawdown);
    }
    if (account.loss_gate_memory.enabled) {
        account.realized_decay_valid = account.realized_decay_valid
            && hsl_decay_realized_fill(net_pnl, account.loss_gate_memory.drawdown);
    }
    // Keep small fees/profits from disappearing into the larger cash balance.
    // The residual travels with account state through temporal dispatches.
    float adjusted = net_pnl - account.balance_correction;
    float next = account.balance + adjusted;
    account.balance_correction = (next - account.balance) - adjusted;
    account.balance = next;
    account.realized_pnl_total += net_pnl;
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
    advance_joint_realized_memory(account, int(fill_k));
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

#ifndef PASSIVBOT_COIN_OVERRIDES_ABSENT
#define PASSIVBOT_COIN_OVERRIDES_ABSENT 0
#endif

inline float coin_override_or(
    constant float* coin_overrides, int coin, int column, float fallback
) {
#if PASSIVBOT_COIN_OVERRIDES_ABSENT
    return fallback;
#else
    float value = coin_overrides[coin * OVERRIDE_COLS + column];
    return isfinite(value) ? value : fallback;
#endif
}

#pragma STDC FP_CONTRACT OFF
inline float allowed_wallet_exposure_limit(
    float base_limit, float total_limit, float allowance_pct) {
    if (!(isfinite(base_limit) && base_limit > 0.0f)) return 0.0f;
    float raw = fmax(allowance_pct, 0.0f);
    float max_effective = (isfinite(total_limit) && total_limit > 0.0f)
        ? fmax(total_limit / base_limit - 1.0f, 0.0f) : 0.0f;
    float effective = fmin(raw, max_effective);
    return base_limit * (1.0f + effective);
}

inline float coin_hsl_budget_multiplier(
    bool scale, float twel, int slots, constant float* overrides,
    int coin, int wel_col, int allowance_col, float allowance
) {
    scale = coin_override_or(overrides, coin, HSL_OVERRIDE_START + 7,
                             float(scale)) > 0.5f;
    if (!scale || slots <= 0) return 1.0f;
    float authored_wel = coin_override_or(overrides, coin, wel_col, -1.0f);
    float base = authored_wel >= 0.0f ? authored_wel : twel / float(slots);
    if (!(isfinite(base) && base > 0.0f)) return 1.0f;
    return allowed_wallet_exposure_limit(base, twel,
        coin_override_or(overrides, coin, allowance_col, allowance)) / base;
}

// Terminal accounting must use the same configured headroom as mark observations.
inline void prepare_coin_hsl_budgets(
    thread HslState* controllers, bool coin_mode, bool scale, float twel,
    int slots, constant float* overrides, int coins, int wel_col,
    int allowance_col, float allowance
) {
#if !PASSIVBOT_HSL_DISABLED
    if (!coin_mode || slots <= 0) return;
    for (int c = 0; c < coins; ++c) {
        controllers[c].slot_count = float(slots);
        controllers[c].budget_multiplier = coin_hsl_budget_multiplier(
            scale, twel, slots, overrides, c, wel_col, allowance_col, allowance);
    }
#endif
}

#pragma STDC FP_CONTRACT ON

inline bool passes_multicoin_min_effective_cost(
    bool enabled, float balance, float wel,
    float initial_qty_pct, float price,
    constant float* coin_settings ORGANILLO_ARGS, int coin_offset
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
    constant float* bars, constant float* coin_settings ORGANILLO_ARGS,
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

inline int joint_hsl_report_tier(
    thread const HslState& long_hsl,
    thread const HslState& short_hsl
) {
    return max(hsl_report_tier(long_hsl), hsl_report_tier(short_hsl));
}

// Unheld unavailable coins need no valuation. Held positions must remain
// inside their declared range with finite positive H/L/C.
inline bool held_positions_have_missing_prices(
    thread const float* psize,
    constant float* bars,
    constant float* coin_settings ORGANILLO_ARGS,
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

// Compact candle-level moments used by the native CUDA result contract.
#ifndef PASSIVBOT_EXPOSURE_RMS_ENABLED
#define PASSIVBOT_EXPOSURE_RMS_ENABLED 0
#endif
#if PASSIVBOT_EXPOSURE_RMS_ENABLED
struct ExposureRmsState {
    float sum[7];
    float correction[7];
    uint samples;
    float base_wel[2 * MAX_COINS];
};
inline ExposureRmsState init_exposure_rms() {
    ExposureRmsState state;
    state.samples = 0;
    for (int i = 0; i < 7; ++i) { state.sum[i] = 0.0f; state.correction[i] = 0.0f; }
    for (int i = 0; i < 2 * MAX_COINS; ++i) state.base_wel[i] = 0.0f;
    return state;
}
inline void update_exposure_rms_budgets(
    thread ExposureRmsState& state, constant float* bars,
    constant float* settings, constant float* overrides,
    int k, int coins, int slots, bool short_side, float twel, int wel_col
) {
    for (int c = 0; c < coins; ++c) {
        const int offset = c * COIN_COLS;
        const float authored = coin_override_or(overrides, c, wel_col, -1.0f);
        // Runtime WEL updates follow tradability, independently of selection,
        // minimum cost, HSL, and whether a next entry order was generated.
        if (k >= int(settings[offset + 8]) && k <= int(settings[offset + 7])
            && finite_positive(bars[(k * coins + c) * 4 + 2]) && authored != 0.0f) {
            state.base_wel[int(short_side) * MAX_COINS + c] = authored >= 0.0f
                ? authored : (slots > 0 ? twel / float(slots) : 0.0f);
        }
    }
}
inline float exposure_rms_position_moment(float e, float wel, float limit) {
    if (e == 0.0f) return 0.0f;
    return finite_positive(e) && finite_positive(wel) && finite_positive(limit)
        ? e * (e / wel) / limit : NAN;
}
inline float exposure_rms_ratio(float exposure, float limit) {
    if (exposure == 0.0f) return 0.0f;
    return finite_positive(exposure) && finite_positive(limit) ? exposure / limit : NAN;
}
inline void record_exposure_rms(thread ExposureRmsState& state,
    float long_exposure, float short_exposure, float long_moment, float short_moment,
    float long_limit, float short_limit
) {
    const float long_ratio = exposure_rms_ratio(long_exposure, long_limit);
    const float short_ratio = exposure_rms_ratio(short_exposure, short_limit);
    const float gross = long_exposure + short_exposure;
    const float values[7] = {long_exposure * long_exposure, short_exposure * short_exposure,
        gross * gross, long_ratio * long_ratio, short_ratio * short_ratio,
        long_moment, short_moment};
    ++state.samples;
    for (int i = 0; i < 7; ++i) {
        // Compensated sums prevent long float32 histories losing small samples.
        const float y = values[i] - state.correction[i];
        const float next = state.sum[i] + y;
        state.correction[i] = (next - state.sum[i]) - y;
        state.sum[i] = next;
    }
}
inline void finish_exposure_rms(thread const ExposureRmsState& state,
    device float* scalars, int offset
) {
    for (int i = 0; i < 7; ++i)
        scalars[offset + i] = state.samples > 0 ? sqrt(state.sum[i] / float(state.samples)) : NAN;
}
#endif
