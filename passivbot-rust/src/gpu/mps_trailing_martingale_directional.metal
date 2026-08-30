#include <metal_stdlib>
using namespace metal;

#ifndef PASSIVBOT_TM_TRAILING_ENTRY_ONLY
#define PASSIVBOT_TM_TRAILING_ENTRY_ONLY 0
#endif

#ifndef PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY
#define PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY 0
#endif
#ifndef PASSIVBOT_TM_TRAILING_CLOSE_ONLY
#define PASSIVBOT_TM_TRAILING_CLOSE_ONLY 0
#endif
#ifndef PASSIVBOT_TM_REDUCERS_DISABLED
#define PASSIVBOT_TM_REDUCERS_DISABLED 0
#endif
#ifndef PASSIVBOT_TM_MARKET_ORDERS_DISABLED
#define PASSIVBOT_TM_MARKET_ORDERS_DISABLED 0
#endif
#ifndef PASSIVBOT_TM_LOSS_GATE_DISABLED
#define PASSIVBOT_TM_LOSS_GATE_DISABLED 0
#endif
#ifndef PASSIVBOT_TM_VOLATILITY_DISABLED
#define PASSIVBOT_TM_VOLATILITY_DISABLED 0
#endif

#if PASSIVBOT_BTC_RISK_ENABLED
constant int DAILY_COLS = 11;
#else
constant int DAILY_COLS = 8;
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
constant int SCALAR_COLS = 72;
#elif PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
constant int SCALAR_COLS = 70;
#elif PASSIVBOT_HSL_EMA_TAIL_ENABLED
constant int SCALAR_COLS = 68;
#else
constant int SCALAR_COLS = 66;
#endif
constant int GAP_BINS = 128;
constant int SIDE_PARAMS = 52;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
constant float RECOVERY_FAIL_CLOSED_SENTINEL = -3.402823466e+38f;
#endif

inline int elapsed_fill_day_bucket(float k, float first_eq_k, float interval_ms) {
    const float fill_day_candles = 86400000.0f / interval_ms;
    return int((k - first_eq_k) / fill_day_candles);
}

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
    float market_order_near_touch_threshold
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
        <= fmax(market_order_near_touch_threshold, 0.0f);
}

inline float ordinary_market_fill_price(
    float market_price,
    bool buy_order,
    float market_order_slippage_pct,
    float price_step
) {
    float slipped = market_price * (
        buy_order
            ? 1.0f + market_order_slippage_pct
            : 1.0f - market_order_slippage_pct
    );
    return fmax(
        buy_order ? ceil_step(slipped, price_step) : floor_step(slipped, price_step),
        price_step
    );
}

inline float resize_market_close_qty(
    float qty,
    float psize,
    float market_price,
    float qty_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    if (!(qty > 0.0f) || !(psize > 0.0f)) return 0.0f;
    float minimum = min_entry_qty(
        market_price, qty_step, min_qty, min_cost, c_mult
    );
    float tolerance = 1.0e-12f * fmax(qty, minimum) * 4.0f;
    if (qty + tolerance >= minimum || psize <= qty) return fmin(qty, psize);
    float resized = fmin(minimum, psize);
    float remainder = psize - resized;
    if (remainder > 0.0f && remainder + tolerance < minimum) resized = psize;
    return resized;
}

inline bool passes_min_effective_cost(
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

inline bool realized_loss_proxy_allows_close(
    float qty, float close_price, float pprice, bool is_long,
    float c_mult, float fee_rate, bool gate_enabled
) {
    if (!gate_enabled) return true;
    if (!(qty > 0.0f && close_price > 0.0f && pprice > 0.0f)) return false;
    float gross_pnl = qty * c_mult
        * (is_long ? close_price - pprice : pprice - close_price);
    float fee = qty * close_price * c_mult * fee_rate;
    float net_pnl = gross_pnl - fee;
    // Enumerating and reserving the recursive TM close ladder at every candle
    // would make screening proportional to its 500-rung exact upper bound.
    // Instead, the proxy uses a zero-loss envelope whenever Rust's gate is
    // active. Require 1024 float32 unit roundoffs: twice the recursive rung
    // bound, covering accumulated position-price drift plus this projection,
    // so a rounded break-even result is not admitted as non-loss-making.
    // Exact Rust remains authoritative.
    float arithmetic_scale = fabs(gross_pnl) + fabs(fee)
        + qty * fabs(c_mult) * (fabs(close_price) + fabs(pprice));
    float margin = 1.220703125e-4f * arithmetic_scale;
    return isfinite(net_pnl) && net_pnl > margin;
}

inline float float32_floor_nonnegative(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return fmax(value, 0.0f);
    return as_type<float>(as_type<uint>(value) - 1u);
}

inline bool realized_loss_proxy_allows_reducer(
    float qty,
    float close_price,
    float pprice,
    bool is_long,
    bool is_unstuck,
    float c_mult,
    float fee_rate,
    bool gate_enabled,
    float balance,
    float realized_pnl_cumsum_last,
    float realized_pnl_cumsum_max,
    float max_realized_loss_pct
) {
    if (!gate_enabled) return true;
    if (!is_unstuck) {
        return realized_loss_proxy_allows_close(
            qty, close_price, pprice, is_long,
            c_mult, fee_rate, true
        );
    }
    if (!(qty > 0.0f && close_price > 0.0f && pprice > 0.0f)) return false;
    float gross_pnl = qty * c_mult
        * (is_long ? close_price - pprice : pprice - close_price);
    float fee = qty * close_price * c_mult * fee_rate;
    float net_pnl = gross_pnl - fee;
    if (!isfinite(net_pnl)) return false;
    if (net_pnl >= 0.0f) return true;
    float balance_peak = balance
        + (realized_pnl_cumsum_max - realized_pnl_cumsum_last);
    float allowed_loss_budget = float32_floor_nonnegative(
        balance_peak * fmax(max_realized_loss_pct, 0.0f)
    );
    float current_realized_loss = fmax(
        realized_pnl_cumsum_max - realized_pnl_cumsum_last, 0.0f
    );
    float remaining_loss_budget = float32_floor_nonnegative(
        fmax(allowed_loss_budget - current_realized_loss, 0.0f)
    );
    float arithmetic_scale = fabs(gross_pnl) + fabs(fee)
        + qty * fabs(c_mult) * (fabs(close_price) + fabs(pprice));
    float margin = 1.220703125e-4f * arithmetic_scale;
    return -net_pnl + margin <= remaining_loss_budget;
}

// PASSIVBOT_HSL_COMMON

// PASSIVBOT_BTC_RISK_COMMON

// PASSIVBOT_EQUITY_BALANCE_DIFF_COMMON

// PASSIVBOT_ENTRY_INTERVAL_COMMON

inline void record_realized_net(
    float net_pnl,
    thread float& realized_pnl_cumsum_last,
    thread float& realized_pnl_cumsum_max,
    thread float& realized_pnl_cumsum_long,
    thread float& realized_pnl_cumsum_short,
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
    day_fill_count += 1.0f;
    fill_count += 1.0f;
    if (is_entry) fill_count_entry += 1.0f;
    if (is_long) fill_count_long += 1.0f;
    realized_pnl_cumsum_last += net_pnl;
    if (is_long) realized_pnl_cumsum_long += net_pnl;
    else realized_pnl_cumsum_short += net_pnl;
    realized_pnl_cumsum_max = fmax(
        realized_pnl_cumsum_max, realized_pnl_cumsum_last
    );
    if (realized_pnl_cumsum_last > pnl_recovery_peak) {
        if (pnl_recovery_peak_k >= 0.0f) {
            pnl_recovery_max_min = fmax(
                pnl_recovery_max_min, fill_k - pnl_recovery_peak_k
            );
        }
        pnl_recovery_peak = realized_pnl_cumsum_last;
        pnl_recovery_peak_k = fill_k;
    }
}

inline void record_gross_pnl(
    float pnl, thread float& profit_sum, thread float& loss_sum
) {
    if (pnl > 0.0f) profit_sum += pnl;
    else loss_sum += fabs(pnl);
}

inline void record_directional_gross_pnl(
    float pnl,
    thread float& profit_sum,
    thread float& loss_sum,
    thread float& side_profit_sum,
    thread float& side_loss_sum
) {
    record_gross_pnl(pnl, profit_sum, loss_sum);
    record_gross_pnl(pnl, side_profit_sum, side_loss_sum);
}

struct TmSide {
    float alpha0, alpha1, alpha2;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    float alpha1m, alpha1h;
#endif
    float ddf, initial_ema_dist, initial_qty_pct;
    float entry_threshold_base, entry_threshold_we;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    float entry_threshold_v1h, entry_threshold_v1m;
#endif
    float entry_retracement_base, entry_retracement_we;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    float entry_retracement_v1h, entry_retracement_v1m;
#endif
    float close_qty_pct, close_threshold_base, close_threshold_we;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    float close_threshold_v1h, close_threshold_v1m;
    float close_retracement_base, close_retracement_v1h, close_retracement_v1m;
#else
    float close_retracement_base;
#endif
    float cooldown_min, twel, allowed_wel, entry_cap;
    bool gate_initial, gate_reentry, twel_entry_gate_enabled;
    bool wel_enforcer_enabled;
    float wel_enforcer_threshold;
    bool twel_enforcer_enabled;
    float twel_enforcer_threshold;
    bool unstuck_enabled;
    bool unstuck_ema_gating_enabled;
    float unstuck_close_pct;
    float unstuck_ema_dist;
    float unstuck_loss_allowance_pct;
    float unstuck_threshold;
    float unstuck_allowance;
    bool close_is_exposure_reducer, close_is_twel_reducer;
    bool close_is_unstuck_reducer;
    bool close_loss_gate_disabled_reducers;
    bool close_is_panic;
    bool entry_market, close_market, secondary_close_market;
    bool market_orders_allowed;
    float market_order_near_touch_threshold;
    float ema0, ema1, ema2;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    float vol1m, vol1h;
#endif
    float psize, pprice, last_inc_k, pos_open_k;
    int entry_ticks, close_ticks, secondary_close_ticks;
    float entry_price, close_price, entry_qty, entry_strategy_qty, close_qty;
    float secondary_close_price, secondary_close_qty;
    float entry_gen_balance, entry_gen_psize, entry_gen_pprice, entry_gen_kf;
    float entry_gen_market_price;
    int entry_gen_touch_ticks;
    float close_gen_balance, close_gen_psize, close_gen_pprice;
    float close_gen_market_price;
    float close_gen_realized_pnl_cumsum_last;
    float close_gen_realized_pnl_cumsum_max;
    bool close_gen_unstuck_candidate_enabled;
    int close_gen_touch_down_ticks, close_gen_touch_up_ticks;
    int close_gen_touch_nearest_ticks;
    float close_gen_touch_min_qty;
    int close_gen_touch_min_qty_relation;
    float min_since_open, max_since_min, max_since_open, min_since_max;
};

inline void clear_pending_tm_orders(thread TmSide& side) {
    side.entry_qty = 0.0f;
    side.close_qty = 0.0f;
    side.secondary_close_qty = 0.0f;
}

struct CloseGroup {
    int ticks;
    float price;
    float qty;
    bool market;
};

struct ReducerCandidate {
    float requested_qty;
    float finalized_qty;
    int ticks;
    float price;
    int order_type_id;
    bool market;
    bool is_twel;
    bool is_unstuck;
};

struct RecursiveCloseAllocation {
    float reducer_qty;
    float ordinary_budget;
    float minimum_any;
    float dust_remainder;
    int last_kept_rank;
    int collapse_ordinary_rank;
    bool normalize_close_groups;
};

inline ReducerCandidate empty_reducer_candidate() {
    ReducerCandidate candidate;
    candidate.requested_qty = 0.0f;
    candidate.finalized_qty = 0.0f;
    candidate.ticks = 0;
    candidate.price = 0.0f;
    candidate.order_type_id = 0;
    candidate.market = false;
    candidate.is_twel = false;
    candidate.is_unstuck = false;
    return candidate;
}

inline void install_hsl_panic_close(
    thread TmSide& side,
    bool is_long,
    int touch_down_tick,
    int touch_up_tick,
    float price_step
) {
    side.close_ticks = is_long
        ? max(touch_down_tick - 1, 1)
        : max(touch_up_tick + 1, 1);
    side.close_price = float(side.close_ticks) * price_step;
    side.close_qty = side.psize;
    side.secondary_close_qty = 0.0f;
    side.close_is_exposure_reducer = false;
    side.close_is_twel_reducer = false;
    side.close_is_unstuck_reducer = false;
    // HSL replaces the generated ordinary close. Panic execution policy is
    // carried separately by close_is_panic and the side-specific runner
    // setting; ordinary market flags must not leak into a limit panic.
    side.close_market = false;
    side.secondary_close_market = false;
    side.close_is_panic = true;
}

inline float directional_equity_at_close(
    float balance,
    thread TmSide& long_side,
    thread TmSide& short_side,
    float close,
    float c_mult
) {
    return balance
        + (long_side.psize > 0.0f
            ? long_side.psize * c_mult * (close - long_side.pprice) : 0.0f)
        + (short_side.psize > 0.0f
            ? short_side.psize * c_mult * (short_side.pprice - close) : 0.0f);
}

inline TmSide load_side(constant float* p, int o, float seed) {
    TmSide s;
    float span0 = p[o + 0], span1 = p[o + 1], span2 = sqrt(span0 * span1);
    float lo = fmin(span0, fmin(span1, span2));
    float hi = fmax(span0, fmax(span1, span2));
    float mid = span0 + span1 + span2 - lo - hi;
    s.alpha0 = clamp(2.0f / (lo + 1.0f), 0.0f, 1.0f);
    s.alpha1 = clamp(2.0f / (mid + 1.0f), 0.0f, 1.0f);
    s.alpha2 = clamp(2.0f / (hi + 1.0f), 0.0f, 1.0f);
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    s.alpha1h = p[o + 2] > 0.0f ? 2.0f / (fmax(p[o + 2], 1.0f) + 1.0f) : 0.0f;
    s.alpha1m = p[o + 3] > 0.0f ? clamp(2.0f / (p[o + 3] + 1.0f), 0.0f, 1.0f) : 0.0f;
#endif
    s.ddf = p[o + 4];
    s.initial_ema_dist = p[o + 5];
    s.initial_qty_pct = p[o + 6];
    s.entry_threshold_base = p[o + 7];
    s.entry_threshold_we = p[o + 8];
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    s.entry_threshold_v1h = p[o + 9];
    s.entry_threshold_v1m = p[o + 10];
#endif
    s.entry_retracement_base = p[o + 11];
    s.entry_retracement_we = p[o + 12];
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    s.entry_retracement_v1h = p[o + 13];
    s.entry_retracement_v1m = p[o + 14];
#endif
    s.close_qty_pct = p[o + 15];
    s.close_threshold_base = p[o + 16];
    s.close_threshold_we = p[o + 17];
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    s.close_threshold_v1h = p[o + 18];
    s.close_threshold_v1m = p[o + 19];
#endif
    s.close_retracement_base = p[o + 20];
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    s.close_retracement_v1h = p[o + 21];
    s.close_retracement_v1m = p[o + 22];
#endif
    s.cooldown_min = ceil(p[o + 23]);
    s.twel = p[o + 24];
    s.gate_initial = p[o + 25] > 0.5f;
    s.gate_reentry = p[o + 26] > 0.5f;
    float allowance_pct = fmax(p[o + 27], 0.0f);
    bool legacy_raw_allowance = p[o + 28] > 0.5f;
    float base_wel = p[o + 51];
    if (!(isfinite(base_wel) && base_wel >= 0.0f)) base_wel = s.twel;
    float effective_allowance_pct = allowance_pct;
    if (!legacy_raw_allowance) {
        float max_effective = base_wel > 0.0f
            ? fmax(s.twel / base_wel - 1.0f, 0.0f) : 0.0f;
        effective_allowance_pct = fmin(allowance_pct, max_effective);
    }
    s.allowed_wel = base_wel > 0.0f
        ? base_wel * (1.0f + effective_allowance_pct) : 0.0f;
    s.twel_entry_gate_enabled = p[o + 29] > 0.5f;
    float twel_threshold = p[o + 30];
    float gate_cap = s.twel;
    if (isfinite(twel_threshold) && twel_threshold > 0.0f) {
        gate_cap = fmin(s.twel, s.twel * twel_threshold);
    }
    s.entry_cap = s.twel_entry_gate_enabled
        ? fmin(s.allowed_wel, gate_cap) : s.allowed_wel;
    s.wel_enforcer_enabled = !PASSIVBOT_TM_REDUCERS_DISABLED
        && p[o + 31] > 0.5f;
    s.wel_enforcer_threshold = p[o + 32];
    s.twel_enforcer_enabled = !PASSIVBOT_TM_REDUCERS_DISABLED
        && p[o + 33] > 0.5f;
    s.twel_enforcer_threshold = twel_threshold;
    s.unstuck_enabled = !PASSIVBOT_TM_REDUCERS_DISABLED
        && p[o + 34] > 0.5f;
    s.unstuck_ema_gating_enabled = p[o + 35] > 0.5f;
    s.unstuck_close_pct = p[o + 36];
    s.unstuck_ema_dist = p[o + 37];
    s.unstuck_loss_allowance_pct = p[o + 38];
    s.unstuck_threshold = p[o + 39];
    s.unstuck_allowance = 0.0f;
    s.close_is_exposure_reducer = false;
    s.close_is_twel_reducer = false;
    s.close_is_unstuck_reducer = false;
    s.close_loss_gate_disabled_reducers = false;
    s.close_is_panic = false;
    s.entry_market = false;
    s.close_market = false;
    s.secondary_close_market = false;
    s.market_orders_allowed = false;
    s.market_order_near_touch_threshold = 0.0f;
    s.ema0 = seed; s.ema1 = seed; s.ema2 = seed;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    s.vol1m = 0.0f; s.vol1h = 0.0f;
#endif
    s.psize = 0.0f; s.pprice = 0.0f;
    s.last_inc_k = -1.0f; s.pos_open_k = -1.0f;
    s.entry_ticks = 0; s.entry_qty = 0.0f; s.entry_strategy_qty = 0.0f;
    s.close_ticks = 0; s.close_qty = 0.0f;
    s.entry_price = 0.0f; s.close_price = 0.0f;
    s.secondary_close_ticks = 0; s.secondary_close_qty = 0.0f;
    s.secondary_close_price = 0.0f;
    s.entry_gen_balance = 0.0f;
    s.entry_gen_psize = 0.0f; s.entry_gen_pprice = 0.0f;
    s.entry_gen_kf = -1.0f; s.entry_gen_touch_ticks = 0;
    s.entry_gen_market_price = 0.0f;
    s.close_gen_balance = 0.0f;
    s.close_gen_psize = 0.0f; s.close_gen_pprice = 0.0f;
    s.close_gen_market_price = 0.0f;
    s.close_gen_realized_pnl_cumsum_last = 0.0f;
    s.close_gen_realized_pnl_cumsum_max = 0.0f;
    s.close_gen_unstuck_candidate_enabled = false;
    s.close_gen_touch_down_ticks = 0; s.close_gen_touch_up_ticks = 0;
    s.close_gen_touch_nearest_ticks = 0;
    s.close_gen_touch_min_qty = 0.0f;
    s.close_gen_touch_min_qty_relation = 0;
    s.min_since_open = INFINITY; s.max_since_min = 0.0f;
    s.max_since_open = 0.0f; s.min_since_max = INFINITY;
    return s;
}

inline void update_indicators(
    thread TmSide& s, float close, float lr, float hour_lr, bool valid, bool hour_valid
) {
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    if (hour_valid && s.alpha1h > 0.0f)
        s.vol1h = fma(s.alpha1h, hour_lr - s.vol1h, s.vol1h);
#endif
    if (valid) {
        s.ema0 = fma(s.alpha0, close - s.ema0, s.ema0);
        s.ema1 = fma(s.alpha1, close - s.ema1, s.ema1);
        s.ema2 = fma(s.alpha2, close - s.ema2, s.ema2);
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
        if (s.alpha1m > 0.0f)
            s.vol1m = fma(s.alpha1m, lr - s.vol1m, s.vol1m);
#endif
    }
}

inline void update_trailing(
    thread TmSide& s, float high, float low, float close, bool valid, bool filled
) {
    if (!valid || s.psize <= 0.0f) return;
    if (filled) {
        s.min_since_open = INFINITY; s.max_since_min = 0.0f;
        s.max_since_open = 0.0f; s.min_since_max = INFINITY;
    } else {
        if (low < s.min_since_open) {
            s.min_since_open = low; s.max_since_min = close;
        } else s.max_since_min = fmax(s.max_since_min, high);
        if (high > s.max_since_open) {
            s.max_since_open = high; s.min_since_max = close;
        } else s.min_since_max = fmin(s.min_since_max, low);
    }
}

inline int directional_ticks(float price, float step, bool up) {
    return up ? int(ceil(price / step - 1.0e-6f))
              : int(floor(price / step + 1.0e-6f));
}

inline float crop_entry(
    thread TmSide& s, float balance, float price, float qty,
    float qty_step, float min_qty, float min_cost, float c_mult
) {
    if (qty <= 0.0f) return 0.0f;
    float cost = s.psize * s.pprice * c_mult;
    float we_if = (cost + qty * price * c_mult) / fmax(balance, 1.0e-9f);
    if (we_if <= s.allowed_wel * 1.01f) return qty;
    float q = round_step(
        (s.allowed_wel * balance - cost)
            / fmax(price * c_mult, 1.0e-12f),
        qty_step
    );
    float mq = min_entry_qty(price, qty_step, min_qty, min_cost, c_mult);
    q = fmax(q, mq);
    return q < qty ? q : qty;
}

inline float gate_entry_by_twel_strict(
    thread TmSide& s, float balance, float price, float qty,
    float qty_step, float min_qty, float min_cost, float c_mult
) {
    if (!(qty > 0.0f && balance > 0.0f && price > 0.0f
        && qty_step > 0.0f && c_mult > 0.0f && s.entry_cap > 0.0f)) {
        return 0.0f;
    }
    float current_cost = s.psize * s.pprice * c_mult;
    float remaining_cost = s.entry_cap * balance - current_cost;
    if (!(remaining_cost > 0.0f)) return 0.0f;
    float gated_qty = fmin(
        qty,
        floor_step(remaining_cost / (price * c_mult), qty_step)
    );
    if (!(gated_qty > 0.0f)) return 0.0f;

    // Exact Rust requires post-fill exposure to remain strictly below the
    // total-exposure cap.  A quantity exactly on the cap is not eligible.
    float gated_exposure = (current_cost + gated_qty * price * c_mult)
        / balance;
    if (gated_exposure >= s.entry_cap) {
        float previous_qty = gated_qty;
        float decremented_qty = gated_qty - qty_step;
        if (!(decremented_qty < gated_qty)) {
            decremented_qty = gated_qty - fmax(
                qty_step,
                fabs(gated_qty) * 1.1920928955078125e-7f
            );
        }
        gated_qty = floor_step(decremented_qty, qty_step);
        if (!(gated_qty < previous_qty)) return 0.0f;
    }
    float executable_min = min_entry_qty(
        price, qty_step, min_qty, min_cost, c_mult
    );
    return gated_qty >= executable_min ? gated_qty : 0.0f;
}

inline float calc_close_qty(
    thread TmSide& s, float balance, float mq, int mq_relation, float pct,
    float qty_step, float c_mult
) {
    float full = balance * s.allowed_wel / fmax(s.pprice * c_mult, 1.0e-12f);
    float qty = fmin(
        round_step(s.psize, qty_step),
        fmax(mq, ceil_step(full * pct + fmax(s.psize - full, 0.0f), qty_step))
    );
    float remainder = s.psize - qty;
    bool remainder_below_mq = remainder < mq
        || (remainder == mq && mq_relation > 0);
    if (qty > 0.0f && qty < s.psize && remainder_below_mq) qty = s.psize;
    if (s.psize < mq * (1.0f - 1.0e-6f) && qty > 0.0f) qty = s.psize;
    else if (qty > 0.0f && qty * (1.0f + 1.0e-6f) < mq) qty = 0.0f;
    return qty;
}

inline bool quantity_is_meaningfully_below(float quantity, float boundary) {
    return quantity * (1.0f + 1.0e-6f) < boundary;
}

inline float exposure_reducer_qty(
    float psize, float pprice, float balance, float target_exposure,
    float reducer_price, float qty_step, float min_qty, float min_cost,
    float c_mult
) {
    if (!(balance > 0.0f && psize > 0.0f && pprice > 0.0f
        && target_exposure > 0.0f && reducer_price > 0.0f)) {
        return 0.0f;
    }
    float current_exposure = psize * pprice * c_mult / balance;
    if (!(current_exposure > target_exposure)) return 0.0f;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float target_psize = target_exposure * balance
        / fmax(pprice * c_mult, 1.0e-12f);
    float reduce_qty = fmax(psize - target_psize, 0.0f);
    if (reduce_qty <= 2.220446e-16f) reduce_qty = qty_step;
    float reducer_qty = 0.0f;
    for (int steps = 0; steps <= 10000; ++steps) {
        reducer_qty = fmin(
            psize, fmax(reducer_min, ceil_step(reduce_qty, qty_step))
        );
        float new_psize = fmax(
            round_step(psize - reducer_qty, qty_step), 0.0f
        );
        if (new_psize < target_psize || new_psize <= 2.220446e-16f) {
            break;
        }
        reduce_qty += qty_step;
    }
    return reducer_qty;
}

inline float total_exposure_reducer_qty(
    float psize, float pprice, float balance, float target_exposure,
    float reducer_price, float qty_step, float min_qty, float min_cost,
    float c_mult
) {
    if (!(balance > 0.0f && psize > 0.0f && pprice > 0.0f
        && target_exposure > 0.0f && reducer_price > 0.0f)) {
        return 0.0f;
    }
    float current_exposure = psize * pprice * c_mult / balance;
    if (!(current_exposure > target_exposure + 1.0e-9f)) return 0.0f;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float exposure_to_cut = current_exposure - target_exposure;
    float requested_qty = ceil_step(
        exposure_to_cut * balance / fmax(pprice * c_mult, 1.0e-12f),
        qty_step
    );
    float reducer_qty = fmin(
        psize, fmax(reducer_min, requested_qty)
    );
    return fmin(psize, ceil_step(reducer_qty, qty_step));
}

inline float finalized_reducer_qty(
    float psize, float reducer_qty, float reducer_price,
    float qty_step, float min_qty, float min_cost, float c_mult
) {
    if (reducer_qty <= 0.0f || reducer_price <= 0.0f) return 0.0f;
    float remainder = fmax(
        round_step(psize - reducer_qty, qty_step), 0.0f
    );
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    // Reducer selection finalizes each competing protective close separately.
    // In a WEL/TWEL competition the strategy WEL has already suppressed the
    // ordinary ladder, so a sub-minimum remainder is absorbed by the candidate.
    return remainder > 0.0f && remainder < reducer_min
        ? psize : reducer_qty;
}

inline float finalized_reducer_qty_with_ordinary(
    float psize,
    float reducer_qty,
    float reducer_price,
    float ordinary_qty,
    float ordinary_min,
    int ordinary_min_relation,
    float qty_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    if (reducer_qty <= 0.0f || reducer_price <= 0.0f) return 0.0f;
    reducer_qty = fmin(psize, reducer_qty);
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    if (ordinary_qty > 0.0f) {
        if (ordinary_qty + reducer_qty > psize) {
            ordinary_qty = fmax(
                round_step(psize - reducer_qty, qty_step), 0.0f
            );
        }
        bool ordinary_below_minimum = ordinary_qty < ordinary_min
            || (ordinary_qty == ordinary_min && ordinary_min_relation > 0);
        if (!ordinary_below_minimum) {
            float remainder = fmax(
                round_step(psize - reducer_qty - ordinary_qty, qty_step),
                0.0f
            );
            float minimum_any = fmin(ordinary_min, reducer_min);
            if (remainder > 0.0f && remainder < minimum_any) {
                ordinary_qty = fmin(
                    psize - reducer_qty,
                    round_step(ordinary_qty + remainder, qty_step)
                );
            }
            if (ordinary_qty > 0.0f) return reducer_qty;
        }
    }
    float remainder = fmax(
        round_step(psize - reducer_qty, qty_step), 0.0f
    );
    return remainder > 0.0f && remainder < reducer_min
        ? psize : reducer_qty;
}

inline bool reducer_candidate_preferred(
    float left_qty,
    int left_ticks,
    int left_order_type_id,
    float right_qty,
    int right_ticks,
    int right_order_type_id,
    bool is_long
) {
    if (!(left_qty > 0.0f)) return false;
    if (!(right_qty > 0.0f)) return true;
    if (left_qty != right_qty) return left_qty > right_qty;
    if (left_ticks != right_ticks) {
        return is_long ? left_ticks < right_ticks : left_ticks > right_ticks;
    }
    return left_order_type_id < right_order_type_id;
}

inline float calc_unstuck_allowance(
    thread const TmSide& s,
    float balance,
    float balance_peak
) {
    if (!(s.unstuck_enabled
        && s.unstuck_loss_allowance_pct > 0.0f
        && s.twel > 0.0f
        && balance > 0.0f
        && balance_peak > 0.0f)) {
        return 0.0f;
    }
    float allowance_pct = s.unstuck_loss_allowance_pct * s.twel;
    return float32_floor_nonnegative(
        fmax(balance - balance_peak * (1.0f - allowance_pct), 0.0f)
    );
}

inline bool unstuck_eligible(
    thread const TmSide& s,
    bool is_long,
    float balance,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float price_step,
    float c_mult
) {
    if (!(s.unstuck_enabled
        && s.unstuck_allowance > 0.0f
        && s.unstuck_close_pct > 0.0f
        && s.unstuck_threshold > 0.0f
        && s.psize > 0.0f
        && s.pprice > 0.0f
        && s.allowed_wel > 0.0f
        && balance > 0.0f
        && price_now > 0.0f)) {
        return false;
    }
    float wallet_exposure = s.psize * s.pprice * c_mult / balance;
    if (!(wallet_exposure / s.allowed_wel > s.unstuck_threshold)) {
        return false;
    }
    if (!s.unstuck_ema_gating_enabled) return true;
    float lower = fmin(s.ema0, fmin(s.ema1, s.ema2));
    float upper = fmax(s.ema0, fmax(s.ema1, s.ema2));
    int trigger_ticks = is_long
        ? int(ceil(
            upper * (1.0f + s.unstuck_ema_dist) / price_step - 1.0e-6f
        ))
        : int(floor(
            lower * (1.0f - s.unstuck_ema_dist) / price_step + 1.0e-6f
        ));
    return is_long
        ? touch_down_ticks >= trigger_ticks
        : touch_up_ticks <= trigger_ticks;
}

inline float unstuck_reducer_qty(
    thread const TmSide& s,
    bool is_long,
    float balance,
    float reducer_price,
    float qty_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    if (!(s.unstuck_enabled && s.unstuck_allowance > 0.0f)) return 0.0f;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float target_qty = floor_step(
        balance * s.allowed_wel * s.unstuck_close_pct
            / fmax(reducer_price * c_mult, 1.0e-12f),
        qty_step
    );
    float qty = fmin(s.psize, fmax(reducer_min, target_qty));
    float gross_pnl = qty * c_mult * (
        is_long ? reducer_price - s.pprice : s.pprice - reducer_price
    );
    if (gross_pnl < 0.0f && -gross_pnl > s.unstuck_allowance) {
        float scaled_qty = fmin(
            s.psize, qty * s.unstuck_allowance / -gross_pnl
        );
        qty = fmin(
            s.psize,
            fmax(reducer_min, floor_step(scaled_qty, qty_step))
        );
    }
    return qty;
}

inline void generate_orders(
    thread TmSide& s, bool is_long, float balance, float price_now,
    int touch_down_ticks, int touch_up_ticks, int touch_nearest_ticks,
    float touch_min_qty, int touch_min_qty_relation,
    float qty_step, float price_step, float min_qty, float min_cost, float c_mult,
    float kf, bool block_initial
) {
    bool entry_up = !is_long;
    bool close_up = is_long;
    float band = is_long ? fmin(s.ema0, fmin(s.ema1, s.ema2))
                         : fmax(s.ema0, fmax(s.ema1, s.ema2));
    int entry_touch = is_long ? touch_down_ticks : touch_up_ticks;
    // Exact Rust builds a recursive entry ladder from one immutable generation
    // snapshot. Preserve that snapshot so all rungs touched by the next candle
    // can be reconstructed without fee-adjusted sizing drift.
    s.entry_gen_balance = balance;
    s.entry_gen_psize = s.psize;
    s.entry_gen_pprice = s.pprice;
    s.entry_gen_kf = kf;
    s.entry_gen_touch_ticks = entry_touch;
    s.entry_gen_market_price = price_now;
    s.close_gen_balance = balance;
    s.close_gen_psize = s.psize;
    s.close_gen_pprice = s.pprice;
    s.close_gen_market_price = price_now;
    s.close_gen_touch_down_ticks = touch_down_ticks;
    s.close_gen_touch_up_ticks = touch_up_ticks;
    s.close_gen_touch_nearest_ticks = touch_nearest_ticks;
    s.close_gen_touch_min_qty = touch_min_qty;
    s.close_gen_touch_min_qty_relation = touch_min_qty_relation;
    s.close_is_exposure_reducer = false;
    s.close_is_twel_reducer = false;
    s.close_is_unstuck_reducer = false;
    s.close_loss_gate_disabled_reducers = false;
    s.secondary_close_ticks = 0;
    s.secondary_close_price = 0.0f;
    s.secondary_close_qty = 0.0f;
    s.secondary_close_market = false;
    int band_ticks = directional_ticks(
        band * (is_long ? 1.0f - s.initial_ema_dist
                        : 1.0f + s.initial_ema_dist),
        price_step, entry_up
    );
    float band_price = float(band_ticks) * price_step;
    // Compare float64-derived directional touch ticks before selecting raw
    // entry targets. A raw touch and a neighboring tick target may collapse
    // to equality in float32 even though exact Rust's min/max selects the tick.
    bool initial_touch_controls = !s.gate_initial || (entry_up
        ? touch_down_ticks >= band_ticks : touch_up_ticks <= band_ticks);
    int initial_ticks = initial_touch_controls ? entry_touch : band_ticks;
    // Exact Rust chooses the controlling raw/target value in float64, then
    // finalize_next_entry quantizes the executable order directionally.
    float initial_price = float(initial_ticks) * price_step;
    float min_iq = min_entry_qty(initial_price, qty_step, min_qty, min_cost, c_mult);
    float iq = fmax(min_iq, round_step(
        balance * s.allowed_wel * s.initial_qty_pct
            / fmax(initial_price * c_mult, 1.0e-12f), qty_step
    ));
    bool flat = s.psize <= 0.0f;
    bool partial = !flat && s.psize < iq * 0.8f;
    float iq_partial = fmax(min_iq, floor_step(iq - s.psize, qty_step));
    float iq_effective = !flat && s.psize < iq
        ? fmax(round_step(s.psize, qty_step), min_iq) : iq;
    float we = !flat && balance > 0.0f
        ? s.psize * s.pprice * c_mult / balance : 0.0f;
    float wer = we / fmax(s.allowed_wel, 1.0e-12f);
#if PASSIVBOT_TM_VOLATILITY_DISABLED
    float tm = fmax(1.0f + wer * s.entry_threshold_we, 1.0f);
#if !PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY
    float rm = fmax(1.0f + wer * s.entry_retracement_we, 1.0f);
#endif
#else
    float tm = fmax(
        1.0f + s.vol1h * s.entry_threshold_v1h
            + s.vol1m * s.entry_threshold_v1m + wer * s.entry_threshold_we,
        1.0f
    );
#if !PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY
    float rm = fmax(
        1.0f + s.vol1h * s.entry_retracement_v1h
            + s.vol1m * s.entry_retracement_v1m + wer * s.entry_retracement_we,
        1.0f
    );
#endif
#endif
    float threshold = fmax(s.entry_threshold_base, 0.0f) * tm;
#if PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY
    const bool trailing_entry = false;
    bool entry_triggered = true;
    float reentry_target = s.pprice * (
        is_long ? 1.0f - s.entry_threshold_base * tm
                : 1.0f + s.entry_threshold_base * tm
    );
#else
    float retracement = fmax(s.entry_retracement_base, 0.0f) * rm;
    const bool trailing_entry = PASSIVBOT_TM_TRAILING_ENTRY_ONLY
        || s.entry_retracement_base > 0.0f;
    bool retraced_entry = is_long
        ? s.max_since_min > s.min_since_open * (1.0f + retracement)
        : s.min_since_max < s.max_since_open * (1.0f - retracement);
    bool crossed_entry = is_long
        ? s.min_since_open < s.pprice * (1.0f - threshold)
        : s.max_since_open > s.pprice * (1.0f + threshold);
    bool entry_triggered = true;
    float reentry_target = s.pprice * (
        is_long ? 1.0f - s.entry_threshold_base * tm
                : 1.0f + s.entry_threshold_base * tm
    );
    if (trailing_entry) {
        if (threshold <= 0.0f) {
            entry_triggered = retracement > 0.0f && retraced_entry;
            reentry_target = price_now;
        } else if (retracement <= 0.0f) {
            reentry_target = s.pprice * (is_long ? 1.0f - threshold : 1.0f + threshold);
        } else {
            entry_triggered = crossed_entry && retraced_entry;
            reentry_target = s.pprice * (
                is_long ? 1.0f - threshold + retracement
                        : 1.0f + threshold - retracement
            );
        }
    }
#endif
    bool reentry_target_is_touch = trailing_entry && threshold <= 0.0f;
    int raw_reentry_ticks = reentry_target_is_touch
        ? entry_touch : directional_ticks(reentry_target, price_step, entry_up);
    bool reentry_touch_controls = reentry_target_is_touch || (entry_up
        ? touch_down_ticks >= raw_reentry_ticks
        : touch_up_ticks <= raw_reentry_ticks);
    int reentry_ticks = reentry_touch_controls ? entry_touch : raw_reentry_ticks;
    float reentry_price = float(reentry_ticks) * price_step;
    if (s.gate_reentry) {
        bool band_controls = entry_up
            ? band_ticks >= reentry_ticks : band_ticks <= reentry_ticks;
        if (band_controls) {
            reentry_ticks = band_ticks;
            reentry_price = band_price;
        }
    }
    float min_rq = min_entry_qty(reentry_price, qty_step, min_qty, min_cost, c_mult);
    float rq = fmax(iq_effective, fmax(min_rq, round_step(
        fmax(
            s.psize * s.ddf,
            balance * s.allowed_wel * s.initial_qty_pct
                / fmax(reentry_price * c_mult, 1.0e-12f)
        ), qty_step
    )));
    float we_if = (s.psize * s.pprice + rq * reentry_price)
        * c_mult / fmax(balance, 1.0e-9f);
    float crop_fraction = (s.allowed_wel - we)
        / fmax(we_if - we, 1.0e-12f);
    float rq_crop = fmax(round_step(rq * crop_fraction, qty_step), min_rq);
    if (we_if > s.allowed_wel * 1.01f && rq_crop < rq) rq = rq_crop;
    bool cap_hit = trailing_entry
        ? we > s.allowed_wel * 0.999f : we >= s.allowed_wel * 0.999f;
    bool reentry_ok = !flat && !partial && !cap_hit && reentry_ticks > 1
        && (!trailing_entry || entry_triggered);
    float eqty = flat ? iq : (partial ? iq_partial : (reentry_ok ? rq : 0.0f));
    int eticks = flat || partial ? initial_ticks : reentry_ticks;
    float eprice = flat || partial ? initial_price : reentry_price;
    bool entry_market = should_use_ordinary_market_execution(
        eticks, is_long, price_now, price_step, s.market_orders_allowed,
        s.market_order_near_touch_threshold
    );
    float market_min_q = entry_market
        ? min_entry_qty(price_now, qty_step, min_qty, min_cost, c_mult)
        : min_entry_qty(eprice, qty_step, min_qty, min_cost, c_mult);
    bool cooldown = s.cooldown_min > 0.0f && s.last_inc_k >= 0.0f
        && kf < s.last_inc_k + s.cooldown_min;
    if (cooldown || balance <= 0.0f || s.initial_qty_pct <= 0.0f
        || s.allowed_wel <= 0.0f || eticks <= 1
        || (block_initial && flat)) eqty = 0.0f;
    s.entry_ticks = eticks;
    s.entry_price = eprice;
    s.entry_qty = crop_entry(
        s, balance, eprice, eqty,
        qty_step, min_qty, min_cost, c_mult
    );
    // Rust builds the immutable recursive strategy ladder before market
    // executable-minimum sizing and portfolio TWEL gating mutate its orders.
    s.entry_strategy_qty = s.entry_qty;
    if (!is_long && entry_market && s.entry_qty > 0.0f
        && s.entry_qty < market_min_q) {
        s.entry_qty = market_min_q;
    }
    if (s.twel_entry_gate_enabled) {
        float entry_gate_price = entry_market ? price_now : eprice;
        s.entry_qty = gate_entry_by_twel_strict(
            s, balance, entry_gate_price, s.entry_qty,
            qty_step, min_qty, min_cost, c_mult
        );
    }
    if (!is_long && entry_market && s.entry_qty + 1.0e-12f < market_min_q) {
        s.entry_qty = 0.0f;
    }
    s.entry_market = entry_market && s.entry_qty > 0.0f;

    // Exact Rust selects the largest protective reducer before allocating the
    // ordinary close ladder.  Model both the per-position WEL repair and the
    // side-wide TWEL repair, retaining the winning reducer's own price.
    float wel_target = s.allowed_wel * s.wel_enforcer_threshold;
    int wel_ticks = is_long ? touch_up_ticks : touch_down_ticks;
    float wel_price = float(wel_ticks) * price_step;
    float wel_qty = s.wel_enforcer_enabled && s.wel_enforcer_threshold > 0.0f
        ? exposure_reducer_qty(
            s.psize, s.pprice, balance, wel_target, wel_price,
            qty_step, min_qty, min_cost, c_mult
        ) : 0.0f;
    bool wel_market = wel_qty > 0.0f
        && should_use_ordinary_market_execution(
            wel_ticks, !is_long, price_now, price_step,
            s.market_orders_allowed, s.market_order_near_touch_threshold
        );
    if (wel_market) {
        wel_qty = resize_market_close_qty(
            wel_qty, s.psize, price_now,
            qty_step, min_qty, min_cost, c_mult
        );
    }
    float wel_exec_price = wel_market ? price_now : wel_price;

    float twel_target = s.twel * s.twel_enforcer_threshold;
    int twel_ticks = is_long
        ? int(floor(price_now * 0.9995f / price_step + 1.0e-6f))
        : int(ceil(price_now * 1.0005f / price_step - 1.0e-6f));
    twel_ticks = max(twel_ticks, 1);
    float twel_price = float(twel_ticks) * price_step;
    float twel_qty = s.twel_enforcer_enabled
        && s.twel_enforcer_threshold > 0.0f
        ? total_exposure_reducer_qty(
            s.psize, s.pprice, balance, twel_target, twel_price,
            qty_step, min_qty, min_cost, c_mult
        ) : 0.0f;
    bool twel_market = twel_qty > 0.0f
        && should_use_ordinary_market_execution(
            twel_ticks, !is_long, price_now, price_step,
            s.market_orders_allowed, s.market_order_near_touch_threshold
        );
    if (twel_market) {
        twel_qty = resize_market_close_qty(
            twel_qty, s.psize, price_now,
            qty_step, min_qty, min_cost, c_mult
        );
    }
    float twel_exec_price = twel_market ? price_now : twel_price;

    int unstuck_ticks = is_long ? touch_up_ticks : touch_down_ticks;
    float unstuck_price = float(unstuck_ticks) * price_step;
    float unstuck_qty = s.unstuck_enabled
        ? unstuck_reducer_qty(
            s, is_long, balance, unstuck_price,
            qty_step, min_qty, min_cost, c_mult
        )
        : 0.0f;
    bool unstuck_market = unstuck_qty > 0.0f
        && should_use_ordinary_market_execution(
            unstuck_ticks, !is_long, price_now, price_step,
            s.market_orders_allowed, s.market_order_near_touch_threshold
        );
    if (unstuck_market) {
        unstuck_qty = resize_market_close_qty(
            unstuck_qty, s.psize, price_now,
            qty_step, min_qty, min_cost, c_mult
        );
    }
    float unstuck_exec_price = unstuck_market ? price_now : unstuck_price;

#if PASSIVBOT_TM_VOLATILITY_DISABLED
    float ct = s.close_threshold_base + wer * s.close_threshold_we;
    float cr = fmax(s.close_retracement_base, 0.0f);
#else
    float ct = s.close_threshold_base + wer * s.close_threshold_we
        + s.vol1h * s.close_threshold_v1h + s.vol1m * s.close_threshold_v1m;
    float cr = fmax(s.close_retracement_base, 0.0f) * fmax(
        1.0f + s.vol1h * s.close_retracement_v1h
            + s.vol1m * s.close_retracement_v1m,
        1.0f
    );
#endif
    const bool trailing_close = PASSIVBOT_TM_TRAILING_CLOSE_ONLY
        || s.close_retracement_base > 0.0f;
    bool retraced_close = is_long
        ? s.min_since_max < s.max_since_open * (1.0f - cr)
        : s.max_since_min > s.min_since_open * (1.0f + cr);
    bool crossed_close = is_long
        ? s.max_since_open > s.pprice * (1.0f + ct)
        : s.min_since_open < s.pprice * (1.0f - ct);
    bool close_triggered = true;
    float close_target = s.pprice * (is_long ? 1.0f + ct : 1.0f - ct);
    if (trailing_close) {
        if (ct <= 0.0f) {
            close_triggered = cr > 0.0f && retraced_close;
            close_target = price_now;
        } else if (cr > 0.0f) {
            close_triggered = crossed_close && retraced_close;
            close_target = s.pprice * (
                is_long ? 1.0f + ct - cr : 1.0f - ct + cr
            );
        }
    }
    int target_ticks = directional_ticks(close_target, price_step, close_up);
    int close_touch = close_up ? touch_up_ticks : touch_down_ticks;
    // Compare the float64-derived directional touch ticks before choosing the
    // raw touch. Nearby raw and tick prices may be equal after float32
    // conversion even though exact Rust's max/min selects the raw value.
    bool touch_controls = (trailing_close && ct <= 0.0f) || (close_up
        ? close_touch > target_ticks : close_touch < target_ticks);
    // calc_closes_long/short quantizes the selected close to nearest tick.
    int cticks = touch_controls ? touch_nearest_ticks : target_ticks;
    float close_price = float(cticks) * price_step;
    // Rust sizes the selected raw touch before calc_closes_* quantizes its
    // executable price. Python preserves the float64 minimum's ordering
    // relative to its float32 representation for the remainder comparison.
    float close_mq = touch_controls
        ? touch_min_qty
        : min_entry_qty(close_price, qty_step, min_qty, min_cost, c_mult);
    int close_mq_relation = touch_controls ? touch_min_qty_relation : 0;
    float pct = trailing_close ? s.close_qty_pct
        : (s.close_threshold_we == 0.0f ? 1.0f : s.close_qty_pct);
    s.close_ticks = cticks;
    s.close_price = close_price;
    s.close_qty = s.psize > 0.0f && close_price > 0.0f
            && (!trailing_close || close_triggered)
        ? calc_close_qty(
            s, balance, close_mq, close_mq_relation, pct, qty_step, c_mult
        ) : 0.0f;
    s.close_market = s.close_qty > 0.0f
        && should_use_ordinary_market_execution(
            cticks, !is_long, price_now, price_step,
            s.market_orders_allowed, s.market_order_near_touch_threshold
        );
    if (s.close_market) {
        s.close_qty = resize_market_close_qty(
            s.close_qty, s.psize, price_now,
            qty_step, min_qty, min_cost, c_mult
        );
    }
    float ordinary_min = s.close_market
        ? min_entry_qty(price_now, qty_step, min_qty, min_cost, c_mult)
        : close_mq;
    int ordinary_min_relation = s.close_market ? 0 : close_mq_relation;
    bool ordinary_can_accompany_reducer = wel_qty <= 0.0f
        && trailing_close && s.close_qty > 0.0f;
    float finalized_wel_qty = finalized_reducer_qty(
        s.psize, wel_qty, wel_exec_price,
        qty_step, min_qty, min_cost, c_mult
    );
    float finalized_twel_qty = ordinary_can_accompany_reducer
        ? finalized_reducer_qty_with_ordinary(
            s.psize, twel_qty, twel_exec_price, s.close_qty, ordinary_min,
            ordinary_min_relation, qty_step, min_qty, min_cost, c_mult
        )
        : finalized_reducer_qty(
            s.psize, twel_qty, twel_exec_price,
            qty_step, min_qty, min_cost, c_mult
        );
    float finalized_unstuck_qty = ordinary_can_accompany_reducer
        ? finalized_reducer_qty_with_ordinary(
            s.psize, unstuck_qty, unstuck_exec_price, s.close_qty, ordinary_min,
            ordinary_min_relation, qty_step, min_qty, min_cost, c_mult
        )
        : finalized_reducer_qty(
            s.psize, unstuck_qty, unstuck_exec_price,
            qty_step, min_qty, min_cost, c_mult
        );
    int unstuck_order_type_id = is_long ? 9 : 20;
    bool use_twel = finalized_twel_qty > finalized_wel_qty;
    float finalized_exposure_qty = use_twel
        ? finalized_twel_qty : finalized_wel_qty;
    int exposure_ticks = use_twel ? twel_ticks : wel_ticks;
    int exposure_order_type_id = use_twel
        ? (is_long ? 10 : 21) : (is_long ? 24 : 25);
    // Rust chooses by finalized size, then strict distance to the executable
    // touch, and finally stable order fields. Unstuck and WEL share the touch,
    // so the lower unstuck order-type id wins only that exact-price tie.
    bool use_unstuck = reducer_candidate_preferred(
        finalized_unstuck_qty, unstuck_ticks, unstuck_order_type_id,
        finalized_exposure_qty, exposure_ticks, exposure_order_type_id,
        is_long
    );
    // Final sizing is only the selection key. Preserve the requested quantity
    // for the winning reducer so the ordinary close allocator below can absorb
    // dust exactly as Rust does when TWEL is the sole strategy-external close.
    float reducer_qty = use_unstuck
        ? unstuck_qty : (use_twel ? twel_qty : wel_qty);
    int reducer_ticks = use_unstuck
        ? unstuck_ticks : (use_twel ? twel_ticks : wel_ticks);
    float reducer_price = use_unstuck
        ? unstuck_price : (use_twel ? twel_price : wel_price);
    bool reducer_market = use_unstuck
        ? unstuck_market : (use_twel ? twel_market : wel_market);
    float reducer_exec_price = reducer_market ? price_now : reducer_price;
    if (reducer_qty > 0.0f && reducer_ticks > 0) {
        float reducer_min = min_entry_qty(
            reducer_exec_price, qty_step, min_qty, min_cost, c_mult
        );
        // WEL is emitted by the strategy itself and therefore suppresses a
        // trailing close. TWEL is appended by orchestration, so it retains an
        // independently generated trailing close only when no strategy WEL
        // would have taken precedence during calc_closes_*.
        if ((use_twel || use_unstuck) && wel_qty <= 0.0f
            && trailing_close && s.close_qty > 0.0f) {
            float ordinary_qty = s.close_qty;
            if (ordinary_qty + reducer_qty > s.psize) {
                ordinary_qty = fmax(
                    round_step(s.psize - reducer_qty, qty_step), 0.0f
                );
            }
            bool ordinary_below_minimum = ordinary_qty < ordinary_min
                || (ordinary_qty == ordinary_min
                    && ordinary_min_relation > 0);
            if (!ordinary_below_minimum) {
                float remainder = fmax(
                    round_step(
                        s.psize - reducer_qty - ordinary_qty, qty_step
                    ),
                    0.0f
                );
                float minimum_any = fmin(ordinary_min, reducer_min);
                if (remainder > 0.0f && remainder < minimum_any) {
                    ordinary_qty = fmin(
                        s.psize - reducer_qty,
                        round_step(ordinary_qty + remainder, qty_step)
                    );
                }
                s.secondary_close_ticks = s.close_ticks;
                s.secondary_close_price = s.close_price;
                s.secondary_close_qty = ordinary_qty;
                s.secondary_close_market = s.close_market;
            }
        }
        if (s.secondary_close_qty <= 0.0f) {
            float remainder = fmax(
                round_step(s.psize - reducer_qty, qty_step), 0.0f
            );
            if (remainder > 0.0f && remainder < reducer_min) {
                reducer_qty = s.psize;
            }
        }
        s.close_ticks = reducer_ticks;
        s.close_price = reducer_price;
        s.close_qty = reducer_qty;
        s.close_market = reducer_market;
        s.close_is_exposure_reducer = true;
        s.close_is_twel_reducer = use_twel && !use_unstuck;
        s.close_is_unstuck_reducer = use_unstuck;
    }
}

inline void generate_long_orders(
    thread TmSide& s, float balance, float price_now, float qty_step,
    int touch_down_ticks, int touch_up_ticks, int touch_nearest_ticks,
    float touch_min_qty, int touch_min_qty_relation,
    float price_step, float min_qty, float min_cost, float c_mult,
    float kf, bool block_initial
) {
    generate_orders(
        s, true, balance, price_now, touch_down_ticks, touch_up_ticks,
        touch_nearest_ticks, touch_min_qty, touch_min_qty_relation,
        qty_step, price_step, min_qty, min_cost,
        c_mult, kf, block_initial
    );
}

inline void generate_short_orders(
    thread TmSide& s, float balance, float price_now, float qty_step,
    int touch_down_ticks, int touch_up_ticks, int touch_nearest_ticks,
    float touch_min_qty, int touch_min_qty_relation,
    float price_step, float min_qty, float min_cost, float c_mult,
    float kf, bool block_initial
) {
    generate_orders(
        s, false, balance, price_now, touch_down_ticks, touch_up_ticks,
        touch_nearest_ticks, touch_min_qty, touch_min_qty_relation,
        qty_step, price_step, min_qty, min_cost,
        c_mult, kf, block_initial
    );
}

// Rebuild Rust's immutable recursive close grid and return its number of
// duplicate-merged price groups. If wanted_group is present, also return that
// group. Keeping only one requested group avoids a large per-GPU-thread array;
// close grids are rebuilt only after a conservative fill-reachability check.
inline int recursive_close_groups(
    thread const TmSide& source, bool is_long, int wanted_group,
    float qty_step, float price_step, float min_qty, float min_cost, float c_mult,
    float market_resize_psize,
    int prefix_merge_ticks, float prefix_merge_qty,
    int max_rungs,
    thread CloseGroup& selected
) {
    selected.ticks = 0;
    selected.price = 0.0f;
    selected.qty = 0.0f;
    selected.market = false;
    TmSide sim = source;
    sim.psize = source.close_gen_psize;
    sim.pprice = source.close_gen_pprice;
    // Reconstruct Rust's passive strategy ladder first. Ordinary market
    // execution is a policy applied to each completed price group against the
    // market captured at generation time; applying it inside generate_orders
    // would compare against close_gen_pprice and permanently distort sizing.
    sim.market_orders_allowed = false;
    bool have_group = false;
    int group_count = 0;
    int group_ticks = 0;
    float group_price = 0.0f;
    float group_qty = 0.0f;

    for (int rung = 0; rung < max_rungs; ++rung) {
        generate_orders(
            sim, is_long, source.close_gen_balance, source.close_gen_pprice,
            source.close_gen_touch_down_ticks, source.close_gen_touch_up_ticks,
            source.close_gen_touch_nearest_ticks, source.close_gen_touch_min_qty,
            source.close_gen_touch_min_qty_relation, qty_step, price_step,
            min_qty, min_cost, c_mult, 0.0f, false
        );
        float qty = round_step(sim.close_qty, qty_step);
        if (qty <= 0.0f || sim.close_ticks <= 0) break;

        if (!have_group) {
            have_group = true;
            group_ticks = sim.close_ticks;
            group_price = sim.close_price;
            group_qty = qty;
        } else if (sim.close_ticks == group_ticks) {
            group_qty = round_step(group_qty + qty, qty_step);
        } else {
            if (group_count == wanted_group) {
                selected.ticks = group_ticks;
                selected.price = group_price;
                selected.qty = group_count == 0
                        && group_ticks == prefix_merge_ticks
                    ? round_step(group_qty + prefix_merge_qty, qty_step)
                    : group_qty;
            }
            ++group_count;
            group_ticks = sim.close_ticks;
            group_price = sim.close_price;
            group_qty = qty;
        }

        sim.psize = fmax(round_step(sim.psize - qty, qty_step), 0.0f);
        if (sim.psize <= 0.0f) break;
    }
    if (have_group) {
        if (group_count == wanted_group) {
            selected.ticks = group_ticks;
            selected.price = group_price;
            selected.qty = group_count == 0
                    && group_ticks == prefix_merge_ticks
                ? round_step(group_qty + prefix_merge_qty, qty_step)
                : group_qty;
        }
        ++group_count;
    }
    if (selected.qty > 0.0f && selected.ticks > 0) {
        selected.market = should_use_ordinary_market_execution(
            selected.ticks, !is_long, source.close_gen_market_price,
            price_step, source.market_orders_allowed,
            source.market_order_near_touch_threshold
        );
        if (selected.market) {
            selected.qty = resize_market_close_qty(
                selected.qty, market_resize_psize,
                source.close_gen_market_price,
                qty_step, min_qty, min_cost, c_mult
            );
        }
    }
    return group_count;
}

// Re-run Rust's close allocator for one protective reducer against the fully
// reconstructed ordinary ladder.  Each WEL, TWEL, and unstuck candidate must
// be finalized independently: reducer size changes which ordinary closes are
// trimmed, which in turn changes minimum filtering and dust absorption.
inline RecursiveCloseAllocation recursive_close_allocation(
    thread const TmSide& source,
    bool is_long,
    int group_count,
    float position_size,
    float reducer_requested_qty,
    float reducer_price,
    bool reducer_market,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    int prefix_merge_ticks,
    float prefix_merge_qty,
    int grid_rung_limit
) {
    RecursiveCloseAllocation allocation;
    float requested_reducer_qty = fmin(
        round_step(reducer_requested_qty, qty_step), position_size
    );
    bool include_reducer = requested_reducer_qty > 0.0f;
    // At most one retry is needed: if the reducer itself is below minimum but
    // a mixed-price ordinary rung is executable, Rust filters that reducer and
    // trims the ordinary ladder again without reserving any reducer quantity.
    for (int allocation_pass = 0; allocation_pass < 2; ++allocation_pass) {
        allocation.reducer_qty = include_reducer
            ? requested_reducer_qty : 0.0f;
        bool has_reducer = allocation.reducer_qty > 0.0f;
        float reducer_min = has_reducer
            ? min_entry_qty(
                reducer_market ? source.close_gen_market_price : reducer_price,
                qty_step, min_qty, min_cost, c_mult
            ) : 1.0e30f;
        allocation.ordinary_budget = fmax(
            round_step(
                position_size - allocation.reducer_qty, qty_step
            ),
            0.0f
        );
        float remaining_budget = allocation.ordinary_budget;
        float kept_ordinary = 0.0f;
        allocation.minimum_any = reducer_min;
        bool all_below_min = !has_reducer
            || quantity_is_meaningfully_below(position_size, reducer_min);
        bool any_group_market = false;
        allocation.last_kept_rank = -1;
        bool reverse = source.close_threshold_we > 0.0f;
        CloseGroup group;
        for (int trim_rank = 0; trim_rank < group_count; ++trim_rank) {
            int wanted = reverse ? group_count - trim_rank - 1 : trim_rank;
            recursive_close_groups(
                source, is_long, wanted, qty_step, price_step,
                min_qty, min_cost, c_mult, position_size,
                prefix_merge_ticks, prefix_merge_qty,
                grid_rung_limit, group
            );
            any_group_market = any_group_market || group.market;
            float trimmed_qty = fmin(group.qty, remaining_budget);
            float group_min = min_entry_qty(
                group.market ? source.close_gen_market_price : group.price,
                qty_step, min_qty, min_cost, c_mult
            );
            all_below_min = all_below_min
                && quantity_is_meaningfully_below(position_size, group_min);
            bool partial_trim = quantity_is_meaningfully_below(
                trimmed_qty, group.qty
            );
            if (quantity_is_meaningfully_below(trimmed_qty, group_min)) {
                trimmed_qty = 0.0f;
                if (partial_trim) remaining_budget = 0.0f;
            }
            if (trimmed_qty > 0.0f) {
                kept_ordinary += trimmed_qty;
                remaining_budget = fmax(
                    round_step(remaining_budget - trimmed_qty, qty_step),
                    0.0f
                );
                allocation.minimum_any = fmin(
                    allocation.minimum_any, group_min
                );
                allocation.last_kept_rank = trim_rank;
            }
        }
        bool reducer_below_min = has_reducer
            && quantity_is_meaningfully_below(
                allocation.reducer_qty, reducer_min
            );
        if (reducer_below_min && !all_below_min) {
            include_reducer = false;
            continue;
        }
        // The reducer-drop retry still represents a normalized close set:
        // minimum filtering and aggregate position trimming above must remain
        // authoritative even when every surviving ordinary group is passive.
        allocation.normalize_close_groups = allocation_pass > 0
            || has_reducer || any_group_market;
        allocation.collapse_ordinary_rank = -1;
        if (allocation.normalize_close_groups && all_below_min
            && !has_reducer && group_count > 0) {
            // Rust's below-minimum position exception keeps one
            // closest-to-fill ordinary close at the full position size.
            allocation.collapse_ordinary_rank = 0;
            kept_ordinary = position_size;
            allocation.last_kept_rank = 0;
        }
        allocation.dust_remainder = allocation.normalize_close_groups
            ? fmax(
                round_step(
                    position_size - allocation.reducer_qty - kept_ordinary,
                    qty_step
                ),
                0.0f
            ) : 0.0f;
        if (allocation.dust_remainder > 0.0f
            && allocation.dust_remainder < allocation.minimum_any
            && allocation.last_kept_rank < 0 && has_reducer) {
            allocation.reducer_qty = fmin(
                position_size,
                round_step(
                    allocation.reducer_qty + allocation.dust_remainder,
                    qty_step
                )
            );
            allocation.ordinary_budget = fmax(
                round_step(
                    position_size - allocation.reducer_qty, qty_step
                ),
                0.0f
            );
            allocation.dust_remainder = 0.0f;
        }
        return allocation;
    }
    return allocation;
}

// Select from independently finalized candidates, retrying in Rust's
// preference order when the generation-time realized-loss gate rejects the
// current winner.
inline int select_recursive_close_reducer(
    thread const ReducerCandidate* candidates,
    bool is_long,
    thread const TmSide& source,
    float price_step,
    float market_order_slippage_pct,
    float maker_fee,
    float taker_fee,
    float c_mult,
    bool loss_gate_enabled,
    float max_realized_loss_pct
) {
    bool candidate_available[3] = {true, true, true};
    for (int attempt = 0; attempt < 3; ++attempt) {
        int preferred_idx = -1;
        for (int candidate_idx = 0; candidate_idx < 3; ++candidate_idx) {
            if (!candidate_available[candidate_idx]
                || !(candidates[candidate_idx].finalized_qty > 0.0f)) {
                continue;
            }
            if (preferred_idx < 0 || reducer_candidate_preferred(
                    candidates[candidate_idx].finalized_qty,
                    candidates[candidate_idx].ticks,
                    candidates[candidate_idx].order_type_id,
                    candidates[preferred_idx].finalized_qty,
                    candidates[preferred_idx].ticks,
                    candidates[preferred_idx].order_type_id,
                    is_long)) {
                preferred_idx = candidate_idx;
            }
        }
        if (preferred_idx < 0) break;
        ReducerCandidate preferred = candidates[preferred_idx];
        float gate_price = preferred.market
            ? ordinary_market_fill_price(
                source.close_gen_market_price, !is_long,
                market_order_slippage_pct, price_step
            ) : preferred.price;
        float gate_fee_rate = preferred.market ? taker_fee : maker_fee;
        if (realized_loss_proxy_allows_reducer(
                preferred.finalized_qty, gate_price,
                source.close_gen_pprice, is_long,
                preferred.is_unstuck, c_mult, gate_fee_rate,
                loss_gate_enabled, source.close_gen_balance,
                source.close_gen_realized_pnl_cumsum_last,
                source.close_gen_realized_pnl_cumsum_max,
                max_realized_loss_pct)) {
            return preferred_idx;
        }
        candidate_available[preferred_idx] = false;
    }
    return -1;
}

// The next-candle strategy contract expands recursive closes only when one of
// the immutable passive strategy orders would strictly fill. Conservative
// tick bounds decide when this exact probe is necessary; this reconstruction
// filters their harmless false positives before market policy can expose the
// otherwise-unemitted suffix. Strategy WEL remains part of this pre-loss-gate
// decision even if it is rejected later. Portfolio TWEL and unstuck reducers
// are appended after strategy generation, so they must not decide expansion.
inline bool recursive_strategy_close_would_expand(
    thread const TmSide& source, bool is_long,
    int high_fill_max_tick, int low_nonfill_max_tick,
    float qty_step, float price_step, float min_qty, float min_cost, float c_mult
) {
    TmSide sim = source;
    sim.psize = source.close_gen_psize;
    sim.pprice = source.close_gen_pprice;
    sim.market_orders_allowed = false;
    sim.twel_enforcer_enabled = false;
    sim.unstuck_enabled = false;
    for (int rung = 0; rung < 500; ++rung) {
        generate_orders(
            sim, is_long, source.close_gen_balance, source.close_gen_pprice,
            source.close_gen_touch_down_ticks, source.close_gen_touch_up_ticks,
            source.close_gen_touch_nearest_ticks, source.close_gen_touch_min_qty,
            source.close_gen_touch_min_qty_relation, qty_step, price_step,
            min_qty, min_cost, c_mult, 0.0f, false
        );
        float qty = round_step(sim.close_qty, qty_step);
        if (qty <= 0.0f || sim.close_ticks <= 0) break;
        bool reachable = is_long
            ? sim.close_ticks <= high_fill_max_tick
            : sim.close_ticks > low_nonfill_max_tick;
        if (reachable) return true;
        sim.psize = fmax(round_step(sim.psize - qty, qty_step), 0.0f);
        if (sim.psize <= 0.0f) break;
    }
    return false;
}

inline bool force_close_delisted_position(
    thread float& psize,
    thread float& pprice,
    thread float& pos_open_k,
    thread HslState& hsl,
    bool is_long,
    float close,
    float other_unrealized,
    float price_step,
    float c_mult,
    float taker_fee,
    float market_order_slippage_pct,
    float kf,
    thread float& balance,
    thread float& realized_pnl_cumsum_last,
    thread float& realized_pnl_cumsum_max,
    thread float& realized_pnl_cumsum_long,
    thread float& realized_pnl_cumsum_short,
    thread float& day_fill_count,
    thread float& fill_count,
    thread float& fill_count_entry,
    thread float& fill_count_long,
    thread float& pnl_recovery_peak,
    thread float& pnl_recovery_peak_k,
    thread float& pnl_recovery_max_min,
    thread HslRollingPnlWindow& rolling_pnl,
    device float2* rolling_pnl_values,
    device int2* rolling_pnl_indices,
    int rolling_base,
    int rolling_capacity,
    int pnl_lookback_bars,
    bool coin_hsl_rolling,
    thread float& profit_sum,
    thread float& loss_sum,
    thread float& side_profit_sum,
    thread float& side_loss_sum,
    thread float& held_max_min,
    thread float& held_sum_min,
    thread float& held_count,
    thread float& day_volume
) {
    if (!(psize > 0.0f && pprice > 0.0f)) return false;
    const float close_price = ordinary_market_fill_price(
        close, !is_long, market_order_slippage_pct, price_step
    );
    const float close_qty = psize;
    const float pnl = close_qty * c_mult * (
        is_long ? close_price - pprice : pprice - close_price
    );
    const float fee = close_qty * close_price * c_mult * taker_fee;
    const float net_pnl = pnl - fee;
    const float own_unrealized = close_qty * c_mult * (
        is_long ? close - pprice : pprice - close
    );
    record_hsl_panic_fill(
        hsl, net_pnl, balance + own_unrealized + other_unrealized
    );
    record_directional_gross_pnl(
        pnl, profit_sum, loss_sum, side_profit_sum, side_loss_sum
    );
    balance += net_pnl;
    record_realized_net(
        net_pnl, realized_pnl_cumsum_last, realized_pnl_cumsum_max,
        realized_pnl_cumsum_long, realized_pnl_cumsum_short,
        day_fill_count, fill_count, fill_count_entry, fill_count_long,
        pnl_recovery_peak, pnl_recovery_peak_k, pnl_recovery_max_min, kf,
        false, is_long
    );
    record_hsl_rolling_pnl(
        rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
        rolling_base, rolling_capacity, int(kf), pnl_lookback_bars,
        coin_hsl_rolling, net_pnl
    );
    if (pos_open_k >= 0.0f) {
        const float held_min = kf - pos_open_k;
        held_max_min = fmax(held_max_min, held_min);
        held_sum_min += held_min;
        held_count += 1.0f;
    }
    day_volume += close_qty * close_price / balance;
    psize = 0.0f;
    pprice = 0.0f;
    pos_open_k = -1.0f;
    return true;
}

inline void passivbot_single_coin_impl(
    constant float* bars,
    constant int* flags,
    constant float* params,
    constant float* settings,
    constant int* sizes,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float2* rolling_pnl_values,
    device int2* rolling_pnl_indices,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
    uint b
) {
    const int B = sizes[0];
    const int T = sizes[1];
    const int D = sizes[2];
    const int P = sizes[3];
    const int first_valid = sizes[4];
    const int last_valid = sizes[7];
    const int bounded_history_start = sizes[10];
    const int bounded_trade_start = sizes[11];
    const bool recent_history_window = bounded_history_start >= 0;
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    const int bounded_first_hour_step = sizes[13];
    const bool bounded_first_hour_ready = sizes[14] != 0;
    const int bounded_first_next_window_start = sizes[15];
#endif
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    const int recovery_stride = sizes[8];
    const int recovery_sample_capacity = sizes[9];
    const int recovery_sample_count = sizes[12];
#endif
    if (b >= uint(B)) return;

    const float qty_step = settings[0];
    const float price_step = settings[1];
    const float min_qty = settings[2];
    const float min_cost = settings[3];
    const float c_mult = settings[4];
    const float maker_fee = settings[5];
    const float starting_balance = settings[6];
    const float liq_floor = settings[7];
    const float interval_ms = settings[8];
#if defined(PASSIVBOT_TRAILING_LONG_ONLY)
    const bool long_enabled = true;
    const bool short_enabled = false;
#elif defined(PASSIVBOT_TRAILING_SHORT_ONLY)
    const bool long_enabled = false;
    const bool short_enabled = true;
#else
    const bool long_enabled = settings[9] > 0.5f;
    const bool short_enabled = settings[10] > 0.5f;
#endif
    const bool hedge_mode = settings[11] > 0.5f;
    const bool filter_by_min_effective_cost = settings[12] > 0.5f;
    const float max_effective_min_cost = settings[13];
    const float max_realized_loss_pct = settings[14];
    const float taker_fee = settings[15];
    const float market_order_slippage_pct = fmax(settings[16], 0.0f);
    const bool long_hsl_panic_market = settings[17] > 0.5f;
    const bool short_hsl_panic_market = settings[18] > 0.5f;
    const bool market_orders_allowed = !PASSIVBOT_TM_MARKET_ORDERS_DISABLED
        && settings[19] > 0.5f;
    const float market_order_near_touch_threshold = fmax(settings[20], 0.0f);
    const int pnl_lookback_bars = max(sizes[6], 0);
    const int rolling_capacity = sizes[5];
    const bool loss_gate_enabled = !PASSIVBOT_TM_LOSS_GATE_DISABLED
        && max_realized_loss_pct < 1.0f;
    const float log_bin_scale = 127.0f / log(4000001.0f);

    const int po = int(b) * P;
    const int seed_k = recent_history_window
        ? clamp(bounded_history_start, 0, T - 1)
        : clamp(first_valid, 0, T - 1);
    const float seed_close = bars[seed_k * 5 + 2];
    TmSide long_side = load_side(params, po, seed_close);
    TmSide short_side = load_side(params, po + SIDE_PARAMS, seed_close);
    long_side.market_orders_allowed = market_orders_allowed;
    long_side.market_order_near_touch_threshold = market_order_near_touch_threshold;
    short_side.market_orders_allowed = market_orders_allowed;
    short_side.market_order_near_touch_threshold = market_order_near_touch_threshold;
    HslState long_hsl = load_hsl(params, po, 40);
    HslState short_hsl = load_hsl(params, po + SIDE_PARAMS, 40);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    HslStrategyEquityStats long_hsl_strategy_eq = init_hsl_strategy_equity_stats();
    HslStrategyEquityStats short_hsl_strategy_eq = init_hsl_strategy_equity_stats();
#endif
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    HslDrawdownEmaTailStats long_hsl_ema_tail = init_hsl_drawdown_ema_tail_stats();
    HslDrawdownEmaTailStats short_hsl_ema_tail = init_hsl_drawdown_ema_tail_stats();
#endif
#if defined(PASSIVBOT_TRAILING_HSL_DISABLED)
    long_hsl.enabled = false;
    short_hsl.enabled = false;
#endif
    const bool long_coin_hsl_rolling = long_hsl.enabled
        && long_hsl.signal_mode == HSL_SIGNAL_COIN && pnl_lookback_bars > 0;
    const bool short_coin_hsl_rolling = short_hsl.enabled
        && short_hsl.signal_mode == HSL_SIGNAL_COIN && pnl_lookback_bars > 0;
    HslRollingPnlWindow long_rolling_pnl = init_hsl_rolling_pnl_window();
    HslRollingPnlWindow short_rolling_pnl = init_hsl_rolling_pnl_window();
    const int long_rolling_base = int(b) * 2 * rolling_capacity;
    const int short_rolling_base = long_rolling_base + rolling_capacity;
    const bool hsl_modes_valid = long_hsl.signal_mode == short_hsl.signal_mode;

    float balance = hsl_modes_valid ? starting_balance : 0.0f;
    float realized_pnl_cumsum_last = 0.0f;
    float realized_pnl_cumsum_max = 0.0f;
    float realized_pnl_cumsum_long = 0.0f;
    float realized_pnl_cumsum_short = 0.0f;
    float pnl_recovery_peak = -INFINITY;
    float pnl_recovery_peak_k = -1.0f;
    float pnl_recovery_max_min = 0.0f;
    float profit_sum = 0.0f;
    float loss_sum = 0.0f;
    float profit_sum_long = 0.0f;
    float loss_sum_long = 0.0f;
    float profit_sum_short = 0.0f;
    float loss_sum_short = 0.0f;
    float fill_count = 0.0f;
    float fill_count_entry = 0.0f;
    float fill_count_long = 0.0f;
    float fills_active_days_count = 0.0f;
    int last_active_fill_day = -1;
    bool alive = hsl_modes_valid;
    bool min_cost_exact_open_uncertain = false;
    int liq_day = hsl_modes_valid ? -1 : 0;
    float held_max_min = 0.0f;
    float held_sum_min = 0.0f;
    float held_count = 0.0f;
    float position_unchanged_max_min = 0.0f;
    float long_position_last_fill_k = -1.0f;
    float short_position_last_fill_k = -1.0f;
    float last_fill_k = -1.0f;
    float first_fill_k = -1.0f;
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    float long_last_initial_entry_k = -1.0f;
    float short_last_initial_entry_k = -1.0f;
#endif
    float gap_max_min = 0.0f;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float total_wallet_exposure_max = 0.0f;
    float total_wallet_exposure_mean = 0.0f;
    float total_wallet_exposure_samples = 0.0f;
    float last_high_k = -1.0f;
    float recovery_max_min = 0.0f;
    float account_peak = -INFINITY;
    float account_peak_k = -1.0f;
    float account_recovery_max_min = 0.0f;
    float first_eq_k = -1.0f;
    float last_eq_k = -1.0f;
    bool eq_started = false;
#if PASSIVBOT_BTC_RISK_ENABLED
    BtcRiskState btc_risk = init_btc_risk_state();
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    EquityBalanceDiffState equity_balance_diff_state =
        init_equity_balance_diff_state();
#endif
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    int recovery_start_k = -1;
#endif
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    float hsl_tier_samples_total = 0.0f;
    float hsl_tier_samples_yellow = 0.0f;
    float hsl_tier_samples_orange = 0.0f;
    float hsl_tier_samples_red = 0.0f;
#endif

    int cur_day = flags[seed_k * 11 + 2];
    bool day_touched = false;
    float day_end = 0.0f;
    float day_min = INFINITY;
    float day_dd = 0.0f;
    float day_volume = 0.0f;
    float day_has_fill = 0.0f;
    float day_start_balance = balance;
    float day_fill_count = 0.0f;

    for (int j = 0; j < GAP_BINS; ++j) {
        gap_hist[int(b) * GAP_BINS + j] = 0;
    }
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    init_entry_interval_output(
        entry_interval_stats, entry_interval_counts, b
    );
#endif

#if !PASSIVBOT_TM_VOLATILITY_DISABLED
    float bounded_hour_high = -INFINITY;
    float bounded_hour_low = INFINITY;
    float bounded_hour_latest = 0.0f;
    bool bounded_hour_has_price = false;
    bool bounded_hour_latest_valid = false;
    bool bounded_hour_synced = false;
    if (recent_history_window && seed_k <= last_valid) {
        const float seed_high = bars[seed_k * 5 + 0];
        const float seed_low = bars[seed_k * 5 + 1];
        if (isfinite(seed_high) && isfinite(seed_low)
                && seed_high > 0.0f && seed_low > 0.0f) {
            bounded_hour_high = seed_high;
            bounded_hour_low = seed_low;
            bounded_hour_has_price = true;
        }
    }
#endif

    const int loop_start = recent_history_window ? max(seed_k + 1, 1) : 1;
    for (int k = loop_start; k < T - 1; ++k) {
        const int bo = k * 5;
        const int fo = k * 11;
        const float high = bars[bo + 0];
        const float low = bars[bo + 1];
        const float close = bars[bo + 2];
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
        const float log_range = bars[bo + 3];
        float hour_lr = bars[bo + 4];
#endif
        const bool valid = flags[fo + 0] != 0;
        const bool can_gen = flags[fo + 1] != 0
            && (!recent_history_window || k >= bounded_trade_start);
        const int di = flags[fo + 2];
#if !PASSIVBOT_TM_VOLATILITY_DISABLED
        const int hour_flags = flags[fo + 3];
        bool hour_valid = (hour_flags & 1) != 0;
        if (recent_history_window && !bounded_hour_synced) {
            const bool hour_boundary = (hour_flags & 2) != 0;
            if (hour_boundary) {
                const bool bounded_window_ready = k == bounded_first_hour_step
                    ? bounded_first_hour_ready
                    : (hour_flags & 4) != 0;
                if (bounded_window_ready && bounded_hour_has_price
                        && bounded_hour_high > 0.0f && bounded_hour_low > 0.0f) {
                    bounded_hour_latest = log(
                        bounded_hour_high / bounded_hour_low
                    );
                    bounded_hour_latest_valid = isfinite(bounded_hour_latest);
                    bounded_hour_synced = bounded_hour_latest_valid;
                }
                hour_lr = bounded_hour_latest;
                hour_valid = bounded_hour_latest_valid;
                if (!bounded_hour_synced) {
                    const int next_window_start = k == bounded_first_hour_step
                        ? bounded_first_next_window_start
                        : ((hour_flags & 8) != 0 ? k - 1 : k);
                    bounded_hour_high = -INFINITY;
                    bounded_hour_low = INFINITY;
                    bounded_hour_has_price = false;
                    for (int hour_k = max(seed_k, next_window_start);
                            hour_k <= k && hour_k <= last_valid; ++hour_k) {
                        const float hour_high = bars[hour_k * 5 + 0];
                        const float hour_low = bars[hour_k * 5 + 1];
                        if (isfinite(hour_high) && isfinite(hour_low)
                                && hour_high > 0.0f && hour_low > 0.0f) {
                            bounded_hour_high = fmax(
                                bounded_hour_high, hour_high
                            );
                            bounded_hour_low = fmin(
                                bounded_hour_low, hour_low
                            );
                            bounded_hour_has_price = true;
                        }
                    }
                }
            } else {
                hour_valid = false;
                if (k <= last_valid && isfinite(high) && isfinite(low)
                        && high > 0.0f && low > 0.0f) {
                    bounded_hour_high = fmax(bounded_hour_high, high);
                    bounded_hour_low = fmin(bounded_hour_low, low);
                    bounded_hour_has_price = true;
                }
            }
        }
#endif
        const int high_fill_max_tick = flags[fo + 4];
        const int low_nonfill_max_tick = flags[fo + 5];
        const int touch_down_tick = flags[fo + 6];
        const int touch_up_tick = flags[fo + 7];
        const int touch_nearest_tick = flags[fo + 8];
        const float touch_min_qty = as_type<float>(flags[fo + 9]);
        const int touch_min_qty_relation = flags[fo + 10];
        const float kf = float(k);

        if (!valid) {
            clear_pending_tm_orders(long_side);
            clear_pending_tm_orders(short_side);
        }

        if (di != cur_day) {
            if (day_touched && cur_day >= 0 && cur_day < D) {
                int o = (int(b) * D + cur_day) * DAILY_COLS;
                daily[o + 0] = day_end;
                daily[o + 1] = day_min;
                daily[o + 2] = day_dd;
                daily[o + 3] = day_volume;
                daily[o + 4] = day_has_fill;
                daily[o + 5] = balance - day_start_balance;
                daily[o + 6] = balance;
                daily[o + 7] = day_fill_count;
#if PASSIVBOT_BTC_RISK_ENABLED
                write_btc_risk_day(btc_risk, daily, o, 8);
#endif
            }
            cur_day = di;
            day_touched = false;
            day_end = 0.0f;
            day_min = INFINITY;
            day_dd = 0.0f;
            day_volume = 0.0f;
            day_has_fill = 0.0f;
            day_start_balance = balance;
            day_fill_count = 0.0f;
#if PASSIVBOT_BTC_RISK_ENABLED
            reset_btc_risk_day(btc_risk);
#endif
        }

        bool long_close_fill = false;
        bool long_entry_fill = false;
        bool short_close_fill = false;
        bool short_entry_fill = false;
#if !defined(PASSIVBOT_TRAILING_SHORT_ONLY)
        bool long_close_ready = valid && alive && long_enabled
            && long_side.close_qty > 0.0f && long_side.psize > 0.0f;
        bool long_secondary_close_fill = valid && alive && long_enabled
            && long_side.secondary_close_qty > 0.0f
            && (long_side.secondary_close_market
                || long_side.secondary_close_ticks <= high_fill_max_tick)
            && long_side.psize > 0.0f;
        const bool long_recursive_close = !PASSIVBOT_TM_TRAILING_CLOSE_ONLY
            && long_side.close_retracement_base <= 0.0f;
        bool long_close_fill_ready = long_close_ready
            && ((long_side.close_is_panic && long_hsl_panic_market)
                || long_side.close_market
                || long_side.close_ticks <= high_fill_max_tick);
        // Exact Rust expands an immutable recursive ladder only when at least
        // one original passive close would fill the next candle. A market-only
        // next close remains a single order and must not expose farther rungs.
        bool long_expand_close_grid = long_close_ready && long_recursive_close
            && !long_side.close_is_panic
            && long_side.close_ticks <= high_fill_max_tick;
        if (long_close_ready && long_recursive_close
            && long_side.close_is_exposure_reducer) {
            // A touch-clamped grid order uses nearest-tick quantization while
            // the passive reducer rounds up.  The grid may therefore be one
            // tick nearer and independently reachable.
            long_expand_close_grid = long_expand_close_grid
                || long_side.close_gen_touch_nearest_ticks <= high_fill_max_tick;
        }
        if (long_close_ready && long_recursive_close
            && long_side.close_threshold_we > 0.0f) {
            // Positive WE weight makes later generated long closes nearer.
            // The zero-WE target is a conservative lower bound: reconstruct
            // the sorted grid only if this candle can reach that bound.
#if PASSIVBOT_TM_VOLATILITY_DISABLED
            float threshold_floor = long_side.close_threshold_base;
#else
            float threshold_floor = long_side.close_threshold_base
                + long_side.vol1h * long_side.close_threshold_v1h
                + long_side.vol1m * long_side.close_threshold_v1m;
#endif
            int target_ticks = directional_ticks(
                long_side.close_gen_pprice * (1.0f + threshold_floor),
                price_step, true
            );
            bool touch_controls = long_side.close_gen_touch_up_ticks > target_ticks;
            int nearest_ticks = touch_controls
                ? long_side.close_gen_touch_nearest_ticks : target_ticks;
            long_expand_close_grid = long_expand_close_grid
                || nearest_ticks <= high_fill_max_tick;
        }
        if (long_expand_close_grid) {
            long_expand_close_grid = recursive_strategy_close_would_expand(
                long_side, true, high_fill_max_tick, low_nonfill_max_tick,
                qty_step, price_step, min_qty, min_cost, c_mult
            );
        }
        if (long_expand_close_grid) {
            TmSide grid_source = long_side;
            int strategy_wel_ticks = long_side.close_gen_touch_up_ticks;
            float strategy_wel_price = float(strategy_wel_ticks) * price_step;
            float strategy_wel_qty = long_side.wel_enforcer_enabled
                ? exposure_reducer_qty(
                    long_side.close_gen_psize,
                    long_side.close_gen_pprice,
                    long_side.close_gen_balance,
                    long_side.allowed_wel * long_side.wel_enforcer_threshold,
                    strategy_wel_price,
                    qty_step, min_qty, min_cost, c_mult
                ) : 0.0f;
            grid_source.close_gen_psize = fmax(
                round_step(
                    long_side.close_gen_psize - strategy_wel_qty, qty_step
                ),
                0.0f
            );
            grid_source.wel_enforcer_enabled = false;
            grid_source.twel_enforcer_enabled = false;
            grid_source.unstuck_enabled = false;
            int prefix_merge_ticks = strategy_wel_qty > 0.0f
                ? strategy_wel_ticks : 0;
            float prefix_merge_qty = strategy_wel_qty > 0.0f
                ? strategy_wel_qty : 0.0f;
            CloseGroup group;
            int grid_rung_limit = strategy_wel_qty > 0.0f ? 499 : 500;
            int group_count = recursive_close_groups(
                grid_source, true, -1, qty_step, price_step,
                min_qty, min_cost, c_mult, long_side.close_gen_psize,
                prefix_merge_ticks, prefix_merge_qty,
                grid_rung_limit, group
            );
            bool strategy_wel_merged = false;
            if (prefix_merge_qty > 0.0f && group_count > 0) {
                recursive_close_groups(
                    grid_source, true, 0, qty_step, price_step,
                    min_qty, min_cost, c_mult, long_side.close_gen_psize,
                    prefix_merge_ticks, prefix_merge_qty,
                    grid_rung_limit, group
                );
                strategy_wel_merged = group.ticks == strategy_wel_ticks;
            }

            ReducerCandidate selected_reducer = empty_reducer_candidate();
            RecursiveCloseAllocation allocation;
#if PASSIVBOT_TM_REDUCERS_DISABLED
            allocation = recursive_close_allocation(
                grid_source, true, group_count,
                long_side.close_gen_psize,
                0.0f, 0.0f, false,
                qty_step, price_step, min_qty, min_cost, c_mult,
                prefix_merge_ticks, prefix_merge_qty,
                grid_rung_limit
            );
            long_side.close_is_exposure_reducer = false;
            long_side.close_is_twel_reducer = false;
            long_side.close_is_unstuck_reducer = false;
            long_side.close_loss_gate_disabled_reducers = false;
#else
            ReducerCandidate candidates[3];
            RecursiveCloseAllocation candidate_allocations[3];
            for (int candidate_idx = 0; candidate_idx < 3; ++candidate_idx) {
                candidates[candidate_idx] = empty_reducer_candidate();
            }

            candidates[0].requested_qty = strategy_wel_merged
                ? 0.0f : strategy_wel_qty;
            candidates[0].ticks = strategy_wel_ticks;
            candidates[0].price = strategy_wel_price;
            candidates[0].order_type_id = 24;
            candidates[0].market = candidates[0].requested_qty > 0.0f
                && should_use_ordinary_market_execution(
                    candidates[0].ticks, false,
                    long_side.close_gen_market_price, price_step,
                    long_side.market_orders_allowed,
                    long_side.market_order_near_touch_threshold
                );
            if (candidates[0].market) {
                candidates[0].requested_qty = resize_market_close_qty(
                    candidates[0].requested_qty,
                    long_side.close_gen_psize,
                    long_side.close_gen_market_price,
                    qty_step, min_qty, min_cost, c_mult
                );
            }

            candidates[1].ticks = max(
                int(floor(
                    long_side.close_gen_market_price * 0.9995f / price_step
                        + 1.0e-6f
                )),
                1
            );
            candidates[1].price = float(candidates[1].ticks) * price_step;
            candidates[1].requested_qty = long_side.twel_enforcer_enabled
                    && long_side.twel_enforcer_threshold > 0.0f
                ? total_exposure_reducer_qty(
                    long_side.close_gen_psize,
                    long_side.close_gen_pprice,
                    long_side.close_gen_balance,
                    long_side.twel * long_side.twel_enforcer_threshold,
                    candidates[1].price,
                    qty_step, min_qty, min_cost, c_mult
                ) : 0.0f;
            candidates[1].order_type_id = 10;
            candidates[1].is_twel = true;
            candidates[1].market = candidates[1].requested_qty > 0.0f
                && should_use_ordinary_market_execution(
                    candidates[1].ticks, false,
                    long_side.close_gen_market_price, price_step,
                    long_side.market_orders_allowed,
                    long_side.market_order_near_touch_threshold
                );
            if (candidates[1].market) {
                candidates[1].requested_qty = resize_market_close_qty(
                    candidates[1].requested_qty,
                    long_side.close_gen_psize,
                    long_side.close_gen_market_price,
                    qty_step, min_qty, min_cost, c_mult
                );
            }

            TmSide unstuck_source = long_side;
            unstuck_source.psize = long_side.close_gen_psize;
            unstuck_source.pprice = long_side.close_gen_pprice;
            unstuck_source.unstuck_enabled =
                long_side.close_gen_unstuck_candidate_enabled;
            candidates[2].ticks = strategy_wel_ticks;
            candidates[2].price = strategy_wel_price;
            candidates[2].requested_qty = unstuck_reducer_qty(
                unstuck_source, true, long_side.close_gen_balance,
                candidates[2].price,
                qty_step, min_qty, min_cost, c_mult
            );
            candidates[2].order_type_id = 9;
            candidates[2].is_unstuck = true;
            candidates[2].market = candidates[2].requested_qty > 0.0f
                && should_use_ordinary_market_execution(
                    candidates[2].ticks, false,
                    long_side.close_gen_market_price, price_step,
                    long_side.market_orders_allowed,
                    long_side.market_order_near_touch_threshold
                );
            if (candidates[2].market) {
                candidates[2].requested_qty = resize_market_close_qty(
                    candidates[2].requested_qty,
                    long_side.close_gen_psize,
                    long_side.close_gen_market_price,
                    qty_step, min_qty, min_cost, c_mult
                );
            }

            for (int candidate_idx = 0; candidate_idx < 3; ++candidate_idx) {
                candidate_allocations[candidate_idx] =
                    recursive_close_allocation(
                        grid_source, true, group_count,
                        long_side.close_gen_psize,
                        candidates[candidate_idx].requested_qty,
                        candidates[candidate_idx].price,
                        candidates[candidate_idx].market,
                        qty_step, price_step, min_qty, min_cost, c_mult,
                        prefix_merge_ticks, prefix_merge_qty,
                        grid_rung_limit
                    );
                candidates[candidate_idx].finalized_qty =
                    candidate_allocations[candidate_idx].reducer_qty;
            }

            int selected_candidate_idx = select_recursive_close_reducer(
                candidates, true, long_side, price_step,
                market_order_slippage_pct, maker_fee, taker_fee, c_mult,
                loss_gate_enabled, max_realized_loss_pct
            );

            if (selected_candidate_idx >= 0) {
                selected_reducer = candidates[selected_candidate_idx];
                allocation = candidate_allocations[selected_candidate_idx];
            } else {
                allocation = recursive_close_allocation(
                    grid_source, true, group_count,
                    long_side.close_gen_psize,
                    0.0f, 0.0f, false,
                    qty_step, price_step, min_qty, min_cost, c_mult,
                    prefix_merge_ticks, prefix_merge_qty,
                    grid_rung_limit
                );
            }
            long_side.close_is_exposure_reducer =
                selected_candidate_idx >= 0;
            long_side.close_is_twel_reducer = selected_reducer.is_twel;
            long_side.close_is_unstuck_reducer = selected_reducer.is_unstuck;
            long_side.close_loss_gate_disabled_reducers =
                selected_candidate_idx < 0
                && (candidates[0].finalized_qty > 0.0f
                    || candidates[1].finalized_qty > 0.0f
                    || candidates[2].finalized_qty > 0.0f);
#endif

            float reducer_qty = allocation.reducer_qty;
            int reducer_ticks = selected_reducer.ticks;
            float reducer_price = selected_reducer.price;
            bool reducer_market = selected_reducer.market;
            bool reverse = grid_source.close_threshold_we > 0.0f;
            float ordinary_budget = allocation.ordinary_budget;
            float remaining_budget = ordinary_budget;
            float minimum_any = allocation.minimum_any;
            int last_kept_rank = allocation.last_kept_rank;
            int collapse_ordinary_rank = allocation.collapse_ordinary_rank;
            float dust_remainder = allocation.dust_remainder;
            bool normalize_close_groups = allocation.normalize_close_groups;
            bool reducer_reachable = reducer_qty > 0.0f
                && (reducer_market || reducer_ticks <= high_fill_max_tick);
            bool reducer_executed = false;
            remaining_budget = ordinary_budget;
            for (int rank = 0; rank < group_count; ++rank) {
                int wanted = reverse ? group_count - rank - 1 : rank;
                recursive_close_groups(
                    grid_source, true, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, long_side.close_gen_psize,
                    prefix_merge_ticks, prefix_merge_qty,
                    grid_rung_limit, group
                );
                if (group.qty <= 0.0f) break;
                float trimmed_group_qty = group.qty;
                if (collapse_ordinary_rank >= 0) {
                    trimmed_group_qty = rank == collapse_ordinary_rank
                        ? long_side.close_gen_psize : 0.0f;
                } else if (normalize_close_groups) {
                    float group_min = min_entry_qty(
                        group.market ? long_side.close_gen_market_price
                                     : group.price,
                        qty_step, min_qty, min_cost, c_mult
                    );
                    trimmed_group_qty = fmin(group.qty, remaining_budget);
                    bool partial_trim = quantity_is_meaningfully_below(
                        trimmed_group_qty, group.qty
                    );
                    if (quantity_is_meaningfully_below(
                            trimmed_group_qty, group_min
                        )) {
                        trimmed_group_qty = 0.0f;
                        if (partial_trim) remaining_budget = 0.0f;
                    }
                    if (trimmed_group_qty > 0.0f) {
                        remaining_budget = fmax(
                            round_step(
                                remaining_budget - trimmed_group_qty, qty_step
                            ),
                            0.0f
                        );
                        if (rank == last_kept_rank && dust_remainder > 0.0f
                            && dust_remainder < minimum_any) {
                            trimmed_group_qty = round_step(
                                trimmed_group_qty + dust_remainder, qty_step
                            );
                        }
                    }
                }
                bool reducer_before_group = reducer_reachable
                    && !reducer_executed && reducer_ticks < group.ticks;
                if (reducer_before_group) {
                    float reducer_fill_price = reducer_market
                        ? ordinary_market_fill_price(
                            close, false, market_order_slippage_pct, price_step
                        ) : reducer_price;
                    float reducer_fee_rate = reducer_market
                        ? taker_fee : maker_fee;
                    float pnl = reducer_qty * c_mult
                        * (reducer_fill_price - long_side.pprice);
                    float fee = reducer_qty * reducer_fill_price * c_mult
                        * reducer_fee_rate;
                    // The finalized reducer was admitted against the order
                    // book captured at generation.  The next-candle price is
                    // only the realized fill price; it must not re-gate an
                    // already emitted market order.
                    {
                        if (long_side.close_is_panic) {
                            record_hsl_panic_fill(
                                long_hsl, pnl - fee,
                                directional_equity_at_close(
                                    balance, long_side, short_side, close, c_mult
                                )
                            );
                        }
                        record_directional_gross_pnl(
                            pnl, profit_sum, loss_sum,
                            profit_sum_long, loss_sum_long
                        );
                        balance += pnl - fee;
                        record_realized_net(
                            pnl - fee,
                            realized_pnl_cumsum_last,
                            realized_pnl_cumsum_max,
                            realized_pnl_cumsum_long,
                            realized_pnl_cumsum_short,
                            day_fill_count,
                            fill_count,
                            fill_count_entry,
                            fill_count_long,
                            pnl_recovery_peak,
                            pnl_recovery_peak_k,
                            pnl_recovery_max_min,
                            kf,
                            false,
                            true
                        );
                        record_hsl_rolling_pnl(
                            long_rolling_pnl,
                            rolling_pnl_values, rolling_pnl_indices,
                            long_rolling_base, rolling_capacity, int(kf),
                            pnl_lookback_bars, long_coin_hsl_rolling, pnl - fee
                        );
                        long_side.psize = fmax(
                            round_step(
                                long_side.psize - reducer_qty, qty_step
                            ),
                            0.0f
                        );
                        day_volume += reducer_qty * reducer_fill_price / balance;
                        long_close_fill = true;
                        reducer_executed = true;
                    }
                }
                if (!group.market && group.ticks > high_fill_max_tick) break;
                float group_qty = trimmed_group_qty;
                if (group_qty <= 0.0f) continue;
                float adj = fmin(round_step(group_qty, qty_step), long_side.psize);
                float group_fill_price = group.market
                    ? ordinary_market_fill_price(
                        close, false, market_order_slippage_pct, price_step
                    ) : group.price;
                float group_gate_price = group.market
                    ? ordinary_market_fill_price(
                        long_side.close_gen_market_price, false,
                        market_order_slippage_pct, price_step
                    ) : group.price;
                float group_fee_rate = group.market ? taker_fee : maker_fee;
                float pnl = adj * c_mult
                    * (group_fill_price - long_side.pprice);
                float fee = adj * group_fill_price * c_mult * group_fee_rate;
                if (!realized_loss_proxy_allows_close(
                        adj, group_gate_price, long_side.pprice, true,
                        c_mult, group_fee_rate, loss_gate_enabled)) continue;
                if (long_side.close_is_panic) {
                    record_hsl_panic_fill(
                        long_hsl, pnl - fee,
                        directional_equity_at_close(
                            balance, long_side, short_side, close, c_mult
                        )
                    );
                }
                record_directional_gross_pnl(
                    pnl, profit_sum, loss_sum, profit_sum_long, loss_sum_long
                );
                balance += pnl - fee;
                record_realized_net(
                    pnl - fee,
                    realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    realized_pnl_cumsum_long,
                    realized_pnl_cumsum_short,
                    day_fill_count,
                    fill_count,
                    fill_count_entry,
                    fill_count_long,
                    pnl_recovery_peak,
                    pnl_recovery_peak_k,
                    pnl_recovery_max_min,
                    kf,
                    false,
                    true
                );
                record_hsl_rolling_pnl(
                    long_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                    long_rolling_base, rolling_capacity, int(kf),
                    pnl_lookback_bars, long_coin_hsl_rolling, pnl - fee
                );
                float new_psize = fmax(
                    round_step(long_side.psize - adj, qty_step), 0.0f
                );
                bool went_flat = new_psize <= 0.0f;
                long_side.psize = new_psize;
                day_volume += fabs(adj) * group_fill_price / balance;
                long_close_fill = true;
                if (went_flat) {
                    long_side.pprice = 0.0f;
                    if (long_side.pos_open_k >= 0.0f) {
                        float held_min = kf - long_side.pos_open_k;
                        held_max_min = fmax(held_max_min, held_min);
                        held_sum_min += held_min;
                        held_count += 1.0f;
                    }
                    long_side.pos_open_k = -1.0f;
                    break;
                }
            }
            if (reducer_reachable && !reducer_executed
                && long_side.psize > 0.0f) {
                float adj = fmin(reducer_qty, long_side.psize);
                float reducer_fill_price = reducer_market
                    ? ordinary_market_fill_price(
                        close, false, market_order_slippage_pct, price_step
                    ) : reducer_price;
                float reducer_fee_rate = reducer_market
                    ? taker_fee : maker_fee;
                float pnl = adj * c_mult
                    * (reducer_fill_price - long_side.pprice);
                float fee = adj * reducer_fill_price * c_mult
                    * reducer_fee_rate;
                // Selection already applied the generation-time loss gate.
                {
                    if (long_side.close_is_panic) {
                        record_hsl_panic_fill(
                            long_hsl, pnl - fee,
                            directional_equity_at_close(
                                balance, long_side, short_side, close, c_mult
                            )
                        );
                    }
                    record_directional_gross_pnl(
                        pnl, profit_sum, loss_sum, profit_sum_long, loss_sum_long
                    );
                    balance += pnl - fee;
                    record_realized_net(
                        pnl - fee,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        realized_pnl_cumsum_long,
                        realized_pnl_cumsum_short,
                        day_fill_count,
                        fill_count,
                        fill_count_entry,
                        fill_count_long,
                        pnl_recovery_peak,
                        pnl_recovery_peak_k,
                        pnl_recovery_max_min,
                        kf,
                        false,
                        true
                    );
                    record_hsl_rolling_pnl(
                        long_rolling_pnl,
                        rolling_pnl_values, rolling_pnl_indices,
                        long_rolling_base, rolling_capacity, int(kf),
                        pnl_lookback_bars, long_coin_hsl_rolling, pnl - fee
                    );
                    long_side.psize = fmax(
                        round_step(long_side.psize - adj, qty_step), 0.0f
                    );
                    day_volume += adj * reducer_fill_price / balance;
                    long_close_fill = true;
                }
            }
            if (long_close_fill && long_side.psize <= 0.0f
                && long_side.pprice > 0.0f) {
                long_side.pprice = 0.0f;
                if (long_side.pos_open_k >= 0.0f) {
                    float held_min = kf - long_side.pos_open_k;
                    held_max_min = fmax(held_max_min, held_min);
                    held_sum_min += held_min;
                    held_count += 1.0f;
                }
                long_side.pos_open_k = -1.0f;
            }
            if (long_close_fill) long_side.close_qty = 0.0f;
        } else if (long_close_fill_ready || long_secondary_close_fill) {
            bool secondary_first = long_secondary_close_fill
                && (!long_close_fill_ready
                    || long_side.secondary_close_price
                        <= long_side.close_price);
            for (int rank = 0; rank < 2; ++rank) {
                bool use_secondary = secondary_first ? rank == 0 : rank == 1;
                bool reachable = use_secondary
                    ? long_secondary_close_fill : long_close_fill_ready;
                if (!reachable || long_side.psize <= 0.0f) continue;
                bool market_panic = !use_secondary && long_side.close_is_panic
                    && long_hsl_panic_market;
                bool ordinary_market = use_secondary
                    ? long_side.secondary_close_market
                    : long_side.close_market;
                bool market_execution = market_panic || ordinary_market;
                float cp = market_execution
                        ? float(max(directional_ticks(
                            close * (1.0f - market_order_slippage_pct),
                            price_step, false
                        ), 1)) * price_step
                        : use_secondary ? long_side.secondary_close_price
                                        : long_side.close_price;
                float requested_qty = use_secondary
                    ? long_side.secondary_close_qty : long_side.close_qty;
                float adj = fmin(
                    round_step(requested_qty, qty_step), long_side.psize
                );
                float pnl = adj * c_mult * (cp - long_side.pprice);
                float fee = adj * cp * c_mult
                    * (market_execution ? taker_fee : maker_fee);
                bool selected_unstuck = !use_secondary
                    && long_side.close_is_unstuck_reducer;
                if (!long_side.close_is_panic
                    && !realized_loss_proxy_allows_reducer(
                        adj, cp, long_side.pprice, true, selected_unstuck,
                        c_mult, market_execution ? taker_fee : maker_fee,
                        loss_gate_enabled, balance,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        max_realized_loss_pct)) continue;
                if (!use_secondary && long_side.close_is_panic) {
                    record_hsl_panic_fill(
                        long_hsl, pnl - fee,
                        directional_equity_at_close(
                            balance, long_side, short_side, close, c_mult
                        )
                    );
                }
                record_directional_gross_pnl(
                    pnl, profit_sum, loss_sum, profit_sum_long, loss_sum_long
                );
                balance += pnl - fee;
                record_realized_net(
                    pnl - fee,
                    realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    realized_pnl_cumsum_long,
                    realized_pnl_cumsum_short,
                    day_fill_count,
                    fill_count,
                    fill_count_entry,
                    fill_count_long,
                    pnl_recovery_peak,
                    pnl_recovery_peak_k,
                    pnl_recovery_max_min,
                    kf,
                    false,
                    true
                );
                record_hsl_rolling_pnl(
                    long_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                    long_rolling_base, rolling_capacity, int(kf),
                    pnl_lookback_bars, long_coin_hsl_rolling, pnl - fee
                );
                long_side.psize = fmax(
                    round_step(long_side.psize - adj, qty_step), 0.0f
                );
                day_volume += fabs(adj) * cp / balance;
                long_close_fill = true;
            }
            bool went_flat = long_side.psize <= 0.0f;
            if (went_flat) {
                long_side.pprice = 0.0f;
                if (long_side.pos_open_k >= 0.0f) {
                    float held_min = kf - long_side.pos_open_k;
                    held_max_min = fmax(held_max_min, held_min);
                    held_sum_min += held_min;
                    held_count += 1.0f;
                }
                long_side.pos_open_k = -1.0f;
            }
            long_side.close_qty = 0.0f;
            long_side.secondary_close_qty = 0.0f;
        }

        bool long_entry_passive_reachable = long_side.entry_qty > 0.0f
            && long_side.entry_ticks > low_nonfill_max_tick;
        long_entry_fill = valid && alive && long_enabled
            && long_side.entry_qty > 0.0f
            && (long_side.entry_market
                || long_entry_passive_reachable);
        if (long_entry_fill) {
            long_entry_fill = false;
            TmSide ladder_side = long_side;
            ladder_side.psize = long_side.entry_gen_psize;
            ladder_side.pprice = long_side.entry_gen_pprice;
            // Rust constructs the immutable recursive strategy ladder before
            // orchestration promotes individual rungs to market execution.
            ladder_side.market_orders_allowed = false;
            const float ladder_balance = long_side.entry_gen_balance;
            const float ladder_kf = long_side.entry_gen_kf;
            int ladder_touch_ticks = long_side.entry_gen_touch_ticks;
            float ladder_market_price = long_side.entry_gen_market_price;
            TmSide gate_side = ladder_side;
            // Rebuild the immutable strategy ladder against allowed WEL only.
            // Portfolio TWEL gating is streamed separately through gate_side.
            ladder_side.twel_entry_gate_enabled = false;
            int previous_ticks = 0;
            for (int rung = 0; rung < 500; ++rung) {
                int entry_ticks = rung == 0
                    ? long_side.entry_ticks : ladder_side.entry_ticks;
                float ep = rung == 0
                    ? long_side.entry_price : ladder_side.entry_price;
                float strategy_eq = round_step(
                    rung == 0
                        ? long_side.entry_strategy_qty
                        : ladder_side.entry_qty,
                    qty_step
                );
                if (strategy_eq <= 0.0f) break;
                float eq = rung == 0 ? long_side.entry_qty : strategy_eq;
                float ungated_eq = strategy_eq;
                bool entry_market = should_use_ordinary_market_execution(
                    entry_ticks, true, ladder_market_price, price_step,
                    long_side.market_orders_allowed,
                    long_side.market_order_near_touch_threshold
                );
                if (rung > 0 && gate_side.twel_entry_gate_enabled) {
                    float entry_gate_price = entry_market
                        ? ladder_market_price : ep;
                    eq = gate_entry_by_twel_strict(
                        gate_side, ladder_balance, entry_gate_price, eq,
                        qty_step, min_qty, min_cost, c_mult
                    );
                    // Exact Rust removes farthest entries first.  A rejected
                    // boundary rung therefore closes the retained prefix;
                    // smaller later quantities may not reappear behind it.
                    if (!(eq > 0.0f)) break;
                }
                bool twel_boundary_partial = gate_side.twel_entry_gate_enabled
                    && eq > 0.0f && eq < ungated_eq;
                if ((!entry_market && entry_ticks <= low_nonfill_max_tick)
                    || (rung > 0 && entry_ticks == previous_ticks)) break;
                if (eq > 0.0f) {
                    float entry_fill_price = entry_market
                        ? ordinary_market_fill_price(
                            close, true, market_order_slippage_pct, price_step
                        ) : ep;
                    float fee = eq * entry_fill_price * c_mult
                        * (entry_market ? taker_fee : maker_fee);
                    balance -= fee;
                    record_realized_net(
                        -fee,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        realized_pnl_cumsum_long,
                        realized_pnl_cumsum_short,
                        day_fill_count,
                        fill_count,
                        fill_count_entry,
                        fill_count_long,
                        pnl_recovery_peak,
                        pnl_recovery_peak_k,
                        pnl_recovery_max_min,
                        kf,
                        true,
                        true
                    );
                    record_hsl_rolling_pnl(
                        long_rolling_pnl, rolling_pnl_values,
                        rolling_pnl_indices, long_rolling_base,
                        rolling_capacity, int(kf), pnl_lookback_bars,
                        long_coin_hsl_rolling, -fee
                    );
                    bool was_flat = long_side.psize <= 0.0f;
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
                    if (rung == 0 && long_side.entry_gen_psize <= 0.0f) {
                        record_initial_entry_interval(
                            entry_interval_stats, entry_interval_counts, b,
                            long_last_initial_entry_k, kf
                        );
                    }
#endif
                    float new_psize = round_step(
                        long_side.psize + eq, qty_step
                    );
                    float new_pprice = was_flat ? entry_fill_price
                        : long_side.pprice
                            * (long_side.psize / fmax(new_psize, 1.0e-12f))
                            + entry_fill_price
                                * (eq / fmax(new_psize, 1.0e-12f));
                    if (was_flat) long_side.pos_open_k = kf;
                    long_side.psize = new_psize;
                    long_side.pprice = new_pprice;
                    long_side.last_inc_k = kf;
                    day_volume += fabs(eq) * entry_fill_price / balance;
                    long_entry_fill = true;

                    if (gate_side.twel_entry_gate_enabled) {
                        float entry_gate_price = entry_market
                            ? ladder_market_price : ep;
                        bool gate_flat = gate_side.psize <= 0.0f;
                        float gate_psize = round_step(
                            gate_side.psize + eq, qty_step
                        );
                        gate_side.pprice = gate_flat ? entry_gate_price
                            : gate_side.pprice * (
                                gate_side.psize / fmax(gate_psize, 1.0e-12f)
                            ) + entry_gate_price * (
                                eq / fmax(gate_psize, 1.0e-12f)
                            );
                        gate_side.psize = gate_psize;
                    }
                }
                // Exact Rust keeps at most one partially retained boundary
                // order after removing the farther suffix.
                if (twel_boundary_partial) break;

                bool sim_flat = ladder_side.psize <= 0.0f;
                float sim_psize = round_step(
                    ladder_side.psize + strategy_eq, qty_step
                );
                ladder_side.pprice = sim_flat ? ep
                    : ladder_side.pprice
                        * (ladder_side.psize / fmax(sim_psize, 1.0e-12f))
                        + ep * (strategy_eq / fmax(sim_psize, 1.0e-12f));
                ladder_side.psize = sim_psize;
                previous_ticks = entry_ticks;
                ladder_touch_ticks = min(ladder_touch_ticks, entry_ticks);
                // Exact Rust expands the immutable recursive ladder only when
                // the original passive next entry strictly crosses the next
                // candle. Market promotion guarantees rung zero's execution,
                // but does not itself authorize expansion.
                if (rung == 0 && !long_entry_passive_reachable) break;
#if PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY
                if (long_side.cooldown_min != 0.0f) break;
#else
                if (PASSIVBOT_TM_TRAILING_ENTRY_ONLY
                    || long_side.entry_retracement_base > 0.0f
                    || long_side.cooldown_min != 0.0f) break;
#endif
                generate_long_orders(
                    ladder_side, ladder_balance, ep, qty_step,
                    ladder_touch_ticks, ladder_touch_ticks, ladder_touch_ticks,
                    touch_min_qty, touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, ladder_kf, false
                );
            }
            long_side.entry_qty = 0.0f;
        }
#endif

#if !defined(PASSIVBOT_TRAILING_LONG_ONLY)
        bool short_close_ready = valid && alive && short_enabled
            && short_side.close_qty > 0.0f && short_side.psize > 0.0f;
        bool short_secondary_close_fill = valid && alive && short_enabled
            && short_side.secondary_close_qty > 0.0f
            && (short_side.secondary_close_market
                || short_side.secondary_close_ticks > low_nonfill_max_tick)
            && short_side.psize > 0.0f;
        const bool short_recursive_close = !PASSIVBOT_TM_TRAILING_CLOSE_ONLY
            && short_side.close_retracement_base <= 0.0f;
        bool short_close_fill_ready = short_close_ready
            && ((short_side.close_is_panic && short_hsl_panic_market)
                || short_side.close_market
                || short_side.close_ticks > low_nonfill_max_tick);
        bool short_expand_close_grid = short_close_ready && short_recursive_close
            && !short_side.close_is_panic
            && short_side.close_ticks > low_nonfill_max_tick;
        if (short_close_ready && short_recursive_close
            && short_side.close_is_exposure_reducer) {
            // Mirror the long-side nearest-tick scan: a touch-clamped grid
            // order can sit one tick above the down-rounded reducer.
            short_expand_close_grid = short_expand_close_grid
                || short_side.close_gen_touch_nearest_ticks > low_nonfill_max_tick;
        }
        if (short_close_ready && short_recursive_close
            && short_side.close_threshold_we > 0.0f) {
            // Positive WE weight makes later generated short closes nearer.
            // The zero-WE target is a conservative upper bound.
#if PASSIVBOT_TM_VOLATILITY_DISABLED
            float threshold_floor = short_side.close_threshold_base;
#else
            float threshold_floor = short_side.close_threshold_base
                + short_side.vol1h * short_side.close_threshold_v1h
                + short_side.vol1m * short_side.close_threshold_v1m;
#endif
            int target_ticks = directional_ticks(
                short_side.close_gen_pprice * (1.0f - threshold_floor),
                price_step, false
            );
            bool touch_controls = short_side.close_gen_touch_down_ticks < target_ticks;
            int nearest_ticks = touch_controls
                ? short_side.close_gen_touch_nearest_ticks : target_ticks;
            short_expand_close_grid = short_expand_close_grid
                || nearest_ticks > low_nonfill_max_tick;
        }
        if (short_expand_close_grid) {
            short_expand_close_grid = recursive_strategy_close_would_expand(
                short_side, false, high_fill_max_tick, low_nonfill_max_tick,
                qty_step, price_step, min_qty, min_cost, c_mult
            );
        }
        if (short_expand_close_grid) {
            TmSide grid_source = short_side;
            int strategy_wel_ticks = short_side.close_gen_touch_down_ticks;
            float strategy_wel_price = float(strategy_wel_ticks) * price_step;
            float strategy_wel_qty = short_side.wel_enforcer_enabled
                ? exposure_reducer_qty(
                    short_side.close_gen_psize,
                    short_side.close_gen_pprice,
                    short_side.close_gen_balance,
                    short_side.allowed_wel * short_side.wel_enforcer_threshold,
                    strategy_wel_price,
                    qty_step, min_qty, min_cost, c_mult
                ) : 0.0f;
            grid_source.close_gen_psize = fmax(
                round_step(
                    short_side.close_gen_psize - strategy_wel_qty, qty_step
                ),
                0.0f
            );
            grid_source.wel_enforcer_enabled = false;
            grid_source.twel_enforcer_enabled = false;
            grid_source.unstuck_enabled = false;
            int prefix_merge_ticks = strategy_wel_qty > 0.0f
                ? strategy_wel_ticks : 0;
            float prefix_merge_qty = strategy_wel_qty > 0.0f
                ? strategy_wel_qty : 0.0f;
            CloseGroup group;
            int grid_rung_limit = strategy_wel_qty > 0.0f ? 499 : 500;
            int group_count = recursive_close_groups(
                grid_source, false, -1, qty_step, price_step,
                min_qty, min_cost, c_mult, short_side.close_gen_psize,
                prefix_merge_ticks, prefix_merge_qty,
                grid_rung_limit, group
            );
            bool strategy_wel_merged = false;
            if (prefix_merge_qty > 0.0f && group_count > 0) {
                recursive_close_groups(
                    grid_source, false, 0, qty_step, price_step,
                    min_qty, min_cost, c_mult, short_side.close_gen_psize,
                    prefix_merge_ticks, prefix_merge_qty,
                    grid_rung_limit, group
                );
                strategy_wel_merged = group.ticks == strategy_wel_ticks;
            }

            ReducerCandidate selected_reducer = empty_reducer_candidate();
            RecursiveCloseAllocation allocation;
#if PASSIVBOT_TM_REDUCERS_DISABLED
            allocation = recursive_close_allocation(
                grid_source, false, group_count,
                short_side.close_gen_psize,
                0.0f, 0.0f, false,
                qty_step, price_step, min_qty, min_cost, c_mult,
                prefix_merge_ticks, prefix_merge_qty,
                grid_rung_limit
            );
            short_side.close_is_exposure_reducer = false;
            short_side.close_is_twel_reducer = false;
            short_side.close_is_unstuck_reducer = false;
            short_side.close_loss_gate_disabled_reducers = false;
#else
            ReducerCandidate candidates[3];
            RecursiveCloseAllocation candidate_allocations[3];
            for (int candidate_idx = 0; candidate_idx < 3; ++candidate_idx) {
                candidates[candidate_idx] = empty_reducer_candidate();
            }

            candidates[0].requested_qty = strategy_wel_merged
                ? 0.0f : strategy_wel_qty;
            candidates[0].ticks = strategy_wel_ticks;
            candidates[0].price = strategy_wel_price;
            candidates[0].order_type_id = 25;
            candidates[0].market = candidates[0].requested_qty > 0.0f
                && should_use_ordinary_market_execution(
                    candidates[0].ticks, true,
                    short_side.close_gen_market_price, price_step,
                    short_side.market_orders_allowed,
                    short_side.market_order_near_touch_threshold
                );
            if (candidates[0].market) {
                candidates[0].requested_qty = resize_market_close_qty(
                    candidates[0].requested_qty,
                    short_side.close_gen_psize,
                    short_side.close_gen_market_price,
                    qty_step, min_qty, min_cost, c_mult
                );
            }

            candidates[1].ticks = max(
                int(ceil(
                    short_side.close_gen_market_price * 1.0005f / price_step
                        - 1.0e-6f
                )),
                1
            );
            candidates[1].price = float(candidates[1].ticks) * price_step;
            candidates[1].requested_qty = short_side.twel_enforcer_enabled
                    && short_side.twel_enforcer_threshold > 0.0f
                ? total_exposure_reducer_qty(
                    short_side.close_gen_psize,
                    short_side.close_gen_pprice,
                    short_side.close_gen_balance,
                    short_side.twel * short_side.twel_enforcer_threshold,
                    candidates[1].price,
                    qty_step, min_qty, min_cost, c_mult
                ) : 0.0f;
            candidates[1].order_type_id = 21;
            candidates[1].is_twel = true;
            candidates[1].market = candidates[1].requested_qty > 0.0f
                && should_use_ordinary_market_execution(
                    candidates[1].ticks, true,
                    short_side.close_gen_market_price, price_step,
                    short_side.market_orders_allowed,
                    short_side.market_order_near_touch_threshold
                );
            if (candidates[1].market) {
                candidates[1].requested_qty = resize_market_close_qty(
                    candidates[1].requested_qty,
                    short_side.close_gen_psize,
                    short_side.close_gen_market_price,
                    qty_step, min_qty, min_cost, c_mult
                );
            }

            TmSide unstuck_source = short_side;
            unstuck_source.psize = short_side.close_gen_psize;
            unstuck_source.pprice = short_side.close_gen_pprice;
            unstuck_source.unstuck_enabled =
                short_side.close_gen_unstuck_candidate_enabled;
            candidates[2].ticks = strategy_wel_ticks;
            candidates[2].price = strategy_wel_price;
            candidates[2].requested_qty = unstuck_reducer_qty(
                unstuck_source, false, short_side.close_gen_balance,
                candidates[2].price,
                qty_step, min_qty, min_cost, c_mult
            );
            candidates[2].order_type_id = 20;
            candidates[2].is_unstuck = true;
            candidates[2].market = candidates[2].requested_qty > 0.0f
                && should_use_ordinary_market_execution(
                    candidates[2].ticks, true,
                    short_side.close_gen_market_price, price_step,
                    short_side.market_orders_allowed,
                    short_side.market_order_near_touch_threshold
                );
            if (candidates[2].market) {
                candidates[2].requested_qty = resize_market_close_qty(
                    candidates[2].requested_qty,
                    short_side.close_gen_psize,
                    short_side.close_gen_market_price,
                    qty_step, min_qty, min_cost, c_mult
                );
            }

            for (int candidate_idx = 0; candidate_idx < 3; ++candidate_idx) {
                candidate_allocations[candidate_idx] =
                    recursive_close_allocation(
                        grid_source, false, group_count,
                        short_side.close_gen_psize,
                        candidates[candidate_idx].requested_qty,
                        candidates[candidate_idx].price,
                        candidates[candidate_idx].market,
                        qty_step, price_step, min_qty, min_cost, c_mult,
                        prefix_merge_ticks, prefix_merge_qty,
                        grid_rung_limit
                    );
                candidates[candidate_idx].finalized_qty =
                    candidate_allocations[candidate_idx].reducer_qty;
            }

            int selected_candidate_idx = select_recursive_close_reducer(
                candidates, false, short_side, price_step,
                market_order_slippage_pct, maker_fee, taker_fee, c_mult,
                loss_gate_enabled, max_realized_loss_pct
            );

            if (selected_candidate_idx >= 0) {
                selected_reducer = candidates[selected_candidate_idx];
                allocation = candidate_allocations[selected_candidate_idx];
            } else {
                allocation = recursive_close_allocation(
                    grid_source, false, group_count,
                    short_side.close_gen_psize,
                    0.0f, 0.0f, false,
                    qty_step, price_step, min_qty, min_cost, c_mult,
                    prefix_merge_ticks, prefix_merge_qty,
                    grid_rung_limit
                );
            }
            short_side.close_is_exposure_reducer =
                selected_candidate_idx >= 0;
            short_side.close_is_twel_reducer = selected_reducer.is_twel;
            short_side.close_is_unstuck_reducer = selected_reducer.is_unstuck;
            short_side.close_loss_gate_disabled_reducers =
                selected_candidate_idx < 0
                && (candidates[0].finalized_qty > 0.0f
                    || candidates[1].finalized_qty > 0.0f
                    || candidates[2].finalized_qty > 0.0f);
#endif

            float reducer_qty = allocation.reducer_qty;
            int reducer_ticks = selected_reducer.ticks;
            float reducer_price = selected_reducer.price;
            bool reducer_market = selected_reducer.market;
            bool reverse = grid_source.close_threshold_we > 0.0f;
            float ordinary_budget = allocation.ordinary_budget;
            float remaining_budget = ordinary_budget;
            float minimum_any = allocation.minimum_any;
            int last_kept_rank = allocation.last_kept_rank;
            int collapse_ordinary_rank = allocation.collapse_ordinary_rank;
            float dust_remainder = allocation.dust_remainder;
            bool normalize_close_groups = allocation.normalize_close_groups;
            bool reducer_reachable = reducer_qty > 0.0f
                && (reducer_market || reducer_ticks > low_nonfill_max_tick);
            bool reducer_executed = false;
            remaining_budget = ordinary_budget;
            for (int rank = 0; rank < group_count; ++rank) {
                int wanted = reverse ? group_count - rank - 1 : rank;
                recursive_close_groups(
                    grid_source, false, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, short_side.close_gen_psize,
                    prefix_merge_ticks, prefix_merge_qty,
                    grid_rung_limit, group
                );
                if (group.qty <= 0.0f) break;
                float trimmed_group_qty = group.qty;
                if (collapse_ordinary_rank >= 0) {
                    trimmed_group_qty = rank == collapse_ordinary_rank
                        ? short_side.close_gen_psize : 0.0f;
                } else if (normalize_close_groups) {
                    float group_min = min_entry_qty(
                        group.market ? short_side.close_gen_market_price
                                     : group.price,
                        qty_step, min_qty, min_cost, c_mult
                    );
                    trimmed_group_qty = fmin(group.qty, remaining_budget);
                    bool partial_trim = quantity_is_meaningfully_below(
                        trimmed_group_qty, group.qty
                    );
                    if (quantity_is_meaningfully_below(
                            trimmed_group_qty, group_min
                        )) {
                        trimmed_group_qty = 0.0f;
                        if (partial_trim) remaining_budget = 0.0f;
                    }
                    if (trimmed_group_qty > 0.0f) {
                        remaining_budget = fmax(
                            round_step(
                                remaining_budget - trimmed_group_qty, qty_step
                            ),
                            0.0f
                        );
                        if (rank == last_kept_rank && dust_remainder > 0.0f
                            && dust_remainder < minimum_any) {
                            trimmed_group_qty = round_step(
                                trimmed_group_qty + dust_remainder, qty_step
                            );
                        }
                    }
                }
                bool reducer_before_group = reducer_reachable
                    && !reducer_executed && reducer_ticks > group.ticks;
                if (reducer_before_group) {
                    float reducer_fill_price = reducer_market
                        ? ordinary_market_fill_price(
                            close, true, market_order_slippage_pct, price_step
                        ) : reducer_price;
                    float reducer_fee_rate = reducer_market
                        ? taker_fee : maker_fee;
                    float pnl = reducer_qty * c_mult
                        * (short_side.pprice - reducer_fill_price);
                    float fee = reducer_qty * reducer_fill_price * c_mult
                        * reducer_fee_rate;
                    // The finalized reducer was admitted against the order
                    // book captured at generation.  The next-candle price is
                    // only the realized fill price; it must not re-gate an
                    // already emitted market order.
                    {
                        if (short_side.close_is_panic) {
                            record_hsl_panic_fill(
                                short_hsl, pnl - fee,
                                directional_equity_at_close(
                                    balance, long_side, short_side, close, c_mult
                                )
                            );
                        }
                        record_directional_gross_pnl(
                            pnl, profit_sum, loss_sum,
                            profit_sum_short, loss_sum_short
                        );
                        balance += pnl - fee;
                        record_realized_net(
                            pnl - fee,
                            realized_pnl_cumsum_last,
                            realized_pnl_cumsum_max,
                            realized_pnl_cumsum_long,
                            realized_pnl_cumsum_short,
                            day_fill_count,
                            fill_count,
                            fill_count_entry,
                            fill_count_long,
                            pnl_recovery_peak,
                            pnl_recovery_peak_k,
                            pnl_recovery_max_min,
                            kf,
                            false,
                            false
                        );
                        record_hsl_rolling_pnl(
                            short_rolling_pnl,
                            rolling_pnl_values, rolling_pnl_indices,
                            short_rolling_base, rolling_capacity, int(kf),
                            pnl_lookback_bars, short_coin_hsl_rolling, pnl - fee
                        );
                        short_side.psize = fmax(
                            round_step(
                                short_side.psize - reducer_qty, qty_step
                            ),
                            0.0f
                        );
                        day_volume += reducer_qty * reducer_fill_price / balance;
                        short_close_fill = true;
                        reducer_executed = true;
                    }
                }
                if (!group.market && group.ticks <= low_nonfill_max_tick) break;
                float group_qty = trimmed_group_qty;
                if (group_qty <= 0.0f) continue;
                float adj = fmin(round_step(group_qty, qty_step), short_side.psize);
                float group_fill_price = group.market
                    ? ordinary_market_fill_price(
                        close, true, market_order_slippage_pct, price_step
                    ) : group.price;
                float group_gate_price = group.market
                    ? ordinary_market_fill_price(
                        short_side.close_gen_market_price, true,
                        market_order_slippage_pct, price_step
                    ) : group.price;
                float group_fee_rate = group.market ? taker_fee : maker_fee;
                float pnl = adj * c_mult
                    * (short_side.pprice - group_fill_price);
                float fee = adj * group_fill_price * c_mult * group_fee_rate;
                if (!realized_loss_proxy_allows_close(
                        adj, group_gate_price, short_side.pprice, false,
                        c_mult, group_fee_rate, loss_gate_enabled)) continue;
                if (short_side.close_is_panic) {
                    record_hsl_panic_fill(
                        short_hsl, pnl - fee,
                        directional_equity_at_close(
                            balance, long_side, short_side, close, c_mult
                        )
                    );
                }
                record_directional_gross_pnl(
                    pnl, profit_sum, loss_sum, profit_sum_short, loss_sum_short
                );
                balance += pnl - fee;
                record_realized_net(
                    pnl - fee,
                    realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    realized_pnl_cumsum_long,
                    realized_pnl_cumsum_short,
                    day_fill_count,
                    fill_count,
                    fill_count_entry,
                    fill_count_long,
                    pnl_recovery_peak,
                    pnl_recovery_peak_k,
                    pnl_recovery_max_min,
                    kf,
                    false,
                    false
                );
                record_hsl_rolling_pnl(
                    short_rolling_pnl,
                    rolling_pnl_values, rolling_pnl_indices,
                    short_rolling_base, rolling_capacity, int(kf),
                    pnl_lookback_bars, short_coin_hsl_rolling, pnl - fee
                );
                float new_psize = fmax(
                    round_step(short_side.psize - adj, qty_step), 0.0f
                );
                bool went_flat = new_psize <= 0.0f;
                short_side.psize = new_psize;
                day_volume += fabs(adj) * group_fill_price / balance;
                short_close_fill = true;
                if (went_flat) {
                    short_side.pprice = 0.0f;
                    if (short_side.pos_open_k >= 0.0f) {
                        float held_min = kf - short_side.pos_open_k;
                        held_max_min = fmax(held_max_min, held_min);
                        held_sum_min += held_min;
                        held_count += 1.0f;
                    }
                    short_side.pos_open_k = -1.0f;
                    break;
                }
            }
            if (reducer_reachable && !reducer_executed
                && short_side.psize > 0.0f) {
                float adj = fmin(reducer_qty, short_side.psize);
                float reducer_fill_price = reducer_market
                    ? ordinary_market_fill_price(
                        close, true, market_order_slippage_pct, price_step
                    ) : reducer_price;
                float reducer_fee_rate = reducer_market
                    ? taker_fee : maker_fee;
                float pnl = adj * c_mult
                    * (short_side.pprice - reducer_fill_price);
                float fee = adj * reducer_fill_price * c_mult
                    * reducer_fee_rate;
                // Selection already applied the generation-time loss gate.
                {
                    if (short_side.close_is_panic) {
                        record_hsl_panic_fill(
                            short_hsl, pnl - fee,
                            directional_equity_at_close(
                                balance, long_side, short_side, close, c_mult
                            )
                        );
                    }
                    record_directional_gross_pnl(
                        pnl, profit_sum, loss_sum, profit_sum_short, loss_sum_short
                    );
                    balance += pnl - fee;
                    record_realized_net(
                        pnl - fee,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        realized_pnl_cumsum_long,
                        realized_pnl_cumsum_short,
                        day_fill_count,
                        fill_count,
                        fill_count_entry,
                        fill_count_long,
                        pnl_recovery_peak,
                        pnl_recovery_peak_k,
                        pnl_recovery_max_min,
                        kf,
                        false,
                        false
                    );
                    record_hsl_rolling_pnl(
                        short_rolling_pnl,
                        rolling_pnl_values, rolling_pnl_indices,
                        short_rolling_base, rolling_capacity, int(kf),
                        pnl_lookback_bars, short_coin_hsl_rolling, pnl - fee
                    );
                    short_side.psize = fmax(
                        round_step(short_side.psize - adj, qty_step), 0.0f
                    );
                    day_volume += adj * reducer_fill_price / balance;
                    short_close_fill = true;
                }
            }
            if (short_close_fill && short_side.psize <= 0.0f
                && short_side.pprice > 0.0f) {
                short_side.pprice = 0.0f;
                if (short_side.pos_open_k >= 0.0f) {
                    float held_min = kf - short_side.pos_open_k;
                    held_max_min = fmax(held_max_min, held_min);
                    held_sum_min += held_min;
                    held_count += 1.0f;
                }
                short_side.pos_open_k = -1.0f;
            }
            if (short_close_fill) short_side.close_qty = 0.0f;
        } else if (short_close_fill_ready || short_secondary_close_fill) {
            bool secondary_first = short_secondary_close_fill
                && (!short_close_fill_ready
                    || short_side.secondary_close_price
                        >= short_side.close_price);
            for (int rank = 0; rank < 2; ++rank) {
                bool use_secondary = secondary_first ? rank == 0 : rank == 1;
                bool reachable = use_secondary
                    ? short_secondary_close_fill : short_close_fill_ready;
                if (!reachable || short_side.psize <= 0.0f) continue;
                bool market_panic = !use_secondary && short_side.close_is_panic
                    && short_hsl_panic_market;
                bool ordinary_market = use_secondary
                    ? short_side.secondary_close_market
                    : short_side.close_market;
                bool market_execution = market_panic || ordinary_market;
                float cp = market_execution
                        ? float(max(directional_ticks(
                            close * (1.0f + market_order_slippage_pct),
                            price_step, true
                        ), 1)) * price_step
                        : use_secondary ? short_side.secondary_close_price
                                        : short_side.close_price;
                float requested_qty = use_secondary
                    ? short_side.secondary_close_qty : short_side.close_qty;
                float adj = fmin(
                    round_step(requested_qty, qty_step), short_side.psize
                );
                float pnl = adj * c_mult * (short_side.pprice - cp);
                float fee = adj * cp * c_mult
                    * (market_execution ? taker_fee : maker_fee);
                bool selected_unstuck = !use_secondary
                    && short_side.close_is_unstuck_reducer;
                if (!short_side.close_is_panic
                    && !realized_loss_proxy_allows_reducer(
                        adj, cp, short_side.pprice, false, selected_unstuck,
                        c_mult, market_execution ? taker_fee : maker_fee,
                        loss_gate_enabled, balance,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        max_realized_loss_pct)) continue;
                if (!use_secondary && short_side.close_is_panic) {
                    record_hsl_panic_fill(
                        short_hsl, pnl - fee,
                        directional_equity_at_close(
                            balance, long_side, short_side, close, c_mult
                        )
                    );
                }
                record_directional_gross_pnl(
                    pnl, profit_sum, loss_sum, profit_sum_short, loss_sum_short
                );
                balance += pnl - fee;
                record_realized_net(
                    pnl - fee,
                    realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    realized_pnl_cumsum_long,
                    realized_pnl_cumsum_short,
                    day_fill_count,
                    fill_count,
                    fill_count_entry,
                    fill_count_long,
                    pnl_recovery_peak,
                    pnl_recovery_peak_k,
                    pnl_recovery_max_min,
                    kf,
                    false,
                    false
                );
                record_hsl_rolling_pnl(
                    short_rolling_pnl,
                    rolling_pnl_values, rolling_pnl_indices,
                    short_rolling_base, rolling_capacity, int(kf),
                    pnl_lookback_bars, short_coin_hsl_rolling, pnl - fee
                );
                short_side.psize = fmax(
                    round_step(short_side.psize - adj, qty_step), 0.0f
                );
                day_volume += fabs(adj) * cp / balance;
                short_close_fill = true;
            }
            bool went_flat = short_side.psize <= 0.0f;
            if (went_flat) {
                short_side.pprice = 0.0f;
                if (short_side.pos_open_k >= 0.0f) {
                    float held_min = kf - short_side.pos_open_k;
                    held_max_min = fmax(held_max_min, held_min);
                    held_sum_min += held_min;
                    held_count += 1.0f;
                }
                short_side.pos_open_k = -1.0f;
            }
            short_side.close_qty = 0.0f;
            short_side.secondary_close_qty = 0.0f;
        }

        bool short_entry_passive_reachable = short_side.entry_qty > 0.0f
            && short_side.entry_ticks <= high_fill_max_tick;
        short_entry_fill = valid && alive && short_enabled
            && short_side.entry_qty > 0.0f
            && (short_side.entry_market
                || short_entry_passive_reachable);
        if (short_entry_fill) {
            short_entry_fill = false;
            TmSide ladder_side = short_side;
            ladder_side.psize = short_side.entry_gen_psize;
            ladder_side.pprice = short_side.entry_gen_pprice;
            ladder_side.market_orders_allowed = false;
            const float ladder_balance = short_side.entry_gen_balance;
            const float ladder_kf = short_side.entry_gen_kf;
            int ladder_touch_ticks = short_side.entry_gen_touch_ticks;
            float ladder_market_price = short_side.entry_gen_market_price;
            TmSide gate_side = ladder_side;
            ladder_side.twel_entry_gate_enabled = false;
            int previous_ticks = 0;
            for (int rung = 0; rung < 500; ++rung) {
                int entry_ticks = rung == 0
                    ? short_side.entry_ticks : ladder_side.entry_ticks;
                float ep = rung == 0
                    ? short_side.entry_price : ladder_side.entry_price;
                float strategy_eq = round_step(
                    rung == 0
                        ? short_side.entry_strategy_qty
                        : ladder_side.entry_qty,
                    qty_step
                );
                if (strategy_eq <= 0.0f) break;
                float eq = rung == 0 ? short_side.entry_qty : strategy_eq;
                float ungated_eq = strategy_eq;
                bool entry_market = should_use_ordinary_market_execution(
                    entry_ticks, false, ladder_market_price, price_step,
                    short_side.market_orders_allowed,
                    short_side.market_order_near_touch_threshold
                );
                if (entry_market) {
                    float market_min_q = min_entry_qty(
                        ladder_market_price, qty_step, min_qty, min_cost, c_mult
                    );
                    if (eq < market_min_q) eq = market_min_q;
                    if (ungated_eq < market_min_q) ungated_eq = market_min_q;
                }
                if (rung > 0 && gate_side.twel_entry_gate_enabled) {
                    float entry_gate_price = entry_market
                        ? ladder_market_price : ep;
                    eq = gate_entry_by_twel_strict(
                        gate_side, ladder_balance, entry_gate_price, eq,
                        qty_step, min_qty, min_cost, c_mult
                    );
                    if (!(eq > 0.0f)) break;
                }
                bool twel_boundary_partial = gate_side.twel_entry_gate_enabled
                    && eq > 0.0f && eq < ungated_eq;
                if ((!entry_market && entry_ticks > high_fill_max_tick)
                    || (rung > 0 && entry_ticks == previous_ticks)) break;
                if (eq > 0.0f) {
                    float entry_fill_price = entry_market
                        ? ordinary_market_fill_price(
                            close, false, market_order_slippage_pct, price_step
                        ) : ep;
                    float fee = eq * entry_fill_price * c_mult
                        * (entry_market ? taker_fee : maker_fee);
                    balance -= fee;
                    record_realized_net(
                        -fee,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        realized_pnl_cumsum_long,
                        realized_pnl_cumsum_short,
                        day_fill_count,
                        fill_count,
                        fill_count_entry,
                        fill_count_long,
                        pnl_recovery_peak,
                        pnl_recovery_peak_k,
                        pnl_recovery_max_min,
                        kf,
                        true,
                        false
                    );
                    record_hsl_rolling_pnl(
                        short_rolling_pnl, rolling_pnl_values,
                        rolling_pnl_indices, short_rolling_base,
                        rolling_capacity, int(kf), pnl_lookback_bars,
                        short_coin_hsl_rolling, -fee
                    );
                    bool was_flat = short_side.psize <= 0.0f;
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
                    if (rung == 0 && short_side.entry_gen_psize <= 0.0f) {
                        record_initial_entry_interval(
                            entry_interval_stats, entry_interval_counts, b,
                            short_last_initial_entry_k, kf
                        );
                    }
#endif
                    float new_psize = round_step(
                        short_side.psize + eq, qty_step
                    );
                    float new_pprice = was_flat ? entry_fill_price
                        : short_side.pprice
                            * (short_side.psize / fmax(new_psize, 1.0e-12f))
                            + entry_fill_price
                                * (eq / fmax(new_psize, 1.0e-12f));
                    if (was_flat) short_side.pos_open_k = kf;
                    short_side.psize = new_psize;
                    short_side.pprice = new_pprice;
                    short_side.last_inc_k = kf;
                    day_volume += fabs(eq) * entry_fill_price / balance;
                    short_entry_fill = true;

                    if (gate_side.twel_entry_gate_enabled) {
                        float entry_gate_price = entry_market
                            ? ladder_market_price : ep;
                        bool gate_flat = gate_side.psize <= 0.0f;
                        float gate_psize = round_step(
                            gate_side.psize + eq, qty_step
                        );
                        gate_side.pprice = gate_flat ? entry_gate_price
                            : gate_side.pprice * (
                                gate_side.psize / fmax(gate_psize, 1.0e-12f)
                            ) + entry_gate_price * (
                                eq / fmax(gate_psize, 1.0e-12f)
                            );
                        gate_side.psize = gate_psize;
                    }
                }
                if (twel_boundary_partial) break;

                bool sim_flat = ladder_side.psize <= 0.0f;
                float sim_psize = round_step(
                    ladder_side.psize + strategy_eq, qty_step
                );
                ladder_side.pprice = sim_flat ? ep
                    : ladder_side.pprice
                        * (ladder_side.psize / fmax(sim_psize, 1.0e-12f))
                        + ep * (strategy_eq / fmax(sim_psize, 1.0e-12f));
                ladder_side.psize = sim_psize;
                previous_ticks = entry_ticks;
                ladder_touch_ticks = max(ladder_touch_ticks, entry_ticks);
                if (rung == 0 && !short_entry_passive_reachable) break;
#if PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY
                if (short_side.cooldown_min != 0.0f) break;
#else
                if (PASSIVBOT_TM_TRAILING_ENTRY_ONLY
                    || short_side.entry_retracement_base > 0.0f
                    || short_side.cooldown_min != 0.0f) break;
#endif
                generate_short_orders(
                    ladder_side, ladder_balance, ep, qty_step,
                    ladder_touch_ticks, ladder_touch_ticks, ladder_touch_ticks,
                    touch_min_qty, touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, ladder_kf, false
                );
            }
            short_side.entry_qty = 0.0f;
        }
#endif

        // Exact Rust generates this candle's bundles first, then force-closes and
        // clears both bundles.  Closing here and suppressing only that dead order
        // generation is equivalent while retaining `gen` for equity/HSL sampling.
        const bool forced_delist = valid && k == last_valid
            && last_valid + 1400 < T;
        bool forced_delist_closed_any = false;
        if (forced_delist && alive && balance > 0.0f) {
#if !defined(PASSIVBOT_TRAILING_SHORT_ONLY)
#if defined(PASSIVBOT_TRAILING_LONG_ONLY)
            float short_unrealized = 0.0f;
#else
            float short_unrealized = short_side.psize > 0.0f
                ? short_side.psize * c_mult * (short_side.pprice - close)
                : 0.0f;
#endif
            bool forced_long_close = force_close_delisted_position(
                long_side.psize, long_side.pprice, long_side.pos_open_k,
                long_hsl, true, close, short_unrealized, price_step,
                c_mult, taker_fee, market_order_slippage_pct, kf, balance,
                realized_pnl_cumsum_last, realized_pnl_cumsum_max,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                day_fill_count, fill_count, fill_count_entry, fill_count_long,
                pnl_recovery_peak, pnl_recovery_peak_k, pnl_recovery_max_min,
                long_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                long_rolling_base, rolling_capacity, pnl_lookback_bars,
                long_coin_hsl_rolling, profit_sum, loss_sum, profit_sum_long,
                loss_sum_long, held_max_min, held_sum_min, held_count, day_volume
            );
#else
            bool forced_long_close = false;
#endif
#if !defined(PASSIVBOT_TRAILING_LONG_ONLY)
#if defined(PASSIVBOT_TRAILING_SHORT_ONLY)
            float long_unrealized = 0.0f;
#else
            float long_unrealized = long_side.psize > 0.0f
                ? long_side.psize * c_mult * (close - long_side.pprice)
                : 0.0f;
#endif
            bool forced_short_close = force_close_delisted_position(
                short_side.psize, short_side.pprice, short_side.pos_open_k,
                short_hsl, false, close, long_unrealized, price_step,
                c_mult, taker_fee, market_order_slippage_pct, kf, balance,
                realized_pnl_cumsum_last, realized_pnl_cumsum_max,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                day_fill_count, fill_count, fill_count_entry, fill_count_long,
                pnl_recovery_peak, pnl_recovery_peak_k, pnl_recovery_max_min,
                short_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                short_rolling_base, rolling_capacity, pnl_lookback_bars,
                short_coin_hsl_rolling, profit_sum, loss_sum, profit_sum_short,
                loss_sum_short, held_max_min, held_sum_min, held_count, day_volume
            );
#else
            bool forced_short_close = false;
#endif
            long_close_fill = long_close_fill || forced_long_close;
            short_close_fill = short_close_fill || forced_short_close;
            forced_delist_closed_any = forced_long_close || forced_short_close;
            if (forced_delist_closed_any) {
#if !defined(PASSIVBOT_TRAILING_SHORT_ONLY)
                long_side.entry_qty = 0.0f;
                long_side.close_qty = 0.0f;
                long_side.secondary_close_qty = 0.0f;
#endif
#if !defined(PASSIVBOT_TRAILING_LONG_ONLY)
                short_side.entry_qty = 0.0f;
                short_side.close_qty = 0.0f;
                short_side.secondary_close_qty = 0.0f;
#endif
            }
        }

#if !defined(PASSIVBOT_TRAILING_SHORT_ONLY)
        if (long_close_fill || long_entry_fill) {
            if (long_position_last_fill_k >= 0.0f) {
                position_unchanged_max_min = fmax(
                    position_unchanged_max_min, kf - long_position_last_fill_k
                );
            }
            long_position_last_fill_k = long_side.psize > 0.0f ? kf : -1.0f;
        }
#endif
#if !defined(PASSIVBOT_TRAILING_LONG_ONLY)
        if (short_close_fill || short_entry_fill) {
            if (short_position_last_fill_k >= 0.0f) {
                position_unchanged_max_min = fmax(
                    position_unchanged_max_min, kf - short_position_last_fill_k
                );
            }
            short_position_last_fill_k = short_side.psize > 0.0f ? kf : -1.0f;
        }
#endif

        bool any_fill = long_close_fill || long_entry_fill
            || short_close_fill || short_entry_fill;
        if (any_fill) {
            day_has_fill = 1.0f;
            day_touched = true;
            if (last_fill_k >= 0.0f) {
                float gap = kf - last_fill_k;
                int bin = clamp(
                    int(log(fmax(gap, 0.0f) + 1.0f) * log_bin_scale), 0, 127
                );
                gap_hist[int(b) * GAP_BINS + bin] += 1;
                gap_max_min = fmax(gap_max_min, gap);
            }
            if (first_fill_k < 0.0f) first_fill_k = kf;
            last_fill_k = kf;
        }

#if !defined(PASSIVBOT_TRAILING_SHORT_ONLY)
        if (long_enabled) {
#if PASSIVBOT_TM_VOLATILITY_DISABLED
            update_indicators(long_side, close, 0.0f, 0.0f, valid, false);
#else
            update_indicators(long_side, close, log_range, hour_lr, valid, hour_valid);
#endif
        }
#endif
#if !defined(PASSIVBOT_TRAILING_LONG_ONLY)
        if (short_enabled) {
#if PASSIVBOT_TM_VOLATILITY_DISABLED
            update_indicators(short_side, close, 0.0f, 0.0f, valid, false);
#else
            update_indicators(short_side, close, log_range, hour_lr, valid, hour_valid);
#endif
        }
#endif
#if !defined(PASSIVBOT_TRAILING_SHORT_ONLY)
        if (long_enabled) {
            update_trailing(
                long_side, high, low, close, valid,
                long_close_fill || long_entry_fill
            );
        }
#endif
#if !defined(PASSIVBOT_TRAILING_LONG_ONLY)
        if (short_enabled) {
            update_trailing(
                short_side, high, low, close, valid,
                short_close_fill || short_entry_fill
            );
        }
#endif

        bool gen = can_gen && alive;
        eq_started = eq_started || gen;
        int long_hsl_mode = hsl_mode(long_hsl, long_side.psize > 0.0f);
        int short_hsl_mode = hsl_mode(short_hsl, short_side.psize > 0.0f);
        if (gen && !forced_delist_closed_any) {
            long_side.close_is_panic = false;
            short_side.close_is_panic = false;
            if (long_side.psize > 0.0f || short_side.psize > 0.0f) {
                min_cost_exact_open_uncertain = true;
            }
            float guaranteed_balance_lower = min_cost_exact_open_uncertain
                ? 0.0f : liq_floor;
            bool long_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                long_side.allowed_wel, long_side.initial_qty_pct,
                max_effective_min_cost
            );
            bool short_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                short_side.allowed_wel, short_side.initial_qty_pct,
                max_effective_min_cost
            );
            if (filter_by_min_effective_cost
                && !min_cost_exact_open_uncertain
                && (
                    (long_enabled && long_side.psize <= 0.0f
                        && !long_min_cost_eligible)
                    || (short_enabled && short_side.psize <= 0.0f
                        && !short_min_cost_eligible)
                )) {
                min_cost_exact_open_uncertain = true;
                guaranteed_balance_lower = 0.0f;
                long_min_cost_eligible = passes_min_effective_cost(
                    true, guaranteed_balance_lower, long_side.allowed_wel,
                    long_side.initial_qty_pct, max_effective_min_cost
                );
                short_min_cost_eligible = passes_min_effective_cost(
                    true, guaranteed_balance_lower, short_side.allowed_wel,
                    short_side.initial_qty_pct, max_effective_min_cost
                );
            }
            bool block_long_initial = !long_min_cost_eligible || long_hsl_mode != 0;
            bool block_short_initial = !short_min_cost_eligible || short_hsl_mode != 0;
            if (long_enabled && short_enabled && !hedge_mode) {
                if (long_side.psize > 0.0f) {
                    block_short_initial = true;
                } else if (short_side.psize > 0.0f) {
                    block_long_initial = true;
                } else if (long_min_cost_eligible && !short_min_cost_eligible) {
                    block_short_initial = true;
                } else if (!long_min_cost_eligible && short_min_cost_eligible) {
                    block_long_initial = true;
                } else if (long_min_cost_eligible && short_min_cost_eligible) {
                    float long_lower = fmin(
                        long_side.ema0, fmin(long_side.ema1, long_side.ema2)
                    );
                    float short_upper = fmax(
                        short_side.ema0, fmax(short_side.ema1, short_side.ema2)
                    );
                    float dist_long = long_lower
                        * (1.0f - long_side.initial_ema_dist) / close - 1.0f;
                    float dist_short = 1.0f - short_upper
                        * (1.0f + short_side.initial_ema_dist) / close;
                    if (dist_long >= dist_short) {
                        block_short_initial = true;
                    } else {
                        block_long_initial = true;
                    }
                }
            }
            float balance_peak = balance
                + (realized_pnl_cumsum_max - realized_pnl_cumsum_last);
            long_side.unstuck_allowance = calc_unstuck_allowance(
                long_side, balance, balance_peak
            );
            short_side.unstuck_allowance = calc_unstuck_allowance(
                short_side, balance, balance_peak
            );
            bool configured_long_unstuck = long_side.unstuck_enabled;
            bool configured_short_unstuck = short_side.unstuck_enabled;
            bool long_unstuck_selected = long_enabled && unstuck_eligible(
                long_side, true, balance, close, touch_down_tick,
                touch_up_tick, price_step, c_mult
            );
            bool short_unstuck_selected = short_enabled && unstuck_eligible(
                short_side, false, balance, close, touch_down_tick,
                touch_up_tick, price_step, c_mult
            );
            if (long_unstuck_selected && short_unstuck_selected) {
                float long_diff = 1.0f - close / long_side.pprice;
                float short_diff = close / short_side.pprice - 1.0f;
                if (long_diff <= short_diff) {
                    short_unstuck_selected = false;
                } else {
                    long_unstuck_selected = false;
                }
            }
            long_side.unstuck_enabled = long_unstuck_selected;
            short_side.unstuck_enabled = short_unstuck_selected;
            if (long_enabled) {
                generate_long_orders(
                    long_side, balance, close, qty_step, touch_down_tick,
                    touch_up_tick, touch_nearest_tick, touch_min_qty,
                    touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, kf, block_long_initial
                );
            }
            if (short_enabled) {
                generate_short_orders(
                    short_side, balance, close, qty_step, touch_down_tick,
                    touch_up_tick, touch_nearest_tick, touch_min_qty,
                    touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, kf, block_short_initial
                );
            }
            if (filter_by_min_effective_cost && long_enabled && short_enabled) {
                min_cost_exact_open_uncertain = true;
            }
            // Exact Rust tries the next-largest protective reducer when the
            // winner is loss-gated. Auto-unstuck may consume the conservative
            // all-history budget; other TM reducers retain the zero-loss envelope.
            bool long_wel_enabled = long_side.wel_enforcer_enabled;
            bool long_twel_enabled = long_side.twel_enforcer_enabled;
            for (int retry = 0; retry < 3 && long_enabled
                && loss_gate_enabled
                && long_side.close_is_exposure_reducer
                && !realized_loss_proxy_allows_reducer(
                    long_side.close_qty,
                    long_side.close_market
                        ? ordinary_market_fill_price(
                            close, false, market_order_slippage_pct, price_step
                        )
                        : long_side.close_price,
                    long_side.pprice, true,
                    long_side.close_is_unstuck_reducer,
                    c_mult,
                    long_side.close_market ? taker_fee : maker_fee,
                    true, balance,
                    realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    max_realized_loss_pct
                ); ++retry) {
                if (long_side.close_is_unstuck_reducer) {
                    long_side.unstuck_enabled = false;
                } else if (long_side.close_is_twel_reducer) {
                    long_side.twel_enforcer_enabled = false;
                } else {
                    long_side.wel_enforcer_enabled = false;
                }
                generate_long_orders(
                    long_side, balance, close, qty_step, touch_down_tick,
                    touch_up_tick, touch_nearest_tick, touch_min_qty,
                    touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, kf, block_long_initial
                );
                if (!long_side.close_is_exposure_reducer) {
                    long_side.close_loss_gate_disabled_reducers = true;
                }
            }
            bool short_wel_enabled = short_side.wel_enforcer_enabled;
            bool short_twel_enabled = short_side.twel_enforcer_enabled;
            for (int retry = 0; retry < 3 && short_enabled
                && loss_gate_enabled
                && short_side.close_is_exposure_reducer
                && !realized_loss_proxy_allows_reducer(
                    short_side.close_qty,
                    short_side.close_market
                        ? ordinary_market_fill_price(
                            close, true, market_order_slippage_pct, price_step
                        )
                        : short_side.close_price,
                    short_side.pprice, false,
                    short_side.close_is_unstuck_reducer,
                    c_mult,
                    short_side.close_market ? taker_fee : maker_fee,
                    true, balance,
                    realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    max_realized_loss_pct
                ); ++retry) {
                if (short_side.close_is_unstuck_reducer) {
                    short_side.unstuck_enabled = false;
                } else if (short_side.close_is_twel_reducer) {
                    short_side.twel_enforcer_enabled = false;
                } else {
                    short_side.wel_enforcer_enabled = false;
                }
                generate_short_orders(
                    short_side, balance, close, qty_step, touch_down_tick,
                    touch_up_tick, touch_nearest_tick, touch_min_qty,
                    touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, kf, block_short_initial
                );
                if (!short_side.close_is_exposure_reducer) {
                    short_side.close_loss_gate_disabled_reducers = true;
                }
            }
            // Recursive next-candle expansion must reconstruct every reducer
            // candidate which existed at this generation, even when singular
            // selection or its loss-gate fallback temporarily disabled it.
            long_side.close_gen_realized_pnl_cumsum_last =
                realized_pnl_cumsum_last;
            long_side.close_gen_realized_pnl_cumsum_max =
                realized_pnl_cumsum_max;
            short_side.close_gen_realized_pnl_cumsum_last =
                realized_pnl_cumsum_last;
            short_side.close_gen_realized_pnl_cumsum_max =
                realized_pnl_cumsum_max;
            long_side.close_gen_unstuck_candidate_enabled =
                long_unstuck_selected;
            short_side.close_gen_unstuck_candidate_enabled =
                short_unstuck_selected;
            long_side.wel_enforcer_enabled = long_wel_enabled;
            long_side.twel_enforcer_enabled = long_twel_enabled;
            long_side.unstuck_enabled = configured_long_unstuck;
            short_side.wel_enforcer_enabled = short_wel_enabled;
            short_side.twel_enforcer_enabled = short_twel_enabled;
            short_side.unstuck_enabled = configured_short_unstuck;
            if (long_enabled && long_hsl_mode >= 2) {
                long_side.entry_qty = 0.0f;
            }
            if (short_enabled && short_hsl_mode >= 2) {
                short_side.entry_qty = 0.0f;
            }
            if (long_enabled && long_hsl_mode == 3) {
                install_hsl_panic_close(
                    long_side, true, touch_down_tick, touch_up_tick, price_step
                );
            }
            if (short_enabled && short_hsl_mode == 3) {
                install_hsl_panic_close(
                    short_side, false, touch_down_tick, touch_up_tick, price_step
                );
            }
        }

        // Exact Rust keeps sampling account equity through a short invalid
        // tail, but excludes positions whose coin is no longer valid.  Such a
        // tail is non-tradable and therefore contributes balance-only equity.
        const bool after_valid_tail = k > last_valid;
        float long_unreal = valid && long_side.psize > 0.0f
            ? long_side.psize * c_mult * (close - long_side.pprice) : 0.0f;
        float short_unreal = valid && short_side.psize > 0.0f
            ? short_side.psize * c_mult * (short_side.pprice - close) : 0.0f;
        float equity = balance + long_unreal + short_unreal;
        const bool rolling_pnl_overflowed =
            (long_coin_hsl_rolling && long_rolling_pnl.overflowed)
            || (short_coin_hsl_rolling && short_rolling_pnl.overflowed);
        if (rolling_pnl_overflowed) {
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
            // A bounded rolling-PnL overflow invalidates the proxy candidate.
            // The postprocessor maps this impossible equity to the maximum
            // bounded duration for every minimized recovery statistic.
            recovery_samples[int(b) * recovery_sample_capacity]
                = RECOVERY_FAIL_CLOSED_SENTINEL;
#endif
            balance = 0.0f;
            alive = false;
            liq_day = di;
        }
        const bool hsl_step = gen || (eq_started && after_valid_tail);
        if (hsl_step && alive && balance > 0.0f && equity > liq_floor) {
#if defined(PASSIVBOT_TRAILING_LONG_ONLY)
            bool long_blocking_orders = valid && long_hsl_mode != 3 && (
                long_side.entry_qty > 0.0f || long_side.close_qty > 0.0f
                    || long_side.secondary_close_qty > 0.0f
            );
            prepare_coin_hsl_rolling_signal(
                long_hsl, long_rolling_pnl,
                rolling_pnl_values, rolling_pnl_indices,
                long_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, realized_pnl_cumsum_long
            );
            float long_triggers_before = long_hsl.triggers;
            const bool long_hsl_sample_enabled = long_hsl.enabled
                && (long_hsl.signal_mode == HSL_SIGNAL_COIN || !long_hsl.halted);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
            if (long_hsl_sample_enabled) {
                update_hsl_strategy_equity_stats(
                    long_hsl_strategy_eq,
                    starting_balance + realized_pnl_cumsum_long + long_unreal,
                    di
                );
            }
#endif
            update_one_side_hsl(
                long_hsl, balance, starting_balance,
                realized_pnl_cumsum_long, long_unreal,
                long_side.psize > 0.0f, long_blocking_orders,
                kf, interval_ms
            );
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
            if (long_hsl_sample_enabled) {
                update_hsl_drawdown_ema_tail_stats(
                    long_hsl_ema_tail, long_hsl.drawdown_ema
                );
            }
#endif
            if (long_hsl.triggers > long_triggers_before) {
                reset_hsl_rolling_pnl_window(long_rolling_pnl);
            }
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
            if (long_hsl.enabled) {
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow += long_hsl.tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange += long_hsl.tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red += long_hsl.tier == 3 ? 1.0f : 0.0f;
            }
#endif
            try_restart_hsl(long_hsl, kf, equity);
#elif defined(PASSIVBOT_TRAILING_SHORT_ONLY)
            bool short_blocking_orders = valid && short_hsl_mode != 3 && (
                short_side.entry_qty > 0.0f || short_side.close_qty > 0.0f
                    || short_side.secondary_close_qty > 0.0f
            );
            prepare_coin_hsl_rolling_signal(
                short_hsl, short_rolling_pnl,
                rolling_pnl_values, rolling_pnl_indices,
                short_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, realized_pnl_cumsum_short
            );
            float short_triggers_before = short_hsl.triggers;
            const bool short_hsl_sample_enabled = short_hsl.enabled
                && (short_hsl.signal_mode == HSL_SIGNAL_COIN || !short_hsl.halted);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
            if (short_hsl_sample_enabled) {
                update_hsl_strategy_equity_stats(
                    short_hsl_strategy_eq,
                    starting_balance + realized_pnl_cumsum_short + short_unreal,
                    di
                );
            }
#endif
            update_one_side_hsl(
                short_hsl, balance, starting_balance,
                realized_pnl_cumsum_short, short_unreal,
                short_side.psize > 0.0f, short_blocking_orders,
                kf, interval_ms
            );
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
            if (short_hsl_sample_enabled) {
                update_hsl_drawdown_ema_tail_stats(
                    short_hsl_ema_tail, short_hsl.drawdown_ema
                );
            }
#endif
            if (short_hsl.triggers > short_triggers_before) {
                reset_hsl_rolling_pnl_window(short_rolling_pnl);
            }
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
            if (short_hsl.enabled) {
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow += short_hsl.tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange += short_hsl.tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red += short_hsl.tier == 3 ? 1.0f : 0.0f;
            }
#endif
            try_restart_hsl(short_hsl, kf, equity);
#else
            bool long_blocking_orders = valid && long_hsl_mode != 3 && (
                long_side.entry_qty > 0.0f || long_side.close_qty > 0.0f
                    || long_side.secondary_close_qty > 0.0f
            );
            bool short_blocking_orders = valid && short_hsl_mode != 3 && (
                short_side.entry_qty > 0.0f || short_side.close_qty > 0.0f
                    || short_side.secondary_close_qty > 0.0f
            );
            prepare_coin_hsl_rolling_signal(
                long_hsl, long_rolling_pnl,
                rolling_pnl_values, rolling_pnl_indices,
                long_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, realized_pnl_cumsum_long
            );
            prepare_coin_hsl_rolling_signal(
                short_hsl, short_rolling_pnl,
                rolling_pnl_values, rolling_pnl_indices,
                short_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, realized_pnl_cumsum_short
            );
            float long_triggers_before = long_hsl.triggers;
            float short_triggers_before = short_hsl.triggers;
            const bool unified_hsl = long_hsl.signal_mode == HSL_SIGNAL_UNIFIED;
            const bool long_hsl_sample_enabled = hsl_modes_valid && long_enabled
                && long_hsl.enabled
                && (long_hsl.signal_mode == HSL_SIGNAL_COIN || !long_hsl.halted);
            const bool short_hsl_sample_enabled = hsl_modes_valid && short_enabled
                && short_hsl.enabled
                && (short_hsl.signal_mode == HSL_SIGNAL_COIN || !short_hsl.halted);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
            if (long_hsl_sample_enabled) {
                update_hsl_strategy_equity_stats(
                    long_hsl_strategy_eq,
                    starting_balance + (
                        unified_hsl ? realized_pnl_cumsum_last
                            : realized_pnl_cumsum_long
                    ) + (unified_hsl ? long_unreal + short_unreal : long_unreal),
                    di
                );
            }
            if (short_hsl_sample_enabled) {
                update_hsl_strategy_equity_stats(
                    short_hsl_strategy_eq,
                    starting_balance + (
                        unified_hsl ? realized_pnl_cumsum_last
                            : realized_pnl_cumsum_short
                    ) + (unified_hsl ? long_unreal + short_unreal : short_unreal),
                    di
                );
            }
#endif
            bool hsl_update_valid = update_dual_side_hsl(
                long_hsl, short_hsl, balance, starting_balance,
                realized_pnl_cumsum_last,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                long_unreal, short_unreal,
                long_side.psize > 0.0f, short_side.psize > 0.0f,
                long_blocking_orders, short_blocking_orders,
                kf, interval_ms
            );
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
            if (hsl_update_valid && long_hsl_sample_enabled) {
                update_hsl_drawdown_ema_tail_stats(
                    long_hsl_ema_tail, long_hsl.drawdown_ema
                );
            }
            if (hsl_update_valid && short_hsl_sample_enabled) {
                update_hsl_drawdown_ema_tail_stats(
                    short_hsl_ema_tail, short_hsl.drawdown_ema
                );
            }
#endif
            if (long_hsl.triggers > long_triggers_before) {
                reset_hsl_rolling_pnl_window(long_rolling_pnl);
            }
            if (short_hsl.triggers > short_triggers_before) {
                reset_hsl_rolling_pnl_window(short_rolling_pnl);
            }
            if (!hsl_update_valid) {
                balance = 0.0f;
                alive = false;
                liq_day = di;
            }
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
            if (hsl_update_valid && (long_hsl.enabled || short_hsl.enabled)) {
                int hsl_tier = max(long_hsl.tier, short_hsl.tier);
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow += hsl_tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange += hsl_tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red += hsl_tier == 3 ? 1.0f : 0.0f;
            }
#endif
            if (hsl_update_valid) {
                try_restart_hsl(long_hsl, kf, equity);
                try_restart_hsl(short_hsl, kf, equity);
            }
#endif
        }
        // Exact Rust records an equity sample at every tracked timestamp.
        // Invalid candles are non-tradable and contribute balance-only equity,
        // just like the already-supported tail after last_valid.
        bool active = eq_started && alive;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = kf;
            last_eq_k = kf;
            if (any_fill) {
                int active_fill_day = elapsed_fill_day_bucket(
                    kf, first_eq_k, interval_ms
                );
                if (active_fill_day != last_active_fill_day) {
                    fills_active_days_count += 1.0f;
                    last_active_fill_day = active_fill_day;
                }
            }
            bool liq = balance <= 0.0f || equity <= liq_floor;
            float eqf = liq ? liq_floor : equity;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
            if (recovery_stride > 0 && recovery_start_k < 0) {
                recovery_start_k = k;
                recovery_samples[int(b) * recovery_sample_capacity] = eqf;
            } else if (recovery_stride > 0) {
                const int recovery_elapsed = k - recovery_start_k;
                const bool recovery_terminal = liq || k == T - 2;
                const bool recovery_regular = recovery_elapsed % recovery_stride == 0;
                if (recovery_regular || recovery_terminal) {
                    const int sample_index = recovery_terminal
                        ? (recovery_elapsed + recovery_stride - 1) / recovery_stride
                        : recovery_elapsed / recovery_stride;
                    if (sample_index < recovery_sample_count) {
                        recovery_samples[
                            int(b) * recovery_sample_capacity + sample_index
                        ] = eqf;
                    }
                }
            }
#endif
            if (eqf >= account_peak) {
                if (account_peak_k >= 0.0f) {
                    account_recovery_max_min = fmax(
                        account_recovery_max_min, kf - account_peak_k
                    );
                }
                account_peak = eqf;
                account_peak_k = kf;
            }
            if (eqf > run_peak) {
                if (last_high_k >= 0.0f) {
                    recovery_max_min = fmax(recovery_max_min, kf - last_high_k);
                }
                last_high_k = kf;
                run_peak = eqf;
            }
            float dd = fmax(
                (run_peak - eqf) / fmax(fabs(run_peak), 1.0e-12f), 0.0f
            );
            max_dd = fmax(max_dd, dd);
            day_end = eqf;
            day_min = fmin(day_min, eqf);
            day_dd = fmax(day_dd, dd);
#if PASSIVBOT_BTC_RISK_ENABLED
            update_btc_risk_state(btc_risk, eqf, btc_prices[k]);
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
            update_equity_balance_diff_state(
                equity_balance_diff_state, balance, eqf, btc_prices, k,
                starting_balance, any_fill
            );
#endif
            day_touched = true;
            if (!liq) {
                float twe_net = (
                    long_side.psize * long_side.pprice
                    - short_side.psize * short_side.pprice
                ) * c_mult / balance;
                float twe_abs = fabs(twe_net);
                total_wallet_exposure_samples += 1.0f;
                total_wallet_exposure_mean += (
                    twe_abs - total_wallet_exposure_mean
                ) / total_wallet_exposure_samples;
                total_wallet_exposure_max = fmax(
                    total_wallet_exposure_max, twe_abs
                );
            }
            if (liq) {
                liq_day = di;
                alive = false;
            }
        }
    }

    if (day_touched && cur_day >= 0 && cur_day < D) {
        int o = (int(b) * D + cur_day) * DAILY_COLS;
        daily[o + 0] = day_end;
        daily[o + 1] = day_min;
        daily[o + 2] = day_dd;
        daily[o + 3] = day_volume;
        daily[o + 4] = day_has_fill;
        daily[o + 5] = balance - day_start_balance;
        daily[o + 6] = balance;
        daily[o + 7] = day_fill_count;
#if PASSIVBOT_BTC_RISK_ENABLED
        write_btc_risk_day(btc_risk, daily, o, 8);
#endif
    }

    if (long_side.pos_open_k >= 0.0f && last_eq_k >= 0.0f) {
        float held_min = last_eq_k - long_side.pos_open_k;
        held_max_min = fmax(held_max_min, held_min);
        held_sum_min += held_min;
        held_count += 1.0f;
    }
    if (short_side.pos_open_k >= 0.0f && last_eq_k >= 0.0f) {
        float held_min = last_eq_k - short_side.pos_open_k;
        held_max_min = fmax(held_max_min, held_min);
        held_sum_min += held_min;
        held_count += 1.0f;
    }
    if (long_position_last_fill_k >= 0.0f && last_eq_k >= 0.0f) {
        position_unchanged_max_min = fmax(
            position_unchanged_max_min, last_eq_k - long_position_last_fill_k
        );
    }
    if (short_position_last_fill_k >= 0.0f && last_eq_k >= 0.0f) {
        position_unchanged_max_min = fmax(
            position_unchanged_max_min, last_eq_k - short_position_last_fill_k
        );
    }
    if (pnl_recovery_peak_k >= 0.0f && last_eq_k >= 0.0f) {
        pnl_recovery_max_min = fmax(
            pnl_recovery_max_min, last_eq_k - pnl_recovery_peak_k
        );
    }
    int so = int(b) * SCALAR_COLS;
    scalars[so + 0] = max_dd;
    scalars[so + 1] = held_max_min * interval_ms;
    scalars[so + 2] = gap_max_min * interval_ms;
    scalars[so + 3] = first_fill_k >= 0.0f ? first_fill_k * interval_ms : -1.0f;
    scalars[so + 4] = last_fill_k >= 0.0f ? last_fill_k * interval_ms : -1.0f;
    scalars[so + 5] = recovery_max_min * interval_ms;
    scalars[so + 6] = last_high_k >= 0.0f ? last_high_k * interval_ms : -1.0f;
    scalars[so + 7] = first_eq_k >= 0.0f ? first_eq_k * interval_ms : -1.0f;
    scalars[so + 8] = last_eq_k >= 0.0f ? last_eq_k * interval_ms : -1.0f;
    scalars[so + 9] = float(liq_day);
    scalars[so + 10] = balance;
    scalars[so + 11] = long_side.psize;
    scalars[so + 12] = long_side.pprice;
    scalars[so + 13] = alive ? 1.0f : 0.0f;
    scalars[so + 14] = long_side.psize > 0.0f || short_side.psize > 0.0f ? 1.0f : 0.0f;
    scalars[so + 15] = short_side.psize;
    scalars[so + 16] = short_side.pprice;
    scalars[so + 17] = 0.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    float long_terminal_count = long_hsl.halted
        && long_hsl.current_halt_start_k >= 0.0f && last_eq_k >= 0.0f
        ? 1.0f : 0.0f;
    float short_terminal_count = short_hsl.halted
        && short_hsl.current_halt_start_k >= 0.0f && last_eq_k >= 0.0f
        ? 1.0f : 0.0f;
    float long_terminal_duration = long_terminal_count > 0.0f
        ? fmax(last_eq_k - long_hsl.current_halt_start_k, 0.0f) : 0.0f;
    float short_terminal_duration = short_terminal_count > 0.0f
        ? fmax(last_eq_k - short_hsl.current_halt_start_k, 0.0f) : 0.0f;
    float terminal_count = long_terminal_count + short_terminal_count;
    scalars[so + 18] = long_hsl.enabled ? 1.0f : 0.0f;
    scalars[so + 19] = short_hsl.enabled ? 1.0f : 0.0f;
    scalars[so + 20] = long_hsl.triggers;
    scalars[so + 21] = short_hsl.triggers;
    scalars[so + 22] = long_hsl.restarts;
    scalars[so + 23] = short_hsl.restarts;
    scalars[so + 24] = hsl_tier_samples_total;
    scalars[so + 25] = hsl_tier_samples_yellow;
    scalars[so + 26] = hsl_tier_samples_orange;
    scalars[so + 27] = hsl_tier_samples_red;
    scalars[so + 28] = long_hsl.halt_duration_sum_steps
        + short_hsl.halt_duration_sum_steps
        + long_terminal_duration + short_terminal_duration;
    scalars[so + 29] = fmax(
        fmax(long_hsl.halt_duration_max_steps, short_hsl.halt_duration_max_steps),
        fmax(long_terminal_duration, short_terminal_duration)
    );
    scalars[so + 30] = long_hsl.halt_duration_count
        + short_hsl.halt_duration_count + terminal_count;
    scalars[so + 31] = long_hsl.trigger_drawdown_sum
        + short_hsl.trigger_drawdown_sum;
    scalars[so + 32] = long_hsl.trigger_drawdown_count
        + short_hsl.trigger_drawdown_count;
    scalars[so + 33] = long_hsl.flatten_time_sum_steps
        + short_hsl.flatten_time_sum_steps;
    scalars[so + 34] = long_hsl.flatten_time_count
        + short_hsl.flatten_time_count;
    scalars[so + 35] = long_hsl.restart_retrigger_count
        + short_hsl.restart_retrigger_count;
    float panic_drawdown_count = long_hsl.panic_loss_drawdown_count
        + short_hsl.panic_loss_drawdown_count;
    float panic_drawdown_min = long_hsl.panic_loss_drawdown_count > 0.0f
        ? (short_hsl.panic_loss_drawdown_count > 0.0f
            ? fmin(
                long_hsl.panic_loss_drawdown_min,
                short_hsl.panic_loss_drawdown_min
            ) : long_hsl.panic_loss_drawdown_min)
        : (short_hsl.panic_loss_drawdown_count > 0.0f
            ? short_hsl.panic_loss_drawdown_min : 0.0f);
    scalars[so + 36] = long_hsl.halt_to_restart_equity_loss
        + short_hsl.halt_to_restart_equity_loss;
    scalars[so + 37] = long_hsl.panic_close_loss_sum
        + short_hsl.panic_close_loss_sum;
    scalars[so + 38] = fmax(
        long_hsl.panic_close_loss_max, short_hsl.panic_close_loss_max
    );
    scalars[so + 39] = panic_drawdown_min;
    scalars[so + 40] = long_hsl.panic_loss_drawdown_sum
        + short_hsl.panic_loss_drawdown_sum;
    scalars[so + 41] = fmax(
        long_hsl.panic_loss_drawdown_max,
        short_hsl.panic_loss_drawdown_max
    );
    scalars[so + 42] = panic_drawdown_count;
#else
    for (int column = 18; column <= 42; ++column) {
        scalars[so + column] = 0.0f;
    }
#endif
    scalars[so + 43] = profit_sum;
    scalars[so + 44] = loss_sum;
    scalars[so + 45] = position_unchanged_max_min * interval_ms;
    scalars[so + 46] = long_enabled
        ? long_side.allowed_wel * long_side.initial_qty_pct : 0.0f;
    scalars[so + 47] = short_enabled
        ? short_side.allowed_wel * short_side.initial_qty_pct : 0.0f;
    scalars[so + 48] = total_wallet_exposure_max;
    scalars[so + 49] = total_wallet_exposure_mean;
    scalars[so + 50] = fill_count;
    scalars[so + 51] = fill_count_entry;
    scalars[so + 52] = fill_count_long;
    scalars[so + 53] = fills_active_days_count;
    scalars[so + 54] = pnl_recovery_max_min * interval_ms;
    scalars[so + 55] = held_sum_min * interval_ms;
    scalars[so + 56] = held_count;
    scalars[so + 57] = account_recovery_max_min * interval_ms;
    scalars[so + 58] = profit_sum_long;
    scalars[so + 59] = loss_sum_long;
    scalars[so + 60] = profit_sum_short;
    scalars[so + 61] = loss_sum_short;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    scalars[so + 62] = long_hsl.enabled ? long_hsl.drawdown_ema_max : 0.0f;
    scalars[so + 63] = short_hsl.enabled ? short_hsl.drawdown_ema_max : 0.0f;
    scalars[so + 64] = hsl_strategy_equity_recovery_max_steps(
        long_hsl_strategy_eq
    ) * interval_ms;
    scalars[so + 65] = hsl_strategy_equity_recovery_max_steps(
        short_hsl_strategy_eq
    ) * interval_ms;
#else
    scalars[so + 62] = 0.0f;
    scalars[so + 63] = 0.0f;
    scalars[so + 64] = 0.0f;
    scalars[so + 65] = 0.0f;
#endif
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    scalars[so + 66] = hsl_drawdown_ema_mean_worst_1pct(long_hsl_ema_tail);
    scalars[so + 67] = hsl_drawdown_ema_mean_worst_1pct(short_hsl_ema_tail);
#endif
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    scalars[so + 68] = hsl_strategy_equity_drawdown_max(long_hsl_strategy_eq);
    scalars[so + 69] = hsl_strategy_equity_drawdown_max(short_hsl_strategy_eq);
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    scalars[so + 70] = hsl_strategy_equity_drawdown_mean_worst_1pct(
        long_hsl_strategy_eq
    );
    scalars[so + 71] = hsl_strategy_equity_drawdown_mean_worst_1pct(
        short_hsl_strategy_eq
    );
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    write_equity_balance_diff_state(
        equity_balance_diff_state, equity_balance_diff, b
    );
#endif
}

kernel void passivbot_trailing_martingale(
    constant float* bars,
    constant int* flags,
    constant float* params,
    constant float* settings,
    constant int* sizes,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float2* rolling_pnl_values,
    device int2* rolling_pnl_indices,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
    uint b [[thread_position_in_grid]]
) {
    passivbot_single_coin_impl(
        bars, flags, params, settings, sizes,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
        entry_interval_stats, entry_interval_counts,
#endif
        daily, scalars, gap_hist,
        rolling_pnl_values, rolling_pnl_indices,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b
    );
}
