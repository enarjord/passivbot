#include <metal_stdlib>
using namespace metal;

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
constant int SIDE_PARAMS = 35;
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
        buy_order ? ceil_step(slipped, price_step) : floor_step(slipped, price_step),
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
    float tolerance = 1.0e-12f * fmax(requested_qty, minimum_qty) * 4.0f;
    if (requested_qty + tolerance >= minimum_qty) return requested_qty;
    float resized = fmin(minimum_qty, position_size);
    float remainder = position_size - resized;
    if (remainder > 0.0f && remainder + tolerance < minimum_qty) {
        resized = position_size;
    }
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

inline bool realized_loss_gate_allows(
    float net_pnl, float remaining_loss_budget,
    bool gate_enabled
) {
    return !gate_enabled || net_pnl >= 0.0f
        || -net_pnl <= remaining_loss_budget;
}

inline float float32_floor_nonnegative(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return fmax(value, 0.0f);
    return as_type<float>(as_type<uint>(value) - 1u);
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

struct EmaSide {
    float alpha0;
    float alpha1;
    float alpha2;
    float alpha1m;
    float alpha1h;
    float base_qty_pct;
    float ddf;
    float offset;
    float psize_weight;
    float w1h;
    float w1m;
    float cooldown_min;
    float twel;
    float base_wel;
    float allowed_wel;
    float entry_cap;
    float twel_enforcer_threshold;
    bool twel_enforcer_enabled;
    bool unstuck_enabled;
    bool unstuck_ema_gating_enabled;
    float unstuck_close_pct;
    float unstuck_ema_dist;
    float unstuck_loss_allowance_pct;
    float unstuck_threshold;
    float ema0;
    float ema1;
    float ema2;
    float vol1m;
    float vol1h;
    float psize;
    float pprice;
    float last_inc_k;
    float pos_open_k;
    int entry_ticks;
    float entry_qty;
    bool entry_market;
    int close_ticks;
    float close_qty;
    bool close_market;
    int secondary_close_ticks;
    float secondary_close_qty;
    bool secondary_close_market;
    int close_without_reducer_ticks;
    float close_without_reducer_qty;
    bool close_is_protective_reducer;
    bool close_is_panic;
};

inline void clear_pending_ema_orders(thread EmaSide& side) {
    side.entry_qty = 0.0f;
    side.close_qty = 0.0f;
    side.secondary_close_qty = 0.0f;
}

struct ReducerVariant {
    bool valid;
    bool is_unstuck;
    bool is_panic;
    int ticks;
    float qty;
    bool market;
    int secondary_ticks;
    float secondary_qty;
    bool secondary_market;
};

inline EmaSide load_side(constant float* params, int po, float seed_close) {
    EmaSide side;
    float span0 = params[po + 1];
    float span1 = params[po + 2];
    float span2 = sqrt(span0 * span1);
    float lo_span = fmin(span0, fmin(span1, span2));
    float hi_span = fmax(span0, fmax(span1, span2));
    float mid_span = span0 + span1 + span2 - lo_span - hi_span;
    side.alpha0 = clamp(2.0f / (lo_span + 1.0f), 0.0f, 1.0f);
    side.alpha1 = clamp(2.0f / (mid_span + 1.0f), 0.0f, 1.0f);
    side.alpha2 = clamp(2.0f / (hi_span + 1.0f), 0.0f, 1.0f);
    float span_h = params[po + 8];
    float span_m = params[po + 9];
    side.alpha1h = span_h > 0.0f ? 2.0f / (fmax(span_h, 1.0f) + 1.0f) : 0.0f;
    side.alpha1m = span_m > 0.0f
        ? clamp(2.0f / (span_m + 1.0f), 0.0f, 1.0f) : 0.0f;
    side.base_qty_pct = params[po + 0];
    side.ddf = params[po + 3];
    side.offset = params[po + 4];
    side.psize_weight = params[po + 5];
    side.w1h = params[po + 6];
    side.w1m = params[po + 7];
    side.cooldown_min = ceil(params[po + 10]);
    side.twel = params[po + 11];
    float allowance_pct = fmax(params[po + 12], 0.0f);
    bool legacy_raw_allowance = params[po + 13] > 0.5f;
    float base_wel = params[po + 34];
    if (!(isfinite(base_wel) && base_wel >= 0.0f)) base_wel = side.twel;
    side.base_wel = base_wel;
    float effective_allowance_pct = allowance_pct;
    if (!legacy_raw_allowance) {
        float max_effective = base_wel > 0.0f
            ? fmax(side.twel / base_wel - 1.0f, 0.0f) : 0.0f;
        effective_allowance_pct = fmin(allowance_pct, max_effective);
    }
    side.allowed_wel = base_wel > 0.0f
        ? base_wel * (1.0f + effective_allowance_pct) : 0.0f;
    bool twel_entry_gate_enabled = params[po + 14] > 0.5f;
    float twel_threshold = params[po + 15];
    float gate_cap = side.twel;
    if (isfinite(twel_threshold) && twel_threshold > 0.0f) {
        gate_cap = fmin(side.twel, side.twel * twel_threshold);
    }
    side.entry_cap = twel_entry_gate_enabled ? gate_cap : INFINITY;
    side.twel_enforcer_threshold = twel_threshold;
    side.twel_enforcer_enabled = params[po + 16] > 0.5f;
    side.unstuck_enabled = params[po + 17] > 0.5f;
    side.unstuck_ema_gating_enabled = params[po + 18] > 0.5f;
    side.unstuck_close_pct = params[po + 19];
    side.unstuck_ema_dist = params[po + 20];
    side.unstuck_loss_allowance_pct = params[po + 21];
    side.unstuck_threshold = params[po + 22];
    side.ema0 = seed_close;
    side.ema1 = seed_close;
    side.ema2 = seed_close;
    side.vol1m = 0.0f;
    side.vol1h = 0.0f;
    side.psize = 0.0f;
    side.pprice = 0.0f;
    side.last_inc_k = -1.0f;
    side.pos_open_k = -1.0f;
    side.entry_ticks = 0;
    side.entry_qty = 0.0f;
    side.entry_market = false;
    side.close_ticks = 0;
    side.close_qty = 0.0f;
    side.close_market = false;
    side.secondary_close_ticks = 0;
    side.secondary_close_qty = 0.0f;
    side.secondary_close_market = false;
    side.close_without_reducer_ticks = 0;
    side.close_without_reducer_qty = 0.0f;
    side.close_is_protective_reducer = false;
    side.close_is_panic = false;
    return side;
}

inline ReducerVariant empty_reducer_variant() {
    ReducerVariant result;
    result.valid = false;
    result.is_unstuck = false;
    result.is_panic = false;
    result.ticks = 0;
    result.qty = 0.0f;
    result.market = false;
    result.secondary_ticks = 0;
    result.secondary_qty = 0.0f;
    result.secondary_market = false;
    return result;
}

inline ReducerVariant finalize_reducer_variant(
    float psize,
    int ordinary_ticks,
    float ordinary_qty,
    int reducer_ticks,
    float reducer_qty,
    bool is_unstuck,
    bool is_long,
    float market_price,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    ReducerVariant result = empty_reducer_variant();
    if (!(psize > 0.0f && reducer_ticks > 0 && reducer_qty > 0.0f)) {
        return result;
    }
    float reducer_price = float(reducer_ticks) * price_step;
    bool reducer_market = should_use_ordinary_market_execution(
        reducer_ticks, !is_long, market_price, price_step,
        market_orders_allowed, market_order_near_touch_threshold
    );
    if (reducer_market) {
        reducer_qty = resize_market_close_qty(
            reducer_qty, psize, market_price, qty_step,
            min_qty, min_cost, c_mult
        );
    }
    float reducer_min_price = reducer_market ? market_price : reducer_price;
    float reducer_min = min_entry_qty(
        reducer_min_price, qty_step, min_qty, min_cost, c_mult
    );
    reducer_qty = fmin(psize, reducer_qty);
    if (ordinary_qty > 0.0f && ordinary_ticks > 0) {
        float ordinary_price = float(ordinary_ticks) * price_step;
        bool ordinary_market = should_use_ordinary_market_execution(
            ordinary_ticks, !is_long, market_price, price_step,
            market_orders_allowed, market_order_near_touch_threshold
        );
        if (ordinary_market) {
            ordinary_qty = resize_market_close_qty(
                ordinary_qty, psize, market_price, qty_step,
                min_qty, min_cost, c_mult
            );
        }
        float ordinary_min_price = ordinary_market ? market_price : ordinary_price;
        float ordinary_min = min_entry_qty(
            ordinary_min_price, qty_step, min_qty, min_cost, c_mult
        );
        if (ordinary_qty + reducer_qty > psize) {
            ordinary_qty = fmax(round_step(psize - reducer_qty, qty_step), 0.0f);
        }
        if (ordinary_qty >= ordinary_min) {
            float remainder = fmax(
                round_step(psize - reducer_qty - ordinary_qty, qty_step), 0.0f
            );
            float minimum_any = fmin(ordinary_min, reducer_min);
            if (remainder > 0.0f && remainder < minimum_any) {
                ordinary_qty = fmin(
                    psize - reducer_qty,
                    round_step(ordinary_qty + remainder, qty_step)
                );
            }
            result.secondary_ticks = ordinary_ticks;
            result.secondary_qty = ordinary_qty;
            result.secondary_market = ordinary_market;
        }
    }
    if (result.secondary_qty <= 0.0f) {
        float remainder = fmax(round_step(psize - reducer_qty, qty_step), 0.0f);
        if (remainder > 0.0f && remainder < reducer_min) {
            reducer_qty = psize;
        }
    }
    result.valid = reducer_qty > 0.0f;
    result.is_unstuck = is_unstuck;
    result.is_panic = false;
    result.ticks = reducer_ticks;
    result.qty = reducer_qty;
    result.market = reducer_market;
    return result;
}

inline void apply_reducer_variant(
    thread EmaSide& side,
    ReducerVariant variant
) {
    side.close_ticks = variant.ticks;
    side.close_qty = variant.qty;
    side.close_market = variant.market;
    side.secondary_close_ticks = variant.secondary_ticks;
    side.secondary_close_qty = variant.secondary_qty;
    side.secondary_close_market = variant.secondary_market;
    side.close_is_protective_reducer = variant.valid;
    side.close_is_panic = variant.is_panic;
}

inline void restore_ordinary_close(thread EmaSide& side) {
    side.close_ticks = side.close_without_reducer_ticks;
    side.close_qty = side.close_without_reducer_qty;
    side.close_market = false;
    side.secondary_close_ticks = 0;
    side.secondary_close_qty = 0.0f;
    side.secondary_close_market = false;
    side.close_is_protective_reducer = false;
    side.close_is_panic = false;
}

inline void prepare_ordinary_market_close(
    thread EmaSide& side,
    bool is_long,
    float market_price,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    side.close_market = side.close_qty > 0.0f
        && should_use_ordinary_market_execution(
            side.close_ticks, !is_long, market_price, price_step,
            market_orders_allowed, market_order_near_touch_threshold
        );
    if (side.close_market) {
        side.close_qty = resize_market_close_qty(
            side.close_qty, side.psize, market_price, qty_step,
            min_qty, min_cost, c_mult
        );
    }
}

inline ReducerVariant generated_reducer_variant(thread EmaSide& side) {
    if (!side.close_is_protective_reducer || !(side.close_qty > 0.0f)) {
        return empty_reducer_variant();
    }
    ReducerVariant result;
    result.valid = true;
    result.is_unstuck = false;
    result.is_panic = false;
    result.ticks = side.close_ticks;
    result.qty = side.close_qty;
    result.market = side.close_market;
    result.secondary_ticks = side.secondary_close_ticks;
    result.secondary_qty = side.secondary_close_qty;
    result.secondary_market = side.secondary_close_market;
    return result;
}

inline ReducerVariant panic_reducer_variant(
    thread EmaSide& side,
    bool is_long,
    int touch_down_ticks,
    int touch_up_ticks
) {
    ReducerVariant result = empty_reducer_variant();
    if (!(side.psize > 0.0f)) return result;
    result.valid = true;
    result.is_panic = true;
    result.ticks = max(
        is_long ? touch_down_ticks - 1 : touch_up_ticks + 1, 1
    );
    result.qty = side.psize;
    return result;
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

inline void apply_twel_reducer(
    thread EmaSide& side,
    bool is_long,
    float balance,
    float price_now,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    bool market_orders_allowed,
    float market_order_near_touch_threshold
) {
    side.secondary_close_ticks = 0;
    side.secondary_close_qty = 0.0f;
    side.close_is_protective_reducer = false;
    float target = side.twel * side.twel_enforcer_threshold;
    int reducer_ticks = is_long
        ? int(floor(price_now * 0.9995f / price_step + 1.0e-6f))
        : int(ceil(price_now * 1.0005f / price_step - 1.0e-6f));
    reducer_ticks = max(reducer_ticks, 1);
    float reducer_price = float(reducer_ticks) * price_step;
    float reducer_qty = side.twel_enforcer_enabled
            && side.twel_enforcer_threshold > 0.0f
        ? total_exposure_reducer_qty(
            side.psize, side.pprice, balance, target, reducer_price,
            qty_step, min_qty, min_cost, c_mult
        )
        : 0.0f;
    if (!(reducer_qty > 0.0f)) return;

    apply_reducer_variant(
        side,
        finalize_reducer_variant(
            side.psize, side.close_ticks, side.close_qty,
            reducer_ticks, reducer_qty, false, is_long, price_now,
            market_orders_allowed, market_order_near_touch_threshold,
            qty_step, price_step,
            min_qty, min_cost, c_mult
        )
    );
}

inline ReducerVariant unstuck_reducer_variant(
    thread EmaSide& side,
    bool is_long,
    float balance,
    float balance_peak,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    bool market_orders_allowed,
    float market_order_near_touch_threshold
) {
    ReducerVariant none = empty_reducer_variant();
    if (!(side.unstuck_enabled
        && side.unstuck_loss_allowance_pct > 0.0f
        && side.unstuck_close_pct > 0.0f
        && side.unstuck_threshold > 0.0f
        && balance > 0.0f
        && balance_peak > 0.0f
        && side.psize > 0.0f
        && side.pprice > 0.0f
        && side.allowed_wel > 0.0f
        && price_now > 0.0f)) {
        return none;
    }
    float allowance_pct = side.unstuck_loss_allowance_pct * side.twel;
    float allowance = float32_floor_nonnegative(
        fmax(balance - balance_peak * (1.0f - allowance_pct), 0.0f)
    );
    if (!(allowance > 0.0f)) return none;
    float wallet_exposure = side.psize * side.pprice * c_mult / balance;
    if (!(wallet_exposure / side.allowed_wel > side.unstuck_threshold)) {
        return none;
    }
    if (side.unstuck_ema_gating_enabled) {
        float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
        float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
        int trigger_ticks = is_long
            ? int(ceil(
                upper * (1.0f + side.unstuck_ema_dist) / price_step
                    - 1.0e-6f
            ))
            : int(floor(
                lower * (1.0f - side.unstuck_ema_dist) / price_step
                    + 1.0e-6f
            ));
        bool triggered = is_long
            ? touch_down_ticks >= trigger_ticks
            : touch_up_ticks <= trigger_ticks;
        if (!triggered) return none;
    }

    int reducer_ticks = max(
        is_long ? touch_up_ticks : touch_down_ticks, 1
    );
    float reducer_price = float(reducer_ticks) * price_step;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float target_qty = floor_step(
        balance * side.allowed_wel * side.unstuck_close_pct
            / fmax(reducer_price * c_mult, 1.0e-12f),
        qty_step
    );
    float reducer_qty = fmin(
        side.psize, fmax(reducer_min, target_qty)
    );
    float gross_pnl = reducer_qty * c_mult * (
        is_long ? reducer_price - side.pprice : side.pprice - reducer_price
    );
    if (gross_pnl < 0.0f && -gross_pnl > allowance) {
        float scaled_qty = fmin(
            side.psize, reducer_qty * allowance / -gross_pnl
        );
        reducer_qty = fmin(
            side.psize,
            fmax(reducer_min, floor_step(scaled_qty, qty_step))
        );
    }
    return finalize_reducer_variant(
        side.psize,
        side.close_without_reducer_ticks,
        side.close_without_reducer_qty,
        reducer_ticks,
        reducer_qty,
        true,
        is_long,
        price_now,
        market_orders_allowed,
        market_order_near_touch_threshold,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult
    );
}

inline bool reducer_variant_preferred(
    ReducerVariant left,
    ReducerVariant right,
    bool is_long
) {
    if (!left.valid) return false;
    if (!right.valid) return true;
    if (left.qty != right.qty) return left.qty > right.qty;
    if (left.is_panic != right.is_panic) return left.is_panic;
    if (left.ticks != right.ticks) {
        return is_long ? left.ticks < right.ticks : left.ticks > right.ticks;
    }
    if (left.is_unstuck != right.is_unstuck) return left.is_unstuck;
    return false;
}

inline void order_reducer_variants(
    ReducerVariant first,
    ReducerVariant second,
    bool is_long,
    thread ReducerVariant& preferred,
    thread ReducerVariant& fallback
) {
    if (reducer_variant_preferred(second, first, is_long)) {
        preferred = second;
        fallback = first;
    } else {
        preferred = first;
        fallback = second;
    }
}

inline ReducerVariant reducer_variant_at(
    ReducerVariant preferred,
    ReducerVariant fallback,
    int index
) {
    return index == 0 ? preferred : fallback;
}

inline bool global_reducer_prefers_long(
    ReducerVariant long_candidate,
    ReducerVariant short_candidate
) {
    if (long_candidate.qty != short_candidate.qty) {
        return long_candidate.qty > short_candidate.qty;
    }
    if (long_candidate.is_panic != short_candidate.is_panic) {
        return long_candidate.is_panic;
    }
    return true;
}

inline bool gate_reducer_variant(
    ReducerVariant variant,
    float pprice,
    bool is_long,
    float price_step,
    float c_mult,
    float maker_fee,
    float taker_fee,
    float market_order_slippage_pct,
    float market_price,
    bool gate_enabled,
    thread float& remaining_loss_budget
) {
    if (!variant.valid || !(variant.qty > 0.0f && pprice > 0.0f)) {
        return false;
    }
    if (variant.is_panic) return true;
    float price = variant.market
        ? ordinary_market_fill_price(
            market_price, !is_long, market_order_slippage_pct, price_step
        )
        : float(variant.ticks) * price_step;
    float fee_rate = variant.market ? taker_fee : maker_fee;
    float gross_pnl = variant.qty * c_mult * (
        is_long ? price - pprice : pprice - price
    );
    float net_pnl = gross_pnl - variant.qty * price * c_mult * fee_rate;
    if (!realized_loss_gate_allows(
            net_pnl, remaining_loss_budget, gate_enabled)) {
        return false;
    }
    if (gate_enabled && net_pnl < 0.0f) {
        remaining_loss_budget = float32_floor_nonnegative(
            fmax(remaining_loss_budget + net_pnl, 0.0f)
        );
    }
    return true;
}

inline void update_indicators(
    thread EmaSide& side,
    float close,
    float log_range,
    float hour_lr,
    bool valid,
    bool hour_valid
) {
    if (hour_valid && side.alpha1h > 0.0f) {
        side.vol1h = fma(side.alpha1h, hour_lr - side.vol1h, side.vol1h);
    }
    if (valid) {
        side.ema0 = fma(side.alpha0, close - side.ema0, side.ema0);
        side.ema1 = fma(side.alpha1, close - side.ema1, side.ema1);
        side.ema2 = fma(side.alpha2, close - side.ema2, side.ema2);
        if (side.alpha1m > 0.0f) {
            side.vol1m = fma(side.alpha1m, log_range - side.vol1m, side.vol1m);
        }
    }
}

inline void generate_long_orders(
    thread EmaSide& side,
    float balance,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    float kf,
    bool block_initial,
    bool market_orders_allowed,
    float market_order_near_touch_threshold
) {
    float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
    float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
    float mult = fmax(1.0f + side.vol1h * side.w1h + side.vol1m * side.w1m, 1.0f);
    float eff_off = side.offset * mult;
    float current_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * price_now * c_mult / balance : 0.0f;
    float current_cost_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * side.pprice * c_mult / balance : 0.0f;
    float swer = side.psize > 0.0f && balance > 0.0f
        ? current_we / fmax(side.base_wel, 1.0e-12f) : 0.0f;
    float inv_shift = swer * side.psize_weight;

    int bid_ticks = min(
        int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
        touch_down_ticks
    );
    float bid_price = float(bid_ticks) * price_step;
    bool entry_market = should_use_ordinary_market_execution(
        bid_ticks, true, price_now, price_step, market_orders_allowed,
        market_order_near_touch_threshold
    );
    float entry_exposure_price = entry_market ? price_now : bid_price;
    float min_q = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
    float base_q = fmax(min_q, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(bid_price, 1.0e-12f) / c_mult,
        qty_step
    ));
    float e_qty = round_step(
        base_q * fmax(1.0f + fmax(swer, 0.0f) * side.ddf, 1.0f), qty_step
    );
    bool cooldown = side.cooldown_min > 0.0f && side.last_inc_k >= 0.0f
        && kf < side.last_inc_k + side.cooldown_min;
    float cap = side.entry_cap - 1.0e-7f;
    float headroom = (cap * balance - side.psize * side.pprice * c_mult)
        / fmax(entry_exposure_price * c_mult, 1.0e-12f);
    bool over = (side.psize * side.pprice + e_qty * entry_exposure_price) * c_mult
        / fmax(balance, 1.0e-9f) >= cap;
    float capped = floor_step(headroom, qty_step);
    if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
    if (current_cost_we >= cap || cooldown || bid_price <= 0.0f || balance <= 0.0f
        || side.base_qty_pct <= 0.0f
        || (block_initial && side.psize <= 0.0f)) {
        e_qty = 0.0f;
    }
    side.entry_ticks = bid_ticks;
    side.entry_qty = e_qty;
    side.entry_market = entry_market && e_qty > 0.0f;

    int ask_ticks = max(
        int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
        touch_up_ticks
    );
    float ask_price = float(ask_ticks) * price_step;
    float min_cq = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
    float clip = fmin(side.psize, fmax(min_cq, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(ask_price, 1.0e-12f) / c_mult,
        qty_step
    )));
    float c_qty = side.psize <= min_cq || side.psize - clip < min_cq
        ? side.psize : clip;
    if (side.psize <= 0.0f || ask_price <= 0.0f) c_qty = 0.0f;
    side.close_ticks = ask_ticks;
    side.close_qty = c_qty;
    side.close_without_reducer_ticks = ask_ticks;
    side.close_without_reducer_qty = c_qty;
    apply_twel_reducer(
        side, true, balance, price_now, qty_step, price_step,
        min_qty, min_cost, c_mult, market_orders_allowed,
        market_order_near_touch_threshold
    );
}

inline void generate_short_orders(
    thread EmaSide& side,
    float balance,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    float kf,
    bool block_initial,
    bool market_orders_allowed,
    float market_order_near_touch_threshold
) {
    float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
    float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
    float mult = fmax(1.0f + side.vol1h * side.w1h + side.vol1m * side.w1m, 1.0f);
    float eff_off = side.offset * mult;
    float swer = side.psize > 0.0f && balance > 0.0f
        ? -side.psize * price_now * c_mult / balance
            / fmax(side.base_wel, 1.0e-12f)
        : 0.0f;
    float current_cost_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * side.pprice * c_mult / balance : 0.0f;
    float inv_shift = swer * side.psize_weight;

    int ask_ticks = max(
        int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
        touch_up_ticks
    );
    float ask_price = float(ask_ticks) * price_step;
    bool entry_market = should_use_ordinary_market_execution(
        ask_ticks, false, price_now, price_step, market_orders_allowed,
        market_order_near_touch_threshold
    );
    float entry_exposure_price = entry_market ? price_now : ask_price;
    float min_q = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
    float base_q = fmax(min_q, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(ask_price, 1.0e-12f) / c_mult,
        qty_step
    ));
    float e_qty = round_step(
        base_q * fmax(1.0f + fmax(-swer, 0.0f) * side.ddf, 1.0f), qty_step
    );
    float market_min_q = entry_market
        ? min_entry_qty(price_now, qty_step, min_qty, min_cost, c_mult)
        : min_q;
    if (entry_market && e_qty < market_min_q) e_qty = market_min_q;
    bool cooldown = side.cooldown_min > 0.0f && side.last_inc_k >= 0.0f
        && kf < side.last_inc_k + side.cooldown_min;
    float cap = side.entry_cap - 1.0e-7f;
    float headroom = (cap * balance - side.psize * side.pprice * c_mult)
        / fmax(entry_exposure_price * c_mult, 1.0e-12f);
    bool over = (side.psize * side.pprice + e_qty * entry_exposure_price) * c_mult
        / fmax(balance, 1.0e-9f) >= cap;
    float capped = floor_step(headroom, qty_step);
    if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
    if (entry_market && e_qty + 1.0e-12f < market_min_q) e_qty = 0.0f;
    if (current_cost_we >= cap || cooldown || ask_price <= 0.0f || balance <= 0.0f
        || side.base_qty_pct <= 0.0f
        || (block_initial && side.psize <= 0.0f)) {
        e_qty = 0.0f;
    }
    side.entry_ticks = ask_ticks;
    side.entry_qty = e_qty;
    side.entry_market = entry_market && e_qty > 0.0f;

    int bid_ticks = min(
        int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
        touch_down_ticks
    );
    float bid_price = float(bid_ticks) * price_step;
    float min_cq = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
    float clip = fmin(side.psize, fmax(min_cq, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(bid_price, 1.0e-12f) / c_mult,
        qty_step
    )));
    float c_qty = side.psize <= min_cq || side.psize - clip < min_cq
        ? side.psize : clip;
    if (side.psize <= 0.0f || bid_price <= 0.0f) c_qty = 0.0f;
    side.close_ticks = bid_ticks;
    side.close_qty = c_qty;
    side.close_without_reducer_ticks = bid_ticks;
    side.close_without_reducer_qty = c_qty;
    apply_twel_reducer(
        side, false, balance, price_now, qty_step, price_step,
        min_qty, min_cost, c_mult, market_orders_allowed,
        market_order_near_touch_threshold
    );
}

inline void gate_generated_close(
    thread float& qty,
    int ticks,
    float pprice,
    bool is_long,
    float price_step,
    float c_mult,
    float maker_fee,
    float taker_fee,
    float market_order_slippage_pct,
    float market_price,
    bool market_execution,
    bool gate_enabled,
    thread float& remaining_loss_budget
) {
    if (!(qty > 0.0f && ticks > 0 && pprice > 0.0f)) return;
    float price = market_execution
        ? ordinary_market_fill_price(
            market_price, !is_long, market_order_slippage_pct, price_step
        )
        : float(ticks) * price_step;
    float fee_rate = market_execution ? taker_fee : maker_fee;
    float gross_pnl = qty * c_mult
        * (is_long ? price - pprice : pprice - price);
    float net_pnl = gross_pnl - qty * price * c_mult * fee_rate;
    if (!realized_loss_gate_allows(
            net_pnl, remaining_loss_budget, gate_enabled)) {
        qty = 0.0f;
        return;
    }
    if (gate_enabled && net_pnl < 0.0f) {
        remaining_loss_budget = float32_floor_nonnegative(
            fmax(remaining_loss_budget + net_pnl, 0.0f)
        );
    }
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
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    const int recovery_stride = sizes[8];
    const int recovery_sample_count = sizes[9];
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
#if defined(PASSIVBOT_EMA_LONG_ONLY)
    const bool long_enabled = true;
    const bool short_enabled = false;
#elif defined(PASSIVBOT_EMA_SHORT_ONLY)
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
    const bool market_orders_allowed = settings[19] > 0.5f;
    const float market_order_near_touch_threshold = fmax(settings[20], 0.0f);
    const int pnl_lookback_bars = max(sizes[6], 0);
    const int rolling_capacity = sizes[5];
    const float log_bin_scale = 127.0f / log(4000001.0f);

    const int po = int(b) * P;
    const int seed_k = clamp(first_valid, 0, T - 1);
    const float seed_close = bars[seed_k * 5 + 2];
    EmaSide long_side = load_side(params, po, seed_close);
    EmaSide short_side = load_side(params, po + SIDE_PARAMS, seed_close);
    HslState long_hsl = load_hsl(params, po, 23);
    HslState short_hsl = load_hsl(params, po + SIDE_PARAMS, 23);
    HslStrategyEquityStats long_hsl_strategy_eq = init_hsl_strategy_equity_stats();
    HslStrategyEquityStats short_hsl_strategy_eq = init_hsl_strategy_equity_stats();
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    HslDrawdownEmaTailStats long_hsl_ema_tail = init_hsl_drawdown_ema_tail_stats();
    HslDrawdownEmaTailStats short_hsl_ema_tail = init_hsl_drawdown_ema_tail_stats();
#endif
#if defined(PASSIVBOT_EMA_HSL_DISABLED)
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
    float hsl_tier_samples_total = 0.0f;
    float hsl_tier_samples_yellow = 0.0f;
    float hsl_tier_samples_orange = 0.0f;
    float hsl_tier_samples_red = 0.0f;

    int cur_day = flags[2];
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

    for (int k = 1; k < T - 1; ++k) {
        const int bo = k * 5;
        const int fo = k * 11;
        const float high = bars[bo + 0];
        const float low = bars[bo + 1];
        const float close = bars[bo + 2];
        const float log_range = bars[bo + 3];
        const float hour_lr = bars[bo + 4];
        const bool valid = flags[fo + 0] != 0;
        const bool can_gen = flags[fo + 1] != 0;
        const int di = flags[fo + 2];
        const bool hour_valid = flags[fo + 3] != 0;
        const int high_fill_max_tick = flags[fo + 4];
        const int low_nonfill_max_tick = flags[fo + 5];
        const int touch_down_tick = flags[fo + 6];
        const int touch_up_tick = flags[fo + 7];
        const float kf = float(k);

        if (!valid) {
            clear_pending_ema_orders(long_side);
            clear_pending_ema_orders(short_side);
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
        bool long_primary_market = long_side.close_is_panic
            ? long_hsl_panic_market : long_side.close_market;
        bool long_primary_close_fill = valid && alive && long_enabled
            && long_side.close_qty > 0.0f && long_side.psize > 0.0f
            && (long_primary_market
                || long_side.close_ticks <= high_fill_max_tick);
        bool long_secondary_close_fill = valid && alive && long_enabled
            && long_side.secondary_close_qty > 0.0f && long_side.psize > 0.0f
            && (long_side.secondary_close_market
                || long_side.secondary_close_ticks <= high_fill_max_tick);
        bool long_secondary_first = long_secondary_close_fill
            && (!long_primary_close_fill
                || long_side.secondary_close_ticks < long_side.close_ticks);
        for (int rank = 0; rank < 2; ++rank) {
            bool use_secondary = long_secondary_first ? rank == 0 : rank == 1;
            bool reachable = use_secondary
                ? long_secondary_close_fill : long_primary_close_fill;
            if (!reachable || long_side.psize <= 0.0f) continue;
            int close_ticks = use_secondary
                ? long_side.secondary_close_ticks : long_side.close_ticks;
            float requested_qty = use_secondary
                ? long_side.secondary_close_qty : long_side.close_qty;
            bool market_execution = use_secondary
                ? long_side.secondary_close_market : long_primary_market;
            float cp = market_execution
                ? ordinary_market_fill_price(
                    close, false, market_order_slippage_pct, price_step
                )
                : float(close_ticks) * price_step;
            float adj = fmin(round_step(requested_qty, qty_step), long_side.psize);
            if (!(adj > 0.0f)) continue;
            float pnl = adj * c_mult * (cp - long_side.pprice);
            float fee = adj * cp * c_mult
                * (market_execution ? taker_fee : maker_fee);
            float net_pnl = pnl - fee;
            if (!use_secondary && long_side.close_is_panic) {
                float current_equity = balance
                    + long_side.psize * c_mult * (close - long_side.pprice)
                    + (short_side.psize > 0.0f
                        ? short_side.psize * c_mult * (short_side.pprice - close)
                        : 0.0f);
                record_hsl_panic_fill(long_hsl, net_pnl, current_equity);
            }
            record_directional_gross_pnl(
                pnl, profit_sum, loss_sum, profit_sum_long, loss_sum_long
            );
            balance += net_pnl;
            record_realized_net(
                net_pnl, realized_pnl_cumsum_last, realized_pnl_cumsum_max,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                day_fill_count, fill_count, fill_count_entry, fill_count_long,
                pnl_recovery_peak, pnl_recovery_peak_k,
                pnl_recovery_max_min, kf,
                false, true
            );
            record_hsl_rolling_pnl(
                long_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                long_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, long_coin_hsl_rolling, net_pnl
            );
            float new_psize = fmax(round_step(long_side.psize - adj, qty_step), 0.0f);
            bool went_flat = new_psize <= 0.0f;
            long_side.psize = new_psize;
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
            day_volume += fabs(adj) * cp / balance;
            long_close_fill = true;
            if (use_secondary) {
                long_side.secondary_close_qty = 0.0f;
            } else {
                long_side.close_qty = 0.0f;
            }
        }

        bool long_entry_fill = valid && alive && long_enabled
            && long_side.entry_qty > 0.0f
            && (long_side.entry_market
                || long_side.entry_ticks > low_nonfill_max_tick);
        if (long_entry_fill) {
            float ep = long_side.entry_market
                ? ordinary_market_fill_price(
                    close, true, market_order_slippage_pct, price_step
                )
                : float(long_side.entry_ticks) * price_step;
            float eq = round_step(long_side.entry_qty, qty_step);
            float fee = eq * ep * c_mult
                * (long_side.entry_market ? taker_fee : maker_fee);
            balance -= fee;
            record_realized_net(
                -fee, realized_pnl_cumsum_last, realized_pnl_cumsum_max,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                day_fill_count, fill_count, fill_count_entry, fill_count_long,
                pnl_recovery_peak, pnl_recovery_peak_k,
                pnl_recovery_max_min, kf,
                true, true
            );
            record_hsl_rolling_pnl(
                long_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                long_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, long_coin_hsl_rolling, -fee
            );
            bool was_flat = long_side.psize <= 0.0f;
            float new_psize = round_step(long_side.psize + eq, qty_step);
            float new_pprice = was_flat ? ep
                : long_side.pprice * (long_side.psize / fmax(new_psize, 1.0e-12f))
                    + ep * (eq / fmax(new_psize, 1.0e-12f));
            if (was_flat) long_side.pos_open_k = kf;
            long_side.psize = new_psize;
            long_side.pprice = new_pprice;
            long_side.last_inc_k = kf;
            day_volume += fabs(eq) * ep / balance;
            long_side.entry_qty = 0.0f;
        }

        bool short_close_fill = false;
        bool short_primary_market = short_side.close_is_panic
            ? short_hsl_panic_market : short_side.close_market;
        bool short_primary_close_fill = valid && alive && short_enabled
            && short_side.close_qty > 0.0f && short_side.psize > 0.0f
            && (short_primary_market
                || short_side.close_ticks > low_nonfill_max_tick);
        bool short_secondary_close_fill = valid && alive && short_enabled
            && short_side.secondary_close_qty > 0.0f && short_side.psize > 0.0f
            && (short_side.secondary_close_market
                || short_side.secondary_close_ticks > low_nonfill_max_tick);
        bool short_secondary_first = short_secondary_close_fill
            && (!short_primary_close_fill
                || short_side.secondary_close_ticks > short_side.close_ticks);
        for (int rank = 0; rank < 2; ++rank) {
            bool use_secondary = short_secondary_first ? rank == 0 : rank == 1;
            bool reachable = use_secondary
                ? short_secondary_close_fill : short_primary_close_fill;
            if (!reachable || short_side.psize <= 0.0f) continue;
            int close_ticks = use_secondary
                ? short_side.secondary_close_ticks : short_side.close_ticks;
            float requested_qty = use_secondary
                ? short_side.secondary_close_qty : short_side.close_qty;
            bool market_execution = use_secondary
                ? short_side.secondary_close_market : short_primary_market;
            float cp = market_execution
                ? ordinary_market_fill_price(
                    close, true, market_order_slippage_pct, price_step
                )
                : float(close_ticks) * price_step;
            float adj = fmin(round_step(requested_qty, qty_step), short_side.psize);
            if (!(adj > 0.0f)) continue;
            float pnl = adj * c_mult * (short_side.pprice - cp);
            float fee = adj * cp * c_mult
                * (market_execution ? taker_fee : maker_fee);
            float net_pnl = pnl - fee;
            if (!use_secondary && short_side.close_is_panic) {
                float current_equity = balance
                    + (long_side.psize > 0.0f
                        ? long_side.psize * c_mult * (close - long_side.pprice)
                        : 0.0f)
                    + short_side.psize * c_mult * (short_side.pprice - close);
                record_hsl_panic_fill(short_hsl, net_pnl, current_equity);
            }
            record_directional_gross_pnl(
                pnl, profit_sum, loss_sum, profit_sum_short, loss_sum_short
            );
            balance += net_pnl;
            record_realized_net(
                net_pnl, realized_pnl_cumsum_last, realized_pnl_cumsum_max,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                day_fill_count, fill_count, fill_count_entry, fill_count_long,
                pnl_recovery_peak, pnl_recovery_peak_k,
                pnl_recovery_max_min, kf,
                false, false
            );
            record_hsl_rolling_pnl(
                short_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                short_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, short_coin_hsl_rolling, net_pnl
            );
            float new_psize = fmax(round_step(short_side.psize - adj, qty_step), 0.0f);
            bool went_flat = new_psize <= 0.0f;
            short_side.psize = new_psize;
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
            day_volume += fabs(adj) * cp / balance;
            short_close_fill = true;
            if (use_secondary) {
                short_side.secondary_close_qty = 0.0f;
            } else {
                short_side.close_qty = 0.0f;
            }
        }

        bool short_entry_fill = valid && alive && short_enabled
            && short_side.entry_qty > 0.0f
            && (short_side.entry_market
                || short_side.entry_ticks <= high_fill_max_tick);
        if (short_entry_fill) {
            float ep = short_side.entry_market
                ? ordinary_market_fill_price(
                    close, false, market_order_slippage_pct, price_step
                )
                : float(short_side.entry_ticks) * price_step;
            float eq = round_step(short_side.entry_qty, qty_step);
            float fee = eq * ep * c_mult
                * (short_side.entry_market ? taker_fee : maker_fee);
            balance -= fee;
            record_realized_net(
                -fee, realized_pnl_cumsum_last, realized_pnl_cumsum_max,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                day_fill_count, fill_count, fill_count_entry, fill_count_long,
                pnl_recovery_peak, pnl_recovery_peak_k,
                pnl_recovery_max_min, kf,
                true, false
            );
            record_hsl_rolling_pnl(
                short_rolling_pnl, rolling_pnl_values, rolling_pnl_indices,
                short_rolling_base, rolling_capacity, int(kf),
                pnl_lookback_bars, short_coin_hsl_rolling, -fee
            );
            bool was_flat = short_side.psize <= 0.0f;
            float new_psize = round_step(short_side.psize + eq, qty_step);
            float new_pprice = was_flat ? ep
                : short_side.pprice * (short_side.psize / fmax(new_psize, 1.0e-12f))
                    + ep * (eq / fmax(new_psize, 1.0e-12f));
            if (was_flat) short_side.pos_open_k = kf;
            short_side.psize = new_psize;
            short_side.pprice = new_pprice;
            short_side.last_inc_k = kf;
            day_volume += fabs(eq) * ep / balance;
            short_side.entry_qty = 0.0f;
        }

        // Exact Rust generates this candle's bundles first, then force-closes and
        // clears both bundles.  Closing here and suppressing only that dead order
        // generation is equivalent while retaining `gen` for equity/HSL sampling.
        const bool forced_delist = valid && k == last_valid
            && last_valid + 1400 < T;
        bool forced_delist_closed_any = false;
        if (forced_delist && alive && balance > 0.0f) {
            float short_unrealized = short_side.psize > 0.0f
                ? short_side.psize * c_mult * (short_side.pprice - close)
                : 0.0f;
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
            float long_unrealized = long_side.psize > 0.0f
                ? long_side.psize * c_mult * (close - long_side.pprice)
                : 0.0f;
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
            long_close_fill = long_close_fill || forced_long_close;
            short_close_fill = short_close_fill || forced_short_close;
            forced_delist_closed_any = forced_long_close || forced_short_close;
            if (forced_delist_closed_any) {
                long_side.entry_qty = 0.0f;
                long_side.close_qty = 0.0f;
                long_side.secondary_close_qty = 0.0f;
                short_side.entry_qty = 0.0f;
                short_side.close_qty = 0.0f;
                short_side.secondary_close_qty = 0.0f;
            }
        }

        if (long_close_fill || long_entry_fill) {
            if (long_position_last_fill_k >= 0.0f) {
                position_unchanged_max_min = fmax(
                    position_unchanged_max_min, kf - long_position_last_fill_k
                );
            }
            long_position_last_fill_k = long_side.psize > 0.0f ? kf : -1.0f;
        }
        if (short_close_fill || short_entry_fill) {
            if (short_position_last_fill_k >= 0.0f) {
                position_unchanged_max_min = fmax(
                    position_unchanged_max_min, kf - short_position_last_fill_k
                );
            }
            short_position_last_fill_k = short_side.psize > 0.0f ? kf : -1.0f;
        }

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

        if (long_enabled) {
            update_indicators(long_side, close, log_range, hour_lr, valid, hour_valid);
        }
        if (short_enabled) {
            update_indicators(short_side, close, log_range, hour_lr, valid, hour_valid);
        }

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
                long_side.allowed_wel, long_side.base_qty_pct, max_effective_min_cost
            );
            bool short_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                short_side.allowed_wel, short_side.base_qty_pct, max_effective_min_cost
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
                    long_side.base_qty_pct, max_effective_min_cost
                );
                short_min_cost_eligible = passes_min_effective_cost(
                    true, guaranteed_balance_lower, short_side.allowed_wel,
                    short_side.base_qty_pct, max_effective_min_cost
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
                    float dist_long = long_lower * (1.0f - long_side.offset)
                        / close - 1.0f;
                    float dist_short = 1.0f
                        - short_upper * (1.0f + short_side.offset) / close;
                    if (dist_long >= dist_short) {
                        block_short_initial = true;
                    } else {
                        block_long_initial = true;
                    }
                }
            }
            if (long_enabled) {
                generate_long_orders(
                    long_side, balance, close, touch_down_tick, touch_up_tick,
                    qty_step, price_step, min_qty, min_cost, c_mult, kf,
                    block_long_initial, market_orders_allowed,
                    market_order_near_touch_threshold
                );
            }
            if (short_enabled) {
                generate_short_orders(
                    short_side, balance, close, touch_down_tick, touch_up_tick,
                    qty_step, price_step, min_qty, min_cost, c_mult, kf,
                    block_short_initial, market_orders_allowed,
                    market_order_near_touch_threshold
                );
            }
            if (filter_by_min_effective_cost && long_enabled && short_enabled) {
                min_cost_exact_open_uncertain = true;
            }

            bool loss_gate_enabled = max_realized_loss_pct < 1.0f;
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

            ReducerVariant long_twel = generated_reducer_variant(long_side);
            ReducerVariant short_twel = generated_reducer_variant(short_side);
            ReducerVariant long_unstuck = long_enabled
                ? unstuck_reducer_variant(
                    long_side, true, balance, balance_peak, close,
                    touch_down_tick, touch_up_tick, qty_step, price_step,
                    min_qty, min_cost, c_mult, market_orders_allowed,
                    market_order_near_touch_threshold
                )
                : empty_reducer_variant();
            ReducerVariant short_unstuck = short_enabled
                ? unstuck_reducer_variant(
                    short_side, false, balance, balance_peak, close,
                    touch_down_tick, touch_up_tick, qty_step, price_step,
                    min_qty, min_cost, c_mult, market_orders_allowed,
                    market_order_near_touch_threshold
                )
                : empty_reducer_variant();

            // Exact Rust emits at most one global unstuck intent: the eligible
            // long/short position with the lowest pside-aware price distance.
            if (long_unstuck.valid && short_unstuck.valid) {
                float long_diff = 1.0f - close / long_side.pprice;
                float short_diff = close / short_side.pprice - 1.0f;
                if (long_diff <= short_diff) {
                    short_unstuck = empty_reducer_variant();
                } else {
                    long_unstuck = empty_reducer_variant();
                }
            }
            if (long_enabled && long_hsl_mode == 3) {
                long_twel = panic_reducer_variant(
                    long_side, true, touch_down_tick, touch_up_tick
                );
                long_unstuck = empty_reducer_variant();
            }
            if (short_enabled && short_hsl_mode == 3) {
                short_twel = panic_reducer_variant(
                    short_side, false, touch_down_tick, touch_up_tick
                );
                short_unstuck = empty_reducer_variant();
            }

            ReducerVariant long_preferred, long_fallback;
            ReducerVariant short_preferred, short_fallback;
            order_reducer_variants(
                long_twel, long_unstuck, true,
                long_preferred, long_fallback
            );
            order_reducer_variants(
                short_twel, short_unstuck, false,
                short_preferred, short_fallback
            );
            restore_ordinary_close(long_side);
            restore_ordinary_close(short_side);
            if (long_enabled) {
                prepare_ordinary_market_close(
                    long_side, true, close, market_orders_allowed,
                    market_order_near_touch_threshold, qty_step, price_step,
                    min_qty, min_cost, c_mult
                );
            }
            if (short_enabled) {
                prepare_ordinary_market_close(
                    short_side, false, close, market_orders_allowed,
                    market_order_near_touch_threshold, qty_step, price_step,
                    min_qty, min_cost, c_mult
                );
            }

            ReducerVariant selected_long = empty_reducer_variant();
            ReducerVariant selected_short = empty_reducer_variant();
            int long_candidate_index = 0;
            int short_candidate_index = 0;
            bool long_resolved = !long_preferred.valid;
            bool short_resolved = !short_preferred.valid;
            // Final reducer sizes compete globally largest-first. A blocked
            // candidate advances only its own position to the next reducer.
            for (int attempt = 0; attempt < 4; ++attempt) {
                ReducerVariant long_candidate = reducer_variant_at(
                    long_preferred, long_fallback, long_candidate_index
                );
                ReducerVariant short_candidate = reducer_variant_at(
                    short_preferred, short_fallback, short_candidate_index
                );
                bool long_available = !long_resolved && long_candidate.valid;
                bool short_available = !short_resolved && short_candidate.valid;
                if (!long_available && !short_available) break;
                bool use_long = long_available && (
                    !short_available || global_reducer_prefers_long(
                        long_candidate, short_candidate
                    )
                );
                if (use_long) {
                    if (gate_reducer_variant(
                            long_candidate, long_side.pprice, true,
                            price_step, c_mult, maker_fee, taker_fee,
                            market_order_slippage_pct, close, loss_gate_enabled,
                            remaining_loss_budget)) {
                        selected_long = long_candidate;
                        long_resolved = true;
                    } else {
                        long_candidate_index += 1;
                        long_resolved = long_candidate_index > 1
                            || !long_fallback.valid;
                    }
                } else {
                    if (gate_reducer_variant(
                            short_candidate, short_side.pprice, false,
                            price_step, c_mult, maker_fee, taker_fee,
                            market_order_slippage_pct, close, loss_gate_enabled,
                            remaining_loss_budget)) {
                        selected_short = short_candidate;
                        short_resolved = true;
                    } else {
                        short_candidate_index += 1;
                        short_resolved = short_candidate_index > 1
                            || !short_fallback.valid;
                    }
                }
            }
            if (selected_long.valid) {
                apply_reducer_variant(long_side, selected_long);
            }
            if (selected_short.valid) {
                apply_reducer_variant(short_side, selected_short);
            }

            // Ordinary closes consume any remaining allowance in canonical
            // long-then-short order after protective reducers are finalized.
            if (long_enabled) {
                if (long_side.close_is_protective_reducer) {
                    gate_generated_close(
                        long_side.secondary_close_qty,
                        long_side.secondary_close_ticks, long_side.pprice, true,
                        price_step, c_mult, maker_fee, taker_fee,
                        market_order_slippage_pct, close,
                        long_side.secondary_close_market, loss_gate_enabled,
                        remaining_loss_budget
                    );
                } else {
                    gate_generated_close(
                        long_side.close_qty, long_side.close_ticks,
                        long_side.pprice, true, price_step, c_mult, maker_fee,
                        taker_fee, market_order_slippage_pct, close,
                        long_side.close_market,
                        loss_gate_enabled, remaining_loss_budget
                    );
                }
            }
            if (short_enabled) {
                if (short_side.close_is_protective_reducer) {
                    gate_generated_close(
                        short_side.secondary_close_qty,
                        short_side.secondary_close_ticks, short_side.pprice,
                        false, price_step, c_mult, maker_fee, taker_fee,
                        market_order_slippage_pct, close,
                        short_side.secondary_close_market, loss_gate_enabled,
                        remaining_loss_budget
                    );
                } else {
                    gate_generated_close(
                        short_side.close_qty, short_side.close_ticks,
                        short_side.pprice, false, price_step, c_mult, maker_fee,
                        taker_fee, market_order_slippage_pct, close,
                        short_side.close_market,
                        loss_gate_enabled, remaining_loss_budget
                    );
                }
            }

            if (long_enabled && long_hsl_mode >= 2) {
                long_side.entry_qty = 0.0f;
            }
            if (short_enabled && short_hsl_mode >= 2) {
                short_side.entry_qty = 0.0f;
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
            recovery_samples[int(b) * recovery_sample_count]
                = RECOVERY_FAIL_CLOSED_SENTINEL;
#endif
            balance = 0.0f;
            alive = false;
            liq_day = di;
        }
        const bool hsl_step = gen || (eq_started && after_valid_tail);
        if (hsl_step && alive && balance > 0.0f && equity > liq_floor) {
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
            if (hsl_update_valid && (long_hsl.enabled || short_hsl.enabled)) {
                int hsl_tier = max(long_hsl.tier, short_hsl.tier);
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow += hsl_tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange += hsl_tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red += hsl_tier == 3 ? 1.0f : 0.0f;
            }
            if (hsl_update_valid) {
                try_restart_hsl(long_hsl, kf, equity);
                try_restart_hsl(short_hsl, kf, equity);
            }
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
                recovery_samples[int(b) * recovery_sample_count] = eqf;
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
                            int(b) * recovery_sample_count + sample_index
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
    scalars[so + 43] = profit_sum;
    scalars[so + 44] = loss_sum;
    scalars[so + 45] = position_unchanged_max_min * interval_ms;
    scalars[so + 46] = long_enabled
        ? long_side.allowed_wel * long_side.base_qty_pct : 0.0f;
    scalars[so + 47] = short_enabled
        ? short_side.allowed_wel * short_side.base_qty_pct : 0.0f;
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
    scalars[so + 62] = long_hsl.enabled ? long_hsl.drawdown_ema_max : 0.0f;
    scalars[so + 63] = short_hsl.enabled ? short_hsl.drawdown_ema_max : 0.0f;
    scalars[so + 64] = hsl_strategy_equity_recovery_max_steps(
        long_hsl_strategy_eq
    ) * interval_ms;
    scalars[so + 65] = hsl_strategy_equity_recovery_max_steps(
        short_hsl_strategy_eq
    ) * interval_ms;
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

kernel void passivbot_ema_anchor(
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
        daily, scalars, gap_hist,
        rolling_pnl_values, rolling_pnl_indices,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b
    );
}
