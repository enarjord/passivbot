#include <metal_stdlib>
using namespace metal;

constant int DAILY_COLS = 8;
constant int SCALAR_COLS = 62;
constant int GAP_BINS = 128;
constant int SIDE_PARAMS = 51;

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
    float c_mult, float maker_fee, bool gate_enabled
) {
    if (!gate_enabled) return true;
    if (!(qty > 0.0f && close_price > 0.0f && pprice > 0.0f)) return false;
    float gross_pnl = qty * c_mult
        * (is_long ? close_price - pprice : pprice - close_price);
    float fee = qty * close_price * c_mult * maker_fee;
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
    float maker_fee,
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
            c_mult, maker_fee, true
        );
    }
    if (!(qty > 0.0f && close_price > 0.0f && pprice > 0.0f)) return false;
    float gross_pnl = qty * c_mult
        * (is_long ? close_price - pprice : pprice - close_price);
    float fee = qty * close_price * c_mult * maker_fee;
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
    float alpha0, alpha1, alpha2, alpha1m, alpha1h;
    float ddf, initial_ema_dist, initial_qty_pct;
    float entry_threshold_base, entry_threshold_we;
    float entry_threshold_v1h, entry_threshold_v1m;
    float entry_retracement_base, entry_retracement_we;
    float entry_retracement_v1h, entry_retracement_v1m;
    float close_qty_pct, close_threshold_base, close_threshold_we;
    float close_threshold_v1h, close_threshold_v1m;
    float close_retracement_base, close_retracement_v1h, close_retracement_v1m;
    float cooldown_min, twel, allowed_wel, entry_cap;
    bool gate_initial, gate_reentry;
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
    float ema0, ema1, ema2, vol1m, vol1h;
    float psize, pprice, last_inc_k, pos_open_k;
    int entry_ticks, close_ticks, secondary_close_ticks;
    float entry_price, close_price, entry_qty, close_qty;
    float secondary_close_price, secondary_close_qty;
    float entry_gen_balance, entry_gen_psize, entry_gen_pprice, entry_gen_kf;
    int entry_gen_touch_ticks;
    float close_gen_balance, close_gen_psize, close_gen_pprice;
    int close_gen_touch_down_ticks, close_gen_touch_up_ticks;
    int close_gen_touch_nearest_ticks;
    float close_gen_touch_min_qty;
    int close_gen_touch_min_qty_relation;
    float min_since_open, max_since_min, max_since_open, min_since_max;
};

struct CloseGroup {
    int ticks;
    float price;
    float qty;
};

// PASSIVBOT_HSL_COMMON

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
    s.alpha1h = p[o + 2] > 0.0f ? 2.0f / (fmax(p[o + 2], 1.0f) + 1.0f) : 0.0f;
    s.alpha1m = p[o + 3] > 0.0f ? clamp(2.0f / (p[o + 3] + 1.0f), 0.0f, 1.0f) : 0.0f;
    s.ddf = p[o + 4];
    s.initial_ema_dist = p[o + 5];
    s.initial_qty_pct = p[o + 6];
    s.entry_threshold_base = p[o + 7];
    s.entry_threshold_we = p[o + 8];
    s.entry_threshold_v1h = p[o + 9];
    s.entry_threshold_v1m = p[o + 10];
    s.entry_retracement_base = p[o + 11];
    s.entry_retracement_we = p[o + 12];
    s.entry_retracement_v1h = p[o + 13];
    s.entry_retracement_v1m = p[o + 14];
    s.close_qty_pct = p[o + 15];
    s.close_threshold_base = p[o + 16];
    s.close_threshold_we = p[o + 17];
    s.close_threshold_v1h = p[o + 18];
    s.close_threshold_v1m = p[o + 19];
    s.close_retracement_base = p[o + 20];
    s.close_retracement_v1h = p[o + 21];
    s.close_retracement_v1m = p[o + 22];
    s.cooldown_min = ceil(p[o + 23]);
    s.twel = p[o + 24];
    s.gate_initial = p[o + 25] > 0.5f;
    s.gate_reentry = p[o + 26] > 0.5f;
    float allowance_pct = fmax(p[o + 27], 0.0f);
    bool legacy_raw_allowance = p[o + 28] > 0.5f;
    s.allowed_wel = s.twel * (
        1.0f + (legacy_raw_allowance ? allowance_pct : 0.0f)
    );
    bool twel_entry_gate_enabled = p[o + 29] > 0.5f;
    float twel_threshold = p[o + 30];
    float gate_cap = s.twel;
    if (isfinite(twel_threshold) && twel_threshold > 0.0f) {
        gate_cap = fmin(s.twel, s.twel * twel_threshold);
    }
    s.entry_cap = twel_entry_gate_enabled
        ? fmin(s.allowed_wel, gate_cap) : s.allowed_wel;
    s.wel_enforcer_enabled = p[o + 31] > 0.5f;
    s.wel_enforcer_threshold = p[o + 32];
    s.twel_enforcer_enabled = p[o + 33] > 0.5f;
    s.twel_enforcer_threshold = twel_threshold;
    s.unstuck_enabled = p[o + 34] > 0.5f;
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
    s.ema0 = seed; s.ema1 = seed; s.ema2 = seed;
    s.vol1m = 0.0f; s.vol1h = 0.0f;
    s.psize = 0.0f; s.pprice = 0.0f;
    s.last_inc_k = -1.0f; s.pos_open_k = -1.0f;
    s.entry_ticks = 0; s.entry_qty = 0.0f;
    s.close_ticks = 0; s.close_qty = 0.0f;
    s.entry_price = 0.0f; s.close_price = 0.0f;
    s.secondary_close_ticks = 0; s.secondary_close_qty = 0.0f;
    s.secondary_close_price = 0.0f;
    s.entry_gen_balance = 0.0f;
    s.entry_gen_psize = 0.0f; s.entry_gen_pprice = 0.0f;
    s.entry_gen_kf = -1.0f; s.entry_gen_touch_ticks = 0;
    s.close_gen_balance = 0.0f;
    s.close_gen_psize = 0.0f; s.close_gen_pprice = 0.0f;
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
    if (hour_valid && s.alpha1h > 0.0f)
        s.vol1h = fma(s.alpha1h, hour_lr - s.vol1h, s.vol1h);
    if (valid) {
        s.ema0 = fma(s.alpha0, close - s.ema0, s.ema0);
        s.ema1 = fma(s.alpha1, close - s.ema1, s.ema1);
        s.ema2 = fma(s.alpha2, close - s.ema2, s.ema2);
        if (s.alpha1m > 0.0f)
            s.vol1m = fma(s.alpha1m, lr - s.vol1m, s.vol1m);
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
    if (we_if <= s.entry_cap * 1.01f) return qty;
    float q = round_step(
        (s.entry_cap * balance - cost) / fmax(price * c_mult, 1.0e-12f), qty_step
    );
    float mq = min_entry_qty(price, qty_step, min_qty, min_cost, c_mult);
    q = fmax(q, mq);
    return q < qty ? q : qty;
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
    s.close_gen_balance = balance;
    s.close_gen_psize = s.psize;
    s.close_gen_pprice = s.pprice;
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
    float tm = fmax(
        1.0f + s.vol1h * s.entry_threshold_v1h
            + s.vol1m * s.entry_threshold_v1m + wer * s.entry_threshold_we,
        1.0f
    );
    float rm = fmax(
        1.0f + s.vol1h * s.entry_retracement_v1h
            + s.vol1m * s.entry_retracement_v1m + wer * s.entry_retracement_we,
        1.0f
    );
    float threshold = fmax(s.entry_threshold_base, 0.0f) * tm;
    float retracement = fmax(s.entry_retracement_base, 0.0f) * rm;
    bool trailing_entry = s.entry_retracement_base > 0.0f;
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
    float crop_fraction = (s.entry_cap - we) / fmax(we_if - we, 1.0e-12f);
    float rq_crop = fmax(round_step(rq * crop_fraction, qty_step), min_rq);
    if (we_if > s.entry_cap * 1.01f && rq_crop < rq) rq = rq_crop;
    bool cap_hit = trailing_entry
        ? we > s.entry_cap * 0.999f : we >= s.entry_cap * 0.999f;
    bool reentry_ok = !flat && !partial && !cap_hit && reentry_ticks > 1
        && (!trailing_entry || entry_triggered);
    float eqty = flat ? iq : (partial ? iq_partial : (reentry_ok ? rq : 0.0f));
    int eticks = flat || partial ? initial_ticks : reentry_ticks;
    float eprice = flat || partial ? initial_price : reentry_price;
    bool cooldown = s.cooldown_min > 0.0f && s.last_inc_k >= 0.0f
        && kf < s.last_inc_k + s.cooldown_min;
    if (cooldown || balance <= 0.0f || s.initial_qty_pct <= 0.0f
        || s.allowed_wel <= 0.0f || s.entry_cap <= 0.0f || eticks <= 1
        || (block_initial && flat)) eqty = 0.0f;
    s.entry_ticks = eticks;
    s.entry_price = eprice;
    s.entry_qty = crop_entry(
        s, balance, eprice, eqty,
        qty_step, min_qty, min_cost, c_mult
    );

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

    int unstuck_ticks = is_long ? touch_up_ticks : touch_down_ticks;
    float unstuck_price = float(unstuck_ticks) * price_step;
    float unstuck_qty = s.unstuck_enabled
        ? unstuck_reducer_qty(
            s, is_long, balance, unstuck_price,
            qty_step, min_qty, min_cost, c_mult
        )
        : 0.0f;

    float ct = s.close_threshold_base + wer * s.close_threshold_we
        + s.vol1h * s.close_threshold_v1h + s.vol1m * s.close_threshold_v1m;
    float cr = fmax(s.close_retracement_base, 0.0f) * fmax(
        1.0f + s.vol1h * s.close_retracement_v1h
            + s.vol1m * s.close_retracement_v1m,
        1.0f
    );
    bool trailing_close = s.close_retracement_base > 0.0f;
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
    bool ordinary_can_accompany_reducer = wel_qty <= 0.0f
        && trailing_close && s.close_qty > 0.0f;
    float finalized_wel_qty = finalized_reducer_qty(
        s.psize, wel_qty, wel_price,
        qty_step, min_qty, min_cost, c_mult
    );
    float finalized_twel_qty = ordinary_can_accompany_reducer
        ? finalized_reducer_qty_with_ordinary(
            s.psize, twel_qty, twel_price, s.close_qty, close_mq,
            close_mq_relation, qty_step, min_qty, min_cost, c_mult
        )
        : finalized_reducer_qty(
            s.psize, twel_qty, twel_price,
            qty_step, min_qty, min_cost, c_mult
        );
    float finalized_unstuck_qty = ordinary_can_accompany_reducer
        ? finalized_reducer_qty_with_ordinary(
            s.psize, unstuck_qty, unstuck_price, s.close_qty, close_mq,
            close_mq_relation, qty_step, min_qty, min_cost, c_mult
        )
        : finalized_reducer_qty(
            s.psize, unstuck_qty, unstuck_price,
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
    if (reducer_qty > 0.0f && reducer_ticks > 0) {
        float reducer_min = min_entry_qty(
            reducer_price, qty_step, min_qty, min_cost, c_mult
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
            bool ordinary_below_minimum = ordinary_qty < close_mq
                || (ordinary_qty == close_mq && close_mq_relation > 0);
            if (!ordinary_below_minimum) {
                float remainder = fmax(
                    round_step(
                        s.psize - reducer_qty - ordinary_qty, qty_step
                    ),
                    0.0f
                );
                float minimum_any = fmin(close_mq, reducer_min);
                if (remainder > 0.0f && remainder < minimum_any) {
                    ordinary_qty = fmin(
                        s.psize - reducer_qty,
                        round_step(ordinary_qty + remainder, qty_step)
                    );
                }
                s.secondary_close_ticks = s.close_ticks;
                s.secondary_close_price = s.close_price;
                s.secondary_close_qty = ordinary_qty;
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
    int max_rungs,
    thread CloseGroup& selected
) {
    selected.ticks = 0;
    selected.price = 0.0f;
    selected.qty = 0.0f;
    TmSide sim = source;
    sim.psize = source.close_gen_psize;
    sim.pprice = source.close_gen_pprice;
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
                selected.qty = group_qty;
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
            selected.qty = group_qty;
        }
        ++group_count;
    }
    return group_count;
}

inline void passivbot_single_coin_impl(
    constant float* bars,
    constant int* flags,
    constant float* params,
    constant float* settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b
) {
    const int B = sizes[0];
    const int T = sizes[1];
    const int D = sizes[2];
    const int P = sizes[3];
    const int first_valid = sizes[4];
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
    const bool long_enabled = settings[9] > 0.5f;
    const bool short_enabled = settings[10] > 0.5f;
    const bool hedge_mode = settings[11] > 0.5f;
    const bool filter_by_min_effective_cost = settings[12] > 0.5f;
    const float max_effective_min_cost = settings[13];
    const float max_realized_loss_pct = settings[14];
    const float taker_fee = settings[15];
    const float market_order_slippage_pct = fmax(settings[16], 0.0f);
    const bool long_hsl_panic_market = settings[17] > 0.5f;
    const bool short_hsl_panic_market = settings[18] > 0.5f;
    const bool loss_gate_enabled = max_realized_loss_pct < 1.0f;
    const float log_bin_scale = 127.0f / log(4000001.0f);

    const int po = int(b) * P;
    const int seed_k = clamp(first_valid, 0, T - 1);
    const float seed_close = bars[seed_k * 5 + 2];
    TmSide long_side = load_side(params, po, seed_close);
    TmSide short_side = load_side(params, po + SIDE_PARAMS, seed_close);
    HslState long_hsl = load_hsl(params, po, 40);
    HslState short_hsl = load_hsl(params, po + SIDE_PARAMS, 40);
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
        const int touch_nearest_tick = flags[fo + 8];
        const float touch_min_qty = as_type<float>(flags[fo + 9]);
        const int touch_min_qty_relation = flags[fo + 10];
        const float kf = float(k);

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
        }

        bool long_close_fill = false;
        bool long_close_ready = valid && alive && long_enabled
            && long_side.close_qty > 0.0f && long_side.psize > 0.0f;
        bool long_secondary_close_fill = valid && alive && long_enabled
            && long_side.secondary_close_qty > 0.0f
            && long_side.secondary_close_ticks <= high_fill_max_tick
            && long_side.psize > 0.0f;
        bool long_recursive_close = long_side.close_retracement_base <= 0.0f;
        bool long_scan_close_grid = long_close_ready
            && ((long_side.close_is_panic && long_hsl_panic_market)
                || long_side.close_ticks <= high_fill_max_tick);
        if (long_close_ready && long_recursive_close
            && long_side.close_is_exposure_reducer) {
            // A touch-clamped grid order uses nearest-tick quantization while
            // the passive reducer rounds up.  The grid may therefore be one
            // tick nearer and independently reachable.
            long_scan_close_grid = long_scan_close_grid
                || long_side.close_gen_touch_nearest_ticks <= high_fill_max_tick;
        }
        if (long_close_ready && long_recursive_close
            && long_side.close_threshold_we > 0.0f) {
            // Positive WE weight makes later generated long closes nearer.
            // The zero-WE target is a conservative lower bound: reconstruct
            // the sorted grid only if this candle can reach that bound.
            float threshold_floor = long_side.close_threshold_base
                + long_side.vol1h * long_side.close_threshold_v1h
                + long_side.vol1m * long_side.close_threshold_v1m;
            int target_ticks = directional_ticks(
                long_side.close_gen_pprice * (1.0f + threshold_floor),
                price_step, true
            );
            bool touch_controls = long_side.close_gen_touch_up_ticks > target_ticks;
            int nearest_ticks = touch_controls
                ? long_side.close_gen_touch_nearest_ticks : target_ticks;
            long_scan_close_grid = long_scan_close_grid
                || nearest_ticks <= high_fill_max_tick;
        }
        if (long_scan_close_grid && long_recursive_close
            && !long_side.close_is_panic) {
            TmSide grid_source = long_side;
            if (long_side.close_loss_gate_disabled_reducers) {
                grid_source.wel_enforcer_enabled = false;
                grid_source.twel_enforcer_enabled = false;
                grid_source.unstuck_enabled = false;
            }
            float reducer_qty = 0.0f;
            int reducer_ticks = 0;
            float reducer_price = 0.0f;
            float strategy_wel_qty = 0.0f;
            if (long_side.close_is_exposure_reducer) {
                reducer_qty = fmin(
                    round_step(long_side.close_qty, qty_step), long_side.psize
                );
                reducer_ticks = long_side.close_ticks;
                reducer_price = long_side.close_price;
                strategy_wel_qty = reducer_qty;
                if (long_side.close_is_twel_reducer
                    || long_side.close_is_unstuck_reducer) {
                    float wel_price = float(
                        long_side.close_gen_touch_up_ticks
                    ) * price_step;
                    strategy_wel_qty = long_side.wel_enforcer_enabled
                        ? exposure_reducer_qty(
                            long_side.close_gen_psize,
                            long_side.close_gen_pprice,
                            long_side.close_gen_balance,
                            long_side.allowed_wel
                                * long_side.wel_enforcer_threshold,
                            wel_price, qty_step, min_qty, min_cost, c_mult
                        ) : 0.0f;
                }
                grid_source.close_gen_psize = fmax(
                    round_step(
                        long_side.close_gen_psize - strategy_wel_qty, qty_step
                    ),
                    0.0f
                );
                grid_source.wel_enforcer_enabled = false;
                grid_source.twel_enforcer_enabled = false;
                grid_source.unstuck_enabled = false;
            }
            CloseGroup group;
            int grid_rung_limit = strategy_wel_qty > 0.0f ? 499 : 500;
            int group_count = recursive_close_groups(
                grid_source, true, -1, qty_step, price_step,
                min_qty, min_cost, c_mult, grid_rung_limit, group
            );
            bool reverse = grid_source.close_threshold_we > 0.0f;
            float ordinary_budget = fmax(
                round_step(
                    long_side.close_gen_psize - reducer_qty, qty_step
                ),
                0.0f
            );
            bool trim_for_reducer = long_side.close_is_exposure_reducer
                && reducer_qty > 0.0f;
            float remaining_budget = ordinary_budget;
            float kept_ordinary = 0.0f;
            float minimum_any = trim_for_reducer
                ? min_entry_qty(
                    reducer_price, qty_step, min_qty, min_cost, c_mult
                ) : 0.0f;
            int last_kept_rank = -1;
            for (int trim_rank = 0;
                trim_for_reducer && trim_rank < group_count;
                ++trim_rank) {
                int wanted = reverse
                    ? group_count - trim_rank - 1 : trim_rank;
                recursive_close_groups(
                    grid_source, true, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, grid_rung_limit, group
                );
                float trimmed_qty = fmin(group.qty, remaining_budget);
                float group_min = min_entry_qty(
                    group.price, qty_step, min_qty, min_cost, c_mult
                );
                bool partial_trim = trimmed_qty + 1.0e-6f < group.qty;
                if (trimmed_qty + 1.0e-6f < group_min) {
                    trimmed_qty = 0.0f;
                    if (partial_trim) remaining_budget = 0.0f;
                }
                if (trimmed_qty > 0.0f) {
                    kept_ordinary += trimmed_qty;
                    remaining_budget = fmax(
                        round_step(remaining_budget - trimmed_qty, qty_step),
                        0.0f
                    );
                    minimum_any = fmin(minimum_any, group_min);
                    last_kept_rank = trim_rank;
                }
            }
            float dust_remainder = fmax(
                round_step(
                    long_side.close_gen_psize - reducer_qty - kept_ordinary,
                    qty_step
                ),
                0.0f
            );
            if (dust_remainder > 0.0f && dust_remainder < minimum_any
                && last_kept_rank < 0) {
                reducer_qty = fmin(
                    long_side.close_gen_psize,
                    round_step(reducer_qty + dust_remainder, qty_step)
                );
                dust_remainder = 0.0f;
            }
            bool reducer_reachable = reducer_qty > 0.0f
                && reducer_ticks <= high_fill_max_tick;
            bool reducer_executed = false;
            remaining_budget = ordinary_budget;
            for (int rank = 0; rank < group_count; ++rank) {
                int wanted = reverse ? group_count - rank - 1 : rank;
                recursive_close_groups(
                    grid_source, true, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, grid_rung_limit, group
                );
                if (group.qty <= 0.0f) break;
                float trimmed_group_qty = group.qty;
                if (trim_for_reducer) {
                    float group_min = min_entry_qty(
                        group.price, qty_step, min_qty, min_cost, c_mult
                    );
                    trimmed_group_qty = fmin(group.qty, remaining_budget);
                    bool partial_trim = trimmed_group_qty + 1.0e-6f
                        < group.qty;
                    if (trimmed_group_qty + 1.0e-6f < group_min) {
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
                    float pnl = reducer_qty * c_mult
                        * (reducer_price - long_side.pprice);
                    float fee = reducer_qty * reducer_price * c_mult * maker_fee;
                    if (!realized_loss_proxy_allows_reducer(
                            reducer_qty, reducer_price, long_side.pprice, true,
                            long_side.close_is_unstuck_reducer,
                            c_mult, maker_fee, loss_gate_enabled, balance,
                            realized_pnl_cumsum_last,
                            realized_pnl_cumsum_max,
                            max_realized_loss_pct)) {
                        reducer_reachable = false;
                    } else {
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
                        long_side.psize = fmax(
                            round_step(
                                long_side.psize - reducer_qty, qty_step
                            ),
                            0.0f
                        );
                        day_volume += reducer_qty * reducer_price / balance;
                        long_close_fill = true;
                        reducer_executed = true;
                    }
                }
                if (group.ticks > high_fill_max_tick) break;
                float group_qty = trimmed_group_qty;
                if (group_qty <= 0.0f) continue;
                float adj = fmin(round_step(group_qty, qty_step), long_side.psize);
                float pnl = adj * c_mult * (group.price - long_side.pprice);
                float fee = adj * group.price * c_mult * maker_fee;
                if (!realized_loss_proxy_allows_close(
                        adj, group.price, long_side.pprice, true,
                        c_mult, maker_fee, loss_gate_enabled)) continue;
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
                float new_psize = fmax(
                    round_step(long_side.psize - adj, qty_step), 0.0f
                );
                bool went_flat = new_psize <= 0.0f;
                long_side.psize = new_psize;
                day_volume += fabs(adj) * group.price / balance;
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
                float pnl = adj * c_mult
                    * (reducer_price - long_side.pprice);
                float fee = adj * reducer_price * c_mult * maker_fee;
                if (realized_loss_proxy_allows_reducer(
                        adj, reducer_price, long_side.pprice, true,
                        long_side.close_is_unstuck_reducer,
                        c_mult, maker_fee, loss_gate_enabled, balance,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        max_realized_loss_pct)) {
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
                    long_side.psize = fmax(
                        round_step(long_side.psize - adj, qty_step), 0.0f
                    );
                    day_volume += adj * reducer_price / balance;
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
        } else if (long_scan_close_grid || long_secondary_close_fill) {
            bool secondary_first = long_secondary_close_fill
                && (!long_scan_close_grid
                    || long_side.secondary_close_price
                        <= long_side.close_price);
            for (int rank = 0; rank < 2; ++rank) {
                bool use_secondary = secondary_first ? rank == 0 : rank == 1;
                bool reachable = use_secondary
                    ? long_secondary_close_fill : long_scan_close_grid;
                if (!reachable || long_side.psize <= 0.0f) continue;
                bool market_panic = !use_secondary && long_side.close_is_panic
                    && long_hsl_panic_market;
                float cp = use_secondary ? long_side.secondary_close_price
                    : market_panic
                        ? float(max(directional_ticks(
                            close * (1.0f - market_order_slippage_pct),
                            price_step, false
                        ), 1)) * price_step
                        : long_side.close_price;
                float requested_qty = use_secondary
                    ? long_side.secondary_close_qty : long_side.close_qty;
                float adj = fmin(
                    round_step(requested_qty, qty_step), long_side.psize
                );
                float pnl = adj * c_mult * (cp - long_side.pprice);
                float fee = adj * cp * c_mult
                    * (market_panic ? taker_fee : maker_fee);
                bool selected_unstuck = !use_secondary
                    && long_side.close_is_unstuck_reducer;
                if (!long_side.close_is_panic
                    && !realized_loss_proxy_allows_reducer(
                        adj, cp, long_side.pprice, true, selected_unstuck,
                        c_mult, maker_fee, loss_gate_enabled, balance,
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

        bool long_entry_fill = valid && alive && long_enabled
            && long_side.entry_qty > 0.0f
            && long_side.entry_ticks > low_nonfill_max_tick;
        if (long_entry_fill) {
            TmSide ladder_side = long_side;
            ladder_side.psize = long_side.entry_gen_psize;
            ladder_side.pprice = long_side.entry_gen_pprice;
            const float ladder_balance = long_side.entry_gen_balance;
            const float ladder_kf = long_side.entry_gen_kf;
            int ladder_touch_ticks = long_side.entry_gen_touch_ticks;
            int previous_ticks = 0;
            for (int rung = 0; rung < 500; ++rung) {
                int entry_ticks = rung == 0
                    ? long_side.entry_ticks : ladder_side.entry_ticks;
                float ep = rung == 0
                    ? long_side.entry_price : ladder_side.entry_price;
                float eq = round_step(
                    rung == 0 ? long_side.entry_qty : ladder_side.entry_qty,
                    qty_step
                );
                if (eq <= 0.0f || entry_ticks <= low_nonfill_max_tick
                    || (rung > 0 && entry_ticks == previous_ticks)) break;

                float fee = eq * ep * c_mult * maker_fee;
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

                bool sim_flat = ladder_side.psize <= 0.0f;
                float sim_psize = round_step(ladder_side.psize + eq, qty_step);
                ladder_side.pprice = sim_flat ? ep
                    : ladder_side.pprice
                        * (ladder_side.psize / fmax(sim_psize, 1.0e-12f))
                        + ep * (eq / fmax(sim_psize, 1.0e-12f));
                ladder_side.psize = sim_psize;
                previous_ticks = entry_ticks;
                ladder_touch_ticks = min(ladder_touch_ticks, entry_ticks);
                if (long_side.entry_retracement_base > 0.0f
                    || long_side.cooldown_min != 0.0f) break;
                generate_long_orders(
                    ladder_side, ladder_balance, ep, qty_step,
                    ladder_touch_ticks, ladder_touch_ticks, ladder_touch_ticks,
                    touch_min_qty, touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, ladder_kf, false
                );
            }
            long_side.entry_qty = 0.0f;
        }

        bool short_close_fill = false;
        bool short_close_ready = valid && alive && short_enabled
            && short_side.close_qty > 0.0f && short_side.psize > 0.0f;
        bool short_secondary_close_fill = valid && alive && short_enabled
            && short_side.secondary_close_qty > 0.0f
            && short_side.secondary_close_ticks > low_nonfill_max_tick
            && short_side.psize > 0.0f;
        bool short_recursive_close = short_side.close_retracement_base <= 0.0f;
        bool short_scan_close_grid = short_close_ready
            && ((short_side.close_is_panic && short_hsl_panic_market)
                || short_side.close_ticks > low_nonfill_max_tick);
        if (short_close_ready && short_recursive_close
            && short_side.close_is_exposure_reducer) {
            // Mirror the long-side nearest-tick scan: a touch-clamped grid
            // order can sit one tick above the down-rounded reducer.
            short_scan_close_grid = short_scan_close_grid
                || short_side.close_gen_touch_nearest_ticks > low_nonfill_max_tick;
        }
        if (short_close_ready && short_recursive_close
            && short_side.close_threshold_we > 0.0f) {
            // Positive WE weight makes later generated short closes nearer.
            // The zero-WE target is a conservative upper bound.
            float threshold_floor = short_side.close_threshold_base
                + short_side.vol1h * short_side.close_threshold_v1h
                + short_side.vol1m * short_side.close_threshold_v1m;
            int target_ticks = directional_ticks(
                short_side.close_gen_pprice * (1.0f - threshold_floor),
                price_step, false
            );
            bool touch_controls = short_side.close_gen_touch_down_ticks < target_ticks;
            int nearest_ticks = touch_controls
                ? short_side.close_gen_touch_nearest_ticks : target_ticks;
            short_scan_close_grid = short_scan_close_grid
                || nearest_ticks > low_nonfill_max_tick;
        }
        if (short_scan_close_grid && short_recursive_close
            && !short_side.close_is_panic) {
            TmSide grid_source = short_side;
            if (short_side.close_loss_gate_disabled_reducers) {
                grid_source.wel_enforcer_enabled = false;
                grid_source.twel_enforcer_enabled = false;
                grid_source.unstuck_enabled = false;
            }
            float reducer_qty = 0.0f;
            int reducer_ticks = 0;
            float reducer_price = 0.0f;
            float strategy_wel_qty = 0.0f;
            if (short_side.close_is_exposure_reducer) {
                reducer_qty = fmin(
                    round_step(short_side.close_qty, qty_step), short_side.psize
                );
                reducer_ticks = short_side.close_ticks;
                reducer_price = short_side.close_price;
                strategy_wel_qty = reducer_qty;
                if (short_side.close_is_twel_reducer
                    || short_side.close_is_unstuck_reducer) {
                    float wel_price = float(
                        short_side.close_gen_touch_down_ticks
                    ) * price_step;
                    strategy_wel_qty = short_side.wel_enforcer_enabled
                        ? exposure_reducer_qty(
                            short_side.close_gen_psize,
                            short_side.close_gen_pprice,
                            short_side.close_gen_balance,
                            short_side.allowed_wel
                                * short_side.wel_enforcer_threshold,
                            wel_price, qty_step, min_qty, min_cost, c_mult
                        ) : 0.0f;
                }
                grid_source.close_gen_psize = fmax(
                    round_step(
                        short_side.close_gen_psize - strategy_wel_qty, qty_step
                    ),
                    0.0f
                );
                grid_source.wel_enforcer_enabled = false;
                grid_source.twel_enforcer_enabled = false;
                grid_source.unstuck_enabled = false;
            }
            CloseGroup group;
            int grid_rung_limit = strategy_wel_qty > 0.0f ? 499 : 500;
            int group_count = recursive_close_groups(
                grid_source, false, -1, qty_step, price_step,
                min_qty, min_cost, c_mult, grid_rung_limit, group
            );
            bool reverse = grid_source.close_threshold_we > 0.0f;
            float ordinary_budget = fmax(
                round_step(
                    short_side.close_gen_psize - reducer_qty, qty_step
                ),
                0.0f
            );
            bool trim_for_reducer = short_side.close_is_exposure_reducer
                && reducer_qty > 0.0f;
            float remaining_budget = ordinary_budget;
            float kept_ordinary = 0.0f;
            float minimum_any = trim_for_reducer
                ? min_entry_qty(
                    reducer_price, qty_step, min_qty, min_cost, c_mult
                ) : 0.0f;
            int last_kept_rank = -1;
            for (int trim_rank = 0;
                trim_for_reducer && trim_rank < group_count;
                ++trim_rank) {
                int wanted = reverse
                    ? group_count - trim_rank - 1 : trim_rank;
                recursive_close_groups(
                    grid_source, false, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, grid_rung_limit, group
                );
                float trimmed_qty = fmin(group.qty, remaining_budget);
                float group_min = min_entry_qty(
                    group.price, qty_step, min_qty, min_cost, c_mult
                );
                bool partial_trim = trimmed_qty + 1.0e-6f < group.qty;
                if (trimmed_qty + 1.0e-6f < group_min) {
                    trimmed_qty = 0.0f;
                    if (partial_trim) remaining_budget = 0.0f;
                }
                if (trimmed_qty > 0.0f) {
                    kept_ordinary += trimmed_qty;
                    remaining_budget = fmax(
                        round_step(remaining_budget - trimmed_qty, qty_step),
                        0.0f
                    );
                    minimum_any = fmin(minimum_any, group_min);
                    last_kept_rank = trim_rank;
                }
            }
            float dust_remainder = fmax(
                round_step(
                    short_side.close_gen_psize - reducer_qty - kept_ordinary,
                    qty_step
                ),
                0.0f
            );
            if (dust_remainder > 0.0f && dust_remainder < minimum_any
                && last_kept_rank < 0) {
                reducer_qty = fmin(
                    short_side.close_gen_psize,
                    round_step(reducer_qty + dust_remainder, qty_step)
                );
                dust_remainder = 0.0f;
            }
            bool reducer_reachable = reducer_qty > 0.0f
                && reducer_ticks > low_nonfill_max_tick;
            bool reducer_executed = false;
            remaining_budget = ordinary_budget;
            for (int rank = 0; rank < group_count; ++rank) {
                int wanted = reverse ? group_count - rank - 1 : rank;
                recursive_close_groups(
                    grid_source, false, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, grid_rung_limit, group
                );
                if (group.qty <= 0.0f) break;
                float trimmed_group_qty = group.qty;
                if (trim_for_reducer) {
                    float group_min = min_entry_qty(
                        group.price, qty_step, min_qty, min_cost, c_mult
                    );
                    trimmed_group_qty = fmin(group.qty, remaining_budget);
                    bool partial_trim = trimmed_group_qty + 1.0e-6f
                        < group.qty;
                    if (trimmed_group_qty + 1.0e-6f < group_min) {
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
                    float pnl = reducer_qty * c_mult
                        * (short_side.pprice - reducer_price);
                    float fee = reducer_qty * reducer_price * c_mult * maker_fee;
                    if (!realized_loss_proxy_allows_reducer(
                            reducer_qty, reducer_price, short_side.pprice, false,
                            short_side.close_is_unstuck_reducer,
                            c_mult, maker_fee, loss_gate_enabled, balance,
                            realized_pnl_cumsum_last,
                            realized_pnl_cumsum_max,
                            max_realized_loss_pct)) {
                        reducer_reachable = false;
                    } else {
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
                        short_side.psize = fmax(
                            round_step(
                                short_side.psize - reducer_qty, qty_step
                            ),
                            0.0f
                        );
                        day_volume += reducer_qty * reducer_price / balance;
                        short_close_fill = true;
                        reducer_executed = true;
                    }
                }
                if (group.ticks <= low_nonfill_max_tick) break;
                float group_qty = trimmed_group_qty;
                if (group_qty <= 0.0f) continue;
                float adj = fmin(round_step(group_qty, qty_step), short_side.psize);
                float pnl = adj * c_mult * (short_side.pprice - group.price);
                float fee = adj * group.price * c_mult * maker_fee;
                if (!realized_loss_proxy_allows_close(
                        adj, group.price, short_side.pprice, false,
                        c_mult, maker_fee, loss_gate_enabled)) continue;
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
                float new_psize = fmax(
                    round_step(short_side.psize - adj, qty_step), 0.0f
                );
                bool went_flat = new_psize <= 0.0f;
                short_side.psize = new_psize;
                day_volume += fabs(adj) * group.price / balance;
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
                float pnl = adj * c_mult
                    * (short_side.pprice - reducer_price);
                float fee = adj * reducer_price * c_mult * maker_fee;
                if (realized_loss_proxy_allows_reducer(
                        adj, reducer_price, short_side.pprice, false,
                        short_side.close_is_unstuck_reducer,
                        c_mult, maker_fee, loss_gate_enabled, balance,
                        realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max,
                        max_realized_loss_pct)) {
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
                    short_side.psize = fmax(
                        round_step(short_side.psize - adj, qty_step), 0.0f
                    );
                    day_volume += adj * reducer_price / balance;
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
        } else if (short_scan_close_grid || short_secondary_close_fill) {
            bool secondary_first = short_secondary_close_fill
                && (!short_scan_close_grid
                    || short_side.secondary_close_price
                        >= short_side.close_price);
            for (int rank = 0; rank < 2; ++rank) {
                bool use_secondary = secondary_first ? rank == 0 : rank == 1;
                bool reachable = use_secondary
                    ? short_secondary_close_fill : short_scan_close_grid;
                if (!reachable || short_side.psize <= 0.0f) continue;
                bool market_panic = !use_secondary && short_side.close_is_panic
                    && short_hsl_panic_market;
                float cp = use_secondary ? short_side.secondary_close_price
                    : market_panic
                        ? float(max(directional_ticks(
                            close * (1.0f + market_order_slippage_pct),
                            price_step, true
                        ), 1)) * price_step
                        : short_side.close_price;
                float requested_qty = use_secondary
                    ? short_side.secondary_close_qty : short_side.close_qty;
                float adj = fmin(
                    round_step(requested_qty, qty_step), short_side.psize
                );
                float pnl = adj * c_mult * (short_side.pprice - cp);
                float fee = adj * cp * c_mult
                    * (market_panic ? taker_fee : maker_fee);
                bool selected_unstuck = !use_secondary
                    && short_side.close_is_unstuck_reducer;
                if (!short_side.close_is_panic
                    && !realized_loss_proxy_allows_reducer(
                        adj, cp, short_side.pprice, false, selected_unstuck,
                        c_mult, maker_fee, loss_gate_enabled, balance,
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

        bool short_entry_fill = valid && alive && short_enabled
            && short_side.entry_qty > 0.0f
            && short_side.entry_ticks <= high_fill_max_tick;
        if (short_entry_fill) {
            TmSide ladder_side = short_side;
            ladder_side.psize = short_side.entry_gen_psize;
            ladder_side.pprice = short_side.entry_gen_pprice;
            const float ladder_balance = short_side.entry_gen_balance;
            const float ladder_kf = short_side.entry_gen_kf;
            int ladder_touch_ticks = short_side.entry_gen_touch_ticks;
            int previous_ticks = 0;
            for (int rung = 0; rung < 500; ++rung) {
                int entry_ticks = rung == 0
                    ? short_side.entry_ticks : ladder_side.entry_ticks;
                float ep = rung == 0
                    ? short_side.entry_price : ladder_side.entry_price;
                float eq = round_step(
                    rung == 0 ? short_side.entry_qty : ladder_side.entry_qty,
                    qty_step
                );
                if (eq <= 0.0f || entry_ticks > high_fill_max_tick
                    || (rung > 0 && entry_ticks == previous_ticks)) break;

                float fee = eq * ep * c_mult * maker_fee;
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

                bool sim_flat = ladder_side.psize <= 0.0f;
                float sim_psize = round_step(ladder_side.psize + eq, qty_step);
                ladder_side.pprice = sim_flat ? ep
                    : ladder_side.pprice
                        * (ladder_side.psize / fmax(sim_psize, 1.0e-12f))
                        + ep * (eq / fmax(sim_psize, 1.0e-12f));
                ladder_side.psize = sim_psize;
                previous_ticks = entry_ticks;
                ladder_touch_ticks = max(ladder_touch_ticks, entry_ticks);
                if (short_side.entry_retracement_base > 0.0f
                    || short_side.cooldown_min != 0.0f) break;
                generate_short_orders(
                    ladder_side, ladder_balance, ep, qty_step,
                    ladder_touch_ticks, ladder_touch_ticks, ladder_touch_ticks,
                    touch_min_qty, touch_min_qty_relation, price_step, min_qty,
                    min_cost, c_mult, ladder_kf, false
                );
            }
            short_side.entry_qty = 0.0f;
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
        if (long_enabled) {
            update_trailing(
                long_side, high, low, close, valid,
                long_close_fill || long_entry_fill
            );
        }
        if (short_enabled) {
            update_trailing(
                short_side, high, low, close, valid,
                short_close_fill || short_entry_fill
            );
        }

        bool gen = can_gen && alive;
        eq_started = eq_started || gen;
        int long_hsl_mode = hsl_mode(long_hsl, long_side.psize > 0.0f);
        int short_hsl_mode = hsl_mode(short_hsl, short_side.psize > 0.0f);
        if (gen) {
            long_side.close_is_panic = false;
            short_side.close_is_panic = false;
            // When both sides are flat, an exact Rust path that remains alive has
            // balance above liq_floor. If either side is open, equity cannot bound
            // exact cash balance, so flat-side eligibility uses zero and fails closed.
            float guaranteed_balance_lower =
                long_side.psize <= 0.0f && short_side.psize <= 0.0f
                ? liq_floor : 0.0f;
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
            // Exact Rust tries the next-largest protective reducer when the
            // winner is loss-gated. Auto-unstuck may consume the conservative
            // all-history budget; other TM reducers retain the zero-loss envelope.
            bool long_wel_enabled = long_side.wel_enforcer_enabled;
            bool long_twel_enabled = long_side.twel_enforcer_enabled;
            for (int retry = 0; retry < 3 && long_enabled
                && loss_gate_enabled
                && long_side.close_is_exposure_reducer
                && !realized_loss_proxy_allows_reducer(
                    long_side.close_qty, long_side.close_price,
                    long_side.pprice, true,
                    long_side.close_is_unstuck_reducer,
                    c_mult, maker_fee, true, balance,
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
                    short_side.close_qty, short_side.close_price,
                    short_side.pprice, false,
                    short_side.close_is_unstuck_reducer,
                    c_mult, maker_fee, true, balance,
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
                long_side.close_ticks = max(touch_down_tick - 1, 1);
                long_side.close_price = float(long_side.close_ticks) * price_step;
                long_side.close_qty = long_side.psize;
                long_side.secondary_close_qty = 0.0f;
                long_side.close_is_exposure_reducer = false;
                long_side.close_is_twel_reducer = false;
                long_side.close_is_unstuck_reducer = false;
                long_side.close_is_panic = true;
            }
            if (short_enabled && short_hsl_mode == 3) {
                short_side.close_ticks = max(touch_up_tick + 1, 1);
                short_side.close_price = float(short_side.close_ticks) * price_step;
                short_side.close_qty = short_side.psize;
                short_side.secondary_close_qty = 0.0f;
                short_side.close_is_exposure_reducer = false;
                short_side.close_is_twel_reducer = false;
                short_side.close_is_unstuck_reducer = false;
                short_side.close_is_panic = true;
            }
        }

        float long_unreal = long_side.psize > 0.0f
            ? long_side.psize * c_mult * (close - long_side.pprice) : 0.0f;
        float short_unreal = short_side.psize > 0.0f
            ? short_side.psize * c_mult * (short_side.pprice - close) : 0.0f;
        float equity = balance + long_unreal + short_unreal;
        if (gen && valid && alive && balance > 0.0f && equity > liq_floor) {
            bool long_blocking_orders = long_hsl_mode != 3 && (
                long_side.entry_qty > 0.0f || long_side.close_qty > 0.0f
                    || long_side.secondary_close_qty > 0.0f
            );
            bool short_blocking_orders = short_hsl_mode != 3 && (
                short_side.entry_qty > 0.0f || short_side.close_qty > 0.0f
                    || short_side.secondary_close_qty > 0.0f
            );
            bool hsl_update_valid = update_dual_side_hsl(
                long_hsl, short_hsl, balance, starting_balance,
                realized_pnl_cumsum_last,
                realized_pnl_cumsum_long, realized_pnl_cumsum_short,
                long_unreal, short_unreal,
                long_side.psize > 0.0f, short_side.psize > 0.0f,
                long_blocking_orders, short_blocking_orders,
                kf, interval_ms
            );
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
        bool active = eq_started && alive && valid;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = kf;
            last_eq_k = kf;
            if (any_fill) {
                int active_fill_day = int(kf - first_eq_k) / 1440;
                if (active_fill_day != last_active_fill_day) {
                    fills_active_days_count += 1.0f;
                    last_active_fill_day = active_fill_day;
                }
            }
            bool liq = balance <= 0.0f || equity <= liq_floor;
            float eqf = liq ? liq_floor : equity;
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
}

kernel void passivbot_trailing_martingale(
    constant float* bars,
    constant int* flags,
    constant float* params,
    constant float* settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b [[thread_position_in_grid]]
) {
    passivbot_single_coin_impl(
        bars, flags, params, settings, sizes, daily, scalars, gap_hist, b
    );
}
