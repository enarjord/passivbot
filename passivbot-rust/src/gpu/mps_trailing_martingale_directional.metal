#include <metal_stdlib>
using namespace metal;

constant int DAILY_COLS = 5;
constant int SCALAR_COLS = 18;
constant int GAP_BINS = 128;
constant int SIDE_PARAMS = 27;

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
    float cooldown_min, twel;
    bool gate_initial, gate_reentry;
    float ema0, ema1, ema2, vol1m, vol1h;
    float psize, pprice, last_inc_k, pos_open_k;
    int entry_ticks, close_ticks;
    float entry_price, close_price, entry_qty, close_qty;
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
    s.ema0 = seed; s.ema1 = seed; s.ema2 = seed;
    s.vol1m = 0.0f; s.vol1h = 0.0f;
    s.psize = 0.0f; s.pprice = 0.0f;
    s.last_inc_k = -1.0f; s.pos_open_k = -1.0f;
    s.entry_ticks = 0; s.entry_qty = 0.0f;
    s.close_ticks = 0; s.close_qty = 0.0f;
    s.entry_price = 0.0f; s.close_price = 0.0f;
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
    if (we_if <= s.twel * 1.01f) return qty;
    float q = round_step(
        (s.twel * balance - cost) / fmax(price * c_mult, 1.0e-12f), qty_step
    );
    float mq = min_entry_qty(price, qty_step, min_qty, min_cost, c_mult);
    q = fmax(q, mq);
    return q < qty ? q : qty;
}

inline float calc_close_qty(
    thread TmSide& s, float balance, float mq, int mq_relation, float pct,
    float qty_step, float c_mult
) {
    float full = balance * s.twel / fmax(s.pprice * c_mult, 1.0e-12f);
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
        balance * s.twel * s.initial_qty_pct
            / fmax(initial_price * c_mult, 1.0e-12f), qty_step
    ));
    bool flat = s.psize <= 0.0f;
    bool partial = !flat && s.psize < iq * 0.8f;
    float iq_partial = fmax(min_iq, floor_step(iq - s.psize, qty_step));
    float iq_effective = !flat && s.psize < iq
        ? fmax(round_step(s.psize, qty_step), min_iq) : iq;
    float we = !flat && balance > 0.0f
        ? s.psize * s.pprice * c_mult / balance : 0.0f;
    float wer = we / fmax(s.twel, 1.0e-12f);
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
            balance * s.twel * s.initial_qty_pct
                / fmax(reentry_price * c_mult, 1.0e-12f)
        ), qty_step
    )));
    float we_if = (s.psize * s.pprice + rq * reentry_price)
        * c_mult / fmax(balance, 1.0e-9f);
    float crop_fraction = (s.twel - we) / fmax(we_if - we, 1.0e-12f);
    float rq_crop = fmax(round_step(rq * crop_fraction, qty_step), min_rq);
    if (we_if > s.twel * 1.01f && rq_crop < rq) rq = rq_crop;
    bool cap_hit = trailing_entry ? we > s.twel * 0.999f : we >= s.twel * 0.999f;
    bool reentry_ok = !flat && !partial && !cap_hit && reentry_ticks > 1
        && (!trailing_entry || entry_triggered);
    float eqty = flat ? iq : (partial ? iq_partial : (reentry_ok ? rq : 0.0f));
    int eticks = flat || partial ? initial_ticks : reentry_ticks;
    float eprice = flat || partial ? initial_price : reentry_price;
    bool cooldown = s.cooldown_min > 0.0f && s.last_inc_k >= 0.0f
        && kf < s.last_inc_k + s.cooldown_min;
    if (cooldown || balance <= 0.0f || s.initial_qty_pct <= 0.0f
        || s.twel <= 0.0f || eticks <= 1
        || (block_initial && flat)) eqty = 0.0f;
    s.entry_ticks = eticks;
    s.entry_price = eprice;
    s.entry_qty = crop_entry(
        s, balance, eprice, eqty,
        qty_step, min_qty, min_cost, c_mult
    );

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
    const float log_bin_scale = 127.0f / log(4000001.0f);

    const int po = int(b) * P;
    const int seed_k = clamp(first_valid, 0, T - 1);
    const float seed_close = bars[seed_k * 5 + 2];
    TmSide long_side = load_side(params, po, seed_close);
    TmSide short_side = load_side(params, po + SIDE_PARAMS, seed_close);

    float balance = starting_balance;
    bool alive = true;
    int liq_day = -1;
    float held_max_min = 0.0f;
    float last_fill_k = -1.0f;
    float first_fill_k = -1.0f;
    float gap_max_min = 0.0f;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float last_high_k = -1.0f;
    float recovery_max_min = 0.0f;
    float first_eq_k = -1.0f;
    float last_eq_k = -1.0f;
    bool eq_started = false;

    int cur_day = flags[2];
    bool day_touched = false;
    float day_end = 0.0f;
    float day_min = INFINITY;
    float day_dd = 0.0f;
    float day_volume = 0.0f;
    float day_has_fill = 0.0f;

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
            }
            cur_day = di;
            day_touched = false;
            day_end = 0.0f;
            day_min = INFINITY;
            day_dd = 0.0f;
            day_volume = 0.0f;
            day_has_fill = 0.0f;
        }

        bool long_close_fill = false;
        bool long_close_ready = valid && alive && long_enabled
            && long_side.close_qty > 0.0f && long_side.psize > 0.0f;
        bool long_recursive_close = long_side.close_retracement_base <= 0.0f;
        bool long_scan_close_grid = long_close_ready
            && long_side.close_ticks <= high_fill_max_tick;
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
            long_scan_close_grid = nearest_ticks <= high_fill_max_tick;
        }
        if (long_scan_close_grid && long_recursive_close) {
            CloseGroup group;
            int group_count = recursive_close_groups(
                long_side, true, -1, qty_step, price_step,
                min_qty, min_cost, c_mult, group
            );
            bool reverse = long_side.close_threshold_we > 0.0f;
            for (int rank = 0; rank < group_count; ++rank) {
                int wanted = reverse ? group_count - rank - 1 : rank;
                recursive_close_groups(
                    long_side, true, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, group
                );
                if (group.qty <= 0.0f || group.ticks > high_fill_max_tick) break;
                float adj = fmin(round_step(group.qty, qty_step), long_side.psize);
                float pnl = adj * c_mult * (group.price - long_side.pprice);
                float fee = adj * group.price * c_mult * maker_fee;
                balance += pnl - fee;
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
                        held_max_min = fmax(
                            held_max_min, kf - long_side.pos_open_k
                        );
                    }
                    long_side.pos_open_k = -1.0f;
                    break;
                }
            }
            if (long_close_fill) long_side.close_qty = 0.0f;
        } else if (long_scan_close_grid) {
            float cp = long_side.close_price;
            float adj = fmin(round_step(long_side.close_qty, qty_step), long_side.psize);
            float pnl = adj * c_mult * (cp - long_side.pprice);
            float fee = adj * cp * c_mult * maker_fee;
            balance += pnl - fee;
            float new_psize = fmax(round_step(long_side.psize - adj, qty_step), 0.0f);
            bool went_flat = new_psize <= 0.0f;
            long_side.psize = new_psize;
            if (went_flat) {
                long_side.pprice = 0.0f;
                if (long_side.pos_open_k >= 0.0f) {
                    held_max_min = fmax(held_max_min, kf - long_side.pos_open_k);
                }
                long_side.pos_open_k = -1.0f;
            }
            day_volume += fabs(adj) * cp / balance;
            long_side.close_qty = 0.0f;
            long_close_fill = true;
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
        bool short_recursive_close = short_side.close_retracement_base <= 0.0f;
        bool short_scan_close_grid = short_close_ready
            && short_side.close_ticks > low_nonfill_max_tick;
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
            short_scan_close_grid = nearest_ticks > low_nonfill_max_tick;
        }
        if (short_scan_close_grid && short_recursive_close) {
            CloseGroup group;
            int group_count = recursive_close_groups(
                short_side, false, -1, qty_step, price_step,
                min_qty, min_cost, c_mult, group
            );
            bool reverse = short_side.close_threshold_we > 0.0f;
            for (int rank = 0; rank < group_count; ++rank) {
                int wanted = reverse ? group_count - rank - 1 : rank;
                recursive_close_groups(
                    short_side, false, wanted, qty_step, price_step,
                    min_qty, min_cost, c_mult, group
                );
                if (group.qty <= 0.0f || group.ticks <= low_nonfill_max_tick) break;
                float adj = fmin(round_step(group.qty, qty_step), short_side.psize);
                float pnl = adj * c_mult * (short_side.pprice - group.price);
                float fee = adj * group.price * c_mult * maker_fee;
                balance += pnl - fee;
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
                        held_max_min = fmax(
                            held_max_min, kf - short_side.pos_open_k
                        );
                    }
                    short_side.pos_open_k = -1.0f;
                    break;
                }
            }
            if (short_close_fill) short_side.close_qty = 0.0f;
        } else if (short_scan_close_grid) {
            float cp = short_side.close_price;
            float adj = fmin(round_step(short_side.close_qty, qty_step), short_side.psize);
            float pnl = adj * c_mult * (short_side.pprice - cp);
            float fee = adj * cp * c_mult * maker_fee;
            balance += pnl - fee;
            float new_psize = fmax(round_step(short_side.psize - adj, qty_step), 0.0f);
            bool went_flat = new_psize <= 0.0f;
            short_side.psize = new_psize;
            if (went_flat) {
                short_side.pprice = 0.0f;
                if (short_side.pos_open_k >= 0.0f) {
                    held_max_min = fmax(held_max_min, kf - short_side.pos_open_k);
                }
                short_side.pos_open_k = -1.0f;
            }
            day_volume += fabs(adj) * cp / balance;
            short_side.close_qty = 0.0f;
            short_close_fill = true;
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
        if (gen) {
            // When both sides are flat, an exact Rust path that remains alive has
            // balance above liq_floor. If either side is open, equity cannot bound
            // exact cash balance, so flat-side eligibility uses zero and fails closed.
            float guaranteed_balance_lower =
                long_side.psize <= 0.0f && short_side.psize <= 0.0f
                ? liq_floor : 0.0f;
            bool long_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                long_side.twel, long_side.initial_qty_pct,
                max_effective_min_cost
            );
            bool short_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                short_side.twel, short_side.initial_qty_pct,
                max_effective_min_cost
            );
            bool block_long_initial = !long_min_cost_eligible;
            bool block_short_initial = !short_min_cost_eligible;
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
        }

        float long_unreal = long_side.psize > 0.0f
            ? long_side.psize * c_mult * (close - long_side.pprice) : 0.0f;
        float short_unreal = short_side.psize > 0.0f
            ? short_side.psize * c_mult * (short_side.pprice - close) : 0.0f;
        float equity = balance + long_unreal + short_unreal;
        bool active = eq_started && alive && valid;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = kf;
            last_eq_k = kf;
            bool liq = balance <= 0.0f || equity <= liq_floor;
            float eqf = liq ? liq_floor : equity;
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
    }

    if (long_side.pos_open_k >= 0.0f && last_eq_k >= 0.0f) {
        held_max_min = fmax(held_max_min, last_eq_k - long_side.pos_open_k);
    }
    if (short_side.pos_open_k >= 0.0f && last_eq_k >= 0.0f) {
        held_max_min = fmax(held_max_min, last_eq_k - short_side.pos_open_k);
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
