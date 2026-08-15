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
    float entry_qty, close_qty;
    float min_since_open, max_since_min, max_since_open, min_since_max;
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
    s.min_since_open = INFINITY; s.max_since_min = 0.0f;
    s.max_since_open = 0.0f; s.min_since_max = INFINITY;
    return s;
}

inline void update_indicators(
    thread TmSide& s, float close, float lr, float hour_lr, bool valid, bool hour_valid
) {
    if (hour_valid && s.alpha1h > 0.0f)
        s.vol1h = s.alpha1h * hour_lr + (1.0f - s.alpha1h) * s.vol1h;
    if (valid) {
        s.ema0 = s.alpha0 * close + (1.0f - s.alpha0) * s.ema0;
        s.ema1 = s.alpha1 * close + (1.0f - s.alpha1) * s.ema1;
        s.ema2 = s.alpha2 * close + (1.0f - s.alpha2) * s.ema2;
        if (s.alpha1m > 0.0f)
            s.vol1m = s.alpha1m * lr + (1.0f - s.alpha1m) * s.vol1m;
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

inline int nearest_ticks(float price, float step) {
    return int(floor(price / step + 0.5f));
}

inline int touch_clamp(int target, int touch, bool up) {
    return up ? max(target, touch) : min(target, touch);
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
    thread TmSide& s, float balance, float price, float pct,
    float qty_step, float min_qty, float min_cost, float c_mult
) {
    float mq = min_entry_qty(price, qty_step, min_qty, min_cost, c_mult);
    float full = balance * s.twel / fmax(s.pprice * c_mult, 1.0e-12f);
    float qty = fmin(
        round_step(s.psize, qty_step),
        fmax(mq, ceil_step(full * pct + fmax(s.psize - full, 0.0f), qty_step))
    );
    if (qty > 0.0f && qty < s.psize && s.psize - qty < mq) qty = s.psize;
    if (s.psize < mq * (1.0f - 1.0e-6f) && qty > 0.0f) qty = s.psize;
    else if (qty > 0.0f && qty * (1.0f + 1.0e-6f) < mq) qty = 0.0f;
    return qty;
}

inline void generate_orders(
    thread TmSide& s, bool is_long, float balance, float price_now,
    float qty_step, float price_step, float min_qty, float min_cost, float c_mult,
    float kf, bool block_initial
) {
    bool entry_up = !is_long;
    bool close_up = is_long;
    float band = is_long ? fmin(s.ema0, fmin(s.ema1, s.ema2))
                         : fmax(s.ema0, fmax(s.ema1, s.ema2));
    int entry_touch = directional_ticks(price_now, price_step, entry_up);
    int band_ticks = directional_ticks(
        band * (is_long ? 1.0f - s.initial_ema_dist
                        : 1.0f + s.initial_ema_dist),
        price_step, entry_up
    );
    int initial_ticks = s.gate_initial
        ? touch_clamp(band_ticks, entry_touch, entry_up) : entry_touch;
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
    int reentry_ticks = touch_clamp(
        directional_ticks(reentry_target, price_step, entry_up), entry_touch, entry_up
    );
    if (s.gate_reentry)
        reentry_ticks = touch_clamp(band_ticks, reentry_ticks, entry_up);
    float reentry_price = float(reentry_ticks) * price_step;
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
    bool cooldown = s.cooldown_min > 0.0f && s.last_inc_k >= 0.0f
        && kf < s.last_inc_k + s.cooldown_min;
    if (cooldown || balance <= 0.0f || s.initial_qty_pct <= 0.0f
        || s.twel <= 0.0f || eticks <= 1 || (block_initial && flat)) eqty = 0.0f;
    s.entry_ticks = eticks;
    s.entry_qty = crop_entry(
        s, balance, float(eticks) * price_step, eqty,
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
    float rounded_target = float(target_ticks) * price_step;
    bool touch_controls = (trailing_close && ct <= 0.0f) || (close_up
        ? price_now > rounded_target : price_now < rounded_target);
    int cticks = touch_controls ? nearest_ticks(price_now, price_step) : target_ticks;
    float close_price = float(cticks) * price_step;
    float pct = trailing_close ? s.close_qty_pct
        : (s.close_threshold_we == 0.0f ? 1.0f : s.close_qty_pct);
    s.close_ticks = cticks;
    s.close_qty = s.psize > 0.0f && close_price > 0.0f
            && (!trailing_close || close_triggered)
        ? calc_close_qty(
            s, balance, close_price, pct, qty_step, min_qty, min_cost, c_mult
        ) : 0.0f;
}

inline void generate_long_orders(
    thread TmSide& s, float balance, float price_now, float qty_step,
    float price_step, float min_qty, float min_cost, float c_mult,
    float kf, bool block_initial
) {
    generate_orders(
        s, true, balance, price_now, qty_step, price_step, min_qty, min_cost,
        c_mult, kf, block_initial
    );
}

inline void generate_short_orders(
    thread TmSide& s, float balance, float price_now, float qty_step,
    float price_step, float min_qty, float min_cost, float c_mult,
    float kf, bool block_initial
) {
    generate_orders(
        s, false, balance, price_now, qty_step, price_step, min_qty, min_cost,
        c_mult, kf, block_initial
    );
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
        const int fo = k * 4;
        const float high = bars[bo + 0];
        const float low = bars[bo + 1];
        const float close = bars[bo + 2];
        const float log_range = bars[bo + 3];
        const float hour_lr = bars[bo + 4];
        const bool valid = flags[fo + 0] != 0;
        const bool can_gen = flags[fo + 1] != 0;
        const int di = flags[fo + 2];
        const bool hour_valid = flags[fo + 3] != 0;
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

        bool long_close_fill = valid && alive && long_enabled
            && long_side.close_qty > 0.0f && long_side.psize > 0.0f
            && high > float(long_side.close_ticks) * price_step;
        if (long_close_fill) {
            float cp = float(long_side.close_ticks) * price_step;
            float adj = fmin(round_step(long_side.close_qty, qty_step), long_side.psize);
            float pnl = adj * c_mult * (cp - long_side.pprice);
            balance += pnl - adj * cp * c_mult * maker_fee;
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
        }

        bool long_entry_fill = valid && alive && long_enabled
            && long_side.entry_qty > 0.0f
            && low < float(long_side.entry_ticks) * price_step;
        if (long_entry_fill) {
            float ep = float(long_side.entry_ticks) * price_step;
            float eq = round_step(long_side.entry_qty, qty_step);
            balance -= eq * ep * c_mult * maker_fee;
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

        bool short_close_fill = valid && alive && short_enabled
            && short_side.close_qty > 0.0f && short_side.psize > 0.0f
            && low < float(short_side.close_ticks) * price_step;
        if (short_close_fill) {
            float cp = float(short_side.close_ticks) * price_step;
            float adj = fmin(round_step(short_side.close_qty, qty_step), short_side.psize);
            float pnl = adj * c_mult * (short_side.pprice - cp);
            balance += pnl - adj * cp * c_mult * maker_fee;
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
        }

        bool short_entry_fill = valid && alive && short_enabled
            && short_side.entry_qty > 0.0f
            && high > float(short_side.entry_ticks) * price_step;
        if (short_entry_fill) {
            float ep = float(short_side.entry_ticks) * price_step;
            float eq = round_step(short_side.entry_qty, qty_step);
            balance -= eq * ep * c_mult * maker_fee;
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
            bool block_long_initial = false;
            bool block_short_initial = false;
            if (long_enabled && short_enabled && !hedge_mode) {
                if (long_side.psize > 0.0f) {
                    block_short_initial = true;
                } else if (short_side.psize > 0.0f) {
                    block_long_initial = true;
                } else {
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
                    long_side, balance, close, qty_step, price_step, min_qty,
                    min_cost, c_mult, kf, block_long_initial
                );
            }
            if (short_enabled) {
                generate_short_orders(
                    short_side, balance, close, qty_step, price_step, min_qty,
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
