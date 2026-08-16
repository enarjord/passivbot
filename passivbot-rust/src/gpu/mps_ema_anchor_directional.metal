#include <metal_stdlib>
using namespace metal;

constant int DAILY_COLS = 5;
constant int SCALAR_COLS = 18;
constant int GAP_BINS = 128;
constant int SIDE_PARAMS = 12;

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
    int close_ticks;
    float close_qty;
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
    side.close_ticks = 0;
    side.close_qty = 0.0f;
    return side;
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
    bool block_initial
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
        ? current_we / fmax(side.twel, 1.0e-12f) : 0.0f;
    float inv_shift = swer * side.psize_weight;

    int bid_ticks = min(
        int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
        touch_down_ticks
    );
    float bid_price = float(bid_ticks) * price_step;
    float min_q = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
    float base_q = fmax(min_q, round_step(
        balance * side.twel * side.base_qty_pct
            / fmax(bid_price, 1.0e-12f) / c_mult,
        qty_step
    ));
    float e_qty = round_step(
        base_q * fmax(1.0f + fmax(swer, 0.0f) * side.ddf, 1.0f), qty_step
    );
    bool cooldown = side.cooldown_min > 0.0f && side.last_inc_k >= 0.0f
        && kf < side.last_inc_k + side.cooldown_min;
    float cap = side.twel - 1.0e-7f;
    float headroom = (cap * balance - side.psize * side.pprice * c_mult)
        / fmax(bid_price * c_mult, 1.0e-12f);
    bool over = (side.psize * side.pprice + e_qty * bid_price) * c_mult
        / fmax(balance, 1.0e-9f) >= cap;
    float capped = floor_step(headroom, qty_step);
    if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
    if (current_cost_we >= cap || cooldown || bid_price <= 0.0f || balance <= 0.0f
        || side.base_qty_pct <= 0.0f || (block_initial && side.psize <= 0.0f)) {
        e_qty = 0.0f;
    }
    side.entry_ticks = bid_ticks;
    side.entry_qty = e_qty;

    int ask_ticks = max(
        int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
        touch_up_ticks
    );
    float ask_price = float(ask_ticks) * price_step;
    float min_cq = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
    float clip = fmin(side.psize, fmax(min_cq, round_step(
        balance * side.twel * side.base_qty_pct
            / fmax(ask_price, 1.0e-12f) / c_mult,
        qty_step
    )));
    float c_qty = side.psize <= min_cq || side.psize - clip < min_cq
        ? side.psize : clip;
    if (side.psize <= 0.0f || ask_price <= 0.0f) c_qty = 0.0f;
    side.close_ticks = ask_ticks;
    side.close_qty = c_qty;
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
    bool block_initial
) {
    float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
    float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
    float mult = fmax(1.0f + side.vol1h * side.w1h + side.vol1m * side.w1m, 1.0f);
    float eff_off = side.offset * mult;
    float swer = side.psize > 0.0f && balance > 0.0f
        ? -side.psize * price_now * c_mult / balance / fmax(side.twel, 1.0e-12f)
        : 0.0f;
    float current_cost_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * side.pprice * c_mult / balance : 0.0f;
    float inv_shift = swer * side.psize_weight;

    int ask_ticks = max(
        int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
        touch_up_ticks
    );
    float ask_price = float(ask_ticks) * price_step;
    float min_q = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
    float base_q = fmax(min_q, round_step(
        balance * side.twel * side.base_qty_pct
            / fmax(ask_price, 1.0e-12f) / c_mult,
        qty_step
    ));
    float e_qty = round_step(
        base_q * fmax(1.0f + fmax(-swer, 0.0f) * side.ddf, 1.0f), qty_step
    );
    bool cooldown = side.cooldown_min > 0.0f && side.last_inc_k >= 0.0f
        && kf < side.last_inc_k + side.cooldown_min;
    float cap = side.twel - 1.0e-7f;
    float headroom = (cap * balance - side.psize * side.pprice * c_mult)
        / fmax(ask_price * c_mult, 1.0e-12f);
    bool over = (side.psize * side.pprice + e_qty * ask_price) * c_mult
        / fmax(balance, 1.0e-9f) >= cap;
    float capped = floor_step(headroom, qty_step);
    if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
    if (current_cost_we >= cap || cooldown || ask_price <= 0.0f || balance <= 0.0f
        || side.base_qty_pct <= 0.0f || (block_initial && side.psize <= 0.0f)) {
        e_qty = 0.0f;
    }
    side.entry_ticks = ask_ticks;
    side.entry_qty = e_qty;

    int bid_ticks = min(
        int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
        touch_down_ticks
    );
    float bid_price = float(bid_ticks) * price_step;
    float min_cq = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
    float clip = fmin(side.psize, fmax(min_cq, round_step(
        balance * side.twel * side.base_qty_pct
            / fmax(bid_price, 1.0e-12f) / c_mult,
        qty_step
    )));
    float c_qty = side.psize <= min_cq || side.psize - clip < min_cq
        ? side.psize : clip;
    if (side.psize <= 0.0f || bid_price <= 0.0f) c_qty = 0.0f;
    side.close_ticks = bid_ticks;
    side.close_qty = c_qty;
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
    EmaSide long_side = load_side(params, po, seed_close);
    EmaSide short_side = load_side(params, po + SIDE_PARAMS, seed_close);

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
            && long_side.close_ticks <= high_fill_max_tick;
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
            && long_side.entry_ticks > low_nonfill_max_tick;
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
            && short_side.close_ticks > low_nonfill_max_tick;
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
            && short_side.entry_ticks <= high_fill_max_tick;
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
                    block_long_initial
                );
            }
            if (short_enabled) {
                generate_short_orders(
                    short_side, balance, close, touch_down_tick, touch_up_tick,
                    qty_step, price_step, min_qty, min_cost, c_mult, kf,
                    block_short_initial
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

kernel void passivbot_ema_anchor(
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
