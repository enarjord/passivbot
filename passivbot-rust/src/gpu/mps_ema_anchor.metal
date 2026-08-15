#include <metal_stdlib>
using namespace metal;

constant int DAILY_COLS = 5;
constant int SCALAR_COLS = 15;
constant int GAP_BINS = 128;

inline float safe_div(float a, float b) {
    return a / fmax(fabs(b), 1.0e-12f);
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
    const float log_bin_scale = 127.0f / log(4000001.0f);

    const int po = int(b) * P;
    float ema0;
    float ema1;
    float ema2;
    float alpha0;
    float alpha1;
    float alpha2;
    float alpha1m;
    float alpha1h;
    float twel;
    float cooldown_min;

    // EMA-anchor parameters.
    float base_qty_pct = 0.0f;
    float ddf = 0.0f;
    float offset = 0.0f;
    float psize_weight = 0.0f;
    float w1h = 0.0f;
    float w1m = 0.0f;

    float span0 = params[po + 1];
    float span1 = params[po + 2];
    float span2 = sqrt(span0 * span1);
    float lo_span = fmin(span0, fmin(span1, span2));
    float hi_span = fmax(span0, fmax(span1, span2));
    float mid_span = span0 + span1 + span2 - lo_span - hi_span;
    alpha0 = clamp(2.0f / (lo_span + 1.0f), 0.0f, 1.0f);
    alpha1 = clamp(2.0f / (mid_span + 1.0f), 0.0f, 1.0f);
    alpha2 = clamp(2.0f / (hi_span + 1.0f), 0.0f, 1.0f);
    float span_h = params[po + 8];
    float span_m = params[po + 9];
    alpha1h = span_h > 0.0f ? 2.0f / (fmax(span_h, 1.0f) + 1.0f) : 0.0f;
    alpha1m = span_m > 0.0f ? clamp(2.0f / (span_m + 1.0f), 0.0f, 1.0f) : 0.0f;
    base_qty_pct = params[po + 0];
    ddf = params[po + 3];
    offset = params[po + 4];
    psize_weight = params[po + 5];
    w1h = params[po + 6];
    w1m = params[po + 7];
    cooldown_min = ceil(params[po + 10]);
    twel = params[po + 11];

    const int seed_k = clamp(first_valid, 0, T - 1);
    const float seed_close = bars[seed_k * 5 + 2];
    ema0 = seed_close;
    ema1 = seed_close;
    ema2 = seed_close;
    float vol1m = 0.0f;
    float vol1h = 0.0f;
    float psize = 0.0f;
    float pprice = 0.0f;
    float balance = starting_balance;
    bool alive = true;
    int liq_day = -1;
    float last_inc_k = -1.0f;
    int entry_ticks = 0;
    float entry_qty = 0.0f;
    int close_ticks = 0;
    float close_qty = 0.0f;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float pos_open_k = -1.0f;
    float held_max_min = 0.0f;
    float last_fill_k = -1.0f;
    float first_fill_k = -1.0f;
    float gap_max_min = 0.0f;
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

    for (int j = 0; j < GAP_BINS; ++j) gap_hist[int(b) * GAP_BINS + j] = 0;

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

        bool fill_close = valid && alive && close_qty > 0.0f && psize > 0.0f
            && high > float(close_ticks) * price_step;
        float pnl = 0.0f;
        bool went_flat = false;
        if (fill_close) {
            float cp = float(close_ticks) * price_step;
            float adj = fmin(round_step(close_qty, qty_step), psize);
            pnl = adj * c_mult * (cp - pprice);
            balance += pnl - adj * cp * c_mult * maker_fee;
            float new_psize = fmax(round_step(psize - adj, qty_step), 0.0f);
            went_flat = new_psize <= 0.0f;
            psize = new_psize;
            if (went_flat) pprice = 0.0f;
            day_volume += fabs(adj) * cp / balance;
            if (went_flat) {
                if (pos_open_k >= 0.0f) held_max_min = fmax(held_max_min, float(k) - pos_open_k);
                pos_open_k = -1.0f;
            }
            close_qty = 0.0f;
        }

        bool fill_entry = valid && alive && entry_qty > 0.0f
            && low < float(entry_ticks) * price_step;
        bool was_flat = psize <= 0.0f;
        if (fill_entry) {
            float ep = float(entry_ticks) * price_step;
            float eq = round_step(entry_qty, qty_step);
            balance -= eq * ep * c_mult * maker_fee;
            float new_psize = round_step(psize + eq, qty_step);
            float new_pprice = was_flat ? ep
                : pprice * (psize / fmax(new_psize, 1.0e-12f))
                    + ep * (eq / fmax(new_psize, 1.0e-12f));
            if (was_flat) pos_open_k = float(k);
            psize = new_psize;
            pprice = new_pprice;
            last_inc_k = float(k);
            day_volume += fabs(eq) * ep / balance;
            entry_qty = 0.0f;
        }

        bool any_fill = fill_close || fill_entry;
        if (any_fill) {
            day_has_fill = 1.0f;
            if (last_fill_k >= 0.0f) {
                float gap = float(k) - last_fill_k;
                int bin = clamp(int(log(fmax(gap, 0.0f) + 1.0f) * log_bin_scale), 0, 127);
                gap_hist[int(b) * GAP_BINS + bin] += 1;
                gap_max_min = fmax(gap_max_min, gap);
            }
            if (first_fill_k < 0.0f) first_fill_k = float(k);
            last_fill_k = float(k);
        }

        if (hour_valid && alpha1h > 0.0f) vol1h = alpha1h * hour_lr + (1.0f - alpha1h) * vol1h;
        if (valid) {
            ema0 = alpha0 * close + (1.0f - alpha0) * ema0;
            ema1 = alpha1 * close + (1.0f - alpha1) * ema1;
            ema2 = alpha2 * close + (1.0f - alpha2) * ema2;
            if (alpha1m > 0.0f) vol1m = alpha1m * log_range + (1.0f - alpha1m) * vol1m;
        }

        bool gen = can_gen && alive;
        eq_started = eq_started || gen;
        float lower = fmin(ema0, fmin(ema1, ema2));
        float upper = fmax(ema0, fmax(ema1, ema2));
        float price_now = close;
        float current_we = psize > 0.0f && balance > 0.0f
            ? psize * price_now * c_mult / balance : 0.0f;
        float current_cost_we = psize > 0.0f && balance > 0.0f
            ? psize * pprice * c_mult / balance : 0.0f;

        if (gen) {
            float mult = fmax(1.0f + vol1h * w1h + vol1m * w1m, 1.0f);
            float eff_off = offset * mult;
            float swer = psize > 0.0f && balance > 0.0f
                ? current_we / fmax(twel, 1.0e-12f) : 0.0f;
            float inv_shift = swer * psize_weight;
            int bid_ticks = min(int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
                                int(floor(price_now / price_step + 1.0e-6f)));
            float bid_price = float(bid_ticks) * price_step;
            float min_q = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
            float base_q = fmax(min_q, round_step(
                balance * twel * base_qty_pct / fmax(bid_price, 1.0e-12f) / c_mult, qty_step));
            float e_qty = round_step(base_q * fmax(1.0f + swer * ddf, 1.0f), qty_step);
            bool cooldown = cooldown_min > 0.0f && last_inc_k >= 0.0f
                && float(k) < last_inc_k + cooldown_min;
            float cap = twel - 1.0e-7f;
            float headroom = (cap * balance - psize * pprice * c_mult)
                / fmax(bid_price * c_mult, 1.0e-12f);
            bool over = (psize * pprice + e_qty * bid_price) * c_mult
                / fmax(balance, 1.0e-9f) >= cap;
            float capped = floor_step(headroom, qty_step);
            if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
            if (current_cost_we >= cap || cooldown || bid_price <= 0.0f || balance <= 0.0f
                || base_qty_pct <= 0.0f) e_qty = 0.0f;
            entry_ticks = bid_ticks;
            entry_qty = e_qty;

            int ask_ticks = max(int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
                                int(ceil(price_now / price_step - 1.0e-6f)));
            float ask_price = float(ask_ticks) * price_step;
            float min_cq = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
            float clip = fmin(psize, fmax(min_cq, round_step(
                balance * twel * base_qty_pct / fmax(ask_price, 1.0e-12f) / c_mult, qty_step)));
            float c_qty = psize <= min_cq || psize - clip < min_cq ? psize : clip;
            if (psize <= 0.0f || ask_price <= 0.0f) c_qty = 0.0f;
            close_ticks = ask_ticks;
            close_qty = c_qty;
        }

        float unreal = psize > 0.0f ? psize * c_mult * (price_now - pprice) : 0.0f;
        float equity = balance + unreal;
        bool active = eq_started && alive && valid;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = float(k);
            last_eq_k = float(k);
            bool liq = balance <= 0.0f || equity <= liq_floor;
            float eqf = liq ? liq_floor : equity;
            if (eqf > run_peak) {
                if (last_high_k >= 0.0f) recovery_max_min = fmax(recovery_max_min, float(k) - last_high_k);
                last_high_k = float(k);
                run_peak = eqf;
            }
            float dd = fmax((run_peak - eqf) / fmax(fabs(run_peak), 1.0e-12f), 0.0f);
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

    if (pos_open_k >= 0.0f && last_eq_k >= 0.0f)
        held_max_min = fmax(held_max_min, last_eq_k - pos_open_k);
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
    scalars[so + 11] = psize;
    scalars[so + 12] = pprice;
    scalars[so + 13] = alive ? 1.0f : 0.0f;
    scalars[so + 14] = psize > 0.0f ? 1.0f : 0.0f;
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
