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
// Require a unique round-trip before subtracting counts. Leave the output
// unchanged when the proof is unavailable.
inline bool try_aligned_qty_difference_steps(
    float total, float held, float step, thread float& difference_steps
) {
    float total_count = rint(total / step);
    float held_count = rint(held / step);
    bool counts_resolvable = total_count >= 0.0f && total_count < 0x1p24f
        && held_count >= 0.0f && held_count < 0x1p24f
        && total - nextafter(total, -INFINITY) <= step
        && nextafter(held, INFINITY) - held <= step;
    bool on_grid = total_count * step == total && held_count * step == held;
    if (counts_resolvable && on_grid) {
        difference_steps = fmax(total_count - held_count, 0.0f);
        return true;
    }
    return false;
}

inline float aligned_qty_difference(float total, float held, float step) {
    float difference_steps;
    if (try_aligned_qty_difference_steps(total, held, step, difference_steps)) return difference_steps * step;
    return fmax(floor_step(total - held, step), 0.0f);
}

// Preserve the selected count for callers that must compare venue quantities
// without re-dividing the returned float32 minimum.
inline float min_entry_qty(
    float price, float qty_step, float min_qty, float min_cost, float c_mult,
    thread float* minimum_steps = nullptr
) {
    float raw_min = fmax(min_qty, min_cost / fmax(price, 1.0e-12f) / c_mult);
    float raw_steps = raw_min / qty_step;
    float nearest_count = floor(raw_steps + 0.5f);
    float nearest = nearest_count * qty_step;
    float representation_tolerance = 1.1920928955078125e-7f
        * fmax(fabs(raw_min), fabs(nearest)) * 4.0f;
    bool aligned = nearest_count > 0.0f && fabs(raw_steps - nearest_count) <= 1.0e-8f
        && (nearest >= raw_min || raw_min - nearest <= representation_tolerance);
    float steps = aligned ? nearest_count : ceil(raw_steps);
    if (minimum_steps != nullptr) *minimum_steps = steps;
    return aligned ? fmax(nearest, raw_min) : steps * qty_step;
}

inline bool should_use_ordinary_market_execution(
    int order_ticks,
    bool buy_order,
    float market_price,
    float price_step,
    bool market_orders_allowed,
    float near_touch_fraction
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
        <= fmax(near_touch_fraction, 0.0f);
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

// Causal close allocation for candle replay. A quoted limit remains maker
// liquidity in both replay engines. A market close uses the staged execution
// tick for PnL and the observed, unslipped touch for its venue minimum.
#pragma STDC FP_CONTRACT OFF
struct ReplayCloseProjection {
    float execution_price;
    float minimum_qty;
    float minimum_lots;
    float position_price;
    float c_mult;
    float fee_fraction;
    bool is_long;
    bool immediately_executable;
};

inline float replay_lot_count(float qty, float step, bool upward) {
    float ratio = qty / step;
    if (!isfinite(ratio)) return NAN;
    float nearest = rint(ratio);
    if (nearest * step == qty) return nearest;

    float count = upward ? ceil(ratio) : floor(ratio);
    float rounded = count * step;
    if (upward && rounded < qty) {
        count = ceil(nextafter(count, INFINITY));
    } else if (!upward && rounded > qty) {
        count = floor(nextafter(count, 0.0f));
    }
    return count;
}

inline bool project_replay_close(
    bool is_long, float position_price, float quote, float touch, bool use_market,
    int market_tick, float price_step, float qty_step, float min_qty, float min_cost,
    float c_mult, float maker_fee, float taker_fee, thread ReplayCloseProjection& projection
) {
    if (!(finite_positive(position_price) && finite_positive(quote)
          && finite_positive(touch) && finite_positive(price_step)
          && finite_positive(qty_step) && finite_positive(c_mult)
          && isfinite(min_qty) && min_qty >= 0.0f && isfinite(min_cost) && min_cost >= 0.0f
          && isfinite(maker_fee) && isfinite(taker_fee))
        || (use_market && market_tick <= 0)) return false;

    float price = use_market ? ordinary_market_fill_price(market_tick, price_step) : quote;
    float minimum_price = use_market ? touch : quote;
    float raw_minimum = fmax(min_qty, (min_cost / minimum_price) / c_mult);
    float lots = replay_lot_count(raw_minimum, qty_step, true);
    float minimum = lots * qty_step;
    if (!(finite_positive(price) && isfinite(lots) && isfinite(minimum))) return false;

    projection.execution_price = price;
    projection.minimum_qty = minimum;
    projection.minimum_lots = lots;
    projection.position_price = position_price;
    projection.c_mult = c_mult;
    projection.fee_fraction = use_market ? taker_fee : maker_fee;
    projection.is_long = is_long;
    projection.immediately_executable = use_market;
    return true;
}

inline float replay_close_net_pnl(ReplayCloseProjection projection, float qty) {
    float price_diff = projection.is_long
        ? projection.execution_price - projection.position_price
        : projection.position_price - projection.execution_price;
    // One net price increment makes the projection monotone in quantity.
    // Separately rounding gross and fee totals can reverse nearby lots when
    // the two almost cancel, breaking both integer searches.
    float net_price_move = fma(-projection.fee_fraction, projection.execution_price, price_diff);
    return (fabs(qty) * projection.c_mult) * net_price_move;
}

// Exact integer lot indexes where binary32 resolves every count. At very
// large sizes, search the ordered positive encodings of representable lot
// counts, as the device position state cannot distinguish adjacent raw lots.
inline uint replay_lot_index(float count, bool direct) {
    return direct ? uint(count) : as_type<uint>(count);
}

inline float replay_lots_at(uint index, bool direct) {
    return direct ? float(index) : floor(as_type<float>(index));
}

inline bool size_replay_close(
    float proposed_abs_qty, float position_abs_qty, float qty_step,
    ReplayCloseProjection projection, bool budget_enabled, float headroom,
    bool allow_below_minimum_full_close, thread float& qty
) {
    qty = 0.0f;
    if (!(finite_positive(proposed_abs_qty) && finite_positive(position_abs_qty)
          && finite_positive(qty_step))
        || (budget_enabled && !(isfinite(headroom) && headroom >= 0.0f))) return false;

    float upper = fmin(proposed_abs_qty, position_abs_qty);
    bool full_close_exception = allow_below_minimum_full_close
        && upper == position_abs_qty && upper < projection.minimum_qty;
    float lots = full_close_exception ? 0.0f : replay_lot_count(upper, qty_step, false);
    float upper_qty = full_close_exception ? upper : lots * qty_step;
    float least_lots = fmax(projection.minimum_lots, 1.0f);
    if (!(finite_positive(upper_qty) && upper_qty <= upper)
        || (!full_close_exception && lots < least_lots)) return false;

    float pnl = replay_close_net_pnl(projection, upper_qty);
    if (!isfinite(pnl)) return false;
    if (!budget_enabled || pnl >= -headroom) {
        qty = upper_qty;
        return true;
    }
    if (!(headroom > 0.0f)) return false;
    if (full_close_exception) {
        qty = upper_qty;
        return true;
    }

    // Compare the projected losses of integer lots directly. An exact hit
    // requires zero extra lots, even when division would round upward.
    bool direct = lots <= 0x1p24f;
    uint lo = replay_lot_index(least_lots, direct);
    uint hi = replay_lot_index(lots, direct);
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2u;
        float lot_qty = replay_lots_at(mid, direct) * qty_step;
        if (-replay_close_net_pnl(projection, lot_qty) >= headroom) hi = mid;
        else lo = mid + 1u;
    }
    qty = replay_lots_at(lo, direct) * qty_step;
    return true;
}

inline bool replay_unstuck_exposure_after(
    float position_abs_qty, float qty, ReplayCloseProjection projection,
    float balance, float effective_wel, thread float& ratio
) {
    float after = balance + replay_close_net_pnl(projection, qty);
    if (!(finite_positive(after) && qty >= 0.0f && qty < position_abs_qty)) return false;
    float budget = after * effective_wel;
    float entry_cost = projection.position_price * projection.c_mult;
    ratio = ((position_abs_qty - qty) * entry_cost) / budget;
    return finite_positive(budget) && isfinite(ratio);
}

inline bool upper_replay_unstuck_qty(
    float position_abs_qty, float qty_step, ReplayCloseProjection projection,
    float balance, float effective_wel, float trigger, thread float& qty
) {
    qty = 0.0f;
    float before;
    if (!(finite_positive(qty_step) && finite_positive(effective_wel) && trigger > 0.0f)
        || !replay_unstuck_exposure_after(
            position_abs_qty, 0.0f, projection, balance, effective_wel, before)
        || !(trigger < before)) return false;

    // The finite positive-balance branch is strictly decreasing only when
    // Q * pnl_per_unit + B > 0. Solve on integer lots against the actual
    // projected balance of each candidate rather than dividing by a near-zero
    // slope or allowing a full flatten.
    float direction = fma(position_abs_qty, replay_close_net_pnl(projection, 1.0f), balance);
    if (!finite_positive(direction)) return false;
    float lots = replay_lot_count(position_abs_qty, qty_step, false);
    if (!(lots >= 1.0f && isfinite(lots))) return false;
    bool direct = lots <= 0x1p24f;
    uint lo = 0u;
    uint hi = replay_lot_index(lots, direct);
    while (lo < hi) {
        uint mid = lo + (hi - lo + 1u) / 2u;
        float after;
        float lot_qty = replay_lots_at(mid, direct) * qty_step;
        if (replay_unstuck_exposure_after(
                position_abs_qty, lot_qty, projection, balance, effective_wel, after)
            && after >= trigger) lo = mid;
        else hi = mid - 1u;
    }

    float floor_qty = replay_lots_at(lo, direct) * qty_step;
    float projected_ratio;
    if (!(floor_qty > 0.0f)
        || !replay_unstuck_exposure_after(
            position_abs_qty, floor_qty, projection, balance, effective_wel, projected_ratio)
        || !(projected_ratio >= trigger && projected_ratio < before)) return false;
    qty = floor_qty;
    return true;
}

inline bool size_capped_replay_unstuck(
    float exposure_cap, float position_abs_qty, float qty_step, ReplayCloseProjection projection,
    float balance, float effective_wel, float trigger, float unstuck_headroom,
    bool gate_enabled, float gate_headroom, thread float& qty
) {
    qty = 0.0f;
    if (!finite_positive(unstuck_headroom)
        || (gate_enabled && !(isfinite(gate_headroom) && gate_headroom >= 0.0f))) return false;
    float headroom = gate_enabled ? fmin(unstuck_headroom, gate_headroom) : unstuck_headroom;
    float sized;
    float projected_ratio;
    float before;
    if (!size_replay_close(
            exposure_cap, position_abs_qty, qty_step, projection, true, headroom, false, sized)
        || !replay_unstuck_exposure_after(
            position_abs_qty, 0.0f, projection, balance, effective_wel, before)
        || !(trigger > 0.0f && trigger < before)
        || !replay_unstuck_exposure_after(
            position_abs_qty, sized, projection, balance, effective_wel, projected_ratio)
        || !(projected_ratio >= trigger && projected_ratio < before)) return false;
    qty = sized;
    return true;
}

inline bool size_replay_unstuck(
    float position_abs_qty, float qty_step, ReplayCloseProjection projection,
    float balance, float effective_wel, float trigger, float unstuck_headroom,
    bool gate_enabled, float gate_headroom, thread float& qty
) {
    qty = 0.0f;
    float exposure_cap;
    return upper_replay_unstuck_qty(
        position_abs_qty, qty_step, projection, balance, effective_wel, trigger, exposure_cap
    ) && size_capped_replay_unstuck(
        exposure_cap, position_abs_qty, qty_step, projection, balance, effective_wel,
        trigger, unstuck_headroom, gate_enabled, gate_headroom, qty
    );
}

struct ReplayClosePriority {
    bool immediately_executable;
    float resting_gap_rank;
    float signed_entry_distance_rank;
    int stable_coin_index;
    int position_side;
    int order_type;
    float price;
    float signed_qty;
};

inline float replay_close_priority_touch(float raw_touch, int touch_down, int touch_up, float price_step) {
    // Matching staged ticks prove alignment in the source price. Reconstruct
    // that touch like the equally priced quote so packing cannot invent a gap.
    return touch_down > 0 && touch_down == touch_up ? float(touch_down) * price_step : raw_touch;
}

// Same normalized priority lattice as the Rust close allocator. Every
// binary32 value with magnitude at least 16 is already on this lattice.
inline float replay_close_priority_rank(float fraction) {
    float rank = fabs(fraction) >= 16.0f
        ? fraction : round(fraction * 0x1p20f) * 0x1p-20f;
    return rank == 0.0f ? 0.0f : rank;
}

inline ReplayClosePriority replay_close_priority(
    ReplayCloseProjection projection, float quote, float touch, int stable_coin_index,
    int order_type, float proposed_abs_qty
) {
    float reference_price = projection.immediately_executable ? touch : quote;
    float entry_distance = (reference_price - projection.position_price) / projection.position_price;
    ReplayClosePriority result;
    result.immediately_executable = projection.immediately_executable;
    result.resting_gap_rank = projection.immediately_executable
        ? 0.0f : replay_close_priority_rank(fabs((quote - touch) / touch));
    result.signed_entry_distance_rank =
        replay_close_priority_rank(projection.is_long ? entry_distance : -entry_distance);
    result.stable_coin_index = stable_coin_index;
    result.position_side = projection.is_long ? 0 : 1;
    result.order_type = order_type;
    result.price = quote;
    result.signed_qty = projection.is_long ? -proposed_abs_qty : proposed_abs_qty;
    return result;
}

inline bool replay_close_preferred(ReplayClosePriority candidate, ReplayClosePriority incumbent) {
    if (candidate.immediately_executable != incumbent.immediately_executable)
        return candidate.immediately_executable;
    if (candidate.resting_gap_rank != incumbent.resting_gap_rank)
        return candidate.resting_gap_rank < incumbent.resting_gap_rank;
    if (candidate.signed_entry_distance_rank != incumbent.signed_entry_distance_rank)
        return candidate.signed_entry_distance_rank > incumbent.signed_entry_distance_rank;
    if (candidate.stable_coin_index != incumbent.stable_coin_index)
        return candidate.stable_coin_index < incumbent.stable_coin_index;
    if (candidate.position_side != incumbent.position_side)
        return candidate.position_side < incumbent.position_side;
    if (candidate.order_type != incumbent.order_type)
        return candidate.order_type < incumbent.order_type;
    if (candidate.price != incumbent.price)
        return candidate.price < incumbent.price;
    return candidate.signed_qty < incumbent.signed_qty;
}

struct ReplayLossBudget {
    float remaining;
    float error;
};

inline float replay_loss_headroom(ReplayLossBudget budget) {
    // Projected losses are representable float32 values. Round the quota up
    // for the first-lot comparison while retaining its residual for spending.
    return budget.error > 0.0f ? nextafter(budget.remaining, INFINITY) : budget.remaining;
}

inline float replay_budget_sum_error(float left, float right, float sum) {
    float right_high = sum - left;
    return (left - (sum - right_high)) + (right - right_high);
}

inline void spend_replay_loss_budget(thread ReplayLossBudget& budget, float net_pnl) {
    if (!(budget.remaining > 0.0f && net_pnl < 0.0f)) return;
    // Recover each subtraction's rounding error and renormalize the two
    // limbs. Repeated sub-ULP losses must deplete the same account quota.
    float difference = budget.remaining + net_pnl;
    float residual = budget.error + replay_budget_sum_error(budget.remaining, net_pnl, difference);
    float remaining = difference + residual;
    if (remaining > 0.0f) {
        budget.remaining = remaining;
        budget.error = replay_budget_sum_error(difference, residual, remaining);
    } else {
        budget.remaining = 0.0f;
        budget.error = 0.0f;
    }
}
#pragma STDC FP_CONTRACT ON

inline int multicoin_interval_minutes(float interval_ms) {
    return max(1, int(interval_ms / 60000.0f + 0.5f));
}

inline bool multicoin_tail_indicates_delisting(
    int last_valid, int timestep_count, float interval_ms
) {
    return timestep_count - 1 - last_valid > 1440 / multicoin_interval_minutes(interval_ms);
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
    RealizedDecayMemory realized_memory;
    int realized_decay_interval_minutes;
    bool realized_decay_valid;
#if PASSIVBOT_CORE_REQUEST_MASK != 0
    CoreGpuState core;
#endif
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
    account.realized_memory.beta = 0.0f;
    account.realized_memory.drawdown = 0.0f;
    account.realized_memory.minute = 0;
    account.realized_memory.enabled = false;
    account.realized_decay_interval_minutes = 1;
    account.realized_decay_valid = true;
#if PASSIVBOT_CORE_REQUEST_MASK != 0
    account.core = init_core_gpu(starting_balance, 0);
#endif
    return account;
}

// Each actual account net fill updates the shared realized memory once.
inline bool joint_realized_loss_allowance(
    thread JointPortfolioAccount& account, float fraction, thread float& allowance
) {
    if (!hsl_decay_realized_allowance(
            account.balance, fraction, account.realized_memory.drawdown, allowance)) {
        account.realized_decay_valid = false;
        return false;
    }
    return true;
}

inline void configure_joint_realized_memory(
    thread JointPortfolioAccount& account, float memory_span,
    bool enabled, int interval_minutes
) {
    if (enabled) {
        float alpha;
        account.realized_decay_valid = account.realized_decay_valid
            && hsl_decay_rate(memory_span, alpha, account.realized_memory.beta);
    }
    account.realized_memory.enabled = enabled;
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
    advance_realized_memory(account.realized_memory, minute, account.realized_decay_valid);
}
#pragma STDC FP_CONTRACT ON

inline void record_joint_portfolio_fill(
    thread JointPortfolioAccount& account,
    float net_pnl,
    bool is_long
) {
    if (account.realized_memory.enabled) {
        account.realized_decay_valid = account.realized_decay_valid
            && hsl_decay_realized_fill(net_pnl, account.realized_memory.drawdown);
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
    thread float& fill_count,
    thread float& fill_count_entry,
    thread float& fill_count_long,
    float fill_k,
    bool is_entry,
    bool is_long
) {
#if PASSIVBOT_CORE_REQUEST_MASK != 0
    account.core.pre_fill_balance = account.balance;
    core_record_fill(account.core, int(fill_k), net_pnl, is_long);
#endif
    advance_joint_realized_memory(account, int(fill_k));
    record_joint_portfolio_fill(account, net_pnl, is_long);
    fill_count += 1.0f;
    if (is_entry) fill_count_entry += 1.0f;
    if (is_long) fill_count_long += 1.0f;

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
inline float allowed_wallet_exposure_limit_ratio(
    float base_limit, float total_limit, float allowance_fraction) {
    if (!(isfinite(base_limit) && base_limit > 0.0f)) return 0.0f;
    float raw = fmax(allowance_fraction, 0.0f);
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
    return allowed_wallet_exposure_limit_ratio(base, twel,
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

// Age is initial-entry admission, independent of indicator/data readiness.
// Columns are exact replay indices prepared by the Rust age predicate.
inline bool passes_multicoin_age(
    constant float* coin_settings, int coin_offset, int k, bool short_side
) {
    return k >= int(coin_settings[coin_offset + (short_side ? 14 : 13)]);
}

inline bool passes_multicoin_min_effective_cost(
    bool enabled, float balance, float wel,
    float initial_qty_fraction, float price,
    constant float* coin_settings ORGANILLO_ARGS, int coin_offset
) {
    if (!enabled) return true;
    if (!finite_positive(price)) return false;
    const float c_mult = coin_settings[coin_offset + 4];
    const float minimum_cost = min_entry_qty(
        price, coin_settings[coin_offset + 0], coin_settings[coin_offset + 2],
        coin_settings[coin_offset + 3], c_mult
    ) * price * c_mult;
    const float projected_cost = balance * wel * initial_qty_fraction;
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
    float base_wel[2 * MAX_COINS];
};
inline ExposureRmsState init_exposure_rms() {
    ExposureRmsState state;
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



#endif
