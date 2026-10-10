// Compose a fresh, causally clipped scope from reconstructed pair facts. No
// previous panic decision or reconstructed curve is accepted as authority.
struct HslScopePair {
    HslPairFacts facts;
    HslPairHistory history;
    device HslPairEvent* events;
    device const int* sequences; // Global sequence, with a caller-owned stride.
    int sequence_stride;
    device const float* prices; // Aligned minute marks for this exact snapshot.
    int price_stride;
    constant float* candles; // Native aligned close grid; null for prepared prices.
    int candle_stride;
    int candle_first;
    int candle_last;
    float fallback_price;
    float current_size;
    float current_basis;
    float current_mark;
    float multiplier;
    bool short_side;
    int consumed;
    float boundary_size;
};

inline float hsl_scope_price_at(thread const HslScopePair& pair, int minute, int start, int end) {
    if (pair.candles == nullptr) return pair.prices[(minute - start) * pair.price_stride];
    // Candles name their open; close j becomes available at minute j+1.
    // Projection is evaluation-local: forward-fill gaps, then backfill a missing
    // prefix from the first causal close. Never read the end minute's future bar.
    int first = max(max(start - 1, 0), pair.candle_first);
    int last = min(end - 1, pair.candle_last);
    for (int j = min(max(minute - 1, 0), last); j >= first; --j) {
        float value = pair.candles[j * pair.candle_stride];
        if (isfinite(value) && value > 0.0f) return value;
    }
    for (int j = first; j <= last; ++j) {
        float value = pair.candles[j * pair.candle_stride];
        if (isfinite(value) && value > 0.0f) return value;
    }
    return pair.fallback_price;
}

struct HslScopePoint {
    int minute;
    float pnl; // Centered against the common current factual cashflow prefix.
    float upnl;
    int exposed;
    int flatten;
    float raw;
    float ema;
    int action; // 0 normal, 1 halted, 3 panic; same as the native controller.
};

struct HslScopeResult {
    float raw;
    float ema;
    int action;
    int flat_minute;
    int point_count;
    int latest_flat_minute;
    float latest_flat_raw;
    float latest_flat_ema;
};

struct HslScopeSignal {
    float peak;
    float raw;
    float ema;
    float baseline;
    float reference;
    int minute;
    bool ready;
    bool baseline_ready;
};

// Disposable scalar continuation of a known exposed episode. The caller must
// independently prove unchanged factual inputs; no cached permission is stored.
struct HslScopeCursor {
    float peak;
    float ema;
    float pnl;
    float budget;
    float alpha;
    int minute;
    int start;
    int point_count;
    bool valid;
};

inline void hsl_scope_signal_reset(thread HslScopeSignal& s, float reference) {
    s.peak = -INFINITY;
    s.raw = s.ema = s.baseline = 0.0f;
    s.reference = reference;
    s.minute = -1;
    s.ready = s.baseline_ready = false;
}

inline float hsl_scope_cash_difference(
    thread const HslPairSum& prefix, thread const HslPairSum& anchor
) {
    HslPairSum centered;
    hsl_pair_sum_reset(centered, 0.0f);
    hsl_pair_sum_add(centered, prefix.value);
    hsl_pair_sum_add(centered, prefix.correction);
    hsl_pair_sum_add(centered, -anchor.value);
    hsl_pair_sum_add(centered, -anchor.correction);
    return hsl_pair_sum_value(centered);
}

inline bool hsl_scope_visit(
    thread HslScopeSignal& signal, float budget, float alpha, float threshold,
    float cooldown, bool never_restart, int start, bool reopened,
    thread HslScopePoint& point, thread HslScopeResult& result,
    device HslScopePoint* trace, int trace_capacity
) {
    float delta = point.pnl + point.upnl;
    if (!isfinite(delta)) return false;
    if (signal.minute != point.minute) {
        signal.baseline = signal.ema;
        signal.baseline_ready = signal.ready;
    }
    if (!signal.ready) signal.peak = fmax(signal.reference, delta);
    else signal.peak = fmax(signal.peak, delta);
    float denominator = budget + signal.peak;
    signal.raw = denominator > 0.0f ? (signal.peak - delta) / denominator : 1.0f;
    signal.ema = signal.baseline_ready
        ? alpha * signal.raw + (1.0f - alpha) * signal.baseline : signal.raw;
    if (!isfinite(signal.raw) || !isfinite(signal.ema)) return false;
    signal.ready = true;
    signal.minute = point.minute;
    if (point.exposed || reopened) result.flat_minute = -1;
    bool red = fmin(signal.raw, signal.ema) > threshold;
    if (point.flatten) {
        result.flat_minute = red ? point.minute : -1;
        result.latest_flat_minute = point.minute;
        result.latest_flat_raw = signal.raw;
        result.latest_flat_ema = signal.ema;
    }
    if (result.flat_minute >= 0 && (result.flat_minute < start
        || (!never_restart && float(point.minute - result.flat_minute) >= cooldown)))
        result.flat_minute = -1;
    result.raw = point.raw = signal.raw;
    result.ema = point.ema = signal.ema;
    result.action = point.action = point.exposed && red ? 3
        : (!point.exposed && result.flat_minute >= 0 ? 1 : 0);
    if (trace != nullptr) {
        if (result.point_count >= trace_capacity) return false;
        trace[result.point_count] = point;
    }
    ++result.point_count;
    return true;
}

// Native candles have evaluation-local forward/backfill. With no retained
// execution in any selected pair, reconstruction is flat until the actual
// exposed endpoint; fresh composition replaces that curve by this singleton.
// Reuse the same signal visitor, never yesterday's peak, EMA or permission.
inline bool hsl_compose_empty_native_scope(
    thread HslScopePair* pairs, int pair_count, int start, int end,
    float budget, float span, float threshold, float cooldown, bool never_restart,
    thread HslScopeResult& result
) {
    if (pair_count < 1 || end < start || !isfinite(budget) || budget <= 0.0f
        || !isfinite(span) || span < 1.0f || !isfinite(threshold)
        || threshold <= 0.0f || threshold > 1.0f || !isfinite(cooldown)
        || cooldown < 0.0f) return false;
    HslPairSum upnl;
    hsl_pair_sum_reset(upnl, 0.0f);
    bool exposed = false;
    for (int p = 0; p < pair_count; ++p) {
        thread const HslScopePair& pair = pairs[p];
        if (pair.facts.count != 0 || pair.candles == nullptr
            || pair.history.opening_size != 0.0f || pair.history.flat_correction_minute >= 0
            || pair.sequence_stride < 1 || pair.price_stride < 1) return false;
        HslPairSample sample;
        if (!hsl_sample_pair(pair.history, pair.events, 0, 0, end, end, false,
            pair.current_size, pair.current_basis, pair.current_mark,
            hsl_scope_price_at(pair, end, start, end), pair.multiplier,
            pair.short_side, sample)) return false;
        exposed = exposed || sample.size != 0.0f;
        hsl_pair_sum_add(upnl, sample.upnl);
    }
    if (!exposed) return false;
    result.raw = result.ema = result.latest_flat_raw = result.latest_flat_ema = 0.0f;
    result.action = result.point_count = 0;
    result.flat_minute = result.latest_flat_minute = -1;
    HslScopeSignal signal;
    hsl_scope_signal_reset(signal, 0.0f);
    HslScopePoint point;
    point.minute = end; point.pnl = 0.0f; point.upnl = hsl_pair_sum_value(upnl);
    point.exposed = 1; point.flatten = 0;
    return hsl_scope_visit(signal, budget, 2.0f / (span + 1.0f), threshold,
        cooldown, never_restart, start, false, point, result, nullptr, 0);
}

inline int hsl_scope_pair_sequence(thread const HslScopePair& pair, int logical) {
    return pair.sequences[hsl_pair_slot(pair.facts, logical) * pair.sequence_stride];
}

inline bool hsl_advance_scope(
    thread HslScopeCursor& cursor, int start, int minute, float upnl,
    float budget, float alpha, float threshold, float cooldown, bool never_restart,
    thread HslScopeResult& result
) {
    if (!cursor.valid || start != cursor.start || minute != cursor.minute + 1
        || budget != cursor.budget || alpha != cursor.alpha
        || !isfinite(upnl) || !isfinite(budget) || budget <= 0.0f
        || !isfinite(alpha) || alpha <= 0.0f || alpha > 1.0f
        || !isfinite(threshold) || threshold <= 0.0f || threshold > 1.0f
        || !isfinite(cooldown) || cooldown < 0.0f) return false;
    HslScopeSignal signal;
    signal.peak = cursor.peak;
    signal.raw = 0.0f;
    signal.ema = signal.baseline = cursor.ema;
    signal.reference = -INFINITY;
    signal.minute = cursor.minute;
    signal.ready = signal.baseline_ready = true;
    float denominator = budget + fmax(signal.peak, cursor.pnl + upnl);
    if (!isfinite(denominator) || denominator <= 0.0f) return false;
    result.raw = result.ema = result.latest_flat_raw = result.latest_flat_ema = 0.0f;
    result.action = 0;
    result.flat_minute = result.latest_flat_minute = -1;
    result.point_count = cursor.point_count;
    HslScopePoint point;
    point.minute = minute; point.pnl = cursor.pnl; point.upnl = upnl;
    point.exposed = 1; point.flatten = 0;
    if (!hsl_scope_visit(signal, budget, alpha, threshold, cooldown, never_restart,
        start, false, point, result, nullptr, 0)) return false;
    // Fresh replay adjudicates cancellation-sensitive policy comparisons.
    float score = fmin(signal.raw, signal.ema);
    if (fabs(score - threshold) <= 16.0f * 1.1920928955078125e-7f
        * fmax(1.0f, fabs(score))) return false;
    cursor.peak = signal.peak; cursor.ema = signal.ema;
    cursor.minute = minute; cursor.point_count = result.point_count;
    return true;
}

inline int hsl_scope_next_opening(thread HslScopePair* pairs, int pair_count) {
    int opening = -1;
    for (int p = 0; p < pair_count; ++p) {
        thread HslScopePair& pair = pairs[p];
        for (int i = pair.consumed; i < pair.facts.count; ++i) {
            if (pair.events[i].before == 0.0f && pair.events[i].after > 0.0f) {
                int minute = hsl_pair_minute(pair.facts, i);
                opening = opening < 0 ? minute : min(opening, minute);
                break;
            }
        }
    }
    return opening;
}

inline bool hsl_compose_scope(
    thread HslScopePair* pairs, int pair_count, int start, int end,
    bool before_price, float budget, float span, float threshold,
    float cooldown, bool never_restart, thread HslScopeResult& result,
    device HslScopePoint* trace = nullptr, int trace_capacity = 0,
    thread HslScopeCursor* cursor = nullptr
) {
    if (cursor != nullptr) cursor->valid = false;
    result.raw = result.ema = result.latest_flat_raw = result.latest_flat_ema = 0.0f;
    result.action = result.point_count = 0;
    result.flat_minute = result.latest_flat_minute = -1;
    if (pair_count < 1 || end < start || !isfinite(budget) || budget <= 0.0f
        || !isfinite(span) || span < 1.0f || !isfinite(threshold)
        || threshold <= 0.0f || threshold > 1.0f || !isfinite(cooldown)
        || cooldown < 0.0f) return false;
    HslPairSum anchor, cash;
    hsl_pair_sum_reset(anchor, 0.0f);
    hsl_pair_sum_reset(cash, 0.0f);
    bool estimated = false, current_exposed = false, reconstructed_flat = true;
    for (int p = 0; p < pair_count; ++p) {
        thread HslScopePair& pair = pairs[p];
        if (pair.sequence_stride < 1 || pair.price_stride < 1) return false;
        pair.consumed = 0;
        pair.boundary_size = fabs(pair.history.opening_size);
        estimated = estimated || pair.history.opening_size != 0.0f;
        current_exposed = current_exposed || pair.current_size != 0.0f;
        reconstructed_flat = reconstructed_flat && (pair.facts.count == 0
            ? pair.history.opening_size == 0.0f
            : pair.events[pair.facts.count - 1].after == 0.0f);
        for (int i = 0; i < pair.facts.count; ++i) {
            HslPairFact f = hsl_pair_fact(pair.facts, i);
            hsl_pair_sum_add(anchor, f.realized);
            hsl_pair_sum_add(anchor, f.fee);
        }
    }
    if (!isfinite(hsl_pair_sum_value(anchor))) return false;
    HslScopeSignal signal;
    hsl_scope_signal_reset(signal, estimated ? -hsl_pair_sum_value(anchor) : -INFINITY);
    float alpha = 2.0f / (span + 1.0f);
    int episode_trace_start = 0;
    int episode_opening = -1;
    float final_pnl = 0.0f;
    for (int minute = start; minute <= end; ++minute) {
        while (true) {
            int selected = -1, selected_minute = 0, sequence = 0;
            for (int p = 0; p < pair_count; ++p) {
                thread HslScopePair& pair = pairs[p];
                int i = pair.consumed;
                if (i >= pair.facts.count) continue;
                int t = hsl_pair_minute(pair.facts, i);
                if (!hsl_pair_fill_precedes_price(t, minute, end, before_price)) continue;
                int seq = hsl_scope_pair_sequence(pair, i);
                if (selected < 0 || t < selected_minute
                    || (t == selected_minute && seq < sequence)) {
                    selected = p; selected_minute = t; sequence = seq;
                }
            }
            if (selected < 0) break;
            thread HslScopePair& pair = pairs[selected];
            int i = pair.consumed++;
            HslPairEvent event = pair.events[i];
            bool had_exposure = event.before > 0.0f || event.after > 0.0f;
            for (int p = 0; p < pair_count; ++p)
                had_exposure = had_exposure || pairs[p].boundary_size != 0.0f;
            pair.boundary_size = i + 1 == pair.facts.count
                && pair.history.flat_correction_minute >= 0 ? 0.0f : event.after;
            HslPairFact fact = hsl_pair_fact(pair.facts, i);
            hsl_pair_sum_add(cash, fact.realized);
            hsl_pair_sum_add(cash, fact.fee);
            bool flat = true;
            for (int p = 0; p < pair_count; ++p)
                flat = flat && pairs[p].boundary_size == 0.0f;
            if (flat && had_exposure) {
                HslScopePoint point;
                point.minute = selected_minute;
                point.pnl = hsl_scope_cash_difference(cash, anchor);
                point.upnl = 0.0f;
                point.exposed = 0; point.flatten = 1;
                bool reopened = result.point_count > episode_trace_start
                    && episode_opening >= 0 && point.minute >= episode_opening;
                if (!hsl_scope_visit(signal, budget, alpha, threshold, cooldown,
                    never_restart, start, reopened, point, result, trace, trace_capacity))
                    return false;
                // A genuine flat seeds the following episode at exactly the
                // consumed cashflow prefix, including its closing fee.
                hsl_scope_signal_reset(signal, -INFINITY);
                episode_trace_start = result.point_count;
                // Lifecycle uses the causal fill time, even when the caller's
                // candle phase samples inventory before a same-time fill.
                episode_opening = hsl_scope_next_opening(pairs, pair_count);
                point.flatten = 0;
                if (!hsl_scope_visit(signal, budget, alpha, threshold, cooldown,
                    never_restart, start, false, point, result, trace, trace_capacity)) return false;
            }
        }
        HslPairSum upnl;
        hsl_pair_sum_reset(upnl, 0.0f);
        bool exposed = false;
        for (int p = 0; p < pair_count; ++p) {
            thread HslScopePair& pair = pairs[p];
            HslPairSample sample;
            float price = hsl_scope_price_at(pair, minute, start, end);
            if (!hsl_sample_pair(pair.history, pair.events, pair.consumed,
                pair.facts.count, minute, end, before_price, pair.current_size,
                pair.current_basis, pair.current_mark, price, pair.multiplier,
                pair.short_side, sample)) return false;
            hsl_pair_sum_add(upnl, sample.upnl);
            exposed = exposed || sample.size != 0.0f;
        }
        if (minute == end && current_exposed && reconstructed_flat) {
            // Current positions prove a new opening absent from the retained
            // tail. Its scoped entry value seeds one real current observation.
            hsl_scope_signal_reset(signal, 0.0f);
            result.point_count = episode_trace_start;
        }
        HslScopePoint point;
        point.minute = minute;
        point.pnl = hsl_scope_cash_difference(cash, anchor);
        point.upnl = hsl_pair_sum_value(upnl);
        if (cursor != nullptr && minute == end) final_pnl = point.pnl;
        point.exposed = exposed ? 1 : 0; point.flatten = 0;
        bool reopened = result.point_count > episode_trace_start
            && episode_opening >= 0 && minute >= episode_opening;
        if (!hsl_scope_visit(signal, budget, alpha, threshold, cooldown,
            never_restart, start, reopened, point, result, trace, trace_capacity)) return false;
        // A completed flat episode seeds an exact zero signal. With every fill
        // consumed and no current exposure, later idle marks cannot change it.
        // Keep the logical observation count, then visit the real endpoint so
        // cooldown/lookback expiry still uses current time. Debug traces retain
        // every observation; this is evaluation-local compaction, not a cache.
        if (trace == nullptr && !current_exposed && !exposed
            && signal.raw == 0.0f && signal.ema == 0.0f && minute + 1 < end) {
            bool complete = true;
            for (int p = 0; p < pair_count; ++p)
                complete = complete && pairs[p].consumed == pairs[p].facts.count
                    && pairs[p].boundary_size == 0.0f;
            if (complete) {
                result.point_count += end - minute - 1;
                minute = end - 1;
            }
        }
    }
    if (cursor != nullptr) {
        bool complete = true, stable_prefix = true;
        for (int p = 0; p < pair_count; ++p) {
            complete = complete && pairs[p].consumed == pairs[p].facts.count;
            // Today's endpoint becomes a historical sample on the next call.
            // Same-minute fills shift to the causal following price sample, and
            // reconstructed inventory/basis can differ from the actual endpoint.
            // Continue only when that reinterpretation cannot revise this sample.
            thread const HslScopePair& pair = pairs[p];
            if (pair.facts.count > 0) {
                int last = pair.facts.count - 1;
                stable_prefix = stable_prefix && hsl_pair_minute(pair.facts, last) < end
                    && pair.events[last].after == fabs(pair.current_size)
                    && pair.events[last].basis == pair.current_basis;
            }
            // Until a causal close exists, a later first quote can revise the
            // evaluation-local backfill of earlier marks. Keep that case fresh.
            if (pair.candles != nullptr) {
                int bar = end - 1;
                bool available = bar >= pair.candle_first && bar <= pair.candle_last;
                float close = available ? pair.candles[bar * pair.candle_stride] : 0.0f;
                stable_prefix = stable_prefix && available && isfinite(close) && close > 0.0f;
            }
        }
        float denominator = budget + signal.peak;
        cursor->valid = current_exposed && !estimated && !reconstructed_flat
            && !before_price && complete && stable_prefix && signal.ready
            && result.flat_minute < 0 && result.latest_flat_minute < 0
            && isfinite(signal.peak) && isfinite(final_pnl)
            && isfinite(denominator) && denominator > 0.0f;
        cursor->peak = signal.peak; cursor->ema = signal.ema;
        cursor->pnl = final_pnl; cursor->budget = budget; cursor->alpha = alpha;
        cursor->minute = end; cursor->start = start;
        cursor->point_count = result.point_count;
    }
    return true;
}
