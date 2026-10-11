// Unified candidate-owned cache; execution rings and candles remain authority.
// Disabled builds retain the independent factual composer with no cache state.
#if PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
#ifndef PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY
#define PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY 0
#endif
#define HSL_NATIVE_CACHE_BLOCK 64
#ifndef PASSIVBOT_HSL_NATIVE_CACHE_FLAT_SUFFIX_ENABLED
#define PASSIVBOT_HSL_NATIVE_CACHE_FLAT_SUFFIX_ENABLED 1
#endif

// Guarded controller arithmetic over the reconciled current factual episode.
// Inputs must already be reconciled and grouped within ONE factual episode:
// maximum sees every same-minute boundary, final sees only its last sample.
// This index cannot prove history completeness or authorize clipped replay.
struct HslBlockSample {
    float maximum;
    float final;
    int minute;
    int episode;
};

struct HslBlockGroup {
    float peak;
    float prefix_weight;
    float prefix_loss; // Sum w*(this prefix's peak-Q); always nonnegative.
    float loss; // Sum w*(this group's peak-Q); always nonnegative.
    int first;
    int count;
    int next_loss; // First nonzero-loss group at/after this group, or groups.
};

struct HslBlockSummary {
    int first;
    int count;
    int groups;
    int episode;
    float alpha;
    float decay;
    float minimum;
    int valid;
};

struct HslBlockAnswer {
    float peak;
    float raw;
    float ema;
    float error;
    int scalar_samples;
    int full_blocks;
    int positive_terms;
    int fallback; // 0 indexed; 1 absent/invalid cache; 2 conditioning; 3 threshold.
};
static_assert(sizeof(HslBlockSample) == 16, "Grouped fact layout");
static_assert(sizeof(HslBlockGroup) == 28, "Peak plateau layout");
static_assert(sizeof(HslBlockSummary) == 32, "Block layout");

inline float hsl_block_raw(float peak, float q, float k) {
    float denominator = k + peak;
    return denominator > 0.0f ? (peak - q) / denominator : 1.0f;
}

inline bool hsl_block_build(
    device const HslBlockSample* samples, int available, int first, int count,
    float alpha, device HslBlockGroup* groups, int capacity,
    device HslBlockSummary& summary
) {
    summary.valid = 0;
    if (samples == nullptr || groups == nullptr || first < 0 || count < 1
        || first > available - count || capacity < count || !isfinite(alpha)
        || alpha <= 0.0f || alpha > 1.0f) return false;
    float peak = -INFINITY, minimum = INFINITY;
    int n = 0, episode = samples[first].episode;
    for (int i = first; i < first + count; ++i) {
        HslBlockSample sample = samples[i];
        if (!isfinite(sample.maximum) || !isfinite(sample.final)
            || sample.maximum < sample.final || sample.episode != episode
            || (i > first && sample.minute <= samples[i - 1].minute)) return false;
        minimum = fmin(minimum, sample.final);
        if (sample.maximum > peak) {
            peak = sample.maximum;
            HslBlockGroup group;
            group.peak = peak; group.first = i; group.count = 0;
            group.prefix_weight = group.prefix_loss = group.loss = 0.0f;
            groups[n++] = group;
        }
        ++groups[n - 1].count;
    }
    // Reverse generation avoids pow per sample and an underflowed first
    // weight which cannot be recovered by division. Span 1 also works here.
    float beta = 1.0f - alpha, weight = alpha;
    for (int g = n - 1; g >= 0; --g) {
        HslPairSum weights, losses;
        hsl_pair_sum_reset(weights, 0.0f); hsl_pair_sum_reset(losses, 0.0f);
        for (int i = groups[g].first + groups[g].count - 1; i >= groups[g].first; --i) {
            hsl_pair_sum_add(weights, weight);
            hsl_pair_sum_add(losses, weight * (groups[g].peak - samples[i].final));
            weight *= beta;
        }
        groups[g].prefix_weight = hsl_pair_sum_value(weights);
        groups[g].loss = hsl_pair_sum_value(losses);
    }
    HslPairSum weights, losses;
    hsl_pair_sum_reset(weights, 0.0f); hsl_pair_sum_reset(losses, 0.0f);
    for (int g = 0; g < n; ++g) {
        if (g > 0) hsl_pair_sum_add(losses,
            (groups[g].peak - groups[g - 1].peak) * hsl_pair_sum_value(weights));
        hsl_pair_sum_add(weights, groups[g].prefix_weight);
        hsl_pair_sum_add(losses, groups[g].loss);
        groups[g].prefix_weight = hsl_pair_sum_value(weights);
        groups[g].prefix_loss = hsl_pair_sum_value(losses);
        if (!isfinite(groups[g].loss) || !isfinite(groups[g].prefix_loss)
            || !isfinite(groups[g].prefix_weight)) return false;
    }
    int next_loss = n;
    for (int g = n - 1; g >= 0; --g) {
        if (groups[g].loss != 0.0f) next_loss = g;
        groups[g].next_loss = next_loss;
    }
    summary.first = first; summary.count = count; summary.groups = n;
    summary.episode = episode; summary.alpha = alpha;
    summary.minimum = minimum;
    summary.decay = pow(beta, float(count));
    summary.valid = 1;
    return true;
}

inline int hsl_block_upper_bound(device const HslBlockGroup* groups, int n, float peak) {
    int low = 0, high = n;
    while (low < high) {
        int middle = low + (high - low) / 2;
        if (groups[middle].peak <= peak) low = middle + 1;
        else high = middle;
    }
    return low;
}

inline bool hsl_block_denominator_conditioned(float k, float peak) {
    float denominator = k + peak;
    // Addition close to zero can amplify currency rounding into arbitrary
    // ratios, or change the raw=1 branch. Recompute the scalar reference.
    float uncertainty = 64.0f * 0x1p-24f * (fabs(k) + fabs(peak) + 1.0f);
    return isfinite(denominator) && fabs(denominator) > uncertainty;
}

inline bool hsl_block_apply(
    device const HslBlockGroup* groups, HslBlockSummary summary, float k,
    thread HslBlockAnswer& answer
) {
    const float incoming = answer.peak;
    int prefix = hsl_block_upper_bound(groups, summary.groups, incoming);
    int nonpositive = max(prefix, hsl_block_upper_bound(groups, summary.groups, -k));
    HslPairSum contribution;
    hsl_pair_sum_reset(contribution, 0.0f);
    // Positive weighted arithmetic scales with the raw/EMA magnitude, not with
    // a unit-sized drawdown. A tiny floor also covers underflow/flush concerns.
    float bound = 0x1p-100f;
    if (prefix > 0) {
        if (!hsl_block_denominator_conditioned(k, incoming)) return false;
        HslBlockGroup group = groups[prefix - 1];
        if (k + incoming <= 0.0f) {
            hsl_pair_sum_add(contribution, group.prefix_weight);
            bound = 1.0f;
        }
        else {
            // Both terms are nonnegative: no subtraction of large moments.
            float numerator = (incoming - group.peak) * group.prefix_weight + group.prefix_loss;
            hsl_pair_sum_add(contribution, numerator / (k + incoming));
            bound = fmax(bound, (incoming - summary.minimum) / (k + incoming));
        }
    }
    if (nonpositive > prefix) {
        // Guard the branch boundary, including zero-loss groups.
        if (!hsl_block_denominator_conditioned(k, groups[nonpositive - 1].peak)) return false;
        float before = prefix == 0 ? 0.0f : groups[prefix - 1].prefix_weight;
        hsl_pair_sum_add(contribution, groups[nonpositive - 1].prefix_weight - before);
        bound = fmax(bound, 1.0f);
    }
    if (nonpositive < summary.groups) {
        // Positive denominators increase with group peak. The first is the
        // worst-conditioned one; later zero-loss groups need no per-group work.
        if (!hsl_block_denominator_conditioned(k, groups[nonpositive].peak)) return false;
        bound = fmax(bound, (fmax(incoming, groups[summary.groups - 1].peak)
            - summary.minimum) / (k + groups[nonpositive].peak));
        for (int g = groups[nonpositive].next_loss; g < summary.groups;
            g = g + 1 < summary.groups ? groups[g + 1].next_loss : summary.groups) {
            hsl_pair_sum_add(contribution, groups[g].loss / (k + groups[g].peak));
            ++answer.positive_terms;
        }
    }
    float old_ema = answer.ema;
    answer.ema = summary.decay * old_ema + hsl_pair_sum_value(contribution);
    answer.peak = fmax(incoming, groups[summary.groups - 1].peak);
    // Deliberately loose component error allowance for regrouping weights,
    // positive moments and divisions versus a scalar float32 recurrence.
    // It is an experimental guard, not yet a qualified native risk bound.
    float operations = 64.0f * float(summary.count);
    float gamma = operations * 0x1p-24f;
    if (gamma >= 0.25f || !isfinite(answer.ema) || !isfinite(bound)) return false;
    float local_error = gamma / (1.0f - gamma)
        * fmax(bound, fmax(fabs(old_ema), fabs(answer.ema)));
    answer.error = summary.decay * answer.error + local_error;
    ++answer.full_blocks;
    return true;
}

struct HslNativeCacheFlat { int minute; int sequence; float cash; };
struct HslNativeCacheHeader {
    int valid;
    int minute;
    int first;
    int pair_count;
    HslScopeCutoff cutoff;
    HslPairSum cash;
    float alpha;
    HslNativeCacheFlat latest_flat;
    int reserved[11];
};
struct HslNativeCachePair {
    HslNativePairCursor cursor;
    HslPairHistory history;
    float current_size;
    float current_basis;
    HslPairRecord last_record;
    int head;
    int count;
    int first_sequence;
    int last_sequence;
    float quantity_step;
    float multiplier;
    int pair_id;
    int valid;
    int reserved[2];
};
static_assert(sizeof(HslNativeCacheHeader) == 96, "Unified cache header layout");
static_assert(sizeof(HslNativeCachePair) == 128, "Pair cache header layout");
static_assert(sizeof(HslNativeCacheFlat) == 12, "Ordered scope flat layout");

inline int hsl_native_cache_align(int bytes) { return (bytes + 15) / 16 * 16; }
inline int hsl_native_cache_bytes(int pair_capacity, int fact_capacity) {
    int width = PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY;
    if (width < 1 || pair_capacity < 1 || fact_capacity < 1) return 0;
    return hsl_native_cache_align(sizeof(HslNativeCacheHeader))
        + hsl_native_cache_align(pair_capacity * sizeof(HslNativeCachePair))
        + hsl_native_cache_align(width * sizeof(HslBlockSample))
        + hsl_native_cache_align(width * sizeof(HslPairSum))
        + hsl_native_cache_align(width * sizeof(float))
        + hsl_native_cache_align(width * sizeof(HslBlockGroup))
        + hsl_native_cache_align((width / HSL_NATIVE_CACHE_BLOCK) * sizeof(HslBlockSummary))
        + hsl_native_cache_align(fact_capacity * sizeof(HslPairEvent));
}

struct HslNativeCache {
    device HslNativeCacheHeader* header;
    device HslNativeCachePair* pairs;
    device HslBlockSample* samples;
    device HslPairSum* upnl;
    device float* cash;
    device HslBlockGroup* groups;
    device HslBlockSummary* summaries;
    device HslPairEvent* temporary;
    int capacity;
    int pair_capacity;
};
inline HslNativeCache hsl_native_cache_view(device int* words, int pair_capacity, int fact_capacity) {
    HslNativeCache c;
    device char* at = reinterpret_cast<device char*>(words);
    c.capacity = PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY;
    c.pair_capacity = pair_capacity;
    c.header = reinterpret_cast<device HslNativeCacheHeader*>(at);
    at += hsl_native_cache_align(sizeof(HslNativeCacheHeader));
    c.pairs = reinterpret_cast<device HslNativeCachePair*>(at);
    at += hsl_native_cache_align(pair_capacity * sizeof(HslNativeCachePair));
    c.samples = reinterpret_cast<device HslBlockSample*>(at);
    at += hsl_native_cache_align(c.capacity * sizeof(HslBlockSample));
    c.upnl = reinterpret_cast<device HslPairSum*>(at);
    at += hsl_native_cache_align(c.capacity * sizeof(HslPairSum));
    c.cash = reinterpret_cast<device float*>(at);
    at += hsl_native_cache_align(c.capacity * sizeof(float));
    c.groups = reinterpret_cast<device HslBlockGroup*>(at);
    at += hsl_native_cache_align(c.capacity * sizeof(HslBlockGroup));
    c.summaries = reinterpret_cast<device HslBlockSummary*>(at);
    at += hsl_native_cache_align((c.capacity / HSL_NATIVE_CACHE_BLOCK) * sizeof(HslBlockSummary));
    c.temporary = reinterpret_cast<device HslPairEvent*>(at);
    return c;
}

inline bool hsl_native_cache_pair_advance(
    thread HslNativePairCursor& cursor, HslPairFact fact, bool short_side, float quantity_step
) {
    if (!isfinite(fact.delta) || fact.delta == 0.0f || !isfinite(fact.price)
        || fact.price <= 0.0f || !isfinite(fact.realized) || !isfinite(fact.fee)
        || !isfinite(quantity_step) || quantity_step <= 0.0f) return false;
    float delta = fact.delta * (short_side ? -1.0f : 1.0f);
    float before = cursor.size;
    cursor.scale = fmax(cursor.scale, fmax(before, fabs(delta)));
    hsl_pair_sum_add(cursor.inventory, delta);
    float after = hsl_pair_round_quantity(hsl_pair_sum_value(cursor.inventory), cursor.scale, quantity_step);
    // An inventory correction would alter an earlier inferred reduction run.
    if (!isfinite(after) || after < 0.0f) return false;
    if (delta > 0.0f) {
        cursor.basis = fma(delta / (before + delta), fact.price - cursor.basis, cursor.basis);
        cursor.has_increase = 1;
    }
    cursor.size = after;
    hsl_pair_sum_add(cursor.cash, fact.realized); hsl_pair_sum_add(cursor.cash, fact.fee);
    if (after == 0.0f) {
        cursor.basis = cursor.scale = 0.0f; hsl_pair_sum_reset(cursor.inventory, 0.0f);
    }
    ++cursor.consumed;
    return isfinite(cursor.basis) && isfinite(hsl_pair_sum_value(cursor.cash));
}

inline bool hsl_native_cache_sample_valid(thread const HslNativeCache& c, int minute) {
    return minute >= 0 && c.samples[minute % c.capacity].minute == minute;
}
inline void hsl_native_cache_store(
    thread const HslNativeCache& c, int minute, HslPairSum upnl, float cash
) {
    int slot = minute % c.capacity;
    c.upnl[slot] = upnl; c.cash[slot] = cash;
    float q = cash + hsl_pair_sum_value(upnl);
    HslBlockSample sample; sample.maximum = sample.final = q;
    sample.minute = minute; sample.episode = 0;
    c.samples[slot] = sample;
    c.summaries[slot / HSL_NATIVE_CACHE_BLOCK].valid = 0;
}
inline bool hsl_native_cache_repair_mark(
    thread const HslNativeCache& c, int minute, float before, float after
) {
    if (!hsl_native_cache_sample_valid(c, minute) || !isfinite(before) || !isfinite(after))
        return false;
    int slot = minute % c.capacity;
    HslPairSum sum = c.upnl[slot];
    hsl_pair_sum_add(sum, -before); hsl_pair_sum_add(sum, after);
    hsl_native_cache_store(c, minute, sum, c.cash[slot]);
    return isfinite(c.samples[slot].final);
}
inline bool hsl_native_cache_query(
    thread const HslNativeCache& c, int first, int end, float first_maximum,
    float first_final, float budget, float alpha, float threshold,
    thread HslScopeResult& result
) {
    if (first < 0 || end < first || end - first >= c.capacity
        || !isfinite(first_maximum) || !isfinite(first_final) || first_maximum < first_final
        || !isfinite(budget) || budget <= 0.0f || !isfinite(alpha)
        || alpha <= 0.0f || alpha > 1.0f || !isfinite(threshold)
        || threshold <= 0.0f || threshold > 1.0f) return false;
    const int width = HSL_NATIVE_CACHE_BLOCK;
    HslPairSum cash_sum = c.header->cash;
    float cash = hsl_pair_sum_value(cash_sum), k = budget - cash;
    HslBlockAnswer answer;
    answer.peak = first_maximum; answer.raw = answer.ema = hsl_block_raw(answer.peak, first_final, k);
    answer.error = 0.0f; answer.scalar_samples = 1;
    answer.full_blocks = answer.positive_terms = answer.fallback = 0;
    float minimum_denominator = k + first_maximum;
    if (!hsl_block_denominator_conditioned(k, first_maximum)
        || minimum_denominator <= 0.0f) return false;
    float currency_scale = fmax(fabs(cash), fmax(fabs(first_maximum), fabs(first_final)));
    for (int minute = first + 1; minute <= end;) {
        int slot = minute % c.capacity;
        if (minute % width == 0 && minute + width - 1 <= end) {
            int block = slot / width;
            HslBlockSummary summary = c.summaries[block];
            if (!summary.valid || summary.first != minute || summary.alpha != alpha) {
                for (int j = 0; j < width; ++j)
                    if (!hsl_native_cache_sample_valid(c, minute + j)) return false;
                if (!hsl_block_build(c.samples, c.capacity, slot, width, alpha,
                    c.groups + slot, width, c.summaries[block])) return false;
                c.summaries[block].first = minute;
                summary = c.summaries[block];
            }
            if (summary.groups < 1 || summary.groups > width || summary.count != width
                || summary.episode != 0 || !hsl_block_apply(c.groups + slot, summary, k, answer))
                return false;
            currency_scale = fmax(currency_scale,
                fmax(fabs(summary.minimum), fabs(c.groups[slot + summary.groups - 1].peak)));
            minute += width;
        } else {
            if (!hsl_native_cache_sample_valid(c, minute)) return false;
            HslBlockSample sample = c.samples[slot];
            if (!isfinite(sample.maximum) || !isfinite(sample.final)
                || sample.maximum < sample.final || sample.episode != 0) return false;
            answer.peak = fmax(answer.peak, sample.maximum);
            if (!hsl_block_denominator_conditioned(k, answer.peak)) return false;
            answer.raw = hsl_block_raw(answer.peak, sample.final, k);
            answer.ema = alpha * answer.raw + (1.0f - alpha) * answer.ema;
            answer.error = (1.0f - alpha) * answer.error + 8.0f * 0x1p-24f
                * fmax(0x1p-100f, fmax(fabs(answer.raw), fabs(answer.ema)));
            currency_scale = fmax(currency_scale, fmax(fabs(sample.maximum), fabs(sample.final)));
            ++answer.scalar_samples; ++minute;
        }
    }
    if (!hsl_native_cache_sample_valid(c, end)) return false;
    answer.raw = hsl_block_raw(answer.peak, c.samples[end % c.capacity].final, k);
    // Experimental numerical guard for fixed-coordinate formation and weighted
    // regrouping. The authoritative factual composer adjudicates every decline;
    // the component's transformed-Q scalar fallback never supplies native risk.
    float formation_margin = 32.0f * 0x1p-24f * (currency_scale + fabs(k) + 1.0f)
        / minimum_denominator + 16.0f * 0x1p-23f;
    float score = fmin(answer.raw, answer.ema);
    if (!isfinite(answer.raw) || !isfinite(answer.ema) || !isfinite(formation_margin)
        || fabs(score - threshold) <= answer.error + formation_margin) return false;
    result.raw = answer.raw; result.ema = answer.ema;
    result.action = score > threshold ? 3 : 0;
    result.flat_minute = -1; result.point_count = end - first + 1;
    result.latest_flat_minute = -1;
    result.latest_flat_raw = result.latest_flat_ema = 0.0f;
    return true;
}

// Only this unified owner stores events beside their physical fact slots.
// Independent factual reconstruction/composition continues to use logical rows.
inline HslPairEvent hsl_native_cache_event(thread const HslScopePair& pair, int logical) {
    return pair.events[hsl_pair_slot(pair.facts, logical)];
}
inline void hsl_native_cache_physical_events(
    thread const HslNativeCache& cache, thread const HslScopePair& pair
) {
    for (int i = 0; i < pair.facts.count; ++i) cache.temporary[i] = pair.events[i];
    for (int i = 0; i < pair.facts.count; ++i)
        pair.events[hsl_pair_slot(pair.facts, i)] = cache.temporary[i];
}

inline int hsl_native_pair_lower_bound(thread const HslScopePair& pair, int minute) {
    int left = 0, right = pair.facts.count;
    while (left < right) {
        int middle = left + (right - left) / 2;
        if (hsl_pair_minute(pair.facts, middle) < minute) left = middle + 1;
        else right = middle;
    }
    return left;
}
inline bool hsl_native_pair_precedes(thread HslScopePair* pairs, int a, int b) {
    int ta = hsl_pair_minute(pairs[a].facts, pairs[a].consumed);
    int tb = hsl_pair_minute(pairs[b].facts, pairs[b].consumed);
    return ta != tb ? ta < tb
        : hsl_scope_pair_sequence(pairs[a], pairs[a].consumed)
            < hsl_scope_pair_sequence(pairs[b], pairs[b].consumed);
}
inline void hsl_native_heads_down(
    thread HslScopePair* pairs, thread int* heads, int count, int index
) {
    while (index * 2 + 1 < count) {
        int child = index * 2 + 1;
        if (child + 1 < count && hsl_native_pair_precedes(pairs, heads[child + 1], heads[child]))
            ++child;
        if (!hsl_native_pair_precedes(pairs, heads[child], heads[index])) break;
        int swap = heads[index]; heads[index] = heads[child]; heads[child] = swap; index = child;
    }
}
inline int hsl_native_heads_initialize(
    thread HslScopePair* pairs, int count, int minute, thread int* heads,
    thread int& exposed
) {
    int n = 0; exposed = 0;
    for (int p = 0; p < count; ++p) {
        thread HslScopePair& pair = pairs[p];
        pair.consumed = hsl_native_pair_lower_bound(pair, minute);
        pair.boundary_size = pair.consumed > 0 ? hsl_native_cache_event(pair, pair.consumed - 1).after
            : fabs(pair.history.opening_size);
        if (pair.consumed == pair.facts.count && pair.consumed > 0
            && pair.history.flat_correction_minute >= 0) pair.boundary_size = 0.0f;
        exposed += pair.boundary_size != 0.0f;
        if (pair.consumed < pair.facts.count) heads[n++] = p;
    }
    for (int i = n / 2 - 1; i >= 0; --i) hsl_native_heads_down(pairs, heads, n, i);
    return n;
}
inline bool hsl_native_heads_consume(
    thread HslScopePair* pairs, thread int* heads, thread int& count,
    thread int& exposed, thread HslPairSum& cash, thread HslNativeCacheFlat& flat
) {
    int selected = heads[0]; thread HslScopePair& pair = pairs[selected];
    int i = pair.consumed++;
    HslPairEvent event = hsl_native_cache_event(pair, i);
    bool had_exposure = exposed > 0 || event.before > 0.0f || event.after > 0.0f;
    bool before = pair.boundary_size != 0.0f;
    pair.boundary_size = pair.consumed == pair.facts.count && pair.history.flat_correction_minute >= 0
        ? 0.0f : event.after;
    exposed += (pair.boundary_size != 0.0f) - before;
    HslPairFact fact = hsl_pair_fact(pair.facts, i);
    hsl_pair_sum_add(cash, fact.realized); hsl_pair_sum_add(cash, fact.fee);
    if (!isfinite(hsl_pair_sum_value(cash))) return false;
    if (exposed == 0 && had_exposure) {
        flat.minute = hsl_pair_minute(pair.facts, i);
        flat.sequence = hsl_scope_pair_sequence(pair, i);
        flat.cash = hsl_pair_sum_value(cash);
    }
    if (pair.consumed == pair.facts.count) heads[0] = heads[--count];
    hsl_native_heads_down(pairs, heads, count, 0);
    return true;
}
inline bool hsl_native_cache_latest_flat(
    thread HslScopePair* pairs, int count, int first, int end, float cash_start,
    thread HslNativeCacheFlat& flat,
    thread const HslPairSum* initial_cash = nullptr, thread HslPairSum* final_cash = nullptr
) {
    flat.minute = flat.sequence = -1; flat.cash = 0.0f;
    if (end < first) return true;
    int heads[2 * MAX_COINS], exposed;
    int n = hsl_native_heads_initialize(pairs, count, first, heads, exposed);
    HslPairSum cash; hsl_pair_sum_reset(cash, cash_start);
    if (initial_cash != nullptr) cash = *initial_cash;
    while (n > 0 && hsl_pair_minute(pairs[heads[0]].facts, pairs[heads[0]].consumed) <= end)
        if (!hsl_native_heads_consume(pairs, heads, n, exposed, cash, flat)) return false;
    if (final_cash != nullptr) *final_cash = cash;
    return true;
}
inline bool hsl_native_cache_mark(
    thread const HslScopePair& pair, int minute, int first, int end,
    thread HslPairSample& sample, bool physical = true
) {
    int consumed = hsl_native_pair_lower_bound(pair, minute);
    if (minute == end) consumed = pair.facts.count;
    device const HslPairEvent* events = pair.events;
    int available = pair.facts.count;
    if (physical && consumed > 0) {
        // The sampler only needs the last consumed event. A one-row view avoids
        // constructing pointers before the allocation when the ring wraps.
        events += hsl_pair_slot(pair.facts, consumed - 1);
        consumed = available = 1;
    }
    return hsl_sample_pair(pair.history, events, consumed, available,
        minute, end, false, pair.current_size, pair.current_basis, pair.current_mark,
        hsl_scope_price_at(pair, minute, first, end), pair.multiplier, pair.short_side, sample);
}

inline bool hsl_native_record_equal(HslPairRecord a, HslPairRecord b) {
    return a.minute == b.minute && a.first_sequence == b.first_sequence
        && a.last_sequence == b.last_sequence && a.actual_size_after == b.actual_size_after
        && a.fact.delta == b.fact.delta && a.fact.price == b.fact.price
        && a.fact.realized == b.fact.realized && a.fact.fee == b.fact.fee;
}
inline HslPairRecord hsl_native_cache_record(thread const HslScopePair& pair, int logical) {
    device const HslPairRecord* records = reinterpret_cast<device const HslPairRecord*>(pair.facts.values);
    return records[hsl_pair_slot(pair.facts, logical)];
}
inline HslNativeCachePair hsl_native_cache_pair_snapshot(
    thread const HslScopePair& pair, HslNativePairCursor cursor, int pair_id, float quantity_step
) {
    HslNativeCachePair saved = {};
    saved.cursor = cursor; saved.history = pair.history;
    saved.current_size = pair.current_size; saved.current_basis = pair.current_basis;
    saved.head = pair.facts.head; saved.count = pair.facts.count;
    saved.first_sequence = saved.last_sequence = -1;
    if (pair.facts.count > 0) {
        saved.first_sequence = hsl_scope_pair_sequence(pair, 0);
        saved.last_record = hsl_native_cache_record(pair, pair.facts.count - 1);
        saved.last_sequence = saved.last_record.last_sequence;
    }
    saved.quantity_step = quantity_step; saved.multiplier = pair.multiplier;
    saved.pair_id = pair_id; saved.valid = 1;
    saved.reserved[0] = saved.reserved[1] = 0;
    return saved;
}
inline bool hsl_native_cache_quotes_stable(thread const HslScopePair& pair, int first, int end) {
    if (pair.candles == nullptr || pair.candle_stride < 1 || first > end) return false;
    int a = max(first - 1, 0), b = end - 1;
    if (a < pair.candle_first || a > pair.candle_last || b < pair.candle_first || b > pair.candle_last)
        return false;
    float opening = pair.candles[a * pair.candle_stride], current = pair.candles[b * pair.candle_stride];
    return isfinite(opening) && opening > 0.0f && isfinite(current) && current > 0.0f
        && isfinite(pair.multiplier) && pair.multiplier > 0.0f;
}

// Builds disposable arithmetic only after the independent factual composer has
// supplied the current observation. Failure simply leaves this cache invalid.
inline bool hsl_native_cache_initialize(
    thread const HslNativeCache& cache, thread HslScopePair* pairs, int count,
    thread const int* pair_ids, thread const float* quantity_steps, int first, int end,
    HslScopeCutoff cutoff, float alpha
) {
    cache.header->valid = 0;
    if (count < 1 || count > cache.pair_capacity || first < 0 || end < first
        || end - first >= cache.capacity || !isfinite(alpha) || alpha <= 0.0f || alpha > 1.0f)
        return false;
    for (int p = 0; p < count; ++p) {
        if (!hsl_native_cache_quotes_stable(pairs[p], first, end)
            || (pairs[p].facts.count > 0
                && hsl_pair_minute(pairs[p].facts, pairs[p].facts.count - 1) >= end)) return false;
        HslNativePairCursor cursor;
        if (!hsl_reconstruct_pair(pairs[p].facts, pairs[p].current_size, pairs[p].current_basis,
            pairs[p].short_side, quantity_steps[p], pairs[p].events, pairs[p].history, &cursor)) return false;
        hsl_native_cache_physical_events(cache, pairs[p]);
        cache.pairs[p] = hsl_native_cache_pair_snapshot(pairs[p], cursor, pair_ids[p], quantity_steps[p]);
    }
    int heads[2 * MAX_COINS], exposed;
    int n = hsl_native_heads_initialize(pairs, count, first, heads, exposed);
    HslPairSum cash; hsl_pair_sum_reset(cash, 0.0f);
    HslNativeCacheFlat flat; flat.minute = flat.sequence = -1; flat.cash = 0.0f;
    for (int minute = first; minute <= end; ++minute) {
        while (n > 0 && hsl_pair_fill_precedes_price(
            hsl_pair_minute(pairs[heads[0]].facts, pairs[heads[0]].consumed), minute, end, false))
            if (!hsl_native_heads_consume(pairs, heads, n, exposed, cash, flat)) return false;
        HslPairSum upnl; hsl_pair_sum_reset(upnl, 0.0f);
        for (int p = 0; p < count; ++p) {
            HslPairSample sample;
            if (!hsl_native_cache_mark(pairs[p], minute, first, end, sample)) return false;
            hsl_pair_sum_add(upnl, sample.upnl);
        }
        hsl_native_cache_store(cache, minute, upnl, hsl_pair_sum_value(cash));
        if (!isfinite(cache.samples[minute % cache.capacity].final)) return false;
    }
    float anchor = hsl_pair_sum_value(cash);
    // Center once at fresh initialization. Later fills extend this owner prefix;
    // expiry changes inferred UPNL, not the already recorded execution cash.
    for (int minute = first; minute <= end; ++minute) {
        int slot = minute % cache.capacity;
        HslPairSum upnl = cache.upnl[slot];
        hsl_native_cache_store(cache, minute, upnl, cache.cash[slot] - anchor);
    }
    if (flat.minute >= 0) flat.cash -= anchor;
    cache.header->cash.value = cache.header->cash.correction = 0.0f;
    cache.header->minute = end; cache.header->first = first; cache.header->pair_count = count;
    cache.header->cutoff = cutoff; cache.header->alpha = alpha; cache.header->latest_flat = flat;
    cache.header->valid = 1;
    return true;
}

// A retained reconstructed zero followed by a retained increase resets both
// the reverse reduction requirement and forward inventory/basis arithmetic.
// Reconstruct only the changed left edge; unchanged suffix slots remain owned
// by the same factual ring. Declines always return to independent composition.
inline bool hsl_native_cache_retained_old(
    thread HslScopePair& old, thread const HslScopePair& pair,
    HslNativeCachePair saved, int retained
) {
    int dropped = (pair.facts.head - saved.head + pair.facts.capacity) % pair.facts.capacity;
    if (dropped < 1 || dropped >= saved.count || retained != saved.count - dropped
        || retained < 1) return false;
    // Removed factual slots may already contain new fills, but their cached
    // predecessor event still supplies the old left-edge price sample.
    HslPairEvent opening = pair.events[(saved.head + dropped - 1) % pair.facts.capacity];
    old.facts = pair.facts; old.facts.count = retained;
    old.history.opening_size = (pair.short_side ? -1.0f : 1.0f) * opening.after;
    old.history.opening_basis = opening.basis;
    return true;
}

// 1 reuses the certified suffix; 0 requests complete pair repair; -1 rejects
// this cache update after an aggregate mutation. No rejected attempt supplies risk.
inline int hsl_native_cache_flat_suffix(
    thread const HslNativeCache& cache, thread HslScopePair& pair,
    thread HslScopePair& old, HslNativeCachePair saved, int retained,
    float quantity_step, int first, int end,
    thread HslNativePairCursor& cursor, thread int& affected_until
) {
#if PASSIVBOT_HSL_NATIVE_CACHE_FLAT_SUFFIX_ENABLED
    if (retained < 2 || saved.cursor.has_increase == 0) return 0;
    int flat = -1;
    for (int i = 0; i + 1 < retained; ++i) {
        HslPairEvent event = hsl_native_cache_event(pair, i);
        float next = hsl_pair_fact(pair.facts, i + 1).delta * (pair.short_side ? -1.0f : 1.0f);
        if (event.after == 0.0f && event.basis == 0.0f && next > 0.0f) { flat = i; break; }
    }
    if (flat < 0) return 0;
    HslScopePair prefix = pair;
    prefix.facts.count = flat + 1; prefix.events = cache.temporary;
    // Keep the real observation end when sampling: the closing fill's own price
    // minute precedes that fill and must not see an artificial flat endpoint.
    HslNativePairCursor proof;
    if (!hsl_reconstruct_pair(prefix.facts, 0.0f, 0.0f, pair.short_side, quantity_step,
        prefix.events, prefix.history, &proof) || proof.size != 0.0f || proof.basis != 0.0f
        || proof.scale != 0.0f || hsl_pair_sum_value(proof.inventory) != 0.0f
        || prefix.history.flat_correction_minute >= 0) return 0;
    cursor = saved.cursor; cursor.consumed = retained;
    for (int i = retained; i < pair.facts.count; ++i) {
        float before = cursor.size;
        if (!hsl_native_cache_pair_advance(cursor, hsl_pair_fact(pair.facts, i), pair.short_side,
            quantity_step)) return 0;
        HslPairEvent event; event.before = before; event.after = cursor.size;
        event.basis = cursor.basis; event.realized = hsl_pair_sum_value(cursor.cash);
        pair.events[hsl_pair_slot(pair.facts, i)] = event;
    }
    affected_until = min(hsl_pair_minute(pair.facts, flat), end - 2);
    for (int minute = first; minute <= affected_until; ++minute) {
        HslPairSample a, b;
        if (!hsl_native_cache_mark(old, minute, first, end - 1, a)
            || !hsl_native_cache_mark(prefix, minute, first, end, b, false)
            || !hsl_native_cache_repair_mark(cache, minute, a.upnl, b.upnl)) return -1;
    }
    for (int i = 0; i <= flat; ++i) pair.events[hsl_pair_slot(pair.facts, i)] = prefix.events[i];
    pair.history = prefix.history;
    pair.history.flat_correction_minute = pair.current_size == 0.0f && cursor.size != 0.0f
        ? hsl_pair_minute(pair.facts, pair.facts.count - 1) : -1;
    return 1;
#else
    return 0;
#endif
}

// All inputs are native views AFTER the authoritative reverse-actual cutoff.
// A decline is disposable: the caller must run independent factual composition.
inline bool hsl_native_cache_replay(
    thread const HslNativeCache& cache, thread HslScopePair* pairs, int count,
    thread const int* pair_ids, thread const float* quantity_steps, int first, int end,
    HslScopeCutoff cutoff, float budget, float alpha, float threshold,
    thread HslScopeResult& result
) {
    HslNativeCacheHeader prior = *cache.header;
    cache.header->valid = 0;
    if (!prior.valid || count < 1 || count > cache.pair_capacity || count != prior.pair_count
        || end != prior.minute + 1 || first < prior.first || end - first >= cache.capacity
        || cutoff.found != prior.cutoff.found || cutoff.minute != prior.cutoff.minute
        || cutoff.sequence != prior.cutoff.sequence || alpha != prior.alpha
        || !hsl_native_cache_sample_valid(cache, end - 1)) return false;
    HslPairSum previous_upnl, current_upnl;
    hsl_pair_sum_reset(previous_upnl, 0.0f); hsl_pair_sum_reset(current_upnl, 0.0f);
    int repair_until = first - 1;
    bool estimated = false, reconstructed_flat = true;
    for (int p = 0; p < count; ++p) {
        thread HslScopePair& pair = pairs[p];
        HslNativeCachePair saved = cache.pairs[p];
        if (!saved.valid || saved.pair_id != pair_ids[p] || saved.multiplier != pair.multiplier
            || saved.quantity_step != quantity_steps[p] || pair.facts.value_stride != 2
            || pair.facts.minute_stride != 8 || pair.sequence_stride != 8
            || !hsl_native_cache_quotes_stable(pair, first, end)
            || saved.count < 0 || saved.count > pair.facts.capacity) return false;
        HslScopePair old = pair;
        old.facts.head = saved.head; old.facts.count = saved.count;
        old.history = saved.history; old.current_size = saved.current_size;
        old.current_basis = saved.current_basis;
        bool rolling = pair.facts.head != saved.head;
        if (saved.count > 0 && (saved.last_record.minute >= prior.minute
            || (!rolling && hsl_scope_pair_sequence(old, 0) != saved.first_sequence)
            || !hsl_native_record_equal(hsl_native_cache_record(old, saved.count - 1), saved.last_record)))
            return false; // A coalesced or overwritten prior fact has no reusable prefix.
        // New execution facts belong to the causal bar immediately preceding
        // this observation. Earlier insertions or a repeated-minute callback
        // cannot extend the owner's already-accounted cash prefix.
        int left = 0, right = pair.facts.count;
        while (left < right) {
            int middle = left + (right - left) / 2;
            if (hsl_scope_pair_sequence(pair, middle) <= saved.last_sequence) left = middle + 1;
            else right = middle;
        }
        for (int i = left; i < pair.facts.count; ++i)
            if (hsl_pair_minute(pair.facts, i) != end - 1) return false;
        HslNativePairCursor cursor = saved.cursor;
        int affected_until = first - 1;
        bool suffix = false;
        if (rolling) {
            if (!hsl_native_cache_retained_old(old, pair, saved, left)) return false;
            int reused = hsl_native_cache_flat_suffix(cache, pair, old, saved, left,
                quantity_steps[p], first, end, cursor, affected_until);
            if (reused < 0) return false;
            suffix = reused > 0;
        }
        bool append = pair.facts.head == saved.head && pair.facts.count >= saved.count
            && (cursor.has_increase || saved.count == 0 || pair.facts.count == saved.count)
            && (pair.facts.count != saved.count
                || (pair.current_size == saved.current_size && pair.current_basis == saved.current_basis));
        if (!rolling && append) {
            for (int i = saved.count; i < pair.facts.count; ++i) {
                if (hsl_pair_minute(pair.facts, i) != end - 1) { append = false; break; }
                float before = cursor.size;
                if (!hsl_native_cache_pair_advance(cursor, hsl_pair_fact(pair.facts, i), pair.short_side,
                    quantity_steps[p])) { append = false; break; }
                HslPairEvent event; event.before = before; event.after = cursor.size;
                event.basis = cursor.basis; event.realized = hsl_pair_sum_value(cursor.cash);
                pair.events[hsl_pair_slot(pair.facts, i)] = event;
            }
            if (append) {
                pair.history = saved.history;
                pair.history.flat_correction_minute = pair.current_size == 0.0f && cursor.size != 0.0f
                    && pair.facts.count > 0 ? hsl_pair_minute(pair.facts, pair.facts.count - 1) : -1;
            }
        }
        if (!suffix && !append) {
            for (int i = 0; i < old.facts.count; ++i) cache.temporary[i] = hsl_native_cache_event(old, i);
            old.events = cache.temporary;
            if (!hsl_reconstruct_pair(pair.facts, pair.current_size, pair.current_basis,
                pair.short_side, quantity_steps[p], pair.events, pair.history, &cursor)) return false;
        }
        bool same_history = old.history.opening_size == pair.history.opening_size
            && old.history.opening_basis == pair.history.opening_basis
            && old.history.flat_correction_minute == pair.history.flat_correction_minute;
        if (!suffix && (!append || !same_history)) {
            HslPairSample a, b;
            if (!hsl_native_cache_mark(old, first, first, end - 1, a, append)
                || !hsl_native_cache_mark(pair, first, first, end, b, append)) return false;
            int latest_changed = a.size != b.size || a.basis != b.basis ? first : first - 1;
            int i = hsl_native_pair_lower_bound(old, first);
            for (int j = 0; j < pair.facts.count && hsl_pair_minute(pair.facts, j) < end - 1; ++j) {
                int sequence = hsl_scope_pair_sequence(pair, j);
                while (i < old.facts.count && hsl_scope_pair_sequence(old, i) < sequence) ++i;
                if (i == old.facts.count || hsl_scope_pair_sequence(old, i) != sequence) continue;
                HslPairEvent before = append ? hsl_native_cache_event(old, i) : old.events[i];
                HslPairEvent after = append ? hsl_native_cache_event(pair, j) : pair.events[j];
                if (before.before != after.before || before.after != after.after || before.basis != after.basis)
                    latest_changed = hsl_pair_minute(pair.facts, j);
                if (before.after == 0.0f && after.after == 0.0f && latest_changed >= first) {
                    affected_until = hsl_pair_minute(pair.facts, j); latest_changed = first - 1;
                }
            }
            if (latest_changed >= first || old.history.flat_correction_minute != pair.history.flat_correction_minute)
                affected_until = end - 2;
        }
        for (int minute = first; !suffix && minute <= affected_until; ++minute) {
            HslPairSample a, b;
            if (!hsl_native_cache_mark(old, minute, first, end - 1, a, append)
                || !hsl_native_cache_mark(pair, minute, first, end, b, append)
                || !hsl_native_cache_repair_mark(cache, minute, a.upnl, b.upnl)) return false;
        }
        if (!suffix && !append) hsl_native_cache_physical_events(cache, pair);
        repair_until = max(repair_until, affected_until);
        HslPairSample previous, current;
        if (!hsl_native_cache_mark(pair, end - 1, first, end, previous)
            || !hsl_native_cache_mark(pair, end, first, end, current)) return false;
        hsl_pair_sum_add(previous_upnl, previous.upnl); hsl_pair_sum_add(current_upnl, current.upnl);
        estimated = estimated || pair.history.opening_size != 0.0f;
        reconstructed_flat = reconstructed_flat && (pair.facts.count == 0
            ? pair.history.opening_size == 0.0f : hsl_native_cache_event(pair, pair.facts.count - 1).after == 0.0f);
        cache.pairs[p] = hsl_native_cache_pair_snapshot(pair, cursor, pair_ids[p], quantity_steps[p]);
    }
    float previous_cash = hsl_pair_sum_value(prior.cash);
    hsl_native_cache_store(cache, end - 1, previous_upnl, previous_cash);
    HslNativeCacheFlat prefix, tail, latest;
    latest.minute = latest.sequence = -1; latest.cash = 0.0f;
    if (repair_until >= first) {
        if (!hsl_native_cache_sample_valid(cache, first)
            || !hsl_native_cache_latest_flat(pairs, count, first, repair_until,
                cache.cash[first % cache.capacity], prefix)) return false;
        latest = prefix;
    }
    if (prior.latest_flat.minute >= first && prior.latest_flat.minute > repair_until
        && prior.latest_flat.minute < end - 1) latest = prior.latest_flat;
    HslPairSum current_cash;
    if (!hsl_native_cache_latest_flat(pairs, count, end - 1, end - 1,
        previous_cash, tail, &prior.cash, &current_cash)) return false;
    if (tail.minute >= 0) latest = tail;
    cache.header->cash = current_cash;
    hsl_native_cache_store(cache, end, current_upnl, hsl_pair_sum_value(current_cash));
    cache.header->minute = end; cache.header->first = first; cache.header->pair_count = count;
    cache.header->cutoff = cutoff; cache.header->alpha = alpha; cache.header->latest_flat = latest;
    // The source's actual singleton drops the preceding curve. Keep that narrow
    // case on the independent composer unless the existing empty shortcut applies.
    if (reconstructed_flat) return false;
    int episode_first = latest.minute >= first ? latest.minute : first;
    if (!hsl_native_cache_sample_valid(cache, episode_first)) return false;
    HslBlockSample initial = cache.samples[episode_first % cache.capacity];
    float maximum = initial.maximum, final = initial.final;
    if (latest.minute >= first) maximum = final = latest.cash;
    else if (estimated) maximum = fmax(maximum, cache.cash[first % cache.capacity]);
    bool accepted = hsl_native_cache_query(cache, episode_first, end, maximum, final,
        budget, alpha, threshold, result);
    cache.header->valid = 1;
    return accepted;
}
#endif
