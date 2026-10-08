// Retained finite simulator fill facts for linear instruments. The factual
// view is worker-owned, ordered and already clipped to the requested horizon.
// This reconstructs Rust hsl_history's pair inputs; it is not a controller.
struct HslPairFact {
    float delta; // Signed position quantity, separate from order side.
    float price;
    float realized;
    float fee;
};

struct HslPairFacts {
    device const HslPairFact* values;
    device const int* minutes;
    int head;
    int count;
    int capacity;
    int value_stride;
    int minute_stride;
};

struct HslPairEvent {
    float before; // Absolute pair inventory, as in Rust's reconstructed events.
    float after;
    float basis;
    float realized;
};

struct HslPairHistory {
    float opening_size; // Signed, as in Rust's pair history.
    float opening_basis;
    int flat_correction_minute;
};

struct HslPairSample {
    float size; // Signed position quantity.
    float basis;
    float realized;
    float upnl;
};

// Preserve small net cashflows and lot residuals across mixed-magnitude facts.
// This is disposable reconstruction arithmetic, not additional simulator state.
struct HslPairSum { float value; float correction; };

inline void hsl_pair_sum_reset(thread HslPairSum& sum, float value) {
    sum.value = value;
    sum.correction = 0.0f;
}

inline void hsl_pair_sum_add(thread HslPairSum& sum, float value) {
    float next = sum.value + value;
    sum.correction += fabs(sum.value) >= fabs(value)
        ? (sum.value - next) + value : (value - next) + sum.value;
    sum.value = next;
}

inline float hsl_pair_sum_value(thread const HslPairSum& sum) {
    return sum.value + sum.correction;
}

inline int hsl_pair_slot(thread const HslPairFacts& facts, int logical) {
    return (facts.head + logical) % facts.capacity;
}

inline HslPairFact hsl_pair_fact(thread const HslPairFacts& facts, int logical) {
    return facts.values[hsl_pair_slot(facts, logical) * facts.value_stride];
}

inline int hsl_pair_minute(thread const HslPairFacts& facts, int logical) {
    return facts.minutes[hsl_pair_slot(facts, logical) * facts.minute_stride];
}

inline float hsl_pair_round_quantity(float value, float scale, float step) {
    // Use the reconstruction's roundoff rule at float32 precision. The quantum
    // guard prevents a valid tick from being erased as inventory grows.
    float tolerance = 64.0f * 1.1920928955078125e-7f * fmax(scale, fabs(value));
    return value != 0.0f && fabs(value) <= tolerance && fabs(value) < step * 0.5f
        ? 0.0f : value;
}

inline bool hsl_reconstruct_pair(
    thread const HslPairFacts& facts, float current_size, float current_basis,
    bool short_side, float quantity_step, device HslPairEvent* events,
    thread HslPairHistory& history
) {
    float direction = short_side ? -1.0f : 1.0f;
    if (facts.capacity < 1 || facts.count < 0 || facts.count > facts.capacity
        || facts.head < 0 || facts.head >= facts.capacity
        || facts.value_stride < 1 || facts.minute_stride < 1
        || !isfinite(current_size) || current_size * direction < 0.0f
        || !isfinite(current_basis) || (current_size != 0.0f && current_basis <= 0.0f)
        || !isfinite(quantity_step) || quantity_step <= 0.0f) return false;
    // Store each suffix reduction requirement in scratch, then overwrite it with
    // its event. No quadratic scan, heap allocation or retained inferred state.
    HslPairSum reductions;
    hsl_pair_sum_reset(reductions, 0.0f);
    bool only_reductions = facts.count > 0;
    for (int i = facts.count - 1; i >= 0; --i) {
        HslPairFact f = hsl_pair_fact(facts, i);
        if (!isfinite(f.delta) || f.delta == 0.0f || !isfinite(f.price) || f.price <= 0.0f
            || !isfinite(f.realized) || !isfinite(f.fee)) return false;
        float delta = f.delta * direction;
        if (delta > 0.0f) { hsl_pair_sum_reset(reductions, 0.0f); only_reductions = false; }
        else hsl_pair_sum_add(reductions, -delta);
        float needed = hsl_pair_sum_value(reductions);
        if (!isfinite(needed)) return false;
        events[i].before = needed;
    }
    float initial = facts.count > 0 ? events[0].before : 0.0f;
    if (only_reductions) initial += fabs(current_size);
    float size = initial;
    float basis = initial > 0.0f
        ? hsl_pair_fact(facts, 0).price : 0.0f;
    history.opening_size = direction * initial;
    history.opening_basis = basis;
    history.flat_correction_minute = -1;
    float scale = initial;
    HslPairSum inventory, cash;
    hsl_pair_sum_reset(inventory, initial);
    hsl_pair_sum_reset(cash, 0.0f);
    for (int i = 0; i < facts.count; ++i) {
        HslPairFact f = hsl_pair_fact(facts, i);
        float delta = f.delta * direction;
        scale = fmax(scale, fmax(size, fabs(delta)));
        float before = size;
        if (hsl_pair_round_quantity(before - events[i].before, scale, quantity_step) < 0.0f) {
            before = events[i].before;
            hsl_pair_sum_reset(inventory, before);
        }
        hsl_pair_sum_add(inventory, delta);
        size = fmax(0.0f, hsl_pair_round_quantity(hsl_pair_sum_value(inventory), scale, quantity_step));
        if (before > 0.0f && basis == 0.0f) basis = f.price;
        if (delta > 0.0f) {
            float weight = delta / (before + delta);
            basis = fma(weight, f.price - basis, basis);
        }
        hsl_pair_sum_add(cash, f.realized);
        hsl_pair_sum_add(cash, f.fee);
        if (size == 0.0f) { basis = 0.0f; scale = 0.0f; hsl_pair_sum_reset(inventory, 0.0f); }
        if (!isfinite(size) || !isfinite(basis) || !isfinite(hsl_pair_sum_value(cash))) return false;
        events[i].before = before;
        events[i].after = size;
        events[i].basis = basis;
        events[i].realized = hsl_pair_sum_value(cash);
    }
    if (current_size == 0.0f && size != 0.0f && facts.count > 0)
        history.flat_correction_minute = hsl_pair_minute(facts, facts.count - 1);
    return true;
}

inline bool hsl_pair_fill_precedes_price(int fill, int price, int end, bool before) {
    return fill < price || (fill == price && (before || price == end));
}

inline bool hsl_sample_pair(
    thread const HslPairHistory& history, device const HslPairEvent* events,
    int consumed, int count, int minute, int end, bool before_price,
    float current_size, float current_basis, float mark, float historical_price,
    float multiplier, bool short_side, thread HslPairSample& sample
) {
    if (consumed < 0 || consumed > count || !isfinite(mark) || mark <= 0.0f
        || !isfinite(historical_price) || historical_price <= 0.0f
        || !isfinite(multiplier) || multiplier <= 0.0f) return false;
    float direction = short_side ? -1.0f : 1.0f;
    float size = consumed > 0 ? events[consumed - 1].after : fabs(history.opening_size);
    float basis = consumed > 0 ? events[consumed - 1].basis : history.opening_basis;
    sample.realized = consumed > 0 ? events[consumed - 1].realized : 0.0f;
    if (minute == end || (history.flat_correction_minute >= 0
            && hsl_pair_fill_precedes_price(history.flat_correction_minute, minute, end, before_price))) {
        size = fabs(current_size);
        basis = current_basis;
    }
    float price = minute == end ? mark : historical_price;
    sample.size = direction * size;
    sample.basis = basis;
    sample.upnl = size == 0.0f ? 0.0f : direction * size * multiplier * (price - basis);
    return isfinite(sample.upnl);
}

// One record is one globally consecutive same-pair/direction/time run. Actual
// post-fill position and global sequence remain factual lifecycle inputs.
struct HslPairRecord {
    HslPairFact fact;
    int minute;
    int first_sequence;
    int last_sequence;
    float actual_size_after;
};

// Fits in two 32-byte node-sized slots, separate from controller/window state.
struct HslPairRingState {
    int head;
    int count;
    int version;
    int failure; // 1: overflow; 2: malformed producer fact.
    int latest_minute;
    int latest_sequence;
    HslPairSum delta;
    HslPairSum realized;
    HslPairSum fee;
    float current_size;
    float current_basis;
};

struct HslPairRing {
    device HslPairRingState* state;
    device HslPairRecord* records;
    int capacity;
    int lookback;
    bool enabled;
};

inline void hsl_pair_ring_reset(thread HslPairRing& ring) {
    ring.state->head = ring.state->count = ring.state->version = ring.state->failure = 0;
    ring.state->latest_minute = ring.state->latest_sequence = -1;
    HslPairSum sum;
    hsl_pair_sum_reset(sum, 0.0f);
    ring.state->delta = ring.state->realized = ring.state->fee = sum;
    ring.state->current_size = ring.state->current_basis = 0.0f;
}

inline bool hsl_pair_ring_append(
    thread HslPairRing& ring, HslPairFact fact, int minute, int sequence,
    float actual_size_after, bool short_side
) {
    if (!ring.enabled) return true;
    float direction = short_side ? -1.0f : 1.0f;
    if (ring.state->failure != 0) return false;
    if (ring.capacity < 1 || ring.lookback < 0 || minute < 0 || sequence < 0
        || ring.state->head < 0 || ring.state->head >= ring.capacity
        || ring.state->count < 0 || ring.state->count > ring.capacity
        || minute < ring.state->latest_minute || sequence <= ring.state->latest_sequence
        || !isfinite(fact.delta) || fact.delta == 0.0f
        || !isfinite(fact.price) || fact.price <= 0.0f
        || !isfinite(fact.realized) || !isfinite(fact.fee)
        || !isfinite(actual_size_after) || actual_size_after * direction < 0.0f) {
        ring.state->failure = 2;
        return false;
    }
    while (ring.state->count > 0
        && ring.records[ring.state->head].minute < minute - ring.lookback) {
        ring.state->head = (ring.state->head + 1) % ring.capacity;
        --ring.state->count;
        ++ring.state->version;
    }
    int last = (ring.state->head + ring.state->count - 1) % ring.capacity;
    bool merge = ring.state->count > 0
        && ring.records[last].minute == minute
        && ring.records[last].last_sequence == sequence - 1
        && (ring.records[last].fact.delta > 0.0f) == (fact.delta > 0.0f);
    if (ring.state->count > 0 && (ring.records[last].minute > minute
        || ring.records[last].last_sequence >= sequence)) {
        ring.state->failure = 2;
        return false;
    }
    // Never merge a close through an actual flat boundary. A later malformed
    // reduction remains a separate disclosed fact, not a hidden merged episode.
    if (merge && fact.delta * direction < 0.0f
        && ring.records[last].actual_size_after == 0.0f) merge = false;
    if (!merge && ring.state->count == ring.capacity) {
        ring.state->failure = 1;
        return false;
    }
    HslPairSum delta, realized, fee;
    if (merge) {
        delta = ring.state->delta;
        realized = ring.state->realized;
        fee = ring.state->fee;
    } else {
        hsl_pair_sum_reset(delta, 0.0f);
        hsl_pair_sum_reset(realized, 0.0f);
        hsl_pair_sum_reset(fee, 0.0f);
    }
    hsl_pair_sum_add(delta, fact.delta);
    hsl_pair_sum_add(realized, fact.realized);
    hsl_pair_sum_add(fee, fact.fee);
    HslPairRecord record;
    record.fact.delta = hsl_pair_sum_value(delta);
    record.fact.realized = hsl_pair_sum_value(realized);
    record.fact.fee = hsl_pair_sum_value(fee);
    record.fact.price = merge ? ring.records[last].fact.price : fact.price;
    if (merge && fact.delta * direction > 0.0f) {
        float weight = fact.delta / record.fact.delta;
        record.fact.price = fma(weight, fact.price - record.fact.price, record.fact.price);
    }
    if (!isfinite(record.fact.delta) || !isfinite(record.fact.realized)
        || !isfinite(record.fact.fee) || !isfinite(record.fact.price)) {
        ring.state->failure = 2;
        return false;
    }
    record.minute = minute;
    record.first_sequence = merge ? ring.records[last].first_sequence : sequence;
    record.last_sequence = sequence;
    record.actual_size_after = actual_size_after;
    int slot = merge ? last : (ring.state->head + ring.state->count) % ring.capacity;
    ring.records[slot] = record;
    ring.state->latest_minute = minute;
    ring.state->latest_sequence = sequence;
    ring.state->delta = delta;
    ring.state->realized = realized;
    ring.state->fee = fee;
    if (!merge) ++ring.state->count;
    ++ring.state->version;
    return true;
}

// Native fill producers provide the actual endpoint even when their caller
// applies the position mutation after accounting a close. This is simulator
// truth, separate from inferred historical inventory and disposable replay.
inline bool hsl_pair_ring_capture(
    thread HslPairRing& ring, HslPairFact fact, int minute, int sequence,
    float actual_size_after, float actual_basis_after, bool short_side
) {
    if (!ring.enabled) return true;
    if (!isfinite(actual_basis_after) || (actual_size_after != 0.0f
        ? actual_basis_after <= 0.0f : actual_basis_after != 0.0f)) {
        ring.state->failure = 2;
        return false;
    }
    if (!hsl_pair_ring_append(ring, fact, minute, sequence, actual_size_after, short_side))
        return false;
    ring.state->current_size = actual_size_after;
    ring.state->current_basis = actual_basis_after;
    return true;
}

inline HslPairFacts hsl_pair_ring_view(thread const HslPairRing& ring, int start) {
    HslPairFacts facts;
    facts.values = reinterpret_cast<device const HslPairFact*>(ring.records);
    facts.minutes = reinterpret_cast<device const int*>(ring.records) + 4;
    facts.head = ring.state->head;
    facts.count = ring.state->count;
    facts.capacity = ring.capacity;
    facts.value_stride = 2;
    facts.minute_stride = 8;
    while (facts.count > 0 && hsl_pair_minute(facts, 0) < start) {
        facts.head = (facts.head + 1) % facts.capacity;
        --facts.count;
    }
    return facts;
}

// A cutoff is a factual flat prefix, including its globally ordered execution.
// Current exposure needs the latest flatten; current flatness needs the preceding
// flatten so the just-completed episode remains available for cooldown replay.
struct HslScopeCutoff {
    int minute;
    int sequence;
    bool found;
};

// Simulator-only memo of a factual flat prefix. No signal or permission is
// retained. Pair identities and current exposure are exact masks; every append,
// coalescence and pruning advances a monotonic native ring version. A new
// candidate/retry resets this memo together with its factual rings.
struct HslScopeCutoffCache {
    HslScopeCutoff cutoff;
    ulong version_sum;
    uint selected[4];
    uint exposed[4];
    bool valid;
};

inline void hsl_scope_cutoff_cache_reset(thread HslScopeCutoffCache& cache) {
    cache.cutoff.minute = cache.cutoff.sequence = -1;
    cache.cutoff.found = false;
    cache.version_sum = 0;
    cache.valid = false;
    for (int i = 0; i < 4; ++i) cache.selected[i] = cache.exposed[i] = 0;
}

inline bool hsl_scope_history_cutoff(
    thread const HslPairRing* rings, thread const float* current_sizes,
    thread const float* quantity_steps, int pair_count,
    thread int* cursors, thread float* sizes, thread HslScopeCutoff& cutoff
) {
    cutoff.minute = cutoff.sequence = -1;
    cutoff.found = false;
    if (pair_count < 1) return false;
    int exposed = 0, first_minute = -1;
    for (int p = 0; p < pair_count; ++p) {
        const HslPairRing ring = rings[p];
        if (!ring.enabled || ring.state->failure != 0 || ring.capacity < 1
            || ring.state->head < 0 || ring.state->head >= ring.capacity
            || ring.state->count < 0 || ring.state->count > ring.capacity
            || !isfinite(current_sizes[p]) || !isfinite(quantity_steps[p])
            || quantity_steps[p] <= 0.0f) return false;
        sizes[p] = current_sizes[p];
        exposed += sizes[p] != 0.0f ? 1 : 0;
        cursors[p] = ring.state->count - 1;
        if (ring.state->count > 0) {
            int first = ring.records[ring.state->head].minute;
            first_minute = first_minute < 0 ? first : min(first_minute, first);
        }
    }
    const int needed = exposed == 0 ? 2 : 1;
    int boundaries = 0;
    while (true) {
        int selected = -1, sequence = -1;
        for (int p = 0; p < pair_count; ++p) {
            if (cursors[p] < 0) continue;
            const HslPairRing ring = rings[p];
            int slot = (ring.state->head + cursors[p]) % ring.capacity;
            int candidate = ring.records[slot].last_sequence;
            if (candidate > sequence) { selected = p; sequence = candidate; }
        }
        if (selected < 0) break;
        const HslPairRing ring = rings[selected];
        int slot = (ring.state->head + cursors[selected]) % ring.capacity;
        HslPairRecord record = ring.records[slot];
        // Same as the simulator's factual cutoff: reverse actual post-fill
        // inventory by the signed execution quantity, on its exchange quantum.
        float before = round((record.actual_size_after - record.fact.delta)
            / quantity_steps[selected]) * quantity_steps[selected];
        if (!isfinite(before)) return false;
        if (exposed == 0 && before != 0.0f && ++boundaries == needed) {
            cutoff.minute = record.minute;
            cutoff.sequence = record.last_sequence;
            cutoff.found = true;
            return true;
        }
        exposed -= sizes[selected] != 0.0f ? 1 : 0;
        exposed += before != 0.0f ? 1 : 0;
        sizes[selected] = before;
        --cursors[selected];
    }
    // The retained prefix proves flatness only when this reverse walk reaches
    // zero. An exposed clipped prefix keeps the caller's configured lookback;
    // it must not receive an invented flat seed before its first retained fill.
    if (first_minute >= 0 && exposed == 0) {
        cutoff.minute = first_minute - 1;
        cutoff.found = true;
    }
    return true;
}

inline bool hsl_scope_history_cutoff_cached(
    thread const HslPairRing* rings, thread const float* current_sizes,
    thread const float* quantity_steps, thread const int* pair_ids, int pair_count,
    thread int* cursors, thread float* sizes, thread HslScopeCutoffCache& cache,
    thread HslScopeCutoff& cutoff, thread bool& reused
) {
    reused = false;
    if (pair_count < 1 || pair_count > 128) return false;
    uint selected[4] = {0, 0, 0, 0}, exposed[4] = {0, 0, 0, 0};
    ulong version_sum = 0;
    for (int p = 0; p < pair_count; ++p) {
        const HslPairRing ring = rings[p];
        if (!ring.enabled || ring.state->failure != 0 || ring.capacity < 1
            || ring.state->head < 0 || ring.state->head >= ring.capacity
            || ring.state->count < 0 || ring.state->count > ring.capacity
            || ring.state->version < 0 || !isfinite(current_sizes[p])
            || !isfinite(quantity_steps[p]) || quantity_steps[p] <= 0.0f
            || pair_ids[p] < 0 || pair_ids[p] >= 128) return false;
        int word = pair_ids[p] / 32;
        uint bit = 1u << (pair_ids[p] % 32);
        if ((selected[word] & bit) != 0) return false;
        selected[word] |= bit;
        if (current_sizes[p] != 0.0f) exposed[word] |= bit;
        version_sum += ulong(ring.state->version);
    }
    bool same = cache.valid && cache.version_sum == version_sum;
    for (int i = 0; i < 4; ++i)
        same = same && cache.selected[i] == selected[i] && cache.exposed[i] == exposed[i];
    if (same) {
        cutoff = cache.cutoff;
        reused = true;
        return true;
    }
    if (!hsl_scope_history_cutoff(rings, current_sizes, quantity_steps, pair_count,
        cursors, sizes, cutoff)) return false;
    cache.cutoff = cutoff;
    cache.version_sum = version_sum;
    for (int i = 0; i < 4; ++i) {
        cache.selected[i] = selected[i];
        cache.exposed[i] = exposed[i];
    }
    cache.valid = true;
    return true;
}

inline HslPairFacts hsl_pair_ring_view_after(
    thread const HslPairRing& ring, int start, thread const HslScopeCutoff& cutoff
) {
    HslPairFacts facts = hsl_pair_ring_view(ring, start);
    while (facts.count > 0 && cutoff.found && cutoff.sequence >= 0
        && ring.records[facts.head].last_sequence <= cutoff.sequence) {
        facts.head = (facts.head + 1) % facts.capacity;
        --facts.count;
    }
    return facts;
}
