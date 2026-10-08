#ifndef PASSIVBOT_HSL_FACTUAL_ONLY
#define PASSIVBOT_HSL_FACTUAL_ONLY 0
#endif

#ifndef PASSIVBOT_HSL_FACTS_ENABLED
#define PASSIVBOT_HSL_FACTS_ENABLED 0
#endif

// HSL arithmetic over factual PNL + UPNL samples. No trading state or
// episode inference lives here. Both Metal and CUDA consume this scalar source.
// A candidate owns a bounded ring of compact samples and a tree over 64-sample
// blocks. Partial boundary blocks are visited directly; complete blocks aggregate.
struct HslNode {
    float peak;
    float first;
    float last;
    float loss;
    float weight;
    float decay;
    int count;
    int rising;
};

inline int hsl_storage_nodes(int capacity, int tree_size, int fact_capacity = 0) {
#if PASSIVBOT_HSL_FACTUAL_ONLY
    // Native replay owns only factual headers, records and disposable events.
    return fact_capacity > 0 ? 2 + fact_capacity + (fact_capacity + 1) / 2 : 0;
#else
    // Four two-float samples fit in one 32-byte node-sized allocation.
    int nodes = 2 * tree_size + (capacity + 3) / 4;
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    nodes += 2 + fact_capacity + (fact_capacity + 1) / 2;
#endif
    return nodes;
#endif
}

#if !PASSIVBOT_HSL_FACTUAL_ONLY
struct HslWindow {
    int head;
    int count;
    int capacity;
    int tree_size;
    float alpha;
    int last_minute;
    HslNode block_prefix; // Completed samples before the replaceable last row.
};

inline HslNode hsl_empty_node() {
    HslNode n;
    n.peak = -INFINITY;
    n.first = n.last = n.loss = n.weight = 0.0f;
    n.decay = 1.0f;
    n.count = 0;
    n.rising = 1;
    return n;
}

inline HslNode hsl_join(HslNode l, HslNode r) {
    if (l.count == 0) return r;
    if (r.count == 0) return l;
    HslNode n;
    n.peak = fmax(l.peak, r.peak);
    n.first = l.first;
    n.last = r.last;
    n.loss = (l.loss + (n.peak - l.peak) * l.weight) * r.decay
        + r.loss + (n.peak - r.peak) * r.weight;
    n.weight = l.weight * r.decay + r.weight;
    n.decay = l.decay * r.decay;
    n.count = l.count + r.count;
    n.rising = l.rising && r.rising && l.last <= r.first;
    return n;
}


inline device float2* hsl_samples(device HslNode* tree, int tree_size) {
    return reinterpret_cast<device float2*>(tree + 2 * tree_size);
}

inline HslNode hsl_sample(float2 row, float alpha) {
    HslNode n;
    n.peak = row.y;
    n.first = n.last = row.x;
    n.loss = alpha * (row.y - row.x);
    n.weight = alpha;
    n.decay = 1.0f - alpha;
    n.count = 1;
    n.rising = row.x == row.y;
    return n;
}

inline void hsl_set(
    device HslNode* tree, int tree_size, int slot, HslNode n
) {
    int index = tree_size + slot;
    tree[index] = n;
    // Only complete aligned blocks can be consumed by a chronological range
    // query. Updating completed right edges makes append amortized O(1).
    while (index > 1 && (index & 1) != 0) {
        index /= 2;
        tree[index] = hsl_join(tree[index * 2], tree[index * 2 + 1]);
    }
}

inline HslWindow hsl_init(
    device HslNode* tree, int capacity, int tree_size, float span
) {
    HslWindow w;
    w.head = w.count = 0;
    w.capacity = capacity;
    w.tree_size = tree_size;
    w.alpha = 2.0f / (span + 1.0f);
    w.last_minute = -1;
    w.block_prefix = hsl_empty_node();
    for (int i = 0; i < 2 * tree_size; ++i) tree[i] = hsl_empty_node();
    return w;
}

inline void hsl_expire(
    thread HslWindow& w, device HslNode* tree,
    device int* times, int first_minute
) {
    while (w.count > 0 && times[w.head] < first_minute) {
        // Expired leaves lie outside both queried ring ranges.
        w.head = (w.head + 1) % w.capacity;
        --w.count;
    }
}

inline bool hsl_push(
    thread HslWindow& w, device HslNode* tree,
    device int* times, int minute, float value
) {
    if (!isfinite(value) || minute < w.last_minute) return false;
    const bool replace = w.count > 0 && minute == w.last_minute;
    if (!replace && w.count == w.capacity) return false;
    int slot = (w.head + w.count - (replace ? 1 : 0)) % w.capacity;
    device float2* samples = hsl_samples(tree, w.tree_size);
    float peak = replace ? fmax(samples[slot].y, value) : value;
    samples[slot] = float2(value, peak);
    int block = slot / 64;
    if (!replace) {
        w.block_prefix = slot % 64 == 0 ? hsl_empty_node()
            : tree[w.tree_size + block];
    }
    HslNode n = hsl_join(w.block_prefix,
        hsl_sample(samples[slot], w.alpha));
    tree[w.tree_size + block] = n;
    if (slot % 64 == 63) hsl_set(tree, w.tree_size, block, n);
    times[slot] = minute;
    w.last_minute = minute;
    if (!replace) ++w.count;
    return true;
}

// Iterative left-to-right traversal; Metal prohibits recursive functions.
// At most one right sibling per level is pending (90d of minutes < 2^18).
inline void hsl_visit(
    device HslNode* tree, int tree_size, int begin, int end,
    float offset, float alpha, thread float& peak, thread float& ema
) {
    int pending_index[32];
    int pending_lo[32];
    int pending_hi[32];
    int count = 1;
    pending_index[0] = 1;
    pending_lo[0] = 0;
    pending_hi[0] = tree_size * 64;
    while (count > 0) {
        --count;
        int index = pending_index[count];
        int lo = pending_lo[count];
        int hi = pending_hi[count];
        if (hi <= begin || end <= lo) continue;
        HslNode n = tree[index];
        if (begin <= lo && hi <= end) {
            if (n.count == 0) continue;
            if (n.peak <= peak) {
                float contribution = offset + peak > 0.0f
                    ? (n.loss + (peak - n.peak) * n.weight) / (offset + peak)
                    : n.weight;
                ema = ema * n.decay + contribution;
                continue;
            }
            if (n.rising && n.first >= peak && offset + n.first > 0.0f) {
                peak = n.peak;
                ema *= n.decay;
                continue;
            }
            if (offset + n.peak <= 0.0f) {
                peak = fmax(peak, n.peak);
                ema = ema * n.decay + n.weight;
                continue;
            }
            if (n.count == 1) {
                peak = fmax(peak, n.peak);
                float raw = offset + peak > 0.0f
                    ? (peak - n.last) / (offset + peak) : 1.0f;
                ema = ema * n.decay + raw * n.weight;
                continue;
            }
        }
        if (hi - lo == 64) {
            device float2* samples = hsl_samples(tree, tree_size);
            for (int slot = max(lo, begin); slot < min(hi, end); ++slot) {
                float2 row = samples[slot];
                peak = fmax(peak, row.y);
                float raw = offset + peak > 0.0f
                    ? (peak - row.x) / (offset + peak) : 1.0f;
                ema = ema * (1.0f - alpha) + raw * alpha;
            }
            continue;
        }
        int mid = (lo + hi) / 2;
        pending_index[count] = index * 2 + 1;
        pending_lo[count] = mid;
        pending_hi[count++] = hi;
        pending_index[count] = index * 2;
        pending_lo[count] = lo;
        pending_hi[count++] = mid;
    }
}

inline float2 hsl_signal_peak(
    thread HslWindow& w, device HslNode* tree,
    float offset, float entry_reference, thread float& result_peak
) {
    if (w.count == 0 || !isfinite(offset)) return float2(NAN, NAN);
    device float2* samples = hsl_samples(tree, w.tree_size);
    float2 first = samples[w.head];
    float peak = fmax(first.y, entry_reference);
    float ema = offset + peak > 0.0f
        ? (peak - first.x) / (offset + peak) : 1.0f;
    // Seed EMA with the first raw drawdown, then visit the remaining ring.
    int begin = (w.head + 1) % w.capacity;
    int remaining = w.count - 1;
    int n = min(remaining, w.capacity - begin);
    hsl_visit(tree, w.tree_size, begin, begin + n, offset, w.alpha, peak, ema);
    hsl_visit(tree, w.tree_size, 0, remaining - n, offset, w.alpha, peak, ema);
    float last = samples[(w.head + w.count - 1) % w.capacity].x;
    float raw = offset + peak > 0.0f ? (peak - last) / (offset + peak) : 1.0f;
    result_peak = peak;
    return float2(raw, ema);
}

inline float2 hsl_signal(
    thread HslWindow& w, device HslNode* tree,
    float offset, float entry_reference
) {
    float peak;
    return hsl_signal_peak(w, tree, offset, entry_reference, peak);
}

#endif // Legacy observation window.

#if PASSIVBOT_HSL_FACTUAL_ONLY
// Reporting/control state shared with fresh factual reconstruction. No window,
// retained observation caches or device pointers belong to this controller.
struct HslController {
    float raw;
    float ema;
    int last_observed;
    int flat_minute;
    int action;
    bool exposed;
};

inline HslController hsl_controller_init(
    device HslNode* tree, int capacity, int tree_size, float span
) {
    HslController h;
    h.raw = h.ema = 0.0f;
    h.last_observed = h.flat_minute = -1;
    h.action = 0;
    h.exposed = false;
    return h;
}
#else
// Simulator policy over the latest factual episode. Storage is caller-owned so
// temporal replay can rebind buffers without retaining device pointers in state.
struct HslController {
    HslWindow window;
    float origin;
    float last_realized;
    float raw;
    float ema;
    int last_observed;
    int mark_head;
    int mark_count;
    int mark_observed;
    int episode_seed;
    int flat_minute;
    float completed_entry_reference;
    int completed_reference_minute;
    int action; // 0 normal, 1 cooldown, 3 current panic
    bool exposed;
    bool completed;
    bool scalar_ready;
    float scalar_offset;
    float scalar_peak;
    float scalar_raw;
    float scalar_ema;
    float scalar_baseline;
    int scalar_minute;
};

inline HslController hsl_controller_init(
    device HslNode* tree, int capacity, int tree_size, float span
) {
    HslController h;
    h.window = hsl_init(tree, capacity, tree_size, span);
    h.origin = h.last_realized = h.raw = h.ema = 0.0f;
    h.last_observed = h.episode_seed = h.flat_minute = -1;
    h.mark_head = h.mark_count = 0;
    h.mark_observed = -1;
    h.completed_entry_reference = -INFINITY;
    h.completed_reference_minute = -1;
    h.action = 0;
    h.exposed = h.completed = h.scalar_ready = false;
    h.scalar_offset = h.scalar_peak = h.scalar_raw = h.scalar_ema = h.scalar_baseline = 0.0f;
    h.scalar_minute = -1;
    return h;
}

inline bool hsl_record(
    thread HslController& h, device HslNode* tree,
    device int* times, device float* realized_rows,
    int minute, float realized, float upnl
) {
    float relative = realized - h.origin;
    if (!hsl_push(h.window, tree, times, minute, relative + upnl)) return false;
    realized_rows[(h.window.head + h.window.count - 1) % h.window.capacity] = relative;
    return true;
}

inline bool hsl_observe(
    thread HslController& h, device HslNode* tree,
    device int* times, device float* realized_rows,
    int minute, int lookback, float budget, float realized, float upnl,
    bool exposed, bool terminal, float threshold, float cooldown, bool never_restart
) {
    if (!(budget > 0.0f) || !isfinite(budget)
        || !isfinite(realized) || !isfinite(upnl) || (terminal && exposed)
        || !(threshold > 0.0f && threshold <= 1.0f) || !isfinite(cooldown)
        || cooldown < 0.0f || lookback < 1 || !(h.window.alpha > 0.0f)) return false;
    if (minute < h.last_observed) {
        // Forced delisting follows order construction. Its factual fill is at
        // the bar start, before the mark used to construct those orders.
        // Retract only that latest provisional observation, including coarser
        // candle intervals, before observing the terminal
        // fill; ordinary out-of-order observations remain invalid. The caller's
        // lookback + 2 capacity preserves the preceding window during append.
        if (!(terminal && !exposed && h.exposed
                && minute >= h.mark_observed
                && h.window.last_minute == h.last_observed
                && h.mark_count < h.window.capacity)) return false;
        h.window.head = h.mark_head;
        h.window.count = h.mark_count;
        h.last_observed = h.mark_observed;
        h.scalar_ready = false;
        int last = (h.window.head + h.window.count - 1) % h.window.capacity;
        if (h.window.count == 0 || times[last] > minute) {
            // First observed exposure: its flat seed and closing fill share
            // this factual minute, without an invented preceding EMA step.
            h.window.count = 0;
            h.window.last_minute = -1;
            h.window.block_prefix = hsl_empty_node();
            h.episode_seed = minute;
            if (!hsl_record(h, tree, times, realized_rows,
                    minute, h.origin, 0.0f)) return false;
        } else {
            h.window.last_minute = times[last];
            h.window.block_prefix = hsl_empty_node();
            device float2* rows = hsl_samples(tree, h.window.tree_size);
            for (int slot = (last / 64) * 64; slot < last; ++slot) {
                h.window.block_prefix = hsl_join(h.window.block_prefix,
                    hsl_sample(rows[slot], h.window.alpha));
            }
            HslNode block = hsl_join(h.window.block_prefix,
                hsl_sample(rows[last], h.window.alpha));
            tree[h.window.tree_size + last / 64] = block;
            if (last % 64 == 63) hsl_set(tree, h.window.tree_size, last / 64, block);
        }
    }
    int start = minute - lookback;
    if ((exposed || terminal) && !h.exposed) {
        // A new exposure, even during cooldown, discards the completed episode.
        // The last flat observation seeds the next episode's zero drawdown.
        h.window.head = h.window.count = 0;
        h.window.last_minute = -1;
        h.origin = h.last_realized;
        h.completed = false;
        h.completed_entry_reference = -INFINITY;
        h.completed_reference_minute = -1;
        h.scalar_ready = false;
        h.flat_minute = -1;
        h.episode_seed = max(start, h.last_observed >= 0 ? h.last_observed : minute);
        if (!hsl_record(h, tree, times, realized_rows,
                h.episode_seed, h.origin, 0.0f)) return false;
    }
    if (minute > h.last_observed) {
        h.mark_head = h.window.head;
        h.mark_count = h.window.count;
        h.mark_observed = h.last_observed;
    }
    bool clipped = h.window.count > 0 && times[h.window.head] < start;
    hsl_expire(h.window, tree, times, start);
    if (exposed || terminal) {
        if (!hsl_record(h, tree, times, realized_rows, minute, realized, upnl)) return false;
        if (terminal) {
            h.completed = true;
            h.flat_minute = minute;
        }
    }
    h.action = 0;
    h.raw = h.ema = 0.0f;
    if (exposed || terminal || (h.completed && h.flat_minute >= start)) {
        // An active incomplete episode needs its estimated entry-loss peak.
        // At flattening, bind that reference to the first retained observation,
        // like Rust's initial entry_reference_delta. Budget changes or rebuilding
        // the scalar cache preserve it, but expiration of that observation drops
        // it. Never synthesize a new reference for a completed flat episode.
        float reference = -INFINITY;
        if (exposed || terminal) {
            if (start > h.episode_seed && h.window.count > 0) {
                reference = realized_rows[h.window.head];
            }
            if (terminal) {
                h.completed_entry_reference = reference;
                h.completed_reference_minute = isfinite(reference)
                    ? times[h.window.head] : -1;
            }
        } else if (h.completed_reference_minute >= start) {
            reference = h.completed_entry_reference;
        }
        float offset = budget - (realized - h.origin);
        float2 score;
        if (h.scalar_ready && !clipped && offset == h.scalar_offset) {
            if (exposed || terminal) {
                float value = (realized - h.origin) + upnl;
                h.scalar_peak = fmax(h.scalar_peak, value);
                h.scalar_raw = offset + h.scalar_peak > 0.0f
                    ? (h.scalar_peak - value) / (offset + h.scalar_peak) : 1.0f;
                if (minute != h.scalar_minute) h.scalar_baseline = h.scalar_ema;
                h.scalar_ema = h.window.count == 1 ? h.scalar_raw
                    : h.window.alpha * h.scalar_raw + (1.0f - h.window.alpha) * h.scalar_baseline;
            }
            score = float2(h.scalar_raw, h.scalar_ema);
        } else {
            score = hsl_signal_peak(h.window, tree, offset, reference, h.scalar_peak);
            h.scalar_raw = score.x;
            h.scalar_ema = score.y;
            h.scalar_baseline = 0.0f;
            if (h.window.count > 1) {
                --h.window.count;
                h.scalar_baseline = hsl_signal(h.window, tree, offset, reference).y;
                ++h.window.count;
            }
            h.scalar_offset = offset;
            h.scalar_ready = true;
        }
        h.scalar_minute = h.window.last_minute;
        if (!isfinite(score.x) || !isfinite(score.y)) return false;
        bool red = fmin(score.x, score.y) > threshold;
        if (exposed || terminal) { h.raw = score.x; h.ema = score.y; }
        h.action = exposed ? (red ? 3 : 0)
            : (red && (never_restart || float(minute) < float(h.flat_minute) + cooldown) ? 1 : 0);
    } else if (h.completed && h.flat_minute < start) {
        h.completed = false;
        h.completed_entry_reference = -INFINITY;
        h.completed_reference_minute = -1;
    }
    h.last_observed = minute;
    h.last_realized = realized;
    h.exposed = exposed;
    return true;
}

#endif // Legacy observation controller.
