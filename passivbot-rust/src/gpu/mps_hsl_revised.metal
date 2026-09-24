// Revised HSL arithmetic over factual PNL + UPNL samples. No trading state or
// episode inference lives here. Both Metal and CUDA consume this scalar source.
// A candidate owns a bounded ring of compact samples and a tree over 64-sample
// blocks. Partial boundary blocks are visited directly; complete blocks aggregate.
struct RevisedHslNode {
    float peak;
    float first;
    float last;
    float loss;
    float weight;
    float decay;
    int count;
    int rising;
};

struct RevisedHslWindow {
    int head;
    int count;
    int capacity;
    int tree_size;
    float alpha;
    int last_minute;
    RevisedHslNode block_prefix; // Completed samples before the replaceable last row.
};

inline RevisedHslNode revised_hsl_empty_node() {
    RevisedHslNode n;
    n.peak = -INFINITY;
    n.first = n.last = n.loss = n.weight = 0.0f;
    n.decay = 1.0f;
    n.count = 0;
    n.rising = 1;
    return n;
}

inline RevisedHslNode revised_hsl_join(RevisedHslNode l, RevisedHslNode r) {
    if (l.count == 0) return r;
    if (r.count == 0) return l;
    RevisedHslNode n;
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

inline int revised_hsl_storage_nodes(int capacity, int tree_size) {
    // Four two-float samples fit in one 32-byte node-sized allocation.
    return 2 * tree_size + (capacity + 3) / 4;
}

inline device float2* revised_hsl_samples(device RevisedHslNode* tree, int tree_size) {
    return reinterpret_cast<device float2*>(tree + 2 * tree_size);
}

inline RevisedHslNode revised_hsl_sample(float2 row, float alpha) {
    RevisedHslNode n;
    n.peak = row.y;
    n.first = n.last = row.x;
    n.loss = alpha * (row.y - row.x);
    n.weight = alpha;
    n.decay = 1.0f - alpha;
    n.count = 1;
    n.rising = row.x == row.y;
    return n;
}

inline void revised_hsl_set(
    device RevisedHslNode* tree, int tree_size, int slot, RevisedHslNode n
) {
    int index = tree_size + slot;
    tree[index] = n;
    // Only complete aligned blocks can be consumed by a chronological range
    // query. Updating completed right edges makes append amortized O(1).
    while (index > 1 && (index & 1) != 0) {
        index /= 2;
        tree[index] = revised_hsl_join(tree[index * 2], tree[index * 2 + 1]);
    }
}

inline RevisedHslWindow revised_hsl_init(
    device RevisedHslNode* tree, int capacity, int tree_size, float span
) {
    RevisedHslWindow w;
    w.head = w.count = 0;
    w.capacity = capacity;
    w.tree_size = tree_size;
    w.alpha = 2.0f / (span + 1.0f);
    w.last_minute = -1;
    w.block_prefix = revised_hsl_empty_node();
    for (int i = 0; i < 2 * tree_size; ++i) tree[i] = revised_hsl_empty_node();
    return w;
}

inline void revised_hsl_expire(
    thread RevisedHslWindow& w, device RevisedHslNode* tree,
    device int* times, int first_minute
) {
    while (w.count > 0 && times[w.head] < first_minute) {
        // Expired leaves lie outside both queried ring ranges.
        w.head = (w.head + 1) % w.capacity;
        --w.count;
    }
}

inline bool revised_hsl_push(
    thread RevisedHslWindow& w, device RevisedHslNode* tree,
    device int* times, int minute, float value
) {
    if (!isfinite(value) || minute < w.last_minute) return false;
    const bool replace = w.count > 0 && minute == w.last_minute;
    if (!replace && w.count == w.capacity) return false;
    int slot = (w.head + w.count - (replace ? 1 : 0)) % w.capacity;
    device float2* samples = revised_hsl_samples(tree, w.tree_size);
    float peak = replace ? fmax(samples[slot].y, value) : value;
    samples[slot] = float2(value, peak);
    int block = slot / 64;
    if (!replace) {
        w.block_prefix = slot % 64 == 0 ? revised_hsl_empty_node()
            : tree[w.tree_size + block];
    }
    RevisedHslNode n = revised_hsl_join(w.block_prefix,
        revised_hsl_sample(samples[slot], w.alpha));
    tree[w.tree_size + block] = n;
    if (slot % 64 == 63) revised_hsl_set(tree, w.tree_size, block, n);
    times[slot] = minute;
    w.last_minute = minute;
    if (!replace) ++w.count;
    return true;
}

// Iterative left-to-right traversal; Metal prohibits recursive functions.
// At most one right sibling per level is pending (90d of minutes < 2^18).
inline void revised_hsl_visit(
    device RevisedHslNode* tree, int tree_size, int begin, int end,
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
        RevisedHslNode n = tree[index];
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
            device float2* samples = revised_hsl_samples(tree, tree_size);
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

inline float2 revised_hsl_signal_peak(
    thread RevisedHslWindow& w, device RevisedHslNode* tree,
    float offset, float entry_reference, thread float& result_peak
) {
    if (w.count == 0 || !isfinite(offset)) return float2(NAN, NAN);
    device float2* samples = revised_hsl_samples(tree, w.tree_size);
    float2 first = samples[w.head];
    float peak = fmax(first.y, entry_reference);
    float ema = offset + peak > 0.0f
        ? (peak - first.x) / (offset + peak) : 1.0f;
    // Seed EMA with the first raw drawdown, then visit the remaining ring.
    int begin = (w.head + 1) % w.capacity;
    int remaining = w.count - 1;
    int n = min(remaining, w.capacity - begin);
    revised_hsl_visit(tree, w.tree_size, begin, begin + n, offset, w.alpha, peak, ema);
    revised_hsl_visit(tree, w.tree_size, 0, remaining - n, offset, w.alpha, peak, ema);
    float last = samples[(w.head + w.count - 1) % w.capacity].x;
    float raw = offset + peak > 0.0f ? (peak - last) / (offset + peak) : 1.0f;
    result_peak = peak;
    return float2(raw, ema);
}

inline float2 revised_hsl_signal(
    thread RevisedHslWindow& w, device RevisedHslNode* tree,
    float offset, float entry_reference
) {
    float peak;
    return revised_hsl_signal_peak(w, tree, offset, entry_reference, peak);
}

// Simulator policy over the latest factual episode. Storage is caller-owned so
// temporal replay can rebind buffers without retaining device pointers in state.
struct RevisedHslController {
    RevisedHslWindow window;
    float origin;
    float last_realized;
    float raw;
    float ema;
    int last_observed;
    int episode_seed;
    int flat_minute;
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

inline RevisedHslController revised_hsl_controller_init(
    device RevisedHslNode* tree, int capacity, int tree_size, float span
) {
    RevisedHslController h;
    h.window = revised_hsl_init(tree, capacity, tree_size, span);
    h.origin = h.last_realized = h.raw = h.ema = 0.0f;
    h.last_observed = h.episode_seed = h.flat_minute = -1;
    h.action = 0;
    h.exposed = h.completed = h.scalar_ready = false;
    h.scalar_offset = h.scalar_peak = h.scalar_raw = h.scalar_ema = h.scalar_baseline = 0.0f;
    h.scalar_minute = -1;
    return h;
}

inline bool revised_hsl_record(
    thread RevisedHslController& h, device RevisedHslNode* tree,
    device int* times, device float* realized_rows,
    int minute, float realized, float upnl
) {
    float relative = realized - h.origin;
    if (!revised_hsl_push(h.window, tree, times, minute, relative + upnl)) return false;
    realized_rows[(h.window.head + h.window.count - 1) % h.window.capacity] = relative;
    return true;
}

inline bool revised_hsl_observe(
    thread RevisedHslController& h, device RevisedHslNode* tree,
    device int* times, device float* realized_rows,
    int minute, int lookback, float budget, float realized, float upnl,
    bool exposed, bool terminal, float threshold, float cooldown, bool never_restart
) {
    if (minute < h.last_observed || !(budget > 0.0f) || !isfinite(budget)
        || !isfinite(realized) || !isfinite(upnl) || (terminal && exposed)
        || !(threshold > 0.0f && threshold <= 1.0f) || !isfinite(cooldown)
        || cooldown < 0.0f || lookback < 1 || !(h.window.alpha > 0.0f)) return false;
    int start = minute - lookback;
    if ((exposed || terminal) && !h.exposed) {
        // A new exposure, even during cooldown, discards the completed episode.
        // The last flat observation seeds the next episode's zero drawdown.
        h.window.head = h.window.count = 0;
        h.window.last_minute = -1;
        h.origin = h.last_realized;
        h.completed = false;
        h.scalar_ready = false;
        h.flat_minute = -1;
        h.episode_seed = max(start, h.last_observed >= 0 ? h.last_observed : minute);
        if (!revised_hsl_record(h, tree, times, realized_rows,
                h.episode_seed, h.origin, 0.0f)) return false;
    }
    bool clipped = h.window.count > 0 && times[h.window.head] < start;
    revised_hsl_expire(h.window, tree, times, start);
    if (exposed || terminal) {
        if (!revised_hsl_record(h, tree, times, realized_rows, minute, realized, upnl)) return false;
        if (terminal) {
            h.completed = true;
            h.flat_minute = minute;
        }
    }
    h.action = 0;
    h.raw = h.ema = 0.0f;
    if (exposed || terminal || (h.completed && h.flat_minute >= start)) {
        // Once the opening leaves the bounded tape, its synthetic entry basis
        // contributes a peak reference but never an extra EMA time step.
        float reference = start > h.episode_seed && h.window.count > 0
            ? realized_rows[h.window.head] : -INFINITY;
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
            score = revised_hsl_signal_peak(h.window, tree, offset, reference, h.scalar_peak);
            h.scalar_raw = score.x;
            h.scalar_ema = score.y;
            h.scalar_baseline = 0.0f;
            if (h.window.count > 1) {
                --h.window.count;
                h.scalar_baseline = revised_hsl_signal(h.window, tree, offset, reference).y;
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
    }
    h.last_observed = minute;
    h.last_realized = realized;
    h.exposed = exposed;
    return true;
}
