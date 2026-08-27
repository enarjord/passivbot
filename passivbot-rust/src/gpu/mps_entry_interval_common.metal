// Shared opt-in initial-entry interval accumulator.
//
// Canonical Rust measures gaps between EntryInitialNormal fills independently
// for each coin and position side.  Strategy kernels own the per-stream last
// entry timestamp; this helper owns the candidate-wide bounded distribution.

#ifndef PASSIVBOT_ENTRY_INTERVAL_ENABLED
#define PASSIVBOT_ENTRY_INTERVAL_ENABLED 0
#endif

#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
constant int ENTRY_INTERVAL_BINS = 128;
constant int ENTRY_INTERVAL_STAT_COLS = 2;
constant int ENTRY_INTERVAL_COUNT_COLS = 129;

inline void init_entry_interval_output(
    device float* stats,
    device int* counts,
    uint candidate
) {
    const int stat_offset = int(candidate) * ENTRY_INTERVAL_STAT_COLS;
    for (int column = 0; column < ENTRY_INTERVAL_STAT_COLS; ++column) {
        stats[stat_offset + column] = 0.0f;
    }
    const int count_offset = int(candidate) * ENTRY_INTERVAL_COUNT_COLS;
    for (int column = 0; column < ENTRY_INTERVAL_COUNT_COLS; ++column) {
        counts[count_offset + column] = 0;
    }
}

inline void record_initial_entry_interval(
    device float* stats,
    device int* counts,
    uint candidate,
    thread float& last_initial_entry_k,
    float current_k
) {
    if (last_initial_entry_k >= 0.0f) {
        const float gap = fmax(current_k - last_initial_entry_k, 0.0f);
        const int stat_offset = int(candidate) * ENTRY_INTERVAL_STAT_COLS;
        stats[stat_offset + 0] += gap;
        stats[stat_offset + 1] = fmax(stats[stat_offset + 1], gap);
        const int count_offset = int(candidate) * ENTRY_INTERVAL_COUNT_COLS;
        counts[count_offset + 0] += 1;
        const float log_bin_scale = 127.0f / log(4000001.0f);
        const int bin = clamp(
            int(log(gap + 1.0f) * log_bin_scale), 0, 127
        );
        counts[count_offset + 1 + bin] += 1;
    }
    last_initial_entry_k = current_k;
}
#endif
