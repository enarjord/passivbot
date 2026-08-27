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
constant int ENTRY_INTERVAL_COLS = 131;

inline void init_entry_interval_output(
    device float* output,
    uint candidate
) {
    const int offset = int(candidate) * ENTRY_INTERVAL_COLS;
    for (int column = 0; column < ENTRY_INTERVAL_COLS; ++column) {
        output[offset + column] = 0.0f;
    }
}

inline void record_initial_entry_interval(
    device float* output,
    uint candidate,
    thread float& last_initial_entry_k,
    float current_k
) {
    if (last_initial_entry_k >= 0.0f) {
        const float gap = fmax(current_k - last_initial_entry_k, 0.0f);
        const int offset = int(candidate) * ENTRY_INTERVAL_COLS;
        output[offset + 0] += gap;
        output[offset + 1] += 1.0f;
        output[offset + 2] = fmax(output[offset + 2], gap);
        const float log_bin_scale = 127.0f / log(4000001.0f);
        const int bin = clamp(
            int(log(gap + 1.0f) * log_bin_scale), 0, 127
        );
        output[offset + 3 + bin] += 1.0f;
    }
    last_initial_entry_k = current_k;
}
#endif
