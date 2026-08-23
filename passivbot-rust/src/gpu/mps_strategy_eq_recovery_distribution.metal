#include <metal_stdlib>
using namespace metal;

constant int RECOVERY_METRIC_COLS = 7;
constant float RECOVERY_FAIL_CLOSED_SENTINEL = -3.402823466e+38f;

inline uint recovery_histogram_value_at_rank(
    device const uint* histogram,
    const uint offset,
    const uint slot_count,
    const uint rank
) {
    uint cumulative = 0;
    for (uint duration = 0; duration < slot_count; ++duration) {
        cumulative += histogram[offset + duration];
        if (cumulative > rank) return duration;
    }
    return slot_count > 0 ? slot_count - 1 : 0;
}

inline float recovery_histogram_percentile(
    device const uint* histogram,
    const uint offset,
    const uint slot_count,
    const uint sample_count,
    const float percentile
) {
    if (sample_count <= 1) {
        return float(recovery_histogram_value_at_rank(
            histogram, offset, slot_count, 0
        ));
    }
    const float rank = clamp(percentile, 0.0f, 100.0f)
        * 0.01f * float(sample_count - 1);
    const uint lower_rank = uint(floor(rank));
    const uint upper_rank = uint(ceil(rank));
    const float lower = float(recovery_histogram_value_at_rank(
        histogram, offset, slot_count, lower_rank
    ));
    if (lower_rank == upper_rank) return lower;
    const float upper = float(recovery_histogram_value_at_rank(
        histogram, offset, slot_count, upper_rank
    ));
    return lower + (upper - lower) * (rank - float(lower_rank));
}

inline float recovery_histogram_mean_worst_pct(
    device const uint* histogram,
    const uint offset,
    const uint slot_count,
    const uint sample_count,
    const float percentile
) {
    if (sample_count == 0) return 0.0f;
    const uint worst_count = min(
        sample_count,
        max(uint(1), uint(float(sample_count) * percentile * 0.01f))
    );
    uint remaining = worst_count;
    float total = 0.0f;
    for (uint duration = slot_count; duration > 0 && remaining > 0; --duration) {
        const uint value = duration - 1;
        const uint take = min(histogram[offset + value], remaining);
        total += float(take) * float(value);
        remaining -= take;
    }
    return total / float(worst_count);
}

kernel void passivbot_strategy_eq_recovery_distribution(
    device const float* strategy_equity_samples,
    device uint* stack_indices,
    device uint* duration_histogram,
    device float* output,
    constant int* sizes,
    uint candidate [[thread_position_in_grid]]
) {
    const uint batch_size = uint(sizes[0]);
    const uint slot_count = uint(sizes[1]);
    if (candidate >= batch_size || slot_count == 0) return;

    const uint offset = candidate * slot_count;
    const uint output_offset = candidate * RECOVERY_METRIC_COLS;
    if (strategy_equity_samples[offset] == RECOVERY_FAIL_CLOSED_SENTINEL) {
        const float full_horizon = float(slot_count - 1);
        for (uint metric = 0; metric < RECOVERY_METRIC_COLS; ++metric) {
            output[output_offset + metric] = full_horizon;
        }
        return;
    }
    uint stack_size = 0;
    uint sample_count = 0;
    uint final_slot = 0;

    for (uint slot = 0; slot < slot_count; ++slot) {
        const float value = strategy_equity_samples[offset + slot];
        if (!isfinite(value)) continue;
        sample_count += 1;
        final_slot = slot;
        while (stack_size > 0) {
            const uint pending_slot = stack_indices[offset + stack_size - 1];
            if (!(value > strategy_equity_samples[offset + pending_slot])) break;
            duration_histogram[offset + slot - pending_slot] += 1;
            stack_size -= 1;
        }
        stack_indices[offset + stack_size] = slot;
        stack_size += 1;
    }

    while (stack_size > 0) {
        const uint pending_slot = stack_indices[offset + stack_size - 1];
        duration_histogram[offset + final_slot - pending_slot] += 1;
        stack_size -= 1;
    }

    if (sample_count == 0) {
        for (uint metric = 0; metric < RECOVERY_METRIC_COLS; ++metric) {
            output[output_offset + metric] = 0.0f;
        }
        return;
    }

    float duration_sum = 0.0f;
    uint duration_max = 0;
    for (uint duration = 0; duration < slot_count; ++duration) {
        const uint count = duration_histogram[offset + duration];
        duration_sum += float(count) * float(duration);
        if (count > 0) duration_max = duration;
    }
    output[output_offset + 0] = duration_sum / float(sample_count);
    output[output_offset + 1] = recovery_histogram_percentile(
        duration_histogram, offset, slot_count, sample_count, 50.0f
    );
    output[output_offset + 2] = recovery_histogram_percentile(
        duration_histogram, offset, slot_count, sample_count, 95.0f
    );
    output[output_offset + 3] = recovery_histogram_percentile(
        duration_histogram, offset, slot_count, sample_count, 99.0f
    );
    output[output_offset + 4] = recovery_histogram_mean_worst_pct(
        duration_histogram, offset, slot_count, sample_count, 5.0f
    );
    output[output_offset + 5] = recovery_histogram_mean_worst_pct(
        duration_histogram, offset, slot_count, sample_count, 1.0f
    );
    output[output_offset + 6] = float(duration_max);
}
