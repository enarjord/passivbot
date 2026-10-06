#include <metal_stdlib>
using namespace metal;

// Each replay slot holds the sum of abs(fill_qty)*price/post-fill balance
// and a fill-presence flag. Presence is separate: a zero contribution still
// makes its UTC day a filled day in the canonical CPU definition.
kernel void passivbot_weighted_volume(
    device const float2* volume_samples,
    device const float2* equity_bounds_ms,
    device float* output,
    constant int* sizes,
    uint candidate [[thread_position_in_grid]]
) {
    const int B = sizes[0];
    const int T = sizes[1];
    const int start_minute = sizes[2];
    const int interval_minutes = sizes[3];
    if (int(candidate) >= B) return;
    const float2 bounds = equity_bounds_ms[candidate];
    output[candidate] = 0.0f;
    if (isnan(bounds.x) && isnan(bounds.y)) return; // No recorded equities.
    if (!isfinite(bounds.x) || !isfinite(bounds.y)
        || bounds.x < 0.0f || bounds.y < bounds.x
        || bounds.y / (60000.0f * float(interval_minutes)) >= float(T)) {
        output[candidate] = NAN;
        return;
    }
    const int first = int(round(bounds.x / (60000.0f * float(interval_minutes))));
    const int last = int(round(bounds.y / (60000.0f * float(interval_minutes))));
    if (last >= T) {
        output[candidate] = NAN;
        return;
    }

    const int n = last - first + 1;
    int cutoffs[10];
    cutoffs[0] = 0; // The full analysis includes every actual fill.
    int subset_count = 1;
    for (int denominator = 2; denominator <= 10; ++denominator) {
        // round(n - n/denominator), ties upward, without float32 rounding.
        const int remainder = n % denominator;
        const int offset = n - n / denominator
            - (2 * remainder > denominator ? 1 : 0);
        if (offset >= n) break;
        cutoffs[subset_count++] = first + offset;
    }

    int pending = subset_count - 1;
    float total_volume = 0.0f;
    int filled_days = 0;
    long previous_fill_day = -1;
    float subset_sum = 0.0f;
    const uint base = candidate * uint(T);
    for (int k = T - 1; k >= 0; --k) {
        const float2 sample = volume_samples[base + uint(k)];
        if (sample.y > 0.0f) {
            total_volume += sample.x;
            const long day = (long(start_minute)
                + long(k) * long(interval_minutes)) / 1440;
            if (day != previous_fill_day) {
                filled_days += 1;
                previous_fill_day = day;
            }
        }
        while (pending >= 0 && k == cutoffs[pending]) {
            subset_sum += filled_days > 0
                ? total_volume / float(filled_days) : 0.0f;
            pending -= 1;
        }
    }
    output[candidate] = subset_sum / float(subset_count);
}
