"""Bounded TM close streaming at the recursive limit and policy boundaries."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    assert not getattr(pbr, "__is_stub__", False)
    return pbr


def test_tm_close_grid_streams_bounded_groups_and_market_policy(
    require_real_passivbot_rust_module,
):
    from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
    from rust_utils import verify_loaded_runtime_extension

    pbr = require_real_passivbot_rust_module
    verify_loaded_runtime_extension()
    probe = r"""
kernel void tm_close_grid_stream_probe(
    constant float* cases [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint b [[thread_position_in_grid]]
) {
    bool short_side = cases[b * 4] > 0.5f;
    float slope = cases[b * 4 + 1];
    bool prefix = cases[b * 4 + 2] > 0.5f;
    bool market = cases[b * 4 + 3] > 0.5f;
    TmCloseGridContext context = {
        short_side, 1000.0f, 100.0f, 1000.0f, 100.0f,
        90000, 110000, 100000, 0.001f, 0,
        0.001f, 0.3f, slope, 0.0f, 0.0f, 0.0f, 0.0f,
        0.001f, 0.001f, 0.001f, 0.0f, 1.0f,
        0, 0.0f, 500, 100.0f, market, 2.0f, prefix ? 1001.0f : 1000.0f
    };
    CloseGroup first;
    recursive_grid_close_groups_after_reducer(context, 0, first);
    if (prefix) {
        context.prefix_merge_tick = first.ticks;
        context.prefix_merge_qty = 1.0f;
    }
    CloseGroup merged_wel = {0, 0.0f, 0.0f, false};
    bool has_merged_wel = merged_tm_wel_prefix_group(context, merged_wel);
    bool merged_wel_valid = prefix
        ? (has_merged_wel
           && merged_wel.ticks == first.ticks
           && merged_wel.price == first.price
           && !merged_wel.market)
        : (!has_merged_wel && merged_wel.ticks == 0 && merged_wel.qty == 0.0f);
    CloseGroup first_ordinary;
    int count = recursive_grid_close_groups_after_reducer(context, 0, first_ordinary);
    TmCloseGridIterator iterator = tm_close_grid_iterator(context);
    CloseGroup group;
    int seen = 0;
    int markets = 0;
    float quantity = 0.0f;
    bool selections_match = true;
    bool ordered = true;
    int previous_tick = has_merged_wel ? merged_wel.ticks : 0;
    bool have_previous_tick = has_merged_wel;
    while (next_tm_close_grid_group(context, iterator, group)) {
        finalize_tm_close_grid_group(context, group);
        CloseGroup selected;
        int selected_count = recursive_grid_close_groups_after_reducer(
            context, seen, selected
        );
        selections_match = selections_match && selected_count == count
            && group.ticks == selected.ticks && group.price == selected.price
            && group.qty == selected.qty && group.market == selected.market;
        if (have_previous_tick) {
            bool ascending = short_side ? slope > 0.0f : slope < 0.0f;
            ordered = ordered && (ascending
                ? previous_tick < group.ticks : previous_tick > group.ticks);
        }
        previous_tick = group.ticks;
        have_previous_tick = true;
        quantity += group.qty;
        markets += group.market ? 1 : 0;
        ++seen;
    }
    bool exhausted = !next_tm_close_grid_group(context, iterator, group);
    CloseGroup missing;
    recursive_grid_close_groups_after_reducer(context, count, missing);
    int offset = b * 10;
    output[offset] = float(count);
    output[offset + 1] = float(seen);
    output[offset + 2] = quantity;
    output[offset + 3] = merged_wel.qty;
    output[offset + 4] = float(markets);
    output[offset + 5] = selections_match ? 1.0f : 0.0f;
    output[offset + 6] = ordered ? 1.0f : 0.0f;
    output[offset + 7] = exhausted ? 1.0f : 0.0f;
    output[offset + 8] = missing.qty == 0.0f && missing.ticks == 0 ? 1.0f : 0.0f;
    output[offset + 9] = merged_wel_valid ? 1.0f : 0.0f;
}
"""
    cases = np.array(
        [(side, slope, prefix, market)
         for side in (0, 1) for slope in (-0.1, 0.0, 0.1)
         for prefix in (0, 1) for market in (0, 1)],
        dtype=np.float32,
    )
    device = gpu_device()
    inputs = torch.tensor(cases, device=device)
    output = torch.zeros((len(cases), 10), dtype=torch.float32, device=device)
    # Standalone probes need declarations for the unused portfolio kernels,
    # without allocating their HSL histories.
    source = (
        "#define PASSIVBOT_HSL_CAPACITY 1\n"
        "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 0\n"
        + pbr.mps_trailing_martingale_multicoin_source_py() + probe
    )
    library = compile_shader(
        source,
        cuda_coin_capacity=1,
        mps_coin_capacity=1,
    )
    library.tm_close_grid_stream_probe(inputs, output, threads=(len(cases), 1, 1))
    synchronize()
    for (_, slope, prefix, market), result in zip(cases, output.cpu().numpy()):
        # Nonzero exposure slope emits 500 separate one-unit rungs. A zero
        # slope requests the full position at one duplicate-merged price.
        # When the leading close shares the WEL reducer tick, that prefix is
        # merged into the protective WEL group and omitted from ordinary closes.
        raw_groups = 1 if slope == 0.0 else 500
        expected_ordinary_groups = raw_groups - (1 if prefix else 0)
        expected_ordinary_quantity = (
            (0 if prefix else 1000)
            if slope == 0.0
            else (499 if prefix else 500)
        )
        expected_merged_wel_quantity = (
            (1001 if slope == 0.0 else 2) if prefix else 0
        )
        np.testing.assert_allclose(
            result,
            [
                expected_ordinary_groups,
                expected_ordinary_groups,
                expected_ordinary_quantity,
                expected_merged_wel_quantity,
                expected_ordinary_groups if market else 0,
                1,
                1,
                1,
                1,
                1,
            ],
            rtol=0.0,
            atol=1e-4,
        )
