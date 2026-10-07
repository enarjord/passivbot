"""Exercise the shared GPU episode boundary code on CUDA and Apple MPS."""

from functools import lru_cache
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import compile_shader, gpu_device, synchronize
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable"
)

_GPU = Path(__file__).resolve().parents[2] / "passivbot-rust" / "src" / "gpu"
_KERNELS = (
    "mps_ema_anchor_directional.metal",
    "mps_trailing_martingale_directional.metal",
    "mps_ema_anchor_multicoin_long.metal",
    "mps_trailing_martingale_multicoin.metal",
)


def _source(name):
    source = (_GPU / name).read_text()
    source = source.replace(
        "// PASSIVBOT_ADAPTIVE_TIMING", (_GPU / "mps_adaptive_timing.metal").read_text()
    )
    for marker, filename in (
        ("UNSTUCK_EMA", "mps_unstuck_ema_common.metal"),
        ("HSL", "mps_hsl_common.metal"),
        ("BTC_RISK", "mps_btc_risk_common.metal"),
        ("EQUITY_BALANCE_DIFF", "mps_equity_balance_diff_common.metal"),
        ("ENTRY_INTERVAL", "mps_entry_interval_common.metal"),
        ("MULTICOIN", "mps_multicoin_common.metal"),
    ):
        source = source.replace(
            f"// PASSIVBOT_{marker}_COMMON", (_GPU / filename).read_text()
        )
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "#define PASSIVBOT_HSL_CAPACITY 64\n"
        "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
        "#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 1\n"
        + (_GPU / "mps_hsl.metal").read_text()
        + source
    )


_PROBE = r"""
kernel void episode_boundary_probe(
    constant float* params, device HslNode* trees, device int* rows,
    device float* output, constant int& scope_held, constant int& opposite_held,
    constant float& terminal_loss, constant int& short_owner,
    uint b [[thread_position_in_grid]]
) {
    HslState h = load_hsl(params, 0, 0);
    HslState opposite = load_hsl(params, 0, 0);
    bind_hsl(h, trees, rows, 0, 64, 1, 1440, true, short_owner == 0);
    bind_hsl(opposite, trees, rows, 1, 64, 1, 1440, true, short_owner != 0);
    thread HslState& owner = h.signal_mode == HSL_SIGNAL_UNIFIED && short_owner != 0
        ? opposite : h;
    observe_hsl(owner, 1000, 0, 0, false, 0, false);
    observe_hsl(owner, 1000, 0, -400, true, 1, false);
    output[0] = owner.red_active_now;
    bool finished = finish_hsl_scoped_episode_at_flat(
        h, &opposite, scope_held != 0, opposite_held != 0,
        1000 - terminal_loss, 1000, -terminal_loss, -terminal_loss, 2, 60000);
    output[1] = finished;
    output[2] = owner.hsl.action;
    output[3] = owner.sampled_drawdown_raw;
    output[4] = owner.triggers;
    output[5] = h.hsl.action;
    output[6] = opposite.hsl.action;
    if (finished) {
        // A later balance change reclassifies the completed episode. There is
        // no remembered RED decision, even with restart policy never.
        observe_hsl(owner, 100000, -terminal_loss, 0, false, 3, false);
        output[7] = owner.hsl.action;
        observe_hsl(owner, 1000 - terminal_loss, -terminal_loss, 0, false, 4, false);
        output[8] = owner.hsl.action;
        // Renewed exposure discards the previous episode and its cooldown.
        observe_hsl(owner, 1000 - terminal_loss, -terminal_loss, 0, true, 4, false);
        output[9] = owner.hsl.action;
        output[10] = owner.sampled_drawdown_raw;
        output[11] = owner.hsl_valid;
    }
}
"""


@lru_cache(maxsize=None)
def _library(name):
    return compile_shader(_source(name) + _PROBE)


def _run(
    name,
    mode=2,
    loss=400,
    scope_held=False,
    opposite_held=False,
    short_owner=False,
    never=False,
):
    params = torch.tensor(
        [1, 0.1, 1, 60, 2 if never else 0, mode, 1], dtype=torch.float32, device=gpu_device()
    )
    trees = torch.empty((36, 32), dtype=torch.uint8, device=gpu_device())
    rows = torch.empty(256, dtype=torch.int32, device=gpu_device())
    output = torch.zeros(12, dtype=torch.float32, device=gpu_device())
    _library(name).episode_boundary_probe(
        params,
        trees,
        rows,
        output,
        int(scope_held),
        int(opposite_held),
        float(loss),
        int(short_owner),
        threads=1,
    )
    synchronize()
    return output.cpu().tolist()


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["unified", "pside", "coin"])
@pytest.mark.parametrize("never", [False, True])
def test_terminal_red_starts_reclassifiable_cooldown_and_reentry_clears_it(
    name, mode, never
):
    values = _run(name, mode, never=never)
    assert values[:3] == [1, 1, 1]
    assert values[3] == pytest.approx(0.4)
    assert values[4:6] == [1, 1]
    assert values[6] == (1 if mode == 0 else 0)
    assert values[7:] == [0, 1, 0, 0, 1]


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_green_terminal_voids_prior_red_without_cooldown(name, mode):
    values = _run(name, mode, loss=10)
    assert values[:3] == [1, 1, 0]
    assert values[3] == pytest.approx(0.01)
    assert values[4:] == [0, 0, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize(
    "mode,scope_held,opposite_held,finished",
    [
        (0, False, True, False),
        (0, True, False, False),
        (1, True, False, False),
        (1, False, True, True),
        (2, False, True, True),
    ],
)
def test_only_configured_scope_flatness_finalizes_episode(
    mode, scope_held, opposite_held, finished
):
    values = _run(_KERNELS[0], mode, scope_held=scope_held, opposite_held=opposite_held)
    assert values[1] == int(finished)
    assert values[2] == (1 if finished else 3)
    assert values[4] == int(finished)


@pytest.mark.parametrize("name", _KERNELS)
def test_unified_short_owner_mirrors_finalized_signal(name):
    values = _run(name, mode=0, short_owner=True)
    assert values[2] == values[5] == values[6] == 1
    assert values[7:] == [0, 1, 0, 0, 1]


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("balance", [0.0, -100.0, float("nan")])
def test_terminal_cash_exhaustion_leaves_liquidation_to_kernel(name, mode, balance):
    # Closing losses exhaust raw cash before the end-of-bar liquidation sample.
    # A non-finite budget is still unavailable, rather than an ordinary loss.
    probe = _PROBE.replace(
        "if (finished) {",
        "output[7] = owner.hsl_valid; output[8] = owner.hsl.last_observed; return;\n"
        "    if (finished) {",
    )
    library = compile_shader(_source(name) + probe)
    params = torch.tensor([1, 0.1, 1, 60, 0, mode, 1], device=gpu_device())
    trees = torch.empty((36, 32), dtype=torch.uint8, device=gpu_device())
    rows = torch.empty(256, dtype=torch.int32, device=gpu_device())
    output = torch.zeros(12, device=gpu_device())
    library.episode_boundary_probe(
        params, trees, rows, output, 0, 0, 1000.0 - balance, 0, threads=1
    )
    values = output.cpu().tolist()
    assert values[1] == 1
    assert values[4] == 0  # No invented terminal RED/cooldown event.
    assert values[7] == int(balance == balance)
    assert values[8] == 1  # No observation with a depleted or unusable budget.


@pytest.mark.parametrize("finish", ["flat", "green", "censored"])
def test_panic_loss_report_finishes_segments_without_retaining_panic(finish):
    probe = r"""
kernel void loss_report_probe(constant float* params, device HslNode* trees,
    device int* rows, device float* output, constant int& finish,
    uint b [[thread_position_in_grid]]) {
    HslState h = load_hsl(params, 0, 0);
    bind_hsl(h, trees, rows, 0, 64, 1, 1440, true, true);
    observe_hsl(h, 1000, 0, 0, false, 0, false);
    observe_hsl(h, 1000, 0, -400, true, 1, false);
    record_hsl_panic_fill(h, -10, 1000);
    record_hsl_panic_fill(h, -15, 975);
    if (finish == 0) {
        finish_hsl_episode_at_flat(h, 600, 1000, -400, 2, 60000);
    } else if (finish == 1) {
        observe_hsl(h, 975, -25, 0, true, 2, false);
    } else {
        // Repeated exports include a still-open segment without consuming it.
        HslOutputAggregate report = init_hsl_output_aggregate(0, 0);
        accumulate_hsl_output(report, h, false, 2);
        output[8] = report.panic_loss_drawdown_sum;
        output[9] = h.panic_event_loss;
        report = init_hsl_output_aggregate(0, 0);
        accumulate_hsl_output(report, h, false, 2);
        output[10] = report.panic_loss_drawdown_count;
        output[11] = h.panic_loss_drawdown_count;
        finish_hsl_panic_loss(h);
    }
    output[0] = h.panic_loss_drawdown_sum;
    output[1] = h.panic_loss_drawdown_count;
    output[2] = h.hsl.action;
    // Finishing twice must not duplicate accounting; a new segment uses a new denominator.
    finish_hsl_panic_loss(h);
    record_hsl_panic_fill(h, -40, 800);
    finish_hsl_panic_loss(h);
    output[3] = h.panic_loss_drawdown_min;
    output[4] = h.panic_loss_drawdown_max;
    output[5] = h.panic_loss_drawdown_sum;
    output[6] = h.panic_loss_drawdown_count;
    output[7] = h.panic_close_loss_sum;
}
"""
    library = compile_shader(_source(_KERNELS[0]) + probe)
    params = torch.tensor([1, 0.1, 1, 60, 0, 2, 1], device=gpu_device())
    trees = torch.empty((36, 32), dtype=torch.uint8, device=gpu_device())
    rows = torch.empty(256, dtype=torch.int32, device=gpu_device())
    result = torch.zeros(12, device=gpu_device())
    library.loss_report_probe(
        params,
        trees,
        rows,
        result,
        ["flat", "green", "censored"].index(finish),
        threads=1,
    )
    synchronize()
    values = result.cpu().tolist()
    assert values[:2] == pytest.approx([0.025, 1])
    assert values[2] == {"flat": 1, "green": 0, "censored": 3}[finish]
    assert values[3:8] == pytest.approx([0.025, 0.05, 0.075, 2, 65])
    if finish == "censored":
        assert values[8:] == pytest.approx([0.025, 25, 1, 0])
