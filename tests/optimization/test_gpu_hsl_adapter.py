"""Actual-device checks of the shared native risk adapter's event boundaries."""

from pathlib import Path

import numpy as np
import pytest


_PROBE = r"""
kernel void v9_coin_budget_probe(
    constant float* params, constant float* settings, device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b != 0) return;
    HslState aggregate = load_hsl(params, 0, 0);
    HslState coins[2];
    HslReplayContext context = hsl_replay_context(params, settings, 2, 8, true);
    bind_hsl(aggregate, nullptr, nullptr, 0, 0, 0, 0, true, true);
    for (int c = 0; c < 2; ++c) {
        coins[c] = load_hsl(params, 0, 0);
        bind_hsl(coins[c], nullptr, nullptr, 0, 0, 0, 0, true, true);
        coins[c].slot_count = 2.0f;
        coins[c].risk_size = c == 0 ? 10.0f : 1.0f;
        coins[c].risk_basis = 100.0f;
        coins[c].risk_mark = c == 0 ? 100.0f : 80.0f;
    }
    attach_hsl_replay_side(context, aggregate, coins, 0, false);
    for (int c = 0; c < 2; ++c)
        hsl_observe_current_scope(coins[c], context, 0, c, 1000.0f, 0.0f, 0);
    output[0] = coins[1].decay.drawdown;
    output[1] = coins[1].decay.ratio;
    // Coin zero realizes a $600 loss. Coin one's inventory and causal mark do
    // not move, but its contemporaneous budget falls from $500 to $200.
    hsl_before_fill(coins[0], 1000.0f, 40.0f, 0);
    hsl_after_fill(coins[0], 400.0f, -600.0f, 0.0f, 0.0f, 40.0f, 0);
    output[2] = coins[1].decay.drawdown;
    output[3] = coins[1].decay.ratio;
    output[4] = coins[1].decay.ema;
    output[5] = float(hsl_mode(coins[1], true));
    output[6] = coins[1].risk_mark;
    output[7] = float(coins[0].hsl_valid && coins[1].hsl_valid);
    HslState inactive = load_hsl(params, 0, 0);
    bind_hsl(inactive, nullptr, nullptr, 0, 0, 0, 0, true, true);
    output[8] = float(hsl_coin_scope_active(inactive, 0, false));
    output[9] = float(inactive.hsl_valid);
    output[10] = float(inactive.hsl.last_observed);
    hsl_coin_scope_active(inactive, 0, true);
    output[11] = float(inactive.hsl_valid);
}

kernel void v9_entry_red_time_probe(
    constant float* params, constant float* settings, device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b != 0) return;
    HslState aggregate = load_hsl(params, 0, 0);
    HslState coins[1];
    coins[0] = load_hsl(params, 0, 0);
    bind_hsl(aggregate, nullptr, nullptr, 0, 0, 0, 0, true, true);
    bind_hsl(coins[0], nullptr, nullptr, 0, 0, 0, 0, true, true);
    HslReplayContext context = hsl_replay_context(params, settings, 1, 8, true);
    attach_hsl_replay_side(context, aggregate, coins, 0, false);
    HslTimeObservation time = init_hsl_time_observation();
    record_hsl_time_observation(time, 0.0f, 0);
    hsl_before_fill(coins[0], 1000.0f, 100.0f, 1);
    hsl_after_fill(coins[0], 999.9f, -0.1f, 1.0f, 100.0f, 100.0f, 1);
    HslFillTimeReport fill = observed_multicoin_hsl_fill_time(aggregate, coins, 1, 1, 1);
    if (fill.observed) record_hsl_time_observation(time, 0.0f, fill.tier);
    thread HslState& scope = aggregate.signal_mode == HSL_SIGNAL_COIN ? coins[0] : aggregate;
    hsl_observe_current_scope(scope, context, 0,
        aggregate.signal_mode == HSL_SIGNAL_COIN ? 0 : -1, 999.9f, 0.0f, 2);
    record_hsl_time_observation(time, 1.0f, hsl_report_tier(scope));
    output[0] = float(fill.observed);
    output[1] = time.red_steps;
    output[2] = time.observed_steps;
}
"""

_ENTRY_PROBE = r"""
kernel void v9_pending_entry_probe(
    constant float* params, constant float* settings, constant float* bars,
    constant int* ticks, constant int* touch_min_qty_relation,
    constant float* overrides, constant int& preceding_close,
    device float* output,
    uint b [[thread_position_in_grid]]
) {
    if (b != 0) return;
#if V9_TM_PROBE
    TrailingMartingaleMulticoinSideState side = {};
#else
    EmaMulticoinSideState side = {};
#endif
    side.hsl = load_hsl(params, 0, 0);
    bind_hsl(side.hsl, nullptr, nullptr, 0, 0, 0, 0, true, true);
    for (int c = 0; c < 2; ++c) {
        side.coin_hsl[c] = load_hsl(params, 0, 0);
        bind_hsl(side.coin_hsl[c], nullptr, nullptr, 0, 0, 0, 0, true, true);
        side.coin_hsl[c].slot_count = 2.0f;
    }
    HslReplayContext context = hsl_replay_context(bars, settings, 2, COIN_COLS, true);
    attach_hsl_replay_side(context, side.hsl, side.coin_hsl, 0, false);
    side.psize[1] = side.coin_hsl[1].risk_size = 1.0f;
    side.pprice[1] = side.coin_hsl[1].risk_basis = 100.0f;
    side.coin_hsl[1].risk_mark = 80.0f;
    if (preceding_close) {
        side.psize[0] = side.coin_hsl[0].risk_size = 10.0f;
        side.pprice[0] = side.coin_hsl[0].risk_basis = 100.0f;
        side.coin_hsl[0].risk_mark = 100.0f;
        side.close_qty[0] = 10.0f;
        side.close_tick[0] = 40;
        side.close_market[0] = true;
    }
    float initial_balance = preceding_close ? 1000.0f : 400.0f;
    if (side.hsl.signal_mode == HSL_SIGNAL_COIN) {
        for (int c = 0; c < 2; ++c)
            hsl_observe_current_scope(side.coin_hsl[c], context, 0, c,
                initial_balance, 0.0f, 0);
    } else hsl_observe_current_scope(side.hsl, context, 0, -1,
        preceding_close ? 1000.0f : 200.0f, 0.0f, 0);
    side.entry_qty[1] = 1.0f;
    side.entry_tick[1] = 80;
    JointPortfolioAccount account = init_joint_portfolio_account(initial_balance);
#if V9_TM_PROBE
    TrailingMartingaleMulticoinFillState fills = init_trailing_martingale_multicoin_fill_state();
    TrailingMartingaleMulticoinSideConfig config = {};
    config.coin_hsl_mode = side.hsl.signal_mode == HSL_SIGNAL_COIN;
    float hsl_equity = initial_balance;
    bool filled = process_tm_multicoin_side_fills(side, config, account, fills,
        bars, ticks, ticks, ticks, ticks, touch_min_qty_relation, settings, overrides, output,
        0, 0, 2, false, true, false, false, 1.0f, false, 0.0f, 0.0f,
        false, hsl_equity);
#else
    EmaMulticoinFillState fills = init_ema_multicoin_fill_state();
    bool filled = process_ema_multicoin_side_fills(side, account, fills,
        bars, ticks, settings, overrides, output, 0, 0, 2, false, true,
        false, 0.0f, false, 400.0f);
#endif
    output[0] = side.psize[1];
    output[1] = fills.fill_count;
    output[2] = float(filled);
    output[3] = account.balance;
}
"""

_TERMINAL_PROBE = r"""
kernel void v9_terminal_mark_probe(
    constant float* params, constant float* settings, constant float* bars,
    device float* output, uint b [[thread_position_in_grid]]
) {
    if (b != 0) return;
    HslState aggregate = load_hsl(params, 0, 0);
    HslState coins[1];
    coins[0] = load_hsl(params, 0, 0);
    bind_hsl(aggregate, nullptr, nullptr, 0, 0, 0, 0, true, true);
    bind_hsl(coins[0], nullptr, nullptr, 0, 0, 0, 0, true, true);
    HslReplayContext context = hsl_replay_context(bars, settings, 1, COIN_COLS, true);
    attach_hsl_replay_side(context, aggregate, coins, 0, false);
    hsl_before_fill(coins[0], 1000.0f, 100.0f, 1);
    hsl_after_fill(coins[0], 1000.0f, 0.0f, 20.0f, 100.0f, 100.0f, 1);
    thread HslState& scope = aggregate.signal_mode == HSL_SIGNAL_COIN ? coins[0] : aggregate;
    output[0] = float(scope.hsl.action);
    const float balance = 1000.0f;
    const float hsl_unrealized = -1980.0f;
    const float hsl_equity = balance + hsl_unrealized;
    const float liquidation_floor = 50.0f;
    const bool post_fill_balance_depleted = false;
    const bool any_fill = true;
    // V9_TERMINAL_PHASE_GATE
    if (hsl_mark_terminal_observation_eligible(balance, hsl_equity, 50.0f,
            hsl_report_at_fill_boundary)) {
        hsl_completed_marks(context, 0);
        hsl_observe_current_scope(scope, context, 0,
            aggregate.signal_mode == HSL_SIGNAL_COIN ? 0 : -1, balance, 0.0f, 2);
    }
    output[1] = float(scope.hsl.action);
    output[2] = float(scope.hsl.last_observed);
    output[3] = scope.decay.drawdown;
    output[4] = scope.decay.ratio;
}
"""

_REALIZED_CLOCK_PROBE = r"""
kernel void v9_aggregated_realized_clock_probe(device float* output,
    uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    const float interval_ms = 300000.0f;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    configure_joint_realized_memory(account, 3.0f, 3.0f, false, true,
        multicoin_interval_minutes(interval_ms));
    record_joint_portfolio_fill(account, -10.0f, true);
    int k = 0;
    // V9_REALIZED_COMPLETED_CLOCK
    output[0] = account.loss_gate_memory.drawdown;
    float day_count=0, count=0, entries=0, longs=0, peak=0, peak_k=0, recovery=0;
    record_realized_net(-10.0f, account, day_count, count, entries, longs,
        peak, peak_k, recovery, 1.0f, false, true);
    output[1] = account.loss_gate_memory.drawdown;
    k = 1;
    // V9_REALIZED_COMPLETED_CLOCK
    output[2] = account.loss_gate_memory.drawdown;
    output[3] = float(account.realized_decay_valid);
}
"""


def _native_source(root, strategy):
    body = ("mps_trailing_martingale_multicoin.metal" if strategy == "tm"
            else "mps_ema_anchor_multicoin_long.metal")
    source = (root / body).read_text()
    source = source.replace("using namespace metal;", "using namespace metal;\n"
                            + (root / "mps_hsl_decay.metal").read_text(), 1)
    replacements = {
        "// PASSIVBOT_HSL_COMMON": ["mps_hsl.metal", "mps_hsl_common.metal"],
        "// PASSIVBOT_MULTICOIN_COMMON": ["mps_decimal_rounding.metal", "mps_multicoin_common.metal"],
        "// PASSIVBOT_ADAPTIVE_TIMING": ["mps_adaptive_timing.metal"],
        "// PASSIVBOT_UNSTUCK_EMA_COMMON": ["mps_unstuck_ema_common.metal"],
        "// PASSIVBOT_BTC_RISK_COMMON": ["mps_btc_risk_common.metal"],
        "// PASSIVBOT_EQUITY_BALANCE_DIFF_COMMON": ["mps_equity_balance_diff_common.metal"],
        "// PASSIVBOT_ENTRY_INTERVAL_COMMON": ["mps_entry_interval_common.metal"],
    }
    for marker, paths in replacements.items():
        assert source.count(marker) == 1
        source = source.replace(marker, "\n".join((root / p).read_text() for p in paths), 1)
    return ("#define PASSIVBOT_HSL_LOOKBACK 1\n"
            + (root / "mps_organillo.metal").read_text() + source)


_REALIZED_GATE_PROBE = r"""
kernel void shared_gate_event_probe(
    constant float* settings, constant float* events, device float* output,
    uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    HslDecayPolicy policy;
    if (!hsl_decay_policy(settings[0], 1.0f, 0.1f, 0, policy)) return;
    JointPortfolioAccount account = init_joint_portfolio_account(settings[2]);
    const bool loss_gate_enabled = settings[3] > 0.5f;
    configure_joint_realized_memory(account, settings[0], settings[0],
        false, loss_gate_enabled, int(settings[4]));
    for (int event = 0; event < 8; ++event) {
        advance_joint_realized_memory(account, int(events[event * 2]));
        record_joint_portfolio_fill(account, events[event * 2 + 1], event % 2 == 0);
        const float before_query = account.loss_gate_memory.drawdown;
        float allowance = 0.0f;
        bool valid = joint_realized_loss_allowance(account, settings[1], allowance);
        output[event * 4] = account.loss_gate_memory.drawdown;
        output[event * 4 + 1] = account.balance;
        output[event * 4 + 2] = allowance;
        output[event * 4 + 3] = float(valid && account.realized_decay_valid
            && before_query == account.loss_gate_memory.drawdown);
    }
}

kernel void shared_gate_invalid_probe(
    constant float* inputs, device float* output, uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    JointPortfolioAccount account = init_joint_portfolio_account(inputs[0]);
    account.loss_gate_memory.drawdown = inputs[2];
    float allowance = NAN;
    output[0] = float(joint_realized_loss_allowance(account, inputs[1], allowance));
    output[1] = float(account.realized_decay_valid);
}
"""


@pytest.fixture(scope="module", params=["ema", "tm"])
def shared_gate_program(request):
    import re

    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual supported Metal/CUDA execution required")
    from optimization.gpu.runtime import compile_shader, gpu_device

    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    source = _native_source(root, request.param)
    assert source.count("configure_joint_realized_memory(account, params[po +") == 2
    probe = _REALIZED_GATE_PROBE
    # Gate-only production must keep the fill producer when unstuck is compiled out.
    source = "#define PASSIVBOT_UNSTUCK_ENABLED 0\n" + source + probe
    return torch, compile_shader(source, mps_coin_capacity=2), gpu_device(torch)


@pytest.mark.parametrize("span,fraction,interval", [(3.0, 0.1, 1), (3.0, 0.0, 1), (8.0, 0.99999999, 5)])
def test_gate_only_ordered_fills_share_exact_decaying_memory(shared_gate_program, span, fraction, interval):
    from v9_risk_reference import add, allowance, coefficients, f32, fma, sub

    torch, library, device = shared_gate_program
    events = [(0, -10.0), (0, 20.0), (0, -5.0), (0, -0.25),
              (0, 0.125), (5, 0.0), (5, -2.0), (6, 3.0)]
    settings = torch.tensor([span, fraction, 1000.0, 1.0, interval], dtype=torch.float32, device=device)
    packed = torch.tensor(events, dtype=torch.float32, device=device)
    output = torch.full((8, 4), float("nan"), dtype=torch.float32, device=device)
    library.shared_gate_event_probe(settings, packed, output, threads=1)
    _, beta = coefficients(span)
    expected, drawdown, balance, previous_minute = [], 0.0, 1000.0, 0
    for bar, net in events:
        minute = bar * interval
        for _ in range(previous_minute, minute):
            drawdown = fma(beta, drawdown, 0.0)
        previous_minute = minute
        drawdown = max(0.0, sub(drawdown, f32(net)))
        balance = add(balance, f32(net))  # Exact dyadic fixture amounts.
        expected.append([drawdown, balance, allowance(balance, fraction, drawdown), 1.0])
    assert output.cpu().tolist() == expected


@pytest.mark.parametrize("inputs", [(2e38, 0.1, 2e38), (1000.0, 0.1, -1.0)])
def test_shared_gate_invalid_memory_is_not_a_neutral_allowance(shared_gate_program, inputs):
    torch, library, device = shared_gate_program
    values = torch.tensor(inputs, dtype=torch.float32, device=device)
    output = torch.full((2,), float("nan"), dtype=torch.float32, device=device)
    library.shared_gate_invalid_probe(values, output, threads=1)
    assert output.cpu().tolist() == [0.0, 0.0]


@pytest.fixture(scope="module")
def adapter_program():
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual supported Metal/CUDA execution required")
    from optimization.gpu.runtime import compile_shader, gpu_device

    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    source = ("#include <metal_stdlib>\n"
              "using namespace metal;\n#define PASSIVBOT_HSL_LOOKBACK 1\n")
    source += "\n".join((root / name).read_text() for name in (
        "mps_hsl_decay.metal", "mps_hsl.metal", "mps_hsl_common.metal"))
    return torch, compile_shader(source + _PROBE), gpu_device(torch)


@pytest.fixture(scope="module", params=["ema", "tm"])
def native_entry_program(request):
    import re

    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual supported Metal/CUDA execution required")
    from optimization.gpu.runtime import compile_shader, gpu_device

    root = Path(__file__).resolve().parents[2] / "passivbot-rust/src/gpu"
    body = (root / ("mps_trailing_martingale_multicoin.metal" if request.param == "tm"
                    else "mps_ema_anchor_multicoin_long.metal")).read_text()
    # Exercise the production phase gate, rather than a copied test condition.
    gates = re.findall(r"const bool hsl_report_at_fill_boundary =[^;]+;", body)
    assert len(gates) == 2  # One-side and joint-account entry points.
    terminal_probe = "\n".join(
        _TERMINAL_PROBE.replace("// V9_TERMINAL_PHASE_GATE", gate)
        .replace("v9_terminal_mark_probe", f"v9_terminal_mark_probe_{index}")
        for index, gate in enumerate(gates)
    )
    clock_calls = re.findall(r"advance_joint_realized_memory\(account, k \+ 1\);", body)
    assert len(clock_calls) == 2
    clock_probe = _REALIZED_CLOCK_PROBE.replace("// V9_REALIZED_COMPLETED_CLOCK", clock_calls[0])
    source = (f"#define V9_TM_PROBE {int(request.param == 'tm')}\n"
              "#define PASSIVBOT_UNSTUCK_ENABLED 1\n")
    source += _native_source(root, request.param) + _ENTRY_PROBE + terminal_probe + clock_probe
    return torch, compile_shader(source, mps_coin_capacity=2), gpu_device(torch), request.param


def test_other_coin_fill_reprices_current_budget_before_pending_entry(adapter_program):
    torch, library, device = adapter_program
    params = torch.tensor([1, .08, 1, 0, 3, 2, 0], dtype=torch.float32, device=device)
    settings = torch.zeros((2, 8), dtype=torch.float32, device=device)
    settings[:, 4] = 1
    output = torch.full((12,), float("nan"), dtype=torch.float32, device=device)
    library.v9_coin_budget_probe(params, settings, output, threads=1)
    np.testing.assert_allclose(output.cpu().numpy(),
                               [20, .04, 20, .1, .1, 3, 80, 1, 0, 1, -1, 0],
                               rtol=2e-7, atol=0)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_entry_red_fill_reports_the_following_interval(adapter_program, mode):
    torch, library, device = adapter_program
    params = torch.tensor([1, 1e-8, 1, 0, 3, mode, 0], dtype=torch.float32, device=device)
    settings = torch.zeros((1, 8), dtype=torch.float32, device=device)
    settings[:, 4] = 1
    output = torch.full((3,), float("nan"), dtype=torch.float32, device=device)
    library.v9_entry_red_time_probe(params, settings, output, threads=1)
    np.testing.assert_array_equal(output.cpu().numpy(), [1, 1, 1])


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("preceding_close", [False, True])
def test_pending_native_entry_checks_current_scope_permission(native_entry_program, mode, preceding_close):
    torch, library, device, strategy = native_entry_program
    params = torch.tensor([1, .08, 1, 0, 3, mode, 0], dtype=torch.float32, device=device)
    settings = torch.tensor([[1, 1, 1, 0, 1, 0, 0, 1, 0, 100, 1, 0, 0]] * 2,
                            dtype=torch.float32, device=device)
    bars = torch.tensor([[120, 20, 40, 1], [120, 60, 80, 1]], dtype=torch.float32, device=device)
    ticks = torch.tensor([[120, 60]] * 2, dtype=torch.int32, device=device)
    overrides = torch.full((2, 75 if strategy == "tm" else 34), float("nan"),
                            dtype=torch.float32, device=device)
    output = torch.full((4,), float("nan"), dtype=torch.float32, device=device)
    relation = torch.zeros((2, 2), dtype=torch.int8 if device == "cuda" else torch.int32, device=device)
    library.v9_pending_entry_probe(params, settings, bars, ticks, relation, overrides,
                                   int(preceding_close), output, threads=1)
    np.testing.assert_array_equal(output.cpu().numpy(),
                                  [1, int(preceding_close), int(preceding_close), 400])


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("topology", [0, 1], ids=["first", "second"])
def test_terminal_completed_mark_survives_a_start_fill(native_entry_program, mode, topology):
    torch, library, device, _ = native_entry_program
    params = torch.tensor([1, .08, 1, 0, 3, mode, 0], dtype=torch.float32, device=device)
    settings = torch.zeros((1, 15), dtype=torch.float32, device=device)
    settings[:, 4] = 1
    bars = torch.tensor([[2, 1, 1, 1]], dtype=torch.float32, device=device)
    output = torch.full((5,), float("nan"), dtype=torch.float32, device=device)
    getattr(library, f"v9_terminal_mark_probe_{topology}")(
        params, settings, bars, output, threads=1)
    np.testing.assert_array_equal(output.cpu().numpy(), [0, 3, 2, 1980, 1])


def test_realized_memory_uses_five_integer_minutes_per_five_minute_bar(native_entry_program):
    from fractions import Fraction

    torch, library, device, _ = native_entry_program
    output = torch.full((4,), float("nan"), dtype=torch.float32, device=device)
    library.v9_aggregated_realized_clock_probe(output, threads=1)
    # Span3 gives exact beta1/2: five authored minute recurrences, then a
    # same-clock loss fill and five more. These rational values are exactly f32.
    first = Fraction(10, 2**5)
    second = first + 10
    np.testing.assert_array_equal(output.cpu().numpy(),
        [float(first), float(second), float(second / 2**5), 1])
