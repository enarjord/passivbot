"""Exercise the checked-in Metal episode boundary code on Apple MPS."""

from functools import lru_cache
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
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
    for marker, filename in (
        ("HSL", "mps_hsl_common.metal"),
        ("BTC_RISK", "mps_btc_risk_common.metal"),
        ("EQUITY_BALANCE_DIFF", "mps_equity_balance_diff_common.metal"),
        ("ENTRY_INTERVAL", "mps_entry_interval_common.metal"),
        ("MULTICOIN", "mps_multicoin_common.metal"),
    ):
        source = source.replace(f"// PASSIVBOT_{marker}_COMMON", (_GPU / filename).read_text())
    return source


def _params(mode=2, red_threshold=0.2):
    return torch.tensor(
        [1, red_threshold, 3, 60, 0.8, 0, 0.5, 0.8, 0, mode, 1],
        dtype=torch.float32,
        device="mps",
    )


_SHARED_PROBE = r"""
kernel void episode_boundary_probe(
    constant float* params,
    device float* output,
    constant int& scope_held,
    constant int& opposite_held,
    constant int& prior_red,
    uint b [[thread_position_in_grid]]
) {
    HslState h = load_hsl(params, 0, 0);
    h.initialized = true;
    h.peak_strategy_pnl = 0.0f;
    h.no_restart_peak_strategy_equity = 1234.0f;
    h.red_latched = prior_red != 0;
    h.tier = prior_red != 0 ? 3 : 0;
    HslState opposite = h;
    bool reset = finish_hsl_scoped_episode_at_flat(
        h, &opposite, scope_held != 0, opposite_held != 0,
        900.0f, 1000.0f, -100.0f, -100.0f, 10.0f, 60000.0f
    );
    output[0] = reset;
    output[1] = h.initialized;
    output[2] = h.coin_realized_baseline;
    output[3] = h.no_restart_peak_strategy_equity;
    output[4] = h.red_latched;
    output[5] = h.flat_confirmations;
    output[6] = h.drawdown_ema;
    output[7] = opposite.coin_realized_baseline;
    // An ordinary bar read must not refinalize an already recorded stop.
    update_hsl(h, 900.0f, 1000.0f, -100.0f, 0.0f,
               false, false, 10.0f, 60000.0f);
    output[8] = h.flat_confirmations;
    output[9] = h.drawdown_ema;
    // The next entry is in the same bar, after the actual flatten boundary.
    HslSignal signal;
    signal.drawdown_raw = -1.0f;
    derive_hsl_signal(h, 900.0f, 1000.0f, -100.0f, -50.0f, signal);
    output[10] = signal.drawdown_raw;
    output[11] = h.halted;
    output[12] = h.cooldown_until_k;
    output[13] = h.triggers;
    // A resting entry after this boundary cannot erase the finalized stop.
    update_hsl(h, 900.0f, 1000.0f, -100.0f, -50.0f,
               true, false, 10.0f, 60000.0f);
    output[14] = h.pending_stop_k;
    output[15] = h.triggers;
}
"""


@lru_cache(maxsize=None)
def _shared_library(name):
    return torch.mps.compile_shader(_source(name) + _SHARED_PROBE + _SAME_MINUTE_PROBE + _EMA_REPLACEMENT_PROBE)


def _run_shared(name, mode=2, threshold=0.2, scope_held=False, opposite_held=False, prior_red=False):
    output = torch.zeros(16, dtype=torch.float32, device="mps")
    _shared_library(name).episode_boundary_probe(
        _params(mode, threshold), output, int(scope_held), int(opposite_held), int(prior_red),
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()
    return output.cpu().tolist()


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["unified", "pside", "coin"])
def test_ordinary_episode_reset_preserves_persistent_peak_and_same_bar_reentry(name, mode):
    values = _run_shared(name, mode)
    assert values[:4] == [1, 1, -100, 1234]
    assert values[4:7] == [0, 0, 0]
    assert values[7] == (-100 if mode == 0 else 0)
    assert values[10] == pytest.approx(50 / 900, abs=1e-6)


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("prior_red", [False, True], ids=["first-red-on-close", "already-red"])
def test_closing_loss_can_first_trigger_red_without_double_sampling(name, prior_red):
    values = _run_shared(name, threshold=0.2 if prior_red else 0.05, prior_red=prior_red)
    assert values[0] == 0
    assert values[1] == 1
    assert values[2] == -100
    assert values[4:6] == [1, 2]
    assert values[6] == pytest.approx(0.5 * 100 / 900, abs=1e-6)
    assert values[8] == 2
    assert values[9] == values[6]
    assert values[11:] == [1, 70, 1, 10, 1]


@pytest.mark.parametrize("mode,scope_held,opposite_held,reset", [
    (0, False, True, False),
    (0, True, False, False),
    (1, True, False, False),
    (1, False, True, True),
    (2, False, True, True),
])
def test_episode_resets_only_when_configured_scope_is_flat(mode, scope_held, opposite_held, reset):
    values = _run_shared(_KERNELS[0], mode, scope_held=scope_held, opposite_held=opposite_held)
    assert values[0] == int(reset)


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
@pytest.mark.parametrize("hsl_disabled", [False, True])
def test_trailing_directional_specializations_compile(direction, hsl_disabled):
    prefix = f"#define PASSIVBOT_TRAILING_{direction}_ONLY\n"
    if hsl_disabled:
        prefix += "#define PASSIVBOT_TRAILING_HSL_DISABLED\n"
    torch.mps.compile_shader(prefix + _source(_KERNELS[1]))


_MULTICOIN_PROBE = r"""
kernel void multicoin_close_episode_probe(
    constant float* params,
    device float* output,
    constant int& other_coin_held,
    constant int& opposite_held,
    constant int& short_side_raw,
    uint b [[thread_position_in_grid]]
) {
    SIDE_STATE side;
    side.hsl = load_hsl(params, 0, 0);
    side.hsl.initialized = true;
    side.hsl.peak_strategy_pnl = 0.0f;
    side.psize[0] = 1.0f;
    side.psize[1] = float(other_coin_held);
    side.coin_realized_pnl[0] = 0.0f;
    side.coin_realized_pnl[1] = 0.0f;
    side.coin_hsl[0] = side.hsl;
    side.coin_hsl[1] = side.hsl;
    HslState opposite = side.hsl;
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    FILL_STATE fills = INIT_FILLS();
    float equity = 901.0f;
    RECORD_FILL(side, account, fills, output, 0, 2, 0, 10,
        -99.0f, -100.0f, 1.0f, 1000.0f, 901.0f, 1.0f,
        short_side_raw != 0, false, false, equity, &opposite, opposite_held != 0);
    thread HslState& h = side.hsl.signal_mode == HSL_SIGNAL_COIN
        ? side.coin_hsl[0] : side.hsl;
    output[0] = h.coin_realized_baseline == -100.0f;
    output[1] = h.coin_realized_baseline;
    output[2] = h.red_latched;
    output[3] = account.balance;
    output[4] = side.coin_realized_pnl[0];
    output[5] = opposite.coin_realized_baseline;
}
"""


@lru_cache(maxsize=None)
def _multicoin_library(strategy):
    if strategy == "ema":
        name, side, fills, init, record = (
            _KERNELS[2], "EmaMulticoinSideState", "EmaMulticoinFillState",
            "init_ema_multicoin_fill_state", "record_ema_multicoin_close_fill",
        )
    else:
        name, side, fills, init, record = (
            _KERNELS[3], "TrailingMartingaleMulticoinSideState",
            "TrailingMartingaleMulticoinFillState", "init_trailing_martingale_multicoin_fill_state",
            "record_tm_multicoin_close_fill",
        )
    probe = _MULTICOIN_PROBE
    for key, value in (("SIDE_STATE", side), ("FILL_STATE", fills), ("INIT_FILLS", init), ("RECORD_FILL", record)):
        probe = probe.replace(key, value)
    return torch.mps.compile_shader(_source(name) + probe)


@pytest.mark.parametrize("strategy", ["ema", "trailing"])
@pytest.mark.parametrize("short_side", [False, True], ids=["long", "short"])
@pytest.mark.parametrize("mode,other_coin_held,opposite_held,reset", [
    (2, True, True, True),
    (1, True, False, False),
    (1, False, True, True),
    (0, False, True, False),
    (0, False, False, True),
])
def test_multicoin_closing_fill_resets_scoped_state_after_accounting_for_fee(
    strategy, short_side, mode, other_coin_held, opposite_held, reset
):
    output = torch.zeros(6, dtype=torch.float32, device="mps")
    _multicoin_library(strategy).multicoin_close_episode_probe(
        _params(mode), output, int(other_coin_held), int(opposite_held), int(short_side),
        threads=(1, 1, 1),
    )
    torch.mps.synchronize()
    values = output.cpu().tolist()
    assert values[0] == int(reset)
    assert values[1] == (-100 if reset else 0)
    assert values[2:5] == [0, 900, -100]
    assert values[5] == (-100 if mode == 0 and reset else 0)


_SAME_MINUTE_PROBE = r"""
kernel void same_minute_boundary_probe(
    constant float* params, device float* output,
    uint b [[thread_position_in_grid]]
) {
    HslState h = load_hsl(params, 0, 0);
    h.initialized = true;
    h.peak_strategy_pnl = 0.0f;
    // Prime the minute's normal sample before two separate position episodes.
    update_hsl(h, 1000.0f, 1000.0f, 0.0f, 0.0f, true, false, 10.0f, 60000.0f);
    output[0] = finish_hsl_episode_at_flat(h, 990.0f, 1000.0f, -10.0f, 10.0f, 60000.0f);
    output[1] = h.drawdown_ema;
    // Another entry and close in this minute lose a further 100 including fees.
    output[2] = finish_hsl_episode_at_flat(h, 890.0f, 1000.0f, -110.0f, 10.0f, 60000.0f);
    output[3] = h.sampled_drawdown_raw;
    output[4] = h.drawdown_ema;
    output[5] = h.halted;
    output[6] = h.pending_stop_k;
    output[7] = h.cooldown_until_k;
    output[8] = h.triggers;
    output[9] = h.no_restart_latched;
    update_hsl(h, 890.0f, 1000.0f, -110.0f, -50.0f, true, false, 10.0f, 60000.0f);
    output[10] = h.drawdown_ema;
    output[11] = h.triggers;
}
"""


@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["unified", "pside", "coin"])
@pytest.mark.parametrize("terminal", [False, True])
def test_second_same_minute_closing_loss_finalizes_red_before_reentry(name, mode, terminal):
    params = _params(mode, 0.02)
    if terminal:
        params[4] = 0.04
        params[5] = 1
    output = torch.zeros(12, dtype=torch.float32, device="mps")
    _shared_library(name).same_minute_boundary_probe(params, output, threads=(1, 1, 1))
    torch.mps.synchronize()
    values = output.cpu().tolist()
    raw = 100 / (890 if mode == 2 else 990)
    assert values[:3] == [1, 0, 0]
    assert values[3:5] == pytest.approx([raw, 0.5 * raw], abs=1e-6)
    assert values[5:10] == [1, 10, -1 if terminal else 70, 1, int(terminal)]
    assert values[10] == values[4]
    assert values[11] == 1


_EMA_REPLACEMENT_PROBE = r"""
kernel void boundary_ema_replacement_probe(constant float* params, device float* output,
    uint b [[thread_position_in_grid]]) {
    HslState h = load_hsl(params, 0, 0);
    h.initialized = true;
    h.drawdown_ema = 0.2f;
    h.last_sample_k = 8.0f;
    HslSignal signal;
    signal.strategy_equity = 900.0f;
    signal.peak_strategy_equity = 1000.0f;
    signal.drawdown_raw = 0.1f;
    update_hsl_from_signal(h, signal, 0.0f, true, false, 10.0f, 60000.0f);
    output[0] = h.drawdown_ema;
    signal.drawdown_raw = 0.3f;
    update_hsl_from_signal(h, signal, 0.0f, true, false, 10.0f, 60000.0f, true);
    output[1] = h.drawdown_ema;
    signal.drawdown_raw = 0.9f;
    update_hsl_from_signal(h, signal, 0.0f, true, false, 10.0f, 60000.0f);
    output[2] = h.drawdown_ema;
    output[3] = h.sampled_drawdown_raw;
}
"""


@pytest.mark.parametrize("name", _KERNELS)
def test_distinct_boundary_replaces_cached_minute_without_compounding_ema(name):
    output = torch.zeros(4, dtype=torch.float32, device="mps")
    _shared_library(name).boundary_ema_replacement_probe(_params(red_threshold=0.9), output, threads=(1, 1, 1))
    torch.mps.synchronize()
    assert output.cpu().tolist() == pytest.approx([0.125, 0.275, 0.275, 0.3], abs=1e-6)
