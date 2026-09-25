from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

import passivbot_monitor as monitor
from config.strategy_spec import get_strategy_defaults


@pytest.fixture
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("strategy defaults require the compiled Rust strategy registry")


def test_trailing_snapshot_reuses_settings_and_observes_next_snapshot_changes(
    monkeypatch, require_real_passivbot_rust_module
):
    # Inspect the complete diagnostic inputs; native result parity is covered by
    # the real-extension trailing diagnostic and monitor tests.
    monkeypatch.setattr(monitor, 'build_trailing_entry_diagnostic', lambda inputs: deepcopy(inputs))
    monkeypatch.setattr(monitor, 'build_trailing_martingale_close_diagnostic', lambda inputs: deepcopy(inputs))
    symbols = ['A/USDT:USDT', 'B/USDT:USDT']
    calls = Counter()
    defaults = get_strategy_defaults('trailing_martingale')
    overrides = {('long', symbols[0]): .017, ('short', symbols[1]): .028}
    def params(side, symbol):
        calls[side, symbol] += 1
        result = deepcopy(defaults[side])
        result['entry']['initial_qty_pct'] = overrides.get((side, symbol), .011)
        return result
    def bp(side, key, symbol=None):
        return 'bounded' if key == 'risk_we_excess_allowance_mode' else .2
    bot = SimpleNamespace(
        config={'live': {'strategy_kind': 'trailing_martingale'}},
        positions={s: {'long': {'size': 1., 'price': 100.}, 'short': {'size': -.5, 'price': 100.}} for s in symbols},
        _strategy_params_to_rust_dict=params,
        _orchestrator_exchange_params=lambda s: {'qty_step': .01},
        _bot_params_to_rust_dict=lambda side, symbol: {'n_positions': 2},
        bp=bp, bot_value=lambda *args: 1.,
        **{k: {s: .01 for s in symbols} for k in ['qty_steps', 'price_steps', 'min_qtys', 'min_costs', 'c_mults']},
    )
    market = {s: {'last_price': 101., 'ema_bands': {'long': {'lower': 100., 'upper': 102.}}} for s in symbols}
    def build():
        return monitor._build_monitor_trailing_section(bot, balance_raw=1000., market=market)
    cached = build()
    assert calls == Counter({(side, s): 1 for s in symbols for side in ('long', 'short')})
    original = monitor._monitor_strategy_params
    with monkeypatch.context() as m:
        m.setattr(monitor, '_monitor_strategy_params', lambda bot, side, symbol, cache=None: original(bot, side, symbol))
        assert build() == cached
    assert max(calls.values()) > 10  # the comparison exercised independent resolution
    calls.clear()
    overrides['long', symbols[0]] = .043
    fresh = build()
    assert fresh[symbols[0]]['long']['entry']['entry_initial_qty_pct'] == .043
    assert fresh[symbols[1]] == cached[symbols[1]]
    assert calls == Counter({(side, s): 1 for s in symbols for side in ('long', 'short')})


def test_failed_strategy_resolution_is_not_cached():
    calls = []
    def getter(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('invalid strategy')
        return {'entry': {'initial_qty_pct': .02}}
    bot = SimpleNamespace(_strategy_params_to_rust_dict=getter)
    cache = {}
    with pytest.raises(ValueError):
        monitor._monitor_strategy_value(bot, 'long', 'entry_initial_qty_pct', 'A', strategy_cache=cache)
    assert cache == {}
    assert monitor._monitor_strategy_value(bot, 'long', 'entry_initial_qty_pct', 'A', strategy_cache=cache) == .02
