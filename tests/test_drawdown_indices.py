"""Full Rust replay exports and scores minute-sampled drawdown indices."""
import math

import numpy as np
import pytest


def test_real_replay_drawdown_indices_match_saved_full_resolution_curve():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension
    assert not getattr(pbr, '__is_stub__', False)
    verify_loaded_runtime_extension()
    from backtest import build_backtest_payload, execute_backtest
    from pathlib import Path
    from config_utils import load_config
    from config.scoring import normalize_scoring_entries

    config = load_config(str(Path(__file__).resolve().parents[1] /
                             'configs/examples/default_trailing_martingale_long.json'),
                         verbose=False)
    config['backtest'].update(exchanges=['binance'], coins={'binance': ['BTC']},
                              start_date='2021-01-01', end_date='2021-02-05')
    config['bot']['portfolio']['warmup'].update(span_multiplier=0., max_minutes=None)
    config['bot']['long']['enabled'] = True
    config['bot']['short']['enabled'] = False
    config['bot']['long']['universe']['minimum_coin_age_days'] = 0
    n = 50000
    timestamps = 1609459200000 + np.arange(n, dtype=np.int64) * 60_000
    price = 100 + 10 * np.sin(np.arange(n) / 75)
    candles = np.stack([price + .5, price - .5, price, np.ones(n)], axis=1)[:, None, :]
    mss = {'BTC': dict(qty_step=.001, price_step=.1, min_qty=0., min_cost=0.,
                      c_mult=1., maker=.0002, taker=.0005, exchange='binance',
                      first_valid_index=0, last_valid_index=n-1),
           '__meta__': dict(requested_start_ts=int(timestamps[0]),
                            requested_start_date='2021-01-01', warmup_minutes_requested=0,
                            warmup_minutes_provided=0)}
    payload = build_backtest_payload(candles, mss, config, 'binance',
                                     np.full(n, 20000.), timestamps)
    _, equities, analysis = execute_backtest(payload, config)
    strategy_equity = equities[:, 3]
    assert len(strategy_equity) > 1000
    assert analysis['fills_count'] > 0
    peaks = np.maximum.accumulate(strategy_equity)
    drawdowns = (peaks - strategy_equity) / np.maximum(np.abs(peaks), 1e-12)
    assert analysis['pain_index_strategy_eq'] == pytest.approx(np.mean(drawdowns), abs=1e-12)
    assert analysis['ulcer_index_strategy_eq'] == pytest.approx(math.sqrt(np.mean(drawdowns**2)), abs=1e-12)
    assert analysis['ulcer_index_strategy_eq'] >= analysis['pain_index_strategy_eq'] > 0
    assert not any(key.startswith('strategy_eq_underwater_pct_') for key in analysis)
    specs, _ = normalize_scoring_entries(['pain_index_strategy_eq', 'ulcer_index_strategy_eq'])
    assert [spec.goal for spec in specs] == ['min', 'min']

    payload = build_backtest_payload(candles, mss, config, 'binance',
                                     np.full(n, 20000.), timestamps, metrics_only=True)
    _, omitted_equities, metrics_only = execute_backtest(payload, config)
    assert omitted_equities is None
    for metric in ['pain_index_strategy_eq', 'ulcer_index_strategy_eq']:
        assert metrics_only[metric] == analysis[metric]
