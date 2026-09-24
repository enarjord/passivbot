"""Device integration for compatible suite batching and scenario result routing."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config.schema import get_template_config
from optimization.backends.gpu_backend import _evaluate_gpu_suite_proxies
from optimization.gpu.service import MpsMulticoinProxy
from tools.gpu_proxy_benchmark import _synthetic_hlcvs


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Apple MPS and NVIDIA CUDA unavailable",
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_suite_batches_preserve_metrics_defaults_and_dispatch_bounds(side):
    count = 9001
    values, timestamps = _synthetic_hlcvs(count, 3, 7)
    coins = ["BTC", "ETH", "SOL"]
    markets = {
        coin: dict(
            qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=0.0,
            c_mult=1.0, maker=0.0002, taker=0.0006, exchange="bybit",
            first_valid_index=0, last_valid_index=count - 1, warmup_minutes=60,
        )
        for coin in coins
    }
    markets["__meta__"] = dict(
        requested_start_ts=int(timestamps[60]), warmup_minutes_requested=60,
    )
    config = get_template_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    config["live"]["max_warmup_minutes"] = 60
    config["backtest"]["coins"] = {"bybit": coins}
    config["backtest"]["exchanges"] = ["bybit"]
    for pside in ("long", "short"):
        config["bot"][pside]["risk"]["n_positions"] = 1 if pside == side else 0
        config["bot"][pside]["risk"]["total_wallet_exposure_limit"] = 1.0 if pside == side else 0.0
        config["bot"][pside]["hsl"]["enabled"] = False
        config["bot"][pside]["unstuck"]["enabled"] = False
    strategy = config["bot"][side]["strategy"]["trailing_martingale"]
    strategy["entry"].update(ema_span_0=17.25, ema_span_1=53.5, initial_ema_dist=0.001)
    strategy["close"].update(threshold_base_pct=0.002, retracement_base_pct=0.001)
    needed = {"adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day"}
    cache = {}
    scenarios = []
    interrupt_check = lambda: None
    for n_positions in (1, 2, 3):
        scenario = copy.deepcopy(config)
        scenario["bot"][side]["risk"]["n_positions"] = n_positions
        # Different defaults must survive materialization even when not supplied
        # as explicit suite overrides or candidate values.
        scenario["bot"][side]["risk"]["total_wallet_exposure_limit"] = n_positions / 3
        proxy = MpsMulticoinProxy(
            config=scenario, hlcvs=values, mss=markets,
            btc=np.full(count, 50000.0), timestamps=timestamps,
            exchange="bybit", batch_size=64, max_dispatch_candidate_bars=1600000,
            needed_metrics=needed, prepared_data_cache=cache, interrupt_check=interrupt_check,
        )
        proxy.profile_enabled = True
        scenarios.append((SimpleNamespace(label=str(n_positions)), [("bybit", proxy)], {}))
    assert len({item[1][0][1].suite_batch_key() for item in scenarios}) == 1
    assert scenarios[0][1][0][1].suite_batch_key() is not None
    candidates = [{f"{side}_entry_initial_qty_pct": 0.1 + i / 1000} for i in range(35)]
    original = copy.deepcopy(candidates)

    class Suite:
        @staticmethod
        def score_scenario_results(results):
            metrics = [result.metrics["stats"] for result in results]
            # Keep each scenario's reductions observable, including its order.
            objectives = tuple(-row["adg_strategy_eq"]["mean"] for row in metrics)
            return dict(
                objectives=objectives, unpenalized_objectives=objectives,
                constraint_violation=0.0, suite_metrics=metrics,
            )

    separate = _evaluate_gpu_suite_proxies(
        Suite(), scenarios, candidates,
        batch_compatible_scenarios=False,
    )
    batched = _evaluate_gpu_suite_proxies(
        Suite(), scenarios, candidates,
        batch_compatible_scenarios=True,
    )
    np.testing.assert_equal(batched, separate)
    assert candidates == original
    profiles = [item[1][0][1].last_profile for item in scenarios]
    assert profiles[0]["actual_dispatch_batch_sizes"] == [64, 41]
    assert all(not profile for profile in profiles[1:])
    assert profiles[0]["dispatch_count"] > 2  # temporal chunks remain bounded
