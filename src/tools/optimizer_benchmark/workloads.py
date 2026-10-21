"""Generate once, serialize, then replay the same effective candidates everywhere."""

from copy import deepcopy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import random

from .catalogue import declared_gpu_support, select_cases

VERSION = 2
CORE_METRICS = ["adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day", "backtest_completion_ratio"]


def harness_digest():
    result = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(memoryview(array).cast("B"))
    return result.hexdigest()


def set_path(config, path, value):
    node = config
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def get_path(config, path):
    for part in path:
        config = config[part]
    return config


def apply_features(config, case, coins, markets, bars):
    flags = set(case.features)
    if flags & {"hsl_never", "hsl_threshold", "hsl_unified", "entry_gate_off"}:
        raise ValueError("retired benchmark feature; freeze a current v9 workload")
    if "hsl_scaled" in flags:
        flags.add("excess")
    active = ("long", "short") if case.sides == "both" else (case.sides,)
    config["backtest"].update(offline=True, dynamic_wel_by_tradability="dynamic_wel_off" not in flags)
    config['bot']['global']['hedge_mode'] = case.sides == "both" and "one_way" not in flags
    if "loss_gate" in flags:
        config["bot"]["global"]["loss_gate"]["max_drawdown_ratio"] = 0.001
    if "min_cost" in flags:
        config["backtest"]["filter_by_min_effective_cost"] = True
    if "fill_buffer" in flags:
        config["backtest"]["limit_order_fill_buffer_fraction"] = 0.001
    if "slippage" in flags:
        config["backtest"]["market_order_slippage_fraction"] = 0.003
    if "btc_collateral" in flags:
        config["backtest"]["btc_collateral_cap"] = 0.25
    if "eligibility" in flags and len(coins) > 1:
        markets[coins[-1]]["first_valid_index"] = bars // 2
        markets[coins[0]]["last_valid_index"] = bars - bars // 4 - 1
    for side in ("long", "short"):
        bot = config["bot"][side]
        slots = 1 if case.topology == "single" else max(1, len(coins) // 2) if case.topology == "forager" else len(coins)
        bot['risk'].update({'max_positions': slots if side in active else 0, 'total_wallet_exposure_limit': 1.0 if side in active else 0.0, 'we_excess_allowance_fraction': 0.2 if 'excess' in flags else 0.0, 'position_exposure_enforcer': {'enabled': 'position_enforcer' in flags, 'threshold_ratio': 0.8}, 'total_exposure_enforcer': {'enabled': bool(flags & {'total_enforcer', 'total_enforcer_portfolio'}), 'policy': 'reduce_portfolio' if 'total_enforcer_portfolio' in flags else 'reduce_overweight', 'threshold_ratio': 0.8}})
        bot["unstuck"].update(enabled=side in active and bool(flags & {"unstuck", "unstuck_no_ema"}),
                              ema_gating_enabled="unstuck_no_ema" not in flags)
        bot["entry_cooldown"].update(base_duration_minutes=5.0 if flags & {"cooldown", "adaptive_cooldown"} else 0.0)
        if "adaptive_cooldown" in flags:
            bot["entry_cooldown"].update(min_duration_minutes=1.0, max_duration_minutes=60.0,
                                          weights_minutes=dict(exposure_ratio=10.0, adverse_directionality=5.0))
        weights = bot["forager"]["score_weights"]
        weights.update(volume=0.3, volatility=0.3, ema_readiness=0.4, unilateralness=0.2 if "forager_unilateral" in flags else 0.0)
        if case.topology != "forager":
            weights.update({key: 0.0 for key in weights})
        if "forager_hysteresis" in flags:
            config['bot']['global']['forager']['score_hysteresis_margin'] = 0.1
        policy = config["bot"]["global"]["hsl"] if config["bot"]["global"]["hsl"]["scope"] == "portfolio" else bot["hsl"]
        if "hsl_slow_memory" in flags:
            policy["memory_ema_span_minutes"] = 10.5
        if "hsl_zero_cooldown" in flags:
            policy["cooldown_minutes_after_red"] = 0.0
        policy["scale_budget_with_excess_allowance"] = "hsl_scaled" in flags
        policy["panic_close_order_type"] = "market" if "hsl_market" in flags else "limit"
        strategy = bot["strategy"][case.strategy]
        if case.strategy == "trailing_martingale":
            for leg in ("entry", "close"):
                for key in list(strategy[leg]):
                    if "weight" in key:
                        strategy[leg][key] = 0.0
                strategy[leg]["retracement_base_fraction"] = 0.001 if f"{leg}_trailing" in flags else 0.0
            for flag, mode in (("gate_disabled", "disabled"), ("gate_all", "all"), ("gate_reentry", "reentry")):
                if flag in flags:
                    strategy["entry"]["ema_gate_mode"] = mode
            if "volatility" in flags:
                for leg in ("entry", "close"):
                    for key in strategy[leg]:
                        if "volatility" in key and "weight" in key:
                            strategy[leg][key] = 0.1
            if "exposure_weight" in flags:
                strategy["entry"]["threshold_we_weight"] = 0.01
                strategy["close"]["threshold_we_weight"] = -0.01
        elif case.strategy == "ema_anchor":
            for key in strategy:
                if "weight" in key:
                    strategy[key] = 0.1 if "volatility" in flags else 0.0
    if "overrides" in flags:
        config["coin_overrides"] = {coins[0]: {"bot": {side: {"entry_cooldown": {"base_duration_minutes": 3.0},
            "risk": {"we_excess_allowance_fraction": 0.13}} for side in active}}}


def sampled_population(config, case, count, seed, bounds_config=None, boundary_candidates=False):
    import numpy as np
    from config.hsl import generated_template, validate_hsl_paths
    from config.schema import get_template_config
    from optimization.config_adapter import _flatten_bounds_for_config, get_optimization_key_paths
    from optimization.bounds import Bound, enforce_bounds

    search = deepcopy(config)
    source_bounds = bounds_config
    if source_bounds is None:
        template = generated_template(get_template_config(), config["bot"]["global"]["hsl"]["scope"])
        source_bounds = template["optimize"]["bounds"]
        # This generated benchmark profile explicitly searches every positive
        # score coordinate. An authored bounds_config remains authoritative.
        for side in ("long", "short"):
            if config["bot"][side]["forager"]["score_weights"]["unilateralness"] > 0:
                source_bounds[side]["forager"].setdefault("score_weights", {})["unilateralness"] = [0, 1, 0.01]
    search["optimize"]["bounds"] = deepcopy(source_bounds)
    validate_hsl_paths(search)
    flat = _flatten_bounds_for_config(search, search["optimize"]["bounds"])
    paths = get_optimization_key_paths(search)
    flags = set(case.features)
    active = ("long", "short") if case.sides == "both" else (case.sides,)
    specs = []
    for key, path in paths:
        bound = Bound.from_config(key, flat[key])
        value = get_path(config, path)
        portfolio_hsl = path[:3] == ("bot", "global", "hsl")
        enabled = (portfolio_hsl and config["bot"]["global"]["hsl"]["scope"] == "portfolio" and
                   any(flag.startswith("hsl") for flag in flags)) or path[1] in active and (
            "strategy" in path or path[2] == "risk" or
            path[2] == "forager" and case.topology == "forager" or
            path[2] == "unstuck" and bool(flags & {"unstuck", "unstuck_no_ema"}) or
            path[2] == "entry_cooldown" and bool(flags & {"cooldown", "adaptive_cooldown"}) or
            path[2] == "hsl" and config["bot"]["global"]["hsl"]["scope"] != "portfolio" and
            any(flag.startswith("hsl") for flag in flags))
        # Preserve the class: policies are fixed; zero weights/retracements stay zero.
        if value is None or isinstance(value, bool):
            continue
        if path[-1] == "max_positions" or value == 0.0 and ("weight" in path[-1] or "retracement" in path[-1]):
            enabled = False
        if path[-1] == "memory_ema_span_minutes" and bounds_config is None:
            enabled = False
        if path[-1] == "we_excess_allowance_fraction" and "excess" not in flags:
            enabled = False
        if path[2] == "risk" and path[-1] == "total_wallet_exposure_limit" and path[1] in active:
            # The public short template pins TWEL=0. The short class needs positive bounds.
            bound = Bound(0.5, 1.5, 0.01) if bounds_config is None else bound
            if bound.low <= 0:
                raise ValueError("active-side exposure bounds must stay positive")
        if path[2] == "hsl" and enabled and bounds_config is None:
            bound = {"drawdown_ratio_threshold": Bound(.002, .01, .001), "drawdown_ratio_ema_span_minutes": Bound(2, 20, 1),
                     "cooldown_minutes_after_red": Bound(5, 30, 1)}[path[-1]]
        if not enabled:
            bound = Bound(float(value), float(value))
        elif value != 0 and ("weight" in path[-1] or "retracement_base_fraction" == path[-1]) and bound.low == bound.high == 0:
            bound = Bound(float(value), float(value))
        specs.append((key, path, bound))
    bounds = [item[2] for item in specs]
    saved = random.getstate()
    saved_numpy = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        vectors = [enforce_bounds([bound.random_on_grid() for bound in bounds], bounds,
                                 config["optimize"]["round_to_n_significant_digits"]) for _ in range(count)]
        if boundary_candidates:
            if count < 3:
                raise ValueError("boundary sampling requires at least three candidates including random candidates")
            vectors[:2] = [enforce_bounds([getattr(bound, endpoint) for bound in bounds], bounds,
                                         config["optimize"]["round_to_n_significant_digits"])
                           for endpoint in ("low", "high")]
    finally:
        random.setstate(saved)
        np.random.set_state(saved_numpy)
    candidates = []
    for vector in vectors:
        candidate = deepcopy(config)
        for (_, path, _), value in zip(specs, vector):
            set_path(candidate, path, value)
        candidates.append(candidate)
    return candidates, vectors, [dict(key=key, path=path, low=b.low, high=b.high, step=b.step) for key, path, b in specs]


def freeze(directory, *, profile="smoke", patterns=(), candidates=8, seeds=(7,), bars=512,
           coins=4, bounds_config=None, boundary_candidates=False):
    import numpy as np
    from tools.gpu_parity import build_parser, fixture_inputs
    from config_utils import clean_config

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("workload destination must be empty")
    if not 1 <= candidates <= 4096 or not 128 <= bars <= 5_300_000 or not 2 <= coins <= 64:
        raise ValueError("require 1..4096 candidates, 128..5300000 bars and 2..64 multicoin coins")
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for case in select_cases(profile, patterns):
        for seed in seeds:
            hsl = next((mode for mode in ("coin", "pside", "portfolio") if f"hsl_{mode}" in case.features),
                       "coin" if any(f.startswith("hsl") for f in case.features) else "disabled")
            # Catalogue feature IDs remain stable; authored scope uses its full name.
            hsl = "position_side" if hsl == "pside" else hsl
            options = ["--fixture", case.strategy,
                       "--sides", case.sides, "--coins", str(1 if case.topology == "single" else coins),
                       "--bars", str(bars), "--seed", str(seed), "--hsl", hsl,
                       "--hsl-red-threshold", "0.005", "--hsl-drawdown-ratio-ema-span-minutes", "5", "--hsl-cooldown-minutes", "10"]
            config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args(options))
            source_coins = config["backtest"]["coins"]
            config = clean_config(config)
            config["backtest"]["coins"] = source_coins
            if "coarse_candles" in case.features:
                timestamps = timestamps[0] + np.arange(bars, dtype=np.int64) * 300_000
                timestamps -= (int(timestamps[-1]) + 300_000) % 86_400_000
                config["backtest"].update(candle_interval_minutes=5,
                    start_date=datetime.fromtimestamp(int(timestamps[0]) / 1000, timezone.utc).isoformat(),
                    end_date=datetime.fromtimestamp((int(timestamps[-1]) + 300_000) / 1000, timezone.utc).isoformat())
            if case.regime == "shock":
                candles[bars // 2:, 0, :3] *= 0.7
                if candles.shape[1] > 1:
                    candles[bars * 3 // 4:, 1, :3] *= 1.3
            if case.regime == "trend":
                candles[:, :, :3] *= np.linspace(1, 1.5, bars)[:, None, None]
            selected_coins = config["backtest"]["coins"]["binance"]
            apply_features(config, case, selected_coins, markets, bars)
            population, vectors, bounds = sampled_population(config, case, candidates, seed, bounds_config, boundary_candidates)
            metrics = list(CORE_METRICS)
            if any(f.startswith("hsl") for f in case.features):
                metrics += ["hard_stop_triggers_per_year", "hard_stop_time_in_red_pct"]
            if "equity_metrics" in case.features:
                metrics += ["adg_strategy_eq", "fills_gap_p95_hours", "strategy_eq_recovery_days_p95"]
            entry = dict(case=case.record(), seed=seed, metrics=metrics, configs=population,
                         vectors=vectors, bounds=bounds, markets=markets, exchange="binance",
                         data_sha256=array_digest(candles, btc, timestamps),
                         declared_gpu_limitation=declared_gpu_support(case),
                         boundary_candidates=2 if boundary_candidates else 0,
                         duplicate_candidates=len(vectors) - len({tuple(v) for v in vectors}))
            entry["diagnostic_limits"] = config["optimize"]["limits"]
            entry["id"] = digest(entry)
            entry["data_file"] = entry["id"] + ".npz"
            np.savez(directory / entry["data_file"], hlcvs=candles, btc=btc, timestamps=timestamps)
            entries.append(entry)
    manifest = dict(version=VERSION, profile=profile, entries=entries)
    manifest["sha256"] = digest(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return json.loads(json.dumps(manifest))


def load_manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    recorded = manifest.pop("sha256")
    if manifest["version"] != VERSION or digest(manifest) != recorded:
        raise ValueError("workload manifest version or integrity mismatch")
    manifest["sha256"] = recorded
    for entry in manifest["entries"]:
        if Path(entry["data_file"]).name != entry["data_file"]:
            raise ValueError("dataset filenames must be local basenames")
        identity = {key: value for key, value in entry.items() if key not in {"id", "data_file"}}
        if digest(identity) != entry["id"]:
            raise ValueError("candidate identity mismatch")
    return manifest
