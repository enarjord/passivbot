"""Exercise an installed research wheel using only a synthetic local candle cache."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import passivbot_rust

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore
from procedures import FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION
from rust_utils import verify_loaded_runtime_extension
from tools.synthetic_backtest_data import synthetic_hlcvs


START = 1704067200000
SYMBOL = "BTC/USDT:USDT"


def write_offline_fixture():
    caches = Path("caches")
    (caches / "binance").mkdir(parents=True)
    market = {
        "id": "BTCUSDT",
        "symbol": SYMBOL,
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "type": "swap",
        "swap": True,
        "linear": True,
        "active": True,
        "spot": False,
        "future": False,
        "contract": True,
        "maker": 0.0002,
        "taker": 0.0005,
        "contractSize": 1.0,
        "precision": {"amount": 0.001, "price": 0.1},
        "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
    }
    inception = START - 86400000
    metadata = {
        "binance/markets.json": {SYMBOL: market},
        "binance/first_timestamps.json": {"BTC": inception},
        "first_ohlcv_timestamps_unified.json": {"BTC": inception},
        "first_ohlcv_timestamps_unified_exchange_specific.json": {
            "BTC": {"binanceusdm": inception}
        },
        "first_ohlcv_timestamps_unified_exchange_specific_symbols.json": {
            "BTC": {"binanceusdm": SYMBOL}
        },
    }
    for name, value in metadata.items():
        (caches / name).write_text(json.dumps(value), encoding="utf-8")
    (caches / "first_ohlcv_timestamps_unified.version").write_text(
        str(FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION), encoding="utf-8"
    )
    root = caches / "ohlcvs"
    store = OhlcvStore(root, OhlcvCatalog(root / "catalog.sqlite"))
    candles, _ = synthetic_hlcvs(1442, 1, seed=42)
    timestamps = START + np.arange(-1, 1441, dtype=np.int64) * 60_000
    store.write_rows("binance", "1m", SYMBOL, timestamps, candles[:, 0].astype(np.float32))

    no_hourly_volatility = {
        "threshold_log_range_1h_coefficient": 0.0,
        "retracement_log_range_1h_coefficient": 0.0,
    }
    config = {
        "config_version": "v9.0.0",
        "bot": {
            "global": {
                "universe": {"approved_coins": ["BTC"], "minimum_coin_age_days": 0},
                "strategy": {
                    "trailing_martingale": {
                        "entry": {
                            **no_hourly_volatility,
                            "ema_span_0_minutes": 5,
                            "ema_span_1_minutes": 10,
                        },
                        "close": no_hourly_volatility,
                    }
                },
            },
            "portfolio": {"warmup": {"max_minutes": 1, "activation_span_fraction": 1}},
            "long": {"enabled": True},
        },
        "backtest": {
            "offline": True,
            "exchanges": ["binance"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "base_dir": "backtest_results",
            "compress_cache": False,
            "gap_tolerance_ohlcvs_minutes": 0,
        },
        "optimize": {
            "backend": "pymoo",
            "iters": 4,
            "n_cpus": 1,
            "population_size": 4,
            "seed": 42,
            "write_all_results": True,
            "scoring": ["cagr_fraction_strategy_equity", "ulcer_index_fraction_strategy_equity"],
            "limits": [],
            "bounds": {
                "global": {
                    "strategy": {
                        "trailing_martingale": {"entry": {"initial_qty_fraction": [0.006, 0.018]}}
                    }
                }
            },
        },
    }
    Path("config.json").write_text(json.dumps(config), encoding="utf-8")


def main(expected_source):
    # A working directory can have unrelated sources. The wheel must neither
    # fingerprint those files nor try to compile them.
    decoy = Path("passivbot-rust")
    decoy.mkdir()
    (decoy / "Cargo.toml").write_text("unrelated sources\n", encoding="utf-8")
    runtime = verify_loaded_runtime_extension()
    assert Path(passivbot_rust.__file__).is_relative_to(Path(sys.prefix))
    assert runtime["expected_source_fingerprint"] == expected_source
    assert runtime["runtime_compiled_source_stamp"] == expected_source
    assert len(runtime["runtime_compiled_sha256"]) == 64

    write_offline_fixture()
    guard = Path("socket_guard")
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import socket\n"
        "_connect = socket.socket.connect\n"
        "def deny(*args, **kwargs): raise RuntimeError('offline wheel smoke attempted networking')\n"
        "def connect(sock, address):\n"
        "    if sock.family == socket.AF_UNIX: return _connect(sock, address)\n"
        "    return deny()\n"
        "socket.socket.connect = connect\n"
        "socket.getaddrinfo = deny\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(guard.resolve()))
    env.pop("SKIP_RUST_COMPILE", None)
    cli = Path(sys.executable).with_name("passivbot")
    for args in (
        ("tool", "config-schema", "--check"),
        ("download", "config.json", "--for-optimize", "--plan"),
        ("backtest", "config.json", "--plots", "none"),
        ("optimize", "config.json"),
    ):
        subprocess.run([str(cli), *args], env=env, check=True, timeout=120)

    backtest, = Path("backtest_results").rglob("session.json")
    identity = json.loads(backtest.read_text(encoding="utf-8"))["setup"]["implementation"]
    assert identity["rust_source_sha256"] == expected_source
    assert identity["rust_artifact_sha256"] == runtime["runtime_compiled_sha256"]
    fills = (backtest.parent / "fills.csv").read_text(encoding="utf-8").splitlines()
    assert len(fills) > 1
    results, = Path("optimize_results").rglob("all_results.bin")
    assert results.stat().st_size > 0
    assert (results.parent / "checkpoint.pkl").is_file()
    optimizer_session = json.loads((results.parent / "session.json").read_text(encoding="utf-8"))
    optimizer_identity = optimizer_session["setup"]["evaluation_contract"]["implementation"]
    assert optimizer_identity["rust_source_sha256"] == expected_source
    assert optimizer_identity["rust_artifact_sha256"] == runtime["runtime_compiled_sha256"]
    assert not (decoy / ".compile.lock").exists()
    print("Installed wheel completed offline planning, backtesting and CPU optimization.")


if __name__ == "__main__":
    main(sys.argv[1])
