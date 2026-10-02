"""Offline synthetic timing fixture for the internal hsl GPU runners."""

import argparse
import hashlib
import json
import statistics

import numpy as np

from optimization.gpu.model import (
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS as KEYS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS as MULTI_KEYS,
)
from optimization.gpu.mps_kernel import (
    MpsTrailingMartingaleRunner,
    MpsTrailingMartingaleMulticoinFusedRunner,
)
from optimization.gpu.runtime import gpu_device
from optimization.gpu.service import build_mps_data
from optimization.gpu.model import build_mps_multicoin_data
from tools.gpu_proxy_benchmark import (
    _market_and_run,
    _parameter_matrix,
    _synthetic_hlcvs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", type=int, default=1)
    parser.add_argument("--mode", choices=["coin", "pside", "unified"], default="coin")
    parser.add_argument("--minutes", type=int, default=4000)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--lookback-days", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument(
        "--scratch-mib",
        type=int,
        default=None,
        help="Single-coin HSL scratch ceiling; compare complete output digests",
    )
    args = parser.parse_args()
    if (
        args.minutes < 3
        or args.candidates < 1
        or args.runs < 1
        or not 1 <= args.coins <= 8
        or (args.scratch_mib is not None and (args.scratch_mib <= 0 or args.coins != 1))
    ):
        parser.error(
            "minutes >= 3, candidates >= 1, runs >= 1, coins in [1,8]; scratch requires one coin and positive MiB"
        )
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    candles, timestamps = _synthetic_hlcvs(args.minutes, args.coins, 43)
    market, run = _market_and_run(timestamps, args.minutes)
    data = build_mps_data(
        candles[:, 0, 0], candles[:, 0, 1], candles[:, 0, 2], timestamps, run, market
    )
    if args.coins > 1:
        data = build_mps_multicoin_data(
            candles,
            timestamps,
            [run] * args.coins,
            [market] * args.coins,
            include_hourly_ranges=True,
        )
    params = _parameter_matrix(
        KEYS if args.coins == 1 else MULTI_KEYS,
        args.candidates,
        43,
        value_overrides={
            "hsl_enabled": 1.0,
            "hsl_restart_policy": 0.0,
            "hsl_signal_mode": {"coin": 2.0, "pside": 1.0, "unified": 0.0}[args.mode],
            "hsl_red_threshold": args.threshold,
            "total_wallet_exposure_limit": 5.0,
            "entry_double_down_factor": 2.0,
        },
    )
    matrix = np.concatenate([params, params], axis=1)
    if args.coins == 1:
        runner = MpsTrailingMartingaleRunner(
            market,
            run,
            data,
            long_enabled=True,
            short_enabled=True,
            pnl_lookback_bars=args.lookback_days * 1440,
        )
    else:
        runner = MpsTrailingMartingaleMulticoinFusedRunner(
            run, data, pnl_lookback_bars=args.lookback_days * 1440
        )
    timings = []
    if args.scratch_mib is not None:
        runner.hsl_scratch_budget_bytes = args.scratch_mib * 1024**2
    for iteration in range(args.runs + 1):
        output = runner.run(matrix, profile=True)
        if iteration:
            timings.append(runner.last_profile["kernel_seconds"])
    digest = hashlib.sha256()
    for key, value in sorted(output.items()):
        array = value.cpu().numpy()
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    print(
        json.dumps(
            dict(
                device=str(gpu_device()),
                minutes=args.minutes,
                coins=args.coins,
                mode=args.mode,
                candidates=args.candidates,
                lookback_days=args.lookback_days,
                scratch_mib=args.scratch_mib,
                result_sha256=digest.hexdigest(),
                warm_kernel_seconds=timings,
                median_kernel_seconds=statistics.median(timings),
                fills=float(output["fill_count"].sum()),
                stops=float(
                    (output["hsl_triggers_long"] + output["hsl_triggers_short"]).sum()
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
