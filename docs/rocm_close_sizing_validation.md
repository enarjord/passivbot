# ROCm close-sizing qualification

Validated on gfx1030 (RX 6800 XT), PyTorch 2.10.0+rocm7.0, HIP 7.0.51831 and
CuPy 14.2.0, against fc56fd1f0.

## Execution path

ROCm is not a separate torch implementation of the strategy. Runtime selects
the torch `cuda` device for AMD; `compile_shader` creates `CudaShaderLibrary`,
which translates the Rust-owned Metal strategy source to CUDA C++ and then
`_hipify` translates it for CuPy hipRTC. The close-sizing expressions and shared
loss-memory/unstuck helpers added by #2329 are therefore already on the HIP path.
The missing piece was the vendor admission policy, not another strategy port.
The existing gfx1030 private-secondary-slot workaround remains unchanged.

Enable AMD in the same policy as NVIDIA/Apple. Allow actual HIP devices in the
close-sizing replay/service fixtures; preserve all numerical assertions and
CPU-fallback prohibitions. Update policy tests to accept AMD and still reject
an unavailable target, with the scenario-override regression holding the vendor
fixed across both arms.

## Reproduction

Run from the repository root with the project virtual environment active.

Device suites:

```sh
.venv/bin/python -m pytest tests/optimization/test_gpu_cuda_close_sizing.py     tests/optimization/test_gpu_close_sizing_replay.py     tests/optimization/test_gpu_close_sizing_wide_parity.py -q
.venv/bin/python -m pytest tests/optimization/test_gpu_close_sizing_dispatch.py     tests/optimization/test_native_coupled_spans.py -q
.venv/bin/python -m pytest     tests/optimization/test_gpu_tm_loss_admission.py::test_tm_recursive_loss_plan_matches_finalized_rust_orders -q
```

Policy checks that need no device:

```sh
.venv/bin/python -m pytest tests/optimization/test_gpu_preflight.py -q
```

Reproducible CPU/GPU comparison from tracked fixtures:

```sh
.venv/bin/python src/tools/gpu_parity.py --fixture ema_anchor --sides both --coins 2 --bars 128
.venv/bin/python src/tools/gpu_parity.py --fixture trailing_martingale --sides both --coins 2 --bars 128
```

## Results on gfx1030

| Run | Result |
|---|---|
| close-sizing device suites | 41 passed, 0 failed, 0 skipped |
| dispatch and coupled-span suites | 41 passed, 0 failed, 0 skipped |
| recursive loss plan vs Rust | 12 passed, 0 failed, 0 skipped |
| preflight policy suite | 21 passed, 0 failed, 0 skipped |

The device suites assert exact fill counts, balance and position parity, fee-only
closes, zero allowance, override/candidate isolation, native service execution
without CPU fallback, and 64-coin fused TM parity. The recursive loss plan cases
compare GPU fills, balances and final positions with Rust for position and
portfolio exposure reducers, including market and limit closes.

Tracked fixture comparison with default fixture policy:

| Strategy | Metric | CPU | GPU | Relative error |
|---|---|---:|---:|---:|
| EMA Anchor | CAGR fraction | -0.9978358975547382 | -0.9978360040726858 | 1.07e-07 |
| EMA Anchor | max drawdown | 0.0014010733025554363 | 0.0014010915765538812 | 1.30e-05 |
| EMA Anchor | gross turnover/day | 16.538811489602843 | 16.53881159695712 | 6.49e-09 |
| Trailing Martingale | CAGR fraction | -0.42353811658165785 | -0.42340653768325043 | 3.11e-04 |
| Trailing Martingale | max drawdown | 0.00007522452576857359 | 0.00007525588443968445 | 4.17e-04 |
| Trailing Martingale | gross turnover/day | 0.26449897298185915 | 0.26449897072531964 | 8.53e-09 |

Both fixture runs report `status: passed` under the tool's default tolerances.
They are a smoke check of the shared replay machinery: the fixtures leave the
loss gate and unstuck inactive, so gate-active close sizing is proven by the
device suites above, not by these numbers.

Long-window real-market parity on the mixed15 configuration remains unverified;
that configuration is a local file and is not part of this change.
