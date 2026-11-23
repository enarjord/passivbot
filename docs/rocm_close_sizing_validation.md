# ROCm close-sizing qualification

Validated on gfx1030 (RX 6800 XT), PyTorch 2.10.0+rocm7.0,
HIP 7.0.51831 and CuPy 14.2.0, against fc56fd1f0.

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
an unavailable target.

## Reproduction

From /home/janek/software/pb9:

```sh
.venv/bin/python -m pytest tests/optimization/test_gpu_cuda_close_sizing.py tests/optimization/test_gpu_close_sizing_replay.py tests/optimization/test_gpu_close_sizing_wide_parity.py -q
.venv/bin/python -m pytest tests/optimization/test_gpu_close_sizing_dispatch.py tests/optimization/test_native_coupled_spans.py -q
PYTHONPATH=src:tests/optimization .venv/bin/python artifacts/rocm-close-sizing/parity_active.py
PYTHONPATH=src .venv/bin/python artifacts/rocm-close-sizing/parity_mixed15.py
```

Artifacts (local, not tracked): artifacts/rocm-close-sizing/.

The 41 device tests, 41 policy/native planner tests and 2 close-risk helper tests pass, including exact fill-count checks, balance/position
checks, fee-only closes, zero allowance, override/candidate isolation, native
service execution without CPU fallback, and 64-coin fused TM parity.

Independent native-service reports use 2 coins / 50 bars / both directions.
All three default metrics pass unchanged default tolerances:

| Strategy | Gate cap | CPU fills | CPU max drawdown | GPU max drawdown |
|---|---:|---:|---:|---:|
| EMA Anchor | 0 | 7 | 0.0011831999999997152 | 0.001183227519504726 |
| EMA Anchor | 0.005 | 44 | 0.05535679999999684 | 0.05535681173205376 |
| Trailing Martingale | 0 | 6 | 0.0009475999999997384 | 0.0009476318373344839 |
| Trailing Martingale | 0.005 | 42 | 0.0011071999999971922 | 0.001107177697122097 |

Max absolute error across these reports: CAGR 8.802e-9, max drawdown
3.184e-8, daily gross turnover 3.000e-6. Service tests separately assert GPU
fill counts equal CPU counts and duplicated candidates are identical.

The mixed15 config comparison uses identical CPU/GPU settings, 2 synthetic
coins / 4096 one-minute bars, reduced universe and a 60-minute warmup cap
(required for a short window; unrestricted warmup leaves no evaluated interval).
Unstuck and loss gate stay enabled. Both engines return zero fills and all
three metrics zero. This smoke check is not evidence of active close sizing
under the actual mixed15 market history; active evidence comes from the
adversarial fixtures above. Long-window real-market mixed15 parity remains
unverified.

## Optimization

```sh
cd /home/janek/software/pb9
.venv/bin/python src/optimize.py configs/mixed15_v9.json
```

The config already selects optimize.backend=gpu and enables unstuck/loss gate.
No optimization, push, or paid worker was used for this validation.
