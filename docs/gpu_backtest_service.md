# Prepared CUDA backtest service

`optimization.gpu.native.CudaBacktestService` is an internal execution interface under
development. CPU code registers prepared scenarios, submits identified backtest requests
and receives futures containing compact metrics. Device buffers, packing, replay handles
and residency stay inside the service. No CPU backtest or evolutionary algorithm runs
there. Optimizer integration and practical simulation-parity acceptance remain separate.

## Input ownership

Prepare effective config, market settings and immutable candle/BTC/timestamp arrays on
the CPU. Allocate the arrays once using `SharedArrayManager`; keep the segments immutable
and alive until the service has closed. `PreparedGpuDataset` snapshots metadata and
shared-array descriptors without attaching or copying candle histories at registration.
Read-only attachments protect worker-side views; they do not make another owner's
mutable alias safe to change during a run.

Supply `candle_coins` in the actual source-column order. Optional `coin_indices` select
those columns, and their identities/order must exactly match the scenario's sorted
`config.backtest.coins[exchange]`. Optional `time_range` selects a half-open row interval.
The CPU must prepare config dates and market validity/warmup metadata for that effective
scenario; the service does not infer missing inputs or substitute market settings.
Requested metrics specify backtest work, not scoring directions or constraint penalties.

```python
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.gpu.executor import BacktestRequest
from optimization.gpu.native import CudaBacktestService
from shared_arrays import SharedArrayManager

# config, markets, exchange, ordered_coins, candles, btc and timestamps
# have already been prepared for this effective scenario on the CPU.
arrays = SharedArrayManager()
try:
    specs = [arrays.create_from(value)[0] for value in (candles, btc, timestamps)]
    dataset = PreparedGpuDataset(
        config=config, markets=markets, exchange=exchange,
        hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
        candle_coins=ordered_coins, metrics=["adg_strategy_eq", "fills_per_day"],
    )
    with CudaBacktestService() as service:
        service.register_dataset("scenario", dataset)
        future = service.submit(BacktestRequest("candidate", "scenario", {}))
        result = future.result()
        # The caller scores, reduces, records and selects using result.metrics.
finally:
    arrays.cleanup()
```

An empty parameter mapping evaluates the dataset's base strategy parameters. The
transitional replay adapter also accepts its materialized scalar parameter mapping;
unsupported topology changes require a separately prepared dataset. The request and
dataset IDs are caller-owned identities. Persistent content/evaluation fingerprints,
precision stamps and resume compatibility are subsequent integration work.

## Execution and cleanup

The first accepted request creates one CUDA residency context on its owning worker.
Dataset attachment, subset preparation, replay construction, evaluation and cleanup
belong to that worker. Unused registrations and an unused service do not initialize GPU
dependencies. Repeated requests reuse their replay and compatible immutable packing.

The current policy keeps one active dataset's tensors and one replay's scratch on CUDA.
Other packed inputs and reusable coin subsets live in run-local disk files. Scenario
switches evict device buffers while keeping reusable immutable packing. Initial invariant
inputs must fit the existing 45% free-VRAM budget; scratch and other allocations can
still fail and propagate. This is a bounded device-residency foundation, not a complete
host/disk admission budget or adaptive multi-device scheduler.

Admission bounds queued plus running work. Adjacent compatible requests form bounded
microbatches; each receives an identified future rather than a generation-wide barrier.
`close()` drains accepted work and joins the owner. `close(cancel_pending=True)` cancels
queued work and waits for the running dispatch. An optional interrupt callback is checked
by replay execution to stop at its safe boundaries. Device, preparation and interrupt
failures fail admitted work and stop new admission; no CPU fallback supplies results.
Cleanup closes attachments and removes run-local packing/subset files, preserving an
original failure if cleanup also fails.

The shared-account engine now permits 1..64 selected coins. This facade uses that
implementation internally; legacy optimizer routing is unchanged. Short synthetic
one-coin measurements show a substantial throughput disadvantage against the old
single-coin implementation. Kernel ablation and representative measurements are required
before selecting the final native optimizer's default execution policy.
