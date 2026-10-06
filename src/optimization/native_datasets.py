"""Bind canonical CPU preparation to GPU datasets without copying candle histories.

Existing candle/BTC segments are borrowed; only compact timestamp windows are owned
here and reused by content. Close the service before closing this registry.
"""

import json
import logging

import numpy as np

from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
from optimization.native_results import CanonicalResultScorer
from optimization.prepared_dataset_identity import _array_identity
from shared_arrays import SharedArrayManager


class NativeDatasetRegistry:
    def __init__(self, evaluator, *, standalone_candle_coins=None, metrics=None, overrides_list=None):
        from optimize import config_to_individual, _canonicalize_optimizer_individual

        self._arrays = SharedArrayManager()
        self._timestamps = {}
        self._closed = False
        self.scorer = CanonicalResultScorer(evaluator)
        base = self.scorer.base
        overrides = tuple(overrides_list if overrides_list is not None else
                          base.config["optimize"].get("enable_overrides", []))
        vector = config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
        effective = _canonicalize_optimizer_individual(
            vector, base.config, base.bounds, base.sig_digits, base.key_paths, overrides,
        )
        if isinstance(metrics, str):
            raise ValueError("requested metrics must be a collection of names")
        requested = None if metrics is None else tuple(metrics)
        bindings = []
        try:
            if self.scorer.suite:
                for ctx in evaluator.contexts:
                    scenario = evaluator.build_scenario_candidate_config(effective, ctx)
                    for exchange in ctx.exchanges:
                        lazy = (ctx.master_hlcvs_specs or {}).get(exchange) is not None
                        candles = ctx.master_hlcvs_specs[exchange] if lazy else ctx.hlcvs_specs[exchange]
                        btc = ((ctx.master_btc_specs or {}).get(exchange) if lazy
                               else ctx.btc_usd_specs.get(exchange))
                        span = (ctx.time_slice or {}).get(exchange) if lazy else None
                        indices = ((ctx.coin_slice_indices or {}).get(exchange) if lazy
                                   else ctx.coin_indices.get(exchange))
                        columns = (ctx.candle_coins or {}).get(exchange)
                        if columns is None:
                            raise ValueError("prepared scenario must retain actual candle-column identities")
                        bindings.append(self._binding(
                            ctx.label, exchange, scenario, ctx.msss[exchange], candles, btc,
                            ctx.timestamps.get(exchange), columns, span, indices, requested, len(bindings),
                        ))
            else:
                for exchange in base.exchanges:
                    columns = (standalone_candle_coins or {}).get(exchange)
                    if columns is None:
                        raise ValueError("standalone preparation must supply actual candle-column identities")
                    bindings.append(self._binding(
                        "base", exchange, effective, base.msss[exchange], base.hlcvs_specs[exchange],
                        base.btc_usd_specs.get(exchange), base.timestamps.get(exchange), columns,
                        None, None, requested, len(bindings),
                    ))
            self.bindings = tuple(bindings)
            self.planner = NativeCandidatePlanner(evaluator, self.bindings, overrides_list=overrides)
            # Metric surface and effective static inputs must be valid before registration.
            self.planner.prepare("preparation", vector)
        except BaseException:
            try:
                self.close()
            except BaseException:
                logging.exception("native dataset cleanup failed after preparation failure")
            raise

    def _binding(self, label, exchange, config, markets, candles, btc, timestamps,
                 columns, span, indices, requested, index):
        if timestamps is None:
            raise ValueError("prepared GPU scenario requires actual timestamps")
        timestamps = np.asarray(timestamps)
        if timestamps.ndim != 1 or timestamps.dtype.kind not in "iu":
            raise ValueError("prepared timestamps must be a one-dimensional integer array")
        identity = _array_identity(timestamps)
        key = json.dumps(identity, sort_keys=True)
        if key not in self._timestamps:
            self._timestamps[key] = self._arrays.create_from(timestamps)[0]
        selected_metrics = requested if requested is not None else tuple(sorted(
            self.scorer.required_metrics(label) or {"backtest_completion_ratio"},
        ))
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange=exchange, hlcvs=candles, btc=btc,
            timestamps=self._timestamps[key], timestamp_range=(0, len(timestamps)),
            candle_coins=columns, time_range=span, coin_indices=indices, metrics=selected_metrics,
        )
        return ScenarioBinding(label, f"native:{index}", dataset)

    def register(self, service):
        if self._closed:
            raise RuntimeError("native dataset registry is closed")
        for binding in self.bindings:
            service.register_dataset(binding.dataset_id, binding.dataset)

    def close(self):
        if not self._closed:
            self._closed = True
            failure = None
            for spec in self._timestamps.values():
                try:
                    self._arrays.cleanup([spec])
                except BaseException as error:
                    if failure is None:
                        failure = error
                    else:
                        logging.exception("native dataset cleanup failed after an earlier cleanup failure")
            self._timestamps.clear()
            if failure is not None:
                raise failure

    def __enter__(self):
        if self._closed:
            raise RuntimeError("native dataset registry is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            try:
                self.close()
            except BaseException:
                logging.exception("native dataset cleanup failed after caller failure")
        else:
            self.close()
