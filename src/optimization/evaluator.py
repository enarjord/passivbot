"""
Fitness evaluators for passivbot optimization.

Evaluator: single-scenario backtesting (one config → metrics).
SuiteEvaluator: multi-scenario with aggregation.
"""

import logging
from copy import deepcopy
from typing import List, Dict, Any

import numpy as np

from backtest import build_backtest_payload, execute_backtest
from optimization.bounds import apply_config_overrides, individual_to_config
from optimizer_overrides import optimizer_overrides
from shared_arrays import attach_shared_array
from metrics_schema import build_scenario_metrics
from optimize_suite import ScenarioEvalContext
from suite_runner import (
    SuiteScenario,
    ScenarioResult,
    aggregate_metrics,
    build_suite_metrics_payload,
)


class Evaluator:
    def __init__(
        self,
        hlcvs_specs,
        btc_usd_specs,
        msss,
        config,
        timestamps=None,
        shared_array_manager=None,
    ):
        logging.debug("Initializing Evaluator...")
        self.hlcvs_specs = hlcvs_specs
        self.btc_usd_specs = btc_usd_specs
        self.msss = msss
        self.timestamps = timestamps or {}
        self.exchanges = list(hlcvs_specs.keys())
        self.shared_array_manager = shared_array_manager
        self.shared_hlcvs_np = {}
        self.shared_btc_np = {}
        self._attachments = {"hlcvs": {}, "btc": {}}

        for exchange in self.exchanges:
            logging.debug("Preparing cached parameters for %s...", exchange)
            if self.shared_array_manager is not None:
                self.shared_hlcvs_np[exchange] = self.shared_array_manager.view(
                    self.hlcvs_specs[exchange]
                )
                btc_spec = self.btc_usd_specs.get(exchange)
                if btc_spec is not None:
                    self.shared_btc_np[exchange] = self.shared_array_manager.view(btc_spec)

        self.config = config
        logging.debug("Evaluator initialization complete.")
        logging.info("Evaluator ready | exchanges=%d", len(self.exchanges))

    def _ensure_attached(self, exchange: str) -> None:
        if exchange not in self.shared_hlcvs_np:
            spec = self.hlcvs_specs[exchange]
            attachment = attach_shared_array(spec)
            self._attachments["hlcvs"][exchange] = attachment
            self.shared_hlcvs_np[exchange] = attachment.array
        if exchange not in self.shared_btc_np:
            btc_spec = self.btc_usd_specs.get(exchange)
            if btc_spec is not None:
                attachment = attach_shared_array(btc_spec)
                self._attachments["btc"][exchange] = attachment
                self.shared_btc_np[exchange] = attachment.array

    def evaluate(self, individual, overrides_list):
        config = individual_to_config(individual, optimizer_overrides, overrides_list, self.config)
        analyses = {}
        for exchange in self.exchanges:
            self._ensure_attached(exchange)
            payload = build_backtest_payload(
                self.shared_hlcvs_np[exchange],
                self.msss[exchange],
                config,
                exchange,
                self.shared_btc_np[exchange],
                self.timestamps.get(exchange),
            )
            fills, equities_array, analysis = execute_backtest(payload, config)
            analyses[exchange] = analysis
            del fills
            del equities_array
        scenario_metrics = build_scenario_metrics(analyses)
        return {
            "stats": scenario_metrics.get("stats", {}),
        }

    def __del__(self):
        for attachment_map in self._attachments.values():
            for attachment in attachment_map.values():
                attachment.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("shared_hlcvs_np", None)
        state.pop("shared_btc_np", None)
        state.pop("_attachments", None)
        state.pop("shared_array_manager", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.shared_array_manager = None
        self.shared_hlcvs_np = {}
        self.shared_btc_np = {}
        self._attachments = {"hlcvs": {}, "btc": {}}
        for exchange in self.exchanges:
            self._ensure_attached(exchange)


class SuiteEvaluator:
    def __init__(
        self,
        base_evaluator: Evaluator,
        scenario_contexts: List[ScenarioEvalContext],
        aggregate_cfg: Dict[str, Any],
    ) -> None:
        self.base = base_evaluator
        self.contexts = scenario_contexts
        self.aggregate_cfg = aggregate_cfg
        # Cache for master dataset attachments (shared across scenarios)
        self._master_attachments: Dict[str, Dict[str, Any]] = {"hlcvs": {}, "btc": {}}
        self._master_arrays: Dict[str, Dict[str, np.ndarray]] = {"hlcvs": {}, "btc": {}}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_master_attachments", None)
        state.pop("_master_arrays", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._master_attachments = {"hlcvs": {}, "btc": {}}
        self._master_arrays = {"hlcvs": {}, "btc": {}}

    def _ensure_master_attachment(self, spec, cache_key: str, array_type: str) -> np.ndarray:
        """Attach to master SharedMemory if not already attached."""
        if cache_key not in self._master_arrays[array_type]:
            attachment = attach_shared_array(spec)
            self._master_attachments[array_type][cache_key] = attachment
            self._master_arrays[array_type][cache_key] = attachment.array
        return self._master_arrays[array_type][cache_key]

    def _get_lazy_slice_data(
        self, ctx: ScenarioEvalContext, exchange: str
    ) -> tuple[np.ndarray, np.ndarray | None, list[int] | None]:
        """
        Get data for lazy slicing mode.
        Returns (hlcvs_view, btc_view, coin_indices).

        Only applies TIME slicing here (creates views, O(1) memory).
        Coin subsetting is deferred to build_backtest_payload which does it efficiently.
        """
        master_spec = ctx.master_hlcvs_specs[exchange]
        master_array = self._ensure_master_attachment(master_spec, master_spec.name, "hlcvs")

        time_slice = ctx.time_slice.get(exchange) if ctx.time_slice else None
        coin_indices = ctx.coin_slice_indices.get(exchange) if ctx.coin_slice_indices else None

        # Time slicing creates a VIEW (no copy, O(1) memory)
        if time_slice is not None:
            start_idx, end_idx = time_slice
            hlcvs_view = master_array[start_idx:end_idx]
        else:
            hlcvs_view = master_array

        # BTC slice (time-only slicing creates a view)
        btc_view = None
        master_btc_spec = ctx.master_btc_specs.get(exchange) if ctx.master_btc_specs else None
        if master_btc_spec is not None:
            master_btc = self._ensure_master_attachment(master_btc_spec, master_btc_spec.name, "btc")
            if time_slice is not None:
                start_idx, end_idx = time_slice
                btc_view = master_btc[start_idx:end_idx]
            else:
                btc_view = master_btc

        # Return coin_indices to let build_backtest_payload handle subsetting in one step
        return hlcvs_view, btc_view, coin_indices

    def _uses_lazy_slicing(self, ctx: ScenarioEvalContext, exchange: str) -> bool:
        """Check if context uses lazy slicing for the given exchange."""
        return (
            ctx.master_hlcvs_specs is not None
            and exchange in ctx.master_hlcvs_specs
            and ctx.master_hlcvs_specs[exchange] is not None
        )

    def _ensure_context_attachment(self, ctx: ScenarioEvalContext, exchange: str) -> None:
        """Attach to SharedMemory for non-lazy-slicing contexts only."""
        # Skip if using lazy slicing - slices are computed on-demand in evaluate()
        if self._uses_lazy_slicing(ctx, exchange):
            return

        # Original flow: per-scenario SharedMemory
        if exchange not in ctx.shared_hlcvs_np:
            if exchange in ctx.hlcvs_specs and ctx.hlcvs_specs[exchange] is not None:
                attachment = attach_shared_array(ctx.hlcvs_specs[exchange])
                ctx.attachments["hlcvs"][exchange] = attachment
                ctx.shared_hlcvs_np[exchange] = attachment.array
        if exchange not in ctx.shared_btc_np and exchange in ctx.btc_usd_specs:
            if ctx.btc_usd_specs[exchange] is not None:
                attachment = attach_shared_array(ctx.btc_usd_specs[exchange])
                ctx.attachments["btc"][exchange] = attachment
                ctx.shared_btc_np[exchange] = attachment.array

    def evaluate(self, individual, overrides_list):
        config = individual_to_config(
            individual, optimizer_overrides, overrides_list, self.base.config
        )

        scenario_results: List[ScenarioResult] = []

        from tools.iterative_backtester import combine_analyses as combine

        for ctx in self.contexts:
            scenario_config = deepcopy(config)
            scenario_config["backtest"]["start_date"] = ctx.config["backtest"]["start_date"]
            scenario_config["backtest"]["end_date"] = ctx.config["backtest"]["end_date"]
            scenario_config["backtest"]["coins"] = deepcopy(ctx.config["backtest"]["coins"])
            scenario_config["backtest"]["cache_dir"] = deepcopy(
                ctx.config["backtest"].get("cache_dir", {})
            )
            scenario_config.setdefault("live", {})
            scenario_config["live"]["approved_coins"] = deepcopy(
                ctx.config["live"].get("approved_coins", {})
            )
            scenario_config["live"]["ignored_coins"] = deepcopy(
                ctx.config["live"].get("ignored_coins", {})
            )
            logging.debug(
                "Optimizer scenario %s | start=%s end=%s coins=%s",
                ctx.label,
                scenario_config["backtest"].get("start_date"),
                scenario_config["backtest"].get("end_date"),
                list(scenario_config["backtest"]["coins"].keys()),
            )
            if ctx.overrides:
                apply_config_overrides(scenario_config, ctx.overrides)
            scenario_config["disable_plotting"] = True

            analyses = {}
            for exchange in ctx.exchanges:
                # Get data arrays - either from lazy slicing or cached SharedMemory
                if self._uses_lazy_slicing(ctx, exchange):
                    # Get time-sliced VIEW (O(1) memory) + coin indices
                    # Coin subsetting happens inside build_backtest_payload (single copy)
                    hlcvs_data, btc_data, coin_indices = self._get_lazy_slice_data(ctx, exchange)
                else:
                    self._ensure_context_attachment(ctx, exchange)
                    hlcvs_data = ctx.shared_hlcvs_np[exchange]
                    btc_data = ctx.shared_btc_np.get(exchange)
                    coin_indices = ctx.coin_indices.get(exchange)

                payload = build_backtest_payload(
                    hlcvs_data,
                    ctx.msss[exchange],
                    scenario_config,
                    exchange,
                    btc_data,
                    ctx.timestamps.get(exchange),
                    coin_indices=coin_indices,
                )
                fills, equities_array, analysis = execute_backtest(payload, scenario_config)
                analyses[exchange] = analysis

                # Free backtest results to allow memory reuse
                del fills
                del equities_array
                del payload

            combined_metrics = combine(analyses)
            stats = combined_metrics.get("stats", {})
            logging.debug(
                "Scenario metrics | label=%s adg_pnl=%s peak_recovery_hours_pnl=%s",
                ctx.label,
                (
                    stats.get("adg_pnl", {}).get("mean")
                    if isinstance(stats.get("adg_pnl"), dict)
                    else stats.get("adg_pnl")
                ),
                (
                    stats.get("peak_recovery_hours_pnl", {}).get("mean")
                    if isinstance(stats.get("peak_recovery_hours_pnl"), dict)
                    else stats.get("peak_recovery_hours_pnl")
                ),
            )
            scenario_results.append(
                ScenarioResult(
                    scenario=SuiteScenario(
                        label=ctx.label,
                        start_date=None,
                        end_date=None,
                        coins=None,
                        ignored_coins=None,
                    ),
                    per_exchange={},
                    metrics={"stats": combined_metrics.get("stats", {})},
                    elapsed_seconds=0.0,
                    output_path=None,
                )
            )

        aggregate_summary = aggregate_metrics(scenario_results, self.aggregate_cfg)
        suite_payload = build_suite_metrics_payload(scenario_results, aggregate_summary)
        aggregate_stats = aggregate_summary.get("stats", {})

        # Build flat_stats override for aggregated values
        flat_stats_override = {}
        aggregated_values = aggregate_summary.get("aggregated", {})
        for metric, agg_value in aggregated_values.items():
            flat_stats_override[f"{metric}_mean"] = agg_value

        return {
            "stats": aggregate_stats,
            "suite_metrics": suite_payload,
            "flat_stats_override": flat_stats_override,
        }

    def __del__(self):
        for ctx in self.contexts:
            for attachment in ctx.attachments.get("hlcvs", {}).values():
                try:
                    attachment.close()
                except Exception:
                    logging.exception("Failed to close hlcvs attachment")
            for attachment in ctx.attachments.get("btc", {}).values():
                try:
                    attachment.close()
                except Exception:
                    logging.exception("Failed to close btc attachment")
