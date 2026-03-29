import logging
import math
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import passivbot_rust as pbr

from backtest import build_backtest_payload, execute_backtest
from limit_utils import compute_limit_violation, expand_limit_checks
from metrics_schema import build_scenario_metrics, flatten_metric_stats
from optimization.bounds import enforce_bounds
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimizer_overrides import optimizer_overrides
from optimize_suite import ScenarioEvalContext
from pure_funcs import calc_hash
from shared_arrays import SharedArrayManager, attach_shared_array
from suite_runner import (
    ScenarioResult,
    SuiteScenario,
    aggregate_metrics,
    build_suite_metrics_payload,
)


def apply_config_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    if not overrides:
        return
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str):
            continue
        parts = dotted_path.split(".")
        if not parts:
            continue
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value


def individual_to_config(individual, overrides_func, overrides_list, template):
    config = deepcopy(template)
    i = 0
    for pside in sorted(config["bot"]):
        for key in sorted(config["bot"][pside]):
            config["bot"][pside][key] = individual[i]
            i += 1
    for pside in config["bot"]:
        config = overrides_func(overrides_list, config, pside)
    return config


class Evaluator:
    def __init__(
        self,
        hlcvs_specs,
        btc_usd_specs,
        msss,
        config,
        seen_hashes=None,
        duplicate_counter=None,
        timestamps=None,
        shared_array_manager: SharedArrayManager | None = None,
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
        self.seen_hashes = seen_hashes if seen_hashes is not None else {}
        self.duplicate_counter = duplicate_counter if duplicate_counter is not None else {"count": 0}
        self.bounds = extract_bounds_tuple_list_from_config(self.config)
        self.sig_digits = config.get("optimize", {}).get("round_to_n_significant_digits", 6)

        shared_metric_weights = {
            "positions_held_per_day": 1.0,
            "positions_held_per_day_w": 1.0,
            "position_held_hours_mean": 1.0,
            "position_held_hours_max": 1.0,
            "position_held_hours_median": 1.0,
            "position_unchanged_hours_max": 1.0,
            "high_exposure_hours_mean_long": 1.0,
            "high_exposure_hours_max_long": 1.0,
            "high_exposure_hours_mean_short": 1.0,
            "high_exposure_hours_max_short": 1.0,
            "adg_pnl": -1.0,
            "adg_pnl_w": -1.0,
            "mdg_pnl": -1.0,
            "mdg_pnl_w": -1.0,
            "sharpe_ratio_pnl": -1.0,
            "sharpe_ratio_pnl_w": -1.0,
            "sortino_ratio_pnl": -1.0,
            "sortino_ratio_pnl_w": -1.0,
        }

        currency_metric_weights = {
            "adg": -1.0,
            "adg_per_exposure_long": -1.0,
            "adg_per_exposure_short": -1.0,
            "adg_w": -1.0,
            "adg_w_per_exposure_long": -1.0,
            "adg_w_per_exposure_short": -1.0,
            "calmar_ratio": -1.0,
            "calmar_ratio_w": -1.0,
            "drawdown_worst": 1.0,
            "drawdown_worst_mean_1pct": 1.0,
            "equity_balance_diff_neg_max": 1.0,
            "equity_balance_diff_neg_mean": 1.0,
            "equity_balance_diff_pos_max": 1.0,
            "equity_balance_diff_pos_mean": 1.0,
            "equity_choppiness": 1.0,
            "equity_choppiness_w": 1.0,
            "equity_jerkiness": 1.0,
            "equity_jerkiness_w": 1.0,
            "peak_recovery_hours_equity": 1.0,
            "expected_shortfall_1pct": 1.0,
            "exponential_fit_error": 1.0,
            "exponential_fit_error_w": 1.0,
            "gain": -1.0,
            "gain_per_exposure_long": -1.0,
            "gain_per_exposure_short": -1.0,
            "loss_profit_ratio": 1.0,
            "loss_profit_ratio_w": 1.0,
            "mdg": -1.0,
            "mdg_per_exposure_long": -1.0,
            "mdg_per_exposure_short": -1.0,
            "mdg_w": -1.0,
            "mdg_w_per_exposure_long": -1.0,
            "mdg_w_per_exposure_short": -1.0,
            "omega_ratio": -1.0,
            "omega_ratio_w": -1.0,
            "sharpe_ratio": -1.0,
            "sharpe_ratio_w": -1.0,
            "sortino_ratio": -1.0,
            "sortino_ratio_w": -1.0,
            "sterling_ratio": -1.0,
            "sterling_ratio_w": -1.0,
            "total_wallet_exposure_max": 1.0,
            "total_wallet_exposure_mean": 1.0,
            "total_wallet_exposure_median": 1.0,
            "volume_pct_per_day_avg": -1.0,
            "volume_pct_per_day_avg_w": -1.0,
            "entry_initial_balance_pct_long": -1.0,
            "entry_initial_balance_pct_short": -1.0,
        }

        self.scoring_weights = {}
        self.scoring_weights.update(shared_metric_weights)

        for metric, weight in currency_metric_weights.items():
            self.scoring_weights[f"{metric}_usd"] = weight
            self.scoring_weights[f"{metric}_btc"] = weight
            self.scoring_weights.setdefault(metric, weight)
            self.scoring_weights.setdefault(f"usd_{metric}", weight)
            self.scoring_weights.setdefault(f"btc_{metric}", weight)

        self.build_limit_checks()

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

    def perturb_step_digits(self, individual, change_chance=0.5):
        perturbed = []
        for i, val in enumerate(individual):
            if np.random.random() < change_chance:
                perturbed.append(val)
                continue
            bound = self.bounds[i]
            if bound.high == bound.low:
                perturbed.append(val)
                continue

            if bound.is_stepped:
                step = bound.step
            elif val != 0.0:
                exponent = math.floor(math.log10(abs(val))) - (self.sig_digits - 1)
                step = 10**exponent
            else:
                step = (bound.high - bound.low) * 10 ** -(self.sig_digits - 1)

            direction = np.random.choice([-1.0, 1.0])
            new_val = val + step * direction
            if bound.is_stepped:
                perturbed.append(new_val)
            else:
                perturbed.append(pbr.round_dynamic(new_val, self.sig_digits))

        return perturbed

    def perturb_x_pct(self, individual, magnitude=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            bound = self.bounds[i]
            if bound.high == bound.low:
                perturbed.append(val)
                continue
            new_val = val * (1 + np.random.uniform(-magnitude, magnitude))
            if bound.is_stepped:
                perturbed.append(new_val)
            else:
                perturbed.append(pbr.round_dynamic(new_val, self.sig_digits))
        return perturbed

    def perturb_random_subset(self, individual, frac=0.2):
        perturbed = list(individual)
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            bound = self.bounds[i]
            if bound.low != bound.high:
                if bound.is_stepped:
                    direction = np.random.choice([-1.0, 1.0])
                    perturbed[i] = individual[i] + bound.step * direction
                else:
                    delta = (bound.high - bound.low) * 0.01
                    perturbed[i] = individual[i] + delta * np.random.uniform(-1.0, 1.0)
        return perturbed

    def perturb_sample_some(self, individual, frac=0.2):
        perturbed = list(individual)
        n = len(individual)
        indices = np.random.choice(n, max(1, int(frac * n)), replace=False)
        for i in indices:
            bound = self.bounds[i]
            if bound.low != bound.high:
                perturbed[i] = bound.random_on_grid()
        return perturbed

    def perturb_gaussian(self, individual, scale=0.01):
        perturbed = []
        for i, val in enumerate(individual):
            bound = self.bounds[i]
            if bound.high == bound.low:
                perturbed.append(val)
                continue
            if bound.is_stepped:
                max_steps = (bound.high - bound.low) / bound.step
                n_steps = int(np.random.normal(0, scale * max_steps) + 0.5)
                perturbed.append(val + n_steps * bound.step)
            else:
                noise = np.random.normal(0, scale * (bound.high - bound.low))
                perturbed.append(val + noise)
        return perturbed

    def perturb_large_uniform(self, individual):
        perturbed = []
        for i in range(len(individual)):
            bound = self.bounds[i]
            if bound.low == bound.high:
                perturbed.append(bound.low)
            else:
                perturbed.append(bound.random_on_grid())
        return perturbed

    def evaluate(self, individual, overrides_list):
        individual[:] = enforce_bounds(individual, self.bounds, self.sig_digits)
        config = individual_to_config(individual, optimizer_overrides, overrides_list, self.config)
        individual_hash = calc_hash(individual)
        if individual_hash in self.seen_hashes:
            existing_entry = self.seen_hashes[individual_hash]
            existing_score = None
            existing_penalty = 0.0
            if existing_entry is not None:
                existing_score, existing_penalty = existing_entry
            self.duplicate_counter["total"] += 1
            perturbation_funcs = [
                self.perturb_x_pct,
                self.perturb_step_digits,
                self.perturb_gaussian,
                self.perturb_random_subset,
                self.perturb_sample_some,
                self.perturb_large_uniform,
            ]
            for perturb_fn in perturbation_funcs:
                perturbed = perturb_fn(individual)
                perturbed = enforce_bounds(perturbed, self.bounds, self.sig_digits)
                new_hash = calc_hash(perturbed)
                if new_hash not in self.seen_hashes:
                    individual[:] = perturbed
                    self.seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed, optimizer_overrides, overrides_list, self.config
                    )
                    self.duplicate_counter["resolved"] += 1
                    break
            else:
                if existing_score is not None:
                    self.duplicate_counter["reused"] += 1
                    return tuple(existing_score), existing_penalty, None
        else:
            self.seen_hashes[individual_hash] = None
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
        aggregate_stats = scenario_metrics.get("stats", {})
        flat_stats = flatten_metric_stats(aggregate_stats)
        objectives, total_penalty = self.calc_fitness(flat_stats)
        objectives_map = {f"w_{i}": val for i, val in enumerate(objectives)}
        metrics_payload = {
            "stats": aggregate_stats,
            "objectives": objectives_map,
            "constraint_violation": total_penalty,
        }
        individual.evaluation_metrics = metrics_payload
        actual_hash = calc_hash(individual)
        self.seen_hashes[actual_hash] = (tuple(objectives), total_penalty)
        return tuple(objectives), total_penalty, metrics_payload

    def build_limit_checks(self):
        limits = self.config["optimize"].get("limits", [])
        objective_index_map: Dict[str, List[int]] = {}
        for idx, metric in enumerate(self.config["optimize"].get("scoring", [])):
            objective_index_map.setdefault(metric, []).append(idx)
        self.limit_checks = expand_limit_checks(
            limits,
            self.scoring_weights,
            penalty_weight=1e6,
            objective_index_map=objective_index_map,
        )

    def calc_fitness(self, analyses_combined):
        scoring_keys = self.config["optimize"]["scoring"]
        per_objective_modifier = [0.0] * len(scoring_keys)
        global_modifier = 0.0
        for check in self.limit_checks:
            val = analyses_combined.get(check["metric_key"])
            penalty = compute_limit_violation(check, val)
            if not penalty:
                continue
            targets = check.get("objective_indexes") or []
            if targets:
                for idx in targets:
                    if 0 <= idx < len(per_objective_modifier):
                        per_objective_modifier[idx] += penalty
            else:
                global_modifier += penalty

        total_penalty = global_modifier + sum(per_objective_modifier)
        scores = []
        for idx, sk in enumerate(scoring_keys):
            penalty_total = global_modifier + per_objective_modifier[idx]

            extended_candidates = []
            seen = set()
            for candidate in [sk, sk.replace("_mean", "")]:
                if candidate not in seen:
                    extended_candidates.append(candidate)
                    seen.add(candidate)
                for suffix in ("usd", "btc"):
                    with_suffix = f"{candidate}_{suffix}"
                    if with_suffix not in seen:
                        extended_candidates.append(with_suffix)
                        seen.add(with_suffix)
                    parts_candidate = candidate.split("_")
                    if len(parts_candidate) >= 2:
                        inserted = "_".join(parts_candidate[:-1] + [suffix, parts_candidate[-1]])
                        if inserted not in seen:
                            extended_candidates.append(inserted)
                            seen.add(inserted)

            val = None
            weight = None
            for candidate in extended_candidates:
                metric_key = f"{candidate}_mean"
                if val is None and metric_key in analyses_combined:
                    val = analyses_combined[metric_key]
                if weight is None and candidate in self.scoring_weights:
                    weight = self.scoring_weights[candidate]
                if val is not None and weight is not None:
                    break

            if val is None:
                val = 0
            if weight is None:
                weight = 1.0
            scores.append(val * weight + penalty_total)
        return tuple(scores), total_penalty

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
        self._master_attachments: Dict[str, Dict[str, Any]] = {"hlcvs": {}, "btc": {}}
        self._master_arrays: Dict[str, Dict[str, np.ndarray]] = {"hlcvs": {}, "btc": {}}

    def _ensure_master_attachment(self, spec, cache_key: str, array_type: str) -> np.ndarray:
        if cache_key not in self._master_arrays[array_type]:
            attachment = attach_shared_array(spec)
            self._master_attachments[array_type][cache_key] = attachment
            self._master_arrays[array_type][cache_key] = attachment.array
        return self._master_arrays[array_type][cache_key]

    def _get_lazy_slice_data(
        self, ctx: ScenarioEvalContext, exchange: str
    ) -> tuple[np.ndarray, np.ndarray | None, list[int] | None]:
        master_spec = ctx.master_hlcvs_specs[exchange]
        master_array = self._ensure_master_attachment(master_spec, master_spec.name, "hlcvs")

        time_slice = ctx.time_slice.get(exchange) if ctx.time_slice else None
        coin_indices = ctx.coin_slice_indices.get(exchange) if ctx.coin_slice_indices else None

        if time_slice is not None:
            start_idx, end_idx = time_slice
            hlcvs_view = master_array[start_idx:end_idx]
        else:
            hlcvs_view = master_array

        btc_view = None
        master_btc_spec = ctx.master_btc_specs.get(exchange) if ctx.master_btc_specs else None
        if master_btc_spec is not None:
            master_btc = self._ensure_master_attachment(master_btc_spec, master_btc_spec.name, "btc")
            if time_slice is not None:
                start_idx, end_idx = time_slice
                btc_view = master_btc[start_idx:end_idx]
            else:
                btc_view = master_btc

        return hlcvs_view, btc_view, coin_indices

    def _uses_lazy_slicing(self, ctx: ScenarioEvalContext, exchange: str) -> bool:
        return (
            ctx.master_hlcvs_specs is not None
            and exchange in ctx.master_hlcvs_specs
            and ctx.master_hlcvs_specs[exchange] is not None
        )

    def _ensure_context_attachment(self, ctx: ScenarioEvalContext, exchange: str) -> None:
        if self._uses_lazy_slicing(ctx, exchange):
            return

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
        individual[:] = enforce_bounds(individual, self.base.bounds, self.base.sig_digits)
        config = individual_to_config(
            individual, optimizer_overrides, overrides_list, self.base.config
        )
        individual_hash = calc_hash(individual)
        seen_hashes = self.base.seen_hashes
        duplicate_counter = self.base.duplicate_counter

        if individual_hash in seen_hashes:
            existing_entry = seen_hashes[individual_hash]
            existing_score = None
            existing_penalty = 0.0
            if existing_entry is not None:
                existing_score, existing_penalty = existing_entry
            duplicate_counter["total"] += 1
            perturbation_funcs = [
                self.base.perturb_x_pct,
                self.base.perturb_step_digits,
                self.base.perturb_gaussian,
                self.base.perturb_random_subset,
                self.base.perturb_sample_some,
                self.base.perturb_large_uniform,
            ]
            for perturb_fn in perturbation_funcs:
                perturbed = perturb_fn(individual)
                perturbed = enforce_bounds(perturbed, self.base.bounds, self.base.sig_digits)
                new_hash = calc_hash(perturbed)
                if new_hash not in seen_hashes:
                    individual[:] = perturbed
                    seen_hashes[new_hash] = None
                    config = individual_to_config(
                        perturbed, optimizer_overrides, overrides_list, self.base.config
                    )
                    duplicate_counter["resolved"] += 1
                    break
            else:
                if existing_score is not None:
                    duplicate_counter["reused"] += 1
                    return tuple(existing_score), existing_penalty, None
        else:
            seen_hashes[individual_hash] = None

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
                if self._uses_lazy_slicing(ctx, exchange):
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

        flat_stats = flatten_metric_stats(aggregate_stats)
        aggregated_values = aggregate_summary.get("aggregated", {})
        for metric, agg_value in aggregated_values.items():
            flat_stats[f"{metric}_mean"] = agg_value
        objectives, total_penalty = self.base.calc_fitness(flat_stats)
        objectives_map = {f"w_{i}": val for i, val in enumerate(objectives)}

        metrics_payload = {
            "objectives": objectives_map,
            "suite_metrics": suite_payload,
            "constraint_violation": total_penalty,
        }

        individual.evaluation_metrics = metrics_payload
        actual_hash = calc_hash(individual)
        self.base.seen_hashes[actual_hash] = (tuple(objectives), total_penalty)
        return tuple(objectives), total_penalty, metrics_payload

    def __del__(self):
        for ctx in self.contexts:
            for attachment in ctx.attachments.get("hlcvs", {}).values():
                try:
                    attachment.close()
                except Exception:
                    pass
            for attachment in ctx.attachments.get("btc", {}).values():
                try:
                    attachment.close()
                except Exception:
                    pass
