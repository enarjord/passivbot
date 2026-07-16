from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DATASET_IDENTITY_KEYS = (
    "cache_hash",
    "exchange",
    "coins",
    "requested_start_date",
    "requested_end_date",
)
ANALYSIS_METRICS = (
    "adg_strategy_eq",
    "drawdown_worst_strategy_eq",
    "loss_profit_ratio",
    "position_held_days_max",
    "high_exposure_days_max_long",
    "high_exposure_days_max_short",
    "strategy_eq_recovery_days_max",
)
FULL_FILL_KEY_COLUMNS = (
    "timestamp",
    "coin",
    "pnl",
    "fee_paid",
    "qty",
    "price",
    "psize",
    "pprice",
    "type",
    "liquidity",
    "minute",
)
EVENT_FILL_KEY_COLUMNS = ("timestamp", "coin", "type")


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing backtest artifact file: {path}") from None
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_csv(path: Path, *, compression: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing backtest artifact file: {path}")
    return pd.read_csv(path, compression=compression)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _row_counter(frame: pd.DataFrame, columns: Iterable[str]) -> Counter:
    selected = list(columns)
    return Counter(
        tuple(_normalize_scalar(value) for value in row)
        for row in frame.loc[:, selected].itertuples(index=False, name=None)
    )


def _multiset_match_ratio(
    base: pd.DataFrame, candidate: pd.DataFrame, columns: Iterable[str]
) -> float:
    base_counter = _row_counter(base, columns)
    candidate_counter = _row_counter(candidate, columns)
    matched = sum((base_counter & candidate_counter).values())
    return matched / max(len(base), len(candidate), 1)


def _common_prefix_rows(
    base: pd.DataFrame, candidate: pd.DataFrame, columns: Iterable[str]
) -> int:
    selected = list(columns)
    limit = min(len(base), len(candidate))
    for index in range(limit):
        base_row = tuple(
            _normalize_scalar(value) for value in base.loc[index, selected]
        )
        candidate_row = tuple(
            _normalize_scalar(value) for value in candidate.loc[index, selected]
        )
        if base_row != candidate_row:
            return index
    return limit


def _dataset_identity(base: dict, candidate: dict) -> dict:
    fields = {}
    missing = []
    mismatches = []
    for key in DATASET_IDENTITY_KEYS:
        base_value = base.get(key)
        candidate_value = candidate.get(key)
        present = (
            key in base
            and key in candidate
            and base_value not in (None, "")
            and candidate_value not in (None, "")
        )
        matches = present and base_value == candidate_value
        fields[key] = {
            "base": base_value,
            "candidate": candidate_value,
            "present_in_both": present,
            "matches": matches,
        }
        if not present:
            missing.append(key)
        elif not matches:
            mismatches.append(key)
    status = "mismatch" if mismatches else "unproven" if missing else "same"
    return {
        "status": status,
        "missing_fields": missing,
        "mismatched_fields": mismatches,
        "fields": fields,
    }


def _metric_comparison(base: dict, candidate: dict) -> dict:
    metrics = {}
    for key in ANALYSIS_METRICS:
        if key not in base or key not in candidate:
            continue
        base_value = float(base[key])
        candidate_value = float(candidate[key])
        absolute_delta = candidate_value - base_value
        metrics[key] = {
            "base": base_value,
            "candidate": candidate_value,
            "absolute_delta": absolute_delta,
            "relative_delta": (
                None if base_value == 0.0 else absolute_delta / abs(base_value)
            ),
        }
    return metrics


def _fill_comparison(base: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    missing_event_columns = sorted(
        set(EVENT_FILL_KEY_COLUMNS) - set(base.columns).intersection(candidate.columns)
    )
    if missing_event_columns:
        raise KeyError(
            "fills.csv files must both contain event identity columns: "
            + ", ".join(missing_event_columns)
        )
    exact_columns = tuple(
        column
        for column in FULL_FILL_KEY_COLUMNS
        if column in base.columns and column in candidate.columns
    )
    if not exact_columns:
        raise KeyError("fills.csv files have no common comparison columns")

    base_types = Counter(base["type"].astype(str))
    candidate_types = Counter(candidate["type"].astype(str))
    type_delta = {
        key: candidate_types.get(key, 0) - base_types.get(key, 0)
        for key in sorted(set(base_types) | set(candidate_types))
        if candidate_types.get(key, 0) != base_types.get(key, 0)
    }
    result = {
        "base_count": len(base),
        "candidate_count": len(candidate),
        "count_delta": len(candidate) - len(base),
        "event_multiset_match_ratio": _multiset_match_ratio(
            base, candidate, EVENT_FILL_KEY_COLUMNS
        ),
        "exact_multiset_match_ratio": _multiset_match_ratio(
            base, candidate, exact_columns
        ),
        "exact_match_columns": list(exact_columns),
        "common_prefix_rows": _common_prefix_rows(base, candidate, exact_columns),
        "type_count_delta": type_delta,
    }
    for column in ("pnl", "fee_paid"):
        if column in base.columns and column in candidate.columns:
            result[f"{column}_sum_base"] = float(pd.to_numeric(base[column]).sum())
            result[f"{column}_sum_candidate"] = float(
                pd.to_numeric(candidate[column]).sum()
            )
    return result


def _equity_series(frame: pd.DataFrame, *, source: str) -> pd.Series:
    if "strategy_equity" not in frame.columns:
        raise KeyError(f"{source} missing required column 'strategy_equity'")
    if "timestamp" in frame.columns:
        index = frame["timestamp"].astype(str)
    else:
        index_columns = [
            column for column in frame.columns if str(column).startswith("Unnamed")
        ]
        index = (
            frame[index_columns[0]].astype(str)
            if index_columns
            else frame.index.astype(str)
        )
    series = pd.Series(
        pd.to_numeric(frame["strategy_equity"], errors="raise").to_numpy(dtype=float),
        index=pd.Index(index, name="sample"),
    )
    if series.index.has_duplicates:
        raise ValueError(f"{source} contains duplicate equity sample identifiers")
    return series


def _equity_comparison(base: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    base_series = _equity_series(base, source="base balance_and_equity.csv.gz")
    candidate_series = _equity_series(
        candidate, source="candidate balance_and_equity.csv.gz"
    )
    common_index = base_series.index.intersection(candidate_series.index, sort=False)
    if common_index.empty:
        raise ValueError("balance/equity artifacts have no common samples")
    base_common = base_series.loc[common_index]
    candidate_common = candidate_series.loc[common_index]
    delta = candidate_common - base_common
    initial_equity = abs(float(base_series.iloc[0]))
    if initial_equity == 0.0:
        raise ValueError(
            "base strategy equity starts at zero; percentage comparison is undefined"
        )
    denominator = base_common.abs().replace(0.0, np.nan)
    final_base = float(base_series.iloc[-1])
    final_candidate = float(candidate_series.iloc[-1])
    return {
        "base_sample_count": len(base_series),
        "candidate_sample_count": len(candidate_series),
        "common_sample_count": len(common_index),
        "final_base": final_base,
        "final_candidate": final_candidate,
        "final_pct_delta": (
            None
            if final_base == 0.0
            else (final_candidate - final_base) / abs(final_base)
        ),
        "rmse_pct_of_initial": float(
            np.sqrt(np.mean(np.square(delta))) / initial_equity
        ),
        "max_abs_pct_delta": float((delta.abs() / denominator).max()),
    }


def compare_backtest_artifacts(base_dir: str | Path, candidate_dir: str | Path) -> dict:
    base_dir = Path(base_dir).expanduser().resolve()
    candidate_dir = Path(candidate_dir).expanduser().resolve()
    base_dataset = _load_json(base_dir / "dataset.json")
    candidate_dataset = _load_json(candidate_dir / "dataset.json")
    base_analysis = _load_json(base_dir / "analysis.json")
    candidate_analysis = _load_json(candidate_dir / "analysis.json")
    base_fills = _load_csv(base_dir / "fills.csv")
    candidate_fills = _load_csv(candidate_dir / "fills.csv")
    base_equity = _load_csv(base_dir / "balance_and_equity.csv.gz", compression="gzip")
    candidate_equity = _load_csv(
        candidate_dir / "balance_and_equity.csv.gz", compression="gzip"
    )
    return {
        "base_artifact_dir": str(base_dir),
        "candidate_artifact_dir": str(candidate_dir),
        "dataset_identity": _dataset_identity(base_dataset, candidate_dataset),
        "metrics": _metric_comparison(base_analysis, candidate_analysis),
        "fills": _fill_comparison(base_fills, candidate_fills),
        "strategy_equity": _equity_comparison(base_equity, candidate_equity),
        "interpretation": (
            "Evidence only: this comparison does not declare a migration safe or unsafe. "
            "Review risk policy, exposure duration, and fill-type changes."
        ),
    }


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.3f}%"


def format_human_summary(report: dict) -> str:
    fills = report["fills"]
    equity = report["strategy_equity"]
    lines = [
        f"Dataset identity: {report['dataset_identity']['status']}",
        (
            "Fills: "
            f"{fills['base_count']} -> {fills['candidate_count']} "
            f"({fills['count_delta']:+d})"
        ),
        f"Fill-event match: {fills['event_multiset_match_ratio'] * 100:.2f}%",
        f"Exact fill-row match: {fills['exact_multiset_match_ratio'] * 100:.2f}%",
        f"Final strategy equity delta: {_format_pct(equity['final_pct_delta'])}",
        f"Equity RMSE vs initial balance: {equity['rmse_pct_of_initial'] * 100:.3f}%",
    ]
    for key in (
        "adg_strategy_eq",
        "drawdown_worst_strategy_eq",
        "position_held_days_max",
        "high_exposure_days_max_long",
        "high_exposure_days_max_short",
    ):
        metric = report["metrics"].get(key)
        if metric is None:
            continue
        lines.append(
            f"{key}: {metric['base']:.8g} -> {metric['candidate']:.8g} "
            f"(relative {_format_pct(metric['relative_delta'])})"
        )
    type_deltas = sorted(
        fills["type_count_delta"].items(), key=lambda item: (-abs(item[1]), item[0])
    )
    if type_deltas:
        lines.append("Largest fill-type count changes:")
        lines.extend(f"  {name}: {delta:+d}" for name, delta in type_deltas[:10])
    lines.append(report["interpretation"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool compare-backtests",
        description="Compare two completed Passivbot backtest artifact directories.",
    )
    parser.add_argument("base_artifact_dir", type=Path)
    parser.add_argument("candidate_artifact_dir", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of text."
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to write the JSON report."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_backtest_artifacts(
            args.base_artifact_dir, args.candidate_artifact_dir
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        print(f"compare-backtests: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload if args.json else format_human_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
