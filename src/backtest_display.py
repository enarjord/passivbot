"""Terminal-only projection of canonical backtest metrics and support facts."""

from collections.abc import Iterable, Mapping
from decimal import Decimal
import math

from config.metrics import CORE_METRIC_CATALOG


_SIDE_METRICS = {
    "long": {"position_exposure_utilization_rms_ratio_long"},
    "short": {
        "position_exposure_utilization_rms_ratio_short",
        "short_net_profit_contribution_fraction",
    },
}
_HSL_METRICS = {name for name in CORE_METRIC_CATALOG if name.startswith("hsl_")}
_STATUS_LABELS = {
    "positive_infinity": "+inf",
    "negative_infinity": "-inf",
    "undefined": "undefined",
    "invalid": "invalid",
}


def _disabled_zero_metrics(analysis, config, metadata, fill_types):
    """Hide only explicit inactivity; unknown or contrary observations stay visible."""
    hidden = set()
    bot = config.get("bot", {})
    for side, names in _SIDE_METRICS.items():
        if (
            bot.get(side, {}).get("enabled") is False
            and not any(kind.endswith("_" + side) for kind in fill_types)
            and all(analysis[name] == 0 for name in names if name in analysis)
        ):
            hidden.update(names)
    if (
        metadata.get("hsl_enabled") is False
        and not any(kind.startswith("close_panic_") for kind in fill_types)
        and all(analysis[name] == 0 for name in _HSL_METRICS if name in analysis)
        and all(metadata.get(name, 0) == 0 for name in (
            "hsl_trigger_count", "hsl_execution_episode_count"
        ))
    ):
        hidden.update(_HSL_METRICS)
    return hidden


def _metric_value(name, value, status):
    if value is None:
        tag = status.get("status", "unavailable")
        reason = status.get("reason") or ""
        return _STATUS_LABELS.get(tag, tag), str(reason).replace("_", " ")
    if CORE_METRIC_CATALOG[name]["unit"] == "fraction":
        percentage = value * 100
        if not math.isfinite(percentage):
            # Scaling a genuine finite endpoint must not display a fake infinity.
            percentage = Decimal(str(value)).scaleb(2)
        return f"{percentage:.5g}%", ""
    return f"{value:.5g}", ""


def format_backtest_metrics(
    analysis: Mapping,
    config: Mapping,
    *,
    fill_types: Iterable[str] = (),
) -> str:
    """Format five significant digits without changing values, requests or artifacts.

    Applicability comes from explicit side permission and native HSL metadata.
    Actual fills and nonzero/unavailable metrics prevent a family being hidden.
    """
    metadata = analysis.get("metric_metadata", {})
    statuses = analysis.get("metric_status", {})
    fill_types = set(fill_types)
    hidden = _disabled_zero_metrics(analysis, config, metadata, fill_types)
    rows = []
    for name in CORE_METRIC_CATALOG:
        if name not in analysis or name in hidden:
            continue
        value, note = _metric_value(name, analysis[name], statuses.get(name, {}))
        if name == "peak_recovery_max_days_strategy_equity" and metadata.get(
            "peak_recovery_tail_censored"
        ) is True:
            note = "; ".join(filter(None, (note, "unfinished tail")))
        if name == "cagr_fraction_strategy_equity_ew" and "ew_half_life_days" in metadata:
            half_life = metadata["ew_half_life_days"]
            note = "; ".join(filter(None, (note, f"half-life {half_life:.5g} days")))
        rows.append((name, value, note))

    lines = []
    if "effective_start_date" in analysis and "effective_end_date" in analysis:
        lines.append(f"Period: {analysis['effective_start_date']} to {analysis['effective_end_date']}")
    if "liquidated" in analysis:
        lines.append("Liquidated: " + ("yes" if analysis["liquidated"] else "no"))
    if rows:
        name_width = max(len("Metric"), *(len(name) for name, _, _ in rows))
        value_width = max(len("Value"), *(len(value) for _, value, _ in rows))
        show_notes = any(note for _, _, note in rows)
        lines.extend((
            f"{'Metric':<{name_width}}  {'Value':>{value_width}}" + ("  Note" if show_notes else ""),
            f"{'-' * name_width}  {'-' * value_width}" + ("  ----" if show_notes else ""),
        ))
        lines.extend(
            f"{name:<{name_width}}  {value:>{value_width}}" + (f"  {note}" if note else "")
            for name, value, note in rows
        )
    support = [
        f"{metadata[key]:,} {label}"
        for key, label in (
            ("daily_return_count", "daily returns"),
            ("equity_observation_count", "equity observations"),
            ("fill_count", "fills"),
            ("holding_episode_count", "holding episodes"),
        ) if key in metadata
    ]
    if support:
        lines.append("Support: " + ", ".join(support))
    if metadata.get("hsl_enabled") is True or _HSL_METRICS.intersection(analysis) - hidden:
        hsl = [
            f"{metadata[key]:,} {label}"
            for key, label in (
                ("hsl_trigger_count", "triggers"),
                ("hsl_execution_episode_count", "execution episodes"),
            ) if key in metadata
        ]
        if hsl:
            lines.append("HSL: " + ", ".join(hsl))
    return "\n".join(lines)
