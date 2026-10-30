"""Views of saved native backtest facts; no strategy or HSL replay lives here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from backtest_results import ResultStore, extrema_indices, write_json

PLOTS = {"equity", "hsl", "hsl_episodes", "coin_fills", "twe"}


@dataclass(frozen=True)
class ReportOptions:
    plots: tuple[str, ...] = ("equity", "hsl")
    equity_scale: str = "log"
    btc: bool = False
    equity_csv: bool = False
    scope: str | None = None

    def __post_init__(self):
        if set(self.plots) - PLOTS or self.equity_scale not in {"log", "linear"}:
            raise ValueError("Invalid backtest report options")


def parse_plots(text):
    if text == "all":
        return tuple(sorted(PLOTS))
    if text == "none":
        return ()
    values = tuple(
        dict.fromkeys(part.strip() for part in text.split(",") if part.strip())
    )
    if not values or set(values) - PLOTS:
        raise ValueError(
            "Plots must be equity,hsl,hsl_episodes,coin_fills,twe,all or none"
        )
    return values


def add_report_arguments(parser):
    parser.add_argument(
        "--plots",
        type=parse_plots,
        default="equity,hsl",
        help="Plot groups: equity,hsl,hsl_episodes,coin_fills,twe,all,none.",
    )
    parser.add_argument("--equity-scale", choices=("log", "linear"), default="log")
    parser.add_argument(
        "--report-btc",
        action="store_true",
        help="Include optional BTC-denominated equity view.",
    )
    parser.add_argument(
        "--equity-csv",
        action="store_true",
        help="Also export full-resolution equity as gzip CSV.",
    )


def options_from_args(args):
    return ReportOptions(
        args.plots, args.equity_scale, args.report_btc, args.equity_csv
    )


def scope_name(report, scope):
    side, coin = scope["side"], scope["coin"]
    if side is None:
        return "portfolio"
    return (f"{report['coins'][coin]}:" if coin is not None else "") + (
        "long",
        "short",
    )[side]


def scope_trace(report, trace, scope):
    if trace is None:
        return None
    mask = (trace["side"] == (-1 if scope["side"] is None else scope["side"])) & (
        trace["coin"] == (-1 if scope["coin"] is None else scope["coin"])
    )
    return {
        key: values[mask] for key, values in trace.items() if key != "schema_version"
    }


def strategy_drawdown(equity):
    values = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(values) if len(values) else values
    # Equity has a positive initial balance. Preserve any later nonpositive
    # terminal observation instead of hiding it behind a log plot.
    return np.divide(
        peak - values, peak, out=np.full_like(values, np.nan), where=peak > 0
    )


def equity_view(
    store: ResultStore, *, max_points=2000, start=None, end=None, btc=False
):
    columns = ["strategy_equity", "usd_total_balance"]
    if btc:
        columns += ["btc_total_equity", "btc_total_balance"]
    # Derive drawdown before selecting a window; the prior peak is part of the
    # quantity even when the viewer is zoomed into a later interval.
    frame = store.equity(columns=columns)
    timestamps = frame.index.as_unit("ms").asi8
    drawdown = strategy_drawdown(frame.strategy_equity.to_numpy())
    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps <= end
    frame = frame.loc[mask]
    timestamps, drawdown = timestamps[mask], drawdown[mask]
    values = np.column_stack([frame.to_numpy(), drawdown])
    selected = extrema_indices(values, max_points=max_points)
    return {
        "timestamp": timestamps[selected],
        **{key: frame[key].to_numpy()[selected] for key in columns},
        "drawdown_fraction": drawdown[selected],
        "observations": len(frame),
    }


def create_equity_overview(store, options):
    import matplotlib.pyplot as plt

    data = equity_view(store, btc=options.btc)
    panels = 3 if options.btc else 2
    fig, axes = plt.subplots(
        panels, 1, sharex=True, figsize=(14, 8 if panels == 2 else 11)
    )
    x = pd.to_datetime(data["timestamp"], unit="ms")
    ax = axes[0]
    ax.plot(x, data["strategy_equity"], label="Strategy equity")
    balance = data["usd_total_balance"]
    if not np.array_equal(balance, data["strategy_equity"], equal_nan=True):
        ax.plot(x, balance, label="Account balance", alpha=0.7)
    scale = options.equity_scale
    plotted = np.concatenate([data["strategy_equity"], balance])
    if scale == "log" and np.any(plotted[np.isfinite(plotted)] <= 0):
        scale = "linear"
        ax.set_title("Strategy equity — linear scale includes nonpositive observations")
    else:
        ax.set_title(f"Strategy equity — {scale} scale")
    ax.set_yscale(scale)
    ax.set_ylabel("USD")
    ax.legend(loc="upper left")
    axes[1].fill_between(
        x, 0, data["drawdown_fraction"] * 100, alpha=0.25, color="tab:red"
    )
    axes[1].plot(x, data["drawdown_fraction"] * 100, color="tab:red", linewidth=1)
    maximum = float(np.nanmax(data["drawdown_fraction"])) if len(x) else None
    axes[1].set_title(
        "Strategy equity drawdown"
        + (f" — maximum {maximum:.3%}" if maximum is not None else "")
    )
    axes[1].set_ylabel("Drawdown (%)")
    if options.btc:
        axes[2].plot(x, data["btc_total_equity"], label="Equity in BTC")
        axes[2].plot(x, data["btc_total_balance"], label="Balance in BTC", alpha=0.7)
        if (
            scale == "log"
            and np.all(data["btc_total_equity"] > 0)
            and np.all(
                data["btc_total_balance"][np.isfinite(data["btc_total_balance"])] > 0
            )
        ):
            axes[2].set_yscale("log")
        axes[2].set_ylabel("BTC")
        axes[2].legend(loc="upper left")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (UTC)")
    fig.tight_layout()
    return fig


def _gap_lines(timestamps, values, gap_before):
    # Missing minutes in this native trace are explicit idle gaps. Do not draw
    # an interpolated signal across two different exposure episodes.
    x = np.asarray(timestamps, dtype=np.int64)
    y = np.asarray(values, dtype=np.float64)
    gaps = np.flatnonzero(gap_before)
    # Keep the UTC clock integer-valued. A repeated factual clock paired with
    # a display-only NaN breaks the path without pandas float-clock rounding.
    return np.insert(x, gaps, x[gaps]), np.insert(y, gaps, np.nan)


def hsl_view(store, *, scope=None, max_points=2000, start=None, end=None):
    report, trace = store.hsl(), store.hsl_trace()
    if report is None:
        return None
    groups = []
    for policy in report["scopes"]:
        name = scope_name(report, policy)
        if scope is not None and scope != name:
            continue
        rows = scope_trace(report, trace, policy)
        if rows is None:
            # A released detailed JSON report is an explicit factual source.
            legacy = [
                r
                for r in report.get("samples", [])
                if r["side"] == policy["side"]
                and r["coin"] == policy["coin"]
                and r["action"] is not None
            ]
            if legacy:
                rows = {
                    key: np.asarray([r[key] for r in legacy])
                    for key in (
                        "timestamp",
                        "sequence",
                        "signal_raw_fraction",
                        "signal_ema_fraction",
                    )
                }
                rows["action"] = np.asarray(
                    [
                        {"normal": 0, "panic": 1, "halted": 2}[r["action"]]
                        for r in legacy
                    ]
                )
        if rows is None or not len(rows["timestamp"]):
            continue
        mask = np.ones(len(rows["timestamp"]), dtype=bool)
        if start is not None:
            mask &= rows["timestamp"] >= start
        if end is not None:
            mask &= rows["timestamp"] <= end
        rows = {key: value[mask] for key, value in rows.items()}
        n = len(rows["timestamp"])
        if not n:
            continue
        raw, ema = rows["signal_raw_fraction"], rows["signal_ema_fraction"]
        trigger = rows.get("trigger_signal_fraction", np.minimum(raw, ema))
        changes = np.flatnonzero(np.diff(rows["action"]) != 0) + 1
        gaps = np.flatnonzero(np.diff(rows["timestamp"]) > 60_000) + 1
        forced = np.concatenate([changes, changes - 1, gaps, gaps - 1])
        chosen = extrema_indices(
            np.column_stack([raw, ema, trigger]), max_points=max_points, forced=forced
        )
        gap_before = np.concatenate([[False], np.diff(rows["timestamp"]) > 60_000])
        states = [
            (
                int(rows["sequence"][i]),
                int(rows["timestamp"][i]),
                int(rows["action"][i]),
            )
            for i in chosen
        ]
        for event in report["events"]:
            if event["side"] != policy["side"] or event["coin"] != policy["coin"]:
                continue
            if (
                start is not None
                and event["observed_at"] < start
                or end is not None
                and event["observed_at"] > end
            ):
                continue
            states.append(
                (
                    event["sequence"],
                    event["observed_at"],
                    {"red": 1, "flat": 2, "restart": 0}[event["kind"]],
                )
            )
        states.sort(key=lambda row: row[0])
        groups.append(
            {
                "name": name,
                "policy": policy["policy"],
                "gap_before": gap_before[chosen],
                "state_timestamp": np.array([row[1] for row in states], dtype=np.int64),
                "state_action": np.array([row[2] for row in states], dtype=np.uint8),
                "observations": n,
                **{key: value[chosen] for key, value in rows.items()},
            }
        )
    if scope is not None and scope not in {
        scope_name(report, policy) for policy in report["scopes"]
    }:
        raise ValueError("Unknown HSL controller scope")
    return {
        "report": report,
        "scopes": groups,
        "trace_available": trace is not None or bool(report.get("samples")),
    }


def create_hsl_overview(store, options, *, start=None, end=None, title=None):
    import matplotlib.pyplot as plt

    view = hsl_view(store, scope=options.scope, start=start, end=end)
    if view is None or not view["report"]["scopes"]:
        return None
    fig, (signal, state) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    groups = view["scopes"]
    if not groups:
        message = (
            "No exposed HSL observations"
            if view["trace_available"]
            else "Native HSL trace unavailable in this legacy result"
        )
        signal.text(
            0.5, 0.5, message, transform=signal.transAxes, ha="center", va="center"
        )
    for group in groups:
        p = group["policy"]
        ts = group["timestamp"]
        if len(groups) == 1:
            for key, label in [
                ("signal_raw_fraction", "Raw drawdown"),
                ("signal_ema_fraction", "Drawdown EMA"),
                ("trigger_signal_fraction", "Native trigger signal"),
            ]:
                if key not in group:
                    continue
                xx, yy = _gap_lines(ts, group[key] * 100, group["gap_before"])
                signal.plot(pd.to_datetime(xx, unit="ms"), yy, label=label, linewidth=1)
            signal.axhline(
                p["drawdown_threshold_fraction"] * 100,
                linestyle="--",
                color="firebrick",
                label="RED threshold",
            )
            signal.set_ylabel("Drawdown signal (%)")
            signal.set_title(
                title
                or f"HSL {group['name']} — memory {p['memory_ema_span_minutes']:g} min, signal EMA {p['drawdown_fraction_ema_span_minutes']:g} min"
            )
        else:
            gap = (
                p["drawdown_threshold_fraction"]
                - group.get(
                    "trigger_signal_fraction",
                    np.minimum(
                        group["signal_raw_fraction"], group["signal_ema_fraction"]
                    ),
                )
            ) * 100
            xx, yy = _gap_lines(ts, gap, group["gap_before"])
            signal.plot(
                pd.to_datetime(xx, unit="ms"), yy, label=group["name"], linewidth=1
            )
            signal.set_ylabel("Distance below RED (percentage points)")
            signal.set_title(
                title or "HSL active controllers — negative values exceed the threshold"
            )
        state.step(
            pd.to_datetime(group["state_timestamp"], unit="ms"),
            group["state_action"],
            where="post",
            label=group["name"],
        )
    if len(groups) > 1:
        signal.axhline(0, linestyle="--", color="firebrick", linewidth=1)
    for event in view["report"]["events"]:
        if (
            start is not None
            and event["observed_at"] < start
            or end is not None
            and event["observed_at"] > end
        ):
            continue
        if options.scope and scope_name(view["report"], event) != options.scope:
            continue
        if event["kind"] == "red":
            state.axvline(
                pd.Timestamp(event["observed_at"], unit="ms"),
                linestyle=":",
                color="firebrick",
                linewidth=0.7,
            )
    state.set_yticks([0, 1, 2], ["Normal", "Panic", "Cooldown"])
    state.set_ylim(-0.2, 2.2)
    state.set_xlabel("Time (UTC)")
    for ax in (signal, state):
        ax.grid(True, alpha=0.25)
        if groups and len(groups) <= 6:
            ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def episode_frame(report):
    columns = [
        "scope",
        "trigger_time_utc",
        "first_fill_time_utc",
        "finished_time_utc",
        "fill_count",
        "loss_usd",
        "account_balance_usd",
        "loss_fraction",
        "unfinished",
    ]
    records = []
    for episode in report.get("execution_episodes", []):
        record = {
            key: episode.get(key)
            for key in columns
            if key not in {"scope", "unfinished"}
        }
        record["scope"] = scope_name(report, episode)
        record["unfinished"] = episode["finished_timestamp_ms"] is None
        for stem in ("trigger", "first_fill", "finished"):
            value = episode[f"{stem}_timestamp_ms"]
            record[f"{stem}_time_utc"] = (
                None
                if value is None
                else pd.Timestamp(value, unit="ms", tz="UTC").isoformat()
            )
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def render_reports(store, directory=None, *, options=None, disabled=()):
    import matplotlib.pyplot as plt

    options = options or ReportOptions()
    directory = (
        Path(directory) if directory is not None else store.directory / "reports"
    )
    directory.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": 1,
        "status": "rendering",
        "options": asdict(options),
        "artifacts": [],
        "timings_seconds": {},
    }
    write_json(directory / "report.json", status)
    aliases = {"equity": "balance", "hsl": "hard_stop", "hsl_episodes": "hard_stop"}

    def save(name, figure):
        if figure is None:
            return
        try:
            from session_artifacts import safe_component

            filename = safe_component(name) + ".png"
            figure.savefig(directory / filename)
            status["artifacts"].append(filename)
        finally:
            plt.close(figure)

    try:
        for group in options.plots:
            if aliases.get(group, group) in disabled:
                continue
            start = time.perf_counter()
            logging.info("[report] rendering %s", group)
            if group == "equity":
                save("equity", create_equity_overview(store, options))
            elif group == "hsl":
                save("hsl", create_hsl_overview(store, options))
            elif group == "hsl_episodes":
                report = store.hsl()
                if report is not None:
                    for i, episode in enumerate(report.get("execution_episodes", [])):
                        name = scope_name(report, episode)
                        selected = ReportOptions(
                            ("hsl",), options.equity_scale, options.btc, scope=name
                        )
                        first = (
                            episode["trigger_timestamp_ms"]
                            if episode["trigger_timestamp_ms"] is not None
                            else episode["first_fill_timestamp_ms"]
                        )
                        if first is None:
                            continue
                        last = episode["finished_timestamp_ms"]
                        save(
                            f"hsl_episode_{i + 1}",
                            create_hsl_overview(
                                store,
                                selected,
                                start=first - 24 * 3_600_000,
                                end=None if last is None else last + 6 * 3_600_000,
                                title=f"HSL execution episode {i + 1} — {name}",
                            ),
                        )
            elif group == "twe":
                from plotting import create_forager_twe_figure

                for name, fig in create_forager_twe_figure(
                    store.fills(), autoplot=False, return_figures=True
                ).items():
                    fig.axes[0].set_title(
                        "Fill-observed wallet exposure — hourly fill-event mean"
                    )
                    save(name, fig)
            elif group == "coin_fills":
                from backtest_artifacts import load_backtest_artifact
                from plotting import create_forager_coin_figures

                artifact = load_backtest_artifact(store.directory)
                create_forager_coin_figures(
                    artifact.coins,
                    artifact.fills,
                    artifact.hlcvs,
                    timestamps=artifact.timestamps,
                    on_figure=save,
                    close_after_callback=False,
                )
            status["timings_seconds"][group] = time.perf_counter() - start
        report = store.hsl()
        if report is not None and "execution_episodes" in report:
            episode_frame(report).to_csv(directory / "hsl_episodes.csv", index=False)
            status["artifacts"].append("hsl_episodes.csv")
        if options.equity_csv:
            start = time.perf_counter()
            store.equity().to_csv(
                directory / "balance_and_equity.csv.gz",
                compression={"method": "gzip", "compresslevel": 1},
            )
            status["artifacts"].append("balance_and_equity.csv.gz")
            status["timings_seconds"]["equity_csv"] = time.perf_counter() - start
        status["status"] = "complete"
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        raise
    finally:
        write_json(directory / "report.json", status)
    return status
