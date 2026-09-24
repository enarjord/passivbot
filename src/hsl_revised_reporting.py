"""Observational revised HSL artifacts and plots, using native results only."""
import logging


def revised_report(plot_data):
    """Return the native envelope without deriving a replacement risk signal."""
    if plot_data is None or "revised" not in plot_data:
        return None
    report = plot_data["revised"]
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("engine") != "revised"
            or report.get("mode") not in {"coin", "pside", "unified"}):
        raise ValueError("unsupported revised HSL report schema")
    return report


def create_revised_hsl_figures(report, *, figsize, autoplot, return_figures, display=None):
    import matplotlib.pyplot as plt
    import pandas as pd

    if report is None:
        logging.warning("Revised HSL plots unavailable: native report was not supplied")
        return {}
    if not report["detailed"]:
        logging.info("Revised HSL drawdown plots omitted: enable backtest.hsl_detailed_report for per-minute samples (unavailable in metrics-only runs)")
        return {}

    samples = {}
    events = {}
    for row in report["samples"]:
        samples.setdefault((row["side"], row["coin"]), []).append(row)
    for row in report["events"]:
        events.setdefault((row["side"], row["coin"]), []).append(row)

    figures = {}
    for scope in report["scopes"]:
        side, coin = scope["side"], scope["coin"]
        rows = samples.get((side, coin), [])
        if not rows or not any(row["action"] is not None for row in rows):
            continue
        side_name = ("long", "short")[side] if side is not None else None
        if report["mode"] == "coin":
            name = f"{report['coins'][coin]} {side_name}"
            key = f"hard_stop_drawdown_coin_{coin}_{side_name}"
        elif report["mode"] == "pside":
            name, key = side_name, f"hard_stop_drawdown_{side_name}"
        else:
            name, key = "portfolio", "hard_stop_drawdown_unified"
        policy = scope["policy"]
        frame = pd.DataFrame(rows)
        x = pd.to_datetime(frame["timestamp"], unit="ms")
        fig, (signal_ax, state_ax) = plt.subplots(
            2, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
        signal_ax.plot(x, frame["raw"].astype(float), label="Raw drawdown", alpha=.6)
        signal_ax.plot(x, frame["ema"].astype(float), label="EMA drawdown")
        signal_ax.axhline(policy["red_threshold"], color="firebrick", linestyle="--",
                         label="RED threshold")
        signal_ax.set_title(f"Revised HSL: {name} (EMA span {policy['ema_span_minutes']:g} min)")
        signal_ax.set_ylabel("Drawdown")
        # Preserve native same-timestamp observation order, including fill-boundary transitions.
        states = frame["action"].map({"normal": 0, "panic": 1, "halted": 1})
        if (frame["action"].notna() & states.isna()).any():
            plt.close(fig)
            raise ValueError("unsupported revised HSL report action")
        transitions = [(row["sequence"], row["timestamp"], state)
                       for row, state in zip(rows, states)]
        for event in events.get((side, coin), []):
            transitions.append((event["sequence"], event["observed_at"],
                                {"red": 1, "flat": 1, "restart": 0}[event["kind"]]))
        transitions.sort(key=lambda row: row[0])
        state_ax.step(pd.to_datetime([row[1] for row in transitions], unit="ms"),
                      [row[2] for row in transitions], where="post", color="firebrick",
                      label="Controller state")
        state_ax.set_yticks([0, 1], ["GREEN", "RED"])
        state_ax.set_ylim(-.15, 1.15)
        seen = set()
        for event in events.get((side, coin), []):
            kind = event["kind"]
            color = {"red": "firebrick", "flat": "slategray", "restart": "seagreen"}[kind]
            state_ax.axvline(pd.to_datetime(event["observed_at"], unit="ms"), color=color,
                             linestyle=":", label=kind if kind not in seen else None)
            seen.add(kind)
        for ax in (signal_ax, state_ax):
            ax.grid(True, alpha=.25)
            ax.legend(loc="upper left")
        state_ax.set_xlabel("Time (UTC)")
        fig.tight_layout()
        if return_figures:
            figures[key] = fig
        if autoplot:
            if display is not None:
                display(fig)
            else:
                fig.show()
        if not return_figures:
            plt.close(fig)
    return figures
