import argparse
from collections import deque
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MS_PER_DAY = 86_400_000


def load_inputs(backtest_dir: Path):
    balance_path = backtest_dir / "balance_and_equity.csv.gz"
    fills_path = backtest_dir / "fills.csv"
    config_path = backtest_dir / "config.json"
    if not balance_path.exists():
        raise FileNotFoundError(balance_path)
    if not fills_path.exists():
        raise FileNotFoundError(fills_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    balance = pd.read_csv(balance_path)
    fills = pd.read_csv(fills_path)
    config = json.loads(config_path.read_text())
    return balance, fills, config


def prepare_series(balance: pd.DataFrame, fills: pd.DataFrame, config: dict) -> pd.DataFrame:
    balance = balance.copy()
    fills = fills.copy()

    balance["timestamp"] = pd.to_datetime(balance["Unnamed: 0"], utc=True)
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    balance = balance.sort_values("timestamp").reset_index(drop=True)
    fills = fills.sort_values("timestamp").reset_index(drop=True)

    fills["realized_pnl_net"] = fills["pnl"] + fills["fee_paid"]
    fills["realized_pnl_net_cumsum"] = fills["realized_pnl_net"].cumsum()
    realized = fills[["timestamp", "realized_pnl_net_cumsum"]].copy()

    df = pd.merge_asof(
        balance,
        realized,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    df["realized_pnl_net_cumsum"] = df["realized_pnl_net_cumsum"].fillna(0.0)

    df["equity_usd"] = df["usd_total_equity"]
    df["balance_usd"] = df["usd_total_balance"]
    df["unrealized_pnl"] = df["equity_usd"] - df["balance_usd"]
    df["strategy_pnl"] = df["realized_pnl_net_cumsum"] + df["unrealized_pnl"]

    starting_balance = float(config["backtest"]["starting_balance"])
    lookback_days = float(config["live"]["pnls_max_lookback_days"])
    lookback_ms = max(1, int(round(lookback_days * MS_PER_DAY)))

    df["strategy_equity_start_rebased"] = starting_balance + df["strategy_pnl"]
    df["strategy_equity_current_rebased"] = (
        (df["balance_usd"] - df["realized_pnl_net_cumsum"]) + df["strategy_pnl"]
    )
    df["baseline_balance"] = df["balance_usd"] - df["realized_pnl_net_cumsum"]

    rolling_peak_strategy_pnl = []
    q = deque()
    for ts, strategy_pnl in zip(df["timestamp"], df["strategy_pnl"]):
        ts_ms = int(ts.value // 1_000_000)
        while q and ts_ms - q[0][0] > lookback_ms:
            q.popleft()
        while q and q[-1][1] <= strategy_pnl:
            q.pop()
        q.append((ts_ms, strategy_pnl))
        rolling_peak_strategy_pnl.append(q[0][1])
    df["rolling_peak_strategy_pnl"] = rolling_peak_strategy_pnl
    df["peak_strategy_equity_hsl"] = (
        df["baseline_balance"] + df["rolling_peak_strategy_pnl"]
    ).clip(lower=df["equity_usd"])

    df["drawdown_usd_equity"] = 1.0 - df["equity_usd"] / df["equity_usd"].cummax().clip(lower=1e-12)
    df["drawdown_hsl_style"] = 1.0 - df["equity_usd"] / df["peak_strategy_equity_hsl"].clip(lower=1e-12)
    df["current_rebased_minus_equity"] = (
        df["strategy_equity_current_rebased"] - df["equity_usd"]
    )
    return df


def save_plot_equity_curves(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["timestamp"], df["equity_usd"], label="Actual equity (USD)", linewidth=2)
    ax.plot(df["timestamp"], df["balance_usd"], label="Actual balance (USD)", alpha=0.8)
    ax.plot(
        df["timestamp"],
        df["strategy_equity_start_rebased"],
        label="Starting-balance rebased strategy equity",
        linewidth=2,
    )
    ax.plot(
        df["timestamp"],
        df["strategy_equity_current_rebased"],
        label="Current-balance rebased strategy equity",
        linestyle="--",
        alpha=0.9,
    )
    ax.set_title("Equity Curves: Actual vs Rebased Strategy Equity")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_equity_curves.png", dpi=150)
    plt.close(fig)


def save_plot_pnl_decomposition(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["timestamp"], df["realized_pnl_net_cumsum"], label="Realized pnl net cumsum", linewidth=2)
    ax.plot(df["timestamp"], df["unrealized_pnl"], label="Unrealized pnl", linewidth=2)
    ax.plot(df["timestamp"], df["strategy_pnl"], label="Strategy pnl", linewidth=2.2)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Strategy PnL Decomposition")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_pnl_decomposition.png", dpi=150)
    plt.close(fig)


def save_plot_hsl_denominator(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["timestamp"], df["equity_usd"], label="Current equity", linewidth=2)
    ax.plot(df["timestamp"], df["peak_strategy_equity_hsl"], label="HSL rebased peak strategy equity", linewidth=2)
    ax.plot(df["timestamp"], df["baseline_balance"], label="Baseline balance = balance - realized pnl", alpha=0.8)
    ax.set_title("HSL Drawdown Denominator Construction")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_equity_hsl_denominator.png", dpi=150)
    plt.close(fig)


def save_plot_drawdowns(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["timestamp"], df["drawdown_usd_equity"], label="Raw USD equity drawdown", linewidth=2)
    ax.plot(df["timestamp"], df["drawdown_hsl_style"], label="HSL-style drawdown", linewidth=2)
    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_equity_drawdown_comparison.png", dpi=150)
    plt.close(fig)


def save_plot_collateral_fx_context(df: pd.DataFrame, out_dir: Path):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df["timestamp"], df["equity_usd"], label="Actual equity (USD)", linewidth=2)
    ax1.plot(
        df["timestamp"],
        df["strategy_equity_start_rebased"],
        label="Starting-balance rebased strategy equity",
        linewidth=2,
    )
    ax1.set_ylabel("USD")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    btc_price = (df["usd_total_balance"] / df["btc_total_balance"]).replace([pd.NA], float("nan"))
    ax2.plot(
        df["timestamp"],
        btc_price,
        label="Implied BTC price from balance series",
        color="black",
        alpha=0.45,
        linewidth=1.5,
    )
    ax2.set_ylabel("BTC/USD")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Collateral FX Context: Equity Curves vs BTC Price")
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_equity_collateral_fx_context.png", dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, config: dict, out_dir: Path):
    lines = [
        "Strategy equity comparison summary",
        "",
        f"starting_balance: {float(config['backtest']['starting_balance']):.10f}",
        f"btc_collateral_cap: {float(config['backtest'].get('btc_collateral_cap', 0.0) or 0.0):.10f}",
        f"pnls_max_lookback_days: {float(config['live']['pnls_max_lookback_days']):.10f}",
        "",
        "Series definitions:",
        "  equity_usd = actual account equity in USD terms",
        "  balance_usd = actual account balance in USD terms",
        "  realized_pnl_net_cumsum = cumulative (pnl + fee_paid) from fills.csv",
        "  unrealized_pnl = equity_usd - balance_usd",
        "  strategy_pnl = realized_pnl_net_cumsum + unrealized_pnl",
        "  strategy_equity_start_rebased = starting_balance + strategy_pnl",
        "  strategy_equity_current_rebased = (balance_usd - realized_pnl_net_cumsum) + strategy_pnl",
        "  peak_strategy_equity_hsl = max(equity_usd, (balance_usd - realized_pnl_net_cumsum) + rolling_peak_strategy_pnl)",
        "",
        "Key identity:",
        "  strategy_equity_current_rebased = equity_usd",
        f"  max_abs(strategy_equity_current_rebased - equity_usd) = {df['current_rebased_minus_equity'].abs().max():.16f}",
        "",
        "Interpretation:",
        "  The current-balance rebased construction collapses to actual equity.",
        "  That means HSL-style rebasing is meaningful for the drawdown denominator,",
        "  but it does not create a distinct growth-series family for ADG/MDG.",
        "  The only distinct collateral-agnostic growth curve here is the starting-balance-rebased one.",
    ]
    (out_dir / "strategy_equity_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backtest_dir")
    args = parser.parse_args()

    backtest_dir = Path(args.backtest_dir)
    if not backtest_dir.exists():
        raise FileNotFoundError(backtest_dir)

    balance, fills, config = load_inputs(backtest_dir)
    df = prepare_series(balance, fills, config)
    save_plot_equity_curves(df, backtest_dir)
    save_plot_pnl_decomposition(df, backtest_dir)
    save_plot_hsl_denominator(df, backtest_dir)
    save_plot_drawdowns(df, backtest_dir)
    save_plot_collateral_fx_context(df, backtest_dir)
    write_summary(df, config, backtest_dir)
    print(f"Wrote strategy-equity comparison plots to {backtest_dir}")


if __name__ == "__main__":
    main()
