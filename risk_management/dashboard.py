"""Terminal dashboard for monitoring trading portfolios.

The module consumes a JSON snapshot describing one or more accounts along with
alert thresholds.  It renders a textual dashboard summarising exposure, profit
and loss, and risk metrics while also highlighting any triggered alerts.

The command line interface can either read a static snapshot file or, when
configured with realtime credentials, fetch fresh account information from the
supported exchanges on a configurable interval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .configuration import CustomEndpointSettings, load_realtime_config


@dataclass
class Position:
    """A lightweight representation of a trading position."""

    symbol: str
    side: str
    notional: float
    entry_price: float
    mark_price: float
    liquidation_price: float | None
    wallet_exposure_pct: float | None
    unrealized_pnl: float
    max_drawdown_pct: float | None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    size: float | None = None
    signed_notional: float | None = None

    def exposure_relative_to(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return abs(self.notional) / balance

    def pnl_pct(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return self.unrealized_pnl / balance


@dataclass
class Order:
    """Representation of an open order."""

    symbol: str
    side: str
    order_type: str
    price: float | None
    amount: float | None
    remaining: float | None
    status: str
    reduce_only: bool
    stop_price: float | None = None
    notional: float | None = None
    order_id: str | None = None
    created_at: str | None = None


@dataclass
class Account:
    """Account level snapshot."""

    name: str
    balance: float
    positions: Sequence[Position]
    orders: Sequence[Order] = ()

    def total_abs_notional(self) -> float:
        return sum(abs(p.notional) for p in self.positions)

    def total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)

    def exposure_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return self.total_abs_notional() / self.balance

    def net_notional(self) -> float:
        total = 0.0
        for position in self.positions:
            if position.signed_notional is not None:
                total += position.signed_notional
            else:
                total += position.notional if position.side.lower() == "long" else -position.notional
        return total

    def gross_exposure_pct(self) -> float:
        return self.exposure_pct()

    def net_exposure_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return self.net_notional() / self.balance

    def exposures_by_symbol(self) -> Dict[str, Dict[str, float]]:
        exposures: Dict[str, Dict[str, float]] = {}
        for position in self.positions:
            signed = (
                position.signed_notional
                if position.signed_notional is not None
                else position.notional if position.side.lower() == "long" else -position.notional
            )
            data = exposures.setdefault(position.symbol, {"gross": 0.0, "net": 0.0})
            data["gross"] += abs(signed)
            data["net"] += signed
        return exposures


@dataclass
class AlertThresholds:
    wallet_exposure_pct: float = 0.6
    position_wallet_exposure_pct: float = 0.25
    max_drawdown_pct: float = 0.3
    loss_threshold_pct: float = -0.12


logger = logging.getLogger(__name__)


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def _format_price(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:,.2f}"


def _parse_position(raw: Dict[str, Any]) -> Position:
    required = [
        "symbol",
        "side",
        "notional",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
    ]
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"Position missing required fields: {missing}")

    size_raw = raw.get("size")
    signed_notional_raw = raw.get("signed_notional")
    return Position(
        symbol=str(raw["symbol"]),
        side=str(raw.get("side", "")),
        notional=float(raw["notional"]),
        entry_price=float(raw["entry_price"]),
        mark_price=float(raw["mark_price"]),
        liquidation_price=(
            float(raw["liquidation_price"])
            if raw.get("liquidation_price") is not None
            else None
        ),
        wallet_exposure_pct=(
            float(raw["wallet_exposure_pct"])
            if raw.get("wallet_exposure_pct") is not None
            else None
        ),
        unrealized_pnl=float(raw["unrealized_pnl"]),
        max_drawdown_pct=(
            float(raw["max_drawdown_pct"])
            if raw.get("max_drawdown_pct") is not None
            else None
        ),
        take_profit_price=(
            float(raw["take_profit_price"])
            if raw.get("take_profit_price") is not None
            else None
        ),
        stop_loss_price=(
            float(raw["stop_loss_price"])
            if raw.get("stop_loss_price") is not None
            else None
        ),
        size=float(size_raw) if size_raw not in (None, "") else None,
        signed_notional=(
            float(signed_notional_raw)
            if signed_notional_raw not in (None, "")
            else None
        ),
    )


def _parse_account(raw: Dict[str, Any]) -> Account:
    if "name" not in raw or "balance" not in raw:
        raise ValueError("Account entries must include 'name' and 'balance'.")

    positions_raw = raw.get("positions", [])
    positions = [_parse_position(pos) for pos in positions_raw]
    orders_raw = raw.get("open_orders") or raw.get("orders") or []
    orders = [_parse_order(order) for order in orders_raw]
    return Account(
        name=str(raw["name"]),
        balance=float(raw["balance"]),
        positions=positions,
        orders=orders,
    )


def _parse_order(raw: Mapping[str, Any]) -> Order:
    symbol = str(raw.get("symbol", ""))
    side = str(raw.get("side", "")).lower()
    order_type = str(raw.get("type") or raw.get("order_type") or "").lower()
    price = raw.get("price")
    amount = raw.get("amount")
    remaining = raw.get("remaining" if "remaining" in raw else "remaining_amount")
    reduce_only_raw = raw.get("reduce_only") if "reduce_only" in raw else raw.get("reduceOnly")
    stop_price = raw.get("stop_price") if "stop_price" in raw else raw.get("stopPrice")
    notional = raw.get("notional")
    created_at = raw.get("created_at") if "created_at" in raw else raw.get("createdAt")
    order_id = raw.get("order_id") if "order_id" in raw else raw.get("orderId") or raw.get("id")
    return Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        price=float(price) if price not in (None, "") else None,
        amount=float(amount) if amount not in (None, "") else None,
        remaining=float(remaining) if remaining not in (None, "") else None,
        status=str(raw.get("status", "")),
        reduce_only=bool(reduce_only_raw),
        stop_price=float(stop_price) if stop_price not in (None, "") else None,
        notional=float(notional) if notional not in (None, "") else None,
        order_id=str(order_id) if order_id not in (None, "") else None,
        created_at=str(created_at) if created_at not in (None, "") else None,
    )


def _parse_thresholds(raw: Dict[str, Any]) -> AlertThresholds:
    thresholds = AlertThresholds()
    for key in (
        "wallet_exposure_pct",
        "position_wallet_exposure_pct",
        "max_drawdown_pct",
        "loss_threshold_pct",
    ):
        if key in raw:
            setattr(thresholds, key, float(raw[key]))
    return thresholds


def load_snapshot(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def parse_snapshot(data: Dict[str, Any]) -> tuple[datetime, Sequence[Account], AlertThresholds, Sequence[str]]:
    generated_at_raw = data.get("generated_at")
    if generated_at_raw:
        try:
            generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
        except ValueError:
            generated_at = datetime.now(timezone.utc)
    else:
        generated_at = datetime.now(timezone.utc)

    accounts = [_parse_account(acc) for acc in data.get("accounts", [])]
    thresholds = _parse_thresholds(data.get("alert_thresholds", {}))
    notifications = [str(channel) for channel in data.get("notification_channels", [])]
    return generated_at, accounts, thresholds, notifications


def evaluate_alerts(accounts: Sequence[Account], thresholds: AlertThresholds) -> List[str]:
    alerts: List[str] = []
    for account in accounts:
        exposure = account.exposure_pct()
        if exposure > thresholds.wallet_exposure_pct:
            alerts.append(
                f"{account.name}: wallet exposure {exposure:.2%} exceeds limit {thresholds.wallet_exposure_pct:.2%}"
            )

        balance = account.balance
        unrealized_pct = account.total_unrealized() / balance if balance else 0.0
        if unrealized_pct < thresholds.loss_threshold_pct:
            alerts.append(
                f"{account.name}: unrealized PnL {unrealized_pct:.2%} below loss limit {thresholds.loss_threshold_pct:.2%}"
            )

        for position in account.positions:
            pos_exposure = position.exposure_relative_to(balance)
            if pos_exposure > thresholds.position_wallet_exposure_pct:
                alerts.append(
                    f"{account.name} {position.symbol}: exposure {pos_exposure:.2%} exceeds {thresholds.position_wallet_exposure_pct:.2%}"
                )
            if (
                position.max_drawdown_pct is not None
                and position.max_drawdown_pct > thresholds.max_drawdown_pct
            ):
                alerts.append(
                    f"{account.name} {position.symbol}: drawdown {position.max_drawdown_pct:.2%} exceeds {thresholds.max_drawdown_pct:.2%}"
                )
    return alerts


def render_dashboard(
    generated_at: datetime,
    accounts: Sequence[Account],
    alerts: Sequence[str],
    notifications: Sequence[str],
    account_messages: Mapping[str, str] | None = None,
) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("Risk Management Dashboard")
    lines.append(f"Snapshot generated at: {generated_at.astimezone(timezone.utc).isoformat()}")
    lines.append("=" * 80)
    lines.append("")

    account_messages = account_messages or {}

    if not accounts:
        lines.append("No accounts available in the snapshot.")
    for account in accounts:
        lines.append(f"Account: {account.name}")
        lines.append(f"  Balance: {_format_currency(account.balance)}")
        lines.append(f"  Exposure: {_format_pct(account.exposure_pct())}")
        lines.append(f"  Unrealized PnL: {_format_currency(account.total_unrealized())}")
        status_message = account_messages.get(account.name)
        if status_message:
            lines.append(f"  Status: {status_message}")
        lines.append("")
        if account.positions:
            header = (
                f"    {'Symbol':<10}{'Side':<6}{'Exposure':>10}{'PnL':>12}{'Entry':>12}{'Mark':>12}"
                f"{'Liq.':>12}{'Max DD':>10}{'TP':>12}{'SL':>12}"
            )
            lines.append(header)
            lines.append("    " + "-" * (len(header) - 4))
            for position in account.positions:
                balance = account.balance
                pos_exposure = position.exposure_relative_to(balance)
                pnl_pct = position.pnl_pct(balance)
                lines.append(
                    "    "
                    + f"{position.symbol:<10}{position.side:<6}{_format_pct(pos_exposure):>10}"
                    + f"{_format_pct(pnl_pct):>12}{_format_price(position.entry_price):>12}"
                    + f"{_format_price(position.mark_price):>12}{_format_price(position.liquidation_price):>12}"
                    + f"{_format_pct(position.max_drawdown_pct or 0.0):>10}"
                    + f"{_format_price(position.take_profit_price):>12}{_format_price(position.stop_loss_price):>12}"
                )
        else:
            lines.append("    No open positions.")
        lines.append("")

    lines.append("Alerts")
    lines.append("-" * 80)
    if alerts:
        for item in alerts:
            lines.append(f"• {item}")
    else:
        lines.append("No active alerts. All monitored metrics are within thresholds.")
    lines.append("")

    if notifications:
        lines.append("Notification channels")
        lines.append("-" * 80)
        for channel in notifications:
            lines.append(f"• {channel}")
    else:
        lines.append("No notification channels configured.")

    lines.append("=" * 80)
    return "\n".join(lines)


def run_dashboard(config_path: Path) -> str:
    snapshot = load_snapshot(config_path)
    return build_dashboard(snapshot)


def build_dashboard(snapshot: Dict[str, Any]) -> str:
    generated_at, accounts, thresholds, notifications = parse_snapshot(snapshot)
    alerts = evaluate_alerts(accounts, thresholds)
    account_messages = snapshot.get("account_messages", {}) if isinstance(snapshot, Mapping) else {}
    return render_dashboard(generated_at, accounts, alerts, notifications, account_messages=account_messages)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the risk management dashboard")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("dashboard_config.json"),
        help="Path to the JSON snapshot used for the dashboard.",
    )
    parser.add_argument(
        "--realtime-config",
        type=Path,
        default=None,
        help="Path to a realtime configuration file. Overrides --config when provided.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Set to >0 to continuously monitor the file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to render the dashboard when --interval is set. Use 0 to loop indefinitely.",
    )
    parser.add_argument(
        "--custom-endpoints",
        help=(
            "Override custom endpoint behaviour. Provide a JSON file path to reuse the same "
            "proxy configuration as the trading system, 'auto' to enable auto-discovery, or 'none' to "
            "disable overrides."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.interval < 0:
        parser.error("--interval must be >= 0")
    if args.iterations < 0:
        parser.error("--iterations must be >= 0")

    try:
        return asyncio.run(_run_cli(args))
    except FileNotFoundError:
        parser.error(f"Snapshot file not found: {args.config}")
    except json.JSONDecodeError as exc:
        parser.error(f"Snapshot file is not valid JSON: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    return 1


async def _run_cli(args: argparse.Namespace) -> int:
    from .realtime import RealtimeDataFetcher

    realtime_fetcher: RealtimeDataFetcher | None = None
    if args.realtime_config:
        realtime_config = load_realtime_config(Path(args.realtime_config))
        override = args.custom_endpoints
        if override is not None:
            override_normalized = override.strip()
            if not override_normalized:
                realtime_config.custom_endpoints = None
            else:
                lowered = override_normalized.lower()
                if lowered in {"none", "off", "disable"}:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=None, autodiscover=False
                    )
                elif lowered in {"auto", "autodiscover", "default"}:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=None, autodiscover=True
                    )
                else:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=override_normalized, autodiscover=False
                    )
        realtime_fetcher = RealtimeDataFetcher(realtime_config)
        logger.info("Starting realtime dashboard using %s", args.realtime_config)

    try:
        iteration = 0
        while True:
            if realtime_fetcher is not None:
                snapshot = await realtime_fetcher.fetch_snapshot()
            else:
                snapshot = load_snapshot(Path(args.config))
            dashboard = build_dashboard(snapshot)
            print(dashboard)
            iteration += 1
            if args.iterations and iteration >= args.iterations:
                break
            if args.interval == 0:
                break
            await asyncio.sleep(args.interval)
        return 0
    finally:
        if realtime_fetcher is not None:
            await realtime_fetcher.close()


if __name__ == "__main__":
    sys.exit(main())

