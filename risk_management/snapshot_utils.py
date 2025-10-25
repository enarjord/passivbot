"""Helpers for transforming risk management snapshots into presentation data."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .dashboard import (
    Account,
    AlertThresholds,
    Order,
    Position,
    evaluate_alerts,
    parse_snapshot,
)


def build_presentable_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a snapshot payload into UI friendly structures."""

    generated_at, accounts, thresholds, notifications = parse_snapshot(dict(snapshot))
    alerts = evaluate_alerts(accounts, thresholds)
    account_messages = snapshot.get("account_messages", {}) if isinstance(snapshot, Mapping) else {}
    account_views = _build_account_views(accounts, account_messages)
    portfolio = _build_portfolio_view(accounts)

    payload: Dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "accounts": account_views["visible"],
        "alerts": alerts,
        "notifications": notifications,
        "thresholds": _thresholds_to_view(thresholds),
        "portfolio": portfolio,
    }

    if account_views["hidden"]:
        payload["hidden_accounts"] = account_views["hidden"]

    return payload


def _build_account_views(
    accounts: Sequence[Account], account_messages: Mapping[str, str]
) -> Dict[str, list[Dict[str, Any]]]:
    visible_accounts: list[Dict[str, Any]] = []
    hidden_accounts: list[Dict[str, Any]] = []

    for account in accounts:
        view = _build_account_view(account, account_messages)
        if view["message"]:
            hidden_accounts.append({"name": view["name"], "message": view["message"]})
            continue
        visible_accounts.append(view)

    return {"visible": visible_accounts, "hidden": hidden_accounts}


def _build_portfolio_view(accounts: Sequence[Account]) -> Dict[str, Any]:
    total_balance = sum(account.balance for account in accounts)
    gross_notional = sum(account.total_abs_notional() for account in accounts)
    net_notional = sum(account.net_notional() for account in accounts)
    symbols: Dict[str, Dict[str, float]] = {}
    for account in accounts:
        for symbol, values in account.exposures_by_symbol().items():
            entry = symbols.setdefault(symbol, {"gross": 0.0, "net": 0.0})
            entry["gross"] += values["gross"]
            entry["net"] += values["net"]
    symbol_entries: List[Dict[str, Any]] = []
    for symbol, values in symbols.items():
        gross_pct = values["gross"] / total_balance if total_balance else 0.0
        net_pct = values["net"] / total_balance if total_balance else 0.0
        symbol_entries.append(
            {
                "symbol": symbol,
                "gross_notional": values["gross"],
                "net_notional": values["net"],
                "gross_pct": gross_pct,
                "net_pct": net_pct,
            }
        )
    symbol_entries.sort(key=lambda item: item["gross_notional"], reverse=True)
    gross_pct_total = gross_notional / total_balance if total_balance else 0.0
    net_pct_total = net_notional / total_balance if total_balance else 0.0
    return {
        "balance": total_balance,
        "gross_exposure": gross_notional,
        "net_exposure": net_notional,
        "gross_exposure_pct": gross_pct_total,
        "net_exposure_pct": net_pct_total,
        "symbols": symbol_entries,
    }


def _build_account_view(account: Account, account_messages: Mapping[str, str]) -> Dict[str, Any]:
    positions = [_build_position_view(position, account.balance) for position in account.positions]
    orders = [_build_order_view(order) for order in account.orders]
    message = account_messages.get(account.name)
    symbol_exposures = _build_symbol_exposures(account)
    return {
        "name": account.name,
        "balance": account.balance,
        "exposure": account.exposure_pct(),
        "gross_exposure": account.gross_exposure_pct(),
        "gross_exposure_notional": account.total_abs_notional(),
        "net_exposure": account.net_exposure_pct(),
        "net_exposure_notional": account.net_notional(),
        "unrealized_pnl": account.total_unrealized(),
        "positions": positions,
        "symbol_exposures": symbol_exposures,
        "orders": orders,
        "message": message,
    }


def _build_position_view(position: Position, balance: float) -> Dict[str, Any]:
    exposure = position.exposure_relative_to(balance)
    pnl_pct = position.pnl_pct(balance)
    return {
        "symbol": position.symbol,
        "side": position.side,
        "notional": position.notional,
        "entry_price": position.entry_price,
        "mark_price": position.mark_price,
        "liquidation_price": position.liquidation_price,
        "wallet_exposure_pct": position.wallet_exposure_pct if position.wallet_exposure_pct is not None else exposure,
        "exposure": exposure,
        "unrealized_pnl": position.unrealized_pnl,
        "pnl_pct": pnl_pct,
        "max_drawdown_pct": position.max_drawdown_pct,
        "take_profit_price": position.take_profit_price,
        "stop_loss_price": position.stop_loss_price,
        "size": position.size,
        "signed_notional": position.signed_notional,
    }


def _build_symbol_exposures(account: Account) -> List[Dict[str, Any]]:
    exposures = account.exposures_by_symbol()
    items: List[Dict[str, Any]] = []
    balance = account.balance or 0.0
    for symbol, values in exposures.items():
        gross = values["gross"]
        net = values["net"]
        gross_pct = gross / balance if balance else 0.0
        net_pct = net / balance if balance else 0.0
        items.append(
            {
                "symbol": symbol,
                "gross_notional": gross,
                "net_notional": net,
                "gross_pct": gross_pct,
                "net_pct": net_pct,
            }
        )
    items.sort(key=lambda item: item["gross_notional"], reverse=True)
    return items


def _build_order_view(order: Order) -> Dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side,
        "type": order.order_type,
        "price": order.price,
        "amount": order.amount,
        "remaining": order.remaining,
        "status": order.status,
        "reduce_only": order.reduce_only,
        "stop_price": order.stop_price,
        "notional": order.notional,
        "order_id": order.order_id,
        "created_at": order.created_at,
    }


def _thresholds_to_view(thresholds: AlertThresholds) -> Dict[str, float]:
    return {
        "wallet_exposure_pct": thresholds.wallet_exposure_pct,
        "position_wallet_exposure_pct": thresholds.position_wallet_exposure_pct,
        "max_drawdown_pct": thresholds.max_drawdown_pct,
        "loss_threshold_pct": thresholds.loss_threshold_pct,
    }
