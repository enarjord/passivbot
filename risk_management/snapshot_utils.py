"""Helpers for transforming risk management snapshots into presentation data."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .dashboard import (
    Account,
    AlertThresholds,
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

    payload: Dict[str, Any] = {
        "generated_at": generated_at.isoformat(),
        "accounts": account_views["visible"],
        "alerts": alerts,
        "notifications": notifications,
        "thresholds": _thresholds_to_view(thresholds),
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


def _build_account_view(account: Account, account_messages: Mapping[str, str]) -> Dict[str, Any]:
    positions = [_build_position_view(position, account.balance) for position in account.positions]
    message = account_messages.get(account.name)
    return {
        "name": account.name,
        "balance": account.balance,
        "exposure": account.exposure_pct(),
        "unrealized_pnl": account.total_unrealized(),
        "positions": positions,
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
    }


def _thresholds_to_view(thresholds: AlertThresholds) -> Dict[str, float]:
    return {
        "wallet_exposure_pct": thresholds.wallet_exposure_pct,
        "position_wallet_exposure_pct": thresholds.position_wallet_exposure_pct,
        "max_drawdown_pct": thresholds.max_drawdown_pct,
        "loss_threshold_pct": thresholds.loss_threshold_pct,
    }
