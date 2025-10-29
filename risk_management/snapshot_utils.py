"""Helpers for transforming risk management snapshots into presentation data."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .dashboard import evaluate_alerts, parse_snapshot
from .domain.models import Account, AlertThresholds, Order, Position


DEFAULT_ACCOUNTS_PAGE_SIZE = 25
MAX_ACCOUNTS_PAGE_SIZE = 200
DEFAULT_ACCOUNT_SORT_KEY = "balance"
DEFAULT_ACCOUNT_SORT_ORDER = "desc"
ACCOUNT_SORT_FIELDS: Dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "name": lambda account: str(account.get("name", "")),
    "balance": lambda account: float(account.get("balance", 0.0)),
    "gross": lambda account: float(account.get("gross_exposure_notional", 0.0)),
    "net": lambda account: float(account.get("net_exposure_notional", 0.0)),
    "unrealized": lambda account: float(account.get("unrealized_pnl", 0.0)),
    "daily_realized": lambda account: float(account.get("daily_realized_pnl", 0.0)),
}

EXPOSURE_FILTERS = {"any", "gross", "net_long", "net_short", "flat"}


def build_presentable_snapshot(
    snapshot: Mapping[str, Any],
    *,
    account_name: Optional[str] = None,
    search: Optional[str] = None,
    exposure_filter: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    sort_key: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a snapshot payload into UI friendly structures."""

    generated_at, accounts, thresholds, notifications = parse_snapshot(dict(snapshot))
    alerts = evaluate_alerts(accounts, thresholds)
    account_messages = snapshot.get("account_messages", {}) if isinstance(snapshot, Mapping) else {}
    performance = snapshot.get("performance") if isinstance(snapshot, Mapping) else None
    account_performance = (
        performance.get("accounts")
        if isinstance(performance, Mapping)
        else {}
    )
    portfolio_performance = (
        performance.get("portfolio") if isinstance(performance, Mapping) else None
    )
    account_stop_losses = (
        snapshot.get("account_stop_losses") if isinstance(snapshot, Mapping) else None
    )
    account_views = _build_account_views(
        accounts,
        account_messages,
        account_stop_losses,
        account_performance,
    )
    portfolio = _build_portfolio_view(accounts, portfolio_performance)

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
    stop_loss = snapshot.get("portfolio_stop_loss") if isinstance(snapshot, Mapping) else None
    if isinstance(stop_loss, Mapping):
        payload["portfolio_stop_loss"] = dict(stop_loss)
    account_stop_loss_view = _normalise_account_stop_losses(account_stop_losses)
    if account_stop_loss_view:
        payload["account_stop_losses"] = account_stop_loss_view

    (accounts_page, meta) = _slice_accounts(
        account_views["visible"],
        account_name=account_name,
        search=search,
        exposure_filter=exposure_filter,
        page=page,
        page_size=page_size,
        sort_key=sort_key,
        sort_order=sort_order,
    )
    payload["accounts"] = accounts_page
    payload["accounts_meta"] = meta

    return payload


def _slice_accounts(
    accounts: Sequence[Mapping[str, Any]],
    *,
    account_name: Optional[str] = None,
    search: Optional[str] = None,
    exposure_filter: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    sort_key: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply filtering, sorting and pagination to the account list."""

    all_accounts = list(accounts)
    total_accounts = len(all_accounts)

    requested_sort_key = sort_key or DEFAULT_ACCOUNT_SORT_KEY
    if requested_sort_key not in ACCOUNT_SORT_FIELDS:
        requested_sort_key = DEFAULT_ACCOUNT_SORT_KEY

    requested_sort_order = (sort_order or DEFAULT_ACCOUNT_SORT_ORDER).lower()
    if requested_sort_order not in {"asc", "desc"}:
        requested_sort_order = DEFAULT_ACCOUNT_SORT_ORDER

    requested_page_size = page_size or DEFAULT_ACCOUNTS_PAGE_SIZE
    if requested_page_size <= 0:
        requested_page_size = DEFAULT_ACCOUNTS_PAGE_SIZE
    requested_page_size = min(requested_page_size, MAX_ACCOUNTS_PAGE_SIZE)

    requested_page = page or 1
    if requested_page <= 0:
        requested_page = 1

    account_name_normalised = account_name.lower() if account_name else None
    exposure_mode = (exposure_filter or "any").lower()
    if exposure_mode not in EXPOSURE_FILTERS:
        exposure_mode = "any"

    def _matches_account_filter(account: Mapping[str, Any]) -> bool:
        if account_name_normalised and account.get("name", "").lower() != account_name_normalised:
            return False
        if exposure_mode != "any" and not _matches_exposure_filter(account, exposure_mode):
            return False
        if search:
            return _matches_search(account, search)
        return True

    filtered_accounts = [account for account in all_accounts if _matches_account_filter(account)]
    filtered_count = len(filtered_accounts)

    key_function = ACCOUNT_SORT_FIELDS[requested_sort_key]
    sorted_accounts = sorted(
        filtered_accounts,
        key=lambda account: key_function(account),
        reverse=requested_sort_order == "desc",
    )

    total_pages = (filtered_count + requested_page_size - 1) // requested_page_size if filtered_count else 1
    if total_pages <= 0:
        total_pages = 1
    current_page = min(max(1, requested_page), total_pages)

    start_index = (current_page - 1) * requested_page_size
    end_index = start_index + requested_page_size
    page_accounts = [dict(account) for account in sorted_accounts[start_index:end_index]]

    meta: Dict[str, Any] = {
        "total": total_accounts,
        "filtered": filtered_count,
        "page": current_page,
        "pages": total_pages,
        "page_size": requested_page_size,
        "sort_key": requested_sort_key,
        "sort_order": requested_sort_order,
        "account": account_name,
        "search": search or "",
        "exposure_filter": exposure_mode,
        "has_next": current_page < total_pages,
        "has_previous": current_page > 1,
    }

    return page_accounts, meta


def _matches_search(account: Mapping[str, Any], raw_query: str) -> bool:
    query = raw_query.lower()
    if query in str(account.get("name", "")).lower():
        return True
    for key in ("symbol_exposures", "positions", "orders"):
        items = account.get(key)
        if not isinstance(items, Iterable):
            continue
        for item in items:
            symbol = ""
            if isinstance(item, Mapping):
                symbol = str(item.get("symbol", ""))
            elif hasattr(item, "get"):
                symbol = str(item.get("symbol", ""))  # type: ignore[attr-defined]
            if query and query in symbol.lower():
                return True
    return False


def _matches_exposure_filter(account: Mapping[str, Any], mode: str) -> bool:
    gross_notional = float(account.get("gross_exposure_notional", 0.0) or 0.0)
    net_notional = float(account.get("net_exposure_notional", 0.0) or 0.0)
    if mode == "gross":
        return abs(gross_notional) > 0.0
    if mode == "net_long":
        return net_notional > 0.0
    if mode == "net_short":
        return net_notional < 0.0
    if mode == "flat":
        return abs(gross_notional) == 0.0 and abs(net_notional) == 0.0
    return True


def _build_account_views(
    accounts: Sequence[Account],
    account_messages: Mapping[str, str],
    account_stop_losses: Any,
    account_performance: Any,
) -> Dict[str, list[Dict[str, Any]]]:
    visible_accounts: list[Dict[str, Any]] = []
    hidden_accounts: list[Dict[str, Any]] = []

    for account in accounts:
        view = _build_account_view(
            account,
            account_messages,
            account_stop_losses,
            account_performance,
        )
        if view["message"]:
            hidden_accounts.append({"name": view["name"], "message": view["message"]})
            continue
        visible_accounts.append(view)

    return {"visible": visible_accounts, "hidden": hidden_accounts}


def _build_portfolio_view(
    accounts: Sequence[Account], performance: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    total_balance = sum(account.balance for account in accounts)
    gross_notional = sum(account.total_abs_notional() for account in accounts)
    net_notional = sum(account.net_notional() for account in accounts)
    daily_realized = sum(account.total_daily_realized() for account in accounts)

    symbol_data: Dict[str, Dict[str, Any]] = {}
    portfolio_volatility: Dict[str, float] = {}
    portfolio_funding: Dict[str, float] = {}
    volatility_weights: Dict[str, float] = {}
    funding_weights: Dict[str, float] = {}

    for account in accounts:
        for position in account.positions:
            signed = (
                position.signed_notional
                if position.signed_notional is not None
                else position.notional if position.side.lower() == "long" else -position.notional
            )
            gross_value = abs(signed)
            entry = symbol_data.setdefault(
                position.symbol,
                {
                    "gross": 0.0,
                    "net": 0.0,
                    "volatility": {},
                    "vol_weights": {},
                    "funding": {},
                    "funding_weights": {},
                },
            )
            entry["gross"] += gross_value
            entry["net"] += signed
            weight = gross_value or abs(position.notional) or 0.0
            if weight:
                _accumulate_metric(entry["volatility"], entry["vol_weights"], position.volatility, weight)
                _accumulate_metric(entry["funding"], entry["funding_weights"], position.funding_rates, weight)
                _accumulate_metric(portfolio_volatility, volatility_weights, position.volatility, weight)
                _accumulate_metric(portfolio_funding, funding_weights, position.funding_rates, weight)

    symbol_entries: List[Dict[str, Any]] = []
    for symbol, values in symbol_data.items():
        gross_pct = values["gross"] / total_balance if total_balance else 0.0
        net_pct = values["net"] / total_balance if total_balance else 0.0
        symbol_entries.append(
            {
                "symbol": symbol,
                "gross_notional": values["gross"],
                "net_notional": values["net"],
                "gross_pct": gross_pct,
                "net_pct": net_pct,
                "volatility": _finalise_metric(values["volatility"], values["vol_weights"]),
                "funding_rates": _finalise_metric(values["funding"], values["funding_weights"]),
            }
        )
    symbol_entries.sort(key=lambda item: item["gross_notional"], reverse=True)
    gross_pct_total = gross_notional / total_balance if total_balance else 0.0
    net_pct_total = net_notional / total_balance if total_balance else 0.0
    payload: Dict[str, Any] = {
        "balance": total_balance,
        "gross_exposure": gross_notional,
        "net_exposure": net_notional,
        "gross_exposure_pct": gross_pct_total,
        "net_exposure_pct": net_pct_total,
        "daily_realized_pnl": daily_realized,
        "volatility": _finalise_metric(portfolio_volatility, volatility_weights),
        "funding_rates": _finalise_metric(portfolio_funding, funding_weights),
        "symbols": symbol_entries,
    }
    if performance:
        payload["performance"] = _normalise_performance(performance)
    return payload


def _build_account_view(
    account: Account,
    account_messages: Mapping[str, str],
    account_stop_losses: Any,
    account_performance: Any,
) -> Dict[str, Any]:
    positions = [_build_position_view(position, account.balance) for position in account.positions]
    orders = [_build_order_view(order) for order in account.orders]
    message = account_messages.get(account.name)
    symbol_exposures = _build_symbol_exposures(account)
    volatility = _aggregate_position_metrics(account.positions, "volatility")
    funding_rates = _aggregate_position_metrics(account.positions, "funding_rates")
    stop_loss_state = None
    if isinstance(account_stop_losses, Mapping):
        raw_stop_loss = account_stop_losses.get(account.name)
        stop_loss_state = _normalise_stop_loss(raw_stop_loss)
    performance_state = None
    if isinstance(account_performance, Mapping):
        performance_state = _normalise_performance(account_performance.get(account.name))
    return {
        "name": account.name,
        "balance": account.balance,
        "exposure": account.exposure_pct(),
        "gross_exposure": account.gross_exposure_pct(),
        "gross_exposure_notional": account.total_abs_notional(),
        "net_exposure": account.net_exposure_pct(),
        "net_exposure_notional": account.net_notional(),
        "unrealized_pnl": account.total_unrealized(),
        "daily_realized_pnl": account.daily_realized_pnl or account.total_daily_realized(),
        "positions": positions,
        "symbol_exposures": symbol_exposures,
        "orders": orders,
        "message": message,
        "volatility": volatility,
        "funding_rates": funding_rates,
        "stop_loss": stop_loss_state,
        "performance": performance_state,
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
        "daily_realized_pnl": position.daily_realized_pnl,
        "volatility": dict(position.volatility) if position.volatility else {},
        "funding_rates": dict(position.funding_rates) if position.funding_rates else {},
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


def _normalise_account_stop_losses(data: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(data, Mapping):
        return {}
    normalised: Dict[str, Dict[str, Any]] = {}
    for name, value in data.items():
        state = _normalise_stop_loss(value)
        if state is not None:
            normalised[str(name)] = state
    return normalised


def _normalise_stop_loss(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, Mapping):
        return None
    result: Dict[str, Any] = {
        "threshold_pct": _to_optional_float(data.get("threshold_pct")),
        "baseline_balance": _to_optional_float(data.get("baseline_balance")),
        "current_balance": _to_optional_float(data.get("current_balance")),
        "current_drawdown_pct": _to_optional_float(data.get("current_drawdown_pct")),
        "triggered": bool(data.get("triggered")),
        "active": bool(data.get("active", True)),
        "triggered_at": data.get("triggered_at"),
    }
    return result


def _normalise_performance(data: Any) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    summary: Dict[str, Any] = {
        "current_balance": _to_optional_float(data.get("current_balance")),
        "latest_snapshot": None,
        "daily": None,
        "weekly": None,
        "monthly": None,
        "daily_pct": None,
        "weekly_pct": None,
        "monthly_pct": None,
    }
    latest = data.get("latest_snapshot")
    if isinstance(latest, Mapping):
        summary["latest_snapshot"] = {
            "date": latest.get("date"),
            "timestamp": latest.get("timestamp"),
            "balance": _to_optional_float(latest.get("balance")),
        }
    since_map: Dict[str, Any] = {}
    reference_balances: Dict[str, Any] = {}
    for key in ("daily", "weekly", "monthly"):
        change = data.get(key)
        if isinstance(change, Mapping):
            pnl = change.get("pnl")
            pct = change.get("pct_change")
            summary[key] = float(pnl) if pnl is not None else None
            summary[f"{key}_pct"] = float(pct) if pct is not None else None
            if change.get("since") is not None:
                since_map[key] = change.get("since")
            if change.get("reference_balance") is not None:
                reference_balances[key] = float(change.get("reference_balance"))
        elif isinstance(change, (int, float)):
            summary[key] = float(change)
            summary[f"{key}_pct"] = None
        else:
            summary[key] = None
            summary[f"{key}_pct"] = None
    if since_map:
        summary["since"] = since_map
    if reference_balances:
        summary["reference_balances"] = reference_balances
    return summary


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _accumulate_metric(
    totals: MutableMapping[str, float],
    weights: MutableMapping[str, float],
    metrics: Optional[Mapping[str, float]],
    weight: float,
) -> None:
    if not metrics or weight <= 0:
        return
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        totals[key] = totals.get(key, 0.0) + numeric * weight
        weights[key] = weights.get(key, 0.0) + weight


def _finalise_metric(
    totals: Mapping[str, float], weights: Mapping[str, float]
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for key, total in totals.items():
        weight = weights.get(key, 0.0)
        if weight:
            results[key] = total / weight
    return results


def _aggregate_position_metrics(
    positions: Sequence[Position], attribute: str
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for position in positions:
        metrics = getattr(position, attribute, None)
        if not metrics:
            continue
        weight = abs(position.notional) or abs(position.signed_notional or 0.0)
        if not weight:
            continue
        _accumulate_metric(totals, weights, metrics, weight)
    return _finalise_metric(totals, weights)


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
