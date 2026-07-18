"""Bounded, connector-proven balance-composition diagnostics.

The helpers in this module are observability-only.  They must never be used
to calculate the trading balance or to decide whether an account is ready.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


ASSET_BALANCE_MAX_ROWS = 8
ASSET_BALANCE_CONSOLE_SAMPLE_MAX = 2
_ASSET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _asset_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    asset = value.strip().upper()
    return asset if _ASSET_RE.fullmatch(asset) else None


def _collateral_enabled(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def _signature(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalize_snapshot(
    *,
    status: str,
    source: str,
    assets: list[dict[str, Any]],
    reason: str | None = None,
    invalid_row_count: int = 0,
) -> dict[str, Any]:
    ordered_assets = sorted(assets, key=lambda row: row["asset"])
    count = len(ordered_assets)
    retained_assets = ordered_assets[:ASSET_BALANCE_MAX_ROWS]
    snapshot: dict[str, Any] = {
        "status": status,
        "source": source,
        "asset_balances": retained_assets,
        "count": count,
        "retained": len(retained_assets),
        "truncated": max(0, count - len(retained_assets)),
    }
    if reason is not None:
        snapshot["reason"] = reason
    if invalid_row_count:
        snapshot["invalid_row_count"] = int(invalid_row_count)
    signature_payload = dict(snapshot)
    signature_payload["asset_balances"] = ordered_assets
    snapshot["_signature"] = _signature(signature_payload)
    return snapshot


def unavailable_balance_composition(*, reason: str = "unsupported_connector") -> dict[str, Any]:
    """Return one stable explicit state for connectors without a parser."""
    return _finalize_snapshot(
        status="unavailable",
        source="unsupported",
        reason=reason,
        assets=[],
    )


def malformed_balance_composition(*, source: str, reason: str) -> dict[str, Any]:
    """Return a bounded diagnostic failure without exposing response content."""
    return _finalize_snapshot(
        status="malformed",
        source=source,
        reason=reason,
        assets=[],
    )


def normalize_okx_balance_composition(fetched: Any) -> dict[str, Any]:
    """Normalize only documented OKX account-detail fields from one fetched response."""
    source = "okx.info.data[0].details"
    if not isinstance(fetched, Mapping):
        return malformed_balance_composition(source=source, reason="invalid_payload")
    info = fetched.get("info")
    if not isinstance(info, Mapping):
        return malformed_balance_composition(source=source, reason="missing_info")
    data = info.get("data")
    if not isinstance(data, list) or not data:
        return malformed_balance_composition(source=source, reason="missing_data")
    account = data[0]
    if not isinstance(account, Mapping):
        return malformed_balance_composition(source=source, reason="invalid_account")
    details = account.get("details")
    if not isinstance(details, list):
        return malformed_balance_composition(source=source, reason="missing_details")

    assets: list[dict[str, Any]] = []
    invalid_row_count = 0
    for detail in details:
        if not isinstance(detail, Mapping):
            invalid_row_count += 1
            continue
        asset = _asset_name(detail.get("ccy"))
        if asset is None:
            invalid_row_count += 1
            continue
        row: dict[str, Any] = {
            "asset": asset,
            "field_provenance": {"asset": "ccy"},
        }
        for field, payload_key in (
            ("amount", "cashBal"),
            ("usd_value", "eqUsd"),
            ("unrealized_pnl", "upl"),
            ("liability", "liab"),
        ):
            value = _finite_number(detail.get(payload_key))
            if value is not None:
                row[field] = value
                row["field_provenance"][field] = payload_key
        collateral = _collateral_enabled(detail.get("collateralEnabled"))
        if collateral is not None:
            row["collateral_enabled"] = collateral
            row["field_provenance"]["collateral_enabled"] = "collateralEnabled"
        assets.append(row)
    return _finalize_snapshot(
        status="available",
        source=source,
        assets=assets,
        invalid_row_count=invalid_row_count,
    )


def public_balance_composition(value: Any) -> dict[str, Any] | None:
    """Copy only the bounded event contract; never publish an internal signature."""
    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    source = value.get("source")
    assets = value.get("asset_balances")
    count = value.get("count")
    retained = value.get("retained")
    truncated = value.get("truncated")
    if (
        status not in {"available", "unavailable", "malformed"}
        or source not in {"unsupported", "normalizer", "okx.info.data[0].details"}
        or not isinstance(assets, list)
        or not all(isinstance(item, int) and item >= 0 for item in (count, retained, truncated))
    ):
        return None
    normalized_assets: list[dict[str, Any]] = []
    for item in assets[:ASSET_BALANCE_MAX_ROWS]:
        if not isinstance(item, Mapping):
            return None
        asset = _asset_name(item.get("asset"))
        if asset is None:
            return None
        row: dict[str, Any] = {"asset": asset}
        for key in ("amount", "usd_value", "unrealized_pnl", "liability"):
            number = _finite_number(item.get(key))
            if number is not None:
                row[key] = number
        if isinstance(item.get("collateral_enabled"), bool):
            row["collateral_enabled"] = item["collateral_enabled"]
        provenance = item.get("field_provenance")
        if isinstance(provenance, Mapping):
            allowed_sources = {
                "asset": "ccy",
                "amount": "cashBal",
                "usd_value": "eqUsd",
                "unrealized_pnl": "upl",
                "liability": "liab",
                "collateral_enabled": "collateralEnabled",
            }
            clean_provenance = {
                key: expected
                for key, expected in allowed_sources.items()
                if provenance.get(key) == expected and (key == "asset" or key in row)
            }
            if clean_provenance:
                row["field_provenance"] = clean_provenance
        normalized_assets.append(row)
    out: dict[str, Any] = {
        "status": status,
        "source": source,
        "asset_balances": normalized_assets,
        "count": count,
        "retained": retained,
        "truncated": truncated,
    }
    reason = value.get("reason")
    if isinstance(reason, str) and re.fullmatch(r"[a-z_]{1,48}", reason):
        out["reason"] = reason
    invalid_row_count = value.get("invalid_row_count")
    if isinstance(invalid_row_count, int) and invalid_row_count >= 0:
        out["invalid_row_count"] = invalid_row_count
    return out


def balance_composition_signature(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    signature = value.get("_signature")
    return signature if isinstance(signature, str) and len(signature) == 64 else None


def format_balance_composition_sample(value: Any) -> str | None:
    """Render at most two sanitized asset rows for an operator-facing balance line."""
    public = public_balance_composition(value)
    if not public or public.get("status") != "available":
        return None
    parts = []
    for row in public["asset_balances"][:ASSET_BALANCE_CONSOLE_SAMPLE_MAX]:
        asset = _asset_name(row.get("asset"))
        if asset is None:
            continue
        amount = _finite_number(row.get("amount"))
        usd_value = _finite_number(row.get("usd_value"))
        liability = _finite_number(row.get("liability"))
        values = []
        if amount is not None:
            values.append(f"a={amount:.6g}")
        if usd_value is not None:
            values.append(f"usd={usd_value:.6g}")
        if liability is not None:
            values.append(f"liab={liability:.6g}")
        if row.get("collateral_enabled") is True:
            values.append("coll=yes")
        elif row.get("collateral_enabled") is False:
            values.append("coll=no")
        if values:
            parts.append(f"{asset}:{','.join(values)}")
    if not parts:
        return None
    omitted = max(0, int(public["count"]) - len(parts))
    suffix = f"+{omitted} more" if omitted else ""
    rendered = ";".join(parts)
    return f"{rendered} {suffix}" if suffix else rendered
