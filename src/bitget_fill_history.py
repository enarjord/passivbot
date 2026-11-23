"""Traverse Bitget's Classic and UTA private fill-history pages."""

from collections.abc import AsyncIterator

from live.state_refresh import AuthoritativeSurfaceUnavailable


async def bitget_fill_pages(
    api,
    *,
    uta: bool,
    product_type: str,
    since_ms: int | None,
    until_ms: int,
    limit: int,
    max_fetches: int,
) -> AsyncIterator[list[dict]]:
    if not 1 <= limit <= 100:
        raise ValueError("Bitget fill-history limit must be in [1, 100]")
    if uta:
        fetch = api.private_uta_get_v3_trade_fills
        params = {"category": product_type, "limit": limit}
        rows_key, id_key, cursor_key, request_cursor = (
            "list",
            "execId",
            "cursor",
            "cursor",
        )
        window_ms = 30 * 86_400_000
    else:
        fetch = api.private_mix_get_v2_mix_order_fill_history
        params = {"productType": product_type, "limit": limit}
        rows_key, id_key, cursor_key, request_cursor = (
            "fillList",
            "tradeId",
            "endId",
            "idLessThan",
        )
        window_ms = 7 * 86_400_000

    if since_ms is not None and since_ms > until_ms:
        raise ValueError("Bitget fill-history start exceeds end")
    timestamp_key = "createdTime" if uta else "cTime"
    end = int(until_ms)
    fetch_count = 0
    while True:
        window = {**params, "endTime": end}
        if since_ms is not None:
            window["startTime"] = max(since_ms, end - window_ms + 1)
        cursors = set()
        seen_ids = set()
        while True:
            if fetch_count >= max_fetches:
                raise AuthoritativeSurfaceUnavailable(
                    "fills",
                    f"Bitget incomplete history after pagination cap ({max_fetches})",
                )
            fetch_count += 1
            payload = await fetch(dict(window))
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict) or not isinstance(data.get(rows_key), list):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", f"Bitget expected data.{rows_key} list"
                )
            if payload.get("code") != "00000":
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget fill history returned an unsuccessful response"
                )
            rows = data[rows_key]
            page_ids = set()
            for row in rows:
                row_id = row.get(id_key) if isinstance(row, dict) else None
                if not isinstance(row_id, str) or not row_id.strip():
                    raise AuthoritativeSurfaceUnavailable(
                        "fills", f"Bitget fill missing {id_key}"
                    )
                timestamp = row.get(timestamp_key)
                if (
                    not isinstance(timestamp, (str, int))
                    or not str(timestamp).isascii()
                    or not str(timestamp).isdecimal()
                ):
                    raise ValueError(f"Bitget fill missing or invalid {timestamp_key}")
                if not window.get("startTime", 0) <= int(timestamp) <= end:
                    raise AuthoritativeSurfaceUnavailable(
                        "fills", "Bitget fill outside requested window"
                    )
                page_ids.add(row_id)
            if rows and not page_ids.difference(seen_ids):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget fill-history page made no progress"
                )
            if len(rows) > limit:
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget fill-history page exceeds requested limit"
                )
            if not rows:
                break
            cursor = data.get(cursor_key)
            if len(rows) < limit and cursor in (None, ""):
                yield rows
                break
            if (
                not isinstance(cursor, str)
                or not cursor
                or cursor != cursor.strip()
                or cursor in cursors
            ):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", f"Bitget invalid or repeated fill-history {cursor_key}"
                )
            previous = window.get(request_cursor)
            if (
                not uta
                and previous
                and previous.isdecimal()
                and cursor.isdecimal()
                and int(cursor) >= int(previous)
            ):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget endId did not move to an older ID"
                )
            cursors.add(cursor)
            seen_ids.update(page_ids)
            yield rows
            window[request_cursor] = cursor
        if since_ms is None or window["startTime"] == since_ms:
            return
        end = window["startTime"] - 1
