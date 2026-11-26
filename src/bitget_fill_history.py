"""Traverse Bitget's Classic and UTA private fill-history pages."""

from collections.abc import AsyncIterator, Iterator

from live.state_refresh import AuthoritativeSurfaceUnavailable


BITGET_UTA_RETENTION_MS = 90 * 86_400_000
BITGET_UTA_WINDOW_MS = 30 * 86_400_000


def _uta_windows(start: int, end: int, observed_at: int) -> Iterator[tuple[int, int]]:
    if start == end:
        # The venue rejects equal timestamps. Widen by one millisecond on
        # the side that stays inside the observed retention window.
        yield (start, end + 1) if end < observed_at else (start - 1, end)
        return
    while start <= end:
        stop = min(start + BITGET_UTA_WINDOW_MS - 1, end)
        if stop == end - 1:
            # Leave two timestamps for the final request, not a singleton.
            stop -= 1
        yield start, stop
        start = stop + 1


async def bitget_fill_pages(
    api,
    *,
    uta: bool,
    product_type: str,
    since_ms: int | None,
    until_ms: int,
    observed_at_ms: int,
    limit: int,
    max_fetches: int,
) -> AsyncIterator[list[dict]]:
    if not 1 <= limit <= 100:
        raise ValueError("Bitget fill-history limit must be in [1, 100]")
    if uta:
        oldest_requested = until_ms if since_ms is None else since_ms
        if oldest_requested < observed_at_ms - BITGET_UTA_RETENTION_MS:
            raise AuthoritativeSurfaceUnavailable(
                "fills", "Bitget UTA cannot verify fills outside its 90-day retention"
            )
        fetch = api.private_uta_get_v3_trade_fills
        params = {"category": product_type, "limit": limit}
        rows_key, id_key, cursor_key, request_cursor = (
            "list",
            "execId",
            "cursor",
            "cursor",
        )
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
    padded_singleton = uta and since_ms == until_ms
    if since_ms is None:
        windows = ((None, until_ms),)
    elif uta:
        # Read the oldest interval first before later pages consume retention.
        windows = _uta_windows(since_ms, until_ms, observed_at_ms)
    else:
        windows = (
            (max(since_ms, end - window_ms + 1), end)
            for end in range(until_ms, since_ms - 1, -window_ms)
        )
    fetch_count = 0
    for start, end in windows:
        window = {**params, "endTime": end}
        if start is not None:
            window["startTime"] = start
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
            if not isinstance(payload, dict) or payload.get("code") != "00000":
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget fill history returned an unsuccessful response"
                )
            data = payload.get("data")
            if not isinstance(data, dict):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", f"Bitget expected data.{rows_key} list"
                )
            rows = data.get(rows_key)
            cursor = data.get(cursor_key)
            # UTA returns explicit nulls both for an empty query and after
            # following a cursor past the last fill.
            if uta and rows is None and cursor is None and {rows_key, cursor_key} <= data.keys():
                break
            if not isinstance(rows, list):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", f"Bitget expected data.{rows_key} list"
                )
            page_ids = set()
            requested_rows = []
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
                # Do not leak the extra timestamp used to query a singleton.
                if not padded_singleton or int(timestamp) == since_ms:
                    requested_rows.append(row)
            if rows and not page_ids.difference(seen_ids):
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget fill-history page made no progress"
                )
            if len(rows) > limit:
                raise AuthoritativeSurfaceUnavailable(
                    "fills", "Bitget fill-history page exceeds requested limit"
                )
            if not rows:
                if cursor not in (None, ""):
                    raise AuthoritativeSurfaceUnavailable(
                        "fills",
                        f"Bitget empty fill page has a continuation {cursor_key}",
                    )
                break
            if len(rows) < limit and cursor in (None, ""):
                if requested_rows:
                    yield requested_rows
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
            if requested_rows:
                yield requested_rows
            window[request_cursor] = cursor
