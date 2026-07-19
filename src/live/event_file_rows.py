from __future__ import annotations

import gzip
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EventRowWindow:
    limited: bool = False
    skipped_lines: int | None = None
    skipped_lines_exact: bool = True
    skipped_bytes: int = 0
    line_numbers_exact: bool = True
    method: str = "full_scan"
    physical_bytes_read: int | None = None
    decoded_bytes_read: int | None = None


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _plain_tail_rows(path: Path, *, max_lines: int) -> tuple[list[tuple[int, str]], EventRowWindow]:
    if max_lines <= 0:
        return [], EventRowWindow()
    size = path.stat().st_size
    if size <= 0:
        return [], EventRowWindow(
            limited=True,
            skipped_lines=0,
            skipped_lines_exact=True,
            skipped_bytes=0,
            line_numbers_exact=True,
            method="seek_tail",
            physical_bytes_read=0,
            decoded_bytes_read=0,
        )

    chunk_size = 64 * 1024
    chunks: list[bytes] = []
    pos = size
    newline_count = 0
    with open(path, "rb") as stream:
        while pos > 0 and newline_count <= max_lines:
            read_size = min(chunk_size, pos)
            pos -= read_size
            stream.seek(pos)
            chunk = stream.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    reached_start = pos == 0
    data = b"".join(reversed(chunks))
    raw_lines = data.splitlines()
    if not reached_start:
        raw_lines = raw_lines[-max_lines:]
    elif len(raw_lines) > max_lines:
        raw_lines = raw_lines[-max_lines:]

    decoded = [line.decode("utf-8", errors="replace") for line in raw_lines]
    skipped_lines = None
    if reached_start:
        skipped_lines = max(0, len(data.splitlines()) - len(decoded))
    skipped_bytes = max(0, pos)
    line_numbers_exact = skipped_lines is not None
    line_offset = int(skipped_lines or 0)
    rows = [(line_offset + idx, line) for idx, line in enumerate(decoded, start=1)]
    return rows, EventRowWindow(
        limited=True,
        skipped_lines=skipped_lines,
        skipped_lines_exact=skipped_lines is not None,
        skipped_bytes=skipped_bytes,
        line_numbers_exact=line_numbers_exact,
        method="seek_tail",
        physical_bytes_read=len(data),
        decoded_bytes_read=len(data),
    )


@contextmanager
def event_file_rows(path: Path, *, max_tail_lines: int = 0, text_opener=None):
    tail_lines = max(0, int(max_tail_lines))
    opener = _open_text if text_opener is None else text_opener
    if tail_lines <= 0:
        try:
            physical_bytes_read = int(path.stat().st_size)
        except OSError:
            physical_bytes_read = None
        window = EventRowWindow(
            physical_bytes_read=physical_bytes_read,
            decoded_bytes_read=(
                None if path.name.endswith(".gz") else physical_bytes_read
            ),
        )
        with opener(path) as stream:
            yield enumerate(stream, start=1), window
            if path.name.endswith(".gz"):
                buffer = getattr(stream, "buffer", None)
                if buffer is not None:
                    window.decoded_bytes_read = int(buffer.tell())
        return
    if not path.name.endswith(".gz"):
        rows, window = _plain_tail_rows(path, max_lines=tail_lines)
        yield rows, window
        return
    physical_bytes_read = int(path.stat().st_size)
    with _open_text(path) as stream:
        rows = deque(enumerate(stream, start=1), maxlen=tail_lines)
        decoded_bytes_read = int(stream.buffer.tell())
    skipped_lines = max(0, int(rows[0][0]) - 1) if rows else 0
    yield rows, EventRowWindow(
        limited=True,
        skipped_lines=skipped_lines,
        skipped_lines_exact=True,
        skipped_bytes=0,
        line_numbers_exact=True,
        method="sequential_gzip_tail",
        physical_bytes_read=physical_bytes_read,
        decoded_bytes_read=decoded_bytes_read,
    )
