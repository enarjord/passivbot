"""Optional, externally authored daily UTC entry eligibility. No selection rules."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property, lru_cache
import hashlib
import io
from pathlib import Path

import numpy as np

DAY_MS = 86_400_000


@dataclass(frozen=True, eq=False)
class Selection:
    values: np.ndarray
    rows: np.ndarray
    sha256: str

    @cached_property
    def identity(self):
        return {"sha256": self.sha256, "shape": list(self.values.shape),
                "values_sha256": hashlib.sha256(self.values.tobytes()).hexdigest(),
                "rows_sha256": hashlib.sha256(self.rows.tobytes()).hexdigest(), "contract": 1}


def read_csv(path):
    """Validate every cell, including rows outside the requested interval."""
    path = Path(path).expanduser().resolve()
    try:
        content = path.read_bytes()
        records = list(csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Organillo CSV {path}: {exc}") from exc
    if not records or len(records[0]) < 2 or records[0][0] != "date":
        raise ValueError(f"Organillo CSV {path}: expected header date,COIN,...")
    coins = tuple(records[0][1:])
    if any(not c or c != c.strip() for c in coins) or len(set(coins)) != len(coins):
        raise ValueError(f"Organillo CSV {path}: coin headers must be nonempty and unique")
    dates, values = [], []
    for line, record in enumerate(records[1:], 2):
        if len(record) != len(coins) + 1:
            raise ValueError(f"Organillo CSV {path}, row {line}: expected {len(coins)+1} cells")
        try:
            date = datetime.strptime(record[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if date.strftime("%Y-%m-%d") != record[0]:
                raise ValueError("noncanonical date")
        except ValueError as exc:
            raise ValueError(f"Organillo CSV {path}, row {line}: date must be YYYY-MM-DD") from exc
        stamp = int(date.timestamp() * 1000)
        if dates and stamp <= dates[-1]:
            raise ValueError(f"Organillo CSV {path}, row {line}: dates must increase uniquely")
        for column, value in zip(coins, record[1:], strict=True):
            if value not in ("0", "1"):
                raise ValueError(f"Organillo CSV {path}, row {line}, column {column}: expected 0 or 1")
        dates.append(stamp)
        values.append([int(v) for v in record[1:]])
    if not dates:
        raise ValueError(f"Organillo CSV {path}: at least one date row is required")
    dates = np.asarray(dates, dtype=np.int64)
    values = np.asarray(values, dtype=np.uint8)
    dates.flags.writeable = values.flags.writeable = False
    return coins, dates, values, hashlib.sha256(content).hexdigest()


@lru_cache(maxsize=64)
def _source(path, expected_hash):
    result = read_csv(path)
    if expected_hash is not None and result[3] != expected_hash:
        raise ValueError(f"Organillo CSV {path}: SHA-256 differs from the prepared/saved input")
    return result


def freeze_config(config):
    """Validate once at dataset preparation and pin the immutable run input."""
    bt = config["backtest"]
    path = bt.get("organillo_csv")
    if path is None:
        return
    if not isinstance(path, str) or not path.strip():
        raise ValueError("backtest.organillo_csv must be a nonempty path or null")
    path = str(Path(path).expanduser().resolve())
    result = read_csv(path)
    expected = bt.get("organillo_sha256")
    if expected is not None and expected != result[3]:
        raise ValueError(f"Organillo CSV {path}: SHA-256 differs from the saved input")
    bt["organillo_csv"] = path
    bt["organillo_sha256"] = result[3]
    _source(path, result[3])


@lru_cache(maxsize=128)
def _align(path, expected_hash, coins, first, length, interval, operational_start):
    source_coins, dates, values, digest = _source(path, expected_hash)
    missing = sorted(set(coins) - set(source_coins))
    if missing:
        raise ValueError(f"Organillo CSV {path}: missing coin columns {', '.join(missing)}")
    stamps = first + np.arange(length, dtype=np.int64) * interval
    days = stamps // DAY_MS * DAY_MS
    rows = np.searchsorted(dates, days).astype(np.int32)
    required = stamps >= operational_start
    bounded = np.minimum(rows, len(dates) - 1)
    covered = (rows < len(dates)) & (dates[bounded] == days)
    if np.any(required & ~covered):
        day = int(days[np.flatnonzero(required & ~covered)[0]])
        date = datetime.fromtimestamp(day / 1000, timezone.utc).strftime("%Y-%m-%d")
        raise ValueError(f"Organillo CSV {path}: missing required UTC date {date}")
    # Warmup-only rows are inert, explicitly mapped to an appended zero row.
    aligned = np.ascontiguousarray(values[:, [source_coins.index(c) for c in coins]])
    aligned = np.concatenate((aligned, np.zeros((1, len(coins)), dtype=np.uint8)))
    rows[~required] = len(dates)
    aligned.flags.writeable = rows.flags.writeable = False
    return Selection(aligned, rows, digest)


def prepare_selection(config, coins, timestamps):
    bt = config["backtest"]
    path = bt.get("organillo_csv")
    if path is None:
        return None
    stamps = np.asarray(timestamps)
    if stamps.ndim != 1 or len(stamps) < 2 or stamps.dtype.kind not in "iu":
        raise ValueError("Organillo requires actual integer candle timestamps")
    interval = int(stamps[1] - stamps[0])
    if interval <= 0 or np.any(np.diff(stamps) != interval):
        raise ValueError("Organillo requires strictly increasing, evenly spaced candle timestamps")
    from utils import date_to_ts
    start = int(date_to_ts(bt["start_date"]))
    return _align(str(Path(path).expanduser().resolve()), bt.get("organillo_sha256"),
                  tuple(coins), int(stamps[0]), len(stamps), interval, start)


def validate_buffers(values, rows, bars, coins):
    if (values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != coins
            or not len(values) or np.any(values > 1)):
        raise ValueError("Organillo eligibility must be a binary uint8 matrix with one column per coin")
    if (rows.dtype != np.int32 or rows.shape != (bars,) or np.any(rows < 0)
            or np.any(rows >= len(values))):
        raise ValueError("Organillo row indices must be int32, in range, and match candle rows")


@lru_cache(maxsize=128)
def _period_union(path, digest, start, end):
    if end <= start:
        raise ValueError(f"Organillo CSV {path}: operational end date must follow start date")
    coins, dates, values, _digest = _source(path, digest)
    days = np.arange(start // DAY_MS * DAY_MS, end, DAY_MS, dtype=np.int64)
    rows = np.searchsorted(dates, days)
    if np.any(rows >= len(dates)) or np.any(dates[np.minimum(rows, len(dates)-1)] != days):
        raise ValueError(f"Organillo CSV {path}: missing required daily coverage for requested period")
    included = np.any(values[rows] != 0, axis=0)
    # An all-zero run still needs candle identities to produce a valid empty-trading replay.
    return frozenset(c for i, c in enumerate(coins) if included[i]) or frozenset(coins)


def period_coins(config, approved_coins=None):
    from utils import date_to_ts, format_end_date
    bt = config["backtest"]
    path = str(Path(bt["organillo_csv"]).expanduser().resolve())
    if approved_coins is not None:
        columns = _source(path, bt.get("organillo_sha256"))[0]
        unresolved = sorted(set(columns) - set(approved_coins))
        if unresolved:
            raise ValueError(f"Organillo CSV {path}: coin columns outside resolved approved universe: "
                             + ", ".join(unresolved) + "; use exact canonical coin identities")
    return _period_union(path, bt.get("organillo_sha256"), int(date_to_ts(bt["start_date"])),
                         int(date_to_ts(format_end_date(bt["end_date"]))))


def share_selection(config, coins, timestamps, manager):
    selection = prepare_selection(config, coins, timestamps)
    return (manager.create_from(selection.values)[0], manager.create_from(selection.rows)[0], selection.sha256)


def borrow_selection(specs, attachments):
    from shared_arrays import attach_shared_array
    values, rows, digest = specs
    for spec in (values, rows):
        if spec not in attachments:
            attachments[spec] = attach_shared_array(spec)
        attachments[spec].array.flags.writeable = False
    return Selection(attachments[values].array, attachments[rows].array, digest)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file")
    parser.add_argument("--coins", nargs="+", help="Exact canonical prepared coin identities")
    parser.add_argument("--start-date", help="First operational UTC date (inclusive)")
    parser.add_argument("--end-date", help="Operational end UTC date (exclusive)")
    args = parser.parse_args()
    try:
        coins, dates, _values, digest = read_csv(args.csv_file)
        if bool(args.start_date) != bool(args.end_date):
            raise ValueError("supply both --start-date and --end-date")
        missing = set(args.coins or ()) - set(coins)
        if missing:
            raise ValueError("missing coin columns " + ", ".join(sorted(missing)))
        if args.start_date:
            config = {"backtest": {"organillo_csv": args.csv_file, "organillo_sha256": digest,
                                    "start_date": args.start_date,
                                    "end_date": args.end_date}}
            period_coins(config)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Valid Organillo CSV: {len(dates)} dates, {len(coins)} coins, SHA-256 {digest}")


if __name__ == "__main__":
    main()
