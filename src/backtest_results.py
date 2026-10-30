"""Versioned, lossless backtest facts and lazy readers shared by reports/tools.

Numeric archives never contain pickles. Rendering choices do not change these
facts or the native metric population. Legacy CSV results remain readable.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

RESULT_VERSION = 1
EQUITY_FILE = "balance_and_equity.npz"
TRACE_FILE = "hsl_trace.npz"
EQUITY_COLUMNS = (
    "usd_total_equity",
    "btc_total_equity",
    "strategy_equity",
    "usd_cash_wallet",
    "usd_total_balance",
    "btc_cash_wallet",
    "btc_total_balance",
)
TRACE_DTYPES = {
    "sequence": "uint64",
    "timestamp": "int64",
    "side": "int8",
    "coin": "int64",
    "phase": "uint8",
    "action": "uint8",
    "exposed": "uint8",
    "signal_raw_fraction": "float64",
    "signal_ema_fraction": "float64",
    "trigger_signal_fraction": "float64",
    "nonpositive_equity": "uint8",
    "drawdown_memory_usd": "float64",
    "budget_usd": "float64",
    "upnl_usd": "float64",
}


@contextmanager
def atomic_output(path: Path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as stream:
            yield stream
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value) -> None:
    with atomic_output(path, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_numeric_archive(path, **arrays):
    # NPZ is an ordinary ZIP of NPY columns. Level 1 keeps serialization fast
    # while retaining the same exact numeric bits and standard NumPy reader.
    with atomic_output(Path(path)) as stream:
        with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for name, value in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as column:
                    np.lib.format.write_array(
                        column, np.asarray(value), allow_pickle=False
                    )


def _version(value, label):
    value = np.asarray(value)
    if value.ndim != 0 or value.dtype.kind not in "iu" or int(value) != 1:
        raise ValueError(f"Unsupported {label} schema version")


def _clocks(values, *, strict=True):
    values = np.asarray(values)
    if values.dtype != np.dtype("int64") or values.ndim != 1:
        raise ValueError("Result clocks must be one-dimensional int64 milliseconds")
    if np.any(values < 0) or np.any(values[1:] < values[:-1]):
        raise ValueError("Result clocks must be nonnegative and ordered")
    if strict and np.any(values[1:] == values[:-1]):
        raise ValueError("Equity clocks must be unique")
    return values


def write_equity(directory: Path, frame: pd.DataFrame) -> dict:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Equity export requires a DatetimeIndex")
    timestamps = _clocks(frame.index.as_unit("ms").asi8)
    if not set(frame.columns) <= set(EQUITY_COLUMNS):
        raise ValueError("Unsupported equity export columns")
    if not {"usd_total_equity", "btc_total_equity", "strategy_equity"} <= set(frame):
        raise ValueError("Equity export is missing native equity quantities")
    arrays = {name: frame[name].to_numpy(dtype=np.float64) for name in frame}
    for name, values in arrays.items():
        if np.isinf(values).any() or (
            name.endswith("equity") and not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid native equity export: {name}")
    write_numeric_archive(
        Path(directory) / EQUITY_FILE,
        schema_version=np.int64(1),
        timestamp_ms=timestamps,
        **arrays,
    )
    return {
        "file": EQUITY_FILE,
        "schema_version": 1,
        "rows": len(frame),
        "columns": list(frame.columns),
        "timestamp_unit": "ms",
        "value_dtype": "float64",
    }


def validate_trace(trace, report=None) -> dict[str, np.ndarray]:
    _version(trace["schema_version"], "HSL trace")
    if set(trace) != {"schema_version", *TRACE_DTYPES}:
        raise ValueError("HSL trace column schema mismatch")
    columns = {key: np.asarray(trace[key]) for key in TRACE_DTYPES}
    n = len(columns["timestamp"])
    for key, dtype in TRACE_DTYPES.items():
        value = columns[key]
        if value.dtype != np.dtype(dtype) or value.ndim != 1 or len(value) != n:
            raise ValueError(f"Invalid HSL trace column: {key}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite HSL trace facts: {key}")
    _clocks(columns["timestamp"], strict=False)
    seq = columns["sequence"]
    if np.any(seq[1:] <= seq[:-1]):
        raise ValueError("HSL trace sequence must preserve native observation order")
    if np.any(~np.isin(columns["side"], [-1, 0, 1])) or np.any(columns["coin"] < -1):
        raise ValueError("Invalid HSL trace scope identity")
    if (
        np.any(columns["phase"] > 2)
        or np.any(columns["action"] > 2)
        or np.any(columns["exposed"] > 1)
        or np.any(columns["nonpositive_equity"] > 1)
    ):
        raise ValueError("Unsupported HSL trace observation/state code")
    if report is not None:
        mode = report["mode"]
        side, coin = columns["side"], columns["coin"]
        valid = (
            ((side == -1) & (coin == -1))
            if mode == "portfolio"
            else (
                ((side >= 0) & (coin == -1))
                if mode == "pside"
                else ((side >= 0) & (coin >= 0) & (coin < len(report["coins"])))
            )
        )
        if not valid.all():
            raise ValueError("HSL trace scope disagrees with its report policy")
    return columns


def write_hsl(directory: Path, report: dict | None, *, trace=None) -> dict | None:
    if report is None:
        return None
    metadata = dict(report)
    if trace is not None:
        columns = validate_trace(trace, report)
        write_numeric_archive(
            Path(directory) / TRACE_FILE, schema_version=np.int64(1), **columns
        )
        metadata["trace"] = {
            "schema_version": 1,
            "file": TRACE_FILE,
            "rows": len(columns["timestamp"]),
            "timestamp_unit": "ms",
            "value_dtype": "float64",
            "retention": "exposed, cooldown and final inactive boundary",
        }
    write_json(Path(directory) / "hsl_report.json", metadata)
    return metadata


def load_equity_frame(directory, columns=None) -> pd.DataFrame:
    return ResultStore(directory).equity(columns=columns)


class ResultStore:
    """Open metadata first; equity columns, fills and HSL are loaded on demand."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        manifest = self.directory / "result.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else None
        if self.manifest is not None:
            version = self.manifest.get("schema_version")
            if type(version) is not int or version != RESULT_VERSION:
                raise ValueError("Unsupported backtest result schema version")
        self._arrays: dict[str, np.ndarray] = {}
        self._fills = None
        self._trace = None
        self._lock = RLock()

    def json(self, name, *, optional=False):
        if name not in {
            "config.json",
            "analysis.json",
            "dataset.json",
            "session.json",
            "hsl_report.json",
            "timings.json",
        }:
            raise ValueError("Unsupported result metadata file")
        path = self.directory / name
        if optional and not path.exists():
            return None
        return json.loads(path.read_text())

    def _column(self, name):
        with self._lock:
            if name not in self._arrays:
                with np.load(
                    self.directory / EQUITY_FILE, allow_pickle=False
                ) as archive:
                    _version(archive["schema_version"], "equity")
                    if not set(archive.files) <= {
                        "schema_version",
                        "timestamp_ms",
                        *EQUITY_COLUMNS,
                    }:
                        raise ValueError("Unsupported equity archive columns")
                    value = archive[name]
                if name == "timestamp_ms":
                    _clocks(value)
                elif (
                    value.ndim != 1
                    or value.dtype != np.dtype("float64")
                    or np.isinf(value).any()
                    or name.endswith("equity")
                    and not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid stored equity column: {name}")
                self._arrays[name] = value
            return self._arrays[name]

    def equity(self, *, columns=None, start=None, end=None) -> pd.DataFrame:
        if columns is not None and not set(columns) <= set(EQUITY_COLUMNS):
            raise ValueError("Unknown equity columns requested")
        path = self.directory / EQUITY_FILE
        if path.exists():
            timestamps = self._column("timestamp_ms")
            if columns is None:
                with np.load(path, allow_pickle=False) as archive:
                    columns = [key for key in EQUITY_COLUMNS if key in archive.files]
            first = (
                0
                if start is None
                else int(np.searchsorted(timestamps, start, side="left"))
            )
            last = (
                len(timestamps)
                if end is None
                else int(np.searchsorted(timestamps, end, side="right"))
            )
            data = {}
            for name in columns:
                values = self._column(name)
                if len(values) != len(timestamps):
                    raise ValueError("Equity columns and clocks have different lengths")
                data[name] = values[first:last]
            return pd.DataFrame(
                data, index=pd.to_datetime(timestamps[first:last], unit="ms")
            )
        # Released CSV files can have a named timestamp or a generated index.
        frame = pd.read_csv(
            self.directory / "balance_and_equity.csv.gz", float_precision="round_trip"
        )
        name = "timestamp" if "timestamp" in frame else frame.columns[0]
        stamps = frame.pop(name)
        if pd.api.types.is_numeric_dtype(stamps):
            if name != "timestamp":
                raise ValueError("Legacy equity is missing an explicit timestamp clock")
            frame.index = pd.to_datetime(stamps, unit="ms", errors="raise")
        else:
            frame.index = pd.to_datetime(stamps, errors="raise")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Legacy equity clocks must be unique and ordered")
        if columns is not None:
            frame = frame[list(columns)]
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start, unit="ms")]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end, unit="ms")]
        return frame

    def fills(self) -> pd.DataFrame:
        with self._lock:
            if self._fills is None:
                frame = pd.read_csv(
                    self.directory / "fills.csv", float_precision="round_trip"
                )
                frame = frame.drop(
                    columns=[key for key in frame if str(key).startswith("Unnamed")]
                )
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
                self._fills = frame
            return self._fills.copy()

    def hsl(self):
        from hsl_reporting import hsl_report

        report = self.json("hsl_report.json", optional=True)
        return None if report is None else hsl_report({"hsl": report})

    def hsl_trace(self):
        with self._lock:
            if self._trace is not None:
                return dict(self._trace)
            report = self.hsl()
            if report is None or "trace" not in report:
                return None  # Legacy detailed reports remain handled by their renderer.
            description = report["trace"]
            if description.get("file") != TRACE_FILE:
                raise ValueError("Unsupported HSL trace artifact path")
            with np.load(self.directory / TRACE_FILE, allow_pickle=False) as archive:
                columns = validate_trace(
                    {key: archive[key] for key in archive.files}, report
                )
            if len(columns["timestamp"]) != description["rows"]:
                raise ValueError("HSL trace row count disagrees with metadata")
            for values in columns.values():
                values.flags.writeable = False
            self._trace = {"schema_version": 1, **columns}
            return dict(self._trace)


def extrema_indices(values, *, max_points=2000, forced=()) -> np.ndarray:
    """Select original rows, preserving extrema of each quantity and event rows.

    This bounds rendering work, never saved data. All returned rows keep their
    real order/timestamps; extrema of separate series are never combined.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    bins = max(1, max_points // max(2, 2 * values.shape[1]))
    selected = {0, n - 1}
    selected.update(int(i) for i in forced if 0 <= i < n)
    for first, last in zip(
        np.linspace(0, n, bins + 1, dtype=int)[:-1],
        np.linspace(0, n, bins + 1, dtype=int)[1:],
    ):
        block = values[first:last]
        for col in range(values.shape[1]):
            finite = np.flatnonzero(np.isfinite(block[:, col]))
            if len(finite):
                data = block[finite, col]
                selected.add(first + int(finite[np.argmin(data)]))
                selected.add(first + int(finite[np.argmax(data)]))
    return np.array(sorted(selected), dtype=np.int64)
