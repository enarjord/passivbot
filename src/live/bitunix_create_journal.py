"""Durable transport facts for the single outstanding Bitunix create."""

import json
import os
import re
import sqlite3
import stat
import tempfile
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Iterator


CLIENT_ORDER_ID_PATTERN = re.compile(r"[.A-Z:/a-z0-9_-]{1,36}")
WIRE_TOKEN_PATTERN = re.compile(r"[!-~]+")
_WIRE_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_LIMIT_EFFECTS = {"GTC", "IOC", "FOK", "POST_ONLY"}
_APPLICATION_ID = 0x50424352  # PBCR
_VERSION = 1
_SCHEMA = (
    "CREATE TABLE pending_create ("
    "slot INTEGER PRIMARY KEY CHECK (slot = 1), attempt TEXT NOT NULL, "
    "order_id TEXT, filled_qty TEXT, conflicting_order_id TEXT)"
)


class BitunixCreateJournalError(RuntimeError):
    """The pending create state could not be read or durably updated."""


class BitunixCreateAlreadyPending(BitunixCreateJournalError):
    """The account already has a create whose outcome is unresolved."""


def _matches(pattern: re.Pattern, value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _positive_decimal(value: object) -> bool:
    return _matches(_WIRE_DECIMAL, value) and Decimal(value) > 0


@dataclass(frozen=True)
class BitunixCreateAttempt:
    client_id: str
    symbol: str
    position_side: str
    side: str
    reduce_only: bool
    position_id: str | None
    order_type: str
    qty: str
    price: str | None
    effect: str | None
    submitted_at_ms: int

    def __post_init__(self) -> None:
        if not _matches(CLIENT_ORDER_ID_PATTERN, self.client_id):
            raise ValueError("Invalid Bitunix create attempt client ID")
        if not _matches(WIRE_TOKEN_PATTERN, self.symbol):
            raise ValueError("Invalid Bitunix create attempt symbol")
        if (
            self.position_side not in ("long", "short")
            or type(self.reduce_only) is not bool
        ):
            raise ValueError(
                "Invalid Bitunix create attempt position side or reduce-only flag"
            )
        opens_with_buy = self.position_side == "long"
        if self.side != ("buy" if opens_with_buy != self.reduce_only else "sell"):
            raise ValueError("Invalid Bitunix create attempt action side")
        if self.reduce_only:
            if not _matches(WIRE_TOKEN_PATTERN, self.position_id):
                raise ValueError("Bitunix close attempt requires a position ID")
        elif self.position_id is not None:
            raise ValueError("Bitunix entry attempt includes an unexpected position ID")
        if not _positive_decimal(self.qty):
            raise ValueError("Invalid Bitunix create attempt quantity")
        if self.order_type == "limit":
            if not _positive_decimal(self.price) or self.effect not in _LIMIT_EFFECTS:
                raise ValueError("Invalid Bitunix limit attempt price or time in force")
        elif (
            self.order_type != "market"
            or self.price is not None
            or self.effect is not None
        ):
            raise ValueError("Invalid Bitunix create attempt order type")
        if type(self.submitted_at_ms) is not int or self.submitted_at_ms <= 0:
            raise ValueError("Invalid Bitunix create attempt timestamp")

    @property
    def wire_side(self) -> str:
        return "BUY" if self.position_side == "long" else "SELL"

    @property
    def trade_side(self) -> str:
        return "CLOSE" if self.reduce_only else "OPEN"

    @classmethod
    def from_wire(cls, body: dict, *, submitted_at_ms: int) -> "BitunixCreateAttempt":
        raw_side = body.get("side")
        if raw_side not in ("BUY", "SELL"):
            raise ValueError("Invalid Bitunix create request hedge side")
        reduce_only = body.get("reduceOnly")
        trade_side = body.get("tradeSide")
        if trade_side != ("CLOSE" if reduce_only is True else "OPEN"):
            raise ValueError("Invalid Bitunix create request trade side")
        order_type = body.get("orderType")
        if order_type not in ("LIMIT", "MARKET"):
            raise ValueError("Invalid Bitunix create request order type")
        return cls(
            client_id=body.get("clientId"),
            symbol=body.get("symbol"),
            position_side="long" if raw_side == "BUY" else "short",
            side="buy" if (raw_side == "BUY") != (reduce_only is True) else "sell",
            reduce_only=reduce_only,
            position_id=body.get("positionId"),
            order_type=order_type.lower(),
            qty=body.get("qty"),
            price=body.get("price"),
            effect=body.get("effect"),
            submitted_at_ms=submitted_at_ms,
        )


@dataclass(frozen=True)
class BitunixCreateObservation:
    order_id: str
    filled_qty: Decimal
    conflicting_order_id: str | None = None

    def __post_init__(self) -> None:
        if not _matches(WIRE_TOKEN_PATTERN, self.order_id):
            raise ValueError("Invalid Bitunix observed order ID")
        if (
            not isinstance(self.filled_qty, Decimal)
            or not self.filled_qty.is_finite()
            or self.filled_qty < 0
        ):
            raise ValueError("Invalid Bitunix observed executed quantity")
        if self.conflicting_order_id is not None and (
            not _matches(WIRE_TOKEN_PATTERN, self.conflicting_order_id)
            or self.conflicting_order_id == self.order_id
        ):
            raise ValueError("Invalid Bitunix conflicting order ID")


PendingCreate = tuple[BitunixCreateAttempt, BitunixCreateObservation | None]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate field in Bitunix create attempt")
    return result


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(
        sqlite3.connect(
            path.as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=0.25
        )
    ) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("PRAGMA synchronous=EXTRA")
        conn.execute("PRAGMA fullfsync=ON")
        if conn.execute("PRAGMA journal_mode").fetchone() != ("delete",):
            raise BitunixCreateJournalError("Unsupported Bitunix create journal mode")
        yield conn


def _read_pending(conn: sqlite3.Connection) -> PendingCreate | None:
    if (
        conn.execute("PRAGMA application_id").fetchone() != (_APPLICATION_ID,)
        or conn.execute("PRAGMA user_version").fetchone() != (_VERSION,)
        or conn.execute("SELECT type, name, sql FROM sqlite_schema").fetchall()
        != [("table", "pending_create", _SCHEMA)]
    ):
        raise BitunixCreateJournalError("Invalid Bitunix create journal schema")
    rows = conn.execute(
        "SELECT slot, attempt, order_id, filled_qty, conflicting_order_id FROM pending_create"
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1 or rows[0][0] != 1 or not isinstance(rows[0][1], str):
        raise BitunixCreateJournalError("Invalid Bitunix create journal rows")
    _, payload, order_id, filled_qty, conflicting_order_id = rows[0]
    try:
        attempt = BitunixCreateAttempt(
            **json.loads(payload, object_pairs_hook=_unique_json_object)
        )
        if order_id is None and filled_qty is None and conflicting_order_id is None:
            return attempt, None
        if not _matches(_WIRE_DECIMAL, filled_qty):
            raise ValueError("Invalid Bitunix journal executed quantity")
        observation = BitunixCreateObservation(
            order_id, Decimal(filled_qty), conflicting_order_id
        )
        if observation.filled_qty > Decimal(attempt.qty):
            raise ValueError(
                "Bitunix journal executed quantity exceeds submitted amount"
            )
        return attempt, observation
    except (ValueError, TypeError) as exc:
        raise BitunixCreateJournalError(
            "Invalid Bitunix create journal pending state"
        ) from exc


def _insert(conn: sqlite3.Connection, attempt: BitunixCreateAttempt) -> None:
    payload = json.dumps(asdict(attempt), separators=(",", ":"), allow_nan=False)
    conn.execute("INSERT INTO pending_create (slot, attempt) VALUES (1, ?)", (payload,))


def _sync_directories(path: Path) -> None:
    if os.name == "nt":
        return
    path = path.resolve(strict=True)
    for path in (path, *path.parents):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class BitunixCreateJournal:
    """Persist at most one possibly transmitted create per account."""

    def __init__(self, path: Path):
        self.path = Path(path).absolute()

    def _exists(self) -> bool:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode) or (
            os.name != "nt" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
        ):
            raise BitunixCreateJournalError(
                "Bitunix create journal must be a private regular file"
            )
        return True

    def load(self) -> tuple[BitunixCreateAttempt, ...]:
        pending = self.load_pending()
        return (pending[0],) if pending is not None else ()

    def load_pending(self) -> PendingCreate | None:
        try:
            if not self._exists():
                return None
            with _connect(self.path) as conn:
                conn.execute("BEGIN")
                return _read_pending(conn)
        except (sqlite3.Error, OSError) as exc:
            raise BitunixCreateJournalError(
                "Could not read the Bitunix create journal"
            ) from exc

    def _create(self, attempt: BitunixCreateAttempt) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Publish only an initialized, committed database. A competing writer must
        # never replace the account's existing reservation with its own.
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        temp_path = Path(name)
        os.close(fd)
        try:
            with _connect(temp_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                conn.execute(f"PRAGMA user_version={_VERSION}")
                conn.execute(_SCHEMA)
                _insert(conn, attempt)
                conn.commit()
            try:
                os.link(temp_path, self.path)
            except FileExistsError:
                return False
        finally:
            temp_path.unlink()
        _sync_directories(self.path.parent)
        return True

    def record(self, attempt: BitunixCreateAttempt) -> None:
        """Commit the exact attempt before the transport can submit it."""
        if not isinstance(attempt, BitunixCreateAttempt):
            raise TypeError("Expected a BitunixCreateAttempt")
        try:
            if not self._exists() and self._create(attempt):
                return
            if not self._exists():
                raise BitunixCreateJournalError("Bitunix create journal disappeared")
            with _connect(self.path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                if _read_pending(conn):
                    raise BitunixCreateAlreadyPending(
                        "A Bitunix create is already pending"
                    )
                _insert(conn, attempt)
                conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise BitunixCreateJournalError(
                "Could not record the Bitunix create attempt"
            ) from exc

    def observe(
        self, client_id: str, order_id: str, filled_qty: Decimal
    ) -> BitunixCreateObservation | None:
        """Accumulate exact venue observations without allowing a regression."""
        if not _matches(CLIENT_ORDER_ID_PATTERN, client_id):
            raise ValueError("Invalid Bitunix observation client ID")
        observed = BitunixCreateObservation(order_id, filled_qty)
        try:
            if not self._exists():
                return None
            with _connect(self.path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                pending = _read_pending(conn)
                if pending is None or pending[0].client_id != client_id:
                    return None
                attempt, previous = pending
                if observed.filled_qty > Decimal(attempt.qty):
                    raise ValueError(
                        "Bitunix executed quantity exceeds submitted amount"
                    )
                if previous is not None:
                    if previous.order_id != order_id:
                        observed = replace(
                            previous,
                            conflicting_order_id=previous.conflicting_order_id
                            or order_id,
                        )
                    else:
                        observed = replace(
                            previous, filled_qty=max(previous.filled_qty, filled_qty)
                        )
                if observed != previous:
                    conn.execute(
                        "UPDATE pending_create SET order_id = ?, filled_qty = ?, "
                        "conflicting_order_id = ? WHERE slot = 1",
                        (
                            observed.order_id,
                            (
                                format(observed.filled_qty, "f")
                                if observed.filled_qty
                                else "0"
                            ),
                            observed.conflicting_order_id,
                        ),
                    )
                    conn.commit()
                return observed
        except (sqlite3.Error, OSError) as exc:
            raise BitunixCreateJournalError(
                "Could not persist the Bitunix create observation"
            ) from exc

    def resolve(
        self,
        client_id: str,
        *,
        order_id: str | None = None,
        filled_qty: Decimal | None = None,
    ) -> bool:
        """Release a proven negative or a reconciled matching venue observation."""
        if not _matches(CLIENT_ORDER_ID_PATTERN, client_id):
            raise ValueError("Invalid Bitunix create resolution client ID")
        if (order_id is None) != (filled_qty is None):
            raise ValueError(
                "Bitunix positive resolution requires order ID and quantity"
            )
        proof = (
            BitunixCreateObservation(order_id, filled_qty)
            if order_id is not None
            else None
        )
        try:
            if not self._exists():
                return False
            with _connect(self.path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                pending = _read_pending(conn)
                if pending is None or pending[0].client_id != client_id:
                    return False
                attempt, observed = pending
                if proof is None:
                    if observed is not None:
                        return False
                elif (
                    observed is None
                    or observed.conflicting_order_id is not None
                    or observed.order_id != proof.order_id
                    or not observed.filled_qty
                    <= proof.filled_qty
                    <= Decimal(attempt.qty)
                ):
                    return False
                conn.execute("DELETE FROM pending_create WHERE slot = 1")
                conn.commit()
                return True
        except (sqlite3.Error, OSError) as exc:
            raise BitunixCreateJournalError(
                "Could not resolve the Bitunix create attempt"
            ) from exc
