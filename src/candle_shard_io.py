"""Read current candle bytes while reusing bounded, validated NPY header metadata."""

from functools import lru_cache
from io import BytesIO
import struct

import numpy as np


@lru_cache(maxsize=128)
def _parsed_header(version, encoded):
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    return reader(BytesIO(encoded), max_header_size=10_000)


def load_candle_array(handle, candle_dtype):
    """Optimize only ordinary one-dimensional, non-object candle NPY files.

    Identity is the actual version and complete header bytes read on this call,
    never a filename, timestamp or size. Payloads are always read afresh. Other
    formats retain NumPy's loader, including its allow_pickle=False boundary.
    """
    start = handle.tell()

    def ordinary_load():
        handle.seek(start)
        return np.load(handle, allow_pickle=False)

    try:
        version = np.lib.format.read_magic(handle)
    except (ValueError, EOFError):
        return ordinary_load()
    if version not in ((1, 0), (2, 0)):
        return ordinary_load()
    length_size = 2 if version == (1, 0) else 4
    length_bytes = handle.read(length_size)
    if len(length_bytes) != length_size:
        return ordinary_load()
    length = struct.unpack("<H" if length_size == 2 else "<I", length_bytes)[0]
    # Bound both retained metadata and parsing to NumPy's default safety limit.
    if length > 10_000:
        return ordinary_load()
    header = handle.read(length)
    if len(header) != length:
        return ordinary_load()
    shape, fortran, dtype = _parsed_header(version, length_bytes + header)
    if len(shape) != 1 or shape[0] < 0 or fortran or dtype != candle_dtype or dtype.hasobject:
        return ordinary_load()
    rows = np.fromfile(handle, dtype=dtype, count=shape[0])
    # NumPy's loader also rejects short payloads rather than returning partial rows.
    rows.shape = shape
    return rows
