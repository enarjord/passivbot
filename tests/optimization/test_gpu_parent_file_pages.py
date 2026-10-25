"""Advisory parent-page release preserves ownership and immutable data."""
import mmap
from pathlib import Path

import numpy as np
import pytest

from shared_arrays import SharedArrayManager, attach_shared_array


def test_parent_advice_retains_owned_files_and_readback(tmp_path):
    original = np.arange(300, dtype=np.float64).reshape(25, 3, 4)
    path = tmp_path / "source.npy"
    np.save(path, original)
    manager = SharedArrayManager(file_backed=True)
    try:
        values = manager.load_numpy_artifact(path)
        spec, _ = manager.create_from(values)
        assert manager.file_backed
        advised = manager.discard_file_pages()
        assert advised == int(hasattr(mmap, "MADV_DONTNEED") and hasattr(values._mmap, "madvise"))
        assert path.exists() and not values._mmap.closed
        np.testing.assert_array_equal(manager.view(spec), original)
        with_array = attach_shared_array(spec)
        try:
            np.testing.assert_array_equal(with_array.array, original)
        finally:
            with_array.close()
    finally:
        manager.cleanup()
    assert path.exists()


def test_unused_native_close_does_not_call_page_callback():
    from optimization.gpu.native import NativeBacktestService
    calls = []
    with NativeBacktestService(on_inputs_prepared=lambda: calls.append(1)):
        pass
    assert calls == []
    with pytest.raises(TypeError, match="callback"):
        NativeBacktestService(on_inputs_prepared=1)
