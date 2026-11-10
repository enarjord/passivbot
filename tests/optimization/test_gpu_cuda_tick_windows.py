"""CUDA input windows preserve signed tick bytes, indices and queue ownership."""

import ast
from contextlib import contextmanager
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu.cuda_kernel import CudaShaderLibrary, cuda_source


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def helpers():
    path = ROOT / "src/optimization/gpu/mps_kernel.py"
    names = {"_METAL_WINDOWED_TICKS", "_with_metal_tick_windows",
             "_with_compact_metal_hourly_ranges", "_metal_tick_window_paths",
             "_metal_tick_window_bounds", "_cuda_tick_window_arguments"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if ((isinstance(node, ast.FunctionDef) and node.name in names)
                 or (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in names
                             for target in node.targets)))]
    namespace = dict(Path=Path, re=re, np=np, contextmanager=contextmanager, logging=logging)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def tick_data(tmp_path, *, n=19, coins=3):
    data = dict(n=n, n_coins=coins, stream_tick_inputs=True)
    for ordinal, (name, width) in enumerate((
        ("fill_ticks", 2), ("touch_ticks", 2), ("touch_nearest_ticks", 1),
        ("touch_min_qty_bits", 1), ("touch_min_qty_relation", 1),
    )):
        shape = (n, coins) + ((2,) if width == 2 else ())
        if name == "touch_min_qty_relation":
            values = (np.arange(np.prod(shape)) % 3 - 1).astype(np.int8).reshape(shape)
        else:
            values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) + ordinal * 1000
        data[name] = tmp_path / f"{name}.npy"
        np.save(data[name], values, allow_pickle=False)
    return data


def _windowed_source(helpers):
    source = (ROOT / "passivbot-rust/src/gpu/mps_multicoin_common.metal").read_text()
    source += (ROOT / "passivbot-rust/src/gpu/mps_trailing_martingale_multicoin.metal").read_text()
    source = helpers["_with_metal_tick_windows"](source)
    return helpers["_with_compact_metal_hourly_ranges"](source)


def test_cuda_lowering_keeps_byte_relation_window_and_device_callable_views(helpers, monkeypatch):
    source = _windowed_source(helpers)
    lowered = cuda_source(source, coin_capacity=4)
    assert lowered.count("const signed char* window_touch_min_qty_relation") == 3
    assert "PassivbotTickWindow touch_min_qty_relation" not in lowered
    assert lowered.count("PassivbotByteTickWindow touch_min_qty_relation") == 7
    assert "__device__ inline int operator[](int absolute_index)" in lowered
    assert "__device__ inline float operator[](int absolute_index)" in lowered
    assert "return values[absolute_index - first_element];" in lowered
    assert "hour_log_ranges[j * coin_count + coin]" in lowered
    assert "const int hour = (start_minute_of_hour + k) / 60 - 1;" in lowered
    assert lowered.count("const float* compact_hour_log_ranges") == 3
    # Public wrapper arguments, not the internal view type, define tensor dtype.
    module = SimpleNamespace(compile=lambda: None)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(RawModule=lambda **kwargs: module))
    library = CudaShaderLibrary(source, coin_capacity=4)
    for kernel in ("passivbot_trailing_martingale_multicoin",
                   "passivbot_trailing_martingale_multicoin_long",
                   "passivbot_trailing_martingale_multicoin_fused"):
        assert library._byte_parameters[kernel] == {6}


def test_cuda_lowering_streams_ema_anchor_tick_arguments_without_rebinding_hourly_slot(helpers):
    source = (ROOT / "passivbot-rust/src/gpu/mps_multicoin_common.metal").read_text()
    source += (ROOT / "passivbot-rust/src/gpu/mps_ema_anchor_multicoin_long.metal").read_text()
    source = helpers["_with_metal_tick_windows"](source, strategy="ema_anchor")
    source = helpers["_with_compact_metal_hourly_ranges"](source, strategy="ema_anchor")
    lowered = cuda_source(source, coin_capacity=4)
    assert lowered.count("const int* window_fill_ticks,") == 3
    assert lowered.count("const int* window_touch_ticks,") == 3
    assert lowered.count("const float* compact_hour_log_ranges,") == 3
    assert "hour_log_ranges[j * coin_count + coin]" in lowered


def test_cuda_disk_window_contract_accepts_only_original_signed_byte_relation(helpers, tmp_path):
    data = tick_data(tmp_path)
    check = helpers["_metal_tick_window_paths"]
    assert len(check(data, device="cuda", supported=True, temporal=True)) == 5
    np.save(data["touch_min_qty_relation"], np.zeros((19, 3), dtype=np.int32), allow_pickle=False)
    with pytest.raises(ValueError, match="Invalid streamed GPU tick array: touch_min_qty_relation"):
        check(data, device="cuda", supported=True, temporal=True)


def test_cuda_window_uploads_preserve_bytes_and_halos_until_existing_wait(helpers, tmp_path, monkeypatch):
    from optimization.gpu import residency

    data = tick_data(tmp_path)
    paths = {name: data[name] for name, _ in helpers["_METAL_WINDOWED_TICKS"]}
    original_load = np.load
    mappings, waits, uploads = [], [], []

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        mappings.append(array)
        return array

    def upload(array, *, device):
        assert device == "cuda"
        uploads.append(array.dtype)
        return SimpleNamespace(contiguous=lambda: np.array(array, copy=True))

    wait = lambda: waits.append(1)
    helpers["torch"] = SimpleNamespace(as_tensor=upload, cuda=SimpleNamespace(synchronize=wait))
    monkeypatch.setattr(np, "load", load)
    for begin, stop in ((1, 5), (5, 11), (11, 18)):
        sentinel = object()
        arguments = (sentinel, sentinel, *paths.values(), sentinel)
        with helpers["_cuda_tick_window_arguments"](
            arguments, paths, begin=begin, stop=stop, available=data["n"],
        ) as (actual, start):
            assert start == begin - 1
            assert actual[:2] == arguments[:2] and actual[-1] is sentinel
            for slot, path in enumerate(paths.values(), start=2):
                expected = original_load(path)[start:min(data["n"], stop + 1)]
                np.testing.assert_array_equal(actual[slot], expected)
                assert actual[slot].dtype == expected.dtype
            assert all(not mapping._mmap.closed for mapping in mappings[-5:])
            wait()  # The existing production replay boundary.
        assert all(mapping._mmap.closed for mapping in mappings)
    assert waits == [1, 1, 1]
    assert uploads == [np.dtype(np.int32)] * 4 + [np.dtype(np.int8)] \
        + [np.dtype(np.int32)] * 4 + [np.dtype(np.int8)] \
        + [np.dtype(np.int32)] * 4 + [np.dtype(np.int8)]
    assert not residency._cuda_upload_quarantine_snapshot()


def test_failed_cuda_wait_quarantines_sources_until_confirmed_quiescence(
    helpers, tmp_path, monkeypatch, caplog,
):
    from optimization.gpu import residency

    assert not residency._cuda_upload_quarantine_snapshot()
    data = tick_data(tmp_path)
    paths = {name: data[name] for name, _ in helpers["_METAL_WINDOWED_TICKS"]}
    mappings = []
    original_load = np.load
    original_error = RuntimeError("CUDA window upload failed")

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        mappings.append(array)
        return array

    def upload(array, **kwargs):
        if len(mappings) == 2:
            raise original_error
        return SimpleNamespace(contiguous=lambda: np.array(array, copy=True))

    def failed_wait():
        raise RuntimeError("CUDA queue status unknown")

    fake_torch = SimpleNamespace(as_tensor=upload, cuda=SimpleNamespace(synchronize=failed_wait))
    helpers["torch"] = fake_torch
    monkeypatch.setattr(np, "load", load)
    try:
        with pytest.raises(RuntimeError) as caught:
            with helpers["_cuda_tick_window_arguments"](
                (None,) * 8, paths, begin=1, stop=5, available=19,
            ):
                pytest.fail("failed upload cannot dispatch")
        assert caught.value is original_error
        assert len(residency._cuda_upload_quarantine_snapshot()) == len(mappings) == 2
        assert all(not array._mmap.closed for array in mappings)
        assert "CUDA input-window cleanup synchronization failed" in caplog.text
    finally:
        fake_torch.cuda.synchronize = lambda: None
        residency._synchronize_cuda_uploads(fake_torch)
    assert not residency._cuda_upload_quarantine_snapshot()
    assert all(array._mmap.closed for array in mappings)


def test_cuda_relation_window_executes_signed_values_at_absolute_offsets():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA unavailable")
    library = CudaShaderLibrary("""
        struct PassivbotTickWindow {
            constant int* values;
            int first_element;
            inline int operator[](int absolute_index) const {
                return values[absolute_index - first_element];
            }
        };
        kernel void relations(constant int* window_touch_min_qty_relation,
                              constant int* offset, device int* output,
                              uint i [[thread_position_in_grid]]) {
            const PassivbotTickWindow touch_min_qty_relation = {
                window_touch_min_qty_relation, offset[0]};
            output[i] = touch_min_qty_relation[int(i) + offset[0]];
        }
    """)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.tensor([-1, 0, 1] * 43, device="cuda", dtype=torch.int8)
        offset = torch.tensor([321], device="cuda", dtype=torch.int32)
        output = torch.empty(len(values), device="cuda", dtype=torch.int32)
        library.relations(values, offset, output, threads=len(values))
        np.testing.assert_array_equal(output.cpu().numpy(), values.cpu().numpy())
        with pytest.raises(ValueError, match="signed int8"):
            library.relations(values.to(torch.int32), offset, output, threads=len(values))
