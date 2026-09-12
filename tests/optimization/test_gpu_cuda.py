"""CUDA launch contract, independent of the shared strategy regression suite."""

from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu.cuda_kernel import cuda_source
from optimization.gpu.runtime import gpu_device


def test_runtime_selects_cuda_without_mps():
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    assert gpu_device(torch) == "cuda"
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="GPU optimization requires"):
        gpu_device(torch)


@pytest.mark.parametrize(
    "source", ["kernel void missing_index() {}", "kernel void k(uint i [[other]]) {}"]
)
def test_unknown_shader_dialect_is_rejected(source):
    with pytest.raises(ValueError, match="Unsupported GPU shader"):
        cuda_source(source)


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA unavailable")
    from optimization.gpu.cuda_kernel import CudaShaderLibrary

    return torch, CudaShaderLibrary


@pytest.mark.parametrize("count", [1, 63, 64, 65, 129])
def test_cuda_dispatch_bounds_and_current_stream(cuda, count):
    torch, library_cls = cuda
    library = library_cls("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void transform(constant float* input, device float* output,
                              uint i [[thread_position_in_grid]]) {
            output[i] = fma(input[i], 2.0f, 1.0f);
        }
    """)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.arange(count + 64, device="cuda", dtype=torch.float32) * 3
        output = torch.full_like(values, -1)
        library.transform(values, output, threads=(count, 1, 1))
        actual = output.cpu().numpy()
    np.testing.assert_array_equal(actual[:count], np.arange(count, dtype=np.float32) * 6 + 1)
    np.testing.assert_array_equal(actual[count:], -np.ones(64))
    for invalid in [0, 2**32]:
        with pytest.raises(ValueError, match="dispatch must contain"):
            library.transform(values, output, threads=invalid)
    with pytest.raises(ValueError, match="one-dimensional thread groups"):
        library.transform(values, output, threads=count, group_size=(64, 2, 1))
    with pytest.raises(ValueError, match="contiguous"):
        library.transform(values[::2], output, threads=count)
    with pytest.raises(ValueError, match="contiguous"):
        library.transform(values.cpu(), output, threads=count)


def test_cuda_vector_and_bitcast_contract(cuda):
    torch, library_cls = cuda
    library = library_cls("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void check(device float* output, uint i [[thread_position_in_grid]]) {
            float3 value = clamp(2.0f / (float3(1.0f, 3.0f, 7.0f) + 1.0f), 0.0f, 1.0f);
            value = fma(value, float3(8.0f) - float3(4.0f), float3(4.0f));
            output[0] = value.x; output[1] = value.y; output[2] = value.z;
            output[3] = as_type<float>(as_type<uint>(1.0f) - 1u);
            output[4] = INFINITY;
            output[5] = NAN;
        }
    """)
    output = torch.empty(6, device="cuda", dtype=torch.float32)
    library.check(output, threads=1)
    np.testing.assert_array_equal(output[:3].cpu().numpy(), [8, 6, 5])
    assert output[3].item() == np.nextafter(np.float32(1), np.float32(0))
    assert torch.isinf(output[4])
    assert torch.isnan(output[5])


def test_shader_buffer_annotations_preserve_argument_order():
    source = """kernel void copy(constant float* x [[buffer(0)]],
        device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        y[i] = x[i];
    }"""
    assert "[[" not in cuda_source(source)
    with pytest.raises(ValueError, match="positional argument order"):
        cuda_source(source.replace("buffer(1)", "buffer(0)"))
    with pytest.raises(ValueError, match="positional argument order"):
        cuda_source(source.replace(" [[buffer(0)]]", "").replace("buffer(1)", "buffer(0)"))


def test_checkpoint_keeps_mps_compatibility_and_identifies_cuda(monkeypatch):
    import sys
    from optimization.gpu.runtime import checkpoint_runtime

    runtime = SimpleNamespace(
        __version__="test-torch",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 6)),
    )
    assert checkpoint_runtime(runtime) == {}
    runtime.backends.mps.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__version__="test-cupy"))
    assert checkpoint_runtime(runtime) == {
        "cuda_runtime": {
            "kernel_contract": 1,
            "torch": "test-torch",
            "cupy": "test-cupy",
            "compute_capability": [8, 6],
        }
    }


def test_cuda_constant_reference_buffer(cuda):
    torch, library_cls = cuda
    library = library_cls("""
        kernel void scale(constant int& factor [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
            output[i] = float(factor) * float(i + 1);
        }
    """)
    factor = torch.tensor([3], dtype=torch.int32, device="cuda")
    output = torch.empty(5, device="cuda")
    for value in [factor, 3]:
        library.scale(value, output, threads=5)
        np.testing.assert_array_equal(output.cpu().numpy(), [3, 6, 9, 12, 15])
