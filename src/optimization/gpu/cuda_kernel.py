"""Compile the repository's scalar Metal kernel dialect as CUDA C++.

Only address-space qualifiers, the grid index, and the few Metal vector operations
used by these kernels differ. Multi-coin private arrays may use a smaller validated
capacity. Strategy expressions remain in the Rust-owned source.
CuPy compiles with NVRTC and shares PyTorch tensors and its current CUDA stream.
"""

from __future__ import annotations

import re
import numpy as np

_CUDA_PREAMBLE = r"""
#include <cuda_runtime.h>
#define INFINITY __int_as_float(0x7f800000)
#define NAN __int_as_float(0x7fffffff)
using uint = unsigned int;
using ulong = unsigned long long;
template <typename T> __device__ inline T clamp(T v, T lo, T hi) {
    return min(max(v, lo), hi);
}
template <typename To, typename From> __device__ inline To as_type(From value) {
    static_assert(sizeof(To) == sizeof(From), "bitcast size mismatch");
    union { From from; To to; } bits;
    bits.from = value;
    return bits.to;
}
struct alignas(8) PBFloat2 {
    float x, y;
    PBFloat2() = default;
    __device__ PBFloat2(float a, float b): x(a), y(b) {}
};
struct alignas(8) PBInt2 {
    int x, y;
    PBInt2() = default;
    __device__ PBInt2(int a, int b): x(a), y(b) {}
};
struct alignas(16) PBFloat3 {
    float x, y, z;
    PBFloat3() = default;
    __device__ PBFloat3(float a): x(a), y(a), z(a) {}
    __device__ PBFloat3(float a, float b, float c): x(a), y(b), z(c) {}
};
__device__ inline PBFloat3 operator+(PBFloat3 a, float b) {
    return PBFloat3(a.x+b, a.y+b, a.z+b);
}
__device__ inline PBFloat3 operator-(PBFloat3 a, PBFloat3 b) {
    return PBFloat3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ inline PBFloat3 operator/(float a, PBFloat3 b) {
    return PBFloat3(a/b.x, a/b.y, a/b.z);
}
__device__ inline PBFloat3 clamp(PBFloat3 a, float lo, float hi) {
    return PBFloat3(clamp(a.x,lo,hi),clamp(a.y,lo,hi),clamp(a.z,lo,hi));
}
__device__ inline PBFloat3 fma(PBFloat3 a, PBFloat3 b, PBFloat3 c) {
    return PBFloat3(fmaf(a.x,b.x,c.x),fmaf(a.y,b.y,c.y),fmaf(a.z,b.z,c.z));
}
"""


def cuda_source(source: str, *, coin_capacity: int | None = None) -> str:
    """Lower only the explicitly supported scalar shader dialect."""
    if coin_capacity is not None:
        declarations = list(re.finditer(
            r"\bconstant\s+int\s+MAX_COINS\s*=\s*(\d+)\s*;", source
        ))
        if len(declarations) != 1:
            raise ValueError("CUDA coin specialization requires one MAX_COINS declaration")
        declaration = declarations[0]
        if (
            type(coin_capacity) is not int
            or not 1 <= coin_capacity <= int(declaration[1])
        ):
            raise ValueError("CUDA coin capacity must fit the shader's MAX_COINS limit")
        source = (
            source[:declaration.start(1)]
            + str(coin_capacity)
            + source[declaration.end(1):]
        )
    for signature in re.findall(r"kernel\s+void\s+\w+\((.*?)\)\s*\{", source, re.S):
        for position, argument in enumerate(signature.split(",")):
            slot = re.search(r"\[\[buffer\((\d+)\)\]\]", argument)
            if slot is not None and int(slot[1]) != position:
                raise ValueError("CUDA shader buffers must follow positional argument order")
    source = re.sub(r"\s*\[\[buffer\(\d+\)\]\]", "", source)

    def lower_kernel_references(match):
        aliases = []

        def lower_reference(ref):
            type_name, name = ref.groups()
            pointer = f"passivbot_ref_{name}"
            aliases.append(f"const {type_name}& {name} = *{pointer};")
            return f"constant {type_name}* {pointer}"

        signature = re.sub(r"\bconstant\s+(\w+)\s*&\s*(\w+)", lower_reference, match[0])
        return signature + "\n" + "\n".join(aliases) if aliases else signature

    source = re.sub(r"kernel\s+void\s+\w+\(.*?\)\s*\{", lower_kernel_references, source, flags=re.S)
    source = source.replace("#include <metal_stdlib>", "")
    source = source.replace("using namespace metal;", "")
    source = re.sub(r"\bconstant\s+(\w+)\s*\*", r"const \1*", source)
    source = re.sub(r"\bconstant\b", "constexpr", source)
    source = re.sub(r"\b(?:device|thread)\s+", "", source)
    source = re.sub(r"\binline\b", "__device__ inline", source)
    source = re.sub(r"\bkernel\s+void\b", 'extern "C" __global__ void', source)
    for before, after in [("float2", "PBFloat2"), ("int2", "PBInt2"), ("float3", "PBFloat3")]:
        source = re.sub(r"\b" + before + r"\b", after, source)
    source, count = re.subn(
        r"uint\s+(\w+)\s*\[\[thread_position_in_grid\]\]\s*\)\s*\{",
        lambda m: "unsigned int passivbot_dispatch_count) {\n"
        + f"const uint {m[1]} = blockIdx.x * blockDim.x + threadIdx.x;\n"
        + f"if ({m[1]} >= passivbot_dispatch_count) return;\n",
        source,
    )
    if count == 0 or "[[" in source:
        raise ValueError("Unsupported GPU shader grid signature or Metal attribute")
    return _CUDA_PREAMBLE + source


class CudaShaderLibrary:
    def __init__(self, source: str, *, coin_capacity: int | None = None):
        import cupy

        self._cupy = cupy
        self._scalar_parameters = {}
        for name, signature in re.findall(r"kernel\s+void\s+(\w+)\((.*?)\)\s*\{", source, re.S):
            self._scalar_parameters[name] = {
                index: match[1]
                for index, argument in enumerate(signature.split(","))
                if (match := re.search(r"\bconstant\s+(\w+)\s*&", argument))
            }
        self._module = cupy.RawModule(
            code=cuda_source(source, coin_capacity=coin_capacity),
            options=("--std=c++17", "--fmad=false"),
        )
        self._module.compile()

    def __getattr__(self, name: str):
        kernel = self._module.get_function(name)

        def launch(*args, threads, group_size=None):
            import torch

            count = int(threads if isinstance(threads, int) else threads[0])
            if not isinstance(threads, int) and tuple(threads[1:]) != (1, 1):
                raise ValueError("CUDA proxy only supports one-dimensional dispatch")
            if not 1 <= count <= np.iinfo(np.uint32).max:
                raise ValueError("CUDA dispatch must contain 1..2**32-1 threads")
            if group_size is not None and tuple(group_size[1:]) != (1, 1):
                raise ValueError("CUDA proxy only supports one-dimensional thread groups")
            block = 64 if group_size is None else int(group_size[0])
            if not 1 <= block <= 1024:
                raise ValueError("Invalid CUDA thread block size")
            scalar_parameters = self._scalar_parameters.get(name, {})
            if scalar_parameters:
                device = next(
                    (arg.device for arg in args if isinstance(arg, torch.Tensor)),
                    torch.device("cuda"),
                )
                scalar_types = {
                    "int": torch.int32,
                    "uint": torch.uint32,
                    "ulong": torch.uint64,
                    "float": torch.float32,
                    "bool": torch.bool,
                }
                args = tuple(
                    (
                        torch.tensor(
                            [arg], device=device, dtype=scalar_types[scalar_parameters[index]]
                        )
                        if index in scalar_parameters and isinstance(arg, (int, float))
                        else arg
                    )
                    for index, arg in enumerate(args)
                )
            if not args or any(
                not isinstance(arg, torch.Tensor)
                or arg.device.type != "cuda"
                or not arg.is_contiguous()
                or arg.device != args[0].device
                for arg in args
            ):
                raise ValueError(
                    "CUDA kernel arguments must be contiguous tensors on one CUDA device"
                )
            device = args[0].device
            with torch.cuda.device(device), self._cupy.cuda.Device(device.index):
                stream = self._cupy.cuda.ExternalStream(
                    torch.cuda.current_stream(device).cuda_stream
                )
                with stream:
                    arrays = tuple(self._cupy.from_dlpack(arg) for arg in args)
                    kernel(((count + block - 1) // block,), (block,), arrays + (np.uint32(count),))

        return launch
