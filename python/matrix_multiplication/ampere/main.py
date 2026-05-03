import os
import sys
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

# These kernels are tuned for SM80 (A100, RTX 30/40-series datacenter parts)
# and request 128 KB of dynamic shared memory per block. Consumer Blackwell
# (RTX 5090) caps opt-in shared memory at ~99 KB, so the launch fails. Skip
# gracefully on such devices.
REQUIRED_SMEM = 128 * 1024
props = torch.cuda.get_device_properties(0)
if props.shared_memory_per_block_optin < REQUIRED_SMEM:
    print(f"Skipping: device {props.name} has {props.shared_memory_per_block_optin} bytes "
          f"opt-in shared memory, kernels need {REQUIRED_SMEM}.")
    sys.exit(0)

SRC = os.path.join(os.path.dirname(__file__), "../../../cpp/matrix_multiplication/ampere")

sources = [
    f"{SRC}/naive.cu",
    f"{SRC}/optimize_1.cu",
    f"{SRC}/optimize_2.cu",
    f"{SRC}/optimize_3.cu",
    f"{SRC}/optimize_4.cu",
    f"{SRC}/optimize_5.cu",
    f"{SRC}/optimize_6.cu",
    f"{SRC}/optimize_7.cu",
    f"{SRC}/optimize_8.cu",
    f"{SRC}/cublas.cu",
    f"{SRC}/bindings.cpp",
]

mod = torch.utils.cpp_extension.load(
    name="matmul_ampere",
    sources=sources,
    # Generate SASS for the actual device + PTX fallback for forward compat.
    extra_cuda_cflags=["-O3", "--use_fast_math",
                       "--generate-code=arch=compute_80,code=compute_80"],
    extra_ldflags=["-lcublas"],
    verbose=False,
)

M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.randn(M, N, device="cuda", dtype=torch.float16)
alpha, beta = 1.0, 0.5

ref = (alpha * (A.float() @ B.float()) + beta * C.float()).half()

variants = ["naive", "opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8", "cublas"]

for name in variants:
    fn = getattr(mod, f"matmul_ampere_{name}")
    out = fn(alpha, A, B, beta, C)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)
    print(f"matmul_ampere_{name} PASS")

print()
for name in variants:
    fn = getattr(mod, f"matmul_ampere_{name}")
    print(f"matmul_ampere_{name:8s} {do_bench(lambda: fn(alpha, A, B, beta, C)):.3f} ms")
print(f"{'torch (fp16 mm)':22s} {do_bench(lambda: alpha * (A @ B) + beta * C):.3f} ms")
