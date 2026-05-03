import os
import sys
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

cap = torch.cuda.get_device_capability()
# These kernels use SM90a-specific TMA / WGMMA instructions; they don't run on
# pre-Hopper devices, and SM90a code is not forward-compatible with Blackwell
# (SM100/SM120) which has its own MMA generation. Gate on exact major == 9.
if cap[0] != 9:
    print(f"Skipping: Hopper-only kernels (got capability {cap[0]}.{cap[1]}).")
    sys.exit(0)

SRC = os.path.join(os.path.dirname(__file__), "../../../cpp/matrix_multiplication/hopper")
CUTLASS_INC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../3rdparty/cutlass/include"))

mod = torch.utils.cpp_extension.load(
    name="matmul_hopper",
    sources=[
        f"{SRC}/optimize_1.cu",
        f"{SRC}/optimize_2.cu",
        f"{SRC}/optimize_3.cu",
        f"{SRC}/bindings.cpp",
    ],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-DCOMPILE_3X_HOPPER",
        "--generate-code=arch=compute_90a,code=[sm_90a]",
    ],
    extra_include_paths=[CUTLASS_INC],
    verbose=False,
)

M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
alpha, beta = 1.0, 0.0

ref = (alpha * (A.float() @ B.float()) + beta * C.float()).half()

for i in [1, 2, 3]:
    fn = getattr(mod, f"matmul_hopper_opt{i}")
    out = fn(alpha, A, B, beta, C)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)
    print(f"matmul_hopper_opt{i} PASS")

print()
for i in [1, 2, 3]:
    fn = getattr(mod, f"matmul_hopper_opt{i}")
    print(f"matmul_hopper_opt{i} {do_bench(lambda: fn(alpha, A, B, beta, C)):.3f} ms")
print(f"torch (fp16 mm)    {do_bench(lambda: A @ B):.3f} ms")
