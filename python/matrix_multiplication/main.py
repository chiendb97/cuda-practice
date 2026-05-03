import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/matrix_multiplication")

mod = torch.utils.cpp_extension.load(
    name="matrix_multiplication",
    sources=[f"{SRC}/cuda_core.cu", f"{SRC}/tensor_core.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")
ref = A @ B

torch.testing.assert_close(mod.matmul_cuda_core(A, B), ref, rtol=1e-3, atol=1e-3)
print("matmul_cuda_core PASS")
torch.testing.assert_close(mod.matmul_cuda_core_opt(A, B), ref, rtol=1e-3, atol=1e-3)
print("matmul_cuda_core_opt PASS")

Ah = A.half()
Bh = B.half()
out_tc = mod.matmul_tensor_core(Ah, Bh)
torch.testing.assert_close(out_tc, ref, rtol=1e-2, atol=1e-1)
print("matmul_tensor_core PASS")

print()
print(f"matmul_cuda_core     {do_bench(lambda: mod.matmul_cuda_core(A, B)):.3f} ms")
print(f"matmul_cuda_core_opt {do_bench(lambda: mod.matmul_cuda_core_opt(A, B)):.3f} ms")
print(f"matmul_tensor_core   {do_bench(lambda: mod.matmul_tensor_core(Ah, Bh)):.3f} ms")
print(f"torch.mm (fp32)      {do_bench(lambda: A @ B):.3f} ms")
print(f"torch.mm (fp16)      {do_bench(lambda: Ah @ Bh):.3f} ms")
