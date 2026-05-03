import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/l1_normalization")

mod = torch.utils.cpp_extension.load(
    name="l1_normalization",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/optimize.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

x = torch.randn(512, 1024, device="cuda")
eps = 1e-6
ref = x / x.abs().sum(dim=-1, keepdim=True).clamp(min=eps)

torch.testing.assert_close(mod.l1_norm(x, eps), ref, rtol=1e-4, atol=1e-5)
print("l1_norm PASS")
torch.testing.assert_close(mod.l1_norm_opt(x, eps), ref, rtol=1e-4, atol=1e-5)
print("l1_norm_opt PASS")

print()
print(f"l1_norm     {do_bench(lambda: mod.l1_norm(x, eps)):.3f} ms")
print(f"l1_norm_opt {do_bench(lambda: mod.l1_norm_opt(x, eps)):.3f} ms")
print(f"torch (ref) {do_bench(lambda: x / x.abs().sum(dim=-1, keepdim=True).clamp(min=eps)):.3f} ms")
