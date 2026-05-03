import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/prefix_multiple")

mod = torch.utils.cpp_extension.load(
    name="prefix_multiple",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

n = 1024
x = 1.0 + 0.001 * torch.randn(n, device="cuda")
ref = torch.cumprod(x, dim=0)

torch.testing.assert_close(mod.prefix_multiple(x), ref, rtol=1e-3, atol=1e-3)
print("prefix_multiple PASS")

print()
print(f"prefix_multiple {do_bench(lambda: mod.prefix_multiple(x)):.3f} ms")
print(f"torch.cumprod   {do_bench(lambda: torch.cumprod(x, dim=0)):.3f} ms")
