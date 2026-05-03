import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/prefix_sum")

mod = torch.utils.cpp_extension.load(
    name="prefix_sum",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/optimize.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

n = 1 << 20
x = torch.randn(n, device="cuda")
ref = torch.cumsum(x, dim=0)

torch.testing.assert_close(mod.prefix_sum_cub(x), ref, rtol=1e-4, atol=1e-3)
print("prefix_sum_cub PASS")
torch.testing.assert_close(mod.prefix_sum_opt(x), ref, rtol=1e-4, atol=1e-3)
print("prefix_sum_opt PASS")

print()
print(f"prefix_sum_cub {do_bench(lambda: mod.prefix_sum_cub(x)):.3f} ms")
print(f"prefix_sum_opt {do_bench(lambda: mod.prefix_sum_opt(x)):.3f} ms")
print(f"torch.cumsum   {do_bench(lambda: torch.cumsum(x, dim=0)):.3f} ms")
