import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/l2_normalization")

mod = torch.utils.cpp_extension.load(
    name="l2_normalization",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

x = torch.randn(512, 1024, device="cuda")
ref = F.normalize(x, p=2, dim=-1)

torch.testing.assert_close(mod.l2_norm(x), ref, rtol=1e-4, atol=1e-5)
print("l2_norm PASS")

print()
print(f"l2_norm     {do_bench(lambda: mod.l2_norm(x)):.3f} ms")
print(f"torch (ref) {do_bench(lambda: F.normalize(x, p=2, dim=-1)):.3f} ms")
