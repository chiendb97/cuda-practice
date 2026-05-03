import os
import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

SRC = os.path.join(os.path.dirname(__file__), "../../cpp/relu")

mod = torch.utils.cpp_extension.load(
    name="relu",
    sources=[f"{SRC}/baseline.cu", f"{SRC}/optimize.cu", f"{SRC}/bindings.cpp"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

x = torch.randn(1024, 1024, device="cuda")
ref = x.clamp(min=0)

torch.testing.assert_close(mod.relu_baseline(x), ref)
print("relu_baseline PASS")
torch.testing.assert_close(mod.relu_optimize(x), ref)
print("relu_optimize PASS")

print()
print(f"relu_baseline {do_bench(lambda: mod.relu_baseline(x)):.3f} ms")
print(f"relu_optimize {do_bench(lambda: mod.relu_optimize(x)):.3f} ms")
print(f"torch.relu    {do_bench(lambda: torch.relu(x)):.3f} ms")
